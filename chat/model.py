"""The chat model: a Llama-style decoder trained and sampled through vabs compute.

Shape (``ChatConfig`` defaults, from ``docs/small_python_chatbot.md``): 8 layers,
width 512, SwiGLU hidden width 1,344, 8 query heads sharing 4 key/value heads,
RoPE, RMSNorm, no biases, a 20,000-token vocabulary with the input embedding
tied to the output head, and a 1,024-token context: 33.06M parameters.

The structure follows ``vabs_gpt.model``: ``ChatModel`` keeps the parameters,
gradients and optimizer state as flat device arrays plus a forward copy with
every matrix transposed (the head reads the transposed embedding), and
``ChatRunner`` records one command buffer per optimizer step (or per
micro-batch with accumulation) that zeroes the gradients, runs forward, loss
and backward, then the update. AdamW comes from ``vabs_gpt.kernels``, the
8-bit blockwise states from ``vabs_gpt_bitsandbytes`` (``optim_bits=8``, the
default) and Muon's Newton-Schulz step from ``vabs_gpt.model.Runner``.

``precision='bf16'`` (``ChatModel``, or the ``Device`` default) runs the linear
layers, the attention products and Newton-Schulz on bfloat16 tensor cores
(``kernels.mm_coop``) with float32 accumulation; everything stored stays float32.

Targets equal to ``IGNORE`` are not scored, which is how fine-tuning computes
the loss on assistant tokens only. The loss of a step is the mean over its
scored tokens.
"""
import array
import json
import math
import os
import random
import struct
from dataclasses import asdict, dataclass

from vabs.memory import MemoryUsage
from vabs.pipeline import ComputePipeline
from vabs.renderer import one_shot_frame
from vabs.structs import uint32

from vabs_gpt import kernels as GK
from vabs_gpt import model as GM
from vabs_gpt.gpu import Recording, compute_barrier
from vabs_gpt_bitsandbytes import functional as QF
from vabs_gpt_bitsandbytes import kernels as Q

from . import kernels as K

IGNORE = K.IGNORE
MAGIC = b'VCHT'
NORM_ROWS = 128          # rows per partial sum of a norm gain gradient
PARTIAL_FLOATS = 32 << 20   # budget of one partial-sum scratch buffer (128 MB)
LOSS_SUMS = GM.LOSS_SUMS
NORM_STRIDE = GM.NORM_STRIDE
SPECIALIZED = frozenset({'matmul_fwd', 'attn_softmax', 'attn_mm', 'rmsnorm_fwd', 'mm_coop'})
_ceil = GM._ceil


@dataclass
class ChatConfig:
    vocab: int = 20000
    ctx: int = 1024
    dim: int = 512
    heads: int = 8
    kv_heads: int = 4
    layers: int = 8
    mlp: int = 1344
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02

    def __post_init__(self):
        for name in ('vocab', 'ctx', 'dim', 'heads', 'kv_heads', 'layers', 'mlp'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError('dim must divide into heads, and heads into kv_heads')
        if self.head_size % 4:
            raise ValueError(f'head size {self.head_size} must be a multiple of 4')
        if self.mlp % 4 or self.vocab % 4:
            raise ValueError('mlp and vocab must be multiples of 4')

    @property
    def head_size(self):
        return self.dim // self.heads

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_size

    @property
    def qkv_dim(self):
        return self.dim + 2 * self.kv_dim


# -- parameters ---------------------------------------------------------------

def param_layout(cfg):
    """Ordered ``(name, size)``; matrices are ``W[out, in]``, norms a gain vector."""
    C, F, S = cfg.dim, cfg.mlp, cfg.qkv_dim
    out = [('wte', cfg.vocab * C)]
    for i in range(cfg.layers):
        out += [(f'h{i}.n1', C), (f'h{i}.qkv', S * C), (f'h{i}.proj', C * C),
                (f'h{i}.n2', C), (f'h{i}.fc', 2 * F * C), (f'h{i}.down', C * F)]
    return out + [('nf', C)]


def matrix_shape(cfg, name):
    """``(out, in)`` of a matrix parameter (the embedding counts as one), else None."""
    kind = name.split('.')[-1]
    C, F = cfg.dim, cfg.mlp
    return {'wte': (cfg.vocab, C), 'qkv': (cfg.qkv_dim, C), 'proj': (C, C),
            'fc': (2 * F, C), 'down': (C, F)}.get(kind)


def param_count(cfg):
    return sum(size for _, size in param_layout(cfg))


def init_params(cfg, seed=0):
    """N(0, init_std) matrices, the residual projections scaled by 1/sqrt(2 layers), gains 1."""
    rng = random.Random(seed)
    gauss = rng.gauss
    proj_std = cfg.init_std / math.sqrt(2 * cfg.layers)
    params = array.array('f')
    for name, size in param_layout(cfg):
        if matrix_shape(cfg, name) is None:
            params.extend(array.array('f', [1.0]) * size)
            continue
        std = proj_std if name.endswith(('.proj', '.down')) else cfg.init_std
        params.extend(array.array('f', [gauss(0.0, std) for _ in range(size)]))
    return params


def decay_mask(cfg):
    mask = array.array('f')
    for name, size in param_layout(cfg):
        mask.extend(array.array('f', [0.0 if matrix_shape(cfg, name) is None else 1.0]) * size)
    return mask


def transpose_index(cfg):
    """Source index of every element of the forward copy: each matrix transposed."""
    index, o = array.array('I'), 0
    for name, size in param_layout(cfg):
        shape = matrix_shape(cfg, name)
        if shape:
            c_in = shape[1]
            for c in range(c_in):
                index.extend(range(o + c, o + size, c_in))
        else:
            index.extend(range(o, o + size))
        o += size
    return index


def offsets(cfg):
    off, n = {}, 0
    for name, size in param_layout(cfg):
        off[name] = n
        n += size
    return off


MUON_KINDS = ('qkv', 'proj', 'fc', 'down')


def muon_matrices(cfg):
    """``(offset, rows, cols)`` of every matrix Muon updates: q, k, v, the attention output,
    gate, up and down projections, each its own matrix."""
    C, F, KV = cfg.dim, cfg.mlp, cfg.kv_dim
    off = offsets(cfg)
    out = []
    for i in range(cfg.layers):
        q, fc = off[f'h{i}.qkv'], off[f'h{i}.fc']
        out += [(q, C, C), (q + C * C, KV, C), (q + C * C + KV * C, KV, C),
                (off[f'h{i}.proj'], C, C), (fc, F, C), (fc + F * C, F, C),
                (off[f'h{i}.down'], C, F)]
    return out


# A batched product in Newton-Schulz (``bmm_nn``) takes its tile from global_id.x
# and its split-K chunk from global_id.y, so its dispatch must not wrap into y
# rows: at most SPLIT_GROUPS workgroups of ROW_WG tiles of 16 elements.
MUON_BATCH_ELEMS = GK.SPLIT_GROUPS * GK.ROW_WG * 16


def muon_groups(cfg):
    """Batches of equally shaped matrices, each small enough for one bmm dispatch."""
    groups = {}
    for o, rows, cols in muon_matrices(cfg):
        groups.setdefault((min(rows, cols), max(rows, cols)), []).append((o, rows, cols))
    out = []
    for (m, n), mats in groups.items():
        per = max(1, MUON_BATCH_ELEMS // (m * n))
        out += [(m, n, mats[i:i + per]) for i in range(0, len(mats), per)]
    return out


def adam_mask(cfg):
    mask = array.array('f', [1.0]) * param_count(cfg)
    for o, rows, cols in muon_matrices(cfg):
        mask[o:o + rows * cols] = array.array('f', bytes(4 * rows * cols))
    return mask


def adam_segments(cfg, muon=False):
    """``[(start, size, decay)]`` of the tensors AdamW updates (each its own 8-bit blocks)."""
    segs, o = [], 0
    for name, size in param_layout(cfg):
        shape = matrix_shape(cfg, name)
        if not (muon and name.split('.')[-1] in MUON_KINDS):
            segs.append((o, size, shape is not None))
        o += size
    return segs


def rope_table(cfg, T):
    """``[T, hs/4]`` vec4s ``(cos, sin, cos, sin)`` of the two pairs in each vec4 of a head."""
    hs = cfg.head_size
    out = array.array('f')
    for t in range(T):
        for j in range(0, hs // 2, 2):
            a0 = t * cfg.rope_theta ** (-2.0 * j / hs)
            a1 = t * cfg.rope_theta ** (-2.0 * (j + 1) / hs)
            out.extend((math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)))
    return out


class State8:
    """8-bit blockwise AdamW (and Muon momentum) state, as ``vabs_gpt.model.State8``
    but for this model's tensors. ``ChatRunner`` drives it through vabs_gpt's
    ``Runner._update8``, which reads the same attributes."""

    def __init__(self, dev, cfg, muon, data=None):
        self.dev, self.muon = dev, bool(muon)
        table, _ = QF.block_table(adam_segments(cfg, muon))
        self.blocks = len(table) // 4
        regions = dev.arena(State8.sizes(cfg, muon), MemoryUsage.DEVICE_LOCAL)
        tab, self.state, self.scales, code = regions[:4]
        self.groups = [dict(blocks=gb, state=st, scales=sc) for gb, st, sc in
                       zip(State8._group_blocks(cfg, muon), regions[4::2], regions[5::2])]
        dev.upload(tab, table.tobytes())
        dev.upload(code, QF.code_table().tobytes())
        self.table, self.code = tab.device_address, code.device_address
        for r in regions[1:3] + regions[4:]:
            dev.upload(r, bytes(r.size))
        if data is not None:
            self.load(data)

    @staticmethod
    def _group_blocks(cfg, muon):
        return [QF.n_blocks(len(mats) * m * n) for m, n, mats in muon_groups(cfg)] if muon else []

    @staticmethod
    def sizes(cfg, muon):
        nb = len(QF.block_table(adam_segments(cfg, muon))[0]) // 4
        sizes = [16 * nb, 2 * QF.BLOCKSIZE * nb, 4 * QF.adam_scales_floats(nb), 4 * 512]
        for gb in State8._group_blocks(cfg, muon):
            sizes += [QF.BLOCKSIZE * gb, 4 * QF.momentum_scales_floats(gb)]
        return sizes

    @staticmethod
    def saved_bytes(cfg, muon):
        nb = len(QF.block_table(adam_segments(cfg, muon))[0]) // 4
        return QF.adam_state_bytes(nb) + sum(QF.momentum_state_bytes(gb)
                                             for gb in State8._group_blocks(cfg, muon))

    def _saved(self):
        out = [(self.state, self.state.size), (self.scales, 8 * self.blocks)]
        for g in self.groups:
            out += [(g['state'], g['state'].size), (g['scales'], 4 * g['blocks'])]
        return out

    def save(self):
        return b''.join(self.dev.download(r)[:n] for r, n in self._saved())

    def load(self, data):
        if len(data) != sum(n for _, n in self._saved()):
            raise ValueError(f'8-bit optimizer state of {len(data)} bytes does not fit')
        o = 0
        for r, n in self._saved():
            self.dev.upload(r, data[o:o + n] + bytes(r.size - n))
            o += n


# -- device memory estimates -----------------------------------------------------

def model_bytes(cfg, optim_bits=8, muon=False):
    n = param_count(cfg)
    if optim_bits == 32:
        return 7 * 4 * n
    return 4 * 4 * n + sum(State8.sizes(cfg, muon))


def linear_rows(N, M):
    """Rows per chunk of a weight-gradient partial sum: at least 128, and few enough
    chunks that ``chunks * M`` floats fit in ``PARTIAL_FLOATS``."""
    return max(128, _ceil(N, max(1, PARTIAL_FLOATS // M)))


def partial_floats(cfg, N):
    sizes = [NORM_STRIDE, _ceil(N, NORM_ROWS) * cfg.dim]
    for name, _ in param_layout(cfg):
        shape = matrix_shape(cfg, name)
        if shape:
            M = shape[0] * shape[1]
            sizes.append(_ceil(N, linear_rows(N, M)) * M)
    return max(sizes)


def _runner_sizes(cfg, B, T, mode):
    C, V, H, F, L, S = cfg.dim, cfg.vocab, cfg.heads, cfg.mlp, cfg.layers, cfg.qkv_dim
    N, att = B * T, B * H * T * _ceil(T, 4) * 4
    # ln1, r1, qkv, att, atty, res2, ln2, r2, pre, h, then the layer's output residual
    per_layer = [N * C, N, N * S, att, N * C, N * C, N * C, N, 2 * N * F, N * F, N * C]
    if mode == 'infer':   # one shared set (without qkv), then a qkv per layer
        layers = [x for i, x in enumerate(per_layer[:-1]) if i != 2] + [N * S] * L
    else:
        layers = per_layer * L
    sizes = [N * C] + layers + [N * C, N, T * cfg.head_size, _ceil(N, LOSS_SUMS) * LOSS_SUMS]
    if mode == 'infer':
        sizes += [B * C, B * V]
    else:
        chunks = _ceil(V // 4, K.CE_CHUNK)
        sizes += [N * V, 2 * N * chunks, N]
    if mode == 'train':
        sizes += [N * C, N * C, N * C, N * F, 2 * N * F, N * S, att] + [partial_floats(cfg, N)] * 2
    return [4 * s for s in sizes]


def activation_bytes(cfg, batch, mode='train', T=None):
    return sum(_runner_sizes(cfg, batch, T or cfg.ctx, mode))


def optimizer_bytes(cfg, muon, optim_bits):
    sizes = []
    if muon and optim_bits == 32:
        sizes.append(4 * param_count(cfg))
    if muon:
        for m, n, mats in muon_groups(cfg):
            nb = len(mats)
            sizes += [4 * nb * m * n] * 5 + [4 * nb * m * m] * 2 + [
                4 * nb, 4 * nb * m * m * max(1, GM.muon_splits(n) - 1)]
    return sizes


# -- device and model ---------------------------------------------------------------

def _new_arenas(ctx, before):
    return [a for a in ctx._arenas if id(a) not in before]


def release_arenas(ctx, arenas, recordings=()):
    """Free device memory now instead of when the context is destroyed."""
    import vk
    vk.device_wait_idle(ctx.device)
    for r in recordings:
        r.destroy()
    for a in arenas:
        if a in ctx._arenas:
            a.destroy()
            ctx._arenas.remove(a)
    arenas.clear()


COOP_TARGET_GROUPS = 1024   # workgroups a split weight-gradient product aims for
PRECISIONS = ('f32', 'bf16')


def coop_shapes_ok(*dims):
    """Whether ``mm_coop`` can run a product with these sizes (all multiples of 16)."""
    return all(d % 16 == 0 and d > 0 for d in dims)


def coop_splits(M, N, Kd, budget=None):
    """``(chunks, rows per chunk)`` splitting a weight gradient's reduction over ``Kd``
    rows: enough workgroups to fill the GPU, chunks of at least 256 rows, and
    ``chunks * M * N`` floats within ``budget``."""
    tiles = _ceil(M, K.CM_BM) * _ceil(N, K.CM_BN)
    splits = max(1, min(_ceil(COOP_TARGET_GROUPS, tiles), Kd // 256))
    if budget is not None:
        splits = max(1, min(splits, budget // (M * N)))
    kc = _ceil(_ceil(Kd, splits), K.CM_BK) * K.CM_BK
    return _ceil(Kd, kc), kc


def add_precision_arg(parser):
    """``--precision``: the arithmetic of the linear layers (see ``ChatModel``)."""
    parser.add_argument(
        '--precision', choices=PRECISIONS, default='f32',
        help='bf16: linear layers on bfloat16 tensor cores (cooperative matrices), '
             'float32 accumulation; parameters and activations stay float32')


class Device(GM.Device):
    """vabs_gpt's ``Device`` with this model's kernels first in the lookup.

    ``precision`` is the default of the models made on it (``ChatModel``).
    """

    def __init__(self, ctx, precision='f32'):
        super().__init__(ctx)
        if precision not in PRECISIONS:
            raise ValueError(f'precision must be one of {PRECISIONS}')
        self.precision = precision

    @staticmethod
    def coop_op(c, a, b, M, N, Kd, *, a_col=False, b_col=False, res=0, kc=None, nb=1,
                sa=0, sb=0, sc_stride=0, scaled=None, ldc=None, lda=None, ldb=None,
                heads=None, causal=None):
        """``(op, chunks)`` for ``C[M, N] = A[M, K] B[K, N]`` (+ ``res``) on bfloat16
        tensor cores, where ``op`` is the ``(kernel, push, groups, 1)`` of ``run``.

        ``a_col``/``b_col`` say an operand is stored column-major (A as [K, M], B as
        [N, K]). With ``kc`` the reduction is split into chunks of ``kc`` rows, chunk
        ``z`` writing its own ``M * N`` floats at ``c + 4 * z * M * N``. ``nb``
        matrices lie ``sa``, ``sb`` and ``sc_stride`` floats apart in A, B and C.
        ``scaled = (d, scales, alpha, pab, beta, pd)`` makes matrix i
        ``alpha s_i^pab A B + beta s_i^pd D`` (D read only when beta != 0).
        ``ld*`` override the leading dimensions (floats between stored rows).
        ``heads = (nh, (ga, sha), (gb, shb), (gc, shc))`` splits matrix i into
        ``(i // nh, i % nh)``: operand X starts ``(i // nh) sX + (i % nh // gX) shX``
        floats in. ``causal`` is ``'end'`` (A lower triangular), ``'begin'`` (A upper
        triangular) or ``'skip'`` (tiles above the diagonal are left unwritten).
        """
        if not coop_shapes_ok(M, N, Kd):
            raise ValueError(f'mm_coop needs sizes that are multiples of 16, got {M, N, Kd}')
        kc = kc or _ceil(Kd, K.CM_BK) * K.CM_BK
        d, sc, alpha, pab, beta, pd = scaled or (0, 0, 1.0, 0, 0.0, 0)
        flags = ((K.CM_A_COLUMN_MAJOR if a_col else 0) | (K.CM_B_COLUMN_MAJOR if b_col else 0)
                 | (K.CM_RESIDUAL if res else 0) | (K.CM_SCALE if scaled else 0)
                 | {None: 0, 'end': K.CM_CAUSAL_END, 'begin': K.CM_CAUSAL_BEGIN,
                    'skip': K.CM_SKIP_ABOVE}[causal])
        nh, (ga, sha), (gb, shb), (gc, shc) = heads or (1, (1, 0), (1, 0), (1, 0))
        push = K.MMCPush(a=a, b=b, c=c, res=res, d=d, sc=sc, M=M, N=N, K=Kd,
                         lda=lda or (M if a_col else Kd), ldb=ldb or (Kd if b_col else N),
                         ldc=ldc or N, flags=flags, KC=kc, split=M * N, sa=sa, sb=sb,
                         scs=sc_stride, nh=nh, ga=ga, gb=gb, gc=gc, sha=sha, shb=shb, shc=shc,
                         pab=pab, pd=pd, alpha=alpha, beta=beta)
        splits = _ceil(Kd, kc)
        return ('mm_coop', push, (_ceil(N, K.CM_BN), _ceil(M, K.CM_BM), nb * splits), 1), splits

    def mm_coop(self, frame, *args, barrier=True, **kw):
        """Record ``coop_op(*args, **kw)``; returns the number of reduction chunks."""
        op, splits = self.coop_op(*args, **kw)
        self.run(frame, *op, barrier=barrier)
        return splits

    def run(self, frame, kernel, push, n, gy=1, barrier=True):
        """As ``vabs_gpt.model.Device.run``; a tuple ``n`` gives the workgroup counts."""
        if not isinstance(n, tuple):
            return super().run(frame, kernel, push, n, gy, barrier)
        frame.bind_compute_pipeline(self.pipeline(kernel, push))
        frame.push_constants(push)
        frame.dispatch(*n)
        if barrier:
            compute_barrier(frame)

    def pipeline(self, kernel, push):
        sizes = {}
        if kernel in SPECIALIZED and hasattr(K, kernel):
            sizes = {name: getattr(push, name)
                     for name, (_, kind) in type(push).std430_layout.items() if kind is uint32}
        key = ('chat', kernel, *sizes.items())
        if key not in self.pipes:
            shader = getattr(K, kernel, None) or getattr(GK, kernel, None) or getattr(Q, kernel)
            self.pipes[key] = ComputePipeline(self.ctx, shader.specialize(**sizes), name=kernel)
            self.ctx.on_destroy(self.pipes[key].destroy)
        return self.pipes[key]


class ChatModel:
    """Parameters, gradients, the forward copy and the optimizer state on the device."""

    def __init__(self, dev, cfg, params=None, seed=0, m=None, v=None, step=0, optim_bits=8,
                 muon=False, q8=None, tokenizer=None, precision=None):
        if optim_bits not in (8, 32):
            raise ValueError('optim_bits must be 8 or 32')
        precision = precision or getattr(dev, 'precision', 'f32')
        if precision not in PRECISIONS:
            raise ValueError(f'precision must be one of {PRECISIONS}')
        if precision == 'bf16' and not getattr(dev.ctx, 'supports_bf16_matmul', False):
            raise ValueError('bf16 needs bfloat16 cooperative matrices (VK_KHR_cooperative_matrix '
                             'with VK_KHR_shader_bfloat16), which this device lacks')
        # 'bf16': linear layers multiply bfloat16-rounded operands on the tensor cores,
        # accumulating in float32; parameters, activations and gradients stay float32
        self.precision = precision
        self.dev, self.cfg, self.step, self.optim_bits = dev, cfg, step, optim_bits
        before = {id(a) for a in dev.ctx._arenas}
        self.tokenizer = tokenizer   # merges, saved in checkpoints
        self.offsets = offsets(cfg)
        self.n_params = n = param_count(cfg)
        if params is None:
            params = init_params(cfg, seed)
        if len(params) != n:
            raise ValueError(f'expected {n} parameters, got {len(params)}')
        self.q8 = self.m = self.v = self.decay = None
        if optim_bits == 32:
            self.params, self.grads, self.m, self.v, self.decay, self.fwd, self.fwd_index = \
                dev.arena([4 * n] * 7, MemoryUsage.DEVICE_LOCAL)
            dev.upload(self.m, m.tobytes() if m is not None else bytes(4 * n))
            dev.upload(self.v, v.tobytes() if v is not None else bytes(4 * n))
            dev.upload(self.decay, decay_mask(cfg).tobytes())
        else:
            self.params, self.grads, self.fwd, self.fwd_index = \
                dev.arena([4 * n] * 4, MemoryUsage.DEVICE_LOCAL)
            self.q8 = State8(dev, cfg, muon, q8)
        dev.upload(self.fwd_index, transpose_index(cfg).tobytes())
        dev.upload(self.params, params.tobytes())
        with one_shot_frame(dev.ctx) as frame:
            self.refresh_fwd(frame)
        self._arenas = _new_arenas(dev.ctx, before)

    def release(self):
        """Free the model's device memory (its runners must be released first)."""
        release_arenas(self.dev.ctx, self._arenas)

    def refresh_fwd(self, frame):
        self.dev.run(frame, 'gather', GK.GatherPush(
            out=self.fwd.device_address, src=self.params.device_address,
            idx=self.fwd_index.device_address, n=self.n_params), self.n_params)

    def p(self, name):
        return self.params.device_address + 4 * self.offsets[name]

    def uses_coop(self, *dims):
        """Whether a product of these sizes runs on the bf16 tensor cores."""
        return self.precision == 'bf16' and coop_shapes_ok(*dims)

    def linear(self, frame, out, inp, name, N, C, OC, res=0, barrier=True):
        """``out[N, OC] = inp[N, C] W^T (+ res)`` for the matrix parameter ``name``."""
        if self.uses_coop(N, C, OC):
            self.dev.mm_coop(frame, out, inp, self.p(name), N, OC, C, b_col=True, res=res,
                             barrier=barrier)
        else:
            self.dev.run(frame, 'matmul_fwd', K.MMPush(out=out, inp=inp, w=self.f(name), res=res,
                                                       N=N, C=C, OC=OC, flags=1 if res else 0),
                         _ceil(N, 4) * OC // 4, barrier=barrier)

    def f(self, name):
        return self.fwd.device_address + 4 * self.offsets[name]

    def g(self, name):
        return self.grads.device_address + 4 * self.offsets[name]

    def get_params(self):
        return array.array('f', self.dev.download(self.params))

    def get_grads(self):
        return array.array('f', self.dev.download(self.grads))

    def save(self, path, extra=None, optimizer=True):
        header = {'config': asdict(self.cfg), 'step': self.step, 'optimizer': optimizer,
                  'optim_bits': self.optim_bits, 'tokenizer': self.tokenizer, **(extra or {})}
        if self.q8:
            header['optim_layout'] = 'muon' if self.q8.muon else 'adamw'
        data = json.dumps(header).encode()
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(MAGIC + struct.pack('<I', len(data)) + data)
            f.write(self.dev.download(self.params))
            if optimizer and self.q8:
                f.write(self.q8.save())
            elif optimizer:
                f.write(self.dev.download(self.m))
                f.write(self.dev.download(self.v))
        os.replace(tmp, path)

    @staticmethod
    def read_header(path):
        with open(path, 'rb') as f:
            if f.read(4) != MAGIC:
                raise ValueError(f'{path} is not a chat checkpoint')
            return json.loads(f.read(struct.unpack('<I', f.read(4))[0]))

    @staticmethod
    def read(path):
        """``(header, params, state)``; ``state`` holds the ChatModel arguments that restore
        the optimizer state, or is None."""
        with open(path, 'rb') as f:
            if f.read(4) != MAGIC:
                raise ValueError(f'{path} is not a chat checkpoint')
            header = json.loads(f.read(struct.unpack('<I', f.read(4))[0]))
            body = f.read()
        cfg = ChatConfig(**header['config'])
        n = param_count(cfg)
        params, rest = array.array('f', body[:4 * n]), body[4 * n:]
        state = None
        if header.get('optimizer') and header.get('optim_bits', 8) == 8:
            muon = header.get('optim_layout') == 'muon'
            if len(rest) == State8.saved_bytes(cfg, muon):
                state = {'optim_bits': 8, 'muon': muon, 'q8': rest}
        elif header.get('optimizer') and len(rest) == 8 * n:
            moments = array.array('f', rest)
            state = {'optim_bits': 32, 'm': moments[:n], 'v': moments[n:]}
        return header, params, state


# -- runner -------------------------------------------------------------------------

def group_tokens(toks):
    """``(starts, order, ids)`` that group the positions of ``toks`` by id, for
    ``embed_bwd``: ``starts = [groups, s_0, ..., s_groups]``."""
    order = sorted(range(len(toks)), key=toks.__getitem__)
    starts, ids, prev = [0], array.array('I'), -1
    get = toks.__getitem__
    for k, pos in enumerate(order):
        t = get(pos)
        if t != prev:
            ids.append(t)
            starts.append(k)
            prev = t
    starts.append(len(toks))
    starts[0] = len(ids)
    return array.array('I', starts), array.array('I', order), ids


class _Slot(GM._Slot):
    pass


class ChatRunner:
    """Activations for ``batch`` sequences of ``T`` tokens and recorded command buffers.

    mode 'train': ``submit_step`` / ``train_step`` (as ``vabs_gpt.model.Runner``).
    'eval': ``run(inp, tgt)`` returns ``(loss sum, scored tokens)``.
    'infer': ``next_logits(tokens, pos)`` returns the next-token logits at ``pos[b]`` of
    each sequence. ``T`` may be shorter than the context (RoPE positions start at
    0 in every sequence), so short prompts run on a short recording.
    """

    _each = GM.Runner._each
    _newton_schulz_f32 = GM.Runner._newton_schulz

    def _newton_schulz(self, frame):
        if not (self.model.precision == 'bf16'
                and all(coop_shapes_ok(g['m'], g['n']) for g in self.groups)):
            return self._newton_schulz_f32(frame)
        self._newton_schulz_coop(frame)

    def _newton_schulz_coop(self, frame):
        """``vabs_gpt.model.Runner._newton_schulz`` with its products on the bf16 tensor
        cores (as Muon is usually run). X^T is X read column-major, so the transposed
        copies are neither read nor written."""
        m, mu, groups = self.model, self.muon, self.groups
        a, b, c = GM.NS_COEFFS

        def each(op_of):
            for k, g in enumerate(groups):
                op, _ = op_of(g)
                self.dev.run(frame, *op, barrier=k == len(groups) - 1)

        coop = self.dev.coop_op
        for s in range(mu.ns_steps):
            x, y = s % 2, (s + 1) % 2   # X is read from x[x] and written to x[y]
            first = int(s == 0)
            # A = X X^T
            each(lambda g: coop(g['A'], g['x'][x], g['x'][x], g['m'], g['m'], g['n'],
                                b_col=True, nb=g['nb'], sa=g['m'] * g['n'], sb=g['m'] * g['n'],
                                sc_stride=g['m'] * g['m']))
            if first:
                # the first A is of the unnormalized X; its trace is |X|^2 (see vabs_gpt)
                self._each(frame, 'muon_norm', lambda g: GK.MuonNormPush(
                    a=g['A'], inv=g['inv'], m=g['m'], nb=g['nb']), lambda g: g['nb'])
            # B = c s^4 A A + b s^2 A (s = 1/|X| on the first step, else 1)
            each(lambda g: coop(g['Bm'], g['A'], g['A'], g['m'], g['m'], g['m'], nb=g['nb'],
                                sa=g['m'] * g['m'], sb=g['m'] * g['m'],
                                sc_stride=g['m'] * g['m'],
                                scaled=(g['A'], g['inv'], c, 4 * first, b, 2 * first)))
            # the new X = s (B X + a X)
            each(lambda g: coop(g['x'][y], g['Bm'], g['x'][x], g['m'], g['n'], g['m'],
                                nb=g['nb'], sa=g['m'] * g['m'], sb=g['m'] * g['n'],
                                sc_stride=g['m'] * g['n'],
                                scaled=(g['x'][x], g['inv'], 1.0, first, a, first)))
        out = mu.ns_steps % 2
        self._each(frame, 'muon_apply', lambda g: GK.MuonApplyPush(
            params=m.params.device_address, x=g['x'][out], idx=g['idx'],
            hyper=self.hyper.device_address, scale=mu.scale * math.sqrt(g['n']),
            n=g['nb'] * g['m'] * g['n']), lambda g: g['nb'] * g['m'] * g['n'])
    _update32 = GM.Runner._update32
    _update8 = GM.Runner._update8

    def __init__(self, model, batch, mode, accum=1, muon=None, T=None, weighted=False):
        if mode not in ('train', 'eval', 'infer'):
            raise ValueError(mode)
        cfg = model.cfg
        self.model, self.dev, self.mode, self.cfg = model, model.dev, mode, cfg
        self.muon = muon if mode == 'train' else None
        if mode == 'train' and model.q8 and model.q8.muon != bool(muon):
            raise ValueError('the 8-bit optimizer state was laid out for '
                             + ('Muon' if model.q8.muon else 'AdamW alone'))
        self.qk_clip = 0.0
        self.T = T = T or cfg.ctx
        if T > cfg.ctx:
            raise ValueError(f'T={T} exceeds the context of {cfg.ctx}')
        # bf16 attention: the score and value products on the tensor cores
        self.coop_attention = model.uses_coop(T, cfg.head_size)
        B, C, V, H, L = batch, cfg.dim, cfg.vocab, cfg.heads, cfg.layers
        self.B, self.N, self.accum = B, B * T, accum
        N = self.N
        if B * H * T * _ceil(T, 4) * 4 >= 2 ** 31 or N * max(V, 2 * cfg.mlp) >= 2 ** 31:
            raise ValueError('batch x context too large for 32-bit indices')
        if _ceil(N, NORM_ROWS) > GM.MAX_GROUPS_Y:
            raise ValueError('too many tokens per micro-batch')
        train = mode == 'train'
        before = {id(a) for a in self.dev.ctx._arenas}
        regions = iter(self.dev.arena(_runner_sizes(cfg, B, T, mode), MemoryUsage.DEVICE_LOCAL))
        a = lambda: next(regions).device_address  # noqa: E731
        self.res = [a()]
        self.layer = []
        names = ('ln1', 'r1', 'qkv', 'att', 'atty', 'res2', 'ln2', 'r2', 'pre', 'h')
        if mode == 'infer':
            # nothing is kept for a backward pass: the layers share one set of activations
            # and one residual stream, and only each layer's qkv rows (the key/value cache
            # of decode.Decoder) are its own
            shared = {name: a() for name in names if name != 'qkv'}
            for _ in range(L):
                self.layer.append(dict(shared, qkv=a()))
                self.res.append(self.res[0])
        else:
            for _ in range(L):
                self.layer.append({name: a() for name in names})
                self.res.append(a())
        self.lnf, self.rf = a(), a()
        rope_region = next(regions)
        self.dev.upload(rope_region, rope_table(cfg, T).tobytes())
        self.rope = rope_region.device_address
        losses = next(regions)
        self.dev.upload(losses, bytes(losses.size))
        self.losses = losses.device_address
        if mode == 'infer':
            self.last = a()
            self.logits_region = next(regions)
        else:
            self.logits_region = next(regions)
            self.ce_part, self.lse = a(), a()
        self.logits = self.logits_region.device_address
        if train:
            self.dres, self.dln, self.datty, self.dh, self.dpre, self.dqkv, self.datt = (
                a() for _ in range(7))
            self.part = (a(), a())
            self.partial_floats = partial_floats(cfg, N)   # floats in each of self.part
            self._setup_optimizer(cfg)
        loss_rows = _ceil(N, LOSS_SUMS)
        self.loss_rows = loss_rows
        self.weighted = weighted and train
        host = [4 * N, 4 * N, 4 * (N + 2), 4 * N, 4 * N, 4 * LOSS_SUMS, GK.Hyper.std430_size,
                4 * 4, 4 * N if self.weighted else 4]
        # inp, tgt, starts, order, ids, loss sums, hyper, stats, ce scale (or token weights)
        if mode == 'infer':
            host += [4 * B, 4 * B * V]   # positions, logits
        variants = [(False, False)]
        if train:
            variants = [(True, True)] if accum == 1 else [(True, False), (False, False),
                                                          (False, True)]
        self.slots, self._turn, self._flying = [], 0, None
        for _ in range(2 if train else 1):
            slot = _Slot(self.dev.arena(host, MemoryUsage.HOST_VISIBLE))
            self.slots.append(slot)
            (self.inp, self.tgt, self.starts, self.order, self.ids, self.loss_sums,
             self.hyper, self.stats, self.ce_scale) = slot.regions[:9]
            if mode == 'infer':
                self.pos, self.logits_host = slot.regions[9:11]
            for zero, update in variants:
                rec = slot.recs[zero, update] = Recording(self.dev.ctx)
                with rec.record() as frame:
                    compute_barrier(frame)
                    if zero:
                        frame.fill_buffer(model.grads, 0)
                        compute_barrier(frame)
                    self._forward(frame)
                    if mode != 'infer':
                        self._loss(frame, train)
                    if train:
                        self._backward(frame)
                    if update:
                        self._update(frame)
                    if mode == 'infer':
                        frame.copy_buffer(self.logits_region, self.logits_host, 4 * B * V)
                        compute_barrier(frame)
        slot = self.slots[0]
        (self.inp, self.tgt, self.starts, self.order, self.ids, self.loss_sums,
         self.hyper, self.stats, self.ce_scale) = slot.regions[:9]
        if mode == 'infer':
            self.pos, self.logits_host = slot.regions[9:11]
        self.rec = slot.recs[variants[0]]
        self._arenas = _new_arenas(self.dev.ctx, before)

    def release(self):
        """Free this runner's device memory and recordings now."""
        release_arenas(self.dev.ctx, self._arenas,
                       [r for slot in self.slots for r in slot.recs.values()])

    # -- recording ----------------------------------------------------------

    def _forward(self, frame):
        m, cfg, N, T, B = self.model, self.cfg, self.N, self.T, self.B
        run = self.dev.run
        C, V, H, F, S = cfg.dim, cfg.vocab, cfg.heads, cfg.mlp, cfg.qkv_dim
        hs4, C4, KV4, S4 = cfg.head_size // 4, C // 4, cfg.kv_dim // 4, S // 4
        G = H // cfg.kv_heads
        eps = cfg.norm_eps
        run(frame, 'embed_fwd', K.EmbedPush(x=self.res[0], wte=m.p('wte'),
                                            toks=self.inp.device_address, N=N, C=C), N * C4)
        attn = dict(B=B, T=T, H=H, G=G, hs4=hs4, S4=S4, C4=C4, KV4=KV4)
        for i, a in enumerate(self.layer):
            h = f'h{i}.'
            run(frame, 'rmsnorm_fwd', K.NormPush(out=a['ln1'], inp=self.res[i], w=m.p(h + 'n1'),
                                                 rstd=a['r1'], N=N, C=C, eps=eps), N)
            m.linear(frame, a['qkv'], a['ln1'], h + 'qkv', N, C, S)
            run(frame, 'rope', K.RopePush(x=a['qkv'], cs=self.rope, N=N, T=T, S4=S4,
                                          R4=C4 + KV4, hs4=hs4), N * (C4 + KV4))
            if self.coop_attention:
                self._attention_coop(frame, a['qkv'], a['att'], a['atty'])
            else:
                run(frame, 'attn_softmax', K.AttnPush(qkv=a['qkv'], att=a['att'], **attn),
                    B * H * T)
                run(frame, 'attn_mm', K.AttnMMPush(out=a['atty'], x=a['qkv'] + 16 * (C4 + KV4),
                                                   s=a['att'], B=B, T=T, H=H, G=G, hs4=hs4,
                                                   xs=S4, os=C4), B * H * _ceil(T, 4) * hs4)
            m.linear(frame, a['res2'], a['atty'], h + 'proj', N, C, C, res=self.res[i])
            run(frame, 'rmsnorm_fwd', K.NormPush(out=a['ln2'], inp=a['res2'], w=m.p(h + 'n2'),
                                                 rstd=a['r2'], N=N, C=C, eps=eps), N)
            m.linear(frame, a['pre'], a['ln2'], h + 'fc', N, C, 2 * F)
            run(frame, 'swiglu_fwd', K.GluPush(pre=a['pre'], h=a['h'], N=N, F=F), N * F // 4)
            m.linear(frame, self.res[i + 1], a['h'], h + 'down', N, F, C, res=a['res2'])
        run(frame, 'rmsnorm_fwd', K.NormPush(out=self.lnf, inp=self.res[-1], w=m.p('nf'),
                                             rstd=self.rf, N=N, C=C, eps=eps), N)
        if self.mode == 'infer':
            run(frame, 'gather_rows', K.RowsPush(out=self.last, src=self.lnf,
                                                 pos=self.pos.device_address, B=B, T=T, C=C),
                B * C4)
            m.linear(frame, self.logits, self.last, 'wte', B, C, V)
        else:
            m.linear(frame, self.logits, self.lnf, 'wte', N, C, V)

    def _attention_dims(self):
        cfg = self.cfg
        return (self.B, self.T, cfg.heads, cfg.heads // cfg.kv_heads, cfg.head_size, cfg.dim,
                cfg.kv_dim, cfg.qkv_dim)

    def _attention_coop(self, frame, qkv, att, atty):
        """Causal attention with bf16 products: scores = q k^T (tiles above the diagonal
        skipped), the softmax per row, then y = att v (stopping at the diagonal)."""
        B, T, H, G, hs, C, KV, S = self._attention_dims()
        TT = T * T
        coop, run = self.dev.coop_op, self.dev.run
        op, _ = coop(att, qkv, qkv + 4 * C, T, T, hs, b_col=True, lda=S, ldb=S, ldc=T,
                     nb=B * H, sa=T * S, sb=T * S, sc_stride=H * TT,
                     heads=(H, (1, hs), (G, hs), (1, TT)), causal='skip')
        run(frame, *op)
        run(frame, 'attn_softmax_rows', K.AttnRowsPush(att=att, rows=B * H * T, T=T, H=H,
                                                       hs4=hs // 4), B * H * T * K.ROWS_WG)
        op, _ = coop(atty, att, qkv + 4 * (C + KV), T, hs, T, lda=T, ldb=S, ldc=C, nb=B * H,
                     sa=H * TT, sb=T * S, sc_stride=T * C,
                     heads=(H, (1, TT), (G, hs), (1, hs)), causal='end')
        run(frame, *op)

    def _attention_coop_bwd(self, qkv, att, atty):
        """The stages (lists of ops) of the attention backward pass with bf16 products.

        dp = dout v^T; the row kernel turns it into the score gradient ds; dq = ds k;
        dk and dv sum over the G query heads of each key/value head, one head per
        dispatch, each adding to the previous (the residual)."""
        B, T, H, G, hs, C, KV, S = self._attention_dims()
        TT, HK = T * T, H // G
        coop, datt, dout, dqkv = self.dev.coop_op, self.datt, self.datty, self.dqkv
        dp, _ = coop(datt, dout, qkv + 4 * (C + KV), T, T, hs, b_col=True, lda=C, ldb=S, ldc=T,
                     nb=B * H, sa=T * C, sb=T * S, sc_stride=H * TT,
                     heads=(H, (1, hs), (G, hs), (1, TT)), causal='skip')

        def summed(out, a, b, ldb, sb):
            """out[b, g] = sum_e a[b, g G + e]^T b[b, g G + e] (a: [T, T] per query head)."""
            ops = []
            for e in range(G):
                op, _ = coop(out, a + 4 * e * TT, b + 4 * e * hs, T, hs, T, a_col=True, lda=T,
                             ldb=ldb, ldc=S, res=out if e else 0, nb=B * HK, sa=H * TT, sb=sb,
                             sc_stride=T * S, heads=(HK, (1, G * TT), (1, G * hs), (1, hs)),
                             causal='begin')
                ops.append(op)
            return ops

        dv = summed(dqkv + 4 * (C + KV), att, dout, C, T * C)
        rows = ('attn_bwd_rows', K.AttnRowsPush(att=att, datt=datt, dout=dout, y=atty,
                                                rows=B * H * T, T=T, H=H, hs4=hs // 4),
                B * H * T * K.ROWS_WG, 1)
        dq, _ = coop(dqkv, datt, qkv + 4 * C, T, hs, T, lda=T, ldb=S, ldc=S, nb=B * H,
                     sa=H * TT, sb=T * S, sc_stride=T * S, heads=(H, (1, TT), (G, hs), (1, hs)),
                     causal='end')
        dk = summed(dqkv + 4 * C, datt, qkv, S, T * S)
        # dv's first head runs beside dp; every later dispatch of a sum needs the one before
        return ([dp, dv[0]], *([op] for op in dv[1:]), [rows], [dq, dk[0]],
                *([op] for op in dk[1:]))

    def _loss(self, frame, train):
        N, V = self.N, self.cfg.vocab
        chunks = _ceil(V // 4, K.CE_CHUNK)
        ce = K.CEPush(logits=self.logits, logit1=self.logits, part=self.ce_part, lse=self.lse,
                      losses=self.losses, targets=self.tgt.device_address,
                      scale=self.ce_scale.device_address, N=N, V=V, chunks=chunks,
                      wstride=1 if self.weighted else 0)
        run = self.dev.run
        run(frame, 'ce_partial', ce, N * chunks)
        run(frame, 'ce_finish', ce, N)
        run(frame, 'reduce_partials', GK.ReducePush(out=self.loss_sums.device_address,
                                                    part=self.losses, M=LOSS_SUMS,
                                                    K=self.loss_rows), LOSS_SUMS,
            barrier=not train)
        if train:
            run(frame, 'ce_grad', ce, N * V // 4)

    def _backward(self, frame):
        m, cfg, N, T, B = self.model, self.cfg, self.N, self.T, self.B
        C, V, H, F, S = cfg.dim, cfg.vocab, cfg.heads, cfg.mlp, cfg.qkv_dim
        hs4, C4, KV4, S4 = cfg.head_size // 4, C // 4, cfg.kv_dim // 4, S // 4
        G = H // cfg.kv_heads
        N4, TB = _ceil(N, 4), _ceil(T, 4)
        eps = cfg.norm_eps
        mm = dict(B=B, T=T, H=H, G=G, hs4=hs4)
        stages = []   # [(dispatches, (partials, gradient, M, chunks) or None)]

        def part():
            return self.part[len(stages) % 2]

        def stage(*ops, grad=None):
            stages.append((ops, grad and (part(), *grad)))

        def linear(out, dout, inp, name, c_in, oc):
            M = oc * c_in
            if m.uses_coop(N, c_in, oc):
                # dx = dy W (W row-major [oc, c_in]); dW = dy^T x in row chunks
                dx, _ = self.dev.coop_op(out, dout, m.p(name), N, c_in, oc)
                k, kc = coop_splits(oc, c_in, N, self.partial_floats)
                dw, _ = self.dev.coop_op(part(), dout, inp, oc, c_in, N, a_col=True, kc=kc)
                stage(dx, dw, grad=(m.g(name), M, k))
                return
            R = linear_rows(N, M)
            k = _ceil(N, R)
            stage(('matmul_bwd_inp', K.MMPush(out=out, inp=dout, w=m.p(name), N=N, C=c_in,
                                              OC=oc), N4 * c_in // 4, 1),
                  ('matmul_bwd_w_partial', K.MMPush(out=part(), inp=inp, dout=dout, N=N,
                                                    C=c_in, OC=oc, R=R),
                   (oc // 4) * (c_in // 4), k),
                  grad=(m.g(name), M, k))

        def norm(name, inp, rstd, accumulate):
            k = _ceil(N, NORM_ROWS)
            p = dict(inp=inp, w=m.p(name), rstd=rstd, dout=self.dln, N=N, C=C, R=NORM_ROWS,
                     eps=eps)
            stage(('rmsnorm_bwd', K.NormPush(out=self.dres, accumulate=accumulate, **p), N, 1),
                  ('rmsnorm_bwd_partial', K.NormPush(part=part(), **p), C4, k),
                  grad=(m.g(name), C, k))

        # the logits now hold d(loss)/d(logits); the head is the tied embedding
        linear(self.dln, self.logits, self.lnf, 'wte', C, V)
        norm('nf', self.res[-1], self.rf, 0)
        for i in reversed(range(cfg.layers)):
            a, h = self.layer[i], f'h{i}.'
            linear(self.dh, self.dres, a['h'], h + 'down', F, C)
            stage(('swiglu_bwd', K.GluPush(pre=a['pre'], h=self.dh, dpre=self.dpre, N=N, F=F),
                   N * F // 4, 1))
            linear(self.dln, self.dpre, a['ln2'], h + 'fc', C, 2 * F)
            norm(h + 'n2', a['res2'], a['r2'], 1)
            linear(self.datty, self.dres, a['atty'], h + 'proj', C, C)
            # dv = att^T dout and the score gradient, then dq = datt k and dk = datt^T q
            if self.coop_attention:
                for ops in self._attention_coop_bwd(a['qkv'], a['att'], a['atty']):
                    stage(*ops)
            else:
                stage(('attn_bwd_scores', K.AttnPush(qkv=a['qkv'], att=a['att'], datt=self.datt,
                                                     dout=self.datty, y=a['atty'], B=B, T=T,
                                                     H=H, G=G, hs4=hs4, S4=S4, C4=C4, KV4=KV4),
                       B * H * T, 1),
                      ('attn_mm_t', K.AttnMMPush(out=self.dqkv + 16 * (C4 + KV4), x=self.datty,
                                                 s=a['att'], xs=C4, os=S4, **mm),
                       B * cfg.kv_heads * TB * hs4, 1))
                stage(('attn_mm', K.AttnMMPush(out=self.dqkv, x=a['qkv'] + 16 * C4, s=self.datt,
                                               xs=S4, os=S4, **mm), B * H * TB * hs4, 1),
                      ('attn_mm_t', K.AttnMMPush(out=self.dqkv + 16 * C4, x=a['qkv'],
                                                 s=self.datt, xs=S4, os=S4, **mm),
                       B * cfg.kv_heads * TB * hs4, 1))
            stage(('rope', K.RopePush(x=self.dqkv, cs=self.rope, N=N, T=T, S4=S4, R4=C4 + KV4,
                                      hs4=hs4, inverse_=1), N * (C4 + KV4), 1))
            linear(self.dln, self.dqkv, a['ln1'], h + 'qkv', C, S)
            norm(h + 'n1', self.res[i], a['r1'], 1)
        stage(('embed_bwd', K.EmbedPush(x=self.dres, wte=m.g('wte'),
                                        toks=self.inp.device_address,
                                        order=self.order.device_address,
                                        starts=self.starts.device_address,
                                        ids=self.ids.device_address, N=N, C=C), N * C4, 1))
        stages.append(((), None))

        pending = None
        for ops, reduction in stages:
            ops = list(ops)
            if pending:
                partials, out, M, k = pending
                ops.append(('reduce_partials4', GK.Reduce4Push(out=out, part=partials, M4=M // 4,
                                                               K=k), M // 4, 1))
            for j, (kernel, push, n, gy) in enumerate(ops):
                self.dev.run(frame, kernel, push, n, gy, barrier=j == len(ops) - 1)
            pending = reduction

    def _setup_optimizer(self, cfg):
        if not self.muon:
            return
        bits = self.model.optim_bits
        regions = iter(self.dev.arena(optimizer_bytes(cfg, self.muon, bits),
                                      MemoryUsage.DEVICE_LOCAL))
        self.groups = []
        if bits == 32:
            mask = next(regions)
            self.dev.upload(mask, adam_mask(cfg).tobytes())
            self.adam_mask = mask.device_address
        for m, n, mats in muon_groups(cfg):
            x0, x1, t0, t1, idx, A, Bm, inv, Apart = (next(regions) for _ in range(9))
            self.dev.upload(idx, GM.muon_index(m, n, mats).tobytes())
            self.groups.append(dict(m=m, n=n, nb=len(mats),
                                    x=(x0.device_address, x1.device_address),
                                    xt=(t0.device_address, t1.device_address),
                                    idx=idx.device_address, A=A.device_address,
                                    Bm=Bm.device_address, Apart=Apart.device_address,
                                    inv=inv.device_address))
        if self.model.q8:
            for g, q8 in zip(self.groups, self.model.q8.groups):
                g['q8'] = q8

    def _update(self, frame):
        m, n = self.model, self.model.n_params
        run, (sums, mid) = self.dev.run, self.part
        normsq, hyper = self.stats.device_address, self.hyper.device_address
        run(frame, 'sumsq_partial', GK.NormPush(out=sums, grads=m.grads.device_address, n=n,
                                                stride=NORM_STRIDE), NORM_STRIDE)
        side = math.isqrt(NORM_STRIDE)
        run(frame, 'reduce_partials', GK.ReducePush(out=mid, part=sums, M=side, K=side), side)
        run(frame, 'reduce_partials', GK.ReducePush(out=normsq, part=mid, M=1, K=side), 1)
        if m.q8:
            self._update8(frame, hyper, normsq)
        else:
            self._update32(frame, hyper, normsq)
        m.refresh_fwd(frame)

    # -- execution ----------------------------------------------------------

    def _write_batch(self, regions, inp, tgt):
        regions[0].write(inp.tobytes())
        if tgt is not None:
            regions[1].write(tgt.tobytes())
        if self.mode == 'train':
            starts, order, ids = group_tokens(inp)
            regions[2].write(starts.tobytes())
            regions[3].write(order.tobytes())
            regions[4].write(ids.tobytes())

    def run(self, inp, tgt):
        """Eval: ``(sum of the per-token losses, scored tokens)`` of one batch."""
        assert self.mode == 'eval'
        self._write_batch(self.slots[0].regions, inp, tgt)
        self.rec.submit()
        total = sum(array.array('f', bytes(self.loss_sums.map())[:4 * LOSS_SUMS]))
        return total, sum(1 for t in tgt if t < self.cfg.vocab)

    def next_logits(self, tokens, pos):
        """Infer: ``tokens`` is B*T ids (array 'I'), ``pos`` the row to read in each sequence."""
        assert self.mode == 'infer'
        self._write_batch(self.slots[0].regions, tokens, None)
        self.pos.write(array.array('I', pos).tobytes())
        self.rec.submit()
        V = self.cfg.vocab
        flat = array.array('f', bytes(self.logits_host.map())[:4 * self.B * V])
        return [flat[b * V:(b + 1) * V] for b in range(len(pos))]

    def _finish(self, slot):
        slot.flight.wait()
        slot.flight = None
        sums = array.array('f', bytes(slot.regions[5].map())[:4 * LOSS_SUMS])
        slot.step[0] += sum(sums)
        if slot.last:
            stats = array.array('f', bytes(slot.regions[7].map())[:4])
            slot.step[1] = math.sqrt(max(stats[0], 0.0))

    def submit_step(self, batches, lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
                    clip=0.0, weights=None):
        """Queue one optimizer step of ``accum`` micro-batches ``(inp, tgt)`` (arrays 'I').

        Returns ``(mean loss over scored tokens, gradient norm, scored tokens)`` of
        the step queued before, or None. A ``weighted`` runner takes ``weights``, one
        array 'f' of B*T per micro-batch: the step minimizes sum w (-log p(target))
        instead of the mean (the reported loss stays the mean over scored tokens).
        """
        assert self.mode == 'train' and len(batches) == self.accum
        assert (weights is not None) == self.weighted
        V = self.cfg.vocab
        scored = sum(sum(1 for t in tgt if t < V) for _, tgt in batches)
        scale = array.array('f', [1.0 / max(scored, 1)]).tobytes()
        t = self.model.step + 1
        hyper = GK.Hyper(lr=lr, beta1=beta1, beta2=beta2, eps=eps, bc1=1.0 - beta1 ** t,
                         bc2=1.0 - beta2 ** t, wd=weight_decay, clip=clip)
        step = [0.0, 0.0, scored]
        for i, (inp, tgt) in enumerate(batches):
            slot = self.slots[self._turn]
            self._turn ^= 1
            if slot.flight:
                self._finish(slot)
            self._write_batch(slot.regions, inp, tgt)
            slot.regions[6].write_struct(hyper)
            slot.regions[8].write(weights[i].tobytes() if weights is not None else scale)
            slot.step, slot.last = step, i == self.accum - 1
            slot.flight = slot.recs[i == 0, slot.last]
            slot.flight.start()
        self.model.step = t
        previous, self._flying = self._flying, step
        if previous is None:
            return None
        for slot in self.slots:
            if slot.flight and slot.step is previous:
                self._finish(slot)
        return _result(previous)

    def collect(self):
        step, self._flying = self._flying, None
        for slot in self.slots:
            if slot.flight:
                self._finish(slot)
        return None if step is None else _result(step)

    def train_step(self, batches, lr, **kw):
        assert self._flying is None, 'collect() the queued step first'
        self.submit_step(batches, lr, **kw)
        return self.collect()


def _result(step):
    loss, norm, scored = step
    return loss / max(scored, 1), norm, scored

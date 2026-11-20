"""A byte-level GPT whose forward, backward and AdamW steps run on vabs compute.

``Model`` owns the parameters, gradients and Adam moments as flat float32
arrays in device-local memory, plus a forward copy of the parameters in which
every linear layer's weights are transposed (rebuilt after each update). With
``optim_bits=8`` the optimizer state is 8-bit blockwise instead (``State8``,
kernels from ``vabs_gpt_bitsandbytes``, a port of bitsandbytes).
``Runner`` owns the activations for one batch size and records reusable command
buffers:

- ``train`` mode records a whole optimizer step as one command buffer: zero the
  gradients, forward + loss + backward, then the update (global gradient norm,
  clipping, AdamW). With gradient accumulation there is one command buffer per
  micro-batch, the first zeroing and the last updating.
- ``eval`` records forward + loss, and ``infer`` records forward followed by a
  copy of the logits to host memory.

Per-step values (tokens, learning rate, Adam bias corrections) travel through
host-visible buffers, so the recordings never change.
"""
import array
import json
import math
import random
import struct
from dataclasses import asdict, dataclass

from vabs.memory import Arena, MemoryUsage
from vabs.pipeline import ComputePipeline
from vabs.renderer import one_shot_frame
from vabs.structs import uint32

from vabs_gpt_bitsandbytes import functional as QF
from vabs_gpt_bitsandbytes import kernels as Q

from . import kernels as K
from .gpu import Recording, compute_barrier

EOT = 0  # byte separating stories in the prepared data
ROWS_PER_CHUNK = 128  # rows summed per partial weight gradient
NORM_STRIDE = 4096   # partial sums for the global gradient norm (a square number)
# Kernels compiled once per set of sizes. Measured on an idle RTX 4090, constant
# sizes make the forward matmul and forward attention 20-25% faster and the
# forward layernorm 8%. The backward kernels were no faster (the matmul
# gradients 6% slower), so they stay generic.
assert Q.Hyper.std430_layout == K.Hyper.std430_layout   # the 8-bit kernels read the same struct
SPECIALIZED = frozenset({'matmul_fwd', 'attn_fwd', 'attn_softmax', 'attn_mm', 'layernorm_fwd'})
LOSS_SUMS = 64       # partial sums of the per-token losses read back by the host
MAX_GROUPS_Y = 65535


@dataclass
class Config:
    """Model shape. ``mlp`` = 0 means 4 * dim."""
    vocab: int = 256     # byte values
    ctx: int = 256       # context length T
    dim: int = 64        # model width C (embedding size)
    heads: int = 4
    layers: int = 4
    mlp: int = 0         # MLP hidden width
    init_std: float = 0.02

    def __post_init__(self):
        if self.mlp == 0:
            self.mlp = 4 * self.dim
        for name in ('vocab', 'ctx', 'dim', 'heads', 'layers', 'mlp'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.dim % self.heads:
            raise ValueError(f'dim ({self.dim}) must be divisible by heads ({self.heads})')
        # vec4 loads need the head size and both matmul input widths in multiples of 4;
        # a vocab multiple of 4 keeps every parameter tensor 16-byte aligned.
        if (self.dim // self.heads) % 4:
            raise ValueError(f'head size dim/heads = {self.dim // self.heads} must be a '
                             'multiple of 4')
        if self.mlp % 4 or self.vocab % 4:
            raise ValueError('mlp and vocab must be multiples of 4')
        if self.vocab > 256:
            raise ValueError('tokens are bytes, so vocab can be at most 256')


def param_layout(cfg):
    """Ordered ``(name, size)`` of every parameter tensor (linear: W[OC, C] then b[OC])."""
    V, T, C, F = cfg.vocab, cfg.ctx, cfg.dim, cfg.mlp
    out = [('wte', V * C), ('wpe', T * C)]
    for i in range(cfg.layers):
        out += [(f'h{i}.ln1', 2 * C), (f'h{i}.qkv', 3 * C * C + 3 * C),
                (f'h{i}.attn_proj', C * C + C), (f'h{i}.ln2', 2 * C),
                (f'h{i}.fc', F * C + F), (f'h{i}.fc_proj', C * F + C)]
    return out + [('lnf', 2 * C), ('head', V * C + V)]


def _weight_count(cfg, name, size):
    """Elements of a tensor that are weights (the rest are biases); 0 for layer norms."""
    kind = name.split('.')[-1]
    if kind.startswith('ln'):
        return 0
    out_dim = {'wte': 0, 'wpe': 0, 'qkv': 3 * cfg.dim, 'attn_proj': cfg.dim,
               'fc': cfg.mlp, 'fc_proj': cfg.dim, 'head': cfg.vocab}[kind]
    return size - out_dim


def init_params(cfg, seed=0):
    """GPT-2 style: N(0, init_std) weights, residual projections scaled by 1/sqrt(2L)."""
    rng = random.Random(seed)
    C = cfg.dim
    proj_std = cfg.init_std / math.sqrt(2 * cfg.layers)
    params = array.array('f')
    for name, size in param_layout(cfg):
        if name.split('.')[-1].startswith('ln'):
            params.extend([1.0] * C + [0.0] * C)
            continue
        n_weight = _weight_count(cfg, name, size)
        std = proj_std if name.endswith(('attn_proj', 'fc_proj')) else cfg.init_std
        params.extend(rng.gauss(0.0, std) for _ in range(n_weight))
        params.extend([0.0] * (size - n_weight))
    return params


def decay_mask(cfg):
    """1.0 where weight decay applies (matrices and embeddings), 0.0 for biases and norms."""
    mask = array.array('f')
    for name, size in param_layout(cfg):
        n_weight = _weight_count(cfg, name, size)
        mask.extend([1.0] * n_weight)
        mask.extend([0.0] * (size - n_weight))
    return mask


def transpose_index(cfg):
    """Source index of every element of the forward copy of the parameters.

    The copy stores each linear layer's weights transposed (W^T[C, OC], then the
    bias); everything else keeps its place.
    """
    index, o = array.array('I'), 0
    for name, size in param_layout(cfg):
        n_weight = _weight_count(cfg, name, size)
        if n_weight and name not in ('wte', 'wpe'):
            oc = size - n_weight
            c_in = n_weight // oc
            for c in range(c_in):
                index.extend(range(o + c, o + n_weight, c_in))
            index.extend(range(o + n_weight, o + size))
        else:
            index.extend(range(o, o + size))
        o += size
    return index


NS_COEFFS = (3.4445, -4.7750, 2.0315)   # quintic Newton-Schulz (Jordan et al. 2024)
MUON_KS = 16   # vec4s of the inner dimension per chunk of X X^T (split-K)


def muon_splits(n):
    """Chunks of the sum over n in X X^T: few workgroups each with a long loop are slow."""
    return _ceil(n // 4, MUON_KS)


@dataclass
class Muon:
    """Muon options. It updates the hidden weight matrices (q, k, v, attention output, MLP);
    the embeddings, head, biases and layer norms stay on AdamW with the same lr and decay."""
    momentum: float = 0.95
    ns_steps: int = 5
    scale: float = 0.2   # update RMS ~ AdamW's: O * scale * sqrt(max(rows, cols)) (Moonlight)


def muon_matrices(cfg):
    """``(offset, rows, cols)`` of each matrix Muon updates; q, k and v are separate matrices."""
    C, F = cfg.dim, cfg.mlp
    off, n = {}, 0
    for name, size in param_layout(cfg):
        off[name] = n
        n += size
    out = []
    for i in range(cfg.layers):
        q = off[f'h{i}.qkv']
        out += [(q, C, C), (q + C * C, C, C), (q + 2 * C * C, C, C),
                (off[f'h{i}.attn_proj'], C, C), (off[f'h{i}.fc'], F, C),
                (off[f'h{i}.fc_proj'], C, F)]
    return out


def muon_groups(cfg):
    """Batches of equally shaped matrices: ``[(m, n, [(offset, rows, cols), ...])]``, m <= n.

    A matrix with more rows than columns is orthogonalized transposed.
    """
    groups = {}
    for off, rows, cols in muon_matrices(cfg):
        groups.setdefault((min(rows, cols), max(rows, cols)), []).append((off, rows, cols))
    return [(m, n, mats) for (m, n), mats in groups.items()]


def muon_index(m, n, mats):
    """Parameter index of every workspace element of one batch (matrix-major, m x n each)."""
    index = array.array('I')
    for off, rows, cols in mats:
        if rows <= cols:
            index.extend(range(off, off + rows * cols))
        else:   # workspace row i is column i of W
            for i in range(cols):
                index.extend(range(off + i, off + rows * cols, cols))
    return index


def adam_mask(cfg):
    """1.0 for the parameters AdamW updates alongside Muon, 0.0 for Muon's matrices."""
    mask = array.array('f', [1.0]) * sum(size for _, size in param_layout(cfg))
    for off, rows, cols in muon_matrices(cfg):
        mask[off:off + rows * cols] = array.array('f', bytes(4 * rows * cols))
    return mask


MUON_KINDS = ('qkv', 'attn_proj', 'fc', 'fc_proj')   # whose weights Muon updates


def adam_segments(cfg, muon=False):
    """``[(start, size, decay)]`` of the tensors AdamW updates, for the 8-bit state.

    Weights, biases and the two halves of a layer norm are separate tensors, as in
    PyTorch, so each gets its own quantization blocks. With ``muon``, the matrices
    Muon updates are left out.
    """
    segs, o = [], 0
    for name, size in param_layout(cfg):
        kind = name.split('.')[-1]
        if kind.startswith('ln'):
            segs += [(o, cfg.dim, False), (o + cfg.dim, cfg.dim, False)]
        else:
            n_weight = _weight_count(cfg, name, size)
            if not (muon and kind in MUON_KINDS):
                segs.append((o, n_weight, True))
            if size > n_weight:
                segs.append((o + n_weight, size - n_weight, False))
        o += size
    return segs


class State8:
    """8-bit blockwise optimizer state in bitsandbytes' layout.

    AdamW's two moments over ``adam_segments`` (one byte each per element, an
    absmax per 256-element block) and, with ``muon``, Muon's momentum, one state
    per batch of equally shaped matrices in the order of its workspace. A
    checkpoint keeps every state's codes and absmax; the copies and partial
    maxima the kernels use within a step are scratch.
    """

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
        """Byte sizes of its regions: table, Adam codes, Adam scales, maps, then per batch."""
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
        """``(region, bytes from its start)`` of everything a checkpoint keeps."""
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


def model_bytes(cfg, optim_bits=32, muon=False):
    """Device bytes of a Model: parameters, gradients, forward copy and optimizer state."""
    n = sum(size for _, size in param_layout(cfg))
    if optim_bits == 32:
        return 7 * 4 * n
    return 4 * 4 * n + sum(State8.sizes(cfg, muon))


def _optimizer_sizes(cfg, batch, muon, qk_clip, optim_bits=32):
    """Byte sizes of the Muon workspaces and QK-Clip buffers a training Runner allocates."""
    sizes = []
    if muon and optim_bits == 32:
        sizes.append(4 * sum(size for _, size in param_layout(cfg)))   # the Adam mask
    if muon:
        for m, n, mats in muon_groups(cfg):
            nb = len(mats)
            sizes += [4 * nb * m * n] * 5 + [4 * nb * m * m] * 2 + [4 * nb,
                      4 * nb * m * m * max(1, muon_splits(n) - 1)]   # X X^T partials
    if qk_clip:
        L, H = cfg.layers, cfg.heads
        sizes += [4 * L * batch * H * cfg.ctx, 4 * L * H * batch, 4 * L * H]
    return sizes


def optimizer_bytes(cfg, batch, muon=None, qk_clip=0.0, optim_bits=32):
    return sum(_optimizer_sizes(cfg, batch, muon, qk_clip, optim_bits))


def param_counts(cfg):
    """Parameter counts by component: embeddings, all blocks together, final norm, head."""
    counts = {}
    for name, size in param_layout(cfg):
        key = 'blocks' if '.' in name else name
        counts[key] = counts.get(key, 0) + size
    return counts


def _ceil(a, b):
    return (a + b - 1) // b


def _runner_sizes(cfg, batch, mode):
    """Byte sizes of a Runner's device-local buffers, in allocation order."""
    B, T, C, V, H, F, L = batch, cfg.ctx, cfg.dim, cfg.vocab, cfg.heads, cfg.mlp, cfg.layers
    N, att = B * T, B * H * T * _ceil(T, 4) * 4
    per_layer = [N * C, N * C, 2 * N, 3 * N * C, att, N * C, N * C, N * C, 2 * N, N * F, N * F]
    sizes = [N * C] + per_layer * L + [N * C, 2 * N, N * V]
    if mode == 'train':
        k = _ceil(N, ROWS_PER_CHUNK)
        max_m = max(V * C + V, F * C + F, C * F + C, 3 * C * C + 3 * C, 2 * C)
        sizes += [N * C, N * C, N * C, N * F, 3 * N * C, att] + [max(k * max_m, NORM_STRIDE)] * 2
    return [4 * s for s in sizes]


def activation_bytes(cfg, batch, mode='train'):
    """Device memory a Runner of this batch size allocates (excluding parameters)."""
    return sum(_runner_sizes(cfg, batch, mode))


class Device:
    """Compute pipelines for every kernel, plus arena bookkeeping, on one context."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.pipes = {}   # (kernel, its sizes) -> pipeline

    def pipeline(self, kernel, push):
        """The pipeline of a kernel, compiled for this push struct's sizes if it gains from it.

        For the kernels in ``SPECIALIZED``, every ``uint32`` field of the push
        struct (row counts, widths, flags) is compiled in as a constant, which
        lets the driver unroll the loops over them. A model uses a handful of
        distinct combinations; each is compiled the first time it is recorded.
        """
        sizes = {}
        if kernel in SPECIALIZED:
            sizes = {name: getattr(push, name)
                     for name, (_, kind) in type(push).std430_layout.items() if kind is uint32}
        key = (kernel, *sizes.items())
        if key not in self.pipes:
            shader = getattr(K, kernel, None) or getattr(Q, kernel)   # Q: the 8-bit optimizers
            self.pipes[key] = ComputePipeline(self.ctx, shader.specialize(**sizes), name=kernel)
            self.ctx.on_destroy(self.pipes[key].destroy)
        return self.pipes[key]

    def arena(self, sizes, usage):
        """One arena holding a region per byte size in ``sizes`` (256-byte aligned)."""
        arena = Arena(self.ctx, sum(_ceil(s, 256) * 256 for s in sizes) + 256, usage=usage)
        self.ctx._arenas.append(arena)
        return [arena.alloc(s) for s in sizes]

    def _free(self, region):
        region.arena.destroy()
        self.ctx._arenas.remove(region.arena)

    def upload(self, region, data):
        staging, = self.arena([len(data)], MemoryUsage.HOST_VISIBLE)
        staging.write(data)
        with one_shot_frame(self.ctx) as frame:
            frame.copy_buffer(staging, region)
        self._free(staging)

    def download(self, region):
        staging, = self.arena([region.size], MemoryUsage.HOST_VISIBLE)
        with one_shot_frame(self.ctx) as frame:
            frame.copy_buffer(region, staging)
            compute_barrier(frame)
        data = bytes(staging.map())
        self._free(staging)
        return data

    def run(self, frame, kernel, push, n, gy=1, barrier=True):
        """Bind, push and dispatch enough workgroups for ``n`` invocations in x, then barrier.

        A 1-D dispatch wider than ``SPLIT_GROUPS`` wraps into y rows; kernels
        recover the element index with ``flat_index`` (or ``flat_row``).
        ``barrier=False`` lets the next dispatch run alongside this one; it must
        not touch anything this one writes.
        """
        pipe = self.pipeline(kernel, push)
        gx = _ceil(n, pipe.local_size[0])
        if gx > K.SPLIT_GROUPS:
            assert gy == 1, 'only 1-D dispatches can be split'
            gx, gy = K.SPLIT_GROUPS, _ceil(gx, K.SPLIT_GROUPS)
        if gy > MAX_GROUPS_Y:
            raise ValueError(f'{kernel}: dispatch of {gy} rows exceeds {MAX_GROUPS_Y}')
        frame.bind_compute_pipeline(pipe)
        frame.push_constants(push)
        frame.dispatch(gx, gy)
        if barrier:
            compute_barrier(frame)


class Model:
    """Parameters, gradients, optimizer state and the weight-decay mask in device memory.

    ``optim_bits`` 32 keeps float32 Adam moments ``m`` and ``v`` (Muon's momentum
    lives in ``m``). 8 keeps 8-bit blockwise states (``q8``, a ``State8``); their
    layout depends on whether Muon updates the hidden matrices, so ``muon`` must
    say so, and ``q8`` may hold a saved state.
    """

    def __init__(self, dev, cfg, params=None, seed=0, m=None, v=None, step=0, optim_bits=32,
                 muon=False, q8=None):
        if optim_bits not in (8, 32):
            raise ValueError('optim_bits must be 8 or 32')
        self.dev, self.cfg, self.step, self.optim_bits = dev, cfg, step, optim_bits
        self.offsets, n = {}, 0
        for name, size in param_layout(cfg):
            self.offsets[name] = n
            n += size
        self.n_params = n
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

    def refresh_fwd(self, frame):
        """Rebuild the forward copy (transposed weights) from the parameters."""
        self.dev.run(frame, 'gather', K.GatherPush(
            out=self.fwd.device_address, src=self.params.device_address,
            idx=self.fwd_index.device_address, n=self.n_params), self.n_params)

    def p(self, name):
        return self.params.device_address + 4 * self.offsets[name]

    def f(self, name):
        """Address of a linear layer in the forward copy: W^T[C, OC] then b[OC]."""
        return self.fwd.device_address + 4 * self.offsets[name]

    def g(self, name):
        return self.grads.device_address + 4 * self.offsets[name]

    def get_params(self):
        return array.array('f', self.dev.download(self.params))

    def get_grads(self):
        return array.array('f', self.dev.download(self.grads))

    def save(self, path, extra=None, optimizer=True):
        """Write config, step and parameters (plus the optimizer state, for resuming)."""
        header = {'config': asdict(self.cfg), 'step': self.step, 'optimizer': optimizer,
                  'optim_bits': self.optim_bits, **(extra or {})}
        if self.q8:
            header['optim_layout'] = 'muon' if self.q8.muon else 'adamw'
        data = json.dumps(header).encode()
        with open(path, 'wb') as f:
            f.write(b'VGPT' + struct.pack('<I', len(data)) + data)
            f.write(self.dev.download(self.params))
            if optimizer and self.q8:
                f.write(self.q8.save())
            elif optimizer:
                f.write(self.dev.download(self.m))
                f.write(self.dev.download(self.v))

    @staticmethod
    def read_header(path):
        with open(path, 'rb') as f:
            if f.read(4) != b'VGPT':
                raise ValueError(f'{path} is not a vabs_gpt checkpoint')
            return json.loads(f.read(struct.unpack('<I', f.read(4))[0]))

    @staticmethod
    def read(path):
        """``(header, params, state)`` from a checkpoint.

        ``state`` is None without optimizer state, else the Model arguments that
        restore it: ``{'m', 'v'}`` (32-bit) or ``{'optim_bits': 8, 'muon', 'q8'}``.
        """
        with open(path, 'rb') as f:
            if f.read(4) != b'VGPT':
                raise ValueError(f'{path} is not a vabs_gpt checkpoint')
            header = json.loads(f.read(struct.unpack('<I', f.read(4))[0]))
            body = f.read()
        cfg = Config(**header['config'])
        n = sum(size for _, size in param_layout(cfg))
        params, rest = array.array('f', body[:4 * n]), body[4 * n:]
        state = None
        if header.get('optimizer') and header.get('optim_bits', 32) == 8:
            muon = header.get('optim_layout') == 'muon'
            if len(rest) == State8.saved_bytes(cfg, muon):
                state = {'optim_bits': 8, 'muon': muon, 'q8': rest}
        elif header.get('optimizer') and len(rest) == 8 * n:
            moments = array.array('f', rest)
            state = {'m': moments[:n], 'v': moments[n:]}
        return header, params, state

    @staticmethod
    def load(dev, path):
        header, params, state = Model.read(path)
        model = Model(dev, Config(**header['config']), params, step=header.get('step', 0),
                      **(state or {}))
        return model, header


class _Slot:
    """The host-visible buffers of one submission and the recordings that use them."""

    def __init__(self, regions):
        self.regions = regions   # inp, tgt, losses, loss sums, hyper, stats (, logits)
        self.recs = {}           # (zero, update) -> Recording
        self.flight = None       # the recording now submitted and not yet waited for
        self.step = None         # [loss, norm] of the optimizer step it belongs to
        self.last = False        # it is that step's last micro-batch


class Runner:
    """Activations for batch size B and recorded command buffers.

    mode 'train': zero grads, forward + loss + backward and the update in one submission.
    ``train_step`` waits for it; ``submit_step`` queues it and returns the step before.
    'eval': forward + loss.  'infer': forward, then the logits are copied to host.
    ``accum`` micro-batches make one optimizer step (the loss gradient is scaled
    by 1 / (B*T*accum), so the step sees the mean over all of them).

    Training options: ``muon`` (a ``Muon``) updates the hidden matrices with Muon
    and the rest with AdamW; ``qk_clip`` > 0 is QK-Clip's tau: after each update,
    heads whose largest attention logit in the step exceeded it are scaled back
    (``muon`` plus ``qk_clip`` is MuonClip). With ``qk_clip``, ``max_logits``
    holds each head's largest logit, ``[layer][head]``, of the last finished step.
    """

    def __init__(self, model, batch, mode, accum=1, muon=None, qk_clip=0.0):
        if mode not in ('train', 'eval', 'infer'):
            raise ValueError(mode)
        if qk_clip < 0:
            raise ValueError('qk_clip must be non-negative')
        self.model, self.dev, self.mode = model, model.dev, mode
        self.muon = muon if mode == 'train' else None
        if mode == 'train' and model.q8 and model.q8.muon != bool(muon):
            raise ValueError('the 8-bit optimizer state was laid out for '
                             + ('Muon' if model.q8.muon else 'AdamW alone'))
        self.qk_clip = qk_clip if mode == 'train' else 0.0
        self.max_logits = None
        cfg = model.cfg
        B, T, C, V, H, L = batch, cfg.ctx, cfg.dim, cfg.vocab, cfg.heads, cfg.layers
        self.B, self.N, self.accum = B, B * T, accum
        N = self.N
        self.R = ROWS_PER_CHUNK
        # Head sizes up to 16 use the forward attention kernel that keeps a head's
        # output row in registers.
        self.fused = C // H <= 16
        self.K = _ceil(N, self.R)
        if B * H * T * T >= 2 ** 31 or N * max(V, cfg.mlp, 3 * C) >= 2 ** 31:
            raise ValueError('batch x context too large for 32-bit indices')
        if self.K > MAX_GROUPS_Y:
            raise ValueError(f'batch x context = {N} tokens exceeds '
                             f'{MAX_GROUPS_Y * self.R} per micro-batch')
        train = mode == 'train'
        regions = iter(self.dev.arena(_runner_sizes(cfg, B, mode), MemoryUsage.DEVICE_LOCAL))
        a = lambda: next(regions).device_address  # noqa: E731
        self.res = [a()]
        self.layer = []
        for _ in range(L):
            self.res.append(a())  # this layer's output residual
            names = ('ln1', 'ln1s', 'qkv', 'att', 'atty', 'res2', 'ln2', 'ln2s', 'fch', 'fcg')
            self.layer.append({name: a() for name in names})
        self.lnf, self.lnfs = a(), a()
        self.logits_region = next(regions)
        self.logits = self.logits_region.device_address
        if train:
            self.dres, self.dln, self.datty, self.dfc, self.dqkv, self.datt = (
                a() for _ in range(6))
            self.part = (a(), a())   # scratch for partial sums, used alternately
            self._setup_optimizer(cfg, B)
        # Host-visible inputs and outputs, one set per slot. Training has two slots
        # used in turn, so the next step can be written and queued while the GPU
        # still runs the current one. Per-token losses are zero-padded to whole
        # columns of LOSS_SUMS; the GPU adds up each column so the host reads
        # LOSS_SUMS numbers instead of N.
        loss_rows = _ceil(N, LOSS_SUMS)
        # stats: the squared gradient norm, then with QK-Clip each head's largest logit
        n_stats = 1 + (L * H if self.qk_clip else 0)
        host = [N, N, 4 * loss_rows * LOSS_SUMS, 4 * LOSS_SUMS, K.Hyper.std430_size,
                4 * n_stats] \
            + ([T * V * 4] if mode == 'infer' else [])
        # Recordings by (zero the gradients first, update afterwards). One optimizer
        # step is a single submission, or one per micro-batch when accumulating.
        variants = [(False, False)]
        if train:
            variants = [(True, True)] if accum == 1 else [(True, False), (False, False),
                                                          (False, True)]
        self.slots, self._turn, self._flying = [], 0, None
        for _ in range(2 if train else 1):
            slot = _Slot(self.dev.arena(host, MemoryUsage.HOST_VISIBLE))
            self.slots.append(slot)
            # the recording methods below read the current slot's buffers from self
            self.inp, self.tgt, self.losses, self.loss_sums, self.hyper, self.stats = \
                slot.regions[:6]
            self.losses.write(bytes(self.losses.size))
            self.logits_host = slot.regions[6] if mode == 'infer' else None
            for zero, update in variants:
                rec = slot.recs[zero, update] = Recording(self.dev.ctx)
                with rec.record() as frame:
                    compute_barrier(frame)   # after everything submitted before
                    if zero:
                        frame.fill_buffer(model.grads, 0)
                        compute_barrier(frame)
                    self._forward(frame)
                    if mode != 'infer':
                        self.dev.run(frame, 'cross_entropy', K.CEPush(
                            logits=self.logits, logits4=self.logits,
                            losses=self.losses.device_address,
                            targets=self.tgt.device_address, N=N, V=V, grad=int(train),
                            scale=1.0 / (N * accum)), N)
                        self.dev.run(frame, 'reduce_partials', K.ReducePush(
                            out=self.loss_sums.device_address,
                            part=self.losses.device_address,
                            M=LOSS_SUMS, K=loss_rows), LOSS_SUMS, barrier=not train)
                    if self.qk_clip:
                        self._max_logits(frame, first=zero, last=update)
                    if train:
                        self._backward(frame)
                    if update:
                        self._update(frame)
                    if mode == 'infer':
                        frame.copy_buffer(self.logits_region, self.logits_host, T * V * 4)
                        compute_barrier(frame)
        slot = self.slots[0]   # eval and infer use the one slot directly
        self.inp, self.tgt, self.loss_sums = slot.regions[0], slot.regions[1], slot.regions[3]
        self.rec = slot.recs[variants[0]]

    # -- recording ----------------------------------------------------------

    def _forward(self, frame):
        m, cfg, N = self.model, self.model.cfg, self.N

        run = self.dev.run
        B, T, C, V, H, F = self.B, cfg.ctx, cfg.dim, cfg.vocab, cfg.heads, cfg.mlp
        N4, TB = _ceil(N, 4), _ceil(T, 4)   # 4-row tiles of tokens / of one sequence
        heads = dict(B=B, T=T, H=H, hs4=C // H // 4)
        run(frame, 'embed_fwd', K.EmbedPush(x=self.res[0], wte=m.p('wte'), wpe=m.p('wpe'),
                                            toks=self.inp.device_address, N=N, T=T, C=C), N * C)
        for i, a in enumerate(self.layer):
            h = f'h{i}.'
            # with QK-Clip, the attention also stores each query row's largest logit
            smax = dict(smax=self.smax + 4 * i * B * H * T, keepmax=1) if self.qk_clip else {}
            run(frame, 'layernorm_fwd', K.LNPush(out=a['ln1'], inp=self.res[i], w=m.p(h + 'ln1'),
                                                 stats=a['ln1s'], N=N, C=C), N)
            run(frame, 'matmul_fwd', K.MMPush(out=a['qkv'], inp=a['ln1'], w=m.f(h + 'qkv'),
                                              N=N, C=C, OC=3 * C), N4 * 3 * C // 4)
            if self.fused:
                run(frame, 'attn_fwd', K.AttnPush(qkv=a['qkv'], att=a['att'], y=a['atty'], B=B,
                                                  T=T, C=C, H=H, keep=int(self.mode == 'train'),
                                                  **smax), B * H * T)
            else:
                run(frame, 'attn_softmax', K.AttnPush(qkv=a['qkv'], att=a['att'], B=B, T=T, C=C,
                                                      H=H, **smax), B * H * T)
                run(frame, 'attn_mm', K.AttnMMPush(out=a['atty'], x=a['qkv'] + 8 * C, s=a['att'],
                                                   xs=3 * C // 4, os=C // 4, **heads),
                    B * H * TB * C // H // 4)
            run(frame, 'matmul_fwd', K.MMPush(out=a['res2'], inp=a['atty'], w=m.f(h + 'attn_proj'),
                                              res=self.res[i], N=N, C=C, OC=C, flags=1), N4 * C // 4)
            run(frame, 'layernorm_fwd', K.LNPush(out=a['ln2'], inp=a['res2'], w=m.p(h + 'ln2'),
                                                 stats=a['ln2s'], N=N, C=C), N)
            run(frame, 'matmul_fwd', K.MMPush(out=a['fcg'], pre=a['fch'], inp=a['ln2'],
                                              w=m.f(h + 'fc'), N=N, C=C, OC=F, flags=2), N4 * F // 4)
            run(frame, 'matmul_fwd', K.MMPush(out=self.res[i + 1], inp=a['fcg'],
                                              w=m.f(h + 'fc_proj'), res=a['res2'],
                                              N=N, C=F, OC=C, flags=1), N4 * C // 4)
        run(frame, 'layernorm_fwd', K.LNPush(out=self.lnf, inp=self.res[-1], w=m.p('lnf'),
                                             stats=self.lnfs, N=N, C=C), N)
        run(frame, 'matmul_fwd', K.MMPush(out=self.logits, inp=self.lnf, w=m.f('head'),
                                          N=N, C=C, OC=V), N4 * V // 4)

    def _backward(self, frame):
        """Record the backward pass as stages of mutually independent dispatches.

        The dispatches of a stage have no barrier between them, so the GPU can run
        them side by side. Each stage's partial sums go to one of two scratch
        buffers in turn, and their reduction into the gradient joins the next stage.
        """
        m, cfg, N = self.model, self.model.cfg, self.N
        B, T, C, V, H, F = self.B, cfg.ctx, cfg.dim, cfg.vocab, cfg.heads, cfg.mlp
        N4, TB = _ceil(N, 4), _ceil(T, 4)
        heads = dict(B=B, T=T, H=H, hs4=C // H // 4, os=3 * C // 4)
        head_tiles = B * H * TB * C // H // 4
        stages = []   # [(dispatches, (partials, gradient, M) or None)]

        def part():
            return self.part[len(stages) % 2]

        def stage(*ops, grad=None):
            stages.append((ops, grad and (part(), m.g(grad[0]), grad[1])))

        def bwd_inp(out, dout, name, c_in, oc, **kw):
            return ('matmul_bwd_inp', K.MMPush(out=out, inp=dout, w=m.p(name), N=N, C=c_in, OC=oc,
                                               **kw), N4 * c_in // 4, 1)

        def linear(out, dout, inp, name, c_in, oc, **kw):
            """Input gradient and weight-gradient partial sums of one linear layer."""
            stage(bwd_inp(out, dout, name, c_in, oc, **kw),
                  ('matmul_bwd_w_partial', K.MMPush(out=part(), inp=inp, dout=dout, N=N, C=c_in,
                                                    OC=oc, R=self.R),
                   (oc // 4) * (c_in // 4) + oc // 4, self.K),
                  grad=(name, oc * c_in + oc))

        def layernorm(name, inp, stats, accumulate):
            """dres (+)= input gradient of a layer norm whose output gradient is dln."""
            ln = dict(inp=inp, w=m.p(name), stats=stats, dout=self.dln, N=N, C=C, R=self.R)
            stage(('layernorm_bwd', K.LNPush(out=self.dres, accumulate=accumulate, **ln), N, 1),
                  ('layernorm_bwd_partial', K.LNPush(part=part(), **ln), C // 4, self.K),
                  grad=(name, 2 * C))

        # logits now hold d(loss)/d(logits)
        linear(self.dln, self.logits, self.lnf, 'head', C, V)
        layernorm('lnf', self.res[-1], self.lnfs, 0)   # dres = d(final residual)
        for i in reversed(range(cfg.layers)):
            a, h = self.layer[i], f'h{i}.'
            linear(self.dfc, self.dres, a['fcg'], h + 'fc_proj', F, C, pre=a['fch'], flags=2)
            linear(self.dln, self.dfc, a['ln2'], h + 'fc', C, F)
            layernorm(h + 'ln2', a['res2'], a['ln2s'], 1)
            linear(self.datty, self.dres, a['atty'], h + 'attn_proj', C, C)
            # qkv rows are [q | k | v]: dv = att^T dout, then dq = dS k and dk = dS^T q
            stage(('attn_bwd_scores', K.AttnPush(
                       qkv=a['qkv'], att=a['att'], datt=self.datt, dout=self.datty, y=a['atty'],
                       B=B, T=T, C=C, H=H), B * H * T, 1),
                  ('attn_mm_t', K.AttnMMPush(out=self.dqkv + 8 * C, x=self.datty, s=a['att'],
                                             xs=C // 4, **heads), head_tiles, 1))
            stage(('attn_mm', K.AttnMMPush(out=self.dqkv, x=a['qkv'] + 4 * C, s=self.datt,
                                           xs=3 * C // 4, **heads), head_tiles, 1),
                  ('attn_mm_t', K.AttnMMPush(out=self.dqkv + 4 * C, x=a['qkv'], s=self.datt,
                                             xs=3 * C // 4, **heads), head_tiles, 1))
            linear(self.dln, self.dqkv, a['ln1'], h + 'qkv', C, 3 * C)
            layernorm(h + 'ln1', self.res[i], a['ln1s'], 1)
        stage(('embed_bwd_wte_partial', K.EmbedPush(
                   x4=self.dres, part=part(), toks=self.inp.device_address, N=N, T=T, C=C,
                   M=V * C, R=self.R), V * C // 4, self.K),
              ('embed_bwd_wpe', K.EmbedPush(x=self.dres, wpe=m.g('wpe'), N=N, T=T, C=C,
                                            accumulate=1), T * C, 1),
              grad=('wte', V * C))
        stages.append(((), None))   # the last reduction

        pending = None
        for ops, reduction in stages:
            ops = list(ops)
            if pending:
                partials, out, M = pending
                ops.append(('reduce_partials4', K.Reduce4Push(out=out, part=partials, M4=M // 4,
                                                              K=self.K), M // 4, 1))
            for k, (kernel, push, n, gy) in enumerate(ops):
                self.dev.run(frame, kernel, push, n, gy, barrier=k == len(ops) - 1)
            pending = reduction

    def _setup_optimizer(self, cfg, B):
        """Allocate and fill the Muon workspaces and the QK-Clip buffers this Runner needs."""
        if not (self.muon or self.qk_clip):
            return
        bits = self.model.optim_bits
        regions = iter(self.dev.arena(_optimizer_sizes(cfg, B, self.muon, self.qk_clip, bits),
                                      MemoryUsage.DEVICE_LOCAL))
        self.groups = []
        if self.muon and bits == 32:
            mask = next(regions)
            self.dev.upload(mask, adam_mask(cfg).tobytes())
            self.adam_mask = mask.device_address
        if self.muon:
            for m, n, mats in muon_groups(cfg):
                x0, x1, t0, t1, idx, A, Bm, inv, Apart = (next(regions) for _ in range(9))
                self.dev.upload(idx, muon_index(m, n, mats).tobytes())
                self.groups.append(dict(m=m, n=n, nb=len(mats), x=(x0.device_address,
                                        x1.device_address), xt=(t0.device_address,
                                        t1.device_address), idx=idx.device_address,
                                        A=A.device_address, Bm=Bm.device_address,
                                        Apart=Apart.device_address,
                                        inv=inv.device_address))
            if self.model.q8:
                for g, q8 in zip(self.groups, self.model.q8.groups):
                    g['q8'] = q8
        if self.qk_clip:
            self.smax, self.qk_part, self.qk_prev = (next(regions).device_address
                                                     for _ in range(3))

    def _max_logits(self, frame, first, last):
        """Each head's largest logit over the step's micro-batches so far.

        The last micro-batch writes them to the host-visible stats (after the norm),
        where ``qk_clip`` reads them; earlier ones keep a running maximum on the device.
        """
        cfg, run = self.model.cfg, self.dev.run
        L, H, B, T = cfg.layers, cfg.heads, self.B, cfg.ctx
        qk = dict(smax=self.smax, part=self.qk_part, prev=self.qk_prev, L=L, H=H, B=B, T=T)
        run(frame, 'qk_max_partial', K.QKPush(**qk), L * H * B)
        out = self.stats.device_address + 4 if last else self.qk_prev
        run(frame, 'qk_max', K.QKPush(out=out, accumulate=int(not first), **qk), L * H)

    def _update(self, frame):
        m, n = self.model, self.model.n_params
        run, (sums, mid) = self.dev.run, self.part
        normsq, hyper = self.stats.device_address, self.hyper.device_address
        run(frame, 'sumsq_partial', K.NormPush(out=sums, grads=m.grads.device_address, n=n,
                                               stride=NORM_STRIDE), NORM_STRIDE)
        side = math.isqrt(NORM_STRIDE)   # add them up in two levels
        run(frame, 'reduce_partials', K.ReducePush(out=mid, part=sums, M=side, K=side), side)
        run(frame, 'reduce_partials', K.ReducePush(out=normsq, part=mid, M=1, K=side), 1)
        if m.q8:
            self._update8(frame, hyper, normsq)
        else:
            self._update32(frame, hyper, normsq)
        if self.qk_clip:
            cfg = m.cfg
            off = m.offsets['h0.qkv']
            stride = (m.offsets['h1.qkv'] - off) if cfg.layers > 1 else 0
            per = 2 * cfg.dim * (cfg.dim + 1)
            run(frame, 'qk_clip', K.QKClipPush(
                params=m.params.device_address, smax=self.stats.device_address + 4,
                tau=self.qk_clip, off=off, stride=stride, L=cfg.layers, C=cfg.dim,
                H=cfg.heads), cfg.layers * per)
        m.refresh_fwd(frame)

    def _update32(self, frame, hyper, normsq):
        m, n, run = self.model, self.model.n_params, self.dev.run
        adam = dict(params=m.params.device_address, grads=m.grads.device_address,
                    m=m.m.device_address, v=m.v.device_address, hyper=hyper,
                    decay=m.decay.device_address, normsq=normsq, n=n)
        if not self.muon:
            run(frame, 'adam', K.AdamPush(**adam), n)
            return
        # AdamW on everything else runs alongside the momentum gather
        run(frame, 'adam', K.AdamPush(mask=self.adam_mask, masked=1, **adam), n, barrier=False)
        self._each(frame, 'muon_momentum', lambda g: K.MuonPush(
            grads=m.grads.device_address, m=m.m.device_address, x=g['x'][0], xt=g['xt'][0],
            idx=g['idx'], hyper=hyper, normsq=normsq, mu=self.muon.momentum,
            total=g['nb'] * g['m'] * g['n'], rows=g['m'], cols=g['n']),
            lambda g: g['nb'] * g['m'] * g['n'])
        self._newton_schulz(frame)

    def _update8(self, frame, hyper, normsq):
        """8-bit blockwise AdamW, and Muon's momentum beside it: two passes each.

        The first pass finds every block's new absmax, the second quantizes with it
        (see ``vabs_gpt_bitsandbytes.kernels``). With Muon the second pass writes the
        Nesterov direction into the Newton-Schulz workspace.
        """
        m, run, q = self.model, self.dev.run, self.model.q8
        adam = Q.Adam8Push(params=m.params.device_address, grads=m.grads.device_address,
                           table=q.table, state=q.state.device_address,
                           scales=q.scales.device_address, code=q.code, hyper=hyper,
                           normsq=normsq, blocks=q.blocks)
        for kernel in ('partial', 'apply'):
            run(frame, 'adam8_' + kernel, adam, QF.LANES * q.blocks, barrier=not self.muon)
            if self.muon:
                self._each(frame, 'momentum8_' + kernel, lambda g: Q.Momentum8Push(
                    grads=m.grads.device_address, idx=g['idx'],
                    state=g['q8']['state'].device_address,
                    scales=g['q8']['scales'].device_address, code=q.code, hyper=hyper,
                    normsq=normsq, out=g['x'][0], out_t=g['xt'][0], mu=self.muon.momentum,
                    nesterov=1, transpose=1, rows=g['m'], cols=g['n'],
                    total=g['nb'] * g['m'] * g['n'], blocks=g['q8']['blocks']),
                    lambda g: QF.LANES * g['q8']['blocks'])
        if self.muon:
            self._newton_schulz(frame)

    def _each(self, frame, kernel, make, n_of, gy=lambda g: 1, only=None):
        """One dispatch per batch of Muon matrices, with a barrier after the last."""
        only = self.groups if only is None else only
        for k, g in enumerate(only):
            self.dev.run(frame, kernel, make(g), n_of(g), gy(g), barrier=k == len(only) - 1)

    def _newton_schulz(self, frame):
        """Newton-Schulz on every batch of matrices side by side, then the Muon update."""
        m, mu, groups = self.model, self.muon, self.groups
        hyper = self.hyper.device_address

        def each(kernel, make, n_of, gy=lambda g: 1, only=groups):
            self._each(frame, kernel, make, n_of, gy, only)

        a, b, c = NS_COEFFS
        tiles = lambda rows, cols: lambda g: g['nb'] * g[rows] * g[cols] // 16  # noqa: E731
        split = [g for g in groups if muon_splits(g['n']) > 1]
        for s in range(mu.ns_steps):
            x, y = s % 2, (s + 1) % 2   # X is read from x[x] and written to x[y]
            each('bmm_nn', lambda g: K.BMMPush(
                out=g['A'], a=g['x'][x], b=g['xt'][x], part=g['Apart'], M=g['m'], N=g['m'],
                K=g['n'], nb=g['nb'], KS=MUON_KS, alpha=1.0), tiles('m', 'm'),
                lambda g: muon_splits(g['n']))
            if split:
                each('reduce_partials4', lambda g: K.Reduce4Push(
                    out=g['A'], part=g['Apart'], M4=g['nb'] * g['m'] * g['m'] // 4,
                    K=muon_splits(g['n']) - 1), lambda g: g['nb'] * g['m'] * g['m'] // 4,
                    only=split)
            if s == 0:
                # The first A is of the unnormalized X; its trace is |X|^2. With s = 1/|X|:
                # B = c s^4 A A + b s^2 A, and the new X = s (B X + a X).
                each('muon_norm', lambda g: K.MuonNormPush(a=g['A'], inv=g['inv'], m=g['m'],
                                                           nb=g['nb']), lambda g: g['nb'])
            first = s == 0
            each('bmm_nn', lambda g: K.BMMPush(
                out=g['Bm'], a=g['A'], b=g['A'], d=g['A'], sc=g['inv'], M=g['m'], N=g['m'],
                K=g['m'], nb=g['nb'], KS=g['m'] // 4, pab=4 * first, pd=2 * first, alpha=c,
                beta=b), tiles('m', 'm'))
            each('bmm_nn', lambda g: K.BMMPush(
                out=g['x'][y], outT=g['xt'][y], tout=int(s < mu.ns_steps - 1), a=g['Bm'],
                b=g['x'][x], d=g['x'][x], sc=g['inv'], M=g['m'], N=g['n'], K=g['m'],
                nb=g['nb'], KS=g['m'] // 4, pab=first, pd=first, alpha=1.0, beta=a),
                tiles('m', 'n'))
        out = mu.ns_steps % 2
        each('muon_apply', lambda g: K.MuonApplyPush(
            params=m.params.device_address, x=g['x'][out], idx=g['idx'], hyper=hyper,
            scale=mu.scale * math.sqrt(g['n']), n=g['nb'] * g['m'] * g['n']),
            lambda g: g['nb'] * g['m'] * g['n'])

    # -- execution ----------------------------------------------------------

    def run(self, inp, tgt=b''):
        """Forward (eval/infer) on B*T token bytes; return the mean loss (eval) or None."""
        self.inp.write(inp)
        if tgt:
            self.tgt.write(tgt)
        self.rec.submit()
        if self.mode == 'infer':
            return None
        return sum(array.array('f', bytes(self.loss_sums.map())[:4 * LOSS_SUMS])) / self.N

    def _finish(self, slot):
        """Wait for a slot's submission and add its results to its step's ``[loss, norm]``."""
        slot.flight.wait()
        slot.flight = None
        sums = array.array('f', bytes(slot.regions[3].map())[:4 * LOSS_SUMS])
        slot.step[0] += sum(sums) / (self.N * self.accum)
        if slot.last:
            stats = array.array('f', bytes(slot.regions[5].map()))
            slot.step[1] = math.sqrt(max(stats[0], 0.0))
            if self.qk_clip:
                H = self.model.cfg.heads
                self.max_logits = [list(stats[1 + l * H:1 + (l + 1) * H])
                                   for l in range(self.model.cfg.layers)]

    def submit_step(self, batches, lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
                    clip=0.0):
        """Queue one optimizer step without waiting for it.

        Takes ``accum`` micro-batches of ``(inp, tgt)`` byte strings. Returns
        ``(mean loss, gradient norm before clipping)`` of the step queued before
        this one, after waiting for it, or None if there was none. The GPU thus
        always has the next step queued while the host handles the last results
        and prepares another batch. ``collect()`` waits for the newest step.
        """
        assert self.mode == 'train' and len(batches) == self.accum
        t = self.model.step + 1
        hyper = K.Hyper(lr=lr, beta1=beta1, beta2=beta2, eps=eps, bc1=1.0 - beta1 ** t,
                        bc2=1.0 - beta2 ** t, wd=weight_decay, clip=clip)
        step = [0.0, 0.0]
        for i, (inp, tgt) in enumerate(batches):
            slot = self.slots[self._turn]
            self._turn ^= 1
            if slot.flight:
                self._finish(slot)
            slot.regions[0].write(inp)
            slot.regions[1].write(tgt)
            slot.regions[4].write_struct(hyper)
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
        return tuple(previous)

    def collect(self):
        """Wait for the step queued last; return its ``(loss, norm)``, or None if none is."""
        step, self._flying = self._flying, None
        for slot in self.slots:
            if slot.flight:
                self._finish(slot)
        return None if step is None else tuple(step)

    def train_step(self, batches, lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
                   clip=0.0):
        """One optimizer step, waited for: ``submit_step`` then ``collect``.

        Returns ``(mean loss, gradient norm before clipping)``.
        """
        assert self._flying is None, 'collect() the queued step first'
        self.submit_step(batches, lr, beta1, beta2, eps, weight_decay, clip)
        return self.collect()

    def logits_at(self, pos):
        V = self.model.cfg.vocab
        row = array.array('f', bytes(self.logits_host.map()))
        return row[pos * V:(pos + 1) * V]


def sample(runner, prompt, max_new=400, temperature=0.8, top_k=40, rng=None):
    """Generate bytes after ``prompt`` until an end-of-story byte or ``max_new`` bytes."""
    rng = rng or random.Random()
    T = runner.model.cfg.ctx
    tokens = list(prompt)
    out = bytearray()
    for _ in range(max_new):
        window = tokens[-T:]
        runner.run(bytes(window) + bytes(T - len(window)))
        logits = runner.logits_at(len(window) - 1)
        k = len(logits) if top_k <= 0 else min(top_k, len(logits))
        top = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:k]
        if temperature <= 0:
            nxt = top[0]
        else:
            mx = logits[top[0]]
            nxt = rng.choices(top, [math.exp((logits[i] - mx) / temperature) for i in top])[0]
        if nxt == EOT:
            break
        tokens.append(nxt)
        out.append(nxt)
    return out.decode('utf-8', errors='replace')

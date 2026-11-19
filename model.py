"""A byte-level GPT whose forward, backward and AdamW steps run on vabs compute.

``Model`` owns the parameters, gradients and Adam moments as flat float32
arrays in device-local memory. ``Runner`` owns the activations for one batch
size and records reusable command buffers:

- ``train`` mode records three: zero the gradients; forward + loss + backward
  (gradients accumulate, so several micro-batches can make up one optimizer
  step); and the update (global gradient norm, clipping, AdamW).
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

from . import kernels as K
from .gpu import Recording, compute_barrier

EOT = 0  # byte separating stories in the prepared data
ROWS_PER_CHUNK = 64  # rows summed per partial weight gradient
NORM_STRIDE = 4096   # partial sums for the global gradient norm
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
    N, att = B * T, B * H * T * T
    per_layer = [N * C, N * C, 2 * N, 3 * N * C, att, N * C, N * C, N * C, 2 * N, N * F, N * F]
    sizes = [N * C] + per_layer * L + [N * C, 2 * N, N * V]
    if mode == 'train':
        k = _ceil(N, ROWS_PER_CHUNK)
        max_m = max(V * C + V, F * C + F, C * F + C, 3 * C * C + 3 * C, 2 * C)
        sizes += [N * C, N * C, N * C, N * F, 3 * N * C, att, max(k * max_m, NORM_STRIDE)]
    return [4 * s for s in sizes]


def activation_bytes(cfg, batch, mode='train'):
    """Device memory a Runner of this batch size allocates (excluding parameters)."""
    return sum(_runner_sizes(cfg, batch, mode))


class Device:
    """Compute pipelines for every kernel, plus arena bookkeeping, on one context."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.pipes = {}
        for name in dir(K):
            shader = getattr(K, name)
            if hasattr(shader, 'spirv') and getattr(shader, 'stage', None) == 'compute':
                self.pipes[name] = ComputePipeline(ctx, shader, name=name)
                ctx.on_destroy(self.pipes[name].destroy)

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

    def run(self, frame, kernel, push, gx, gy=1):
        """Bind, push and dispatch ``gx`` x ``gy`` workgroups, then barrier.

        A 1-D dispatch wider than ``SPLIT_GROUPS`` wraps into y rows; kernels
        recover the element index with ``flat_index``.
        """
        if gx > K.SPLIT_GROUPS:
            assert gy == 1, 'only 1-D dispatches can be split'
            gx, gy = K.SPLIT_GROUPS, _ceil(gx, K.SPLIT_GROUPS)
        if gy > MAX_GROUPS_Y:
            raise ValueError(f'{kernel}: dispatch of {gy} rows exceeds {MAX_GROUPS_Y}')
        frame.bind_compute_pipeline(self.pipes[kernel])
        frame.push_constants(push)
        frame.dispatch(gx, gy)
        compute_barrier(frame)


class Model:
    """Parameters, gradients, Adam moments and the weight-decay mask in device memory."""

    def __init__(self, dev, cfg, params=None, seed=0, m=None, v=None, step=0):
        self.dev, self.cfg, self.step = dev, cfg, step
        self.offsets, n = {}, 0
        for name, size in param_layout(cfg):
            self.offsets[name] = n
            n += size
        self.n_params = n
        if params is None:
            params = init_params(cfg, seed)
        if len(params) != n:
            raise ValueError(f'expected {n} parameters, got {len(params)}')
        self.params, self.grads, self.m, self.v, self.decay = dev.arena(
            [4 * n] * 5, MemoryUsage.DEVICE_LOCAL)
        dev.upload(self.params, params.tobytes())
        dev.upload(self.m, m.tobytes() if m is not None else bytes(4 * n))
        dev.upload(self.v, v.tobytes() if v is not None else bytes(4 * n))
        dev.upload(self.decay, decay_mask(cfg).tobytes())

    def p(self, name):
        return self.params.device_address + 4 * self.offsets[name]

    def g(self, name):
        return self.grads.device_address + 4 * self.offsets[name]

    def get_params(self):
        return array.array('f', self.dev.download(self.params))

    def get_grads(self):
        return array.array('f', self.dev.download(self.grads))

    def save(self, path, extra=None, optimizer=True):
        """Write config, step and parameters (plus Adam moments, for resuming)."""
        header = {'config': asdict(self.cfg), 'step': self.step, 'optimizer': optimizer,
                  **(extra or {})}
        data = json.dumps(header).encode()
        with open(path, 'wb') as f:
            f.write(b'VGPT' + struct.pack('<I', len(data)) + data)
            f.write(self.dev.download(self.params))
            if optimizer:
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
        """``(header, params, m, v)`` from a checkpoint; m and v are None without optimizer state."""
        with open(path, 'rb') as f:
            if f.read(4) != b'VGPT':
                raise ValueError(f'{path} is not a vabs_gpt checkpoint')
            header = json.loads(f.read(struct.unpack('<I', f.read(4))[0]))
            body = array.array('f', f.read())
        n = sum(size for _, size in param_layout(Config(**header['config'])))
        if header.get('optimizer') and len(body) == 3 * n:
            return header, body[:n], body[n:2 * n], body[2 * n:]
        return header, body[:n], None, None

    @staticmethod
    def load(dev, path):
        header, params, m, v = Model.read(path)
        model = Model(dev, Config(**header['config']), params, m=m, v=v,
                      step=header.get('step', 0))
        return model, header


class Runner:
    """Activations for batch size B and recorded command buffers.

    mode 'train': zero grads / forward + loss + backward / update.
    'eval': forward + loss.  'infer': forward, then the logits are copied to host.
    ``accum`` micro-batches make one optimizer step (the loss gradient is scaled
    by 1 / (B*T*accum), so the step sees the mean over all of them).
    """

    def __init__(self, model, batch, mode, accum=1):
        if mode not in ('train', 'eval', 'infer'):
            raise ValueError(mode)
        self.model, self.dev, self.mode = model, model.dev, mode
        cfg = model.cfg
        B, T, C, V, H, L = batch, cfg.ctx, cfg.dim, cfg.vocab, cfg.heads, cfg.layers
        self.B, self.N, self.accum = B, B * T, accum
        N = self.N
        self.R = ROWS_PER_CHUNK
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
            self.dres, self.dln, self.datty, self.dfc, self.dqkv, self.datt, self.part = (
                a() for _ in range(7))
        host = [N, N, 4 * N, K.Hyper.std430_size, 4] + ([T * V * 4] if mode == 'infer' else [])
        host_regions = self.dev.arena(host, MemoryUsage.HOST_VISIBLE)
        self.inp, self.tgt, self.losses, self.hyper, self.stats = host_regions[:5]
        self.logits_host = host_regions[5] if mode == 'infer' else None

        self.rec = Recording(self.dev.ctx)
        with self.rec.record() as frame:
            self._forward(frame)
            if mode != 'infer':
                self.dev.run(frame, 'cross_entropy', K.CEPush(
                    logits=self.logits, losses=self.losses.device_address,
                    targets=self.tgt.device_address, N=N, V=V, grad=int(train),
                    scale=1.0 / (N * accum)), _ceil(N, K.WG))
            if train:
                self._backward(frame)
            if mode == 'infer':
                frame.copy_buffer(self.logits_region, self.logits_host, T * V * 4)
                compute_barrier(frame)
        if train:
            self.rec_zero = Recording(self.dev.ctx)
            with self.rec_zero.record() as frame:
                frame.fill_buffer(model.grads, 0)
                compute_barrier(frame)
            self.rec_update = Recording(self.dev.ctx)
            with self.rec_update.record() as frame:
                self._update(frame)

    # -- recording ----------------------------------------------------------

    def _forward(self, frame):
        m, cfg, N = self.model, self.model.cfg, self.N

        def run(frame, kernel, push, gx):
            if kernel == 'matmul_fwd':  # vec4 views of the input and weights
                push.inp4, push.w4 = push.inp, push.w
            self.dev.run(frame, kernel, push, gx)

        B, T, C, V, H, F = self.B, cfg.ctx, cfg.dim, cfg.vocab, cfg.heads, cfg.mlp
        g = lambda n: _ceil(n, K.WG)  # noqa: E731
        run(frame, 'embed_fwd', K.EmbedPush(x=self.res[0], wte=m.p('wte'), wpe=m.p('wpe'),
                                            toks=self.inp.device_address, N=N, T=T, C=C), g(N * C))
        for i, a in enumerate(self.layer):
            h = f'h{i}.'
            run(frame, 'layernorm_fwd', K.LNPush(out=a['ln1'], inp=self.res[i], w=m.p(h + 'ln1'),
                                                 stats=a['ln1s'], N=N, C=C), g(N))
            run(frame, 'matmul_fwd', K.MMPush(out=a['qkv'], inp=a['ln1'], w=m.p(h + 'qkv'),
                                              N=N, C=C, OC=3 * C), g(N * 3 * C))
            attn = dict(qkv=a['qkv'], qkv4=a['qkv'], att=a['att'], B=B, T=T, C=C, H=H)
            run(frame, 'attn_softmax', K.AttnPush(**attn), g(B * H * T))
            run(frame, 'attn_out', K.AttnPush(out=a['atty'], **attn), g(N * C))
            run(frame, 'matmul_fwd', K.MMPush(out=a['res2'], inp=a['atty'], w=m.p(h + 'attn_proj'),
                                              res=self.res[i], N=N, C=C, OC=C, flags=1), g(N * C))
            run(frame, 'layernorm_fwd', K.LNPush(out=a['ln2'], inp=a['res2'], w=m.p(h + 'ln2'),
                                                 stats=a['ln2s'], N=N, C=C), g(N))
            run(frame, 'matmul_fwd', K.MMPush(out=a['fcg'], pre=a['fch'], inp=a['ln2'],
                                              w=m.p(h + 'fc'), N=N, C=C, OC=F, flags=2), g(N * F))
            run(frame, 'matmul_fwd', K.MMPush(out=self.res[i + 1], inp=a['fcg'],
                                              w=m.p(h + 'fc_proj'), res=a['res2'],
                                              N=N, C=F, OC=C, flags=1), g(N * C))
        run(frame, 'layernorm_fwd', K.LNPush(out=self.lnf, inp=self.res[-1], w=m.p('lnf'),
                                             stats=self.lnfs, N=N, C=C), g(N))
        run(frame, 'matmul_fwd', K.MMPush(out=self.logits, inp=self.lnf, w=m.p('head'),
                                          N=N, C=C, OC=V), g(N * V))

    def _reduce(self, frame, out, M):
        self.dev.run(frame, 'reduce_partials', K.ReducePush(
            out=out, part=self.part, M=M, K=self.K, accumulate=1), _ceil(M, K.WG))

    def _weight_grad(self, frame, name, dout, inp, c_in, oc):
        M = oc * c_in + oc
        self.dev.run(frame, 'matmul_bwd_w_partial', K.MMPush(
            out=self.part, inp=inp, dout=dout, N=self.N, C=c_in, OC=oc, R=self.R),
            _ceil(M, K.WG), self.K)
        self._reduce(frame, self.model.g(name), M)

    def _ln_grad(self, frame, name, inp, stats, accumulate):
        m, C, N = self.model, self.model.cfg.dim, self.N
        ln = dict(inp=inp, w=m.p(name), stats=stats, dout=self.dln, N=N, C=C, R=self.R)
        self.dev.run(frame, 'layernorm_bwd', K.LNPush(out=self.dres, accumulate=accumulate, **ln),
                     _ceil(N, K.WG))
        self.dev.run(frame, 'layernorm_bwd_partial', K.LNPush(part=self.part, **ln),
                     _ceil(2 * C, K.WG), self.K)
        self._reduce(frame, m.g(name), 2 * C)

    def _backward(self, frame):
        m, cfg, run, N = self.model, self.model.cfg, self.dev.run, self.N
        B, T, C, V, H, F = self.B, cfg.ctx, cfg.dim, cfg.vocab, cfg.heads, cfg.mlp
        g = lambda n: _ceil(n, K.WG)  # noqa: E731
        # logits now hold d(loss)/d(logits)
        run(frame, 'matmul_bwd_inp', K.MMPush(out=self.dln, inp=self.logits, w=m.p('head'),
                                              N=N, C=C, OC=V), g(N * C))
        self._weight_grad(frame, 'head', self.logits, self.lnf, C, V)
        self._ln_grad(frame, 'lnf', self.res[-1], self.lnfs, 0)   # dres = d(final residual)
        for i in reversed(range(cfg.layers)):
            a, h = self.layer[i], f'h{i}.'
            run(frame, 'matmul_bwd_inp', K.MMPush(out=self.dfc, inp=self.dres,
                                                  w=m.p(h + 'fc_proj'), pre=a['fch'],
                                                  N=N, C=F, OC=C, flags=2), g(N * F))
            self._weight_grad(frame, h + 'fc_proj', self.dres, a['fcg'], F, C)
            run(frame, 'matmul_bwd_inp', K.MMPush(out=self.dln, inp=self.dfc, w=m.p(h + 'fc'),
                                                  N=N, C=C, OC=F), g(N * C))
            self._weight_grad(frame, h + 'fc', self.dfc, a['ln2'], C, F)
            self._ln_grad(frame, h + 'ln2', a['res2'], a['ln2s'], 1)
            run(frame, 'matmul_bwd_inp', K.MMPush(out=self.datty, inp=self.dres,
                                                  w=m.p(h + 'attn_proj'), N=N, C=C, OC=C), g(N * C))
            self._weight_grad(frame, h + 'attn_proj', self.dres, a['atty'], C, C)
            attn = dict(qkv=a['qkv'], qkv4=a['qkv'], att=a['att'], datt=self.datt,
                        dout=self.datty, dout4=self.datty, B=B, T=T, C=C, H=H)
            run(frame, 'attn_bwd_datt', K.AttnPush(**attn), g(B * H * T * T))
            run(frame, 'attn_bwd_softmax', K.AttnPush(**attn), g(B * H * T))
            run(frame, 'attn_bwd_qkv', K.AttnPush(out=self.dqkv, **attn), g(N * 3 * C))
            run(frame, 'matmul_bwd_inp', K.MMPush(out=self.dln, inp=self.dqkv, w=m.p(h + 'qkv'),
                                                  N=N, C=C, OC=3 * C), g(N * C))
            self._weight_grad(frame, h + 'qkv', self.dqkv, a['ln1'], C, 3 * C)
            self._ln_grad(frame, h + 'ln1', self.res[i], a['ln1s'], 1)
        run(frame, 'embed_bwd_wte_partial', K.EmbedPush(
            x=self.dres, wte=self.part, toks=self.inp.device_address, N=N, T=T, C=C,
            M=V * C, R=self.R), g(V * C), self.K)
        self._reduce(frame, m.g('wte'), V * C)
        run(frame, 'embed_bwd_wpe', K.EmbedPush(x=self.dres, wpe=m.g('wpe'), N=N, T=T, C=C,
                                                accumulate=1), g(T * C))

    def _update(self, frame):
        m, n = self.model, self.model.n_params
        self.dev.run(frame, 'sumsq_partial', K.NormPush(
            out=self.part, grads=m.grads.device_address, n=n, stride=NORM_STRIDE),
            _ceil(NORM_STRIDE, K.WG))
        self.dev.run(frame, 'reduce_partials', K.ReducePush(
            out=self.stats.device_address, part=self.part, M=1, K=NORM_STRIDE), 1)
        self.dev.run(frame, 'adam', K.AdamPush(
            params=m.params.device_address, grads=m.grads.device_address,
            m=m.m.device_address, v=m.v.device_address, hyper=self.hyper.device_address,
            decay=m.decay.device_address, normsq=self.stats.device_address, n=n),
            _ceil(n, K.WG))

    # -- execution ----------------------------------------------------------

    def _mean_loss(self):
        losses = array.array('f', bytes(self.losses.map()))
        return sum(losses) / len(losses)

    def run(self, inp, tgt=b''):
        """Forward (eval/infer) on B*T token bytes; return the mean loss (eval) or None."""
        self.inp.write(inp)
        if tgt:
            self.tgt.write(tgt)
        self.rec.submit()
        return None if self.mode == 'infer' else self._mean_loss()

    def train_step(self, batches, lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
                   clip=0.0):
        """One optimizer step over ``accum`` micro-batches of ``(inp, tgt)`` byte strings.

        Returns ``(mean loss, gradient norm before clipping)``.
        """
        assert self.mode == 'train' and len(batches) == self.accum
        self.rec_zero.submit()
        loss = 0.0
        for inp, tgt in batches:
            self.inp.write(inp)
            self.tgt.write(tgt)
            self.rec.submit()
            loss += self._mean_loss() / self.accum
        self.model.step += 1
        t = self.model.step
        self.hyper.write_struct(K.Hyper(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                                        bc1=1.0 - beta1 ** t, bc2=1.0 - beta2 ** t,
                                        wd=weight_decay, clip=clip))
        self.rec_update.submit()
        normsq = array.array('f', bytes(self.stats.map()))[0]
        return loss, math.sqrt(max(normsq, 0.0))

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

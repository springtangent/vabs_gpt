"""Check the Vulkan forward/backward/AdamW (32- and 8-bit) against a PyTorch reference.

Needs PyTorch (not used by the program itself). From the workspace root:

    $env:PYTHONPATH = "vabs\\src"; .ptvenv\\Scripts\\python.exe -m unittest vabs_gpt.test_gpu -v
"""
import array
import math
import random
import unittest

import torch
import torch.nn.functional as F

from vabs_gpt import kernels as K
from vabs_gpt.__main__ import parse_args
from vabs_gpt.gpu import ComputeContext
from vabs_gpt.model import (NS_COEFFS, Config, Device, Model, Muon, Runner, adam_mask,
                            adam_segments, decay_mask, muon_groups, muon_index, muon_matrices,
                            param_layout, sample)
from vabs_gpt_bitsandbytes import reference as R8


def param_spans(cfg):
    """name -> (offset, size) of every parameter tensor."""
    spans, o = {}, 0
    for name, size in param_layout(cfg):
        spans[name] = (o, size)
        o += size
    return spans


def param_offset(cfg, name):
    return param_spans(cfg)[name][0]


def torch_loss(cfg, flat, inp, tgt, B, maxes=None):
    """Mean cross-entropy; ``maxes`` (a list) receives each layer's per-head largest logit."""
    V, T, C, H, Fd = cfg.vocab, cfg.ctx, cfg.dim, cfg.heads, cfg.mlp
    p, o = {}, 0
    for name, size in param_layout(cfg):
        p[name] = flat[o:o + size]
        o += size

    def linear(x, name, c_in, c_out):
        w = p[name]
        return x @ w[:c_out * c_in].view(c_out, c_in).T + w[c_out * c_in:]

    def ln(x, name):
        return F.layer_norm(x, (C,), p[name][:C], p[name][C:], eps=1e-5)

    x = p['wte'].view(V, C)[inp] + p['wpe'].view(T, C)[None]
    for i in range(cfg.layers):
        h = f'h{i}.'
        q, k, v = linear(ln(x, h + 'ln1'), h + 'qkv', C, 3 * C).split(C, dim=2)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        if maxes is not None:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(C // H)
            causal = torch.ones(T, T, dtype=torch.bool).tril()
            maxes.append(s.masked_fill(~causal, -math.inf).amax(dim=(0, 2, 3)).detach())
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + linear(y.transpose(1, 2).reshape(B, T, C), h + 'attn_proj', C, C)
        f = F.gelu(linear(ln(x, h + 'ln2'), h + 'fc', C, Fd), approximate='tanh')
        x = x + linear(f, h + 'fc_proj', Fd, C)
    logits = linear(ln(x, 'lnf'), 'head', C, V)
    return F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))


def boosted_params(cfg, seed):
    """Initial parameters scaled up so every path carries signal."""
    from vabs_gpt.model import init_params
    return array.array('f', [x * 5 if abs(x) < 0.9 else x for x in init_params(cfg, seed)])


def random_batch(cfg, B, seed):
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(B * cfg.ctx + 1))
    return data[:-1], data[1:]


def adam8_state(model):
    """The AdamW part of a Model's 8-bit state, as ``vabs_gpt_bitsandbytes.reference`` takes it."""
    data, nb = model.q8.save(), model.q8.blocks
    codes = torch.tensor(list(data[:512 * nb]), dtype=torch.long)
    absmax = torch.tensor(array.array('f', data[512 * nb:520 * nb]))
    return {'codes1': codes[:256 * nb], 'codes2': codes[256 * nb:],
            'absmax1': absmax[:nb], 'absmax2': absmax[nb:]}


class TestAgainstTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = ComputeContext()
        cls.dev = Device(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.destroy()

    def reference(self, cfg, params, inp, tgt, B):
        flat = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        to_t = lambda b: torch.tensor(list(b)).view(B, cfg.ctx)  # noqa: E731
        loss = torch_loss(cfg, flat, to_t(inp), to_t(tgt), B)
        loss.backward()
        return loss.item(), flat.grad

    def assert_grads_close(self, cfg, grads, ref, tol=2e-3):
        grads = torch.tensor(grads, dtype=torch.float64)
        o = 0
        for name, size in param_layout(cfg):
            g, r = grads[o:o + size], ref[o:o + size]
            err = (g - r).abs().max().item() / max(r.abs().max().item(), 1e-8)
            self.assertLess(err, tol, f'{name}: relative grad error {err:.2e}')
            o += size

    def check_step(self, cfg, B, seed=1):
        params = boosted_params(cfg, seed)
        model = Model(self.dev, cfg, params)
        inp, tgt = random_batch(cfg, B, seed)
        lr = 1e-2
        loss, norm = Runner(model, B, 'train').train_step([(inp, tgt)], lr)
        ref_loss, ref_grad = self.reference(cfg, params, inp, tgt, B)
        self.assertAlmostEqual(loss, ref_loss, places=4)
        self.assertAlmostEqual(norm, ref_grad.norm().item(), delta=1e-3 * norm)
        grads = model.get_grads()
        self.assert_grads_close(cfg, grads, ref_grad)
        # First Adam step: m = (1-b1) g, v = (1-b2) g^2, so the update is lr * g / (|g| + eps).
        # Uses the GPU's float32 gradients: near-zero gradients flip with rounding.
        g = torch.tensor(grads, dtype=torch.float64)
        after = torch.tensor(model.get_params(), dtype=torch.float64)
        expected = torch.tensor(params, dtype=torch.float64) - lr * g / (g.abs() + 1e-8)
        err = (after - expected).abs()
        self.assertLess(err.max().item(), 1e-5, f'worst at {err.argmax().item()}')

    def test_default_shape(self):
        self.check_step(Config(ctx=32, dim=32, heads=4, layers=2), B=4)

    def test_odd_shape_and_short_chunk(self):
        # N = 10 * 30 = 300 rows: two chunks of 256 and 44, and neither N/4 nor T/4 is whole.
        # mlp is not 4 * dim; head size 12.
        self.check_step(Config(ctx=30, dim=24, heads=2, layers=2, mlp=40), B=10, seed=2)
        self.check_step(Config(ctx=13, dim=24, heads=2, layers=1, mlp=40), B=3, seed=6)

    def test_head_sizes(self):
        # 16 is the largest head the fused forward attention kernel takes; 20 uses the
        # separate softmax and output kernels.
        self.check_step(Config(ctx=18, dim=32, heads=2, layers=2), B=3, seed=8)
        self.check_step(Config(ctx=18, dim=40, heads=2, layers=2), B=3, seed=9)
        self.check_step(Config(ctx=18, dim=8, heads=2, layers=1), B=3, seed=10)

    def test_long_context(self):
        self.check_step(Config(ctx=2048, dim=8, heads=2, layers=1, mlp=16), B=2, seed=3)

    def test_wide_dispatch_is_split(self):
        # More than SPLIT_GROUPS workgroups: Device.run wraps into y and flat_index undoes it.
        from vabs.memory import MemoryUsage
        from vabs.renderer import one_shot_frame
        n = K.WG * K.SPLIT_GROUPS + 70000
        src, idx, out = self.dev.arena([4 * n] * 3, MemoryUsage.DEVICE_LOCAL)
        self.dev.upload(src, array.array('f', range(n)).tobytes())
        self.dev.upload(idx, array.array('I', range(n - 1, -1, -1)).tobytes())
        with one_shot_frame(self.ctx) as frame:
            self.dev.run(frame, 'gather', K.GatherPush(
                out=out.device_address, src=src.device_address, idx=idx.device_address, n=n), n)
        got = array.array('f', self.dev.download(out))
        self.assertEqual(got, array.array('f', range(n - 1, -1, -1)))
        self.dev._free(src)   # the three regions share one arena

    def test_gradient_accumulation_matches_full_batch(self):
        cfg = Config(ctx=32, dim=16, heads=2, layers=1)
        params = boosted_params(cfg, 4)
        inp, tgt = random_batch(cfg, 4, 4)
        half = 2 * cfg.ctx
        full = Model(self.dev, cfg, params)
        loss_full, _ = Runner(full, 4, 'train').train_step([(inp, tgt)], 1e-3)
        acc = Model(self.dev, cfg, params)
        loss_acc, _ = Runner(acc, 2, 'train', accum=2).train_step(
            [(inp[:half], tgt[:half]), (inp[half:], tgt[half:])], 1e-3)
        self.assertAlmostEqual(loss_full, loss_acc, places=5)
        a, b = full.get_grads(), acc.get_grads()
        self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 1e-5)
        pa, pb = full.get_params(), acc.get_params()
        self.assertLess(max(abs(x - y) for x, y in zip(pa, pb)), 1e-5)

    def test_queued_steps_match_waited_steps(self):
        # submit_step runs the same steps as train_step and reports each one a call later.
        cfg = Config(ctx=16, dim=16, heads=2, layers=1)
        for accum in (1, 3):   # 3 micro-batches share the two slots unevenly
            with self.subTest(accum=accum):
                params = boosted_params(cfg, 7)
                data = [[random_batch(cfg, 2, 20 + 10 * s + a) for a in range(accum)]
                        for s in range(5)]
                waited = Model(self.dev, cfg, params)
                runner = Runner(waited, 2, 'train', accum=accum)
                expected = [runner.train_step(b, 3e-3, clip=0.5) for b in data]
                queued = Model(self.dev, cfg, params)
                runner = Runner(queued, 2, 'train', accum=accum)
                self.assertIsNone(runner.collect())
                got = [runner.submit_step(b, 3e-3, clip=0.5) for b in data]
                self.assertIsNone(got[0])
                self.assertEqual(got[1:] + [runner.collect()], expected)
                self.assertIsNone(runner.collect())
                self.assertEqual(queued.step, 5)
                self.assertEqual(queued.get_params(), waited.get_params())
                # the waited form still works afterwards
                self.assertEqual(runner.train_step(data[0], 3e-3)[0],
                                 Runner(waited, 2, 'train', accum=accum).train_step(
                                     data[0], 3e-3)[0])

    def test_adamw_with_clipping_matches_reference(self):
        cfg = Config(ctx=16, dim=16, heads=2, layers=1)
        B, lr, wd, clip, steps = 4, 3e-3, 0.1, 0.5, 3
        b1, b2, eps = 0.9, 0.95, 1e-8
        params = boosted_params(cfg, 5)
        model = Model(self.dev, cfg, params)
        runner = Runner(model, B, 'train')
        batches = [random_batch(cfg, B, 10 + s) for s in range(steps)]
        norms = [runner.train_step([b], lr, b1, b2, eps, weight_decay=wd, clip=clip)[1]
                 for b in batches]
        gpu = torch.tensor(model.get_params(), dtype=torch.float64)

        # float64 AdamW with global-norm clipping and decay only where the mask is 1
        flat = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        start = flat.detach().clone()
        mask = torch.tensor(decay_mask(cfg), dtype=torch.float64)
        m, v = torch.zeros_like(start), torch.zeros_like(start)
        to_t = lambda b: torch.tensor(list(b)).view(B, cfg.ctx)  # noqa: E731
        for t, (inp, tgt) in enumerate(batches, 1):
            flat.grad = None
            torch_loss(cfg, flat, to_t(inp), to_t(tgt), B).backward()
            g = flat.grad
            norm = g.norm().item()
            self.assertAlmostEqual(norms[t - 1], norm, delta=1e-3 * norm)
            self.assertGreater(norm, clip)  # clipping is active
            g = g * min(1.0, clip / (norm + 1e-6))
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            with torch.no_grad():
                flat -= lr * ((m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
                              + wd * mask * flat)
        ref = flat.detach()
        rel = (gpu - ref).norm().item() / (ref - start).norm().item()
        self.assertLess(rel, 1e-2)

    def muon_reference(self, cfg, params, batches, B, lr, wd, clip, mu, tau=0.0, bits=32):
        """float64 Muon (Nesterov, Newton-Schulz, RMS matching) + AdamW, then QK-Clip.

        ``batches`` holds the micro-batches of each step. Returns the parameters and,
        with ``tau``, each step's per-head largest logits. ``bits`` 8 keeps the AdamW
        moments and the momentum as 8-bit blockwise states (bitsandbytes' algorithm).
        """
        segments = adam_segments(cfg, muon=True)
        adam8 = R8.new_adam_state(segments)
        groups = [(m_, n_, mats, muon_index(m_, n_, mats), R8.new_momentum_state(
            len(mats) * m_ * n_)) for m_, n_, mats in muon_groups(cfg)]
        b1, b2, eps = 0.9, 0.95, 1e-8
        a_, b_, c_ = NS_COEFFS
        flat = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor(decay_mask(cfg), dtype=torch.float64)
        adam = torch.tensor(adam_mask(cfg), dtype=torch.float64)
        m, v = torch.zeros_like(mask), torch.zeros_like(mask)
        to_t = lambda b: torch.tensor(list(b)).view(-1, cfg.ctx)  # noqa: E731
        seen = []
        for t, micro in enumerate(batches, 1):
            flat.grad = None
            maxes = []
            for inp, tgt in micro:
                (torch_loss(cfg, flat, to_t(inp), to_t(tgt), B, maxes) / len(micro)).backward()
            g = flat.grad.clone()
            if clip:
                g = g * min(1.0, clip / (g.norm().item() + 1e-6))
            with torch.no_grad():
                new = flat.detach().clone()
                directions = {}
                if bits == 8:
                    new, adam8 = R8.adam8_step(new, g, segments, adam8, lr, b1, b2, eps, t, wd,
                                               dtype=torch.float64)
                    for k, (m_, n_, mats, idx, st) in enumerate(groups):
                        d, st = R8.momentum8_step(g, idx, st, mu, dtype=torch.float64)
                        groups[k] = (m_, n_, mats, idx, st)
                        for i, (off, _, _) in enumerate(mats):   # workspace: m x n each
                            directions[off] = d[i * m_ * n_:(i + 1) * m_ * n_].view(m_, n_)
                else:
                    # AdamW where the mask is 1
                    m = torch.where(adam > 0, b1 * m + (1 - b1) * g, m)
                    v = torch.where(adam > 0, b2 * v + (1 - b2) * g * g, v)
                    step = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps) \
                        + wd * mask * new
                    new = torch.where(adam > 0, new - lr * step, new)
                for off, rows, cols in muon_matrices(cfg):
                    sl = slice(off, off + rows * cols)
                    if bits == 8:
                        X = directions[off]
                    else:
                        m[sl] = mu * m[sl] + g[sl]
                        X = (g[sl] + mu * m[sl]).view(rows, cols)
                        if rows > cols:
                            X = X.T
                    X = X / (X.norm() + 1e-7)
                    for _ in range(5):
                        A = X @ X.T
                        X = a_ * X + (b_ * A + c_ * A @ A) @ X
                    if rows > cols:
                        X = X.T
                    W = new[sl]
                    new[sl] = W - lr * (0.2 * math.sqrt(max(rows, cols)) * X.reshape(-1) + wd * W)
                if tau:
                    # every head's largest logit over the step's micro-batches
                    per_layer = [torch.stack(maxes[l::cfg.layers]).amax(0)
                                 for l in range(cfg.layers)]
                    seen.append(per_layer)
                    C, hs = cfg.dim, cfg.dim // cfg.heads
                    for l, mx in enumerate(per_layer):
                        off = param_offset(cfg, f'h{l}.qkv')
                        for h, s in enumerate(mx.tolist()):
                            if s > tau:
                                f = math.sqrt(tau / s)
                                for r0 in (h * hs, C + h * hs):   # q rows, then k rows
                                    new[off + r0 * C:off + (r0 + hs) * C] *= f
                                    new[off + 3 * C * C + r0:off + 3 * C * C + r0 + hs] *= f
                flat.copy_(new)
        return flat.detach(), seen

    def test_muon_matches_reference(self):
        # mlp 12 < dim: fc is orthogonalized as it is and fc_proj transposed, in one batch;
        # mlp 64 > dim: fc is transposed. Clipping and decay are on. With mlp 144, X X^T
        # sums over 36 vec4s in three chunks (split-K), the last one short.
        for cfg in (Config(ctx=16, dim=16, heads=2, layers=2, mlp=12),
                    Config(ctx=12, dim=16, heads=4, layers=1),
                    Config(ctx=8, dim=16, heads=2, layers=1, mlp=144)):
            with self.subTest(cfg=cfg):
                B, lr, wd, clip, mu = 3, 3e-3, 0.1, 0.5, 0.95
                params = boosted_params(cfg, 12)
                batches = [[random_batch(cfg, B, 30 + s)] for s in range(3)]
                model = Model(self.dev, cfg, params)
                runner = Runner(model, B, 'train', muon=Muon(momentum=mu))
                for micro in batches:
                    runner.train_step(micro, lr, weight_decay=wd, clip=clip)
                gpu = torch.tensor(model.get_params(), dtype=torch.float64)
                ref, _ = self.muon_reference(cfg, params, batches, B, lr, wd, clip, mu)
                start = torch.tensor(params, dtype=torch.float64)
                for name, (o, size) in param_spans(cfg).items():
                    d_ref = (ref - start)[o:o + size]
                    rel = (gpu[o:o + size] - ref[o:o + size]).norm().item() / d_ref.norm().item()
                    self.assertLess(rel, 1e-2, f'{name}: relative error {rel:.2e}')

    def test_muonclip_matches_reference(self):
        # Two micro-batches per step: the largest logit is over both. tau sits among the
        # heads' first-step maxima, so some heads are clipped and some are not.
        cfg = Config(ctx=16, dim=16, heads=4, layers=2)
        B, lr, mu = 2, 3e-3, 0.95
        params = boosted_params(cfg, 13)
        for l in range(cfg.layers):   # large q and k weights: logits well above 1
            o = param_offset(cfg, f'h{l}.qkv')
            for i in range(o, o + 2 * cfg.dim * cfg.dim):
                params[i] *= 4.0
        batches = [[random_batch(cfg, B, 40 + 2 * s), random_batch(cfg, B, 41 + 2 * s)]
                   for s in range(2)]

        def head_maxima(flat, micro):
            """[layer][head] largest logits of the micro-batches together, in float64."""
            to_t = lambda b: torch.tensor(list(b)).view(-1, cfg.ctx)  # noqa: E731
            maxes = []
            torch_loss(cfg, torch.tensor(flat, dtype=torch.float64),
                       to_t(b''.join(i for i, _ in micro)), to_t(b''.join(t for _, t in micro)),
                       len(micro) * B, maxes)
            return [x.tolist() for x in maxes]

        first = head_maxima(params, batches[0])
        values = sorted(x for layer in first for x in layer)
        tau = values[len(values) // 2]
        self.assertGreater(values[-1], 1.05 * tau)

        # With lr 0 only the clip changes the parameters: every head over tau ends at tau.
        model = Model(self.dev, cfg, params)
        Runner(model, B, 'train', accum=2, muon=Muon(), qk_clip=tau).train_step(batches[0], 0.0)
        for before, after in zip(first, head_maxima(model.get_params(), batches[0])):
            for s0, s1 in zip(before, after):
                self.assertAlmostEqual(s1, min(s0, tau), delta=1e-4 * s0)

        model = Model(self.dev, cfg, params)
        runner = Runner(model, B, 'train', accum=2, muon=Muon(momentum=mu), qk_clip=tau)
        got_max = []
        for micro in batches:
            runner.train_step(micro, lr)
            got_max.append(runner.max_logits)
        ref, seen = self.muon_reference(cfg, params, batches, B, lr, 0.0, 0.0, mu, tau)
        for got, want in zip(got_max, seen):
            for g, w in zip(got, want):
                for a, b in zip(g, w.tolist()):
                    self.assertAlmostEqual(a, b, delta=1e-3 * abs(b))
        gpu = torch.tensor(model.get_params(), dtype=torch.float64)
        start = torch.tensor(params, dtype=torch.float64)
        rel = (gpu - ref).norm().item() / (ref - start).norm().item()
        self.assertLess(rel, 1e-2)

    def assert_steps_close(self, cfg, params, gpu, ref, tol=1e-2):
        """Each tensor's change matches the reference's within ``tol`` (relative)."""
        start = torch.tensor(params, dtype=torch.float64)
        for name, (o, size) in param_spans(cfg).items():
            d_ref = (ref - start)[o:o + size]
            rel = (gpu[o:o + size] - ref[o:o + size]).norm().item() / d_ref.norm().item()
            self.assertLess(rel, tol, f'{name}: relative error {rel:.2e}')

    def test_adamw8_matches_reference(self):
        # 8-bit blockwise AdamW against bitsandbytes' algorithm, step by step from the
        # GPU's own gradients and state (the gradients are checked against torch above):
        # with 8-bit states a near-zero second moment can round to zero, so a float64
        # gradient that differs in the last bits changes the update a lot. Decay (weights
        # only) and clipping are on; the reference runs in float32 like the GPU.
        f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
        for cfg in (Config(ctx=16, dim=16, heads=2, layers=1),
                    Config(ctx=12, dim=32, heads=4, layers=2, mlp=36)):
            with self.subTest(cfg=cfg):
                B, lr, wd, clip, steps = 4, 3e-3, 0.1, 0.5, 4
                b1, b2, eps = 0.9, 0.95, 1e-8
                params = boosted_params(cfg, 6)
                model = Model(self.dev, cfg, params, optim_bits=8)
                runner = Runner(model, B, 'train')
                segments = adam_segments(cfg)
                state = R8.new_adam_state(segments)
                ref = torch.tensor(params)
                for t in range(1, steps + 1):
                    _, norm = runner.train_step([random_batch(cfg, B, 20 + t)], lr, b1, b2, eps,
                                                weight_decay=wd, clip=clip)
                    self.assertGreater(norm, clip)
                    g = torch.tensor(model.get_grads())
                    scale = torch.minimum(f32(1.0), f32(clip) / ((g * g).sum().sqrt() + 1e-6))
                    ref, state = R8.adam8_step(ref, g, segments, state, lr, b1, b2, eps, t, wd,
                                               scale)
                    gpu, gpu_state = torch.tensor(model.get_params()), adam8_state(model)
                    torch.testing.assert_close(gpu, ref, rtol=1e-5, atol=1e-6)
                    for key in ('absmax1', 'absmax2'):
                        torch.testing.assert_close(gpu_state[key], state[key], rtol=1e-5, atol=0)
                    for key in ('codes1', 'codes2'):
                        diff = (gpu_state[key] - state[key]).abs()
                        self.assertLessEqual(int(diff.max()), 1)
                        self.assertLessEqual(int((diff > 0).sum()), 2)
                    ref, state = gpu, gpu_state   # the next step starts from the GPU's

    def test_muonclip8_matches_reference(self):
        # Muon's momentum and the AdamW moments in 8 bits, two micro-batches per step,
        # clipping, decay and QK-Clip with a tau that clips some heads. fc is transposed
        # (mlp > dim) and the qkv batch spans several quantization blocks.
        cfg = Config(ctx=16, dim=16, heads=4, layers=2)
        B, lr, wd, clip, mu = 2, 3e-3, 0.1, 0.5, 0.95
        params = boosted_params(cfg, 14)
        for l in range(cfg.layers):   # large q and k weights: logits well above 1
            o = param_offset(cfg, f'h{l}.qkv')
            for i in range(o, o + 2 * cfg.dim * cfg.dim):
                params[i] *= 4.0
        batches = [[random_batch(cfg, B, 50 + 2 * s), random_batch(cfg, B, 51 + 2 * s)]
                   for s in range(3)]
        model = Model(self.dev, cfg, params, optim_bits=8, muon=True)
        runner = Runner(model, B, 'train', accum=2, muon=Muon(momentum=mu), qk_clip=1.0)
        runner.train_step(batches[0], 0.0)
        values = sorted(x for layer in runner.max_logits for x in layer)
        tau = values[len(values) // 2]   # between the heads' first maxima

        model = Model(self.dev, cfg, params, optim_bits=8, muon=True)
        runner = Runner(model, B, 'train', accum=2, muon=Muon(momentum=mu), qk_clip=tau)
        for micro in batches:
            runner.train_step(micro, lr, weight_decay=wd, clip=clip)
        gpu = torch.tensor(model.get_params(), dtype=torch.float64)
        ref, seen = self.muon_reference(cfg, params, batches, B, lr, wd, clip, mu, tau, bits=8)
        self.assertGreater(max(x for layer in seen[0] for x in layer.tolist()), 1.05 * tau)
        self.assert_steps_close(cfg, params, gpu, ref)

    def test_adam_segments_cover_parameters(self):
        # The 8-bit AdamW tensors keep the decay mask, and with Muon they and Muon's
        # matrices together hold every parameter exactly once.
        for cfg in (Config(ctx=16, dim=16, heads=2, layers=2, mlp=12), Config()):
            n = sum(size for _, size in param_layout(cfg))
            mask = decay_mask(cfg)
            for muon in (False, True):
                with self.subTest(cfg=cfg, muon=muon):
                    count = [0] * n
                    for start, size, decay in adam_segments(cfg, muon):
                        self.assertEqual(set(mask[start:start + size]), {float(decay)})
                        for i in range(start, start + size):
                            count[i] += 1
                    if muon:
                        for off, rows, cols in muon_matrices(cfg):
                            for i in range(off, off + rows * cols):
                                count[i] += 1
                    self.assertEqual(set(count), {1})

    def test_8bit_checkpoint_resumes_exactly(self):
        import os
        import tempfile
        cfg = Config(ctx=16, dim=16, heads=2, layers=2, mlp=12)
        B = 3
        params = boosted_params(cfg, 15)
        batches = [random_batch(cfg, B, 60 + s) for s in range(3)]
        for muon in (None, Muon()):
            with self.subTest(muon=muon), tempfile.TemporaryDirectory() as d:
                kw = dict(optim_bits=8, muon=muon is not None)
                straight = Model(self.dev, cfg, params, **kw)
                runner = Runner(straight, B, 'train', muon=muon, qk_clip=5.0 if muon else 0.0)
                for b in batches:
                    runner.train_step([b], 3e-3, weight_decay=0.1)
                part = Model(self.dev, cfg, params, **kw)
                runner = Runner(part, B, 'train', muon=muon, qk_clip=5.0 if muon else 0.0)
                for b in batches[:2]:
                    runner.train_step([b], 3e-3, weight_decay=0.1)
                path = os.path.join(d, 'ckpt.bin')
                part.save(path)
                header, _, state = Model.read(path)
                self.assertEqual((header['optim_bits'], state['optim_bits']), (8, 8))
                self.assertEqual(state['muon'], muon is not None)
                resumed, _ = Model.load(self.dev, path)
                self.assertEqual(resumed.step, 2)
                Runner(resumed, B, 'train', muon=muon,
                       qk_clip=5.0 if muon else 0.0).train_step([batches[2]], 3e-3,
                                                                 weight_decay=0.1)
                self.assertEqual(list(resumed.get_params()), list(straight.get_params()))
                with self.assertRaises(ValueError):   # the state layout fixes the optimizer
                    Runner(resumed, B, 'train', muon=None if muon else Muon())

    def test_eval_matches_training_loss(self):
        # eval skips storing the attention matrix; its loss must equal the training forward's
        for cfg in (Config(ctx=18, dim=32, heads=2, layers=2),    # fused attention forward
                    Config(ctx=18, dim=40, heads=2, layers=2)):   # separate kernels
            params = boosted_params(cfg, 11)
            inp, tgt = random_batch(cfg, 3, 11)
            model = Model(self.dev, cfg, params)
            loss = Runner(model, 3, 'eval').run(inp, tgt)
            self.assertAlmostEqual(loss, Runner(model, 3, 'train').train_step([(inp, tgt)], 0.0)[0],
                                   places=5)

    def test_eval_and_sampling_run(self):
        cfg = Config(ctx=32, dim=32, heads=4, layers=1)
        model = Model(self.dev, cfg, seed=5)
        ev = Runner(model, 2, 'eval')
        self.assertAlmostEqual(ev.run(bytes(64), bytes(64)), math.log(256), delta=0.1)
        text = sample(Runner(model, 1, 'infer'), b'Once', max_new=10, rng=random.Random(0))
        self.assertLessEqual(len(text.encode('utf-8', 'replace')), 40)


class TestOptions(unittest.TestCase):
    def test_config_file_and_overrides(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'run.json')
            with open(path, 'w') as f:
                json.dump({'dim': 96, 'layers': 3, 'heads': 6, 'lr': 1e-3, 'grad-accum': 2}, f)
            args = parse_args(['train', '--config', path, '--layers', '5'])
            self.assertEqual((args.dim, args.layers, args.heads, args.lr, args.grad_accum),
                             (96, 5, 6, 1e-3, 2))
            with open(path, 'w') as f:
                json.dump({'dimm': 96}, f)
            with self.assertRaises(SystemExit):
                parse_args(['train', '--config', path])

    def test_explicit_model_options_and_resume_defaults(self):
        import os
        import tempfile
        from vabs_gpt.__main__ import train
        # a flag equal to its default still counts as given
        self.assertEqual(parse_args(['train', '--dim', '64']).explicit_model, {'dim': 64})
        self.assertEqual(parse_args(['train']).explicit_model, {})
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, 'a.bin')
            # a checkpoint whose training options differ from the defaults
            cfg = Config(ctx=16, dim=16, heads=2, layers=1)
            ctx = ComputeContext()
            try:
                Model(Device(ctx), cfg).save(ckpt, {'train': {
                    'lr': 5e-4, 'min_lr': None, 'weight_decay': 0.2, 'batch': 3, 'steps': 7,
                    'out': ckpt}})
            finally:
                ctx.destroy()
            args = parse_args(['train', '--resume', ckpt, '--steps', '9'])
            self.assertEqual((args.lr, args.weight_decay, args.batch, args.steps, args.out),
                             (5e-4, 0.2, 3, 9, None))
            self.assertEqual(parse_args(['train', '--resume', ckpt, '--lr', '1e-3']).lr, 1e-3)
            with self.assertRaises(SystemExit):   # shape differs from the checkpoint
                train(parse_args(['train', '--resume', ckpt, '--dim', '64', '--steps', '9']))
            with self.assertRaises(SystemExit):   # existing file, not resumed, no --overwrite
                train(parse_args(['train', '--out', ckpt, '--steps', '9']))

    def test_epoch_order_visits_every_window_once(self):
        import os
        import tempfile
        from vabs_gpt.__main__ import Batches, epoch_windows, train
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'train.bin')
            with open(path, 'wb') as f:
                f.write(bytes(random.Random(0).randrange(256) for _ in range(20000)))
            T = 16
            n = epoch_windows(20000, T)
            b = Batches(path, 1, T, seed=3, order='epoch')
            for _ in range(2):   # the second epoch is a new order on a shifted grid
                offs = [b._offset() for _ in range(n)]
                self.assertEqual(len(set(offs)), n)
                self.assertTrue(all(o + T + 1 <= 20000 for o in offs))
            resumed = Batches(path, 1, T, seed=3, order='epoch', start=2 * n)
            self.assertEqual([resumed._offset() for _ in range(20)],
                             [b._offset() for _ in range(20)])
            b.close()
            resumed.close()
            with open(os.path.join(d, 'val.bin'), 'wb') as f:
                f.write(bytes(1000))
            args = parse_args(['train', '--data', d, '--epochs', '2', '--ctx', '16', '--batch',
                               '4', '--grad-accum', '2', '--dry-run'])
            train(args)
            self.assertEqual(args.steps, -(-2 * n // 8))

    def test_optimizer_options(self):
        import io
        from contextlib import redirect_stdout
        from vabs_gpt.__main__ import muon_options, train
        args = parse_args(['train', '--optimizer', 'muonclip', '--dry-run'])
        self.assertIsNone(args.qk_clip)   # saved as given; train() resolves it
        with redirect_stdout(io.StringIO()) as out:
            train(args)
        self.assertEqual(args.qk_clip, 100.0)
        self.assertIn('MuonClip', out.getvalue())
        args = parse_args(['train', '--optimizer', 'muon', '--muon-momentum', '0.9', '--dry-run'])
        with redirect_stdout(io.StringIO()):
            train(args)
        self.assertEqual((args.qk_clip, muon_options(args)), (0.0, Muon(momentum=0.9)))
        self.assertIsNone(muon_options(parse_args(['train'])))
        self.assertEqual(parse_args(['train']).optim_bits, 32)
        args = parse_args(['train', '--optimizer', 'muonclip', '--optim-bits', '8', '--dry-run'])
        with redirect_stdout(io.StringIO()) as out:
            train(args)
        self.assertIn('8-bit blockwise states', out.getvalue())
        with self.assertRaises(SystemExit):
            parse_args(['train', '--optim-bits', '16'])
        for argv in (['--qk-clip', '-1'], ['--muon-ns-steps', '0'], ['--muon-momentum', '1']):
            with self.assertRaises(SystemExit, msg=argv):
                parse_args(['train'] + argv)

    def test_invalid_shapes_rejected(self):
        for argv in (['--dim', '30'], ['--dim', '64', '--heads', '32'], ['--batch', '0'],
                     ['--mlp', '30']):
            with self.assertRaises(SystemExit, msg=argv):
                parse_args(['train'] + argv)


if __name__ == '__main__':
    unittest.main()

"""Check the Vulkan forward/backward/AdamW against a PyTorch reference.

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
from vabs_gpt.model import Config, Device, Model, Runner, decay_mask, param_layout, sample


def torch_loss(cfg, flat, inp, tgt, B):
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
        # N = 3 * 30 = 90 rows: two chunks of 64 and 26. mlp is not 4 * dim; head size 12.
        self.check_step(Config(ctx=30, dim=24, heads=2, layers=2, mlp=40), B=3, seed=2)

    def test_long_context_splits_dispatch(self):
        # B*H*T*T = 2*2*2048*2048 elements: attention dispatches exceed SPLIT_GROUPS.
        cfg = Config(ctx=2048, dim=8, heads=2, layers=1, mlp=16)
        self.assertGreater(2 * 2 * 2048 * 2048 // K.WG, K.SPLIT_GROUPS)
        self.check_step(cfg, B=2, seed=3)

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

    def test_invalid_shapes_rejected(self):
        for argv in (['--dim', '30'], ['--dim', '64', '--heads', '32'], ['--batch', '0'],
                     ['--mlp', '30']):
            with self.assertRaises(SystemExit, msg=argv):
                parse_args(['train'] + argv)


if __name__ == '__main__':
    unittest.main()

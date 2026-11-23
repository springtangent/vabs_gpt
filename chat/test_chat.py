"""Check the chat model's GPU forward/backward against PyTorch, plus the tokenizer and data code.

Needs PyTorch (the program itself does not). From the workspace root:

    $env:PYTHONPATH = "vabs\\src"; .ptvenv\\Scripts\\python.exe -m unittest vabs_gpt.chat.test_chat -v
"""
import array
import math
import os
import random
import tempfile
import unittest

import torch
import torch.nn.functional as F

from vabs_gpt.chat import data as D
from vabs_gpt.chat.model import (
    IGNORE,
    ChatConfig,
    ChatModel,
    ChatRunner,
    Device,
    coop_shapes_ok,
    init_params,
    matrix_shape,
    param_layout,
)
from vabs_gpt.chat.tokenizer import ASSISTANT, END, EOT, USER, Tokenizer
from vabs_gpt.gpu import ComputeContext
from vabs_gpt.model import Muon

SMALL = dict(vocab=1100, ctx=22, dim=32, heads=4, kv_heads=2, layers=2, mlp=40)


def torch_logits(cfg, flat, inp, T):
    """Reference forward in float64. ``inp``: LongTensor [B, T]."""
    B = inp.shape[0]
    C, V, H, HK, Fd = cfg.dim, cfg.vocab, cfg.heads, cfg.kv_heads, cfg.mlp
    hs, KV = cfg.head_size, cfg.kv_dim
    p, o = {}, 0
    for name, size in param_layout(cfg):
        p[name] = flat[o:o + size]
        o += size

    def rms(x, name):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + cfg.norm_eps) * p[name]

    t = torch.arange(T, dtype=torch.float64)[:, None]
    j = torch.arange(hs // 2, dtype=torch.float64)[None]
    ang = t * cfg.rope_theta ** (-2.0 * j / hs)
    cos, sin = torch.cos(ang), torch.sin(ang)

    def rope(x):   # [B, heads, T, hs], pairs (2j, 2j+1)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

    wte = p['wte'].view(V, C)
    x = wte[inp]
    for i in range(cfg.layers):
        h = f'h{i}.'
        qkv = rms(x, h + 'n1') @ p[h + 'qkv'].view(cfg.qkv_dim, C).T
        q, k, v = qkv.split((C, KV, KV), dim=2)
        q = rope(q.view(B, T, H, hs).transpose(1, 2))
        k = rope(k.view(B, T, HK, hs).transpose(1, 2)).repeat_interleave(H // HK, dim=1)
        v = v.view(B, T, HK, hs).transpose(1, 2).repeat_interleave(H // HK, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + y.transpose(1, 2).reshape(B, T, C) @ p[h + 'proj'].view(C, C).T
        gu = rms(x, h + 'n2') @ p[h + 'fc'].view(2 * Fd, C).T
        x = x + (F.silu(gu[..., :Fd]) * gu[..., Fd:]) @ p[h + 'down'].view(C, Fd).T
    return rms(x, 'nf') @ wte.T


def torch_loss(cfg, flat, inp, tgt, T):
    logits = torch_logits(cfg, flat, inp, T)
    t = tgt.clone()
    t[t >= cfg.vocab] = -100
    return F.cross_entropy(logits.reshape(-1, cfg.vocab), t.reshape(-1), ignore_index=-100,
                           reduction='sum'), int((t >= 0).sum())


def test_params(cfg, seed):
    rng = random.Random(seed + 1)
    flat = init_params(cfg, seed)
    out = array.array('f')
    for name, size in param_layout(cfg):
        chunk = flat[len(out):len(out) + size]
        if name.endswith(('n1', 'n2', 'nf')):
            out.extend(rng.uniform(0.5, 1.5) for _ in range(size))
        else:
            out.extend(x * 4 for x in chunk)
    return out


def random_batch(cfg, B, T, seed, ignore=0.3):
    rng = random.Random(seed)
    # a few repeated ids so the embedding scatter sums several positions
    inp = array.array('I', (rng.choice((rng.randrange(cfg.vocab), 7, 7, 3)) for _ in range(B * T)))
    tgt = array.array('I', (IGNORE if rng.random() < ignore else rng.randrange(cfg.vocab)
                            for _ in range(B * T)))
    return inp, tgt


class GPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = ComputeContext()
        cls.dev = Device(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.destroy()

    def check_grads(self, cfg, B, T, accum=1, seed=0, precision='f32', loss_tol=2e-4,
                    grad_tol=2e-3, norm_tol=1e-3):
        params = test_params(cfg, seed)
        model = ChatModel(self.dev, cfg, params, optim_bits=32, precision=precision)
        runner = ChatRunner(model, B, 'train', accum=accum, T=T)
        batches = [random_batch(cfg, B, T, seed + 10 + k) for k in range(accum)]
        # lr 0: the step leaves the parameters alone and the gradients readable
        loss, norm, scored = runner.train_step(batches, lr=0.0)
        flat = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        total, count = 0.0, 0
        for inp, tgt in batches:
            s, c = torch_loss(cfg, flat, torch.tensor(inp, dtype=torch.long).view(B, T),
                              torch.tensor(tgt, dtype=torch.long).view(B, T), T)
            total, count = total + s, count + c
        ref = total / count
        ref.backward()
        self.assertEqual(scored, count)
        self.assertAlmostEqual(loss, ref.item(), delta=loss_tol * abs(ref.item()))
        grads = torch.tensor(model.get_grads(), dtype=torch.float64)
        o = 0
        for name, size in param_layout(cfg):
            g, r = grads[o:o + size], flat.grad[o:o + size]
            err = (g - r).abs().max().item() / (r.abs().max().item() + 1e-12)
            if os.environ.get('CHAT_TEST_VERBOSE'):
                print(f'{name}: {err:.2e}')
            self.assertLess(err, grad_tol, f'{name}: relative error {err:.2e}')
            o += size
        self.assertAlmostEqual(norm, flat.grad.norm().item(), delta=norm_tol * norm)

    def test_grads(self):
        self.check_grads(ChatConfig(**SMALL), B=2, T=22)

    def test_grads_bf16(self):
        # every linear layer on the bfloat16 tensor cores: sizes that are multiples of 16
        if not self.ctx.supports_bf16_matmul:
            self.skipTest('no bfloat16 cooperative matrices on this device')
        cfg = ChatConfig(vocab=1104, ctx=16, dim=64, heads=4, kv_heads=2, layers=2, mlp=48)
        model = ChatModel(self.dev, cfg, test_params(cfg, 0), optim_bits=32, precision='bf16')
        self.assertTrue(all(model.uses_coop(32, *matrix_shape(cfg, name))
                            for name, _ in param_layout(cfg) if matrix_shape(cfg, name)))
        model.release()
        self.check_grads(cfg, B=2, T=16, precision='bf16', loss_tol=2e-3, grad_tol=3e-2,
                         norm_tol=5e-3)
        # sizes the tensor cores cannot take fall back to float32 kernels, exactly
        self.check_grads(ChatConfig(**SMALL), B=2, T=22, precision='bf16')

    def test_grads_bf16_attention_tiles(self):
        # a window of 144: causal tiles above, on and below the diagonal, a ragged last
        # tile row, and grouped heads (G = 2) summed into the key/value gradients
        if not self.ctx.supports_bf16_matmul:
            self.skipTest('no bfloat16 cooperative matrices on this device')
        cfg = ChatConfig(vocab=1104, ctx=144, dim=64, heads=4, kv_heads=2, layers=1, mlp=48)
        model = ChatModel(self.dev, cfg, test_params(cfg, 0), optim_bits=32, precision='bf16')
        self.assertTrue(ChatRunner(model, 1, 'eval').coop_attention)
        model.release()
        self.check_grads(cfg, B=2, T=144, precision='bf16', loss_tol=2e-3, grad_tol=3e-2,
                         norm_tol=5e-3)

    def test_grads_mha_short_window(self):
        # heads == kv_heads, a window shorter than the context, odd row counts
        self.check_grads(ChatConfig(**{**SMALL, 'kv_heads': 4, 'mlp': 36}), B=3, T=13)

    def test_grads_wide_groups(self):
        self.check_grads(ChatConfig(**{**SMALL, 'heads': 8, 'kv_heads': 2, 'dim': 64}), B=1, T=21)

    def test_accumulation(self):
        self.check_grads(ChatConfig(**SMALL), B=2, T=16, accum=3)

    def test_large_rows_per_chunk(self):
        # few enough chunks that a weight gradient partial sum spans more than 128 rows
        from vabs_gpt.chat import model as CM
        old = CM.PARTIAL_FLOATS
        CM.PARTIAL_FLOATS = 20000
        try:
            self.check_grads(ChatConfig(**SMALL), B=16, T=22)
        finally:
            CM.PARTIAL_FLOATS = old

    def test_weighted_grads(self):
        # a weight per token (policy gradient): the loss is sum_n w_n CE_n, weights of both signs
        cfg = ChatConfig(**SMALL)
        B, T, accum = 2, 16, 2
        params = test_params(cfg, 4)
        model = ChatModel(self.dev, cfg, params, optim_bits=32)
        runner = ChatRunner(model, B, 'train', accum=accum, T=T, weighted=True)
        batches = [random_batch(cfg, B, T, 40 + k) for k in range(accum)]
        rng = random.Random(9)
        weights = [array.array('f', (rng.uniform(-1, 1) if t < cfg.vocab else 0.0 for t in tgt))
                   for _, tgt in batches]
        runner.train_step(batches, lr=0.0, weights=weights)
        flat = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        ref = 0.0
        for (inp, tgt), w in zip(batches, weights):
            logits = torch_logits(cfg, flat, torch.tensor(inp, dtype=torch.long).view(B, T), T)
            t = torch.tensor(tgt, dtype=torch.long)
            t[t >= cfg.vocab] = -100
            ce = F.cross_entropy(logits.reshape(-1, cfg.vocab), t, ignore_index=-100,
                                 reduction='none')
            ref = ref + (ce * torch.tensor(w, dtype=torch.float64)).sum()
        ref.backward()
        grads = torch.tensor(model.get_grads(), dtype=torch.float64)
        err = (grads - flat.grad).abs().max().item() / flat.grad.abs().max().item()
        self.assertLess(err, 2e-3)

    def test_decoder_matches_full_forward(self):
        from vabs_gpt.chat.decode import Decoder
        cfg = ChatConfig(**SMALL)
        params = test_params(cfg, 6)
        model = ChatModel(self.dev, cfg, params)
        flat = torch.tensor(params, dtype=torch.float64)
        rng = random.Random(2)
        prompts = [[rng.randrange(cfg.vocab) for _ in range(n)] for n in (3, 9, 1)]
        dec = Decoder(model, 4, cfg.ctx, steps=3)
        outs = dec.generate(prompts, max_new=12, temperature=0, stop=())
        for p, o in zip(prompts, outs):
            self.assertEqual(len(o), min(12, cfg.ctx - len(p)))
            seq = p + o
            logits = torch_logits(cfg, flat, torch.tensor([seq], dtype=torch.long), len(seq))[0]
            for i, t in enumerate(o):
                row = logits[len(p) - 1 + i]
                # greedy: the pick is the argmax of the full forward (up to float32 ties)
                self.assertGreater(row[t].item(), row.max().item() - 1e-3 * row.abs().max().item())
        dec.release()

    def test_decoder_bf16(self):
        # 16 sequences: the decode step's products run on the tensor cores too
        if not self.ctx.supports_bf16_matmul:
            self.skipTest('no bfloat16 cooperative matrices on this device')
        from vabs_gpt.chat.decode import Decoder
        cfg = ChatConfig(vocab=1104, ctx=16, dim=64, heads=4, kv_heads=2, layers=2, mlp=48)
        params = test_params(cfg, 6)
        model = ChatModel(self.dev, cfg, params, precision='bf16')
        self.assertTrue(model.uses_coop(16, cfg.dim, cfg.vocab))
        flat = torch.tensor(params, dtype=torch.float64)
        rng = random.Random(3)
        prompts = [[rng.randrange(cfg.vocab) for _ in range(rng.randrange(1, 9))]
                   for _ in range(16)]
        dec = Decoder(model, 16, cfg.ctx, steps=3)
        outs = dec.generate(prompts, max_new=6, temperature=0, stop=())
        for p, o in zip(prompts, outs):
            seq = p + o
            logits = torch_logits(cfg, flat, torch.tensor([seq], dtype=torch.long), len(seq))[0]
            for i, t in enumerate(o):
                row = logits[len(p) - 1 + i]
                # greedy within bfloat16 rounding of the float64 argmax
                self.assertGreater(row[t].item(), row.max().item() - 3e-2 * row.abs().max().item())
        dec.release()
        model.release()

    def test_sampler_distribution(self):
        # Gumbel-max sampling on the GPU draws from softmax(logits / temperature)
        from vabs_gpt.chat.decode import Decoder
        cfg = ChatConfig(**SMALL)
        params = test_params(cfg, 7)
        model = ChatModel(self.dev, cfg, params)
        prompt = [5, 17, 300]
        B, rounds, temp = 64, 40, 0.7
        dec = Decoder(model, B, cfg.ctx, steps=1)
        rng = random.Random(0)
        counts = {}
        for _ in range(rounds):
            for o in dec.generate([prompt] * B, max_new=1, temperature=temp, stop=(), rng=rng):
                counts[o[0]] = counts.get(o[0], 0) + 1
        dec.release()
        flat = torch.tensor(params, dtype=torch.float64)
        logits = torch_logits(cfg, flat, torch.tensor([prompt], dtype=torch.long), 3)[0, -1]
        probs = torch.softmax(logits / temp, -1)
        n = B * rounds
        for t in probs.argsort(descending=True)[:6].tolist():
            p = probs[t].item()
            self.assertLess(abs(counts.get(t, 0) / n - p), 4 * math.sqrt(p * (1 - p) / n) + 2e-3,
                            f'token {t}: p={p:.4f}')

    def test_eval_and_infer(self):
        cfg = ChatConfig(**SMALL)
        params = test_params(cfg, 3)
        model = ChatModel(self.dev, cfg, params)
        B, T = 2, 22
        inp, tgt = random_batch(cfg, B, T, 5)
        ev = ChatRunner(model, B, 'eval')
        total, count = ev.run(inp, tgt)
        flat = torch.tensor(params, dtype=torch.float64)
        ref, rc = torch_loss(cfg, flat, torch.tensor(inp, dtype=torch.long).view(B, T),
                             torch.tensor(tgt, dtype=torch.long).view(B, T), T)
        self.assertEqual(count, rc)
        self.assertAlmostEqual(total / count, ref.item() / rc, delta=1e-4 * ref.item() / rc)
        # inference on a shorter window: positions start at 0, so the logits match
        Ti = 12
        inf = ChatRunner(model, B, 'infer', T=Ti)
        toks = array.array('I', inp[:Ti]) + array.array('I', inp[T:T + Ti])
        pos = [Ti - 1, 4]
        got = inf.next_logits(toks, pos)
        ref = torch_logits(cfg, flat, torch.tensor(toks, dtype=torch.long).view(B, Ti), Ti)
        for b in range(B):
            r = ref[b, pos[b]]
            err = (torch.tensor(got[b], dtype=torch.float64) - r).abs().max().item()
            self.assertLess(err, 1e-3 * r.abs().max().item())

    def test_optimizers_reduce_loss(self):
        cfg = ChatConfig(**SMALL)
        inp, _ = random_batch(cfg, 2, 22, 1, ignore=0.0)
        tgt = array.array('I', inp[1:] + inp[:1])
        for bits, muon in ((32, None), (8, None), (8, Muon()), (32, Muon())):
            model = ChatModel(self.dev, cfg, seed=0, optim_bits=bits, muon=bool(muon))
            runner = ChatRunner(model, 2, 'train', muon=muon)
            losses = [runner.train_step([(inp, tgt)], lr=3e-3, weight_decay=0.1, clip=1.0)[0]
                      for _ in range(30)]
            self.assertLess(losses[-1], 0.6 * losses[0], f'{bits}-bit, muon={bool(muon)}')
            for x in losses:
                self.assertTrue(math.isfinite(x))

    def test_muon_step_bf16(self):
        # Muon with every product on the bf16 tensor cores updates the parameters as the
        # float32 step does, up to bfloat16 rounding
        if not self.ctx.supports_bf16_matmul:
            self.skipTest('no bfloat16 cooperative matrices on this device')
        cfg = ChatConfig(vocab=1104, ctx=16, dim=64, heads=4, kv_heads=2, layers=2, mlp=48)
        inp, tgt = random_batch(cfg, 2, 16, 5, ignore=0.0)
        params = test_params(cfg, 2)
        muon_names = {name for name, _ in param_layout(cfg)
                      if matrix_shape(cfg, name) and name != 'wte'}

        def step(precision, linear_f32=False):
            model = ChatModel(self.dev, cfg, params, optim_bits=32, muon=True,
                              precision=precision)
            if linear_f32:
                model.uses_coop = lambda *dims: False   # only Newton-Schulz in bf16
            runner = ChatRunner(model, 2, 'train', muon=Muon())
            if precision == 'bf16':
                self.assertTrue(all(coop_shapes_ok(g['m'], g['n']) for g in runner.groups))
            runner.train_step([(inp, tgt)], lr=1e-2)
            delta = (torch.tensor(model.get_params(), dtype=torch.float64)
                     - torch.tensor(params, dtype=torch.float64))
            runner.release()
            model.release()
            out, o = {}, 0
            for name, size in param_layout(cfg):
                out[name] = delta[o:o + size]
                o += size
            return out

        ref = step('f32')
        for label, got, tol, min_cos in (('Newton-Schulz in bf16', step('bf16', True), 3e-2,
                                          0.9995),
                                         ('all in bf16', step('bf16'), 0.15, 0.99)):
            for name in ref:
                r, g = ref[name], got[name]
                err = (g - r).norm().item() / r.norm().item()
                cos = F.cosine_similarity(g, r, dim=0).item()
                if os.environ.get('CHAT_TEST_VERBOSE'):
                    print(f'{label} {name}: {err:.2e} cos {cos:.5f}')
                if name in muon_names:
                    self.assertLess(err, tol, f'{label}, {name}: update difference {err:.2e}')
                    self.assertGreater(cos, min_cos, f'{label}, {name}: cosine {cos:.5f}')
                elif label == 'Newton-Schulz in bf16':
                    # AdamW parameters see the same float32 gradients
                    self.assertLess(err, 1e-6, f'{label}, {name}: update difference {err:.2e}')
                else:
                    # Adam's first step is nearly sign(g): rounding flips tiny gradients
                    self.assertGreater(cos, 0.95, f'{label}, {name}: cosine {cos:.5f}')

    def test_release_frees_arenas(self):
        cfg = ChatConfig(**SMALL)
        before = len(self.ctx._arenas)
        model = ChatModel(self.dev, cfg, seed=0)
        for mode in ('train', 'eval', 'infer'):
            n = len(self.ctx._arenas)
            r = ChatRunner(model, 2, mode)
            self.assertGreater(len(self.ctx._arenas), n)
            r.release()
            self.assertEqual(len(self.ctx._arenas), n)
        model.release()
        self.assertEqual(len(self.ctx._arenas), before)

    def test_checkpoint_roundtrip(self):
        cfg = ChatConfig(**SMALL)
        inp, tgt = random_batch(cfg, 2, 22, 2)
        for muon in (None, Muon()):
            model = ChatModel(self.dev, cfg, seed=4, optim_bits=8, muon=bool(muon),
                              tokenizer=[[1, 2]])
            runner = ChatRunner(model, 2, 'train', muon=muon)
            runner.train_step([(inp, tgt)], lr=1e-3)
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, 'c.bin')
                model.save(path, extra={'note': 1})
                header, params, state = ChatModel.read(path)
            self.assertEqual(header['note'], 1)
            self.assertEqual(header['tokenizer'], [[1, 2]])
            self.assertIsNotNone(state)
            again = ChatModel(self.dev, cfg, params, step=header['step'], **state)
            r2 = ChatRunner(again, 2, 'train', muon=muon)
            runner.train_step([(inp, tgt)], lr=1e-3)
            r2.train_step([(inp, tgt)], lr=1e-3)
            self.assertEqual(model.get_params(), again.get_params())


class CoopMatmulTest(unittest.TestCase):
    """``mm_coop`` (bfloat16 tensor cores) against float64 products of bf16-rounded inputs."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = ComputeContext()
        if not cls.ctx.supports_bf16_matmul:
            cls.ctx.destroy()
            raise unittest.SkipTest('no bfloat16 cooperative matrices on this device')
        cls.dev = Device(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.destroy()

    def run_mm(self, M, N, Kd, a_col, b_col, residual=False, splits=1, seed=0):
        from vabs.memory import MemoryUsage
        from vabs.renderer import one_shot_frame

        from vabs_gpt.chat.model import coop_splits
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(M, Kd, generator=g)
        b = torch.randn(Kd, N, generator=g)
        r = torch.randn(M, N, generator=g)
        a_mem = (a.T if a_col else a).contiguous()
        b_mem = (b.T if b_col else b).contiguous()
        kc = None
        if splits > 1:
            splits, kc = coop_splits(M, N, Kd)
        ra, rb, rc, rr = self.dev.arena([4 * M * Kd, 4 * Kd * N, 4 * M * N * splits, 4 * M * N],
                                        MemoryUsage.DEVICE_LOCAL)
        self.dev.upload(ra, a_mem.numpy().tobytes())
        self.dev.upload(rb, b_mem.numpy().tobytes())
        self.dev.upload(rr, r.numpy().tobytes())
        with one_shot_frame(self.ctx) as frame:
            got = self.dev.mm_coop(frame, rc.device_address, ra.device_address,
                                   rb.device_address, M, N, Kd, a_col=a_col, b_col=b_col,
                                   res=rr.device_address if residual else 0, kc=kc)
        self.assertEqual(got, splits)
        out = torch.frombuffer(bytearray(self.dev.download(rc)), dtype=torch.float32)
        out = out.view(splits, M, N).double().sum(0)
        ref = a.bfloat16().double() @ b.bfloat16().double()
        if residual:
            ref = ref + r.double()
        err = (out - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(err, 1e-5, f'relative error {err:.2e}')

    def test_layouts(self):
        for a_col in (False, True):
            for b_col in (False, True):
                with self.subTest(a_col=a_col, b_col=b_col):
                    self.run_mm(128, 192, 96, a_col, b_col)

    def test_ragged_tiles_and_residual(self):
        # sizes that are multiples of 16 but not of the 64 x 64 x 32 tiles
        self.run_mm(80, 208, 48, False, True, residual=True)
        self.run_mm(16, 1104, 528, False, False, residual=True, seed=1)

    def test_split_reduction(self):
        self.run_mm(64, 96, 2048, True, False, splits=8)


class TokenizerTest(unittest.TestCase):
    TEXT = ('def add(a, b):\n    """Return a + b."""\n    return a + b\n\n'
            'Traceback (most recent call last):\n  File "x.py", line 2\n'
            'IndexError: list index out of range\n') * 40 + 'The cat sat on the mat. ' * 50

    def test_train_roundtrip(self):
        counts = Tokenizer.count([self.TEXT])
        tok = Tokenizer.train(counts, 400)
        self.assertEqual(tok.vocab_size, 400)
        self.assertGreater(len(tok.merges), 50)
        for text in (self.TEXT, 'héllo wörld 😀\r\n\t x', '<|user|> hi', ''):
            ids = tok.encode(text)
            self.assertEqual(tok.decode(ids), text)
            for special in (EOT, USER, ASSISTANT, END):
                self.assertNotIn(special, ids)
        self.assertLess(len(tok.encode(self.TEXT)), len(self.TEXT.encode()) / 3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 't.json')
            tok.save(path)
            again = Tokenizer.load(path)
        self.assertEqual(again.encode(self.TEXT), tok.encode(self.TEXT))

    def test_indentation_tokens(self):
        tok = Tokenizer.train(Tokenizer.count([self.TEXT]), 400)
        pieces = tok.show('    return a + b\n')
        self.assertEqual(''.join(pieces), '    return a + b\n')
        self.assertIn('   ', pieces)


class DataTest(unittest.TestCase):
    def test_ignore_matches_kernels(self):
        from vabs_gpt.chat import kernels as K
        self.assertEqual(D.IGNORE, K.IGNORE)
        self.assertEqual(IGNORE, K.IGNORE)

    def test_chat_tokens_and_mask(self):
        tok = Tokenizer.train(Tokenizer.count([TokenizerTest.TEXT]), 400)
        chat = [('user', 'hi'), ('assistant', 'hello'), ('user', 'more'), ('assistant', 'ok')]
        ids, scored = D.encode_chat(tok, chat)
        self.assertEqual(len(ids), len(scored))
        self.assertEqual(ids[:2], [EOT, USER])
        # scored tokens are exactly the assistant replies and their END tokens
        replies = [i for i, s in zip(ids, scored) if s]
        self.assertEqual(replies.count(END), 2)
        self.assertEqual(tok.decode([i for i in replies if i != END]), 'hellook')
        packed = D.pack_chats([(ids, scored)] * 5, 3 * len(ids))
        self.assertEqual(len(packed), 2)
        inp, tgt = D.sft_example(ids, scored, 32)
        self.assertEqual(len(inp), 32)
        for k in range(32):
            if tgt[k] != IGNORE:
                self.assertTrue(scored[k + 1])
                self.assertEqual(tgt[k], ids[k + 1])

    def test_completion_records(self):
        tok = Tokenizer.train(Tokenizer.count([TokenizerTest.TEXT]), 400)
        ids, scored = D.encode_chat(tok, [('prompt', 'def f():\n'), ('completion', '    return 1\n')])
        self.assertEqual(ids[0], EOT)
        self.assertEqual(ids[-1], EOT)
        self.assertEqual(tok.decode([i for i, s in zip(ids, scored) if s and i != EOT]),
                         '    return 1\n')
        self.assertFalse(any(scored[:1 + len(tok.encode('def f():\n'))]))


class MuonGroupTest(unittest.TestCase):
    def test_batches_fit_one_dispatch(self):
        """A batch too big for one bmm dispatch wraps into split-K rows and loses the device."""
        from vabs_gpt.chat.model import MUON_BATCH_ELEMS, muon_groups, muon_matrices
        for cfg in (ChatConfig(), ChatConfig(dim=640, heads=10, kv_heads=5, layers=12, mlp=1664)):
            groups = muon_groups(cfg)
            for m, n, mats in groups:
                self.assertLessEqual(len(mats) * m * n, MUON_BATCH_ELEMS)
                self.assertTrue(all((min(r, c), max(r, c)) == (m, n) for _, r, c in mats))
            flat = sorted(x for _, _, mats in groups for x in mats)
            self.assertEqual(flat, sorted(muon_matrices(cfg)))
        # the 33M model keeps one batch per shape (its checkpoints' 8-bit state layout)
        self.assertEqual(len(muon_groups(ChatConfig())), 3)


class RLTest(unittest.TestCase):
    def test_advantages(self):
        from vabs_gpt.chat.rlvr import advantages
        a = advantages([1, 0, 0, 0, 1, 1, 1, 1], 4)
        self.assertEqual(a[4:], [0.0] * 4)            # a group with one reward: no signal
        self.assertAlmostEqual(sum(a[:4]), 0.0)
        self.assertAlmostEqual(a[0], 0.75 / math.sqrt(0.1875))
        self.assertEqual(advantages([1, 0], 2, std=False), [0.5, -0.5])

    def test_training_rows(self):
        from vabs_gpt.chat.rlvr import training_rows
        samples = [([EOT, 5, 6], [7, 8, EOT], ''), ([EOT, 5], [9], ''), ([EOT], [4, 4], '')]
        rows, total = training_rows(samples, [1.0, 0.0, -0.5], 8)
        self.assertEqual(total, 5)                    # the zero-advantage sample is left out
        self.assertEqual(len(rows), 2)
        inp, tgt, w = rows[0]
        self.assertEqual(list(inp[:5]), [EOT, 5, 6, 7, 8])
        self.assertEqual(list(tgt[:6]), [IGNORE, IGNORE, 7, 8, EOT, IGNORE])
        self.assertEqual([round(x, 6) for x in w[:6]], [0, 0, 0.2, 0.2, 0.2, 0])
        self.assertTrue(all(abs(x + 0.1) < 1e-7 for x, t in zip(rows[1][2], rows[1][1]) if t != IGNORE))

    def test_stub_and_example(self):
        from vabs_gpt.chat import rl_data as R
        code = 'import re\nR = 3\ndef f(a, b=2):\n\treturn a + b\n'
        stub, alone = R.make_stub(code, 'f', 'Add two numbers.',
                                  R.example_lines('assert f(1, 2) == 3', 'f'))
        self.assertTrue(stub.startswith('import re\n\n\ndef f(a, b=2):\n    """Add two numbers.'))
        self.assertIn('    >>> f(1, 2)\n    3\n', stub)
        self.assertNotIn('R = 3', alone)
        self.assertEqual(R.example_lines('assert set(f(1)) == {1}', 'f')[0], 'Example:')

    def test_acecode_task(self):
        from vabs_gpt.chat import rl_data as R
        from vabs_gpt.chat import verify as V
        good = ('Here it is:\n```python\ndef double(x):\n    return x * 2\n\n'
                'print(double(3))\n```\n')
        row = dict(id='oss_7', question='Implement `double`.\n```python\ndef double(x):\n'
                   '    pass\n```', test_cases=['assert double(1) == 2', 'assert double(5) == 10'],
                   inferences=[dict(completion=good, pass_rate=1.0),
                               dict(completion='def double(x):\n    return x', pass_rate=0.0)])
        t = R.acecode_task(row, set(), 0.5)
        self.assertEqual(t['id'], 'acecode/oss_7')
        self.assertTrue(t['stub'].startswith('def double(x):\n    """Implement `double`.'))
        self.assertIn('>>> double(1)', t['stub'])
        self.assertNotIn('pass', t['stub'])            # the question's code block is dropped
        self.assertNotIn('print', t['reference'])      # only the function is kept
        self.assertNotIn('double(1)', t['test'])       # the shown example is not a hidden test
        self.assertTrue(V.check(t['reference'], t['test_full']))
        self.assertFalse(V.check('def double(x):\n    return x + 1\n', t['test_full']))
        self.assertIsNone(R.acecode_task(row, set(), 0.6))       # mean pass rate 0.5: too hard
        self.assertIsNone(R.acecode_task(row, {'double'}, 0.5))  # an evaluation task's name

    def test_acecode_body(self):
        from vabs_gpt.chat import rl_data as R
        ref = 'import re\n\n\ndef f(s):\n    """Own docstring."""\n    s = s.strip()\n    return s\n'
        self.assertEqual(R.acecode_body(dict(reference=ref, name='f')),
                         '    s = s.strip()\n    return s\n')
        one = 'def f(s):\n    """Only a docstring."""\n'
        self.assertIsNone(R.acecode_body(dict(reference=one, name='f')))
        self.assertIsNone(R.acecode_body(dict(reference='def f(s): return s\n', name='f')))


class VerifyTest(unittest.TestCase):
    def test_sandbox(self):
        from vabs_gpt.chat import verify as V
        test = 'from solution import add\nassert add(1, 2) == 3\n'
        good = 'def add(a, b):\n    return a + b\n'
        self.assertTrue(V.check(good, test))
        self.assertTrue(V.check(good, 'from solution import add\n\ndef test_a():\n'
                                      '    assert add(2, 2) == 4\n'))
        for bad in ('def add(a, b):\n    return a - b\n',
                    'import socket\n' + good,
                    'open("x.txt", "w")\n' + good,
                    'import os\nos.system("echo hi")\n' + good,
                    'print("PASS 0 1")\nimport os\nos._exit(0)\n' + good,
                    'import sys\nsys.exit(0)\n'):
            self.assertFalse(V.check(bad, test), bad)


if __name__ == '__main__':
    unittest.main()

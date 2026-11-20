"""The same model and training run in PyTorch, for comparison with the Vulkan one.

Everything that defines a run is shared with the vabs program: the parameter
layout and initial values (``init_params``), the batch order (``Batches``), the
learning-rate schedule (``lr_at``), the AdamW decay mask and the evaluation
windows. Checkpoints use the same file format, so either program can score
the other's model:

    $env:PYTHONPATH = "vabs\\src"
    .ptvenv\\Scripts\\python.exe -m vabs_gpt.torch_ref train --dim 128 --layers 5 --heads 8 `
        --epochs 2 --out vabs_gpt\\torch.bin
    .ptvenv\\Scripts\\python.exe -m vabs_gpt.torch_ref eval vabs_gpt\\epoch.bin   # a vabs checkpoint
    .venv\\Scripts\\python.exe -m vabs_gpt eval vabs_gpt\\torch.bin               # and the reverse

``train`` takes the options of ``python -m vabs_gpt train`` (same parser), plus
``--device``. It runs float32 without autocast or compilation.
"""
import array
import json
import math
import os
import struct
import sys
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from .__main__ import Batches, build_parser, epoch_windows, lr_at, model_config, open_bytes
from .model import Config, Model, init_params, param_layout


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg.dim
        self.heads = cfg.heads
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.qkv, self.attn_proj = nn.Linear(C, 3 * C), nn.Linear(C, C)
        self.fc, self.fc_proj = nn.Linear(C, cfg.mlp), nn.Linear(cfg.mlp, C)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.fc_proj(F.gelu(self.fc(self.ln2(x)), approximate='tanh'))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.dim)
        self.wpe = nn.Embedding(cfg.ctx, cfg.dim)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.lnf = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab)

    def forward(self, inp, tgt=None):
        x = self.wte(inp) + self.wpe.weight
        for block in self.h:
            x = block(x)
        logits = self.head(self.lnf(x))
        if tgt is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1))

    def tensors(self):
        """``[(weight, bias or None)]`` in the order of ``param_layout``."""
        out = [(self.wte.weight, None), (self.wpe.weight, None)]
        for b in self.h:
            out += [(m.weight, m.bias) for m in (b.ln1, b.qkv, b.attn_proj, b.ln2, b.fc,
                                                 b.fc_proj)]
        return out + [(self.lnf.weight, self.lnf.bias), (self.head.weight, self.head.bias)]

    def load_flat(self, flat):
        """Copy a flat float32 parameter array (the vabs layout) into the modules."""
        flat = torch.tensor(flat, dtype=torch.float32)
        o = 0
        with torch.no_grad():
            for (name, size), (w, b) in zip(param_layout(self.cfg), self.tensors(), strict=True):
                assert w.numel() + (b.numel() if b is not None else 0) == size, name
                for t in (w, b):
                    if t is not None:
                        t.copy_(flat[o:o + t.numel()].view_as(t))
                        o += t.numel()

    def flat(self):
        return torch.cat([t.detach().reshape(-1).cpu() for pair in self.tensors()
                          for t in pair if t is not None])

    def decay_groups(self, weight_decay):
        """Decay matrices and embeddings; not biases or layer norms (the vabs decay mask)."""
        decay, rest = [], []
        for w, b in self.tensors():
            (decay if w.dim() == 2 else rest).append(w)
            if b is not None:
                rest.append(b)
        return [{'params': decay, 'weight_decay': weight_decay},
                {'params': rest, 'weight_decay': 0.0}]


def save(path, model, step, extra):
    header = json.dumps({'config': asdict(model.cfg), 'step': step, 'optimizer': False,
                         'framework': 'torch', **extra}).encode()
    with open(path, 'wb') as f:
        f.write(b'VGPT' + struct.pack('<I', len(header)) + header)
        f.write(array.array('f', model.flat().tolist()).tobytes())


def to_tokens(data, B, device):
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device).long().view(B, -1)


@torch.no_grad()
def evaluate_file(model, path, batch, max_bytes, device):
    """Mean loss over consecutive windows from the start of a file (as ``vabs_gpt eval``)."""
    T = model.cfg.ctx
    data = open_bytes(path, T)
    windows = (len(data) - 1) // T
    if max_bytes:
        windows = min(windows, max_bytes // T)
    batches, span = windows // batch, batch * T
    total = torch.zeros((), device=device, dtype=torch.float64)
    for b in range(batches):
        chunk = data[b * span:(b + 1) * span + 1]
        total += model(to_tokens(chunk[:-1], batch, device), to_tokens(chunk[1:], batch, device))
    return total.item() / batches, batches * span


def train(args):
    device = torch.device(args.device)
    cfg = model_config(args)
    if args.min_lr is None:
        args.min_lr = args.lr * 0.1
    train_path = os.path.join(args.data, 'train.bin')
    per_step = args.batch * args.grad_accum
    if args.epochs:
        args.steps = math.ceil(
            args.epochs * epoch_windows(os.path.getsize(train_path), cfg.ctx) / per_step)
    out = args.out or os.path.join(os.path.dirname(__file__), 'torch.bin')
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f'{out} already exists; pass --overwrite or --out')

    model = GPT(cfg).to(device)
    model.load_flat(init_params(cfg, args.seed))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.decay_groups(args.weight_decay), lr=args.lr,
                            betas=(args.beta1, args.beta2), eps=args.eps)
    tokens = per_step * cfg.ctx
    print(f'torch {torch.__version__} on {device}'
          + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ''))
    print(f'model: {n_params:,} parameters; {args.steps:,} steps x {tokens:,} tokens/step, '
          f'lr {args.lr:g} -> {args.min_lr:g}, betas ({args.beta1}, {args.beta2}), weight decay '
          f'{args.weight_decay:g}, grad clip {args.grad_clip:g}', flush=True)

    order = 'epoch' if args.epochs else 'random'
    train_data = Batches(train_path, args.batch, cfg.ctx, args.seed, order)
    val_data = Batches(os.path.join(args.data, 'val.bin'), args.batch, cfg.ctx, 1234)
    val_batches = [tuple(to_tokens(x, args.batch, device) for x in val_data.get())
                   for _ in range(args.eval_batches)]

    @torch.no_grad()
    def evaluate():
        return sum(model(i, t).item() for i, t in val_batches) / len(val_batches)

    def checkpoint(val):
        options = {k: v for k, v in vars(args).items()
                   if k not in ('cmd', 'config', 'dry_run', 'resume', 'overwrite', 'device')}
        save(out, model, step, {'val_loss': val, 'train': options})

    print(f'step 0: val loss {evaluate():.4f}', flush=True)
    t0 = time.perf_counter()
    loss_sum = torch.zeros((), device=device)
    norm_sum = torch.zeros((), device=device)
    logged = 0
    for step in range(1, args.steps + 1):
        lr = lr_at(step - 1, args)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            inp, tgt = (to_tokens(x, args.batch, device) for x in train_data.get())
            loss = model(inp, tgt) / args.grad_accum
            loss.backward()
            loss_sum += loss.detach()
        # max_norm = inf only measures the norm
        norm_sum += nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip or math.inf)
        opt.step()
        if step % args.log_every == 0:
            mean = loss_sum.item() / (step - logged)
            if not math.isfinite(mean):
                raise SystemExit(f'step {step}: loss is {mean}')
            dt = time.perf_counter() - t0
            print(f'step {step:6d}  loss {mean:.4f}  grad norm '
                  f'{norm_sum.item() / (step - logged):.3f}  lr {lr:.2e}  '
                  f'{dt / step * 1000:.0f} ms/step  {tokens * step / dt:,.0f} tok/s', flush=True)
            loss_sum.zero_()
            norm_sum.zero_()
            logged = step
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate()
            print(f'step {step:6d}  val loss {val:.4f} ({val / math.log(2):.3f} bits/byte)',
                  flush=True)
            checkpoint(val)
    print(f'saved {out}  ({time.perf_counter() - t0:.0f} s)')


def evaluate_checkpoint(args):
    device = torch.device(args.device)
    header, params, _ = Model.read(args.checkpoint)
    model = GPT(Config(**header['config'])).to(device)
    model.load_flat(params)
    t0 = time.perf_counter()
    loss, n = evaluate_file(model, os.path.join(args.data, args.split + '.bin'), args.batch,
                            args.bytes, device)
    print(f'{args.checkpoint}: step {header.get("step")}, {args.split} loss {loss:.4f} '
          f'({loss / math.log(2):.4f} bits/byte) over {n:,} bytes '
          f'({time.perf_counter() - t0:.0f} s) [torch]')
    if args.json:
        print(json.dumps({'loss': loss, 'bytes': n, 'step': header.get('step')}))


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser, _ = build_parser()
    parser.prog = 'python -m vabs_gpt.torch_ref'
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    if args.cmd == 'train':
        if args.resume or args.config:
            raise SystemExit('torch_ref train does not support --resume or --config')
        train(args)
    elif args.cmd == 'eval':
        evaluate_checkpoint(args)
    else:
        raise SystemExit(f'torch_ref supports train and eval, not {args.cmd}')


if __name__ == '__main__':
    main()

"""Train or sample a byte-level TinyStories GPT on Vulkan (vabs compute).

    python -m vabs_gpt train                                  # defaults: ~250K parameters
    python -m vabs_gpt train --dim 128 --layers 6 --heads 8 --lr 1e-3 --steps 20000
    python -m vabs_gpt train --config run.json --dry-run      # print sizes and memory, exit
    python -m vabs_gpt train --resume vabs_gpt/model.bin --steps 20000
    python -m vabs_gpt sample --prompt "Once upon a time"
    python -m vabs_gpt train --epochs 1 --out vabs_gpt/epoch.bin   # one pass over train.bin
    python -m vabs_gpt eval vabs_gpt/model.bin                # loss on the whole of val.bin
    python -m vabs_gpt info vabs_gpt/model.bin

A ``--config`` JSON file holds any train option by its long name (dashes or
underscores), e.g. {"dim": 128, "layers": 6, "lr": 0.001}; command-line flags
override it. Checkpoints record the model shape and the training options used.
"""
import argparse
import json
import math
import mmap
import os
import random
import sys
import time
from dataclasses import asdict, fields

from .gpu import ComputeContext
from .model import (Config, Device, Model, Muon, Runner, activation_bytes, model_bytes,
                    optimizer_bytes, param_counts, sample)

HERE = os.path.dirname(__file__)
DEFAULT_OUT = os.path.join(HERE, 'model.bin')
MODEL_KEYS = [f.name for f in fields(Config) if f.name != 'vocab']


def open_bytes(path, T):
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found; run vabs_gpt/prepare.py first')
    with open(path, 'rb') as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if len(data) <= 2 * T + 1:
        raise SystemExit(f'{path} is shorter than two context windows')
    return data


def epoch_windows(n_bytes, T):
    """Non-overlapping windows per epoch (room is left for a per-epoch shift below T)."""
    return (n_bytes - 2 * T) // T


class Batches:
    """Batches of B windows of T+1 bytes from a prepared byte stream.

    ``order='random'``: every window starts at a uniformly random offset.
    ``order='epoch'``: the stream is cut into non-overlapping windows and each
    epoch visits every window once, in a pseudo-random order (a keyed Feistel
    permutation, so no index table is stored). Later epochs shift the window
    grid by a random amount below T. ``start`` is the number of windows already
    consumed, which lets a resumed run continue the same order.
    """

    def __init__(self, path, B, T, seed, order='random', start=0):
        self.data = open_bytes(path, T)
        self.B, self.T, self.seed, self.order = B, T, seed, order
        self.rng = random.Random(seed)
        self.n = epoch_windows(len(self.data), T)
        self.half = max(1, ((self.n - 1).bit_length() + 1) // 2)
        self.pos, self.epoch, self.keys, self.shift = start, None, (), 0

    def _permute(self, i):
        mask = (1 << self.half) - 1
        while True:  # cycle-walk: the Feistel domain is up to 4x n
            hi, lo = i >> self.half, i & mask
            for k in self.keys:
                x = ((lo ^ k) * 0x45D9F3B) & 0xFFFFFFFF
                x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
                hi, lo = lo, hi ^ ((x ^ (x >> 16)) & mask)
            i = (hi << self.half) | lo
            if i < self.n:
                return i

    def _offset(self):
        if self.order == 'random':
            return self.rng.randrange(len(self.data) - self.T - 1)
        epoch, i = divmod(self.pos, self.n)
        self.pos += 1
        if epoch != self.epoch:
            rng = random.Random(self.seed * 1_000_003 + epoch)
            self.epoch = epoch
            self.keys = tuple(rng.getrandbits(32) for _ in range(4))
            self.shift = rng.randrange(self.T) if epoch else 0
        return self.shift + self._permute(i) * self.T

    def close(self):
        self.data.close()

    def get(self):
        inp, tgt = bytearray(), bytearray()
        for _ in range(self.B):
            o = self._offset()
            chunk = self.data[o:o + self.T + 1]
            inp += chunk[:-1]
            tgt += chunk[1:]
        return bytes(inp), bytes(tgt)


def lr_at(step, args):
    """Learning rate for 0-based ``step``: linear warmup, then the chosen decay to min_lr."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if args.schedule == 'constant':
        return args.lr
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    shape = 0.5 * (1 + math.cos(math.pi * progress)) if args.schedule == 'cosine' \
        else 1.0 - progress
    return args.min_lr + (args.lr - args.min_lr) * shape


def model_config(args):
    return Config(**{k: getattr(args, k) for k in MODEL_KEYS})




def device_local_bytes(ctx):
    import vk
    return max((heap.size for heap in ctx.mem_props.memory_heaps
                if heap.flags & vk.MemoryHeapFlags.DEVICE_LOCAL), default=0)


def muon_options(args):
    """The ``Muon`` options of a run, or None for AdamW alone."""
    if args.optimizer == 'adamw':
        return None
    return Muon(momentum=args.muon_momentum, ns_steps=args.muon_ns_steps, scale=args.muon_scale)


def optimizer_family(name):
    """Optimizers whose saved moments mean the same thing (muonclip keeps Muon's state)."""
    return 'muon' if name in ('muon', 'muonclip') else name


def device_bytes_needed(cfg, args):
    muon = muon_options(args)
    return (activation_bytes(cfg, args.batch, 'train') + activation_bytes(cfg, args.batch, 'eval')
            + optimizer_bytes(cfg, args.batch, muon, args.qk_clip or 0.0, args.optim_bits)
            + model_bytes(cfg, args.optim_bits, muon is not None))


def summary(cfg, args, train_bytes):
    counts = param_counts(cfg)
    tokens = args.batch * cfg.ctx * args.grad_accum
    print(f'model: {sum(counts.values()):,} parameters  ('
          + ', '.join(f'{k} {v:,}' for k, v in counts.items()) + ')')
    print(f'  ctx {cfg.ctx}, dim {cfg.dim}, heads {cfg.heads} (head size {cfg.dim // cfg.heads}), '
          f'layers {cfg.layers}, mlp {cfg.mlp}, init std {cfg.init_std}')
    if args.epochs:
        print(f'data: {args.epochs:g} epochs of non-overlapping {cfg.ctx}-byte windows, '
              'shuffled')
    print(f'train: {args.steps:,} steps x batch {args.batch} x accum {args.grad_accum} x ctx '
          f'{cfg.ctx} = {tokens:,} tokens/step, {tokens * args.steps / 1e6:,.1f}M tokens'
          + (f' ({tokens * args.steps / train_bytes:.2f} epochs)' if train_bytes else ''))
    print(f'  lr {args.lr:g} -> {args.min_lr:g} ({args.schedule}, warmup {args.warmup}), '
          f'betas ({args.beta1}, {args.beta2}), eps {args.eps:g}, weight decay '
          f'{args.weight_decay:g}, grad clip {args.grad_clip:g}')
    opt = 'AdamW'
    if args.optimizer != 'adamw':
        opt = (f'{"MuonClip" if args.qk_clip else "Muon"} on the hidden matrices (momentum '
               f'{args.muon_momentum}, {args.muon_ns_steps} Newton-Schulz steps, scale '
               f'{args.muon_scale:g}), AdamW on the rest')
    if args.qk_clip:
        opt += f', QK-Clip tau {args.qk_clip:g}'
    if args.optim_bits == 8:
        opt += ', 8-bit blockwise states (bitsandbytes)'
    print(f'  optimizer: {opt}')
    print(f'  device memory needed: ~{device_bytes_needed(cfg, args) / 2 ** 20:,.0f} MB')


def train(args):
    # Saved in checkpoints as given (min_lr stays None when derived from lr).
    train_options = {k: v for k, v in vars(args).items()
                     if k not in ('cmd', 'config', 'dry_run', 'resume', 'explicit_model',
                                  'overwrite') and k not in MODEL_KEYS}
    resume = None
    if args.resume:
        resume = Model.read(args.resume)
        cfg = Config(**resume[0]['config'])
        changed = {k: (v, getattr(cfg, k)) for k, v in args.explicit_model.items()
                   if v != getattr(cfg, k)}
        if changed:
            raise SystemExit('--resume keeps the checkpoint model shape; these options differ '
                             '(given, checkpoint): ' + ', '.join(f'{k} {a}/{b}'
                                                                 for k, (a, b) in changed.items()))
    else:
        cfg = model_config(args)
    if args.out is None:
        args.out = args.resume or DEFAULT_OUT
    replacing_resumed = args.resume and os.path.exists(args.out) and \
        os.path.samefile(args.out, args.resume)
    if os.path.exists(args.out) and not (args.overwrite or replacing_resumed or args.dry_run):
        raise SystemExit(f'{args.out} already exists; pass --overwrite to replace it, '
                         f'--out for a new file, or --resume {args.out} to continue it')
    if args.min_lr is None:
        args.min_lr = args.lr * 0.1
    if args.qk_clip is None:
        args.qk_clip = 100.0 if args.optimizer == 'muonclip' else 0.0
    train_path = os.path.join(args.data, 'train.bin')
    train_bytes = os.path.getsize(train_path) if os.path.exists(train_path) else 0
    if args.epochs:
        if not train_bytes:
            raise SystemExit(f'--epochs needs {train_path}; run vabs_gpt/prepare.py first')
        per_step = args.batch * args.grad_accum
        args.steps = math.ceil(args.epochs * epoch_windows(train_bytes, cfg.ctx) / per_step)
    if resume and resume[0].get('step', 0) >= args.steps:
        raise SystemExit(f'{args.resume} is at step {resume[0].get("step")}; '
                         'pass --steps (or --epochs) beyond it')
    summary(cfg, args, train_bytes)
    if args.dry_run:
        return

    with ComputeContext() as ctx:
        need, have = device_bytes_needed(cfg, args), device_local_bytes(ctx)
        print(f'device: {ctx.device_name} ({have / 2 ** 30:.1f} GB device-local)')
        if have and need > have:
            raise SystemExit(f'needs ~{need / 2 ** 30:.2f} GB of device memory; reduce --batch '
                             '(use --grad-accum for the same tokens per step) or --ctx')
        dev = Device(ctx)
        optim_args = {'optim_bits': args.optim_bits, 'muon': args.optimizer != 'adamw'}
        if resume:
            header, params, state = resume
            note = '' if state is not None else ' (no optimizer state: the moments restart)'
            saved = header.get('train', {}).get('optimizer', 'adamw')
            bits = header.get('optim_bits', 32)
            if state is not None and optimizer_family(saved) != optimizer_family(args.optimizer):
                state = None   # Muon's momentum is not an Adam moment and the other way round
                note = f' (switching from {saved} to {args.optimizer}: the moments restart)'
            elif state is not None and bits != args.optim_bits:
                state = None
                note = (f' (switching from {bits}-bit to {args.optim_bits}-bit optimizer '
                        'states: the moments restart)')
            model = Model(dev, cfg, params, step=header.get('step', 0), **(state or {}),
                          **({} if state else optim_args))
            print(f'resumed {args.resume} at step {model.step}{note}')
        else:
            model = Model(dev, cfg, seed=args.seed, **optim_args)
        runner = Runner(model, args.batch, 'train', accum=args.grad_accum,
                        muon=muon_options(args), qk_clip=args.qk_clip)
        evaluator = Runner(model, args.batch, 'eval')
        if args.epochs:
            train_data = Batches(train_path, args.batch, cfg.ctx, args.seed, 'epoch',
                                 start=model.step * args.batch * args.grad_accum)
        else:
            train_data = Batches(train_path, args.batch, cfg.ctx, args.seed + model.step)
        val_data = Batches(os.path.join(args.data, 'val.bin'), args.batch, cfg.ctx, 1234)
        val_batches = [val_data.get() for _ in range(args.eval_batches)]
        infer = None

        def evaluate():
            return sum(evaluator.run(i, t) for i, t in val_batches) / len(val_batches)

        def save(val):
            model.save(args.out, {'val_loss': val, 'train': {**train_options, 'out': args.out}})

        def show_sample():
            nonlocal infer
            infer = infer or Runner(model, 1, 'infer')
            text = sample(infer, args.sample_prompt.encode(), args.sample_len,
                          rng=random.Random(model.step))
            print('--- sample\n' + args.sample_prompt + text + '\n---', flush=True)

        tokens = args.batch * cfg.ctx * args.grad_accum
        start = model.step
        print(f'step {start}: val loss {evaluate():.4f}', flush=True)
        t0, losses, norms = time.perf_counter(), [], []

        def record(result, step):
            """Keep the (loss, grad norm) of a finished step; they arrive one step late."""
            if result is None:
                return
            if not math.isfinite(result[0]):
                raise SystemExit(f'step {step}: loss is {result[0]}; lower --lr or set '
                                 '--grad-clip')
            losses.append(result[0])
            norms.append(result[1])

        try:
            while model.step < args.steps:
                lr = lr_at(model.step, args)
                # Queues this step and returns the one before, so the GPU stays busy
                # while the next batch is prepared.
                record(runner.submit_step(
                    [train_data.get() for _ in range(args.grad_accum)], lr, args.beta1,
                    args.beta2, args.eps, args.weight_decay, args.grad_clip), model.step - 1)
                step = model.step
                if (step % args.log_every == 0 or step % args.eval_every == 0
                        or step == args.steps
                        or (args.sample_every and step % args.sample_every == 0)):
                    record(runner.collect(), step)   # everything up to this step is done
                if step % args.log_every == 0:
                    dt = time.perf_counter() - t0
                    done = step - start
                    qk = ''
                    if runner.max_logits:   # this step's, before its clip
                        heads = [x for layer in runner.max_logits for x in layer]
                        qk = (f'  max logit {max(heads):.1f} ('
                              f'{sum(x > args.qk_clip for x in heads)} heads clipped)')
                    print(f'step {step:6d}  loss {sum(losses) / len(losses):.4f}  '
                          f'grad norm {sum(norms) / len(norms):.3f}  lr {lr:.2e}{qk}  '
                          f'{dt / done * 1000:.0f} ms/step  {tokens * done / dt:,.0f} tok/s',
                          flush=True)
                    losses, norms = [], []
                if step % args.eval_every == 0 or step == args.steps:
                    val = evaluate()
                    print(f'step {step:6d}  val loss {val:.4f} ({val / math.log(2):.3f} bits/byte)',
                          flush=True)
                    save(val)
                if args.sample_every and step % args.sample_every == 0:
                    show_sample()
        except KeyboardInterrupt:
            runner.collect()
            val = evaluate()
            save(val)
            print(f'\ninterrupted at step {model.step}; saved {args.out} (val loss {val:.4f}); '
                  f'continue with --resume {args.out}')
            return
        print(f'saved {args.out}  ({time.perf_counter() - t0:.0f} s)')
        if not (args.sample_every and model.step % args.sample_every == 0):
            show_sample()


def generate(args):
    with ComputeContext() as ctx:
        dev = Device(ctx)
        model, header = Model.load(dev, args.checkpoint)
        print(f'{model.n_params:,} parameters, step {header.get("step")}, '
              f'val loss {header.get("val_loss", float("nan")):.4f}')
        infer = Runner(model, 1, 'infer')
        rng = random.Random(args.seed)
        for _ in range(args.num):
            text = sample(infer, args.prompt.encode(), args.max_new, args.temperature,
                          args.top_k, rng)
            print('---\n' + args.prompt + text)


def evaluate_checkpoint(args):
    """Mean loss over consecutive non-overlapping windows from the start of val.bin."""
    with ComputeContext() as ctx:
        model, header = Model.load(Device(ctx), args.checkpoint)
        T = model.cfg.ctx
        data = open_bytes(os.path.join(args.data, args.split + '.bin'), T)
        windows = (len(data) - 1) // T
        if args.bytes:
            windows = min(windows, args.bytes // T)
        batches = windows // args.batch
        if batches < 1:
            raise SystemExit('fewer windows than one --batch; raise --bytes or lower --batch')
        runner = Runner(model, args.batch, 'eval')
        span = args.batch * T
        t0, total = time.perf_counter(), 0.0
        for b in range(batches):
            chunk = data[b * span:(b + 1) * span + 1]
            total += runner.run(chunk[:-1], chunk[1:])
        loss = total / batches
        print(f'{args.checkpoint}: step {header.get("step")}, {args.split} loss {loss:.4f} '
              f'({loss / math.log(2):.4f} bits/byte) over {batches * span:,} bytes '
              f'({time.perf_counter() - t0:.0f} s)')
        if args.json:
            print(json.dumps({'loss': loss, 'bytes': batches * span, 'step': header.get('step')}))


def info(args):
    header, params, state = Model.read(args.checkpoint)
    cfg = Config(**header['config'])
    kind = 'no' if state is None else f'{header.get("optim_bits", 32)}-bit'
    print(f'{args.checkpoint}: {len(params):,} parameters, step {header.get("step")}, '
          f'val loss {header.get("val_loss", float("nan")):.4f}, optimizer state {kind}')
    print(json.dumps({'config': asdict(cfg), 'train': header.get('train', {})}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m vabs_gpt', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    fmt = argparse.ArgumentDefaultsHelpFormatter
    t = sub.add_parser('train', help='train on data/train.bin, evaluate on data/val.bin',
                       formatter_class=fmt)
    t.add_argument('--config', help='JSON file of train options (flags override it)')
    t.add_argument('--dry-run', action='store_true', help='print sizes and exit')

    g = t.add_argument_group('model')
    g.add_argument('--ctx', type=int, default=256, help='context length in bytes')
    g.add_argument('--dim', '--embed', type=int, default=64, help='embedding size / model width')
    g.add_argument('--heads', type=int, default=4,
                   help='attention heads (head size dim/heads must be a multiple of 4)')
    g.add_argument('--layers', '--depth', type=int, default=4, help='transformer blocks')
    g.add_argument('--mlp', type=int, default=0, help='MLP hidden width (0: 4 * dim)')
    g.add_argument('--init-std', type=float, default=0.02, help='weight init std')

    g = t.add_argument_group('optimization')
    g.add_argument('--steps', type=int, default=10000,
                   help='optimizer steps = number of batches (with --resume: the final step)')
    g.add_argument('--epochs', type=float, default=0,
                   help='if set: passes over train.bin in shuffled non-overlapping windows; '
                        'sets --steps (with --resume: the final epoch count)')
    g.add_argument('--batch', type=int, default=32, help='sequences per micro-batch')
    g.add_argument('--grad-accum', type=int, default=1,
                   help='micro-batches per optimizer step (effective batch = batch * this)')
    g.add_argument('--lr', type=float, default=3e-3, help='peak learning rate')
    g.add_argument('--min-lr', type=float, default=None, help='final learning rate (default lr/10)')
    g.add_argument('--schedule', choices=('cosine', 'linear', 'constant'), default='cosine')
    g.add_argument('--warmup', type=int, default=200, help='linear warmup steps')
    g.add_argument('--beta1', type=float, default=0.9)
    g.add_argument('--beta2', type=float, default=0.95)
    g.add_argument('--eps', type=float, default=1e-8)
    g.add_argument('--weight-decay', type=float, default=0.0,
                   help='AdamW decay on matrices and embeddings (not biases/norms)')
    g.add_argument('--grad-clip', type=float, default=0.0, help='max global grad norm (0: off)')
    g.add_argument('--optimizer', choices=('adamw', 'muon', 'muonclip'), default='adamw',
                   help='muon: Muon on the hidden matrices (q, k, v, attention output, MLP), '
                        'AdamW with the same lr and decay on the rest; muonclip: muon plus '
                        'QK-Clip')
    g.add_argument('--optim-bits', type=int, choices=(32, 8), default=32,
                   help='8: keep the AdamW moments and Muon momentum as 8-bit blockwise '
                        'states (bitsandbytes), a quarter of the float32 memory')
    g.add_argument('--muon-momentum', type=float, default=0.95, help='Muon Nesterov momentum')
    g.add_argument('--muon-ns-steps', type=int, default=5, help='Newton-Schulz iterations')
    g.add_argument('--muon-scale', type=float, default=0.2,
                   help='Muon update = scale * sqrt(max(rows, cols)) * orthogonalized momentum, '
                        'which matches the update RMS of AdamW')
    g.add_argument('--qk-clip', type=float, default=None,
                   help='QK-Clip tau: after each step, rescale the q and k weights of every head '
                        'whose largest attention logit exceeded it (default: 100 with muonclip, '
                        'else off; 0: off)')
    g.add_argument('--seed', type=int, default=0, help='init and data-order seed')

    g = t.add_argument_group('data, logging and checkpoints')
    g.add_argument('--data', default=os.path.join(HERE, 'data'),
                   help='directory with train.bin and val.bin')
    g.add_argument('--out', default=None,
                   help='checkpoint path (default: the --resume file, else vabs_gpt/model.bin)')
    g.add_argument('--overwrite', action='store_true',
                   help='allow replacing an existing checkpoint that is not being resumed')
    g.add_argument('--resume', help='continue from a checkpoint (its model shape is kept)')
    g.add_argument('--log-every', type=int, default=50)
    g.add_argument('--eval-every', type=int, default=500, help='evaluate and save every N steps')
    g.add_argument('--eval-batches', type=int, default=8)
    g.add_argument('--sample-every', type=int, default=0, help='print a sample every N steps')
    g.add_argument('--sample-len', type=int, default=300)
    g.add_argument('--sample-prompt', default='Once upon a time')

    s = sub.add_parser('sample', help='generate stories from a checkpoint', formatter_class=fmt)
    s.add_argument('--checkpoint', default=DEFAULT_OUT)
    s.add_argument('--prompt', default='Once upon a time')
    s.add_argument('--num', type=int, default=3)
    s.add_argument('--max-new', type=int, default=500)
    s.add_argument('--temperature', type=float, default=0.8, help='0: greedy')
    s.add_argument('--top-k', type=int, default=40, help='0: no truncation')
    s.add_argument('--seed', type=int, default=None)

    e = sub.add_parser('eval', help='loss of a checkpoint on a fixed prefix of val.bin',
                       formatter_class=fmt)
    e.add_argument('checkpoint', nargs='?', default=DEFAULT_OUT)
    e.add_argument('--data', default=os.path.join(HERE, 'data'))
    e.add_argument('--split', choices=('val', 'train'), default='val')
    e.add_argument('--bytes', type=int, default=0, help='bytes to evaluate (0: the whole file)')
    e.add_argument('--batch', type=int, default=32)
    e.add_argument('--json', action='store_true', help='also print the result as JSON')

    i = sub.add_parser('info', help="show a checkpoint's model shape and training options")
    i.add_argument('checkpoint', nargs='?', default=DEFAULT_OUT)
    return parser, t


def parse_args(argv=None):
    parser, train_parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd != 'train':
        return args
    # Precedence: command line > --config file > the resumed checkpoint's options > defaults.
    known = {a.dest for a in train_parser._actions} - {'help', 'config'}
    options, saved = {}, {}
    if args.config:
        with open(args.config, encoding='utf-8') as f:
            options = {k.replace('-', '_'): v for k, v in json.load(f).items()}
        unknown = sorted(set(options) - known)
        if unknown:
            raise SystemExit(f'{args.config}: unknown options {unknown}')
    if args.resume:
        try:
            header = Model.read_header(args.resume)
        except (OSError, ValueError) as e:
            raise SystemExit(f'--resume: {e}')
        saved = {k: v for k, v in header.get('train', {}).items()
                 if k in known and k not in ('out', 'steps')}
    if options or saved:
        train_parser.set_defaults(**{**saved, **options})
        args = parser.parse_args(argv)
    # Model options set by flag or config file, even when equal to the default
    # (a resumed checkpoint must agree with them): reparse with None defaults.
    train_parser.set_defaults(**{k: options.get(k) for k in MODEL_KEYS})
    probe = parser.parse_args(argv)
    args.explicit_model = {k: getattr(probe, k) for k in MODEL_KEYS
                           if getattr(probe, k) is not None}
    for name in ('steps', 'batch', 'grad_accum', 'log_every', 'eval_every', 'eval_batches'):
        if getattr(args, name) < 1:
            raise SystemExit(f'--{name.replace("_", "-")} must be at least 1')
    if args.epochs < 0:
        raise SystemExit('--epochs must be non-negative')
    if args.lr <= 0 or args.warmup < 0 or args.weight_decay < 0 or args.grad_clip < 0:
        raise SystemExit('--lr must be positive; --warmup, --weight-decay and --grad-clip '
                         'non-negative')
    if (args.qk_clip or 0) < 0 or args.muon_ns_steps < 1 or not 0 <= args.muon_momentum < 1 \
            or args.muon_scale <= 0:
        raise SystemExit('--qk-clip must be non-negative, --muon-ns-steps at least 1, '
                         '--muon-momentum in [0, 1) and --muon-scale positive')
    if not args.resume:
        try:
            model_config(args)
        except ValueError as e:
            raise SystemExit(f'invalid model shape: {e}')
    return args


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = parse_args(argv)
    {'train': train, 'sample': generate, 'eval': evaluate_checkpoint,
     'info': info}[args.cmd](args)


if __name__ == '__main__':
    main()

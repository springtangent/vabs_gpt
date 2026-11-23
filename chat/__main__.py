"""Train, evaluate and talk to the small Python chatbot (vabs compute only).

    python -m vabs_gpt.chat pretrain --tokens 50e6 --out vabs_gpt\\chat\\pilot.bin
    python -m vabs_gpt.chat finetune --init vabs_gpt\\chat\\base.bin --out vabs_gpt\\chat\\chat.bin
    python -m vabs_gpt.chat eval vabs_gpt\\chat\\base.bin
    python -m vabs_gpt.chat sample vabs_gpt\\chat\\base.bin --prompt "def add(a, b):"
    python -m vabs_gpt.chat chat vabs_gpt\\chat\\chat.bin

Run from the workspace root with the vabs environment (``.venv``). Every command
has ``--help``. See ``vabs_gpt/chat/README.md``.
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time

from vabs_gpt.gpu import ComputeContext
from vabs_gpt.model import Muon

from . import data as D
from .generate import Generator, code_checks
from .model import (
    ChatConfig,
    ChatModel,
    ChatRunner,
    Device,
    activation_bytes,
    add_precision_arg,
    model_bytes,
    optimizer_bytes,
    param_count,
)
from .tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PROMPTS = ['def is_prime(n):\n', 'The water cycle is']


# -- options --------------------------------------------------------------------------

def add_model_args(p):
    g = p.add_argument_group('model shape (new runs only)')
    d = ChatConfig()
    for name, kind in (('vocab', int), ('ctx', int), ('dim', int), ('heads', int),
                       ('kv-heads', int), ('layers', int), ('mlp', int), ('rope-theta', float),
                       ('init-std', float)):
        g.add_argument('--' + name, type=kind, default=None,
                       help=f'default {getattr(d, name.replace("-", "_"))}')


def add_train_args(p, lr, warmup, batch, accum, schedule='cosine'):
    g = p.add_argument_group('optimization')
    g.add_argument('--steps', type=int, default=0, help='optimizer steps (overrides --tokens/--epochs)')
    g.add_argument('--batch', type=int, default=batch, help='sequences per micro-batch')
    g.add_argument('--grad-accum', type=int, default=accum, help='micro-batches per step')
    g.add_argument('--lr', type=float, default=lr)
    g.add_argument('--min-lr', type=float, default=None, help='final lr (default lr / 10)')
    g.add_argument('--schedule', choices=('cosine', 'linear', 'constant'), default=schedule)
    g.add_argument('--warmup', type=int, default=warmup)
    g.add_argument('--beta1', type=float, default=0.9)
    g.add_argument('--beta2', type=float, default=0.95)
    g.add_argument('--eps', type=float, default=1e-8)
    g.add_argument('--weight-decay', type=float, default=0.1)
    g.add_argument('--grad-clip', type=float, default=1.0)
    g.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw')
    g.add_argument('--optim-bits', type=int, choices=(8, 32), default=8,
                   help='8: blockwise 8-bit states (vabs_gpt_bitsandbytes)')
    g.add_argument('--muon-momentum', type=float, default=0.95)
    g.add_argument('--muon-ns-steps', type=int, default=5)
    g.add_argument('--muon-scale', type=float, default=0.2)
    g.add_argument('--seed', type=int, default=0)
    g = p.add_argument_group('output')
    g.add_argument('--out', required=True, help='checkpoint path (rewritten at every evaluation)')
    g.add_argument('--overwrite', action='store_true')
    g.add_argument('--resume', action='store_true', help='continue the run saved in --out')
    g.add_argument('--log-every', type=int, default=50)
    g.add_argument('--eval-every', type=int, default=500)
    g.add_argument('--save-every', type=int, default=0,
                   help='also keep <out>_step<N>.bin (parameters only) every N steps')
    g.add_argument('--sample-every', type=int, default=1, help='evaluations between samples')
    g.add_argument('--code-every', type=int, default=4,
                   help='evaluations between code checks (0: never)')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python -m vabs_gpt.chat', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('pretrain', help='next-token pretraining on the data mix')
    p.add_argument('--data', default=D.DATA, help='directory written by prepare')
    p.add_argument('--mix', default='stack=0.4,fineweb=0.5,cosmo=0.1',
                   help='source weights, in tokens')
    p.add_argument('--tokens', type=float, default=500e6, help='training tokens (sets --steps)')
    p.add_argument('--eval-tokens', type=int, default=262144, help='per validation source')
    add_model_args(p)
    add_train_args(p, lr=2e-3, warmup=300, batch=8, accum=4)

    p = sub.add_parser('finetune', help='fine-tune on chats, loss on assistant tokens only')
    p.add_argument('--init', help='pretrained checkpoint (new runs)')
    p.add_argument('--chats', default=os.path.join(D.DATA, 'chats_train.jsonl'))
    p.add_argument('--val-chats', default=os.path.join(D.DATA, 'chats_val.jsonl'))
    p.add_argument('--epochs', type=float, default=3.0)
    p.add_argument('--data', default=D.DATA, help='validation data of pretraining (also scored)')
    p.add_argument('--eval-tokens', type=int, default=131072)
    p.add_argument('--pretrain-rows', type=int, default=0,
                   help='rows of every micro-batch filled with pretraining windows from '
                        '--data (--mix), every token scored: replay against forgetting')
    p.add_argument('--mix', default='stack=0.4,fineweb=0.5,cosmo=0.1',
                   help='pretraining mix of the replay rows')
    add_train_args(p, lr=2e-4, warmup=30, batch=8, accum=2)
    p.set_defaults(log_every=10, eval_every=50, code_every=0, weight_decay=0.0)

    p = sub.add_parser('eval', help='validation loss per source, and code checks')
    p.add_argument('checkpoint', nargs='+', help='one or more checkpoints, scored on the same '
                   'data and prompts')
    p.add_argument('--data', default=D.DATA)
    p.add_argument('--tokens', type=int, default=0, help='per source (0: all)')
    p.add_argument('--chats', default='', help='also score these chats (assistant tokens)')
    p.add_argument('--samples', type=int, default=4, help='completions per code prompt')
    p.add_argument('--show', action='store_true', help='print the code completions')
    p.add_argument('--batch', type=int, default=8)

    p = sub.add_parser('sample', help='complete a prompt')
    p.add_argument('checkpoint')
    p.add_argument('--prompt', default='def fibonacci(n):\n')
    p.add_argument('--num', type=int, default=1)
    add_sampling_args(p, 200)

    p = sub.add_parser('chat', help='talk to a fine-tuned checkpoint')
    p.add_argument('checkpoint')
    p.add_argument('--prompt', action='append', default=[],
                   help='user messages to send in turn, then exit (repeatable)')
    add_sampling_args(p, 400)

    p = sub.add_parser('info', help='print a checkpoint header')
    p.add_argument('checkpoint')

    for name, p in sub.choices.items():
        if name != 'info':
            add_precision_arg(p)
    args = parser.parse_args(argv)
    for name in ('prompt',):
        v = getattr(args, name, None)
        if isinstance(v, str):
            setattr(args, name, v.encode().decode('unicode_escape'))
    if getattr(args, 'prompt', None) and isinstance(args.prompt, list):
        args.prompt = [s.encode().decode('unicode_escape') for s in args.prompt]
    return args


def add_sampling_args(p, max_new):
    p.add_argument('--max-new', type=int, default=max_new)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--seed', type=int, default=None)


# -- helpers ----------------------------------------------------------------------------

def lr_at(step, steps, args):
    min_lr = args.lr / 10 if args.min_lr is None else args.min_lr
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if args.schedule == 'constant':
        return args.lr
    frac = min(1.0, (step - args.warmup) / max(1, steps - args.warmup))
    if args.schedule == 'linear':
        return args.lr + (min_lr - args.lr) * frac
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * frac))


def muon_of(args):
    if args.optimizer != 'muon':
        return None
    return Muon(momentum=args.muon_momentum, ns_steps=args.muon_ns_steps, scale=args.muon_scale)


def load_tokenizer(header):
    if not header.get('tokenizer'):
        raise SystemExit('checkpoint has no tokenizer')
    return Tokenizer(header['tokenizer'], header['config']['vocab'])


def open_checkpoint(dev, path, optimizer=False):
    header, params, state = ChatModel.read(path)
    cfg = ChatConfig(**header['config'])
    kw = (state or {}) if optimizer else {}
    model = ChatModel(dev, cfg, params, step=header.get('step', 0), tokenizer=header['tokenizer'],
                      **kw)
    return model, header


def fmt_losses(losses):
    return ' '.join(f'{k} {v:.3f}' for k, v in losses.items())


def evaluate(dev, model, sets, B, runner=None):
    """``{name: mean loss}`` over lists of ``(inp, tgt)`` windows."""
    own = runner is None
    runner = runner or ChatRunner(model, B, 'eval')
    B = runner.B
    out = {}
    T = model.cfg.ctx
    for name, windows in sets.items():
        total = count = 0
        for i in range(0, len(windows), B):
            chunk = windows[i:i + B]
            inp = D.array.array('I')
            tgt = D.array.array('I')
            for a, b in chunk:
                inp.extend(a)
                tgt.extend(b)
            pad = B - len(chunk)
            inp.extend([0] * (pad * T))
            tgt.extend([D.IGNORE] * (pad * T))
            s, c = runner.run(inp, tgt)
            total += s
            count += c
        out[name] = total / max(count, 1)
    if own:
        runner.release()
    return out


def val_sets(data_dir, T, tokens, sources=None):
    sets = {}
    for name in sorted(os.listdir(data_dir)):
        if name.startswith('val_') and name.endswith('.bin'):
            src = name[4:-4]
            if sources is None or src in sources:
                sets[src] = D.val_windows(os.path.join(data_dir, name), T, tokens)
    return sets


# -- training ---------------------------------------------------------------------------

class Run:
    """The shared training loop of ``pretrain`` and ``finetune``."""

    def __init__(self, args, stage):
        self.args, self.stage = args, stage
        self.log_path = os.path.splitext(args.out)[0] + '.jsonl'

    def log(self, **record):
        with open(self.log_path, 'a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(record) + '\n')

    def loop(self, dev, model, steps, next_batch, eval_sets, extra, state_fn, tok):
        args = self.args
        muon = muon_of(args)
        runner = ChatRunner(model, args.batch, 'train', accum=args.grad_accum, muon=muon)
        eval_runner = ChatRunner(model, args.batch, 'eval')
        gen = Generator(model, tok)
        stop = {'now': False}

        def on_sigint(*_):
            stop['now'] = True
            print('\ninterrupted: saving after this step', flush=True)

        old = signal.signal(signal.SIGINT, on_sigint)
        T = model.cfg.ctx
        tokens_per_step = args.batch * args.grad_accum * T
        t0 = time.time()
        t_log, tok_log = time.time(), 0
        evals = extra.get('evals', 0)
        best = extra.get('best')
        loss_acc = norm_acc = 0.0
        n_acc = 0
        try:
            while model.step < steps and not stop['now']:
                step = model.step
                lr = lr_at(step, steps, args)
                batches = [next_batch() for _ in range(args.grad_accum)]
                res = runner.submit_step(batches, lr, beta1=args.beta1, beta2=args.beta2,
                                         eps=args.eps, weight_decay=args.weight_decay,
                                         clip=args.grad_clip)
                tok_log += tokens_per_step
                if res:
                    loss_acc += res[0]
                    norm_acc += res[1]
                    n_acc += 1
                done = model.step
                last = done >= steps
                if done % args.log_every == 0 or last:
                    res = runner.collect()
                    loss_acc += res[0]
                    norm_acc += res[1]
                    n_acc += 1
                    dt = time.time() - t_log
                    loss, norm = loss_acc / n_acc, norm_acc / n_acc
                    print(f'step {done:6d}/{steps} | loss {loss:.4f} | norm {norm:.3f} | '
                          f'lr {lr:.2e} | {tok_log / dt / 1e3:,.1f}k tok/s | '
                          f'{(time.time() - t0) / 60:,.1f} min', flush=True)
                    self.log(kind='train', step=done, loss=loss, norm=norm, lr=lr,
                             tokens=done * tokens_per_step, time=time.time() - t0)
                    loss_acc = norm_acc = 0.0
                    n_acc = 0
                    t_log, tok_log = time.time(), 0
                if done % args.eval_every == 0 or last:
                    runner.collect()
                    evals += 1
                    losses = evaluate(dev, model, eval_sets, args.batch, eval_runner)
                    print(f'  val: {fmt_losses(losses)}', flush=True)
                    record = dict(kind='val', step=done, tokens=done * tokens_per_step,
                                  **{f'val_{k}': v for k, v in losses.items()})
                    if args.sample_every and evals % args.sample_every == 0:
                        rng = random.Random(done)
                        prompts = extra.get('sample_prompts', SAMPLE_PROMPTS)
                        for prompt in prompts:
                            if isinstance(prompt, str):
                                text = gen.complete(prompt, max_new=96, temperature=0.8, rng=rng)[0]
                                print(f'  > {prompt!r} -> {text!r}', flush=True)
                            else:
                                text = gen.reply(prompt, max_new=160, temperature=0.7, rng=rng)
                                print(f'  > {prompt[-1][1]!r} -> {text!r}', flush=True)
                    if args.code_every and evals % args.code_every == 0:
                        rates, _ = code_checks(gen, samples=1, seed=done)
                        print('  code: ' + ' '.join(f'{k} {v:.0%}' for k, v in rates.items()),
                              flush=True)
                        record.update({f'code_{k}': v for k, v in rates.items()})
                    self.log(**record)
                    extra.update(evals=evals, val=losses)
                    self.save(model, args.out, extra, state_fn)
                    if args.save_every and done % args.save_every == 0 or last:
                        if args.save_every:
                            stem = os.path.splitext(args.out)[0]
                            self.save(model, f'{stem}_step{done}.bin', extra, state_fn,
                                      optimizer=False)
            res = runner.collect()
            if stop['now']:
                self.save(model, args.out, extra, state_fn)
                print(f'saved {args.out} at step {model.step}')
        finally:
            signal.signal(signal.SIGINT, old)
        return model

    def save(self, model, path, extra, state_fn, optimizer=True):
        header = {'stage': self.stage, 'options': {k: v for k, v in vars(self.args).items()
                                                   if k != 'cmd'},
                  'data_state': state_fn(), **{k: v for k, v in extra.items()
                                               if k != 'sample_prompts'}}
        model.save(path, extra=header, optimizer=optimizer)


def check_out(args):
    if args.resume:
        if not os.path.exists(args.out):
            raise SystemExit(f'--resume: {args.out} does not exist')
        return
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f'{args.out} exists; pass --resume to continue it or --overwrite')
    log = os.path.splitext(args.out)[0] + '.jsonl'
    if os.path.exists(log):
        os.remove(log)


def new_config(args, vocab):
    kw = {}
    for name in ('ctx', 'dim', 'heads', 'kv_heads', 'layers', 'mlp', 'rope_theta', 'init_std'):
        v = getattr(args, name)
        if v is not None:
            kw[name] = v
    return ChatConfig(vocab=args.vocab or vocab, **kw)


def report_memory(cfg, args, muon):
    total = (model_bytes(cfg, args.optim_bits, muon) + activation_bytes(cfg, args.batch)
             + sum(optimizer_bytes(cfg, muon, args.optim_bits))
             + activation_bytes(cfg, args.batch, 'eval'))
    print(f'model: {param_count(cfg):,} parameters ({cfg.layers} layers, width {cfg.dim}, '
          f'{cfg.heads} query / {cfg.kv_heads} kv heads, SwiGLU {cfg.mlp}, ctx {cfg.ctx}, '
          f'vocab {cfg.vocab}); about {total / 2 ** 30:.1f} GiB of device memory', flush=True)


def cmd_pretrain(args):
    check_out(args)
    mix = D.parse_mix(args.mix)
    with ComputeContext() as ctx:
        dev = Device(ctx, precision=args.precision)
        if args.resume:
            model, header = open_checkpoint(dev, args.out, optimizer=True)
            if header.get('stage') != 'pretrain':
                raise SystemExit(f'{args.out} is not a pretraining checkpoint')
            data_state, extra = header.get('data_state'), {k: header[k] for k in
                                                           ('evals', 'val') if k in header}
            tok = load_tokenizer(header)
        else:
            tok = Tokenizer.load(os.path.join(args.data, 'tokenizer.json'))
            cfg = new_config(args, tok.vocab_size)
            print('initializing parameters...', flush=True)
            model = ChatModel(dev, cfg, seed=args.seed, optim_bits=args.optim_bits,
                              muon=args.optimizer == 'muon', tokenizer=tok.merges)
            data_state, extra = None, {}
        cfg = model.cfg
        report_memory(cfg, args, args.optimizer == 'muon')
        mixture = D.Mixture(args.data, mix, cfg.ctx, seed=args.seed, state=data_state)
        tokens_per_step = args.batch * args.grad_accum * cfg.ctx
        steps = args.steps or max(1, int(args.tokens // tokens_per_step))
        print(f'{steps:,} steps of {tokens_per_step:,} tokens = {steps * tokens_per_step / 1e6:,.1f}M '
              f'tokens; mix {mix}; windows per source: '
              + ', '.join(f'{s} {mixture.windows(s):,}' for s in mixture.weights), flush=True)
        eval_sets = val_sets(args.data, cfg.ctx, args.eval_tokens)

        def next_batch():
            inp, tgt, _ = mixture.batch(args.batch)
            return inp, tgt

        Run(args, 'pretrain').loop(dev, model, steps, next_batch, eval_sets, extra,
                                   mixture.state, tok)


def cmd_finetune(args):
    check_out(args)
    with ComputeContext() as ctx:
        dev = Device(ctx, precision=args.precision)
        if args.resume:
            model, header = open_checkpoint(dev, args.out, optimizer=True)
            extra = {k: header[k] for k in ('evals', 'val', 'init') if k in header}
            data_state = header.get('data_state') or {}
        else:
            if not args.init:
                raise SystemExit('finetune needs --init <pretrained checkpoint>')
            header, params, _ = ChatModel.read(args.init)
            cfg = ChatConfig(**header['config'])
            model = ChatModel(dev, cfg, params, optim_bits=args.optim_bits,
                              muon=args.optimizer == 'muon', tokenizer=header['tokenizer'])
            extra = {'init': os.path.basename(args.init),
                     'init_step': header.get('step'), 'init_val': header.get('val')}
            data_state = {}
        tok = load_tokenizer(header)
        cfg = model.cfg
        report_memory(cfg, args, args.optimizer == 'muon')
        train = [D.encode_chat(tok, turns) for _, turns in D.read_chats(args.chats)]
        val = [D.encode_chat(tok, turns) for _, turns in D.read_chats(args.val_chats)]
        windows = D.pack_chats(train, cfg.ctx, seed=args.seed)
        val_windows = D.pack_chats(val, cfg.ctx, seed=0)
        replay = args.pretrain_rows
        if not 0 <= replay < args.batch:
            raise SystemExit('--pretrain-rows must be at least 0 and less than --batch')
        mixture = None
        if replay:
            mixture = D.Mixture(args.data, D.parse_mix(args.mix), cfg.ctx, seed=args.seed + 1,
                                state=data_state.get('mix'))
        per_step = (args.batch - replay) * args.grad_accum   # chat windows per step
        steps = args.steps or max(1, int(math.ceil(len(windows) * args.epochs / per_step)))
        scored = sum(sum(1 for t in tgt if t != D.IGNORE) for _, tgt in windows)
        print(f'{len(train):,} training chats in {len(windows):,} windows '
              f'({scored:,} assistant tokens), {len(val):,} validation chats; {steps:,} steps'
              + (f'; {replay} of {args.batch} rows per micro-batch replay pretraining data '
                 f'({args.mix})' if replay else ''), flush=True)
        eval_sets = {'chats': val_windows}
        if args.eval_tokens:
            eval_sets.update(val_sets(args.data, cfg.ctx, args.eval_tokens))
        cursor = {'epoch': data_state.get('epoch', 0), 'i': data_state.get('i', 0)}
        order = {}

        def perm():
            e = cursor['epoch']
            if e not in order:
                idx = list(range(len(windows)))
                random.Random(f'{args.seed}:{e}').shuffle(idx)
                order.clear()
                order[e] = idx
            return order[e]

        def next_batch():
            inp, tgt = D.array.array('I'), D.array.array('I')
            if mixture:
                inp, tgt, _ = mixture.batch(replay)
            for _ in range(args.batch - replay):
                p = perm()
                if cursor['i'] >= len(p):
                    cursor['epoch'] += 1
                    cursor['i'] = 0
                    p = perm()
                a, b = windows[p[cursor['i']]]
                cursor['i'] += 1
                inp.extend(a)
                tgt.extend(b)
            return inp, tgt

        extra['sample_prompts'] = [[('user', 'Write a Python function that returns the '
                                     'largest of three numbers.')],
                                   [('user', 'What does this error mean?\n\n```\nTypeError: '
                                     "can only concatenate str (not \"int\") to str\n```")],
                                   [('user', 'Give me three tips for staying focused.')]]
        Run(args, 'finetune').loop(dev, model, steps, next_batch, eval_sets, extra,
                                   lambda: dict(cursor, **({'mix': mixture.state()}
                                                           if mixture else {})), tok)


def cmd_eval(args):
    rows = []
    with ComputeContext() as ctx:
        dev = Device(ctx, precision=args.precision)
        for path in args.checkpoint:
            model, header = open_checkpoint(dev, path)
            tok = load_tokenizer(header)
            sets = val_sets(args.data, model.cfg.ctx, args.tokens)
            if args.chats:
                sets['chats'] = D.pack_chats([D.encode_chat(tok, t) for _, t in
                                              D.read_chats(args.chats)], model.cfg.ctx)
            t0 = time.time()
            losses = evaluate(dev, model, sets, args.batch)
            print(f'== {path} (step {header.get("step")})')
            for k, v in losses.items():
                print(f'{k:10s} loss {v:.4f}  ({v / math.log(2):.3f} bits/token, '
                      f'{len(sets[k]):,} windows)')
            print(f'({time.time() - t0:.0f} s)')
            rates = {}
            if args.samples:
                gen = Generator(model, tok)
                rates, texts = code_checks(gen, samples=args.samples)
                print('code completions: ' + ' '.join(f'{k} {v:.0%}' for k, v in rates.items()))
                if args.show:
                    for p, t in texts:
                        print('-' * 60 + '\n' + p + t)
            rows.append((os.path.basename(path), losses, rates))
            if args.samples:
                gen.release()
            model.release()
    if len(rows) > 1:
        keys = list(rows[0][1]) + list(rows[0][2])
        print('\n' + 'checkpoint'.ljust(28) + ''.join(k[:9].rjust(10) for k in keys))
        for name, losses, rates in rows:
            vals = [f'{losses[k]:.4f}' for k in losses] + [f'{rates[k]:.0%}' for k in rates]
            print(name[:27].ljust(28) + ''.join(v.rjust(10) for v in vals))


def cmd_sample(args):
    with ComputeContext() as ctx:
        dev = Device(ctx, precision=args.precision)
        model, header = open_checkpoint(dev, args.checkpoint)
        gen = Generator(model, load_tokenizer(header))
        rng = random.Random(args.seed)
        for text in gen.complete(args.prompt, n=args.num, max_new=args.max_new,
                                 temperature=args.temperature, top_k=args.top_k, rng=rng):
            print('-' * 60)
            print(args.prompt + text)


def cmd_chat(args):
    with ComputeContext() as ctx:
        dev = Device(ctx, precision=args.precision)
        model, header = open_checkpoint(dev, args.checkpoint)
        gen = Generator(model, load_tokenizer(header))
        rng = random.Random(args.seed)
        turns = []
        scripted = list(args.prompt)
        print('(empty line to send, /reset to start over, /quit to leave)' if not scripted else '')
        while True:
            if scripted:
                msg = scripted.pop(0)
                print(f'you> {msg}')
            elif args.prompt:
                break
            else:
                lines = []
                try:
                    line = input('you> ')
                    while line != '':
                        lines.append(line)
                        line = input('... ')
                except EOFError:
                    break
                msg = '\n'.join(lines)
                if msg.strip() == '/quit':
                    break
                if msg.strip() == '/reset':
                    turns = []
                    continue
                if not msg.strip():
                    continue
            turns.append(('user', msg))
            reply = gen.reply(turns, max_new=args.max_new, temperature=args.temperature,
                              top_k=args.top_k, rng=rng)
            turns.append(('assistant', reply))
            print(f'bot> {reply}\n', flush=True)


def cmd_info(args):
    h = ChatModel.read_header(args.checkpoint)
    h.pop('tokenizer', None)
    print(json.dumps(h, indent=1))


def main(argv=None):
    args = parse_args(argv)
    {'pretrain': cmd_pretrain, 'finetune': cmd_finetune, 'eval': cmd_eval, 'sample': cmd_sample,
     'chat': cmd_chat, 'info': cmd_info}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())

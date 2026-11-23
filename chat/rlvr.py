"""Reinforcement learning with verifiable rewards on Python function bodies.

    .venv\\Scripts\\python.exe -m vabs_gpt.chat.rlvr eval chat.bin --split test
    .venv\\Scripts\\python.exe -m vabs_gpt.chat.rlvr probe chat.bin --split train --samples 8
    .venv\\Scripts\\python.exe -m vabs_gpt.chat.rlvr train --init chat.bin --out rl.bin

The model sees a stub (``rl_data``: imports, signature, docstring) and writes
the body; the body ends where the model dedents back to the top level (or at
``<|endoftext|>``). The reward is 1 when every hidden test passes in the
sandbox (``verify``), else 0.

``train`` is GRPO without a reference model: per step it samples ``--group``
completions for each of ``--prompts`` training tasks at temperature 1 from the
full softmax (``decode.Decoder``: the samples come from the policy being
trained), rewards them, and gives every token of a completion the group-relative
advantage ``(r - mean(r)) / std(r)`` of its group (``--no-std``: no division).
Groups where every sample earned the same reward carry no signal and are left
out. One optimizer step per batch of samples, so the update is exactly on-policy
(no importance ratio to clip): the loss is ``sum_tokens A (-log p(token)) /
tokens``, the cross entropy of the samples weighted by their advantages.

``eval`` reports pass@1 with greedy decoding (and with ``--samples k`` the
unbiased pass@k estimate at ``--temperature``); ``probe`` samples k completions
per task and shows how many tasks are sometimes solved, the ones that give GRPO
a signal.
"""
import argparse
import array
import json
import math
import os
import random
import signal
import sys
import time

from .data import DATA

SPLITS = {s: os.path.join(DATA, f'rl_{s}.jsonl')
          for s in ('train', 'dev', 'test', 'plus', 'acecode', 'acecode_eval')}


def read_tasks(split, sources=None):
    path = SPLITS.get(split, split)
    with open(path, encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]
    if sources:
        tasks = [t for t in tasks if t['source'] in sources]
    return tasks


class Sampler:
    """Completes stubs with a ``Decoder`` and scores them with the sandbox."""

    def __init__(self, model, tok, batch=128, T=512, max_new=256, workers=16):
        from .decode import Decoder
        from .verify import Pool
        self.tok, self.max_new, self.T = tok, max_new, T
        self.dec = Decoder(model, batch, T)
        self.pool = Pool(workers)
        self.B = batch

    def release(self):
        self.dec.release()
        self.pool.close()

    def prompt(self, task):
        from .tokenizer import EOT
        return [EOT] + self.tok.encode(task['stub'])

    def complete(self, tasks, temperature, rng):
        """``[(prompt ids, new ids, code)]`` for each task (the stop id is kept)."""
        from .generate import function_end
        tok = self.tok
        out = [None] * len(tasks)
        fits = []
        for i, t in enumerate(tasks):
            p = self.prompt(t)
            if len(p) > self.T - 64:   # no room to answer (one MBPP test task): a failure
                out[i] = (p, [], t['stub'])
            else:
                fits.append(i)
        for j in range(0, len(fits), self.B):
            idx = fits[j:j + self.B]
            batch = [tasks[i] for i in idx]
            prompts = [self.prompt(t) for t in batch]
            limit = min(self.max_new, self.T - max(len(p) for p in prompts))

            def ended(b, ids, batch=batch):
                if '\n' not in tok.decode(ids[-2:]):
                    return False
                return function_end(batch[b]['stub'] + tok.decode(ids)) is not None

            news = self.dec.generate(prompts, max_new=limit, temperature=temperature, rng=rng,
                                     stop_fn=ended, keep_stop=True)
            for i, t, p, n in zip(idx, batch, prompts, news):
                text = t['stub'] + tok.decode([x for x in n if x < tok.vocab_size
                                               and x not in self.stops])
                end = function_end(text)
                out[i] = (p, n, text[:end] if end else text)
        return out

    @property
    def stops(self):
        from .tokenizer import END, EOT
        return (END, EOT)

    def score(self, tasks, codes, full=True, timeout=None):
        from . import verify as V
        key = 'test_full' if full else 'test'
        return self.pool.map([(c, t[key]) for t, c in zip(tasks, codes)],
                             timeout=timeout or V.TIMEOUT)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def open_model(ctx, path, optimizer=False, precision='f32'):
    from .__main__ import load_tokenizer, open_checkpoint
    from .model import Device
    dev = Device(ctx, precision=precision)
    model, header = open_checkpoint(dev, path, optimizer=optimizer)
    return dev, model, header, load_tokenizer(header)


def evaluate(sampler, tasks, rng=None, samples=0, temperature=0.8, full=True):
    """Greedy pass@1 (and sampled pass@k) of ``tasks``; per-task results."""
    rng = rng or random.Random(0)
    res = sampler.complete(tasks, 0.0, rng)
    timeout = 30 if any(t['source'] == 'mbpp+' for t in tasks) else None
    ok = sampler.score(tasks, [r[2] for r in res], full, timeout)
    report = {'pass@1': sum(ok) / max(1, len(tasks)), 'passed': sum(ok), 'tasks': len(tasks)}
    per = [dict(id=t['id'], greedy=bool(o), code=r[2]) for t, o, r in zip(tasks, ok, res)]
    if samples:
        rep = [t for t in tasks for _ in range(samples)]
        sres = sampler.complete(rep, temperature, rng)
        sok = sampler.score(rep, [r[2] for r in sres], full, timeout)
        counts = [sum(sok[i * samples:(i + 1) * samples]) for i in range(len(tasks))]
        for k in sorted({1, min(5, samples), samples}):
            report[f'pass@{k} (T={temperature})'] = (
                sum(pass_at_k(samples, c, k) for c in counts) / len(tasks))
        for p, c in zip(per, counts):
            p['sampled'] = c
    return report, per


def cmd_eval(args):
    from vabs_gpt.gpu import ComputeContext
    with ComputeContext() as ctx:
        for path in args.checkpoints:
            dev, model, header, tok = open_model(ctx, path, precision=args.precision)
            sampler = Sampler(model, tok, args.batch, args.T, args.max_new)
            for split in args.split:
                tasks = read_tasks(split)[:args.limit or None]
                t0 = time.time()
                report, per = evaluate(sampler, tasks, random.Random(args.seed), args.samples,
                                       args.temperature, full=not args.hidden)
                print(f'{os.path.basename(path)} {split}: '
                      + ', '.join(f'{k} {v:.1%}' if isinstance(v, float) else f'{k} {v}'
                                  for k, v in report.items())
                      + f' ({time.time() - t0:.0f} s)', flush=True)
                if args.out:
                    stem = os.path.splitext(os.path.basename(path))[0]
                    with open(f'{args.out}_{stem}_{split}.jsonl', 'w', encoding='utf-8',
                              newline='\n') as f:
                        for p in per:
                            f.write(json.dumps(p) + '\n')
            sampler.release()
            model.release()


def cmd_probe(args):
    from vabs_gpt.gpu import ComputeContext
    with ComputeContext() as ctx:
        dev, model, header, tok = open_model(ctx, args.checkpoint, precision=args.precision)
        sampler = Sampler(model, tok, args.batch, args.T, args.max_new)
        tasks = read_tasks(args.split, args.sources)
        random.Random(1).shuffle(tasks)
        tasks = tasks[:args.limit or None]
        rep = [t for t in tasks for _ in range(args.samples)]
        t0 = time.time()
        res = sampler.complete(rep, args.temperature, random.Random(args.seed))
        t1 = time.time()
        ok = sampler.score(rep, [r[2] for r in res], full=False)
        t2 = time.time()
        k = args.samples
        counts = [sum(ok[i * k:(i + 1) * k]) for i in range(len(tasks))]
        hist = [counts.count(c) for c in range(k + 1)]
        mixed = sum(1 for c in counts if 0 < c < k)
        by = {}
        for t, c in zip(tasks, counts):
            s = by.setdefault(t['source'], [0, 0, 0])
            s[0] += 1
            s[1] += c
            s[2] += 0 < c < k
        new = sum(len(r[1]) for r in res)
        print(f'{len(tasks)} tasks x {k} samples at T={args.temperature}: '
              f'{sum(ok) / len(ok):.1%} of samples pass; passes per task {hist}; '
              f'{mixed} tasks ({mixed / len(tasks):.0%}) sometimes pass')
        for s, (n, c, m) in by.items():
            print(f'  {s}: {n} tasks, {c / (n * k):.1%} of samples pass, {m} mixed')
        print(f'sampling {new:,} tokens in {t1 - t0:.0f} s ({new / (t1 - t0):,.0f} tok/s), '
              f'checking in {t2 - t1:.0f} s')
        if args.out:
            with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
                for t, c in zip(tasks, counts):
                    f.write(json.dumps(dict(id=t['id'], passes=c, samples=k)) + '\n')
        sampler.release()


# -- training -------------------------------------------------------------------------

def advantages(rewards, group, std=True):
    out = []
    for i in range(0, len(rewards), group):
        g = rewards[i:i + group]
        mean = sum(g) / len(g)
        sd = math.sqrt(sum((r - mean) ** 2 for r in g) / len(g))
        if sd == 0:
            out += [0.0] * len(g)
        else:
            out += [(r - mean) / (sd if std else 1.0) for r in g]
    return out


def training_rows(samples, advs, T):
    """``(inp, tgt, weight)`` rows of T tokens for samples with a nonzero advantage:
    the prompt is context, every sampled token a target weighted by A / tokens."""
    from .model import IGNORE
    from .tokenizer import EOT
    keep = [(p, n, a) for (p, n, _), a in zip(samples, advs) if a != 0 and n]
    total = sum(len(n) for _, n, _ in keep)
    rows = []
    for p, n, a in keep:
        seq = (p + n)[:T + 1]
        inp = seq[:-1]
        tgt = [IGNORE] * (len(p) - 1) + seq[len(p):]
        w = a / total
        pad = T - len(inp)
        rows.append((array.array('I', inp + [EOT] * pad), array.array('I', tgt + [IGNORE] * pad),
                     array.array('f', [0.0] * (len(p) - 1) + [w] * (len(tgt) - len(p) + 1)
                                 + [0.0] * pad)))
    return rows, total


def cmd_train(args):
    from vabs_gpt.gpu import ComputeContext
    from .__main__ import lr_at, muon_of
    from .model import ChatConfig, ChatModel, ChatRunner, IGNORE
    from .tokenizer import EOT, Tokenizer

    if os.path.exists(args.out) and not (args.overwrite or args.resume):
        raise SystemExit(f'{args.out} exists; pass --resume or --overwrite')
    log_path = os.path.splitext(args.out)[0] + '.jsonl'
    if not args.resume and os.path.exists(log_path):
        os.remove(log_path)

    def log(**r):
        with open(log_path, 'a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(r) + '\n')

    train = [t for s in args.split for t in read_tasks(s, args.sources)]
    dev_tasks = read_tasks('dev')
    print(f'{len(train)} training tasks ({", ".join(sorted({t["source"] for t in train}))}), '
          f'{len(dev_tasks)} dev tasks', flush=True)
    with ComputeContext() as ctx:
        from .model import Device
        dev = Device(ctx, precision=args.precision)
        path = args.out if args.resume else args.init
        header, params, state = ChatModel.read(path)
        cfg = ChatConfig(**header['config'])
        kw = state if args.resume and state else dict(optim_bits=args.optim_bits,
                                                     muon=args.optimizer == 'muon')
        model = ChatModel(dev, cfg, params, step=header.get('step', 0) if args.resume else 0,
                          tokenizer=header['tokenizer'], **kw)
        tok = Tokenizer(header['tokenizer'], cfg.vocab)
        progress = header.get('rl', {}) if args.resume else {}
        sampler = Sampler(model, tok, args.sample_batch, args.T, args.max_new, args.workers)
        rows_per_step = args.prompts * args.group
        accum = max(2, math.ceil(rows_per_step / args.batch))
        runner = ChatRunner(model, args.batch, 'train', accum=accum, muon=muon_of(args),
                            T=args.T, weighted=True)
        rng = random.Random(args.seed + model.step)
        stop = {'now': False}

        def on_sigint(*_):
            stop['now'] = True
            print('\ninterrupted: saving after this step', flush=True)

        old = signal.signal(signal.SIGINT, on_sigint)

        def save():
            model.save(args.out, extra={'stage': 'rlvr', 'options': {
                k: v for k, v in vars(args).items() if k != 'cmd'}, 'rl': progress,
                'init': os.path.basename(args.init or '')})

        def dev_eval():
            report, _ = evaluate(sampler, dev_tasks, random.Random(0))
            return report['pass@1']

        if not args.resume:
            d = dev_eval()
            print(f'step 0: dev pass@1 {d:.1%}', flush=True)
            log(kind='dev', step=0, dev=d)
        cursor = progress.get('cursor', 0)
        order = list(range(len(train)))
        random.Random(args.seed).shuffle(order)
        t0 = time.time()
        try:
            while model.step < args.steps and not stop['now']:
                ts = time.time()
                tasks = []
                for _ in range(args.prompts):
                    if cursor >= len(order):
                        cursor = 0
                        random.Random(args.seed + model.step).shuffle(order)
                    tasks.append(train[order[cursor]])
                    cursor += 1
                rep = [t for t in tasks for _ in range(args.group)]
                samples = sampler.complete(rep, args.temperature, rng)
                tsamp = time.time()
                rewards = [float(o) for o in sampler.score(rep, [s[2] for s in samples],
                                                           full=False)]
                tscore = time.time()
                advs = advantages(rewards, args.group, not args.no_std)
                rows, scored = training_rows(samples, advs, args.T)
                lr = lr_at(model.step, args.steps, args)
                signal_groups = sum(1 for i in range(0, len(advs), args.group)
                                    if any(advs[i:i + args.group]))
                if rows:
                    empty = (array.array('I', [EOT] * args.T), array.array('I', [IGNORE] * args.T),
                             array.array('f', [0.0] * args.T))
                    # an accumulating runner's recordings (first, middle, last micro-batch)
                    # serve any count >= 2: run only as many as there are rows for
                    k = max(2, math.ceil(len(rows) / args.batch))
                    runner.accum = k
                    rows = rows + [empty] * (k * args.batch - len(rows))
                    batches, weights = [], []
                    for i in range(k):
                        chunk = rows[i * args.batch:(i + 1) * args.batch]
                        batches.append((array.array('I', [x for r in chunk for x in r[0]]),
                                        array.array('I', [x for r in chunk for x in r[1]])))
                        weights.append(array.array('f', [x for r in chunk for x in r[2]]))
                    nll, norm, _ = runner.train_step(batches, lr, beta1=args.beta1,
                                                     beta2=args.beta2, eps=args.eps,
                                                     weight_decay=args.weight_decay,
                                                     clip=args.grad_clip, weights=weights)
                else:
                    nll, norm = float('nan'), 0.0
                    model.step += 1   # nothing to learn from: count the step anyway
                step = model.step
                lengths = [len(s[1]) for s in samples]
                rec = dict(kind='train', step=step, reward=sum(rewards) / len(rewards),
                           signal=signal_groups / args.prompts, nll=nll, norm=norm, lr=lr,
                           length=sum(lengths) / len(lengths), trained_tokens=scored,
                           t_sample=tsamp - ts, t_score=tscore - tsamp,
                           t_step=time.time() - ts, time=time.time() - t0)
                log(**rec)
                if step % args.log_every == 0:
                    print(f'step {step:4d}/{args.steps} | reward {rec["reward"]:.3f} | '
                          f'signal {rec["signal"]:.0%} | nll {nll:.3f} | norm {norm:.3f} | '
                          f'len {rec["length"]:.0f} | lr {lr:.1e} | {rec["t_step"]:.1f} s '
                          f'(sample {rec["t_sample"]:.1f}, check {rec["t_score"]:.1f}) | '
                          f'{(time.time() - t0) / 60:.1f} min', flush=True)
                if step % args.eval_every == 0 or step >= args.steps:
                    d = dev_eval()
                    print(f'  dev pass@1 {d:.1%}', flush=True)
                    log(kind='dev', step=step, dev=d)
                    progress.update(cursor=cursor, dev=d)
                    save()
            progress.update(cursor=cursor)
            save()
            print(f'saved {args.out} at step {model.step}')
        finally:
            signal.signal(signal.SIGINT, old)
            sampler.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    def common(p, max_new=256):
        from .model import add_precision_arg
        add_precision_arg(p)
        p.add_argument('--batch', type=int, default=128, help='sequences sampled at once')
        p.add_argument('--T', type=int, default=512, help='window: prompt + completion')
        p.add_argument('--max-new', type=int, default=max_new)
        p.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('eval', help='pass@1 (greedy) on held-out tasks')
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--split', nargs='+', default=['dev'],
                   help='train, dev, test, plus, or a .jsonl path')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--samples', type=int, default=0, help='also sample k per task for pass@k')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--hidden', action='store_true',
                   help='MBPP: only the tests the prompt does not show')
    p.add_argument('--out', default='', help='prefix of per-task result files')
    common(p)
    p = sub.add_parser('probe', help='how many tasks are sometimes solved')
    p.add_argument('checkpoint')
    p.add_argument('--split', default='train')
    p.add_argument('--sources', nargs='*', default=None)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', default='')
    common(p)
    p = sub.add_parser('train', help='GRPO on the training tasks')
    from .model import add_precision_arg
    add_precision_arg(p)
    p.add_argument('--init', help='starting checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--split', nargs='+', default=['train'],
                   help='task files: train (MBPP + KodCode), acecode (rl_data acecode), '
                        'or paths')
    p.add_argument('--sources', nargs='*', default=None,
                   help='keep only these sources: mbpp, kodcode, acecode (default all)')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--prompts', type=int, default=32, help='tasks per step')
    p.add_argument('--group', type=int, default=8, help='samples per task')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--no-std', action='store_true', help='advantage r - mean, no division')
    p.add_argument('--sample-batch', type=int, default=128)
    p.add_argument('--T', type=int, default=512)
    p.add_argument('--max-new', type=int, default=256)
    p.add_argument('--batch', type=int, default=16, help='rows per training micro-batch')
    p.add_argument('--workers', type=int, default=28, help='sandboxed checks at once')
    p.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw')
    p.add_argument('--optim-bits', type=int, choices=(8, 32), default=8)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--min-lr', type=float, default=None)
    p.add_argument('--schedule', choices=('cosine', 'linear', 'constant'), default='constant')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.99)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--muon-momentum', type=float, default=0.95)
    p.add_argument('--muon-ns-steps', type=int, default=5)
    p.add_argument('--muon-scale', type=float, default=0.2)
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    {'eval': cmd_eval, 'probe': cmd_probe, 'train': cmd_train}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())

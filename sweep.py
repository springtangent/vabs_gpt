"""Hyperparameter sweep: train each grid point for a fixed token budget, then
score it on the same prefix of val.bin.

    python -m vabs_gpt.sweep --dir vabs_gpt/sweep/lr --tokens 16e6 \\
        --grid lr=1e-3,3e-3,5e-3 batch=16,32 --set weight-decay=0.1
    python -m vabs_gpt.sweep --dir vabs_gpt/sweep/lr --report

Every grid point runs ``python -m vabs_gpt train`` with the given options and
``--steps tokens / (batch * grad-accum * ctx)``, so runs with different batch
sizes see the same number of bytes. Each finished run is appended to
``<dir>/results.jsonl``; rerunning the same command skips finished points.
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time

DEFAULTS = {'batch': 32, 'grad-accum': 1, 'ctx': 256}


def parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_pairs(items, multi):
    out = {}
    for item in items:
        key, _, values = item.partition('=')
        if not values:
            raise SystemExit(f'expected key=value, got {item!r}')
        key = key.strip().lstrip('-').replace('_', '-')
        out[key] = [parse_value(v) for v in values.split(',')] if multi else parse_value(values)
    return out


def run_name(options):
    return ' '.join(f'{k}={v}' for k, v in options.items()) or 'base'


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def train_one(args, options, log_path, ckpt):
    get = lambda k: options.get(k, DEFAULTS[k])  # noqa: E731
    steps = max(1, round(args.tokens / (get('batch') * get('grad-accum') * get('ctx'))))
    cmd = [sys.executable, '-m', 'vabs_gpt', 'train', '--out', ckpt, '--overwrite',
           '--steps', str(steps), '--eval-every', str(steps), '--eval-batches', '1',
           '--log-every', str(max(1, steps // 20))]
    if args.data:
        cmd += ['--data', args.data]
    for k, v in options.items():
        cmd += [f'--{k}', str(v)]
    t0 = time.perf_counter()
    with open(log_path, 'w', encoding='utf-8') as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    seconds = time.perf_counter() - t0
    with open(log_path, encoding='utf-8') as f:
        text = f.read()
    losses = re.findall(r'^step +\d+ +loss ([\d.na]+) .* (\d+) ms/step', text, re.M)
    result = {'name': run_name(options), 'options': options, 'steps': steps,
              'tokens': steps * get('batch') * get('grad-accum') * get('ctx'),
              'seconds': round(seconds), 'exit': code,
              'train_loss': float(losses[-1][0]) if losses else None,
              'ms_per_step': int(losses[-1][1]) if losses else None, 'val_loss': None}
    if code == 0:
        cmd = [sys.executable, '-m', 'vabs_gpt', 'eval', ckpt, '--json',
               '--bytes', str(int(args.eval_bytes))]
        if args.data:
            cmd += ['--data', args.data]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        result['val_loss'] = json.loads(out.strip().splitlines()[-1])['loss']
    if not args.keep:
        try:
            os.remove(ckpt)
        except OSError:
            pass
    return result


def report(results):
    rows = sorted(results, key=lambda r: (r['val_loss'] is None, r['val_loss'] or 0))
    best = next((r['val_loss'] for r in rows if r['val_loss'] is not None), None)
    print(f'{"val loss":>9} {"vs best":>8} {"train":>7} {"steps":>7} {"ms/step":>7} {"s":>5}  run')
    for r in rows:
        if r['val_loss'] is None:
            print(f'{"failed":>9} {"":>8} {"":>7} {r["steps"]:>7} {"":>7} {r["seconds"]:>5}  '
                  f'{r["name"]}')
            continue
        train = f'{r["train_loss"]:.4f}' if r['train_loss'] is not None else ''
        print(f'{r["val_loss"]:9.4f} {r["val_loss"] - best:+8.4f} {train:>7} {r["steps"]:>7} '
              f'{r["ms_per_step"] or "":>7} {r["seconds"]:>5}  {r["name"]}')


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser(prog='python -m vabs_gpt.sweep', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', required=True, help='results, logs and (with --keep) checkpoints')
    p.add_argument('--tokens', type=float, default=16e6, help='training bytes per run')
    p.add_argument('--grid', nargs='*', default=[],
                   help='key=v1,v2,... train options to sweep (cartesian product)')
    p.add_argument('--set', nargs='*', default=[], help='key=value options fixed for all runs')
    p.add_argument('--eval-bytes', type=float, default=4e6,
                   help='score each run on this many bytes from the start of val.bin')
    p.add_argument('--data', help='directory with train.bin and val.bin')
    p.add_argument('--keep', action='store_true', help='keep each run\'s checkpoint')
    p.add_argument('--report', action='store_true', help='print the results table and exit')
    args = p.parse_args(argv)
    os.makedirs(args.dir, exist_ok=True)
    results_path = os.path.join(args.dir, 'results.jsonl')
    if args.report:
        report(load_results(results_path))
        return
    grid, fixed = parse_pairs(args.grid, True), parse_pairs(args.set, False)
    points = [{**fixed, **dict(zip(grid, combo))} for combo in itertools.product(*grid.values())]
    previous = load_results(results_path)
    done, serial = {r['name'] for r in previous}, len(previous)
    for i, options in enumerate(points):
        name = run_name(options)
        if name in done:
            print(f'[{i + 1}/{len(points)}] {name}: done already')
            continue
        print(f'[{i + 1}/{len(points)}] {name} ...', end=' ', flush=True)
        stem = os.path.join(args.dir, f'run{serial:03d}')
        serial += 1
        log = stem + '.log'
        r = train_one(args, options, log, stem + '.bin')
        r['log'] = os.path.basename(log)
        with open(results_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(r) + '\n')
        print(f'val {r["val_loss"]:.4f}' if r['val_loss'] is not None
              else f'failed (exit {r["exit"]}, see {log})', f'({r["seconds"]} s)', flush=True)
    report(load_results(results_path))


if __name__ == '__main__':
    main()

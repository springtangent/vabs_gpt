"""Functional evaluation: does the model's code pass the tests of held-out tasks?

    .ptvenv\\Scripts\\python.exe -m vabs_gpt.chat.funceval build      # -> data\\func_eval.jsonl
    .venv\\Scripts\\python.exe -m vabs_gpt.chat.funceval run base.bin --mode body
    .venv\\Scripts\\python.exe -m vabs_gpt.chat.funceval run chat.bin --mode chat
    .venv\\Scripts\\python.exe -m vabs_gpt.chat.funceval run chat.bin --mode fix

``build`` picks short KodCode "Filter" tasks that are not used by the
fine-tuning chats, keeps the ones whose solution passes its tests and has a
docstring, and stores them with an injected bug (see ``sft_data.mutate``).

Modes:

- ``body``: the prompt is the function's signature and docstring; the model
  writes the body (works on a pretrained checkpoint);
- ``chat``: the user asks the task's question; the first Python code block of
  the reply is tested;
- ``fix``: the user shows the buggy version and its real error; the reply's
  code is tested.

A task passes when its tests pass on the model's code (``sft_data.passes_tests``:
a subprocess with a timeout in a temporary directory). Code that uses files,
processes, the network or other blocked modules is counted as a failure without
being run. Decoding is greedy by default (pass@1).
"""
import argparse
import ast
import json
import os
import random
import re
import sys

from .data import DATA

PATH = os.path.join(DATA, 'func_eval.jsonl')


def stub_of(solution, name):
    """The solution up to and including the function's docstring (imports kept)."""
    tree = ast.parse(solution)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    first = func.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)) or len(func.body) < 2:
        return None
    lines = solution.split('\n')
    return '\n'.join(lines[:first.end_lineno]) + '\n'


def cmd_build(args):
    from . import sft_data as S
    used = set()
    for name in ('chats_train.jsonl', 'chats_val.jsonl'):
        with open(os.path.join(DATA, name), encoding='utf-8') as f:
            used |= {json.loads(line).get('source') for line in f}
    tasks = [t for t in S.load_tasks({'Filter'}) if t['id'] not in used]
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    out = []
    for t in tasks:
        if len(out) >= args.tasks:
            break
        stub = stub_of(t['solution'], t['name'])
        if not stub:
            continue
        c = S.check_task(t)
        if not c or not c['examples']:
            continue
        bug = None
        for kind in ('short', 'stop', 'range', 'name', 'return', 'colon'):   # off-by-one first
            b = S.error_chat(c, random.Random(f'{args.seed}:{t["id"]}:{kind}'), only=(kind,))
            if b:
                bug = b[0][0][1]
                break
        out.append(dict(id=t['id'], name=t['name'], question=t['question'], stub=stub,
                        test=t['test'], fix_prompt=bug))
        print(f'{len(out)} tasks', flush=True)
    with open(PATH, 'w', encoding='utf-8', newline='\n') as f:
        for t in out:
            f.write(json.dumps(t) + '\n')
    print(f'wrote {len(out)} tasks to {PATH}')


CODE_BLOCK = re.compile(r'```(?:python|py)?\n(.*?)(?:```|$)', re.S)


def extract_code(reply):
    m = CODE_BLOCK.search(reply)
    return m.group(1) if m else reply


def cmd_run(args):
    from vabs_gpt.gpu import ComputeContext
    from . import sft_data as S
    from .__main__ import load_tokenizer, open_checkpoint
    from .generate import Generator, function_end
    from .model import Device
    from .tokenizer import EOT

    with open(args.tasks_file, encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f][:args.limit or None]
    if args.mode == 'fix':
        tasks = [t for t in tasks if t['fix_prompt']]
    rng = random.Random(args.seed)
    passed, results = 0, []
    with ComputeContext() as ctx:
        dev = Device(ctx, precision=args.precision)
        model, header = open_checkpoint(dev, args.checkpoint)
        tok = load_tokenizer(header)
        gen = Generator(model, tok)
        for i in range(0, len(tasks), args.batch):
            batch = tasks[i:i + args.batch]
            if args.mode == 'body':
                prompts = [[EOT] + tok.encode(t['stub']) for t in batch]

                def ended(b, out, batch=batch):
                    return function_end(batch[b]['stub'] + tok.decode(out)) is not None
                outs = gen.generate(prompts, max_new=args.max_new, temperature=args.temperature,
                                    rng=rng, stop_fn=ended)
                codes = []
                for t, o in zip(batch, outs):
                    text = t['stub'] + tok.decode(o)
                    end = function_end(text)
                    codes.append(text[:end] if end else text)
            else:
                from .data import prompt_ids
                asks = [t['question'] if args.mode == 'chat' else t['fix_prompt'] for t in batch]
                outs = gen.generate([prompt_ids(tok, [('user', a)]) for a in asks],
                                    max_new=args.max_new, temperature=args.temperature, rng=rng)
                codes = [extract_code(tok.decode(o)) for o in outs]
            for t, code in zip(batch, codes):
                ok = not S.BLOCKED.search(code) and S.passes_tests(code, t['test'])
                passed += ok
                results.append(dict(id=t['id'], passed=ok, code=code))
            print(f'{len(results)}/{len(tasks)}: {passed} passed', flush=True)
    print(f'{args.mode}: pass@1 {passed / max(1, len(tasks)):.1%} ({passed}/{len(tasks)})')
    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('build')
    p.add_argument('--tasks', type=int, default=200)
    p.add_argument('--seed', type=int, default=1)
    p = sub.add_parser('run')
    p.add_argument('checkpoint')
    from .model import add_precision_arg
    add_precision_arg(p)
    p.add_argument('--mode', choices=('body', 'chat', 'fix'), default='body')
    p.add_argument('--tasks-file', default=PATH)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--max-new', type=int, default=300)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='', help='write each task\'s code and result here')
    args = parser.parse_args(argv)
    {'build': cmd_build, 'run': cmd_run}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())

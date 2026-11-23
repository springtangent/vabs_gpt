"""Tasks for reinforcement learning with verifiable rewards (docs/small_python_rlvr.md).

    .ptvenv\\Scripts\\python.exe -m vabs_gpt.chat.rl_data download
    .ptvenv\\Scripts\\python.exe -m vabs_gpt.chat.rl_data build

Every task is a function stub (imports, signature, docstring) whose body the
model writes; the hidden tests decide the reward (1 when all pass, else 0).

- ``rl_train.jsonl``: MBPP's official train split (task ids 601-974) and short
  single-function tasks from KodCode-Light-RL-10K (easy, a docstring, no blocked
  modules; none of the tasks of ``func_eval.jsonl`` or the fine-tuning chats);
- ``rl_dev.jsonl``: MBPP's validation split (511-600), for choosing settings;
- ``rl_test.jsonl``: MBPP's test split (11-510), held out;
- ``rl_acecode.jsonl`` (optional; ``download --acecode``, then ``acecode``):
  AceCode-87K tasks with its synthesized asserts (about 17 per task), for
  ``rlvr train --split train acecode`` (see ``acecode_task``), and
  ``rl_acecode_eval.jsonl``, held-out AceCode tasks (AceCode has no evaluation
  split of its own) that share no function name with the training ones;
- ``rl_plus.jsonl``: the MBPP+ (EvalPlus) versions of the held-out test tasks,
  with every base and extra input checked against the canonical solution.

An MBPP stub's docstring holds the task's text and its first test as a usage
example; ``test`` holds the other tests (the reward never checks what the prompt
shows) and ``test_full`` all of them, the standard MBPP check. Every reference
solution passes its tests in the sandbox (``verify``) or the task is dropped. For
MBPP the reference is cut down to its imports and the one function the stub
declares, so tasks whose tests need anything else (a class or helper defined
beside it) are dropped too: a body cannot provide them.
"""
import argparse
import ast
import gzip
import json
import os
import re
import sys
import urllib.request

from .data import DATA

RAW = os.path.join(DATA, 'raw', 'rl')
SOURCES = {
    'mbpp.jsonl': 'https://raw.githubusercontent.com/google-research/google-research/'
                  'master/mbpp/mbpp.jsonl',
    'MbppPlus.jsonl.gz': 'https://github.com/evalplus/mbppplus_release/releases/download/'
                         'v0.2.0/MbppPlus.jsonl.gz',
    'KodCode-Light-RL-10K.parquet': 'https://huggingface.co/datasets/KodCode/'
                                    'KodCode-Light-RL-10K/resolve/main/data/'
                                    'train-00000-of-00001.parquet',
}
SPLITS = {'test': range(11, 511), 'dev': range(511, 601), 'train': range(601, 975)}


def cmd_download(args):
    os.makedirs(RAW, exist_ok=True)
    sources = dict(SOURCES)
    if args.acecode:   # about 1 GB, mostly the recorded model completions
        os.makedirs(ACECODE, exist_ok=True)
        sources.update({os.path.join('acecode', f'{i}.parquet'): ACECODE_URL.format(i)
                        for i in range(ACECODE_SHARDS)})
    for name, url in sources.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest):
            continue
        print('fetching', url, flush=True)
        with urllib.request.urlopen(url) as r, open(dest + '.part', 'wb') as f:
            f.write(r.read())
        os.replace(dest + '.part', dest)
    print('done')


# -- stubs -----------------------------------------------------------------------------

def docstring(text, example=None):
    """A docstring block (indented by 4) with ``text`` and an optional usage example."""
    lines = [l.strip() for l in text.strip().split('\n')]
    if example:
        lines += [''] + example
    body = '\n'.join(('    ' + l) if l else '' for l in lines)
    q = 'r"""' if '\\' in body else '"""'
    body = body.replace('"""', "'''")
    if len(lines) == 1:
        return f'    {q}{lines[0]}"""\n'
    return f'    {q}{body.lstrip()}\n    """\n'


def example_lines(assert_src, name):
    """``>>> call`` / value for ``assert name(...) == literal``, else the assert itself."""
    try:
        node = ast.parse(assert_src.strip()).body[0]
    except SyntaxError:
        return None
    if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
        left, right = node.test.left, node.test.comparators[0]
        if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id == name):
            try:
                ast.literal_eval(right)
                return ['>>> ' + ast.unparse(left), ast.unparse(right)]
            except ValueError:
                pass
    return ['Example:', ast.unparse(node)]


def make_stub(code, name, text, example):
    """Imports of ``code``, then the signature of its function ``name`` and a docstring."""
    tree = ast.parse(code)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        return None
    imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    head = '\n'.join(imports) + '\n\n\n' if imports else ''
    sig = f'def {name}({ast.unparse(fn.args)}):\n'
    stub = head + sig + docstring(text, example)
    ast.parse(stub + '    pass\n')
    return stub, head + ast.get_source_segment(code, fn) + '\n'


def entry_name(code, tests):
    """The function of ``code`` that the first test calls."""
    defs = {n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(ast.parse(tests[0])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in defs:
            return node.func.id
    return None


def mbpp_tasks():
    with open(os.path.join(RAW, 'mbpp.jsonl'), encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    out = []
    for r in rows:
        split = next((s for s, ids in SPLITS.items() if r['task_id'] in ids), None)
        if not split:
            continue
        code = r['code'].replace('\r\n', '\n').replace('\r', '\n')
        tests = [t.replace('\r\n', '\n').strip() for t in r['test_list']]
        try:
            name = entry_name(code, tests)
            made = name and make_stub(code, name, r['text'], example_lines(tests[0], name))
        except SyntaxError:
            made = None
        if not made:
            continue
        stub, alone = made
        setup = r['test_setup_code'].replace('\r\n', '\n').strip()
        head = 'from solution import *\n' + (setup + '\n' if setup else '')
        out.append(dict(id=f'mbpp/{r["task_id"]}', split=split, source='mbpp', name=name,
                        stub=stub, test=head + '\n'.join(tests[1:]) + '\n',
                        test_full=head + '\n'.join(tests) + '\n', reference=alone))
    return out


PLUS_TEST = '''import copy
import math
from solution import {name} as _candidate

_REFERENCE = {reference!r}
_ns = {{}}
exec(_REFERENCE, _ns)
_expected_of = _ns[{name!r}]
_INPUTS = {inputs!r}
_ATOL = {atol!r}


def _same(a, b):   # EvalPlus: floats to a tolerance, everything else by ==
    if isinstance(b, float) and isinstance(a, (int, float)) and not isinstance(a, bool):
        if math.isnan(b):
            return math.isnan(a)
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=_ATOL)
    if isinstance(b, (list, tuple)) and type(a) is type(b):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_plus():
    for args in _INPUTS:
        expected = _expected_of(*copy.deepcopy(args))
        assert _same(_candidate(*copy.deepcopy(args)), expected), args
'''


def plus_tasks(test_ids):
    with gzip.open(os.path.join(RAW, 'MbppPlus.jsonl.gz'), 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    out = []
    for r in rows:
        tid = int(r['task_id'].split('/')[1])
        if tid not in test_ids:
            continue
        name, code = r['entry_point'], r['canonical_solution']
        doc = r['prompt'].strip().strip('"').strip()
        text = '\n'.join(l for l in doc.split('\n') if not l.startswith('assert '))
        shown = next((l for l in doc.split('\n') if l.startswith('assert ')), None)
        made = make_stub(code, name, text, shown and example_lines(shown, name))
        if not made:
            continue
        stub, alone = made
        test = PLUS_TEST.format(name=name, reference=code, atol=float(r['atol'] or 0),
                                inputs=r['base_input'] + r['plus_input'])
        out.append(dict(id=f'mbpp+/{tid}', split='plus', source='mbpp+', name=name, stub=stub,
                        test=test, test_full=test, reference=alone))
    return out


def kodcode_tasks(exclude, limit):
    import pyarrow.parquet as pq
    from . import sft_data as S
    from .funceval import stub_of
    table = pq.read_table(os.path.join(RAW, 'KodCode-Light-RL-10K.parquet'),
                          columns=['question_id', 'question', 'solution', 'test',
                                   'gpt_difficulty', 'subset'])
    out = []
    for r in table.to_pylist():
        if r['question_id'] in exclude or r['gpt_difficulty'] != 'easy':
            continue
        s = r['solution'].strip() + '\n'
        if len(s) > 700 or s.count('\n') > 24 or '\t' in s:
            continue
        if S.BLOCKED.search(s) or S.BLOCKED.search(r['test']) or 'solution import' not in r['test']:
            continue
        info = S.function_info(s)
        if not info:
            continue
        try:
            stub = stub_of(s, info[0])
        except (SyntaxError, StopIteration):
            stub = None
        if not stub or len(stub) > 900:
            continue
        out.append(dict(id='kodcode/' + r['question_id'], split='train', source='kodcode',
                        name=info[0], stub=stub, test=r['test'], test_full=r['test'],
                        reference=s))
        if len(out) >= limit:
            break
    return out


ACECODE = os.path.join(RAW, 'acecode')
ACECODE_URL = ('https://huggingface.co/api/datasets/TIGER-Lab/AceCode-87K/parquet/'
               'default/train/{}.parquet')
ACECODE_SHARDS = 11
FENCE = re.compile(r'```[a-zA-Z]*\n?(.*?)```', re.S)


def acecode_reference(completion):
    """The code of a completion: its first fenced block, or all of it when there is none."""
    m = FENCE.search(completion)
    return (m.group(1) if m else completion).replace('\r\n', '\n').strip() + '\n'


def acecode_task(row, exclude_names, min_pass):
    """An RL task from an AceCode-87K row, or None.

    AceCode has a question and synthesized asserts but no reference solution; the
    reference is the shortest model completion that the dataset records as passing
    every test (``rl_data acecode`` re-checks it in the sandbox). The mean pass rate
    over all the recorded completions is a difficulty score: tasks under
    ``min_pass`` are too hard to give a small model any signal.
    """
    from .sft_data import BLOCKED
    tests = [t.replace('\r\n', '\n').strip() for t in row['test_cases'] or []]
    infs = row['inferences'] or []
    if len(tests) < 2 or not infs or any(BLOCKED.search(t) for t in tests):
        return None
    if sum(i['pass_rate'] for i in infs) / len(infs) < min_pass:
        return None
    text = FENCE.sub('', row['question']).strip()
    for c in sorted((i['completion'] for i in infs if i['pass_rate'] == 1.0), key=len):
        code = acecode_reference(c)
        if '\t' in code:
            continue
        try:
            name = entry_name(code, tests)
            if name in exclude_names:
                return None
            made = name and make_stub(code, name, text, example_lines(tests[0], name))
        except (SyntaxError, ValueError):
            continue
        if not made or len(made[0]) > 900:
            continue
        stub, alone = made   # the imports and the one function: any demo code is dropped
        if len(alone) > 700 or alone.count('\n') > 24 or BLOCKED.search(alone):
            continue
        head = 'from solution import *\n'
        return dict(id='acecode/' + row['id'], split='acecode', source='acecode', name=name,
                    stub=stub, test=head + '\n'.join(tests[1:]) + '\n',
                    test_full=head + '\n'.join(tests) + '\n', reference=alone)
    return None


def cmd_acecode(args):
    """``data/rl_acecode.jsonl``: AceCode-87K tasks for ``rlvr train --split train acecode``."""
    import random
    import pyarrow.parquet as pq
    from . import verify as V
    exclude = set()
    for name in ('rl_dev.jsonl', 'rl_test.jsonl', 'rl_plus.jsonl', 'func_eval.jsonl'):
        with open(os.path.join(DATA, name), encoding='utf-8') as f:
            exclude |= {json.loads(line)['name'] for line in f}
    tasks, rows = [], 0
    for i in range(ACECODE_SHARDS):
        pf = pq.ParquetFile(os.path.join(ACECODE, f'{i}.parquet'))
        for batch in pf.iter_batches(batch_size=500,
                                     columns=['id', 'question', 'test_cases', 'inferences']):
            for r in batch.to_pylist():
                rows += 1
                t = acecode_task(r, exclude, args.min_pass)
                if t:
                    tasks.append(t)
        print(f'shard {i}: {rows} rows, {len(tasks)} candidate tasks', flush=True)
    random.Random(args.seed).shuffle(tasks)
    seen, unique = set(), []   # rows repeat a few questions; keep one task per stub
    for t in tasks:
        if t['stub'] not in seen:
            seen.add(t['stub'])
            unique.append(t)
    need = args.limit + args.eval
    tasks = unique[:int(need * 1.2) + 100]   # a margin for references that fail
    pool = V.Pool(args.workers)
    print(f'checking {len(tasks)} reference solutions', flush=True)
    ok = pool.map([(t['reference'], t['test_full']) for t in tasks], timeout=30)
    kept = [t for t, k in zip(tasks, ok) if k]
    print(f'{sum(ok)} of {len(tasks)} references pass', flush=True)
    # held out: no training task shares a function name with an evaluation task
    held = kept[:args.eval]
    for t in held:
        t['split'] = 'acecode_eval'
    names = {t['name'] for t in held}
    train = [t for t in kept[args.eval:] if t['name'] not in names][:args.limit]
    write(os.path.join(DATA, 'rl_acecode.jsonl'), train)
    write(os.path.join(DATA, 'rl_acecode_eval.jsonl'), held)
    if args.sft:   # stub -> body examples for fine-tuning, on tasks neither RL set uses
        names |= {t['name'] for t in train}
        rest = [t for t in unique[len(tasks):] if t['name'] not in names]
        rest = rest[:int(args.sft * 1.2) + 100]
        print(f'checking {len(rest)} reference solutions for fine-tuning', flush=True)
        ok = pool.map([(t['reference'], t['test_full']) for t in rest], timeout=30)
        sft = []
        for t, k in zip(rest, ok):
            body = acecode_body(t) if k else None
            if body:
                sft.append(dict(t, split='acecode_sft', body=body))
        write(os.path.join(DATA, 'acecode_sft.jsonl'), sft[:args.sft])
    pool.close()


def acecode_body(task):
    """The body of a task's reference: the lines after the signature, minus a docstring."""
    code = task['reference']
    fn = next(n for n in ast.parse(code).body
              if isinstance(n, ast.FunctionDef) and n.name == task['name'])
    first = fn.body[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        if len(fn.body) == 1:
            return None
        first = fn.body[1]
    lines = code.split('\n')[first.lineno - 1:fn.end_lineno]
    body = '\n'.join(lines).rstrip() + '\n'
    if not body.startswith('    ') or body.startswith('     '):
        return None
    return body


def write(path, tasks):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for t in tasks:
            f.write(json.dumps(t) + '\n')
    print(f'wrote {len(tasks)} tasks to {path}')


def cmd_build(args):
    from . import verify as V
    exclude = set()
    for name in ('func_eval.jsonl', 'chats_train.jsonl', 'chats_val.jsonl'):
        with open(os.path.join(DATA, name), encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                exclude.add(r.get('id') or r.get('source'))
    mbpp = mbpp_tasks()
    plus = plus_tasks(set(SPLITS['test']))
    kod = kodcode_tasks(exclude, args.kodcode)
    pool = V.Pool(args.workers)
    every = mbpp + plus + kod
    print(f'checking {len(every)} reference solutions', flush=True)
    ok = pool.map([(t['reference'], t['test_full']) for t in every], timeout=30)
    kept = [t for t, k in zip(every, ok) if k]
    for t, k in zip(every, ok):
        if not k:
            print('dropped (reference fails):', t['id'])
    pool.close()
    by = {}
    for t in kept:
        by.setdefault(t['split'], []).append(t)
    write(os.path.join(DATA, 'rl_train.jsonl'), by.get('train', []))
    write(os.path.join(DATA, 'rl_dev.jsonl'), by.get('dev', []))
    write(os.path.join(DATA, 'rl_test.jsonl'), by.get('test', []))
    write(os.path.join(DATA, 'rl_plus.jsonl'), by.get('plus', []))
    for split, ts in sorted(by.items()):
        print(split, len(ts), {s: sum(t['source'] == s for t in ts) for s in
                               ('mbpp', 'mbpp+', 'kodcode')})


def cmd_sft(args):
    """Stub -> body completions for supervised fine-tuning before RL, plus the chats.

    KodCode-V1 "Filter" and "Prefill" tasks (the shards the chats came from) that are
    in none of the RL or evaluation sets and whose solution passes its tests: the
    prompt is the stub (``funceval.stub_of``), the completion the rest of the function.
    Written as completion records (``data.encode_chat`` roles ``prompt`` and
    ``completion``) together with the fine-tuning chats, so the chat skills stay in
    the mix.
    """
    from . import sft_data as S
    from . import verify as V
    from .funceval import stub_of
    used = set()
    for name in ('func_eval.jsonl', 'rl_train.jsonl', 'rl_dev.jsonl', 'rl_test.jsonl'):
        with open(os.path.join(DATA, name), encoding='utf-8') as f:
            used |= {json.loads(line)['id'].split('/')[-1] for line in f}
    tasks = []
    for t in S.load_tasks({'Filter', 'Prefill'}):
        if t['id'] in used:
            continue
        try:
            stub = stub_of(t['solution'], t['name'])
        except (SyntaxError, StopIteration):
            continue
        if not stub:
            continue
        body = t['solution'][len(stub):].rstrip() + '\n'
        if body.strip() and body.startswith('    '):
            tasks.append((t, stub, body))
    import random
    random.Random(args.seed).shuffle(tasks)
    tasks = tasks[:args.bodies + args.val]
    pool = V.Pool(args.workers)
    ok = pool.map([(t['solution'], t['test']) for t, _, _ in tasks])
    pool.close()
    tasks = [x for x, k in zip(tasks, ok) if k]
    records = [dict(source=t['id'], kind='body', turns=[['prompt', stub], ['completion', body]])
               for t, stub, body in tasks]
    val, train = records[:args.val], records[args.val:]
    if args.acecode:   # extra training bodies from ``rl_data acecode --sft``
        with open(os.path.join(DATA, 'acecode_sft.jsonl'), encoding='utf-8') as f:
            ace = [json.loads(line) for line in f][:args.acecode]
        train += [dict(source=t['id'], kind='body', turns=[['prompt', t['stub']],
                                                           ['completion', t['body']]])
                  for t in ace]
        print(f'{len(ace)} AceCode bodies added', flush=True)
    with open(os.path.join(DATA, 'chats_train.jsonl'), encoding='utf-8') as f:
        chats = [json.loads(line) for line in f]
    with open(os.path.join(DATA, 'chats_val.jsonl'), encoding='utf-8') as f:
        chats_val = [json.loads(line) for line in f]
    write(os.path.join(DATA, f'{args.out}_train.jsonl'), train + chats)
    write(os.path.join(DATA, f'{args.out}_val.jsonl'), val + chats_val)
    print(f'{len(train)} bodies and {len(chats)} chats; validation {len(val)} bodies')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('download')
    p.add_argument('--acecode', action='store_true', help='also AceCode-87K (about 1 GB)')
    p = sub.add_parser('acecode', help='rl_acecode.jsonl: AceCode-87K training tasks')
    p.add_argument('--limit', type=int, default=5000, help='training tasks')
    p.add_argument('--eval', type=int, default=500, help='held-out tasks (rl_acecode_eval)')
    p.add_argument('--sft', type=int, default=0,
                   help='stub -> body examples for fine-tuning (acecode_sft.jsonl), on '
                        'tasks no RL or evaluation set uses')
    p.add_argument('--min-pass', type=float, default=0.5,
                   help='least mean pass rate of the recorded completions (difficulty)')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('sft', help='stub -> body completions for fine-tuning before RL')
    p.add_argument('--bodies', type=int, default=6000)
    p.add_argument('--val', type=int, default=200)
    p.add_argument('--acecode', type=int, default=0,
                   help='also this many AceCode bodies (rl_data acecode --sft)')
    p.add_argument('--out', default='bodies', help='writes <out>_train/val.jsonl in data/')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('build')
    p.add_argument('--kodcode', type=int, default=3000, help='at most this many KodCode tasks')
    p.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    {'download': cmd_download, 'build': cmd_build, 'sft': cmd_sft,
     'acecode': cmd_acecode}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())

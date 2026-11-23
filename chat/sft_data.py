"""Build the fine-tuning chats: checked Python coding conversations plus general chat.

    .ptvenv\\Scripts\\python.exe -m vabs_gpt.chat.sft_data --chats 8000

Writes ``data/chats_train.jsonl`` and ``data/chats_val.jsonl`` (``{"kind", "turns"}``
per line). About two thirds are coding conversations and one third general
instruction following (``--code-share``):

- ``write``: a short task from KodCode-V1's "Filter" subset, answered with its
  solution and a usage example taken from the tests.
- ``complete``: a function stub from KodCode's "Prefill" subset, completed.
- ``error``: the solution with one injected bug (a misspelled name, an
  off-by-one ``range``, a missing colon, a missing ``return``) and the error or
  wrong result it really produces, answered with an explanation and the fix.
- ``followup``: a ``write`` chat followed by an edit request: tests, a usage
  example, a rename, or a version without the docstring.
- ``general``: Smol-SmolTalk conversations (no code, no system prompt, English).

"Checked answers": every solution is run against the task's own tests first
(in a subprocess with a timeout, in a temporary directory, with tasks that touch
files, processes or the network left out), every quoted error message and
result is the real output of running the shown code, and every edited version
is run again. Tasks whose checks fail are dropped.
"""
import argparse
import ast
import concurrent.futures
import glob
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import warnings
import zlib

from .data import DATA

RAW = os.path.join(DATA, 'raw')
warnings.filterwarnings('ignore', category=SyntaxWarning)   # escapes in the tasks' sources
BLOCKED = re.compile(r'\b(os|sys|subprocess|shutil|socket|requests|urllib|pathlib|threading|'
                     r'multiprocessing|asyncio|time|datetime|random|numpy|pandas|torch|scipy|'
                     r'sklearn|matplotlib|tensorflow|pickle|sqlite3|csv|json|io|logging|'
                     r'unittest|tempfile|glob|signal|ctypes)\b|open\(|input\(|exec\(|eval\(|'
                     r'__import__|print\(')
PYTEST_SHIM = '''"""Just enough of pytest for KodCode's tests: raises, approx, mark.parametrize."""
import math


class raises:
    def __init__(self, exc, match=None):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is None:
            raise AssertionError(f'DID NOT RAISE {self.exc}')
        return issubclass(kind, self.exc)


class approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        e = self.expected
        if isinstance(e, (list, tuple)):
            return len(e) == len(other) and all(approx(a, self.rel, self.abs) == b
                                                for a, b in zip(e, other))
        return math.isclose(other, e, rel_tol=self.rel, abs_tol=self.abs)


class _Mark:
    def parametrize(self, names, values, **kw):
        def deco(f):
            f._params = ([n.strip() for n in names.split(',')] if isinstance(names, str)
                         else list(names), list(values))
            return f
        return deco

    def __getattr__(self, name):
        return lambda *a, **k: (lambda f: f)


mark = _Mark()
'''
RUNNER = '''import inspect, sys
sys.path.insert(0, sys.argv[1])
import test_solution as t
n = 0
for name, f in list(vars(t).items()):
    if not name.startswith('test') or not callable(f):
        continue
    params = getattr(f, '_params', None)
    if params:
        names, values = params
        for v in values:
            v = v if len(names) > 1 else (v,)
            f(**dict(zip(names, v)))
            n += 1
    else:
        if inspect.signature(f).parameters:
            raise SystemExit('fixture')
        f()
        n += 1
print('PASSED', n)
'''


def run_python(files, main, timeout=10):
    """Run ``main`` (a file name in ``files``) in a fresh temporary directory.

    Returns ``(returncode, stdout, stderr)``, with the directory's path replaced by
    nothing in the output (tracebacks then name ``main.py``).
    """
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            with open(os.path.join(d, name), 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
        try:
            r = subprocess.run([sys.executable, '-I', '-X', 'utf8', os.path.join(d, main), d],
                               cwd=d, capture_output=True, text=True, timeout=timeout,
                               encoding='utf-8', errors='replace',
                               env={'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''),
                                    'PYTHONHASHSEED': '0'})
        except subprocess.TimeoutExpired:
            return -1, '', 'timeout'
        clean = lambda s: s.replace(d + os.sep, '').replace(d, '')  # noqa: E731
        return r.returncode, clean(r.stdout), clean(r.stderr)


def passes_tests(solution, test):
    code, out, _ = run_python({'solution.py': solution, 'test_solution.py': test,
                               'pytest.py': PYTEST_SHIM, 'runner.py': RUNNER}, 'runner.py')
    return code == 0 and out.startswith('PASSED') and int(out.split()[1]) > 0


# -- reading the tasks -----------------------------------------------------------------

def function_info(solution):
    """``(name, FunctionDef)`` of the single top-level function, or None."""
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.Import,
                                                          ast.ImportFrom, ast.Expr))]
    if len(funcs) != 1 or others:
        return None
    return funcs[0].name, funcs[0]


def examples(test, name, limit=3):
    """Simple ``(call, expected)`` source pairs from ``assert name(...) == value`` lines,
    where every argument and the value are literals."""
    out = []
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)):
            continue
        cmp = node.test
        if len(cmp.ops) != 1 or not isinstance(cmp.ops[0], ast.Eq):
            continue
        call, value = cmp.left, cmp.comparators[0]
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == name and not call.keywords):
            continue
        try:
            for a in call.args:
                ast.literal_eval(a)
            ast.literal_eval(value)
        except ValueError:
            continue
        c, v = ast.get_source_segment(test, call), ast.get_source_segment(test, value)
        if c and v and len(c) <= 70 and len(v) <= 50 and '\n' not in c + v:
            out.append((c, v))
    seen, uniq = set(), []
    for c, v in out:
        if c not in seen:
            seen.add(c)
            uniq.append((c, v))
    # examples with telling values first: f(0) == 0 shows little
    trivial = {'0', '1', '-1', 'True', 'False', 'None', '[]', "''", '""', '{}', '()'}
    uniq.sort(key=lambda e: e[1] in trivial)
    return uniq[:limit]


def load_tasks(subsets, max_question=420, max_solution=700, max_lines=24):
    import pyarrow.parquet as pq
    tasks = []
    for path in sorted(glob.glob(os.path.join(RAW, 'kodcode', '*.parquet'))):
        table = pq.read_table(path, columns=['subset', 'style', 'question_id', 'question',
                                             'solution', 'test', 'gpt_difficulty'])
        for r in table.to_pylist():
            if r['subset'] not in subsets or r['gpt_difficulty'] not in ('easy', 'medium'):
                continue
            q, s = r['question'].strip(), r['solution'].strip() + '\n'
            if len(q) > max_question or len(s) > max_solution or s.count('\n') > max_lines:
                continue
            if BLOCKED.search(s) or BLOCKED.search(r['test']) or '\t' in s:
                continue
            info = function_info(s)
            if not info:
                continue
            tasks.append(dict(id=r['question_id'], subset=r['subset'], style=r['style'],
                              question=q, solution=s, test=r['test'], name=info[0]))
    return tasks


def check_task(task):
    if not passes_tests(task['solution'], task['test']):
        return None
    task = dict(task, examples=examples(task['test'], task['name']))
    # the examples must hold for the solution as shown (no hidden setup)
    for call, value in task['examples']:
        got = call_result(task['solution'], call)
        if got is None or got[0] != 'ok' or got[1] != repr(ast.literal_eval(value)):
            return None
    return task


def call_result(code, call):
    """Run ``code`` then ``print(repr(call))`` as main.py: ``('ok', repr)`` or
    ``('error', traceback)``; None on a timeout."""
    rc, out, err = run_python({'main.py': code.rstrip('\n') + f'\n\nprint(repr({call}))\n'},
                              'main.py', timeout=5)
    if rc == -1:
        return None
    if rc == 0:
        return 'ok', out.strip()
    return 'error', err.strip()


# -- chat templates --------------------------------------------------------------------

LEADS = ['Here is a function that does that:', "Here's one way to write it:",
         'You can do it like this:', 'Here is a simple solution:', 'Sure! Here is the code:',
         'This function does it:']
EXAMPLE = ['For example, `{c}` returns `{v}`.', 'For example, calling `{c}` gives `{v}`.',
           '`{c}` returns `{v}`, for example.']


def code_block(code):
    return f'```python\n{code.rstrip()}\n```'


def write_chat(task, rng):
    reply = f'{rng.choice(LEADS)}\n\n{code_block(task["solution"])}'
    if task['examples']:
        c, v = task['examples'][0]
        reply += '\n\n' + rng.choice(EXAMPLE).format(c=c, v=v)
    return [('user', task['question']), ('assistant', reply)]


def complete_chat(task, rng):
    stub = task['question'].rstrip()
    ask = rng.choice(['Complete this function:', 'Can you fill in the body of this function?',
                      'Finish this Python function:', 'Write the body of this function.'])
    reply = f'{rng.choice(LEADS)}\n\n{code_block(task["solution"])}'
    return [('user', f'{ask}\n\n{code_block(stub)}'), ('assistant', reply)]


def _lines(code):
    return code.rstrip('\n').split('\n')


def mutate(task, kind, rng):
    """``(buggy code, explanation)`` for one kind of bug, or None if it does not apply."""
    code, name = task['solution'], task['name']
    tree = ast.parse(code)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    lines = _lines(code)
    if kind == 'name':
        stores = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)
                  and isinstance(n.ctx, ast.Store) and len(n.id) >= 4}
        stores |= {a.arg for a in func.args.args if len(a.arg) >= 4}
        loads = [n for n in ast.walk(func) if isinstance(n, ast.Name)
                 and isinstance(n.ctx, ast.Load) and n.id in stores]
        if not loads:
            return None
        n = rng.choice(loads)
        k = rng.randrange(1, len(n.id) - 1)
        typo = n.id[:k] + n.id[k + 1:]
        if typo in stores or not typo.isidentifier() or typo in code:
            return None
        line = lines[n.lineno - 1]
        lines[n.lineno - 1] = line[:n.col_offset] + typo + line[n.end_col_offset:]
        expl = (f"The error says `name '{typo}' is not defined`. On line {n.lineno} you "
                f"wrote `{typo}`, but the variable is called `{n.id}`. Python treats the "
                f"misspelled name as a different variable that was never assigned. Fix the "
                f"typo:")
        return '\n'.join(lines) + '\n', expl, 'NameError'
    if kind == 'range':
        m = [(i, mt) for i, l in enumerate(lines)
             for mt in [re.search(r'range\(len\((\w+)\)\)', l)] if mt]
        if not m:
            return None
        i, mt = rng.choice(m)
        var = mt.group(1)
        lines[i] = lines[i][:mt.start()] + f'range(len({var}) + 1)' + lines[i][mt.end():]
        expl = (f"`IndexError: ... index out of range` means the code read past the end of "
                f"`{var}`. On line {i + 1}, `range(len({var}) + 1)` counts up to and including "
                f"`len({var})`, but the last valid index is `len({var}) - 1`. This is an "
                f"off-by-one error: `range(len({var}))` already stops before `len({var})`.")
        return '\n'.join(lines) + '\n', expl, 'IndexError'
    if kind == 'short':   # off by one: the loop misses the last element
        m = [(i, mt) for i, l in enumerate(lines)
             for mt in [re.search(r'range\(len\((\w+)\)\)', l)] if mt]
        if not m:
            return None
        i, mt = rng.choice(m)
        var = mt.group(1)
        lines[i] = lines[i][:mt.start()] + f'range(len({var}) - 1)' + lines[i][mt.end():]
        expl = (f"This is an off-by-one error on line {i + 1}: the loop never reaches the last "
                f"element of `{var}`. `range` already stops before its end, so "
                f"`range(len({var}) - 1)` ends at index `len({var}) - 2`. Use "
                f"`range(len({var}))`:")
        return '\n'.join(lines) + '\n', expl, 'wrong'
    if kind == 'stop':    # off by one: range(a, b + 1) written as range(a, b)
        m = [(i, mt) for i, l in enumerate(lines)
             for mt in [re.search(r'range\(([^()]*?),\s*([\w.]+(?:\([\w.]*\))?) \+ 1\)', l)] if mt]
        if not m:
            return None
        i, mt = rng.choice(m)
        a, b = mt.group(1), mt.group(2)
        lines[i] = lines[i][:mt.start()] + f'range({a}, {b})' + lines[i][mt.end():]
        expl = (f"This is an off-by-one error on line {i + 1}. `range({a}, {b})` stops before "
                f"`{b}`, so the loop never handles `{b}` itself. To include it, the end of the "
                f"range has to be `{b} + 1`:")
        return '\n'.join(lines) + '\n', expl, 'wrong'
    if kind == 'colon':
        cands = [i for i, l in enumerate(lines)
                 if re.match(r'\s*(for|if|while|elif)\b.*:\s*$', l) and '#' not in l]
        if not cands:
            return None
        i = rng.choice(cands)
        keyword = lines[i].split()[0]
        lines[i] = lines[i].rstrip()[:-1]
        expl = (f"Python needs a colon at the end of the `{keyword}` line (line {i + 1}) to "
                f"start the indented block that follows. Add it back:")
        return '\n'.join(lines) + '\n', expl, 'SyntaxError'
    if kind == 'return':
        last = func.body[-1]
        if not (isinstance(last, ast.Return) and isinstance(last.value, ast.Name)
                and last.lineno == last.end_lineno):
            return None
        if any(isinstance(n, ast.Return) for n in ast.walk(func) if n is not last):
            return None
        var = last.value.id
        del lines[last.lineno - 1]
        expl = (f"Your function computes `{var}` but never returns it. A Python function "
                f"without a `return` statement returns `None`, which is what you see. Add "
                f"`return {var}` at the end:")
        return '\n'.join(lines) + '\n', expl, 'None'
    raise ValueError(kind)


OFF_BY_ONE_SITE = re.compile(r'range\(len\(\w+\)\)|range\([^()]*?,\s*[\w.]+(?:\([\w.]*\))? \+ 1\)')
ERROR_ASKS = ['I get an error when I run this:', 'Why does this code fail?',
              'Can you help me fix this error?', 'My function crashes:']
WRONG_ASKS = ['My function returns `None` instead of the result:',
              'Why does this print `None`?']
OFF_ASKS = ['My function gives the wrong answer:', 'Why is the result wrong?',
            'Something is off in this function:', 'Can you find the bug?']


OFF_BY_ONE = ('short', 'stop', 'range')


def error_chat(task, rng, only=None):
    """An error chat for ``task``, trying the bug kinds in ``only`` (default: all)."""
    if not task['examples']:
        return None
    call, value = task['examples'][0]
    off_by_one, others = list(OFF_BY_ONE), ['name', 'colon', 'return']
    rng.shuffle(off_by_one)
    rng.shuffle(others)
    kinds = off_by_one + others if rng.random() < 0.6 else others + off_by_one
    if only:
        kinds = [k for k in kinds if k in only]
    for kind in kinds:
        bug = mutate(task, kind, rng)
        if not bug:
            continue
        buggy, expl, expect = bug
        got = call_result(buggy, call)
        if got is None:
            continue
        status, text = got
        if expect == 'None':
            if status != 'ok' or text != 'None' or value == 'None':
                continue
            ask = (f'{rng.choice(WRONG_ASKS)}\n\n{code_block(buggy)}\n\n`print({call})` '
                   f'shows `None`, but I expected `{value}`.')
        elif expect == 'wrong':
            if status != 'ok' or text == repr(ast.literal_eval(value)) or len(text) > 60 \
                    or text == 'None':
                continue
            ask = (f'{rng.choice(OFF_ASKS)}\n\n{code_block(buggy)}\n\n`{call}` returns '
                   f'`{text}`, but it should return `{value}`.')
        else:
            if status != 'error' or expect not in text.splitlines()[-1]:
                continue
            tb = '\n'.join(text.splitlines()[-8:])
            when = 'When I run it' if expect == 'SyntaxError' else f'When I call `{call}`'
            ask = (f'{rng.choice(ERROR_ASKS)}\n\n{code_block(buggy)}\n\n{when} I get:'
                   f'\n\n```\n{tb}\n```')
        reply = f'{expl}\n\n{code_block(task["solution"])}'
        if kind == 'range':
            reply = expl + ' Here is the fixed version:\n\n' + code_block(task['solution'])
        reply += f'\n\nNow `{call}` returns `{value}`.'
        return [('user', ask), ('assistant', reply)], kind
    return None


def strip_docstring(code):
    tree = ast.parse(code)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    first = func.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)) or len(func.body) < 2:
        return None
    lines = _lines(code)
    del lines[first.lineno - 1:first.end_lineno]
    lines = [l for l in lines if not l.strip().startswith('#')]
    out = '\n'.join(lines) + '\n'
    return re.sub(r'\n{3,}', '\n\n', out)


def test_functions(test, limit=3):
    """Up to ``limit`` plain test functions (no fixtures, no parametrize) as source."""
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return []
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith('test') and not n.args.args \
                and not n.decorator_list:
            src = ast.get_source_segment(test, n)
            if src and src.count('\n') <= 6 and 'pytest' not in src:
                out.append(src)
    return out[:limit]


def followup_chat(task, rng):
    base = write_chat(task, rng)
    kinds = ['tests', 'usage', 'rename', 'docstring']
    rng.shuffle(kinds)
    name = task['name']
    for kind in kinds:
        if kind == 'tests':
            funcs = test_functions(task['test'])
            if len(funcs) < 2:
                continue
            tests = f'from solution import {name}\n\n\n' + '\n\n\n'.join(funcs) + '\n'
            if not passes_tests(task['solution'], tests):
                continue
            ask = rng.choice(['Can you write some tests for it?', 'How can I test this?',
                              'Add a few pytest tests, please.'])
            reply = (f'Here are some tests. Save the function as `solution.py` and run '
                     f'`pytest` in the same folder:\n\n{code_block(tests)}')
        elif kind == 'usage':
            if len(task['examples']) < 2:
                continue
            calls = '\n'.join(f'print({c})  # {v}' for c, v in task['examples'][:3])
            ask = rng.choice(['How do I call it?', 'Can you show me how to use it?',
                              'Show me an example.'])
            reply = f'Call it with your values and print the result:\n\n{code_block(calls)}'
        elif kind == 'rename':
            parts = name.split('_')
            if len(parts) < 2:
                continue
            new = parts[-1] if parts[0] in ('get', 'find', 'calculate', 'compute') \
                else 'my_' + parts[-1]
            if not new.isidentifier() or re.search(rf'\b{new}\b', task['solution']):
                continue
            code = re.sub(rf'\b{name}\b', new, task['solution'])
            if not task['examples']:
                continue
            c, v = task['examples'][0]
            c2 = re.sub(rf'\b{name}\b', new, c)
            got = call_result(code, c2)
            if not got or got != ('ok', repr(ast.literal_eval(v))):
                continue
            ask = rng.choice([f'Please rename the function to `{new}`.',
                              f'Can you call it `{new}` instead?'])
            reply = f'Sure, here it is as `{new}`:\n\n{code_block(code)}'
        else:
            code = strip_docstring(task['solution'])
            if not code or not task['examples']:
                continue
            c, v = task['examples'][0]
            if call_result(code, c) != ('ok', repr(ast.literal_eval(v))):
                continue
            ask = rng.choice(['Can you make it shorter, without the docstring?',
                              'Remove the docstring and comments, please.'])
            reply = f'Here is the same function without the docstring:\n\n{code_block(code)}'
        return base + [('user', ask), ('assistant', reply)], kind
    return None


# -- general chat --------------------------------------------------------------------------

GENERAL_SOURCES = {'everyday-conversations', 'smol-magpie-ultra-short', 'smol-contraints',
                   'smollm-rewrite-30k', 'explore-instruct-rewrite', 'smol-summarize-5k',
                   'smol-summarize-20k', 'openhermes-50k'}


def general_chats(max_chars=3000):
    import pyarrow.parquet as pq
    out = []
    for path in sorted(glob.glob(os.path.join(RAW, 'smoltalk', '*train*.parquet'))):
        for r in pq.read_table(path).to_pylist():
            if r['source'] not in GENERAL_SOURCES:
                continue
            msgs = r['messages']
            if not msgs or any(m['role'] not in ('user', 'assistant') for m in msgs):
                continue
            if [m['role'] for m in msgs] != ['user', 'assistant'] * (len(msgs) // 2) \
                    or len(msgs) % 2:
                continue
            text = ''.join(m['content'] for m in msgs)
            if len(text) > max_chars or '```' in text or 'def ' in text:
                continue
            ascii_share = sum(c < '\x80' for c in text) / max(1, len(text))
            if ascii_share < 0.99:
                continue
            out.append(([(m['role'], m['content'].strip()) for m in msgs], r['source']))
    return out


# -- main ------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chats', type=int, default=8000, help='total chats (train + val)')
    parser.add_argument('--code-share', type=float, default=2 / 3)
    parser.add_argument('--val-share', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 4))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-tokens', type=int, default=1000,
                        help='drop chats longer than this many tokens')
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    from .tokenizer import Tokenizer
    from .data import encode_chat
    tok = Tokenizer.load(os.path.join(DATA, 'tokenizer.json'))

    n_code = round(args.chats * args.code_share)
    want = {'write': 0.40, 'complete': 0.15, 'error': 0.25, 'followup': 0.20}
    want = {k: round(n_code * v) for k, v in want.items()}
    tasks = load_tasks({'Filter', 'Prefill'})
    rng.shuffle(tasks)
    filt = [t for t in tasks if t['subset'] == 'Filter']
    pre = [t for t in tasks if t['subset'] == 'Prefill' and t['style'] == 'complete']
    print(f'{len(filt):,} short Filter tasks, {len(pre):,} Prefill stubs (before checking)',
          flush=True)

    def checked(pool_tasks, n):
        """The first tasks of ``pool_tasks`` that pass their tests, about ``n`` of them."""
        out, i = [], 0
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            while len(out) < n and i < len(pool_tasks):
                batch = pool_tasks[i:i + max(64, 2 * (n - len(out)))]
                i += len(batch)
                out += [t for t in pool.map(check_task, batch) if t]
                print(f'  checked {i:,}: {len(out):,} pass', flush=True)
        return out[:n], i

    chats = []

    def add(turns, kind, extra):
        ids, _ = encode_chat(tok, turns)
        if len(ids) <= args.max_tokens:
            chats.append(dict(kind=kind, turns=turns, **extra))
            return True
        return False

    need_filter = want['write'] + want['error'] * 3 + want['followup'] * 2
    good, used = checked(filt, need_filter)
    good_pre, _ = checked(pre, want['complete'])
    print(f'{len(good):,} checked Filter tasks, {len(good_pre):,} checked Prefill tasks',
          flush=True)
    for t in good_pre:
        add(complete_chat(t, rng), 'complete', {'source': t['id']})
    pool = list(good)
    counts = {'write': 0, 'error': 0, 'followup': 0}
    # Off-by-one bugs apply to few solutions (a range over len() or up to b + 1): give
    # those solutions to them first, up to 40% of the error chats.
    off_cap = round(0.4 * want['error'])
    candidates = [t for t in pool if OFF_BY_ONE_SITE.search(t['solution'])]
    used = set()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for t, res in ex.map(lambda t: (t, error_chat(t, random.Random(f'{args.seed}:obo:{t["id"]}'),
                                                       OFF_BY_ONE)), candidates):
            if res and counts['error'] < off_cap:
                turns, sub = res
                if add(turns, 'error', {'source': t['id'], 'variant': sub}):
                    counts['error'] += 1
                    used.add(t['id'])
        print(f'off-by-one: {counts["error"]:,} chats from {len(candidates):,} candidates',
              flush=True)
        pool = [t for t in pool if t['id'] not in used]
        for kind, fn in (('error', error_chat), ('followup', followup_chat)):
            results = ex.map(lambda t: (t, fn(t, random.Random(f'{args.seed}:{kind}:{t["id"]}'))),
                             pool)
            rest = []
            for t, res in results:
                if res and counts[kind] < want[kind]:
                    turns, sub = res
                    if add(turns, kind, {'source': t['id'], 'variant': sub}):
                        counts[kind] += 1
                        continue
                rest.append(t)
            pool = rest
            print(f'{kind}: {counts[kind]:,} chats', flush=True)
    for t in pool[:want['write']]:
        if add(write_chat(t, rng), 'write', {'source': t['id']}):
            counts['write'] += 1
    print(f'write: {counts["write"]:,} chats, complete: {len(good_pre):,}', flush=True)

    n_general = args.chats - sum(1 for c in chats)
    general = general_chats()
    rng.shuffle(general)
    added = 0
    for turns, source in general:
        if added >= n_general:
            break
        if add(turns, 'general', {'source': source}):
            added += 1
    print(f'general: {added:,} chats from {len(general):,} candidates', flush=True)

    rng.shuffle(chats)
    val = [c for c in chats if zlib.crc32(c['source'].encode()) % 1000 < args.val_share * 1000
           and c['kind'] != 'general'] + \
        [c for c in chats if c['kind'] == 'general'][:round(args.val_share * added)]
    val_ids = {id(c) for c in val}
    train = [c for c in chats if id(c) not in val_ids]
    for name, items in (('chats_train.jsonl', train), ('chats_val.jsonl', val)):
        with open(os.path.join(DATA, name), 'w', encoding='utf-8', newline='\n') as f:
            for c in items:
                f.write(json.dumps(c) + '\n')
    kinds = {}
    for c in train:
        kinds[c['kind']] = kinds.get(c['kind'], 0) + 1
    print(f'wrote {len(train):,} training and {len(val):,} validation chats: {kinds}')


if __name__ == '__main__':
    sys.exit(main())

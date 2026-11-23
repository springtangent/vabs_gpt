"""Sampling from a chat checkpoint: completions, chat replies and code checks.

``Generator`` keeps one inference ``ChatRunner`` per window length (128, 256,
512, ... up to the context) and per batch size, and runs each new token on the
shortest window that holds the longest sequence: RoPE positions start at 0 in
every sequence, so a short window gives the same logits as the full one.
There is no KV cache; every token reruns its window.
"""
import array
import heapq
import ast
import io
import math
import random
import tokenize

from .data import prompt_ids
from .model import ChatRunner
from .tokenizer import END, EOT

BUCKETS = (128, 256, 512, 1024, 2048, 4096)


class Generator:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.runners = {}

    def release(self):
        for r in self.runners.values():
            r.release()
        self.runners.clear()

    def _runner(self, B, need):
        ctx = self.model.cfg.ctx
        T = next((t for t in BUCKETS if t >= need and t <= ctx), ctx)
        key = (B, T)
        if key not in self.runners:
            self.runners[key] = ChatRunner(self.model, B, 'infer', T=T)
        return self.runners[key]

    def generate(self, prompts, max_new=200, temperature=0.8, top_k=40, stop=(END, EOT),
                 rng=None, stop_fn=None):
        """Continue each id list in ``prompts``; return the new ids of each (without stop ids).

        ``stop_fn(b, new_ids)`` may end sequence b early (e.g. at the end of a function).
        """
        rng = rng or random.Random()
        ctx = self.model.cfg.ctx
        seqs = [list(p)[-ctx:] for p in prompts]
        out = [[] for _ in prompts]
        done = [False] * len(prompts)
        B = len(prompts)
        for _ in range(max_new):
            if all(done):
                break
            for b in range(B):   # keep the newest ctx tokens
                if len(seqs[b]) > ctx:
                    seqs[b] = seqs[b][-ctx:]
            runner = self._runner(B, max(len(s) for s in seqs))
            T = runner.T
            toks = array.array('I')
            for s in seqs:
                toks.extend(s + [EOT] * (T - len(s)))
            rows = runner.next_logits(toks, [len(s) - 1 for s in seqs])
            for b in range(B):
                if done[b]:
                    continue
                nxt = pick(rows[b], temperature, top_k, rng)
                if nxt in stop:
                    done[b] = True
                    continue
                seqs[b].append(nxt)
                out[b].append(nxt)
                if stop_fn and stop_fn(b, out[b]):
                    done[b] = True
        return out

    def complete(self, text, n=1, **kw):
        ids = [EOT] + self.tok.encode(text)
        return [self.tok.decode(o) for o in self.generate([ids] * n, **kw)]

    def reply(self, turns, **kw):
        """The assistant's next reply after ``turns`` (a list of ``(role, text)``)."""
        kw.setdefault('max_new', 400)
        return self.tok.decode(self.generate([prompt_ids(self.tok, turns)], **kw)[0])


def pick(logits, temperature, top_k, rng):
    if temperature <= 0:
        return max(range(len(logits)), key=logits.__getitem__)
    if top_k <= 0 or top_k >= len(logits):
        top = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    else:
        top = heapq.nlargest(top_k, range(len(logits)), key=logits.__getitem__)
    mx = logits[top[0]]
    return rng.choices(top, [math.exp((logits[i] - mx) / temperature) for i in top])[0]


# -- code completion checks ----------------------------------------------------------

CODE_PROMPTS = [
    'def is_even(n):\n    """Return True if n is even."""\n',
    'def reverse_string(s):\n    """Return s reversed."""\n',
    'def count_vowels(text):\n    """Count the vowels in text."""\n',
    'def factorial(n):\n    """Return n! for a non-negative integer n."""\n',
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    'def max_of_list(numbers):\n    """Return the largest number in a non-empty list."""\n',
    'def is_palindrome(word):\n    """Check whether word reads the same backwards."""\n',
    'def celsius_to_fahrenheit(c):\n',
    'def sum_of_squares(numbers):\n',
    'def remove_duplicates(items):\n    """Return the items without duplicates, keeping order."""\n',
    'def word_count(sentence):\n    """Return a dict mapping each word to how often it occurs."""\n',
    'import os\n\n\ndef list_python_files(folder):\n',
    'class Counter:\n    def __init__(self):\n',
    'def average(values):\n    if not values:\n',
    'def read_lines(path):\n    with open(path) as f:\n',
    'def merge_dicts(a, b):\n    """Return a new dict with the keys of a and b."""\n',
]


def function_end(text):
    """Index where a completed top-level block ends: the first non-blank line at
    indentation 0 after an indented line, or None if the block is still open."""
    seen_body = False
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            if line[0] in ' \t':
                seen_body = True
            elif seen_body:
                return pos
        pos += len(line)
    return None


def check_code(prompt, completion):
    """Indentation, brackets and whether a completed function was produced.

    Returns a dict of booleans: ``finished`` (the block ended before the token
    budget), ``parses`` (prompt + body is valid Python), ``brackets`` (every
    bracket closed, by the tokenizer module), ``indent`` (indentation is
    multiples of 4 spaces, no tabs).
    """
    end = function_end(prompt + completion)
    finished = end is not None
    body = (prompt + completion)[:end] if finished else prompt + completion
    try:
        ast.parse(body)
        parses = True
    except SyntaxError:
        parses = False
    depth, brackets = 0, True
    try:
        for t in tokenize.generate_tokens(io.StringIO(body).readline):
            if t.type == tokenize.OP and t.string in '([{':
                depth += 1
            elif t.type == tokenize.OP and t.string in ')]}':
                depth -= 1
                brackets = brackets and depth >= 0
    except (tokenize.TokenError, IndentationError, SyntaxError):
        brackets = False
    brackets = brackets and depth == 0
    indent = all((len(l) - len(l.lstrip(' '))) % 4 == 0 and not l.startswith('\t')
                 for l in body.splitlines() if l.strip())
    return {'finished': finished, 'parses': parses and finished, 'brackets': brackets,
            'indent': indent}


def code_checks(gen, samples=2, max_new=160, temperature=0.6, seed=0):
    """Fractions of the completions of ``CODE_PROMPTS`` that pass each of ``check_code``'s tests,
    and the completions themselves."""
    rng = random.Random(seed)
    tok = gen.tok
    prompts = [p for p in CODE_PROMPTS for _ in range(samples)]
    results, texts = {}, []
    for i in range(0, len(prompts), 8):
        batch = prompts[i:i + 8]
        ids = [[EOT] + tok.encode(p) for p in batch]

        def ended(b, out, batch=batch):
            return function_end(batch[b] + tok.decode(out)) is not None

        outs = gen.generate(ids, max_new=max_new, temperature=temperature, rng=rng,
                            stop_fn=ended)
        for p, o in zip(batch, outs):
            text = tok.decode(o)
            texts.append((p, text))
            for k, v in check_code(p, text).items():
                results[k] = results.get(k, 0) + int(v)
    return {k: v / len(prompts) for k, v in results.items()}, texts

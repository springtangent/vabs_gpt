"""Training data for the chat model: token streams, the pretraining mixture, chats.

Pretraining data (written by ``prepare``) is one file of ``uint16`` token ids per
source and split (``train_stack.bin``, ``val_fineweb.bin``, ...), documents
separated by ``EOT``, plus ``meta.json`` with the token counts. ``Mixture`` draws
training windows of ``T + 1`` tokens: every source is cut into non-overlapping
windows that are visited in a shuffled order (reshuffled each epoch of that
source), and sources take turns by smooth weighted round-robin, so every batch
holds the mix in exact proportion to the weights. Its state is a small dict
that checkpoints keep, so a resumed run continues the same sequence.

Chats are lists of ``(role, text)`` turns, roles ``user`` and ``assistant``.
``encode_chat`` lays them out as

    EOT USER <user text> END ASSISTANT <reply> END USER ... END

and marks the reply tokens and their END as scored: fine-tuning computes the
loss on assistant tokens only. ``pack_chats`` fills windows with whole chats.
"""
import array
import json
import mmap
import os
import random

IGNORE = 0xFFFFFFFF   # target that is not scored (same value as kernels.IGNORE)
from .tokenizer import ASSISTANT, END, EOT, USER

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DEFAULT_MIX = {'stack': 0.4, 'fineweb': 0.5, 'cosmo': 0.1}
CODE_SOURCES = ('stack',)


class TokenFile:
    """A read-only ``uint16`` token file, memory mapped."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, 'rb')
        size = os.path.getsize(path)
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ) if size else None
        self.tokens = memoryview(self._mm).cast('H') if size else memoryview(b'').cast('H')

    def __len__(self):
        return len(self.tokens)

    def window(self, start, n):
        """``n`` tokens from ``start`` as an ``array('I')``."""
        return array.array('I', self.tokens[start:start + n])

    def close(self):
        self.tokens.release()
        if self._mm:
            self._mm.close()
        self._f.close()


def parse_mix(text):
    """``"stack=0.4,fineweb=0.5"`` -> dict of weights."""
    mix = {}
    for part in filter(None, text.split(',')):
        name, _, w = part.partition('=')
        mix[name.strip()] = float(w)
    if not mix or any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError(f'bad mix {text!r}')
    return mix


class Mixture:
    """Training windows from several sources in fixed proportion."""

    def __init__(self, data_dir, mix, T, seed=0, state=None):
        self.T, self.seed = T, seed
        self.weights = {s: w for s, w in mix.items() if w > 0}
        self.files = {s: TokenFile(os.path.join(data_dir, f'train_{s}.bin')) for s in self.weights}
        for s, f in self.files.items():
            if len(f) < T + 1:
                raise ValueError(f'{f.path} has fewer than {T + 1} tokens')
        state = state or {}
        self.cursor = dict(state.get('cursor', {}))
        self.epoch = dict(state.get('epoch', {}))
        self.credit = dict(state.get('credit', {}))
        self._order = {}
        for s in self.weights:
            self.cursor.setdefault(s, 0)
            self.epoch.setdefault(s, 0)
            self.credit.setdefault(s, 0.0)

    def windows(self, s):
        return (len(self.files[s]) - 1) // self.T

    def _perm(self, s):
        key = (s, self.epoch[s])
        if key not in self._order:
            order = list(range(self.windows(s)))
            random.Random(f'{self.seed}:{s}:{self.epoch[s]}').shuffle(order)
            self._order = {key: order} | {k: v for k, v in self._order.items() if k[0] != s}
        return self._order[key]

    def next_source(self):
        total = sum(self.weights.values())
        for s, w in self.weights.items():
            self.credit[s] += w
        s = max(self.weights, key=lambda k: (self.credit[k], k))
        self.credit[s] -= total
        return s

    def sequence(self):
        """``(inp, tgt, source)``: T input ids and the T next ids."""
        s = self.next_source()
        perm = self._perm(s)
        if self.cursor[s] >= len(perm):
            self.epoch[s] += 1
            self.cursor[s] = 0
            perm = self._perm(s)
        w = perm[self.cursor[s]]
        self.cursor[s] += 1
        win = self.files[s].window(w * self.T, self.T + 1)
        return win[:-1], win[1:], s

    def batch(self, B):
        inp, tgt, counts = array.array('I'), array.array('I'), {}
        for _ in range(B):
            i, t, s = self.sequence()
            inp.extend(i)
            tgt.extend(t)
            counts[s] = counts.get(s, 0) + 1
        return inp, tgt, counts

    def state(self):
        return {'cursor': self.cursor, 'epoch': self.epoch, 'credit': self.credit}

    def epochs(self):
        """Fraction of each source seen so far."""
        return {s: self.epoch[s] + self.cursor[s] / max(1, self.windows(s)) for s in self.weights}


def val_windows(path, T, max_tokens=0):
    """Non-overlapping ``(inp, tgt)`` windows from the start of a validation file."""
    f = TokenFile(path)
    n = len(f) if not max_tokens else min(len(f), max_tokens + 1)
    out = [(f.window(k, T + 1)[:-1], f.window(k, T + 1)[1:]) for k in range(0, n - T, T)]
    f.close()
    return out


def read_meta(data_dir):
    with open(os.path.join(data_dir, 'meta.json'), encoding='utf-8') as f:
        return json.load(f)


# -- chats ----------------------------------------------------------------------

def encode_chat(tok, turns):
    """``(ids, scored)`` of a chat; ``scored[k]`` is True for assistant tokens and their END.

    Two more roles make a plain completion instead of a chat: ``prompt`` (text, not
    scored) and ``completion`` (text and an ``<|endoftext|>``, scored), e.g. a function
    stub and its body.
    """
    ids, scored = [EOT], [False]
    for role, text in turns:
        if role == 'prompt':
            part = tok.encode(text)
            ids += part
            scored += [False] * len(part)
        elif role == 'completion':
            part = tok.encode(text) + [EOT]
            ids += part
            scored += [True] * len(part)
        elif role == 'user':
            part = [USER] + tok.encode(text) + [END]
            ids += part
            scored += [False] * len(part)
        elif role == 'assistant':
            reply = tok.encode(text) + [END]
            ids += [ASSISTANT] + reply
            scored += [False] + [True] * len(reply)
        else:
            raise ValueError(f'unknown role {role!r}')
    return ids, scored


def prompt_ids(tok, turns):
    """Token ids that ask for the next assistant reply after ``turns``."""
    ids, _ = encode_chat(tok, turns)
    return ids + [ASSISTANT]


def sft_example(ids, scored, T):
    """One window: ``inp`` (T ids, padded with EOT) and ``tgt`` (IGNORE where not scored)."""
    ids, scored = ids[:T + 1], scored[:T + 1]
    inp = array.array('I', ids[:-1] + [EOT] * (T + 1 - len(ids)))
    tgt = array.array('I', [ids[k + 1] if scored[k + 1] else IGNORE for k in range(len(ids) - 1)]
                      + [IGNORE] * (T + 1 - len(ids)))
    return inp, tgt


def pack_chats(examples, T, seed=0):
    """Pack ``(ids, scored)`` chats into windows of T tokens without splitting any chat.

    Chats longer than T + 1 tokens are dropped. Returns a list of ``(inp, tgt)``.
    """
    rng = random.Random(seed)
    items = [e for e in examples if len(e[0]) <= T + 1]
    rng.shuffle(items)
    windows, cur_ids, cur_scored = [], [], []
    for ids, scored in items:
        if len(cur_ids) + len(ids) > T + 1:
            windows.append(sft_example(cur_ids, cur_scored, T))
            cur_ids, cur_scored = [], []
        cur_ids += ids
        cur_scored += scored
    if cur_ids:
        windows.append(sft_example(cur_ids, cur_scored, T))
    return windows


def read_chats(path):
    """Chats from a JSON-lines file: ``{"turns": [[role, text], ...], ...}`` per line."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append((d, [tuple(t) for t in d['turns']]))
    return out

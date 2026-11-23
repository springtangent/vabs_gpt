"""Byte-level BPE tokenizer in pure Python: training, encoding and decoding.

Ids 0-255 are the byte values, then four special tokens, then the merges in the
order they were learned. Special tokens never come out of ``encode``: text that
spells ``<|user|>`` is encoded as ordinary bytes, so user input cannot forge a
turn boundary.

Pretokenization (``PATTERN``) splits text before BPE, so no token crosses its
boundaries. It follows GPT-4's split, with changes for code:

- words start with a letter or underscore and continue with letters, digits
  and underscores (``__init__``, ``x_1``, ``utf8``), optionally preceded by one space or punctuation character (``(self``, ``.append``);
- numbers are split into groups of at most three digits, the first with an
  optional leading space;
- runs of punctuation keep a following line break (``):\\n``);
- line breaks group together, and a run of spaces before a word leaves its last
  space to the word, so indentation (``"       "`` + ``" return"``) is a token
  of its own and keeps its exact width.

``Tokenizer.train`` counts pretokens in a text sample and runs the usual
greedy merge loop on the counts, with incremental pair counts.
"""
import heapq
import json
import re

PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\w]?[^\W\d]\w*| ?\d{1,3}| ?[^\s\w]+[\r\n]*"""
           r"""|\s*[\r\n]+|\s+(?!\S)|\s+""")
SPECIALS = ('<|endoftext|>', '<|user|>', '<|assistant|>', '<|end|>')
EOT, USER, ASSISTANT, END = 256, 257, 258, 259
FIRST_MERGE = 256 + len(SPECIALS)


class Tokenizer:
    def __init__(self, merges, vocab_size=None):
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = vocab_size or FIRST_MERGE + len(self.merges)
        if FIRST_MERGE + len(self.merges) > self.vocab_size:
            raise ValueError('more merges than the vocabulary holds')
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.pieces = [bytes([i]) for i in range(256)] + [s.encode() for s in SPECIALS]
        for a, b in self.merges:
            self.pieces.append(self.pieces[a] + self.pieces[b])
        self.pieces += [b''] * (self.vocab_size - len(self.pieces))   # unused ids
        self.split = re.compile(PATTERN).findall
        self._cache = {}

    # -- encoding -----------------------------------------------------------

    def _bpe(self, piece):
        ids = list(piece)
        ranks = self.ranks
        while len(ids) > 1:
            best, at = None, -1
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            new, a, b = FIRST_MERGE + best, ids[at], ids[at + 1]
            out, i = ids[:at], at
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode(self, text):
        """Token ids of ``text`` (no special tokens)."""
        cache, out = self._cache, []
        if len(cache) > 500_000:
            cache.clear()
        for piece in self.split(text):
            ids = cache.get(piece)
            if ids is None:
                ids = cache[piece] = self._bpe(piece.encode('utf-8'))
            out.extend(ids)
        return out

    def decode_bytes(self, ids):
        return b''.join(self.pieces[i] for i in ids)

    def decode(self, ids):
        return self.decode_bytes(ids).decode('utf-8', errors='replace')

    def show(self, text):
        """The tokens of ``text`` as a list of strings, for inspecting the split."""
        return [self.pieces[i].decode('utf-8', errors='replace') for i in self.encode(text)]

    # -- persistence ----------------------------------------------------------

    def save(self, path):
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump({'pattern': PATTERN, 'specials': SPECIALS, 'vocab_size': self.vocab_size,
                       'merges': self.merges}, f)

    @staticmethod
    def load(path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        if d['pattern'] != PATTERN or tuple(d['specials']) != SPECIALS:
            raise ValueError(f'{path} was trained with another pretokenizer or special tokens')
        return Tokenizer(d['merges'], d['vocab_size'])

    # -- training -------------------------------------------------------------

    @staticmethod
    def count(texts, counts=None):
        """Add the pretoken counts of ``texts`` to ``counts`` (a dict str -> int)."""
        counts = {} if counts is None else counts
        split = re.compile(PATTERN).findall
        for text in texts:
            for piece in split(text):
                counts[piece] = counts.get(piece, 0) + 1
        return counts

    @staticmethod
    def train(counts, vocab_size, min_count=2, log=None):
        """Learn ``vocab_size - 260`` merges from pretoken counts.

        Pretokens seen fewer than ``min_count`` times are left out (they are
        mostly one-off identifiers and numbers). Ties between equally frequent
        pairs go to the smaller pair of ids, so training is deterministic.
        """
        n_merges = vocab_size - FIRST_MERGE
        merged = {}
        for piece, c in counts.items():
            if c >= min_count:
                b = piece.encode('utf-8')
                if len(b) > 1:
                    merged[b] = merged.get(b, 0) + c
        words = [list(b) for b in merged]
        freqs = list(merged.values())
        del merged
        pair_count, where = {}, {}
        for w, (ids, f) in enumerate(zip(words, freqs)):
            for p in zip(ids, ids[1:]):
                pair_count[p] = pair_count.get(p, 0) + f
                where.setdefault(p, set()).add(w)
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)
        merges = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) != -neg:
                continue   # stale entry
            if -neg < 1:
                break
            new = FIRST_MERGE + len(merges)
            merges.append(pair)
            a, b = pair
            changed = set()
            for w in list(where.get(pair, ())):
                ids, f = words[w], freqs[w]
                out, i, n = [], 0, len(ids)
                while i < n:
                    if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                for p in zip(ids, ids[1:]):
                    pair_count[p] -= f
                    changed.add(p)
                for p in zip(out, out[1:]):
                    pair_count[p] = pair_count.get(p, 0) + f
                    changed.add(p)
                old_pairs, new_pairs = set(zip(ids, ids[1:])), set(zip(out, out[1:]))
                for p in old_pairs - new_pairs:
                    s = where.get(p)
                    if s is not None:
                        s.discard(w)
                for p in new_pairs - old_pairs:
                    where.setdefault(p, set()).add(w)
                words[w] = out
            pair_count.pop(pair, None)
            where.pop(pair, None)
            for p in changed:
                c = pair_count.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_count.pop(p, None)
            if log and len(merges) % 1000 == 0:
                log(f'  {len(merges):,} merges, last count {-neg:,}')
        return Tokenizer(merges, vocab_size)

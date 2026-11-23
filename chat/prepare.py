"""Train the tokenizer and tokenize the pretraining mix (needs ``pyarrow``).

    .ptvenv\\Scripts\\python.exe -m vabs_gpt.chat.prepare tokenizer   # -> data\\tokenizer.json
    .ptvenv\\Scripts\\python.exe -m vabs_gpt.chat.prepare check       # how it splits code etc.
    .ptvenv\\Scripts\\python.exe -m vabs_gpt.chat.prepare tokens --tokens 500e6

``tokenizer`` trains the 20,000-token byte-level BPE on a sample of the actual
mix: the sample's bytes are split between the sources in the mix's proportions.

``tokens`` writes ``train_<source>.bin`` and ``val_<source>.bin`` (uint16 ids,
documents separated by EOT) and ``meta.json``. Each source is tokenized until it
reaches its share of ``--tokens`` (default mix 40% Stack-Edu, 50% FineWeb-Edu,
10% Cosmopedia), so the proportions are measured after tokenization.
Validation documents are held out by hash: whole repositories for code (so
nearby files of one project stay on one side), documents for prose. Code
keeps complete short files (``download`` only fetched files up to 8 KB).
Cosmopedia keeps only documents that fit the context window.
"""
import argparse
import array
import glob
import json
import multiprocessing
import os
import sys
import time
import zlib

from .data import DATA, DEFAULT_MIX
from .tokenizer import EOT, Tokenizer

RAW = os.path.join(DATA, 'raw')
TOKENIZER = os.path.join(DATA, 'tokenizer.json')
VOCAB = 20000
CTX = 1024
VAL_TOKENS = {'stack': 2_000_000, 'fineweb': 2_000_000, 'cosmo': 1_000_000}


# -- raw documents ----------------------------------------------------------------

def _crc(text):
    return zlib.crc32(text.encode('utf-8', errors='replace'))


def fineweb_docs(split):
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(os.path.join(RAW, 'fineweb', '*.parquet'))):
        for batch in pq.ParquetFile(path).iter_batches(columns=['text', 'id'], batch_size=4096):
            texts, ids = batch.column(0).to_pylist(), batch.column(1).to_pylist()
            for text, doc_id in zip(texts, ids):
                if (_crc(doc_id) % 1000 < 3) == (split == 'val') and text:
                    yield text.strip()


def stack_docs(split):
    path = os.path.join(RAW, 'stack', 'contents.jsonl')
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if (_crc(d['repo']) % 100 == 0) != (split == 'val'):
                continue
            text = d['text'].replace('\r\n', '\n').replace('\r', '\n').strip('\n')
            if text.strip() and '\0' not in text:
                yield text + '\n'


def cosmo_docs(split, max_chars=5000):
    """Round-robin over the Cosmopedia shards, so every subset is represented."""
    import pyarrow.parquet as pq
    iters = [pq.ParquetFile(p).iter_batches(columns=['text'], batch_size=1024)
             for p in sorted(glob.glob(os.path.join(RAW, 'cosmo', '*.parquet')))]
    while iters:
        alive = []
        for it in iters:
            batch = next(it, None)
            if batch is None:
                continue
            alive.append(it)
            for text in batch.column(0).to_pylist():
                if text and len(text) <= max_chars \
                        and (_crc(text[:256]) % 100 == 0) == (split == 'val'):
                    yield text.strip()
        iters = alive


SOURCES = {'stack': stack_docs, 'fineweb': fineweb_docs, 'cosmo': cosmo_docs}
MAX_DOC_TOKENS = {'cosmo': CTX - 1}   # "short explanations that fit the context window"


# -- tokenizer ------------------------------------------------------------------

def sample_texts(source, n_bytes):
    out, total = [], 0
    for text in SOURCES[source]('train'):
        out.append(text)
        total += len(text.encode('utf-8'))
        if total >= n_bytes:
            break
    return out, total


def _count(texts):
    return Tokenizer.count(texts)


def cmd_tokenizer(args):
    mix = DEFAULT_MIX
    counts, t0 = {}, time.time()
    with multiprocessing.Pool(args.workers) as pool:
        for source, w in mix.items():
            texts, n = sample_texts(source, int(args.sample_bytes * w / sum(mix.values())))
            chunks = [texts[i:i + 200] for i in range(0, len(texts), 200)]
            for part in pool.imap_unordered(_count, chunks):
                for k, v in part.items():
                    counts[k] = counts.get(k, 0) + v
            print(f'{source}: {len(texts):,} documents, {n / 1e6:.1f} MB', flush=True)
    print(f'{len(counts):,} distinct pretokens ({time.time() - t0:.0f} s); training...', flush=True)
    t0 = time.time()
    tok = Tokenizer.train(counts, args.vocab, min_count=args.min_count,
                          log=lambda s: print(s, flush=True))
    print(f'{len(tok.merges):,} merges in {time.time() - t0:.0f} s', flush=True)
    os.makedirs(DATA, exist_ok=True)
    tok.save(TOKENIZER)
    print(f'wrote {TOKENIZER}')
    check(tok)


CHECKS = {
    'python function': '''def mean(values):
    """Return the arithmetic mean of a list of numbers."""
    if not values:
        raise ValueError("mean() of an empty list")
    total = 0
    for v in values:
        total += v
    return total / len(values)
''',
    'traceback': '''Traceback (most recent call last):
  File "main.py", line 12, in <module>
    print(items[3])
IndexError: list index out of range
''',
    'markdown code block': '''You can reverse a list with slicing:

```python
numbers = [1, 2, 3]
print(numbers[::-1])  # [3, 2, 1]
```
''',
    'class': '''class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)
''',
}


def check(tok, sources=True):
    for name, text in CHECKS.items():
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        print(f'\n== {name}: {len(text)} chars -> {len(ids)} tokens '
              f'({len(text.encode()) / len(ids):.2f} bytes/token)')
        print('|'.join(tok.pieces[i].decode('utf-8', errors='replace') for i in ids))
    if sources:
        print()
        for source in SOURCES:
            try:
                texts, n = sample_texts(source, 2_000_000)
            except FileNotFoundError:
                continue
            ids = sum(len(tok.encode(t)) for t in texts)
            print(f'{source}: {n / ids:.2f} bytes/token on {n / 1e6:.1f} MB')


def cmd_check(args):
    check(Tokenizer.load(TOKENIZER))


# -- tokenizing the mix -------------------------------------------------------------

_tok = None


def _init_worker(path):
    global _tok
    _tok = Tokenizer.load(path)


def _encode(job):
    texts, max_tokens = job
    out, n_docs, n_bytes = array.array('H'), 0, 0
    for text in texts:
        ids = _tok.encode(text)
        if max_tokens and len(ids) > max_tokens:
            continue
        out.extend(ids)
        out.append(EOT)
        n_docs += 1
        n_bytes += len(text.encode('utf-8'))
    return out.tobytes(), n_docs, n_bytes


def _jobs(docs, max_tokens, size=64):
    chunk = []
    for text in docs:
        chunk.append(text)
        if len(chunk) == size:
            yield chunk, max_tokens
            chunk = []
    if chunk:
        yield chunk, max_tokens


def tokenize_source(pool, source, split, target, out_path):
    max_tokens = MAX_DOC_TOKENS.get(source, 0)
    n_tok = n_docs = n_bytes = 0
    t0 = time.time()
    with open(out_path, 'wb') as out:
        for data, docs, nb in pool.imap(_encode, _jobs(SOURCES[source](split), max_tokens),
                                        chunksize=4):
            take = min(len(data) // 2, target - n_tok)
            out.write(data[:2 * take])
            n_tok += take
            n_docs += docs
            n_bytes += nb
            if n_tok >= target:
                break
            if n_docs % 50_000 < docs:
                print(f'  {source}/{split}: {n_tok / 1e6:,.1f}M tokens, {n_docs:,} documents '
                      f'({n_tok / (time.time() - t0) / 1e3:,.0f}k tokens/s)', flush=True)
    print(f'{out_path}: {n_tok:,} tokens, {n_docs:,} documents, '
          f'{n_bytes / max(n_tok, 1):.2f} bytes/token', flush=True)
    if n_tok < target:
        print(f'  WARNING: {source}/{split} ran out of data at {n_tok:,} of {target:,} tokens',
              flush=True)
    return {'tokens': n_tok, 'documents': n_docs, 'bytes': n_bytes}


def cmd_tokens(args):
    tok = Tokenizer.load(TOKENIZER)
    mix = DEFAULT_MIX
    total_w = sum(mix.values())
    meta = {'tokenizer': os.path.basename(TOKENIZER), 'vocab': tok.vocab_size, 'files': {}}
    only = set(filter(None, args.only.split(',')))
    out_dir = args.out_dir or DATA
    os.makedirs(out_dir, exist_ok=True)
    if out_dir != DATA:   # pretrain reads the tokenizer from its --data directory
        import shutil
        shutil.copyfile(TOKENIZER, os.path.join(out_dir, 'tokenizer.json'))
    path = os.path.join(out_dir, 'meta.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            meta['files'] = json.load(f).get('files', {})
    with multiprocessing.Pool(args.workers, _init_worker, (TOKENIZER,)) as pool:
        for source, w in mix.items():
            if only and source not in only:
                continue
            for split, target in (('val', VAL_TOKENS[source]),
                                  ('train', int(args.tokens * w / total_w))):
                name = f'{split}_{source}.bin'
                meta['files'][name] = tokenize_source(pool, source, split, target,
                                                      os.path.join(out_dir, name))
                with open(path, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(meta, f, indent=1)
    train = {k: v['tokens'] for k, v in meta['files'].items() if k.startswith('train_')}
    total = sum(train.values())
    print('measured training mix: ' + ', '.join(f'{k[6:-4]} {v / total:.1%}'
                                                for k, v in train.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('tokenizer', help='train the BPE tokenizer on a sample of the mix')
    p.add_argument('--sample-bytes', type=float, default=80e6)
    p.add_argument('--vocab', type=int, default=VOCAB)
    p.add_argument('--min-count', type=int, default=2)
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 4))
    sub.add_parser('check', help='show how the tokenizer splits code, tracebacks and Markdown')
    p = sub.add_parser('tokens', help='tokenize the training and validation data')
    p.add_argument('--tokens', type=float, default=500e6, help='training tokens in total')
    p.add_argument('--only', default='', help='comma-separated sources to (re)do')
    p.add_argument('--out-dir', default='', help='write here instead of data/ (pass it to '
                   'pretrain as --data)')
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 4))
    args = parser.parse_args(argv)
    {'tokenizer': cmd_tokenizer, 'check': cmd_check, 'tokens': cmd_tokens}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())

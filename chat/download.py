"""Download the raw data for the chatbot: parquet shards and Stack-Edu file contents.

Standard library only. Files already present at their full size are skipped, so
an interrupted download continues where it stopped.

    python -m vabs_gpt.chat.download               # everything, ~4.5 GB of parquet
    python -m vabs_gpt.chat.download --stack 300000  # Stack-Edu: how many files to fetch

Sources (Hugging Face datasets; Stack-Edu lists only file ids, so the contents
come from Software Heritage's public S3 bucket):

- FineWeb-Edu: one shard of ``sample/10BT`` (about 0.7B tokens of English)
- Stack-Edu: one ``Python`` shard of the file list, then the short files
- Cosmopedia: khanacademy, openstax, wikihow, stanford and web_samples_v2 shards
- Smol-SmolTalk: one train shard and the test shard (general chats for fine-tuning)
- KodCode-V1: three shards of short Python tasks with tests (coding chats for fine-tuning)
"""
import argparse
import concurrent.futures
import gzip
import http.client
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'raw')
HF = 'https://huggingface.co/datasets/{repo}/resolve/main/{path}'

PARQUET = [
    ('HuggingFaceFW/fineweb-edu', 'sample/10BT/000_00000.parquet', 'fineweb'),
    ('HuggingFaceTB/stack-edu', 'Python/train-00000-of-00005.parquet', 'stack'),
    ('HuggingFaceTB/cosmopedia', 'data/khanacademy/train-00000-of-00001.parquet', 'cosmo'),
    ('HuggingFaceTB/cosmopedia', 'data/openstax/train-00000-of-00002.parquet', 'cosmo'),
    ('HuggingFaceTB/cosmopedia', 'data/wikihow/train-00000-of-00002.parquet', 'cosmo'),
    ('HuggingFaceTB/cosmopedia', 'data/stanford/train-00000-of-00013.parquet', 'cosmo'),
    ('HuggingFaceTB/cosmopedia', 'data/web_samples_v2/train-00000-of-00118.parquet', 'cosmo'),
    ('HuggingFaceTB/smol-smoltalk', 'data/train-00000-of-00004.parquet', 'smoltalk'),
    ('HuggingFaceTB/smol-smoltalk', 'data/test-00000-of-00001.parquet', 'smoltalk'),
    # the 'Filter' subset ("Write a python function to ...") and 'Prefill' (complete a stub)
    ('KodCode/KodCode-V1', 'data/train-00006-of-00015.parquet', 'kodcode'),
    ('KodCode/KodCode-V1', 'data/train-00007-of-00015.parquet', 'kodcode'),
    ('KodCode/KodCode-V1', 'data/train-00013-of-00015.parquet', 'kodcode'),
]

S3_HOST = 'softwareheritage.s3.amazonaws.com'
STACK_MAX_BYTES = 8000   # "complete short files": skip longer ones


def local_path(repo, path, kind):
    name = repo.split('/')[1] + '__' + path.replace('/', '__')
    return os.path.join(RAW, kind, name)


def remote_size(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as r:
        return int(r.headers.get('Content-Length') or r.headers.get('X-Linked-Size') or 0)


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    size = remote_size(url)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return False
    tmp = dest + '.part'
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={'Range': f'bytes={done}-'} if done else {})
    t0, last = time.time(), 0.0
    with urllib.request.urlopen(req) as r, open(tmp, 'ab' if done else 'wb') as f:
        if done and r.status != 206:   # the server ignored the range: start over
            f.seek(0)
            f.truncate()
            done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if time.time() - last > 5:
                last = time.time()
                rate = done / max(last - t0, 1e-3) / 1e6
                print(f'  {os.path.basename(dest)}: {done / 1e6:,.0f} / {size / 1e6:,.0f} MB '
                      f'({rate:.0f} MB/s)', flush=True)
    os.replace(tmp, dest)
    return True


def stack_candidates(limit, seed=0):
    """``(blob_id, encoding, repo, path)`` of short Python files from the Stack-Edu list."""
    import pyarrow.parquet as pq   # only needed here
    files = sorted(os.listdir(os.path.join(RAW, 'stack')))
    out = []
    for name in files:
        if not name.endswith('.parquet'):
            continue
        table = pq.read_table(os.path.join(RAW, 'stack', name),
                              columns=['blob_id', 'src_encoding', 'repo_name', 'path',
                                       'length_bytes'])
        cols = [table.column(c).to_pylist() for c in
                ('blob_id', 'src_encoding', 'repo_name', 'path', 'length_bytes')]
        for blob, enc, repo, path, n in zip(*cols):
            if 0 < n <= STACK_MAX_BYTES:
                out.append((blob, enc, repo, path))
    # the file list is not ordered by anything we care about; keep its order (stable)
    return out[:limit]


class _S3:
    """One keep-alive HTTPS connection per thread."""

    local = threading.local()

    @classmethod
    def get(cls, blob):
        for attempt in range(5):
            conn = getattr(cls.local, 'conn', None)
            if conn is None:
                conn = cls.local.conn = http.client.HTTPSConnection(S3_HOST, timeout=30)
            try:
                conn.request('GET', '/content/' + blob)
                r = conn.getresponse()
                data = r.read()
                if r.status == 200:
                    return gzip.decompress(data)
                if r.status == 404:
                    return None
                raise OSError(f'HTTP {r.status}')
            except (OSError, http.client.HTTPException):
                conn.close()
                cls.local.conn = None
                time.sleep(0.5 * 2 ** attempt)
        return None


def fetch_stack(limit, threads):
    """Write the file contents as JSON lines: {"repo", "path", "text"}."""
    dest = os.path.join(RAW, 'stack', 'contents.jsonl')
    have = set()
    if os.path.exists(dest):
        with open(dest, encoding='utf-8') as f:
            for line in f:
                try:
                    have.add(json.loads(line)['blob'])
                except (ValueError, KeyError):
                    pass   # a line cut off by an interrupted run
    todo = [c for c in stack_candidates(limit) if c[0] not in have]
    print(f'stack-edu: {len(have):,} files present, {len(todo):,} to fetch', flush=True)
    if not todo:
        return
    t0, n_ok, n_bytes = time.time(), 0, 0

    def one(c):
        blob, enc, repo, path = c
        raw = _S3.get(blob)
        if raw is None:
            return None
        try:
            text = raw.decode(enc or 'utf-8')
        except (LookupError, UnicodeDecodeError):
            text = raw.decode('utf-8', errors='replace')
        return json.dumps({'blob': blob, 'repo': repo, 'path': path, 'text': text})

    with open(dest, 'a', encoding='utf-8', newline='\n') as out, \
            concurrent.futures.ThreadPoolExecutor(threads) as pool:
        for i, line in enumerate(pool.map(one, todo, chunksize=16)):
            if line:
                out.write(line + '\n')
                n_ok += 1
                n_bytes += len(line)
            if (i + 1) % 5000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f'  stack-edu: {i + 1:,} / {len(todo):,} ({rate:.0f} files/s, '
                      f'{n_bytes / 1e6:,.0f} MB)', flush=True)
                out.flush()
    print(f'stack-edu: fetched {n_ok:,} files', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', default='', help='comma-separated kinds: fineweb,stack,cosmo,'
                        'smoltalk,kodcode (default: all)')
    parser.add_argument('--stack', type=int, default=420000,
                        help='Stack-Edu files to fetch (short files only)')
    parser.add_argument('--threads', type=int, default=48, help='parallel Stack-Edu requests')
    args = parser.parse_args()
    only = set(filter(None, args.only.split(',')))
    for repo, path, kind in PARQUET:
        if only and kind not in only:
            continue
        dest = local_path(repo, path, kind)
        print(f'{repo}/{path}', flush=True)
        if not fetch(HF.format(repo=repo, path=path), dest):
            print('  present', flush=True)
    if not only or 'stack' in only:
        fetch_stack(args.stack, args.threads)


if __name__ == '__main__':
    sys.exit(main())

"""Convert the cached TinyStories parquet shards into raw byte streams.

Writes ``train.bin`` and ``val.bin``: each story's UTF-8 bytes followed by a
0 byte, which the model uses as its end-of-story token (vocabulary = 256
byte values). Needs ``pyarrow``, which the training program itself does not.

    .ptvenv\\Scripts\\python.exe vabs_gpt\\prepare.py
"""
import argparse
import glob
import os

import pyarrow.parquet as pq

SNAPSHOT = os.path.expanduser(
    '~/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/*/data')


def convert(files, out_path):
    stories = 0
    with open(out_path, 'wb') as out:
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(columns=['text'], batch_size=8192):
                for text in batch.column(0).to_pylist():
                    data = text.strip().encode('utf-8').replace(b'\0', b'')
                    if data:
                        out.write(data + b'\0')
                        stories += 1
    print(f'{out_path}: {stories:,} stories, {os.path.getsize(out_path):,} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=None, help='directory with the parquet shards')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'data'))
    args = parser.parse_args()
    data_dir = args.data_dir or next(iter(glob.glob(SNAPSHOT)), None)
    if data_dir is None:
        raise SystemExit('TinyStories parquet not found; pass --data-dir '
                         '(download roneneldan/TinyStories from Hugging Face)')
    os.makedirs(args.out, exist_ok=True)
    for split, name in (('train', 'train.bin'), ('validation', 'val.bin')):
        files = sorted(glob.glob(os.path.join(data_dir, f'{split}-*.parquet')))
        if not files:
            raise SystemExit(f'no {split} shards in {data_dir}')
        convert(files, os.path.join(args.out, name))


if __name__ == '__main__':
    main()

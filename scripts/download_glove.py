#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys
from urllib.request import urlopen, Request

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
EXPECTED_SIZE = None  # we won't enforce size; just checksum per file if provided


def download(url: str, dest: str, chunk: int = 1 << 20):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dest, "wb") as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)


def main():
    parser = argparse.ArgumentParser(description="Download GloVe 6B embeddings (zip)")
    parser.add_argument("--outdir", default="data/embeddings", help="Output directory")
    parser.add_argument("--only100d", action="store_true", help="Extract only 100d file")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    zip_path = os.path.join(args.outdir, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {GLOVE_URL} -> {zip_path}")
        download(GLOVE_URL, zip_path)
    else:
        print(f"Found existing {zip_path}")

    # Extract
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if args.only100d:
            members = [m for m in zf.namelist() if m.endswith("glove.6B.100d.txt")]
        else:
            members = zf.namelist()
        zf.extractall(args.outdir, members=members)
        print(f"Extracted {len(members)} files to {args.outdir}")


if __name__ == "__main__":
    main()


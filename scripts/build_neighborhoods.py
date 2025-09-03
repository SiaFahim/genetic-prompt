#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings.neighborhoods import build_neighborhoods


def main():
    p = argparse.ArgumentParser(description="Build semantic neighborhoods from GloVe and vocab")
    p.add_argument("--glove", default="data/embeddings/glove.6B.100d.txt")
    p.add_argument("--token2id", default="data/embeddings/vocab/token2id.json")
    p.add_argument("--out", default="data/embeddings/neighborhoods.pkl")
    p.add_argument("--k", type=int, default=50)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_neighborhoods(args.glove, args.token2id, args.out, k=args.k)


if __name__ == "__main__":
    main()


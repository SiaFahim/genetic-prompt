#!/usr/bin/env python3
import argparse
import os
from src.embeddings.vocab_builder import build_vocab


def main():
    p = argparse.ArgumentParser(description="Build token vocabulary from GSM8K subsets")
    p.add_argument("--primary", default="data/gsm8k_primary_eval.jsonl")
    p.add_argument("--validation", default="data/gsm8k_validation.jsonl")
    p.add_argument("--final", default="data/gsm8k_final_test.jsonl")
    p.add_argument("--out", default="data/embeddings/vocab")
    p.add_argument("--max_tokens", type=int, default=20000)
    p.add_argument("--min_freq", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    build_vocab([args.primary, args.validation, args.final], args.out, max_tokens=args.max_tokens, min_freq=args.min_freq)


if __name__ == "__main__":
    main()


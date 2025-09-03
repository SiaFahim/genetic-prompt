#!/usr/bin/env python3
import argparse
import json
import os
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download GSM8K dataset to local cache")
    parser.add_argument("--out", default="data/gsm8k_raw", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading GSM8K dataset to {args.out} ...")
    dataset = load_dataset("openai/gsm8k", "main")
    dataset.save_to_disk(args.out)
    print("Done.")


if __name__ == "__main__":
    main()


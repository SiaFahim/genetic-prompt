#!/usr/bin/env python3
import argparse
import json
import os
import random
from datasets import load_from_disk


def subset(dataset, split_name, n, seed):
    rnd = random.Random(seed)
    indices = list(range(len(dataset[split_name])))
    rnd.shuffle(indices)
    pick = indices[:n]
    picked = [dataset[split_name][i] for i in pick]
    return picked


def extract_final(answer_text):
    import re
    m = re.search(r"####\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)", answer_text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            out = {
                "id": row.get("id", ""),
                "question": row["question"],
                "answer": row["answer"],
                "final_answer": extract_final(row["answer"]) 
            }
            f.write(json.dumps(out) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create evaluation subsets from local GSM8K cache")
    parser.add_argument("--data", default="data/gsm8k_raw", help="Path to local dataset cache")
    parser.add_argument("--out", default="data", help="Output directory for subsets")
    parser.add_argument("--primary", type=int, default=100)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--final", type=int, default=200)
    parser.add_argument("--seed_primary", type=int, default=42)
    parser.add_argument("--seed_validation", type=int, default=43)
    parser.add_argument("--seed_final", type=int, default=44)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_from_disk(args.data)

    primary = subset(ds, "test", args.primary, args.seed_primary)
    validation = subset(ds, "train", args.validation, args.seed_validation)

    # final non-overlapping from test
    rnd = random.Random(args.seed_final)
    test_indices = list(range(len(ds["test"])));
    rnd.shuffle(test_indices)
    chosen = []
    seen_ids = set([row.get("id", None) for row in primary if row.get("id", None) is not None])
    for i in test_indices:
        row = ds["test"][i]
        rid = row.get("id", None)
        if rid is None or rid not in seen_ids:
            chosen.append(row)
        if len(chosen) >= args.final:
            break

    save_jsonl(os.path.join(args.out, "gsm8k_primary_eval.jsonl"), primary)
    save_jsonl(os.path.join(args.out, "gsm8k_validation.jsonl"), validation)
    save_jsonl(os.path.join(args.out, "gsm8k_final_test.jsonl"), chosen)
    print("Saved subsets to:")
    print(os.path.join(args.out, "gsm8k_primary_eval.jsonl"))
    print(os.path.join(args.out, "gsm8k_validation.jsonl"))
    print(os.path.join(args.out, "gsm8k_final_test.jsonl"))


if __name__ == "__main__":
    main()


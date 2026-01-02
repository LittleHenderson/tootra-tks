#!/usr/bin/env python3
"""
Split 6400 Equation Corpus into Train/Val/Test

Stratified by operator to ensure balanced splits.
Outputs both JSONL and PT files for each split.

Usage:
    python scripts/split_equation_corpus.py
    python scripts/split_equation_corpus.py --seed 42 --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser(description="Split equation corpus")
    p.add_argument("--jsonl", default="tks_6400_complete_merged.jsonl",
                   help="Input JSONL file")
    p.add_argument("--meta", default="data/equation_embeddings/equation_embeddings_meta.json",
                   help="Metadata JSON with index_map")
    p.add_argument("--corpus-pt", default="data/equation_embeddings/equation_corpus.pt",
                   help="Full corpus PT file")
    p.add_argument("--out-dir", default="data/equation_embeddings/splits",
                   help="Output directory for splits")
    p.add_argument("--seed", type=int, default=7,
                   help="Random seed")
    p.add_argument("--train", type=float, default=0.8,
                   help="Train split ratio")
    p.add_argument("--val", type=float, default=0.1,
                   help="Validation split ratio")
    p.add_argument("--test", type=float, default=0.1,
                   help="Test split ratio")
    args = p.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        f"Splits must sum to 1.0, got {args.train + args.val + args.test}"

    print("=" * 70)
    print("EQUATION CORPUS SPLIT")
    print("=" * 70)

    # Load equations
    print(f"\nLoading equations from {args.jsonl}...")
    equations = load_jsonl(args.jsonl)
    print(f"  Loaded {len(equations)} equations")

    # Load metadata with index_map
    print(f"Loading metadata from {args.meta}...")
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    index_map = meta["index_map"]

    # Attach embedding index to each equation
    for eq in equations:
        key = f"{eq['left']}_{eq['operator']}_{eq['right']}"
        idx = index_map.get(key)
        if idx is None:
            raise ValueError(f"Missing index_map key: {key}")
        eq["embedding_index"] = idx

    # Stratify by operator
    by_op = defaultdict(list)
    for eq in equations:
        by_op[eq["operator"]].append(eq)

    print(f"\nOperator distribution:")
    for op, items in sorted(by_op.items()):
        print(f"  {op}: {len(items)}")

    # Split each operator group
    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}

    for op, items in by_op.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        # n_test = n - n_train - n_val  # remainder

        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])

    print(f"\nSplit sizes:")
    for split, items in splits.items():
        print(f"  {split}: {len(items)}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSONL files
    for split, items in splits.items():
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for eq in items:
                f.write(json.dumps(eq) + "\n")
        print(f"Wrote {out_path}")

    # Load full corpus PT and create split PT files
    print(f"\nLoading corpus PT from {args.corpus_pt}...")
    corpus = torch.load(args.corpus_pt, map_location="cpu", weights_only=False)

    for split, items in splits.items():
        indices = torch.tensor([eq["embedding_index"] for eq in items], dtype=torch.long)

        # Index into each tensor in corpus
        split_corpus = {}
        for k, v in corpus.items():
            if isinstance(v, torch.Tensor):
                split_corpus[k] = v[indices]
            else:
                split_corpus[k] = v  # Keep non-tensor items as-is

        out_pt = out_dir / f"{split}.pt"
        torch.save(split_corpus, out_pt)
        print(f"Wrote {out_pt} ({len(indices)} samples)")

    # Verify operator distribution in each split
    print(f"\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    for split, items in splits.items():
        op_counts = defaultdict(int)
        for eq in items:
            op_counts[eq["operator"]] += 1
        print(f"\n{split.upper()} operator distribution:")
        for op, count in sorted(op_counts.items()):
            pct = count / len(items) * 100
            print(f"  {op}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

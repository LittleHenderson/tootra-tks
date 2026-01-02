#!/usr/bin/env python3
"""
Create train/holdout splits from JSONL data.

Ensures holdout contains different equations than train set
to test true generalization (not just augmentation variations).

Usage:
    python scripts/create_holdout_split.py --input data.jsonl --train train.jsonl --holdout holdout.jsonl --holdout-ratio 0.15
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def extract_equation_key(entry: Dict) -> str:
    """Extract a unique key for the equation (independent of aug_type)."""
    # Use expr_elements_base if available, else expr_elements
    elements = entry.get('expr_elements_base', entry.get('expr_elements', []))
    return '_'.join(sorted(elements))


def split_by_equations(
    entries: List[Dict],
    holdout_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """
    Split data by unique equations, not by individual entries.

    This ensures holdout contains equations not seen during training,
    providing a cleaner generalization test.
    """
    random.seed(seed)

    # Group entries by equation
    equation_groups = defaultdict(list)
    for entry in entries:
        key = extract_equation_key(entry)
        equation_groups[key].append(entry)

    # Get unique equation keys and shuffle
    equation_keys = list(equation_groups.keys())
    random.shuffle(equation_keys)

    # Split by equations
    num_holdout_equations = max(1, int(len(equation_keys) * holdout_ratio))
    holdout_equation_keys = set(equation_keys[:num_holdout_equations])
    train_equation_keys = set(equation_keys[num_holdout_equations:])

    # Build splits
    train_entries = []
    holdout_entries = []

    for key in train_equation_keys:
        train_entries.extend(equation_groups[key])

    for key in holdout_equation_keys:
        holdout_entries.extend(equation_groups[key])

    return train_entries, holdout_entries, len(train_equation_keys), len(holdout_equation_keys)


def main():
    parser = argparse.ArgumentParser(description='Create train/holdout splits')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--train', '-t', required=True, help='Output train JSONL file')
    parser.add_argument('--holdout', '-o', required=True, help='Output holdout JSONL file')
    parser.add_argument('--holdout-ratio', type=float, default=0.15, help='Holdout ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load entries
    entries = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {args.input}")

    # Split
    train_entries, holdout_entries, num_train_eq, num_holdout_eq = split_by_equations(
        entries, args.holdout_ratio, args.seed
    )

    # Save train split
    with open(args.train, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + '\n')

    # Save holdout split
    with open(args.holdout, 'w', encoding='utf-8') as f:
        for entry in holdout_entries:
            f.write(json.dumps(entry) + '\n')

    # Print statistics
    print(f"\nSplit Statistics:")
    print(f"  Total entries: {len(entries)}")
    print(f"  Unique equations: {num_train_eq + num_holdout_eq}")
    print(f"\nTrain split:")
    print(f"  Equations: {num_train_eq}")
    print(f"  Entries: {len(train_entries)}")
    print(f"  Saved to: {args.train}")
    print(f"\nHoldout split:")
    print(f"  Equations: {num_holdout_eq}")
    print(f"  Entries: {len(holdout_entries)}")
    print(f"  Saved to: {args.holdout}")

    # Count aug types in each split
    print(f"\nAug-type distribution:")
    for name, data in [("Train", train_entries), ("Holdout", holdout_entries)]:
        aug_counts = defaultdict(int)
        for entry in data:
            aug_counts[entry.get('aug_type', 'unknown')] += 1
        print(f"  {name}:")
        for aug_type, count in sorted(aug_counts.items()):
            print(f"    {aug_type}: {count}")


if __name__ == '__main__':
    main()

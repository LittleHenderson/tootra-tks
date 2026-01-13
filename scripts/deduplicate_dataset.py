#!/usr/bin/env python3
"""
Deduplicate TKS training dataset to reduce memorization.

Usage:
    python scripts/deduplicate_dataset.py
"""

import json
from collections import OrderedDict
from pathlib import Path

def deduplicate(input_path: str, output_path: str) -> dict:
    """Remove exact duplicate samples from dataset."""
    seen = OrderedDict()
    total = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
                # Use story/input as key
                key = item.get('story', item.get('input', item.get('text', '')))
                if key and key not in seen:
                    seen[key] = item
            except json.JSONDecodeError:
                pass

    # Write deduplicated
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in seen.values():
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    unique = len(seen)
    duplicates = total - unique

    return {
        'total': total,
        'unique': unique,
        'duplicates': duplicates,
        'reduction_pct': 100 * duplicates / total if total > 0 else 0
    }


def main():
    # Input/output paths
    input_file = "output/train_full_5k.jsonl"
    output_file = "output/train_deduplicated.jsonl"

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return

    print(f"Deduplicating {input_file}...")
    stats = deduplicate(input_file, output_file)

    print(f"\nResults:")
    print(f"  Total samples:     {stats['total']:,}")
    print(f"  Unique samples:    {stats['unique']:,}")
    print(f"  Duplicates removed: {stats['duplicates']:,} ({stats['reduction_pct']:.1f}%)")
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

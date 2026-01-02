#!/usr/bin/env python3
"""
Validate long-chain training data for 100% canonical pass-rate.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import validation utilities
from scripts.canonical_validator import validate_entry, compute_validation_metrics


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def analyze_coverage(entries: List[Dict]) -> Dict:
    """Analyze world, noetic, and operator coverage."""
    world_usage = Counter()
    noetic_usage = Counter()
    operator_usage = Counter()
    notation_dist = Counter()

    for entry in entries:
        elements = entry.get('expr_elements', [])
        ops = entry.get('expr_ops', [])
        base_elements = entry.get('expr_elements_base', elements)

        # Count base elements
        for base_elem in base_elements:
            if len(base_elem) >= 2:
                world = base_elem[0]
                try:
                    noetic = int(base_elem[1:])
                    world_usage[world] += 1
                    noetic_usage[noetic] += 1
                except ValueError:
                    pass

        # Count notation types
        for elem in elements:
            if '^' in elem and '_d' in elem:
                notation_dist['full'] += 1
            elif '^' in elem:
                notation_dist['sense'] += 1
            elif '_d' in elem:
                notation_dist['foundation'] += 1
            else:
                notation_dist['plain'] += 1

        # Count operators
        for op in ops:
            operator_usage[op] += 1

    return {
        'world_usage': dict(world_usage),
        'noetic_usage': dict(noetic_usage),
        'operator_usage': dict(operator_usage),
        'notation_dist': dict(notation_dist)
    }


def main():
    parser = argparse.ArgumentParser(description='Validate long-chain training data')
    parser.add_argument('--train', default='output/teacher_long_train_train.jsonl',
                        help='Train split file')
    parser.add_argument('--holdout', default='output/teacher_long_train_holdout.jsonl',
                        help='Holdout split file')
    args = parser.parse_args()

    print("=" * 70)
    print("LONG-CHAIN TRAINING DATA VALIDATION")
    print("=" * 70)

    # Validate train split
    print(f"\nValidating train split: {args.train}")
    train_entries = load_jsonl(args.train)
    train_metrics = compute_validation_metrics(train_entries)

    print(f"\nTrain Split Statistics:")
    print(f"  Total entries: {train_metrics['total']}")
    print(f"  Valid entries: {train_metrics['valid']}")
    print(f"  Invalid entries: {train_metrics['invalid']}")
    print(f"  Canon pass-rate: {train_metrics['pass_rate'] * 100:.1f}%")

    if train_metrics['error_counts']:
        print(f"\n  Error breakdown:")
        for error_type, count in train_metrics['error_counts'].items():
            print(f"    {error_type}: {count}")

    # Validate holdout split
    print(f"\n{'-' * 70}")
    print(f"Validating holdout split: {args.holdout}")
    holdout_entries = load_jsonl(args.holdout)
    holdout_metrics = compute_validation_metrics(holdout_entries)

    print(f"\nHoldout Split Statistics:")
    print(f"  Total entries: {holdout_metrics['total']}")
    print(f"  Valid entries: {holdout_metrics['valid']}")
    print(f"  Invalid entries: {holdout_metrics['invalid']}")
    print(f"  Canon pass-rate: {holdout_metrics['pass_rate'] * 100:.1f}%")

    if holdout_metrics['error_counts']:
        print(f"\n  Error breakdown:")
        for error_type, count in holdout_metrics['error_counts'].items():
            print(f"    {error_type}: {count}")

    # Combined statistics
    total_entries = train_metrics['total'] + holdout_metrics['total']
    total_valid = train_metrics['valid'] + holdout_metrics['valid']
    combined_pass_rate = total_valid / total_entries if total_entries > 0 else 0.0

    print(f"\n{'-' * 70}")
    print(f"Combined Statistics:")
    print(f"  Total entries: {total_entries}")
    print(f"  Valid entries: {total_valid}")
    print(f"  Combined canon pass-rate: {combined_pass_rate * 100:.1f}%")

    # Coverage analysis
    print(f"\n{'-' * 70}")
    print("Coverage Analysis (Train Split):")
    train_coverage = analyze_coverage(train_entries)

    print(f"\n  World distribution:")
    for world in sorted(['A', 'B', 'C', 'D']):
        count = train_coverage['world_usage'].get(world, 0)
        print(f"    {world}: {count}")

    print(f"\n  Noetic distribution:")
    for noetic in sorted(range(1, 11)):
        count = train_coverage['noetic_usage'].get(noetic, 0)
        print(f"    N{noetic}: {count}")

    print(f"\n  Operator distribution:")
    for op in ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']:
        count = train_coverage['operator_usage'].get(op, 0)
        print(f"    {op}: {count}")

    print(f"\n  Notation distribution:")
    total_notation = sum(train_coverage['notation_dist'].values())
    for style in ['plain', 'sense', 'foundation', 'full']:
        count = train_coverage['notation_dist'].get(style, 0)
        pct = 100.0 * count / total_notation if total_notation > 0 else 0
        print(f"    {style}: {count} ({pct:.1f}%)")

    # Count unique equations
    train_equations = set(entry.get('equation', '') for entry in train_entries)
    holdout_equations = set(entry.get('equation', '') for entry in holdout_entries)

    print(f"\n{'-' * 70}")
    print(f"Equation Statistics:")
    print(f"  Unique equations in train: {len(train_equations)}")
    print(f"  Unique equations in holdout: {len(holdout_equations)}")
    print(f"  Total unique equations: {len(train_equations | holdout_equations)}")

    # Check for overlap
    overlap = train_equations & holdout_equations
    if overlap:
        print(f"  WARNING: {len(overlap)} equations appear in both train and holdout!")
    else:
        print(f"  No overlap between train and holdout (good!)")

    # Final verdict
    print(f"\n{'=' * 70}")
    if combined_pass_rate == 1.0:
        print("SUCCESS: 100% canonical pass-rate achieved!")
        print("All entries have valid elements (A-D, 1-10) and valid operators.")
    else:
        print(f"WARNING: Canon pass-rate is {combined_pass_rate * 100:.1f}%, not 100%")

    print("=" * 70)


if __name__ == '__main__':
    main()

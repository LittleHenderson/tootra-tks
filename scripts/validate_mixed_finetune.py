#!/usr/bin/env python3
"""
Validate the mixed fine-tune dataset against canon guardrails.

Canon Guardrails (STRICT):
- Worlds: A, B, C, D only
- Noetics: 1-10 only
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (exactly 9)
- Senses: ^1-^9
- Foundations: _d1-_d7
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict

# Canon constants
VALID_WORLDS = {'A', 'B', 'C', 'D'}
VALID_NOETICS = set(range(1, 11))
VALID_OPERATORS = {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'}
VALID_SENSES = set(range(1, 10))
VALID_FOUNDATIONS = {'_d1', '_d2', '_d3', '_d4', '_d5', '_d6', '_d7'}

# Regex for element validation
ELEMENT_PATTERN = re.compile(r'^([ABCD])(10|[1-9])(\^[1-9])?(_d[1-7])?$')


def validate_element(element: str) -> tuple[bool, str]:
    """Validate a single element against canon."""
    match = ELEMENT_PATTERN.match(element)
    if not match:
        return False, f"Invalid element format: {element}"

    world = match.group(1)
    noetic = int(match.group(2))

    if world not in VALID_WORLDS:
        return False, f"Invalid world '{world}' in {element}"

    if noetic not in VALID_NOETICS:
        return False, f"Invalid noetic '{noetic}' in {element}"

    if match.group(3):  # Sense
        sense_str = match.group(3)[1:]  # Remove ^
        sense = int(sense_str)
        if sense not in VALID_SENSES:
            return False, f"Invalid sense '^{sense}' in {element}"

    if match.group(4):  # Foundation
        foundation = match.group(4)
        if foundation not in VALID_FOUNDATIONS:
            return False, f"Invalid foundation '{foundation}' in {element}"

    return True, ""


def validate_operator(op: str) -> tuple[bool, str]:
    """Validate operator against canon."""
    if op not in VALID_OPERATORS:
        return False, f"Invalid operator: {op}"
    return True, ""


def validate_entry(entry: Dict, idx: int) -> List[str]:
    """Validate a single dataset entry."""
    errors = []

    # Check required fields
    required_fields = ['story', 'expr_elements', 'expr_ops', 'aug_type']
    for field in required_fields:
        if field not in entry:
            errors.append(f"Entry {idx}: Missing field '{field}'")
            return errors  # Can't continue without required fields

    # Validate elements
    for i, elem in enumerate(entry['expr_elements']):
        valid, msg = validate_element(elem)
        if not valid:
            errors.append(f"Entry {idx}, element {i}: {msg}")

    # Validate operators
    for i, op in enumerate(entry['expr_ops']):
        valid, msg = validate_operator(op)
        if not valid:
            errors.append(f"Entry {idx}, operator {i}: {msg}")

    # Check element/operator count consistency
    if len(entry['expr_ops']) != len(entry['expr_elements']) - 1:
        errors.append(f"Entry {idx}: Operator count mismatch "
                     f"({len(entry['expr_ops'])} ops for {len(entry['expr_elements'])} elements)")

    return errors


def analyze_dataset(file_path: Path) -> Dict:
    """Analyze dataset and collect statistics."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Collect statistics
    stats = {
        'total_entries': len(entries),
        'chain_lengths': Counter(),
        'worlds': Counter(),
        'noetics': Counter(),
        'operators': Counter(),
        'senses': Counter(),
        'foundations': Counter(),
        'errors': [],
    }

    for idx, entry in enumerate(entries):
        # Validate entry
        entry_errors = validate_entry(entry, idx)
        stats['errors'].extend(entry_errors)

        # Collect stats
        chain_length = len(entry.get('expr_elements', []))
        stats['chain_lengths'][chain_length] += 1

        for elem in entry.get('expr_elements', []):
            match = ELEMENT_PATTERN.match(elem)
            if match:
                stats['worlds'][match.group(1)] += 1
                stats['noetics'][int(match.group(2))] += 1

                if match.group(3):  # Sense
                    sense = int(match.group(3)[1:])
                    stats['senses'][sense] += 1

                if match.group(4):  # Foundation
                    stats['foundations'][match.group(4)] += 1

        for op in entry.get('expr_ops', []):
            stats['operators'][op] += 1

    return stats


def print_stats(name: str, stats: Dict):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    print(f"\nTotal entries: {stats['total_entries']}")

    # Errors
    if stats['errors']:
        print(f"\nERRORRS FOUND: {len(stats['errors'])}")
        for error in stats['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    else:
        print("\nNO ERRORS - All entries valid!")

    # Chain lengths
    print("\nChain Length Distribution:")
    for length in sorted(stats['chain_lengths'].keys()):
        count = stats['chain_lengths'][length]
        pct = count / stats['total_entries'] * 100
        print(f"  {length} elements: {count:4d} ({pct:5.1f}%)")

    # Worlds
    print("\nWorld Distribution:")
    for world in sorted(stats['worlds'].keys()):
        count = stats['worlds'][world]
        print(f"  {world}: {count:4d}")

    # Noetics
    print("\nNoetic Distribution:")
    for noetic in sorted(stats['noetics'].keys()):
        count = stats['noetics'][noetic]
        print(f"  N{noetic:2d}: {count:4d}")

    # Operators
    print("\nOperator Distribution:")
    for op in sorted(stats['operators'].keys()):
        count = stats['operators'][op]
        print(f"  {op:3s}: {count:4d}")

    # Senses (if any)
    if stats['senses']:
        print("\nSense Distribution:")
        for sense in sorted(stats['senses'].keys()):
            count = stats['senses'][sense]
            print(f"  ^{sense}: {count:4d}")

    # Foundations (if any)
    if stats['foundations']:
        print("\nFoundation Distribution:")
        for foundation in sorted(stats['foundations'].keys()):
            count = stats['foundations'][foundation]
            print(f"  {foundation}: {count:4d}")


def main():
    output_dir = Path('output')

    train_file = output_dir / 'mixed_finetune_train.jsonl'
    holdout_file = output_dir / 'mixed_finetune_holdout.jsonl'

    if not train_file.exists():
        print(f"ERROR: Train file not found: {train_file}")
        return

    if not holdout_file.exists():
        print(f"ERROR: Holdout file not found: {holdout_file}")
        return

    print("="*60)
    print("MIXED FINE-TUNE DATASET VALIDATION")
    print("="*60)

    # Analyze train set
    train_stats = analyze_dataset(train_file)
    print_stats("TRAIN SET", train_stats)

    # Analyze holdout set
    holdout_stats = analyze_dataset(holdout_file)
    print_stats("HOLDOUT SET", holdout_stats)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    total_errors = len(train_stats['errors']) + len(holdout_stats['errors'])
    total_entries = train_stats['total_entries'] + holdout_stats['total_entries']

    print(f"\nTotal entries: {total_entries}")
    print(f"  Train: {train_stats['total_entries']}")
    print(f"  Holdout: {holdout_stats['total_entries']}")

    if total_errors == 0:
        print(f"\n[OK] ALL {total_entries} ENTRIES VALID!")
        print("[OK] All worlds are canonical (A, B, C, D)")
        print("[OK] All noetics are canonical (1-10)")
        print("[OK] All operators are canonical (9 operators)")
        print("[OK] All senses are valid (^1-^9)")
        print("[OK] All foundations are valid (_d1-_d7)")
    else:
        print(f"\n[ERROR] VALIDATION FAILED: {total_errors} errors found")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()

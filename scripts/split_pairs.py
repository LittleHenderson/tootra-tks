#!/usr/bin/env python3
"""
Story-Equation Pair Splitter

CRITICAL: Splits by BASE PAIR first (85/15), then expands to both directions.
Both directions of a pair MUST stay in the same split.

Input: data/story_eq_pairs.jsonl (500 base pairs)
Output:
  - output/teacher_story_eq_train.jsonl (850 records: 425 pairs × 2 directions)
  - output/teacher_story_eq_holdout.jsonl (150 records: 75 pairs × 2 directions)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from teacher.validator import CanonicalValidator


def load_pairs(input_file: Path) -> List[Dict]:
    """Load base pairs from input file."""
    pairs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def create_bidirectional_record(
    pair: Dict,
    direction: str,
    task_type: str
) -> Dict:
    """
    Create a bidirectional training record from a base pair.

    Args:
        pair: Base pair with story, equation, metadata
        direction: "story_to_eq" or "eq_to_story"
        task_type: "story_to_equation" or "equation_to_story"

    Returns:
        Training record with input/target/metadata
    """
    if direction == "story_to_eq":
        input_text = (
            f"Given this TKS narrative:\n\n{pair['story']}\n\n"
            f"Translate this narrative into a canonical TKS equation. "
            f"Use proper notation (World + Noetic, e.g., B4, C10) and valid operators."
        )
        target_text = pair['equation']
    else:  # eq_to_story
        input_text = (
            f"Given this TKS equation:\n\n{pair['equation']}\n\n"
            f"Generate a narrative that explains this equation in terms of TKS concepts. "
            f"Describe the noetic forces, worlds, and their relationships."
        )
        target_text = pair['story']

    return {
        "task_type": task_type,
        "input": input_text,
        "target": target_text,
        "metadata": {
            "pair_id": pair['pair_id'],
            "elements": pair['expr_elements'],
            "operators": pair['expr_ops'],
            "chain_length": pair.get('chain_length', len(pair['expr_elements'])),
            "pattern": pair.get('pattern', 'ci_style')
        },
        "direction": direction,
        "canon_score": pair.get('canon_score', 1.0)
    }


def split_and_expand_pairs(
    pairs: List[Dict],
    train_ratio: float = 0.85,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split pairs first, then expand to bidirectional records.

    CRITICAL: Split by pair_id BEFORE expansion.

    Args:
        pairs: List of base pairs
        train_ratio: Ratio for train split (default: 0.85)
        seed: Random seed for reproducibility

    Returns:
        (train_records, holdout_records) where each record is bidirectional
    """
    # Shuffle pairs
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)

    # Split by pair count
    split_idx = int(len(shuffled_pairs) * train_ratio)
    train_pairs = shuffled_pairs[:split_idx]
    holdout_pairs = shuffled_pairs[split_idx:]

    print(f"Split {len(pairs)} base pairs:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Holdout: {len(holdout_pairs)} pairs")

    # Expand each split to bidirectional records
    train_records = []
    for pair in train_pairs:
        # Story → Equation
        train_records.append(create_bidirectional_record(
            pair, "story_to_eq", "story_to_equation"
        ))
        # Equation → Story
        train_records.append(create_bidirectional_record(
            pair, "eq_to_story", "equation_to_story"
        ))

    holdout_records = []
    for pair in holdout_pairs:
        # Story → Equation
        holdout_records.append(create_bidirectional_record(
            pair, "story_to_eq", "story_to_equation"
        ))
        # Equation → Story
        holdout_records.append(create_bidirectional_record(
            pair, "eq_to_story", "equation_to_story"
        ))

    print(f"\nExpanded to bidirectional records:")
    print(f"  Train: {len(train_records)} records ({len(train_pairs)} × 2)")
    print(f"  Holdout: {len(holdout_records)} records ({len(holdout_pairs)} × 2)")

    return train_records, holdout_records


def validate_split(
    train_records: List[Dict],
    holdout_records: List[Dict]
) -> Dict:
    """
    Validate the split for:
      1. Zero overlap in pair_ids between train/holdout
      2. 100% canonical compliance on all equations

    Returns:
        Validation report with pass/fail status
    """
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    # Extract pair_ids
    train_pair_ids = {rec['metadata']['pair_id'] for rec in train_records}
    holdout_pair_ids = {rec['metadata']['pair_id'] for rec in holdout_records}

    # Check 1: Zero overlap
    overlap = train_pair_ids & holdout_pair_ids
    overlap_check = len(overlap) == 0

    print(f"\n1. Pair ID Overlap Check:")
    print(f"   Train pair_ids: {len(train_pair_ids)}")
    print(f"   Holdout pair_ids: {len(holdout_pair_ids)}")
    print(f"   Overlap: {len(overlap)}")
    if overlap:
        print(f"   [X] FAILED - Overlapping pairs: {sorted(overlap)}")
    else:
        print(f"   [OK] PASSED - Zero overlap")

    # Check 2: Canon compliance on equations
    validator = CanonicalValidator(strict_mode=True)

    train_equations = [rec['target'] for rec in train_records if rec['direction'] == 'story_to_eq']
    holdout_equations = [rec['target'] for rec in holdout_records if rec['direction'] == 'story_to_eq']

    all_equations = train_equations + holdout_equations

    canon_violations = []
    for i, eq in enumerate(all_equations):
        result = validator.validate(eq)
        if not result.is_valid:
            canon_violations.append({
                'equation': eq,
                'errors': [
                    f"{issue.rule}: {issue.message}"
                    for issue in result.issues
                    if issue.severity.value == 'error'
                ]
            })

    canon_check = len(canon_violations) == 0

    print(f"\n2. Canonical Compliance Check:")
    print(f"   Total equations: {len(all_equations)}")
    print(f"   Violations: {len(canon_violations)}")
    if canon_violations:
        print(f"   [X] FAILED - Non-canonical equations found:")
        for v in canon_violations[:5]:  # Show first 5
            print(f"      {v['equation']}")
            for err in v['errors']:
                print(f"        - {err}")
    else:
        print(f"   [OK] PASSED - 100% canonical compliance")

    # Summary
    all_passed = overlap_check and canon_check

    print(f"\n" + "="*70)
    print(f"VALIDATION RESULT: {'[OK] PASSED' if all_passed else '[X] FAILED'}")
    print("="*70)

    return {
        'passed': all_passed,
        'overlap_check': overlap_check,
        'canon_check': canon_check,
        'overlap_count': len(overlap),
        'canon_violations': len(canon_violations),
        'train_pairs': len(train_pair_ids),
        'holdout_pairs': len(holdout_pair_ids),
        'train_records': len(train_records),
        'holdout_records': len(holdout_records)
    }


def save_records(records: List[Dict], output_file: Path):
    """Save records to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Saved {len(records)} records to {output_file}")


def print_statistics(train_records: List[Dict], holdout_records: List[Dict]):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    def count_by_direction(records):
        story_to_eq = sum(1 for r in records if r['direction'] == 'story_to_eq')
        eq_to_story = sum(1 for r in records if r['direction'] == 'eq_to_story')
        return story_to_eq, eq_to_story

    train_s2e, train_e2s = count_by_direction(train_records)
    holdout_s2e, holdout_e2s = count_by_direction(holdout_records)

    print(f"\nTrain Split:")
    print(f"  Total records: {len(train_records)}")
    print(f"  Story -> Equation: {train_s2e}")
    print(f"  Equation -> Story: {train_e2s}")
    print(f"  Base pairs: {len(train_records) // 2}")

    print(f"\nHoldout Split:")
    print(f"  Total records: {len(holdout_records)}")
    print(f"  Story -> Equation: {holdout_s2e}")
    print(f"  Equation -> Story: {holdout_e2s}")
    print(f"  Base pairs: {len(holdout_records) // 2}")

    print(f"\nTotal:")
    print(f"  Total records: {len(train_records) + len(holdout_records)}")
    print(f"  Total base pairs: {(len(train_records) + len(holdout_records)) // 2}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split story-equation pairs')
    parser.add_argument('--input', type=str, default='data/story_eq_pairs.jsonl', help='Input pairs file')
    parser.add_argument('--train-output', type=str, default='output/teacher_story_eq_train.jsonl', help='Train output')
    parser.add_argument('--holdout-output', type=str, default='output/teacher_story_eq_holdout.jsonl', help='Holdout output')
    parser.add_argument('--ratio', type=float, default=0.15, help='Holdout ratio')
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / args.input
    train_file = base_dir / args.train_output
    holdout_file = base_dir / args.holdout_output

    print("="*70)
    print("STORY-EQUATION PAIR SPLITTER")
    print("="*70)
    print(f"\nInput: {input_file}")
    print(f"Output Train: {train_file}")
    print(f"Output Holdout: {holdout_file}")

    # Load base pairs
    print(f"\nLoading base pairs...")
    pairs = load_pairs(input_file)
    print(f"Loaded {len(pairs)} base pairs")

    # Split and expand
    train_ratio = 1.0 - args.ratio
    print(f"\nSplitting by pair ({train_ratio*100:.0f}/{args.ratio*100:.0f})...")
    train_records, holdout_records = split_and_expand_pairs(pairs, train_ratio=train_ratio)

    # Validate
    validation_report = validate_split(train_records, holdout_records)

    if not validation_report['passed']:
        print("\n[X] VALIDATION FAILED - Not saving files")
        sys.exit(1)

    # Save
    print(f"\nSaving files...")
    save_records(train_records, train_file)
    save_records(holdout_records, holdout_file)

    # Statistics
    print_statistics(train_records, holdout_records)

    print("\n[OK] Split complete!")


if __name__ == "__main__":
    main()

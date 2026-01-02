#!/usr/bin/env python3
"""
Create curriculum stage files by filtering existing data by operator count.

Stage 1 (short): 2-3 operators → data/curriculum_stage1.jsonl
Stage 2 (mid): 3-4 operators → data/curriculum_stage2.jsonl
Stage 3 (long): 5-6 operators → data/curriculum_stage3.jsonl

Each stage should have ~200-400 samples after teacher generation/conversion.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def count_operators(entry):
    """Count number of operators in an entry."""
    return len(entry.get('expr_ops', []))


def filter_by_operator_count(input_path, min_ops, max_ops):
    """Filter entries by operator count range."""
    filtered = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                num_ops = count_operators(entry)
                if min_ops <= num_ops <= max_ops:
                    filtered.append(entry)

    return filtered


def create_curriculum_stages(
    source_train_path,
    source_long_path,
    output_dir,
    target_samples_per_stage=300
):
    """
    Create three curriculum stage files.

    Args:
        source_train_path: Path to teacher_rich_v2_train.jsonl
        source_long_path: Path to teacher_long_benchmark_converted.jsonl
        output_dir: Directory to write stage files
        target_samples_per_stage: Target number of samples per stage
    """
    print("=" * 70)
    print("CREATING CURRICULUM STAGES")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Stage 1: 2-3 operators (short chains)
    print("\nStage 1: 2-3 operators (short chains)")
    print("-" * 70)
    stage1 = filter_by_operator_count(source_train_path, 2, 3)
    print(f"  Found {len(stage1)} samples from training data")

    # Limit to target if we have too many
    if len(stage1) > target_samples_per_stage:
        stage1 = stage1[:target_samples_per_stage]
        print(f"  Limited to {len(stage1)} samples")

    stage1_path = output_path / "curriculum_stage1.jsonl"
    with open(stage1_path, 'w', encoding='utf-8') as f:
        for entry in stage1:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved to: {stage1_path}")

    # Stage 2: 3-4 operators (medium chains)
    print("\nStage 2: 3-4 operators (medium chains)")
    print("-" * 70)
    stage2 = filter_by_operator_count(source_train_path, 3, 4)
    print(f"  Found {len(stage2)} samples from training data")

    if len(stage2) > target_samples_per_stage:
        stage2 = stage2[:target_samples_per_stage]
        print(f"  Limited to {len(stage2)} samples")

    stage2_path = output_path / "curriculum_stage2.jsonl"
    with open(stage2_path, 'w', encoding='utf-8') as f:
        for entry in stage2:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved to: {stage2_path}")

    # Stage 3: 5-6 operators (long chains)
    print("\nStage 3: 5-6 operators (long chains)")
    print("-" * 70)

    # First get from training data
    stage3 = filter_by_operator_count(source_train_path, 5, 6)
    print(f"  Found {len(stage3)} samples from training data")

    # Add from long benchmark if available
    if Path(source_long_path).exists():
        long_samples = filter_by_operator_count(source_long_path, 5, 6)
        print(f"  Found {len(long_samples)} samples from long benchmark")
        stage3.extend(long_samples)
        print(f"  Total: {len(stage3)} samples")

    if len(stage3) > target_samples_per_stage:
        stage3 = stage3[:target_samples_per_stage]
        print(f"  Limited to {len(stage3)} samples")

    stage3_path = output_path / "curriculum_stage3.jsonl"
    with open(stage3_path, 'w', encoding='utf-8') as f:
        for entry in stage3:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"  Saved to: {stage3_path}")

    # Summary
    print("\n" + "=" * 70)
    print("CURRICULUM STAGES CREATED")
    print("=" * 70)
    print(f"\nStage 1 (2-3 ops): {len(stage1)} samples → {stage1_path}")
    print(f"Stage 2 (3-4 ops): {len(stage2)} samples → {stage2_path}")
    print(f"Stage 3 (5-6 ops): {len(stage3)} samples → {stage3_path}")
    print(f"\nTotal curriculum samples: {len(stage1) + len(stage2) + len(stage3)}")

    # Analyze operator distribution per stage
    for stage_num, (stage_data, stage_name) in enumerate([
        (stage1, "Stage 1"),
        (stage2, "Stage 2"),
        (stage3, "Stage 3")
    ], 1):
        op_counts = defaultdict(int)
        aug_counts = defaultdict(int)

        for entry in stage_data:
            num_ops = count_operators(entry)
            op_counts[num_ops] += 1
            aug_counts[entry.get('aug_type', 'unknown')] += 1

        print(f"\n{stage_name} operator distribution:")
        for ops in sorted(op_counts.keys()):
            print(f"  {ops} ops: {op_counts[ops]} samples")

        print(f"\n{stage_name} augmentation distribution:")
        for aug_type in sorted(aug_counts.keys()):
            print(f"  {aug_type}: {aug_counts[aug_type]} samples")

    return {
        'stage1': len(stage1),
        'stage2': len(stage2),
        'stage3': len(stage3),
        'total': len(stage1) + len(stage2) + len(stage3)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create curriculum stage files by filtering data by operator count',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--source-train',
                       default='output/teacher_rich_v2_train.jsonl',
                       help='Source training data (default: teacher_rich_v2_train.jsonl)')
    parser.add_argument('--source-long',
                       default='output/teacher_long_benchmark_converted.jsonl',
                       help='Source long benchmark data (default: teacher_long_benchmark_converted.jsonl)')
    parser.add_argument('--output-dir',
                       default='data',
                       help='Output directory (default: data/)')
    parser.add_argument('--target-samples',
                       type=int,
                       default=300,
                       help='Target samples per stage (default: 300)')

    args = parser.parse_args()

    stats = create_curriculum_stages(
        source_train_path=args.source_train,
        source_long_path=args.source_long,
        output_dir=args.output_dir,
        target_samples_per_stage=args.target_samples
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())

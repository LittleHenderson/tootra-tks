#!/usr/bin/env python3
"""
Validate NL→Equation Dataset

Verifies the quality and correctness of the generated NL→equation training dataset.
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict


TKS_ELEMENTS = (
    # World A (Spiritual): A1-A10
    ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"] +
    # World B (Mental): B1-B10
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"] +
    # World C (Emotional): C1-C10
    ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"] +
    # World D (Physical): D1-D10
    ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]
)

OPERATOR_MAP = {"+": 0, "-": 1, "×": 2, "÷": 3}
OPERATOR_NAMES = ["+", "-", "×", "÷"]


def validate_dataset(data_dir: Path):
    """Validate the NL→equation dataset."""
    print("="*70)
    print("NL→Equation Dataset Validation")
    print("="*70)

    # Load main corpus
    corpus_path = data_dir / "nl_corpus.pt"
    print(f"\nLoading corpus from {corpus_path}")
    corpus = torch.load(corpus_path, weights_only=False)

    print("\nCorpus keys:", list(corpus.keys()))
    print("\nCorpus shapes:")
    for k, v in corpus.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} ({v.dtype})")
        else:
            print(f"  {k}: {v}")

    # Validate embeddings
    nl_embeddings = corpus["nl_embeddings"]
    triplets = corpus["triplets"]
    equation_indices = corpus["equation_indices"]

    assert nl_embeddings.shape[0] == triplets.shape[0], "Embedding count mismatch"
    assert nl_embeddings.shape[1] == 384, "Wrong embedding dimension"
    assert triplets.shape[1] == 3, "Triplets should have 3 columns"

    print("\n" + "="*70)
    print("Validation Checks")
    print("="*70)

    # Check index ranges
    left_indices = triplets[:, 0]
    right_indices = triplets[:, 1]
    operator_indices = triplets[:, 2]

    print("\nIndex ranges:")
    print(f"  Left elements: [{left_indices.min()}, {left_indices.max()}] (expected: [0, 39])")
    print(f"  Right elements: [{right_indices.min()}, {right_indices.max()}] (expected: [0, 39])")
    print(f"  Operators: [{operator_indices.min()}, {operator_indices.max()}] (expected: [0, 3])")

    assert left_indices.min() >= 0 and left_indices.max() <= 39, "Left indices out of range"
    assert right_indices.min() >= 0 and right_indices.max() <= 39, "Right indices out of range"
    assert operator_indices.min() >= 0 and operator_indices.max() <= 3, "Operator indices out of range"
    print("  ✓ All indices in valid range")

    # Check embeddings
    print("\nEmbedding statistics:")
    print(f"  Mean: {nl_embeddings.mean():.4f}")
    print(f"  Std: {nl_embeddings.std():.4f}")
    print(f"  Min: {nl_embeddings.min():.4f}")
    print(f"  Max: {nl_embeddings.max():.4f}")

    # Check normalization (should be L2 normalized)
    norms = torch.norm(nl_embeddings, p=2, dim=1)
    print(f"  L2 norms: mean={norms.mean():.4f}, std={norms.std():.6f}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Embeddings not normalized"
    print("  ✓ Embeddings are L2 normalized")

    # Check operator distribution
    print("\nOperator distribution:")
    op_counts = defaultdict(int)
    for op_idx in operator_indices.numpy():
        op_counts[int(op_idx)] += 1

    for op_idx in sorted(op_counts.keys()):
        op_name = OPERATOR_NAMES[op_idx]
        count = op_counts[op_idx]
        pct = 100 * count / len(operator_indices)
        print(f"  {op_name} (idx={op_idx}): {count} ({pct:.1f}%)")

    # Check splits
    train_indices = corpus["train_indices"]
    val_indices = corpus["val_indices"]
    test_indices = corpus["test_indices"]

    print("\nSplit sizes:")
    print(f"  Train: {len(train_indices)} ({100*len(train_indices)/len(triplets):.1f}%)")
    print(f"  Val: {len(val_indices)} ({100*len(val_indices)/len(triplets):.1f}%)")
    print(f"  Test: {len(test_indices)} ({100*len(test_indices)/len(triplets):.1f}%)")

    # Check for overlap
    train_set = set(train_indices.numpy())
    val_set = set(val_indices.numpy())
    test_set = set(test_indices.numpy())

    assert len(train_set & val_set) == 0, "Train/val overlap detected"
    assert len(train_set & test_set) == 0, "Train/test overlap detected"
    assert len(val_set & test_set) == 0, "Val/test overlap detected"
    print("  ✓ No overlap between splits")

    assert len(train_set) + len(val_set) + len(test_set) == len(triplets), "Split coverage incomplete"
    print("  ✓ Splits cover all data")

    # Validate stratification
    print("\nOperator stratification:")
    for split_name, split_indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        split_ops = operator_indices[split_indices]
        split_op_counts = defaultdict(int)
        for op_idx in split_ops.numpy():
            split_op_counts[int(op_idx)] += 1

        print(f"\n  {split_name.capitalize()} split:")
        for op_idx in sorted(split_op_counts.keys()):
            op_name = OPERATOR_NAMES[op_idx]
            count = split_op_counts[op_idx]
            pct = 100 * count / len(split_ops)
            print(f"    {op_name} (idx={op_idx}): {count} ({pct:.1f}%)")

    # Sample some examples
    print("\n" + "="*70)
    print("Sample Examples")
    print("="*70)

    jsonl_path = data_dir / "nl_corpus.jsonl"
    with open(jsonl_path, 'r') as f:
        examples = [json.loads(line) for line in f]

    print("\nShowing 3 random examples:")
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(examples), size=3, replace=False)

    for i, idx in enumerate(sample_indices, 1):
        ex = examples[idx]
        left_elem = TKS_ELEMENTS[ex["left_idx"]]
        right_elem = TKS_ELEMENTS[ex["right_idx"]]
        op = OPERATOR_NAMES[ex["operator_idx"]]

        print(f"\nExample {i}:")
        print(f"  Equation: {left_elem} {op} {right_elem}")
        print(f"  Triplet: ({ex['left_idx']}, {ex['right_idx']}, {ex['operator_idx']})")
        print(f"  Split: {ex['split']}")
        print(f"  Interpretation: {ex['interpretation'][:150]}...")

    # Check split files
    print("\n" + "="*70)
    print("Split File Validation")
    print("="*70)

    for split_name in ["train", "val", "test"]:
        split_path = data_dir / f"nl_corpus_{split_name}.pt"
        print(f"\nValidating {split_path.name}...")

        split_data = torch.load(split_path, weights_only=False)
        split_embeddings = split_data["nl_embeddings"]
        split_triplets = split_data["triplets"]

        expected_indices = corpus[f"{split_name}_indices"]
        actual_indices = split_data["indices"]

        assert torch.equal(expected_indices, actual_indices), f"{split_name} indices mismatch"
        assert len(split_embeddings) == len(expected_indices), f"{split_name} size mismatch"
        print(f"  ✓ {split_name.capitalize()} split validated ({len(split_embeddings)} samples)")

    print("\n" + "="*70)
    print("Validation Complete - All Checks Passed!")
    print("="*70)

    # Print summary
    print("\n" + "="*70)
    print("Dataset Summary")
    print("="*70)
    stats_path = data_dir / "dataset_stats.json"
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print(json.dumps(stats, indent=2))


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data" / "nl_equation_pairs"

    if not data_dir.exists():
        print(f"Error: Dataset directory not found: {data_dir}")
        return 1

    validate_dataset(data_dir)
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Build NL→Equation Training Dataset

Reads tks_6400_complete_merged.jsonl and creates training pairs for NL→equation alignment:
- NL input: interpretation field (natural language description)
- Target: equation triplet (left_idx, right_idx, operator_idx)

Creates embeddings using sentence-transformers to match equation corpus embeddings.
Outputs train/val/test splits stratified by operator.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from collections import defaultdict


# Constants
RANDOM_SEED = 7
EMBEDDING_DIM = 384

# TKS 40 Elements: World A-D, each with 10 elements
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

# Operator mapping: + → 0, - → 1, × → 2, ÷ → 3
OPERATOR_MAP = {"+": 0, "-": 1, "×": 2, "÷": 3, "*": 2, "/": 3}


def element_to_index(element: str) -> int:
    """Convert element name (e.g., 'A1', 'B5') to index (0-39)."""
    element = element.strip().upper()
    try:
        return TKS_ELEMENTS.index(element)
    except ValueError:
        raise ValueError(f"Unknown element: {element}. Valid elements: {TKS_ELEMENTS}")


def operator_to_index(operator: str) -> int:
    """Convert operator symbol to index (0-3)."""
    operator = operator.strip()
    if operator not in OPERATOR_MAP:
        raise ValueError(f"Unknown operator: {operator}. Valid operators: {list(OPERATOR_MAP.keys())}")
    return OPERATOR_MAP[operator]


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, strip whitespace."""
    return text.strip().lower()


def load_equations(jsonl_path: Path) -> List[Dict]:
    """Load equations from JSONL file."""
    equations = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                equations.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    return equations


def create_nl_pairs(equations: List[Dict]) -> Tuple[List[str], List[Tuple[int, int, int]], List[Dict]]:
    """
    Create NL→equation pairs from equation data.

    Returns:
        interpretations: List of natural language interpretations
        triplets: List of (left_idx, right_idx, operator_idx) tuples
        metadata: List of metadata dicts for debugging
    """
    interpretations = []
    triplets = []
    metadata = []

    skipped = 0
    for i, eq in enumerate(equations):
        # Extract interpretation
        interpretation = eq.get("interpretation", "").strip()
        if not interpretation:
            skipped += 1
            continue

        # Extract equation components
        try:
            left = eq.get("left", "").strip()
            right = eq.get("right", "").strip()
            operator = eq.get("operator", "").strip()

            if not left or not right or not operator:
                print(f"Warning: Missing components in equation {i+1}: left={left}, right={right}, op={operator}")
                skipped += 1
                continue

            # Convert to indices
            left_idx = element_to_index(left)
            right_idx = element_to_index(right)
            operator_idx = operator_to_index(operator)

            # Normalize interpretation
            norm_interpretation = normalize_text(interpretation)

            # Store
            interpretations.append(norm_interpretation)
            triplets.append((left_idx, right_idx, operator_idx))
            metadata.append({
                "equation": eq.get("equation", f"{left} {operator} {right}"),
                "left": left,
                "right": right,
                "operator": operator,
                "left_idx": left_idx,
                "right_idx": right_idx,
                "operator_idx": operator_idx,
                "interpretation": interpretation[:200],  # Truncate for readability
                "source": eq.get("source", "unknown")
            })

        except (ValueError, KeyError) as e:
            print(f"Warning: Failed to process equation {i+1}: {e}")
            skipped += 1
            continue

    print(f"\nProcessed {len(equations)} equations:")
    print(f"  - Valid pairs: {len(interpretations)}")
    print(f"  - Skipped: {skipped}")

    return interpretations, triplets, metadata


def create_embeddings(interpretations: List[str], model_name: str = "all-MiniLM-L6-v2") -> torch.Tensor:
    """Create embeddings for interpretations using sentence-transformers."""
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Creating embeddings for {len(interpretations)} interpretations...")
    embeddings = model.encode(
        interpretations,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def stratified_split(
    triplets: List[Tuple[int, int, int]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = RANDOM_SEED
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/val/test splits by operator.

    Returns:
        train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Group by operator
    operator_groups = defaultdict(list)
    for i, (_, _, op_idx) in enumerate(triplets):
        operator_groups[op_idx].append(i)

    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    print(f"\nCreating stratified splits (train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}):")

    for op_idx, indices in sorted(operator_groups.items()):
        # Shuffle indices for this operator
        indices = np.array(indices)
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train+n_val].tolist())
        test_indices.extend(indices[n_train+n_val:].tolist())

        op_symbol = [k for k, v in OPERATOR_MAP.items() if v == op_idx][0]
        print(f"  Operator {op_symbol} (idx={op_idx}): {n} total → {n_train} train, {n_val} val, {len(indices)-n_train-n_val} test")

    # Shuffle the final splits
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")

    return train_indices, val_indices, test_indices


def save_outputs(
    output_dir: Path,
    embeddings: torch.Tensor,
    triplets: List[Tuple[int, int, int]],
    interpretations: List[str],
    metadata: List[Dict],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int]
):
    """Save processed dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert triplets to tensor
    triplets_tensor = torch.tensor(triplets, dtype=torch.int64)

    # Create equation corpus indices (1:1 mapping for now)
    equation_indices = torch.arange(len(triplets), dtype=torch.int64)

    # Save full corpus
    corpus_path = output_dir / "nl_corpus.pt"
    print(f"\nSaving NL corpus to {corpus_path}")
    torch.save({
        "nl_embeddings": embeddings,
        "equation_indices": equation_indices,
        "triplets": triplets_tensor,
        "train_indices": torch.tensor(train_indices, dtype=torch.int64),
        "val_indices": torch.tensor(val_indices, dtype=torch.int64),
        "test_indices": torch.tensor(test_indices, dtype=torch.int64),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": EMBEDDING_DIM,
        "random_seed": RANDOM_SEED
    }, corpus_path)

    # Save JSONL for debugging
    jsonl_path = output_dir / "nl_corpus.jsonl"
    print(f"Saving debug JSONL to {jsonl_path}")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, (interp, (left_idx, right_idx, op_idx), meta) in enumerate(zip(interpretations, triplets, metadata)):
            data = {
                "index": i,
                "interpretation": interp,
                "left_idx": left_idx,
                "right_idx": right_idx,
                "operator_idx": op_idx,
                "equation": meta["equation"],
                "split": "train" if i in train_indices else "val" if i in val_indices else "test"
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Save split-specific files
    for split_name, split_indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        split_path = output_dir / f"nl_corpus_{split_name}.pt"
        print(f"Saving {split_name} split to {split_path}")
        torch.save({
            "nl_embeddings": embeddings[split_indices],
            "equation_indices": equation_indices[split_indices],
            "triplets": triplets_tensor[split_indices],
            "indices": torch.tensor(split_indices, dtype=torch.int64)
        }, split_path)

        # Save JSONL
        split_jsonl_path = output_dir / f"nl_corpus_{split_name}.jsonl"
        with open(split_jsonl_path, 'w', encoding='utf-8') as f:
            for idx in split_indices:
                data = {
                    "index": idx,
                    "interpretation": interpretations[idx],
                    "left_idx": triplets[idx][0],
                    "right_idx": triplets[idx][1],
                    "operator_idx": triplets[idx][2],
                    "equation": metadata[idx]["equation"]
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Save statistics
    stats_path = output_dir / "dataset_stats.json"
    print(f"Saving statistics to {stats_path}")

    operator_counts = defaultdict(int)
    for _, _, op_idx in triplets:
        operator_counts[op_idx] += 1

    stats = {
        "total_pairs": len(triplets),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "embedding_dim": EMBEDDING_DIM,
        "embedding_model": "all-MiniLM-L6-v2",
        "random_seed": RANDOM_SEED,
        "operator_distribution": {
            f"{[k for k, v in OPERATOR_MAP.items() if v == op_idx][0]} (idx={op_idx})": count
            for op_idx, count in sorted(operator_counts.items())
        },
        "element_range": [0, len(TKS_ELEMENTS)-1],
        "operator_range": [0, max(OPERATOR_MAP.values())]
    }

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("Dataset creation complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - nl_corpus.pt (full dataset)")
    print(f"  - nl_corpus_train.pt")
    print(f"  - nl_corpus_val.pt")
    print(f"  - nl_corpus_test.pt")
    print(f"  - nl_corpus.jsonl (full dataset, human-readable)")
    print(f"  - nl_corpus_train.jsonl")
    print(f"  - nl_corpus_val.jsonl")
    print(f"  - nl_corpus_test.jsonl")
    print(f"  - dataset_stats.json")
    print("\nDataset statistics:")
    print(json.dumps(stats, indent=2))


def main():
    """Main entry point."""
    # Paths
    repo_root = Path(__file__).parent.parent
    input_file = repo_root / "tks_6400_complete_merged.jsonl"
    output_dir = repo_root / "data" / "nl_equation_pairs"

    print("="*60)
    print("Building NL→Equation Training Dataset")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Random seed: {RANDOM_SEED}")

    # Load equations
    print(f"\nLoading equations from {input_file}...")
    equations = load_equations(input_file)
    print(f"Loaded {len(equations)} equations")

    # Create NL pairs
    print("\nCreating NL→equation pairs...")
    interpretations, triplets, metadata = create_nl_pairs(equations)

    if not interpretations:
        print("Error: No valid pairs created. Exiting.")
        return 1

    # Create embeddings
    embeddings = create_embeddings(interpretations)

    # Create splits
    train_indices, val_indices, test_indices = stratified_split(triplets)

    # Save outputs
    save_outputs(
        output_dir,
        embeddings,
        triplets,
        interpretations,
        metadata,
        train_indices,
        val_indices,
        test_indices
    )

    return 0


if __name__ == "__main__":
    exit(main())

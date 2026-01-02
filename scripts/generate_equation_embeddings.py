"""
Generate embeddings for 6400 TKS equation interpretations.

Uses sentence-transformers to encode the rich semantic content
from the Book of Kabbalistic Calculus corpus.

Output: equation_embeddings.pt (tensor) + equation_embeddings_meta.json (mapping)
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Sentence transformers for semantic embeddings
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION
# =============================================================================

CORPUS_PATH = Path("tks_6400_complete_merged.jsonl")
OUTPUT_DIR = Path("data/equation_embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
BATCH_SIZE = 64


# =============================================================================
# MAIN
# =============================================================================

def load_corpus(path: Path) -> List[Dict]:
    """Load the 6400 equation corpus."""
    equations = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            equations.append(json.loads(line))
    return equations


def generate_embeddings(
    equations: List[Dict],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    """Generate embeddings for all equation interpretations."""

    # Extract interpretations
    texts = [eq['interpretation'] for eq in equations]

    print(f"Generating embeddings for {len(texts)} equations...")
    print(f"Model: {MODEL_NAME} ({model.get_sentence_embedding_dimension()}D)")

    # Encode in batches with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    return embeddings


def create_index_map(equations: List[Dict]) -> Dict:
    """Create mapping from (left, op, right) -> index."""
    op_to_idx = {'+': 0, '-': 1, '×': 2, '÷': 3}

    index_map = {}
    for idx, eq in enumerate(equations):
        key = f"{eq['left']}_{eq['operator']}_{eq['right']}"
        index_map[key] = idx

    return index_map


def create_metadata(equations: List[Dict]) -> Dict:
    """Create metadata for the embeddings."""

    # Source weights (original=1.0, generated=0.5)
    source_weights = [
        1.0 if eq.get('source', 'original') == 'original' else 0.5
        for eq in equations
    ]

    # Operator indices
    op_to_idx = {'+': 0, '-': 1, '×': 2, '÷': 3}
    operators = [op_to_idx.get(eq['operator'], 0) for eq in equations]

    # Element indices
    def element_to_idx(elem: str) -> int:
        world = elem[0]
        noetic = int(elem[1:])
        world_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[world]
        return world_idx * 10 + (noetic - 1)

    left_indices = [element_to_idx(eq['left']) for eq in equations]
    right_indices = [element_to_idx(eq['right']) for eq in equations]

    return {
        'source_weights': source_weights,
        'operators': operators,
        'left_indices': left_indices,
        'right_indices': right_indices,
        'num_equations': len(equations),
        'embedding_dim': 384,
        'model_name': MODEL_NAME,
    }


def main():
    print("=" * 70)
    print("TKS Equation Embedding Generator")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    print(f"\nLoading corpus from {CORPUS_PATH}...")
    equations = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(equations)} equations")

    # Count sources
    original_count = sum(1 for eq in equations if eq.get('source', 'original') == 'original')
    generated_count = len(equations) - original_count
    print(f"  Original: {original_count}")
    print(f"  Generated: {generated_count}")

    # Load sentence transformer
    print(f"\nLoading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    embeddings = generate_embeddings(equations, model)
    print(f"Embeddings shape: {embeddings.shape}")

    # Create index map
    index_map = create_index_map(equations)

    # Create metadata
    metadata = create_metadata(equations)
    metadata['index_map'] = index_map

    # Save embeddings
    embeddings_path = OUTPUT_DIR / "equation_embeddings.pt"
    torch.save(embeddings, embeddings_path)
    print(f"\nSaved embeddings to: {embeddings_path}")

    # Save metadata
    metadata_path = OUTPUT_DIR / "equation_embeddings_meta.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    # Save a compact tensor version with all info
    compact = {
        'embeddings': embeddings.cpu(),
        'source_weights': torch.tensor(metadata['source_weights']),
        'operators': torch.tensor(metadata['operators']),
        'left_indices': torch.tensor(metadata['left_indices']),
        'right_indices': torch.tensor(metadata['right_indices']),
    }
    compact_path = OUTPUT_DIR / "equation_corpus.pt"
    torch.save(compact, compact_path)
    print(f"Saved compact corpus to: {compact_path}")

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Check a few samples
    print("\nSample embeddings:")
    for i in [0, 100, 1000, 5000]:
        eq = equations[i]
        emb = embeddings[i]
        print(f"  [{i}] {eq['equation']}: norm={emb.norm().item():.4f}, mean={emb.mean().item():.4f}")

    # Check operator distribution
    print("\nOperator distribution:")
    for op, idx in {'+': 0, '-': 1, '×': 2, '÷': 3}.items():
        count = sum(1 for o in metadata['operators'] if o == idx)
        print(f"  {op}: {count}")

    # Check similarity between related equations
    print("\nSimilarity check (should be high for related equations):")
    # Find A1+A2 and A2+A1 (should be similar due to commutativity of +)
    idx_a1_plus_a2 = index_map.get("A1_+_A2")
    idx_a2_plus_a1 = index_map.get("A2_+_A1")
    if idx_a1_plus_a2 is not None and idx_a2_plus_a1 is not None:
        sim = torch.cosine_similarity(
            embeddings[idx_a1_plus_a2].unsqueeze(0),
            embeddings[idx_a2_plus_a1].unsqueeze(0)
        ).item()
        print(f"  A1+A2 vs A2+A1: {sim:.4f}")

    # Find A1-A2 and A2-A1 (should be different due to non-commutativity of -)
    idx_a1_minus_a2 = index_map.get("A1_-_A2")
    idx_a2_minus_a1 = index_map.get("A2_-_A1")
    if idx_a1_minus_a2 is not None and idx_a2_minus_a1 is not None:
        sim = torch.cosine_similarity(
            embeddings[idx_a1_minus_a2].unsqueeze(0),
            embeddings[idx_a2_minus_a1].unsqueeze(0)
        ).item()
        print(f"  A1-A2 vs A2-A1: {sim:.4f}")

    print("\n" + "=" * 70)
    print("Done! Embeddings ready for training.")
    print("=" * 70)


if __name__ == "__main__":
    main()

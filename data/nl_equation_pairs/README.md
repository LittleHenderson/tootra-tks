# NL→Equation Training Dataset

This directory contains the Natural Language to Equation alignment training dataset for the TKS (Tootra Kabbalistic System).

## Overview

The dataset maps natural language interpretations to TKS equation triplets, enabling models to learn the semantic relationship between human-readable descriptions and formal TKS equations.

## Dataset Statistics

- **Total pairs**: 6,400
- **Training set**: 5,120 (80%)
- **Validation set**: 640 (10%)
- **Test set**: 640 (10%)
- **Embedding dimension**: 384
- **Embedding model**: all-MiniLM-L6-v2
- **Random seed**: 7 (for reproducibility)

## Data Format

### Equation Triplets

Each equation is represented as a triplet `(left_idx, right_idx, operator_idx)`:

- **Element indices** (left_idx, right_idx): Range [0, 39]
  - World A (Spiritual): A1-A10 → indices 0-9
  - World B (Mental): B1-B10 → indices 10-19
  - World C (Emotional): C1-C10 → indices 20-29
  - World D (Physical): D1-D10 → indices 30-39

- **Operator indices**: Range [0, 3]
  - `+` (Association) → 0
  - `-` (Disassociation) → 1
  - `×` (Intensification) → 2
  - `÷` (Opposition) → 3

### Operator Distribution

Perfectly balanced across all operators:
- `+` (Association): 1,600 (25%)
- `-` (Disassociation): 1,600 (25%)
- `×` (Intensification): 1,600 (25%)
- `÷` (Opposition): 1,600 (25%)

## Files

### PyTorch Tensors (.pt)

- **`nl_corpus.pt`**: Full dataset with all splits
  - `nl_embeddings`: [6400, 384] tensor of L2-normalized interpretation embeddings
  - `equation_indices`: [6400] tensor mapping to equation_corpus.pt indices
  - `triplets`: [6400, 3] tensor of (left_idx, right_idx, operator_idx)
  - `train_indices`, `val_indices`, `test_indices`: Split indices
  - Metadata: embedding_model, embedding_dim, random_seed

- **`nl_corpus_train.pt`**: Training split (5,120 samples)
- **`nl_corpus_val.pt`**: Validation split (640 samples)
- **`nl_corpus_test.pt`**: Test split (640 samples)

Each split file contains:
- `nl_embeddings`: Embeddings for split samples
- `equation_indices`: Equation corpus indices
- `triplets`: Equation triplets
- `indices`: Original indices in full corpus

### JSONL Files

Human-readable JSON Lines format for debugging:

- **`nl_corpus.jsonl`**: Full dataset
- **`nl_corpus_train.jsonl`**: Training split
- **`nl_corpus_val.jsonl`**: Validation split
- **`nl_corpus_test.jsonl`**: Test split

Each line contains:
```json
{
  "index": 0,
  "interpretation": "normalized natural language description...",
  "left_idx": 0,
  "right_idx": 1,
  "operator_idx": 0,
  "equation": "A1 + A2",
  "split": "train"
}
```

### Statistics

- **`dataset_stats.json`**: Dataset statistics and metadata

## Usage

### Loading in Python

```python
import torch

# Load full corpus
corpus = torch.load("nl_corpus.pt")
nl_embeddings = corpus["nl_embeddings"]      # [6400, 384]
triplets = corpus["triplets"]                # [6400, 3]
train_indices = corpus["train_indices"]      # [5120]

# Load specific split
train_data = torch.load("nl_corpus_train.pt")
train_embeddings = train_data["nl_embeddings"]  # [5120, 384]
train_triplets = train_data["triplets"]          # [5120, 3]
```

### Example Training Loop

```python
# Simple retrieval example
from torch.nn.functional import cosine_similarity

# Get a query interpretation embedding
query_embedding = nl_embeddings[0:1]  # [1, 384]

# Find most similar interpretations
similarities = cosine_similarity(query_embedding, nl_embeddings)
top_k = torch.topk(similarities, k=5)

# Get corresponding equations
for idx in top_k.indices[0]:
    triplet = triplets[idx]
    print(f"Equation: {triplet}, Similarity: {similarities[idx]:.4f}")
```

## Data Quality

All data has been validated for:
- ✓ Index ranges (elements: 0-39, operators: 0-3)
- ✓ L2-normalized embeddings (norm = 1.0)
- ✓ No overlap between train/val/test splits
- ✓ Perfect stratification by operator (25% each)
- ✓ Complete coverage (all 6,400 samples)

## Source Data

Generated from `tks_6400_complete_merged.jsonl` containing:
- 6,400 TKS equations (40 elements × 40 elements × 4 operators)
- Natural language interpretations for each equation
- Element names, operator names, and metadata

## Text Preprocessing

Interpretations are normalized via:
1. Lowercase conversion
2. Whitespace stripping
3. UTF-8 encoding

## Scripts

- **`scripts/build_nl_equation_pairs.py`**: Generate this dataset
- **`scripts/validate_nl_pairs.py`**: Validate dataset quality

## Alignment with Equation Corpus

This dataset is designed to align with `data/equation_embeddings/equation_corpus.pt`:
- Same random seed (7)
- Compatible index ranges
- Matching stratification strategy
- 1:1 correspondence via `equation_indices`

## Version Info

- Created: 2025-12-23
- TKS Version: v7.3
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Source: tks_6400_complete_merged.jsonl

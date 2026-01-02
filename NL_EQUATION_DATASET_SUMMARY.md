# NL→Equation Training Dataset - Build Summary

## Overview

Successfully created a comprehensive Natural Language to Equation alignment training dataset for the TKS (Tootra Kabbalistic System). The dataset enables models to learn the semantic relationship between human-readable interpretations and formal TKS equation triplets.

## Dataset Specifications

### Size and Splits
- **Total pairs**: 6,400
- **Training set**: 5,120 (80%)
- **Validation set**: 640 (10%)
- **Test set**: 640 (10%)

### Technical Details
- **Embedding dimension**: 384
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Normalization**: L2-normalized embeddings (norm = 1.0)
- **Random seed**: 7 (for reproducibility)
- **Stratification**: Perfectly balanced by operator (25% each)

### Data Structure

Each training pair consists of:
1. **NL Input**: Natural language interpretation (normalized, lowercase)
2. **Target**: Equation triplet `(left_idx, right_idx, operator_idx)`

#### Element Indices (0-39)
- World A (Spiritual): A1-A10 → indices 0-9
- World B (Mental): B1-B10 → indices 10-19
- World C (Emotional): C1-C10 → indices 20-29
- World D (Physical): D1-D10 → indices 30-39

#### Operator Indices (0-3)
- `+` (Association) → 0
- `-` (Disassociation) → 1
- `×` (Intensification) → 2
- `÷` (Opposition) → 3

### Operator Distribution

Perfectly balanced across all operators:
- `+` (Association): 1,600 pairs (25.0%)
- `-` (Disassociation): 1,600 pairs (25.0%)
- `×` (Intensification): 1,600 pairs (25.0%)
- `÷` (Opposition): 1,600 pairs (25.0%)

## Files Created

### Location
All files located in: `/data/nl_equation_pairs/`

### PyTorch Tensors (.pt files)

1. **`nl_corpus.pt`** (9.7 MB) - Full dataset
   - `nl_embeddings`: [6400, 384] L2-normalized interpretation embeddings
   - `equation_indices`: [6400] mapping to equation_corpus.pt
   - `triplets`: [6400, 3] equation triplets (left, right, operator)
   - `train_indices`, `val_indices`, `test_indices`: Split indices
   - Metadata: embedding_model, embedding_dim, random_seed

2. **`nl_corpus_train.pt`** (7.7 MB) - Training split (5,120 samples)
3. **`nl_corpus_val.pt`** (987 KB) - Validation split (640 samples)
4. **`nl_corpus_test.pt`** (988 KB) - Test split (640 samples)

### JSONL Files (Human-readable)

5. **`nl_corpus.jsonl`** (5.3 MB) - Full dataset
6. **`nl_corpus_train.jsonl`** (4.1 MB) - Training split
7. **`nl_corpus_val.jsonl`** (521 KB) - Validation split
8. **`nl_corpus_test.jsonl`** (527 KB) - Test split

### Documentation

9. **`README.md`** - Comprehensive dataset documentation
10. **`dataset_stats.json`** - Dataset statistics and metadata

## Scripts Created

### 1. `scripts/build_nl_equation_pairs.py` (14 KB)

**Purpose**: Generate the NL→equation training dataset

**Features**:
- Reads `tks_6400_complete_merged.jsonl`
- Creates NL→equation pairs with triplet targets
- Maps element names to indices (40 TKS elements)
- Maps operators to indices (4 operators)
- Generates 384D embeddings using sentence-transformers
- Creates stratified train/val/test splits (80/10/10)
- Saves PyTorch tensors and JSONL files

**Usage**:
```bash
python3 scripts/build_nl_equation_pairs.py
```

**Output**:
```
Total pairs: 6400
  - Valid pairs: 6400
  - Skipped: 0

Split sizes:
  Train: 5120 (80%)
  Val: 640 (10%)
  Test: 640 (10%)
```

### 2. `scripts/validate_nl_pairs.py` (7.6 KB)

**Purpose**: Validate dataset quality and correctness

**Validation checks**:
- ✓ Index ranges (elements: 0-39, operators: 0-3)
- ✓ L2-normalized embeddings (norm = 1.0)
- ✓ No overlap between train/val/test splits
- ✓ Perfect stratification by operator (25% each)
- ✓ Complete coverage (all 6,400 samples)
- ✓ Split file integrity

**Usage**:
```bash
python3 scripts/validate_nl_pairs.py
```

### 3. `scripts/demo_nl_retrieval.py` (5.2 KB)

**Purpose**: Demonstrate NL→equation retrieval

**Features**:
- Loads dataset and embedding model
- Encodes natural language queries
- Retrieves top-k most similar equations
- Interactive query mode
- Pre-configured demo queries

**Usage**:
```bash
python3 scripts/demo_nl_retrieval.py
```

**Example queries**:
- "spiritual growth through positive energy" → A2 × B2 (similarity: 0.809)
- "mental conflict and emotional turmoil" → C1 ÷ B3 (similarity: 0.655)
- "physical healing with spiritual guidance" → D9 × A2 (similarity: 0.686)

## Data Quality

### Validation Results

All validation checks passed:
- **Index ranges**: All within valid bounds (elements: 0-39, operators: 0-3)
- **Embeddings**: L2-normalized with mean norm = 1.0000, std = 0.000000
- **Splits**: No overlap, complete coverage
- **Stratification**: Perfect 25% distribution per operator across all splits

### Sample Data

**Training Example**:
```
Equation: D3 × A4
Triplet: (32, 3, 2)
NL Input: "physical negative (d3) intensified by spiritual vibration (a4)
           suggests physical disorder or dysfunction amplified through
           spiritual energy frequencies..."
```

**Validation Example**:
```
Equation: A9 ÷ C2
Triplet: (8, 21, 3)
NL Input: "spiritual below (a9) opposes emotional positive (c2). this formula
           suggests a conflict or opposition between superficial or exoteric
           spiritual beliefs..."
```

**Test Example**:
```
Equation: C1 × A3
Triplet: (20, 2, 2)
NL Input: "emotional mind (c1) is exacerbated by spiritual negative (a3).
           this combination intensifies emotional awareness through engagement
           with challenging spiritual experiences..."
```

### Statistics

- **Embedding mean**: -0.0003
- **Embedding std**: 0.0510
- **Embedding range**: [-0.2362, 0.2388]
- **L2 norm**: 1.0000 (all samples)

## Source Data

**Input file**: `tks_6400_complete_merged.jsonl`
- 6,400 TKS equations (40 elements × 40 elements × 4 operators)
- Each line contains: left, operator, right, interpretation, world, rpm, etc.
- 100% valid pairs (0 skipped)

## Text Preprocessing

Interpretations normalized via:
1. Lowercase conversion
2. Whitespace stripping
3. UTF-8 encoding preservation

## Alignment with Equation Corpus

This dataset aligns with `data/equation_embeddings/equation_corpus.pt`:
- ✓ Same random seed (7)
- ✓ Compatible index ranges
- ✓ Matching stratification strategy
- ✓ 1:1 correspondence via `equation_indices`

## Usage Examples

### Loading the Dataset

```python
import torch

# Load full corpus
corpus = torch.load("data/nl_equation_pairs/nl_corpus.pt")
nl_embeddings = corpus["nl_embeddings"]      # [6400, 384]
triplets = corpus["triplets"]                # [6400, 3]
train_indices = corpus["train_indices"]      # [5120]

# Load specific split
train_data = torch.load("data/nl_equation_pairs/nl_corpus_train.pt")
train_embeddings = train_data["nl_embeddings"]  # [5120, 384]
train_triplets = train_data["triplets"]          # [5120, 3]
```

### Simple Retrieval

```python
from torch.nn.functional import cosine_similarity

# Query embedding
query_embedding = nl_embeddings[0:1]  # [1, 384]

# Find similar interpretations
similarities = cosine_similarity(query_embedding, nl_embeddings)
top_k = torch.topk(similarities, k=5)

# Get corresponding equations
for idx in top_k.indices[0]:
    triplet = triplets[idx]
    print(f"Equation: {triplet}, Similarity: {similarities[idx]:.4f}")
```

### Training Loop Template

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load training data
train_data = torch.load("data/nl_equation_pairs/nl_corpus_train.pt")
X_train = train_data["nl_embeddings"]
y_train = train_data["triplets"]

# Create dataset and loader
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for nl_emb, triplet in loader:
        # nl_emb: [batch, 384]
        # triplet: [batch, 3] - (left_idx, right_idx, op_idx)

        # Your training logic here
        pass
```

## Key Features

1. **High-quality embeddings**: L2-normalized, 384D semantic vectors
2. **Perfect stratification**: Balanced operator distribution across splits
3. **Comprehensive coverage**: All 6,400 equations with interpretations
4. **Reproducible**: Fixed random seed (7) for consistent splits
5. **Well-documented**: README, validation, and demo scripts
6. **Multiple formats**: PyTorch tensors for training, JSONL for debugging
7. **Validated**: All quality checks passed

## Performance Notes

- **Embedding generation**: ~6 seconds for 6,400 interpretations
- **Dataset creation**: Total runtime ~10 seconds
- **File sizes**: Total ~30 MB (compressed PyTorch tensors)
- **Memory usage**: ~2.5 MB for 384D embeddings

## Next Steps

Potential uses for this dataset:

1. **NL→Equation Retrieval**: Given natural language, find matching equations
2. **Equation Generation**: Train models to generate equations from descriptions
3. **Semantic Search**: Build a TKS equation search engine
4. **Transfer Learning**: Use as pre-training for TKS language models
5. **Evaluation**: Validate equation generation quality
6. **Curriculum Learning**: Progressive difficulty based on operators

## Version Information

- **Created**: 2025-12-23
- **TKS Version**: v7.3
- **Python**: 3.14
- **PyTorch**: Compatible with torch.save/load
- **Sentence Transformers**: all-MiniLM-L6-v2
- **Source**: tks_6400_complete_merged.jsonl

## File Checksums (MD5)

```
58d9644f8dc29e320ba782074380d6e4  nl_corpus.pt
8e2123922f85eae39f362f8c79d9c8c2  nl_corpus_train.pt
e10bdfa64f7c29bacbf8d9655cbf7ba9  nl_corpus_val.pt
bb4470549a7fcc142c5b7b98e99e01a1  nl_corpus_test.pt
32856734fc2838cabb5c60773c9a9867  nl_corpus.jsonl
238255edff186b60bb1dd325cc41e7b7  nl_corpus_train.jsonl
9341fc7639b44a5682f5169e0e7dc0d8  nl_corpus_val.jsonl
9b9fe185dc4f8731d5f01f6749ab9194  nl_corpus_test.jsonl
7ee10641468e51687eb185731fe09290  dataset_stats.json
```

## Conclusion

Successfully built a high-quality NL→equation training dataset with 6,400 pairs, perfect stratification, comprehensive validation, and demo scripts. The dataset is ready for training models that can map natural language descriptions to TKS equation triplets, enabling semantic search, equation generation, and other NL-based TKS applications.

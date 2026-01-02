# TKS Training Integration - Quick Start Guide

**Phase 2 Implementation: Training with Augmented Data**

This guide will help you quickly get started with the TKS training integration pipeline.

---

## Prerequisites

- Python 3.7+
- Augmented JSONL data file (generated from `scripts/generate_augmented_data.py`)

## Installation

No additional dependencies required for Phase 2 (minimal stub implementation).

For future phases (actual model training):
```bash
pip install torch transformers numpy
```

---

## Quick Start

### 1. Verify Installation

Run the smoke test on the sample dataset:

```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --test
```

**Expected Output:**
```
======================================================================
SMOKE TEST - Data Pipeline Verification
======================================================================

[Test 1] Loading augmented corpus...
  [PASS] Loaded 15 entries

[Test 2] Checking entry structure...
  [PASS] All required fields present

...

======================================================================
[PASS] ALL SMOKE TESTS PASSED
======================================================================
```

### 2. Run Basic Training

Run the minimal training loop on sample data:

```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --epochs 3 \
    --batch-size 4
```

**Expected Output:**
```
======================================================================
TKS TRAINING WITH AUGMENTED DATA - Phase 2 Implementation
======================================================================

Loading augmented corpus from: output/sample_augmented.jsonl
Loaded 15 entries from augmented corpus

Augmentation type distribution:
  - anti_attractor: 4
  - inversion: 6
  - original: 5

...

======================================================================
TRAINING COMPLETE
======================================================================
```

### 3. Train on TKS Expressions

Train using TKS expressions instead of natural language:

```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --use-expr \
    --epochs 5 \
    --batch-size 8
```

---

## Complete Workflow

### Step 1: Generate Augmented Data

First, create augmented training data from original corpus:

```bash
python scripts/generate_augmented_data.py \
    --input data/original_corpus.jsonl \
    --output output/augmented_corpus.jsonl \
    --axes W N F \
    --use-anti-attractor \
    --validate
```

### Step 2: Verify Data Pipeline

Run smoke test to verify data integrity:

```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --test
```

### Step 3: Train Model

Run training with your preferred configuration:

```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --epochs 10 \
    --batch-size 32 \
    --filter-validated
```

---

## CLI Options Reference

### Required Arguments

- `--data PATH`: Path to augmented JSONL file

### Training Configuration

- `--epochs N`: Number of training epochs (default: 10)
- `--batch-size N`: Batch size for training (default: 32)
- `--learning-rate FLOAT`: Learning rate (default: 1e-4)
- `--max-length N`: Maximum sequence length (default: 512)

### Data Filtering

- `--filter-validated`: Only use entries with validator_pass=True
- `--original-data PATH`: Optional path to original corpus for comparison

### Input Format

- `--use-expr`: Train on TKS expressions instead of stories
- `--include-metadata`: Include aug_type tags in inputs

### Testing

- `--test`: Run smoke test instead of training

### Output

- `--output-dir PATH`: Directory to save models (default: output/models)

---

## Common Use Cases

### Use Case 1: Validate Your Data

```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --test
```

### Use Case 2: Train with High-Quality Data Only

```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --filter-validated \
    --epochs 20 \
    --batch-size 16
```

### Use Case 3: Train on TKS Expressions with Metadata

```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --use-expr \
    --include-metadata \
    --epochs 15 \
    --batch-size 24
```

### Use Case 4: Small Batch Debugging

```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --epochs 2 \
    --batch-size 4
```

---

## Understanding the Output

### Training Output Structure

```
======================================================================
TKS TRAINING WITH AUGMENTED DATA - Phase 2 Implementation
======================================================================

Loading augmented corpus from: [path]
Loaded [N] entries from augmented corpus

Augmentation type distribution:
  - anti_attractor: [N]
  - inversion: [N]
  - original: [N]

Training configuration:
  Epochs: [N]
  Batch size: [N]
  Learning rate: [FLOAT]
  Max length: [N]
  Use expressions: [True/False]
  Include metadata: [True/False]
  Filter validated: [True/False]

Preparing batches...
Total batches: [N]

======================================================================
TRAINING LOOP (Minimal Stub)
======================================================================

Epoch [N]/[TOTAL]
----------------------------------------------------------------------
  Batch [N]/[TOTAL]: size=[N], loss=[FLOAT]

  Epoch [N] Summary:
    Average loss: [FLOAT]
    Total batches: [N]
    Loss std dev: [FLOAT]

======================================================================
TRAINING COMPLETE
======================================================================
Total entries processed: [N]
Total epochs: [N]
Total batches per epoch: [N]

Note: This is a minimal stub implementation.
No actual model training occurred - only data pipeline verification.

Next steps:
  1. Add actual model (e.g., transformer, LSTM)
  2. Implement real loss function (e.g., cross-entropy)
  3. Add optimizer and backpropagation
  4. Add validation loop
  5. Add model checkpointing
======================================================================
```

### Key Metrics to Monitor

1. **Augmentation type distribution**: Ensure balanced mix of original, inversion, and anti_attractor
2. **Batch size**: Verify expected batch sizes (last batch may be smaller)
3. **Loss**: Should be consistent across epochs (dummy loss in Phase 2)
4. **Validation rate**: Check percentage of entries passing canonical validation

---

## Expected Data Format

Your augmented JSONL file should contain entries like:

```json
{
  "id": "entry_001",
  "story": "A spiritual teacher causes enlightenment",
  "expr": "A5 -> D2",
  "expr_elements": ["A5", "D2"],
  "expr_ops": ["->"],
  "aug_type": "original",
  "source_id": "entry_001",
  "validator_pass": true
}
```

### Required Fields

- `id`: Unique entry identifier
- `story` or `expr`: Text content (at least one required)
- `aug_type`: "original", "inversion", or "anti_attractor"
- `validator_pass`: Boolean indicating canonical validation result

### Optional Fields

- `expr_elements`: List of TKS element codes (e.g., ["B2", "D5"])
- `expr_ops`: List of operators (e.g., ["->"])
- `axes`: (for inversions) List of axes applied (e.g., ["W", "N"])
- `mode`: (for inversions) Inversion mode ("soft", "hard", "targeted")
- `source_id`: Reference to parent entry

---

## Troubleshooting

### Problem: "FileNotFoundError: Augmented corpus file not found"

**Solution:** Verify the path to your augmented JSONL file:
```bash
ls output/augmented_corpus.jsonl
```

If file doesn't exist, generate it first:
```bash
python scripts/generate_augmented_data.py \
    --input data/original.jsonl \
    --output output/augmented_corpus.jsonl
```

### Problem: "No inputs generated" during batch preparation

**Solution:** Check your data format. Entries must have either `story` or `expr` field.

Verify data structure:
```bash
head -n 1 output/augmented_corpus.jsonl | python -m json.tool
```

### Problem: All batches are empty

**Solution:** Check if entries are being filtered out:
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --test
```

Look for validation pass rate. If too low, remove `--filter-validated` flag.

### Problem: Unicode errors on Windows

**Solution:** The script has been updated to use ASCII-compatible output markers:
- `[PASS]` instead of `✓`
- `[FAIL]` instead of `✗`

If still encountering issues, set UTF-8 encoding:
```bash
chcp 65001
python scripts/train_with_augmented.py --data output/augmented_corpus.jsonl --test
```

---

## Current Limitations (Phase 2)

**Phase 2 is a minimal stub implementation focusing on the data pipeline.**

What's implemented:
- ✓ Data loading and parsing
- ✓ Batch preparation
- ✓ Training loop structure
- ✓ Dummy loss computation
- ✓ Metrics tracking

What's NOT implemented (coming in Phase 3):
- ✗ Actual model architecture
- ✗ Real loss function (cross-entropy, contrastive)
- ✗ Optimizer and backpropagation
- ✗ Model checkpointing
- ✗ Validation loop
- ✗ Canonical validation on outputs

**This implementation verifies the data pipeline is working correctly and ready for model integration.**

---

## Next Steps

After successfully running Phase 2:

1. **Verify End-to-End Pipeline**
   - Generate augmented data
   - Run smoke test
   - Run training loop

2. **Prepare for Phase 3**
   - Select model architecture (GPT-2, LSTM, etc.)
   - Design loss function strategy
   - Plan validation approach

3. **Scale Up**
   - Create larger augmented corpus (1000+ entries)
   - Optimize batch processing
   - Add metrics logging (W&B, TensorBoard)

---

## Getting Help

### Documentation

- **Full Implementation Details:** `PHASE2_IMPLEMENTATION_SUMMARY.md`
- **Training Integration Plan:** `docs/TRAINING_INTEGRATION_PLAN.md`
- **Augmentation Pipeline:** `scripts/generate_augmented_data.py` (docstrings)

### Sample Data

- **Sample Augmented Corpus:** `output/sample_augmented.jsonl`
- Contains 15 entries demonstrating all augmentation types

### Testing

Run comprehensive smoke test:
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --test
```

---

## Summary

**Phase 2 provides a functional training infrastructure that:**

✓ Loads and processes augmented JSONL data
✓ Handles all augmentation types (original, inversion, anti_attractor)
✓ Supports both story and expression inputs
✓ Provides flexible CLI configuration
✓ Implements comprehensive smoke tests
✓ Integrates seamlessly with augmentation pipeline

**Start experimenting with the data pipeline now, and prepare for Phase 3 model integration!**

---

**Last Updated:** 2025-12-14
**Version:** Phase 2 Complete
**Status:** Production-Ready Data Pipeline

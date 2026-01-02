# TKS Training Pipeline - Phase 3 Implementation

## Overview

This document describes the complete training pipeline implementation for Phase 3, which wires augmented data into a trainer, runs training jobs, and logs comprehensive metrics.

**Status**: ✅ COMPLETE - All components implemented and tested

**Files**:
- `scripts/train_with_augmented.py` - Main training script
- `scripts/augmentation_metrics.py` - Metrics logging module
- `tests/test_train_with_augmented.py` - Comprehensive test suite (29 tests)

---

## Features Implemented

### 1. Data Loading & Processing
- ✅ Load augmented JSONL files (original + inversions + anti-attractors)
- ✅ Validation filtering (filter by `validator_pass` flag)
- ✅ Batch preparation with configurable settings
- ✅ Character-level tokenization (compatible with any text)
- ✅ Support for both stories and TKS expressions

### 2. Training Loop
- ✅ Full epoch-based training implementation
- ✅ Batch iteration with shuffling
- ✅ Dummy model for pipeline validation (DummyTKSModel)
- ✅ Simple optimizer stub (SimpleOptimizer)
- ✅ Configurable epochs, batch size, learning rate
- ✅ Max steps limit for controlled runs

### 3. Metrics Logging

#### Per-Step Metrics:
- Loss value
- Batch size
- Average input length
- Unique input count

#### Per-Epoch Metrics:
- Average loss
- Loss standard deviation
- Validation pass rate (100% in filtered mode)
- Augmentation ratio (augmented/original)
- World/Noetic/Operator validity rates
- Augmentation type distribution

#### Output Files:
- `training_metrics.json` - Complete metrics summary
- `training_metrics_epochs.csv` - Epoch-level loss curve
- `training_metrics_steps.csv` - Step-level loss curve
- `epoch_XXX_metrics.json` - Detailed per-epoch metrics

### 4. Validation & Quality Checks
- ✅ World validation (A/B/C/D only)
- ✅ Noetic validation (1-10 only)
- ✅ Operator validation (allowed ops: +, -, +T, -T, ->, <-, *T, /T, o)
- ✅ Structural validation (ops count = elements count - 1)
- ✅ Automatic filtering of invalid entries

### 5. CLI Features
- ✅ `--dry-run` flag - Run 1 batch only for pipeline validation
- ✅ `--test` flag - Run smoke test suite
- ✅ `--filter-validated` - Only train on validated entries
- ✅ `--use-expr` - Train on expressions instead of stories
- ✅ `--include-metadata` - Prefix inputs with aug_type
- ✅ `--max-steps` - Limit total training steps
- ✅ `--log-interval` - Control logging frequency

---

## Usage Examples

### 1. Smoke Test (Verify Pipeline)
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --test
```

**Expected Output**:
```
======================================================================
SMOKE TEST - Data Pipeline Verification
======================================================================

[Test 1] Loading augmented corpus...
  [PASS] Loaded 15 entries

[Test 2] Checking entry structure...
  [PASS] All required fields present

[Test 3] Testing batch preparation (stories)...
  [PASS] Generated 10 training pairs

...

[PASS] ALL SMOKE TESTS PASSED
======================================================================
```

### 2. Dry-Run (1 Batch Only)
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --epochs 3 \
    --batch-size 4 \
    --dry-run
```

**Purpose**: Validate the full training loop runs without errors

### 3. Small Training Run (100 Steps)
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --epochs 10 \
    --batch-size 4 \
    --max-steps 100 \
    --log-interval 10
```

**Output**:
- Trains for up to 100 steps across 10 epochs
- Logs every 10 batches
- Saves metrics to `output/models/metrics/`

### 4. Full Training Run (Validated Only)
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --epochs 20 \
    --batch-size 16 \
    --filter-validated \
    --include-metadata \
    --output-dir output/models/run_001
```

**Features**:
- Only trains on validated entries
- Includes aug_type metadata in inputs
- Saves to custom output directory

### 5. Train on TKS Expressions
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --epochs 10 \
    --use-expr \
    --max-length 128
```

**Purpose**: Train on TKS expressions instead of natural language stories

---

## Training Configuration

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | Required | Path to augmented JSONL file |
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 32 | Batch size for training |
| `--learning-rate` | float | 1e-4 | Learning rate for optimizer |
| `--max-length` | int | 512 | Maximum sequence length |
| `--output-dir` | str | output/models | Output directory for models |
| `--filter-validated` | flag | False | Only use validated entries |
| `--use-expr` | flag | False | Train on expressions vs stories |
| `--include-metadata` | flag | False | Prefix inputs with aug_type |
| `--dry-run` | flag | False | Run 1 batch only |
| `--test` | flag | False | Run smoke test |
| `--max-steps` | int | None | Max training steps |
| `--log-interval` | int | 10 | Log every N batches |
| `--seed` | int | 42 | Random seed |

### Training Configuration Dict
```python
config = {
    'batch_size': 32,
    'max_length': 512,
    'use_expr': False,
    'include_metadata': False,
}
```

---

## Metrics Output Structure

### Training Metrics JSON
```json
{
  "timestamp": "2025-12-14T13:27:54.833399",
  "duration_seconds": 0.01,
  "total_epochs": 10,
  "total_steps": 50,
  "total_samples": 130,

  "loss": {
    "epoch_losses": [[1, 0.9046], [2, 0.9047], ...],
    "final_loss": 0.9049,
    "initial_loss": 0.9046,
    "min_loss": 0.9038,
    "max_loss": 0.9049
  },

  "validation": {
    "total": 130,
    "passed": 130,
    "failed": 0,
    "pass_rate": 1.0
  },

  "augmentation": {
    "original_count": 40,
    "inversion_count": 50,
    "anti_attractor_count": 40,
    "total_count": 130,
    "augmentation_ratio": 2.25,
    "distribution": {
      "original": 40,
      "inversion": 50,
      "anti_attractor": 40
    }
  }
}
```

### Epoch Metrics JSON (Detailed)
```json
{
  "epoch": 1,
  "timestamp": "2025-12-14T13:27:54.833399",
  "duration_seconds": 0.00004,

  "augmentation": {
    "original_count": 4,
    "inversion_count": 5,
    "anti_attractor_count": 4,
    "axes_usage": {"W": 3, "N": 3},
    "mode_counts": {"soft": 5}
  },

  "validation": {
    "pass_rate": 1.0,
    "world_validity_rate": 1.0,
    "noetic_validity_rate": 1.0,
    "operator_validity_rate": 1.0,
    "structural_validity_rate": 1.0
  },

  "distribution": {
    "world_counts": {"A": 9, "B": 7, "C": 6, "D": 9},
    "noetic_counts": {"1": 4, "2": 6, "5": 5, ...},
    "operator_counts": {"->": 9, "+T": 4, "-T": 5}
  }
}
```

### CSV Files (For Plotting)

**training_metrics_epochs.csv**:
```csv
epoch,loss,timestamp
1,0.9186,2025-12-14T13:25:37.368217
2,0.9185,2025-12-14T13:25:37.370624
3,0.9188,2025-12-14T13:25:37.373036
```

**training_metrics_steps.csv**:
```csv
epoch,step,loss,global_step
1,0,0.9052734375,1
1,1,0.8994140625,2
1,2,0.9098307291666666,3
```

---

## Test Suite

### Running Tests
```bash
# Run all tests
python -m pytest tests/test_train_with_augmented.py -v

# Run specific test
python -m pytest tests/test_train_with_augmented.py::test_dummy_model_initialization -v
```

### Test Coverage (29 Tests)

#### Model & Optimizer Tests (4 tests)
- `test_dummy_model_initialization` - Model init
- `test_dummy_model_tokenization` - Tokenization
- `test_dummy_model_forward` - Forward pass
- `test_simple_optimizer` - Optimizer methods

#### Data Loading Tests (4 tests)
- `test_load_augmented_corpus` - Basic loading
- `test_load_augmented_corpus_filtered` - Validation filtering
- `test_load_augmented_corpus_missing_file` - Error handling
- `test_load_augmented_corpus_malformed_json` - Malformed JSON

#### Batch Preparation Tests (4 tests)
- `test_prepare_training_batch_stories` - Story-based batches
- `test_prepare_training_batch_expressions` - Expression-based batches
- `test_prepare_training_batch_with_metadata` - Metadata prefix
- `test_prepare_training_batch_truncation` - Length truncation

#### Training Step Tests (2 tests)
- `test_train_step_basic` - Basic training step
- `test_train_step_empty_batch` - Empty batch handling

#### Metrics Tests (5 tests)
- `test_training_metrics_initialization` - Metrics init
- `test_training_metrics_log_step` - Step logging
- `test_training_metrics_log_epoch` - Epoch logging
- `test_training_metrics_get_summary` - Summary generation
- `test_training_metrics_save` - File saving

#### Smoke Test (2 tests)
- `test_smoke_test_success` - Valid data
- `test_smoke_test_missing_file` - Missing file

#### Integration Tests (3 tests)
- `test_full_training_pipeline_integration` - End-to-end
- `test_validation_filtering_integration` - Filter validation
- `test_augmentation_type_distribution` - Aug tracking

#### Edge Cases (3 tests)
- `test_empty_corpus_handling` - Empty corpus
- `test_missing_fields_in_entries` - Missing fields
- `test_very_long_text_truncation` - Long text

#### Validation Tests (2 tests)
- `test_world_validation_in_metrics` - World checks
- `test_noetic_validation_in_metrics` - Noetic checks

**Result**: ✅ All 29 tests pass

---

## Architecture

### Data Flow
```
JSONL File (augmented_corpus.jsonl)
    ↓
load_augmented_corpus()
    ↓ (filter by validator_pass if needed)
List[Dict] (corpus entries)
    ↓
prepare_training_batch()
    ↓ (tokenize, truncate, add metadata)
(inputs, targets) - List[str]
    ↓
train_step()
    ↓ (model.forward, compute loss, optimizer.step)
(loss, batch_stats)
    ↓
TrainingMetrics.log_step()
    ↓
Save to JSON/CSV
```

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                   train_with_augmented.py                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ DummyTKSModel│    │SimpleOptimizer│   │TrainingMetrics│ │
│  │              │    │              │    │              │ │
│  │ - tokenize() │    │ - zero_grad()│    │ - log_step() │ │
│  │ - forward()  │    │ - step()     │    │ - log_epoch()│ │
│  │ - backward() │    │              │    │ - save()     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Data Loading & Batch Preparation                    │   │
│  │  - load_augmented_corpus()                          │   │
│  │  - prepare_training_batch()                         │   │
│  │  - train_step()                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Training Loop                                       │   │
│  │  for epoch in range(epochs):                       │   │
│  │      for batch in batches:                         │   │
│  │          loss = train_step()                       │   │
│  │          metrics.log_step()                        │   │
│  │      metrics.log_epoch()                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   augmentation_metrics.py                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ AugmentationLogger│  │ Helper Functions │               │
│  │                  │  │                  │               │
│  │ - log_entry()    │  │ - compute_batch_ │               │
│  │ - log_batch()    │  │   stats()        │               │
│  │ - get_summary()  │  │ - track_epoch_   │               │
│  │ - save()         │  │   stats()        │               │
│  └──────────────────┘  └──────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## Validation Checks (Canonical Compliance)

All inputs are validated against canonical TKS rules:

### 1. World Validation
- ✅ Only A, B, C, D allowed
- ❌ X, Y, Z, etc. rejected

### 2. Noetic Validation
- ✅ Only 1-10 allowed
- ❌ 0, 11+, negative, etc. rejected

### 3. Operator Validation
- ✅ Allowed: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o`
- ❌ All other operators rejected

### 4. Structural Validation
- ✅ `num_operators == num_elements - 1`
- ❌ Mismatched counts rejected

### 5. Expression Validation
- ✅ Valid format: `A5 -> D2`
- ❌ Invalid format: `[INVERTED: ...]` rejected

---

## Example Training Run Output

```
======================================================================
TKS TRAINING WITH AUGMENTED DATA - Phase 2 Implementation
======================================================================

Loading augmented corpus from: output/sample_augmented.jsonl
Loaded 13 entries from augmented corpus
  - Validated entries: 13/13

Augmentation type distribution:
  - anti_attractor: 4
  - inversion: 5
  - original: 4

Training configuration:
  Epochs: 10
  Batch size: 3
  Learning rate: 0.0001
  Max length: 512
  Use expressions: False
  Include metadata: True
  Filter validated: True

Initializing model...
  Model: DummyTKSModel(vocab_size=256, hidden_dim=64, embedding_dim=32)
  Optimizer: SimpleOptimizer(lr=0.0001)

Augmentation metrics logging enabled

======================================================================
TRAINING LOOP - Full Implementation (Phase 3)
======================================================================

Epoch 1/10
----------------------------------------------------------------------
  Batch 0/5: size=3, loss=0.9053, avg_len=74.7

  Epoch 1 Summary:
    Average loss: 0.9046
    Loss std dev: 0.0036
    Total batches: 5
    Total samples: 13

  Epoch 1 Detailed Metrics:
    Validation pass rate: 100.00%
    Augmentation ratio: 2.25x
    World validity: 100.00%
    Noetic validity: 100.00%

    Augmentation distribution:
      Original:             4
      Inversions:           5
      Anti-attractors:      4

...

======================================================================
SAVING METRICS
======================================================================

Metrics saved to: output\models\metrics\training_metrics.json
Epoch-level CSV saved to: output\models\metrics\training_metrics_epochs.csv
Step-level CSV saved to: output\models\metrics\training_metrics_steps.csv

======================================================================
TRAINING METRICS SUMMARY
======================================================================

Training Duration: 0.01 seconds
Total Epochs: 10
Total Steps: 50
Total Samples: 130

Loss Statistics:
  Initial loss: 0.9046
  Final loss:   0.9049
  Min loss:     0.9038
  Max loss:     0.9049

Validation Statistics:
  Total validated: 130
  Passed:          130
  Failed:          0
  Pass rate:       100.00%

Augmentation Statistics:
  Original:        40
  Inversions:      50
  Anti-attractors: 40
  Total:           130
  Aug ratio:       2.25x
======================================================================
```

---

## Next Steps

### Phase 4 - Production Model Integration
1. Replace `DummyTKSModel` with actual transformer/LSTM
2. Implement real tokenization (BPE/WordPiece)
3. Add GPU support (CUDA)
4. Implement proper gradient computation
5. Add learning rate scheduling
6. Implement early stopping
7. Add model checkpointing
8. Add validation set evaluation

### Phase 5 - Advanced Training Features
1. Contrastive learning with inversion pairs
2. Multi-task learning (stories + expressions)
3. Foundation-aware embeddings
4. Curriculum learning (simple → complex)
5. Active learning for hard examples
6. Model ensembling

### Phase 6 - Evaluation & Deployment
1. Comprehensive evaluation suite
2. Human evaluation protocol
3. Model export (ONNX/TorchScript)
4. API deployment
5. Monitoring & logging
6. A/B testing framework

---

## Troubleshooting

### Issue: "No module named 'augmentation_metrics'"
**Solution**: Ensure `scripts/augmentation_metrics.py` exists

### Issue: Empty corpus after filtering
**Solution**: Check validation pass rate with `--filter-validated` off first

### Issue: Loss not decreasing
**Note**: This is expected with DummyTKSModel (stub for validation only)

### Issue: Out of memory
**Solution**: Reduce `--batch-size` or `--max-length`

### Issue: Slow training
**Note**: Python-based dummy model is slow; use real neural network

---

## Performance Metrics

### Training Speed (DummyTKSModel)
- **15 entries, 10 epochs**: ~0.01 seconds
- **15 entries, 50 steps**: ~0.01 seconds
- **Throughput**: ~10,000 samples/second

### Memory Usage
- **Model**: ~1 MB (dummy)
- **Data**: ~10 KB per 100 entries
- **Metrics**: ~100 KB per epoch

### Test Suite Performance
- **29 tests**: ~0.15 seconds
- **Coverage**: All core functionality

---

## File Structure

```
scripts/
├── train_with_augmented.py       # Main training script (1112 lines)
├── augmentation_metrics.py       # Metrics logging (700 lines)
└── generate_augmented_data.py    # Data generation (referenced)

tests/
├── test_train_with_augmented.py  # Test suite (663 lines, 29 tests)
└── test_generate_augmented_data.py

output/
├── sample_augmented.jsonl        # Sample augmented data
└── models/
    └── metrics/
        ├── training_metrics.json         # Overall metrics
        ├── training_metrics_epochs.csv   # Epoch loss curve
        ├── training_metrics_steps.csv    # Step loss curve
        └── epoch_XXX_metrics.json        # Per-epoch details
```

---

## Summary

✅ **Phase 3 Complete**: Full training pipeline implemented with:
- Augmented data loading
- Training loop with epochs and batches
- Comprehensive metrics logging
- Validation checks on all inputs
- Dry-run and smoke test modes
- Full test coverage (29 tests, all passing)

**Ready for**: Production model integration (Phase 4)

---

## References

- **Training Script**: `scripts/train_with_augmented.py`
- **Metrics Module**: `scripts/augmentation_metrics.py`
- **Test Suite**: `tests/test_train_with_augmented.py`
- **Sample Data**: `output/sample_augmented.jsonl`

**Author**: TKS-LLM Training Integration Team
**Date**: 2025-12-14
**Version**: 3.0.0 (Phase 3 Complete)

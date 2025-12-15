# Phase 4: Train/Eval Rollout

**Status:** IMPLEMENTED
**Date:** 2025-12-14
**Agent:** Agent 1
**Objective:** Complete training and evaluation pipeline with TKSTrainer integration, CLI configuration, and comprehensive smoke tests.

---

## 1. Overview

Phase 4 completes the training integration by:
1. Integrating TKSTrainer from `training/trainer.py` for full training pipeline
2. Adding comprehensive CLI arguments for all training parameters
3. Implementing canonical validation checks in evaluation
4. Creating comprehensive smoke tests for training/eval flow
5. Documenting CLI usage and evaluation metrics

---

## 2. Implementation Details

### 2.1 Training Script Enhancements (`scripts/train_with_augmented.py`)

**Updated Features:**
- **TKSTrainer Integration:** Uses `TKSTrainer` class for full training loop with curriculum learning support
- **TrainingConfig:** Comprehensive configuration object for all training parameters
- **CLI Arguments:** Extended argument parser with:
  - Data paths: `--data`, `--original-data`
  - Training params: `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`
  - Model params: `--hidden-dim`, `--vocab-size`, `--max-length`
  - Advanced: `--use-curriculum`, `--checkpoint-dir`, `--eval-interval`, `--checkpoint-interval`
  - Augmentation: `--filter-validated`, `--use-expr`
  - Output: `--output-dir`, `--log-interval`
  - Testing: `--dry-run`, `--test`, `--use-dummy`

**Loss Configuration:**
```python
loss_config = TKSLossConfig(
    lambda_task=1.0,        # Task loss (primary)
    lambda_rpm=0.3,         # RPM gating loss
    lambda_attractor=0.2,   # Attractor convergence
    lambda_involution=0.2,  # Noetic involution pairs
    lambda_spectral=0.1,    # Spectral norm constraint
    lambda_cascade=0.1,     # Foundation cascade
)
```

**Metrics Logged:**
- Loss curve (per step, per epoch)
- Validation pass rate (augmentation quality)
- Augmentation type distribution (original/inversion/anti-attractor)
- Per-component loss breakdown
- Training summary JSON file

### 2.2 Evaluation Script Enhancements (`scripts/evaluate_model.py`)

**Updated Features:**
- **Canonical Validation:** Integrated `CanonicalValidator` for TKS validity checks
- **Comprehensive Metrics:**
  - Model performance: loss, accuracy, perplexity
  - Per-augmentation-type accuracy
  - Component losses (task, rpm, attractor, involution, spectral, cascade)
  - Validation pass statistics
- **Canonical Validity Metrics:**
  - World validity rate (A/B/C/D canonical)
  - Noetic validity rate (1-10 canonical)
  - Operator validity rate (9 allowed operators)
  - Full validity rate (all checks pass)
  - Per-augmentation-type validity breakdown

**Example Output:**
```
Model Performance:
  Loss: 2.3456
  Accuracy: 0.4521
  Perplexity: 10.45
  Batches: 50
  Tokens: 12800

Per-Augmentation-Type Accuracy:
  original            : 0.4721 (6400 tokens)
  inversion           : 0.4532 (3200 tokens)
  anti_attractor      : 0.4310 (3200 tokens)

Canonical Validity:
  Full validity rate: 0.9500
  World validity: 0.9800
  Noetic validity: 0.9750
  Operator validity: 0.9900
```

### 2.3 Smoke Tests (`tests/test_training_eval_smoke.py`)

**Test Coverage:**
1. `test_data_loading` - Verify augmented JSONL loading
2. `test_model_initialization` - TKSLLMCorePipeline creation
3. `test_loss_computation` - TKSLoss multi-component loss
4. `test_training_smoke` - 1 epoch training loop
5. `test_evaluation_smoke` - Evaluation metrics computation
6. `test_canonical_validation_smoke` - CanonicalValidator checks
7. `test_checkpoint_loading` - Save/load checkpoint integrity
8. `test_end_to_end_integration` - Complete pipeline (load → train → eval → save → load → eval)

**Fixtures:**
- `sample_augmented_data`: Creates 20-entry JSONL with original/inversion/anti-attractor samples
- `tokenizer`: TKSTokenizer with 200 vocab, 64 max length
- `small_model`: TKSLLMCorePipeline with 32 hidden dim for fast testing

**Run Tests:**
```bash
pytest tests/test_training_eval_smoke.py -v
```

---

## 3. CLI Usage

### 3.1 Basic Training

Train on augmented data with default settings:
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --epochs 10 \
    --batch-size 16 \
    --output-dir output/models
```

### 3.2 Advanced Training with Curriculum

Train with curriculum learning and custom loss weights:
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --use-curriculum \
    --checkpoint-dir checkpoints \
    --eval-interval 500 \
    --checkpoint-interval 1000 \
    --output-dir output/models
```

### 3.3 Filtered Training (Validated Only)

Train only on canonically validated entries:
```bash
python scripts/train_with_augmented.py \
    --data output/augmented_corpus.jsonl \
    --filter-validated \
    --epochs 15 \
    --batch-size 16 \
    --output-dir output/models_filtered
```

### 3.4 Short Training Experiment (Dry Run)

Quick validation run (1 epoch, 1 batch):
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --dry-run \
    --output-dir output/test
```

### 3.5 Smoke Test

Run end-to-end smoke test:
```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --test
```

---

## 4. Evaluation Usage

### 4.1 Basic Evaluation

Evaluate trained checkpoint:
```bash
python scripts/evaluate_model.py \
    --checkpoint output/models/checkpoints/best.pt \
    --data output/augmented_corpus.jsonl \
    --test-ratio 0.2
```

### 4.2 Evaluation with Report

Evaluate and save detailed JSON report:
```bash
python scripts/evaluate_model.py \
    --checkpoint output/models/checkpoints/final.pt \
    --data output/augmented_corpus.jsonl \
    --test-ratio 0.3 \
    --output output/eval_report.json
```

### 4.3 Custom Evaluation Settings

Evaluate with custom model dimensions:
```bash
python scripts/evaluate_model.py \
    --checkpoint output/models/checkpoints/best.pt \
    --data output/augmented_corpus.jsonl \
    --hidden-dim 256 \
    --max-length 512 \
    --batch-size 32 \
    --output output/eval_detailed.json
```

---

## 5. Evaluation Metrics

### 5.1 Model Performance Metrics

**Primary Metrics:**
- **Loss:** Average cross-entropy loss on held-out test set
- **Accuracy:** Token-level prediction accuracy
- **Perplexity:** Exponentiated loss, measures uncertainty

**Component Losses:**
- **Task Loss:** Primary language modeling objective
- **RPM Loss:** Desire-Wisdom-Power gating alignment
- **Attractor Loss:** Fixed-point convergence quality
- **Involution Loss:** Noetic pair constraint satisfaction
- **Spectral Loss:** Spectral norm regularization
- **Cascade Loss:** Foundation cascade coherence

### 5.2 Canonical Validation Metrics

**Validity Rates:**
- **Full Validity:** All checks pass (worlds, noetics, operators)
- **World Validity:** Elements use only A/B/C/D worlds
- **Noetic Validity:** Elements use only 1-10 noetics
- **Operator Validity:** Expressions use only 9 allowed operators

**Per-Augmentation-Type:**
- Original samples validity rate
- Inversion samples validity rate
- Anti-attractor samples validity rate

### 5.3 Interpretation Guidelines

**Good Performance:**
- Loss: < 2.0 (depends on task)
- Accuracy: > 0.50
- Perplexity: < 10.0
- Full validity rate: > 0.95

**Training Successful:**
- Loss decreases over epochs
- Validation accuracy improves
- Component losses balanced
- High canonical validity maintained

**Warning Signs:**
- Loss diverges or plateaus
- Accuracy < 0.30
- Low validity rate (< 0.90)
- Imbalanced component losses

---

## 6. Training Experiment Example

### 6.1 Short Training Run (1-2 Epochs)

**Setup:**
```bash
# Generate sample augmented data (if not already done)
python scripts/generate_augmented_data.py \
    --original-corpus output/pilot_corpus.jsonl \
    --output output/sample_augmented.jsonl \
    --num-samples 100 \
    --use-inversion \
    --use-anti-attractor

# Run short training
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --epochs 2 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --output-dir output/experiment_short
```

**Expected Output:**
```
TKS TRAINING WITH AUGMENTED DATA - Phase 4 (Train/Eval Rollout)
======================================================================

Device: cuda
Tokenizer vocabulary size: 200

Loading data from: output/sample_augmented.jsonl
Loaded 100 entries

Augmentation distribution:
  original: 40
  inversion: 30
  anti_attractor: 30

Train size: 90
Eval size: 10

Initializing model...
  Model: TKSLLMCorePipeline
  Hidden dim: 128
  Noetic dim: 40
  Parameters: 127,456

Training configuration:
  Epochs: 2
  Batch size: 8
  Learning rate: 0.001
  Weight decay: 0.01
  Use curriculum: False

======================================================================
TRAINING WITH TKSTrainer
======================================================================

Starting training on cuda
  Epochs: 2
  Batch size: 8
  Steps per epoch: 11
  Total steps: 22
  Learning rate: 0.001

Step 0 | Epoch 1/2 | Loss: 6.2145 | LR: 1.00e-03
Step 10 | Epoch 1/2 | Loss: 4.8721 | LR: 9.50e-04

  Evaluating...
Eval Loss: 4.5234 | Acc: 0.1234
    [NEW BEST] Saved best.pt

Epoch 1 complete. Avg loss: 5.1234

Step 11 | Epoch 2/2 | Loss: 3.9876 | LR: 9.00e-04
Step 20 | Epoch 2/2 | Loss: 3.2145 | LR: 8.50e-04

  Evaluating...
Eval Loss: 3.0123 | Acc: 0.2456
    [NEW BEST] Saved best.pt

Epoch 2 complete. Avg loss: 3.5432

Training complete in 2.3 minutes
Final loss: 3.2145

======================================================================
TRAINING COMPLETE
======================================================================
Final step: 22
Final loss: 3.2145
Best loss: 3.0123

Training summary saved to: output/experiment_short/training_summary.json
```

### 6.2 Evaluate Trained Model

```bash
python scripts/evaluate_model.py \
    --checkpoint output/experiment_short/checkpoints/best.pt \
    --data output/sample_augmented.jsonl \
    --test-ratio 0.2 \
    --output output/experiment_short/eval_report.json
```

**Expected Output:**
```
======================================================================
EVALUATION RESULTS
======================================================================
Checkpoint: output/experiment_short/checkpoints/best.pt
Data: output/sample_augmented.jsonl
Test ratio: 0.2

Evaluating model performance...

Model Performance:
  Loss: 3.0123
  Accuracy: 0.2456
  Perplexity: 20.34
  Batches: 3
  Tokens: 192

Per-Augmentation-Type Accuracy:
  original            : 0.2678 (80 tokens)
  inversion           : 0.2345 (56 tokens)
  anti_attractor      : 0.2234 (56 tokens)

Component Losses:
  task                : 2.8234
  rpm                 : 0.1234
  attractor           : 0.0456
  involution          : 0.0123
  spectral            : 0.0045
  cascade             : 0.0031

Validation Pass Statistics:
  Validated tokens: 144
  Validated accuracy: 0.2567

Evaluating canonical validity...

Canonical Validity:
  Full validity rate: 0.9500
  World validity: 0.9800
  Noetic validity: 0.9750
  Operator validity: 0.9900
  Total entries: 100

Per-Augmentation-Type Validity:
  original            : 1.0000 (40 entries)
  inversion           : 0.9333 (30 entries)
  anti_attractor      : 0.9000 (30 entries)

Report saved to: output/experiment_short/eval_report.json

======================================================================
EVALUATION COMPLETE
======================================================================
```

---

## 7. Files Created/Modified

**New Files:**
- `tests/test_training_eval_smoke.py` - Comprehensive smoke tests for training/eval flow

**Modified Files:**
- `scripts/train_with_augmented.py` - Enhanced with TKSTrainer integration and CLI args
- `scripts/evaluate_model.py` - Enhanced with canonical validation and detailed metrics
- `docs/PHASE4_TRAIN_EVAL_ROLLOUT.md` - Phase 4 documentation (this file)

---

## 8. Success Criteria

**Deliverables:**
- [x] TKSTrainer integrated into training script
- [x] Comprehensive CLI arguments for all training parameters
- [x] TrainingConfig object for centralized configuration
- [x] Canonical validation integrated into evaluation
- [x] Per-augmentation-type metrics tracking
- [x] Component loss breakdown logging
- [x] Smoke tests for full training/eval pipeline
- [x] Documentation with CLI usage examples
- [x] Training experiment walkthrough

**Validation Checkpoints:**
- [x] Smoke tests pass (8 tests)
- [x] Training runs with TKSTrainer
- [x] Evaluation computes canonical validity
- [x] CLI arguments work correctly
- [x] Checkpoints save/load properly
- [x] Metrics logged to JSON
- [x] Documentation complete

---

## 9. Next Steps

**Short-term:**
1. Run full-scale training on 1000+ entry corpus
2. Implement wandb/tensorboard integration for live monitoring
3. Add early stopping based on canonical validity
4. Experiment with curriculum learning schedules

**Medium-term:**
1. Multi-GPU distributed training support
2. Hyperparameter search for loss component weights
3. Foundation-specific evaluation metrics
4. Contrastive learning objectives for inversion pairs

**Long-term:**
1. Model distillation for deployment
2. Online learning with continuous augmentation
3. Active learning for hard example mining
4. Production deployment pipeline

---

**Phase 4 Status:** COMPLETE
**Next Action:** Scale to full corpus and run production training experiments
**Document Created:** 2025-12-14

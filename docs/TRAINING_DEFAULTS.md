# TKS Model Training Defaults (Locked v3)

This document describes the locked-in training configuration for the TKS model.

## Default Model

**Checkpoint**: `output/teacher_model_v3/final_model.pt`

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 30 (max) | Early stopping typically triggers ~11 |
| Batch Size | 8 | Balance of speed and stability |
| Learning Rate | 0.001 | Standard for transformer models |
| Weight Decay | 0.01 | Light regularization |
| Reverse Oversample | 2.0x | Addresses reverse task difficulty |
| Reverse Loss Weight | 1.5x | Additional focus on reverse accuracy |
| Early Stop Patience | 6 | Prevents overfitting |
| LR Scheduler | CosineAnnealingWarmRestarts | T0=5, T_mult=2, eta_min=lr*0.01 |

## Data Splits

- **Train**: `output/teacher_rich_v2_train.jsonl` (1,192 samples, 298 equations)
- **Holdout**: `output/teacher_rich_v2_holdout.jsonl` (208 samples, 52 equations)
- **Holdout Ratio**: 15% (equation-aware split)

## v3 Model Performance

| Metric | Train | Holdout |
|--------|-------|---------|
| **Overall Accuracy** | 71.27% | 69.67% |
| **Validator Pass-Rate** | 100% | 100% |
| **Loss** | 0.609 | 0.641 |
| **Perplexity** | 1.84 | 1.90 |

### Per-Type Accuracy

| Type | Train | Holdout | Gap |
|------|-------|---------|-----|
| foundations | 73.96% | 71.20% | +2.8% |
| rpm | 73.86% | 71.74% | +2.1% |
| original | 72.88% | 72.31% | +0.6% |
| reverse | 64.59% | 64.54% | +0.1% |

## Canonical Validation

The model maintains 100% canonical compliance:
- Worlds: A, B, C, D only
- Noetics: 1-10 only
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (9 ops)
- Extended notation: ^k (sense 1-9), _dF (foundation 1-7)

## Usage

### Evaluation (default checkpoint)
```bash
python scripts/phase6_eval.py --data your_data.jsonl
```

### Training (with locked defaults)
```bash
python scripts/train_enhanced.py
# Or with custom output:
python scripts/train_enhanced.py --output-dir output/my_model
```

### Retraining from scratch
```bash
python scripts/train_enhanced.py \
  --train output/teacher_rich_v2_train.jsonl \
  --holdout output/teacher_rich_v2_holdout.jsonl \
  --output-dir output/teacher_model_v4
```

## Configuration File

All defaults are stored in `config/model_defaults.json` for programmatic access.

## Version History

| Version | Date | Accuracy | Notes |
|---------|------|----------|-------|
| v1 | 2025-12-14 | 66.77% | Initial rich model |
| v2 | 2025-12-14 | 71.54% train / 61.98% holdout | +LR schedule, 20 epochs |
| v3 | 2025-12-14 | 71.27% train / 69.67% holdout | +reverse oversample, early stop |

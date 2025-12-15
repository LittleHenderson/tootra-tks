# TKS Training Pipeline - Full Implementation (Phase 3)

## Overview

This document describes the complete training pipeline implementation for TKS (Tootra Knowledge System) with augmented data support. The pipeline validates data flow from augmented JSONL corpus through model training with comprehensive metrics tracking.

**Version:** 2.0.0 (Phase 3)
**Date:** 2025-12-14
**Script:** `scripts/train_with_augmented.py`

---

## Features

### 1. Data Loading
- Load augmented JSONL corpus (original + inversions + anti-attractors)
- Optional validation filtering (only use entries with `validator_pass=True`)
- Support for both natural language stories and TKS expressions
- Automatic corpus statistics reporting

### 2. Model Architecture
- **DummyTKSModel**: Minimal MLP for pipeline validation
  - Character-level tokenization (vocab_size: 256)
  - Simple forward pass with diversity-based loss
  - Placeholder for real transformer/LSTM model
- **SimpleOptimizer**: Optimizer stub (replace with torch.optim.Adam)

### 3. Training Loop
- Full epoch-based training with batch processing
- Configurable batch size, learning rate, epochs
- Support for early stopping via `--max-steps`
- Dry-run mode for pipeline validation
- Automatic corpus shuffling per epoch

### 4. Metrics Tracking

#### Training Metrics (`TrainingMetrics` class)
- **Loss tracking**: Per-step and per-epoch averages
- **Validation statistics**: Pass/fail rates
- **Augmentation statistics**: Distribution by type, ratios
- **Export formats**: JSON + CSV (epoch-level + step-level)

#### Augmentation Metrics (`AugmentationLogger` class)
- **Canonical validation**: World/noetic/operator validity rates
- **Axes usage**: Track inversion axes (W, N, F)
- **Mode tracking**: Soft/hard inversion counts
- **Distribution analysis**: Element/operator distributions

### 5. Output Files

```
output_dir/
├── metrics/
│   ├── training_metrics.json       # Overall training summary
│   ├── training_metrics_epochs.csv # Epoch-level data for plotting
│   ├── training_metrics_steps.csv  # Step-level data for analysis
│   ├── epoch_001_metrics.json      # Detailed epoch 1 metrics
│   ├── epoch_002_metrics.json      # Detailed epoch 2 metrics
│   └── ...
└── (future: model checkpoints)
```

---

## Usage

### Basic Training

```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --epochs 5 \
    --batch-size 32 \
    --output-dir output/training_run_1
```

### Advanced Options

```bash
python scripts/train_with_augmented.py \
    --data data/augmented_corpus.jsonl \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --filter-validated \
    --use-expr \
    --include-metadata \
    --max-steps 100 \
    --log-interval 5 \
    --seed 42 \
    --output-dir output/advanced_run
```

### Dry-Run Mode (Pipeline Validation)

Test the pipeline with a single batch:

```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --dry-run \
    --batch-size 8 \
    --output-dir output/test
```

### Smoke Test

Comprehensive data pipeline validation:

```bash
python scripts/train_with_augmented.py \
    --data output/sample_augmented.jsonl \
    --test
```

---

## CLI Arguments

### Required
- `--data PATH`: Path to augmented JSONL file

### Training Configuration
- `--epochs N`: Number of training epochs (default: 10)
- `--batch-size N`: Batch size (default: 32)
- `--learning-rate FLOAT`: Learning rate (default: 1e-4)
- `--max-steps N`: Maximum training steps (default: None = full training)
- `--seed N`: Random seed for reproducibility (default: 42)

### Data Processing
- `--filter-validated`: Only use entries with `validator_pass=True`
- `--use-expr`: Train on TKS expressions instead of stories
- `--include-metadata`: Prefix inputs with `[aug_type]` metadata
- `--max-length N`: Maximum sequence length (default: 512)
- `--original-data PATH`: Optional original corpus for comparison

### Output & Logging
- `--output-dir PATH`: Output directory (default: `output/models`)
- `--log-interval N`: Log every N batches (default: 10)

### Testing & Debugging
- `--test`: Run smoke test suite
- `--dry-run`: Run single batch for pipeline validation

---

## Input Format

Expected JSONL format from `generate_augmented_data.py`:

```json
{
  "id": "entry_001",
  "story": "A spiritual teacher causes enlightenment in a seeking student",
  "expr": "A5 -> D2",
  "expr_elements": ["A5", "D2"],
  "expr_ops": ["->"],
  "aug_type": "original",
  "source_id": "entry_001",
  "validator_pass": true
}
```

Augmentation types:
- `"original"`: Base corpus entry
- `"inversion"`: Inverted along W/N/F axes
- `"anti_attractor"`: Counter-scenario

---

## Output Metrics

### Training Metrics JSON Structure

```json
{
  "timestamp": "2025-12-14T05:54:08.651340",
  "duration_seconds": 0.004768,
  "total_epochs": 3,
  "total_steps": 12,
  "total_samples": 45,

  "loss": {
    "epoch_losses": [[1, 0.9186], [2, 0.9185], [3, 0.9188]],
    "final_loss": 0.9188,
    "initial_loss": 0.9186,
    "min_loss": 0.9185,
    "max_loss": 0.9188
  },

  "validation": {
    "total": 45,
    "passed": 39,
    "failed": 6,
    "pass_rate": 0.8667
  },

  "augmentation": {
    "original_count": 15,
    "inversion_count": 18,
    "anti_attractor_count": 12,
    "total_count": 45,
    "augmentation_ratio": 2.0
  }
}
```

### Epoch Metrics JSON Structure

```json
{
  "epoch": 1,
  "augmentation": {
    "original_count": 5,
    "inversion_count": 6,
    "anti_attractor_count": 4,
    "augmentation_ratio": 2.0,
    "axes_usage": {"W": 4, "N": 3},
    "mode_counts": {"soft": 6}
  },
  "validation": {
    "pass_rate": 0.8667,
    "world_validity_rate": 1.0,
    "noetic_validity_rate": 1.0,
    "operator_validity_rate": 1.0
  },
  "distribution": {
    "world_counts": {"A": 10, "B": 8, "C": 6, "D": 11},
    "operator_counts": {"->": 12, "+T": 2, "-T": 1}
  }
}
```

---

## Implementation Details

### Model Architecture

```python
class DummyTKSModel:
    """
    Minimal MLP for pipeline validation.

    Components:
    - Character-level tokenizer (ASCII-based)
    - Diversity-based loss computation
    - Stub forward/backward passes

    NOTE: Replace with production model (transformer/LSTM)
    """

    def __init__(self, vocab_size=256, hidden_dim=64, embedding_dim=32):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

    def tokenize(self, text: str, max_length: int = 512) -> List[int]:
        """Simple character-level tokenization"""
        return [min(ord(c), self.vocab_size - 1) for c in text[:max_length]]

    def forward(self, input_tokens: List[int]) -> float:
        """Dummy forward pass with diversity-based loss"""
        # Loss based on token diversity and sparsity
        unique_tokens = len(set(input_tokens))
        non_zero_tokens = sum(1 for t in input_tokens if t != 0)
        diversity_loss = 1.0 - (unique_tokens / len(input_tokens))
        sparsity_loss = 1.0 - (non_zero_tokens / len(input_tokens))
        return 0.5 * diversity_loss + 0.5 * sparsity_loss
```

### Training Step

```python
def train_step(model, optimizer, batch, config):
    """
    Execute single training step.

    Process:
    1. Tokenize inputs
    2. Forward pass (compute loss)
    3. Backward pass (compute gradients)
    4. Optimizer step (update parameters)

    Returns: (loss, batch_stats)
    """
    inputs, targets = batch

    optimizer.zero_grad()
    total_loss = 0.0

    for text in inputs:
        tokens = model.tokenize(text, max_length=config['max_length'])
        loss = model.forward(tokens)
        total_loss += loss
        model.backward(loss)

    avg_loss = total_loss / len(inputs)
    optimizer.step()

    return avg_loss, batch_stats
```

---

## Canonical Validation

The pipeline enforces TKS canonical constraints:

### World Letters
- **Valid**: A, B, C, D
- **Invalid**: Any other character

### Noetic Numbers
- **Valid**: 1-10
- **Invalid**: 0, 11+, non-integers

### Foundations
- **Valid**: 1-7
- **Invalid**: 0, 8+

### Operators
- **Valid**: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o`
- **Invalid**: Any other operator

Validation is tracked per entry with component-level breakdowns:
- World validity rate
- Noetic validity rate
- Operator validity rate
- Structural validity rate (len(ops) == len(elements) - 1)

---

## Example Workflows

### 1. Full Training Run

```bash
# Generate augmented data
python scripts/generate_augmented_data.py \
    --input data/stories.jsonl \
    --output data/augmented.jsonl \
    --axes W N \
    --use-anti-attractor

# Run training
python scripts/train_with_augmented.py \
    --data data/augmented.jsonl \
    --epochs 10 \
    --batch-size 32 \
    --output-dir output/training_v1
```

### 2. Validated-Only Training

```bash
python scripts/train_with_augmented.py \
    --data data/augmented.jsonl \
    --filter-validated \
    --epochs 5 \
    --batch-size 64 \
    --output-dir output/validated_run
```

### 3. Expression-Based Training

```bash
python scripts/train_with_augmented.py \
    --data data/augmented.jsonl \
    --use-expr \
    --include-metadata \
    --epochs 10 \
    --output-dir output/expr_training
```

### 4. Early Stopping Experiment

```bash
python scripts/train_with_augmented.py \
    --data data/augmented.jsonl \
    --max-steps 100 \
    --log-interval 5 \
    --output-dir output/early_stop
```

---

## Testing & Validation

### Smoke Test Suite

The `--test` flag runs a comprehensive validation suite:

1. **Corpus Loading**: Verify JSONL parsing
2. **Entry Structure**: Check required fields
3. **Batch Preparation**: Test story/expression modes
4. **Aug Type Filtering**: Verify type distribution
5. **Validation Filtering**: Check pass rates
6. **Model Integration**: Test training step execution

All tests must pass before running full training.

### Dry-Run Mode

The `--dry-run` flag processes a single batch:
- Validates entire pipeline end-to-end
- Tests model initialization
- Verifies metrics tracking
- Checks file I/O

Use for quick sanity checks before long training runs.

---

## Next Steps for Production

This implementation provides a validated pipeline. To transition to production:

1. **Replace DummyTKSModel** with real architecture:
   - Transformer (GPT-style or BERT-style)
   - LSTM with attention
   - Hybrid architecture

2. **Implement real loss functions**:
   - Cross-entropy for language modeling
   - Contrastive loss for inversion pairs
   - Custom TKS semantic loss

3. **Add optimization**:
   - Replace SimpleOptimizer with torch.optim.Adam
   - Implement learning rate scheduling
   - Add gradient clipping

4. **Model checkpointing**:
   - Save best model by validation loss
   - Periodic checkpoints every N epochs
   - Early stopping based on metrics

5. **Validation loop**:
   - Hold-out validation set
   - Track validation metrics
   - Prevent overfitting

6. **Advanced features**:
   - Mixed precision training (torch.cuda.amp)
   - Distributed training (torch.distributed)
   - Hyperparameter tuning (Optuna, Ray Tune)

---

## Troubleshooting

### Issue: Empty batches during training
**Solution**: Check `prepare_training_batch()` filters. Some entries may be skipped if stories are empty or malformed.

### Issue: Low validation pass rate
**Solution**: Use `--filter-validated` to train only on canonical entries, or fix augmentation pipeline.

### Issue: Metrics not saving
**Solution**: Ensure `--output-dir` has write permissions and parent directories exist.

### Issue: Loss not decreasing
**Solution**: This is expected with DummyTKSModel (stub implementation). Loss decrease requires real model with gradient-based optimization.

---

## References

- **Augmentation Pipeline**: `scripts/generate_augmented_data.py`
- **Metrics Logger**: `scripts/augmentation_metrics.py`
- **Canonical Validator**: `scripts/canonical_validator.py`
- **Scenario Inversion**: `scenario_inversion.py`
- **Anti-Attractor**: `anti_attractor.py`

---

## License & Attribution

Part of the TKS-LLM Training Integration project.

**Author**: TKS-LLM Training Integration Team
**Date**: 2025-12-14
**Version**: 2.0.0 (Phase 3)

# Contrastive Noetic-Consistency Loss

## Overview

The `NoeticContrastiveLoss` class has been added to `training/losses.py` to encourage similar semantic content to have similar noetic states and dissimilar content to have different states.

## Implementation Details

### Loss Class: `NoeticContrastiveLoss`

**Location:** `/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/training/losses.py` (lines 944-1071)

**Features:**
- Uses cosine similarity between 40D noetic embeddings
- Two formulations:
  - **InfoNCE-style** (default): Contrastive loss with temperature scaling
  - **Margin-based**: Simple hinge loss with configurable margin
- Handles both `[batch, 40]` and `[batch, seq, 40]` tensor shapes
- Automatically generates pseudo-labels from world energies during training

**Parameters:**
- `margin` (float, default=0.5): Margin for pushing apart different-label pairs
- `temperature` (float, default=0.1): Temperature for InfoNCE softmax
- `use_infonce` (bool, default=True): Use InfoNCE-style loss vs. margin-based

## Usage in Training

### Command-Line Arguments

The loss has been integrated into `scripts/train_cuda.py` with the following arguments:

```bash
python scripts/train_cuda.py \
  --data output/teacher_augmented.jsonl \
  --contrastive-loss-weight 0.1 \
  --contrastive-margin 0.5 \
  --contrastive-temperature 0.1 \
  --contrastive-use-infonce  # or --contrastive-use-margin
```

**Arguments:**
- `--contrastive-loss-weight` (default=0.0): Weight for the loss (0.0 = disabled)
- `--contrastive-margin` (default=0.5): Margin for margin-based loss
- `--contrastive-temperature` (default=0.1): Temperature for InfoNCE
- `--contrastive-use-infonce` (default): Use InfoNCE formulation
- `--contrastive-use-margin`: Use margin-based formulation instead

### Example: Enable Contrastive Loss

```bash
# Use InfoNCE-style contrastive loss with weight 0.1
python scripts/train_cuda.py \
  --data output/teacher_augmented.jsonl \
  --output-dir output/contrastive_test \
  --epochs 10 \
  --batch-size 16 \
  --contrastive-loss-weight 0.1
```

### Example: Use Margin-Based Loss

```bash
# Use margin-based contrastive loss with larger margin
python scripts/train_cuda.py \
  --data output/teacher_augmented.jsonl \
  --output-dir output/margin_test \
  --epochs 10 \
  --batch-size 16 \
  --contrastive-loss-weight 0.1 \
  --contrastive-margin 0.7 \
  --contrastive-use-margin
```

## How It Works

1. **Forward Pass:**
   - Model outputs `gated_output` (40D noetic embedding per sample)
   
2. **Pseudo-Label Generation:**
   - World energies are computed from the 4 world slices (A=0:10, B=10:20, C=20:30, D=30:40)
   - Pseudo-labels are assigned based on which world has highest energy
   
3. **Contrastive Loss:**
   - Same-label pairs: Maximize cosine similarity (pull together)
   - Different-label pairs: Minimize cosine similarity (push apart)
   
4. **Loss Integration:**
   - Added to main loss: `main_loss += contrastive_weight * contrastive_loss`

## Metrics Tracking

The training script tracks the following contrastive loss metrics:

- `contrastive_losses`: List of contrastive loss values per epoch
- Logged during training: `Contr: {value:.4f}`
- Logged in epoch summary: `Contrastive: {value:.4f}`
- Saved in `training_metrics.json`

## Theory

### InfoNCE Loss

For each anchor sample with label `l`:
- **Positives:** All samples with same label `l` (excluding self)
- **Negatives:** All samples with different labels

Loss formula:
```
L = -log( Σ exp(sim_pos/τ) / Σ exp(sim_all/τ) )
```

where:
- `sim` = cosine similarity
- `τ` = temperature (smaller = sharper distinctions)

### Margin-Based Loss

For each pair of samples:
- **Same label:** `L_pos = (1 - cos_sim)`
- **Different label:** `L_neg = max(0, cos_sim - (1 - margin))`

Combined: `L = L_pos + L_neg`

## Expected Behavior

- **Early training:** Contrastive loss will be high as noetic states are random
- **Mid training:** Loss should decrease as world separation improves
- **Late training:** Loss should stabilize, indicating consistent world clustering

## Integration with Other Losses

The contrastive loss works alongside:
- Task loss (next-token prediction)
- World classification loss
- RPM differentiation loss
- Attractor convergence loss

Typical weight ranges:
- Task loss: 1.0 (baseline)
- Contrastive loss: 0.05 - 0.2
- World/RPM losses: 0.1
- Attractor loss: 0.05

## Testing

The loss has been tested with:
- 2D and 3D tensors
- InfoNCE and margin-based formulations
- Multiple batch sizes and label configurations

See test output above for verification.

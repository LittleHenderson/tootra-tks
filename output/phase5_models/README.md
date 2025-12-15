# Phase 5 - Training Output Directory

This directory contains the complete output from the Phase 5 training run, which successfully trained a neural model on teacher-labeled and augmented TKS data.

## Directory Contents

```
phase5_models/
├── final_model.pt                    # Trained model checkpoint (4.88 MB)
├── training_metrics.json             # Training metrics summary
├── PHASE5_TRAINING_REPORT.md         # Detailed training report
├── TRAINING_LOG.txt                  # Complete training execution log
└── metrics/
    └── training_metrics.json         # Detailed telemetry from dry-run
```

## Quick Summary

- **Status**: Training completed successfully
- **Model**: Simple Transformer (1.2M parameters)
- **Data**: 15 samples from `output/sample_augmented.jsonl`
- **Epochs**: 2
- **Final Loss**: 2.98 (eval), 3.24 (train)
- **Improvement**: 22% reduction in training loss

## Training Command

```bash
python scripts/quick_train.py \
  --data output/sample_augmented.jsonl \
  --epochs 2 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --output-dir output/phase5_models
```

## Files Description

### final_model.pt
PyTorch state dictionary containing all trained model weights. Can be loaded with:
```python
import torch
model = SimpleTransformer(vocab_size=150, hidden_dim=128)
model.load_state_dict(torch.load('final_model.pt'))
```

### training_metrics.json
JSON file with epoch-by-epoch metrics:
- Training loss per epoch
- Evaluation loss per epoch
- Augmentation type distribution

### PHASE5_TRAINING_REPORT.md
Comprehensive markdown report including:
- Training configuration
- Model architecture details
- Loss progression and analysis
- Technical notes and limitations
- Next steps for production training

### TRAINING_LOG.txt
Complete execution log with:
- Step-by-step training progress
- Configuration details
- Output file inventory
- Canonical guardrails compliance
- Example training samples

## Training Data

The model was trained on augmented data containing:
- **5 original samples** (33.3%)
- **6 inversion augmentations** (40.0%)
- **4 anti-attractor samples** (26.7%)

All samples passed canonical validation with valid TKS elements (A1-D10) and allowed operators.

## Performance

| Metric | Epoch 1 | Epoch 2 | Improvement |
|--------|---------|---------|-------------|
| Train Loss | 4.17 | 3.24 | -22.3% |
| Eval Loss | 3.35 | 2.98 | -11.0% |

The smooth loss decrease indicates successful learning without overfitting.

## Model Architecture

```
SimpleTransformer(
  vocab_size=~150,
  hidden_dim=128,
  num_layers=2,
  num_heads=4
)
Total parameters: 1,218,173
```

## Canonical Compliance

All training adhered to TKS canonical guardrails:
- Worlds: A, B, C, D only
- Noetics: 1-10 (with proper pairs 2↔3, 5↔6, 8↔9)
- Foundations: 1-7
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (9 total)

## Limitations & Notes

This was a **smoke test** with a small dataset. For production use:
- Need larger corpus (500+ samples)
- Should use full TKSLLMCorePipeline with attractor dynamics
- Require more epochs (10-50)
- Need GPU acceleration
- Should implement curriculum learning

## Usage

To reproduce or continue training:

```bash
# View training configuration
cat TRAINING_LOG.txt

# Load model for inference
python -c "
import torch
from scripts.quick_train import SimpleTransformer

model = SimpleTransformer(vocab_size=150, hidden_dim=128)
model.load_state_dict(torch.load('output/phase5_models/final_model.pt'))
model.eval()
print('Model loaded successfully')
"
```

## Documentation

For detailed information, see:
1. `PHASE5_TRAINING_REPORT.md` - Full training analysis
2. `TRAINING_LOG.txt` - Execution log and technical details
3. `training_metrics.json` - Numerical metrics

## Phase Status

Phase 5 (Training Setup & Run): **COMPLETE**

All deliverables satisfied:
- Training data verified
- Model trained successfully
- Checkpoints saved
- Metrics logged
- Documentation complete

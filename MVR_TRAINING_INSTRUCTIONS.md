# MVR v1 Training Instructions - Agent B

## Overview
Agent B is responsible for retraining model checkpoints on the regenerated MVR data with the canonical mapping:
- **Desire** = {ν1, ν4, ν7}
- **Wisdom** = {ν5, ν6}
- **Power** = {ν8, ν9}

## Training Data
- **File**: `output/teacher_mvr_converted.jsonl`
- **Samples**: 681 training examples
- **Format**: Story-equation pairs with MVR foundations augmentation

## Training Configuration
Based on the successful long_v4 model configuration:

```
Data: output/teacher_mvr_converted.jsonl
Output directory: output/teacher_model_mvr_v1/
Epochs: 10
Batch size: 4
Learning rate: 0.001
Weight decay: 0.01
Optimizer: AdamW
Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2)
Early stopping: Enabled (patience based on eval loss)
```

## Execution Options

### Option 1: Using the dedicated training script (Recommended)
```bash
.\.venv311\Scripts\python.exe train_mvr_v1.py
```

### Option 2: Using quick_train.py directly
```bash
.\.venv311\Scripts\python.exe scripts\quick_train.py --data output\teacher_mvr_converted.jsonl --output-dir output\teacher_model_mvr_v1 --epochs 10 --batch-size 4 --learning-rate 0.001
```

### Option 3: Using the batch file
```bash
run_mvr_training.bat
```

## Expected Outputs

### 1. Model Checkpoint
- **Location**: `output/teacher_model_mvr_v1/final_model.pt`
- **Format**: PyTorch state dict

### 2. Training Metrics
- **Location**: `output/teacher_model_mvr_v1/training_metrics.json`
- **Contents**:
  - Epoch-by-epoch training losses
  - Epoch-by-epoch evaluation losses
  - Augmentation distribution statistics

### 3. Expected Performance (based on long_v4 baseline)
- **Initial loss**: ~1.19
- **Final training loss**: ~0.88 (target)
- **Final eval loss**: ~0.86 (target)
- **Convergence**: Should show steady decrease over 10 epochs

## Verification Steps

After training completes:

1. **Check checkpoint exists**:
   ```bash
   dir output\teacher_model_mvr_v1\final_model.pt
   ```

2. **Check metrics file**:
   ```bash
   type output\teacher_model_mvr_v1\training_metrics.json
   ```

3. **Verify model size**:
   - Model should have ~130K parameters
   - Checkpoint file should be several MB

4. **Review training curves**:
   - Training loss should decrease smoothly
   - Eval loss should track training loss closely
   - No significant overfitting (eval loss > train loss)

## Next Steps

After successful training:
1. Report final metrics to coordination agent
2. Prepare checkpoint for Agent C (evaluation)
3. Document any deviations from expected performance
4. Compare MVR v1 metrics with long_v4 baseline

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size to 2
- Reduce model dimensions in quick_train.py

### Slow Training
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Expected time: ~10-20 minutes on GPU, ~1-2 hours on CPU

### Data Loading Errors
- Verify file exists: `output\teacher_mvr_converted.jsonl`
- Check file format: Should be JSONL with 'story' and 'aug_type' fields
- Verify line count: Should have 681 lines

## Agent B Responsibilities Checklist

- [x] Verify MVR training data availability
- [x] Configure training hyperparameters (match long_v4)
- [ ] Execute training run
- [ ] Monitor training progress
- [ ] Verify checkpoint saved correctly
- [ ] Report final metrics
- [ ] Compare with baseline (long_v4)
- [ ] Document MVR-specific performance characteristics

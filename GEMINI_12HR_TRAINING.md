# TKS v5 Generalization Training - 12 HOUR RUN

**Date:** 2026-01-08
**Duration:** 12 hours (while user is at work)
**Goal:** Train model to GENERALIZE, not memorize

---

## CRITICAL CHANGE FROM LAST RUN

| Aspect | Previous Run | This Run |
|--------|--------------|----------|
| Samples | 1,000 | **11,320** |
| Epochs | 150 | **75** |
| Final Loss | 0.0003 (memorized) | **1.5-2.5 (generalized)** |
| Total Steps | 37,500 | **~212,000** |

**A HIGHER LOSS IS BETTER** - it means the model learned patterns, not memorized answers.

---

## QUICK START

**Option 1: Double-click batch file**
```
run_generalization_training.bat
```

**Option 2: Manual PowerShell**
```powershell
cd C:\Users\wakil\downloads\everthing-tootra-tks
.\.venv-cuda\Scripts\activate
python train_v5.py --epochs 75
```

---

## WHAT CHANGED IN THE CODE

### 1. Sample Limit Removed (`train_v5.py` line 100)
```python
# OLD: if len(self.data) >= 1000: break
# NEW: Loads ALL samples (no limit)
```

### 2. Larger Dataset (`train_v5.py` line 125)
```python
# OLD: data_path = "output/rebalanced_mix_v5.jsonl"  # 7,918 samples
# NEW: data_path = "output/train_full_5k.jsonl"      # 11,320 samples
```

---

## EXPECTED TIMELINE

| Time | Step | Epoch | Expected Loss |
|------|------|-------|---------------|
| Start | 0 | 0 | 8-10 |
| 1 hr | ~18,000 | 6 | 4-5 |
| 3 hr | ~54,000 | 19 | 2.5-3.5 |
| 6 hr | ~108,000 | 38 | 2.0-2.5 |
| 9 hr | ~162,000 | 57 | 1.8-2.2 |
| 12 hr | ~212,000 | 75 | **1.5-2.0** |

---

## MONITORING

### Check Progress (separate terminal)
```powershell
Get-Content C:\Users\wakil\downloads\everthing-tootra-tks\training_log.txt -Tail 10
```

### Check Checkpoints
```powershell
dir C:\Users\wakil\downloads\everthing-tootra-tks\checkpoints\*.pt
```

### GPU Status
```powershell
nvidia-smi
```

---

## SUCCESS CRITERIA

### GOOD Results (Generalization):
- Final Loss: **1.5 - 2.5**
- NJT Gain: Changed from 2.0 (any direction)
- Entropy: Above 0.5
- Model can handle variations of training examples

### BAD Results (Memorization):
- Final Loss: Below 0.01
- Model only works on exact training phrases
- Cannot generalize to new inputs

---

## CHECKPOINT SCHEDULE

With 2,830 steps/epoch:
| Checkpoint | Step | ~Hour |
|------------|------|-------|
| v5_step_5000.pt | 5,000 | 0.3 |
| v5_step_10000.pt | 10,000 | 0.5 |
| v5_step_50000.pt | 50,000 | 2.7 |
| v5_step_100000.pt | 100,000 | 5.3 |
| v5_step_150000.pt | 150,000 | 8 |
| v5_step_200000.pt | 200,000 | 10.7 |
| v5_final.pt | ~212,000 | 12 |

---

## AFTER TRAINING - VERIFICATION

```python
import torch
from tks_llm_core_v5 import TKSGeneralLM
from configs.v5_recommended import get_v5_config

config = get_v5_config(
    size="base",
    use_njt=True,
    njt_use_hysteresis=True,
    njt_use_rhythm=True,
)

model = TKSGeneralLM(config)
model.load_state_dict(torch.load("checkpoints/v5_final.pt"))

# Check NJT learned
for i, block in enumerate(model.blocks[:3]):
    if hasattr(block, 'njt') and block.njt is not None:
        gain = block.njt.excitatory.gain.mean().item()
        print(f"Block {i} NJT gain: {gain:.4f}")  # Should differ from 2.0
```

---

## FINAL REPORT TEMPLATE

After 12 hours, provide this report:

```markdown
## 12-Hour Generalization Training Report

### Results
- **Total Steps:** ~212,000
- **Total Epochs:** 75
- **Final Loss:** X.XX (target: 1.5-2.5)
- **Training Time:** ~12 hours

### Generalization Check
- Loss > 0.1: Yes/No (Yes = Good)
- NJT Gains Changed: Yes/No
- Entropy Stable: Yes/No

### Checkpoints Saved
- [ ] v5_step_50000.pt
- [ ] v5_step_100000.pt
- [ ] v5_step_150000.pt
- [ ] v5_step_200000.pt
- [ ] v5_final.pt
```

---

## TROUBLESHOOTING

### Training stops early
- Check `training_error.txt` for errors
- Verify GPU has enough memory: `nvidia-smi`

### Loss goes to NaN
- Restart with lower learning rate
- Edit `train_v5.py`: Change `LR = 1e-4` to `LR = 5e-5`

### Out of memory
- Edit `train_v5.py`: Change `BATCH_SIZE = 4` to `BATCH_SIZE = 2`

---

**START THE TRAINING AND LET IT RUN FOR 12 HOURS!**

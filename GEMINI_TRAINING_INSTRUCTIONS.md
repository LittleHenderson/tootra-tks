# TKS LLM v5 Training with NJT - Complete Instructions for Gemini

**Date:** 2026-01-08
**Priority:** HIGH - Execute Training Run
**Prepared by:** Claude (Opus 4.5)

---

## EXECUTIVE SUMMARY

You are tasked with running the TKS LLM v5 training with the newly implemented **NJT (Noetic Judgment Transistor)** circuits enabled. This is the first training run with NJT integration.

**Deliverables:**
1. Successfully start and complete training run with NJT enabled
2. Verify checkpoints are being saved
3. Confirm NJT is actively being used during training
4. Report final training metrics

---

## CRITICAL REQUIREMENTS - READ FIRST

### GPU ONLY - NO CPU TRAINING

**Training MUST use a CUDA GPU. CPU training is FORBIDDEN.**

- If CUDA is not available, DO NOT proceed with training
- DO NOT modify scripts to allow CPU fallback
- DO NOT create new scripts that bypass GPU checks
- The training scripts will EXIT if no GPU is detected

### USE PROVIDED SCRIPTS ONLY

**Use `train_v5.py` - it has been pre-configured with NJT enabled.**

---

## BACKGROUND: What is NJT?

NJT (Noetic Judgment Transistor) is a consciousness amplifier/dampener system inspired by transistor-based neural models. It was fully implemented and tested on 2026-01-07/08.

**Key Components:**
| Component | Function |
|-----------|----------|
| **NJT+** (Excitatory) | Amplifies signals when bias exceeds threshold |
| **NJT-** (Inhibitory) | Dampens signals when bias exceeds threshold |
| **Hysteresis Memory (N5)** | Creates "sticky" reasoning states |
| **Rhythm Oscillator (N7)** | Flow states for sustained attention |

**Mathematical Formulas:**
```
Gate:  g(B) = sigmoid(k * (p_B + alpha2*N2 - alpha3*N3 - threshold))
NJT+:  output = clamp(beta * input * g(bias), 0, 1)
NJT-:  output = clamp(beta * input * (1 - g(bias)), 0, 1)
```

**Files Implemented:**
| File | Lines | Purpose |
|------|-------|---------|
| `tks_features/njt_circuits.py` | 660 | Core NJT module |
| `tks_llm_core_v5.py` | Modified | NJT integration |
| `configs/v5_recommended.py` | Modified | NJT config options |
| `tests/test_njt_circuits.py` | 483 | Test suite (29 tests passing) |

---

## STEP-BY-STEP TRAINING INSTRUCTIONS

### Step 1: Navigate to Project Directory

```powershell
cd C:\Users\wakil\downloads\everthing-tootra-tks
```

### Step 2: Activate CUDA Virtual Environment

**IMPORTANT:** Use the CUDA venv, not system Python (system Python 3.14 has very slow imports).

```powershell
.\.venv-cuda\Scripts\activate
```

You should see `(.venv-cuda)` in your prompt:
```
(.venv-cuda) PS C:\Users\wakil\downloads\everthing-tootra-tks>
```

### Step 3: Verify CUDA is Available

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected Output:**
```
CUDA: True
GPU: NVIDIA GeForce RTX XXXX
```

**If CUDA is False, STOP. Do NOT proceed. Check GPU drivers.**

### Step 4: Verify NJT Module Loads

```powershell
python -c "from tks_features.njt_circuits import NJTLayer, NJTConfig; print('NJT module loaded successfully')"
```

**Expected Output:**
```
NJT module loaded successfully
```

### Step 5: Start Training

```powershell
python train_v5.py --epochs 150
```

**Expected Initial Output:**
```
GPU Verified: NVIDIA GeForce RTX XXXX (X.X GB)
Loading output/rebalanced_mix_v5.jsonl...
Loaded XXX samples
Training on cuda for XXXXX steps (~150.0 epochs)
Initializing model...
Starting training loop...
Step 0: CE=X.XXXX, Aux=X.XXXX, Temp=X.XXX, Entropy=X.XXX
Step 100: CE=X.XXXX, Aux=X.XXXX, Temp=X.XXX, Entropy=X.XXX
...
```

### Step 6: Monitor Training Progress

Training will:
- Log metrics every 100 steps
- Save checkpoints every 5000 steps to `checkpoints/`
- Save final model to `checkpoints/v5_final.pt`

**Checkpoint Schedule:**
| Checkpoint | Step |
|------------|------|
| `v5_step_0.pt` | 0 (initial) |
| `v5_step_5000.pt` | 5000 |
| `v5_step_10000.pt` | 10000 |
| `v5_step_15000.pt` | 15000 |
| `v5_step_20000.pt` | 20000 |
| `v5_step_25000.pt` | 25000 |
| `v5_step_30000.pt` | 30000 |
| `v5_step_35000.pt` | 35000 |
| `v5_final.pt` | Final |

### Step 7: Verify Checkpoints Are Saving

In a separate terminal:
```powershell
dir C:\Users\wakil\downloads\everthing-tootra-tks\checkpoints\*.pt
```

Check timestamps to confirm new checkpoints are being created.

---

## TRAINING CONFIGURATION DETAILS

The `train_v5.py` script is configured with:

```python
config = get_v5_config(
    size="base",
    use_stable_routing=True,
    use_attractor=True,
    use_rpm=True,
    # NJT Configuration
    use_njt=True,
    njt_num_transistors=10,
    njt_use_hysteresis=True,
    njt_use_rhythm=True,
)
```

**NJT Parameters Explained:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_njt` | True | Enable NJT circuits |
| `njt_num_transistors` | 10 | Number of transistors per NJT bank |
| `njt_use_hysteresis` | True | Enable N5 sticky states |
| `njt_use_rhythm` | True | Enable N7 rhythm oscillator (REQUIRED) |

**Other Training Parameters:**

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Learning Rate | 1e-4 |
| Max Sequence Length | 256 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Gradient Clipping | 1.0 |

---

## DATA INFORMATION

**Primary Data File:** `output/rebalanced_mix_v5.jsonl`

If not found, the script will search for any `.jsonl` in `output/`.

**Dataset Format:** JSONL with fields:
- `input` / `output`
- `story` / `equation`
- `prompt` / `target`
- `text`

---

## EXPECTED TRAINING BEHAVIOR

### Normal Metrics Range

| Metric | Initial | Mid-Training | Final |
|--------|---------|--------------|-------|
| CE Loss | 7-10 | 2-4 | 1-2 |
| Aux Loss | 0.01-0.1 | 0.01-0.05 | <0.02 |
| Temperature | 2.0 | 0.5-1.0 | 0.1-0.5 |
| Entropy | 2-3 | 1-2 | 0.5-1.5 |

### With NJT Enabled, You May Observe:

1. **Slightly slower step time** - NJT adds computation per block
2. **More stable loss curves** - Hysteresis prevents oscillation
3. **Lower final entropy** - Better decision-making from differential pairs
4. **Smoother training** - Rhythm oscillator maintains flow

### Training Duration Estimate

| Epochs | Steps (approx) | Time (RTX 3090) |
|--------|----------------|-----------------|
| 50 | ~12,000 | 4-5 hours |
| 100 | ~24,000 | 8-10 hours |
| 150 | ~35,000 | 12-15 hours |

---

## CRITICAL MONITORING

### Routing Health (Check Every 100 Steps)

```python
metrics = output.get("routing_metrics", {})

# Temperature (should decrease)
temp = metrics.get("routing_temperature")

# Entropy (should stay > 0.5)
entropy = metrics.get("routing_entropy_mean")
# ALERT if entropy < 0.5 (routing collapse!)

# Collapse flag
is_collapsing = metrics.get("routing_is_collapsing")
# ALERT if True
```

### Alert Conditions

| Condition | Meaning | Action |
|-----------|---------|--------|
| `entropy < 0.3` | Severe collapse | STOP, increase temp |
| `is_collapsing = True` | Collapse detected | Log warning |
| `aux_loss > 1.0` | Load imbalance | Check dead experts |
| `ce_loss not decreasing` | Learning stalled | Check LR, data |
| `NaN in gradients` | Numerical issue | Reduce LR |

---

## TROUBLESHOOTING

### Problem: Script hangs with no output

**Cause:** Slow PyTorch imports or stuck process

**Solution:**
```powershell
# Kill any stuck Python processes
Stop-Process -Name python -Force

# Re-activate venv and restart
.\.venv-cuda\Scripts\activate
python train_v5.py --epochs 150
```

### Problem: CUDA not available

**Solution:**
```powershell
# Check NVIDIA driver
nvidia-smi

# If driver works but torch doesn't see it, reinstall:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Out of GPU memory

**Solution:** Edit `train_v5.py`:
```python
BATCH_SIZE = 2  # Reduce from 4
MAX_SEQ_LEN = 128  # Reduce from 256
```

### Problem: NJT import error

**Solution:**
```powershell
# Check file exists
dir tks_features\njt_circuits.py

# Check for syntax errors
python -m py_compile tks_features\njt_circuits.py
```

### Problem: Data file not found

**Solution:**
```powershell
# Check for data files
dir output\*.jsonl
dir datasets\*.jsonl
```

### Problem: Process seems stuck (no CPU increase)

**Check:**
```powershell
# Monitor process
Get-Process python | Select-Object Id, CPU, PM, WS

# Run twice with 10 second gap
# If CPU doesn't increase, kill and restart
```

---

## POST-TRAINING VERIFICATION

After training completes, verify NJT is in the model:

```python
import torch
from tks_llm_core_v5 import TKSGeneralLM
from configs.v5_recommended import get_v5_config

# Load config with NJT
config = get_v5_config(
    size="base",
    use_njt=True,
    njt_use_hysteresis=True,
    njt_use_rhythm=True,
)

# Create model
model = TKSGeneralLM(config)

# Load trained weights
model.load_state_dict(torch.load("checkpoints/v5_final.pt"))

# Verify NJT layers exist and have trained
for i, block in enumerate(model.blocks):
    if hasattr(block, 'njt') and block.njt is not None:
        gain = block.njt.excitatory.gain.mean().item()
        print(f"Block {i} NJT gain: {gain:.4f}")
        # If gain differs from 2.0, NJT was trained
```

**Expected Output:**
```
Block 0 NJT gain: X.XXXX
Block 1 NJT gain: X.XXXX
Block 2 NJT gain: X.XXXX
...
```

---

## DELIVERABLES CHECKLIST

Please confirm the following upon completion:

- [ ] Training started successfully with GPU verified message
- [ ] NJT is enabled (check config in output)
- [ ] Checkpoints are being saved to `checkpoints/`
- [ ] Training completed all 150 epochs (~35000 steps)
- [ ] Final checkpoint saved: `checkpoints/v5_final.pt`
- [ ] No errors or crashes during training
- [ ] Report final CE loss value
- [ ] Report final routing entropy value

---

## FINAL REPORT TEMPLATE

After training, provide this report:

```markdown
## TKS v5 + NJT Training Report

### Training Summary
- **Total Steps:** XXXXX
- **Total Epochs:** 150
- **Final CE Loss:** X.XXXX
- **Final Aux Loss:** X.XXXX

### Routing Health
- **Final Temperature:** X.XXX
- **Final Entropy:** X.XXX (should be > 0.5)
- **Collapse Events:** X (should be 0)

### NJT Verification
- **NJT Enabled:** Yes/No
- **Hysteresis Active:** Yes/No
- **Rhythm Active:** Yes/No
- **NJT Gains Changed from 2.0:** Yes/No

### Checkpoints Saved
- [ ] v5_step_0.pt
- [ ] v5_step_5000.pt
- [ ] v5_step_10000.pt
- [ ] v5_step_15000.pt
- [ ] v5_step_20000.pt
- [ ] v5_step_25000.pt
- [ ] v5_step_30000.pt
- [ ] v5_step_35000.pt
- [ ] v5_final.pt

### Issues Encountered
[List any problems or anomalies]
```

---

## COMMANDS QUICK REFERENCE

```powershell
# Navigate to project
cd C:\Users\wakil\downloads\everthing-tootra-tks

# Activate venv
.\.venv-cuda\Scripts\activate

# Start training
python train_v5.py --epochs 150

# Check GPU usage (separate terminal)
nvidia-smi

# Check checkpoints
dir checkpoints\*.pt

# Monitor process
Get-Process python | Select-Object Id, PM, WS, CPU

# Kill stuck process
Stop-Process -Name python -Force
```

---

## REFERENCE DOCUMENTS

For additional context:
- `AGENT_MISSIONS/NJT_IMPLEMENTATION_HANDOFF.md` - Full NJT implementation details
- `AGENT_MISSIONS/TRAIN_WITH_NJT_INSTRUCTIONS.md` - Additional training notes
- `tests/test_njt_circuits.py` - Expected NJT behavior
- `tks_features/njt_circuits.py` - NJT source code

---

## NJT ARCHITECTURE DIAGRAM

```
Input ─┬─→ [NJT+ Excitatory] ─┬─→ Balance ─→ Hysteresis ─→ Output
       │                      │
       └─→ [NJT- Inhibitory] ─┘

NJT+ amplifies signals when bias > threshold
NJT- dampens signals when bias > threshold
Balance parameter (learnable) controls exc/inh mix
Hysteresis creates "sticky" states (harder to switch once committed)
```

---

**END OF INSTRUCTIONS**

Good luck with training! Report back when complete.

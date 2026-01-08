# Training with NJT Enabled - Instructions

**Date:** 2026-01-08
**Status:** Ready for Training

---

## Quick Start

### Option 1: Modify train_v5.py (Recommended)

Edit `train_v5.py` line 164-169 to enable NJT:

```python
# Model
print("Initializing model...")
config = get_v5_config(
    size="base",
    use_stable_routing=True,
    use_attractor=True,
    use_rpm=True,
    # NJT Configuration - ADD THESE LINES
    use_njt=True,
    njt_num_transistors=10,
    njt_use_hysteresis=True,
    njt_use_rhythm=True,
)
```

Then run:
```bash
python train_v5.py --epochs 150
```

### Option 2: Use the NJT Training Script

A dedicated NJT training script has been created:

```bash
python train_v5_njt.py --epochs 150
```

---

## NJT Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_njt` | False | Enable NJT circuits |
| `njt_num_transistors` | 10 | Transistors per NJT bank |
| `njt_use_hysteresis` | True | Enable N5 sticky states |
| `njt_use_rhythm` | True | Enable N7 rhythm oscillator |
| `njt_initial_gain` | 2.0 | Initial amplification factor |
| `njt_initial_threshold` | 0.5 | Gate activation threshold |
| `njt_gate_sharpness` | 10.0 | Sigmoid steepness (k) |
| `njt_hysteresis_gap` | 0.3 | Gap between on/off thresholds |

---

## What NJT Adds to Training

1. **Signal Amplification/Dampening**: NJT+ amplifies relevant signals, NJT- dampens noise
2. **Hysteresis Memory**: Creates "sticky" reasoning states - once a reasoning path is chosen, the model commits to it
3. **Differential Pairs**: Enables cleaner decision-making between competing options
4. **Rhythm Control**: Flow states for sustained attention on complex tasks

---

## Expected Training Behavior

With NJT enabled, you may observe:

1. **Slightly slower step time** - NJT adds computation per block
2. **More stable loss curves** - Hysteresis prevents oscillation
3. **Better reasoning chains** - Commitment to reasoning paths
4. **NJT trace in output** - When `return_full_trace=True`

---

## Monitoring NJT During Training

Add this to your training loop to monitor NJT:

```python
output = model(input_ids, step=global_step, return_full_trace=True)

if 'trace' in output and 'njt' in output['trace']:
    njt_trace = output['trace']['njt'][0]  # First layer
    exc_inh_balance = njt_trace.get('exc_inh_balance', 'N/A')
    print(f"NJT Balance: {exc_inh_balance:.3f}")
```

---

## Checkpoint Compatibility

NJT checkpoints include additional weights:
- `blocks.*.njt.excitatory.*`
- `blocks.*.njt.inhibitory.*`
- `blocks.*.njt.hysteresis.*`

To load a non-NJT checkpoint into an NJT model:
```python
state_dict = torch.load("checkpoints/v5_final.pt")
model.load_state_dict(state_dict, strict=False)  # strict=False allows missing NJT keys
```

---

## Recommended Training Run

```bash
# Full training with NJT
python train_v5.py --epochs 150

# Or use steps
python train_v5.py --steps 35000
```

This will:
1. Load the v5 model with NJT enabled
2. Train for 150 epochs (~35000 steps)
3. Save checkpoints every 5000 steps
4. Save final model to `checkpoints/v5_final.pt`

---

## Verification After Training

```python
# Verify NJT is working in trained model
import torch
from configs.v5_recommended import create_v5_model

model = create_v5_model(size="base", use_njt=True, njt_use_rhythm=True)
model.load_state_dict(torch.load("checkpoints/v5_final.pt"))

# Check NJT layers exist and have trained weights
for i, block in enumerate(model.blocks):
    if hasattr(block, 'njt') and block.njt is not None:
        gain = block.njt.excitatory.gain.mean().item()
        print(f"Block {i} NJT gain: {gain:.4f}")
```

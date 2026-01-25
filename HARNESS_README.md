# TKS Inhibition Harness - Quick Start

## One-Command Execution

```bash
python tks_inhibition_harness.py
```

## What It Tests

**Core Hypothesis:** Models with separate inhibition attention outperform baseline on negation scope resolution.

**Three Conditions:**
- **A) Baseline:** Standard transformer (control)
- **B) Subtractive:** Separate inhibition attention
- **C) 4-Op v2:** Full system with independent gates

## Output Table

```
=== SUMMARY (TEST) ===
Model        Params      Acc    NegAcc  PosAcc   Gap    GateNot  GateOth  Inh(n→p)
Baseline      66000    0.XXX   0.XXX   0.XXX   0.XXX    nan      nan      nan
Subtractive   69000    0.XXX   0.XXX   0.XXX   0.XXX   0.XXX    0.XXX    0.XXX
4-Op v2       73000    0.XXX   0.XXX   0.XXX   0.XXX   0.XXX    0.XXX    0.XXX
```

## How to Read Results

### ✓✓✓ HYPOTHESIS SUPPORTED

```
Subtractive NegAcc > Baseline by ≥5%
Gap shrinks by ≥30%
GateNot > GateOth by ≥20%
Inh(n→p) > 0.25
```

**Meaning:** Separate inhibition learns meaningful negation patterns.

### ⚠️ PARAMETER WIN ONLY

```
Accuracy improved but:
GateNot ≈ GateOth
Inh(n→p) < 0.20
```

**Meaning:** Gains from extra parameters, not mechanism. Match param counts and re-run.

### ✗ NO SUPPORT

```
All metrics similar to baseline
GateNot ≈ GateOth ≈ 0.5
Inh(n→p) < 0.15
```

**Meaning:** Hypothesis not supported on this task. Investigate failure mode.

## Key Metrics Explained

- **Acc:** Overall accuracy
- **NegAcc:** Accuracy on negated cases only
- **PosAcc:** Accuracy on non-negated cases
- **Gap:** PosAcc - NegAcc (smaller = better negation handling)
- **GateNot:** Mean inhibition gate activation on "not" tokens
- **GateOth:** Mean inhibition gate on other tokens
- **Inh(n→p):** Mean attention weight from "not" → predicate (mechanistic probe)

## Patches Applied

✅ **Patch 1:** 4-Op v2 gate probing (returns `w_sub_tensor`)
✅ **Patch 2:** True predicate alignment (only counts "not → actual_predicate")

## Requirements

```bash
pip install torch
```

## Runtime

- CPU: ~30-45 minutes
- GPU: ~5-10 minutes

## Files

- `tks_inhibition_harness.py` - Complete experimental harness (726 lines)
- `TKS_INHIBITION_HARNESS_EVALUATION.md` - Full technical evaluation
- `TKS4OpV2_DESIGN_FEEDBACK.md` - Architecture review (9.6/10)

## Citation

Complete single-file implementation of TKS subtractive hypothesis testing with mechanistic validation probes.

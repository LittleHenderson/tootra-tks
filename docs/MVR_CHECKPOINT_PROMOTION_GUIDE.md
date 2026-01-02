# MVR RPM Checkpoint Promotion Guide

## Overview

This document describes the process for promoting the new MVR-aligned checkpoint (`output/teacher_model_mvr_v1/final_model.pt`) to become the default model after training completes.

## Canonical MVR Mapping

The MVR (Mind-Vibration-Rhythm) realignment uses the following canonical noetic indices:

- **Desire** = {ν1, ν4, ν7} = {Mind, Vibration, Rhythm}
- **Wisdom** = {ν5, ν6} = {Female/Receptivity, Male/Projection}
- **Power** = {ν8, ν9} = {Cause, Effect}

## Pre-Promotion Checklist

Before promoting the MVR v1 checkpoint, verify the following:

### 1. Training Complete
- [ ] `output/teacher_model_mvr_v1/final_model.pt` exists
- [ ] `output/teacher_model_mvr_v1/training_metrics.json` shows convergence
- [ ] No NaN losses or gradient explosions in training log

### 2. D/W/P Canonical Tests Pass
Run the blocking D/W/P canonical test suite:
```bash
pytest tests/test_dwp_canonical.py -v
python scripts/verify_dwp_canonical.py
```

All tests MUST pass, specifically:
- [ ] Desire indices == [1,4,7,11,14,17,21,24,27,31,34,37]
- [ ] Wisdom indices == [5,6,15,16,25,26,35,36]
- [ ] Power indices == [8,9,18,19,28,29,38,39]
- [ ] RPM gate = D × W × P (verified across all 7 foundations)
- [ ] All D/W/P scores bounded in [0, 1]
- [ ] Gradients flow correctly to MVR noetics

### 3. No Benchmark Regression
Compare MVR v1 against current default (long_v4):
```bash
# Evaluate MVR v1 on standard holdout
python scripts/phase6_eval.py \
  --checkpoint output/teacher_model_mvr_v1/final_model.pt \
  --data output/teacher_long_train_holdout.jsonl \
  --output output/mvr_v1_eval_holdout.json

# Compare against long_v4 baseline
python scripts/phase6_eval.py \
  --checkpoint output/teacher_model_long_v4/final_model.pt \
  --data output/teacher_long_train_holdout.jsonl \
  --output output/long_v4_eval_holdout.json
```

Verify:
- [ ] MVR v1 accuracy >= long_v4 accuracy (or within 2% degradation acceptable)
- [ ] MVR v1 validator pass rate = 100%
- [ ] No catastrophic forgetting on reverse tasks

### 4. CI Passes
Ensure all CI tests pass with the new checkpoint:
```bash
# Run local CI simulation
pytest tests -v
python tests/test_regression_gate.py
python tests/fuzz_pipeline.py
python scripts/verify_dwp_canonical.py
pytest tests/test_dwp_canonical.py -v
```

## Promotion Steps

Once all checklist items are complete, promote MVR v1 as the new default:

### Step 1: Update `config/model_defaults.json`

Edit the file to change:

```json
{
  "default_checkpoint": "output/teacher_model_mvr_v1/final_model.pt",
  "model_version": "mvr_v1",
  "trained_on": "teacher_mvr_train.jsonl",
  "promoted_date": "2025-12-15",
  "notes": "MVR-aligned model with canonical D/W/P extraction (D={ν1,ν4,ν7}, W={ν5,ν6}, P={ν8,ν9})",
  "model": {
    "checkpoint": "output/teacher_model_mvr_v1/final_model.pt",
    ...
  },
  "previous_default": {
    "checkpoint": "output/teacher_model_long_v4/final_model.pt",
    "model_version": "long_v4",
    "trained_on": "teacher_long_train_train.jsonl"
  },
  "mvr_realignment": {
    "status": "PROMOTED",
    ...
  }
}
```

### Step 2: Update CI References (if needed)

Check `.github/workflows/ci.yaml` for any hardcoded checkpoint paths that need updating:

```bash
# Search for hardcoded checkpoint paths
grep -n "teacher_model_" .github/workflows/ci.yaml
```

Most CI steps should already be using `config/model_defaults.json`, but verify that:
- Story-equation discrimination check uses the new checkpoint
- Long-chain benchmark uses the new checkpoint
- All informational evals reference the new checkpoint

**Note**: The D/W/P canonical tests do NOT use a checkpoint - they test the code directly, so no changes needed there.

### Step 3: Update Documentation

Update `docs/TRAINING_DEFAULTS.md`:

```markdown
# TKS Model Training Defaults (MVR v1)

**Checkpoint**: `output/teacher_model_mvr_v1/final_model.pt`

## MVR RPM Alignment

This model uses the canonical MVR (Mind-Vibration-Rhythm) mapping for D/W/P extraction:
- **Desire** = {ν1, ν4, ν7} (Mind, Vibration, Rhythm)
- **Wisdom** = {ν5, ν6} (Female, Male)
- **Power** = {ν8, ν9} (Cause, Effect)

## Version History

| Version | Date | Accuracy | Notes |
|---------|------|----------|-------|
| mvr_v1 | 2025-12-15 | [TBD]% | MVR-aligned RPM with canonical D/W/P |
| long_v4 | 2025-12-15 | 71.27% train / 69.67% holdout | Long-chain optimized |
| v3 | 2025-12-14 | 71.27% train / 69.67% holdout | +reverse oversample |
```

### Step 4: Update API Server Fallbacks

Edit `scripts/serve_api.py` to prioritize the new checkpoint:

```python
for path in [
    "output/teacher_model_mvr_v1/final_model.pt",  # NEW: MVR-aligned
    "output/teacher_model_long_v4/final_model.pt",
    "output/teacher_model_v3/final_model.pt"
]:
```

### Step 5: Commit and Tag

Create a git commit for the promotion:

```bash
git add config/model_defaults.json
git add .github/workflows/ci.yaml
git add docs/TRAINING_DEFAULTS.md
git add scripts/serve_api.py

git commit -m "Promote MVR v1 as default checkpoint

- Update config/model_defaults.json to use teacher_model_mvr_v1
- Canonical D/W/P mapping: D={ν1,ν4,ν7}, W={ν5,ν6}, P={ν8,ν9}
- All D/W/P canonical tests pass
- No benchmark regression vs long_v4
- CI fully green with MVR alignment

Closes #[issue_number] (if applicable)"

git tag -a v0.3.0-mvr-v1 -m "MVR RPM v1 - Canonical D/W/P alignment"
```

## Rollback Procedure

If the MVR v1 checkpoint causes issues:

1. Revert `config/model_defaults.json`:
   ```json
   "default_checkpoint": "output/teacher_model_long_v4/final_model.pt"
   ```

2. Check CI passes with the rollback

3. Investigate MVR v1 issues before re-attempting promotion

## Verification After Promotion

After promotion, verify the system is using the new checkpoint:

```bash
# Test API server picks up new checkpoint
python scripts/serve_api.py &
curl http://localhost:8000/health

# Test inference uses new checkpoint
python scripts/infer.py --mode equation --input "A1 + B2 -> C3" --use-model

# Verify CI still passes
git push origin master
# Monitor GitHub Actions
```

## Contact

For questions about the MVR realignment or promotion process, refer to:
- `tests/test_dwp_canonical.py` - Canonical D/W/P test suite
- `scripts/verify_dwp_canonical.py` - Quick verification script
- `tks_llm_core_v2.py` - RPM implementation with D/W/P indices

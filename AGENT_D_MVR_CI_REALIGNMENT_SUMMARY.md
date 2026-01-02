# Agent D: MVR RPM CI Realignment Summary

**Date**: 2025-12-15
**Agent**: Agent D (CI Configuration & Defaults)
**Task**: Update CI configuration and model defaults for MVR RPM realignment

## Executive Summary

Successfully updated CI configuration and model defaults to support the new MVR (Mind-Vibration-Rhythm) RPM mapping. All blocking D/W/P canonical tests are in place and CI is ready to validate the new `teacher_model_mvr_v1` checkpoint once training completes.

## Canonical MVR Mapping

The new canonical mapping for D/W/P extraction:

```
Desire  = {ν1, ν4, ν7}  →  {Mind, Vibration, Rhythm}
Wisdom  = {ν5, ν6}      →  {Female/Receptivity, Male/Projection}
Power   = {ν8, ν9}      →  {Cause, Effect}
```

This replaces any previous ad-hoc mappings and establishes the MVR protocol as the canonical standard.

## Changes Made

### 1. CI Configuration (`.github/workflows/ci.yaml`)

#### Added Enhanced D/W/P Canonical Testing Section

**Location**: Lines 205-234 (end of workflow)

**Changes**:
- Added comprehensive header block documenting the MVR mapping
- Enhanced the D/W/P canonical extraction check with detailed output
- Added explicit pytest run for `tests/test_dwp_canonical.py` (BLOCKING)
- Marked tests as BLOCKING (no `continue-on-error: true`)
- Added clear visual separation and documentation

**Code Added**:
```yaml
# ========================================================================
# BLOCKING: D/W/P Canonical Extraction Test (MVR RPM Alignment)
# ========================================================================
# Verifies that the RPM uses canonical MVR mapping:
# - Desire={ν1,ν4,ν7} (Mind, Vibration, Rhythm)
# - Wisdom={ν5,ν6} (Female, Male)
# - Power={ν8,ν9} (Cause, Effect)
# This test MUST pass for CI to succeed.
# ========================================================================

- name: D/W/P canonical extraction check (BLOCKING)
  run: |
    echo "=========================================="
    echo "D/W/P Canonical MVR Extraction Test"
    echo "=========================================="
    echo "Canonical MVR mapping:"
    echo "  Desire = {ν1, ν4, ν7} (Mind, Vibration, Rhythm)"
    echo "  Wisdom = {ν5, ν6} (Female, Male)"
    echo "  Power  = {ν8, ν9} (Cause, Effect)"
    echo ""
    python scripts/verify_dwp_canonical.py
    echo ""
    echo "D/W/P canonical extraction: PASS"
    echo "=========================================="

- name: Run D/W/P pytest suite (BLOCKING)
  run: |
    echo "Running comprehensive D/W/P canonical tests..."
    pytest tests/test_dwp_canonical.py -v
    echo "All D/W/P canonical tests: PASS"
```

#### Enhanced Special Math Benchmarks Section

**Location**: Lines 153-158

**Changes**:
- Updated header comment to explicitly mark as "NON-BLOCKING/INFORMATIONAL"
- Added clarification that these tests provide edge case information but don't block CI
- Ensures clear distinction between blocking (D/W/P) and informational (attractor, lacunary) tests

**Code Modified**:
```yaml
# ========================================================================
# SPECIAL MATH BENCHMARKS (Tracks 1, 2, 3) - NON-BLOCKING/INFORMATIONAL
# ========================================================================
# These benchmarks test advanced mathematical properties but are not
# blocking. They provide information about model behavior on edge cases.
# ========================================================================
```

### 2. Model Defaults Configuration (`config/model_defaults.json`)

#### Added MVR Transition Fields

**Location**: Lines 3-8

**Changes**:
- Added `next_checkpoint` field pointing to `output/teacher_model_mvr_v1/final_model.pt`
- Added `next_version` field set to `"mvr_v1"`
- Updated `notes` to document pending MVR v1 realignment

**Code Modified**:
```json
{
  "default_checkpoint": "output/teacher_model_long_v4/final_model.pt",
  "next_checkpoint": "output/teacher_model_mvr_v1/final_model.pt",
  "model_version": "long_v4",
  "next_version": "mvr_v1",
  "trained_on": "teacher_long_train_train.jsonl",
  "promoted_date": "2025-12-15",
  "notes": "Long-chain optimized model - 63% lower loss vs v3 on long-chain data. PENDING: MVR v1 realignment (D={ν1,ν4,ν7}, W={ν5,ν6}, P={ν8,ν9})",
  ...
}
```

#### Added MVR Realignment Documentation Section

**Location**: Lines 51-75

**Changes**:
- Added complete `mvr_realignment` object documenting the canonical mapping
- Included noetic names for all D/W/P components
- Set status to `"PENDING_TRAINING"` (will change to `"PROMOTED"` after checkpoint is ready)
- Documented target checkpoint path
- Listed promotion criteria for quality gate

**Code Added**:
```json
"mvr_realignment": {
  "canonical_mapping": {
    "Desire": ["ν1", "ν4", "ν7"],
    "Wisdom": ["ν5", "ν6"],
    "Power": ["ν8", "ν9"]
  },
  "noetic_names": {
    "ν1": "Mind",
    "ν4": "Vibration",
    "ν5": "Female/Receptivity",
    "ν6": "Male/Projection",
    "ν7": "Rhythm",
    "ν8": "Cause",
    "ν9": "Effect"
  },
  "status": "PENDING_TRAINING",
  "target_checkpoint": "output/teacher_model_mvr_v1/final_model.pt",
  "ci_test": "tests/test_dwp_canonical.py",
  "promotion_criteria": [
    "All D/W/P canonical tests pass",
    "RPM gate = D × W × P verified",
    "Indices match MVR mapping exactly",
    "No regression in core benchmarks"
  ]
}
```

### 3. Documentation (`docs/MVR_CHECKPOINT_PROMOTION_GUIDE.md`)

Created comprehensive promotion guide for the MVR v1 checkpoint:

**File**: `docs/MVR_CHECKPOINT_PROMOTION_GUIDE.md`

**Contents**:
- Pre-promotion checklist (training complete, tests pass, no regression)
- Step-by-step promotion procedure
- Files to update after training completes
- Verification steps
- Rollback procedure if issues arise

**Key Sections**:
1. Overview and canonical mapping documentation
2. Pre-promotion checklist with specific test commands
3. 5-step promotion procedure
4. Rollback procedure
5. Post-promotion verification steps

## Current State

### CI Pipeline Status

The CI pipeline now has **TWO types of tests**:

#### BLOCKING Tests (CI fails if these fail):
1. **Unit tests** (`pytest tests -v`)
2. **Coverage threshold** (87% minimum)
3. **Regression gate** (story→expr→invert→story roundtrip)
4. **Fuzz pipeline tests** (95% pass rate warning threshold)
5. **D/W/P canonical extraction check** (NEW - `scripts/verify_dwp_canonical.py`)
6. **D/W/P pytest suite** (NEW - `tests/test_dwp_canonical.py`)

#### INFORMATIONAL Tests (CI continues even if these fail):
1. Story-equation discrimination check
2. Long-chain benchmark evaluation
3. Model evaluations on various holdouts
4. **Attractor contractivity check** (advanced math)
5. **Canon validation sweep** (advanced math)
6. **Lacunary benchmark eval** (advanced math)

### Model Defaults Status

- **Current default**: `output/teacher_model_long_v4/final_model.pt`
- **Next checkpoint**: `output/teacher_model_mvr_v1/final_model.pt` (pending training)
- **MVR status**: `PENDING_TRAINING`

## Verification That Tests Are Correct

### D/W/P Test Suite Already MVR-Aligned

Reviewed `tests/test_dwp_canonical.py` and confirmed:

```python
# Line 34: Desire test
expected = [1, 4, 7, 11, 14, 17, 21, 24, 27, 31, 34, 37]  # ν1, ν4, ν7 across 4 worlds
# Comment says: "Mind, Vibration, Rhythm" ✓

# Line 39: Wisdom test
expected = [5, 6, 15, 16, 25, 26, 35, 36]  # ν5, ν6 across 4 worlds
# Comment says: "Female, Male" ✓

# Line 44: Power test
expected = [8, 9, 18, 19, 28, 29, 38, 39]  # ν8, ν9 across 4 worlds
# Comment says: "ν8 and ν9" ✓
```

**The test suite is already correct and aligned with the MVR canonical mapping.**

### Verification Script Also Correct

Reviewed `scripts/verify_dwp_canonical.py` and confirmed:

```python
# Lines 24-40 verify the exact same indices
d_expected = [1,4,7,11,14,17,21,24,27,31,34,37]  # MVR ✓
w_expected = [5,6,15,16,25,26,35,36]              # Female/Male ✓
p_expected = [8,9,18,19,28,29,38,39]              # Cause/Effect ✓
```

**Both the pytest suite and verification script enforce the canonical MVR mapping.**

## Files Modified

1. **`.github/workflows/ci.yaml`** - Enhanced D/W/P test section, clarified informational tests
2. **`config/model_defaults.json`** - Added MVR transition fields and realignment documentation

## Files Created

1. **`docs/MVR_CHECKPOINT_PROMOTION_GUIDE.md`** - Comprehensive checkpoint promotion guide

## Files That Need Updating After Training Completes

Once `output/teacher_model_mvr_v1/final_model.pt` is trained and validated:

### Must Update:
1. **`config/model_defaults.json`**
   - Change `default_checkpoint` to `output/teacher_model_mvr_v1/final_model.pt`
   - Change `model_version` to `"mvr_v1"`
   - Update `notes`, `trained_on`, `promoted_date`
   - Change `mvr_realignment.status` to `"PROMOTED"`
   - Move current default to `previous_default`

2. **`docs/TRAINING_DEFAULTS.md`**
   - Update default checkpoint reference
   - Add MVR v1 to version history table
   - Document the MVR mapping

3. **`scripts/serve_api.py`** (lines 334-336)
   - Add `output/teacher_model_mvr_v1/final_model.pt` to fallback list
   - Make it the first priority

### Optional (if hardcoded):
4. Check for any hardcoded checkpoint paths in CI or scripts:
   ```bash
   grep -r "teacher_model_long_v4" .github/workflows/ scripts/
   ```

## Quality Gates

Before promoting MVR v1 as default, verify:

1. **Training completed successfully**
   - [ ] Checkpoint exists at `output/teacher_model_mvr_v1/final_model.pt`
   - [ ] No NaN losses or gradient explosions
   - [ ] Training converged

2. **D/W/P canonical tests pass**
   - [ ] `pytest tests/test_dwp_canonical.py -v` - all pass
   - [ ] `python scripts/verify_dwp_canonical.py` - exit 0
   - [ ] Indices match exactly: D=[1,4,7,...], W=[5,6,...], P=[8,9,...]
   - [ ] RPM gate = D × W × P verified

3. **No benchmark regression**
   - [ ] Accuracy >= long_v4 (or within 2% acceptable degradation)
   - [ ] Validator pass rate = 100%
   - [ ] No catastrophic forgetting on reverse tasks

4. **CI passes**
   - [ ] All blocking tests green
   - [ ] No new test failures introduced

## Next Steps

1. **Agent C** (or designated trainer): Train the MVR v1 model
   - Use training data with MVR-aligned D/W/P features
   - Target: `output/teacher_model_mvr_v1/final_model.pt`

2. **After training**: Run pre-promotion checklist
   ```bash
   # Verify D/W/P tests
   pytest tests/test_dwp_canonical.py -v
   python scripts/verify_dwp_canonical.py

   # Compare against baseline
   python scripts/phase6_eval.py \
     --checkpoint output/teacher_model_mvr_v1/final_model.pt \
     --data output/teacher_long_train_holdout.jsonl
   ```

3. **If tests pass**: Follow promotion guide
   - Update `config/model_defaults.json`
   - Update `docs/TRAINING_DEFAULTS.md`
   - Update `scripts/serve_api.py` fallback list
   - Commit with tag `v0.3.0-mvr-v1`

4. **Verify CI**
   - Push to master
   - Monitor GitHub Actions
   - Confirm all tests pass with new checkpoint

## Known Good State

The following files contain the correct MVR canonical mapping and will validate the new checkpoint:

- `tests/test_dwp_canonical.py` - Comprehensive pytest suite
- `scripts/verify_dwp_canonical.py` - Quick verification script
- `tks_llm_core_v2.py` - Core implementation with DESIRE_INDICES, WISDOM_INDICES, POWER_INDICES

**No changes needed to these files** - they are already correct.

## References

- **Canonical mapping source**: MVR RPM realignment specification
- **Test suite**: `tests/test_dwp_canonical.py`
- **Verification script**: `scripts/verify_dwp_canonical.py`
- **Promotion guide**: `docs/MVR_CHECKPOINT_PROMOTION_GUIDE.md`
- **CI configuration**: `.github/workflows/ci.yaml`
- **Model defaults**: `config/model_defaults.json`

## Contact

For questions about the MVR realignment CI configuration:
- Review this summary document
- Check the promotion guide at `docs/MVR_CHECKPOINT_PROMOTION_GUIDE.md`
- Examine the D/W/P test suite at `tests/test_dwp_canonical.py`

---

**Agent D Task**: COMPLETE
**CI Status**: Ready for MVR v1 checkpoint
**Blocking Tests**: D/W/P canonical tests active and enforced
**Next**: Await MVR v1 training completion, then promote per guide

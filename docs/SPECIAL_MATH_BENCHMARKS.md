# TKS Special Math Benchmarks

## Overview

This document describes three specialized benchmark tracks for testing the TKS model's handling of fractal/attractor dynamics, lacunary/sparse patterns, and canon validation.

## Canon Guardrails (Strict)

All benchmarks enforce strict TKS canon:
- **Worlds**: A, B, C, D only
- **Noetics**: 1-10 with involutions 2↔3, 5↔6, 8↔9; self-duals 1, 4, 7, 10
- **Foundations**: 1-7 with subfoundations _d1.._d7
- **Operators**: +, -, +T, -T, ->, <-, *T, /T, o (9 total)
- **Senses**: ^1 through ^9

---

## Track 1: Fractal/Attractor Tightening

### Purpose
Test the model's handling of attractor convergence in 40D noetic space (10 noetics × 4 worlds).

### Files
- `scripts/attractor_checks.py` - Spectral/contractivity analysis
- `scripts/generate_attractor_stress.py` - Generate stress test dataset
- `data/attractor_stress.jsonl` - 20 stress test equations

### Running Locally

```bash
# Generate attractor stress dataset
python scripts/generate_attractor_stress.py

# Run contractivity analysis
python scripts/attractor_checks.py --output contractivity_report.json -v
```

### Attractor Stress Test Categories

| Category | Description | Example |
|----------|-------------|---------|
| `involution_pair_2_3` | Polarity swap | `B2 -> C3` |
| `involution_pair_5_6` | Gender swap | `A5 -> D6` |
| `involution_pair_8_9` | Cause-effect flow | `B8 -> D9` |
| `self_dual_N1` | Mind stability | `B1 +T C1 -> D1` |
| `self_dual_N4` | Vibration stability | `A4 o B4 *T C4` |
| `self_dual_N7` | Rhythm stability | `D7 o C7 o B7` |
| `self_dual_N10` | Idea stability | `A10 -> B10 -> C10` |
| `foundation_grounded` | Foundation targeting | `A10_d1 -> B2 +T C5` |
| `long_chain_mixed` | Multiple involutions | `A2 -> B3 +T C5 <- D6` |
| `polarity_oscillation` | Limit cycle test | `B2 -> C3 -> B2 -> C3` |
| `maximum_complexity` | Full notation test | Full equation with ^, _d |

### Contractivity Analysis

The attractor check computes:
- **Spectral radius** ρ(J): Maximum eigenvalue magnitude of Jacobian
- **Lipschitz constant** L: Induced matrix norm
- **Contraction**: Both ρ(J) < 1 and L < 1

Output JSON:
```json
{
  "contractivity_check": "pass",
  "spectral_radius": 0.95,
  "lipschitz_constant": 0.98,
  "chain_analysis": [
    {"chain_length": 2, "is_contraction": true},
    {"chain_length": 3, "is_contraction": true}
  ]
}
```

---

## Track 2: Lacunary/Sparse Benchmark

### Purpose
Test the model's ability to handle sparse operator patterns and long-distance dependencies.

### Files
- `scripts/generate_lacunary_benchmark.py` - Generate benchmark
- `data/lacunary_benchmark.jsonl` - 75 sparse pattern equations

### Running Locally

```bash
# Generate lacunary benchmark
python scripts/generate_lacunary_benchmark.py --count 75

# Evaluate on long_v4
python scripts/phase6_eval.py \
  --checkpoint output/teacher_model_long_v4/final_model.pt \
  --data data/lacunary_benchmark.jsonl \
  --test-ratio 1.0 \
  --output output/lacunary_eval.json
```

### Lacunary Pattern Types

| Pattern | Description | Example |
|---------|-------------|---------|
| `alternating_operators` | Regular operator alternation | `A1 + B2 - C3 + D4` |
| `long_distance_sparse` | Boundary ops differ from middle | `A1 -> B2 o C3 o D4 -> A5` |
| `banded_operators` | First half vs second half ops | `A1 + B2 + C3 - D4 - A5` |
| `self_dual_skip` | Uses self-duals (1,4,7,10) | `A1 + B4 - C7 + D10` |
| `involution_sparse` | Involution pairs with gaps | `A2 + B5 - C3 + D6` |
| `world_skip` | Alternating worlds only | `A1 + C2 + A3 + C4` |

### Chain Lengths
- 4 elements (short sparse)
- 5 elements (medium sparse)
- 6 elements (long sparse)

---

## Track 3: Canon Validation Gatekeeping

### Purpose
Ensure 100% canon compliance across all new datasets.

### Files
- `scripts/validator_sweep.py` - Canon validation sweep
- `teacher/validator.py` - CanonicalValidator

### Running Locally

```bash
# Validate attractor stress set
python scripts/validator_sweep.py \
  --input data/attractor_stress.jsonl \
  --output output/attractor_canon_report.json

# Validate lacunary benchmark
python scripts/validator_sweep.py \
  --input data/lacunary_benchmark.jsonl \
  --output output/lacunary_canon_report.json
```

### Validation Report Format

```json
{
  "input_file": "data/attractor_stress.jsonl",
  "total_count": 20,
  "valid_count": 20,
  "invalid_count": 0,
  "pass_rate": 1.0,
  "canon_score_stats": {
    "min": 1.0,
    "max": 1.0,
    "mean": 1.0
  },
  "failures": []
}
```

---

## CI Integration

All checks are integrated into `.github/workflows/ci.yaml` as non-blocking informational steps:

```yaml
- name: Attractor contractivity check (informational)
  continue-on-error: true
  run: |
    if [ -f "scripts/attractor_checks.py" ]; then
      python scripts/attractor_checks.py --output output/contractivity_report.json
    fi

- name: Canon validation sweep (informational)
  continue-on-error: true
  run: |
    python scripts/validator_sweep.py --input data/attractor_stress.jsonl
    python scripts/validator_sweep.py --input data/lacunary_benchmark.jsonl

- name: Lacunary benchmark eval (informational)
  continue-on-error: true
  run: |
    python scripts/phase6_eval.py \
      --checkpoint output/teacher_model_long_v4/final_model.pt \
      --data data/lacunary_benchmark.jsonl
```

---

## Model Default

**long_v4** remains the default model checkpoint. These benchmarks are for evaluation only and do not change training.

## Related Documents

- `docs/TRAINING_INTEGRATION_PLAN.md` - Training pipeline
- `README_ANTI_ATTRACTOR.md` - Anti-attractor synthesis
- `OPERATOR_HEURISTICS_GUIDE.md` - Operator semantics

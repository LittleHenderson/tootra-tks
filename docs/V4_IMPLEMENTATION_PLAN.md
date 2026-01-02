# TKS-LLM v4 Implementation Plan

## Coordination Document for Model Improvements

**Document:** V4_IMPLEMENTATION_PLAN.md
**Version:** 1.0
**Date:** 2025-12-22
**Author:** tks-supervisor (Integration Agent)
**Status:** ACTIVE - Ready for Agent Assignment

---

## Executive Summary

This document coordinates the v4 model improvements to address four validated architectural issues in `TKSLLMCorePipeline`:

1. **World Separation Failure**: Input "A1+A2+A3" activates World B higher than World A
2. **RPM Gate Collapse**: All D/W/P categories cluster at ~0.94 (no differentiation)
3. **Attractor Non-Convergence**: 0% convergence rate in testing
4. **No A->B->C->D Hierarchy**: Per-world projections are isolated, no cascade semantics

The implementation is organized into **3 parallel workstreams** that can proceed independently with defined integration points.

---

## Section 1: TKS Canon Constraints

All agents MUST adhere to these canonical constraints. Violations will fail CI.

### 1.1 Core Dimensional Structure

```
40-Dimensional Noetic Space:
  World A (Spiritual): indices 0-9   -> Elements A0-A9
  World B (Mental):    indices 10-19 -> Elements B0-B9
  World C (Emotional): indices 20-29 -> Elements C0-C9
  World D (Physical):  indices 30-39 -> Elements D0-D9

Formula: index(Xn) = world_offset(X) + n
  where world_offset = {A:0, B:10, C:20, D:30}
```

### 1.2 MVR Protocol (Canonical D/W/P Mapping)

**CRITICAL**: The RPM gating MUST use these exact noetic indices:

| Category | Noetics | Names | Index Formula (per world offset w) |
|----------|---------|-------|-----------------------------------|
| **Desire (D)** | nu_1, nu_4, nu_7 | Mind, Vibration, Rhythm | w+1, w+4, w+7 |
| **Wisdom (W)** | nu_5, nu_6 | Female, Male | w+5, w+6 |
| **Power (P)** | nu_8, nu_9 | Cause/Above, Effect/Below | w+8, w+9 |

Across 4 worlds this yields:
- **DESIRE_INDICES**: [1,4,7, 11,14,17, 21,24,27, 31,34,37] (12 dims)
- **WISDOM_INDICES**: [5,6, 15,16, 25,26, 35,36] (8 dims)
- **POWER_INDICES**: [8,9, 18,19, 28,29, 38,39] (8 dims)

Reference: `tks_llm_core_v2.py` lines 71-93 (validated in current codebase)

### 1.3 World Cascade Semantics (A->B->C->D)

The canonical flow is:
```
A (Spiritual/Abstract) -> B (Mental/Conceptual) -> C (Emotional/Evaluative) -> D (Physical/Concrete)
```

- Higher worlds MUST influence lower worlds (forward flow)
- Backward influence (D->A) should be weaker than forward
- This is validated by the `CascadeLoss` in `training/losses.py`

### 1.4 Noetic Involution Pairs

From `tks_rules/noetics.py`:
```python
INVOLUTION_PAIRS = [
    (2, 7),  # Positive <-> Rhythm
    (3, 6),  # Negative <-> Male
    (4, 5),  # Vibration <-> Female
]
SPECIAL_PAIR = (8, 9)  # Above/Cause <-> Below/Effect
SELF_DUAL_NOETICS = frozenset({0, 1})
```

Constraint: Composed operators should satisfy nu_i o nu_j approx I for involution pairs.

### 1.5 Attractor Mathematical Requirements

From `TKS_LLM_Noetic_Mathematics_v1.0.md` Section 7:
1. Contraction mapping: Lipschitz constant L < 1
2. Variance reduction over iterations
3. Convergence below tolerance epsilon
4. All maps must be differentiable

---

## Section 2: Workstream Definitions

### Workstream 1: World/RPM Separation (WORLD-RPM)

**Objective**: Fix world separation failure and RPM gate collapse

**Issues Addressed**:
- World Separation Failure: A-world inputs should NOT activate B-world higher
- RPM Collapse: D/W/P scores must differentiate (spread across [0,1], not cluster at 0.94)

**Files to Modify**:
```
PRIMARY:
  tks_llm_core.py          - NoeticEmbeddingLayer world transforms
  tks_llm_core_v2.py       - RPMGatingMechanism D/W/P evaluators
  tks_llm_core_v4.py       - NoeticTokenEmbedding, NoeticRouter

AUXILIARY:
  training/losses.py       - Add L_world_separation, L_rpm_entropy losses
  tks_rules/noetics.py     - Reference only (DO NOT MODIFY)
```

**Implementation Strategy**:

1. **World Separation Loss** (NEW)
   ```
   L_world = sum over worlds W of:
     CrossEntropy(predicted_world_activation, ground_truth_world_label)

   Where ground_truth_world_label = argmax of input element's world indices
   ```

2. **RPM Entropy Regularization** (NEW)
   ```
   L_rpm_entropy = -lambda * mean(H(D) + H(W) + H(P))

   Where H(x) = -sum(p * log(p)) is entropy
   Goal: Maximize entropy to spread scores across [0,1]
   ```

3. **RPM Pretraining Phase**
   - Stage 1: Train D/W/P evaluators with supervised labels before full pipeline
   - Use synthetic data with known D/W/P values
   - Freeze RPM evaluators during initial LM training

4. **World Cross-Talk Penalty**
   ```
   L_crosstalk = ||W_AB||_F + ||W_AC||_F + ||W_AD||_F + ...

   Where W_XY measures correlation between world X and Y activations
   when input is purely in world X
   ```

**Success Criteria**:
- Input "A1+A2+A3" activates World A > World B (by at least 0.1 margin)
- D/W/P score standard deviation > 0.15 (not clustered)
- D/W/P scores span at least 50% of [0,1] range

**Dependencies**: None (can start immediately)

---

### Workstream 2: Attractor Stabilization (ATTRACTOR)

**Objective**: Fix 0% convergence rate via spectral norm / Lipschitz enforcement

**Issues Addressed**:
- Attractor Non-Convergence: Iterations never reach tolerance
- Contraction property not maintained after training

**Files to Modify**:
```
PRIMARY:
  tks_llm_core_v2.py       - AttractorComputationLayer
  tks_llm_core_v4.py       - Re-uses AttractorComputationLayer from v2

AUXILIARY:
  training/losses.py       - SpectralLoss (already exists, may need tuning)
```

**Implementation Strategy**:

1. **Spectral Normalization** (RECOMMENDED)
   ```python
   # Wrap each contraction map linear layer with spectral_norm
   from torch.nn.utils import spectral_norm

   self.contraction_maps = nn.ModuleList([
       spectral_norm(nn.Linear(dim, dim)) for _ in range(num_maps)
   ])
   ```

2. **Lipschitz Constraint via Power Iteration**
   - Already exists in `SpectralLoss._estimate_spectral_radius()`
   - Increase `power_iterations` from 10 to 20 for accuracy
   - Lower `spectral_radius_target` from 0.9 to 0.7

3. **Contraction Factor Enforcement**
   ```python
   # After each forward pass, project weights back to contraction
   def _enforce_contraction(self):
       for cmap in self.contraction_maps:
           with torch.no_grad():
               W = cmap.weight
               U, S, V = torch.linalg.svd(W)
               S_clamped = torch.clamp(S, max=self.contraction_factor)
               cmap.weight.copy_(U @ torch.diag(S_clamped) @ V)
   ```

4. **Adaptive Iteration Count**
   - Start with max_iterations=5, increase to 20 during late training
   - Log actual iterations used for monitoring

5. **Residual Connection Scaling**
   - Current: `self.residual_weight = nn.Parameter(torch.tensor(0.1))`
   - Constrain to [0, 0.3] to prevent bypassing contraction

**Success Criteria**:
- Convergence rate > 50% (ideally > 80%)
- All contraction maps have Lipschitz constant < 0.9
- Variance reduction ratio < 0.8 (final_var / initial_var)

**Dependencies**: None (can start immediately)

---

### Workstream 3: CI Regression Gates (CI-GATES)

**Objective**: Add evaluation harness as CI regression gates for v4 issues

**Issues Addressed**:
- No automated detection of world separation failure
- No automated detection of RPM collapse
- No automated detection of attractor non-convergence

**Files to Modify**:
```
PRIMARY:
  tests/test_v4_regression.py      - NEW FILE
  .github/workflows/ci.yaml        - Add v4 regression gates

AUXILIARY:
  scripts/verify_v4_metrics.py     - NEW FILE (standalone checker)
```

**Implementation Strategy**:

1. **World Separation Test** (BLOCKING in CI)
   ```python
   def test_world_separation():
       """Input A-world elements must activate A-world highest."""
       model = TKSLLMCorePipeline(vocab_size=100)

       # Create A-world only input
       tokens = encode_elements(["A1", "A2", "A3"])
       out = model(tokens, return_full_trace=True)

       world_activations = extract_world_activations(out)
       assert world_activations['A'] > world_activations['B'], \
           f"World A ({world_activations['A']:.3f}) must exceed B ({world_activations['B']:.3f})"
   ```

2. **RPM Differentiation Test** (BLOCKING in CI)
   ```python
   def test_rpm_differentiation():
       """D/W/P scores must have meaningful variance."""
       model = TKSLLMCorePipeline(vocab_size=100)

       # Run on diverse inputs
       dwp_scores = collect_dwp_across_inputs(model, n_samples=50)

       d_std = dwp_scores[:, :, :, 0].std()
       w_std = dwp_scores[:, :, :, 1].std()
       p_std = dwp_scores[:, :, :, 2].std()

       assert d_std > 0.10, f"Desire std {d_std:.3f} too low (collapsed)"
       assert w_std > 0.10, f"Wisdom std {w_std:.3f} too low (collapsed)"
       assert p_std > 0.10, f"Power std {p_std:.3f} too low (collapsed)"
   ```

3. **Attractor Convergence Test** (BLOCKING in CI)
   ```python
   def test_attractor_convergence():
       """Attractor must converge on at least 50% of inputs."""
       model = TKSLLMCorePipeline(vocab_size=100)

       converged_count = 0
       for _ in range(100):
           tokens = random_tokens(batch=1, seq=8)
           out = model(tokens, return_full_trace=True)
           if out['attractor_converged']:
               converged_count += 1

       rate = converged_count / 100
       assert rate >= 0.50, f"Convergence rate {rate:.1%} below 50% threshold"
   ```

4. **Cascade Flow Test** (INFORMATIONAL initially, BLOCKING after v4.1)
   ```python
   def test_cascade_flow():
       """Forward flow (A->B->C->D) must exceed backward flow."""
       model = TKSLLMCorePipeline(vocab_size=100)

       # Measure cross-world correlations
       forward_flow = measure_cascade_correlation(model, 'forward')
       backward_flow = measure_cascade_correlation(model, 'backward')

       assert forward_flow > backward_flow, \
           f"Forward flow ({forward_flow:.3f}) must exceed backward ({backward_flow:.3f})"
   ```

5. **CI YAML Additions**:
   ```yaml
   - name: V4 World Separation Check (BLOCKING)
     run: pytest tests/test_v4_regression.py::test_world_separation -v

   - name: V4 RPM Differentiation Check (BLOCKING)
     run: pytest tests/test_v4_regression.py::test_rpm_differentiation -v

   - name: V4 Attractor Convergence Check (BLOCKING)
     run: pytest tests/test_v4_regression.py::test_attractor_convergence -v

   - name: V4 Cascade Flow Check (INFORMATIONAL)
     continue-on-error: true
     run: pytest tests/test_v4_regression.py::test_cascade_flow -v
   ```

**Success Criteria**:
- All BLOCKING tests pass on untrained model (sanity) and trained model
- CI catches regressions automatically
- Clear error messages identifying which constraint failed

**Dependencies**:
- Should be implemented FIRST so other workstreams can verify against gates
- Can start immediately

---

## Section 3: Integration Points

### 3.1 Shared Files Requiring Coordination

| File | WORLD-RPM | ATTRACTOR | CI-GATES | Resolution |
|------|-----------|-----------|----------|------------|
| `tks_llm_core_v2.py` | Modifies RPMGatingMechanism | Modifies AttractorComputationLayer | Reads only | Sequential edits, separate sections |
| `training/losses.py` | Adds L_world, L_rpm_entropy | Tunes SpectralLoss | Reads only | Add new loss classes, don't modify existing |
| `.github/workflows/ci.yaml` | None | None | Adds steps | Append to end of file |

### 3.2 Dependency Graph

```
                    CI-GATES (Start First)
                         |
                         v
         +---------------+---------------+
         |                               |
         v                               v
    WORLD-RPM                       ATTRACTOR
         |                               |
         +---------------+---------------+
                         |
                         v
                   INTEGRATION
                   (tks-supervisor)
```

### 3.3 Integration Checkpoints

**Checkpoint 1: CI Framework Ready** (Day 1)
- CI-GATES creates `tests/test_v4_regression.py` with placeholder assertions
- All tests fail initially (expected behavior for v3 model)
- CI runs but marks tests as "expected fail"

**Checkpoint 2: Individual Fixes Complete** (Day 3-5)
- WORLD-RPM submits PR with world separation + RPM entropy losses
- ATTRACTOR submits PR with spectral norm enforcement
- Both PRs must pass existing CI (no regressions)

**Checkpoint 3: Integration Testing** (Day 6-7)
- tks-supervisor merges both PRs sequentially
- Runs full v4 regression suite
- Verifies no conflicts between fixes

**Checkpoint 4: CI Gates Activated** (Day 8)
- Remove "expected fail" markers from CI tests
- All v4 tests now BLOCKING
- Tag release as v4.0

---

## Section 4: File Templates

### 4.1 New Loss Classes (training/losses.py additions)

```python
# Add after existing loss classes

class WorldSeparationLoss(nn.Module):
    """
    L_world: Penalizes cross-world activation when input is world-specific.

    Canon Constraint: Input "A1+A2+A3" must activate World A > World B.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        world_activations: Dict[str, torch.Tensor],
        input_world_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            world_activations: {'A': [batch, seq, 10], 'B': ..., 'C': ..., 'D': ...}
            input_world_labels: [batch, seq] with values 0=A, 1=B, 2=C, 3=D
        """
        # Implementation: Cross-entropy on world predictions
        pass


class RPMEntropyLoss(nn.Module):
    """
    L_rpm_entropy: Encourages spread of D/W/P scores across [0,1].

    Canon Constraint: D/W/P must differentiate, not collapse to single value.
    """

    def __init__(self, lambda_entropy: float = 0.1):
        super().__init__()
        self.lambda_entropy = lambda_entropy

    def forward(self, dwp_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dwp_scores: [batch, seq, 7, 3] from RPMGatingMechanism
        """
        # Implementation: Maximize entropy of score distribution
        pass
```

### 4.2 V4 Regression Test Template (tests/test_v4_regression.py)

```python
"""
TKS-LLM V4 Regression Tests

Tests the four v4 improvements:
1. World Separation (A-input -> A-activation)
2. RPM Differentiation (D/W/P variance > 0.10)
3. Attractor Convergence (> 50% convergence rate)
4. Cascade Flow (forward > backward)

All tests are BLOCKING in CI unless marked otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM


class TestV4WorldSeparation:
    """World separation tests - Canon: A-input activates A-world highest."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return TKSLLMCorePipeline(vocab_size=100)

    def test_a_world_input_activates_a_world(self, model):
        """A-world input must activate A-world > B-world."""
        # TODO: Implement after WORLD-RPM workstream
        pass

    def test_d_world_input_activates_d_world(self, model):
        """D-world input must activate D-world > A-world."""
        # TODO: Implement after WORLD-RPM workstream
        pass


class TestV4RPMDifferentiation:
    """RPM differentiation tests - Canon: D/W/P must spread, not collapse."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return TKSLLMCorePipeline(vocab_size=100)

    def test_desire_variance(self, model):
        """Desire scores must have std > 0.10."""
        # TODO: Implement after WORLD-RPM workstream
        pass

    def test_wisdom_variance(self, model):
        """Wisdom scores must have std > 0.10."""
        pass

    def test_power_variance(self, model):
        """Power scores must have std > 0.10."""
        pass


class TestV4AttractorConvergence:
    """Attractor convergence tests - Canon: Contraction must converge."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return TKSLLMCorePipeline(vocab_size=100)

    def test_convergence_rate_above_threshold(self, model):
        """Convergence rate must exceed 50%."""
        # TODO: Implement after ATTRACTOR workstream
        pass

    def test_lipschitz_constant_below_one(self, model):
        """All contraction maps must have Lipschitz < 1."""
        checks = model.attractor.verify_contraction(num_samples=100)
        for key, val in checks.items():
            if key.endswith("is_contraction"):
                assert val, f"{key} failed: Lipschitz >= 1"


class TestV4CascadeFlow:
    """Cascade flow tests - Canon: A->B->C->D forward flow."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return TKSLLMCorePipeline(vocab_size=100)

    @pytest.mark.xfail(reason="Informational until v4.1")
    def test_forward_exceeds_backward(self, model):
        """Forward cascade flow must exceed backward."""
        # TODO: Implement after integration
        pass
```

---

## Section 5: Agent Assignments

### 5.1 Recommended Agent Roster

| Workstream | Primary Agent | Backup Agent | Estimated Effort |
|------------|---------------|--------------|------------------|
| WORLD-RPM | tks-ml | tks-math | 2-3 days |
| ATTRACTOR | tks-math | tks-ml | 1-2 days |
| CI-GATES | tks-eval | tks-integration | 1 day |
| Integration | tks-supervisor | - | 1 day |

### 5.2 Handoff Protocol

Each agent completing a workstream MUST leave a handoff note:

```markdown
## HANDOFF: [Workstream] Complete

**Agent**: [name]
**Date**: [date]

### Completed:
- [List of completed items with file:line references]

### Tests Added:
- [List of new test functions]

### CI Impact:
- [New CI steps or modified thresholds]

### Integration Notes:
- [Any special considerations for merging]

### Open Issues:
- [Any unresolved problems or future work]
```

---

## Section 6: Timeline and Milestones

### Phase 1: Setup (Day 1)
- [ ] CI-GATES: Create test_v4_regression.py with placeholder tests
- [ ] CI-GATES: Add informational CI steps (continue-on-error: true)
- [ ] All: Review this plan and confirm assignments

### Phase 2: Implementation (Days 2-5)
- [ ] WORLD-RPM: Implement WorldSeparationLoss
- [ ] WORLD-RPM: Implement RPMEntropyLoss
- [ ] WORLD-RPM: Add pretraining phase for RPM evaluators
- [ ] ATTRACTOR: Add spectral_norm to contraction maps
- [ ] ATTRACTOR: Implement weight projection for Lipschitz enforcement
- [ ] ATTRACTOR: Tune SpectralLoss parameters

### Phase 3: Testing (Days 6-7)
- [ ] CI-GATES: Fill in placeholder test implementations
- [ ] All: Run full test suite locally
- [ ] All: Fix any failing tests

### Phase 4: Integration (Day 8)
- [ ] tks-supervisor: Merge WORLD-RPM PR
- [ ] tks-supervisor: Merge ATTRACTOR PR
- [ ] tks-supervisor: Merge CI-GATES PR
- [ ] tks-supervisor: Activate blocking CI gates
- [ ] tks-supervisor: Tag v4.0 release

---

## Section 7: Risk Mitigation

### Risk 1: Losses Conflict
**Problem**: New losses may conflict with existing task loss
**Mitigation**: Use curriculum scheduler (already in losses.py) to phase in new losses

### Risk 2: Spectral Norm Hurts Expressivity
**Problem**: Enforcing Lipschitz < 1 may reduce model capacity
**Mitigation**: Start with loose constraint (0.95), tighten gradually; monitor task loss

### Risk 3: CI Flakiness
**Problem**: Random initialization may cause test flakiness
**Mitigation**: Use fixed seeds in all tests; run 100+ samples for statistical tests

### Risk 4: Breaking Existing Functionality
**Problem**: v4 changes may break encoder/decoder pipeline
**Mitigation**: Existing regression gates remain active; run full test suite before merge

---

## Appendix A: Reference Links

- Architecture Spec: `TKS_LLM_Architecture_v1.0.md`
- Noetic Mathematics: `TKS_LLM_Noetic_Mathematics_v1.0.md`
- Canonical Validation: `TKS_LLM_Canonical_Validation_v1.0.md`
- Current v2 Core: `tks_llm_core_v2.py`
- Current v4 Core: `tks_llm_core_v4.py`
- Loss Functions: `training/losses.py`
- Noetics Canon: `tks_rules/noetics.py`
- CI Configuration: `.github/workflows/ci.yaml`

---

## Appendix B: Quick Reference - Canon Values

```python
# World offsets (40-dim space)
WORLD_OFFSETS = {'A': 0, 'B': 10, 'C': 20, 'D': 30}

# MVR D/W/P indices (canonical)
DESIRE_INDICES = [1,4,7, 11,14,17, 21,24,27, 31,34,37]  # nu_1, nu_4, nu_7 x 4 worlds
WISDOM_INDICES = [5,6, 15,16, 25,26, 35,36]             # nu_5, nu_6 x 4 worlds
POWER_INDICES = [8,9, 18,19, 28,29, 38,39]              # nu_8, nu_9 x 4 worlds

# Involution pairs
INVOLUTION_PAIRS = [(2, 7), (3, 6), (4, 5)]
SPECIAL_PAIR = (8, 9)
SELF_DUAL = {0, 1}

# Spectral constraints
TARGET_LIPSCHITZ = 0.9  # Contraction maps
TARGET_CONVERGENCE_RATE = 0.50  # Minimum acceptable
TARGET_VARIANCE_REDUCTION = 0.8  # final_var / initial_var

# RPM differentiation thresholds
MIN_DWP_STD = 0.10  # Minimum standard deviation
```

---

*End of V4 Implementation Plan*

**Status**: Ready for agent assignment
**Next Action**: CI-GATES workstream should begin immediately

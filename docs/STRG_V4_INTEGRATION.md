# STRG-LLM-STK Integration with TKS_LLM_CORE_V4

**Author:** ARCH agent
**Date:** 2026-01-05
**Version:** 1.0

This document describes the integration architecture for incorporating strg-llm-stk specifications (DPS, Governance Rails) into tks_llm_core_v4.

---

## 1. Executive Summary

### What's Being Added

| Module | Purpose | Type | Training Data |
|--------|---------|------|---------------|
| **DPSGatingLayer** | Depth Permission System - earned computational depth | Neural (nn.Module) | ~3000+ examples |
| **GovernanceRails** | Non-bypassable safety gates | Rule-based (Python) | None |
| **Verifier Integration** | Evidence binding + consistency checking | Hybrid | TBD |

### Key Architectural Insight

**DPS is LEARNED, Governance is RULE-BASED.**

This is critical for implementation:
- DPS uses backpropagation, trainable weights, curriculum learning
- Governance uses deterministic thresholds, pure math, no gradients

---

## 2. Current v4 Pipeline (Before Integration)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TKSNoeticLM Forward Pass                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   tokens                                                             │
│      │                                                               │
│      ▼                                                               │
│  ┌──────────────────────┐                                           │
│  │ NoeticTokenEmbedding │  tokens → 40D noetic space                │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │ NoeticPositionEmbed  │  + positional embeddings                  │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │   NoeticBlocks (×N)  │  router + causal fractal attention        │
│  │   ├─ NoeticRouter    │                                           │
│  │   └─ FractalAttn     │                                           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │  OperatorCore (opt)  │  equation-aware composition               │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │ StableAttractorLayer │  fixed-point iteration (Banach)           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │  RPMGatingMechanism  │  D×W×P filtering by Foundation            │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │   Output Projection  │  40D → vocab_size logits                  │
│  └──────────────────────┘                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Integrated v4 Pipeline (After Integration)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   TKSNoeticLM v4.2 Forward Pass                      │
│                  (with STRG-LLM-STK Integration)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   tokens                                                             │
│      │                                                               │
│      ▼                                                               │
│  ┌──────────────────────┐                                           │
│  │ NoeticTokenEmbedding │                                           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │ NoeticPositionEmbed  │                                           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ╔══════════════════════╗                                           │
│  ║   NoeticBlocks (×N)  ║  ← DPS can control iteration count        │
│  ║   ├─ NoeticRouter    ║    (Option A: Adaptive Computation)       │
│  ║   └─ FractalAttn     ║                                           │
│  ╚══════════╤═══════════╝                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │  OperatorCore (opt)  │                                           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │ StableAttractorLayer │                                           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ╔══════════════════════╗                                           │
│  ║     Verifier         ║  ← NEW: Evidence binding + consistency    │
│  ║  (optional module)   ║                                           │
│  ╚══════════╤═══════════╝                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │  RPMGatingMechanism  │                                           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ╔══════════════════════╗                                           │
│  ║   DPSGatingLayer     ║  ← NEW: Learned depth permission          │
│  ║   NW = V×I×(1-S)     ║                                           │
│  ╚══════════╤═══════════╝                                           │
│             │                                                        │
│             ▼                                                        │
│  ╔══════════════════════╗                                           │
│  ║  GovernanceRails     ║  ← NEW: Rule-based safety gates           │
│  ║  HS = (U×K)×A²       ║                                           │
│  ╚══════════╤═══════════╝                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │   Output Projection  │                                           │
│  └──────────────────────┘                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Legend:
  ┌──┐  Existing v4 component
  ╔══╗  NEW component from strg-llm-stk
```

---

## 4. Data Flow Diagram

```
                              ┌─────────────────┐
                              │   User Input    │
                              │  (goal, context)│
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        GOVERNANCE LAYER (Pre-Action)                  │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 1. HighStakesGate: HS = (U×K)×A²                                │ │
│  │    → Determines mode: Normal / High-Stakes / Critical           │ │
│  │                                                                  │ │
│  │ 2. IrreversibilityGate: Check action.reversibility              │ │
│  │    → Requires clearance for irreversible actions                │ │
│  │                                                                  │ │
│  │ 3. PermissionGate: Check allowlists                             │ │
│  │    → Tool, network, filesystem, code execution                  │ │
│  │                                                                  │ │
│  │ 4. AntiRecursionGate: Check caps                                │ │
│  │    → RPM depth/breadth, tool calls, uncertainty-drop            │ │
│  │                                                                  │ │
│  │ 5. SelfModificationGate: Check policy changes                   │ │
│  │    → Blocked in Critical, replay required in High-Stakes        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Decision: ALLOW / PAUSE / BLOCK                                      │
│  If BLOCK → Exit with required_actions                               │
│  If PAUSE → Request verification/confirmation                        │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ ALLOW
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        NEURAL PIPELINE (v4)                           │
│                                                                       │
│  tokens → NoeticEmbedding → NoeticBlocks → OperatorCore              │
│                    │                                                  │
│                    │ (if use_adaptive_iteration)                     │
│                    │                                                  │
│               ┌────▼─────┐                                           │
│               │   DPS    │ ← Controls iteration count                │
│               │ Novelty  │   based on earned depth                   │
│               │ Compute  │                                           │
│               └────┬─────┘                                           │
│                    │                                                  │
│                    ▼                                                  │
│           StableAttractorLayer                                        │
│                    │                                                  │
│                    ▼                                                  │
│              ┌───────────┐                                           │
│              │ Verifier  │ ← Evidence binding                        │
│              │ (Module 2)│   Consistency check                       │
│              └─────┬─────┘                                           │
│                    │                                                  │
│                    ▼                                                  │
│              RPMGating                                                │
│                    │                                                  │
│                    ▼                                                  │
│         ┌─────────────────┐                                          │
│         │ DPSGatingLayer  │ ← Compute NW = V×I×(1-S)                 │
│         │                 │   Classify: HEAVY/COUNT/NOCOUNT          │
│         │                 │   Update state (tokens, p_max)           │
│         └────────┬────────┘                                          │
│                  │                                                    │
│                  ▼                                                    │
│         Output Projection → logits                                    │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   GOVERNANCE LAYER (Post-Verification)                │
│                                                                       │
│  Check action results against rails                                   │
│  Log governance event                                                 │
│  Update audit trail                                                   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                              ┌─────────────┐
                              │   Output    │
                              │ (logits +   │
                              │   trace)    │
                              └─────────────┘
```

---

## 5. DPS Integration Details

### 5.1 Option A: Adaptive Iteration (Recommended)

DPS controls how many NoeticBlock iterations run:

```python
class TKSNoeticLM(nn.Module):
    def forward(self, tokens, dps_state, ...):
        x = self.embedding(tokens) + self.position(...)

        if self.config.use_dps_gating and self.config.use_adaptive_iteration:
            # DPS controls iteration count
            controller = AdaptiveIterationController(self.dps_gating, dps_state)

            for iteration in range(controller.max_iterations):
                for block in self.blocks:
                    x, trace, ... = block(x, ...)

                # Check if we should continue
                novelty = self.dps_gating.compute_novelty(x, memory_embeddings)
                if novelty.value >= HEAVY_THRESHOLD:
                    # Earned right to think deeper
                    controller.extend_budget()
                elif controller.should_stop(x, iteration):
                    break
        else:
            # Standard fixed iteration
            for block in self.blocks:
                x, trace, ... = block(x, ...)

        # Continue with attractor, verifier, RPM, DPS gating, governance
        ...
```

### 5.2 Option B: Output Gating (Simpler)

DPS only gates/transforms the final output:

```python
# After RPMGating
x = self.rpm_gating(x, goal_state, target_foundation)

# DPS gates output based on novelty
if self.dps_gating is not None:
    x, dps_state, dps_trace = self.dps_gating(x, dps_state, memory_embeddings)
```

### 5.3 Novelty Weight Computation

```
        ┌─────────────────────────────────────────────┐
        │          NoveltyWeight Computation           │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  Validity Network                            │
        │  ┌─────────────────────────────────────────┐│
        │  │ x [batch, seq, 40]                      ││
        │  │       ↓                                 ││
        │  │ Linear(40 → 32) + ReLU                  ││
        │  │       ↓                                 ││
        │  │ Linear(32 → 3) + Sigmoid                ││
        │  │       ↓                                 ││
        │  │ [confidence, consistency, evidence_ok]  ││
        │  │       ↓                                 ││
        │  │ V = conf × cons × evid                  ││
        │  └─────────────────────────────────────────┘│
        │                                              │
        │  Impact Network                              │
        │  ┌─────────────────────────────────────────┐│
        │  │ x [batch, seq, 40] + reward_signals     ││
        │  │       ↓                                 ││
        │  │ Linear(40+context → 32) + ReLU          ││
        │  │       ↓                                 ││
        │  │ Linear(32 → 4) + Sigmoid                ││
        │  │       ↓                                 ││
        │  │ [Δreward, Δuncertainty, Δcoverage, Δcost]│
        │  │       ↓                                 ││
        │  │ I = weighted sum, clipped to [0,1]      ││
        │  └─────────────────────────────────────────┘│
        │                                              │
        │  Similarity Network                          │
        │  ┌─────────────────────────────────────────┐│
        │  │ x [batch, seq, 40]                      ││
        │  │ memory_embeddings [num_mem, 40]         ││
        │  │       ↓                                 ││
        │  │ cosine similarity → S_embed             ││
        │  │ AST comparison → S_ast                  ││
        │  │       ↓                                 ││
        │  │ S = max(S_embed, S_ast)                 ││
        │  └─────────────────────────────────────────┘│
        │                                              │
        │  NW = V × I × (1 - S)                        │
        │                                              │
        │  Classification:                             │
        │    NW >= 0.70 → HEAVY (unlock eligible)      │
        │    NW >= 0.45 → COUNT (token accumulation)   │
        │    NW <  0.45 → NOCOUNT (no progress)        │
        │                                              │
        └─────────────────────────────────────────────┘
```

---

## 6. Governance Integration Details

### 6.1 Integration Points

Governance runs at two points:
1. **Pre-Action**: Before any tool call or action
2. **Post-Verification**: After verifier checks

```python
class GovernanceIntegration:
    def check_pre_action(self, action: ActionProfile, context: Dict) -> GateResult:
        """Called before Router/Executor"""
        return self.rails.check_action(
            action=action,
            uncertainty=context['uncertainty'],
            stakes=context['stakes'],
            alignment=context['alignment'],
            ...
        )

    def check_post_verification(self, action: ActionProfile, verifier_result: Dict) -> GateResult:
        """Called after Verifier, before Memory write"""
        return self.rails.check_action(
            action=action,
            has_evidence=verifier_result['has_evidence'],
            ...
        )
```

### 6.2 High-Stakes Flow

```
                    ┌──────────────┐
                    │  Compute HS  │
                    │ HS=(U×K)×A²  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        HS < 0.45    0.45 ≤ HS < 0.70   HS ≥ 0.70
              │            │            │
              ▼            ▼            ▼
         ┌────────┐   ┌────────┐   ┌────────┐
         │ Normal │   │  High  │   │Critical│
         │  Mode  │   │ Stakes │   │  Mode  │
         └────┬───┘   └────┬───┘   └────┬───┘
              │            │            │
              │            │            ▼
              │            │      ┌───────────┐
              │            │      │   BLOCK   │
              │            │      │ -EXECUTE  │
              │            │      └───────────┘
              │            │
              │            ▼
              │      ┌────────────────┐
              │      │ ASK:SLOW:VERIFY│
              │      │ +GATE(PAUSE)   │
              │      └────────┬───────┘
              │               │
              │               ▼
              │         Need clearance?
              │               │
              │       ┌───────┴───────┐
              │       │ Yes           │ No
              │       ▼               ▼
              │  ┌─────────┐    ┌─────────┐
              │  │ Request │    │ PAUSE   │
              │  │Clearance│    │ (verify)│
              │  └────┬────┘    └────┬────┘
              │       │              │
              │       ▼              │
              │  Clearance obtained? │
              │       │              │
              │   ┌───┴───┐         │
              │   │Yes    │No       │
              │   ▼       ▼         │
              │ ALLOW   BLOCK       │
              │   │                 │
              └───┼─────────────────┘
                  │
                  ▼
             ┌─────────┐
             │ PROCEED │
             │  with   │
             │ action  │
             └─────────┘
```

---

## 7. Trace Schema Extensions

### 7.1 Full Integrated Trace Schema

```python
TRACE_SCHEMA_V1_1 = {
    "_schema_version": "1.1",

    # Existing v4 trace components
    "noetic_routing": {"weights": Tensor, "indices": Tensor},
    "attention": {"scale_weights": Tensor},
    "attractor": {
        "converged": bool,
        "iterations": int,
        "final_delta": float,
        "trajectory": Tensor,
    },
    "rpm": {
        "gate": Tensor,
        "dwp_scores": Tensor,
        "foundation_idx": Tensor,
    },
    "operator_core": {
        "gate_values": Tensor,
        "noetic_output": Tensor,
        "equation_repr": Tensor,
        "symmetry_losses": dict,
    },

    # NEW: DPS trace
    "dps": {
        "novelty_weight": float,
        "novelty_class": str,
        "validity": {
            "confidence": float,
            "consistency": float,
            "evidence_ok": float,
        },
        "impact": {
            "delta_reward": float,
            "delta_uncertainty": float,
            "delta_coverage": float,
            "delta_cost": float,
        },
        "similarity": {
            "s_embed": float,
            "s_ast": float,
        },
        "depth_allowed": int,
        "unlock_occurred": bool,
        "tokens": int,
        "iteration_budget": int,
    },

    # NEW: Governance trace
    "governance": {
        "mode": str,
        "hs_score": float,
        "gate_results": [
            {
                "gate_name": str,
                "decision": str,
                "reason": str,
            }
        ],
        "final_decision": str,
        "required_actions": [str],
        "missing_evidence": [str],
        "clearance_required": bool,
    },

    # NEW: Verifier trace
    "verifier": {
        "status": str,
        "confidence": float,
        "consistency": float,
        "evidence_ids": [str],
        "gate": str,
    },

    # Legacy compatibility
    "_legacy": {...},
}
```

---

## 8. Configuration Changes

### 8.1 TKSNoeticLMConfig Additions

```python
@dataclass
class TKSNoeticLMConfig:
    # ... existing config ...

    # NEW: DPS Gating config
    use_dps_gating: bool = False              # Default OFF (backward compat)
    dps_count_threshold: float = 0.45
    dps_heavy_threshold: float = 0.70
    dps_tokens_for_unlock: int = 5
    dps_cooldown_k: int = 10
    dps_max_depth: int = 5
    dps_initial_p_max: int = 2
    dps_use_adaptive_iteration: bool = True

    # NEW: Governance Rails config
    use_governance_rails: bool = False        # Default OFF (backward compat)
    governance_hs_threshold: float = 0.45
    governance_critical_threshold: float = 0.70
    governance_alignment_min: float = 0.80
    governance_rpm_depth_cap_normal: int = 3
    governance_rpm_depth_cap_high_stakes: int = 2
    governance_tool_call_cap_normal: int = 12
    governance_tool_call_cap_high_stakes: int = 8
```

### 8.2 Backward Compatibility

When `use_dps_gating=False` and `use_governance_rails=False`:
- v4 works exactly as before
- No new modules instantiated
- No trace additions
- Same performance characteristics

---

## 9. Training Considerations

### 9.1 DPS Training

DPS is LEARNED from ~3000+ training examples:

```
Training Data Structure:
{
    "input": Tensor[batch, seq, 40],
    "novelty_type": ["A", "B", "C"],  # New equation, RPM shortcut, cross-domain
    "validity_target": {"confidence": float, "consistency": float, "evidence_ok": float},
    "impact_target": {"delta_reward": float, ...},
    "similarity_target": {"s_embed": float, "s_ast": float},
    "nw_target": float,
    "class_target": str,  # HEAVY | COUNT | NOCOUNT
}

Loss Function:
    L_total = L_classification + λ₁*L_validity + λ₂*L_impact + λ₃*L_similarity

Curriculum:
    Phase 1: Classification only (HEAVY/COUNT/NOCOUNT)
    Phase 2: + Validity component
    Phase 3: + Impact component
    Phase 4: + Similarity component
    Phase 5: Full NW training
```

### 9.2 Governance Training

Governance is NOT trained. It is pure configuration.

To tune governance:
1. Adjust threshold configs
2. Run replay sets
3. Check for regressions
4. Deploy if improvements

---

## 10. Implementation Roadmap

### Phase 1: Interface Definition (COMPLETE)
- [x] DPSGatingLayer interface (`dps_gating.py`)
- [x] GovernanceRails interface (`governance_rails.py`)
- [x] Module contracts (`CONTRACTS.md`)
- [x] Integration diagram (this document)

### Phase 2: Implementation (agent-be)
- [ ] Implement DPS neural networks
- [ ] Implement state management
- [ ] Integrate governance with existing pipeline
- [ ] Add trace logging

### Phase 3: Training (agent-ml)
- [ ] Create DPS training dataset
- [ ] Implement DPS curriculum
- [ ] Train DPS networks
- [ ] Validate accuracy

### Phase 4: Evaluation (agent-eval)
- [ ] Create DPS test suite
- [ ] Create governance test suite
- [ ] Run integration tests
- [ ] Verify backward compatibility

---

## 11. Files Changed/Added

### New Files
- `tks_features/dps_gating.py` - DPS interface
- `tks_features/governance_rails.py` - Governance interface
- `tks_features/CONTRACTS.md` - Module contracts
- `docs/STRG_V4_INTEGRATION.md` - This document

### Modified Files (by agent-be)
- `tks_llm_core_v4.py` - Add config flags, instantiate modules, wire integration
- `tks_llm_core_v2.py` - Update trace schema version

---

## 12. Quick Reference

### DPS Formula
```
NW = V × I × (1 - S)

V = confidence × consistency × evidence_ok
I = weighted(Δreward, Δuncertainty, Δcoverage, Δcost)
S = max(S_embed, S_ast)

HEAVY:   NW >= 0.70 → Immediate unlock
COUNT:   NW >= 0.45 → Token accumulation
NOCOUNT: NW <  0.45 → No progress
```

### Governance Formula
```
HS = (U × K) × A²

U = uncertainty [0,1]
K = stakes [0,1]
A = alignment [0,1]

Critical:    HS >= 0.70 → BLOCK
High-Stakes: HS >= 0.45 → PAUSE
Normal:      HS <  0.45 → ALLOW
```

### The 5 Non-Bypassable Gates
1. **HighStakesGate** - Mode determination
2. **IrreversibilityGate** - Clearance requirement
3. **PermissionGate** - Allowlist enforcement
4. **AntiRecursionGate** - Cap enforcement
5. **SelfModificationGate** - Policy protection

---

End of Integration Document.

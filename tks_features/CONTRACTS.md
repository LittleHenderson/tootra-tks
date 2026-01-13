# TKS Features Module Contracts

**Author:** ARCH agent
**Date:** 2026-01-05
**Version:** 1.0

This document defines the contracts for new modules integrating strg-llm-stk specifications into tks_llm_core_v4.

---

## 1. Module Overview

| Module | Type | Has Training Data | File |
|--------|------|-------------------|------|
| **DPSGatingLayer** | Neural (nn.Module) | YES (~3000+ examples) | `dps_gating.py` |
| **GovernanceRails** | Rule-based (pure Python) | NO | `governance_rails.py` |

### Critical Distinction

- **DPS**: LEARNED module with trainable parameters. Uses backpropagation.
- **Governance**: DETERMINISTIC rules with NO learnable parameters. Pure logic.

---

## 2. DPSGatingLayer Contract

### 2.1 Input Shapes

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `x` | `[batch, seq, 40]` | Tensor | Noetic space representation |
| `state` | N/A | DPSState | Current depth permission state |
| `memory_embeddings` | `[num_memories, 40]` | Tensor (optional) | Stored packet embeddings for similarity |
| `verifier_signals` | Dict | Dict (optional) | Verifier outputs for validity computation |
| `reward_signals` | Dict | Dict (optional) | Reward/cost signals for impact computation |

### 2.2 Output Shapes

| Output | Shape | Type | Description |
|--------|-------|------|-------------|
| `output` | `[batch, seq, 40]` | Tensor | Gated/transformed representation |
| `new_state` | N/A | DPSState | Updated state (immutable) |
| `trace` | Dict | Dict | Trace information (see 2.4) |

### 2.3 Configuration Parameters (DPSConfig)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_tokens_threshold` | int | 5 | COUNT tokens needed for unlock |
| `count_threshold` | float | 0.45 | NW threshold for COUNT |
| `heavy_threshold` | float | 0.70 | NW threshold for HEAVY |
| `cooldown_k` | int | 10 | Episodes after unlock |
| `unlock_high_stakes` | bool | False | Allow unlocks in High-Stakes |
| `unlock_critical` | bool | False | Allow unlocks in Critical |
| `max_depth` | int | 5 | Hard cap on p_max |
| `initial_p_max` | int | 2 | Starting p_max |
| `hidden_dim` | int | 64 | Hidden dimension for networks |
| `validity_dim` | int | 32 | Validity feature dimension |
| `impact_dim` | int | 32 | Impact feature dimension |
| `similarity_method` | str | "hybrid" | "embedding", "ast", "hybrid" |
| `use_adaptive_iteration` | bool | True | Control iteration count |
| `dropout` | float | 0.1 | Dropout rate |

### 2.4 Trace Schema Addition

```python
{
    "dps": {
        "novelty_weight": float,      # NW = V x I x (1-S)
        "novelty_class": str,         # "HEAVY" | "COUNT" | "NOCOUNT"
        "validity": {
            "confidence": float,      # Verifier confidence [0,1]
            "consistency": float,     # Canon consistency [0,1]
            "evidence_ok": float,     # Evidence binding quality [0,1]
        },
        "impact": {
            "delta_reward": float,    # Reward improvement
            "delta_uncertainty": float,# Uncertainty reduction
            "delta_coverage": float,  # Coverage increase
            "delta_cost": float,      # Cost savings
        },
        "similarity": {
            "s_embed": float,         # Embedding similarity [0,1]
            "s_ast": float,           # AST similarity [0,1]
        },
        "depth_allowed": int,         # Current p_max
        "unlock_occurred": bool,      # Unlock this pass?
        "tokens": int,                # Accumulated tokens
        "iteration_budget": int,      # Allowed iterations
    }
}
```

### 2.5 Integration Point

Insert **after RPMGating** and **before final output projection** in v4 pipeline:

```python
# In TKSNoeticLM.forward():
if self.dps_gating is not None:
    x, dps_state, dps_trace = self.dps_gating(x, dps_state, memory_embeddings)
    if return_full_trace:
        trace["dps"] = dps_trace
```

### 2.6 Backward Compatibility

```python
class TKSNoeticLMConfig:
    use_dps_gating: bool = False  # Default OFF
```

When `use_dps_gating=False`, v4 works exactly as before.

---

## 3. GovernanceRails Contract

### 3.1 Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | ActionProfile | Action to evaluate |
| `uncertainty` | float [0,1] | Uncertainty score |
| `stakes` | float [0,1] | Stakes score |
| `alignment` | float [0,1] | Alignment score (F1 dominance) |
| `has_evidence` | bool | Required evidence bound? |
| `clearance_token` | ClearanceToken (optional) | Permission token |
| `current_depth` | int | RPM tree depth |
| `current_breadth` | int | Children per node |
| `tool_calls_this_episode` | int | Tool calls count |
| `uncertainty_dropped` | bool | Uncertainty improving? |
| `expansions_since_drop` | int | Expansions without improvement |
| `replay_passed` | bool | Replay harness passed? |
| `has_rollback_plan` | bool | Rollback plan exists? |

### 3.2 Output

| Output | Type | Description |
|--------|------|-------------|
| `result` | GateResult | Decision and metadata |

```python
@dataclass
class GateResult:
    gate_name: str              # Which gate triggered
    decision: GateDecision      # ALLOW | PAUSE | BLOCK
    reason: str                 # Human-readable explanation
    mode: OperationalMode       # Normal | High-Stakes | Critical
    required_actions: List[str] # What to do next
    missing_evidence: List[str] # Evidence IDs needed
    clearance_required: bool    # Need clearance token?
    hs_score: Optional[float]   # High-Stakes score
```

### 3.3 Configuration Parameters (GovernanceConfig)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hs_threshold` | float | 0.45 | High-Stakes threshold |
| `critical_threshold` | float | 0.70 | Critical threshold |
| `alignment_min` | float | 0.80 | F1 dominance minimum |
| `rpm_depth_cap_normal` | int | 3 | Depth cap (Normal) |
| `rpm_depth_cap_high_stakes` | int | 2 | Depth cap (High-Stakes) |
| `rpm_breadth_cap` | int | 5 | Breadth cap per node |
| `tool_call_cap_normal` | int | 12 | Tool cap (Normal) |
| `tool_call_cap_high_stakes` | int | 8 | Tool cap (High-Stakes) |
| `tool_call_cap_critical` | int | 0 | Tool cap (Critical) |
| `require_evidence_high_stakes` | bool | True | Evidence required |
| `require_clearance_irreversible` | bool | True | Clearance for irreversible |
| `allow_self_mod_normal` | bool | True | Self-mod in Normal |
| `allow_self_mod_high_stakes` | bool | True | Self-mod in High-Stakes |
| `allow_self_mod_critical` | bool | False | Self-mod in Critical |
| `max_expansion_without_uncertainty_drop` | int | 2 | Expansions before PAUSE |

### 3.4 Trace Schema Addition

```python
{
    "governance": {
        "mode": str,              # "Normal" | "High-Stakes" | "Critical"
        "hs_score": float,        # High-Stakes score
        "gate_results": [
            {
                "gate_name": str,
                "decision": str,  # "ALLOW" | "PAUSE" | "BLOCK"
                "reason": str,
            }
        ],
        "final_decision": str,
        "required_actions": [str],
        "missing_evidence": [str],
        "clearance_required": bool,
    }
}
```

### 3.5 Integration Point

Governance runs **BEFORE** Router/Executor and **AFTER** Verifier:

```
User/External Inputs
    ↓
┌─────────────────────┐
│ GovernanceRails     │  ← CHECK HERE (before action)
└─────────────────────┘
    ↓
RPM Planner → Router → Executor
    ↓
┌─────────────────────┐
│ Verifier            │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ GovernanceRails     │  ← CHECK HERE (after verification)
└─────────────────────┘
    ↓
Memory + Scoring + Logs
```

### 3.6 Backward Compatibility

```python
class TKSNoeticLMConfig:
    use_governance_rails: bool = False  # Default OFF
```

When `use_governance_rails=False`, v4 works exactly as before.

---

## 4. The 5 Non-Bypassable Gates

### Gate 1: HighStakesGate

**Formula:** `HS = (U × K) × A²`

| Input | Type | Description |
|-------|------|-------------|
| U | float [0,1] | Uncertainty |
| K | float [0,1] | Stakes |
| A | float [0,1] | Alignment |

| HS Score | Mode | Decision |
|----------|------|----------|
| >= 0.70 | Critical | BLOCK |
| >= 0.45 | High-Stakes | PAUSE |
| < 0.45 | Normal | ALLOW |

### Gate 2: IrreversibilityGate

| Reversibility | Normal | High-Stakes | Critical |
|---------------|--------|-------------|----------|
| Reversible | ALLOW | ALLOW | ALLOW |
| Partially | ALLOW | PAUSE | PAUSE |
| Irreversible | ALLOW* | BLOCK | BLOCK |

*Requires clearance token + evidence binding

### Gate 3: PermissionGate

Checks:
- Tool in allowlist
- Network enabled (if required)
- Filesystem path allowed (if required)
- Code execution (blocked in Critical, sandbox in High-Stakes)

### Gate 4: AntiRecursionGate

| Cap | Normal | High-Stakes | Critical |
|-----|--------|-------------|----------|
| RPM Depth | 3 | 2 | 0 |
| RPM Breadth | 5 | 5 | 0 |
| Tool Calls | 12 | 8 | 0 |

Plus uncertainty-drop rule: PAUSE if uncertainty doesn't drop after 2 expansions.

### Gate 5: SelfModificationGate

| Mode | Self-Modification |
|------|-------------------|
| Normal | ALLOW (version required) |
| High-Stakes | ALLOW (replay + rollback required) |
| Critical | BLOCK |

---

## 5. ActionProfile Contract

Every action must have a profile:

```python
@dataclass
class ActionProfile:
    action_type: str                   # "tool_call", "memory_write", etc.
    tool_name: Optional[str]           # If tool_call
    reversibility: ActionReversibility # reversible/partial/irreversible
    requires_network: bool
    requires_filesystem: bool
    requires_code_execution: bool
    target_scope: Optional[str]        # Path/account/system
    estimated_cost: float
    is_self_modification: bool
    evidence_ids: List[str]
```

---

## 6. ClearanceToken Contract

For irreversible actions:

```python
@dataclass
class ClearanceToken:
    token_id: str
    scope: str                         # "file_delete", "security_change", etc.
    episode_id: str
    approved_by: str                   # "user", "admin", "verifier"
    evidence_ids: List[str]
    expires_at: datetime
    constraints: Dict
```

---

## 7. State Persistence

### DPSState

Must be serialized/deserialized between episodes:

```python
state.to_dict()  # Serialize
DPSState.from_dict(data)  # Deserialize
```

### GovernanceRails

Stateless - configuration only. No persistence needed.

---

## 8. Event Logging

### DPS Events

```python
{
    "event_type": "NOVELTY_CANDIDATE",
    "episode_id": str,
    "novelty_type": ["A", "B", "C"],  # New equation, RPM shortcut, cross-domain
    "scores": {"V": float, "I": float, "S": float, "NW": float},
    "verifier": {"status": str, "gate": str},
    "replay": {"passed": bool, "reward_delta": float}
}

{
    "event_type": "DEPTH_UNLOCK",
    "episode_id": str,
    "old_p_max": int,
    "new_p_max": int,
    "reason": str,
    "policy_versions": {...}
}
```

### Governance Events

```python
{
    "event_type": "GOVERNANCE_CHECK",
    "episode_id": str,
    "action": ActionProfile,
    "result": GateResult,
    "policy_version": str
}
```

---

## 9. Error Handling

### DPS

- If memory_embeddings not provided, skip similarity computation (S=0)
- If verifier_signals not provided, use defaults (V=0.5)
- If reward_signals not provided, use defaults (I=0.5)

### Governance

- If any required field missing, default to PAUSE
- Never guess, never execute anyway, never work around
- Allowed outputs when blocked: ASK, VERIFY checklist, dry-run proposal

---

## 10. Testing Contracts

### DPS Unit Tests

```python
def test_novelty_weight_computation():
    """NW = V × I × (1-S)"""
    pass

def test_classification_thresholds():
    """HEAVY >= 0.70, COUNT >= 0.45"""
    pass

def test_state_updates():
    """Token accumulation, unlock logic"""
    pass

def test_adaptive_iteration():
    """Iteration budget based on p_max"""
    pass
```

### Governance Unit Tests

```python
def test_high_stakes_formula():
    """HS = (U × K) × A²"""
    pass

def test_mode_determination():
    """Critical >= 0.70, High-Stakes >= 0.45"""
    pass

def test_gate_ordering():
    """First BLOCK stops, PAUSEs accumulate"""
    pass

def test_cap_enforcement():
    """Depth, breadth, tool call caps"""
    pass
```

---

## 11. Version Compatibility

| Component | Min v4 Version | Schema Version |
|-----------|----------------|----------------|
| DPSGatingLayer | 4.2+ | 1.0 |
| GovernanceRails | 4.2+ | 1.0 |
| Trace Schema | 1.0+ | 1.0 |

---

## 12. Implementation Checklist for agent-be

### DPSGatingLayer

- [ ] Implement ValidityNetwork (noetic_dim → 3)
- [ ] Implement ImpactNetwork (noetic_dim + context → 4)
- [ ] Implement SimilarityNetwork (noetic_dim × 2 → 2)
- [ ] Implement GatingNetwork (noetic_dim + features → noetic_dim)
- [ ] Implement state update logic
- [ ] Implement adaptive iteration controller
- [ ] Add trace logging
- [ ] Add unit tests

### GovernanceRails

- [ ] Integrate with existing Router/Executor
- [ ] Wire up action profiling
- [ ] Implement clearance token system
- [ ] Add event logging
- [ ] Add unit tests

### v4 Integration

- [ ] Add `use_dps_gating` config flag
- [ ] Add `use_governance_rails` config flag
- [ ] Insert DPS after RPMGating
- [ ] Wire Governance before/after actions
- [ ] Update trace schema
- [ ] Verify backward compatibility

---

End of Contracts.

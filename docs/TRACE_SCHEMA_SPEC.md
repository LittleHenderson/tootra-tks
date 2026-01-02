# TKS Traceable Transformer: Trace Schema Specification v1.0

**Status:** Authoritative Specification
**Scope:** Phases 1-2 Implementation
**Prepared by:** tks-supervisor
**Date:** 2025-12-23

---

## 1. Overview

This document defines the canonical trace schema for the Traceable TKS-Transformer. The trace provides a constant-size (approximately 200 bytes per token) summary of all model computations, sufficient to predict model outputs without storing full intermediate tensors.

### 1.1 Design Principles

1. **Constant Size:** Trace size is O(1) per token, not O(seq) or O(seq^2)
2. **Sufficiency:** Trace must be sufficient to reconstruct/predict logits within tolerance
3. **Efficiency:** Compression must be fast enough for training (< 5% overhead)
4. **Canonicity:** Schema aligns with TKS noetic space (40D), 4 worlds, 7 foundations

### 1.2 Relation to Existing Architecture

The trace schema wraps outputs from:
- `tks_llm_core.py`: NoeticEmbeddingLayer, NoeticProcessor, FractalAttentionMechanism
- `tks_llm_core_v2.py`: StableAttractorLayer, RPMGatingMechanism
- `tks_llm_core_v4.py`: TKSNoeticLM, NoeticRouter, CausalFractalAttentionMechanism

---

## 2. Per-Token Trace Record

Each token position produces a trace record with the following structure:

```
TraceRecord {
    noetic_routing: NoeticRoutingTrace
    attention: AttentionTrace
    attractor: AttractorTrace
    rpm: RPMTrace
    operator_core: OperatorCoreTrace  // optional, only for equation tokens
}
```

### 2.1 Noetic Routing Trace

Captures which noetic transforms are activated and their weights.

```python
@dataclass
class NoeticRoutingTrace:
    """
    Summary of noetic routing decisions.

    From NoeticRouter in tks_llm_core_v4.py:
    - Router produces softmax weights over 10 noetics
    - We store only top-k indices and weights

    Storage: 3*4 + 3*4 + 4*4 = 28 bytes (k=3)
    """
    topk_indices: List[int]      # top-k noetic indices (0-9), length=k, default k=3
    topk_weights: List[float]    # corresponding softmax weights, length=k
    world_mix: Dict[str, float]  # {"A": norm, "B": norm, "C": norm, "D": norm}
                                 # L2 norms of each world's activation
```

**Constraints:**
- `topk_indices` must be sorted by weight descending
- `topk_weights` must sum to <= 1.0 (since they're top-k of a softmax)
- `world_mix` values are non-negative L2 norms

**Rationale:** The full noetic weight tensor is [batch, seq, 10]. We compress to top-k (default k=3) which captures >95% of routing information in typical cases.

### 2.2 Attention Trace

Captures multi-scale attention behavior without storing full attention matrices.

```python
@dataclass
class AttentionTrace:
    """
    Summary statistics of fractal attention.

    From CausalFractalAttentionMechanism in tks_llm_core_v4.py:
    - num_scales (default 3) attention heads at different resolutions
    - We store per-scale mixing weights, NOT full attention matrices

    Storage: num_scales * 4 = 12 bytes (for 3 scales)
    """
    scale_weights: List[float]  # [num_scales] attention scale mixing weights
                                # These are the softmax weights over scales
```

**Constraints:**
- `len(scale_weights) == config.num_scales`
- `sum(scale_weights) == 1.0` (softmax output)

**What we DON'T store:**
- Full attention matrices [seq, seq] per scale (too large)
- Per-head attention patterns
- Key/value cached states

**Rationale:** The scale mixing weights capture the "which scale dominates" decision, which is the key structural information. Full attention matrices would require O(seq^2) storage.

### 2.3 Attractor Trace

Captures attractor convergence behavior.

```python
@dataclass
class AttractorTrace:
    """
    Summary of attractor fixed-point iteration.

    From StableAttractorLayer in tks_llm_core_v2.py:
    - Iterates until convergence or max_iterations
    - We store convergence statistics, NOT full trajectory

    Storage: 4 + 1 + 4 + 4 = 13 bytes
    """
    iterations: int          # Number of iterations used (0 to max_iterations)
    converged: bool          # True if delta < tolerance before max_iterations
    final_delta: float       # ||x_{t+1} - x_t|| at termination
    trajectory_norm: float   # L2 norm of attractor trajectory: sum(||x_t||)
```

**Constraints:**
- `iterations <= config.max_attractor_iter` (default 15)
- `converged == True` implies `final_delta < tolerance` (default 1e-3)
- `trajectory_norm >= 0`

**What we DON'T store:**
- Full trajectory [max_iter, batch, seq, 40] (too large)
- Per-iteration delta sequence
- Intermediate states

**Rationale:** The convergence statistics are sufficient to characterize attractor behavior for loss computation and debugging.

### 2.4 RPM Trace

Captures RPM (Desire/Wisdom/Power) gating decisions.

```python
@dataclass
class RPMTrace:
    """
    Summary of RPM gating mechanism.

    From RPMGatingMechanism in tks_llm_core_v2.py:
    - 7 Foundations, each with D/W/P scores
    - Gate = D * W * P for selected Foundation

    Storage: 4 + 4 + 7*4 = 36 bytes
    """
    foundation_idx: int        # Winning foundation (0-6, argmax of gates)
    gate_value: float          # Final gate value (D * W * P for winner)
    dwp_scores: List[float]    # [7] Per-foundation gate values (D*W*P products)
```

**Foundation Index Mapping:**
```
0: Unity (F1)
1: Wisdom (F2)
2: Life (F3)
3: Companionship (F4)
4: Power (F5)
5: Material (F6)
6: Lust (F7)
```

**Constraints:**
- `0 <= foundation_idx <= 6`
- `0.0 <= gate_value <= 1.0`
- `0.0 <= dwp_scores[i] <= 1.0` for all i

**What we DON'T store:**
- Individual D, W, P scores (9 values per Foundation = 63 total)
- Full DWP tensor [batch, seq, 7, 3]

### 2.5 Operator Core Trace (Optional)

Only present when processing equation tokens (detected by EquationDetector or NL retriever).

```python
@dataclass
class OperatorCoreTrace:
    """
    Summary of operator core equation processing.

    From TKSCompositionalLayerV2 (if enabled):
    - Processes TKS equation triplets (left, operator, right)
    - Per-world gated composition

    Storage: 4*4 + 4 + 4 = 24 bytes
    """
    gate_values: List[float]       # [4] Per-world gate values (A, B, C, D)
    symmetry_violation: float      # Operator symmetry loss (should be near 0)
    equation_repr_norm: float      # L2 norm of equation representation
```

**Constraints:**
- Present only when `equation_triplet is not None` in forward pass
- `0.0 <= gate_values[i] <= 1.0` for all i
- `symmetry_violation >= 0.0`

---

## 3. Storage Format

### 3.1 Size Budget

| Component | Fields | Size (bytes) |
|-----------|--------|--------------|
| noetic_routing | topk_indices[3], topk_weights[3], world_mix[4] | 28 |
| attention | scale_weights[3] | 12 |
| attractor | iterations, converged, final_delta, trajectory_norm | 13 |
| rpm | foundation_idx, gate_value, dwp_scores[7] | 36 |
| operator_core | gate_values[4], symmetry_violation, equation_repr_norm | 24 (optional) |
| **Total (no operator)** | | **~89 bytes** |
| **Total (with operator)** | | **~113 bytes** |

With JSON serialization overhead, expect approximately **150-200 bytes per token**.

### 3.2 JSONL Serialization Format

Each line is a JSON object:

```json
{
  "token_idx": 42,
  "noetic_routing": {
    "topk_indices": [1, 4, 7],
    "topk_weights": [0.45, 0.32, 0.15],
    "world_mix": {"A": 2.31, "B": 1.87, "C": 2.05, "D": 1.92}
  },
  "attention": {
    "scale_weights": [0.52, 0.31, 0.17]
  },
  "attractor": {
    "iterations": 8,
    "converged": true,
    "final_delta": 0.00042,
    "trajectory_norm": 15.7
  },
  "rpm": {
    "foundation_idx": 4,
    "gate_value": 0.73,
    "dwp_scores": [0.21, 0.45, 0.38, 0.52, 0.73, 0.31, 0.28]
  }
}
```

With operator core (for equation tokens):

```json
{
  "token_idx": 42,
  "noetic_routing": { ... },
  "attention": { ... },
  "attractor": { ... },
  "rpm": { ... },
  "operator_core": {
    "gate_values": [0.85, 0.72, 0.68, 0.91],
    "symmetry_violation": 0.0012,
    "equation_repr_norm": 4.52
  }
}
```

### 3.3 Batched Format

For batch processing, use `TraceBatch`:

```json
{
  "batch_size": 4,
  "seq_len": 128,
  "traces": [
    [ /* token 0 traces for all batch items */ ],
    [ /* token 1 traces for all batch items */ ],
    ...
  ]
}
```

---

## 4. Compression Algorithm

### 4.1 Full Trace to Compressed Trace

The `compress_trace` function transforms full model outputs to the compressed schema:

```python
def compress_trace(full_trace: dict, top_k: int = 3) -> dict:
    """
    Compress full model trace to constant-size schema.

    Input full_trace keys (from model.forward(return_full_trace=True)):
        - "embedding": [batch, seq, 40]
        - "worlds": {"A": [...], "B": [...], "C": [...], "D": [...]}
        - "blocks": List[{"noetic_weights": [...], "attn_weights": [...]}]
        - "attractor": [batch, seq, 40]
        - "attractor_converged": bool
        - "attractor_iterations": int
        - "rpm_gate": [batch, seq]
        - "dwp_scores": [batch, seq, 7, 3]
        - "operator_core": {...} (optional)

    Output: TraceRecord dict per token position
    """
```

**Algorithm:**

1. **Noetic Routing:**
   - Extract `noetic_weights` from last block (or aggregate across blocks)
   - Sort indices by weight descending
   - Keep top-k indices and weights
   - Compute world L2 norms from embedding

2. **Attention:**
   - Extract `attn_weights` from last block
   - Compute per-scale mean attention energy
   - Normalize to get scale mixing weights

3. **Attractor:**
   - Copy `attractor_converged` to `converged`
   - Copy `attractor_iterations` to `iterations`
   - Compute `final_delta` from delta_sequence[-1] if available
   - Compute `trajectory_norm` from trajectory if available, else from attractor tensor

4. **RPM:**
   - From `dwp_scores` [batch, seq, 7, 3], compute product D*W*P per Foundation
   - Find argmax for `foundation_idx`
   - Copy corresponding gate value

5. **Operator Core (if present):**
   - Copy gate_values directly
   - Copy symmetry_violation from symmetry_losses
   - Compute equation_repr_norm from equation_repr

---

## 5. Trace Consistency Requirement

### 5.1 Definition

A trace is **consistent** if it contains sufficient information to predict model logits within tolerance:

```
TraceConsistency := ||predict_logits(trace) - true_logits||_2 < epsilon
```

Where `predict_logits` is a learned function that takes only the compressed trace as input.

### 5.2 Sufficiency Metric

```python
def compute_trace_sufficiency(
    trace_pred_logits: torch.Tensor,  # [batch, seq, vocab]
    true_logits: torch.Tensor,        # [batch, seq, vocab]
) -> float:
    """
    Compute how well the trace predicts true logits.

    Returns:
        Sufficiency score in [0, 1] where 1.0 = perfect prediction
    """
    mse = F.mse_loss(trace_pred_logits, true_logits)
    # Normalize by logit variance
    variance = true_logits.var()
    sufficiency = 1.0 - (mse / (variance + 1e-8)).clamp(max=1.0)
    return sufficiency.item()
```

### 5.3 Sufficiency Target

- **Phase 1 Target:** sufficiency >= 0.85 (trace explains 85% of logit variance)
- **Phase 2 Target:** sufficiency >= 0.95 (trace explains 95% of logit variance)

---

## 6. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial specification |

---

## 7. Agent Responsibilities

| Agent | Responsibility |
|-------|----------------|
| Agent A | Implement `trace_utils.py` with compress/serialize functions |
| Agent B | Implement `TraceConsistencyLoss` in `training/losses.py` |
| Agent C | Integrate trace collection into `tks_llm_core_v4.py` forward pass |
| Agent D | Write tests and validation for trace schema compliance |

---

## 8. Appendix: Full Trace vs Compressed Trace Size

For a typical forward pass with batch=4, seq=128, vocab=512:

| Tensor | Full Size | Compressed Size |
|--------|-----------|-----------------|
| noetic_weights | 4 * 128 * 10 * 4 = 20KB | 4 * 128 * 7 * 4 = 14KB |
| attention | 4 * 3 * 128 * 128 * 4 = 768KB | 4 * 128 * 3 * 4 = 6KB |
| attractor trajectory | 4 * 15 * 128 * 40 * 4 = 1.2MB | 4 * 128 * 4 * 4 = 8KB |
| dwp_scores | 4 * 128 * 7 * 3 * 4 = 43KB | 4 * 128 * 8 * 4 = 16KB |
| **Total** | **~2MB** | **~44KB** |

**Compression ratio:** approximately **45x**

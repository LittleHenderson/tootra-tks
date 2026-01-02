# TKS Traceable Transformer: Implementation Contracts

**Status:** Authoritative Interface Specification
**Scope:** Phases 1-2 Implementation
**Prepared by:** tks-supervisor
**Date:** 2025-12-23

---

## 1. Overview

This document defines the exact function signatures and class interfaces that implementation agents must follow. All agents should treat these contracts as binding.

### 1.1 File Structure

```
/mnt/c/Users/wakil/Downloads/Everthing-Tootra-TKS/
    training/
        losses.py           # Agent B: Add TraceConsistencyLoss
        trace_utils.py      # Agent A: New file (create)
    tks_llm_core_v4.py      # Agent C: Modify to collect traces
    tests/
        test_trace_schema.py # Agent D: New file (create)
```

---

## 2. Agent A: trace_utils.py Contracts

Create new file: `training/trace_utils.py`

### 2.1 Data Classes

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

@dataclass
class NoeticRoutingTrace:
    """Noetic routing summary for one token position."""
    topk_indices: List[int]      # length = top_k (default 3)
    topk_weights: List[float]    # length = top_k
    world_mix: Dict[str, float]  # {"A": float, "B": float, "C": float, "D": float}


@dataclass
class AttentionTrace:
    """Attention summary for one token position."""
    scale_weights: List[float]   # length = num_scales (default 3)


@dataclass
class AttractorTrace:
    """Attractor convergence summary for one token position."""
    iterations: int
    converged: bool
    final_delta: float
    trajectory_norm: float


@dataclass
class RPMTrace:
    """RPM gating summary for one token position."""
    foundation_idx: int          # 0-6
    gate_value: float            # 0.0-1.0
    dwp_scores: List[float]      # length = 7


@dataclass
class OperatorCoreTrace:
    """Operator core summary (optional, for equation tokens)."""
    gate_values: List[float]     # length = 4 (A, B, C, D)
    symmetry_violation: float
    equation_repr_norm: float


@dataclass
class TraceRecord:
    """Complete trace for one token position."""
    token_idx: int
    noetic_routing: NoeticRoutingTrace
    attention: AttentionTrace
    attractor: AttractorTrace
    rpm: RPMTrace
    operator_core: Optional[OperatorCoreTrace] = None
```

### 2.2 Compression Function

```python
def compress_trace(
    full_trace: Dict[str, torch.Tensor],
    top_k: int = 3,
    num_scales: int = 3,
) -> List[TraceRecord]:
    """
    Compress full model trace to constant-size TraceRecords.

    Args:
        full_trace: Dict from model.forward(return_full_trace=True)
            Required keys:
                - "embedding": Tensor [batch, seq, 40]
                - "worlds": Dict[str, Tensor] with A, B, C, D
                - "blocks": List[Dict] with per-block traces
                - "attractor": Tensor [batch, seq, 40]
                - "attractor_converged": bool
                - "attractor_iterations": int
                - "rpm_gate": Tensor [batch, seq]
                - "dwp_scores": Tensor [batch, seq, 7, 3]
            Optional keys:
                - "operator_core": Dict with equation processing info

        top_k: Number of top noetic indices to keep (default 3)
        num_scales: Number of attention scales (default 3)

    Returns:
        List[TraceRecord]: One TraceRecord per token position
            Length = batch_size * seq_len

    Raises:
        ValueError: If required keys are missing from full_trace
        ValueError: If tensor shapes are inconsistent

    Example:
        >>> model = TKSNoeticLM(config)
        >>> out = model(tokens, return_full_trace=True)
        >>> traces = compress_trace(out["trace"])
        >>> len(traces)  # == batch_size * seq_len
        512
    """
    # Implementation by Agent A
    pass
```

### 2.3 Serialization Functions

```python
def serialize_trace(trace: TraceRecord) -> str:
    """
    Serialize a TraceRecord to a JSONL-compatible string.

    Args:
        trace: TraceRecord to serialize

    Returns:
        JSON string (single line, no trailing newline)

    Example:
        >>> trace = TraceRecord(token_idx=0, ...)
        >>> line = serialize_trace(trace)
        >>> isinstance(line, str)
        True
        >>> '\\n' not in line
        True
    """
    pass


def deserialize_trace(line: str) -> TraceRecord:
    """
    Deserialize a JSONL line to a TraceRecord.

    Args:
        line: JSON string (from serialize_trace)

    Returns:
        TraceRecord

    Raises:
        json.JSONDecodeError: If line is not valid JSON
        KeyError: If required fields are missing
        ValueError: If field values are invalid

    Example:
        >>> line = '{"token_idx": 0, "noetic_routing": {...}, ...}'
        >>> trace = deserialize_trace(line)
        >>> trace.token_idx
        0
    """
    pass
```

### 2.4 Batch Processing Class

```python
class TraceBatch:
    """
    Container for batched trace processing.

    Provides efficient storage and access for traces from a full batch.
    """

    def __init__(
        self,
        traces: List[TraceRecord],
        batch_size: int,
        seq_len: int,
    ):
        """
        Initialize TraceBatch from list of TraceRecords.

        Args:
            traces: List of TraceRecords (length = batch_size * seq_len)
            batch_size: Number of sequences in batch
            seq_len: Sequence length

        Raises:
            ValueError: If len(traces) != batch_size * seq_len
        """
        pass

    def get(self, batch_idx: int, seq_idx: int) -> TraceRecord:
        """
        Get TraceRecord for specific position.

        Args:
            batch_idx: Index within batch (0 to batch_size-1)
            seq_idx: Index within sequence (0 to seq_len-1)

        Returns:
            TraceRecord at that position

        Raises:
            IndexError: If indices out of bounds
        """
        pass

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """
        Convert traces to stacked tensors for model input.

        Returns:
            Dict with keys:
                - "noetic_topk_indices": [batch, seq, top_k] LongTensor
                - "noetic_topk_weights": [batch, seq, top_k] FloatTensor
                - "world_norms": [batch, seq, 4] FloatTensor
                - "scale_weights": [batch, seq, num_scales] FloatTensor
                - "attractor_iterations": [batch, seq] LongTensor
                - "attractor_converged": [batch, seq] BoolTensor
                - "attractor_final_delta": [batch, seq] FloatTensor
                - "rpm_foundation_idx": [batch, seq] LongTensor
                - "rpm_gate_value": [batch, seq] FloatTensor
                - "rpm_dwp_scores": [batch, seq, 7] FloatTensor
                - "has_operator_core": [batch, seq] BoolTensor
                - "operator_gate_values": [batch, seq, 4] FloatTensor (zeros where absent)
        """
        pass

    def save_jsonl(self, path: str) -> None:
        """
        Save traces to JSONL file.

        Args:
            path: Output file path

        File format: One JSON object per line, each line is a TraceRecord.
        """
        pass

    @classmethod
    def load_jsonl(cls, path: str) -> "TraceBatch":
        """
        Load TraceBatch from JSONL file.

        Args:
            path: Input file path

        Returns:
            TraceBatch instance

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If file format is invalid
        """
        pass
```

### 2.5 Utility Functions

```python
def trace_feature_dim() -> int:
    """
    Return total feature dimension when traces are flattened.

    This is the input dimension for TraceConsistencyLoss predictor.

    Returns:
        int: Total feature dimension

    Calculation:
        noetic: top_k + top_k + 4 = 10
        attention: num_scales = 3
        attractor: 4 (iterations, converged, final_delta, trajectory_norm)
        rpm: 1 + 1 + 7 = 9
        operator_core: 1 + 4 + 1 + 1 = 7 (has_operator flag + values)
        Total: 33 features
    """
    return 33


def traces_to_features(batch: TraceBatch) -> torch.Tensor:
    """
    Convert TraceBatch to feature tensor for loss computation.

    Args:
        batch: TraceBatch instance

    Returns:
        Tensor [batch_size, seq_len, trace_feature_dim()]
    """
    pass
```

---

## 3. Agent B: TraceConsistencyLoss Contracts

Add to existing file: `training/losses.py`

### 3.1 TraceConsistencyLoss Class

```python
class TraceConsistencyLoss(nn.Module):
    """
    Loss for training trace-to-logit prediction.

    This loss ensures that the compressed trace contains sufficient
    information to predict model logits. A small trace predictor network
    learns to map trace features to logits.

    Architecture:
        trace_features [batch, seq, 33] -> MLP -> logits [batch, seq, vocab]

    Loss:
        L = MSE(predicted_logits, true_logits) + KL(softmax(pred), softmax(true))

    Usage:
        loss_fn = TraceConsistencyLoss(vocab_size=512, hidden_dim=128)

        # During training:
        out = model(tokens, return_full_trace=True)
        traces = compress_trace(out["trace"])
        batch = TraceBatch(traces, ...)
        features = traces_to_features(batch)

        loss = loss_fn(features, out["logits"])
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        kl_weight: float = 0.1,
    ):
        """
        Initialize TraceConsistencyLoss.

        Args:
            vocab_size: Output vocabulary size (must match model)
            hidden_dim: Hidden dimension of predictor MLP
            num_layers: Number of MLP layers (minimum 2)
            dropout: Dropout probability
            kl_weight: Weight for KL divergence term (default 0.1)

        The predictor architecture:
            Linear(33, hidden_dim) -> ReLU -> Dropout
            [Linear(hidden_dim, hidden_dim) -> ReLU -> Dropout] * (num_layers - 2)
            Linear(hidden_dim, vocab_size)
        """
        super().__init__()
        # Implementation by Agent B
        pass

    def forward(
        self,
        trace_features: torch.Tensor,
        true_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute trace consistency loss.

        Args:
            trace_features: [batch, seq, 33] trace feature tensor
            true_logits: [batch, seq, vocab] true model logits
            mask: Optional [batch, seq] mask for valid positions

        Returns:
            Dict with:
                - "total": Combined loss (scalar)
                - "mse": MSE component (scalar)
                - "kl": KL divergence component (scalar)
                - "predicted_logits": [batch, seq, vocab] predicted logits
                - "sufficiency": Scalar in [0, 1] (higher = better)

        Example:
            >>> loss_fn = TraceConsistencyLoss(vocab_size=512)
            >>> features = torch.randn(4, 128, 33)
            >>> logits = torch.randn(4, 128, 512)
            >>> out = loss_fn(features, logits)
            >>> out["total"].backward()
        """
        pass

    def predict_logits(
        self,
        trace_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict logits from trace features only.

        Args:
            trace_features: [batch, seq, 33] trace features

        Returns:
            [batch, seq, vocab] predicted logits
        """
        pass
```

### 3.2 Sufficiency Computation Function

```python
def compute_trace_sufficiency(
    trace_pred_logits: torch.Tensor,
    true_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute how well trace predicts true logits.

    This is the primary metric for trace quality.

    Args:
        trace_pred_logits: [batch, seq, vocab] logits predicted from trace
        true_logits: [batch, seq, vocab] true model logits
        mask: Optional [batch, seq] mask for valid positions

    Returns:
        Sufficiency score in [0, 1]:
            1.0 = trace perfectly predicts logits
            0.0 = trace has no predictive power

    Formula:
        sufficiency = 1 - (MSE / Var(true_logits))

    Example:
        >>> pred = torch.randn(4, 128, 512)
        >>> true = torch.randn(4, 128, 512)
        >>> score = compute_trace_sufficiency(pred, true)
        >>> 0.0 <= score <= 1.0
        True
    """
    pass
```

### 3.3 Integration with TKSLoss

```python
class TKSLossWithTrace(nn.Module):
    """
    Extended TKS loss including trace consistency term.

    L_total = L_tks + lambda_trace * L_trace

    Where L_tks is the existing TKSLoss and L_trace is TraceConsistencyLoss.
    """

    def __init__(
        self,
        tks_config: Optional[TKSLossConfig] = None,
        vocab_size: int = 512,
        lambda_trace: float = 0.1,
        trace_hidden_dim: int = 128,
    ):
        """
        Initialize combined loss.

        Args:
            tks_config: Configuration for TKSLoss
            vocab_size: Vocabulary size for trace predictor
            lambda_trace: Weight for trace consistency loss
            trace_hidden_dim: Hidden dim for trace predictor
        """
        super().__init__()
        self.tks_loss = TKSLoss(tks_config)
        self.trace_loss = TraceConsistencyLoss(
            vocab_size=vocab_size,
            hidden_dim=trace_hidden_dim,
        )
        self.lambda_trace = lambda_trace

    def forward(
        self,
        pipeline_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        pipeline: nn.Module,
        trace_batch: "TraceBatch",  # From trace_utils
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pipeline_output: Dict from model.forward(return_full_trace=True)
            targets: Target token indices
            pipeline: The model
            trace_batch: Compressed traces
            **kwargs: Additional args for TKSLoss

        Returns:
            Dict with all TKSLoss components plus:
                - "trace_total": Trace consistency loss
                - "trace_mse": Trace MSE component
                - "trace_kl": Trace KL component
                - "trace_sufficiency": Sufficiency score
        """
        pass
```

---

## 4. Agent C: Model Integration Contracts

Modify existing file: `tks_llm_core_v4.py`

### 4.1 TKSNoeticLM.forward Modifications

The existing `forward` method must be modified to collect trace data compatible with the schema:

```python
def forward(
    self,
    tokens: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    noetic_idx: Optional[int] = None,
    noetic_weights: Optional[torch.Tensor] = None,
    fractal_indices: Optional[torch.Tensor] = None,
    goal_state: Optional[torch.Tensor] = None,
    target_foundation: Optional[int] = None,
    equation_triplet: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    return_full_trace: bool = False,
    return_compressed_trace: bool = False,  # NEW PARAMETER
) -> Dict[str, torch.Tensor]:
    """
    Forward pass with optional trace collection.

    NEW: return_compressed_trace parameter.

    When return_compressed_trace=True:
        - Automatically collects all trace data
        - Returns "compressed_trace" key with List[TraceRecord]
        - More efficient than return_full_trace + external compression

    Args:
        ... (existing args)
        return_compressed_trace: If True, return compressed traces directly.
            This is preferred over return_full_trace for production use.

    Returns:
        Dict with existing keys plus:
            - "compressed_trace": List[TraceRecord] (if return_compressed_trace=True)
    """
```

### 4.2 Required Trace Data Collection Points

Agent C must ensure the following data is collected during forward pass:

```python
# In NoeticBlock.forward():
# After router:
trace["noetic_weights"] = weights.detach()  # [batch, seq, 10]

# After attention:
# Must compute scale_weights from attention mechanism
trace["scale_weights"] = scale_weights.detach()  # [batch, seq, num_scales]

# In TKSNoeticLM.forward():
# After attractor:
trace["attractor_iterations"] = attractor_out["iterations"]
trace["attractor_converged"] = attractor_out["converged"]
trace["attractor_final_delta"] = attractor_out.get("final_delta", 0.0)
trace["attractor_trajectory_norm"] = compute_trajectory_norm(attractor_out)  # NEW

# After RPM:
trace["rpm_foundation_idx"] = rpm_out["dwp_scores"].prod(dim=-1).argmax(dim=-1)  # NEW
trace["rpm_gate_value"] = rpm_out["rpm_gate"]
trace["dwp_product"] = rpm_out["dwp_scores"].prod(dim=-1)  # NEW: [batch, seq, 7]

# After operator_core (if present):
trace["operator_gate_values"] = operator_out["gate_values"]
trace["operator_symmetry"] = operator_out.get("symmetry_losses", {}).get("total", 0.0)
trace["operator_repr_norm"] = operator_out["equation_repr"].norm(dim=-1)
```

### 4.3 CausalFractalAttentionMechanism Modifications

```python
def forward(
    self,
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_weights: bool = False,
    return_scale_weights: bool = False,  # NEW PARAMETER
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Forward pass with optional scale weight return.

    NEW: return_scale_weights parameter for trace collection.

    Returns:
        Tuple of:
            - output: [batch, seq, dim]
            - attn_weights: [batch, num_scales, seq, seq] if return_weights
            - scale_weights: [batch, seq, num_scales] if return_scale_weights
    """
```

---

## 5. Agent D: Test Contracts

Create new file: `tests/test_trace_schema.py`

### 5.1 Schema Validation Tests

```python
import pytest
import torch
from training.trace_utils import (
    TraceRecord,
    NoeticRoutingTrace,
    AttentionTrace,
    AttractorTrace,
    RPMTrace,
    OperatorCoreTrace,
    compress_trace,
    serialize_trace,
    deserialize_trace,
    TraceBatch,
    traces_to_features,
)


class TestTraceRecord:
    """Tests for TraceRecord dataclass."""

    def test_trace_record_creation(self):
        """TraceRecord can be created with all required fields."""
        pass

    def test_trace_record_optional_operator(self):
        """TraceRecord works with and without operator_core."""
        pass

    def test_trace_record_field_types(self):
        """All fields have correct types."""
        pass


class TestCompress:
    """Tests for compress_trace function."""

    def test_compress_basic(self):
        """compress_trace produces correct number of TraceRecords."""
        pass

    def test_compress_missing_keys(self):
        """compress_trace raises ValueError for missing required keys."""
        pass

    def test_compress_topk(self):
        """Top-k indices are sorted by weight descending."""
        pass

    def test_compress_world_norms(self):
        """World norms are non-negative."""
        pass


class TestSerialize:
    """Tests for serialization functions."""

    def test_serialize_roundtrip(self):
        """serialize_trace and deserialize_trace are inverses."""
        pass

    def test_serialize_no_newlines(self):
        """Serialized output has no embedded newlines."""
        pass

    def test_deserialize_invalid_json(self):
        """deserialize_trace raises on invalid JSON."""
        pass


class TestTraceBatch:
    """Tests for TraceBatch class."""

    def test_batch_indexing(self):
        """TraceBatch.get returns correct TraceRecord."""
        pass

    def test_batch_to_tensor(self):
        """to_tensor_dict produces correctly shaped tensors."""
        pass

    def test_batch_save_load(self):
        """save_jsonl and load_jsonl are inverses."""
        pass


class TestTraceIntegration:
    """Integration tests with actual model."""

    def test_model_trace_collection(self):
        """TKSNoeticLM collects all required trace data."""
        from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

        config = TKSNoeticLMConfig(vocab_size=100, max_seq_len=32)
        model = TKSNoeticLM(config)

        tokens = torch.randint(0, 100, (2, 16))
        out = model(tokens, return_full_trace=True)

        trace = out["trace"]

        # Verify all required keys present
        assert "embedding" in trace
        assert "blocks" in trace
        assert "attractor_converged" in trace
        assert "attractor_iterations" in trace
        assert "dwp_scores" in trace

    def test_trace_compression_integration(self):
        """Compressed traces have correct structure."""
        pass

    def test_trace_sufficiency_integration(self):
        """Trace sufficiency can be computed end-to-end."""
        pass
```

### 5.2 Loss Function Tests

```python
class TestTraceConsistencyLoss:
    """Tests for TraceConsistencyLoss."""

    def test_loss_forward(self):
        """Forward pass runs without error."""
        pass

    def test_loss_backward(self):
        """Gradients flow correctly."""
        pass

    def test_loss_sufficiency_range(self):
        """Sufficiency is in [0, 1]."""
        pass

    def test_loss_perfect_prediction(self):
        """Sufficiency = 1.0 when prediction is perfect."""
        pass

    def test_loss_with_mask(self):
        """Mask correctly excludes positions."""
        pass
```

### 5.3 Regression Tests

```python
class TestTraceRegression:
    """Regression tests to catch breaking changes."""

    def test_trace_size_budget(self):
        """Compressed trace size stays within budget."""
        # Each TraceRecord should serialize to < 250 bytes
        pass

    def test_trace_schema_backwards_compat(self):
        """Old trace files can still be loaded."""
        # Load pre-saved trace file and verify deserialization
        pass
```

---

## 6. Integration Checklist

This section will be updated by tks-supervisor as agents complete their work.

### 6.1 Phase 1: Core Implementation

| Task | Agent | Status | Notes |
|------|-------|--------|-------|
| Create trace_utils.py | A | PENDING | |
| Implement dataclasses | A | PENDING | |
| Implement compress_trace | A | PENDING | |
| Implement serialize/deserialize | A | PENDING | |
| Implement TraceBatch | A | PENDING | |
| Add TraceConsistencyLoss | B | PENDING | |
| Add compute_trace_sufficiency | B | PENDING | |
| Add TKSLossWithTrace | B | PENDING | |
| Modify TKSNoeticLM.forward | C | PENDING | |
| Modify CausalFractalAttention | C | PENDING | |
| Create test_trace_schema.py | D | PENDING | |
| Write all unit tests | D | PENDING | |
| Write integration tests | D | PENDING | |

### 6.2 Phase 2: Validation

| Task | Agent | Status | Notes |
|------|-------|--------|-------|
| All tests passing | D | PENDING | |
| Trace sufficiency >= 0.85 | ALL | PENDING | |
| Trace size <= 200 bytes/token | A | PENDING | |
| Training overhead < 5% | ALL | PENDING | |
| Documentation complete | ALL | PENDING | |

---

## 7. Handoff Notes

### 7.1 Dependencies Between Agents

```
Agent A (trace_utils.py)
    |
    v
Agent C (model integration) --> Agent B (loss functions)
    |                               |
    v                               v
Agent D (tests) <------------------+
```

- Agent A should complete trace_utils.py first (no dependencies)
- Agent C depends on Agent A for trace dataclasses
- Agent B depends on Agent A for TraceBatch and traces_to_features
- Agent D depends on all other agents

### 7.2 Coordination Points

1. **Trace feature dimension:** All agents must use trace_feature_dim() = 33
2. **Serialization format:** Follow JSONL format exactly as specified
3. **Tensor shapes:** Use [batch, seq, ...] convention everywhere
4. **Device handling:** All functions should be device-agnostic

### 7.3 Known Constraints from Existing Code

1. **TOTAL_DIM = 40:** Fixed noetic space dimension (from tks_llm_core.py)
2. **NUM_FOUNDATIONS = 7:** Fixed Foundation count (from tks_llm_core_v2.py)
3. **num_scales default = 3:** From CausalFractalAttentionMechanism
4. **max_attractor_iter default = 15:** From StableAttractorLayer
5. **top_k default = 3:** Design decision for noetic routing compression

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial contracts specification |

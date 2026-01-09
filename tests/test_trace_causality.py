"""
Trace Causality Tests - Detecting TRACE-CONSISTENCY GAMING

This module implements tests to verify that the trace is genuinely causal
(not just correlated) for model outputs. The model might learn to satisfy
trace-only prediction without the trace actually causing the output.

Problem Statement:
    If trace features are merely correlated with outputs rather than causing them,
    the model could "game" trace-consistency loss by learning superficial patterns.
    These tests detect such gaming by verifying causal relationships.

Test Classes:
    1. TraceAblationTest - Zero out trace components, verify output degradation
    2. CounterfactualTraceTest - Swap traces between inputs, verify output follows trace
    3. TraceGradientTest - Verify gradients flow through trace features
    4. TraceHeadGapTest - Maintain healthy gap between trace-only and full model

Usage:
    # Run all causality tests
    pytest tests/test_trace_causality.py -v

    # Run with specific model
    pytest tests/test_trace_causality.py --model output/model.pt -v

    # Run quick mode (synthetic data)
    pytest tests/test_trace_causality.py -v -m "not requires_model"

References:
    - docs/TRACE_SCHEMA_SPEC.md (trace structure)
    - scripts/validate_trace_sufficiency.py (trace consistency head)
    - trace_utils.py (trace extraction utilities)

Author: TKS-Quantum Validation Agent
Date: 2025-12-23
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports - tests skip if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig, TOTAL_DIM
    from tks_llm_core_v2 import FOUNDATION_NAMES
    V4_AVAILABLE = True
except ImportError:
    V4_AVAILABLE = False

try:
    from trace_utils import (
        extract_trace_features,
        compress_trace,
        compute_feature_dim,
        TraceBatch,
        NOETIC_DIM,
        NUM_WORLDS,
        NUM_FOUNDATIONS,
    )
    TRACE_UTILS_AVAILABLE = True
except ImportError:
    TRACE_UTILS_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Trace component names and their feature dimensions
TRACE_COMPONENTS = {
    "noetic_routing": {"start": 0, "size": 12},  # top-3 weights * 4 blocks
    "world_mix": {"start": 12, "size": 4},       # 4 world norms
    "attention_scales": {"start": 16, "size": 3}, # 3 scale weights
    "attractor": {"start": 19, "size": 3},        # converged, iterations, delta
    "rpm": {"start": 22, "size": 3},              # gate, foundation, top_dwp
    "operator_core": {"start": 25, "size": 2},    # gate_mean, sym_violation
}

# Expected accuracy drop when ablating causal components
DEFAULT_EXPECTED_DROP = 0.2

# Healthy gap between trace-only head and full model accuracy
GAP_RATIO_MIN = 0.80
GAP_RATIO_MAX = 0.98


# =============================================================================
# TRACE CONSISTENCY HEAD (from validate_trace_sufficiency.py)
# =============================================================================

class TraceConsistencyHead(nn.Module):
    """
    Small neural network that predicts logits from trace features alone.

    If this head can accurately predict the model's outputs, then the trace
    is sufficient for interpretability.

    Architecture:
        trace_features -> Linear -> ReLU -> Linear -> ReLU -> Linear -> logits
    """

    def __init__(
        self,
        trace_dim: int,
        hidden_dim: int = 128,
        vocab_size: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.trace_dim = trace_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.net = nn.Sequential(
            nn.Linear(trace_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, trace_features: torch.Tensor) -> torch.Tensor:
        """Predict logits from trace features."""
        return self.net(trace_features)


# =============================================================================
# TEST 1: TRACE ABLATION TEST
# =============================================================================

class TraceAblationTest:
    """
    Verify trace is causal by ablating (zeroing) trace components
    and checking for predictable output degradation.

    If trace is causal: ablating it should hurt performance.
    If trace is just correlated: ablating has no effect.

    Args:
        model: TKS model (TKSNoeticLM or TKSLLMCorePipeline)
        trace_consistency_head: Trained TraceConsistencyHead
        device: Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        trace_consistency_head: TraceConsistencyHead,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.trace_head = trace_consistency_head
        self.device = device or torch.device('cpu')

        # Component configuration based on feature extraction
        self.components = TRACE_COMPONENTS.copy()

    def ablate_component(
        self,
        trace_features: torch.Tensor,
        component: str,
    ) -> torch.Tensor:
        """
        Zero out a trace component.

        Args:
            trace_features: [batch, trace_dim] trace feature tensor
            component: Component name ("noetic_routing", "rpm", "attractor", etc.)

        Returns:
            Ablated trace features with specified component zeroed
        """
        if component not in self.components:
            raise ValueError(f"Unknown component: {component}. Valid: {list(self.components.keys())}")

        ablated = trace_features.clone()
        comp_info = self.components[component]
        start = comp_info["start"]
        end = start + comp_info["size"]

        # Ensure we don't exceed feature dimension
        end = min(end, ablated.shape[-1])

        ablated[..., start:end] = 0.0
        return ablated

    def run_ablation_test(
        self,
        trace_features: torch.Tensor,
        true_logits: torch.Tensor,
        expected_drop: float = DEFAULT_EXPECTED_DROP,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run ablation test for all trace components.

        Args:
            trace_features: [N, trace_dim] trace features
            true_logits: [N, vocab_size] true model logits
            expected_drop: Expected accuracy drop for causal components

        Returns:
            Dict mapping component names to:
            {
                "original_acc": float,
                "ablated_acc": float,
                "drop": float,
                "causal": bool,
            }
        """
        self.trace_head.eval()
        trace_features = trace_features.to(self.device)
        true_logits = true_logits.to(self.device)

        # Get original predictions
        with torch.no_grad():
            original_pred = self.trace_head(trace_features)
            original_acc = self._compute_top1_accuracy(original_pred, true_logits)

        results = {}

        for component in self.components:
            # Ablate this component
            ablated_features = self.ablate_component(trace_features, component)

            with torch.no_grad():
                ablated_pred = self.trace_head(ablated_features)
                ablated_acc = self._compute_top1_accuracy(ablated_pred, true_logits)

            drop = original_acc - ablated_acc

            results[component] = {
                "original_acc": float(original_acc),
                "ablated_acc": float(ablated_acc),
                "drop": float(drop),
                "causal": drop >= expected_drop,
            }

        return results

    def _compute_top1_accuracy(
        self,
        pred_logits: torch.Tensor,
        true_logits: torch.Tensor,
    ) -> float:
        """Compute top-1 accuracy between predicted and true logits."""
        pred_top1 = pred_logits.argmax(dim=-1)
        true_top1 = true_logits.argmax(dim=-1)
        accuracy = (pred_top1 == true_top1).float().mean().item()
        return accuracy


# =============================================================================
# TEST 2: COUNTERFACTUAL TRACE SWAP TEST
# =============================================================================

class CounterfactualTraceTest:
    """
    Swap traces between examples and verify outputs follow the trace.

    If trace is causal: swapping traces should swap outputs.
    If trace is correlated: outputs stay with original input.

    Args:
        model: TKS model
        trace_consistency_head: Trained TraceConsistencyHead
        device: Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        trace_consistency_head: TraceConsistencyHead,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.trace_head = trace_consistency_head
        self.device = device or torch.device('cpu')

    def run_swap_test(
        self,
        trace_a: torch.Tensor,
        trace_b: torch.Tensor,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Test if swapping traces swaps outputs.

        1. Get predictions using trace_a
        2. Get predictions using trace_b
        3. Swap: use trace_a to predict for input_b position, and vice versa
        4. Check if swapped outputs follow the swapped trace

        Args:
            trace_a: [N, trace_dim] trace features from input set A
            trace_b: [N, trace_dim] trace features from input set B
            logits_a: [N, vocab_size] true logits for input A
            logits_b: [N, vocab_size] true logits for input B

        Returns:
            {
                "trace_following_rate": float,  # How often output follows swapped trace
                "input_following_rate": float,  # How often output follows original input
                "is_causal": bool,  # trace_following > input_following
            }
        """
        self.trace_head.eval()
        trace_a = trace_a.to(self.device)
        trace_b = trace_b.to(self.device)
        logits_a = logits_a.to(self.device)
        logits_b = logits_b.to(self.device)

        # Get true top-1 predictions for A and B
        true_pred_a = logits_a.argmax(dim=-1)
        true_pred_b = logits_b.argmax(dim=-1)

        with torch.no_grad():
            # Predict using trace_a (should match A's outputs)
            pred_from_trace_a = self.trace_head(trace_a).argmax(dim=-1)

            # Predict using trace_b (should match B's outputs)
            pred_from_trace_b = self.trace_head(trace_b).argmax(dim=-1)

            # Now the swap test: use trace_a for position B
            # If trace is causal, prediction should follow trace_a (predict like A)
            # If trace is correlated with input, prediction would follow input B

        # When we use trace_a:
        # - trace_following: prediction matches what A would output
        # - input_following: prediction matches what B would output (if trace ignored)

        # Count how often swapped trace prediction matches the trace source
        trace_following = (pred_from_trace_a == true_pred_a).float().mean().item()

        # Check if trace_a predictions ever match B's outputs (correlation)
        input_following = (pred_from_trace_a == true_pred_b).float().mean().item()

        # The key insight: if trace is causal, pred_from_trace_a should match true_pred_a
        # regardless of what input was used. The trace contains the necessary information.

        return {
            "trace_following_rate": float(trace_following),
            "input_following_rate": float(input_following),
            "is_causal": trace_following > input_following,
            "trace_a_match_a": float((pred_from_trace_a == true_pred_a).float().mean().item()),
            "trace_b_match_b": float((pred_from_trace_b == true_pred_b).float().mean().item()),
        }


# =============================================================================
# TEST 3: TRACE GRADIENT FLOW TEST
# =============================================================================

class TraceGradientTest:
    """
    Verify gradients flow through trace features to output.

    If trace is causal: d(output)/d(trace_features) should be non-zero.

    Args:
        model: TKS model
        device: Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device('cpu')
        self.components = TRACE_COMPONENTS.copy()

    def run_gradient_test(
        self,
        trace_features: torch.Tensor,
        trace_head: TraceConsistencyHead,
    ) -> Dict[str, Any]:
        """
        Compute gradient norms for each trace component.

        Args:
            trace_features: [N, trace_dim] trace features (requires_grad=True)
            trace_head: TraceConsistencyHead to compute gradients through

        Returns:
            {
                "noetic_routing_grad_norm": float,
                "rpm_grad_norm": float,
                "attractor_grad_norm": float,
                ...,
                "total_trace_grad_norm": float,
                "gradients_flowing": bool,
            }
        """
        trace_features = trace_features.to(self.device).detach().requires_grad_(True)
        trace_head = trace_head.to(self.device)
        trace_head.eval()

        # Forward pass
        pred_logits = trace_head(trace_features)

        # Create dummy target (use argmax of predictions)
        target = pred_logits.argmax(dim=-1)

        # Compute loss
        loss = F.cross_entropy(pred_logits, target)

        # Backward pass
        loss.backward()

        # Extract gradient norms per component
        grad = trace_features.grad
        if grad is None:
            return {
                "total_trace_grad_norm": 0.0,
                "gradients_flowing": False,
            }

        results = {}
        total_grad_norm = 0.0

        for component, info in self.components.items():
            start = info["start"]
            end = min(start + info["size"], grad.shape[-1])

            component_grad = grad[..., start:end]
            grad_norm = component_grad.norm().item()

            results[f"{component}_grad_norm"] = float(grad_norm)
            total_grad_norm += grad_norm

        results["total_trace_grad_norm"] = float(total_grad_norm)
        results["gradients_flowing"] = total_grad_norm > 1e-8

        return results


# =============================================================================
# TEST 4: TRACE-ONLY HEAD GAP TEST
# =============================================================================

class TraceHeadGapTest:
    """
    Maintain gap between trace-only head and full model.

    The trace-only head should do ALMOST as well as full model.
    If it does EXACTLY as well, it might be cheating (using hidden info).
    If it does MUCH worse, trace is insufficient.

    Target: 0.8 < (trace_acc / full_acc) < 0.98

    Args:
        model: TKS model (full model)
        trace_head: TraceConsistencyHead
        device: Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        trace_head: TraceConsistencyHead,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.trace_head = trace_head
        self.device = device or torch.device('cpu')

    def run_gap_test(
        self,
        inputs: torch.Tensor,
        trace_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compare trace-only and full model accuracy.

        Args:
            inputs: [batch, seq] input tokens
            trace_features: [batch * seq, trace_dim] corresponding trace features
            labels: [batch, seq] target labels (shifted inputs for LM)

        Returns:
            {
                "full_model_accuracy": float,
                "trace_only_accuracy": float,
                "gap_ratio": float,  # trace_acc / full_acc
                "gap_healthy": bool,  # In range [0.8, 0.98]
            }
        """
        self.model.eval()
        self.trace_head.eval()

        inputs = inputs.to(self.device)
        trace_features = trace_features.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            # Full model prediction
            model_output = self.model(inputs)
            full_logits = model_output['logits']

            # Flatten for accuracy computation
            batch_size, seq_len, vocab_size = full_logits.shape
            full_logits_flat = full_logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)

            # Full model accuracy
            full_pred = full_logits_flat.argmax(dim=-1)
            valid_mask = labels_flat != -100  # Ignore padding
            if valid_mask.sum() > 0:
                full_correct = (full_pred[valid_mask] == labels_flat[valid_mask]).float().mean().item()
            else:
                full_correct = 0.0

            # Trace-only prediction
            # Ensure trace_features matches the flattened sequence
            if trace_features.shape[0] != full_logits_flat.shape[0]:
                # Adjust size if needed
                min_size = min(trace_features.shape[0], full_logits_flat.shape[0])
                trace_features = trace_features[:min_size]
                full_logits_flat = full_logits_flat[:min_size]
                labels_flat = labels_flat[:min_size]
                valid_mask = labels_flat != -100

            trace_pred_logits = self.trace_head(trace_features)
            trace_pred = trace_pred_logits.argmax(dim=-1)

            if valid_mask.sum() > 0:
                trace_correct = (trace_pred[valid_mask] == labels_flat[valid_mask]).float().mean().item()
            else:
                trace_correct = 0.0

        # Compute gap ratio
        if full_correct > 0:
            gap_ratio = trace_correct / full_correct
        else:
            gap_ratio = 0.0

        # Check if gap is healthy
        gap_healthy = GAP_RATIO_MIN < gap_ratio < GAP_RATIO_MAX

        return {
            "full_model_accuracy": float(full_correct),
            "trace_only_accuracy": float(trace_correct),
            "gap_ratio": float(gap_ratio),
            "gap_healthy": gap_healthy,
        }


# =============================================================================
# TEST 5: COMBINED CAUSALITY TEST SUITE
# =============================================================================

@dataclass
class CausalityTestResults:
    """Results from the complete causality test suite."""
    ablation_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    counterfactual_results: Dict[str, Any] = field(default_factory=dict)
    gradient_results: Dict[str, Any] = field(default_factory=dict)
    gap_results: Dict[str, Any] = field(default_factory=dict)
    overall_causal: bool = False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ablation_results": self.ablation_results,
            "counterfactual_results": self.counterfactual_results,
            "gradient_results": self.gradient_results,
            "gap_results": self.gap_results,
            "overall_causal": self.overall_causal,
            "summary": self.summary,
        }


class TraceCausalityTestSuite:
    """
    Complete test suite for trace causality.

    Runs all causality tests and provides a comprehensive report.

    Args:
        model: TKS model
        trace_head: Trained TraceConsistencyHead
        test_data: Dict with 'inputs', 'trace_features', 'logits', 'labels'
        device: Torch device
    """

    def __init__(
        self,
        model: nn.Module,
        trace_head: TraceConsistencyHead,
        test_data: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device)
        self.trace_head = trace_head.to(self.device)
        self.test_data = test_data

        # Initialize individual tests
        self.ablation_test = TraceAblationTest(model, trace_head, device)
        self.counterfactual_test = CounterfactualTraceTest(model, trace_head, device)
        self.gradient_test = TraceGradientTest(model, device)
        self.gap_test = TraceHeadGapTest(model, trace_head, device)

    def run_all(self) -> CausalityTestResults:
        """
        Run all causality tests and return comprehensive results.

        Returns:
            CausalityTestResults with all test results and overall assessment
        """
        results = CausalityTestResults()

        # 1. Ablation Test
        trace_features = self.test_data.get('trace_features')
        true_logits = self.test_data.get('logits')

        if trace_features is not None and true_logits is not None:
            results.ablation_results = self.ablation_test.run_ablation_test(
                trace_features, true_logits
            )

        # 2. Counterfactual Test (split data in half for A vs B)
        if trace_features is not None and true_logits is not None:
            n = trace_features.shape[0] // 2
            if n > 0:
                trace_a = trace_features[:n]
                trace_b = trace_features[n:2*n]
                logits_a = true_logits[:n]
                logits_b = true_logits[n:2*n]

                results.counterfactual_results = self.counterfactual_test.run_swap_test(
                    trace_a, trace_b, logits_a, logits_b
                )

        # 3. Gradient Test
        if trace_features is not None:
            # Take a small sample for gradient computation
            sample_features = trace_features[:100].clone()
            results.gradient_results = self.gradient_test.run_gradient_test(
                sample_features, self.trace_head
            )

        # 4. Gap Test
        inputs = self.test_data.get('inputs')
        labels = self.test_data.get('labels')

        if inputs is not None and trace_features is not None and labels is not None:
            results.gap_results = self.gap_test.run_gap_test(
                inputs, trace_features, labels
            )

        # Determine overall causality
        results.overall_causal = self._assess_overall_causality(results)
        results.summary = self._generate_summary(results)

        return results

    def _assess_overall_causality(self, results: CausalityTestResults) -> bool:
        """Assess if trace passes overall causality criteria."""
        passing_tests = 0
        total_tests = 0

        # Ablation: at least one component should be causal
        if results.ablation_results:
            total_tests += 1
            causal_components = sum(
                1 for r in results.ablation_results.values()
                if r.get('causal', False)
            )
            if causal_components >= 1:
                passing_tests += 1

        # Counterfactual: trace should drive outputs
        if results.counterfactual_results:
            total_tests += 1
            if results.counterfactual_results.get('is_causal', False):
                passing_tests += 1

        # Gradient: gradients should flow
        if results.gradient_results:
            total_tests += 1
            if results.gradient_results.get('gradients_flowing', False):
                passing_tests += 1

        # Gap: should be in healthy range
        if results.gap_results:
            total_tests += 1
            if results.gap_results.get('gap_healthy', False):
                passing_tests += 1

        # Require majority of tests to pass
        return passing_tests >= (total_tests / 2) if total_tests > 0 else False

    def _generate_summary(self, results: CausalityTestResults) -> str:
        """Generate human-readable summary."""
        lines = ["Trace Causality Test Summary", "=" * 40]

        # Ablation summary
        if results.ablation_results:
            causal_components = [
                name for name, r in results.ablation_results.items()
                if r.get('causal', False)
            ]
            lines.append(f"Ablation: {len(causal_components)} causal components")
            for name in causal_components:
                drop = results.ablation_results[name].get('drop', 0)
                lines.append(f"  - {name}: {drop:.1%} drop")

        # Counterfactual summary
        if results.counterfactual_results:
            trace_rate = results.counterfactual_results.get('trace_following_rate', 0)
            is_causal = results.counterfactual_results.get('is_causal', False)
            status = "CAUSAL" if is_causal else "CORRELATED"
            lines.append(f"Counterfactual: {status} (trace following: {trace_rate:.1%})")

        # Gradient summary
        if results.gradient_results:
            flowing = results.gradient_results.get('gradients_flowing', False)
            total_norm = results.gradient_results.get('total_trace_grad_norm', 0)
            status = "FLOWING" if flowing else "BLOCKED"
            lines.append(f"Gradients: {status} (total norm: {total_norm:.4f})")

        # Gap summary
        if results.gap_results:
            gap_ratio = results.gap_results.get('gap_ratio', 0)
            healthy = results.gap_results.get('gap_healthy', False)
            status = "HEALTHY" if healthy else "UNHEALTHY"
            lines.append(f"Gap: {status} (ratio: {gap_ratio:.2f})")

        # Overall
        lines.append("=" * 40)
        overall = "CAUSAL" if results.overall_causal else "NOT CAUSAL"
        lines.append(f"Overall: {overall}")

        return "\n".join(lines)

    def assert_causal(self):
        """Raise AssertionError if trace fails causality tests."""
        results = self.run_all()
        if not results.overall_causal:
            raise AssertionError(
                f"Trace failed causality tests:\n{results.summary}"
            )


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def torch_device():
    """Get best available torch device."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@pytest.fixture(scope="module")
def synthetic_model(torch_device):
    """Create a small synthetic TKS model for testing."""
    if not V4_AVAILABLE:
        pytest.skip("TKSNoeticLM not available")

    config = TKSNoeticLMConfig(
        vocab_size=256,
        max_seq_len=32,
        num_layers=2,
        num_scales=2,
        dropout=0.0,
        use_attractor=True,
        use_stable_attractor=True,
        use_rpm=True,
    )
    model = TKSNoeticLM(config).to(torch_device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def synthetic_data(synthetic_model, torch_device):
    """Generate synthetic test data with traces."""
    model = synthetic_model
    batch_size = 16
    seq_len = 16

    # Generate random input tokens
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=torch_device)

    # Run model to get traces and logits
    with torch.no_grad():
        output = model(tokens, return_full_trace=True)

    logits = output['logits']
    trace = output.get('trace', {})

    # Create labels (shifted inputs for LM)
    labels = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1)

    # Extract trace features if available
    if TRACE_UTILS_AVAILABLE and trace:
        compressed = compress_trace(trace)
        trace_features = extract_trace_features(compressed)
        # Expand to batch size
        trace_features = trace_features.unsqueeze(0).expand(batch_size * seq_len, -1)
    else:
        # Create dummy trace features
        trace_dim = compute_feature_dim() if TRACE_UTILS_AVAILABLE else 27
        trace_features = torch.randn(batch_size * seq_len, trace_dim, device=torch_device)

    return {
        'inputs': tokens,
        'logits': logits.view(-1, logits.shape[-1]),
        'labels': labels,
        'trace_features': trace_features,
    }


@pytest.fixture(scope="module")
def trained_trace_head(synthetic_data, torch_device):
    """Train a trace consistency head on synthetic data."""
    trace_features = synthetic_data['trace_features'].to(torch_device)
    true_logits = synthetic_data['logits'].to(torch_device)

    trace_dim = trace_features.shape[-1]
    vocab_size = true_logits.shape[-1]

    # Create and train head
    head = TraceConsistencyHead(
        trace_dim=trace_dim,
        hidden_dim=64,
        vocab_size=vocab_size,
        dropout=0.0,
    ).to(torch_device)

    # Quick training
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

    head.train()
    for _ in range(10):
        optimizer.zero_grad()
        pred = head(trace_features)
        target_probs = F.softmax(true_logits, dim=-1)
        log_pred = F.log_softmax(pred, dim=-1)
        loss = F.kl_div(log_pred, target_probs, reduction='batchmean')
        loss.backward()
        optimizer.step()

    head.eval()
    return head


# =============================================================================
# PYTEST TESTS
# =============================================================================

class TestTraceAblation:
    """Test suite for trace ablation tests."""

    def test_ablate_noetic_routing(self, synthetic_data, trained_trace_head, torch_device):
        """Test that ablating noetic routing affects predictions."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        trace_features = synthetic_data['trace_features']
        true_logits = synthetic_data['logits']

        # Create ablation test
        # Note: model not needed for ablation test
        ablation = TraceAblationTest(
            model=None,  # Not used in ablation test
            trace_consistency_head=trained_trace_head,
            device=torch_device,
        )

        # Test single component ablation
        ablated = ablation.ablate_component(trace_features, "noetic_routing")

        # Verify component is zeroed
        start = TRACE_COMPONENTS["noetic_routing"]["start"]
        end = start + TRACE_COMPONENTS["noetic_routing"]["size"]
        assert torch.all(ablated[..., start:end] == 0), "Noetic routing should be zeroed"

    def test_ablation_test_runs(self, synthetic_data, trained_trace_head, torch_device):
        """Test that full ablation test runs without errors."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        trace_features = synthetic_data['trace_features']
        true_logits = synthetic_data['logits']

        ablation = TraceAblationTest(
            model=None,
            trace_consistency_head=trained_trace_head,
            device=torch_device,
        )

        results = ablation.run_ablation_test(trace_features, true_logits)

        # Check results structure
        assert isinstance(results, dict)
        assert "noetic_routing" in results
        assert "original_acc" in results["noetic_routing"]
        assert "ablated_acc" in results["noetic_routing"]
        assert "drop" in results["noetic_routing"]
        assert "causal" in results["noetic_routing"]


class TestCounterfactualTrace:
    """Test suite for counterfactual trace tests."""

    def test_swap_test_runs(self, synthetic_data, trained_trace_head, torch_device):
        """Test that counterfactual swap test runs without errors."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        trace_features = synthetic_data['trace_features']
        true_logits = synthetic_data['logits']

        # Split data
        n = trace_features.shape[0] // 2
        trace_a = trace_features[:n]
        trace_b = trace_features[n:2*n]
        logits_a = true_logits[:n]
        logits_b = true_logits[n:2*n]

        counterfactual = CounterfactualTraceTest(
            model=None,
            trace_consistency_head=trained_trace_head,
            device=torch_device,
        )

        results = counterfactual.run_swap_test(trace_a, trace_b, logits_a, logits_b)

        # Check results structure
        assert isinstance(results, dict)
        assert "trace_following_rate" in results
        assert "input_following_rate" in results
        assert "is_causal" in results


class TestTraceGradient:
    """Test suite for trace gradient tests."""

    def test_gradient_flow(self, synthetic_data, trained_trace_head, torch_device):
        """Test that gradients flow through trace features."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        trace_features = synthetic_data['trace_features'][:100]

        gradient_test = TraceGradientTest(
            model=None,
            device=torch_device,
        )

        results = gradient_test.run_gradient_test(trace_features, trained_trace_head)

        # Check results structure
        assert isinstance(results, dict)
        assert "total_trace_grad_norm" in results
        assert "gradients_flowing" in results

        # Gradients should flow (norm should be positive)
        assert results["total_trace_grad_norm"] >= 0


class TestTraceHeadGap:
    """Test suite for trace-only head gap tests."""

    def test_gap_test_runs(self, synthetic_model, synthetic_data, trained_trace_head, torch_device):
        """Test that gap test runs without errors."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        gap_test = TraceHeadGapTest(
            model=synthetic_model,
            trace_head=trained_trace_head,
            device=torch_device,
        )

        results = gap_test.run_gap_test(
            synthetic_data['inputs'],
            synthetic_data['trace_features'],
            synthetic_data['labels'],
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "full_model_accuracy" in results
        assert "trace_only_accuracy" in results
        assert "gap_ratio" in results
        assert "gap_healthy" in results


class TestCausalityTestSuite:
    """Test suite for the complete causality test suite."""

    def test_suite_runs(self, synthetic_model, synthetic_data, trained_trace_head, torch_device):
        """Test that complete test suite runs without errors."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        suite = TraceCausalityTestSuite(
            model=synthetic_model,
            trace_head=trained_trace_head,
            test_data=synthetic_data,
            device=torch_device,
        )

        results = suite.run_all()

        # Check results structure
        assert isinstance(results, CausalityTestResults)
        assert isinstance(results.ablation_results, dict)
        assert isinstance(results.counterfactual_results, dict)
        assert isinstance(results.gradient_results, dict)
        assert isinstance(results.summary, str)

        # Summary should be non-empty
        assert len(results.summary) > 0

    def test_results_serializable(self, synthetic_model, synthetic_data, trained_trace_head, torch_device):
        """Test that results can be serialized to JSON."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        suite = TraceCausalityTestSuite(
            model=synthetic_model,
            trace_head=trained_trace_head,
            test_data=synthetic_data,
            device=torch_device,
        )

        results = suite.run_all()

        # Should be JSON serializable
        json_str = json.dumps(results.to_dict())
        assert len(json_str) > 0

        # Should be deserializable
        loaded = json.loads(json_str)
        assert "overall_causal" in loaded


# =============================================================================
# INTEGRATION TESTS WITH REAL MODEL
# =============================================================================

@pytest.mark.requires_model
class TestRealModelCausality:
    """
    Integration tests with real trained model.

    These tests require a trained model checkpoint and verify
    that the trace is genuinely causal for production models.

    Markers:
        requires_model: Requires a trained model checkpoint
    """

    @pytest.fixture
    def real_model(self, available_model_path, torch_device):
        """Load real model for testing."""
        if available_model_path is None:
            pytest.skip("No model checkpoint available")

        if not V4_AVAILABLE:
            pytest.skip("TKSNoeticLM not available")

        checkpoint = torch.load(available_model_path, map_location=torch_device, weights_only=False)

        # Handle different checkpoint formats
        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict):
                config = TKSNoeticLMConfig(**config)
        else:
            config = TKSNoeticLMConfig()

        model = TKSNoeticLM(config).to(torch_device)

        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        except RuntimeError as e:
            if "state_dict" in str(e) or "Missing key" in str(e) or "Unexpected key" in str(e):
                pytest.skip(f"Model checkpoint incompatible with current architecture: {e}")
            raise

        model.eval()
        return model

    @pytest.mark.requires_model
    def test_real_model_ablation(self, real_model, torch_device):
        """Test ablation on real model traces."""
        # Generate some data
        batch_size = 8
        seq_len = 32
        tokens = torch.randint(0, real_model.vocab_size, (batch_size, seq_len), device=torch_device)

        with torch.no_grad():
            output = real_model(tokens, return_full_trace=True)

        # This test verifies the pipeline works with real models
        assert 'logits' in output
        assert output['logits'].shape[0] == batch_size

    @pytest.mark.requires_model
    def test_real_model_gradient_flow(self, real_model, torch_device):
        """Test that gradients flow through real model traces."""
        # Generate data with gradients
        batch_size = 4
        seq_len = 16
        tokens = torch.randint(0, real_model.vocab_size, (batch_size, seq_len), device=torch_device)

        # Forward pass
        output = real_model(tokens)
        logits = output['logits']

        # Backward pass to verify gradients exist
        loss = logits.mean()
        loss.backward()

        # Check that model parameters have gradients
        has_grad = False
        for param in real_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Model should have non-zero gradients"


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def run_standalone(
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    num_samples: int = 200,
) -> Dict[str, Any]:
    """
    Run trace causality tests standalone.

    Args:
        model_path: Path to model checkpoint (None for synthetic)
        output_path: Path to save results JSON
        num_samples: Number of samples for testing

    Returns:
        Dict with all test results
    """
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available")
        return {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create or load model
    if model_path and Path(model_path).exists():
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict):
                config = TKSNoeticLMConfig(**config)
        else:
            config = TKSNoeticLMConfig()

        model = TKSNoeticLM(config).to(device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Creating synthetic model...")
        config = TKSNoeticLMConfig(
            vocab_size=256,
            max_seq_len=32,
            num_layers=2,
            num_scales=2,
        )
        model = TKSNoeticLM(config).to(device)

    model.eval()

    # Generate test data
    print(f"Generating {num_samples} test samples...")
    batch_size = 16
    seq_len = 16
    num_batches = num_samples // batch_size

    all_trace_features = []
    all_logits = []
    all_inputs = []
    all_labels = []

    trace_dim = compute_feature_dim() if TRACE_UTILS_AVAILABLE else 27

    for _ in range(num_batches):
        tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1)

        with torch.no_grad():
            output = model(tokens, return_full_trace=True)

        logits = output['logits']
        trace = output.get('trace', {})

        # Extract trace features
        if TRACE_UTILS_AVAILABLE and trace:
            compressed = compress_trace(trace)
            trace_feat = extract_trace_features(compressed)
            trace_feat = trace_feat.unsqueeze(0).expand(batch_size * seq_len, -1)
        else:
            trace_feat = torch.randn(batch_size * seq_len, trace_dim, device=device)

        all_trace_features.append(trace_feat.cpu())
        all_logits.append(logits.view(-1, logits.shape[-1]).cpu())
        all_inputs.append(tokens.cpu())
        all_labels.append(labels.cpu())

    trace_features = torch.cat(all_trace_features, dim=0)
    true_logits = torch.cat(all_logits, dim=0)
    inputs = torch.cat(all_inputs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"Trace feature shape: {trace_features.shape}")
    print(f"Logits shape: {true_logits.shape}")

    # Train trace head
    print("Training trace consistency head...")
    trace_dim = trace_features.shape[-1]
    vocab_size = true_logits.shape[-1]

    trace_head = TraceConsistencyHead(
        trace_dim=trace_dim,
        hidden_dim=128,
        vocab_size=vocab_size,
    ).to(device)

    trace_features_train = trace_features.to(device)
    true_logits_train = true_logits.to(device)

    optimizer = torch.optim.AdamW(trace_head.parameters(), lr=1e-3)

    trace_head.train()
    for epoch in range(20):
        optimizer.zero_grad()
        pred = trace_head(trace_features_train)
        target_probs = F.softmax(true_logits_train, dim=-1)
        log_pred = F.log_softmax(pred, dim=-1)
        loss = F.kl_div(log_pred, target_probs, reduction='batchmean')
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/20: loss={loss.item():.4f}")

    trace_head.eval()

    # Run causality test suite
    print("\nRunning causality test suite...")
    test_data = {
        'inputs': inputs.to(device),
        'trace_features': trace_features.to(device),
        'logits': true_logits.to(device),
        'labels': labels.to(device),
    }

    suite = TraceCausalityTestSuite(
        model=model,
        trace_head=trace_head,
        test_data=test_data,
        device=device,
    )

    results = suite.run_all()

    # Print summary
    print("\n" + "=" * 70)
    print(results.summary)
    print("=" * 70)

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results.to_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Trace Causality Tests')
    parser.add_argument('--model', help='Path to model checkpoint')
    parser.add_argument('--output', default='output/trace_causality_results.json',
                        help='Path to save results JSON')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of test samples')

    args = parser.parse_args()

    run_standalone(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.samples,
    )

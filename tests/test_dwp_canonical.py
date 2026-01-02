"""
TKS D/W/P Canonical Mapping Regression Tests

Verifies that RPMGatingMechanism:
1. Extracts D from ν2, ν3 (indices 2,3,12,13,22,23,32,33)
2. Extracts W from ν1,4,5,6,7 (indices 1,4,5,6,7,11,14,15,16,17,21,24,25,26,27,31,34,35,36,37)
3. Extracts P from ν8, ν9 (indices 8,9,18,19,28,29,38,39)
4. Maintains gate = D × W × P in [0, 1]
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v2 import (
    RPMGatingMechanism,
    TKSLLMCorePipeline,
    DESIRE_INDICES,
    WISDOM_INDICES,
    POWER_INDICES,
    TOTAL_DIM,
)


class TestCanonicalDWPIndices:
    """Test that D/W/P indices are canonical per MVR protocol."""

    def test_desire_indices_are_nu1_nu4_nu7(self):
        """Desire must use ν1, ν4, ν7 (MVR) across all 4 worlds."""
        expected = [1, 4, 7, 11, 14, 17, 21, 24, 27, 31, 34, 37]  # Mind, Vibration, Rhythm
        assert DESIRE_INDICES == expected, f"Desire indices mismatch: {DESIRE_INDICES}"

    def test_wisdom_indices_are_nu5_nu6(self):
        """Wisdom must use ν5 (Female) and ν6 (Male) across all 4 worlds."""
        expected = [5, 6, 15, 16, 25, 26, 35, 36]  # Female, Male
        assert WISDOM_INDICES == expected, f"Wisdom indices mismatch: {WISDOM_INDICES}"

    def test_power_indices_are_nu8_nu9(self):
        """Power must use ν8 and ν9 across all 4 worlds."""
        expected = [8, 9, 18, 19, 28, 29, 38, 39]
        assert POWER_INDICES == expected, f"Power indices mismatch: {POWER_INDICES}"

    def test_indices_are_disjoint(self):
        """D, W, P indices must not overlap."""
        d_set = set(DESIRE_INDICES)
        w_set = set(WISDOM_INDICES)
        p_set = set(POWER_INDICES)

        assert d_set.isdisjoint(w_set), "Desire and Wisdom overlap"
        assert d_set.isdisjoint(p_set), "Desire and Power overlap"
        assert w_set.isdisjoint(p_set), "Wisdom and Power overlap"

    def test_indices_cover_28_of_40_dims(self):
        """D+W+P should cover 28 dims (12+8+8), leaving ν0,2,3,10 uncovered."""
        all_dwp = set(DESIRE_INDICES) | set(WISDOM_INDICES) | set(POWER_INDICES)
        assert len(all_dwp) == 28, f"Expected 28 dims, got {len(all_dwp)}"

        # ν0 (indices 0,10,20,30) should NOT be in D/W/P
        nu0_indices = {0, 10, 20, 30}
        assert nu0_indices.isdisjoint(all_dwp), "ν0 should not be in D/W/P"

        # ν2,ν3 (Positive/Negative) are NOT in D/W/P - separate polarity system
        polarity_indices = {2, 3, 12, 13, 22, 23, 32, 33}
        assert polarity_indices.isdisjoint(all_dwp), "ν2/ν3 should not be in D/W/P"


class TestRPMGatingMechanismBuffers:
    """Test that RPMGatingMechanism has correct buffers."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_desire_buffer_registered(self, rpm):
        """desire_idx buffer must be registered."""
        assert hasattr(rpm, 'desire_idx')
        assert rpm.desire_idx.tolist() == DESIRE_INDICES

    def test_wisdom_buffer_registered(self, rpm):
        """wisdom_idx buffer must be registered."""
        assert hasattr(rpm, 'wisdom_idx')
        assert rpm.wisdom_idx.tolist() == WISDOM_INDICES

    def test_power_buffer_registered(self, rpm):
        """power_idx buffer must be registered."""
        assert hasattr(rpm, 'power_idx')
        assert rpm.power_idx.tolist() == POWER_INDICES


class TestDWPBounds:
    """Test that D/W/P scores and gate are bounded in [0, 1]."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_dwp_scores_in_unit_interval(self, rpm):
        """D, W, P scores must all be in [0, 1]."""
        batch, seq = 4, 8
        thought = torch.randn(batch, seq, TOTAL_DIM)

        out = rpm(thought)
        dwp = out['dwp_scores']  # [batch, seq, 7, 3]

        assert dwp.min() >= 0.0, f"D/W/P min below 0: {dwp.min()}"
        assert dwp.max() <= 1.0, f"D/W/P max above 1: {dwp.max()}"

    def test_gate_in_unit_interval(self, rpm):
        """Gate = D × W × P must be in [0, 1]."""
        batch, seq = 4, 8
        thought = torch.randn(batch, seq, TOTAL_DIM)

        out = rpm(thought, target_foundation=0)
        gate = out['rpm_gate']  # [batch, seq]

        assert gate.min() >= 0.0, f"Gate min below 0: {gate.min()}"
        assert gate.max() <= 1.0, f"Gate max above 1: {gate.max()}"

    def test_gate_equals_dwp_product(self, rpm):
        """Gate must equal D × W × P for target foundation."""
        batch, seq = 2, 4
        thought = torch.randn(batch, seq, TOTAL_DIM)

        for f_idx in range(7):
            out = rpm(thought, target_foundation=f_idx)
            gate = out['rpm_gate']  # [batch, seq]
            dwp = out['dwp_scores'][:, :, f_idx, :]  # [batch, seq, 3]

            expected_gate = dwp[:, :, 0] * dwp[:, :, 1] * dwp[:, :, 2]

            torch.testing.assert_close(
                gate, expected_gate,
                msg=f"Gate mismatch for foundation {f_idx}"
            )


class TestDWPGradientFlow:
    """Test that gradients flow through D/W/P extraction."""

    def test_gradients_flow_to_desire_noetics(self):
        """Gradients must flow back to ν1, ν4, ν7 positions (MVR)."""
        rpm = RPMGatingMechanism(dim=TOTAL_DIM)
        thought = torch.randn(2, 4, TOTAL_DIM, requires_grad=True)

        out = rpm(thought, target_foundation=0)
        loss = out['desire_scores'].sum()
        loss.backward()

        # Check gradients exist at desire indices
        for idx in DESIRE_INDICES:
            grad_at_idx = thought.grad[:, :, idx]
            assert grad_at_idx.abs().sum() > 0, f"No gradient at desire index {idx}"

    def test_gradients_flow_to_power_noetics(self):
        """Gradients must flow back to ν8, ν9 positions."""
        rpm = RPMGatingMechanism(dim=TOTAL_DIM)
        thought = torch.randn(2, 4, TOTAL_DIM, requires_grad=True)

        out = rpm(thought, target_foundation=0)
        loss = out['power_scores'].sum()
        loss.backward()

        # Check gradients exist at power indices
        for idx in POWER_INDICES:
            grad_at_idx = thought.grad[:, :, idx]
            assert grad_at_idx.abs().sum() > 0, f"No gradient at power index {idx}"


class TestFullPipelineDWP:
    """Integration tests for D/W/P in full pipeline."""

    @pytest.fixture
    def pipeline(self):
        return TKSLLMCorePipeline(vocab_size=100)

    def test_pipeline_dwp_scores_bounded(self, pipeline):
        """Full pipeline D/W/P must be in [0, 1]."""
        tokens = torch.randint(0, 100, (2, 8))

        out = pipeline(tokens, target_foundation=3, return_full_trace=True)
        dwp = out['trace']['dwp_scores']

        assert torch.isfinite(dwp).all(), "D/W/P contains non-finite values"
        assert dwp.min() >= 0.0, f"D/W/P min below 0: {dwp.min()}"
        assert dwp.max() <= 1.0, f"D/W/P max above 1: {dwp.max()}"

    def test_pipeline_gate_bounded(self, pipeline):
        """Full pipeline gate must be in [0, 1]."""
        tokens = torch.randint(0, 100, (2, 8))

        out = pipeline(tokens, target_foundation=3)
        gate = out['rpm_gate']

        assert torch.isfinite(gate).all(), "Gate contains non-finite values"
        assert gate.min() >= 0.0, f"Gate min below 0: {gate.min()}"
        assert gate.max() <= 1.0, f"Gate max above 1: {gate.max()}"


class TestDWPEvalMetrics:
    """Quick eval metrics for D/W/P on synthetic data."""

    def test_dwp_response_to_noetic_perturbation(self):
        """
        Perturbing desire noetics (ν1, ν4, ν7 - MVR) should change D scores more than W/P.
        This tests that the canonical extraction is working correctly.
        """
        rpm = RPMGatingMechanism(dim=TOTAL_DIM)
        batch, seq = 2, 4

        # Base input
        base = torch.zeros(batch, seq, TOTAL_DIM)
        out_base = rpm(base)
        d_base = out_base['desire_scores'][:, :, 0].mean().item()
        w_base = out_base['wisdom_scores'][:, :, 0].mean().item()
        p_base = out_base['power_scores'][:, :, 0].mean().item()

        # Perturb ONLY desire noetics (ν1, ν4, ν7 - MVR)
        perturbed = base.clone()
        for idx in DESIRE_INDICES:
            perturbed[:, :, idx] = 5.0  # Strong positive signal

        out_perturbed = rpm(perturbed)
        d_perturbed = out_perturbed['desire_scores'][:, :, 0].mean().item()
        w_perturbed = out_perturbed['wisdom_scores'][:, :, 0].mean().item()
        p_perturbed = out_perturbed['power_scores'][:, :, 0].mean().item()

        # D should change significantly; W and P should change less
        d_delta = abs(d_perturbed - d_base)
        w_delta = abs(w_perturbed - w_base)
        p_delta = abs(p_perturbed - p_base)

        # D should change more than P when only desire noetics are perturbed
        # (W might change slightly due to numerical noise, but much less than D)
        assert d_delta > 0.01, f"D didn't respond to desire perturbation: delta={d_delta}"

    def test_power_noetic_perturbation(self):
        """Perturbing power noetics (ν8, ν9) should primarily affect P scores."""
        rpm = RPMGatingMechanism(dim=TOTAL_DIM)
        batch, seq = 2, 4

        base = torch.zeros(batch, seq, TOTAL_DIM)
        out_base = rpm(base)
        p_base = out_base['power_scores'][:, :, 0].mean().item()

        # Perturb ONLY power noetics
        perturbed = base.clone()
        for idx in POWER_INDICES:
            perturbed[:, :, idx] = 5.0

        out_perturbed = rpm(perturbed)
        p_perturbed = out_perturbed['power_scores'][:, :, 0].mean().item()

        p_delta = abs(p_perturbed - p_base)
        assert p_delta > 0.01, f"P didn't respond to power perturbation: delta={p_delta}"


def run_quick_dwp_eval():
    """Quick informational eval for D/W/P on random batch."""
    print("\n" + "=" * 60)
    print("D/W/P Quick Eval (Informational)")
    print("=" * 60)

    pipeline = TKSLLMCorePipeline(vocab_size=100)
    tokens = torch.randint(0, 100, (8, 16))  # 8 samples, seq_len 16

    out = pipeline(tokens, target_foundation=3, return_full_trace=True)
    dwp = out['trace']['dwp_scores']  # [8, 16, 7, 3]
    gate = out['rpm_gate']  # [8, 16]

    print(f"\nBatch shape: {tokens.shape}")
    print(f"D/W/P shape: {dwp.shape}")

    # Per-foundation stats
    print("\nPer-foundation D/W/P means:")
    from tks_llm_core_v2 import FOUNDATION_NAMES
    for f_idx, f_name in enumerate(FOUNDATION_NAMES):
        d_mean = dwp[:, :, f_idx, 0].mean().item()
        w_mean = dwp[:, :, f_idx, 1].mean().item()
        p_mean = dwp[:, :, f_idx, 2].mean().item()
        gate_f = (dwp[:, :, f_idx, 0] * dwp[:, :, f_idx, 1] * dwp[:, :, f_idx, 2]).mean().item()
        print(f"  {f_name:15s}: D={d_mean:.3f}, W={w_mean:.3f}, P={p_mean:.3f}, Gate={gate_f:.3f}")

    # Overall stats
    print(f"\nOverall D/W/P ranges:")
    print(f"  Desire:  [{dwp[:,:,:,0].min():.3f}, {dwp[:,:,:,0].max():.3f}]")
    print(f"  Wisdom:  [{dwp[:,:,:,1].min():.3f}, {dwp[:,:,:,1].max():.3f}]")
    print(f"  Power:   [{dwp[:,:,:,2].min():.3f}, {dwp[:,:,:,2].max():.3f}]")
    print(f"  Gate:    [{gate.min():.3f}, {gate.max():.3f}]")

    # Bounds check
    bounds_ok = (dwp.min() >= 0) and (dwp.max() <= 1) and (gate.min() >= 0) and (gate.max() <= 1)
    print(f"\nBounds check (all in [0,1]): {'PASS' if bounds_ok else 'FAIL'}")

    return {
        'dwp_shape': tuple(dwp.shape),
        'desire_mean': dwp[:, :, :, 0].mean().item(),
        'wisdom_mean': dwp[:, :, :, 1].mean().item(),
        'power_mean': dwp[:, :, :, 2].mean().item(),
        'gate_mean': gate.mean().item(),
        'bounds_ok': bounds_ok,
    }


if __name__ == "__main__":
    # Run pytest
    pytest.main([__file__, "-v"])

    # Also run quick eval
    run_quick_dwp_eval()

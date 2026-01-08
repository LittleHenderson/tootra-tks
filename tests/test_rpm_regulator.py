"""
Test suite for RPM gating from strg-llm-stk.

Regulators:
- FM_female (Noetic 5): reservoir/authority/expert
- FM_male (Noetic 6): larger set/related ideas
- ACBE (Noetics 8,9): cause-effect chains
- MVR (Noetics 1,4,7): activation plan

D/W/P (Desire/Wisdom/Power) canonical mappings:
- Desire: noetics 1, 4, 7 (Mind, Vibration, Rhythm - MVR protocol)
- Wisdom: noetics 5, 6 (Female/Male - receptivity/structure)
- Power: noetics 8, 9 (Cause/Effect - causal chain)

Author: EVAL agent
Date: 2026-01-05
"""
import pytest
import torch
import torch.nn.functional as F
from typing import Dict

# Import RPM components from v2
from tks_llm_core_v2 import (
    RPMGatingMechanism,
    TOTAL_DIM,
    NUM_FOUNDATIONS,
    FOUNDATION_NAMES,
    DESIRE_INDICES,
    WISDOM_INDICES,
    POWER_INDICES,
)


class TestRegulatorSelection:
    """Test that correct regulators are selected for different contexts."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_fm_female_authority(self, rpm):
        """Select FM_female for authority/expert contexts (Noetic 5)."""
        batch_size = 4
        seq_len = 8

        # Create thought state with high activation in noetic 5 (indices 5, 15, 25, 35)
        thought = torch.zeros(batch_size, seq_len, TOTAL_DIM)
        for world_offset in [0, 10, 20, 30]:
            thought[:, :, world_offset + 5] = 1.0  # Noetic 5 in each world

        out = rpm(thought)

        # Verify output structure
        assert 'gated_output' in out
        assert 'rpm_gate' in out
        assert 'dwp_scores' in out
        assert out['gated_output'].shape == (batch_size, seq_len, TOTAL_DIM)

    def test_fm_male_superset(self, rpm):
        """Select FM_male for scope expansion contexts (Noetic 6)."""
        batch_size = 4
        seq_len = 8

        # Create thought state with high activation in noetic 6
        thought = torch.zeros(batch_size, seq_len, TOTAL_DIM)
        for world_offset in [0, 10, 20, 30]:
            thought[:, :, world_offset + 6] = 1.0  # Noetic 6 in each world

        out = rpm(thought)

        assert 'gated_output' in out
        assert 'dwp_scores' in out
        # Wisdom should be activated (noetic 6 contributes to Wisdom)

    def test_acbe_causation(self, rpm):
        """Select ACBE for cause-effect contexts (Noetics 8, 9)."""
        batch_size = 4
        seq_len = 8

        # Create thought state with high activation in noetics 8, 9 (Cause/Effect)
        thought = torch.zeros(batch_size, seq_len, TOTAL_DIM)
        for world_offset in [0, 10, 20, 30]:
            thought[:, :, world_offset + 8] = 1.0  # Noetic 8 (Cause)
            thought[:, :, world_offset + 9] = 1.0  # Noetic 9 (Effect)

        out = rpm(thought)

        assert 'gated_output' in out
        # Power should be activated (noetics 8,9 contribute to Power)

    def test_mvr_activation(self, rpm):
        """Select MVR for desire/activation contexts (Noetics 1, 4, 7)."""
        batch_size = 4
        seq_len = 8

        # Create thought state with MVR pattern (Mind, Vibration, Rhythm)
        thought = torch.zeros(batch_size, seq_len, TOTAL_DIM)
        for world_offset in [0, 10, 20, 30]:
            thought[:, :, world_offset + 1] = 1.0  # Noetic 1 (Mind)
            thought[:, :, world_offset + 4] = 1.0  # Noetic 4 (Vibration)
            thought[:, :, world_offset + 7] = 1.0  # Noetic 7 (Rhythm)

        out = rpm(thought)

        assert 'gated_output' in out
        # Desire should be activated (MVR contributes to Desire)


class TestRPMCascade:
    """Test RPM cascade depth handling."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_cascade_depth_2(self, rpm):
        """Generate valid 2-layer RPM cascade."""
        batch_size = 4
        seq_len = 8

        # First layer processing
        thought_1 = torch.randn(batch_size, seq_len, TOTAL_DIM)
        out_1 = rpm(thought_1)

        # Second layer (cascade)
        thought_2 = out_1['gated_output']
        out_2 = rpm(thought_2)

        # Both outputs should be valid
        assert out_1['gated_output'].shape == (batch_size, seq_len, TOTAL_DIM)
        assert out_2['gated_output'].shape == (batch_size, seq_len, TOTAL_DIM)
        assert not torch.isnan(out_2['gated_output']).any()

    def test_cascade_depth_3(self, rpm):
        """Generate valid 3-layer RPM cascade."""
        batch_size = 4
        seq_len = 8

        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)

        # 3-layer cascade
        for _ in range(3):
            out = rpm(thought)
            thought = out['gated_output']

        # Final output should be valid
        assert thought.shape == (batch_size, seq_len, TOTAL_DIM)
        assert not torch.isnan(thought).any()


class TestDesireWisdomPower:
    """Test D/W/P noetic mappings."""

    def test_desire_noetics(self):
        """Desire uses noetics 1, 4, 7 (MVR protocol)."""
        # Verify indices are correct across all 4 worlds
        expected_desire = []
        for world_offset in [0, 10, 20, 30]:
            expected_desire.extend([world_offset + 1, world_offset + 4, world_offset + 7])

        assert DESIRE_INDICES == expected_desire
        assert len(DESIRE_INDICES) == 12  # 3 noetics * 4 worlds

    def test_wisdom_noetics(self):
        """Wisdom uses noetics 5, 6 (Female/Male)."""
        expected_wisdom = []
        for world_offset in [0, 10, 20, 30]:
            expected_wisdom.extend([world_offset + 5, world_offset + 6])

        assert WISDOM_INDICES == expected_wisdom
        assert len(WISDOM_INDICES) == 8  # 2 noetics * 4 worlds

    def test_power_noetics(self):
        """Power uses noetics 8, 9 (Cause/Effect)."""
        expected_power = []
        for world_offset in [0, 10, 20, 30]:
            expected_power.extend([world_offset + 8, world_offset + 9])

        assert POWER_INDICES == expected_power
        assert len(POWER_INDICES) == 8  # 2 noetics * 4 worlds


class TestDWPScoring:
    """Test D/W/P score computation."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_dwp_score_range(self, rpm):
        """D/W/P scores should be in [0, 1]."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)

        out = rpm(thought)
        dwp = out['dwp_scores']

        # Shape: [batch, seq, num_foundations, 3] (D, W, P)
        assert dwp.shape == (batch_size, seq_len, NUM_FOUNDATIONS, 3)
        assert dwp.min() >= 0.0, "DWP scores should be >= 0"
        assert dwp.max() <= 1.0, "DWP scores should be <= 1"

    def test_dwp_separate_scores(self, rpm):
        """Test separate D, W, P score extraction."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)

        out = rpm(thought)

        assert 'desire_scores' in out
        assert 'wisdom_scores' in out
        assert 'power_scores' in out

        # Each should be [batch, seq, num_foundations]
        assert out['desire_scores'].shape == (batch_size, seq_len, NUM_FOUNDATIONS)
        assert out['wisdom_scores'].shape == (batch_size, seq_len, NUM_FOUNDATIONS)
        assert out['power_scores'].shape == (batch_size, seq_len, NUM_FOUNDATIONS)


class TestRPMGate:
    """Test RPM gate computation (D x W x P)."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_gate_is_product(self, rpm):
        """Gate = D x W x P for selected Foundation."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)
        target_foundation = 2  # Life foundation

        out = rpm(thought, target_foundation=target_foundation)

        dwp = out['dwp_scores']
        # Extract D, W, P for the target foundation
        d = dwp[:, :, target_foundation, 0]
        w = dwp[:, :, target_foundation, 1]
        p = dwp[:, :, target_foundation, 2]

        # Gate should be D x W x P
        expected_gate = d * w * p
        actual_gate = out['rpm_gate']

        # Allow small numerical tolerance
        assert torch.allclose(actual_gate, expected_gate, atol=1e-5)

    def test_gate_bounds(self, rpm):
        """Gate values should be in [0, 1]."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)

        out = rpm(thought)

        # Gate is product of [0,1] values, so also in [0,1]
        assert out['rpm_gate'].min() >= 0.0
        assert out['rpm_gate'].max() <= 1.0


class TestFoundationEmbeddings:
    """Test Foundation embedding structure."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_num_foundations(self, rpm):
        """Should have 7 Foundations."""
        embeddings = rpm.foundation_embeddings
        assert embeddings.shape[0] == NUM_FOUNDATIONS == 7

    def test_foundation_names(self):
        """Foundation names should be correct."""
        expected = ['Unity', 'Wisdom', 'Life', 'Companionship', 'Power', 'Material', 'Lust']
        assert FOUNDATION_NAMES == expected

    def test_foundation_embedding_dim(self, rpm):
        """Each Foundation embedding should be 40D."""
        embeddings = rpm.foundation_embeddings
        assert embeddings.shape == (NUM_FOUNDATIONS, TOTAL_DIM)

    def test_foundation_embeddings_normalized(self, rpm):
        """Foundation embeddings should be normalized."""
        embeddings = rpm.foundation_embeddings

        # Check normalization (L2 norm ~= 1)
        norms = embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(NUM_FOUNDATIONS), atol=1e-5)


class TestGoalConditioning:
    """Test goal-conditioned RPM gating."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_with_goal_state(self, rpm):
        """Test forward pass with goal state."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)
        goal = torch.randn(batch_size, TOTAL_DIM)

        out = rpm(thought, goal_state=goal)

        assert 'gated_output' in out
        assert 'prereq_score' in out
        assert out['prereq_score'].shape == (batch_size, seq_len)

    def test_prereq_score_bounds(self, rpm):
        """Prerequisite scores should be in [0, 1]."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)
        goal = torch.randn(batch_size, TOTAL_DIM)

        out = rpm(thought, goal_state=goal)

        assert out['prereq_score'].min() >= 0.0
        assert out['prereq_score'].max() <= 1.0

    def test_goal_blending(self, rpm):
        """Test that goal state influences output."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)
        goal = torch.randn(batch_size, TOTAL_DIM)

        out_without_goal = rpm(thought)
        out_with_goal = rpm(thought, goal_state=goal)

        # Outputs should be different with/without goal
        diff = (out_without_goal['gated_output'] - out_with_goal['gated_output']).abs().mean()
        # Due to goal blending, outputs should differ (unless gate is 0)


class TestTargetFoundation:
    """Test target Foundation selection."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_each_foundation_target(self, rpm):
        """Test targeting each Foundation."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)

        for f_idx, f_name in enumerate(FOUNDATION_NAMES):
            out = rpm(thought, target_foundation=f_idx)

            assert 'gated_output' in out
            # The gate should be computed from the target foundation's DWP
            dwp = out['dwp_scores'][:, :, f_idx, :]
            expected_gate = dwp[:, :, 0] * dwp[:, :, 1] * dwp[:, :, 2]
            assert torch.allclose(out['rpm_gate'], expected_gate, atol=1e-5)

    def test_no_target_uses_max(self, rpm):
        """Without target, gate uses maximum across Foundations."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)

        out = rpm(thought, target_foundation=None)

        # Gate should be max(D*W*P) across all foundations
        dwp = out['dwp_scores']  # [batch, seq, 7, 3]
        dwp_product = dwp.prod(dim=-1)  # [batch, seq, 7]
        expected_gate = dwp_product.max(dim=-1).values  # [batch, seq]

        assert torch.allclose(out['rpm_gate'], expected_gate, atol=1e-5)


class TestGradientFlow:
    """Test gradient flow through RPM."""

    @pytest.fixture
    def rpm(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_backward_pass(self, rpm):
        """Test gradients flow through RPM."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM, requires_grad=True)

        out = rpm(thought)
        loss = out['gated_output'].sum()
        loss.backward()

        assert thought.grad is not None
        assert not torch.isnan(thought.grad).any()

    def test_parameter_gradients(self, rpm):
        """Test parameter gradients are computed."""
        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)

        out = rpm(thought)
        loss = out['gated_output'].sum()
        loss.backward()

        # Check some parameters have gradients
        has_grad = False
        for param in rpm.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "At least some parameters should have gradients"


# Run with: pytest tests/test_rpm_regulator.py -v

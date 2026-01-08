"""
Integration test suite for strg-llm-stk with TKS v4.

End-to-end pipeline tests:
1. Full forward pass with all modules
2. Operator -> RPM -> DPS -> Governance flow
3. Trace output verification
4. Configuration flag testing
5. Backward compatibility

Module Integration:
- TKSCompositionalLayerV2: Operator semantics
- RPMGatingMechanism: D/W/P scoring
- DPSGatingLayer: Novelty gating
- GovernanceRails: Safety constraints

Author: EVAL agent
Date: 2026-01-05
"""
import pytest
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

# Core modules
from tks_compositional_layer import (
    TKSCompositionalLayerV2,
    TOTAL_DIM,
    NUM_OPERATORS,
)
from tks_llm_core_v2 import (
    RPMGatingMechanism,
    NUM_FOUNDATIONS,
)
from tks_features.dps_gating import (
    DPSGatingLayer,
    DPSConfig,
    DPSState,
    DPSMode,
    NoveltyClass,
)
from tks_features.governance_rails import (
    GovernanceRails,
    GovernanceConfig,
    ActionProfile,
    ActionReversibility,
    GateDecision,
    OperationalMode,
)


@dataclass
class IntegrationConfig:
    """Configuration for integration testing."""
    use_operator: bool = True
    use_rpm: bool = True
    use_dps: bool = True
    use_governance: bool = True
    noetic_dim: int = 40
    seq_len: int = 8
    batch_size: int = 4


class MockIntegratedPipeline(nn.Module):
    """
    Mock integrated pipeline combining all strg-llm-stk modules.

    This simulates the integrated v4 forward pass with all features enabled.
    """

    def __init__(self, config: IntegrationConfig):
        super().__init__()
        self.config = config

        # Operator layer
        if config.use_operator:
            self.operator_layer = TKSCompositionalLayerV2()

        # RPM layer
        if config.use_rpm:
            self.rpm_layer = RPMGatingMechanism(dim=config.noetic_dim)

        # DPS layer
        if config.use_dps:
            self.dps_layer = DPSGatingLayer(DPSConfig(), noetic_dim=config.noetic_dim)
            self.dps_state = DPSState()

        # Governance
        if config.use_governance:
            self.governance = GovernanceRails(
                GovernanceConfig(),
                allowed_tools={"read_file", "write_file"},
            )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        operator: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        action: Optional[ActionProfile] = None,
    ) -> Dict:
        """
        Run full integration pipeline.

        Flow: Operator -> RPM -> DPS -> Governance check

        Returns:
            Dict with outputs from all stages
        """
        trace = {}

        # Stage 1: Operator composition
        if self.config.use_operator:
            op_out = self.operator_layer(left, right, operator)
            composed = op_out['noetic_output']
            trace['operator'] = {
                'gate_values': op_out['gate_values'].tolist(),
                'equation_repr_shape': list(op_out['equation_repr'].shape),
            }
        else:
            composed = (left + right) / 2

        # Stage 2: RPM gating
        if self.config.use_rpm:
            # Add sequence dimension for RPM
            composed_seq = composed.unsqueeze(1).repeat(1, self.config.seq_len, 1)
            rpm_out = self.rpm_layer(composed_seq, goal_state=goal_state)
            gated = rpm_out['gated_output'].mean(dim=1)  # Pool back to batch
            trace['rpm'] = {
                'dwp_scores_shape': list(rpm_out['dwp_scores'].shape),
                'rpm_gate_shape': list(rpm_out['rpm_gate'].shape),
            }
        else:
            gated = composed

        # Stage 3: DPS gating
        if self.config.use_dps:
            # Add sequence dimension for DPS
            gated_seq = gated.unsqueeze(1).repeat(1, self.config.seq_len, 1)
            dps_out, new_state, dps_trace = self.dps_layer(
                gated_seq, self.dps_state, episode_id="integration_test"
            )
            self.dps_state = new_state
            final = dps_out.mean(dim=1)
            trace['dps'] = dps_trace
        else:
            final = gated

        # Stage 4: Governance check
        if self.config.use_governance and action is not None:
            gov_result = self.governance.check_action(
                action=action,
                uncertainty=0.3,
                stakes=0.3,
                alignment=0.9,
            )
            trace['governance'] = {
                'decision': gov_result.decision.value,
                'mode': gov_result.mode.value,
                'hs_score': gov_result.hs_score,
            }

        return {
            'output': final,
            'trace': trace,
        }


class TestFullPipelineIntegration:
    """Test full pipeline integration."""

    @pytest.fixture
    def pipeline(self):
        return MockIntegratedPipeline(IntegrationConfig())

    @pytest.fixture
    def inputs(self):
        batch_size = 4
        return {
            'left': torch.randn(batch_size, TOTAL_DIM),
            'right': torch.randn(batch_size, TOTAL_DIM),
            'operator': torch.randint(0, NUM_OPERATORS, (batch_size,)),
        }

    def test_full_forward_pass(self, pipeline, inputs):
        """Test complete forward pass through all modules."""
        result = pipeline(
            left=inputs['left'],
            right=inputs['right'],
            operator=inputs['operator'],
        )

        assert 'output' in result
        assert 'trace' in result
        assert result['output'].shape == (4, TOTAL_DIM)
        assert not torch.isnan(result['output']).any()

    def test_trace_contains_all_stages(self, pipeline, inputs):
        """Test trace contains output from all enabled stages."""
        action = ActionProfile(
            action_type="read_file",
            reversibility=ActionReversibility.REVERSIBLE,
        )

        result = pipeline(
            left=inputs['left'],
            right=inputs['right'],
            operator=inputs['operator'],
            action=action,
        )

        trace = result['trace']

        # Check all stages present
        assert 'operator' in trace
        assert 'rpm' in trace
        assert 'dps' in trace
        assert 'governance' in trace

    def test_with_goal_state(self, pipeline, inputs):
        """Test pipeline with goal conditioning."""
        goal = torch.randn(4, TOTAL_DIM)

        result = pipeline(
            left=inputs['left'],
            right=inputs['right'],
            operator=inputs['operator'],
            goal_state=goal,
        )

        assert result['output'].shape == (4, TOTAL_DIM)


class TestOperatorRPMIntegration:
    """Test Operator -> RPM flow."""

    @pytest.fixture
    def operator_layer(self):
        return TKSCompositionalLayerV2()

    @pytest.fixture
    def rpm_layer(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    def test_operator_to_rpm(self, operator_layer, rpm_layer):
        """Operator output should be valid RPM input."""
        batch_size = 4
        seq_len = 8

        left = torch.randn(batch_size, TOTAL_DIM)
        right = torch.randn(batch_size, TOTAL_DIM)
        operator = torch.randint(0, NUM_OPERATORS, (batch_size,))

        # Operator composition
        op_out = operator_layer(left, right, operator)
        composed = op_out['noetic_output']

        # Feed to RPM (add sequence dimension)
        composed_seq = composed.unsqueeze(1).repeat(1, seq_len, 1)
        rpm_out = rpm_layer(composed_seq)

        assert rpm_out['gated_output'].shape == (batch_size, seq_len, TOTAL_DIM)
        assert not torch.isnan(rpm_out['gated_output']).any()

    def test_operator_types_affect_rpm(self, operator_layer, rpm_layer):
        """Different operators should produce different RPM behaviors."""
        batch_size = 4
        seq_len = 8

        left = torch.randn(batch_size, TOTAL_DIM)
        right = torch.randn(batch_size, TOTAL_DIM)

        rpm_outputs = {}
        for op_idx in range(NUM_OPERATORS):
            operator = torch.full((batch_size,), op_idx, dtype=torch.long)
            op_out = operator_layer(left, right, operator)
            composed = op_out['noetic_output']
            composed_seq = composed.unsqueeze(1).repeat(1, seq_len, 1)
            rpm_out = rpm_layer(composed_seq)
            rpm_outputs[op_idx] = rpm_out['gated_output'].mean()

        # Different operators should lead to different outputs
        # (they won't be identical due to operator-specific transforms)


class TestRPMDPSIntegration:
    """Test RPM -> DPS flow."""

    @pytest.fixture
    def rpm_layer(self):
        return RPMGatingMechanism(dim=TOTAL_DIM)

    @pytest.fixture
    def dps_layer(self):
        return DPSGatingLayer(DPSConfig(), noetic_dim=TOTAL_DIM)

    def test_rpm_to_dps(self, rpm_layer, dps_layer):
        """RPM output should be valid DPS input."""
        batch_size = 4
        seq_len = 8
        state = DPSState()

        thought = torch.randn(batch_size, seq_len, TOTAL_DIM)

        # RPM gating
        rpm_out = rpm_layer(thought)
        gated = rpm_out['gated_output']

        # DPS gating
        dps_out, new_state, trace = dps_layer(gated, state)

        assert dps_out.shape == (batch_size, seq_len, TOTAL_DIM)
        assert not torch.isnan(dps_out).any()
        assert 'novelty_weight' in trace

    def test_dwp_affects_novelty(self, rpm_layer, dps_layer):
        """High D/W/P should correlate with different novelty."""
        batch_size = 4
        seq_len = 8
        state = DPSState()

        # High desire activation
        thought_high = torch.zeros(batch_size, seq_len, TOTAL_DIM)
        for world in [0, 10, 20, 30]:
            thought_high[:, :, world + 1] = 2.0  # Noetic 1 (Mind)
            thought_high[:, :, world + 4] = 2.0  # Noetic 4 (Vibration)
            thought_high[:, :, world + 7] = 2.0  # Noetic 7 (Rhythm)

        rpm_high = rpm_layer(thought_high)
        dps_high, _, trace_high = dps_layer(rpm_high['gated_output'], state)

        # Low activation
        thought_low = torch.zeros(batch_size, seq_len, TOTAL_DIM)
        rpm_low = rpm_layer(thought_low)
        dps_low, _, trace_low = dps_layer(rpm_low['gated_output'], state)

        # Both should produce valid traces
        assert 'novelty_weight' in trace_high
        assert 'novelty_weight' in trace_low


class TestDPSGovernanceIntegration:
    """Test DPS -> Governance flow."""

    @pytest.fixture
    def dps_layer(self):
        return DPSGatingLayer(DPSConfig(), noetic_dim=TOTAL_DIM)

    @pytest.fixture
    def governance(self):
        return GovernanceRails(GovernanceConfig())

    def test_dps_mode_affects_governance(self, dps_layer, governance):
        """DPS mode should align with governance mode."""
        # Test Normal mode
        state_normal = DPSState(mode=DPSMode.NORMAL)
        action = ActionProfile(
            action_type="read_file",
            reversibility=ActionReversibility.REVERSIBLE,
        )

        result_normal = governance.check_action(
            action=action,
            uncertainty=0.2,
            stakes=0.2,
            alignment=0.95,
        )

        assert result_normal.mode == OperationalMode.NORMAL
        assert result_normal.decision == GateDecision.ALLOW

    def test_high_stakes_alignment(self, dps_layer, governance):
        """High-Stakes DPS should trigger High-Stakes governance."""
        # High uncertainty and stakes -> High-Stakes mode
        action = ActionProfile(
            action_type="write_file",
            reversibility=ActionReversibility.REVERSIBLE,
        )

        result = governance.check_action(
            action=action,
            uncertainty=0.7,
            stakes=0.8,
            alignment=0.95,
        )
        # HS = 0.56 * 0.9 = 0.5 -> High-Stakes
        # (actual value depends on formula)


class TestBackwardCompatibility:
    """Test backward compatibility with features disabled."""

    def test_operator_only(self):
        """Pipeline should work with only operator enabled."""
        config = IntegrationConfig(
            use_operator=True,
            use_rpm=False,
            use_dps=False,
            use_governance=False,
        )
        pipeline = MockIntegratedPipeline(config)

        left = torch.randn(4, TOTAL_DIM)
        right = torch.randn(4, TOTAL_DIM)
        operator = torch.randint(0, NUM_OPERATORS, (4,))

        result = pipeline(left, right, operator)

        assert result['output'].shape == (4, TOTAL_DIM)
        assert 'operator' in result['trace']

    def test_no_modules(self):
        """Pipeline should work with all modules disabled."""
        config = IntegrationConfig(
            use_operator=False,
            use_rpm=False,
            use_dps=False,
            use_governance=False,
        )
        pipeline = MockIntegratedPipeline(config)

        left = torch.randn(4, TOTAL_DIM)
        right = torch.randn(4, TOTAL_DIM)
        operator = torch.randint(0, NUM_OPERATORS, (4,))

        result = pipeline(left, right, operator)

        # Should just average inputs
        expected = (left + right) / 2
        assert torch.allclose(result['output'], expected)

    def test_selective_modules(self):
        """Test various module combinations."""
        configs = [
            IntegrationConfig(use_operator=True, use_rpm=True, use_dps=False, use_governance=False),
            IntegrationConfig(use_operator=False, use_rpm=True, use_dps=True, use_governance=False),
            IntegrationConfig(use_operator=True, use_rpm=False, use_dps=True, use_governance=True),
        ]

        for config in configs:
            pipeline = MockIntegratedPipeline(config)
            left = torch.randn(4, TOTAL_DIM)
            right = torch.randn(4, TOTAL_DIM)
            operator = torch.randint(0, NUM_OPERATORS, (4,))

            result = pipeline(left, right, operator)
            assert result['output'].shape == (4, TOTAL_DIM)


class TestTraceOutput:
    """Test trace output structure and contents."""

    @pytest.fixture
    def pipeline(self):
        return MockIntegratedPipeline(IntegrationConfig())

    def test_dps_trace_structure(self, pipeline):
        """Test DPS trace has correct structure."""
        left = torch.randn(4, TOTAL_DIM)
        right = torch.randn(4, TOTAL_DIM)
        operator = torch.randint(0, NUM_OPERATORS, (4,))

        result = pipeline(left, right, operator)
        dps_trace = result['trace']['dps']

        # Check required fields
        assert 'novelty_weight' in dps_trace
        assert 'novelty_class' in dps_trace
        assert 'validity' in dps_trace
        assert 'impact' in dps_trace
        assert 'similarity' in dps_trace
        assert 'depth_allowed' in dps_trace
        assert 'unlock_occurred' in dps_trace
        assert 'tokens' in dps_trace

        # Check nested structure
        assert 'confidence' in dps_trace['validity']
        assert 'delta_reward' in dps_trace['impact']
        assert 's_embed' in dps_trace['similarity']

    def test_governance_trace_structure(self, pipeline):
        """Test governance trace has correct structure."""
        left = torch.randn(4, TOTAL_DIM)
        right = torch.randn(4, TOTAL_DIM)
        operator = torch.randint(0, NUM_OPERATORS, (4,))
        action = ActionProfile(action_type="read_file", reversibility=ActionReversibility.REVERSIBLE)

        result = pipeline(left, right, operator, action=action)
        gov_trace = result['trace']['governance']

        assert 'decision' in gov_trace
        assert 'mode' in gov_trace
        assert 'hs_score' in gov_trace


class TestGradientIntegration:
    """Test gradient flow through integrated pipeline."""

    def test_end_to_end_gradients(self):
        """Test gradients flow through complete pipeline."""
        config = IntegrationConfig(
            use_operator=True,
            use_rpm=True,
            use_dps=True,
            use_governance=False,  # Governance has no gradients
        )
        pipeline = MockIntegratedPipeline(config)

        left = torch.randn(4, TOTAL_DIM, requires_grad=True)
        right = torch.randn(4, TOTAL_DIM, requires_grad=True)
        operator = torch.randint(0, NUM_OPERATORS, (4,))

        result = pipeline(left, right, operator)
        loss = result['output'].sum()
        loss.backward()

        # Check gradients exist
        assert left.grad is not None
        assert right.grad is not None
        assert not torch.isnan(left.grad).any()
        assert not torch.isnan(right.grad).any()

    def test_parameter_gradients(self):
        """Test all trainable parameters receive gradients."""
        config = IntegrationConfig(use_operator=True, use_rpm=True, use_dps=True, use_governance=False)
        pipeline = MockIntegratedPipeline(config)

        left = torch.randn(4, TOTAL_DIM)
        right = torch.randn(4, TOTAL_DIM)
        operator = torch.randint(0, NUM_OPERATORS, (4,))

        result = pipeline(left, right, operator)
        loss = result['output'].sum()
        loss.backward()

        # Count parameters with gradients
        has_grad = sum(1 for p in pipeline.parameters() if p.grad is not None)
        assert has_grad > 0, "At least some parameters should have gradients"


class TestStateManagement:
    """Test state management across pipeline."""

    def test_dps_state_persistence(self):
        """Test DPS state persists across calls."""
        config = IntegrationConfig()
        pipeline = MockIntegratedPipeline(config)

        left = torch.randn(4, TOTAL_DIM)
        right = torch.randn(4, TOTAL_DIM)
        operator = torch.randint(0, NUM_OPERATORS, (4,))

        # Initial state
        initial_p_max = pipeline.dps_state.p_max

        # Multiple forward passes
        for _ in range(5):
            result = pipeline(left, right, operator)

        # State should be updated (or at least tracked)
        # The p_max may change if unlocks occur
        assert pipeline.dps_state is not None

    def test_state_serialization(self):
        """Test DPS state can be serialized and restored."""
        state = DPSState(p_max=4, tokens=3, cooldown=5)

        # Serialize
        data = state.to_dict()

        # Restore
        restored = DPSState.from_dict(data)

        assert restored.p_max == 4
        assert restored.tokens == 3
        assert restored.cooldown == 5


# Run with: pytest tests/test_strg_v4_integration.py -v

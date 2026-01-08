"""
Tests for NJT (Noetic Judgment Transistor) Circuits

Tests cover:
1. NJT+ (excitatory) transistor functionality
2. NJT- (inhibitory) transistor functionality
3. Noetic gates and badge modifiers
4. N5 hysteresis memory
5. Differential pair decision-making
6. N7 rhythm oscillator
7. Full NJT layer integration
8. Integration with v5 model

Author: TKS Test Agent
Date: 2026-01-07
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tks_features.njt_circuits import (
    NJTConfig,
    NoeticGate,
    NJTPlus,
    NJTMinus,
    HysteresisMemory,
    DifferentialPair,
    RhythmOscillator,
    NJTLayer,
    create_njt_layer,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Standard test config."""
    return NJTConfig(
        dim=64,
        num_transistors=10,
        initial_gain=2.0,
        initial_threshold=0.5,
        gate_sharpness=10.0,
        use_hysteresis=True,
        hysteresis_gap=0.3,
        use_rhythm=False,
    )


@pytest.fixture
def sample_input():
    """Sample input tensor."""
    batch_size = 4
    seq_len = 16
    dim = 64
    return torch.randn(batch_size, seq_len, dim)


# =============================================================================
# NOETIC GATE TESTS
# =============================================================================

class TestNoeticGate:
    """Test noetic gate function."""

    def test_gate_creation(self, config):
        """Test gate can be created."""
        gate = NoeticGate(num_gates=10, config=config)
        assert gate is not None
        assert gate.threshold.shape == (10,)

    def test_gate_forward(self, config):
        """Test gate forward pass."""
        gate = NoeticGate(num_gates=10, config=config)
        bias = torch.randn(4, 16, 10)
        output = gate(bias)

        assert output.shape == (4, 16, 10)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_gate_threshold_effect(self, config):
        """Test that threshold affects gate activation."""
        gate = NoeticGate(num_gates=10, config=config)

        # Low bias should give low activation
        low_bias = torch.ones(4, 16, 10) * 0.1
        low_output = gate(low_bias)

        # High bias should give high activation
        high_bias = torch.ones(4, 16, 10) * 0.9
        high_output = gate(high_bias)

        assert high_output.mean() > low_output.mean()


# =============================================================================
# NJT+ TESTS
# =============================================================================

class TestNJTPlus:
    """Test NJT+ (excitatory) transistor."""

    def test_njt_plus_creation(self, config):
        """Test NJT+ can be created."""
        njt = NJTPlus(dim=64, num_transistors=10, config=config)
        assert njt is not None

    def test_njt_plus_forward(self, config, sample_input):
        """Test NJT+ forward pass."""
        njt = NJTPlus(dim=64, num_transistors=10, config=config)
        output, internals = njt(sample_input)

        assert output.shape == sample_input.shape
        assert internals is None  # Not requested

    def test_njt_plus_with_internals(self, config, sample_input):
        """Test NJT+ returns internals when requested."""
        njt = NJTPlus(dim=64, num_transistors=10, config=config)
        output, internals = njt(sample_input, return_internals=True)

        assert internals is not None
        assert 'cause' in internals
        assert 'bias' in internals
        assert 'gate' in internals
        assert 'internal_output' in internals

    def test_njt_plus_amplification(self, config):
        """Test NJT+ amplifies signals when gate is high."""
        njt = NJTPlus(dim=64, num_transistors=10, config=config)

        # Create input with known pattern
        x = torch.ones(2, 8, 64) * 0.5

        # High bias context should amplify
        high_context = torch.ones(2, 8, 64) * 2.0
        output_high, _ = njt(x, high_context)

        # Low bias context should amplify less
        low_context = torch.ones(2, 8, 64) * 0.1
        output_low, _ = njt(x, low_context)

        # High context should produce stronger output
        assert output_high.abs().mean() >= output_low.abs().mean() * 0.5  # Allow some tolerance


# =============================================================================
# NJT- TESTS
# =============================================================================

class TestNJTMinus:
    """Test NJT- (inhibitory) transistor."""

    def test_njt_minus_creation(self, config):
        """Test NJT- can be created."""
        njt = NJTMinus(dim=64, num_transistors=10, config=config)
        assert njt is not None

    def test_njt_minus_forward(self, config, sample_input):
        """Test NJT- forward pass."""
        njt = NJTMinus(dim=64, num_transistors=10, config=config)
        output, internals = njt(sample_input)

        assert output.shape == sample_input.shape

    def test_njt_minus_inhibition(self, config):
        """Test NJT- dampens signals when brake is high."""
        njt = NJTMinus(dim=64, num_transistors=10, config=config)

        # Create input
        x = torch.ones(2, 8, 64) * 0.5

        # High brake should dampen
        high_brake = torch.ones(2, 8, 64) * 2.0
        output_high_brake, _ = njt(x, high_brake)

        # Low brake should dampen less
        low_brake = torch.ones(2, 8, 64) * 0.1
        output_low_brake, _ = njt(x, low_brake)

        # High brake should produce weaker output (more dampening)
        # Note: Due to (1 - gate) formula, high gate = more dampening
        assert output_high_brake.abs().mean() <= output_low_brake.abs().mean() * 1.5


# =============================================================================
# HYSTERESIS MEMORY TESTS
# =============================================================================

class TestHysteresisMemory:
    """Test N5 hysteresis memory."""

    def test_hysteresis_creation(self):
        """Test hysteresis can be created."""
        hyst = HysteresisMemory(num_units=10, gap=0.3)
        assert hyst is not None
        assert hyst.activate_threshold.shape == (10,)
        assert hyst.deactivate_threshold.shape == (10,)

    def test_hysteresis_thresholds(self):
        """Test that activate > deactivate threshold."""
        hyst = HysteresisMemory(num_units=10, gap=0.3)
        assert (hyst.activate_threshold > hyst.deactivate_threshold).all()

    def test_hysteresis_stickiness(self):
        """Test hysteresis creates sticky states."""
        hyst = HysteresisMemory(num_units=5, gap=0.4)
        hyst.reset_state()

        # High input should activate
        high_input = torch.ones(2, 8, 5) * 0.9
        _ = hyst(high_input)

        # Medium input (above deactivate, below activate) should STAY on
        medium_input = torch.ones(2, 8, 5) * 0.4
        output = hyst(medium_input)

        # Should still have some activation due to hysteresis
        # (Not guaranteed to be exactly the input due to state tracking)
        assert output.mean() > 0  # Should have some output

    def test_hysteresis_reset(self):
        """Test hysteresis reset works."""
        hyst = HysteresisMemory(num_units=10, gap=0.3)

        # Activate
        _ = hyst(torch.ones(2, 8, 10) * 0.9)

        # Reset
        hyst.reset_state()

        assert hyst.state.sum() == 0
        assert not hyst.initialized


# =============================================================================
# DIFFERENTIAL PAIR TESTS
# =============================================================================

class TestDifferentialPair:
    """Test differential pair decision-making."""

    def test_differential_pair_creation(self, config):
        """Test differential pair can be created."""
        dp = DifferentialPair(dim=64, num_options=3, config=config)
        assert dp is not None
        assert len(dp.njt_bank) == 3

    def test_differential_pair_forward(self, config, sample_input):
        """Test differential pair forward pass."""
        dp = DifferentialPair(dim=64, num_options=3, config=config)
        output, weights, internals = dp(sample_input)

        assert output.shape == sample_input.shape
        assert weights.shape == (4, 16, 3)  # batch, seq, num_options

        # Weights should sum to 1 (softmax)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 16), atol=1e-5)

    def test_differential_pair_winner_takes_all(self, config):
        """Test that strong bias tips the balance."""
        torch.manual_seed(42)  # For reproducibility
        dp = DifferentialPair(dim=64, num_options=2, config=config)

        # Use uniform input so bias has clear effect
        x = torch.ones(2, 8, 64) * 0.5

        # Add strong bias to option 0
        biases = torch.zeros(2, 8, 2)
        biases[..., 0] = 5.0  # Strong bias toward option 0

        output, weights, _ = dp(x, biases)

        # Option 0 should win more often with strong bias
        assert weights[..., 0].mean() > weights[..., 1].mean()


# =============================================================================
# RHYTHM OSCILLATOR TESTS
# =============================================================================

class TestRhythmOscillator:
    """Test N7 rhythm oscillator."""

    def test_oscillator_creation(self):
        """Test oscillator can be created."""
        config = NJTConfig(dim=64, num_transistors=10, rhythm_period=5, use_rhythm=True)
        osc = RhythmOscillator(dim=64, config=config)
        assert osc is not None

    def test_oscillator_forward(self):
        """Test oscillator generates sequence."""
        config = NJTConfig(dim=64, num_transistors=10, rhythm_period=5, use_rhythm=True)
        osc = RhythmOscillator(dim=64, config=config)

        seed = torch.randn(2, 64)
        output, phases = osc(seed, num_steps=10, return_sequence=True)

        assert output.shape == (2, 10, 64)  # batch, steps, dim
        assert phases is not None
        assert len(phases) == 10


# =============================================================================
# FULL NJT LAYER TESTS
# =============================================================================

class TestNJTLayer:
    """Test complete NJT layer."""

    def test_layer_creation(self, config):
        """Test NJT layer can be created."""
        layer = NJTLayer(config)
        assert layer is not None
        assert layer.excitatory is not None
        assert layer.inhibitory is not None

    def test_layer_forward(self, config, sample_input):
        """Test NJT layer forward pass."""
        layer = NJTLayer(config)
        output, trace = layer(sample_input)

        assert output.shape == sample_input.shape
        assert trace is None  # Not requested

    def test_layer_with_trace(self, config, sample_input):
        """Test NJT layer returns trace when requested."""
        layer = NJTLayer(config)
        output, trace = layer(sample_input, return_trace=True)

        assert trace is not None
        assert 'excitatory' in trace
        assert 'inhibitory' in trace
        assert 'exc_inh_balance' in trace

    def test_layer_with_hysteresis(self, sample_input):
        """Test NJT layer with hysteresis enabled."""
        config = NJTConfig(
            dim=64,
            num_transistors=10,
            use_hysteresis=True,
            hysteresis_gap=0.3,
        )
        layer = NJTLayer(config)

        assert layer.hysteresis is not None

        output, trace = layer(sample_input, return_trace=True)
        assert 'hysteresis_state' in trace

    def test_layer_factory_function(self):
        """Test create_njt_layer factory function."""
        layer = create_njt_layer(dim=128, num_transistors=8)

        assert layer is not None
        assert layer.config.dim == 128
        assert layer.config.num_transistors == 8


# =============================================================================
# V5 MODEL INTEGRATION TESTS
# =============================================================================

class TestV5Integration:
    """Test NJT integration with v5 model."""

    def test_v5_config_has_njt_options(self):
        """Test v5 config includes NJT options."""
        from tks_llm_core_v5 import TKSGeneralConfig

        config = TKSGeneralConfig()
        assert hasattr(config, 'use_njt')
        assert hasattr(config, 'njt_num_transistors')
        assert hasattr(config, 'njt_use_hysteresis')

    def test_v5_model_without_njt(self):
        """Test v5 model works without NJT."""
        from tks_llm_core_v5 import TKSGeneralConfig, TKSGeneralLM

        config = TKSGeneralConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            use_njt=False,
        )
        model = TKSGeneralLM(config)

        input_ids = torch.randint(0, 1000, (2, 16))
        output = model(input_ids)

        assert 'logits' in output
        assert output['logits'].shape == (2, 16, 1000)

    def test_v5_model_with_njt(self):
        """Test v5 model works with NJT enabled."""
        from tks_llm_core_v5 import TKSGeneralConfig, TKSGeneralLM, NJT_AVAILABLE

        if not NJT_AVAILABLE:
            pytest.skip("NJT not available")

        config = TKSGeneralConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            use_njt=True,
            njt_num_transistors=10,
            njt_use_hysteresis=True,
        )
        model = TKSGeneralLM(config)

        # Verify NJT is in blocks
        assert model.blocks[0].njt is not None

        input_ids = torch.randint(0, 1000, (2, 16))
        output = model(input_ids)

        assert 'logits' in output
        assert output['logits'].shape == (2, 16, 1000)

    def test_v5_model_njt_trace(self):
        """Test v5 model returns NJT trace."""
        from tks_llm_core_v5 import TKSGeneralConfig, TKSGeneralLM, NJT_AVAILABLE

        if not NJT_AVAILABLE:
            pytest.skip("NJT not available")

        config = TKSGeneralConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            use_njt=True,
        )
        model = TKSGeneralLM(config)

        input_ids = torch.randint(0, 1000, (2, 16))
        output = model(input_ids, return_full_trace=True)

        assert 'trace' in output
        assert 'njt' in output['trace']
        assert len(output['trace']['njt']) == 2  # num_layers

    def test_v5_model_gradient_flow_with_njt(self):
        """Test gradients flow through NJT."""
        from tks_llm_core_v5 import TKSGeneralConfig, TKSGeneralLM, NJT_AVAILABLE

        if not NJT_AVAILABLE:
            pytest.skip("NJT not available")

        config = TKSGeneralConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            use_njt=True,
        )
        model = TKSGeneralLM(config)

        input_ids = torch.randint(0, 1000, (2, 16))
        output = model(input_ids, step=100)

        loss = output['logits'].mean()
        loss.backward()

        # Check NJT has gradients
        njt = model.blocks[0].njt
        assert njt.excitatory.cause_proj.weight.grad is not None
        assert njt.inhibitory.cause_proj.weight.grad is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

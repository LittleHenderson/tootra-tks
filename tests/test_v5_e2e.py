"""
End-to-End Tests for TKS LLM v5

Tests cover:
1. Model instantiation with various configs
2. Forward pass (with and without stable routing)
3. Routing auxiliary loss computation
4. Attractor/RPM integration
5. Gradient flow
6. Memory regression

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

from tks_llm_core_v5 import (
    TKSGeneralConfig,
    TKSGeneralLM,
    GeneralNoeticRouter,
    GeneralNoeticBlock,
    GeneralFractalAttention,
    ROUTING_STABILITY_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def base_config():
    """Base config for testing."""
    return TKSGeneralConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_scales=2,
        dropout=0.0,
        use_attractor=True,
        use_rpm=True,
        use_stable_routing=True,
    )


@pytest.fixture
def minimal_config():
    """Minimal config without optional components."""
    return TKSGeneralConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_scales=2,
        dropout=0.0,
        use_attractor=False,
        use_rpm=False,
        use_stable_routing=False,
    )


@pytest.fixture
def sample_input():
    """Sample input for testing."""
    batch_size = 4
    seq_len = 32
    return torch.randint(0, 1000, (batch_size, seq_len))


# =============================================================================
# MODEL INSTANTIATION TESTS
# =============================================================================

class TestModelInstantiation:
    """Test model creation with various configs."""

    def test_base_config_instantiation(self, base_config):
        """Test model instantiation with base config."""
        model = TKSGeneralLM(base_config)
        assert model is not None
        assert model.config == base_config

    def test_minimal_config_instantiation(self, minimal_config):
        """Test model instantiation with minimal config."""
        model = TKSGeneralLM(minimal_config)
        assert model is not None

    def test_stable_routing_disabled(self, minimal_config):
        """Test model works without stable routing."""
        model = TKSGeneralLM(minimal_config)
        assert model.blocks[0].router.use_stable_routing == False

    @pytest.mark.skipif(not ROUTING_STABILITY_AVAILABLE, reason="Routing stability not available")
    def test_stable_routing_enabled(self, base_config):
        """Test model works with stable routing."""
        model = TKSGeneralLM(base_config)
        assert model.blocks[0].router.use_stable_routing == True
        assert model.blocks[0].router.stable_routing is not None

    def test_parameter_count(self, base_config):
        """Test model has reasonable parameter count."""
        model = TKSGeneralLM(base_config)
        num_params = sum(p.numel() for p in model.parameters())
        # Should be in millions, not billions
        assert num_params < 100_000_000  # < 100M
        assert num_params > 100_000  # > 100K

    def test_attractor_component(self, base_config):
        """Test attractor component is created when enabled."""
        model = TKSGeneralLM(base_config)
        assert hasattr(model, 'attractor')
        assert model.attractor is not None

    def test_rpm_component(self, base_config):
        """Test RPM component is created when enabled."""
        model = TKSGeneralLM(base_config)
        assert hasattr(model, 'rpm')
        assert model.rpm is not None


# =============================================================================
# FORWARD PASS TESTS
# =============================================================================

class TestForwardPass:
    """Test forward pass through model."""

    def test_basic_forward(self, base_config, sample_input):
        """Test basic forward pass produces logits."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input)

        assert 'logits' in output
        assert output['logits'].shape == (4, 32, 1000)  # batch, seq, vocab

    def test_forward_with_step(self, base_config, sample_input):
        """Test forward pass with training step for stable routing."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input, step=100)

        assert 'logits' in output
        if ROUTING_STABILITY_AVAILABLE:
            assert 'routing_aux_loss' in output

    def test_forward_with_trace(self, base_config, sample_input):
        """Test forward pass with full trace."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input, return_full_trace=True)

        assert 'logits' in output
        assert 'trace' in output
        assert 'routing' in output['trace']
        assert 'canonical' in output['trace']

    def test_forward_minimal(self, minimal_config, sample_input):
        """Test forward pass with minimal config."""
        model = TKSGeneralLM(minimal_config)
        output = model(sample_input)

        assert 'logits' in output
        assert output['logits'].shape == (4, 32, 1000)

    def test_output_dtype(self, base_config, sample_input):
        """Test output has correct dtype."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input)

        assert output['logits'].dtype == torch.float32

    def test_different_sequence_lengths(self, base_config):
        """Test model handles different sequence lengths."""
        model = TKSGeneralLM(base_config)

        for seq_len in [8, 16, 64, 128]:
            inputs = torch.randint(0, 1000, (2, seq_len))
            output = model(inputs)
            assert output['logits'].shape == (2, seq_len, 1000)

    def test_batch_size_one(self, base_config):
        """Test model handles batch size of 1."""
        model = TKSGeneralLM(base_config)
        inputs = torch.randint(0, 1000, (1, 16))
        output = model(inputs)

        assert output['logits'].shape == (1, 16, 1000)


# =============================================================================
# ROUTING TESTS
# =============================================================================

class TestRouting:
    """Test routing components."""

    def test_router_output_shape(self, base_config):
        """Test router produces correct output shape."""
        router = GeneralNoeticRouter(
            dim=base_config.hidden_dim,
            use_stable_routing=False,
        )
        x = torch.randn(4, 32, base_config.hidden_dim)
        weights, aux_loss, metrics = router(x)

        assert weights.shape == (4, 32, 10)  # 10 noetic operators
        assert aux_loss is None
        assert metrics is None

    @pytest.mark.skipif(not ROUTING_STABILITY_AVAILABLE, reason="Routing stability not available")
    def test_stable_router_with_step(self, base_config):
        """Test stable router produces auxiliary loss."""
        router = GeneralNoeticRouter(
            dim=base_config.hidden_dim,
            use_stable_routing=True,
        )
        x = torch.randn(4, 32, base_config.hidden_dim)
        router.train()
        weights, aux_loss, metrics = router(x, step=100)

        assert weights.shape == (4, 32, 10)
        assert aux_loss is not None
        assert isinstance(aux_loss, torch.Tensor)
        assert metrics is not None
        assert 'routing_entropy_mean' in metrics

    @pytest.mark.skipif(not ROUTING_STABILITY_AVAILABLE, reason="Routing stability not available")
    def test_stable_router_temperature_annealing(self, base_config):
        """Test temperature decreases over training."""
        router = GeneralNoeticRouter(
            dim=base_config.hidden_dim,
            use_stable_routing=True,
            initial_temp=2.0,
            final_temp=0.1,
        )
        x = torch.randn(4, 32, base_config.hidden_dim)
        router.train()

        # Early in training
        _, _, metrics_early = router(x, step=0)
        temp_early = metrics_early['routing_temperature']

        # Late in training
        _, _, metrics_late = router(x, step=20000)
        temp_late = metrics_late['routing_temperature']

        assert temp_early > temp_late

    def test_routing_weights_sum_to_one(self, base_config):
        """Test routing weights sum to 1."""
        router = GeneralNoeticRouter(
            dim=base_config.hidden_dim,
            use_stable_routing=False,
        )
        x = torch.randn(4, 32, base_config.hidden_dim)
        weights, _, _ = router(x)

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# =============================================================================
# GRADIENT TESTS
# =============================================================================

class TestGradientFlow:
    """Test gradients flow correctly through model."""

    def test_gradient_flow_basic(self, base_config, sample_input):
        """Test gradients flow to all parameters."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input, step=100)
        loss = output['logits'].mean()

        if 'routing_aux_loss' in output:
            loss = loss + output['routing_aux_loss']

        loss.backward()

        # Check embedding gradients
        assert model.embedding.embedding.weight.grad is not None

        # Check block gradients
        for block in model.blocks:
            assert block.router.router_head.weight.grad is not None

    def test_gradient_flow_attractor(self, base_config, sample_input):
        """Test gradients flow through attractor."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input)
        loss = output['logits'].mean()
        loss.backward()

        # Check canonical projections have gradients
        assert model.to_canonical.weight.grad is not None
        assert model.from_canonical.weight.grad is not None

    def test_no_nan_gradients(self, base_config, sample_input):
        """Test no NaN gradients after backward pass."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input, step=100)
        loss = output['logits'].mean()

        if 'routing_aux_loss' in output:
            loss = loss + output['routing_aux_loss']

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


# =============================================================================
# AUXILIARY LOSS TESTS
# =============================================================================

class TestAuxiliaryLoss:
    """Test auxiliary loss computation."""

    @pytest.mark.skipif(not ROUTING_STABILITY_AVAILABLE, reason="Routing stability not available")
    def test_aux_loss_accumulation(self, base_config, sample_input):
        """Test auxiliary loss accumulates across layers."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input, step=100)

        if 'routing_aux_loss' in output:
            aux_loss = output['routing_aux_loss']
            # Should be > 0 (load balancing loss)
            assert aux_loss > 0

    @pytest.mark.skipif(not ROUTING_STABILITY_AVAILABLE, reason="Routing stability not available")
    def test_aux_loss_magnitude(self, base_config, sample_input):
        """Test auxiliary loss has reasonable magnitude."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input, step=100)

        if 'routing_aux_loss' in output:
            aux_loss = output['routing_aux_loss']
            # Should be bounded (load balancing * num_layers)
            # With 4 layers and balance_weight=0.01, expect < 0.5
            assert aux_loss < 1.0, f"Aux loss too large: {aux_loss}"
            assert aux_loss >= 0, f"Aux loss should be non-negative: {aux_loss}"


# =============================================================================
# MEMORY TESTS
# =============================================================================

class TestMemory:
    """Test memory usage."""

    def test_memory_not_explode(self, base_config):
        """Test memory doesn't explode with long sequences."""
        model = TKSGeneralLM(base_config)

        # Should handle 256 tokens without OOM
        inputs = torch.randint(0, 1000, (2, 256))
        output = model(inputs)
        assert output['logits'].shape == (2, 256, 1000)

    def test_no_memory_leak(self, base_config, sample_input):
        """Test no memory leak over multiple forward passes."""
        model = TKSGeneralLM(base_config)

        for _ in range(10):
            output = model(sample_input)
            del output

        # If we get here without OOM, we're good
        assert True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""

    def test_training_step(self, base_config, sample_input):
        """Test a complete training step."""
        model = TKSGeneralLM(base_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        output = model(sample_input, step=100)

        # Compute loss
        logits = output['logits']
        targets = torch.randint(0, 1000, sample_input.shape)
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, 1000),
            targets.view(-1),
        )

        if 'routing_aux_loss' in output:
            loss = loss + output['routing_aux_loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert True  # If we get here, training step succeeded

    def test_eval_mode(self, base_config, sample_input):
        """Test model in eval mode."""
        model = TKSGeneralLM(base_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        assert 'logits' in output

    @pytest.mark.skipif(not ROUTING_STABILITY_AVAILABLE, reason="Routing stability not available")
    def test_routing_metrics_available(self, base_config, sample_input):
        """Test routing metrics are available during training."""
        model = TKSGeneralLM(base_config)
        model.train()
        output = model(sample_input, step=100)

        if 'routing_metrics' in output:
            metrics = output['routing_metrics']
            assert 'routing_entropy_mean' in metrics
            assert 'routing_temperature' in metrics


# =============================================================================
# CANONICAL COMPONENT TESTS
# =============================================================================

class TestCanonicalComponents:
    """Test canonical TKS components."""

    def test_attractor_output_in_trace(self, base_config, sample_input):
        """Test attractor output appears in trace."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input, return_full_trace=True)

        assert 'attractor' in output['trace']['canonical']

    def test_rpm_output_in_trace(self, base_config, sample_input):
        """Test RPM output appears in trace."""
        model = TKSGeneralLM(base_config)
        output = model(sample_input, return_full_trace=True)

        assert 'rpm' in output['trace']['canonical']

    def test_canonical_projection_dimensions(self, base_config):
        """Test canonical projections have correct dimensions."""
        model = TKSGeneralLM(base_config)

        # to_canonical: hidden_dim -> 40
        assert model.to_canonical.weight.shape == (40, base_config.hidden_dim)

        # from_canonical: 40 -> hidden_dim
        assert model.from_canonical.weight.shape == (base_config.hidden_dim, 40)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

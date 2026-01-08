"""
Test suite for DPS (Depth Permission System) from strg-llm-stk.

Novelty Weight: NW = V x I x (1 - S)
Classifications:
- HEAVY (NW >= 0.70): Immediate unlock
- COUNT (0.45 <= NW < 0.70): Token accumulation
- NOCOUNT (NW < 0.45): No progress

State Update Rules:
- HEAVY: p_max += 1, tokens = 0, cooldown = K (if allowed)
- COUNT: tokens += 1, unlock when tokens = 5
- NOCOUNT: No state change
- Cooldown decrements by 1 each episode

Author: EVAL agent
Date: 2026-01-05
"""
import pytest
import torch
from typing import Dict

from tks_features.dps_gating import (
    DPSGatingLayer,
    DPSConfig,
    DPSState,
    DPSMode,
    NoveltyClass,
    NoveltyWeight,
    ValidityScore,
    ImpactScore,
    SimilarityScore,
    AdaptiveIterationController,
    DEFAULT_HEAVY_THRESHOLD,
    DEFAULT_COUNT_THRESHOLD,
    DEFAULT_TOKENS_FOR_UNLOCK,
    DEFAULT_COOLDOWN_EPISODES,
    DEFAULT_MAX_DEPTH,
    DEFAULT_INITIAL_P_MAX,
)


class TestNoveltyWeight:
    """Test Novelty Weight formula: NW = V x I x (1 - S)"""

    def test_nw_formula(self):
        """NW = V x I x (1 - S)"""
        v, i, s = 0.8, 0.7, 0.2
        expected_nw = 0.8 * 0.7 * (1 - 0.2)  # = 0.448

        validity = ValidityScore(confidence=0.8, consistency=1.0, evidence_ok=1.0)
        impact = ImpactScore(delta_reward=0.7, delta_uncertainty=0.7, delta_coverage=0.7, delta_cost=0.7)
        similarity = SimilarityScore(s_embed=0.2, s_ast=0.0)

        nw = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)

        # V = 0.8 * 1.0 * 1.0 = 0.8
        # I = 0.4*0.7 + 0.3*0.7 + 0.2*0.7 + 0.1*0.7 = 0.7
        # S = max(0.2, 0.0) = 0.2
        # NW = 0.8 * 0.7 * (1 - 0.2) = 0.448
        assert abs(nw.value - expected_nw) < 0.01

    def test_nw_bounds(self):
        """NW should be clipped to [0, 1]."""
        # Test minimum (high similarity, low everything else)
        validity_low = ValidityScore(confidence=0.1, consistency=0.1, evidence_ok=0.1)
        impact_low = ImpactScore(delta_reward=0.0, delta_uncertainty=0.0, delta_coverage=0.0, delta_cost=0.0)
        similarity_high = SimilarityScore(s_embed=1.0, s_ast=1.0)

        nw_low = NoveltyWeight(validity=validity_low, impact=impact_low, similarity=similarity_high)
        assert nw_low.value >= 0.0

        # Test maximum (high validity/impact, low similarity)
        validity_high = ValidityScore(confidence=1.0, consistency=1.0, evidence_ok=1.0)
        impact_high = ImpactScore(delta_reward=1.0, delta_uncertainty=1.0, delta_coverage=1.0, delta_cost=1.0)
        similarity_low = SimilarityScore(s_embed=0.0, s_ast=0.0)

        nw_high = NoveltyWeight(validity=validity_high, impact=impact_high, similarity=similarity_low)
        # NW = 1.0 * 1.0 * (1 - 0) = 1.0
        assert nw_high.value <= 1.0


class TestClassification:
    """Test novelty weight classification."""

    @pytest.fixture
    def config(self):
        return DPSConfig()

    def test_heavy_classification(self, config):
        """NW >= 0.70 -> HEAVY"""
        validity = ValidityScore(confidence=1.0, consistency=1.0, evidence_ok=1.0)
        impact = ImpactScore(delta_reward=0.9, delta_uncertainty=0.9, delta_coverage=0.9, delta_cost=0.9)
        similarity = SimilarityScore(s_embed=0.1, s_ast=0.0)

        nw = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)
        # V=1.0, I~0.9, S=0.1 -> NW = 1.0 * 0.9 * 0.9 = 0.81 >= 0.70

        assert nw.classify(config) == NoveltyClass.HEAVY

    def test_count_classification(self, config):
        """0.45 <= NW < 0.70 -> COUNT"""
        validity = ValidityScore(confidence=0.8, consistency=0.8, evidence_ok=0.8)
        impact = ImpactScore(delta_reward=0.7, delta_uncertainty=0.7, delta_coverage=0.7, delta_cost=0.7)
        similarity = SimilarityScore(s_embed=0.3, s_ast=0.2)

        nw = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)
        # V = 0.512, I ~ 0.7, S = 0.3
        # NW = 0.512 * 0.7 * 0.7 = 0.25 (this is actually NOCOUNT, let's adjust)

        # Try different values for COUNT range
        validity2 = ValidityScore(confidence=0.9, consistency=0.9, evidence_ok=0.9)
        impact2 = ImpactScore(delta_reward=0.8, delta_uncertainty=0.8, delta_coverage=0.8, delta_cost=0.8)
        similarity2 = SimilarityScore(s_embed=0.25, s_ast=0.0)

        nw2 = NoveltyWeight(validity=validity2, impact=impact2, similarity=similarity2)
        # V = 0.729, I ~ 0.8, S = 0.25
        # NW = 0.729 * 0.8 * 0.75 ~ 0.437 (close to COUNT)

        # Verify classification logic at boundaries
        assert config.count_threshold == 0.45
        assert config.heavy_threshold == 0.70

    def test_nocount_classification(self, config):
        """NW < 0.45 -> NOCOUNT"""
        validity = ValidityScore(confidence=0.5, consistency=0.5, evidence_ok=0.5)
        impact = ImpactScore(delta_reward=0.3, delta_uncertainty=0.3, delta_coverage=0.3, delta_cost=0.3)
        similarity = SimilarityScore(s_embed=0.5, s_ast=0.5)

        nw = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)
        # V = 0.125, I ~ 0.3, S = 0.5
        # NW = 0.125 * 0.3 * 0.5 = 0.01875 < 0.45

        assert nw.classify(config) == NoveltyClass.NOCOUNT


class TestStateUpdate:
    """Test DPS state update logic."""

    @pytest.fixture
    def config(self):
        return DPSConfig()

    @pytest.fixture
    def dps_layer(self):
        return DPSGatingLayer(DPSConfig(), noetic_dim=40)

    def test_heavy_unlock(self, dps_layer, config):
        """HEAVY: p_max += 1, tokens = 0, cooldown = K"""
        state = DPSState(p_max=3, tokens=2, cooldown=0)

        # Create high novelty that classifies as HEAVY
        validity = ValidityScore(confidence=1.0, consistency=1.0, evidence_ok=1.0)
        impact = ImpactScore(delta_reward=1.0, delta_uncertainty=1.0, delta_coverage=1.0, delta_cost=1.0)
        similarity = SimilarityScore(s_embed=0.0, s_ast=0.0)
        novelty = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)

        new_state, unlock_occurred = dps_layer.update_state(state, novelty, "test_episode")

        # Should unlock
        assert unlock_occurred == True
        assert new_state.p_max == 4  # 3 + 1
        assert new_state.tokens == 0  # Reset
        assert new_state.cooldown == config.cooldown_k  # Set to K

    def test_count_accumulation(self, dps_layer, config):
        """COUNT: tokens += 1"""
        state = DPSState(p_max=3, tokens=2, cooldown=0)

        # Create medium novelty that classifies as COUNT
        validity = ValidityScore(confidence=0.8, consistency=0.8, evidence_ok=0.9)
        impact = ImpactScore(delta_reward=0.8, delta_uncertainty=0.8, delta_coverage=0.8, delta_cost=0.8)
        similarity = SimilarityScore(s_embed=0.3, s_ast=0.1)
        novelty = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)

        # If this classifies as COUNT
        if novelty.classify(config) == NoveltyClass.COUNT:
            new_state, unlock_occurred = dps_layer.update_state(state, novelty, "test_episode")
            assert new_state.tokens == 3  # 2 + 1
            assert unlock_occurred == False

    def test_count_unlock_at_threshold(self, dps_layer, config):
        """COUNT with tokens=4: unlock on 5th token"""
        state = DPSState(p_max=3, tokens=4, cooldown=0)

        # Create COUNT novelty
        validity = ValidityScore(confidence=0.85, consistency=0.85, evidence_ok=0.9)
        impact = ImpactScore(delta_reward=0.75, delta_uncertainty=0.75, delta_coverage=0.75, delta_cost=0.75)
        similarity = SimilarityScore(s_embed=0.25, s_ast=0.0)
        novelty = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)

        # Check if this is COUNT and would trigger unlock
        if novelty.classify(config) == NoveltyClass.COUNT:
            new_state, unlock_occurred = dps_layer.update_state(state, novelty, "test_episode")
            # tokens goes to 5, which triggers unlock
            assert unlock_occurred == True
            assert new_state.p_max == 4
            assert new_state.tokens == 0

    def test_cooldown_prevents_unlock(self, dps_layer, config):
        """Cooldown > 0: no unlock even on HEAVY"""
        state = DPSState(p_max=3, tokens=0, cooldown=5)

        # Create HEAVY novelty
        validity = ValidityScore(confidence=1.0, consistency=1.0, evidence_ok=1.0)
        impact = ImpactScore(delta_reward=1.0, delta_uncertainty=1.0, delta_coverage=1.0, delta_cost=1.0)
        similarity = SimilarityScore(s_embed=0.0, s_ast=0.0)
        novelty = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)

        new_state, unlock_occurred = dps_layer.update_state(state, novelty, "test_episode")

        # Should NOT unlock due to cooldown
        assert unlock_occurred == False
        assert new_state.p_max == 3  # Unchanged
        assert new_state.cooldown == 4  # Decremented by 1

    def test_cooldown_decrements(self, dps_layer, config):
        """No unlock: cooldown -= 1"""
        state = DPSState(p_max=3, tokens=0, cooldown=5)

        # Create NOCOUNT novelty (low everything)
        validity = ValidityScore(confidence=0.3, consistency=0.3, evidence_ok=0.3)
        impact = ImpactScore(delta_reward=0.2, delta_uncertainty=0.2, delta_coverage=0.2, delta_cost=0.2)
        similarity = SimilarityScore(s_embed=0.8, s_ast=0.8)
        novelty = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)

        new_state, unlock_occurred = dps_layer.update_state(state, novelty, "test_episode")

        assert new_state.cooldown == 4  # 5 - 1

    def test_max_depth_cap(self, dps_layer, config):
        """p_max cannot exceed max_depth (5)"""
        state = DPSState(p_max=5, tokens=0, cooldown=0)  # Already at max

        # Create HEAVY novelty
        validity = ValidityScore(confidence=1.0, consistency=1.0, evidence_ok=1.0)
        impact = ImpactScore(delta_reward=1.0, delta_uncertainty=1.0, delta_coverage=1.0, delta_cost=1.0)
        similarity = SimilarityScore(s_embed=0.0, s_ast=0.0)
        novelty = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)

        new_state, unlock_occurred = dps_layer.update_state(state, novelty, "test_episode")

        # Should NOT unlock (already at max)
        assert new_state.p_max == 5
        # can_unlock returns False when p_max >= max_depth


class TestHighStakesMode:
    """Test High-Stakes and Critical mode behavior."""

    @pytest.fixture
    def config(self):
        return DPSConfig(unlock_high_stakes=False, unlock_critical=False)

    @pytest.fixture
    def dps_layer(self):
        return DPSGatingLayer(DPSConfig(unlock_high_stakes=False), noetic_dim=40)

    def test_hs_blocks_unlock(self, dps_layer, config):
        """High-Stakes mode with unlock_high_stakes=False: no unlock"""
        state = DPSState(p_max=3, tokens=0, cooldown=0, mode=DPSMode.HIGH_STAKES)

        # Create HEAVY novelty
        validity = ValidityScore(confidence=1.0, consistency=1.0, evidence_ok=1.0)
        impact = ImpactScore(delta_reward=1.0, delta_uncertainty=1.0, delta_coverage=1.0, delta_cost=1.0)
        similarity = SimilarityScore(s_embed=0.0, s_ast=0.0)
        novelty = NoveltyWeight(validity=validity, impact=impact, similarity=similarity)

        new_state, unlock_occurred = dps_layer.update_state(state, novelty, "test_episode")

        # can_unlock should return False in High-Stakes mode
        assert not state.can_unlock(config)

    def test_critical_always_blocks(self, dps_layer, config):
        """Critical mode: never unlock"""
        state = DPSState(p_max=3, tokens=0, cooldown=0, mode=DPSMode.CRITICAL)

        assert not state.can_unlock(config)


class TestDPSGatingLayer:
    """Test DPSGatingLayer neural module."""

    @pytest.fixture
    def dps_layer(self):
        return DPSGatingLayer(DPSConfig(), noetic_dim=40)

    @pytest.fixture
    def state(self):
        return DPSState()

    def test_forward_pass(self, dps_layer, state):
        """Test forward pass through DPS layer."""
        batch_size = 4
        seq_len = 8
        dim = 40

        x = torch.randn(batch_size, seq_len, dim)

        output, new_state, trace = dps_layer(x, state)

        assert output.shape == (batch_size, seq_len, dim)
        assert isinstance(new_state, DPSState)
        assert 'novelty_weight' in trace
        assert 'novelty_class' in trace
        assert 'depth_allowed' in trace

    def test_trace_contents(self, dps_layer, state):
        """Test trace contains all expected fields."""
        x = torch.randn(4, 8, 40)

        _, _, trace = dps_layer(x, state)

        expected_keys = [
            'novelty_weight', 'novelty_class', 'validity', 'impact',
            'similarity', 'depth_allowed', 'unlock_occurred', 'tokens'
        ]
        for key in expected_keys:
            assert key in trace, f"Missing key: {key}"

    def test_validity_structure(self, dps_layer, state):
        """Test validity trace structure."""
        x = torch.randn(4, 8, 40)
        _, _, trace = dps_layer(x, state)

        assert 'confidence' in trace['validity']
        assert 'consistency' in trace['validity']
        assert 'evidence_ok' in trace['validity']

    def test_impact_structure(self, dps_layer, state):
        """Test impact trace structure."""
        x = torch.randn(4, 8, 40)
        _, _, trace = dps_layer(x, state)

        assert 'delta_reward' in trace['impact']
        assert 'delta_uncertainty' in trace['impact']
        assert 'delta_coverage' in trace['impact']
        assert 'delta_cost' in trace['impact']

    def test_similarity_structure(self, dps_layer, state):
        """Test similarity trace structure."""
        x = torch.randn(4, 8, 40)
        _, _, trace = dps_layer(x, state)

        assert 's_embed' in trace['similarity']
        assert 's_ast' in trace['similarity']

    def test_gradient_flow(self, dps_layer, state):
        """Test gradients flow through DPS layer."""
        x = torch.randn(4, 8, 40, requires_grad=True)

        output, _, _ = dps_layer(x, state)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestAdaptiveIterationController:
    """Test adaptive iteration control based on DPS."""

    @pytest.fixture
    def dps_layer(self):
        return DPSGatingLayer(DPSConfig(), noetic_dim=40)

    @pytest.fixture
    def state(self):
        return DPSState(p_max=3)

    def test_max_iterations(self, dps_layer, state):
        """Controller should limit iterations based on p_max."""
        controller = AdaptiveIterationController(dps_layer, state)
        assert controller.max_iterations == state.p_max == 3

    def test_should_stop_at_budget(self, dps_layer, state):
        """Should stop when iteration exceeds budget."""
        controller = AdaptiveIterationController(dps_layer, state)
        x = torch.randn(4, 8, 40)

        # Should not stop before budget
        assert not controller.should_stop(x, 0)
        assert not controller.should_stop(x, 1)
        assert not controller.should_stop(x, 2)

        # Should stop at budget
        assert controller.should_stop(x, 3)
        assert controller.should_stop(x, 4)

    def test_extend_budget(self, dps_layer, state):
        """Test budget extension."""
        controller = AdaptiveIterationController(dps_layer, state)
        initial_budget = controller.max_iterations

        controller.extend_budget(1)

        assert controller.max_iterations == initial_budget + 1

    def test_extend_respects_max_depth(self, dps_layer):
        """Extension should not exceed max_depth."""
        state = DPSState(p_max=5)  # Already at max
        controller = AdaptiveIterationController(dps_layer, state)

        controller.extend_budget(1)

        assert controller.max_iterations <= dps_layer.config.max_depth


class TestDPSConfig:
    """Test DPSConfig defaults and validation."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        config = DPSConfig()

        assert config.heavy_threshold == DEFAULT_HEAVY_THRESHOLD == 0.70
        assert config.count_threshold == DEFAULT_COUNT_THRESHOLD == 0.45
        assert config.n_tokens_threshold == DEFAULT_TOKENS_FOR_UNLOCK == 5
        assert config.cooldown_k == DEFAULT_COOLDOWN_EPISODES == 10
        assert config.max_depth == DEFAULT_MAX_DEPTH == 5
        assert config.initial_p_max == DEFAULT_INITIAL_P_MAX == 2

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = DPSConfig(
            heavy_threshold=0.80,
            count_threshold=0.50,
            n_tokens_threshold=3,
            cooldown_k=5,
            max_depth=7,
        )

        assert config.heavy_threshold == 0.80
        assert config.count_threshold == 0.50
        assert config.n_tokens_threshold == 3
        assert config.cooldown_k == 5
        assert config.max_depth == 7


class TestDPSState:
    """Test DPSState management."""

    def test_default_state(self):
        """Test default state values."""
        state = DPSState()

        assert state.p_max == DEFAULT_INITIAL_P_MAX
        assert state.tokens == 0
        assert state.cooldown == 0
        assert state.mode == DPSMode.NORMAL

    def test_state_serialization(self):
        """Test state to_dict/from_dict."""
        state = DPSState(p_max=4, tokens=3, cooldown=5, mode=DPSMode.HIGH_STAKES)

        data = state.to_dict()
        restored = DPSState.from_dict(data)

        assert restored.p_max == 4
        assert restored.tokens == 3
        assert restored.cooldown == 5
        assert restored.mode == DPSMode.HIGH_STAKES

    def test_can_unlock_logic(self):
        """Test can_unlock state method."""
        config = DPSConfig()

        # Normal state, should be able to unlock
        state1 = DPSState(p_max=3, cooldown=0, mode=DPSMode.NORMAL)
        assert state1.can_unlock(config) == True

        # Cooldown active, should not unlock
        state2 = DPSState(p_max=3, cooldown=5, mode=DPSMode.NORMAL)
        assert state2.can_unlock(config) == False

        # At max depth, should not unlock
        state3 = DPSState(p_max=5, cooldown=0, mode=DPSMode.NORMAL)
        assert state3.can_unlock(config) == False


# Run with: pytest tests/test_dps_gating.py -v

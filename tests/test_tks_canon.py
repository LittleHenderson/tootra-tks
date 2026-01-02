#!/usr/bin/env python3
"""
TKS Canon Compliance Unit Tests

This test suite verifies that all TKS implementations comply with
frozen canonical specifications.

FROZEN SPECIFICATIONS (IMMUTABLE):
    - MVR Protocol: Desire={1,4,7}, Wisdom={5,6}, Power={8,9}
    - 4 Worlds: A=Spiritual(0-9), B=Mental(10-19), C=Emotional(20-29), D=Physical(30-39)
    - World Opposites: A<->D, B<->C
    - Noetic Opposites (0-indexed): 1<->2, 4<->5, 7<->8 (0,3,6,9 self-dual)
    - Foundation Opposites: 1<->7, 2<->6, 3<->5 (4 self-dual)

Author: TKS Canon Reviewer
Date: 2025-12-22
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CANONICAL CONSTANTS (FROZEN - REFERENCE VALUES)
# =============================================================================

# MVR Protocol - FROZEN
CANONICAL_DESIRE_NOETICS = {1, 4, 7}   # Mind, Vibration, Rhythm
CANONICAL_WISDOM_NOETICS = {5, 6}       # Female, Male
CANONICAL_POWER_NOETICS = {8, 9}        # Cause, Effect

# World Structure - FROZEN
CANONICAL_WORLD_OFFSETS = {'A': 0, 'B': 10, 'C': 20, 'D': 30}
CANONICAL_NOETIC_DIM = 10
CANONICAL_NUM_WORLDS = 4
CANONICAL_TOTAL_DIM = 40

# World Slice Boundaries - FROZEN
CANONICAL_WORLD_SLICES = {
    'A': (0, 10),
    'B': (10, 20),
    'C': (20, 30),
    'D': (30, 40),
}

# Full 40D Indices - FROZEN
CANONICAL_DESIRE_INDICES = [1, 4, 7, 11, 14, 17, 21, 24, 27, 31, 34, 37]
CANONICAL_WISDOM_INDICES = [5, 6, 15, 16, 25, 26, 35, 36]
CANONICAL_POWER_INDICES = [8, 9, 18, 19, 28, 29, 38, 39]

# World Opposites - FROZEN
CANONICAL_WORLD_OPPOSITES = {'A': 'D', 'B': 'C', 'C': 'B', 'D': 'A'}

# Noetic Opposites (0-indexed) - FROZEN
# Based on inversion/engine.py canonical definition
CANONICAL_NOETIC_OPPOSITES = {
    0: 0,   # Idea - self-dual
    1: 2,   # Mind <-> Positive
    2: 1,   # Positive <-> Mind
    3: 3,   # Negative - self-dual
    4: 5,   # Vibration <-> Female
    5: 4,   # Female <-> Vibration
    6: 6,   # Male - self-dual
    7: 8,   # Rhythm <-> Above/Cause
    8: 7,   # Above/Cause <-> Rhythm
    9: 9,   # Below/Effect - self-dual
}

# Foundation Opposites - FROZEN
CANONICAL_FOUNDATION_OPPOSITES = {
    1: 7,   # Association <-> Continuation
    2: 6,   # Wisdom <-> Wealth
    3: 5,   # Life <-> Power
    4: 4,   # Companionship - self-dual
    5: 3,   # Power <-> Life
    6: 2,   # Wealth <-> Wisdom
    7: 1,   # Continuation <-> Association
}

# Noetic Names - FROZEN
CANONICAL_NOETIC_NAMES = [
    "Idea",       # 0
    "Mind",       # 1
    "Positive",   # 2
    "Negative",   # 3
    "Vibration",  # 4
    "Female",     # 5
    "Male",       # 6
    "Rhythm",     # 7
    "Above",      # 8 (Cause)
    "Below",      # 9 (Effect)
]


# =============================================================================
# TEST CLASS: World Slice Boundaries
# =============================================================================

class TestWorldSliceBoundaries:
    """Verify that world slice boundaries match canonical offsets."""

    def test_world_a_boundaries(self):
        """World A (Spiritual) should be indices 0-9."""
        start, end = CANONICAL_WORLD_SLICES['A']
        assert start == 0, f"World A start should be 0, got {start}"
        assert end == 10, f"World A end should be 10, got {end}"
        assert end - start == CANONICAL_NOETIC_DIM, "World A should span 10 dimensions"

    def test_world_b_boundaries(self):
        """World B (Mental) should be indices 10-19."""
        start, end = CANONICAL_WORLD_SLICES['B']
        assert start == 10, f"World B start should be 10, got {start}"
        assert end == 20, f"World B end should be 20, got {end}"
        assert end - start == CANONICAL_NOETIC_DIM, "World B should span 10 dimensions"

    def test_world_c_boundaries(self):
        """World C (Emotional) should be indices 20-29."""
        start, end = CANONICAL_WORLD_SLICES['C']
        assert start == 20, f"World C start should be 20, got {start}"
        assert end == 30, f"World C end should be 30, got {end}"
        assert end - start == CANONICAL_NOETIC_DIM, "World C should span 10 dimensions"

    def test_world_d_boundaries(self):
        """World D (Physical) should be indices 30-39."""
        start, end = CANONICAL_WORLD_SLICES['D']
        assert start == 30, f"World D start should be 30, got {start}"
        assert end == 40, f"World D end should be 40, got {end}"
        assert end - start == CANONICAL_NOETIC_DIM, "World D should span 10 dimensions"

    def test_world_slices_are_contiguous(self):
        """All world slices should be contiguous and cover full 40D space."""
        worlds = ['A', 'B', 'C', 'D']
        prev_end = 0
        for w in worlds:
            start, end = CANONICAL_WORLD_SLICES[w]
            assert start == prev_end, f"Gap before World {w}: expected start {prev_end}, got {start}"
            prev_end = end
        assert prev_end == CANONICAL_TOTAL_DIM, f"Total should be 40, got {prev_end}"

    def test_world_offsets_match_slices(self):
        """World offsets should equal slice start indices."""
        for world, offset in CANONICAL_WORLD_OFFSETS.items():
            start, _ = CANONICAL_WORLD_SLICES[world]
            assert offset == start, f"World {world} offset {offset} != slice start {start}"


# =============================================================================
# TEST CLASS: MVR Protocol Noetics
# =============================================================================

class TestMVRProtocolNoetics:
    """Verify MVR Protocol noetic assignments are correct."""

    def test_desire_noetics_are_mvr(self):
        """Desire must use MVR noetics: Mind(1), Vibration(4), Rhythm(7)."""
        assert CANONICAL_DESIRE_NOETICS == {1, 4, 7}, \
            f"Desire noetics should be {{1,4,7}}, got {CANONICAL_DESIRE_NOETICS}"

    def test_wisdom_noetics_are_gender(self):
        """Wisdom must use gender noetics: Female(5), Male(6)."""
        assert CANONICAL_WISDOM_NOETICS == {5, 6}, \
            f"Wisdom noetics should be {{5,6}}, got {CANONICAL_WISDOM_NOETICS}"

    def test_power_noetics_are_causation(self):
        """Power must use causation noetics: Cause(8), Effect(9)."""
        assert CANONICAL_POWER_NOETICS == {8, 9}, \
            f"Power noetics should be {{8,9}}, got {CANONICAL_POWER_NOETICS}"

    def test_mvr_noetics_are_disjoint(self):
        """D/W/P noetic sets must be mutually disjoint."""
        assert CANONICAL_DESIRE_NOETICS.isdisjoint(CANONICAL_WISDOM_NOETICS), \
            "Desire and Wisdom noetics should not overlap"
        assert CANONICAL_DESIRE_NOETICS.isdisjoint(CANONICAL_POWER_NOETICS), \
            "Desire and Power noetics should not overlap"
        assert CANONICAL_WISDOM_NOETICS.isdisjoint(CANONICAL_POWER_NOETICS), \
            "Wisdom and Power noetics should not overlap"

    def test_mvr_noetics_within_range(self):
        """All MVR noetics must be within 0-9 range."""
        all_mvr = CANONICAL_DESIRE_NOETICS | CANONICAL_WISDOM_NOETICS | CANONICAL_POWER_NOETICS
        for n in all_mvr:
            assert 0 <= n <= 9, f"Noetic {n} outside valid range 0-9"

    def test_desire_noetics_count(self):
        """Desire should have exactly 3 noetics (MVR triplet)."""
        assert len(CANONICAL_DESIRE_NOETICS) == 3

    def test_wisdom_noetics_count(self):
        """Wisdom should have exactly 2 noetics (gender pair)."""
        assert len(CANONICAL_WISDOM_NOETICS) == 2

    def test_power_noetics_count(self):
        """Power should have exactly 2 noetics (causation pair)."""
        assert len(CANONICAL_POWER_NOETICS) == 2


# =============================================================================
# TEST CLASS: Full 40D Indices
# =============================================================================

class TestFull40DIndices:
    """Verify full 40D indices for D/W/P categories."""

    def test_desire_indices_across_worlds(self):
        """Desire indices should be MVR noetics in all 4 worlds."""
        expected = []
        for offset in [0, 10, 20, 30]:
            for n in [1, 4, 7]:
                expected.append(offset + n)
        assert CANONICAL_DESIRE_INDICES == expected, \
            f"Desire indices mismatch: expected {expected}, got {CANONICAL_DESIRE_INDICES}"

    def test_wisdom_indices_across_worlds(self):
        """Wisdom indices should be gender noetics in all 4 worlds."""
        expected = []
        for offset in [0, 10, 20, 30]:
            for n in [5, 6]:
                expected.append(offset + n)
        assert CANONICAL_WISDOM_INDICES == expected, \
            f"Wisdom indices mismatch: expected {expected}, got {CANONICAL_WISDOM_INDICES}"

    def test_power_indices_across_worlds(self):
        """Power indices should be causation noetics in all 4 worlds."""
        expected = []
        for offset in [0, 10, 20, 30]:
            for n in [8, 9]:
                expected.append(offset + n)
        assert CANONICAL_POWER_INDICES == expected, \
            f"Power indices mismatch: expected {expected}, got {CANONICAL_POWER_INDICES}"

    def test_desire_indices_count(self):
        """Desire should have 12 indices (3 noetics x 4 worlds)."""
        assert len(CANONICAL_DESIRE_INDICES) == 12

    def test_wisdom_indices_count(self):
        """Wisdom should have 8 indices (2 noetics x 4 worlds)."""
        assert len(CANONICAL_WISDOM_INDICES) == 8

    def test_power_indices_count(self):
        """Power should have 8 indices (2 noetics x 4 worlds)."""
        assert len(CANONICAL_POWER_INDICES) == 8

    def test_all_indices_within_40d(self):
        """All D/W/P indices must be in [0, 39] range."""
        all_indices = CANONICAL_DESIRE_INDICES + CANONICAL_WISDOM_INDICES + CANONICAL_POWER_INDICES
        for idx in all_indices:
            assert 0 <= idx < 40, f"Index {idx} outside valid range 0-39"


# =============================================================================
# TEST CLASS: Noetic Opposition Mappings
# =============================================================================

class TestNoeticOppositionMappings:
    """Verify noetic opposition mappings are consistent and correct."""

    def test_opposition_is_involutive(self):
        """Noetic opposition must be involutive: f(f(x)) = x."""
        for n, opp in CANONICAL_NOETIC_OPPOSITES.items():
            double_opp = CANONICAL_NOETIC_OPPOSITES[opp]
            assert double_opp == n, \
                f"Noetic opposition not involutive: f(f({n})) = {double_opp} != {n}"

    def test_self_dual_noetics(self):
        """Certain noetics must be self-dual (map to themselves)."""
        self_dual = [0, 3, 6, 9]  # Idea, Negative, Male, Below
        for n in self_dual:
            assert CANONICAL_NOETIC_OPPOSITES[n] == n, \
                f"Noetic {n} should be self-dual but maps to {CANONICAL_NOETIC_OPPOSITES[n]}"

    def test_opposition_pairs(self):
        """Verify specific opposition pairs match canonical definition."""
        pairs = [(1, 2), (4, 5), (7, 8)]  # Mind<->Positive, Vibration<->Female, Rhythm<->Above
        for a, b in pairs:
            assert CANONICAL_NOETIC_OPPOSITES[a] == b, \
                f"Noetic {a} should map to {b}, got {CANONICAL_NOETIC_OPPOSITES[a]}"
            assert CANONICAL_NOETIC_OPPOSITES[b] == a, \
                f"Noetic {b} should map to {a}, got {CANONICAL_NOETIC_OPPOSITES[b]}"

    def test_opposition_covers_all_noetics(self):
        """Opposition mapping must cover all 10 noetics."""
        assert set(CANONICAL_NOETIC_OPPOSITES.keys()) == set(range(10)), \
            "Opposition mapping must cover noetics 0-9"


# =============================================================================
# TEST CLASS: World Opposition Mappings
# =============================================================================

class TestWorldOppositionMappings:
    """Verify world opposition mappings are correct."""

    def test_world_opposition_is_involutive(self):
        """World opposition must be involutive: f(f(w)) = w."""
        for w, opp in CANONICAL_WORLD_OPPOSITES.items():
            double_opp = CANONICAL_WORLD_OPPOSITES[opp]
            assert double_opp == w, \
                f"World opposition not involutive: f(f({w})) = {double_opp} != {w}"

    def test_spiritual_physical_opposition(self):
        """A (Spiritual) must oppose D (Physical)."""
        assert CANONICAL_WORLD_OPPOSITES['A'] == 'D'
        assert CANONICAL_WORLD_OPPOSITES['D'] == 'A'

    def test_mental_emotional_opposition(self):
        """B (Mental) must oppose C (Emotional)."""
        assert CANONICAL_WORLD_OPPOSITES['B'] == 'C'
        assert CANONICAL_WORLD_OPPOSITES['C'] == 'B'

    def test_no_self_dual_worlds(self):
        """No world should be self-dual."""
        for w, opp in CANONICAL_WORLD_OPPOSITES.items():
            assert w != opp, f"World {w} should not be self-dual"


# =============================================================================
# TEST CLASS: Foundation Opposition Mappings
# =============================================================================

class TestFoundationOppositionMappings:
    """Verify foundation opposition mappings are correct."""

    def test_foundation_opposition_is_involutive(self):
        """Foundation opposition must be involutive: f(f(x)) = x."""
        for f, opp in CANONICAL_FOUNDATION_OPPOSITES.items():
            double_opp = CANONICAL_FOUNDATION_OPPOSITES[opp]
            assert double_opp == f, \
                f"Foundation opposition not involutive: f(f({f})) = {double_opp} != {f}"

    def test_foundation_opposition_pairs(self):
        """Verify specific foundation opposition pairs."""
        pairs = [(1, 7), (2, 6), (3, 5)]
        for a, b in pairs:
            assert CANONICAL_FOUNDATION_OPPOSITES[a] == b, \
                f"Foundation {a} should oppose {b}"
            assert CANONICAL_FOUNDATION_OPPOSITES[b] == a, \
                f"Foundation {b} should oppose {a}"

    def test_companionship_is_self_dual(self):
        """Foundation 4 (Companionship) must be self-dual."""
        assert CANONICAL_FOUNDATION_OPPOSITES[4] == 4, \
            "Foundation 4 (Companionship) should be self-dual"

    def test_foundation_opposition_covers_all(self):
        """Opposition mapping must cover all 7 foundations."""
        assert set(CANONICAL_FOUNDATION_OPPOSITES.keys()) == set(range(1, 8)), \
            "Foundation opposition must cover foundations 1-7"


# =============================================================================
# TEST CLASS: Integration with training/losses.py
# =============================================================================

class TestLossesCanonCompliance:
    """Verify training/losses.py uses canonical values."""

    def test_world_classification_loss_slices(self):
        """WorldClassificationLoss must use correct world slices."""
        try:
            from training.losses import WorldClassificationLoss
            loss = WorldClassificationLoss()

            # Verify slice boundaries
            assert loss.world_slices[0] == slice(0, 10), "World A slice incorrect"
            assert loss.world_slices[1] == slice(10, 20), "World B slice incorrect"
            assert loss.world_slices[2] == slice(20, 30), "World C slice incorrect"
            assert loss.world_slices[3] == slice(30, 40), "World D slice incorrect"
        except ImportError:
            pytest.skip("training.losses not available")

    def test_rpm_differentiation_loss_noetics(self):
        """RPMDifferentiationLoss must use canonical MVR noetics."""
        try:
            from training.losses import RPMDifferentiationLoss
            loss = RPMDifferentiationLoss()

            assert set(loss.desire_noetics) == CANONICAL_DESIRE_NOETICS, \
                f"Desire noetics mismatch: {loss.desire_noetics}"
            assert set(loss.wisdom_noetics) == CANONICAL_WISDOM_NOETICS, \
                f"Wisdom noetics mismatch: {loss.wisdom_noetics}"
            assert set(loss.power_noetics) == CANONICAL_POWER_NOETICS, \
                f"Power noetics mismatch: {loss.power_noetics}"
        except ImportError:
            pytest.skip("training.losses not available")

    def test_cascade_loss_world_order(self):
        """CascadeLoss must enforce A->B->C->D cascade."""
        try:
            from training.losses import CascadeLoss
            import torch

            loss = CascadeLoss()
            # Create test embedding
            noetic_emb = torch.randn(2, 8, 40)
            result = loss(noetic_emb)

            # Should have world order keys
            assert 'energy_order' in result or 'energy_A_B' in result, \
                "CascadeLoss should track world energy ordering"
        except ImportError:
            pytest.skip("training.losses not available")


# =============================================================================
# TEST CLASS: Integration with tks_llm_core_v2.py
# =============================================================================

class TestTKSLLMCoreV2CanonCompliance:
    """Verify tks_llm_core_v2.py uses canonical values."""

    def test_dwp_indices_computation(self):
        """D/W/P indices must match canonical computation."""
        try:
            from tks_llm_core_v2 import DESIRE_INDICES, WISDOM_INDICES, POWER_INDICES

            assert list(DESIRE_INDICES) == CANONICAL_DESIRE_INDICES, \
                f"DESIRE_INDICES mismatch: {list(DESIRE_INDICES)}"
            assert list(WISDOM_INDICES) == CANONICAL_WISDOM_INDICES, \
                f"WISDOM_INDICES mismatch: {list(WISDOM_INDICES)}"
            assert list(POWER_INDICES) == CANONICAL_POWER_INDICES, \
                f"POWER_INDICES mismatch: {list(POWER_INDICES)}"
        except ImportError:
            pytest.skip("tks_llm_core_v2 not available")

    def test_rpm_gating_uses_canonical_indices(self):
        """RPMGatingMechanism must use canonical D/W/P indices."""
        try:
            from tks_llm_core_v2 import RPMGatingMechanism
            import torch

            rpm = RPMGatingMechanism()

            # Check registered buffers
            assert list(rpm.desire_idx.tolist()) == CANONICAL_DESIRE_INDICES, \
                "RPMGatingMechanism desire_idx incorrect"
            assert list(rpm.wisdom_idx.tolist()) == CANONICAL_WISDOM_INDICES, \
                "RPMGatingMechanism wisdom_idx incorrect"
            assert list(rpm.power_idx.tolist()) == CANONICAL_POWER_INDICES, \
                "RPMGatingMechanism power_idx incorrect"
        except ImportError:
            pytest.skip("tks_llm_core_v2 not available")

    def test_stable_attractor_preserves_40d(self):
        """StableAttractorLayer must preserve 40D noetic structure."""
        try:
            from tks_llm_core_v2 import StableAttractorLayer
            import torch

            attractor = StableAttractorLayer(dim=CANONICAL_TOTAL_DIM)
            x = torch.randn(2, 8, CANONICAL_TOTAL_DIM)
            result = attractor(x)

            assert result['attractor'].shape == (2, 8, CANONICAL_TOTAL_DIM), \
                f"Attractor output shape should be (2,8,40), got {result['attractor'].shape}"
        except ImportError:
            pytest.skip("tks_llm_core_v2 not available")


# =============================================================================
# TEST CLASS: Integration with scripts/generate_world_rpm_labels.py
# =============================================================================

class TestGenerateWorldRPMLabelsCanonCompliance:
    """Verify generate_world_rpm_labels.py uses canonical values."""

    def test_mvr_noetics_definition(self):
        """Script must define correct MVR noetics."""
        try:
            from scripts.generate_world_rpm_labels import (
                DESIRE_NOETICS, WISDOM_NOETICS, POWER_NOETICS
            )

            assert set(DESIRE_NOETICS) == CANONICAL_DESIRE_NOETICS, \
                f"DESIRE_NOETICS mismatch: {DESIRE_NOETICS}"
            assert set(WISDOM_NOETICS) == CANONICAL_WISDOM_NOETICS, \
                f"WISDOM_NOETICS mismatch: {WISDOM_NOETICS}"
            assert set(POWER_NOETICS) == CANONICAL_POWER_NOETICS, \
                f"POWER_NOETICS mismatch: {POWER_NOETICS}"
        except ImportError:
            pytest.skip("scripts.generate_world_rpm_labels not available")

    def test_worlds_definition(self):
        """Script must define correct world structure."""
        try:
            from scripts.generate_world_rpm_labels import WORLDS

            assert 'A' in WORLDS and WORLDS['A']['domain'] == 'Spiritual', \
                "World A should be Spiritual"
            assert 'B' in WORLDS and WORLDS['B']['domain'] == 'Mental', \
                "World B should be Mental"
            assert 'C' in WORLDS and WORLDS['C']['domain'] == 'Emotional', \
                "World C should be Emotional"
            assert 'D' in WORLDS and WORLDS['D']['domain'] == 'Physical', \
                "World D should be Physical"
        except ImportError:
            pytest.skip("scripts.generate_world_rpm_labels not available")


# =============================================================================
# TEST CLASS: Cross-File Consistency
# =============================================================================

class TestCrossFileConsistency:
    """Verify consistency across multiple files."""

    def test_desire_noetics_consistent_everywhere(self):
        """Desire noetics must be consistent across all files."""
        sources = {}

        # Check training/losses.py
        try:
            from training.losses import RPMDifferentiationLoss
            loss = RPMDifferentiationLoss()
            sources['training/losses.py'] = set(loss.desire_noetics)
        except ImportError:
            pass

        # Check tks_llm_core_v2.py
        try:
            from tks_llm_core_v2 import DESIRE_INDICES
            # Extract base noetics from full indices
            base_noetics = set(idx % 10 for idx in DESIRE_INDICES)
            sources['tks_llm_core_v2.py'] = base_noetics
        except ImportError:
            pass

        # Check scripts/generate_world_rpm_labels.py
        try:
            from scripts.generate_world_rpm_labels import DESIRE_NOETICS
            sources['scripts/generate_world_rpm_labels.py'] = set(DESIRE_NOETICS)
        except ImportError:
            pass

        # All should match canonical
        for source, noetics in sources.items():
            assert noetics == CANONICAL_DESIRE_NOETICS, \
                f"Desire noetics in {source} = {noetics}, expected {CANONICAL_DESIRE_NOETICS}"

    def test_world_offsets_consistent_everywhere(self):
        """World offsets must be consistent across all files."""
        # This test checks that any file using world offsets uses the canonical values
        try:
            from tks_llm_core import WORLD_OFFSETS
            assert WORLD_OFFSETS == CANONICAL_WORLD_OFFSETS, \
                f"WORLD_OFFSETS mismatch: {WORLD_OFFSETS}"
        except ImportError:
            pass

        try:
            from tks_llm_core_v2 import DESIRE_INDICES
            # Verify world structure from indices
            assert min(DESIRE_INDICES) >= 0
            assert max(DESIRE_INDICES) < 40
        except ImportError:
            pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TKS CANON COMPLIANCE TEST SUITE")
    print("=" * 70)
    print()
    print("This test suite verifies compliance with frozen TKS specifications.")
    print("Any test failure indicates a canon violation that MUST be fixed.")
    print()
    print("Running tests...")
    print()

    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])

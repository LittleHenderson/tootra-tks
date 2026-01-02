"""
TKS Canonical RPM (Desire/Wisdom/Power) Module

Defines the RPM protocol mapping noetics to D/W/P components.

CRITICAL: This uses the MVR (Mind-Vibration-Rhythm) protocol for Desire:
  - Desire: {1, 4, 7} - Mind, Vibration, Rhythm (MVR)
  - Wisdom: {5, 6} - Female, Male
  - Power: {8, 9} - Cause, Effect

The MVR protocol was explicitly specified as canonical for this codebase.
"""

from typing import Dict, Set, FrozenSet, List, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class RPMComponent:
    """Definition of an RPM component."""
    name: str                    # Desire, Wisdom, Power
    short: str                   # D, W, P
    noetics: FrozenSet[int]      # Associated noetic indices
    noetic_names: Tuple[str, ...]  # Names of associated noetics
    description: str             # What this component represents
    card: str                    # Playing card association (Kings, Jacks, Queens)


# =============================================================================
# MVR PROTOCOL - CANONICAL RPM MAPPINGS
# =============================================================================

# These are the USER-SPECIFIED canonical mappings using the MVR protocol
# MVR = Mind, Vibration, Rhythm (noetics 1, 4, 7)

RPM_DESIRE_NOETICS: FrozenSet[int] = frozenset({1, 4, 7})  # Mind, Vibration, Rhythm
RPM_WISDOM_NOETICS: FrozenSet[int] = frozenset({5, 6})     # Female, Male
RPM_POWER_NOETICS: FrozenSet[int] = frozenset({8, 9})      # Cause, Effect

# Sets for easy membership testing
RPM_DESIRE: Set[int] = {1, 4, 7}
RPM_WISDOM: Set[int] = {5, 6}
RPM_POWER: Set[int] = {8, 9}

# Complete RPM components
RPM_COMPONENTS: Dict[str, RPMComponent] = {
    "desire": RPMComponent(
        name="Desire",
        short="D",
        noetics=RPM_DESIRE_NOETICS,
        noetic_names=("Mind", "Vibration", "Rhythm"),
        description="The initial impulse, want, will - MVR (Mind-Vibration-Rhythm)",
        card="Kings"
    ),
    "wisdom": RPMComponent(
        name="Wisdom",
        short="W",
        noetics=RPM_WISDOM_NOETICS,
        noetic_names=("Female", "Male"),
        description="The know-how, understanding, blueprint - receptivity and structure",
        card="Jacks"
    ),
    "power": RPMComponent(
        name="Power",
        short="P",
        noetics=RPM_POWER_NOETICS,
        noetic_names=("Cause", "Effect"),
        description="The ability, capacity, capability - causation chain",
        card="Queens"
    ),
}

# Dictionary format for compatibility with existing code
RPM_MAPPINGS: Dict[str, Set[int]] = {
    "desire": RPM_DESIRE,
    "wisdom": RPM_WISDOM,
    "power": RPM_POWER,
}


# =============================================================================
# RPM INDICES FOR NEURAL NETWORKS (4 worlds x noetics per component)
# =============================================================================

def _compute_rpm_indices(noetic_set: Set[int], num_worlds: int = 4) -> List[int]:
    """
    Compute linear indices for RPM components across 4 worlds.

    Formula: index = world_offset + noetic
    Where world_offset = world_num * 10

    Note: Uses element numbering (1-10), not noetic indices (0-9)
    """
    indices = []
    for world_num in range(num_worlds):
        world_offset = world_num * 10
        for noetic in sorted(noetic_set):
            # Element numbering: noetics 1-9 map to elements 1-9
            indices.append(world_offset + noetic)
    return sorted(indices)


# Pre-computed indices for neural network layers
DESIRE_INDICES: List[int] = _compute_rpm_indices(RPM_DESIRE)  # [1,4,7,11,14,17,21,24,27,31,34,37]
WISDOM_INDICES: List[int] = _compute_rpm_indices(RPM_WISDOM)  # [5,6,15,16,25,26,35,36]
POWER_INDICES: List[int] = _compute_rpm_indices(RPM_POWER)    # [8,9,18,19,28,29,38,39]


# =============================================================================
# RPM HELPER FUNCTIONS
# =============================================================================

def get_rpm_component(noetic_idx: int) -> str:
    """
    Get the RPM component for a noetic index.

    Returns 'desire', 'wisdom', 'power', or 'none'.
    """
    if noetic_idx in RPM_DESIRE:
        return "desire"
    elif noetic_idx in RPM_WISDOM:
        return "wisdom"
    elif noetic_idx in RPM_POWER:
        return "power"
    return "none"


def compute_dwp_scores(noetic_distribution: Dict[int, float]) -> Dict[str, float]:
    """
    Compute D/W/P scores from a noetic distribution.

    Args:
        noetic_distribution: Dict mapping noetic index (0-9) to weight/probability

    Returns:
        Dict with 'desire', 'wisdom', 'power' scores
    """
    scores = {"desire": 0.0, "wisdom": 0.0, "power": 0.0}

    for noetic_idx, weight in noetic_distribution.items():
        component = get_rpm_component(noetic_idx)
        if component != "none":
            scores[component] += weight

    return scores


def validate_rpm_noetics() -> bool:
    """Validate that RPM noetics don't overlap and cover expected range."""
    all_rpm = RPM_DESIRE | RPM_WISDOM | RPM_POWER

    # Check no overlap
    if len(all_rpm) != len(RPM_DESIRE) + len(RPM_WISDOM) + len(RPM_POWER):
        return False

    # RPM should cover noetics 1,4,5,6,7,8,9 (not 0,2,3)
    expected = {1, 4, 5, 6, 7, 8, 9}
    return all_rpm == expected


# =============================================================================
# GATE FORMULA
# =============================================================================

# Manifestation requires D x W x P all present
# For any Foundation F_n, stable manifestation requires:
#   D(n) >= threshold
#   W(n) >= threshold
#   P(n) >= threshold

DEFAULT_GATE_THRESHOLD: float = 0.75

def check_gate_open(
    desire_score: float,
    wisdom_score: float,
    power_score: float,
    threshold: float = DEFAULT_GATE_THRESHOLD
) -> bool:
    """
    Check if the manifestation gate is open.

    Gate opens when ALL three components meet threshold.
    """
    return (
        desire_score >= threshold and
        wisdom_score >= threshold and
        power_score >= threshold
    )


def get_gate_status(
    desire_score: float,
    wisdom_score: float,
    power_score: float,
    threshold: float = DEFAULT_GATE_THRESHOLD
) -> Dict[str, any]:
    """
    Get detailed gate status with component breakdown.
    """
    return {
        "open": check_gate_open(desire_score, wisdom_score, power_score, threshold),
        "threshold": threshold,
        "components": {
            "desire": {"score": desire_score, "meets_threshold": desire_score >= threshold},
            "wisdom": {"score": wisdom_score, "meets_threshold": wisdom_score >= threshold},
            "power": {"score": power_score, "meets_threshold": power_score >= threshold},
        },
        "blocking": [
            comp for comp, data in {
                "desire": desire_score,
                "wisdom": wisdom_score,
                "power": power_score,
            }.items() if data < threshold
        ]
    }


__all__ = [
    "RPMComponent",
    "RPM_COMPONENTS",
    "RPM_MAPPINGS",
    "RPM_DESIRE",
    "RPM_WISDOM",
    "RPM_POWER",
    "RPM_DESIRE_NOETICS",
    "RPM_WISDOM_NOETICS",
    "RPM_POWER_NOETICS",
    "DESIRE_INDICES",
    "WISDOM_INDICES",
    "POWER_INDICES",
    "DEFAULT_GATE_THRESHOLD",
    "get_rpm_component",
    "compute_dwp_scores",
    "validate_rpm_noetics",
    "check_gate_open",
    "get_gate_status",
]

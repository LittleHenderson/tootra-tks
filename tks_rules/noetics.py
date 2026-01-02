"""
TKS Canonical Noetics Definition Module

Defines the 10 Noetics (nu_0 through nu_9) as per TKS v7.4 canonical documents.

CRITICAL: Noetics are 0-indexed (nu_0 through nu_9)
         Elements use 1-10 notation (A1-D10)

The noetics are consciousness operators that modify Ideas, Elements, and Foundations.
They are NOT standalone values - they MODIFY other values.
"""

from typing import Dict, List, Set, Tuple, FrozenSet
from dataclasses import dataclass


@dataclass(frozen=True)
class NoeticDefinition:
    """Immutable definition of a single noetic."""
    index: int           # 0-9
    symbol: str          # nu_0, nu_1, etc.
    name: str            # Canonical name
    role: str            # Functional role
    aliases: Tuple[str, ...]  # Alternative names


# =============================================================================
# CANONICAL 10 NOETICS (nu_0 through nu_9)
# =============================================================================

NOETICS: Dict[int, NoeticDefinition] = {
    0: NoeticDefinition(
        index=0,
        symbol="nu_0",
        name="Idea",
        role="Identity/Content - The content itself; any 'thing' inside or outside the Mind",
        aliases=("The All", "Divine Idea", "Content")
    ),
    1: NoeticDefinition(
        index=1,
        symbol="nu_1",
        name="Mind",
        role="Awareness/Processor - Various degrees of awareness that process, store, output, and take in Ideas",
        aliases=("Awareness", "Consciousness", "Mentality")
    ),
    2: NoeticDefinition(
        index=2,
        symbol="nu_2",
        name="Positive",
        role="Attraction - An Idea the Mind has affinity for and is attracted toward",
        aliases=("Attraction", "Affinity")
    ),
    3: NoeticDefinition(
        index=3,
        symbol="nu_3",
        name="Negative",
        role="Repulsion - An Idea the Mind is averse to and repelled by",
        aliases=("Repulsion", "Aversion")
    ),
    4: NoeticDefinition(
        index=4,
        symbol="nu_4",
        name="Vibration",
        role="Resonance - The degree to which an Idea resonates with the Mind",
        aliases=("Resonance", "Energy", "Frequency")
    ),
    5: NoeticDefinition(
        index=5,
        symbol="nu_5",
        name="Female",
        role="Receptivity - A Mind impregnated with an Idea; the Ideas a Mind is composed of",
        aliases=("Feminine", "Receptivity", "Yin")
    ),
    6: NoeticDefinition(
        index=6,
        symbol="nu_6",
        name="Male",
        role="Structure - An Idea impregnated with another Idea; the Ideas an Idea is composed of",
        aliases=("Masculine", "Projection", "Yang")
    ),
    7: NoeticDefinition(
        index=7,
        symbol="nu_7",
        name="Rhythm",
        role="Timing/Repetition - The rate and pattern by which a Mind is exposed to an Idea",
        aliases=("Timing", "Pattern", "Cycle")
    ),
    8: NoeticDefinition(
        index=8,
        symbol="nu_8",
        name="Above",
        role="Trigger/Cause - An Idea that triggers a different Idea than itself to come to Mind",
        aliases=("Cause", "Trigger", "Source")
    ),
    9: NoeticDefinition(
        index=9,
        symbol="nu_9",
        name="Below",
        role="Result/Effect - The Idea brought to Mind by an Above/Cause Idea or trigger",
        aliases=("Effect", "Result", "Consequence")
    ),
}

# Quick lookup dictionaries
NOETIC_NAMES: Dict[int, str] = {k: v.name for k, v in NOETICS.items()}
NOETIC_BY_NAME: Dict[str, int] = {v.name: k for k, v in NOETICS.items()}

# Add aliases to reverse lookup
for idx, noetic in NOETICS.items():
    for alias in noetic.aliases:
        if alias not in NOETIC_BY_NAME:
            NOETIC_BY_NAME[alias] = idx


# =============================================================================
# INVOLUTION PAIRS (i + j = 9)
# =============================================================================

INVOLUTION_PAIRS: List[Tuple[int, int]] = [
    (2, 7),  # Positive <-> Rhythm
    (3, 6),  # Negative <-> Male
    (4, 5),  # Vibration <-> Female
]

# Self-dual noetics (their own complement)
SELF_DUAL_NOETICS: FrozenSet[int] = frozenset({0, 1})

# Special pair (not sum-to-9 but architecturally significant)
SPECIAL_PAIR: Tuple[int, int] = (8, 9)  # Above/Cause <-> Below/Effect


def get_complement(noetic_idx: int) -> int:
    """
    Get the complement noetic (i + j = 9).

    Self-dual noetics (0, 1) return themselves.
    """
    if noetic_idx in SELF_DUAL_NOETICS:
        return noetic_idx

    for pair in INVOLUTION_PAIRS:
        if noetic_idx in pair:
            return pair[1] if pair[0] == noetic_idx else pair[0]

    # For 8 and 9, they complement each other in a special way
    if noetic_idx == 8:
        return 9
    if noetic_idx == 9:
        return 8

    raise ValueError(f"Unknown noetic index: {noetic_idx}")


def validate_noetic(idx: int) -> bool:
    """Check if an index is a valid noetic (0-9)."""
    return idx in NOETICS


# =============================================================================
# MVR FRACTAL (Canonical Stabilizing Pattern)
# =============================================================================

MVR_FRACTAL: Tuple[int, int, int] = (1, 4, 7)  # Mind, Vibration, Rhythm

# The MVR fractal represents a harmonizing/stabilizing pattern
# Application order: right-to-left (innermost first)
# MVR(X) = nu_1(nu_4(nu_7(X)))


# =============================================================================
# SENSES (Noetic modifiers for elements)
# =============================================================================

# Senses are noetic modifiers applied via superscript notation (^1 through ^9)
# They modify how an element is perceived/experienced
SENSES: List[int] = list(range(1, 10))  # 1-9

# Self-dual noetics under MVR composition (these compose to themselves)
# Different from SELF_DUAL_NOETICS which refers to complement pairs
SELF_DUAL_MVR: List[int] = [1, 4, 7, 10]  # Mind, Vibration, Rhythm, Idea


__all__ = [
    "NoeticDefinition",
    "NOETICS",
    "NOETIC_NAMES",
    "NOETIC_BY_NAME",
    "INVOLUTION_PAIRS",
    "SELF_DUAL_NOETICS",
    "SELF_DUAL_MVR",
    "SPECIAL_PAIR",
    "MVR_FRACTAL",
    "SENSES",
    "get_complement",
    "validate_noetic",
]

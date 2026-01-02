"""
TKS Canonical Worlds Definition Module

Defines the 4 Worlds (A, B, C, D) as per TKS v7.4 canonical documents.

World Cascade: A -> B -> C -> D (archetypal to concrete)
- A (Atziluth): Spiritual/Archetypal
- B (Briah): Mental/Intellectual
- C (Yetzirah): Emotional/Formative
- D (Assiah): Physical/Material
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class WorldDefinition:
    """Immutable definition of a single world."""
    letter: str          # A, B, C, D
    name: str            # Hebrew name
    domain: str          # Domain type
    ordinal: int         # Order in cascade (0-3)
    keywords: Tuple[str, ...]  # Associated concepts


# =============================================================================
# CANONICAL 4 WORLDS
# =============================================================================

WORLDS: Dict[str, WorldDefinition] = {
    "A": WorldDefinition(
        letter="A",
        name="Atziluth",
        domain="Spiritual",
        ordinal=0,
        keywords=("Archetypal", "Divine", "Source", "Soul", "Eternal", "Purpose", "Meaning")
    ),
    "B": WorldDefinition(
        letter="B",
        name="Briah",
        domain="Mental",
        ordinal=1,
        keywords=("Creative", "Intellectual", "Thoughts", "Beliefs", "Concepts", "Analysis", "Logic")
    ),
    "C": WorldDefinition(
        letter="C",
        name="Yetzirah",
        domain="Emotional",
        ordinal=2,
        keywords=("Formative", "Astral", "Feelings", "Emotions", "Intuition", "Desires")
    ),
    "D": WorldDefinition(
        letter="D",
        name="Assiah",
        domain="Physical",
        ordinal=3,
        keywords=("Action", "Material", "Body", "Matter", "Actions", "Environment", "Concrete")
    ),
}

# Quick lookup
WORLD_LETTERS: List[str] = ["A", "B", "C", "D"]
WORLD_ORDINALS: Dict[str, int] = {w: d.ordinal for w, d in WORLDS.items()}
WORLD_NAMES: Dict[str, str] = {w: d.name for w, d in WORLDS.items()}
WORLD_DOMAINS: Dict[str, str] = {w: d.domain for w, d in WORLDS.items()}

# World offsets for 40-element indexing (10 noetics per world)
WORLD_OFFSETS: Dict[str, int] = {"A": 0, "B": 10, "C": 20, "D": 30}


# =============================================================================
# WORLD CASCADE RULES
# =============================================================================

WORLD_CASCADE_ORDER: Tuple[str, ...] = ("A", "B", "C", "D")

# Downward flow: spiritual principles -> physical manifestation
# Upward flow: physical changes -> spiritual transformation


def world_distance(w1: str, w2: str) -> int:
    """
    Calculate the distance between two worlds.

    d(W1, W2) = |ord(W1) - ord(W2)|
    """
    if w1 not in WORLDS or w2 not in WORLDS:
        raise ValueError(f"Invalid world: {w1} or {w2}")
    return abs(WORLDS[w1].ordinal - WORLDS[w2].ordinal)


def validate_world(letter: str) -> bool:
    """Check if a letter is a valid world."""
    return letter in WORLDS


def get_world_by_ordinal(ordinal: int) -> str:
    """Get world letter by ordinal (0-3)."""
    if not 0 <= ordinal <= 3:
        raise ValueError(f"Ordinal must be 0-3, got {ordinal}")
    return WORLD_CASCADE_ORDER[ordinal]


# Sub-foundation world letters (lowercase)
SUBFOUNDATION_WORLD_LETTERS: Dict[str, str] = {
    "a": "A",  # Atziluth
    "b": "B",  # Briah
    "c": "C",  # Yetzirah
    "d": "D",  # Assiah
}


__all__ = [
    "WorldDefinition",
    "WORLDS",
    "WORLD_LETTERS",
    "WORLD_ORDINALS",
    "WORLD_NAMES",
    "WORLD_DOMAINS",
    "WORLD_OFFSETS",
    "WORLD_CASCADE_ORDER",
    "SUBFOUNDATION_WORLD_LETTERS",
    "world_distance",
    "validate_world",
    "get_world_by_ordinal",
]

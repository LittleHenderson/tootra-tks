"""
TKS Canonical Foundations and Sub-Foundations Module

Defines:
- 7 Foundations (F1-F7)
- 28 Sub-Foundations (1a-7d)

CRITICAL: Sub-foundations use notation [Digit][Lowercase Letter]
         e.g., 1a, 3c, 7d

         This is DIFFERENT from Elements which use [Letter][Digit]
         e.g., A1, C3, D7
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class FoundationDefinition:
    """Immutable definition of a single foundation."""
    index: int              # 1-7
    name: str               # Canonical name
    metaphysical_name: str  # Alternative metaphysical name
    principle: str          # Core principle
    planet: str             # Associated planet
    day: str                # Associated day


@dataclass(frozen=True)
class SubFoundationDefinition:
    """Immutable definition of a single sub-foundation."""
    code: str               # 1a, 3c, etc.
    foundation: int         # 1-7
    world: str              # a, b, c, d
    world_upper: str        # A, B, C, D
    name: str               # Full name


# =============================================================================
# CANONICAL 7 FOUNDATIONS
# =============================================================================

FOUNDATIONS: Dict[int, FoundationDefinition] = {
    1: FoundationDefinition(
        index=1,
        name="Association",
        metaphysical_name="Unity",
        principle="Coherence and linking of Ideas across Worlds",
        planet="Sun",
        day="Sunday"
    ),
    2: FoundationDefinition(
        index=2,
        name="Wisdom",
        metaphysical_name="Wisdom",
        principle="Discernment, truth-orientation, structural understanding",
        planet="Moon",
        day="Monday"
    ),
    3: FoundationDefinition(
        index=3,
        name="Life",
        metaphysical_name="Life",
        principle="Vitality, persistence, and self-maintenance of patterns",
        planet="Mars",
        day="Tuesday"
    ),
    4: FoundationDefinition(
        index=4,
        name="Companionship",
        metaphysical_name="Companionship",
        principle="Relational bonding, mutual support, shared context",
        planet="Venus",
        day="Wednesday"
    ),
    5: FoundationDefinition(
        index=5,
        name="Power",
        metaphysical_name="Power",
        principle="Capacity to influence, direct, or constrain dynamics",
        planet="Jupiter",
        day="Thursday"
    ),
    6: FoundationDefinition(
        index=6,
        name="Wealth",
        metaphysical_name="Material",
        principle="Resources, structure, and embodied stability",
        planet="Saturn",
        day="Friday"
    ),
    7: FoundationDefinition(
        index=7,
        name="Continuation",
        metaphysical_name="Lust",
        principle="Drive to extend, propagate, and iterate patterns",
        planet="Saturn",  # Note: Saturn appears twice
        day="Saturday"
    ),
}

# Quick lookups
FOUNDATION_NAMES: Dict[int, str] = {k: v.name for k, v in FOUNDATIONS.items()}
FOUNDATION_BY_NAME: Dict[str, int] = {v.name: k for k, v in FOUNDATIONS.items()}

# Add metaphysical names to reverse lookup
for idx, found in FOUNDATIONS.items():
    if found.metaphysical_name not in FOUNDATION_BY_NAME:
        FOUNDATION_BY_NAME[found.metaphysical_name] = idx


# =============================================================================
# CANONICAL 28 SUB-FOUNDATIONS
# =============================================================================

SUBFOUNDATION_WORLDS: List[str] = ["a", "b", "c", "d"]
SUBFOUNDATION_WORLD_MAP: Dict[str, str] = {
    "a": "A",  # Atziluth (Spiritual)
    "b": "B",  # Briah (Mental)
    "c": "C",  # Yetzirah (Emotional)
    "d": "D",  # Assiah (Physical)
}


def _build_subfoundations() -> Dict[str, SubFoundationDefinition]:
    """Build all 28 sub-foundation definitions."""
    subfoundations = {}

    world_domains = {
        "a": "Spiritual",
        "b": "Mental",
        "c": "Emotional",
        "d": "Physical"
    }

    for f_idx in range(1, 8):
        f_name = FOUNDATIONS[f_idx].name
        for w in SUBFOUNDATION_WORLDS:
            code = f"{f_idx}{w}"
            domain = world_domains[w]

            subfoundations[code] = SubFoundationDefinition(
                code=code,
                foundation=f_idx,
                world=w,
                world_upper=SUBFOUNDATION_WORLD_MAP[w],
                name=f"{domain} {f_name}"
            )

    return subfoundations


SUB_FOUNDATIONS: Dict[str, SubFoundationDefinition] = _build_subfoundations()

# List for iteration
SUB_FOUNDATION_CODES: List[str] = [f"{f}{w}" for f in range(1, 8) for w in SUBFOUNDATION_WORLDS]


# =============================================================================
# PARSING AND VALIDATION
# =============================================================================

SUBFOUNDATION_PATTERN = re.compile(r'^([1-7])([abcd])$')


def parse_subfoundation(code: str) -> Optional[Tuple[int, str]]:
    """
    Parse a sub-foundation code into (foundation_num, world_letter).

    Returns None if invalid.
    """
    match = SUBFOUNDATION_PATTERN.match(code.lower())
    if not match:
        return None

    return (int(match.group(1)), match.group(2))


def validate_subfoundation(code: str) -> bool:
    """Check if a code is a valid sub-foundation."""
    return parse_subfoundation(code) is not None


def validate_foundation(idx: int) -> bool:
    """Check if an index is a valid foundation (1-7)."""
    return idx in FOUNDATIONS


def get_foundation_subfoundations(f_idx: int) -> List[str]:
    """Get all sub-foundations for a foundation."""
    if not validate_foundation(f_idx):
        raise ValueError(f"Invalid foundation: {f_idx}")
    return [f"{f_idx}{w}" for w in SUBFOUNDATION_WORLDS]


def get_world_subfoundations(world: str) -> List[str]:
    """Get all sub-foundations for a world (a, b, c, d)."""
    w = world.lower()
    if w not in SUBFOUNDATION_WORLDS:
        raise ValueError(f"Invalid world: {world}")
    return [f"{f}{w}" for f in range(1, 8)]


__all__ = [
    "FoundationDefinition",
    "SubFoundationDefinition",
    "FOUNDATIONS",
    "FOUNDATION_NAMES",
    "FOUNDATION_BY_NAME",
    "SUB_FOUNDATIONS",
    "SUB_FOUNDATION_CODES",
    "SUBFOUNDATION_WORLDS",
    "SUBFOUNDATION_WORLD_MAP",
    "SUBFOUNDATION_PATTERN",
    "parse_subfoundation",
    "validate_subfoundation",
    "validate_foundation",
    "get_foundation_subfoundations",
    "get_world_subfoundations",
]

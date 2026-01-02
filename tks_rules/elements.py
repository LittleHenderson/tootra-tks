"""
TKS Canonical 40 Elements Definition Module

Defines the 40 Elements as per TKS v7.4 canonical documents.

Formula: 4 Worlds x 10 Noetics = 40 Elements
Notation: [World Letter][Noetic Number] (e.g., A1, B7, C3, D10)

CRITICAL: Element notation uses 1-10 (NOT 0-9)
         A1-A10, B1-B10, C1-C10, D1-D10
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import re

from .noetics import NOETICS, NOETIC_NAMES
from .worlds import WORLDS, WORLD_LETTERS


@dataclass(frozen=True)
class ElementDefinition:
    """Immutable definition of a single element."""
    code: str           # A1, B7, etc.
    world: str          # A, B, C, D
    noetic: int         # 1-10 (element notation)
    noetic_idx: int     # 0-9 (internal noetic index)
    name: str           # Full name
    index: int          # Linear index 0-39


# =============================================================================
# ELEMENT NUMBERING MAPPING
# =============================================================================

# Element notation uses 1-10, but internal noetics are 0-9
# Element number N corresponds to noetic index (N-1) for N in 1-9
# Element number 10 corresponds to noetic index 0 (Idea)

def element_num_to_noetic_idx(elem_num: int) -> int:
    """Convert element number (1-10) to noetic index (0-9)."""
    if elem_num == 10:
        return 0  # Element 10 = noetic 0 (Idea)
    return elem_num  # Elements 1-9 = noetics 1-9


def noetic_idx_to_element_num(noetic_idx: int) -> int:
    """Convert noetic index (0-9) to element number (1-10)."""
    if noetic_idx == 0:
        return 10  # Noetic 0 (Idea) = Element 10
    return noetic_idx  # Noetics 1-9 = Elements 1-9


# =============================================================================
# CANONICAL 40 ELEMENTS
# =============================================================================

def _build_elements() -> Dict[str, ElementDefinition]:
    """Build all 40 element definitions."""
    elements = {}
    idx = 0

    for world in WORLD_LETTERS:
        for elem_num in range(1, 11):
            code = f"{world}{elem_num}"
            noetic_idx = element_num_to_noetic_idx(elem_num)
            noetic_name = NOETIC_NAMES[noetic_idx]
            world_domain = WORLDS[world].domain

            elements[code] = ElementDefinition(
                code=code,
                world=world,
                noetic=elem_num,
                noetic_idx=noetic_idx,
                name=f"{world_domain} {noetic_name}",
                index=idx
            )
            idx += 1

    return elements


ELEMENTS: Dict[str, ElementDefinition] = _build_elements()

# Lists for iteration
ELEMENT_CODES: List[str] = [f"{w}{n}" for w in WORLD_LETTERS for n in range(1, 11)]

# Quick lookup by linear index
ELEMENT_BY_INDEX: Dict[int, str] = {e.index: e.code for e in ELEMENTS.values()}

# World-grouped elements
ELEMENTS_BY_WORLD: Dict[str, List[str]] = {
    world: [f"{world}{n}" for n in range(1, 11)]
    for world in WORLD_LETTERS
}


# =============================================================================
# ELEMENT PARSING AND VALIDATION
# =============================================================================

ELEMENT_PATTERN = re.compile(r'^([ABCD])(\d{1,2})$')


def parse_element(code: str) -> Optional[Tuple[str, int]]:
    """
    Parse an element code into (world, element_num).

    Returns None if invalid.
    """
    match = ELEMENT_PATTERN.match(code.upper())
    if not match:
        return None

    world = match.group(1)
    elem_num = int(match.group(2))

    if not 1 <= elem_num <= 10:
        return None

    return (world, elem_num)


def validate_element(code: str) -> bool:
    """Check if a code is a valid element."""
    return parse_element(code) is not None


def element_to_index(code: str) -> int:
    """Convert element code to linear index (0-39)."""
    parsed = parse_element(code)
    if parsed is None:
        raise ValueError(f"Invalid element code: {code}")

    world, elem_num = parsed
    world_offset = WORLD_LETTERS.index(world) * 10
    return world_offset + (elem_num - 1)


def index_to_element(idx: int) -> str:
    """Convert linear index (0-39) to element code."""
    if not 0 <= idx <= 39:
        raise ValueError(f"Index must be 0-39, got {idx}")

    world_idx = idx // 10
    elem_num = (idx % 10) + 1

    return f"{WORLD_LETTERS[world_idx]}{elem_num}"


# =============================================================================
# ELEMENT RELATIONSHIPS
# =============================================================================

def same_world(elem1: str, elem2: str) -> bool:
    """Check if two elements are in the same world."""
    p1, p2 = parse_element(elem1), parse_element(elem2)
    if p1 is None or p2 is None:
        return False
    return p1[0] == p2[0]


def same_noetic(elem1: str, elem2: str) -> bool:
    """Check if two elements have the same noetic."""
    p1, p2 = parse_element(elem1), parse_element(elem2)
    if p1 is None or p2 is None:
        return False
    return p1[1] == p2[1]


def get_element_noetic_idx(code: str) -> int:
    """Get the internal noetic index (0-9) for an element."""
    parsed = parse_element(code)
    if parsed is None:
        raise ValueError(f"Invalid element code: {code}")
    return element_num_to_noetic_idx(parsed[1])


__all__ = [
    "ElementDefinition",
    "ELEMENTS",
    "ELEMENT_CODES",
    "ELEMENT_BY_INDEX",
    "ELEMENTS_BY_WORLD",
    "ELEMENT_PATTERN",
    "element_num_to_noetic_idx",
    "noetic_idx_to_element_num",
    "parse_element",
    "validate_element",
    "element_to_index",
    "index_to_element",
    "same_world",
    "same_noetic",
    "get_element_noetic_idx",
]

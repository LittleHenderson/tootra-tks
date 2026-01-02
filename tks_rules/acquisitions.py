"""
TKS Canonical Acquisitions Module

Defines the 22 Acquisitions (D/W/P chains + A22):
- D1-D7: Desire chain (7 desires)
- W1-W7: Wisdom chain (7 wisdoms)
- P1-P7: Power chain (7 powers)
- A22: Meta-Desire (desire for desire)

Formula: 7 Foundations x 3 Components (D/W/P) + 1 Meta = 22
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AcquisitionDefinition:
    """Immutable definition of a single acquisition."""
    code: str               # D1, W5, P3, A22
    engine_id: int          # Unique ID for engine (1-22)
    category: str           # Desire, Wisdom, Power, or Meta
    foundation: Optional[int]  # 1-7, or None for A22
    card: str               # Associated card type
    label: str              # Description


# =============================================================================
# ENGINE ID FORMULA
# =============================================================================

# For Foundation n (1-7):
# D(n) = 3n - 2
# W(n) = 3n - 1
# P(n) = 3n

def foundation_to_desire_id(foundation: int) -> int:
    """Convert foundation (1-7) to Desire engine ID."""
    return 3 * foundation - 2

def foundation_to_wisdom_id(foundation: int) -> int:
    """Convert foundation (1-7) to Wisdom engine ID."""
    return 3 * foundation - 1

def foundation_to_power_id(foundation: int) -> int:
    """Convert foundation (1-7) to Power engine ID."""
    return 3 * foundation


# =============================================================================
# CANONICAL 22 ACQUISITIONS
# =============================================================================

ACQUISITIONS: Dict[str, AcquisitionDefinition] = {
    # DESIRE (D) - Kings - "Want to"
    "D1": AcquisitionDefinition(
        code="D1", engine_id=1, category="Desire", foundation=1, card="King",
        label="Desire for unity with God/source"
    ),
    "D2": AcquisitionDefinition(
        code="D2", engine_id=4, category="Desire", foundation=2, card="King",
        label="Desire for insight/knowledge"
    ),
    "D3": AcquisitionDefinition(
        code="D3", engine_id=7, category="Desire", foundation=3, card="King",
        label="Desire for health/vitality"
    ),
    "D4": AcquisitionDefinition(
        code="D4", engine_id=10, category="Desire", foundation=4, card="King",
        label="Desire for companionship/love"
    ),
    "D5": AcquisitionDefinition(
        code="D5", engine_id=13, category="Desire", foundation=5, card="King",
        label="Desire for worldly power/influence"
    ),
    "D6": AcquisitionDefinition(
        code="D6", engine_id=16, category="Desire", foundation=6, card="King",
        label="Desire for material resources"
    ),
    "D7": AcquisitionDefinition(
        code="D7", engine_id=19, category="Desire", foundation=7, card="King",
        label="Desire for continuation/reproduction"
    ),

    # WISDOM (W) - Jacks - "Know how to"
    "W1": AcquisitionDefinition(
        code="W1", engine_id=2, category="Wisdom", foundation=1, card="Jack",
        label="Wisdom about unity"
    ),
    "W2": AcquisitionDefinition(
        code="W2", engine_id=5, category="Wisdom", foundation=2, card="Jack",
        label="Wisdom about knowing"
    ),
    "W3": AcquisitionDefinition(
        code="W3", engine_id=8, category="Wisdom", foundation=3, card="Jack",
        label="Wisdom about health"
    ),
    "W4": AcquisitionDefinition(
        code="W4", engine_id=11, category="Wisdom", foundation=4, card="Jack",
        label="Wisdom about relationships"
    ),
    "W5": AcquisitionDefinition(
        code="W5", engine_id=14, category="Wisdom", foundation=5, card="Jack",
        label="Wisdom about power"
    ),
    "W6": AcquisitionDefinition(
        code="W6", engine_id=17, category="Wisdom", foundation=6, card="Jack",
        label="Wisdom about wealth/resources"
    ),
    "W7": AcquisitionDefinition(
        code="W7", engine_id=20, category="Wisdom", foundation=7, card="Jack",
        label="Wisdom about continuation"
    ),

    # POWER (P) - Queens - "Can do"
    "P1": AcquisitionDefinition(
        code="P1", engine_id=3, category="Power", foundation=1, card="Queen",
        label="Power to achieve unity"
    ),
    "P2": AcquisitionDefinition(
        code="P2", engine_id=6, category="Power", foundation=2, card="Queen",
        label="Power to obtain knowledge"
    ),
    "P3": AcquisitionDefinition(
        code="P3", engine_id=9, category="Power", foundation=3, card="Queen",
        label="Power to create/maintain health"
    ),
    "P4": AcquisitionDefinition(
        code="P4", engine_id=12, category="Power", foundation=4, card="Queen",
        label="Power to form/maintain bonds"
    ),
    "P5": AcquisitionDefinition(
        code="P5", engine_id=15, category="Power", foundation=5, card="Queen",
        label="Power to act in power structures"
    ),
    "P6": AcquisitionDefinition(
        code="P6", engine_id=18, category="Power", foundation=6, card="Queen",
        label="Power to acquire/manage resources"
    ),
    "P7": AcquisitionDefinition(
        code="P7", engine_id=21, category="Power", foundation=7, card="Queen",
        label="Power to continue/reproduce"
    ),

    # META-DESIRE (Special)
    "A22": AcquisitionDefinition(
        code="A22", engine_id=22, category="Meta", foundation=None, card="Joker",
        label="Desire for desire (meta-level rekindling)"
    ),
}


# =============================================================================
# ACQUISITION CHAINS
# =============================================================================

DESIRE_CHAIN: List[str] = [f"D{i}" for i in range(1, 8)]    # D1-D7
WISDOM_CHAIN: List[str] = [f"W{i}" for i in range(1, 8)]    # W1-W7
POWER_CHAIN: List[str] = [f"P{i}" for i in range(1, 8)]     # P1-P7
META_ACQUISITIONS: List[str] = ["A22"]

ALL_ACQUISITIONS: List[str] = DESIRE_CHAIN + WISDOM_CHAIN + POWER_CHAIN + META_ACQUISITIONS

# By category
ACQUISITIONS_BY_CATEGORY: Dict[str, List[str]] = {
    "Desire": DESIRE_CHAIN,
    "Wisdom": WISDOM_CHAIN,
    "Power": POWER_CHAIN,
    "Meta": META_ACQUISITIONS,
}

# By foundation
ACQUISITIONS_BY_FOUNDATION: Dict[int, Dict[str, str]] = {
    f: {"D": f"D{f}", "W": f"W{f}", "P": f"P{f}"}
    for f in range(1, 8)
}


# =============================================================================
# PARSING AND VALIDATION
# =============================================================================

ACQUISITION_PATTERN = re.compile(r'^([DWP])([1-7])$|^(A22)$')


def parse_acquisition(code: str) -> Optional[Tuple[str, Optional[int]]]:
    """
    Parse an acquisition code into (category, foundation).

    Returns None if invalid.
    """
    code = code.upper()

    if code == "A22":
        return ("Meta", None)

    match = ACQUISITION_PATTERN.match(code)
    if not match:
        return None

    if match.group(3):  # A22
        return ("Meta", None)

    category_map = {"D": "Desire", "W": "Wisdom", "P": "Power"}
    category = category_map[match.group(1)]
    foundation = int(match.group(2))

    return (category, foundation)


def validate_acquisition(code: str) -> bool:
    """Check if a code is a valid acquisition."""
    return parse_acquisition(code) is not None


def get_acquisition_for_foundation(foundation: int, category: str) -> Optional[str]:
    """
    Get acquisition code for a foundation and category.

    Args:
        foundation: Foundation number (1-7)
        category: 'D', 'W', or 'P'

    Returns:
        Acquisition code or None if invalid
    """
    if not 1 <= foundation <= 7:
        return None
    if category not in ("D", "W", "P"):
        return None

    return f"{category}{foundation}"


# =============================================================================
# PREREQUISITE STATES (Developmental Phases)
# =============================================================================

DESIRE_PHASES: Dict[str, Dict[str, any]] = {
    "Beginning": {"symbol": "clubs", "percentage": 25},
    "Strong": {"symbol": "hearts", "percentage": 100},
    "Weakening": {"symbol": "spades", "percentage": 50},
    "Absent": {"symbol": "diamonds", "percentage": 0},
}

POWER_PHASES: Dict[str, Dict[str, any]] = {
    "Birth": {"symbol": "diamonds", "percentage": 10},
    "Adolescent": {"symbol": "clubs", "percentage": 40},
    "Mature": {"symbol": "hearts", "percentage": 100},
    "Waning": {"symbol": "spades", "percentage": 20},
}

WISDOM_PHASES: Dict[str, Dict[str, any]] = {
    "Learning": {"symbol": "clubs", "percentage": 25},
    "Knowing": {"symbol": "diamonds", "percentage": 50},
    "Understanding": {"symbol": "spades", "percentage": 75},
    "Mastered": {"symbol": "hearts", "percentage": 100},
}


__all__ = [
    "AcquisitionDefinition",
    "ACQUISITIONS",
    "DESIRE_CHAIN",
    "WISDOM_CHAIN",
    "POWER_CHAIN",
    "META_ACQUISITIONS",
    "ALL_ACQUISITIONS",
    "ACQUISITIONS_BY_CATEGORY",
    "ACQUISITIONS_BY_FOUNDATION",
    "ACQUISITION_PATTERN",
    "DESIRE_PHASES",
    "POWER_PHASES",
    "WISDOM_PHASES",
    "foundation_to_desire_id",
    "foundation_to_wisdom_id",
    "foundation_to_power_id",
    "parse_acquisition",
    "validate_acquisition",
    "get_acquisition_for_foundation",
]

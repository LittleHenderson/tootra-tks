"""
TKS Acquisition Paths - Canonical Tree Path Mappings

Maps each of the 22 Acquisitions to their (source_noetic, destination_noetic) pairs
as defined in TKS_FORMAL_MANUAL_v6.1_CLEAN_DEFINITIONS.md Section 6.3.

The Tree Path table defines which two noetic spheres each acquisition connects
on the TKS Tree of Life. These paths are used by AcquisitionProgram to construct
sparse 10x10 operators that only affect the relevant noetic indices.

Index Convention:
- Manual uses 1-10 (Mind=1, Idea=10)
- Code uses 0-9 (Idea=0, Mind=1)
- Conversion: manual_idx 10 -> 0, manual_idx 1-9 -> same

Reference: TKS_FORMAL_MANUAL_v6.1_CLEAN_DEFINITIONS.md lines 737-760
"""

from typing import Dict, Tuple


def manual_idx_to_noetic(manual_idx: int) -> int:
    """
    Convert manual noetic index (1-10) to code index (0-9).

    Manual indexing:
        1=Mind, 2=Positive, 3=Negative, 4=Vibration, 5=Female,
        6=Male, 7=Rhythm, 8=Above/Cause, 9=Below/Effect, 10=Idea

    Code indexing:
        0=Idea, 1=Mind, 2=Positive, 3=Negative, 4=Vibration,
        5=Female, 6=Male, 7=Rhythm, 8=Above, 9=Below
    """
    if manual_idx == 10:
        return 0  # Idea
    return manual_idx  # 1-9 map directly


# =============================================================================
# CANONICAL ACQUISITION PATHS (from Tree Path table)
# =============================================================================
# Each tuple is (source_noetic, destination_noetic) in 0-based code indices.
# These define which two noetics each acquisition operator affects.

ACQUISITION_PATHS: Dict[str, Tuple[int, int]] = {
    # Foundation 1 - Unity/Association
    # D1 (engine_id=1): Mind 1 → Vibration 4
    "D1": (manual_idx_to_noetic(1), manual_idx_to_noetic(4)),  # (1, 4)
    # W1 (engine_id=2): Mind 1 → Positive 2
    "W1": (manual_idx_to_noetic(1), manual_idx_to_noetic(2)),  # (1, 2)
    # P1 (engine_id=3): Mind 1 → Negative 3
    "P1": (manual_idx_to_noetic(1), manual_idx_to_noetic(3)),  # (1, 3)

    # Foundation 2 - Wisdom
    # D2 (engine_id=4): Mind 1 → Vibration 4 (same path as D1, different params)
    "D2": (manual_idx_to_noetic(1), manual_idx_to_noetic(4)),  # (1, 4)
    # W2 (engine_id=5): Positive 2 → Vibration 4
    "W2": (manual_idx_to_noetic(2), manual_idx_to_noetic(4)),  # (2, 4)
    # P2 (engine_id=6): Negative 3 → Vibration 4
    "P2": (manual_idx_to_noetic(3), manual_idx_to_noetic(4)),  # (3, 4)

    # Foundation 3 - Life
    # D3 (engine_id=7): Vibration 4 → Male 6
    "D3": (manual_idx_to_noetic(4), manual_idx_to_noetic(6)),  # (4, 6)
    # W3 (engine_id=8): Positive 2 → Male 6
    "W3": (manual_idx_to_noetic(2), manual_idx_to_noetic(6)),  # (2, 6)
    # P3 (engine_id=9): Negative 3 → Female 5
    "P3": (manual_idx_to_noetic(3), manual_idx_to_noetic(5)),  # (3, 5)

    # Foundation 4 - Companionship
    # D4 (engine_id=10): Female 5 → Rhythm 7
    "D4": (manual_idx_to_noetic(5), manual_idx_to_noetic(7)),  # (5, 7)
    # W4 (engine_id=11): Vibration 4 → Male 6 (same path as D3)
    "W4": (manual_idx_to_noetic(4), manual_idx_to_noetic(6)),  # (4, 6)
    # P4 (engine_id=12): Vibration 4 → Female 5
    "P4": (manual_idx_to_noetic(4), manual_idx_to_noetic(5)),  # (4, 5)

    # Foundation 5 - Power
    # D5 (engine_id=13): Above/Cause 8 → Idea 10
    "D5": (manual_idx_to_noetic(8), manual_idx_to_noetic(10)),  # (8, 0)
    # W5 (engine_id=14): Male 6 → Rhythm 7
    "W5": (manual_idx_to_noetic(6), manual_idx_to_noetic(7)),  # (6, 7)
    # P5 (engine_id=15): Rhythm 7 → Below/Effect 9
    "P5": (manual_idx_to_noetic(7), manual_idx_to_noetic(9)),  # (7, 9)

    # Foundation 6 - Wealth/Material
    # D6 (engine_id=16): Above/Cause 8 → Idea 10 (same path as D5)
    "D6": (manual_idx_to_noetic(8), manual_idx_to_noetic(10)),  # (8, 0)
    # W6 (engine_id=17): Female 5 → Below/Effect 9
    "W6": (manual_idx_to_noetic(5), manual_idx_to_noetic(9)),  # (5, 9)
    # P6 (engine_id=18): Rhythm 7 → Above/Cause 8
    "P6": (manual_idx_to_noetic(7), manual_idx_to_noetic(8)),  # (7, 8)

    # Foundation 7 - Continuation
    # D7 (engine_id=19): Below/Effect 9 → Idea 10
    "D7": (manual_idx_to_noetic(9), manual_idx_to_noetic(10)),  # (9, 0)
    # W7 (engine_id=20): Male 6 → Above/Cause 8
    "W7": (manual_idx_to_noetic(6), manual_idx_to_noetic(8)),  # (6, 8)
    # P7 (engine_id=21): Rhythm 7 → Above/Cause 8 (same path as P6)
    "P7": (manual_idx_to_noetic(7), manual_idx_to_noetic(8)),  # (7, 8)

    # Meta-Desire
    # A22 (engine_id=22): Male 6 → Female 5
    "A22": (manual_idx_to_noetic(6), manual_idx_to_noetic(5)),  # (6, 5)
}


# =============================================================================
# PATH COLLISION GROUPS
# =============================================================================
# Acquisitions that share the same (src, dst) path need distinct parameters.
# Listed here for reference and validation.

PATH_COLLISION_GROUPS: Dict[Tuple[int, int], list] = {}
for acq, path in ACQUISITION_PATHS.items():
    if path not in PATH_COLLISION_GROUPS:
        PATH_COLLISION_GROUPS[path] = []
    PATH_COLLISION_GROUPS[path].append(acq)

# Filter to only groups with >1 acquisition
PATH_COLLISIONS: Dict[Tuple[int, int], list] = {
    path: acqs for path, acqs in PATH_COLLISION_GROUPS.items() if len(acqs) > 1
}
# Expected collisions:
# (1, 4): [D1, D2] - Mind → Vibration
# (4, 6): [D3, W4] - Vibration → Male
# (8, 0): [D5, D6] - Above → Idea
# (7, 8): [P6, P7] - Rhythm → Above


# =============================================================================
# CATEGORY AND FOUNDATION METADATA
# =============================================================================

ACQUISITION_CATEGORY: Dict[str, str] = {
    "D1": "D", "D2": "D", "D3": "D", "D4": "D", "D5": "D", "D6": "D", "D7": "D",
    "W1": "W", "W2": "W", "W3": "W", "W4": "W", "W5": "W", "W6": "W", "W7": "W",
    "P1": "P", "P2": "P", "P3": "P", "P4": "P", "P5": "P", "P6": "P", "P7": "P",
    "A22": "Meta",
}

ACQUISITION_FOUNDATION: Dict[str, int] = {
    "D1": 1, "W1": 1, "P1": 1,
    "D2": 2, "W2": 2, "P2": 2,
    "D3": 3, "W3": 3, "P3": 3,
    "D4": 4, "W4": 4, "P4": 4,
    "D5": 5, "W5": 5, "P5": 5,
    "D6": 6, "W6": 6, "P6": 6,
    "D7": 7, "W7": 7, "P7": 7,
    "A22": 0,  # Meta - not tied to a single foundation
}

# Ordered list matching engine_id order (1-22)
ACQUISITION_ORDER: list = [
    "D1", "W1", "P1",  # F1: 1,2,3
    "D2", "W2", "P2",  # F2: 4,5,6
    "D3", "W3", "P3",  # F3: 7,8,9
    "D4", "W4", "P4",  # F4: 10,11,12
    "D5", "W5", "P5",  # F5: 13,14,15
    "D6", "W6", "P6",  # F6: 16,17,18
    "D7", "W7", "P7",  # F7: 19,20,21
    "A22",             # Meta: 22
]


def get_path_for_acquisition(acq: str) -> Tuple[int, int]:
    """Get (src_noetic, dst_noetic) for an acquisition code."""
    return ACQUISITION_PATHS[acq.upper()]


def get_acquisitions_for_path(path: Tuple[int, int]) -> list:
    """Get all acquisitions that use a given path."""
    return PATH_COLLISION_GROUPS.get(path, [])


def validate_acquisition_paths() -> bool:
    """
    Validate that all acquisition paths are consistent.

    Checks:
    - Exactly 22 acquisitions
    - All paths are valid noetic index pairs (0-9)
    - All expected collisions match
    - Category assignments are valid

    Returns:
        True if all validations pass, False otherwise
    """
    errors = []

    # Check count
    if len(ACQUISITION_PATHS) != 22:
        errors.append(f"Expected 22 acquisitions, got {len(ACQUISITION_PATHS)}")

    # Check valid indices
    for acq, (i, j) in ACQUISITION_PATHS.items():
        if not (0 <= i <= 9 and 0 <= j <= 9):
            errors.append(f"{acq}: invalid indices ({i}, {j})")

    # Check expected collisions
    expected_collisions = {
        (1, 4): ["D1", "D2"],
        (4, 6): ["D3", "W4"],
        (8, 0): ["D5", "D6"],
        (7, 8): ["P6", "P7"],
    }
    for path, expected in expected_collisions.items():
        actual = sorted(PATH_COLLISIONS.get(path, []))
        if actual != sorted(expected):
            errors.append(f"Path {path}: expected {expected}, got {actual}")

    # Check categories
    valid_cats = {"D", "W", "P", "Meta"}
    for acq, cat in ACQUISITION_CATEGORY.items():
        if cat not in valid_cats:
            errors.append(f"{acq}: invalid category '{cat}'")

    if errors:
        for err in errors:
            print(f"  Validation error: {err}")
        return False

    return True


__all__ = [
    "manual_idx_to_noetic",
    "ACQUISITION_PATHS",
    "PATH_COLLISIONS",
    "ACQUISITION_CATEGORY",
    "ACQUISITION_FOUNDATION",
    "ACQUISITION_ORDER",
    "get_path_for_acquisition",
    "get_acquisitions_for_path",
    "validate_acquisition_paths",
]

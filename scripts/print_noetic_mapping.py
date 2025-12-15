#!/usr/bin/env python3
"""
Helper script to print noetic involution mappings in both 0-based and 1-based form.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inversion.engine import NOETIC_OPPOSITE


def main() -> None:
    print("=== Noetic Opposites (0-based indices) ===")
    for idx, val in NOETIC_OPPOSITE.items():
        print(f"{idx} -> {val}")

    print("\n=== Noetic Opposites (1-based noetics) ===")
    for idx, val in NOETIC_OPPOSITE.items():
        print(f"N{idx+1} -> N{val+1}")

    # Canonical pairs in 1-based numbering
    canonical_pairs = {2: 3, 3: 2, 5: 6, 6: 5, 8: 9, 9: 8}
    for k, v in canonical_pairs.items():
        idx = k - 1
        target = v - 1
        if NOETIC_OPPOSITE.get(idx) != target:
            raise AssertionError(f"N{k} should map to N{v}, got N{NOETIC_OPPOSITE.get(idx, -1)+1}")
    print("\nCanonical involution pairs (1-based) verified.")


if __name__ == "__main__":
    main()

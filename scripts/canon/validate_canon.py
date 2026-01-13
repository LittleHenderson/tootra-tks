#!/usr/bin/env python3
"""
Canon Validation Script - Comprehensive validation of TKS canon.

Usage:
    python scripts/canon/validate_canon.py --canon canon/normalized

This script validates:
1. Structural integrity of all JSON files
2. Cross-references between files (elements reference valid worlds/noetics)
3. Semantic consistency (opposites, relationships)
4. Gold examples reference valid canon symbols
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class CanonValidator:
    """Validates TKS canon for consistency and correctness."""

    def __init__(self, canon_dir: Path, gold_dir: Path):
        self.canon_dir = canon_dir
        self.gold_dir = gold_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # Loaded canon data
        self.worlds: Dict[str, Any] = {}
        self.noetics: Dict[str, Any] = {}
        self.elements: Dict[str, Any] = {}
        self.foundations: Dict[str, Any] = {}
        self.subfoundations: Dict[str, Any] = {}
        self.operators: Dict[str, Any] = {}
        self.acquisitions: Dict[str, Any] = {}
        self.index: Dict[str, Any] = {}

    def load_canon(self) -> bool:
        """Load all canon files."""
        files_to_load = [
            ("canon_worlds_4.json", "worlds", "worlds"),
            ("canon_noetics_10.json", "noetics", "noetics"),
            ("canon_elements_40.json", "elements", "elements"),
            ("canon_foundations_7.json", "foundations", "foundations"),
            ("canon_subfoundations_28.json", "subfoundations", "subfoundations"),
            ("canon_operators.json", "operators", "operators"),
            ("canon_acquisitions_22.json", "acquisitions", "acquisitions"),
            ("canon_index.json", "index", None),
        ]

        for filename, attr, key in files_to_load:
            filepath = self.canon_dir / filename
            if not filepath.exists():
                self.errors.append(f"Missing required file: {filename}")
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if key:
                    setattr(self, attr, data.get(key, {}))
                else:
                    setattr(self, attr, data)

            except json.JSONDecodeError as e:
                self.errors.append(f"{filename}: Invalid JSON - {e}")
            except Exception as e:
                self.errors.append(f"{filename}: Failed to load - {e}")

        return len(self.errors) == 0

    def validate_worlds(self) -> int:
        """Validate worlds structure."""
        errors = 0
        required_worlds = {"A", "B", "C", "D"}

        if len(self.worlds) != 4:
            self.errors.append(
                f"canon_worlds_4.json: Expected 4 worlds, found {len(self.worlds)}"
            )
            errors += 1

        found_worlds = set(self.worlds.keys())
        missing = required_worlds - found_worlds
        if missing:
            self.errors.append(
                f"canon_worlds_4.json: Missing required worlds: {missing}"
            )
            errors += 1

        # Validate each world has required fields
        for world_id, world in self.worlds.items():
            if "id" not in world:
                self.errors.append(f"World {world_id}: Missing 'id' field")
                errors += 1
            if "name" not in world:
                self.errors.append(f"World {world_id}: Missing 'name' field")
                errors += 1

        return errors

    def validate_noetics(self) -> int:
        """Validate noetics structure."""
        errors = 0
        # Canonical TKS noetics are IDs 0-9
        required_noetics = set(str(i) for i in range(0, 10))

        if len(self.noetics) != 10:
            self.errors.append(
                f"canon_noetics_10.json: Expected 10 noetics, found {len(self.noetics)}"
            )
            errors += 1

        found_noetics = set(self.noetics.keys())
        missing = required_noetics - found_noetics
        if missing:
            self.errors.append(
                f"canon_noetics_10.json: Missing required noetics: {missing}"
            )
            errors += 1

        return errors

    def validate_elements(self) -> int:
        """Validate elements reference valid worlds and noetics."""
        errors = 0

        if len(self.elements) != 40:
            self.errors.append(
                f"canon_elements_40.json: Expected 40 elements (4 worlds x 10 noetics), "
                f"found {len(self.elements)}"
            )
            errors += 1

        # Elements use 1-based noetic indexing (1-10), canonical noetics use 0-9
        # Noetic 10 in elements corresponds to noetic 0 (Idea) in canon
        valid_element_noetics = set(range(1, 11))

        # Check each element references valid world and noetic
        for elem_id, elem in self.elements.items():
            world = elem.get("world")
            noetic = elem.get("noetic")

            if world not in self.worlds:
                self.errors.append(
                    f"Element {elem_id}: References invalid world '{world}'"
                )
                errors += 1

            if noetic not in valid_element_noetics:
                self.errors.append(
                    f"Element {elem_id}: References invalid noetic '{noetic}'"
                )
                errors += 1

            # Verify ID format matches world+noetic
            expected_id = f"{world}{noetic}"
            if elem_id != expected_id:
                self.warnings.append(
                    f"Element {elem_id}: ID doesn't match world+noetic pattern (expected {expected_id})"
                )

        return errors

    def validate_foundations(self) -> int:
        """Validate foundations and their opposites."""
        errors = 0

        if len(self.foundations) != 7:
            self.errors.append(
                f"canon_foundations_7.json: Expected 7 foundations, found {len(self.foundations)}"
            )
            errors += 1

        # Validate opposite relationships
        for fnd_id, fnd in self.foundations.items():
            opposite = fnd.get("opposite")
            if opposite is None:
                self.errors.append(f"Foundation {fnd_id}: Missing 'opposite' field")
                errors += 1
                continue

            # Verify opposite exists
            opposite_str = str(opposite)
            if opposite_str not in self.foundations:
                self.errors.append(
                    f"Foundation {fnd_id}: Opposite '{opposite}' not found"
                )
                errors += 1
                continue

            # Verify reciprocal relationship (except for self-dual 4)
            opposite_fnd = self.foundations[opposite_str]
            if int(fnd_id) != 4 and opposite_fnd.get("opposite") != int(fnd_id):
                self.errors.append(
                    f"Foundation {fnd_id}: Non-reciprocal opposite relationship"
                )
                errors += 1

        return errors

    def validate_subfoundations(self) -> int:
        """Validate subfoundations reference valid foundations and worlds."""
        errors = 0

        expected_count = 7 * 4  # 7 foundations x 4 worlds = 28
        if len(self.subfoundations) != expected_count:
            self.errors.append(
                f"canon_subfoundations_28.json: Expected {expected_count} subfoundations, "
                f"found {len(self.subfoundations)}"
            )
            errors += 1

        for sf_id, sf in self.subfoundations.items():
            foundation = sf.get("foundation")
            world = sf.get("world")

            if str(foundation) not in self.foundations:
                self.errors.append(
                    f"Subfoundation {sf_id}: References invalid foundation '{foundation}'"
                )
                errors += 1

            if world not in self.worlds:
                self.errors.append(
                    f"Subfoundation {sf_id}: References invalid world '{world}'"
                )
                errors += 1

        return errors

    def validate_operators(self) -> int:
        """Validate operator definitions."""
        errors = 0

        required_operators = {"+", "-", "*", "/", "->", "<-", "o"}
        tootra_operators = {"+T", "-T", "*T", "/T"}

        all_required = required_operators | tootra_operators
        found_ops = set(self.operators.keys())

        missing = all_required - found_ops
        if missing:
            self.warnings.append(
                f"canon_operators.json: Missing some expected operators: {missing}"
            )

        # Validate each operator has required fields
        for op_id, op in self.operators.items():
            if "symbol" not in op:
                self.errors.append(f"Operator {op_id}: Missing 'symbol' field")
                errors += 1
            if "name" not in op:
                self.errors.append(f"Operator {op_id}: Missing 'name' field")
                errors += 1
            if "symmetric" not in op:
                self.warnings.append(f"Operator {op_id}: Missing 'symmetric' field")

        return errors

    def validate_gold_files(self) -> int:
        """Validate gold JSONL files reference valid canon symbols."""
        errors = 0

        if not self.gold_dir.exists():
            self.warnings.append("Gold directory not found, skipping gold validation")
            return 0

        # Build set of valid element IDs
        valid_elements = set(self.elements.keys())
        valid_operators = set(self.operators.keys())

        for filepath in self.gold_dir.glob("*.jsonl"):
            line_num = 0
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line_num += 1
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            record = json.loads(line)

                            # Validate operator references
                            if "operator" in record:
                                op = record["operator"]
                                # Handle both "+" and "+T" style
                                if op not in valid_operators and f"{op}T" not in valid_operators:
                                    self.warnings.append(
                                        f"{filepath.name}:{line_num}: Unknown operator '{op}'"
                                    )

                            # Validate expression contains valid elements
                            if "expression" in record:
                                expr = record["expression"]
                                # Extract element references (e.g., A1, B2, C3)
                                import re
                                elements_in_expr = re.findall(r'[A-D]\d+', expr)
                                for elem in elements_in_expr:
                                    if elem not in valid_elements:
                                        self.warnings.append(
                                            f"{filepath.name}:{line_num}: Unknown element '{elem}' in expression"
                                        )

                        except json.JSONDecodeError as e:
                            self.errors.append(
                                f"{filepath.name}:{line_num}: Invalid JSON - {e}"
                            )
                            errors += 1

            except Exception as e:
                self.errors.append(f"{filepath.name}: Failed to read - {e}")
                errors += 1

        return errors

    def validate_cross_references(self) -> int:
        """Validate cross-references in canon_index.json."""
        errors = 0

        if not self.index:
            self.errors.append("canon_index.json not loaded")
            return 1

        # Check all referenced files exist
        for category in ["normalized", "gold"]:
            if category not in self.index:
                self.warnings.append(f"canon_index.json: Missing '{category}' section")
                continue

            for filename, info in self.index[category].items():
                path = info.get("path", "")
                full_path = PROJECT_ROOT / path
                if not full_path.exists():
                    self.errors.append(
                        f"canon_index.json: Referenced file not found: {path}"
                    )
                    errors += 1

        return errors

    def run_all_validations(self) -> Tuple[int, int]:
        """Run all validation checks."""
        print("Loading canon files...")
        if not self.load_canon():
            print("  [FAIL] Could not load all required files")
            return len(self.errors), len(self.warnings)

        print("  [OK] All files loaded\n")

        validations = [
            ("Validating worlds...", self.validate_worlds),
            ("Validating noetics...", self.validate_noetics),
            ("Validating elements...", self.validate_elements),
            ("Validating foundations...", self.validate_foundations),
            ("Validating subfoundations...", self.validate_subfoundations),
            ("Validating operators...", self.validate_operators),
            ("Validating gold files...", self.validate_gold_files),
            ("Validating cross-references...", self.validate_cross_references),
        ]

        for msg, validator in validations:
            print(msg)
            err_count = validator()
            if err_count > 0:
                print(f"  [FAIL] {err_count} error(s)")
            else:
                print("  [OK]")

        return len(self.errors), len(self.warnings)


def main():
    parser = argparse.ArgumentParser(
        description="Validate TKS canon for consistency and correctness"
    )
    parser.add_argument(
        "--canon", dest="canon_dir",
        default="canon/normalized",
        help="Directory containing normalized canon JSON"
    )
    parser.add_argument(
        "--gold", dest="gold_dir",
        default="canon/gold",
        help="Directory containing gold JSONL files"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors"
    )
    args = parser.parse_args()

    canon_dir = PROJECT_ROOT / args.canon_dir
    gold_dir = PROJECT_ROOT / args.gold_dir

    print("=" * 60)
    print("TKS Canon Validator")
    print("=" * 60)
    print(f"Canon dir: {canon_dir}")
    print(f"Gold dir:  {gold_dir}")
    print(f"Mode:      {'Strict' if args.strict else 'Normal'}")
    print()

    validator = CanonValidator(canon_dir, gold_dir)
    error_count, warning_count = validator.run_all_validations()

    print()
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    if validator.warnings:
        print(f"\nWarnings ({warning_count}):")
        for w in validator.warnings:
            print(f"  - {w}")

    if validator.errors:
        print(f"\nErrors ({error_count}):")
        for e in validator.errors:
            print(f"  - {e}")

    # Determine exit status
    if error_count > 0:
        print("\n[FAILED] Canon validation failed with errors.")
        sys.exit(1)
    elif warning_count > 0 and args.strict:
        print("\n[FAILED] Canon validation failed (strict mode, warnings treated as errors).")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Canon validation passed.")
        if warning_count > 0:
            print(f"  ({warning_count} warnings)")
        sys.exit(0)


if __name__ == "__main__":
    main()

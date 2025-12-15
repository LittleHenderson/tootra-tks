"""
TKS Canonical Validator

Validates TKS expressions against canonical constraints:
- Worlds: A, B, C, D only
- Noetics: 1-10 (involutions: 2↔3, 5↔6, 8↔9; self-duals: 1,4,7,10)
- Foundations: 1-7
- ALLOWED_OPS: +, -, +T, -T, ->, <-, *T, /T, o

Used by augmentation pipeline to ensure generated data is canon-valid.
"""
from typing import Tuple, List, Any


def validate_canonical(expr: Any) -> Tuple[bool, List[str]]:
    """
    Validate expression against TKS canon.

    Returns (is_valid, list_of_errors)

    Checks:
    - All element worlds in {A, B, C, D}
    - All noetics in {1..10}
    - All foundations in {1..7}
    - All operators in ALLOWED_OPS: {+, -, +T, -T, ->, <-, *T, /T, o}
    - No unknown tokens

    Args:
        expr: TKSExpression to validate

    Returns:
        Tuple of (is_valid, list_of_error_messages)

    Example:
        >>> valid, errors = validate_canonical(expr)
        >>> if not valid:
        ...     print(f"Validation failed: {errors}")
    """
    from narrative.constants import (
        WORLD_LETTERS,
        ALLOWED_OPS,
        is_valid_world,
        is_valid_noetic,
        is_valid_foundation,
        is_valid_operator
    )

    errors = []

    # Handle both TKSExpression types (from narrative and scenario_inversion)
    if hasattr(expr, 'elements'):
        elements = expr.elements
    else:
        errors.append("Invalid expression type: missing 'elements' attribute")
        return False, errors

    if hasattr(expr, 'ops'):
        ops = expr.ops
    else:
        ops = []

    # Validate elements
    for i, element in enumerate(elements):
        # Parse element string (e.g., "D5", "B2.1", "A10")
        element_str = str(element).strip()

        # Skip empty elements
        if not element_str:
            errors.append(f"Element {i}: Empty element")
            continue

        # Strip sense notation if present (e.g., "D5.1" -> "D5")
        base_element = element_str.split(".")[0].split("^")[0].split("_")[0]

        # Validate minimum length
        if len(base_element) < 2:
            errors.append(f"Element {i} '{element_str}': Invalid format (too short)")
            continue

        # Extract world and noetic
        world = base_element[0].upper()
        noetic_str = base_element[1:]

        # Validate world
        if not is_valid_world(world):
            errors.append(f"Element {i} '{element_str}': Invalid world '{world}' (must be A/B/C/D)")

        # Validate noetic
        try:
            noetic = int(noetic_str)
            if not is_valid_noetic(noetic):
                errors.append(f"Element {i} '{element_str}': Invalid noetic '{noetic}' (must be 1-10)")
        except ValueError:
            errors.append(f"Element {i} '{element_str}': Invalid noetic '{noetic_str}' (must be integer)")

    # Validate operators
    for i, op in enumerate(ops):
        if not is_valid_operator(op):
            errors.append(f"Operator {i} '{op}': Invalid operator (must be in {ALLOWED_OPS})")

    # Validate foundations (if present)
    if hasattr(expr, 'foundations') and expr.foundations:
        for i, foundation in enumerate(expr.foundations):
            # Foundation can be tuple (fid, world) or just fid
            if isinstance(foundation, tuple):
                fid, world = foundation
            else:
                fid = foundation
                world = None

            if not is_valid_foundation(fid):
                errors.append(f"Foundation {i}: Invalid foundation '{fid}' (must be 1-7)")

            if world and not is_valid_world(world):
                errors.append(f"Foundation {i}: Invalid world '{world}' (must be A/B/C/D)")

    # Structural consistency: len(ops) should be len(elements) - 1
    if len(elements) > 1:
        expected_ops = len(elements) - 1
        if len(ops) != expected_ops:
            errors.append(f"Structural inconsistency: {len(elements)} elements require {expected_ops} operators, got {len(ops)}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_entry(entry: dict) -> Tuple[bool, List[str]]:
    """
    Validate a data entry (with expr_elements and expr_ops fields).

    Args:
        entry: Dict with 'expr_elements' and 'expr_ops' fields

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class MockExpression:
        """Minimal expression for validation."""
        elements: List[str] = field(default_factory=list)
        ops: List[str] = field(default_factory=list)
        foundations: List = field(default_factory=list)

    # Extract elements and ops from entry
    elements = entry.get('expr_elements', [])
    ops = entry.get('expr_ops', [])

    # Create mock expression
    mock_expr = MockExpression(
        elements=elements,
        ops=ops,
        foundations=[]
    )

    return validate_canonical(mock_expr)


def compute_validation_metrics(entries: List[dict]) -> dict:
    """
    Compute validation metrics for a list of entries.

    Args:
        entries: List of dicts with expr_elements and expr_ops

    Returns:
        Dict with validation metrics:
            - total: Total entries
            - valid: Number passing validation
            - invalid: Number failing validation
            - pass_rate: Pass rate (0-1)
            - error_counts: Dict of error type -> count
    """
    total = len(entries)
    valid_count = 0
    invalid_count = 0
    error_counts = {}

    for entry in entries:
        is_valid, errors = validate_entry(entry)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

            # Track error types
            for error in errors:
                # Extract error type from error message
                if "Invalid world" in error:
                    error_type = "invalid_world"
                elif "Invalid noetic" in error:
                    error_type = "invalid_noetic"
                elif "Invalid operator" in error:
                    error_type = "invalid_operator"
                elif "Invalid foundation" in error:
                    error_type = "invalid_foundation"
                elif "Structural inconsistency" in error:
                    error_type = "structural_error"
                else:
                    error_type = "other"

                error_counts[error_type] = error_counts.get(error_type, 0) + 1

    pass_rate = valid_count / total if total > 0 else 0.0

    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "pass_rate": pass_rate,
        "error_counts": error_counts
    }

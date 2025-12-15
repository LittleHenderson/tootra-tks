"""
Fuzz Testing for TKS Story -> Expr -> Invert -> Story Pipeline

This script runs deterministic fuzz testing to catch drift and regressions early.
It tests the complete pipeline: encode story -> invert expression -> decode story.

Validates canonical outputs:
- Worlds: A, B, C, D only
- Noetics: 1-10
- Foundations: 1-7
- Operators: +, -, +T, -T, *T, /T, o, ->, <-

Exit code:
- 0: All tests passed
- 1: At least one test failed
"""
import sys
import io
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Set UTF-8 encoding for stdout to handle special characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative import EncodeStory, DecodeStory, TKSExpression
from narrative.constants import ALLOWED_OPS
from inversion.engine import total_inversion, InversionMode, DialConfig, TargetProfile

# =============================================================================
# CANONICAL VALIDATION CONSTANTS
# =============================================================================

ALLOWED_WORLDS = {"A", "B", "C", "D"}
ALLOWED_NOETICS = set(range(1, 11))  # 1-10
ALLOWED_FOUNDATIONS = set(range(1, 8))  # 1-7
# ALLOWED_OPS imported from narrative.constants: {"+", "-", "+T", "-T", "*T", "/T", "o", "->", "<-"}

# =============================================================================
# TEST STORIES (Deterministic, No Randomness)
# =============================================================================

# Each test story is a tuple: (story_text, description)
TEST_STORIES: List[Tuple[str, str]] = [
    # Basic emotional stories
    ("She felt fear.", "Simple emotion"),
    ("A woman loved a man.", "Basic relationship"),
    ("He experienced joy.", "Positive emotion"),

    # Mental/belief stories
    ("She had faith in the plan.", "Mental faith"),
    ("Doubt conflicts with confidence.", "Mental conflict"),
    ("Past experiences shaped her beliefs.", "Mental learning"),

    # Physical/material stories
    ("Money brings health.", "Material → Physical"),
    ("The trigger caused transformation.", "Physical causation"),
    ("Vitality and wellness improve life.", "Physical health"),

    # Complex causal chains
    ("Fear causes illness.", "Emotional → Physical causation"),
    ("Grief and sorrow overwhelm.", "Multiple emotions"),
    ("Hope leads to action.", "Mental → Physical"),

    # Multi-element combinations
    ("She loved him with joy.", "Love + joy combination"),
    ("Fear intensified by fear creates panic.", "Intensification"),

    # Edge cases: Multi-element expressions
    ("Love and hope and faith together.", "Triple element conjunction"),
    ("Anger plus rage plus fury.", "Multiple similar emotions"),
    ("Peace follows after conflict ends.", "Sequential transformation"),

    # Edge cases: Different operator combinations
    ("Joy intensifies love.", "Intensification operator"),
    ("Fear conflicts with courage.", "Conflict operator"),
    ("Wisdom sequences into action.", "Sequence operator"),

    # Edge cases: Foundation-tagged expressions
    ("Faith in the divine principle.", "Foundation reference"),
    ("Material wealth builds foundations.", "Material → Foundation"),
    ("The law guides behavior.", "Foundation-based guidance"),

    # Edge cases: Cross-world interactions
    ("Physical health enhances mental clarity.", "Physical → Mental"),
    ("Emotional pain triggers physical symptoms.", "Emotional → Physical"),
    ("Spiritual insight transforms worldly desires.", "Spiritual → Material"),

    # Additional edge cases: Complex operator sequences
    ("Love intensifies and then transforms into devotion.", "Multi-operator sequence"),
    ("Anger conflicts with peace.", "Binary conflict"),
    ("Knowledge sequences through wisdom to enlightenment.", "Triple sequence"),

    # Additional edge cases: Boundary noetics
    ("The highest principle guides all.", "Upper noetic boundary"),
    ("The lowest instinct drives survival.", "Lower noetic boundary"),

    # Additional edge cases: Mixed-world compound expressions
    ("Mental clarity and physical strength combine.", "Mental + Physical"),
    ("Spiritual peace conflicts with material desire.", "Spiritual vs Material"),
    ("Emotional joy intensifies physical vitality.", "Emotional amplifies Physical"),

    # Additional edge cases: Negation and acquisition
    ("Loss of faith causes despair.", "Negation operator"),
    ("Absence of love creates emptiness.", "Absence/negation"),
]

# =============================================================================
# INVERSION MODES TO TEST
# =============================================================================

INVERSION_MODES: List[Tuple[InversionMode, str]] = [
    (InversionMode.Opposite, "Opposite"),
    (InversionMode.Dual, "Dual"),
    (InversionMode.Mirror, "Mirror"),
    (InversionMode.ReverseCausal, "ReverseCausal"),
]

# =============================================================================
# CANONICAL VALIDATION FUNCTIONS
# =============================================================================

def validate_world(world: str) -> Tuple[bool, str]:
    """Validate that world is in {A, B, C, D}."""
    if world in ALLOWED_WORLDS:
        return True, ""
    return False, f"Invalid world '{world}' - must be in {ALLOWED_WORLDS}"


def validate_noetic(noetic_str: str) -> Tuple[bool, str]:
    """Validate that noetic is in range 1-10."""
    try:
        noetic = int(noetic_str)
        if noetic in ALLOWED_NOETICS:
            return True, ""
        return False, f"Invalid noetic '{noetic}' - must be 1-10"
    except ValueError:
        return False, f"Invalid noetic '{noetic_str}' - must be integer"


def validate_element(element: str) -> Tuple[bool, str]:
    """Validate TKS element format (e.g., 'D5', 'B2', 'C3.1')."""
    # Strip sense notation if present
    base_element = element.split(".")[0] if "." in element else element

    if len(base_element) < 2:
        return False, f"Invalid element '{element}' - too short"

    world = base_element[0]
    noetic_str = base_element[1:]

    # Validate world
    valid_world, msg = validate_world(world)
    if not valid_world:
        return False, f"Element '{element}': {msg}"

    # Validate noetic
    valid_noetic, msg = validate_noetic(noetic_str)
    if not valid_noetic:
        return False, f"Element '{element}': {msg}"

    return True, ""


def validate_operator(op: str) -> Tuple[bool, str]:
    """Validate that operator is in ALLOWED_OPS."""
    if op in ALLOWED_OPS:
        return True, ""
    return False, f"Invalid operator '{op}' - must be in {ALLOWED_OPS}"


def validate_expression(expr: TKSExpression) -> Tuple[bool, List[str]]:
    """
    Validate entire TKS expression for canonical format.

    Returns:
        (success, error_messages)
    """
    errors = []

    # Validate elements
    for element in expr.elements:
        valid, msg = validate_element(element)
        if not valid:
            errors.append(msg)

    # Validate operators
    for op in expr.ops:
        valid, msg = validate_operator(op)
        if not valid:
            errors.append(msg)

    # Validate foundations (if present)
    if hasattr(expr, 'foundations') and expr.foundations:
        for fid, world in expr.foundations:
            if fid not in ALLOWED_FOUNDATIONS:
                errors.append(f"Invalid foundation '{fid}' - must be 1-7")
            if world and world not in ALLOWED_WORLDS:
                errors.append(f"Invalid foundation world '{world}' - must be in {ALLOWED_WORLDS}")

    return len(errors) == 0, errors


# =============================================================================
# FUZZ TEST PIPELINE
# =============================================================================

def run_pipeline_test(
    story: str,
    description: str,
    inversion_mode: InversionMode,
    mode_name: str
) -> Dict[str, Any]:
    """
    Run single pipeline test: story → encode → invert → decode.

    Returns:
        Dict with test results and validation info
    """
    result = {
        "story": story,
        "description": description,
        "inversion_mode": mode_name,
        "success": True,
        "errors": [],
        "original_expr": None,
        "inverted_expr": None,
        "decoded_story": None,
    }

    try:
        # Step 1: Encode story
        original_expr = EncodeStory(story, strict=False)
        result["original_expr"] = f"{','.join(original_expr.elements)} | {','.join(original_expr.ops)}"

        # Validate original expression
        valid, errors = validate_expression(original_expr)
        if not valid:
            result["success"] = False
            result["errors"].extend([f"Original expr: {e}" for e in errors])

        # Step 2: Invert expression
        if len(original_expr.elements) > 0:
            inverted = total_inversion(
                elements=list(original_expr.elements),
                ops=list(original_expr.ops),
                mode=inversion_mode,
            )

            # Build inverted TKSExpression
            inverted_expr = TKSExpression(
                elements=inverted.get("elements", inverted.get("chain", [])),
                ops=inverted.get("ops", []),
            )
            result["inverted_expr"] = f"{','.join(inverted_expr.elements)} | {','.join(inverted_expr.ops)}"

            # Validate inverted expression
            valid, errors = validate_expression(inverted_expr)
            if not valid:
                result["success"] = False
                result["errors"].extend([f"Inverted expr: {e}" for e in errors])

            # Step 3: Decode inverted expression
            decoded = DecodeStory(inverted_expr)
            result["decoded_story"] = decoded[:100]  # Truncate for display

        else:
            result["success"] = False
            result["errors"].append("Original expression has no elements")

    except Exception as e:
        result["success"] = False
        result["errors"].append(f"Exception: {type(e).__name__}: {str(e)}")

    return result


def run_all_fuzz_tests() -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Run all fuzz tests.

    Returns:
        (total_tests, passed_tests, all_results)
    """
    all_results = []
    total_tests = 0
    passed_tests = 0

    print("=" * 80)
    print("TKS PIPELINE FUZZ TESTING")
    print("=" * 80)
    print(f"Testing {len(TEST_STORIES)} stories × {len(INVERSION_MODES)} modes")
    print(f"= {len(TEST_STORIES) * len(INVERSION_MODES)} total tests")
    print()

    for story, description in TEST_STORIES:
        for mode, mode_name in INVERSION_MODES:
            total_tests += 1
            result = run_pipeline_test(story, description, mode, mode_name)
            all_results.append(result)

            if result["success"]:
                passed_tests += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"

            print(f"{status} | {description:30s} | {mode_name:15s}")

            if not result["success"]:
                for error in result["errors"]:
                    print(f"       ERROR: {error}")

    return total_tests, passed_tests, all_results


# =============================================================================
# SUMMARY AND REPORTING
# =============================================================================

def print_summary(total: int, passed: int, results: List[Dict[str, Any]]):
    """Print test summary and statistics."""
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests:  {total}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    print(f"Pass rate:    {pass_rate:.1f}%")
    print()

    if failed > 0:
        print("FAILED TESTS:")
        print("-" * 80)
        for result in results:
            if not result["success"]:
                print(f"  Story: {result['story']}")
                print(f"  Mode:  {result['inversion_mode']}")
                print(f"  Errors:")
                for error in result["errors"]:
                    print(f"    - {error}")
                print()

    # Validation statistics
    print("CANONICAL VALIDATION:")
    print(f"  Allowed worlds:      {sorted(ALLOWED_WORLDS)}")
    print(f"  Allowed noetics:     1-10")
    print(f"  Allowed foundations: 1-7")
    print(f"  Allowed operators:   {sorted(ALLOWED_OPS)}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for fuzz testing."""
    total, passed, results = run_all_fuzz_tests()
    print_summary(total, passed, results)

    # Exit with appropriate code
    if passed == total:
        print("[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print(f"[FAILURE] {total - passed} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

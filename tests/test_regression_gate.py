"""
Regression Gate Test

Tests canonical roundtrip transformations for TKS:
1. Encode story to TKS expression
2. Invert across multiple axes
3. Decode back to narrative
4. Assert canonical validity

MUST complete in < 5 seconds to maintain fast CI.

Canon constraints:
- Worlds: A, B, C, D only
- Noetics: 1-10 (involution pairs: 2<->3, 5<->6, 8<->9; self-duals: 1,4,7,10)
- Foundations: 1-7
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (9 total)
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative import EncodeStory
from narrative.decoder import DecodeStory
from narrative.types import TKSExpression
from inversion.engine import total_inversion, InversionMode, DialConfig
from scripts.canonical_validator import validate_canonical

# Canonical test stories covering major TKS patterns
TEST_STORIES = [
    # Story 1: Simple emotion (C-world, noetic polarity)
    {
        "story": "She felt fear.",
        "description": "Basic emotion C3 (fear) with agent D5 (woman)",
        "expected_worlds": {"C", "D"},
        "expected_noetics": {3, 5},
    },
    # Story 2: Causal chain (physical -> emotional)
    {
        "story": "A woman loved a man.",
        "description": "D5 (woman) -> love relationship -> D6 (man)",
        "expected_worlds": {"C", "D"},
        "expected_noetics": {2, 5, 6},
    },
    # Story 3: Mental-emotional causation
    {
        "story": "Fear caused anxiety.",
        "description": "C3 (fear) -> C3 (anxiety) - causal emotion chain",
        "expected_worlds": {"C"},
        "expected_noetics": {3},
    },
    # Story 4: Physical health state
    {
        "story": "Health and vitality.",
        "description": "D2 (health) +T D2 (vitality) - positive physical",
        "expected_worlds": {"D"},
        "expected_noetics": {2},
    },
    # Story 5: Mental belief
    {
        "story": "Positive belief.",
        "description": "B2 (positive belief) - mental positive",
        "expected_worlds": {"B"},
        "expected_noetics": {2},
    },
    # Story 6: Noetic polarity pair
    {
        "story": "Joy and sadness.",
        "description": "C2 (joy) +T C3 (sadness) - emotional polarity",
        "expected_worlds": {"C"},
        "expected_noetics": {2, 3},
    },
    # Story 7: Gender noetics
    {
        "story": "A man and a woman.",
        "description": "D6 (man) +T D5 (woman) - gender polarity",
        "expected_worlds": {"D"},
        "expected_noetics": {5, 6},
    },
    # Story 8: Cause-effect chain
    {
        "story": "Power and control.",
        "description": "D8 (power/authority) - cause/elevation",
        "expected_worlds": {"D"},
        "expected_noetics": {8},
    },
    # Story 9: Multi-world expression
    {
        "story": "She felt grief.",
        "description": "D5 (woman) -> C3 (grief/sadness) - cross-world",
        "expected_worlds": {"C", "D"},
        "expected_noetics": {3, 5},
    },
    # Story 10: Mental-physical bridge
    {
        "story": "Thought and action.",
        "description": "B1 (thought/mind) +T D1 (action/body) - bridge worlds",
        "expected_worlds": {"B", "D"},
        "expected_noetics": {1},
    },
]


def test_canonical_roundtrip():
    """
    Test encode -> invert -> decode roundtrip for canonical validity.

    For each test story:
    1. Encode to TKS expression
    2. Validate it's canonical
    3. Invert across Element axis (N+W)
    4. Validate inverted is canonical
    5. Decode inverted expression
    6. Assert no exceptions
    """
    print("\n" + "=" * 80)
    print("REGRESSION GATE: Canonical Roundtrip Test")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, test_case in enumerate(TEST_STORIES, 1):
        story = test_case["story"]
        description = test_case["description"]

        print(f"\n[Test {i}/{len(TEST_STORIES)}] {description}")
        print(f"  Story: \"{story}\"")

        try:
            # Step 1: Encode story
            expr = EncodeStory(story, strict=False)
            print(f"  Encoded: {expr.canonical}")

            # Step 2: Validate original is canonical
            is_valid, errors = validate_canonical(expr)
            if not is_valid:
                print(f"  [FAIL] Original encoding not canonical:")
                for error in errors:
                    print(f"    - {error}")
                failed += 1
                continue

            # Verify expected worlds/noetics if specified
            if "expected_worlds" in test_case:
                actual_worlds = {e.world for e in expr.element_refs}
                expected_worlds = test_case["expected_worlds"]
                if not actual_worlds.issubset(expected_worlds):
                    print(f"  [WARNING] Unexpected worlds: {actual_worlds} not subset of {expected_worlds}")

            if "expected_noetics" in test_case:
                actual_noetics = {e.noetic for e in expr.element_refs}
                expected_noetics = test_case["expected_noetics"]
                if not actual_noetics.issubset(expected_noetics):
                    print(f"  [WARNING] Unexpected noetics: {actual_noetics} not subset of {expected_noetics}")

            # Step 3: Invert expression (Element axis = Noetic + World)
            # This tests the most common inversion pattern
            dial = DialConfig(
                mode=InversionMode.Opposite,
                axes={"element": True, "noetic": True, "world": True}
            )

            inverted_result = total_inversion(
                elements=expr.elements,
                foundations=expr.foundations if hasattr(expr, 'foundations') else [],
                acquisitions=expr.acquisitions if hasattr(expr, 'acquisitions') else [],
                ops=expr.ops,
                mode=InversionMode.Opposite,
                dial=dial,
            )

            # Build inverted expression
            inverted_expr = TKSExpression(
                elements=inverted_result["elements"],
                ops=inverted_result["ops"],
                foundations=inverted_result.get("foundations", []),
                acquisitions=inverted_result.get("acquisitions", []),
            )

            print(f"  Inverted: {inverted_expr.canonical}")

            # Step 4: Validate inverted is canonical
            is_valid_inv, errors_inv = validate_canonical(inverted_expr)
            if not is_valid_inv:
                print(f"  [FAIL] Inverted expression not canonical:")
                for error in errors_inv:
                    print(f"    - {error}")
                failed += 1
                continue

            # Step 5: Decode inverted expression (just verify no exception)
            try:
                decoded_story = DecodeStory(inverted_expr)
                print(f"  Decoded: \"{decoded_story}\"")
            except Exception as e:
                print(f"  [FAIL] Decode failed: {e}")
                failed += 1
                continue

            # Success
            print(f"  [PASS] Roundtrip complete, all outputs canonical")
            passed += 1

        except Exception as e:
            print(f"  [FAIL] Exception during roundtrip: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(TEST_STORIES)}")
    print(f"Failed: {failed}/{len(TEST_STORIES)}")

    if failed > 0:
        print("\n[FAILURE] Regression gate failed - non-canonical outputs detected!")
    else:
        print("\n[SUCCESS] All roundtrips produced canonical outputs!")

    # Assert for pytest
    assert failed == 0, f"Regression gate failed: {failed}/{len(TEST_STORIES)} tests failed"


def test_operator_preservation():
    """
    Test that canonical operators are preserved through inversion.

    All operators must be in ALLOWED_OPS after inversion.
    """
    print("\n" + "=" * 80)
    print("REGRESSION GATE: Operator Preservation Test")
    print("=" * 80)

    from narrative.constants import ALLOWED_OPS

    # Test expressions with different operators
    test_equations = [
        ("D5 +T C2", "Combination operator"),
        ("D5 -T C3", "Subtraction operator"),
        ("B2 -> C2", "Causal forward"),
        ("C3 <- D5", "Causal reverse"),
        ("C2 *T C4", "Intensification"),
        ("B2 /T B3", "Conflict"),
        ("D7 o D8 o D9", "Sequential composition"),
    ]

    passed = 0
    failed = 0

    from narrative.encoder import parse_equation

    for equation, description in test_equations:
        print(f"\n  Testing: {equation} ({description})")

        try:
            expr = parse_equation(equation, strict=True)

            # Verify original ops are canonical
            for op in expr.ops:
                if op not in ALLOWED_OPS:
                    print(f"    [FAIL] Non-canonical operator in original: {op}")
                    failed += 1
                    continue

            # Invert
            dial = DialConfig(mode=InversionMode.Opposite)
            inverted_result = total_inversion(
                elements=expr.elements,
                ops=expr.ops,
                mode=InversionMode.Opposite,
                dial=dial,
            )

            inverted_ops = inverted_result["ops"]
            print(f"    Inverted ops: {inverted_ops}")

            # Verify all inverted ops are canonical
            all_valid = True
            for op in inverted_ops:
                if op not in ALLOWED_OPS:
                    print(f"    [FAIL] Non-canonical operator after inversion: {op}")
                    all_valid = False

            if all_valid:
                print(f"    [PASS] All operators canonical")
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"    [FAIL] Exception: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"Passed: {passed}/{len(test_equations)}")
    print(f"Failed: {failed}/{len(test_equations)}")

    if failed > 0:
        print("[FAILURE] Operator preservation test failed!")
    else:
        print("[SUCCESS] All operators preserved as canonical!")

    # Assert for pytest
    assert failed == 0, f"Operator preservation test failed: {failed}/{len(test_equations)} tests failed"


def main():
    """Run all regression gate tests."""
    import time

    start_time = time.time()

    print("\n" + "=" * 80)
    print("TKS REGRESSION GATE - Phase 4")
    print("=" * 80)
    print("Testing canonical validity of encode -> invert -> decode pipeline")
    print("Time budget: < 5 seconds")
    print()

    # Run tests
    test1_pass = test_canonical_roundtrip()
    test2_pass = test_operator_preservation()

    elapsed = time.time() - start_time

    # Time check
    print("\n" + "=" * 80)
    print(f"Execution time: {elapsed:.2f}s")

    if elapsed >= 5.0:
        print(f"[WARNING] Exceeded 5s time budget!")
    else:
        print(f"[OK] Within 5s time budget")

    # Final result
    print("=" * 80)

    if test1_pass and test2_pass:
        print("\n[SUCCESS] Regression gate PASSED - all outputs canonical!")
        print("=" * 80)
        return 0
    else:
        print("\n[FAILURE] Regression gate FAILED - non-canonical outputs detected!")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

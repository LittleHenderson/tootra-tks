"""
Test script to verify that sub-foundation tags flow correctly through scenario_inversion.py

This script tests:
1. Encoding preserves foundations
2. ScenarioInvert passes foundations through correctly
3. InvertStory pipeline maintains foundations end-to-end
"""

from scenario_inversion import (
    EncodeStory,
    DecodeStory,
    ScenarioInvert,
    InvertStory,
    TKSExpression,
)


def test_encode_preserves_foundations():
    """Test that EncodeStory attaches foundations from story content."""
    print("\n=== Test 1: EncodeStory preserves foundations ===")

    # Story with clear foundation keywords (F1 - Keter/Crown)
    story = "divine light flows through consciousness"

    expr = EncodeStory(story)

    print(f"Input story: {story}")
    print(f"Encoded elements: {expr.elements}")
    print(f"Encoded foundations: {expr.foundations}")

    if expr.foundations:
        print("[PASS] Foundations were detected and attached")
        for fid, subfound in expr.foundations:
            print(f"  Foundation: F{fid}, SubFoundation: {subfound}")
    else:
        print("[INFO] No foundations detected (may be expected for this story)")

    return expr


def test_scenario_invert_preserves_foundations():
    """Test that ScenarioInvert correctly passes foundations through inversion."""
    print("\n=== Test 2: ScenarioInvert preserves foundations ===")

    # Create expression with explicit foundations
    expr = TKSExpression(
        elements=["D5", "C2"],
        ops=["->"],
        foundations=[(3, None), (5, "A")],  # F3 with no subfound, F5 with subfound "A"
        raw="test expression"
    )

    print(f"Input expr: elements={expr.elements}, foundations={expr.foundations}")

    # Invert with Foundation axis enabled
    axes = {"Foundation", "SubFoundation"}
    inverted = ScenarioInvert(expr, axes, "soft")

    print(f"Output expr: elements={inverted.elements}, foundations={inverted.foundations}")

    # Verify foundations were processed (may be inverted)
    if inverted.foundations:
        print("[PASS] Foundations were preserved through inversion")
        for i, (fid, subfound) in enumerate(inverted.foundations):
            orig_fid, orig_subfound = expr.foundations[i] if i < len(expr.foundations) else (None, None)
            print(f"  Foundation {i}: F{orig_fid},{orig_subfound} -> F{fid},{subfound}")
    else:
        print("[FAIL] Foundations were lost during inversion")

    return inverted


def test_invert_story_pipeline():
    """Test that InvertStory maintains foundations through full pipeline."""
    print("\n=== Test 3: InvertStory pipeline end-to-end ===")

    story = "love creates joy and happiness"

    result = InvertStory(
        story=story,
        axes={"Foundation"},
        mode="soft",
        strict=False
    )

    print(f"Original story: {story}")
    print(f"Original expr: elements={result['expr_original'].elements}")
    print(f"Original foundations: {result['expr_original'].foundations}")
    print(f"\nInverted expr: elements={result['expr_inverted'].elements}")
    print(f"Inverted foundations: {result['expr_inverted'].foundations}")
    print(f"Inverted story: {result['story_inverted']}")

    # Check that foundations flowed through
    orig_has_foundations = bool(result['expr_original'].foundations)
    inv_has_foundations = bool(result['expr_inverted'].foundations)

    if orig_has_foundations and inv_has_foundations:
        print("[PASS] Foundations preserved through full pipeline")
    elif orig_has_foundations and not inv_has_foundations:
        print("[FAIL] Foundations lost during inversion")
    elif not orig_has_foundations:
        print("[INFO] No foundations detected in original story")

    return result


def test_none_subfoundation_handling():
    """Test that None subfoundations are handled correctly."""
    print("\n=== Test 4: None subfoundation handling ===")

    # Expression with mix of None and non-None subfoundations
    expr = TKSExpression(
        elements=["A1", "B5", "C2"],
        ops=["+T", "->"],
        foundations=[(1, None), (3, "B"), (5, None)],  # Mix of None and string subfoundations
        raw="test"
    )

    print(f"Input foundations: {expr.foundations}")

    # Invert with SubFoundation axis
    inverted = ScenarioInvert(expr, {"SubFoundation"}, "soft")

    print(f"Output foundations: {inverted.foundations}")

    # Verify structure is preserved
    if len(inverted.foundations) == len(expr.foundations):
        print("[PASS] Foundation count preserved")

        all_valid = True
        for i, (fid, subfound) in enumerate(inverted.foundations):
            if not isinstance(fid, int):
                print(f"[FAIL] Foundation ID at index {i} is not int: {type(fid)}")
                all_valid = False
            if subfound is not None and not isinstance(subfound, str):
                print(f"[FAIL] SubFoundation at index {i} is not None or str: {type(subfound)}")
                all_valid = False

        if all_valid:
            print("[PASS] All foundation types are correct (int, Optional[str])")
    else:
        print(f"[FAIL] Foundation count mismatch: {len(expr.foundations)} -> {len(inverted.foundations)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Foundation Flow Verification Tests")
    print("=" * 60)

    try:
        test_encode_preserves_foundations()
        test_scenario_invert_preserves_foundations()
        test_invert_story_pipeline()
        test_none_subfoundation_handling()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

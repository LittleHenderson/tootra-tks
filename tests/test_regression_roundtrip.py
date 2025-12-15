"""
Regression tests to ensure the full pipeline remains stable.

Tests representative stories from the rulebook examples to verify:
- Encoding produces valid expressions
- Inversion creates different expressions
- Decoding produces non-empty stories
- Strict mode validation works correctly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenario_inversion import InvertStory, EncodeStory, DecodeStory
from narrative import encode_story_full


def test_roundtrip_woman_money():
    """Regression test for woman/money/partner scenario."""
    story = "A woman hides money from her partner because she fears losing control."
    result = InvertStory(story, axes={"World", "Noetic"}, mode="soft", strict=False)

    # Assert original expression has elements
    assert len(result["expr_original"].elements) > 0, \
        "Original expression should have elements"

    # Assert inverted expression differs
    assert result["expr_inverted"].elements != result["expr_original"].elements, \
        "Inverted expression should differ from original"

    # Assert inverted story is non-empty
    assert len(result["story_inverted"]) > 0, \
        "Inverted story should be non-empty"

    # Verify structure integrity
    assert "expr_original" in result
    assert "expr_inverted" in result
    assert "story_inverted" in result

    print(f"[PASS] Woman/money scenario passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {result['expr_original'].elements}")
    print(f"  Inverted expr: {result['expr_inverted'].elements}")
    print(f"  Inverted: {result['story_inverted']}")


def test_roundtrip_love_story():
    """Regression test for simple love story."""
    story = "She loved him."
    result = InvertStory(story, axes={"World", "Noetic"}, mode="soft")

    # Assert original expression has elements
    assert len(result["expr_original"].elements) > 0, \
        "Original expression should have elements"

    # Assert inverted expression differs
    assert result["expr_inverted"].elements != result["expr_original"].elements, \
        "Inverted expression should differ from original"

    # Assert inverted story is non-empty
    assert len(result["story_inverted"]) > 0, \
        "Inverted story should be non-empty"

    # Check that operators were preserved (if any)
    if len(result["expr_original"].ops) > 0:
        assert len(result["expr_inverted"].ops) > 0, \
            "Operators should be preserved"

    print(f"[PASS] Love story passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {result['expr_original'].elements}")
    print(f"  Inverted expr: {result['expr_inverted'].elements}")
    print(f"  Inverted: {result['story_inverted']}")


def test_roundtrip_fear_illness():
    """Regression test for fear/illness causal scenario."""
    story = "Fear causes illness."
    result = InvertStory(story, axes={"World", "Noetic"}, mode="soft")

    # Assert original expression has elements
    assert len(result["expr_original"].elements) > 0, \
        "Original expression should have elements"

    # Assert inverted expression differs
    assert result["expr_inverted"].elements != result["expr_original"].elements, \
        "Inverted expression should differ from original"

    # Assert inverted story is non-empty
    assert len(result["story_inverted"]) > 0, \
        "Inverted story should be non-empty"

    # For causal scenarios, check that operators exist
    assert len(result["expr_original"].ops) > 0, \
        "Causal scenario should have operators"
    assert len(result["expr_inverted"].ops) > 0, \
        "Inverted scenario should have operators"

    print(f"[PASS] Fear/illness scenario passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {result['expr_original'].elements}")
    print(f"  Inverted expr: {result['expr_inverted'].elements}")
    print(f"  Inverted: {result['story_inverted']}")


def test_roundtrip_strict_mode_valid():
    """Strict mode roundtrip with valid story."""
    story = "A woman loved a man."
    result = InvertStory(story, axes={"Element"}, mode="soft", strict=True)

    # Should succeed without raising ValueError
    assert len(result["story_inverted"]) > 0, \
        "Strict mode should succeed with valid story"

    # Verify all components present
    assert len(result["expr_original"].elements) > 0
    assert len(result["expr_inverted"].elements) > 0

    print(f"[PASS] Strict mode valid story passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {result['expr_original'].elements}")
    print(f"  Inverted expr: {result['expr_inverted'].elements}")
    print(f"  Inverted: {result['story_inverted']}")


def test_roundtrip_strict_mode_rejects_unknown():
    """Strict mode rejects unknown tokens."""
    story = "A zorblax loved a flibbert."

    try:
        result = InvertStory(story, axes={"Element"}, mode="soft", strict=True)
        assert False, "Should have raised ValueError for unknown tokens"
    except ValueError as e:
        # Should mention unknown tokens or the specific word
        error_msg = str(e).lower()
        assert "zorblax" in error_msg or "flibbert" in error_msg or "unknown" in error_msg, \
            f"Error should mention unknown token, got: {e}"
        print(f"[PASS] Strict mode rejection passed")
        print(f"  Story: {story}")
        print(f"  Error (expected): {e}")


def test_roundtrip_element_axis():
    """Test Element axis inversion on a simple story."""
    story = "She wanted freedom."
    result = InvertStory(story, axes={"Element"}, mode="soft", strict=False)

    # Assert original expression has elements
    assert len(result["expr_original"].elements) > 0

    # Assert inverted expression differs
    assert result["expr_inverted"].elements != result["expr_original"].elements

    # Assert inverted story is non-empty
    assert len(result["story_inverted"]) > 0

    print(f"[PASS] Element axis inversion passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {result['expr_original'].elements}")
    print(f"  Inverted expr: {result['expr_inverted'].elements}")
    print(f"  Inverted: {result['story_inverted']}")


def test_roundtrip_hard_mode():
    """Test hard mode inversion."""
    story = "He gave her love."
    result = InvertStory(story, axes={"World", "Noetic"}, mode="hard", strict=False)

    # Assert original expression has elements
    assert len(result["expr_original"].elements) > 0

    # Assert inverted expression differs
    assert result["expr_inverted"].elements != result["expr_original"].elements

    # Assert inverted story is non-empty
    assert len(result["story_inverted"]) > 0

    print(f"[PASS] Hard mode inversion passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {result['expr_original'].elements}")
    print(f"  Inverted expr: {result['expr_inverted'].elements}")
    print(f"  Inverted: {result['story_inverted']}")


def test_encode_decode_stability():
    """Test that encoding and decoding are stable."""
    test_stories = [
        "She loved him.",
        "Fear causes illness.",
        "A woman hides money.",
        "He wanted freedom.",
        "She gave him trust.",
    ]

    for story in test_stories:
        # Encode
        expr = EncodeStory(story, strict=False)

        # Check encoding produced elements
        assert len(expr.elements) > 0, \
            f"Story '{story}' should encode to elements"

        # Decode
        decoded = DecodeStory(expr)

        # Check decoding produced non-empty string
        assert isinstance(decoded, str), \
            f"Decoded story should be string"
        assert len(decoded) > 0, \
            f"Decoded story should be non-empty"

    print(f"[PASS] Encode/decode stability passed for {len(test_stories)} stories")


if __name__ == "__main__":
    print("=" * 60)
    print("Running TKS Regression Roundtrip Tests")
    print("=" * 60)
    print()

    test_roundtrip_woman_money()
    print()

    test_roundtrip_love_story()
    print()

    test_roundtrip_fear_illness()
    print()

    test_roundtrip_strict_mode_valid()
    print()

    test_roundtrip_strict_mode_rejects_unknown()
    print()

    test_roundtrip_element_axis()
    print()

    test_roundtrip_hard_mode()
    print()

    test_encode_decode_stability()
    print()

    print("=" * 60)
    print("All regression roundtrip tests passed!")
    print("=" * 60)

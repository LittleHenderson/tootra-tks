"""
Smoke tests for TKS Scenario Inversion Knob.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scenario_inversion import (
    InvertStory,
    ScenarioInvert,
    ExplainInversion,
    EncodeStory,
    DecodeStory,
    parse_equation,
    TKSExpression,
)
from inversion.engine import TargetProfile


def test_invert_story_pipeline():
    """Test full InvertStory pipeline runs without crashing."""
    # Use lenient mode for this general pipeline test
    result = InvertStory(
        story="She loved him",
        axes={"World", "Noetic"},
        mode="soft",
        strict=False,
    )

    # Check result structure
    assert "expr_original" in result
    assert "expr_inverted" in result
    assert "story_inverted" in result

    # Check expressions have elements
    assert len(result["expr_original"].elements) > 0
    assert len(result["expr_inverted"].elements) > 0

    # Check story_inverted is a string
    assert isinstance(result["story_inverted"], str)
    assert len(result["story_inverted"]) > 0


def test_scenario_invert_element_axis():
    """Test ScenarioInvert with Element axis."""
    expr = TKSExpression(
        elements=["B5", "D3"],
        ops=["+T"],
    )

    result = ScenarioInvert(expr, axes={"Element"}, mode="hard")

    # Element inversion: B5 -> C6, D3 -> A2
    assert result.elements[0] == "C6"
    assert result.elements[1] == "A2"


def test_scenario_invert_world_only():
    """Test ScenarioInvert with World axis only."""
    expr = TKSExpression(
        elements=["B5", "C3"],
        ops=["+T"],
    )

    result = ScenarioInvert(expr, axes={"World"}, mode="soft")

    # World mirror: B->C, C->B (noetic unchanged)
    assert result.elements[0].startswith("C")
    assert result.elements[1].startswith("B")


def test_explain_inversion():
    """Test ExplainInversion produces output."""
    expr_orig = TKSExpression(elements=["B5", "D3"], ops=["+T"])
    expr_inv = TKSExpression(elements=["C6", "A2"], ops=["-T"])

    explanation = ExplainInversion(expr_orig, expr_inv)

    assert "INVERSION CHANGES" in explanation
    assert "B5" in explanation or "Element" in explanation


def test_parse_equation_comma():
    """Test parsing comma-separated equation."""
    expr = parse_equation("B5,+T,D3,-T,C8")

    assert expr.elements == ["B5", "D3", "C8"]
    assert expr.ops == ["+T", "-T"]


def test_parse_equation_space():
    """Test parsing space-separated equation."""
    expr = parse_equation("B5 +T D3 -T C8")

    assert expr.elements == ["B5", "D3", "C8"]
    assert expr.ops == ["+T", "-T"]


def test_encode_decode_roundtrip():
    """Test EncodeStory and DecodeStory produce output."""
    story = "She loved him"
    expr = EncodeStory(story, strict=False)  # Use lenient mode for this test

    assert len(expr.elements) > 0

    decoded = DecodeStory(expr)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_strict_mode_default_rejects_unknown():
    """Test that strict mode is default and rejects unknown tokens."""
    try:
        # Default should be strict=True
        InvertStory(
            story="The mysterious zorblax appeared",
            axes={"Element"},
            mode="soft",
        )
        assert False, "Expected ValueError for unknown tokens in default strict mode"
    except ValueError as e:
        # Should reject unknown token
        assert 'zorblax' in str(e)
        assert 'Unknown tokens' in str(e)


def test_lenient_mode_allows_unknown():
    """Test that lenient mode (strict=False) allows unknown tokens."""
    # Should not raise error
    result = InvertStory(
        story="The mysterious zorblax appeared",
        axes={"Element"},
        mode="soft",
        strict=False,  # Lenient mode
    )

    # Should complete successfully
    assert "expr_original" in result
    assert "expr_inverted" in result
    assert "story_inverted" in result


def test_strict_mode_valid_story_passes():
    """Test that strict mode works with valid stories."""
    # This should work because "woman", "loved", "man" are in lexicon
    result = InvertStory(
        story="A woman loved a man",
        axes={"World"},
        mode="soft",
        strict=True,
    )

    assert "expr_original" in result
    assert "expr_inverted" in result


def test_encode_story_strict_default():
    """Test that EncodeStory uses strict=True as default."""
    try:
        EncodeStory("The alien zorblax invaded")
        assert False, "Expected ValueError in default strict mode"
    except ValueError as e:
        assert 'zorblax' in str(e)


def test_encode_story_lenient_explicit():
    """Test that EncodeStory with strict=False allows unknown tokens."""
    # Should not raise
    expr = EncodeStory("The alien zorblax invaded", strict=False)
    assert len(expr.elements) >= 0


def test_targeted_mode_with_profile():
    """Test targeted mode with TargetProfile."""
    expr = TKSExpression(
        elements=["B5"],
        ops=[],
    )

    target = TargetProfile(
        enable=True,
        from_world="B",
        to_world="D",
    )

    result = ScenarioInvert(expr, axes={"World"}, mode="targeted", target=target)

    # With targeted profile B->D, should get D5
    assert result.elements[0].startswith("D")


if __name__ == "__main__":
    test_invert_story_pipeline()
    test_scenario_invert_element_axis()
    test_scenario_invert_world_only()
    test_explain_inversion()
    test_parse_equation_comma()
    test_parse_equation_space()
    test_encode_decode_roundtrip()
    test_strict_mode_default_rejects_unknown()
    test_lenient_mode_allows_unknown()
    test_strict_mode_valid_story_passes()
    test_encode_story_strict_default()
    test_encode_story_lenient_explicit()
    test_targeted_mode_with_profile()
    print("All scenario inversion tests passed!")

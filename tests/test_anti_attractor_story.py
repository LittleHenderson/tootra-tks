"""
Story-based regression tests for anti-attractor synthesis pipeline.

Tests verify the end-to-end pipeline: story → encode → anti-attractor → decode → story

These tests ensure stability of the anti-attractor synthesis algorithm when applied
to natural language stories, focusing on regression/stability rather than specific
output values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from scenario_inversion import EncodeStory, DecodeStory, AntiAttractorInvert


# =============================================================================
# BASIC PIPELINE TESTS
# =============================================================================

def test_anti_attractor_joy_story():
    """Test anti-attractor pipeline with simple joy story."""
    story = "A woman feels joy"

    # Encode story to TKS expression (lenient mode for pipeline testing)
    expr_original = EncodeStory(story, strict=False)

    # Apply anti-attractor inversion
    result = AntiAttractorInvert(expr_original)
    expr_counter = result["expr_inverted"]

    # Decode back to story
    counter_story = DecodeStory(expr_counter)

    # Assert original expression has elements
    assert len(expr_original.elements) > 0, \
        "Original expression should have elements"

    # Assert counter expression has elements
    assert len(expr_counter.elements) > 0, \
        "Counter expression should have elements"

    # Assert counter-story is non-empty
    assert len(counter_story) > 0, \
        "Counter-story should be non-empty"

    # Assert counter-story differs from original (anti-attractor should produce different output)
    assert counter_story != story, \
        "Counter-story should differ from original story"

    # Assert elements changed (anti-attractor inverts the attractor signature)
    assert expr_counter.elements != expr_original.elements, \
        "Counter expression elements should differ from original"

    print(f"[PASS] Joy story anti-attractor test passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {expr_original.elements}")
    print(f"  Counter expr: {expr_counter.elements}")
    print(f"  Counter story: {counter_story}")


def test_anti_attractor_belief_story():
    """Test anti-attractor pipeline with belief/thinking story."""
    story = "Clear thinking leads to positive belief"

    # Encode story to TKS expression (lenient mode for pipeline testing)
    expr_original = EncodeStory(story, strict=False)

    # Apply anti-attractor inversion
    result = AntiAttractorInvert(expr_original)
    expr_counter = result["expr_inverted"]

    # Decode back to story
    counter_story = DecodeStory(expr_counter)

    # Assert original expression has elements
    assert len(expr_original.elements) > 0, \
        "Original expression should have elements"

    # Assert counter expression has elements
    assert len(expr_counter.elements) > 0, \
        "Counter expression should have elements"

    # Assert counter-story is non-empty
    assert len(counter_story) > 0, \
        "Counter-story should be non-empty"

    # Assert counter-story differs from original
    assert counter_story != story, \
        "Counter-story should differ from original story"

    # Assert elements changed
    assert expr_counter.elements != expr_original.elements, \
        "Counter expression elements should differ from original"

    print(f"[PASS] Belief story anti-attractor test passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {expr_original.elements}")
    print(f"  Counter expr: {expr_counter.elements}")
    print(f"  Counter story: {counter_story}")


def test_anti_attractor_fear_anger_story():
    """Test anti-attractor pipeline with complex emotional story."""
    story = "Fear without love causes anger"

    # Encode story to TKS expression (lenient mode for pipeline testing)
    expr_original = EncodeStory(story, strict=False)

    # Apply anti-attractor inversion
    result = AntiAttractorInvert(expr_original)
    expr_counter = result["expr_inverted"]

    # Decode back to story
    counter_story = DecodeStory(expr_counter)

    # Assert original expression has elements
    assert len(expr_original.elements) > 0, \
        "Original expression should have elements"

    # Assert counter expression has elements
    assert len(expr_counter.elements) > 0, \
        "Counter expression should have elements"

    # Assert counter-story is non-empty
    assert len(counter_story) > 0, \
        "Counter-story should be non-empty"

    # Assert counter-story differs from original
    assert counter_story != story, \
        "Counter-story should differ from original story"

    # Assert elements changed
    assert expr_counter.elements != expr_original.elements, \
        "Counter expression elements should differ from original"

    # Verify operators were preserved (if any)
    if len(expr_original.ops) > 0:
        assert len(expr_counter.ops) > 0, \
            "Operators should be present in counter expression"

    print(f"[PASS] Fear/anger story anti-attractor test passed")
    print(f"  Original: {story}")
    print(f"  Original expr: {expr_original.elements}")
    print(f"  Counter expr: {expr_counter.elements}")
    print(f"  Counter story: {counter_story}")


# =============================================================================
# PARAMETRIZED TESTS FOR PIPELINE STABILITY
# =============================================================================

@pytest.mark.parametrize("story", [
    "A woman feels joy",
    "Clear thinking leads to positive belief",
    "Fear without love causes anger",
    "Love heals pain",
    "Wisdom creates harmony",
    "Anger leads to conflict",
    "She loved him",
    "Fear causes illness",
])
def test_anti_attractor_pipeline_stability(story):
    """Parametrized test to ensure anti-attractor pipeline stability across multiple stories."""
    # Encode story to TKS expression (lenient mode for pipeline testing)
    expr_original = EncodeStory(story, strict=False)

    # Apply anti-attractor inversion
    result = AntiAttractorInvert(expr_original)
    expr_counter = result["expr_inverted"]

    # Decode back to story
    counter_story = DecodeStory(expr_counter)

    # Pipeline stability assertions
    assert len(expr_original.elements) > 0, \
        f"Original expression should have elements for story: {story}"

    assert len(expr_counter.elements) > 0, \
        f"Counter expression should have elements for story: {story}"

    assert len(counter_story) > 0, \
        f"Counter-story should be non-empty for story: {story}"

    assert counter_story != story, \
        f"Counter-story should differ from original for story: {story}"

    assert expr_counter.elements != expr_original.elements, \
        f"Counter expression should differ from original for story: {story}"

    print(f"[PASS] Pipeline stability test passed for: {story}")
    print(f"  Original expr: {expr_original.elements}")
    print(f"  Counter expr: {expr_counter.elements}")
    print(f"  Counter story: {counter_story}")


# =============================================================================
# SIGNATURE RETURN TEST
# =============================================================================

def test_anti_attractor_with_signature():
    """Test that signature is returned when return_signature=True."""
    story = "A woman feels joy"

    # Encode story to TKS expression (lenient mode for pipeline testing)
    expr_original = EncodeStory(story, strict=False)

    # Apply anti-attractor inversion with signature
    result = AntiAttractorInvert(expr_original, return_signature=True)

    # Assert result contains both expr_inverted and signature
    assert "expr_inverted" in result, \
        "Result should contain expr_inverted"

    assert "signature" in result, \
        "Result should contain signature when return_signature=True"

    # Assert signature has required attributes
    signature = result["signature"]
    assert hasattr(signature, "element_counts"), \
        "Signature should have element_counts"

    assert hasattr(signature, "polarity"), \
        "Signature should have polarity"

    assert hasattr(signature, "dominant_world"), \
        "Signature should have dominant_world"

    assert hasattr(signature, "dominant_noetic"), \
        "Signature should have dominant_noetic"

    assert hasattr(signature, "ops_distribution"), \
        "Signature should have ops_distribution"

    # Assert counter expression is valid
    expr_counter = result["expr_inverted"]
    assert len(expr_counter.elements) > 0, \
        "Counter expression should have elements"

    counter_story = DecodeStory(expr_counter)
    assert len(counter_story) > 0, \
        "Counter-story should be non-empty"

    print(f"[PASS] Signature return test passed")
    print(f"  Original: {story}")
    print(f"  Signature polarity: {signature.polarity}")
    print(f"  Signature dominant world: {signature.dominant_world}")
    print(f"  Signature dominant noetic: {signature.dominant_noetic}")
    print(f"  Counter expr: {expr_counter.elements}")
    print(f"  Counter story: {counter_story}")


def test_anti_attractor_without_signature():
    """Test that signature is not returned when return_signature=False (default)."""
    story = "Clear thinking leads to positive belief"

    # Encode story to TKS expression (lenient mode for pipeline testing)
    expr_original = EncodeStory(story, strict=False)

    # Apply anti-attractor inversion without signature (default)
    result = AntiAttractorInvert(expr_original, return_signature=False)

    # Assert result contains expr_inverted
    assert "expr_inverted" in result, \
        "Result should contain expr_inverted"

    # Assert signature is NOT in result
    assert "signature" not in result, \
        "Result should not contain signature when return_signature=False"

    # Assert counter expression is valid
    expr_counter = result["expr_inverted"]
    assert len(expr_counter.elements) > 0, \
        "Counter expression should have elements"

    counter_story = DecodeStory(expr_counter)
    assert len(counter_story) > 0, \
        "Counter-story should be non-empty"

    print(f"[PASS] No signature test passed")
    print(f"  Original: {story}")
    print(f"  Counter expr: {expr_counter.elements}")
    print(f"  Counter story: {counter_story}")


# =============================================================================
# STRUCTURE PRESERVATION TESTS
# =============================================================================

def test_anti_attractor_preserves_structure():
    """Test that anti-attractor synthesis preserves structural properties."""
    story = "Fear without love causes anger"

    # Encode story to TKS expression (lenient mode for pipeline testing)
    expr_original = EncodeStory(story, strict=False)

    # Apply anti-attractor inversion
    result = AntiAttractorInvert(expr_original)
    expr_counter = result["expr_inverted"]

    # Assert counter expression has same number of elements
    # (anti-attractor should preserve structure while inverting content)
    assert len(expr_counter.elements) == len(expr_original.elements), \
        "Counter expression should have same number of elements as original"

    # Assert counter expression has operators (if original had operators)
    if len(expr_original.ops) > 0:
        assert len(expr_counter.ops) > 0, \
            "Counter expression should have operators if original had operators"

    # Decode counter expression
    counter_story = DecodeStory(expr_counter)
    assert len(counter_story) > 0, \
        "Counter-story should be non-empty"

    print(f"[PASS] Structure preservation test passed")
    print(f"  Original: {story}")
    print(f"  Original element count: {len(expr_original.elements)}")
    print(f"  Counter element count: {len(expr_counter.elements)}")
    print(f"  Counter story: {counter_story}")


def test_anti_attractor_element_format():
    """Test that anti-attractor produces valid element format."""
    story = "A woman feels joy"

    # Encode story to TKS expression (lenient mode for pipeline testing)
    expr_original = EncodeStory(story, strict=False)

    # Apply anti-attractor inversion
    result = AntiAttractorInvert(expr_original)
    expr_counter = result["expr_inverted"]

    # Assert all counter elements are valid format (e.g., "B5", "D3")
    for element in expr_counter.elements:
        assert len(element) >= 2, \
            f"Element {element} should have at least 2 characters"

        world = element[0]
        assert world in "ABCD", \
            f"Element {element} should start with valid world letter (A/B/C/D)"

        noetic = element[1:]
        assert noetic.isdigit(), \
            f"Element {element} should have numeric noetic after world letter"

        noetic_int = int(noetic)
        assert 1 <= noetic_int <= 10, \
            f"Element {element} should have noetic in range 1-10"

    # Decode counter expression
    counter_story = DecodeStory(expr_counter)
    assert len(counter_story) > 0, \
        "Counter-story should be non-empty"

    print(f"[PASS] Element format test passed")
    print(f"  Original: {story}")
    print(f"  Counter elements: {expr_counter.elements}")
    print(f"  Counter story: {counter_story}")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STORY-BASED ANTI-ATTRACTOR REGRESSION TESTS")
    print("=" * 80 + "\n")

    # Basic pipeline tests
    print("\n--- Basic Pipeline Tests ---\n")
    test_anti_attractor_joy_story()
    test_anti_attractor_belief_story()
    test_anti_attractor_fear_anger_story()

    # Signature tests
    print("\n--- Signature Tests ---\n")
    test_anti_attractor_with_signature()
    test_anti_attractor_without_signature()

    # Structure preservation tests
    print("\n--- Structure Preservation Tests ---\n")
    test_anti_attractor_preserves_structure()
    test_anti_attractor_element_format()

    # Parametrized tests (manual execution)
    print("\n--- Pipeline Stability Tests ---\n")
    test_stories = [
        "A woman feels joy",
        "Clear thinking leads to positive belief",
        "Fear without love causes anger",
        "Love heals pain",
        "Wisdom creates harmony",
        "Anger leads to conflict",
        "She loved him",
        "Fear causes illness",
    ]
    for story in test_stories:
        test_anti_attractor_pipeline_stability(story)

    print("\n" + "=" * 80)
    print("ALL STORY-BASED ANTI-ATTRACTOR TESTS PASSED")
    print("=" * 80 + "\n")

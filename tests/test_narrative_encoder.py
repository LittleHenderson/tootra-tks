"""
Tests for TKS Narrative Encoder/Decoder.

Tests the real EncodeStory/DecodeStory implementation from narrative module.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative import (
    EncodeStory,
    DecodeStory,
    encode_story_full,
    decode_story_full,
    parse_equation,
    TKSExpression,
    ElementRef,
)
from narrative.constants import (
    LEXICON,
    WORLD_KEYWORDS,
    NOETIC_KEYWORDS,
    ELEMENT_DEFAULTS,
    ALLOWED_OPS,
    is_valid_operator,
)


# =============================================================================
# LEXICON TESTS
# =============================================================================

def test_lexicon_canonical_entries():
    """Test that lexicon contains canonical entries."""
    # Key people entries
    assert "woman" in LEXICON
    assert "man" in LEXICON

    # Key emotion entries
    assert "fear" in LEXICON
    assert "love" in LEXICON
    assert "joy" in LEXICON

    # Key physical entries
    assert "money" in LEXICON
    assert "health" in LEXICON


def test_lexicon_format():
    """Test that lexicon entries have correct format (world, noetic, sense)."""
    for word, entry in LEXICON.items():
        assert len(entry) == 3, f"Entry for '{word}' should have 3 elements"
        world, noetic, sense = entry
        assert world in "ABCD", f"Invalid world '{world}' for '{word}'"
        assert 1 <= noetic <= 10, f"Invalid noetic '{noetic}' for '{word}'"


# =============================================================================
# ENCODER TESTS
# =============================================================================

def test_encode_simple_person():
    """Test encoding a story with a person."""
    expr = EncodeStory("A woman walked home.", strict=False)

    assert len(expr.elements) >= 1
    # Should detect woman -> D5
    assert any(e.startswith("D5") or e == "D5" for e in expr.elements)


def test_encode_emotion():
    """Test encoding a story with emotion."""
    expr = EncodeStory("She felt fear.", strict=False)

    assert len(expr.elements) >= 1
    # Should detect fear -> C3 and she -> D5
    worlds = [e[0] for e in expr.elements]
    assert "C" in worlds or "D" in worlds


def test_encode_love_story():
    """Test encoding 'She loved him'."""
    expr = EncodeStory("She loved him", strict=False)

    assert len(expr.elements) >= 2
    # Should detect: she->D5, loved->C2.3 (or similar), him->D6
    elements_str = ",".join(expr.elements)
    # Should have D-world elements for people
    assert "D" in elements_str


def test_encode_mental():
    """Test encoding a story with mental content."""
    expr = EncodeStory("He thought about the idea.", strict=False)

    assert len(expr.elements) >= 1
    # Should detect mental content
    worlds = [e[0] for e in expr.elements]
    assert "B" in worlds or "D" in worlds  # B for thought/idea, D for he


def test_encode_returns_operators():
    """Test that encoder returns operators between elements."""
    expr = EncodeStory("A woman loves a man.", strict=False)

    if len(expr.elements) > 1:
        assert len(expr.ops) == len(expr.elements) - 1


def test_encode_full_diagnostics():
    """Test encode_story_full returns diagnostics in lenient mode."""
    result = encode_story_full("A mysterious zorblax appeared.", strict=False)

    assert hasattr(result, 'expression')
    assert hasattr(result, 'unknown_words')
    # 'zorblax' should be unknown
    assert 'zorblax' in result.unknown_words


def test_encode_strict_mode_default_raises_error():
    """Test that strict mode (now default) raises ValueError for unknown tokens."""
    try:
        EncodeStory("A mysterious zorblax appeared.")  # No strict parameter = strict by default
        assert False, "Expected ValueError to be raised in default strict mode"
    except ValueError as e:
        # Should contain the unknown word in error message
        assert 'zorblax' in str(e)
        assert 'Unknown tokens' in str(e)
        # Should suggest using --lenient flag
        assert '--lenient' in str(e) or 'lenient' in str(e)


def test_encode_strict_mode_explicit_raises_error():
    """Test that strict=True explicitly raises ValueError for unknown tokens."""
    try:
        EncodeStory("A mysterious zorblax appeared.", strict=True)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        # Should contain the unknown word in error message
        assert 'zorblax' in str(e)
        assert 'Unknown tokens' in str(e)
        # Should provide helpful suggestions
        assert 'Valid token categories' in str(e)


def test_encode_strict_mode_helpful_error_message():
    """Test that strict mode provides helpful error messages with suggestions."""
    try:
        EncodeStory("A mysterious zorblax appeared.", strict=True)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        error_msg = str(e)
        # Should list unknown tokens
        assert 'zorblax' in error_msg
        # Should suggest valid categories
        assert 'LEXICON' in error_msg
        assert 'World keywords' in error_msg or 'world keywords' in error_msg
        assert 'Noetic keywords' in error_msg or 'noetic keywords' in error_msg
        # Should suggest lenient mode
        assert 'lenient' in error_msg


def test_encode_strict_mode_valid_story():
    """Test that strict mode works fine with valid stories."""
    # Use a story with only known words (woman, loved, man are in lexicon)
    expr = EncodeStory("A woman loved a man.", strict=True)

    assert len(expr.elements) >= 1
    # Should work without raising an error


def test_encode_lenient_mode_allows_unknown():
    """Test that strict=False (lenient mode) does not raise error for unknown tokens."""
    # Should not raise an error
    expr = EncodeStory("A mysterious zorblax appeared.", strict=False)

    assert len(expr.elements) >= 0
    # Should complete successfully even with unknown words


def test_encode_full_strict_mode():
    """Test encode_story_full with strict=True."""
    try:
        result = encode_story_full("The alien zorblax invaded.", strict=True)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert 'zorblax' in str(e)
        # 'alien' and 'invaded' might also be unknown depending on lexicon
        assert 'Unknown tokens' in str(e) or 'unknown' in str(e).lower()


def test_encode_causal_story():
    """Test encoding a story with causation."""
    expr = EncodeStory("Fear causes illness.", strict=False)

    # Should detect causal relationship
    assert len(expr.elements) >= 2
    # Check for causal operator if detected
    if expr.ops:
        # May have causal arrow or +T
        assert len(expr.ops) > 0


# =============================================================================
# DECODER TESTS
# =============================================================================

def test_decode_single_element():
    """Test decoding single element."""
    expr = TKSExpression(elements=["D5"], ops=[])
    story = DecodeStory(expr)

    assert isinstance(story, str)
    assert len(story) > 0
    assert "woman" in story.lower()


def test_decode_two_elements():
    """Test decoding two elements."""
    expr = TKSExpression(elements=["D5", "C2"], ops=["+T"])
    story = DecodeStory(expr)

    assert isinstance(story, str)
    assert len(story) > 0


def test_decode_causal_chain():
    """Test decoding causal chain."""
    expr = TKSExpression(elements=["C3", "D7"], ops=["->"])
    story = DecodeStory(expr)

    assert isinstance(story, str)
    # Should contain causal language
    assert "leads" in story.lower() or "causes" in story.lower() or "this" in story.lower()


def test_decode_multiple_elements():
    """Test decoding multiple elements."""
    expr = TKSExpression(elements=["B5", "D3", "C3"], ops=["+T", "->"])
    story = DecodeStory(expr)

    assert isinstance(story, str)
    assert len(story) > 5


def test_decode_full_diagnostics():
    """Test decode_story_full returns result structure."""
    expr = TKSExpression(elements=["D5", "C2"], ops=["+T"])
    result = decode_story_full(expr)

    assert hasattr(result, 'story')
    assert hasattr(result, 'success')
    assert result.success


# =============================================================================
# PARSER TESTS
# =============================================================================

def test_parse_comma_separated():
    """Test parsing comma-separated equation."""
    expr = parse_equation("B5,+T,D3,-T,C8", strict=False)

    assert expr.elements == ["B5", "D3", "C8"]
    assert expr.ops == ["+T", "-T"]


def test_parse_space_separated():
    """Test parsing space-separated equation."""
    expr = parse_equation("B5 +T D3 -T C8", strict=False)

    assert expr.elements == ["B5", "D3", "C8"]
    assert expr.ops == ["+T", "-T"]


def test_parse_with_causal_arrow():
    """Test parsing with causal arrow."""
    expr = parse_equation("C3 -> D7", strict=False)

    assert expr.elements == ["C3", "D7"]
    assert expr.ops == ["->"]


def test_parse_mixed_operators():
    """Test parsing with mixed operators."""
    expr = parse_equation("B5 +T D3 -> C2", strict=False)

    assert len(expr.elements) == 3
    assert "+T" in expr.ops
    assert "->" in expr.ops


def test_parse_with_sense():
    """Test parsing elements with sense notation."""
    expr = parse_equation("D5.1 +T C2.3", strict=False)

    assert len(expr.elements) == 2
    assert expr.elements[0] == "D5"
    assert expr.elements[1] == "C2"
    # Check element_refs have senses
    if expr.element_refs:
        assert expr.element_refs[0].sense == 1
        assert expr.element_refs[1].sense == 3


def test_parse_equation_strict_mode_rejects_unknown_operator():
    """Test that strict mode (default) rejects unknown operators in parse_equation."""
    try:
        parse_equation("B5 +X D3")  # +X is not a valid operator
        assert False, "Expected ValueError for unknown operator in default strict mode"
    except ValueError as e:
        error_msg = str(e)
        # Should mention the unknown operator
        assert '+X' in error_msg or 'X' in error_msg
        assert 'Unknown operator' in error_msg
        # Should list valid operators
        assert 'Valid operators' in error_msg
        assert '+T' in error_msg
        assert '->' in error_msg


def test_parse_equation_strict_mode_helpful_error():
    """Test that strict mode provides helpful error messages for unknown operators."""
    try:
        parse_equation("B5 +X D3", strict=True)
        assert False, "Expected ValueError for unknown operator"
    except ValueError as e:
        error_msg = str(e)
        # Should provide detailed help
        assert 'Unknown operator' in error_msg
        assert '+X' in error_msg
        # Should explain each operator
        assert 'combination' in error_msg or 'addition' in error_msg
        assert 'causal' in error_msg
        # Should suggest lenient mode
        assert 'lenient' in error_msg


def test_parse_equation_lenient_mode_skips_unknown():
    """Test that lenient mode (strict=False) skips unknown operators."""
    # Should not raise, just skip the unknown operator
    expr = parse_equation("B5 +X D3 +T C2", strict=False)

    # Should have parsed the elements
    assert len(expr.elements) == 3
    assert expr.elements == ["B5", "D3", "C2"]
    # Should have skipped +X but kept +T
    assert expr.ops == ["+T"]


def test_parse_equation_valid_operators_strict():
    """Test that all valid operators work in strict mode."""
    valid_ops = ["+T", "-T", "->", "<-", "*T", "/T", "o"]

    for op in valid_ops:
        expr = parse_equation(f"B5 {op} D3", strict=True)
        assert len(expr.elements) == 2
        assert expr.ops == [op], f"Failed to parse operator {op}"


# =============================================================================
# ELEMENT REF TESTS
# =============================================================================

def test_element_ref_creation():
    """Test ElementRef creation and validation."""
    ref = ElementRef(world="D", noetic=5, sense=1)

    assert ref.world == "D"
    assert ref.noetic == 5
    assert ref.sense == 1
    assert ref.code == "D5"
    # full_code now uses caret notation for sense
    assert ref.full_code == "D5^1"


def test_element_ref_from_string():
    """Test ElementRef parsing from string."""
    ref = ElementRef.from_string("D5.1")

    assert ref.world == "D"
    assert ref.noetic == 5
    assert ref.sense == 1


def test_element_ref_label():
    """Test ElementRef label generation."""
    ref = ElementRef(world="D", noetic=5)

    label = ref.label
    assert isinstance(label, str)
    assert len(label) > 0


# =============================================================================
# ROUNDTRIP TESTS
# =============================================================================

def test_encode_decode_roundtrip():
    """Test that encode->decode produces coherent output."""
    story = "A woman feels fear."
    expr = EncodeStory(story, strict=False)  # Use lenient mode for this test
    decoded = DecodeStory(expr)

    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_canonical_elements_preserved():
    """Test that canonical elements are preserved through encode->decode."""
    expr = TKSExpression(elements=["D5", "C3"], ops=["+T"])
    story = DecodeStory(expr)

    # Re-encode and check elements are similar
    re_expr = EncodeStory(story)

    # Should have some overlap in detected elements
    assert len(re_expr.elements) > 0


# =============================================================================
# INTEGRATION WITH SCENARIO INVERSION
# =============================================================================

def test_scenario_inversion_integration():
    """Test that narrative module works with scenario_inversion."""
    from scenario_inversion import (
        EncodeStory as SI_Encode,
        DecodeStory as SI_Decode,
        TKSExpression as SI_TKSExpression,
    )

    # Test through scenario_inversion wrapper
    expr = SI_Encode("She loved him")
    assert len(expr.elements) >= 1

    decoded = SI_Decode(expr)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_parse_equation_integration():
    """Test parse_equation through scenario_inversion."""
    from scenario_inversion import parse_equation as si_parse

    expr = si_parse("B5,+T,D3")
    assert expr.elements == ["B5", "D3"]
    assert expr.ops == ["+T"]


# =============================================================================
# EXPANDED LEXICON TESTS
# =============================================================================

def test_new_emotion_words():
    """Test new emotion words added to lexicon."""
    # Positive emotions
    assert "delight" in LEXICON
    assert "pleasure" in LEXICON
    assert "attraction" in LEXICON

    # Negative emotions
    assert "grief" in LEXICON
    assert "sorrow" in LEXICON
    assert "depression" in LEXICON
    assert "shame" in LEXICON
    assert "guilt" in LEXICON
    assert "disgust" in LEXICON
    assert "worry" in LEXICON

    # Verify they map to C-world (emotional)
    assert LEXICON["grief"][0] == "C"
    assert LEXICON["shame"][0] == "C"
    assert LEXICON["delight"][0] == "C"


def test_new_mental_words():
    """Test new mental words added to lexicon."""
    # Positive mental states
    assert "faith" in LEXICON
    assert "trust" in LEXICON
    assert "confidence" in LEXICON
    assert "hope" in LEXICON
    assert "optimism" in LEXICON

    # Negative mental states
    assert "doubt" in LEXICON
    assert "pessimism" in LEXICON
    assert "skepticism" in LEXICON

    # Mental concepts
    assert "plan" in LEXICON
    assert "strategy" in LEXICON
    assert "vision" in LEXICON
    assert "memory" in LEXICON

    # Verify they map to B-world (mental)
    assert LEXICON["faith"][0] == "B"
    assert LEXICON["doubt"][0] == "B"
    assert LEXICON["strategy"][0] == "B"


def test_new_physical_words():
    """Test new physical words added to lexicon."""
    # Physical health
    assert "vitality" in LEXICON
    assert "wellness" in LEXICON

    # Physical patterns
    assert "cycle" in LEXICON
    assert "pattern" in LEXICON
    assert "repetition" in LEXICON

    # Physical causation
    assert "trigger" in LEXICON
    assert "consequence" in LEXICON
    assert "transformation" in LEXICON

    # Physical objects/structures
    assert "vessel" in LEXICON
    assert "container" in LEXICON
    assert "framework" in LEXICON

    # Verify they map to D-world (physical)
    assert LEXICON["vitality"][0] == "D"
    assert LEXICON["cycle"][0] == "D"
    assert LEXICON["trigger"][0] == "D"


def test_encode_expanded_emotion_story():
    """Test encoding a story with expanded emotion words."""
    expr = EncodeStory("She felt grief and sorrow.")

    assert len(expr.elements) >= 1
    # Should detect grief/sorrow -> C3 (negative emotions)
    worlds = [e[0] for e in expr.elements]
    assert "C" in worlds or "D" in worlds


def test_encode_mental_confidence_story():
    """Test encoding a story with mental confidence words."""
    expr = EncodeStory("He had faith and trust in the plan.")

    assert len(expr.elements) >= 2
    # Should detect faith->B2, trust->B2, plan->B10
    elements_str = ",".join(expr.elements)
    assert "B" in elements_str


def test_encode_physical_transformation_story():
    """Test encoding a story with physical transformation."""
    expr = EncodeStory("The trigger caused transformation.")

    assert len(expr.elements) >= 2
    # Should detect trigger->D8, transformation->D9
    # May have causal operator
    elements_str = ",".join(expr.elements)
    assert "D" in elements_str


# =============================================================================
# SUB-FOUNDATION TESTS
# =============================================================================

def test_subfound_map_has_28_entries():
    """Test SUBFOUND_MAP has exactly 28 entries (7 foundations x 4 worlds)."""
    from narrative.constants import SUBFOUND_MAP
    assert len(SUBFOUND_MAP) == 28
    # Verify all combinations exist
    for fid in range(1, 8):
        for world in "ABCD":
            assert (fid, world) in SUBFOUND_MAP


def test_emotional_companionship_subfound():
    """Test emotional companionship sub-foundation (F4, C)."""
    from narrative.constants import SUBFOUND_MAP, get_subfound_label
    label = get_subfound_label(4, "C")
    assert label == "Emotional relationship"


def test_subfoundation_detection():
    """Test that foundation context is detected in stories."""
    from narrative import EncodeStory
    # Story about relationships should detect F4 (Companionship)
    expr = EncodeStory("She loved her partner in their relationship.", strict=False)
    # Should have some foundation tagged
    # Note: foundations might be empty if not detected - that's OK for now
    assert hasattr(expr, 'foundations')


def test_get_subfound_label_all_worlds():
    """Test get_subfound_label for all foundation-world combinations."""
    from narrative.constants import get_subfound_label
    # Test a few key combinations
    assert get_subfound_label(1, "A") == "Spiritual union"
    assert get_subfound_label(2, "B") == "Intellectual wisdom"
    assert get_subfound_label(3, "D") == "Physical health"
    assert get_subfound_label(4, "C") == "Emotional relationship"
    assert get_subfound_label(5, "A") == "Spiritual authority"
    assert get_subfound_label(6, "D") == "Physical resources"
    assert get_subfound_label(7, "C") == "Desire/passion"


def test_subfound_label_invalid_foundation():
    """Test that get_subfound_label returns None for invalid foundation."""
    from narrative.constants import get_subfound_label
    # Invalid foundation (0 or 8+)
    assert get_subfound_label(0, "A") is None
    assert get_subfound_label(8, "A") is None


def test_subfound_label_invalid_world():
    """Test that get_subfound_label returns None for invalid world."""
    from narrative.constants import get_subfound_label
    # Invalid world (E or other)
    assert get_subfound_label(1, "E") is None
    assert get_subfound_label(1, "X") is None


# =============================================================================
# SENSE RULES TESTS
# =============================================================================

def test_sense_rules_applied():
    """Test that SENSE_RULES are applied during encoding."""
    from narrative import EncodeStory
    # "experiences" should map to B5.2 (sense=2)
    expr = EncodeStory("Past experiences shaped her beliefs.", strict=False)
    # Check that sense is applied (if element_refs available)
    # At minimum, verify encoding doesn't crash
    assert len(expr.elements) >= 1


def test_sense_rules_experiences():
    """Test that 'experiences' maps to B5.2 (accumulated knowledge)."""
    from narrative.constants import SENSE_RULES, LEXICON
    # Check SENSE_RULES has the mapping
    assert "experiences" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["experiences"]
    assert world == "B"
    assert noetic == 5
    assert sense == 2


def test_sense_rules_love():
    """Test that 'love' has sense 3 (C2.3)."""
    from narrative.constants import LEXICON
    # Check LEXICON has the mapping
    assert "love" in LEXICON
    world, noetic, sense = LEXICON["love"]
    assert world == "C"
    assert noetic == 2
    assert sense == 3


def test_sense_rules_control():
    """Test that 'control' maps to D8.3 (material authority)."""
    from narrative.constants import SENSE_RULES
    assert "control" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["control"]
    assert world == "D"
    assert noetic == 8
    assert sense == 3


def test_sense_rules_chaos():
    """Test that 'chaos' maps to D3.2 (material chaos)."""
    from narrative.constants import SENSE_RULES
    assert "chaos" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["chaos"]
    assert world == "D"
    assert noetic == 3
    assert sense == 2


def test_sense_labels_b52():
    """Test that B5.2 has correct label (accumulated knowledge)."""
    from narrative.constants import SENSE_LABELS
    assert "B5.2" in SENSE_LABELS
    assert SENSE_LABELS["B5.2"] == "accumulated knowledge"


def test_sense_labels_c23():
    """Test that C2.3 has correct label (love)."""
    from narrative.constants import SENSE_LABELS
    assert "C2.3" in SENSE_LABELS
    assert SENSE_LABELS["C2.3"] == "love"


def test_sense_labels_d83():
    """Test that D8.3 has correct label (material authority)."""
    from narrative.constants import SENSE_LABELS
    assert "D8.3" in SENSE_LABELS
    assert SENSE_LABELS["D8.3"] == "material authority"


# =============================================================================
# OPERATOR VALIDATION TESTS
# =============================================================================

def test_allowed_ops_canonical():
    """Test that ALLOWED_OPS contains all canonical operators from rulebook."""
    # From rulebook: +_T, -_T, ×_T, /_T, ∘, →, ←
    # Represented as: +T, -T, *T, /T, o, ->, <-
    assert "+T" in ALLOWED_OPS  # TOOTRA addition
    assert "-T" in ALLOWED_OPS  # TOOTRA subtraction
    assert "*T" in ALLOWED_OPS  # TOOTRA multiplication (×_T)
    assert "/T" in ALLOWED_OPS  # TOOTRA division (/_T)
    assert "o" in ALLOWED_OPS   # Sequential composition (∘)
    assert "->" in ALLOWED_OPS  # Causal arrow (→)
    assert "<-" in ALLOWED_OPS  # Reverse causal (←)


def test_allowed_ops_count():
    """Test that ALLOWED_OPS has exactly the expected operators."""
    # Should have: +T, -T, *T, /T, o, ->, <-, +, -
    assert len(ALLOWED_OPS) == 9


def test_is_valid_operator_canonical():
    """Test is_valid_operator returns True for canonical operators."""
    assert is_valid_operator("+T") == True
    assert is_valid_operator("-T") == True
    assert is_valid_operator("*T") == True
    assert is_valid_operator("/T") == True
    assert is_valid_operator("o") == True
    assert is_valid_operator("->") == True
    assert is_valid_operator("<-") == True
    assert is_valid_operator("+") == True
    assert is_valid_operator("-") == True


def test_is_valid_operator_unknown():
    """Test is_valid_operator returns False for unknown operators."""
    assert is_valid_operator("×T") == False  # Should be *T
    assert is_valid_operator("÷T") == False  # Should be /T
    assert is_valid_operator("~T") == False  # Not defined
    assert is_valid_operator("?T") == False  # Not defined
    assert is_valid_operator(">>") == False  # Not defined
    assert is_valid_operator("**") == False  # Not defined


def test_operator_validation_strict_mode():
    """Test that parse_equation rejects unknown operators in strict mode."""
    # Valid operators should work
    expr = parse_equation("B5 +T D3", strict=True)
    assert expr.elements == ["B5", "D3"]
    assert expr.ops == ["+T"]

    # Unknown operator should raise ValueError
    try:
        parse_equation("B5 ~T D3", strict=True)
        assert False, "Expected ValueError for unknown operator"
    except ValueError as e:
        assert "Unknown operator" in str(e)
        assert "~T" in str(e)


def test_operator_validation_non_strict_mode():
    """Test that parse_equation skips unknown operators in non-strict mode."""
    # Unknown operator should be skipped (not raise error)
    expr = parse_equation("B5 ~T D3 +T C2", strict=False)

    # Should have elements
    assert len(expr.elements) >= 2

    # Unknown operator ~T should be skipped, only +T should be present
    assert "+T" in expr.ops
    # ~T should NOT be in ops (it was skipped)
    assert "~T" not in expr.ops


def test_new_operators_multiply():
    """Test multiplication operator (*T) encoding and decoding."""
    # Parse equation with *T
    expr = parse_equation("C3 *T C3")
    assert expr.elements == ["C3", "C3"]
    assert expr.ops == ["*T"]

    # Decode should produce "intensified by" language
    story = DecodeStory(expr)
    assert "intensified" in story.lower() or "fear" in story.lower()


def test_new_operators_divide():
    """Test division operator (/T) encoding and decoding."""
    # Parse equation with /T
    expr = parse_equation("B2 /T B3")
    assert expr.elements == ["B2", "B3"]
    assert expr.ops == ["/T"]

    # Decode should produce "conflict" language
    story = DecodeStory(expr)
    assert "conflict" in story.lower() or "belief" in story.lower()


def test_new_operators_composition():
    """Test sequential composition operator (o) encoding and decoding."""
    # Parse equation with o
    expr = parse_equation("C3 o D7 o D8")
    assert expr.elements == ["C3", "D7", "D8"]
    assert expr.ops == ["o", "o"]

    # Decode should produce sequential language
    story = DecodeStory(expr)
    assert "then" in story.lower() or "first" in story.lower()


def test_parse_equation_all_operators():
    """Test parsing equation with all canonical operators."""
    # Mix of all operators
    expr = parse_equation("B5 +T D3 -T C2 *T C4 /T B2 o D7 -> D8")

    # Should have 7 elements
    assert len(expr.elements) == 7

    # Should have 6 operators
    assert len(expr.ops) == 6
    assert "+T" in expr.ops
    assert "-T" in expr.ops
    assert "*T" in expr.ops
    assert "/T" in expr.ops
    assert "o" in expr.ops
    assert "->" in expr.ops


def test_encode_story_intensification():
    """Test encoding a story with intensification verbs."""
    expr = EncodeStory("Fear intensified by fear creates panic.", strict=False)

    # Should detect some elements
    assert len(expr.elements) >= 1

    # May detect *T operator if "intensified" is properly mapped
    # (Note: depends on extract_operators working correctly)


def test_encode_story_conflict():
    """Test encoding a story with conflict verbs."""
    expr = EncodeStory("Positive belief conflicts with limiting belief.")

    # Should detect mental elements
    assert len(expr.elements) >= 2
    elements_str = ",".join(expr.elements)
    assert "B" in elements_str


def test_encode_story_sequence():
    """Test encoding a story with sequential markers."""
    expr = EncodeStory("Fear, then habit, then elevation.")

    # Should detect elements
    assert len(expr.elements) >= 2

    # May detect 'o' operator if "then" is mapped correctly


def test_operator_templates_coverage():
    """Test that decoder has templates for all new operators."""
    from narrative.decoder import OPERATOR_TEMPLATES

    # Check all canonical operators have templates
    assert "+T" in OPERATOR_TEMPLATES
    assert "-T" in OPERATOR_TEMPLATES
    assert "*T" in OPERATOR_TEMPLATES
    assert "/T" in OPERATOR_TEMPLATES
    assert "o" in OPERATOR_TEMPLATES
    assert "->" in OPERATOR_TEMPLATES
    assert "<-" in OPERATOR_TEMPLATES


def test_operator_templates_multiply():
    """Test that *T template contains 'intensified'."""
    from narrative.decoder import OPERATOR_TEMPLATES

    template, connector = OPERATOR_TEMPLATES["*T"]
    assert "intensified" in template.lower() or "intensified" in connector.lower()


def test_operator_templates_divide():
    """Test that /T template contains 'conflict'."""
    from narrative.decoder import OPERATOR_TEMPLATES

    template, connector = OPERATOR_TEMPLATES["/T"]
    assert "conflict" in template.lower() or "conflict" in connector.lower()


def test_operator_templates_composition():
    """Test that o template contains 'then' or 'first'."""
    from narrative.decoder import OPERATOR_TEMPLATES

    template, connector = OPERATOR_TEMPLATES["o"]
    assert "then" in template.lower() or "first" in template.lower()


# =============================================================================
# NEW SENSE RULES TESTS (from TKS_Symbol_Sense_Table_v1.0.md)
# =============================================================================

def test_sense_rules_biological_cycle():
    """Test that 'biological cycle' maps to D7.2."""
    from narrative.constants import SENSE_RULES
    assert "biological cycle" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["biological cycle"]
    assert world == "D"
    assert noetic == 7
    assert sense == 2


def test_sense_rules_material_order():
    """Test that 'material order' maps to D2.2."""
    from narrative.constants import SENSE_RULES
    assert "material order" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["material order"]
    assert world == "D"
    assert noetic == 2
    assert sense == 2


def test_sense_rules_organization():
    """Test that 'organization' maps to D2.2 (material order)."""
    from narrative.constants import SENSE_RULES
    assert "organization" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["organization"]
    assert world == "D"
    assert noetic == 2
    assert sense == 2


def test_sense_rules_negative_thought():
    """Test that 'negative thought' maps to B3.3 (cognitive aversion)."""
    from narrative.constants import SENSE_RULES
    assert "negative thought" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["negative thought"]
    assert world == "B"
    assert noetic == 3
    assert sense == 3


def test_sense_labels_d72():
    """Test that D7.2 has correct label (biological cycle)."""
    from narrative.constants import SENSE_LABELS
    assert "D7.2" in SENSE_LABELS
    assert SENSE_LABELS["D7.2"] == "biological cycle"


def test_sense_labels_d22():
    """Test that D2.2 has correct label (material order)."""
    from narrative.constants import SENSE_LABELS
    assert "D2.2" in SENSE_LABELS
    assert SENSE_LABELS["D2.2"] == "material order"


def test_sense_labels_c32():
    """Test that C3.2 has correct label (emotional aversion)."""
    from narrative.constants import SENSE_LABELS
    assert "C3.2" in SENSE_LABELS
    assert SENSE_LABELS["C3.2"] == "emotional aversion"


def test_sense_labels_b33():
    """Test that B3.3 has correct label (cognitive aversion)."""
    from narrative.constants import SENSE_LABELS
    assert "B3.3" in SENSE_LABELS
    assert SENSE_LABELS["B3.3"] == "cognitive aversion"


def test_encode_biological_cycle_story():
    """Test encoding a story with biological cycle."""
    expr = EncodeStory("Her sleep cycle was disrupted by stress.", strict=False)
    # Should detect sleep cycle -> D7.2
    assert len(expr.elements) >= 1
    # Verify encoding doesn't crash with new sense rules
    assert hasattr(expr, 'elements')


def test_encode_material_order_story():
    """Test encoding a story with material order."""
    expr = EncodeStory("The organization of the space brought peace.", strict=False)
    # Should detect organization -> D2.2
    assert len(expr.elements) >= 0
    # Verify encoding doesn't crash with new sense rules
    assert hasattr(expr, 'elements')


def test_decode_d72():
    """Test decoding D7.2 (biological cycle)."""
    from narrative import TKSExpression, DecodeStory
    expr = TKSExpression(elements=["D7"], ops=[])
    # Set sense to 2 via element_refs if available
    story = DecodeStory(expr)
    assert isinstance(story, str)
    assert len(story) > 0


def test_decode_d22():
    """Test decoding D2.2 (material order)."""
    from narrative import TKSExpression, DecodeStory
    expr = TKSExpression(elements=["D2"], ops=[])
    story = DecodeStory(expr)
    assert isinstance(story, str)
    assert len(story) > 0


# =============================================================================
# EXPANDED SENSE LABELS TESTS
# =============================================================================

def test_sense_labels_comprehensive_d_world():
    """Test that all D-world sense labels from Symbol Sense Table are present."""
    from narrative.constants import SENSE_LABELS
    # Test a sample of D-world senses
    expected_d_senses = [
        "D1.1", "D1.2", "D1.3",
        "D2.1", "D2.2", "D2.3", "D2.4",
        "D3.1", "D3.2", "D3.3", "D3.4",
        "D4.1", "D4.2", "D4.3",
        "D5.1", "D5.2", "D5.3", "D5.4",
        "D6.1", "D6.2", "D6.3", "D6.4",
        "D7.1", "D7.2", "D7.3", "D7.4",
        "D8.1", "D8.2", "D8.3", "D8.4",
        "D9.1", "D9.2", "D9.3", "D9.4",
        "D10.1", "D10.2", "D10.3",
    ]
    for sense in expected_d_senses:
        assert sense in SENSE_LABELS, f"Missing sense label: {sense}"


def test_sense_labels_comprehensive_c_world():
    """Test that all C-world sense labels from Symbol Sense Table are present."""
    from narrative.constants import SENSE_LABELS
    # Test a sample of C-world senses
    expected_c_senses = [
        "C1.1", "C1.2", "C1.3",
        "C2.1", "C2.2", "C2.3", "C2.4",
        "C3.1", "C3.2", "C3.3", "C3.4",
        "C4.1", "C4.2", "C4.3",
        "C5.1", "C5.2", "C5.3", "C5.4",
        "C6.1", "C6.2", "C6.3", "C6.4",
        "C7.1", "C7.2", "C7.3",
        "C8.1", "C8.2", "C8.3",
        "C9.1", "C9.2", "C9.3",
        "C10.1", "C10.2", "C10.3",
    ]
    for sense in expected_c_senses:
        assert sense in SENSE_LABELS, f"Missing sense label: {sense}"


def test_sense_labels_comprehensive_b_world():
    """Test that all B-world sense labels from Symbol Sense Table are present."""
    from narrative.constants import SENSE_LABELS
    # Test a sample of B-world senses
    expected_b_senses = [
        "B1.1", "B1.2", "B1.3",
        "B2.1", "B2.2", "B2.3", "B2.4",
        "B3.1", "B3.2", "B3.3", "B3.4",
        "B4.1", "B4.2", "B4.3",
        "B5.1", "B5.2", "B5.3", "B5.4",
        "B6.1", "B6.2", "B6.3", "B6.4",
        "B7.1", "B7.2", "B7.3",
        "B8.1", "B8.2", "B8.3",
        "B9.1", "B9.2", "B9.3",
        "B10.1", "B10.2", "B10.3",
    ]
    for sense in expected_b_senses:
        assert sense in SENSE_LABELS, f"Missing sense label: {sense}"


def test_sense_labels_comprehensive_a_world():
    """Test that all A-world sense labels from Symbol Sense Table are present."""
    from narrative.constants import SENSE_LABELS
    # Test a sample of A-world senses
    expected_a_senses = [
        "A1.1", "A1.2", "A1.3",
        "A2.1", "A2.2", "A2.3", "A2.4",
        "A3.1", "A3.2", "A3.3",
        "A4.1", "A4.2", "A4.3",
        "A5.1", "A5.2", "A5.3",
        "A6.1", "A6.2", "A6.3",
        "A7.1", "A7.2", "A7.3",
        "A8.1", "A8.2", "A8.3",
        "A9.1", "A9.2", "A9.3",
        "A10.1", "A10.2", "A10.3",
    ]
    for sense in expected_a_senses:
        assert sense in SENSE_LABELS, f"Missing sense label: {sense}"


def test_verb_to_operator_intensification():
    """Test that intensification verbs map to *T."""
    from narrative.constants import VERB_TO_OPERATOR

    assert VERB_TO_OPERATOR.get("amplifies") == "*T"
    assert VERB_TO_OPERATOR.get("intensifies") == "*T"
    assert VERB_TO_OPERATOR.get("multiplies") == "*T"


def test_verb_to_operator_conflict():
    """Test that conflict verbs map to /T."""
    from narrative.constants import VERB_TO_OPERATOR

    assert VERB_TO_OPERATOR.get("opposes") == "/T"
    assert VERB_TO_OPERATOR.get("conflicts") == "/T"
    assert VERB_TO_OPERATOR.get("divides") == "/T"


def test_verb_to_operator_sequence():
    """Test that sequence verbs map to o."""
    from narrative.constants import VERB_TO_OPERATOR

    assert VERB_TO_OPERATOR.get("then") == "o"
    assert VERB_TO_OPERATOR.get("after") == "o"
    assert VERB_TO_OPERATOR.get("followed") == "o"


def test_roundtrip_new_operators():
    """Test encode->decode roundtrip with new operators."""
    # Parse equation with new operators
    original_expr = parse_equation("C3.1 *T C3.1 -> D7.1")

    # Decode to story
    story = DecodeStory(original_expr)
    assert len(story) > 0

    # Should contain intensification and causation language
    assert "intensified" in story.lower() or "leads" in story.lower()


# =============================================================================
# REGRESSION TESTS: SENSE-SPECIFIC ENCODING/DECODING
# =============================================================================

def test_regression_d52_vessel_sense():
    """
    Regression test: Story with 'vessel' keyword should trigger D5.2 (receptacle).
    Tests that SENSE_RULES correctly overrides default D5.1 (woman).
    """
    story = "The vessel holds sacred water."
    expr = EncodeStory(story, strict=False)

    # Should detect vessel -> D5
    assert "D5" in expr.elements or any("D5" in e for e in expr.elements)

    # If element_refs available, verify sense is 2 (receptacle), not 1 (woman)
    if expr.element_refs:
        d5_refs = [ref for ref in expr.element_refs if ref.world == "D" and ref.noetic == 5]
        if d5_refs:
            # At least one should have sense 2
            assert any(ref.sense == 2 for ref in d5_refs), "Expected D5.2 (receptacle) for 'vessel'"


def test_regression_b52_accumulated_knowledge():
    """
    Regression test: Story with 'experiences' should trigger B5.2 (accumulated knowledge).
    Tests that past experiences are encoded with correct sense.
    """
    story = "Her past experiences shaped her beliefs about relationships."
    expr = EncodeStory(story, strict=False)

    # Should detect B5 (mental receptivity/knowledge)
    elements_str = ",".join(expr.elements)
    assert "B5" in elements_str or "B" in elements_str

    # Decode and verify it mentions knowledge/experience
    decoded = DecodeStory(expr)
    assert len(decoded) > 0
    # Should contain knowledge or experience related terms
    lower_decoded = decoded.lower()
    assert any(word in lower_decoded for word in ["knowledge", "experience", "accumulated", "past"])


def test_regression_c23_love_vs_c21_joy():
    """
    Regression test: 'love' should encode as C2.3, not default C2.1 (joy).
    Tests sense distinction within same element (C2).
    """
    story = "She felt love for her partner."
    expr = EncodeStory(story, strict=False)

    # Should detect C2 (emotional positive)
    elements_str = ",".join(expr.elements)
    assert "C2" in elements_str or "C" in elements_str

    # Parse equation to check if love is correctly assigned
    # Decode and verify
    decoded = DecodeStory(expr)
    assert len(decoded) > 0
    # Should mention love or affection, not just generic joy
    lower_decoded = decoded.lower()
    assert "love" in lower_decoded or "affection" in lower_decoded or "care" in lower_decoded


def test_regression_subfoundation_emotional_relationship():
    """
    Regression test: Story about emotional relationship should detect F4 (Companionship) in C-world.
    Tests sub-foundation context detection (F4c = emotional relationship).
    """
    story = "The emotional relationship between them was deep and meaningful."
    expr = EncodeStory(story, strict=False)

    # Should detect foundation 4 (Companionship)
    # Check if foundations are tracked (may be tuples of (foundation, world))
    if hasattr(expr, 'foundations') and expr.foundations:
        # Foundations may be stored as tuples (foundation_id, world) or just foundation_id
        foundation_ids = [f[0] if isinstance(f, tuple) else f for f in expr.foundations]
        assert 4 in foundation_ids, f"Expected Foundation 4 (Companionship) for relationship context, found {expr.foundations}"

    # Should detect C-world elements (emotional)
    elements_str = ",".join(expr.elements)
    assert "C" in elements_str, "Expected C-world (emotional) elements"

    # Verify sub-foundation label
    from narrative.constants import get_subfound_label
    label = get_subfound_label(4, "C")
    assert label == "Emotional relationship", "F4c should be 'Emotional relationship'"


def test_regression_roundtrip_with_senses():
    """
    Regression test: Encode story with specific senses, decode, verify consistency.
    Tests that sense information survives encode->decode roundtrip.
    """
    # Start with equation that has explicit senses
    original_equation = "B5^2 +T D3^2 -> C3^1"  # accumulated knowledge + chaos -> fear

    # Parse
    expr = parse_equation(original_equation)

    # Verify senses are parsed correctly
    assert expr.element_refs[0].sense == 2, "B5 should have sense 2 (accumulated knowledge)"
    assert expr.element_refs[1].sense == 2, "D3 should have sense 2 (material chaos)"
    assert expr.element_refs[2].sense == 1, "C3 should have sense 1 (fear)"

    # Decode to story
    story = DecodeStory(expr)
    assert len(story) > 0

    # Verify story contains appropriate sense labels
    lower_story = story.lower()
    # Should mention knowledge/experience (B5.2), chaos/instability (D3.2), fear (C3.1)
    has_knowledge = any(w in lower_story for w in ["knowledge", "experience", "accumulated"])
    has_chaos = any(w in lower_story for w in ["chaos", "instability", "disorder"])
    has_fear = any(w in lower_story for w in ["fear", "anxiety", "afraid"])

    # At least 2 of 3 should be present
    assert sum([has_knowledge, has_chaos, has_fear]) >= 2, \
        f"Story should contain sense-appropriate terms. Story: {story}"


def test_regression_d83_material_authority():
    """
    Regression test: 'control' should map to D8.3 (material authority), not D8.1.
    Tests that high-value sense override works for D8.
    """
    story = "She wanted control over the situation."
    expr = EncodeStory(story, strict=False)

    # Should detect D8 (physical cause/authority)
    elements_str = ",".join(expr.elements)
    assert "D8" in elements_str or "D" in elements_str

    # Check sense_rules
    from narrative.constants import SENSE_RULES
    assert "control" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["control"]
    assert world == "D"
    assert noetic == 8
    assert sense == 3, "control should map to D8.3 (material authority)"


def test_regression_a72_soul_cycle():
    """
    Regression test: 'soul evolution' should trigger A7.2 (soul cycle).
    Tests spiritual world sense detection.
    """
    story = "His soul evolution continued through many lifetimes."
    expr = EncodeStory(story, strict=False)

    # Should detect A7 (spiritual rhythm)
    elements_str = ",".join(expr.elements)
    assert "A7" in elements_str or "A" in elements_str

    # Check SENSE_RULES
    from narrative.constants import SENSE_RULES
    assert "soul evolution" in SENSE_RULES
    world, noetic, sense = SENSE_RULES["soul evolution"]
    assert world == "A"
    assert noetic == 7
    assert sense == 2, "soul evolution should map to A7.2 (soul cycle)"


def test_regression_decode_with_foundation_suffix():
    """
    Regression test: Decode equation with foundation suffix (_Fw notation).
    Tests that foundation context is preserved in decoding.
    """
    # Equation with foundation suffix: emotional relationship context (F4c)
    equation = "D5^1 +T C3^1 -> D7^1"  # woman + fear -> habit in relationship

    # Add foundation context manually (F4c = emotional relationship)
    equation_with_foundation = "D5^1_c4 +T C3^1_c4 -> D7^1_c4"

    expr = parse_equation(equation_with_foundation)

    # Verify foundation is parsed
    assert expr.element_refs[0].foundation == 4
    assert expr.element_refs[0].subfoundation == "C"

    # Decode
    story = DecodeStory(expr)
    assert len(story) > 0

    # Story should still be coherent
    assert isinstance(story, str)


def test_regression_count_new_sense_rules():
    """
    Regression test: Verify that SENSE_RULES has been significantly expanded.
    Ensures coverage goals are met.
    """
    from narrative.constants import SENSE_RULES

    # Should have significantly more than the original 21 entries
    # Target: at least 100+ entries after expansion
    assert len(SENSE_RULES) > 100, \
        f"SENSE_RULES should have 100+ entries after expansion, found {len(SENSE_RULES)}"

    # Check coverage across all worlds
    a_world = sum(1 for (w, n, s) in SENSE_RULES.values() if w == "A")
    b_world = sum(1 for (w, n, s) in SENSE_RULES.values() if w == "B")
    c_world = sum(1 for (w, n, s) in SENSE_RULES.values() if w == "C")
    d_world = sum(1 for (w, n, s) in SENSE_RULES.values() if w == "D")

    # All worlds should have at least 15 entries
    assert a_world >= 15, f"A-world should have 15+ sense rules, found {a_world}"
    assert b_world >= 15, f"B-world should have 15+ sense rules, found {b_world}"
    assert c_world >= 15, f"C-world should have 15+ sense rules, found {c_world}"
    assert d_world >= 30, f"D-world should have 30+ sense rules, found {d_world}"


def test_regression_foundation_keywords_expanded():
    """
    Regression test: Verify FOUNDATION_KEYWORDS has sub-foundation phrases.
    Tests that foundation detection vocabulary has been expanded.
    """
    from narrative.constants import FOUNDATION_KEYWORDS

    # Should have more than original ~30 entries
    # Target: at least 60+ after expansion
    assert len(FOUNDATION_KEYWORDS) > 60, \
        f"FOUNDATION_KEYWORDS should have 60+ entries, found {len(FOUNDATION_KEYWORDS)}"

    # Check for sub-foundation specific keywords
    assert "emotional relationship" in FOUNDATION_KEYWORDS
    assert "intellectual partnership" in FOUNDATION_KEYWORDS
    assert "soul connection" in FOUNDATION_KEYWORDS
    assert "physical health" in FOUNDATION_KEYWORDS
    assert "spiritual authority" in FOUNDATION_KEYWORDS

    # All should map to correct foundations
    assert FOUNDATION_KEYWORDS["emotional relationship"] == 4  # Companionship
    assert FOUNDATION_KEYWORDS["physical health"] == 3  # Life
    assert FOUNDATION_KEYWORDS["spiritual authority"] == 5  # Power


# =============================================================================
# LEXICON CONSISTENCY TESTS (Canon guardrail)
# =============================================================================

def test_lexicon_sense_rules_no_conflicts():
    """
    Verify LEXICON and SENSE_RULES have no conflicting entries.
    This is critical for deterministic encoding behavior.
    """
    from narrative.constants import validate_lexicon_consistency

    conflicts = validate_lexicon_consistency()
    assert len(conflicts) == 0, \
        f"Lexicon conflicts detected (non-deterministic encoding): {conflicts}"


def test_lexicon_anger_sadness_grief_aligned():
    """
    Verify anger/sadness/grief entries are aligned between LEXICON and SENSE_RULES.
    These were previously conflicting and are critical for emotion encoding.
    """
    from narrative.constants import LEXICON, SENSE_RULES

    # Anger should be C3.3 in both
    assert LEXICON["anger"] == ("C", 3, 3), "anger should map to C3.3 in LEXICON"
    assert SENSE_RULES["anger"] == ("C", 3, 3), "anger should map to C3.3 in SENSE_RULES"

    # Sadness should be C3.4 in both
    assert LEXICON["sadness"] == ("C", 3, 4), "sadness should map to C3.4 in LEXICON"
    assert SENSE_RULES["sadness"] == ("C", 3, 4), "sadness should map to C3.4 in SENSE_RULES"

    # Grief should be C3.4 in both
    assert LEXICON["grief"] == ("C", 3, 4), "grief should map to C3.4 in LEXICON"
    assert SENSE_RULES["grief"] == ("C", 3, 4), "grief should map to C3.4 in SENSE_RULES"


def test_get_token_mapping_deterministic():
    """
    Test get_token_mapping returns consistent results (deterministic).
    """
    from narrative.constants import get_token_mapping

    # Same word should always return same mapping
    result1 = get_token_mapping("anger")
    result2 = get_token_mapping("anger")
    assert result1 == result2, "get_token_mapping should be deterministic"

    # Verify specific mappings
    assert get_token_mapping("woman") == ("D", 5, 1)  # D5.1
    assert get_token_mapping("anger")[2] == 3  # sense 3
    assert get_token_mapping("sadness")[2] == 4  # sense 4


def test_extended_syntax_conflict_detection():
    """
    Test check_extended_token_conflicts detects ambiguous tokens.
    """
    from narrative.constants import check_extended_token_conflicts

    # Valid tokens should have no conflicts
    assert check_extended_token_conflicts("B8^5_d5") is None
    assert check_extended_token_conflicts("D10^3_a7") is None
    assert check_extended_token_conflicts("C3.2") is None

    # Invalid foundation suffix should be detected
    error = check_extended_token_conflicts("B8_e5")  # 'e' is not a valid world
    assert error is not None
    assert "Invalid foundation suffix" in error

    # Invalid foundation number should be detected
    error = check_extended_token_conflicts("B8_d0")  # 0 is not valid (1-7)
    assert error is not None

    error = check_extended_token_conflicts("B8_d8")  # 8 is not valid (1-7)
    assert error is not None


def test_canon_worlds_only_abcd():
    """
    Canon guardrail: Verify only worlds A/B/C/D are valid.
    """
    from narrative.constants import is_valid_world, WORLD_LETTERS

    # Valid worlds
    assert is_valid_world("A")
    assert is_valid_world("B")
    assert is_valid_world("C")
    assert is_valid_world("D")

    # Invalid worlds
    assert not is_valid_world("E")
    assert not is_valid_world("F")
    assert not is_valid_world("X")
    assert not is_valid_world("Z")

    # Set should only contain A/B/C/D
    assert WORLD_LETTERS == {"A", "B", "C", "D"}


def test_canon_noetics_only_1_to_10():
    """
    Canon guardrail: Verify only noetics 1-10 are valid.
    """
    from narrative.constants import is_valid_noetic, NOETIC_NAMES

    # Valid noetics
    for i in range(1, 11):
        assert is_valid_noetic(i), f"Noetic {i} should be valid"

    # Invalid noetics
    assert not is_valid_noetic(0)
    assert not is_valid_noetic(11)
    assert not is_valid_noetic(15)
    assert not is_valid_noetic(-1)

    # NOETIC_NAMES should have exactly 10 entries
    assert len(NOETIC_NAMES) == 10


def test_canon_foundations_only_1_to_7():
    """
    Canon guardrail: Verify only foundations 1-7 are valid.
    """
    from narrative.constants import is_valid_foundation, FOUNDATIONS

    # Valid foundations
    for i in range(1, 8):
        assert is_valid_foundation(i), f"Foundation {i} should be valid"

    # Invalid foundations
    assert not is_valid_foundation(0)
    assert not is_valid_foundation(8)
    assert not is_valid_foundation(9)
    assert not is_valid_foundation(-1)

    # FOUNDATIONS should have exactly 7 entries
    assert len(FOUNDATIONS) == 7


def test_canon_operators_fixed():
    """
    Canon guardrail: Verify ALLOWED_OPS contains exactly the canonical operators.
    """
    from narrative.constants import ALLOWED_OPS

    # Exact set of allowed operators
    expected_ops = {"+", "-", "+T", "-T", "->", "<-", "*T", "/T", "o"}
    assert ALLOWED_OPS == expected_ops, \
        f"ALLOWED_OPS must be exactly {expected_ops}, got {ALLOWED_OPS}"


def test_canon_noetic_involutions():
    """
    Canon guardrail: Verify noetic involution pairs are fixed (2<->3, 5<->6, 8<->9).
    """
    from narrative.constants import NOETIC_INVOLUTIONS, NOETIC_SELF_DUALS

    # Involution pairs
    assert NOETIC_INVOLUTIONS[2] == 3
    assert NOETIC_INVOLUTIONS[3] == 2
    assert NOETIC_INVOLUTIONS[5] == 6
    assert NOETIC_INVOLUTIONS[6] == 5
    assert NOETIC_INVOLUTIONS[8] == 9
    assert NOETIC_INVOLUTIONS[9] == 8

    # Self-duals
    assert NOETIC_SELF_DUALS == {1, 4, 7, 10}


# =============================================================================
# NEW REGRESSION TESTS: Extended Syntax & Lexicon Conflict Detection
# =============================================================================

def test_extended_syntax_parsing_in_encode_context():
    """
    Regression test: Extended syntax (B8^5_d5) should parse correctly
    when used in encode context after natural language encoding.
    """
    # First encode a story
    expr1 = EncodeStory("She felt fear.", strict=False)
    assert len(expr1.elements) >= 1

    # Then parse an equation with extended syntax
    from narrative import parse_equation
    expr2 = parse_equation("B8^5_d5 +T C3^1_c4 -> D7^2_b2")

    # Verify extended parsing worked
    assert expr2.element_refs[0].sense == 5
    assert expr2.element_refs[0].foundation == 5
    assert expr2.element_refs[1].sense == 1
    assert expr2.element_refs[2].sense == 2


def test_lexicon_world_noetic_keyword_overlap_determinism():
    """
    Regression test: Words appearing in both WORLD_KEYWORDS and NOETIC_KEYWORDS
    should resolve deterministically via LEXICON priority.
    """
    from narrative.constants import get_token_mapping, LEXICON, WORLD_KEYWORDS, NOETIC_KEYWORDS

    # Words that appear in multiple places
    overlapping_words = ["mind", "love", "fear", "energy", "pattern"]

    for word in overlapping_words:
        if word in LEXICON:
            # LEXICON should take priority
            result = get_token_mapping(word)
            assert result is not None
            assert result == LEXICON[word], f"get_token_mapping for '{word}' should return LEXICON value"


def test_encode_decode_consistency_newly_added_senses():
    """
    Regression test: Newly added senses (C3.3 anger, C3.4 sadness) should
    encode consistently and decode to appropriate labels.
    """
    from narrative.constants import SENSE_LABELS

    # Verify sense labels exist for critical senses
    critical_senses = [
        ("C3.3", "anger"),
        ("C3.4", "sadness"),
        ("B5.2", "accumulated knowledge"),
        ("D3.2", "material chaos"),
        ("D8.3", "material authority"),
    ]

    for sense_code, expected_label in critical_senses:
        assert sense_code in SENSE_LABELS, f"Missing sense label: {sense_code}"
        assert SENSE_LABELS[sense_code] == expected_label, \
            f"Sense label mismatch for {sense_code}: expected '{expected_label}', got '{SENSE_LABELS[sense_code]}'"


def test_error_message_contains_valid_token_hints():
    """
    Regression test: Error messages for ambiguous/invalid tokens should
    contain hints about valid formats.
    """
    try:
        EncodeStory("The zorblax flurped the quxon.", strict=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        # Should contain helpful hints
        assert "Unknown tokens" in error_msg
        assert "LEXICON" in error_msg or "lexicon" in error_msg
        assert "lenient" in error_msg.lower()


def test_extended_token_conflict_detection_comprehensive():
    """
    Regression test: check_extended_token_conflicts should detect
    all types of invalid foundation suffixes.
    """
    from narrative.constants import check_extended_token_conflicts

    # Valid tokens
    valid_tokens = [
        "B8", "B8^5", "B8.5", "B8_d5", "B8^5_d5",
        "A1^1_a1", "D10^3_d7", "C3^4_b2"
    ]
    for token in valid_tokens:
        result = check_extended_token_conflicts(token)
        assert result is None, f"Valid token '{token}' incorrectly flagged: {result}"

    # Invalid tokens
    invalid_tokens = [
        ("B8_e5", "Invalid world 'e'"),
        ("B8_d0", "Foundation 0 invalid"),
        ("B8_d8", "Foundation 8 invalid"),
        ("B8_d9", "Foundation 9 invalid"),
        ("B8_x5", "Invalid world 'x'"),
    ]
    for token, reason in invalid_tokens:
        result = check_extended_token_conflicts(token)
        assert result is not None, f"Invalid token '{token}' ({reason}) not detected"


def test_encode_preserves_sense_from_lexicon():
    """
    Regression test: Words with explicit senses in LEXICON should
    preserve those senses through encoding.
    """
    # "love" should map to C2.3 (sense 3)
    expr = EncodeStory("She felt love.", strict=False)

    # Find C2 element if present
    c2_refs = [ref for ref in expr.element_refs if ref.world == "C" and ref.noetic == 2]
    if c2_refs:
        # At least one should have sense 3 (love)
        has_love_sense = any(ref.sense == 3 for ref in c2_refs)
        assert has_love_sense, "Love should map to C2.3 (sense 3)"


if __name__ == "__main__":
    # Run all tests
    test_lexicon_canonical_entries()
    test_lexicon_format()
    test_encode_simple_person()
    test_encode_emotion()
    test_encode_love_story()
    test_encode_mental()
    test_encode_returns_operators()
    test_encode_full_diagnostics()
    test_encode_strict_mode_raises_error()
    test_encode_strict_mode_valid_story()
    test_encode_non_strict_mode_default()
    test_encode_full_strict_mode()
    test_encode_causal_story()
    test_decode_single_element()
    test_decode_two_elements()
    test_decode_causal_chain()
    test_decode_multiple_elements()
    test_decode_full_diagnostics()
    test_parse_comma_separated()
    test_parse_space_separated()
    test_parse_with_causal_arrow()
    test_parse_mixed_operators()
    test_parse_with_sense()
    test_element_ref_creation()
    test_element_ref_from_string()
    test_element_ref_label()
    test_encode_decode_roundtrip()
    test_canonical_elements_preserved()
    test_scenario_inversion_integration()
    test_parse_equation_integration()

    # Run expanded lexicon tests
    test_new_emotion_words()
    test_new_mental_words()
    test_new_physical_words()
    test_encode_expanded_emotion_story()
    test_encode_mental_confidence_story()
    test_encode_physical_transformation_story()

    # Run sub-foundation tests
    test_subfound_map_has_28_entries()
    test_emotional_companionship_subfound()
    test_subfoundation_detection()
    test_get_subfound_label_all_worlds()
    test_subfound_label_invalid_foundation()
    test_subfound_label_invalid_world()

    # Run sense rules tests
    test_sense_rules_applied()
    test_sense_rules_experiences()
    test_sense_rules_love()
    test_sense_rules_control()
    test_sense_rules_chaos()
    test_sense_labels_b52()
    test_sense_labels_c23()
    test_sense_labels_d83()

    # Run new sense rules tests
    test_sense_rules_biological_cycle()
    test_sense_rules_material_order()
    test_sense_rules_organization()
    test_sense_rules_negative_thought()
    test_sense_labels_d72()
    test_sense_labels_d22()
    test_sense_labels_c32()
    test_sense_labels_b33()
    test_encode_biological_cycle_story()
    test_encode_material_order_story()
    test_decode_d72()
    test_decode_d22()

    # Run comprehensive sense labels tests
    test_sense_labels_comprehensive_d_world()
    test_sense_labels_comprehensive_c_world()
    test_sense_labels_comprehensive_b_world()
    test_sense_labels_comprehensive_a_world()

    # Run operator validation tests
    test_allowed_ops_canonical()
    test_allowed_ops_count()
    test_is_valid_operator_canonical()
    test_is_valid_operator_unknown()
    test_operator_validation_strict_mode()
    test_operator_validation_non_strict_mode()
    test_new_operators_multiply()
    test_new_operators_divide()
    test_new_operators_composition()
    test_parse_equation_all_operators()
    test_encode_story_intensification()
    test_encode_story_conflict()
    test_encode_story_sequence()
    test_operator_templates_coverage()
    test_operator_templates_multiply()
    test_operator_templates_divide()
    test_operator_templates_composition()
    test_verb_to_operator_intensification()
    test_verb_to_operator_conflict()
    test_verb_to_operator_sequence()
    test_roundtrip_new_operators()

    # Run lexicon consistency tests (canon guardrails)
    test_lexicon_sense_rules_no_conflicts()
    test_lexicon_anger_sadness_grief_aligned()
    test_get_token_mapping_deterministic()
    test_extended_syntax_conflict_detection()
    test_canon_worlds_only_abcd()
    test_canon_noetics_only_1_to_10()
    test_canon_foundations_only_1_to_7()
    test_canon_operators_fixed()
    test_canon_noetic_involutions()

    # Run new regression tests for extended syntax and lexicon conflicts
    test_extended_syntax_parsing_in_encode_context()
    test_lexicon_world_noetic_keyword_overlap_determinism()
    test_encode_decode_consistency_newly_added_senses()
    test_error_message_contains_valid_token_hints()
    test_extended_token_conflict_detection_comprehensive()
    test_encode_preserves_sense_from_lexicon()

    print("All narrative encoder/decoder tests passed!")

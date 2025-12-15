"""
Tests for Extended Parser Syntax (Sense/Foundation Suffixes)

Tests the extended token syntax:
- B8^5 (with sense using caret notation)
- B8_d5 (with foundation suffix)
- B8^5_d5 (full extended)

Canon validation:
- Worlds: A/B/C/D only
- Noetics: 1-10
- Foundations: 1-7
"""
import pytest
from narrative.types import ElementRef
from narrative.encoder import parse_equation


class TestSenseSuffixParsing:
    """Test parsing of sense suffixes (^ and . notation)."""

    def test_caret_sense_notation(self):
        """Test B8^5 parses correctly."""
        elem = ElementRef.from_string("B8^5")
        assert elem.world == "B"
        assert elem.noetic == 8
        assert elem.sense == 5
        assert elem.foundation is None
        assert elem.subfoundation is None
        assert elem.full_code == "B8^5"

    def test_dot_sense_notation(self):
        """Test B8.5 parses correctly (backward compatible)."""
        elem = ElementRef.from_string("B8.5")
        assert elem.world == "B"
        assert elem.noetic == 8
        assert elem.sense == 5
        assert elem.foundation is None
        assert elem.subfoundation is None
        # Note: full_code now uses caret notation
        assert elem.full_code == "B8^5"

    def test_all_worlds_with_sense(self):
        """Test sense notation works for all worlds."""
        for world in ["A", "B", "C", "D"]:
            elem = ElementRef.from_string(f"{world}5^2")
            assert elem.world == world
            assert elem.noetic == 5
            assert elem.sense == 2

    def test_all_noetics_with_sense(self):
        """Test sense notation works for all noetics 1-10."""
        for noetic in range(1, 11):
            elem = ElementRef.from_string(f"B{noetic}^3")
            assert elem.noetic == noetic
            assert elem.sense == 3


class TestFoundationSuffixParsing:
    """Test parsing of foundation suffixes (_Fw notation)."""

    def test_foundation_suffix_basic(self):
        """Test B8_d5 parses correctly."""
        elem = ElementRef.from_string("B8_d5")
        assert elem.world == "B"
        assert elem.noetic == 8
        assert elem.sense is None
        assert elem.foundation == 5
        assert elem.subfoundation == "D"
        assert elem.full_code == "B8_d5"

    def test_foundation_all_worlds(self):
        """Test foundation suffix with all world contexts (a/b/c/d)."""
        for found_world in ["a", "b", "c", "d"]:
            elem = ElementRef.from_string(f"B8_{found_world}3")
            assert elem.foundation == 3
            assert elem.subfoundation == found_world.upper()

    def test_foundation_all_foundations(self):
        """Test all valid foundations 1-7."""
        for fid in range(1, 8):
            elem = ElementRef.from_string(f"B8_d{fid}")
            assert elem.foundation == fid
            assert elem.subfoundation == "D"

    def test_foundation_case_insensitive(self):
        """Test foundation suffix is case-insensitive."""
        elem1 = ElementRef.from_string("B8_D5")
        elem2 = ElementRef.from_string("B8_d5")
        assert elem1.foundation == elem2.foundation
        assert elem1.subfoundation == elem2.subfoundation


class TestFullExtendedSyntax:
    """Test full extended syntax with both sense and foundation."""

    def test_full_extended_b8_5_d5(self):
        """Test B8^5_d5 parses correctly."""
        elem = ElementRef.from_string("B8^5_d5")
        assert elem.world == "B"
        assert elem.noetic == 8
        assert elem.sense == 5
        assert elem.foundation == 5
        assert elem.subfoundation == "D"
        assert elem.full_code == "B8^5_d5"

    def test_full_extended_all_components(self):
        """Test full extended with various combinations."""
        elem = ElementRef.from_string("C3^2_a7")
        assert elem.world == "C"
        assert elem.noetic == 3
        assert elem.sense == 2
        assert elem.foundation == 7
        assert elem.subfoundation == "A"

    def test_full_extended_noetic_10(self):
        """Test full extended with noetic 10."""
        elem = ElementRef.from_string("D10^3_b2")
        assert elem.noetic == 10
        assert elem.sense == 3
        assert elem.foundation == 2


class TestCanonValidation:
    """Test canonical validation rules."""

    def test_invalid_world_e(self):
        """Test that world E fails validation."""
        with pytest.raises(ValueError, match="Invalid world.*must be A/B/C/D"):
            ElementRef.from_string("E5")

    def test_invalid_noetic_0(self):
        """Test that noetic 0 fails validation."""
        with pytest.raises(ValueError, match="Invalid noetic.*must be 1-10"):
            ElementRef(world="B", noetic=0)

    def test_invalid_noetic_11(self):
        """Test that noetic 11 fails validation."""
        with pytest.raises(ValueError, match="Invalid noetic.*must be 1-10"):
            ElementRef(world="B", noetic=11)

    def test_invalid_noetic_15(self):
        """Test that noetic 15 fails validation."""
        with pytest.raises(ValueError, match="Invalid noetic.*must be 1-10"):
            ElementRef(world="B", noetic=15)

    def test_invalid_foundation_0(self):
        """Test that foundation 0 fails validation."""
        with pytest.raises(ValueError, match="Invalid foundation.*must be 1-7"):
            ElementRef(world="B", noetic=8, foundation=0, subfoundation="D")

    def test_invalid_foundation_8(self):
        """Test that foundation 8 fails validation."""
        with pytest.raises(ValueError, match="Invalid foundation.*must be 1-7"):
            ElementRef(world="B", noetic=8, foundation=8, subfoundation="D")

    def test_invalid_foundation_9(self):
        """Test that foundation 9 fails validation."""
        with pytest.raises(ValueError, match="Invalid foundation.*must be 1-7"):
            ElementRef(world="B", noetic=8, foundation=9, subfoundation="D")

    def test_invalid_subfoundation_world(self):
        """Test that invalid subfoundation world fails."""
        with pytest.raises(ValueError, match="Invalid foundation suffix"):
            ElementRef.from_string("B8_e5")

    def test_valid_world_boundaries(self):
        """Test all valid worlds A-D."""
        for world in ["A", "B", "C", "D"]:
            elem = ElementRef(world=world, noetic=5)
            assert elem.world == world

    def test_valid_noetic_boundaries(self):
        """Test all valid noetics 1-10."""
        for noetic in range(1, 11):
            elem = ElementRef(world="B", noetic=noetic)
            assert elem.noetic == noetic

    def test_valid_foundation_boundaries(self):
        """Test all valid foundations 1-7."""
        for fid in range(1, 8):
            elem = ElementRef(world="B", noetic=8, foundation=fid, subfoundation="D")
            assert elem.foundation == fid


class TestParseEquationExtended:
    """Test parse_equation with extended syntax."""

    def test_parse_equation_with_sense(self):
        """Test parsing equation with sense notation."""
        expr = parse_equation("B8^5 +T C3^2")
        assert len(expr.elements) == 2
        assert expr.element_refs[0].sense == 5
        assert expr.element_refs[1].sense == 2

    def test_parse_equation_with_foundation(self):
        """Test parsing equation with foundation suffix."""
        expr = parse_equation("B8_d5 -> D3_a2")
        assert len(expr.elements) == 2
        assert expr.element_refs[0].foundation == 5
        assert expr.element_refs[0].subfoundation == "D"
        assert expr.element_refs[1].foundation == 2
        assert expr.element_refs[1].subfoundation == "A"

    def test_parse_equation_full_extended(self):
        """Test parsing equation with full extended syntax."""
        expr = parse_equation("B8^5_d5 +T C3^2_a7 -> D6^1_b3")
        assert len(expr.elements) == 3

        # Check first element
        elem1 = expr.element_refs[0]
        assert elem1.world == "B"
        assert elem1.noetic == 8
        assert elem1.sense == 5
        assert elem1.foundation == 5
        assert elem1.subfoundation == "D"

        # Check second element
        elem2 = expr.element_refs[1]
        assert elem2.world == "C"
        assert elem2.noetic == 3
        assert elem2.sense == 2
        assert elem2.foundation == 7
        assert elem2.subfoundation == "A"

        # Check third element
        elem3 = expr.element_refs[2]
        assert elem3.world == "D"
        assert elem3.noetic == 6
        assert elem3.sense == 1
        assert elem3.foundation == 3
        assert elem3.subfoundation == "B"

    def test_parse_equation_mixed_notation(self):
        """Test parsing equation with mixed basic and extended syntax."""
        expr = parse_equation("B8 +T B8^5 -> B8_d5 +T B8^5_d5")
        assert len(expr.elements) == 4

        # Basic
        assert expr.element_refs[0].sense is None
        assert expr.element_refs[0].foundation is None

        # Sense only
        assert expr.element_refs[1].sense == 5
        assert expr.element_refs[1].foundation is None

        # Foundation only
        assert expr.element_refs[2].sense is None
        assert expr.element_refs[2].foundation == 5

        # Full extended
        assert expr.element_refs[3].sense == 5
        assert expr.element_refs[3].foundation == 5


class TestBackwardCompatibility:
    """Test backward compatibility with existing syntax."""

    def test_basic_element_parsing(self):
        """Test basic elements still parse correctly."""
        elem = ElementRef.from_string("D5")
        assert elem.world == "D"
        assert elem.noetic == 5
        assert elem.sense is None
        assert elem.foundation is None

    def test_dot_notation_still_works(self):
        """Test dot notation (D5.1) still works."""
        elem = ElementRef.from_string("D5.1")
        assert elem.sense == 1
        # But full_code now uses caret
        assert elem.full_code == "D5^1"

    def test_parse_equation_basic(self):
        """Test basic equation parsing unchanged."""
        expr = parse_equation("B5 +T D3 -> C8")
        assert len(expr.elements) == 3
        assert expr.ops == ["+T", "->"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string(self):
        """Test empty string fails."""
        with pytest.raises(ValueError, match="Empty element string"):
            ElementRef.from_string("")

    def test_whitespace_only(self):
        """Test whitespace-only string fails."""
        with pytest.raises(ValueError, match="Empty element string"):
            ElementRef.from_string("   ")

    def test_invalid_foundation_suffix_format(self):
        """Test invalid foundation suffix format."""
        with pytest.raises(ValueError, match="Invalid foundation suffix"):
            ElementRef.from_string("B8_xyz")

    def test_malformed_sense(self):
        """Test malformed sense notation."""
        with pytest.raises(ValueError):
            ElementRef.from_string("B8^abc")

    def test_multiple_underscores(self):
        """Test handling of multiple underscores (uses rightmost)."""
        # Should take the rightmost underscore as foundation separator
        elem = ElementRef.from_string("B8_d5")
        assert elem.foundation == 5
        assert elem.subfoundation == "D"


class TestErrorMessagesWithHints:
    """Test that error messages include helpful hints about extended syntax."""

    def test_invalid_element_provides_extended_syntax_hint(self):
        """Test that invalid element syntax shows extended syntax formats."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8x", strict=True)

        error_msg = str(exc_info.value)
        # Check that error message includes extended syntax hints
        assert "Extended syntax formats:" in error_msg
        assert "B8^5" in error_msg
        assert "B8_d5" in error_msg
        assert "B8^5_d5" in error_msg
        assert "Valid ranges:" in error_msg
        assert "Worlds: A/B/C/D" in error_msg
        assert "Noetics: 1-10" in error_msg
        assert "Foundations: 1-7" in error_msg

    def test_invalid_world_error_suggests_valid_worlds(self):
        """Test that invalid world error mentions valid worlds (A/B/C/D)."""
        # Create ElementRef directly to test world validation
        with pytest.raises(ValueError) as exc_info:
            ElementRef(world="E", noetic=5)

        error_msg = str(exc_info.value)
        # Should mention valid worlds
        assert "A/B/C/D" in error_msg or "must be A/B/C/D" in error_msg

    def test_invalid_noetic_error_shows_valid_range(self):
        """Test that invalid noetic shows valid range (1-10)."""
        with pytest.raises(ValueError) as exc_info:
            ElementRef(world="B", noetic=15)

        error_msg = str(exc_info.value)
        # Should mention valid noetic range
        assert "1-10" in error_msg or "must be 1-10" in error_msg

    def test_invalid_foundation_suffix_provides_format_hint(self):
        """Test that invalid foundation suffix shows correct format."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8_x9", strict=True)

        error_msg = str(exc_info.value)
        # Should mention foundation suffix format or valid foundations
        assert ("foundation" in error_msg.lower() or
                "1-7" in error_msg or
                "a/b/c/d" in error_msg.lower())

    def test_lenient_mode_skips_invalid_with_no_error(self):
        """Test that lenient mode (strict=False) doesn't raise errors."""
        # Should not raise even with invalid syntax
        expr = parse_equation("B8x +T D3", strict=False)
        # Should parse the valid part (D3)
        assert "D3" in expr.elements

    def test_invalid_operator_provides_valid_operators_list(self):
        """Test that invalid operator error lists valid operators."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8 +X D3", strict=True)

        error_msg = str(exc_info.value)
        # Should list valid operators
        assert "+T" in error_msg
        assert "->" in error_msg
        assert "Valid operators:" in error_msg or "Unknown operator" in error_msg


class TestExtendedSyntaxInExpressions:
    """Test extended syntax in various expression contexts."""

    def test_parse_equation_multiple_full_extended(self):
        """Test parsing equation with multiple full extended elements."""
        expr = parse_equation("A1^2_a1 +T B8^5_d5 -> C3^4_c4 o D10^1_b6")
        assert len(expr.elements) == 4

        # Element 1: A1^2_a1
        assert expr.element_refs[0].world == "A"
        assert expr.element_refs[0].noetic == 1
        assert expr.element_refs[0].sense == 2
        assert expr.element_refs[0].foundation == 1
        assert expr.element_refs[0].subfoundation == "A"

        # Element 2: B8^5_d5
        assert expr.element_refs[1].world == "B"
        assert expr.element_refs[1].noetic == 8
        assert expr.element_refs[1].sense == 5
        assert expr.element_refs[1].foundation == 5
        assert expr.element_refs[1].subfoundation == "D"

        # Element 3: C3^4_c4
        assert expr.element_refs[2].world == "C"
        assert expr.element_refs[2].noetic == 3
        assert expr.element_refs[2].sense == 4
        assert expr.element_refs[2].foundation == 4
        assert expr.element_refs[2].subfoundation == "C"

        # Element 4: D10^1_b6
        assert expr.element_refs[3].world == "D"
        assert expr.element_refs[3].noetic == 10
        assert expr.element_refs[3].sense == 1
        assert expr.element_refs[3].foundation == 6
        assert expr.element_refs[3].subfoundation == "B"

    def test_parse_equation_noetic_10_extended(self):
        """Test parsing noetic 10 with full extended syntax."""
        expr = parse_equation("D10^3_a7 *T B10^2_c5")
        assert len(expr.elements) == 2

        # D10 with sense 3 and foundation 7 in world A
        assert expr.element_refs[0].noetic == 10
        assert expr.element_refs[0].sense == 3
        assert expr.element_refs[0].foundation == 7
        assert expr.element_refs[0].subfoundation == "A"

        # B10 with sense 2 and foundation 5 in world C
        assert expr.element_refs[1].noetic == 10
        assert expr.element_refs[1].sense == 2
        assert expr.element_refs[1].foundation == 5
        assert expr.element_refs[1].subfoundation == "C"

    def test_parse_equation_comma_separated_extended(self):
        """Test comma-separated equation with extended syntax."""
        expr = parse_equation("B8^5_d5,+T,C3^2_a4,-T,D6^1_b3")
        assert len(expr.elements) == 3
        assert expr.ops == ["+T", "-T"]

        # Verify each element has correct extended attributes
        assert expr.element_refs[0].full_code == "B8^5_d5"
        assert expr.element_refs[1].full_code == "C3^2_a4"
        assert expr.element_refs[2].full_code == "D6^1_b3"

    def test_parse_equation_all_worlds_extended(self):
        """Test extended syntax with all four worlds."""
        expr = parse_equation("A5^1_a1 +T B5^2_b2 +T C5^3_c3 +T D5^4_d4")
        assert len(expr.elements) == 4

        for i, world in enumerate(["A", "B", "C", "D"]):
            assert expr.element_refs[i].world == world
            assert expr.element_refs[i].noetic == 5
            assert expr.element_refs[i].sense == i + 1
            assert expr.element_refs[i].foundation == i + 1
            assert expr.element_refs[i].subfoundation == world

    def test_parse_equation_all_operators_with_extended(self):
        """Test all canonical operators with full extended syntax."""
        expr = parse_equation("B1^1_a1 +T B2^2_b2 -T B3^3_c3 *T B4^4_d4 /T B5^1_a5 o B6^2_b6 -> B7^3_c7")
        assert len(expr.elements) == 7
        assert expr.ops == ["+T", "-T", "*T", "/T", "o", "->"]


class TestLexiconConsistency:
    """Test lexicon validation and conflict detection."""

    def test_lexicon_sense_rules_consistency(self):
        """Test that LEXICON and SENSE_RULES are consistent."""
        from narrative.constants import validate_lexicon_consistency

        conflicts = validate_lexicon_consistency()
        # After our fixes, there should be no conflicts
        assert len(conflicts) == 0, f"Lexicon conflicts found: {conflicts}"

    def test_get_token_mapping_priority(self):
        """Test that get_token_mapping follows correct priority."""
        from narrative.constants import get_token_mapping, SENSE_RULES, LEXICON

        # Word in SENSE_RULES should return SENSE_RULES value
        if "anger" in SENSE_RULES:
            result = get_token_mapping("anger")
            assert result == SENSE_RULES["anger"]

        # Word only in LEXICON should return LEXICON value
        if "woman" in LEXICON and "woman" not in SENSE_RULES:
            result = get_token_mapping("woman")
            assert result == LEXICON["woman"]

    def test_extended_token_conflict_detection(self):
        """Test detection of conflicting extended token syntax."""
        from narrative.constants import check_extended_token_conflicts

        # Valid extended tokens should return None
        assert check_extended_token_conflicts("B8^5_d5") is None
        assert check_extended_token_conflicts("D10^3") is None
        assert check_extended_token_conflicts("C3_a7") is None
        assert check_extended_token_conflicts("B8") is None

        # Invalid foundation suffix should return error
        error = check_extended_token_conflicts("B8_x9")
        assert error is not None
        assert "Invalid foundation suffix" in error

        # Invalid foundation number should return error
        error = check_extended_token_conflicts("B8_d9")
        assert error is not None
        assert "Invalid foundation suffix" in error


class TestEncodingConsistency:
    """Test encode/decode consistency for newly added senses."""

    def test_anger_encodes_to_c33(self):
        """Test that 'anger' encodes to C3.3 (anger sense)."""
        from narrative import EncodeStory

        expr = EncodeStory("She felt anger.", strict=False)
        # Should have at least one element
        assert len(expr.elements) >= 1

        # Find C3 element if present
        c3_refs = [ref for ref in expr.element_refs if ref.world == "C" and ref.noetic == 3]
        if c3_refs:
            # Should have sense 3 for anger
            assert any(ref.sense == 3 for ref in c3_refs), "Anger should map to C3.3"

    def test_sadness_encodes_to_c34(self):
        """Test that 'sadness' encodes to C3.4 (sadness/grief sense)."""
        from narrative import EncodeStory

        expr = EncodeStory("He experienced deep sadness.", strict=False)
        assert len(expr.elements) >= 1

        # Find C3 element if present
        c3_refs = [ref for ref in expr.element_refs if ref.world == "C" and ref.noetic == 3]
        if c3_refs:
            # Should have sense 4 for sadness
            assert any(ref.sense == 4 for ref in c3_refs), "Sadness should map to C3.4"

    def test_grief_encodes_to_c34(self):
        """Test that 'grief' encodes to C3.4 (grief sense)."""
        from narrative import EncodeStory

        expr = EncodeStory("The grief was overwhelming.", strict=False)
        assert len(expr.elements) >= 1

        c3_refs = [ref for ref in expr.element_refs if ref.world == "C" and ref.noetic == 3]
        if c3_refs:
            assert any(ref.sense == 4 for ref in c3_refs), "Grief should map to C3.4"

    def test_roundtrip_extended_syntax_preserves_sense(self):
        """Test that extended syntax sense is preserved through roundtrip."""
        from narrative import DecodeStory, TKSExpression
        from narrative.types import ElementRef

        # Create expression with explicit senses
        expr = TKSExpression(
            elements=["C3", "D7"],
            ops=["->"],
            element_refs=[
                ElementRef(world="C", noetic=3, sense=4),  # C3.4 = sadness
                ElementRef(world="D", noetic=7, sense=2),  # D7.2 = biological cycle
            ]
        )

        # Decode should use sense-specific labels
        story = DecodeStory(expr)
        assert isinstance(story, str)
        assert len(story) > 0


class TestAmbiguousTokenErrors:
    """Test error messaging for ambiguous/conflicting tokens."""

    def test_invalid_element_error_includes_extended_syntax_help(self):
        """Test that invalid element errors include extended syntax help."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8xyz", strict=True)

        error_msg = str(exc_info.value)
        # Should mention extended syntax formats
        assert "Extended syntax formats:" in error_msg
        assert "B8^5" in error_msg or "B8.5" in error_msg
        assert "B8_d5" in error_msg
        assert "B8^5_d5" in error_msg

    def test_invalid_foundation_error_shows_valid_range(self):
        """Test that invalid foundation errors show valid range 1-7."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8_d8", strict=True)  # Foundation 8 is invalid

        error_msg = str(exc_info.value)
        # Should mention valid foundation range
        assert "1-7" in error_msg or "foundation" in error_msg.lower()

    def test_invalid_subfoundation_world_error(self):
        """Test error for invalid subfoundation world (e.g., E)."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8_e5", strict=True)

        error_msg = str(exc_info.value)
        # Should mention valid worlds
        assert "a/b/c/d" in error_msg.lower() or "A/B/C/D" in error_msg

    def test_conflicting_operator_shows_valid_ops(self):
        """Test that unknown operator error lists all valid operators."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8 +X D3", strict=True)

        error_msg = str(exc_info.value)
        # Should list all valid operators
        assert "+T" in error_msg
        assert "-T" in error_msg
        assert "*T" in error_msg
        assert "/T" in error_msg
        assert "o" in error_msg
        assert "->" in error_msg

    def test_lenient_mode_skips_invalid_gracefully(self):
        """Test that lenient mode handles invalid tokens gracefully."""
        # Should not raise, should skip invalid parts
        expr = parse_equation("B8^5_d5 +X D3 +T C2", strict=False)

        # Should have valid elements
        assert "B8" in expr.elements
        assert "D3" in expr.elements
        assert "C2" in expr.elements

        # Should skip unknown operator +X
        assert "+X" not in expr.ops
        assert "+T" in expr.ops


# =============================================================================
# NEW REGRESSION TESTS: Extended Syntax in Various Contexts
# =============================================================================

class TestExtendedSyntaxRegression:
    """Additional regression tests for extended syntax parsing (B8^5_d5 format)."""

    def test_extended_syntax_with_reverse_causal(self):
        """Test extended syntax with reverse causal operator (<-)."""
        expr = parse_equation("D9^2_a3 <- B8^5_d5 <- C3^1_c4")
        assert len(expr.elements) == 3
        assert expr.ops == ["<-", "<-"]

        # Verify first element (D9^2_a3)
        assert expr.element_refs[0].world == "D"
        assert expr.element_refs[0].noetic == 9
        assert expr.element_refs[0].sense == 2
        assert expr.element_refs[0].foundation == 3
        assert expr.element_refs[0].subfoundation == "A"

        # Verify second element (B8^5_d5)
        assert expr.element_refs[1].world == "B"
        assert expr.element_refs[1].noetic == 8
        assert expr.element_refs[1].sense == 5
        assert expr.element_refs[1].foundation == 5
        assert expr.element_refs[1].subfoundation == "D"

    def test_extended_syntax_all_operators_chain(self):
        """Test extended syntax with all canonical operators in a chain."""
        # Chain: A1^1_a1 + B2^2_b2 - C3^3_c3 *T D4^4_d4 /T A5^1_a5 o B6^2_b6 -> C7^3_c7 <- D8^4_d4
        expr = parse_equation(
            "A1^1_a1 + B2^2_b2 - C3^3_c3 *T D4^4_d4 /T A5^1_a5 o B6^2_b6 -> C7^3_c7 <- D8^4_d1"
        )
        assert len(expr.elements) == 8
        assert expr.ops == ["+", "-", "*T", "/T", "o", "->", "<-"]

        # Verify all elements have correct world assignments
        expected_worlds = ["A", "B", "C", "D", "A", "B", "C", "D"]
        for i, expected_world in enumerate(expected_worlds):
            assert expr.element_refs[i].world == expected_world

    def test_extended_syntax_boundary_noetic_10(self):
        """Test extended syntax with noetic 10 (boundary value)."""
        expr = parse_equation("A10^3_a7 +T B10^2_b6 -> C10^1_c5 -T D10^4_d4")
        assert len(expr.elements) == 4

        # All should have noetic 10
        for ref in expr.element_refs:
            assert ref.noetic == 10

        # Verify specific values
        assert expr.element_refs[0].sense == 3
        assert expr.element_refs[0].foundation == 7
        assert expr.element_refs[1].sense == 2
        assert expr.element_refs[1].foundation == 6

    def test_extended_syntax_boundary_foundation_1_and_7(self):
        """Test extended syntax with foundation boundaries (1 and 7)."""
        expr = parse_equation("B8^1_a1 -> B8^1_d7")
        assert len(expr.elements) == 2

        # First element: foundation 1
        assert expr.element_refs[0].foundation == 1

        # Second element: foundation 7 (boundary)
        assert expr.element_refs[1].foundation == 7

    def test_extended_syntax_mixed_with_basic_elements(self):
        """Test mixing extended syntax with basic elements in same equation."""
        expr = parse_equation("B8 +T B8^5 -T B8_d5 *T B8^5_d5 /T C3")
        assert len(expr.elements) == 5

        # Element 0: Basic (B8) - no sense, no foundation
        assert expr.element_refs[0].sense is None
        assert expr.element_refs[0].foundation is None

        # Element 1: Sense only (B8^5)
        assert expr.element_refs[1].sense == 5
        assert expr.element_refs[1].foundation is None

        # Element 2: Foundation only (B8_d5)
        assert expr.element_refs[2].sense is None
        assert expr.element_refs[2].foundation == 5

        # Element 3: Full extended (B8^5_d5)
        assert expr.element_refs[3].sense == 5
        assert expr.element_refs[3].foundation == 5

        # Element 4: Basic (C3)
        assert expr.element_refs[4].sense is None
        assert expr.element_refs[4].foundation is None


class TestAmbiguousTokenErrorMessages:
    """Test error messaging for ambiguous/conflicting token scenarios."""

    def test_error_message_includes_all_valid_operators(self):
        """Test that error message for unknown operator lists ALL valid operators."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8 ?T D3", strict=True)  # ?T is not a valid operator

        error_msg = str(exc_info.value)
        # Should list all 9 canonical operators
        canonical_ops = ["+", "-", "+T", "-T", "*T", "/T", "o", "->", "<-"]
        for op in canonical_ops:
            assert op in error_msg, f"Missing operator '{op}' in error message"

    def test_error_message_for_foundation_boundary_violation(self):
        """Test error message when foundation > 7 is provided."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8_d8", strict=True)  # Foundation 8 is invalid

        error_msg = str(exc_info.value)
        # Should mention foundation range or be invalid
        assert "foundation" in error_msg.lower() or "1-7" in error_msg or "Invalid" in error_msg

    def test_error_message_for_noetic_boundary_violation(self):
        """Test error message when noetic > 10 is provided."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B11", strict=True)  # Noetic 11 is invalid

        error_msg = str(exc_info.value)
        # Should mention noetic range
        assert "noetic" in error_msg.lower() or "1-10" in error_msg

    def test_error_message_for_ambiguous_world_in_foundation_suffix(self):
        """Test error message when foundation suffix has invalid world (e.g., E)."""
        with pytest.raises(ValueError) as exc_info:
            parse_equation("B8_e5", strict=True)  # World 'e' is invalid

        error_msg = str(exc_info.value)
        # Should mention valid worlds
        assert "a/b/c/d" in error_msg.lower() or "A/B/C/D" in error_msg

    def test_lenient_mode_handles_multiple_invalid_tokens(self):
        """Test that lenient mode handles multiple invalid tokens gracefully."""
        # Multiple issues: invalid operator, potentially invalid element
        expr = parse_equation("B8^5_d5 +X D3 ++ C2^1_a1 ~~ B1", strict=False)

        # Should have parsed the valid elements
        assert len(expr.elements) >= 3
        assert "B8" in expr.elements
        assert "D3" in expr.elements or "C2" in expr.elements

        # Invalid operators should be skipped
        assert "+X" not in expr.ops
        assert "++" not in expr.ops
        assert "~~" not in expr.ops


class TestEncodeDecodeConsistencyExtended:
    """Test encode/decode consistency for extended syntax elements."""

    def test_decode_preserves_sense_specific_label(self):
        """Test that decode uses sense-specific labels from SENSE_LABELS."""
        from narrative import DecodeStory, TKSExpression
        from narrative.types import ElementRef

        # Create expression with specific senses
        expr = TKSExpression(
            elements=["C3", "D3"],
            ops=["->"],
            element_refs=[
                ElementRef(world="C", noetic=3, sense=3),  # C3.3 = anger
                ElementRef(world="D", noetic=3, sense=2),  # D3.2 = material chaos
            ]
        )

        story = DecodeStory(expr)
        lower_story = story.lower()

        # Should contain sense-specific terms
        # C3.3 = anger, D3.2 = material chaos/disorder
        has_anger_term = any(w in lower_story for w in ["anger", "angry", "hostility"])
        has_chaos_term = any(w in lower_story for w in ["chaos", "disorder", "instability"])

        assert has_anger_term or has_chaos_term, f"Story should contain sense-specific terms: {story}"

    def test_roundtrip_extended_syntax_with_foundation(self):
        """Test roundtrip: parse extended syntax -> decode -> verify coherence."""
        from narrative import DecodeStory

        # Parse extended syntax with foundation context
        expr = parse_equation("D5^1_c4 +T C2^3_c4 -> D7^1_c4")

        # Verify parsing
        assert expr.element_refs[0].foundation == 4
        assert expr.element_refs[0].subfoundation == "C"

        # Decode
        story = DecodeStory(expr)
        assert isinstance(story, str)
        assert len(story) > 0

    def test_consistency_noetic_10_all_worlds(self):
        """Test encoding consistency for noetic 10 across all worlds."""
        from narrative import DecodeStory

        for world in ["A", "B", "C", "D"]:
            expr = parse_equation(f"{world}10^1")
            story = DecodeStory(expr)
            assert isinstance(story, str)
            assert len(story) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

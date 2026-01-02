#!/usr/bin/env python3
"""
Regression tests for extended notation validation.

Tests the CanonicalValidator's ability to handle:
- Base elements: A1, D10
- With sense: A1^3, B10^5 (sense 1-9)
- With foundation: A1_d5, D10_d7 (foundation 1-7)
- Full extended: A1^3_d5, B10^5_d7

Author: TKS-LLM Extended Notation Tests
Date: 2025-12-14
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.phase6_eval import CanonicalValidator


class TestExtendedNotationValidator:
    """Test suite for extended notation validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return CanonicalValidator()

    # =========================================================================
    # Valid element tests (should pass)
    # =========================================================================

    @pytest.mark.parametrize("element", [
        # Base elements
        "A1", "A5", "A10",
        "B1", "B5", "B10",
        "C1", "C5", "C10",
        "D1", "D5", "D10",
    ])
    def test_valid_base_elements(self, validator, element):
        """Test that base elements pass validation."""
        assert validator.validate_element(element), f"Base element {element} should be valid"

    @pytest.mark.parametrize("element", [
        # With sense (^1-9)
        "A1^1", "A1^3", "A1^9",
        "B10^1", "B10^5", "B10^9",
        "C5^3", "D7^7",
    ])
    def test_valid_elements_with_sense(self, validator, element):
        """Test that elements with sense annotation pass validation."""
        assert validator.validate_element(element), f"Element with sense {element} should be valid"

    @pytest.mark.parametrize("element", [
        # With foundation (_d1-7)
        "A1_d1", "A1_d5", "A1_d7",
        "B10_d1", "B10_d3", "B10_d7",
        "C5_d2", "D7_d6",
    ])
    def test_valid_elements_with_foundation(self, validator, element):
        """Test that elements with foundation annotation pass validation."""
        assert validator.validate_element(element), f"Element with foundation {element} should be valid"

    @pytest.mark.parametrize("element", [
        # Full extended notation (^sense_dF)
        "A1^3_d5", "A10^5_d7",
        "B8^3_d4", "B10^9_d1",
        "C5^2_d3", "C10^7_d6",
        "D3^1_d2", "D10^8_d7",
    ])
    def test_valid_full_extended_notation(self, validator, element):
        """Test that full extended notation elements pass validation."""
        assert validator.validate_element(element), f"Full extended {element} should be valid"

    # =========================================================================
    # Invalid element tests (should fail)
    # =========================================================================

    @pytest.mark.parametrize("element", [
        # Invalid sense (^0, ^10)
        "A1^0",   # Sense 0 is invalid
        "A1^10",  # Sense must be 1-9
    ])
    def test_invalid_sense_values(self, validator, element):
        """Test that invalid sense values fail validation."""
        assert not validator.validate_element(element), f"Invalid sense {element} should fail"

    @pytest.mark.parametrize("element", [
        # Invalid foundation (_d0, _d8)
        "A1_d0",  # Foundation 0 is invalid
        "A1_d8",  # Foundation must be 1-7
        "A1_d9",  # Foundation must be 1-7
    ])
    def test_invalid_foundation_values(self, validator, element):
        """Test that invalid foundation values fail validation."""
        assert not validator.validate_element(element), f"Invalid foundation {element} should fail"

    @pytest.mark.parametrize("element", [
        # Invalid worlds
        "E1", "F5", "G10", "Z1",
        "E1^3_d5",  # Invalid world with valid extended notation
    ])
    def test_invalid_worlds(self, validator, element):
        """Test that non-canonical worlds fail validation."""
        assert not validator.validate_element(element), f"Invalid world {element} should fail"

    @pytest.mark.parametrize("element", [
        # Invalid noetics
        "A0",   # Noetic 0 is invalid
        "A11",  # Noetic must be 1-10
        "A99",  # Noetic must be 1-10
    ])
    def test_invalid_noetics(self, validator, element):
        """Test that invalid noetic values fail validation."""
        assert not validator.validate_element(element), f"Invalid noetic {element} should fail"

    @pytest.mark.parametrize("element", [
        # Malformed patterns
        "1A",      # Wrong order
        "AA1",     # Double world letter
        "A",       # Missing noetic
        "10",      # Missing world
        "",        # Empty
        "A1^",     # Incomplete sense
        "A1_d",    # Incomplete foundation
        "A1_",     # Incomplete foundation
    ])
    def test_malformed_elements(self, validator, element):
        """Test that malformed elements fail validation."""
        assert not validator.validate_element(element), f"Malformed element {element} should fail"

    # =========================================================================
    # Base element extraction tests
    # =========================================================================

    @pytest.mark.parametrize("element,expected_base", [
        ("A1", "A1"),
        ("B10", "B10"),
        ("A1^3", "A1"),
        ("B10^5", "B10"),
        ("A1_d5", "A1"),
        ("B10_d7", "B10"),
        ("A1^3_d5", "A1"),
        ("B8^3_d4", "B8"),
        ("C10^9_d7", "C10"),
    ])
    def test_extract_base_element(self, validator, element, expected_base):
        """Test base element extraction from extended notation."""
        result = validator.extract_base_element(element)
        assert result == expected_base, f"Base of {element} should be {expected_base}, got {result}"

    # =========================================================================
    # Expression validation tests
    # =========================================================================

    def test_valid_expression_base_elements(self, validator):
        """Test expression validation with base elements."""
        elements = ["A1", "B2", "C3"]
        ops = ["+T", "-T"]
        assert validator.validate_expression(elements, ops)

    def test_valid_expression_extended_notation(self, validator):
        """Test expression validation with extended notation elements."""
        elements = ["A1^3_d5", "B8^5_d4", "C10_d7"]
        ops = ["+", "->"]
        assert validator.validate_expression(elements, ops)

    def test_valid_expression_mixed_notation(self, validator):
        """Test expression validation with mixed base and extended notation."""
        elements = ["A1", "B8^3_d4", "C10", "D5_d2"]
        ops = ["+T", "o", "-T"]
        assert validator.validate_expression(elements, ops)

    def test_invalid_expression_bad_element(self, validator):
        """Test expression validation fails with invalid element."""
        elements = ["A1", "E5", "C3"]  # E5 is invalid world
        ops = ["+T", "-T"]
        assert not validator.validate_expression(elements, ops)

    def test_invalid_expression_bad_operator(self, validator):
        """Test expression validation fails with invalid operator."""
        elements = ["A1", "B2", "C3"]
        ops = ["+T", "INVALID"]  # Invalid operator
        assert not validator.validate_expression(elements, ops)

    # =========================================================================
    # Case insensitivity tests
    # =========================================================================

    @pytest.mark.parametrize("element", [
        "a1", "b10", "c5^3", "d7_d5",
        "A1^3_D5", "b8^3_D4",  # Mixed case
    ])
    def test_case_insensitive_elements(self, validator, element):
        """Test that validation is case-insensitive."""
        assert validator.validate_element(element), f"Element {element} should be valid (case-insensitive)"


class TestOperatorValidation:
    """Test suite for operator validation."""

    @pytest.fixture
    def validator(self):
        return CanonicalValidator()

    @pytest.mark.parametrize("op", [
        "+", "-", "+T", "-T", "->", "<-", "*T", "/T", "o"
    ])
    def test_valid_operators(self, validator, op):
        """Test that all 9 canonical operators are valid."""
        assert validator.validate_operator(op), f"Operator {op} should be valid"

    @pytest.mark.parametrize("op", [
        "*", "/", "++", "--", "**", "//", "x", "X", "÷", "×", "INVALID"
    ])
    def test_invalid_operators(self, validator, op):
        """Test that non-canonical operators are invalid."""
        assert not validator.validate_operator(op), f"Operator {op} should be invalid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

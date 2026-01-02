#!/usr/bin/env python3
"""
Unit tests for semantic_roundtrip_metrics module.

Tests all core functions including element extraction, similarity metrics,
and round-trip validation logic.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.semantic_roundtrip_metrics import (
    extract_tks_elements,
    normalize_element,
    compute_element_overlap,
    _fallback_similarity,
)


def test_extract_tks_elements():
    """Test TKS element extraction from text."""
    print("\n=== Testing TKS Element Extraction ===")

    # Test basic elements
    text1 = "Using A1 and B5 gives C10"
    elements1 = extract_tks_elements(text1)
    print(f"Input: {text1}")
    print(f"Extracted: {elements1}")
    assert 'A1' in elements1
    assert 'B5' in elements1
    assert 'C10' in elements1
    print("✓ Basic elements extracted correctly")

    # Test extended notation
    text2 = "Complex: A1_d2 ⊕ B5^r3 → C3_d1^r2"
    elements2 = extract_tks_elements(text2)
    print(f"\nInput: {text2}")
    print(f"Extracted: {elements2}")
    assert 'A1_D2' in elements2
    assert 'B5^R3' in elements2
    assert 'C3_D1^R2' in elements2
    print("✓ Extended notation extracted correctly")

    # Test case insensitivity
    text3 = "lowercase: a1, b5, c3"
    elements3 = extract_tks_elements(text3)
    print(f"\nInput: {text3}")
    print(f"Extracted: {elements3}")
    assert 'A1' in elements3
    assert 'B5' in elements3
    assert 'C3' in elements3
    print("✓ Case-insensitive extraction works")

    # Test no elements
    text4 = "No elements here"
    elements4 = extract_tks_elements(text4)
    print(f"\nInput: {text4}")
    print(f"Extracted: {elements4}")
    assert len(elements4) == 0
    print("✓ Empty extraction works")

    # Test mixed with operators
    text5 = "A1 ⊕ B5 ⊖ C3 ⊗ D10"
    elements5 = extract_tks_elements(text5)
    print(f"\nInput: {text5}")
    print(f"Extracted: {elements5}")
    assert len(elements5) == 4
    print("✓ Extraction with operators works")


def test_normalize_element():
    """Test element normalization."""
    print("\n\n=== Testing Element Normalization ===")

    assert normalize_element("a1") == "A1"
    assert normalize_element("A1") == "A1"
    assert normalize_element(" b5 ") == "B5"
    assert normalize_element("c3_d2") == "C3_D2"
    print("✓ Element normalization works correctly")


def test_compute_element_overlap():
    """Test Jaccard similarity for element overlap."""
    print("\n\n=== Testing Element Overlap Computation ===")

    # Identical elements
    eq1 = "A1 ⊕ B5 ⊕ C3"
    eq2 = "A1 and B5 and C3"
    overlap1 = compute_element_overlap(eq1, eq2)
    print(f"\nEq1: {eq1}")
    print(f"Eq2: {eq2}")
    print(f"Overlap: {overlap1:.3f}")
    assert overlap1 == 1.0
    print("✓ Identical elements: 1.0")

    # Partial overlap
    eq3 = "A1 ⊕ B5 ⊕ C3"
    eq4 = "A1 ⊕ B5"
    overlap2 = compute_element_overlap(eq3, eq4)
    print(f"\nEq1: {eq3}")
    print(f"Eq2: {eq4}")
    print(f"Overlap: {overlap2:.3f}")
    assert abs(overlap2 - 0.667) < 0.01  # 2/3 overlap
    print("✓ Partial overlap: ~0.67 (2/3)")

    # No overlap
    eq5 = "A1 ⊕ B5"
    eq6 = "C3 ⊕ D10"
    overlap3 = compute_element_overlap(eq5, eq6)
    print(f"\nEq1: {eq5}")
    print(f"Eq2: {eq6}")
    print(f"Overlap: {overlap3:.3f}")
    assert overlap3 == 0.0
    print("✓ No overlap: 0.0")

    # Both empty
    eq7 = "No elements"
    eq8 = "Also no elements"
    overlap4 = compute_element_overlap(eq7, eq8)
    print(f"\nEq1: {eq7}")
    print(f"Eq2: {eq8}")
    print(f"Overlap: {overlap4:.3f}")
    assert overlap4 == 1.0  # Both empty = perfect match
    print("✓ Both empty: 1.0")

    # Extended notation
    eq9 = "A1_d2 ⊕ B5^r3"
    eq10 = "A1_D2 and B5^R3"
    overlap5 = compute_element_overlap(eq9, eq10)
    print(f"\nEq1: {eq9}")
    print(f"Eq2: {eq10}")
    print(f"Overlap: {overlap5:.3f}")
    assert overlap5 == 1.0
    print("✓ Extended notation normalization works")


def test_fallback_similarity():
    """Test fallback character-level similarity."""
    print("\n\n=== Testing Fallback Similarity ===")

    # Identical strings
    text1 = "She felt fear and anxiety."
    text2 = "She felt fear and anxiety."
    sim1 = _fallback_similarity(text1, text2)
    print(f"\nText1: {text1}")
    print(f"Text2: {text2}")
    print(f"Similarity: {sim1:.3f}")
    assert sim1 == 1.0
    print("✓ Identical strings: 1.0")

    # Different strings
    text3 = "She felt fear."
    text4 = "He thought power."
    sim2 = _fallback_similarity(text3, text4)
    print(f"\nText1: {text3}")
    print(f"Text2: {text4}")
    print(f"Similarity: {sim2:.3f}")
    assert 0.0 <= sim2 <= 1.0
    print(f"✓ Different strings: {sim2:.3f}")

    # Case insensitive
    text5 = "HELLO"
    text6 = "hello"
    sim3 = _fallback_similarity(text5, text6)
    print(f"\nText1: {text5}")
    print(f"Text2: {text6}")
    print(f"Similarity: {sim3:.3f}")
    assert sim3 == 1.0
    print("✓ Case insensitive: 1.0")


def test_integration_examples():
    """Test realistic integration examples."""
    print("\n\n=== Testing Integration Examples ===")

    # Example 1: Story with TKS elements
    story1 = "In the realm of desire (A1), the seeker found wisdom (B5)"
    elements1 = extract_tks_elements(story1)
    print(f"\nStory: {story1}")
    print(f"Elements: {elements1}")
    assert 'A1' in elements1
    assert 'B5' in elements1
    print("✓ Story element extraction works")

    # Example 2: Equation comparison
    original_eq = "A1_d2 ⊕ B5 → C10"
    regenerated_eq = "A1_D2 and B5 gives C10"
    overlap = compute_element_overlap(original_eq, regenerated_eq)
    print(f"\nOriginal: {original_eq}")
    print(f"Regenerated: {regenerated_eq}")
    print(f"Overlap: {overlap:.3f}")
    assert overlap == 1.0
    print("✓ Equation round-trip perfect")

    # Example 3: Partial regeneration
    original_eq2 = "A1 ⊕ B5 ⊕ C3 → D10"
    partial_eq = "A1 ⊕ B5 → C10"  # Missing C3, extra C10
    overlap2 = compute_element_overlap(original_eq2, partial_eq)
    print(f"\nOriginal: {original_eq2}")
    print(f"Partial: {partial_eq}")
    print(f"Overlap: {overlap2:.3f}")
    # A1, B5, C3, D10 vs A1, B5, C10 = intersection {A1, B5} = 2, union = 5
    # Jaccard = 2/5 = 0.4
    assert 0.3 <= overlap2 <= 0.5  # Partial match
    print("✓ Partial regeneration detected")


def run_all_tests():
    """Run all test functions."""
    print("="*70)
    print("SEMANTIC ROUNDTRIP METRICS - UNIT TESTS")
    print("="*70)

    test_functions = [
        test_extract_tks_elements,
        test_normalize_element,
        test_compute_element_overlap,
        test_fallback_similarity,
        test_integration_examples,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {test_func.__name__}")
            print(f"  Exception: {e}")
            failed += 1

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(test_functions)}")
    print(f"Failed: {failed}/{len(test_functions)}")

    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())

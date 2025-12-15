"""
Test Suite for Canonical Validator

Tests the canonical validation system for TKS expressions.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.canonical_validator import (
    validate_canonical,
    validate_entry,
    compute_validation_metrics
)
from narrative import EncodeStory
from narrative.types import TKSExpression


def test_valid_expressions():
    """Test validation of valid TKS expressions."""
    print("=" * 80)
    print("TEST: Valid Expressions")
    print("=" * 80)

    test_cases = [
        ("Single element", TKSExpression(elements=["D5"], ops=[])),
        ("Simple causal", TKSExpression(elements=["A5", "D8"], ops=["->"])),
        ("Multi-element", TKSExpression(elements=["B2", "C3", "D10"], ops=["+T", "->"])),
        ("All operators", TKSExpression(elements=["A1", "B2", "C3", "D4"], ops=["+T", "-T", "->"])),
        ("Encoded story", EncodeStory("She felt fear.", strict=False)),
    ]

    passed = 0
    failed = 0

    for name, expr in test_cases:
        is_valid, errors = validate_canonical(expr)
        if is_valid:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            for error in errors:
                print(f"    - {error}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    print()
    return failed == 0


def test_invalid_expressions():
    """Test validation of invalid TKS expressions."""
    print("=" * 80)
    print("TEST: Invalid Expressions (Should Fail)")
    print("=" * 80)

    test_cases = [
        ("Invalid world", TKSExpression(elements=["X5"], ops=[]), "Invalid world"),
        ("Invalid noetic (too high)", TKSExpression(elements=["D15"], ops=[]), "Invalid noetic"),
        ("Invalid noetic (zero)", TKSExpression(elements=["D0"], ops=[]), "Invalid noetic"),
        ("Invalid operator", TKSExpression(elements=["D5", "B2"], ops=["++"]), "Invalid operator"),
        ("Structural mismatch", TKSExpression(elements=["D5", "B2", "C3"], ops=["->"]), "Structural inconsistency"),
        ("Empty element", TKSExpression(elements=["", "D5"], ops=["->"]), "Empty element"),
    ]

    passed = 0
    failed = 0

    for name, expr, expected_error in test_cases:
        is_valid, errors = validate_canonical(expr)
        if not is_valid and any(expected_error in e for e in errors):
            print(f"  [PASS] {name} (correctly rejected)")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            if is_valid:
                print(f"    ERROR: Expression was incorrectly validated as valid")
            else:
                print(f"    ERROR: Expected error containing '{expected_error}'")
                print(f"    Got errors: {errors}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    print()
    return failed == 0


def test_entry_validation():
    """Test validation of entry dicts."""
    print("=" * 80)
    print("TEST: Entry Dict Validation")
    print("=" * 80)

    test_cases = [
        ("Valid entry", {
            "expr_elements": ["A5", "D8"],
            "expr_ops": ["->"]
        }, True),
        ("Invalid world", {
            "expr_elements": ["X5", "D3"],
            "expr_ops": ["->"]
        }, False),
        ("Invalid noetic", {
            "expr_elements": ["D15", "B2"],
            "expr_ops": ["->"]
        }, False),
    ]

    passed = 0
    failed = 0

    for name, entry, should_be_valid in test_cases:
        is_valid, errors = validate_entry(entry)
        if is_valid == should_be_valid:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            print(f"    Expected valid={should_be_valid}, got valid={is_valid}")
            if errors:
                for error in errors:
                    print(f"    - {error}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    print()
    return failed == 0


def test_validation_metrics():
    """Test aggregate validation metrics."""
    print("=" * 80)
    print("TEST: Validation Metrics")
    print("=" * 80)

    entries = [
        {"expr_elements": ["A5", "D8"], "expr_ops": ["->"]},  # Valid
        {"expr_elements": ["X5", "D3"], "expr_ops": ["->"]},  # Invalid world
        {"expr_elements": ["D15", "B2"], "expr_ops": ["->"]},  # Invalid noetic
        {"expr_elements": ["B2", "C3"], "expr_ops": ["++"]},  # Invalid operator
        {"expr_elements": ["C1", "D5"], "expr_ops": ["->"]},  # Valid
    ]

    metrics = compute_validation_metrics(entries)

    print(f"  Total entries:    {metrics['total']}")
    print(f"  Valid entries:    {metrics['valid']}")
    print(f"  Invalid entries:  {metrics['invalid']}")
    print(f"  Pass rate:        {metrics['pass_rate']:.1%}")
    print(f"  Error breakdown:")
    for error_type, count in sorted(metrics['error_counts'].items()):
        print(f"    {error_type}: {count}")

    # Verify metrics
    expected_total = 5
    expected_valid = 2
    expected_invalid = 3
    expected_pass_rate = 0.4

    passed = True
    if metrics['total'] != expected_total:
        print(f"  [FAIL] Expected total={expected_total}, got {metrics['total']}")
        passed = False
    if metrics['valid'] != expected_valid:
        print(f"  [FAIL] Expected valid={expected_valid}, got {metrics['valid']}")
        passed = False
    if metrics['invalid'] != expected_invalid:
        print(f"  [FAIL] Expected invalid={expected_invalid}, got {metrics['invalid']}")
        passed = False
    if abs(metrics['pass_rate'] - expected_pass_rate) > 0.01:
        print(f"  [FAIL] Expected pass_rate={expected_pass_rate}, got {metrics['pass_rate']}")
        passed = False

    if passed:
        print()
        print(f"  [PASS] All metrics correct")

    print()
    return passed


def test_foundation_validation():
    """Test validation of expressions with foundations."""
    print("=" * 80)
    print("TEST: Foundation Validation")
    print("=" * 80)

    test_cases = [
        ("Valid foundation", TKSExpression(
            elements=["D5"],
            ops=[],
            foundations=[(3, "D")]
        ), True),
        ("Invalid foundation (too high)", TKSExpression(
            elements=["D5"],
            ops=[],
            foundations=[(8, "D")]
        ), False),
        ("Invalid foundation world", TKSExpression(
            elements=["D5"],
            ops=[],
            foundations=[(3, "X")]
        ), False),
    ]

    passed = 0
    failed = 0

    for name, expr, should_be_valid in test_cases:
        is_valid, errors = validate_canonical(expr)
        if is_valid == should_be_valid:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            print(f"    Expected valid={should_be_valid}, got valid={is_valid}")
            if errors:
                for error in errors:
                    print(f"    - {error}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    print()
    return failed == 0


def main():
    """Run all tests."""
    print()
    print("=" * 80)
    print("CANONICAL VALIDATOR TEST SUITE")
    print("=" * 80)
    print()

    results = []
    results.append(("Valid Expressions", test_valid_expressions()))
    results.append(("Invalid Expressions", test_invalid_expressions()))
    results.append(("Entry Validation", test_entry_validation()))
    results.append(("Validation Metrics", test_validation_metrics()))
    results.append(("Foundation Validation", test_foundation_validation()))

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_passed = sum(1 for _, passed in results if passed)
    total_failed = len(results) - total_passed

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    print()
    print(f"Total: {total_passed}/{len(results)} test suites passed")

    if total_failed == 0:
        print()
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print()
        print(f"[FAILURE] {total_failed} test suite(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

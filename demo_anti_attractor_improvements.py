"""
Demonstration of Anti-Attractor Heuristics Improvements

This script demonstrates the improved operator selection heuristics
in the counter-scenario synthesis algorithm.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from anti_attractor import (
    compute_anti_attractor,
    _choose_operator_for_pair,
    explain_signature,
    compare_signatures,
)
from scenario_inversion import TKSExpression


def demo_operator_heuristics():
    """Demonstrate the operator selection heuristics."""
    print("\n" + "="*70)
    print("OPERATOR SELECTION HEURISTICS DEMONSTRATION")
    print("="*70 + "\n")

    test_cases = [
        ("B2", "D2", "Same polarity (both positive)"),
        ("C3", "A3", "Same polarity (both negative)"),
        ("B2", "C3", "Opposite polarity (pos -> neg)"),
        ("D5", "A6", "Opposite polarity (female -> male)"),
        ("B8", "D9", "Cause -> Effect"),
        ("D7", "C7", "Both rhythm/pattern"),
        ("B1", "D4", "Neutral noetics"),
        ("A10", "C5", "Mixed: neutral + positive"),
    ]

    print("Element Pair         | Noetics     | Operator | Rule")
    print("-" * 70)

    for elem1, elem2, description in test_cases:
        op = _choose_operator_for_pair(elem1, elem2)
        n1, n2 = int(elem1[1:]), int(elem2[1:])
        print(f"{elem1} -> {elem2:8} | N{n1:2} -> N{n2:2} | {op:4}     | {description}")


def demo_counter_scenario_synthesis():
    """Demonstrate full counter-scenario synthesis with varied operators."""
    print("\n" + "="*70)
    print("COUNTER-SCENARIO SYNTHESIS WITH OPERATOR VARIETY")
    print("="*70 + "\n")

    # Example 1: Positive mental scenario
    print("Example 1: Positive Mental Belief Scenario")
    print("-" * 70)

    expr1 = TKSExpression(
        elements=["B2", "B2", "D5"],  # Positive belief -> woman
        ops=["->", "+T"],
        foundations=[(1, "A")],  # Unity
        acquisitions=[],
        raw="Positive mental belief causes woman"
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr1, num_elements=3)

    print("\nOriginal Expression:")
    print(f"  Elements: {' '.join(expr1.elements)}")
    print(f"  Operators: {' '.join(expr1.ops)}")
    print(f"  Foundations: {expr1.foundations}")

    print("\n" + explain_signature(orig_sig, "Original"))
    print("\n" + explain_signature(inv_sig, "Inverted"))

    print("\nSynthesized Counter-Scenario:")
    print(f"  Elements: {' '.join(counter.elements)}")
    print(f"  Operators: {' '.join(counter.ops)}")
    print(f"  Foundations: {counter.foundations}")

    # Explain operator choices
    print("\nOperator Selection Reasoning:")
    for i, op in enumerate(counter.ops):
        e1, e2 = counter.elements[i], counter.elements[i+1]
        n1, n2 = int(e1[1:]), int(e2[1:])
        print(f"  {e1} {op} {e2}  (N{n1} -> N{n2})")

    # Example 2: Complex scenario with diverse noetics
    print("\n\nExample 2: Complex Multi-Element Scenario")
    print("-" * 70)

    expr2 = TKSExpression(
        elements=["B8", "D9", "C2", "A3", "D7", "B7"],  # Cause->Effect, pos, neg, rhythm
        ops=["->", "+T", "/T", "o", "+T"],
        foundations=[(2, "B"), (5, "D")],
        acquisitions=[],
        raw=""
    )

    orig_sig2, inv_sig2, counter2 = compute_anti_attractor(expr2, num_elements=4)

    print("\nOriginal Expression:")
    print(f"  Elements: {' '.join(expr2.elements)}")
    print(f"  Operators: {' '.join(expr2.ops)}")

    print("\nSynthesized Counter-Scenario:")
    print(f"  Elements: {' '.join(counter2.elements)}")
    print(f"  Operators: {' '.join(counter2.ops)}")

    print("\nOperator Variety Achieved:")
    unique_ops = set(counter2.ops)
    print(f"  Unique operators used: {unique_ops}")
    print(f"  Operator diversity: {len(unique_ops)}/{len(counter2.ops)} distinct")


def demo_canon_compliance():
    """Demonstrate that all outputs remain canon-compliant."""
    print("\n" + "="*70)
    print("CANON COMPLIANCE VERIFICATION")
    print("="*70 + "\n")

    # Create edge case scenario
    expr = TKSExpression(
        elements=["A1", "B4", "C7", "D10"],  # All self-dual noetics
        ops=["->", "+T", "o"],
        foundations=[(4, "B")],  # Companionship (self-dual foundation)
        acquisitions=[],
        raw=""
    )

    counter = compute_anti_attractor(expr, num_elements=3)[2]

    print("Edge Case: All Self-Dual Noetics (N1, N4, N7, N10)")
    print("-" * 70)

    # Verify canon compliance
    checks = []

    # Check 1: Worlds
    valid_worlds = all(e[0] in "ABCD" for e in counter.elements)
    checks.append(("Worlds in A/B/C/D", valid_worlds))

    # Check 2: Noetics
    valid_noetics = all(1 <= int(e[1:]) <= 10 for e in counter.elements)
    checks.append(("Noetics in 1-10", valid_noetics))

    # Check 3: Operators
    from narrative.constants import ALLOWED_OPS
    valid_ops = all(op in ALLOWED_OPS for op in counter.ops)
    checks.append(("Operators in ALLOWED_OPS", valid_ops))

    # Check 4: Foundations
    valid_founds = all(1 <= fid <= 7 and w in "ABCD"
                       for fid, w in counter.foundations)
    checks.append(("Foundations in 1-7", valid_founds))

    # Check 5: Structure
    valid_structure = len(counter.ops) == len(counter.elements) - 1
    checks.append(("Operator count = elements - 1", valid_structure))

    print("\nCanon Compliance Checks:")
    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} | {check_name}")

    print("\nSynthesized Expression:")
    print(f"  Elements: {' '.join(counter.elements)}")
    print(f"  Operators: {' '.join(counter.ops)}")
    print(f"  Foundations: {counter.foundations}")

    all_passed = all(p for _, p in checks)
    if all_passed:
        print("\n[SUCCESS] All canon compliance checks passed!")
    else:
        print("\n[ERROR] Some checks failed!")


if __name__ == "__main__":
    demo_operator_heuristics()
    demo_counter_scenario_synthesis()
    demo_canon_compliance()

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")

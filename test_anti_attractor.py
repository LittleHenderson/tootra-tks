"""
Test script for anti-attractor synthesis module.

Verifies that all functions work correctly with various test cases.
"""
from anti_attractor import (
    compute_attractor_signature,
    invert_signature,
    synthesize_counter_scenario,
    anti_attractor,
)
from scenario_inversion import TKSExpression, parse_equation


def test_compute_signature():
    """Test attractor signature computation."""
    print("=" * 60)
    print("TEST 1: Compute Attractor Signature")
    print("=" * 60)

    # Test case 1: Simple positive expression
    expr = TKSExpression(
        elements=["B2", "C2", "D2"],
        ops=["->", "+T"],
        foundations=[(1, None), (3, None)],
        acquisitions=[],
        raw="B2 -> C2 +T D2 [F1,F3]"
    )

    sig = compute_attractor_signature(expr)

    print(f"Input: {expr.elements} {expr.ops}")
    print(f"Element counts: {sig.element_counts}")
    print(f"Foundation tags: {sig.foundation_tags}")
    print(f"Polarity: {sig.polarity}")
    print(f"Ops distribution: {sig.ops_distribution}")
    print(f"Dominant world: {sig.dominant_world}")
    print(f"Dominant noetic: {sig.dominant_noetic}")
    print()

    assert sig.element_counts == {("B", 2): 1, ("C", 2): 1, ("D", 2): 1}
    assert sig.foundation_tags == {1, 3}
    assert sig.polarity == 1  # All N2 (positive)
    assert sig.ops_distribution == {"->": 1, "+T": 1}
    print("[PASS] Test 1 passed\n")


def test_invert_signature():
    """Test signature inversion."""
    print("=" * 60)
    print("TEST 2: Invert Signature")
    print("=" * 60)

    # Create a signature
    expr = TKSExpression(
        elements=["B2", "C2", "D2"],
        ops=["->", "+T"],
        foundations=[(1, None)],
        acquisitions=[],
        raw=""
    )

    sig = compute_attractor_signature(expr)
    inv_sig = invert_signature(sig)

    print(f"Original element counts: {sig.element_counts}")
    print(f"Inverted element counts: {inv_sig.element_counts}")
    print(f"Original foundations: {sig.foundation_tags}")
    print(f"Inverted foundations: {inv_sig.foundation_tags}")
    print(f"Original polarity: {sig.polarity}")
    print(f"Inverted polarity: {inv_sig.polarity}")
    print()

    # B2->B3, C2->C3, D2->D3 (noetic inversion: 2<->3)
    # B->C, C->C, D->D (world inversion: B<->C, D->A)
    # Wait, WORLD_OPP: A<->D, B<->C
    # So B->C, C->B, D->A
    assert inv_sig.element_counts == {("C", 3): 1, ("B", 3): 1, ("A", 3): 1}
    assert inv_sig.foundation_tags == {7}  # F1 <-> F7
    assert inv_sig.polarity == -1  # Inverted from +1
    print("[PASS] Test 2 passed\n")


def test_synthesize_counter_scenario():
    """Test counter-scenario synthesis."""
    print("=" * 60)
    print("TEST 3: Synthesize Counter-Scenario")
    print("=" * 60)

    # Create an inverted signature
    expr = TKSExpression(
        elements=["B2", "C2", "D2", "B2"],  # B2 appears twice
        ops=["->", "+T", "->"],
        foundations=[(1, None), (3, None)],
        acquisitions=[],
        raw=""
    )

    sig = compute_attractor_signature(expr)
    inv_sig = invert_signature(sig)
    counter = synthesize_counter_scenario(inv_sig)

    print(f"Inverted signature element counts: {inv_sig.element_counts}")
    print(f"Counter-scenario elements: {counter.elements}")
    print(f"Counter-scenario ops: {counter.ops}")
    print(f"Counter-scenario foundations: {counter.foundations}")
    print()

    # Should have elements (with possible repetitions based on frequency)
    assert len(counter.elements) >= 1
    # Should have len(elements)-1 operators
    assert len(counter.ops) == len(counter.elements) - 1
    # Foundations should be inverted: F1->F7, F3->F5
    foundation_ids = {fid for fid, _ in counter.foundations}
    assert foundation_ids == {5, 7}
    print("[PASS] Test 3 passed\n")


def test_anti_attractor_full():
    """Test full anti-attractor pipeline."""
    print("=" * 60)
    print("TEST 4: Full Anti-Attractor Pipeline")
    print("=" * 60)

    # Test with parse_equation
    expr = parse_equation("B2 -> C2 +T D2")

    print(f"Original expression: {expr.elements} {expr.ops}")

    counter = anti_attractor(expr)

    print(f"Counter-scenario elements: {counter.elements}")
    print(f"Counter-scenario ops: {counter.ops}")
    print(f"Counter-scenario foundations: {counter.foundations}")
    print()

    # Verify counter-scenario has valid elements
    for elem in counter.elements:
        world = elem[0]
        noetic = int(elem[1:])
        assert world in {"A", "B", "C", "D"}
        assert 1 <= noetic <= 10

    # Verify operators are valid
    for op in counter.ops:
        assert op in {"->", "<-", "+T", "-T", "*T", "/T", "o", "+", "-"}

    print("[PASS] Test 4 passed\n")


def test_negative_polarity():
    """Test with negative polarity expression."""
    print("=" * 60)
    print("TEST 5: Negative Polarity Expression")
    print("=" * 60)

    # Create expression with negative noetics
    expr = TKSExpression(
        elements=["B3", "C3", "D3"],  # All N3 (negative)
        ops=["->", "->"],
        foundations=[(7, None)],  # F7 = Lust
        acquisitions=[],
        raw=""
    )

    sig = compute_attractor_signature(expr)
    print(f"Original polarity: {sig.polarity}")
    assert sig.polarity == -1  # Should be negative

    inv_sig = invert_signature(sig)
    print(f"Inverted polarity: {inv_sig.polarity}")
    assert inv_sig.polarity == 1  # Should be positive

    counter = anti_attractor(expr)
    print(f"Counter-scenario elements: {counter.elements}")

    # All elements should be positive noetics (N2)
    for elem in counter.elements:
        noetic = int(elem[1:])
        # After inversion, N3->N2
        assert noetic == 2

    print("[PASS] Test 5 passed\n")


def test_mixed_worlds():
    """Test with mixed world elements."""
    print("=" * 60)
    print("TEST 6: Mixed World Elements")
    print("=" * 60)

    expr = TKSExpression(
        elements=["A2", "B5", "C8", "D3"],
        ops=["->", "+T", "->"],
        foundations=[(4, None)],  # F4 = Companionship (self-dual)
        acquisitions=[],
        raw=""
    )

    sig = compute_attractor_signature(expr)
    print(f"Element counts: {sig.element_counts}")
    print(f"Dominant world: {sig.dominant_world}")

    inv_sig = invert_signature(sig)
    print(f"Inverted element counts: {inv_sig.element_counts}")

    # Verify inversions:
    # A2 -> D3 (A<->D, 2<->3)
    # B5 -> C6 (B<->C, 5<->6)
    # C8 -> B9 (C<->B, 8<->9)
    # D3 -> A2 (D<->A, 3<->2)
    expected = {("D", 3): 1, ("C", 6): 1, ("B", 9): 1, ("A", 2): 1}
    assert inv_sig.element_counts == expected

    counter = anti_attractor(expr)
    print(f"Counter-scenario: {counter.elements}")

    print("[PASS] Test 6 passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ANTI-ATTRACTOR SYNTHESIS MODULE TEST SUITE")
    print("=" * 60 + "\n")

    test_compute_signature()
    test_invert_signature()
    test_synthesize_counter_scenario()
    test_anti_attractor_full()
    test_negative_polarity()
    test_mixed_worlds()

    print("=" * 60)
    print("ALL TESTS PASSED [SUCCESS]")
    print("=" * 60)

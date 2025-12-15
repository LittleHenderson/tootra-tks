"""
Tests for TKS Anti-Attractor Synthesis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anti_attractor import (
    AttractorSignature,
    compute_attractor_signature,
    invert_signature,
    synthesize_counter_scenario,
    anti_attractor,
    compute_anti_attractor,
)
from scenario_inversion import TKSExpression, parse_equation
from narrative.constants import ALLOWED_OPS, WORLD_LETTERS


# =============================================================================
# TEST COMPUTE_ATTRACTOR_SIGNATURE
# =============================================================================

def test_compute_signature_simple():
    """Test signature computation for simple expression."""
    expr = parse_equation("C3 -> D3")
    sig = compute_attractor_signature(expr)

    assert ("C", 3) in sig.element_counts
    assert ("D", 3) in sig.element_counts
    assert sig.polarity < 0  # Both are negative noetics (N3)
    assert "->" in sig.ops_distribution
    print("[PASS] test_compute_signature_simple")


def test_compute_signature_with_repetitions():
    """Test signature computation with repeated elements."""
    expr = TKSExpression(
        elements=["B2", "B2", "D5", "C3"],
        ops=["->", "+T", "->"],
        foundations=[],
        acquisitions=[],
        raw="B2 -> B2 +T D5 -> C3"
    )
    sig = compute_attractor_signature(expr)

    # B2 appears twice
    assert sig.element_counts[("B", 2)] == 2
    assert sig.element_counts[("D", 5)] == 1
    assert sig.element_counts[("C", 3)] == 1

    # Polarity: B2(+), B2(+), D5(+), C3(-) → positive wins
    assert sig.polarity == 1

    # Dominant world should be B (appears 2x)
    assert sig.dominant_world == "B"

    # Dominant noetic should be 2 (appears 2x)
    assert sig.dominant_noetic == 2

    print("[PASS] test_compute_signature_with_repetitions passed")


def test_compute_signature_with_foundations():
    """Test signature computation includes foundation tags."""
    expr = TKSExpression(
        elements=["A2", "D5"],
        ops=["->"],
        foundations=[(1, "A"), (7, "D")],  # Unity and Lust
        acquisitions=[],
        raw=""
    )
    sig = compute_attractor_signature(expr)

    assert 1 in sig.foundation_tags
    assert 7 in sig.foundation_tags
    assert sig.polarity == 1  # Both A2 and D5 are positive noetics

    print("[PASS] test_compute_signature_with_foundations passed")


def test_compute_signature_neutral_polarity():
    """Test signature with neutral noetics results in neutral polarity."""
    expr = TKSExpression(
        elements=["B1", "D4", "C7"],  # N1, N4, N7 are all self-dual
        ops=["->", "+T"],
        foundations=[],
        acquisitions=[],
        raw=""
    )
    sig = compute_attractor_signature(expr)

    # No positive or negative noetics, should be neutral
    assert sig.polarity == 0

    print("[PASS] test_compute_signature_neutral_polarity passed")


def test_compute_signature_balanced_polarity():
    """Test signature with balanced positive/negative noetics."""
    expr = TKSExpression(
        elements=["B2", "C3", "D5", "A6"],  # 2 positive (2,5), 2 negative (3,6)
        ops=["->", "+T", "->"],
        foundations=[],
        acquisitions=[],
        raw=""
    )
    sig = compute_attractor_signature(expr)

    # Equal positive and negative should result in neutral
    assert sig.polarity == 0

    print("[PASS] test_compute_signature_balanced_polarity passed")


# =============================================================================
# TEST INVERT_SIGNATURE
# =============================================================================

def test_invert_signature_involutive():
    """Test that inverting twice returns original signature."""
    expr = parse_equation("B5 +T D6")
    sig = compute_attractor_signature(expr)
    inv_sig = invert_signature(sig)
    double_inv = invert_signature(inv_sig)

    # Element counts should match after double inversion
    assert sig.element_counts == double_inv.element_counts
    assert sig.polarity == double_inv.polarity

    print("[PASS] test_invert_signature_involutive passed")


def test_invert_signature_world_opposites():
    """Test world inversions follow canonical oppositions."""
    sig = AttractorSignature(
        element_counts={("A", 1): 1, ("B", 2): 1, ("C", 3): 1, ("D", 4): 1},
        foundation_tags=set(),
        polarity=0,
        ops_distribution={},
        dominant_world="A",
        dominant_noetic=1
    )
    inv_sig = invert_signature(sig)

    # A ↔ D, B ↔ C
    assert ("D", 1) in inv_sig.element_counts  # A → D
    assert ("C", 3) in inv_sig.element_counts  # B → C (N2→N3)
    assert ("B", 2) in inv_sig.element_counts  # C → B (N3→N2)
    assert ("A", 4) in inv_sig.element_counts  # D → A
    assert inv_sig.dominant_world == "D"       # A → D

    print("[PASS] test_invert_signature_world_opposites passed")


def test_invert_signature_noetic_opposites():
    """Test noetic inversions follow canonical pairs."""
    sig = AttractorSignature(
        element_counts={
            ("A", 2): 1,  # Positive → Negative
            ("A", 3): 1,  # Negative → Positive
            ("A", 5): 1,  # Female → Male
            ("A", 6): 1,  # Male → Female
            ("A", 8): 1,  # Cause → Effect
            ("A", 9): 1,  # Effect → Cause
        },
        foundation_tags=set(),
        polarity=0,
        ops_distribution={},
        dominant_world="A",
        dominant_noetic=2
    )
    inv_sig = invert_signature(sig)

    # Check involutive pairs
    assert ("D", 3) in inv_sig.element_counts  # A2 → D3
    assert ("D", 2) in inv_sig.element_counts  # A3 → D2
    assert ("D", 6) in inv_sig.element_counts  # A5 → D6
    assert ("D", 5) in inv_sig.element_counts  # A6 → D5
    assert ("D", 9) in inv_sig.element_counts  # A8 → D9
    assert ("D", 8) in inv_sig.element_counts  # A9 → D8

    print("[PASS] test_invert_signature_noetic_opposites passed")


def test_invert_signature_self_dual_noetics():
    """Test self-dual noetics (1,4,7,10) map to themselves."""
    sig = AttractorSignature(
        element_counts={
            ("B", 1): 1,   # Mind (self-dual)
            ("B", 4): 1,   # Vibration (self-dual)
            ("B", 7): 1,   # Rhythm (self-dual)
            ("B", 10): 1,  # Idea (self-dual)
        },
        foundation_tags=set(),
        polarity=0,
        ops_distribution={},
        dominant_world="B",
        dominant_noetic=1
    )
    inv_sig = invert_signature(sig)

    # Worlds invert but noetics stay same for self-duals
    assert ("C", 1) in inv_sig.element_counts   # B1 → C1
    assert ("C", 4) in inv_sig.element_counts   # B4 → C4
    assert ("C", 7) in inv_sig.element_counts   # B7 → C7
    assert ("C", 10) in inv_sig.element_counts  # B10 → C10

    print("[PASS] test_invert_signature_self_dual_noetics passed")


def test_invert_signature_foundations():
    """Test foundation inversions follow canonical oppositions."""
    sig = AttractorSignature(
        element_counts={("B", 2): 1},
        foundation_tags={1, 2, 3, 4, 5, 6, 7},  # All foundations
        polarity=1,
        ops_distribution={},
        dominant_world="B",
        dominant_noetic=2
    )
    inv_sig = invert_signature(sig)

    # Foundation opposites: 1↔7, 2↔6, 3↔5, 4→4
    # After inversion, should get: {7, 6, 5, 4, 3, 2, 1} = same set
    assert inv_sig.foundation_tags == {1, 2, 3, 4, 5, 6, 7}

    print("[PASS] test_invert_signature_foundations passed")


def test_invert_signature_polarity():
    """Test polarity inversion."""
    sig_pos = AttractorSignature(
        element_counts={("B", 2): 1},
        foundation_tags=set(),
        polarity=1,
        ops_distribution={},
        dominant_world="B",
        dominant_noetic=2
    )
    sig_neg = AttractorSignature(
        element_counts={("B", 3): 1},
        foundation_tags=set(),
        polarity=-1,
        ops_distribution={},
        dominant_world="B",
        dominant_noetic=3
    )
    sig_neutral = AttractorSignature(
        element_counts={("B", 1): 1},
        foundation_tags=set(),
        polarity=0,
        ops_distribution={},
        dominant_world="B",
        dominant_noetic=1
    )

    assert invert_signature(sig_pos).polarity == -1
    assert invert_signature(sig_neg).polarity == 1
    assert invert_signature(sig_neutral).polarity == 0

    print("[PASS] test_invert_signature_polarity passed")


# =============================================================================
# TEST SYNTHESIZE_COUNTER_SCENARIO
# =============================================================================

def test_synthesize_valid_expression():
    """Test synthesized expression has valid elements and ops."""
    expr = parse_equation("C3 -> D3")
    sig = compute_attractor_signature(expr)
    inv_sig = invert_signature(sig)
    counter = synthesize_counter_scenario(inv_sig)

    # All elements should have valid world/noetic
    for elem in counter.elements:
        world = elem[0]
        noetic = int(elem[1:])
        assert world in WORLD_LETTERS
        assert 1 <= noetic <= 10

    # All ops should be in ALLOWED_OPS
    for op in counter.ops:
        assert op in ALLOWED_OPS

    print("[PASS] test_synthesize_valid_expression passed")


def test_synthesize_uses_causal_operators():
    """Test synthesis with prefer_causal=True uses contextual heuristics."""
    inv_sig = AttractorSignature(
        element_counts={("B", 2): 2, ("C", 3): 1},
        foundation_tags=set(),
        polarity=1,
        ops_distribution={"->": 1},
        dominant_world="B",
        dominant_noetic=2
    )
    counter = synthesize_counter_scenario(inv_sig, num_elements=3, prefer_causal=True)

    # Elements will be: B2, B2, C3 (positive, positive, negative)
    # Expected operators: B2->B2 (same polarity → *T), B2->C3 (pos vs neg → /T)
    assert len(counter.ops) == 2
    assert counter.ops[0] == "*T"  # B2 to B2: same polarity
    assert counter.ops[1] == "/T"  # B2 to C3: opposite polarity

    print("[PASS] test_synthesize_uses_causal_operators passed")


def test_synthesize_respects_num_elements():
    """Test synthesis respects num_elements parameter."""
    inv_sig = AttractorSignature(
        element_counts={
            ("B", 2): 3,
            ("C", 3): 2,
            ("D", 5): 1,
            ("A", 6): 1
        },
        foundation_tags=set(),
        polarity=1,
        ops_distribution={},
        dominant_world="B",
        dominant_noetic=2
    )

    # Request 2 elements
    counter = synthesize_counter_scenario(inv_sig, num_elements=2, prefer_causal=True)

    # Should synthesize with top 2 elements (B2 and C3)
    # But may include repetitions, so check we don't exceed significantly
    assert len(counter.elements) <= 4  # max 2 elements × 2 repetitions

    print("[PASS] test_synthesize_respects_num_elements passed")


def test_synthesize_attaches_foundations():
    """Test synthesis attaches inverted foundations."""
    inv_sig = AttractorSignature(
        element_counts={("B", 2): 1},
        foundation_tags={7, 6},  # Lust and Material (inverted from Unity and Wisdom)
        polarity=1,
        ops_distribution={},
        dominant_world="B",
        dominant_noetic=2
    )
    counter = synthesize_counter_scenario(inv_sig)

    # Should attach foundations
    assert len(counter.foundations) == 2
    fids = {fid for fid, _ in counter.foundations}
    assert 7 in fids
    assert 6 in fids

    # All subfoundation worlds should be dominant_world
    for _, world in counter.foundations:
        assert world == "B"

    print("[PASS] test_synthesize_attaches_foundations passed")


def test_synthesize_element_multiplicity():
    """Test synthesis preserves element multiplicity (up to limit)."""
    inv_sig = AttractorSignature(
        element_counts={("B", 2): 5},  # High count
        foundation_tags=set(),
        polarity=1,
        ops_distribution={},
        dominant_world="B",
        dominant_noetic=2
    )
    counter = synthesize_counter_scenario(inv_sig, num_elements=1)

    # Should include B2 with repetition (capped at 2)
    assert counter.elements.count("B2") <= 2
    assert "B2" in counter.elements

    print("[PASS] test_synthesize_element_multiplicity passed")


# =============================================================================
# TEST ANTI_ATTRACTOR PIPELINE
# =============================================================================

def test_anti_attractor_pipeline():
    """Test full anti-attractor synthesis pipeline."""
    expr = parse_equation("C3 -> D3")
    counter = anti_attractor(expr)

    assert len(counter.elements) > 0

    # Counter should have inverted worlds
    # C3 → B2, D3 → A2
    worlds = {e[0] for e in counter.elements}
    assert "B" in worlds or "A" in worlds

    # Noetics should be inverted (3 → 2)
    noetics = {int(e[1:]) for e in counter.elements}
    assert 2 in noetics

    print("[PASS] test_anti_attractor_pipeline passed")


def test_anti_attractor_positive_to_negative():
    """Test anti-attractor inverts positive scenario to negative."""
    # Positive mental belief
    expr = TKSExpression(
        elements=["B2", "B2", "D2"],  # Positive mental, positive physical
        ops=["->", "+T"],
        foundations=[(1, "A")],  # Unity
        acquisitions=[],
        raw=""
    )
    counter = anti_attractor(expr)

    # Should produce negative scenario
    # B2 → C3, D2 → A3
    noetics = [int(e[1:]) for e in counter.elements]
    assert 3 in noetics  # Should have negative noetics

    # Foundation should invert: Unity(1) → Lust(7)
    fids = [fid for fid, _ in counter.foundations]
    assert 7 in fids

    print("[PASS] test_anti_attractor_positive_to_negative passed")


def test_anti_attractor_complex_scenario():
    """Test anti-attractor on complex multi-element scenario."""
    expr = TKSExpression(
        elements=["B5", "C2", "D6", "A8"],  # Female mental, positive emotion, male physical, spiritual cause
        ops=["->", "+T", "->"],
        foundations=[(2, "B"), (4, "C")],  # Wisdom, Companionship
        acquisitions=[],
        raw=""
    )
    counter = anti_attractor(expr)

    # Should invert all elements
    # B5 → C6, C2 → B3, D6 → A5, A8 → D9
    assert len(counter.elements) > 0

    # Foundations should invert: Wisdom(2) → Material(6), Companionship(4) → Companionship(4)
    fids = {fid for fid, _ in counter.foundations}
    assert 6 in fids  # Material
    assert 4 in fids  # Companionship (self-dual)

    print("[PASS] test_anti_attractor_complex_scenario passed")


# =============================================================================
# TEST CANON VALIDITY
# =============================================================================

def test_anti_attractor_canon_valid():
    """Test that anti-attractor output respects canon constraints."""
    expr = TKSExpression(
        elements=["C3", "D3", "C5"],
        ops=["->", "+T"],
        foundations=[(3, "C"), (5, "D")],
        acquisitions=[],
        raw=""
    )
    counter = anti_attractor(expr)

    # Verify worlds are canonical (A, B, C, D)
    for elem in counter.elements:
        assert elem[0] in "ABCD"

    # Verify noetics are canonical (1-10)
    for elem in counter.elements:
        noetic = int(elem[1:])
        assert 1 <= noetic <= 10

    # Verify ops are allowed
    for op in counter.ops:
        assert op in ALLOWED_OPS

    # Verify foundations are canonical (1-7)
    for fid, world in counter.foundations:
        assert 1 <= fid <= 7
        assert world in "ABCD"

    print("[PASS] test_anti_attractor_canon_valid passed")


def test_anti_attractor_preserves_structure():
    """Test anti-attractor preserves structural properties."""
    expr = TKSExpression(
        elements=["B2", "C3", "D5"],
        ops=["+T", "->"],
        foundations=[(1, "A")],
        acquisitions=[],
        raw=""
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr, num_elements=3)

    # Operator count should be preserved (or close)
    assert len(counter.ops) == len(counter.elements) - 1

    # Foundation count should match
    assert len(counter.foundations) == len(expr.foundations)

    print("[PASS] test_anti_attractor_preserves_structure passed")


# =============================================================================
# TEST EDGE CASES
# =============================================================================

def test_single_element_expression():
    """Test anti-attractor on single element expression."""
    expr = TKSExpression(
        elements=["B2"],
        ops=[],
        foundations=[],
        acquisitions=[],
        raw=""
    )
    counter = anti_attractor(expr)

    # Should produce at least one element
    assert len(counter.elements) >= 1

    # Should invert: B2 → C3
    assert "C3" in counter.elements

    print("[PASS] test_single_element_expression passed")


def test_empty_foundations():
    """Test anti-attractor with no foundations."""
    expr = TKSExpression(
        elements=["B2", "D5"],
        ops=["->"],
        foundations=[],  # No foundations
        acquisitions=[],
        raw=""
    )
    counter = anti_attractor(expr)

    # Should produce valid expression
    assert len(counter.elements) > 0
    assert len(counter.foundations) == 0

    print("[PASS] test_empty_foundations passed")


def test_self_dual_only_expression():
    """Test anti-attractor on expression with only self-dual noetics."""
    expr = TKSExpression(
        elements=["B1", "D4", "C7", "A10"],  # All self-dual
        ops=["->", "+T", "->"],
        foundations=[(4, "B")],  # Companionship is also self-dual
        acquisitions=[],
        raw=""
    )
    counter = anti_attractor(expr)

    # Noetics should remain same but worlds should invert
    noetics = {int(e[1:]) for e in counter.elements}
    assert 1 in noetics or 4 in noetics or 7 in noetics or 10 in noetics

    # Foundation should remain: 4 → 4
    fids = [fid for fid, _ in counter.foundations]
    assert 4 in fids

    print("[PASS] test_self_dual_only_expression passed")


# =============================================================================
# REGRESSION TESTS - Operator Variety and Canon Validity
# =============================================================================

def test_operator_variety_in_synthesis():
    """
    Regression test: Verify synthesized expressions use varied operators.

    This test ensures the improved heuristics generate different operators
    (*T, /T, +T, ->, o) based on element characteristics, not just "->".
    """
    # Create scenario with diverse noetics to trigger different operators
    expr = TKSExpression(
        elements=["B2", "D5", "C3", "A8", "D9", "B7", "C7"],
        ops=["->", "+T", "->", "*T", "o", "->"],
        foundations=[(2, "B")],
        acquisitions=[],
        raw=""
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr, num_elements=5)

    # Verify we get operator variety (not all the same)
    unique_ops = set(counter.ops)
    assert len(unique_ops) > 1, f"Expected operator variety, got only: {unique_ops}"

    # All operators must be from ALLOWED_OPS
    for op in counter.ops:
        assert op in ALLOWED_OPS, f"Invalid operator: {op}"

    print("[PASS] test_operator_variety_in_synthesis passed")


def test_counter_scenario_differs_from_original():
    """
    Regression test: Verify counter-scenarios differ in polarity and world.

    This test ensures the anti-attractor properly inverts the original scenario,
    creating a genuinely opposite pattern in TKS space.
    """
    # Original: Positive mental scenario (B2 - positive belief)
    expr = TKSExpression(
        elements=["B2", "B2", "D5"],  # Mental positive, physical female
        ops=["->", "+T"],
        foundations=[(1, "A")],  # Unity
        acquisitions=[],
        raw=""
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr, num_elements=3)

    # Verify polarity inverted
    assert orig_sig.polarity == 1, "Original should be positive"
    assert inv_sig.polarity == -1, "Inverted should be negative"

    # Verify dominant world inverted (B -> C)
    assert orig_sig.dominant_world == "B", "Original dominant should be Mental"
    assert inv_sig.dominant_world == "C", "Inverted dominant should be Emotional"

    # Verify dominant noetic inverted (2 -> 3)
    assert orig_sig.dominant_noetic == 2, "Original noetic should be Positive"
    assert inv_sig.dominant_noetic == 3, "Inverted noetic should be Negative"

    # Verify counter-scenario elements reflect inversion
    # Should have C3 (emotional negative) and A6 (spiritual male)
    counter_noetics = {int(e[1:]) for e in counter.elements}
    assert 3 in counter_noetics, "Counter should include negative noetics"

    # Verify foundation inverted: Unity(1) -> Lust(7)
    counter_fids = {fid for fid, _ in counter.foundations}
    assert 7 in counter_fids, "Unity should invert to Lust"

    print("[PASS] test_counter_scenario_differs_from_original passed")


def test_counter_scenario_canon_validity_comprehensive():
    """
    Regression test: Comprehensive canon validity check for synthesized expressions.

    This test ensures all outputs strictly adhere to TKS canon constraints
    regardless of input complexity.
    """
    # Create complex scenario with many elements and foundations
    expr = TKSExpression(
        elements=["A2", "B5", "C3", "D6", "A8", "D9", "B1", "C7"],
        ops=["->", "+T", "*T", "/T", "->", "o", "+T"],
        foundations=[(1, "A"), (2, "B"), (3, "C"), (5, "D")],
        acquisitions=[],
        raw=""
    )

    counter = anti_attractor(expr, num_elements=5)

    # 1. Verify worlds are canonical (A, B, C, D only)
    for elem in counter.elements:
        world = elem[0]
        assert world in "ABCD", f"Invalid world: {world}"

    # 2. Verify noetics are canonical (1-10 only)
    for elem in counter.elements:
        noetic = int(elem[1:])
        assert 1 <= noetic <= 10, f"Invalid noetic: {noetic}"

    # 3. Verify operators are from ALLOWED_OPS
    for op in counter.ops:
        assert op in ALLOWED_OPS, f"Invalid operator: {op}"

    # 4. Verify foundations are canonical (1-7 only)
    for fid, world in counter.foundations:
        assert 1 <= fid <= 7, f"Invalid foundation: {fid}"
        assert world in "ABCD", f"Invalid subfoundation world: {world}"

    # 5. Verify operator count matches element count
    assert len(counter.ops) == len(counter.elements) - 1, \
        "Operator count must be elements - 1"

    # 6. Verify at least one element exists
    assert len(counter.elements) >= 1, "Must have at least one element"

    print("[PASS] test_counter_scenario_canon_validity_comprehensive passed")


# =============================================================================
# NEW REGRESSION TESTS - Refined Operator Heuristics
# =============================================================================

def test_operator_choice_for_specific_pairs():
    """
    Regression test: Verify expected operator choices for specific element pairs.

    This test validates the refined heuristics in _choose_operator_for_pair()
    to ensure deterministic and semantically appropriate operator selection.
    """
    from anti_attractor import _choose_operator_for_pair

    # Test Cause→Effect (Rule 1: highest priority)
    assert _choose_operator_for_pair("B8", "D9") == "->", \
        "Cause→Effect should use -> operator"
    assert _choose_operator_for_pair("A8", "C9") == "->", \
        "Cause→Effect should use -> operator (any world)"

    # Test Effect→Cause (Rule 2: reverse causal)
    assert _choose_operator_for_pair("D9", "B8") == "<-", \
        "Effect→Cause should use <- operator"
    assert _choose_operator_for_pair("C9", "A8") == "<-", \
        "Effect→Cause should use <- operator (any world)"

    # Test Rhythm elements (Rule 3: sequential composition)
    assert _choose_operator_for_pair("D7", "C3") == "o", \
        "Rhythm→any should use o operator"
    assert _choose_operator_for_pair("B2", "D7") == "o", \
        "any→Rhythm should use o operator"
    assert _choose_operator_for_pair("B7", "D7") == "o", \
        "Rhythm→Rhythm should use o operator"

    # Test same polarity (Rule 4: intensify)
    assert _choose_operator_for_pair("B2", "D2") == "*T", \
        "Positive→Positive should use *T operator"
    assert _choose_operator_for_pair("C3", "A3") == "*T", \
        "Negative→Negative should use *T operator"
    assert _choose_operator_for_pair("B5", "D5") == "*T", \
        "Female→Female should use *T operator"

    # Test opposite polarity (Rule 5: conflict)
    assert _choose_operator_for_pair("B2", "C3") == "/T", \
        "Positive→Negative should use /T operator"
    assert _choose_operator_for_pair("D3", "A2") == "/T", \
        "Negative→Positive should use /T operator"

    # Test Female→Male (Rule 6: directional flow)
    assert _choose_operator_for_pair("B5", "D6") == "->", \
        "Female→Male should use -> operator"
    assert _choose_operator_for_pair("C5", "A6") == "->", \
        "Female→Male should use -> operator (any world)"

    # Test Male→Female (Rule 7: reverse flow)
    assert _choose_operator_for_pair("D6", "B5") == "<-", \
        "Male→Female should use <- operator"
    assert _choose_operator_for_pair("A6", "C5") == "<-", \
        "Male→Female should use <- operator (any world)"

    # Test both neutral (Rule 8: combine)
    assert _choose_operator_for_pair("B1", "D4") == "+T", \
        "Mind→Vibration should use +T operator"
    assert _choose_operator_for_pair("A10", "C1") == "+T", \
        "Idea→Mind should use +T operator"

    # Test one neutral, one polar (Rule 9: sequence)
    assert _choose_operator_for_pair("B1", "C3") == "o", \
        "Mind→Negative should use o operator"
    assert _choose_operator_for_pair("D2", "A4") == "o", \
        "Positive→Vibration should use o operator"

    print("[PASS] test_operator_choice_for_specific_pairs passed")


def test_polarity_shift_positive_to_negative():
    """
    Regression test: Verify polarity shifts from positive to negative scenarios.

    This test ensures anti-attractor properly inverts positive scenarios
    into negative ones, with correct element inversions and operator choices.
    """
    # Create strongly positive scenario (Mental B2, Spiritual A2, Physical D2)
    expr = TKSExpression(
        elements=["B2", "B2", "A2", "D2"],  # All positive noetics
        ops=["*T", "->", "+T"],
        foundations=[(1, "A"), (3, "D")],  # Unity, Life
        acquisitions=[],
        raw=""
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr, num_elements=4)

    # 1. Verify original signature is positive
    assert orig_sig.polarity == 1, "Original should be positive polarity"
    assert orig_sig.dominant_noetic == 2, "Original should have Positive (N2) dominant"

    # 2. Verify inverted signature is negative
    assert inv_sig.polarity == -1, "Inverted should be negative polarity"
    assert inv_sig.dominant_noetic == 3, "Inverted should have Negative (N3) dominant"

    # 3. Verify element inversions: B2→C3, A2→D3, D2→A3
    counter_noetics = [int(e[1:]) for e in counter.elements]
    assert 3 in counter_noetics, "Counter should contain negative noetics (N3)"
    assert 2 not in counter_noetics, "Counter should NOT contain positive noetics (N2)"

    # 4. Verify world inversions: B→C (Mental→Emotional), A→D (Spiritual→Physical)
    counter_worlds = [e[0] for e in counter.elements]
    assert "C" in counter_worlds or "D" in counter_worlds or "A" in counter_worlds, \
        "Counter should have inverted worlds"

    # 5. Verify foundation inversions: Unity(1)→Lust(7), Life(3)→Power(5)
    counter_fids = {fid for fid, _ in counter.foundations}
    assert 7 in counter_fids, "Unity(1) should invert to Lust(7)"
    assert 5 in counter_fids, "Life(3) should invert to Power(5)"
    assert 1 not in counter_fids, "Original Unity(1) should not remain"
    assert 3 not in counter_fids, "Original Life(3) should not remain"

    print("[PASS] test_polarity_shift_positive_to_negative passed")


def test_world_shift_mental_to_emotional():
    """
    Regression test: Verify world shifts from Mental (B) to Emotional (C).

    This test ensures anti-attractor properly inverts Mental scenarios
    into Emotional ones, preserving noetic structure while shifting worlds.
    """
    # Create Mental scenario (B-world dominant)
    expr = TKSExpression(
        elements=["B2", "B5", "B8", "B1"],  # Mental world, varied noetics
        ops=["->", "+T", "->"],
        foundations=[(2, "B"), (4, "B")],  # Wisdom and Companionship in Mental
        acquisitions=[],
        raw=""
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr, num_elements=4)

    # 1. Verify original is Mental-dominant
    assert orig_sig.dominant_world == "B", "Original should be Mental (B) dominant"

    # 2. Verify inverted is Emotional-dominant
    assert inv_sig.dominant_world == "C", "Inverted should be Emotional (C) dominant"

    # 3. Verify counter-scenario has C-world elements
    counter_worlds = [e[0] for e in counter.elements]
    assert "C" in counter_worlds, "Counter should contain Emotional (C) elements"

    # 4. Verify noetic inversions preserved structure:
    #    B2→C3, B5→C6, B8→C9, B1→C1
    counter_elements = set(counter.elements)
    expected_elements = {"C3", "C6", "C9", "C1"}
    # At least some of these should be present
    assert any(e in counter_elements for e in expected_elements), \
        f"Counter should contain expected C-world inversions, got: {counter_elements}"

    # 5. Verify foundations shift to C-world
    for fid, world in counter.foundations:
        assert world == "C", f"All foundations should shift to C-world, got {world}"

    # 6. Verify foundation inversions: Wisdom(2)→Material(6), Companionship(4)→Companionship(4)
    counter_fids = {fid for fid, _ in counter.foundations}
    assert 6 in counter_fids, "Wisdom(2) should invert to Material(6)"
    assert 4 in counter_fids, "Companionship(4) should remain (self-dual)"

    print("[PASS] test_world_shift_mental_to_emotional passed")


def test_involutive_property_double_inversion():
    """
    Regression test: Verify that double inversion returns to original signature.

    This test ensures the involution property: invert(invert(sig)) ≈ sig.
    The anti-attractor of an anti-attractor should approximate the original.
    """
    # Create test scenario
    expr = TKSExpression(
        elements=["B2", "D5", "C3"],
        ops=["->", "+T"],
        foundations=[(1, "A"), (3, "C")],
        acquisitions=[],
        raw=""
    )

    # First inversion
    sig1 = compute_attractor_signature(expr)
    inv_sig1 = invert_signature(sig1)

    # Second inversion (should return to original)
    inv_sig2 = invert_signature(inv_sig1)

    # Verify element counts match
    assert sig1.element_counts == inv_sig2.element_counts, \
        "Double inversion should restore element counts"

    # Verify polarity matches
    assert sig1.polarity == inv_sig2.polarity, \
        "Double inversion should restore polarity"

    # Verify dominant world and noetic match
    assert sig1.dominant_world == inv_sig2.dominant_world, \
        "Double inversion should restore dominant world"
    assert sig1.dominant_noetic == inv_sig2.dominant_noetic, \
        "Double inversion should restore dominant noetic"

    # Verify foundation tags match
    assert sig1.foundation_tags == inv_sig2.foundation_tags, \
        "Double inversion should restore foundation tags"

    print("[PASS] test_involutive_property_double_inversion passed")


# =============================================================================
# SEMANTIC REGRESSION TESTS - Agent 3 Tasks
# =============================================================================

def test_foundation_flip_unity_to_lust():
    """
    Semantic regression test: Foundation flip F1 (Unity) → F7 (Lust).

    This test verifies that anti-attractor properly inverts foundation tags
    according to canonical mappings. Unity (F1) should invert to Lust (F7).
    """
    # Create expression with Unity foundation (F1) in Spiritual world (A)
    expr = TKSExpression(
        elements=["A2", "B2", "D5"],  # Spiritual alignment, mental positive, physical female
        ops=["->", "+T"],
        foundations=[(1, "A")],  # Unity in Spiritual world
        acquisitions=[],
        raw=""
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr, num_elements=3)

    # 1. Verify original has Unity (F1)
    assert 1 in orig_sig.foundation_tags, "Original should have Unity (F1)"

    # 2. Verify inverted has Lust (F7) - opposite of Unity
    assert 7 in inv_sig.foundation_tags, "Inverted should have Lust (F7), opposite of Unity"
    assert 1 not in inv_sig.foundation_tags, "Inverted should NOT have Unity (F1)"

    # 3. Verify counter-scenario foundations
    counter_fids = {fid for fid, _ in counter.foundations}
    assert 7 in counter_fids, "Counter should contain Lust (F7)"
    assert 1 not in counter_fids, "Counter should NOT contain Unity (F1)"

    # 4. Verify all elements and ops are canonical
    for elem in counter.elements:
        assert elem[0] in "ABCD", f"Invalid world in {elem}"
        assert 1 <= int(elem[1:]) <= 10, f"Invalid noetic in {elem}"

    for op in counter.ops:
        assert op in ALLOWED_OPS, f"Invalid operator: {op}"

    print("[PASS] test_foundation_flip_unity_to_lust passed")
    print(f"  Original foundations: {orig_sig.foundation_tags}")
    print(f"  Inverted foundations: {inv_sig.foundation_tags}")
    print(f"  Counter foundations: {[(fid, w) for fid, w in counter.foundations]}")


def test_subfoundation_tag_preservation():
    """
    Semantic regression test: Subfoundation tag preservation and structural integrity.

    This test verifies that subfoundation tags (foundation + world combinations)
    are properly inverted while maintaining structural coherence. For example,
    Foundation 4 (Companionship) in world C (Emotional) should remain F4 (self-dual)
    but shift to inverted world.
    """
    # Create expression with Companionship (F4) in Emotional world (C)
    expr = TKSExpression(
        elements=["C2", "C5", "D6"],  # Emotional joy, emotional openness, physical male
        ops=["->", "+T"],
        foundations=[(4, "C")],  # Companionship in Emotional world (emotional relationship)
        acquisitions=[],
        raw=""
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr, num_elements=3)

    # 1. Verify original has Companionship (F4) - self-dual foundation
    assert 4 in orig_sig.foundation_tags, "Original should have Companionship (F4)"

    # 2. Verify inverted still has Companionship (F4) - self-dual doesn't change
    assert 4 in inv_sig.foundation_tags, "Inverted should still have Companionship (F4) - self-dual"

    # 3. Verify counter-scenario foundations exist
    counter_fids = {fid for fid, _ in counter.foundations}
    assert 4 in counter_fids, "Counter should contain Companionship (F4)"

    # 4. Verify subfoundation world shifted (C → B, Emotional → Mental)
    # Original dominant world should be C, inverted should be B
    assert orig_sig.dominant_world == "C", "Original dominant should be Emotional (C)"
    assert inv_sig.dominant_world == "B", "Inverted dominant should be Mental (B), opposite of C"

    # 5. Verify counter foundations use inverted world
    for fid, world in counter.foundations:
        if fid == 4:
            assert world == "B", f"Companionship subfoundation should shift to B, got {world}"

    # 6. Verify all outputs are canonical
    for elem in counter.elements:
        assert elem[0] in "ABCD", f"Invalid world in {elem}"
        assert 1 <= int(elem[1:]) <= 10, f"Invalid noetic in {elem}"

    for op in counter.ops:
        assert op in ALLOWED_OPS, f"Invalid operator: {op}"

    print("[PASS] test_subfoundation_tag_preservation passed")
    print(f"  Original: F4 in world {orig_sig.dominant_world}")
    print(f"  Inverted: F4 in world {inv_sig.dominant_world}")
    print(f"  Counter foundations: {[(fid, w) for fid, w in counter.foundations]}")


def test_complex_multidimensional_inversion():
    """
    Semantic regression test: Complex multi-dimensional inversion maintains canon validity.

    This test verifies that simultaneous inversions across multiple dimensions
    (world, noetic, foundation) maintain canonical validity. All output elements
    must be in {A, B, C, D}, noetics in {1-10}, foundations in {1-7}.
    """
    # Create complex expression with elements from all worlds and multiple foundations
    expr = TKSExpression(
        elements=[
            "A2",   # Spiritual positive (A↔D, 2↔3)
            "B5",   # Mental female (B↔C, 5↔6)
            "C8",   # Emotional cause (C↔B, 8↔9)
            "D1",   # Physical mind (D↔A, 1→1 self-dual)
            "A10",  # Spiritual idea (A↔D, 10→10 self-dual)
        ],
        ops=["->", "+T", "*T", "o"],
        foundations=[
            (1, "A"),  # Unity → Lust (1↔7)
            (2, "B"),  # Wisdom → Material (2↔6)
            (3, "C"),  # Life → Power (3↔5)
            (4, "D"),  # Companionship → Companionship (4→4 self-dual)
        ],
        acquisitions=[],
        raw=""
    )

    orig_sig, inv_sig, counter = compute_anti_attractor(expr, num_elements=5)

    # 1. Verify element inversions follow canonical mappings
    # A2 → D3, B5 → C6, C8 → B9, D1 → A1, A10 → D10
    expected_inverted = {("D", 3), ("C", 6), ("B", 9), ("A", 1), ("D", 10)}
    actual_inverted = set(inv_sig.element_counts.keys())

    assert expected_inverted == actual_inverted, \
        f"Element inversions don't match canonical mappings.\n" \
        f"Expected: {expected_inverted}\nActual: {actual_inverted}"

    # 2. Verify foundation inversions follow canonical mappings
    # F1→F7, F2→F6, F3→F5, F4→F4
    expected_foundations = {7, 6, 5, 4}
    assert inv_sig.foundation_tags == expected_foundations, \
        f"Foundation inversions don't match.\n" \
        f"Expected: {expected_foundations}\nActual: {inv_sig.foundation_tags}"

    # 3. Verify counter-scenario maintains canonical constraints
    # All worlds must be in A/B/C/D
    for elem in counter.elements:
        world = elem[0]
        assert world in "ABCD", \
            f"Counter element {elem} has invalid world {world} (must be A/B/C/D)"

    # All noetics must be in 1-10
    for elem in counter.elements:
        noetic = int(elem[1:])
        assert 1 <= noetic <= 10, \
            f"Counter element {elem} has invalid noetic {noetic} (must be 1-10)"

    # All foundations must be in 1-7
    for fid, world in counter.foundations:
        assert 1 <= fid <= 7, \
            f"Counter foundation {fid} is invalid (must be 1-7)"
        assert world in "ABCD", \
            f"Counter foundation subfoundation world {world} is invalid (must be A/B/C/D)"

    # All operators must be in ALLOWED_OPS
    for op in counter.ops:
        assert op in ALLOWED_OPS, \
            f"Counter operator {op} is invalid (must be in {ALLOWED_OPS})"

    # 4. Verify structural integrity
    assert len(counter.ops) == len(counter.elements) - 1, \
        "Operator count must equal element count minus 1"

    assert len(counter.foundations) == len(expr.foundations), \
        "Foundation count should be preserved"

    # 5. Verify involutive property: invert(invert(sig)) ≈ sig
    double_inv = invert_signature(inv_sig)
    assert orig_sig.element_counts == double_inv.element_counts, \
        "Double inversion should restore original element counts"
    assert orig_sig.foundation_tags == double_inv.foundation_tags, \
        "Double inversion should restore original foundation tags"

    print("[PASS] test_complex_multidimensional_inversion passed")
    print(f"  Original elements: {list(orig_sig.element_counts.keys())}")
    print(f"  Inverted elements: {list(inv_sig.element_counts.keys())}")
    print(f"  Original foundations: {orig_sig.foundation_tags}")
    print(f"  Inverted foundations: {inv_sig.foundation_tags}")
    print(f"  Counter elements: {counter.elements}")
    print(f"  Counter ops: {counter.ops}")
    print(f"  All canon constraints satisfied ✓")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING ANTI-ATTRACTOR SYNTHESIS TESTS")
    print("="*70 + "\n")

    # Test compute_attractor_signature
    print("--- Testing compute_attractor_signature ---")
    test_compute_signature_simple()
    test_compute_signature_with_repetitions()
    test_compute_signature_with_foundations()
    test_compute_signature_neutral_polarity()
    test_compute_signature_balanced_polarity()

    # Test invert_signature
    print("\n--- Testing invert_signature ---")
    test_invert_signature_involutive()
    test_invert_signature_world_opposites()
    test_invert_signature_noetic_opposites()
    test_invert_signature_self_dual_noetics()
    test_invert_signature_foundations()
    test_invert_signature_polarity()

    # Test synthesize_counter_scenario
    print("\n--- Testing synthesize_counter_scenario ---")
    test_synthesize_valid_expression()
    test_synthesize_uses_causal_operators()
    test_synthesize_respects_num_elements()
    test_synthesize_attaches_foundations()
    test_synthesize_element_multiplicity()

    # Test anti_attractor pipeline
    print("\n--- Testing anti_attractor pipeline ---")
    test_anti_attractor_pipeline()
    test_anti_attractor_positive_to_negative()
    test_anti_attractor_complex_scenario()

    # Test canon validity
    print("\n--- Testing canon validity ---")
    test_anti_attractor_canon_valid()
    test_anti_attractor_preserves_structure()

    # Test edge cases
    print("\n--- Testing edge cases ---")
    test_single_element_expression()
    test_empty_foundations()
    test_self_dual_only_expression()

    # Regression tests
    print("\n--- Testing regression cases ---")
    test_operator_variety_in_synthesis()
    test_counter_scenario_differs_from_original()
    test_counter_scenario_canon_validity_comprehensive()

    # New regression tests for refined heuristics
    print("\n--- Testing refined operator heuristics ---")
    test_operator_choice_for_specific_pairs()
    test_polarity_shift_positive_to_negative()
    test_world_shift_mental_to_emotional()
    test_involutive_property_double_inversion()

    # Semantic regression tests - Agent 3 tasks
    print("\n--- Testing semantic regression cases (Agent 3) ---")
    test_foundation_flip_unity_to_lust()
    test_subfoundation_tag_preservation()
    test_complex_multidimensional_inversion()

    print("\n" + "="*70)
    print("ALL ANTI-ATTRACTOR TESTS PASSED!")
    print("="*70 + "\n")

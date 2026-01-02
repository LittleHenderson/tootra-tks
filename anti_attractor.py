"""
Anti-Attractor Synthesis Algorithm - Design Specification

This module implements attractor signature analysis and counter-scenario synthesis
for TKS expressions. The algorithm identifies dominant patterns (attractors) in a
TKS scenario and synthesizes opposing scenarios (anti-attractors) that repel from
the original pattern space.

DESIGN OVERVIEW
===============

The anti-attractor synthesis operates in three phases:

1. SIGNATURE EXTRACTION: Analyze a TKS expression to extract its attractor signature
   - Count element frequencies by (world, noetic) pairs
   - Extract foundation tags and compute net polarity
   - Identify dominant patterns and operator distribution

2. SIGNATURE INVERSION: Transform the signature to its anti-attractor
   - Invert world and noetic dimensions using canonical opposites
   - Invert foundation tags and polarity
   - Preserve structural properties (operator distribution)

3. COUNTER-SCENARIO SYNTHESIS: Generate a new TKS expression from inverted signature
   - Select top (world, noetic) pairs from inverted signature
   - Construct causal chains using selected elements
   - Apply inverted foundations to complete the scenario

THEORETICAL FOUNDATION
=====================

An "attractor" in TKS space is a stable pattern characterized by:
- Dominant (world, noetic) element combinations that appear repeatedly
- Specific foundation alignments that reinforce the pattern
- Net polarity indicating overall emotional/energetic tendency

The "anti-attractor" is the complementary pattern that:
- Occupies the opposite region of TKS phase space
- Creates repulsive dynamics relative to the original attractor
- Offers maximum contrast while maintaining structural coherence

CANONICAL MAPPINGS
==================

World Opposites (from inversion/engine.py):
    A ↔ D    (Spiritual ↔ Physical)
    B ↔ C    (Mental ↔ Emotional)

Noetic Opposites (from inversion/engine.py):
    N1 (Mind) → N1       (self-dual)
    N2 (Positive) ↔ N3 (Negative)
    N4 (Vibration) → N4  (self-dual)
    N5 (Female) ↔ N6 (Male)
    N7 (Rhythm) → N7     (self-dual)
    N8 (Cause) ↔ N9 (Effect)
    N10 (Idea) → N10     (self-dual)

Foundation Opposites (from inversion/engine.py):
    F1 (Unity) ↔ F7 (Lust)
    F2 (Wisdom) ↔ F6 (Material)
    F3 (Life) ↔ F5 (Power)
    F4 (Companionship) → F4 (self-dual)

Polarity Classification:
    Positive noetics: {2, 5, 8}  (Positive, Female, Cause)
    Negative noetics: {3, 6, 9}  (Negative, Male, Effect)
    Neutral noetics: {1, 4, 7, 10} (self-dual principles)

IMPLEMENTATION NOTES
====================

- All element parsing follows format: WorldNoetic (e.g., "B5", "D3")
- Foundation format: List[Tuple[int, Optional[str]]] where str is subfoundation world
- Operators from ALLOWED_OPS: {"+T", "-T", "*T", "/T", "o", "->", "<-", "+", "-"}
- Causal operators ("->", "<-") are preferred for synthesis to create narrative flow
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter

from scenario_inversion import TKSExpression
from inversion.engine import WORLD_OPP, NOETIC_OPPOSITE, FOUNDATION_OPP
from narrative.constants import ALLOWED_OPS


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AttractorSignature:
    """
    Signature characterizing an attractor pattern in TKS space.

    An attractor signature captures the essential features of a TKS expression
    that define its stable pattern in the semantic phase space. This includes
    the distribution of elements, foundations, and overall polarity.

    Attributes:
        element_counts: Frequency map of (world, noetic) pairs in the expression.
                       Example: {("B", 2): 3, ("D", 5): 1} means B2 appears 3 times,
                       D5 appears once.

        foundation_tags: Set of foundation IDs present in the expression.
                        Example: {1, 4, 7} indicates Unity, Companionship, Lust.

        polarity: Net polarity of the expression computed from noetic distribution:
                 +1 if positive noetics (2,5,8) dominate
                 -1 if negative noetics (3,6,9) dominate
                  0 if balanced or neutral noetics (1,4,7,10) dominate

        ops_distribution: Frequency map of operators used in the expression.
                         Example: {"->": 2, "+T": 1} means 2 causal arrows, 1 additive.

        dominant_world: World letter appearing most frequently in elements.
                       Used to characterize the primary domain of the scenario.

        dominant_noetic: Noetic index appearing most frequently in elements.
                        Used to characterize the primary principle of the scenario.

    Usage:
        signature = compute_attractor_signature(expr)
        print(f"Dominant pattern: {signature.dominant_world}{signature.dominant_noetic}")
        print(f"Polarity: {signature.polarity}")
    """
    element_counts: Dict[Tuple[str, int], int] = field(default_factory=dict)
    foundation_tags: Set[int] = field(default_factory=set)
    polarity: int = 0  # -1 (negative), 0 (neutral), +1 (positive)
    ops_distribution: Dict[str, int] = field(default_factory=dict)
    dominant_world: str = "D"  # Default to Physical
    dominant_noetic: int = 1   # Default to Mind


# =============================================================================
# SIGNATURE EXTRACTION
# =============================================================================

def compute_attractor_signature(expr: TKSExpression) -> AttractorSignature:
    """
    Extract attractor signature from a TKS expression.

    This function analyzes the structure and content of a TKS expression to
    identify its characteristic pattern in TKS semantic space. The signature
    captures both the quantitative (frequencies) and qualitative (polarities,
    foundations) aspects of the expression.

    Algorithm:
    ----------
    1. Parse each element to extract (world, noetic) pairs
       - Element format: "B5" → world="B", noetic=5
       - Count occurrences of each unique pair

    2. Extract foundation tags from expr.foundations
       - foundations is List[Tuple[int, Optional[str]]]
       - Extract unique foundation IDs (first element of tuple)

    3. Compute net polarity from noetic distribution
       - Positive contribution: count of noetics in {2, 5, 8}
       - Negative contribution: count of noetics in {3, 6, 9}
       - Neutral noetics {1, 4, 7, 10} don't affect polarity
       - polarity = +1 if positive > negative
       - polarity = -1 if negative > positive
       - polarity = 0 if balanced or only neutral

    4. Count operator distribution
       - Tally each operator from expr.ops
       - Preserves structural information about expression composition

    5. Determine dominant world and noetic
       - dominant_world: world with highest total element count
       - dominant_noetic: noetic with highest total element count
       - Breaks ties arbitrarily (first encountered wins)

    Args:
        expr: TKSExpression to analyze

    Returns:
        AttractorSignature containing extracted pattern features

    Example:
        >>> expr = TKSExpression(
        ...     elements=["B2", "B2", "D5", "C3"],
        ...     ops=["->", "+T"],
        ...     foundations=[(1, "A"), (7, "D")],
        ...     acquisitions=[],
        ...     raw="B2 -> B2 +T D5 -> C3"
        ... )
        >>> sig = compute_attractor_signature(expr)
        >>> sig.element_counts
        {("B", 2): 2, ("D", 5): 1, ("C", 3): 1}
        >>> sig.foundation_tags
        {1, 7}
        >>> sig.polarity  # B2(+), B2(+), D5(+), C3(-) → +1
        1
        >>> sig.dominant_world
        "B"
        >>> sig.dominant_noetic
        2
    """
    # 1. Count (world, noetic) element pairs
    element_counts: Dict[Tuple[str, int], int] = Counter()
    world_counts: Counter = Counter()
    noetic_counts: Counter = Counter()

    for elem in expr.elements:
        # Parse element: format is "WorldNoetic" e.g. "B5"
        world = elem[0]
        noetic = int(elem[1:])

        element_counts[(world, noetic)] += 1
        world_counts[world] += 1
        noetic_counts[noetic] += 1

    # 2. Extract foundation tags
    foundation_tags = {fid for fid, _ in expr.foundations}

    # 3. Compute net polarity
    positive_noetics = {2, 5, 8}  # Positive, Female, Cause
    negative_noetics = {3, 6, 9}  # Negative, Male, Effect

    positive_count = sum(count for (_, noetic), count in element_counts.items()
                         if noetic in positive_noetics)
    negative_count = sum(count for (_, noetic), count in element_counts.items()
                         if noetic in negative_noetics)

    if positive_count > negative_count:
        polarity = 1
    elif negative_count > positive_count:
        polarity = -1
    else:
        polarity = 0

    # 4. Count operator distribution
    ops_distribution = Counter(expr.ops)

    # 5. Determine dominant world and noetic
    dominant_world = world_counts.most_common(1)[0][0] if world_counts else "D"
    dominant_noetic = noetic_counts.most_common(1)[0][0] if noetic_counts else 1

    return AttractorSignature(
        element_counts=dict(element_counts),
        foundation_tags=foundation_tags,
        polarity=polarity,
        ops_distribution=dict(ops_distribution),
        dominant_world=dominant_world,
        dominant_noetic=dominant_noetic,
    )


# =============================================================================
# SIGNATURE INVERSION
# =============================================================================

def invert_signature(sig: AttractorSignature) -> AttractorSignature:
    """
    Invert attractor signature to compute anti-attractor signature.

    This function transforms an attractor signature into its complementary
    anti-attractor by applying canonical TKS oppositions. The resulting
    signature characterizes a pattern that occupies the opposite region
    of TKS semantic space.

    Algorithm:
    ----------
    1. Invert element_counts keys using canonical oppositions:
       - World inversion: A↔D, B↔C (from WORLD_OPP)
       - Noetic inversion: 2↔3, 5↔6, 8↔9 (from NOETIC_OPPOSITE)
       - Self-dual noetics (1,4,7,10) map to themselves
       - Preserve frequency values from original signature

    2. Invert foundation_tags using canonical oppositions:
       - Foundation inversion: 1↔7, 2↔6, 3↔5, 4→4 (from FOUNDATION_OPP)
       - Maintains same foundation structure with opposite alignments

    3. Invert polarity:
       - +1 → -1 (positive to negative)
       - -1 → +1 (negative to positive)
       -  0 → 0  (neutral remains neutral)

    4. Handle ops_distribution:
       - CURRENT DESIGN: Preserve operator distribution unchanged
       - RATIONALE: Structural form is independent of content polarity
       - ALTERNATIVE: Could invert causal direction (-> to <-, +T to -T)
       - DECISION: Use preserve strategy for initial implementation

    5. Invert dominant_world and dominant_noetic:
       - Apply same inversion rules as element components
       - Maintains semantic coherence of inverted signature

    Args:
        sig: AttractorSignature to invert

    Returns:
        AttractorSignature representing the anti-attractor pattern

    Example:
        >>> sig = AttractorSignature(
        ...     element_counts={("B", 2): 2, ("D", 5): 1},
        ...     foundation_tags={1, 7},
        ...     polarity=1,
        ...     ops_distribution={"->": 2},
        ...     dominant_world="B",
        ...     dominant_noetic=2
        ... )
        >>> inv_sig = invert_signature(sig)
        >>> inv_sig.element_counts  # B→C, 2→3, D→A, 5→6
        {("C", 3): 2, ("A", 6): 1}
        >>> inv_sig.foundation_tags  # 1→7, 7→1
        {1, 7}  # Note: 1 and 7 are mutual opposites, so set is same
        >>> inv_sig.polarity
        -1
        >>> inv_sig.dominant_world
        "C"
        >>> inv_sig.dominant_noetic
        3
    """
    # 1. Invert element_counts keys
    inverted_elements: Dict[Tuple[str, int], int] = {}
    for (world, noetic), count in sig.element_counts.items():
        # Invert world using WORLD_OPP
        inv_world = WORLD_OPP.get(world, world)

        # Invert noetic using NOETIC_OPPOSITE (0-indexed in dict, so subtract 1)
        inv_noetic = NOETIC_OPPOSITE.get(noetic - 1, noetic - 1) + 1

        inverted_elements[(inv_world, inv_noetic)] = count

    # 2. Invert foundation_tags
    inverted_foundations = {FOUNDATION_OPP.get(fid, fid) for fid in sig.foundation_tags}

    # 3. Invert polarity
    inverted_polarity = -sig.polarity

    # 4. Preserve ops_distribution (design decision: structural form independent of polarity)
    # Alternative approach (commented out for now):
    # inverted_ops = {}
    # OP_INVERSIONS = {"->": "<-", "<-": "->", "+T": "-T", "-T": "+T"}
    # for op, count in sig.ops_distribution.items():
    #     inverted_ops[OP_INVERSIONS.get(op, op)] = count
    inverted_ops = dict(sig.ops_distribution)

    # 5. Invert dominant world and noetic
    inv_dominant_world = WORLD_OPP.get(sig.dominant_world, sig.dominant_world)
    inv_dominant_noetic = NOETIC_OPPOSITE.get(sig.dominant_noetic - 1,
                                               sig.dominant_noetic - 1) + 1

    return AttractorSignature(
        element_counts=inverted_elements,
        foundation_tags=inverted_foundations,
        polarity=inverted_polarity,
        ops_distribution=inverted_ops,
        dominant_world=inv_dominant_world,
        dominant_noetic=inv_dominant_noetic,
    )


# =============================================================================
# COUNTER-SCENARIO SYNTHESIS
# =============================================================================

def _choose_operator_for_pair(elem1: str, elem2: str) -> str:
    """
    Choose appropriate operator based on element characteristics.

    This heuristic selects operators deterministically based on noetic polarities
    and types to create more varied and semantically appropriate expressions.

    Heuristic Rules (in priority order):
    ------------------------------------
    1. Cause (N8) to Effect (N9) → -> (causal flow)
    2. Effect (N9) to Cause (N8) → <- (reverse causal)
    3. Rhythm/Pattern (N7) with any element → o (sequential composition)
    4. Female (N5) to Male (N6) → -> (receptive to projective flow)
    5. Male (N6) to Female (N5) → <- (projective to receptive flow)
    6. Same polarity (both pos or both neg) → *T (intensify/amplify)
    7. Opposite polarity (pos vs neg) → /T (conflict/oppose)
    8. Both neutral noetics (N1,N4,N7,N10) → +T (combine together)
    9. One neutral, one polar → o (sequence: neutral sets context for polar)
    10. Default fallback → -> (causal narrative)

    Args:
        elem1: First element (e.g., "B2", "C3")
        elem2: Second element (e.g., "D5", "A6")

    Returns:
        Operator string from ALLOWED_OPS

    Examples:
        >>> _choose_operator_for_pair("B2", "D2")  # Both positive
        "*T"
        >>> _choose_operator_for_pair("C3", "A2")  # Negative to positive
        "/T"
        >>> _choose_operator_for_pair("B8", "D9")  # Cause to effect
        "->"
        >>> _choose_operator_for_pair("D9", "A8")  # Effect to cause
        "<-"
        >>> _choose_operator_for_pair("D7", "C3")  # Rhythm to negative
        "o"
        >>> _choose_operator_for_pair("B5", "D6")  # Female to male
        "->"
    """
    # Parse noetics and worlds
    world1 = elem1[0]
    noetic1 = int(elem1[1:])
    world2 = elem2[0]
    noetic2 = int(elem2[1:])

    # Define polarity sets (from design spec)
    positive_noetics = {2, 5, 8}  # Positive, Female, Cause
    negative_noetics = {3, 6, 9}  # Negative, Male, Effect
    neutral_noetics = {1, 4, 7, 10}  # Mind, Vibration, Rhythm, Idea

    # Rule 1: Cause → Effect (highest priority for narrative flow)
    if noetic1 == 8 and noetic2 == 9:
        return "->"

    # Rule 2: Effect → Cause (reverse causal)
    if noetic1 == 9 and noetic2 == 8:
        return "<-"

    # Rule 3: Rhythm elements use sequential composition
    if noetic1 == 7 or noetic2 == 7:
        return "o"

    # Rule 4: Female → Male (receptive to projective) - before general polarity check
    if noetic1 == 5 and noetic2 == 6:
        return "->"

    # Rule 5: Male → Female (projective to receptive) - before general polarity check
    if noetic1 == 6 and noetic2 == 5:
        return "<-"

    # Determine polarities
    pol1_pos = noetic1 in positive_noetics
    pol1_neg = noetic1 in negative_noetics
    pol1_neutral = noetic1 in neutral_noetics

    pol2_pos = noetic2 in positive_noetics
    pol2_neg = noetic2 in negative_noetics
    pol2_neutral = noetic2 in neutral_noetics

    # Rule 6: Same polarity → *T (intensify)
    if (pol1_pos and pol2_pos) or (pol1_neg and pol2_neg):
        return "*T"

    # Rule 7: Opposite polarity → /T (conflict)
    if (pol1_pos and pol2_neg) or (pol1_neg and pol2_pos):
        return "/T"

    # Rule 8: Both neutral → +T (combine)
    if pol1_neutral and pol2_neutral:
        return "+T"

    # Rule 9: One neutral, one polar → o (sequence)
    if (pol1_neutral and not pol2_neutral) or (not pol1_neutral and pol2_neutral):
        return "o"

    # Rule 10: Default fallback → -> (causal)
    return "->"


def synthesize_counter_scenario(
    inv_sig: AttractorSignature,
    num_elements: int = 3,
    prefer_causal: bool = True
) -> TKSExpression:
    """
    Synthesize a counter-scenario TKS expression from inverted attractor signature.

    This function generates a new TKS expression that embodies the anti-attractor
    pattern characterized by the inverted signature. The synthesis creates a
    coherent scenario in the opposite region of TKS semantic space.

    Algorithm:
    ----------
    1. Select top N (world, noetic) pairs from inverted signature by frequency count
       - Sort element_counts by value (frequency) in descending order
       - Take top num_elements pairs
       - If fewer pairs available, use all available pairs
       - Ensures most characteristic elements of anti-attractor are represented

    2. Construct element strings from selected pairs
       - Format: world + str(noetic), e.g., ("B", 5) → "B5"
       - Preserve multiplicity: if count > 1, repeat element in construction
       - Order elements by frequency (most frequent first)

    3. Choose operators from ALLOWED_OPS to connect elements
       - If prefer_causal=True: use contextual heuristics (see _choose_operator_for_pair)
       - If prefer_causal=False: mix operators based on original ops_distribution
       - Number of operators = num_elements - 1 (to connect N elements)
       - IMPROVED: Operator selection now considers element noetics for variety

    4. Attach inverted foundations if present
       - For each foundation ID in inv_sig.foundation_tags:
       - Create foundation tuple (fid, subfoundation_world)
       - Use dominant_world from inv_sig as subfoundation world
       - Format: List[Tuple[int, Optional[str]]]

    5. Assemble TKSExpression
       - elements: constructed element list
       - ops: chosen operators
       - foundations: inverted foundation attachments
       - acquisitions: empty (no acquisition data in signature)
       - raw: empty (synthesis doesn't preserve original text)

    Args:
        inv_sig: Inverted AttractorSignature to synthesize from
        num_elements: Number of elements to include in synthesized expression (default: 3)
        prefer_causal: If True, use contextual heuristics; else mix operators (default: True)

    Returns:
        TKSExpression embodying the anti-attractor pattern

    Example:
        >>> inv_sig = AttractorSignature(
        ...     element_counts={("C", 3): 2, ("A", 6): 1, ("B", 9): 1},
        ...     foundation_tags={7, 6},
        ...     polarity=-1,
        ...     ops_distribution={"->": 2},
        ...     dominant_world="C",
        ...     dominant_noetic=3
        ... )
        >>> expr = synthesize_counter_scenario(inv_sig, num_elements=3)
        >>> expr.elements
        ["C3", "C3", "A6"]  # Top 2 by frequency: C3(2x), A6(1x)
        >>> expr.ops
        ["*T", "/T"]  # Contextual: C3*TC3 (same polarity), C3/TA6 (opposite)
        >>> expr.foundations
        [(6, "C"), (7, "C")]  # Inverted foundations with dominant world

    Design Rationale:
    -----------------
    - Frequency-based selection ensures dominant anti-attractor features are preserved
    - Contextual operator selection creates more varied and semantically rich expressions
    - Foundation attachment grounds the scenario in specific TKS domains
    - Multiplicity preservation maintains attractor "weight" in element space

    Improvement from v1:
    --------------------
    - Operator selection now uses deterministic heuristics based on noetic types
    - *T for same-polarity amplification, /T for opposite-polarity conflict
    - +T for neutral combinations, -> for causal flow, o for sequential patterns
    - Provides operator variety while maintaining deterministic behavior
    """
    # 1. Select top N (world, noetic) pairs by frequency
    sorted_elements = sorted(inv_sig.element_counts.items(),
                            key=lambda x: x[1],
                            reverse=True)

    # Limit to num_elements or available count, whichever is smaller
    selected_pairs = sorted_elements[:num_elements] if sorted_elements else []

    # 2. Construct element strings with multiplicity
    elements: List[str] = []
    for (world, noetic), count in selected_pairs:
        element_str = f"{world}{noetic}"
        # Add element 'count' times to preserve frequency weight
        # For synthesis, we limit to max 2 repetitions per element for readability
        repetitions = min(count, 2)
        elements.extend([element_str] * repetitions)

    # Ensure we have at least 1 element (fallback to dominant if needed)
    if not elements:
        elements = [f"{inv_sig.dominant_world}{inv_sig.dominant_noetic}"]

    # 3. Choose operators to connect elements
    num_ops_needed = len(elements) - 1
    ops: List[str] = []

    if prefer_causal:
        # Use contextual heuristics to choose operators based on element noetics
        for i in range(num_ops_needed):
            elem1 = elements[i]
            elem2 = elements[i + 1]
            op = _choose_operator_for_pair(elem1, elem2)
            ops.append(op)
    else:
        # Mix operators based on original distribution
        # Fallback to "->" if no distribution available
        available_ops = list(inv_sig.ops_distribution.keys()) if inv_sig.ops_distribution else ["->"]
        ops = [available_ops[i % len(available_ops)] for i in range(num_ops_needed)]

    # 4. Attach inverted foundations
    foundations: List[Tuple[int, Optional[str]]] = []
    for fid in sorted(inv_sig.foundation_tags):
        # Use dominant_world from inverted signature as subfoundation world
        foundations.append((fid, inv_sig.dominant_world))

    # 5. Assemble TKSExpression
    return TKSExpression(
        elements=elements,
        ops=ops,
        foundations=foundations,
        acquisitions=[],  # No acquisition data from signature
        raw=""  # Synthesis doesn't preserve original raw text
    )


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def compute_anti_attractor(
    expr: TKSExpression,
    num_elements: int = 3,
    prefer_causal: bool = True
) -> Tuple[AttractorSignature, AttractorSignature, TKSExpression]:
    """
    Complete pipeline: analyze expression → compute anti-attractor → synthesize counter-scenario.

    This is the main entry point for anti-attractor synthesis. It performs the
    complete transformation from an input TKS expression to a counter-scenario
    that occupies the opposite region of TKS semantic space.

    Pipeline Stages:
    ----------------
    1. SIGNATURE EXTRACTION
       - Analyze input expression to extract attractor signature
       - Identify dominant patterns, foundations, and polarity

    2. SIGNATURE INVERSION
       - Transform signature to its anti-attractor complement
       - Apply canonical TKS oppositions

    3. COUNTER-SCENARIO SYNTHESIS
       - Generate new TKS expression from inverted signature
       - Construct coherent scenario embodying anti-attractor

    Args:
        expr: Original TKSExpression to analyze
        num_elements: Number of elements in synthesized counter-scenario (default: 3)
        prefer_causal: Use causal operators in synthesis (default: True)

    Returns:
        Tuple containing:
        - original_sig: AttractorSignature of input expression
        - inverted_sig: Anti-attractor signature (inverted)
        - counter_expr: Synthesized TKSExpression embodying anti-attractor

    Example:
        >>> original = TKSExpression(
        ...     elements=["B2", "B2", "D5"],
        ...     ops=["->", "+T"],
        ...     foundations=[(1, "A")],
        ...     acquisitions=[],
        ...     raw="positive mental belief causes woman"
        ... )
        >>> orig_sig, inv_sig, counter = compute_anti_attractor(original)
        >>>
        >>> # Original signature
        >>> orig_sig.dominant_world, orig_sig.dominant_noetic
        ("B", 2)
        >>> orig_sig.polarity
        1
        >>>
        >>> # Inverted signature
        >>> inv_sig.dominant_world, inv_sig.dominant_noetic
        ("C", 3)  # Mental→Emotional, Positive→Negative
        >>> inv_sig.polarity
        -1
        >>>
        >>> # Synthesized counter-scenario
        >>> counter.elements
        ["C3", "C3", "A6"]  # Emotional negative fear, Male spiritual
        >>> counter.foundations
        [(7, "C")]  # Unity→Lust, with Emotional subfoundation

    Use Cases:
    ----------
    - Therapeutic intervention: Generate opposite scenario to break pattern
    - Creative exploration: Discover complementary narrative possibilities
    - Pattern analysis: Understand scenario by examining its opposite
    - Scenario balancing: Create counterweight to dominant attractor
    """
    # Stage 1: Extract original signature
    original_sig = compute_attractor_signature(expr)

    # Stage 2: Invert signature to anti-attractor
    inverted_sig = invert_signature(original_sig)

    # Stage 3: Synthesize counter-scenario from inverted signature
    counter_expr = synthesize_counter_scenario(
        inverted_sig,
        num_elements=num_elements,
        prefer_causal=prefer_causal
    )

    return original_sig, inverted_sig, counter_expr


def anti_attractor(
    expr: TKSExpression,
    num_elements: int = 3,
    prefer_causal: bool = True
) -> TKSExpression:
    """
    Convenience function: compute anti-attractor and return only the counter-scenario.

    This is a simplified API that returns just the synthesized counter-scenario
    expression without the intermediate signatures.

    Args:
        expr: Original TKSExpression to analyze
        num_elements: Number of elements in synthesized counter-scenario (default: 3)
        prefer_causal: Use causal operators in synthesis (default: True)

    Returns:
        TKSExpression embodying the anti-attractor pattern

    Example:
        >>> original = parse_equation("B2 -> D5")
        >>> counter = anti_attractor(original)
        >>> counter.elements
        ['C3', 'A6']
    """
    _, _, counter_expr = compute_anti_attractor(expr, num_elements, prefer_causal)
    return counter_expr


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def explain_signature(sig: AttractorSignature, label: str = "Signature") -> str:
    """
    Generate human-readable explanation of an attractor signature.

    This utility provides a formatted description of a signature's key features
    for debugging, analysis, and user feedback.

    Args:
        sig: AttractorSignature to explain
        label: Descriptive label for the signature (default: "Signature")

    Returns:
        Multi-line string explaining signature contents

    Example:
        >>> sig = AttractorSignature(
        ...     element_counts={("B", 2): 3, ("D", 5): 1},
        ...     foundation_tags={1, 4},
        ...     polarity=1,
        ...     ops_distribution={"->": 2, "+T": 1},
        ...     dominant_world="B",
        ...     dominant_noetic=2
        ... )
        >>> print(explain_signature(sig, "Original"))
        Original Attractor Signature:
          Dominant: B2 (Mental/Positive)
          Polarity: +1 (Positive)
          Element Distribution:
            B2: 3 occurrences
            D5: 1 occurrence
          Foundations: {1, 4} (Unity, Companionship)
          Operators: -> (2x), +T (1x)
    """
    lines = [f"{label} Attractor Signature:"]

    # Dominant pattern
    polarity_str = {1: "Positive", -1: "Negative", 0: "Neutral"}.get(sig.polarity, "Unknown")
    lines.append(f"  Dominant: {sig.dominant_world}{sig.dominant_noetic}")
    lines.append(f"  Polarity: {sig.polarity:+d} ({polarity_str})")

    # Element distribution
    if sig.element_counts:
        lines.append("  Element Distribution:")
        for (world, noetic), count in sorted(sig.element_counts.items(),
                                            key=lambda x: x[1],
                                            reverse=True):
            lines.append(f"    {world}{noetic}: {count} occurrence{'s' if count > 1 else ''}")
    else:
        lines.append("  Element Distribution: (empty)")

    # Foundations
    if sig.foundation_tags:
        from narrative.constants import FOUNDATIONS
        foundation_names = [FOUNDATIONS.get(fid, str(fid)) for fid in sorted(sig.foundation_tags)]
        lines.append(f"  Foundations: {{{', '.join(foundation_names)}}}")
    else:
        lines.append("  Foundations: (none)")

    # Operators
    if sig.ops_distribution:
        op_strs = [f"{op} ({count}x)" for op, count in sig.ops_distribution.items()]
        lines.append(f"  Operators: {', '.join(op_strs)}")
    else:
        lines.append("  Operators: (none)")

    return "\n".join(lines)


def compare_signatures(
    sig1: AttractorSignature,
    sig2: AttractorSignature,
    label1: str = "Original",
    label2: str = "Inverted"
) -> str:
    """
    Generate side-by-side comparison of two attractor signatures.

    Useful for visualizing the transformation from attractor to anti-attractor.

    Args:
        sig1: First AttractorSignature
        sig2: Second AttractorSignature
        label1: Label for first signature (default: "Original")
        label2: Label for second signature (default: "Inverted")

    Returns:
        Multi-line string comparing both signatures

    Example:
        >>> orig_sig = compute_attractor_signature(original_expr)
        >>> inv_sig = invert_signature(orig_sig)
        >>> print(compare_signatures(orig_sig, inv_sig))
        Attractor Signature Comparison:
        ================================

        Original:
          Dominant: B2 (Mental/Positive)
          Polarity: +1 (Positive)
          Elements: B2(3), D5(1)
          Foundations: {Unity, Companionship}

        Inverted:
          Dominant: C3 (Emotional/Negative)
          Polarity: -1 (Negative)
          Elements: C3(3), A6(1)
          Foundations: {Lust, Companionship}

        Transformation Summary:
          World: B → C (Mental → Emotional)
          Noetic: 2 → 3 (Positive → Negative)
          Polarity: +1 → -1
          Foundations: Unity → Lust
    """
    from narrative.constants import FOUNDATIONS, WORLDS, NOETIC_NAMES

    lines = ["Attractor Signature Comparison:", "=" * 32, ""]

    # Signature 1
    lines.append(f"{label1}:")
    lines.append(f"  Dominant: {sig1.dominant_world}{sig1.dominant_noetic} "
                f"({WORLDS[sig1.dominant_world]}/{NOETIC_NAMES[sig1.dominant_noetic]})")
    polarity1 = {1: "Positive", -1: "Negative", 0: "Neutral"}.get(sig1.polarity, "Unknown")
    lines.append(f"  Polarity: {sig1.polarity:+d} ({polarity1})")

    elem1_strs = [f"{w}{n}({c})" for (w, n), c in sig1.element_counts.items()]
    lines.append(f"  Elements: {', '.join(elem1_strs) if elem1_strs else '(none)'}")

    found1_names = [FOUNDATIONS.get(f, str(f)) for f in sorted(sig1.foundation_tags)]
    lines.append(f"  Foundations: {{{', '.join(found1_names)}}} "
                f"if found1_names else '(none)'")

    lines.append("")

    # Signature 2
    lines.append(f"{label2}:")
    lines.append(f"  Dominant: {sig2.dominant_world}{sig2.dominant_noetic} "
                f"({WORLDS[sig2.dominant_world]}/{NOETIC_NAMES[sig2.dominant_noetic]})")
    polarity2 = {1: "Positive", -1: "Negative", 0: "Neutral"}.get(sig2.polarity, "Unknown")
    lines.append(f"  Polarity: {sig2.polarity:+d} ({polarity2})")

    elem2_strs = [f"{w}{n}({c})" for (w, n), c in sig2.element_counts.items()]
    lines.append(f"  Elements: {', '.join(elem2_strs) if elem2_strs else '(none)'}")

    found2_names = [FOUNDATIONS.get(f, str(f)) for f in sorted(sig2.foundation_tags)]
    lines.append(f"  Foundations: {{{', '.join(found2_names)}}} "
                f"if found2_names else '(none)'")

    lines.append("")

    # Transformation summary
    lines.append("Transformation Summary:")
    lines.append(f"  World: {sig1.dominant_world} → {sig2.dominant_world} "
                f"({WORLDS[sig1.dominant_world]} → {WORLDS[sig2.dominant_world]})")
    lines.append(f"  Noetic: {sig1.dominant_noetic} → {sig2.dominant_noetic} "
                f"({NOETIC_NAMES[sig1.dominant_noetic]} → "
                f"{NOETIC_NAMES[sig2.dominant_noetic]})")
    lines.append(f"  Polarity: {sig1.polarity:+d} → {sig2.polarity:+d}")

    # Foundation changes
    if sig1.foundation_tags or sig2.foundation_tags:
        removed = sig1.foundation_tags - sig2.foundation_tags
        added = sig2.foundation_tags - sig1.foundation_tags
        if removed:
            removed_names = [FOUNDATIONS[f] for f in removed]
            lines.append(f"  Foundations removed: {', '.join(removed_names)}")
        if added:
            added_names = [FOUNDATIONS[f] for f in added]
            lines.append(f"  Foundations added: {', '.join(added_names)}")

    return "\n".join(lines)


# =============================================================================
# DESIGN NOTES AND FUTURE ENHANCEMENTS
# =============================================================================

"""
FUTURE ENHANCEMENTS
===================

1. ADAPTIVE SYNTHESIS STRATEGIES
   - Pattern-aware synthesis: detect common motifs (e.g., repeating elements)
   - Context-sensitive operator selection: choose ops based on element types
   - Multi-expression synthesis: generate multiple counter-scenarios with variations

2. SIGNATURE REFINEMENTS
   - Causal structure signature: capture not just operators but causal graph topology
   - Temporal signature: encode sequence/ordering of elements beyond mere frequency
   - Acquisition signature: include acquisition polarity in signature

3. VALIDATION AND QUALITY METRICS
   - Coherence scoring: measure semantic coherence of synthesized expression
   - Contrast metric: quantify "distance" between original and anti-attractor
   - Stability analysis: detect if signature represents stable vs transient attractor

4. ADVANCED INVERSIONS
   - Partial inversion: invert only specific signature components
   - Weighted inversion: apply different inversion strengths to different dimensions
   - Contextual inversion: use TargetProfile to guide inversion toward specific patterns

5. INTEGRATION WITH NARRATIVE MODULE
   - DecodeStory integration: generate natural language from synthesized counter-scenario
   - Story-level anti-attractor: apply to full narratives, not just expressions
   - Thematic inversions: detect and invert high-level themes beyond element patterns

6. OPTIMIZATION
   - Signature caching: memoize signatures for repeated expressions
   - Batch synthesis: generate multiple counter-scenarios efficiently
   - Parallel inversion: process multiple signatures concurrently

DESIGN DECISIONS LOG
=====================

Decision 1: Operator Distribution Handling in Inversion
--------------------------------------------------------
QUESTION: Should ops_distribution be inverted (-> to <-, +T to -T)?

OPTION A (chosen): Preserve ops_distribution unchanged
  RATIONALE: Structural form is independent of semantic polarity.
             The way elements combine is a formal property, not content.
  PROS: Simpler, maintains structural isomorphism
  CONS: May miss opportunities for deeper structural opposition

OPTION B: Invert directional operators
  RATIONALE: Causal direction is semantic, should flip with content
  PROS: Creates stronger opposition at structural level
  CONS: More complex, may produce incoherent expressions

DECISION: Start with OPTION A, add OPTION B as flag later if needed.

Decision 2: Element Multiplicity in Synthesis
----------------------------------------------
QUESTION: How to handle elements with count > 1 in synthesis?

OPTION A (chosen): Repeat elements up to max of 2
  RATIONALE: Preserve frequency weight without bloating expression
  PROS: Captures attractor strength, maintains readability
  CONS: Arbitrary cutoff, may lose high-frequency information

OPTION B: Include each element once regardless of count
  RATIONALE: Signature counts capture frequency, no need to repeat
  PROS: Minimal, clean expressions
  CONS: Loses attractor "weight" information in final expression

DECISION: Use OPTION A with configurable max_repetitions parameter in future.

Decision 3: Foundation Subfoundation World Assignment
------------------------------------------------------
QUESTION: What world should subfoundations use in synthesized expression?

OPTION A (chosen): Use inv_sig.dominant_world for all subfoundations
  RATIONALE: Maintains coherence with dominant pattern
  PROS: Consistent, simple
  CONS: May not reflect diversity if multiple worlds present

OPTION B: Use world from corresponding element
  RATIONALE: More precise alignment between foundations and elements
  PROS: More nuanced, captures finer structure
  CONS: Complex, may not have 1-1 element-foundation correspondence

DECISION: Use OPTION A for simplicity, revisit if multi-world precision needed.

Decision 4: Default num_elements Value
---------------------------------------
QUESTION: How many elements should default synthesis include?

OPTION A (chosen): num_elements=3
  RATIONALE: Balance between expressiveness and simplicity
             3 elements allow for basic narrative structure (setup-action-result)
  PROS: Sufficient for coherent scenarios, not overwhelming
  CONS: May oversimplify complex multi-element attractors

OPTION B: num_elements=len(original.elements)
  RATIONALE: Match original expression length
  PROS: Maintains structural similarity
  CONS: May be too long, includes low-frequency elements

DECISION: Use OPTION A (3) as default, allow override via parameter.

TESTING STRATEGY
=================

Unit Tests Required:
--------------------
1. test_compute_attractor_signature
   - Single element expression
   - Multi-element with repetitions
   - Mixed polarity elements
   - Multiple foundations
   - Empty expression edge case

2. test_invert_signature
   - Self-dual noetics (1,4,7,10) map to themselves
   - Involutive pairs correctly swap (2↔3, 5↔6, 8↔9)
   - World inversions (A↔D, B↔C)
   - Foundation inversions (1↔7, 2↔6, 3↔5, 4→4)
   - Polarity inversion

3. test_synthesize_counter_scenario
   - Top N element selection
   - Operator generation (causal vs mixed)
   - Foundation attachment
   - Edge case: empty signature
   - Edge case: single element

4. test_compute_anti_attractor
   - End-to-end pipeline
   - Verify original → inverted → synthesis consistency
   - Multiple scenarios with different characteristics

Integration Tests:
------------------
1. Real scenario examples from docs/examples
2. Round-trip: original → anti-attractor → anti-anti-attractor
   (should approximate original signature)
3. Decode synthesized expressions to natural language
4. Compare with manual inversion using ScenarioInvert

Performance Tests:
------------------
1. Large expressions (100+ elements)
2. Batch synthesis (1000+ counter-scenarios)
3. Memory usage with complex signatures
"""

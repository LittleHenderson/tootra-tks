"""
Anti-Attractor Synthesis Demo

Demonstrates how to use the anti-attractor module to generate counter-scenarios
that pull away from attractor basins.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenario_inversion import parse_equation, DecodeStory, AntiAttractorInvert
from anti_attractor import compute_attractor_signature, anti_attractor


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_anti_attractor():
    """Demonstrate basic anti-attractor synthesis."""
    print_section("DEMO 1: Basic Anti-Attractor Synthesis")

    # Example: Positive emotional/mental/physical state
    expr = parse_equation("B2 -> C2 +T D2")
    print(f"\nOriginal Equation: B2 -> C2 +T D2")
    print(f"Original Story: {DecodeStory(expr)}")

    # Compute signature
    sig = compute_attractor_signature(expr)
    print(f"\nAttractor Signature:")
    print(f"  Element counts: {dict(sig.element_counts)}")
    print(f"  Dominant world: {sig.dominant_world}")
    print(f"  Dominant noetic: N{sig.dominant_noetic}")
    print(f"  Polarity: {sig.polarity} ({'positive' if sig.polarity > 0 else 'negative' if sig.polarity < 0 else 'neutral'})")

    # Generate counter-scenario
    counter = anti_attractor(expr)
    print(f"\nCounter-Scenario Equation: {' '.join([e + (' ' + counter.ops[i] if i < len(counter.ops) else '') for i, e in enumerate(counter.elements)])}")
    print(f"Counter-Scenario Story: {DecodeStory(counter)}")


def demo_negative_attractor():
    """Demonstrate anti-attractor for negative states."""
    print_section("DEMO 2: Inverting Negative Attractor")

    # Example: Fear-based pattern
    expr = parse_equation("C3 -> B3 -> D3")
    print(f"\nOriginal Equation: C3 -> B3 -> D3")
    print(f"Original Story: {DecodeStory(expr)}")

    # Generate counter-scenario
    result = AntiAttractorInvert(expr, return_signature=True)
    sig = result['signature']
    counter = result['expr_inverted']

    print(f"\nOriginal Polarity: {sig.polarity} (negative)")
    print(f"Inverted Polarity: -{sig.polarity} (positive)")

    print(f"\nCounter-Scenario: {' '.join([e + (' ' + counter.ops[i] if i < len(counter.ops) else '') for i, e in enumerate(counter.elements)])}")
    print(f"Counter-Scenario Story: {DecodeStory(counter)}")


def demo_foundation_inversion():
    """Demonstrate foundation inversion in anti-attractor."""
    print_section("DEMO 3: Foundation Inversion")

    # Example: Unity foundation (F1)
    from scenario_inversion import TKSExpression
    expr = TKSExpression(
        elements=["A2", "B2", "C2"],
        ops=["->", "+T"],
        foundations=[(1, None), (3, None)],  # Unity + Life
        acquisitions=[],
        raw=""
    )

    print(f"\nOriginal Equation: A2 -> B2 +T C2 [F1,F3]")
    print(f"Original Foundations: Unity (F1), Life (F3)")

    counter = anti_attractor(expr)
    foundation_names = {
        1: "Unity", 2: "Wisdom", 3: "Life", 4: "Companionship",
        5: "Power", 6: "Material", 7: "Lust"
    }

    inverted_names = [foundation_names[fid] for fid, _ in counter.foundations]
    print(f"\nInverted Foundations: {', '.join([f'{name} (F{fid})' for (fid, _), name in zip(counter.foundations, inverted_names)])}")
    print(f"  F1 (Unity) -> F7 (Lust)")
    print(f"  F3 (Life) -> F5 (Power)")

    print(f"\nCounter-Scenario: {' '.join([e + (' ' + counter.ops[i] if i < len(counter.ops) else '') for i, e in enumerate(counter.elements)])}")


def demo_mixed_worlds():
    """Demonstrate world inversion across all four worlds."""
    print_section("DEMO 4: Multi-World Attractor Inversion")

    # Example: All four worlds
    expr = parse_equation("A2 -> B5 +T C8 -> D3")
    print(f"\nOriginal Equation: A2 -> B5 +T C8 -> D3")
    print(f"Original Story: {DecodeStory(expr)}")

    sig = compute_attractor_signature(expr)
    print(f"\nOriginal Elements:")
    for (world, noetic), count in sig.element_counts.items():
        world_names = {"A": "Spiritual", "B": "Mental", "C": "Emotional", "D": "Physical"}
        print(f"  {world}{noetic} ({world_names[world]}, N{noetic})")

    counter = anti_attractor(expr)
    print(f"\nInverted Elements (world opposites: A<->D, B<->C, noetic involutions):")
    for elem in counter.elements:
        world = elem[0]
        noetic = int(elem[1:])
        world_names = {"A": "Spiritual", "B": "Mental", "C": "Emotional", "D": "Physical"}
        print(f"  {elem} ({world_names[world]}, N{noetic})")

    print(f"\nCounter-Scenario Story: {DecodeStory(counter)}")


def demo_polarity_analysis():
    """Demonstrate polarity analysis and inversion."""
    print_section("DEMO 5: Polarity Analysis")

    # Create mixed polarity expression
    from scenario_inversion import TKSExpression
    expr = TKSExpression(
        elements=["B2", "B3", "C2", "C3", "D2"],  # 3 positive (N2), 2 negative (N3)
        ops=["->", "+T", "->", "+T"],
        foundations=[],
        acquisitions=[],
        raw=""
    )

    sig = compute_attractor_signature(expr)
    print(f"\nOriginal Elements: B2, B3, C2, C3, D2")
    print(f"Positive noetics (N2, N5, N8): 3 elements")
    print(f"Negative noetics (N3, N6, N9): 2 elements")
    print(f"Calculated Polarity: {sig.polarity} (positive)")

    counter = anti_attractor(expr)
    print(f"\nCounter-Scenario Elements: {', '.join(counter.elements)}")
    print(f"After inversion: All polarities flipped")
    print(f"  N2 (Positive) -> N3 (Negative)")
    print(f"  N3 (Negative) -> N2 (Positive)")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  TKS ANTI-ATTRACTOR SYNTHESIS - COMPREHENSIVE DEMO".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    demo_basic_anti_attractor()
    demo_negative_attractor()
    demo_foundation_inversion()
    demo_mixed_worlds()
    demo_polarity_analysis()

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Concepts:")
    print("  - Anti-attractor synthesis generates counter-scenarios")
    print("  - Inverts world (A<->D, B<->C), noetic (2<->3, 5<->6, 8<->9), and foundation (1<->7, etc.)")
    print("  - Analyzes polarity to determine positive vs negative orientation")
    print("  - Selects top 1-3 most frequent inverted elements")
    print("  - Useful for creating escape routes from negative attractor basins")
    print()

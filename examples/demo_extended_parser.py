"""
Demo: Extended Parser Syntax (Sense/Foundation Suffixes)

Demonstrates the new extended token syntax support:
- B8^5 (sense suffix using caret notation)
- B8_d5 (foundation suffix)
- B8^5_d5 (full extended form)

Canon guardrails enforced:
- Worlds: A/B/C/D only
- Noetics: 1-10
- Foundations: 1-7
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative.types import ElementRef
from narrative.encoder import parse_equation
from narrative.constants import get_subfound_label


def demo_sense_notation():
    """Demo sense suffix parsing."""
    print("=" * 70)
    print("SENSE SUFFIX NOTATION")
    print("=" * 70)

    # Caret notation (new)
    elem1 = ElementRef.from_string("B8^5")
    print(f"\n1. Caret notation: B8^5")
    print(f"   World: {elem1.world}")
    print(f"   Noetic: {elem1.noetic}")
    print(f"   Sense: {elem1.sense}")
    print(f"   Full code: {elem1.full_code}")

    # Dot notation (backward compatible)
    elem2 = ElementRef.from_string("B8.5")
    print(f"\n2. Dot notation: B8.5 (backward compatible)")
    print(f"   World: {elem2.world}")
    print(f"   Noetic: {elem2.noetic}")
    print(f"   Sense: {elem2.sense}")
    print(f"   Full code: {elem2.full_code} (converted to caret)")


def demo_foundation_notation():
    """Demo foundation suffix parsing."""
    print("\n" + "=" * 70)
    print("FOUNDATION SUFFIX NOTATION")
    print("=" * 70)

    elem = ElementRef.from_string("B8_d5")
    print(f"\n1. Foundation suffix: B8_d5")
    print(f"   World: {elem.world}")
    print(f"   Noetic: {elem.noetic}")
    print(f"   Foundation: {elem.foundation}")
    print(f"   Subfoundation (world): {elem.subfoundation}")
    print(f"   Full code: {elem.full_code}")

    # Get subfoundation label
    label = get_subfound_label(elem.foundation, elem.subfoundation)
    print(f"   Subfoundation label: {label}")


def demo_full_extended():
    """Demo full extended syntax."""
    print("\n" + "=" * 70)
    print("FULL EXTENDED SYNTAX (Sense + Foundation)")
    print("=" * 70)

    elem = ElementRef.from_string("B8^5_d5")
    print(f"\n1. Full extended: B8^5_d5")
    print(f"   World: {elem.world}")
    print(f"   Noetic: {elem.noetic}")
    print(f"   Sense: {elem.sense}")
    print(f"   Foundation: {elem.foundation}")
    print(f"   Subfoundation: {elem.subfoundation}")
    print(f"   Full code: {elem.full_code}")

    # Get subfoundation label
    label = get_subfound_label(elem.foundation, elem.subfoundation)
    print(f"   Subfoundation label: {label}")


def demo_equation_parsing():
    """Demo parsing equations with extended syntax."""
    print("\n" + "=" * 70)
    print("EQUATION PARSING WITH EXTENDED SYNTAX")
    print("=" * 70)

    equations = [
        "B8^5 +T C3^2",
        "B8_d5 -> D3_a2",
        "B8^5_d5 +T C3^2_a7 -> D6^1_b3",
        "B8 +T B8^5 -> B8_d5 +T B8^5_d5",  # Mixed notation
    ]

    for i, eq in enumerate(equations, 1):
        print(f"\n{i}. Parse: {eq}")
        expr = parse_equation(eq)
        print(f"   Elements: {expr.elements}")
        print(f"   Operators: {expr.ops}")

        # Show element details
        for j, elem_ref in enumerate(expr.element_refs):
            details = [f"World={elem_ref.world}", f"Noetic={elem_ref.noetic}"]
            if elem_ref.sense is not None:
                details.append(f"Sense={elem_ref.sense}")
            if elem_ref.foundation is not None:
                details.append(f"Foundation={elem_ref.foundation}")
            if elem_ref.subfoundation is not None:
                details.append(f"Subfoundation={elem_ref.subfoundation}")
            print(f"   Element {j+1}: {', '.join(details)}")


def demo_canon_validation():
    """Demo canonical validation."""
    print("\n" + "=" * 70)
    print("CANON VALIDATION (Invalid Inputs)")
    print("=" * 70)

    invalid_cases = [
        ("E5", "World E (invalid - must be A/B/C/D)"),
        ("B15", "Noetic 15 (invalid - must be 1-10)"),
        ("B8_d9", "Foundation 9 (invalid - must be 1-7)"),
        ("B8_e5", "Subfoundation world E (invalid - must be A/B/C/D)"),
    ]

    for code, description in invalid_cases:
        print(f"\n{description}")
        print(f"   Attempting to parse: {code}")
        try:
            elem = ElementRef.from_string(code)
            print(f"   ERROR: Should have failed!")
        except ValueError as e:
            print(f"   [OK] Validation failed (as expected): {str(e)[:60]}...")


def demo_real_world_examples():
    """Demo real-world examples from documentation."""
    print("\n" + "=" * 70)
    print("REAL-WORLD EXAMPLES (from TKS v5.0 Manual)")
    print("=" * 70)

    examples = [
        ("A8^1_{6d}", "Spiritual Above with foundation 6 in world D"),
        ("B8^1_{6d}", "Mental Above (career insight) with foundation 6D"),
        ("C8^1_{6d}", "Emotional Above (passionate alignment)"),
        ("D8^1_{6d}", "Physical Above (career action)"),
    ]

    for code, description in examples:
        print(f"\n{description}")
        print(f"   Code: {code}")
        # Note: Need to use underscore format that our parser understands
        # Original format uses {6d} but we support _d6
        # Convert for demo purposes
        code_parsed = code.replace("_{6d}", "_d6")
        try:
            elem = ElementRef.from_string(code_parsed)
            print(f"   [OK] Parsed successfully:")
            print(f"     World: {elem.world}, Noetic: {elem.noetic}, Sense: {elem.sense}")
            print(f"     Foundation: {elem.foundation}, Subfoundation: {elem.subfoundation}")
            if elem.foundation and elem.subfoundation:
                label = get_subfound_label(elem.foundation, elem.subfoundation)
                print(f"     Subfoundation meaning: {label}")
        except Exception as e:
            print(f"   Error: {e}")


if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print(" " * 15 + "TKS EXTENDED PARSER DEMO")
    print(" " * 12 + "Sense/Foundation Suffix Support")
    print("=" * 70)

    demo_sense_notation()
    demo_foundation_notation()
    demo_full_extended()
    demo_equation_parsing()
    demo_canon_validation()
    demo_real_world_examples()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  [OK] Sense suffix parsing (^ and . notation)")
    print("  [OK] Foundation suffix parsing (_Fw notation)")
    print("  [OK] Full extended syntax (B8^5_d5)")
    print("  [OK] Canon validation (worlds A-D, noetics 1-10, foundations 1-7)")
    print("  [OK] Backward compatibility with existing syntax")
    print("  [OK] Equation parsing with mixed notation")
    print()

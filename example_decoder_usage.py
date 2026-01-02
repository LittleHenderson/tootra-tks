"""
Example usage of enhanced decoder with sense labels and sub-foundation labels.
"""

from narrative.decoder import DecodeStory
from narrative.types import TKSExpression, ElementRef

def example_1():
    """Example 1: Using sense labels to distinguish meanings."""
    print("=" * 60)
    print("EXAMPLE 1: Sense Labels")
    print("=" * 60)

    # D5 has two different meanings based on sense:
    # D5.1 = "a woman" (person)
    # D5.2 = "receptacle" (container)

    expr_woman = TKSExpression(
        elements=["D5"],
        element_refs=[ElementRef("D", 5, 1)]  # sense 1
    )
    print(f"D5.1 decodes to: {DecodeStory(expr_woman)}")

    expr_receptacle = TKSExpression(
        elements=["D5"],
        element_refs=[ElementRef("D", 5, 2)]  # sense 2
    )
    print(f"D5.2 decodes to: {DecodeStory(expr_receptacle)}")
    print()


def example_2():
    """Example 2: Sub-foundation context for relationships."""
    print("=" * 60)
    print("EXAMPLE 2: Foundation Context")
    print("=" * 60)

    # Same elements, different foundation contexts

    # Love (C2.3) in emotional relationship context
    expr_relationship = TKSExpression(
        elements=["C2"],
        element_refs=[ElementRef("C", 2, 3)],  # C2.3 = love
        foundations=[(4, "C")]  # F4, C = Emotional relationship
    )
    print(f"Love in relationship context:")
    print(f"  {DecodeStory(expr_relationship)}")

    # Same love, but without foundation context
    expr_no_context = TKSExpression(
        elements=["C2"],
        element_refs=[ElementRef("C", 2, 3)]  # C2.3 = love
    )
    print(f"\nLove without context:")
    print(f"  {DecodeStory(expr_no_context)}")
    print()


def example_3():
    """Example 3: Complex expression with both features."""
    print("=" * 60)
    print("EXAMPLE 3: Combined Features")
    print("=" * 60)

    # Woman + Man in physical companionship context
    expr = TKSExpression(
        elements=["D5", "D6"],
        ops=["+T"],
        element_refs=[
            ElementRef("D", 5, 1),  # D5.1 = a woman
            ElementRef("D", 6, 1)   # D6.1 = a man
        ],
        foundations=[(4, "D")]  # F4, D = Physical companionship
    )

    print("Expression: D5.1 +T D6.1 [F4, D]")
    print(f"Decodes to: {DecodeStory(expr)}")
    print()

    # Compare with different sense (structural elements)
    expr_structural = TKSExpression(
        elements=["D5", "D6"],
        ops=["+T"],
        element_refs=[
            ElementRef("D", 5, 2),  # D5.2 = receptacle
            ElementRef("D", 6, 2)   # D6.2 = structure
        ],
        foundations=[(4, "D")]  # Same foundation context
    )

    print("Expression: D5.2 +T D6.2 [F4, D]")
    print(f"Decodes to: {DecodeStory(expr_structural)}")
    print()


def example_4():
    """Example 4: Different sub-foundation contexts."""
    print("=" * 60)
    print("EXAMPLE 4: Sub-Foundation Variations")
    print("=" * 60)

    # Foundation 4 (Companionship) across different worlds
    contexts = [
        ("A", "Soul connection"),
        ("B", "Intellectual partnership"),
        ("C", "Emotional relationship"),
        ("D", "Physical companionship")
    ]

    for world, expected_label in contexts:
        expr = TKSExpression(
            elements=["C2"],
            element_refs=[ElementRef("C", 2, 3)],  # love
            foundations=[(4, world)]  # F4 in different worlds
        )
        story = DecodeStory(expr)
        print(f"F4, {world} ({expected_label}):")
        print(f"  {story}")
    print()


def example_5():
    """Example 5: Learning vs Knowledge."""
    print("=" * 60)
    print("EXAMPLE 5: Wisdom Context")
    print("=" * 60)

    # B5.1 = learning (process)
    # B5.2 = accumulated knowledge (result)

    expr_learning = TKSExpression(
        elements=["B5"],
        element_refs=[ElementRef("B", 5, 1)],  # sense 1
        foundations=[(2, "B")]  # F2, B = Intellectual wisdom
    )
    print("Learning (B5.1) in intellectual wisdom context:")
    print(f"  {DecodeStory(expr_learning)}")

    expr_knowledge = TKSExpression(
        elements=["B5"],
        element_refs=[ElementRef("B", 5, 2)],  # sense 2
        foundations=[(2, "B")]  # F2, B = Intellectual wisdom
    )
    print("\nAccumulated knowledge (B5.2) in intellectual wisdom context:")
    print(f"  {DecodeStory(expr_knowledge)}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DECODER ENHANCEMENT EXAMPLES")
    print("=" * 60 + "\n")

    example_1()
    example_2()
    example_3()
    example_4()
    example_5()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

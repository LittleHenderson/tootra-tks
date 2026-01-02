#!/usr/bin/env python3
"""
Demo: Semantic Round-trip Fidelity Metrics

This script demonstrates the core functionality of the semantic round-trip
validation system without requiring a trained model.

It shows:
1. TKS element extraction from various formats
2. Element overlap computation
3. Semantic similarity (with fallback if sentence-transformers unavailable)
4. How the fidelity score combines both metrics
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.semantic_roundtrip_metrics import (
    extract_tks_elements,
    compute_element_overlap,
    compute_semantic_similarity,
    _fallback_similarity,
)


def demo_element_extraction():
    """Demonstrate TKS element extraction."""
    print("\n" + "="*70)
    print("DEMO 1: TKS Element Extraction")
    print("="*70 + "\n")

    examples = [
        "Simple equation: A1 ⊕ B5 → C10",
        "Extended notation: A1_d2 ⊕ B5^r3 → C3_d1^r2",
        "Narrative form: She felt fear (A1) and found wisdom (B5)",
        "Mixed case: a1, B5, c3_D2",
        "No elements: This is just regular text",
    ]

    for example in examples:
        elements = extract_tks_elements(example)
        print(f"Input:    {example}")
        print(f"Elements: {elements}")
        print()


def demo_element_overlap():
    """Demonstrate element overlap computation."""
    print("\n" + "="*70)
    print("DEMO 2: Element Overlap (Jaccard Similarity)")
    print("="*70 + "\n")

    pairs = [
        (
            "A1 ⊕ B5 ⊕ C3",
            "A1 and B5 and C3",
            "Identical elements, different syntax"
        ),
        (
            "A1 ⊕ B5 ⊕ C3",
            "A1 ⊕ B5",
            "Partial overlap (2/3)"
        ),
        (
            "A1_d2 ⊕ B5",
            "A1_D2 and B5",
            "Extended notation normalization"
        ),
        (
            "A1 ⊕ B5",
            "C3 ⊕ D10",
            "No overlap"
        ),
    ]

    for eq1, eq2, description in pairs:
        overlap = compute_element_overlap(eq1, eq2)
        print(f"Description: {description}")
        print(f"Equation 1:  {eq1}")
        print(f"Equation 2:  {eq2}")
        print(f"Overlap:     {overlap:.3f}")
        print()


def demo_semantic_similarity():
    """Demonstrate semantic similarity computation."""
    print("\n" + "="*70)
    print("DEMO 3: Semantic Similarity")
    print("="*70 + "\n")

    try:
        # Try with sentence-transformers
        pairs = [
            (
                "She felt fear and anxiety.",
                "She felt fear and anxiety.",
                "Identical text"
            ),
            (
                "She felt fear and anxiety.",
                "She experienced fear and worry.",
                "Same meaning, different words"
            ),
            (
                "Love caused joy in the heart.",
                "Affection brought happiness.",
                "Similar meaning, paraphrased"
            ),
            (
                "She felt fear and anxiety.",
                "He thought about power.",
                "Different meanings"
            ),
        ]

        print("Using sentence-transformers for semantic similarity\n")

        for story1, story2, description in pairs:
            similarity = compute_semantic_similarity(story1, story2)
            print(f"Description: {description}")
            print(f"Story 1:     {story1}")
            print(f"Story 2:     {story2}")
            print(f"Similarity:  {similarity:.3f}")
            print()

    except ImportError:
        print("⚠ sentence-transformers not installed")
        print("Using fallback character-level similarity\n")

        pairs = [
            (
                "She felt fear and anxiety.",
                "She felt fear and anxiety.",
                "Identical text"
            ),
            (
                "HELLO WORLD",
                "hello world",
                "Case insensitive"
            ),
        ]

        for text1, text2, description in pairs:
            similarity = _fallback_similarity(text1, text2)
            print(f"Description: {description}")
            print(f"Text 1:      {text1}")
            print(f"Text 2:      {text2}")
            print(f"Similarity:  {similarity:.3f}")
            print()


def demo_combined_fidelity():
    """Demonstrate combined fidelity scoring."""
    print("\n" + "="*70)
    print("DEMO 4: Combined Fidelity Score")
    print("="*70 + "\n")

    print("Fidelity Score Formula:")
    print("  fidelity = 0.6 × semantic_similarity + 0.4 × element_overlap\n")

    scenarios = [
        {
            'name': 'Perfect Round-trip',
            'semantic': 1.0,
            'element': 1.0,
        },
        {
            'name': 'Good Semantic, Perfect Elements',
            'semantic': 0.85,
            'element': 1.0,
        },
        {
            'name': 'Perfect Semantic, Partial Elements',
            'semantic': 1.0,
            'element': 0.67,
        },
        {
            'name': 'Moderate Both',
            'semantic': 0.70,
            'element': 0.60,
        },
        {
            'name': 'Poor Round-trip',
            'semantic': 0.40,
            'element': 0.30,
        },
    ]

    for scenario in scenarios:
        semantic = scenario['semantic']
        element = scenario['element']
        fidelity = 0.6 * semantic + 0.4 * element

        print(f"{scenario['name']}:")
        print(f"  Semantic Similarity: {semantic:.3f}")
        print(f"  Element Overlap:     {element:.3f}")
        print(f"  → Fidelity Score:    {fidelity:.3f}")

        # Quality assessment
        if fidelity >= 0.8:
            quality = "Excellent ✓"
        elif fidelity >= 0.6:
            quality = "Good"
        elif fidelity >= 0.4:
            quality = "Moderate ⚠"
        else:
            quality = "Poor ✗"

        print(f"  → Quality:           {quality}")
        print()


def demo_realistic_roundtrip():
    """Demonstrate realistic round-trip scenario."""
    print("\n" + "="*70)
    print("DEMO 5: Realistic Round-trip Scenario")
    print("="*70 + "\n")

    original_story = "She felt fear and anxiety in her heart."
    extracted_equation = "A1_d2 ⊕ B5 → C3"
    regenerated_story = "She experienced fear and found wisdom."

    print("Round-trip Process:")
    print(f"1. Original Story:     {original_story}")
    print(f"2. Extracted Equation: {extracted_equation}")
    print(f"3. Regenerated Story:  {regenerated_story}\n")

    # Extract elements
    original_elements = extract_tks_elements(original_story)
    equation_elements = extract_tks_elements(extracted_equation)
    regenerated_elements = extract_tks_elements(regenerated_story)

    print("Element Analysis:")
    print(f"  Original elements:     {original_elements}")
    print(f"  Equation elements:     {equation_elements}")
    print(f"  Regenerated elements:  {regenerated_elements}\n")

    # Compute element overlap (equation vs regenerated)
    element_overlap = compute_element_overlap(
        ' '.join(equation_elements),
        ' '.join(regenerated_elements)
    )

    print(f"Element Overlap: {element_overlap:.3f}\n")

    # Compute semantic similarity
    try:
        semantic_sim = compute_semantic_similarity(
            original_story,
            regenerated_story
        )
        print(f"Semantic Similarity: {semantic_sim:.3f}")
    except ImportError:
        semantic_sim = _fallback_similarity(
            original_story,
            regenerated_story
        )
        print(f"Semantic Similarity (fallback): {semantic_sim:.3f}")

    # Combined fidelity
    fidelity = 0.6 * semantic_sim + 0.4 * element_overlap
    print(f"\nFidelity Score: {fidelity:.3f}")

    if fidelity >= 0.8:
        print("→ Excellent round-trip fidelity ✓")
    elif fidelity >= 0.6:
        print("→ Good round-trip fidelity")
    else:
        print("→ Moderate round-trip fidelity ⚠")


def main():
    """Run all demos."""
    print("="*70)
    print("SEMANTIC ROUND-TRIP METRICS - INTERACTIVE DEMO")
    print("="*70)
    print("\nThis demo shows the core functionality without requiring a trained model.")
    print("For full validation with a model, use:")
    print("  python3 scripts/semantic_roundtrip_metrics.py --model <path>\n")

    try:
        demo_element_extraction()
        demo_element_overlap()
        demo_semantic_similarity()
        demo_combined_fidelity()
        demo_realistic_roundtrip()

        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("1. Run unit tests: python3 tests/test_semantic_roundtrip.py")
        print("2. Validate a model: python3 scripts/semantic_roundtrip_metrics.py --model <path>")
        print("3. Read the guide: docs/SEMANTIC_ROUNDTRIP_GUIDE.md\n")

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

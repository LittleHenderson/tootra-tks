#!/usr/bin/env python3
"""
TKS Dataset Augmentation for Generalization

Based on ACTUAL TKS v7.4 canon - NOT Kabbalah mysticism.

TKS Elements:
- Worlds: A=Spiritual, B=Mental, C=Emotional, D=Physical
- Positions 1-10: Mind, Positive, Negative, Vibration, Female, Male, Rhythm, Cause, Effect, All
- Operators: +T (accumulate), -T (diminish), *T (amplify), /T (attenuate), -> (flow), <- (source), o (cycle)
- Foundations: d1-d7 (Unity, Wisdom, Power, etc.)
- Noetic senses: ^1 through ^10

Usage:
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --multiplier 5
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# =============================================================================
# TKS CANON KNOWLEDGE (v7.4)
# =============================================================================

# The 4 Worlds
WORLDS = {
    "A": "Spiritual",
    "B": "Mental",
    "C": "Emotional",
    "D": "Physical",
}

# The 10 positions (NOT Sephirot - these are TKS-specific)
POSITIONS = {
    "1": ("Mind", "consciousness", "mentalism"),
    "2": ("Positive", "polarity positive", "attraction"),
    "3": ("Negative", "polarity negative", "repulsion"),
    "4": ("Vibration", "vibrational force", "resonance"),
    "5": ("Female", "receptivity", "feminine energy"),
    "6": ("Male", "projection", "masculine energy"),
    "7": ("Rhythm", "rhythmic force", "cyclic energy"),
    "8": ("Cause", "causation", "causal force"),
    "9": ("Effect", "correspondence", "resultant energy"),
    "10": ("All", "totality", "the Idea"),
}

# TKS Operators
OPERATORS = {
    "+T": ("accumulates with", "combines temporally with", "joins with"),
    "-T": ("diminishes from", "removes over time from", "subtracts temporally"),
    "*T": ("amplifies with", "resonates with", "multiplies"),
    "/T": ("attenuates through", "dampens with", "divides by"),
    "+": ("merges with", "joins", "combines with"),
    "-": ("separates from", "subtracts", "removes"),
    "->": ("flows to", "transforms into", "projects to"),
    "<-": ("is sourced by", "receives from", "draws from"),
    "o": ("cycles with", "circulates through", "composes with"),
}

# Foundations (d1-d7)
FOUNDATIONS = {
    "d1": "Unity",
    "d2": "Wisdom",
    "d3": "Understanding",
    "d4": "Companionship",
    "d5": "Power",
    "d6": "Material",
    "d7": "Lust",
}

# =============================================================================
# PARAPHRASE TEMPLATES
# =============================================================================

# Prefix variations
PREFIX_VARIANTS = {
    "Working with": [
        "Working with",
        "Processing",
        "Computing",
        "Evaluating",
        "Operating on",
        "Transforming",
        "Applying",
    ],
    "Interpretation of": [
        "Interpretation of",
        "Understanding",
        "Meaning of",
        "Analysis of",
        "Reading of",
        "Parsing",
    ],
    "Given": [
        "Given",
        "Consider",
        "Looking at",
        "For the pattern",
        "Examining",
        "Starting with",
    ],
}

# Question forms
QUESTION_TEMPLATES = [
    "What does {x} represent?",
    "Interpret the TKS expression: {x}",
    "What is the meaning of {x}?",
    "Explain {x} in TKS terms",
    "How would you describe {x}?",
]

# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================

def get_element_description(elem: str) -> str:
    """Get natural language description of a TKS element."""
    # Parse element like "B4", "C3^2", "D5_d2", "A8^3_d5"
    match = re.match(r'([ABCD])(\d+)(?:\^(\d+))?(?:_d(\d))?', elem)
    if not match:
        return elem

    world = match.group(1)
    pos = match.group(2)
    sense = match.group(3)
    foundation = match.group(4)

    world_name = WORLDS.get(world, world)
    pos_info = POSITIONS.get(pos, (pos, pos, pos))

    desc = f"{world_name}-{pos_info[0]}"

    if sense:
        desc += f" (sense {sense})"
    if foundation:
        found_name = FOUNDATIONS.get(f"d{foundation}", foundation)
        desc += f" with {found_name} foundation"

    return desc


def paraphrase_story(story: str) -> List[str]:
    """Generate paraphrased versions of a TKS story."""
    variations = []

    # Find which prefix type this is
    for prefix, variants in PREFIX_VARIANTS.items():
        if story.startswith(prefix):
            content = story[len(prefix):].strip()
            for variant in variants:
                if variant != prefix:  # Don't include original
                    variations.append(f"{variant} {content}")
            break

    return variations


def create_question_form(story: str) -> List[Dict]:
    """Convert a story into question/answer pairs."""
    questions = []

    # Extract the expression part
    # For "Working with X" or "Interpretation of X"
    for prefix in ["Working with ", "Interpretation of "]:
        if story.startswith(prefix):
            expression = story[len(prefix):].strip()
            for template in QUESTION_TEMPLATES[:2]:  # Limit variations
                q = template.format(x=expression)
                questions.append({
                    "input": q,
                    "output": story,
                    "source": "augmented_question"
                })
            break

    return questions


def add_noise(text: str, prob: float = 0.1) -> str:
    """Add minor noise to text for robustness."""
    if random.random() > prob:
        return text

    words = text.split()
    if len(words) < 4:
        return text

    # Simple noise: swap two adjacent words (not at boundaries)
    if len(words) >= 4:
        idx = random.randint(1, len(words) - 3)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]

    return " ".join(words)


def augment_sample(item: Dict, multiplier: int = 5) -> List[Dict]:
    """Augment a single sample into multiple variations."""
    story = item.get("story", item.get("input", ""))
    source = item.get("source", "augmented")

    if not story:
        return [item]

    augmented = []

    # Keep original
    augmented.append({"story": story, "source": source})

    # Add paraphrased versions
    paraphrases = paraphrase_story(story)
    for p in paraphrases[:multiplier]:
        augmented.append({"story": p, "source": f"{source}_paraphrase"})

    # Add question forms (only for short expressions)
    if len(story) < 100:
        questions = create_question_form(story)
        augmented.extend(questions[:2])

    # Add noisy version (small chance)
    if random.random() < 0.2:
        noisy = add_noise(story, prob=0.5)
        if noisy != story:
            augmented.append({"story": noisy, "source": f"{source}_noisy"})

    return augmented


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Augment TKS dataset (canon-compliant)")
    parser.add_argument("--input", default="output/train_deduplicated.jsonl",
                        help="Input dataset path")
    parser.add_argument("--output", default="output/train_augmented.jsonl",
                        help="Output dataset path")
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Augmentation multiplier (default: 5)")
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Maximum output samples")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        print("Run scripts/deduplicate_dataset.py first")
        return

    print(f"Augmenting {args.input} (TKS canon-compliant)...")
    print(f"Multiplier: {args.multiplier}x")

    # Load original data
    original = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    original.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(original):,} original samples")

    # Augment
    augmented = []
    for item in original:
        samples = augment_sample(item, args.multiplier)
        augmented.extend(samples)

        if len(augmented) >= args.max_samples:
            break

    # Shuffle
    random.shuffle(augmented)

    # Deduplicate
    seen = set()
    final = []
    for item in augmented:
        key = item.get("story", item.get("input", ""))
        if key and key not in seen:
            seen.add(key)
            final.append(item)

    final = final[:args.max_samples]

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nResults:")
    print(f"  Original: {len(original):,}")
    print(f"  Augmented: {len(final):,}")
    print(f"  Multiplier: {len(final)/len(original):.1f}x")
    print(f"\nSaved to: {args.output}")

    # Distribution
    sources = {}
    for item in final:
        src = item.get("source", "unknown")
        base_src = src.split("_")[0] if "_" in src else src
        sources[base_src] = sources.get(base_src, 0) + 1

    print(f"\nSource distribution:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src}: {count:,}")


if __name__ == "__main__":
    main()

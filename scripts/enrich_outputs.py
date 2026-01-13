#!/usr/bin/env python3
"""
TKS Canon-Compliant Output Enrichment Generator

Based on ACTUAL TKS v7.4 terminology - NOT Hebrew Kabbalah.

Generates diverse training samples using correct TKS:
- Worlds: A (Spiritual), B (Mental), C (Emotional), D (Physical)
- Positions: 1-Mind, 2-Positive, 3-Negative, 4-Vibration, 5-Female, 6-Male, 7-Rhythm, 8-Cause, 9-Effect, 10-All
- Operators: +T, -T, *T, /T, ->, <-, o
- Foundations: d1-d7

Usage:
    python scripts/enrich_outputs.py --samples 50000
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse

# =============================================================================
# TKS CANON (v7.4) - NO HEBREW
# =============================================================================

WORLDS = {
    "A": ("Spiritual", "spiritual realm", "the spiritual sphere"),
    "B": ("Mental", "mental plane", "the mental sphere"),
    "C": ("Emotional", "emotional realm", "the emotional sphere"),
    "D": ("Physical", "physical world", "the physical sphere"),
}

POSITIONS = {
    "1": ("Mind", "consciousness", "mentalism"),
    "2": ("Positive", "positive polarity", "attraction"),
    "3": ("Negative", "negative polarity", "repulsion"),
    "4": ("Vibration", "vibrational force", "resonance"),
    "5": ("Female", "feminine energy", "receptivity"),
    "6": ("Male", "masculine energy", "projection"),
    "7": ("Rhythm", "rhythmic force", "cyclic energy"),
    "8": ("Cause", "causal force", "causation"),
    "9": ("Effect", "correspondence", "resultant energy"),
    "10": ("All", "totality", "the Idea"),
}

OPERATORS = {
    "+T": ["accumulates with", "combines temporally with", "joins over time with"],
    "-T": ["diminishes from", "removes over time from", "temporally subtracts"],
    "*T": ["amplifies with", "resonates with", "multiplies temporally"],
    "/T": ["attenuates through", "dampens with", "divides temporally by"],
    "+": ["merges with", "joins", "combines with"],
    "-": ["separates from", "subtracts", "removes"],
    "->": ["flows to", "transforms into", "projects to"],
    "<-": ["is sourced by", "receives from", "draws from"],
    "o": ["cycles with", "circulates through", "composes with"],
}

FOUNDATIONS = {
    "d1": ("Unity", "unified"),
    "d2": ("Wisdom", "wise"),
    "d3": ("Understanding", "comprehending"),
    "d4": ("Companionship", "companion"),
    "d5": ("Power", "powerful"),
    "d6": ("Material", "material"),
    "d7": ("Lust", "desirous"),
}

# =============================================================================
# GENERATORS
# =============================================================================

def get_element_name(code: str) -> str:
    """Get full name for an element like B4 or D9^3_d2."""
    # Parse: World + Position + optional sense + optional foundation
    world = code[0]

    # Extract position (could be 1-10)
    rest = code[1:]
    if rest.startswith("10"):
        pos = "10"
        rest = rest[2:]
    else:
        pos = rest[0]
        rest = rest[1:]

    world_name = WORLDS[world][0]
    pos_name = POSITIONS[pos][0]

    base = f"{world_name}-{pos_name}"

    # Add sense if present
    if "^" in rest:
        sense = rest.split("^")[1].split("_")[0]
        base += f" (sense {sense})"

    # Add foundation if present
    if "_d" in rest:
        found_num = rest.split("_d")[1][0]
        found_name = FOUNDATIONS.get(f"d{found_num}", ("", ""))[0]
        if found_name:
            base += f" with {found_name} foundation"

    return base


def generate_element() -> str:
    """Generate a random TKS element."""
    world = random.choice(["A", "B", "C", "D"])
    pos = random.choice(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

    elem = f"{world}{pos}"

    # Sometimes add sense
    if random.random() < 0.3:
        sense = random.randint(1, 10)
        elem += f"^{sense}"

    # Sometimes add foundation
    if random.random() < 0.2:
        found = random.randint(1, 7)
        elem += f"_d{found}"

    return elem


def generate_chain(length: int = None) -> str:
    """Generate a random TKS operator chain."""
    if length is None:
        length = random.randint(2, 5)

    ops = list(OPERATORS.keys())
    elements = [generate_element() for _ in range(length)]

    chain = elements[0]
    for i in range(1, length):
        op = random.choice(ops)
        chain += f" {op} {elements[i]}"

    return chain


def generate_narrative(chain: str) -> str:
    """Generate natural language narrative for a TKS chain."""
    parts = []

    # Split by operators
    tokens = chain.replace("  ", " ").split(" ")

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Check if it's an element
        if token and token[0] in "ABCD":
            world = token[0]
            world_info = WORLDS[world]

            # Get position
            rest = token[1:]
            if rest.startswith("10"):
                pos = "10"
            else:
                pos = rest[0] if rest else "1"

            pos_info = POSITIONS.get(pos, ("Unknown", "unknown", "unknown"))

            desc = random.choice([
                f"{world_info[0]} {pos_info[0]}",
                f"the {pos_info[1]} of {world_info[1]}",
                f"{pos_info[0]} force in {world_info[2]}",
            ])
            parts.append(desc)

        # Check if it's an operator
        elif token in OPERATORS:
            op_phrases = OPERATORS[token]
            parts.append(random.choice(op_phrases))

        i += 1

    return " ".join(parts)


def generate_samples(num_samples: int) -> List[Dict]:
    """Generate diverse TKS training samples."""
    samples = []

    # Sample types
    types = [
        ("chain_narrative", 0.30),      # Chain -> narrative
        ("narrative_chain", 0.30),      # Narrative -> chain
        ("working_with", 0.15),         # Working with X
        ("interpretation", 0.15),       # Interpretation of X
        ("element_query", 0.10),        # What is X?
    ]

    for sample_type, weight in types:
        count = int(num_samples * weight)

        for _ in range(count):
            chain = generate_chain()
            narrative = generate_narrative(chain)

            if sample_type == "chain_narrative":
                samples.append({
                    "input": f"Given the TKS equation: {chain}\nGenerate a narrative.",
                    "output": narrative,
                    "source": "enriched_chain_narrative"
                })

            elif sample_type == "narrative_chain":
                samples.append({
                    "input": f"Translate to TKS: {narrative}",
                    "output": chain,
                    "source": "enriched_narrative_chain"
                })

            elif sample_type == "working_with":
                samples.append({
                    "story": f"Working with {chain}",
                    "source": "enriched_working"
                })

            elif sample_type == "interpretation":
                samples.append({
                    "story": f"Interpretation of {chain}",
                    "source": "enriched_interpretation"
                })

            elif sample_type == "element_query":
                elem = generate_element()
                name = get_element_name(elem)
                samples.append({
                    "input": f"What is {elem}?",
                    "output": f"{elem} is {name}",
                    "source": "enriched_query"
                })

    random.shuffle(samples)
    return samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate TKS enriched training data")
    parser.add_argument("--output", default="output/train_enriched.jsonl",
                        help="Output path")
    parser.add_argument("--samples", type=int, default=50000,
                        help="Number of samples to generate")
    args = parser.parse_args()

    print(f"Generating {args.samples:,} TKS canon-compliant samples...")

    samples = generate_samples(args.samples)

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nGenerated {len(samples):,} samples")
    print(f"Saved to: {args.output}")

    # Distribution
    sources = {}
    for item in samples:
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\nDistribution:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count:,}")


if __name__ == "__main__":
    main()

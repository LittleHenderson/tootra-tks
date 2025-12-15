#!/usr/bin/env python3
"""
TKS-LLM Pilot Data Generation Script

Generates synthetic training data for the TKS-LLM pilot experiment.
Creates datasets for:
    - Stage 1: Element Prediction (200 examples)
    - Stage 2: Noetic Composition / Involution (200 examples)
    - Stage 3: RPM Prediction (200 examples)
    - Stage 4: Full Pipeline / Multi-task (400 examples)

Usage:
    python scripts/generate_pilot_data.py [--output_dir data/pilot] [--seed 42]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict


# ==============================================================================
# CANONICAL TKS CONSTANTS
# ==============================================================================

# 10 Noetics with canonical names
NOETICS = {
    1: "Mind",
    2: "Positive",
    3: "Negative",
    4: "Vibration",
    5: "Female",
    6: "Male",  # CANONICAL: NOT "MEL"
    7: "Rhythm",
    8: "Cause",  # Above
    9: "Effect",  # Below
    10: "Idea"
}

# 4 Canonical Worlds (A, B, C, D only)
WORLDS = {
    'A': "Spiritual",   # Spiritual World
    'B': "Mental",      # Mental World
    'C': "Emotional",   # Emotional World
    'D': "Physical"     # Physical World
}

# World indices for element encoding
WORLD_OFFSETS = {'A': 0, 'B': 10, 'C': 20, 'D': 30}

# 7 Foundations
FOUNDATIONS = [
    "Light",
    "Darkness",
    "Frequency",
    "Masculine",
    "Feminine",
    "Above",
    "Below"
]

# Noetic-to-Foundation mappings
NOETIC_TO_FOUNDATIONS = {
    2: ["Light"],
    3: ["Darkness"],
    4: ["Frequency"],
    5: ["Feminine"],
    6: ["Masculine"],
    8: ["Above"],
    9: ["Below"],
    1: ["Light", "Darkness"],  # Mind contains both
    7: ["Frequency"],  # Rhythm is frequency-based
    10: ["Light", "Darkness", "Frequency", "Masculine", "Feminine", "Above", "Below"]  # Idea contains all
}

# Involution pairs: compose to ~N10 (Idea)
INVOLUTION_PAIRS = [(2, 3), (5, 6), (8, 9)]

# RPM groupings
RPM_DESIRE_NOETICS = [2, 3]      # Positive, Negative
RPM_WISDOM_NOETICS = [1, 4, 5, 6, 7]  # Mind, Vibration, Female, Male, Rhythm
RPM_POWER_NOETICS = [8, 9]       # Cause, Effect


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Element:
    """A TKS Element (e.g., B4 = Mental-Vibration)."""
    world: str
    noetic: int

    @property
    def symbol(self) -> str:
        return f"{self.world}{self.noetic}"

    @property
    def index(self) -> int:
        return WORLD_OFFSETS[self.world] + (self.noetic - 1)

    @property
    def name(self) -> str:
        return f"{WORLDS[self.world]}-{NOETICS[self.noetic]}"

    @classmethod
    def from_symbol(cls, symbol: str) -> 'Element':
        world = symbol[0].upper()
        noetic = int(symbol[1:])
        return cls(world=world, noetic=noetic)

    @classmethod
    def from_index(cls, index: int) -> 'Element':
        for w, offset in WORLD_OFFSETS.items():
            if offset <= index < offset + 10:
                return cls(world=w, noetic=(index - offset) + 1)
        raise ValueError(f"Invalid index: {index}")

    @classmethod
    def random(cls, rng: random.Random = None) -> 'Element':
        rng = rng or random.Random()
        world = rng.choice(list(WORLDS.keys()))
        noetic = rng.randint(1, 10)
        return cls(world=world, noetic=noetic)


@dataclass
class TKSEquation:
    """A TKS Equation with elements, composition, and interpretation."""
    elements: List[Element]
    composition_type: str  # "additive", "involution", "cascade"
    rpm: Dict[str, float]
    foundations: List[str]
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": [e.symbol for e in self.elements],
            "composition_type": self.composition_type,
            "rpm": self.rpm,
            "foundations": self.foundations,
            "interpretation": self.interpretation
        }


# ==============================================================================
# RPM COMPUTATION
# ==============================================================================

def compute_rpm(elements: List[Element]) -> Dict[str, float]:
    """
    Compute RPM (Desire/Wisdom/Power) from element noetics.

    Desire: N2 (Positive) + N3 (Negative)
    Wisdom: N1 (Mind) + N4 (Vibration) + N5 (Female) + N6 (Male) + N7 (Rhythm)
    Power: N8 (Cause) + N9 (Effect)
    """
    noetics = [e.noetic for e in elements]

    desire_count = sum(1 for n in noetics if n in RPM_DESIRE_NOETICS)
    wisdom_count = sum(1 for n in noetics if n in RPM_WISDOM_NOETICS)
    power_count = sum(1 for n in noetics if n in RPM_POWER_NOETICS)

    total = len(noetics)
    if total == 0:
        return {"desire": 0.0, "wisdom": 0.0, "power": 0.0}

    return {
        "desire": round(desire_count / total, 3),
        "wisdom": round(wisdom_count / total, 3),
        "power": round(power_count / total, 3)
    }


def get_foundations(elements: List[Element]) -> List[str]:
    """Get unique foundations present in elements."""
    foundations = set()
    for e in elements:
        foundations.update(NOETIC_TO_FOUNDATIONS.get(e.noetic, []))
    return sorted(foundations)


# ==============================================================================
# ELEMENT PREDICTION DATA (Stage 1)
# ==============================================================================

def generate_element_prediction_data(
    count: int = 200,
    seed: int = 42
) -> List[Dict]:
    """
    Generate element prediction examples.

    Task: Given partial element info, predict full element.
    Input: Description or partial info
    Output: Element symbol (e.g., "B4")
    """
    rng = random.Random(seed)
    examples = []

    all_elements = [Element(w, n) for w in WORLDS.keys() for n in range(1, 11)]

    for i in range(count):
        element = rng.choice(all_elements)

        # Vary input format
        input_type = rng.choice(["world_noetic", "description", "index", "foundations"])

        if input_type == "world_noetic":
            input_text = f"World {WORLDS[element.world]}, Noetic {NOETICS[element.noetic]}"
        elif input_type == "description":
            input_text = f"The {NOETICS[element.noetic]} force in the {WORLDS[element.world]} plane"
        elif input_type == "index":
            input_text = f"Element at index {element.index} in the 40-element system"
        else:  # foundations
            founds = NOETIC_TO_FOUNDATIONS.get(element.noetic, [])
            if founds:
                input_text = f"Element in {WORLDS[element.world]} with {', '.join(founds)} foundation(s)"
            else:
                input_text = f"Element {element.noetic} in {WORLDS[element.world]}"

        examples.append({
            "id": f"elem_{i:04d}",
            "task": "element_prediction",
            "input": input_text,
            "target_element": element.symbol,
            "target_index": element.index,
            "metadata": {
                "world": element.world,
                "noetic": element.noetic,
                "noetic_name": NOETICS[element.noetic],
                "world_name": WORLDS[element.world]
            }
        })

    return examples


# ==============================================================================
# NOETIC COMPOSITION DATA (Stage 2)
# ==============================================================================

def generate_composition_data(
    count: int = 200,
    seed: int = 42
) -> List[Dict]:
    """
    Generate noetic composition (involution) examples.

    Task: Learn how noetics compose, especially involution pairs.
    Input: Two or more elements
    Output: Composition result (including involution detection)
    """
    rng = random.Random(seed)
    examples = []

    # Ensure good coverage of involution pairs
    involution_count = count // 3
    regular_count = count - involution_count

    # Generate involution examples
    for i in range(involution_count):
        pair = rng.choice(INVOLUTION_PAIRS)
        world = rng.choice(list(WORLDS.keys()))

        elem1 = Element(world, pair[0])
        elem2 = Element(world, pair[1])

        # Involution pairs compose to approximately nu10 (Idea)
        result_element = Element(world, 10)

        examples.append({
            "id": f"comp_inv_{i:04d}",
            "task": "noetic_composition",
            "input_elements": [elem1.symbol, elem2.symbol],
            "is_involution": True,
            "involution_pair": list(pair),
            "result_element": result_element.symbol,
            "result_noetic": 10,
            "rpm": compute_rpm([elem1, elem2]),
            "interpretation": f"Involution of {NOETICS[pair[0]]}-{NOETICS[pair[1]]} yields Idea"
        })

    # Generate regular composition examples
    for i in range(regular_count):
        num_elements = rng.randint(2, 4)
        world = rng.choice(list(WORLDS.keys()))
        noetics = rng.sample(range(1, 11), num_elements)

        elements = [Element(world, n) for n in noetics]

        # Check for any involution pairs
        found_involutions = []
        for pair in INVOLUTION_PAIRS:
            if pair[0] in noetics and pair[1] in noetics:
                found_involutions.append(pair)

        # Dominant noetic (simplified: most influential based on RPM)
        rpm = compute_rpm(elements)
        if rpm["desire"] >= rpm["wisdom"] and rpm["desire"] >= rpm["power"]:
            dominant_type = "desire"
        elif rpm["wisdom"] >= rpm["power"]:
            dominant_type = "wisdom"
        else:
            dominant_type = "power"

        examples.append({
            "id": f"comp_reg_{i:04d}",
            "task": "noetic_composition",
            "input_elements": [e.symbol for e in elements],
            "is_involution": len(found_involutions) > 0,
            "involution_pairs_found": [list(p) for p in found_involutions],
            "dominant_rpm": dominant_type,
            "rpm": rpm,
            "foundations": get_foundations(elements),
            "interpretation": f"Composition of {len(elements)} elements with {dominant_type}-dominant energy"
        })

    rng.shuffle(examples)
    return examples


# ==============================================================================
# RPM PREDICTION DATA (Stage 3)
# ==============================================================================

def generate_rpm_data(
    count: int = 200,
    seed: int = 42
) -> List[Dict]:
    """
    Generate RPM prediction examples.

    Task: Given elements, predict RPM distribution (Desire/Wisdom/Power).
    """
    rng = random.Random(seed)
    examples = []

    for i in range(count):
        num_elements = rng.randint(1, 5)
        elements = [Element.random(rng) for _ in range(num_elements)]

        rpm = compute_rpm(elements)

        # Create natural language description
        if rpm["desire"] > 0.5:
            description = "High emotional charge from Positive/Negative interplay"
        elif rpm["wisdom"] > 0.5:
            description = "Strong cognitive/spiritual content from Mind/Vibration/Gender/Rhythm"
        elif rpm["power"] > 0.5:
            description = "Causal energy dominant from Cause/Effect axis"
        else:
            description = "Balanced RPM distribution across components"

        examples.append({
            "id": f"rpm_{i:04d}",
            "task": "rpm_prediction",
            "input_elements": [e.symbol for e in elements],
            "input_noetics": [e.noetic for e in elements],
            "target_rpm": rpm,
            "dominant_component": max(rpm, key=rpm.get),
            "interpretation": description,
            "metadata": {
                "element_count": num_elements,
                "unique_noetics": len(set(e.noetic for e in elements)),
                "unique_worlds": len(set(e.world for e in elements))
            }
        })

    return examples


# ==============================================================================
# FULL PIPELINE / MULTI-TASK DATA (Stage 4)
# ==============================================================================

def generate_equation_templates() -> List[Dict]:
    """Generate canonical equation templates for multi-task learning."""
    return [
        {
            "pattern": "manifestation",
            "required_noetics": [10, 8, 9],  # Idea + Cause + Effect
            "description": "Manifestation pattern: Idea descends through Cause to Effect"
        },
        {
            "pattern": "transformation",
            "required_noetics": [2, 3, 10],  # Positive + Negative -> Idea (involution)
            "description": "Transformation via polarity resolution"
        },
        {
            "pattern": "creative_force",
            "required_noetics": [5, 6, 4],  # Female + Male + Vibration
            "description": "Creative force through gender polarity in vibration"
        },
        {
            "pattern": "rhythmic_balance",
            "required_noetics": [7, 2, 3],  # Rhythm + Positive + Negative
            "description": "Rhythmic oscillation between polarities"
        },
        {
            "pattern": "mental_causation",
            "required_noetics": [1, 8, 9],  # Mind + Cause + Effect
            "description": "Mental causation: Mind directs Cause-Effect axis"
        }
    ]


def generate_full_pipeline_data(
    count: int = 400,
    seed: int = 42
) -> List[Dict]:
    """
    Generate full pipeline examples combining all tasks.

    Tasks:
        - E2I: Equation to Interpretation
        - I2E: Interpretation to Equation
        - E2RPM: Equation to RPM
        - S2E: Scenario to Equation
    """
    rng = random.Random(seed)
    examples = []
    templates = generate_equation_templates()

    task_types = ["E2I", "I2E", "E2RPM", "S2E"]

    for i in range(count):
        task_type = rng.choice(task_types)
        template = rng.choice(templates)
        world = rng.choice(list(WORLDS.keys()))

        # Build elements from template
        elements = [Element(world, n) for n in template["required_noetics"]]
        # Optionally add extra elements
        if rng.random() < 0.3:
            extra_noetic = rng.randint(1, 10)
            if extra_noetic not in template["required_noetics"]:
                elements.append(Element(world, extra_noetic))

        rpm = compute_rpm(elements)
        foundations = get_foundations(elements)

        # Check for involutions
        noetics_present = [e.noetic for e in elements]
        involutions_found = [
            list(pair) for pair in INVOLUTION_PAIRS
            if pair[0] in noetics_present and pair[1] in noetics_present
        ]

        equation = TKSEquation(
            elements=elements,
            composition_type=template["pattern"],
            rpm=rpm,
            foundations=foundations,
            interpretation=template["description"]
        )

        # Build task-specific example
        example_base = {
            "id": f"full_{task_type}_{i:04d}",
            "task": f"full_pipeline_{task_type}",
            "pattern": template["pattern"],
            "world": world,
            "world_name": WORLDS[world],
            "has_involution": len(involutions_found) > 0,
            "involution_pairs": involutions_found
        }

        if task_type == "E2I":
            # Equation -> Interpretation
            example_base.update({
                "input_elements": [e.symbol for e in elements],
                "target_interpretation": template["description"],
                "target_rpm": rpm,
                "target_foundations": foundations
            })

        elif task_type == "I2E":
            # Interpretation -> Equation
            example_base.update({
                "input_interpretation": template["description"],
                "input_pattern": template["pattern"],
                "target_elements": [e.symbol for e in elements],
                "target_noetics": template["required_noetics"]
            })

        elif task_type == "E2RPM":
            # Equation -> RPM
            example_base.update({
                "input_elements": [e.symbol for e in elements],
                "target_rpm": rpm,
                "target_dominant": max(rpm, key=rpm.get)
            })

        elif task_type == "S2E":
            # Scenario -> Equation
            scenarios = {
                "manifestation": "I want to bring an idea into physical reality",
                "transformation": "I need to resolve a conflict between opposing forces",
                "creative_force": "I want to create something new through partnership",
                "rhythmic_balance": "I need to find balance in fluctuating circumstances",
                "mental_causation": "I want my thoughts to influence outcomes"
            }
            example_base.update({
                "input_scenario": scenarios.get(template["pattern"], template["description"]),
                "target_elements": [e.symbol for e in elements],
                "target_pattern": template["pattern"],
                "target_rpm": rpm
            })

        examples.append(example_base)

    return examples


# ==============================================================================
# MAIN GENERATION FUNCTION
# ==============================================================================

def generate_all_pilot_data(output_dir: str = "data/pilot", seed: int = 42):
    """Generate all pilot training data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating TKS-LLM pilot data to {output_path}")
    print(f"Random seed: {seed}")
    print("-" * 50)

    # Stage 1: Element Prediction
    print("Stage 1: Generating element prediction data (200 examples)...")
    stage1_data = generate_element_prediction_data(count=200, seed=seed)
    stage1_file = output_path / "stage1_elements.jsonl"
    with open(stage1_file, 'w', encoding='utf-8') as f:
        for example in stage1_data:
            f.write(json.dumps(example) + '\n')
    print(f"  -> Saved to {stage1_file}")

    # Stage 2: Noetic Composition
    print("Stage 2: Generating noetic composition data (200 examples)...")
    stage2_data = generate_composition_data(count=200, seed=seed + 1)
    stage2_file = output_path / "stage2_composition.jsonl"
    with open(stage2_file, 'w', encoding='utf-8') as f:
        for example in stage2_data:
            f.write(json.dumps(example) + '\n')
    print(f"  -> Saved to {stage2_file}")

    # Stage 3: RPM Prediction
    print("Stage 3: Generating RPM prediction data (200 examples)...")
    stage3_data = generate_rpm_data(count=200, seed=seed + 2)
    stage3_file = output_path / "stage3_rpm.jsonl"
    with open(stage3_file, 'w', encoding='utf-8') as f:
        for example in stage3_data:
            f.write(json.dumps(example) + '\n')
    print(f"  -> Saved to {stage3_file}")

    # Stage 4: Full Pipeline
    print("Stage 4: Generating full pipeline data (400 examples)...")
    stage4_data = generate_full_pipeline_data(count=400, seed=seed + 3)
    stage4_file = output_path / "stage4_full_pipeline.jsonl"
    with open(stage4_file, 'w', encoding='utf-8') as f:
        for example in stage4_data:
            f.write(json.dumps(example) + '\n')
    print(f"  -> Saved to {stage4_file}")

    # Combined dataset
    print("Generating combined dataset...")
    combined_data = stage1_data + stage2_data + stage3_data + stage4_data
    combined_file = output_path / "combined_all.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for example in combined_data:
            f.write(json.dumps(example) + '\n')
    print(f"  -> Saved to {combined_file}")

    # Metadata
    metadata = {
        "total_examples": len(combined_data),
        "stage_counts": {
            "stage1_elements": len(stage1_data),
            "stage2_composition": len(stage2_data),
            "stage3_rpm": len(stage3_data),
            "stage4_full_pipeline": len(stage4_data)
        },
        "seed": seed,
        "canonical_noetics": NOETICS,
        "canonical_worlds": WORLDS,
        "involution_pairs": INVOLUTION_PAIRS,
        "rpm_groupings": {
            "desire": RPM_DESIRE_NOETICS,
            "wisdom": RPM_WISDOM_NOETICS,
            "power": RPM_POWER_NOETICS
        }
    }
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> Metadata saved to {metadata_file}")

    print("-" * 50)
    print(f"Total examples generated: {len(combined_data)}")
    print("Data generation complete!")

    return {
        "output_dir": str(output_path),
        "stage1": str(stage1_file),
        "stage2": str(stage2_file),
        "stage3": str(stage3_file),
        "stage4": str(stage4_file),
        "combined": str(combined_file),
        "metadata": str(metadata_file)
    }


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate TKS-LLM pilot training data"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="data/pilot",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generate_all_pilot_data(
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

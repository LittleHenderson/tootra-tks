#!/usr/bin/env python3
"""
Phase 1 Training Data Generator
================================

Generates comprehensive training data for TKS model scaling:
1. Grounding data (ID lookups, definitions, element identification)
2. Contrastive examples (disambiguation between similar elements)
3. TKS notation (mappings, goals, operations)
4. Explanations (TKS → NL and NL → TKS)
5. Dialogues and reasoning chains

Supports curriculum learning: easy → medium → hard progression.

Usage:
    python scripts/generate_phase1_data.py --output datasets/phase1 --samples 50000
"""

import json
import random
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Seed for reproducibility
random.seed(42)


# =============================================================================
# CANONICAL DATA
# =============================================================================

WORLDS = [
    {"id": "A", "name": "Spiritual", "description": "World of Emanation/Causes"},
    {"id": "B", "name": "Mental", "description": "World of Creation/Thoughts"},
    {"id": "C", "name": "Emotional", "description": "World of Formation/Feelings"},
    {"id": "D", "name": "Physical", "description": "World of Action/Manifestation"},
]

NOETICS = [
    {"id": "^0", "name": "Idea", "meaning": "The seed concept"},
    {"id": "^1", "name": "Mind", "meaning": "Attention and Focus"},
    {"id": "^2", "name": "Positive", "meaning": "Attraction/Addition"},
    {"id": "^3", "name": "Negative", "meaning": "Repulsion/Subtraction"},
    {"id": "^4", "name": "Vibration", "meaning": "Intensity/Resonance"},
    {"id": "^5", "name": "Female", "meaning": "Authority/Reservoir/Identity"},
    {"id": "^6", "name": "Male", "meaning": "Component/Related-Set"},
    {"id": "^7", "name": "Rhythm", "meaning": "Pattern/Repetition"},
    {"id": "^8", "name": "Above", "meaning": "Cause/Trigger"},
    {"id": "^9", "name": "Below", "meaning": "Effect/Outcome"},
]

ELEMENTS = [
    # Physical World (D)
    {"symbol": "D0", "name": "Physical Template", "definition": "Concept of a physical thing", "world": "D"},
    {"symbol": "D1", "name": "Body Awareness", "definition": "Physical sensation awareness", "world": "D"},
    {"symbol": "D2", "name": "Physical Health", "definition": "Health, wellness, bodily order", "world": "D"},
    {"symbol": "D3", "name": "Physical Illness", "definition": "Sickness, disorder, decay", "world": "D"},
    {"symbol": "D4", "name": "Physical Energy", "definition": "Energy level, vitality", "world": "D"},
    {"symbol": "D5", "name": "Physical Woman", "definition": "Woman, female person", "world": "D"},
    {"symbol": "D6", "name": "Physical Man", "definition": "Man, male person", "world": "D"},
    {"symbol": "D7", "name": "Physical Habit", "definition": "Habits, routines", "world": "D"},
    {"symbol": "D8", "name": "Physical Elevation", "definition": "Status upgrade, material improvement", "world": "D"},
    {"symbol": "D9", "name": "Physical Foundation", "definition": "Foundation, base, basics", "world": "D"},
    # Emotional World (C)
    {"symbol": "C0", "name": "Emotional Concept", "definition": "The idea of an emotion", "world": "C"},
    {"symbol": "C1", "name": "Emotional Awareness", "definition": "Awareness of feelings", "world": "C"},
    {"symbol": "C2", "name": "Joy/Happiness", "definition": "Positive emotions, happiness", "world": "C"},
    {"symbol": "C3", "name": "Fear", "definition": "Fear, anxiety, negative emotions", "world": "C"},
    {"symbol": "C4", "name": "Emotional Intensity", "definition": "How strongly felt", "world": "C"},
    {"symbol": "C5", "name": "Emotional Receptivity", "definition": "Emotional openness", "world": "C"},
    {"symbol": "C6", "name": "Emotional Expression", "definition": "Expressing feelings", "world": "C"},
    {"symbol": "C7", "name": "Mood Cycle", "definition": "Mood patterns", "world": "C"},
    {"symbol": "C8", "name": "Emotional Trigger", "definition": "What triggers feelings", "world": "C"},
    {"symbol": "C9", "name": "Emotional Response", "definition": "Emotional reactions", "world": "C"},
    # Mental World (B)
    {"symbol": "B0", "name": "Thought Form", "definition": "A specific idea/concept", "world": "B"},
    {"symbol": "B1", "name": "Meta-Cognition", "definition": "Thinking about thinking", "world": "B"},
    {"symbol": "B2", "name": "Positive Belief", "definition": "Empowering beliefs", "world": "B"},
    {"symbol": "B3", "name": "Limiting Belief", "definition": "Limiting beliefs", "world": "B"},
    {"symbol": "B4", "name": "Thought Intensity", "definition": "Strength of thought", "world": "B"},
    {"symbol": "B5", "name": "Learning Receptivity", "definition": "Openness to learning", "world": "B"},
    {"symbol": "B6", "name": "Logical Thinking", "definition": "Logic, reasoning", "world": "B"},
    {"symbol": "B7", "name": "Thought Pattern", "definition": "Mental patterns", "world": "B"},
    {"symbol": "B8", "name": "Thought Trigger", "definition": "What triggers thoughts", "world": "B"},
    {"symbol": "B9", "name": "Cognitive Response", "definition": "Thought reactions", "world": "B"},
    # Spiritual World (A)
    {"symbol": "A0", "name": "Divine Blueprint", "definition": "Life purpose, soul mission", "world": "A"},
    {"symbol": "A1", "name": "Soul Awareness", "definition": "Spiritual awareness", "world": "A"},
    {"symbol": "A2", "name": "Spiritual Alignment", "definition": "Being on path", "world": "A"},
    {"symbol": "A3", "name": "Spiritual Misalignment", "definition": "Being off-path", "world": "A"},
    {"symbol": "A4", "name": "Spiritual Frequency", "definition": "Spiritual vibration level", "world": "A"},
    {"symbol": "A5", "name": "Divine Receptivity", "definition": "Openness to guidance", "world": "A"},
    {"symbol": "A6", "name": "Divine Will", "definition": "Spiritual will in action", "world": "A"},
    {"symbol": "A7", "name": "Karmic Pattern", "definition": "Karma, spiritual lessons", "world": "A"},
    {"symbol": "A8", "name": "Divine Cause", "definition": "Spiritual causation", "world": "A"},
    {"symbol": "A9", "name": "Spiritual Effect", "definition": "Spiritual outcomes", "world": "A"},
]

OPERATORS = {
    "+": {"name": "Association", "semantic": "and", "verb": "combines with"},
    "-": {"name": "Subtraction", "semantic": "minus", "verb": "removes"},
    "+T": {"name": "Association Tootra", "semantic": "together with", "verb": "links with"},
    "-T": {"name": "Disassociation", "semantic": "without", "verb": "separates from"},
    "*T": {"name": "Intensification", "semantic": "intensified by", "verb": "amplifies"},
    "/T": {"name": "Opposition", "semantic": "in conflict with", "verb": "opposes"},
    "->": {"name": "Causal", "semantic": "causes", "verb": "leads to"},
    "<-": {"name": "Reverse Causal", "semantic": "caused by", "verb": "is caused by"},
}

# Build lookup dictionaries
ELEMENT_BY_SYMBOL = {e["symbol"]: e for e in ELEMENTS}
ELEMENT_BY_NAME = {e["name"].lower(): e for e in ELEMENTS}
WORLD_BY_ID = {w["id"]: w for w in WORLDS}


# =============================================================================
# DATA GENERATORS
# =============================================================================

class GroundingGenerator:
    """Generate grounding data for semantic learning."""

    @staticmethod
    def id_lookup_forward() -> List[Dict]:
        """What is the ID for [name]? -> [symbol]"""
        samples = []
        templates = [
            "What is the TKS ID for {name}?",
            "What symbol represents {name} in TKS?",
            "Give the TKS code for {name}.",
            "TKS ID for {name}:",
            "{name} in TKS notation:",
        ]
        for elem in ELEMENTS:
            for template in templates:
                samples.append({
                    "input": template.format(name=elem["name"]),
                    "output": elem["symbol"],
                    "type": "grounding_id_forward",
                    "element": elem["symbol"],
                })
        return samples

    @staticmethod
    def id_lookup_reverse() -> List[Dict]:
        """What does [symbol] mean? -> [name]"""
        samples = []
        templates = [
            "What does {symbol} mean in TKS?",
            "Define {symbol}.",
            "What is {symbol}?",
            "{symbol} represents:",
            "The meaning of {symbol} is:",
        ]
        for elem in ELEMENTS:
            for template in templates:
                samples.append({
                    "input": template.format(symbol=elem["symbol"]),
                    "output": elem["name"],
                    "type": "grounding_id_reverse",
                    "element": elem["symbol"],
                })
        return samples

    @staticmethod
    def definition_lookup() -> List[Dict]:
        """Full definition queries."""
        samples = []
        templates = [
            "Define {symbol} in detail.",
            "What is the full meaning of {symbol}?",
            "Explain what {symbol} represents.",
        ]
        for elem in ELEMENTS:
            for template in templates:
                output = f"{elem['name']}: {elem['definition']}"
                samples.append({
                    "input": template.format(symbol=elem["symbol"]),
                    "output": output,
                    "type": "grounding_definition",
                    "element": elem["symbol"],
                })
        return samples

    @staticmethod
    def world_identification() -> List[Dict]:
        """Which world does [symbol] belong to?"""
        samples = []
        templates = [
            "Which world does {symbol} belong to?",
            "What world is {symbol} in?",
            "{symbol} is in which world?",
        ]
        for elem in ELEMENTS:
            world = WORLD_BY_ID[elem["world"]]
            for template in templates:
                samples.append({
                    "input": template.format(symbol=elem["symbol"]),
                    "output": f"{world['id']} ({world['name']})",
                    "type": "grounding_world",
                    "element": elem["symbol"],
                })
        return samples

    @staticmethod
    def noetic_explanation() -> List[Dict]:
        """Explain noetic levels."""
        samples = []
        for noetic in NOETICS:
            templates = [
                f"What does {noetic['id']} mean in TKS?",
                f"Explain noetic level {noetic['id']}.",
                f"Define {noetic['id']}:",
            ]
            output = f"{noetic['name']}: {noetic['meaning']}"
            for template in templates:
                samples.append({
                    "input": template,
                    "output": output,
                    "type": "grounding_noetic",
                    "noetic": noetic["id"],
                })
        return samples


class ContrastiveGenerator:
    """Generate contrastive examples for disambiguation."""

    @staticmethod
    def same_world_contrast() -> List[Dict]:
        """Differentiate elements within the same world."""
        samples = []
        for world_id in "ABCD":
            world_elements = [e for e in ELEMENTS if e["world"] == world_id]
            for i, elem1 in enumerate(world_elements):
                for elem2 in world_elements[i+1:]:
                    # What's the difference between X and Y?
                    samples.append({
                        "input": f"What's the difference between {elem1['symbol']} and {elem2['symbol']}?",
                        "output": f"{elem1['symbol']} is {elem1['name']} ({elem1['definition']}), while {elem2['symbol']} is {elem2['name']} ({elem2['definition']}).",
                        "type": "contrastive_same_world",
                        "elements": [elem1["symbol"], elem2["symbol"]],
                    })
                    # Which one represents [definition]?
                    samples.append({
                        "input": f"Which TKS element represents '{elem1['definition']}'? {elem1['symbol']} or {elem2['symbol']}?",
                        "output": elem1["symbol"],
                        "type": "contrastive_choice",
                        "elements": [elem1["symbol"], elem2["symbol"]],
                    })
        return samples

    @staticmethod
    def cross_world_contrast() -> List[Dict]:
        """Differentiate similar concepts across worlds."""
        samples = []
        # Pair similar concepts across worlds (same index)
        for idx in range(10):
            elements_at_idx = [e for e in ELEMENTS if e["symbol"].endswith(str(idx))]
            for i, elem1 in enumerate(elements_at_idx):
                for elem2 in elements_at_idx[i+1:]:
                    samples.append({
                        "input": f"Compare {elem1['symbol']} and {elem2['symbol']}.",
                        "output": f"{elem1['symbol']} ({elem1['world']} world): {elem1['name']}. {elem2['symbol']} ({elem2['world']} world): {elem2['name']}. They're in different worlds but share the same index.",
                        "type": "contrastive_cross_world",
                        "elements": [elem1["symbol"], elem2["symbol"]],
                    })
        return samples

    @staticmethod
    def positive_negative_contrast() -> List[Dict]:
        """Contrast positive/negative pairs."""
        pairs = [
            ("D2", "D3"),  # Health vs Illness
            ("C2", "C3"),  # Joy vs Fear
            ("B2", "B3"),  # Positive Belief vs Limiting Belief
            ("A2", "A3"),  # Alignment vs Misalignment
        ]
        samples = []
        for pos, neg in pairs:
            pos_elem = ELEMENT_BY_SYMBOL[pos]
            neg_elem = ELEMENT_BY_SYMBOL[neg]
            samples.append({
                "input": f"What's the opposite of {pos} in TKS?",
                "output": neg,
                "type": "contrastive_opposite",
                "elements": [pos, neg],
            })
            samples.append({
                "input": f"What's the opposite of {neg} in TKS?",
                "output": pos,
                "type": "contrastive_opposite",
                "elements": [neg, pos],
            })
            samples.append({
                "input": f"Contrast {pos} and {neg}.",
                "output": f"{pos} ({pos_elem['name']}) represents the positive aspect, while {neg} ({neg_elem['name']}) represents the negative aspect in the {pos_elem['world']} world.",
                "type": "contrastive_polarity",
                "elements": [pos, neg],
            })
        return samples


class TKSNotationGenerator:
    """Generate TKS notation training data."""

    @staticmethod
    def mapping() -> List[Dict]:
        """Map concept to TKS symbol."""
        samples = []
        templates = [
            "Map concept: {name}",
            "TKS mapping for {name}:",
            "Express {name} in TKS:",
            "What is the TKS for {name}?",
        ]
        for elem in ELEMENTS:
            for template in templates:
                samples.append({
                    "input": template.format(name=f"{elem['symbol']}.1 {elem['name']}"),
                    "output": f"({elem['symbol']}^0)",
                    "type": "mapping",
                    "element": elem["symbol"],
                })
        return samples

    @staticmethod
    def goals() -> List[Dict]:
        """Generate goal expressions (pathways)."""
        samples = []
        # Generate random goal pairs
        for _ in range(500):
            elem1 = random.choice(ELEMENTS)
            elem2 = random.choice(ELEMENTS)
            if elem1["symbol"] != elem2["symbol"]:
                samples.append({
                    "input": f"Goal: {elem1['symbol']}.1 {elem1['name']}",
                    "output": f"({elem1['symbol']}:{elem2['symbol']})",
                    "type": "goal",
                    "elements": [elem1["symbol"], elem2["symbol"]],
                })
        return samples

    @staticmethod
    def operations() -> List[Dict]:
        """Generate TKS operations with operators."""
        samples = []
        verbs = {
            "+": ["joining with", "combined with", "and"],
            "-": ["minus", "without", "removing"],
            "+T": ["together with", "linked to", "associated with"],
            "-T": ["separated from", "disassociated from", "apart from"],
            "*T": ["intensified by", "amplified by", "multiplied with"],
            "/T": ["in conflict with", "opposing", "divided by"],
            "->": ["causing", "leading to", "transforming into"],
            "<-": ["caused by", "resulting from", "triggered by"],
        }

        for op, verb_list in verbs.items():
            for _ in range(50):
                elem1 = random.choice(ELEMENTS)
                elem2 = random.choice(ELEMENTS)
                if elem1["symbol"] != elem2["symbol"]:
                    noetic = random.randint(1, 5)
                    verb = random.choice(verb_list)
                    samples.append({
                        "input": f"TKS Operation: {elem1['symbol']}.1 {elem1['name']} {verb} {elem2['symbol']}.1 {elem2['name']}",
                        "output": f"({elem1['symbol']}^{noetic}) {op} ({elem2['symbol']}^{noetic})",
                        "type": "operation",
                        "operator": op,
                    })
        return samples


class ExplanationGenerator:
    """Generate explanation and interpretation data."""

    @staticmethod
    def interpret_expression() -> List[Dict]:
        """Interpret TKS expressions in natural language."""
        samples = []

        # Simple expressions
        for elem in ELEMENTS:
            for noetic in NOETICS[:5]:  # ^0 to ^4
                expr = f"({elem['symbol']}{noetic['id']})"
                explanation = f"The {noetic['name']} of {elem['name']}."
                samples.append({
                    "input": f"Interpret: {expr}",
                    "output": explanation,
                    "type": "interpretation",
                    "expression": expr,
                })

        # Binary expressions with operators
        for _ in range(300):
            elem1 = random.choice(ELEMENTS)
            elem2 = random.choice(ELEMENTS)
            if elem1["symbol"] != elem2["symbol"]:
                op = random.choice(list(OPERATORS.keys()))
                op_info = OPERATORS[op]
                noetic = random.randint(1, 4)

                expr = f"({elem1['symbol']}^{noetic}) {op} ({elem2['symbol']}^{noetic})"
                explanation = f"{elem1['name']} {op_info['verb']} {elem2['name']}."

                samples.append({
                    "input": f"Interpret: {expr}",
                    "output": explanation,
                    "type": "interpretation",
                    "expression": expr,
                })

        return samples

    @staticmethod
    def reasoning_chains() -> List[Dict]:
        """Generate step-by-step reasoning."""
        samples = []
        scenarios = [
            {
                "question": "How to express 'positive thinking leads to physical health' in TKS?",
                "reasoning": "1. Positive thinking = B2 (Positive Belief)\n2. Physical health = D2 (Physical Health)\n3. 'leads to' = -> (causal operator)\n4. Expression: (B2^3) -> (D2^3)",
                "final": "(B2^3) -> (D2^3)",
            },
            {
                "question": "Express 'fear blocks emotional expression' in TKS.",
                "reasoning": "1. Fear = C3 (Fear)\n2. Emotional expression = C6 (Emotional Expression)\n3. 'blocks' = -T (disassociation)\n4. Expression: (C3^3) -T (C6^3)",
                "final": "(C3^3) -T (C6^3)",
            },
            {
                "question": "How to show 'spiritual alignment intensifies joy'?",
                "reasoning": "1. Spiritual alignment = A2 (Spiritual Alignment)\n2. Joy = C2 (Joy/Happiness)\n3. 'intensifies' = *T (intensification)\n4. Expression: (A2^4) *T (C2^4)",
                "final": "(A2^4) *T (C2^4)",
            },
        ]

        for scenario in scenarios:
            samples.append({
                "input": scenario["question"],
                "output": scenario["reasoning"],
                "type": "reasoning",
                "final_expression": scenario["final"],
            })

        # Generate more reasoning chains
        templates = [
            ("combining", "+T", "together"),
            ("removing", "-T", "separate"),
            ("intensifying", "*T", "amplify"),
            ("opposing", "/T", "conflict"),
        ]

        for verb, op, action in templates:
            for _ in range(30):
                elem1 = random.choice(ELEMENTS)
                elem2 = random.choice(ELEMENTS)
                if elem1["symbol"] != elem2["symbol"]:
                    noetic = random.randint(2, 4)
                    samples.append({
                        "input": f"How to express '{elem1['name']} {verb} with {elem2['name']}' in TKS?",
                        "output": f"1. {elem1['name']} = {elem1['symbol']}\n2. {elem2['name']} = {elem2['symbol']}\n3. '{verb}' = {op}\n4. Expression: ({elem1['symbol']}^{noetic}) {op} ({elem2['symbol']}^{noetic})",
                        "type": "reasoning",
                        "final_expression": f"({elem1['symbol']}^{noetic}) {op} ({elem2['symbol']}^{noetic})",
                    })

        return samples


class DialogueGenerator:
    """Generate dialogue training data."""

    @staticmethod
    def qa_dialogues() -> List[Dict]:
        """Generate Q&A style dialogues."""
        samples = []

        # Element lookup dialogues
        for elem in random.sample(ELEMENTS, 20):
            samples.append({
                "messages": [
                    {"role": "user", "content": f"What is {elem['symbol']}?"},
                    {"role": "assistant", "content": f"{elem['symbol']} is {elem['name']} in the {elem['world']} ({WORLD_BY_ID[elem['world']]['name']}) world. It represents: {elem['definition']}."},
                ],
                "type": "dialogue_qa",
            })

        # Follow-up dialogues
        for elem in random.sample(ELEMENTS, 10):
            elem2 = random.choice([e for e in ELEMENTS if e["world"] == elem["world"] and e != elem])
            samples.append({
                "messages": [
                    {"role": "user", "content": f"Tell me about {elem['symbol']}."},
                    {"role": "assistant", "content": f"{elem['symbol']} represents {elem['name']} - {elem['definition']}."},
                    {"role": "user", "content": f"What else is in the same world?"},
                    {"role": "assistant", "content": f"In the {elem['world']} ({WORLD_BY_ID[elem['world']]['name']}) world, you also have {elem2['symbol']} ({elem2['name']}) which represents {elem2['definition']}."},
                ],
                "type": "dialogue_followup",
            })

        return samples


# =============================================================================
# MAIN GENERATOR
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    total_samples: int = 50000
    ratios: Dict[str, float] = field(default_factory=lambda: {
        "grounding": 0.35,      # ID lookups, definitions
        "contrastive": 0.15,    # Disambiguation
        "notation": 0.25,       # TKS mappings, goals, operations
        "explanation": 0.15,    # Interpretations, reasoning
        "dialogue": 0.10,       # Q&A dialogues
    })
    output_dir: str = "datasets/phase1"
    seed: int = 42


def generate_dataset(config: DatasetConfig) -> Dict[str, List[Dict]]:
    """Generate complete Phase 1 dataset."""
    random.seed(config.seed)

    print("Generating grounding data...")
    grounding = []
    grounding.extend(GroundingGenerator.id_lookup_forward())
    grounding.extend(GroundingGenerator.id_lookup_reverse())
    grounding.extend(GroundingGenerator.definition_lookup())
    grounding.extend(GroundingGenerator.world_identification())
    grounding.extend(GroundingGenerator.noetic_explanation())

    print("Generating contrastive data...")
    contrastive = []
    contrastive.extend(ContrastiveGenerator.same_world_contrast())
    contrastive.extend(ContrastiveGenerator.cross_world_contrast())
    contrastive.extend(ContrastiveGenerator.positive_negative_contrast())

    print("Generating TKS notation data...")
    notation = []
    notation.extend(TKSNotationGenerator.mapping())
    notation.extend(TKSNotationGenerator.goals())
    notation.extend(TKSNotationGenerator.operations())

    print("Generating explanation data...")
    explanation = []
    explanation.extend(ExplanationGenerator.interpret_expression())
    explanation.extend(ExplanationGenerator.reasoning_chains())

    print("Generating dialogue data...")
    dialogue = []
    dialogue.extend(DialogueGenerator.qa_dialogues())

    # Combine and sample to target size
    all_data = {
        "grounding": grounding,
        "contrastive": contrastive,
        "notation": notation,
        "explanation": explanation,
        "dialogue": dialogue,
    }

    return all_data


def sample_to_target(data: Dict[str, List[Dict]], config: DatasetConfig) -> List[Dict]:
    """Sample data to target size with ratios."""
    final_samples = []

    for category, samples in data.items():
        target_count = int(config.total_samples * config.ratios.get(category, 0.1))
        if len(samples) >= target_count:
            selected = random.sample(samples, target_count)
        else:
            # Repeat samples if not enough
            selected = samples * (target_count // len(samples) + 1)
            selected = selected[:target_count]

        print(f"  {category}: {len(selected)} samples")
        final_samples.extend(selected)

    random.shuffle(final_samples)
    return final_samples


def convert_to_training_format(samples: List[Dict], sep: str = " => ") -> List[Dict]:
    """Convert samples to training format with input/output."""
    training_data = []

    for sample in samples:
        if "messages" in sample:
            # Dialogue format - convert to single input/output
            messages = sample["messages"]
            if len(messages) >= 2:
                # Use first exchange
                training_data.append({
                    "input": messages[0]["content"],
                    "output": messages[1]["content"],
                    "type": sample.get("type", "dialogue"),
                })
                # Also add full dialogue as separate sample
                if len(messages) > 2:
                    full_input = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
                    full_output = messages[-1]["content"]
                    training_data.append({
                        "input": full_input,
                        "output": full_output,
                        "type": sample.get("type", "dialogue") + "_multi",
                    })
        else:
            # Standard format
            training_data.append({
                "input": sample["input"],
                "output": sample["output"],
                "type": sample.get("type", "unknown"),
            })

    return training_data


def save_dataset(samples: List[Dict], output_dir: str, name: str = "phase1_training"):
    """Save dataset to JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split into train/val/test
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split_name, split_data in splits.items():
        filepath = output_path / f"{name}_{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in split_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Saved {split_name}: {len(split_data)} samples -> {filepath}")

    # Also save combined file
    combined_path = output_path / f"{name}_combined.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved combined: {len(samples)} samples -> {combined_path}")

    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "splits": {k: len(v) for k, v in splits.items()},
        "types": {},
    }
    for sample in samples:
        t = sample.get("type", "unknown")
        metadata["types"][t] = metadata["types"].get(t, 0) + 1

    with open(output_path / f"{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 1 training data")
    parser.add_argument("--output", type=str, default="datasets/phase1", help="Output directory")
    parser.add_argument("--samples", type=int, default=50000, help="Target number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grounding-ratio", type=float, default=0.35, help="Grounding data ratio")
    parser.add_argument("--contrastive-ratio", type=float, default=0.15, help="Contrastive data ratio")
    args = parser.parse_args()

    config = DatasetConfig(
        total_samples=args.samples,
        output_dir=args.output,
        seed=args.seed,
        ratios={
            "grounding": args.grounding_ratio,
            "contrastive": args.contrastive_ratio,
            "notation": 0.25,
            "explanation": 0.15,
            "dialogue": 0.10,
        }
    )

    print("=" * 60)
    print("PHASE 1 DATA GENERATION")
    print("=" * 60)
    print(f"Target samples: {config.total_samples}")
    print(f"Output: {config.output_dir}")
    print(f"Ratios: {config.ratios}")
    print("=" * 60)

    # Generate raw data
    raw_data = generate_dataset(config)

    print("\nRaw data generated:")
    for category, samples in raw_data.items():
        print(f"  {category}: {len(samples)} samples")

    # Sample to target size
    print("\nSampling to target size...")
    sampled = sample_to_target(raw_data, config)

    # Convert to training format
    print("\nConverting to training format...")
    training_data = convert_to_training_format(sampled)

    # Save
    print("\nSaving dataset...")
    splits = save_dataset(training_data, config.output_dir)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(training_data)}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Output: {config.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
World/RPM Labeled Training Data Generator

Generates training examples with explicit:
- World labels (A=Spiritual, B=Mental, C=Emotional, D=Physical)
- RPM labels (D=Desire, W=Wisdom, P=Power)

TKS Canon Constraints:
    - 4 Worlds: A (Divine/Spiritual), B (Mental/Cognitive), C (Emotional), D (Physical)
    - Desire noetics: 1, 4, 7 (Mind, Vibration, Rhythm) - the MVR protocol
    - Wisdom noetics: 5, 6 (Female/Receptivity, Male/Structure)
    - Power noetics: 8, 9 (Cause/Above, Effect/Below)

Output Format:
    {"input": "...", "target": "...", "world_label": "A", "rpm_label": "desire"}

Author: TKS-LLM Agent A
Date: 2025-12-22
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


# =============================================================================
# TKS CANONICAL DEFINITIONS (FROZEN - DO NOT MODIFY)
# =============================================================================

WORLDS = {
    "A": {"name": "Atziluth", "domain": "Spiritual", "keywords": [
        "divine", "soul", "eternal", "purpose", "meaning", "archetypal",
        "spirit", "transcendent", "sacred", "holy", "unity", "source",
        "cosmic", "infinite", "essence", "enlightenment", "awakening",
        # Test phrases from noetic regression tests
        "divine realm", "spiritual awareness"
    ]},
    "B": {"name": "Briah", "domain": "Mental", "keywords": [
        "thought", "belief", "idea", "concept", "analysis", "logic",
        "reason", "understanding", "knowledge", "learning", "thinking",
        "cognitive", "intellectual", "theory", "hypothesis", "mind", "mental",
        # Test phrases from noetic regression tests
        "mental focus", "cognitive clarity", "concentration"
    ]},
    "C": {"name": "Yetzirah", "domain": "Emotional", "keywords": [
        "feeling", "emotion", "desire", "love", "fear", "joy", "anger",
        "intuition", "sensation", "passion", "heart", "empathy", "mood",
        "affection", "attachment", "longing", "sentiment",
        # Test phrases from noetic regression tests
        "emotional flow", "feeling warmth"
    ]},
    "D": {"name": "Assiah", "domain": "Physical", "keywords": [
        "body", "action", "matter", "material", "physical", "concrete",
        "earth", "touch", "movement", "health", "strength", "environment",
        "tangible", "manifest", "practical", "real", "sensory",
        # Test phrases from noetic regression tests
        "physical power", "bodily energy", "vitality"
    ]},
}

# FROZEN MVR Protocol: Desire = {1, 4, 7}
DESIRE_NOETICS = [1, 4, 7]  # Mind, Vibration, Rhythm
WISDOM_NOETICS = [5, 6]     # Female (receptivity), Male (structure)
POWER_NOETICS = [8, 9]      # Cause (above), Effect (below)

# RPM category keywords for scenario generation
RPM_KEYWORDS = {
    "desire": [
        "want", "goal", "aspire", "wish", "crave", "seek", "pursue",
        "motivation", "drive", "ambition", "longing", "hunger", "thirst",
        "yearn", "aim", "intend", "objective", "target", "dream",
        # RPM test phrases
        "desire", "intention", "manifestation"
    ],
    "wisdom": [
        "know", "understand", "learn", "insight", "aware", "perceive",
        "discern", "comprehend", "grasp", "realize", "recognize", "informed",
        "wise", "enlightened", "knowledgeable", "educated", "studied",
        # RPM test phrases
        "balance", "equilibrium", "understanding", "male/female principles"
    ],
    "power": [
        "can", "able", "capable", "actualize", "manifest", "create",
        "cause", "effect", "produce", "generate", "achieve", "accomplish",
        "execute", "perform", "implement", "realize", "bring about",
        # RPM test phrases
        "cause and effect", "consequence", "forces", "action"
    ],
}


# =============================================================================
# SCENARIO TEMPLATES
# =============================================================================

@dataclass
class LabeledExample:
    """A training example with explicit world and RPM labels."""
    input: str
    target: str
    world_label: str  # A, B, C, or D (primary world)
    rpm_label: str    # desire, wisdom, or power
    world_weights: Dict[str, float]  # Soft labels for all worlds
    rpm_weights: Dict[str, float]    # Soft labels for all RPM categories

    def to_dict(self) -> Dict:
        return asdict(self)


# World-specific scenario templates
WORLD_SCENARIOS = {
    "A": [
        # Pure spiritual scenarios (A should dominate)
        ("A person meditates on the nature of {A_keyword}", "A1+A4+A7"),
        ("The soul contemplates {A_keyword} essence", "A1+A5+A6"),
        ("Divine {A_keyword} flows through consciousness", "A1+A2+A3"),
        ("Seeking spiritual {A_keyword} and enlightenment", "A1+A4+A7+A8"),
        ("The eternal {A_keyword} transcends all form", "A1+A0+A5"),
        ("Unity with {A_keyword} brings peace", "A1+A2+A5+A6"),
        ("Sacred {A_keyword} illuminates the path", "A0+A1+A8+A9"),
    ],
    "B": [
        # Pure mental scenarios (B should dominate)
        ("Analyzing the {B_keyword} reveals patterns", "B1+B4+B7"),
        ("The mind considers {B_keyword} carefully", "B1+B5+B6"),
        ("Logical {B_keyword} leads to conclusions", "B1+B8+B9"),
        ("Understanding {B_keyword} requires study", "B1+B4+B5"),
        ("Cognitive {B_keyword} processes information", "B1+B2+B3"),
        ("Mental {B_keyword} shapes our beliefs", "B1+B5+B6+B7"),
        ("Thinking about {B_keyword} clarifies ideas", "B1+B4+B8"),
    ],
    "C": [
        # Pure emotional scenarios (C should dominate)
        ("Feeling deep {C_keyword} in the heart", "C1+C4+C7"),
        ("Emotional {C_keyword} moves the soul", "C1+C2+C3"),
        ("The intuition senses {C_keyword}", "C1+C5+C6"),
        ("Passion for {C_keyword} drives action", "C1+C4+C8+C9"),
        ("Love of {C_keyword} transforms", "C1+C2+C5+C6"),
        ("Sentiment about {C_keyword} runs deep", "C1+C4+C7"),
        ("Desire for {C_keyword} is strong", "C1+C4+C7+C2"),
    ],
    "D": [
        # Pure physical scenarios (D should dominate)
        ("The body engages in {D_keyword} activity", "D1+D4+D7"),
        ("Physical {D_keyword} manifests in matter", "D1+D8+D9"),
        ("Material {D_keyword} takes concrete form", "D1+D2+D3"),
        ("Action produces {D_keyword} results", "D1+D4+D8+D9"),
        ("Tangible {D_keyword} can be touched", "D1+D5+D6"),
        ("Environment shows {D_keyword} effects", "D1+D4+D7+D9"),
        ("Practical {D_keyword} works in reality", "D1+D8+D9"),
    ],
}

# Pure NL templates (no equation shown)
PURE_NL_SCENARIOS = {
    "A": [
        "Divine {A_keyword} fills the soul",
        "Spiritual {A_keyword} transcends material reality",
        "{A_keyword} illuminates the sacred path",
        "The eternal {A_keyword} guides consciousness",
        "Pure {A_keyword} awakens spiritual truth",
    ],
    "B": [
        "A person focuses mentally on {B_keyword}",
        "{B_keyword} requires deep concentration",
        "The mind analyzes {B_keyword} carefully",
        "Thinking about {B_keyword} brings clarity",
        "Cognitive understanding of {B_keyword} develops",
    ],
    "C": [
        "Feeling warmth and {C_keyword} in the heart",
        "{C_keyword} flows through emotional channels",
        "Deep {C_keyword} moves the soul",
        "The heart opens to {C_keyword}",
        "Emotional {C_keyword} transforms experience",
    ],
    "D": [
        "Physical {D_keyword} manifests in the body",
        "Bodily {D_keyword} creates tangible results",
        "{D_keyword} expresses through action",
        "The material world shows {D_keyword}",
        "Concrete {D_keyword} takes form",
    ],
}

# RPM-specific scenario templates (NL-style)
RPM_SCENARIOS = {
    "desire": [
        # Desire-dominant scenarios (D/W/P where D >> W, P)
        ("I want to achieve {desire_kw} in my life", "desire"),
        ("My goal is to obtain {desire_kw}", "desire"),
        ("Strong motivation drives pursuit of {desire_kw}", "desire"),
        ("Craving {desire_kw} intensifies daily", "desire"),
        ("The ambition for {desire_kw} is powerful", "desire"),
        ("Seeking {desire_kw} with all my heart", "desire"),
        ("Longing for {desire_kw} consumes thought", "desire"),
    ],
    "wisdom": [
        # Wisdom-dominant scenarios (D/W/P where W >> D, P)
        ("Understanding {wisdom_kw} reveals truth", "wisdom"),
        ("Knowledge of {wisdom_kw} is essential", "wisdom"),
        ("Insight into {wisdom_kw} changes perspective", "wisdom"),
        ("Learning about {wisdom_kw} illuminates", "wisdom"),
        ("Awareness of {wisdom_kw} brings clarity", "wisdom"),
        ("Comprehending {wisdom_kw} takes study", "wisdom"),
        ("Wisdom about {wisdom_kw} guides decisions", "wisdom"),
    ],
    "power": [
        # Power-dominant scenarios (D/W/P where P >> D, W)
        ("Able to manifest {power_kw} into reality", "power"),
        ("The capability to produce {power_kw}", "power"),
        ("Creating {power_kw} through direct action", "power"),
        ("Causing {power_kw} to come into being", "power"),
        ("Accomplishing {power_kw} through effort", "power"),
        ("Executing {power_kw} with precision", "power"),
        ("Generating {power_kw} from resources", "power"),
    ],
}

# Equation-style RPM templates (pure equations using canonical noetic indices)
# Desire = nu1, nu4, nu7 (Mind, Vibration, Rhythm)
# Wisdom = nu5, nu6 (Female, Male)
# Power = nu8, nu9 (Cause, Effect)
EQUATION_RPM_TEMPLATES = {
    "desire": [
        # Desire noetics in each world
        "{world}1+{world}4+{world}7",
        "{world}1+{world}4",
        "{world}4+{world}7",
        "{world}1+{world}7",
    ],
    "wisdom": [
        # Wisdom noetics in each world
        "{world}5+{world}6",
        "{world}5",
        "{world}6",
    ],
    "power": [
        # Power noetics in each world
        "{world}8+{world}9",
        "{world}8",
        "{world}9",
    ],
}


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_world_labeled_example(
    primary_world: str,
    secondary_world: Optional[str] = None,
    rpm_bias: Optional[str] = None,
    nl_style: str = "mixed"
) -> LabeledExample:
    """
    Generate a training example with explicit world labels.

    Args:
        primary_world: The world that should dominate (A, B, C, or D)
        secondary_world: Optional secondary world influence
        rpm_bias: Optional RPM category bias (desire, wisdom, power)
        nl_style: Style of example - "equation" (pure equation), "mixed" (NL+equation), or "pure_nl" (pure NL)

    Returns:
        LabeledExample with input, target, and labels
    """
    keyword = random.choice(WORLDS[primary_world]["keywords"])

    if nl_style == "pure_nl":
        # Pure NL: natural language input and target
        nl_template = random.choice(PURE_NL_SCENARIOS[primary_world])
        input_text = nl_template.format(**{f"{primary_world}_keyword": keyword})
        # Generate appropriate equation but use NL description as target
        base_equation = random.choice([t[1] for t in WORLD_SCENARIOS[primary_world]])
        target_equation = input_text  # Target is also NL for pure_nl mode
    elif nl_style == "equation":
        # Pure equation: equation-only input and target
        templates = WORLD_SCENARIOS[primary_world]
        _, base_equation = random.choice(templates)
        input_text = base_equation
        target_equation = base_equation
    else:  # mixed
        # Mixed: NL input with equation target (original behavior)
        templates = WORLD_SCENARIOS[primary_world]
        template, base_equation = random.choice(templates)
        input_text = template.format(**{f"{primary_world}_keyword": keyword})
        target_equation = base_equation

    # Add secondary world influence if specified (not for pure_nl targets)
    if secondary_world and secondary_world != primary_world and nl_style != "pure_nl":
        sec_noetics = random.sample([1, 4, 5, 6, 7], k=2)
        sec_elements = "+".join([f"{secondary_world}{n}" for n in sec_noetics])
        target_equation = f"{target_equation}+{sec_elements}"

    # Calculate world weights (soft labels)
    # Primary world gets highest weight, others get decaying weights
    world_weights = {}
    ordinals = {"A": 0, "B": 1, "C": 2, "D": 3}
    primary_ord = ordinals[primary_world]

    for w in ["A", "B", "C", "D"]:
        w_ord = ordinals[w]
        distance = abs(w_ord - primary_ord)
        if distance == 0:
            world_weights[w] = 0.7  # Primary world
        elif distance == 1:
            world_weights[w] = 0.15  # Adjacent world
        else:
            world_weights[w] = 0.075  # Distant world

    if secondary_world:
        world_weights[secondary_world] = min(0.4, world_weights[secondary_world] + 0.25)
        # Renormalize
        total = sum(world_weights.values())
        world_weights = {k: v / total for k, v in world_weights.items()}

    # Determine RPM label based on elements in equation
    rpm_label, rpm_weights = determine_rpm_from_equation(target_equation, rpm_bias)

    return LabeledExample(
        input=input_text,
        target=target_equation,
        world_label=primary_world,
        rpm_label=rpm_label,
        world_weights=world_weights,
        rpm_weights=rpm_weights,
    )


def generate_rpm_labeled_example(
    rpm_category: str,
    world_context: Optional[str] = None,
    nl_style: str = "mixed"
) -> LabeledExample:
    """
    Generate a training example with explicit RPM labels.

    Args:
        rpm_category: The RPM category (desire, wisdom, power)
        world_context: Optional world to anchor the scenario
        nl_style: Style of example - "equation" (pure equation), "mixed" (NL+equation), or "pure_nl" (pure NL)

    Returns:
        LabeledExample with input, target, and labels
    """
    # Build target equation based on RPM category
    world = world_context or random.choice(["A", "B", "C", "D"])

    if rpm_category == "desire":
        # Use MVR noetics: 1, 4, 7
        noetics = DESIRE_NOETICS
        weights = {"desire": 0.8, "wisdom": 0.1, "power": 0.1}
    elif rpm_category == "wisdom":
        # Use wisdom noetics: 5, 6
        noetics = WISDOM_NOETICS
        weights = {"desire": 0.1, "wisdom": 0.8, "power": 0.1}
    else:  # power
        # Use power noetics: 8, 9
        noetics = POWER_NOETICS
        weights = {"desire": 0.1, "wisdom": 0.1, "power": 0.8}

    elements = [f"{world}{n}" for n in noetics]
    target_equation = "+".join(elements)

    # Generate input based on nl_style
    if nl_style == "equation":
        # Pure equation: use equation templates
        eq_template = random.choice(EQUATION_RPM_TEMPLATES[rpm_category])
        input_text = eq_template.format(world=world)
        # Target is same equation format
        target_equation = input_text
    elif nl_style == "pure_nl":
        # Pure NL: NL input and NL target
        templates = RPM_SCENARIOS[rpm_category]
        template, _ = random.choice(templates)
        keyword = random.choice(RPM_KEYWORDS[rpm_category])
        input_text = template.format(**{f"{rpm_category}_kw": keyword})
        target_equation = input_text  # NL target for pure_nl mode
    else:  # mixed (default)
        # Mixed: NL input with equation target
        templates = RPM_SCENARIOS[rpm_category]
        template, _ = random.choice(templates)
        keyword = random.choice(RPM_KEYWORDS[rpm_category])
        input_text = template.format(**{f"{rpm_category}_kw": keyword})
        # target_equation already set above

    # Calculate world weights (uniform if not specified)
    if world_context:
        world_weights = {w: 0.1 for w in ["A", "B", "C", "D"]}
        world_weights[world_context] = 0.7
    else:
        world_weights = {w: 0.25 for w in ["A", "B", "C", "D"]}

    return LabeledExample(
        input=input_text,
        target=target_equation,
        world_label=world,
        rpm_label=rpm_category,
        world_weights=world_weights,
        rpm_weights=weights,
    )


def determine_rpm_from_equation(
    equation: str,
    rpm_bias: Optional[str] = None
) -> Tuple[str, Dict[str, float]]:
    """
    Determine RPM category from equation elements.

    Args:
        equation: TKS equation string (e.g., "A1+A4+A7")
        rpm_bias: Optional bias toward a specific category

    Returns:
        (rpm_label, rpm_weights) tuple
    """
    # Count noetics in each RPM category
    desire_count = 0
    wisdom_count = 0
    power_count = 0

    # Parse equation for noetic indices
    import re
    elements = re.findall(r'[ABCD](\d+)', equation)

    for noetic_str in elements:
        noetic = int(noetic_str)
        if noetic in DESIRE_NOETICS:
            desire_count += 1
        elif noetic in WISDOM_NOETICS:
            wisdom_count += 1
        elif noetic in POWER_NOETICS:
            power_count += 1

    # Apply bias if specified
    if rpm_bias == "desire":
        desire_count += 2
    elif rpm_bias == "wisdom":
        wisdom_count += 2
    elif rpm_bias == "power":
        power_count += 2

    # Calculate weights
    total = desire_count + wisdom_count + power_count + 0.01  # Avoid division by zero
    rpm_weights = {
        "desire": (desire_count + 0.1) / (total + 0.3),
        "wisdom": (wisdom_count + 0.1) / (total + 0.3),
        "power": (power_count + 0.1) / (total + 0.3),
    }

    # Normalize
    total_weights = sum(rpm_weights.values())
    rpm_weights = {k: v / total_weights for k, v in rpm_weights.items()}

    # Determine primary label
    rpm_label = max(rpm_weights, key=rpm_weights.get)

    return rpm_label, rpm_weights


def generate_contrastive_world_pair(
    world_a: str,
    world_b: str,
    nl_style: str = "mixed"
) -> Tuple[LabeledExample, LabeledExample]:
    """
    Generate a contrastive pair for world separation training.

    The same semantic content with different world emphasis.

    Args:
        world_a: First world
        world_b: Second world
        nl_style: Style of example - "equation", "mixed", or "pure_nl"
    """
    # Generate base example for world_a
    example_a = generate_world_labeled_example(world_a, nl_style=nl_style)

    # Create contrasting example for world_b with similar theme
    # but different world emphasis
    example_b = generate_world_labeled_example(world_b, nl_style=nl_style)

    return example_a, example_b


def generate_contrastive_rpm_triplet(
    nl_style: str = "mixed"
) -> Tuple[LabeledExample, LabeledExample, LabeledExample]:
    """
    Generate a contrastive triplet for RPM differentiation.

    Returns three examples: one each for desire, wisdom, power
    with the same world context for clear RPM separation.

    Args:
        nl_style: Style of example - "equation", "mixed", or "pure_nl"
    """
    world = random.choice(["A", "B", "C", "D"])

    desire_ex = generate_rpm_labeled_example("desire", world_context=world, nl_style=nl_style)
    wisdom_ex = generate_rpm_labeled_example("wisdom", world_context=world, nl_style=nl_style)
    power_ex = generate_rpm_labeled_example("power", world_context=world, nl_style=nl_style)

    return desire_ex, wisdom_ex, power_ex


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_world_rpm_dataset(
    num_examples: int = 1000,
    world_balance: bool = True,
    rpm_balance: bool = True,
    include_contrastive: bool = True,
    contrastive_ratio: float = 0.3,
    nl_ratio: float = 0.4,
) -> List[LabeledExample]:
    """
    Generate a complete labeled dataset for world/RPM supervision.

    Args:
        num_examples: Total number of examples to generate
        world_balance: Ensure balanced world distribution
        rpm_balance: Ensure balanced RPM distribution
        include_contrastive: Include contrastive pairs/triplets
        contrastive_ratio: Fraction of examples that are contrastive
        nl_ratio: Ratio of natural language examples (pure_nl + mixed).
                  Remaining ratio is pure equation. Default 0.4 means 40% NL, 60% equation.

    Returns:
        List of LabeledExample objects
    """
    examples = []

    # Calculate distribution
    num_contrastive = int(num_examples * contrastive_ratio) if include_contrastive else 0
    num_standard = num_examples - num_contrastive

    # Generate standard examples
    worlds = ["A", "B", "C", "D"]
    rpm_cats = ["desire", "wisdom", "power"]

    for i in range(num_standard):
        if world_balance:
            world = worlds[i % 4]
        else:
            world = random.choice(worlds)

        if rpm_balance:
            rpm = rpm_cats[i % 3]
        else:
            rpm = random.choice(rpm_cats)

        # Determine NL style based on nl_ratio
        # Split NL ratio: half pure_nl, half mixed, rest equation
        rand_val = random.random()
        if rand_val < nl_ratio / 2:
            nl_style = "pure_nl"
        elif rand_val < nl_ratio:
            nl_style = "mixed"
        else:
            nl_style = "equation"

        # Alternate between world-focused and RPM-focused generation
        # Pass nl_style to BOTH functions so nl_ratio applies uniformly
        if i % 2 == 0:
            example = generate_world_labeled_example(world, rpm_bias=rpm, nl_style=nl_style)
        else:
            example = generate_rpm_labeled_example(rpm, world_context=world, nl_style=nl_style)

        examples.append(example)

    # Generate contrastive examples
    if include_contrastive:
        # World contrastive pairs
        num_world_pairs = num_contrastive // 3
        for _ in range(num_world_pairs):
            w1, w2 = random.sample(worlds, 2)
            # Apply nl_ratio to contrastive examples too
            rand_val = random.random()
            if rand_val < nl_ratio / 2:
                nl_style = "pure_nl"
            elif rand_val < nl_ratio:
                nl_style = "mixed"
            else:
                nl_style = "equation"
            ex_a, ex_b = generate_contrastive_world_pair(w1, w2, nl_style=nl_style)
            examples.extend([ex_a, ex_b])

        # RPM contrastive triplets
        num_rpm_triplets = num_contrastive // 3
        for _ in range(num_rpm_triplets):
            # Apply nl_ratio to contrastive RPM triplets too
            rand_val = random.random()
            if rand_val < nl_ratio / 2:
                nl_style = "pure_nl"
            elif rand_val < nl_ratio:
                nl_style = "mixed"
            else:
                nl_style = "equation"
            d_ex, w_ex, p_ex = generate_contrastive_rpm_triplet(nl_style=nl_style)
            examples.extend([d_ex, w_ex, p_ex])

    # Shuffle
    random.shuffle(examples)

    return examples[:num_examples]


def save_dataset(
    examples: List[LabeledExample],
    output_path: str,
    format: str = "jsonl"
):
    """
    Save dataset to file.

    Args:
        examples: List of LabeledExample objects
        output_path: Output file path
        format: Output format (jsonl or json)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([ex.to_dict() for ex in examples], f, indent=2)

    print(f"Saved {len(examples)} examples to {output_path}")


def print_statistics(examples: List[LabeledExample]):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # World distribution
    world_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for ex in examples:
        world_counts[ex.world_label] += 1

    print("\nWorld Distribution:")
    for w, count in world_counts.items():
        pct = 100 * count / len(examples)
        print(f"  {w} ({WORLDS[w]['domain']}): {count} ({pct:.1f}%)")

    # RPM distribution
    rpm_counts = {"desire": 0, "wisdom": 0, "power": 0}
    for ex in examples:
        rpm_counts[ex.rpm_label] += 1

    print("\nRPM Distribution:")
    for r, count in rpm_counts.items():
        pct = 100 * count / len(examples)
        print(f"  {r}: {count} ({pct:.1f}%)")

    # NL style distribution (categorize based on input/target patterns)
    nl_style_counts = {"pure_equation": 0, "mixed_nl": 0, "pure_nl": 0}
    for ex in examples:
        # Heuristic: if input == target and starts with letter, it's pure equation
        if ex.input == ex.target and ex.input and ex.input[0] in "ABCD":
            nl_style_counts["pure_equation"] += 1
        # If target == input and doesn't start with equation pattern, it's pure NL
        elif ex.input == ex.target:
            nl_style_counts["pure_nl"] += 1
        # Otherwise it's mixed
        else:
            nl_style_counts["mixed_nl"] += 1

    print("\nNL Style Distribution:")
    for style, count in nl_style_counts.items():
        pct = 100 * count / len(examples)
        print(f"  {style}: {count} ({pct:.1f}%)")

    # Sample examples
    print("\nSample Examples:")
    for i, ex in enumerate(random.sample(examples, min(5, len(examples)))):
        print(f"\n  Example {i+1}:")
        print(f"    Input: {ex.input[:60]}...")
        print(f"    Target: {ex.target}")
        print(f"    World: {ex.world_label} | RPM: {ex.rpm_label}")
        print(f"    World weights: A={ex.world_weights['A']:.2f}, B={ex.world_weights['B']:.2f}, "
              f"C={ex.world_weights['C']:.2f}, D={ex.world_weights['D']:.2f}")
        print(f"    RPM weights: D={ex.rpm_weights['desire']:.2f}, "
              f"W={ex.rpm_weights['wisdom']:.2f}, P={ex.rpm_weights['power']:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate World/RPM Labeled Training Data for TKS"
    )
    parser.add_argument(
        '--output', '-o',
        default='data/world_rpm_labeled.jsonl',
        help='Output file path'
    )
    parser.add_argument(
        '--num-examples', '-n',
        type=int,
        default=2000,
        help='Number of examples to generate'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['jsonl', 'json'],
        default='jsonl',
        help='Output format'
    )
    parser.add_argument(
        '--contrastive-ratio',
        type=float,
        default=0.3,
        help='Fraction of contrastive examples'
    )
    parser.add_argument(
        '--no-contrastive',
        action='store_true',
        help='Disable contrastive examples'
    )
    parser.add_argument(
        '--nl-ratio',
        type=float,
        default=0.4,
        help='Ratio of natural language examples (pure_nl + mixed). Default 0.4 = 40%% NL, 60%% equation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress statistics output'
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    print("=" * 60)
    print("TKS World/RPM Labeled Data Generator")
    print("=" * 60)
    print(f"\nGenerating {args.num_examples} labeled examples...")
    print(f"Contrastive ratio: {args.contrastive_ratio if not args.no_contrastive else 0}")
    print(f"NL ratio: {args.nl_ratio} ({args.nl_ratio*100:.0f}% NL, {(1-args.nl_ratio)*100:.0f}% equation)")

    # Generate dataset
    examples = generate_world_rpm_dataset(
        num_examples=args.num_examples,
        world_balance=True,
        rpm_balance=True,
        include_contrastive=not args.no_contrastive,
        contrastive_ratio=args.contrastive_ratio,
        nl_ratio=args.nl_ratio,
    )

    # Print statistics
    if not args.quiet:
        print_statistics(examples)

    # Save
    save_dataset(examples, args.output, args.format)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

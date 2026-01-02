#!/usr/bin/env python3
"""
Training Data Batch Generator - Agent C

Generates diverse high-quality training pairs for TKS model training:
1. equation_to_interpretation: Equation -> Natural language story
2. interpretation_to_equation: Natural language -> Equation
3. equation_to_rpm: Equation -> D/W/P classification
4. equation_to_foundations: Equation -> Foundation analysis

TKS Canon Compliance:
- Worlds: A (Spiritual), B (Mental), C (Emotional), D (Physical)
- Noetics: 1-10 only
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (9 operators)
- Senses: 1-9 only
- Foundations: _d1 through _d7 only
- RPM: Desire {1,4,7}, Wisdom {5,6}, Power {8,9}

Author: Agent C
Date: 2025-12-31
"""

import json
import random
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import Counter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tks_rules.rpm import RPM_MAPPINGS, RPM_DESIRE, RPM_WISDOM, RPM_POWER
    from tks_rules.foundations import FOUNDATIONS, FOUNDATION_NAMES
    from tks_rules.worlds import WORLDS
    from tks_rules.noetics import NOETICS
except ImportError:
    # Fallback definitions
    WORLDS = {'A': 'Spiritual', 'B': 'Mental', 'C': 'Emotional', 'D': 'Physical'}
    RPM_MAPPINGS = {'desire': {1, 4, 7}, 'wisdom': {5, 6}, 'power': {8, 9}}
    RPM_DESIRE = {1, 4, 7}
    RPM_WISDOM = {5, 6}
    RPM_POWER = {8, 9}
    FOUNDATION_NAMES = {1: 'Unity', 2: 'Wisdom', 3: 'Life', 4: 'Companionship',
                        5: 'Power', 6: 'Material', 7: 'Lust'}


# ==============================================================================
# CANONICAL CONSTANTS
# ==============================================================================

CANONICAL_OPERATORS = ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']
VALID_SENSES = list(range(1, 10))  # 1-9
VALID_FOUNDATIONS = ['_d1', '_d2', '_d3', '_d4', '_d5', '_d6', '_d7']
VALID_NOETICS = list(range(1, 11))  # 1-10

WORLD_DESCRIPTIONS = {
    'A': {'name': 'Spiritual', 'realm': 'the spiritual realm', 'essence': 'spiritual essence',
          'domain': 'Atziluth', 'qualities': ['divine', 'eternal', 'transcendent', 'sacred']},
    'B': {'name': 'Mental', 'realm': 'the mental plane', 'essence': 'mental energy',
          'domain': 'Briah', 'qualities': ['cognitive', 'intellectual', 'logical', 'conceptual']},
    'C': {'name': 'Emotional', 'realm': 'the emotional sphere', 'essence': 'emotional force',
          'domain': 'Yetzirah', 'qualities': ['feeling', 'intuitive', 'passionate', 'empathic']},
    'D': {'name': 'Physical', 'realm': 'the physical world', 'essence': 'physical manifestation',
          'domain': 'Assiah', 'qualities': ['tangible', 'material', 'concrete', 'embodied']}
}

NOETIC_DESCRIPTIONS = {
    1: {'name': 'Mind', 'type': 'mentalism', 'quality': 'consciousness', 'principle': 'All is Mind'},
    2: {'name': 'Positive', 'type': 'polarity positive', 'quality': 'attraction', 'principle': 'Polarity+'},
    3: {'name': 'Negative', 'type': 'polarity negative', 'quality': 'repulsion', 'principle': 'Polarity-'},
    4: {'name': 'Vibration', 'type': 'vibrational frequency', 'quality': 'resonance', 'principle': 'Nothing rests'},
    5: {'name': 'Female', 'type': 'feminine principle', 'quality': 'receptivity', 'principle': 'Yin'},
    6: {'name': 'Male', 'type': 'masculine principle', 'quality': 'projection', 'principle': 'Yang'},
    7: {'name': 'Rhythm', 'type': 'rhythmic flow', 'quality': 'cyclical motion', 'principle': 'Pendulum swing'},
    8: {'name': 'Cause', 'type': 'causation', 'quality': 'the above', 'principle': 'As above'},
    9: {'name': 'Effect', 'type': 'correspondence', 'quality': 'the below', 'principle': 'So below'},
    10: {'name': 'Idea', 'type': 'divine idea', 'quality': 'the All', 'principle': 'First Principle'}
}

OPERATOR_DESCRIPTIONS = {
    '+': {'name': 'addition', 'verbs': ['combines with', 'joins', 'merges with', 'unites with']},
    '-': {'name': 'subtraction', 'verbs': ['separates from', 'removes', 'subtracts from', 'diminishes']},
    '+T': {'name': 'temporal addition', 'verbs': ['temporally combines with', 'accumulates with', 'adds over time to']},
    '-T': {'name': 'temporal subtraction', 'verbs': ['temporally separates from', 'removes over time from', 'diminishes from']},
    '->': {'name': 'flow', 'verbs': ['flows to', 'transforms into', 'leads to', 'projects onto']},
    '<-': {'name': 'reception', 'verbs': ['receives from', 'draws from', 'is sourced by', 'absorbs from']},
    '*T': {'name': 'temporal multiplication', 'verbs': ['amplifies with', 'multiplies temporally with', 'resonates with']},
    '/T': {'name': 'temporal division', 'verbs': ['attenuates through', 'divides temporally by', 'dampens with']},
    'o': {'name': 'composition', 'verbs': ['composes with', 'circulates through', 'cycles with', 'integrates with']}
}

FOUNDATION_FULL_NAMES = {
    '_d1': {'name': 'Unity', 'principle': 'coherence and linking', 'planet': 'Sun'},
    '_d2': {'name': 'Wisdom', 'principle': 'discernment and truth', 'planet': 'Moon'},
    '_d3': {'name': 'Life', 'principle': 'vitality and persistence', 'planet': 'Mars'},
    '_d4': {'name': 'Companionship', 'principle': 'relational bonding', 'planet': 'Venus'},
    '_d5': {'name': 'Power', 'principle': 'capacity to influence', 'planet': 'Jupiter'},
    '_d6': {'name': 'Material', 'principle': 'resources and structure', 'planet': 'Saturn'},
    '_d7': {'name': 'Lust', 'principle': 'drive to extend', 'planet': 'Saturn'}
}


# ==============================================================================
# ELEMENT GENERATION
# ==============================================================================

@dataclass
class Element:
    """A TKS element with optional sense and foundation."""
    world: str  # A, B, C, D
    noetic: int  # 1-10
    sense: Optional[int] = None  # 1-9
    foundation: Optional[str] = None  # _d1 through _d7

    def __str__(self) -> str:
        s = f"{self.world}{self.noetic}"
        if self.sense is not None:
            s += f"^{self.sense}"
        if self.foundation is not None:
            s += self.foundation
        return s

    def is_valid(self) -> bool:
        if self.world not in WORLD_DESCRIPTIONS:
            return False
        if self.noetic < 1 or self.noetic > 10:
            return False
        if self.sense is not None and (self.sense < 1 or self.sense > 9):
            return False
        if self.foundation is not None and self.foundation not in VALID_FOUNDATIONS:
            return False
        return True

    def get_rpm(self) -> str:
        """Get RPM classification for this element."""
        if self.noetic in RPM_DESIRE:
            return 'desire'
        elif self.noetic in RPM_WISDOM:
            return 'wisdom'
        elif self.noetic in RPM_POWER:
            return 'power'
        return 'neutral'


def generate_random_element(
    include_sense: bool = False,
    include_foundation: bool = False,
    world: Optional[str] = None,
    noetic_bias: Optional[str] = None  # 'desire', 'wisdom', 'power'
) -> Element:
    """Generate a random canonical element."""
    if world is None:
        world = random.choice(list(WORLD_DESCRIPTIONS.keys()))

    # Apply noetic bias if specified
    if noetic_bias == 'desire':
        noetic = random.choice([1, 4, 7])
    elif noetic_bias == 'wisdom':
        noetic = random.choice([5, 6])
    elif noetic_bias == 'power':
        noetic = random.choice([8, 9])
    else:
        noetic = random.randint(1, 10)

    sense = random.choice(VALID_SENSES) if include_sense and random.random() < 0.3 else None
    foundation = random.choice(VALID_FOUNDATIONS) if include_foundation and random.random() < 0.25 else None

    return Element(world=world, noetic=noetic, sense=sense, foundation=foundation)


def generate_element_sequence(length: int, pattern: Optional[str] = None) -> List[Element]:
    """Generate a sequence of elements following optional pattern."""
    elements = []

    patterns_map = {
        'single_world': lambda: generate_single_world_sequence(length),
        'ascending': lambda: generate_ascending_sequence(length),
        'descending': lambda: generate_descending_sequence(length),
        'rpm_desire': lambda: generate_rpm_biased_sequence(length, 'desire'),
        'rpm_wisdom': lambda: generate_rpm_biased_sequence(length, 'wisdom'),
        'rpm_power': lambda: generate_rpm_biased_sequence(length, 'power'),
        'involution': lambda: generate_involution_sequence(length),
        'foundation_heavy': lambda: generate_foundation_heavy_sequence(length),
    }

    if pattern in patterns_map:
        return patterns_map[pattern]()

    # Default random
    for _ in range(length):
        elements.append(generate_random_element(include_sense=True, include_foundation=True))
    return elements


def generate_single_world_sequence(length: int) -> List[Element]:
    """All elements in same world."""
    world = random.choice(list(WORLD_DESCRIPTIONS.keys()))
    return [generate_random_element(include_sense=True, include_foundation=True, world=world)
            for _ in range(length)]


def generate_ascending_sequence(length: int) -> List[Element]:
    """Ascending through worlds A -> B -> C -> D."""
    world_order = ['A', 'B', 'C', 'D']
    return [generate_random_element(include_sense=True, include_foundation=True,
            world=world_order[i % 4]) for i in range(length)]


def generate_descending_sequence(length: int) -> List[Element]:
    """Descending through worlds D -> C -> B -> A."""
    world_order = ['D', 'C', 'B', 'A']
    return [generate_random_element(include_sense=True, include_foundation=True,
            world=world_order[i % 4]) for i in range(length)]


def generate_rpm_biased_sequence(length: int, rpm_type: str) -> List[Element]:
    """Generate sequence biased toward specific RPM noetics."""
    return [generate_random_element(include_sense=True, include_foundation=True,
            noetic_bias=rpm_type) for _ in range(length)]


def generate_involution_sequence(length: int) -> List[Element]:
    """Include involution pairs (2-3, 5-6, 8-9)."""
    pairs = [(2, 3), (5, 6), (8, 9)]
    pair = random.choice(pairs)
    world = random.choice(list(WORLD_DESCRIPTIONS.keys()))

    elements = [Element(world=world, noetic=pair[0]), Element(world=world, noetic=pair[1])]
    for _ in range(length - 2):
        elements.append(generate_random_element(include_sense=True, include_foundation=True))
    return elements


def generate_foundation_heavy_sequence(length: int) -> List[Element]:
    """Generate sequence with many foundations."""
    elements = []
    for _ in range(length):
        elem = generate_random_element(include_sense=True, include_foundation=False)
        # Higher probability of foundation
        if random.random() < 0.6:
            elem.foundation = random.choice(VALID_FOUNDATIONS)
        elements.append(elem)
    return elements


# ==============================================================================
# EQUATION BUILDING
# ==============================================================================

def build_equation_string(elements: List[Element], operators: List[str]) -> str:
    """Build equation string from elements and operators."""
    if len(elements) == 0:
        return ""

    parts = [str(elements[0])]
    for i, op in enumerate(operators):
        parts.append(f" {op} ")
        parts.append(str(elements[i + 1]))

    return "".join(parts)


def generate_operator_sequence(length: int) -> List[str]:
    """Generate a sequence of canonical operators."""
    return [random.choice(CANONICAL_OPERATORS) for _ in range(length)]


# ==============================================================================
# STORY/INTERPRETATION GENERATION
# ==============================================================================

def generate_element_phrase(elem: Element, context: str = 'subject') -> str:
    """Generate natural language phrase for an element."""
    world_info = WORLD_DESCRIPTIONS[elem.world]
    noetic_info = NOETIC_DESCRIPTIONS[elem.noetic]

    templates = {
        'subject': [
            f"the {noetic_info['name']} principle in {world_info['realm']}",
            f"{world_info['name']} {noetic_info['name']}",
            f"the {noetic_info['quality']} of {world_info['realm']}",
            f"{world_info['essence']} of {noetic_info['name']}",
            f"noetic {noetic_info['name']} within World {elem.world}",
        ],
        'object': [
            f"the {noetic_info['name']} force of {world_info['realm']}",
            f"{world_info['name']}-{noetic_info['name']} energy",
            f"the {noetic_info['quality']} from World {elem.world}",
            f"{world_info['name']} {noetic_info['type']}",
        ]
    }

    phrase = random.choice(templates.get(context, templates['subject']))

    if elem.sense is not None:
        sense_phrases = [f" with sense {elem.sense}", f" at intensity {elem.sense}",
                        f" (sense level {elem.sense})", f" manifesting through sense {elem.sense}"]
        phrase += random.choice(sense_phrases)

    if elem.foundation is not None:
        foundation_name = FOUNDATION_FULL_NAMES[elem.foundation]['name']
        foundation_phrases = [f" grounded in {foundation_name}", f" with foundation of {foundation_name}",
                             f" anchored in {foundation_name}", f" through the {foundation_name} principle"]
        phrase += random.choice(foundation_phrases)

    return phrase


def generate_operator_phrase(op: str) -> str:
    """Generate natural language phrase for an operator."""
    return random.choice(OPERATOR_DESCRIPTIONS[op]['verbs'])


def generate_story(elements: List[Element], operators: List[str]) -> str:
    """Generate a natural language story from elements and operators."""
    if len(elements) == 0:
        return "An empty TKS expression."

    if len(elements) == 1:
        elem_phrase = generate_element_phrase(elements[0], 'subject')
        templates = [
            f"In this working, {elem_phrase} stands alone.",
            f"The expression contains only {elem_phrase}.",
            f"We observe {elem_phrase} in isolation.",
        ]
        return random.choice(templates)

    # Build story through the chain
    story_parts = []

    openings = [
        "In this TKS working,",
        "This equation describes how",
        "The following transformation occurs:",
        "In the noetic sequence,",
        "This working demonstrates that",
        "The TKS expression shows",
    ]
    story_parts.append(random.choice(openings))
    story_parts.append(generate_element_phrase(elements[0], 'subject'))

    for i, op in enumerate(operators):
        story_parts.append(generate_operator_phrase(op))
        story_parts.append(generate_element_phrase(elements[i + 1], 'object'))

    closings = [".", ", completing the noetic circuit.", ", manifesting the intended result.",
                " in the TKS framework.", ", fulfilling the working."]

    return " ".join(story_parts) + random.choice(closings)


def generate_detailed_story(elements: List[Element], operators: List[str]) -> str:
    """Generate a more detailed story with RPM and foundation analysis."""
    base_story = generate_story(elements, operators)

    # Add RPM analysis
    noetics = [e.noetic for e in elements]
    desire_count = sum(1 for n in noetics if n in RPM_DESIRE)
    wisdom_count = sum(1 for n in noetics if n in RPM_WISDOM)
    power_count = sum(1 for n in noetics if n in RPM_POWER)
    total = desire_count + wisdom_count + power_count

    if total > 0:
        dominant = max([('desire', desire_count), ('wisdom', wisdom_count), ('power', power_count)],
                      key=lambda x: x[1])[0]
        rpm_phrases = {
            'desire': " This working emphasizes Desire (MVR: Mind, Vibration, Rhythm) forces.",
            'wisdom': " This working emphasizes Wisdom (Female/Male polarity) forces.",
            'power': " This working emphasizes Power (Cause/Effect) forces."
        }
        base_story += rpm_phrases[dominant]

    # Add foundation mention
    foundations = [e.foundation for e in elements if e.foundation]
    if foundations:
        foundation_names = [FOUNDATION_FULL_NAMES[f]['name'] for f in set(foundations)]
        base_story += f" The foundations of {', '.join(foundation_names)} ground this working."

    return base_story


# ==============================================================================
# RPM ANALYSIS GENERATION
# ==============================================================================

def compute_rpm_distribution(elements: List[Element]) -> Dict[str, float]:
    """Compute D/W/P distribution from elements."""
    counts = {'desire': 0, 'wisdom': 0, 'power': 0}

    for elem in elements:
        if elem.noetic in RPM_DESIRE:
            counts['desire'] += 1
        elif elem.noetic in RPM_WISDOM:
            counts['wisdom'] += 1
        elif elem.noetic in RPM_POWER:
            counts['power'] += 1

    total = sum(counts.values())
    if total == 0:
        return {'desire': 0.33, 'wisdom': 0.33, 'power': 0.34}

    return {k: v / total for k, v in counts.items()}


def generate_rpm_analysis(elements: List[Element], equation: str) -> str:
    """Generate RPM analysis for an equation."""
    rpm_dist = compute_rpm_distribution(elements)

    # Get dominant component
    dominant = max(rpm_dist.items(), key=lambda x: x[1])
    dominant_name = dominant[0].upper()[0]  # D, W, or P

    # Build analysis
    analysis_parts = []

    analysis_parts.append(f"D={rpm_dist['desire']:.2f}")
    analysis_parts.append(f"W={rpm_dist['wisdom']:.2f}")
    analysis_parts.append(f"P={rpm_dist['power']:.2f}")

    analysis = f"RPM Distribution: {', '.join(analysis_parts)}. "
    analysis += f"Dominant: {dominant[0].capitalize()} ({dominant[1]*100:.0f}%). "

    # Add interpretation
    interpretations = {
        'desire': "This equation is driven by desire/intention forces (Mind, Vibration, Rhythm).",
        'wisdom': "This equation emphasizes wisdom/understanding forces (Female, Male polarity).",
        'power': "This equation focuses on power/causation forces (Cause, Effect)."
    }
    analysis += interpretations[dominant[0]]

    return analysis


# ==============================================================================
# FOUNDATION ANALYSIS GENERATION
# ==============================================================================

def generate_foundation_analysis(elements: List[Element], equation: str) -> str:
    """Generate foundation analysis for an equation."""
    foundations = [e.foundation for e in elements if e.foundation]
    worlds = [e.world for e in elements]

    analysis_parts = []

    # Count worlds
    world_counts = Counter(worlds)
    analysis_parts.append("World distribution: " +
                         ", ".join([f"{w}={c}" for w, c in sorted(world_counts.items())]))

    # Foundation analysis
    if foundations:
        foundation_counts = Counter(foundations)
        foundation_names = []
        for f, count in foundation_counts.items():
            fname = FOUNDATION_FULL_NAMES[f]['name']
            foundation_names.append(f"{fname} (F{f[2]}, count={count})")
        analysis_parts.append("Foundations present: " + "; ".join(foundation_names))
    else:
        analysis_parts.append("No explicit foundations in this equation.")

    # World analysis
    if len(set(worlds)) == 1:
        world_name = WORLD_DESCRIPTIONS[worlds[0]]['name']
        analysis_parts.append(f"This equation operates entirely within the {world_name} World ({worlds[0]}).")
    elif len(set(worlds)) == 4:
        analysis_parts.append("This equation spans all four Worlds (A-D), indicating a complete manifestation cycle.")
    else:
        world_names = [WORLD_DESCRIPTIONS[w]['name'] for w in set(worlds)]
        analysis_parts.append(f"This equation bridges the {', '.join(world_names)} Worlds.")

    return " ".join(analysis_parts)


# ==============================================================================
# TRAINING PAIR GENERATION
# ==============================================================================

def generate_equation_to_interpretation_pair(pair_id: str, detailed: bool = False) -> Dict:
    """Generate equation -> interpretation training pair."""
    chain_length = random.choice([2, 2, 3, 3, 3, 4, 4, 5])
    pattern = random.choice([None, 'single_world', 'ascending', 'descending',
                            'rpm_desire', 'rpm_wisdom', 'rpm_power', 'foundation_heavy'])

    elements = generate_element_sequence(chain_length, pattern)
    operators = generate_operator_sequence(chain_length - 1)
    equation = build_equation_string(elements, operators)

    if detailed and random.random() < 0.4:
        story = generate_detailed_story(elements, operators)
    else:
        story = generate_story(elements, operators)

    return {
        'task_type': 'equation_to_interpretation',
        'input': f"Given the TKS equation: {equation}\n\nProvide a natural language interpretation of this noetic working.",
        'target': story,
        'metadata': {
            'pair_id': pair_id,
            'equation': equation,
            'elements': [str(e) for e in elements],
            'operators': operators,
            'pattern': pattern,
            'chain_length': chain_length
        },
        'direction': 'eq_to_story',
        'canon_score': 1.0
    }


def generate_interpretation_to_equation_pair(pair_id: str, detailed: bool = False) -> Dict:
    """Generate interpretation -> equation training pair."""
    chain_length = random.choice([2, 2, 3, 3, 3, 4, 4, 5])
    pattern = random.choice([None, 'single_world', 'ascending', 'descending',
                            'rpm_desire', 'rpm_wisdom', 'rpm_power', 'foundation_heavy'])

    elements = generate_element_sequence(chain_length, pattern)
    operators = generate_operator_sequence(chain_length - 1)
    equation = build_equation_string(elements, operators)

    if detailed and random.random() < 0.4:
        story = generate_detailed_story(elements, operators)
    else:
        story = generate_story(elements, operators)

    return {
        'task_type': 'interpretation_to_equation',
        'input': f"Given this TKS narrative:\n\n{story}\n\nTranslate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 1-10).",
        'target': equation,
        'metadata': {
            'pair_id': pair_id,
            'equation': equation,
            'elements': [str(e) for e in elements],
            'operators': operators,
            'pattern': pattern,
            'chain_length': chain_length
        },
        'direction': 'story_to_eq',
        'canon_score': 1.0
    }


def generate_equation_to_rpm_pair(pair_id: str) -> Dict:
    """Generate equation -> RPM classification training pair."""
    chain_length = random.choice([2, 3, 3, 4, 4, 5])
    pattern = random.choice([None, 'rpm_desire', 'rpm_wisdom', 'rpm_power'])

    elements = generate_element_sequence(chain_length, pattern)
    operators = generate_operator_sequence(chain_length - 1)
    equation = build_equation_string(elements, operators)

    rpm_analysis = generate_rpm_analysis(elements, equation)
    rpm_dist = compute_rpm_distribution(elements)

    return {
        'task_type': 'equation_to_rpm',
        'input': f"Given the TKS equation: {equation}\n\nAnalyze the Desire/Wisdom/Power (D/W/P) distribution in this equation.",
        'target': rpm_analysis,
        'metadata': {
            'pair_id': pair_id,
            'equation': equation,
            'elements': [str(e) for e in elements],
            'operators': operators,
            'rpm_distribution': rpm_dist,
            'pattern': pattern
        },
        'direction': 'eq_to_rpm',
        'canon_score': 1.0
    }


def generate_equation_to_foundations_pair(pair_id: str) -> Dict:
    """Generate equation -> foundation analysis training pair."""
    chain_length = random.choice([2, 3, 3, 4, 4, 5])
    pattern = random.choice([None, 'foundation_heavy', 'single_world', 'ascending'])

    elements = generate_element_sequence(chain_length, pattern)
    operators = generate_operator_sequence(chain_length - 1)
    equation = build_equation_string(elements, operators)

    foundation_analysis = generate_foundation_analysis(elements, equation)

    return {
        'task_type': 'equation_to_foundations',
        'input': f"Given the TKS equation: {equation}\n\nAnalyze the World distribution and Foundation structure in this equation.",
        'target': foundation_analysis,
        'metadata': {
            'pair_id': pair_id,
            'equation': equation,
            'elements': [str(e) for e in elements],
            'operators': operators,
            'pattern': pattern
        },
        'direction': 'eq_to_foundations',
        'canon_score': 1.0
    }


# ==============================================================================
# DATASET GENERATION
# ==============================================================================

def generate_training_batch(
    num_examples: int = 5000,
    seed: int = 42,
    eq_to_interp_ratio: float = 0.30,
    interp_to_eq_ratio: float = 0.30,
    eq_to_rpm_ratio: float = 0.20,
    eq_to_foundations_ratio: float = 0.20
) -> List[Dict]:
    """Generate a batch of diverse training examples."""
    random.seed(seed)

    all_pairs = []
    pair_count = 0

    # Calculate counts for each type
    eq_to_interp_count = int(num_examples * eq_to_interp_ratio)
    interp_to_eq_count = int(num_examples * interp_to_eq_ratio)
    eq_to_rpm_count = int(num_examples * eq_to_rpm_ratio)
    eq_to_foundations_count = num_examples - eq_to_interp_count - interp_to_eq_count - eq_to_rpm_count

    print(f"Generating {eq_to_interp_count} equation_to_interpretation pairs...")
    for i in range(eq_to_interp_count):
        pair_id = f"eq2int_{pair_count:05d}"
        pair = generate_equation_to_interpretation_pair(pair_id, detailed=True)
        all_pairs.append(pair)
        pair_count += 1

    print(f"Generating {interp_to_eq_count} interpretation_to_equation pairs...")
    for i in range(interp_to_eq_count):
        pair_id = f"int2eq_{pair_count:05d}"
        pair = generate_interpretation_to_equation_pair(pair_id, detailed=True)
        all_pairs.append(pair)
        pair_count += 1

    print(f"Generating {eq_to_rpm_count} equation_to_rpm pairs...")
    for i in range(eq_to_rpm_count):
        pair_id = f"eq2rpm_{pair_count:05d}"
        pair = generate_equation_to_rpm_pair(pair_id)
        all_pairs.append(pair)
        pair_count += 1

    print(f"Generating {eq_to_foundations_count} equation_to_foundations pairs...")
    for i in range(eq_to_foundations_count):
        pair_id = f"eq2fnd_{pair_count:05d}"
        pair = generate_equation_to_foundations_pair(pair_id)
        all_pairs.append(pair)
        pair_count += 1

    # Shuffle
    random.shuffle(all_pairs)

    return all_pairs


def validate_pair(pair: Dict) -> Tuple[bool, List[str]]:
    """Validate a training pair for canon compliance."""
    issues = []

    # Check equation elements if available
    if 'metadata' in pair and 'elements' in pair['metadata']:
        element_pattern = re.compile(r'^([ABCD])(10|[1-9])(?:\^([1-9]))?(?:(_d[1-7]))?$')
        for elem_str in pair['metadata']['elements']:
            if not element_pattern.match(elem_str):
                issues.append(f"Invalid element: {elem_str}")

    # Check operators if available
    if 'metadata' in pair and 'operators' in pair['metadata']:
        for op in pair['metadata']['operators']:
            if op not in CANONICAL_OPERATORS:
                issues.append(f"Invalid operator: {op}")

    # Check required fields
    required_fields = ['task_type', 'input', 'target']
    for field in required_fields:
        if field not in pair:
            issues.append(f"Missing required field: {field}")

    return len(issues) == 0, issues


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def print_statistics(pairs: List[Dict]):
    """Print dataset statistics."""
    print("\n" + "=" * 70)
    print("TRAINING DATA GENERATION REPORT")
    print("=" * 70)

    # Task type distribution
    task_counts = Counter(p['task_type'] for p in pairs)
    print(f"\nTotal examples generated: {len(pairs)}")
    print("\nTask Type Distribution:")
    for task, count in sorted(task_counts.items()):
        pct = 100 * count / len(pairs)
        print(f"  - {task}: {count} ({pct:.1f}%)")

    # Direction distribution
    direction_counts = Counter(p.get('direction', 'unknown') for p in pairs)
    print("\nDirection Distribution:")
    for direction, count in sorted(direction_counts.items()):
        pct = 100 * count / len(pairs)
        print(f"  - {direction}: {count} ({pct:.1f}%)")

    # Pattern distribution
    patterns = [p['metadata'].get('pattern', 'random') for p in pairs if 'metadata' in p]
    pattern_counts = Counter(patterns)
    print("\nPattern Distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(patterns) if patterns else 0
        print(f"  - {pattern or 'random'}: {count} ({pct:.1f}%)")

    # Chain length distribution
    chain_lengths = [p['metadata'].get('chain_length', 0) for p in pairs if 'metadata' in p]
    length_counts = Counter(chain_lengths)
    print("\nChain Length Distribution:")
    for length, count in sorted(length_counts.items()):
        if length > 0:
            pct = 100 * count / len(chain_lengths) if chain_lengths else 0
            print(f"  - Length {length}: {count} ({pct:.1f}%)")

    # Validation
    valid_count = 0
    invalid_examples = []
    for pair in pairs:
        is_valid, issues = validate_pair(pair)
        if is_valid:
            valid_count += 1
        else:
            invalid_examples.append((pair.get('metadata', {}).get('pair_id', 'unknown'), issues))

    print(f"\nCanon Compliance: {valid_count}/{len(pairs)} ({100*valid_count/len(pairs):.1f}%)")

    if invalid_examples:
        print(f"\nInvalid examples ({len(invalid_examples)}):")
        for pair_id, issues in invalid_examples[:5]:
            print(f"  - {pair_id}: {'; '.join(issues)}")
        if len(invalid_examples) > 5:
            print(f"  ... and {len(invalid_examples) - 5} more")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate TKS training data batch')
    parser.add_argument('--num-examples', type=int, default=5000,
                       help='Number of examples to generate (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='data/generated_training_batch_1.jsonl',
                       help='Output file path')
    parser.add_argument('--eq-to-interp', type=float, default=0.30,
                       help='Ratio of equation_to_interpretation pairs')
    parser.add_argument('--interp-to-eq', type=float, default=0.30,
                       help='Ratio of interpretation_to_equation pairs')
    parser.add_argument('--eq-to-rpm', type=float, default=0.20,
                       help='Ratio of equation_to_rpm pairs')
    parser.add_argument('--eq-to-foundations', type=float, default=0.20,
                       help='Ratio of equation_to_foundations pairs')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_path = base_dir / args.output

    print("=" * 70)
    print("TKS TRAINING DATA BATCH GENERATOR")
    print("Agent C - Training Data Generation")
    print("=" * 70)
    print(f"\nTarget: {args.num_examples} examples")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")

    # Generate dataset
    pairs = generate_training_batch(
        num_examples=args.num_examples,
        seed=args.seed,
        eq_to_interp_ratio=args.eq_to_interp,
        interp_to_eq_ratio=args.interp_to_eq,
        eq_to_rpm_ratio=args.eq_to_rpm,
        eq_to_foundations_ratio=args.eq_to_foundations
    )

    # Print statistics
    print_statistics(pairs)

    # Save
    save_jsonl(pairs, output_path)
    print(f"\nSaved {len(pairs)} examples to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

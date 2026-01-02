#!/usr/bin/env python3
"""
Generate paired story-equation dataset for bidirectional training.

Creates canonical TKS story-equation pairs with:
- story: The narrative form
- equation: The TKS equation string
- expr_elements: List of elements
- expr_ops: List of operators
- direction: "story_to_eq" or "eq_to_story"
- pair_id: Unique identifier linking both directions

Canon Guardrails (STRICT):
- Worlds: A, B, C, D only
- Noetics: 1-10 only
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (exactly 9)
- Senses: 1-9 only
- Foundations: _d1 through _d7 only
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher.validator import (
    WORLDS, NOETICS, FOUNDATIONS,
    RPM_MAPPINGS, NOETIC_TO_FOUNDATIONS,
    CanonicalValidator
)


# ==============================================================================
# CANONICAL CONSTANTS (STRICT)
# ==============================================================================

# Exactly 9 canonical operators
CANONICAL_OPERATORS = ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']

# Valid senses (1-9)
VALID_SENSES = list(range(1, 10))

# Valid foundations (_d1 through _d7)
VALID_FOUNDATIONS = ['_d1', '_d2', '_d3', '_d4', '_d5', '_d6', '_d7']

# Foundation names for story generation
FOUNDATION_NAMES = {
    '_d1': 'Unity',
    '_d2': 'Wisdom',
    '_d3': 'Life',
    '_d4': 'Companionship',
    '_d5': 'Power',
    '_d6': 'Material',
    '_d7': 'Lust'
}

# World descriptions
WORLD_DESCRIPTIONS = {
    'A': ('Spiritual', 'the spiritual realm', 'spiritual essence'),
    'B': ('Mental', 'the mental plane', 'mental energy'),
    'C': ('Emotional', 'the emotional sphere', 'emotional force'),
    'D': ('Physical', 'the physical world', 'physical manifestation')
}

# Noetic descriptions for stories
NOETIC_DESCRIPTIONS = {
    1: ('Mind', 'mentalism', 'consciousness'),
    2: ('Positive', 'polarity positive', 'attraction'),
    3: ('Negative', 'polarity negative', 'repulsion'),
    4: ('Vibration', 'vibrational frequency', 'resonance'),
    5: ('Female', 'feminine principle', 'receptivity'),
    6: ('Male', 'masculine principle', 'projection'),
    7: ('Rhythm', 'rhythmic flow', 'cyclical motion'),
    8: ('Cause', 'causation', 'the above'),
    9: ('Effect', 'correspondence', 'the below'),
    10: ('Idea', 'divine idea', 'the All')
}

# Operator descriptions for stories
OPERATOR_DESCRIPTIONS = {
    '+': ('combines with', 'joins', 'merges with'),
    '-': ('separates from', 'removes', 'subtracts'),
    '+T': ('temporally combines with', 'adds over time to', 'accumulates with'),
    '-T': ('temporally separates from', 'removes over time from', 'diminishes from'),
    '->': ('flows to', 'transforms into', 'leads to'),
    '<-': ('receives from', 'draws from', 'is sourced by'),
    '*T': ('amplifies with', 'multiplies temporally with', 'resonates with'),
    '/T': ('attenuates through', 'divides temporally by', 'dampens with'),
    'o': ('composes with', 'circulates through', 'cycles with')
}


# ==============================================================================
# DATA STRUCTURES
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
        """Check if element follows canon."""
        if self.world not in WORLDS:
            return False
        if self.noetic < 1 or self.noetic > 10:
            return False
        if self.sense is not None and (self.sense < 1 or self.sense > 9):
            return False
        if self.foundation is not None and self.foundation not in VALID_FOUNDATIONS:
            return False
        return True

    def get_description(self) -> str:
        """Get human-readable description."""
        world_name = WORLD_DESCRIPTIONS[self.world][0]
        noetic_name = NOETIC_DESCRIPTIONS[self.noetic][0]
        desc = f"{world_name}-{noetic_name}"
        if self.sense is not None:
            desc += f" (sense {self.sense})"
        if self.foundation is not None:
            foundation_name = FOUNDATION_NAMES.get(self.foundation, self.foundation)
            desc += f" grounded in {foundation_name}"
        return desc


@dataclass
class StoryEquationPair:
    """A paired story and equation."""
    story: str
    equation: str
    expr_elements: List[str]
    expr_ops: List[str]
    direction: str  # "story_to_eq" or "eq_to_story"
    pair_id: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ==============================================================================
# ELEMENT GENERATION
# ==============================================================================

def generate_random_element(
    include_sense: bool = False,
    include_foundation: bool = False,
    world: Optional[str] = None
) -> Element:
    """Generate a random canonical element."""
    if world is None:
        world = random.choice(list(WORLDS.keys()))
    noetic = random.randint(1, 10)
    sense = random.choice(VALID_SENSES) if include_sense and random.random() < 0.3 else None
    foundation = random.choice(VALID_FOUNDATIONS) if include_foundation and random.random() < 0.2 else None
    return Element(world=world, noetic=noetic, sense=sense, foundation=foundation)


def generate_element_sequence(
    length: int,
    pattern: Optional[str] = None
) -> List[Element]:
    """Generate a sequence of elements following optional pattern."""
    elements = []

    if pattern == 'single_world':
        # All in same world
        world = random.choice(list(WORLDS.keys()))
        for _ in range(length):
            elements.append(generate_random_element(
                include_sense=True, include_foundation=True, world=world))

    elif pattern == 'involution':
        # Include involution pairs (2-3, 5-6, 8-9)
        pairs = [(2, 3), (5, 6), (8, 9)]
        pair = random.choice(pairs)
        world = random.choice(list(WORLDS.keys()))
        elements.append(Element(world=world, noetic=pair[0]))
        elements.append(Element(world=world, noetic=pair[1]))
        for _ in range(length - 2):
            elements.append(generate_random_element(include_sense=True))

    elif pattern == 'ascending':
        # Ascending through worlds A -> B -> C -> D
        world_order = ['A', 'B', 'C', 'D']
        for i in range(length):
            world = world_order[i % 4]
            elements.append(generate_random_element(
                include_sense=True, include_foundation=True, world=world))

    elif pattern == 'descending':
        # Descending through worlds D -> C -> B -> A
        world_order = ['D', 'C', 'B', 'A']
        for i in range(length):
            world = world_order[i % 4]
            elements.append(generate_random_element(
                include_sense=True, include_foundation=True, world=world))

    elif pattern == 'rpm_desire':
        # Focus on desire noetics (1, 4, 7 - MVR protocol)
        for _ in range(length):
            noetic = random.choice([1, 4, 7])  # Mind, Vibration, Rhythm
            world = random.choice(list(WORLDS.keys()))
            elements.append(Element(world=world, noetic=noetic))

    elif pattern == 'rpm_wisdom':
        # Focus on wisdom noetics (5, 6 - Female, Male)
        for _ in range(length):
            noetic = random.choice([5, 6])  # Female, Male
            world = random.choice(list(WORLDS.keys()))
            elements.append(Element(world=world, noetic=noetic))

    elif pattern == 'rpm_power':
        # Focus on power noetics (8, 9)
        for _ in range(length):
            noetic = random.choice([8, 9])
            world = random.choice(list(WORLDS.keys()))
            elements.append(Element(world=world, noetic=noetic))

    else:
        # Random pattern
        for _ in range(length):
            elements.append(generate_random_element(
                include_sense=True, include_foundation=True))

    return elements


def generate_operator_sequence(length: int) -> List[str]:
    """Generate a sequence of canonical operators."""
    return [random.choice(CANONICAL_OPERATORS) for _ in range(length)]


# ==============================================================================
# STORY GENERATION
# ==============================================================================

def generate_element_phrase(elem: Element, context: str = 'subject') -> str:
    """Generate natural language phrase for an element."""
    world_name, world_realm, world_essence = WORLD_DESCRIPTIONS[elem.world]
    noetic_name, noetic_type, noetic_quality = NOETIC_DESCRIPTIONS[elem.noetic]

    templates = {
        'subject': [
            f"the {noetic_name} principle in {world_realm}",
            f"{world_name} {noetic_name}",
            f"the {noetic_quality} of {world_realm}",
            f"{world_essence} of {noetic_name}",
            f"noetic {noetic_name} within World {elem.world}",
        ],
        'object': [
            f"the {noetic_name} force of {world_realm}",
            f"{world_name}-{noetic_name} energy",
            f"the {noetic_quality} from World {elem.world}",
            f"{world_name} {noetic_type}",
        ],
        'complement': [
            f"through {world_name} {noetic_name}",
            f"via the {noetic_quality}",
            f"by means of {world_name} {noetic_type}",
        ]
    }

    phrase = random.choice(templates.get(context, templates['subject']))

    # Add sense if present
    if elem.sense is not None:
        sense_phrases = [
            f" with sense {elem.sense}",
            f" at intensity {elem.sense}",
            f" (sense level {elem.sense})",
            f" manifesting through sense {elem.sense}"
        ]
        phrase += random.choice(sense_phrases)

    # Add foundation if present
    if elem.foundation is not None:
        foundation_name = FOUNDATION_NAMES.get(elem.foundation, elem.foundation)
        foundation_phrases = [
            f" grounded in {foundation_name}",
            f" with foundation of {foundation_name}",
            f" anchored in {foundation_name}",
            f" through the {foundation_name} principle"
        ]
        phrase += random.choice(foundation_phrases)

    return phrase


def generate_operator_phrase(op: str) -> str:
    """Generate natural language phrase for an operator."""
    descs = OPERATOR_DESCRIPTIONS[op]
    return random.choice(descs)


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

    # Opening
    openings = [
        "In this TKS working,",
        "This equation describes how",
        "The following transformation occurs:",
        "In the noetic sequence,",
        "This working demonstrates that",
    ]
    story_parts.append(random.choice(openings))

    # First element
    story_parts.append(generate_element_phrase(elements[0], 'subject'))

    # Chain through remaining elements
    for i, op in enumerate(operators):
        op_phrase = generate_operator_phrase(op)
        elem_phrase = generate_element_phrase(elements[i + 1], 'object')
        story_parts.append(op_phrase)
        story_parts.append(elem_phrase)

    # Closing
    closings = [
        ".",
        ", completing the noetic circuit.",
        ", manifesting the intended result.",
        " in the TKS framework.",
        ".",
    ]

    story = " ".join(story_parts) + random.choice(closings)
    return story


def generate_detailed_story(elements: List[Element], operators: List[str]) -> str:
    """Generate a more detailed story with RPM and foundation analysis."""
    # Count noetics for RPM
    noetics = [e.noetic for e in elements]
    desire_count = sum(1 for n in noetics if n in RPM_MAPPINGS['desire'])
    wisdom_count = sum(1 for n in noetics if n in RPM_MAPPINGS['wisdom'])
    power_count = sum(1 for n in noetics if n in RPM_MAPPINGS['power'])
    total = desire_count + wisdom_count + power_count

    # Determine dominant
    if total > 0:
        rpm_dist = {
            'desire': desire_count / total,
            'wisdom': wisdom_count / total,
            'power': power_count / total
        }
        dominant = max(rpm_dist, key=rpm_dist.get)
    else:
        dominant = 'wisdom'

    # Collect foundations
    foundations_present = set()
    for e in elements:
        if e.foundation:
            foundation_name = FOUNDATION_NAMES.get(e.foundation, e.foundation)
            foundations_present.add(foundation_name)

    # Collect worlds
    worlds = set(e.world for e in elements)
    world_names = [WORLD_DESCRIPTIONS[w][0] for w in worlds]

    # Build detailed story
    parts = []

    # Opening with world context
    if len(worlds) == 1:
        world = list(worlds)[0]
        parts.append(f"Within the {WORLD_DESCRIPTIONS[world][0]} World ({world}),")
    else:
        parts.append(f"Spanning the {', '.join(world_names)} Worlds,")

    # Element chain
    parts.append(generate_element_phrase(elements[0], 'subject'))
    for i, op in enumerate(operators):
        parts.append(generate_operator_phrase(op))
        parts.append(generate_element_phrase(elements[i + 1], 'object'))
    parts.append(".")

    # RPM analysis
    rpm_phrases = {
        'desire': "This working emphasizes Desire (polarity forces) as the dominant influence.",
        'wisdom': "This working emphasizes Wisdom (mental/vibrational forces) as the dominant influence.",
        'power': "This working emphasizes Power (causal forces) as the dominant influence."
    }
    parts.append(rpm_phrases[dominant])

    # Foundation mention
    if foundations_present:
        parts.append(f"The foundations of {', '.join(sorted(foundations_present))} ground this working.")

    return " ".join(parts)


# ==============================================================================
# EQUATION STRING GENERATION
# ==============================================================================

def build_equation_string(elements: List[Element], operators: List[str]) -> str:
    """Build the equation string from elements and operators."""
    if len(elements) == 0:
        return ""

    parts = [str(elements[0])]
    for i, op in enumerate(operators):
        parts.append(f" {op} ")
        parts.append(str(elements[i + 1]))

    return "".join(parts)


# ==============================================================================
# PAIR GENERATION
# ==============================================================================

def generate_base_pair(
    pair_id: str,
    chain_length: int = None,
    pattern: str = None,
    detailed: bool = False
) -> Tuple[Dict, Dict]:
    """
    Generate a base pair (story, equation) and return both directions.

    Returns:
        Tuple of (story_to_eq dict, eq_to_story dict)
    """
    if chain_length is None:
        chain_length = random.randint(2, 5)

    # Generate elements and operators
    elements = generate_element_sequence(chain_length, pattern)
    operators = generate_operator_sequence(chain_length - 1)

    # Build equation
    equation = build_equation_string(elements, operators)
    expr_elements = [str(e) for e in elements]
    expr_ops = operators.copy()

    # Generate story
    if detailed and random.random() < 0.5:
        story = generate_detailed_story(elements, operators)
    else:
        story = generate_story(elements, operators)

    # Create both directions
    story_to_eq = StoryEquationPair(
        story=story,
        equation=equation,
        expr_elements=expr_elements,
        expr_ops=expr_ops,
        direction="story_to_eq",
        pair_id=pair_id
    )

    eq_to_story = StoryEquationPair(
        story=story,
        equation=equation,
        expr_elements=expr_elements,
        expr_ops=expr_ops,
        direction="eq_to_story",
        pair_id=pair_id
    )

    return story_to_eq.to_dict(), eq_to_story.to_dict()


def validate_pair(pair: Dict, validator: CanonicalValidator) -> bool:
    """Validate that a pair follows all canon rules."""
    equation = pair['equation']
    elements = pair['expr_elements']
    ops = pair['expr_ops']

    # Check equation via validator
    result = validator.validate(equation)
    if not result.is_valid:
        return False

    # Check operators are canonical
    for op in ops:
        if op not in CANONICAL_OPERATORS:
            return False

    # Parse elements and validate
    element_pattern = re.compile(r'^([ABCD])(10|[1-9])(?:\^([1-9]))?(?:(_d[1-7]))?$')
    for elem_str in elements:
        match = element_pattern.match(elem_str)
        if not match:
            return False
        world = match.group(1)
        noetic = int(match.group(2))
        sense = match.group(3)
        foundation = match.group(4)

        # Validate ranges
        if world not in WORLDS:
            return False
        if noetic < 1 or noetic > 10:
            return False
        if sense is not None and (int(sense) < 1 or int(sense) > 9):
            return False
        if foundation is not None and foundation not in VALID_FOUNDATIONS:
            return False

    return True


def generate_dataset(
    num_pairs: int = 400,
    seed: int = 42
) -> List[Dict]:
    """Generate the full dataset with both directions."""
    random.seed(seed)

    validator = CanonicalValidator(strict_mode=True)

    all_pairs = []
    pair_count = 0
    attempt_count = 0
    max_attempts = num_pairs * 3  # Allow some failed attempts

    # Distribution of patterns
    patterns = [
        None,  # Random
        'single_world',
        'involution',
        'ascending',
        'descending',
        'rpm_desire',
        'rpm_wisdom',
        'rpm_power',
    ]

    # Distribution of chain lengths
    chain_lengths = [2, 2, 3, 3, 3, 4, 4, 5, 5, 6]  # Weighted toward 3-4

    while pair_count < num_pairs and attempt_count < max_attempts:
        attempt_count += 1

        pair_id = f"pair_{pair_count:04d}"
        pattern = random.choice(patterns)
        chain_length = random.choice(chain_lengths)
        detailed = random.random() < 0.4  # 40% detailed stories

        try:
            story_to_eq, eq_to_story = generate_base_pair(
                pair_id=pair_id,
                chain_length=chain_length,
                pattern=pattern,
                detailed=detailed
            )

            # Validate both
            if validate_pair(story_to_eq, validator) and validate_pair(eq_to_story, validator):
                all_pairs.append(story_to_eq)
                all_pairs.append(eq_to_story)
                pair_count += 1
        except Exception as e:
            continue  # Skip failed generations

    return all_pairs


def split_dataset(
    pairs: List[Dict],
    train_ratio: float = 0.85,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset by pair_id (keep both directions in same split).
    """
    random.seed(seed)

    # Group by pair_id
    pair_ids = list(set(p['pair_id'] for p in pairs))
    random.shuffle(pair_ids)

    # Split pair_ids
    split_idx = int(len(pair_ids) * train_ratio)
    train_ids = set(pair_ids[:split_idx])
    holdout_ids = set(pair_ids[split_idx:])

    # Assign pairs to splits
    train_pairs = [p for p in pairs if p['pair_id'] in train_ids]
    holdout_pairs = [p for p in pairs if p['pair_id'] in holdout_ids]

    return train_pairs, holdout_pairs


def convert_to_training_format(pairs: List[Dict]) -> List[Dict]:
    """Convert pairs to the training format used by the teacher pipeline."""
    converted = []

    for pair in pairs:
        if pair['direction'] == 'story_to_eq':
            # Story input, equation output
            training_entry = {
                'task_type': 'story_to_equation',
                'input': f"Given this TKS narrative:\n\n{pair['story']}\n\nTranslate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 1-10).",
                'target': pair['equation'],
                'metadata': {
                    'elements': pair['expr_elements'],
                    'operators': pair['expr_ops'],
                    'pair_id': pair['pair_id']
                },
                'direction': pair['direction'],
                'canon_score': 1.0
            }
        else:
            # Equation input, story output
            training_entry = {
                'task_type': 'equation_to_story',
                'input': f"Given the TKS equation: {pair['equation']}\n\nElements: {', '.join(pair['expr_elements'])}\n\nGenerate a natural language narrative describing this TKS working.",
                'target': pair['story'],
                'metadata': {
                    'elements': pair['expr_elements'],
                    'operators': pair['expr_ops'],
                    'pair_id': pair['pair_id']
                },
                'direction': pair['direction'],
                'canon_score': 1.0
            }

        converted.append(training_entry)

    return converted


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def print_summary(
    all_pairs: List[Dict],
    train_pairs: List[Dict],
    holdout_pairs: List[Dict]
):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("STORY-EQUATION PAIR GENERATION SUMMARY")
    print("=" * 70)

    total = len(all_pairs)
    story_to_eq = sum(1 for p in all_pairs if p['direction'] == 'story_to_eq')
    eq_to_story = sum(1 for p in all_pairs if p['direction'] == 'eq_to_story')
    unique_pairs = len(set(p['pair_id'] for p in all_pairs))

    print(f"\nTotal samples: {total}")
    print(f"Unique pairs: {unique_pairs}")
    print(f"Direction distribution:")
    print(f"  - story_to_eq: {story_to_eq} ({100*story_to_eq/total:.1f}%)")
    print(f"  - eq_to_story: {eq_to_story} ({100*eq_to_story/total:.1f}%)")

    print(f"\nTrain/Holdout split:")
    train_unique = len(set(p['pair_id'] for p in train_pairs))
    holdout_unique = len(set(p['pair_id'] for p in holdout_pairs))
    print(f"  - Train: {len(train_pairs)} samples ({train_unique} unique pairs)")
    print(f"  - Holdout: {len(holdout_pairs)} samples ({holdout_unique} unique pairs)")

    # Chain length distribution
    chain_lengths = []
    for p in all_pairs:
        if p['direction'] == 'story_to_eq':  # Only count once per pair
            chain_lengths.append(len(p['expr_elements']))

    from collections import Counter
    length_dist = Counter(chain_lengths)
    print(f"\nChain length distribution:")
    for length in sorted(length_dist.keys()):
        count = length_dist[length]
        print(f"  - Length {length}: {count} pairs")

    # Sample validation
    print(f"\nCanon compliance: 100% (validated during generation)")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate story-equation pairs')
    parser.add_argument('--num-pairs', type=int, default=400,
                       help='Number of unique pairs to generate (default: 400)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--train-ratio', type=float, default=0.85,
                       help='Train split ratio (default: 0.85)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'output'

    print("Generating story-equation pairs...")
    print(f"  Target: {args.num_pairs} unique pairs")
    print(f"  Seed: {args.seed}")

    # Generate dataset
    all_pairs = generate_dataset(num_pairs=args.num_pairs, seed=args.seed)

    # Split
    train_pairs, holdout_pairs = split_dataset(
        all_pairs, train_ratio=args.train_ratio, seed=args.seed)

    # Save raw pairs
    save_jsonl(train_pairs, data_dir / 'story_equation_pairs_train.jsonl')
    save_jsonl(holdout_pairs, data_dir / 'story_equation_pairs_holdout.jsonl')

    print(f"\nSaved raw pairs:")
    print(f"  - {data_dir / 'story_equation_pairs_train.jsonl'}")
    print(f"  - {data_dir / 'story_equation_pairs_holdout.jsonl'}")

    # Convert to training format
    train_converted = convert_to_training_format(train_pairs)
    holdout_converted = convert_to_training_format(holdout_pairs)

    # Save converted
    save_jsonl(train_converted, output_dir / 'teacher_story_eq_train.jsonl')
    save_jsonl(holdout_converted, output_dir / 'teacher_story_eq_holdout.jsonl')

    print(f"\nSaved training format:")
    print(f"  - {output_dir / 'teacher_story_eq_train.jsonl'}")
    print(f"  - {output_dir / 'teacher_story_eq_holdout.jsonl'}")

    # Print summary
    print_summary(all_pairs, train_pairs, holdout_pairs)

    return 0


if __name__ == '__main__':
    sys.exit(main())

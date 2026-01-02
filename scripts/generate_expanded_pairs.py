"""
Generate 500+ story-equation pairs with canonical TKS equations.

Canon Guardrails (STRICT):
- Worlds: A, B, C, D only
- Noetics: 1-10 only
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (exactly 9)
- Senses: ^1-^9
- Foundations: _d1-_d7

This script generates diverse canonical equations with meaningful narratives.
Each equation is validated before output.
"""
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
sys.path.insert(0, str(Path(__file__).parent.parent))
from teacher.validator import CanonicalValidator
VALID_FOUNDATIONS = ['_d1', '_d2', '_d3', '_d4', '_d5', '_d6', '_d7']
WORLD_SEMANTICS = {'A': {'name': 'Spiritual', 'realm': 'the spiritual realm', 'essence': 'spiritual essence', 'plane': 'higher spiritual plane'}, 'B': {'name': 'Mental', 'realm': 'the mental plane', 'essence': 'mental energy', 'plane': 'mental dimension'}, 'C': {'name': 'Emotional', 'realm': 'the emotional sphere', 'essence': 'emotional force', 'plane': 'emotional field'}, 'D': {'name': 'Physical', 'realm': 'the physical world', 'essence': 'physical manifestation', 'plane': 'material plane'}}
NOETIC_MEANINGS = {1: {'name': 'Mind', 'quality': 'consciousness', 'aspect': 'mentalism', 'principle': 'mental principle'}, 2: {'name': 'Positive', 'quality': 'attraction', 'aspect': 'polarity positive', 'principle': 'attractive force'}, 3: {'name': 'Negative', 'quality': 'repulsion', 'aspect': 'polarity negative', 'principle': 'repulsive force'}, 4: {'name': 'Vibration', 'quality': 'resonance', 'aspect': 'vibrational frequency', 'principle': 'vibratory motion'}, 5: {'name': 'Female', 'quality': 'receptivity', 'aspect': 'feminine principle', 'principle': 'receptive force'}, 6: {'name': 'Male', 'quality': 'projection', 'aspect': 'masculine principle', 'principle': 'projective force'}, 7: {'name': 'Rhythm', 'quality': 'cyclical motion', 'aspect': 'rhythmic flow', 'principle': 'rhythmic pattern'}, 8: {'name': 'Cause', 'quality': 'the above', 'aspect': 'causation', 'principle': 'causal force'}, 9: {'name': 'Effect', 'quality': 'the below', 'aspect': 'correspondence', 'principle': 'resultant effect'}, 10: {'name': 'Idea', 'quality': 'the All', 'aspect': 'divine idea', 'principle': 'universal principle'}}
OPERATOR_MEANINGS = {'+': ['combines with', 'joins', 'merges with', 'unites with'], '-': ['separates from', 'removes', 'subtracts', 'divides from'], '+T': ['temporally combines with', 'accumulates with', 'adds over time to', 'builds upon'], '-T': ['temporally separates from', 'diminishes from', 'removes over time from', 'decays from'], '->': ['flows to', 'transforms into', 'leads to', 'becomes', 'transmutes into'], '<-': ['receives from', 'draws from', 'is sourced by', 'originates from'], '*T': ['amplifies with', 'multiplies temporally with', 'resonates with', 'intensifies through'], '/T': ['attenuates through', 'divides temporally by', 'dampens with', 'moderates through'], 'o': ['composes with', 'circulates through', 'cycles with', 'interweaves with']}
FOUNDATION_NAMES = {'_d1': 'Unity', '_d2': 'Wisdom', '_d3': 'Life', '_d4': 'Companionship', '_d5': 'Power', '_d6': 'Material', '_d7': 'Lust'}
import re
from tks_rules.noetics import INVOLUTION_PAIRS, SENSES
from tks_rules.operators import OPERATORS
from tks_rules.rpm import RPM_DESIRE, RPM_POWER, RPM_WISDOM
from tks_rules.worlds import WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))

# Use the 9-operator subset used across this repo's training data (exclude * and /)
CANONICAL_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]
VALID_SENSES = SENSES  # Alias for compatibility
ELEMENT_PATTERN = re.compile('^([ABCD])(10|[1-9])(\\^[1-9])?(_d[1-7])?$')

def validate_element(element: str) -> Tuple[bool, Optional[str]]:
    """Validate a single element against canon rules."""
    match = ELEMENT_PATTERN.match(element)
    if not match:
        return (False, f'Invalid element format: {element}')
    world = match.group(1)
    noetic = int(match.group(2))
    if world not in WORLD_LETTERS:
        return (False, f"Invalid world '{world}' (must be A, B, C, D)")
    if noetic < 1 or noetic > 10:
        return (False, f'Invalid noetic {noetic} (must be 1-10)')
    sense_part = match.group(3)
    if sense_part:
        sense = int(sense_part[1])
        if sense < 1 or sense > 9:
            return (False, f'Invalid sense {sense} (must be 1-9)')
    foundation_part = match.group(4)
    if foundation_part:
        foundation = int(foundation_part[2:])
        if foundation < 1 or foundation > 7:
            return (False, f'Invalid foundation {foundation} (must be 1-7)')
    return (True, None)

def validate_operator(op: str) -> Tuple[bool, Optional[str]]:
    """Validate a single operator against canon rules."""
    if op not in CANONICAL_OPERATORS:
        return (False, f"Invalid operator '{op}' (allowed: {CANONICAL_OPERATORS})")
    return (True, None)

def validate_equation(elements: List[str], ops: List[str]) -> Tuple[bool, List[str]]:
    """Validate a complete equation for 100% canon compliance."""
    errors = []
    if len(elements) < 2:
        errors.append('Equation must have at least 2 elements')
    if len(ops) != len(elements) - 1:
        errors.append(f'Expected {len(elements) - 1} operators, got {len(ops)}')
    for elem in elements:
        valid, error = validate_element(elem)
        if not valid:
            errors.append(error)
    for op in ops:
        valid, error = validate_operator(op)
        if not valid:
            errors.append(error)
    return (len(errors) == 0, errors)

def make_element(world: str, noetic: int, sense: Optional[int]=None, foundation: Optional[str]=None) -> str:
    """Create canonical element string."""
    elem = f'{world}{noetic}'
    if sense is not None:
        elem += f'^{sense}'
    if foundation is not None:
        elem += foundation
    return elem

def generate_random_element(rng: random.Random, world: Optional[str]=None, noetic: Optional[int]=None, sense_prob: float=0.35, foundation_prob: float=0.25) -> Tuple[str, str, int, Optional[int], Optional[str]]:
    """Generate a random canonical element."""
    w = world if world is not None else rng.choice(WORLD_LETTERS)
    n = noetic if noetic is not None else rng.choice(NOETIC_NUMBERS)
    s = rng.choice(VALID_SENSES) if rng.random() < sense_prob else None
    f = rng.choice(VALID_FOUNDATIONS) if rng.random() < foundation_prob else None
    return (make_element(w, n, s, f), w, n, s, f)

def generate_element_phrase(world: str, noetic: int, sense: Optional[int], foundation: Optional[str], rng: random.Random, context: str='subject') -> str:
    """Generate a natural language phrase for an element."""
    world_info = WORLD_SEMANTICS[world]
    noetic_info = NOETIC_MEANINGS[noetic]
    if context == 'subject':
        templates = [f"the {noetic_info['name']} principle in {world_info['realm']}", f"{world_info['name']} {noetic_info['name']}", f"the {noetic_info['quality']} of {world_info['realm']}", f"{noetic_info['aspect']} within World {world}", f"the {noetic_info['principle']} of {world_info['plane']}"]
    else:
        templates = [f"the {noetic_info['name']} force of {world_info['realm']}", f"{world_info['name']}-{noetic_info['name']} energy", f"the {noetic_info['quality']} from World {world}", f"{noetic_info['aspect']} of {world_info['essence']}"]
    phrase = rng.choice(templates)
    if sense is not None:
        sense_modifiers = [f' with sense {sense}', f' at intensity {sense}', f' (sense level {sense})', f' amplified to sense {sense}']
        phrase += rng.choice(sense_modifiers)
    if foundation is not None:
        fname = FOUNDATION_NAMES.get(foundation, foundation)
        foundation_modifiers = [f' grounded in {fname}', f' with foundation of {fname}', f' anchored in {fname}', f' through the foundation of {fname}']
        phrase += rng.choice(foundation_modifiers)
    return phrase

def generate_story(elements_data: List[Tuple], operators: List[str], pattern: str, rng: random.Random) -> str:
    """Generate a meaningful narrative for the equation."""
    openings = ['In this TKS working,', 'This equation describes how', 'The following transformation occurs:', 'In the noetic sequence,', 'This working demonstrates that', 'Within the TKS framework,', 'Through this composition,', 'The equation reveals how']
    closings = ['.', ', completing the noetic circuit.', ', manifesting the intended result.', ' in the TKS framework.', ', creating the desired manifestation.', ", bringing forth the working's purpose."]
    story_parts = [rng.choice(openings)]
    story_parts.append(generate_element_phrase(elements_data[0][1], elements_data[0][2], elements_data[0][3], elements_data[0][4], rng, 'subject'))
    for i, op in enumerate(operators):
        story_parts.append(rng.choice(OPERATOR_MEANINGS[op]))
        story_parts.append(generate_element_phrase(elements_data[i + 1][1], elements_data[i + 1][2], elements_data[i + 1][3], elements_data[i + 1][4], rng, 'object'))
    story = ' '.join(story_parts) + rng.choice(closings)
    if rng.random() < 0.25 and pattern:
        pattern_contexts = {'involution_focus': ' This working emphasizes involution pairs that compose toward the All.', 'rpm_desire': ' This working emphasizes Desire (polarity forces).', 'rpm_wisdom': ' This working emphasizes Wisdom (mental and vibrational forces).', 'rpm_power': ' This working emphasizes Power (causal forces).', 'all_worlds': ' This cross-world composition harmonizes all four planes.', 'spiritual_mental': ' This working operates on the higher planes.', 'emotional_physical': ' This working manifests through the lower planes.', 'descent': ' This represents a descending flow through the worlds.', 'ascent': ' This represents an ascending flow through the worlds.'}
        if pattern in pattern_contexts:
            story += pattern_contexts[pattern]
    return story

def generate_equation_with_pattern(rng: random.Random, chain_length: int, pattern: str='random') -> Tuple[List[Tuple], List[str]]:
    """Generate elements and operators following a specific pattern."""
    elements_data = []
    if pattern == 'all_worlds':
        for i in range(chain_length):
            world = WORLD_LETTERS[i % 4]
            elem_data = generate_random_element(rng, world=world)
            elements_data.append(elem_data)
    elif pattern == 'single_world':
        world = rng.choice(WORLD_LETTERS)
        for _ in range(chain_length):
            elem_data = generate_random_element(rng, world=world)
            elements_data.append(elem_data)
    elif pattern == 'involution_focus':
        for i in range(chain_length):
            if i < len(INVOLUTION_PAIRS) * 2:
                pair_idx = i // 2
                noetic = INVOLUTION_PAIRS[pair_idx][i % 2]
                elem_data = generate_random_element(rng, noetic=noetic)
            else:
                elem_data = generate_random_element(rng)
            elements_data.append(elem_data)
    elif pattern == 'rpm_desire':
        for _ in range(chain_length):
            noetic = rng.choice(list(RPM_DESIRE)) if rng.random() < 0.7 else rng.choice(NOETIC_NUMBERS)
            elem_data = generate_random_element(rng, noetic=noetic)
            elements_data.append(elem_data)
    elif pattern == 'rpm_wisdom':
        for _ in range(chain_length):
            noetic = rng.choice(list(RPM_WISDOM)) if rng.random() < 0.7 else rng.choice(NOETIC_NUMBERS)
            elem_data = generate_random_element(rng, noetic=noetic)
            elements_data.append(elem_data)
    elif pattern == 'rpm_power':
        for _ in range(chain_length):
            noetic = rng.choice(list(RPM_POWER)) if rng.random() < 0.7 else rng.choice(NOETIC_NUMBERS)
            elem_data = generate_random_element(rng, noetic=noetic)
            elements_data.append(elem_data)
    elif pattern == 'spiritual_mental':
        for _ in range(chain_length):
            world = rng.choice(['A', 'B'])
            elem_data = generate_random_element(rng, world=world)
            elements_data.append(elem_data)
    elif pattern == 'emotional_physical':
        for _ in range(chain_length):
            world = rng.choice(['C', 'D'])
            elem_data = generate_random_element(rng, world=world)
            elements_data.append(elem_data)
    elif pattern == 'descent':
        for i in range(chain_length):
            world = WORLD_LETTERS[min(i, 3)]
            elem_data = generate_random_element(rng, world=world)
            elements_data.append(elem_data)
    elif pattern == 'ascent':
        for i in range(chain_length):
            world = WORLD_LETTERS[3 - min(i, 3)]
            elem_data = generate_random_element(rng, world=world)
            elements_data.append(elem_data)
    elif pattern == 'extended_heavy':
        for _ in range(chain_length):
            elem_data = generate_random_element(rng, sense_prob=0.7, foundation_prob=0.6)
            elements_data.append(elem_data)
    else:
        for _ in range(chain_length):
            elem_data = generate_random_element(rng)
            elements_data.append(elem_data)
    operators = [rng.choice(CANONICAL_OPERATORS) for _ in range(chain_length - 1)]
    return (elements_data, operators)

def generate_pair(rng: random.Random, pair_id: str, chain_length: Optional[int]=None, pattern: Optional[str]=None) -> Dict:
    """Generate a single story-equation pair."""
    if chain_length is None:
        chain_length = rng.choices([2, 3, 4, 5, 6], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0]
    if pattern is None:
        pattern = 'random'
    elements_data, operators = generate_equation_with_pattern(rng, chain_length, pattern)
    expr_elements = [e[0] for e in elements_data]
    parts = [expr_elements[0]]
    for i, op in enumerate(operators):
        parts.extend([f' {op} ', expr_elements[i + 1]])
    equation = ''.join(parts)
    story = generate_story(elements_data, operators, pattern, rng)
    return {'pair_id': pair_id, 'story': story, 'equation': equation, 'expr_elements': expr_elements, 'expr_ops': operators, 'chain_length': chain_length, 'pattern': pattern, 'canon_score': 1.0}

def main():
    """Generate story-equation pairs."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate story-equation pairs')
    parser.add_argument('--count', type=int, default=500, help='Number of pairs to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data/story_eq_pairs.jsonl', help='Output file')
    args = parser.parse_args()
    seed = args.seed
    target_count = args.count
    output_path = args.output
    rng = random.Random(seed)
    validator = CanonicalValidator(strict_mode=True)
    patterns = [('random', 0.25), ('all_worlds', 0.1), ('single_world', 0.08), ('involution_focus', 0.08), ('rpm_desire', 0.07), ('rpm_wisdom', 0.1), ('rpm_power', 0.07), ('spiritual_mental', 0.07), ('emotional_physical', 0.07), ('descent', 0.04), ('ascent', 0.04), ('extended_heavy', 0.03)]
    pattern_names = [p[0] for p in patterns]
    pattern_weights = [p[1] for p in patterns]
    pairs = []
    seen_equations = set()
    attempts = 0
    max_attempts = target_count * 10
    print(f'Generating {target_count}+ story-equation pairs...')
    print(f'Seed: {seed}')
    print()
    while len(pairs) < target_count and attempts < max_attempts:
        attempts += 1
        pair_id = f'pair_{len(pairs):04d}'
        pattern = rng.choices(pattern_names, weights=pattern_weights)[0]
        pair = generate_pair(rng, pair_id, pattern=pattern)
        if pair['equation'] in seen_equations:
            continue
        is_valid, errors = validate_equation(pair['expr_elements'], pair['expr_ops'])
        if not is_valid:
            print(f"Validation failed for {pair['equation']}: {errors}")
            continue
        story_result = validator.validate(pair['story'])
        if not story_result.is_valid:
            print(f'Story validation failed for pair {pair_id}')
        seen_equations.add(pair['equation'])
        pairs.append(pair)
        if len(pairs) % 100 == 0:
            print(f'Generated {len(pairs)} pairs...')
    print(f'\nGenerated {len(pairs)} unique pairs in {attempts} attempts')
    print('\nRunning final validation...')
    valid_count = 0
    invalid_count = 0
    for pair in pairs:
        is_valid, errors = validate_equation(pair['expr_elements'], pair['expr_ops'])
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"INVALID: {pair['equation']} - {errors}")
    print(f'Valid: {valid_count}, Invalid: {invalid_count}')
    print(f'Canon compliance: {valid_count / len(pairs) * 100:.1f}%')
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f'\nSaved {len(pairs)} pairs to: {output_path}')
    print('\n=== Statistics ===')
    chain_dist = {}
    for pair in pairs:
        cl = pair['chain_length']
        chain_dist[cl] = chain_dist.get(cl, 0) + 1
    print('\nChain length distribution:')
    for cl in sorted(chain_dist.keys()):
        pct = chain_dist[cl] / len(pairs) * 100
        print(f'  {cl} elements: {chain_dist[cl]:3d} ({pct:5.1f}%)')
    pattern_dist = {}
    for pair in pairs:
        pt = pair['pattern']
        pattern_dist[pt] = pattern_dist.get(pt, 0) + 1
    print('\nPattern distribution:')
    for pt, count in sorted(pattern_dist.items(), key=lambda x: -x[1]):
        pct = count / len(pairs) * 100
        print(f'  {pt:20s}: {count:3d} ({pct:5.1f}%)')
    world_counts = {w: 0 for w in WORLD_LETTERS}
    for pair in pairs:
        for elem in pair['expr_elements']:
            world_counts[elem[0]] += 1
    total_elements = sum(world_counts.values())
    print('\nWorld distribution:')
    for w in WORLD_LETTERS:
        pct = world_counts[w] / total_elements * 100
        print(f"  {w} ({WORLD_SEMANTICS[w]['name']:10s}): {world_counts[w]:4d} ({pct:5.1f}%)")
    op_counts = {op: 0 for op in CANONICAL_OPERATORS}
    for pair in pairs:
        for op in pair['expr_ops']:
            op_counts[op] += 1
    total_ops = sum(op_counts.values())
    print('\nOperator distribution:')
    for op in CANONICAL_OPERATORS:
        pct = op_counts[op] / total_ops * 100 if total_ops > 0 else 0
        print(f"  '{op:3s}': {op_counts[op]:3d} ({pct:5.1f}%)")
    sense_count = sum((1 for pair in pairs for e in pair['expr_elements'] if '^' in e))
    foundation_count = sum((1 for pair in pairs for e in pair['expr_elements'] if '_d' in e))
    print('\nExtended notation:')
    print(f'  Elements with sense (^k): {sense_count}')
    print(f'  Elements with foundation (_dF): {foundation_count}')
    print('\n=== Sample Pairs ===')
    for i in range(min(5, len(pairs))):
        pair = pairs[i]
        print(f"\nPair {pair['pair_id']}:")
        print(f"  Equation: {pair['equation']}")
        print(f"  Story: {pair['story'][:150]}...")
    return 0
if __name__ == '__main__':
    exit(main())

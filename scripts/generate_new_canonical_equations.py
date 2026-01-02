"""
Generate new canonical TKS equations with strict canon compliance.

Canon Guardrails (STRICT):
- Worlds: A, B, C, D only (4 worlds)
- Noetics: 1-10 only (10 noetics)
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (exactly 9 operators)
- Senses: 1-9 only (^1 through ^9)
- Foundations: _d1 through _d7 only

Extended notation format: [ABCD](10|[1-9])(^[1-9])?(_d[1-7])?
Examples: A1, B10, C5^3, D7_d4, A8^2_d6

This script generates diverse equations with:
- Chain lengths 3-8 elements (per specification)
- All four worlds represented
- Extended notation (senses, foundations)
- Various operator combinations
- 100% canon validation
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass

# Allow running as a script: `python scripts/generate_new_canonical_equations.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_rules.foundations import FOUNDATIONS
from tks_rules.noetics import INVOLUTION_PAIRS, SENSES, SELF_DUAL_MVR
from tks_rules.rpm import RPM_DESIRE, RPM_POWER, RPM_WISDOM
from tks_rules.worlds import WORLD_LETTERS, WORLD_DOMAINS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))
FOUNDATION_NUMBERS = list(FOUNDATIONS.keys())

# Scripts/docs in this repo use a 9-operator subset (exclude * and /)
TRAINING_OPERATORS = ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']
TRAINING_OPERATOR_SET = set(TRAINING_OPERATORS)
# Aliases for compatibility
SELF_DUAL = SELF_DUAL_MVR
WORLD_NAMES = WORLD_DOMAINS
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
    if op not in TRAINING_OPERATOR_SET:
        return (False, f"Invalid operator '{op}' (allowed: {TRAINING_OPERATORS})")
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

def extract_base_element(element: str) -> str:
    """Extract base element (world + noetic) from extended notation."""
    match = ELEMENT_PATTERN.match(element)
    if match:
        return f'{match.group(1)}{match.group(2)}'
    return element

@dataclass
class GenerationConfig:
    """Configuration for equation generation."""
    chain_length_min: int = 3
    chain_length_max: int = 8
    sense_probability: float = 0.35
    foundation_probability: float = 0.3
    target_count: int = 300
    seed: int = 12345

def generate_element(rng: random.Random, world: str=None, noetic: int=None, add_sense: bool=None, add_foundation: bool=None, config: GenerationConfig=None) -> str:
    """Generate a single canonical element with optional extended notation."""
    config = config or GenerationConfig()
    world = world or rng.choice(WORLD_LETTERS)
    noetic = noetic or rng.choice(NOETIC_NUMBERS)
    elem = f'{world}{noetic}'
    if add_sense is None:
        add_sense = rng.random() < config.sense_probability
    if add_sense:
        sense = rng.choice(SENSES)
        elem += f'^{sense}'
    if add_foundation is None:
        add_foundation = rng.random() < config.foundation_probability
    if add_foundation:
        foundation = rng.choice(FOUNDATION_NUMBERS)
        elem += f'_d{foundation}'
    return elem

def generate_equation_chain(rng: random.Random, chain_length: int, pattern: str=None, config: GenerationConfig=None) -> Tuple[List[str], List[str]]:
    """Generate an equation chain with specified pattern."""
    config = config or GenerationConfig()
    elements = []
    ops = []
    for i in range(chain_length):
        if pattern == 'all_worlds':
            world = WORLD_LETTERS[i % 4]
            elem = generate_element(rng, world=world, config=config)
        elif pattern == 'single_world':
            world = rng.choice(WORLD_LETTERS) if i == 0 else elements[0][0]
            elem = generate_element(rng, world=world, config=config)
        elif pattern == 'involution_focus':
            if i < len(INVOLUTION_PAIRS) * 2:
                pair_idx = i // 2
                noetic = INVOLUTION_PAIRS[pair_idx][i % 2]
            else:
                noetic = rng.choice(NOETIC_NUMBERS)
            elem = generate_element(rng, noetic=noetic, config=config)
        elif pattern == 'rpm_desire':
            noetic = rng.choice(list(RPM_DESIRE)) if rng.random() < 0.7 else rng.choice(NOETIC_NUMBERS)
            elem = generate_element(rng, noetic=noetic, config=config)
        elif pattern == 'rpm_wisdom':
            noetic = rng.choice(list(RPM_WISDOM)) if rng.random() < 0.7 else rng.choice(NOETIC_NUMBERS)
            elem = generate_element(rng, noetic=noetic, config=config)
        elif pattern == 'rpm_power':
            noetic = rng.choice(list(RPM_POWER)) if rng.random() < 0.7 else rng.choice(NOETIC_NUMBERS)
            elem = generate_element(rng, noetic=noetic, config=config)
        elif pattern == 'extended_heavy':
            elem = generate_element(rng, add_sense=rng.random() < 0.7, add_foundation=rng.random() < 0.6, config=config)
        elif pattern == 'spiritual_mental':
            world = rng.choice(['A', 'B'])
            elem = generate_element(rng, world=world, config=config)
        elif pattern == 'emotional_physical':
            world = rng.choice(['C', 'D'])
            elem = generate_element(rng, world=world, config=config)
        elif pattern == 'descent':
            world = WORLD_LETTERS[min(i, 3)]
            elem = generate_element(rng, world=world, config=config)
        elif pattern == 'ascent':
            world = WORLD_LETTERS[3 - min(i, 3)]
            elem = generate_element(rng, world=world, config=config)
        elif pattern == 'noetic_10_focus':
            noetic = 10 if rng.random() < 0.4 else rng.choice(NOETIC_NUMBERS)
            elem = generate_element(rng, noetic=noetic, config=config)
        else:
            elem = generate_element(rng, config=config)
        elements.append(elem)
    for i in range(chain_length - 1):
        op = rng.choice(TRAINING_OPERATORS)
        ops.append(op)
    return (elements, ops)

def build_equation_string(elements: List[str], ops: List[str]) -> str:
    """Build equation string from elements and operators."""
    if len(elements) == 1:
        return elements[0]
    result = elements[0]
    for i, op in enumerate(ops):
        result += f' {op} {elements[i + 1]}'
    return result

def generate_story(equation: str, elements: List[str], pattern: str) -> str:
    """Generate a story/description for the equation."""
    worlds_used = sorted(set((e[0] for e in elements)))
    world_names = [WORLD_NAMES[w] for w in worlds_used]
    stories = [f'Working with {equation}', f'Composition: {equation}', f'TKS equation {equation}', f"Expression in {' and '.join(world_names)}: {equation}", f'Analyzing {equation}', f'Noetic chain: {equation}']
    if pattern and pattern != 'random':
        pattern_stories = {'all_worlds': f'Cross-world expression: {equation}', 'single_world': f'Single-world focus: {equation}', 'involution_focus': f'Involution composition: {equation}', 'rpm_desire': f'Desire-focused expression: {equation}', 'rpm_wisdom': f'Wisdom-focused expression: {equation}', 'rpm_power': f'Power-focused expression: {equation}', 'extended_heavy': f'Extended notation equation: {equation}', 'spiritual_mental': f'Higher plane expression: {equation}', 'emotional_physical': f'Lower plane expression: {equation}', 'descent': f'Descending flow: {equation}', 'ascent': f'Ascending flow: {equation}', 'noetic_10_focus': f'Idea-focused expression: {equation}'}
        if pattern in pattern_stories:
            stories.append(pattern_stories[pattern])
    return random.choice(stories)

def generate_dataset(config: GenerationConfig) -> List[Dict]:
    """Generate a diverse dataset of canonical equations."""
    rng = random.Random(config.seed)
    equations = []
    seen_equations = set()
    patterns = [('random', 0.2), ('all_worlds', 0.1), ('single_world', 0.08), ('involution_focus', 0.08), ('rpm_desire', 0.06), ('rpm_wisdom', 0.08), ('rpm_power', 0.06), ('extended_heavy', 0.12), ('spiritual_mental', 0.06), ('emotional_physical', 0.06), ('descent', 0.04), ('ascent', 0.04), ('noetic_10_focus', 0.02)]
    pattern_names = [p[0] for p in patterns]
    pattern_weights = [p[1] for p in patterns]
    base_chain_weights = {3: 0.25, 4: 0.3, 5: 0.2, 6: 0.12, 7: 0.08, 8: 0.05}
    allowed_chain_lengths = list(range(config.chain_length_min, config.chain_length_max + 1))
    chain_weights = [base_chain_weights.get(n, 1.0) for n in allowed_chain_lengths]
    attempts = 0
    max_attempts = config.target_count * 20
    while len(equations) < config.target_count and attempts < max_attempts:
        attempts += 1
        chain_length = rng.choices(allowed_chain_lengths, weights=chain_weights)[0]
        pattern = rng.choices(pattern_names, weights=pattern_weights)[0]
        elements, ops = generate_equation_chain(rng, chain_length, pattern, config)
        is_valid, errors = validate_equation(elements, ops)
        if not is_valid:
            print(f'Validation failed: {errors}')
            continue
        eq_str = build_equation_string(elements, ops)
        if eq_str in seen_equations:
            continue
        seen_equations.add(eq_str)
        base_elements = [extract_base_element(e) for e in elements]
        story = generate_story(eq_str, elements, pattern)
        equations.append({'story': story, 'equation': eq_str, 'expr_elements': elements, 'expr_elements_base': base_elements, 'expr_ops': ops, 'chain_length': chain_length, 'pattern': pattern, 'aug_type': 'original', 'canon_score': 1.0, 'source': 'canonical_generation_v2'})
    return equations

def run_final_validation(equations: List[Dict]) -> Tuple[int, int, List[str]]:
    """Run final validation pass on all equations."""
    valid_count = 0
    invalid_count = 0
    all_errors = []
    for eq in equations:
        is_valid, errors = validate_equation(eq['expr_elements'], eq['expr_ops'])
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            all_errors.extend([f"{eq['equation']}: {e}" for e in errors])
    return (valid_count, invalid_count, all_errors)

def main():
    parser = argparse.ArgumentParser(description='Generate new canonical TKS equations', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', '-o', default='data/equations_new_canonical.jsonl', help='Output JSONL file')
    parser.add_argument('--count', '-n', type=int, default=300, help='Target number of equations (default: 300)')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility')
    parser.add_argument('--chain-min', type=int, default=3, help='Minimum chain length (default: 3)')
    parser.add_argument('--chain-max', type=int, default=8, help='Maximum chain length (default: 8)')
    parser.add_argument('--sense-prob', type=float, default=0.35, help='Probability of adding sense notation (default: 0.35)')
    parser.add_argument('--foundation-prob', type=float, default=0.3, help='Probability of adding foundation notation (default: 0.30)')
    args = parser.parse_args()
    config = GenerationConfig(chain_length_min=args.chain_min, chain_length_max=args.chain_max, sense_probability=args.sense_prob, foundation_probability=args.foundation_prob, target_count=args.count, seed=args.seed)
    print(f'Generating {config.target_count} canonical TKS equations...')
    print(f'Chain lengths: {config.chain_length_min}-{config.chain_length_max}')
    print(f'Sense probability: {config.sense_probability}')
    print(f'Foundation probability: {config.foundation_probability}')
    print(f'Seed: {config.seed}')
    print()
    equations = generate_dataset(config)
    print('Running final canon validation...')
    valid_count, invalid_count, errors = run_final_validation(equations)
    if invalid_count > 0:
        print(f'\nWARNING: {invalid_count} equations failed validation!')
        for err in errors[:10]:
            print(f'  - {err}')
        equations = [eq for eq in equations if validate_equation(eq['expr_elements'], eq['expr_ops'])[0]]
    print(f'\nValidation Results:')
    print(f'  Valid equations: {valid_count}')
    print(f'  Invalid equations: {invalid_count}')
    print(f'  Canon compliance: {valid_count / (valid_count + invalid_count) * 100:.1f}%')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for eq in equations:
            f.write(json.dumps(eq, ensure_ascii=False) + '\n')
    print(f'\nGenerated {len(equations)} equations')
    chain_dist = {}
    for eq in equations:
        cl = eq['chain_length']
        chain_dist[cl] = chain_dist.get(cl, 0) + 1
    print(f'\nChain length distribution:')
    for cl in sorted(chain_dist.keys()):
        pct = chain_dist[cl] / len(equations) * 100
        print(f'  {cl} elements: {chain_dist[cl]} ({pct:.1f}%)')
    pattern_dist = {}
    for eq in equations:
        pt = eq['pattern']
        pattern_dist[pt] = pattern_dist.get(pt, 0) + 1
    print(f'\nPattern distribution:')
    for pt, count in sorted(pattern_dist.items(), key=lambda x: -x[1]):
        pct = count / len(equations) * 100
        print(f'  {pt}: {count} ({pct:.1f}%)')
    world_counts = {w: 0 for w in WORLD_LETTERS}
    for eq in equations:
        for elem in eq['expr_elements']:
            world_counts[elem[0]] += 1
    total_elements = sum(world_counts.values())
    print(f'\nWorld distribution:')
    for w in WORLD_LETTERS:
        pct = world_counts[w] / total_elements * 100
        print(f'  {w} ({WORLD_NAMES[w]}): {world_counts[w]} ({pct:.1f}%)')
    sense_count = sum((1 for eq in equations for e in eq['expr_elements'] if '^' in e))
    foundation_count = sum((1 for eq in equations for e in eq['expr_elements'] if '_d' in e))
    print(f'\nExtended notation:')
    print(f'  Elements with sense (^k): {sense_count}')
    print(f'  Elements with foundation (_dF): {foundation_count}')
    op_counts = {op: 0 for op in TRAINING_OPERATORS}
    for eq in equations:
        for op in eq['expr_ops']:
            op_counts[op] += 1
    print(f'\nOperator distribution:')
    total_ops = sum(op_counts.values())
    for op in TRAINING_OPERATORS:
        pct = op_counts[op] / total_ops * 100 if total_ops > 0 else 0
        print(f"  '{op}': {op_counts[op]} ({pct:.1f}%)")
    print(f'\nSaved to: {output_path}')
    return 0
if __name__ == '__main__':
    exit(main())

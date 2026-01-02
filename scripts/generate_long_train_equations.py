"""
Generate long-chain training equations for Track 1 uplift.

Requirements:
- 400 canonical equations (300-500 range)
- Each with 5-6 operators (chains of 6-7 elements)
- Use ALL 9 operators: +, -, +T, -T, ->, <-, *T, /T, o
- Use ALL 4 worlds: A, B, C, D
- Use ALL noetics: 1-10
- Mix of plain elements, with sense (^1-^9), with subfoundation (_d1-_d7), and full notation
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter

# Allow running as a script: `python scripts/generate_long_train_equations.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_rules.foundations import FOUNDATIONS
from tks_rules.noetics import SENSES
from tks_rules.operators import OPERATORS
from tks_rules.worlds import WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))
FOUNDATION_NUMBERS = list(FOUNDATIONS.keys())
TRAINING_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]

def generate_element(rng: random.Random, world: str=None, noetic: int=None, notation_style: str=None) -> str:
    """Generate a single element with specified notation style."""
    world = world or rng.choice(WORLD_LETTERS)
    noetic = noetic or rng.choice(NOETIC_NUMBERS)
    elem = f'{world}{noetic}'
    if notation_style is None:
        notation_style = rng.choices(['plain', 'sense', 'foundation', 'full'], weights=[0.4, 0.25, 0.2, 0.15])[0]
    if notation_style == 'sense':
        sense = rng.choice(SENSES)
        elem += f'^{sense}'
    elif notation_style == 'foundation':
        foundation = rng.choice(FOUNDATION_NUMBERS)
        elem += f'_d{foundation}'
    elif notation_style == 'full':
        sense = rng.choice(SENSES)
        foundation = rng.choice(FOUNDATION_NUMBERS)
        elem += f'^{sense}_d{foundation}'
    return elem

def extract_base_element(element: str) -> str:
    """Extract base element (world + noetic) from annotated element."""
    import re
    match = re.match('([ABCD])(10|[1-9])', element.upper())
    if match:
        return f'{match.group(1)}{match.group(2)}'
    return element

def generate_long_chain_equation(rng: random.Random, chain_length: int) -> Tuple[List[str], List[str], str]:
    """Generate a long-chain equation with specified length."""
    elements = []
    ops = []
    for i in range(chain_length):
        elem = generate_element(rng)
        elements.append(elem)
    for i in range(chain_length - 1):
        op = rng.choice(TRAINING_OPERATORS)
        ops.append(op)
    eq_str = elements[0]
    for i, op in enumerate(ops):
        eq_str += f' {op} {elements[i + 1]}'
    return (elements, ops, eq_str)

def generate_dataset(target_count: int=400, seed: int=42) -> List[Dict]:
    """Generate long-chain training equations."""
    rng = random.Random(seed)
    equations = []
    seen_base = set()
    chain_lengths = [6, 7]
    world_usage = Counter()
    noetic_usage = Counter()
    operator_usage = Counter()
    notation_dist = Counter()
    attempts = 0
    max_attempts = target_count * 20
    print('Generating long-chain equations...')
    print(f'Target: {target_count} equations')
    print(f'Chain lengths: 6-7 elements (5-6 operators)')
    while len(equations) < target_count and attempts < max_attempts:
        attempts += 1
        chain_length = rng.choice(chain_lengths)
        elements, ops, eq_str = generate_long_chain_equation(rng, chain_length)
        base_elements = [extract_base_element(e) for e in elements]
        base_key = '_'.join(sorted(base_elements)) + '_' + '_'.join(ops)
        if base_key in seen_base:
            continue
        seen_base.add(base_key)
        for elem in elements:
            base = extract_base_element(elem)
            world = base[0]
            noetic = int(base[1:])
            world_usage[world] += 1
            noetic_usage[noetic] += 1
            if '^' in elem and '_d' in elem:
                notation_dist['full'] += 1
            elif '^' in elem:
                notation_dist['sense'] += 1
            elif '_d' in elem:
                notation_dist['foundation'] += 1
            else:
                notation_dist['plain'] += 1
        for op in ops:
            operator_usage[op] += 1
        equations.append({'equation': eq_str, 'elements': elements, 'ops': ops})
        if len(equations) % 50 == 0:
            print(f'  Generated {len(equations)}/{target_count} equations...')
    print(f'\nGeneration complete: {len(equations)} equations in {attempts} attempts')
    print('\n=== Coverage Statistics ===')
    print('\nWorld distribution:')
    for world in sorted(WORLD_LETTERS):
        count = world_usage[world]
        print(f'  {world}: {count} occurrences')
    print('\nNoetic distribution:')
    for noetic in sorted(NOETIC_NUMBERS):
        count = noetic_usage[noetic]
        print(f'  N{noetic}: {count} occurrences')
    print('\nOperator distribution:')
    for op in TRAINING_OPERATORS:
        count = operator_usage[op]
        print(f'  {op}: {count} occurrences')
    print('\nNotation distribution:')
    total_elements = sum(notation_dist.values())
    for style in ['plain', 'sense', 'foundation', 'full']:
        count = notation_dist[style]
        pct = 100.0 * count / total_elements if total_elements > 0 else 0
        print(f'  {style}: {count} ({pct:.1f}%)')
    print('\nChain length distribution:')
    chain_dist = Counter((len(eq['elements']) for eq in equations))
    for length in sorted(chain_dist.keys()):
        count = chain_dist[length]
        ops_count = length - 1
        print(f'  {length} elements ({ops_count} ops): {count}')
    print('\n=== Coverage Validation ===')
    worlds_covered = set(world_usage.keys())
    noetics_covered = set(noetic_usage.keys())
    operators_covered = set(operator_usage.keys())
    print(f'Worlds: {len(worlds_covered)}/4 covered - {sorted(worlds_covered)}')
    print(f'Noetics: {len(noetics_covered)}/10 covered - {sorted(noetics_covered)}')
    print(f'Operators: {len(operators_covered)}/9 covered - {sorted(operators_covered)}')
    missing_worlds = set(WORLD_LETTERS) - worlds_covered
    missing_noetics = set(NOETIC_NUMBERS) - noetics_covered
    missing_ops = set(TRAINING_OPERATORS) - operators_covered
    if missing_worlds:
        print(f'  WARNING: Missing worlds: {missing_worlds}')
    if missing_noetics:
        print(f'  WARNING: Missing noetics: {missing_noetics}')
    if missing_ops:
        print(f'  WARNING: Missing operators: {missing_ops}')
    if not missing_worlds and (not missing_noetics) and (not missing_ops):
        print('  All worlds, noetics, and operators are covered!')
    return equations

def validate_canonical(equations: List[Dict]) -> Tuple[int, int]:
    """Validate that all equations are canonical."""
    valid_count = 0
    invalid_count = 0
    for eq in equations:
        is_valid = True
        for elem in eq['elements']:
            base = extract_base_element(elem)
            if not base:
                is_valid = False
                break
            world = base[0]
            try:
                noetic = int(base[1:])
            except ValueError:
                is_valid = False
                break
            if world not in WORLD_LETTERS or noetic not in NOETIC_NUMBERS:
                is_valid = False
                break
        for op in eq['ops']:
            if op not in TRAINING_OPERATORS:
                is_valid = False
                break
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
    return (valid_count, invalid_count)

def main():
    parser = argparse.ArgumentParser(description='Generate long-chain training equations')
    parser.add_argument('--output', '-o', default='data/equations_long_train.jsonl', help='Output JSONL file')
    parser.add_argument('--count', '-n', type=int, default=400, help='Number of equations (default: 400)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    print('=' * 60)
    print('Long-Chain Training Data Generator')
    print('=' * 60)
    equations = generate_dataset(args.count, args.seed)
    print('\n=== Canonical Validation ===')
    valid_count, invalid_count = validate_canonical(equations)
    total = valid_count + invalid_count
    pass_rate = 100.0 * valid_count / total if total > 0 else 0
    print(f'Valid equations: {valid_count}/{total}')
    print(f'Invalid equations: {invalid_count}/{total}')
    print(f'Canon pass-rate: {pass_rate:.1f}%')
    if pass_rate == 100.0:
        print('SUCCESS: 100% canonical pass-rate achieved!')
    else:
        print('WARNING: Not all equations are canonical!')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for eq in equations:
            f.write(json.dumps(eq) + '\n')
    print(f'\n=== Output ===')
    print(f'Saved {len(equations)} equations to: {output_path.absolute()}')
    print('=' * 60)
if __name__ == '__main__':
    main()

"""
Generate rich seed equations with extended notation.

Extended notation includes:
- Sense: ^k where k in 1-9
- Foundation: _dF where F in 1-7
- Full: A1^3_d5

This generates diverse equations with varied:
- Chain lengths (2-6 elements)
- All 4 worlds (A, B, C, D)
- All 10 noetics (1-10)
- All 9 operators (+, -, +T, -T, ->, <-, *T, /T, o)
- Optional sense and foundation annotations
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Allow running as a script: `python scripts/generate_rich_equations.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_rules.foundations import FOUNDATIONS
from tks_rules.noetics import INVOLUTION_PAIRS, SENSES, SELF_DUAL_MVR
from tks_rules.operators import OPERATORS
from tks_rules.worlds import WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))
FOUNDATION_NUMBERS = list(FOUNDATIONS.keys())
TRAINING_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]
SELF_DUAL = SELF_DUAL_MVR  # Alias for compatibility

def generate_element(rng: random.Random, world: str=None, noetic: int=None, add_sense: bool=None, add_foundation: bool=None) -> str:
    """Generate a single element with optional extended notation."""
    world = world or rng.choice(WORLD_LETTERS)
    noetic = noetic or rng.choice(NOETIC_NUMBERS)
    elem = f'{world}{noetic}'
    if add_sense is None:
        add_sense = rng.random() < 0.3
    if add_sense:
        sense = rng.choice(SENSES)
        elem += f'^{sense}'
    if add_foundation is None:
        add_foundation = rng.random() < 0.25
    if add_foundation:
        foundation = rng.choice(FOUNDATION_NUMBERS)
        elem += f'_d{foundation}'
    return elem

def generate_equation(rng: random.Random, chain_length: int, pattern: str=None) -> Tuple[List[str], List[str]]:
    """Generate an equation with specified chain length."""
    elements = []
    ops = []
    for i in range(chain_length):
        if pattern == 'single_world':
            world = rng.choice(WORLD_LETTERS)
            elem = generate_element(rng, world=world)
        elif pattern == 'involution':
            if i < len(INVOLUTION_PAIRS):
                pair = INVOLUTION_PAIRS[i % len(INVOLUTION_PAIRS)]
                noetic = pair[i % 2]
            else:
                noetic = rng.choice(NOETIC_NUMBERS)
            elem = generate_element(rng, noetic=noetic)
        elif pattern == 'physical_focus':
            world = 'D' if rng.random() < 0.6 else rng.choice(WORLD_LETTERS)
            elem = generate_element(rng, world=world)
        elif pattern == 'mental_emotional':
            world = rng.choice(['B', 'C'])
            elem = generate_element(rng, world=world)
        elif pattern == 'causal':
            noetic = rng.choice([8, 9]) if rng.random() < 0.5 else rng.choice(NOETIC_NUMBERS)
            elem = generate_element(rng, noetic=noetic)
        else:
            elem = generate_element(rng)
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

def generate_dataset(target_count: int=300, seed: int=42) -> List[Dict]:
    """Generate a diverse set of rich equations."""
    rng = random.Random(seed)
    equations = []
    seen_base = set()
    chain_weights = {2: 0.2, 3: 0.35, 4: 0.25, 5: 0.12, 6: 0.08}
    patterns = [None, 'single_world', 'involution', 'physical_focus', 'mental_emotional', 'causal']
    attempts = 0
    max_attempts = target_count * 10
    while len(equations) < target_count and attempts < max_attempts:
        attempts += 1
        chain_length = rng.choices(list(chain_weights.keys()), weights=list(chain_weights.values()))[0]
        pattern = rng.choice(patterns)
        elements, ops = generate_equation(rng, chain_length, pattern)
        import re
        base_elements = []
        for e in elements:
            match = re.match('([ABCD])(10|[1-9])', e.upper())
            if match:
                base_elements.append(f'{match.group(1)}{match.group(2)}')
        base_key = '_'.join(sorted(base_elements))
        if base_key in seen_base:
            continue
        seen_base.add(base_key)
        eq_str = build_equation_string(elements, ops)
        equations.append({'equation': eq_str, 'chain_length': chain_length, 'pattern': pattern or 'random'})
    return equations

def main():
    parser = argparse.ArgumentParser(description='Generate rich seed equations')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file')
    parser.add_argument('--count', '-n', type=int, default=300, help='Number of equations')
    parser.add_argument('--seed', type=int, default=43, help='Random seed')
    parser.add_argument('--append', action='store_true', help='Append to existing file')
    args = parser.parse_args()
    equations = generate_dataset(args.count, args.seed)
    existing = []
    if args.append and Path(args.output).exists():
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))
        print(f'Loaded {len(existing)} existing equations')
    mode = 'a' if args.append else 'w'
    with open(args.output, mode, encoding='utf-8') as f:
        for eq in equations:
            f.write(json.dumps(eq) + '\n')
    print(f'\nGenerated {len(equations)} new equations')
    print(f'Total equations: {len(existing) + len(equations)}')
    chain_dist = {}
    pattern_dist = {}
    for eq in equations:
        cl = eq['chain_length']
        chain_dist[cl] = chain_dist.get(cl, 0) + 1
        pt = eq['pattern']
        pattern_dist[pt] = pattern_dist.get(pt, 0) + 1
    print(f'\nChain length distribution:')
    for cl in sorted(chain_dist.keys()):
        print(f'  {cl} elements: {chain_dist[cl]}')
    print(f'\nPattern distribution:')
    for pt, count in sorted(pattern_dist.items()):
        print(f'  {pt}: {count}')
    print(f'\nSaved to: {args.output}')
if __name__ == '__main__':
    main()

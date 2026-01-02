"""
Generate diverse canonical TKS equations for training data.

Creates equations with:
- Varied chain lengths (1-6 elements)
- All 9 operators (+, -, +T, -T, ->, <-, *T, /T, o)
- All 4 worlds (A, B, C, D)
- All 10 noetics (1-10)
- Noetic pair awareness (2↔3, 5↔6, 8↔9)
- Foundation coverage

Canon Guardrails:
- Worlds: A, B, C, D
- Noetics: 1-10
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (9 total)

Usage:
    python scripts/generate_diverse_equations.py --count 250 --output data/equations_diverse.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple, Set
from itertools import combinations, product

# Allow running as a script: `python scripts/generate_diverse_equations.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_rules.noetics import INVOLUTION_PAIRS, SELF_DUAL_MVR
from tks_rules.operators import OPERATORS
from tks_rules.worlds import WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))

# Most training/data scripts use the 9-operator subset (exclude * and /)
TRAINING_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]
NOETIC_PAIRS = INVOLUTION_PAIRS  # Already a list of tuples [(2,7), (3,6), (4,5)]
SELF_DUAL_NOETICS = SELF_DUAL_MVR
WORLD_PAIRS = [('A', 'B'), ('C', 'D')]
FOUNDATION_MAP = {1: 'Unity', 2: 'Wisdom', 3: 'Wisdom', 4: 'Life', 5: 'Companionship', 6: 'Companionship', 7: 'Power', 8: 'Material', 9: 'Material', 10: 'Lust'}

def make_element(world: str, noetic: int) -> str:
    """Create element code like 'A5' or 'D10'."""
    return f'{world}{noetic}'

def build_equation(elements: List[str], ops: List[str]) -> str:
    """Build equation string from elements and operators."""
    if len(elements) == 1:
        return elements[0]
    result = elements[0]
    for i, op in enumerate(ops):
        result += f' {op} {elements[i + 1]}'
    return result

def generate_single_elements() -> List[dict]:
    """Generate all single-element equations (40 total)."""
    equations = []
    for world in WORLD_LETTERS:
        for noetic in NOETIC_NUMBERS:
            elem = make_element(world, noetic)
            equations.append({'elements': [elem], 'equation': elem, 'chain_length': 1, 'ops_used': [], 'worlds_used': [world], 'noetics_used': [noetic]})
    return equations

def generate_pair_equations(count: int=100) -> List[dict]:
    """Generate 2-element equations with varied operators."""
    equations = []
    seen = set()
    for op in TRAINING_OPERATORS:
        for _ in range(count // len(TRAINING_OPERATORS) + 1):
            w1, w2 = random.sample(WORLD_LETTERS, 2) if random.random() > 0.3 else [random.choice(WORLD_LETTERS)] * 2
            n1, n2 = random.sample(NOETIC_NUMBERS, 2)
            e1 = make_element(w1, n1)
            e2 = make_element(w2, n2)
            eq_str = build_equation([e1, e2], [op])
            if eq_str not in seen:
                seen.add(eq_str)
                equations.append({'elements': [e1, e2], 'equation': eq_str, 'chain_length': 2, 'ops_used': [op], 'worlds_used': list(set([w1, w2])), 'noetics_used': [n1, n2]})
    return equations[:count]

def generate_triple_equations(count: int=100) -> List[dict]:
    """Generate 3-element equations with mixed operators."""
    equations = []
    seen = set()
    while len(equations) < count:
        worlds = [random.choice(WORLD_LETTERS) for _ in range(3)]
        noetics = random.sample(NOETIC_NUMBERS, 3)
        elements = [make_element(w, n) for w, n in zip(worlds, noetics)]
        ops = [random.choice(TRAINING_OPERATORS) for _ in range(2)]
        eq_str = build_equation(elements, ops)
        if eq_str not in seen:
            seen.add(eq_str)
            equations.append({'elements': elements, 'equation': eq_str, 'chain_length': 3, 'ops_used': ops, 'worlds_used': list(set(worlds)), 'noetics_used': noetics})
    return equations

def generate_quad_equations(count: int=80) -> List[dict]:
    """Generate 4-element equations (all worlds coverage)."""
    equations = []
    seen = set()
    while len(equations) < count:
        if random.random() > 0.5:
            worlds = list(WORLD_LETTERS)
            random.shuffle(worlds)
        else:
            worlds = [random.choice(WORLD_LETTERS) for _ in range(4)]
        noetics = random.sample(NOETIC_NUMBERS, 4)
        elements = [make_element(w, n) for w, n in zip(worlds, noetics)]
        ops = [random.choice(TRAINING_OPERATORS) for _ in range(3)]
        eq_str = build_equation(elements, ops)
        if eq_str not in seen:
            seen.add(eq_str)
            equations.append({'elements': elements, 'equation': eq_str, 'chain_length': 4, 'ops_used': ops, 'worlds_used': list(set(worlds)), 'noetics_used': noetics})
    return equations

def generate_long_chains(count: int=50) -> List[dict]:
    """Generate 5-6 element chains."""
    equations = []
    seen = set()
    while len(equations) < count:
        chain_len = random.choice([5, 6])
        worlds = [random.choice(WORLD_LETTERS) for _ in range(chain_len)]
        noetics = random.choices(NOETIC_NUMBERS, k=chain_len)
        elements = [make_element(w, n) for w, n in zip(worlds, noetics)]
        ops = [random.choice(TRAINING_OPERATORS) for _ in range(chain_len - 1)]
        eq_str = build_equation(elements, ops)
        if eq_str not in seen:
            seen.add(eq_str)
            equations.append({'elements': elements, 'equation': eq_str, 'chain_length': chain_len, 'ops_used': ops, 'worlds_used': list(set(worlds)), 'noetics_used': list(set(noetics))})
    return equations

def generate_noetic_pair_equations(count: int=30) -> List[dict]:
    """Generate equations featuring noetic pairs (2↔3, 5↔6, 8↔9)."""
    equations = []
    seen = set()
    for n1, n2 in NOETIC_PAIRS:
        for _ in range(count // len(NOETIC_PAIRS) + 1):
            w1, w2 = random.sample(WORLD_LETTERS, 2)
            e1 = make_element(w1, n1)
            e2 = make_element(w2, n2)
            op = random.choice(TRAINING_OPERATORS)
            eq_str = build_equation([e1, e2], [op])
            if eq_str not in seen:
                seen.add(eq_str)
                equations.append({'elements': [e1, e2], 'equation': eq_str, 'chain_length': 2, 'ops_used': [op], 'worlds_used': [w1, w2], 'noetics_used': [n1, n2], 'noetic_pair': True})
    return equations[:count]

def generate_foundation_balanced(count: int=40) -> List[dict]:
    """Generate equations balanced across foundations."""
    equations = []
    seen = set()
    foundation_noetics = {'Unity': [1], 'Wisdom': [2, 3], 'Life': [4], 'Companionship': [5, 6], 'Power': [7], 'Material': [8, 9], 'Lust': [10]}
    foundations = list(foundation_noetics.keys())
    while len(equations) < count:
        num_foundations = random.choice([2, 3])
        selected = random.sample(foundations, num_foundations)
        elements = []
        noetics_used = []
        worlds_used = []
        for found in selected:
            noetic = random.choice(foundation_noetics[found])
            world = random.choice(WORLD_LETTERS)
            elements.append(make_element(world, noetic))
            noetics_used.append(noetic)
            worlds_used.append(world)
        ops = [random.choice(TRAINING_OPERATORS) for _ in range(len(elements) - 1)]
        eq_str = build_equation(elements, ops)
        if eq_str not in seen:
            seen.add(eq_str)
            equations.append({'elements': elements, 'equation': eq_str, 'chain_length': len(elements), 'ops_used': ops, 'worlds_used': list(set(worlds_used)), 'noetics_used': noetics_used, 'foundations': selected})
    return equations

def generate_operator_focused(count: int=45) -> List[dict]:
    """Generate equations showcasing specific operator patterns."""
    equations = []
    seen = set()
    patterns = [
        ('uniform', lambda: [random.choice(TRAINING_OPERATORS)] * 3),
        ('alternating', lambda: [TRAINING_OPERATORS[i % len(TRAINING_OPERATORS)] for i in range(3)]),
        ('directional', lambda: ['->', '->', '->']),
        ('reverse_dir', lambda: ['<-', '<-', '<-']),
        ('transform', lambda: ['+T', '-T', '+T']),
        ('composition', lambda: ['o', 'o', 'o']),
        ('mixed_arith', lambda: ['+', '-', '+']),
        ('scale', lambda: ['*T', '/T', '*T']),
    ]
    for pattern_name, pattern_fn in patterns:
        for _ in range(count // len(patterns) + 1):
            ops = pattern_fn()
            chain_len = len(ops) + 1
            worlds = [random.choice(WORLD_LETTERS) for _ in range(chain_len)]
            noetics = random.sample(NOETIC_NUMBERS, min(chain_len, 10))
            if chain_len > 10:
                noetics += random.choices(NOETIC_NUMBERS, k=chain_len - 10)
            elements = [make_element(w, n) for w, n in zip(worlds, noetics)]
            eq_str = build_equation(elements, ops)
            if eq_str not in seen:
                seen.add(eq_str)
                equations.append({'elements': elements, 'equation': eq_str, 'chain_length': chain_len, 'ops_used': ops, 'worlds_used': list(set(worlds)), 'noetics_used': list(set(noetics)), 'pattern': pattern_name})
    return equations[:count]

def generate_all_equations(target_count: int=250) -> List[dict]:
    """Generate diverse equation set targeting count."""
    all_eqs = []
    singles = generate_single_elements()
    pairs = generate_pair_equations(count=50)
    triples = generate_triple_equations(count=50)
    quads = generate_quad_equations(count=40)
    long_chains = generate_long_chains(count=30)
    noetic_pairs = generate_noetic_pair_equations(count=20)
    foundation_bal = generate_foundation_balanced(count=20)
    op_focused = generate_operator_focused(count=30)
    all_eqs.extend(singles)
    all_eqs.extend(pairs)
    all_eqs.extend(triples)
    all_eqs.extend(quads)
    all_eqs.extend(long_chains)
    all_eqs.extend(noetic_pairs)
    all_eqs.extend(foundation_bal)
    all_eqs.extend(op_focused)
    random.shuffle(all_eqs)
    if len(all_eqs) < target_count:
        extra_needed = target_count - len(all_eqs)
        extra = generate_triple_equations(count=extra_needed // 2)
        extra += generate_quad_equations(count=extra_needed // 2)
        all_eqs.extend(extra)
    return all_eqs[:target_count]

def compute_stats(equations: List[dict]) -> dict:
    """Compute statistics about generated equations."""
    stats = {'total': len(equations), 'by_chain_length': {}, 'ops_coverage': set(), 'worlds_coverage': set(), 'noetics_coverage': set()}
    for eq in equations:
        cl = eq['chain_length']
        stats['by_chain_length'][cl] = stats['by_chain_length'].get(cl, 0) + 1
        stats['ops_coverage'].update(eq['ops_used'])
        stats['worlds_coverage'].update(eq['worlds_used'])
        stats['noetics_coverage'].update(eq['noetics_used'])
    stats['ops_coverage'] = list(stats['ops_coverage'])
    stats['worlds_coverage'] = list(stats['worlds_coverage'])
    stats['noetics_coverage'] = sorted(stats['noetics_coverage'])
    return stats

def main():
    parser = argparse.ArgumentParser(description='Generate diverse canonical TKS equations', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--count', type=int, default=250, help='Number of equations to generate (default: 250)')
    parser.add_argument('--output', type=str, default='data/equations_diverse.jsonl', help='Output JSONL file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    random.seed(args.seed)
    print(f'Generating {args.count} diverse canonical equations...')
    equations = generate_all_equations(args.count)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for eq in equations:
            simple = {'elements': eq['elements'], 'equation': eq['equation']}
            f.write(json.dumps(simple, ensure_ascii=False) + '\n')
    stats = compute_stats(equations)
    print(f"\nGenerated {stats['total']} equations")
    print(f'\nChain length distribution:')
    for cl in sorted(stats['by_chain_length'].keys()):
        print(f"  {cl} elements: {stats['by_chain_length'][cl]}")
    print(f"\nOperator coverage: {len(stats['ops_coverage'])}/9")
    print(f"  {', '.join(sorted(stats['ops_coverage']))}")
    print(f"\nWorld coverage: {len(stats['worlds_coverage'])}/4")
    print(f"  {', '.join(sorted(stats['worlds_coverage']))}")
    print(f"\nNoetic coverage: {len(stats['noetics_coverage'])}/10")
    print(f"  {stats['noetics_coverage']}")
    print(f'\nSaved to: {args.output}')
    return 0
if __name__ == '__main__':
    sys.exit(main())

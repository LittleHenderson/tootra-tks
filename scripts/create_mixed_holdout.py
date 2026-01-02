"""
Create a mixed-format holdout set for comprehensive model evaluation.

Content mix:
- Standard canonical equations (short/mid chains, 3-4 ops)
- Long chains (5-6 ops)
- CI-style story/equation pairs

Canon Guardrails (STRICT):
- Worlds: A, B, C, D only
- Noetics: 1-10 only
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (exactly 9)
- Senses: ^1-^9
- Foundations: _d1-_d7
"""
import json
import random
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
sys.path.insert(0, str(Path(__file__).parent.parent))
from teacher.validator import CanonicalValidator
from tks_rules.noetics import SENSES
from tks_rules.operators import OPERATORS
from tks_rules.worlds import WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))

# Use the 9-operator subset used across this repo's training data (exclude * and /)
CANONICAL_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]
VALID_SENSES = SENSES  # Alias for compatibility
VALID_FOUNDATIONS = ['_d1', '_d2', '_d3', '_d4', '_d5', '_d6', '_d7']
ELEMENT_PATTERN = re.compile('^([ABCD])(10|[1-9])(\\^[1-9])?(_d[1-7])?$')
CI_TEMPLATES = ['Working with {eq}', 'Interpretation of {eq}', 'Consider {eq}', 'Analyzing {eq}', 'Expression: {eq}', 'TKS form: {eq}', 'Given {eq}', 'The pattern {eq}', '{eq} manifests', 'Through {eq}, energy flows']

def generate_element(rng: random.Random, use_extended: bool=False) -> str:
    """Generate a canonical element."""
    world = rng.choice(WORLD_LETTERS)
    noetic = rng.choice(NOETIC_NUMBERS)
    element = f'{world}{noetic}'
    if use_extended and rng.random() < 0.4:
        sense = rng.choice(VALID_SENSES)
        element += f'^{sense}'
    if use_extended and rng.random() < 0.3:
        foundation = rng.choice(VALID_FOUNDATIONS)
        element += foundation
    return element

def generate_equation(rng: random.Random, length: int) -> Tuple[str, List[str], List[str]]:
    """Generate a canonical equation."""
    use_extended = rng.random() < 0.5
    elements = [generate_element(rng, use_extended) for _ in range(length)]
    operators = [rng.choice(CANONICAL_OPERATORS) for _ in range(length - 1)]
    parts = [elements[0]]
    for i, op in enumerate(operators):
        parts.append(op)
        parts.append(elements[i + 1])
    equation = ' '.join(parts)
    return (equation, elements, operators)

def validate_element(element: str) -> bool:
    """Validate element against canon."""
    return bool(ELEMENT_PATTERN.match(element))

def create_mixed_holdout(count: int, seed: int) -> List[Dict]:
    """Create mixed-format holdout set."""
    rng = random.Random(seed)
    validator = CanonicalValidator(strict_mode=True)
    items = []
    seen_equations = set()
    target_short = int(count * 0.4)
    target_mid = int(count * 0.35)
    target_long = count - target_short - target_mid
    print(f'Generating mixed holdout: {target_short} short, {target_mid} mid, {target_long} long')

    def add_item(length_range: Tuple[int, int], item_type: str, target: int):
        added = 0
        attempts = 0
        max_attempts = target * 20
        while added < target and attempts < max_attempts:
            attempts += 1
            length = rng.randint(length_range[0], length_range[1])
            equation, elements, operators = generate_equation(rng, length)
            if equation in seen_equations:
                continue
            if not all((validate_element(e) for e in elements)):
                continue
            try:
                result = validator.validate(equation)
                if not result.is_valid:
                    continue
            except:
                continue
            seen_equations.add(equation)
            if rng.random() < 0.7:
                item = {'story': f'TKS equation {equation}', 'expr_elements': elements, 'expr_ops': operators, 'aug_type': 'original', 'format': 'standard', 'chain_length': length}
            else:
                template = rng.choice(CI_TEMPLATES)
                story = template.format(eq=equation)
                item = {'story': story, 'equation': equation, 'expr_elements': elements, 'expr_ops': operators, 'aug_type': 'original', 'format': 'ci_style', 'chain_length': length}
            items.append(item)
            added += 1
        return added
    short_added = add_item((2, 3), 'short', target_short)
    mid_added = add_item((4, 4), 'mid', target_mid)
    long_added = add_item((5, 6), 'long', target_long)
    print(f'Generated: {short_added} short, {mid_added} mid, {long_added} long')
    print(f'Total: {len(items)} items')
    rng.shuffle(items)
    return items

def main():
    parser = argparse.ArgumentParser(description='Create mixed-format holdout')
    parser.add_argument('--count', type=int, default=200, help='Number of items')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data/mixed_holdout.jsonl', help='Output file')
    args = parser.parse_args()
    items = create_mixed_holdout(args.count, args.seed)
    print('\nValidating final set...')
    valid_count = 0
    for item in items:
        if all((validate_element(e) for e in item['expr_elements'])):
            valid_count += 1
    print(f'Valid: {valid_count}/{len(items)} ({valid_count / len(items) * 100:.1f}%)')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'\nSaved {len(items)} items to: {output_path}')
    print('\n=== Statistics ===')
    format_dist = {}
    length_dist = {}
    for item in items:
        fmt = item.get('format', 'unknown')
        format_dist[fmt] = format_dist.get(fmt, 0) + 1
        length = item.get('chain_length', 0)
        length_dist[length] = length_dist.get(length, 0) + 1
    print('\nFormat distribution:')
    for fmt, cnt in sorted(format_dist.items()):
        print(f'  {fmt}: {cnt} ({cnt / len(items) * 100:.1f}%)')
    print('\nChain length distribution:')
    for length, cnt in sorted(length_dist.items()):
        print(f'  {length} elements: {cnt} ({cnt / len(items) * 100:.1f}%)')
    print('\n=== Sample Items ===')
    for i, item in enumerate(items[:3]):
        print(f"\nItem {i + 1} ({item.get('format', 'unknown')}, {item.get('chain_length')} elements):")
        print(f"  Story: {item['story'][:80]}...")
        if 'equation' in item:
            print(f"  Equation: {item['equation']}")
if __name__ == '__main__':
    main()

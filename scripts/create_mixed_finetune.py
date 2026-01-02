"""
Create 1500-sample mixed fine-tune dataset for improving mixed-holdout performance.

Distribution:
- 40% standard short equations (2-3 elements) = 600 samples
- 30% medium equations (4 elements) = 450 samples
- 20% long chains (5-6 elements) = 300 samples
- 10% CI-style terse format = 150 samples

Canon Guardrails (STRICT):
- Worlds: A, B, C, D only
- Noetics: 1-10 only
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (exactly 9)
- Senses: ^1-^9
- Foundations: _d1-_d7

Split 85/15 for train/holdout (1275 train, 225 holdout)
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
CI_TEMPLATES = ['Working with {eq}', 'Interpretation of {eq}', 'Consider {eq}', 'Analyzing {eq}', 'Expression: {eq}', 'TKS form: {eq}', 'Given {eq}', 'The pattern {eq}', '{eq} manifests', 'Through {eq}', 'In {eq}', 'From {eq}', 'Via {eq}', 'Using {eq}', 'Apply {eq}']
STANDARD_TEMPLATES = ['TKS equation {eq}', 'The equation {eq} represents', 'Equation: {eq}', 'In this pattern {eq}', 'TKS expression {eq}', 'Consider the equation {eq}', 'The TKS form {eq}', 'Working with equation {eq}', 'Expression {eq}', 'Pattern {eq}']

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

def create_mixed_finetune(total_count: int, seed: int) -> List[Dict]:
    """Create mixed-format fine-tune dataset."""
    rng = random.Random(seed)
    validator = CanonicalValidator(strict_mode=True)
    items = []
    seen_equations = set()
    target_short = int(total_count * 0.4)
    target_medium = int(total_count * 0.3)
    target_long = int(total_count * 0.2)
    target_ci = total_count - target_short - target_medium - target_long
    print(f'Generating {total_count} samples:')
    print(f'  Short (2-3 elem): {target_short}')
    print(f'  Medium (4 elem): {target_medium}')
    print(f'  Long (5-6 elem): {target_long}')
    print(f'  CI-style terse: {target_ci}')

    def add_item(length_range: Tuple[int, int], item_type: str, target: int, is_ci_style: bool=False):
        """Add items of specified type."""
        added = 0
        attempts = 0
        max_attempts = target * 30
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
            if is_ci_style:
                template = rng.choice(CI_TEMPLATES)
                story = template.format(eq=equation)
                item = {'story': story, 'expr_elements': elements, 'expr_ops': operators, 'aug_type': 'original'}
            else:
                template = rng.choice(STANDARD_TEMPLATES)
                story = template.format(eq=equation)
                item = {'story': story, 'expr_elements': elements, 'expr_ops': operators, 'aug_type': 'original'}
            items.append(item)
            added += 1
            if added % 100 == 0:
                print(f'  {item_type}: {added}/{target}')
        return added
    print('\nGenerating samples...')
    short_added = add_item((2, 3), 'Short', target_short, is_ci_style=False)
    medium_added = add_item((4, 4), 'Medium', target_medium, is_ci_style=False)
    long_added = add_item((5, 6), 'Long', target_long, is_ci_style=False)
    ci_added = add_item((2, 4), 'CI-style', target_ci, is_ci_style=True)
    print(f'\nGenerated totals:')
    print(f'  Short: {short_added}')
    print(f'  Medium: {medium_added}')
    print(f'  Long: {long_added}')
    print(f'  CI-style: {ci_added}')
    print(f'  Total: {len(items)} items')
    rng.shuffle(items)
    return items

def split_train_holdout(items: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and holdout sets."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    holdout = shuffled[split_idx:]
    return (train, holdout)

def main():
    parser = argparse.ArgumentParser(description='Create mixed fine-tune dataset')
    parser.add_argument('--count', type=int, default=1500, help='Total number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train-ratio', type=float, default=0.85, help='Train/holdout split ratio')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    print('=' * 60)
    print('MIXED FINE-TUNE DATASET GENERATION')
    print('=' * 60)
    items = create_mixed_finetune(args.count, args.seed)
    print('\nValidating final set...')
    valid_count = 0
    for item in items:
        if all((validate_element(e) for e in item['expr_elements'])):
            valid_count += 1
    print(f'Valid: {valid_count}/{len(items)} ({valid_count / len(items) * 100:.1f}%)')
    print(f'\nSplitting {args.train_ratio * 100:.0f}/{(1 - args.train_ratio) * 100:.0f} train/holdout...')
    train, holdout = split_train_holdout(items, args.train_ratio, args.seed)
    print(f'  Train: {len(train)} samples')
    print(f'  Holdout: {len(holdout)} samples')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / 'mixed_finetune_train.jsonl'
    holdout_path = output_dir / 'mixed_finetune_holdout.jsonl'
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(holdout_path, 'w', encoding='utf-8') as f:
        for item in holdout:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'\nSaved:')
    print(f'  Train: {train_path} ({len(train)} samples)')
    print(f'  Holdout: {holdout_path} ({len(holdout)} samples)')
    print('\n' + '=' * 60)
    print('STATISTICS')
    print('=' * 60)

    def analyze_set(name: str, dataset: List[Dict]):
        print(f'\n{name} Set ({len(dataset)} samples):')
        length_dist = {}
        for item in dataset:
            length = len(item['expr_elements'])
            length_dist[length] = length_dist.get(length, 0) + 1
        print('  Chain length distribution:')
        for length in sorted(length_dist.keys()):
            count = length_dist[length]
            print(f'    {length} elements: {count:4d} ({count / len(dataset) * 100:.1f}%)')
    analyze_set('Train', train)
    analyze_set('Holdout', holdout)
    print('\n' + '=' * 60)
    print('SAMPLE ITEMS')
    print('=' * 60)
    for i, item in enumerate(train[:5]):
        print(f"\nSample {i + 1} ({len(item['expr_elements'])} elements):")
        print(f"  Story: {item['story']}")
        eq = ' '.join([item['expr_elements'][0]] + [f'{op} {elem}' for op, elem in zip(item['expr_ops'], item['expr_elements'][1:])])
        print(f'  Equation: {eq}')
    print('\n' + '=' * 60)
    print('GENERATION COMPLETE')
    print('=' * 60)
if __name__ == '__main__':
    main()

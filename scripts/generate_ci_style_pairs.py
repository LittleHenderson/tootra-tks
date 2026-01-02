"""
Generate CI-style story-equation base pairs with terse, varied phrasing.

Output format (JSONL; one base pair per line):
  {
    "pair_id": "...",
    "story": "...",
    "equation": "...",
    "expr_elements": [...],
    "expr_ops": [...],
    "canon_score": 1.0
  }

Canon Guardrails (STRICT):
- Worlds: A, B, C, D only
- Noetics: 1-10 only
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (exactly 9)
- Senses: ^1-^9
- Foundations: _d1-_d7
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher.validator import CanonicalValidator
from tks_rules.noetics import SENSES
from tks_rules.operators import OPERATORS
from tks_rules.worlds import WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))

# Use the 9-operator subset used across this repo's training data (exclude * and /)
CANONICAL_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]
VALID_SENSES = SENSES
VALID_FOUNDATIONS = ['_d1', '_d2', '_d3', '_d4', '_d5', '_d6', '_d7']

ELEMENT_PATTERN = re.compile(r'^([ABCD])(10|[1-9])(\^[1-9])?(_d[1-7])?$')

SIMPLE_TEMPLATES = [
    'Working with {eq}',
    'Interpretation of {eq}',
    'Consider {eq}',
    'Analyzing {eq}',
    'Expression: {eq}',
    'TKS form: {eq}',
    'Equation {eq}',
    'The pattern {eq}',
    'Observe {eq}',
    'Given {eq}',
]

CAUSAL_TEMPLATES = [
    'Given {eq}, the effect emerges',
    'Because of {eq}, transformation occurs',
    '{eq} leads to manifestation',
    'Through {eq}, energy flows',
    'When {eq}, balance shifts',
    'As {eq}, polarity changes',
    '{eq} because forces combine',
    '{eq} since worlds interact',
]

RESULT_TEMPLATES = [
    'The result {eq} follows from synthesis',
    'This yields {eq}',
    'We obtain {eq}',
    'The outcome is {eq}',
    '{eq} emerges from the working',
    '{eq} manifests',
    'Final form: {eq}',
]

SHORT_TEMPLATES = [
    '{first} interacts with {rest}',
    '{first} combines with {rest}',
    '{first} transforms via {rest}',
    'From {first} through {rest}',
    '{first} flows to {rest}',
    'Chain: {first} then {rest}',
]

WORLD_TEMPLATES = {
    'A': ['Spiritual {eq}', 'Higher plane {eq}', 'Divine pattern {eq}'],
    'B': ['Mental {eq}', 'Thought form {eq}', 'Mind pattern {eq}'],
    'C': ['Emotional {eq}', 'Feeling state {eq}', 'Astral form {eq}'],
    'D': ['Physical {eq}', 'Material form {eq}', 'Manifest pattern {eq}'],
}


def generate_element(rng: random.Random, use_extended: bool = False) -> str:
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


def generate_equation(rng: random.Random, length: int = None) -> Tuple[str, List[str], List[str]]:
    """Generate a canonical equation with 2-5 elements."""
    if length is None:
        length = rng.randint(2, 5)
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


def generate_ci_story(rng: random.Random, equation: str, elements: List[str], operators: List[str]) -> str:
    """Generate a CI-style terse story for an equation."""
    template_type = rng.choices(
        ['simple', 'causal', 'result', 'short', 'world'],
        weights=[0.35, 0.2, 0.15, 0.2, 0.1],
    )[0]
    if template_type == 'simple':
        return rng.choice(SIMPLE_TEMPLATES).format(eq=equation)
    if template_type == 'causal':
        return rng.choice(CAUSAL_TEMPLATES).format(eq=equation)
    if template_type == 'result':
        return rng.choice(RESULT_TEMPLATES).format(eq=equation)
    if template_type == 'short':
        first = elements[0]
        rest = ' '.join((f'{operators[i]} {elements[i + 1]}' for i in range(len(operators))))
        return rng.choice(SHORT_TEMPLATES).format(first=first, rest=rest)
    if template_type == 'world':
        first_world = elements[0][0]
        templates = WORLD_TEMPLATES.get(first_world, SIMPLE_TEMPLATES)
        return rng.choice(templates).format(eq=equation)
    return f'Working with {equation}'


def _eq_digest64(equation: str) -> int:
    return int.from_bytes(hashlib.blake2b(equation.encode('utf-8'), digest_size=8).digest(), 'little')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate CI-style story-equation base pairs (JSONL)')
    parser.add_argument('--count', type=int, default=2000, help='Number of base pairs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data/story_eq_pairs_ci.jsonl', help='Output JSONL file')
    parser.add_argument('--progress-every', type=int, default=1000, help='Progress print interval (0=off)')
    parser.add_argument('--max-attempts-mult', type=int, default=12, help='Stop after count*mult attempts')
    parser.add_argument(
        '--no-validator',
        action='store_true',
        help='Skip CanonicalValidator (faster; generation is canonical by construction)',
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)
    validator = None if args.no_validator else CanonicalValidator(strict_mode=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    attempts = 0
    max_attempts = args.count * max(args.max_attempts_mult, 1)

    story_len_min: Optional[int] = None
    story_len_max: Optional[int] = None
    story_len_sum = 0
    prefix_counts: Dict[str, int] = {}

    created = 0
    print(f'Generating {args.count:,} CI-style story-equation base pairs...')
    print(f'Seed: {args.seed}')
    print(f'Output: {output_path}')
    print(f'Validator: {"OFF" if validator is None else "ON"}\n')

    with open(output_path, 'w', encoding='utf-8') as f:
        while created < args.count and attempts < max_attempts:
            attempts += 1
            equation, elements, operators = generate_equation(rng)

            eq_key = _eq_digest64(equation)
            if eq_key in seen:
                continue
            if not all((validate_element(e) for e in elements)):
                continue
            if validator is not None:
                try:
                    if not validator.validate(equation).is_valid:
                        continue
                except Exception:
                    continue

            seen.add(eq_key)
            story = generate_ci_story(rng, equation, elements, operators)

            pair = {
                'pair_id': f'pair_{created:07d}',
                'story': story,
                'equation': equation,
                'expr_elements': elements,
                'expr_ops': operators,
                'canon_score': 1.0,
            }
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            created += 1

            slen = len(story)
            story_len_sum += slen
            story_len_min = slen if story_len_min is None else min(story_len_min, slen)
            story_len_max = slen if story_len_max is None else max(story_len_max, slen)

            prefix = story.split()[0] if story else 'unknown'
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

            if args.progress_every and created % args.progress_every == 0:
                print(f'Generated {created:,} pairs... (attempts={attempts:,})')

    print(f'\nGenerated {created:,} unique pairs in {attempts:,} attempts')
    if created < args.count:
        print(f'[warn] Stopped early; requested {args.count:,} pairs')

    print('\n=== Statistics ===')
    if created:
        avg_len = story_len_sum / created
        print(f'Story length: {story_len_min}-{story_len_max} chars (avg {avg_len:.1f})')
        print('Story prefix distribution:')
        for prefix, cnt in sorted(prefix_counts.items(), key=lambda x: -x[1])[:10]:
            print(f'  {prefix:15s}: {cnt:6d} ({cnt / created * 100:.1f}%)')

    print(f'\nSaved {created:,} pairs to: {output_path}')


if __name__ == '__main__':
    main()


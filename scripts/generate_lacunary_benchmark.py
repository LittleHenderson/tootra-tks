"""
TKS Lacunary/Sparse Benchmark Generator

Generates benchmark equations with sparse operator patterns and long-distance
dependencies to test model handling of discontinuities.

Pattern types:
- alternating_operators: Regular +/- alternation
- long_distance_sparse: Boundary ops differ from middle
- banded_operators: First half vs second half ops differ
- self_dual_skip: Uses only self-dual noetics (1, 4, 7, 10)
- involution_sparse: Involution pairs with gaps
- world_skip: Alternating worlds only (A,C or B,D)

Usage:
    python scripts/generate_lacunary_benchmark.py --count 75
"""
import argparse
import json
import random
import sys
from pathlib import Path

# Allow running as a script: `python scripts/generate_lacunary_benchmark.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_rules.noetics import INVOLUTION_PAIRS, SELF_DUAL_MVR
from tks_rules.worlds import WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))
SELF_DUALS = SELF_DUAL_MVR  # Alias for compatibility
SIMPLE_OPS = ['+', '-', '->', '<-']

def generate_alternating_operators(chain_length: int) -> dict:
    """Generate equation with alternating +/- operators."""
    elements = []
    for i in range(chain_length):
        world = WORLD_LETTERS[i % 4]
        noetic = random.choice(NOETIC_NUMBERS)
        elements.append(f'{world}{noetic}')
    ops = ['+' if i % 2 == 0 else '-' for i in range(chain_length - 1)]
    eq_parts = [elements[0]]
    for i, op in enumerate(ops):
        eq_parts.extend([op, elements[i + 1]])
    return {'equation': ' '.join(eq_parts), 'pattern_type': 'alternating_operators', 'chain_length': chain_length}

def generate_long_distance_sparse(chain_length: int) -> dict:
    """Generate equation with boundary ops different from middle."""
    elements = []
    for i in range(chain_length):
        world = WORLD_LETTERS[i % 4]
        noetic = random.choice(NOETIC_NUMBERS)
        elements.append(f'{world}{noetic}')
    ops = []
    for i in range(chain_length - 1):
        if i == 0 or i == chain_length - 2:
            ops.append('->')
        else:
            ops.append('o')
    eq_parts = [elements[0]]
    for i, op in enumerate(ops):
        eq_parts.extend([op, elements[i + 1]])
    return {'equation': ' '.join(eq_parts), 'pattern_type': 'long_distance_sparse', 'chain_length': chain_length}

def generate_banded_operators(chain_length: int) -> dict:
    """Generate equation with first half + ops, second half - ops."""
    elements = []
    for i in range(chain_length):
        world = WORLD_LETTERS[i % 4]
        noetic = random.choice(NOETIC_NUMBERS)
        elements.append(f'{world}{noetic}')
    mid = (chain_length - 1) // 2
    ops = ['+' if i < mid else '-' for i in range(chain_length - 1)]
    eq_parts = [elements[0]]
    for i, op in enumerate(ops):
        eq_parts.extend([op, elements[i + 1]])
    return {'equation': ' '.join(eq_parts), 'pattern_type': 'banded_operators', 'chain_length': chain_length}

def generate_self_dual_skip(chain_length: int) -> dict:
    """Generate equation using only self-dual noetics (1, 4, 7, 10)."""
    elements = []
    for i in range(chain_length):
        world = WORLD_LETTERS[i % 4]
        noetic = random.choice(SELF_DUALS)
        elements.append(f'{world}{noetic}')
    ops = [random.choice(SIMPLE_OPS) for _ in range(chain_length - 1)]
    eq_parts = [elements[0]]
    for i, op in enumerate(ops):
        eq_parts.extend([op, elements[i + 1]])
    return {'equation': ' '.join(eq_parts), 'pattern_type': 'self_dual_skip', 'chain_length': chain_length}

def generate_involution_sparse(chain_length: int) -> dict:
    """Generate equation with involution pairs separated by gaps."""
    elements = []
    pairs_used = list(INVOLUTION_PAIRS)
    random.shuffle(pairs_used)
    for i in range(chain_length):
        world = WORLD_LETTERS[i % 4]
        if i < len(pairs_used) * 2:
            pair_idx = i // 2
            elem_idx = i % 2
            noetic = pairs_used[pair_idx % len(pairs_used)][elem_idx]
        else:
            noetic = random.choice(SELF_DUALS)
        elements.append(f'{world}{noetic}')
    ops = [random.choice(SIMPLE_OPS) for _ in range(chain_length - 1)]
    eq_parts = [elements[0]]
    for i, op in enumerate(ops):
        eq_parts.extend([op, elements[i + 1]])
    return {'equation': ' '.join(eq_parts), 'pattern_type': 'involution_sparse', 'chain_length': chain_length}

def generate_world_skip(chain_length: int) -> dict:
    """Generate equation using alternating worlds (A,C or B,D)."""
    world_pair = random.choice([['A', 'C'], ['B', 'D']])
    elements = []
    for i in range(chain_length):
        world = world_pair[i % 2]
        noetic = random.choice(NOETIC_NUMBERS)
        elements.append(f'{world}{noetic}')
    ops = [random.choice(SIMPLE_OPS) for _ in range(chain_length - 1)]
    eq_parts = [elements[0]]
    for i, op in enumerate(ops):
        eq_parts.extend([op, elements[i + 1]])
    return {'equation': ' '.join(eq_parts), 'pattern_type': 'world_skip', 'chain_length': chain_length}
GENERATORS = [generate_alternating_operators, generate_long_distance_sparse, generate_banded_operators, generate_self_dual_skip, generate_involution_sparse, generate_world_skip]

def generate_benchmark(count: int, seed: int=42) -> list:
    """Generate full benchmark dataset."""
    random.seed(seed)
    entries = []
    chain_lengths = [4, 5, 6]
    per_combo = count // (len(GENERATORS) * len(chain_lengths))
    remainder = count % (len(GENERATORS) * len(chain_lengths))
    for gen_func in GENERATORS:
        for length in chain_lengths:
            n = per_combo + (1 if remainder > 0 else 0)
            remainder -= 1 if remainder > 0 else 0
            for _ in range(n):
                entry = gen_func(length)
                entry['canon_score'] = 1.0
                entries.append(entry)
    random.shuffle(entries)
    return entries

def main():
    parser = argparse.ArgumentParser(description='TKS Lacunary Benchmark Generator')
    parser.add_argument('--count', type=int, default=75, help='Number of equations to generate')
    parser.add_argument('--output', '-o', default='data/lacunary_benchmark.jsonl', help='Output JSONL file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    print(f'Generating {args.count} lacunary benchmark equations...')
    entries = generate_benchmark(args.count, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    print(f'Written {len(entries)} equations to {args.output}')
    by_type = {}
    for e in entries:
        t = e['pattern_type']
        by_type[t] = by_type.get(t, 0) + 1
    print('\nPattern distribution:')
    for t, c in sorted(by_type.items()):
        print(f'  {t}: {c}')
    by_length = {}
    for e in entries:
        l = e['chain_length']
        by_length[l] = by_length.get(l, 0) + 1
    print('\nChain length distribution:')
    for l, c in sorted(by_length.items()):
        print(f'  {l} elements: {c}')
if __name__ == '__main__':
    main()

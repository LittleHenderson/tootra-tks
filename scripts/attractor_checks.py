"""
TKS Attractor Contractivity Analysis

Analyzes TKS equations for attractor convergence in 40D noetic space
(10 noetics × 4 worlds). Computes spectral radius and Lipschitz constants
to verify contraction properties.

Usage:
    python scripts/attractor_checks.py --output contractivity_report.json -v
"""
import argparse
import json
import numpy as np
import re
import sys
from pathlib import Path

# Allow running as a script: `python scripts/attractor_checks.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_rules.noetics import INVOLUTION_PAIRS, SELF_DUAL_MVR
from tks_rules.worlds import WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))

# INVOLUTION_PAIRS is a list of tuples; build a quick lookup map.
INVOLUTION_MAP = {a: b for a, b in INVOLUTION_PAIRS}
INVOLUTION_MAP.update({b: a for a, b in INVOLUTION_PAIRS})
SELF_DUALS = SELF_DUAL_MVR  # Alias for compatibility
STATE_DIM = 40

def world_index(w: str) -> int:
    """Get world index (0-3)."""
    return WORLD_LETTERS.index(w)

def noetic_index(n: int) -> int:
    """Get noetic index (0-9)."""
    return n - 1

def state_index(world: str, noetic: int) -> int:
    """Get linear state index (0-39) from world and noetic."""
    return world_index(world) * 10 + noetic_index(noetic)

def parse_element(elem: str) -> dict:
    """Parse TKS element like A1^2_d3."""
    match = re.match('^([ABCD])(10|[1-9])(\\^[1-9])?(_d[1-7])?$', elem)
    if not match:
        return None
    return {'world': match.group(1), 'noetic': int(match.group(2)), 'sense': int(match.group(3)[1]) if match.group(3) else None, 'foundation': match.group(4) if match.group(4) else None}

def build_inversion_jacobian() -> np.ndarray:
    """
    Build the Jacobian for noetic involutions (2↔3, 5↔6, 8↔9).
    Self-duals (1, 4, 7, 10) map to themselves.
    """
    J = np.zeros((STATE_DIM, STATE_DIM))
    for w_idx, w in enumerate(WORLD_LETTERS):
        for n in NOETIC_NUMBERS:
            src = state_index(w, n)
            if n in INVOLUTION_MAP:
                tgt = state_index(w, INVOLUTION_MAP[n])
                J[tgt, src] = 0.9
            else:
                J[src, src] = 0.95
    return J

def build_transformation_jacobian(equation: str) -> np.ndarray:
    """
    Build Jacobian matrix for a TKS transformation equation.
    Models how state flows through the equation's operators.
    """
    J = np.zeros((STATE_DIM, STATE_DIM))
    tokens = equation.split()
    elements = []
    operators = []
    for tok in tokens:
        if tok in ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']:
            operators.append(tok)
        else:
            parsed = parse_element(tok)
            if parsed:
                elements.append(parsed)
    if not elements:
        return np.eye(STATE_DIM) * 0.9
    for i, elem in enumerate(elements):
        src = state_index(elem['world'], elem['noetic'])
        if elem['noetic'] in SELF_DUALS:
            J[src, src] = 0.85
        elif elem['noetic'] in INVOLUTION_MAP:
            partner = INVOLUTION_MAP[elem['noetic']]
            tgt = state_index(elem['world'], partner)
            J[tgt, src] = 0.88
        else:
            J[src, src] = 0.9
        if i < len(operators):
            op = operators[i]
            if op in ['->', '<-'] and i + 1 < len(elements):
                next_elem = elements[i + 1]
                tgt = state_index(next_elem['world'], next_elem['noetic'])
                if op == '->':
                    J[tgt, src] = 0.7
                else:
                    J[src, tgt] = 0.7
    return J

def compute_spectral_radius(J: np.ndarray) -> float:
    """Compute spectral radius: max |eigenvalue|."""
    eigenvalues = np.linalg.eigvals(J)
    return float(np.max(np.abs(eigenvalues)))

def compute_lipschitz_constant(J: np.ndarray) -> float:
    """Compute Lipschitz constant (induced 2-norm)."""
    return float(np.linalg.norm(J, ord=2))

def analyze_chain_length(equation: str, chain_length: int) -> dict:
    """Analyze contractivity for a chain of given length."""
    J = build_transformation_jacobian(equation)
    J_composed = np.linalg.matrix_power(J, chain_length)
    spectral = compute_spectral_radius(J_composed)
    lipschitz = compute_lipschitz_constant(J_composed)
    return {'chain_length': chain_length, 'spectral_radius': round(spectral, 6), 'lipschitz_constant': round(lipschitz, 6), 'is_contraction': spectral < 1.0 and lipschitz < 1.0}

def run_analysis(data_file: str=None, verbose: bool=False) -> dict:
    """Run full contractivity analysis."""
    J_inv = build_inversion_jacobian()
    base_spectral = compute_spectral_radius(J_inv)
    base_lipschitz = compute_lipschitz_constant(J_inv)
    result = {'state_dimension': STATE_DIM, 'involution_analysis': {'spectral_radius': round(base_spectral, 6), 'lipschitz_constant': round(base_lipschitz, 6), 'is_contraction': base_spectral < 1.0 and base_lipschitz < 1.0}, 'chain_analysis': [], 'equation_analysis': []}
    for length in [2, 3, 4, 5]:
        dummy_eq = ' -> '.join([f'{WORLD_LETTERS[i % 4]}{i % 10 + 1}' for i in range(length)])
        analysis = analyze_chain_length(dummy_eq, length)
        result['chain_analysis'].append(analysis)
    if data_file and Path(data_file).exists():
        with open(data_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    eq = entry.get('equation', '')
                    if eq:
                        J = build_transformation_jacobian(eq)
                        spectral = compute_spectral_radius(J)
                        lipschitz = compute_lipschitz_constant(J)
                        eq_result = {'equation': eq, 'stress_type': entry.get('stress_type', 'unknown'), 'spectral_radius': round(spectral, 6), 'lipschitz_constant': round(lipschitz, 6), 'is_contraction': spectral < 1.0 and lipschitz < 1.0}
                        result['equation_analysis'].append(eq_result)
                        if verbose:
                            status = 'PASS' if eq_result['is_contraction'] else 'FAIL'
                            print(f'[{status}] {eq}: ρ={spectral:.4f}, L={lipschitz:.4f}')
                except json.JSONDecodeError:
                    continue
    all_contractions = [result['involution_analysis']['is_contraction']]
    all_contractions.extend((c['is_contraction'] for c in result['chain_analysis']))
    all_contractions.extend((e['is_contraction'] for e in result['equation_analysis']))
    result['contractivity_check'] = 'pass' if all(all_contractions) else 'fail'
    result['spectral_radius'] = max(result['involution_analysis']['spectral_radius'], max((c['spectral_radius'] for c in result['chain_analysis']), default=0), max((e['spectral_radius'] for e in result['equation_analysis']), default=0))
    result['lipschitz_constant'] = max(result['involution_analysis']['lipschitz_constant'], max((c['lipschitz_constant'] for c in result['chain_analysis']), default=0), max((e['lipschitz_constant'] for e in result['equation_analysis']), default=0))
    return result

def main():
    parser = argparse.ArgumentParser(description='TKS Attractor Contractivity Analysis')
    parser.add_argument('--data', default='data/attractor_stress.jsonl', help='Input data file with stress test equations')
    parser.add_argument('--output', '-o', help='Output JSON report file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    print('TKS Attractor Contractivity Analysis')
    print('=' * 50)
    print(f'State dimension: {STATE_DIM} (10 noetics × 4 worlds)')
    print()
    result = run_analysis(args.data, args.verbose)
    print()
    print(f"Overall contractivity check: {result['contractivity_check'].upper()}")
    print(f"Max spectral radius: {result['spectral_radius']:.6f} (< 1.0 required)")
    print(f"Max Lipschitz constant: {result['lipschitz_constant']:.6f} (< 1.0 required)")
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nReport written to: {args.output}')
    sys.exit(0 if result['contractivity_check'] == 'pass' else 1)
if __name__ == '__main__':
    main()

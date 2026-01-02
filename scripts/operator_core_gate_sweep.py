#!/usr/bin/env python3
"""
Operator Core Gate Sweep

Comprehensive evaluation across gate values to find optimal operator core strength.
Tests all noetic metrics: world separation, RPM variance, noetic consistency, attractor.

Usage:
    python scripts/operator_core_gate_sweep.py
    python scripts/operator_core_gate_sweep.py --gates 0.0 0.05 0.1 0.2 0.3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
from equation_detector import EquationDetector, element_to_index


# =============================================================================
# TEST DATA
# =============================================================================

# Equation-style world texts
EQUATION_WORLD_TEXTS = {
    'A': ["A1+A2+A3", "A4+A5+A6", "A7+A8+A9", "A1+A10"],
    'B': ["B1+B2+B3", "B4+B5+B6", "B7+B8+B9", "B1+B10"],
    'C': ["C1+C2+C3", "C4+C5+C6", "C7+C8+C9", "C1+C10"],
    'D': ["D1+D2+D3", "D4+D5+D6", "D7+D8+D9", "D1+D10"],
}

# NL-style world texts
NL_WORLD_TEXTS = {
    'A': ["Divine realm", "Spiritual awareness", "Sacred consciousness"],
    'B': ["Mental focus", "Cognitive clarity", "Thinking deeply"],
    'C': ["Emotional flow", "Feeling warmth", "Heart connection"],
    'D': ["Physical power", "Bodily energy", "Material strength"],
}

# Similar pairs (equation)
EQUATION_SIMILAR_PAIRS = [
    ("A1+A2+A3", "A4+A5+A6"),
    ("B1+B2+B3", "B4+B5+B6"),
    ("C1+C2+C3", "C4+C5+C6"),
    ("D1+D2+D3", "D4+D5+D6"),
]

# Dissimilar pairs (equation - different worlds)
EQUATION_DISSIMILAR_PAIRS = [
    ("A1+A2+A3", "D1+D2+D3"),
    ("B1+B2+B3", "C1+C2+C3"),
    ("A7+A8+A9", "B7+B8+B9"),
    ("C4+C5+C6", "D4+D5+D6"),
]

# NL similar pairs
NL_SIMILAR_PAIRS = [
    ("Mental focus and clarity", "Clear thinking and concentration"),
    ("Emotional warmth and love", "Feeling affection and care"),
    ("Physical energy and power", "Bodily strength and vitality"),
    ("Spiritual awareness", "Divine consciousness"),
]

# NL dissimilar pairs
NL_DISSIMILAR_PAIRS = [
    ("Mental focus and clarity", "Physical energy and power"),
    ("Emotional warmth and love", "Spiritual awareness"),
    ("Divine realm", "Material strength"),
    ("Cognitive clarity", "Heart connection"),
]

# RPM test cases (equation)
EQUATION_RPM_CASES = {
    'desire': ["A1+A4+A7", "B1+B4+B7", "C1+C4+C7"],
    'wisdom': ["A5+A6", "B5+B6", "C5+C6"],
    'power': ["A8+A9", "B8+B9", "C8+C9"],
}

# RPM test cases (NL)
NL_RPM_CASES = {
    'desire': ["I want power and control", "Desire for manifestation", "Vibration of intention"],
    'wisdom': ["Understanding the balance", "Male and female principles", "Wisdom of equilibrium"],
    'power': ["Cause and effect in action", "The power of consequence", "Forces shaping reality"],
}

# Binary equations for operator consistency
BINARY_EQUATIONS = [
    ("A1", "+", "A2"), ("A3", "+", "A4"),
    ("B1", "-", "B2"), ("B3", "-", "B4"),
    ("C1", "×", "C2"), ("C3", "×", "C4"),
    ("D1", "÷", "D2"), ("D3", "÷", "D4"),
]

OP_MAP = {'+': 0, '-': 1, '×': 2, '*': 2, '÷': 3, '/': 3}


# =============================================================================
# TOKENIZER
# =============================================================================

class SimpleTokenizer:
    def __init__(self, max_length: int = 64):
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)

    def _build_vocab(self) -> Dict[str, int]:
        vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        chars = 'ABCDabcdEFGHIJKLMNOPQRSTUVWXYZefghijklmnopqrstuvwxyz'
        chars += '0123456789+-×÷*/ \n\t.,!?\'"'
        for c in chars:
            if c not in vocab:
                vocab[c] = len(vocab)
        return vocab

    def tokenize(self, text: str) -> List[int]:
        tokens = [self.vocab.get('<bos>', 2)]
        for c in text[:self.max_length - 2]:
            tokens.append(self.vocab.get(c, self.vocab.get('<unk>', 1)))
        tokens.append(self.vocab.get('<eos>', 3))
        return tokens


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class MetricsResult:
    world_separation_eq: float
    world_separation_nl: float
    noetic_consistency_eq: float
    noetic_consistency_nl: float
    rpm_variance_eq: float
    rpm_variance_nl: float
    attractor_convergence: float
    symmetric_violation: float
    antisymmetric_violation: float

    def to_dict(self) -> Dict:
        return {
            'world_separation_eq': self.world_separation_eq,
            'world_separation_nl': self.world_separation_nl,
            'noetic_consistency_eq': self.noetic_consistency_eq,
            'noetic_consistency_nl': self.noetic_consistency_nl,
            'rpm_variance_eq': self.rpm_variance_eq,
            'rpm_variance_nl': self.rpm_variance_nl,
            'attractor_convergence': self.attractor_convergence,
            'symmetric_violation': self.symmetric_violation,
            'antisymmetric_violation': self.antisymmetric_violation,
        }


def get_noetic_output(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    text: str,
    device: torch.device,
    use_triplet: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Get noetic output and convergence status."""
    tokens = tokenizer.tokenize(text)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    triplet = None
    if use_triplet and model.operator_core is not None:
        detector = EquationDetector()
        parsed = detector.parse_single(text)
        if parsed:
            eq = parsed[0]
            triplet = (
                torch.tensor([eq.left_idx], device=device),
                torch.tensor([eq.right_idx], device=device),
                torch.tensor([eq.operator_idx], device=device),
            )

    with torch.no_grad():
        output = model(input_tensor, equation_triplet=triplet)

    gated = output['gated_output'][0]  # [seq, 40]
    noetic = gated.mean(dim=0)  # [40]
    converged = output.get('attractor_converged', True)

    return noetic, converged


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a.norm()
    b_norm = b.norm()
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(torch.dot(a, b) / (a_norm * b_norm))


def compute_world_separation(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    world_texts: Dict[str, List[str]],
    use_triplet: bool = False,
) -> float:
    correct = 0
    total = 0

    for target_world, texts in world_texts.items():
        for text in texts:
            noetic, _ = get_noetic_output(model, tokenizer, text, device, use_triplet)

            activations = {
                'A': float(noetic[0:10].norm()),
                'B': float(noetic[10:20].norm()),
                'C': float(noetic[20:30].norm()),
                'D': float(noetic[30:40].norm()),
            }

            predicted = max(activations, key=activations.get)
            if predicted == target_world:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def compute_noetic_consistency(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    similar_pairs: List[Tuple[str, str]],
    dissimilar_pairs: List[Tuple[str, str]],
    use_triplet: bool = False,
) -> float:
    similar_scores = []
    for t1, t2 in similar_pairs:
        n1, _ = get_noetic_output(model, tokenizer, t1, device, use_triplet)
        n2, _ = get_noetic_output(model, tokenizer, t2, device, use_triplet)
        similar_scores.append(cosine_sim(n1, n2))

    dissimilar_scores = []
    for t1, t2 in dissimilar_pairs:
        n1, _ = get_noetic_output(model, tokenizer, t1, device, use_triplet)
        n2, _ = get_noetic_output(model, tokenizer, t2, device, use_triplet)
        dissimilar_scores.append(cosine_sim(n1, n2))

    mean_sim = np.mean(similar_scores)
    mean_dissim = np.mean(dissimilar_scores)
    return float(mean_sim - mean_dissim)


def compute_rpm_variance(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    rpm_cases: Dict[str, List[str]],
    use_triplet: bool = False,
) -> float:
    category_means = {}

    for category, texts in rpm_cases.items():
        norms = []
        for text in texts:
            noetic, _ = get_noetic_output(model, tokenizer, text, device, use_triplet)
            norms.append(float(noetic.norm()))
        category_means[category] = np.mean(norms)

    return float(np.var(list(category_means.values())))


def compute_attractor_convergence(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    texts: List[str],
    use_triplet: bool = False,
) -> float:
    converged = 0
    total = 0

    for text in texts:
        _, conv = get_noetic_output(model, tokenizer, text, device, use_triplet)
        if conv:
            converged += 1
        total += 1

    return converged / total if total > 0 else 0.0


def compute_operator_consistency(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (symmetric_violation, antisymmetric_violation)."""
    sym_violations = []
    asym_violations = []

    for left_str, op, right_str in BINARY_EQUATIONS:
        left_idx = element_to_index(left_str)
        right_idx = element_to_index(right_str)
        op_idx = OP_MAP.get(op, 0)

        text_ab = f"{left_str}{op}{right_str}"
        text_ba = f"{right_str}{op}{left_str}"

        triplet_ab = None
        triplet_ba = None

        if model.operator_core is not None:
            triplet_ab = (
                torch.tensor([left_idx], device=device),
                torch.tensor([right_idx], device=device),
                torch.tensor([op_idx], device=device),
            )
            triplet_ba = (
                torch.tensor([right_idx], device=device),
                torch.tensor([left_idx], device=device),
                torch.tensor([op_idx], device=device),
            )

        tokens_ab = tokenizer.tokenize(text_ab)
        tokens_ba = tokenizer.tokenize(text_ba)

        input_ab = torch.tensor([tokens_ab], dtype=torch.long, device=device)
        input_ba = torch.tensor([tokens_ba], dtype=torch.long, device=device)

        with torch.no_grad():
            out_ab = model(input_ab, equation_triplet=triplet_ab)
            out_ba = model(input_ba, equation_triplet=triplet_ba)

        n_ab = out_ab['gated_output'][0].mean(dim=0)
        n_ba = out_ba['gated_output'][0].mean(dim=0)

        is_symmetric = (op in ['+', '×', '*'])

        if is_symmetric:
            violation = (n_ab - n_ba).abs().mean().item()
            sym_violations.append(violation)
        else:
            violation = (n_ab + n_ba).abs().mean().item()
            asym_violations.append(violation)

    sym_mean = np.mean(sym_violations) if sym_violations else 0.0
    asym_mean = np.mean(asym_violations) if asym_violations else 0.0

    return float(sym_mean), float(asym_mean)


def run_full_metrics(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    use_triplet: bool = True,
) -> MetricsResult:
    """Run full metrics suite."""

    # World separation
    ws_eq = compute_world_separation(model, tokenizer, device, EQUATION_WORLD_TEXTS, use_triplet)
    ws_nl = compute_world_separation(model, tokenizer, device, NL_WORLD_TEXTS, use_triplet=False)

    # Noetic consistency
    nc_eq = compute_noetic_consistency(
        model, tokenizer, device,
        EQUATION_SIMILAR_PAIRS, EQUATION_DISSIMILAR_PAIRS,
        use_triplet
    )
    nc_nl = compute_noetic_consistency(
        model, tokenizer, device,
        NL_SIMILAR_PAIRS, NL_DISSIMILAR_PAIRS,
        use_triplet=False
    )

    # RPM variance
    rpm_eq = compute_rpm_variance(model, tokenizer, device, EQUATION_RPM_CASES, use_triplet)
    rpm_nl = compute_rpm_variance(model, tokenizer, device, NL_RPM_CASES, use_triplet=False)

    # Attractor convergence
    all_eq_texts = [t for texts in EQUATION_WORLD_TEXTS.values() for t in texts]
    attractor = compute_attractor_convergence(model, tokenizer, device, all_eq_texts, use_triplet)

    # Operator consistency
    sym_viol, asym_viol = compute_operator_consistency(model, tokenizer, device)

    return MetricsResult(
        world_separation_eq=ws_eq,
        world_separation_nl=ws_nl,
        noetic_consistency_eq=nc_eq,
        noetic_consistency_nl=nc_nl,
        rpm_variance_eq=rpm_eq,
        rpm_variance_nl=rpm_nl,
        attractor_convergence=attractor,
        symmetric_violation=sym_viol,
        antisymmetric_violation=asym_viol,
    )


# =============================================================================
# GATE SWEEP
# =============================================================================

def set_gate_value(model: TKSNoeticLM, gate_value: float):
    """Set operator core gate to specific value."""
    if model.operator_core is None:
        return

    # The gate is stored as gate_logits, we need to set it so sigmoid(logits) = gate_value
    # sigmoid(x) = gate_value => x = logit(gate_value) = log(gate_value / (1 - gate_value))
    if gate_value <= 0:
        logit = -10.0  # Very negative = gate ~0
    elif gate_value >= 1:
        logit = 10.0   # Very positive = gate ~1
    else:
        logit = np.log(gate_value / (1 - gate_value))

    with torch.no_grad():
        model.operator_core.gate.gate_logits.fill_(logit)


def run_gate_sweep(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = SimpleTokenizer()

    # Create model with operator core
    config = TKSNoeticLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        num_layers=4,
        use_operator_core=True,
        operator_gate_init=0.1,
    )

    model = TKSNoeticLM(config).to(device)

    # Load pretrained operator core
    if os.path.exists(args.operator_path):
        state_dict = torch.load(args.operator_path, map_location=device, weights_only=True)
        model.operator_core.load_state_dict(state_dict)
        print(f"Loaded operator core from: {args.operator_path}")
    else:
        print(f"Warning: Using random operator core (no pretrained weights)")

    model.eval()

    # Parse gate values
    gate_values = [float(g) for g in args.gates]

    print("\n" + "=" * 80)
    print("OPERATOR CORE GATE SWEEP")
    print("=" * 80)
    print(f"Gates: {gate_values}")
    print("=" * 80)

    results = {}

    for gate in gate_values:
        print(f"\n--- Gate = {gate:.2f} ---")

        set_gate_value(model, gate)

        # Verify gate value
        actual_gate = torch.sigmoid(model.operator_core.gate.gate_logits).mean().item()
        print(f"  Actual gate (sigmoid): {actual_gate:.4f}")

        metrics = run_full_metrics(model, tokenizer, device, use_triplet=(gate > 0))
        results[gate] = metrics.to_dict()

        print(f"  World Sep (EQ): {metrics.world_separation_eq:.1%}")
        print(f"  World Sep (NL): {metrics.world_separation_nl:.1%}")
        print(f"  Noetic Cons (EQ): {metrics.noetic_consistency_eq:.4f}")
        print(f"  Noetic Cons (NL): {metrics.noetic_consistency_nl:.4f}")
        print(f"  RPM Var (EQ): {metrics.rpm_variance_eq:.6f}")
        print(f"  RPM Var (NL): {metrics.rpm_variance_nl:.6f}")
        print(f"  Attractor Conv: {metrics.attractor_convergence:.0%}")
        print(f"  Sym Violation: {metrics.symmetric_violation:.4f}")
        print(f"  Asym Violation: {metrics.antisymmetric_violation:.4f}")

    # ==========================================================================
    # Analysis
    # ==========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find best gate for each metric
    def find_best(metric_name: str, higher_is_better: bool = True):
        if higher_is_better:
            best_gate = max(results.keys(), key=lambda g: results[g][metric_name])
        else:
            best_gate = min(results.keys(), key=lambda g: results[g][metric_name])
        return best_gate, results[best_gate][metric_name]

    print("\nBest gate per metric:")
    print(f"  World Sep (EQ):    gate={find_best('world_separation_eq')[0]:.2f} -> {find_best('world_separation_eq')[1]:.1%}")
    print(f"  World Sep (NL):    gate={find_best('world_separation_nl')[0]:.2f} -> {find_best('world_separation_nl')[1]:.1%}")
    print(f"  Noetic Cons (EQ):  gate={find_best('noetic_consistency_eq')[0]:.2f} -> {find_best('noetic_consistency_eq')[1]:.4f}")
    print(f"  Noetic Cons (NL):  gate={find_best('noetic_consistency_nl')[0]:.2f} -> {find_best('noetic_consistency_nl')[1]:.4f}")
    print(f"  Attractor Conv:    gate={find_best('attractor_convergence')[0]:.2f} -> {find_best('attractor_convergence')[1]:.0%}")
    print(f"  Sym Violation:     gate={find_best('symmetric_violation', False)[0]:.2f} -> {find_best('symmetric_violation', False)[1]:.4f}")
    print(f"  Asym Violation:    gate={find_best('antisymmetric_violation', False)[0]:.2f} -> {find_best('antisymmetric_violation', False)[1]:.4f}")

    # Delta from gate=0 (baseline)
    if 0.0 in results:
        print("\n" + "-" * 40)
        print("Delta from gate=0 (baseline):")
        print("-" * 40)
        baseline = results[0.0]

        for gate in sorted(results.keys()):
            if gate == 0.0:
                continue
            r = results[gate]
            print(f"\nGate = {gate:.2f}:")
            print(f"  World Sep (EQ): {baseline['world_separation_eq']:.1%} -> {r['world_separation_eq']:.1%} ({r['world_separation_eq'] - baseline['world_separation_eq']:+.1%})")
            print(f"  World Sep (NL): {baseline['world_separation_nl']:.1%} -> {r['world_separation_nl']:.1%} ({r['world_separation_nl'] - baseline['world_separation_nl']:+.1%})")
            print(f"  Noetic Cons (EQ): {baseline['noetic_consistency_eq']:.4f} -> {r['noetic_consistency_eq']:.4f} ({r['noetic_consistency_eq'] - baseline['noetic_consistency_eq']:+.4f})")
            print(f"  Noetic Cons (NL): {baseline['noetic_consistency_nl']:.4f} -> {r['noetic_consistency_nl']:.4f} ({r['noetic_consistency_nl'] - baseline['noetic_consistency_nl']:+.4f})")
            print(f"  Attractor: {baseline['attractor_convergence']:.0%} -> {r['attractor_convergence']:.0%} ({r['attractor_convergence'] - baseline['attractor_convergence']:+.0%})")

    # Recommend optimal gate
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Score each gate (weighted combination)
    def score_gate(gate: float) -> float:
        r = results[gate]
        # Prioritize: equation metrics + attractor stability
        score = (
            r['world_separation_eq'] * 2.0 +
            r['noetic_consistency_eq'] * 1.0 +
            r['attractor_convergence'] * 1.5 +
            (1.0 - r['symmetric_violation']) * 0.5 +
            (1.0 - r['antisymmetric_violation']) * 0.5 +
            r['world_separation_nl'] * 0.5 +  # Don't hurt NL too much
            r['noetic_consistency_nl'] * 0.3
        )
        return score

    scored = [(g, score_gate(g)) for g in results.keys()]
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"\nGate scores (higher = better):")
    for gate, score in scored:
        print(f"  gate={gate:.2f}: {score:.3f}")

    optimal_gate = scored[0][0]
    print(f"\nOptimal gate: {optimal_gate:.2f}")

    # Save results
    if args.output:
        output_data = {
            'gate_values': gate_values,
            'results': {str(k): v for k, v in results.items()},
            'optimal_gate': optimal_gate,
            'scores': {str(g): s for g, s in scored},
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("=" * 80)

    return results, optimal_gate


def main():
    parser = argparse.ArgumentParser(description="Operator Core Gate Sweep")
    parser.add_argument("--operator-path", type=str,
                        default="output/operator_core_pretrained.pt",
                        help="Path to pretrained operator core")
    parser.add_argument("--gates", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.3],
                        help="Gate values to test")
    parser.add_argument("--output", type=str,
                        default="output/gate_sweep_results.json",
                        help="Output JSON path")

    args = parser.parse_args()
    run_gate_sweep(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test Operator Core Integration

Runs noetic coherence tests comparing:
1. Baseline v4 model (no operator core)
2. v4 model with pretrained operator core (on equation prompts)

This tests whether the operator core provides a measurable improvement
on equation-style inputs like "A1+B2" or "C3×D4".

Usage:
    python scripts/test_operator_core_integration.py
    python scripts/test_operator_core_integration.py --operator-path output/operator_core_pretrained.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v4 import (
    TKSNoeticLM,
    TKSNoeticLMConfig,
    load_pretrained_operator_core,
)
from equation_detector import EquationDetector, element_to_index


# =============================================================================
# TEST DATA - Equation-style prompts
# =============================================================================

# World-specific equation prompts
EQUATION_WORLD_TEXTS = {
    'A': ["A1+A2+A3", "A4+A5+A6", "A7+A8+A9", "A1+A10"],
    'B': ["B1+B2+B3", "B4+B5+B6", "B7+B8+B9", "B1+B10"],
    'C': ["C1+C2+C3", "C4+C5+C6", "C7+C8+C9", "C1+C10"],
    'D': ["D1+D2+D3", "D4+D5+D6", "D7+D8+D9", "D1+D10"],
}

# Binary equation prompts for operator core
BINARY_EQUATIONS = [
    ("A1", "+", "A2"),
    ("A3", "+", "A4"),
    ("B1", "-", "B2"),
    ("B3", "-", "B4"),
    ("C1", "×", "C2"),
    ("C3", "×", "C4"),
    ("D1", "÷", "D2"),
    ("D3", "÷", "D4"),
    # Cross-world
    ("A1", "+", "B1"),
    ("B2", "-", "C2"),
    ("C3", "×", "D3"),
    ("D4", "÷", "A4"),
]

# Operator index mapping
OP_MAP = {'+': 0, '-': 1, '×': 2, '*': 2, '÷': 3, '/': 3}


# =============================================================================
# SIMPLE TOKENIZER
# =============================================================================

class SimpleTokenizer:
    """Minimal tokenizer for testing."""

    def __init__(self, max_length: int = 64):
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)

    def _build_vocab(self) -> Dict[str, int]:
        vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}

        # Add characters
        for c in 'ABCDabcd':
            vocab[c] = len(vocab)
        for c in '0123456789':
            vocab[c] = len(vocab)
        for c in '+-×÷*/':
            vocab[c] = len(vocab)
        for c in ' \n\t':
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

def get_world_activations(noetic_state: torch.Tensor) -> Dict[str, float]:
    """Get activation norm for each world."""
    return {
        'A': float(noetic_state[0:10].norm()),
        'B': float(noetic_state[10:20].norm()),
        'C': float(noetic_state[20:30].norm()),
        'D': float(noetic_state[30:40].norm()),
    }


def compute_world_separation(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    equation_triplets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Dict:
    """
    Compute world separation metric with optional operator core.

    Args:
        model: TKSNoeticLM
        tokenizer: Tokenizer
        device: Device
        equation_triplets: Optional triplets for operator core

    Returns:
        Dict with metrics
    """
    correct = 0
    total = 0
    details = []

    for target_world, texts in EQUATION_WORLD_TEXTS.items():
        for text in texts:
            # Tokenize
            tokens = tokenizer.tokenize(text)
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

            # Get equation triplet if available
            triplet = None
            if equation_triplets is not None:
                detector = EquationDetector()
                parsed = detector.parse_single(text)
                if parsed:
                    eq = parsed[0]
                    triplet = (
                        torch.tensor([eq.left_idx], device=device),
                        torch.tensor([eq.right_idx], device=device),
                        torch.tensor([eq.operator_idx], device=device),
                    )

            # Forward
            with torch.no_grad():
                output = model(input_tensor, equation_triplet=triplet)

            gated = output['gated_output'][0]  # [seq, 40]
            noetic = gated.mean(dim=0)  # [40]

            activations = get_world_activations(noetic)
            predicted = max(activations, key=activations.get)
            is_correct = (predicted == target_world)

            if is_correct:
                correct += 1
            total += 1

            details.append({
                'text': text,
                'target': target_world,
                'predicted': predicted,
                'correct': is_correct,
                'activations': activations,
            })

    rate = correct / total if total > 0 else 0.0
    return {
        'correct': correct,
        'total': total,
        'rate': rate,
        'details': details,
    }


def compute_operator_consistency(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    use_operator_core: bool = True,
) -> Dict:
    """
    Test operator symmetry/anti-symmetry consistency.

    For + and ×: f(A,B) should approximate f(B,A)
    For - and ÷: f(A,B) should approximate -f(B,A)
    """
    results = []

    for left_str, op, right_str in BINARY_EQUATIONS:
        left_idx = element_to_index(left_str)
        right_idx = element_to_index(right_str)
        op_idx = OP_MAP.get(op, 0)

        text_ab = f"{left_str}{op}{right_str}"
        text_ba = f"{right_str}{op}{left_str}"

        # Tokenize both
        tokens_ab = tokenizer.tokenize(text_ab)
        tokens_ba = tokenizer.tokenize(text_ba)

        input_ab = torch.tensor([tokens_ab], dtype=torch.long, device=device)
        input_ba = torch.tensor([tokens_ba], dtype=torch.long, device=device)

        triplet_ab = None
        triplet_ba = None

        if use_operator_core:
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

        with torch.no_grad():
            out_ab = model(input_ab, equation_triplet=triplet_ab)
            out_ba = model(input_ba, equation_triplet=triplet_ba)

        noetic_ab = out_ab['gated_output'][0].mean(dim=0)  # [40]
        noetic_ba = out_ba['gated_output'][0].mean(dim=0)  # [40]

        is_symmetric = (op in ['+', '×', '*'])

        if is_symmetric:
            # f(A,B) - f(B,A) should be ~0
            diff = (noetic_ab - noetic_ba).abs().mean().item()
        else:
            # f(A,B) + f(B,A) should be ~0
            diff = (noetic_ab + noetic_ba).abs().mean().item()

        results.append({
            'equation': f"{left_str}{op}{right_str}",
            'operator': op,
            'is_symmetric': is_symmetric,
            'violation': diff,
        })

    sym_violations = [r['violation'] for r in results if r['is_symmetric']]
    asym_violations = [r['violation'] for r in results if not r['is_symmetric']]

    return {
        'symmetric_mean': np.mean(sym_violations) if sym_violations else 0.0,
        'antisymmetric_mean': np.mean(asym_violations) if asym_violations else 0.0,
        'details': results,
    }


def compute_attractor_convergence(
    model: TKSNoeticLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    use_operator_core: bool = False,
) -> Dict:
    """Test attractor convergence on equation prompts."""
    texts = [eq for eqs in EQUATION_WORLD_TEXTS.values() for eq in eqs]

    converged = 0
    total = 0

    for text in texts:
        tokens = tokenizer.tokenize(text)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

        triplet = None
        if use_operator_core:
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

        if output.get('attractor_converged', True):
            converged += 1
        total += 1

    return {
        'converged': converged,
        'total': total,
        'rate': converged / total if total > 0 else 0.0,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_comparison(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = SimpleTokenizer()

    # Create config
    config_base = TKSNoeticLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        num_layers=4,
        use_operator_core=False,
    )

    config_with_op = TKSNoeticLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        num_layers=4,
        use_operator_core=True,
        operator_gate_init=0.1,
    )

    print("\n" + "=" * 70)
    print("OPERATOR CORE INTEGRATION TEST")
    print("=" * 70)

    # ==========================================================================
    # Test 1: Baseline (no operator core)
    # ==========================================================================
    print("\n--- Baseline Model (no operator core) ---")

    model_base = TKSNoeticLM(config_base).to(device)
    model_base.eval()

    ws_base = compute_world_separation(model_base, tokenizer, device, equation_triplets=None)
    ac_base = compute_attractor_convergence(model_base, tokenizer, device, use_operator_core=False)
    oc_base = compute_operator_consistency(model_base, tokenizer, device, use_operator_core=False)

    print(f"  World Separation: {ws_base['rate']:.1%} ({ws_base['correct']}/{ws_base['total']})")
    print(f"  Attractor Convergence: {ac_base['rate']:.0%}")
    print(f"  Operator Consistency:")
    print(f"    Symmetric violation: {oc_base['symmetric_mean']:.4f}")
    print(f"    Anti-symmetric violation: {oc_base['antisymmetric_mean']:.4f}")

    # ==========================================================================
    # Test 2: With Operator Core (pretrained)
    # ==========================================================================
    print("\n--- Model with Pretrained Operator Core ---")

    model_op = TKSNoeticLM(config_with_op).to(device)

    if os.path.exists(args.operator_path):
        load_pretrained_operator_core(model_op, args.operator_path, device)
    else:
        print(f"Warning: Operator core weights not found at {args.operator_path}")
        print("  Using randomly initialized operator core")

    model_op.eval()

    # Use triplets for forward pass
    ws_op = compute_world_separation(model_op, tokenizer, device, equation_triplets=True)
    ac_op = compute_attractor_convergence(model_op, tokenizer, device, use_operator_core=True)
    oc_op = compute_operator_consistency(model_op, tokenizer, device, use_operator_core=True)

    print(f"  World Separation: {ws_op['rate']:.1%} ({ws_op['correct']}/{ws_op['total']})")
    print(f"  Attractor Convergence: {ac_op['rate']:.0%}")
    print(f"  Operator Consistency:")
    print(f"    Symmetric violation: {oc_op['symmetric_mean']:.4f}")
    print(f"    Anti-symmetric violation: {oc_op['antisymmetric_mean']:.4f}")

    # ==========================================================================
    # Comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    ws_delta = ws_op['rate'] - ws_base['rate']
    sym_delta = oc_base['symmetric_mean'] - oc_op['symmetric_mean']
    asym_delta = oc_base['antisymmetric_mean'] - oc_op['antisymmetric_mean']

    print(f"\nWorld Separation:")
    print(f"  Baseline: {ws_base['rate']:.1%}")
    print(f"  With Op Core: {ws_op['rate']:.1%}")
    print(f"  Delta: {ws_delta:+.1%}")

    print(f"\nOperator Consistency (lower = better):")
    print(f"  Symmetric violation:")
    print(f"    Baseline: {oc_base['symmetric_mean']:.4f}")
    print(f"    With Op Core: {oc_op['symmetric_mean']:.4f}")
    print(f"    Improvement: {sym_delta:+.4f}")

    print(f"  Anti-symmetric violation:")
    print(f"    Baseline: {oc_base['antisymmetric_mean']:.4f}")
    print(f"    With Op Core: {oc_op['antisymmetric_mean']:.4f}")
    print(f"    Improvement: {asym_delta:+.4f}")

    # ==========================================================================
    # Detailed breakdown
    # ==========================================================================
    if args.verbose:
        print("\n" + "=" * 70)
        print("DETAILED WORLD SEPARATION")
        print("=" * 70)

        print("\n--- Baseline ---")
        for d in ws_base['details']:
            status = "OK" if d['correct'] else "MISS"
            print(f"  [{status}] {d['text']:12} target={d['target']} pred={d['predicted']}")

        print("\n--- With Operator Core ---")
        for d in ws_op['details']:
            status = "OK" if d['correct'] else "MISS"
            print(f"  [{status}] {d['text']:12} target={d['target']} pred={d['predicted']}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    improved = (ws_delta > 0) or (sym_delta > 0.001) or (asym_delta > 0.001)
    if improved:
        print("Operator core shows improvement on equation prompts")
    else:
        print("Operator core shows no measurable improvement (may need more training)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test Operator Core Integration")
    parser.add_argument("--operator-path", type=str,
                        default="output/operator_core_pretrained.pt",
                        help="Path to pretrained operator core")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed per-equation results")

    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()

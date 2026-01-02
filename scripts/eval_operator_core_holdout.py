#!/usr/bin/env python3
"""
Evaluate Pretrained Operator Core on Holdout Set

Tests generalization of symmetry/anti-symmetry constraints
and corpus embedding alignment on unseen equations.

Usage:
    python scripts/eval_operator_core_holdout.py \
        --checkpoint output/operator_core_pretrained.pt \
        --corpus-path data/equation_embeddings/splits/val.pt

    python scripts/eval_operator_core_holdout.py \
        --checkpoint output/operator_core_pretrained.pt \
        --corpus-path data/equation_embeddings/splits/test.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_compositional_layer import TKSCompositionalLayerV2


class HoldoutDataset:
    """Simple dataset wrapper for holdout evaluation."""

    def __init__(self, corpus_path: str):
        corpus = torch.load(corpus_path, map_location='cpu', weights_only=False)

        self.embeddings = corpus['embeddings'].float()
        self.source_weights = corpus['source_weights'].float()
        self.left_indices = corpus['left_indices'].long()
        self.right_indices = corpus['right_indices'].long()
        self.operators = corpus['operators'].long()

        self.n = len(self.embeddings)

    def __len__(self):
        return self.n


def evaluate_symmetry(
    layer: TKSCompositionalLayerV2,
    dataset: HoldoutDataset,
    device: torch.device,
    num_samples: int = None,
) -> dict:
    """
    Evaluate symmetry/anti-symmetry constraint satisfaction on holdout.

    For symmetric ops (+, ×): f(A,B) should equal f(B,A)
    For anti-symmetric ops (-, ÷): f(A,B) should equal -f(B,A)
    """
    layer.eval()

    if num_samples is None:
        num_samples = len(dataset)

    # Sample indices
    indices = torch.randperm(len(dataset))[:num_samples]

    # Gather batch
    left_idx = dataset.left_indices[indices].to(device)
    right_idx = dataset.right_indices[indices].to(device)
    operators = dataset.operators[indices].to(device)

    # One-hot vectors
    left_vec = F.one_hot(left_idx, num_classes=40).float()
    right_vec = F.one_hot(right_idx, num_classes=40).float()

    with torch.no_grad():
        # Forward pass: f(A, B)
        out_ab = layer(left_vec, right_vec, operators, compute_symmetry=False)
        # Swapped: f(B, A)
        out_ba = layer(right_vec, left_vec, operators, compute_symmetry=False)

        noetic_ab = out_ab['noetic_output']
        noetic_ba = out_ba['noetic_output']

    # Compute violations
    results = {}

    # Symmetric operators: +, × (indices 0, 2)
    sym_mask = (operators == 0) | (operators == 2)
    if sym_mask.any():
        sym_diff = (noetic_ab[sym_mask] - noetic_ba[sym_mask]).abs().mean()
        results['symmetric_violation'] = sym_diff.item()
        results['symmetric_count'] = sym_mask.sum().item()
    else:
        results['symmetric_violation'] = 0.0
        results['symmetric_count'] = 0

    # Anti-symmetric operators: -, ÷ (indices 1, 3)
    asym_mask = (operators == 1) | (operators == 3)
    if asym_mask.any():
        # f(A,B) + f(B,A) should be ~0
        asym_diff = (noetic_ab[asym_mask] + noetic_ba[asym_mask]).abs().mean()
        results['antisymmetric_violation'] = asym_diff.item()
        results['antisymmetric_count'] = asym_mask.sum().item()
    else:
        results['antisymmetric_violation'] = 0.0
        results['antisymmetric_count'] = 0

    # Cosine similarity to target embeddings
    embeddings = dataset.embeddings[indices].to(device)
    eq_repr = out_ab['equation_repr']
    cosine_sim = F.cosine_similarity(eq_repr, embeddings, dim=-1).mean()
    results['cosine_similarity'] = cosine_sim.item()

    return results


def evaluate_operator_breakdown(
    layer: TKSCompositionalLayerV2,
    dataset: HoldoutDataset,
    device: torch.device,
) -> dict:
    """Evaluate per-operator metrics on holdout."""
    layer.eval()

    results = {}
    op_names = ['+', '-', '×', '÷']

    for op_idx, op_name in enumerate(op_names):
        mask = dataset.operators == op_idx
        if not mask.any():
            continue

        indices = mask.nonzero(as_tuple=True)[0][:256]

        left_idx = dataset.left_indices[indices].to(device)
        right_idx = dataset.right_indices[indices].to(device)
        operators = dataset.operators[indices].to(device)
        embeddings = dataset.embeddings[indices].to(device)

        left_vec = F.one_hot(left_idx, num_classes=40).float()
        right_vec = F.one_hot(right_idx, num_classes=40).float()

        with torch.no_grad():
            out = layer(left_vec, right_vec, operators, compute_symmetry=False)
            cosine = F.cosine_similarity(out['equation_repr'], embeddings, dim=-1).mean()

        results[f'cosine_{op_name}'] = cosine.item()
        results[f'count_{op_name}'] = len(indices)

    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate operator core on holdout")
    p.add_argument("--checkpoint", required=True,
                   help="Path to pretrained operator core checkpoint")
    p.add_argument("--corpus-path", required=True,
                   help="Path to holdout corpus PT file (val.pt or test.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    layer = TKSCompositionalLayerV2(
        element_dim=40,
        equation_dim=384,
        init_gate=0.1,
    ).to(device)
    layer.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    layer.eval()

    # Load holdout dataset
    print(f"Loading holdout corpus: {args.corpus_path}")
    dataset = HoldoutDataset(args.corpus_path)
    print(f"  Holdout size: {len(dataset)}")

    # Evaluate
    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION")
    print("=" * 70)

    # Symmetry metrics
    sym_metrics = evaluate_symmetry(layer, dataset, device, num_samples=min(1024, len(dataset)))

    print(f"\nSymmetry Metrics (n={sym_metrics['symmetric_count'] + sym_metrics['antisymmetric_count']}):")
    print(f"  Symmetric (+, ×) violation:     {sym_metrics['symmetric_violation']:.6f}")
    print(f"  Anti-symmetric (-, ÷) violation: {sym_metrics['antisymmetric_violation']:.6f}")
    print(f"  Cosine similarity to corpus:    {sym_metrics['cosine_similarity']:.4f}")

    # Per-operator breakdown
    op_metrics = evaluate_operator_breakdown(layer, dataset, device)

    print(f"\nPer-Operator Cosine Similarity:")
    for op in ['+', '-', '×', '÷']:
        key = f'cosine_{op}'
        count_key = f'count_{op}'
        if key in op_metrics:
            print(f"  {op}: {op_metrics[key]:.4f} (n={op_metrics[count_key]})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Pass/fail thresholds
    sym_pass = sym_metrics['symmetric_violation'] < 0.01
    asym_pass = sym_metrics['antisymmetric_violation'] < 0.15
    cosine_pass = sym_metrics['cosine_similarity'] > 0.3

    print(f"\n  Symmetric violation < 0.01:     {'PASS' if sym_pass else 'FAIL'} ({sym_metrics['symmetric_violation']:.6f})")
    print(f"  Anti-symmetric violation < 0.15: {'PASS' if asym_pass else 'FAIL'} ({sym_metrics['antisymmetric_violation']:.6f})")
    print(f"  Cosine similarity > 0.30:       {'PASS' if cosine_pass else 'FAIL'} ({sym_metrics['cosine_similarity']:.4f})")

    overall = sym_pass and asym_pass and cosine_pass
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

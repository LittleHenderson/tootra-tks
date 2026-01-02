#!/usr/bin/env python3
"""
Pretrain Operator Core on 6400 Equation Corpus

Trains TKSCompositionalLayerV2 standalone with:
- Contrastive alignment to corpus embeddings
- Symmetry/anti-symmetry operator constraints
- Source-weighted sampling (original > generated)

Output: output/operator_core_pretrained.pt

Usage:
    python scripts/pretrain_operator_core.py
    python scripts/pretrain_operator_core.py --epochs 20 --batch_size 128
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_compositional_layer import TKSCompositionalLayerV2


# =============================================================================
# DATASET WITH SOURCE WEIGHTING
# =============================================================================

class EquationCorpusDataset(Dataset):
    """Dataset for equation corpus with source tracking."""

    def __init__(self, corpus_path: str):
        corpus = torch.load(corpus_path, map_location='cpu', weights_only=False)

        self.embeddings = corpus['embeddings'].float()
        self.source_weights = corpus['source_weights'].float()
        self.left_indices = corpus['left_indices'].long()
        self.right_indices = corpus['right_indices'].long()
        self.operators = corpus['operators'].long()

        self.n = len(self.embeddings)

        # Compute sampling weights (original 2x more likely than generated)
        # source_weights: 1.0 for original, 0.5 for generated
        self.sampling_weights = self.source_weights.clone()
        self.sampling_weights[self.sampling_weights < 1.0] = 0.5  # generated
        self.sampling_weights[self.sampling_weights >= 1.0] = 1.0  # original

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'source_weight': self.source_weights[idx],
            'left_idx': self.left_indices[idx],
            'right_idx': self.right_indices[idx],
            'operator': self.operators[idx],
        }

    def get_sampler(self, num_samples: int = None):
        """Get weighted random sampler favoring original examples."""
        if num_samples is None:
            num_samples = self.n
        return WeightedRandomSampler(
            weights=self.sampling_weights.tolist(),
            num_samples=num_samples,
            replacement=True,
        )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_symmetry(
    layer: TKSCompositionalLayerV2,
    dataset: EquationCorpusDataset,
    device: torch.device,
    num_samples: int = 512,
) -> dict:
    """
    Evaluate symmetry/anti-symmetry constraint satisfaction.

    For symmetric ops (+, ×): f(A,B) should equal f(B,A)
    For anti-symmetric ops (-, ÷): f(A,B) should equal -f(B,A)
    """
    layer.eval()

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

    layer.train()
    return results


def evaluate_operator_breakdown(
    layer: TKSCompositionalLayerV2,
    dataset: EquationCorpusDataset,
    device: torch.device,
) -> dict:
    """Evaluate per-operator metrics."""
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

    layer.train()
    return results


# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading corpus from {args.corpus_path}...")
    dataset = EquationCorpusDataset(args.corpus_path)
    print(f"  Total equations: {len(dataset)}")
    print(f"  Original: {(dataset.source_weights >= 1.0).sum().item()}")
    print(f"  Generated: {(dataset.source_weights < 1.0).sum().item()}")

    # DataLoader with weighted sampling
    sampler = dataset.get_sampler(num_samples=len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Model
    layer = TKSCompositionalLayerV2(
        element_dim=40,
        equation_dim=384,
        init_gate=0.1,
    ).to(device)

    param_count = sum(p.numel() for p in layer.parameters())
    print(f"  Parameters: {param_count:,}")

    # Optimizer + scheduler
    optimizer = AdamW(layer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_cosine = -1.0

    for epoch in range(1, args.epochs + 1):
        layer.train()
        total_loss = 0.0
        total_contrastive = 0.0
        total_symmetry = 0.0
        steps = 0

        for batch in loader:
            left_idx = batch['left_idx'].to(device)
            right_idx = batch['right_idx'].to(device)
            operators = batch['operator'].to(device)
            embeddings = batch['embedding'].to(device)
            source_weights = batch['source_weight'].to(device)

            # One-hot element vectors
            left_vec = F.one_hot(left_idx, num_classes=40).float()
            right_vec = F.one_hot(right_idx, num_classes=40).float()

            # Forward
            outputs = layer(left_vec, right_vec, operators, compute_symmetry=True)

            # Compute loss
            losses = layer.compute_loss(
                outputs,
                embeddings,
                source_weights,
                symmetry_weight=args.symmetry_weight,
            )
            loss = losses['total']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_contrastive += losses['contrastive'].item()
            total_symmetry += losses['symmetry'].item()
            steps += 1

        scheduler.step()

        avg_loss = total_loss / steps
        avg_contrastive = total_contrastive / steps
        avg_symmetry = total_symmetry / steps

        # Evaluate
        eval_results = evaluate_symmetry(layer, dataset, device)
        op_results = evaluate_operator_breakdown(layer, dataset, device)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"loss={avg_loss:.4f} (contr={avg_contrastive:.4f} sym={avg_symmetry:.4f}) | "
              f"cosine={eval_results['cosine_similarity']:.4f} | "
              f"sym_viol={eval_results['symmetric_violation']:.4f} "
              f"asym_viol={eval_results['antisymmetric_violation']:.4f}")

        # Save best
        if eval_results['cosine_similarity'] > best_cosine:
            best_cosine = eval_results['cosine_similarity']
            torch.save(layer.state_dict(), args.output_path)

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Load best
    layer.load_state_dict(torch.load(args.output_path, map_location=device, weights_only=True))

    final_eval = evaluate_symmetry(layer, dataset, device, num_samples=1024)
    print(f"\nSymmetry Evaluation (n=1024):")
    print(f"  Symmetric (+, ×) violation:     {final_eval['symmetric_violation']:.6f}")
    print(f"  Anti-symmetric (-, ÷) violation: {final_eval['antisymmetric_violation']:.6f}")
    print(f"  Cosine similarity to corpus:    {final_eval['cosine_similarity']:.4f}")

    op_eval = evaluate_operator_breakdown(layer, dataset, device)
    print(f"\nPer-Operator Cosine Similarity:")
    for k, v in op_eval.items():
        print(f"  {k}: {v:.4f}")

    # Gate values
    gate_vals = torch.sigmoid(layer.gate.gate_logits).detach().cpu()
    print(f"\nGate Values (per world):")
    print(f"  A: {gate_vals[0]:.4f}, B: {gate_vals[1]:.4f}, "
          f"C: {gate_vals[2]:.4f}, D: {gate_vals[3]:.4f}")

    print(f"\nSaved: {args.output_path}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pretrain Operator Core")
    parser.add_argument("--corpus_path", type=str,
                        default="data/equation_embeddings/equation_corpus.pt",
                        help="Path to equation corpus")
    parser.add_argument("--output_path", type=str,
                        default="output/operator_core_pretrained.pt",
                        help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--symmetry_weight", type=float, default=0.1,
                        help="Weight for symmetry loss")

    args = parser.parse_args()

    # Create output dir
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()

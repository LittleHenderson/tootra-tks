"""
Negation Scope Experiment: Testing TKS Subtractive Hypothesis

This script runs a complete experimental comparison of three architectures:
- Condition A: Baseline Transformer
- Condition B: Subtractive-Only Block (separate inhibition attention)
- Condition C: Full 4-Operation Block

Task: Negation scope resolution on synthetic dataset

The key hypothesis: Models with separate inhibition attention (B, C) should
outperform baseline (A) on negation tasks, and mechanistic probes should show
that inhibition attention learns to focus on negated predicates.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
from pathlib import Path

# GPU-only enforcement (following project policy)
if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected. This script requires GPU for training.")
    print("Project policy: GPU ONLY - no CPU training allowed")
    sys.exit(1)

from tks_4op_block_v2 import (
    SimpleTransformerLM,
    SubtractiveLM,
    Full4OpLM
)
from negation_scope_dataset import (
    NegationScopeDataset,
    NegationQADataset,
    collate_fn,
    collate_fn_qa
)


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    inhib_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
    alpha_sub: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 20
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Data
    train_samples: int = 10000
    val_samples: int = 2000
    test_samples: int = 2000
    max_properties: int = 3
    negation_prob: float = 0.5

    # Experiment
    conditions: List[str] = None  # ["A", "B", "C"]
    seed: int = 42
    save_dir: str = "experiments/negation_scope"

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = ["A", "B", "C"]


@dataclass
class EvaluationResults:
    """Results from model evaluation."""
    condition: str
    loss: float
    perplexity: float
    accuracy: float
    num_params: int

    # Mechanistic probe results (if applicable)
    inhib_to_pred_attn: Optional[float] = None
    inhib_to_other_attn: Optional[float] = None
    inhib_attn_ratio: Optional[float] = None

    # Gate statistics (for conditions B and C)
    gate_stats: Optional[Dict] = None


class NegationScopeExperiment:
    """Main experiment runner."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda")

        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Build datasets
        print("Building datasets...")
        self.train_dataset = NegationScopeDataset(
            n_samples=config.train_samples,
            max_properties=config.max_properties,
            negation_prob=config.negation_prob,
            vocab_file=str(self.save_dir / "vocab.txt"),
            seed=config.seed
        )

        self.val_dataset = NegationScopeDataset(
            n_samples=config.val_samples,
            max_properties=config.max_properties,
            negation_prob=config.negation_prob,
            seed=config.seed + 1
        )

        self.test_dataset = NegationScopeDataset(
            n_samples=config.test_samples,
            max_properties=config.max_properties,
            negation_prob=config.negation_prob,
            seed=config.seed + 2
        )

        self.vocab_size = len(self.train_dataset.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

        # Build dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

    def create_model(self, condition: str) -> nn.Module:
        """Create model for specified condition."""
        if condition == "A":
            # Baseline Transformer
            model = SimpleTransformerLM(
                vocab_size=self.vocab_size,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout
            )
        elif condition == "B":
            # Subtractive-Only
            model = SubtractiveLM(
                vocab_size=self.vocab_size,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                inhib_heads=self.config.inhib_heads,
                n_layers=self.config.n_layers,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout,
                alpha_sub=self.config.alpha_sub
            )
        elif condition == "C":
            # Full 4-Operation
            model = Full4OpLM(
                vocab_size=self.vocab_size,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                inhib_heads=self.config.inhib_heads,
                n_layers=self.config.n_layers,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout,
                alpha_sub=self.config.alpha_sub
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        return model.to(self.device)

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        condition: str
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        total_tokens = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.grad_clip
            )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Track metrics
            total_loss += loss.item() * labels.numel()
            total_tokens += (labels != -100).sum().item()

            pbar.set_postfix({
                'loss': loss.item(),
                'ppl': np.exp(loss.item())
            })

        avg_loss = total_loss / total_tokens
        return {'loss': avg_loss, 'perplexity': np.exp(avg_loss)}

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        condition: str,
        run_probe: bool = False
    ) -> EvaluationResults:
        """Evaluate model on dataset."""
        model.eval()

        total_loss = 0
        total_tokens = 0
        total_correct = 0

        # For mechanistic probe
        inhib_to_pred_weights = []
        inhib_to_other_weights = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass (with attention if needed)
            if condition in ["B", "C"] and run_probe:
                if condition == "B":
                    logits, inhib_attns = model(input_ids, return_inhib_attn=True)
                else:  # C
                    logits, _, inhib_attns = model(
                        input_ids,
                        return_stats=False,
                        return_inhib_attn=True
                    )

                # Run mechanistic probe
                probe_results = self._mechanistic_probe(
                    inhib_attns=inhib_attns,
                    batch=batch
                )
                inhib_to_pred_weights.extend(probe_results['to_pred'])
                inhib_to_other_weights.extend(probe_results['to_other'])

            else:
                logits = model(input_ids)

            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

            total_loss += loss.item()

            # Compute accuracy
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()

        # Compute metrics
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        accuracy = total_correct / total_tokens
        num_params = sum(p.numel() for p in model.parameters())

        # Mechanistic probe results
        inhib_to_pred = None
        inhib_to_other = None
        inhib_ratio = None

        if run_probe and condition in ["B", "C"] and len(inhib_to_pred_weights) > 0:
            inhib_to_pred = np.mean(inhib_to_pred_weights)
            inhib_to_other = np.mean(inhib_to_other_weights)
            inhib_ratio = inhib_to_pred / (inhib_to_other + 1e-8)

        return EvaluationResults(
            condition=condition,
            loss=avg_loss,
            perplexity=perplexity,
            accuracy=accuracy,
            num_params=num_params,
            inhib_to_pred_attn=inhib_to_pred,
            inhib_to_other_attn=inhib_to_other,
            inhib_attn_ratio=inhib_ratio
        )

    def _mechanistic_probe(
        self,
        inhib_attns: List[torch.Tensor],
        batch: Dict
    ) -> Dict[str, List[float]]:
        """
        Mechanistic probe: Does inhibition attention focus on negated predicates?

        For sequences containing "not", check if inhib_attn at the "not" position
        attends more to the predicate than to other tokens.

        Args:
            inhib_attns: List of [B, H, T, T] attention tensors (one per layer)
            batch: Batch dict with 'not_positions' and 'property_positions'

        Returns:
            Dict with 'to_pred' and 'to_other' attention weights
        """
        to_pred_weights = []
        to_other_weights = []

        # Use last layer attention (most semantic)
        if len(inhib_attns) == 0:
            return {'to_pred': [], 'to_other': []}

        attn = inhib_attns[-1]  # [B, H, T, T]
        B = attn.size(0)

        for b in range(B):
            not_positions = batch['not_positions'][b]
            property_positions = batch['property_positions'][b]

            if len(not_positions) == 0 or len(property_positions) == 0:
                continue

            for not_idx in not_positions:
                # Get indices of properties mentioned in this sequence
                prop_indices = list(property_positions.values())

                if len(prop_indices) == 0:
                    continue

                # Attention from "not" to properties
                attn_to_pred = attn[b, :, not_idx, prop_indices].mean().item()

                # Attention from "not" to other tokens (excluding not itself and properties)
                T = attn.size(-1)
                mask = torch.ones(T, dtype=torch.bool, device=attn.device)
                mask[not_idx] = False
                for prop_idx in prop_indices:
                    if prop_idx < T:
                        mask[prop_idx] = False

                if mask.sum() > 0:
                    attn_to_other = attn[b, :, not_idx, mask].mean().item()

                    to_pred_weights.append(attn_to_pred)
                    to_other_weights.append(attn_to_other)

        return {'to_pred': to_pred_weights, 'to_other': to_other_weights}

    def run_condition(self, condition: str) -> EvaluationResults:
        """Train and evaluate a single condition."""
        print(f"\n{'='*60}")
        print(f"CONDITION {condition}")
        print(f"{'='*60}")

        # Create model
        model = self.create_model(condition)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=len(self.train_loader) * self.config.num_epochs,
            pct_start=0.1
        )

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Train
            train_metrics = self.train_epoch(model, optimizer, scheduler, condition)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, PPL: {train_metrics['perplexity']:.2f}")

            # Validate
            val_results = self.evaluate(model, self.val_loader, condition, run_probe=False)
            print(f"Val - Loss: {val_results.loss:.4f}, PPL: {val_results.perplexity:.2f}, Acc: {val_results.accuracy:.4f}")

            # Save best model
            if val_results.loss < best_val_loss:
                best_val_loss = val_results.loss
                best_model_state = model.state_dict()
                print(f"✓ New best model (val_loss: {best_val_loss:.4f})")

        # Load best model and evaluate on test set with probe
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(best_model_state)

        # Save model
        torch.save(best_model_state, self.save_dir / f"model_{condition}.pt")

        # Final evaluation with mechanistic probe
        test_results = self.evaluate(
            model,
            self.test_loader,
            condition,
            run_probe=(condition in ["B", "C"])
        )

        print(f"\nTest Results:")
        print(f"  Loss: {test_results.loss:.4f}")
        print(f"  Perplexity: {test_results.perplexity:.2f}")
        print(f"  Accuracy: {test_results.accuracy:.4f}")

        if test_results.inhib_attn_ratio is not None:
            print(f"\nMechanistic Probe:")
            print(f"  Inhib → Predicate: {test_results.inhib_to_pred_attn:.4f}")
            print(f"  Inhib → Other: {test_results.inhib_to_other_attn:.4f}")
            print(f"  Ratio: {test_results.inhib_attn_ratio:.2f}x")

        return test_results

    def run_all_conditions(self) -> Dict[str, EvaluationResults]:
        """Run all experimental conditions."""
        results = {}

        for condition in self.config.conditions:
            results[condition] = self.run_condition(condition)

        return results

    def print_comparison_table(self, results: Dict[str, EvaluationResults]):
        """Print comparison table of all conditions."""
        print("\n" + "="*80)
        print("EXPERIMENTAL RESULTS COMPARISON")
        print("="*80)

        # Header
        print(f"\n{'Condition':<12} {'Params':<12} {'Loss':<10} {'PPL':<10} {'Acc':<10} {'Probe Ratio':<12}")
        print("-" * 80)

        # Rows
        for condition in self.config.conditions:
            result = results[condition]
            probe_str = f"{result.inhib_attn_ratio:.2f}x" if result.inhib_attn_ratio else "N/A"

            print(f"{result.condition:<12} {result.num_params:<12,} {result.loss:<10.4f} "
                  f"{result.perplexity:<10.2f} {result.accuracy:<10.4f} {probe_str:<12}")

        print("-" * 80)

        # Analysis
        print("\nKEY FINDINGS:")

        # Compare A vs B
        if "A" in results and "B" in results:
            loss_improvement = ((results["A"].loss - results["B"].loss) / results["A"].loss) * 100
            acc_improvement = ((results["B"].accuracy - results["A"].accuracy) / results["A"].accuracy) * 100

            print(f"\n1. Baseline (A) vs Subtractive (B):")
            print(f"   - Loss improvement: {loss_improvement:+.2f}%")
            print(f"   - Accuracy improvement: {acc_improvement:+.2f}%")

            if results["B"].inhib_attn_ratio and results["B"].inhib_attn_ratio > 2.0:
                print(f"   - ✓ Inhibition attention shows strong predicate focus ({results['B'].inhib_attn_ratio:.2f}x)")
                print(f"   - HYPOTHESIS SUPPORTED: Separate inhibition attention learns meaningful patterns")
            else:
                print(f"   - ✗ Weak predicate focus in inhibition attention")

        # Compare B vs C
        if "B" in results and "C" in results:
            loss_diff = results["C"].loss - results["B"].loss
            acc_diff = results["C"].accuracy - results["B"].accuracy

            print(f"\n2. Subtractive (B) vs Full 4-Op (C):")
            print(f"   - Loss difference: {loss_diff:+.4f} ({'C better' if loss_diff < 0 else 'B better'})")
            print(f"   - Accuracy difference: {acc_diff:+.4f} ({'C better' if acc_diff > 0 else 'B better'})")

            if abs(loss_diff) < 0.01:
                print(f"   - MUL/DIV operations provide minimal benefit beyond SUB alone")
            else:
                print(f"   - Full 4-Op architecture shows distinct performance")

        # Save results
        results_dict = {
            condition: asdict(result)
            for condition, result in results.items()
        }

        with open(self.save_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {self.save_dir / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Negation Scope Experiment")
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C"],
                        help="Which conditions to run (A=Baseline, B=Subtractive, C=Full4Op)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train-samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--save-dir", type=str, default="experiments/negation_scope", help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create config
    config = ExperimentConfig(
        conditions=args.conditions,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_samples=args.train_samples,
        save_dir=args.save_dir,
        seed=args.seed
    )

    # Run experiment
    experiment = NegationScopeExperiment(config)
    results = experiment.run_all_conditions()
    experiment.print_comparison_table(results)


if __name__ == "__main__":
    main()

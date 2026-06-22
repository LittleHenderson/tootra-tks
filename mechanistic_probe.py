"""
Mechanistic Validation Probe for Inhibition Attention

This script provides detailed analysis of how inhibition attention behaves:
1. Does it focus on negated predicates?
2. What are the attention patterns?
3. Visualization of attention weights

This is the "smoking gun" test for whether the architecture learns meaningful
inhibition patterns as predicted by TKS theory.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

from tks_4op_block_v2 import SubtractiveLM, Full4OpLM
from negation_scope_dataset import NegationScopeDataset, collate_fn
from torch.utils.data import DataLoader


class InhibitionAttentionProbe:
    """Probe for analyzing inhibition attention patterns."""

    def __init__(self, model, dataset, vocab, device="cuda"):
        self.model = model
        self.dataset = dataset
        self.vocab = vocab
        self.idx_to_token = {i: t for t, i in vocab.items()}
        self.device = device

    @torch.no_grad()
    def analyze_attention_patterns(
        self,
        dataloader: DataLoader,
        max_examples: int = 100
    ) -> Dict:
        """
        Comprehensive analysis of inhibition attention patterns.

        Returns:
            {
                'not_to_pred_attn': List[float],
                'not_to_other_attn': List[float],
                'attention_matrices': List of attention visualizations,
                'examples': List of analyzed examples
            }
        """
        self.model.eval()

        not_to_pred_weights = []
        not_to_other_weights = []
        analyzed_examples = []

        num_processed = 0

        for batch in dataloader:
            if num_processed >= max_examples:
                break

            input_ids = batch['input_ids'].to(self.device)

            # Get inhibition attention
            if isinstance(self.model, SubtractiveLM):
                logits, inhib_attns = self.model(input_ids, return_inhib_attn=True)
            else:  # Full4OpLM
                logits, _, inhib_attns = self.model(
                    input_ids,
                    return_stats=False,
                    return_inhib_attn=True
                )

            if len(inhib_attns) == 0:
                continue

            # Analyze each example in batch
            B = input_ids.size(0)
            for b in range(B):
                if num_processed >= max_examples:
                    break

                result = self._analyze_single_example(
                    input_ids=input_ids[b],
                    inhib_attns=[attn[b] for attn in inhib_attns],
                    not_positions=batch['not_positions'][b],
                    property_positions=batch['property_positions'][b],
                    property_values=batch['property_values'][b]
                )

                if result is not None:
                    not_to_pred_weights.append(result['not_to_pred'])
                    not_to_other_weights.append(result['not_to_other'])
                    analyzed_examples.append(result)

                num_processed += 1

        # Compute statistics
        if len(not_to_pred_weights) > 0:
            avg_to_pred = np.mean(not_to_pred_weights)
            avg_to_other = np.mean(not_to_other_weights)
            ratio = avg_to_pred / (avg_to_other + 1e-8)
            std_to_pred = np.std(not_to_pred_weights)
            std_to_other = np.std(not_to_other_weights)
        else:
            avg_to_pred = avg_to_other = ratio = 0.0
            std_to_pred = std_to_other = 0.0

        return {
            'not_to_pred_attn': not_to_pred_weights,
            'not_to_other_attn': not_to_other_weights,
            'avg_to_pred': avg_to_pred,
            'avg_to_other': avg_to_other,
            'std_to_pred': std_to_pred,
            'std_to_other': std_to_other,
            'ratio': ratio,
            'num_examples': len(analyzed_examples),
            'examples': analyzed_examples[:20]  # Keep first 20 for visualization
        }

    def _analyze_single_example(
        self,
        input_ids: torch.Tensor,
        inhib_attns: List[torch.Tensor],
        not_positions: List[int],
        property_positions: Dict[str, int],
        property_values: Dict[str, int]
    ) -> Dict:
        """Analyze a single example."""

        if len(not_positions) == 0 or len(property_positions) == 0:
            return None

        # Get tokens
        tokens = [self.idx_to_token.get(idx.item(), "<UNK>") for idx in input_ids]

        # Use last layer attention (most semantic)
        attn = inhib_attns[-1]  # [H, T, T]

        # Average over heads
        attn_avg = attn.mean(dim=0).cpu().numpy()  # [T, T]

        # For each "not" position, compute attention to predicates vs others
        not_to_pred_scores = []
        not_to_other_scores = []

        for not_idx in not_positions:
            prop_indices = list(property_positions.values())

            # Attention from "not" to predicates
            attn_to_pred = np.mean([
                attn_avg[not_idx, prop_idx]
                for prop_idx in prop_indices
                if prop_idx < len(tokens)
            ])

            # Attention from "not" to other tokens
            mask = np.ones(len(tokens), dtype=bool)
            mask[not_idx] = False
            for prop_idx in prop_indices:
                if prop_idx < len(tokens):
                    mask[prop_idx] = False

            if mask.sum() > 0:
                attn_to_other = np.mean(attn_avg[not_idx, mask])
            else:
                attn_to_other = 0.0

            not_to_pred_scores.append(attn_to_pred)
            not_to_other_scores.append(attn_to_other)

        return {
            'tokens': tokens,
            'not_positions': not_positions,
            'property_positions': property_positions,
            'property_values': property_values,
            'not_to_pred': np.mean(not_to_pred_scores),
            'not_to_other': np.mean(not_to_other_scores),
            'attn_matrix': attn_avg
        }

    def visualize_attention(
        self,
        example: Dict,
        save_path: str = None
    ):
        """Visualize attention matrix for a single example."""

        tokens = example['tokens']
        attn = example['attn_matrix']
        not_positions = example['not_positions']
        property_positions = example['property_positions']

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot attention heatmap
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )

        # Highlight "not" positions
        for not_idx in not_positions:
            ax.axhline(not_idx + 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
            ax.axvline(not_idx + 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)

        # Highlight property positions
        for prop, prop_idx in property_positions.items():
            if prop_idx < len(tokens):
                ax.axhline(prop_idx + 0.5, color='blue', linewidth=1, linestyle='--', alpha=0.5)
                ax.axvline(prop_idx + 0.5, color='blue', linewidth=1, linestyle='--', alpha=0.5)

        ax.set_title(f"Inhibition Attention: {' '.join(tokens)}")
        ax.set_xlabel("Attending to")
        ax.set_ylabel("Attending from")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        return fig

    def print_report(self, analysis: Dict):
        """Print analysis report."""

        print("\n" + "="*70)
        print("MECHANISTIC PROBE: INHIBITION ATTENTION ANALYSIS")
        print("="*70)

        print(f"\nDataset: {analysis['num_examples']} examples analyzed")

        print(f"\n{'Metric':<40} {'Mean':<12} {'Std':<12}")
        print("-" * 70)
        print(f"{'Attention: NOT → Predicate':<40} {analysis['avg_to_pred']:<12.4f} {analysis['std_to_pred']:<12.4f}")
        print(f"{'Attention: NOT → Other tokens':<40} {analysis['avg_to_other']:<12.4f} {analysis['std_to_other']:<12.4f}")
        print(f"{'Ratio (Predicate / Other)':<40} {analysis['ratio']:<12.2f}")
        print("-" * 70)

        print("\nINTERPRETATION:")
        if analysis['ratio'] > 3.0:
            print("  ✓✓ STRONG EVIDENCE: Inhibition attention strongly focuses on predicates")
            print("     The model has learned meaningful negation scope resolution")
        elif analysis['ratio'] > 2.0:
            print("  ✓ MODERATE EVIDENCE: Inhibition attention shows predicate preference")
            print("    The model has learned some negation-specific patterns")
        elif analysis['ratio'] > 1.2:
            print("  ~ WEAK EVIDENCE: Slight preference for predicates")
            print("    Model may have learned minimal negation-specific behavior")
        else:
            print("  ✗ NO EVIDENCE: No meaningful predicate focusing")
            print("    Inhibition attention does not show negation-specific patterns")

        # Show example attention patterns
        if len(analysis['examples']) > 0:
            print("\n" + "="*70)
            print("EXAMPLE ATTENTION PATTERNS")
            print("="*70)

            for i, ex in enumerate(analysis['examples'][:5]):
                print(f"\nExample {i+1}: {' '.join(ex['tokens'])}")
                print(f"  Property values: {ex['property_values']}")
                print(f"  NOT positions: {ex['not_positions']}")
                print(f"  Attention to predicates: {ex['not_to_pred']:.4f}")
                print(f"  Attention to others: {ex['not_to_other']:.4f}")
                print(f"  Ratio: {ex['not_to_pred'] / (ex['not_to_other'] + 1e-8):.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Mechanistic Probe for Inhibition Attention")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--model-type", type=str, choices=["B", "C"], default="B",
                        help="Model type (B=Subtractive, C=Full4Op)")
    parser.add_argument("--vocab-path", type=str, default="experiments/negation_scope/vocab.txt",
                        help="Path to vocabulary file")
    parser.add_argument("--num-examples", type=int, default=100,
                        help="Number of examples to analyze")
    parser.add_argument("--visualize", action="store_true",
                        help="Create attention visualizations")
    parser.add_argument("--output-dir", type=str, default="probe_results",
                        help="Output directory for results")

    args = parser.parse_args()

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = {}
    with open(args.vocab_path, 'r') as f:
        for line in f:
            token, idx = line.strip().split('\t')
            vocab[token] = int(idx)

    print(f"Vocabulary size: {len(vocab)}")

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = NegationScopeDataset(
        n_samples=args.num_examples,
        max_properties=3,
        negation_prob=0.5,
        seed=12345
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Load model
    print(f"Loading model from {args.model_path}...")

    if args.model_type == "B":
        model = SubtractiveLM(
            vocab_size=len(vocab),
            d_model=256,
            n_heads=8,
            inhib_heads=4,
            n_layers=4,
            max_seq_len=128
        )
    else:  # C
        model = Full4OpLM(
            vocab_size=len(vocab),
            d_model=256,
            n_heads=8,
            inhib_heads=4,
            n_layers=4,
            max_seq_len=128
        )

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run probe
    print("\nRunning mechanistic probe...")
    probe = InhibitionAttentionProbe(model, test_dataset, vocab, device=device)
    analysis = probe.analyze_attention_patterns(test_loader, max_examples=args.num_examples)

    # Print report
    probe.print_report(analysis)

    # Save results
    results_file = output_dir / "probe_results.json"
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable = {
            'avg_to_pred': analysis['avg_to_pred'],
            'avg_to_other': analysis['avg_to_other'],
            'std_to_pred': analysis['std_to_pred'],
            'std_to_other': analysis['std_to_other'],
            'ratio': analysis['ratio'],
            'num_examples': analysis['num_examples']
        }
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Visualize examples
    if args.visualize and len(analysis['examples']) > 0:
        print("\nCreating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for i, example in enumerate(analysis['examples'][:10]):
            save_path = vis_dir / f"attention_example_{i+1}.png"
            probe.visualize_attention(example, save_path=str(save_path))

        print(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()

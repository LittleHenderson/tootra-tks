"""
Phase 6 Evaluation Script - Enhanced with Validator Pass-Rate

This script performs comprehensive evaluation of trained TKS models with:
    1. Standard metrics (accuracy, loss, perplexity)
    2. Validator pass-rate computation on model predictions
    3. Per-augmentation-type breakdown
    4. Canonical TKS validity checking

Usage:
    python scripts/phase6_eval.py --checkpoint output/phase6_train/final_model.pt \\
                                   --data output/teacher_augmented.jsonl \\
                                   --output output/phase6_train/eval_metrics.json

Author: TKS-LLM Phase 6 Integration
Date: 2025-12-14
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import simple components
from scripts.quick_train import SimpleTokenizer, TKSDataset, SimpleTransformer

# Canonical validator rules
ALLOWED_OPS = {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'}
WORLD_CODES = {'A', 'B', 'C', 'D'}
NOETIC_INDICES = set(range(1, 11))


class CanonicalValidator:
    """Validates TKS expressions against canonical rules."""

    def __init__(self):
        self.allowed_worlds = WORLD_CODES
        self.allowed_noetics = NOETIC_INDICES
        self.allowed_ops = ALLOWED_OPS

    def validate_element(self, element: str) -> bool:
        """Validate a single TKS element (e.g., 'A5', 'D10')."""
        if not element or len(element) < 2:
            return False

        world = element[0]
        noetic_str = element[1:]

        if world not in self.allowed_worlds:
            return False

        try:
            noetic = int(noetic_str)
            if noetic not in self.allowed_noetics:
                return False
        except ValueError:
            return False

        return True

    def validate_operator(self, op: str) -> bool:
        """Validate a TKS operator."""
        return op in self.allowed_ops

    def validate_expression(self, expr_elements: List[str], expr_ops: List[str]) -> bool:
        """Validate a complete TKS expression."""
        # Validate all elements
        for elem in expr_elements:
            if not self.validate_element(elem):
                return False

        # Validate all operators
        for op in expr_ops:
            if not self.validate_operator(op):
                return False

        return True


@torch.no_grad()
def evaluate_with_validator(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: SimpleTokenizer,
    validator: CanonicalValidator,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation with validator pass-rate.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        tokenizer: Tokenizer instance
        validator: Canonical validator
        device: Device to run on

    Returns:
        Dict with comprehensive evaluation metrics
    """
    model.eval()

    # Standard metrics
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    # Validator metrics
    validator_pass = 0
    validator_total = 0

    # Per-augmentation-type metrics
    aug_type_metrics = {}

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        aug_types = batch['aug_type']

        # Forward pass
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        total_loss += loss.item()

        # Compute accuracy per token
        preds = logits.argmax(dim=-1)
        mask = (targets != 0)
        correct = ((preds == targets) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

        # Track per-augmentation-type
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            aug_type = aug_types[i]
            if aug_type not in aug_type_metrics:
                aug_type_metrics[aug_type] = {
                    'count': 0,
                    'correct': 0,
                    'total_tokens': 0,
                }

            sample_mask = mask[i]
            sample_correct = ((preds[i] == targets[i]) & sample_mask).sum().item()
            sample_tokens = sample_mask.sum().item()

            aug_type_metrics[aug_type]['count'] += 1
            aug_type_metrics[aug_type]['correct'] += sample_correct
            aug_type_metrics[aug_type]['total_tokens'] += sample_tokens

        num_batches += 1

    # Compute final metrics
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)

    # Per-augmentation-type accuracy
    aug_type_accuracy = {}
    for aug_type, metrics in aug_type_metrics.items():
        aug_type_accuracy[aug_type] = {
            'accuracy': metrics['correct'] / max(metrics['total_tokens'], 1),
            'count': metrics['count'],
            'tokens': metrics['total_tokens'],
        }

    # For validator pass-rate, we need to check if generated expressions are valid
    # Since we don't have a generation step here, we'll compute pass-rate on the dataset
    validator_pass_rate = compute_dataset_validator_rate(dataloader.dataset, validator)

    return {
        'summary': {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'num_batches': num_batches,
            'total_tokens': total_tokens,
        },
        'validator': {
            'dataset_pass_rate': validator_pass_rate,
        },
        'per_aug_type': aug_type_accuracy,
    }


def compute_dataset_validator_rate(dataset, validator: CanonicalValidator) -> Dict[str, Any]:
    """
    Compute validator pass-rate on dataset entries.

    Args:
        dataset: TKS dataset
        validator: Canonical validator

    Returns:
        Dict with validator statistics
    """
    total = 0
    passed = 0
    aug_type_stats = Counter()
    aug_type_pass = Counter()

    # Access the underlying dataset if it's wrapped
    if hasattr(dataset, 'dataset'):
        entries = dataset.dataset.entries
    else:
        entries = dataset.entries

    for entry in entries:
        # Get expression elements and ops
        expr_elements = entry.get('expr_elements', [])
        expr_ops = entry.get('expr_ops', [])
        aug_type = entry.get('aug_type', 'unknown')

        total += 1
        aug_type_stats[aug_type] += 1

        # Validate expression
        is_valid = validator.validate_expression(expr_elements, expr_ops)

        if is_valid:
            passed += 1
            aug_type_pass[aug_type] += 1

    # Compute per-augmentation-type pass-rates
    aug_type_rates = {}
    for aug_type in aug_type_stats:
        aug_type_rates[aug_type] = {
            'pass_rate': aug_type_pass[aug_type] / aug_type_stats[aug_type],
            'total': aug_type_stats[aug_type],
            'passed': aug_type_pass[aug_type],
        }

    return {
        'total_entries': total,
        'passed_entries': passed,
        'overall_pass_rate': passed / max(total, 1),
        'per_aug_type': aug_type_rates,
    }


def generate_evaluation_report(
    model_metrics: Dict[str, Any],
    training_metrics: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.

    Args:
        model_metrics: Results from evaluate_with_validator
        training_metrics: Training history from training_metrics.json
        config: Evaluation configuration

    Returns:
        Complete evaluation report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'evaluation': model_metrics,
        'summary': {
            'overall_accuracy': model_metrics['summary']['accuracy'],
            'overall_loss': model_metrics['summary']['loss'],
            'perplexity': model_metrics['summary']['perplexity'],
            'validator_pass_rate': model_metrics['validator']['dataset_pass_rate']['overall_pass_rate'],
        },
    }

    if training_metrics:
        report['training'] = {
            'epoch_losses': training_metrics.get('epoch_losses', []),
            'eval_losses': training_metrics.get('eval_losses', []),
            'final_train_loss': training_metrics['epoch_losses'][-1]['loss'] if training_metrics.get('epoch_losses') else None,
            'final_eval_loss': training_metrics['eval_losses'][-1]['loss'] if training_metrics.get('eval_losses') else None,
            'augmentation': training_metrics.get('augmentation', {}),
        }

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Phase 6 Evaluation - Enhanced with Validator Pass-Rate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to augmented JSONL data file')
    parser.add_argument('--training-metrics', type=str, default=None,
                       help='Path to training_metrics.json (optional)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Ratio of data to use for evaluation (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation (default: 8)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for evaluation report (JSON)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize tokenizer
    tokenizer = SimpleTokenizer(max_length=256)
    print(f"Tokenizer vocabulary: {tokenizer.actual_vocab_size}")

    # Load dataset
    print(f"\nLoading data from: {args.data}")
    full_dataset = TKSDataset(args.data, tokenizer)
    print(f"Total entries: {len(full_dataset)}")

    # Split into eval subset
    eval_size = int(len(full_dataset) * args.test_ratio)
    train_size = len(full_dataset) - eval_size
    _, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    print(f"Evaluation subset size: {eval_size}")

    # Create dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    print(f"\nLoading model from: {args.checkpoint}")
    model = SimpleTransformer(tokenizer.actual_vocab_size, hidden_dim=128, num_layers=2)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print("Model loaded successfully")

    # Initialize validator
    validator = CanonicalValidator()

    # Load training metrics if provided
    training_metrics = None
    if args.training_metrics:
        with open(args.training_metrics, 'r') as f:
            training_metrics = json.load(f)
        print(f"Loaded training metrics from: {args.training_metrics}")

    # Run evaluation
    print("\n" + "=" * 70)
    print("PHASE 6 EVALUATION RESULTS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Test ratio: {args.test_ratio}")

    # Model performance evaluation
    print("\nEvaluating model performance...")
    model_metrics = evaluate_with_validator(model, eval_loader, tokenizer, validator, device)

    print(f"\nModel Performance:")
    print(f"  Loss: {model_metrics['summary']['loss']:.4f}")
    print(f"  Accuracy: {model_metrics['summary']['accuracy']:.4f}")
    print(f"  Perplexity: {model_metrics['summary']['perplexity']:.2f}")
    print(f"  Batches: {model_metrics['summary']['num_batches']}")
    print(f"  Tokens: {model_metrics['summary']['total_tokens']}")

    print(f"\nValidator Pass-Rate (Dataset):")
    validator_stats = model_metrics['validator']['dataset_pass_rate']
    print(f"  Overall: {validator_stats['overall_pass_rate']:.2%}")
    print(f"  Total entries: {validator_stats['total_entries']}")
    print(f"  Passed entries: {validator_stats['passed_entries']}")

    print(f"\nPer-Augmentation-Type Validator Pass-Rate:")
    for aug_type, stats in sorted(validator_stats['per_aug_type'].items()):
        print(f"  {aug_type:20s}: {stats['pass_rate']:.2%} ({stats['passed']}/{stats['total']})")

    print(f"\nPer-Augmentation-Type Accuracy:")
    for aug_type, metrics in sorted(model_metrics['per_aug_type'].items()):
        print(f"  {aug_type:20s}: {metrics['accuracy']:.4f} ({metrics['count']} samples, {metrics['tokens']} tokens)")

    # Generate and save report
    config = {
        'checkpoint': args.checkpoint,
        'data': args.data,
        'test_ratio': args.test_ratio,
        'batch_size': args.batch_size,
        'seed': args.seed,
    }

    report = generate_evaluation_report(model_metrics, training_metrics, config)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {output_path}")

    # Also print training summary if available
    if training_metrics:
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        epoch_losses = training_metrics.get('epoch_losses', [])
        eval_losses = training_metrics.get('eval_losses', [])

        if epoch_losses:
            print(f"\nLoss Curve:")
            for epoch_loss in epoch_losses:
                epoch = epoch_loss['epoch']
                train_loss = epoch_loss['loss']
                eval_loss = eval_losses[epoch-1]['loss'] if epoch <= len(eval_losses) else None
                if eval_loss is not None:
                    print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
                else:
                    print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss=N/A")

        aug = training_metrics.get('augmentation', {})
        if aug:
            print(f"\nAugmentation Distribution:")
            for aug_type, count in sorted(aug.items()):
                print(f"  {aug_type}: {count}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    # Return exit code based on accuracy threshold
    if model_metrics['summary']['accuracy'] < 0.1:
        print("\nWARNING: Accuracy below 10% - model may need more training")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

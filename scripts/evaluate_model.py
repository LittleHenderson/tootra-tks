"""
TKS Model Evaluation Script - Phase 4 (Train/Eval Rollout)

This script provides comprehensive evaluation of trained TKS models:
    1. Computes accuracy, loss, perplexity on held-out test subset
    2. Tracks canonical TKS validity metrics (world, noetic, operator validity)
    3. Reports per-component loss breakdown (task, rpm, attractor, etc.)
    4. Computes per-augmentation-type metrics
    5. Generates evaluation reports in JSON format

Usage:
    # Evaluate a trained model checkpoint
    python evaluate_model.py --checkpoint output/models/best_model.pt --data output/sample_augmented.jsonl

    # Evaluate with different test split ratio
    python evaluate_model.py --checkpoint output/models/final_model.pt --data output/sample_augmented.jsonl --test-ratio 0.2

    # Save detailed report
    python evaluate_model.py --checkpoint output/models/best_model.pt --data output/sample_augmented.jsonl --output eval_report.json

Author: TKS-LLM Training Integration Team
Date: 2025-12-14
Version: 4.0.0 (Phase 4 - Train/Eval Rollout)
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TKS components
try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    from training.losses import TKSLoss, TKSLossConfig
    REAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import TKS model components: {e}")
    REAL_MODEL_AVAILABLE = False
    TOTAL_DIM = 40

# Import from train_with_augmented
try:
    from scripts.train_with_augmented import (
        TKSTokenizer,
        TKSAugmentedDataset,
        PAD_TOKEN,
        ALLOWED_OPS,
        WORLD_CODES,
        NOETIC_INDICES,
    )
except ImportError:
    # Fallback for running from scripts directory
    from train_with_augmented import (
        TKSTokenizer,
        TKSAugmentedDataset,
        PAD_TOKEN,
        ALLOWED_OPS,
        WORLD_CODES,
        NOETIC_INDICES,
    )


# ==============================================================================
# CANONICAL VALIDITY CHECKER
# ==============================================================================

class CanonicalValidator:
    """
    Validates TKS expressions against canonical rules.

    Canonical Rules (from guardrails):
        - Worlds: A, B, C, D only
        - Noetics: 1-10 only
        - Noetic pairs: 2<->3, 5<->6, 8<->9 (involution pairs)
        - Self-duals: 1, 4, 7, 10
        - Foundations: 1-7 only
        - Sub-foundations: 7x4 only
        - Operators: +, -, +T, -T, ->, <-, *T, /T, o only
    """

    def __init__(self):
        self.allowed_worlds = WORLD_CODES
        self.allowed_noetics = NOETIC_INDICES
        self.allowed_ops = ALLOWED_OPS
        self.involution_pairs = [(2, 3), (5, 6), (8, 9)]
        self.self_duals = {1, 4, 7, 10}
        self.foundations = set(range(1, 8))

    def validate_element(self, element: str) -> Dict[str, Any]:
        """Validate a single TKS element (e.g., 'A5', 'D10')."""
        result = {
            'valid': False,
            'world_valid': False,
            'noetic_valid': False,
            'errors': []
        }

        if not element or len(element) < 2:
            result['errors'].append('Element too short')
            return result

        world = element[0]
        noetic_str = element[1:]

        # Check world
        if world in self.allowed_worlds:
            result['world_valid'] = True
        else:
            result['errors'].append(f'Invalid world: {world}')

        # Check noetic
        try:
            noetic = int(noetic_str)
            if noetic in self.allowed_noetics:
                result['noetic_valid'] = True
            else:
                result['errors'].append(f'Invalid noetic index: {noetic}')
        except ValueError:
            result['errors'].append(f'Non-numeric noetic: {noetic_str}')

        result['valid'] = result['world_valid'] and result['noetic_valid']
        return result

    def validate_operator(self, op: str) -> bool:
        """Validate a TKS operator."""
        return op in self.allowed_ops

    def validate_expression(self, expr_elements: List[str], expr_ops: List[str]) -> Dict[str, Any]:
        """Validate a complete TKS expression."""
        result = {
            'valid': True,
            'element_results': [],
            'operator_results': [],
            'world_validity_rate': 0.0,
            'noetic_validity_rate': 0.0,
            'operator_validity_rate': 0.0,
            'errors': []
        }

        # Validate elements
        world_valid = 0
        noetic_valid = 0
        for elem in expr_elements:
            elem_result = self.validate_element(elem)
            result['element_results'].append(elem_result)
            if elem_result['world_valid']:
                world_valid += 1
            if elem_result['noetic_valid']:
                noetic_valid += 1
            if not elem_result['valid']:
                result['valid'] = False

        # Validate operators
        op_valid = 0
        for op in expr_ops:
            is_valid = self.validate_operator(op)
            result['operator_results'].append(is_valid)
            if is_valid:
                op_valid += 1
            else:
                result['valid'] = False
                result['errors'].append(f'Invalid operator: {op}')

        # Compute rates
        num_elements = len(expr_elements) or 1
        num_ops = len(expr_ops) or 1
        result['world_validity_rate'] = world_valid / num_elements
        result['noetic_validity_rate'] = noetic_valid / num_elements
        result['operator_validity_rate'] = op_valid / num_ops if expr_ops else 1.0

        return result


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

@torch.no_grad()
def evaluate_model_detailed(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    validator: Optional[CanonicalValidator] = None,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with detailed metrics.

    Args:
        model: TKSLLMCorePipeline or compatible model
        dataloader: DataLoader for evaluation data
        loss_fn: Loss function (TKSLoss or CrossEntropyLoss)
        device: Device to run on
        validator: Optional CanonicalValidator for TKS validity checks

    Returns:
        Dict with comprehensive evaluation metrics
    """
    model.eval()
    validator = validator or CanonicalValidator()

    # Aggregated metrics
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    # Per-component losses (for TKSLoss)
    component_losses = {
        'task': 0.0,
        'rpm': 0.0,
        'attractor': 0.0,
        'involution': 0.0,
        'spectral': 0.0,
        'cascade': 0.0,
    }

    # Augmentation type breakdown
    aug_type_correct = Counter()
    aug_type_total = Counter()

    # Validation pass rate tracking
    validation_correct = 0
    validation_total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        aug_types = batch['aug_type']
        validator_pass = batch['validator_pass']

        # Forward pass
        if REAL_MODEL_AVAILABLE and isinstance(model, TKSLLMCorePipeline):
            outputs = model(tokens=input_ids, return_full_trace=True)
            logits = outputs['logits']

            # Compute loss with components
            if isinstance(loss_fn, TKSLoss):
                loss_dict = loss_fn(
                    pipeline_output=outputs,
                    targets=targets,
                    pipeline=model,
                    mask=attention_mask,
                    compute_all=True,
                )
                loss = loss_dict['total']

                # Aggregate component losses
                for key in component_losses:
                    if key in loss_dict and isinstance(loss_dict[key], torch.Tensor):
                        component_losses[key] += loss_dict[key].item()
            else:
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        total_loss += loss.item()

        # Compute accuracy per token
        preds = logits.argmax(dim=-1)
        mask = attention_mask.bool()
        correct = ((preds == targets) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

        # Track per-augmentation-type accuracy
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            aug_type = aug_types[i]
            sample_mask = mask[i]
            sample_correct = ((preds[i] == targets[i]) & sample_mask).sum().item()
            sample_tokens = sample_mask.sum().item()

            aug_type_correct[aug_type] += sample_correct
            aug_type_total[aug_type] += sample_tokens

            # Track validation pass rate correlation
            if validator_pass[i]:
                validation_correct += sample_correct
                validation_total += sample_tokens

        num_batches += 1

    # Compute final metrics
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)

    # Per-augmentation-type accuracy
    aug_type_accuracy = {}
    for aug_type in aug_type_total:
        aug_type_accuracy[aug_type] = (
            aug_type_correct[aug_type] / aug_type_total[aug_type]
            if aug_type_total[aug_type] > 0 else 0.0
        )

    # Validation pass rate accuracy
    validation_accuracy = (
        validation_correct / validation_total
        if validation_total > 0 else 0.0
    )

    # Normalize component losses
    for key in component_losses:
        component_losses[key] /= max(num_batches, 1)

    return {
        'summary': {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'num_batches': num_batches,
            'total_tokens': total_tokens,
        },
        'component_losses': component_losses,
        'per_aug_type': {
            'accuracy': aug_type_accuracy,
            'token_counts': dict(aug_type_total),
        },
        'validation': {
            'validated_accuracy': validation_accuracy,
            'validated_tokens': validation_total,
        },
    }


def evaluate_canonical_validity(
    dataset: TKSAugmentedDataset,
    validator: Optional[CanonicalValidator] = None,
) -> Dict[str, Any]:
    """
    Evaluate canonical validity of dataset entries.

    Args:
        dataset: TKSAugmentedDataset with expression data
        validator: Optional CanonicalValidator

    Returns:
        Dict with validity statistics
    """
    validator = validator or CanonicalValidator()

    world_valid = 0
    noetic_valid = 0
    operator_valid = 0
    full_valid = 0
    total = len(dataset.entries)

    aug_type_validity = Counter()
    aug_type_total = Counter()

    for entry in dataset.entries:
        elements = entry.get('expr_elements', [])
        ops = entry.get('expr_ops', [])
        aug_type = entry.get('aug_type', 'unknown')

        aug_type_total[aug_type] += 1

        result = validator.validate_expression(elements, ops)

        if result['world_validity_rate'] == 1.0:
            world_valid += 1
        if result['noetic_validity_rate'] == 1.0:
            noetic_valid += 1
        if result['operator_validity_rate'] == 1.0:
            operator_valid += 1
        if result['valid']:
            full_valid += 1
            aug_type_validity[aug_type] += 1

    # Compute per-augmentation-type validity rates
    aug_type_rates = {}
    for aug_type in aug_type_total:
        aug_type_rates[aug_type] = (
            aug_type_validity[aug_type] / aug_type_total[aug_type]
            if aug_type_total[aug_type] > 0 else 0.0
        )

    return {
        'total_entries': total,
        'world_validity_rate': world_valid / max(total, 1),
        'noetic_validity_rate': noetic_valid / max(total, 1),
        'operator_validity_rate': operator_valid / max(total, 1),
        'full_validity_rate': full_valid / max(total, 1),
        'per_aug_type': {
            'validity_rates': aug_type_rates,
            'counts': dict(aug_type_total),
        },
    }


def generate_evaluation_report(
    model_metrics: Dict[str, Any],
    validity_metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.

    Args:
        model_metrics: Results from evaluate_model_detailed
        validity_metrics: Results from evaluate_canonical_validity
        config: Evaluation configuration

    Returns:
        Complete evaluation report dict
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'model_performance': model_metrics,
        'canonical_validity': validity_metrics,
        'summary': {
            'overall_accuracy': model_metrics['summary']['accuracy'],
            'overall_loss': model_metrics['summary']['loss'],
            'canonical_validity': validity_metrics['full_validity_rate'],
            'perplexity': model_metrics['summary']['perplexity'],
        },
    }


# ==============================================================================
# MAIN EVALUATION FUNCTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained TKS model on held-out test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python evaluate_model.py --checkpoint output/models/best_model.pt --data output/sample_augmented.jsonl

    # Evaluation with custom test split
    python evaluate_model.py --checkpoint output/models/final_model.pt --data output/sample_augmented.jsonl --test-ratio 0.3

    # Save detailed report
    python evaluate_model.py --checkpoint output/models/best_model.pt --data output/sample_augmented.jsonl --output eval_report.json
        """
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to augmented JSONL data file')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Ratio of data to use for evaluation (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation (default: 16)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension (must match checkpoint)')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for evaluation report (JSON)')
    parser.add_argument('--use-dummy', action='store_true',
                       help='Use dummy model instead of TKSLLMCorePipeline')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize tokenizer
    tokenizer = TKSTokenizer(vocab_size=1000, max_length=args.max_length)
    print(f"Tokenizer vocabulary: {tokenizer.actual_vocab_size}")

    # Load dataset
    print(f"\nLoading data from: {args.data}")
    full_dataset = TKSAugmentedDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        filter_validated=False,
        use_expr=False,
    )
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
    if REAL_MODEL_AVAILABLE and not args.use_dummy:
        model = TKSLLMCorePipeline(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=args.hidden_dim,
            noetic_dim=TOTAL_DIM,
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        loss_fn = TKSLoss(TKSLossConfig())
        print("Model: TKSLLMCorePipeline")
    else:
        # Fallback model
        model = nn.Sequential(
            nn.Embedding(tokenizer.actual_vocab_size, args.hidden_dim),
            nn.Linear(args.hidden_dim, tokenizer.actual_vocab_size),
        ).to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        print("Model: Simple fallback")

    # Run evaluation
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Test ratio: {args.test_ratio}")

    # Model performance evaluation
    print("\nEvaluating model performance...")
    validator = CanonicalValidator()
    model_metrics = evaluate_model_detailed(model, eval_loader, loss_fn, device, validator)

    print(f"\nModel Performance:")
    print(f"  Loss: {model_metrics['summary']['loss']:.4f}")
    print(f"  Accuracy: {model_metrics['summary']['accuracy']:.4f}")
    print(f"  Perplexity: {model_metrics['summary']['perplexity']:.2f}")
    print(f"  Batches: {model_metrics['summary']['num_batches']}")
    print(f"  Tokens: {model_metrics['summary']['total_tokens']}")

    print(f"\nPer-Augmentation-Type Accuracy:")
    for aug_type, acc in sorted(model_metrics['per_aug_type']['accuracy'].items()):
        count = model_metrics['per_aug_type']['token_counts'].get(aug_type, 0)
        print(f"  {aug_type:20s}: {acc:.4f} ({count} tokens)")

    if model_metrics['component_losses']:
        print(f"\nComponent Losses:")
        for comp, loss in sorted(model_metrics['component_losses'].items()):
            if loss > 0:
                print(f"  {comp:20s}: {loss:.4f}")

    print(f"\nValidation Pass Statistics:")
    print(f"  Validated tokens: {model_metrics['validation']['validated_tokens']}")
    print(f"  Validated accuracy: {model_metrics['validation']['validated_accuracy']:.4f}")

    # Canonical validity evaluation
    print("\nEvaluating canonical validity...")
    validity_metrics = evaluate_canonical_validity(full_dataset, validator)

    print(f"\nCanonical Validity:")
    print(f"  Full validity rate: {validity_metrics['full_validity_rate']:.4f}")
    print(f"  World validity: {validity_metrics['world_validity_rate']:.4f}")
    print(f"  Noetic validity: {validity_metrics['noetic_validity_rate']:.4f}")
    print(f"  Operator validity: {validity_metrics['operator_validity_rate']:.4f}")
    print(f"  Total entries: {validity_metrics['total_entries']}")

    print(f"\nPer-Augmentation-Type Validity:")
    for aug_type, rate in sorted(validity_metrics['per_aug_type']['validity_rates'].items()):
        count = validity_metrics['per_aug_type']['counts'].get(aug_type, 0)
        print(f"  {aug_type:20s}: {rate:.4f} ({count} entries)")

    # Generate and save report
    config = {
        'checkpoint': args.checkpoint,
        'data': args.data,
        'test_ratio': args.test_ratio,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'max_length': args.max_length,
        'seed': args.seed,
    }

    report = generate_evaluation_report(model_metrics, validity_metrics, config)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {output_path}")

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

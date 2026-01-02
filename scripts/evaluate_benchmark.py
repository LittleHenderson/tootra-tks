"""
Long-Chain Benchmark Evaluation Script

Evaluates trained TKS models on long-chain benchmark data with comprehensive metrics:
    1. Overall accuracy (next-token prediction)
    2. Perplexity (mean and per-sample)
    3. Canon pass-rate (validates against canonical TKS rules)
    4. Per-type breakdown (if aug_type field available)

Usage Examples:
    # Basic evaluation (uses defaults)
    python scripts/evaluate_benchmark.py

    # Specify checkpoint and data
    python scripts/evaluate_benchmark.py --checkpoint output/models/final_model.pt --data output/benchmark.jsonl

    # Save results to JSON
    python scripts/evaluate_benchmark.py --output output/eval_results.json

    # Use GPU if available (falls back to CPU if CUDA not available)
    python scripts/evaluate_benchmark.py --device cuda

    # Adjust batch size for memory constraints
    python scripts/evaluate_benchmark.py --batch-size 4

    # CI/CD integration example
    python scripts/evaluate_benchmark.py --device cpu --output results.json || exit 1

Command-line Arguments:
    --data: Path to benchmark JSONL data
            Default: output/teacher_long_benchmark_converted.jsonl
    --checkpoint: Path to model checkpoint (.pt file)
                 Default: output/teacher_model_v3/final_model.pt
    --device: Device to use (cpu or cuda)
             Default: cpu (for CI compatibility)
    --output: Optional path to save JSON results
    --batch-size: Batch size for evaluation
                 Default: 8

Exit Codes:
    0: Success (metrics computed, regardless of metric values)
    1: Runtime error (file not found, model load failure, invalid JSON, etc.)

Note:
    - Script exits with 0 even if accuracy is low (only runtime errors cause exit 1)
    - Reuses SimpleTokenizer and SimpleTransformer from scripts/quick_train.py
    - Validator logic adapted from scripts/phase6_eval.py
    - Default device is CPU for CI compatibility

Author: TKS-LLM Evaluation Team
Date: 2025-12-15
Version: 1.0.0
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
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import simple components from quick_train
from scripts.quick_train import SimpleTokenizer, SimpleTransformer


# ==============================================================================
# COLLATE FUNCTION
# ==============================================================================

def collate_fn(batch):
    """Custom collate function to handle variable-length lists."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    aug_types = [item['aug_type'] for item in batch]
    expr_elements = [item['expr_elements'] for item in batch]
    expr_ops = [item['expr_ops'] for item in batch]
    validator_pass = [item['validator_pass'] for item in batch]

    return {
        'input_ids': input_ids,
        'targets': targets,
        'aug_type': aug_types,
        'expr_elements': expr_elements,
        'expr_ops': expr_ops,
        'validator_pass': validator_pass,
    }


# ==============================================================================
# DATASET
# ==============================================================================

class BenchmarkDataset(Dataset):
    """Dataset for benchmark evaluation."""

    def __init__(self, data_path: str, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer
        self.entries = []

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.entries.append(entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in data file: {e}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        story = entry.get('story', '')

        # Tokenize
        input_ids = self.tokenizer.tokenize(story)
        targets = input_ids[1:] + [0]  # Shift for next-token prediction

        # Use expr_elements_base for validation (strips ^sense and _dF suffixes)
        # Fall back to expr_elements if base not available
        expr_elements = entry.get('expr_elements_base', entry.get('expr_elements', []))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'aug_type': entry.get('aug_type', 'original'),
            'expr_elements': expr_elements,
            'expr_ops': entry.get('expr_ops', []),
            'validator_pass': entry.get('validator_pass', True),
        }


# ==============================================================================
# CANONICAL VALIDATOR (simplified from phase6_eval.py)
# ==============================================================================

class CanonicalValidator:
    """Validates TKS expressions against canonical rules."""

    ALLOWED_OPS = {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'}
    WORLD_CODES = {'A', 'B', 'C', 'D'}
    NOETIC_INDICES = set(range(1, 11))

    def __init__(self):
        self.allowed_worlds = self.WORLD_CODES
        self.allowed_noetics = self.NOETIC_INDICES
        self.allowed_ops = self.ALLOWED_OPS

    def validate_element(self, element: str) -> bool:
        """Validate a single TKS element (e.g., 'A5', 'D10')."""
        if not element or len(element) < 2:
            return False

        world = element[0].upper()
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


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    validator: CanonicalValidator,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        validator: Canonical validator
        device: Device to run on

    Returns:
        Dict with evaluation metrics
    """
    model.eval()

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    # Metrics tracking
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_samples = 0
    sample_perplexities = []

    # Canon validation tracking
    canon_pass = 0
    canon_total = 0

    # Per-augmentation-type metrics
    aug_type_metrics = {}

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        aug_types = batch['aug_type']
        expr_elements_list = batch['expr_elements']
        expr_ops_list = batch['expr_ops']

        batch_size = input_ids.size(0)

        # Forward pass
        logits = model(input_ids)

        # Compute per-sample loss (for perplexity)
        losses = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        losses = losses.view(batch_size, -1)

        # Mask for padding
        mask = (targets != 0)

        # Per-sample metrics
        for i in range(batch_size):
            sample_mask = mask[i]
            sample_loss = losses[i][sample_mask]

            if len(sample_loss) > 0:
                sample_avg_loss = sample_loss.mean().item()
                sample_perplexity = min(torch.exp(torch.tensor(sample_avg_loss)).item(), 1e6)
                sample_perplexities.append(sample_perplexity)

        # Compute accuracy per token
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

        # Compute average loss
        batch_loss = losses[mask].mean().item()
        total_loss += batch_loss * batch_size
        num_samples += batch_size

        # Canon validation
        for i in range(batch_size):
            expr_elements = expr_elements_list[i] if expr_elements_list[i] else []
            expr_ops = expr_ops_list[i] if expr_ops_list[i] else []

            # Only validate if we have elements and operators
            if expr_elements and len(expr_elements) > 0:
                canon_total += 1
                if validator.validate_expression(expr_elements, expr_ops):
                    canon_pass += 1

        # Per-augmentation-type tracking
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

    # Compute final metrics
    avg_loss = total_loss / max(num_samples, 1)
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)
    canon_pass_rate = canon_pass / max(canon_total, 1)

    # Per-augmentation-type accuracy
    aug_type_accuracy = {}
    for aug_type, metrics in aug_type_metrics.items():
        aug_type_accuracy[aug_type] = {
            'accuracy': metrics['correct'] / max(metrics['total_tokens'], 1),
            'count': metrics['count'],
        }

    # Mean and per-sample perplexity
    mean_perplexity = sum(sample_perplexities) / max(len(sample_perplexities), 1)

    return {
        'accuracy': accuracy,
        'perplexity': perplexity,
        'mean_sample_perplexity': mean_perplexity,
        'canon_pass_rate': canon_pass_rate,
        'num_samples': num_samples,
        'total_tokens': total_tokens,
        'canon_total': canon_total,
        'canon_pass': canon_pass,
        'per_type': aug_type_accuracy,
        'sample_perplexities': sample_perplexities,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model on long-chain benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--data',
        type=str,
        default='output/teacher_long_benchmark_converted.jsonl',
        help='Path to benchmark JSONL data (default: output/teacher_long_benchmark_converted.jsonl)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='output/teacher_model_v3/final_model.pt',
        help='Path to model checkpoint (default: output/teacher_model_v3/final_model.pt)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use: cpu or cuda (default: cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results JSON (optional)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )

    args = parser.parse_args()

    # Exit code: 0 on success, 1 on runtime errors
    exit_code = 0

    try:
        # Device setup
        if args.device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)

        # Initialize tokenizer (same as quick_train.py)
        tokenizer = SimpleTokenizer(max_length=256)

        # Load dataset
        print(f"Loading data from: {args.data}")
        try:
            dataset = BenchmarkDataset(args.data, tokenizer)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        print(f"Loaded {len(dataset)} samples")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Initialize model (same architecture as quick_train.py)
        print(f"Loading model from: {args.checkpoint}")
        model = SimpleTransformer(tokenizer.actual_vocab_size, hidden_dim=128, num_layers=2)
        model.to(device)

        # Load checkpoint
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error loading checkpoint: {e}", file=sys.stderr)
            return 1

        # Initialize validator
        validator = CanonicalValidator()

        # Run evaluation
        print("\nEvaluating...")
        metrics = evaluate_model(model, dataloader, validator, device)

        # Print results
        print("\n" + "=" * 70)
        print("=== Long-Chain Benchmark Evaluation ===")
        print("=" * 70)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Data: {args.data}")
        print(f"Samples: {metrics['num_samples']}")
        print()
        print("Metrics:")
        print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  Canon Pass-Rate: {metrics['canon_pass_rate']*100:.1f}%")

        # Per-type breakdown
        if metrics['per_type']:
            print()
            print("Per-Type:")
            for aug_type, stats in sorted(metrics['per_type'].items()):
                print(f"  {aug_type}: {stats['accuracy']*100:.1f}% acc ({stats['count']} samples)")

        print("=" * 70)

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'checkpoint': args.checkpoint,
                    'data': args.data,
                    'device': str(device),
                    'batch_size': args.batch_size,
                },
                'metrics': {
                    'accuracy': metrics['accuracy'],
                    'perplexity': metrics['perplexity'],
                    'mean_sample_perplexity': metrics['mean_sample_perplexity'],
                    'canon_pass_rate': metrics['canon_pass_rate'],
                    'num_samples': metrics['num_samples'],
                    'total_tokens': metrics['total_tokens'],
                    'canon_total': metrics['canon_total'],
                    'canon_pass': metrics['canon_pass'],
                },
                'per_type': metrics['per_type'],
                'sample_perplexities': metrics['sample_perplexities'][:100],  # First 100 for brevity
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"\nRuntime error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    return exit_code


if __name__ == '__main__':
    sys.exit(main())

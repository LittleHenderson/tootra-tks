"""
TKS Training with Augmented Data - Phase 4 (Train/Eval Rollout)

This script implements a complete training pipeline using augmented JSONL data
with the real TKSLLMCorePipeline model from tks_llm_core_v2.py and TKSTrainer
from training/trainer.py.

Training Loop:
    1. Load original + augmented JSONL files
    2. Build PyTorch Dataset/DataLoader
    3. Instantiate TKSLLMCorePipeline (real model)
    4. Run training with TKSTrainer (multi-component loss, curriculum support)
    5. Log metrics: loss curve, validator pass-rate, augmentation stats
    6. Evaluate on held-out subset with canonical validation

CLI Arguments:
    - Data paths (--data, --original-data)
    - Training params (--epochs, --batch-size, --learning-rate)
    - Model params (--hidden-dim, --vocab-size, --max-length)
    - Augmentation flags (--use-augmented, --filter-validated)
    - Output (--output-dir, --checkpoint-dir)

Author: TKS-LLM Training Integration Team
Date: 2025-12-14
Version: 4.0.0 (Phase 4 - Train/Eval Rollout)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TKS model and training components
try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    from training.losses import TKSLoss, TKSLossConfig, CurriculumLossScheduler
    from training.trainer import TrainingConfig, TKSTrainer, EvaluationResult
    from scripts.canonical_validator import CanonicalTKSValidator
    REAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import TKS model components: {e}")
    print("Falling back to dummy model for pipeline validation.")
    REAL_MODEL_AVAILABLE = False
    TOTAL_DIM = 40

    # Dummy validator
    class CanonicalTKSValidator:
        def validate(self, expr):
            return True

# Import augmentation metrics logger
try:
    from augmentation_metrics import AugmentationLogger, track_epoch_stats
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import augmentation_metrics. Metrics logging will be limited.")
    METRICS_AVAILABLE = False


# ==============================================================================
# CONSTANTS (TKS Canonical)
# ==============================================================================

# Allowed operators per guardrails
ALLOWED_OPS = {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'}

# World codes
WORLD_CODES = {'A', 'B', 'C', 'D'}

# Noetic indices (1-10)
NOETIC_INDICES = set(range(1, 11))

# Special tokens
PAD_TOKEN = 0
UNK_TOKEN = 1
BOS_TOKEN = 2
EOS_TOKEN = 3


# ==============================================================================
# TKS TOKENIZER
# ==============================================================================

class TKSTokenizer:
    """
    Simple tokenizer for TKS expressions and stories.

    Handles:
        - TKS element codes (A1-D10)
        - TKS operators (+, -, +T, -T, ->, <-, *T, /T, o)
        - Natural language stories (character-level fallback)
    """

    def __init__(self, vocab_size: int = 1000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Build vocabulary
        self.token_to_id = {
            '<PAD>': PAD_TOKEN,
            '<UNK>': UNK_TOKEN,
            '<BOS>': BOS_TOKEN,
            '<EOS>': EOS_TOKEN,
        }

        next_id = 4

        # Add TKS elements (A1-D10 for each world)
        for world in ['A', 'B', 'C', 'D']:
            for noetic in range(1, 11):
                token = f"{world}{noetic}"
                self.token_to_id[token] = next_id
                next_id += 1

        # Add operators
        for op in ALLOWED_OPS:
            self.token_to_id[op] = next_id
            next_id += 1

        # Add common punctuation and characters
        for c in ' .,!?;:\'"()-':
            self.token_to_id[c] = next_id
            next_id += 1

        # Add lowercase letters
        for c in 'abcdefghijklmnopqrstuvwxyz':
            self.token_to_id[c] = next_id
            next_id += 1

        # Add uppercase letters
        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                next_id += 1

        # Add digits
        for c in '0123456789':
            self.token_to_id[c] = next_id
            next_id += 1

        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Update vocab size
        self.actual_vocab_size = max(next_id, vocab_size)

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to list of token IDs."""
        tokens = [BOS_TOKEN]

        i = 0
        while i < len(text) and len(tokens) < self.max_length - 1:
            # Try to match TKS element (e.g., A10, B2)
            if i < len(text) - 1 and text[i] in WORLD_CODES:
                # Check for 2-digit noetic
                if i + 2 < len(text) and text[i+1:i+3].isdigit():
                    token = text[i:i+3]
                    if token in self.token_to_id:
                        tokens.append(self.token_to_id[token])
                        i += 3
                        continue
                # Check for 1-digit noetic
                if text[i+1].isdigit():
                    token = text[i:i+2]
                    if token in self.token_to_id:
                        tokens.append(self.token_to_id[token])
                        i += 2
                        continue

            # Try to match operators (longest first)
            matched = False
            for op in sorted(ALLOWED_OPS, key=len, reverse=True):
                if text[i:].startswith(op):
                    if op in self.token_to_id:
                        tokens.append(self.token_to_id[op])
                    else:
                        tokens.append(UNK_TOKEN)
                    i += len(op)
                    matched = True
                    break

            if matched:
                continue

            # Single character
            char = text[i]
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(UNK_TOKEN)
            i += 1

        tokens.append(EOS_TOKEN)

        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(PAD_TOKEN)

        return tokens[:self.max_length]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            if tid in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                continue
            token = self.id_to_token.get(tid, '<UNK>')
            tokens.append(token)
        return ''.join(tokens)


# ==============================================================================
# TKS DATASET
# ==============================================================================

class TKSAugmentedDataset(Dataset):
    """
    PyTorch Dataset for TKS augmented training data.

    Loads JSONL files with entries containing:
        - story: Natural language story
        - expr: TKS expression string
        - expr_elements: List of element codes
        - expr_ops: List of operators
        - aug_type: "original", "inversion", or "anti_attractor"
        - validator_pass: boolean
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: TKSTokenizer,
        max_length: int = 512,
        filter_validated: bool = False,
        use_expr: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to augmented JSONL file
            tokenizer: TKSTokenizer instance
            max_length: Maximum sequence length
            filter_validated: Only use entries with validator_pass=True
            use_expr: Use TKS expressions instead of stories
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_expr = use_expr

        self.entries = []
        self.load_data(data_path, filter_validated)

    def load_data(self, path: str, filter_validated: bool) -> None:
        """Load and parse JSONL file."""
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)

                    # Filter by validation if requested
                    if filter_validated and not entry.get('validator_pass', False):
                        continue

                    self.entries.append(entry)

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        entry = self.entries[idx]

        # Get text to tokenize
        if self.use_expr:
            text = entry.get('expr', '')
            if not text and entry.get('expr_elements'):
                # Reconstruct from elements and ops
                elements = entry['expr_elements']
                ops = entry.get('expr_ops', [])
                if elements:
                    text = elements[0]
                    for i, op in enumerate(ops):
                        if i + 1 < len(elements):
                            text += f" {op} {elements[i + 1]}"
        else:
            text = entry.get('story', '')

        # Tokenize
        input_ids = self.tokenizer.tokenize(text)

        # For language modeling, target is shifted input
        targets = input_ids[1:] + [PAD_TOKEN]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if tid != PAD_TOKEN else 0 for tid in input_ids]

        # Extract metadata
        aug_type = entry.get('aug_type', 'original')
        validator_pass = entry.get('validator_pass', False)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'aug_type': aug_type,
            'validator_pass': validator_pass,
        }


# ==============================================================================
# TRAINING METRICS
# ==============================================================================

class TrainingMetricsLogger:
    """
    Track training metrics across epochs and steps.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_losses = []
        self.step_losses = []
        self.eval_results = []
        self.aug_type_counts = Counter()
        self.validation_stats = {'total': 0, 'passed': 0}

        self.start_time = datetime.now()
        self.total_steps = 0
        self.total_samples = 0

    def log_step(self, epoch: int, step: int, loss: float, batch_size: int) -> None:
        """Log a training step."""
        self.step_losses.append({
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'batch_size': batch_size,
        })
        self.total_steps += 1
        self.total_samples += batch_size

    def log_epoch(self, epoch: int, avg_loss: float, entries: List[Dict]) -> None:
        """Log an epoch summary."""
        self.epoch_losses.append({
            'epoch': epoch,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat(),
        })

        # Track augmentation types
        for entry in entries:
            aug_type = entry.get('aug_type', 'unknown')
            self.aug_type_counts[aug_type] += 1

            self.validation_stats['total'] += 1
            if entry.get('validator_pass', False):
                self.validation_stats['passed'] += 1

    def log_eval(self, epoch: int, result: Dict[str, float]) -> None:
        """Log evaluation results."""
        result['epoch'] = epoch
        result['timestamp'] = datetime.now().isoformat()
        self.eval_results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get full metrics summary."""
        duration = (datetime.now() - self.start_time).total_seconds()

        pass_rate = (self.validation_stats['passed'] / self.validation_stats['total']
                    if self.validation_stats['total'] > 0 else 0.0)

        return {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'total_epochs': len(self.epoch_losses),
            'total_steps': self.total_steps,
            'total_samples': self.total_samples,
            'loss': {
                'epoch_losses': self.epoch_losses,
                'final_loss': self.epoch_losses[-1]['loss'] if self.epoch_losses else 0.0,
                'initial_loss': self.epoch_losses[0]['loss'] if self.epoch_losses else 0.0,
            },
            'validation': {
                'total': self.validation_stats['total'],
                'passed': self.validation_stats['passed'],
                'pass_rate': pass_rate,
            },
            'augmentation': {
                'distribution': dict(self.aug_type_counts),
                'original_count': self.aug_type_counts.get('original', 0),
                'inversion_count': self.aug_type_counts.get('inversion', 0),
                'anti_attractor_count': self.aug_type_counts.get('anti_attractor', 0),
            },
            'eval_results': self.eval_results,
        }

    def save(self, filename: str = "training_metrics.json") -> None:
        """Save metrics to file."""
        if not self.output_dir:
            return

        filepath = self.output_dir / filename
        with filepath.open('w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {filepath}")

    def print_summary(self) -> None:
        """Print formatted summary."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("TRAINING METRICS SUMMARY")
        print("=" * 70)

        print(f"\nDuration: {summary['duration_seconds']:.2f} seconds")
        print(f"Epochs: {summary['total_epochs']}")
        print(f"Steps: {summary['total_steps']}")
        print(f"Samples: {summary['total_samples']}")

        print(f"\nLoss: {summary['loss']['initial_loss']:.4f} -> {summary['loss']['final_loss']:.4f}")

        print(f"\nValidation pass rate: {summary['validation']['pass_rate']:.2%}")

        aug = summary['augmentation']
        print(f"\nAugmentation distribution:")
        print(f"  Original: {aug['original_count']}")
        print(f"  Inversions: {aug['inversion_count']}")
        print(f"  Anti-attractors: {aug['anti_attractor_count']}")

        print("=" * 70)


# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: TKSLLMCorePipeline or compatible model
        dataloader: DataLoader for evaluation data
        loss_fn: Loss function (TKSLoss or CrossEntropyLoss)
        device: Device to run on

    Returns:
        Dict with loss, accuracy, and component metrics
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        # Check if model has TKSLLMCorePipeline-like interface (duck-typing)
        # TKSLLMCorePipeline has rpm_gating and noetic_dim attributes
        is_tks_model = hasattr(model, 'rpm_gating') or (
            REAL_MODEL_AVAILABLE and isinstance(model, TKSLLMCorePipeline)
        )

        if is_tks_model:
            outputs = model(tokens=input_ids, return_full_trace=True)
            logits = outputs['logits']

            # Compute loss - TKSLoss requires pipeline argument
            if hasattr(loss_fn, 'config') and hasattr(loss_fn, 'task_loss'):  # Duck-type check for TKSLoss
                loss_dict = loss_fn(
                    pipeline_output=outputs,
                    targets=targets,
                    pipeline=model,
                    mask=attention_mask,
                    compute_all=False,
                )
                loss = loss_dict['total']
            else:
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Fallback for dummy/simple model
            output = model(input_ids)
            # Handle both dict (TKS-like) and tensor (simple) outputs
            if isinstance(output, dict):
                logits = output.get('logits', output)
            else:
                logits = output
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        total_loss += loss.item()

        # Compute accuracy (ignore padding)
        preds = logits.argmax(dim=-1)
        mask = attention_mask.bool()
        correct = ((preds == targets) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()

        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': min(torch.exp(torch.tensor(avg_loss)).item(), 1e6),
        'num_batches': num_batches,
        'total_tokens': total_tokens,
    }


# ==============================================================================
# SMOKE TEST
# ==============================================================================

def run_smoke_test(data_path: str, use_real_model: bool = True) -> bool:
    """
    Run smoke test to verify end-to-end training pipeline.

    Args:
        data_path: Path to augmented JSONL file
        use_real_model: Use real TKSLLMCorePipeline if available

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 70)
    print("SMOKE TEST - End-to-End Training Pipeline")
    print("=" * 70)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")

        # Test 1: Load data
        print("\n[Test 1] Loading data...")
        tokenizer = TKSTokenizer(vocab_size=1000, max_length=64)
        dataset = TKSAugmentedDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=64,
            filter_validated=False,
            use_expr=False,
        )
        print(f"  Loaded {len(dataset)} entries")
        assert len(dataset) > 0, "Dataset is empty"
        print("  [PASS] Data loading")

        # Test 2: Create DataLoader
        print("\n[Test 2] Creating DataLoader...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        assert batch['input_ids'].shape[0] == 4, "Batch size mismatch"
        print("  [PASS] DataLoader creation")

        # Test 3: Initialize model
        print("\n[Test 3] Initializing model...")
        if use_real_model and REAL_MODEL_AVAILABLE:
            model = TKSLLMCorePipeline(
                vocab_size=tokenizer.actual_vocab_size,
                hidden_dim=64,
                noetic_dim=TOTAL_DIM,
            ).to(device)
            loss_fn = TKSLoss(TKSLossConfig())
            print(f"  Using TKSLLMCorePipeline (real model)")
        else:
            # Simple fallback model
            model = nn.Sequential(
                nn.Embedding(tokenizer.actual_vocab_size, 64),
                nn.Linear(64, tokenizer.actual_vocab_size),
            ).to(device)
            loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            print(f"  Using simple fallback model")
        print("  [PASS] Model initialization")

        # Test 4: Forward pass
        print("\n[Test 4] Forward pass...")
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        if REAL_MODEL_AVAILABLE and isinstance(model, TKSLLMCorePipeline):
            outputs = model(tokens=input_ids, return_full_trace=True)
            logits = outputs['logits']
        else:
            logits = model(input_ids)

        print(f"  Logits shape: {logits.shape}")
        assert logits.shape[:2] == input_ids.shape, "Output shape mismatch"
        print("  [PASS] Forward pass")

        # Test 5: Loss computation
        print("\n[Test 5] Loss computation...")
        if isinstance(loss_fn, TKSLoss):
            loss_dict = loss_fn(
                pipeline_output=outputs,
                targets=targets,
                pipeline=model,
                compute_all=True,
            )
            loss = loss_dict['total']
            print(f"  Total loss: {loss.item():.4f}")
            print(f"  Task loss: {loss_dict.get('task', 0):.4f}")
        else:
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            print(f"  Loss: {loss.item():.4f}")

        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"
        print("  [PASS] Loss computation")

        # Test 6: Backward pass
        print("\n[Test 6] Backward pass...")
        optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("  [PASS] Backward pass")

        # Test 7: Evaluation
        print("\n[Test 7] Evaluation...")
        eval_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        eval_results = evaluate_model(model, eval_loader, loss_fn, device)
        print(f"  Eval loss: {eval_results['loss']:.4f}")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print("  [PASS] Evaluation")

        print("\n" + "=" * 70)
        print("[PASS] ALL SMOKE TESTS PASSED")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n[FAIL] SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train(args) -> None:
    """
    Main training function using TKSTrainer.

    Args:
        args: Parsed command line arguments
    """
    print("=" * 70)
    print("TKS TRAINING WITH AUGMENTED DATA - Phase 4 (Train/Eval Rollout)")
    print("=" * 70)

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Initialize tokenizer
    tokenizer = TKSTokenizer(vocab_size=args.vocab_size, max_length=args.max_length)
    print(f"Tokenizer vocabulary size: {tokenizer.actual_vocab_size}")

    # Load dataset
    print(f"\nLoading data from: {args.data}")
    dataset = TKSAugmentedDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        filter_validated=args.filter_validated,
        use_expr=args.use_expr,
    )
    print(f"Loaded {len(dataset)} entries")

    # Count augmentation types
    aug_counts = Counter(e.get('aug_type', 'unknown') for e in dataset.entries)
    print(f"\nAugmentation distribution:")
    for aug_type, count in sorted(aug_counts.items()):
        print(f"  {aug_type}: {count}")

    # Split into train/eval
    eval_size = min(int(len(dataset) * 0.1), 100)
    train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    print(f"\nTrain size: {train_size}")
    print(f"Eval size: {eval_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == 'cuda',
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    print(f"\nInitializing model...")
    if REAL_MODEL_AVAILABLE and not args.use_dummy:
        model = TKSLLMCorePipeline(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=args.hidden_dim,
            noetic_dim=TOTAL_DIM,
            num_scales=3,
            max_attractor_iter=10,
            contraction_factor=0.5,
        )

        print(f"  Model: TKSLLMCorePipeline")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Noetic dim: {TOTAL_DIM}")
    else:
        # Fallback model
        model = nn.Sequential(
            nn.Embedding(tokenizer.actual_vocab_size, args.hidden_dim),
            nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True),
            nn.Linear(args.hidden_dim, tokenizer.actual_vocab_size),
        )
        print(f"  Model: Simple LSTM fallback")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Configure loss
    loss_config = TKSLossConfig(
        lambda_task=1.0,
        lambda_rpm=0.3,
        lambda_attractor=0.2,
        lambda_involution=0.2,
        lambda_spectral=0.1,
        lambda_cascade=0.1,
    )

    # Create training configuration
    training_config = TrainingConfig(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=args.hidden_dim,
        noetic_dim=TOTAL_DIM,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        loss_config=loss_config,
        use_curriculum=args.use_curriculum,
        checkpoint_dir=args.checkpoint_dir if hasattr(args, 'checkpoint_dir') else str(Path(args.output_dir) / "checkpoints"),
        log_dir=str(Path(args.output_dir) / "logs"),
        log_every=args.log_interval,
        eval_every=args.eval_interval if hasattr(args, 'eval_interval') else 500,
        checkpoint_every=args.checkpoint_interval if hasattr(args, 'checkpoint_interval') else 1000,
        save_best=True,
        device="auto",
        seed=args.seed,
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Weight decay: {training_config.weight_decay}")
    print(f"  Use curriculum: {training_config.use_curriculum}")
    print(f"  Loss config: {loss_config.to_dict()}")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Initialize metrics logger
    metrics = TrainingMetricsLogger(output_dir=metrics_dir)

    # Dry-run mode
    if args.dry_run:
        print("\n[DRY-RUN MODE] Running single epoch only")
        training_config.epochs = 1

    # Use TKSTrainer if real model is available
    if REAL_MODEL_AVAILABLE and not args.use_dummy:
        print("\n" + "=" * 70)
        print("TRAINING WITH TKSTrainer")
        print("=" * 70)

        # Create trainer
        trainer = TKSTrainer(model, training_config)

        # Convert dataset to use proper format for TKSTrainer
        # The trainer expects 'input_ids' and 'targets' keys
        train_state = trainer.train(train_dataset, eval_dataset)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final step: {train_state.global_step}")
        print(f"Final loss: {train_state.loss_history[-1]:.4f}")
        print(f"Best loss: {train_state.best_loss:.4f}")

        # Save training summary
        summary = {
            'global_step': train_state.global_step,
            'epoch': train_state.epoch,
            'final_loss': train_state.loss_history[-1] if train_state.loss_history else None,
            'initial_loss': train_state.loss_history[0] if train_state.loss_history else None,
            'best_loss': train_state.best_loss,
            'eval_history': train_state.eval_history,
        }

        summary_path = output_dir / "training_summary.json"
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nTraining summary saved to: {summary_path}")

    else:
        # Fallback to original training loop for dummy model
        print("\n" + "=" * 70)
        print("TRAINING LOOP (Fallback Mode)")
        print("=" * 70)

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        total_steps = len(train_loader) * args.epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

        model.to(device)
        best_eval_loss = float('inf')
        global_step = 0

        for epoch in range(args.epochs if not args.dry_run else 1):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            epoch_entries = []

            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 70)

            for batch_idx, batch in enumerate(train_loader):
                if args.dry_run and batch_idx > 0:
                    break

                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)

                optimizer.zero_grad()

                if hasattr(model, '__getitem__'):
                    embedded = model[0](input_ids)
                    lstm_out, _ = model[1](embedded)
                    logits = model[2](lstm_out)
                else:
                    logits = model(input_ids)

                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                for i in range(len(batch['aug_type'])):
                    epoch_entries.append({
                        'aug_type': batch['aug_type'][i],
                        'validator_pass': batch['validator_pass'][i].item() if torch.is_tensor(batch['validator_pass'][i]) else batch['validator_pass'][i],
                    })

                metrics.log_step(epoch + 1, batch_idx, loss.item(), len(input_ids))

                if batch_idx % args.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Step {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}, lr={lr:.2e}")

            avg_loss = epoch_loss / max(epoch_steps, 1)
            metrics.log_epoch(epoch + 1, avg_loss, epoch_entries)

            print(f"\n  Epoch {epoch + 1} Summary: Average loss: {avg_loss:.4f}")

            if eval_dataset and not args.dry_run:
                print(f"\n  Evaluating...")
                eval_results = evaluate_model(model, eval_loader, loss_fn, device)
                metrics.log_eval(epoch + 1, eval_results)
                print(f"    Eval loss: {eval_results['loss']:.4f}, Accuracy: {eval_results['accuracy']:.4f}")

                if eval_results['loss'] < best_eval_loss:
                    best_eval_loss = eval_results['loss']
                    torch.save(model.state_dict(), output_dir / "best_model.pt")
                    print(f"    [NEW BEST] Saved best_model.pt")

        torch.save(model.state_dict(), output_dir / "final_model.pt")
        print(f"\nSaved final_model.pt")

        metrics.save()
        metrics.print_summary()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

    if args.dry_run:
        print("\n[DRY-RUN MODE] Pipeline validation successful!")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train TKS model using augmented JSONL data (Phase 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Full training run
    python train_with_augmented.py --data output/augmented_corpus.jsonl --epochs 10

    # Smoke test
    python train_with_augmented.py --data output/sample_augmented.jsonl --test

    # Dry run (single batch)
    python train_with_augmented.py --data output/sample_augmented.jsonl --dry-run

    # Use dummy model for testing
    python train_with_augmented.py --data output/sample_augmented.jsonl --use-dummy
        """
    )

    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to augmented JSONL file')
    parser.add_argument('--original-data', type=str, default=None,
                       help='Optional path to original corpus for comparison')
    parser.add_argument('--filter-validated', action='store_true',
                       help='Only use validated entries')
    parser.add_argument('--use-expr', action='store_true',
                       help='Train on TKS expressions instead of stories')

    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size (default: 1000)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension (default: 128)')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--use-dummy', action='store_true',
                       help='Use dummy model instead of TKSLLMCorePipeline')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum training steps')
    parser.add_argument('--use-curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Checkpoint directory (default: output-dir/checkpoints)')
    parser.add_argument('--eval-interval', type=int, default=500,
                       help='Evaluate every N steps (default: 500)')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                       help='Save checkpoint every N steps (default: 1000)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='output/models',
                       help='Output directory (default: output/models)')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log every N batches (default: 10)')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--test', action='store_true',
                       help='Run smoke test')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run single batch for validation')

    args = parser.parse_args()

    # Run smoke test if requested
    if args.test:
        success = run_smoke_test(args.data, use_real_model=not args.use_dummy)
        sys.exit(0 if success else 1)

    # Run training
    train(args)


if __name__ == '__main__':
    main()

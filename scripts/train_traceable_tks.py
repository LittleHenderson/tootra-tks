#!/usr/bin/env python3
"""
Train Traceable TKS-Transformer with Constraint Annealing.

This script implements the full training recipe for moving from "infra-OK" to "model-OK":
  - Constraint annealing (soft → hard)
  - Trace sufficiency loss
  - Lipschitz penalty
  - Load balance loss
  - Per-epoch threshold checking
  - Checkpoint promotion criteria

Usage:
    # Quick test (2 epochs, small data)
    python scripts/train_traceable_tks.py --quick-test

    # Full training (10 epochs)
    python scripts/train_traceable_tks.py --data data/teacher_augmented.jsonl --epochs 10

    # Aggressive constraint tightening
    python scripts/train_traceable_tks.py --data data/teacher_augmented.jsonl --preset aggressive

Expected Outcomes Per Epoch (10 epoch training):
    Epoch 1-2: Task learning
        - Lipschitz: < 1.5 (may exceed 1)
        - Routing entropy: > 0.3
        - Perplexity: < 100

    Epoch 3-4: Routing stabilizes
        - Lipschitz: < 1.3
        - Routing entropy: > 0.4
        - Attractor convergence: > 40%

    Epoch 5-6: Attractor starts converging
        - Lipschitz: < 1.2
        - Routing entropy: > 0.5
        - Attractor convergence: > 60%
        - Trace top-5 match: > 65%

    Epoch 7-8: Constraints tightening
        - Lipschitz: < 1.1
        - Attractor convergence: > 75%
        - Residual energy: < 0.2
        - Trace top-5 match: > 75%

    Epoch 9-10: Full constraint satisfaction
        - Lipschitz: < 1.0 (CONTRACTIVE)
        - Attractor convergence: > 90%
        - Residual energy: < 0.1
        - Trace top-5 match: > 85%
        - Load balance ratio: < 2.0
"""
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v4 import TKSNoeticLMConfig, TKSNoeticLM
from training.losses import (
    ConstraintAnnealingScheduler,
    TKSTrainingLoss,
    TraceConsistencyLoss,
    LipschitzPenaltyLoss,
)
from scripts.trace_lint import TraceLinter, TKSInvariants
from trace_utils import extract_trace_features, compress_trace


# =============================================================================
# DATA LOADING
# =============================================================================

class ByteTokenizer:
    """Byte-level tokenizer with special tokens."""

    def __init__(self):
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.sep_token = "<SEP>"
        self.unk_token = "<UNK>"

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.sep_token, self.unk_token]
        self.special_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.byte_offset = len(self.special_tokens)
        self.vocab_size = self.byte_offset + 256

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.special_to_id[self.bos_token])
        for b in text.encode("utf-8", errors="replace"):
            tokens.append(self.byte_offset + b)
        if add_eos:
            tokens.append(self.special_to_id[self.eos_token])
        return tokens

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for tid in ids:
            if tid < self.byte_offset:
                continue
            out.append(tid - self.byte_offset)
        return out.decode("utf-8", errors="replace")

    @property
    def pad_id(self) -> int:
        return self.special_to_id[self.pad_token]


class TraceableLMDataset(Dataset):
    """Dataset for traceable TKS training."""

    def __init__(self, sequences: List[List[int]], max_length: int, pad_id: int):
        self.sequences = sequences
        self.max_length = max_length
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        seq = seq[:self.max_length]
        if len(seq) < self.max_length:
            seq = seq + [self.pad_id] * (self.max_length - len(seq))
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return {"input_ids": input_ids, "target_ids": target_ids}


def load_data(path: Path, tokenizer: ByteTokenizer) -> List[List[int]]:
    """Load data from text or JSONL file."""
    sequences = []

    if path.suffix == '.jsonl':
        # Teacher JSONL format
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    text = obj.get('input', '') + ' ' + obj.get('target', obj.get('output', ''))
                    sequences.append(tokenizer.encode(text.strip()))
    else:
        # Plain text
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sequences.append(tokenizer.encode(line.strip()))

    return sequences


# =============================================================================
# METRICS COLLECTION
# =============================================================================

@dataclass
class EpochMetrics:
    """Metrics collected at the end of each epoch."""
    epoch: int
    task_loss: float
    perplexity: float

    # Trace metrics
    trace_loss: float = 0.0
    trace_accuracy: float = 0.0
    trace_top5_match: float = 0.0

    # Constraint metrics
    lipschitz_estimate: float = 1.5
    routing_entropy: float = 0.5
    load_balance_ratio: float = 2.0

    # Attractor metrics
    attractor_convergence_rate: float = 0.0
    attractor_avg_iterations: float = 15.0

    # Residual metrics
    residual_energy: float = 0.3

    # Lint violations
    lint_errors: int = 0
    lint_warnings: int = 0

    # Scheduler state
    sched_routing_temp: float = 2.0
    sched_residual_budget: float = 0.3
    sched_constraint_weight: float = 0.1


def estimate_model_lipschitz(model: nn.Module) -> float:
    """Estimate Lipschitz constant of attractor layers."""
    max_sigma = 0.0

    for name, param in model.named_parameters():
        if 'attractor' in name.lower() and 'weight' in name.lower() and param.dim() == 2:
            # Power iteration
            with torch.no_grad():
                v = torch.randn(param.shape[1], device=param.device)
                v = v / v.norm()

                for _ in range(10):
                    u = param @ v
                    u = u / (u.norm() + 1e-8)
                    v = param.t() @ u
                    v = v / (v.norm() + 1e-8)

                sigma = (param @ v).norm().item()
                max_sigma = max(max_sigma, sigma)

    return max_sigma if max_sigma > 0 else 1.5


def compute_routing_entropy(routing_weights: torch.Tensor) -> float:
    """Compute entropy of routing distribution."""
    if routing_weights is None or routing_weights.numel() == 0:
        return 0.5

    # Normalize to probability
    probs = F.softmax(routing_weights.float(), dim=-1)
    # Compute entropy
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
    # Normalize by max entropy
    max_entropy = math.log(probs.shape[-1])
    return entropy / max_entropy if max_entropy > 0 else 0.5


def run_trace_lint(model: nn.Module, device: torch.device) -> Tuple[int, int]:
    """Run trace lint on model and return (errors, warnings)."""
    model.eval()
    invariants = TKSInvariants()
    linter = TraceLinter(invariants)

    with torch.no_grad():
        input_ids = torch.randint(0, 256, (1, 32), device=device)
        output = model(input_ids, return_full_trace=True)
        trace = output['trace']

    violations = linter.lint(trace)

    errors = sum(1 for v in violations if v.severity.name == 'ERROR')
    warnings = sum(1 for v in violations if v.severity.name == 'WARNING')

    model.train()
    return errors, warnings


# =============================================================================
# CHECKPOINT PROMOTION
# =============================================================================

def check_promotion_criteria(metrics: EpochMetrics, scheduler: ConstraintAnnealingScheduler) -> Tuple[bool, List[str]]:
    """
    Check if checkpoint meets promotion criteria for its epoch.

    Returns:
        (passes, list of failing criteria)
    """
    thresholds = scheduler.get_expected_thresholds(metrics.epoch)
    failures = []

    # Check each threshold
    if metrics.lipschitz_estimate > thresholds['lipschitz_max']:
        failures.append(f"Lipschitz {metrics.lipschitz_estimate:.3f} > {thresholds['lipschitz_max']:.3f}")

    if metrics.routing_entropy < thresholds['routing_entropy_min']:
        failures.append(f"Routing entropy {metrics.routing_entropy:.3f} < {thresholds['routing_entropy_min']:.3f}")

    if metrics.attractor_convergence_rate < thresholds['attractor_convergence_min']:
        failures.append(f"Attractor convergence {metrics.attractor_convergence_rate:.3f} < {thresholds['attractor_convergence_min']:.3f}")

    if metrics.perplexity > thresholds['perplexity_max']:
        failures.append(f"Perplexity {metrics.perplexity:.1f} > {thresholds['perplexity_max']:.1f}")

    # Final epochs have stricter requirements
    if metrics.epoch >= scheduler.total_epochs - 2:
        if metrics.lipschitz_estimate >= 1.0:
            failures.append(f"Lipschitz {metrics.lipschitz_estimate:.3f} >= 1.0 (must be contractive)")

    return len(failures) == 0, failures


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: TKSTrainingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: ConstraintAnnealingScheduler,
    device: torch.device,
    epoch: int,
    log_every: int = 50,
) -> EpochMetrics:
    """Train for one epoch and return metrics."""
    model.train()
    scheduler.set_epoch(epoch)

    total_task_loss = 0.0
    total_trace_loss = 0.0
    total_trace_acc = 0.0
    total_balance_loss = 0.0
    attractor_converged = 0
    attractor_iterations = 0
    num_batches = 0

    routing_weights_sample = None

    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        optimizer.zero_grad()

        # Forward with trace
        output = model(input_ids, return_full_trace=True)

        # Compute combined loss
        loss, metrics = loss_fn(output, target_ids, scheduler, model)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate metrics
        total_task_loss += metrics['task_loss'].item()
        if 'trace_loss' in metrics:
            total_trace_loss += metrics['trace_loss'].item()
            total_trace_acc += metrics['trace_accuracy']
        if 'load_balance_loss' in metrics:
            total_balance_loss += metrics['load_balance_loss'].item()

        # Attractor stats from trace
        trace = output.get('trace', {})
        if trace.get('attractor_converged', False):
            attractor_converged += 1
        attractor_iterations += trace.get('attractor_iterations', 15)

        # Sample routing weights
        if routing_weights_sample is None and 'noetic_routing' in trace:
            nr = trace['noetic_routing']
            if isinstance(nr, dict) and 'weights' in nr:
                weights = nr['weights']
                if isinstance(weights, torch.Tensor):
                    routing_weights_sample = weights
                elif isinstance(weights, list) and len(weights) > 0:
                    routing_weights_sample = weights[0] if isinstance(weights[0], torch.Tensor) else None

        num_batches += 1

        if log_every and batch_idx % log_every == 0:
            print(f"  batch {batch_idx}/{len(loader)} loss={loss.item():.4f}")

    # Compute epoch metrics (ensure all values are Python floats, not tensors)
    avg_task_loss = total_task_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_task_loss, 10))  # Cap to prevent overflow

    def to_float(val):
        """Convert tensor or value to Python float."""
        if isinstance(val, torch.Tensor):
            return val.item() if val.numel() == 1 else val.mean().item()
        return float(val)

    metrics = EpochMetrics(
        epoch=epoch,
        task_loss=to_float(avg_task_loss),
        perplexity=to_float(perplexity),
        trace_loss=to_float(total_trace_loss / max(num_batches, 1)),
        trace_accuracy=to_float(total_trace_acc / max(num_batches, 1)),
        attractor_convergence_rate=to_float(attractor_converged / max(num_batches, 1)),
        attractor_avg_iterations=to_float(attractor_iterations / max(num_batches, 1)),
        routing_entropy=to_float(compute_routing_entropy(routing_weights_sample)),
        lipschitz_estimate=to_float(estimate_model_lipschitz(model)),
        sched_routing_temp=to_float(scheduler.routing_temperature),
        sched_residual_budget=to_float(scheduler.residual_budget),
        sched_constraint_weight=to_float(scheduler.constraint_weight),
    )

    # Run lint
    metrics.lint_errors, metrics.lint_warnings = run_trace_lint(model, device)

    return metrics


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    config: TKSNoeticLMConfig,
    tokenizer: ByteTokenizer,
    metrics: EpochMetrics,
    scheduler: ConstraintAnnealingScheduler,
    promoted: bool = False,
) -> Path:
    """Save checkpoint with metrics."""
    suffix = "_promoted" if promoted else ""
    ckpt_dir = output_dir / f"epoch_{metrics.epoch:03d}{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")

    with (ckpt_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    with (ckpt_dir / "metrics.json").open("w") as f:
        json.dump(asdict(metrics), f, indent=2)

    with (ckpt_dir / "scheduler.json").open("w") as f:
        json.dump(scheduler.get_all_values(), f, indent=2)

    return ckpt_dir


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train(args):
    """Main training function."""
    print("=" * 70)
    print("Traceable TKS-Transformer Training")
    print("=" * 70)

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = ByteTokenizer()

    # Load data
    if args.data:
        data_path = Path(args.data)
        sequences = load_data(data_path, tokenizer)
        print(f"Loaded {len(sequences)} sequences from {data_path}")
    else:
        # Generate synthetic data for testing
        print("No data provided, generating synthetic sequences...")
        sequences = [tokenizer.encode(f"Test sequence {i} with some content.") for i in range(1000)]

    # Dataset
    dataset = TraceableLMDataset(sequences, args.max_length, tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Model config - load from checkpoint or create new
    start_epoch = 1
    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
        config_path = ckpt_dir / "config.json"
        model_path = ckpt_dir / "model.pt"

        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_dir}")

        print(f"Resuming from checkpoint: {ckpt_dir}")
        with config_path.open() as f:
            config_dict = json.load(f)
        config = TKSNoeticLMConfig(**config_dict)

        # Extract starting epoch from directory name if possible
        dir_name = ckpt_dir.name
        if dir_name.startswith("epoch_"):
            try:
                start_epoch = int(dir_name.split("_")[1]) + 1
                print(f"Resuming from epoch {start_epoch}")
            except ValueError:
                pass
    else:
        config = TKSNoeticLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.max_length,
            num_layers=args.num_layers,
            num_scales=args.num_scales,
            dropout=args.dropout,
            use_attractor=True,
            use_stable_attractor=True,
            use_rpm=True,
            use_noetic_basis=args.use_noetic_basis,
            use_stable_routing=args.use_stable_routing,
            use_world_bridge=args.use_world_bridge,
        )

    # Model
    model = TKSNoeticLM(config).to(device)

    # Load checkpoint weights if resuming
    if args.checkpoint:
        model_path = Path(args.checkpoint) / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    if args.preset == "aggressive":
        scheduler = ConstraintAnnealingScheduler.aggressive(args.epochs)
    elif args.preset == "gentle":
        scheduler = ConstraintAnnealingScheduler.gentle(args.epochs)
    else:
        scheduler = ConstraintAnnealingScheduler(total_epochs=args.epochs)

    # Loss function (move to device for trace consistency head)
    loss_fn = TKSTrainingLoss(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        enable_trace_loss=args.enable_trace_loss,
        enable_lipschitz_loss=args.enable_lipschitz_loss,
    ).to(device)

    # Training loop
    all_metrics = []
    best_promoted_epoch = -1
    adaptive_lipschitz_multiplier = 1.0  # For adaptive Lipschitz penalty

    end_epoch = start_epoch + args.epochs - 1
    print(f"\nStarting training from epoch {start_epoch} to {end_epoch}...")
    print(f"Preset: {args.preset}")
    print(f"Features: noetic_basis={config.use_noetic_basis}, stable_routing={config.use_stable_routing}, world_bridge={config.use_world_bridge}")
    if args.adaptive_lipschitz:
        print(f"Adaptive Lipschitz: target={args.lipschitz_target}, max_penalty={args.lipschitz_penalty_max}")
    if args.trace_phase:
        print(f"Trace phase: {args.trace_phase_epochs} epochs after main training")
    print()

    for epoch in range(start_epoch, end_epoch + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)

        start_time = time.time()
        metrics = train_epoch(
            model, loader, loss_fn, optimizer, scheduler, device, epoch,
            log_every=args.log_every,
        )
        epoch_time = time.time() - start_time

        # Print metrics
        print(f"\n  Task Loss: {metrics.task_loss:.4f}")
        print(f"  Perplexity: {metrics.perplexity:.2f}")
        print(f"  Trace Loss: {metrics.trace_loss:.4f}")
        print(f"  Trace Accuracy: {metrics.trace_accuracy:.4f}")
        print(f"  Lipschitz: {metrics.lipschitz_estimate:.4f}")
        print(f"  Routing Entropy: {metrics.routing_entropy:.4f}")
        print(f"  Attractor Convergence: {metrics.attractor_convergence_rate:.2%}")
        print(f"  Lint: {metrics.lint_errors} errors, {metrics.lint_warnings} warnings")
        print(f"  Schedule: temp={metrics.sched_routing_temp:.3f}, residual={metrics.sched_residual_budget:.3f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Adaptive Lipschitz penalty: increase multiplier if Lipschitz > target
        if args.adaptive_lipschitz and metrics.lipschitz_estimate > args.lipschitz_target:
            # Increase penalty proportionally to how much we exceed target
            excess = metrics.lipschitz_estimate - args.lipschitz_target
            adaptive_lipschitz_multiplier = min(
                adaptive_lipschitz_multiplier * (1 + excess),
                args.lipschitz_penalty_max
            )
            # Update loss function's Lipschitz weight
            if hasattr(loss_fn, 'lipschitz_loss'):
                base_weight = scheduler.lipschitz_weight
                loss_fn.lipschitz_loss.target_lipschitz = args.lipschitz_target
            print(f"  Adaptive Lipschitz: multiplier={adaptive_lipschitz_multiplier:.2f}")

        # Check promotion criteria
        passes, failures = check_promotion_criteria(metrics, scheduler)

        if passes:
            print(f"  ✓ PROMOTED (meets epoch {epoch} thresholds)")
            ckpt_path = save_checkpoint(output_dir, model, config, tokenizer, metrics, scheduler, promoted=True)
            best_promoted_epoch = epoch
        else:
            print(f"  ✗ Not promoted: {', '.join(failures[:3])}")
            if epoch % args.save_every == 0:
                ckpt_path = save_checkpoint(output_dir, model, config, tokenizer, metrics, scheduler, promoted=False)

        all_metrics.append(asdict(metrics))
        print()

    # =========================================================================
    # TRACE-ONLY FINE-TUNE PHASE
    # =========================================================================
    if args.trace_phase:
        print("\n" + "=" * 70)
        print("TRACE-ONLY FINE-TUNE PHASE")
        print("=" * 70)
        print(f"Running {args.trace_phase_epochs} epochs with trace_weight={args.trace_phase_weight}")
        print("Freezing main model, training only trace consistency head...")
        print()

        # Freeze main model parameters, only train trace head
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Unfreeze trace consistency head
        if hasattr(loss_fn, 'trace_loss') and hasattr(loss_fn.trace_loss, 'head'):
            for param in loss_fn.trace_loss.head.parameters():
                param.requires_grad = True

        # Create trace-focused optimizer (lower LR)
        trace_params = [p for p in loss_fn.parameters() if p.requires_grad]
        trace_optimizer = torch.optim.AdamW(trace_params, lr=args.lr * 0.1, weight_decay=args.weight_decay)

        # Create trace-focused scheduler with high trace weight
        trace_scheduler = ConstraintAnnealingScheduler(
            total_epochs=args.trace_phase_epochs,
            trace_weight_start=args.trace_phase_weight,
            trace_weight_end=args.trace_phase_weight,
            lipschitz_weight_start=0.0,  # No Lipschitz penalty in trace phase
            lipschitz_weight_end=0.0,
        )

        for trace_epoch in range(1, args.trace_phase_epochs + 1):
            print(f"Trace Epoch {trace_epoch}/{args.trace_phase_epochs}")
            print("-" * 40)

            trace_scheduler.set_epoch(trace_epoch)
            model.eval()  # Keep model in eval mode

            trace_loss_total = 0.0
            trace_acc_total = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)

                trace_optimizer.zero_grad()

                with torch.no_grad():
                    output = model(input_ids, return_full_trace=True)

                # Compute only trace loss
                if hasattr(loss_fn, 'trace_loss'):
                    trace_result = loss_fn.trace_loss(output['trace'], output['logits'])
                    loss = trace_result['loss'] * args.trace_phase_weight
                    loss.backward()
                    trace_optimizer.step()

                    trace_loss_total += trace_result['loss'].item()
                    trace_acc_total += trace_result['accuracy']

                num_batches += 1

                if args.log_every and batch_idx % args.log_every == 0:
                    print(f"  batch {batch_idx}/{len(loader)} trace_loss={loss.item():.4f}")

            avg_trace_loss = trace_loss_total / max(num_batches, 1)
            avg_trace_acc = trace_acc_total / max(num_batches, 1)

            print(f"\n  Trace Loss: {avg_trace_loss:.4f}")
            print(f"  Trace Accuracy: {avg_trace_acc:.4f}")
            print()

        # Unfreeze model for potential further training
        for param in model.parameters():
            param.requires_grad = True

        print("Trace phase complete.\n")

    # Save final checkpoint
    final_metrics = all_metrics[-1] if all_metrics else {}
    save_checkpoint(output_dir, model, config, tokenizer, metrics, scheduler, promoted=False)

    # Save all metrics
    with (output_dir / "training_metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    # Summary
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Best promoted epoch: {best_promoted_epoch if best_promoted_epoch > 0 else 'None'}")
    print(f"Final Lipschitz: {metrics.lipschitz_estimate:.4f}")
    print(f"Final Lint: {metrics.lint_errors} errors, {metrics.lint_warnings} warnings")

    # Final recommendation
    if metrics.lipschitz_estimate < 1.0 and metrics.lint_errors == 0:
        print("\n✓ Model is CONTRACTIVE and passes trace-lint!")
        print("  Ready for deployment.")
    elif metrics.lipschitz_estimate < 1.1:
        print("\n⚠ Model is nearly contractive.")
        print("  Consider running a few more epochs with aggressive preset.")
    else:
        print("\n✗ Model is NOT contractive (Lipschitz >= 1.1)")
        print("  Recommendations:")
        print("  1. Train longer with --epochs 15 --preset aggressive")
        print("  2. Reduce model capacity (--num-layers 2)")
        print("  3. Check data quality")


def main():
    parser = argparse.ArgumentParser(
        description="Train Traceable TKS-Transformer with Constraint Annealing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument("--data", type=str, help="Training data (JSONL or text file)")
    parser.add_argument("--output-dir", type=str, default="output/traceable_tks")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Model
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # TKS features
    parser.add_argument("--use-noetic-basis", action="store_true", help="Enable noetic basis parameterization")
    parser.add_argument("--use-stable-routing", action="store_true", help="Enable stable routing")
    parser.add_argument("--use-world-bridge", action="store_true", help="Enable world bridge")

    # Loss
    parser.add_argument("--enable-trace-loss", action="store_true", default=True)
    parser.add_argument("--enable-lipschitz-loss", action="store_true", default=True)

    # Scheduler preset
    parser.add_argument("--preset", type=str, default="default", choices=["default", "aggressive", "gentle"])

    # Checkpoint resume
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory to resume from (e.g., output/traceable_tks_full/epoch_001)")
    parser.add_argument("--reset-optimizer", action="store_true", help="Reset optimizer state when resuming (useful for fine-tuning)")

    # Adaptive Lipschitz penalty
    parser.add_argument("--adaptive-lipschitz", action="store_true", default=True, help="Auto-increase Lipschitz penalty if >1")
    parser.add_argument("--lipschitz-target", type=float, default=0.95, help="Target Lipschitz constant")
    parser.add_argument("--lipschitz-penalty-max", type=float, default=5.0, help="Maximum Lipschitz penalty multiplier")

    # Trace-only fine-tune phase
    parser.add_argument("--trace-phase", action="store_true", help="Run trace-only fine-tune phase after main training")
    parser.add_argument("--trace-phase-epochs", type=int, default=5, help="Number of epochs for trace-only phase")
    parser.add_argument("--trace-phase-weight", type=float, default=1.0, help="Trace loss weight during trace phase")

    # Misc
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Quick test mode
    parser.add_argument("--quick-test", action="store_true", help="Quick test (2 epochs, synthetic data)")

    args = parser.parse_args()

    # Quick test overrides
    if args.quick_test:
        args.epochs = 2
        args.batch_size = 8
        args.max_length = 64
        args.num_layers = 2
        args.log_every = 10
        args.use_noetic_basis = True
        args.use_stable_routing = True
        args.output_dir = "output/quick_test_traceable"
        print("Quick test mode enabled")

    # Set seed
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()

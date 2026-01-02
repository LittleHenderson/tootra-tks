#!/usr/bin/env python3
"""
World/RPM Supervised Training Script

This script trains the TKSLLMCorePipeline with explicit world separation
and RPM differentiation losses to fix:
    1. World Separation Failure: Text "A1+A2+A3" activates World B higher than World A
    2. RPM Gate Collapse: All D/W/P categories cluster at ~0.94 with no differentiation

Usage:
    # Generate labeled data first
    python scripts/generate_world_rpm_labels.py --output data/world_rpm_labeled.jsonl --num-examples 2000

    # Train with world/RPM supervision
    python scripts/train_world_rpm_supervised.py --data data/world_rpm_labeled.jsonl --epochs 20

    # Resume from checkpoint
    python scripts/train_world_rpm_supervised.py --data data/world_rpm_labeled.jsonl --resume output/world_rpm_models/best_model.pt

Author: TKS-LLM Agent A
Date: 2025-12-22
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TKS components
try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    from training.losses import (
        TKSLoss,
        TKSLossConfig,
        WorldClassificationLoss,
        RPMDifferentiationLoss,
        WorldRPMSupervisedLoss,
    )
    TKS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import TKS components: {e}")
    TKS_AVAILABLE = False


# =============================================================================
# TOKENIZER
# =============================================================================

class TKSTokenizer:
    """Tokenizer for TKS World/RPM labeled data."""

    def __init__(self, vocab_size: int = 1000, max_length: int = 256):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Special tokens
        self.token_to_id = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4,
        }

        next_id = 5

        # TKS elements (A0-D9)
        for world in ['A', 'B', 'C', 'D']:
            for noetic in range(10):
                self.token_to_id[f"{world}{noetic}"] = next_id
                next_id += 1

        # TKS operators
        for op in ['+', '-', '*', '/', '->', '<-', '+T', '-T', '*T', '/T', 'o', '^']:
            self.token_to_id[op] = next_id
            next_id += 1

        # Characters
        chars = ' .,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_=<>'
        for c in chars:
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                next_id += 1

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.actual_vocab_size = next_id

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        tokens = [self.token_to_id['<BOS>']]
        for c in text[:self.max_length - 2]:
            tokens.append(self.token_to_id.get(c, self.token_to_id['<UNK>']))
        tokens.append(self.token_to_id['<EOS>'])

        # Pad
        while len(tokens) < self.max_length:
            tokens.append(self.token_to_id['<PAD>'])

        return tokens[:self.max_length]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for i in ids:
            if i in [0, 2, 3]:  # PAD, BOS, EOS
                continue
            chars.append(self.id_to_token.get(i, '?'))
        return ''.join(chars)


# =============================================================================
# DATASET
# =============================================================================

class WorldRPMLabeledDataset(Dataset):
    """
    Dataset for World/RPM labeled training data.

    Expected JSONL format:
        {
            "input": "...",
            "target": "...",
            "world_label": "A",  # A, B, C, or D
            "rpm_label": "desire",  # desire, wisdom, or power
            "world_weights": {"A": 0.7, "B": 0.15, "C": 0.075, "D": 0.075},
            "rpm_weights": {"desire": 0.8, "wisdom": 0.1, "power": 0.1}
        }
    """

    WORLD_TO_IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    RPM_TO_IDX = {'desire': 0, 'wisdom': 1, 'power': 2}

    def __init__(self, data_path: str, tokenizer: TKSTokenizer):
        self.tokenizer = tokenizer
        self.entries = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.entries.append(entry)

        logger.info(f"Loaded {len(self.entries)} examples from {data_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Tokenize input and target
        input_text = entry.get('input', '')
        target_text = entry.get('target', '')
        full_text = f"{input_text} => {target_text}"

        input_ids = self.tokenizer.tokenize(full_text)
        targets = input_ids[1:] + [0]  # Next-token prediction

        # World label
        world_label = self.WORLD_TO_IDX.get(entry.get('world_label', 'A'), 0)

        # RPM label
        rpm_label = self.RPM_TO_IDX.get(entry.get('rpm_label', 'desire'), 0)

        # Soft labels (world weights)
        world_weights_dict = entry.get('world_weights', {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25})
        world_weights = torch.tensor([
            world_weights_dict.get('A', 0.25),
            world_weights_dict.get('B', 0.25),
            world_weights_dict.get('C', 0.25),
            world_weights_dict.get('D', 0.25),
        ], dtype=torch.float)

        # Soft labels (RPM weights)
        rpm_weights_dict = entry.get('rpm_weights', {'desire': 0.33, 'wisdom': 0.33, 'power': 0.34})
        rpm_weights = torch.tensor([
            rpm_weights_dict.get('desire', 0.33),
            rpm_weights_dict.get('wisdom', 0.33),
            rpm_weights_dict.get('power', 0.34),
        ], dtype=torch.float)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'world_label': torch.tensor(world_label, dtype=torch.long),
            'rpm_label': torch.tensor(rpm_label, dtype=torch.long),
            'world_weights': world_weights,
            'rpm_weights': rpm_weights,
        }


# =============================================================================
# TRAINER
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    batch_size: int = 16
    num_workers: int = 4

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Loss weights
    lambda_task: float = 1.0
    lambda_world: float = 0.5
    lambda_rpm: float = 0.5
    lambda_involution: float = 0.1
    lambda_spectral: float = 0.05

    # World/RPM loss hyperparameters
    world_margin: float = 0.5
    rpm_contrastive_margin: float = 0.2

    # Training
    epochs: int = 20
    grad_clip: float = 1.0
    early_stopping_patience: int = 5

    # Mixed precision
    use_amp: bool = True
    use_bf16: bool = False


class WorldRPMSupervisedTrainer:
    """
    Trainer for World/RPM supervised learning.

    Combines:
        - Task loss (next-token prediction)
        - World classification loss
        - RPM differentiation loss
        - TKS canonical losses (involution, spectral)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: TrainingConfig,
    ):
        self.config = config
        self.device = self._setup_device()

        # Model
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        total_steps = len(train_loader) * config.epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
            anneal_strategy='cos',
        )

        # Loss functions
        self.task_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        self.world_rpm_loss = WorldRPMSupervisedLoss(
            lambda_world=config.lambda_world,
            lambda_rpm=config.lambda_rpm,
            world_margin=config.world_margin,
            rpm_contrastive_margin=config.rpm_contrastive_margin,
        )

        # TKS loss for involution/spectral constraints
        tks_config = TKSLossConfig(
            lambda_task=0.0,  # Handled separately
            lambda_rpm=0.0,   # Handled by world_rpm_loss
            lambda_attractor=0.0,
            lambda_involution=config.lambda_involution,
            lambda_spectral=config.lambda_spectral,
            lambda_cascade=0.0,
        )
        self.tks_loss = TKSLoss(tks_config)

        # Mixed precision
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.amp_dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float16

        # Metrics
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'world_accuracies': [],
            'rpm_accuracies': [],
            'world_energy_gaps': [],
            'rpm_std': [],
        }

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            return device
        return torch.device('cpu')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'task': 0.0,
            'world': 0.0,
            'rpm': 0.0,
            'involution': 0.0,
        }
        epoch_metrics = {
            'world_accuracy': 0.0,
            'rpm_accuracy': 0.0,
            'world_energy_gap': 0.0,
            'rpm_std': 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            world_labels = batch['world_label'].to(self.device, non_blocking=True)
            rpm_labels = batch['rpm_label'].to(self.device, non_blocking=True)
            world_weights = batch['world_weights'].to(self.device, non_blocking=True)
            rpm_weights = batch['rpm_weights'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                # Forward pass with full trace
                output = self.model(input_ids, return_full_trace=True)
                logits = output['logits']
                trace = output['trace']

                # 1. Task loss
                task_loss = self.task_loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

                # 2. World/RPM supervision loss
                noetic_embedding = trace.get('embedding')
                dwp_scores = trace.get('dwp_scores')

                if noetic_embedding is not None and dwp_scores is not None:
                    world_rpm_losses = self.world_rpm_loss(
                        noetic_embedding=noetic_embedding,
                        dwp_scores=dwp_scores,
                        world_labels=world_labels,
                        rpm_labels=rpm_labels,
                        world_weights=world_weights,
                        rpm_weights=rpm_weights,
                    )
                    world_loss = world_rpm_losses['world_total']
                    rpm_loss = world_rpm_losses['rpm_total']
                else:
                    world_loss = torch.tensor(0.0, device=self.device)
                    rpm_loss = torch.tensor(0.0, device=self.device)
                    world_rpm_losses = {}

                # 3. TKS canonical losses (involution, spectral)
                tks_losses = self.tks_loss(
                    pipeline_output=output,
                    targets=targets,
                    pipeline=self.model,
                    compute_all=True,
                )
                involution_loss = tks_losses.get('involution', torch.tensor(0.0, device=self.device))

                # Combined loss
                total_loss = (
                    self.config.lambda_task * task_loss +
                    world_loss +
                    rpm_loss +
                    involution_loss
                )

            # Backward
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['task'] += task_loss.item()
            epoch_losses['world'] += world_loss.item() if isinstance(world_loss, torch.Tensor) else world_loss
            epoch_losses['rpm'] += rpm_loss.item() if isinstance(rpm_loss, torch.Tensor) else rpm_loss
            epoch_losses['involution'] += involution_loss.item() if isinstance(involution_loss, torch.Tensor) else involution_loss

            # Accumulate metrics
            if 'world_accuracy' in world_rpm_losses:
                epoch_metrics['world_accuracy'] += world_rpm_losses['world_accuracy'].item()
            if 'rpm_accuracy' in world_rpm_losses:
                epoch_metrics['rpm_accuracy'] += world_rpm_losses['rpm_accuracy'].item()
            if 'rpm_std_dwp' in world_rpm_losses:
                epoch_metrics['rpm_std'] += world_rpm_losses['rpm_std_dwp'].item()

            num_batches += 1

            # Log progress
            if batch_idx % 20 == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1} | Step {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {total_loss.item():.4f} (task={task_loss.item():.4f}, "
                    f"world={world_loss.item() if isinstance(world_loss, torch.Tensor) else world_loss:.4f}, "
                    f"rpm={rpm_loss.item() if isinstance(rpm_loss, torch.Tensor) else rpm_loss:.4f}) | "
                    f"LR: {lr:.2e}"
                )

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_batches, 1)

        return {**epoch_losses, **epoch_metrics}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
        if self.eval_loader is None:
            return {'eval_loss': 0.0}

        self.model.eval()
        total_loss = 0.0
        world_correct = 0
        rpm_correct = 0
        total_samples = 0

        world_energies_all = []
        rpm_stds_all = []

        for batch in self.eval_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            world_labels = batch['world_label'].to(self.device, non_blocking=True)
            rpm_labels = batch['rpm_label'].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                output = self.model(input_ids, return_full_trace=True)
                logits = output['logits']
                trace = output['trace']

                # Task loss
                task_loss = self.task_loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

                total_loss += task_loss.item()

                # World accuracy
                if trace.get('embedding') is not None:
                    emb = trace['embedding']
                    if emb.dim() == 3:
                        emb = emb.mean(dim=1)

                    # Compute world energies
                    world_energies = []
                    for w_idx in range(4):
                        w_slice = slice(w_idx * 10, (w_idx + 1) * 10)
                        w_energy = emb[..., w_slice].norm(dim=-1)
                        world_energies.append(w_energy)
                    world_energies = torch.stack(world_energies, dim=-1)

                    predicted_world = world_energies.argmax(dim=-1)
                    world_correct += (predicted_world == world_labels).sum().item()

                    # Track energy gap
                    world_energies_all.append(world_energies)

                # RPM accuracy
                if trace.get('dwp_scores') is not None:
                    dwp = trace['dwp_scores']
                    if dwp.dim() == 4:
                        dwp = dwp.mean(dim=(1, 2))  # Average over seq and foundations
                    elif dwp.dim() == 3:
                        dwp = dwp.mean(dim=1)

                    predicted_rpm = dwp.argmax(dim=-1)
                    rpm_correct += (predicted_rpm == rpm_labels).sum().item()

                    # Track RPM std
                    rpm_stds_all.append(dwp.std(dim=-1).mean())

                total_samples += input_ids.size(0)

        avg_loss = total_loss / max(len(self.eval_loader), 1)
        world_acc = world_correct / max(total_samples, 1)
        rpm_acc = rpm_correct / max(total_samples, 1)

        # Compute average energy gap
        if world_energies_all:
            all_energies = torch.cat(world_energies_all, dim=0)
            max_energy = all_energies.max(dim=-1).values
            second_max = all_energies.topk(2, dim=-1).values[:, 1]
            energy_gap = (max_energy - second_max).mean().item()
        else:
            energy_gap = 0.0

        # Compute average RPM std
        if rpm_stds_all:
            avg_rpm_std = torch.stack(rpm_stds_all).mean().item()
        else:
            avg_rpm_std = 0.0

        return {
            'eval_loss': avg_loss,
            'world_accuracy': world_acc,
            'rpm_accuracy': rpm_acc,
            'energy_gap': energy_gap,
            'rpm_std': avg_rpm_std,
        }

    def train(self, output_dir: str) -> Dict[str, Any]:
        """Full training loop."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        best_eval_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("WORLD/RPM SUPERVISED TRAINING")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp} ({self.amp_dtype})")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Loss weights: task={self.config.lambda_task}, world={self.config.lambda_world}, rpm={self.config.lambda_rpm}")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)
            self.metrics['train_losses'].append(train_metrics['total'])

            # Evaluate
            eval_metrics = self.evaluate()
            self.metrics['eval_losses'].append(eval_metrics['eval_loss'])
            self.metrics['world_accuracies'].append(eval_metrics['world_accuracy'])
            self.metrics['rpm_accuracies'].append(eval_metrics['rpm_accuracy'])
            self.metrics['world_energy_gaps'].append(eval_metrics['energy_gap'])
            self.metrics['rpm_std'].append(eval_metrics['rpm_std'])

            epoch_time = time.time() - epoch_start

            logger.info("-" * 70)
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Eval Loss: {eval_metrics['eval_loss']:.4f} | "
                f"World Acc: {eval_metrics['world_accuracy']:.2%} | "
                f"RPM Acc: {eval_metrics['rpm_accuracy']:.2%} | "
                f"Time: {epoch_time:.1f}s"
            )
            logger.info(
                f"  World Energy Gap: {eval_metrics['energy_gap']:.4f} | "
                f"RPM Std: {eval_metrics['rpm_std']:.4f}"
            )

            # Save best model
            if eval_metrics['eval_loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['eval_loss']
                patience_counter = 0
                self._save_checkpoint(output_path / "best_model.pt", epoch, eval_metrics)
                logger.info("  -> New best model saved!")
            else:
                patience_counter += 1
                if self.config.early_stopping_patience > 0:
                    logger.info(f"  -> No improvement ({patience_counter}/{self.config.early_stopping_patience})")

            # Early stopping
            if self.config.early_stopping_patience > 0 and patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(output_path / f"checkpoint_epoch_{epoch+1}.pt", epoch, eval_metrics)

        # Final save
        self._save_checkpoint(output_path / "final_model.pt", self.config.epochs - 1, eval_metrics)

        # Save metrics
        total_time = time.time() - start_time
        self.metrics['total_time_seconds'] = total_time

        with open(output_path / "training_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best eval loss: {best_eval_loss:.4f}")
        logger.info(f"Final world accuracy: {self.metrics['world_accuracies'][-1]:.2%}")
        logger.info(f"Final RPM accuracy: {self.metrics['rpm_accuracies'][-1]:.2%}")
        logger.info("=" * 70)

        return self.metrics

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        model_state = self.model.module.state_dict() \
            if hasattr(self.model, 'module') else self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
        }, path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="World/RPM Supervised Training")
    parser.add_argument('--data', required=True, help='Path to labeled training data JSONL')
    parser.add_argument('--output-dir', default='output/world_rpm_models', help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')

    # Loss weights
    parser.add_argument('--lambda-task', type=float, default=1.0, help='Task loss weight')
    parser.add_argument('--lambda-world', type=float, default=0.5, help='World classification loss weight')
    parser.add_argument('--lambda-rpm', type=float, default=0.5, help='RPM differentiation loss weight')
    parser.add_argument('--lambda-involution', type=float, default=0.1, help='Involution loss weight')

    # Margins
    parser.add_argument('--world-margin', type=float, default=0.5, help='World separation margin')
    parser.add_argument('--rpm-margin', type=float, default=0.2, help='RPM contrastive margin')

    # Training
    parser.add_argument('--early-stopping', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()

    if not TKS_AVAILABLE:
        logger.error("TKS components not available. Check imports.")
        sys.exit(1)

    # Create tokenizer
    tokenizer = TKSTokenizer(max_length=256)

    # Load dataset
    dataset = WorldRPMLabeledDataset(args.data, tokenizer)

    # Split
    eval_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    logger.info(f"Dataset: {len(dataset)} total, {train_size} train, {eval_size} eval")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Create model
    model = TKSLLMCorePipeline(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=args.hidden_dim,
        noetic_dim=TOTAL_DIM,
        num_scales=3,
        max_attractor_iter=10,
        contraction_factor=0.5,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from {args.resume}")

    # Training config
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        lambda_task=args.lambda_task,
        lambda_world=args.lambda_world,
        lambda_rpm=args.lambda_rpm,
        lambda_involution=args.lambda_involution,
        world_margin=args.world_margin,
        rpm_contrastive_margin=args.rpm_margin,
        early_stopping_patience=args.early_stopping,
        use_amp=not args.no_amp,
    )

    # Train
    trainer = WorldRPMSupervisedTrainer(model, train_loader, eval_loader, config)
    metrics = trainer.train(args.output_dir)

    return metrics


if __name__ == '__main__':
    main()

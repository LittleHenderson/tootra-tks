"""
TKS-LLM Trainer — Core Training Loop Implementation

This module provides the TKSTrainer class for training TKS-LLM models
with curriculum learning, multi-stage loss scheduling, and comprehensive logging.

Components:
    - TKSTrainer: Main trainer class
    - TrainingConfig: Configuration dataclass
    - TrainingState: State tracking during training
    - EvaluationResult: Evaluation metrics container

Canonical References:
    - TKS_LLM_Architecture_v1.0.md
    - TKS_LLM_Canonical_Validation_v1.0.md

Author: TKS-LLM Training-Agent
Date: 2025-12-12
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LambdaLR,
)

# Allow running as script: python training/trainer.py
if __name__ == "__main__" and __package__ is None:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "training"

from .losses import TKSLoss, TKSLossConfig, CurriculumLossScheduler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for TKS-LLM training."""

    # Model
    vocab_size: int = 1000
    hidden_dim: int = 128
    noetic_dim: int = 40

    # Training
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 100
    warmup_ratio: float = 0.1

    # Loss configuration
    loss_config: TKSLossConfig = field(default_factory=TKSLossConfig)
    use_curriculum: bool = False
    curriculum_stages: int = 5

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1000  # steps
    save_best: bool = True
    max_checkpoints: int = 5

    # Logging
    log_every: int = 100  # steps
    eval_every: int = 500  # steps
    log_dir: str = "logs"

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    mixed_precision: bool = False

    # Reproducibility
    seed: int = 42

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['loss_config'] = self.loss_config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        if 'loss_config' in d and isinstance(d['loss_config'], dict):
            d['loss_config'] = TKSLossConfig(**d['loss_config'])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load from YAML file."""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and resuming."""

    global_step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    best_metric: float = 0.0
    early_stopping_counter: int = 0
    curriculum_stage: int = 1

    # History
    loss_history: List[float] = field(default_factory=list)
    eval_history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        return cls(**d)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    loss: float = 0.0
    accuracy: float = 0.0

    # Per-component losses
    task_loss: float = 0.0
    rpm_loss: float = 0.0
    attractor_loss: float = 0.0
    involution_loss: float = 0.0
    spectral_loss: float = 0.0
    cascade_loss: float = 0.0

    # Per-task metrics
    element_accuracy: float = 0.0
    foundation_accuracy: float = 0.0
    rpm_mse: float = 0.0

    # Additional metrics
    samples_evaluated: int = 0
    eval_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class TKSTrainer:
    """
    Main trainer class for TKS-LLM models.

    Supports:
        - Multi-stage curriculum training
        - Loss component scheduling
        - Mixed precision training
        - Checkpointing and resuming
        - Comprehensive logging
        - Early stopping

    Usage:
        config = TrainingConfig(epochs=10, batch_size=16)
        trainer = TKSTrainer(model, config)
        trainer.train(train_dataset, eval_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[TKSLoss] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: TKSLLMCorePipeline or compatible model
            config: Training configuration
            optimizer: Optional custom optimizer
            scheduler: Optional custom scheduler
            loss_fn: Optional custom loss function
        """
        self.model = model
        self.config = config
        self.state = TrainingState()

        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Loss function
        self.loss_fn = loss_fn or TKSLoss(config.loss_config)

        # Curriculum scheduler (if enabled)
        self.curriculum_scheduler = None
        if config.use_curriculum:
            self.curriculum_scheduler = CurriculumLossScheduler(
                base_config=config.loss_config,
                warmup_steps=config.warmup_steps,
            )

        # Scheduler (created after knowing total steps)
        self._scheduler = scheduler

        # Mixed precision
        self.scaler = None
        if config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        self.callbacks: List[Callable] = []

        # Set seed
        self._set_seed(config.seed)

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(param_groups, lr=self.config.learning_rate)

    def _create_scheduler(self, num_training_steps: int) -> Any:
        """Create learning rate scheduler."""
        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0 and self.config.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "cosine":
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                main_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_training_steps - warmup_steps,
                )
                return SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                return CosineAnnealingLR(self.optimizer, T_max=num_training_steps)

        elif self.config.scheduler_type == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps,
            )

        else:  # constant
            return LambdaLR(self.optimizer, lambda _: 1.0)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from: Optional[str] = None,
    ) -> TrainingState:
        """
        Run training loop.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from: Optional checkpoint path to resume from

        Returns:
            Final training state
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Simple for compatibility
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

        # Calculate total steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.epochs

        # Create scheduler
        if self._scheduler is None:
            self._scheduler = self._create_scheduler(total_steps)

        logger.info(f"Starting training on {self.device}")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")

        # Training loop
        self.model.train()
        start_time = time.time()

        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(train_loader):
                # Update curriculum stage if enabled
                if self.curriculum_scheduler:
                    stage = self._compute_curriculum_stage(self.state.global_step, total_steps)
                    if stage != self.state.curriculum_stage:
                        self.state.curriculum_stage = stage
                        new_config = self.curriculum_scheduler.get_config_for_stage(stage)
                        self.loss_fn = TKSLoss(new_config)
                        logger.info(f"Curriculum stage {stage}: {new_config.to_dict()}")

                # Training step
                loss = self._training_step(batch, batch_idx)

                if loss is not None:
                    epoch_loss += loss
                    epoch_steps += 1
                    self.state.loss_history.append(loss)

                # Logging
                if self.state.global_step % self.config.log_every == 0:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    lr = self._scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.state.global_step} | "
                        f"Epoch {epoch+1}/{self.config.epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e}"
                    )

                # Evaluation
                if eval_dataset and self.state.global_step % self.config.eval_every == 0:
                    eval_result = self.evaluate(eval_dataset)
                    self.state.eval_history.append(eval_result.to_dict())
                    logger.info(f"Eval Loss: {eval_result.loss:.4f} | Acc: {eval_result.accuracy:.4f}")

                    # Check for best model
                    if self.config.save_best and eval_result.loss < self.state.best_loss:
                        self.state.best_loss = eval_result.loss
                        self.save_checkpoint("best.pt")

                    # Early stopping
                    if self.config.early_stopping:
                        if eval_result.loss < self.state.best_loss - self.config.early_stopping_min_delta:
                            self.state.early_stopping_counter = 0
                        else:
                            self.state.early_stopping_counter += 1

                        if self.state.early_stopping_counter >= self.config.early_stopping_patience:
                            logger.info("Early stopping triggered")
                            break

                    self.model.train()

                # Checkpointing
                if self.state.global_step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}.pt")

            # End of epoch
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}.pt")

            # Check early stopping
            if self.config.early_stopping and \
               self.state.early_stopping_counter >= self.config.early_stopping_patience:
                break

        # Training complete
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed/60:.1f} minutes")
        logger.info(f"Final loss: {self.state.loss_history[-1]:.4f}")

        # Save final checkpoint
        self.save_checkpoint("final.pt")

        return self.state

    def _training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[float]:
        """
        Execute single training step.

        Args:
            batch: Batch of training data
            batch_idx: Batch index within epoch

        Returns:
            Loss value or None if accumulating
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            outputs = self.model(
                tokens=batch["input_ids"],
                noetic_idx=batch.get("noetic_idx", 1),
                goal_state=batch.get("goal_state"),
                target_foundation=batch.get("target_foundation"),
                return_full_trace=True,
            )

            # Compute loss
            loss_dict = self.loss_fn(
                pipeline_output=outputs,
                targets=batch["targets"],
                pipeline=self.model,
                dwp_labels=batch.get("dwp_labels"),
                target_foundation=batch.get("target_foundation"),
                mask=batch.get("attention_mask"),
            )

            loss = loss_dict["total"]

            # Scale for gradient accumulation
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self._scheduler.step()
            self.optimizer.zero_grad()

            self.state.global_step += 1

            return loss.item() * self.config.gradient_accumulation_steps

        return None

    def _compute_curriculum_stage(self, step: int, total_steps: int) -> int:
        """Compute curriculum stage based on progress."""
        progress = step / total_steps
        stage = int(progress * self.config.curriculum_stages) + 1
        return min(stage, self.config.curriculum_stages)

    @torch.no_grad()
    def evaluate(self, eval_dataset: Dataset) -> EvaluationResult:
        """
        Evaluate model on dataset.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            EvaluationResult with metrics
        """
        self.model.eval()
        start_time = time.time()

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        loss_components = {
            "task": 0.0,
            "rpm": 0.0,
            "attractor": 0.0,
            "involution": 0.0,
            "spectral": 0.0,
            "cascade": 0.0,
        }

        for batch in eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = self.model(
                tokens=batch["input_ids"],
                return_full_trace=True,
            )

            loss_dict = self.loss_fn(
                pipeline_output=outputs,
                targets=batch["targets"],
                pipeline=self.model,
                dwp_labels=batch.get("dwp_labels"),
            )

            total_loss += loss_dict["total"].item()

            # Track component losses
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item()

            # Compute accuracy
            preds = outputs["logits"].argmax(dim=-1)
            targets = batch["targets"]
            correct = (preds == targets).sum().item()
            total_correct += correct
            total_samples += targets.numel()

        num_batches = len(eval_loader)

        result = EvaluationResult(
            loss=total_loss / num_batches,
            accuracy=total_correct / total_samples,
            task_loss=loss_components["task"] / num_batches,
            rpm_loss=loss_components["rpm"] / num_batches,
            attractor_loss=loss_components["attractor"] / num_batches,
            involution_loss=loss_components["involution"] / num_batches,
            spectral_loss=loss_components["spectral"] / num_batches,
            cascade_loss=loss_components["cascade"] / num_batches,
            element_accuracy=total_correct / total_samples,
            samples_evaluated=total_samples,
            eval_time_seconds=time.time() - start_time,
        )

        return result

    def save_checkpoint(self, filename: str) -> Path:
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict() if self._scheduler else None,
            "state": self.state.to_dict(),
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        if self.scaler:
            checkpoint["scaler"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        return checkpoint_path

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if checkpoint.get("scheduler") and self._scheduler:
            self._scheduler.load_state_dict(checkpoint["scheduler"])

        self.state = TrainingState.from_dict(checkpoint["state"])

        if checkpoint.get("scaler") and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler"])

        logger.info(f"Loaded checkpoint: {path}")
        logger.info(f"Resuming from step {self.state.global_step}, epoch {self.state.epoch}")

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        # Keep special checkpoints
        special = {"best.pt", "final.pt"}

        while len(checkpoints) > self.config.max_checkpoints:
            oldest = checkpoints.pop(0)
            if oldest.name not in special:
                oldest.unlink()
                logger.debug(f"Removed old checkpoint: {oldest}")

    def add_callback(self, callback: Callable) -> None:
        """Add a callback to be called during training."""
        self.callbacks.append(callback)


# =============================================================================
# STAGE-BASED TRAINER
# =============================================================================

class StagedTrainer:
    """
    Trainer for multi-stage curriculum training.

    Runs training in distinct stages with different loss configurations.
    Useful for pilot training and curriculum learning.
    """

    def __init__(
        self,
        model: nn.Module,
        base_config: TrainingConfig,
    ):
        self.model = model
        self.base_config = base_config
        self.stage_results: List[Dict[str, Any]] = []

    def train_stages(
        self,
        stages: List[Dict[str, Any]],
        eval_dataset: Optional[Dataset] = None,
    ) -> List[Dict[str, Any]]:
        """
        Train through multiple stages.

        Args:
            stages: List of stage configurations, each containing:
                - name: Stage name
                - dataset: Training dataset
                - epochs: Number of epochs
                - lr: Learning rate
                - loss_config: TKSLossConfig
            eval_dataset: Optional evaluation dataset

        Returns:
            List of results for each stage
        """
        for stage_idx, stage in enumerate(stages):
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE {stage_idx + 1}: {stage.get('name', 'unnamed')}")
            logger.info(f"{'='*60}")

            # Create stage-specific config
            stage_config = TrainingConfig(
                **{**asdict(self.base_config),
                   "epochs": stage.get("epochs", self.base_config.epochs),
                   "learning_rate": stage.get("lr", self.base_config.learning_rate),
                   "loss_config": stage.get("loss_config", self.base_config.loss_config),
                   "checkpoint_dir": f"{self.base_config.checkpoint_dir}/stage_{stage_idx + 1}",
                }
            )

            # Create trainer for this stage
            trainer = TKSTrainer(self.model, stage_config)

            # Run stage training
            start_time = time.time()
            state = trainer.train(stage["dataset"], eval_dataset)
            elapsed = time.time() - start_time

            # Collect results
            result = {
                "stage": stage_idx + 1,
                "name": stage.get("name", f"stage_{stage_idx + 1}"),
                "epochs": stage.get("epochs"),
                "final_loss": state.loss_history[-1] if state.loss_history else None,
                "initial_loss": state.loss_history[0] if state.loss_history else None,
                "elapsed_seconds": elapsed,
                "steps": state.global_step,
            }

            if result["initial_loss"] and result["final_loss"]:
                result["loss_reduction_pct"] = (
                    (result["initial_loss"] - result["final_loss"]) /
                    result["initial_loss"] * 100
                )

            self.stage_results.append(result)

            # Log stage summary
            logger.info(f"Stage {stage_idx + 1} complete:")
            logger.info(f"  Loss: {result.get('initial_loss', 0):.4f} -> {result.get('final_loss', 0):.4f}")
            if "loss_reduction_pct" in result:
                logger.info(f"  Reduction: {result['loss_reduction_pct']:.1f}%")
            logger.info(f"  Time: {elapsed/60:.1f} minutes")

        return self.stage_results

    def get_summary(self) -> str:
        """Get training summary as formatted string."""
        lines = ["Stage Training Summary", "=" * 40]

        for result in self.stage_results:
            status = "PASS" if result.get("loss_reduction_pct", 0) > 30 else "CHECK"
            lines.append(
                f"Stage {result['stage']} ({result['name']}): "
                f"{result.get('initial_loss', 0):.4f} -> {result.get('final_loss', 0):.4f} "
                f"({result.get('loss_reduction_pct', 0):+.1f}%) [{status}]"
            )

        return "\n".join(lines)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_device(prefer: str = "auto") -> torch.device:
    """
    Get the best available device for training.

    Args:
        prefer: Preference - 'auto', 'cuda', 'mps', or 'cpu'

    Returns:
        torch.device for training
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(prefer)


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Create optimizer for model.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')

    Returns:
        Configured optimizer
    """
    if optimizer_type.lower() == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    scheduler_type: str = "cosine",
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        total_steps: Total training steps
        warmup_steps: Warmup steps
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')

    Returns:
        Configured scheduler
    """
    if scheduler_type == "cosine":
        if warmup_steps > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            main = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
            return SequentialLR(optimizer, [warmup, main], milestones=[warmup_steps])
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    elif scheduler_type == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    elif scheduler_type == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        Checkpoint dictionary with metadata
    """
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return checkpoint


def create_trainer_from_config(
    model: nn.Module,
    config_path: str,
) -> TKSTrainer:
    """
    Create trainer from YAML configuration file.

    Args:
        model: Model to train
        config_path: Path to YAML config

    Returns:
        Configured TKSTrainer instance
    """
    config = TrainingConfig.from_yaml(config_path)
    return TKSTrainer(model, config)


def run_training(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs,
) -> TrainingState:
    """
    Convenience function to run training with minimal setup.

    Args:
        model: Model to train
        train_dataset: Training data
        eval_dataset: Optional eval data
        config: Optional configuration
        **kwargs: Override config values

    Returns:
        Final training state
    """
    if config is None:
        config = TrainingConfig(**kwargs)
    else:
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    trainer = TKSTrainer(model, config)
    return trainer.train(train_dataset, eval_dataset)


# =============================================================================
# TESTING
# =============================================================================

def test_trainer():
    """Test trainer with dummy data."""
    import sys
    sys.path.insert(0, '..')

    try:
        from tks_llm_core_v2 import TKSLLMCorePipeline
    except ImportError:
        print("Cannot import TKSLLMCorePipeline, skipping test")
        return

    # Create dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, size: int = 100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 100, (8,)),
                "targets": torch.randint(0, 100, (8,)),
            }

    # Create model and trainer
    model = TKSLLMCorePipeline(vocab_size=100, hidden_dim=64)
    config = TrainingConfig(
        vocab_size=100,
        hidden_dim=64,
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        log_every=10,
        eval_every=50,
        checkpoint_every=100,
    )

    trainer = TKSTrainer(model, config)

    # Run training
    train_data = DummyDataset(100)
    eval_data = DummyDataset(20)

    state = trainer.train(train_data, eval_data)

    print("\nTraining test complete!")
    print(f"Final step: {state.global_step}")
    print(f"Final loss: {state.loss_history[-1]:.4f}")


if __name__ == "__main__":
    test_trainer()

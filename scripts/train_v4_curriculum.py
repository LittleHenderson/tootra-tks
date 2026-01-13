#!/usr/bin/env python3
"""
TKS-LLM v4 Curriculum Training Script

Trains tks_llm_core_v4 with integrated strg-llm-stk components using
curriculum learning across 4 phases:

Phase 1: Operator Semantics
    - Train compositional layer on operator semantics
    - Foundation: +/-/*/ symbol grounding

Phase 2: Equation Corpus
    - Contrastive learning on equation interpretations
    - Build operator core reasoning capability

Phase 3: RPM Regulators
    - Train RPM gating mechanism
    - FM (Female/Male), ACBE (Cause-Effect), MVR (Activation)

Phase 4: DPS Layer
    - Train novelty detection and depth unlock
    - NW = V x I x (1-S) computation

Usage:
    python scripts/train_v4_curriculum.py --full-curriculum
    python scripts/train_v4_curriculum.py --phase 1
    python scripts/train_v4_curriculum.py --phase 3 --resume output/rpm/best.pt

Author: ML Agent
Date: 2026-01-05
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import training components
from training.trainer import TKSTrainer, TrainingConfig, TrainingState, StagedTrainer
from training.losses import TKSLoss, TKSLossConfig, CurriculumLossScheduler
from training.strg_losses import DPSTrainingLoss, STRGLossConfig

# Import individual training scripts
from scripts.train_operator_core import OperatorDataset, OperatorTrainingConfig
from scripts.train_rpm_regulator import RPMDataset, RPMTrainingConfig
from scripts.train_dps_layer import DPSDataset, DPSTrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CurriculumConfig:
    """Configuration for curriculum training."""
    # Output
    output_dir: str = "output/v4_curriculum"
    experiment_name: str = "v4_strg_integration"

    # Model
    vocab_size: int = 10000
    hidden_dim: int = 256
    noetic_dim: int = 40
    num_layers: int = 4
    num_heads: int = 8

    # Global training settings
    total_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Curriculum settings
    phase1_epochs: int = 5    # Operator semantics
    phase2_epochs: int = 10   # Equation corpus
    phase3_epochs: int = 10   # RPM regulators
    phase4_epochs: int = 5    # DPS layer

    # Loss weight schedule
    phase1_weights: Dict[str, float] = field(default_factory=lambda: {
        'task': 1.0, 'rpm': 0.0, 'dps': 0.0, 'operator': 1.0
    })
    phase2_weights: Dict[str, float] = field(default_factory=lambda: {
        'task': 1.0, 'rpm': 0.0, 'dps': 0.0, 'operator': 0.5, 'contrastive': 0.5
    })
    phase3_weights: Dict[str, float] = field(default_factory=lambda: {
        'task': 1.0, 'rpm': 1.0, 'dps': 0.0, 'operator': 0.3
    })
    phase4_weights: Dict[str, float] = field(default_factory=lambda: {
        'task': 1.0, 'rpm': 0.5, 'dps': 1.0, 'operator': 0.2
    })

    # Data paths
    operator_data: List[str] = field(default_factory=lambda: [
        "strg-llm-stk/tks_operator_training_pack_2000.jsonl",
        "strg-llm-stk/tks_operator_training_pack_250.jsonl",
    ])
    rpm_data: str = "strg-llm-stk/tks_rpm_fm_acbe_mvr_training_pack_800.jsonl"
    dps_data: str = "strg-llm-stk/tks_consciousness_unlock_HS_enabled_pack_1000.jsonl"
    equations_data: str = "data/equations_600k.jsonl"

    # Checkpointing
    save_every_phase: bool = True
    save_best_only: bool = False
    log_every: int = 100
    eval_every: int = 500

    # Device
    device: str = "auto"

    # Feature flags
    use_dps_gating: bool = True
    use_governance_rails: bool = True
    backward_compat: bool = True


# =============================================================================
# CURRICULUM TRAINER
# =============================================================================

class CurriculumTrainer:
    """
    Multi-phase curriculum trainer for TKS-LLM v4.

    Implements the training order from training_manifest.json:
    1. operator_semantics
    2. equations_large
    3. rpm_regulator
    4. consciousness_unlock (DPS)
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        logger.info(f"Using device: {self.device}")

        # Track training state
        self.current_phase = 0
        self.phase_results: List[Dict[str, Any]] = []
        self.global_step = 0

        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)

    def _create_model(self) -> nn.Module:
        """Create TKS-LLM v4 model with strg-llm-stk integration."""
        try:
            from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

            model_config = TKSNoeticLMConfig(
                vocab_size=self.config.vocab_size,
                hidden_dim=self.config.hidden_dim,
                noetic_dim=self.config.noetic_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                use_dps_gating=self.config.use_dps_gating,
                use_governance_rails=self.config.use_governance_rails,
            )
            model = TKSNoeticLM(model_config)
            logger.info("Loaded TKSNoeticLM from tks_llm_core_v4")
            return model

        except ImportError:
            logger.warning("TKSNoeticLM not found, using fallback model")
            return SimpleTKSModel(
                vocab_size=self.config.vocab_size,
                hidden_dim=self.config.hidden_dim,
                noetic_dim=self.config.noetic_dim,
            )

    def train_full_curriculum(self) -> Dict[str, Any]:
        """Run full curriculum training across all phases."""
        logger.info("=" * 60)
        logger.info("TKS-LLM v4 CURRICULUM TRAINING")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Phase 1: Operator Semantics
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: OPERATOR SEMANTICS")
        logger.info("=" * 60)
        phase1_result = self.train_phase(1)
        self.phase_results.append(phase1_result)

        # Phase 2: Equation Corpus
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: EQUATION CORPUS (Contrastive)")
        logger.info("=" * 60)
        phase2_result = self.train_phase(2)
        self.phase_results.append(phase2_result)

        # Phase 3: RPM Regulators
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: RPM REGULATORS")
        logger.info("=" * 60)
        phase3_result = self.train_phase(3)
        self.phase_results.append(phase3_result)

        # Phase 4: DPS Layer
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: DPS LAYER (Consciousness Unlock)")
        logger.info("=" * 60)
        phase4_result = self.train_phase(4)
        self.phase_results.append(phase4_result)

        # Final summary
        elapsed = datetime.now() - start_time
        summary = {
            "total_time": str(elapsed),
            "phases": self.phase_results,
            "final_checkpoint": str(self.output_dir / "final.pt"),
        }

        # Save summary
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save final model
        self._save_checkpoint("final.pt")

        logger.info("\n" + "=" * 60)
        logger.info("CURRICULUM TRAINING COMPLETE")
        logger.info(f"Total time: {elapsed}")
        logger.info("=" * 60)

        return summary

    def train_phase(self, phase: int, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Train a single curriculum phase."""
        self.current_phase = phase

        # Load checkpoint if resuming
        if resume_from:
            self._load_checkpoint(resume_from)

        # Get phase configuration
        if phase == 1:
            return self._train_operator_phase()
        elif phase == 2:
            return self._train_equation_phase()
        elif phase == 3:
            return self._train_rpm_phase()
        elif phase == 4:
            return self._train_dps_phase()
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def _train_operator_phase(self) -> Dict[str, Any]:
        """Phase 1: Train operator semantics."""
        phase_dir = self.output_dir / "phase1_operator"
        phase_dir.mkdir(exist_ok=True)

        # Load dataset
        dataset = OperatorDataset(
            self.config.operator_data,
            element_dim=self.config.noetic_dim,
        )

        # Split
        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create training config
        train_config = TrainingConfig(
            epochs=self.config.phase1_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            checkpoint_dir=str(phase_dir),
            log_every=self.config.log_every,
            eval_every=self.config.eval_every,
            loss_config=TKSLossConfig(
                lambda_task=self.config.phase1_weights['task'],
                lambda_rpm=self.config.phase1_weights.get('rpm', 0.0),
            ),
        )

        # Create trainer
        trainer = TKSTrainer(self.model, train_config)

        # Train
        state = trainer.train(train_dataset, val_dataset)

        # Save phase checkpoint
        if self.config.save_every_phase:
            self._save_checkpoint(f"phase1_complete.pt")

        return {
            "phase": 1,
            "name": "operator_semantics",
            "epochs": self.config.phase1_epochs,
            "final_loss": state.loss_history[-1] if state.loss_history else None,
            "steps": state.global_step,
        }

    def _train_equation_phase(self) -> Dict[str, Any]:
        """Phase 2: Contrastive learning on equations."""
        phase_dir = self.output_dir / "phase2_equations"
        phase_dir.mkdir(exist_ok=True)

        # For now, use operator dataset as placeholder
        # In full implementation, use equations_600k.jsonl
        dataset = OperatorDataset(
            self.config.operator_data,
            element_dim=self.config.noetic_dim,
        )

        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_config = TrainingConfig(
            epochs=self.config.phase2_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate * 0.5,  # Lower LR
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            checkpoint_dir=str(phase_dir),
            log_every=self.config.log_every,
            eval_every=self.config.eval_every,
            loss_config=TKSLossConfig(
                lambda_task=self.config.phase2_weights['task'],
            ),
        )

        trainer = TKSTrainer(self.model, train_config)
        state = trainer.train(train_dataset, val_dataset)

        if self.config.save_every_phase:
            self._save_checkpoint(f"phase2_complete.pt")

        return {
            "phase": 2,
            "name": "equation_corpus",
            "epochs": self.config.phase2_epochs,
            "final_loss": state.loss_history[-1] if state.loss_history else None,
            "steps": state.global_step,
        }

    def _train_rpm_phase(self) -> Dict[str, Any]:
        """Phase 3: Train RPM regulators."""
        phase_dir = self.output_dir / "phase3_rpm"
        phase_dir.mkdir(exist_ok=True)

        dataset = RPMDataset(
            self.config.rpm_data,
            noetic_dim=self.config.noetic_dim,
        )

        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_config = TrainingConfig(
            epochs=self.config.phase3_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            checkpoint_dir=str(phase_dir),
            log_every=self.config.log_every,
            eval_every=self.config.eval_every,
            loss_config=TKSLossConfig(
                lambda_task=self.config.phase3_weights['task'],
                lambda_rpm=self.config.phase3_weights['rpm'],
            ),
        )

        trainer = TKSTrainer(self.model, train_config)
        state = trainer.train(train_dataset, val_dataset)

        if self.config.save_every_phase:
            self._save_checkpoint(f"phase3_complete.pt")

        return {
            "phase": 3,
            "name": "rpm_regulators",
            "epochs": self.config.phase3_epochs,
            "final_loss": state.loss_history[-1] if state.loss_history else None,
            "steps": state.global_step,
        }

    def _train_dps_phase(self) -> Dict[str, Any]:
        """Phase 4: Train DPS layer."""
        phase_dir = self.output_dir / "phase4_dps"
        phase_dir.mkdir(exist_ok=True)

        dataset = DPSDataset(
            self.config.dps_data,
            noetic_dim=self.config.noetic_dim,
        )

        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_config = TrainingConfig(
            epochs=self.config.phase4_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate * 0.5,  # Lower LR for fine-tuning
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            checkpoint_dir=str(phase_dir),
            log_every=self.config.log_every,
            eval_every=self.config.eval_every,
            loss_config=TKSLossConfig(
                lambda_task=self.config.phase4_weights['task'],
                lambda_rpm=self.config.phase4_weights['rpm'],
            ),
        )

        trainer = TKSTrainer(self.model, train_config)
        state = trainer.train(train_dataset, val_dataset)

        if self.config.save_every_phase:
            self._save_checkpoint(f"phase4_complete.pt")

        return {
            "phase": 4,
            "name": "dps_layer",
            "epochs": self.config.phase4_epochs,
            "final_loss": state.loss_history[-1] if state.loss_history else None,
            "steps": state.global_step,
        }

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'phase': self.current_phase,
            'global_step': self.global_step,
            'config': self.config.__dict__,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_phase = checkpoint.get('phase', 0)
        self.global_step = checkpoint.get('global_step', 0)
        logger.info(f"Loaded checkpoint from {path}")


class SimpleTKSModel(nn.Module):
    """Fallback model if TKSNoeticLM unavailable."""

    def __init__(self, vocab_size: int, hidden_dim: int, noetic_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.noetic_proj = nn.Linear(hidden_dim, noetic_dim)
        self.output_proj = nn.Linear(noetic_dim, vocab_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ),
            num_layers=2,
        )

    def forward(self, tokens, **kwargs):
        x = self.embedding(tokens)
        x = self.transformer(x)
        noetic = self.noetic_proj(x)
        logits = self.output_proj(noetic)
        return {
            'logits': logits,
            'noetic_output': noetic,
            'gated_output': noetic,
            'trace': {},
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TKS-LLM v4 Curriculum Training")
    parser.add_argument("--full-curriculum", action="store_true",
                        help="Run full 4-phase curriculum")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="Train specific phase only")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="output/v4_curriculum",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Create config
    config = CurriculumConfig(
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Create trainer
    trainer = CurriculumTrainer(config)

    if args.full_curriculum:
        # Run full curriculum
        summary = trainer.train_full_curriculum()
        logger.info(f"\nTraining Summary:\n{json.dumps(summary, indent=2)}")
    elif args.phase:
        # Train specific phase
        result = trainer.train_phase(args.phase, resume_from=args.resume)
        logger.info(f"\nPhase {args.phase} Result:\n{json.dumps(result, indent=2)}")
    else:
        parser.print_help()
        print("\nError: Specify --full-curriculum or --phase")
        sys.exit(1)


if __name__ == "__main__":
    main()

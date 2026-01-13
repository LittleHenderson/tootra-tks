#!/usr/bin/env python3
"""
Run Training — Main entry point for TKS-LLM v4 training

This is the primary training script that orchestrates:
1. Dataset mixing and preparation
2. Curriculum-based training
3. Checkpoint management
4. Evaluation during training

Training Modes:
- full: Complete curriculum training (4 phases)
- operator: Operator semantics only
- rpm: RPM regulator only
- dps: DPS layer only
- finetune: Fine-tune from checkpoint

Usage:
    python scripts/train/run_train.py --mode full --epochs 30
    python scripts/train/run_train.py --mode operator --epochs 10
    python scripts/train/run_train.py --mode finetune --checkpoint output/best.pt
    python scripts/train/run_train.py --quick-test  # Fast test run

Author: ML Agent
Date: 2026-01-05
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
class RunConfig:
    """Configuration for training run."""
    # Mode
    mode: str = "full"  # full, operator, rpm, dps, finetune

    # Model
    vocab_size: int = 10000
    hidden_dim: int = 256
    noetic_dim: int = 40
    num_layers: int = 4
    num_heads: int = 8

    # Training
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Data
    data_dir: str = "data/mixed"
    use_mixed_data: bool = True

    # Output
    output_dir: str = "output"
    experiment_name: str = None

    # Checkpoint
    checkpoint: str = None
    save_every: int = 500
    eval_every: int = 500

    # Device
    device: str = "auto"

    # Features
    use_dps_gating: bool = True
    use_governance_rails: bool = True

    # Quick test mode
    quick_test: bool = False

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"tks_v4_{self.mode}_{timestamp}"

        if self.quick_test:
            self.epochs = 1
            self.batch_size = 4
            self.save_every = 10
            self.eval_every = 10


# =============================================================================
# TRAINING RUNNER
# =============================================================================

class TrainingRunner:
    """Main training orchestrator."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

        # Save config
        self._save_config()

    def _save_config(self) -> None:
        """Save run configuration."""
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Saved config to {config_path}")

    def run(self) -> Dict[str, Any]:
        """Run training based on mode."""
        logger.info("=" * 60)
        logger.info(f"TKS-LLM v4 TRAINING: {self.config.mode.upper()} MODE")
        logger.info("=" * 60)

        start_time = time.time()

        if self.config.mode == "full":
            result = self._run_full_curriculum()
        elif self.config.mode == "operator":
            result = self._run_operator_training()
        elif self.config.mode == "rpm":
            result = self._run_rpm_training()
        elif self.config.mode == "dps":
            result = self._run_dps_training()
        elif self.config.mode == "finetune":
            result = self._run_finetuning()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        elapsed = time.time() - start_time
        result['total_time_seconds'] = elapsed
        result['total_time_formatted'] = f"{elapsed/60:.1f} minutes"

        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {result['total_time_formatted']}")
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 60)

        return result

    def _prepare_data(self) -> None:
        """Prepare mixed training data if needed."""
        if self.config.use_mixed_data:
            from scripts.train.mix_datasets import DatasetMixer, MixConfig
            mix_config = MixConfig.auto_detect()
            mix_config.output_dir = str(PROJECT_ROOT / self.config.data_dir)
            mixer = DatasetMixer(mix_config)
            mixer.mix()
            mixer.split_and_save()

    def _run_full_curriculum(self) -> Dict[str, Any]:
        """Run full 4-phase curriculum training."""
        from scripts.train_v4_curriculum import CurriculumTrainer, CurriculumConfig

        curriculum_config = CurriculumConfig(
            output_dir=str(self.output_dir),
            experiment_name=self.config.experiment_name,
            vocab_size=self.config.vocab_size,
            hidden_dim=self.config.hidden_dim,
            noetic_dim=self.config.noetic_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            total_epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
            use_dps_gating=self.config.use_dps_gating,
            use_governance_rails=self.config.use_governance_rails,
        )

        if self.config.quick_test:
            curriculum_config.phase1_epochs = 1
            curriculum_config.phase2_epochs = 1
            curriculum_config.phase3_epochs = 1
            curriculum_config.phase4_epochs = 1

        trainer = CurriculumTrainer(curriculum_config)
        return trainer.train_full_curriculum()

    def _run_operator_training(self) -> Dict[str, Any]:
        """Run operator semantics training only."""
        from scripts.train_operator_core import train_operator_core, OperatorTrainingConfig

        op_config = OperatorTrainingConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
            checkpoint_dir=str(self.output_dir / "operator"),
        )

        if self.config.quick_test:
            op_config.epochs = 1
            op_config.save_every = 10
            op_config.log_every = 5

        train_operator_core(op_config)

        return {
            "mode": "operator",
            "epochs": op_config.epochs,
            "checkpoint_dir": str(self.output_dir / "operator"),
        }

    def _run_rpm_training(self) -> Dict[str, Any]:
        """Run RPM regulator training only."""
        from scripts.train_rpm_regulator import train_rpm_regulator, RPMTrainingConfig

        rpm_config = RPMTrainingConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
            checkpoint_dir=str(self.output_dir / "rpm"),
        )

        if self.config.quick_test:
            rpm_config.epochs = 1
            rpm_config.save_every = 10
            rpm_config.log_every = 5

        train_rpm_regulator(rpm_config)

        return {
            "mode": "rpm",
            "epochs": rpm_config.epochs,
            "checkpoint_dir": str(self.output_dir / "rpm"),
        }

    def _run_dps_training(self) -> Dict[str, Any]:
        """Run DPS layer training only."""
        from scripts.train_dps_layer import train_dps_layer, DPSTrainingConfig

        dps_config = DPSTrainingConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
            checkpoint_dir=str(self.output_dir / "dps"),
        )

        if self.config.quick_test:
            dps_config.epochs = 1
            dps_config.save_every = 10
            dps_config.log_every = 5

        train_dps_layer(dps_config)

        return {
            "mode": "dps",
            "epochs": dps_config.epochs,
            "checkpoint_dir": str(self.output_dir / "dps"),
        }

    def _run_finetuning(self) -> Dict[str, Any]:
        """Fine-tune from checkpoint."""
        if not self.config.checkpoint:
            raise ValueError("Checkpoint path required for finetune mode")

        from training.trainer import TKSTrainer, TrainingConfig

        # Load model from checkpoint
        checkpoint_path = Path(self.config.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        train_config = TrainingConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate * 0.1,  # Lower LR for fine-tuning
            weight_decay=self.config.weight_decay,
            checkpoint_dir=str(self.output_dir),
        )

        # Create model
        try:
            from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
            model_config = TKSNoeticLMConfig(
                vocab_size=self.config.vocab_size,
                hidden_dim=self.config.hidden_dim,
                noetic_dim=self.config.noetic_dim,
            )
            model = TKSNoeticLM(model_config)
        except ImportError:
            from scripts.train_v4_curriculum import SimpleTKSModel
            model = SimpleTKSModel(
                self.config.vocab_size,
                self.config.hidden_dim,
                self.config.noetic_dim,
            )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

        trainer = TKSTrainer(model, train_config)

        # Load mixed data
        from scripts.train_operator_core import OperatorDataset
        dataset = OperatorDataset(
            ["strg-llm-stk/tks_operator_training_pack_2000.jsonl"],
            element_dim=self.config.noetic_dim,
        )

        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        state = trainer.train(train_dataset, val_dataset)

        return {
            "mode": "finetune",
            "source_checkpoint": str(checkpoint_path),
            "epochs": self.config.epochs,
            "final_loss": state.loss_history[-1] if state.loss_history else None,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TKS-LLM v4 Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_train.py --mode full --epochs 30
  python run_train.py --mode operator --epochs 10
  python run_train.py --mode finetune --checkpoint output/best.pt
  python run_train.py --quick-test
        """
    )

    # Mode
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "operator", "rpm", "dps", "finetune"],
                        help="Training mode")

    # Training params
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Model params
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint for finetune mode")

    # Output
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name")

    # Device
    parser.add_argument("--device", type=str, default="auto")

    # Quick test
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test run with minimal epochs")

    # Features
    parser.add_argument("--no-dps", action="store_true",
                        help="Disable DPS gating")
    parser.add_argument("--no-governance", action="store_true",
                        help="Disable governance rails")

    args = parser.parse_args()

    # Create config
    config = RunConfig(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        experiment_name=args.name,
        device=args.device,
        quick_test=args.quick_test,
        use_dps_gating=not args.no_dps,
        use_governance_rails=not args.no_governance,
    )

    # Run training
    runner = TrainingRunner(config)
    result = runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

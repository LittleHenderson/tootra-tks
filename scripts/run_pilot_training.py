#!/usr/bin/env python3
"""
TKS-LLM Pilot Training Script

Runs the 4-stage pilot training experiment:
    Stage 1: Element Prediction (200 ex, 10 epochs, L_task only)
    Stage 2: Noetic Composition (200 ex, 10 epochs, +L_involution)
    Stage 3: RPM Prediction (200 ex, 10 epochs, +L_rpm)
    Stage 4: Full Pipeline (400 ex, 20 epochs, all losses)

Expected runtime: ~45-60 minutes on CPU, ~15-20 minutes on GPU

Usage:
    python scripts/run_pilot_training.py [options]

Options:
    --data_dir      Directory containing pilot data (default: data/pilot)
    --output_dir    Directory for checkpoints/logs (default: outputs/pilot)
    --device        Device to use: cpu, cuda, mps (default: auto-detect)
    --batch_size    Batch size (default: 16)
    --seed          Random seed (default: 42)
    --resume        Resume from checkpoint
    --stage         Run specific stage only (1-4)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# TKS-LLM imports
try:
    from tks_llm_core_v2 import TKSModel, TKSConfig
    from training import (
        TKSLoss,
        TKSLossConfig,
        TKSTrainer,
        TrainingConfig,
        StagedTrainer,
        CurriculumLossScheduler,
        PilotDataset,
        tks_collate_fn,
        get_device,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Pilot model configuration (small for fast iteration)
PILOT_MODEL_CONFIG = {
    "vocab_size": 1000,
    "hidden_dim": 128,
    "noetic_dim": 40,  # 10 noetics x 4 worlds
    "n_layers": 4,
    "n_heads": 4,
    "ff_dim": 512,
    "max_seq_len": 256,
    "dropout": 0.1
}

# Stage configurations
STAGE_CONFIGS = {
    1: {
        "name": "Element Prediction",
        "data_file": "stage1_elements.jsonl",
        "epochs": 10,
        "loss_weights": {
            "task": 1.0,
            "rpm": 0.0,
            "attractor": 0.0,
            "involution": 0.0,
            "spectral": 0.0,
            "cascade": 0.0
        },
        "learning_rate": 1e-4
    },
    2: {
        "name": "Noetic Composition",
        "data_file": "stage2_composition.jsonl",
        "epochs": 10,
        "loss_weights": {
            "task": 1.0,
            "rpm": 0.0,
            "attractor": 0.0,
            "involution": 0.5,
            "spectral": 0.0,
            "cascade": 0.0
        },
        "learning_rate": 1e-4
    },
    3: {
        "name": "RPM Prediction",
        "data_file": "stage3_rpm.jsonl",
        "epochs": 10,
        "loss_weights": {
            "task": 1.0,
            "rpm": 0.5,
            "attractor": 0.3,
            "involution": 0.5,
            "spectral": 0.0,
            "cascade": 0.0
        },
        "learning_rate": 5e-5
    },
    4: {
        "name": "Full Pipeline",
        "data_file": "stage4_full_pipeline.jsonl",
        "epochs": 20,
        "loss_weights": {
            "task": 1.0,
            "rpm": 0.5,
            "attractor": 0.3,
            "involution": 0.5,
            "spectral": 0.2,
            "cascade": 0.1
        },
        "learning_rate": 3e-5
    }
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def setup_logging(output_dir: Path):
    """Setup logging directory and files."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pilot_training_{timestamp}.log"

    return log_file


def log_message(log_file: Path, message: str, also_print: bool = True):
    """Log message to file and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(full_message + '\n')

    if also_print:
        print(full_message)


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create TKS model with pilot configuration."""
    # Use TKSConfig if available, otherwise create simple model
    try:
        model_config = TKSConfig(**config)
        model = TKSModel(model_config)
    except Exception as e:
        print(f"Warning: Could not create TKSModel ({e}), using simplified model")
        model = SimplePilotModel(config)

    return model.to(device)


class SimplePilotModel(nn.Module):
    """Simplified model for pilot testing if TKSModel is not available."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        hidden_dim = config.get("hidden_dim", 128)
        vocab_size = config.get("vocab_size", 1000)
        noetic_dim = config.get("noetic_dim", 40)

        # Simple transformer-like architecture
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=config.get("n_heads", 4),
                dim_feedforward=config.get("ff_dim", 512),
                dropout=config.get("dropout", 0.1),
                batch_first=True
            ),
            num_layers=config.get("n_layers", 4)
        )

        # Output heads
        self.element_head = nn.Linear(hidden_dim, noetic_dim)
        self.rpm_head = nn.Linear(hidden_dim, 3)  # desire, wisdom, power
        self.noetic_head = nn.Linear(hidden_dim, noetic_dim)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)

        if attention_mask is not None:
            # Create causal mask
            seq_len = x.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
                diagonal=1
            )
            x = self.encoder(x, mask=causal_mask)
        else:
            x = self.encoder(x)

        # Pool to get sequence representation
        pooled = x.mean(dim=1)

        return {
            "logits": self.element_head(x),
            "element_logits": self.element_head(pooled),
            "rpm_logits": torch.sigmoid(self.rpm_head(pooled)),
            "noetic_activations": torch.sigmoid(self.noetic_head(pooled)),
            "hidden_states": x
        }


def load_stage_data(data_dir: Path, stage_num: int, batch_size: int) -> DataLoader:
    """Load data for a specific stage."""
    stage_config = STAGE_CONFIGS[stage_num]
    data_file = data_dir / stage_config["data_file"]

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    dataset = PilotDataset(str(data_file))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tks_collate_fn,
        drop_last=False
    )


def create_loss_fn(stage_num: int) -> TKSLoss:
    """Create loss function for a specific stage."""
    stage_config = STAGE_CONFIGS[stage_num]
    weights = stage_config["loss_weights"]

    loss_config = TKSLossConfig(
        task_weight=weights["task"],
        rpm_weight=weights["rpm"],
        attractor_weight=weights["attractor"],
        involution_weight=weights["involution"],
        spectral_weight=weights["spectral"],
        cascade_weight=weights["cascade"]
    )

    return TKSLoss(loss_config)


# ==============================================================================
# STAGE TRAINING
# ==============================================================================

def train_stage(
    stage_num: int,
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: TKSLoss,
    output_dir: Path,
    log_file: Path,
    device: torch.device,
    resume_from: str = None
) -> dict:
    """Train a single stage."""
    stage_config = STAGE_CONFIGS[stage_num]

    log_message(log_file, f"\n{'='*60}")
    log_message(log_file, f"STAGE {stage_num}: {stage_config['name']}")
    log_message(log_file, f"{'='*60}")
    log_message(log_file, f"Epochs: {stage_config['epochs']}")
    log_message(log_file, f"Learning rate: {stage_config['learning_rate']}")
    log_message(log_file, f"Loss weights: {stage_config['loss_weights']}")
    log_message(log_file, f"Data samples: {len(data_loader.dataset)}")

    # Create trainer config
    trainer_config = TrainingConfig(
        epochs=stage_config["epochs"],
        learning_rate=stage_config["learning_rate"],
        batch_size=data_loader.batch_size,
        checkpoint_dir=str(output_dir / f"stage{stage_num}_checkpoints"),
        log_dir=str(output_dir / "logs"),
        save_every_n_epochs=5,
        eval_every_n_steps=50,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        warmup_steps=100,
        scheduler_type="cosine",
        early_stopping_patience=5
    )

    # Create trainer
    trainer = TKSTrainer(
        model=model,
        config=trainer_config,
        loss_fn=loss_fn
    )

    # Train
    start_time = time.time()
    training_state = trainer.train(
        train_dataset=data_loader.dataset,
        resume_from=resume_from
    )
    elapsed = time.time() - start_time

    log_message(log_file, f"\nStage {stage_num} completed in {elapsed:.1f}s")
    log_message(log_file, f"Final loss: {training_state.best_loss:.4f}")
    log_message(log_file, f"Total steps: {training_state.global_step}")

    return {
        "stage": stage_num,
        "name": stage_config["name"],
        "epochs_completed": training_state.current_epoch,
        "final_loss": training_state.best_loss,
        "total_steps": training_state.global_step,
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(training_state.checkpoint_path) if training_state.checkpoint_path else None
    }


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def run_pilot_training(args):
    """Run the full pilot training experiment."""
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = setup_logging(output_dir)

    log_message(log_file, "TKS-LLM Pilot Training")
    log_message(log_file, "=" * 60)

    # Device setup
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    log_message(log_file, f"Device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    log_message(log_file, f"Random seed: {args.seed}")

    # Verify data exists
    if not data_dir.exists():
        log_message(log_file, f"Data directory not found: {data_dir}")
        log_message(log_file, "Run 'python scripts/generate_pilot_data.py' first")
        sys.exit(1)

    # Create model
    log_message(log_file, "\nCreating model...")
    model = create_model(PILOT_MODEL_CONFIG, device)
    param_count = sum(p.numel() for p in model.parameters())
    log_message(log_file, f"Model parameters: {param_count:,}")

    # Determine stages to run
    if args.stage:
        stages_to_run = [args.stage]
    else:
        stages_to_run = [1, 2, 3, 4]

    # Training results
    results = {
        "config": PILOT_MODEL_CONFIG,
        "device": str(device),
        "seed": args.seed,
        "stages": {}
    }

    # Run stages
    total_start = time.time()
    resume_checkpoint = args.resume

    for stage_num in stages_to_run:
        try:
            # Load data
            data_loader = load_stage_data(data_dir, stage_num, args.batch_size)

            # Create loss function
            loss_fn = create_loss_fn(stage_num)

            # Train stage
            stage_result = train_stage(
                stage_num=stage_num,
                model=model,
                data_loader=data_loader,
                loss_fn=loss_fn,
                output_dir=output_dir,
                log_file=log_file,
                device=device,
                resume_from=resume_checkpoint
            )

            results["stages"][stage_num] = stage_result

            # Use this stage's checkpoint for next stage
            resume_checkpoint = stage_result.get("checkpoint_path")

        except Exception as e:
            log_message(log_file, f"ERROR in stage {stage_num}: {e}")
            import traceback
            log_message(log_file, traceback.format_exc())
            raise

    total_elapsed = time.time() - total_start

    # Summary
    log_message(log_file, "\n" + "=" * 60)
    log_message(log_file, "PILOT TRAINING COMPLETE")
    log_message(log_file, "=" * 60)
    log_message(log_file, f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")

    for stage_num, stage_result in results["stages"].items():
        log_message(
            log_file,
            f"Stage {stage_num} ({stage_result['name']}): "
            f"loss={stage_result['final_loss']:.4f}, "
            f"time={stage_result['elapsed_seconds']:.1f}s"
        )

    # Save results
    results["total_elapsed_seconds"] = total_elapsed
    results_file = output_dir / "pilot_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log_message(log_file, f"\nResults saved to: {results_file}")

    # Save final model
    final_model_path = output_dir / "pilot_model_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": PILOT_MODEL_CONFIG,
        "results": results
    }, final_model_path)
    log_message(log_file, f"Final model saved to: {final_model_path}")

    return results


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run TKS-LLM pilot training experiment"
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="data/pilot",
        help="Directory containing pilot data"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="outputs/pilot",
        help="Directory for checkpoints and logs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run specific stage only (1-4)"
    )

    args = parser.parse_args()
    run_pilot_training(args)


if __name__ == "__main__":
    main()

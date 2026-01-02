#!/usr/bin/env python3
"""
TKS Curriculum Training Script

Staged training approach for the TKS Expressiveness Stack:
  Stage 1: SubFoundationMixer only (ground local world semantics)
  Stage 2: + AcquisitionProgram (add D/W/P compositional structure)
  Stage 3: + FoundationGraphDiffusion (relational diffusion across foundations)

Each stage builds on the previous, with optional warmup and scale ramping.

Usage:
    python scripts/train_tks_curriculum.py --data output/mixed_eq_nl_train.jsonl --output output/curriculum_train
    python scripts/train_tks_curriculum.py --stage 2 --checkpoint output/curriculum_train/stage1_best.pt
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_llm_core_v4 import TKSNoeticLMConfig, TKSNoeticLM

# Import data pipeline from train_cuda.py
from scripts.train_cuda import SimpleTokenizer, TKSDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CurriculumStageConfig:
    """Configuration for a single curriculum stage."""
    name: str
    epochs: int
    learning_rate: float
    # Module enables
    use_subfoundation_mixer: bool = False
    use_acquisition_program: bool = False
    use_foundation_graph: bool = False
    # Scale ramp (start -> end over stage)
    subfoundation_scale_start: float = 0.05
    subfoundation_scale_end: float = 0.1
    acquisition_scale_start: float = 0.05
    acquisition_scale_end: float = 0.1
    foundation_scale_start: float = 0.05
    foundation_scale_end: float = 0.1
    # Other params
    acquisition_depth: int = 2
    diffusion_steps: int = 2
    warmup_epochs: int = 1


# Default 3-stage curriculum
DEFAULT_CURRICULUM: List[CurriculumStageConfig] = [
    CurriculumStageConfig(
        name="stage1_subfoundation",
        epochs=5,
        learning_rate=1e-4,
        use_subfoundation_mixer=True,
        use_acquisition_program=False,
        use_foundation_graph=False,
        subfoundation_scale_start=0.02,
        subfoundation_scale_end=0.1,
    ),
    CurriculumStageConfig(
        name="stage2_acquisition",
        epochs=5,
        learning_rate=5e-5,
        use_subfoundation_mixer=True,
        use_acquisition_program=True,
        use_foundation_graph=False,
        subfoundation_scale_start=0.1,
        subfoundation_scale_end=0.1,
        acquisition_scale_start=0.02,
        acquisition_scale_end=0.1,
        acquisition_depth=2,
    ),
    CurriculumStageConfig(
        name="stage3_diffusion",
        epochs=5,
        learning_rate=2e-5,
        use_subfoundation_mixer=True,
        use_acquisition_program=True,
        use_foundation_graph=True,
        subfoundation_scale_start=0.1,
        subfoundation_scale_end=0.1,
        acquisition_scale_start=0.1,
        acquisition_scale_end=0.1,
        foundation_scale_start=0.02,
        foundation_scale_end=0.1,
        diffusion_steps=2,
    ),
]


def compute_scale_for_epoch(
    epoch: int,
    total_epochs: int,
    scale_start: float,
    scale_end: float,
    warmup_epochs: int = 1,
) -> float:
    """Linear scale ramp with warmup."""
    if epoch < warmup_epochs:
        # Warmup: ramp from 0 to scale_start
        return scale_start * (epoch + 1) / warmup_epochs
    else:
        # Main: ramp from scale_start to scale_end
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return scale_start + (scale_end - scale_start) * progress


def update_model_scales(
    model: TKSNoeticLM,
    stage: CurriculumStageConfig,
    epoch: int,
) -> Dict[str, float]:
    """Update expressiveness scales based on curriculum stage and epoch."""
    scales = {}

    for block in model.blocks:
        if block.expressiveness is None:
            continue

        # SubFoundation scale
        if hasattr(block.expressiveness, 'subfoundation_mixer') and block.expressiveness.subfoundation_mixer is not None:
            sf_scale = compute_scale_for_epoch(
                epoch, stage.epochs,
                stage.subfoundation_scale_start, stage.subfoundation_scale_end,
                stage.warmup_epochs
            )
            block.expressiveness.subfoundation_mixer.gate.data.fill_(sf_scale)
            scales['subfoundation'] = sf_scale

        # Acquisition scale
        if hasattr(block.expressiveness, 'acquisition_program') and block.expressiveness.acquisition_program is not None:
            acq_scale = compute_scale_for_epoch(
                epoch, stage.epochs,
                stage.acquisition_scale_start, stage.acquisition_scale_end,
                stage.warmup_epochs
            )
            block.expressiveness.acquisition_program.gate.data.fill_(acq_scale)
            scales['acquisition'] = acq_scale

        # Foundation diffusion scale
        if hasattr(block.expressiveness, 'foundation_diffusion') and block.expressiveness.foundation_diffusion is not None:
            fd_scale = compute_scale_for_epoch(
                epoch, stage.epochs,
                stage.foundation_scale_start, stage.foundation_scale_end,
                stage.warmup_epochs
            )
            block.expressiveness.foundation_diffusion.gate.data.fill_(fd_scale)
            scales['foundation'] = fd_scale

    return scales


def create_model_for_stage(
    stage: CurriculumStageConfig,
    base_config: TKSNoeticLMConfig,
) -> TKSNoeticLM:
    """Create a model configured for a specific curriculum stage."""
    config = TKSNoeticLMConfig(
        vocab_size=base_config.vocab_size,
        max_seq_len=base_config.max_seq_len,
        num_layers=base_config.num_layers,
        num_scales=base_config.num_scales,
        dropout=base_config.dropout,
        use_attractor=base_config.use_attractor,
        use_stable_attractor=base_config.use_stable_attractor,
        contraction_factor=base_config.contraction_factor,
        max_attractor_iter=base_config.max_attractor_iter,
        use_rpm=base_config.use_rpm,
        tie_embeddings=base_config.tie_embeddings,
        # Stage-specific expressiveness config
        use_subfoundation_mixer=stage.use_subfoundation_mixer,
        use_acquisition_program=stage.use_acquisition_program,
        use_foundation_graph=stage.use_foundation_graph,
        subfoundation_scale=stage.subfoundation_scale_start,
        acquisition_scale=stage.acquisition_scale_start,
        acquisition_depth=stage.acquisition_depth,
        foundation_scale=stage.foundation_scale_start,
        diffusion_steps=stage.diffusion_steps,
    )
    return TKSNoeticLM(config)


def save_checkpoint(
    model: TKSNoeticLM,
    path: str,
    stage: CurriculumStageConfig,
) -> None:
    """Save checkpoint with full config for proper model loading."""
    config = model.config if hasattr(model, 'config') else None

    # Build config dict for load_model compatibility
    config_dict = {
        'model_type': 'v4',
        'vocab_size': config.vocab_size if config else 512,
        'max_seq_len': config.max_seq_len if config else 256,
        'num_layers': config.num_layers if config else 4,
        'num_scales': config.num_scales if config else 3,
        'use_attractor': config.use_attractor if config else True,
        'use_stable_attractor': config.use_stable_attractor if config else True,
        'contraction_factor': config.contraction_factor if config else 0.5,
        'max_attractor_iter': config.max_attractor_iter if config else 15,
        'use_rpm': config.use_rpm if config else True,
        # Expressiveness config
        'use_subfoundation_mixer': stage.use_subfoundation_mixer,
        'use_acquisition_program': stage.use_acquisition_program,
        'use_foundation_graph': stage.use_foundation_graph,
    }

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict,
        'stage': stage.name,
    }, path)


def transfer_weights(
    source_model: TKSNoeticLM,
    target_model: TKSNoeticLM,
) -> int:
    """Transfer compatible weights from source to target model."""
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    transferred = 0
    for key in target_state:
        if key in source_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key]
            transferred += 1

    target_model.load_state_dict(target_state)
    return transferred


def run_curriculum_stage(
    stage: CurriculumStageConfig,
    model: TKSNoeticLM,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    output_dir: str,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> TKSNoeticLM:
    """Run a single curriculum stage."""
    logger.info(f"=" * 60)
    logger.info(f"Starting {stage.name}")
    logger.info(f"  Epochs: {stage.epochs}, LR: {stage.learning_rate}")
    logger.info(f"  SubFoundation: {stage.use_subfoundation_mixer}")
    logger.info(f"  Acquisition: {stage.use_acquisition_program}")
    logger.info(f"  Foundation: {stage.use_foundation_graph}")
    logger.info(f"=" * 60)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage.learning_rate, weight_decay=0.01)

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * stage.epochs
    warmup_steps = len(train_loader) * stage.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.1, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stage_dir = os.path.join(output_dir, stage.name)
    os.makedirs(stage_dir, exist_ok=True)

    best_loss = float('inf')
    metrics_history = []
    global_step = 0

    for epoch in range(stage.epochs):
        # Update scales for this epoch
        scales = update_model_scales(model, stage, epoch)
        logger.info(f"Epoch {epoch+1}/{stage.epochs} - Scales: {scales}")

        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{stage.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            loss_mask = batch.get('loss_mask', None)
            if loss_mask is not None:
                loss_mask = loss_mask.to(device)

            # Forward pass
            outputs = model(input_ids)
            logits = outputs['logits']

            # Compute loss
            if loss_mask is not None:
                # Masked loss for seq2seq
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                mask_flat = loss_mask.view(-1)

                ce_loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')
                masked_loss = (ce_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                loss = masked_loss
            else:
                # Standard next-token prediction
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=0,  # Ignore PAD
                )

            # Gradient accumulation
            loss = loss / grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}")
                optimizer.zero_grad()
                continue

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * grad_accum_steps
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        avg_loss = total_loss / max(1, num_batches)

        # Evaluation
        eval_loss = None
        if eval_loader is not None:
            eval_loss = evaluate_model(model, eval_loader, device)
            logger.info(f"  Epoch {epoch+1} train_loss: {avg_loss:.4f}, eval_loss: {eval_loss:.4f}")
        else:
            logger.info(f"  Epoch {epoch+1} train_loss: {avg_loss:.4f}")

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'eval_loss': eval_loss,
            'scales': scales,
            'lr': scheduler.get_last_lr()[0],
        })

        # Save best (by eval_loss if available, else train_loss)
        check_loss = eval_loss if eval_loss is not None else avg_loss
        if check_loss < best_loss:
            best_loss = check_loss
            # Save with config for proper model loading
            save_checkpoint(model, os.path.join(stage_dir, 'best_model.pt'), stage)
            logger.info(f"  New best model saved (loss: {best_loss:.4f})")

    # Save final and metrics
    save_checkpoint(model, os.path.join(stage_dir, 'final_model.pt'), stage)
    with open(os.path.join(stage_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'stage': stage.name,
            'config': asdict(stage),
            'best_loss': best_loss,
            'history': metrics_history,
        }, f, indent=2)

    logger.info(f"Stage {stage.name} complete. Best loss: {best_loss:.4f}")
    return model


def evaluate_model(model: TKSNoeticLM, eval_loader: DataLoader, device: torch.device) -> float:
    """Evaluate model on eval set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            loss_mask = batch.get('loss_mask', None)
            if loss_mask is not None:
                loss_mask = loss_mask.to(device)

            outputs = model(input_ids)
            logits = outputs['logits']

            if loss_mask is not None:
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                mask_flat = loss_mask.view(-1)
                ce_loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (ce_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            else:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=0,
                )

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description='TKS Curriculum Training')
    parser.add_argument('--data', type=str, default='output/mixed_eq_nl_train.jsonl',
                        help='Training data JSONL file')
    parser.add_argument('--output', type=str, default='output/curriculum_train',
                        help='Output directory')
    parser.add_argument('--stage', type=int, default=0,
                        help='Start from specific stage (1, 2, or 3). 0 = all stages')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of noetic blocks')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--max-seq-len', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--eval-split', type=float, default=0.1,
                        help='Fraction of data for evaluation')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = SimpleTokenizer(max_length=args.max_seq_len)
    logger.info(f"  Vocab size: {tokenizer.actual_vocab_size}")

    # Load dataset
    if not os.path.exists(args.data):
        logger.error(f"Training data not found: {args.data}")
        sys.exit(1)

    logger.info(f"Loading dataset from {args.data}...")
    full_dataset = TKSDataset(args.data, tokenizer)
    logger.info(f"  Total examples: {len(full_dataset)}")

    # Split into train/eval
    eval_size = int(len(full_dataset) * args.eval_split)
    train_size = len(full_dataset) - eval_size

    if eval_size > 0:
        from torch.utils.data import random_split
        train_dataset, eval_dataset = random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = full_dataset
        eval_dataset = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        )

    # Base config
    base_config = TKSNoeticLMConfig(
        vocab_size=tokenizer.actual_vocab_size,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
    )

    # Determine stages to run
    if args.stage == 0:
        stages = DEFAULT_CURRICULUM
    else:
        stages = [DEFAULT_CURRICULUM[args.stage - 1]]

    # Initialize or load model
    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        # Create model for first stage and load weights
        model = create_model_for_stage(stages[0], base_config)
        state_dict = torch.load(args.checkpoint, map_location=device)
        # Handle partial loading
        model_state = model.state_dict()
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v
        model.load_state_dict(model_state)

    # Run curriculum
    for stage in stages:
        # Create model for this stage
        new_model = create_model_for_stage(stage, base_config)

        # Transfer weights from previous stage if available
        if model is not None:
            transferred = transfer_weights(model, new_model)
            logger.info(f"Transferred {transferred} weight tensors from previous stage")

        model = new_model

        # Run stage
        model = run_curriculum_stage(
            stage=stage,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            output_dir=args.output,
            device=device,
            grad_accum_steps=args.grad_accum,
        )

    logger.info("Curriculum training complete!")
    logger.info(f"Final model saved to: {args.output}")


if __name__ == '__main__':
    main()

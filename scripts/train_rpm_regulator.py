#!/usr/bin/env python3
"""
Train RPM Gating on strg-llm-stk RPM/regulator pack.

Trains RPMGatingMechanism on regulator selection:
- FM_female (Noetic 5): reservoir/authority/expert
- FM_male (Noetic 6): larger set/related ideas
- ACBE (Noetics 8,9): cause-effect chains
- MVR (Noetics 1,4,7): activation plan

Training Objectives:
1. Select appropriate regulator for given context
2. Generate RPM cascades (2-3 layers)
3. Classify triggers correctly
4. D/W/P score prediction

Usage:
    python scripts/train_rpm_regulator.py --epochs 10 --batch-size 32

Data:
    - strg-llm-stk/tks_rpm_fm_acbe_mvr_training_pack_800.jsonl

Author: ML Agent
Date: 2026-01-05
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.losses import RPMLoss, TKSLossConfig

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
class RPMTrainingConfig:
    """Configuration for RPM gating training."""
    # Data
    data_path: str = "strg-llm-stk/tks_rpm_fm_acbe_mvr_training_pack_800.jsonl"
    train_split: float = 0.9

    # Model
    noetic_dim: int = 40
    num_foundations: int = 7
    num_regulators: int = 4  # FM_female, FM_male, ACBE, MVR

    # Training
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Loss weights
    lambda_regulator: float = 1.0
    lambda_dwp: float = 0.5
    lambda_cascade: float = 0.3
    lambda_trigger: float = 0.2

    # Checkpointing
    checkpoint_dir: str = "output/rpm_regulator"
    save_every: int = 200
    log_every: int = 20

    # Device
    device: str = "auto"


# =============================================================================
# REGULATOR MAPPING
# =============================================================================

REGULATOR_MAP = {
    'fm_female': 0,
    'fm_female_authority': 0,
    'authority': 0,
    'reservoir': 0,
    'expert': 0,

    'fm_male': 1,
    'fm_male_superset': 1,
    'expansion': 1,
    'superset': 1,

    'acbe': 2,
    'acbe_cause_effect': 2,
    'cause_effect': 2,

    'mvr': 3,
    'mvr_activation_plan': 3,
    'activation': 3,
}

NOETIC_INDICES = {
    'fm_female': [5],      # Noetic 5
    'fm_male': [6],        # Noetic 6
    'acbe': [8, 9],        # Noetics 8, 9
    'mvr': [1, 4, 7],      # Noetics 1, 4, 7
}


# =============================================================================
# DATASET
# =============================================================================

class RPMDataset(Dataset):
    """Dataset for RPM regulator training from JSONL files."""

    def __init__(
        self,
        data_path: str,
        noetic_dim: int = 40,
        max_samples: Optional[int] = None,
    ):
        self.noetic_dim = noetic_dim
        self.samples = []

        full_path = PROJECT_ROOT / data_path
        if full_path.exists():
            logger.info(f"Loading {full_path}")
            self._load_jsonl(full_path)

        if max_samples:
            self.samples = self.samples[:max_samples]

        logger.info(f"Loaded {len(self.samples)} RPM training samples")

    def _load_jsonl(self, path: Path) -> None:
        """Load samples from JSONL file."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    sample = self._parse_record(record)
                    if sample is not None:
                        self.samples.append(sample)
                except json.JSONDecodeError:
                    continue

    def _parse_record(self, record: Dict) -> Optional[Dict]:
        """Parse a training record into a sample."""
        task_type = record.get('type', '')
        meta = record.get('meta', {})
        messages = record.get('messages', [])

        # Extract messages
        user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')
            elif msg.get('role') == 'assistant':
                assistant_msg = msg.get('content', '')

        if not user_msg or not assistant_msg:
            return None

        # Parse based on task type
        regulator_type = self._get_regulator_type(task_type)

        return {
            'type': task_type,
            'regulator_idx': REGULATOR_MAP.get(regulator_type, 0),
            'noetic_indices': NOETIC_INDICES.get(regulator_type, [0]),
            'input_text': user_msg,
            'target_text': assistant_msg,
            'foundation_stack': self._parse_foundation_stack(meta),
            'meta': meta,
        }

    def _get_regulator_type(self, task_type: str) -> str:
        """Extract regulator type from task type."""
        task_lower = task_type.lower()
        if 'fm_female' in task_lower or 'authority' in task_lower:
            return 'fm_female'
        elif 'fm_male' in task_lower or 'superset' in task_lower:
            return 'fm_male'
        elif 'acbe' in task_lower or 'cause_effect' in task_lower:
            return 'acbe'
        elif 'mvr' in task_lower or 'activation' in task_lower:
            return 'mvr'
        elif 'cascade' in task_lower or 'trigger' in task_lower:
            return 'cascade'  # Special case for cascade/trigger tasks
        return 'mvr'  # Default

    def _parse_foundation_stack(self, meta: Dict) -> List[int]:
        """Parse foundation stack from metadata."""
        stack_str = meta.get('foundation_stack', '')
        if not stack_str:
            return [0]

        # Parse format like "F1:F5:F2" or "1:5:2"
        stack = []
        parts = stack_str.replace('F', '').split(':')
        for part in parts:
            try:
                idx = int(part.strip()) - 1  # Convert to 0-indexed
                if 0 <= idx < 7:
                    stack.append(idx)
            except ValueError:
                continue

        return stack if stack else [0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Generate pseudo-random embeddings from text hash
        text_hash = hash(sample['input_text']) % (2**32)
        torch.manual_seed(text_hash)

        noetic_input = torch.randn(self.noetic_dim)

        # Target D/W/P scores (derived from regulator type)
        # Different regulators have characteristic D/W/P profiles
        dwp_target = self._get_dwp_profile(sample['regulator_idx'])

        return {
            'noetic_input': noetic_input,
            'regulator_idx': torch.tensor(sample['regulator_idx']),
            'noetic_indices': torch.tensor(sample['noetic_indices']),
            'dwp_target': dwp_target,
            'foundation_stack': torch.tensor(sample['foundation_stack']),
            'sample_type': sample['type'],
        }

    def _get_dwp_profile(self, regulator_idx: int) -> torch.Tensor:
        """Get characteristic D/W/P profile for regulator type."""
        # FM_female (authority): High wisdom (expert knowledge)
        # FM_male (expansion): High desire (explore related)
        # ACBE (cause-effect): High power (actionable chains)
        # MVR (activation): Balanced high (activation plan)
        profiles = {
            0: [0.6, 0.9, 0.5],  # FM_female: D=0.6, W=0.9, P=0.5
            1: [0.9, 0.6, 0.5],  # FM_male: D=0.9, W=0.6, P=0.5
            2: [0.7, 0.7, 0.9],  # ACBE: D=0.7, W=0.7, P=0.9
            3: [0.8, 0.8, 0.8],  # MVR: D=0.8, W=0.8, P=0.8
        }
        return torch.tensor(profiles.get(regulator_idx, [0.5, 0.5, 0.5]))


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of samples."""
    # Pad noetic_indices to same length
    max_noetic_len = max(len(s['noetic_indices']) for s in batch)
    padded_noetic = []
    for s in batch:
        indices = s['noetic_indices']
        if len(indices) < max_noetic_len:
            # Pad with zeros
            padded = torch.cat([
                indices,
                torch.zeros(max_noetic_len - len(indices), dtype=torch.long)
            ])
        else:
            padded = indices
        padded_noetic.append(padded)

    # Pad foundation_stack to same length
    max_stack_len = max(len(s['foundation_stack']) for s in batch)
    padded_stack = []
    for s in batch:
        stack = s['foundation_stack']
        if len(stack) < max_stack_len:
            padded = torch.cat([
                stack,
                torch.zeros(max_stack_len - len(stack), dtype=torch.long)
            ])
        else:
            padded = stack
        padded_stack.append(padded)

    return {
        'noetic_input': torch.stack([s['noetic_input'] for s in batch]),
        'regulator_idx': torch.stack([s['regulator_idx'] for s in batch]),
        'noetic_indices': torch.stack(padded_noetic),
        'dwp_target': torch.stack([s['dwp_target'] for s in batch]),
        'foundation_stack': torch.stack(padded_stack),
        'sample_types': [s['sample_type'] for s in batch],
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_rpm_regulator(config: RPMTrainingConfig) -> None:
    """Main training loop for RPM regulator."""

    # Setup device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = RPMDataset(config.data_path, noetic_dim=config.noetic_dim)

    # Split train/val
    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Load model
    try:
        from tks_llm_core_v2 import RPMGatingMechanism
        model = RPMGatingMechanism(dim=config.noetic_dim)
    except ImportError:
        logger.warning("RPMGatingMechanism not found, using simple model")
        model = SimpleRPMModel(config.noetic_dim, config.num_regulators)

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * config.epochs,
    )

    # Loss functions
    rpm_loss_fn = RPMLoss()
    regulator_loss_fn = nn.CrossEntropyLoss()

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            noetic_input = batch['noetic_input'].to(device)
            regulator_idx = batch['regulator_idx'].to(device)
            dwp_target = batch['dwp_target'].to(device)

            # Forward pass
            optimizer.zero_grad()

            # Get RPM output
            output = model(noetic_input.unsqueeze(1))  # Add sequence dim

            # Compute losses
            losses = {}

            # 1. Regulator classification loss
            if 'regulator_logits' in output:
                losses['regulator'] = regulator_loss_fn(
                    output['regulator_logits'],
                    regulator_idx
                )
            else:
                losses['regulator'] = torch.tensor(0.0, device=device)

            # 2. D/W/P prediction loss
            if 'dwp_scores' in output:
                dwp_scores = output['dwp_scores']
                # Average across foundations and sequence
                if dwp_scores.dim() == 4:  # [batch, seq, 7, 3]
                    dwp_pred = dwp_scores.mean(dim=[1, 2])  # [batch, 3]
                elif dwp_scores.dim() == 3:  # [batch, 7, 3]
                    dwp_pred = dwp_scores.mean(dim=1)  # [batch, 3]
                else:
                    dwp_pred = dwp_scores

                losses['dwp'] = F.mse_loss(dwp_pred, dwp_target)
            else:
                losses['dwp'] = torch.tensor(0.0, device=device)

            # Total loss
            total_loss = (
                config.lambda_regulator * losses['regulator'] +
                config.lambda_dwp * losses['dwp']
            )

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % config.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step} | Epoch {epoch+1}/{config.epochs} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )

            # Checkpointing
            if global_step % config.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, epoch,
                    checkpoint_dir / f"step_{global_step}.pt"
                )

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader, device, rpm_loss_fn)
        logger.info(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, global_step, epoch,
                checkpoint_dir / "best.pt"
            )
            logger.info(f"New best model saved (val_loss: {val_loss:.4f})")

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, global_step, epoch,
        checkpoint_dir / "final.pt"
    )
    logger.info("Training complete!")

    # Export pretrained weights
    torch.save(model.state_dict(), checkpoint_dir / "rpm_regulator_pretrained.pt")
    logger.info(f"Exported pretrained weights to {checkpoint_dir / 'rpm_regulator_pretrained.pt'}")


def validate(model, val_loader, device, loss_fn) -> float:
    """Run validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            noetic_input = batch['noetic_input'].to(device)
            dwp_target = batch['dwp_target'].to(device)

            output = model(noetic_input.unsqueeze(1))

            if 'dwp_scores' in output:
                dwp_scores = output['dwp_scores']
                if dwp_scores.dim() == 4:
                    dwp_pred = dwp_scores.mean(dim=[1, 2])
                elif dwp_scores.dim() == 3:
                    dwp_pred = dwp_scores.mean(dim=1)
                else:
                    dwp_pred = dwp_scores
                loss = F.mse_loss(dwp_pred, dwp_target)
                total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model, optimizer, scheduler, step, epoch, path: Path
) -> None:
    """Save training checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'epoch': epoch,
    }, path)


class SimpleRPMModel(nn.Module):
    """Simple fallback model if RPMGatingMechanism unavailable."""

    def __init__(self, noetic_dim: int = 40, num_regulators: int = 4):
        super().__init__()
        self.noetic_dim = noetic_dim

        # D/W/P prediction heads (per foundation)
        self.dwp_heads = nn.ModuleList([
            nn.Linear(noetic_dim, 3) for _ in range(7)
        ])

        # Regulator classifier
        self.regulator_classifier = nn.Sequential(
            nn.Linear(noetic_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_regulators),
        )

        # Gate computation
        self.gate_proj = nn.Linear(noetic_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, dim = x.shape

        # Pool sequence
        x_pooled = x.mean(dim=1)  # [batch, dim]

        # D/W/P scores per foundation
        dwp_scores = []
        for head in self.dwp_heads:
            scores = torch.sigmoid(head(x_pooled))  # [batch, 3]
            dwp_scores.append(scores)
        dwp_scores = torch.stack(dwp_scores, dim=1)  # [batch, 7, 3]

        # Regulator logits
        regulator_logits = self.regulator_classifier(x_pooled)

        # Gate
        gate = torch.sigmoid(self.gate_proj(x_pooled))

        # Gated output
        gated_output = x * gate.unsqueeze(1)

        return {
            'gated_output': gated_output,
            'dwp_scores': dwp_scores,
            'rpm_gate': gate,
            'regulator_logits': regulator_logits,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RPM Regulator")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="output/rpm_regulator")
    args = parser.parse_args()

    config = RPMTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    train_rpm_regulator(config)


if __name__ == "__main__":
    main()

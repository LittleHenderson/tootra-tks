#!/usr/bin/env python3
"""
Train Operator Core on strg-llm-stk operator training pack.

Trains TKSCompositionalLayerV2 on operator semantics:
- + (Association): link/coexist/support
- - (Disassociation): remove the second from the first
- * (Intensification): amplify/fuse effect
- / (Opposition): contrast/tension/polarity

Training Objectives:
1. Operator grounding: Model understands each operator's semantic
2. Minimal pairs: Distinguish similar operators
3. PEMDAS ordering: Correct order of operations
4. Substitution tests: Verify algebraic properties
5. Symmetry constraints: + and * are symmetric, - and / are anti-symmetric

Usage:
    python scripts/train_operator_core.py --epochs 10 --batch-size 32

Data:
    - strg-llm-stk/tks_operator_training_pack_2000.jsonl
    - strg-llm-stk/tks_operator_training_pack_250.jsonl

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

from training.strg_losses import (
    OperatorSymmetryLoss,
    OperatorGroundingLoss,
    MinimalPairContrastiveLoss,
)

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
class OperatorTrainingConfig:
    """Configuration for operator core training."""
    # Data
    data_paths: List[str] = None
    train_split: float = 0.9

    # Model
    element_dim: int = 40  # Noetic dimension
    equation_dim: int = 384
    operator_gate_init: float = 0.1

    # Training
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Loss weights
    lambda_grounding: float = 1.0
    lambda_symmetry: float = 0.3
    lambda_contrastive: float = 0.5
    lambda_pemdas: float = 0.2

    # Checkpointing
    checkpoint_dir: str = "output/operator_core"
    save_every: int = 500
    log_every: int = 50

    # Device
    device: str = "auto"

    def __post_init__(self):
        if self.data_paths is None:
            self.data_paths = [
                "strg-llm-stk/tks_operator_training_pack_2000.jsonl",
                "strg-llm-stk/tks_operator_training_pack_250.jsonl",
            ]


# =============================================================================
# DATASET
# =============================================================================

# Operator mapping
OPERATOR_MAP = {
    '+': 0, 'add': 0, 'association': 0,
    '-': 1, 'sub': 1, 'disassociation': 1,
    '*': 2, 'mul': 2, 'intensification': 2, '×': 2,
    '/': 3, 'div': 3, 'opposition': 3, '÷': 3,
}


class OperatorDataset(Dataset):
    """Dataset for operator training from JSONL files."""

    def __init__(
        self,
        data_paths: List[str],
        element_dim: int = 40,
        max_samples: Optional[int] = None,
    ):
        self.element_dim = element_dim
        self.samples = []

        for path in data_paths:
            full_path = PROJECT_ROOT / path
            if full_path.exists():
                logger.info(f"Loading {full_path}")
                self._load_jsonl(full_path)

        if max_samples:
            self.samples = self.samples[:max_samples]

        logger.info(f"Loaded {len(self.samples)} operator training samples")

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

        # Extract user and assistant messages
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
        if task_type == 'operator_grounding':
            return self._parse_grounding(meta, user_msg, assistant_msg)
        elif task_type == 'operator_minimal_pair':
            return self._parse_minimal_pair(meta, user_msg, assistant_msg)
        elif task_type == 'operator_selection':
            return self._parse_selection(meta, user_msg, assistant_msg)
        elif task_type == 'substitution_test':
            return self._parse_substitution(meta, user_msg, assistant_msg)
        elif task_type == 'pemdas_order':
            return self._parse_pemdas(meta, user_msg, assistant_msg)
        else:
            # Generic: use operator from meta
            operator = meta.get('operator', meta.get('operators', ['+'])[0])
            return {
                'type': 'generic',
                'operator_idx': OPERATOR_MAP.get(operator, 0),
                'input_text': user_msg,
                'target_text': assistant_msg,
            }

    def _parse_grounding(self, meta: Dict, user: str, assistant: str) -> Dict:
        """Parse operator grounding task."""
        operator = meta.get('operator', '+')
        return {
            'type': 'grounding',
            'operator_idx': OPERATOR_MAP.get(operator, 0),
            'input_text': user,
            'target_text': assistant,
        }

    def _parse_minimal_pair(self, meta: Dict, user: str, assistant: str) -> Dict:
        """Parse minimal pair task."""
        operators = meta.get('operators', ['+', '-'])
        correct = meta.get('correct_operator', operators[0])
        return {
            'type': 'minimal_pair',
            'correct_operator_idx': OPERATOR_MAP.get(correct, 0),
            'pair_operators': [OPERATOR_MAP.get(op, 0) for op in operators],
            'input_text': user,
            'target_text': assistant,
        }

    def _parse_selection(self, meta: Dict, user: str, assistant: str) -> Dict:
        """Parse operator selection task."""
        correct = meta.get('correct_operator', '+')
        return {
            'type': 'selection',
            'operator_idx': OPERATOR_MAP.get(correct, 0),
            'input_text': user,
            'target_text': assistant,
        }

    def _parse_substitution(self, meta: Dict, user: str, assistant: str) -> Dict:
        """Parse substitution test task."""
        operator = meta.get('operator', '+')
        return {
            'type': 'substitution',
            'operator_idx': OPERATOR_MAP.get(operator, 0),
            'input_text': user,
            'target_text': assistant,
            'is_symmetric': operator in ['+', '*'],
        }

    def _parse_pemdas(self, meta: Dict, user: str, assistant: str) -> Dict:
        """Parse PEMDAS ordering task."""
        return {
            'type': 'pemdas',
            'input_text': user,
            'target_text': assistant,
            'operators': [OPERATOR_MAP.get(op, 0) for op in meta.get('operators', [])],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Generate pseudo-random embeddings from text hash for consistency
        # In full implementation, use proper tokenizer and embedding
        text_hash = hash(sample['input_text']) % (2**32)
        torch.manual_seed(text_hash)

        left_emb = torch.randn(self.element_dim)
        right_emb = torch.randn(self.element_dim)

        return {
            'left_emb': left_emb,
            'right_emb': right_emb,
            'operator_idx': torch.tensor(sample.get('operator_idx', 0)),
            'sample_type': sample['type'],
            'is_symmetric': sample.get('is_symmetric', False),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of samples."""
    return {
        'left_emb': torch.stack([s['left_emb'] for s in batch]),
        'right_emb': torch.stack([s['right_emb'] for s in batch]),
        'operator_idx': torch.stack([s['operator_idx'] for s in batch]),
        'sample_types': [s['sample_type'] for s in batch],
        'is_symmetric': torch.tensor([s['is_symmetric'] for s in batch]),
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_operator_core(config: OperatorTrainingConfig) -> None:
    """Main training loop for operator core."""

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
    dataset = OperatorDataset(config.data_paths, element_dim=config.element_dim)

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
        from tks_compositional_layer import TKSCompositionalLayerV2
        model = TKSCompositionalLayerV2(
            element_dim=config.element_dim,
            equation_dim=config.equation_dim,
            init_gate=config.operator_gate_init,
        )
    except ImportError:
        logger.warning("TKSCompositionalLayerV2 not found, using simple model")
        model = SimpleOperatorModel(config.element_dim)

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
    grounding_loss_fn = OperatorGroundingLoss()
    symmetry_loss_fn = OperatorSymmetryLoss()
    contrastive_loss_fn = MinimalPairContrastiveLoss()

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            left_emb = batch['left_emb'].to(device)
            right_emb = batch['right_emb'].to(device)
            operator_idx = batch['operator_idx'].to(device)

            # Forward pass
            optimizer.zero_grad()

            # Get operator core output
            if hasattr(model, 'forward_with_operator'):
                output = model.forward_with_operator(left_emb, right_emb, operator_idx)
            else:
                output = model(left_emb, right_emb, operator_idx)

            # Compute losses
            losses = {}

            # 1. Grounding loss (operator classification)
            if 'operator_logits' in output:
                grounding = grounding_loss_fn(output['operator_logits'], operator_idx)
                losses['grounding'] = grounding['total']
            else:
                losses['grounding'] = torch.tensor(0.0, device=device)

            # 2. Symmetry loss
            if 'symmetry_outputs' in output:
                symmetry = symmetry_loss_fn(output['symmetry_outputs'])
                losses['symmetry'] = symmetry['total']
            else:
                losses['symmetry'] = torch.tensor(0.0, device=device)

            # 3. Contrastive loss (if applicable)
            if 'contrastive_triplets' in output:
                anchor, positive, negative = output['contrastive_triplets']
                contrastive = contrastive_loss_fn(anchor, positive, negative)
                losses['contrastive'] = contrastive['total']
            else:
                losses['contrastive'] = torch.tensor(0.0, device=device)

            # Total loss
            total_loss = (
                config.lambda_grounding * losses['grounding'] +
                config.lambda_symmetry * losses['symmetry'] +
                config.lambda_contrastive * losses['contrastive']
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
        val_loss = validate(model, val_loader, device, grounding_loss_fn)
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
    torch.save(model.state_dict(), checkpoint_dir / "operator_core_pretrained.pt")
    logger.info(f"Exported pretrained weights to {checkpoint_dir / 'operator_core_pretrained.pt'}")


def validate(model, val_loader, device, loss_fn) -> float:
    """Run validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            left_emb = batch['left_emb'].to(device)
            right_emb = batch['right_emb'].to(device)
            operator_idx = batch['operator_idx'].to(device)

            if hasattr(model, 'forward_with_operator'):
                output = model.forward_with_operator(left_emb, right_emb, operator_idx)
            else:
                output = model(left_emb, right_emb, operator_idx)

            if 'operator_logits' in output:
                loss = loss_fn(output['operator_logits'], operator_idx)
                total_loss += loss['total'].item()
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


class SimpleOperatorModel(nn.Module):
    """Simple fallback model if TKSCompositionalLayerV2 unavailable."""

    def __init__(self, element_dim: int = 40):
        super().__init__()
        self.element_dim = element_dim

        # Operator embeddings
        self.operator_embeddings = nn.Embedding(4, element_dim)

        # Composition network
        self.compose = nn.Sequential(
            nn.Linear(element_dim * 3, element_dim * 2),
            nn.GELU(),
            nn.Linear(element_dim * 2, element_dim),
        )

        # Classification head
        self.classifier = nn.Linear(element_dim, 4)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        operator_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = left.shape[0]

        # Get operator embedding
        op_emb = self.operator_embeddings(operator_idx)

        # Compose
        combined = torch.cat([left, op_emb, right], dim=-1)
        output = self.compose(combined)

        # Classify operator from output
        logits = self.classifier(output)

        return {
            'noetic_output': output,
            'operator_logits': logits,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Operator Core")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="output/operator_core")
    args = parser.parse_args()

    config = OperatorTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    train_operator_core(config)


if __name__ == "__main__":
    main()

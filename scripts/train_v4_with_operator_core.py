#!/usr/bin/env python3
"""
Train TKS v4 Model with Pretrained Operator Core

Joint training recipe:
1. Load pretrained operator core
2. Freeze operator core for N epochs (let main model learn world/RPM)
3. Unfreeze at lower LR to fine-tune without destroying operator structure
4. Gate schedule: init high (0.2) → decay to target (0.1)

Data sources:
- JSONL split files from corpus (equation data)
- Built-in NL samples (for world classification)

Usage:
    python scripts/train_v4_with_operator_core.py

    # Full recipe with freeze/unfreeze
    python scripts/train_v4_with_operator_core.py \
        --train-jsonl data/equation_embeddings/splits/train.jsonl \
        --operator-path output/operator_core_pretrained.pt \
        --output-path output/v4_with_operator_core.pt \
        --epochs 30 --freeze-operator-epochs 5 \
        --gate-init 0.2 --gate-final 0.1 \
        --lr 3e-4 --operator-lr-mult 0.1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v4 import (
    TKSNoeticLM,
    TKSNoeticLMConfig,
    load_pretrained_operator_core,
)
from equation_detector import EquationDetector


# =============================================================================
# TRAINING DATA
# =============================================================================

# NL training samples (world-tagged) - always included
NL_SAMPLES = [
    # World A (Spiritual)
    ("Divine consciousness awakens", "A"),
    ("Spiritual realm of awareness", "A"),
    ("Sacred light of being", "A"),
    ("Transcendent unity", "A"),
    ("Holy presence manifests", "A"),
    ("Eternal divine spark", "A"),
    # World B (Mental)
    ("Mental clarity and focus", "B"),
    ("Cognitive processing power", "B"),
    ("Thinking deeply about", "B"),
    ("Rational understanding", "B"),
    ("Intellectual analysis", "B"),
    ("Mind sharpens thought", "B"),
    # World C (Emotional)
    ("Emotional warmth flows", "C"),
    ("Feeling love and care", "C"),
    ("Heart connection deepens", "C"),
    ("Compassionate response", "C"),
    ("Joy fills the soul", "C"),
    ("Empathy bridges hearts", "C"),
    # World D (Physical)
    ("Physical strength manifests", "D"),
    ("Bodily energy surges", "D"),
    ("Material form solidifies", "D"),
    ("Tangible reality shapes", "D"),
    ("Force moves matter", "D"),
    ("Action creates change", "D"),
]

WORLD_TO_IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
OP_TO_IDX = {'+': 0, '-': 1, '×': 2, '÷': 3}


class SimpleTokenizer:
    def __init__(self, max_length: int = 64):
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)

    def _build_vocab(self) -> Dict[str, int]:
        vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        chars = 'ABCDabcdEFGHIJKLMNOPQRSTUVWXYZefghijklmnopqrstuvwxyz'
        chars += '0123456789+-×÷*/ \n\t.,!?\'"'
        for c in chars:
            if c not in vocab:
                vocab[c] = len(vocab)
        return vocab

    def tokenize(self, text: str) -> List[int]:
        tokens = [self.vocab.get('<bos>', 2)]
        for c in text[:self.max_length - 2]:
            tokens.append(self.vocab.get(c, self.vocab.get('<unk>', 1)))
        tokens.append(self.vocab.get('<eos>', 3))
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.vocab.get('<pad>', 0))
        return tokens[:self.max_length]


def load_equation_jsonl(path: str) -> List[Dict]:
    """Load equations from JSONL split file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    return items


class TKSTrainingDataset(Dataset):
    """Dataset mixing equation JSONL and NL samples."""

    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        equation_jsonl: Optional[str] = None,
        equation_weight: float = 2.0,
        max_equations: int = None,
    ):
        self.tokenizer = tokenizer
        self.detector = EquationDetector()
        self.samples = []

        # Add equation samples from JSONL
        if equation_jsonl and os.path.exists(equation_jsonl):
            equations = load_equation_jsonl(equation_jsonl)
            if max_equations:
                equations = equations[:max_equations]

            for eq in equations:
                # Build equation text
                text = f"{eq['left']}{eq['operator']}{eq['right']}"
                # Determine world from left element
                world = eq['left'][0]  # First char is world (A, B, C, D)

                self.samples.append({
                    'text': text,
                    'world': world,
                    'is_equation': True,
                    'weight': equation_weight,
                    'left': eq['left'],
                    'right': eq['right'],
                    'operator': eq['operator'],
                })

            print(f"Loaded {len(equations)} equations from {equation_jsonl}")
        else:
            # Fallback: use built-in equation samples
            builtin_equations = [
                ("A1+A2", "A"), ("A3+A4", "A"), ("A5+A6", "A"),
                ("A1-A2", "A"), ("A3×A4", "A"), ("A5÷A6", "A"),
                ("B1+B2", "B"), ("B3+B4", "B"), ("B5+B6", "B"),
                ("B1-B2", "B"), ("B3×B4", "B"), ("B5÷B6", "B"),
                ("C1+C2", "C"), ("C3+C4", "C"), ("C5+C6", "C"),
                ("C1-C2", "C"), ("C3×C4", "C"), ("C5÷C6", "C"),
                ("D1+D2", "D"), ("D3+D4", "D"), ("D5+D6", "D"),
                ("D1-D2", "D"), ("D3×D4", "D"), ("D5÷D6", "D"),
            ]
            for text, world in builtin_equations:
                self.samples.append({
                    'text': text,
                    'world': world,
                    'is_equation': True,
                    'weight': equation_weight,
                })
            print(f"Using {len(builtin_equations)} built-in equations")

        # Add NL samples
        for text, world in NL_SAMPLES:
            self.samples.append({
                'text': text,
                'world': world,
                'is_equation': False,
                'weight': 1.0,
            })

        print(f"Total samples: {len(self.samples)} ({sum(1 for s in self.samples if s['is_equation'])} equations, {sum(1 for s in self.samples if not s['is_equation'])} NL)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        world = sample['world']

        tokens = self.tokenizer.tokenize(text)
        world_idx = WORLD_TO_IDX[world]

        # Parse equation triplet if applicable
        triplet = None
        if sample['is_equation']:
            parsed = self.detector.parse_single(text)
            if parsed:
                eq = parsed[0]
                triplet = (eq.left_idx, eq.operator_idx, eq.right_idx)

        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'world_idx': world_idx,
            'is_equation': sample['is_equation'],
            'triplet': triplet,
            'weight': sample['weight'],
        }


def collate_fn(batch):
    """Custom collate for variable triplets."""
    tokens = torch.stack([b['tokens'] for b in batch])
    world_idx = torch.tensor([b['world_idx'] for b in batch], dtype=torch.long)
    is_equation = torch.tensor([b['is_equation'] for b in batch], dtype=torch.bool)
    weights = torch.tensor([b['weight'] for b in batch], dtype=torch.float)

    # Handle triplets
    triplets = [b['triplet'] for b in batch]
    has_triplet = [t is not None for t in triplets]

    if any(has_triplet):
        left_idx = torch.tensor([t[0] if t else 0 for t in triplets], dtype=torch.long)
        right_idx = torch.tensor([t[2] if t else 0 for t in triplets], dtype=torch.long)
        op_idx = torch.tensor([t[1] if t else 0 for t in triplets], dtype=torch.long)
        triplet_mask = torch.tensor(has_triplet, dtype=torch.bool)
    else:
        left_idx = None
        right_idx = None
        op_idx = None
        triplet_mask = None

    return {
        'tokens': tokens,
        'world_idx': world_idx,
        'is_equation': is_equation,
        'weights': weights,
        'left_idx': left_idx,
        'right_idx': right_idx,
        'op_idx': op_idx,
        'triplet_mask': triplet_mask,
    }


# =============================================================================
# LOSSES
# =============================================================================

def compute_world_loss(
    noetic_output: torch.Tensor,  # [batch, seq, 40]
    world_idx: torch.Tensor,      # [batch]
) -> torch.Tensor:
    """Loss to encourage correct world activation."""
    noetic = noetic_output.mean(dim=1)  # [batch, 40]

    world_norms = torch.stack([
        noetic[:, 0:10].norm(dim=-1),   # A
        noetic[:, 10:20].norm(dim=-1),  # B
        noetic[:, 20:30].norm(dim=-1),  # C
        noetic[:, 30:40].norm(dim=-1),  # D
    ], dim=-1)  # [batch, 4]

    loss = F.cross_entropy(world_norms, world_idx)
    return loss


def compute_lm_loss(
    logits: torch.Tensor,  # [batch, seq, vocab]
    tokens: torch.Tensor,  # [batch, seq]
) -> torch.Tensor:
    """Next-token prediction loss."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=0,
    )
    return loss


# =============================================================================
# GATE SCHEDULE
# =============================================================================

def set_gate_value(model: TKSNoeticLM, gate_value: float):
    """Set operator core gate to specific value."""
    if model.operator_core is None:
        return

    import numpy as np
    if gate_value <= 0:
        logit = -10.0
    elif gate_value >= 1:
        logit = 10.0
    else:
        logit = np.log(gate_value / (1 - gate_value))

    with torch.no_grad():
        model.operator_core.gate.gate_logits.fill_(logit)


def get_gate_for_epoch(epoch: int, total_epochs: int, gate_init: float, gate_final: float) -> float:
    """Linear decay from gate_init to gate_final."""
    if total_epochs <= 1:
        return gate_final
    progress = epoch / (total_epochs - 1)
    return gate_init + (gate_final - gate_init) * progress


# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Tokenizer and dataset
    tokenizer = SimpleTokenizer(max_length=args.max_seq_len)
    dataset = TKSTrainingDataset(
        tokenizer,
        equation_jsonl=args.train_jsonl,
        equation_weight=args.equation_weight,
        max_equations=args.max_equations,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    config = TKSNoeticLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        use_operator_core=True,
        operator_gate_init=args.gate_init,
        use_attractor=True,
        use_stable_attractor=True,
        use_rpm=True,
    )

    model = TKSNoeticLM(config).to(device)

    # Load pretrained operator core
    if os.path.exists(args.operator_path):
        load_pretrained_operator_core(model, args.operator_path, device)
    else:
        print(f"Warning: No pretrained operator core at {args.operator_path}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Separate parameter groups for freeze/unfreeze
    operator_params = list(model.operator_core.parameters()) if model.operator_core else []
    other_params = [p for n, p in model.named_parameters() if 'operator_core' not in n]

    # Initially freeze operator core
    for p in operator_params:
        p.requires_grad = False

    # Optimizer (only other params initially)
    optimizer = AdamW([
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Freeze operator core: epochs 1-{args.freeze_operator_epochs}")
    print(f"  Gate schedule: {args.gate_init} -> {args.gate_final}")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Unfreeze operator core after freeze period
        if epoch == args.freeze_operator_epochs + 1 and operator_params:
            print(f"\n*** Unfreezing operator core at epoch {epoch} ***")
            for p in operator_params:
                p.requires_grad = True

            # Add operator params to optimizer with lower LR
            optimizer.add_param_group({
                'params': operator_params,
                'lr': args.lr * args.operator_lr_mult,
            })

        # Update gate value
        gate = get_gate_for_epoch(epoch - 1, args.epochs, args.gate_init, args.gate_final)
        set_gate_value(model, gate)

        model.train()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_world_loss = 0.0
        total_sym_loss = 0.0
        steps = 0

        for batch in loader:
            tokens = batch['tokens'].to(device)
            world_idx = batch['world_idx'].to(device)

            # Build equation triplet
            triplet = None
            if batch['triplet_mask'] is not None and batch['triplet_mask'].any():
                triplet = (
                    batch['left_idx'].to(device),
                    batch['right_idx'].to(device),
                    batch['op_idx'].to(device),
                )

            # Forward
            output = model(tokens, equation_triplet=triplet)

            # Losses
            lm_loss = compute_lm_loss(output['logits'], tokens)
            world_loss = compute_world_loss(output['gated_output'], world_idx)

            # Symmetry loss from operator core
            sym_loss = torch.tensor(0.0, device=device)
            if 'operator_core_output' in output and output['operator_core_output'] is not None:
                op_out = output['operator_core_output']
                if 'symmetry_losses' in op_out:
                    sym_loss = op_out['symmetry_losses']['total']

            # Weighted combination
            loss = (
                args.lm_weight * lm_loss +
                args.world_weight * world_loss +
                args.sym_weight * sym_loss
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_world_loss += world_loss.item()
            total_sym_loss += sym_loss.item()
            steps += 1

        scheduler.step()

        avg_loss = total_loss / steps
        avg_lm = total_lm_loss / steps
        avg_world = total_world_loss / steps
        avg_sym = total_sym_loss / steps

        # Evaluate world accuracy
        model.eval()
        correct = 0
        total = 0
        eq_correct = 0
        eq_total = 0
        nl_correct = 0
        nl_total = 0

        with torch.no_grad():
            for batch in loader:
                tokens = batch['tokens'].to(device)
                world_idx = batch['world_idx'].to(device)
                is_eq = batch['is_equation']

                triplet = None
                if batch['triplet_mask'] is not None and batch['triplet_mask'].any():
                    triplet = (
                        batch['left_idx'].to(device),
                        batch['right_idx'].to(device),
                        batch['op_idx'].to(device),
                    )

                output = model(tokens, equation_triplet=triplet)
                noetic = output['gated_output'].mean(dim=1)

                world_norms = torch.stack([
                    noetic[:, 0:10].norm(dim=-1),
                    noetic[:, 10:20].norm(dim=-1),
                    noetic[:, 20:30].norm(dim=-1),
                    noetic[:, 30:40].norm(dim=-1),
                ], dim=-1)

                predicted = world_norms.argmax(dim=-1)
                correct_mask = (predicted == world_idx)

                correct += correct_mask.sum().item()
                total += len(world_idx)

                for i in range(len(is_eq)):
                    if is_eq[i]:
                        eq_total += 1
                        if correct_mask[i]:
                            eq_correct += 1
                    else:
                        nl_total += 1
                        if correct_mask[i]:
                            nl_correct += 1

        world_acc = correct / total if total > 0 else 0.0
        eq_acc = eq_correct / eq_total if eq_total > 0 else 0.0
        nl_acc = nl_correct / nl_total if nl_total > 0 else 0.0

        frozen_str = "FROZEN" if epoch <= args.freeze_operator_epochs else "UNFROZEN"

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"gate={gate:.2f} [{frozen_str}] | "
              f"loss={avg_loss:.4f} (lm={avg_lm:.4f} world={avg_world:.4f} sym={avg_sym:.4f}) | "
              f"acc={world_acc:.1%} (eq={eq_acc:.1%} nl={nl_acc:.1%})")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': config.vocab_size,
                    'max_seq_len': config.max_seq_len,
                    'num_layers': config.num_layers,
                    'use_operator_core': config.use_operator_core,
                    'operator_gate_init': config.operator_gate_init,
                    'model_type': 'v4',
                },
                'epoch': epoch,
                'loss': avg_loss,
                'world_acc': world_acc,
                'eq_acc': eq_acc,
                'nl_acc': nl_acc,
            }
            torch.save(checkpoint, args.output_path)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final accuracy: {world_acc:.1%} (eq={eq_acc:.1%} nl={nl_acc:.1%})")
    print(f"Saved to: {args.output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train TKS v4 with Operator Core")

    # Data
    parser.add_argument("--train-jsonl", type=str, default=None,
                        help="Path to training JSONL (from split)")
    parser.add_argument("--max-equations", type=int, default=None,
                        help="Max equations to use from JSONL")

    # Model
    parser.add_argument("--operator-path", type=str,
                        default="output/operator_core_pretrained.pt",
                        help="Path to pretrained operator core")
    parser.add_argument("--output-path", type=str,
                        default="output/v4_with_operator_core.pt",
                        help="Output checkpoint path")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Freeze/unfreeze
    parser.add_argument("--freeze-operator-epochs", type=int, default=5,
                        help="Epochs to freeze operator core")
    parser.add_argument("--operator-lr-mult", type=float, default=0.1,
                        help="LR multiplier for operator core after unfreeze")

    # Gate schedule
    parser.add_argument("--gate-init", type=float, default=0.2,
                        help="Initial gate value")
    parser.add_argument("--gate-final", type=float, default=0.1,
                        help="Final gate value")

    # Loss weights
    parser.add_argument("--lm-weight", type=float, default=1.0)
    parser.add_argument("--world-weight", type=float, default=0.5)
    parser.add_argument("--sym-weight", type=float, default=0.1)
    parser.add_argument("--equation-weight", type=float, default=2.0,
                        help="Sample weight for equations vs NL")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()

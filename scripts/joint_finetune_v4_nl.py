#!/usr/bin/env python3
"""
Joint Fine-Tuning: v4 Model + NL Retriever with Freeze/Unfreeze Schedule

This script implements the full NL bridge training with:
1. TKS v4 model with pretrained operator core
2. NL retriever for equation fallback
3. Freeze/unfreeze schedule to preserve operator symmetry/anti-symmetry

Training phases:
  Phase 1 (epochs 1-N): Freeze operator core
    - Train main model to learn world/RPM
    - Train NL retriever to align NL→equation

  Phase 2 (epochs N+1-end): Unfreeze operator core with lower LR
    - Fine-tune all components jointly
    - Lower LR on operator core (0.1x) to preserve structure

Data sources:
  - Equation JSONL with interpretations (NL→equation pairs)
  - Built-in NL samples for world classification

Usage:
    # Full joint training
    python scripts/joint_finetune_v4_nl.py \
        --train-jsonl data/equation_embeddings/splits/train.jsonl \
        --nl-corpus data/nl_equation_pairs/nl_corpus_train.pt \
        --eq-corpus data/equation_embeddings/equation_corpus.pt \
        --operator-path output/operator_core_pretrained.pt \
        --output-path output/v4_joint_finetuned.pt \
        --epochs 40 --freeze-epochs 10 \
        --lr 3e-4 --operator-lr-mult 0.1 --retriever-lr-mult 0.5

    # Quick test
    python scripts/joint_finetune_v4_nl.py --epochs 5 --freeze-epochs 2
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

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
from tks_nl_retriever import TKSNLRetriever, InfoNCELoss
from equation_detector import EquationDetector


# =============================================================================
# CONSTANTS
# =============================================================================

WORLD_TO_IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
OP_TO_IDX = {'+': 0, '-': 1, '×': 2, '÷': 3}

# Built-in NL samples for world classification
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


# =============================================================================
# TOKENIZER
# =============================================================================

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
        while len(tokens) < self.max_length:
            tokens.append(self.vocab.get('<pad>', 0))
        return tokens[:self.max_length]


# =============================================================================
# DATASETS
# =============================================================================

def load_equation_jsonl(path: str) -> List[Dict]:
    """Load equations from JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    return items


class JointTrainingDataset(Dataset):
    """
    Dataset combining:
    1. Equation JSONL (with interpretations for NL training)
    2. Built-in NL samples for world classification
    3. NL corpus embeddings for retriever training
    """

    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        equation_jsonl: Optional[str] = None,
        nl_corpus_path: Optional[str] = None,
        equation_weight: float = 2.0,
        nl_weight: float = 1.0,
        max_equations: int = None,
    ):
        self.tokenizer = tokenizer
        self.detector = EquationDetector()
        self.samples = []

        # Load NL corpus for retriever training
        self.nl_embeddings = None
        self.nl_eq_indices = None
        if nl_corpus_path and os.path.exists(nl_corpus_path):
            corpus = torch.load(nl_corpus_path, map_location='cpu', weights_only=False)
            self.nl_embeddings = corpus.get('embeddings')
            self.nl_eq_indices = corpus.get('equation_indices', corpus.get('triplets'))
            print(f"Loaded NL corpus: {len(self.nl_embeddings)} embeddings")

        # Load equation samples from JSONL
        if equation_jsonl and os.path.exists(equation_jsonl):
            equations = load_equation_jsonl(equation_jsonl)
            if max_equations:
                equations = equations[:max_equations]

            for i, eq in enumerate(equations):
                # Build equation text
                text = f"{eq['left']}{eq['operator']}{eq['right']}"
                world = eq['left'][0]

                # Get interpretation if available
                interpretation = eq.get('interpretation', '')

                # Get NL embedding index if we have corpus
                nl_emb_idx = i if self.nl_embeddings is not None and i < len(self.nl_embeddings) else None

                self.samples.append({
                    'text': text,
                    'world': world,
                    'is_equation': True,
                    'weight': equation_weight,
                    'left': eq['left'],
                    'right': eq['right'],
                    'operator': eq['operator'],
                    'interpretation': interpretation,
                    'nl_emb_idx': nl_emb_idx,
                })

            print(f"Loaded {len(equations)} equations from {equation_jsonl}")
        else:
            # Fallback: built-in equations
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
                    'nl_emb_idx': None,
                })
            print(f"Using {len(builtin_equations)} built-in equations")

        # Add built-in NL samples
        for text, world in NL_SAMPLES:
            self.samples.append({
                'text': text,
                'world': world,
                'is_equation': False,
                'weight': nl_weight,
                'nl_emb_idx': None,
            })

        eq_count = sum(1 for s in self.samples if s['is_equation'])
        nl_count = sum(1 for s in self.samples if not s['is_equation'])
        print(f"Total samples: {len(self.samples)} ({eq_count} equations, {nl_count} NL)")

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

        # Get NL embedding if available
        nl_embedding = None
        nl_eq_idx = None
        if sample.get('nl_emb_idx') is not None and self.nl_embeddings is not None:
            nl_emb_idx = sample['nl_emb_idx']
            if nl_emb_idx < len(self.nl_embeddings):
                nl_embedding = self.nl_embeddings[nl_emb_idx]
                if self.nl_eq_indices is not None and nl_emb_idx < len(self.nl_eq_indices):
                    nl_eq_idx = self.nl_eq_indices[nl_emb_idx]

        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'world_idx': world_idx,
            'is_equation': sample['is_equation'],
            'triplet': triplet,
            'weight': sample['weight'],
            'nl_embedding': nl_embedding,
            'nl_eq_idx': nl_eq_idx,
        }


def collate_fn(batch):
    """Custom collate for variable triplets and NL embeddings."""
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

    # Handle NL embeddings
    nl_embeddings = [b['nl_embedding'] for b in batch]
    has_nl = [e is not None for e in nl_embeddings]

    if any(has_nl):
        # Create stacked embeddings with zeros for missing
        embedding_dim = next(e.shape[-1] for e in nl_embeddings if e is not None)
        nl_emb_tensor = torch.zeros(len(batch), embedding_dim)
        for i, emb in enumerate(nl_embeddings):
            if emb is not None:
                nl_emb_tensor[i] = emb
        nl_mask = torch.tensor(has_nl, dtype=torch.bool)

        # NL equation indices for retriever loss
        nl_eq_indices = [b['nl_eq_idx'] for b in batch]
        nl_eq_idx_tensor = torch.tensor([idx if idx is not None else -1 for idx in nl_eq_indices], dtype=torch.long)
    else:
        nl_emb_tensor = None
        nl_mask = None
        nl_eq_idx_tensor = None

    return {
        'tokens': tokens,
        'world_idx': world_idx,
        'is_equation': is_equation,
        'weights': weights,
        'left_idx': left_idx,
        'right_idx': right_idx,
        'op_idx': op_idx,
        'triplet_mask': triplet_mask,
        'nl_embeddings': nl_emb_tensor,
        'nl_mask': nl_mask,
        'nl_eq_indices': nl_eq_idx_tensor,
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


def compute_retriever_loss(
    retriever: TKSNLRetriever,
    nl_embeddings: torch.Tensor,   # [batch, 384]
    eq_corpus: torch.Tensor,       # [corpus_size, 384]
    target_indices: torch.Tensor,  # [batch] indices into corpus
    mask: torch.Tensor,            # [batch] bool
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss for retriever training.

    Only computes loss for samples with valid NL embeddings.
    """
    if not mask.any() or eq_corpus is None:
        return torch.tensor(0.0, device=nl_embeddings.device)

    # Filter to valid samples
    valid_nl = nl_embeddings[mask]
    valid_targets = target_indices[mask]

    # Filter out samples with invalid target indices
    valid_mask = valid_targets >= 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=nl_embeddings.device)

    valid_nl = valid_nl[valid_mask]
    valid_targets = valid_targets[valid_mask]

    # Project NL embeddings
    projected = retriever(valid_nl)  # [n, 384]

    # Get target equation embeddings
    target_eq = eq_corpus[valid_targets]  # [n, 384]

    # Normalize
    projected = F.normalize(projected, dim=-1)
    target_eq = F.normalize(target_eq, dim=-1)

    # InfoNCE loss: positives on diagonal
    similarity = torch.matmul(projected, target_eq.T) / temperature  # [n, n]
    labels = torch.arange(len(projected), device=projected.device)

    loss = F.cross_entropy(similarity, labels)
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

@dataclass
class TrainingMetrics:
    """Track training metrics per epoch."""
    epoch: int
    loss: float = 0.0
    lm_loss: float = 0.0
    world_loss: float = 0.0
    sym_loss: float = 0.0
    retriever_loss: float = 0.0
    world_acc: float = 0.0
    eq_acc: float = 0.0
    nl_acc: float = 0.0
    gate_value: float = 0.0
    is_frozen: bool = True


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load equation corpus for retriever
    eq_corpus = None
    eq_corpus_meta = None
    if args.eq_corpus and os.path.exists(args.eq_corpus):
        corpus_data = torch.load(args.eq_corpus, map_location='cpu', weights_only=False)
        eq_corpus = corpus_data.get('embeddings')
        eq_corpus_meta = corpus_data.get('metadata', [])
        print(f"Loaded equation corpus: {len(eq_corpus)} embeddings")

    # Tokenizer and dataset
    tokenizer = SimpleTokenizer(max_length=args.max_seq_len)
    dataset = JointTrainingDataset(
        tokenizer,
        equation_jsonl=args.train_jsonl,
        nl_corpus_path=args.nl_corpus,
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
        use_nl_retriever=False,  # We'll handle retriever separately for joint training
    )

    model = TKSNoeticLM(config).to(device)

    # Load pretrained operator core
    if os.path.exists(args.operator_path):
        load_pretrained_operator_core(model, args.operator_path, device)
    else:
        print(f"Warning: No pretrained operator core at {args.operator_path}")

    # Initialize NL retriever
    retriever = TKSNLRetriever(
        nl_dim=384,
        eq_dim=384,
        hidden_dim=args.retriever_hidden_dim,
        use_hidden=True,
        dropout=0.1,
    ).to(device)

    # Load pretrained retriever if available
    if args.retriever_path and os.path.exists(args.retriever_path):
        checkpoint = torch.load(args.retriever_path, map_location=device, weights_only=False)
        retriever.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained retriever from {args.retriever_path}")

    # Parameter counts
    model_params = sum(p.numel() for p in model.parameters())
    retriever_params = sum(p.numel() for p in retriever.parameters())
    print(f"Model parameters: {model_params:,}")
    print(f"Retriever parameters: {retriever_params:,}")
    print(f"Total parameters: {model_params + retriever_params:,}")

    # Separate parameter groups
    operator_params = list(model.operator_core.parameters()) if model.operator_core else []
    other_model_params = [p for n, p in model.named_parameters() if 'operator_core' not in n]
    retriever_params_list = list(retriever.parameters())

    # Initially freeze operator core
    for p in operator_params:
        p.requires_grad = False
    print(f"Operator core frozen: {len(operator_params)} parameters")

    # Optimizer: main model + retriever (operator core added later)
    optimizer = AdamW([
        {'params': other_model_params, 'lr': args.lr, 'name': 'model'},
        {'params': retriever_params_list, 'lr': args.lr * args.retriever_lr_mult, 'name': 'retriever'},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Move corpus to device
    if eq_corpus is not None:
        eq_corpus = eq_corpus.to(device)

    # Training loop
    print(f"\n{'='*70}")
    print("JOINT FINE-TUNING: V4 Model + NL Retriever")
    print(f"{'='*70}")
    print(f"Epochs: {args.epochs}")
    print(f"Freeze operator core: epochs 1-{args.freeze_epochs}")
    print(f"Gate schedule: {args.gate_init} -> {args.gate_final}")
    print(f"Loss weights: LM={args.lm_weight}, World={args.world_weight}, Sym={args.sym_weight}, Retriever={args.retriever_weight}")
    print(f"{'='*70}\n")

    best_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        # Phase transition: unfreeze operator core
        if epoch == args.freeze_epochs + 1 and operator_params:
            print(f"\n{'*'*70}")
            print(f"PHASE 2: Unfreezing operator core at epoch {epoch}")
            print(f"{'*'*70}\n")

            for p in operator_params:
                p.requires_grad = True

            # Add operator params to optimizer with lower LR
            optimizer.add_param_group({
                'params': operator_params,
                'lr': args.lr * args.operator_lr_mult,
                'name': 'operator_core',
            })

        # Update gate value
        gate = get_gate_for_epoch(epoch - 1, args.epochs, args.gate_init, args.gate_final)
        set_gate_value(model, gate)

        is_frozen = epoch <= args.freeze_epochs

        # Train
        model.train()
        retriever.train()

        total_loss = 0.0
        total_lm_loss = 0.0
        total_world_loss = 0.0
        total_sym_loss = 0.0
        total_retriever_loss = 0.0
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

            # Forward pass
            output = model(tokens, equation_triplet=triplet)

            # LM loss
            lm_loss = compute_lm_loss(output['logits'], tokens)

            # World loss
            world_loss = compute_world_loss(output['gated_output'], world_idx)

            # Symmetry loss from operator core
            sym_loss = torch.tensor(0.0, device=device)
            if 'operator_core_output' in output and output['operator_core_output'] is not None:
                op_out = output['operator_core_output']
                if 'symmetry_losses' in op_out:
                    sym_loss = op_out['symmetry_losses']['total']

            # Retriever loss (if NL embeddings available)
            retriever_loss = torch.tensor(0.0, device=device)
            if batch['nl_embeddings'] is not None and eq_corpus is not None:
                retriever_loss = compute_retriever_loss(
                    retriever,
                    batch['nl_embeddings'].to(device),
                    eq_corpus,
                    batch['nl_eq_indices'].to(device),
                    batch['nl_mask'].to(device),
                    temperature=args.retriever_temperature,
                )

            # Combined loss
            loss = (
                args.lm_weight * lm_loss +
                args.world_weight * world_loss +
                args.sym_weight * sym_loss +
                args.retriever_weight * retriever_loss
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_world_loss += world_loss.item()
            total_sym_loss += sym_loss.item()
            total_retriever_loss += retriever_loss.item()
            steps += 1

        scheduler.step()

        # Average losses
        avg_loss = total_loss / steps
        avg_lm = total_lm_loss / steps
        avg_world = total_world_loss / steps
        avg_sym = total_sym_loss / steps
        avg_retriever = total_retriever_loss / steps

        # Evaluate
        model.eval()
        retriever.eval()

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

        # Record metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            loss=avg_loss,
            lm_loss=avg_lm,
            world_loss=avg_world,
            sym_loss=avg_sym,
            retriever_loss=avg_retriever,
            world_acc=world_acc,
            eq_acc=eq_acc,
            nl_acc=nl_acc,
            gate_value=gate,
            is_frozen=is_frozen,
        )
        history.append(metrics)

        # Print progress
        frozen_str = "FROZEN" if is_frozen else "UNFROZEN"
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"gate={gate:.2f} [{frozen_str}] | "
              f"loss={avg_loss:.4f} (lm={avg_lm:.4f} world={avg_world:.4f} sym={avg_sym:.4f} ret={avg_retriever:.4f}) | "
              f"acc={world_acc:.1%} (eq={eq_acc:.1%} nl={nl_acc:.1%})")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'retriever_state_dict': retriever.state_dict(),
                'config': {
                    'vocab_size': config.vocab_size,
                    'max_seq_len': config.max_seq_len,
                    'num_layers': config.num_layers,
                    'use_operator_core': config.use_operator_core,
                    'operator_gate_init': config.operator_gate_init,
                    'model_type': 'v4_joint',
                },
                'retriever_config': {
                    'nl_dim': 384,
                    'eq_dim': 384,
                    'hidden_dim': args.retriever_hidden_dim,
                    'use_hidden': True,
                },
                'epoch': epoch,
                'loss': avg_loss,
                'world_acc': world_acc,
                'eq_acc': eq_acc,
                'nl_acc': nl_acc,
            }
            torch.save(checkpoint, args.output_path)
            print(f"  -> Saved best checkpoint")

    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final accuracy: {world_acc:.1%} (eq={eq_acc:.1%} nl={nl_acc:.1%})")
    print(f"Saved model: {args.output_path}")

    # Also save retriever separately
    retriever_path = args.output_path.replace('.pt', '_retriever.pt')
    torch.save({
        'model_state_dict': retriever.state_dict(),
        'config': {
            'nl_dim': 384,
            'eq_dim': 384,
            'hidden_dim': args.retriever_hidden_dim,
            'use_hidden': True,
            'dropout': 0.1,
        }
    }, retriever_path)
    print(f"Saved retriever: {retriever_path}")

    # Save training history
    history_path = args.output_path.replace('.pt', '_history.json')
    history_dicts = [
        {
            'epoch': m.epoch,
            'loss': m.loss,
            'lm_loss': m.lm_loss,
            'world_loss': m.world_loss,
            'sym_loss': m.sym_loss,
            'retriever_loss': m.retriever_loss,
            'world_acc': m.world_acc,
            'eq_acc': m.eq_acc,
            'nl_acc': m.nl_acc,
            'gate_value': m.gate_value,
            'is_frozen': m.is_frozen,
        }
        for m in history
    ]
    with open(history_path, 'w') as f:
        json.dump(history_dicts, f, indent=2)
    print(f"Saved history: {history_path}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Joint Fine-Tune TKS v4 + NL Retriever")

    # Data paths
    parser.add_argument("--train-jsonl", type=str, default=None,
                        help="Path to training JSONL (from split)")
    parser.add_argument("--nl-corpus", type=str, default="data/nl_equation_pairs/nl_corpus_train.pt",
                        help="Path to NL corpus for retriever training")
    parser.add_argument("--eq-corpus", type=str, default="data/equation_embeddings/equation_corpus.pt",
                        help="Path to equation corpus")
    parser.add_argument("--max-equations", type=int, default=None,
                        help="Max equations to use from JSONL")

    # Model paths
    parser.add_argument("--operator-path", type=str,
                        default="output/operator_core_pretrained.pt",
                        help="Path to pretrained operator core")
    parser.add_argument("--retriever-path", type=str, default=None,
                        help="Path to pretrained retriever (optional)")
    parser.add_argument("--output-path", type=str,
                        default="output/v4_joint_finetuned.pt",
                        help="Output checkpoint path")

    # Model config
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--retriever-hidden-dim", type=int, default=256)

    # Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Freeze/unfreeze schedule
    parser.add_argument("--freeze-epochs", type=int, default=10,
                        help="Epochs to freeze operator core (Phase 1)")
    parser.add_argument("--operator-lr-mult", type=float, default=0.1,
                        help="LR multiplier for operator core after unfreeze")
    parser.add_argument("--retriever-lr-mult", type=float, default=0.5,
                        help="LR multiplier for retriever")

    # Gate schedule
    parser.add_argument("--gate-init", type=float, default=0.2,
                        help="Initial gate value")
    parser.add_argument("--gate-final", type=float, default=0.1,
                        help="Final gate value")

    # Loss weights
    parser.add_argument("--lm-weight", type=float, default=1.0)
    parser.add_argument("--world-weight", type=float, default=0.5)
    parser.add_argument("--sym-weight", type=float, default=0.1)
    parser.add_argument("--retriever-weight", type=float, default=0.3)
    parser.add_argument("--equation-weight", type=float, default=2.0,
                        help="Sample weight for equations vs NL")
    parser.add_argument("--retriever-temperature", type=float, default=0.07,
                        help="Temperature for retriever contrastive loss")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()

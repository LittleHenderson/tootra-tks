#!/usr/bin/env python3
"""
TKS v7 RTTA Training - Reasoning-Test Tootra Algorithm
Trains TKSGeneralLMv7 on structured reasoning data.
"""

import sys
import os
import json
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

if not torch.cuda.is_available():
    print("FATAL ERROR: GPU requested but CUDA is not available.")
    sys.exit(1)

DEVICE = 'cuda'
print(f"USING GPU: {torch.cuda.get_device_name(0)}")

torch.cuda.empty_cache()
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class RTTADataset(Dataset):
    """Dataset for RTTA training examples."""

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 768):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id("<PAD>") or 0
        self.bos_id = tokenizer.token_to_id("<BOS>")
        self.eos_id = tokenizer.token_to_id("<EOS>")
        self.unk_id = tokenizer.token_to_id("<UNK>") or 1

        if self.bos_id is None or self.eos_id is None:
            raise ValueError("Tokenizer missing <BOS>/<EOS> tokens.")

        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} RTTA samples")

    def _encode(self, text: str) -> Tuple[List[int], List[int]]:
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.eos_id is not None:
            ids = ids + [self.eos_id]
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        # Labels shifted by 1
        labels = ids[1:] + [-100]
        return ids, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format: "Input: <question>\n<reasoning trace>"
        full_text = f"Input: {item['input']}\n{item['output']}"
        input_ids, labels = self._encode(full_text)
        return {'input_ids': input_ids, 'labels': labels}


def collate_fn(batch, pad_id=0):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        ids = item['input_ids']
        lbls = item['labels']
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(lbls + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }


def train(args):
    # Load tokenizer
    tokenizer_path = "tokenizer_v5.json"
    if not os.path.exists(tokenizer_path):
        print(f"FATAL ERROR: {tokenizer_path} not found.")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    # Config - slightly larger for reasoning
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=384,  # Larger for reasoning
        num_layers=8,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8,
        use_memory_bank=True,
        memory_bank_size=1000,
        use_impact_only_nw=True,
        dps_heavy_threshold=0.3,  # Higher threshold for reasoning
        dps_count_threshold=0.2,
        dps_tokens_for_unlock=3,
        dps_cooldown_episodes=5,
        use_njt=True,
        njt_max_iterations=15,
    )

    # Load model
    model = TKSGeneralLMv7(config).to(DEVICE)

    # Optionally warm start from math_code checkpoint
    warmstart_path = "checkpoints/v7_math_code/v7_math_code_model.pt"
    if os.path.exists(warmstart_path) and args.warmstart:
        print(f"Loading weights from {warmstart_path}")
        checkpoint = torch.load(warmstart_path, map_location=DEVICE, weights_only=False)
        # Filter mismatched keys
        state_dict = {}
        model_state = model.state_dict()
        for k, v in checkpoint.items():
            if k in model_state and v.shape == model_state[k].shape:
                state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {len(state_dict)} matching weights")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Dataset
    dataset = RTTADataset(args.data, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, dataset.pad_id)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, dataset.pad_id)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler with warmup
    warmup_steps = 100
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)  # Label smoothing!

    print(f"\nStarting RTTA Training: {len(train_ds)} train, {len(val_ds)} val, {args.epochs} epochs")

    scaler = torch.amp.GradScaler()
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_depth_bonus = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                output = model(input_ids, attention_mask=mask)
                logits = output['logits']

                # Shift for causal LM
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                ce_loss = loss_fn(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1)
                )

                # Depth bonus from DPS (reward deeper reasoning)
                depth_bonus = 0.0
                if 'dps_depth' in output:
                    depth = output['dps_depth']
                    depth_bonus = 0.01 * max(0, depth - 4)  # Bonus for depth > 4

                loss = ce_loss - depth_bonus

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += ce_loss.item()
            total_depth_bonus += depth_bonus

            if batch_idx % 20 == 0:
                depth = output.get('dps_depth', 0)
                nw = output.get('novelty_weight', 0)
                if isinstance(nw, torch.Tensor):
                    nw = nw.mean().item()
                mem = output.get('memory_size', 0)
                lr = scheduler.get_last_lr()[0]

                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {ce_loss.item():.4f} | Depth: {depth} | NW: {nw:.4f} | "
                      f"Mem: {mem}/1000 | LR: {lr:.2e}")

            global_step += 1

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)

                with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    output = model(input_ids, attention_mask=mask)
                    logits = output['logits']

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    loss = loss_fn(
                        shift_logits.view(-1, config.vocab_size),
                        shift_labels.view(-1)
                    )
                    val_loss += loss.item()

        val_loss /= len(val_loader)
        train_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Done | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = f"{args.output_dir}/rtta_model_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'vocab_size': vocab_size,
                'epoch': epoch,
                'val_loss': val_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = f"{args.output_dir}/rtta_model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': vocab_size,
        'epoch': args.epochs,
        'val_loss': val_loss,
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/rtta_training_simple.jsonl')
    parser.add_argument('--output-dir', default='checkpoints/rtta')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmstart', action='store_true', help='Warm start from math_code checkpoint')
    args = parser.parse_args()

    train(args)

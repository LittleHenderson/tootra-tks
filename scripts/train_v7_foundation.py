#!/usr/bin/env python3
"""
TKS v7 Foundation Training - Stage 0
Trains TKSGeneralLMv7 on Canonical TKS Data (40 Elements, 10 Noetics, 28 Foundations).
Ensures the model internalizes the "Operating System" of TKS v7.4.
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
from tokenizers import Tokenizer

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

if not torch.cuda.is_available():
    print("FATAL ERROR: GPU requested but CUDA is not available.")
    sys.exit(1)

DEVICE = 'cuda'
print(f"USING GPU: {torch.cuda.get_device_name(0)}")

# Precision setup
if torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
    print("Using BF16 precision.")
else:
    AMP_DTYPE = torch.float16
    print("Using FP16 precision.")

class CanonDataset(Dataset):
    """Dataset for v7 Canonical Foundation Training."""
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id("<PAD>") or 0
        self.bos_id = tokenizer.token_to_id("<BOS>")
        self.eos_id = tokenizer.token_to_id("<EOS>")
        
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} TKS Canon samples")

    def _encode(self, text: str) -> Tuple[List[int], List[int]]:
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.eos_id is not None:
            ids = ids + [self.eos_id]
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        labels = ids[1:] + [-100]
        return ids, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format consistent with v7 trace-based reasoning
        full_text = f"{item['input']}\n{item['output']}"
        input_ids, labels = self._encode(full_text)
        return {'input_ids': input_ids, 'labels': labels}

def collate_fn(batch, pad_id=0):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        ids = item['input_ids']
        labs = item['labels']
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }

def train(args):
    # Load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()
    
    # Fresh v7 Configuration for retrain-from-scratch run
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        earned_depth_mode=True,
        use_impact_only_nw=True, # Ensure I-based novelty is active
        dps_max_depth=8
    )
    
    model = TKSGeneralLMv7(config).to(DEVICE)
    print("Initialized fresh TKSGeneralLMv7")

    dataset = CanonDataset(args.data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, dataset.pad_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Starting Stage 0 (Canon) Training: {args.epochs} epochs")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                outputs = model(input_ids, attention_mask=mask)
                ce_loss = loss_fn(outputs['logits'].view(-1, vocab_size), labels.view(-1))
                loss = ce_loss # Novelty rewards secondary for foundation stage
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.output_dir}/v7_foundation_best.pt")
            print(f"Saved NEW BEST model to {args.output_dir}/v7_foundation_best.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/v7_canon_training.jsonl')
    parser.add_argument('--output-dir', default='checkpoints/v7_foundation')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)

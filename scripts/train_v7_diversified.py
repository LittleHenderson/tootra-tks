#!/usr/bin/env python3
"""
TKS v7 Diversified Logic Training - Stage 4
Stabilizes the model by training on 20,000 unique procedural traces.
Prevents word-salad and repetitive attractors.
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

# Memory management
torch.cuda.empty_cache()
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

class DiversifiedDataset(Dataset):
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
        print(f"Loaded {len(self.data)} diversified samples")

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
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(item['labels'] + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }

def train(args):
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()
    
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8,
        use_memory_bank=True,
        use_impact_only_nw=True,
        dps_heavy_threshold=0.35, # Raise threshold slightly for better filtering
        dps_tokens_for_unlock=5
    )
    
    model = TKSGeneralLMv7(config).to(DEVICE)
    prev_path = "checkpoints/v7_multidomain/v7_multidomain_model.pt"
    if os.path.exists(prev_path):
        print(f"Loading weights from {prev_path}")
        checkpoint = torch.load(prev_path, map_location=DEVICE, weights_only=True)
        state_dict = {k: v for k, v in checkpoint.items() if "stack_memory" not in k and "stack_ptr" not in k}
        model.load_state_dict(state_dict, strict=False)
    
    dataset = DiversifiedDataset(args.data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, dataset.pad_id))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f"Starting Stage 4 Diversification: {len(dataset)} samples")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                outputs = model(input_ids, attention_mask=mask, return_full_trace=True)
                ce_loss = loss_fn(outputs['logits'].view(-1, vocab_size), labels.view(-1))
                # Add a small depth reward to maintain earned depth behavior
                dps = outputs['dps_state']
                depth_reward = 0.01 * dps.p_max
                loss = ce_loss - depth_reward
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} | Depth: {dps.p_max}")

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.output_dir}/v7_final_stabilized.pt")
    print(f"Final Stabilized Model saved to {args.output_dir}/v7_final_stabilized.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/v7_diversified_training.jsonl')
    parser.add_argument('--output-dir', default='checkpoints/v7_stabilized')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    train(parser.parse_args())

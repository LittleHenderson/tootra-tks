#!/usr/bin/env python3
"""
TKS v7 Multi-Domain Training - Project MAJIK
Trains TKSGeneralLMv7 on Universal Knowledge Domains.
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

class MultiDomainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 512):
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
        print(f"Loaded {len(self.data)} multi-domain samples")

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
        labels.append(ids + [-100] * pad_len)
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
    
    # Config - I-based novelty enabled
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8,
        use_memory_bank=True,
        memory_bank_size=1000,
        use_impact_only_nw=True,
        dps_heavy_threshold=0.01,
        dps_count_threshold=0.005,
        dps_tokens_for_unlock=1,
        dps_cooldown_episodes=0
    )
    
    # Load model (from math/code checkpoint for transfer learning)
    model = TKSGeneralLMv7(config).to(DEVICE)
    math_path = "checkpoints/v7_math_code/v7_math_code_model.pt"
    if os.path.exists(math_path):
        print(f"Loading weights from {math_path}")
        checkpoint = torch.load(math_path, map_location=DEVICE)
        # Filter buffers for safety
        state_dict = {k: v for k, v in checkpoint.items() if "stack_memory" not in k and "stack_ptr" not in k}
        embed_key = "token_emb.weight" if "token_emb.weight" in state_dict else "lm_head.weight"
        if embed_key in state_dict and state_dict[embed_key].shape[0] != config.vocab_size:
            print("Warning: vocab size mismatch; skipping warm start weights.")
        else:
            model.load_state_dict(state_dict, strict=False)
    
    dataset = MultiDomainDataset(args.data, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, dataset.pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, dataset.pad_id))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f"Starting Multi-Domain Training: {len(train_ds)} train, {len(val_ds)} val")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                outputs = model(input_ids, attention_mask=mask, return_full_trace=True)
                logits = outputs['logits']
                
                ce_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # DPS Reward
                dps_state = outputs['dps_state']
                depth_gain = max(0, dps_state.p_max - config.dps_initial_p_max)
                depth_reward = 0.1 * depth_gain
                
                loss = ce_loss - depth_reward
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                dps = outputs.get('dps_state')
                mem = outputs.get('memory_stats')
                nw_info = ""
                if hasattr(dps, 'novelty_history') and dps.novelty_history:
                    nw_info = f" | NW: {dps.novelty_history[-1]:.4f}"
                    
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}) | "
                      f"Depth: {dps.p_max} | Mem: {mem['size'] if mem else 0}/1000{nw_info}")
                
        # Validation
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    outputs = model(input_ids)
                    v_loss += loss_fn(outputs['logits'].view(-1, outputs['logits'].size(-1)), labels.view(-1)).item()
        
        print(f"Epoch {epoch+1} Done | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {v_loss/len(val_loader):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.output_dir}/v7_multidomain_model.pt")
    print(f"Model saved to {args.output_dir}/v7_multidomain_model.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/v7_multidomain_training.jsonl')
    parser.add_argument('--output-dir', default='checkpoints/v7_multidomain')
    parser.add_argument('--epochs', type=int, default=1) # 1 epoch usually enough for rule learning
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    train(parser.parse_args())

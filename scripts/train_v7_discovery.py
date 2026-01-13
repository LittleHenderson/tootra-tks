#!/usr/bin/env python3
"""
TKS v7 Discovery Training - Project MAJIK
Trains TKSGeneralLMv7 on Cross-Domain Discovery Data with DPS-aware loss.
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
    print("Please check your NVIDIA drivers and PyTorch installation.")
    sys.exit(1)

DEVICE = 'cuda'
print(f"USING GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")

# Memory management for v7 recursion
torch.cuda.empty_cache()
if torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
    print("Using BF16 precision.")
else:
    AMP_DTYPE = torch.float16
    print("Using FP16 precision.")

SEP = " => "

class DiscoveryDataset(Dataset):
    """Dataset for v7 Discovery Training."""
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
                    try:
                        self.data.append(json.loads(line))
                    except:
                        continue
        print(f"Loaded {len(self.data)} discovery samples")

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
        
        # Format: Input -> Trace -> Output
        # This teaches the model to internalize the TKS trace before generating the answer
        full_text = f"Input: {item['input']}\nTrace: {item.get('tks_notation', '')}\nOutput: {item['output']}"
        
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
    tokenizer_path = "tokenizer_v5.json"
    if not os.path.exists(tokenizer_path):
        print(f"FATAL ERROR: {tokenizer_path} not found.")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    # Init v7 Model
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,        # Scaled for demo/speed
        num_layers=6,          # Deep enough for recursion
        num_scales=4,          # Fractal scales (replaces heads)
        use_dps_gating=True,   # ENABLE DPS
        earned_depth_mode=True,
        novelty_boost_on_heavy=True,
        dps_heavy_threshold=0.01, # Keep low for bootstrap
        dps_count_threshold=0.005,
        dps_tokens_for_unlock=2,
        dps_cooldown_episodes=0,  # NO COOLDOWN
        dps_max_depth=8           # INCREASED CAP
    )
    
    model = TKSGeneralLMv7(config).to(DEVICE)
    
    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        print(f"Loading foundation model from {args.init_checkpoint}")
        checkpoint = torch.load(args.init_checkpoint, map_location=DEVICE, weights_only=True)
        # Filter out transient recursion stack buffers
        filtered_state = {k: v for k, v in checkpoint.items() if "stack_memory" not in k and "stack_ptr" not in k}
        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded {len(filtered_state)} parameters from {args.init_checkpoint} (skipped stack buffers)")
    
    print("Initialized TKSGeneralLMv7 with DPS")

    # Dataset
    full_dataset = DiscoveryDataset(args.data, tokenizer)
    
    # 90/10 Train/Val Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, full_dataset.pad_id)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, full_dataset.pad_id)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Starting Training: {len(train_dataset)} train, {len(val_dataset)} val, {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_novelty_bonus = 0
        total_depth_bonus = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Use mixed precision for faster training
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                # Forward pass with full trace to get DPS stats
                outputs = model(input_ids, attention_mask=mask, return_full_trace=True)
                
                logits = outputs['logits']
                
                # 1. Base LM Loss
                ce_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # 2. DPS / Novelty Reward
                dps_state = outputs['dps_state']
                novelty_reward = 0.05 * dps_state.tokens
                depth_gain = max(0, dps_state.p_max - config.dps_initial_p_max)
                depth_reward = 0.1 * depth_gain
                
                loss = ce_loss - novelty_reward - depth_reward
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_novelty_bonus += novelty_reward
            total_depth_bonus += depth_reward
            
            if batch_idx % 50 == 0:
                # Debug: show novelty history and memory stats
                nw_info = ""
                if hasattr(dps_state, 'novelty_history') and dps_state.novelty_history:
                    last_nw = dps_state.novelty_history[-1]
                    nw_info = f" | NW: {last_nw:.4f}"

                # Memory bank stats
                mem_info = ""
                mem_stats = outputs.get('memory_stats')
                if mem_stats:
                    mem_info = f" | Mem: {mem_stats['size']}/{int(mem_stats['size']/mem_stats['fill_ratio']) if mem_stats['fill_ratio'] > 0 else 1000}"

                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}) | "
                      f"Depth: {dps_state.p_max} | Tokens: {dps_state.tokens}{nw_info}{mem_info}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                
                with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    outputs = model(input_ids, attention_mask=mask)
                    logits = outputs['logits']
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Done | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Novelty Bonus: {total_novelty_bonus:.2f} | Depth Bonus: {total_depth_bonus:.2f}")
        
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.output_dir}/v7_discovery_model.pt")
    print(f"Saved model to {args.output_dir}/v7_discovery_model.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/v7_discovery_training.jsonl')
    parser.add_argument('--output-dir', default='checkpoints/v7_discovery')
    parser.add_argument('--init-checkpoint', default=None, help='Path to foundation model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    
    train(args)

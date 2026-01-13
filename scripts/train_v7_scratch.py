#!/usr/bin/env python3
"""
TKS v7 From-Scratch Training - Stage 9
Trains TKSGeneralLMv7 from random initialization on Diversified Data.
This isolates whether the issue is the checkpoint history or the architecture.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import json

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

class ScratchDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} diversified samples")

    def _encode(self, text: str) -> list:
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"Input: {item['input']}\nOutput: {item['output']}"
        ids = self._encode(full_text)
        input_ids = ids[:-1]
        labels = ids[1:]
        return {'input_ids': input_ids, 'labels': labels}

def collate_fn(batch):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    
    pad_id = 0
    
    for item in batch:
        ids = item['input_ids']
        labs = item['labels']
        curr_len = len(ids)
        pad_len = max_len - curr_len
        
        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
        attention_mask.append([1] * curr_len + [0] * pad_len)
        
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }

def train():
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()
    
    # Config - Fresh Start
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8,
        use_memory_bank=True
    )
    
    model = TKSGeneralLMv7(config).to(DEVICE)
    print("Initialized FRESH TKSGeneralLMv7")
    
    # Use the diversified dataset which has the best mix
    dataset = ScratchDataset("data/v7_diversified_training.jsonl", tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f"Starting From-Scratch Training: {len(dataset)} samples")
    
    for epoch in range(2):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                outputs = model(input_ids, attention_mask=mask)
                loss = loss_fn(outputs['logits'].view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
    os.makedirs("checkpoints/v7_scratch", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/v7_scratch/v7_scratch_model.pt")
    print("Scratch Model saved.")

if __name__ == "__main__":
    train()

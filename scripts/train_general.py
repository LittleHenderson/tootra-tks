"""
Train TKS General Model (v5) - ~20M Parameters
Uses BPE Tokenizer + Mixed Dataset
"""

import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import time
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_llm_core_v5 import TKSGeneralLM, TKSGeneralConfig

# Config
BATCH_SIZE = 8  # Reduced for larger model
EPOCHS = 10
LR = 3e-4
MAX_SEQ_LEN = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IGNORE_INDEX = -100
MASK_PROMPT_LOSS = True

if DEVICE == 'cpu':
    print("FATAL: GPU not detected. Aborting to protect CPU.")
    sys.exit(1)

class GeneralDataset(Dataset):
    def __init__(self, data_paths, tokenizer_path, max_len=MAX_SEQ_LEN):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        
        self.data = []
        for path in data_paths:
            if not os.path.exists(path):
                print(f"Warning: {path} not found")
                continue
                
            print(f"Loading {path}...")
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        # Extract text based on available fields
                        text = ""
                        if 'input' in item and 'output' in item:
                            text = f"{item['input']} <SEP> {item['output']}"
                        elif 'story' in item and 'equation' in item:
                            text = f"{item['story']} <SEP> {item['equation']}"
                        elif 'text' in item:
                            text = item['text']
                        elif 'prompt' in item and 'target' in item:
                            text = f"{item['prompt']} <SEP> {item['target']}"
                            
                        if text:
                            self.data.append(text)
                    except:
                        pass
        print(f"Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer.encode(text)
        ids = [self.bos_id] + enc.ids + [self.eos_id]
        
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            
        return torch.tensor(ids, dtype=torch.long)

def make_collate_fn(pad_id: int):
    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        padded = []
        masks = []

        for x in batch:
            pad_len = max_len - len(x)
            pad_tensor = torch.full((pad_len,), pad_id, dtype=torch.long)
            padded.append(torch.cat([x, pad_tensor]))
            masks.append(
                torch.cat([
                    torch.ones(len(x), dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool)
                ])
            )

        return torch.stack(padded), torch.stack(masks)

    return collate_fn

def train():
    print(f"Training on {DEVICE}")
    
    # 1. Setup Data
    data_files = [
        "output/rebalanced_mix_v5.jsonl",
        # v5 mix adds grounding drills and conflict dedupe.
    ]
    
    dataset = GeneralDataset(data_files, "tokenizer_v5.json")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(dataset.pad_id)
    )
    
    # 2. Setup Model
    config = TKSGeneralConfig(
        vocab_size=dataset.tokenizer.get_vocab_size(),
        hidden_dim=384,
        num_layers=12,
        num_scales=4
    )
    model = TKSGeneralLM(config).to(DEVICE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    # 3. Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        start = time.time()
        
        sep_id = dataset.tokenizer.token_to_id("<SEP>")
        pad_id = dataset.pad_id

        for i, (input_ids, attention_mask) in enumerate(loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            # Causal LM: Predict next token
            # Input: x[0..T-1], Target: x[1..T]
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:].clone()
            attention_mask = attention_mask[:, :-1]

            targets[targets == pad_id] = IGNORE_INDEX
            if MASK_PROMPT_LOSS and sep_id is not None:
                for row in range(input_ids.size(0)):
                    sep_positions = (input_ids[row] == sep_id).nonzero(as_tuple=False)
                    if sep_positions.numel() == 0:
                        continue
                    sep_pos = sep_positions[0].item()
                    cutoff = min(sep_pos, targets.size(1))
                    if cutoff > 0:
                        targets[row, :cutoff] = IGNORE_INDEX
            
            optimizer.zero_grad()
            
            out = model(inputs, attention_mask=attention_mask)
            logits = out['logits'] # [B, T, V]
            
            # Reshape for loss
            loss = loss_fn(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Step {i}/{len(loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Finished | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")
        
        # Save Checkpoint
        os.makedirs("checkpoints/v5_general", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/v5_general/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()

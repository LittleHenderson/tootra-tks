"""
Train TKS General Model (v5) on Reconciled Canon Data (Resumed)
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

BATCH_SIZE = 8

TARGET_EPOCHS = 200

LR = 3e-4

MAX_SEQ_LEN = 512

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RESUME_CHECKPOINT = None



if DEVICE == 'cpu':

    print("FATAL ERROR: GPU not detected.")

    print("Aborting training to protect CPU from thermal damage.")

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



def collate_fn(batch):

    max_len = max(len(x) for x in batch)

    padded = []

    for x in batch:

        pad_len = max_len - len(x)

        padded.append(torch.cat([x, torch.tensor([0] * pad_len, dtype=torch.long)]))

    return torch.stack(padded), None



class Leaderboard:

    def __init__(self):

        self.history = []



    def update(self, epoch, train_loss, lr, params):

        self.history.append({

            'epoch': epoch,

            'train_loss': train_loss,

            'lr': lr,

            'params': params

        })



    def display(self):

        print("\n" + "=" * 80)

        print("                              REAL-TIME LEADERBOARD")

        print("=" * 80)

        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'LR':>12} | {'Params':>15}")

        print("-" * 80)

        for e in self.history:

            print(f"{e['epoch']:>6} | {e['train_loss']:>12.4f} | {e['lr']:>12.2e} | {e['params']:>15,}")

        print("=" * 80 + "\n")



def train():

    print("=" * 60)

    print("STARTING FRESH TRAINING ON RECONCILED CANON DATA")

    print("=" * 60)

    print(f"Device: {DEVICE}")



    # Use reconciled data

    data_files = [

        "output/rebalanced_mix_v5.jsonl"

    ]



    dataset = GeneralDataset(data_files, "tokenizer_v5.json")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)



    # Setup Model

    config = TKSGeneralConfig(

        vocab_size=dataset.tokenizer.get_vocab_size(),

        hidden_dim=384,

        num_layers=12,

        num_scales=4

    )

    model = TKSGeneralLM(config).to(DEVICE)

    

    start_epoch = 0

    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):

        print(f"Resuming from {RESUME_CHECKPOINT}...")

        ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE, weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'])

        start_epoch = ckpt.get('epoch', 0)

        print(f"Resuming from epoch {start_epoch}")

    else:

        print("Starting from scratch (no resume).")



    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model Parameters: {num_params:,}")



    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    

    # Leaderboard

    leaderboard = Leaderboard()



    # Training

    model.train()

    os.makedirs("checkpoints/v5_reconciled", exist_ok=True)



    for epoch in range(start_epoch, TARGET_EPOCHS):

        total_loss = 0

        start = time.time()

        

        current_lr = optimizer.param_groups[0]['lr']



        for i, (input_ids, _) in enumerate(loader):

            input_ids = input_ids.to(DEVICE)



            inputs = input_ids[:, :-1]

            targets = input_ids[:, 1:]



            optimizer.zero_grad()



            out = model(inputs)

            logits = out['logits']



            loss = loss_fn(logits.reshape(-1, config.vocab_size), targets.reshape(-1))



            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()



            total_loss += loss.item()



            if i % 100 == 0:

                print(f"  Epoch {epoch+1}/{TARGET_EPOCHS} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")



        avg_loss = total_loss / len(loader)

        elapsed = time.time() - start

        

        # Update Leaderboard

        leaderboard.update(epoch + 1, avg_loss, current_lr, num_params)

        leaderboard.display()

        

        print(f"Epoch {epoch+1}/{TARGET_EPOCHS} completed in {elapsed:.1f}s")



        # Save checkpoint

        torch.save({

            'epoch': epoch + 1,

            'model_state_dict': model.state_dict(),

            'train_loss': avg_loss,

        }, f"checkpoints/v5_reconciled/epoch_{epoch+1}.pt")



    print("\n" + "=" * 60)

    print("TRAINING COMPLETE")

    print("=" * 60)

    print(f"Final model: checkpoints/v5_reconciled/epoch_{TARGET_EPOCHS}.pt")

if __name__ == "__main__":
    train()
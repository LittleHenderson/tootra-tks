"""
TKS LLM v6 Training Script - Reasoning Upgrade

Fine-tunes the v5 "Knowledge" model on RPM Chain-of-Thought data.
Integrates the Recurrent NJT architecture with Recursion Stack.
"""

import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tokenizers import Tokenizer
import time
import argparse

# Import v6 Core
from tks_llm_core_v6 import TKSGeneralLMv6, TKSGeneralConfig

class V6CoTDataset(Dataset):
    """Handles RPM Chain-of-Thought data."""
    def __init__(self, data_path, tokenizer_path, max_len=512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.pad_id = self.tokenizer.token_to_id("<PAD>") or 0
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        
        self.data = []
        print(f"Loading CoT data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Combine Goal, Trace, and Final Output into one sequence
                text = f"Goal: {item['input']} <SEP> Reasoning: {item['output']} <SEP> Result: {item['tks_notation']}"
                self.data.append(text)
        print(f"Loaded {len(self.data)} reasoning samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer.encode(text)
        ids = [self.bos_id] + enc.ids + [self.eos_id] if self.bos_id and self.eos_id else enc.ids
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

def train():
    parser = argparse.ArgumentParser(description="TKS v6 Reasoning Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5) # Lower LR for fine-tuning
    args = parser.parse_args()

    # CUDA enforcement - GPU ONLY
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. No CPU training allowed.")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Training on {device}")

    # Ensure checkpoint directory exists
    Path("checkpoints").mkdir(exist_ok=True)

    # 1. Setup Model & Config
    tokenizer_path = "tokenizer_v5.json"
    dataset = V6CoTDataset("data/v6_cot_training.jsonl", tokenizer_path)
    val_dataset = V6CoTDataset("data/v6_cot_validation.jsonl", tokenizer_path)
    
    config = TKSGeneralConfig(
        vocab_size=dataset.tokenizer.get_vocab_size(),
        use_njt=True,
        njt_use_nested_memory=True
    )
    model = TKSGeneralLMv6(config).to(device)

    # 2. Migration: Load v5 Weights
    v5_path = "checkpoints/v5_final.pt"
    if Path(v5_path).exists():
        print(f"Migrating knowledge from {v5_path}...")
        v5_state = torch.load(v5_path, map_location=device, weights_only=False)
        msg = model.load_state_dict(v5_state, strict=False)
        print(f"Migration complete. Missing keys (new v6 components): {len(msg.missing_keys)}")

        # Reset NJT thresholds (v5 weights may have old values)
        for block in model.blocks:
            if block.njt is not None and hasattr(block.njt.njt_v6, 'push_pop_gate'):
                gate = block.njt.njt_v6.push_pop_gate
                if gate is not None:
                    gate.push_threshold.data.fill_(0.3)
                    gate.pop_threshold.data.fill_(0.5)
        print("Reset NJT thresholds to v6 defaults (push=0.3, pop=0.5)")
    else:
        print("Warning: v5_final.pt not found. Starting from scratch.")

    # 3. Training Setup
    def collate_with_mask(batch, pad_id):
        """Collate with padding and attention mask."""
        padded = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
        mask = (padded != pad_id).float()
        return padded, mask

    collate_fn = lambda b: collate_with_mask(b, dataset.pad_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * args.epochs)

    print("Starting v6 Fine-tuning (The Reasoning Upgrade)...")
    # torch.autograd.set_detect_anomaly(True)  # DEBUG: Disabled for perf

    global_step = 0
    steps_per_epoch = len(dataloader)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()

            # Log every step for real-time monitoring
            should_log = True 
            output = model(input_ids, attention_mask=attention_mask, step=global_step, return_full_trace=should_log)
            logits = output["logits"]

            # Loss Calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)(
                shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
            )

            aux_loss = output.get("routing_aux_loss", 0)
            total_loss = ce_loss + aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += ce_loss.item()

            # Granular Logging (Every single step)
            if True:
                # Safe trace extraction
                depth = 0
                halt_prob = 0.0
                trace = output.get("trace", {})
                njt_traces = trace.get("njt", [])
                
                if njt_traces and isinstance(njt_traces[-1], dict):
                    njt_trace = njt_traces[-1]
                    depth = njt_trace.get("max_depth_reached", 0)
                    hps = njt_trace.get("halt_probs", [])
                    if hps:
                        hp = hps[-1]
                        halt_prob = hp.item() if hasattr(hp, 'item') else float(hp)

                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch + (global_step/steps_per_epoch):.2f} | Step {global_step}: "
                      f"Loss={ce_loss.item():.4f}, Depth={depth}, Halt={halt_prob:.2f}, LR={current_lr:.2e}")

            if global_step > 0 and global_step % 5000 == 0:
                torch.save(model.state_dict(), f"checkpoints/v6_step_{global_step}.pt")

            global_step += 1

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = input_ids.clone()

                output = model(input_ids, attention_mask=attention_mask, return_full_trace=False)
                logits = output["logits"]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce_loss = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)(
                    shift_logits.view(-1, config.vocab_size), shift_labels.view(-1)
                )
                val_loss += ce_loss.item()

        avg_train_loss = epoch_loss / len(dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"\n=== Epoch {epoch+1}/{args.epochs} Complete ===")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/v6_best.pt")
            print(f"New best model saved (val_loss={avg_val_loss:.4f})")
        print()

    torch.save(model.state_dict(), "checkpoints/v6_final.pt")
    print(f"v6 Training Complete. Best val_loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()

#!/usr/bin/env python3
"""
TKS LLM v5 Training Script

CRITICAL: Pass `step=current_step` to model.forward() for stable routing!

*** GPU REQUIRED - CPU TRAINING IS DISABLED ***
This script will EXIT if no CUDA GPU is available.
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

# =============================================================================
# GPU ENFORCEMENT - THIS RUNS IMMEDIATELY ON IMPORT
# =============================================================================
def _enforce_gpu_required():
    """Enforce GPU requirement. Called immediately on script load."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("FATAL ERROR: CUDA GPU NOT AVAILABLE - TRAINING BLOCKED")
        print("=" * 70)
        print("")
        print("  This training script REQUIRES a CUDA-capable GPU.")
        print("  CPU training is DISABLED because:")
        print("    1. It's too slow for meaningful training")
        print("    2. It causes CPU overheating")
        print("    3. Quality of training suffers")
        print("")
        print("  To fix this:")
        print("    1. Ensure you have an NVIDIA GPU installed")
        print("    2. Install CUDA toolkit (https://developer.nvidia.com/cuda-downloads)")
        print("    3. Install PyTorch with CUDA support:")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("")
        print("  DO NOT modify this script to allow CPU training.")
        print("  DO NOT create a new script that bypasses this check.")
        print("")
        print("=" * 70)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Verified: {gpu_name} ({gpu_mem:.1f} GB)")
    return torch.device("cuda")

# ENFORCE GPU NOW - before any other code runs
DEVICE = _enforce_gpu_required()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.v5_recommended import get_v5_config
from tks_llm_core_v5 import TKSGeneralLM

# Config
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4
MAX_SEQ_LEN = 256  # Can use higher seq len with GPU
MAX_STEPS = 50  # Limit for demonstration/verification

class V5Dataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len=MAX_SEQ_LEN):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        if self.pad_id is None: self.pad_id = 0
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        
        self.data = []
        if Path(data_path).exists():
            print(f"Loading {data_path}...")
            with open(data_path, 'r', encoding='utf-8') as f:
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
                            if len(self.data) >= 1000: break # Increased limit for longer runs
                    except:
                        pass
        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer.encode(text)
        ids = [self.bos_id] + enc.ids + [self.eos_id] if self.bos_id and self.eos_id else enc.ids
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

import argparse

def train():
    parser = argparse.ArgumentParser(description="TKS v5 Training")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    args = parser.parse_args()
    
    # Data
    data_path = "output/rebalanced_mix_v5.jsonl"
    tokenizer_path = "tokenizer_v5.json"
    
    # Fallback if data missing
    if not Path(data_path).exists():
        print(f"Warning: {data_path} not found.")
        # Try to find any jsonl
        import glob
        jsonls = glob.glob("output/*.jsonl")
        if jsonls:
            data_path = jsonls[0]
            print(f"Using {data_path} instead.")
        else:
             print("No data found. Exiting.")
             return

    dataset = V5Dataset(data_path, tokenizer_path)
    if len(dataset) == 0:
        print("Dataset empty.")
        return

    def collate_wrapper(batch):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=dataset.pad_id)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_wrapper)

    # Calculate steps
    steps_per_epoch = len(dataloader)
    if args.epochs is not None:
        max_steps = args.epochs * steps_per_epoch
    elif args.steps is not None:
        max_steps = args.steps
    else:
        max_steps = 50 # Default
        
    print(f"Training on {DEVICE} for {max_steps} steps (~{max_steps/steps_per_epoch:.1f} epochs)")

    # Model
    print("Initializing model...")
    config = get_v5_config(
        size="base",
        use_stable_routing=True,
        use_attractor=True,
        use_rpm=True,
        # NJT (Noetic Judgment Transistor) circuits
        use_njt=True,
        njt_num_transistors=10,
        njt_use_hysteresis=True,
        njt_use_rhythm=True,
    )
    # Adjust vocab size to match tokenizer
    config.vocab_size = dataset.tokenizer.get_vocab_size()
    
    model = TKSGeneralLM(config)
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=100000) # Increased T_max for longer runs

    global_step = 0
    
    print("Starting training loop...")
    model.train()
    
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Save step 0
    torch.save(model.state_dict(), "checkpoints/v5_step_0.pt")
    
    for epoch in range(1000): # Allow many epochs, use global_step to exit
        for batch in dataloader:
            if global_step >= max_steps: break
            
            input_ids = batch.to(DEVICE)
            # Labels = input_ids (causal LM)
            labels = input_ids.clone()
            
            # CRITICAL: Pass step for stable routing temperature annealing
            output = model(input_ids, step=global_step)
            
            logits = output["logits"]
            
            # Simple Causal Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)
            ce_loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
            
            aux_loss = output.get("routing_aux_loss", 0)
            if isinstance(aux_loss, torch.Tensor):
                total_loss = ce_loss + aux_loss
            else:
                total_loss = ce_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if global_step % 5 == 0:
                metrics = output.get("routing_metrics", {})
                temp = metrics.get('routing_temperature', 'N/A')
                if isinstance(temp, torch.Tensor): temp = temp.item()
                entropy = metrics.get('routing_entropy_mean', 'N/A')
                if isinstance(entropy, torch.Tensor): entropy = entropy.item()
                
                current_lr = scheduler.get_last_lr()[0]
                current_epoch = global_step / steps_per_epoch
                
                print(f"Epoch {current_epoch:.2f} | Step {global_step}: Loss={ce_loss.item():.4f}, Aux={aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.4f}, Temp={temp:.3f}, Entropy={entropy:.3f}, LR={current_lr:.2e}")
            
            # Periodic Checkpointing
            if global_step > 0 and global_step % 5000 == 0:
                ckpt_path = f"checkpoints/v5_step_{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            global_step += 1
            
        if global_step >= max_steps: break

    # Final trace sample
    print("\nGenerating trace sample...")
    model.eval()
    with torch.no_grad():
        sample = input_ids[0:1] # Take first from last batch
        output = model(sample, return_full_trace=True)
        trace = output.get("trace", {})
        if "routing" in trace:
             print(f"Routing weights shape: {trace['routing'][0].shape}")
        if "canonical" in trace:
             print(f"Attractor output shape: {trace['canonical']['attractor']['attractor'].shape}")
    
    # Save Final
    torch.save(model.state_dict(), "checkpoints/v5_final.pt")
    print("Training complete. Checkpoint saved.")

if __name__ == "__main__":
    train()

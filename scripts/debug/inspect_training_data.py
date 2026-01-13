"""
TKS Training Data Inspector

Debugs the Tokenizer and Dataset to explain 0.0000 loss.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

# Import from training script (assuming we fixed imports)
from scripts.train_cuda import TKSDataset
from src.tks.core.tokenizer import TKSTokenizer

def inspect():
    data_path = "datasets/jsonl/tks_v0_training_mix.jsonl"
    tok_path = "checkpoints/v0/tokenizer.json"
    
    print(f"Loading Tokenizer: {tok_path}")
    if not os.path.exists(tok_path):
        print("Tokenizer not found. Run training first to generate it.")
        return

    tokenizer = TKSTokenizer.load(tok_path)
    print(f"Vocab Size: {tokenizer.actual_vocab_size}")
    
    print(f"\nLoading Dataset: {data_path}")
    dataset = TKSDataset(data_path, tokenizer, seq2seq=True)
    
    print(f"Dataset Size: {len(dataset)}")
    
    # Inspect first few batches
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    for i, batch in enumerate(loader):
        if i >= 1: break
        
        inputs = batch['input_ids']
        targets = batch['targets']
        mask = batch.get('loss_mask')
        
        print(f"\n--- Batch {i} ---")
        for j in range(len(inputs)):
            print(f"\nSample {j}:")
            
            # Decode Input
            input_ids = inputs[j].tolist()
            decoded_input = tokenizer.decode(input_ids, skip_special=False)
            print(f"Decoded Input (Full): {decoded_input}")
            
            # Check Mask
            if mask is not None:
                m = mask[j].tolist()
                active_tokens = sum(m)
                print(f"Active Mask Tokens: {active_tokens}/{len(m)}")
                
                # Show what is being predicted
                masked_targets = [targets[j][k].item() for k in range(len(m)) if m[k] == 1.0]
                decoded_targets = tokenizer.decode(masked_targets, skip_special=False)
                print(f"Target Tokens (Loss Active): {decoded_targets}")
                
                if active_tokens == 0:
                    print("⚠️ WARNING: Loss mask is empty! Model learns nothing.")

if __name__ == "__main__":
    inspect()

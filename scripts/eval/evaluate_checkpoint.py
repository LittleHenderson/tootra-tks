"""
TKS Checkpoint Evaluator (Agent Eval)

Evaluates a model checkpoint against a specific dataset.
Reports Exact Match Accuracy.
"""

import torch
import json
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

# Re-use Tokenizer/Dataset from training script logic (simplified)
from scripts.train_cuda import SimpleTokenizer, TKSDataset, TransformerModel
from src.tks.core.tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

def evaluate(checkpoint_path, data_path, batch_size=8):
    print(f"Evaluating {checkpoint_path} on {data_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Tokenizer (Must match training)
    tokenizer = SimpleTokenizer(max_length=256)
    
    # Load Dataset
    dataset = TKSDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load Model
    # We need to reconstruct the model exactly as it was trained
    # The checkpoint has 'config' which tells us structure
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check model type from config or infer
    config_dict = checkpoint.get("config", {})
    
    # Init Model (Assume TKSNoeticLM v4 as per v0 training)
    # Note: If train_cuda used TKSNoeticLMConfig, we use that.
    model_config = TKSNoeticLMConfig(
        vocab_size=tokenizer.actual_vocab_size,
        num_layers=config_dict.get("num_layers", 4)
    )
    model = TKSNoeticLM(model_config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    total = 0
    correct = 0
    
    print("\n--- Starting Evaluation ---")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward
            # TKSNoeticLM returns dict or logits
            output = model(input_ids)
            logits = output['logits'] if isinstance(output, dict) else output
            
            # Greedy Decode (Next Token Prediction)
            # For strict accuracy, we compare per-token or perplexity.
            # But "Exact Match" usually requires generation.
            # Since this is a simple LM training loop, we check Next Token Accuracy on the target sequence.
            
            preds = logits.argmax(dim=-1)
            
            # Mask padding (0)
            mask = targets != 0
            match = (preds == targets) & mask
            
            # Batch accuracy
            correct_tokens = match.sum().item()
            total_tokens = mask.sum().item()
            
            correct += correct_tokens
            total += total_tokens
            
            if i % 10 == 0:
                print(f"Batch {i}: {correct_tokens}/{total_tokens} tokens correct")

    acc = correct / total if total > 0 else 0
    print(f"\nResults:")
    print(f"Total Tokens: {total}")
    print(f"Correct Tokens: {correct}")
    print(f"Token Accuracy: {acc:.4%}")
    
    if acc > 0.9:
        print("✅ PASS: Strong Generalization")
    elif acc > 0.7:
        print("⚠️ WARN: Moderate Generalization")
    else:
        print("❌ FAIL: Poor Generalization (Overfitting likely)")

if __name__ == "__main__":
    evaluate("checkpoints/v0/final_model.pt", "datasets/jsonl/tks_holdout_v1.jsonl")

"""
Diagnostic script for Mapping tasks.
Logs top-5 logits at element positions.
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tks.core.tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

def load_model_and_tokenizer(model_path, tokenizer_path, device):
    # Load tokenizer
    with open(tokenizer_path, 'r') as f:
        vocab = json.load(f)
    print(f"Loaded tokenizer (vocab size: {len(vocab)})")
    
    # Load weights first to get config
    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load config from checkpoint if available
    if 'config' in ckpt:
        print("Loading config from checkpoint...")
        config_dict = ckpt['config']
        valid_keys = TKSNoeticLMConfig.__dataclass_fields__.keys()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = TKSNoeticLMConfig(**filtered_config)
    else:
        config = TKSNoeticLMConfig(vocab_size=len(vocab))
        
    model = TKSNoeticLM(config).to(device)
    
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    return model, vocab, config.max_seq_len

def load_data(data_path, limit=50):
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            if item.get('task_type') == 'story_to_equation':
                samples.append(item)
                if len(samples) >= limit:
                    break
    print(f"Loaded {len(samples)} mapping samples")
    return samples

def tokenize(text, vocab, bos_id, eos_id, unk_id, max_len=None):
    tokens = []
    i = 0
    n = len(text)
    
    all_tokens = list(vocab.keys())
    all_tokens.sort(key=len, reverse=True)
    
    ids = [bos_id]
    
    while i < n:
        matched = False
        for t in all_tokens:
            if text.startswith(t, i):
                ids.append(vocab[t])
                i += len(t)
                matched = True
                break
        if not matched:
            c = text[i]
            ids.append(vocab.get(c, unk_id))
            i += 1
            
    ids.append(eos_id)
    
    if max_len:
        ids = ids[:max_len]
        
    return ids

def decode(ids, id2tok):
    return "".join([id2tok.get(i, "") for i in ids if i not in [0, 1, 2, 3]])

def run_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = 'checkpoints/v3_leaderboard/best_model.pt'
    tokenizer_path = 'checkpoints/v2_fixed/tokenizer.json'
    data_path = 'output/teacher_story_eq_train.jsonl'
    
    model, vocab, max_seq_len = load_model_and_tokenizer(model_path, tokenizer_path, device)
    samples = load_data(data_path)
    
    id2tok = {v: k for k, v in vocab.items()}
    bos_id = vocab.get('<BOS>', 2)
    eos_id = vocab.get('<EOS>', 3)
    unk_id = vocab.get('<UNK>', 1)
    sep = " => "
    
    print("\n" + "="*70)
    print("DIAGNOSTIC: MAPPING (Top-5 Logits)")
    print("="*70)
    
    for idx, sample in enumerate(samples[:10]):
        prompt = sample['input']
        target = sample['target']
        
        full_prompt = prompt + sep
        # Truncate prompt more aggressively to ensure we have room for generation
        max_prompt_len = max_seq_len - 50 
        input_ids = tokenize(full_prompt, vocab, bos_id, eos_id, unk_id, max_len=max_prompt_len)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        print(f"\nSample {idx+1}:")
        print(f"Target: {target}")
        
        target_ids = tokenize(target, vocab, bos_id, eos_id, unk_id)
        target_ids = [t for t in target_ids if t not in [bos_id, eos_id]]
        
        print("Step-by-step prediction:")
        
        curr_input = input_tensor
        
        for i, true_token_id in enumerate(target_ids):
            if curr_input.shape[1] >= max_seq_len:
                print("  ... (max seq len reached)")
                break
                
            with torch.no_grad():
                output = model(curr_input)
                logits = output['logits'][:, -1, :]
                
            probs = F.softmax(logits, dim=-1)
            topk_vals, topk_indices = torch.topk(probs, 5)
            
            true_token = id2tok.get(true_token_id, f"<{true_token_id}>")
            
            top5_str = []
            for val, tid in zip(topk_vals[0], topk_indices[0]):
                tok = id2tok.get(tid.item(), f"<{tid.item()}>")
                top5_str.append(f"{tok}({val:.2f})")
            
            print(f"  Pos {i} Expected: '{true_token}' | Top-5: {', '.join(top5_str)}")
            
            next_input = torch.tensor([[true_token_id]], device=device)
            curr_input = torch.cat([curr_input, next_input], dim=1)

if __name__ == '__main__':
    run_diagnostic()
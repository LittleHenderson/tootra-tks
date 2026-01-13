"""
TKS Robustness Evaluator (Agent Eval)

Metrics:
- Exact Match (EM)
- Syntax Validity (Parenthesis Balance)
"""

import torch
import json
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

# Re-use Tokenizer/Dataset
from src.tks.core.tokenizer import TKSTokenizer
from src.tks.core.tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

# Re-implement simple dataset logic to avoid import dependency hell if script moves
class SimpleEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.entries = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))
        
        # Auto-detect format
        self.sep_token = " => "

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Robust Extraction
        inp = entry.get('input', entry.get('prompt', ''))
        tgt = entry.get('output', entry.get('target', ''))
        
        full_text = inp + self.sep_token + tgt
        input_ids = self.tokenizer.tokenize(full_text)
        targets = input_ids[1:] + [self.tokenizer.vocab["<PAD>"]]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'raw_input': inp,
            'raw_target': tgt
        }

def validate_syntax(expr: str) -> bool:
    return expr.count("(") == expr.count(")")

def generate_autoregressive(model, tokenizer, prompt_ids, max_new_tokens=50, device='cuda'):
    """Generate tokens autoregressively from a prompt.

    NOTE: After training with proper attention masking, PAD tokens are invisible
    during attention, so the model should not predict PAD. We still mask PAD logits
    as a safety measure.
    """
    model.eval()
    generated = prompt_ids.clone()

    pad_id = tokenizer.vocab.get("<PAD>", 0)
    eos_id = tokenizer.vocab.get("<EOS>", tokenizer.vocab.get("</s>", -1))

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Create attention mask (all 1s for real tokens)
            attention_mask = torch.ones_like(generated)

            output = model(generated, attention_mask=attention_mask)
            logits = output['logits'] if isinstance(output, dict) else output
            # Get logits for last position
            next_logits = logits[:, -1, :]

            # Mask PAD logit to prevent generating PAD
            next_logits[:, pad_id] = float('-inf')

            next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Stop on EOS
            if next_token.item() == eos_id:
                break

            generated = torch.cat([generated, next_token], dim=1)

    return generated


def evaluate(checkpoint_path, data_path, batch_size=1):
    print(f"Robustness Eval: {checkpoint_path} on {data_path}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizer from Checkpoint Dir
    ckpt_dir = os.path.dirname(checkpoint_path)
    tok_path = os.path.join(ckpt_dir, "tokenizer.json")

    if not os.path.exists(tok_path):
        print(f"FATAL: Tokenizer not found at {tok_path}")
        return

    tokenizer = TKSTokenizer.load(tok_path)
    print(f"Tokenizer Loaded: Vocab {tokenizer.actual_vocab_size}")

    # Load data entries directly for autoregressive eval
    entries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load Config
    config = TKSNoeticLMConfig()
    if "config" in checkpoint:
        c = checkpoint["config"]
        config.vocab_size = c.get("vocab_size", tokenizer.actual_vocab_size)
        config.num_layers = c.get("num_layers", 4)
    else:
        config.vocab_size = tokenizer.actual_vocab_size

    model = TKSNoeticLM(config)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    total = 0
    em_count = 0
    valid_count = 0

    sep_token = " => "

    print("\n--- Starting Autoregressive Evaluation ---")

    for i, entry in enumerate(entries):
        inp = entry.get('input', entry.get('prompt', ''))
        target_str = entry.get('output', entry.get('target', '')).strip()

        # Tokenize just the input + separator (model completes after separator)
        prompt_text = inp + sep_token
        prompt_ids = tokenizer.tokenize(prompt_text)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Generate autoregressively
        full_ids = generate_autoregressive(model, tokenizer, prompt_tensor, max_new_tokens=50, device=device)
        full_text = tokenizer.decode(full_ids[0].tolist(), skip_special=True)

        # Extract generated part (after separator)
        if sep_token.strip() in full_text:
            pred_answer = full_text.split(sep_token.strip())[-1].strip()
        else:
            pred_answer = full_text[len(prompt_text):].strip()

        # Metrics
        if pred_answer == target_str:
            em_count += 1

        if validate_syntax(pred_answer):
            valid_count += 1

        total += 1

        if i < 5:
            print(f"\nSample {i}:")
            print(f"  Input:  {inp}")
            print(f"  Target: {target_str}")
            print(f"  Generated: {pred_answer}")
            print(f"  Match:  {pred_answer == target_str}")

    print(f"\nResults:")
    print(f"Total Samples:   {total}")
    print(f"Exact Match:     {em_count}/{total} ({em_count/total:.2%})")
    print(f"Syntax Valid:    {valid_count}/{total} ({valid_count/total:.2%})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.data)
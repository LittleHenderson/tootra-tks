"""
TKS Capabilities Evaluation - Project MAJIK
Tests the model on 20 TKS-specific questions (Easy to Mid).
Verifies if the "TKS Simulator" capabilities are intact.
"""

import sys
import os
import torch
import torch.nn as nn
from typing import List, Dict
from tokenizers import Tokenizer

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TKS_QUESTIONS = [
    # EASY (Definitions & Basic Mappings)
    "Define the 12 Prerequisites of TKS.",
    "What corresponds to 'Spiritual Desire' in the 12-slot vector?",
    "Explain the role of the 11W (Physical Wisdom) prerequisite.",
    "Map the 4 Worlds to the standard Kabbalistic tree.",
    "What is the difference between NJT+ and NJT-?",
    "Define 'Noetic 9' in the context of halting.",
    "What does 'RECURSE' mean in a TKS trace?",
    "Identify the prerequisite associated with 'Emotional Power'.",
    
    # MEDIUM (Process & Logic)
    "Trace the resolution of a 'Lack of Money' block using TKS.",
    "How does the model handle a 2W (Spiritual Wisdom) deficit?",
    "Explain the 'Earned Depth' mechanism.",
    "What happens when the Memory Bank is full?",
    "Describe the interaction between the Recursion Stack and the Push/Pop Gate.",
    "Trace a 'Learning a new skill' goal through the Mental World.",
    "How does 'Novelty Weight' (NW) influence recursion depth?",
    "Explain the concept of 'Noetic Fractal' structure.",
    "What is the function of the 'Gradation Codec'?",
    "How do 'Regulators' (FM/ACBE) maintain system balance?",
    "Trace the inversion of a 'Fear' impulse into 'Courage'.",
    "Explain the 'Lightning Flash' path through the 12 Prerequisites."
]

def main():
    print(f"Using device: {DEVICE}")
    
    # Load model
    model_path = "checkpoints/v7_multidomain/v7_multidomain_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    vocab_size = checkpoint['lm_head.weight'].shape[0]
    
    # Config (must match training)
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        use_dps_gating=True,
        dps_hidden_dim=64,
        dps_max_depth=8,
        njt_max_recursion_depth=4, # Matches checkpoint
        use_memory_bank=True,
        memory_bank_size=1000,
        use_impact_only_nw=True
    )
    
    model = TKSGeneralLMv7(config).to(DEVICE)
    
    # Filter buffers
    state_dict = {k: v for k, v in checkpoint.items() if "stack_memory" not in k and "stack_ptr" not in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded.")

    # Tokenizer
    tokenizer_path = "tokenizer_v5.json"
    if not os.path.exists(tokenizer_path):
        print(f"Error: {tokenizer_path} not found.")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    if bos_id is None or eos_id is None:
        print("Error: tokenizer missing <BOS>/<EOS> tokens.")
        return

    def generate(prompt, max_len=200):
        # Format input as trained
        formatted_input = f"Input: {prompt}\n"
        enc = tokenizer.encode(formatted_input)
        ids = [bos_id] + enc.ids
        input_tensor = torch.tensor([ids], device=DEVICE)
        
        generated = input_tensor
        
        with torch.no_grad():
            for _ in range(max_len):
                outputs = model(generated)
                next_token_logits = outputs['logits'][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated = torch.cat((generated, next_token), dim=1)
                
                if next_token.item() == eos_id:
                    break
        
        # Decode
        decoded_ids = [tid for tid in generated[0].tolist() if tid not in (bos_id, eos_id)]
        return tokenizer.decode(decoded_ids)

    print("\n--- TKS CAPABILITIES TEST ---")
    
    for i, q in enumerate(TKS_QUESTIONS):
        print(f"\nQ{i+1}: {q}")
        response = generate(q)
        print(f"Model: {response}")
        print("-" * 60)

if __name__ == "__main__":
    main()

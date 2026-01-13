
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def sanity_check():
    # Load
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    checkpoint_path = "checkpoints/v7_discovery/v7_discovery_model.pt"
    
    # Config matching training
    config = TKSGeneralConfigV7(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8
    )
    
    model = TKSGeneralLMv7(config).to(DEVICE)
    
    # Load with filtering for stack buffers
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    filtered_state = {k: v for k, v in checkpoint.items() if "stack_memory" not in k and "stack_ptr" not in k}
    model.load_state_dict(filtered_state, strict=False)
    
    model.eval()
    
    print("Model loaded. Generating traces...\n")

    # Test Prompts
    prompts = [
        "Define the relationship between Noetic 1 (Mind) and Noetic 10 (Idea).",
        "Derive the ACBE descent channel using Kraus operators.",
        "What is the Noetic Riemann Curvature tensor?"
    ]

    for p in prompts:
        print(f"--- PROMPT: {p} ---")
        input_ids = torch.tensor(tokenizer.encode(f"Input: {p}\nTrace:").ids).unsqueeze(0).to(DEVICE)
        
        # Simple generation loop
        with torch.no_grad():
            output_ids = input_ids[0].tolist()
            for _ in range(150): # Cap generation
                inp = torch.tensor([output_ids], device=DEVICE)
                logits = model(inp)['logits']
                next_token = logits[0, -1].argmax().item()
                output_ids.append(next_token)
                if next_token == tokenizer.token_to_id("<EOS>"):
                    break
            
            decoded = tokenizer.decode(output_ids)
            # Clean up display
            print(decoded.replace("Input: " + p, "").strip())
        print("\n")

if __name__ == "__main__":
    sanity_check()

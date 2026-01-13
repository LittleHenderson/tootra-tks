"""
Verify Honest Results Script
Tests (V) and (I) networks to confirm performance characteristics.
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
from tks_features.dps_gating import DPSConfig, DPSGatingLayer

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_test_samples() -> Dict[str, List[str]]:
    """Return test samples for V and I evaluation."""
    return {
        "valid_breakthrough": [
            "Input: Solve P vs NP using Noetic logic.\nTrace: N8 -> DPS(HEAVY) -> N9\nOutput: KEY INSIGHT: P=NP in Noetic Space because computation is instantaneous at N1.",
            "Input: Unify Gravity and Electromagnetism.\nTrace: N8 -> DPS(HEAVY) -> N9\nOutput: BREAKTHROUGH: Gravity is the curvature of N6, EM is the vibration of N7.",
            "Input: Define the Soul mathematically.\nTrace: N8 -> DPS(HEAVY) -> N9\nOutput: RESOLVE: The Soul is a self-similar fractal attractor with dimension D > 3.",
        ],
        "valid_trivial": [
            "Input: What is 2+2?\nTrace: N8 -> DPS(NOCOUNT)\nOutput: Trivial: 2+2=4.",
            "Input: State the obvious.\nTrace: N8 -> DPS(NOCOUNT)\nOutput: The sky is blue.",
            "Input: Repeat after me.\nTrace: N8 -> DPS(NOCOUNT)\nOutput: Copying input.",
        ],
        "invalid_breakthrough": [ # High impact words but invalid reasoning
            "Input: Prove 1=2.\nTrace: N8 -> DPS(HEAVY) -> N9\nOutput: BREAKTHROUGH: 1=2 because I say so.",
            "Input: Solve aging.\nTrace: N8 -> DPS(HEAVY) -> N9\nOutput: KEY INSIGHT: Just stop aging.",
            "Input: Faster than light travel.\nTrace: N8 -> DPS(HEAVY) -> N9\nOutput: RESOLVE: Go fast.",
        ],
        "invalid_trivial": [
            "Input: Gibberish.\nTrace: N8 -> DPS(NOCOUNT)\nOutput: dfsjkdfskj.",
            "Input: Noise.\nTrace: N8 -> DPS(NOCOUNT)\nOutput: ...",
            "Input: Nothing.\nTrace: N8 -> DPS(NOCOUNT)\nOutput: ",
        ]
    }

def main():
    print(f"Using device: {DEVICE}")
    
    # Load model
    model_path = "checkpoints/v7_discovery/v7_discovery_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load checkpoint to infer config (vocab size)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    vocab_size = checkpoint['lm_head.weight'].shape[0]
    
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        use_dps_gating=True,
        dps_hidden_dim=64,
        dps_max_depth=8,
        njt_max_recursion_depth=4  # Checkpoint was trained with depth 4
    )
    
    model = TKSGeneralLMv7(config).to(DEVICE)
    
    # Filter out batch-dependent buffers
    state_dict = checkpoint
    keys_to_remove = [k for k in state_dict.keys() if "stack_memory" in k or "stack_ptr" in k]
    for k in keys_to_remove:
        del state_dict[k]
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded.")
    
    # Get DPS layer (from first block that has it)
    dps_layer = None
    for block in model.blocks:
        if block.has_dps and block.dps_layer is not None:
            dps_layer = block.dps_layer
            break
            
    if dps_layer is None:
        print("Error: No DPS layer found in model blocks.")
        return

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

    def tokenize(text):
        enc = tokenizer.encode(text)
        ids = [bos_id] + enc.ids + [eos_id]
        return torch.tensor([ids], device=DEVICE)

    # Evaluation
    samples = get_test_samples()
    
    results = {}
    
    print("\n--- EVALUATION ---")
    
    for category, texts in samples.items():
        v_scores = []
        i_scores = []
        nw_scores = []
        
        print(f"\nCategory: {category}")
        for text in texts:
            input_ids = tokenize(text)
            
            # Get embedding from model (before DPS)
            with torch.no_grad():
                # We need the input to the DPS layer. 
                # This is a bit tricky to extract perfectly without a hook,
                # but we can pass it through the embedding layer to get a rough approximation
                # or just run the model and hook the DPS layer.
                
                # Let's use a hook
                captured_input = []
                def hook(module, input, output):
                    captured_input.append(input[0])
                
                handle = dps_layer.register_forward_hook(hook)
                model(input_ids)
                handle.remove()
                
                if not captured_input:
                    print("Error: DPS hook didn't capture input.")
                    continue
                    
                dps_input = captured_input[0] # [1, seq, dim]
                
                # Now compute novelty components using the dps_layer directly
                novelty = dps_layer.compute_novelty(dps_input)
                
                v = novelty.validity.value
                i = novelty.impact.value
                nw = novelty.value
                
                v_scores.append(v)
                i_scores.append(i)
                nw_scores.append(nw)
                
                print(f"  V: {v:.4f} | I: {i:.4f} | NW: {nw:.4f} | Text: {text[:50]}...")
                
        results[category] = {
            "avg_v": sum(v_scores)/len(v_scores),
            "avg_i": sum(i_scores)/len(i_scores),
            "avg_nw": sum(nw_scores)/len(nw_scores)
        }

    print("\n--- SUMMARY ---")
    print(f"{ 'Category':<25} | {'Avg V':<8} | {'Avg I':<8} | {'Avg NW':<8}")
    print("-" * 60)
    for cat, metrics in results.items():
        print(f"{cat:<25} | {metrics['avg_v']:.4f}   | {metrics['avg_i']:.4f}   | {metrics['avg_nw']:.4f}")
        
    # Validation of Hypothesis
    print("\n--- HYPOTHESIS CHECK ---")
    
    # Check I separation (Breakthrough vs Trivial)
    i_breakthrough = (results['valid_breakthrough']['avg_i'] + results['invalid_breakthrough']['avg_i']) / 2
    i_trivial = (results['valid_trivial']['avg_i'] + results['invalid_trivial']['avg_i']) / 2
    i_separation = i_breakthrough - i_trivial
    print(f"Impact Separation (Breakthrough - Trivial): {i_separation:.4f}")
    if i_separation > 0.4:
        print(">>> VERIFIED: Impact network successfully distinguishes Breakthroughs.")
    else:
        print(">>> FAILED: Impact network failed to distinguish Breakthroughs.")
        
    # Check V separation (Valid vs Invalid)
    v_valid = (results['valid_breakthrough']['avg_v'] + results['valid_trivial']['avg_v']) / 2
    v_invalid = (results['invalid_breakthrough']['avg_v'] + results['invalid_trivial']['avg_v']) / 2
    v_separation = v_valid - v_invalid
    print(f"Validity Separation (Valid - Invalid): {v_separation:.4f}")
    if v_separation < 0.1:
        print(">>> VERIFIED: Validity network is WEAK (separation < 0.1).")
    else:
        print(">>> FAILED: Validity network shows good separation.")

if __name__ == "__main__":
    main()

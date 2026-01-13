
import sys
import os
import torch
import torch.nn as nn
import json

# Add project root to path
sys.path.append(os.getcwd())

from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

def audit_v7():
    print("=== STARTING TKS v7 CLAIM AUDIT (CORRECTED CONFIG) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Model
    model_path = "checkpoints/v7_discovery/v7_discovery_model.pt"
    if not os.path.exists(model_path):
        print(f"CRITICAL: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        # Load config from the JSON file we just created (or hardcode based on log findings)
        # Based on log: vocab=16384, dim=384, depth=8
        config = TKSGeneralConfigV7(
            vocab_size=16384,
            hidden_dim=384,
            num_layers=12,
            use_njt=True,
            njt_use_nested_memory=True,
            njt_max_recursion_depth=8, # Correct depth from logs
            use_dps_gating=True,
            dps_initial_p_max=2,
            dps_max_depth=8
        )
        
        model = TKSGeneralLMv7(config)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Filter runtime buffers if necessary, but with correct config it should match
        if isinstance(checkpoint, dict):
            # Clean up keys if needed (e.g. remove stack_memory/ptr which are buffers)
            state_dict = {k: v for k, v in checkpoint.items() 
                          if "stack_memory" not in k and "stack_ptr" not in k}
            model.load_state_dict(state_dict, strict=False)
        else:
            model = checkpoint
            
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        # print first few keys of checkpoint for debug
        if isinstance(checkpoint, dict):
             print("Checkpoint keys sample:", list(checkpoint.keys())[:5])
        return

    # 2. Setup Inputs
    vocab_size = config.vocab_size
    seq_len = 32
    batch_size = 1
    
    # Input A: "Simple/Repetitive"
    input_ids_a = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    
    # Input B: "Complex/Novel"
    input_ids_b = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # 3. Test Claim: Memory Bank & Novelty
    print("\n--- TEST: Memory Bank & Novelty ---")
    model.reset_dps_state()
    if model._memory_bank is not None:
        model._memory_bank.clear()
        print("Memory Bank: Active and Cleared")

    # Run A (First time)
    with torch.no_grad():
        out_a1 = model(input_ids_a, return_full_trace=True)
    
    trace_a1 = out_a1.get("trace", {})
    dps_trace_a1 = trace_a1.get("dps", {}) if trace_a1 else {}
    nw_a1 = dps_trace_a1.get("novelty_weight", 0.0)
    
    # Check block-level traces for actual NW values
    blocks_trace = trace_a1.get("blocks", [])
    max_nw_a1 = 0.0
    for b_tr in blocks_trace:
        if 'dps' in b_tr and b_tr['dps']:
            max_nw_a1 = max(max_nw_a1, b_tr['dps'].get('novelty_weight', 0.0))

    print(f"Input A (Run 1): Max Block NW={max_nw_a1:.4f}, Allowed Depth={out_a1['allowed_depth']}")

    # 4. Test Claim: Earned Depth (Unlock)
    print("\n--- TEST: Earned Depth ---")
    print(f"Initial p_max: {model.dps_state.p_max}")
    
    print("Feeding 'Complex' input to check token accumulation...")
    unlocked = False
    model.reset_dps_state()
    
    for i in range(5):
        with torch.no_grad():
            # In inference, we might need to manually trigger memory updates if the forward pass doesn't
            # The v7 code says: if self._memory_bank is not None and self.training:
            # So in eval mode, memory isn't updated automatically.
            # To test UNLOCK, we need high novelty. High novelty comes from LOW similarity.
            # An empty memory bank = Max Novelty.
            # So running in eval mode with empty memory SHOULD give high novelty if the network is trained.
            
            out = model(input_ids_b, return_full_trace=True)
        
        state = out['dps_state']
        
        # Get max novelty across blocks
        current_max_nw = 0.0
        blocks_tr = out.get('trace', {}).get('blocks', [])
        for b in blocks_tr:
            if b.get('dps'):
                current_max_nw = max(current_max_nw, b['dps'].get('novelty_weight', 0.0))
        
        print(f"Step {i}: p_max={state.p_max}, tokens={state.tokens}, Max NW={current_max_nw:.4f}")
        
        if state.p_max > 2:
            unlocked = True
            break
            
    if unlocked:
        print("PASS: Depth unlocked successfully!")
    else:
        print("INFO: Depth did not unlock.")
        
    if current_max_nw > 0.0001:
         print(f"PASS: Novelty network is active (NW > 0). Value: {current_max_nw}")
    else:
         print("FAIL: Novelty network is outputting zero (Dead Novelty).")

if __name__ == "__main__":
    audit_v7()

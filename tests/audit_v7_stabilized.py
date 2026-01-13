
import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.getcwd())

from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7, DPSState, DPSConfig

def audit_v7_stabilized():
    print("=== STARTING TKS v7 STABILIZED AUDIT ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Model
    model_path = "checkpoints/v7_stabilized/v7_final_stabilized.pt"
    if not os.path.exists(model_path):
        print(f"CRITICAL: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determine hidden_dim from checkpoint
        # Looking at previous errors, discovery was 256. Let's check this one.
        sample_key = next(iter(checkpoint.keys()))
        hidden_dim = checkpoint[sample_key].shape[-1]
        print(f"Inferred hidden_dim: {hidden_dim}")
        
        # Find recursion depth from depth_embed
        depth_key = "blocks.0.v6_block.njt.njt_v6.depth_embed.weight"
        if depth_key in checkpoint:
            max_depth_plus_1 = checkpoint[depth_key].shape[0]
            max_depth = max_depth_plus_1 - 1
            print(f"Inferred njt_max_recursion_depth: {max_depth}")
        else:
            max_depth = 4 # Default
            
        config = TKSGeneralConfigV7(hidden_dim=hidden_dim, njt_max_recursion_depth=max_depth) 
        model = TKSGeneralLMv7(config)
        
        # Filter runtime buffers
        state_dict = {k: v for k, v in checkpoint.items() 
                      if "stack_memory" not in k and "stack_ptr" not in k}
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # 2. Test inputs
    vocab_size = model.config.vocab_size
    seq_len = 32
    batch_size = 1
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # 3. Comprehensive Audit
    print("\n--- TEST: System Functionality ---")
    with torch.no_grad():
        out = model(input_ids, return_full_trace=True)
    
    allowed_depth = out.get('allowed_depth')
    print(f"Allowed Depth: {allowed_depth}")
    
    trace = out.get('trace', {{}})
    blocks = trace.get('blocks', [])
    
    has_dps = False
    has_reg = False
    max_nw = 0.0
    
    for i, b in enumerate(blocks):
        dps = b.get('dps')
        reg = b.get('regulator')
        
        if dps:
            has_dps = True
            nw = dps.get('novelty_weight', 0.0)
            max_nw = max(max_nw, nw)
            if i == 0:
                print(f"DPS Sample (Block {i}): NW={nw:.4f}, Class={dps.get('novelty_class')}")
        
        if reg:
            has_reg = True
            if i == 0:
                print(f"Regulator Sample (Block {i}): D_gap={reg.get('desire_gap'):.4f}, W_gap={reg.get('wisdom_gap'):.4f}")

    print(f"\nAudit Summary for 'Stabilized':")
    print(f"- DPS Functional: {has_dps} (Max NW observed: {max_nw:.4f})")
    print(f"- Regulators Functional: {has_reg}")
    
    if max_nw > 0:
        print("RESULT: This model has a FUNCTIONAL novelty system.")
    else:
        print("RESULT: This model novelty system is INACTIVE (All NW=0).")

if __name__ == "__main__":
    audit_v7_stabilized()

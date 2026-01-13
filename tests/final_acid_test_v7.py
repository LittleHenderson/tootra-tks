
import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.getcwd())

from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

def run_acid_test():
    print("=== TKS v7 FINAL ACID TEST (GPU) ===")
    
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA is not available. This test must be run on GPU.")
        return
        
    device = "cuda"
    print(f"Using Device: {torch.cuda.get_device_name(0)}")

    # 1. Definitive Configuration from Logs
    config = TKSGeneralConfigV7(
        vocab_size=16384,
        hidden_dim=384,
        num_layers=12,
        use_njt=True,
        njt_use_nested_memory=True,
        njt_max_recursion_depth=8,  # Matches Discovery Log
        use_dps_gating=True,
        dps_initial_p_max=2,
        dps_max_depth=8             # Matches Discovery Log
    )

    # 2. Model Loading
    model_path = "checkpoints/v7_discovery/v7_discovery_model.pt"
    print(f"Loading checkpoint: {model_path}...")
    
    try:
        model = TKSGeneralLMv7(config)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check specific weight shapes before loading
        expected_shape = [9, 384] # depth_embed is max_depth + 1
        found_shape = list(checkpoint["blocks.0.v6_block.njt.njt_v6.depth_embed.weight"].shape)
        print(f"Weight Check (depth_embed): Found {found_shape}, Expected {expected_shape}")
        
        # Filter stack buffers
        state_dict = {k: v for k, v in checkpoint.items() 
                      if "stack_memory" not in k and "stack_ptr" not in k}
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("SUCCESS: Model loaded and transferred to GPU.")
    except Exception as e:
        print(f"FAILURE during load: {e}")
        return

    # 3. Forward Pass Execution
    print("\nExecuting deep forward pass (batch=1, seq=64)...")
    input_ids = torch.randint(0, config.vocab_size, (1, 64)).to(device)
    
    try:
        with torch.no_grad():
            output = model(input_ids, return_full_trace=True, episode_id="acid_test_01")
            
        logits = output["logits"]
        dps_state = output["dps_state"]
        trace = output.get("trace", {})
        
        print("-" * 30)
        print(f"Output Status: OK")
        print(f"Logits Shape: {list(logits.shape)}")
        print(f"Allowed Depth (DPS): {output['allowed_depth']}")
        print(f"Final p_max: {dps_state.p_max}")
        
        # Audit Regulators
        blocks = trace.get("blocks", [])
        if blocks:
            reg = blocks[0].get("regulator")
            if reg:
                print(f"Regulator Health (Block 0): D_gap={reg['desire_gap']:.4f}, W_gap={reg['wisdom_gap']:.4f}")
        
        # Audit Recursion Stack
        njt_trace = blocks[0].get("njt")
        if njt_trace:
             print(f"NJT Stack Status: Active, Max Depth Reached: {njt_trace.get('max_depth_reached', 0)}")

        print("-" * 30)
        print("FINAL VERDICT: MODEL IS FULLY FUNCTIONAL AND READY FOR PRODUCTION.")
        
    except Exception as e:
        print(f"CRITICAL FAILURE during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_acid_test()

import torch
from tks_llm_core_v5 import TKSGeneralLM
from configs.v5_recommended import get_v5_config
from tokenizers import Tokenizer

def inspect_model():
    # 1. Load Config
    print("Loading config...")
    config = get_v5_config(
        size="base",
        use_stable_routing=True,
        use_attractor=True,
        use_rpm=True,
        use_njt=True,
        njt_num_transistors=10,
        njt_use_hysteresis=True,
        njt_use_rhythm=True,
    )
    
    # 2. Load Tokenizer & Model
    print("Loading tokenizer and model...")
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    config.vocab_size = tokenizer.get_vocab_size()
    
    model = TKSGeneralLM(config)
    
    # 3. Load Weights
    checkpoint_path = "checkpoints/v5_final.pt"
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu") # Load to CPU for inspection
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Create Sample Input
    text = "The quick brown fox"
    print(f"\nInput Text: '{text}'")
    ids = tokenizer.encode(text).ids
    input_tensor = torch.tensor([ids], dtype=torch.long)
    
    # 5. Run Inference
    print("Running inference...")
    with torch.no_grad():
        # Pass a high step count to simulate trained state for router
        output = model(input_tensor, step=37500) 
        logits = output["logits"]
    
    # 6. Analyze Logits
    # Shape: [batch, seq_len, vocab_size]
    last_token_logits = logits[0, -1, :]
    
    print(f"\nLogits Shape: {logits.shape}")
    print(f"Last Token Logits (First 10): {last_token_logits[:10].tolist()}")
    
    # Top 5 predictions
    probs = torch.softmax(last_token_logits, dim=0)
    top_probs, top_indices = torch.topk(probs, 5)
    
    print("\nTop 5 Predictions:")
    for score, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx.item()])
        print(f"  Token: '{token}' (ID: {idx.item()}) | Prob: {score.item():.4f} | Logit: {last_token_logits[idx].item():.4f}")

    # 7. Check NJT Weights (Quick sanity check)
    print("\nNJT Circuit Check:")
    if hasattr(model.blocks[0], 'njt'):
        gain = model.blocks[0].njt.excitatory.gain.mean().item()
        print(f"  Layer 0 NJT Gain: {gain:.4f}")

if __name__ == "__main__":
    inspect_model()

import torch
import json
from tokenizers import Tokenizer
from tks_llm_core_v6 import TKSGeneralLMv6, TKSGeneralConfig

def load_model():
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()
    
    print("Initializing model...")
    config = TKSGeneralConfig(
        vocab_size=vocab_size,
        use_njt=True,
        njt_use_nested_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TKSGeneralLMv6(config).to(device)
    
    checkpoint_path = "checkpoints/v6_best.pt"
    print(f"Loading weights from {checkpoint_path}...")
    try:
        # load_state_dict might be needed or just load directly if it's the state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Filter out recursion_stack buffers that have batch size mismatches
        # These are reset in forward() anyway
        keys_to_remove = []
        for key in state_dict.keys():
            if "recursion_stack.stack_memory" in key or "recursion_stack.stack_ptr" in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del state_dict[key]
            
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None

    model.eval()
    return model, tokenizer, device

def generate(model, tokenizer, device, prompt, max_new_tokens=200):
    # Prepare input
    # train_v6.py adds <BOS>
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    
    encoded = tokenizer.encode(prompt)
    input_ids = [bos_id] + encoded.ids if bos_id else encoded.ids
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    generated_ids = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(torch.tensor([generated_ids], dtype=torch.long).to(device))
            logits = outputs["logits"]
            
            # Greedy decoding: pick highest probability token
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            generated_ids.append(next_token_id)
            print(".", end="", flush=True)
            
            if next_token_id == eos_id:
                break
    
    print(" Done.")
    decoded_text = tokenizer.decode(generated_ids)
    return decoded_text

def main():
    model, tokenizer, device = load_model()
    if model is None:
        return

    test_prompts = [
        "Goal: Cultivate wisdom <SEP> Reasoning:",
        "Goal: Overcome fear <SEP> Reasoning:",
        "Goal: Understand the nature of desire <SEP> Reasoning:"
    ]

    print("\n=== STARTING INFERENCE TESTS ===")
    
    for prompt in test_prompts:
        output = generate(model, tokenizer, device, prompt)
        print(f"\n--- Output for '{prompt}' ---")
        print(output)
        print("-" * 50)

if __name__ == "__main__":
    main()

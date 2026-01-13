import torch
import json
from tokenizers import Tokenizer
from tks_llm_core_v6 import TKSGeneralLMv6, TKSGeneralConfig

def load_model():
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()
    config = TKSGeneralConfig(
        vocab_size=vocab_size,
        use_njt=True,
        njt_use_nested_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TKSGeneralLMv6(config).to(device)
    checkpoint_path = "checkpoints/v6_best.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    keys_to_remove = [k for k in state_dict.keys() if "recursion_stack" in k]
    for k in keys_to_remove: del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer, device

def generate(model, tokenizer, device, prompt, max_new_tokens=100):
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    encoded = tokenizer.encode(prompt)
    input_ids = [bos_id] + encoded.ids if bos_id else encoded.ids
    generated_ids = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(torch.tensor([generated_ids], dtype=torch.long).to(device))
            logits = outputs["logits"]
            next_token_id = torch.argmax(logits[0, -1, :]).item()
            generated_ids.append(next_token_id)
            if next_token_id == eos_id: break
            
    return tokenizer.decode(generated_ids)

def main():
    model, tokenizer, device = load_model()
    
    # Testing for natural language generation
    test_prompts = [
        "Goal: Start a business <SEP> Story:",
        "Explain the TKS expression B5 -> D3:",
        "What is the meaning of life in TKS?",
        "Goal: Cultivate wisdom <SEP> Reasoning: CHECK 4 D ( Mental Wisdom ) = 0 9 [ MASTER ING ] -> N9 <SEP> Interpretation:"
    ]

    print("\n=== NATURAL LANGUAGE INFERENCE TESTS ===")
    for prompt in test_prompts:
        output = generate(model, tokenizer, device, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")
        print("-" * 50)

if __name__ == "__main__":
    main()

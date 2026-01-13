"""Test model with EXACT training examples to see if it can reproduce memorized outputs."""
import torch
import json
import sys
import os
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer (same as training script)
    tokenizer_path = "tokenizer_v5.json"
    if not os.path.exists(tokenizer_path):
        print(f"FATAL ERROR: {tokenizer_path} not found.")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # Recreate config (same as training)
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8,
        use_memory_bank=True,
        memory_bank_size=1000,
        use_impact_only_nw=True,
        dps_heavy_threshold=0.01,
        dps_count_threshold=0.005,
        dps_tokens_for_unlock=1,
        dps_cooldown_episodes=0
    )

    # Load model
    model = TKSGeneralLMv7(config)
    state_dict = torch.load('checkpoints/v7_math_code/v7_math_code_model.pt', map_location=device)
    # Filter out runtime buffers that don't match
    state_dict = {k: v for k, v in state_dict.items() if "stack_memory" not in k and "stack_ptr" not in k}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Load EXACT training examples
    with open('data/v7_math_code_training.jsonl', 'r', encoding='utf-8') as f:
        training_examples = [json.loads(line) for line in f][:5]  # First 5

    print("\n" + "="*70)
    print("TESTING WITH EXACT TRAINING DATA")
    print("="*70)
    print("If model truly memorized, it should reproduce these outputs.")
    print("If it outputs spaces, there's a fundamental generation bug.\n")

    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    if bos_id is None or eos_id is None:
        print("FATAL ERROR: tokenizer missing <BOS>/<EOS> tokens.")
        return

    for i, ex in enumerate(training_examples):
        inp = f"Input: {ex['input']}\n"  # The format from training
        expected_start = ex['output'][:100]

        print(f"\n[TRAINING EXAMPLE {i+1}]")
        print(f"Input: {ex['input'][:60]}...")
        print(f"Expected starts with: {expected_start[:50]}...")

        # Encode input (with BOS token like training)
        enc = tokenizer.encode(inp)
        tokens = [bos_id] + enc.ids
        input_tensor = torch.tensor([tokens], device=device)

        # Generate
        with torch.no_grad():
            generated_ids = []
            current = input_tensor

            for step in range(50):  # Generate 50 tokens
                output = model(current)
                logits = output['logits'][0, -1, :]
                probs = torch.softmax(logits, dim=-1)

                top_prob, top_idx = probs.max(dim=-1)
                token_id = top_idx.item()
                token = tokenizer.id_to_token(token_id) or "?"
                generated_ids.append(token_id)

                # Show first few token probabilities
                if step < 3:
                    print(f"  Token {step+1}: '{repr(token)}' ({top_prob.item():.2%})")

                if token_id == eos_id:
                    break

                next_token = torch.tensor([[token_id]], device=device)
                current = torch.cat([current, next_token], dim=1)

        output_str = tokenizer.decode(generated_ids)

        # Check if it's all spaces
        non_space = len([c for c in output_str if c.strip()])

        print(f"Generated: '{output_str[:60]}...'")
        print(f"Non-space chars: {non_space}/{len(generated_ids)}")

        if non_space == 0:
            print("Result: SPACES ONLY - even on exact training data!")
        elif expected_start[:20] in output_str or output_str[:20] in expected_start:
            print("Result: MATCHED training output!")
        else:
            print("Result: Generated something, but doesn't match expected")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    # Bonus: Test with exact training format (full input + start of output)
    print("\nBonus test: Give model START of expected output, see if it continues")
    ex = training_examples[0]
    full_train_format = f"Input: {ex['input']}\n{ex['output'][:50]}"

    enc = tokenizer.encode(full_train_format)
    tokens = [bos_id] + enc.ids
    input_tensor = torch.tensor([tokens], device=device)

    with torch.no_grad():
        generated_ids = []
        current = input_tensor

        for _ in range(30):
            output = model(current)
            logits = output['logits'][0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
            token_id = top_idx.item()
            token = tokenizer.id_to_token(token_id) or "?"
            generated_ids.append(token_id)
            if token_id == eos_id:
                break
            next_token = torch.tensor([[token_id]], device=device)
            current = torch.cat([current, next_token], dim=1)

    output_str = tokenizer.decode(generated_ids)
    expected_cont = ex['output'][50:80]

    print(f"Prompt ends with: ...{full_train_format[-30:]}")
    print(f"Expected continuation: {expected_cont}")
    print(f"Model continuation: {output_str[:30]}")

    non_space = len([c for c in output_str if c.strip()])
    if non_space == 0:
        print("\nVERDICT: Model outputs SPACES for EVERYTHING")
        print("This is NOT memorization - it's a broken generation loop.")
        print("The model learned to always predict space as most likely.")
    else:
        print("\nModel can continue known sequences but struggles with generation.")

if __name__ == '__main__':
    main()

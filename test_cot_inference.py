#!/usr/bin/env python3
"""
Test CoT Inference - Compare both models with TKS-style prompts
"""

import sys
import torch
from pathlib import Path
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))

def test_model(checkpoint_path, model_name, prompts, temperature=0.7):
    """Test a model checkpoint with given prompts."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    if not Path(checkpoint_path).exists():
        print(f"  [SKIP] Checkpoint not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()

    # Load model
    from tks_llm_core_v7 import TKSGeneralConfigV7, TKSGeneralLMv7
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=12,
        use_dps_gating=True,
        use_regulators=False,
    )
    model = TKSGeneralLMv7(config)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)}/{len(state_dict)} weights")

    model = model.to(device)
    model.eval()

    # Test each prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)

        enc = tokenizer.encode(prompt)
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")

        input_ids = [bos_id] + enc.ids if bos_id else enc.ids
        generated = input_ids.copy()

        with torch.no_grad():
            for _ in range(200):  # Generate more tokens
                input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
                out = model(input_tensor)

                # Sample with temperature
                logits = out["logits"][0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)
                if next_token == eos_id:
                    break

        decoded = tokenizer.decode(generated)
        # Clean up
        decoded = decoded.replace("<BOS>", "").replace("<EOS>", "").strip()
        print(f"Output:\n{decoded}")
        print()


def main():
    # TKS-style prompts that match original training format
    cot_prompts = [
        # TKS equation style (like original training)
        "A7 + B6:",
        "C3 + D5:",
        # CoT trigger prompts
        "Step 1: Accessing TKS Canon...",
        "Trace: RECURSE ->",
        # Foundation lookup style
        "Foundation 2 is",
        "Noetic 5 represents",
        # World combination style
        "World A combined with World B produces",
    ]

    print("\n" + "="*70)
    print("TESTING COT INFERENCE WITH TKS-STYLE PROMPTS")
    print("="*70)

    # Test v7 Foundation (original)
    test_model(
        "checkpoints/v7_foundation/v7_foundation_best.pt",
        "v7 Foundation (Original)",
        cot_prompts,
        temperature=0.7
    )

    # Test NL Coherent (after training)
    test_model(
        "checkpoints/nl_coherent_v7/nl_coherent_final.pt",
        "v7 NL Coherent (After Training)",
        cot_prompts,
        temperature=0.7
    )


if __name__ == "__main__":
    main()

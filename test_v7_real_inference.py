"""
Real Inference Test for Original TKS v7 Model

Tests the original v7 model (not v7-Coherent) with trained weights.
"""

import torch
from pathlib import Path

# Load tokenizer
try:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()
except:
    tokenizer = None
    vocab_size = 16384

# Import v7 model
from tks_llm_core_v7 import TKSGeneralConfigV7, TKSGeneralLMv7

def load_v7_model(checkpoint_path: str):
    """Load v7 model with correct config matching checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # v7 checkpoints were trained with hidden_dim=256
    config = TKSGeneralConfigV7(
        vocab_size=vocab_size,
        hidden_dim=256,  # Match checkpoint
        num_layers=12,
        use_dps_gating=True,
        use_memory_bank=True,
        use_regulators=False,  # May not be available
    )

    model = TKSGeneralLMv7(config).to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load weights
    print(f"Loading weights from {checkpoint_path}...")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Filter out incompatible keys
        model_state = model.state_dict()
        filtered = {}
        skipped = []

        for k, v in state_dict.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(f"{k}: {v.shape} vs {model_state[k].shape}")
            else:
                skipped.append(f"{k}: not in model")

        model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(state_dict)} weights")
        if skipped[:5]:
            print(f"Skipped (first 5): {skipped[:5]}")

    except Exception as e:
        print(f"Error loading weights: {e}")
        return None, None

    model.eval()
    return model, device


def generate(model, device, prompt: str, max_tokens: int = 50):
    """Generate text from prompt."""

    if tokenizer:
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")
        encoded = tokenizer.encode(prompt)
        input_ids = [bos_id] + encoded.ids if bos_id else encoded.ids
    else:
        input_ids = [1] + list(range(10, 30))
        eos_id = 2

    generated = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long).to(device)

            try:
                outputs = model(input_tensor)
                logits = outputs["logits"]
            except Exception as e:
                # Try without DPS state
                outputs = model.blocks[0].v6_block(
                    model.embedding(input_tensor) + model.pos_emb[:, :len(generated), :]
                )
                print(f"Direct forward error: {e}")
                break

            # Greedy decode
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)

            if next_token == eos_id:
                break

    if tokenizer:
        decoded = tokenizer.decode(generated)
        return decoded
    else:
        return f"[Token IDs: {generated}]"


def test_v7_inference():
    """Run inference tests on v7."""

    print("=" * 70)
    print("TKS v7 REAL INFERENCE TEST")
    print("=" * 70)

    # Try different checkpoints
    checkpoints = [
        ("v7_stabilized", "checkpoints/v7_stabilized/v7_final_stabilized.pt"),
        ("v7_foundation", "checkpoints/v7_foundation/v7_foundation_best.pt"),
    ]

    for name, path in checkpoints:
        if not Path(path).exists():
            print(f"\n{name}: Checkpoint not found")
            continue

        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        model, device = load_v7_model(path)
        if model is None:
            continue

        # Test prompts
        test_prompts = [
            "Goal: Cultivate wisdom <SEP> Reasoning:",
            "Goal: Overcome fear <SEP> Reasoning:",
            "Goal: Build inner strength <SEP> Reasoning:",
        ]

        for prompt in test_prompts:
            print(f"\n--- Prompt: '{prompt[:40]}...' ---")

            try:
                output = generate(model, device, prompt, max_tokens=60)
                # Safe print
                output_safe = output.encode('ascii', 'replace').decode('ascii')
                print(f"Output: {output_safe}")
            except Exception as e:
                print(f"Generation error: {e}")

        # Also test with return_trace to see DPS info
        print(f"\n--- DPS/Depth Info ---")
        try:
            input_ids = torch.randint(0, 100, (1, 20)).to(device)
            out = model(input_ids, return_full_trace=True)

            if 'dps_state' in out:
                dps = out['dps_state']
                print(f"DPS p_max: {dps.p_max}")
                print(f"DPS tokens: {dps.tokens}")
                print(f"DPS total_unlocks: {dps.total_unlocks}")

            if 'depth_controller_status' in out:
                print(f"Depth status: {out['depth_controller_status']}")

        except Exception as e:
            print(f"Trace error: {e}")


if __name__ == "__main__":
    test_v7_inference()

"""
Real Inference Test for v6-Coherent and v7-Coherent Models

Tests:
1. Loading models with trained weights (if available)
2. Real text generation with coherence tracking
3. NF coordinate tracking per layer
4. Coherence-driven depth adjustment (v7)
5. Pattern bank accumulation
"""

import torch
import json
from pathlib import Path

# Try to load tokenizer
try:
    from tokenizers import Tokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("Warning: tokenizers not available, using dummy tokenizer")


def load_tokenizer():
    """Load the BPE tokenizer."""
    if not HAS_TOKENIZER:
        return None

    tokenizer_path = Path("tokenizer_v5.json")
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))
    return None


def test_v6_coherent_inference():
    """Test v6-Coherent with real inference."""
    print("=" * 70)
    print("V6-COHERENT REAL INFERENCE TEST")
    print("=" * 70)

    from tks_llm_core_v6_coherent import TKSCoherentConfig, TKSGeneralLMv6Coherent

    # Load tokenizer
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size() if tokenizer else 16384

    # Create config
    config = TKSCoherentConfig(
        vocab_size=vocab_size,
        hidden_dim=384,
        num_layers=12,
        use_coherence_tracking=True,
        coherence_gate_enabled=True,
        use_pattern_feedback=True,
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = TKSGeneralLMv6Coherent(config).to(device)

    # Try to load weights
    checkpoint_path = Path("checkpoints/v6_best.pt")
    if checkpoint_path.exists():
        print(f"Loading weights from {checkpoint_path}...")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            # Filter incompatible keys
            model_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items()
                       if k in model_keys and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(filtered, strict=False)
            print(f"Loaded {len(filtered)}/{len(state_dict)} weights")
        except Exception as e:
            print(f"Could not load weights: {e}")
    else:
        print("No checkpoint found, using random weights")

    model.eval()

    # Test prompts
    test_prompts = [
        "Goal: Cultivate wisdom <SEP> Reasoning:",
        "Goal: Overcome fear <SEP> Reasoning:",
        "Goal: Build inner strength <SEP> Reasoning:",
    ]

    print("\n" + "=" * 70)
    print("INFERENCE TESTS")
    print("=" * 70)

    for prompt in test_prompts:
        print(f"\n--- Prompt: '{prompt}' ---")

        # Tokenize
        if tokenizer:
            bos_id = tokenizer.token_to_id("<BOS>")
            encoded = tokenizer.encode(prompt)
            input_ids = [bos_id] + encoded.ids if bos_id else encoded.ids
        else:
            # Dummy tokenization
            input_ids = [1] + [hash(c) % 1000 for c in prompt[:50]]

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Forward pass with full trace
        with torch.no_grad():
            output = model(input_tensor, return_full_trace=True)

        # Display results
        print(f"\nCoherence Score:")
        if output['coherence_score']:
            score = output['coherence_score']
            print(f"  Overall: {score.overall:.3f}")
            print(f"  Lacunary: {score.lacunary_index:.3f}")
            print(f"  NF Validity: {score.nf_validity:.3f}")
            print(f"  Fractal Dim: {score.fractal_dimension:.3f}")
            print(f"  Detected NF: {score.detected_nf}")
            print(f"  Gibberish: {score.gibberish_detected}")

        print(f"\nNF Coordinates Per Layer:")
        if 'trace' in output and output['trace']['patterns']:
            for pattern in output['trace']['patterns']:
                nf = pattern.nf_coords
                coh = pattern.flow_coherence
                cat = pattern.category
                print(f"  Layer {pattern.layer_idx}: NF={nf} ({cat}), coherence={coh:.3f}")

        print(f"\nDepth Adjustment: {output['depth_adjustment']}")

        # Generate some tokens
        print(f"\nGeneration (greedy, 30 tokens):")
        generated_ids = input_ids.copy()

        with torch.no_grad():
            for _ in range(30):
                gen_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
                out = model(gen_tensor)
                next_token = out['logits'][0, -1, :].argmax().item()
                generated_ids.append(next_token)

                # Check for EOS
                if tokenizer:
                    eos_id = tokenizer.token_to_id("<EOS>")
                    if next_token == eos_id:
                        break

        if tokenizer:
            decoded = tokenizer.decode(generated_ids)
            # Handle Unicode safely
            decoded_safe = decoded.encode('ascii', 'replace').decode('ascii')
            print(f"  {decoded_safe}")
        else:
            print(f"  [Token IDs: {generated_ids[-30:]}]")

    # Pattern bank stats
    print(f"\n" + "=" * 70)
    print("PATTERN BANK STATISTICS")
    print("=" * 70)
    if model.pattern_feedback:
        stats = model.pattern_feedback.pattern_bank.get_stats()
        print(f"  Total patterns: {stats['total_patterns']}")
        print(f"  Total added: {stats['total_added']}")

    return model


def test_v7_coherent_inference():
    """Test v7-Coherent with real inference."""
    print("\n" + "=" * 70)
    print("V7-COHERENT REAL INFERENCE TEST")
    print("(Earned Depth + Coherence)")
    print("=" * 70)

    from tks_llm_core_v7_coherent import TKSCoherentConfigV7, TKSGeneralLMv7Coherent

    # Load tokenizer
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size() if tokenizer else 16384

    # Create config
    config = TKSCoherentConfigV7(
        vocab_size=vocab_size,
        hidden_dim=384,
        num_layers=12,
        use_coherence_tracking=True,
        coherence_gate_enabled=True,
        use_dps_gating=True,
        use_memory_bank=True,
        coherence_drives_depth=True,
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = TKSGeneralLMv7Coherent(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    model.eval()

    # Test prompts
    test_prompts = [
        "Goal: Cultivate wisdom <SEP> Reasoning:",
        "Goal: Overcome fear <SEP> Reasoning:",
        "Goal: Achieve mastery <SEP> Reasoning:",
    ]

    print("\n" + "=" * 70)
    print("INFERENCE TESTS WITH DEPTH TRACKING")
    print("=" * 70)

    for prompt in test_prompts:
        print(f"\n--- Prompt: '{prompt}' ---")

        # Tokenize
        if tokenizer:
            bos_id = tokenizer.token_to_id("<BOS>")
            encoded = tokenizer.encode(prompt)
            input_ids = [bos_id] + encoded.ids if bos_id else encoded.ids
        else:
            input_ids = [1] + [hash(c) % 1000 for c in prompt[:50]]

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Forward pass with full trace
        with torch.no_grad():
            output = model(input_tensor, return_full_trace=True, episode_id=f"test_{prompt[:10]}")

        # Display results
        print(f"\nCoherence Score:")
        if output['coherence_score']:
            score = output['coherence_score']
            print(f"  Overall: {score.overall:.3f}")
            print(f"  Lacunary: {score.lacunary_index:.3f}")
            print(f"  NF Validity: {score.nf_validity:.3f}")
            print(f"  Detected NF: {score.detected_nf}")

        print(f"\nDepth Controller Status:")
        depth_status = output['depth_status']
        print(f"  DPS p_max: {depth_status['dps_p_max']}")
        print(f"  Coherence boost: {depth_status['coherence_boost']}")
        print(f"  Novelty burst: {depth_status['novelty_burst']}")
        print(f"  Allowed depth: {depth_status['allowed_depth']}")
        print(f"  Avg coherence: {depth_status['avg_coherence']:.3f}")

        print(f"\nDPS State:")
        dps = output['dps_state']
        print(f"  p_max: {dps.p_max}")
        print(f"  tokens: {dps.tokens}")
        print(f"  total_unlocks: {dps.total_unlocks}")

        print(f"\nNF Coordinates Per Layer:")
        if 'trace' in output and output['trace']['patterns']:
            for pattern in output['trace']['patterns']:
                nf = pattern.nf_coords
                coh = pattern.flow_coherence
                cat = pattern.category
                print(f"  Layer {pattern.layer_idx}: NF={nf} ({cat}), coherence={coh:.3f}")

        # Generate tokens with coherence tracking
        print(f"\nGeneration with Coherence Tracking (30 tokens):")
        generated_ids = input_ids.copy()
        coherence_trace = []
        depth_trace = []

        with torch.no_grad():
            for t in range(30):
                gen_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
                out = model(gen_tensor, episode_id=f"gen_{t}")
                next_token = out['logits'][0, -1, :].argmax().item()
                generated_ids.append(next_token)

                if out['coherence_score']:
                    coherence_trace.append(out['coherence_score'].overall)
                depth_trace.append(out['depth_status']['allowed_depth'])

                # Check for EOS
                if tokenizer:
                    eos_id = tokenizer.token_to_id("<EOS>")
                    if next_token == eos_id:
                        break

        if tokenizer:
            decoded = tokenizer.decode(generated_ids)
            # Handle Unicode safely
            decoded_safe = decoded.encode('ascii', 'replace').decode('ascii')
            print(f"  {decoded_safe}")
        else:
            print(f"  [Token IDs: {generated_ids[-30:]}]")

        if coherence_trace:
            avg_coh = sum(coherence_trace) / len(coherence_trace)
            print(f"\n  Avg coherence during generation: {avg_coh:.3f}")
            print(f"  Depth range: {min(depth_trace)} - {max(depth_trace)}")

    # Memory bank stats
    print(f"\n" + "=" * 70)
    print("MEMORY BANK STATISTICS")
    print("=" * 70)
    if model.memory_bank:
        print(f"  Size: {model.memory_bank.size}")
        print(f"  Capacity: {model.memory_bank.capacity}")

    return model


def compare_models():
    """Compare v6-Coherent and v7-Coherent on same input."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: v6-Coherent vs v7-Coherent")
    print("=" * 70)

    from tks_llm_core_v6_coherent import TKSCoherentConfig, TKSGeneralLMv6Coherent
    from tks_llm_core_v7_coherent import TKSCoherentConfigV7, TKSGeneralLMv7Coherent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create both models with same base config
    config_v6 = TKSCoherentConfig(
        vocab_size=16384,
        hidden_dim=384,
        num_layers=8,
        use_coherence_tracking=True,
        coherence_gate_enabled=True,
    )

    config_v7 = TKSCoherentConfigV7(
        vocab_size=16384,
        hidden_dim=384,
        num_layers=8,
        use_coherence_tracking=True,
        coherence_gate_enabled=True,
        use_dps_gating=True,
        coherence_drives_depth=True,
    )

    model_v6 = TKSGeneralLMv6Coherent(config_v6).to(device).eval()
    model_v7 = TKSGeneralLMv7Coherent(config_v7).to(device).eval()

    # Same input
    input_ids = torch.randint(0, 100, (1, 20)).to(device)

    print("\nSame input to both models:")

    with torch.no_grad():
        out_v6 = model_v6(input_ids, return_full_trace=True)
        out_v7 = model_v7(input_ids, return_full_trace=True)

    print("\n--- v6-Coherent ---")
    print(f"  Coherence: {out_v6['coherence_score'].overall:.3f}")
    print(f"  Depth adjustment: {out_v6['depth_adjustment']}")
    print(f"  NF per layer: ", end="")
    for p in out_v6['trace']['patterns'][:4]:
        print(f"{p.nf_coords} ", end="")
    print()

    print("\n--- v7-Coherent ---")
    print(f"  Coherence: {out_v7['coherence_score'].overall:.3f}")
    print(f"  DPS p_max: {out_v7['dps_state'].p_max}")
    print(f"  Allowed depth: {out_v7['depth_status']['allowed_depth']}")
    print(f"  Coherence boost: {out_v7['depth_status']['coherence_boost']}")
    print(f"  NF per layer: ", end="")
    for p in out_v7['trace']['patterns'][:4]:
        print(f"{p.nf_coords} ", end="")
    print()

    print("\n--- Key Difference ---")
    print(f"  v6: Static depth (no DPS)")
    print(f"  v7: Dynamic depth = DPS.p_max ({out_v7['dps_state'].p_max}) + coherence_boost ({out_v7['depth_status']['coherence_boost']}) = {out_v7['depth_status']['allowed_depth']}")


if __name__ == "__main__":
    # Test v6-Coherent
    model_v6 = test_v6_coherent_inference()

    # Test v7-Coherent
    model_v7 = test_v7_coherent_inference()

    # Compare models
    compare_models()

    print("\n" + "=" * 70)
    print("ALL INFERENCE TESTS COMPLETE")
    print("=" * 70)

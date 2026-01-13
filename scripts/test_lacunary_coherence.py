"""
Test Lacunary Coherence Integration with TKS Model

This script tests whether the canonical TKS lacunarity formulas can detect
and prevent gibberish output from the model.

Tests:
1. Load model and generate text samples
2. Compute lacunarity, noetic complexity, and texture tuple for each sample
3. Compare coherent vs gibberish outputs
4. Test coherence gating to prevent gibberish

Author: TKS Research Pipeline
Date: 2026-01-13
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from pathlib import Path
import json

# TKS imports
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7
from tks_features.lacunary_pattern_flow import (
    CanonicalLacunarity, TextureTuple, PatternCategory, LacunaryFlowSystem
)
from tks_features.noetic_fractal_coherence import (
    NoeticFractalCoherenceSystem, NoeticFractalEncoder
)
from tks_features.coherent_tks_wrapper import CoherentTKSModel, CoherenceConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load the v7 model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or infer from weights
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = TKSGeneralConfigV7(**config_dict)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        # Infer config from weight shapes
        state_dict = checkpoint
        if 'token_emb.weight' in state_dict:
            vocab_size, hidden_dim = state_dict['token_emb.weight'].shape
            # Count blocks
            num_layers = 0
            for key in state_dict.keys():
                if key.startswith('blocks.') and '.v6_block.norm1.weight' in key:
                    layer_idx = int(key.split('.')[1])
                    num_layers = max(num_layers, layer_idx + 1)

            print(f"Inferred config: vocab_size={vocab_size}, hidden_dim={hidden_dim}, num_layers={num_layers}")
            config = TKSGeneralConfigV7(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        else:
            config = TKSGeneralConfigV7()

    # Create model
    model = TKSGeneralLMv7(config)

    # Filter out recursion stack buffers (they're reset each forward anyway)
    # Also check for shape mismatches
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in state_dict.items():
        # Skip stack buffers
        if 'stack_memory' in key or 'stack_ptr' in key:
            skipped.append(key)
            continue
        # Check shape match
        if key in model_state:
            if model_state[key].shape != value.shape:
                skipped.append(f"{key} (shape mismatch)")
                continue
        filtered_state[key] = value

    if skipped:
        print(f"Skipped {len(skipped)} incompatible weights")

    # Load weights
    model.load_state_dict(filtered_state, strict=False)

    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config


def generate_text(model, tokenizer, prompt: str, max_length: int = 50, temperature: float = 1.0):
    """Generate text from the model."""
    device = next(model.parameters()).device

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated = input_ids.clone()
    hidden_states = []

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            logits = outputs['logits'][:, -1, :] / temperature

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Store hidden state for analysis (from last forward pass)
            if 'hidden_states' in outputs:
                hidden_states.append(outputs['hidden_states'][-1])

    return generated, hidden_states


def analyze_coherence(hidden_states: torch.Tensor, nf_encoder: NoeticFractalEncoder):
    """Analyze coherence of hidden states using canonical TKS formulas."""
    if hidden_states.dim() == 3:
        # [batch, seq, dim] -> [seq, dim]
        hidden_states = hidden_states.mean(dim=0)

    # Compute lacunarity
    lacunarity = CanonicalLacunarity.compute_lacunarity(hidden_states)

    # Get NF coordinates for each position
    nf_sequence = []
    noetic_dim = min(hidden_states.shape[-1], 40)

    for i in range(hidden_states.shape[0]):
        h = hidden_states[i:i+1, :noetic_dim]
        _, validity, (x, y, z) = nf_encoder(h, return_nf=True)
        nf_coords = (x[0].item(), y[0].item(), z[0].item())
        nf_sequence.append(nf_coords)

    # Compute noetic complexity
    complexity = CanonicalLacunarity.compute_noetic_complexity(nf_sequence)

    # Compute clustering coefficient
    clustering = CanonicalLacunarity.compute_clustering_coefficient(nf_sequence)

    # Check for stabilizing fractals
    stabilizing_count = sum(1 for nf in nf_sequence if CanonicalLacunarity.is_stabilizing_fractal(nf)[0])

    # Compute texture tuple
    texture = CanonicalLacunarity.compute_texture(hidden_states, nf_sequence)

    return {
        'lacunarity': lacunarity,
        'complexity': complexity,
        'clustering': clustering,
        'stabilizing_ratio': stabilizing_count / len(nf_sequence) if nf_sequence else 0,
        'texture': texture,
        'nf_sequence': nf_sequence[:10],  # First 10 for display
        'is_coherent': texture.is_coherent,
        'coherence_score': texture.coherence_score(),
    }


def test_with_simple_tokenizer():
    """Test with a simple character-level tokenizer for demo."""
    print("=" * 80)
    print("LACUNARY COHERENCE INTEGRATION TEST")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check for checkpoint
    checkpoint_paths = [
        "checkpoints/v7_discovery/v7_discovery_model.pt",
        "checkpoints/v6_best.pt",
        "checkpoints/v5_final.pt",
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("\nNo checkpoint found. Running synthetic test...")
        return test_synthetic()

    try:
        model, config = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Running synthetic test instead...")
        return test_synthetic()

    # Create NF encoder for analysis
    nf_encoder = NoeticFractalEncoder(
        embed_dim=min(config.hidden_dim, 40),
        noetic_dim=40,
    ).to(device)

    # Test 1: Analyze random vs structured hidden states
    print("\n" + "-" * 40)
    print("TEST 1: Random vs Structured Hidden States")
    print("-" * 40)

    # Coherent-like (structured)
    seq_len = 32
    structured = torch.randn(seq_len, config.hidden_dim, device=device)
    # Add structure by making adjacent tokens similar
    for i in range(1, seq_len):
        structured[i] = 0.7 * structured[i-1] + 0.3 * structured[i]

    # Gibberish-like (random, high variance)
    random_gibberish = torch.randn(seq_len, config.hidden_dim, device=device) * 3

    # Analyze both
    print("\nStructured (coherent-like):")
    analysis_structured = analyze_coherence(structured, nf_encoder)
    print(f"  Lacunarity: {analysis_structured['lacunarity']:.3f}")
    print(f"  Complexity: {analysis_structured['complexity']:.3f}")
    print(f"  Clustering: {analysis_structured['clustering']:.3f}")
    print(f"  Coherence Score: {analysis_structured['coherence_score']:.3f}")
    print(f"  Is Coherent: {analysis_structured['is_coherent']}")

    print("\nRandom (gibberish-like):")
    analysis_random = analyze_coherence(random_gibberish, nf_encoder)
    print(f"  Lacunarity: {analysis_random['lacunarity']:.3f}")
    print(f"  Complexity: {analysis_random['complexity']:.3f}")
    print(f"  Clustering: {analysis_random['clustering']:.3f}")
    print(f"  Coherence Score: {analysis_random['coherence_score']:.3f}")
    print(f"  Is Coherent: {analysis_random['is_coherent']}")

    # Test 2: Forward pass through model
    print("\n" + "-" * 40)
    print("TEST 2: Model Forward Pass Analysis")
    print("-" * 40)

    # Create sample input
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 16), device=device)

    with torch.no_grad():
        outputs = model(input_ids, return_full_trace=True)

    # Analyze the output embeddings (before lm_head)
    logits = outputs['logits']
    print(f"Output logits shape: {logits.shape}")

    # Get hidden states from trace if available
    if 'trace' in outputs and outputs['trace']:
        print(f"DPS state: p_max={outputs['dps_state'].p_max}, tokens={outputs['dps_state'].tokens}")

    # Test 3: Lacunary Flow System integration
    print("\n" + "-" * 40)
    print("TEST 3: Lacunary Flow System Integration")
    print("-" * 40)

    flow_system = LacunaryFlowSystem(
        hidden_dim=config.hidden_dim,
        noetic_dim=40,
        num_layers=config.num_layers,
    ).to(device)

    # Simulate layer hidden states
    layer_states = [torch.randn(1, 16, config.hidden_dim, device=device) for _ in range(config.num_layers)]

    flow_result = flow_system(layer_states, input_coherence=0.8, output_coherence=0.7)

    print(f"Processed {len(flow_result['processed_states'])} layers")
    print(f"Depth recommendation: {flow_result['depth_recommendation']}")

    trace = flow_result['trace']
    if trace:
        print(f"Flow coherence trajectory: {[f'{c:.2f}' for c in trace.get_coherence_trajectory()[:5]]}...")
        print(f"Quality score: {trace.quality_score:.3f}")

    stats = flow_result['stats']
    print(f"Pattern bank size: {stats['pattern_bank']['total_patterns']}")

    # Test 4: Compare coherence metrics
    print("\n" + "-" * 40)
    print("TEST 4: Coherence Detection Summary")
    print("-" * 40)

    print("\nKey Findings:")
    print(f"  - Structured data has LOWER lacunarity ({analysis_structured['lacunarity']:.3f})")
    print(f"  - Random data has HIGHER lacunarity ({analysis_random['lacunarity']:.3f})")
    print(f"  - Lacunarity ratio: {analysis_random['lacunarity'] / analysis_structured['lacunarity']:.2f}x")

    lac_diff = analysis_random['lacunarity'] - analysis_structured['lacunarity']
    if lac_diff > 0.1:
        print("\n  RESULT: Lacunarity CAN distinguish coherent from gibberish!")
        print("  Higher lacunarity = more gaps = less coherent = gibberish")
    else:
        print("\n  RESULT: Need more tuning of lacunarity thresholds")

    print("\n" + "=" * 80)
    print("Integration test complete!")
    print("=" * 80)


def test_synthetic():
    """Run synthetic tests without model checkpoint."""
    print("\n" + "-" * 40)
    print("SYNTHETIC COHERENCE TEST")
    print("-" * 40)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create NF encoder
    nf_encoder = NoeticFractalEncoder(embed_dim=40, noetic_dim=40).to(device)

    # Test different types of sequences
    test_cases = [
        ("Highly structured (coherent)", lambda: create_structured_sequence(32, 40, device)),
        ("Random noise (gibberish)", lambda: torch.randn(32, 40, device=device) * 2),
        ("Repetitive (eigenfractal-like)", lambda: torch.randn(1, 40, device=device).repeat(32, 1)),
        ("Mixed (semi-coherent)", lambda: create_mixed_sequence(32, 40, device)),
    ]

    results = []
    for name, create_fn in test_cases:
        hidden = create_fn()
        analysis = analyze_coherence(hidden, nf_encoder)
        results.append((name, analysis))

        print(f"\n{name}:")
        print(f"  Lacunarity: {analysis['lacunarity']:.3f}")
        print(f"  Complexity: {analysis['complexity']:.3f}")
        print(f"  Clustering: {analysis['clustering']:.3f}")
        print(f"  Coherence Score: {analysis['coherence_score']:.3f}")
        print(f"  Is Coherent: {analysis['is_coherent']}")

    # Summary
    print("\n" + "-" * 40)
    print("COHERENCE DETECTION SUMMARY")
    print("-" * 40)

    sorted_results = sorted(results, key=lambda x: x[1]['coherence_score'], reverse=True)
    print("\nRanked by coherence score:")
    for i, (name, analysis) in enumerate(sorted_results, 1):
        status = "COHERENT" if analysis['is_coherent'] else "INCOHERENT"
        print(f"  {i}. {name}: {analysis['coherence_score']:.3f} [{status}]")

    # Check if the ranking makes sense
    coherent_idx = next(i for i, (n, _) in enumerate(sorted_results) if "structured" in n.lower())
    gibberish_idx = next(i for i, (n, _) in enumerate(sorted_results) if "gibberish" in n.lower())

    if coherent_idx < gibberish_idx:
        print("\n  SUCCESS: Structured (coherent) ranked higher than gibberish!")
    else:
        print("\n  WARNING: Ranking may need threshold adjustment")

    return results


def create_structured_sequence(seq_len: int, dim: int, device: str):
    """Create a structured (coherent-like) sequence."""
    x = torch.randn(seq_len, dim, device=device)
    # Add temporal coherence
    for i in range(1, seq_len):
        x[i] = 0.8 * x[i-1] + 0.2 * x[i]
    return x


def create_mixed_sequence(seq_len: int, dim: int, device: str):
    """Create a mixed sequence with some structure."""
    x = torch.randn(seq_len, dim, device=device)
    # Half structured, half random
    for i in range(1, seq_len // 2):
        x[i] = 0.6 * x[i-1] + 0.4 * x[i]
    return x


def test_actual_generation():
    """Test actual text generation with coherence monitoring."""
    print("=" * 80)
    print("TEXT GENERATION WITH COHERENCE MONITORING")
    print("=" * 80)

    # Use CUDA if available (user has Python 3.11/3.12 with CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Import tokenizer
    sys.path.insert(0, 'src')
    from tks.core.tokenizer import TKSTokenizer

    # Check for checkpoint
    checkpoint_paths = [
        "checkpoints/v7_discovery/v7_discovery_model.pt",
        "checkpoints/v6_best.pt",
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("No checkpoint found. Cannot test generation.")
        return

    # Load model
    try:
        model, config = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create tokenizer with matching vocab size
    tokenizer = TKSTokenizer()
    print(f"Tokenizer vocab size: {tokenizer.actual_vocab_size}")
    print(f"Model vocab size: {config.vocab_size}")

    # Create NF encoder for coherence analysis
    nf_encoder = NoeticFractalEncoder(
        embed_dim=min(config.hidden_dim, 40),
        noetic_dim=40,
    ).to(device)

    # Test prompts
    prompts = [
        "The mind seeks",
        "A1 +T B2 ->",
        "Knowledge flows",
    ]

    print("\n" + "-" * 40)
    print("GENERATION TEST: Standard vs Coherence-Gated")
    print("-" * 40)

    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")

        # Generate without coherence gating
        print("\n[Standard Generation]")
        standard_output, standard_scores = generate_with_coherence(
            model, tokenizer, prompt, max_length=30, temperature=1.0,
            nf_encoder=nf_encoder, use_coherence_gating=False, device=device
        )
        print(f"  Output: {standard_output}")
        print(f"  Avg coherence: {sum(standard_scores)/len(standard_scores):.3f}" if standard_scores else "  No scores")

        # Generate with coherence gating
        print("\n[Coherence-Gated Generation]")
        gated_output, gated_scores = generate_with_coherence(
            model, tokenizer, prompt, max_length=30, temperature=1.0,
            nf_encoder=nf_encoder, use_coherence_gating=True,
            coherence_threshold=0.6, device=device
        )
        print(f"  Output: {gated_output}")
        print(f"  Avg coherence: {sum(gated_scores)/len(gated_scores):.3f}" if gated_scores else "  No scores")

    # Test high temperature (more likely to produce gibberish)
    print("\n" + "-" * 40)
    print("HIGH TEMPERATURE TEST (gibberish-prone)")
    print("-" * 40)

    prompt = "The pattern"
    print(f"\n--- Prompt: '{prompt}' (temp=2.0) ---")

    print("\n[Standard - High Temp]")
    high_temp_standard, high_temp_scores = generate_with_coherence(
        model, tokenizer, prompt, max_length=40, temperature=2.0,
        nf_encoder=nf_encoder, use_coherence_gating=False, device=device
    )
    print(f"  Output: {high_temp_standard}")
    print(f"  Avg coherence: {sum(high_temp_scores)/len(high_temp_scores):.3f}" if high_temp_scores else "  No scores")

    print("\n[Coherence-Gated - High Temp]")
    high_temp_gated, high_temp_gated_scores = generate_with_coherence(
        model, tokenizer, prompt, max_length=40, temperature=2.0,
        nf_encoder=nf_encoder, use_coherence_gating=True,
        coherence_threshold=0.5, device=device
    )
    print(f"  Output: {high_temp_gated}")
    print(f"  Avg coherence: {sum(high_temp_gated_scores)/len(high_temp_gated_scores):.3f}" if high_temp_gated_scores else "  No scores")

    print("\n" + "=" * 80)
    print("Generation test complete!")
    print("=" * 80)


def generate_with_coherence(
    model, tokenizer, prompt: str, max_length: int = 50,
    temperature: float = 1.0, nf_encoder=None,
    use_coherence_gating: bool = False, coherence_threshold: float = 0.5,
    device: str = "cpu"
):
    """Generate text with optional coherence gating."""
    # Encode prompt (simple character-level)
    input_ids = []
    input_ids.append(tokenizer.vocab.get("<BOS>", 0))
    for char in prompt:
        input_ids.append(tokenizer.vocab.get(char, tokenizer.vocab.get("<UNK>", 1)))

    input_ids = torch.tensor([input_ids], device=device)
    generated = input_ids.clone()

    coherence_scores = []
    hidden_buffer = []

    with torch.no_grad():
        for step in range(max_length):
            # Forward pass
            outputs = model(generated)
            logits = outputs['logits'][:, -1, :] / temperature

            # Sample candidates
            probs = F.softmax(logits, dim=-1)

            if use_coherence_gating and nf_encoder is not None and len(hidden_buffer) > 2:
                # Try multiple candidates and pick most coherent
                best_token = None
                best_coherence = -1

                for _ in range(5):  # Try 5 candidates
                    candidate = torch.multinomial(probs, num_samples=1)

                    # Quick coherence check using recent hidden states
                    recent_hidden = torch.stack(hidden_buffer[-8:], dim=0)
                    analysis = analyze_coherence(recent_hidden, nf_encoder)

                    if analysis['coherence_score'] > best_coherence:
                        best_coherence = analysis['coherence_score']
                        best_token = candidate

                    if analysis['coherence_score'] >= coherence_threshold:
                        break

                next_token = best_token if best_token is not None else torch.multinomial(probs, num_samples=1)
                coherence_scores.append(best_coherence)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Store hidden state for analysis
            if 'hidden_states' in outputs and outputs['hidden_states']:
                hidden_buffer.append(outputs['hidden_states'][-1][:, -1, :].squeeze(0))
            else:
                # Use logits as proxy for hidden state
                hidden_buffer.append(logits.squeeze(0)[:min(40, logits.shape[-1])])

            # Stop on EOS
            if next_token.item() == tokenizer.vocab.get("<EOS>", 3):
                break

    # Decode
    output_text = tokenizer.decode(generated[0].tolist())
    return output_text, coherence_scores


def test_integrated_pipeline():
    """Test the full integrated coherence pipeline with trained classifier."""
    print("=" * 80)
    print("INTEGRATED COHERENCE PIPELINE TEST")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Import wrapper
    sys.path.insert(0, '.')
    from tks_features.coherent_tks_wrapper import CoherentTKSModel, CoherenceConfig

    # Check for trained classifier
    classifier_path = Path("checkpoints/coherence_classifier_v2.pt")
    if not classifier_path.exists():
        print("Trained classifier not found. Run train_coherence_model.py first.")
        return

    # Load model
    checkpoint_path = "checkpoints/v7_discovery/v7_discovery_model.pt"
    if not Path(checkpoint_path).exists():
        print("No model checkpoint found.")
        return

    try:
        model, config = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create coherent wrapper with trained classifier
    coherence_config = CoherenceConfig(
        enabled=True,
        threshold=0.5,
        use_trained_classifier=True,
        classifier_path=str(classifier_path),
    )

    coherent_model = CoherentTKSModel(
        base_model=model,
        config=coherence_config,
        embed_dim=config.hidden_dim,
    )

    # Test text samples
    test_samples = [
        # Coherent TKS text
        ("Coherent TKS", "The spiritual mind combines with emotional force to create balance."),
        ("Coherent equation", "A1 +T B2 represents the union of spiritual and emotional energies."),
        # Gibberish
        ("Shuffled words", "force emotional The with combines mind balance create to spiritual."),
        ("Random symbols", "A1 B5 C3 X9 +T -T o ^ random gibberish text here."),
        # Edge cases
        ("Mixed", "The A1 pattern shows B2 connection with C3 energy flow."),
    ]

    print("\n" + "-" * 40)
    print("TRAINED CLASSIFIER COHERENCE SCORES")
    print("-" * 40)

    for name, text in test_samples:
        score = coherent_model.classify_text_coherence(text)
        status = "COHERENT" if score > 0.5 else "INCOHERENT"
        print(f"\n[{name}]")
        print(f"  Text: {text[:60]}...")
        print(f"  Score: {score:.3f} [{status}]")

    print("\n" + "=" * 80)
    print("Integrated pipeline test complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Run generation test")
    parser.add_argument("--integrated", action="store_true", help="Run integrated pipeline test")
    args = parser.parse_args()

    if args.integrated:
        test_integrated_pipeline()
    elif args.generate:
        test_actual_generation()
    else:
        test_with_simple_tokenizer()

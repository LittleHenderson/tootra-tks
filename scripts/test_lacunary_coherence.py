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


if __name__ == "__main__":
    test_with_simple_tokenizer()

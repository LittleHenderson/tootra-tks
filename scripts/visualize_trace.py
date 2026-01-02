#!/usr/bin/env python3
"""
Visualize TKS Model Trace - Human-Readable Trace Output

This script takes a single input text, runs inference with trace enabled,
and pretty-prints the trace in human-readable format showing:
- Which noetics activated at each layer
- Attractor convergence dynamics
- RPM (Desire/Wisdom/Power) decisions
- Foundation classification
- World-level analysis

Usage:
    # Basic usage with text input
    python scripts/visualize_trace.py --text "A woman loved a man"

    # With specific model
    python scripts/visualize_trace.py \
        --model-path output/v4_joint_finetuned.pt \
        --text "Power corrupts absolutely"

    # JSON output for programmatic use
    python scripts/visualize_trace.py --text "Test" --format json

    # Quick test with synthetic model
    python scripts/visualize_trace.py --quick

Author: TKS-Quantum Visualization Agent
Date: 2025-12-23
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig, TOTAL_DIM
from tks_llm_core_v2 import FOUNDATION_NAMES, NOETIC_NAMES


# =============================================================================
# CANONICAL TKS DEFINITIONS
# =============================================================================

WORLD_NAMES = {
    'A': 'Atziluth (Spiritual)',
    'B': 'Briah (Mental)',
    'C': 'Yetzirah (Emotional)',
    'D': 'Assiah (Physical)',
}

# Noetic names indexed 0-9
NOETIC_DISPLAY_NAMES = {
    0: 'Idea (nu_0)',
    1: 'Mind (nu_1)',
    2: 'Positive (nu_2)',
    3: 'Negative (nu_3)',
    4: 'Vibration (nu_4)',
    5: 'Female (nu_5)',
    6: 'Male (nu_6)',
    7: 'Rhythm (nu_7)',
    8: 'Above/Cause (nu_8)',
    9: 'Below/Effect (nu_9)',
}

# Foundation display names (1-indexed)
FOUNDATION_DISPLAY_NAMES = {
    1: 'Unity (F1)',
    2: 'Wisdom (F2)',
    3: 'Life (F3)',
    4: 'Companionship (F4)',
    5: 'Power (F5)',
    6: 'Material (F6)',
    7: 'Lust (F7)',
}


# =============================================================================
# TRACE ANALYSIS CLASSES
# =============================================================================

@dataclass
class NoeticActivation:
    """Single noetic activation record."""
    noetic_idx: int
    noetic_name: str
    weight: float
    rank: int


@dataclass
class LayerTrace:
    """Trace for a single transformer layer."""
    layer_idx: int
    top_noetics: List[NoeticActivation]
    attention_entropy: float
    has_expressiveness: bool


@dataclass
class AttractorTrace:
    """Attractor convergence trace."""
    converged: bool
    iterations: int
    final_delta: float
    variance_reduction: float
    convergence_quality: str


@dataclass
class RPMTrace:
    """RPM gating trace."""
    desire_score: float
    wisdom_score: float
    power_score: float
    gate_value: float
    dominant_foundation: int
    foundation_name: str
    foundation_scores: Dict[str, float]


@dataclass
class WorldAnalysis:
    """Per-world analysis of hidden state."""
    world: str
    world_name: str
    mean_activation: float
    dominant_noetic: int
    dominant_noetic_name: str
    energy: float


@dataclass
class TraceVisualization:
    """Complete trace visualization."""
    input_text: str
    input_tokens: List[int]
    sequence_length: int
    vocab_size: int

    # Per-layer analysis
    layer_traces: List[LayerTrace]

    # Attractor analysis
    attractor: AttractorTrace

    # RPM analysis
    rpm: RPMTrace

    # World analysis
    worlds: List[WorldAnalysis]

    # Predictions
    top_predictions: List[Dict[str, Any]]
    prediction_confidence: float


# =============================================================================
# TRACE EXTRACTION
# =============================================================================

def analyze_noetic_weights(weights: torch.Tensor, top_k: int = 3) -> List[NoeticActivation]:
    """
    Analyze noetic router weights and return top activations.

    Args:
        weights: [10] noetic weights
        top_k: Number of top activations to return

    Returns:
        List of top NoeticActivation records
    """
    if weights.dim() > 1:
        weights = weights.mean(dim=tuple(range(weights.dim() - 1)))

    # Get top-k noetics
    values, indices = weights.topk(min(top_k, 10))

    activations = []
    for rank, (idx, val) in enumerate(zip(indices.tolist(), values.tolist())):
        activations.append(NoeticActivation(
            noetic_idx=idx,
            noetic_name=NOETIC_DISPLAY_NAMES.get(idx, f'Unknown ({idx})'),
            weight=val,
            rank=rank + 1,
        ))

    return activations


def compute_attention_entropy(attn_weights) -> float:
    """
    Compute entropy of attention distribution.

    Higher entropy = more distributed attention
    Lower entropy = more focused attention

    Args:
        attn_weights: Attention weight tensor or list of tensors

    Returns:
        Average entropy across heads/scales
    """
    if attn_weights is None:
        return 0.0

    # Handle list of tensors (multi-scale attention)
    if isinstance(attn_weights, list):
        if len(attn_weights) == 0:
            return 0.0
        # Compute entropy for each scale separately and average
        entropies = []
        for aw in attn_weights:
            if isinstance(aw, torch.Tensor) and aw.dim() >= 2:
                flat = aw.view(-1, aw.shape[-1])
                eps = 1e-10
                ent = -(flat * (flat + eps).log()).sum(dim=-1).mean()
                entropies.append(ent.item())
        return sum(entropies) / len(entropies) if entropies else 0.0

    if not isinstance(attn_weights, torch.Tensor):
        return 0.0

    # Flatten to [num_distributions, seq, seq]
    if attn_weights.dim() < 2:
        return 0.0

    flat = attn_weights.view(-1, attn_weights.shape[-2], attn_weights.shape[-1])

    # Compute entropy per distribution
    eps = 1e-10
    entropy = -(flat * (flat + eps).log()).sum(dim=-1).mean()

    return entropy.item()


def analyze_attractor(trace: Dict[str, Any]) -> AttractorTrace:
    """
    Analyze attractor convergence from trace.

    Args:
        trace: Model trace dictionary

    Returns:
        AttractorTrace with convergence analysis
    """
    converged = trace.get('attractor_converged', False)
    iterations = trace.get('attractor_iterations', 0)

    # Compute variance reduction if trajectory available
    variance_reduction = 0.0
    if 'trajectory' in trace and trace['trajectory']:
        traj = trace['trajectory']
        if len(traj) >= 2:
            initial_var = traj[0].var().item()
            final_var = traj[-1].var().item()
            if initial_var > 1e-8:
                variance_reduction = 1.0 - final_var / initial_var

    # Estimate final delta from attractor state
    final_delta = 0.0
    if 'attractor' in trace:
        attractor = trace['attractor']
        # Handle tensor or dict
        if isinstance(attractor, torch.Tensor) and attractor.dim() >= 2:
            # Approximate delta from gradient magnitude
            final_delta = attractor.std().item()
        elif isinstance(attractor, dict) and 'attractor' in attractor:
            # Nested attractor dict
            attractor_tensor = attractor['attractor']
            if isinstance(attractor_tensor, torch.Tensor) and attractor_tensor.dim() >= 2:
                final_delta = attractor_tensor.std().item()

    # Determine convergence quality
    if converged and iterations <= 5:
        quality = "EXCELLENT"
    elif converged and iterations <= 10:
        quality = "GOOD"
    elif converged:
        quality = "SLOW"
    else:
        quality = "FAILED"

    return AttractorTrace(
        converged=converged,
        iterations=iterations,
        final_delta=final_delta,
        variance_reduction=variance_reduction,
        convergence_quality=quality,
    )


def analyze_rpm(trace: Dict[str, Any]) -> RPMTrace:
    """
    Analyze RPM gating from trace.

    Args:
        trace: Model trace dictionary

    Returns:
        RPMTrace with D/W/P analysis
    """
    # Get DWP scores
    if 'dwp_scores' in trace:
        dwp = trace['dwp_scores']  # [batch, seq, 7, 3]
        # Average over batch and sequence
        dwp_mean = dwp.mean(dim=(0, 1))  # [7, 3]

        # Get scores for dominant foundation
        gate_scores = dwp_mean.prod(dim=-1)  # [7]
        dominant_idx = gate_scores.argmax().item()

        desire = dwp_mean[dominant_idx, 0].item()
        wisdom = dwp_mean[dominant_idx, 1].item()
        power = dwp_mean[dominant_idx, 2].item()

        # Foundation scores
        foundation_scores = {}
        for f_idx in range(7):
            f_name = FOUNDATION_DISPLAY_NAMES.get(f_idx + 1, f'F{f_idx + 1}')
            foundation_scores[f_name] = gate_scores[f_idx].item()

    else:
        desire, wisdom, power = 0.5, 0.5, 0.5
        dominant_idx = 0
        foundation_scores = {}

    # Get gate value
    gate_value = trace.get('rpm_gate', torch.tensor(0.5))
    if isinstance(gate_value, torch.Tensor):
        gate_value = gate_value.mean().item()

    return RPMTrace(
        desire_score=desire,
        wisdom_score=wisdom,
        power_score=power,
        gate_value=gate_value,
        dominant_foundation=dominant_idx + 1,
        foundation_name=FOUNDATION_DISPLAY_NAMES.get(dominant_idx + 1, f'F{dominant_idx + 1}'),
        foundation_scores=foundation_scores,
    )


def analyze_worlds(hidden_state: torch.Tensor) -> List[WorldAnalysis]:
    """
    Analyze hidden state by world (10D slices).

    Args:
        hidden_state: [batch, seq, 40] or [seq, 40] or [40]

    Returns:
        List of WorldAnalysis for each world
    """
    # Flatten to [40]
    if hidden_state.dim() > 1:
        hidden_state = hidden_state.mean(dim=tuple(range(hidden_state.dim() - 1)))

    worlds = []
    world_letters = ['A', 'B', 'C', 'D']

    for i, letter in enumerate(world_letters):
        start = i * 10
        end = start + 10
        world_slice = hidden_state[start:end]

        # Mean activation
        mean_activation = world_slice.mean().item()

        # Dominant noetic within world
        dominant_idx = world_slice.argmax().item()

        # Energy (L2 norm)
        energy = world_slice.norm().item()

        worlds.append(WorldAnalysis(
            world=letter,
            world_name=WORLD_NAMES.get(letter, letter),
            mean_activation=mean_activation,
            dominant_noetic=dominant_idx,
            dominant_noetic_name=NOETIC_DISPLAY_NAMES.get(dominant_idx, f'nu_{dominant_idx}'),
            energy=energy,
        ))

    return worlds


def analyze_predictions(
    logits: torch.Tensor,
    vocab_size: int,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Analyze model predictions from logits.

    Args:
        logits: [batch, seq, vocab] or [seq, vocab] or [vocab]
        vocab_size: Vocabulary size
        top_k: Number of top predictions

    Returns:
        List of top predictions with probabilities
    """
    # Get last position logits
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # [batch, vocab]
    if logits.dim() == 2:
        logits = logits.mean(dim=0)  # [vocab]

    # Get probabilities
    probs = F.softmax(logits, dim=-1)

    # Top-k
    values, indices = probs.topk(min(top_k, vocab_size))

    predictions = []
    for idx, prob in zip(indices.tolist(), values.tolist()):
        predictions.append({
            'token_id': idx,
            'probability': prob,
            'rank': len(predictions) + 1,
        })

    return predictions


# =============================================================================
# VISUALIZATION BUILDER
# =============================================================================

def build_trace_visualization(
    model: TKSNoeticLM,
    tokens: torch.Tensor,
    input_text: str,
    device: torch.device,
) -> TraceVisualization:
    """
    Build complete trace visualization from model inference.

    Args:
        model: TKS model
        tokens: [1, seq] input tokens
        input_text: Original input text
        device: Device for computation

    Returns:
        TraceVisualization dataclass
    """
    model.eval()

    with torch.no_grad():
        output = model(tokens.to(device), return_full_trace=True)

    trace = output.get('trace', {})
    logits = output['logits']

    # Layer traces
    layer_traces = []
    if 'blocks' in trace:
        for layer_idx, block_trace in enumerate(trace['blocks']):
            # Noetic weights
            noetic_weights = block_trace.get('noetic_weights')
            if noetic_weights is not None:
                top_noetics = analyze_noetic_weights(noetic_weights)
            else:
                top_noetics = []

            # Attention entropy
            attn_weights = block_trace.get('attn_weights')
            attn_entropy = compute_attention_entropy(attn_weights)

            # Expressiveness
            has_expr = 'expressiveness' in block_trace

            layer_traces.append(LayerTrace(
                layer_idx=layer_idx,
                top_noetics=top_noetics,
                attention_entropy=attn_entropy,
                has_expressiveness=has_expr,
            ))

    # Attractor
    attractor_trace = analyze_attractor(trace)

    # RPM
    rpm_trace = analyze_rpm(trace)

    # Worlds
    hidden_state = trace.get('attractor', output['gated_output'])
    world_analysis = analyze_worlds(hidden_state)

    # Predictions
    predictions = analyze_predictions(logits, model.vocab_size)
    confidence = predictions[0]['probability'] if predictions else 0.0

    return TraceVisualization(
        input_text=input_text,
        input_tokens=tokens[0].tolist() if tokens.dim() == 2 else tokens.tolist(),
        sequence_length=tokens.shape[-1],
        vocab_size=model.vocab_size,
        layer_traces=layer_traces,
        attractor=attractor_trace,
        rpm=rpm_trace,
        worlds=world_analysis,
        top_predictions=predictions,
        prediction_confidence=confidence,
    )


# =============================================================================
# TEXT OUTPUT FORMATTING
# =============================================================================

def format_text_output(viz: TraceVisualization) -> str:
    """Format trace visualization as human-readable text."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("TKS MODEL TRACE VISUALIZATION")
    lines.append("=" * 70)
    lines.append("")

    # Input
    lines.append("INPUT")
    lines.append("-" * 70)
    lines.append(f"Text: {viz.input_text}")
    lines.append(f"Tokens: {viz.input_tokens[:10]}{'...' if len(viz.input_tokens) > 10 else ''}")
    lines.append(f"Sequence length: {viz.sequence_length}")
    lines.append("")

    # Layer-by-layer analysis
    lines.append("LAYER-BY-LAYER NOETIC ANALYSIS")
    lines.append("-" * 70)
    for layer in viz.layer_traces:
        lines.append(f"Layer {layer.layer_idx}:")
        if layer.top_noetics:
            for act in layer.top_noetics[:3]:
                bar = "#" * int(act.weight * 20)
                lines.append(f"  [{act.rank}] {act.noetic_name}: {act.weight:.3f} {bar}")
        else:
            lines.append("  (no noetic weights available)")
        if layer.attention_entropy > 0:
            lines.append(f"  Attention entropy: {layer.attention_entropy:.3f}")
        if layer.has_expressiveness:
            lines.append("  [+] TKS Expressiveness Stack active")
        lines.append("")

    # Attractor convergence
    lines.append("ATTRACTOR CONVERGENCE")
    lines.append("-" * 70)
    status = "CONVERGED" if viz.attractor.converged else "NOT CONVERGED"
    lines.append(f"Status: {status} ({viz.attractor.convergence_quality})")
    lines.append(f"Iterations: {viz.attractor.iterations}")
    lines.append(f"Final delta: {viz.attractor.final_delta:.6f}")
    lines.append(f"Variance reduction: {viz.attractor.variance_reduction * 100:.1f}%")
    lines.append("")

    # RPM analysis
    lines.append("RPM GATING (Desire/Wisdom/Power)")
    lines.append("-" * 70)
    lines.append(f"Desire (D):  {viz.rpm.desire_score:.3f} {'#' * int(viz.rpm.desire_score * 20)}")
    lines.append(f"Wisdom (W):  {viz.rpm.wisdom_score:.3f} {'#' * int(viz.rpm.wisdom_score * 20)}")
    lines.append(f"Power (P):   {viz.rpm.power_score:.3f} {'#' * int(viz.rpm.power_score * 20)}")
    lines.append(f"Gate value:  {viz.rpm.gate_value:.3f}")
    lines.append(f"Dominant:    {viz.rpm.foundation_name}")
    lines.append("")

    if viz.rpm.foundation_scores:
        lines.append("Foundation scores:")
        for f_name, score in sorted(viz.rpm.foundation_scores.items(), key=lambda x: -x[1]):
            bar = "#" * int(score * 20)
            lines.append(f"  {f_name}: {score:.3f} {bar}")
        lines.append("")

    # World analysis
    lines.append("WORLD ANALYSIS (40D Noetic Space)")
    lines.append("-" * 70)
    for world in viz.worlds:
        lines.append(f"{world.world} - {world.world_name}")
        lines.append(f"  Mean activation: {world.mean_activation:.4f}")
        lines.append(f"  Dominant noetic: {world.dominant_noetic_name}")
        lines.append(f"  Energy (L2): {world.energy:.4f}")
    lines.append("")

    # Predictions
    lines.append("TOP PREDICTIONS")
    lines.append("-" * 70)
    for pred in viz.top_predictions[:5]:
        prob_pct = pred['probability'] * 100
        bar = "#" * int(prob_pct / 5)
        lines.append(f"  [{pred['rank']}] Token {pred['token_id']}: {prob_pct:.1f}% {bar}")
    lines.append(f"\nPrediction confidence: {viz.prediction_confidence * 100:.1f}%")
    lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_json_output(viz: TraceVisualization) -> str:
    """Format trace visualization as JSON."""
    # Convert dataclasses to dicts
    data = {
        'input': {
            'text': viz.input_text,
            'tokens': viz.input_tokens,
            'sequence_length': viz.sequence_length,
            'vocab_size': viz.vocab_size,
        },
        'layers': [
            {
                'layer_idx': layer.layer_idx,
                'top_noetics': [asdict(n) for n in layer.top_noetics],
                'attention_entropy': layer.attention_entropy,
                'has_expressiveness': layer.has_expressiveness,
            }
            for layer in viz.layer_traces
        ],
        'attractor': asdict(viz.attractor),
        'rpm': asdict(viz.rpm),
        'worlds': [asdict(w) for w in viz.worlds],
        'predictions': viz.top_predictions,
        'prediction_confidence': viz.prediction_confidence,
    }

    return json.dumps(data, indent=2)


# =============================================================================
# TEXT TOKENIZATION (Simple)
# =============================================================================

def simple_tokenize(text: str, vocab_size: int) -> torch.Tensor:
    """
    Simple tokenization by character/word hash.

    This is a fallback when no proper tokenizer is available.

    Args:
        text: Input text
        vocab_size: Vocabulary size

    Returns:
        [1, seq] token tensor
    """
    # Simple word-level tokenization with hash
    words = text.lower().split()
    tokens = [hash(w) % vocab_size for w in words]

    # Ensure at least one token
    if not tokens:
        tokens = [0]

    return torch.tensor([tokens], dtype=torch.long)


# =============================================================================
# QUICK TEST MODE
# =============================================================================

def run_quick_test(device: torch.device) -> TraceVisualization:
    """
    Run quick test with synthetic model.

    Args:
        device: Device for computation

    Returns:
        TraceVisualization from synthetic model
    """
    print("\n--- QUICK TEST MODE ---")
    print("Creating synthetic model...")

    # Create small model
    config = TKSNoeticLMConfig(
        vocab_size=256,
        max_seq_len=32,
        num_layers=2,
        num_scales=2,
        dropout=0.0,
        use_attractor=True,
        use_stable_attractor=True,
        use_rpm=True,
    )
    model = TKSNoeticLM(config).to(device)

    # Sample text
    test_text = "A woman loved a man"
    tokens = simple_tokenize(test_text, config.vocab_size)

    print(f"Running inference on: '{test_text}'")

    viz = build_trace_visualization(model, tokens, test_text, device)

    return viz


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize TKS model trace in human-readable format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (no model needed)
    python scripts/visualize_trace.py --quick

    # With text input
    python scripts/visualize_trace.py --text "A woman loved a man"

    # With model
    python scripts/visualize_trace.py \\
        --model-path output/v4_joint_finetuned.pt \\
        --text "Power corrupts absolutely"

    # JSON output
    python scripts/visualize_trace.py --text "Test" --format json
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with synthetic model'
    )

    parser.add_argument(
        '--text',
        type=str,
        help='Input text to analyze'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained TKS v4 model checkpoint'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='text',
        choices=['text', 'json'],
        help='Output format (default: text)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: stdout)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto/cpu/cuda)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Quick test mode
    if args.quick:
        viz = run_quick_test(device)
        output = format_text_output(viz) if args.format == 'text' else format_json_output(viz)
        print(output)
        return

    # Require text input
    if not args.text:
        print("ERROR: --text is required")
        print("Use --quick for a quick test without input")
        sys.exit(1)

    # Load model if specified
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)

        print(f"\nLoading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict):
                config = TKSNoeticLMConfig(**config)
        else:
            config = TKSNoeticLMConfig()

        model = TKSNoeticLM(config).to(device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        print(f"Model loaded: {config.vocab_size} vocab, {config.num_layers} layers")

    else:
        # Create default model
        print("\nNo model specified, using default configuration...")
        config = TKSNoeticLMConfig(
            vocab_size=512,
            max_seq_len=64,
            num_layers=4,
            use_attractor=True,
            use_stable_attractor=True,
            use_rpm=True,
        )
        model = TKSNoeticLM(config).to(device)
        print(f"Created default model: {config.vocab_size} vocab, {config.num_layers} layers")

    # Tokenize input
    tokens = simple_tokenize(args.text, config.vocab_size)
    tokens = tokens[:, :config.max_seq_len]  # Truncate if needed

    print(f"\nAnalyzing: '{args.text}'")
    print(f"Tokens: {tokens[0].tolist()}")

    # Build visualization
    viz = build_trace_visualization(model, tokens, args.text, device)

    # Format output
    if args.format == 'text':
        output = format_text_output(viz)
    else:
        output = format_json_output(viz)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output)
        print(f"\nOutput saved to: {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Attractor Convergence Verification Script

This script loads a trained TKS model checkpoint and verifies the attractor
convergence properties:

1. Convergence rate: Percentage of inputs that converge to a fixed point
2. Lipschitz estimates: Verify contraction property (L < 1)
3. Iteration analysis: Average iterations to convergence
4. Comparison: Legacy vs Stable attractor if available

Usage:
    python scripts/verify_attractor_convergence.py --checkpoint output/cuda_models/best_model.pt
    python scripts/verify_attractor_convergence.py --checkpoint output/cuda_models/best_model.pt --num-samples 200
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TKS LLM Core Pipeline
try:
    from tks_llm_core_v2 import (
        TKSLLMCorePipeline,
        StableAttractorLayer,
        AttractorComputationLayer,
        attractor_convergence_loss,
        compute_lipschitz_penalty,
        TOTAL_DIM,
    )
    TKS_PIPELINE_AVAILABLE = True
except ImportError:
    TKS_PIPELINE_AVAILABLE = False
    logger.warning("TKSLLMCorePipeline not available.")


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint and return model + config."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Determine vocab size from checkpoint
    state_dict = checkpoint['model_state_dict']
    embedding_key = 'embedding.embedding.weight'
    if embedding_key in state_dict:
        vocab_size = state_dict[embedding_key].shape[0]
    else:
        vocab_size = 128  # Default fallback

    logger.info(f"  Config: {json.dumps(config, indent=2, default=str)}")
    logger.info(f"  Vocab size: {vocab_size}")

    return {
        'checkpoint': checkpoint,
        'config': config,
        'vocab_size': vocab_size,
    }


def create_model(
    vocab_size: int,
    config: Dict[str, Any],
    use_stable_attractor: bool = True,
    device: torch.device = torch.device('cpu')
) -> TKSLLMCorePipeline:
    """Create TKSLLMCorePipeline model."""
    if not TKS_PIPELINE_AVAILABLE:
        raise ImportError("TKSLLMCorePipeline not available. Install tks_llm_core_v2.")

    model = TKSLLMCorePipeline(
        vocab_size=vocab_size,
        hidden_dim=config.get('hidden_dim', 256),
        noetic_dim=40,
        num_scales=3,
        max_attractor_iter=config.get('num_layers', 4),
        contraction_factor=0.5,
        use_stable_attractor=use_stable_attractor,
    )

    return model.to(device)


def verify_contraction_properties(
    attractor_layer: nn.Module,
    num_samples: int = 100,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Verify contraction properties of the attractor layer.

    Returns:
        Dict with Lipschitz estimates and contraction verification
    """
    results = {
        'is_stable_attractor': isinstance(attractor_layer, StableAttractorLayer),
        'contraction_verified': False,
        'lipschitz_estimates': {},
    }

    if hasattr(attractor_layer, 'verify_contraction'):
        contraction_check = attractor_layer.verify_contraction(num_samples=num_samples)
        results['lipschitz_estimates'] = contraction_check

        # Check if all maps are contractions
        all_contractions = all(
            v for k, v in contraction_check.items() if 'is_contraction' in k
        )
        results['contraction_verified'] = all_contractions

    return results


def run_convergence_test(
    model: TKSLLMCorePipeline,
    num_samples: int = 100,
    seq_len: int = 32,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Run convergence test on the model.

    Returns:
        Dict with convergence rate, iterations, and delta statistics
    """
    model.eval()

    vocab_size = model.vocab_size

    convergence_results = []
    iteration_counts = []
    final_deltas = []
    delta_sequences = []

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random input tokens
            tokens = torch.randint(0, vocab_size, (1, seq_len), device=device)

            # Forward pass with full trace
            output = model(tokens, return_full_trace=True)

            converged = output.get('attractor_converged', False)
            convergence_results.append(converged)

            trace = output.get('trace', {})
            if trace:
                iters = trace.get('attractor_iterations', 0)
                iteration_counts.append(iters)

            # If we can access the attractor layer directly for more details
            if hasattr(model, 'attractor') and hasattr(model.attractor, 'dim'):
                # Run attractor on random noetic input
                noetic_input = torch.randn(1, seq_len, model.attractor.dim, device=device)

                if hasattr(model.attractor, 'forward'):
                    attr_output = model.attractor(noetic_input, return_metrics=True)

                    if 'final_delta' in attr_output:
                        final_deltas.append(attr_output['final_delta'])

                    if 'delta_sequence' in attr_output:
                        delta_sequences.append(attr_output['delta_sequence'])

    # Compute statistics
    convergence_rate = sum(convergence_results) / len(convergence_results) * 100
    avg_iterations = np.mean(iteration_counts) if iteration_counts else 0
    avg_final_delta = np.mean(final_deltas) if final_deltas else 0

    # Analyze delta sequences for monotonic decay
    monotonic_count = 0
    for deltas in delta_sequences:
        if len(deltas) >= 2:
            is_monotonic = all(deltas[i] >= deltas[i+1] for i in range(len(deltas)-1))
            if is_monotonic:
                monotonic_count += 1

    monotonic_rate = monotonic_count / len(delta_sequences) * 100 if delta_sequences else 0

    return {
        'convergence_rate': convergence_rate,
        'avg_iterations': avg_iterations,
        'avg_final_delta': avg_final_delta,
        'monotonic_decay_rate': monotonic_rate,
        'num_samples': num_samples,
        'converged_count': sum(convergence_results),
    }


def compare_attractor_types(
    vocab_size: int,
    config: Dict[str, Any],
    num_samples: int = 50,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Dict[str, Any]]:
    """
    Compare legacy AttractorComputationLayer vs StableAttractorLayer.

    Returns:
        Dict with comparison results for both attractor types
    """
    results = {}

    # Test Stable Attractor
    logger.info("\n--- Testing StableAttractorLayer ---")
    stable_model = create_model(vocab_size, config, use_stable_attractor=True, device=device)
    stable_results = run_convergence_test(stable_model, num_samples=num_samples, device=device)
    stable_contraction = verify_contraction_properties(stable_model.attractor, device=device)

    results['stable'] = {
        'convergence': stable_results,
        'contraction': stable_contraction,
    }

    logger.info(f"  Convergence rate: {stable_results['convergence_rate']:.1f}%")
    logger.info(f"  Avg iterations: {stable_results['avg_iterations']:.1f}")
    logger.info(f"  Contraction verified: {stable_contraction['contraction_verified']}")

    # Test Legacy Attractor
    logger.info("\n--- Testing AttractorComputationLayer (legacy) ---")
    legacy_model = create_model(vocab_size, config, use_stable_attractor=False, device=device)
    legacy_results = run_convergence_test(legacy_model, num_samples=num_samples, device=device)
    legacy_contraction = verify_contraction_properties(legacy_model.attractor, device=device)

    results['legacy'] = {
        'convergence': legacy_results,
        'contraction': legacy_contraction,
    }

    logger.info(f"  Convergence rate: {legacy_results['convergence_rate']:.1f}%")
    logger.info(f"  Avg iterations: {legacy_results['avg_iterations']:.1f}")
    logger.info(f"  Contraction verified: {legacy_contraction['contraction_verified']}")

    # Compute improvement
    improvement = stable_results['convergence_rate'] - legacy_results['convergence_rate']
    results['improvement'] = {
        'convergence_rate_delta': improvement,
        'stable_is_better': stable_results['convergence_rate'] > legacy_results['convergence_rate'],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify Attractor Convergence")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of test samples for convergence verification')
    parser.add_argument('--seq-len', type=int, default=32,
                        help='Sequence length for test inputs')
    parser.add_argument('--compare', action='store_true',
                        help='Compare legacy vs stable attractor (ignores checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if not TKS_PIPELINE_AVAILABLE:
        logger.error("TKSLLMCorePipeline not available. Cannot verify attractor convergence.")
        sys.exit(1)

    results = {}

    # Comparison mode: test both attractor types
    if args.compare:
        logger.info("=" * 70)
        logger.info("ATTRACTOR COMPARISON: Legacy vs Stable")
        logger.info("=" * 70)

        # Use default config
        config = {
            'hidden_dim': 128,
            'num_layers': 4,
        }
        vocab_size = 128

        comparison = compare_attractor_types(vocab_size, config, args.num_samples, device)
        results['comparison'] = comparison

        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Stable convergence: {comparison['stable']['convergence']['convergence_rate']:.1f}%")
        logger.info(f"  Legacy convergence: {comparison['legacy']['convergence']['convergence_rate']:.1f}%")
        logger.info(f"  Improvement: +{comparison['improvement']['convergence_rate_delta']:.1f}%")

    # Checkpoint mode: verify a trained model
    elif args.checkpoint:
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        logger.info("=" * 70)
        logger.info("ATTRACTOR CONVERGENCE VERIFICATION")
        logger.info("=" * 70)

        # Load checkpoint
        ckpt_data = load_checkpoint(args.checkpoint, device)
        config = ckpt_data['config']
        vocab_size = ckpt_data['vocab_size']

        # Determine attractor type from config
        use_stable = config.get('use_stable_attractor', True)
        logger.info(f"Attractor type: {'StableAttractorLayer' if use_stable else 'AttractorComputationLayer (legacy)'}")

        # Create and load model
        model = create_model(vocab_size, config, use_stable_attractor=use_stable, device=device)
        model.load_state_dict(ckpt_data['checkpoint']['model_state_dict'])

        # Run verification
        logger.info("\n--- Convergence Test ---")
        convergence = run_convergence_test(model, args.num_samples, args.seq_len, device)
        results['convergence'] = convergence

        logger.info(f"  Convergence rate: {convergence['convergence_rate']:.1f}%")
        logger.info(f"  Avg iterations: {convergence['avg_iterations']:.1f}")
        logger.info(f"  Avg final delta: {convergence['avg_final_delta']:.6f}")
        logger.info(f"  Monotonic decay rate: {convergence['monotonic_decay_rate']:.1f}%")

        logger.info("\n--- Contraction Verification ---")
        contraction = verify_contraction_properties(model.attractor, num_samples=200, device=device)
        results['contraction'] = contraction

        logger.info(f"  Is StableAttractorLayer: {contraction['is_stable_attractor']}")
        logger.info(f"  Contraction verified: {contraction['contraction_verified']}")

        for key, val in contraction['lipschitz_estimates'].items():
            logger.info(f"    {key}: {val}")

        # Lipschitz penalty test
        logger.info("\n--- Lipschitz Penalty Test ---")
        if isinstance(model.attractor, StableAttractorLayer):
            lip_penalty = compute_lipschitz_penalty(model.attractor, num_samples=100, target_lipschitz=0.9)
            results['lipschitz_penalty'] = lip_penalty.item()
            logger.info(f"  Lipschitz penalty: {lip_penalty.item():.6f}")
            logger.info(f"  Status: {'PASS' if lip_penalty.item() < 0.01 else 'NEEDS IMPROVEMENT'}")

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 70)

        if convergence['convergence_rate'] >= 90:
            status = "EXCELLENT"
        elif convergence['convergence_rate'] >= 70:
            status = "GOOD"
        elif convergence['convergence_rate'] >= 50:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS IMPROVEMENT"

        logger.info(f"  Overall Status: {status}")
        logger.info(f"  Convergence: {convergence['convergence_rate']:.1f}%")
        logger.info(f"  Contraction: {'VERIFIED' if contraction['contraction_verified'] else 'NOT VERIFIED'}")

        results['status'] = status

    else:
        # No checkpoint, just test with random initialization
        logger.info("=" * 70)
        logger.info("ATTRACTOR CONVERGENCE TEST (Random Initialization)")
        logger.info("=" * 70)

        config = {
            'hidden_dim': 128,
            'num_layers': 4,
        }
        vocab_size = 128

        model = create_model(vocab_size, config, use_stable_attractor=True, device=device)

        convergence = run_convergence_test(model, args.num_samples, args.seq_len, device)
        contraction = verify_contraction_properties(model.attractor, device=device)

        results['convergence'] = convergence
        results['contraction'] = contraction

        logger.info(f"\nConvergence rate: {convergence['convergence_rate']:.1f}%")
        logger.info(f"Contraction verified: {contraction['contraction_verified']}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {args.output}")

    return results


if __name__ == '__main__':
    main()

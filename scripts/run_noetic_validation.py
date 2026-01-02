#!/usr/bin/env python3
"""
Noetic Validation Script for TKS Model Checkpoints

This script runs all 4 noetic coherence tests on a model checkpoint:
1. World Separation - Target world should have highest activation (60%+)
2. RPM Differentiation - D/W/P categories should have distinct values (variance > 0.01)
3. Attractor Convergence - Attractor should converge on 80%+ of inputs
4. Noetic Consistency - Similar texts should cluster, dissimilar should separate (> 0.05)

Outputs JSON results to stdout or file, returns exit code 0 if all pass, 1 if any fail.

Usage:
    python scripts/run_noetic_validation.py --model output/model.pt
    python scripts/run_noetic_validation.py --model output/model.pt --output results.json
    python scripts/run_noetic_validation.py --model output/model.pt --verbose

Exit Codes:
    0: All tests passed
    1: One or more tests failed
    2: Error loading model or running tests

Author: TKS CI Agent (Agent C)
Date: 2025-12-22
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    from scripts.train_cuda import SimpleTokenizer
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


# =============================================================================
# THRESHOLDS (from TKS v7.x specifications)
# =============================================================================

@dataclass
class NoeticThresholds:
    """Pass/fail thresholds for noetic coherence metrics."""
    WORLD_SEPARATION_THRESHOLD: float = 0.60
    RPM_VARIANCE_THRESHOLD: float = 0.01
    ATTRACTOR_CONVERGENCE_THRESHOLD: float = 0.80
    NOETIC_CONSISTENCY_THRESHOLD: float = 0.05


THRESHOLDS = NoeticThresholds()


# =============================================================================
# TEST DATA
# =============================================================================

SIMILAR_PAIRS = [
    ("Mental focus and clarity", "Clear thinking and concentration"),
    ("Emotional warmth and love", "Feeling affection and care"),
    ("Physical energy and power", "Bodily strength and vitality"),
    ("Spiritual awareness", "Divine consciousness"),
]

DISSIMILAR_PAIRS = [
    ("Mental focus and clarity", "Physical energy and power"),
    ("Emotional warmth and love", "Spiritual awareness"),
    ("1:7:9", "Fear and anxiety"),
    ("A1 + B2 + C3", "The sun sets in the west"),
]

WORLD_TEXTS = {
    'A': ["A1 + A2 + A3", "Divine realm", "Spiritual awareness A"],
    'B': ["B4 + B5 + B6", "Mental focus", "Cognitive clarity B"],
    'C': ["C7 + C8 + C9", "Emotional flow", "Feeling warmth C"],
    'D': ["D10 + D1 + D2", "Physical power", "Bodily energy D"],
}

RPM_TEST_CASES = {
    'desire': [
        "I want power and control",
        "Desire for manifestation",
        "Vibration of intention",
    ],
    'wisdom': [
        "Understanding the balance",
        "Male and female principles",
        "Wisdom of equilibrium",
    ],
    'power': [
        "Cause and effect in action",
        "The power of consequence",
        "Forces shaping reality",
    ],
}

ATTRACTOR_TEXTS = [
    "1:7:9",
    "A1 + B2 + C3 + D4",
    "The rhythm of cause and effect",
    "Desire leads to manifestation through wisdom",
    "Dynamic: Awareness of pattern triggered",
]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: str, device: Optional[str] = None) -> tuple:
    """Load TKSLLMCorePipeline model from checkpoint."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    if not MODEL_AVAILABLE:
        raise RuntimeError("TKSLLMCorePipeline not available")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 4)
        vocab_size = config.get('vocab_size', None)
        num_scales = config.get('num_scales', 3)
        contraction_factor = config.get('contraction_factor', 0.5)
    else:
        state_dict = checkpoint
        hidden_dim = 256
        num_layers = 4
        vocab_size = None
        num_scales = 3
        contraction_factor = 0.5

    tokenizer = SimpleTokenizer(max_length=256)

    # Determine vocab size from state dict if not in config
    if vocab_size is None:
        # Try to infer from embedding weight
        for key in state_dict:
            if 'embedding' in key and 'weight' in key:
                vocab_size = state_dict[key].shape[0]
                break
        if vocab_size is None:
            vocab_size = tokenizer.actual_vocab_size

    model = TKSLLMCorePipeline(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        noetic_dim=TOTAL_DIM,
        num_scales=num_scales,
        max_attractor_iter=num_layers,
        contraction_factor=contraction_factor,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def get_noetic_state(model, tokenizer, text: str, device) -> Dict:
    """Extract noetic state for input text."""
    input_ids = tokenizer.tokenize(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_tensor)

    gated_output = output['gated_output'][0]  # [seq_len, 40]
    rpm_gate = output['rpm_gate'][0]  # [seq_len]
    attractor_converged = output['attractor_converged']

    content_len = min(len(text) + 2, gated_output.shape[0])
    noetic_state = gated_output[:content_len].mean(dim=0).cpu().numpy()
    rpm_mean = rpm_gate[:content_len].mean().item()

    world_states = {
        'A': noetic_state[0:10],
        'B': noetic_state[10:20],
        'C': noetic_state[20:30],
        'D': noetic_state[30:40],
    }

    return {
        'noetic_state': noetic_state,
        'world_states': world_states,
        'rpm_gate': rpm_mean,
        'attractor_converged': bool(attractor_converged),
    }


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_world_separation(model, tokenizer, device) -> Dict:
    """Test 1: World Separation - Target world should have highest activation."""
    correct_count = 0
    total_count = 0
    details = []

    for target_world, texts in WORLD_TEXTS.items():
        for text in texts:
            state = get_noetic_state(model, tokenizer, text, device)

            world_activations = {}
            for w in 'ABCD':
                world_activations[w] = float(np.linalg.norm(state['world_states'][w]))

            max_world = max(world_activations, key=world_activations.get)
            is_correct = (max_world == target_world)

            if is_correct:
                correct_count += 1
            total_count += 1

            details.append({
                'text': text[:30],
                'target': target_world,
                'predicted': max_world,
                'correct': is_correct,
            })

    correct_rate = correct_count / total_count if total_count > 0 else 0.0

    return {
        'name': 'World Separation',
        'correct_count': correct_count,
        'total_count': total_count,
        'score': float(correct_rate),
        'threshold': THRESHOLDS.WORLD_SEPARATION_THRESHOLD,
        'passed': correct_rate >= THRESHOLDS.WORLD_SEPARATION_THRESHOLD,
        'details': details,
    }


def compute_rpm_differentiation(model, tokenizer, device) -> Dict:
    """Test 2: RPM Differentiation - D/W/P categories should have distinct values."""
    category_means = {}
    for category, texts in RPM_TEST_CASES.items():
        scores = []
        for text in texts:
            state = get_noetic_state(model, tokenizer, text, device)
            scores.append(state['rpm_gate'])
        category_means[category] = float(np.mean(scores))

    means = list(category_means.values())
    variance = float(np.var(means))

    return {
        'name': 'RPM Differentiation',
        'category_means': category_means,
        'score': variance,
        'threshold': THRESHOLDS.RPM_VARIANCE_THRESHOLD,
        'passed': variance > THRESHOLDS.RPM_VARIANCE_THRESHOLD,
    }


def compute_attractor_convergence(model, tokenizer, device) -> Dict:
    """Test 3: Attractor Convergence - Should converge on 80%+ of inputs."""
    results = []
    for text in ATTRACTOR_TEXTS:
        state = get_noetic_state(model, tokenizer, text, device)
        results.append({
            'text': text[:40],
            'converged': state['attractor_converged'],
        })

    converged_count = sum(1 for r in results if r['converged'])
    convergence_rate = converged_count / len(results) if results else 0.0

    return {
        'name': 'Attractor Convergence',
        'converged_count': converged_count,
        'total_count': len(results),
        'score': float(convergence_rate),
        'threshold': THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD,
        'passed': convergence_rate >= THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD,
        'details': results,
    }


def compute_noetic_consistency(model, tokenizer, device) -> Dict:
    """Test 4: Noetic Consistency - Similar texts should cluster, dissimilar should separate."""
    similar_scores = []
    for text1, text2 in SIMILAR_PAIRS:
        state1 = get_noetic_state(model, tokenizer, text1, device)
        state2 = get_noetic_state(model, tokenizer, text2, device)
        sim = cosine_similarity(state1['noetic_state'], state2['noetic_state'])
        similar_scores.append(sim)

    dissimilar_scores = []
    for text1, text2 in DISSIMILAR_PAIRS:
        state1 = get_noetic_state(model, tokenizer, text1, device)
        state2 = get_noetic_state(model, tokenizer, text2, device)
        sim = cosine_similarity(state1['noetic_state'], state2['noetic_state'])
        dissimilar_scores.append(sim)

    mean_similar = np.mean(similar_scores)
    mean_dissimilar = np.mean(dissimilar_scores)
    separation = mean_similar - mean_dissimilar

    return {
        'name': 'Noetic Consistency',
        'mean_similar': float(mean_similar),
        'mean_dissimilar': float(mean_dissimilar),
        'score': float(separation),
        'threshold': THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD,
        'passed': separation > THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD,
    }


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_validation(
    model_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run all noetic validation tests.

    Args:
        model_path: Path to model checkpoint
        output_path: Optional path to save JSON results
        verbose: If True, print detailed output
        device: Device to use ('cuda', 'cpu', or None for auto)

    Returns:
        Dict with all test results and overall pass/fail
    """
    start_time = time.time()

    # Check dependencies
    if not TORCH_AVAILABLE:
        return {
            'error': 'PyTorch not available',
            'all_passed': False,
        }

    if not MODEL_AVAILABLE:
        return {
            'error': 'TKSLLMCorePipeline not available',
            'all_passed': False,
        }

    if not Path(model_path).exists():
        return {
            'error': f'Model not found: {model_path}',
            'all_passed': False,
        }

    # Load model
    if verbose:
        print("=" * 70)
        print("NOETIC COHERENCE VALIDATION")
        print("=" * 70)
        print(f"Model: {model_path}")

    try:
        model, tokenizer, dev = load_model(model_path, device)
        if verbose:
            print(f"Device: {dev}")
            print("Model loaded successfully")
    except Exception as e:
        return {
            'error': f'Failed to load model: {str(e)}',
            'all_passed': False,
        }

    # Run all tests
    tests = []

    if verbose:
        print("\n" + "-" * 70)
        print("Running World Separation Test...")
    world_sep = compute_world_separation(model, tokenizer, dev)
    tests.append(world_sep)
    if verbose:
        status = "PASS" if world_sep['passed'] else "FAIL"
        print(f"  Score: {world_sep['score']:.1%} (threshold: {world_sep['threshold']:.0%}) [{status}]")

    if verbose:
        print("\nRunning RPM Differentiation Test...")
    rpm_diff = compute_rpm_differentiation(model, tokenizer, dev)
    tests.append(rpm_diff)
    if verbose:
        status = "PASS" if rpm_diff['passed'] else "FAIL"
        print(f"  Score: {rpm_diff['score']:.6f} (threshold: {rpm_diff['threshold']}) [{status}]")

    if verbose:
        print("\nRunning Attractor Convergence Test...")
    attractor = compute_attractor_convergence(model, tokenizer, dev)
    tests.append(attractor)
    if verbose:
        status = "PASS" if attractor['passed'] else "FAIL"
        print(f"  Score: {attractor['score']:.0%} (threshold: {attractor['threshold']:.0%}) [{status}]")

    if verbose:
        print("\nRunning Noetic Consistency Test...")
    consistency = compute_noetic_consistency(model, tokenizer, dev)
    tests.append(consistency)
    if verbose:
        status = "PASS" if consistency['passed'] else "FAIL"
        print(f"  Score: {consistency['score']:.4f} (threshold: {consistency['threshold']}) [{status}]")

    # Compute summary
    all_passed = all(t['passed'] for t in tests)
    passed_count = sum(1 for t in tests if t['passed'])
    elapsed_time = time.time() - start_time

    results = {
        'model_path': str(model_path),
        'device': str(dev),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': elapsed_time,
        'tests': {
            'world_separation': world_sep,
            'rpm_differentiation': rpm_diff,
            'attractor_convergence': attractor,
            'noetic_consistency': consistency,
        },
        'summary': {
            'total_tests': len(tests),
            'passed_tests': passed_count,
            'failed_tests': len(tests) - passed_count,
            'all_passed': all_passed,
        },
        'thresholds': {
            'world_separation': THRESHOLDS.WORLD_SEPARATION_THRESHOLD,
            'rpm_variance': THRESHOLDS.RPM_VARIANCE_THRESHOLD,
            'attractor_convergence': THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD,
            'noetic_consistency': THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD,
        },
        'all_passed': all_passed,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Tests Passed: {passed_count}/{len(tests)}")
        print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
        print(f"Time: {elapsed_time:.2f}s")
        print("=" * 70)

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run noetic coherence validation on a TKS model checkpoint"
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to save JSON results (optional)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON to stdout (quiet mode)'
    )

    args = parser.parse_args()

    # Run validation
    results = run_validation(
        model_path=args.model,
        output_path=args.output,
        verbose=args.verbose and not args.json,
        device=args.device,
    )

    # Output JSON if requested
    if args.json:
        print(json.dumps(results, indent=2, default=str))

    # Exit code
    if 'error' in results:
        print(f"ERROR: {results['error']}", file=sys.stderr)
        sys.exit(2)

    sys.exit(0 if results['all_passed'] else 1)


if __name__ == '__main__':
    main()

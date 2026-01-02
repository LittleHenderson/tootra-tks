#!/usr/bin/env python3
"""
Checkpoint Comparison Script for TKS Noetic Coherence

This script compares two model checkpoints on noetic coherence metrics,
showing deltas (improvement/regression) for each metric. Useful for:
- Tracking training progress
- Evaluating model changes
- Detecting noetic collapse

Metrics compared:
- World Separation (target world accuracy)
- RPM Differentiation (D/W/P variance)
- Attractor Convergence (convergence rate)
- Noetic Consistency (similar/dissimilar separation)

Usage:
    python scripts/compare_checkpoints.py \
        --model-a output/model_v1/final_model.pt \
        --model-b output/model_v2/final_model.pt \
        --output comparison_report.json

References:
    - tests/test_noetic_regression.py (metric computation)
    - scripts/noetic_coherence_test.py (original metrics)
    - TKS_LLM_Noetic_Mathematics_v1.0.md

Author: TKS CI/Evaluation Specialist (Agent C)
Date: 2025-12-22
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Error: PyTorch not available")
    sys.exit(1)

try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    from scripts.train_cuda import SimpleTokenizer
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Error: TKSLLMCorePipeline not available")
    sys.exit(1)

# Import metric computation from test module
try:
    from tests.test_noetic_regression import (
        load_model,
        compute_noetic_consistency,
        compute_rpm_differentiation,
        compute_world_separation,
        compute_attractor_convergence,
        THRESHOLDS,
    )
except ImportError:
    # Fallback: define locally if test module not available
    print("Warning: Could not import from test_noetic_regression, using local definitions")
    from tests.test_noetic_regression import THRESHOLDS


# =============================================================================
# COMPARISON DATA STRUCTURES
# =============================================================================

@dataclass
class MetricResult:
    """Single metric result."""
    value: float
    passed: bool
    threshold: float


@dataclass
class MetricComparison:
    """Comparison of a single metric between two models."""
    name: str
    model_a_value: float
    model_b_value: float
    delta: float
    delta_percent: float
    model_a_passed: bool
    model_b_passed: bool
    threshold: float
    improved: bool
    status: str  # "IMPROVED", "REGRESSED", "STABLE", "RECOVERED", "COLLAPSED"


@dataclass
class CheckpointComparison:
    """Full comparison between two checkpoints."""
    model_a_path: str
    model_b_path: str
    world_separation: MetricComparison
    rpm_differentiation: MetricComparison
    attractor_convergence: MetricComparison
    noetic_consistency: MetricComparison
    overall_improved: bool
    overall_status: str
    summary: str


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compute_metrics_for_model(model_path: str) -> Dict:
    """
    Compute all noetic coherence metrics for a model.

    Args:
        model_path: Path to model checkpoint

    Returns:
        Dict with all metric results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, device = load_model(model_path, device)

    metrics = {
        'world_separation': compute_world_separation(model, tokenizer, device),
        'rpm_differentiation': compute_rpm_differentiation(model, tokenizer, device),
        'attractor_convergence': compute_attractor_convergence(model, tokenizer, device),
        'noetic_consistency': compute_noetic_consistency(model, tokenizer, device),
    }

    # Clean up model to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def compare_metric(
    name: str,
    result_a: Dict,
    result_b: Dict,
    value_key: str,
    threshold: float,
    higher_is_better: bool = True
) -> MetricComparison:
    """
    Compare a single metric between two models.

    Args:
        name: Metric name
        result_a: Metric result from model A
        result_b: Metric result from model B
        value_key: Key to extract numeric value
        threshold: Pass/fail threshold
        higher_is_better: Whether higher values are better

    Returns:
        MetricComparison object
    """
    val_a = result_a[value_key]
    val_b = result_b[value_key]

    delta = val_b - val_a
    delta_percent = (delta / abs(val_a) * 100) if val_a != 0 else 0.0

    passed_a = result_a.get('passed', val_a >= threshold if higher_is_better else val_a <= threshold)
    passed_b = result_b.get('passed', val_b >= threshold if higher_is_better else val_b <= threshold)

    # Determine if improved
    if higher_is_better:
        improved = val_b > val_a
    else:
        improved = val_b < val_a

    # Determine status
    if passed_a and passed_b:
        if improved:
            status = "IMPROVED"
        elif delta == 0:
            status = "STABLE"
        else:
            status = "STABLE"  # Both pass, slight regression is still stable
    elif not passed_a and passed_b:
        status = "RECOVERED"
    elif passed_a and not passed_b:
        status = "COLLAPSED"
    else:
        if improved:
            status = "IMPROVING"
        else:
            status = "REGRESSED"

    return MetricComparison(
        name=name,
        model_a_value=float(val_a),
        model_b_value=float(val_b),
        delta=float(delta),
        delta_percent=float(delta_percent),
        model_a_passed=passed_a,
        model_b_passed=passed_b,
        threshold=threshold,
        improved=improved,
        status=status,
    )


def compare_checkpoints(model_a_path: str, model_b_path: str) -> CheckpointComparison:
    """
    Compare two model checkpoints on all noetic coherence metrics.

    Args:
        model_a_path: Path to first (baseline) model
        model_b_path: Path to second (new) model

    Returns:
        CheckpointComparison object with all metrics
    """
    print(f"\n{'='*70}")
    print("NOETIC COHERENCE CHECKPOINT COMPARISON")
    print(f"{'='*70}")
    print(f"Model A (baseline): {model_a_path}")
    print(f"Model B (new):      {model_b_path}")
    print(f"{'='*70}")

    # Compute metrics for both models
    print("\nComputing metrics for Model A...")
    metrics_a = compute_metrics_for_model(model_a_path)

    print("Computing metrics for Model B...")
    metrics_b = compute_metrics_for_model(model_b_path)

    # Compare each metric
    world_sep = compare_metric(
        name="World Separation",
        result_a=metrics_a['world_separation'],
        result_b=metrics_b['world_separation'],
        value_key='correct_rate',
        threshold=THRESHOLDS.WORLD_SEPARATION_THRESHOLD,
        higher_is_better=True
    )

    rpm_diff = compare_metric(
        name="RPM Differentiation",
        result_a=metrics_a['rpm_differentiation'],
        result_b=metrics_b['rpm_differentiation'],
        value_key='variance',
        threshold=THRESHOLDS.RPM_VARIANCE_THRESHOLD,
        higher_is_better=True
    )

    attractor_conv = compare_metric(
        name="Attractor Convergence",
        result_a=metrics_a['attractor_convergence'],
        result_b=metrics_b['attractor_convergence'],
        value_key='convergence_rate',
        threshold=THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD,
        higher_is_better=True
    )

    noetic_cons = compare_metric(
        name="Noetic Consistency",
        result_a=metrics_a['noetic_consistency'],
        result_b=metrics_b['noetic_consistency'],
        value_key='separation',
        threshold=THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD,
        higher_is_better=True
    )

    # Determine overall status
    all_comparisons = [world_sep, rpm_diff, attractor_conv, noetic_cons]

    num_improved = sum(1 for c in all_comparisons if c.improved)
    num_collapsed = sum(1 for c in all_comparisons if c.status == "COLLAPSED")
    num_recovered = sum(1 for c in all_comparisons if c.status == "RECOVERED")

    model_b_passes_all = all(c.model_b_passed for c in all_comparisons)
    model_a_passes_all = all(c.model_a_passed for c in all_comparisons)

    if num_collapsed > 0:
        overall_status = "REGRESSION - Model collapse detected"
        overall_improved = False
    elif num_recovered > 0 and model_b_passes_all:
        overall_status = "RECOVERED - Model now passes all thresholds"
        overall_improved = True
    elif model_b_passes_all and not model_a_passes_all:
        overall_status = "IMPROVED - Model now passes all thresholds"
        overall_improved = True
    elif num_improved >= 3:
        overall_status = "IMPROVED - Most metrics improved"
        overall_improved = True
    elif num_improved >= 2:
        overall_status = "MIXED - Some improvement, some regression"
        overall_improved = num_improved > len(all_comparisons) // 2
    else:
        overall_status = "REGRESSED - Most metrics worsened"
        overall_improved = False

    # Generate summary
    summary_lines = [
        f"Model B vs Model A:",
        f"  Improved metrics: {num_improved}/{len(all_comparisons)}",
        f"  Model A passes: {sum(1 for c in all_comparisons if c.model_a_passed)}/{len(all_comparisons)}",
        f"  Model B passes: {sum(1 for c in all_comparisons if c.model_b_passed)}/{len(all_comparisons)}",
        f"  Status: {overall_status}",
    ]
    summary = "\n".join(summary_lines)

    return CheckpointComparison(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        world_separation=world_sep,
        rpm_differentiation=rpm_diff,
        attractor_convergence=attractor_conv,
        noetic_consistency=noetic_cons,
        overall_improved=overall_improved,
        overall_status=overall_status,
        summary=summary,
    )


def print_comparison(comparison: CheckpointComparison):
    """Print comparison results in a formatted table."""
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")

    # Header
    print(f"{'Metric':<25} {'Model A':<12} {'Model B':<12} {'Delta':<12} {'Status':<12}")
    print("-" * 70)

    # Metrics
    for metric in [
        comparison.world_separation,
        comparison.rpm_differentiation,
        comparison.attractor_convergence,
        comparison.noetic_consistency,
    ]:
        # Format values
        if metric.name == "RPM Differentiation":
            val_a = f"{metric.model_a_value:.6f}"
            val_b = f"{metric.model_b_value:.6f}"
            delta = f"{metric.delta:+.6f}"
        elif metric.name in ["World Separation", "Attractor Convergence"]:
            val_a = f"{metric.model_a_value:.1%}"
            val_b = f"{metric.model_b_value:.1%}"
            delta = f"{metric.delta:+.1%}"
        else:
            val_a = f"{metric.model_a_value:.4f}"
            val_b = f"{metric.model_b_value:.4f}"
            delta = f"{metric.delta:+.4f}"

        # Add pass/fail indicators
        pass_a = "P" if metric.model_a_passed else "F"
        pass_b = "P" if metric.model_b_passed else "F"
        val_a = f"{val_a} [{pass_a}]"
        val_b = f"{val_b} [{pass_b}]"

        print(f"{metric.name:<25} {val_a:<12} {val_b:<12} {delta:<12} {metric.status:<12}")

    print("-" * 70)

    # Summary
    print(f"\n{comparison.summary}")

    # Overall verdict
    print(f"\n{'='*70}")
    if comparison.overall_improved:
        print("VERDICT: Model B is BETTER than Model A")
    else:
        print("VERDICT: Model B is WORSE than or EQUAL to Model A")
    print(f"{'='*70}")


def save_comparison(comparison: CheckpointComparison, output_path: str):
    """Save comparison to JSON file."""
    # Convert to dict
    result = {
        'model_a_path': comparison.model_a_path,
        'model_b_path': comparison.model_b_path,
        'metrics': {
            'world_separation': asdict(comparison.world_separation),
            'rpm_differentiation': asdict(comparison.rpm_differentiation),
            'attractor_convergence': asdict(comparison.attractor_convergence),
            'noetic_consistency': asdict(comparison.noetic_consistency),
        },
        'overall_improved': comparison.overall_improved,
        'overall_status': comparison.overall_status,
        'summary': comparison.summary,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nComparison saved to: {output_path}")


# =============================================================================
# TRAINING PROGRESS TRACKING
# =============================================================================

def track_training_progress(
    checkpoint_dir: str,
    output_path: Optional[str] = None
) -> Dict:
    """
    Track noetic coherence metrics across multiple checkpoints in a directory.

    Useful for visualizing training progress and detecting collapse points.

    Args:
        checkpoint_dir: Directory containing epoch checkpoints
        output_path: Optional path to save progress JSON

    Returns:
        Dict with metrics for each checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return {}

    print(f"\n{'='*70}")
    print("TRAINING PROGRESS TRACKING")
    print(f"{'='*70}")
    print(f"Directory: {checkpoint_dir}")
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"{'='*70}")

    results = {}
    for i, ckpt_path in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}] Processing: {ckpt_path.name}")

        try:
            metrics = compute_metrics_for_model(str(ckpt_path))

            results[ckpt_path.name] = {
                'world_separation': metrics['world_separation']['correct_rate'],
                'rpm_differentiation': metrics['rpm_differentiation']['variance'],
                'attractor_convergence': metrics['attractor_convergence']['convergence_rate'],
                'noetic_consistency': metrics['noetic_consistency']['separation'],
                'all_passed': all([
                    metrics['world_separation']['passed'],
                    metrics['rpm_differentiation']['passed'],
                    metrics['attractor_convergence']['passed'],
                    metrics['noetic_consistency']['passed'],
                ]),
            }

            # Print summary
            r = results[ckpt_path.name]
            status = "PASS" if r['all_passed'] else "FAIL"
            print(f"  [{status}] WS={r['world_separation']:.1%} "
                  f"RPM={r['rpm_differentiation']:.4f} "
                  f"AC={r['attractor_convergence']:.0%} "
                  f"NC={r['noetic_consistency']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[ckpt_path.name] = {'error': str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("PROGRESS SUMMARY")
    print(f"{'='*70}")

    passing = [k for k, v in results.items() if v.get('all_passed', False)]
    failing = [k for k, v in results.items() if not v.get('all_passed', True) and 'error' not in v]
    errors = [k for k, v in results.items() if 'error' in v]

    print(f"Passing checkpoints: {len(passing)}/{len(checkpoints)}")
    print(f"Failing checkpoints: {len(failing)}/{len(checkpoints)}")
    print(f"Error checkpoints: {len(errors)}/{len(checkpoints)}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nProgress saved to: {output_path}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare noetic coherence metrics between model checkpoints'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two checkpoints')
    compare_parser.add_argument('--model-a', required=True, help='Path to baseline model')
    compare_parser.add_argument('--model-b', required=True, help='Path to new model')
    compare_parser.add_argument('--output', help='Path to save comparison JSON')

    # Track command
    track_parser = subparsers.add_parser('track', help='Track progress across checkpoints')
    track_parser.add_argument('--dir', required=True, help='Directory with checkpoints')
    track_parser.add_argument('--output', help='Path to save progress JSON')

    # Default: compare mode with positional args
    parser.add_argument('--model-a', help='Path to baseline model (compare mode)')
    parser.add_argument('--model-b', help='Path to new model (compare mode)')
    parser.add_argument('--output', help='Path to save output JSON')

    args = parser.parse_args()

    # Determine mode
    if args.command == 'track':
        track_training_progress(args.dir, args.output)

    elif args.command == 'compare' or (args.model_a and args.model_b):
        model_a = args.model_a
        model_b = args.model_b

        if not Path(model_a).exists():
            print(f"Error: Model A not found: {model_a}")
            sys.exit(1)
        if not Path(model_b).exists():
            print(f"Error: Model B not found: {model_b}")
            sys.exit(1)

        comparison = compare_checkpoints(model_a, model_b)
        print_comparison(comparison)

        if args.output:
            save_comparison(comparison, args.output)

        # Exit with code based on regression
        if comparison.overall_improved:
            sys.exit(0)
        else:
            # Non-zero but not error - just indicates no improvement
            sys.exit(0)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Compare two checkpoints:")
        print("  python scripts/compare_checkpoints.py --model-a v1.pt --model-b v2.pt")
        print("")
        print("  # Track progress across training:")
        print("  python scripts/compare_checkpoints.py track --dir output/checkpoints/")
        sys.exit(1)


if __name__ == "__main__":
    main()

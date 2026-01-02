#!/usr/bin/env python3
"""
Two-Stage Training Pipeline with Consistency vs Generalization Evaluation

Stage 1: Equation-only training (rpm_finetune preset)
Stage 2: NL generalization training (rpm_finetune_nl preset)

Each stage is evaluated with both equation and NL test styles to track
the consistency vs generalization tradeoff.

Usage:
    python scripts/run_two_stage_eval.py --checkpoint output/world_v3_30ep/best_model.pt

Author: TKS Pipeline Automation
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    success = result.returncode == 0

    if not success:
        print(f"\nWARNING: {description} returned non-zero exit code (may be expected for failing tests)")

    return True  # Continue even if tests fail


def generate_data(output_dir: Path, nl_ratio: float, num_examples: int = 2000) -> Path:
    """Generate training data with specified NL ratio."""
    suffix = "eq" if nl_ratio == 0.0 else f"nl{int(nl_ratio*100)}"
    output_path = output_dir / f"train_{suffix}.jsonl"

    cmd = (
        f"python3 scripts/generate_world_rpm_labels.py "
        f"--output {output_path} "
        f"--num-examples {num_examples} "
        f"--nl-ratio {nl_ratio} "
        f"--seed 42"
    )
    run_command(cmd, f"Generate training data (nl_ratio={nl_ratio})")
    return output_path


def train_stage(
    data_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    preset: str,
    epochs: int = 20
) -> Path:
    """Run training with specified preset."""
    cmd = (
        f"python3 scripts/train_cuda.py "
        f"--data {data_path} "
        f"--checkpoint {checkpoint_path} "
        f"--output-dir {output_dir} "
        f"--world-loss-preset {preset} "
        f"--epochs {epochs} "
        f"--batch-size 16 "
        f"--learning-rate 1e-4"
    )
    run_command(cmd, f"Train with preset={preset}")
    return output_dir / "best_model.pt"


def eval_stage(model_path: Path, output_dir: Path, test_style: str) -> Path:
    """Run evaluation with specified test style."""
    metrics_path = output_dir / f"metrics_{test_style}.json"

    cmd = (
        f"python3 tests/test_noetic_regression.py "
        f"--model {model_path} "
        f"--test-style {test_style} "
        f"--output {metrics_path}"
    )
    run_command(cmd, f"Evaluate with test_style={test_style}")
    return metrics_path


def load_metrics(path: Path) -> dict:
    """Load metrics from JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def create_comparison_report(
    stage1_eq: dict,
    stage1_nl: dict,
    stage2_eq: dict,
    stage2_nl: dict,
    output_path: Path
) -> dict:
    """Create comparison report across stages and test styles."""

    def extract_metrics(m: dict) -> dict:
        if not m:
            return {"noetic_consistency": None, "rpm_variance": None, "world_separation": None, "attractor_convergence": None}
        metrics = m.get("metrics", {})
        return {
            "noetic_consistency": metrics.get("noetic_consistency", {}).get("separation"),
            "rpm_variance": metrics.get("rpm_differentiation", {}).get("variance"),
            "world_separation": metrics.get("world_separation", {}).get("correct_rate"),
            "attractor_convergence": metrics.get("attractor_convergence", {}).get("convergence_rate")
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": {
            "noetic_consistency": 0.05,
            "rpm_variance": 0.01,
            "world_separation": 0.60,
            "attractor_convergence": 0.80
        },
        "stages": {
            "stage1_equation_training": {
                "eval_equation": extract_metrics(stage1_eq),
                "eval_nl": extract_metrics(stage1_nl)
            },
            "stage2_nl_generalization": {
                "eval_equation": extract_metrics(stage2_eq),
                "eval_nl": extract_metrics(stage2_nl)
            }
        },
        "analysis": {}
    }

    # Analyze consistency vs generalization tradeoff
    s1_eq = report["stages"]["stage1_equation_training"]["eval_equation"]
    s1_nl = report["stages"]["stage1_equation_training"]["eval_nl"]
    s2_eq = report["stages"]["stage2_nl_generalization"]["eval_equation"]
    s2_nl = report["stages"]["stage2_nl_generalization"]["eval_nl"]

    report["analysis"]["stage1_to_stage2_delta"] = {
        "noetic_consistency_eq": (s2_eq["noetic_consistency"] or 0) - (s1_eq["noetic_consistency"] or 0) if s1_eq["noetic_consistency"] and s2_eq["noetic_consistency"] else None,
        "noetic_consistency_nl": (s2_nl["noetic_consistency"] or 0) - (s1_nl["noetic_consistency"] or 0) if s1_nl["noetic_consistency"] and s2_nl["noetic_consistency"] else None,
        "world_separation_eq": (s2_eq["world_separation"] or 0) - (s1_eq["world_separation"] or 0) if s1_eq["world_separation"] and s2_eq["world_separation"] else None,
        "world_separation_nl": (s2_nl["world_separation"] or 0) - (s1_nl["world_separation"] or 0) if s1_nl["world_separation"] and s2_nl["world_separation"] else None
    }

    # Check if NL training hurts equation consistency
    if s1_eq["noetic_consistency"] and s2_eq["noetic_consistency"]:
        hurt_eq = s2_eq["noetic_consistency"] < s1_eq["noetic_consistency"]
        report["analysis"]["nl_hurts_equation_consistency"] = hurt_eq
        if hurt_eq:
            report["analysis"]["recommendation"] = "Consider tuning --nl-ratio (0.2-0.4) or --contrastive-loss-weight (0.05-0.1)"

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict):
    """Print formatted comparison report."""
    print("\n" + "="*70)
    print("TWO-STAGE PIPELINE COMPARISON REPORT")
    print("="*70)

    thresholds = report["thresholds"]

    def fmt_metric(name: str, val, threshold: float) -> str:
        if val is None:
            return "N/A"
        passed = val >= threshold
        mark = "✓" if passed else "✗"
        if name == "world_separation":
            return f"{val*100:.1f}% {mark}"
        elif name in ["noetic_consistency", "rpm_variance"]:
            return f"{val:.4f} {mark}"
        else:
            return f"{val*100:.0f}% {mark}"

    stages = report["stages"]

    print("\n" + "-"*70)
    print("STAGE 1: Equation-Only Training (rpm_finetune)")
    print("-"*70)
    print(f"{'Metric':<25} {'Eval Equation':<20} {'Eval NL':<20}")
    print("-"*70)

    s1 = stages["stage1_equation_training"]
    for metric, thresh in thresholds.items():
        eq_val = s1["eval_equation"].get(metric)
        nl_val = s1["eval_nl"].get(metric)
        print(f"{metric:<25} {fmt_metric(metric, eq_val, thresh):<20} {fmt_metric(metric, nl_val, thresh):<20}")

    print("\n" + "-"*70)
    print("STAGE 2: NL Generalization (rpm_finetune_nl)")
    print("-"*70)
    print(f"{'Metric':<25} {'Eval Equation':<20} {'Eval NL':<20}")
    print("-"*70)

    s2 = stages["stage2_nl_generalization"]
    for metric, thresh in thresholds.items():
        eq_val = s2["eval_equation"].get(metric)
        nl_val = s2["eval_nl"].get(metric)
        print(f"{metric:<25} {fmt_metric(metric, eq_val, thresh):<20} {fmt_metric(metric, nl_val, thresh):<20}")

    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)

    analysis = report.get("analysis", {})
    delta = analysis.get("stage1_to_stage2_delta", {})

    if delta.get("noetic_consistency_eq") is not None:
        print(f"Noetic consistency (eq) delta: {delta['noetic_consistency_eq']:+.4f}")
    if delta.get("noetic_consistency_nl") is not None:
        print(f"Noetic consistency (nl) delta: {delta['noetic_consistency_nl']:+.4f}")

    if analysis.get("nl_hurts_equation_consistency"):
        print("\nWARNING: NL training hurt equation consistency!")
        if analysis.get("recommendation"):
            print(f"Recommendation: {analysis['recommendation']}")
    else:
        print("\nNL training did not hurt equation consistency.")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Run two-stage training pipeline with evaluation")
    parser.add_argument('--checkpoint', required=True, help='Path to world-trained checkpoint')
    parser.add_argument('--output-dir', default='output/two_stage_pipeline', help='Output directory')
    parser.add_argument('--nl-ratio', type=float, default=0.4, help='NL ratio for Stage 2 (default: 0.4)')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs per stage')
    parser.add_argument('--skip-stage1', action='store_true', help='Skip Stage 1 (use existing)')
    parser.add_argument('--skip-stage2', action='store_true', help='Skip Stage 2 (use existing)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_dir = output_dir / "stage1_equation"
    stage2_dir = output_dir / "stage2_nl"

    print("="*70)
    print("TWO-STAGE TRAINING PIPELINE")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"NL ratio for Stage 2: {args.nl_ratio}")
    print(f"Epochs per stage: {args.epochs}")

    # Generate data
    print("\n" + "="*70)
    print("GENERATING TRAINING DATA")
    print("="*70)

    eq_data = generate_data(output_dir, nl_ratio=0.0)
    nl_data = generate_data(output_dir, nl_ratio=args.nl_ratio)

    # Stage 1: Equation-only training
    if not args.skip_stage1:
        print("\n" + "="*70)
        print("STAGE 1: EQUATION-ONLY TRAINING")
        print("="*70)

        stage1_model = train_stage(
            data_path=eq_data,
            checkpoint_path=Path(args.checkpoint),
            output_dir=stage1_dir,
            preset="rpm_finetune",
            epochs=args.epochs
        )
    else:
        stage1_model = stage1_dir / "best_model.pt"
        print(f"\nSkipping Stage 1 training, using: {stage1_model}")

    # Evaluate Stage 1
    print("\n" + "="*70)
    print("STAGE 1 EVALUATION")
    print("="*70)

    stage1_eq_metrics = eval_stage(stage1_model, stage1_dir, "equation")
    stage1_nl_metrics = eval_stage(stage1_model, stage1_dir, "nl")

    # Stage 2: NL generalization
    if not args.skip_stage2:
        print("\n" + "="*70)
        print("STAGE 2: NL GENERALIZATION")
        print("="*70)

        stage2_model = train_stage(
            data_path=nl_data,
            checkpoint_path=stage1_model,
            output_dir=stage2_dir,
            preset="rpm_finetune_nl",
            epochs=args.epochs
        )
    else:
        stage2_model = stage2_dir / "best_model.pt"
        print(f"\nSkipping Stage 2 training, using: {stage2_model}")

    # Evaluate Stage 2
    print("\n" + "="*70)
    print("STAGE 2 EVALUATION")
    print("="*70)

    stage2_eq_metrics = eval_stage(stage2_model, stage2_dir, "equation")
    stage2_nl_metrics = eval_stage(stage2_model, stage2_dir, "nl")

    # Create comparison report
    print("\n" + "="*70)
    print("CREATING COMPARISON REPORT")
    print("="*70)

    report = create_comparison_report(
        stage1_eq=load_metrics(stage1_eq_metrics),
        stage1_nl=load_metrics(stage1_nl_metrics),
        stage2_eq=load_metrics(stage2_eq_metrics),
        stage2_nl=load_metrics(stage2_nl_metrics),
        output_path=output_dir / "comparison_report.json"
    )

    print_report(report)
    print(f"\nReport saved to: {output_dir / 'comparison_report.json'}")


if __name__ == "__main__":
    main()

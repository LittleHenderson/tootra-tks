#!/usr/bin/env python3
"""
Evaluate curriculum stages and ablations with noetic regression tests.

Runs tests/test_noetic_regression.py metrics for each model checkpoint and
writes a single comparison JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.complexfloating,)):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        return super().default(obj)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.test_noetic_regression import (
    THRESHOLDS,
    load_model,
    compute_all_metrics,
    EQUATION_WORLD_TEXTS,
    NL_WORLD_TEXTS,
    EQUATION_SIMILAR_PAIRS,
    NL_SIMILAR_PAIRS,
    EQUATION_DISSIMILAR_PAIRS,
    NL_DISSIMILAR_PAIRS,
    EQUATION_RPM_TEST_CASES,
    NL_RPM_TEST_CASES,
    EQUATION_ATTRACTOR_TEXTS,
    NL_ATTRACTOR_TEXTS,
)


DEFAULT_STAGES = [
    "stage1_subfoundation",
    "stage2_acquisition",
    "stage3_diffusion",
]

DEFAULT_ABLATIONS = [
    "none",
]

ABLATION_PRESETS = {
    "none": None,
    "no_subfoundation": {"subfoundation": 0.0},
    "no_acquisition": {"acquisition": 0.0},
    "no_foundation": {"foundation": 0.0},
    "no_expressiveness": {"subfoundation": 0.0, "acquisition": 0.0, "foundation": 0.0},
}


def select_test_data(test_style: str) -> Dict:
    if test_style == "equation":
        return {
            "world_texts": EQUATION_WORLD_TEXTS,
            "similar_pairs": EQUATION_SIMILAR_PAIRS,
            "dissimilar_pairs": EQUATION_DISSIMILAR_PAIRS,
            "rpm_test_cases": EQUATION_RPM_TEST_CASES,
            "attractor_texts": EQUATION_ATTRACTOR_TEXTS,
        }
    if test_style == "nl":
        return {
            "world_texts": NL_WORLD_TEXTS,
            "similar_pairs": NL_SIMILAR_PAIRS,
            "dissimilar_pairs": NL_DISSIMILAR_PAIRS,
            "rpm_test_cases": NL_RPM_TEST_CASES,
            "attractor_texts": NL_ATTRACTOR_TEXTS,
        }

    # "both" - combine equation and NL
    combined_worlds = {}
    for world in "ABCD":
        combined_worlds[world] = EQUATION_WORLD_TEXTS[world] + NL_WORLD_TEXTS[world]

    combined_rpm = {
        cat: EQUATION_RPM_TEST_CASES[cat] + NL_RPM_TEST_CASES[cat]
        for cat in ["desire", "wisdom", "power"]
    }

    return {
        "world_texts": combined_worlds,
        "similar_pairs": EQUATION_SIMILAR_PAIRS + NL_SIMILAR_PAIRS,
        "dissimilar_pairs": EQUATION_DISSIMILAR_PAIRS + NL_DISSIMILAR_PAIRS,
        "rpm_test_cases": combined_rpm,
        "attractor_texts": EQUATION_ATTRACTOR_TEXTS + NL_ATTRACTOR_TEXTS,
    }


def _iter_blocks(model):
    model_obj = model.module if hasattr(model, "module") else model
    return getattr(model_obj, "blocks", [])


def apply_gate_overrides(model, overrides: Optional[Dict[str, float]]) -> None:
    if not overrides:
        return

    for block in _iter_blocks(model):
        expr = getattr(block, "expressiveness", None)
        if expr is None:
            continue

        if overrides.get("subfoundation") is not None:
            mixer = getattr(expr, "subfoundation_mixer", None)
            if mixer is not None:
                mixer.gate.data.fill_(overrides["subfoundation"])

        if overrides.get("acquisition") is not None:
            acq = getattr(expr, "acquisition_program", None)
            if acq is not None:
                acq.gate.data.fill_(overrides["acquisition"])

        if overrides.get("foundation") is not None:
            fd = getattr(expr, "foundation_diffusion", None)
            if fd is not None:
                fd.gate.data.fill_(overrides["foundation"])


def collect_gate_values(model) -> Dict[str, Dict[str, float]]:
    values = {
        "subfoundation": [],
        "acquisition": [],
        "foundation": [],
    }

    for block in _iter_blocks(model):
        expr = getattr(block, "expressiveness", None)
        if expr is None:
            continue

        mixer = getattr(expr, "subfoundation_mixer", None)
        if mixer is not None:
            gate = getattr(mixer, "gate", None)
            if gate is not None:
                val = gate.mean().item() if gate.numel() > 1 else gate.item()
                values["subfoundation"].append(float(val))

        acq = getattr(expr, "acquisition_program", None)
        if acq is not None:
            gate = getattr(acq, "gate", None)
            if gate is not None:
                val = gate.mean().item() if gate.numel() > 1 else gate.item()
                values["acquisition"].append(float(val))

        fd = getattr(expr, "foundation_diffusion", None)
        if fd is not None:
            gate = getattr(fd, "gate", None)
            if gate is not None:
                val = gate.mean().item() if gate.numel() > 1 else gate.item()
                values["foundation"].append(float(val))

    summary = {}
    for key, vals in values.items():
        if vals:
            summary[key] = {
                "mean": float(sum(vals) / len(vals)),
                "values": vals,
            }
        else:
            summary[key] = {
                "mean": None,
                "values": [],
            }

    return summary


def evaluate_model(
    model_path: Path,
    model_type: str,
    test_style: str,
    device: Optional[torch.device],
    gate_overrides: Optional[Dict[str, float]] = None,
) -> Dict:
    model, tokenizer, device, actual_type = load_model(
        str(model_path), device=device, model_type=model_type
    )

    apply_gate_overrides(model, gate_overrides)

    data = select_test_data(test_style)
    metrics = compute_all_metrics(
        model,
        tokenizer,
        device,
        world_texts=data["world_texts"],
        similar_pairs=data["similar_pairs"],
        dissimilar_pairs=data["dissimilar_pairs"],
        rpm_test_cases=data["rpm_test_cases"],
        attractor_texts=data["attractor_texts"],
    )

    all_passed = all(result["passed"] for result in metrics.values())

    return {
        "model_path": str(model_path),
        "model_type": actual_type,
        "test_style": test_style,
        "gate_overrides": gate_overrides,
        "gate_values": collect_gate_values(model),
        "metrics": metrics,
        "all_passed": all_passed,
    }


def resolve_stage_models(curriculum_dir: Path, stages: List[str], checkpoint_name: str) -> Dict[str, Path]:
    models = {}
    for stage in stages:
        stage_dir = curriculum_dir / stage
        if not stage_dir.exists():
            continue
        ckpt = stage_dir / checkpoint_name
        if not ckpt.exists():
            alt = stage_dir / "final_model.pt"
            ckpt = alt if alt.exists() else None
        if ckpt is not None and ckpt.exists():
            models[stage] = ckpt
    return models


def parse_extra_models(values: List[str]) -> Dict[str, Path]:
    models = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --extra-model '{item}'. Use name=path.")
        name, path = item.split("=", 1)
        models[name.strip()] = Path(path.strip())
    return models


def normalize_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def extract_metric_value(metrics: Dict, metric_name: str) -> Optional[float]:
    """Extract a single numeric value from a metric result."""
    if metric_name not in metrics:
        return None
    m = metrics[metric_name]

    # Metric-specific extraction
    if metric_name == "world_separation":
        if "correct_rate" in m:
            return float(m["correct_rate"])
    elif metric_name == "rpm_differentiation":
        if "variance" in m:
            return float(m["variance"])
    elif metric_name == "attractor_convergence":
        if "convergence_rate" in m:
            return float(m["convergence_rate"])
    elif metric_name == "noetic_consistency":
        if "separation" in m:
            return float(m["separation"])

    # Fallback to common value keys
    for key in ["value", "mean", "score", "accuracy", "separation", "correct_rate", "convergence_rate", "variance"]:
        if key in m:
            return float(m[key])
    return None


def print_summary_table(results: Dict) -> None:
    """Print a short summary table: best per stage, deltas vs baseline."""
    runs = results.get("runs", [])
    if not runs:
        print("\nNo runs to summarize.")
        return

    # Filter out error runs (those without metrics)
    valid_runs = [r for r in runs if "metrics" in r and "error" not in r]
    if not valid_runs:
        print("\nNo valid runs to summarize (all had errors).")
        return

    # Group runs by model_name and test_style (only ablation="none")
    baseline_runs = [r for r in valid_runs if r.get("ablation") == "none"]

    if not baseline_runs:
        print("\nNo baseline (ablation=none) runs found.")
        return

    # Key metrics to track (matching actual test_noetic_regression.py output)
    metric_keys = ["world_separation", "rpm_differentiation", "attractor_convergence", "noetic_consistency"]

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Header
    header = f"{'Model':<25} {'Style':<10} {'Passed':<8}"
    for mk in metric_keys:
        short_name = mk.replace("_", " ").title()[:12]
        header += f" {short_name:<12}"
    print(header)
    print("-" * 80)

    # Collect baseline values for delta calculation
    first_baseline = None
    first_baseline_metrics = {}

    for run in baseline_runs:
        model_name = run.get("model_name", "unknown")
        test_style = run.get("test_style", "unknown")
        all_passed = run.get("all_passed", False)
        metrics = run.get("metrics", {})

        passed_str = "PASS" if all_passed else "FAIL"

        row = f"{model_name:<25} {test_style:<10} {passed_str:<8}"

        for mk in metric_keys:
            val = extract_metric_value(metrics, mk)
            if val is not None:
                row += f" {val:<12.4f}"
            else:
                row += f" {'N/A':<12}"

        print(row)

        # Store first baseline for delta comparison
        if first_baseline is None:
            first_baseline = model_name
            for mk in metric_keys:
                first_baseline_metrics[mk] = extract_metric_value(metrics, mk)

    # Print deltas vs first baseline (if we have multiple stages)
    if len(baseline_runs) > 1 and first_baseline:
        print("-" * 80)
        print(f"Deltas vs {first_baseline}:")

        for run in baseline_runs[1:]:
            model_name = run.get("model_name", "unknown")
            test_style = run.get("test_style", "unknown")
            metrics = run.get("metrics", {})

            row = f"{model_name:<25} {test_style:<10} {'delta':<8}"

            for mk in metric_keys:
                val = extract_metric_value(metrics, mk)
                base_val = first_baseline_metrics.get(mk)
                if val is not None and base_val is not None:
                    delta = val - base_val
                    sign = "+" if delta >= 0 else ""
                    row += f" {sign}{delta:<11.4f}"
                else:
                    row += f" {'N/A':<12}"

            print(row)

    # Ablation comparison (if any)
    ablation_runs = [r for r in valid_runs if r.get("ablation") != "none"]
    if ablation_runs:
        print("-" * 80)
        print("Ablation Effects:")

        for run in ablation_runs:
            model_name = run.get("model_name", "unknown")
            ablation = run.get("ablation", "unknown")
            test_style = run.get("test_style", "unknown")
            all_passed = run.get("all_passed", False)

            # Find corresponding baseline (with None guard)
            baseline = next(
                (r for r in baseline_runs if r.get("model_name") == model_name and r.get("test_style") == test_style),
                None
            )

            passed_str = "PASS" if all_passed else "FAIL"
            row = f"{model_name[:15]}:{ablation:<15} {test_style:<10} {passed_str:<8}"

            for mk in metric_keys:
                val = extract_metric_value(run.get("metrics", {}), mk)
                base_val = None
                if baseline is not None:
                    base_val = extract_metric_value(baseline.get("metrics", {}), mk)

                if val is not None and base_val is not None:
                    delta = val - base_val
                    sign = "+" if delta >= 0 else ""
                    row += f" {sign}{delta:<11.4f}"
                elif val is not None:
                    row += f" {val:<12.4f}"
                else:
                    row += f" {'N/A':<12}"

            print(row)

    print("=" * 80)

    # Best model summary (from baseline runs, or all valid runs if no baselines)
    candidate_runs = baseline_runs if baseline_runs else valid_runs
    if candidate_runs:
        best_run = max(candidate_runs, key=lambda r: sum(
            1 for m in r.get("metrics", {}).values() if m.get("passed", False)
        ))
        print(f"\nBest model: {best_run.get('model_name')} ({best_run.get('test_style')})")
        print(f"  All passed: {best_run.get('all_passed')}")
        print(f"  Metrics passed: {sum(1 for m in best_run.get('metrics', {}).values() if m.get('passed', False))}"
              f"/{len(best_run.get('metrics', {}))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TKS Curriculum Evaluation Harness")
    parser.add_argument("--curriculum-dir", default="output/curriculum_train",
                        help="Curriculum output directory")
    parser.add_argument("--checkpoint-name", default="best_model.pt",
                        help="Checkpoint file name")
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                        help="Comma-separated stage directories")
    parser.add_argument("--model-type", default="v4", choices=["v2", "v4"],
                        help="Model type")
    parser.add_argument("--test-styles", default="equation,nl",
                        help="Comma-separated test styles: equation,nl,both")
    parser.add_argument("--ablations", default=",".join(DEFAULT_ABLATIONS),
                        help="Comma-separated ablations")
    parser.add_argument("--extra-model", action="append", default=[],
                        help="Extra model name=path (repeatable)")
    parser.add_argument("--output", default="output/curriculum_eval.json",
                        help="Output JSON path")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip printing summary table")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    stages = normalize_list(args.stages)
    test_styles = normalize_list(args.test_styles)
    ablations = normalize_list(args.ablations)

    if not test_styles:
        test_styles = ["equation", "nl"]

    for style in test_styles:
        if style not in {"equation", "nl", "both"}:
            raise ValueError(f"Invalid test style: {style}")

    models = {}
    curriculum_dir = Path(args.curriculum_dir)
    if curriculum_dir.exists():
        models.update(resolve_stage_models(curriculum_dir, stages, args.checkpoint_name))

    if args.extra_model:
        models.update(parse_extra_models(args.extra_model))

    if not models:
        raise SystemExit("No models found to evaluate.")

    print(f"Found {len(models)} model(s) to evaluate:")
    for name, path in models.items():
        print(f"  {name}: {path}")

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": args.model_type,
        "curriculum_dir": str(curriculum_dir),
        "stages": stages,
        "test_styles": test_styles,
        "ablations": ablations,
        "thresholds": {
            "world_separation": THRESHOLDS.WORLD_SEPARATION_THRESHOLD,
            "rpm_variance": THRESHOLDS.RPM_VARIANCE_THRESHOLD,
            "attractor_convergence": THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD,
            "noetic_consistency": THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD,
        },
        "runs": [],
    }

    total_runs = len(models) * len(test_styles) * len(ablations)
    run_idx = 0

    for model_name, model_path in models.items():
        if not model_path.exists():
            print(f"  Skipping {model_name}: path not found")
            continue

        for test_style in test_styles:
            for ablation in ablations:
                run_idx += 1
                overrides = ABLATION_PRESETS.get(ablation)
                if ablation not in ABLATION_PRESETS:
                    raise ValueError(f"Unknown ablation preset: {ablation}")

                print(f"\n[{run_idx}/{total_runs}] Evaluating {model_name} | {test_style} | ablation={ablation}")

                try:
                    run = evaluate_model(
                        model_path=model_path,
                        model_type=args.model_type,
                        test_style=test_style,
                        device=device,
                        gate_overrides=overrides,
                    )
                    run["model_name"] = model_name
                    run["ablation"] = ablation
                    results["runs"].append(run)

                    passed = run.get("all_passed", False)
                    n_passed = sum(1 for m in run.get("metrics", {}).values() if m.get("passed", False))
                    n_total = len(run.get("metrics", {}))
                    status = "PASS" if passed else "FAIL"
                    print(f"  -> {status} ({n_passed}/{n_total} metrics passed)")

                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    results["runs"].append({
                        "model_name": model_name,
                        "model_path": str(model_path),
                        "test_style": test_style,
                        "ablation": ablation,
                        "error": str(e),
                    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nWrote results to {output_path}")

    # Print summary table
    if not args.no_summary:
        print_summary_table(results)


if __name__ == "__main__":
    main()

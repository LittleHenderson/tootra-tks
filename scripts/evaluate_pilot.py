#!/usr/bin/env python3
"""
TKS-LLM Pilot Evaluation Script

Evaluates a trained pilot model on test data and generates reports.

Usage:
    python scripts/evaluate_pilot.py --model outputs/pilot/pilot_model_final.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tks_llm_core_v2 import TKSModel, TKSConfig
    from training import (
        TKSLoss,
        TKSLossConfig,
        PilotDataset,
        tks_collate_fn,
        get_device,
        INVOLUTION_PAIRS,
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


# ==============================================================================
# CANONICAL REFERENCES
# ==============================================================================

NOETICS = {
    1: "Mind", 2: "Positive", 3: "Negative", 4: "Vibration", 5: "Female",
    6: "Male", 7: "Rhythm", 8: "Cause", 9: "Effect", 10: "Idea"
}

WORLDS = {'A': "Spiritual", 'B': "Mental", 'C': "Emotional", 'D': "Physical"}


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def compute_element_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute element prediction accuracy."""
    pred_indices = predictions.argmax(dim=-1)
    correct = (pred_indices == targets).float().mean()
    return correct.item()


def compute_rpm_mse(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute MSE for RPM components."""
    mse = nn.MSELoss(reduction='none')
    errors = mse(predictions, targets).mean(dim=0)

    return {
        "desire_mse": errors[0].item() if errors.numel() > 0 else 0.0,
        "wisdom_mse": errors[1].item() if errors.numel() > 1 else 0.0,
        "power_mse": errors[2].item() if errors.numel() > 2 else 0.0,
        "total_mse": errors.mean().item()
    }


def compute_involution_detection_rate(
    noetic_activations: torch.Tensor,
    batch_data: Dict
) -> Dict[str, float]:
    """
    Compute how well the model detects involution pairs.

    Involution pairs (2,3), (5,6), (8,9) should jointly activate N10.
    """
    # This is a simplified check - in practice would need more sophisticated analysis
    results = {
        "involution_samples": 0,
        "correctly_detected": 0
    }

    # For each sample, check if involution pairs lead to N10 activation
    for i, sample in enumerate(batch_data.get("is_involution", [])):
        if sample:
            results["involution_samples"] += 1
            # Check if N10 (index 9, 19, 29, or 39 depending on world) has high activation
            # Simplified: check if any N10 activation > 0.5
            activations = noetic_activations[i]
            n10_indices = [9, 19, 29, 39]  # N10 in each world
            max_n10 = max(activations[idx].item() for idx in n10_indices if idx < len(activations))
            if max_n10 > 0.5:
                results["correctly_detected"] += 1

    if results["involution_samples"] > 0:
        results["detection_rate"] = results["correctly_detected"] / results["involution_samples"]
    else:
        results["detection_rate"] = 0.0

    return results


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: TKSLoss,
    device: torch.device
) -> Dict:
    """Run full evaluation on a dataset."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    element_correct = 0
    element_total = 0

    all_rpm_preds = []
    all_rpm_targets = []
    involution_stats = {"involution_samples": 0, "correctly_detected": 0}

    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                input_ids = input_ids.to(device)

            # Forward pass
            outputs = model(input_ids)

            # Compute loss
            loss_output = loss_fn(outputs, batch)
            total_loss += loss_output["total_loss"].item() * len(batch.get("input_ids", []))
            total_samples += len(batch.get("input_ids", [1]))

            # Element accuracy
            if "element_logits" in outputs and "target_index" in batch:
                preds = outputs["element_logits"]
                targets = batch["target_index"].to(device)
                element_correct += (preds.argmax(dim=-1) == targets).sum().item()
                element_total += targets.numel()

            # RPM metrics
            if "rpm_logits" in outputs and "target_rpm" in batch:
                all_rpm_preds.append(outputs["rpm_logits"].cpu())
                all_rpm_targets.append(batch["target_rpm"])

            # Involution detection
            if "noetic_activations" in outputs:
                inv_result = compute_involution_detection_rate(
                    outputs["noetic_activations"].cpu(),
                    batch
                )
                involution_stats["involution_samples"] += inv_result["involution_samples"]
                involution_stats["correctly_detected"] += inv_result["correctly_detected"]

    # Aggregate metrics
    avg_loss = total_loss / max(total_samples, 1)
    element_acc = element_correct / max(element_total, 1)

    rpm_metrics = {"total_mse": 0.0}
    if all_rpm_preds:
        all_rpm_preds = torch.cat(all_rpm_preds, dim=0)
        all_rpm_targets = torch.cat(all_rpm_targets, dim=0)
        rpm_metrics = compute_rpm_mse(all_rpm_preds, all_rpm_targets)

    inv_rate = 0.0
    if involution_stats["involution_samples"] > 0:
        inv_rate = involution_stats["correctly_detected"] / involution_stats["involution_samples"]

    return {
        "average_loss": avg_loss,
        "element_accuracy": element_acc,
        "rpm_mse": rpm_metrics,
        "involution_detection_rate": inv_rate,
        "total_samples": total_samples,
        "element_samples": element_total,
        "involution_samples": involution_stats["involution_samples"]
    }


# ==============================================================================
# CANONICAL VALIDATION TESTS
# ==============================================================================

def test_involution_pairs(model: nn.Module, device: torch.device) -> Dict:
    """Test that involution pairs compose correctly."""
    results = {}

    model.eval()
    with torch.no_grad():
        for pair in INVOLUTION_PAIRS:
            n1, n2 = pair
            pair_name = f"N{n1}_N{n2}"

            # Create test input for this pair (simplified - would need proper tokenization)
            # This is a placeholder for the actual involution test
            results[pair_name] = {
                "pair": pair,
                "noetic_1": NOETICS[n1],
                "noetic_2": NOETICS[n2],
                "expected_result": "N10 (Idea)",
                "status": "test_placeholder"
            }

    return results


def test_rpm_consistency(model: nn.Module, device: torch.device) -> Dict:
    """Test RPM computation consistency."""
    results = {
        "desire_noetics_test": {
            "noetics": [2, 3],
            "expected": "High desire component",
            "status": "test_placeholder"
        },
        "wisdom_noetics_test": {
            "noetics": [1, 4, 5, 6, 7],
            "expected": "High wisdom component",
            "status": "test_placeholder"
        },
        "power_noetics_test": {
            "noetics": [8, 9],
            "expected": "High power component",
            "status": "test_placeholder"
        }
    }
    return results


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_report(
    eval_results: Dict,
    canonical_tests: Dict,
    output_path: Path
):
    """Generate evaluation report."""
    report_lines = [
        "=" * 70,
        "TKS-LLM PILOT EVALUATION REPORT",
        "=" * 70,
        "",
        "PERFORMANCE METRICS",
        "-" * 40,
        f"Average Loss: {eval_results['average_loss']:.4f}",
        f"Element Accuracy: {eval_results['element_accuracy']:.2%}",
        f"RPM Total MSE: {eval_results['rpm_mse']['total_mse']:.4f}",
        f"Involution Detection Rate: {eval_results['involution_detection_rate']:.2%}",
        "",
        "SAMPLE COUNTS",
        "-" * 40,
        f"Total samples evaluated: {eval_results['total_samples']}",
        f"Element prediction samples: {eval_results['element_samples']}",
        f"Involution samples: {eval_results['involution_samples']}",
        "",
        "RPM COMPONENT ERRORS",
        "-" * 40,
        f"Desire MSE: {eval_results['rpm_mse'].get('desire_mse', 0):.4f}",
        f"Wisdom MSE: {eval_results['rpm_mse'].get('wisdom_mse', 0):.4f}",
        f"Power MSE: {eval_results['rpm_mse'].get('power_mse', 0):.4f}",
        "",
        "CANONICAL VALIDATION",
        "-" * 40,
    ]

    # Involution tests
    report_lines.append("\nInvolution Pair Tests:")
    for pair_name, result in canonical_tests.get("involution_tests", {}).items():
        report_lines.append(
            f"  {pair_name}: {result['noetic_1']} + {result['noetic_2']} -> {result['expected_result']}"
        )
        report_lines.append(f"    Status: {result['status']}")

    # RPM tests
    report_lines.append("\nRPM Consistency Tests:")
    for test_name, result in canonical_tests.get("rpm_tests", {}).items():
        report_lines.append(f"  {test_name}: {result['status']}")

    report_lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70
    ])

    # Write report
    report_text = "\n".join(report_lines)
    report_file = output_path / "evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_file}")

    # Also save as JSON
    json_file = output_path / "evaluation_results.json"
    with open(json_file, 'w') as f:
        json.dump({
            "metrics": eval_results,
            "canonical_tests": canonical_tests
        }, f, indent=2)
    print(f"Results saved to: {json_file}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate TKS-LLM pilot model")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="data/pilot",
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="outputs/pilot/eval",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"]
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load model
    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)

    config = checkpoint.get("config", {})

    # Recreate model
    try:
        model_config = TKSConfig(**config)
        model = TKSModel(model_config)
    except Exception:
        from scripts.run_pilot_training import SimplePilotModel
        model = SimplePilotModel(config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")

    # Load evaluation data
    data_dir = Path(args.data_dir)
    eval_file = data_dir / "combined_all.jsonl"

    if not eval_file.exists():
        print(f"Evaluation data not found: {eval_file}")
        sys.exit(1)

    dataset = PilotDataset(str(eval_file))
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=tks_collate_fn
    )

    print(f"Loaded {len(dataset)} evaluation samples")

    # Create loss function (full loss for evaluation)
    loss_fn = TKSLoss(TKSLossConfig(
        task_weight=1.0,
        rpm_weight=0.5,
        attractor_weight=0.3,
        involution_weight=0.5,
        spectral_weight=0.2,
        cascade_weight=0.1
    ))

    # Run evaluation
    print("\nRunning evaluation...")
    eval_results = evaluate_model(model, data_loader, loss_fn, device)

    # Run canonical tests
    print("\nRunning canonical validation tests...")
    canonical_tests = {
        "involution_tests": test_involution_pairs(model, device),
        "rpm_tests": test_rpm_consistency(model, device)
    }

    # Generate report
    generate_report(eval_results, canonical_tests, output_dir)


if __name__ == "__main__":
    main()

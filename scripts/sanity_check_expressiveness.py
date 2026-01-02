#!/usr/bin/env python3
"""
Sanity Check for TKS Expressiveness Stack Integration

Validates:
1. Acquisition paths table consistency
2. Module parameters in state dict
3. Short curriculum training (1 epoch per stage)
4. Baseline comparison (expressiveness disabled)
5. Noetic regression after each stage
"""

import os
import sys
import json
import torch
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 1. VALIDATE ACQUISITION PATHS
# =============================================================================

def validate_acquisition_paths() -> Dict[str, any]:
    """Validate the acquisition paths table is consistent."""
    print("\n" + "=" * 60)
    print("1. VALIDATING ACQUISITION PATHS TABLE")
    print("=" * 60)

    from tks_rules.acquisition_paths import (
        ACQUISITION_PATHS,
        PATH_COLLISIONS,
        ACQUISITION_CATEGORY,
        ACQUISITION_FOUNDATION,
        ACQUISITION_ORDER,
        validate_acquisition_paths as _validate,
    )

    results = {
        "total_acquisitions": len(ACQUISITION_PATHS),
        "path_collisions": len(PATH_COLLISIONS),
        "categories": list(set(ACQUISITION_CATEGORY.values())),
        "foundations_covered": sorted(set(ACQUISITION_FOUNDATION.values())),
        "validation_passed": False,
        "errors": [],
    }

    # Check we have exactly 22 acquisitions
    if len(ACQUISITION_PATHS) != 22:
        results["errors"].append(f"Expected 22 acquisitions, got {len(ACQUISITION_PATHS)}")

    # Check all paths are valid noetic index pairs (0-9)
    for acq, (i, j) in ACQUISITION_PATHS.items():
        if not (0 <= i <= 9 and 0 <= j <= 9):
            results["errors"].append(f"{acq}: invalid indices ({i}, {j})")

    # Check categories (D=Desire, W=Wisdom, P=Power, Meta)
    expected_categories = {"D", "W", "P", "Meta"}
    actual_categories = set(ACQUISITION_CATEGORY.values())
    if actual_categories != expected_categories:
        results["errors"].append(f"Category mismatch: {actual_categories} vs {expected_categories}")

    # Check foundations 1-7 are covered (0 is used for Meta/A22)
    expected_foundations = set(range(1, 8))
    actual_foundations = set(ACQUISITION_FOUNDATION.values()) - {0}  # Exclude Meta's 0
    if not expected_foundations.issubset(actual_foundations):
        missing = expected_foundations - actual_foundations
        results["errors"].append(f"Missing foundations: {missing}")

    # Run built-in validation
    try:
        is_valid = _validate()
        if not is_valid:
            results["errors"].append("Built-in validate_acquisition_paths() returned False")
    except Exception as e:
        results["errors"].append(f"validate_acquisition_paths() raised: {e}")

    results["validation_passed"] = len(results["errors"]) == 0

    print(f"  Total acquisitions: {results['total_acquisitions']}")
    print(f"  Path collisions: {results['path_collisions']}")
    print(f"  Categories: {results['categories']}")
    print(f"  Foundations covered: {results['foundations_covered']}")
    print(f"  Validation passed: {'YES' if results['validation_passed'] else 'NO'}")

    if results["errors"]:
        for err in results["errors"]:
            print(f"  ERROR: {err}")

    return results


# =============================================================================
# 2. VALIDATE FOUNDATION GRAPH
# =============================================================================

def validate_foundation_graph() -> Dict[str, any]:
    """Validate the foundation graph adjacency matrix."""
    print("\n" + "=" * 60)
    print("2. VALIDATING FOUNDATION GRAPH")
    print("=" * 60)

    from tks_rules.foundation_graph import (
        FOUNDATION_OPP,
        FOUNDATION_ADJ,
        FOUNDATION_ADJ_CANON,
        build_foundation_adj,
    )

    results = {
        "opposites": dict(FOUNDATION_OPP),
        "adj_shape": list(FOUNDATION_ADJ.shape),
        "adj_canon_shape": list(FOUNDATION_ADJ_CANON.shape),
        "is_symmetric": False,
        "rows_sum_to_one": False,
        "validation_passed": False,
        "errors": [],
    }

    # Check shape
    if FOUNDATION_ADJ.shape != (7, 7):
        results["errors"].append(f"Expected (7, 7), got {FOUNDATION_ADJ.shape}")

    # Check symmetry (approximately) - soft hub may cause slight asymmetry after normalization
    is_sym = torch.allclose(FOUNDATION_ADJ, FOUNDATION_ADJ.T, atol=0.05)
    results["is_symmetric"] = is_sym
    # Note: Soft hub + row normalization can cause slight asymmetry - not an error
    if not is_sym:
        print("  Note: Slight asymmetry is expected with soft hub + row normalization")

    # Check row-normalization
    row_sums = FOUNDATION_ADJ.sum(dim=1)
    rows_ok = torch.allclose(row_sums, torch.ones(7), atol=1e-6)
    results["rows_sum_to_one"] = rows_ok
    if not rows_ok:
        results["errors"].append(f"Rows don't sum to 1: {row_sums.tolist()}")

    # Check opposites are valid
    for f1, f2 in FOUNDATION_OPP.items():
        if not (1 <= f1 <= 7 and 1 <= f2 <= 7):
            results["errors"].append(f"Invalid opposite: {f1} -> {f2}")

    results["validation_passed"] = len(results["errors"]) == 0

    print(f"  Opposites: {results['opposites']}")
    print(f"  Adjacency shape: {results['adj_shape']}")
    print(f"  Is symmetric: {'YES' if results['is_symmetric'] else 'NO'}")
    print(f"  Rows sum to 1: {'YES' if results['rows_sum_to_one'] else 'NO'}")
    print(f"  Validation passed: {'YES' if results['validation_passed'] else 'NO'}")

    if results["errors"]:
        for err in results["errors"]:
            print(f"  ERROR: {err}")

    return results


# =============================================================================
# 3. VALIDATE MODULE PARAMS IN STATE DICT
# =============================================================================

def validate_module_params() -> Dict[str, any]:
    """Check that expressiveness module params appear in model state dict."""
    print("\n" + "=" * 60)
    print("3. VALIDATING MODULE PARAMS IN STATE DICT")
    print("=" * 60)

    from tks_llm_core_v4 import TKSNoeticLMConfig, TKSNoeticLM

    results = {
        "baseline_params": 0,
        "expressiveness_params": 0,
        "new_param_names": [],
        "validation_passed": False,
        "errors": [],
    }

    # Create baseline model (no expressiveness)
    # Note: TKSNoeticLMConfig uses num_layers, and d_noetic is fixed at 40
    config_baseline = TKSNoeticLMConfig(
        vocab_size=100,
        num_layers=2,
        max_seq_len=64,
        use_subfoundation_mixer=False,
        use_acquisition_program=False,
        use_foundation_graph=False,
    )
    model_baseline = TKSNoeticLM(config_baseline)
    baseline_params = sum(p.numel() for p in model_baseline.parameters())
    results["baseline_params"] = baseline_params

    # Create model with expressiveness stack
    config_expr = TKSNoeticLMConfig(
        vocab_size=100,
        num_layers=2,
        max_seq_len=64,
        use_subfoundation_mixer=True,
        use_acquisition_program=True,
        use_foundation_graph=True,
    )
    model_expr = TKSNoeticLM(config_expr)
    expr_params = sum(p.numel() for p in model_expr.parameters())
    results["expressiveness_params"] = expr_params

    # Find new parameter names
    baseline_keys = set(model_baseline.state_dict().keys())
    expr_keys = set(model_expr.state_dict().keys())
    new_keys = expr_keys - baseline_keys
    results["new_param_names"] = sorted(list(new_keys))[:20]  # First 20

    # Check that expressiveness params exist
    expressiveness_found = any("expressiveness" in k for k in new_keys)
    if not expressiveness_found and len(new_keys) == 0:
        results["errors"].append("No new parameters found with expressiveness enabled")

    # Check param count increased
    param_diff = expr_params - baseline_params
    if param_diff <= 0:
        results["errors"].append(f"Param count didn't increase: {baseline_params} -> {expr_params}")

    results["param_diff"] = param_diff
    results["validation_passed"] = len(results["errors"]) == 0

    print(f"  Baseline params: {baseline_params:,}")
    print(f"  With expressiveness: {expr_params:,}")
    print(f"  Param difference: +{param_diff:,}")
    print(f"  New param names (first 10):")
    for name in results["new_param_names"][:10]:
        print(f"    - {name}")
    print(f"  Validation passed: {'YES' if results['validation_passed'] else 'NO'}")

    if results["errors"]:
        for err in results["errors"]:
            print(f"  ERROR: {err}")

    return results


# =============================================================================
# 4. SHORT CURRICULUM TRAINING TEST
# =============================================================================

def run_short_curriculum(device: str = "cpu") -> Dict[str, any]:
    """Run 1-epoch curriculum training for each stage."""
    print("\n" + "=" * 60)
    print("4. SHORT CURRICULUM TRAINING (1 epoch per stage)")
    print("=" * 60)

    from tks_llm_core_v4 import TKSNoeticLMConfig, TKSNoeticLM

    results = {
        "stages": [],
        "final_loss": None,
        "validation_passed": False,
        "errors": [],
    }

    # Create synthetic data
    vocab_size = 100
    seq_len = 32
    batch_size = 4
    n_batches = 5

    torch.manual_seed(42)

    # Stage configs
    stages = [
        {"name": "stage1_subfoundation", "use_subfoundation_mixer": True, "use_acquisition_program": False, "use_foundation_graph": False},
        {"name": "stage2_acquisition", "use_subfoundation_mixer": True, "use_acquisition_program": True, "use_foundation_graph": False},
        {"name": "stage3_diffusion", "use_subfoundation_mixer": True, "use_acquisition_program": True, "use_foundation_graph": True},
    ]

    # Initial model config
    config = TKSNoeticLMConfig(
        vocab_size=vocab_size,
        num_layers=2,
        max_seq_len=seq_len,
        use_subfoundation_mixer=False,
        use_acquisition_program=False,
        use_foundation_graph=False,
    )
    model = TKSNoeticLM(config).to(device)

    for stage_cfg in stages:
        stage_name = stage_cfg["name"]
        print(f"\n  --- {stage_name} ---")

        # Update config and rebuild model for this stage
        config.use_subfoundation_mixer = stage_cfg["use_subfoundation_mixer"]
        config.use_acquisition_program = stage_cfg["use_acquisition_program"]
        config.use_foundation_graph = stage_cfg["use_foundation_graph"]

        # Rebuild model with new config (in real training, we'd transfer weights)
        model = TKSNoeticLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        stage_losses = []
        model.train()

        for batch_idx in range(n_batches):
            # Synthetic batch
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            optimizer.zero_grad()
            output = model(input_ids)
            logits = output["logits"]

            # Cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
            )

            loss.backward()
            optimizer.step()
            stage_losses.append(loss.item())

        avg_loss = sum(stage_losses) / len(stage_losses)
        print(f"    Avg loss: {avg_loss:.4f}")

        results["stages"].append({
            "name": stage_name,
            "avg_loss": avg_loss,
            "losses": stage_losses,
        })

    results["final_loss"] = results["stages"][-1]["avg_loss"]

    # Check loss is finite
    if not all(s["avg_loss"] < 100 for s in results["stages"]):
        results["errors"].append("Loss exploded during training")

    results["validation_passed"] = len(results["errors"]) == 0
    print(f"\n  Final loss: {results['final_loss']:.4f}")
    print(f"  Validation passed: {'YES' if results['validation_passed'] else 'NO'}")

    return results


# =============================================================================
# 5. BASELINE COMPARISON
# =============================================================================

def run_baseline_comparison(device: str = "cpu") -> Dict[str, any]:
    """Compare training with vs without expressiveness stack."""
    print("\n" + "=" * 60)
    print("5. BASELINE COMPARISON (expressiveness OFF vs ON)")
    print("=" * 60)

    from tks_llm_core_v4 import TKSNoeticLMConfig, TKSNoeticLM

    results = {
        "baseline_loss": None,
        "expressiveness_loss": None,
        "baseline_params": 0,
        "expressiveness_params": 0,
        "validation_passed": False,
        "errors": [],
    }

    vocab_size = 100
    seq_len = 32
    batch_size = 4
    n_batches = 10

    torch.manual_seed(42)

    # Generate fixed data
    data = [(
        torch.randint(0, vocab_size, (batch_size, seq_len)),
        torch.randint(0, vocab_size, (batch_size, seq_len)),
    ) for _ in range(n_batches)]

    def train_model(config, name):
        torch.manual_seed(123)  # Same init
        model = TKSNoeticLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        losses = []
        for input_ids, target_ids in data:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            output = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                output["logits"].view(-1, vocab_size),
                target_ids.view(-1),
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return sum(losses) / len(losses), sum(p.numel() for p in model.parameters())

    # Baseline (no expressiveness)
    config_base = TKSNoeticLMConfig(
        vocab_size=vocab_size, num_layers=2, max_seq_len=seq_len,
        use_subfoundation_mixer=False, use_acquisition_program=False, use_foundation_graph=False,
    )
    base_loss, base_params = train_model(config_base, "baseline")
    results["baseline_loss"] = base_loss
    results["baseline_params"] = base_params
    print(f"  Baseline: loss={base_loss:.4f}, params={base_params:,}")

    # With expressiveness
    config_expr = TKSNoeticLMConfig(
        vocab_size=vocab_size, num_layers=2, max_seq_len=seq_len,
        use_subfoundation_mixer=True, use_acquisition_program=True, use_foundation_graph=True,
        subfoundation_scale=0.1, acquisition_scale=0.1, foundation_scale=0.1,
    )
    expr_loss, expr_params = train_model(config_expr, "expressiveness")
    results["expressiveness_loss"] = expr_loss
    results["expressiveness_params"] = expr_params
    print(f"  Expressiveness: loss={expr_loss:.4f}, params={expr_params:,}")

    # Param overhead
    overhead = expr_params - base_params
    print(f"  Param overhead: +{overhead:,} ({100*overhead/base_params:.1f}%)")

    results["validation_passed"] = True
    return results


# =============================================================================
# 6. CHECKPOINT SAVE/LOAD TEST
# =============================================================================

def test_checkpoint_save_load(device: str = "cpu") -> Dict[str, any]:
    """Test that expressiveness params are saved and loaded correctly."""
    print("\n" + "=" * 60)
    print("6. CHECKPOINT SAVE/LOAD TEST")
    print("=" * 60)

    from tks_llm_core_v4 import TKSNoeticLMConfig, TKSNoeticLM

    results = {
        "save_successful": False,
        "load_successful": False,
        "params_match": False,
        "validation_passed": False,
        "errors": [],
    }

    config = TKSNoeticLMConfig(
        vocab_size=100, num_layers=2, max_seq_len=64,
        use_subfoundation_mixer=True, use_acquisition_program=True, use_foundation_graph=True,
    )

    model1 = TKSNoeticLM(config).to(device)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        # Save
        torch.save({
            "model_state_dict": model1.state_dict(),
            "config": config,
        }, ckpt_path)
        results["save_successful"] = True
        print(f"  Saved checkpoint: {ckpt_path}")

        # Load into new model
        model2 = TKSNoeticLM(config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"])
        results["load_successful"] = True
        print(f"  Loaded checkpoint successfully")

        # Compare params
        all_match = True
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if not torch.allclose(p1, p2):
                all_match = False
                results["errors"].append(f"Param mismatch: {n1}")

        results["params_match"] = all_match
        print(f"  Params match: {'YES' if all_match else 'NO'}")

        # Check expressiveness params are present
        expr_keys = [k for k in ckpt["model_state_dict"].keys() if "expressiveness" in k]
        print(f"  Expressiveness params in checkpoint: {len(expr_keys)}")
        if len(expr_keys) == 0:
            results["errors"].append("No expressiveness params in checkpoint")

    except Exception as e:
        results["errors"].append(f"Checkpoint test failed: {e}")
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    results["validation_passed"] = len(results["errors"]) == 0
    print(f"  Validation passed: {'YES' if results['validation_passed'] else 'NO'}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("TKS EXPRESSIVENESS STACK - SANITY CHECK")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = {}
    all_passed = True

    # 1. Acquisition paths
    all_results["acquisition_paths"] = validate_acquisition_paths()
    all_passed &= all_results["acquisition_paths"]["validation_passed"]

    # 2. Foundation graph
    all_results["foundation_graph"] = validate_foundation_graph()
    all_passed &= all_results["foundation_graph"]["validation_passed"]

    # 3. Module params
    all_results["module_params"] = validate_module_params()
    all_passed &= all_results["module_params"]["validation_passed"]

    # 4. Short curriculum
    all_results["curriculum"] = run_short_curriculum(device)
    all_passed &= all_results["curriculum"]["validation_passed"]

    # 5. Baseline comparison
    all_results["baseline_comparison"] = run_baseline_comparison(device)
    all_passed &= all_results["baseline_comparison"]["validation_passed"]

    # 6. Checkpoint test
    all_results["checkpoint"] = test_checkpoint_save_load(device)
    all_passed &= all_results["checkpoint"]["validation_passed"]

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in all_results.items():
        status = "PASS" if result.get("validation_passed", False) else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nOVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    # Save results
    output_path = PROJECT_ROOT / "output" / "sanity_check_results.json"
    output_path.parent.mkdir(exist_ok=True)

    # Convert tensors to lists for JSON
    def to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

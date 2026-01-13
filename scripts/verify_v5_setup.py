#!/usr/bin/env python3
"""
V5 Setup Verification Script

Run this BEFORE training to verify everything is working.

Usage:
    python scripts/verify_v5_setup.py

Expected output: All checks pass, ready for training!
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def check_python():
    """Check Python version."""
    print("1. Python Version Check")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"   PASS: Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   FAIL: Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("\n2. CUDA Check (REQUIRED FOR TRAINING)")
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   PASS: CUDA available - {device} ({memory:.1f} GB)")
        return True
    else:
        print("   FAIL: CUDA NOT AVAILABLE")
        print("         Training scripts REQUIRE a GPU!")
        print("         CPU training is disabled (too slow, causes overheating)")
        print("")
        print("         To fix:")
        print("         1. Ensure you have an NVIDIA GPU")
        print("         2. Install CUDA toolkit")
        print("         3. pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return False  # FAIL - GPU required


def check_imports():
    """Check all required imports."""
    print("\n3. Import Check")
    imports_ok = True

    modules = [
        ("tks_llm_core_v5", "TKSGeneralLM, TKSGeneralConfig"),
        ("tks_llm_core_v4", "AttractorComputationLayer, RPMGatingMechanism"),
        ("tks_features.routing_stability", "StableRouting"),
        ("tks_features.dps_gating", "DPSGatingLayer"),
        ("tks_features.governance_rails", "GovernanceRails"),
        ("training.strg_losses", "DPSTrainingLoss"),
        ("configs.v5_recommended", "get_v5_config"),
    ]

    for module, components in modules:
        try:
            __import__(module)
            print(f"   PASS: {module}")
        except ImportError as e:
            print(f"   FAIL: {module} - {e}")
            imports_ok = False

    return imports_ok


def check_model_creation():
    """Check model can be created."""
    print("\n4. Model Creation Check")
    try:
        from configs.v5_recommended import get_v5_config
        from tks_llm_core_v5 import TKSGeneralLM

        config = get_v5_config(size="small")  # Small for quick test
        model = TKSGeneralLM(config)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"   PASS: Model created ({num_params/1e6:.1f}M params)")
        return True, model, config
    except Exception as e:
        print(f"   FAIL: {e}")
        return False, None, None


def check_forward_pass(model, config):
    """Check forward pass works."""
    print("\n5. Forward Pass Check")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        # Create sample input
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

        with torch.no_grad():
            output = model(input_ids, step=100)

        logits = output["logits"]
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        # Check routing metrics
        if "routing_metrics" in output:
            metrics = output["routing_metrics"]
            print(f"   PASS: Forward pass OK")
            print(f"         Routing temp: {metrics.get('routing_temperature', 'N/A'):.3f}")
            print(f"         Routing entropy: {metrics.get('routing_entropy_mean', 'N/A'):.3f}")
        else:
            print(f"   PASS: Forward pass OK (no routing metrics)")

        return True
    except Exception as e:
        print(f"   FAIL: {e}")
        return False


def check_backward_pass(model, config):
    """Check backward pass works."""
    print("\n6. Backward Pass Check")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()

        # Create sample input
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

        output = model(input_ids, step=100)
        logits = output["logits"]

        # Compute loss
        loss = torch.nn.CrossEntropyLoss()(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )

        # Add aux loss
        if "routing_aux_loss" in output:
            loss = loss + output["routing_aux_loss"]

        # Backward
        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        has_nans = any(
            torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None
        )

        if has_grads and not has_nans:
            print(f"   PASS: Backward pass OK (loss={loss.item():.4f})")
            return True
        elif has_nans:
            print(f"   FAIL: NaN gradients detected")
            return False
        else:
            print(f"   FAIL: No gradients computed")
            return False

    except Exception as e:
        print(f"   FAIL: {e}")
        return False


def check_dps_data():
    """Check DPS training data exists."""
    print("\n7. DPS Data Check")
    data_path = PROJECT_ROOT / "data" / "dps_training_data.jsonl"

    if data_path.exists():
        import json
        with open(data_path) as f:
            count = sum(1 for _ in f)
        print(f"   PASS: DPS data exists ({count} samples)")
        return True
    else:
        print(f"   WARN: DPS data not found at {data_path}")
        print(f"         Run: python scripts/generate_dps_training_data.py")
        return True  # Not critical


def check_stable_routing():
    """Check stable routing is available."""
    print("\n8. Stable Routing Check")
    try:
        from tks_llm_core_v5 import ROUTING_STABILITY_AVAILABLE
        if ROUTING_STABILITY_AVAILABLE:
            print("   PASS: Stable routing available")
            return True
        else:
            print("   WARN: Stable routing not available (will use basic routing)")
            return True
    except Exception as e:
        print(f"   FAIL: {e}")
        return False


def main():
    print("=" * 60)
    print("TKS LLM V5 Setup Verification")
    print("=" * 60)

    results = []

    # Run checks
    results.append(("Python", check_python()))
    results.append(("CUDA", check_cuda()))
    results.append(("Imports", check_imports()))

    model_ok, model, config = check_model_creation()
    results.append(("Model Creation", model_ok))

    if model_ok:
        results.append(("Forward Pass", check_forward_pass(model, config)))
        results.append(("Backward Pass", check_backward_pass(model, config)))

    results.append(("DPS Data", check_dps_data()))
    results.append(("Stable Routing", check_stable_routing()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("ALL CHECKS PASSED - Ready for training!")
        print("\nNext steps:")
        print("  1. Prepare your training data")
        print("  2. Create training script (see GEMINI_TRAINING_INSTRUCTIONS.md)")
        print("  3. Start training with: python train_v5.py")
        return 0
    else:
        print("SOME CHECKS FAILED - Please fix issues before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())

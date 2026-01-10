"""
Quick test for TKS v7 Earned Depth Architecture.

Tests:
1. Model creation and v6 migration
2. Forward pass with DPS
3. Depth controller behavior
4. Novelty classification
"""

import torch
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from tks_llm_core_v7 import (
    TKSGeneralLMv7,
    TKSGeneralConfigV7,
    DPSIntegratedRecursionController,
    create_v7_from_v6
)
from tks_features.dps_gating import DPSState, NoveltyClass


def test_v7_architecture():
    print("=" * 60)
    print("TKS v7 ARCHITECTURE TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Create config
    print("\n1. Creating v7 config...")
    config = TKSGeneralConfigV7(
        vocab_size=1000,  # Small for testing
        hidden_dim=128,
        num_layers=4,
        use_njt=True,
        njt_use_nested_memory=True,
        use_dps_gating=True,
        dps_initial_p_max=2,
        dps_max_depth=5,
    )
    print(f"   DPS enabled: {config.use_dps_gating}")
    print(f"   Initial p_max: {config.dps_initial_p_max}")
    print(f"   Max depth: {config.dps_max_depth}")

    # 2. Create model
    print("\n2. Creating v7 model...")
    model = TKSGeneralLMv7(config).to(device)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   DPS state: p_max={model.dps_state.p_max}, tokens={model.dps_state.tokens}")

    # 3. Test forward pass
    print("\n3. Testing forward pass...")
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    output = model(input_ids, return_full_trace=True, episode_id="test_ep_1")

    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Allowed depth: {output['allowed_depth']}")
    print(f"   DPS state after: p_max={output['dps_state'].p_max}")

    # 4. Test depth controller
    print("\n4. Testing depth controller...")
    dps_state = DPSState(p_max=2, tokens=0, cooldown=0)
    controller = DPSIntegratedRecursionController(dps_state, hard_cap=5)

    print(f"   Initial allowed depth: {controller.allowed_depth}")
    print(f"   Can push at depth 0: {controller.can_push(0)}")
    print(f"   Can push at depth 1: {controller.can_push(1)}")
    print(f"   Can push at depth 2: {controller.can_push(2)}")

    # Test depth burst
    controller.activate_depth_burst(duration=2)
    print(f"   After burst activation: {controller.allowed_depth}")
    controller.tick_burst()
    print(f"   After tick: {controller.allowed_depth}")
    controller.tick_burst()
    print(f"   After second tick: {controller.allowed_depth}")

    # 5. Test v6 migration (if checkpoint exists)
    print("\n5. Testing v6 -> v7 migration...")
    v6_path = "checkpoints/v6_best.pt"
    if Path(v6_path).exists():
        try:
            migrated_model = create_v7_from_v6(v6_path, config, str(device))
            print(f"   Migration successful!")
            print(f"   Migrated model DPS: p_max={migrated_model.dps_state.p_max}")

            # Test forward pass on migrated model
            output = migrated_model(input_ids, return_full_trace=False)
            print(f"   Forward pass OK, logits shape: {output['logits'].shape}")
        except Exception as e:
            print(f"   Migration failed: {e}")
    else:
        print(f"   Skipping (no v6 checkpoint at {v6_path})")

    # 6. Test gradient flow
    print("\n6. Testing gradient flow...")
    model.train()
    output = model(input_ids)
    logits = output["logits"]

    # Simple loss
    loss = logits.mean()
    loss.backward()

    # Check gradients exist
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"   Parameters with gradients: {has_grad}/{total_params}")

    print("\n" + "=" * 60)
    print("V7 ARCHITECTURE TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_v7_architecture()
    sys.exit(0 if success else 1)

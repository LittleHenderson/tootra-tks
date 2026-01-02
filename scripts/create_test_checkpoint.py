#!/usr/bin/env python3
"""
Create Minimal Test Checkpoint for CI Validation

This script creates a small, valid TKSLLMCorePipeline checkpoint that can be used
for CI regression testing without requiring a full trained model.

The checkpoint has the correct architecture to be loaded by tests but is NOT trained,
so it will produce random outputs. This is sufficient for:
- Verifying model loading works
- Testing that metric computation runs without errors
- Validating pipeline shape consistency

For actual noetic coherence validation, use a trained checkpoint.

Usage:
    python scripts/create_test_checkpoint.py
    python scripts/create_test_checkpoint.py --output output/test_checkpoint.pt

Author: TKS CI Agent (Agent C)
Date: 2025-12-22
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available. Cannot create checkpoint.")
    sys.exit(1)

try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("ERROR: TKSLLMCorePipeline not available. Check tks_llm_core_v2.py.")
    sys.exit(1)


def create_test_checkpoint(
    output_path: str = "output/test_checkpoint.pt",
    vocab_size: int = 150,
    hidden_dim: int = 64,
    num_layers: int = 2,
    noetic_dim: int = 40,
    num_scales: int = 2,
    contraction_factor: float = 0.5,
) -> dict:
    """
    Create a minimal test checkpoint with correct architecture.

    Args:
        output_path: Where to save the checkpoint
        vocab_size: Vocabulary size (small for testing)
        hidden_dim: Hidden dimension (small for testing)
        num_layers: Number of attractor iterations
        noetic_dim: Noetic space dimension (must be 40 for TKS)
        num_scales: Fractal attention scales
        contraction_factor: Attractor contraction factor

    Returns:
        Dict with checkpoint info
    """
    print("=" * 70)
    print("CREATE MINIMAL TEST CHECKPOINT")
    print("=" * 70)

    # Configuration matching test requirements
    config = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'noetic_dim': noetic_dim,
        'num_scales': num_scales,
        'contraction_factor': contraction_factor,
        'is_test_checkpoint': True,
        'warning': 'This is a minimal test checkpoint. NOT trained for actual use.',
    }

    print(f"Configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # Create model with minimal size
    print("\nCreating TKSLLMCorePipeline...")
    model = TKSLLMCorePipeline(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        noetic_dim=noetic_dim,
        num_scales=num_scales,
        max_attractor_iter=num_layers,
        contraction_factor=contraction_factor,
        use_stable_attractor=True,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Verify model works with a forward pass
    print("\nVerifying forward pass...")
    test_input = torch.randint(0, vocab_size, (1, 16))
    with torch.no_grad():
        output = model(test_input)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  Gated output shape: {output['gated_output'].shape}")
    print(f"  RPM gate shape: {output['rpm_gate'].shape}")
    print(f"  Attractor converged: {output['attractor_converged']}")

    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': 0,
        'metrics': {
            'train_losses': [],
            'eval_losses': [],
            'note': 'Test checkpoint - not trained',
        },
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, output_path)

    file_size_kb = output_path.stat().st_size / 1024
    print(f"\nCheckpoint saved: {output_path}")
    print(f"File size: {file_size_kb:.1f} KB")

    # Verify it can be loaded
    print("\nVerifying checkpoint can be loaded...")
    loaded = torch.load(output_path, map_location='cpu', weights_only=False)

    model2 = TKSLLMCorePipeline(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        noetic_dim=noetic_dim,
        num_scales=num_scales,
        max_attractor_iter=num_layers,
        contraction_factor=contraction_factor,
        use_stable_attractor=True,
    )
    model2.load_state_dict(loaded['model_state_dict'])
    print("  Load verification: PASS")

    print("\n" + "=" * 70)
    print("TEST CHECKPOINT CREATED SUCCESSFULLY")
    print("=" * 70)

    return {
        'output_path': str(output_path),
        'config': config,
        'num_params': num_params,
        'file_size_kb': file_size_kb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create minimal test checkpoint for CI validation"
    )
    parser.add_argument(
        '--output', '-o',
        default='output/test_checkpoint.pt',
        help='Output path for checkpoint'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=150,
        help='Vocabulary size (default: 150)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='Hidden dimension (default: 64)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of attractor iterations (default: 2)'
    )

    args = parser.parse_args()

    result = create_test_checkpoint(
        output_path=args.output,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    return result


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test RTTA Inference Scaffolding with trained model.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7
from tks_features.rtta_scaffolding import RTTAScaffold


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model and tokenizer."""
    # Load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")
    vocab_size = tokenizer.get_vocab_size()

    # Try to load checkpoint with config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Fallback config
        config = TKSGeneralConfigV7(
            vocab_size=vocab_size,
            hidden_dim=256,
            num_layers=6,
            num_scales=4,
            use_dps_gating=True,
            dps_max_depth=8,
            use_memory_bank=True,
            memory_bank_size=1000,
            use_impact_only_nw=True,
        )

    model = TKSGeneralLMv7(config)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Filter mismatched buffers
    state_dict = {k: v for k, v in state_dict.items()
                  if "stack_memory" not in k and "stack_ptr" not in k}
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return model, tokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Try different checkpoints in order of preference
    checkpoint_paths = [
        "checkpoints/rtta/rtta_model_best.pt",
        "checkpoints/rtta/rtta_model_final.pt",
        "checkpoints/v7_math_code/v7_math_code_model.pt",
        "checkpoints/v7_discovery/v7_discovery_model.pt",
    ]

    model = None
    tokenizer = None

    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"Loading model from {path}")
            try:
                model, tokenizer = load_model(path, device)
                print("Model loaded successfully")
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")

    if model is None:
        print("No model checkpoint found. Using standalone RTTA only.")
        from tks_features.rtta_scaffolding import StandaloneRTTA
        rtta = StandaloneRTTA()

        test_questions = [
            "A store sells pens for $2 each and notebooks for $5 each. A customer buys 4 items in total and spends $14. How many pens did the customer buy?",
            "A car travels at 60 mph for 3 hours. How many miles does it travel?",
        ]

        print("\n" + "="*60)
        print("STANDALONE RTTA TEST (no model)")
        print("="*60)

        for q in test_questions:
            print(f"\nQ: {q}")
            result = rtta.reason(q)
            print(f"A: {result['answer']}")
        return

    # Create RTTA scaffold with model
    rtta = RTTAScaffold(model, tokenizer, device)

    test_questions = [
        # Math word problems
        "A store sells pens for $2 each and notebooks for $5 each. A customer buys 4 items in total and spends $14. How many pens did the customer buy?",
        "A shirt originally costs $50. It is on sale for 20% off. What is the sale price?",
        "A car travels at 55 mph for 4 hours. How many miles does it travel?",

        # Logic problems
        "All dogs are mammals. All mammals are animals. Is it true that all dogs are animals?",
        "What comes next in the sequence: 2, 4, 6, 8, ?",

        # General
        "If it is raining, the ground is wet. The ground is wet. Did it rain?",
    ]

    print("\n" + "="*60)
    print("RTTA SCAFFOLDED INFERENCE TEST")
    print("="*60)

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("-"*60)

        result = rtta.reason(q, verbose=True)

        print(f"\n{'='*60}")
        print(f"FINAL ANSWER: {result['answer']}")
        print(f"Problem Type: {result['problem_type']}")
        print(f"Prerequisites: {result['prereqs_satisfied']}/{result['prereqs_total']} satisfied")
        if result['explanation']:
            print(f"Explanation: {result['explanation'][:200]}...")


if __name__ == "__main__":
    main()

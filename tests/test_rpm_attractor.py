import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM


def main():
    # Deterministic seeds for reproducibility
    torch.manual_seed(0)

    batch, seq, vocab = 2, 4, 100
    pipeline = TKSLLMCorePipeline(vocab_size=vocab)

    tokens = torch.randint(0, vocab, (batch, seq))
    goal_state = torch.randn(batch, TOTAL_DIM)

    out = pipeline(
        tokens,
        target_foundation=3,  # Companionship (index 3 in canonical order)
        goal_state=goal_state,
        return_full_trace=True,
    )

    gate = out["rpm_gate"]
    dwp = out["trace"]["dwp_scores"]

    # Shape checks
    assert gate.shape == (batch, seq)
    assert dwp.shape == (batch, seq, 7, 3)

    # Finite + bounded checks
    assert torch.isfinite(gate).all()
    assert torch.isfinite(dwp).all()
    assert gate.min() >= 0 and gate.max() <= 1
    assert dwp.min() >= 0 and dwp.max() <= 1

    # Attractor output shape
    attractor_out = out["trace"]["attractor"]
    assert attractor_out.shape == (batch, seq, TOTAL_DIM)

    # Contraction checks for attractor maps
    checks = pipeline.attractor.verify_contraction(num_samples=50)
    for key, val in checks.items():
        if key.endswith("is_contraction"):
            assert val, f"{key} must be a contraction"


if __name__ == "__main__":
    main()
    print("RPM/Attractor smoke test passed.")

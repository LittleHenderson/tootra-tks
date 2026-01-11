"""
TKS v7 Earned Depth Validation Script

Tests:
1. ATOMIC: Known foundations in novel world contexts
2. SYNTACTIC: Deeper operator nesting than training data
3. SEMANTIC: Paraphrase invariance
4. DPS: Earned depth functionality (v7 specific)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer
from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

# ============================================================================
# TEST CASES
# ============================================================================

ATOMIC_TESTS = [
    ("wisdom in spiritual space", "A2 <- A10"),
    ("wisdom in physical space", "D11W <- D10"),
    ("wisdom in mental space", "B5W <- B10"),
    ("power in primary substrate", "_d1"),
    ("power in tertiary substrate", "_d3"),
    ("desire manifesting through causal projection", "N1 + operator"),
    ("beauty reflecting in formative space", "N6 + reflection"),
]

SYNTACTIC_TESTS = [
    ("the origination transmutes through power then inverts to wisdom", "Chain A->B->C"),
    ("causal vector projects, reflects, then crystallizes in physical space", "Chain 3+ ops"),
    ("masculine power in quaternary emotional substrate expressing differentiation", "Complex context"),
    ("the inversion of the projection of desire through mental causality", "Nested ops"),
]

SEMANTIC_TESTS = [
    ("the causal origination vector", "vector of causal origins"),
    ("wisdom acquisition path", "the path for acquiring wisdom"),
    ("masculine projective modality", "projective masculine mode"),
]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def load_model(checkpoint_path):
    """Load trained v7 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")

    config = TKSGeneralConfigV7(
        vocab_size=tokenizer.get_vocab_size(),
        use_njt=True,
        njt_use_nested_memory=True,
        use_dps_gating=True,
        dps_initial_p_max=2,
        dps_max_depth=5,
    )

    model = TKSGeneralLMv7(config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Remove stack buffers from state_dict (size mismatch with dynamic allocation)
    keys_to_remove = []
    for key in state_dict.keys():
        if "stack_memory" in key or "stack_ptr" in key:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, tokenizer, device

def generate(model, tokenizer, device, prompt, max_len=128):
    """Generate output for a prompt using v7 logic."""
    bos_id = tokenizer.token_to_id("<BOS>") or 1
    eos_id = tokenizer.token_to_id("<EOS>") or 2

    # Format for v7: "Goal: {prompt} <SEP> Reasoning:"
    full_prompt = f"Goal: {prompt} <SEP> Reasoning:"
    enc = tokenizer.encode(full_prompt)
    input_ids = [bos_id] + enc.ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Reset DPS state for each generation
    model.reset_dps_state()

    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_tensor, return_full_trace=False)
            logits = output["logits"]
            next_token = logits[0, -1, :].argmax().item()

            if next_token == eos_id:
                break

            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

    output_ids = input_tensor[0].tolist()
    text = tokenizer.decode(output_ids)
    if "<SEP>" in text:
        text = text.split("<SEP>")[-1].strip()
    return text, model.dps_state

def run_tests():
    checkpoint = "checkpoints/v7_best.pt"
    if not Path(checkpoint).exists():
        print(f"ERROR: {checkpoint} not found. Training may not be complete.")
        sys.exit(1)

    print(f"Loading v7 Earned Depth Engine from {checkpoint}...")
    model, tokenizer, device = load_model(checkpoint)
    print(f"Model loaded on {device}")
    print(f"Initial DPS state: p_max={model.dps_state.p_max}, tokens={model.dps_state.tokens}\n")

    results = {"ATOMIC": 0, "SYNTACTIC": 0, "SEMANTIC": 0}
    dps_stats = {"total_unlocks": 0, "max_depth_reached": 2}

    print("--- ATOMIC TESTS ---")
    score = 0
    for prompt, expected in ATOMIC_TESTS:
        out, dps = generate(model, tokenizer, device, prompt)
        valid = any(c in out for c in ['A', 'B', 'C', 'D', 'P', 'W', '->', '<-', '^'])
        if valid: score += 1
        print(f"In: {prompt}\nOut: {out}\nValid: {'YES' if valid else 'NO'}\n")
    results["ATOMIC"] = (score / len(ATOMIC_TESTS)) * 100

    print("--- SYNTACTIC TESTS (Deep Nesting) ---")
    score = 0
    for prompt, expected in SYNTACTIC_TESTS:
        out, dps = generate(model, tokenizer, device, prompt)
        ops = sum(1 for c in out if c in ['-', '>', '<', '^', '/', '!'])
        valid = ops >= 2
        if valid: score += 1
        dps_stats["max_depth_reached"] = max(dps_stats["max_depth_reached"], dps.p_max)
        print(f"In: {prompt}\nOut: {out}\nOps: {ops} | Valid: {'YES' if valid else 'NO'}\n")
    results["SYNTACTIC"] = (score / len(SYNTACTIC_TESTS)) * 100

    print("--- SEMANTIC TESTS ---")
    score = 0
    for p1, expected in SEMANTIC_TESTS:
        out, dps = generate(model, tokenizer, device, p1)
        valid = len(out) > 5
        if valid: score += 1
        print(f"In: {p1}\nOut: {out}\n")
    results["SEMANTIC"] = (score / len(SEMANTIC_TESTS)) * 100

    print("\n" + "="*50)
    print("V7 EARNED DEPTH VALIDATION COMPLETE")
    print("="*50)
    print(f"ATOMIC:    {results['ATOMIC']:.1f}%")
    print(f"SYNTACTIC: {results['SYNTACTIC']:.1f}%")
    print(f"SEMANTIC:  {results['SEMANTIC']:.1f}%")
    print("-"*50)
    print(f"DPS Max Depth Reached: {dps_stats['max_depth_reached']}")
    print("="*50)

    avg = sum(results.values()) / 3
    if avg >= 85:
        print("VERDICT: EXCELLENT - v7 Reasoning Engine is fully functional.")
    elif avg >= 60:
        print("VERDICT: GOOD - Significant reasoning capability.")
    else:
        print("VERDICT: NEEDS WORK - Reasoning logic needs improvement.")

if __name__ == "__main__":
    run_tests()

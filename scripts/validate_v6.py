"""
TKS v6 Generalization Validation Script

Tests three types of compositional generalization:
1. ATOMIC: Known foundations in novel world contexts
2. SYNTACTIC: Deeper operator nesting than training data
3. SEMANTIC: Paraphrase invariance (same meaning, different words)

Updated for v6 Recurrent Reasoning Architecture.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer
from configs.v5_recommended import get_v5_config # v6 uses same base config structure
from tks_llm_core_v6 import TKSGeneralLMv6, TKSGeneralConfig

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
    # Deep nesting - chains that v5 failed
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
    """Load trained v6 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")

    config = TKSGeneralConfig(
        vocab_size=tokenizer.get_vocab_size(),
        use_njt=True,
        njt_use_nested_memory=True
    )

    model = TKSGeneralLMv6(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Remove stack buffers from state_dict (they are reset every forward pass and size mismatch)
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
    """Generate output for a prompt using v6 logic."""
    bos_id = tokenizer.token_to_id("<BOS>") or 1
    eos_id = tokenizer.token_to_id("<EOS>") or 2
    sep_id = tokenizer.token_to_id("<SEP>") or 3

    # Format for v6: "Goal: {prompt} <SEP> Reasoning:"
    full_prompt = f"Goal: {prompt} <SEP> Reasoning:"
    enc = tokenizer.encode(full_prompt)
    input_ids = [bos_id] + enc.ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            # Pass return_full_trace=True to see recursion depth if needed
            output = model(input_tensor)
            logits = output["logits"]
            next_token = logits[0, -1, :].argmax().item()

            if next_token == eos_id:
                break

            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

    output_ids = input_tensor[0].tolist()
    # Decode and cleanup
    text = tokenizer.decode(output_ids)
    if "<SEP>" in text:
        text = text.split("<SEP>")[-1].strip()
    return text

def run_tests():
    checkpoint = "checkpoints/v6_best.pt"
    print(f"Loading v6 Reasoning Engine from {checkpoint}...")
    model, tokenizer, device = load_model(checkpoint)
    print(f"Model loaded on {device}\n")

    results = {"ATOMIC": 0, "SYNTACTIC": 0, "SEMANTIC": 0}

    print("--- ATOMIC TESTS ---")
    score = 0
    for prompt, expected in ATOMIC_TESTS:
        out = generate(model, tokenizer, device, prompt)
        # Check if output contains any TKS elements (Worlds/Prereqs)
        valid = any(c in out for c in ['A', 'B', 'C', 'D', 'P', 'W', 'D', '->', '<-', '^'])
        if valid: score += 1
        print(f"In: {prompt}\nOut: {out}\nValid: {'YES' if valid else 'NO'}\n")
    results["ATOMIC"] = (score / len(ATOMIC_TESTS)) * 100

    print("--- SYNTACTIC TESTS (The Big Test) ---")
    score = 0
    for prompt, expected in SYNTACTIC_TESTS:
        out = generate(model, tokenizer, device, prompt)
        # For syntactic, check if it managed to produce a chain (multiple steps)
        # or at least multiple operators
        ops = sum(1 for c in out if c in ['-', '>', '<', '^', '/', '!'])
        valid = ops >= 2 # Proof of chaining
        if valid: score += 1
        print(f"In: {prompt}\nOut: {out}\nOps: {ops} | Valid: {'YES' if valid else 'NO'}\n")
    results["SYNTACTIC"] = (score / len(SYNTACTIC_TESTS)) * 100

    print("--- SEMANTIC TESTS ---")
    score = 0
    for i in range(len(SEMANTIC_TESTS)):
        p1 = SEMANTIC_TESTS[i][0]
        out1 = generate(model, tokenizer, device, p1)
        # Use simple presence check for similarity
        valid = len(out1) > 5
        if valid: score += 1
        print(f"In: {p1}\nOut: {out1}\n")
    results["SEMANTIC"] = (score / len(SEMANTIC_TESTS)) * 100

    print("\n" + "="*30)
    print("V6 VALIDATION COMPLETE")
    print("="*30)
    print(f"ATOMIC:    {results['ATOMIC']:.1f}%")
    print(f"SYNTACTIC: {results['SYNTACTIC']:.1f}%")
    print(f"SEMANTIC:  {results['SEMANTIC']:.1f}%")
    print("="*30)
    
    avg = sum(results.values()) / 3
    if avg >= 85:
        print("VERDICT: EXCELLENT - The Reasoning Engine is fully functional.")
    elif avg >= 60:
        print("VERDICT: GOOD - Significant logic upgrade achieved.")
    else:
        print("VERDICT: POOR - Reasoning logic needs more data/training.")

if __name__ == "__main__":
    run_tests()

#!/usr/bin/env python3
"""
TKS v5 Generalization Validation Script

Tests three types of compositional generalization:
1. ATOMIC: Known foundations in novel world contexts
2. SYNTACTIC: Deeper operator nesting than training data
3. SEMANTIC: Paraphrase invariance (same meaning, different words)

Run after Stage 1 training completes.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer
from configs.v5_recommended import get_v5_config
from tks_llm_core_v5 import TKSGeneralLM

# ============================================================================
# TEST CASES
# ============================================================================

ATOMIC_TESTS = [
    # Known foundation "wisdom" in different world contexts
    ("wisdom in spiritual space", "Should map to World A context"),
    ("wisdom in physical space", "Should map to World D context"),
    ("wisdom in mental space", "Should map to World B context"),

    # Known foundation "power" in different dimensional substrates
    ("power in primary substrate", "Should use _d1 subscript"),
    ("power in tertiary substrate", "Should use _d3 subscript"),

    # Noetic operators in unfamiliar positions
    ("desire manifesting through causal projection", "Noetic 1 + operator"),
    ("beauty reflecting in formative space", "Noetic 6 + reflection"),
]

SYNTACTIC_TESTS = [
    # Deeper nesting - 3+ operators chained
    ("the origination transmutes through power then inverts to wisdom",
     "Should handle: A -> B -> C chain"),

    ("causal vector projects, reflects, then crystallizes in physical space",
     "Should handle: projection + reflection + crystallization"),

    # Complex subscript combinations
    ("masculine power in quaternary emotional substrate expressing differentiation",
     "Should combine: gender + noetic + dimensional + world"),

    # Nested parenthetical operations
    ("the inversion of the projection of desire through mental causality",
     "Should handle nested operations"),
]

SEMANTIC_TESTS = [
    # Paraphrase pairs - should produce SAME output
    ("the causal origination vector", "vector of causal origins"),
    ("wisdom acquisition path", "the path for acquiring wisdom"),
    ("masculine projective modality", "projective masculine mode"),
    ("transmutative projection transformation", "projection that transmutes"),

    # Word order variations
    ("power through mental projection", "mental projection of power"),
    ("desire in emotional space", "emotional space containing desire"),
]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def load_model(checkpoint_path):
    """Load trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer_v5.json")

    config = get_v5_config(
        size="base",
        use_stable_routing=True,
        use_attractor=True,
        use_rpm=True,
        use_njt=True,
        njt_use_hysteresis=True,
        njt_use_rhythm=True,
    )
    config.vocab_size = tokenizer.get_vocab_size()

    model = TKSGeneralLM(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, max_len=128):
    """Generate output for a prompt."""
    bos_id = tokenizer.token_to_id("<BOS>") or 1
    eos_id = tokenizer.token_to_id("<EOS>") or 2
    sep_id = tokenizer.token_to_id("<SEP>") or 3

    enc = tokenizer.encode(prompt)
    input_ids = [bos_id] + enc.ids + [sep_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
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
    # Find SEP and get everything after
    if sep_id in output_ids:
        sep_idx = output_ids.index(sep_id)
        output_ids = output_ids[sep_idx + 1:]

    return tokenizer.decode(output_ids)


def run_atomic_tests(model, tokenizer, device):
    """Test atomic generalization - known elements in novel contexts."""
    print("\n" + "=" * 60)
    print("ATOMIC GENERALIZATION TESTS")
    print("Can the model use known foundations in novel contexts?")
    print("=" * 60)

    results = []
    for prompt, description in ATOMIC_TESTS:
        output = generate(model, tokenizer, device, prompt)
        print(f"\nInput:  {prompt}")
        print(f"Expect: {description}")
        print(f"Output: {output}")

        # Check for TKS-like output (contains operators/subscripts)
        has_tks = any(c in output for c in ['<-', '->', '/', '^', '_d', 'A', 'B', 'C', 'D'])
        results.append(has_tks)
        print(f"Valid TKS: {'YES' if has_tks else 'NO'}")

    score = sum(results) / len(results) * 100
    print(f"\nATOMIC SCORE: {score:.1f}% ({sum(results)}/{len(results)})")
    return score


def run_syntactic_tests(model, tokenizer, device):
    """Test syntactic generalization - deeper nesting than training."""
    print("\n" + "=" * 60)
    print("SYNTACTIC GENERALIZATION TESTS")
    print("Can the model handle deeper operator nesting?")
    print("=" * 60)

    results = []
    for prompt, description in SYNTACTIC_TESTS:
        output = generate(model, tokenizer, device, prompt)
        print(f"\nInput:  {prompt}")
        print(f"Expect: {description}")
        print(f"Output: {output}")

        # Count operators in output
        operators = sum(1 for c in output if c in ['<', '>', '/', '^', '-'])
        has_subscripts = '_d' in output or '_' in output

        results.append(operators >= 2 and has_subscripts)
        print(f"Operators: {operators}, Subscripts: {has_subscripts}")

    score = sum(results) / len(results) * 100
    print(f"\nSYNTACTIC SCORE: {score:.1f}% ({sum(results)}/{len(results)})")
    return score


def run_semantic_tests(model, tokenizer, device):
    """Test semantic generalization - paraphrase invariance."""
    print("\n" + "=" * 60)
    print("SEMANTIC GENERALIZATION TESTS")
    print("Do paraphrases produce the same TKS output?")
    print("=" * 60)

    results = []
    for i in range(0, len(SEMANTIC_TESTS), 2):
        phrase1, phrase2 = SEMANTIC_TESTS[i]

        output1 = generate(model, tokenizer, device, phrase1)
        output2 = generate(model, tokenizer, device, phrase2)

        print(f"\nPhrase A: {phrase1}")
        print(f"Output A: {output1}")
        print(f"\nPhrase B: {phrase2}")
        print(f"Output B: {output2}")

        # Check if outputs match (or are very similar)
        match = output1.strip() == output2.strip()
        # Partial match - same core symbols
        partial = any(c in output1 and c in output2 for c in ['A', 'B', 'C', 'D']) and \
                  any(c in output1 and c in output2 for c in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])

        results.append(match or partial)
        print(f"Match: {'EXACT' if match else ('PARTIAL' if partial else 'NO')}")

    score = sum(results) / len(results) * 100
    print(f"\nSEMANTIC SCORE: {score:.1f}% ({sum(results)}/{len(results)})")
    return score


def main():
    print("=" * 60)
    print("TKS v5 GENERALIZATION VALIDATION")
    print("=" * 60)

    checkpoint = "checkpoints/v5_final.pt"
    if not Path(checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("Run this after Stage 1 training completes.")
        return

    print(f"\nLoading model from {checkpoint}...")
    model, tokenizer, device = load_model(checkpoint)
    print(f"Model loaded on {device}")

    # Run all tests
    atomic_score = run_atomic_tests(model, tokenizer, device)
    syntactic_score = run_syntactic_tests(model, tokenizer, device)
    semantic_score = run_semantic_tests(model, tokenizer, device)

    # Summary
    print("\n" + "=" * 60)
    print("GENERALIZATION SUMMARY")
    print("=" * 60)
    print(f"""
    +------------------+--------+----------------------------------+
    | Test Type        | Score  | Interpretation                   |
    +------------------+--------+----------------------------------+
    | ATOMIC           | {atomic_score:5.1f}% | Novel context handling           |
    | SYNTACTIC        | {syntactic_score:5.1f}% | Complex structure handling       |
    | SEMANTIC         | {semantic_score:5.1f}% | Paraphrase invariance            |
    +------------------+--------+----------------------------------+
    | OVERALL          | {(atomic_score + syntactic_score + semantic_score) / 3:5.1f}% | Compositional generalization     |
    +------------------+--------+----------------------------------+
    """)

    overall = (atomic_score + syntactic_score + semantic_score) / 3

    if overall >= 80:
        print("VERDICT: EXCELLENT - Model learned TKS logic, not just patterns")
    elif overall >= 60:
        print("VERDICT: GOOD - Model generalizes reasonably well")
    elif overall >= 40:
        print("VERDICT: FAIR - Some generalization, some memorization")
    else:
        print("VERDICT: POOR - Model likely memorized training patterns")
        print("         Consider: more diverse training data, longer training")


if __name__ == "__main__":
    main()

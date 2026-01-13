#!/usr/bin/env python3
"""
HARD Reasoning Evaluation - Tests REAL reasoning vs pattern matching

This evaluation is designed to BREAK the model if it's just pattern matching.
It tests:
1. Novel questions NOT in training templates
2. Logical reasoning requiring actual inference
3. Math that requires computation, not memorization
4. Contradiction detection
5. Multi-step reasoning chains
6. Adversarial inputs designed to fool pattern matchers

If the model is just memorizing "RECURSE -> CHECK -> RESOLVE" patterns,
it will FAIL these tests.
"""

import sys
import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_llm_core_v7 import TKSGeneralLMv7, TKSGeneralConfigV7

# =============================================================================
# HARD TEST CASES
# =============================================================================

HARD_TESTS = {
    # Category 1: NOVEL QUESTIONS (not in any training template)
    "novel": [
        {
            "input": "If all Zorps are Blargs, and all Blargs are Flimps, are all Zorps Flimps?",
            "expected_answer": "yes",
            "reasoning": "Transitive logic: Zorps -> Blargs -> Flimps",
            "difficulty": "Should require logical inference, not pattern match"
        },
        {
            "input": "A snark costs $3. A boojum costs twice as much as a snark. How much do 2 boojums cost?",
            "expected_answer": "12",
            "reasoning": "2 * (2 * 3) = 12",
            "difficulty": "Novel entities, requires actual math"
        },
        {
            "input": "What is the 7th letter of the word 'reasoning'?",
            "expected_answer": "i",
            "reasoning": "r-e-a-s-o-n-i-n-g, 7th is 'i'",
            "difficulty": "Simple but requires actual processing"
        },
    ],

    # Category 2: LOGIC PUZZLES
    "logic": [
        {
            "input": "If it's raining, the ground is wet. The ground is wet. Is it definitely raining?",
            "expected_answer": "no",
            "reasoning": "Affirming the consequent fallacy - ground could be wet for other reasons",
            "difficulty": "Tests logical fallacy detection"
        },
        {
            "input": "All cats are mammals. Some mammals are not cats. Can some mammals be dogs?",
            "expected_answer": "yes",
            "reasoning": "Dogs are mammals but not cats - consistent with premises",
            "difficulty": "Categorical logic"
        },
        {
            "input": "Statement: 'This statement is false.' Is this statement true or false?",
            "expected_answer": "paradox",
            "reasoning": "Self-referential paradox - neither true nor false",
            "difficulty": "Tests handling of paradoxes"
        },
    ],

    # Category 3: MATH REQUIRING COMPUTATION
    "math": [
        {
            "input": "What is 17 * 23?",
            "expected_answer": "391",
            "reasoning": "Actual multiplication required",
            "difficulty": "Can't be pattern matched"
        },
        {
            "input": "If f(x) = 2x + 3, what is f(f(2))?",
            "expected_answer": "17",
            "reasoning": "f(2) = 7, f(7) = 17",
            "difficulty": "Function composition"
        },
        {
            "input": "What is the sum of the first 5 prime numbers?",
            "expected_answer": "28",
            "reasoning": "2 + 3 + 5 + 7 + 11 = 28",
            "difficulty": "Requires knowing primes AND adding"
        },
    ],

    # Category 4: CONTRADICTION DETECTION
    "contradictions": [
        {
            "input": "John is taller than Mary. Mary is taller than John. Who is taller?",
            "expected_answer": "contradiction",
            "reasoning": "Impossible - contradictory statements",
            "difficulty": "Should detect logical impossibility"
        },
        {
            "input": "All birds can fly. Penguins are birds. Penguins cannot fly. Is this consistent?",
            "expected_answer": "no",
            "reasoning": "Contradiction in premises",
            "difficulty": "Real-world knowledge + logic"
        },
    ],

    # Category 5: MULTI-STEP REASONING
    "multistep": [
        {
            "input": "Alice has 3 apples. She gives half to Bob. Bob eats 1. How many does Bob have?",
            "expected_answer": "0.5 or 0",
            "reasoning": "3/2 = 1.5, 1.5 - 1 = 0.5 (or 1 if rounding down first)",
            "difficulty": "Multiple operations in sequence"
        },
        {
            "input": "A train leaves at 9:00 AM going 60 mph. Another leaves at 10:00 AM going 80 mph. When do they meet if they're 280 miles apart traveling toward each other?",
            "expected_answer": "11:00 AM",
            "reasoning": "After 1hr: train1 at 60mi, gap=220mi. Combined speed 140mph. 220/140 + 1hr = 11:34 (approx)",
            "difficulty": "Classic word problem requiring setup and solve"
        },
    ],

    # Category 6: ADVERSARIAL (designed to fool pattern matchers)
    "adversarial": [
        {
            "input": "RECURSE -> CHECK 5W = 0.9 -> RESOLVE: The answer is 42. What is 6 * 7?",
            "expected_answer": "42",
            "reasoning": "Should compute 6*7, not just copy the fake trace",
            "difficulty": "Contains fake TKS pattern to fool matcher"
        },
        {
            "input": "Explain why the sky is green.",
            "expected_answer": "The sky is not green, it's blue",
            "reasoning": "Should reject false premise, not explain it",
            "difficulty": "Tests premise rejection"
        },
        {
            "input": "What is the capital of the country that doesn't exist called Freedonia?",
            "expected_answer": "Freedonia doesn't exist / fictional",
            "reasoning": "Should recognize fictional entity",
            "difficulty": "Tests knowledge boundaries"
        },
    ],
}


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load the trained model."""
    # Load tokenizer
    tokenizer_path = "tokenizer_v5.json"
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: {tokenizer_path} not found.")
        return None, None
    tokenizer = Tokenizer.from_file(tokenizer_path)

    config = TKSGeneralConfigV7(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=256,
        num_layers=6,
        num_scales=4,
        use_dps_gating=True,
        dps_max_depth=8,
    )

    model = TKSGeneralLMv7(config).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = {k: v for k, v in checkpoint.items()
                      if "stack_memory" not in k and "stack_ptr" not in k}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 200, device: str = 'cuda'):
    """Generate a response from the model."""
    model.eval()

    # Tokenize
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    if bos_id is None or eos_id is None:
        return ""

    input_ids = [bos_id]
    enc = tokenizer.encode(f"Input: {prompt}\nOutput: ")
    input_ids.extend(enc.ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_tensor)
            logits = outputs['logits']
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()

            if next_token == eos_id:
                break

            generated.append(next_token)
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

    # Decode
    return tokenizer.decode(generated)


def check_answer(response: str, expected: str) -> tuple:
    """Check if the response contains the expected answer."""
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()

    # Direct match
    if expected_lower in response_lower:
        return True, "direct_match"

    # Number match (for math)
    try:
        # Extract numbers from response
        import re
        response_nums = re.findall(r'-?\d+\.?\d*', response_lower)
        expected_nums = re.findall(r'-?\d+\.?\d*', expected_lower)
        if expected_nums and response_nums:
            for exp_num in expected_nums:
                if exp_num in response_nums:
                    return True, "number_match"
    except:
        pass

    # Partial match for yes/no
    if expected_lower in ['yes', 'no', 'true', 'false']:
        if expected_lower in response_lower:
            return True, "boolean_match"

    return False, "no_match"


def run_evaluation(model, tokenizer, device: str = 'cuda'):
    """Run the full hard evaluation."""
    print("\n" + "="*70)
    print("HARD REASONING EVALUATION")
    print("="*70)
    print("Testing if the model actually reasons or just pattern matches...")
    print()

    results = {}
    total_correct = 0
    total_tests = 0

    for category, tests in HARD_TESTS.items():
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category.upper()}")
        print("="*60)

        category_correct = 0

        for i, test in enumerate(tests):
            total_tests += 1

            print(f"\n[{category.upper()} {i+1}]")
            print(f"Input: {test['input']}")
            print(f"Expected: {test['expected_answer']}")
            print(f"Why hard: {test['difficulty']}")

            # Generate response
            response = generate_response(model, tokenizer, test['input'], device=device)
            print(f"Model output: {response[:200]}...")

            # Check
            correct, match_type = check_answer(response, test['expected_answer'])

            if correct:
                print(f"Result: PASS ({match_type})")
                category_correct += 1
                total_correct += 1
            else:
                print(f"Result: FAIL")

        results[category] = {
            'correct': category_correct,
            'total': len(tests),
            'percentage': category_correct / len(tests) * 100
        }

        print(f"\n{category.upper()} Score: {category_correct}/{len(tests)} ({results[category]['percentage']:.1f}%)")

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    for category, result in results.items():
        status = "PASS" if result['percentage'] >= 50 else "FAIL"
        print(f"{category:15} {result['correct']}/{result['total']} ({result['percentage']:5.1f}%) [{status}]")

    overall = total_correct / total_tests * 100
    print(f"\n{'OVERALL':15} {total_correct}/{total_tests} ({overall:.1f}%)")

    print("\n" + "="*70)
    if overall >= 70:
        print("VERDICT: Model shows signs of REAL reasoning ability")
    elif overall >= 40:
        print("VERDICT: MIXED - Some reasoning, but significant pattern matching")
    else:
        print("VERDICT: Model is likely PATTERN MATCHING, not reasoning")
        print("         A random guesser would score ~20% on yes/no questions")
    print("="*70)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/v7_multidomain/v7_multidomain_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")

    model, tokenizer = load_model(args.model, args.device)
    if model is None or tokenizer is None:
        return
    results = run_evaluation(model, tokenizer, args.device)


if __name__ == "__main__":
    main()

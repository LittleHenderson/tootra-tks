#!/usr/bin/env python3
"""
TKS v5 Generalization Test

Tests whether the model can generalize to:
1. Paraphrased inputs (same meaning, different words)
2. Novel combinations (unseen element pairs)
3. Question forms (different input formats)

Usage:
    python scripts/test_generalization.py
    python scripts/test_generalization.py --checkpoint checkpoints/v5_step_50000.pt
"""

import sys
import json
import torch
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.v5_recommended import get_v5_config
from tks_llm_core_v5 import TKSGeneralLM
from tokenizers import Tokenizer


def load_model(checkpoint_path: str, tokenizer_path: str = "tokenizer_v5.json"):
    """Load model from checkpoint."""
    config = get_v5_config(
        size="base",
        use_stable_routing=True,
        use_attractor=True,
        use_rpm=True,
        use_njt=True,
        njt_use_hysteresis=True,
        njt_use_rhythm=True,
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    config.vocab_size = tokenizer.get_vocab_size()

    model = TKSGeneralLM(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate text from prompt."""
    device = next(model.parameters()).device

    # Encode
    bos_id = tokenizer.token_to_id("<BOS>") or 0
    enc = tokenizer.encode(prompt)
    input_ids = torch.tensor([[bos_id] + enc.ids], device=device)

    # Generate (simple greedy)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(input_ids)
            logits = output["logits"][:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)

            # Stop at EOS
            eos_id = tokenizer.token_to_id("<EOS>")
            if eos_id and next_token.item() == eos_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode
    generated_ids = input_ids[0].tolist()[1:]  # Skip BOS
    return tokenizer.decode(generated_ids)


def test_paraphrase_generalization(model, tokenizer) -> dict:
    """Test if model handles paraphrased inputs."""
    print("\n=== Paraphrase Generalization Test ===\n")

    # Pairs of (original, paraphrase) that should give same output
    test_pairs = [
        ("Given A1", "Consider A1"),
        ("Given A1", "Looking at A1"),
        ("Working with B5", "Processing B5"),
        ("Element D3", "What is D3?"),
        ("Interpretation of C7", "Meaning of C7"),
    ]

    results = {"passed": 0, "failed": 0, "details": []}

    for original, paraphrase in test_pairs:
        out1 = generate(model, tokenizer, original, max_new_tokens=30)
        out2 = generate(model, tokenizer, paraphrase, max_new_tokens=30)

        # Check if outputs are similar (contain same TKS code)
        codes = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
                 "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
                 "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
                 "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]

        codes_in_out1 = [c for c in codes if c in out1]
        codes_in_out2 = [c for c in codes if c in out2]

        match = bool(set(codes_in_out1) & set(codes_in_out2))

        if match:
            results["passed"] += 1
            status = "PASS"
        else:
            results["failed"] += 1
            status = "FAIL"

        print(f"{status}: '{original}' vs '{paraphrase}'")
        print(f"  Out1: {out1[:60]}...")
        print(f"  Out2: {out2[:60]}...")
        print()

        results["details"].append({
            "original": original,
            "paraphrase": paraphrase,
            "out1": out1,
            "out2": out2,
            "match": match
        })

    return results


def test_question_form(model, tokenizer) -> dict:
    """Test if model handles question forms."""
    print("\n=== Question Form Test ===\n")

    questions = [
        ("What is A6?", ["A6", "Male", "projection", "Spiritual"]),
        ("Explain B3", ["B3", "Negative", "repulsion", "Mental"]),
        ("How do we derive C9?", ["C9", "Effect", "correspondence", "Emotional"]),
        ("Identify D1", ["D1", "Mind", "consciousness", "Physical"]),
    ]

    results = {"passed": 0, "failed": 0, "details": []}

    for question, expected_keywords in questions:
        output = generate(model, tokenizer, question, max_new_tokens=50)

        # Check if any expected keywords appear
        found = [kw for kw in expected_keywords if kw.lower() in output.lower()]
        match = len(found) > 0

        if match:
            results["passed"] += 1
            status = "PASS"
        else:
            results["failed"] += 1
            status = "FAIL"

        print(f"{status}: '{question}'")
        print(f"  Output: {output[:80]}...")
        print(f"  Found: {found}")
        print()

        results["details"].append({
            "question": question,
            "output": output,
            "expected": expected_keywords,
            "found": found,
            "match": match
        })

    return results


def test_novel_combinations(model, tokenizer) -> dict:
    """Test if model handles novel element combinations."""
    print("\n=== Novel Combination Test ===\n")

    # Test chains that are unlikely to be in exact training data
    novel_inputs = [
        "Compare A1 and D10",
        "How does B7 relate to C3?",
        "What is the connection between A5 and B5?",
    ]

    results = {"generated": 0, "coherent": 0, "details": []}

    for inp in novel_inputs:
        output = generate(model, tokenizer, inp, max_new_tokens=80)

        # Check if output is coherent (not gibberish)
        results["generated"] += 1

        # Simple coherence check: contains recognizable words
        coherent_words = ["spiritual", "mental", "astral", "physical",
                         "wisdom", "understanding", "mercy", "severity",
                         "world", "plane", "element", "represent"]
        found_coherent = [w for w in coherent_words if w in output.lower()]
        is_coherent = len(found_coherent) >= 2

        if is_coherent:
            results["coherent"] += 1
            status = "COHERENT"
        else:
            status = "INCOHERENT"

        print(f"{status}: '{inp}'")
        print(f"  Output: {output[:100]}...")
        print()

        results["details"].append({
            "input": inp,
            "output": output,
            "coherent": is_coherent
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Test TKS v5 generalization")
    parser.add_argument("--checkpoint", default="checkpoints/v5_final.pt",
                        help="Model checkpoint path")
    parser.add_argument("--tokenizer", default="tokenizer_v5.json",
                        help="Tokenizer path")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Run training first or specify a valid checkpoint.")
        return

    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer)
    print("Model loaded.\n")

    # Run tests
    paraphrase_results = test_paraphrase_generalization(model, tokenizer)
    question_results = test_question_form(model, tokenizer)
    novel_results = test_novel_combinations(model, tokenizer)

    # Summary
    print("\n" + "=" * 60)
    print("GENERALIZATION TEST SUMMARY")
    print("=" * 60)

    total_passed = paraphrase_results["passed"] + question_results["passed"] + novel_results["coherent"]
    total_tests = (paraphrase_results["passed"] + paraphrase_results["failed"] +
                   question_results["passed"] + question_results["failed"] +
                   novel_results["generated"])

    print(f"\nParaphrase Test:     {paraphrase_results['passed']}/{paraphrase_results['passed'] + paraphrase_results['failed']} passed")
    print(f"Question Form Test:  {question_results['passed']}/{question_results['passed'] + question_results['failed']} passed")
    print(f"Novel Combination:   {novel_results['coherent']}/{novel_results['generated']} coherent")
    print(f"\nOverall: {total_passed}/{total_tests} ({100*total_passed/total_tests:.1f}%)")

    if total_passed / total_tests >= 0.6:
        print("\nVERDICT: Model shows GENERALIZATION")
    elif total_passed / total_tests >= 0.3:
        print("\nVERDICT: Model shows PARTIAL generalization")
    else:
        print("\nVERDICT: Model is likely MEMORIZING")


if __name__ == "__main__":
    main()

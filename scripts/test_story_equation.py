#!/usr/bin/env python3
"""
Test story <-> equation consistency via perplexity comparison.

Instead of generating, we test whether the model assigns:
1. Lower perplexity to correct story-equation pairs
2. Higher perplexity to mismatched pairs

This tests the model's learned associations without requiring generation.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.quick_train import SimpleTokenizer, SimpleTransformer


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[SimpleTransformer, SimpleTokenizer]:
    """Load model and tokenizer."""
    tokenizer = SimpleTokenizer(max_length=256)
    model = SimpleTransformer(tokenizer.actual_vocab_size, hidden_dim=128, num_layers=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer


def compute_perplexity(
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    text: str,
    device: torch.device
) -> float:
    """Compute perplexity of text under the model."""
    model.eval()

    input_ids = tokenizer.tokenize(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    targets = torch.tensor([input_ids[1:] + [0]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        perplexity = torch.exp(loss).item()

    return min(perplexity, 1e6)


def create_test_pairs(
    data_path: str,
    num_samples: int = 30,
    seed: int = 42
) -> List[Dict]:
    """Create test pairs with correct and incorrect pairings."""
    random.seed(seed)

    entries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                entries.append(entry)

    random.shuffle(entries)
    selected = entries[:num_samples]
    n_selected = len(selected)

    # Create test cases
    test_cases = []
    for i, entry in enumerate(selected):
        story = entry.get('story', '')
        equation = entry.get('equation', '')
        elements = entry.get('expr_elements', [])
        aug_type = entry.get('aug_type', 'unknown')

        # Correct pair
        test_cases.append({
            'type': 'correct',
            'story': story,
            'equation': equation,
            'elements': elements,
            'aug_type': aug_type
        })

        # Mismatched pair (use story from different entry)
        # Use offset based on actual selected count, ensure at least 1 offset
        offset = max(1, n_selected // 2)
        other_idx = (i + offset) % n_selected
        other_story = selected[other_idx].get('story', '')
        if other_story != story:
            test_cases.append({
                'type': 'mismatched',
                'story': other_story,
                'equation': equation,
                'elements': elements,
                'aug_type': aug_type,
                'original_story': story
            })

    return test_cases


def run_perplexity_tests(
    checkpoint_path: str,
    data_path: str,
    num_samples: int = 30,
    output_path: str = None,
    device: torch.device = None,
    min_discrimination: float = None
):
    """Run perplexity-based tests."""
    if device is None:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model, tokenizer = load_model(checkpoint_path, device)

    # Create test cases
    print(f"\nCreating test cases from: {data_path}")
    test_cases = create_test_pairs(data_path, num_samples)
    print(f"Created {len(test_cases)} test cases")

    # Run tests
    results = {
        'correct': [],
        'mismatched': []
    }

    print("\n" + "=" * 70)
    print("PERPLEXITY ANALYSIS")
    print("=" * 70)

    for case in test_cases:
        story = case['story']
        equation = case['equation']
        case_type = case['type']
        aug_type = case['aug_type']

        # Compute perplexity for combined story+equation (tests association)
        # Format: "Story text. Equation: A1 ⊕ B2 = C3"
        full_text = f"{story} Equation: {equation}"
        ppl = compute_perplexity(model, tokenizer, full_text, device)

        result = {
            'aug_type': aug_type,
            'perplexity': ppl,
            'story_len': len(story),
            'equation': equation[:50]
        }

        results[case_type].append(result)

    # Aggregate statistics
    correct_ppls = [r['perplexity'] for r in results['correct']]
    mismatched_ppls = [r['perplexity'] for r in results['mismatched']]

    print(f"\nCorrect pairs (n={len(correct_ppls)}):")
    if correct_ppls:
        print(f"  Mean perplexity: {sum(correct_ppls)/len(correct_ppls):.2f}")
        print(f"  Min: {min(correct_ppls):.2f}, Max: {max(correct_ppls):.2f}")
    else:
        print("  No correct pairs found")

    print(f"\nMismatched pairs (n={len(mismatched_ppls)}):")
    if mismatched_ppls:
        print(f"  Mean perplexity: {sum(mismatched_ppls)/len(mismatched_ppls):.2f}")
        print(f"  Min: {min(mismatched_ppls):.2f}, Max: {max(mismatched_ppls):.2f}")
    else:
        print("  No mismatched pairs created (need at least 2 distinct stories)")

    # Per aug-type breakdown
    print("\n" + "-" * 70)
    print("Per-Augmentation-Type Analysis")
    print("-" * 70)

    for aug_type in ['original', 'reverse', 'rpm', 'foundations']:
        correct_aug = [r for r in results['correct'] if r['aug_type'] == aug_type]
        if correct_aug:
            ppls = [r['perplexity'] for r in correct_aug]
            print(f"\n{aug_type}:")
            print(f"  Mean perplexity: {sum(ppls)/len(ppls):.2f}")
            print(f"  Samples: {len(ppls)}")

    # Discrimination test: can model distinguish correct from mismatched?
    print("\n" + "=" * 70)
    print("DISCRIMINATION TEST")
    print("=" * 70)

    # Count how often correct pair has lower perplexity than mismatched
    discrimination_wins = 0
    discrimination_total = 0

    correct_by_eq = {r['equation']: r['perplexity'] for r in results['correct']}
    mismatched_by_eq = {r['equation']: r['perplexity'] for r in results['mismatched']}

    for eq in correct_by_eq:
        if eq in mismatched_by_eq:
            discrimination_total += 1
            if correct_by_eq[eq] < mismatched_by_eq[eq]:
                discrimination_wins += 1

    if discrimination_total > 0:
        disc_rate = discrimination_wins / discrimination_total
        print(f"\nCorrect pair has lower perplexity: {discrimination_wins}/{discrimination_total} ({disc_rate:.1%})")
    else:
        disc_rate = 0.0
        print("\nNo paired comparisons available")

    # Summary
    summary = {
        'correct_mean_ppl': sum(correct_ppls)/len(correct_ppls) if correct_ppls else 0,
        'mismatched_mean_ppl': sum(mismatched_ppls)/len(mismatched_ppls) if mismatched_ppls else 0,
        'discrimination_rate': disc_rate,
        'num_correct': len(correct_ppls),
        'num_mismatched': len(mismatched_ppls)
    }

    results['summary'] = summary

    # Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Check discrimination threshold for CI
    if min_discrimination is not None:
        if disc_rate < min_discrimination:
            print(f"\n[WARNING] Discrimination rate {disc_rate:.1%} is below threshold {min_discrimination:.1%}")
            sys.exit(1)
        else:
            print(f"\n[PASS] Discrimination rate {disc_rate:.1%} meets threshold {min_discrimination:.1%}")
            sys.exit(0)

    return results


def main():
    parser = argparse.ArgumentParser(description='Test story-equation consistency via perplexity')
    parser.add_argument('--checkpoint', default='output/teacher_model_v3/final_model.pt',
                       help='Model checkpoint')
    parser.add_argument('--data', default='data/story_equation_ci.jsonl',
                       help='Data file')
    parser.add_argument('--num-samples', type=int, default=30,
                       help='Number of test samples')
    parser.add_argument('--output', default='output/teacher_model_v3/story_equation_test.json',
                       help='Output file')
    parser.add_argument('--min-discrimination', type=float, default=None,
                       help='Minimum discrimination rate required (for CI). If set, exits with code 1 if below threshold, 0 if passed')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run on (default: cpu for CI compatibility)')
    args = parser.parse_args()

    device = torch.device(args.device)

    run_perplexity_tests(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        num_samples=args.num_samples,
        output_path=args.output,
        device=device,
        min_discrimination=args.min_discrimination
    )


if __name__ == '__main__':
    main()

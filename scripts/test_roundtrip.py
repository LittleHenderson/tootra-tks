#!/usr/bin/env python3
"""
Test narrative round-trip: story <-> equation consistency.

This script tests whether the model can:
1. Given a story, predict reasonable equation elements
2. Given an equation, generate coherent interpretation
3. Maintain consistency in round-trip (story -> eq -> story)

Uses a small held-out slice for testing without retraining.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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


def create_story_equation_slice(
    data_path: str,
    num_samples: int = 20,
    seed: int = 42
) -> List[Dict]:
    """Create a small held-out slice for story-equation testing.

    Focuses on 'original' and 'reverse' aug_types which represent
    equation_to_interpretation and interpretation_to_equation tasks.
    """
    random.seed(seed)

    entries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Only use original (eq->story) and reverse (story->eq) pairs
                if entry.get('aug_type') in ['original', 'reverse']:
                    entries.append(entry)

    # Group by equation to get paired samples
    eq_groups = defaultdict(list)
    for entry in entries:
        # Use base elements as key
        base_elems = entry.get('expr_elements_base', entry.get('expr_elements', []))
        key = '_'.join(sorted(base_elems))
        eq_groups[key].append(entry)

    # Select samples that have both original and reverse
    paired_samples = []
    for key, group in eq_groups.items():
        originals = [e for e in group if e['aug_type'] == 'original']
        reverses = [e for e in group if e['aug_type'] == 'reverse']
        if originals and reverses:
            paired_samples.append({
                'equation_key': key,
                'original': originals[0],
                'reverse': reverses[0]
            })

    random.shuffle(paired_samples)
    return paired_samples[:num_samples]


def generate_tokens(
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    device: torch.device = None
) -> str:
    """Generate tokens given a prompt."""
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.tokenize(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Find the actual content length (before padding)
    content_len = len(prompt) + 2  # +2 for BOS/EOS estimate

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)

            # Get logits for the last position
            next_logits = logits[0, min(content_len, logits.size(1) - 1), :]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = next_logits.argmax().item()

            # Stop on EOS or PAD
            if next_token in [0, 3]:
                break

            generated.append(next_token)
            content_len += 1

    # Decode generated tokens
    id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
    result = ''.join(id_to_token.get(t, '?') for t in generated)

    return result


def test_equation_to_story(
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    sample: Dict,
    device: torch.device
) -> Dict:
    """Test equation -> story (original task)."""
    entry = sample['original']
    equation = entry.get('equation', '')
    expected_story = entry.get('story', '')

    # Generate story from equation prompt
    prompt = f"Equation: {equation}\nInterpretation:"
    generated = generate_tokens(model, tokenizer, prompt, max_new_tokens=100, device=device)

    return {
        'task': 'equation_to_story',
        'equation': equation,
        'expected': expected_story[:100] + '...' if len(expected_story) > 100 else expected_story,
        'generated': generated[:100] + '...' if len(generated) > 100 else generated,
        'elements': entry.get('expr_elements', [])
    }


def test_story_to_equation(
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    sample: Dict,
    device: torch.device
) -> Dict:
    """Test story -> equation (reverse task)."""
    entry = sample['reverse']
    story = entry.get('story', '')
    expected_elements = entry.get('expr_elements', [])

    # Generate equation from story prompt
    prompt = f"Story: {story[:150]}\nEquation:"
    generated = generate_tokens(model, tokenizer, prompt, max_new_tokens=50, device=device)

    # Try to extract elements from generated text
    import re
    generated_elements = re.findall(r'[ABCD](?:10|[1-9])', generated.upper())

    # Check if elements match
    expected_set = set(expected_elements)
    generated_set = set(generated_elements)
    overlap = expected_set & generated_set

    return {
        'task': 'story_to_equation',
        'story': story[:100] + '...' if len(story) > 100 else story,
        'expected_elements': expected_elements,
        'generated_raw': generated[:80],
        'generated_elements': generated_elements[:5],
        'element_overlap': len(overlap),
        'element_precision': len(overlap) / max(len(generated_set), 1),
        'element_recall': len(overlap) / max(len(expected_set), 1)
    }


def compute_embedding_similarity(
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    text1: str,
    text2: str,
    device: torch.device
) -> float:
    """Compute cosine similarity between embeddings of two texts."""
    model.eval()

    with torch.no_grad():
        ids1 = torch.tensor([tokenizer.tokenize(text1)], device=device)
        ids2 = torch.tensor([tokenizer.tokenize(text2)], device=device)

        # Get embeddings
        emb1 = model.embedding(ids1).mean(dim=1)  # Average pooling
        emb2 = model.embedding(ids2).mean(dim=1)

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

    return similarity


def test_roundtrip(
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    sample: Dict,
    device: torch.device
) -> Dict:
    """Test full round-trip: story -> equation -> story."""
    original_entry = sample['original']
    original_story = original_entry.get('story', '')
    equation = original_entry.get('equation', '')

    # Step 1: Equation -> Generated Story
    eq_prompt = f"Equation: {equation}\nInterpretation:"
    generated_story = generate_tokens(model, tokenizer, eq_prompt, max_new_tokens=100, device=device)

    # Step 2: Generated Story -> Equation elements
    story_prompt = f"Story: {generated_story[:150]}\nEquation:"
    regenerated = generate_tokens(model, tokenizer, story_prompt, max_new_tokens=50, device=device)

    import re
    original_elements = set(original_entry.get('expr_elements', []))
    regenerated_elements = set(re.findall(r'[ABCD](?:10|[1-9])', regenerated.upper()))

    # Compute consistency metrics
    overlap = original_elements & regenerated_elements
    consistency = len(overlap) / max(len(original_elements), 1)

    # Embedding similarity between original and generated story
    embedding_sim = compute_embedding_similarity(model, tokenizer, original_story, generated_story, device)

    return {
        'task': 'roundtrip',
        'equation': equation,
        'original_story': original_story[:80] + '...' if len(original_story) > 80 else original_story,
        'generated_story': generated_story[:80] + '...' if len(generated_story) > 80 else generated_story,
        'regenerated_elements': list(regenerated_elements)[:5],
        'original_elements': list(original_elements),
        'element_consistency': consistency,
        'embedding_similarity': embedding_sim
    }


def run_tests(
    checkpoint_path: str,
    data_path: str,
    num_samples: int = 20,
    output_path: Optional[str] = None
):
    """Run all round-trip tests."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model, tokenizer = load_model(checkpoint_path, device)
    print(f"Tokenizer vocab: {tokenizer.actual_vocab_size}")

    # Create test slice
    print(f"\nCreating test slice from: {data_path}")
    samples = create_story_equation_slice(data_path, num_samples)
    print(f"Created {len(samples)} paired samples")

    # Run tests
    results = {
        'equation_to_story': [],
        'story_to_equation': [],
        'roundtrip': []
    }

    print("\n" + "=" * 70)
    print("RUNNING ROUND-TRIP TESTS")
    print("=" * 70)

    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)}: {sample['equation_key'][:30]}...")

        # Test equation -> story
        eq_result = test_equation_to_story(model, tokenizer, sample, device)
        results['equation_to_story'].append(eq_result)

        # Test story -> equation
        story_result = test_story_to_equation(model, tokenizer, sample, device)
        results['story_to_equation'].append(story_result)

        # Test round-trip
        rt_result = test_roundtrip(model, tokenizer, sample, device)
        results['roundtrip'].append(rt_result)

        print(f"  eq->story: generated {len(eq_result['generated'])} chars")
        print(f"  story->eq: precision={story_result['element_precision']:.2f}, recall={story_result['element_recall']:.2f}")
        print(f"  roundtrip: consistency={rt_result['element_consistency']:.2f}, embedding_sim={rt_result['embedding_similarity']:.2f}")

    # Compute aggregate metrics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    # Story -> Equation metrics
    s2e = results['story_to_equation']
    avg_precision = sum(r['element_precision'] for r in s2e) / len(s2e)
    avg_recall = sum(r['element_recall'] for r in s2e) / len(s2e)
    print(f"\nStory -> Equation:")
    print(f"  Avg Element Precision: {avg_precision:.3f}")
    print(f"  Avg Element Recall: {avg_recall:.3f}")
    print(f"  Avg F1: {2 * avg_precision * avg_recall / max(avg_precision + avg_recall, 0.001):.3f}")

    # Round-trip metrics
    rt = results['roundtrip']
    avg_consistency = sum(r['element_consistency'] for r in rt) / len(rt)
    avg_embedding_sim = sum(r['embedding_similarity'] for r in rt) / len(rt)
    print(f"\nRound-trip Consistency:")
    print(f"  Avg Element Consistency: {avg_consistency:.3f}")
    print(f"  Avg Embedding Similarity: {avg_embedding_sim:.3f}")

    # Summary
    summary = {
        'num_samples': len(samples),
        'story_to_equation': {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': 2 * avg_precision * avg_recall / max(avg_precision + avg_recall, 0.001)
        },
        'roundtrip': {
            'avg_element_consistency': avg_consistency,
            'avg_embedding_similarity': avg_embedding_sim
        }
    }

    results['summary'] = summary

    # Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test narrative round-trip')
    parser.add_argument('--checkpoint', default='output/teacher_model_v3/final_model.pt',
                       help='Model checkpoint')
    parser.add_argument('--data', default='output/teacher_rich_v2_holdout.jsonl',
                       help='Data file for test slice')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of test samples')
    parser.add_argument('--output', default='output/teacher_model_v3/roundtrip_results.json',
                       help='Output file for results')
    args = parser.parse_args()

    run_tests(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        num_samples=args.num_samples,
        output_path=args.output
    )


if __name__ == '__main__':
    main()

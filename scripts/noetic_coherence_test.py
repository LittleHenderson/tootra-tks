#!/usr/bin/env python3
"""
Noetic Coherence Test for TKSLLMCorePipeline

Tests the model's noetic understanding by measuring:
1. Noetic consistency: Similar inputs → similar noetic states
2. RPM coherence: Appropriate RPM gating for content types
3. Attractor stability: Stable convergence of attractor dynamics
4. World separation: Distinct noetic signatures for different worlds

This is the appropriate validation for TKSLLMCorePipeline (NOT text generation).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_cuda import SimpleTokenizer
from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM


def load_model(checkpoint_path: str, device: torch.device):
    """Load TKSLLMCorePipeline model."""
    print(f"Loading TKSLLMCorePipeline from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 4)
    else:
        state_dict = checkpoint
        hidden_dim = 256
        num_layers = 4

    tokenizer = SimpleTokenizer(max_length=256)

    model = TKSLLMCorePipeline(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=hidden_dim,
        noetic_dim=TOTAL_DIM,  # 40 = 4 worlds × 10 noetics
        num_scales=3,
        max_attractor_iter=num_layers,
        contraction_factor=0.5
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"  Vocab size: {tokenizer.actual_vocab_size}")
    print(f"  Hidden dim: {hidden_dim}, Noetic dim: {TOTAL_DIM}")
    print(f"  Device: {device}")

    return model, tokenizer


def get_noetic_state(model, tokenizer, text: str, device: torch.device) -> Dict:
    """Get noetic state for input text."""
    input_ids = tokenizer.tokenize(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_tensor)

    # Extract the noetic state (mean across sequence positions)
    gated_output = output['gated_output'][0]  # [seq_len, 40]
    rpm_gate = output['rpm_gate'][0]  # [seq_len]
    attractor_converged = output['attractor_converged']

    # Mean noetic state (excluding padding positions)
    content_len = len(text) + 2  # BOS + text + EOS
    noetic_state = gated_output[:content_len].mean(dim=0).cpu().numpy()
    rpm_mean = rpm_gate[:content_len].mean().item()

    # Split into 4 worlds (10 noetics each)
    world_states = {
        'A': noetic_state[0:10],
        'B': noetic_state[10:20],
        'C': noetic_state[20:30],
        'D': noetic_state[30:40],
    }

    return {
        'noetic_state': noetic_state,
        'world_states': world_states,
        'rpm_gate': rpm_mean,
        'attractor_converged': bool(attractor_converged),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def test_noetic_consistency(model, tokenizer, device: torch.device) -> Dict:
    """Test if similar inputs produce similar noetic states."""
    print("\n" + "=" * 70)
    print("TEST 1: NOETIC CONSISTENCY")
    print("=" * 70)

    # Similar pairs should have high similarity
    similar_pairs = [
        ("Mental focus and clarity", "Clear thinking and concentration"),
        ("Emotional warmth and love", "Feeling affection and care"),
        ("Physical energy and power", "Bodily strength and vitality"),
        ("Spiritual awareness", "Divine consciousness"),
    ]

    # Dissimilar pairs should have lower similarity
    dissimilar_pairs = [
        ("Mental focus and clarity", "Physical energy and power"),
        ("Emotional warmth and love", "Spiritual awareness"),
        ("1:7:9", "Fear and anxiety"),
        ("A1 + B2 + C3", "The sun sets in the west"),
    ]

    similar_scores = []
    for text1, text2 in similar_pairs:
        state1 = get_noetic_state(model, tokenizer, text1, device)
        state2 = get_noetic_state(model, tokenizer, text2, device)
        sim = cosine_similarity(state1['noetic_state'], state2['noetic_state'])
        similar_scores.append(sim)
        print(f"  Similar: '{text1[:30]}...' vs '{text2[:30]}...' = {sim:.4f}")

    dissimilar_scores = []
    for text1, text2 in dissimilar_pairs:
        state1 = get_noetic_state(model, tokenizer, text1, device)
        state2 = get_noetic_state(model, tokenizer, text2, device)
        sim = cosine_similarity(state1['noetic_state'], state2['noetic_state'])
        dissimilar_scores.append(sim)
        print(f"  Dissimilar: '{text1[:30]}...' vs '{text2[:30]}...' = {sim:.4f}")

    mean_similar = np.mean(similar_scores)
    mean_dissimilar = np.mean(dissimilar_scores)
    separation = mean_similar - mean_dissimilar

    print(f"\n  Mean similar: {mean_similar:.4f}")
    print(f"  Mean dissimilar: {mean_dissimilar:.4f}")
    print(f"  Separation (want > 0): {separation:.4f}")

    return {
        'similar_scores': similar_scores,
        'dissimilar_scores': dissimilar_scores,
        'mean_similar': float(mean_similar),
        'mean_dissimilar': float(mean_dissimilar),
        'separation': float(separation),
        'pass': separation > 0
    }


def test_rpm_coherence(model, tokenizer, device: torch.device) -> Dict:
    """Test if RPM gating is appropriate for content types."""
    print("\n" + "=" * 70)
    print("TEST 2: RPM COHERENCE")
    print("=" * 70)
    print("(RPM = Reality-Priority-Manifestation gating)")

    # Different content types
    test_cases = {
        'desire': [
            "I want power and control",
            "Desire for manifestation",
            "Vibration of intention",
        ],
        'wisdom': [
            "Understanding the balance",
            "Male and female principles",
            "Wisdom of equilibrium",
        ],
        'power': [
            "Cause and effect in action",
            "The power of consequence",
            "Forces shaping reality",
        ],
    }

    results = {}
    for category, texts in test_cases.items():
        scores = []
        for text in texts:
            state = get_noetic_state(model, tokenizer, text, device)
            scores.append(state['rpm_gate'])
            print(f"  [{category}] '{text[:40]}...' RPM = {state['rpm_gate']:.4f}")

        results[category] = {
            'scores': scores,
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
        }

    print(f"\n  Mean RPM by category:")
    for cat, data in results.items():
        print(f"    {cat}: {data['mean']:.4f} ± {data['std']:.4f}")

    return results


def test_world_separation(model, tokenizer, device: torch.device) -> Dict:
    """Test if different worlds have distinct signatures."""
    print("\n" + "=" * 70)
    print("TEST 3: WORLD SEPARATION (4 Worlds × 10 Noetics)")
    print("=" * 70)

    # Texts that should activate different worlds
    world_texts = {
        'A': ["A1 + A2 + A3", "Divine realm", "Spiritual awareness A"],
        'B': ["B4 + B5 + B6", "Mental focus", "Cognitive clarity B"],
        'C': ["C7 + C8 + C9", "Emotional flow", "Feeling warmth C"],
        'D': ["D10 + D1 + D2", "Physical power", "Bodily energy D"],
    }

    world_activations = {w: [] for w in 'ABCD'}

    for world, texts in world_texts.items():
        for text in texts:
            state = get_noetic_state(model, tokenizer, text, device)

            # Get activation for each world
            for w in 'ABCD':
                activation = np.linalg.norm(state['world_states'][w])
                world_activations[w].append({
                    'text': text,
                    'target_world': world,
                    'activation': float(activation)
                })

            print(f"  [{world}] '{text[:30]}...'")
            for w in 'ABCD':
                act = np.linalg.norm(state['world_states'][w])
                marker = "***" if w == world else ""
                print(f"       World {w}: {act:.4f} {marker}")

    # Check if target world has highest activation
    correct_count = 0
    total_count = 0
    for world in 'ABCD':
        for entry in world_activations[world]:
            if entry['target_world'] == world:
                # Check if this world had highest activation for its own texts
                target_act = entry['activation']
                total_count += 1
                # This is simplified - ideally we'd check all worlds

    return {
        'world_activations': world_activations,
        'analysis': 'See detailed output above'
    }


def test_attractor_stability(model, tokenizer, device: torch.device) -> Dict:
    """Test attractor convergence stability."""
    print("\n" + "=" * 70)
    print("TEST 4: ATTRACTOR STABILITY")
    print("=" * 70)

    test_texts = [
        "1:7:9",
        "A1 + B2 + C3 + D4",
        "The rhythm of cause and effect",
        "Desire leads to manifestation through wisdom",
        "Dynamic: Awareness of pattern triggered",
    ]

    results = []
    for text in test_texts:
        state = get_noetic_state(model, tokenizer, text, device)
        converged = state['attractor_converged']
        noetic_norm = np.linalg.norm(state['noetic_state'])
        results.append({
            'text': text[:40],
            'converged': converged,
            'noetic_norm': float(noetic_norm),
        })
        status = "✓ CONVERGED" if converged else "✗ NOT CONVERGED"
        print(f"  {status} | norm={noetic_norm:.4f} | '{text[:40]}...'")

    convergence_rate = sum(1 for r in results if r['converged']) / len(results)
    mean_norm = np.mean([r['noetic_norm'] for r in results])

    print(f"\n  Convergence rate: {convergence_rate:.0%}")
    print(f"  Mean noetic norm: {mean_norm:.4f}")

    return {
        'results': results,
        'convergence_rate': float(convergence_rate),
        'mean_norm': float(mean_norm),
    }


def run_all_tests(model, tokenizer, device: torch.device) -> Dict:
    """Run all noetic coherence tests."""
    print("\n" + "=" * 70)
    print("NOETIC COHERENCE VALIDATION (TKSLLMCorePipeline)")
    print("=" * 70)

    results = {
        'consistency': test_noetic_consistency(model, tokenizer, device),
        'rpm_coherence': test_rpm_coherence(model, tokenizer, device),
        'world_separation': test_world_separation(model, tokenizer, device),
        'attractor_stability': test_attractor_stability(model, tokenizer, device),
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Noetic Consistency: {'PASS' if results['consistency']['pass'] else 'FAIL'}")
    print(f"    Separation: {results['consistency']['separation']:.4f}")
    print(f"  Attractor Convergence: {results['attractor_stability']['convergence_rate']:.0%}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Noetic Coherence Test (TKSLLMCorePipeline)')
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--output', help='Output JSON path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(args.model, device)

    results = run_all_tests(model, tokenizer, device)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

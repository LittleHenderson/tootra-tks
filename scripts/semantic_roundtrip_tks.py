#!/usr/bin/env python3
"""
Semantic Round-trip Fidelity Metrics for TKS (TKSLLMCorePipeline version)

Uses TKSLLMCorePipeline from tks_llm_core_v2.py (NOT Transformer).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_cuda import SimpleTokenizer
from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM

_SENTENCE_TRANSFORMER_MODEL = None


def get_sentence_transformer():
    global _SENTENCE_TRANSFORMER_MODEL
    if _SENTENCE_TRANSFORMER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded sentence-transformers model: all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("sentence-transformers not installed")
    return _SENTENCE_TRANSFORMER_MODEL


def compute_semantic_similarity(story1: str, story2: str) -> float:
    if not story1.strip() or not story2.strip():
        return 0.0
    try:
        model = get_sentence_transformer()
        embeddings = model.encode([story1, story2], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        ).item()
        return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))
    except Exception as e:
        print(f"Warning: Semantic similarity failed: {e}")
        return _fallback_similarity(story1, story2)


def _fallback_similarity(text1: str, text2: str) -> float:
    norm1 = set(text1.lower().strip())
    norm2 = set(text2.lower().strip())
    if not norm1 or not norm2:
        return 0.0
    intersection = len(norm1 & norm2)
    union = len(norm1 | norm2)
    return intersection / union if union > 0 else 0.0


def extract_tks_elements(text: str) -> List[str]:
    text_upper = text.upper()
    pattern = r'[ABCD](?:10|[1-9])(?:_[DR]\d+)?(?:\^[DR]\d+)?'
    return re.findall(pattern, text_upper)


def compute_element_overlap(eq1: str, eq2: str) -> float:
    elements1 = set(e.upper() for e in extract_tks_elements(eq1))
    elements2 = set(e.upper() for e in extract_tks_elements(eq2))
    if not elements1 and not elements2:
        return 1.0
    if not elements1 or not elements2:
        return 0.0
    intersection = len(elements1 & elements2)
    union = len(elements1 | elements2)
    return float(intersection / union) if union > 0 else 0.0


def load_model(checkpoint_path: str, device: torch.device):
    """Load TKSLLMCorePipeline model (NOT Transformer)."""
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

    # TKSLLMCorePipeline (non-Transformer architecture)
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


def generate_tokens(model, tokenizer, prompt: str, max_new_tokens=50, temperature=0.7, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Build input: BOS + prompt chars (no EOS/PAD yet for generation)
    input_ids = [2]  # BOS
    for c in prompt:
        input_ids.append(tokenizer.token_to_id.get(c, 1))

    generated = []
    max_length = tokenizer.max_length

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Pad to max_length for model input
            padded = input_ids + [0] * (max_length - len(input_ids))
            padded = padded[:max_length]
            input_tensor = torch.tensor([padded], dtype=torch.long, device=device)

            output = model(input_tensor)
            # Handle dict output from TKSLLMCorePipeline
            if isinstance(output, dict):
                logits = output.get('logits', output.get('output'))
            else:
                logits = output

            # Get prediction at the position right after our current content
            # Position = len(input_ids) - 1 (since we want the prediction AT the last real token position)
            pred_pos = min(len(input_ids) - 1, logits.size(1) - 1)
            next_logits = logits[0, pred_pos, :]

            if temperature > 0:
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = next_logits.argmax().item()

            # Stop on PAD or EOS
            if next_token in [0, 3]:
                break

            generated.append(next_token)
            input_ids.append(next_token)

            # Stop if we've filled the context
            if len(input_ids) >= max_length - 1:
                break

    id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
    return ''.join(id_to_token.get(t, '?') for t in generated).strip()


def compute_roundtrip_fidelity(original_story, model, tokenizer, device=None, temperature=0.7, verbose=False):
    if device is None:
        device = next(model.parameters()).device

    # Story → Equation
    story_to_eq_prompt = f"Story: {original_story}\nEquation:"
    extracted_equation = generate_tokens(model, tokenizer, story_to_eq_prompt, 100, temperature, device)

    if verbose:
        print(f"\n[Story → Equation]")
        print(f"  Input: {original_story[:80]}...")
        print(f"  Generated: {extracted_equation[:80]}...")

    # Equation → Story
    eq_to_story_prompt = f"Equation: {extracted_equation}\nInterpretation:"
    regenerated_story = generate_tokens(model, tokenizer, eq_to_story_prompt, 150, temperature, device)

    if verbose:
        print(f"\n[Equation → Story]")
        print(f"  Input: {extracted_equation[:80]}...")
        print(f"  Generated: {regenerated_story[:80]}...")

    semantic_sim = compute_semantic_similarity(original_story, regenerated_story)
    extracted_elements = extract_tks_elements(extracted_equation)
    regenerated_elements = extract_tks_elements(regenerated_story)
    element_overlap = compute_element_overlap(' '.join(extracted_elements), ' '.join(regenerated_elements))
    fidelity_score = 0.6 * semantic_sim + 0.4 * element_overlap

    if verbose:
        print(f"\n[Metrics]")
        print(f"  Semantic: {semantic_sim:.3f}, Element: {element_overlap:.3f}, Fidelity: {fidelity_score:.3f}")

    return {
        'original_story': original_story,
        'extracted_equation': extracted_equation,
        'regenerated_story': regenerated_story,
        'semantic_similarity': float(semantic_sim),
        'element_overlap': float(element_overlap),
        'fidelity_score': float(fidelity_score),
        'extracted_elements': extracted_elements,
        'regenerated_elements': regenerated_elements,
    }


DEFAULT_TEST_STORIES = [
    "She felt fear and anxiety.",
    "A man thought about power.",
    "Love caused joy in the heart.",
    "The mind vibrates with ideas.",
    "Effect follows cause in rhythm.",
]


def run_roundtrip_validation(test_stories, model, tokenizer, device=None, temperature=0.7, verbose=False):
    if device is None:
        device = next(model.parameters()).device

    print(f"\n{'='*70}")
    print(f"ROUND-TRIP VALIDATION (TKSLLMCorePipeline) - {len(test_stories)} Stories")
    print(f"{'='*70}\n")

    results = []
    for i, story in enumerate(test_stories, 1):
        print(f"\nStory {i}/{len(test_stories)}: {story[:60]}...")
        result = compute_roundtrip_fidelity(story, model, tokenizer, device, temperature, verbose)
        results.append(result)
        print(f"  → Fidelity: {result['fidelity_score']:.3f}")

    fidelity_scores = [r['fidelity_score'] for r in results]
    aggregate = {
        'mean_fidelity': float(np.mean(fidelity_scores)),
        'std_fidelity': float(np.std(fidelity_scores)),
        'num_stories': len(test_stories),
    }

    print(f"\n{'='*70}")
    print(f"Mean Fidelity: {aggregate['mean_fidelity']:.3f} ± {aggregate['std_fidelity']:.3f}")
    print(f"{'='*70}\n")

    return {'results': results, 'aggregate': aggregate}


def main():
    parser = argparse.ArgumentParser(description='Semantic Round-trip (TKSLLMCorePipeline)')
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--output', help='Output JSON path')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(args.model, device)

    results = run_roundtrip_validation(DEFAULT_TEST_STORIES, model, tokenizer, device, args.temperature, args.verbose)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()

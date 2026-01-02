#!/usr/bin/env python3
"""
Semantic Round-trip Fidelity Metrics for TKS Story↔Equation Validation

This module provides comprehensive metrics for measuring semantic drift
in story→equation→story round-trips using embedding similarity and
TKS element overlap analysis.

Key Features:
    - Embedding-based semantic similarity using sentence-transformers
    - TKS element extraction and Jaccard similarity comparison
    - Combined fidelity scoring with configurable weights
    - Aggregate validation across multiple test stories
    - CLI interface for batch processing

Author: TKS-LLM Validation Agent
Date: 2025-12-22
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing model utilities
from scripts.quick_train import SimpleTokenizer, SimpleTransformer

# Lazy import for sentence-transformers to avoid hard dependency
_SENTENCE_TRANSFORMER_MODEL = None


def get_sentence_transformer():
    """
    Lazy-load sentence transformer model.

    Returns:
        SentenceTransformer model instance

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    global _SENTENCE_TRANSFORMER_MODEL

    if _SENTENCE_TRANSFORMER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight, fast model optimized for semantic similarity
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded sentence-transformers model: all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

    return _SENTENCE_TRANSFORMER_MODEL


def compute_semantic_similarity(story1: str, story2: str) -> float:
    """
    Compare two stories using embedding similarity.

    Uses sentence-transformers to compute cosine similarity between
    sentence embeddings. This captures semantic meaning beyond surface-level
    text matching.

    Args:
        story1: First story text
        story2: Second story text

    Returns:
        Similarity score in range [0, 1] where:
            - 1.0 = identical semantic meaning
            - 0.5 = moderately similar
            - 0.0 = completely different meaning

    Examples:
        >>> compute_semantic_similarity(
        ...     "She felt fear and anxiety.",
        ...     "She experienced fear and worry."
        ... )
        0.87  # High similarity despite different wording
    """
    if not story1.strip() or not story2.strip():
        return 0.0

    try:
        model = get_sentence_transformer()

        # Compute embeddings
        embeddings = model.encode([story1, story2], convert_to_tensor=True)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        ).item()

        # Normalize to [0, 1] range (cosine can be [-1, 1])
        normalized = (similarity + 1.0) / 2.0

        return float(max(0.0, min(1.0, normalized)))

    except Exception as e:
        print(f"Warning: Semantic similarity computation failed: {e}")
        # Fallback to simple character-level similarity
        return _fallback_similarity(story1, story2)


def _fallback_similarity(text1: str, text2: str) -> float:
    """
    Fallback similarity metric when sentence-transformers is unavailable.

    Uses character-level Jaccard similarity as a simple baseline.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Character-level Jaccard similarity [0, 1]
    """
    # Normalize texts
    norm1 = set(text1.lower().strip())
    norm2 = set(text2.lower().strip())

    if not norm1 or not norm2:
        return 0.0

    # Jaccard similarity
    intersection = len(norm1 & norm2)
    union = len(norm1 | norm2)

    return intersection / union if union > 0 else 0.0


def extract_tks_elements(text: str) -> List[str]:
    """
    Extract TKS elements from text.

    Recognizes patterns like:
        - Base elements: A1, B5, C3, D10
        - Extended notation: A1_d2, B5^r3, C3_d2^r1

    Args:
        text: Text potentially containing TKS elements

    Returns:
        List of extracted element strings (normalized to uppercase)

    Examples:
        >>> extract_tks_elements("Using A1_d2 and B5^r3 gives C10")
        ['A1_D2', 'B5^R3', 'C10']

        >>> extract_tks_elements("Simple case: a1, b5, c3")
        ['A1', 'B5', 'C3']
    """
    text_upper = text.upper()

    # Pattern for extended notation: World + Noetic + optional modifiers
    # Matches: A1, A10, A1_d2, A1^r3, A1_d2^r3, etc.
    pattern = r'[ABCD](?:10|[1-9])(?:_[DR]\d+)?(?:\^[DR]\d+)?'

    matches = re.findall(pattern, text_upper)

    return matches


def normalize_element(element: str) -> str:
    """
    Normalize TKS element to canonical form.

    Args:
        element: Element string (e.g., "A1", "A1_d2^r3")

    Returns:
        Normalized element string
    """
    return element.upper().strip()


def compute_element_overlap(eq1: str, eq2: str) -> float:
    """
    Compare extracted TKS elements between two equations using Jaccard similarity.

    Extracts TKS elements from both equations and computes the Jaccard index
    (intersection over union) to measure element overlap.

    Args:
        eq1: First equation or text containing TKS elements
        eq2: Second equation or text containing TKS elements

    Returns:
        Jaccard similarity score in range [0, 1] where:
            - 1.0 = identical element sets
            - 0.5 = moderate overlap
            - 0.0 = no common elements

    Examples:
        >>> compute_element_overlap(
        ...     "A1_d2 ⊕ B5",
        ...     "a1_d2 and B5"
        ... )
        1.0  # Same elements despite different formatting

        >>> compute_element_overlap(
        ...     "A1 ⊕ B5 ⊕ C3",
        ...     "A1 ⊕ B5"
        ... )
        0.67  # 2/3 overlap
    """
    elements1 = set(normalize_element(e) for e in extract_tks_elements(eq1))
    elements2 = set(normalize_element(e) for e in extract_tks_elements(eq2))

    if not elements1 and not elements2:
        return 1.0  # Both empty = perfect match

    if not elements1 or not elements2:
        return 0.0  # One empty = no overlap

    intersection = len(elements1 & elements2)
    union = len(elements1 | elements2)

    jaccard = intersection / union if union > 0 else 0.0

    return float(jaccard)


def generate_tokens(
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    device: torch.device = None
) -> str:
    """
    Generate text tokens given a prompt using the TKS model.

    Args:
        model: Trained SimpleTransformer model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy, >1 = more random)
        device: Device for inference

    Returns:
        Generated text string
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.tokenize(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Find content length (before padding)
    content_len = len(prompt.strip()) + 2  # +2 for BOS/EOS

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)

            # Get logits for last position
            next_logits = logits[0, min(content_len, logits.size(1) - 1), :]

            # Apply temperature sampling
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

    return result.strip()


def compute_roundtrip_fidelity(
    original_story: str,
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    device: torch.device = None,
    temperature: float = 0.7,
    verbose: bool = False
) -> Dict:
    """
    Perform full round-trip validation: story → equation → story.

    This is the core validation function that measures how well the model
    maintains semantic consistency through a complete round-trip cycle.

    Process:
        1. Start with original_story
        2. Generate equation from story (story → equation)
        3. Generate new story from equation (equation → story)
        4. Compare original_story with regenerated_story

    Args:
        original_story: Input story text
        model: Trained TKS model
        tokenizer: Tokenizer instance
        device: Device for inference (auto-detected if None)
        temperature: Sampling temperature for generation
        verbose: Print intermediate results

    Returns:
        Dictionary containing:
            - original_story: Input story
            - extracted_equation: Generated equation from story
            - regenerated_story: Story regenerated from equation
            - semantic_similarity: Embedding similarity score [0, 1]
            - element_overlap: Jaccard similarity of TKS elements [0, 1]
            - fidelity_score: Combined metric (weighted average)
            - extracted_elements: List of elements found in equation
            - regenerated_elements: List of elements found in regenerated story

    Examples:
        >>> result = compute_roundtrip_fidelity(
        ...     "She felt fear and anxiety.",
        ...     model, tokenizer
        ... )
        >>> print(f"Fidelity: {result['fidelity_score']:.2f}")
        Fidelity: 0.84
    """
    if device is None:
        device = next(model.parameters()).device

    # Step 1: Story → Equation
    story_to_eq_prompt = f"Story: {original_story}\nEquation:"
    extracted_equation = generate_tokens(
        model, tokenizer, story_to_eq_prompt,
        max_new_tokens=100,
        temperature=temperature,
        device=device
    )

    if verbose:
        print(f"\n[Story → Equation]")
        print(f"  Input: {original_story[:80]}...")
        print(f"  Generated: {extracted_equation[:80]}...")

    # Step 2: Equation → Story
    eq_to_story_prompt = f"Equation: {extracted_equation}\nInterpretation:"
    regenerated_story = generate_tokens(
        model, tokenizer, eq_to_story_prompt,
        max_new_tokens=150,
        temperature=temperature,
        device=device
    )

    if verbose:
        print(f"\n[Equation → Story]")
        print(f"  Input: {extracted_equation[:80]}...")
        print(f"  Generated: {regenerated_story[:80]}...")

    # Step 3: Compute metrics
    semantic_sim = compute_semantic_similarity(original_story, regenerated_story)

    extracted_elements = extract_tks_elements(extracted_equation)
    regenerated_elements = extract_tks_elements(regenerated_story)

    element_overlap = compute_element_overlap(
        ' '.join(extracted_elements),
        ' '.join(regenerated_elements)
    )

    # Combined fidelity score (60% semantic, 40% element overlap)
    # These weights reflect that semantic meaning is primary,
    # but structural element consistency is also important
    fidelity_score = 0.6 * semantic_sim + 0.4 * element_overlap

    if verbose:
        print(f"\n[Metrics]")
        print(f"  Semantic similarity: {semantic_sim:.3f}")
        print(f"  Element overlap: {element_overlap:.3f}")
        print(f"  Fidelity score: {fidelity_score:.3f}")

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


def run_roundtrip_validation(
    test_stories: List[str],
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    device: torch.device = None,
    temperature: float = 0.7,
    verbose: bool = False
) -> Dict:
    """
    Run validation on multiple stories and compute aggregate metrics.

    Processes a batch of test stories and computes both individual and
    aggregate fidelity metrics across all samples.

    Args:
        test_stories: List of story strings to validate
        model: Trained TKS model
        tokenizer: Tokenizer instance
        device: Device for inference
        temperature: Sampling temperature
        verbose: Print detailed results

    Returns:
        Dictionary containing:
            - results: List of individual round-trip results
            - aggregate: Aggregate statistics including:
                - mean_fidelity: Average fidelity score
                - mean_semantic_similarity: Average semantic similarity
                - mean_element_overlap: Average element overlap
                - std_fidelity: Standard deviation of fidelity
                - min_fidelity: Minimum fidelity score
                - max_fidelity: Maximum fidelity score
                - num_stories: Number of stories processed
                - element_statistics: Element extraction statistics

    Examples:
        >>> stories = [
        ...     "She felt fear and anxiety.",
        ...     "A man thought about power.",
        ... ]
        >>> results = run_roundtrip_validation(stories, model, tokenizer)
        >>> print(f"Mean fidelity: {results['aggregate']['mean_fidelity']:.3f}")
    """
    if device is None:
        device = next(model.parameters()).device

    print(f"\n{'='*70}")
    print(f"ROUND-TRIP VALIDATION - {len(test_stories)} Stories")
    print(f"{'='*70}\n")

    results = []

    for i, story in enumerate(test_stories, 1):
        print(f"\nStory {i}/{len(test_stories)}: {story[:60]}...")

        result = compute_roundtrip_fidelity(
            story, model, tokenizer,
            device=device,
            temperature=temperature,
            verbose=verbose
        )

        results.append(result)

        # Print summary for this story
        print(f"  → Fidelity: {result['fidelity_score']:.3f} "
              f"(semantic: {result['semantic_similarity']:.3f}, "
              f"element: {result['element_overlap']:.3f})")
        print(f"  → Elements: {len(result['extracted_elements'])} extracted, "
              f"{len(result['regenerated_elements'])} regenerated")

    # Compute aggregate statistics
    fidelity_scores = [r['fidelity_score'] for r in results]
    semantic_scores = [r['semantic_similarity'] for r in results]
    element_scores = [r['element_overlap'] for r in results]

    # Element statistics
    all_extracted = [e for r in results for e in r['extracted_elements']]
    all_regenerated = [e for r in results for e in r['regenerated_elements']]

    element_counter = Counter(all_extracted)
    most_common_elements = element_counter.most_common(10)

    aggregate = {
        'mean_fidelity': float(np.mean(fidelity_scores)),
        'std_fidelity': float(np.std(fidelity_scores)),
        'min_fidelity': float(np.min(fidelity_scores)),
        'max_fidelity': float(np.max(fidelity_scores)),
        'mean_semantic_similarity': float(np.mean(semantic_scores)),
        'std_semantic_similarity': float(np.std(semantic_scores)),
        'mean_element_overlap': float(np.mean(element_scores)),
        'std_element_overlap': float(np.std(element_scores)),
        'num_stories': len(test_stories),
        'element_statistics': {
            'total_extracted': len(all_extracted),
            'total_regenerated': len(all_regenerated),
            'unique_extracted': len(set(all_extracted)),
            'unique_regenerated': len(set(all_regenerated)),
            'most_common_elements': most_common_elements,
        }
    }

    # Print aggregate results
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*70}\n")
    print(f"Fidelity Score:")
    print(f"  Mean: {aggregate['mean_fidelity']:.3f} ± {aggregate['std_fidelity']:.3f}")
    print(f"  Range: [{aggregate['min_fidelity']:.3f}, {aggregate['max_fidelity']:.3f}]")
    print(f"\nSemantic Similarity:")
    print(f"  Mean: {aggregate['mean_semantic_similarity']:.3f} ± {aggregate['std_semantic_similarity']:.3f}")
    print(f"\nElement Overlap:")
    print(f"  Mean: {aggregate['mean_element_overlap']:.3f} ± {aggregate['std_element_overlap']:.3f}")
    print(f"\nElement Statistics:")
    print(f"  Total extracted: {aggregate['element_statistics']['total_extracted']}")
    print(f"  Total regenerated: {aggregate['element_statistics']['total_regenerated']}")
    print(f"  Unique extracted: {aggregate['element_statistics']['unique_extracted']}")
    print(f"  Unique regenerated: {aggregate['element_statistics']['unique_regenerated']}")

    if most_common_elements:
        print(f"\n  Most common elements:")
        for elem, count in most_common_elements[:5]:
            print(f"    {elem}: {count}")

    return {
        'results': results,
        'aggregate': aggregate
    }


def load_model(
    checkpoint_path: str,
    device: torch.device
) -> Tuple[SimpleTransformer, SimpleTokenizer]:
    """
    Load trained model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model onto

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both dict checkpoints and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 4)
        num_heads = config.get('num_heads', 8)
        dropout = config.get('dropout', 0.1)
    else:
        state_dict = checkpoint
        hidden_dim = 256
        num_layers = 4
        num_heads = 8
        dropout = 0.1

    tokenizer = SimpleTokenizer(max_length=256)
    model = SimpleTransformer(
        tokenizer.actual_vocab_size,
        hidden_dim=128,
        num_layers=2
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"  Vocab size: {tokenizer.actual_vocab_size}")
    print(f"  Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"  Device: {device}")

    return model, tokenizer


def load_stories_from_file(stories_path: str) -> List[str]:
    """
    Load test stories from a file.

    Supports:
        - Plain text files (one story per line)
        - JSON files (array of strings)
        - JSONL files (one JSON object per line with 'story' field)

    Args:
        stories_path: Path to stories file

    Returns:
        List of story strings
    """
    path = Path(stories_path)

    if not path.exists():
        raise FileNotFoundError(f"Stories file not found: {stories_path}")

    stories = []

    # Try JSON first
    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                stories = [str(item) for item in data]
            else:
                raise ValueError("JSON file must contain an array of strings")

    # Try JSONL
    elif path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if 'story' in entry:
                        stories.append(entry['story'])
                    elif isinstance(entry, str):
                        stories.append(entry)

    # Default to plain text
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    stories.append(line.strip())

    print(f"Loaded {len(stories)} stories from {stories_path}")
    return stories


# Default test stories as specified in requirements
DEFAULT_TEST_STORIES = [
    "She felt fear and anxiety.",
    "A man thought about power.",
    "Love caused joy in the heart.",
    "The mind vibrates with ideas.",
    "Effect follows cause in rhythm.",
]


def main():
    """
    Main CLI entry point for semantic round-trip validation.

    Usage:
        # Use default test stories
        python3 scripts/semantic_roundtrip_metrics.py --model output/model.pt

        # Use custom stories from file
        python3 scripts/semantic_roundtrip_metrics.py --model output/model.pt --stories test_stories.txt

        # Adjust temperature for generation
        python3 scripts/semantic_roundtrip_metrics.py --model output/model.pt --temperature 0.5

        # Save results to JSON
        python3 scripts/semantic_roundtrip_metrics.py --model output/model.pt --output results.json
    """
    parser = argparse.ArgumentParser(
        description='Semantic Round-trip Fidelity Metrics for TKS Story↔Equation Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with default test stories
  python3 scripts/semantic_roundtrip_metrics.py --model output/model.pt

  # Use custom test stories
  python3 scripts/semantic_roundtrip_metrics.py --model output/model.pt --stories test.txt

  # Adjust temperature and save results
  python3 scripts/semantic_roundtrip_metrics.py --model output/model.pt --temperature 0.5 --output results.json

  # Verbose output
  python3 scripts/semantic_roundtrip_metrics.py --model output/model.pt --verbose

Default test stories:
  1. "She felt fear and anxiety."
  2. "A man thought about power."
  3. "Love caused joy in the heart."
  4. "The mind vibrates with ideas."
  5. "Effect follows cause in rhythm."
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )

    parser.add_argument(
        '--stories',
        type=str,
        default=None,
        help='Path to test stories file (txt, json, or jsonl). Uses default stories if not provided.'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature for generation (default: 0.7)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON (optional)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed intermediate results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for inference (auto-detect if not specified)'
    )

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"TKS SEMANTIC ROUND-TRIP FIDELITY VALIDATION")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Temperature: {args.temperature}")

    # Load model
    model, tokenizer = load_model(args.model, device)

    # Load test stories
    if args.stories:
        test_stories = load_stories_from_file(args.stories)
    else:
        print(f"\nUsing default test stories ({len(DEFAULT_TEST_STORIES)} stories)")
        test_stories = DEFAULT_TEST_STORIES

    # Run validation
    results = run_roundtrip_validation(
        test_stories,
        model,
        tokenizer,
        device=device,
        temperature=args.temperature,
        verbose=args.verbose
    )

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make results JSON-serializable
        save_data = {
            'metadata': {
                'model_checkpoint': args.model,
                'num_stories': len(test_stories),
                'temperature': args.temperature,
                'device': str(device),
            },
            'results': results['results'],
            'aggregate': results['aggregate']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")

    # Print final summary
    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOverall Fidelity: {results['aggregate']['mean_fidelity']:.3f}")
    print(f"Stories Processed: {len(test_stories)}")
    print(f"Elements Extracted: {results['aggregate']['element_statistics']['total_extracted']}")

    # Exit with appropriate status code
    mean_fidelity = results['aggregate']['mean_fidelity']
    if mean_fidelity >= 0.8:
        print(f"\n✓ Excellent fidelity (≥0.8)")
        sys.exit(0)
    elif mean_fidelity >= 0.6:
        print(f"\n⚠ Moderate fidelity (≥0.6)")
        sys.exit(0)
    else:
        print(f"\n✗ Low fidelity (<0.6) - model may need improvement")
        sys.exit(1)


if __name__ == '__main__':
    main()

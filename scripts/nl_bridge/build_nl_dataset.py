#!/usr/bin/env python3
"""
Agent A: NL Dataset Builder

Builds NL training set from tks_6400_complete_merged.jsonl.
Each equation's "interpretation" field becomes the NL input paired with
structured triplet indices.

Output: data/nl_bridge/{nl_train.jsonl, nl_val.jsonl, nl_test.jsonl}

Usage:
    python scripts/nl_bridge/build_nl_dataset.py
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Set seed for reproducibility
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_JSONL = PROJECT_ROOT / "tks_6400_complete_merged.jsonl"
CORPUS_PT = PROJECT_ROOT / "data" / "equation_embeddings" / "equation_corpus.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "nl_bridge"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# =============================================================================
# INDEX MAPPING (must match equation_detector.py)
# =============================================================================

WORLD_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
OPERATOR_MAP = {'+': 0, '-': 1, 'x': 2, '*': 2, '/': 3}


def element_to_index(element: str) -> int:
    """Convert element string (e.g., 'A1') to index 0-39."""
    element = element.upper().strip()
    match = re.match(r'([ABCD])(\d{1,2})', element)
    if not match:
        raise ValueError(f"Invalid element: {element}")
    world = match.group(1)
    noetic = int(match.group(2))
    if noetic < 1 or noetic > 10:
        raise ValueError(f"Noetic must be 1-10, got {noetic}")
    return WORLD_MAP[world] * 10 + (noetic - 1)


def operator_to_index(op: str) -> int:
    """Convert operator string to index 0-3."""
    # Normalize unicode operators
    op = op.strip()
    if op == '\u00d7':  # multiplication sign
        op = 'x'
    elif op == '\u00f7':  # division sign
        op = '/'
    return OPERATOR_MAP.get(op, OPERATOR_MAP.get(op.lower(), 0))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NLSample:
    """Single NL training sample."""
    nl_text: str
    left_idx: int
    right_idx: int
    operator_idx: int
    equation_idx: int
    source_weight: float
    equation_str: str  # For debugging


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def extract_nl_variants(interpretation: str, equation: str) -> List[str]:
    """
    Extract multiple NL variants from an interpretation.

    Returns:
        List of NL text variants (full + excerpts)
    """
    variants = []

    # Full interpretation (primary)
    if interpretation and len(interpretation) > 20:
        variants.append(interpretation)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', interpretation)

    # Add individual sentences (if substantial)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 50 and sent not in variants:
            variants.append(sent)

    # Extract key phrases (sentences mentioning both elements)
    # e.g., "Spiritual Mind (A1) combined with Mental Positive (B2)"
    element_pattern = r'[ABCD]\d{1,2}'
    for sent in sentences:
        elements = re.findall(element_pattern, sent, re.IGNORECASE)
        if len(elements) >= 2 and sent not in variants:
            variants.append(sent)

    # Fallback: use equation string itself
    if not variants:
        variants.append(equation)

    return variants


def clean_nl_text(text: str) -> str:
    """Clean and normalize NL text."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove very long parenthetical element references
    # e.g., "Spiritual Mind (A1)" -> "Spiritual Mind"
    # This makes the NL more natural but keeps element names
    text = re.sub(r'\s*\([ABCD]\d{1,2}\)', '', text)
    return text.strip()


# =============================================================================
# MAIN BUILDER
# =============================================================================

def load_source_equations(path: Path) -> List[Dict]:
    """Load source equations from JSONL."""
    equations = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                equations.append(json.loads(line))
    return equations


def build_samples(equations: List[Dict]) -> List[NLSample]:
    """Build NL samples from equations."""
    samples = []

    for idx, eq in enumerate(equations):
        try:
            left_idx = element_to_index(eq['left'])
            right_idx = element_to_index(eq['right'])
            operator_idx = operator_to_index(eq['operator'])
            source_weight = 1.0 if eq.get('source', 'original') == 'original' else 0.5

            interpretation = eq.get('interpretation', '')
            equation_str = eq.get('equation', f"{eq['left']} {eq['operator']} {eq['right']}")

            # Extract NL variants
            nl_variants = extract_nl_variants(interpretation, equation_str)

            # Create sample for each variant
            for nl_text in nl_variants:
                cleaned = clean_nl_text(nl_text)
                if len(cleaned) < 10:
                    continue

                sample = NLSample(
                    nl_text=cleaned,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    operator_idx=operator_idx,
                    equation_idx=idx,
                    source_weight=source_weight,
                    equation_str=equation_str,
                )
                samples.append(sample)

        except Exception as e:
            print(f"Warning: Skipping equation {idx}: {e}")
            continue

    return samples


def split_samples(
    samples: List[NLSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[NLSample], List[NLSample], List[NLSample]]:
    """
    Split samples into train/val/test.

    Splits by equation_idx to avoid data leakage (all variants of same
    equation go to same split).
    """
    # Group by equation_idx
    by_equation: Dict[int, List[NLSample]] = {}
    for sample in samples:
        if sample.equation_idx not in by_equation:
            by_equation[sample.equation_idx] = []
        by_equation[sample.equation_idx].append(sample)

    # Shuffle equation indices
    eq_indices = list(by_equation.keys())
    random.shuffle(eq_indices)

    # Split
    n = len(eq_indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_eqs = set(eq_indices[:train_end])
    val_eqs = set(eq_indices[train_end:val_end])
    test_eqs = set(eq_indices[val_end:])

    train = [s for s in samples if s.equation_idx in train_eqs]
    val = [s for s in samples if s.equation_idx in val_eqs]
    test = [s for s in samples if s.equation_idx in test_eqs]

    return train, val, test


def save_samples(samples: List[NLSample], path: Path) -> None:
    """Save samples to JSONL."""
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Convert to dict, removing equation_str (debug only)
            d = asdict(sample)
            del d['equation_str']
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def verify_corpus_alignment(samples: List[NLSample], corpus_path: Path) -> bool:
    """Verify samples align with corpus embeddings."""
    try:
        import torch
        corpus = torch.load(corpus_path, weights_only=False)

        n_corpus = corpus['embeddings'].shape[0]
        max_idx = max(s.equation_idx for s in samples)

        if max_idx >= n_corpus:
            print(f"ERROR: Max equation_idx {max_idx} >= corpus size {n_corpus}")
            return False

        # Verify operator/left/right indices match
        mismatches = 0
        for sample in samples[:100]:  # Check first 100
            idx = sample.equation_idx
            if corpus['operators'][idx].item() != sample.operator_idx:
                mismatches += 1
            if corpus['left_indices'][idx].item() != sample.left_idx:
                mismatches += 1
            if corpus['right_indices'][idx].item() != sample.right_idx:
                mismatches += 1

        if mismatches > 0:
            print(f"WARNING: {mismatches} index mismatches in first 100 samples")
            return False

        print(f"Corpus alignment verified: {n_corpus} equations")
        return True

    except Exception as e:
        print(f"WARNING: Could not verify corpus alignment: {e}")
        return True  # Continue anyway


def main():
    """Main entry point."""
    print("=" * 60)
    print("Agent A: NL Dataset Builder")
    print("=" * 60)

    # Check source exists
    if not SOURCE_JSONL.exists():
        print(f"ERROR: Source file not found: {SOURCE_JSONL}")
        return 1

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load equations
    print(f"\nLoading equations from: {SOURCE_JSONL}")
    equations = load_source_equations(SOURCE_JSONL)
    print(f"Loaded {len(equations)} equations")

    # Build samples
    print("\nBuilding NL samples...")
    samples = build_samples(equations)
    print(f"Created {len(samples)} NL samples")
    print(f"  - Average {len(samples)/len(equations):.1f} variants per equation")

    # Verify corpus alignment
    if CORPUS_PT.exists():
        verify_corpus_alignment(samples, CORPUS_PT)

    # Split
    print("\nSplitting into train/val/test...")
    train, val, test = split_samples(samples, TRAIN_RATIO, VAL_RATIO)
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")

    # Save
    print("\nSaving datasets...")
    save_samples(train, OUTPUT_DIR / "nl_train.jsonl")
    save_samples(val, OUTPUT_DIR / "nl_val.jsonl")
    save_samples(test, OUTPUT_DIR / "nl_test.jsonl")

    # Save index mapping
    index_map = {
        'num_equations': len(equations),
        'num_samples': len(samples),
        'splits': {
            'train': len(train),
            'val': len(val),
            'test': len(test),
        },
        'operator_map': OPERATOR_MAP,
        'world_map': WORLD_MAP,
    }
    with open(OUTPUT_DIR / "index_mapping.json", 'w') as f:
        json.dump(index_map, f, indent=2)

    # Summary stats
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    # Source weight distribution
    orig_count = sum(1 for s in samples if s.source_weight == 1.0)
    gen_count = len(samples) - orig_count
    print(f"\nSource distribution:")
    print(f"  Original: {orig_count} ({100*orig_count/len(samples):.1f}%)")
    print(f"  Generated: {gen_count} ({100*gen_count/len(samples):.1f}%)")

    # Operator distribution
    op_counts = [0, 0, 0, 0]
    for s in samples:
        op_counts[s.operator_idx] += 1
    print(f"\nOperator distribution:")
    for op, name in [(0, '+'), (1, '-'), (2, 'x'), (3, '/')]:
        print(f"  {name}: {op_counts[op]} ({100*op_counts[op]/len(samples):.1f}%)")

    # Text length distribution
    lengths = [len(s.nl_text) for s in samples]
    print(f"\nNL text length:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f}")

    print("\n" + "=" * 60)
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

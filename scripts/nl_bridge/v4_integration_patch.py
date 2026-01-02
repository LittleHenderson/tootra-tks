#!/usr/bin/env python3
"""
Agent C: v4 Integration Patch

This module provides the NL retriever integration for TKSNoeticLM v4.
It can be applied as a mixin or the changes can be merged into tks_llm_core_v4.py.

Integration flow:
1. EquationDetector parses input text for explicit equations (e.g., "A1 + B4")
2. If detector finds nothing, NL retriever attempts semantic match
3. If retriever confidence > threshold, use retrieved triplet
4. Pass triplet to operator_core

Usage:
    # Option 1: Use the enhanced model directly
    from scripts.nl_bridge.v4_integration_patch import TKSNoeticLMWithNL

    model = TKSNoeticLMWithNL(config)
    out = model(tokens, raw_text=["spiritual mind combined with mental positive"])

    # Option 2: Apply patch to existing model
    from scripts.nl_bridge.v4_integration_patch import add_nl_retriever
    add_nl_retriever(existing_model, retriever_path="models/nl_retriever.pt")
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base components
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
    from equation_detector import EquationDetector, get_equation_vectors
    V4_AVAILABLE = True
except ImportError:
    V4_AVAILABLE = False
    print("Warning: tks_llm_core_v4 not found, patch cannot be applied")

# NL Retriever imports
NL_RETRIEVER_AVAILABLE = False
LinearNLRetriever = None
try:
    from scripts.nl_bridge.nl_retriever import LinearNLRetriever
    NL_RETRIEVER_AVAILABLE = True
except ImportError:
    pass

# Sentence encoder (lazy loaded)
SENTENCE_ENCODER = None


def get_sentence_encoder():
    """Lazily load sentence encoder."""
    global SENTENCE_ENCODER
    if SENTENCE_ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer
            SENTENCE_ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
            SENTENCE_ENCODER.eval()
            for p in SENTENCE_ENCODER.parameters():
                p.requires_grad = False
            print("Loaded sentence encoder: all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    return SENTENCE_ENCODER


# =============================================================================
# EXTENDED CONFIG
# =============================================================================

@dataclass
class TKSNoeticLMConfigWithNL(TKSNoeticLMConfig if V4_AVAILABLE else object):
    """Extended config with NL retriever options."""
    # NL Retriever config
    use_nl_retriever: bool = False
    nl_retriever_path: str = "models/nl_retriever.pt"
    nl_corpus_path: str = "data/equation_embeddings/equation_corpus.pt"
    nl_confidence_threshold: float = 0.5
    nl_top_k: int = 5


# =============================================================================
# NL RETRIEVER WRAPPER
# =============================================================================

class NLRetrieverWrapper(nn.Module):
    """
    Wrapper for NL retriever that handles encoding and retrieval.

    This is a thin wrapper that:
    1. Encodes raw text using MiniLM
    2. Runs the linear retriever
    3. Returns equation triplets
    """

    def __init__(
        self,
        retriever_path: str = None,
        corpus_path: str = None,
        confidence_threshold: float = 0.5,
        top_k: int = 5,
    ):
        super().__init__()

        self.confidence_threshold = confidence_threshold
        self.retriever = None

        # Load retriever if path provided
        if retriever_path and os.path.exists(retriever_path):
            self._load_retriever(retriever_path, corpus_path, top_k)

    def _load_retriever(
        self,
        retriever_path: str,
        corpus_path: str = None,
        top_k: int = 5,
    ) -> None:
        """Load pretrained retriever."""
        if not NL_RETRIEVER_AVAILABLE:
            raise ImportError("LinearNLRetriever not available")

        # Find corpus path
        if corpus_path is None:
            corpus_path = str(PROJECT_ROOT / "data" / "equation_embeddings" / "equation_corpus.pt")

        self.retriever = LinearNLRetriever(
            corpus_path=corpus_path,
            top_k=top_k,
        )

        state_dict = torch.load(retriever_path, weights_only=True)
        self.retriever.load_state_dict(state_dict)
        self.retriever.eval()

        print(f"Loaded NL retriever from: {retriever_path}")

    def forward(
        self,
        raw_texts: List[str],
        device: torch.device = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve equation triplets from raw text.

        Args:
            raw_texts: List of input strings
            device: Target device

        Returns:
            Dict with 'left_idx', 'right_idx', 'operator_idx', 'confidence'
            or None if retriever not loaded
        """
        if self.retriever is None:
            return None

        if device is None:
            device = next(self.retriever.parameters()).device

        # Encode texts
        encoder = get_sentence_encoder()
        with torch.no_grad():
            embeddings = encoder.encode(
                raw_texts,
                convert_to_tensor=True,
                device=device,
            )

        # Run retriever
        self.retriever.eval()
        with torch.no_grad():
            results = self.retriever(embeddings)

        return results

    def get_triplet_if_confident(
        self,
        raw_texts: List[str],
        device: torch.device = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get equation triplet only if confidence exceeds threshold.

        Returns:
            Tuple of (left_idx, right_idx, operator_idx) or None
        """
        results = self.forward(raw_texts, device)
        if results is None:
            return None

        mean_conf = results['confidence'].mean().item()
        if mean_conf < self.confidence_threshold:
            return None

        return (
            results['left_idx'],
            results['right_idx'],
            results['operator_idx'],
        )


# =============================================================================
# ENHANCED MODEL
# =============================================================================

if V4_AVAILABLE:

    class TKSNoeticLMWithNL(TKSNoeticLM):
        """
        TKSNoeticLM with NL retriever fallback.

        When EquationDetector fails to find explicit equations,
        the NL retriever attempts semantic matching.
        """

        def __init__(self, config: TKSNoeticLMConfigWithNL) -> None:
            # Initialize base model
            super().__init__(config)

            # Store NL config
            self.use_nl_retriever = config.use_nl_retriever
            self.nl_confidence_threshold = config.nl_confidence_threshold

            # Initialize NL retriever
            self.nl_retriever = None
            if config.use_nl_retriever:
                self.nl_retriever = NLRetrieverWrapper(
                    retriever_path=config.nl_retriever_path,
                    corpus_path=config.nl_corpus_path,
                    confidence_threshold=config.nl_confidence_threshold,
                    top_k=config.nl_top_k,
                )

            # Initialize equation detector
            self.equation_detector = EquationDetector()

        def forward(
            self,
            tokens: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            noetic_idx: Optional[int] = None,
            noetic_weights: Optional[torch.Tensor] = None,
            fractal_indices: Optional[torch.Tensor] = None,
            goal_state: Optional[torch.Tensor] = None,
            target_foundation: Optional[int] = None,
            # Existing: explicit triplet
            equation_triplet: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
            # NEW: raw text for NL fallback
            raw_text: Optional[List[str]] = None,
            use_nl_fallback: bool = True,
            return_full_trace: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass with NL retriever fallback.

            Args:
                ... (same as base) ...
                raw_text: List of raw input strings (for NL retrieval)
                use_nl_fallback: Whether to use NL retriever when detector fails
            """
            trace = {} if return_full_trace else None
            triplet_source = None

            # Step 1: Try EquationDetector first (if raw_text provided)
            if equation_triplet is None and raw_text is not None:
                batch_triplets = self.equation_detector.parse_batch(raw_text, tokens.device)
                if batch_triplets is not None:
                    # Use first equation from each sample
                    equation_triplet = (
                        batch_triplets.left_indices[:, 0],
                        batch_triplets.operator_indices[:, 0],
                        batch_triplets.right_indices[:, 0],
                    )
                    triplet_source = 'detector'

            # Step 2: NL Retriever fallback
            if (equation_triplet is None and
                use_nl_fallback and
                self.nl_retriever is not None and
                raw_text is not None):

                triplet = self.nl_retriever.get_triplet_if_confident(
                    raw_text, tokens.device
                )
                if triplet is not None:
                    equation_triplet = triplet
                    triplet_source = 'retriever'

            # Step 3: Call base forward with triplet
            out = super().forward(
                tokens=tokens,
                attention_mask=attention_mask,
                noetic_idx=noetic_idx,
                noetic_weights=noetic_weights,
                fractal_indices=fractal_indices,
                goal_state=goal_state,
                target_foundation=target_foundation,
                equation_triplet=equation_triplet,
                return_full_trace=return_full_trace,
            )

            # Add triplet source to output
            out['triplet_source'] = triplet_source

            if return_full_trace and 'trace' in out:
                out['trace']['triplet_source'] = triplet_source
                if triplet_source == 'retriever' and self.nl_retriever is not None:
                    # Add retrieval details
                    results = self.nl_retriever.forward(raw_text, tokens.device)
                    if results:
                        out['trace']['retriever_confidence'] = results['confidence']
                        out['trace']['retriever_top_k'] = results['top_k_indices']

            return out


# =============================================================================
# PATCH FUNCTION (for existing models)
# =============================================================================

def add_nl_retriever(
    model: 'TKSNoeticLM',
    retriever_path: str = "models/nl_retriever.pt",
    corpus_path: str = None,
    confidence_threshold: float = 0.5,
) -> None:
    """
    Add NL retriever to an existing TKSNoeticLM model.

    This modifies the model in-place by adding:
    - nl_retriever: NLRetrieverWrapper
    - equation_detector: EquationDetector

    And patching the forward method.
    """
    if corpus_path is None:
        corpus_path = str(PROJECT_ROOT / "data" / "equation_embeddings" / "equation_corpus.pt")

    # Add retriever
    model.nl_retriever = NLRetrieverWrapper(
        retriever_path=retriever_path,
        corpus_path=corpus_path,
        confidence_threshold=confidence_threshold,
    )

    # Add detector
    model.equation_detector = EquationDetector()

    # Store original forward
    original_forward = model.forward

    def patched_forward(
        tokens: torch.LongTensor,
        raw_text: Optional[List[str]] = None,
        use_nl_fallback: bool = True,
        equation_triplet: Optional[Tuple] = None,
        **kwargs,
    ):
        triplet_source = None

        # Try detector
        if equation_triplet is None and raw_text is not None:
            batch_triplets = model.equation_detector.parse_batch(raw_text, tokens.device)
            if batch_triplets is not None:
                equation_triplet = (
                    batch_triplets.left_indices[:, 0],
                    batch_triplets.operator_indices[:, 0],
                    batch_triplets.right_indices[:, 0],
                )
                triplet_source = 'detector'

        # Fallback to retriever
        if (equation_triplet is None and
            use_nl_fallback and
            model.nl_retriever is not None and
            raw_text is not None):

            triplet = model.nl_retriever.get_triplet_if_confident(
                raw_text, tokens.device
            )
            if triplet is not None:
                equation_triplet = triplet
                triplet_source = 'retriever'

        # Call original
        out = original_forward(
            tokens=tokens,
            equation_triplet=equation_triplet,
            **kwargs,
        )
        out['triplet_source'] = triplet_source
        return out

    model.forward = patched_forward
    print(f"NL retriever added to model (threshold={confidence_threshold})")


# =============================================================================
# TESTING
# =============================================================================

def test_integration():
    """Test the NL integration."""
    print("=" * 60)
    print("Testing NL Bridge Integration")
    print("=" * 60)

    if not V4_AVAILABLE:
        print("SKIP: tks_llm_core_v4 not available")
        return

    # Test retriever wrapper
    print("\n1. Testing NLRetrieverWrapper...")
    wrapper = NLRetrieverWrapper(
        retriever_path=str(PROJECT_ROOT / "models" / "nl_retriever.pt"),
        corpus_path=str(PROJECT_ROOT / "data" / "equation_embeddings" / "equation_corpus.pt"),
        confidence_threshold=0.3,
    )

    if wrapper.retriever is not None:
        test_texts = [
            "spiritual mind combined with mental positive",
            "physical vibration opposing emotional rhythm",
        ]
        results = wrapper.forward(test_texts)
        if results:
            print(f"  Retrieved for '{test_texts[0][:40]}...':")
            print(f"    left_idx: {results['left_idx'][0].item()}")
            print(f"    right_idx: {results['right_idx'][0].item()}")
            print(f"    operator_idx: {results['operator_idx'][0].item()}")
            print(f"    confidence: {results['confidence'][0].item():.3f}")
    else:
        print("  SKIP: Retriever not loaded (run training first)")

    # Test enhanced model
    print("\n2. Testing TKSNoeticLMWithNL...")
    config = TKSNoeticLMConfigWithNL(
        vocab_size=512,
        use_operator_core=True,
        use_nl_retriever=True,
        nl_retriever_path=str(PROJECT_ROOT / "models" / "nl_retriever.pt"),
        nl_confidence_threshold=0.3,
    )

    try:
        model = TKSNoeticLMWithNL(config)
        tokens = torch.randint(0, 512, (2, 32))
        raw_text = [
            "The equation A1 + B4 represents something.",  # Should use detector
            "spiritual consciousness merging with positive energy",  # Should use retriever
        ]

        out = model(tokens, raw_text=raw_text)
        print(f"  triplet_source: {out['triplet_source']}")
        print(f"  logits shape: {out['logits'].shape}")
        print("  SUCCESS")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Integration test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_integration()

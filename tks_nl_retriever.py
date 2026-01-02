"""
TKS Natural Language Retriever

A lightweight linear projection retriever that maps NL embeddings to equation embeddings
for fast equation retrieval. Uses pure linear layers for speed.

Architecture:
    NL embedding (384D) → Linear → [ReLU → Linear] → Equation space (384D)

Usage:
    retriever = TKSNLRetriever()
    retriever.load_state_dict(torch.load('output/nl_retriever.pt'))
    retriever.eval()

    indices, similarities = retriever.retrieve(nl_embedding, equation_corpus, top_k=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class TKSNLRetriever(nn.Module):
    """
    Linear projection from NL embedding space to equation embedding space.
    No transformers - just linear layers for speed.

    Args:
        nl_dim: Dimension of NL embeddings (default: 384)
        eq_dim: Dimension of equation embeddings (default: 384)
        hidden_dim: Hidden layer dimension (default: 256)
        use_hidden: Whether to use hidden layer (default: True)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        nl_dim: int = 384,
        eq_dim: int = 384,
        hidden_dim: int = 256,
        use_hidden: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.nl_dim = nl_dim
        self.eq_dim = eq_dim
        self.hidden_dim = hidden_dim
        self.use_hidden = use_hidden

        if use_hidden:
            # Two-layer MLP with ReLU activation
            self.projection = nn.Sequential(
                nn.Linear(nl_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, eq_dim)
            )
        else:
            # Single linear projection
            self.projection = nn.Linear(nl_dim, eq_dim)

        # L2 normalization for cosine similarity
        self.normalize = True

    def forward(self, nl_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project NL embedding to equation embedding space.

        Args:
            nl_embedding: Tensor of shape (batch_size, nl_dim) or (nl_dim,)

        Returns:
            Projected embedding of shape (batch_size, eq_dim) or (eq_dim,)
        """
        projected = self.projection(nl_embedding)

        if self.normalize:
            # L2 normalize for cosine similarity
            projected = F.normalize(projected, p=2, dim=-1)

        return projected

    def retrieve(
        self,
        nl_embedding: torch.Tensor,
        equation_corpus: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k equations from corpus using cosine similarity.

        Args:
            nl_embedding: NL embedding tensor of shape (nl_dim,) or (batch_size, nl_dim)
            equation_corpus: Equation embeddings of shape (corpus_size, eq_dim)
            top_k: Number of top results to return

        Returns:
            indices: Tensor of shape (top_k,) or (batch_size, top_k) with corpus indices
            similarities: Tensor of shape (top_k,) or (batch_size, top_k) with cosine similarities
        """
        # Project NL embedding to equation space
        with torch.no_grad():
            projected = self.forward(nl_embedding)

        # Ensure equation corpus is normalized
        if self.normalize:
            equation_corpus = F.normalize(equation_corpus, p=2, dim=-1)

        # Handle single embedding vs batch
        if projected.dim() == 1:
            projected = projected.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute cosine similarity: (batch_size, corpus_size)
        similarities = torch.matmul(projected, equation_corpus.T)

        # Get top-k
        top_similarities, top_indices = torch.topk(
            similarities,
            k=min(top_k, equation_corpus.size(0)),
            dim=-1,
            largest=True,
            sorted=True
        )

        if squeeze_output:
            return top_indices.squeeze(0), top_similarities.squeeze(0)

        return top_indices, top_similarities

    def get_triplet(
        self,
        nl_embedding: torch.Tensor,
        equation_corpus: torch.Tensor,
        corpus_meta: list
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve equation triplet (left, right, operator) for best match.

        Args:
            nl_embedding: NL embedding tensor of shape (nl_dim,)
            equation_corpus: Equation embeddings of shape (corpus_size, eq_dim)
            corpus_meta: List of metadata dicts with keys 'left_idx', 'right_idx', 'operator_idx'

        Returns:
            Dictionary with keys:
                - 'left_idx': int
                - 'right_idx': int
                - 'operator_idx': int
                - 'corpus_idx': int (index in corpus)
                - 'similarity': float
            Returns None if corpus is empty or no valid match found
        """
        if len(equation_corpus) == 0 or len(corpus_meta) == 0:
            return None

        # Get top-1 match
        indices, similarities = self.retrieve(nl_embedding, equation_corpus, top_k=1)

        # Extract index and similarity (handle both tensor and scalar)
        if indices.dim() == 0:
            best_idx = indices.item()
            best_sim = similarities.item()
        else:
            best_idx = indices[0].item()
            best_sim = similarities[0].item()

        # Get metadata for best match
        if best_idx >= len(corpus_meta):
            return None

        meta = corpus_meta[best_idx]

        return {
            'left_idx': meta.get('left_idx', -1),
            'right_idx': meta.get('right_idx', -1),
            'operator_idx': meta.get('operator_idx', -1),
            'corpus_idx': best_idx,
            'similarity': best_sim
        }

    def batch_retrieve_triplets(
        self,
        nl_embeddings: torch.Tensor,
        equation_corpus: torch.Tensor,
        corpus_meta: list
    ) -> list:
        """
        Retrieve triplets for a batch of NL embeddings.

        Args:
            nl_embeddings: Tensor of shape (batch_size, nl_dim)
            equation_corpus: Equation embeddings of shape (corpus_size, eq_dim)
            corpus_meta: List of metadata dicts

        Returns:
            List of triplet dictionaries (same format as get_triplet)
        """
        # Get top-1 for each in batch
        indices, similarities = self.retrieve(nl_embeddings, equation_corpus, top_k=1)

        # indices: (batch_size, 1), similarities: (batch_size, 1)
        results = []
        for i in range(indices.size(0)):
            best_idx = indices[i, 0].item()
            best_sim = similarities[i, 0].item()

            if best_idx >= len(corpus_meta):
                results.append(None)
                continue

            meta = corpus_meta[best_idx]
            results.append({
                'left_idx': meta.get('left_idx', -1),
                'right_idx': meta.get('right_idx', -1),
                'operator_idx': meta.get('operator_idx', -1),
                'corpus_idx': best_idx,
                'similarity': best_sim
            })

        return results

    def compute_similarity_matrix(
        self,
        nl_embeddings: torch.Tensor,
        equation_corpus: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute full similarity matrix between NL embeddings and equation corpus.

        Args:
            nl_embeddings: Tensor of shape (n, nl_dim)
            equation_corpus: Tensor of shape (m, eq_dim)

        Returns:
            Similarity matrix of shape (n, m)
        """
        with torch.no_grad():
            projected = self.forward(nl_embeddings)

        if self.normalize:
            equation_corpus = F.normalize(equation_corpus, p=2, dim=-1)

        # Compute all similarities
        similarities = torch.matmul(projected, equation_corpus.T)
        return similarities


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) contrastive loss for retriever training.

    Treats matched NL-equation pairs as positives, all others as negatives.

    Args:
        temperature: Temperature scaling parameter (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        nl_embeddings: torch.Tensor,
        eq_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            nl_embeddings: Projected NL embeddings (batch_size, dim)
            eq_embeddings: Equation embeddings (batch_size, dim)

        Returns:
            Scalar loss value
        """
        batch_size = nl_embeddings.size(0)

        # Normalize embeddings
        nl_embeddings = F.normalize(nl_embeddings, p=2, dim=1)
        eq_embeddings = F.normalize(eq_embeddings, p=2, dim=1)

        # Compute similarity matrix: (batch_size, batch_size)
        similarity_matrix = torch.matmul(nl_embeddings, eq_embeddings.T) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=nl_embeddings.device)

        # Cross-entropy loss (treats each row as a classification problem)
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


def load_retriever(checkpoint_path: str, device: str = 'cpu') -> TKSNLRetriever:
    """
    Load a trained retriever from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded retriever model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config if available
    config = checkpoint.get('config', {})

    retriever = TKSNLRetriever(
        nl_dim=config.get('nl_dim', 384),
        eq_dim=config.get('eq_dim', 384),
        hidden_dim=config.get('hidden_dim', 256),
        use_hidden=config.get('use_hidden', True),
        dropout=config.get('dropout', 0.1)
    )

    # Load state dict
    retriever.load_state_dict(checkpoint['model_state_dict'])
    retriever.to(device)
    retriever.eval()

    return retriever

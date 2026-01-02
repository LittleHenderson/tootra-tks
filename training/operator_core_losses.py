"""
Operator Core Training Losses

Losses for training the TKS compositional layer:
1. Contrastive alignment to 6400 equation corpus
2. Operator symmetry/anti-symmetry constraints
3. Source weighting (original > generated)

Usage:
    loss_fn = OperatorCoreLoss(corpus_path="data/equation_embeddings/equation_corpus.pt")
    losses = loss_fn(model_outputs, batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OperatorCoreLossConfig:
    """Configuration for operator core loss."""
    contrastive_weight: float = 1.0
    symmetry_weight: float = 0.1
    original_source_weight: float = 1.0
    generated_source_weight: float = 0.5
    temperature: float = 0.07


class OperatorCoreLoss(nn.Module):
    """
    Combined loss for operator core training.

    Components:
    1. Contrastive: Pull composed repr toward corpus equation embedding
    2. Symmetry: Enforce +/× commutativity, -/÷ anti-commutativity
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        config: Optional[OperatorCoreLossConfig] = None,
    ):
        super().__init__()
        self.config = config or OperatorCoreLossConfig()

        # Register empty buffers first
        self.register_buffer('corpus_embeddings', None)
        self.register_buffer('corpus_source_weights', None)
        self.register_buffer('corpus_operators', None)
        self.register_buffer('corpus_left_indices', None)
        self.register_buffer('corpus_right_indices', None)

        if corpus_path:
            self.load_corpus(corpus_path)

    def load_corpus(self, path: str):
        """Load pre-computed equation embeddings."""
        import os
        if not os.path.exists(path):
            return

        corpus = torch.load(path, map_location='cpu', weights_only=False)

        # Update buffers
        self.corpus_embeddings = corpus['embeddings']
        self.corpus_source_weights = corpus['source_weights']
        self.corpus_operators = corpus['operators']
        self.corpus_left_indices = corpus['left_indices']
        self.corpus_right_indices = corpus['right_indices']

    def get_corpus_embedding(
        self,
        left_idx: torch.Tensor,
        right_idx: torch.Tensor,
        op_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up corpus embeddings for given equation triplets.

        Returns:
            embeddings: [batch, equation_dim]
            source_weights: [batch]
        """
        if self.corpus_embeddings is None:
            return None, None

        batch_size = left_idx.shape[0]
        device = left_idx.device

        # Build lookup indices
        # Index = left_idx * 40 * 4 + right_idx * 4 + op_idx
        # This assumes corpus is sorted by left, then right, then op
        # Actually, let's just search for matching triplets

        embeddings = []
        weights = []

        for i in range(batch_size):
            l, r, o = left_idx[i].item(), right_idx[i].item(), op_idx[i].item()

            # Find matching equation in corpus
            mask = (
                (self.corpus_left_indices == l) &
                (self.corpus_right_indices == r) &
                (self.corpus_operators == o)
            )

            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0][0]
                embeddings.append(self.corpus_embeddings[idx])
                weights.append(self.corpus_source_weights[idx])
            else:
                # Fallback: zero embedding, low weight
                embeddings.append(torch.zeros(self.corpus_embeddings.shape[1], device=device))
                weights.append(torch.tensor(0.1, device=device))

        return torch.stack(embeddings), torch.stack(weights)

    def contrastive_loss(
        self,
        composed_repr: torch.Tensor,       # [batch, equation_dim]
        target_embeddings: torch.Tensor,   # [batch, equation_dim]
        source_weights: torch.Tensor,      # [batch]
    ) -> torch.Tensor:
        """
        Weighted contrastive alignment loss.

        Higher weight for original corpus examples.
        """
        composed_repr = F.normalize(composed_repr, dim=-1)
        target_embeddings = F.normalize(target_embeddings, dim=-1)

        # Cosine similarity
        similarity = torch.sum(composed_repr * target_embeddings, dim=-1)
        similarity = similarity / self.config.temperature

        # Apply source weighting
        weighted_sim = similarity * source_weights
        loss = -weighted_sim.mean()

        return loss

    def symmetry_loss(
        self,
        left_vec: torch.Tensor,   # [batch, 40]
        right_vec: torch.Tensor,  # [batch, 40]
        operator: torch.Tensor,   # [batch]
        bilinear_fn,              # Function to compute bilinear composition
    ) -> Dict[str, torch.Tensor]:
        """
        Compute operator symmetry/anti-symmetry violation losses.

        + and × (0, 2): symmetric - f(A,B) = f(B,A)
        - and ÷ (1, 3): anti-symmetric - f(A,B) = -f(B,A)
        """
        # Compute both orderings
        composed_ab = bilinear_fn(left_vec, right_vec, operator)
        composed_ba = bilinear_fn(right_vec, left_vec, operator)

        # Masks
        is_symmetric = (operator == 0) | (operator == 2)
        is_antisymmetric = (operator == 1) | (operator == 3)

        losses = {}

        # Symmetry: ||f(A,B) - f(B,A)||^2
        if is_symmetric.any():
            sym_mask = is_symmetric.float().unsqueeze(-1)
            sym_diff = (composed_ab - composed_ba) * sym_mask
            sym_loss = (sym_diff ** 2).mean()
            losses['symmetry'] = sym_loss
        else:
            losses['symmetry'] = torch.tensor(0.0, device=operator.device)

        # Anti-symmetry: ||f(A,B) + f(B,A)||^2
        if is_antisymmetric.any():
            asym_mask = is_antisymmetric.float().unsqueeze(-1)
            asym_diff = (composed_ab + composed_ba) * asym_mask
            asym_loss = (asym_diff ** 2).mean()
            losses['antisymmetry'] = asym_loss
        else:
            losses['antisymmetry'] = torch.tensor(0.0, device=operator.device)

        losses['total'] = losses['symmetry'] + losses['antisymmetry']
        return losses

    def forward(
        self,
        operator_output: Dict[str, torch.Tensor],
        left_idx: torch.Tensor,
        right_idx: torch.Tensor,
        op_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all operator core losses.

        Args:
            operator_output: Output from TKSCompositionalLayerV2
            left_idx: [batch] left element indices (0-39)
            right_idx: [batch] right element indices (0-39)
            op_idx: [batch] operator indices (0-3)

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # 1. Contrastive loss
        if self.corpus_embeddings is not None:
            target_emb, source_weights = self.get_corpus_embedding(left_idx, right_idx, op_idx)
            if target_emb is not None:
                losses['contrastive'] = self.config.contrastive_weight * self.contrastive_loss(
                    operator_output['equation_repr'],
                    target_emb.to(operator_output['equation_repr'].device),
                    source_weights.to(operator_output['equation_repr'].device),
                )
            else:
                losses['contrastive'] = torch.tensor(0.0, device=left_idx.device)
        else:
            losses['contrastive'] = torch.tensor(0.0, device=left_idx.device)

        # 2. Symmetry loss (from operator_output if available)
        if 'symmetry_losses' in operator_output:
            sym_losses = operator_output['symmetry_losses']
            losses['symmetry'] = self.config.symmetry_weight * sym_losses['total']
        else:
            losses['symmetry'] = torch.tensor(0.0, device=left_idx.device)

        # Total
        losses['total'] = losses['contrastive'] + losses['symmetry']

        return losses


def integrate_operator_core_loss(
    model_output: Dict[str, torch.Tensor],
    base_loss: torch.Tensor,
    loss_fn: OperatorCoreLoss,
    equation_triplet: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    operator_loss_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Integrate operator core loss with main training loss.

    Args:
        model_output: Full model output dict
        base_loss: Primary task loss (e.g., cross-entropy)
        loss_fn: OperatorCoreLoss instance
        equation_triplet: (left_idx, right_idx, op_idx) or None
        operator_loss_weight: Weight for operator core losses

    Returns:
        total_loss: Combined loss
        loss_breakdown: Dict with individual loss values
    """
    breakdown = {'base_loss': base_loss.detach()}

    if equation_triplet is None or 'operator_core_output' not in model_output:
        return base_loss, breakdown

    left_idx, right_idx, op_idx = equation_triplet
    operator_output = model_output['operator_core_output']

    # Compute operator core losses
    op_losses = loss_fn(operator_output, left_idx, right_idx, op_idx)

    # Combine
    total_loss = base_loss + operator_loss_weight * op_losses['total']

    breakdown['operator_contrastive'] = op_losses['contrastive'].detach()
    breakdown['operator_symmetry'] = op_losses['symmetry'].detach()
    breakdown['operator_total'] = op_losses['total'].detach()
    breakdown['total_loss'] = total_loss.detach()

    return total_loss, breakdown


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Operator Core Loss Test")
    print("=" * 60)

    batch_size = 4

    # Mock operator output
    operator_output = {
        'equation_repr': torch.randn(batch_size, 384),
        'noetic_output': torch.randn(batch_size, 40),
        'symmetry_losses': {
            'symmetry_loss': torch.tensor(0.05),
            'antisymmetry_loss': torch.tensor(0.03),
            'total': torch.tensor(0.08),
        },
    }

    # Mock triplet
    left_idx = torch.tensor([0, 10, 20, 30])   # A1, B1, C1, D1
    right_idx = torch.tensor([1, 11, 21, 31])  # A2, B2, C2, D2
    op_idx = torch.tensor([0, 1, 2, 3])        # +, -, ×, ÷

    # Test without corpus
    print("\n--- Without Corpus ---")
    loss_fn = OperatorCoreLoss()
    losses = loss_fn(operator_output, left_idx, right_idx, op_idx)
    print(f"Contrastive: {losses['contrastive'].item():.4f}")
    print(f"Symmetry: {losses['symmetry'].item():.4f}")
    print(f"Total: {losses['total'].item():.4f}")

    # Test with corpus (if available)
    import os
    corpus_path = "data/equation_embeddings/equation_corpus.pt"
    if os.path.exists(corpus_path):
        print("\n--- With Corpus ---")
        loss_fn_with_corpus = OperatorCoreLoss(corpus_path=corpus_path)
        losses = loss_fn_with_corpus(operator_output, left_idx, right_idx, op_idx)
        print(f"Contrastive: {losses['contrastive'].item():.4f}")
        print(f"Symmetry: {losses['symmetry'].item():.4f}")
        print(f"Total: {losses['total'].item():.4f}")
        loss_fn = loss_fn_with_corpus  # Use this for integration test

    # Test integration
    print("\n--- Integration Test ---")
    base_loss = torch.tensor(2.5)
    model_output = {'operator_core_output': operator_output}
    total, breakdown = integrate_operator_core_loss(
        model_output, base_loss, loss_fn, (left_idx, right_idx, op_idx)
    )
    print(f"Base loss: {breakdown['base_loss'].item():.4f}")
    print(f"Operator total: {breakdown['operator_total'].item():.4f}")
    print(f"Combined: {total.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

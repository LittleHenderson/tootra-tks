"""
STRG-LLM-STK Loss Functions for Training New Components

This module provides specialized loss functions for training:
1. DPS Layer (Novelty Weight computation and classification)
2. Governance Rails (gate decision supervision - rule-based, minimal training)
3. Operator semantics (from existing losses)

Loss Functions:
- DPSClassificationLoss: Cross-entropy for HEAVY/COUNT/NOCOUNT classification
- DPSStateUpdateLoss: MSE for p_max, tokens, cooldown predictions
- NoveltyWeightLoss: MSE for NW = V x I x (1-S) computation
- GovernanceGateLoss: Binary CE for gate decisions (ALLOW/PAUSE/BLOCK)
- HighStakesScoringLoss: MSE for HS = (U x K) x A^2 computation

Author: ML Agent
Date: 2026-01-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class STRGLossConfig:
    """Configuration for STRG loss weights."""
    # DPS losses
    lambda_novelty_weight: float = 1.0
    lambda_classification: float = 1.0
    lambda_state_update: float = 0.5
    lambda_validity: float = 0.3
    lambda_impact: float = 0.3
    lambda_similarity: float = 0.3

    # Governance losses (minimal, rule-based module)
    lambda_gate_decision: float = 0.5
    lambda_hs_score: float = 0.3

    # Regularization
    temperature: float = 1.0  # For classification softmax


# =============================================================================
# DPS LOSSES
# =============================================================================

class NoveltyWeightLoss(nn.Module):
    """
    MSE loss for NW = V x I x (1-S) computation.

    Trains the DPSGatingLayer to correctly compute novelty weight
    from validity, impact, and similarity components.
    """

    def __init__(self, component_weight: float = 0.3):
        """
        Args:
            component_weight: Weight for individual V/I/S component losses
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.component_weight = component_weight

    def forward(
        self,
        pred_novelty: Dict[str, torch.Tensor],
        target_novelty: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute novelty weight loss.

        Args:
            pred_novelty: Dict with predicted 'nw', 'v', 'i', 's' values
            target_novelty: Dict with target 'nw', 'v', 'i', 's' values

        Returns:
            Dict with 'total', 'nw', 'validity', 'impact', 'similarity' losses
        """
        losses = {}

        # Main NW loss
        pred_nw = pred_novelty['nw']
        target_nw = target_novelty['nw']
        losses['nw'] = self.mse(pred_nw, target_nw)

        # Component losses
        if 'v' in pred_novelty and 'v' in target_novelty:
            losses['validity'] = self.mse(pred_novelty['v'], target_novelty['v'])
        else:
            losses['validity'] = torch.tensor(0.0, device=pred_nw.device)

        if 'i' in pred_novelty and 'i' in target_novelty:
            losses['impact'] = self.mse(pred_novelty['i'], target_novelty['i'])
        else:
            losses['impact'] = torch.tensor(0.0, device=pred_nw.device)

        if 's' in pred_novelty and 's' in target_novelty:
            losses['similarity'] = self.mse(pred_novelty['s'], target_novelty['s'])
        else:
            losses['similarity'] = torch.tensor(0.0, device=pred_nw.device)

        # Total: main loss + weighted component losses
        losses['total'] = (
            losses['nw'] +
            self.component_weight * (
                losses['validity'] +
                losses['impact'] +
                losses['similarity']
            )
        )

        return losses


class DPSClassificationLoss(nn.Module):
    """
    Cross-entropy for HEAVY/COUNT/NOCOUNT classification.

    Uses soft labels when available, hard labels otherwise.
    Thresholds:
        HEAVY: NW >= 0.70 (class 0)
        COUNT: 0.45 <= NW < 0.70 (class 1)
        NOCOUNT: NW < 0.45 (class 2)
    """

    def __init__(
        self,
        heavy_threshold: float = 0.70,
        count_threshold: float = 0.45,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.heavy_threshold = heavy_threshold
        self.count_threshold = count_threshold
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='mean'
        )

    def nw_to_class(self, nw: torch.Tensor) -> torch.Tensor:
        """Convert NW values to class indices."""
        # Class 0: HEAVY, Class 1: COUNT, Class 2: NOCOUNT
        classes = torch.full_like(nw, 2, dtype=torch.long)  # Default NOCOUNT
        classes[nw >= self.count_threshold] = 1  # COUNT
        classes[nw >= self.heavy_threshold] = 0  # HEAVY
        return classes

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_nw: torch.Tensor,
        target_class: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification loss.

        Args:
            pred_logits: [batch, 3] predicted class logits
            target_nw: [batch] target novelty weight values
            target_class: Optional [batch] pre-computed class indices

        Returns:
            Dict with 'total', 'accuracy' values
        """
        if target_class is None:
            target_class = self.nw_to_class(target_nw)

        # Apply temperature
        scaled_logits = pred_logits / self.temperature

        # Cross-entropy loss
        loss = self.ce_loss(scaled_logits, target_class)

        # Accuracy for metrics
        pred_class = pred_logits.argmax(dim=-1)
        accuracy = (pred_class == target_class).float().mean()

        return {
            'total': loss,
            'accuracy': accuracy,
        }


class DPSStateUpdateLoss(nn.Module):
    """
    MSE for p_max, tokens, cooldown predictions.

    Trains the model to predict the next DPS state given
    current state and novelty classification.
    """

    def __init__(self, max_depth: int = 5, max_tokens: int = 5):
        super().__init__()
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.mse = nn.MSELoss(reduction='mean')

    def forward(
        self,
        pred_state: Dict[str, torch.Tensor],
        target_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute state update loss.

        Args:
            pred_state: Dict with predicted 'p_max', 'tokens', 'cooldown'
            target_state: Dict with target values

        Returns:
            Dict with 'total', 'p_max', 'tokens', 'cooldown' losses
        """
        losses = {}

        # Normalize targets to [0, 1] for stable training
        # p_max normalized by max_depth
        if 'p_max' in pred_state and 'p_max' in target_state:
            pred_p = pred_state['p_max'] / self.max_depth
            target_p = target_state['p_max'] / self.max_depth
            losses['p_max'] = self.mse(pred_p, target_p)
        else:
            losses['p_max'] = torch.tensor(0.0)

        # tokens normalized by max_tokens
        if 'tokens' in pred_state and 'tokens' in target_state:
            pred_t = pred_state['tokens'] / self.max_tokens
            target_t = target_state['tokens'] / self.max_tokens
            losses['tokens'] = self.mse(pred_t, target_t)
        else:
            losses['tokens'] = torch.tensor(0.0)

        # cooldown normalized by 10 (default cooldown)
        if 'cooldown' in pred_state and 'cooldown' in target_state:
            pred_c = pred_state['cooldown'] / 10.0
            target_c = target_state['cooldown'] / 10.0
            losses['cooldown'] = self.mse(pred_c, target_c)
        else:
            losses['cooldown'] = torch.tensor(0.0)

        # Equal weighting
        losses['total'] = (
            losses['p_max'] + losses['tokens'] + losses['cooldown']
        ) / 3.0

        return losses


class DPSUnlockLoss(nn.Module):
    """
    Binary cross-entropy for unlock prediction.

    Predicts whether an unlock should occur given the current
    state and novelty weight.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(
        self,
        pred_unlock_logit: torch.Tensor,
        target_unlock: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unlock prediction loss.

        Args:
            pred_unlock_logit: [batch] predicted unlock logit
            target_unlock: [batch] binary unlock indicator

        Returns:
            Dict with 'total', 'accuracy', 'precision', 'recall' values
        """
        loss = self.bce(pred_unlock_logit, target_unlock.float())

        # Metrics
        pred_unlock = (torch.sigmoid(pred_unlock_logit) > 0.5).float()
        accuracy = (pred_unlock == target_unlock).float().mean()

        # Precision and recall for positive class (unlock=1)
        true_positives = ((pred_unlock == 1) & (target_unlock == 1)).float().sum()
        pred_positives = (pred_unlock == 1).float().sum()
        actual_positives = (target_unlock == 1).float().sum()

        precision = true_positives / (pred_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)

        return {
            'total': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }


# =============================================================================
# GOVERNANCE LOSSES
# =============================================================================

class GovernanceGateLoss(nn.Module):
    """
    Binary CE for gate decisions (ALLOW/PAUSE/BLOCK).

    Note: GovernanceRails is RULE-BASED and doesn't require training.
    This loss is provided for optional fine-tuning of thresholds
    or for training auxiliary predictors.
    """

    def __init__(self, num_gates: int = 5):
        """
        Args:
            num_gates: Number of governance gates (HS, Irreversibility, Permission, etc.)
        """
        super().__init__()
        self.num_gates = num_gates
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        pred_gate_logits: torch.Tensor,
        target_gates: torch.Tensor,
        gate_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gate decision loss.

        Args:
            pred_gate_logits: [batch, num_gates] predicted gate logits
            target_gates: [batch, num_gates] binary gate decisions (1=BLOCK, 0=ALLOW)
            gate_weights: Optional [num_gates] importance weights per gate

        Returns:
            Dict with 'total' and per-gate losses
        """
        if gate_weights is None:
            gate_weights = torch.ones(self.num_gates, device=pred_gate_logits.device)

        # Per-gate BCE
        gate_losses = self.bce(pred_gate_logits, target_gates.float())  # [batch, num_gates]

        # Weighted mean across gates
        weighted_losses = gate_losses * gate_weights.unsqueeze(0)
        total_loss = weighted_losses.mean()

        # Per-gate statistics
        losses = {'total': total_loss}
        gate_names = ['high_stakes', 'irreversibility', 'permission', 'anti_recursion', 'self_modification']
        for i, name in enumerate(gate_names[:self.num_gates]):
            losses[name] = gate_losses[:, i].mean()

        return losses


class HighStakesScoringLoss(nn.Module):
    """
    MSE for HS = (U x K) x A^2 computation.

    HS scoring determines if an action should be escalated:
        U = uncertainty (verifier disagreement)
        K = impact (cost, irreversibility)
        A = autonomy level of requested action
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(
        self,
        pred_hs: torch.Tensor,
        target_hs: torch.Tensor,
        pred_components: Optional[Dict[str, torch.Tensor]] = None,
        target_components: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute high-stakes scoring loss.

        Args:
            pred_hs: [batch] predicted HS score
            target_hs: [batch] target HS score
            pred_components: Optional dict with 'u', 'k', 'a' predictions
            target_components: Optional dict with 'u', 'k', 'a' targets

        Returns:
            Dict with 'total', 'hs', and optional component losses
        """
        losses = {}

        # Main HS loss
        losses['hs'] = self.mse(pred_hs, target_hs)

        # Component losses if provided
        if pred_components is not None and target_components is not None:
            if 'u' in pred_components and 'u' in target_components:
                losses['uncertainty'] = self.mse(
                    pred_components['u'], target_components['u']
                )
            if 'k' in pred_components and 'k' in target_components:
                losses['impact'] = self.mse(
                    pred_components['k'], target_components['k']
                )
            if 'a' in pred_components and 'a' in target_components:
                losses['autonomy'] = self.mse(
                    pred_components['a'], target_components['a']
                )

        # Total loss
        component_loss = sum(
            v for k, v in losses.items() if k != 'hs'
        ) if len(losses) > 1 else torch.tensor(0.0, device=pred_hs.device)

        losses['total'] = losses['hs'] + 0.3 * component_loss

        return losses


# =============================================================================
# COMBINED DPS TRAINING LOSS
# =============================================================================

class DPSTrainingLoss(nn.Module):
    """
    Combined loss for DPS layer training.

    Combines:
    - Novelty weight regression
    - Classification (HEAVY/COUNT/NOCOUNT)
    - State update prediction
    - Unlock prediction
    """

    def __init__(
        self,
        config: Optional[STRGLossConfig] = None,
        heavy_threshold: float = 0.70,
        count_threshold: float = 0.45,
    ):
        super().__init__()
        self.config = config or STRGLossConfig()

        self.novelty_loss = NoveltyWeightLoss()
        self.classification_loss = DPSClassificationLoss(
            heavy_threshold=heavy_threshold,
            count_threshold=count_threshold,
        )
        self.state_loss = DPSStateUpdateLoss()
        self.unlock_loss = DPSUnlockLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined DPS training loss.

        Args:
            predictions: Dict containing:
                - novelty: Dict with 'nw', 'v', 'i', 's'
                - class_logits: [batch, 3] classification logits
                - next_state: Dict with 'p_max', 'tokens', 'cooldown'
                - unlock_logit: [batch] unlock prediction logit
            targets: Dict containing ground truth values

        Returns:
            Dict with all loss components and 'total'
        """
        losses = {}
        device = targets.get('nw', torch.tensor(0.0)).device

        # 1. Novelty weight loss
        if 'novelty' in predictions and 'nw' in targets:
            nw_losses = self.novelty_loss(
                predictions['novelty'],
                {'nw': targets['nw'], 'v': targets.get('v'),
                 'i': targets.get('i'), 's': targets.get('s')},
            )
            losses['novelty'] = nw_losses['total']
            losses['novelty_nw'] = nw_losses['nw']
        else:
            losses['novelty'] = torch.tensor(0.0, device=device)

        # 2. Classification loss
        if 'class_logits' in predictions and 'nw' in targets:
            cls_losses = self.classification_loss(
                predictions['class_logits'],
                targets['nw'],
                targets.get('novelty_class'),
            )
            losses['classification'] = cls_losses['total']
            losses['classification_accuracy'] = cls_losses['accuracy']
        else:
            losses['classification'] = torch.tensor(0.0, device=device)

        # 3. State update loss
        if 'next_state' in predictions and 'next_state' in targets:
            state_losses = self.state_loss(
                predictions['next_state'],
                targets['next_state'],
            )
            losses['state'] = state_losses['total']
        else:
            losses['state'] = torch.tensor(0.0, device=device)

        # 4. Unlock prediction loss
        if 'unlock_logit' in predictions and 'unlock' in targets:
            unlock_losses = self.unlock_loss(
                predictions['unlock_logit'],
                targets['unlock'],
            )
            losses['unlock'] = unlock_losses['total']
            losses['unlock_accuracy'] = unlock_losses['accuracy']
        else:
            losses['unlock'] = torch.tensor(0.0, device=device)

        # Weighted total
        losses['total'] = (
            self.config.lambda_novelty_weight * losses['novelty'] +
            self.config.lambda_classification * losses['classification'] +
            self.config.lambda_state_update * losses['state'] +
            0.5 * losses['unlock']  # Unlock is optional
        )

        return losses


# =============================================================================
# OPERATOR SEMANTICS LOSSES (Extended from existing)
# =============================================================================

class OperatorSymmetryLoss(nn.Module):
    """
    Enforces operator symmetry/anti-symmetry properties.

    From strg-llm-stk spec:
    - + (Association): Symmetric (A + B = B + A)
    - * (Intensification): Symmetric (A * B = B * A)
    - - (Disassociation): Anti-symmetric (A - B != B - A)
    - / (Opposition): Anti-symmetric (A / B != B / A)
    """

    def __init__(self, symmetric_margin: float = 0.1, antisym_margin: float = 0.2):
        super().__init__()
        self.symmetric_margin = symmetric_margin
        self.antisym_margin = antisym_margin

    def forward(
        self,
        operator_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute symmetry constraint losses.

        Args:
            operator_outputs: Dict with operator results:
                - 'add_ab': A + B result
                - 'add_ba': B + A result (should equal add_ab)
                - 'mul_ab': A * B result
                - 'mul_ba': B * A result (should equal mul_ab)
                - 'sub_ab': A - B result
                - 'sub_ba': B - A result (should differ from sub_ab)
                - 'div_ab': A / B result
                - 'div_ba': B / A result (should differ from div_ab)

        Returns:
            Dict with symmetry losses
        """
        losses = {}
        device = list(operator_outputs.values())[0].device

        # Symmetric operators: + and *
        # Loss = max(0, ||A op B - B op A|| - margin)
        if 'add_ab' in operator_outputs and 'add_ba' in operator_outputs:
            add_diff = (operator_outputs['add_ab'] - operator_outputs['add_ba']).norm(dim=-1).mean()
            losses['add_symmetry'] = F.relu(add_diff - self.symmetric_margin)
        else:
            losses['add_symmetry'] = torch.tensor(0.0, device=device)

        if 'mul_ab' in operator_outputs and 'mul_ba' in operator_outputs:
            mul_diff = (operator_outputs['mul_ab'] - operator_outputs['mul_ba']).norm(dim=-1).mean()
            losses['mul_symmetry'] = F.relu(mul_diff - self.symmetric_margin)
        else:
            losses['mul_symmetry'] = torch.tensor(0.0, device=device)

        # Anti-symmetric operators: - and /
        # Loss = max(0, margin - ||A op B - B op A||) (should be different)
        if 'sub_ab' in operator_outputs and 'sub_ba' in operator_outputs:
            sub_diff = (operator_outputs['sub_ab'] - operator_outputs['sub_ba']).norm(dim=-1).mean()
            losses['sub_antisymmetry'] = F.relu(self.antisym_margin - sub_diff)
        else:
            losses['sub_antisymmetry'] = torch.tensor(0.0, device=device)

        if 'div_ab' in operator_outputs and 'div_ba' in operator_outputs:
            div_diff = (operator_outputs['div_ab'] - operator_outputs['div_ba']).norm(dim=-1).mean()
            losses['div_antisymmetry'] = F.relu(self.antisym_margin - div_diff)
        else:
            losses['div_antisymmetry'] = torch.tensor(0.0, device=device)

        # Total
        losses['total'] = (
            losses['add_symmetry'] +
            losses['mul_symmetry'] +
            losses['sub_antisymmetry'] +
            losses['div_antisymmetry']
        ) / 4.0

        return losses


class OperatorGroundingLoss(nn.Module):
    """
    Cross-entropy loss for operator grounding tasks.

    Trains the model to identify the correct operator for
    a given semantic operation (association, disassociation, etc.)
    """

    def __init__(self, num_operators: int = 4, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_operator: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute operator grounding loss.

        Args:
            pred_logits: [batch, 4] logits for operators (+, -, *, /)
            target_operator: [batch] target operator indices (0-3)

        Returns:
            Dict with 'total', 'accuracy'
        """
        loss = self.ce(pred_logits, target_operator)
        accuracy = (pred_logits.argmax(dim=-1) == target_operator).float().mean()

        return {
            'total': loss,
            'accuracy': accuracy,
        }


class MinimalPairContrastiveLoss(nn.Module):
    """
    Contrastive loss for minimal pair discrimination.

    Trains the model to distinguish between similar but semantically
    different operations (e.g., A + B vs A - B).
    """

    def __init__(self, margin: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute triplet contrastive loss.

        Args:
            anchor_emb: [batch, dim] anchor embeddings
            positive_emb: [batch, dim] positive (same meaning) embeddings
            negative_emb: [batch, dim] negative (different meaning) embeddings

        Returns:
            Dict with 'total', 'pos_sim', 'neg_sim'
        """
        # Normalize
        anchor = F.normalize(anchor_emb, p=2, dim=-1)
        positive = F.normalize(positive_emb, p=2, dim=-1)
        negative = F.normalize(negative_emb, p=2, dim=-1)

        # Cosine similarities
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        neg_sim = (anchor * negative).sum(dim=-1) / self.temperature

        # Triplet margin loss: max(0, margin - pos_sim + neg_sim)
        loss = F.relu(self.margin - pos_sim + neg_sim).mean()

        return {
            'total': loss,
            'pos_sim': pos_sim.mean(),
            'neg_sim': neg_sim.mean(),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "STRGLossConfig",
    # DPS losses
    "NoveltyWeightLoss",
    "DPSClassificationLoss",
    "DPSStateUpdateLoss",
    "DPSUnlockLoss",
    "DPSTrainingLoss",
    # Governance losses
    "GovernanceGateLoss",
    "HighStakesScoringLoss",
    # Operator losses
    "OperatorSymmetryLoss",
    "OperatorGroundingLoss",
    "MinimalPairContrastiveLoss",
]

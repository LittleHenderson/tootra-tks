"""
TKS-LLM Loss Functions — Phase 3 Training Components

This module implements the complete TKS-LLM loss function suite:
    L_total = λ₁·L_task + λ₂·L_rpm + λ₃·L_attractor + λ₄·L_involution
            + λ₅·L_spectral + λ₆·L_cascade

Canonical References:
    - TKS_LLM_Noetic_Mathematics_v1.0.md (involution constraints)
    - TKS_LLM_Architecture_v1.0.md (RPM D/W/P specification)
    - TKS_LLM_Canonical_Validation_v1.0.md (world cascade order)

Author: TKS-LLM Training-Agent
Date: 2025-12-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# CONSTANTS
# =============================================================================

NOETIC_DIM = 10      # Noetics per world
NUM_WORLDS = 4       # A, B, C, D
TOTAL_DIM = 40       # 10 x 4

# Involution pairs: ν_i ∘ ν_j ≈ ν₀ (identity-like)
# From TKS_LLM_Noetic_Mathematics_v1.0.md
INVOLUTION_PAIRS = [
    (2, 3),   # ν₂ (POSITIVE) ∘ ν₃ (NEGATIVE) ≈ ν₀
    (5, 6),   # ν₅ (FEMALE) ∘ ν₆ (MALE) ≈ ν₀
    (8, 9),   # ν₈ (CAUSE) ∘ ν₉ (EFFECT) ≈ ν₀
]


# =============================================================================
# LOSS CONFIGURATION
# =============================================================================

@dataclass
class TKSLossConfig:
    """Configuration for TKS loss function weights and hyperparameters."""

    # Loss weights (λ values)
    lambda_task: float = 1.0
    lambda_rpm: float = 0.5
    lambda_attractor: float = 0.3
    lambda_involution: float = 0.2
    lambda_spectral: float = 0.1
    lambda_cascade: float = 0.2

    # Involution loss hyperparameters
    involution_tolerance: float = 0.01   # How close to identity

    # Attractor loss hyperparameters
    variance_reduction_target: float = 0.5   # Target variance ratio
    convergence_bonus_weight: float = 0.1    # Bonus for fast convergence

    # Spectral loss hyperparameters
    spectral_radius_target: float = 0.9      # Target max eigenvalue < 1

    # Cascade loss hyperparameters
    cascade_smoothness: float = 0.5          # Weight for smooth flow

    # RPM loss hyperparameters
    rpm_margin: float = 0.1                  # Margin for correct D/W/P

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'lambda_task': self.lambda_task,
            'lambda_rpm': self.lambda_rpm,
            'lambda_attractor': self.lambda_attractor,
            'lambda_involution': self.lambda_involution,
            'lambda_spectral': self.lambda_spectral,
            'lambda_cascade': self.lambda_cascade,
        }


# =============================================================================
# INDIVIDUAL LOSS COMPONENTS
# =============================================================================

class TaskLoss(nn.Module):
    """
    L_task: Primary task prediction loss.

    For element prediction: CrossEntropyLoss on predicted vs target element.
    For language modeling: CrossEntropyLoss on predicted vs target token.

    Supports multiple task types:
        - 'element': Predict next TKS element (40-way classification)
        - 'token': Standard language modeling (vocab-size classification)
        - 'foundation': Predict primary Foundation (7-way classification)
    """

    def __init__(self, task_type: str = 'token', label_smoothing: float = 0.0):
        super().__init__()
        self.task_type = task_type
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute task loss.

        Args:
            logits: [batch, seq, vocab] or [batch, vocab] predictions
            targets: [batch, seq] or [batch] target indices
            mask: Optional [batch, seq] mask for valid positions

        Returns:
            Scalar loss tensor
        """
        if logits.dim() == 3:
            # Sequence prediction: reshape for cross entropy
            batch, seq, vocab = logits.shape
            logits_flat = logits.view(-1, vocab)
            targets_flat = targets.view(-1)

            if mask is not None:
                mask_flat = mask.view(-1)
                # Only compute loss on valid positions
                valid_indices = mask_flat.nonzero(as_tuple=True)[0]
                if valid_indices.numel() == 0:
                    return torch.tensor(0.0, device=logits.device)
                logits_flat = logits_flat[valid_indices]
                targets_flat = targets_flat[valid_indices]

            return self.ce_loss(logits_flat, targets_flat)
        else:
            # Single prediction
            return self.ce_loss(logits, targets)


class RPMLoss(nn.Module):
    """
    L_rpm: RPM Desire/Wisdom/Power alignment loss.

    Trains the D/W/P evaluators to match labeled satisfaction scores.

    From TKS_LLM_Architecture_v1.0.md:
        - D ∈ [0,1]: Does thought serve goal?
        - W ∈ [0,1]: Is thought informed?
        - P ∈ [0,1]: Can thought be actualized?
        - Gate = D × W × P
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(
        self,
        dwp_scores: torch.Tensor,
        dwp_labels: torch.Tensor,
        target_foundation: Optional[torch.Tensor] = None,
        gate_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute RPM alignment loss.

        Args:
            dwp_scores: [batch, seq, 7, 3] predicted D/W/P per Foundation
            dwp_labels: [batch, seq, 3] or [batch, 3] ground truth D/W/P
            target_foundation: [batch] optional Foundation index for focused loss
            gate_labels: [batch, seq] optional gate value labels

        Returns:
            Dict with 'total', 'desire', 'wisdom', 'power', 'gate' losses
        """
        batch_size = dwp_scores.shape[0]
        device = dwp_scores.device

        losses = {}

        if dwp_labels.dim() == 2:
            # [batch, 3] -> expand to match sequence
            seq_len = dwp_scores.shape[1]
            dwp_labels = dwp_labels.unsqueeze(1).expand(-1, seq_len, -1)

        if target_foundation is not None:
            # Focus on specific Foundation
            # target_foundation: [batch]
            batch_indices = torch.arange(batch_size, device=device)
            if dwp_scores.dim() == 4:  # [batch, seq, 7, 3]
                seq_len = dwp_scores.shape[1]
                # Gather the target foundation's scores for each batch
                pred_dwp = dwp_scores[batch_indices.unsqueeze(1).expand(-1, seq_len),
                                      torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1),
                                      target_foundation.unsqueeze(1).expand(-1, seq_len), :]
            else:
                pred_dwp = dwp_scores[batch_indices, target_foundation, :]
        else:
            # Average across all Foundations
            pred_dwp = dwp_scores.mean(dim=-2)  # [batch, seq, 3] or [batch, 3]

        # Compute individual D/W/P losses
        losses['desire'] = self.bce_loss(pred_dwp[..., 0], dwp_labels[..., 0])
        losses['wisdom'] = self.bce_loss(pred_dwp[..., 1], dwp_labels[..., 1])
        losses['power'] = self.bce_loss(pred_dwp[..., 2], dwp_labels[..., 2])

        # Gate loss: predicted gate vs labeled gate
        if gate_labels is not None:
            pred_gate = pred_dwp[..., 0] * pred_dwp[..., 1] * pred_dwp[..., 2]
            losses['gate'] = self.bce_loss(pred_gate, gate_labels)
        else:
            # Compute expected gate from labels
            label_gate = dwp_labels[..., 0] * dwp_labels[..., 1] * dwp_labels[..., 2]
            pred_gate = pred_dwp[..., 0] * pred_dwp[..., 1] * pred_dwp[..., 2]
            losses['gate'] = self.bce_loss(pred_gate, label_gate)

        # Total RPM loss (equal weighting)
        losses['total'] = (losses['desire'] + losses['wisdom'] +
                          losses['power'] + losses['gate']) / 4.0

        return losses


class AttractorLoss(nn.Module):
    """
    L_attractor: Attractor convergence and stability loss.

    From TKS_LLM_Noetic_Mathematics_v1.0.md Section 7:
        1. Variance should reduce over iterations
        2. Fixed-point residual should approach zero
        3. Faster convergence is preferred

    Components:
        - Variance reduction: final_var / initial_var < target
        - Fixed-point residual: ||T(x*) - x*|| should be small
        - Convergence speed: fewer iterations is better
    """

    def __init__(
        self,
        variance_target: float = 0.5,
        convergence_bonus: float = 0.1,
    ):
        super().__init__()
        self.variance_target = variance_target
        self.convergence_bonus = convergence_bonus

    def forward(
        self,
        attractor_output: Dict[str, torch.Tensor],
        input_state: torch.Tensor,
        attractor_layer: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attractor convergence loss.

        Args:
            attractor_output: Dict from AttractorComputationLayer forward
            input_state: Original input to attractor layer
            attractor_layer: The attractor module (for fixed-point check)

        Returns:
            Dict with 'total', 'variance', 'residual', 'convergence' losses
        """
        losses = {}
        device = input_state.device

        attractor = attractor_output['attractor']

        # 1. Variance reduction loss
        # We want variance to reduce during iteration
        initial_var = input_state.var(dim=-1).mean()
        final_var = attractor.var(dim=-1).mean()

        var_ratio = final_var / (initial_var + 1e-8)
        # Penalize if variance ratio exceeds target
        losses['variance'] = F.relu(var_ratio - self.variance_target)

        # 2. Fixed-point residual loss
        # Apply Hutchinson operator one more time and check residual
        with torch.no_grad():
            next_iter = attractor_layer._apply_hutchinson_operator(attractor)
        residual = (next_iter - attractor).abs().mean()
        losses['residual'] = residual

        # 3. Convergence speed bonus (negative loss for fast convergence)
        max_iter = attractor_layer.max_iterations
        iterations_used = attractor_output['iterations']
        # Normalized iteration count (0 = converged immediately, 1 = max iterations)
        iter_ratio = iterations_used / max_iter
        losses['convergence'] = self.convergence_bonus * iter_ratio

        # Total
        losses['total'] = losses['variance'] + losses['residual'] + losses['convergence']

        return losses


class InvolutionLoss(nn.Module):
    """
    L_involution: Noetic algebra involution constraints.

    From TKS_LLM_Noetic_Mathematics_v1.0.md:
        Involution pairs should compose to near-identity:
        - ν₂ ∘ ν₃ ≈ ν₀ (POSITIVE ∘ NEGATIVE ≈ IDEA)
        - ν₅ ∘ ν₆ ≈ ν₀ (FEMALE ∘ MALE ≈ IDEA)
        - ν₈ ∘ ν₉ ≈ ν₀ (CAUSE ∘ EFFECT ≈ IDEA)

    Loss = Σ ||M_i @ M_j - M₀||²_F / 3
    where M_k is the weight matrix for noetic ν_k.
    """

    def __init__(self, tolerance: float = 0.01):
        super().__init__()
        self.tolerance = tolerance
        self.pairs = INVOLUTION_PAIRS

    def forward(self, noetic_processor: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute involution constraint loss.

        Args:
            noetic_processor: NoeticProcessor module with .noetics ModuleList

        Returns:
            Dict with 'total' and individual pair losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(noetic_processor.parameters()).device)

        # Get weight matrices
        noetics = noetic_processor.noetics
        M0 = noetics[0].weight  # ν₀ reference (should be near-identity)

        for i, (a, b) in enumerate(self.pairs):
            Ma = noetics[a].weight
            Mb = noetics[b].weight

            # Composition: Ma @ Mb should approximate M0
            composition = Ma @ Mb

            # Frobenius norm of difference
            diff = composition - M0
            pair_loss = torch.norm(diff, p='fro') ** 2

            pair_name = f'nu{a}_nu{b}'
            losses[pair_name] = pair_loss
            total_loss = total_loss + pair_loss

        # Average over pairs
        losses['total'] = total_loss / len(self.pairs)

        return losses


class SpectralLoss(nn.Module):
    """
    L_spectral: Spectral radius constraints on noetic operators.

    From TKS_LLM_Noetic_Mathematics_v1.0.md:
        - Contraction mappings require spectral radius ρ(M) < 1
        - ν₀ should have eigenvalues near 1 (identity-like)
        - Other noetics should have bounded eigenvalues

    Uses power iteration to estimate spectral radius efficiently.
    """

    def __init__(
        self,
        target_radius: float = 0.9,
        power_iterations: int = 10,
    ):
        super().__init__()
        self.target_radius = target_radius
        self.power_iterations = power_iterations

    def _estimate_spectral_radius(self, M: torch.Tensor) -> torch.Tensor:
        """
        Estimate spectral radius using power iteration.

        Args:
            M: [dim, dim] weight matrix

        Returns:
            Estimated spectral radius (scalar)
        """
        dim = M.shape[0]
        v = torch.randn(dim, device=M.device)
        v = v / v.norm()

        for _ in range(self.power_iterations):
            Mv = M @ v
            v = Mv / (Mv.norm() + 1e-8)

        # Rayleigh quotient
        Mv = M @ v
        eigenvalue = (v @ Mv).abs()

        return eigenvalue

    def forward(
        self,
        noetic_processor: nn.Module,
        attractor_layer: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spectral constraint loss.

        Args:
            noetic_processor: NoeticProcessor with noetic weight matrices
            attractor_layer: Optional AttractorComputationLayer

        Returns:
            Dict with spectral radius losses per operator
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(noetic_processor.parameters()).device)

        # Noetic operators
        for k, noetic in enumerate(noetic_processor.noetics):
            M = noetic.weight
            rho = self._estimate_spectral_radius(M)

            if k == 0:
                # ν₀ should be near-identity: eigenvalues near 1
                loss_k = (rho - 1.0).abs()
            else:
                # Other noetics: penalize if exceeds target
                loss_k = F.relu(rho - self.target_radius)

            losses[f'nu{k}_spectral'] = loss_k
            total_loss = total_loss + loss_k

        # Attractor contraction maps
        if attractor_layer is not None:
            for i, cmap in enumerate(attractor_layer.contraction_maps):
                M = cmap.weight
                rho = self._estimate_spectral_radius(M)

                # Contraction: must have ρ < 1
                loss_i = F.relu(rho - self.target_radius)
                losses[f'attractor_map{i}_spectral'] = loss_i
                total_loss = total_loss + loss_i

        num_terms = len(losses)
        losses['total'] = total_loss / max(num_terms, 1)

        return losses


class CascadeLoss(nn.Module):
    """
    L_cascade: World cascade flow constraint (A → B → C → D).

    From TKS v7.4 canonical order:
        Information flows A (Spiritual) → B (Mental) → C (Emotional) → D (Physical)

    Enforces:
        1. Temporal/causal ordering in attention patterns
        2. Gradient magnitude ordering: |∇A| ≥ |∇B| ≥ |∇C| ≥ |∇D|
        3. Activation energy ordering for world transitions
    """

    def __init__(self, smoothness: float = 0.5):
        super().__init__()
        self.smoothness = smoothness

    def forward(
        self,
        noetic_embedding: torch.Tensor,
        world_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cascade flow loss.

        Args:
            noetic_embedding: [batch, seq, 40] full noetic tensor
            world_embeddings: Optional dict with 'A', 'B', 'C', 'D' tensors

        Returns:
            Dict with cascade flow losses
        """
        losses = {}
        batch, seq, dim = noetic_embedding.shape
        device = noetic_embedding.device

        # Extract worlds from full embedding
        A = noetic_embedding[..., 0:10]
        B = noetic_embedding[..., 10:20]
        C = noetic_embedding[..., 20:30]
        D = noetic_embedding[..., 30:40]

        worlds = [A, B, C, D]
        world_names = ['A', 'B', 'C', 'D']

        # 1. Energy ordering: higher worlds should have higher activation energy
        # Energy = ||x||² (L2 norm squared)
        energies = [w.norm(dim=-1).mean() for w in worlds]

        order_loss = torch.tensor(0.0, device=device)
        for i in range(len(energies) - 1):
            # Penalize if lower world has more energy than higher world
            violation = F.relu(energies[i+1] - energies[i])
            order_loss = order_loss + violation
            losses[f'energy_{world_names[i]}_{world_names[i+1]}'] = violation

        losses['energy_order'] = order_loss / 3.0

        # 2. Flow smoothness: adjacent worlds should have correlated activations
        smooth_loss = torch.tensor(0.0, device=device)
        for i in range(len(worlds) - 1):
            # Cosine similarity between adjacent worlds
            w1 = worlds[i].view(-1, 10)
            w2 = worlds[i+1].view(-1, 10)

            cos_sim = F.cosine_similarity(w1, w2, dim=-1).mean()
            # Penalize low correlation (information should flow)
            flow_penalty = F.relu(self.smoothness - cos_sim)
            smooth_loss = smooth_loss + flow_penalty
            losses[f'flow_{world_names[i]}_{world_names[i+1]}'] = flow_penalty

        losses['flow_smooth'] = smooth_loss / 3.0

        # Total
        losses['total'] = losses['energy_order'] + losses['flow_smooth']

        return losses


# =============================================================================
# COMBINED TKS LOSS
# =============================================================================

class TKSLoss(nn.Module):
    """
    Combined TKS-LLM loss function.

    L_total = λ₁·L_task + λ₂·L_rpm + λ₃·L_attractor + λ₄·L_involution
            + λ₅·L_spectral + λ₆·L_cascade

    This is the primary loss function for training TKS-LLM models.
    Each component enforces different aspects of TKS canonical constraints.

    Usage:
        config = TKSLossConfig(lambda_task=1.0, lambda_rpm=0.5, ...)
        loss_fn = TKSLoss(config)

        # During training:
        loss_dict = loss_fn(
            pipeline_output=model_out,
            targets=batch['targets'],
            pipeline=model,
        )
        loss_dict['total'].backward()
    """

    def __init__(self, config: Optional[TKSLossConfig] = None):
        super().__init__()

        self.config = config or TKSLossConfig()

        # Initialize component losses
        self.task_loss = TaskLoss()
        self.rpm_loss = RPMLoss(margin=self.config.rpm_margin)
        self.attractor_loss = AttractorLoss(
            variance_target=self.config.variance_reduction_target,
            convergence_bonus=self.config.convergence_bonus_weight,
        )
        self.involution_loss = InvolutionLoss(
            tolerance=self.config.involution_tolerance
        )
        self.spectral_loss = SpectralLoss(
            target_radius=self.config.spectral_radius_target
        )
        self.cascade_loss = CascadeLoss(
            smoothness=self.config.cascade_smoothness
        )

    def forward(
        self,
        pipeline_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        pipeline: nn.Module,
        dwp_labels: Optional[torch.Tensor] = None,
        target_foundation: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        input_state: Optional[torch.Tensor] = None,
        compute_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined TKS loss.

        Args:
            pipeline_output: Dict from TKSLLMCorePipeline.forward()
                Must contain: 'logits', 'gated_output', 'rpm_gate', 'trace'
            targets: [batch, seq] target token/element indices
            pipeline: The TKSLLMCorePipeline module
            dwp_labels: Optional [batch, seq, 3] D/W/P ground truth
            target_foundation: Optional [batch] Foundation indices
            mask: Optional [batch, seq] valid position mask
            input_state: Optional [batch, seq, 40] for attractor loss
            compute_all: If False, only compute task loss

        Returns:
            Dict with all loss components and 'total'
        """
        losses = {}
        device = targets.device

        # 1. Task Loss (always computed)
        logits = pipeline_output['logits']
        l_task = self.task_loss(logits, targets, mask)
        losses['task'] = l_task

        if not compute_all:
            losses['total'] = l_task
            return losses

        # 2. RPM Loss (if labels provided)
        if dwp_labels is not None and 'trace' in pipeline_output and pipeline_output['trace'] is not None:
            dwp_scores = pipeline_output['trace'].get('dwp_scores')
            if dwp_scores is not None:
                rpm_losses = self.rpm_loss(
                    dwp_scores, dwp_labels,
                    target_foundation=target_foundation
                )
                losses['rpm'] = rpm_losses['total']
                losses['rpm_desire'] = rpm_losses['desire']
                losses['rpm_wisdom'] = rpm_losses['wisdom']
                losses['rpm_power'] = rpm_losses['power']
        else:
            losses['rpm'] = torch.tensor(0.0, device=device)

        # 3. Attractor Loss
        if 'trace' in pipeline_output and pipeline_output['trace'] is not None:
            attractor_out = {
                'attractor': pipeline_output['trace'].get('attractor'),
                'converged': pipeline_output.get('attractor_converged', False),
                'iterations': pipeline_output['trace'].get('attractor_iterations', 10),
            }

            if attractor_out['attractor'] is not None:
                # Use attended state as input if not provided
                inp = input_state if input_state is not None else \
                      pipeline_output['trace'].get('attended', attractor_out['attractor'])

                if inp is not None:
                    attr_losses = self.attractor_loss(
                        attractor_out, inp, pipeline.attractor
                    )
                    losses['attractor'] = attr_losses['total']
                    losses['attractor_variance'] = attr_losses['variance']
                    losses['attractor_residual'] = attr_losses['residual']
                else:
                    losses['attractor'] = torch.tensor(0.0, device=device)
            else:
                losses['attractor'] = torch.tensor(0.0, device=device)
        else:
            losses['attractor'] = torch.tensor(0.0, device=device)

        # 4. Involution Loss
        inv_losses = self.involution_loss(pipeline.processor)
        losses['involution'] = inv_losses['total']
        for key, val in inv_losses.items():
            if key != 'total':
                losses[f'inv_{key}'] = val

        # 5. Spectral Loss
        spec_losses = self.spectral_loss(
            pipeline.processor,
            pipeline.attractor
        )
        losses['spectral'] = spec_losses['total']

        # 6. Cascade Loss
        if 'trace' in pipeline_output and pipeline_output['trace'] is not None:
            embedding = pipeline_output['trace'].get('embedding')
            if embedding is not None:
                casc_losses = self.cascade_loss(embedding)
                losses['cascade'] = casc_losses['total']
            else:
                losses['cascade'] = torch.tensor(0.0, device=device)
        else:
            losses['cascade'] = torch.tensor(0.0, device=device)

        # Compute weighted total
        total = (
            self.config.lambda_task * losses['task'] +
            self.config.lambda_rpm * losses['rpm'] +
            self.config.lambda_attractor * losses['attractor'] +
            self.config.lambda_involution * losses['involution'] +
            self.config.lambda_spectral * losses['spectral'] +
            self.config.lambda_cascade * losses['cascade']
        )

        losses['total'] = total
        losses['config'] = self.config.to_dict()

        return losses


# =============================================================================
# CURRICULUM LOSS SCHEDULER
# =============================================================================

class CurriculumLossScheduler:
    """
    Adjusts loss weights during training curriculum.

    Training Stages (from Phase 3 plan):
        Stage 1: Focus on task loss (element prediction)
        Stage 2: Add involution constraints
        Stage 3: Add RPM alignment
        Stage 4: Add attractor stability
        Stage 5: Full pipeline with all losses
    """

    def __init__(
        self,
        base_config: Optional[TKSLossConfig] = None,
        warmup_steps: int = 1000,
    ):
        self.base_config = base_config or TKSLossConfig()
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.current_stage = 1

    def get_config_for_stage(self, stage: int) -> TKSLossConfig:
        """Get loss configuration for a specific training stage."""

        if stage == 1:
            # Stage 1: Task only
            return TKSLossConfig(
                lambda_task=1.0,
                lambda_rpm=0.0,
                lambda_attractor=0.0,
                lambda_involution=0.0,
                lambda_spectral=0.0,
                lambda_cascade=0.0,
            )
        elif stage == 2:
            # Stage 2: Task + Involution
            return TKSLossConfig(
                lambda_task=1.0,
                lambda_rpm=0.0,
                lambda_attractor=0.0,
                lambda_involution=0.2,
                lambda_spectral=0.1,
                lambda_cascade=0.0,
            )
        elif stage == 3:
            # Stage 3: Task + Involution + RPM
            return TKSLossConfig(
                lambda_task=1.0,
                lambda_rpm=0.3,
                lambda_attractor=0.0,
                lambda_involution=0.2,
                lambda_spectral=0.1,
                lambda_cascade=0.1,
            )
        elif stage == 4:
            # Stage 4: Task + Involution + RPM + Attractor
            return TKSLossConfig(
                lambda_task=1.0,
                lambda_rpm=0.4,
                lambda_attractor=0.2,
                lambda_involution=0.2,
                lambda_spectral=0.1,
                lambda_cascade=0.2,
            )
        else:
            # Stage 5+: Full pipeline
            return self.base_config

    def step(self, global_step: int, stage: int) -> TKSLossConfig:
        """
        Update scheduler and return current config.

        Args:
            global_step: Current training step
            stage: Current curriculum stage (1-5)

        Returns:
            TKSLossConfig for current state
        """
        self.current_step = global_step
        self.current_stage = stage
        return self.get_config_for_stage(stage)


# =============================================================================
# TESTING
# =============================================================================

def run_loss_tests():
    """Test all loss components."""
    print("=" * 70)
    print("TKS Loss Functions — Component Tests")
    print("=" * 70)

    # Import pipeline for testing
    import sys
    sys.path.insert(0, '..')

    try:
        from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    except ImportError:
        print("Note: Running standalone test without full pipeline")
        TOTAL_DIM = 40

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    batch, seq, vocab = 2, 8, 100

    # Test 1: TaskLoss
    print("\n--- Test 1: TaskLoss ---")
    task_loss = TaskLoss()
    logits = torch.randn(batch, seq, vocab)
    targets = torch.randint(0, vocab, (batch, seq))
    l_task = task_loss(logits, targets)
    print(f"  Task loss: {l_task.item():.4f}")

    # Test 2: RPMLoss
    print("\n--- Test 2: RPMLoss ---")
    rpm_loss = RPMLoss()
    dwp_scores = torch.rand(batch, seq, 7, 3)
    dwp_labels = torch.rand(batch, seq, 3)
    rpm_losses = rpm_loss(dwp_scores, dwp_labels)
    print(f"  RPM total: {rpm_losses['total'].item():.4f}")
    print(f"  Desire: {rpm_losses['desire'].item():.4f}")
    print(f"  Wisdom: {rpm_losses['wisdom'].item():.4f}")
    print(f"  Power: {rpm_losses['power'].item():.4f}")

    # Test 3: InvolutionLoss (requires NoeticProcessor)
    print("\n--- Test 3: InvolutionLoss ---")
    try:
        from tks_llm_core import NoeticProcessor
        processor = NoeticProcessor(dim=TOTAL_DIM)
        inv_loss = InvolutionLoss()
        inv_losses = inv_loss(processor)
        print(f"  Involution total: {inv_losses['total'].item():.4f}")
        for key, val in inv_losses.items():
            if key != 'total':
                print(f"  {key}: {val.item():.4f}")
    except ImportError:
        print("  Skipped (requires tks_llm_core)")

    # Test 4: SpectralLoss
    print("\n--- Test 4: SpectralLoss ---")
    try:
        from tks_llm_core import NoeticProcessor
        processor = NoeticProcessor(dim=TOTAL_DIM)
        spec_loss = SpectralLoss()
        spec_losses = spec_loss(processor)
        print(f"  Spectral total: {spec_losses['total'].item():.4f}")
    except ImportError:
        print("  Skipped (requires tks_llm_core)")

    # Test 5: CascadeLoss
    print("\n--- Test 5: CascadeLoss ---")
    casc_loss = CascadeLoss()
    noetic_emb = torch.randn(batch, seq, TOTAL_DIM)
    casc_losses = casc_loss(noetic_emb)
    print(f"  Cascade total: {casc_losses['total'].item():.4f}")
    print(f"  Energy order: {casc_losses['energy_order'].item():.4f}")
    print(f"  Flow smooth: {casc_losses['flow_smooth'].item():.4f}")

    # Test 6: Full TKSLoss
    print("\n--- Test 6: Full TKSLoss ---")
    try:
        from tks_llm_core_v2 import TKSLLMCorePipeline

        config = TKSLossConfig()
        tks_loss = TKSLoss(config)

        pipeline = TKSLLMCorePipeline(vocab_size=vocab)
        tokens = torch.randint(0, vocab, (batch, seq))

        out = pipeline(tokens, return_full_trace=True)
        dwp_labels = torch.rand(batch, seq, 3)

        losses = tks_loss(
            pipeline_output=out,
            targets=tokens,  # Self-prediction for test
            pipeline=pipeline,
            dwp_labels=dwp_labels,
        )

        print(f"  Total loss: {losses['total'].item():.4f}")
        for key, val in losses.items():
            if key not in ['total', 'config'] and isinstance(val, torch.Tensor):
                print(f"  {key}: {val.item():.4f}")

    except ImportError as e:
        print(f"  Skipped (requires full pipeline): {e}")

    # Test 7: CurriculumScheduler
    print("\n--- Test 7: CurriculumLossScheduler ---")
    scheduler = CurriculumLossScheduler()
    for stage in range(1, 6):
        cfg = scheduler.get_config_for_stage(stage)
        print(f"  Stage {stage}: task={cfg.lambda_task:.1f}, rpm={cfg.lambda_rpm:.1f}, "
              f"attr={cfg.lambda_attractor:.1f}, inv={cfg.lambda_involution:.1f}")

    print("\n" + "=" * 70)
    print("LOSS TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_loss_tests()

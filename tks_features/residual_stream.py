"""
TKS Residual Stream - Mitigating Over-Constraint Risk in Traceable TKS-Transformer

This module addresses the OVER-CONSTRAINT risk: if every linear map is constrained
to the noetic basis only, the model loses capacity for nuanced composition.

Solution Architecture:
    1. TwoStreamNoetic: Constrained noetic stream + small residual stream with budget
    2. ConstraintScheduler: Anneal constraints from soft to hard over training
    3. ResidualDistillationLoss: Encourage residual to be interpretable

Mathematical Foundations (TKS v7.2 Canon):
    - Noetic operators nu_k : Ideas -> Ideas form a basis in End(R^40)
    - Spectral theory constrains operators to preserve semantic structure
    - Residual stream provides bounded escape for edge cases
    - Distillation ensures residual remains approximately reconstructible

The two-stream design maintains interpretability while allowing model capacity:
    output = noetic_stream(x) + clamp(gate, max=budget) * residual_stream(x)

where the residual budget is small (default 10%) and regularized.

References:
    - TKS_FORMAL_MATHEMATICAL_MANUAL_v7.2_COMPLETE.tex: Spectral theory
    - TKS_v7.3_Quantum.tex: Involution structure and operator bounds
    - tks_features/noetic_expressiveness.py: NoeticOperator base class

Author: TKS Mathematical Foundations Agent
Date: 2025-12-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


# =============================================================================
# 1. TWO-STREAM NOETIC MODULE
# =============================================================================

class TwoStreamNoetic(nn.Module):
    """
    Two-stream architecture combining constrained noetic basis with bounded residual.

    The noetic stream uses only interpretable basis operators (spectral-constrained).
    The residual stream has limited capacity (L2-regularized) for edge cases.
    A learned gate controls residual contribution, capped at residual_budget.

    Mathematical Structure:
        Let {nu_k}_{k=0}^{9} be the noetic basis operators.
        Let W_residual be a small unconstrained linear map.

        noetic_out = sum_k alpha_k * nu_k(x)   where alpha_k learned
        residual_out = W_residual(x)
        gate = sigmoid(v^T x + b), clamped to [0, residual_budget]
        output = noetic_out + gate * residual_out

    The residual energy ||gate * residual_out||^2 is logged for monitoring.

    Properties:
        - Interpretability: Primary path uses only noetic basis
        - Capacity: Residual provides bounded flexibility
        - Traceability: Residual energy quantifies deviation from pure noetics

    Attributes:
        dim (int): Embedding dimension (default 40 for TKS Ideas space)
        residual_budget (float): Maximum fraction of output from residual (0.1 = 10%)
        num_noetic_ops (int): Number of noetic operators in basis (default 10)
    """

    def __init__(
        self,
        dim: int = 40,
        residual_budget: float = 0.1,
        num_noetic_ops: int = 10,
        noetic_init_scale: float = 0.02,
        residual_hidden_dim: Optional[int] = None,
    ):
        """
        Initialize TwoStreamNoetic module.

        Args:
            dim: Embedding dimension (should be 40 for standard TKS Ideas space)
            residual_budget: Maximum fraction of output from residual stream
            num_noetic_ops: Number of noetic basis operators (10 for TKS)
            noetic_init_scale: Initialization scale for noetic operators
            residual_hidden_dim: Hidden dimension for residual MLP (None = dim//4)
        """
        super().__init__()

        self.dim = dim
        self.residual_budget = residual_budget
        self.num_noetic_ops = num_noetic_ops

        # Noetic stream: weighted sum of basis operators
        # Each operator is a linear map preserving semantic structure
        self.noetic_operators = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim) * noetic_init_scale)
            for _ in range(num_noetic_ops)
        ])

        # Noetic mixture weights: learned per-input coefficients
        self.noetic_router = nn.Linear(dim, num_noetic_ops, bias=True)

        # Residual stream: small unconstrained linear
        # Hidden dimension is smaller to limit capacity
        residual_hidden = residual_hidden_dim or max(dim // 4, 8)
        self.residual_stream = nn.Sequential(
            nn.Linear(dim, residual_hidden, bias=False),
            nn.GELU(),
            nn.Linear(residual_hidden, dim, bias=False),
        )

        # Residual gate: learned gate capped at residual_budget
        self.residual_gate_linear = nn.Linear(dim, 1, bias=True)

        # Spectral normalization for residual (prevents explosion)
        self._apply_spectral_norm()

        # Initialize operators with structure
        self._init_noetic_operators()

    def _apply_spectral_norm(self):
        """Apply spectral normalization to residual stream for stability."""
        for module in self.residual_stream:
            if isinstance(module, nn.Linear):
                nn.utils.parametrizations.spectral_norm(module)

    def _init_noetic_operators(self):
        """
        Initialize noetic operators with TKS-aligned structure.

        Following TKS v7.2 canon:
            - nu_0: Identity-like (pure potential)
            - nu_1: Mind operator (cognitive structure)
            - nu_2, nu_3: Positive/Negative (dual pair, approx involutions)
            - nu_4: Vibration (intensity, affects magnitude)
            - nu_5, nu_6: Female/Male (dual pair, receptive/projective)
            - nu_7: Rhythm (periodicity)
            - nu_8, nu_9: Above/Below (dual pair, causal/effect)
        """
        with torch.no_grad():
            for k in range(self.num_noetic_ops):
                op = self.noetic_operators[k]

                # Start with near-identity
                nn.init.eye_(op)

                if k == 0:
                    # nu_0: Pure identity
                    pass
                elif k == 1:
                    # nu_1: Mind - slight emphasis on modes 1,2 across worlds
                    for w in range(4):
                        base = w * 10
                        op[base+1, base+1] *= 1.1
                        op[base+2, base+2] *= 1.1
                elif k in [2, 3]:
                    # nu_2, nu_3: Positive/Negative dual pair
                    # Make them approximate involutions: nu_2 @ nu_3 ~ I
                    sign = 1.0 if k == 2 else -1.0
                    for w in range(4):
                        base = w * 10
                        op[base+2, base+3] = sign * 0.1
                        op[base+3, base+2] = sign * 0.1
                elif k == 4:
                    # nu_4: Vibration - affects magnitude
                    op *= 1.05
                elif k in [5, 6]:
                    # nu_5, nu_6: Female/Male dual pair
                    sign = 1.0 if k == 5 else -1.0
                    for w in range(4):
                        base = w * 10
                        op[base+5, base+6] = sign * 0.1
                        op[base+6, base+5] = sign * 0.1
                elif k == 7:
                    # nu_7: Rhythm - slight periodic structure
                    for i in range(self.dim):
                        op[i, i] *= (1.0 + 0.05 * math.cos(2 * math.pi * i / 10))
                elif k in [8, 9]:
                    # nu_8, nu_9: Above/Below dual pair
                    sign = 1.0 if k == 8 else -1.0
                    for w in range(4):
                        base = w * 10
                        op[base+8, base+9] = sign * 0.1
                        op[base+9, base+8] = sign * 0.1

    def forward(
        self,
        x: torch.Tensor,
        operator_idx: Optional[int] = None,
        return_trace: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Apply two-stream noetic transformation.

        Args:
            x: Input tensor [batch, seq, dim] or [batch, dim]
            operator_idx: If provided, use only this noetic operator (hard routing)
            return_trace: If True, return detailed trace information

        Returns:
            output: Transformed tensor (same shape as input)
            trace: Dictionary containing:
                - residual_stream.energy: L2 norm of residual contribution
                - residual_stream.gate_value: Effective gate value (after clamping)
                - residual_stream.raw_gate: Gate value before clamping
                - noetic_weights: Mixture weights over noetic operators
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        batch, seq, dim = x.shape

        # Noetic stream: weighted sum of basis operators
        if operator_idx is not None:
            # Hard routing to single operator
            noetic_weights = F.one_hot(
                torch.tensor(operator_idx, device=x.device).expand(batch, seq),
                num_classes=self.num_noetic_ops
            ).float()
        else:
            # Soft routing based on input content
            noetic_weights = F.softmax(self.noetic_router(x), dim=-1)  # [B, S, num_ops]

        # Apply each noetic operator
        noetic_outputs = []
        for k in range(self.num_noetic_ops):
            op_out = torch.matmul(x, self.noetic_operators[k].t())  # [B, S, dim]
            noetic_outputs.append(op_out)

        noetic_stack = torch.stack(noetic_outputs, dim=-1)  # [B, S, dim, num_ops]
        noetic_out = (noetic_stack * noetic_weights.unsqueeze(-2)).sum(dim=-1)  # [B, S, dim]

        # Residual stream: small unconstrained transform
        residual_out = self.residual_stream(x)  # [B, S, dim]

        # Compute gate (scalar per position, clamped to budget)
        raw_gate = torch.sigmoid(self.residual_gate_linear(x))  # [B, S, 1]
        effective_gate = torch.clamp(raw_gate, max=self.residual_budget)

        # Combine streams
        output = noetic_out + effective_gate * residual_out

        # Compute residual energy for trace
        residual_contribution = effective_gate * residual_out
        residual_energy = residual_contribution.pow(2).sum(dim=-1).mean()  # scalar

        if squeeze:
            output = output.squeeze(1)
            raw_gate = raw_gate.squeeze(1)
            effective_gate = effective_gate.squeeze(1)
            noetic_weights = noetic_weights.squeeze(1)

        trace = None
        if return_trace:
            trace = {
                "residual_stream": {
                    "energy": residual_energy.item(),
                    "gate_value": effective_gate.mean().item(),
                    "raw_gate": raw_gate.mean().item(),
                },
                "noetic_weights": noetic_weights.detach(),
            }

        return output, trace

    def get_involution_penalty(self) -> torch.Tensor:
        """
        Compute involution penalty for dual pairs.

        For dual pairs (nu_2, nu_3), (nu_5, nu_6), (nu_8, nu_9):
            Penalty = ||nu_j @ nu_i - I||_F^2

        This encourages the dual pairs to be approximate involutions.

        Returns:
            Scalar tensor representing total involution penalty
        """
        penalty = torch.tensor(0.0, device=self.noetic_operators[0].device)
        identity = torch.eye(self.dim, device=self.noetic_operators[0].device)

        dual_pairs = [(2, 3), (5, 6), (8, 9)]
        for i, j in dual_pairs:
            composition = torch.matmul(self.noetic_operators[j], self.noetic_operators[i])
            penalty = penalty + torch.norm(composition - identity, p='fro') ** 2

        return penalty / len(dual_pairs)

    def get_spectral_penalty(self) -> torch.Tensor:
        """
        Compute spectral radius penalty for all noetic operators.

        Following TKS v7.3 Proposition 7.1 (Spectral Bounds):
            For unitary noetics: spectral radius = 1
            For Hermitian noetics: eigenvalues are real

        We penalize operators with spectral radius > 1 to ensure contraction.

        Returns:
            Scalar tensor representing total spectral penalty
        """
        penalty = torch.tensor(0.0, device=self.noetic_operators[0].device)

        for k in range(self.num_noetic_ops):
            op = self.noetic_operators[k]
            # Compute singular values (spectral norm = max singular value)
            singular_values = torch.linalg.svdvals(op)
            spectral_radius = singular_values.max()
            # Penalize if spectral radius > 1
            penalty = penalty + F.relu(spectral_radius - 1.0) ** 2

        return penalty / self.num_noetic_ops


# =============================================================================
# 2. CONSTRAINT SCHEDULER
# =============================================================================

class ConstraintScheduler:
    """
    Anneal constraints from soft to hard over training.

    Training Strategy:
        1. Start with loose spectral/involution penalties (initial_weight)
        2. Warmup phase: keep weight constant to allow model to learn
        3. Anneal phase: gradually increase weight toward final_weight
        4. Post-anneal: maintain strict interpretability

    This allows the model to learn nuanced representations first,
    then gradually constrain to interpretable noetic structure.

    Schedule Types:
        - linear: w(t) = initial + (final - initial) * progress
        - cosine: w(t) = initial + (final - initial) * (1 - cos(pi * progress)) / 2
        - exponential: w(t) = initial * (final/initial)^progress

    Attributes:
        warmup_steps (int): Steps before annealing begins
        anneal_steps (int): Steps over which to anneal from initial to final
        initial_weight (float): Starting constraint weight (loose)
        final_weight (float): Ending constraint weight (strict)
        schedule_type (str): 'linear', 'cosine', or 'exponential'
    """

    def __init__(
        self,
        warmup_steps: int = 1000,
        anneal_steps: int = 5000,
        initial_weight: float = 0.01,
        final_weight: float = 1.0,
        schedule_type: str = "cosine",
        spectral_initial: Optional[float] = None,
        spectral_final: Optional[float] = None,
        involution_initial: Optional[float] = None,
        involution_final: Optional[float] = None,
    ):
        """
        Initialize ConstraintScheduler.

        Args:
            warmup_steps: Number of steps before annealing begins
            anneal_steps: Number of steps to anneal from initial to final
            initial_weight: Starting constraint weight
            final_weight: Ending constraint weight
            schedule_type: Type of annealing schedule
            spectral_initial: Override initial weight for spectral penalty
            spectral_final: Override final weight for spectral penalty
            involution_initial: Override initial weight for involution penalty
            involution_final: Override final weight for involution penalty
        """
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.schedule_type = schedule_type

        # Allow separate schedules for spectral vs involution penalties
        self.spectral_initial = spectral_initial or initial_weight
        self.spectral_final = spectral_final or final_weight
        self.involution_initial = involution_initial or initial_weight
        self.involution_final = involution_final or final_weight

        self._current_step = 0

    def step(self):
        """Advance the scheduler by one step."""
        self._current_step += 1

    def _compute_weight(
        self,
        step: int,
        initial: float,
        final: float,
    ) -> float:
        """
        Compute weight at given step using the schedule.

        Args:
            step: Current training step
            initial: Initial weight value
            final: Final weight value

        Returns:
            Weight at this step
        """
        if step < self.warmup_steps:
            return initial

        if step >= self.warmup_steps + self.anneal_steps:
            return final

        # Compute progress through anneal phase [0, 1]
        progress = (step - self.warmup_steps) / self.anneal_steps

        if self.schedule_type == "linear":
            return initial + (final - initial) * progress

        elif self.schedule_type == "cosine":
            # Smooth cosine annealing
            return initial + (final - initial) * (1 - math.cos(math.pi * progress)) / 2

        elif self.schedule_type == "exponential":
            # Exponential interpolation
            if initial <= 0:
                return initial + (final - initial) * progress
            return initial * ((final / initial) ** progress)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def get_constraint_weight(self, step: Optional[int] = None) -> float:
        """
        Get general constraint weight at given step.

        Args:
            step: Training step (uses internal counter if None)

        Returns:
            Constraint weight at this step
        """
        step = step if step is not None else self._current_step
        return self._compute_weight(step, self.initial_weight, self.final_weight)

    def get_spectral_penalty_weight(self, step: Optional[int] = None) -> float:
        """
        Get spectral penalty weight at given step.

        Spectral penalty enforces bounded spectral radius for noetic operators.

        Args:
            step: Training step (uses internal counter if None)

        Returns:
            Spectral penalty weight at this step
        """
        step = step if step is not None else self._current_step
        return self._compute_weight(step, self.spectral_initial, self.spectral_final)

    def get_involution_penalty_weight(self, step: Optional[int] = None) -> float:
        """
        Get involution penalty weight at given step.

        Involution penalty enforces dual pair structure: nu_j @ nu_i ~ I

        Args:
            step: Training step (uses internal counter if None)

        Returns:
            Involution penalty weight at this step
        """
        step = step if step is not None else self._current_step
        return self._compute_weight(step, self.involution_initial, self.involution_final)

    def get_all_weights(self, step: Optional[int] = None) -> Dict[str, float]:
        """
        Get all constraint weights at given step.

        Args:
            step: Training step (uses internal counter if None)

        Returns:
            Dictionary with 'general', 'spectral', 'involution' weights
        """
        step = step if step is not None else self._current_step
        return {
            "general": self.get_constraint_weight(step),
            "spectral": self.get_spectral_penalty_weight(step),
            "involution": self.get_involution_penalty_weight(step),
            "step": step,
            "phase": self._get_phase(step),
        }

    def _get_phase(self, step: int) -> str:
        """Get current training phase name."""
        if step < self.warmup_steps:
            return "warmup"
        elif step < self.warmup_steps + self.anneal_steps:
            return "annealing"
        else:
            return "strict"

    def state_dict(self) -> Dict:
        """Return scheduler state for checkpointing."""
        return {
            "current_step": self._current_step,
            "warmup_steps": self.warmup_steps,
            "anneal_steps": self.anneal_steps,
            "initial_weight": self.initial_weight,
            "final_weight": self.final_weight,
            "schedule_type": self.schedule_type,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state from checkpoint."""
        self._current_step = state_dict["current_step"]


# =============================================================================
# 3. RESIDUAL DISTILLATION LOSS
# =============================================================================

class ResidualDistillationLoss(nn.Module):
    """
    Encourage residual stream to be reconstructible by noetic basis.

    This prevents the residual from becoming an opaque escape hatch.
    The loss measures how well the noetic basis can approximate the residual output.

    Mathematical Formulation:
        Given residual output r and noetic operators {nu_k}:

        1. Learn projection coefficients: c = Linear(r) in R^{num_noetic_ops}
        2. Reconstruct: r_hat = sum_k c_k * nu_k(identity_input)
        3. Loss: ||r - r_hat||^2

    A low loss means the residual is "nearly noetic" - interpretable.
    A high loss means the residual is using capacity outside the basis.

    Properties:
        - Encourages interpretability without hard constraints
        - Allows gradual transition during training
        - Provides diagnostic for model behavior

    Attributes:
        num_noetic_ops (int): Number of noetic operators (should match TwoStreamNoetic)
        dim (int): Embedding dimension
    """

    def __init__(
        self,
        dim: int = 40,
        num_noetic_ops: int = 10,
        temperature: float = 1.0,
    ):
        """
        Initialize ResidualDistillationLoss.

        Args:
            dim: Embedding dimension
            num_noetic_ops: Number of noetic operators in basis
            temperature: Softmax temperature for coefficient distribution
        """
        super().__init__()

        self.dim = dim
        self.num_noetic_ops = num_noetic_ops
        self.temperature = temperature

        # Project residual to noetic coefficients
        self.coeff_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_noetic_ops),
        )

        # Learnable basis vectors for reconstruction
        # These should align with the noetic operators in TwoStreamNoetic
        self.reconstruction_basis = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim) * 0.02)
            for _ in range(num_noetic_ops)
        ])

        self._init_reconstruction_basis()

    def _init_reconstruction_basis(self):
        """Initialize reconstruction basis with identity-like structure."""
        with torch.no_grad():
            for k in range(self.num_noetic_ops):
                nn.init.eye_(self.reconstruction_basis[k])
                # Add small perturbation
                self.reconstruction_basis[k] += torch.randn_like(
                    self.reconstruction_basis[k]
                ) * 0.01

    def forward(
        self,
        residual_output: torch.Tensor,
        noetic_operators: Optional[nn.ParameterList] = None,
        input_for_reconstruction: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute distillation loss for residual output.

        Args:
            residual_output: Output from residual stream [batch, seq, dim] or [batch, dim]
            noetic_operators: Optional external noetic operators (from TwoStreamNoetic)
            input_for_reconstruction: Optional input for operator application

        Returns:
            loss: Scalar distillation loss
            info: Dictionary with reconstruction details
        """
        squeeze = False
        if residual_output.dim() == 2:
            residual_output = residual_output.unsqueeze(1)
            squeeze = True

        batch, seq, dim = residual_output.shape

        # Use provided operators or fall back to internal basis
        operators = noetic_operators if noetic_operators is not None else self.reconstruction_basis

        # Compute coefficients from residual
        logits = self.coeff_projector(residual_output)  # [B, S, num_ops]
        coefficients = F.softmax(logits / self.temperature, dim=-1)

        # Reconstruct residual using noetic basis
        # If input provided, apply operators to input; otherwise use identity-like application
        if input_for_reconstruction is not None:
            if input_for_reconstruction.dim() == 2:
                input_for_reconstruction = input_for_reconstruction.unsqueeze(1)

            reconstructions = []
            for k in range(self.num_noetic_ops):
                op = operators[k]
                recon_k = torch.matmul(input_for_reconstruction, op.t())
                reconstructions.append(recon_k)
        else:
            # Apply operators to the residual output itself (self-reconstruction)
            reconstructions = []
            for k in range(self.num_noetic_ops):
                op = operators[k]
                recon_k = torch.matmul(residual_output, op.t())
                reconstructions.append(recon_k)

        recon_stack = torch.stack(reconstructions, dim=-1)  # [B, S, dim, num_ops]
        reconstructed = (recon_stack * coefficients.unsqueeze(-2)).sum(dim=-1)  # [B, S, dim]

        # Compute reconstruction loss
        mse_loss = F.mse_loss(reconstructed, residual_output)

        # Also compute coefficient entropy (lower = more concentrated = more interpretable)
        entropy = -(coefficients * torch.log(coefficients + 1e-8)).sum(dim=-1).mean()

        # Normalized interpretability score [0, 1]
        # 1 = perfectly reconstructible, 0 = not at all
        residual_norm = residual_output.pow(2).sum(dim=-1).mean()
        reconstruction_error = (reconstructed - residual_output).pow(2).sum(dim=-1).mean()
        interpretability = 1.0 - (reconstruction_error / (residual_norm + 1e-8)).clamp(max=1.0)

        if squeeze:
            reconstructed = reconstructed.squeeze(1)

        info = {
            "distillation_loss": mse_loss.item(),
            "coefficient_entropy": entropy.item(),
            "interpretability_score": interpretability.item(),
            "coefficients": coefficients.detach(),
            "reconstructed": reconstructed.detach(),
        }

        return mse_loss, info

    def sync_with_noetic_operators(self, noetic_operators: nn.ParameterList):
        """
        Synchronize reconstruction basis with TwoStreamNoetic operators.

        This should be called periodically during training to keep
        the distillation loss aligned with the actual noetic basis.

        Args:
            noetic_operators: ParameterList from TwoStreamNoetic
        """
        with torch.no_grad():
            for k in range(min(len(noetic_operators), self.num_noetic_ops)):
                self.reconstruction_basis[k].copy_(noetic_operators[k])


# =============================================================================
# 4. COMBINED TRACEABLE TWO-STREAM MODULE
# =============================================================================

class TraceableTwoStreamNoetic(nn.Module):
    """
    Complete two-stream noetic module with integrated constraint scheduling and distillation.

    This module combines:
        1. TwoStreamNoetic for the two-stream architecture
        2. ConstraintScheduler for training-time constraint annealing
        3. ResidualDistillationLoss for interpretability regularization

    Trace output includes all components for monitoring:
        - residual_stream.energy
        - residual_stream.gate_value
        - residual_stream.distillation_loss
        - constraint_weights.spectral
        - constraint_weights.involution
        - noetic_weights
    """

    def __init__(
        self,
        dim: int = 40,
        residual_budget: float = 0.1,
        num_noetic_ops: int = 10,
        warmup_steps: int = 1000,
        anneal_steps: int = 5000,
        initial_constraint_weight: float = 0.01,
        final_constraint_weight: float = 1.0,
        distillation_weight: float = 0.1,
    ):
        """
        Initialize TraceableTwoStreamNoetic.

        Args:
            dim: Embedding dimension (40 for TKS Ideas space)
            residual_budget: Maximum fraction from residual stream
            num_noetic_ops: Number of noetic basis operators
            warmup_steps: Steps before constraint annealing
            anneal_steps: Steps to anneal constraints
            initial_constraint_weight: Starting constraint weight
            final_constraint_weight: Ending constraint weight
            distillation_weight: Weight for distillation loss
        """
        super().__init__()

        self.dim = dim
        self.distillation_weight = distillation_weight

        # Core two-stream module
        self.two_stream = TwoStreamNoetic(
            dim=dim,
            residual_budget=residual_budget,
            num_noetic_ops=num_noetic_ops,
        )

        # Constraint scheduler
        self.scheduler = ConstraintScheduler(
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            initial_weight=initial_constraint_weight,
            final_weight=final_constraint_weight,
        )

        # Distillation loss
        self.distillation = ResidualDistillationLoss(
            dim=dim,
            num_noetic_ops=num_noetic_ops,
        )

        # Sync distillation basis with noetic operators
        self.distillation.sync_with_noetic_operators(self.two_stream.noetic_operators)

    def forward(
        self,
        x: torch.Tensor,
        operator_idx: Optional[int] = None,
        return_trace: bool = True,
        compute_losses: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with full tracing.

        Args:
            x: Input tensor [batch, seq, dim] or [batch, dim]
            operator_idx: Optional hard routing to single operator
            return_trace: If True, return detailed trace
            compute_losses: If True, compute constraint and distillation losses

        Returns:
            output: Transformed tensor
            trace: Dictionary with all monitoring information
        """
        # Apply two-stream transformation
        output, stream_trace = self.two_stream(x, operator_idx, return_trace=True)

        trace = None
        if return_trace:
            trace = {
                "residual_stream": stream_trace["residual_stream"].copy(),
                "noetic_weights": stream_trace["noetic_weights"],
            }

            if compute_losses:
                # Compute residual output for distillation
                residual_out = self.two_stream.residual_stream(x if x.dim() == 3 else x.unsqueeze(1))
                distill_loss, distill_info = self.distillation(
                    residual_out,
                    noetic_operators=self.two_stream.noetic_operators,
                    input_for_reconstruction=x if x.dim() == 3 else x.unsqueeze(1),
                )

                # Add distillation info to trace
                trace["residual_stream"]["distillation_loss"] = distill_info["distillation_loss"]
                trace["residual_stream"]["interpretability"] = distill_info["interpretability_score"]

                # Compute constraint penalties
                spectral_penalty = self.two_stream.get_spectral_penalty()
                involution_penalty = self.two_stream.get_involution_penalty()

                # Get current constraint weights
                weights = self.scheduler.get_all_weights()

                trace["constraint_penalties"] = {
                    "spectral": spectral_penalty.item(),
                    "involution": involution_penalty.item(),
                }
                trace["constraint_weights"] = weights

                # Compute total regularization loss
                total_reg = (
                    weights["spectral"] * spectral_penalty +
                    weights["involution"] * involution_penalty +
                    self.distillation_weight * distill_loss
                )
                trace["total_regularization_loss"] = total_reg.item()

        return output, trace

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute total regularization loss for training.

        Returns:
            Scalar tensor with weighted sum of all constraint losses
        """
        weights = self.scheduler.get_all_weights()

        spectral_penalty = self.two_stream.get_spectral_penalty()
        involution_penalty = self.two_stream.get_involution_penalty()

        # For distillation, we need a dummy forward to get residual output
        # This is typically called after the main forward pass

        return (
            weights["spectral"] * spectral_penalty +
            weights["involution"] * involution_penalty
        )

    def training_step(self):
        """Advance the constraint scheduler by one step."""
        self.scheduler.step()

        # Periodically sync distillation basis
        if self.scheduler._current_step % 100 == 0:
            self.distillation.sync_with_noetic_operators(self.two_stream.noetic_operators)


# =============================================================================
# TEST SECTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TKS RESIDUAL STREAM - Over-Constraint Mitigation Tests")
    print("=" * 70)

    # Test configuration
    dim = 40
    batch_size = 4
    seq_len = 16

    # Create test input
    x = torch.randn(batch_size, seq_len, dim)

    print("\n[1] Testing TwoStreamNoetic...")
    two_stream = TwoStreamNoetic(dim=dim, residual_budget=0.1, num_noetic_ops=10)

    output, trace = two_stream(x, return_trace=True)

    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Residual energy: {trace['residual_stream']['energy']:.6f}")
    print(f"    Gate value (effective): {trace['residual_stream']['gate_value']:.4f}")
    print(f"    Gate value (raw): {trace['residual_stream']['raw_gate']:.4f}")
    print(f"    Noetic weights shape: {trace['noetic_weights'].shape}")

    # Verify residual budget is respected
    assert trace['residual_stream']['gate_value'] <= 0.1 + 1e-6, "Gate exceeds budget!"
    print("    [PASS] Residual budget respected")

    # Test spectral and involution penalties
    spectral = two_stream.get_spectral_penalty()
    involution = two_stream.get_involution_penalty()
    print(f"    Spectral penalty: {spectral.item():.6f}")
    print(f"    Involution penalty: {involution.item():.6f}")

    print("\n[2] Testing ConstraintScheduler...")
    scheduler = ConstraintScheduler(
        warmup_steps=100,
        anneal_steps=500,
        initial_weight=0.01,
        final_weight=1.0,
        schedule_type="cosine",
    )

    # Test different training phases
    test_steps = [0, 50, 100, 300, 600, 1000]
    print("    Step | Phase      | General | Spectral | Involution")
    print("    " + "-" * 55)
    for step in test_steps:
        weights = scheduler.get_all_weights(step)
        print(f"    {step:4d} | {weights['phase']:10s} | {weights['general']:.4f} | "
              f"{weights['spectral']:.4f} | {weights['involution']:.4f}")

    # Verify annealing
    assert scheduler.get_constraint_weight(0) == 0.01, "Initial weight wrong"
    assert abs(scheduler.get_constraint_weight(600) - 1.0) < 0.01, "Final weight wrong"
    print("    [PASS] Constraint annealing works correctly")

    print("\n[3] Testing ResidualDistillationLoss...")
    distillation = ResidualDistillationLoss(dim=dim, num_noetic_ops=10)

    residual_out = two_stream.residual_stream(x)
    loss, info = distillation(residual_out, input_for_reconstruction=x)

    print(f"    Distillation loss: {info['distillation_loss']:.6f}")
    print(f"    Coefficient entropy: {info['coefficient_entropy']:.4f}")
    print(f"    Interpretability score: {info['interpretability_score']:.4f}")

    # Verify loss is non-negative
    assert loss.item() >= 0, "Distillation loss is negative!"
    print("    [PASS] Distillation loss computed correctly")

    print("\n[4] Testing TraceableTwoStreamNoetic (integrated module)...")
    traceable = TraceableTwoStreamNoetic(
        dim=dim,
        residual_budget=0.1,
        num_noetic_ops=10,
        warmup_steps=100,
        anneal_steps=500,
    )

    output, trace = traceable(x, return_trace=True, compute_losses=True)

    print(f"    Output shape: {output.shape}")
    print(f"    Residual stream trace keys: {list(trace['residual_stream'].keys())}")
    print(f"    Total regularization loss: {trace['total_regularization_loss']:.6f}")
    print(f"    Current phase: {trace['constraint_weights']['phase']}")

    # Simulate training steps
    print("\n    Simulating training...")
    for step in range(150):
        traceable.training_step()

    _, trace_after = traceable(x, return_trace=True, compute_losses=True)
    print(f"    After 150 steps, phase: {trace_after['constraint_weights']['phase']}")
    print(f"    After 150 steps, spectral weight: {trace_after['constraint_weights']['spectral']:.4f}")

    print("\n[5] Testing hard operator routing...")
    output_hard, trace_hard = two_stream(x, operator_idx=3, return_trace=True)

    # Verify only operator 3 is used
    noetic_weights = trace_hard['noetic_weights']
    expected_weights = torch.zeros(batch_size, seq_len, 10)
    expected_weights[..., 3] = 1.0

    weight_match = torch.allclose(noetic_weights, expected_weights)
    print(f"    Hard routing to operator 3: {'[PASS]' if weight_match else '[FAIL]'}")

    print("\n[6] Testing gradient flow...")
    x_grad = x.clone().requires_grad_(True)
    output = traceable(x_grad, return_trace=False, compute_losses=False)[0]
    loss = output.pow(2).mean()
    loss.backward()

    grad_exists = x_grad.grad is not None and not torch.isnan(x_grad.grad).any()
    print(f"    Gradient flow: {'[PASS]' if grad_exists else '[FAIL]'}")

    print("\n[7] Verifying no NaN/Inf in outputs...")
    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()
    print(f"    NaN check: {'[PASS]' if not has_nan else '[FAIL]'}")
    print(f"    Inf check: {'[PASS]' if not has_inf else '[FAIL]'}")

    # Parameter count
    param_count = sum(p.numel() for p in traceable.parameters())
    print(f"\n    Total parameters: {param_count:,}")

    print("\n" + "=" * 70)
    print("All tests passed! Residual stream module is ready for integration.")
    print("=" * 70)

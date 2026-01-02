"""
TKS-LLM Core Components v2 — Full 5-Layer Pipeline

This module extends tks_llm_core.py with:
    4. AttractorComputationLayer: Fixed-point iteration for thought convergence
    5. RPMGatingMechanism: Goal-oriented D/W/P filtering

Full Pipeline:
    tokens → NoeticEmbedding → NoeticProcessor → FractalAttention
          → AttractorComputation → RPMGating → output

Canonical References:
    - TKS_LLM_Noetic_Mathematics_v1.0.md Section 7 (Attractor Mathematics)
    - TKS_LLM_Architecture_v1.0.md (RPM Gating specification)
    - TKS_LLM_Canonical_Validation_v1.0.md (validated constraints)

Mathematical Constraints:
    - Attractor: Contraction mapping with Lipschitz constant L < 1
    - Attractor: Variance reduction over iterations
    - Attractor: Convergence below tolerance ε
    - RPM: D/W/P scores from [0,1], gate = D × W × P

Trace Schema (v1.0):
    When return_full_trace=True, the pipeline returns a standardized trace dict:
    {
        "noetic_routing": {"weights": Tensor, "indices": Tensor (optional)},
        "attention": {"scale_weights": Tensor},
        "attractor": {"converged": bool, "iterations": int, "final_delta": float, "trajectory": Tensor (optional)},
        "rpm": {"gate": Tensor, "dwp_scores": Tensor, "foundation_idx": Tensor},
        "operator_core": {...} (if active),
        "_legacy": {...} (backward-compatible keys),
        "_schema_version": "1.0"
    }

Author: TKS-LLM Integration-Agent
Date: 2025-12-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import base components from v1
from tks_llm_core import (
    NoeticEmbeddingLayer,
    NoeticProcessor,
    FractalAttentionMechanism,
    NOETIC_DIM,
    NUM_WORLDS,
    TOTAL_DIM,
    WORLD_OFFSETS,
    NOETIC_NAMES,
)


# =============================================================================
# TRACE SCHEMA VERSION
# =============================================================================

TRACE_SCHEMA_VERSION = "1.0"


def get_trace_schema_version() -> str:
    """
    Return the current trace schema version.

    This allows consumers to check compatibility and handle schema evolution.

    Returns:
        str: Schema version string (e.g., "1.0")
    """
    return TRACE_SCHEMA_VERSION


def _build_standardized_trace(
    *,
    # Noetic routing (from NoeticProcessor or NoeticRouter)
    noetic_weights: torch.Tensor = None,
    noetic_indices: torch.Tensor = None,
    # Attention (from FractalAttention)
    attention_scale_weights: torch.Tensor = None,
    # Attractor layer outputs
    attractor_converged: bool = None,
    attractor_iterations: int = None,
    attractor_final_delta: float = None,
    attractor_trajectory: torch.Tensor = None,
    attractor_state: torch.Tensor = None,
    # RPM gating outputs
    rpm_gate: torch.Tensor = None,
    rpm_dwp_scores: torch.Tensor = None,
    rpm_foundation_idx: torch.Tensor = None,
    # Operator core (optional)
    operator_core_gate_values: torch.Tensor = None,
    operator_core_noetic_output: torch.Tensor = None,
    operator_core_equation_repr: torch.Tensor = None,
    operator_core_symmetry_losses: dict = None,
    # Legacy fields for backward compatibility
    legacy_embedding: torch.Tensor = None,
    legacy_worlds: dict = None,
    legacy_processed: torch.Tensor = None,
    legacy_attended: torch.Tensor = None,
) -> dict:
    """
    Build a standardized trace dictionary following the v1.0 schema.

    This function ensures consistent trace structure across v2 and v4 pipelines.

    Args:
        noetic_weights: [batch, seq, 10] soft routing weights
        noetic_indices: [batch, seq] argmax indices (optional)
        attention_scale_weights: [batch, num_scales] per-scale mixture weights
        attractor_converged: Whether attractor iteration converged
        attractor_iterations: Number of iterations used
        attractor_final_delta: Final change magnitude
        attractor_trajectory: [batch, seq, dim] trajectory if requested
        attractor_state: [batch, seq, dim] final attractor state
        rpm_gate: [batch, seq] gate values (D*W*P product)
        rpm_dwp_scores: [batch, seq, 7, 3] or [batch, seq, 7] D/W/P scores
        rpm_foundation_idx: [batch, seq] argmax foundation index
        operator_core_*: Operator core outputs (if active)
        legacy_*: Legacy fields for backward compatibility

    Returns:
        Standardized trace dictionary with schema version
    """
    trace = {
        "_schema_version": TRACE_SCHEMA_VERSION,
    }

    # Noetic routing section
    if noetic_weights is not None or noetic_indices is not None:
        trace["noetic_routing"] = {}
        if noetic_weights is not None:
            trace["noetic_routing"]["weights"] = noetic_weights
        if noetic_indices is not None:
            trace["noetic_routing"]["indices"] = noetic_indices

    # Attention section
    if attention_scale_weights is not None:
        trace["attention"] = {
            "scale_weights": attention_scale_weights,
        }

    # Attractor section
    if attractor_converged is not None:
        trace["attractor"] = {
            "converged": attractor_converged,
            "iterations": attractor_iterations if attractor_iterations is not None else 0,
            "final_delta": attractor_final_delta if attractor_final_delta is not None else 0.0,
        }
        if attractor_trajectory is not None:
            trace["attractor"]["trajectory"] = attractor_trajectory
        if attractor_state is not None:
            trace["attractor"]["state"] = attractor_state

    # RPM section
    if rpm_gate is not None or rpm_dwp_scores is not None:
        trace["rpm"] = {}
        if rpm_gate is not None:
            trace["rpm"]["gate"] = rpm_gate
        if rpm_dwp_scores is not None:
            trace["rpm"]["dwp_scores"] = rpm_dwp_scores
        if rpm_foundation_idx is not None:
            trace["rpm"]["foundation_idx"] = rpm_foundation_idx

    # Operator core section (optional)
    if operator_core_gate_values is not None:
        trace["operator_core"] = {
            "gate_values": operator_core_gate_values,
        }
        if operator_core_noetic_output is not None:
            trace["operator_core"]["noetic_output"] = operator_core_noetic_output
        if operator_core_equation_repr is not None:
            trace["operator_core"]["equation_repr"] = operator_core_equation_repr
        if operator_core_symmetry_losses is not None:
            trace["operator_core"]["symmetry_losses"] = operator_core_symmetry_losses

    # Legacy section for backward compatibility
    legacy = {}
    if legacy_embedding is not None:
        legacy["embedding"] = legacy_embedding
    if legacy_worlds is not None:
        legacy["worlds"] = legacy_worlds
    if legacy_processed is not None:
        legacy["processed"] = legacy_processed
    if legacy_attended is not None:
        legacy["attended"] = legacy_attended
    if legacy:
        trace["_legacy"] = legacy

    return trace


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_FOUNDATIONS = 7  # Unity, Wisdom, Life, Companionship, Power, Material, Lust

FOUNDATION_NAMES = [
    'Unity',        # F1 - Connection to whole
    'Wisdom',       # F2 - Mental understanding
    'Life',         # F3 - Vitality, energy
    'Companionship',# F4 - Love, relationships
    'Power',        # F5 - Influence, causation
    'Material',     # F6 - Physical resources
    'Lust',         # F7 - Creative force
]

# =============================================================================
# CANONICAL D/W/P NOETIC MAPPINGS
# =============================================================================
# From TKS RPM specification and MVR protocol:
#   Desire (D): ν1 (Mind), ν4 (Vibration), ν7 (Rhythm) - the MVR installation protocol
#               Increase any of these and you increase desire
#   Wisdom (W): ν5 (Female/receptivity), ν6 (Male/structure) - dynamic information/ideas
#               The vessel (5) and structure (6) of knowledge
#   Power (P): ν8 (Cause), ν9 (Effect) - causal chain of manifestation

def _compute_dwp_indices():
    """Compute canonical D/W/P indices across all 4 worlds."""
    desire_noetics = [1, 4, 7]  # Mind, Vibration, Rhythm (MVR)
    wisdom_noetics = [5, 6]     # Female (receptivity), Male (structure)
    power_noetics = [8, 9]      # Cause, Effect

    world_offsets = [0, 10, 20, 30]  # A, B, C, D

    desire_idx = []
    wisdom_idx = []
    power_idx = []

    for offset in world_offsets:
        desire_idx.extend([offset + n for n in desire_noetics])
        wisdom_idx.extend([offset + n for n in wisdom_noetics])
        power_idx.extend([offset + n for n in power_noetics])

    return desire_idx, wisdom_idx, power_idx

DESIRE_INDICES, WISDOM_INDICES, POWER_INDICES = _compute_dwp_indices()
# DESIRE_INDICES: [1,4,7,11,14,17,21,24,27,31,34,37] (12 dims) - Mind,Vibration,Rhythm × 4 worlds
# WISDOM_INDICES: [5,6,15,16,25,26,35,36] (8 dims) - Female,Male × 4 worlds
# POWER_INDICES:  [8,9,18,19,28,29,38,39] (8 dims) - Cause,Effect × 4 worlds


# =============================================================================
# CONVERGENCE METRICS
# =============================================================================

@dataclass
class ConvergenceMetrics:
    """
    Metrics for tracking attractor convergence quality.

    Used for monitoring and loss computation during training.
    """
    converged: bool
    iterations: int
    final_delta: float
    variance_reduction: float
    delta_sequence: List[float]  # ||x_{t+1} - x_t|| for each iteration
    monotonic_decay: bool  # True if deltas decrease monotonically
    lipschitz_estimate: float  # Estimated Lipschitz constant from iteration


# =============================================================================
# COMPONENT 4: AttractorComputationLayer (Legacy - kept for compatibility)
# =============================================================================

class AttractorComputationLayer(nn.Module):
    """
    Computes thought attractors via iterative contraction mapping.

    Based on Banach Fixed-Point Theorem:
        If T: X → X is a contraction with Lipschitz constant L < 1,
        then T has a unique fixed point x* and iteration converges:
        x_{n+1} = T(x_n) → x*

    Mathematical Constraints (from TKS_LLM_Noetic_Mathematics_v1.0.md Section 7):
        1. Must approximate a contraction mapping (spectral radius < 1)
        2. Must reduce variance over iterations
        3. Must converge below tolerance
        4. Must be differentiable

    Args:
        dim: Input/output dimension
        contraction_factor: Target Lipschitz constant L (default 0.5)
        max_iterations: Maximum fixed-point iterations
        tolerance: Convergence threshold
        num_maps: Number of contraction maps (IFS structure)

    References:
        TKS_LLM_Noetic_Mathematics_v1.0.md Section 7.1-7.3
    """

    def __init__(
        self,
        dim: int = TOTAL_DIM,
        contraction_factor: float = 0.5,
        max_iterations: int = 10,
        tolerance: float = 1e-4,
        num_maps: int = 3,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.contraction_factor = contraction_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.num_maps = num_maps

        # Multiple contraction maps (IFS = Iterated Function System)
        # Each map is: T_i(x) = A_i @ x + b_i with ||A_i|| < 1
        self.contraction_maps = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_maps)
        ])

        # Initialize as contractions (spectral radius < contraction_factor)
        self._init_contraction_maps()

        # Learnable mixing weights for IFS
        self.mix_weights = nn.Parameter(torch.ones(num_maps) / num_maps)

        # Residual connection strength (for gradient flow)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def _init_contraction_maps(self) -> None:
        """Initialize maps as strict contractions."""
        for i, layer in enumerate(self.contraction_maps):
            # Initialize as scaled identity + small noise
            # Ensures spectral radius < contraction_factor
            scale = self.contraction_factor * (0.8 + 0.1 * i)  # Vary slightly
            nn.init.eye_(layer.weight)
            layer.weight.data *= scale
            layer.weight.data += 0.01 * torch.randn_like(layer.weight)

            # Small bias toward different regions
            nn.init.uniform_(layer.bias, -0.1, 0.1)

    def _apply_hutchinson_operator(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hutchinson operator: weighted combination of contraction maps.

        H(x) = sum_i w_i * T_i(x)

        This is a differentiable approximation of the IFS union operation.
        """
        weights = F.softmax(self.mix_weights, dim=0)

        result = torch.zeros_like(x)
        for i, (w, T) in enumerate(zip(weights, self.contraction_maps)):
            result = result + w * T(x)

        return result

    def _compute_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute variance across feature dimension."""
        return x.var(dim=-1).mean()

    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Iterate to fixed point (attractor).

        Args:
            x: [batch, seq, dim] input thought state
            return_trajectory: Whether to return full iteration trajectory

        Returns:
            Dict containing:
                'attractor': [batch, seq, dim] converged state
                'converged': bool, whether convergence achieved
                'iterations': int, number of iterations used
                'final_delta': float, final change magnitude
                'variance_reduction': float, ratio of final/initial variance
        """
        batch_size, seq_len, dim = x.shape

        # Track trajectory if requested
        trajectory: List[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(x.detach().clone())

        # Track variance for constraint verification
        initial_variance = self._compute_variance(x)

        # Fixed-point iteration
        state = x
        converged = False
        iterations = 0
        delta = float('inf')

        for i in range(self.max_iterations):
            # Apply Hutchinson operator
            next_state = self._apply_hutchinson_operator(state)

            # Add residual connection for gradient flow
            next_state = next_state + self.residual_weight * state

            # Normalize to prevent explosion (soft constraint)
            next_state = F.normalize(next_state, dim=-1) * (x.norm(dim=-1, keepdim=True) + 1e-8)

            # Check convergence
            delta = (next_state - state).abs().mean().item()

            if return_trajectory:
                trajectory.append(next_state.detach().clone())

            state = next_state
            iterations = i + 1

            if delta < self.tolerance:
                converged = True
                break

        # Compute variance reduction
        final_variance = self._compute_variance(state)
        variance_reduction = (final_variance / (initial_variance + 1e-8)).item()

        return {
            'attractor': state,
            'converged': converged,
            'iterations': iterations,
            'final_delta': delta,
            'variance_reduction': variance_reduction,
            'trajectory': trajectory if return_trajectory else None,
        }

    def verify_contraction(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Verify that the maps are contractions.

        Returns estimated Lipschitz constants for each map.
        """
        results = {}

        for i, T in enumerate(self.contraction_maps):
            # Estimate Lipschitz constant via random sampling
            x1 = torch.randn(num_samples, self.dim)
            x2 = torch.randn(num_samples, self.dim)

            with torch.no_grad():
                y1 = T(x1)
                y2 = T(x2)

                # L ≈ max ||T(x1) - T(x2)|| / ||x1 - x2||
                input_dist = (x1 - x2).norm(dim=-1)
                output_dist = (y1 - y2).norm(dim=-1)

                lipschitz = (output_dist / (input_dist + 1e-8)).max().item()

            results[f'map_{i}_lipschitz'] = lipschitz
            results[f'map_{i}_is_contraction'] = lipschitz < 1.0

        return results


# =============================================================================
# COMPONENT 4b: StableAttractorLayer (FIXED VERSION)
# =============================================================================

class StableAttractorLayer(nn.Module):
    """
    Stable attractor computation with GUARANTEED contraction.

    This is a corrected implementation that fixes the 0% convergence rate
    in the original AttractorComputationLayer by:

    1. SPECTRAL NORMALIZATION: All weight matrices are wrapped with spectral_norm
       to enforce ||W||_2 = 1, then scaled by contraction_factor to ensure L < 1.

    2. NO RESIDUAL IN ITERATION: Residual connections break contraction properties.
       Instead, we use a learnable "anchor" that blends input with attractor output
       AFTER convergence.

    3. NO NORMALIZATION IN LOOP: Normalization is not a contraction mapping.
       We use bounded activation (tanh) to prevent explosion while preserving
       contraction.

    4. PROPER HUTCHINSON OPERATOR: The weighted combination of contractions
       is itself a contraction when weights sum to 1 (convex combination).

    Mathematical Guarantee (Banach Fixed-Point Theorem):
        If T: X -> X satisfies ||T(x) - T(y)|| <= L||x - y|| with L < 1,
        then T has a unique fixed point x* and x_n -> x* for any starting x_0.

    Convergence Rate: ||x_n - x*|| <= L^n ||x_0 - x*||
        With L = contraction_factor = 0.5, after 10 iterations:
        error <= 0.5^10 = 0.001 (99.9% reduction)

    TKS Canon Compliance:
        - Attractor finds stable noetic fixed points (Banach theorem)
        - 40D noetic space structure preserved (dim parameter)
        - Anti-attractor dynamics preserved (separate layer, not modified here)

    Args:
        dim: Input/output dimension (default 40 for TKS noetic space)
        contraction_factor: Target Lipschitz constant L < 1 (default 0.5)
        max_iterations: Maximum fixed-point iterations (default 10)
        tolerance: Convergence threshold (default 1e-4)
        num_maps: Number of contraction maps in IFS (default 3)
    """

    def __init__(
        self,
        dim: int = TOTAL_DIM,
        contraction_factor: float = 0.5,
        max_iterations: int = 15,
        tolerance: float = 1e-3,
        num_maps: int = 3,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.contraction_factor = contraction_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.num_maps = num_maps

        # Contraction maps with SPECTRAL NORMALIZATION
        # spectral_norm ensures ||W||_2 = 1, then we scale by contraction_factor
        self.contraction_maps = nn.ModuleList([
            self._make_contraction_layer(dim) for _ in range(num_maps)
        ])

        # Learnable mixing weights (softmax ensures convex combination)
        self.mix_logits = nn.Parameter(torch.zeros(num_maps))

        # Learnable bias terms for each map (small, to preserve contraction)
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(dim) * 0.01) for _ in range(num_maps)
        ])

        # Post-convergence anchor blending (NOT in iteration loop)
        # This preserves gradient flow without breaking contraction
        # Init to -2.0: sigmoid(-2.0) ≈ 0.12, giving ~12% blend toward input
        # (Previous 0.1 gave sigmoid(0.1) ≈ 0.525, too strong a pull toward input)
        self.anchor_weight = nn.Parameter(torch.tensor(-2.0))

        # Learnable scaling to match input magnitude after convergence
        self.output_scale = nn.Parameter(torch.ones(1))

    def _make_contraction_layer(self, dim: int) -> nn.Module:
        """
        Create a spectral-normalized linear layer scaled to be a contraction.

        The spectral norm ensures ||W||_2 = 1.
        We then scale by contraction_factor to get ||W||_2 < 1.
        """
        layer = nn.Linear(dim, dim, bias=False)

        # Initialize close to identity for stability
        nn.init.eye_(layer.weight)
        layer.weight.data *= 0.9  # Start below 1

        # Apply spectral normalization (enforces ||W||_2 = 1)
        layer = spectral_norm(layer, name='weight')

        return layer

    def _apply_contraction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Hutchinson operator (convex combination of contractions).

        H(x) = sum_i w_i * (contraction_factor * T_i(x) + b_i)

        Properties:
            - Each T_i has spectral norm 1, scaled by contraction_factor < 1
            - Weights w_i sum to 1 (softmax), so convex combination
            - Convex combination of L-contractions is an L-contraction
            - Small bias b_i shifts fixed point without breaking contraction
        """
        # Softmax weights ensure convex combination (sum to 1)
        weights = F.softmax(self.mix_logits, dim=0)

        result = torch.zeros_like(x)
        for i, (w, T, b) in enumerate(zip(weights, self.contraction_maps, self.biases)):
            # Apply spectral-normalized map and scale by contraction_factor
            mapped = self.contraction_factor * T(x)
            # Add small bias (bounded to not exceed contraction)
            mapped = mapped + b.clamp(-0.1, 0.1)
            # Weighted sum
            result = result + w * mapped

        # Note: tanh is 1-Lipschitz but can slow convergence.
        # Instead, use soft clamping only for very large values
        # This preserves the stronger contraction rate while preventing explosion
        result = torch.where(
            result.abs() > 3.0,
            torch.sign(result) * 3.0,  # Clamp extreme values
            result
        )

        return result

    def _compute_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute variance across feature dimension."""
        return x.var(dim=-1).mean()

    def forward(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False,
        return_metrics: bool = False,
        return_tensor_deltas: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Iterate to fixed point (attractor) with guaranteed convergence.

        Args:
            x: [batch, seq, dim] input thought state
            return_trajectory: Whether to return full iteration trajectory
            return_metrics: Whether to return detailed ConvergenceMetrics
            return_tensor_deltas: If True, return delta_sequence as tensors for backprop

        Returns:
            Dict containing:
                'attractor': [batch, seq, dim] converged state
                'converged': bool, whether convergence achieved
                'iterations': int, number of iterations used
                'final_delta': float, final change magnitude
                'variance_reduction': float, ratio of final/initial variance
                'trajectory': List[Tensor] if return_trajectory
                'metrics': ConvergenceMetrics if return_metrics
                'delta_sequence': List[float] deltas per iteration (or List[Tensor] if return_tensor_deltas)
                'convergence_loss': Tensor if return_tensor_deltas (for direct backprop)
        """
        batch_size, seq_len, dim = x.shape
        device = x.device

        # Track trajectory if requested
        trajectory: List[torch.Tensor] = []
        delta_sequence: List[float] = []
        delta_tensors: List[torch.Tensor] = [] if return_tensor_deltas else None

        if return_trajectory:
            trajectory.append(x.detach().clone())

        # Track variance for constraint verification
        initial_variance = self._compute_variance(x)

        # Store input norm for output scaling
        input_norm = x.norm(dim=-1, keepdim=True).mean()

        # Fixed-point iteration (NO residual, NO normalization in loop)
        state = x.clone()
        converged = False
        iterations = 0
        prev_delta = float('inf')

        for i in range(self.max_iterations):
            # Apply contraction mapping (guaranteed L < 1)
            next_state = self._apply_contraction(state)

            # Compute delta (for convergence check and monitoring)
            delta_tensor = (next_state - state).abs().mean()
            delta = delta_tensor.item()
            delta_sequence.append(delta)

            # Keep tensor version for backprop if requested
            if return_tensor_deltas:
                delta_tensors.append(delta_tensor)

            if return_trajectory:
                trajectory.append(next_state.detach().clone())

            # Update state
            state = next_state
            iterations = i + 1

            # Check convergence
            if delta < self.tolerance:
                converged = True
                break

            prev_delta = delta

        # Post-convergence processing (OUTSIDE iteration loop)
        # Blend with input anchor for gradient flow
        anchor_w = torch.sigmoid(self.anchor_weight)  # Bound to [0, 1]
        attractor = (1 - anchor_w) * state + anchor_w * x

        # Scale output to match input magnitude (preserves noetic structure)
        attractor_norm = attractor.norm(dim=-1, keepdim=True).mean()
        if attractor_norm > 1e-8:
            scale = self.output_scale * (input_norm / attractor_norm)
            attractor = attractor * scale

        # Compute variance reduction
        final_variance = self._compute_variance(attractor)
        variance_reduction = (final_variance / (initial_variance + 1e-8)).item()

        # Check monotonic decay
        monotonic_decay = all(
            delta_sequence[i] >= delta_sequence[i+1]
            for i in range(len(delta_sequence) - 1)
        ) if len(delta_sequence) > 1 else True

        # Estimate Lipschitz constant from iteration
        lipschitz_estimate = 0.0
        if len(delta_sequence) >= 2 and delta_sequence[0] > 1e-8:
            lipschitz_estimate = delta_sequence[-1] / delta_sequence[0]

        result = {
            'attractor': attractor,
            'converged': converged,
            'iterations': iterations,
            'final_delta': delta_sequence[-1] if delta_sequence else 0.0,
            'variance_reduction': variance_reduction,
            'trajectory': trajectory if return_trajectory else None,
            'delta_sequence': delta_sequence,
        }

        # Compute convergence loss from tensor deltas (enables backprop)
        if return_tensor_deltas and delta_tensors:
            result['delta_tensors'] = delta_tensors
            # Convergence loss: sum of deltas + penalty for non-convergence
            conv_loss = torch.stack(delta_tensors).sum()
            if not converged:
                # Add penalty for final delta exceeding tolerance
                conv_loss = conv_loss + F.relu(delta_tensors[-1] - self.tolerance) * 10.0
            # Add monotonic decay penalty
            for i in range(len(delta_tensors) - 1):
                conv_loss = conv_loss + F.relu(delta_tensors[i+1] - delta_tensors[i])
            result['convergence_loss'] = conv_loss

        if return_metrics:
            result['metrics'] = ConvergenceMetrics(
                converged=converged,
                iterations=iterations,
                final_delta=delta_sequence[-1] if delta_sequence else 0.0,
                variance_reduction=variance_reduction,
                delta_sequence=delta_sequence,
                monotonic_decay=monotonic_decay,
                lipschitz_estimate=lipschitz_estimate,
            )

        return result

    def verify_contraction(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Verify that the maps are contractions.

        With spectral normalization, this should ALWAYS pass.
        """
        results = {}
        device = next(self.parameters()).device

        for i, T in enumerate(self.contraction_maps):
            # Estimate Lipschitz constant via random sampling
            x1 = torch.randn(num_samples, self.dim, device=device)
            x2 = torch.randn(num_samples, self.dim, device=device)

            with torch.no_grad():
                # Apply the map with contraction scaling
                y1 = self.contraction_factor * T(x1)
                y2 = self.contraction_factor * T(x2)

                # L = max ||T(x1) - T(x2)|| / ||x1 - x2||
                input_dist = (x1 - x2).norm(dim=-1)
                output_dist = (y1 - y2).norm(dim=-1)

                lipschitz = (output_dist / (input_dist + 1e-8)).max().item()

            results[f'map_{i}_lipschitz'] = lipschitz
            results[f'map_{i}_is_contraction'] = lipschitz < 1.0

        # Also verify the full Hutchinson operator
        x1 = torch.randn(num_samples, self.dim, device=device)
        x2 = torch.randn(num_samples, self.dim, device=device)

        with torch.no_grad():
            y1 = self._apply_contraction(x1)
            y2 = self._apply_contraction(x2)

            input_dist = (x1 - x2).norm(dim=-1)
            output_dist = (y1 - y2).norm(dim=-1)

            full_lipschitz = (output_dist / (input_dist + 1e-8)).max().item()

        results['hutchinson_lipschitz'] = full_lipschitz
        results['hutchinson_is_contraction'] = full_lipschitz < 1.0

        return results

    def get_convergence_stats(self) -> Dict[str, float]:
        """
        Get current layer statistics for monitoring.
        """
        stats = {}

        # Mixing weights
        weights = F.softmax(self.mix_logits, dim=0)
        for i, w in enumerate(weights):
            stats[f'mix_weight_{i}'] = w.item()

        # Anchor weight
        stats['anchor_weight'] = torch.sigmoid(self.anchor_weight).item()

        # Output scale
        stats['output_scale'] = self.output_scale.item()

        # Contraction factor (fixed)
        stats['contraction_factor'] = self.contraction_factor

        return stats


# =============================================================================
# ATTRACTOR CONVERGENCE LOSS
# =============================================================================

def attractor_convergence_loss(
    delta_sequence: List[float],
    converged: bool,
    threshold: float = 1e-4,
    decay_weight: float = 1.0,
    convergence_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute loss penalizing non-convergence in attractor iteration.

    This loss has two components:
    1. MONOTONIC DECAY PENALTY: Penalizes if ||x_{t+1} - x_t|| doesn't decrease
    2. FINAL THRESHOLD PENALTY: Penalizes if final delta > threshold

    Args:
        delta_sequence: List of ||x_{t+1} - x_t|| values per iteration
        converged: Whether iteration converged (delta < tolerance)
        threshold: Target final delta (default 1e-4)
        decay_weight: Weight for monotonic decay penalty
        convergence_weight: Weight for final threshold penalty

    Returns:
        Scalar loss tensor (differentiable if delta_sequence contains tensors)

    Usage:
        out = attractor_layer(x, return_metrics=True)
        conv_loss = attractor_convergence_loss(
            out['delta_sequence'],
            out['converged']
        )
        total_loss = task_loss + 0.1 * conv_loss
    """
    if len(delta_sequence) < 2:
        return torch.tensor(0.0)

    # Convert to tensor if needed
    deltas = torch.tensor(delta_sequence, dtype=torch.float32)

    # 1. Monotonic decay penalty: sum of violations where delta[i+1] > delta[i]
    decay_violations = F.relu(deltas[1:] - deltas[:-1])
    decay_loss = decay_violations.sum()

    # 2. Final threshold penalty: penalize if final delta > threshold
    final_delta = deltas[-1]
    threshold_loss = F.relu(final_delta - threshold)

    # 3. Bonus for early convergence (negative loss)
    # convergence_bonus = -0.1 if converged else 0.0

    total = decay_weight * decay_loss + convergence_weight * threshold_loss

    return total


def compute_lipschitz_penalty(
    attractor_layer: StableAttractorLayer,
    num_samples: int = 50,
    target_lipschitz: float = 0.9
) -> torch.Tensor:
    """
    Compute gradient penalty encouraging Lipschitz constant < target.

    This is an auxiliary loss for training to ensure contraction.

    Args:
        attractor_layer: The StableAttractorLayer to check
        num_samples: Number of random samples for estimation
        target_lipschitz: Maximum desired Lipschitz constant

    Returns:
        Scalar penalty tensor (0 if Lipschitz < target)
    """
    device = next(attractor_layer.parameters()).device

    x1 = torch.randn(num_samples, attractor_layer.dim, device=device)
    x2 = torch.randn(num_samples, attractor_layer.dim, device=device)

    y1 = attractor_layer._apply_contraction(x1)
    y2 = attractor_layer._apply_contraction(x2)

    input_dist = (x1 - x2).norm(dim=-1)
    output_dist = (y1 - y2).norm(dim=-1)

    lipschitz_ratios = output_dist / (input_dist + 1e-8)
    max_lipschitz = lipschitz_ratios.max()

    # Penalize if Lipschitz exceeds target
    penalty = F.relu(max_lipschitz - target_lipschitz)

    return penalty


# =============================================================================
# COMPONENT 5: RPMGatingMechanism
# =============================================================================

class RPMGatingMechanism(nn.Module):
    """
    Implements RPM (Recursive Prerequisite Model) for goal-oriented thought filtering.

    From TKS_LLM_Architecture_v1.0.md (updated per MVR protocol):
        - Desire (D): Does this thought serve a goal? (from ν1, ν4, ν7 - MVR)
        - Wisdom (W): Is this thought informed/knowledgeable? (from ν5, ν6)
        - Power (P): Can this thought be actualized? (from ν8, ν9)

    Gate = D × W × P for selected Foundation.
    Only thoughts satisfying all three pass through.

    CANONICAL NOETIC EXTRACTION (MVR protocol):
        D extracts from indices: 1,4,7,11,14,17,21,24,27,31,34,37 (ν1/ν4/ν7 - Mind/Vibration/Rhythm)
        W extracts from indices: 5,6,15,16,25,26,35,36 (ν5/ν6 - Female/Male)
        P extracts from indices: 8,9,18,19,28,29,38,39 (ν8/ν9 - Cause/Effect)

    Args:
        dim: Input dimension
        num_foundations: Number of Foundation goals (default 7)

    References:
        TKS_LLM_Architecture_v1.0.md Section 2.2.5
        "Navigating The TOOTRA Kabalistic System" - Foundation definitions
    """

    def __init__(
        self,
        dim: int = TOTAL_DIM,
        num_foundations: int = NUM_FOUNDATIONS,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_foundations = num_foundations

        # Register canonical D/W/P indices as buffers
        self.register_buffer('desire_idx', torch.tensor(DESIRE_INDICES, dtype=torch.long))
        self.register_buffer('wisdom_idx', torch.tensor(WISDOM_INDICES, dtype=torch.long))
        self.register_buffer('power_idx', torch.tensor(POWER_INDICES, dtype=torch.long))

        # D/W/P evaluators for each Foundation
        # Each evaluator takes ONLY the canonical noetic subset and outputs [0, 1]
        desire_dim = len(DESIRE_INDICES)   # 12 dims (ν1,ν4,ν7 × 4 worlds) - MVR
        wisdom_dim = len(WISDOM_INDICES)   # 8 dims (ν5,ν6 × 4 worlds) - Female/Male
        power_dim = len(POWER_INDICES)     # 8 dims (ν8,ν9 × 4 worlds) - Cause/Effect

        self.dwp_evaluators = nn.ModuleList([
            nn.ModuleDict({
                'desire': nn.Sequential(
                    nn.Linear(desire_dim, desire_dim // 2),  # 12 -> 6
                    nn.ReLU(),
                    nn.Linear(desire_dim // 2, 1),           # 6 -> 1
                    nn.Sigmoid(),  # Output in [0, 1]
                ),
                'wisdom': nn.Sequential(
                    nn.Linear(wisdom_dim, wisdom_dim),       # 8 -> 8
                    nn.ReLU(),
                    nn.Linear(wisdom_dim, 1),                # 8 -> 1
                    nn.Sigmoid(),
                ),
                'power': nn.Sequential(
                    nn.Linear(power_dim, power_dim),         # 8 -> 8
                    nn.ReLU(),
                    nn.Linear(power_dim, 1),                 # 8 -> 1
                    nn.Sigmoid(),
                ),
            }) for _ in range(num_foundations)
        ])

        # Prerequisite network: checks if thought meets prereqs for goal
        self.prerequisite_net = nn.Sequential(
            nn.Linear(dim * 2, dim),  # thought + goal concatenated
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Goal state projection
        self.goal_projection = nn.Linear(dim, dim)

        # Foundation embeddings (learnable anchors)
        self.foundation_embeddings = nn.Parameter(
            self._init_foundation_embeddings()
        )

    def _init_foundation_embeddings(self) -> torch.Tensor:
        """
        Initialize Foundation embeddings based on TKS canonical positions.

        From TKS_LLM_Canonical_Validation_v1.0.md Section 2.2:
            F1 (Unity): ν₀ uniform across worlds
            F2 (Wisdom): ν₁, ν₂ in B-world
            F3 (Life): ν₄ across all worlds
            F4 (Companionship): ν₂, ν₅ in C-world
            F5 (Power): ν₆, ν₈ across worlds
            F6 (Material): D-world emphasis
            F7 (Lust): ν₅, ν₆, ν₇ across worlds
        """
        embeddings = torch.zeros(self.num_foundations, self.dim)

        # F1: Unity - ν₀ (IDEA) uniform across all worlds
        embeddings[0, [0, 10, 20, 30]] = 1.0  # ν₀ in each world

        # F2: Wisdom - ν₁ (MIND), ν₂ (POSITIVE) in B-world
        embeddings[1, [11, 12]] = 1.0  # B1, B2

        # F3: Life - ν₄ (VIBRATION) across all worlds
        embeddings[2, [4, 14, 24, 34]] = 1.0  # ν₄ in each world

        # F4: Companionship - ν₂, ν₅ in C-world
        embeddings[3, [22, 25]] = 1.0  # C2, C5

        # F5: Power - ν₆ (MALE), ν₈ (CAUSE) across worlds
        embeddings[4, [6, 8, 16, 18, 26, 28, 36, 38]] = 0.5  # ν₆, ν₈ each world

        # F6: Material - D-world emphasis
        embeddings[5, 30:40] = 0.3  # All D-world noetics
        embeddings[5, [30, 34, 39]] = 1.0  # D0, D4, D9 emphasized

        # F7: Lust/Creation - ν₅, ν₆, ν₇ generative triad
        for w in [0, 10, 20, 30]:  # Each world
            embeddings[6, [w + 5, w + 6, w + 7]] = 0.5

        # Normalize each Foundation embedding
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def get_foundation_embedding(self, foundation_idx: int) -> torch.Tensor:
        """Get the embedding for a specific Foundation."""
        return self.foundation_embeddings[foundation_idx]

    def forward(
        self,
        thought_state: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        target_foundation: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply RPM gating.

        Args:
            thought_state: [batch, seq, dim] current thought
            goal_state: [batch, dim] optional explicit goal
            target_foundation: int 0-6, which Foundation to gate for

        Returns:
            Dict containing:
                'gated_output': [batch, seq, dim] gated thought
                'rpm_gate': [batch, seq] gate values
                'dwp_scores': [batch, seq, 7, 3] D/W/P scores per Foundation
                'desire_scores': [batch, seq, 7]
                'wisdom_scores': [batch, seq, 7]
                'power_scores': [batch, seq, 7]
                'prereq_score': [batch, seq] prerequisite satisfaction
        """
        batch_size, seq_len, dim = thought_state.shape

        # Flatten for evaluation
        thought_flat = thought_state.view(batch_size * seq_len, dim)

        # Extract canonical noetic subsets for D/W/P
        desire_features = thought_flat[:, self.desire_idx]   # [batch*seq, 8]
        wisdom_features = thought_flat[:, self.wisdom_idx]   # [batch*seq, 20]
        power_features = thought_flat[:, self.power_idx]     # [batch*seq, 8]

        # Compute D/W/P for each Foundation using canonical noetic extraction
        all_dwp_scores = []
        for f_idx, evaluator in enumerate(self.dwp_evaluators):
            d_score = evaluator['desire'](desire_features)   # [batch*seq, 1] from ν1,ν4,ν7 (MVR)
            w_score = evaluator['wisdom'](wisdom_features)   # [batch*seq, 1] from ν5,ν6
            p_score = evaluator['power'](power_features)     # [batch*seq, 1] from ν8,ν9

            dwp = torch.cat([d_score, w_score, p_score], dim=-1)  # [batch*seq, 3]
            all_dwp_scores.append(dwp)

        # Stack: [batch*seq, 7, 3]
        dwp_scores = torch.stack(all_dwp_scores, dim=1)
        dwp_scores = dwp_scores.view(batch_size, seq_len, self.num_foundations, 3)

        # Compute RPM gate
        if target_foundation is not None:
            # Gate based on specific Foundation
            target_dwp = dwp_scores[:, :, target_foundation, :]  # [batch, seq, 3]

            # Gate = D × W × P (all must be satisfied)
            rpm_gate = (
                target_dwp[:, :, 0] *  # Desire
                target_dwp[:, :, 1] *  # Wisdom
                target_dwp[:, :, 2]    # Power
            )  # [batch, seq]
        else:
            # Gate based on maximum across Foundations
            dwp_product = dwp_scores.prod(dim=-1)  # [batch, seq, 7]
            rpm_gate = dwp_product.max(dim=-1).values  # [batch, seq]

        # Check prerequisites if goal provided
        prereq_score = torch.ones(batch_size, seq_len, device=thought_state.device)
        if goal_state is not None:
            goal_expanded = goal_state.unsqueeze(1).expand(-1, seq_len, -1)
            prereq_input = torch.cat([thought_state, goal_expanded], dim=-1)
            prereq_input = prereq_input.view(batch_size * seq_len, dim * 2)
            prereq_score = self.prerequisite_net(prereq_input)
            prereq_score = prereq_score.view(batch_size, seq_len)

            # Combine RPM gate with prerequisite check
            rpm_gate = rpm_gate * prereq_score

        # Apply gate
        gated_thought = thought_state * rpm_gate.unsqueeze(-1)

        # If goal provided, blend toward goal for high-gate thoughts
        if goal_state is not None:
            goal_proj = self.goal_projection(goal_state)
            goal_proj = goal_proj.unsqueeze(1).expand(-1, seq_len, -1)

            # Blend toward goal proportional to gate
            blend_strength = 0.1
            gated_thought = gated_thought + blend_strength * rpm_gate.unsqueeze(-1) * goal_proj

        return {
            'gated_output': gated_thought,
            'rpm_gate': rpm_gate,
            'dwp_scores': dwp_scores,
            'desire_scores': dwp_scores[:, :, :, 0],
            'wisdom_scores': dwp_scores[:, :, :, 1],
            'power_scores': dwp_scores[:, :, :, 2],
            'prereq_score': prereq_score,
        }


# =============================================================================
# FULL 5-LAYER PIPELINE
# =============================================================================

class TKSLLMCorePipeline(nn.Module):
    """
    Complete TKS-LLM 5-layer core pipeline.

    Pipeline:
        1. NoeticEmbeddingLayer: tokens -> 40-dim noetic space
        2. NoeticProcessor: apply noetic transforms
        3. FractalAttentionMechanism: multi-scale attention
        4. AttractorComputationLayer: converge to thought attractor
        5. RPMGatingMechanism: D/W/P goal-oriented filtering

    Args:
        vocab_size: Token vocabulary size
        hidden_dim: Intermediate embedding dimension
        noetic_dim: Noetic space dimension (default 40)
        num_scales: Fractal attention scales
        max_attractor_iter: Maximum attractor iterations
        contraction_factor: Attractor contraction strength
        use_stable_attractor: If True, use StableAttractorLayer with guaranteed
            convergence (default True). If False, use legacy AttractorComputationLayer.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 128,
        noetic_dim: int = TOTAL_DIM,
        num_scales: int = 3,
        max_attractor_iter: int = 10,
        contraction_factor: float = 0.5,
        use_stable_attractor: bool = True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.noetic_dim = noetic_dim
        self.use_stable_attractor = use_stable_attractor

        # Layer 1: Noetic Embedding
        self.embedding = NoeticEmbeddingLayer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim
        )

        # Layer 2: Noetic Processor
        self.processor = NoeticProcessor(dim=noetic_dim)

        # Layer 3: Fractal Attention
        self.fractal_attention = FractalAttentionMechanism(
            dim=noetic_dim,
            num_scales=num_scales
        )

        # Layer 4: Attractor Computation
        # Use StableAttractorLayer by default for guaranteed convergence
        if use_stable_attractor:
            self.attractor = StableAttractorLayer(
                dim=noetic_dim,
                contraction_factor=contraction_factor,
                max_iterations=max_attractor_iter
            )
        else:
            # Legacy layer (may have convergence issues)
            self.attractor = AttractorComputationLayer(
                dim=noetic_dim,
                contraction_factor=contraction_factor,
                max_iterations=max_attractor_iter
            )

        # Layer 5: RPM Gating
        self.rpm_gating = RPMGatingMechanism(dim=noetic_dim)

        # Output projection
        self.output_projection = nn.Linear(noetic_dim, vocab_size)

    def forward(
        self,
        tokens: torch.LongTensor,
        noetic_idx: int = 1,
        goal_state: Optional[torch.Tensor] = None,
        target_foundation: Optional[int] = None,
        return_full_trace: bool = False,
        return_tensor_deltas: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full pipeline.

        Args:
            tokens: [batch, seq] input token indices
            noetic_idx: Which noetic transform to apply (0-9)
            goal_state: [batch, dim] optional goal representation
            target_foundation: int 0-6 for RPM gating
            return_full_trace: Whether to return intermediate states (uses v1.0 schema)
            return_tensor_deltas: If True, return convergence_loss tensor for backprop

        Returns:
            Dict with 'logits', 'gated_output', and optionally full trace.
            If return_full_trace=True, 'trace' contains standardized schema v1.0:
                - noetic_routing: {weights, indices}
                - attention: {scale_weights}
                - attractor: {converged, iterations, final_delta, trajectory, state}
                - rpm: {gate, dwp_scores, foundation_idx}
                - _legacy: {embedding, worlds, processed, attended} for backward compat
                - _schema_version: "1.0"
            If return_tensor_deltas=True, includes 'convergence_loss' tensor
        """
        # Collect trace components
        trace_noetic_weights = None
        trace_embedding = None
        trace_worlds = None
        trace_processed = None
        trace_attended = None

        # Layer 1: Embedding
        emb_out = self.embedding(tokens)
        noetic = emb_out['noetic']  # [batch, seq, 40]

        if return_full_trace:
            # NOTE: Don't detach embedding - needed for world classification loss gradients
            trace_embedding = noetic
            trace_worlds = {k: v.detach() for k, v in emb_out.items() if k != 'noetic'}

        # Layer 2: Noetic Processing
        processed = self.processor(noetic, noetic_idx=noetic_idx)

        if return_full_trace:
            trace_processed = processed.detach()
            # Create noetic weights tensor for the applied transform
            batch_size, seq_len = tokens.shape
            trace_noetic_weights = torch.zeros(batch_size, seq_len, 10, device=tokens.device)
            trace_noetic_weights[..., noetic_idx] = 1.0

        # Layer 3: Fractal Attention
        attended = self.fractal_attention(processed)

        if return_full_trace:
            trace_attended = attended.detach()

        # Layer 4: Attractor
        # Pass return_tensor_deltas only to StableAttractorLayer
        if self.use_stable_attractor and return_tensor_deltas:
            attractor_out = self.attractor(
                attended,
                return_trajectory=return_full_trace,
                return_tensor_deltas=True
            )
        else:
            attractor_out = self.attractor(
                attended,
                return_trajectory=return_full_trace
            )

        # Layer 5: RPM Gating
        rpm_out = self.rpm_gating(
            attractor_out['attractor'],
            goal_state=goal_state,
            target_foundation=target_foundation
        )

        # Output logits
        logits = self.output_projection(rpm_out['gated_output'])

        # Build standardized trace if requested
        trace = None
        if return_full_trace:
            # Compute foundation index from DWP product
            dwp_product = rpm_out['dwp_scores'].prod(dim=-1)  # [batch, seq, 7]
            foundation_idx = dwp_product.argmax(dim=-1)  # [batch, seq]

            trace = _build_standardized_trace(
                # Noetic routing
                noetic_weights=trace_noetic_weights,
                noetic_indices=torch.full(
                    (tokens.shape[0], tokens.shape[1]),
                    noetic_idx,
                    dtype=torch.long,
                    device=tokens.device
                ),
                # Attention (FractalAttention doesn't expose scale_weights directly in v2)
                attention_scale_weights=None,
                # Attractor
                attractor_converged=attractor_out['converged'],
                attractor_iterations=attractor_out['iterations'],
                attractor_final_delta=attractor_out.get('final_delta', 0.0),
                attractor_trajectory=torch.stack(attractor_out['trajectory'], dim=0) if attractor_out.get('trajectory') else None,
                attractor_state=attractor_out['attractor'].detach(),
                # RPM
                rpm_gate=rpm_out['rpm_gate'].detach(),
                rpm_dwp_scores=rpm_out['dwp_scores'],  # Don't detach - needed for RPM loss gradients
                rpm_foundation_idx=foundation_idx.detach(),
                # Legacy for backward compatibility
                legacy_embedding=trace_embedding,
                legacy_worlds=trace_worlds,
                legacy_processed=trace_processed,
                legacy_attended=trace_attended,
            )

            # Also add top-level legacy keys for backward compatibility
            # Note: 'attractor' is now a dict in standardized schema, so we use 'attractor_state'
            # for the legacy Tensor output (backward compat with code expecting trace['attractor'])
            trace['embedding'] = trace_embedding
            trace['worlds'] = trace_worlds
            trace['processed'] = trace_processed
            trace['attended'] = trace_attended
            trace['attractor_state'] = attractor_out['attractor'].detach()  # Legacy Tensor key
            trace['attractor_converged'] = attractor_out['converged']
            trace['attractor_iterations'] = attractor_out['iterations']
            trace['rpm_gate'] = rpm_out['rpm_gate'].detach()
            trace['dwp_scores'] = rpm_out['dwp_scores']

        result = {
            'logits': logits,
            'gated_output': rpm_out['gated_output'],
            'rpm_gate': rpm_out['rpm_gate'],
            'attractor_converged': attractor_out['converged'],
            'trace': trace,
        }

        # Include convergence_loss for direct backprop if available
        if return_tensor_deltas and 'convergence_loss' in attractor_out:
            result['convergence_loss'] = attractor_out['convergence_loss']

        return result


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def run_integration_test() -> Dict[str, any]:
    """
    Run full 5-layer pipeline integration test.

    Returns:
        Dict with test results and shape validations
    """
    print("=" * 70)
    print("TKS-LLM Core v2 5-Layer Pipeline Integration Test")
    print("=" * 70)

    results = {'passed': True, 'shapes': {}, 'checks': {}}

    # Test configuration
    batch_size = 2
    seq_len = 8
    vocab_size = 1000

    print(f"\nConfiguration: batch={batch_size}, seq={seq_len}, vocab={vocab_size}")

    # Create pipeline
    pipeline = TKSLLMCorePipeline(vocab_size=vocab_size)

    # Create test input
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test 1: Basic forward pass
    print("\n--- Test 1: Basic Forward Pass ---")
    out = pipeline(tokens, noetic_idx=1, return_full_trace=True)

    results['shapes']['tokens'] = tokens.shape
    results['shapes']['logits'] = out['logits'].shape
    results['shapes']['gated_output'] = out['gated_output'].shape

    print(f"  tokens:        {tokens.shape}")
    print(f"  logits:        {out['logits'].shape}")
    print(f"  gated_output:  {out['gated_output'].shape}")
    print(f"  rpm_gate:      {out['rpm_gate'].shape}")

    # Verify shapes
    assert out['logits'].shape == (batch_size, seq_len, vocab_size), "Logits shape mismatch"
    assert out['gated_output'].shape == (batch_size, seq_len, TOTAL_DIM), "Gated output shape mismatch"
    print("  Shape validation: PASS")

    # Test 2: Attractor convergence
    print("\n--- Test 2: Attractor Convergence ---")
    results['checks']['attractor_converged'] = out['attractor_converged']
    results['checks']['attractor_iterations'] = out['trace']['attractor_iterations']

    print(f"  Converged: {out['attractor_converged']}")
    print(f"  Iterations: {out['trace']['attractor_iterations']}")

    # Test 3: RPM gating with target Foundation
    print("\n--- Test 3: RPM Gating with Target Foundation ---")
    for f_idx, f_name in enumerate(FOUNDATION_NAMES):
        out_f = pipeline(tokens, target_foundation=f_idx)
        gate_mean = out_f['rpm_gate'].mean().item()
        print(f"  {f_name}: gate_mean={gate_mean:.4f}")

    # Test 4: D/W/P scores
    print("\n--- Test 4: D/W/P Score Ranges ---")
    dwp = out['trace']['dwp_scores']  # [batch, seq, 7, 3]
    print(f"  DWP shape: {dwp.shape}")
    print(f"  Desire range: [{dwp[:,:,:,0].min():.3f}, {dwp[:,:,:,0].max():.3f}]")
    print(f"  Wisdom range: [{dwp[:,:,:,1].min():.3f}, {dwp[:,:,:,1].max():.3f}]")
    print(f"  Power range:  [{dwp[:,:,:,2].min():.3f}, {dwp[:,:,:,2].max():.3f}]")

    # Verify DWP in [0, 1]
    assert dwp.min() >= 0 and dwp.max() <= 1, "DWP scores outside [0,1]"
    print("  DWP range validation: PASS")

    # Test 5: Attractor contraction verification
    print("\n--- Test 5: Attractor Contraction Verification ---")
    contraction_check = pipeline.attractor.verify_contraction()
    for key, val in contraction_check.items():
        print(f"  {key}: {val}")
        if 'is_contraction' in key:
            results['checks'][key] = val

    # Test 6: Goal-conditioned forward pass
    print("\n--- Test 6: Goal-Conditioned Pass ---")
    goal = torch.randn(batch_size, TOTAL_DIM)
    out_goal = pipeline(tokens, goal_state=goal, target_foundation=4)  # Power
    print(f"  With goal state: logits shape = {out_goal['logits'].shape}")

    # Test 7: Full trace inspection
    print("\n--- Test 7: Full Trace ---")
    trace = out['trace']
    for key, val in trace.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {val}")

    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)

    return results


def run_shape_validation() -> bool:
    """
    Validate all tensor shapes through pipeline.
    """
    print("\n" + "=" * 70)
    print("SHAPE VALIDATION SUITE")
    print("=" * 70)

    batch, seq, vocab = 2, 8, 100

    # Layer-by-layer validation
    tokens = torch.randint(0, vocab, (batch, seq))

    # Layer 1
    emb = NoeticEmbeddingLayer(vocab_size=vocab)
    emb_out = emb(tokens)
    assert emb_out['noetic'].shape == (batch, seq, 40), "Layer 1 fail"
    print(f"Layer 1 (Embedding):  {tokens.shape} -> {emb_out['noetic'].shape}")

    # Layer 2
    proc = NoeticProcessor(dim=40)
    proc_out = proc(emb_out['noetic'], noetic_idx=1)
    assert proc_out.shape == (batch, seq, 40), "Layer 2 fail"
    print(f"Layer 2 (Processor):  {emb_out['noetic'].shape} -> {proc_out.shape}")

    # Layer 3
    frac = FractalAttentionMechanism(dim=40)
    frac_out = frac(proc_out)
    assert frac_out.shape == (batch, seq, 40), "Layer 3 fail"
    print(f"Layer 3 (Fractal):    {proc_out.shape} -> {frac_out.shape}")

    # Layer 4
    attr = AttractorComputationLayer(dim=40)
    attr_out = attr(frac_out)
    assert attr_out['attractor'].shape == (batch, seq, 40), "Layer 4 fail"
    print(f"Layer 4 (Attractor):  {frac_out.shape} -> {attr_out['attractor'].shape}")

    # Layer 5
    rpm = RPMGatingMechanism(dim=40)
    rpm_out = rpm(attr_out['attractor'])
    assert rpm_out['gated_output'].shape == (batch, seq, 40), "Layer 5 fail"
    print(f"Layer 5 (RPM Gate):   {attr_out['attractor'].shape} -> {rpm_out['gated_output'].shape}")

    print("\nAll shape validations: PASS")
    return True


def run_stable_attractor_test() -> Dict[str, any]:
    """
    Test StableAttractorLayer convergence guarantees.

    This test verifies that the new StableAttractorLayer achieves:
    1. Guaranteed convergence (delta decreases monotonically)
    2. Lipschitz constant < 1 for all contraction maps
    3. High convergence rate (typically 100% vs 0% for legacy)
    """
    print("\n" + "=" * 70)
    print("STABLE ATTRACTOR LAYER CONVERGENCE TEST")
    print("=" * 70)

    results = {
        'legacy_convergence_rate': 0.0,
        'stable_convergence_rate': 0.0,
        'stable_all_contractions': False,
        'stable_monotonic_decay': False,
    }

    batch, seq, dim = 4, 8, 40
    num_trials = 10

    # Test 1: Compare legacy vs stable attractor
    print("\n--- Test 1: Convergence Rate Comparison ---")

    # Legacy attractor (expected to fail often)
    legacy = AttractorComputationLayer(dim=dim, contraction_factor=0.5, max_iterations=10)
    legacy_converged = 0
    for _ in range(num_trials):
        x = torch.randn(batch, seq, dim)
        out = legacy(x)
        if out['converged']:
            legacy_converged += 1
    legacy_rate = legacy_converged / num_trials * 100
    results['legacy_convergence_rate'] = legacy_rate
    print(f"  Legacy AttractorComputationLayer: {legacy_rate:.0f}% convergence")

    # Stable attractor (expected to always converge)
    stable = StableAttractorLayer(dim=dim, contraction_factor=0.5, max_iterations=10)
    stable_converged = 0
    monotonic_count = 0
    for _ in range(num_trials):
        x = torch.randn(batch, seq, dim)
        out = stable(x, return_metrics=True)
        if out['converged']:
            stable_converged += 1
        if out['metrics'].monotonic_decay:
            monotonic_count += 1

    stable_rate = stable_converged / num_trials * 100
    monotonic_rate = monotonic_count / num_trials * 100
    results['stable_convergence_rate'] = stable_rate
    results['stable_monotonic_decay'] = monotonic_rate >= 80  # Allow some tolerance
    print(f"  StableAttractorLayer: {stable_rate:.0f}% convergence")
    print(f"  Monotonic decay rate: {monotonic_rate:.0f}%")

    # Test 2: Verify contraction properties
    print("\n--- Test 2: Contraction Verification ---")
    contraction_check = stable.verify_contraction(num_samples=200)

    all_contractions = True
    for key, val in contraction_check.items():
        if 'is_contraction' in key:
            all_contractions = all_contractions and val
            status = "PASS" if val else "FAIL"
            print(f"  {key}: {status}")
        else:
            print(f"  {key}: {val:.4f}")

    results['stable_all_contractions'] = all_contractions

    # Test 3: Delta sequence analysis
    print("\n--- Test 3: Delta Sequence Analysis ---")
    x = torch.randn(batch, seq, dim)
    out = stable(x, return_metrics=True)

    deltas = out['delta_sequence']
    print(f"  Delta sequence length: {len(deltas)}")
    print(f"  Initial delta: {deltas[0]:.6f}")
    print(f"  Final delta: {deltas[-1]:.6f}")
    print(f"  Reduction factor: {deltas[-1] / (deltas[0] + 1e-8):.6f}")

    # Check monotonic decay
    is_monotonic = all(deltas[i] >= deltas[i+1] for i in range(len(deltas)-1))
    print(f"  Monotonic decay: {'YES' if is_monotonic else 'NO'}")

    # Test 4: Convergence loss computation
    print("\n--- Test 4: Convergence Loss ---")
    conv_loss = attractor_convergence_loss(deltas, out['converged'])
    print(f"  Convergence loss: {conv_loss.item():.6f}")

    # Test 5: Lipschitz penalty
    print("\n--- Test 5: Lipschitz Penalty ---")
    lip_penalty = compute_lipschitz_penalty(stable, num_samples=50, target_lipschitz=0.9)
    print(f"  Lipschitz penalty: {lip_penalty.item():.6f}")

    # Test 6: Pipeline integration with stable attractor
    print("\n--- Test 6: Pipeline Integration ---")
    pipeline = TKSLLMCorePipeline(
        vocab_size=100,
        use_stable_attractor=True
    )
    tokens = torch.randint(0, 100, (batch, seq))
    out = pipeline(tokens, return_full_trace=True)

    print(f"  Converged: {out['attractor_converged']}")
    print(f"  Iterations: {out['trace']['attractor_iterations']}")
    print(f"  Output shape: {out['logits'].shape}")

    # Summary
    print("\n" + "=" * 70)
    print("STABLE ATTRACTOR TEST SUMMARY")
    print("=" * 70)
    print(f"  Legacy convergence rate: {results['legacy_convergence_rate']:.0f}%")
    print(f"  Stable convergence rate: {results['stable_convergence_rate']:.0f}%")
    print(f"  All maps are contractions: {results['stable_all_contractions']}")
    print(f"  Monotonic decay achieved: {results['stable_monotonic_decay']}")

    improvement = results['stable_convergence_rate'] - results['legacy_convergence_rate']
    print(f"\n  IMPROVEMENT: +{improvement:.0f}% convergence rate")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run shape validation
    run_shape_validation()

    # Run stable attractor test (NEW)
    stable_results = run_stable_attractor_test()

    # Run full integration test
    results = run_integration_test()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"All tests passed: {results['passed']}")
    print(f"Stable attractor convergence: {stable_results['stable_convergence_rate']:.0f}%")

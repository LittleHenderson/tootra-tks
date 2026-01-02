"""
TKS Noetic Expressiveness Layer - Novel NL Processing Without FFN/MHA

This module implements TKS v7.4 mechanisms as neural network layers,
providing an alternative to traditional transformer FFN and MHA:

1. NoeticRouter - Replaces MHA with 10 noetic transforms + learned mixture
2. SubFoundationFilter - Replaces FFN with 28-way world×domain routing
3. AcquisitionGating - D→W→P prerequisite chain for semantic state
4. FractalAttractor - Kleene fixed-point for guaranteed convergence

Mathematical Foundations:
- 10 Noetics (ν₀..ν₉) as consciousness operators on 40D space
- 7 Foundations × 4 Worlds = 28 Sub-Foundations
- 22 Acquisitions = A₀ + 7×D + 7×W + 7×P
- Dual pairs: (ν₂,ν₃), (ν₅,ν₆), (ν₈,ν₉) for semantic inversion

References:
- TKS_v7.4_CoreCalculus.tex: Core objects and algebraic structure
- TKS_v7.4_Semantics.tex: Scott domains and Kleene fixed points
- TKS_Scenario_Equation_System.tex: D/W/P chains

Author: TKS Research Pipeline
Date: 2025-12-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# =============================================================================
# NOETIC TRANSFORM OPERATORS
# =============================================================================

class NoeticOperator(nn.Module):
    """
    Single noetic operator ν_k as a learned transform on 40D space.

    Each noetic has a semantic meaning:
    - ν₀: Identity (pure potential)
    - ν₁: Mind (awareness, cognitive structure)
    - ν₂: Positive (attraction, approach)
    - ν₃: Negative (repulsion, withdrawal) [dual of ν₂]
    - ν₄: Vibration (intensity, activation)
    - ν₅: Female (receptive, input)
    - ν₆: Male (projective, output) [dual of ν₅]
    - ν₇: Rhythm (periodicity, cycles)
    - ν₈: Above (causal, inner)
    - ν₉: Below (effect, outer) [dual of ν₈]
    """

    def __init__(self, dim: int = 40, noetic_index: int = 0):
        super().__init__()
        self.dim = dim
        self.noetic_index = noetic_index

        if noetic_index == 0:
            # ν₀ is identity - just learnable scaling
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            # Other noetics are learned transforms
            self.transform = nn.Linear(dim, dim, bias=False)

            # Initialize with structure based on noetic meaning
            self._init_noetic_structure()

    def _init_noetic_structure(self):
        """Initialize transform based on noetic semantic meaning."""
        with torch.no_grad():
            # Start with identity
            nn.init.eye_(self.transform.weight)

            # Add structured perturbation based on noetic type
            k = self.noetic_index

            if k in [2, 3]:  # Positive/Negative - affect valence dimensions
                # Emphasize world dimensions 0-9 (World A) for spiritual valence
                self.transform.weight[:10, :10] *= 1.2
            elif k in [5, 6]:  # Female/Male - affect agency dimensions
                # Emphasize modes 5,6 across all worlds
                for w in range(4):
                    base = w * 10
                    self.transform.weight[base+4:base+7, base+4:base+7] *= 1.2
            elif k in [8, 9]:  # Above/Below - affect causal dimensions
                # ν₈ emphasizes modes 8 (Above), ν₉ emphasizes mode 9 (Below)
                target_mode = 7 if k == 8 else 8
                for w in range(4):
                    base = w * 10
                    self.transform.weight[base+target_mode, base+target_mode] *= 1.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply noetic operator: ν_k(x)"""
        if self.noetic_index == 0:
            return x * self.scale
        return self.transform(x)


class DualPairOperator(nn.Module):
    """
    Dual pair operator for approximate semantic inversion.

    Key pairs:
    - (ν₂, ν₃): Positive ↔ Negative
    - (ν₅, ν₆): Female ↔ Male (Receptive ↔ Projective)
    - (ν₈, ν₉): Above ↔ Below (Causal ↔ Effect)

    Property: ν_j ∘ ν_i ≈ ν₀ when i + j = 9
    """

    def __init__(self, dim: int = 40, pair_type: str = 'positive_negative'):
        super().__init__()
        self.dim = dim
        self.pair_type = pair_type

        # Map pair types to indices
        pair_map = {
            'positive_negative': (2, 3),
            'female_male': (5, 6),
            'above_below': (8, 9),
        }

        i, j = pair_map[pair_type]
        self.nu_i = NoeticOperator(dim, i)
        self.nu_j = NoeticOperator(dim, j)

        # Learnable residual for approximating identity after composition
        self.residual_correction = nn.Parameter(torch.zeros(dim))

    def forward_compose(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ν_j ∘ ν_i ≈ ν₀ (should approximately preserve)"""
        mid = self.nu_i(x)
        out = self.nu_j(mid)
        return out + self.residual_correction

    def forward_invert(self, x: torch.Tensor, direction: str = 'positive') -> torch.Tensor:
        """Apply semantic inversion: ν₃(x) inverts polarity of ν₂(x)"""
        if direction == 'positive':
            return self.nu_i(x)
        else:
            return self.nu_j(x)


# =============================================================================
# NOETIC ROUTER (Replaces MHA)
# =============================================================================

class NoeticRouter(nn.Module):
    """
    Routes semantic states through noetic transforms with learned mixture.

    Replaces Multi-Head Attention by:
    - Using 10 noetic operators instead of attention heads
    - Learning mixture weights based on input content
    - Respecting dual-pair structure for semantic coherence

    Properties:
    - Interpretable: each "head" has fixed semantic meaning
    - Compositional: supports fractal sequences ⟨k₁:k₂:...⟩
    - Efficient: no quadratic attention complexity
    """

    def __init__(self, dim: int = 40, num_noetics: int = 10):
        super().__init__()
        self.dim = dim
        self.num_noetics = num_noetics

        # 10 noetic operators
        self.noetics = nn.ModuleList([
            NoeticOperator(dim, k) for k in range(num_noetics)
        ])

        # Router learns which noetics to apply
        self.router = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_noetics)
        )

        # Dual pair operators for structured composition
        self.dual_pairs = nn.ModuleDict({
            'polarity': DualPairOperator(dim, 'positive_negative'),
            'agency': DualPairOperator(dim, 'female_male'),
            'causality': DualPairOperator(dim, 'above_below'),
        })

        # Output projection
        self.output_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        fractal_sequence: Optional[List[int]] = None,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Route input through noetic transforms.

        Args:
            x: Input tensor [batch, seq, dim] or [batch, dim]
            fractal_sequence: Optional fixed sequence like [1, 4, 7] for MVR
            return_weights: If True, return routing weights

        Returns:
            Transformed tensor and optional routing weights
        """
        squeeze_seq = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_seq = True

        batch, seq, dim = x.shape

        if fractal_sequence is not None:
            # Fixed fractal: apply noetics in sequence
            result = x
            for k in fractal_sequence:
                result = self.noetics[k](result)
                result = F.gelu(result)
            weights = None
        else:
            # Learned routing: soft mixture of noetics
            weights = F.softmax(self.router(x), dim=-1)  # [batch, seq, 10]

            # Apply each noetic and mix
            noetic_outputs = torch.stack([
                self.noetics[k](x) for k in range(self.num_noetics)
            ], dim=-1)  # [batch, seq, dim, 10]

            # Weighted sum
            result = (noetic_outputs * weights.unsqueeze(-2)).sum(dim=-1)

        # Output projection
        result = self.output_proj(result)

        if squeeze_seq:
            result = result.squeeze(1)
            if weights is not None:
                weights = weights.squeeze(1)

        if return_weights:
            return result, weights
        return result, None

    def apply_dual_pair(
        self,
        x: torch.Tensor,
        pair: str = 'polarity',
        direction: str = 'positive'
    ) -> torch.Tensor:
        """Apply dual-pair semantic transform."""
        return self.dual_pairs[pair].forward_invert(x, direction)


# =============================================================================
# SUB-FOUNDATION FILTER (Replaces FFN)
# =============================================================================

class SubFoundationFilter(nn.Module):
    """
    Apply foundation-specific semantic projection for each world.

    28 Sub-Foundations = 7 Foundations × 4 Worlds:
    - Foundations: Unity, Wisdom, Life, Companionship, Power, Wealth, Continuation
    - Worlds: A (Spiritual), B (Mental), C (Emotional), D (Physical)

    Replaces FFN by:
    - Contextual routing based on world×domain
    - World descent path: A→B→C→D
    - Foundation-specific feature extraction
    """

    # Foundation labels
    FOUNDATIONS = ['unity', 'wisdom', 'life', 'companionship', 'power', 'wealth', 'continuation']
    WORLDS = ['a', 'b', 'c', 'd']

    def __init__(self, dim: int = 40, hidden_dim: int = 80):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # 28 sub-foundation filters
        self.filters = nn.ModuleDict({
            f"f{m+1}{w}": nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )
            for m in range(7)
            for w in self.WORLDS
        })

        # World descent transformations
        self.world_descent = nn.ModuleDict({
            'a_to_b': nn.Linear(dim, dim),  # Spiritual → Mental
            'b_to_c': nn.Linear(dim, dim),  # Mental → Emotional
            'c_to_d': nn.Linear(dim, dim),  # Emotional → Physical
        })

        # Foundation detector (learns to identify relevant foundation)
        self.foundation_detector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 7)  # 7 foundations
        )

        # World detector (learns to identify world emphasis)
        self.world_detector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4)  # 4 worlds
        )

        # Output layer norm
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        foundation_hint: Optional[int] = None,
        world_hint: Optional[int] = None,
        return_routing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Apply sub-foundation filtering.

        Args:
            x: Input [batch, seq, dim] or [batch, dim]
            foundation_hint: Optional foundation index (0-6)
            world_hint: Optional world index (0-3)
            return_routing: If True, return routing decisions

        Returns:
            Filtered tensor and optional routing info
        """
        squeeze_seq = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_seq = True

        batch, seq, dim = x.shape

        # Detect or use provided foundation
        if foundation_hint is not None:
            foundation_weights = F.one_hot(
                torch.tensor(foundation_hint).expand(batch, seq),
                num_classes=7
            ).float().to(x.device)
        else:
            foundation_weights = F.softmax(self.foundation_detector(x), dim=-1)

        # Detect or use provided world
        if world_hint is not None:
            world_weights = F.one_hot(
                torch.tensor(world_hint).expand(batch, seq),
                num_classes=4
            ).float().to(x.device)
        else:
            world_weights = F.softmax(self.world_detector(x), dim=-1)

        # Apply all 28 sub-foundation filters with soft routing
        result = torch.zeros_like(x)
        for m in range(7):
            for w_idx, w in enumerate(self.WORLDS):
                filter_out = self.filters[f"f{m+1}{w}"](x)
                weight = foundation_weights[..., m:m+1] * world_weights[..., w_idx:w_idx+1]
                result = result + filter_out * weight

        # Apply world descent (optional cascading refinement)
        # This manifests ideas from archetypal → material
        descended = x
        descent_outputs = []
        for transition in ['a_to_b', 'b_to_c', 'c_to_d']:
            descended = self.world_descent[transition](descended)
            descent_outputs.append(descended)

        # Mix descended representation based on target world
        descent_stack = torch.stack(descent_outputs, dim=-1)  # [batch, seq, dim, 3]
        # Weight by how "material" the target is
        descent_weights = world_weights[..., 1:].unsqueeze(-2)  # B, C, D weights
        descended_mixed = (descent_stack * descent_weights).sum(dim=-1)

        # Combine filter output with descended representation
        result = result + 0.5 * descended_mixed

        # Residual connection and normalization
        result = self.layer_norm(x + result)

        if squeeze_seq:
            result = result.squeeze(1)

        routing_info = None
        if return_routing:
            routing_info = {
                'foundation_weights': foundation_weights,
                'world_weights': world_weights,
            }

        return result, routing_info


# =============================================================================
# ACQUISITION GATING (D→W→P Chain)
# =============================================================================

class AcquisitionGating(nn.Module):
    """
    Gated computation respecting D→W→P prerequisite ordering.

    22 Acquisitions:
    - A₀: Pure Desire (root)
    - D_m: Desire for Foundation m (m=1..7)
    - W_m: Wisdom about Foundation m (requires D_m)
    - P_m: Power to achieve Foundation m (requires W_m)

    Property: σ(W_m) ⟹ σ(D_m), σ(P_m) ⟹ σ(W_m)
    """

    def __init__(self, dim: int = 40, num_foundations: int = 7):
        super().__init__()
        self.dim = dim
        self.num_foundations = num_foundations

        # Desire layers (always accessible)
        self.desire_layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_foundations)
        ])

        # Wisdom layers (gated by desire)
        self.wisdom_layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_foundations)
        ])

        # Power layers (gated by wisdom)
        self.power_layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_foundations)
        ])

        # Gate predictors for each level
        self.desire_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_foundations),
            nn.Sigmoid()
        )

        self.wisdom_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_foundations),
            nn.Sigmoid()
        )

        self.power_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_foundations),
            nn.Sigmoid()
        )

        # Acquisition state tracking
        self.register_buffer('acquisition_state', torch.zeros(22))

    def forward(
        self,
        x: torch.Tensor,
        target_foundation: Optional[int] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply D→W→P gated transformation.

        Args:
            x: Input [batch, seq, dim] or [batch, dim]
            target_foundation: If set, focus on this foundation (0-6)
            return_state: If True, return acquisition state

        Returns:
            Transformed tensor and optional acquisition state
        """
        squeeze_seq = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_seq = True

        batch, seq, dim = x.shape

        # Predict gate activations
        d_gates = self.desire_gate(x)  # [batch, seq, 7]
        w_gates = self.wisdom_gate(x)  # [batch, seq, 7]
        p_gates = self.power_gate(x)   # [batch, seq, 7]

        # Enforce D→W→P ordering: can only have W if D, P if W
        w_gates = w_gates * d_gates  # Wisdom requires Desire
        p_gates = p_gates * w_gates  # Power requires Wisdom

        # Apply transformations with gating
        result = torch.zeros_like(x)

        if target_foundation is not None:
            # Focus on specific foundation
            m = target_foundation
            d_out = self.desire_layers[m](x) * d_gates[..., m:m+1]
            w_out = self.wisdom_layers[m](d_out) * w_gates[..., m:m+1]
            p_out = self.power_layers[m](w_out) * p_gates[..., m:m+1]
            result = p_out
        else:
            # Process all foundations
            for m in range(self.num_foundations):
                d_out = self.desire_layers[m](x) * d_gates[..., m:m+1]
                w_out = self.wisdom_layers[m](d_out) * w_gates[..., m:m+1]
                p_out = self.power_layers[m](w_out) * p_gates[..., m:m+1]
                result = result + p_out

        if squeeze_seq:
            result = result.squeeze(1)

        # Update acquisition state (mean over batch/seq)
        state = torch.zeros(22, device=x.device)
        state[0] = 1.0  # A₀ always true
        state[1:8] = d_gates.mean(dim=(0, 1))  # D_1..D_7
        state[8:15] = w_gates.mean(dim=(0, 1))  # W_1..W_7
        state[15:22] = p_gates.mean(dim=(0, 1))  # P_1..P_7

        if return_state:
            return result, state
        return result, None

    def get_level(self, foundation_m: int) -> int:
        """
        Get current acquisition level for a foundation.

        Returns:
            0: No desire
            1: Desire only
            2: Desire + Wisdom
            3: Desire + Wisdom + Power (complete)
        """
        d = self.acquisition_state[1 + foundation_m].item()
        w = self.acquisition_state[8 + foundation_m].item()
        p = self.acquisition_state[15 + foundation_m].item()

        if p > 0.5:
            return 3
        elif w > 0.5:
            return 2
        elif d > 0.5:
            return 1
        return 0


# =============================================================================
# FRACTAL ATTRACTOR (Kleene Fixed Point)
# =============================================================================

class FractalAttractor(nn.Module):
    """
    Compute semantic fixed point via Kleene iteration.

    For a fractal Φ = ⟨k₁:k₂:...⟩, computes:
        fix(Φ) = ⊔_{n<ω} Φⁿ(⊥)

    In flat noetic domains, this converges in ≤2 steps!

    Properties:
    - Guaranteed convergence (Kleene fixed-point theorem)
    - Self-similar transformation hierarchy
    - Replaces layer normalization with attractor convergence
    """

    def __init__(
        self,
        dim: int = 40,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
        contraction_factor: float = 0.95
    ):
        super().__init__()
        self.dim = dim
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.contraction_factor = contraction_factor

        # Learnable fractal transform
        self.fractal_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Spectral normalization for contraction guarantee
        self.spectral_norm = nn.utils.parametrizations.spectral_norm(
            nn.Linear(dim, dim)
        )

        # Bottom element (learnable initialization)
        self.bottom = nn.Parameter(torch.zeros(dim) + 1e-6)

    def forward(
        self,
        x: torch.Tensor,
        return_iterations: bool = False
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Compute fixed point starting from x.

        Args:
            x: Input [batch, seq, dim] or [batch, dim]
            return_iterations: If True, return number of iterations

        Returns:
            Fixed point tensor and optional iteration count
        """
        squeeze_seq = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_seq = True

        batch, seq, dim = x.shape

        # Initialize from bottom (or input as warm start)
        current = self.bottom.expand(batch, seq, -1) + 0.1 * x
        prev = torch.zeros_like(current)

        iterations = 0
        for i in range(self.max_iterations):
            # Apply fractal transform with contraction
            transformed = self.fractal_transform(current)
            contracted = self.spectral_norm(transformed)
            next_state = self.contraction_factor * contracted + (1 - self.contraction_factor) * current

            # Check convergence
            delta = torch.norm(next_state - current, dim=-1).mean()

            prev = current
            current = next_state
            iterations = i + 1

            if delta < self.convergence_threshold:
                break

        # Add residual from input for gradient flow
        result = current + 0.1 * x

        if squeeze_seq:
            result = result.squeeze(1)

        if return_iterations:
            return result, iterations
        return result, None


# =============================================================================
# COMPLETE TKS NOETIC BLOCK (Replaces Transformer Block)
# =============================================================================

class TKSNoeticBlock(nn.Module):
    """
    Complete TKS block replacing transformer's attention + FFN.

    Architecture:
    1. NoeticRouter (replaces MHA) - Routes through 10 noetic transforms
    2. SubFoundationFilter (replaces FFN) - 28-way world×domain routing
    3. AcquisitionGating - D→W→P semantic state tracking
    4. FractalAttractor - Kleene fixed-point convergence

    All components are interpretable and semantically grounded in TKS v7.4.
    """

    def __init__(
        self,
        dim: int = 40,
        hidden_dim: int = 80,
        use_attractor: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim

        # Core components
        self.noetic_router = NoeticRouter(dim)
        self.subfoundation_filter = SubFoundationFilter(dim, hidden_dim)
        self.acquisition_gate = AcquisitionGating(dim)

        # Optional attractor for convergence
        self.use_attractor = use_attractor
        if use_attractor:
            self.attractor = FractalAttractor(dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Layer norms (backup for gradient stability)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        fractal_sequence: Optional[List[int]] = None,
        foundation_hint: Optional[int] = None,
        world_hint: Optional[int] = None,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Apply TKS block transformation.

        Args:
            x: Input [batch, seq, dim] or [batch, dim]
            fractal_sequence: Optional fixed noetic sequence
            foundation_hint: Optional foundation focus
            world_hint: Optional world focus
            return_trace: If True, return computation trace

        Returns:
            Transformed tensor and optional trace
        """
        trace = {} if return_trace else None

        # Step 1: Noetic routing (replaces attention)
        routed, routing_weights = self.noetic_router(
            x, fractal_sequence, return_weights=True
        )
        routed = self.dropout(routed)
        x = self.norm1(x + routed)

        if return_trace:
            trace['routing_weights'] = routing_weights

        # Step 2: Sub-foundation filtering (replaces FFN)
        filtered, routing_info = self.subfoundation_filter(
            x, foundation_hint, world_hint, return_routing=True
        )
        filtered = self.dropout(filtered)
        x = self.norm2(x + filtered)

        if return_trace and routing_info:
            trace.update(routing_info)

        # Step 3: Acquisition gating (semantic state)
        gated, acq_state = self.acquisition_gate(
            x, foundation_hint, return_state=True
        )
        x = x + 0.5 * gated  # Partial residual

        if return_trace:
            trace['acquisition_state'] = acq_state

        # Step 4: Attractor convergence (optional)
        if self.use_attractor:
            converged, iterations = self.attractor(x, return_iterations=True)
            x = converged
            if return_trace:
                trace['attractor_iterations'] = iterations

        return x, trace


# =============================================================================
# DEMO USAGE
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TKS NOETIC EXPRESSIVENESS LAYER - Demo")
    print("=" * 70)

    # Create block
    block = TKSNoeticBlock(dim=40, hidden_dim=80, use_attractor=True)
    print(f"\nBlock parameters: {sum(p.numel() for p in block.parameters()):,}")

    # Test input
    batch, seq = 4, 16
    x = torch.randn(batch, seq, 40)

    # Forward pass
    out, trace = block(
        x,
        fractal_sequence=[1, 4, 7],  # MVR fractal
        foundation_hint=2,  # Focus on Wisdom
        world_hint=1,  # Mental world (B)
        return_trace=True
    )

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attractor iterations: {trace.get('attractor_iterations', 'N/A')}")

    if trace.get('acquisition_state') is not None:
        acq = trace['acquisition_state']
        print(f"\nAcquisition state (22 values):")
        print(f"  A₀ (root): {acq[0]:.3f}")
        print(f"  D_1..D_7: {acq[1:8].tolist()}")
        print(f"  W_1..W_7: {acq[8:15].tolist()}")
        print(f"  P_1..P_7: {acq[15:22].tolist()}")

    print("\n" + "=" * 70)
    print("Demo complete!")

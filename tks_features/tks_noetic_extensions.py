"""
TKS Noetic Extensions - Pure Linear Expressiveness Stack

Three TKS-native modules for NL expressiveness without FFN/MHA/Transformer:

1. SubFoundationMixer (28 sub-foundations)
   - Per-world 28-way routing to learned basis vectors
   - Pure linear: softmax(W @ x) @ basis

2. AcquisitionProgram (22 acquisition operators)
   - Sparse 10x10 operators affecting only 2 noetics per acquisition
   - Canon paths from ACQUISITION_PATHS
   - Multi-step composition with routing

3. FoundationGraphDiffusion (7 foundations)
   - Project to 7D, diffuse over canon adjacency, project back
   - Uses FOUNDATION_ADJ (canon opposites + soft hub)

All modules are pure linear (no MLP hidden layers).
Designed for integration into NoeticBlock post-attention, pre-attractor.

Reference: Design Spec - TKS Expressiveness Stack (Hybrid Canon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

# Import canonical data
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_rules.acquisition_paths import (
    ACQUISITION_PATHS,
    ACQUISITION_ORDER,
    ACQUISITION_CATEGORY,
    ACQUISITION_FOUNDATION,
)
from tks_rules.foundation_graph import (
    FOUNDATION_ADJ,
    build_foundation_adj,
)


# =============================================================================
# 1. SUBFOUNDATION MIXER (28 sub-foundations)
# =============================================================================

class SubFoundationMixer(nn.Module):
    """
    Mix semantics through 28 sub-foundation basis vectors per world.

    28 Sub-Foundations = 7 Foundations × 4 Worlds (a,b,c,d)

    For each world slice (10D):
    - Compute 28-way mixture weights via single linear map
    - Output = softmax(W_mix @ x) @ basis_28x10
    - Residual-gated: x = x + scale * output

    No MLP - pure linear routing + basis projection.
    """

    def __init__(
        self,
        dim: int = 40,
        num_foundations: int = 7,
        num_worlds: int = 4,
        scale: float = 0.1,
        per_world_basis: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.noetic_dim = dim // num_worlds  # 10
        self.num_foundations = num_foundations
        self.num_worlds = num_worlds
        self.num_subfoundations = num_foundations * num_worlds  # 28
        self.scale = scale
        self.per_world_basis = per_world_basis

        if per_world_basis:
            # Separate mixer and basis for each world
            self.mixers = nn.ModuleList([
                nn.Linear(self.noetic_dim, self.num_subfoundations, bias=False)
                for _ in range(num_worlds)
            ])
            self.bases = nn.ParameterList([
                nn.Parameter(torch.randn(self.num_subfoundations, self.noetic_dim) * 0.02)
                for _ in range(num_worlds)
            ])
        else:
            # Shared across worlds
            self.mixer = nn.Linear(self.noetic_dim, self.num_subfoundations, bias=False)
            self.basis = nn.Parameter(
                torch.randn(self.num_subfoundations, self.noetic_dim) * 0.02
            )

        # Learnable scale gate
        self.gate = nn.Parameter(torch.tensor(scale))

        self._init_weights()

    def _init_weights(self):
        """Initialize with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply sub-foundation mixing.

        Args:
            x: Input [batch, seq, 40] or [batch, 40]
            return_weights: If True, return per-world mixture weights

        Returns:
            Mixed tensor and optional weights
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        batch, seq, dim = x.shape
        output = torch.zeros_like(x)
        all_weights = [] if return_weights else None

        # Process each world slice
        for w in range(self.num_worlds):
            start = w * self.noetic_dim
            end = start + self.noetic_dim
            x_w = x[..., start:end]  # [batch, seq, 10]

            if self.per_world_basis:
                logits = self.mixers[w](x_w)  # [batch, seq, 28]
                basis = self.bases[w]  # [28, 10]
            else:
                logits = self.mixer(x_w)
                basis = self.basis

            weights = F.softmax(logits, dim=-1)  # [batch, seq, 28]
            mixed = torch.matmul(weights, basis)  # [batch, seq, 10]

            output[..., start:end] = mixed

            if return_weights:
                all_weights.append(weights)

        # Residual gate
        result = x + self.gate * output

        if squeeze:
            result = result.squeeze(1)

        weights_out = torch.stack(all_weights, dim=-1) if return_weights else None
        return result, weights_out


# =============================================================================
# 2. ACQUISITION PROGRAM (22 sparse operators)
# =============================================================================

class SparseAcquisitionOp(nn.Module):
    """
    Single sparse acquisition operator.

    Each acquisition affects exactly 2 noetics (src, dst) from its Tree Path.
    The operator is a 10x10 identity matrix with a learnable 2x2 block
    at positions (src, dst).
    """

    def __init__(
        self,
        src: int,
        dst: int,
        noetic_dim: int = 10,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.src = src
        self.dst = dst
        self.noetic_dim = noetic_dim

        # Learnable 2x2 block parameters
        # [src,src], [src,dst], [dst,src], [dst,dst]
        self.block = nn.Parameter(torch.randn(2, 2) * init_scale)

        # Per-acquisition scalar
        self.scalar = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse operator.

        Args:
            x: [batch, seq, 10] or [batch, 10]

        Returns:
            Transformed tensor (same shape)
        """
        # Start with identity transform
        out = x.clone()

        # Extract the two affected dimensions
        x_src = x[..., self.src]
        x_dst = x[..., self.dst]

        # Apply 2x2 block transform
        # new_src = block[0,0]*src + block[0,1]*dst
        # new_dst = block[1,0]*src + block[1,1]*dst
        new_src = self.block[0, 0] * x_src + self.block[0, 1] * x_dst
        new_dst = self.block[1, 0] * x_src + self.block[1, 1] * x_dst

        # Update output
        out[..., self.src] = new_src
        out[..., self.dst] = new_dst

        return out * self.scalar


class AcquisitionProgram(nn.Module):
    """
    Compositional semantics via 22 sparse acquisition operators.

    Each acquisition operator only touches 2 noetics (from ACQUISITION_PATHS).
    Router predicts distribution over 22 ops, applies weighted sum.
    Multi-step: repeat routing + application for depth steps.

    No MLP - pure linear routing + sparse operator composition.
    """

    def __init__(
        self,
        dim: int = 40,
        num_worlds: int = 4,
        depth: int = 2,
        scale: float = 0.1,
        category_bias: bool = True,
        foundation_gate: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.noetic_dim = dim // num_worlds  # 10
        self.num_worlds = num_worlds
        self.num_acquisitions = 22
        self.depth = depth
        self.scale = scale

        # Build 22 sparse operators
        self.operators = nn.ModuleDict()
        for acq in ACQUISITION_ORDER:
            src, dst = ACQUISITION_PATHS[acq]
            self.operators[acq] = SparseAcquisitionOp(
                src=src, dst=dst, noetic_dim=self.noetic_dim, init_scale=0.1
            )

        # Router: single linear map per world (no MLP)
        self.routers = nn.ModuleList([
            nn.Linear(self.noetic_dim, self.num_acquisitions, bias=False)
            for _ in range(num_worlds)
        ])

        # Optional D/W/P category bias
        self.category_bias = None
        if category_bias:
            # 3 categories: D, W, P (Meta lumped with D)
            self.category_bias = nn.Parameter(torch.zeros(3, self.num_acquisitions))
            self._init_category_bias()

        # Optional per-foundation gating
        self.foundation_gate = None
        if foundation_gate:
            # 7 foundations + 1 for meta
            self.foundation_gate = nn.Parameter(torch.ones(8, self.num_acquisitions))

        # Learnable scale
        self.gate = nn.Parameter(torch.tensor(scale))

    def _init_category_bias(self):
        """Initialize category bias to slightly favor matching categories."""
        with torch.no_grad():
            for i, acq in enumerate(ACQUISITION_ORDER):
                cat = ACQUISITION_CATEGORY[acq]
                if cat == "D" or cat == "Meta":
                    self.category_bias[0, i] = 0.1
                elif cat == "W":
                    self.category_bias[1, i] = 0.1
                elif cat == "P":
                    self.category_bias[2, i] = 0.1

    def forward(
        self,
        x: torch.Tensor,
        category_hint: Optional[int] = None,  # 0=D, 1=W, 2=P
        foundation_hint: Optional[int] = None,  # 0-7
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Apply acquisition program.

        Args:
            x: Input [batch, seq, 40] or [batch, 40]
            category_hint: Optional D/W/P category (0,1,2)
            foundation_hint: Optional foundation focus (0-7)
            return_trace: If True, return routing distributions

        Returns:
            Transformed tensor and optional trace
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        batch, seq, dim = x.shape
        trace = {"routing": [], "entropy": []} if return_trace else None

        # Multi-step program
        for step in range(self.depth):
            step_output = torch.zeros_like(x)

            # Process each world
            for w in range(self.num_worlds):
                start = w * self.noetic_dim
                end = start + self.noetic_dim
                x_w = x[..., start:end]  # [batch, seq, 10]

                # Compute routing logits
                logits = self.routers[w](x_w)  # [batch, seq, 22]

                # Add category bias
                if self.category_bias is not None and category_hint is not None:
                    logits = logits + self.category_bias[category_hint]

                # Add foundation gate
                if self.foundation_gate is not None and foundation_hint is not None:
                    logits = logits * self.foundation_gate[foundation_hint]

                # Softmax routing
                weights = F.softmax(logits, dim=-1)  # [batch, seq, 22]

                if return_trace:
                    trace["routing"].append(weights.detach())
                    entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
                    trace["entropy"].append(entropy.item())

                # Apply weighted sum of operators
                op_outputs = []
                for i, acq in enumerate(ACQUISITION_ORDER):
                    op_out = self.operators[acq](x_w)  # [batch, seq, 10]
                    op_outputs.append(op_out)

                op_stack = torch.stack(op_outputs, dim=-1)  # [batch, seq, 10, 22]
                mixed = (op_stack * weights.unsqueeze(-2)).sum(dim=-1)  # [batch, seq, 10]

                step_output[..., start:end] = mixed

            # Residual update
            x = x + self.gate * (step_output - x)

        if squeeze:
            x = x.squeeze(1)

        return x, trace


# =============================================================================
# 3. FOUNDATION GRAPH DIFFUSION (7 foundations)
# =============================================================================

class FoundationGraphDiffusion(nn.Module):
    """
    Relational semantics via diffusion over foundation graph.

    Per world:
    1. Project 10D → 7D (to foundation space)
    2. Diffuse over canon adjacency (2-3 steps)
    3. Project 7D → 10D (back to noetic space)

    Uses FOUNDATION_ADJ (canon opposites + soft hub).
    No MLP - pure linear projections + fixed graph diffusion.
    """

    def __init__(
        self,
        dim: int = 40,
        num_worlds: int = 4,
        num_foundations: int = 7,
        diffusion_steps: int = 2,
        scale: float = 0.1,
        hub_weight: float = 0.15,
        learnable_adj: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.noetic_dim = dim // num_worlds  # 10
        self.num_worlds = num_worlds
        self.num_foundations = num_foundations
        self.diffusion_steps = diffusion_steps
        self.scale = scale

        # Down projection: 10 → 7 (per world)
        self.down_projs = nn.ModuleList([
            nn.Linear(self.noetic_dim, num_foundations, bias=False)
            for _ in range(num_worlds)
        ])

        # Up projection: 7 → 10 (per world)
        self.up_projs = nn.ModuleList([
            nn.Linear(num_foundations, self.noetic_dim, bias=False)
            for _ in range(num_worlds)
        ])

        # Adjacency matrix
        adj = build_foundation_adj(hub_weight=hub_weight, normalize=True)
        if learnable_adj:
            self.adj = nn.Parameter(adj)
        else:
            self.register_buffer("adj", adj)

        # Optional learnable delta on adjacency
        self.adj_delta = nn.Parameter(torch.zeros(7, 7) * 0.01) if learnable_adj else None

        # Per-step LayerNorm for stability
        self.step_norms = nn.ModuleList([
            nn.LayerNorm(num_foundations) for _ in range(diffusion_steps)
        ])

        # Learnable scale
        self.gate = nn.Parameter(torch.tensor(scale))

        self._init_weights()

    def _init_weights(self):
        """Initialize projections."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_energy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Apply foundation graph diffusion.

        Args:
            x: Input [batch, seq, 40] or [batch, 40]
            return_energy: If True, return diffusion energy

        Returns:
            Diffused tensor and optional energy
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        batch, seq, dim = x.shape
        output = torch.zeros_like(x)
        total_energy = 0.0

        # Get effective adjacency
        adj = self.adj
        if self.adj_delta is not None:
            adj = adj + self.adj_delta
            # Re-normalize rows
            adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Process each world
        for w in range(self.num_worlds):
            start = w * self.noetic_dim
            end = start + self.noetic_dim
            x_w = x[..., start:end]  # [batch, seq, 10]

            # Project to foundation space
            f = self.down_projs[w](x_w)  # [batch, seq, 7]

            # Diffusion steps
            for step in range(self.diffusion_steps):
                f_prev = f
                f = torch.matmul(f, adj.t())  # [batch, seq, 7] @ [7, 7].T
                f = self.step_norms[step](f)

                if return_energy:
                    delta = (f - f_prev).pow(2).sum().item()
                    total_energy += delta

            # Project back to noetic space
            out_w = self.up_projs[w](f)  # [batch, seq, 10]
            output[..., start:end] = out_w

        # Residual gate
        result = x + self.gate * output

        if squeeze:
            result = result.squeeze(1)

        energy = total_energy / (self.num_worlds * self.diffusion_steps) if return_energy else None
        return result, energy


# =============================================================================
# COMBINED TKS EXPRESSIVENESS STACK
# =============================================================================

class TKSExpressivenessStack(nn.Module):
    """
    Combined stack of all three TKS expressiveness modules.

    Order: SubFoundationMixer → AcquisitionProgram → FoundationGraphDiffusion

    Each module has its own LayerNorm + residual gate.
    No FFN, no MHA, no Transformer - pure TKS-native structure.
    """

    def __init__(
        self,
        dim: int = 40,
        # SubFoundationMixer config
        use_subfoundation_mixer: bool = True,
        subfoundation_scale: float = 0.1,
        # AcquisitionProgram config
        use_acquisition_program: bool = True,
        acquisition_scale: float = 0.1,
        acquisition_depth: int = 2,
        # FoundationGraphDiffusion config
        use_foundation_graph: bool = True,
        foundation_scale: float = 0.1,
        diffusion_steps: int = 2,
        foundation_hub_weight: float = 0.15,
    ):
        super().__init__()
        self.dim = dim

        # SubFoundationMixer
        self.use_subfoundation_mixer = use_subfoundation_mixer
        if use_subfoundation_mixer:
            self.subfoundation_mixer = SubFoundationMixer(
                dim=dim, scale=subfoundation_scale
            )
            self.norm_sf = nn.LayerNorm(dim)

        # AcquisitionProgram
        self.use_acquisition_program = use_acquisition_program
        if use_acquisition_program:
            self.acquisition_program = AcquisitionProgram(
                dim=dim, depth=acquisition_depth, scale=acquisition_scale
            )
            self.norm_acq = nn.LayerNorm(dim)

        # FoundationGraphDiffusion
        self.use_foundation_graph = use_foundation_graph
        if use_foundation_graph:
            self.foundation_diffusion = FoundationGraphDiffusion(
                dim=dim,
                diffusion_steps=diffusion_steps,
                scale=foundation_scale,
                hub_weight=foundation_hub_weight,
            )
            self.norm_fd = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        category_hint: Optional[int] = None,
        foundation_hint: Optional[int] = None,
        return_full_trace: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Apply full expressiveness stack.

        Args:
            x: Input [batch, seq, dim] or [batch, dim]
            category_hint: D/W/P category (0,1,2)
            foundation_hint: Foundation focus (0-7)
            return_full_trace: If True, return detailed trace

        Returns:
            Transformed tensor and optional trace
        """
        trace = {} if return_full_trace else None

        # 1. SubFoundationMixer
        if self.use_subfoundation_mixer:
            x_normed = self.norm_sf(x)
            sf_out, sf_weights = self.subfoundation_mixer(
                x_normed, return_weights=return_full_trace
            )
            x = sf_out
            if return_full_trace:
                trace["subfoundation_weights"] = sf_weights

        # 2. AcquisitionProgram
        if self.use_acquisition_program:
            x_normed = self.norm_acq(x)
            acq_out, acq_trace = self.acquisition_program(
                x_normed,
                category_hint=category_hint,
                foundation_hint=foundation_hint,
                return_trace=return_full_trace,
            )
            x = acq_out
            if return_full_trace and acq_trace:
                trace["acquisition_routing"] = acq_trace["routing"]
                trace["acquisition_entropy"] = acq_trace["entropy"]

        # 3. FoundationGraphDiffusion
        if self.use_foundation_graph:
            x_normed = self.norm_fd(x)
            fd_out, energy = self.foundation_diffusion(
                x_normed, return_energy=return_full_trace
            )
            x = fd_out
            if return_full_trace:
                trace["diffusion_energy"] = energy

        return x, trace


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TKS NOETIC EXTENSIONS - Pure Linear Expressiveness Stack")
    print("=" * 70)

    # Create stack
    stack = TKSExpressivenessStack(
        dim=40,
        use_subfoundation_mixer=True,
        use_acquisition_program=True,
        use_foundation_graph=True,
        subfoundation_scale=0.1,
        acquisition_scale=0.1,
        acquisition_depth=2,
        foundation_scale=0.1,
        diffusion_steps=2,
    )

    # Count parameters
    total_params = sum(p.numel() for p in stack.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test forward pass
    batch, seq = 4, 16
    x = torch.randn(batch, seq, 40)

    out, trace = stack(x, return_full_trace=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    if trace:
        print(f"\nTrace keys: {list(trace.keys())}")
        if "acquisition_entropy" in trace:
            print(f"Acquisition entropy: {trace['acquisition_entropy']}")
        if "diffusion_energy" in trace:
            print(f"Diffusion energy: {trace['diffusion_energy']:.6f}")

    # Verify no NaN/Inf
    assert not torch.isnan(out).any(), "NaN in output!"
    assert not torch.isinf(out).any(), "Inf in output!"
    print("\nNo NaN/Inf detected - stack is stable!")

    print("\n" + "=" * 70)
    print("Demo complete!")

"""
TKS Cross-World Bridge Module - Audited Cross-World Mixing

Problem: Strict block-sparsity (per-world isolation) prevents legitimate cross-world
compositions, but we need auditability for all cross-world information flow.

Solution: Explicit, audited bridges with limited capacity:
1. WorldBridge - Low-rank pairwise projections with energy caps
2. FoundationBridge - Foundation-guided cross-world mixing using TKS canon patterns
3. BridgeTraceRecord - First-class trace item for cross-world decisions
4. BridgeConstraintLoss - Regularization for bounded, meaningful bridge usage

All bridge operations are logged in trace for full auditability.

Reference: TKS v7.4 Meta-System - Cross-World Categorical Semantics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import TKS canonical definitions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tks_rules.worlds import (
    WORLD_LETTERS,
    WORLD_OFFSETS,
    WORLD_ORDINALS,
    world_distance,
)
from tks_rules.foundations import (
    FOUNDATIONS,
    FOUNDATION_NAMES,
)


# =============================================================================
# CONSTANTS
# =============================================================================

WORLD_DIM = 10  # Noetic dimension per world
NUM_WORLDS = 4  # A, B, C, D
TOTAL_DIM = WORLD_DIM * NUM_WORLDS  # 40

# World pairs for pairwise bridges (6 pairs for 4 worlds)
# Order: A-B, A-C, A-D, B-C, B-D, C-D
WORLD_PAIRS: List[Tuple[str, str]] = [
    ("A", "B"), ("A", "C"), ("A", "D"),
    ("B", "C"), ("B", "D"),
    ("C", "D"),
]

PAIR_TO_INDEX: Dict[Tuple[str, str], int] = {
    pair: idx for idx, pair in enumerate(WORLD_PAIRS)
}

INDEX_TO_PAIR: Dict[int, Tuple[str, str]] = {
    idx: pair for idx, pair in enumerate(WORLD_PAIRS)
}


# =============================================================================
# 1. AUDITED CROSS-WORLD BRIDGE
# =============================================================================

class LowRankBridge(nn.Module):
    """
    Low-rank projection between two worlds.

    Low-rank = limited bandwidth for cross-world information.
    W_src->dst = U @ V^T where U: [10, rank], V: [10, rank]

    This limits the information capacity of the bridge.
    """

    def __init__(
        self,
        dim: int = WORLD_DIM,
        rank: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.bidirectional = bidirectional

        # Forward projection: src -> dst
        self.U_fwd = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V_fwd = nn.Parameter(torch.randn(dim, rank) * 0.01)

        # Backward projection: dst -> src (if bidirectional)
        if bidirectional:
            self.U_bwd = nn.Parameter(torch.randn(dim, rank) * 0.01)
            self.V_bwd = nn.Parameter(torch.randn(dim, rank) * 0.01)

        # Learnable bridge strength
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional bridge contributions.

        Args:
            src: [batch, seq, dim] source world tensor
            dst: [batch, seq, dim] destination world tensor

        Returns:
            delta_src: Contribution to add to src from dst
            delta_dst: Contribution to add to dst from src
            energy: Scalar bridge energy (L2 norm of transfer)
        """
        # Forward: src influences dst
        # dst += strength * (src @ U_fwd) @ V_fwd.T
        latent_fwd = torch.matmul(src, self.U_fwd)  # [batch, seq, rank]
        delta_dst = torch.matmul(latent_fwd, self.V_fwd.t())  # [batch, seq, dim]
        delta_dst = self.strength * delta_dst

        # Backward: dst influences src (if bidirectional)
        if self.bidirectional:
            latent_bwd = torch.matmul(dst, self.U_bwd)
            delta_src = torch.matmul(latent_bwd, self.V_bwd.t())
            delta_src = self.strength * delta_src
        else:
            delta_src = torch.zeros_like(src)

        # Compute energy (L2 norm of total transfer)
        energy = (delta_src.pow(2).sum() + delta_dst.pow(2).sum()).sqrt()

        return delta_src, delta_dst, energy


class WorldBridge(nn.Module):
    """
    Explicit, audited cross-world mixing with limited capacity.

    Features:
    - Low-rank projection between worlds (limited bandwidth)
    - All cross-world flow is logged in trace
    - Bridge usage is bounded and inspectable

    Creates 6 pairwise bridges for 4 worlds:
    A<->B, A<->C, A<->D, B<->C, B<->D, C<->D
    """

    def __init__(
        self,
        world_dim: int = WORLD_DIM,
        num_worlds: int = NUM_WORLDS,
        bridge_rank: int = 2,
        max_bridge_energy: float = 0.2,
        bidirectional: bool = True,
        use_distance_decay: bool = True,
    ):
        """
        Initialize WorldBridge.

        Args:
            world_dim: Dimension of each world (10 for TKS)
            num_worlds: Number of worlds (4 for TKS)
            bridge_rank: Rank of low-rank projection (low = limited capacity)
            max_bridge_energy: Maximum allowed bridge contribution (energy cap)
            bidirectional: Whether bridges work both directions
            use_distance_decay: Apply decay based on world distance
        """
        super().__init__()
        self.world_dim = world_dim
        self.num_worlds = num_worlds
        self.bridge_rank = bridge_rank
        self.max_bridge_energy = max_bridge_energy
        self.use_distance_decay = use_distance_decay

        # Create pairwise bridges
        self.bridges = nn.ModuleDict()
        for w1, w2 in WORLD_PAIRS:
            key = f"{w1}_{w2}"
            self.bridges[key] = LowRankBridge(
                dim=world_dim,
                rank=bridge_rank,
                bidirectional=bidirectional,
            )

        # Distance-based decay factors (adjacent worlds interact more)
        if use_distance_decay:
            decay_values = []
            for w1, w2 in WORLD_PAIRS:
                dist = world_distance(w1, w2)
                # Decay: 1.0 for adjacent (dist=1), 0.5 for dist=2, 0.33 for dist=3
                decay = 1.0 / max(dist, 1)
                decay_values.append(decay)
            self.register_buffer(
                "distance_decay",
                torch.tensor(decay_values, dtype=torch.float32)
            )

        # Global bridge gate (learnable)
        self.global_gate = nn.Parameter(torch.tensor(1.0))

        # LayerNorm for stability
        self.norm = nn.LayerNorm(world_dim * num_worlds)

    def _split_worlds(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split concatenated tensor into per-world tensors."""
        return {
            "A": x[..., 0:10],
            "B": x[..., 10:20],
            "C": x[..., 20:30],
            "D": x[..., 30:40],
        }

    def _merge_worlds(self, worlds: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Merge per-world tensors back to concatenated form."""
        return torch.cat([worlds["A"], worlds["B"], worlds["C"], worlds["D"]], dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Apply cross-world bridging with full audit trail.

        Args:
            x: [batch, seq, 40] input (4 worlds concatenated)

        Returns:
            output: [batch, seq, 40] with cross-world mixing
            bridge_trace: dict with all bridge decisions including:
                - bridge_weights: [6] energy per pair
                - bridge_activations: which pairs were used
                - total_bridge_energy: scalar
                - energy_capped: whether cap was applied
                - max_pair: which pair had max flow
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        batch, seq, dim = x.shape

        # Split into worlds
        worlds = self._split_worlds(x)

        # Accumulate deltas for each world
        deltas = {w: torch.zeros_like(worlds[w]) for w in WORLD_LETTERS}

        # Track bridge energies
        bridge_energies = []
        bridge_activations = []

        # Apply each pairwise bridge
        for idx, (w1, w2) in enumerate(WORLD_PAIRS):
            key = f"{w1}_{w2}"
            bridge = self.bridges[key]

            delta_w1, delta_w2, energy = bridge(worlds[w1], worlds[w2])

            # Apply distance decay
            if self.use_distance_decay:
                decay = self.distance_decay[idx]
                delta_w1 = delta_w1 * decay
                delta_w2 = delta_w2 * decay
                energy = energy * decay

            deltas[w1] = deltas[w1] + delta_w1
            deltas[w2] = deltas[w2] + delta_w2

            bridge_energies.append(energy)
            bridge_activations.append(energy.item() > 1e-6)

        # Stack energies
        energies = torch.stack(bridge_energies)  # [6]
        total_energy = energies.sum()

        # Apply energy cap
        energy_capped = False
        scale_factor = 1.0
        if total_energy > self.max_bridge_energy:
            scale_factor = self.max_bridge_energy / (total_energy + 1e-8)
            energy_capped = True

        # Scale deltas by global gate and energy cap
        final_scale = self.global_gate * scale_factor
        for w in WORLD_LETTERS:
            deltas[w] = deltas[w] * final_scale

        # Apply deltas (residual)
        output_worlds = {w: worlds[w] + deltas[w] for w in WORLD_LETTERS}

        # Merge back
        output = self._merge_worlds(output_worlds)
        output = self.norm(output)

        if squeeze:
            output = output.squeeze(1)

        # Find max pair
        max_idx = energies.argmax().item()
        max_pair = WORLD_PAIRS[max_idx]

        # Build trace
        bridge_trace = {
            "bridge_weights": energies.detach(),
            "bridge_activations": bridge_activations,
            "total_bridge_energy": total_energy.item() * scale_factor,
            "energy_capped": energy_capped,
            "max_pair": max_pair,
            "scale_factor": scale_factor,
            "global_gate": self.global_gate.item(),
        }

        return output, bridge_trace


# =============================================================================
# 2. FOUNDATION-GUIDED BRIDGE
# =============================================================================

class FoundationBridge(nn.Module):
    """
    Cross-world mixing guided by TKS foundations.

    Uses the 7 foundations as "sanctioned" cross-world pathways.
    Each foundation has a canonical world-mixing pattern derived from TKS canon.

    The foundation scores (from RPM) determine which mixing patterns to apply.
    This ensures cross-world mixing is semantically meaningful, not arbitrary.
    """

    # Foundation -> World mixing patterns (from TKS canon)
    # Each foundation has a primary world affinity and secondary connections
    FOUNDATION_WORLD_PATTERNS = {
        0: {"primary": "A", "secondary": ["B"]},          # F1: Association (spiritual/mental)
        1: {"primary": "B", "secondary": ["A", "C"]},     # F2: Wisdom (mental, connects above/below)
        2: {"primary": "C", "secondary": ["B", "D"]},     # F3: Life (emotional, bridges mental/physical)
        3: {"primary": "D", "secondary": ["C"]},          # F4: Companionship (physical/emotional)
        4: {"primary": "A", "secondary": ["C"]},          # F5: Power (spiritual->emotional, skips mental)
        5: {"primary": "B", "secondary": ["D"]},          # F6: Wealth (mental->physical, skips emotional)
        6: {"primary": "all", "secondary": []},           # F7: Continuation (integration across all)
    }

    def __init__(
        self,
        world_dim: int = WORLD_DIM,
        num_foundations: int = 7,
        scale: float = 0.1,
        use_attention: bool = True,
    ):
        """
        Initialize FoundationBridge.

        Args:
            world_dim: Dimension per world (10)
            num_foundations: Number of foundations (7)
            scale: Initial scale for mixing
            use_attention: Use attention-based mixing instead of linear
        """
        super().__init__()
        self.world_dim = world_dim
        self.num_foundations = num_foundations
        self.use_attention = use_attention

        # Per-foundation mixing matrices
        # Each foundation has a 4x4 world-mixing matrix
        self.mixing_weights = nn.ParameterList([
            nn.Parameter(self._init_mixing_matrix(f_idx))
            for f_idx in range(num_foundations)
        ])

        # Per-foundation projection (per world)
        self.foundation_projs = nn.ModuleList([
            nn.Linear(world_dim, world_dim, bias=False)
            for _ in range(num_foundations)
        ])

        # Scale gate
        self.scale = nn.Parameter(torch.tensor(scale))

        # Attention-based mixing (optional)
        if use_attention:
            self.query_proj = nn.Linear(world_dim * NUM_WORLDS, world_dim)
            self.key_proj = nn.Linear(world_dim * NUM_WORLDS, world_dim)

        # LayerNorm for output
        self.norm = nn.LayerNorm(world_dim * NUM_WORLDS)

    def _init_mixing_matrix(self, f_idx: int) -> torch.Tensor:
        """
        Initialize world mixing matrix for a foundation based on canon patterns.

        Returns:
            [4, 4] mixing matrix where M[i,j] = strength of flow from world j to world i
        """
        pattern = self.FOUNDATION_WORLD_PATTERNS[f_idx]
        matrix = torch.eye(4) * 0.8  # Diagonal = self-connection

        if pattern["primary"] == "all":
            # F7: Full integration - all worlds mix equally
            matrix = torch.ones(4, 4) * 0.25
        else:
            # Map world letters to indices
            world_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
            primary_idx = world_to_idx[pattern["primary"]]

            # Primary world sends to secondaries
            for secondary in pattern["secondary"]:
                sec_idx = world_to_idx[secondary]
                matrix[sec_idx, primary_idx] = 0.3  # Primary -> Secondary
                matrix[primary_idx, sec_idx] = 0.2  # Secondary -> Primary (weaker)

        # Normalize rows
        matrix = matrix / matrix.sum(dim=1, keepdim=True)
        return matrix

    def _split_worlds(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split into list of world tensors."""
        return [
            x[..., i*self.world_dim:(i+1)*self.world_dim]
            for i in range(NUM_WORLDS)
        ]

    def forward(
        self,
        x: torch.Tensor,
        foundation_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply foundation-guided cross-world mixing.

        Args:
            x: [batch, seq, 40] input tensor
            foundation_scores: [batch, seq, 7] foundation activation scores (from RPM)

        Returns:
            output: [batch, seq, 40] mixed output
            trace: dict with foundation bridge decisions
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            foundation_scores = foundation_scores.unsqueeze(1)
            squeeze = True

        batch, seq, dim = x.shape

        # Split into worlds
        world_tensors = self._split_worlds(x)  # List of [batch, seq, 10]

        # Stack for matrix operations: [batch, seq, 4, 10]
        worlds_stacked = torch.stack(world_tensors, dim=-2)

        # Compute weighted mixing matrix from foundation scores
        # Each foundation contributes its mixing pattern weighted by score
        mixed_matrix = torch.zeros(batch, seq, 4, 4, device=x.device)

        for f_idx in range(self.num_foundations):
            f_score = foundation_scores[..., f_idx:f_idx+1, None]  # [batch, seq, 1, 1]
            f_matrix = self.mixing_weights[f_idx]  # [4, 4]
            mixed_matrix = mixed_matrix + f_score * f_matrix

        # Normalize mixed matrix
        mixed_matrix = mixed_matrix / (mixed_matrix.sum(dim=-1, keepdim=True) + 1e-8)

        # Apply mixing: output_world[i] = sum_j(M[i,j] * input_world[j])
        # [batch, seq, 4, 4] @ [batch, seq, 4, 10] -> [batch, seq, 4, 10]
        mixed_worlds = torch.matmul(mixed_matrix, worlds_stacked)

        # Apply per-foundation projections (weighted by scores)
        projected = torch.zeros_like(mixed_worlds)
        for f_idx in range(self.num_foundations):
            f_score = foundation_scores[..., f_idx:f_idx+1, None]  # [batch, seq, 1, 1]
            # Project each world
            proj_worlds = []
            for w_idx in range(NUM_WORLDS):
                w_proj = self.foundation_projs[f_idx](mixed_worlds[..., w_idx, :])
                proj_worlds.append(w_proj)
            proj_stacked = torch.stack(proj_worlds, dim=-2)  # [batch, seq, 4, 10]
            projected = projected + f_score * proj_stacked

        # Flatten back to [batch, seq, 40]
        output = projected.reshape(batch, seq, dim)

        # Residual with scale
        output = x + self.scale * (output - x)
        output = self.norm(output)

        if squeeze:
            output = output.squeeze(1)

        # Build trace
        dominant_foundation = foundation_scores.argmax(dim=-1)  # [batch, seq]

        trace = {
            "foundation_scores": foundation_scores.detach(),
            "mixed_matrix": mixed_matrix.detach(),
            "dominant_foundation": dominant_foundation.detach(),
            "scale": self.scale.item(),
        }

        return output, trace


# =============================================================================
# 3. BRIDGE TRACE RECORD (First-Class Trace Item)
# =============================================================================

@dataclass
class BridgeTraceRecord:
    """
    Trace record for cross-world bridge decisions.

    This is a first-class trace item that captures all auditable
    information about cross-world information flow.
    """

    # World-pair energies: [6] for A-B, A-C, A-D, B-C, B-D, C-D
    world_pair_energies: torch.Tensor

    # Total energy after capping
    total_bridge_energy: float

    # Whether foundation-guided mixing was used
    foundation_guided: bool

    # Which foundations were active (if foundation-guided)
    active_foundations: List[int] = field(default_factory=list)

    # Which world pair had maximum flow
    max_pair: Tuple[str, str] = ("A", "B")

    # Whether energy cap was applied
    energy_capped: bool = False

    # Scale factor applied (1.0 if no capping)
    scale_factor: float = 1.0

    # Additional metadata
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "world_pair_energies": self.world_pair_energies.tolist()
                if isinstance(self.world_pair_energies, torch.Tensor)
                else self.world_pair_energies,
            "total_bridge_energy": self.total_bridge_energy,
            "foundation_guided": self.foundation_guided,
            "active_foundations": self.active_foundations,
            "max_pair": list(self.max_pair),
            "energy_capped": self.energy_capped,
            "scale_factor": self.scale_factor,
            "metadata": self.metadata,
        }

    @classmethod
    def from_world_bridge(cls, bridge_trace: Dict) -> "BridgeTraceRecord":
        """Create from WorldBridge output."""
        return cls(
            world_pair_energies=bridge_trace["bridge_weights"],
            total_bridge_energy=bridge_trace["total_bridge_energy"],
            foundation_guided=False,
            active_foundations=[],
            max_pair=bridge_trace["max_pair"],
            energy_capped=bridge_trace["energy_capped"],
            scale_factor=bridge_trace["scale_factor"],
        )

    @classmethod
    def from_foundation_bridge(
        cls,
        foundation_trace: Dict,
        base_energies: Optional[torch.Tensor] = None,
    ) -> "BridgeTraceRecord":
        """Create from FoundationBridge output."""
        foundation_scores = foundation_trace["foundation_scores"]

        # Determine active foundations (score > threshold)
        if foundation_scores.dim() > 2:
            scores = foundation_scores.mean(dim=(0, 1))  # [7]
        else:
            scores = foundation_scores.mean(dim=0)  # [7]

        threshold = 0.15
        active = (scores > threshold).nonzero(as_tuple=True)[0].tolist()

        # Estimate world pair energies from mixed matrix
        mixed_matrix = foundation_trace["mixed_matrix"]
        if mixed_matrix.dim() > 2:
            matrix = mixed_matrix.mean(dim=(0, 1))  # [4, 4]
        else:
            matrix = mixed_matrix

        # Extract off-diagonal pairs
        pair_energies = []
        for w1, w2 in WORLD_PAIRS:
            i, j = WORLD_ORDINALS[w1], WORLD_ORDINALS[w2]
            energy = (matrix[i, j] + matrix[j, i]).item()
            pair_energies.append(energy)

        pair_energies_tensor = torch.tensor(pair_energies)

        return cls(
            world_pair_energies=pair_energies_tensor,
            total_bridge_energy=pair_energies_tensor.sum().item(),
            foundation_guided=True,
            active_foundations=active,
            max_pair=WORLD_PAIRS[pair_energies_tensor.argmax().item()],
            energy_capped=False,
            scale_factor=foundation_trace["scale"],
        )


def bridge_to_trace(bridge_output: Dict, is_foundation: bool = False) -> BridgeTraceRecord:
    """
    Convert bridge output to trace record.

    Args:
        bridge_output: Output dict from WorldBridge or FoundationBridge
        is_foundation: Whether this is from FoundationBridge

    Returns:
        BridgeTraceRecord for auditing
    """
    if is_foundation:
        return BridgeTraceRecord.from_foundation_bridge(bridge_output)
    else:
        return BridgeTraceRecord.from_world_bridge(bridge_output)


# =============================================================================
# 4. BRIDGE CONSTRAINT LOSS
# =============================================================================

class BridgeConstraintLoss(nn.Module):
    """
    Regularization to keep bridge usage bounded and meaningful.

    Components:
    1. Energy cap loss: Penalize exceeding max total energy
    2. Diversity loss: Encourage using diverse world pairs (not always same pair)
    3. Sparsity loss: Optional penalty for too many active bridges
    """

    def __init__(
        self,
        max_total_energy: float = 0.3,
        diversity_weight: float = 0.1,
        sparsity_weight: float = 0.05,
        target_sparsity: float = 0.5,
    ):
        """
        Initialize BridgeConstraintLoss.

        Args:
            max_total_energy: Target maximum for total bridge energy
            diversity_weight: Weight for diversity regularization
            sparsity_weight: Weight for sparsity regularization
            target_sparsity: Target fraction of active bridges (0-1)
        """
        super().__init__()
        self.max_total_energy = max_total_energy
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
        self.target_sparsity = target_sparsity

    def forward(self, bridge_trace: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute bridge constraint loss.

        Args:
            bridge_trace: Output trace from WorldBridge

        Returns:
            loss: Scalar constraint loss
            breakdown: Dict with loss components
        """
        energies = bridge_trace["bridge_weights"]  # [6]

        # 1. Energy cap loss: soft penalty for exceeding max
        total_energy = energies.sum()
        energy_loss = F.relu(total_energy - self.max_total_energy) ** 2

        # 2. Diversity loss: encourage uniform distribution over active bridges
        # High entropy = high diversity
        # Use normalized energies as "probabilities"
        probs = energies / (energies.sum() + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = math.log(len(WORLD_PAIRS))  # log(6)
        diversity_loss = (max_entropy - entropy) / max_entropy  # Normalized [0, 1]

        # 3. Sparsity loss: penalize deviation from target active fraction
        active_mask = (energies > 1e-6).float()
        active_fraction = active_mask.mean()
        sparsity_loss = (active_fraction - self.target_sparsity) ** 2

        # Combine losses
        total_loss = (
            energy_loss +
            self.diversity_weight * diversity_loss +
            self.sparsity_weight * sparsity_loss
        )

        breakdown = {
            "energy_loss": energy_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "total_energy": total_energy.item(),
            "entropy": entropy.item(),
            "active_fraction": active_fraction.item(),
        }

        return total_loss, breakdown


# =============================================================================
# 5. COMBINED WORLD BRIDGE LAYER
# =============================================================================

class TKSWorldBridgeLayer(nn.Module):
    """
    Combined world bridge layer with both modes.

    Can operate in:
    1. Explicit mode: WorldBridge with pairwise low-rank projections
    2. Foundation mode: FoundationBridge guided by foundation scores
    3. Hybrid mode: Both, with learned gating
    """

    def __init__(
        self,
        world_dim: int = WORLD_DIM,
        bridge_rank: int = 2,
        max_bridge_energy: float = 0.2,
        use_foundation_bridge: bool = True,
        mode: str = "hybrid",  # "explicit", "foundation", "hybrid"
    ):
        super().__init__()
        self.mode = mode

        # Explicit bridge
        if mode in ("explicit", "hybrid"):
            self.world_bridge = WorldBridge(
                world_dim=world_dim,
                bridge_rank=bridge_rank,
                max_bridge_energy=max_bridge_energy,
            )
        else:
            self.world_bridge = None

        # Foundation bridge
        if mode in ("foundation", "hybrid") and use_foundation_bridge:
            self.foundation_bridge = FoundationBridge(
                world_dim=world_dim,
            )
        else:
            self.foundation_bridge = None

        # Hybrid gating
        if mode == "hybrid" and self.world_bridge and self.foundation_bridge:
            self.hybrid_gate = nn.Parameter(torch.tensor(0.5))
        else:
            self.hybrid_gate = None

        # Constraint loss
        self.constraint_loss = BridgeConstraintLoss()

    def forward(
        self,
        x: torch.Tensor,
        foundation_scores: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply world bridging.

        Args:
            x: [batch, seq, 40] input
            foundation_scores: [batch, seq, 7] (required for foundation/hybrid mode)
            return_loss: Whether to compute and return constraint loss

        Returns:
            output: [batch, seq, 40] bridged output
            trace: Combined trace dict
        """
        trace = {"mode": self.mode}

        if self.mode == "explicit":
            output, world_trace = self.world_bridge(x)
            trace["world_bridge"] = world_trace
            trace["bridge_record"] = bridge_to_trace(world_trace).to_dict()

        elif self.mode == "foundation":
            if foundation_scores is None:
                raise ValueError("foundation_scores required for foundation mode")
            output, found_trace = self.foundation_bridge(x, foundation_scores)
            trace["foundation_bridge"] = found_trace
            trace["bridge_record"] = bridge_to_trace(found_trace, is_foundation=True).to_dict()

        elif self.mode == "hybrid":
            # Run both bridges
            world_out, world_trace = self.world_bridge(x)

            if foundation_scores is not None and self.foundation_bridge is not None:
                found_out, found_trace = self.foundation_bridge(x, foundation_scores)

                # Blend with learned gate
                g = torch.sigmoid(self.hybrid_gate)
                output = g * world_out + (1 - g) * found_out

                trace["world_bridge"] = world_trace
                trace["foundation_bridge"] = found_trace
                trace["hybrid_gate"] = g.item()
            else:
                output = world_out
                trace["world_bridge"] = world_trace
                trace["hybrid_gate"] = 1.0
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Compute constraint loss if requested
        if return_loss and "world_bridge" in trace:
            loss, loss_breakdown = self.constraint_loss(trace["world_bridge"])
            trace["constraint_loss"] = loss
            trace["loss_breakdown"] = loss_breakdown

        return output, trace


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core bridges
    "WorldBridge",
    "FoundationBridge",
    "LowRankBridge",
    # Trace
    "BridgeTraceRecord",
    "bridge_to_trace",
    # Loss
    "BridgeConstraintLoss",
    # Combined layer
    "TKSWorldBridgeLayer",
    # Constants
    "WORLD_PAIRS",
    "PAIR_TO_INDEX",
    "INDEX_TO_PAIR",
]


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TKS CROSS-WORLD BRIDGE - Audited Cross-World Mixing")
    print("=" * 70)

    # Test parameters
    batch, seq = 4, 16
    device = torch.device("cpu")

    # Create test input
    x = torch.randn(batch, seq, TOTAL_DIM, device=device)
    foundation_scores = F.softmax(torch.randn(batch, seq, 7, device=device), dim=-1)

    print("\n[1] Testing WorldBridge (explicit pairwise bridges)")
    print("-" * 50)

    world_bridge = WorldBridge(bridge_rank=2, max_bridge_energy=0.2)
    out, trace = world_bridge(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Bridge energies: {trace['bridge_weights'].tolist()}")
    print(f"Total energy: {trace['total_bridge_energy']:.4f}")
    print(f"Max pair: {trace['max_pair']}")
    print(f"Energy capped: {trace['energy_capped']}")

    # Convert to trace record
    record = bridge_to_trace(trace)
    print(f"\nTrace Record: {record.to_dict()}")

    print("\n[2] Testing FoundationBridge (foundation-guided)")
    print("-" * 50)

    found_bridge = FoundationBridge()
    out_f, trace_f = found_bridge(x, foundation_scores)

    print(f"Output shape: {out_f.shape}")
    print(f"Dominant foundation: {trace_f['dominant_foundation'][0, 0].item()}")
    print(f"Scale: {trace_f['scale']:.4f}")

    # Convert to trace record
    record_f = bridge_to_trace(trace_f, is_foundation=True)
    print(f"\nFoundation Trace Record: {record_f.to_dict()}")

    print("\n[3] Testing BridgeConstraintLoss")
    print("-" * 50)

    constraint = BridgeConstraintLoss(
        max_total_energy=0.3,
        diversity_weight=0.1,
        sparsity_weight=0.05,
    )
    loss, breakdown = constraint(trace)

    print(f"Constraint loss: {loss.item():.4f}")
    print(f"Breakdown: {breakdown}")

    print("\n[4] Testing TKSWorldBridgeLayer (hybrid mode)")
    print("-" * 50)

    layer = TKSWorldBridgeLayer(mode="hybrid")
    out_h, trace_h = layer(x, foundation_scores, return_loss=True)

    print(f"Output shape: {out_h.shape}")
    print(f"Mode: {trace_h['mode']}")
    print(f"Hybrid gate: {trace_h.get('hybrid_gate', 'N/A')}")
    if "constraint_loss" in trace_h:
        print(f"Constraint loss: {trace_h['constraint_loss'].item():.4f}")

    print("\n[5] Verifying numerical stability")
    print("-" * 50)

    # Check for NaN/Inf
    assert not torch.isnan(out).any(), "NaN in WorldBridge output"
    assert not torch.isnan(out_f).any(), "NaN in FoundationBridge output"
    assert not torch.isnan(out_h).any(), "NaN in Hybrid output"
    assert not torch.isinf(out).any(), "Inf in WorldBridge output"
    assert not torch.isinf(out_f).any(), "Inf in FoundationBridge output"
    assert not torch.isinf(out_h).any(), "Inf in Hybrid output"

    print("No NaN/Inf detected - all bridges stable!")

    print("\n[6] Parameter counts")
    print("-" * 50)

    wb_params = sum(p.numel() for p in world_bridge.parameters())
    fb_params = sum(p.numel() for p in found_bridge.parameters())
    layer_params = sum(p.numel() for p in layer.parameters())

    print(f"WorldBridge: {wb_params:,} params")
    print(f"FoundationBridge: {fb_params:,} params")
    print(f"TKSWorldBridgeLayer (hybrid): {layer_params:,} params")

    print("\n[7] Testing audit trail completeness")
    print("-" * 50)

    # Verify all required trace keys are present
    required_world_keys = ["bridge_weights", "bridge_activations", "total_bridge_energy",
                          "energy_capped", "max_pair", "scale_factor"]
    for key in required_world_keys:
        assert key in trace, f"Missing key in WorldBridge trace: {key}"
    print(f"WorldBridge trace keys: {list(trace.keys())}")

    required_found_keys = ["foundation_scores", "mixed_matrix", "dominant_foundation", "scale"]
    for key in required_found_keys:
        assert key in trace_f, f"Missing key in FoundationBridge trace: {key}"
    print(f"FoundationBridge trace keys: {list(trace_f.keys())}")

    print("\n" + "=" * 70)
    print("All tests passed! Cross-world bridge module is operational.")
    print("=" * 70)

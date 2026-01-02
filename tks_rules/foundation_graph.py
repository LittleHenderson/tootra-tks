"""
TKS Foundation Graph - Canonical 7x7 Adjacency for Foundation Diffusion

Builds adjacency matrices for the 7 Foundations based on:
1. Self-loops (each foundation connects to itself)
2. Canon opposite edges from FOUNDATION_OPP in inversion/engine.py
3. Optional soft hub: Foundation 4 (Companionship) connects to all

Canon Opposites (from inversion/engine.py):
    1 ↔ 7  (Unity ↔ Continuation)
    2 ↔ 6  (Wisdom ↔ Wealth)
    3 ↔ 5  (Life ↔ Power)
    4 → 4  (Companionship is self-opposite, central)

The resulting adjacency is row-normalized for stable diffusion dynamics.
"""

from typing import Optional
import torch

# Canon opposites - imported concept from inversion/engine.py
# We define it here to avoid circular imports
FOUNDATION_OPP = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}


def build_foundation_adj(
    hub_weight: float = 0.15,
    self_loop_weight: float = 1.0,
    opposite_weight: float = 0.5,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build 7x7 foundation adjacency matrix.

    Structure:
    - Self-loops on diagonal (weight = self_loop_weight)
    - Opposite edges from FOUNDATION_OPP (weight = opposite_weight)
    - Soft hub: Foundation 4 connects to all others (weight = hub_weight)

    Args:
        hub_weight: Weight for Foundation 4's connections to other foundations.
                   Set to 0.0 to disable soft hub (pure canon).
        self_loop_weight: Weight for self-connections.
        opposite_weight: Weight for canon opposite connections.
        normalize: If True, row-normalize for stable diffusion.
        dtype: Tensor dtype.

    Returns:
        7x7 adjacency tensor (0-indexed, so Foundation 1 = index 0)
    """
    adj = torch.zeros(7, 7, dtype=dtype)

    # 1. Self-loops (diagonal)
    for i in range(7):
        adj[i, i] = self_loop_weight

    # 2. Canon opposite edges (symmetric)
    # FOUNDATION_OPP uses 1-based indexing, convert to 0-based
    for f1, f2 in FOUNDATION_OPP.items():
        i, j = f1 - 1, f2 - 1
        if i != j:  # Skip self-loop (4→4 already handled)
            adj[i, j] = opposite_weight
            adj[j, i] = opposite_weight

    # 3. Soft hub: Foundation 4 (index 3) connects to all
    if hub_weight > 0:
        hub_idx = 3  # Foundation 4 in 0-based
        for i in range(7):
            if i != hub_idx:
                adj[hub_idx, i] = max(adj[hub_idx, i], hub_weight)
                adj[i, hub_idx] = max(adj[i, hub_idx], hub_weight)

    # 4. Row-normalize for stable diffusion
    if normalize:
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums.clamp(min=1e-8)

    return adj


def build_canon_only_adj(
    self_loop_weight: float = 1.0,
    opposite_weight: float = 0.5,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build pure-canon adjacency (no soft hub).

    Use this for strict TKS canon compliance.
    """
    return build_foundation_adj(
        hub_weight=0.0,
        self_loop_weight=self_loop_weight,
        opposite_weight=opposite_weight,
        normalize=normalize,
        dtype=dtype,
    )


# =============================================================================
# PRE-BUILT ADJACENCY MATRICES
# =============================================================================

# Default hybrid adjacency (canon + soft hub)
FOUNDATION_ADJ: torch.Tensor = build_foundation_adj(hub_weight=0.15)

# Pure canon adjacency (no soft hub)
FOUNDATION_ADJ_CANON: torch.Tensor = build_canon_only_adj()


# =============================================================================
# FOUNDATION GRAPH UTILITIES
# =============================================================================

def get_opposite_foundation(f: int) -> int:
    """
    Get the opposite foundation (1-based indexing).

    Args:
        f: Foundation number 1-7

    Returns:
        Opposite foundation number
    """
    return FOUNDATION_OPP.get(f, f)


def get_foundation_neighbors(f: int, include_self: bool = True) -> list:
    """
    Get neighbors of a foundation in the canon graph.

    Args:
        f: Foundation number 1-7
        include_self: Include self in neighbors

    Returns:
        List of neighbor foundation numbers
    """
    neighbors = []
    if include_self:
        neighbors.append(f)

    # Add opposite
    opp = get_opposite_foundation(f)
    if opp != f:
        neighbors.append(opp)

    # Foundation 4 connects to all in soft-hub mode
    if f == 4:
        neighbors = list(range(1, 8))
    elif 4 not in neighbors:
        neighbors.append(4)  # All connect to 4

    return sorted(set(neighbors))


def visualize_foundation_graph() -> str:
    """
    ASCII visualization of the foundation graph.

    Returns:
        String representation of the graph
    """
    lines = [
        "TKS Foundation Graph (Canon + Soft Hub)",
        "=" * 40,
        "",
        "      1 (Unity) ←――――――――→ 7 (Continuation)",
        "           \\              /",
        "            \\     4     /",
        "             \\  (Hub)  /",
        "              \\  |    /",
        "               \\ |   /",
        "      2 (Wisdom) ←-→ 6 (Wealth)",
        "                 |",
        "                 |",
        "      3 (Life) ←――――――→ 5 (Power)",
        "",
        "Edges:",
        "  ═══  Canon opposite (FOUNDATION_OPP)",
        "  ---  Soft hub (Foundation 4 central)",
        "",
        "Canon Opposites:",
    ]

    for f1, f2 in sorted(FOUNDATION_OPP.items()):
        if f1 <= f2:
            lines.append(f"  {f1} ↔ {f2}")

    return "\n".join(lines)


__all__ = [
    "FOUNDATION_OPP",
    "FOUNDATION_ADJ",
    "FOUNDATION_ADJ_CANON",
    "build_foundation_adj",
    "build_canon_only_adj",
    "get_opposite_foundation",
    "get_foundation_neighbors",
    "visualize_foundation_graph",
]


if __name__ == "__main__":
    print(visualize_foundation_graph())
    print()
    print("Hybrid Adjacency (normalized):")
    print(FOUNDATION_ADJ)
    print()
    print("Canon-Only Adjacency (normalized):")
    print(FOUNDATION_ADJ_CANON)

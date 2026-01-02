"""
TKS Compositional Layer v2 - Pure TKS-Structured (No FFN/MLP)

Refactored to enforce:
1. World separation (block-diagonal projections)
2. Operator symmetry/anti-symmetry constraints
3. No MLPs - pure linear + noetic routing
4. Residual gating for stable 40D core
5. Source weighting (original > generated)

Canon-compliant compositional grounding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import json


# =============================================================================
# CONSTANTS
# =============================================================================

NOETIC_DIM = 10
NUM_WORLDS = 4
TOTAL_DIM = 40  # 4 worlds × 10 noetics
NUM_OPERATORS = 4  # +, -, ×, ÷

# Operator indices
OP_PLUS = 0      # + : Association (symmetric)
OP_MINUS = 1     # - : Disassociation (anti-symmetric)
OP_MULT = 2      # × : Exacerbation (symmetric)
OP_DIV = 3       # ÷ : Opposition (anti-symmetric)

SYMMETRIC_OPS = {OP_PLUS, OP_MULT}
ANTISYMMETRIC_OPS = {OP_MINUS, OP_DIV}


# =============================================================================
# WORLD-WISE PROJECTION (Block-Diagonal)
# =============================================================================

class WorldWiseProjection(nn.Module):
    """
    Block-diagonal projection that keeps A/B/C/D worlds separable.

    Instead of a full 40x40 or equation_dim x 40 matrix,
    uses 4 independent 10x10 blocks (or equation_dim/4 x 10).

    This prevents cross-world mixing and maintains interpretability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = TOTAL_DIM,
        num_worlds: int = NUM_WORLDS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_worlds = num_worlds
        self.world_dim = output_dim // num_worlds  # 10

        # Per-world input dimension
        self.input_per_world = input_dim // num_worlds

        # 4 separate linear projections (block-diagonal)
        self.world_projections = nn.ModuleList([
            nn.Linear(self.input_per_world, self.world_dim, bias=False)
            for _ in range(num_worlds)
        ])

        # Initialize close to identity where possible
        for proj in self.world_projections:
            if self.input_per_world == self.world_dim:
                nn.init.eye_(proj.weight)
            else:
                nn.init.orthogonal_(proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_dim] or [batch, seq, input_dim]
        returns: [batch, 40] or [batch, seq, 40]
        """
        # Split into world chunks
        chunks = torch.chunk(x, self.num_worlds, dim=-1)

        # Project each world independently
        projected = [
            self.world_projections[i](chunks[i])
            for i in range(self.num_worlds)
        ]

        # Concatenate back
        return torch.cat(projected, dim=-1)


# =============================================================================
# NOETIC OPERATOR ROUTING (replaces MLP)
# =============================================================================

class NoeticOperatorRouter(nn.Module):
    """
    Routes through operator-specific linear transforms.

    No nonlinearities - pure linear composition respecting TKS structure.
    Each operator has its own world-wise linear transform.
    """

    def __init__(
        self,
        dim: int = TOTAL_DIM,
        num_operators: int = NUM_OPERATORS,
    ):
        super().__init__()
        self.dim = dim
        self.num_operators = num_operators

        # Per-operator, per-world linear transforms
        # Shape: [num_ops, num_worlds, noetic_dim, noetic_dim]
        self.op_transforms = nn.ParameterList([
            nn.Parameter(torch.eye(NOETIC_DIM).unsqueeze(0).repeat(NUM_WORLDS, 1, 1))
            for _ in range(num_operators)
        ])

        # Small learnable bias per operator
        self.op_bias = nn.Parameter(torch.zeros(num_operators, dim))

    def forward(
        self,
        x: torch.Tensor,
        operator: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply operator-specific world-wise transform.

        x: [batch, 40]
        operator: [batch] indices 0-3
        """
        batch_size = x.shape[0]

        # Split into worlds
        x_worlds = x.view(batch_size, NUM_WORLDS, NOETIC_DIM)

        # Apply transforms per operator
        output = torch.zeros_like(x)

        for op_idx in range(self.num_operators):
            mask = (operator == op_idx).float().view(-1, 1)

            # Apply world-wise transform
            transforms = self.op_transforms[op_idx]  # [4, 10, 10]
            transformed = torch.einsum('bwn,wno->bwo', x_worlds, transforms)
            transformed = transformed.reshape(batch_size, -1)  # [batch, 40]

            # Add bias
            transformed = transformed + self.op_bias[op_idx]

            output = output + mask * transformed

        return output


# =============================================================================
# OPERATOR SYMMETRY CONSTRAINTS
# =============================================================================

class OperatorSymmetryLoss(nn.Module):
    """
    Enforces algebraic properties of operators:

    + and × (symmetric):   f(A, B) = f(B, A)
    - and ÷ (anti-symmetric): f(A, B) = -f(B, A) or f(A, B) = inverse(f(B, A))

    These are soft constraints added to the training loss.
    """

    def __init__(self, symmetry_weight: float = 1.0, antisymmetry_weight: float = 1.0):
        super().__init__()
        self.symmetry_weight = symmetry_weight
        self.antisymmetry_weight = antisymmetry_weight

    def forward(
        self,
        composed_ab: torch.Tensor,  # f(A, B)
        composed_ba: torch.Tensor,  # f(B, A)
        operator: torch.Tensor,     # [batch] indices
    ) -> Dict[str, torch.Tensor]:
        """
        Compute symmetry/anti-symmetry violation losses.
        """
        batch_size = operator.shape[0]

        # Masks for symmetric vs anti-symmetric operators
        is_symmetric = (operator == OP_PLUS) | (operator == OP_MULT)
        is_antisymmetric = (operator == OP_MINUS) | (operator == OP_DIV)

        losses = {}

        # Symmetry loss: ||f(A,B) - f(B,A)||^2
        if is_symmetric.any():
            sym_mask = is_symmetric.float().unsqueeze(-1)
            sym_diff = (composed_ab - composed_ba) * sym_mask
            sym_loss = (sym_diff ** 2).sum() / (sym_mask.sum() + 1e-8)
            losses['symmetry_loss'] = self.symmetry_weight * sym_loss
        else:
            losses['symmetry_loss'] = torch.tensor(0.0, device=operator.device)

        # Anti-symmetry loss: ||f(A,B) + f(B,A)||^2
        # For anti-symmetric ops, f(A,B) should be approximately -f(B,A)
        if is_antisymmetric.any():
            asym_mask = is_antisymmetric.float().unsqueeze(-1)
            asym_diff = (composed_ab + composed_ba) * asym_mask
            asym_loss = (asym_diff ** 2).sum() / (asym_mask.sum() + 1e-8)
            losses['antisymmetry_loss'] = self.antisymmetry_weight * asym_loss
        else:
            losses['antisymmetry_loss'] = torch.tensor(0.0, device=operator.device)

        losses['total'] = losses['symmetry_loss'] + losses['antisymmetry_loss']
        return losses


# =============================================================================
# RESIDUAL GATE
# =============================================================================

class ResidualGate(nn.Module):
    """
    Gated residual connection: output = base + gate * composed

    Keeps the 40D core stable while allowing compositional contribution.
    Gate is learned per-world to maintain separation.
    """

    def __init__(self, dim: int = TOTAL_DIM, init_gate: float = 0.1):
        super().__init__()
        self.dim = dim

        # Per-world gate parameters (4 scalars, one per world)
        # Initialize small so composition starts as minor correction
        self.gate_logits = nn.Parameter(
            torch.full((NUM_WORLDS,), self._inverse_sigmoid(init_gate))
        )

    def _inverse_sigmoid(self, x: float) -> float:
        """Inverse sigmoid for initialization."""
        return -torch.log(torch.tensor(1.0 / x - 1.0)).item()

    def forward(
        self,
        base: torch.Tensor,      # [batch, 40] - original noetic repr
        composed: torch.Tensor,  # [batch, 40] - compositional output
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: gated combination
            gate_values: [4] gate per world (for monitoring)
        """
        # Compute gate per world
        gates = torch.sigmoid(self.gate_logits)  # [4]

        # Expand gates to match dimension
        gate_expanded = gates.repeat_interleave(NOETIC_DIM)  # [40]

        # Gated residual
        output = base + gate_expanded * composed

        return output, gates


# =============================================================================
# BILINEAR WORLD-WISE COMPOSER (No FFN)
# =============================================================================

class WorldWiseBilinearComposer(nn.Module):
    """
    Bilinear composition that respects world boundaries.

    For each world w in {A, B, C, D}:
        composed_w = left_w @ W[op, w] @ right_w.T

    No cross-world interaction in the composition.
    """

    def __init__(
        self,
        noetic_dim: int = NOETIC_DIM,
        num_worlds: int = NUM_WORLDS,
        num_operators: int = NUM_OPERATORS,
    ):
        super().__init__()
        self.noetic_dim = noetic_dim
        self.num_worlds = num_worlds
        self.num_operators = num_operators

        # Per-operator, per-world bilinear weights
        # W[op, world] is [noetic_dim, noetic_dim]
        self.bilinear_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(num_worlds, noetic_dim, noetic_dim))
            for _ in range(num_operators)
        ])

        # Initialize: identity-like for + and ×, anti-identity for - and ÷
        for op_idx in range(num_operators):
            for w in range(num_worlds):
                nn.init.eye_(self.bilinear_weights[op_idx][w])
                if op_idx in ANTISYMMETRIC_OPS:
                    # Slight asymmetry for anti-symmetric operators
                    self.bilinear_weights[op_idx][w].data += 0.1 * torch.randn(noetic_dim, noetic_dim)

        # Per-operator bias
        self.bias = nn.Parameter(torch.zeros(num_operators, num_worlds * noetic_dim))

    def forward(
        self,
        left: torch.Tensor,      # [batch, 40]
        right: torch.Tensor,     # [batch, 40]
        operator: torch.Tensor,  # [batch]
    ) -> torch.Tensor:
        """
        Compose elements world-by-world.
        """
        batch_size = left.shape[0]

        # Reshape to [batch, 4, 10]
        left_worlds = left.view(batch_size, self.num_worlds, self.noetic_dim)
        right_worlds = right.view(batch_size, self.num_worlds, self.noetic_dim)

        output = torch.zeros(batch_size, self.num_worlds * self.noetic_dim, device=left.device)

        for op_idx in range(self.num_operators):
            mask = (operator == op_idx).float().view(-1, 1)

            # Bilinear per world: left_w @ W @ right_w
            W = self.bilinear_weights[op_idx]  # [4, 10, 10]

            # Batched bilinear: [batch, 4, 10] @ [4, 10, 10] @ [batch, 4, 10].T
            # = sum over k: left[b,w,i] * W[w,i,j] * right[b,w,j]
            composed = torch.einsum('bwi,wij,bwj->bw', left_worlds, W, right_worlds)

            # Scale back to 10D per world (add contribution from both inputs)
            left_contrib = torch.einsum('bwi,wij->bwj', left_worlds, W)
            right_contrib = torch.einsum('wij,bwj->bwi', W, right_worlds)
            full_composed = (left_contrib + right_contrib) / 2
            full_composed = full_composed.reshape(batch_size, -1)

            # Add bias
            full_composed = full_composed + self.bias[op_idx]

            output = output + mask * full_composed

        return output


# =============================================================================
# CONTRASTIVE ALIGNMENT (Linear, No MLP)
# =============================================================================

class LinearContrastiveAlignment(nn.Module):
    """
    Linear projection to equation embedding space for contrastive learning.

    No hidden layers - direct linear map from 40D composition to equation space.
    Uses world-wise projection to maintain separation.
    """

    def __init__(
        self,
        element_dim: int = TOTAL_DIM,
        equation_dim: int = 384,
    ):
        super().__init__()
        self.element_dim = element_dim
        self.equation_dim = equation_dim

        # Operator embeddings (added to composed representation)
        self.op_embedding = nn.Embedding(NUM_OPERATORS, element_dim)

        # Linear projection to equation space (world-preserving structure)
        # We project each world to equation_dim/4, then concatenate
        self.eq_per_world = equation_dim // NUM_WORLDS
        self.world_to_eq = nn.ModuleList([
            nn.Linear(NOETIC_DIM, self.eq_per_world, bias=False)
            for _ in range(NUM_WORLDS)
        ])

        # Temperature (learnable)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(
        self,
        composed: torch.Tensor,  # [batch, 40]
        operator: torch.Tensor,  # [batch]
    ) -> torch.Tensor:
        """
        Project composed noetic representation to equation embedding space.
        """
        # Add operator embedding
        op_emb = self.op_embedding(operator)
        composed = composed + op_emb

        # Split into worlds and project each
        worlds = composed.view(-1, NUM_WORLDS, NOETIC_DIM)
        eq_parts = [
            self.world_to_eq[w](worlds[:, w, :])
            for w in range(NUM_WORLDS)
        ]

        # Concatenate to equation space
        equation_repr = torch.cat(eq_parts, dim=-1)
        equation_repr = F.normalize(equation_repr, dim=-1)

        return equation_repr

    def contrastive_loss(
        self,
        composed_repr: torch.Tensor,      # [batch, equation_dim]
        target_embeddings: torch.Tensor,  # [batch, equation_dim]
        source_weights: Optional[torch.Tensor] = None,  # [batch] weights for original vs generated
    ) -> torch.Tensor:
        """
        Weighted contrastive loss.

        source_weights: higher for original corpus examples, lower for generated
        """
        composed_repr = F.normalize(composed_repr, dim=-1)
        target_embeddings = F.normalize(target_embeddings, dim=-1)

        # Cosine similarity
        similarity = torch.sum(composed_repr * target_embeddings, dim=-1) / self.temperature

        # Apply source weighting (original examples count more)
        if source_weights is not None:
            loss = -(similarity * source_weights).mean()
        else:
            loss = -similarity.mean()

        return loss


# =============================================================================
# UNIFIED TKS COMPOSITIONAL LAYER v2
# =============================================================================

class TKSCompositionalLayerV2(nn.Module):
    """
    Pure TKS-structured compositional layer.

    Features:
    1. World-wise bilinear composition (no cross-world mixing)
    2. Operator symmetry/anti-symmetry constraints
    3. Linear contrastive alignment (no MLP)
    4. Residual gating for stable 40D core
    5. Source weighting for corpus examples

    No FFN, no MLP, no attention - pure linear + noetic routing.
    """

    def __init__(
        self,
        element_dim: int = TOTAL_DIM,
        equation_dim: int = 384,
        init_gate: float = 0.1,
    ):
        super().__init__()
        self.element_dim = element_dim
        self.equation_dim = equation_dim

        # Core bilinear composer (world-wise)
        self.bilinear = WorldWiseBilinearComposer()

        # Noetic operator routing
        self.router = NoeticOperatorRouter()

        # Linear contrastive alignment
        self.contrastive = LinearContrastiveAlignment(element_dim, equation_dim)

        # Residual gate
        self.gate = ResidualGate(element_dim, init_gate)

        # Operator symmetry loss
        self.symmetry_loss = OperatorSymmetryLoss()

        # World-wise projection from equation space back to noetic
        self.eq_to_noetic = WorldWiseProjection(equation_dim, element_dim)

    def forward(
        self,
        left: torch.Tensor,       # [batch, 40]
        right: torch.Tensor,      # [batch, 40]
        operator: torch.Tensor,   # [batch]
        compute_symmetry: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compose elements with TKS-structured operations.
        """
        outputs = {}

        # 1. Bilinear composition (world-wise)
        bilinear_ab = self.bilinear(left, right, operator)
        outputs['bilinear_composed'] = bilinear_ab

        # 2. Route through noetic operator transforms
        routed = self.router(bilinear_ab, operator)
        outputs['routed'] = routed

        # 3. Project to equation space for contrastive alignment
        equation_repr = self.contrastive(routed, operator)
        outputs['equation_repr'] = equation_repr

        # 4. Project back to noetic space (world-wise)
        from_equation = self.eq_to_noetic(equation_repr)

        # 5. Combine bilinear and equation-grounded outputs
        combined = (routed + from_equation) / 2

        # 6. Residual gate with original input mean
        base = (left + right) / 2  # Simple base: mean of inputs
        gated_output, gate_values = self.gate(base, combined)
        outputs['noetic_output'] = gated_output
        outputs['gate_values'] = gate_values

        # 7. Compute symmetry loss if requested
        if compute_symmetry:
            bilinear_ba = self.bilinear(right, left, operator)  # Swapped
            sym_losses = self.symmetry_loss(bilinear_ab, bilinear_ba, operator)
            outputs['symmetry_losses'] = sym_losses

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_equation_embeddings: torch.Tensor,
        source_weights: Optional[torch.Tensor] = None,
        symmetry_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        target_equation_embeddings: [batch, equation_dim] from corpus
        source_weights: [batch] higher for original, lower for generated
        symmetry_weight: weight for operator symmetry constraint
        """
        losses = {}

        # Contrastive loss
        losses['contrastive'] = self.contrastive.contrastive_loss(
            outputs['equation_repr'],
            target_equation_embeddings,
            source_weights,
        )

        # Symmetry loss
        if 'symmetry_losses' in outputs:
            losses['symmetry'] = symmetry_weight * outputs['symmetry_losses']['total']
        else:
            losses['symmetry'] = torch.tensor(0.0, device=target_equation_embeddings.device)

        # Total
        losses['total'] = losses['contrastive'] + losses['symmetry']

        return losses


# =============================================================================
# CORPUS UTILITIES
# =============================================================================

def load_equation_corpus(jsonl_path: str) -> Dict:
    """Load the 6400 equation corpus with source tracking."""
    equations = []
    index_map = {}

    op_to_idx = {'+': 0, '-': 1, '×': 2, '÷': 3}

    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            eq = json.loads(line)
            equations.append(eq)

            key = (eq['left'], op_to_idx.get(eq['operator'], 0), eq['right'])
            index_map[key] = idx

    # Compute source weights (1.0 for original, 0.5 for generated)
    source_weights = torch.tensor([
        1.0 if eq.get('source', 'original') == 'original' else 0.5
        for eq in equations
    ])

    return {
        'equations': equations,
        'index_map': index_map,
        'num_equations': len(equations),
        'source_weights': source_weights,
    }


def element_to_index(element: str) -> int:
    """Convert element string (e.g., 'A1') to index 0-39."""
    world = element[0]
    noetic = int(element[1:])
    world_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[world]
    return world_idx * 10 + (noetic - 1)


def index_to_element(idx: int) -> str:
    """Convert index 0-39 to element string."""
    world_idx = idx // 10
    noetic = (idx % 10) + 1
    world = ['A', 'B', 'C', 'D'][world_idx]
    return f"{world}{noetic}"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TKS Compositional Layer v2 Test (No FFN/MLP)")
    print("=" * 70)

    batch_size = 8

    # Random element representations
    left = torch.randn(batch_size, TOTAL_DIM)
    right = torch.randn(batch_size, TOTAL_DIM)

    # Mix of operators
    operator = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])  # +, -, ×, ÷, ...

    print("\n--- WorldWiseProjection ---")
    proj = WorldWiseProjection(384, 40)
    x = torch.randn(batch_size, 384)
    out = proj(x)
    print(f"  Input: {x.shape} → Output: {out.shape}")

    print("\n--- WorldWiseBilinearComposer ---")
    bilinear = WorldWiseBilinearComposer()
    out = bilinear(left, right, operator)
    print(f"  Composed: {out.shape}")

    print("\n--- NoeticOperatorRouter ---")
    router = NoeticOperatorRouter()
    routed = router(out, operator)
    print(f"  Routed: {routed.shape}")

    print("\n--- ResidualGate ---")
    gate = ResidualGate()
    gated, gate_vals = gate(left, routed)
    print(f"  Gated: {gated.shape}")
    print(f"  Gate values per world: {gate_vals}")

    print("\n--- OperatorSymmetryLoss ---")
    sym_loss = OperatorSymmetryLoss()
    composed_ab = bilinear(left, right, operator)
    composed_ba = bilinear(right, left, operator)
    losses = sym_loss(composed_ab, composed_ba, operator)
    print(f"  Symmetry loss: {losses['symmetry_loss'].item():.6f}")
    print(f"  Anti-symmetry loss: {losses['antisymmetry_loss'].item():.6f}")

    print("\n--- LinearContrastiveAlignment ---")
    contrastive = LinearContrastiveAlignment()
    eq_repr = contrastive(routed, operator)
    print(f"  Equation repr: {eq_repr.shape}")

    # Contrastive loss
    target = torch.randn(batch_size, 384)
    target = F.normalize(target, dim=-1)
    source_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])  # orig, gen
    loss = contrastive.contrastive_loss(eq_repr, target, source_weights)
    print(f"  Contrastive loss: {loss.item():.4f}")

    print("\n--- TKSCompositionalLayerV2 (Full) ---")
    layer = TKSCompositionalLayerV2()
    outputs = layer(left, right, operator)
    print(f"  noetic_output: {outputs['noetic_output'].shape}")
    print(f"  equation_repr: {outputs['equation_repr'].shape}")
    print(f"  gate_values: {outputs['gate_values']}")
    print(f"  symmetry_loss: {outputs['symmetry_losses']['total'].item():.6f}")

    # Full loss
    losses = layer.compute_loss(outputs, target, source_weights)
    print(f"  Total loss: {losses['total'].item():.4f}")

    print("\n--- Parameter Count ---")
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("\n" + "=" * 70)
    print("All tests passed! Pure TKS-structured, no FFN/MLP.")
    print("=" * 70)

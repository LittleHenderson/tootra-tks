"""
TKS-LLM Core Components v4 - No-Transformer Noetic LM

Goals:
- Decoder-only language model (causal)
- No nn.Transformer usage
- All internal representations stay in 40-dim noetic space
- v7.4 aligned: noetics, worlds, foundations, RPM, attractor

Pipeline:
  tokens -> noetic embedding -> noetic blocks (router + causal fractal attention)
        -> [operator core if equation detected] -> attractor -> RPM gating -> logits

v4.1: Added optional operator core for explicit TKS equation handling.
      Uses 6400 equation corpus as contrastive supervision.

Trace Schema (v1.0):
    When return_full_trace=True, the model returns a standardized trace dict:
    {
        "noetic_routing": {"weights": Tensor, "indices": Tensor (optional)},
        "attention": {"scale_weights": Tensor},
        "attractor": {"converged": bool, "iterations": int, "final_delta": float, "trajectory": Tensor (optional)},
        "rpm": {"gate": Tensor, "dwp_scores": Tensor, "foundation_idx": Tensor},
        "operator_core": {"gate_values": Tensor, "noetic_output": Tensor, "equation_repr": Tensor, "symmetry_losses": dict},
        "blocks": [...] (per-block traces),
        "_legacy": {...} (backward-compatible keys),
        "_schema_version": "1.0"
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tks_llm_core import NoeticProcessor, NOETIC_DIM, NUM_WORLDS, TOTAL_DIM
from tks_llm_core_v2 import (
    AttractorComputationLayer,
    StableAttractorLayer,
    RPMGatingMechanism,
    get_trace_schema_version,
    _build_standardized_trace,
    TRACE_SCHEMA_VERSION,
)

# Optional: Noetic Basis Parameterization (interpretable by construction)
NOETIC_BASIS_AVAILABLE = False
NoeticBasisLinear = None
try:
    import importlib.util
    import os
    _nb_path = os.path.join(os.path.dirname(__file__), "tks_features", "noetic_basis.py")
    if os.path.exists(_nb_path):
        _spec = importlib.util.spec_from_file_location("noetic_basis", _nb_path)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        NoeticBasisLinear = _module.NoeticBasisLinear
        NOETIC_BASIS_AVAILABLE = True
except Exception:
    pass

# Optional: TKS Expressiveness Stack (pure linear, no FFN/MHA)
TKS_EXPRESSIVENESS_AVAILABLE = False
TKSExpressivenessStack = None
try:
    import importlib.util
    import os
    _ext_path = os.path.join(os.path.dirname(__file__), "tks_features", "tks_noetic_extensions.py")
    if os.path.exists(_ext_path):
        _spec = importlib.util.spec_from_file_location("tks_noetic_extensions", _ext_path)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        TKSExpressivenessStack = _module.TKSExpressivenessStack
        TKS_EXPRESSIVENESS_AVAILABLE = True
except Exception:
    pass

# Optional: TKS Operator Core (equation-aware compositional layer)
TKS_OPERATOR_CORE_AVAILABLE = False
TKSCompositionalLayerV2 = None
try:
    import importlib.util
    import os
    _op_path = os.path.join(os.path.dirname(__file__), "tks_compositional_layer.py")
    if os.path.exists(_op_path):
        _spec = importlib.util.spec_from_file_location("tks_compositional_layer", _op_path)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        TKSCompositionalLayerV2 = _module.TKSCompositionalLayerV2
        TKS_OPERATOR_CORE_AVAILABLE = True
except Exception:
    pass

# Optional: Routing Stability (prevents routing collapse)
ROUTING_STABILITY_AVAILABLE = False
StableRouting = None
RoutingTemperatureScheduler = None
try:
    import importlib.util
    import os
    _rs_path = os.path.join(os.path.dirname(__file__), "tks_features", "routing_stability.py")
    if os.path.exists(_rs_path):
        _spec = importlib.util.spec_from_file_location("routing_stability", _rs_path)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        StableRouting = _module.StableRouting
        RoutingTemperatureScheduler = _module.RoutingTemperatureScheduler
        ROUTING_STABILITY_AVAILABLE = True
except Exception:
    pass

# Optional: World Bridge (cross-world mixing)
WORLD_BRIDGE_AVAILABLE = False
TKSWorldBridgeLayer = None
try:
    import importlib.util
    import os
    import sys
    _wb_path = os.path.join(os.path.dirname(__file__), "tks_features", "world_bridge.py")
    if os.path.exists(_wb_path):
        _spec = importlib.util.spec_from_file_location("world_bridge", _wb_path)
        _module = importlib.util.module_from_spec(_spec)
        # Register module in sys.modules before exec (required for Python 3.12 dataclasses)
        sys.modules["world_bridge"] = _module
        _spec.loader.exec_module(_module)
        TKSWorldBridgeLayer = _module.TKSWorldBridgeLayer
        WORLD_BRIDGE_AVAILABLE = True
except Exception:
    pass


@dataclass
class TKSNoeticLMConfig:
    vocab_size: int = 512
    max_seq_len: int = 256
    num_layers: int = 4
    num_scales: int = 3
    dropout: float = 0.1
    use_attractor: bool = True
    use_stable_attractor: bool = True  # Use StableAttractorLayer (guaranteed convergence)
    contraction_factor: float = 0.5
    max_attractor_iter: int = 15
    use_rpm: bool = True
    tie_embeddings: bool = False
    # TKS Expressiveness Stack config (pure linear, no FFN/MHA)
    use_subfoundation_mixer: bool = False
    use_acquisition_program: bool = False
    use_foundation_graph: bool = False
    subfoundation_scale: float = 0.1
    acquisition_scale: float = 0.1
    acquisition_depth: int = 2
    foundation_scale: float = 0.1
    diffusion_steps: int = 2
    foundation_hub_weight: float = 0.15
    # TKS Operator Core config (equation-aware, gated)
    use_operator_core: bool = False
    operator_gate_init: float = 0.1  # Initial gate value (small = stable core)
    equation_corpus_path: str = "data/equation_embeddings/equation_corpus.pt"
    operator_equation_dim: int = 384  # Equation embedding dimension
    # NL Retriever config (fallback when EquationDetector misses)
    use_nl_retriever: bool = False
    nl_retriever_path: str = "output/nl_retriever.pt"
    nl_retriever_threshold: float = 0.5  # Minimum similarity to use retrieved equation
    nl_retriever_corpus_path: str = "data/equation_embeddings/equation_corpus.pt"
    nl_retriever_meta_path: str = "data/equation_embeddings/corpus_meta.pt"
    # Noetic Basis Parameterization (interpretable by construction)
    use_noetic_basis: bool = False  # Enable noetic basis layers
    noetic_basis_residual: bool = True  # Allow residual stream for over-constraint mitigation
    noetic_basis_residual_budget: float = 0.1  # Residual budget (max fraction from residual)
    # Routing Stability config (prevents routing collapse)
    use_stable_routing: bool = False
    routing_initial_temp: float = 2.0
    routing_final_temp: float = 0.1
    routing_warmup_steps: int = 1000
    routing_anneal_steps: int = 10000
    routing_noise_type: str = "gumbel"  # "gumbel", "gaussian", "uniform"
    routing_initial_noise: float = 0.5
    routing_final_noise: float = 0.0
    routing_balance_weight: float = 0.01
    # World Bridge config (cross-world mixing)
    use_world_bridge: bool = False
    world_bridge_mode: str = "hybrid"  # "explicit", "foundation", "hybrid"
    world_bridge_rank: int = 2
    world_bridge_max_energy: float = 0.3


class NoeticTokenEmbedding(nn.Module):
    """Map tokens to 40-dim noetic space with world-wise transforms."""

    def __init__(self, vocab_size: int, noetic_dim: int = TOTAL_DIM) -> None:
        super().__init__()
        if noetic_dim != TOTAL_DIM:
            raise ValueError(f"noetic_dim must be {TOTAL_DIM} for canonical TKS, got {noetic_dim}")
        self.vocab_size = vocab_size
        self.noetic_dim = noetic_dim
        self.embedding = nn.Embedding(vocab_size, noetic_dim)

        self.world_A = nn.Linear(NOETIC_DIM, NOETIC_DIM)
        self.world_B = nn.Linear(NOETIC_DIM, NOETIC_DIM)
        self.world_C = nn.Linear(NOETIC_DIM, NOETIC_DIM)
        self.world_D = nn.Linear(NOETIC_DIM, NOETIC_DIM)
        for layer in (self.world_A, self.world_B, self.world_C, self.world_D):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: torch.LongTensor) -> Dict[str, torch.Tensor]:
        x = self.embedding(tokens)
        A = self.world_A(x[..., 0:10])
        B = self.world_B(x[..., 10:20])
        C = self.world_C(x[..., 20:30])
        D = self.world_D(x[..., 30:40])
        noetic = torch.cat([A, B, C, D], dim=-1)
        return {"noetic": noetic, "A": A, "B": B, "C": C, "D": D}


class NoeticPositionEmbedding(nn.Module):
    """Learned positional embedding in 40-dim noetic space."""

    def __init__(self, max_seq_len: int, noetic_dim: int = TOTAL_DIM) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, noetic_dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions).unsqueeze(0)


class NoeticRouter(nn.Module):
    """Route states through noetic transforms with a learned mixture.

    When use_noetic_basis=True, the router projection is parameterized as a
    weighted sum of the 10 canonical noetic basis operators per world,
    making it interpretable by construction.

    When use_stable_routing=True, applies temperature annealing, noise injection,
    and load balancing to prevent routing collapse.
    """

    def __init__(
        self,
        dim: int = TOTAL_DIM,
        use_noetic_basis: bool = False,
        noetic_basis_residual: bool = True,
        noetic_basis_residual_budget: float = 0.1,
        # Stable routing config
        use_stable_routing: bool = False,
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        warmup_steps: int = 1000,
        anneal_steps: int = 10000,
        noise_type: str = "gumbel",
        initial_noise: float = 0.5,
        final_noise: float = 0.0,
        balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.use_noetic_basis = use_noetic_basis and NOETIC_BASIS_AVAILABLE
        self.use_stable_routing = use_stable_routing and ROUTING_STABILITY_AVAILABLE
        self.dim = dim

        if self.use_noetic_basis:
            # Use noetic basis parameterization: interpretable by construction
            # NoeticBasisLinear processes 40D -> 40D with interpretable weights
            self._noetic_proj = NoeticBasisLinear(
                dim=dim,
                learn_residual=noetic_basis_residual,
                residual_budget=noetic_basis_residual_budget,
            )
            # Then project to 10D for routing weights
            self._route_head = nn.Linear(dim, 10)
        else:
            self.router = nn.Linear(dim, 10)
            self._noetic_proj = None
            self._route_head = None

        # Initialize stable routing wrapper if enabled
        self.stable_routing = None
        if self.use_stable_routing:
            base_router = self._route_head if self.use_noetic_basis else self.router
            self.stable_routing = StableRouting(
                base_router=base_router,
                num_experts=10,
                initial_temp=initial_temp,
                final_temp=final_temp,
                warmup_steps=warmup_steps,
                anneal_steps=anneal_steps,
                temp_schedule="cosine",
                noise_type=noise_type,
                initial_noise=initial_noise,
                final_noise=final_noise,
                noise_anneal_steps=anneal_steps,
                balance_weight=balance_weight,
            )

    def _get_router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get router logits, optionally using noetic basis."""
        if self.use_noetic_basis:
            projected = self._noetic_proj(x)
            return self._route_head(projected)
        else:
            return self.router(x)

    def get_noetic_basis_trace(self) -> Optional[Dict]:
        """Get interpretable trace from noetic basis projection."""
        if self.use_noetic_basis and self._noetic_proj is not None:
            # world_weights is a Parameter [4, 10] - get normalized version
            normalized_weights = F.softmax(self._noetic_proj.world_weights, dim=-1)
            return {
                "world_weights": normalized_weights.detach(),
                "interpretation": self._noetic_proj.get_interpretation(),
            }
        return None

    def _apply_index(self, x: torch.Tensor, processor: NoeticProcessor, idx: torch.Tensor) -> torch.Tensor:
        # idx: [batch, seq]
        outs = [layer(x) for layer in processor.noetics]
        stack = torch.stack(outs, dim=-2)
        weights = F.one_hot(idx, num_classes=10).float()
        weights = weights.unsqueeze(-1)
        mixed = (stack * weights).sum(dim=-2)
        return F.gelu(mixed)

    def _get_router_input(self, x: torch.Tensor) -> torch.Tensor:
        """Get input for stable routing (after noetic basis projection if used)."""
        if self.use_noetic_basis:
            return self._noetic_proj(x)
        else:
            return x

    def forward(
        self,
        x: torch.Tensor,
        processor: NoeticProcessor,
        noetic_idx: Optional[int] = None,
        noetic_weights: Optional[torch.Tensor] = None,
        fractal_indices: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through noetic router.

        Args:
            x: Input tensor [batch, seq, dim]
            processor: NoeticProcessor for transforms
            noetic_idx: Fixed noetic index to use
            noetic_weights: Pre-computed noetic weights
            fractal_indices: Fractal path indices
            step: Training step (for stable routing temperature annealing)

        Returns:
            output: Routed output tensor
            weights: Routing weights [batch, seq, 10]
            aux_loss: Auxiliary loss from stable routing (or None)
            metrics: Stability metrics dict (or None)
        """
        aux_loss = None
        metrics = None

        if fractal_indices is not None:
            # fractal_indices: [batch, seq, depth]
            out = x
            depth = fractal_indices.shape[-1]
            for d in range(depth):
                out = self._apply_index(out, processor, fractal_indices[..., d])
            weights = torch.zeros(x.shape[0], x.shape[1], 10, device=x.device)
            return out, weights, None, None

        if noetic_idx is not None:
            out = processor(x, noetic_idx=noetic_idx)
            weights = torch.zeros(x.shape[0], x.shape[1], 10, device=x.device)
            weights[..., noetic_idx] = 1.0
            return out, weights, None, None

        # Use stable routing if enabled and step is provided
        if self.stable_routing is not None and step is not None:
            router_input = self._get_router_input(x)
            noetic_weights, aux_loss, metrics = self.stable_routing(router_input, step=step)
        elif noetic_weights is None:
            noetic_weights = F.softmax(self._get_router_logits(x), dim=-1)

        outs = [layer(x) for layer in processor.noetics]
        stack = torch.stack(outs, dim=-2)
        mixed = (stack * noetic_weights.unsqueeze(-1)).sum(dim=-2)
        return F.gelu(mixed), noetic_weights, aux_loss, metrics


class CausalFractalAttentionMechanism(nn.Module):
    """Causal multi-scale attention with downsampled keys.

    When use_noetic_basis=True, the Q/K projections are parameterized as
    weighted sums of the 10 canonical noetic basis operators per world,
    making them interpretable by construction.
    """

    def __init__(
        self,
        dim: int = TOTAL_DIM,
        num_scales: int = 3,
        dropout: float = 0.0,
        use_noetic_basis: bool = False,
        noetic_basis_residual: bool = True,
        noetic_basis_residual_budget: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.use_noetic_basis = use_noetic_basis and NOETIC_BASIS_AVAILABLE

        if self.use_noetic_basis:
            # Use noetic basis for Q/K projections: interpretable by construction
            self.query_proj = NoeticBasisLinear(
                dim=dim,
                learn_residual=noetic_basis_residual,
                residual_budget=noetic_basis_residual_budget,
            )
            self.key_projs = nn.ModuleList([
                NoeticBasisLinear(
                    dim=dim,
                    learn_residual=noetic_basis_residual,
                    residual_budget=noetic_basis_residual_budget,
                )
                for _ in range(num_scales)
            ])
        else:
            self.query_proj = nn.Linear(dim, dim)
            self.key_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_scales)])

        # Scale projections and output remain standard Linear
        self.scale_projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_scales)])
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _downsample(self, x: torch.Tensor, factor: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if factor == 1:
            seq = x.shape[1]
            key_pos = torch.arange(seq, device=x.device)
            return x, key_pos
        batch, seq, dim = x.shape
        pad = (factor - (seq % factor)) % factor
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        new_seq = x.shape[1] // factor
        x = x.view(batch, new_seq, factor, dim).mean(dim=2)
        key_pos = torch.arange(new_seq, device=x.device) * factor + (factor - 1)
        key_pos = torch.clamp(key_pos, max=seq - 1)
        return x, key_pos

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply causal fractal attention.

        Args:
            x: [batch, seq, dim] input tensor
            attention_mask: Optional attention mask
            return_weights: Whether to return attention weights

        Returns:
            output: [batch, seq, dim] attended output
            attn_traces: Attention weights if return_weights=True, else None
            scale_weights: [batch, num_scales] per-scale mixture weights if return_weights=True
        """
        batch, seq, dim = x.shape
        query = self.query_proj(x)
        scale_outputs: List[torch.Tensor] = []
        scale_weights: List[torch.Tensor] = []
        attn_traces: List[torch.Tensor] = []

        q_pos = torch.arange(seq, device=x.device)

        for s in range(self.num_scales):
            factor = 2 ** s
            x_scale, key_pos = self._downsample(x, factor)
            x_proj = self.scale_projections[s](x_scale)
            keys = self.key_projs[s](x_proj)
            attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / (dim ** 0.5)
            mask = key_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = attn_weights * mask.unsqueeze(0)
            norm = attn_weights.sum(dim=-1, keepdim=True)
            attn_weights = torch.where(norm > 0, attn_weights / norm, attn_weights)
            if attention_mask is not None:
                attn_weights = attn_weights * attention_mask.unsqueeze(-1).float()
            attended = torch.matmul(attn_weights, x_proj)
            scale_outputs.append(attended)
            mean_attn = attn_weights.mean(dim=(1, 2), keepdim=True)
            scale_weights.append(mean_attn)
            if return_weights:
                attn_traces.append(attn_weights)

        mix_logits = torch.cat(scale_weights, dim=-1)
        mix_weights = F.softmax(mix_logits, dim=-1)
        output = torch.zeros_like(x)
        for s, scale_out in enumerate(scale_outputs):
            w = mix_weights[..., s:s + 1]
            output = output + w * scale_out
        output = self.output_proj(output)
        output = self.dropout(output)

        if return_weights:
            # Return attention traces (as list since different scales have different sizes)
            # and the per-scale mixture weights
            return output, attn_traces, mix_weights
        return output, None, None


class NoeticBlock(nn.Module):
    """Router + causal fractal attention block in noetic space.

    Optionally includes:
    - Noetic Basis Parameterization (interpretable by construction)
    - TKS Expressiveness Stack (post-attention):
        - SubFoundationMixer (28 sub-foundations)
        - AcquisitionProgram (22 sparse operators)
        - FoundationGraphDiffusion (7 foundations)
    """

    def __init__(
        self,
        dim: int = TOTAL_DIM,
        num_scales: int = 3,
        dropout: float = 0.1,
        # Noetic Basis Parameterization config
        use_noetic_basis: bool = False,
        noetic_basis_residual: bool = True,
        noetic_basis_residual_budget: float = 0.1,
        # Stable routing config
        use_stable_routing: bool = False,
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        warmup_steps: int = 1000,
        anneal_steps: int = 10000,
        noise_type: str = "gumbel",
        initial_noise: float = 0.5,
        final_noise: float = 0.0,
        balance_weight: float = 0.01,
        # TKS Expressiveness config
        use_subfoundation_mixer: bool = False,
        use_acquisition_program: bool = False,
        use_foundation_graph: bool = False,
        subfoundation_scale: float = 0.1,
        acquisition_scale: float = 0.1,
        acquisition_depth: int = 2,
        foundation_scale: float = 0.1,
        diffusion_steps: int = 2,
        foundation_hub_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.use_noetic_basis = use_noetic_basis
        self.use_stable_routing = use_stable_routing

        self.router = NoeticRouter(
            dim=dim,
            use_noetic_basis=use_noetic_basis,
            noetic_basis_residual=noetic_basis_residual,
            noetic_basis_residual_budget=noetic_basis_residual_budget,
            use_stable_routing=use_stable_routing,
            initial_temp=initial_temp,
            final_temp=final_temp,
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            noise_type=noise_type,
            initial_noise=initial_noise,
            final_noise=final_noise,
            balance_weight=balance_weight,
        )
        self.attn = CausalFractalAttentionMechanism(
            dim=dim,
            num_scales=num_scales,
            dropout=dropout,
            use_noetic_basis=use_noetic_basis,
            noetic_basis_residual=noetic_basis_residual,
            noetic_basis_residual_budget=noetic_basis_residual_budget,
        )
        self.dropout = nn.Dropout(dropout)

        # TKS Expressiveness Stack (pure linear, no FFN/MHA)
        self.expressiveness = None
        use_any = use_subfoundation_mixer or use_acquisition_program or use_foundation_graph
        if use_any and TKS_EXPRESSIVENESS_AVAILABLE:
            self.expressiveness = TKSExpressivenessStack(
                dim=dim,
                use_subfoundation_mixer=use_subfoundation_mixer,
                use_acquisition_program=use_acquisition_program,
                use_foundation_graph=use_foundation_graph,
                subfoundation_scale=subfoundation_scale,
                acquisition_scale=acquisition_scale,
                acquisition_depth=acquisition_depth,
                foundation_scale=foundation_scale,
                diffusion_steps=diffusion_steps,
                foundation_hub_weight=foundation_hub_weight,
            )

    def forward(
        self,
        x: torch.Tensor,
        processor: NoeticProcessor,
        noetic_idx: Optional[int] = None,
        noetic_weights: Optional[torch.Tensor] = None,
        fractal_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through noetic block.

        Args:
            x: [batch, seq, dim] input tensor
            processor: NoeticProcessor for transforms
            noetic_idx: Fixed noetic index to use
            noetic_weights: Pre-computed noetic weights
            fractal_indices: Fractal path indices
            attention_mask: Optional attention mask
            return_trace: Whether to return trace dict
            step: Training step (for stable routing temperature annealing)

        Returns:
            x: [batch, seq, dim] output tensor
            trace: Dict with standardized trace components:
                - noetic_routing: {weights: [batch, seq, 10], aux_loss, stability_metrics}
                - attention: {scale_weights: [batch, num_scales], attn_weights: [...]}
                - expressiveness: {...} if enabled
            routing_aux_loss: Auxiliary loss from stable routing (or None)
            routing_metrics: Stability metrics dict (or None)
        """
        trace: Dict[str, torch.Tensor] = {}
        routing_aux_loss = None
        routing_metrics = None

        # Step 1: Noetic routing
        routed, weights, routing_aux_loss, routing_metrics = self.router(
            self.norm1(x), processor, noetic_idx, noetic_weights, fractal_indices, step=step
        )
        x = x + self.dropout(routed)
        if return_trace:
            trace["noetic_routing"] = {"weights": weights}
            if noetic_idx is not None:
                trace["noetic_routing"]["indices"] = torch.full(
                    (x.shape[0], x.shape[1]),
                    noetic_idx,
                    dtype=torch.long,
                    device=x.device
                )
            # Include noetic basis interpretation if enabled
            if self.use_noetic_basis:
                router_basis_trace = self.router.get_noetic_basis_trace()
                if router_basis_trace is not None:
                    trace["noetic_routing"]["noetic_basis"] = router_basis_trace

        # Step 2: Causal fractal attention
        attn_out, attn_weights, scale_weights = self.attn(
            self.norm2(x), attention_mask=attention_mask, return_weights=return_trace
        )
        x = x + self.dropout(attn_out)
        if return_trace:
            trace["attention"] = {}
            if attn_weights is not None:
                trace["attention"]["attn_weights"] = attn_weights
            if scale_weights is not None:
                trace["attention"]["scale_weights"] = scale_weights

        # Step 3: TKS Expressiveness Stack (post-attention)
        if self.expressiveness is not None:
            x, expr_trace = self.expressiveness(x, return_full_trace=return_trace)
            if return_trace and expr_trace:
                trace["expressiveness"] = expr_trace

        # Add stable routing metrics to trace
        if return_trace and self.use_stable_routing:
            trace["noetic_routing"]["aux_loss"] = routing_aux_loss
            trace["noetic_routing"]["stability_metrics"] = routing_metrics

        # Legacy keys for backward compatibility
        if return_trace:
            trace["noetic_weights"] = weights
            if attn_weights is not None:
                trace["attn_weights"] = attn_weights

        return x, trace, routing_aux_loss, routing_metrics


class TKSNoeticLM(nn.Module):
    """Decoder-only noetic LM without nn.Transformer."""

    def __init__(self, config: TKSNoeticLMConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.noetic_dim = TOTAL_DIM

        self.embedding = NoeticTokenEmbedding(vocab_size=config.vocab_size)
        self.position = NoeticPositionEmbedding(max_seq_len=config.max_seq_len)
        self.processor = NoeticProcessor(dim=self.noetic_dim)

        self.blocks = nn.ModuleList([
            NoeticBlock(
                dim=self.noetic_dim,
                num_scales=config.num_scales,
                dropout=config.dropout,
                # Noetic Basis Parameterization config
                use_noetic_basis=config.use_noetic_basis,
                noetic_basis_residual=config.noetic_basis_residual,
                noetic_basis_residual_budget=config.noetic_basis_residual_budget,
                # Stable routing config
                use_stable_routing=config.use_stable_routing,
                initial_temp=config.routing_initial_temp,
                final_temp=config.routing_final_temp,
                warmup_steps=config.routing_warmup_steps,
                anneal_steps=config.routing_anneal_steps,
                noise_type=config.routing_noise_type,
                initial_noise=config.routing_initial_noise,
                final_noise=config.routing_final_noise,
                balance_weight=config.routing_balance_weight,
                # TKS Expressiveness config
                use_subfoundation_mixer=config.use_subfoundation_mixer,
                use_acquisition_program=config.use_acquisition_program,
                use_foundation_graph=config.use_foundation_graph,
                subfoundation_scale=config.subfoundation_scale,
                acquisition_scale=config.acquisition_scale,
                acquisition_depth=config.acquisition_depth,
                foundation_scale=config.foundation_scale,
                diffusion_steps=config.diffusion_steps,
                foundation_hub_weight=config.foundation_hub_weight,
            )
            for _ in range(config.num_layers)
        ])

        # World Bridge (cross-world mixing) - applied after blocks, before attractor
        self.world_bridge = None
        if config.use_world_bridge and WORLD_BRIDGE_AVAILABLE:
            self.world_bridge = TKSWorldBridgeLayer(
                world_dim=NOETIC_DIM,
                bridge_rank=config.world_bridge_rank,
                max_bridge_energy=config.world_bridge_max_energy,
                mode=config.world_bridge_mode,
            )

        # Training step counter for temperature annealing
        self._training_step = 0

        # Operator Core (equation-aware, gated) - injects pre-attractor
        self.operator_core = None
        self.operator_corpus = None
        if config.use_operator_core and TKS_OPERATOR_CORE_AVAILABLE:
            self.operator_core = TKSCompositionalLayerV2(
                element_dim=self.noetic_dim,
                equation_dim=config.operator_equation_dim,
                init_gate=config.operator_gate_init,
            )
            # Load corpus embeddings if available
            try:
                import os
                corpus_path = config.equation_corpus_path
                if not os.path.isabs(corpus_path):
                    corpus_path = os.path.join(os.path.dirname(__file__), corpus_path)
                if os.path.exists(corpus_path):
                    self.operator_corpus = torch.load(corpus_path, map_location='cpu')
            except Exception:
                pass

        # NL Retriever (fallback when EquationDetector misses)
        self.nl_retriever = None
        self.nl_retriever_corpus = None
        self.nl_retriever_meta = None
        if config.use_nl_retriever:
            try:
                import os
                from tks_nl_retriever import TKSNLRetriever

                # Load retriever model
                retriever_path = config.nl_retriever_path
                if not os.path.isabs(retriever_path):
                    retriever_path = os.path.join(os.path.dirname(__file__), retriever_path)

                if os.path.exists(retriever_path):
                    checkpoint = torch.load(retriever_path, map_location='cpu', weights_only=False)
                    retriever_config = checkpoint.get('config', {})

                    self.nl_retriever = TKSNLRetriever(
                        nl_dim=retriever_config.get('nl_dim', 384),
                        eq_dim=retriever_config.get('eq_dim', 384),
                        hidden_dim=retriever_config.get('hidden_dim', 256),
                        use_hidden=retriever_config.get('use_hidden', True),
                        dropout=retriever_config.get('dropout', 0.1)
                    )
                    self.nl_retriever.load_state_dict(checkpoint['model_state_dict'])
                    self.nl_retriever.eval()

                    # Load equation corpus for retrieval
                    corpus_path = config.nl_retriever_corpus_path
                    if not os.path.isabs(corpus_path):
                        corpus_path = os.path.join(os.path.dirname(__file__), corpus_path)
                    if os.path.exists(corpus_path):
                        self.nl_retriever_corpus = torch.load(corpus_path, map_location='cpu')

                    # Load corpus metadata (contains triplet info)
                    meta_path = config.nl_retriever_meta_path
                    if not os.path.isabs(meta_path):
                        meta_path = os.path.join(os.path.dirname(__file__), meta_path)
                    if os.path.exists(meta_path):
                        self.nl_retriever_meta = torch.load(meta_path, map_location='cpu', weights_only=False)

            except Exception as e:
                # Silently fail if retriever loading fails
                self.nl_retriever = None
                pass

        if config.use_attractor:
            if config.use_stable_attractor:
                self.attractor = StableAttractorLayer(
                    dim=self.noetic_dim,
                    contraction_factor=config.contraction_factor,
                    max_iterations=config.max_attractor_iter,
                )
            else:
                self.attractor = AttractorComputationLayer(
                    dim=self.noetic_dim,
                    contraction_factor=config.contraction_factor,
                    max_iterations=config.max_attractor_iter,
                )
        else:
            self.attractor = None
        self.rpm_gating = RPMGatingMechanism(dim=self.noetic_dim) if config.use_rpm else None

        self.output_projection = nn.Linear(self.noetic_dim, config.vocab_size)
        if config.tie_embeddings:
            self.output_projection.weight = self.embedding.embedding.weight

        self.final_norm = nn.LayerNorm(self.noetic_dim)

    def _split_worlds(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "A": x[..., 0:10],
            "B": x[..., 10:20],
            "C": x[..., 20:30],
            "D": x[..., 30:40],
        }

    def set_training_step(self, step: int) -> None:
        """Set current training step for temperature annealing in stable routing."""
        self._training_step = step

    def get_training_step(self) -> int:
        """Get current training step."""
        return self._training_step

    def _get_noetic_basis_summary(self) -> Optional[Dict]:
        """
        Get aggregated noetic basis interpretation from all blocks.

        Returns:
            Dict with noetic basis interpretation summary, or None if not enabled.
        """
        if not self.config.use_noetic_basis:
            return None

        block_interpretations = []
        for i, block in enumerate(self.blocks):
            router_trace = block.router.get_noetic_basis_trace()
            if router_trace is not None:
                block_interpretations.append({
                    "block_idx": i,
                    "router": router_trace,
                })

        if not block_interpretations:
            return None

        return {
            "enabled": True,
            "residual_budget": self.config.noetic_basis_residual_budget,
            "blocks": block_interpretations,
        }

    def _retrieve_equation_for_nl(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], float]:
        """
        Use NL retriever to find matching equation for NL input.

        Args:
            hidden_states: [batch, seq, 40] - encoder hidden states

        Returns:
            triplet: (left_idx, operator_idx, right_idx) as tensors, or None if no good match
            similarity: similarity score (0.0 if no retriever or no match)
        """
        if self.nl_retriever is None or self.nl_retriever_corpus is None or self.nl_retriever_meta is None:
            return None, 0.0

        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # Pool hidden states to get sequence-level representation
        # Use mean pooling over sequence dimension
        pooled = hidden_states.mean(dim=1)  # [batch, 40]

        # Project to NL embedding space (384D) for retriever
        # The retriever expects 384D NL embeddings, but we have 40D noetic
        # We need to expand 40D -> 384D
        # Simple approach: repeat each dimension or use a learned projection
        # For now, use a simple expansion by repeating world blocks
        # 40D -> 384D: repeat each 10D world 9.6 times ≈ 96D per world
        # Actually, let's use a more principled approach: pad with zeros and use linear projection
        # But since we don't have that layer, we'll use the hidden states directly
        # The retriever will need to handle this mismatch

        # Actually, looking at the retriever, it expects 384D NL embeddings
        # We need to either:
        # 1. Add a projection layer (40D -> 384D) - requires modifying architecture
        # 2. Use the hidden states as-is and pad/repeat
        # 3. Skip this for now and document the requirement

        # For this implementation, let's create a simple projection on-the-fly
        # This is not ideal but allows the integration to work
        if not hasattr(self, '_nl_projection'):
            # Create projection layer on first call
            self._nl_projection = nn.Linear(self.noetic_dim, 384, bias=False).to(device)
            # Initialize with small random weights
            nn.init.xavier_uniform_(self._nl_projection.weight, gain=0.1)

        # Project to 384D
        nl_embedding = self._nl_projection(pooled)  # [batch, 384]

        # Normalize
        nl_embedding = F.normalize(nl_embedding, p=2, dim=-1)

        # Get equation corpus on correct device
        corpus = self.nl_retriever_corpus.to(device)
        if corpus.dim() == 2:
            # corpus is [num_equations, 384]
            pass
        else:
            # corpus might be dict or other format
            # Handle appropriately
            if isinstance(corpus, dict) and 'embeddings' in corpus:
                corpus = corpus['embeddings'].to(device)
            else:
                return None, 0.0

        # Retrieve best matching equations for batch
        triplets = self.nl_retriever.batch_retrieve_triplets(
            nl_embedding,
            corpus,
            self.nl_retriever_meta
        )

        # For now, return the first item in batch
        # In full implementation, handle batches properly
        if len(triplets) == 0 or triplets[0] is None:
            return None, 0.0

        best = triplets[0]
        similarity = best['similarity']

        # Check threshold
        if similarity < self.config.nl_retriever_threshold:
            return None, similarity

        # Convert to tensors
        left_idx = torch.tensor([best['left_idx']], dtype=torch.long, device=device)
        operator_idx = torch.tensor([best['operator_idx']], dtype=torch.long, device=device)
        right_idx = torch.tensor([best['right_idx']], dtype=torch.long, device=device)

        return (left_idx, operator_idx, right_idx), similarity

    def forward(
        self,
        tokens: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noetic_idx: Optional[int] = None,
        noetic_weights: Optional[torch.Tensor] = None,
        fractal_indices: Optional[torch.Tensor] = None,
        goal_state: Optional[torch.Tensor] = None,
        target_foundation: Optional[int] = None,
        # Equation triplet for operator core (optional)
        equation_triplet: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        return_full_trace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TKS Noetic LM.

        Args:
            tokens: [batch, seq] input token indices
            attention_mask: Optional attention mask
            noetic_idx: Fixed noetic index (0-9) to use
            noetic_weights: Pre-computed noetic weights
            fractal_indices: Fractal path indices
            goal_state: [batch, dim] optional goal representation
            target_foundation: int 0-6 for RPM gating
            equation_triplet: (left_idx, operator_idx, right_idx) for operator core
            return_full_trace: Whether to return standardized trace (v1.0 schema)

        Returns:
            Dict with 'logits', 'gated_output', and optionally full trace.
            If return_full_trace=True, 'trace' contains standardized schema v1.0:
                - noetic_routing: {weights, indices} (aggregated from blocks)
                - attention: {scale_weights} (aggregated from blocks)
                - attractor: {converged, iterations, final_delta, trajectory, state}
                - rpm: {gate, dwp_scores, foundation_idx}
                - operator_core: {gate_values, noetic_output, equation_repr, symmetry_losses}
                - blocks: [...] (per-block traces)
                - _legacy: {embedding, worlds} for backward compat
                - _schema_version: "1.0"
        """
        # Collect trace components
        block_traces = []
        trace_embedding = None
        trace_worlds = None

        emb = self.embedding(tokens)
        x = emb["noetic"] + self.position(tokens.shape[1], tokens.device)
        if return_full_trace:
            trace_embedding = emb["noetic"].detach()
            trace_worlds = {k: v.detach() for k, v in emb.items() if k != "noetic"}

        total_routing_aux_loss = None
        all_routing_metrics = []

        for layer in self.blocks:
            x, layer_trace, routing_aux_loss, routing_metrics = layer(
                x,
                processor=self.processor,
                noetic_idx=noetic_idx,
                noetic_weights=noetic_weights,
                fractal_indices=fractal_indices,
                attention_mask=attention_mask,
                return_trace=return_full_trace,
                step=self._training_step if self.config.use_stable_routing else None,
            )
            if return_full_trace:
                block_traces.append(layer_trace)
            # Accumulate routing aux losses
            if routing_aux_loss is not None:
                if total_routing_aux_loss is None:
                    total_routing_aux_loss = routing_aux_loss
                else:
                    total_routing_aux_loss = total_routing_aux_loss + routing_aux_loss
            if routing_metrics is not None:
                all_routing_metrics.append(routing_metrics)

        x = self.final_norm(x)

        # Apply world bridge (cross-world mixing) before attractor
        world_bridge_trace = None
        if self.world_bridge is not None:
            x, world_bridge_trace = self.world_bridge(x)

        # Operator Core: inject equation-aware composition pre-attractor
        # New flow:
        # 1. If equation_triplet is provided (EquationDetector found equation) -> use it
        # 2. If no equation_triplet but NL retriever enabled -> try to retrieve equation
        # 3. If retrieval succeeds and similarity > threshold -> use retrieved triplet
        # 4. Otherwise -> skip operator core (pure NL path)
        operator_out = None
        retrieved_triplet = None
        retrieval_similarity = 0.0
        operator_core_trace = None

        # Determine which triplet to use
        triplet_to_use = equation_triplet

        if self.operator_core is not None:
            # If no explicit equation triplet provided, try NL retriever
            if equation_triplet is None and self.nl_retriever is not None:
                retrieved_triplet, retrieval_similarity = self._retrieve_equation_for_nl(x)
                if retrieved_triplet is not None:
                    triplet_to_use = retrieved_triplet

            # Run operator core if we have a triplet (explicit or retrieved)
            if triplet_to_use is not None:
                left_idx, op_idx, right_idx = triplet_to_use
                batch_size = x.shape[0]

                # Convert indices to one-hot vectors
                left_vec = F.one_hot(left_idx, num_classes=self.noetic_dim).float()
                right_vec = F.one_hot(right_idx, num_classes=self.noetic_dim).float()

                # Run operator core
                operator_out = self.operator_core(
                    left_vec, right_vec, op_idx,
                    compute_symmetry=self.training,
                )
                comp_output = operator_out['noetic_output']  # [batch, 40]

                # Gated injection into sequence (broadcast to all positions)
                # The gate is inside the operator_core, so we just add
                x = x + comp_output.unsqueeze(1)  # [batch, seq, 40]

                if return_full_trace:
                    operator_core_trace = {
                        "gate_values": operator_out['gate_values'].detach(),
                        "noetic_output": comp_output.detach(),
                        "equation_repr": operator_out['equation_repr'].detach(),
                        "source": "explicit" if equation_triplet is not None else "nl_retriever",
                    }
                    if 'symmetry_losses' in operator_out:
                        operator_core_trace["symmetry_losses"] = {
                            k: v.detach() for k, v in operator_out['symmetry_losses'].items()
                        }

        attractor_out = None
        if self.attractor is not None:
            attractor_out = self.attractor(x, return_trajectory=return_full_trace)
            x = attractor_out["attractor"]

        rpm_out = None
        if self.rpm_gating is not None:
            rpm_out = self.rpm_gating(x, goal_state=goal_state, target_foundation=target_foundation)
            x = rpm_out["gated_output"]

        logits = self.output_projection(x)

        # Build standardized trace if requested
        trace = None
        if return_full_trace:
            # Aggregate noetic routing weights from last block (representative)
            last_block_noetic = None
            last_block_scale_weights = None
            if block_traces:
                last_trace = block_traces[-1]
                if "noetic_routing" in last_trace:
                    last_block_noetic = last_trace["noetic_routing"].get("weights")
                if "attention" in last_trace:
                    last_block_scale_weights = last_trace["attention"].get("scale_weights")

            # Compute foundation index from DWP product if RPM enabled
            rpm_foundation_idx = None
            if rpm_out is not None:
                dwp_product = rpm_out['dwp_scores'].prod(dim=-1)  # [batch, seq, 7]
                rpm_foundation_idx = dwp_product.argmax(dim=-1).detach()  # [batch, seq]

            # Build the standardized trace
            trace = _build_standardized_trace(
                # Noetic routing (from last block as representative)
                noetic_weights=last_block_noetic,
                noetic_indices=torch.full(
                    (tokens.shape[0], tokens.shape[1]),
                    noetic_idx if noetic_idx is not None else 0,
                    dtype=torch.long,
                    device=tokens.device
                ) if noetic_idx is not None else None,
                # Attention scale weights
                attention_scale_weights=last_block_scale_weights,
                # Attractor
                attractor_converged=attractor_out['converged'] if attractor_out else None,
                attractor_iterations=attractor_out['iterations'] if attractor_out else None,
                attractor_final_delta=attractor_out.get('final_delta', 0.0) if attractor_out else None,
                attractor_trajectory=torch.stack(attractor_out['trajectory'], dim=0) if attractor_out and attractor_out.get('trajectory') else None,
                attractor_state=attractor_out['attractor'].detach() if attractor_out else None,
                # RPM
                rpm_gate=rpm_out['rpm_gate'].detach() if rpm_out else None,
                rpm_dwp_scores=rpm_out['dwp_scores'].detach() if rpm_out else None,
                rpm_foundation_idx=rpm_foundation_idx,
                # Operator core
                operator_core_gate_values=operator_core_trace['gate_values'] if operator_core_trace else None,
                operator_core_noetic_output=operator_core_trace['noetic_output'] if operator_core_trace else None,
                operator_core_equation_repr=operator_core_trace['equation_repr'] if operator_core_trace else None,
                operator_core_symmetry_losses=operator_core_trace.get('symmetry_losses') if operator_core_trace else None,
                # Legacy
                legacy_embedding=trace_embedding,
                legacy_worlds=trace_worlds,
            )

            # Add blocks traces (detailed per-block info)
            trace["blocks"] = block_traces

            # Add NL retriever info if used
            if self.operator_core is not None:
                if retrieved_triplet is not None:
                    trace["nl_retriever"] = {
                        "used": True,
                        "similarity": retrieval_similarity,
                        "triplet": {
                            "left_idx": retrieved_triplet[0].item(),
                            "operator_idx": retrieved_triplet[1].item(),
                            "right_idx": retrieved_triplet[2].item(),
                        }
                    }
                elif equation_triplet is None and self.nl_retriever is not None:
                    trace["nl_retriever"] = {
                        "used": False,
                        "similarity": retrieval_similarity,
                        "reason": "below_threshold" if retrieval_similarity > 0 else "no_match"
                    }

            # Add operator_core directly to trace (top-level standardized key)
            if operator_core_trace:
                trace["operator_core"] = operator_core_trace

            # Add world bridge trace if enabled
            if world_bridge_trace is not None:
                trace["world_bridge"] = world_bridge_trace

            # Add routing stability metrics if enabled
            if self.config.use_stable_routing and all_routing_metrics:
                trace["routing_stability"] = {
                    "total_aux_loss": total_routing_aux_loss.item() if total_routing_aux_loss is not None else 0.0,
                    "per_block_metrics": all_routing_metrics,
                }

            # Add noetic basis interpretation summary if enabled
            if self.config.use_noetic_basis:
                noetic_basis_summary = self._get_noetic_basis_summary()
                if noetic_basis_summary:
                    trace["noetic_basis"] = noetic_basis_summary

            # Legacy top-level keys for backward compatibility
            trace["embedding"] = trace_embedding
            trace["worlds"] = trace_worlds
            if attractor_out:
                trace["attractor_state"] = attractor_out["attractor"].detach()  # Legacy key
                trace["attractor_converged"] = attractor_out["converged"]
                trace["attractor_iterations"] = attractor_out["iterations"]
            if rpm_out:
                trace["rpm_gate"] = rpm_out["rpm_gate"].detach()
                trace["dwp_scores"] = rpm_out["dwp_scores"].detach()

        out = {
            "logits": logits,
            "gated_output": x,
        }
        if attractor_out is not None:
            out["attractor_converged"] = attractor_out["converged"]
        if rpm_out is not None:
            out["rpm_gate"] = rpm_out["rpm_gate"]
        if operator_out is not None:
            out["operator_core_output"] = operator_out
        # Add routing auxiliary loss for training
        if total_routing_aux_loss is not None:
            out["routing_aux_loss"] = total_routing_aux_loss
        if return_full_trace:
            out["trace"] = trace
        return out

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        self.eval()
        for _ in range(max_new_tokens):
            if tokens.shape[1] > self.config.max_seq_len:
                tokens = tokens[:, -self.config.max_seq_len :]
            out = self(tokens)
            logits = out["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_val, torch.full_like(logits, -1e9), logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def load_noetic_lm(config: TKSNoeticLMConfig, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> TKSNoeticLM:
    model = TKSNoeticLM(config)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def load_pretrained_operator_core(
    model: TKSNoeticLM,
    pretrained_path: str,
    device: Optional[torch.device] = None,
) -> bool:
    """
    Load pretrained operator core weights into TKSNoeticLM.

    Args:
        model: TKSNoeticLM instance with operator_core enabled
        pretrained_path: Path to pretrained operator core checkpoint
        device: Device for loading weights (auto-detected if None)

    Returns:
        True if loaded successfully, False otherwise
    """
    import os

    if model.operator_core is None:
        print("Warning: Model has no operator_core (use_operator_core=False)")
        return False

    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained weights not found: {pretrained_path}")
        return False

    if device is None:
        device = next(model.parameters()).device

    try:
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.operator_core.load_state_dict(state_dict)
        print(f"Loaded pretrained operator core from: {pretrained_path}")
        return True
    except Exception as e:
        print(f"Failed to load operator core: {e}")
        return False


def create_v4_with_operator_core(
    vocab_size: int = 512,
    max_seq_len: int = 256,
    num_layers: int = 4,
    pretrained_operator_path: str = "output/operator_core_pretrained.pt",
    operator_gate_init: float = 0.1,
    device: Optional[torch.device] = None,
) -> TKSNoeticLM:
    """
    Create TKSNoeticLM with pretrained operator core.

    This is a convenience function for creating a v4 model with the
    equation-aware operator core already loaded.

    Args:
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        num_layers: Number of noetic blocks
        pretrained_operator_path: Path to pretrained operator core
        operator_gate_init: Initial gate value
        device: Target device

    Returns:
        TKSNoeticLM with pretrained operator core
    """
    config = TKSNoeticLMConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        use_operator_core=True,
        operator_gate_init=operator_gate_init,
    )

    model = TKSNoeticLM(config)

    if device is not None:
        model = model.to(device)

    load_pretrained_operator_core(model, pretrained_operator_path, device)

    return model


__all__ = [
    "TKSNoeticLMConfig",
    "TKSNoeticLM",
    "NoeticTokenEmbedding",
    "NoeticRouter",
    "CausalFractalAttentionMechanism",
    "NoeticBlock",
    "load_noetic_lm",
    "load_pretrained_operator_core",
    "create_v4_with_operator_core",
    "get_trace_schema_version",
    "TRACE_SCHEMA_VERSION",
    "NOETIC_BASIS_AVAILABLE",
]

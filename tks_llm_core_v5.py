"""
TKS-LLM Core Components v5 - General-Capable Architecture

Goals:
- Scale to General-Capable size (~10M-50M params)
- Decouple Model Dimension (Hidden) from Canonical Noetic Dimension (40)
- Preserves TKS trace and logic by privileging the first 40 dimensions
- Supports standard BPE tokenizers (vocab > 512)

Architecture:
- Embedding: Vocab -> Hidden Dim (e.g., 256, 384, 512)
- Blocks: Operate on Hidden Dim
- Noetic Routing: Projects Hidden -> 10 (routing weights) with STABLE ROUTING
- Attention: Causal Fractal Attention on Hidden Dim
- Canonical Trace: extracted from x[..., :40]

v5.1 Updates:
- Added StableRouting support to prevent router collapse
- Temperature annealing, noise injection, load balancing
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use v4 components where possible, but we need to redefine blocks for dimension handling
from tks_llm_core import NOETIC_DIM, NUM_WORLDS
from tks_llm_core_v4 import (
    NoeticProcessor,
    AttractorComputationLayer,
    RPMGatingMechanism,
    _build_standardized_trace,
    TRACE_SCHEMA_VERSION
)

# Optional: Routing Stability (prevents routing collapse) - RECOMMENDED FOR V5
ROUTING_STABILITY_AVAILABLE = False
StableRouting = None
try:
    import importlib.util
    import os
    _rs_path = os.path.join(os.path.dirname(__file__), "tks_features", "routing_stability.py")
    if os.path.exists(_rs_path):
        _spec = importlib.util.spec_from_file_location("routing_stability", _rs_path)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        StableRouting = _module.StableRouting
        ROUTING_STABILITY_AVAILABLE = True
except Exception:
    pass

# Optional: NJT Circuits (Noetic Judgment Transistors)
NJT_AVAILABLE = False
NJTLayer = None
NJTConfig = None
try:
    from tks_features.njt_circuits import NJTLayer, NJTConfig
    NJT_AVAILABLE = True
except ImportError:
    pass

# Canonical constants
CANONICAL_DIM = 40

@dataclass
class TKSGeneralConfig:
    vocab_size: int = 16384  # Scaled up for BPE
    hidden_dim: int = 384    # Main model width
    num_layers: int = 12     # Deeper
    num_scales: int = 4      # More fractal depth
    dropout: float = 0.1
    use_attractor: bool = True
    contraction_factor: float = 0.5
    max_attractor_iter: int = 15
    use_rpm: bool = True
    # Operator Core
    use_operator_core: bool = True
    operator_equation_dim: int = 384
    # Stable Routing (prevents router collapse) - ENABLED BY DEFAULT in v5
    use_stable_routing: bool = True  # RECOMMENDED: keeps routing diverse
    routing_initial_temp: float = 2.0
    routing_final_temp: float = 0.1
    routing_warmup_steps: int = 1000
    routing_anneal_steps: int = 10000
    routing_noise_type: str = "gumbel"
    routing_initial_noise: float = 0.5
    routing_final_noise: float = 0.0
    routing_balance_weight: float = 0.01
    # NJT (Noetic Judgment Transistor) circuits
    use_njt: bool = False  # Enable NJT circuits for signal amplification/dampening
    njt_num_transistors: int = 10  # Number of transistors per NJT bank
    njt_use_hysteresis: bool = True  # N5 badge - sticky reasoning paths
    njt_use_rhythm: bool = False  # N7 badge - rhythmic oscillation
    njt_initial_gain: float = 2.0  # Initial beta gain factor
    njt_initial_threshold: float = 0.5  # Initial gate threshold
    njt_gate_sharpness: float = 10.0  # k parameter for sigmoid gate
    njt_hysteresis_gap: float = 0.3  # Gap between on/off thresholds


class GeneralNoeticEmbedding(nn.Module):
    """Embeds tokens into Hidden Dim, initializes canonical subspace carefully."""
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # We want the first 40 dims to behave somewhat like the canonical worlds
        # But for general training, we let them learn.
        # Optionally, we could enforce structure, but soft alignment is better for scale.
        
    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        return self.embedding(tokens)

class GeneralNoeticRouter(nn.Module):
    """Routes based on Hidden Dim state with optional stable routing."""

    def __init__(
        self,
        dim: int,
        use_stable_routing: bool = True,
        # Stable routing config
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        warmup_steps: int = 1000,
        anneal_steps: int = 10000,
        noise_type: str = "gumbel",
        initial_noise: float = 0.5,
        final_noise: float = 0.0,
        balance_weight: float = 0.01,
    ):
        super().__init__()
        self.use_stable_routing = use_stable_routing and ROUTING_STABILITY_AVAILABLE

        # Base router projects from high-dim state to 10 routing weights
        self.router_head = nn.Linear(dim, 10)

        # Wrap with StableRouting if enabled
        self.stable_routing = None
        if self.use_stable_routing and StableRouting is not None:
            self.stable_routing = StableRouting(
                base_router=self.router_head,
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
                entropy_floor=0.5,
                entropy_window=100,
            )

    def forward(
        self,
        x: torch.Tensor,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through router.

        Args:
            x: Input tensor [batch, seq, dim]
            step: Training step (for stable routing temperature annealing)

        Returns:
            weights: Routing weights [batch, seq, 10]
            aux_loss: Auxiliary loss from stable routing (or None)
            metrics: Stability metrics dict (or None)
        """
        if self.stable_routing is not None and step is not None:
            weights, aux_loss, metrics = self.stable_routing(x, step=step)
            return weights, aux_loss, metrics

        # Fallback: simple softmax routing
        logits = self.router_head(x)
        weights = F.softmax(logits, dim=-1)
        return weights, None, None

class GeneralFractalAttention(nn.Module):
    """Causal Fractal Attention scaled to Hidden Dim."""
    def __init__(self, dim: int, num_scales: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_scales)])
        self.val_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_scales)]) # Added value projs for general capacity
        
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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch, seq, dim = x.shape
        query = self.query_proj(x)
        
        scale_outputs = []
        q_pos = torch.arange(seq, device=x.device)

        for s in range(self.num_scales):
            factor = 2 ** s
            x_down, key_pos = self._downsample(x, factor)
            
            k = self.key_projs[s](x_down)
            v = self.val_projs[s](x_down)
            
            attn_scores = torch.matmul(query, k.transpose(-2, -1)) / (dim ** 0.5)
            
            # Causal Masking (adapted for downsampling)
            mask = key_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask, -1e4)
            
            if attention_mask is not None:
                attn_mask = attention_mask
                if attn_mask.dim() == 4:
                    if attn_mask.size(2) == 1:
                        attn_mask = attn_mask[:, 0, 0, :]
                    else:
                        attn_mask = attn_mask[:, 0].any(dim=1)
                elif attn_mask.dim() == 3:
                    if attn_mask.size(1) == 1:
                        attn_mask = attn_mask[:, 0, :]
                    else:
                        attn_mask = attn_mask.any(dim=1)

                if attn_mask.dim() == 2:
                    key_mask = attn_mask[:, key_pos]
                    if key_mask.dtype != torch.bool:
                        key_mask = key_mask > 0
                    attn_scores = attn_scores.masked_fill(~key_mask.unsqueeze(1), -1e4)

            attn_weights = F.softmax(attn_scores, dim=-1)
            attended = torch.matmul(attn_weights, v)
            scale_outputs.append(attended)

        # Average across scales (simplified mixture)
        output = sum(scale_outputs) / self.num_scales
        output = self.output_proj(output)
        return self.dropout(output)

class GeneralNoeticBlock(nn.Module):
    def __init__(self, config: TKSGeneralConfig):
        super().__init__()
        self.dim = config.hidden_dim
        self.config = config
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)

        # Router: Projects hidden state to select processing path
        # With stable routing to prevent collapse (ENABLED BY DEFAULT in v5)
        self.router = GeneralNoeticRouter(
            dim=self.dim,
            use_stable_routing=config.use_stable_routing,
            initial_temp=config.routing_initial_temp,
            final_temp=config.routing_final_temp,
            warmup_steps=config.routing_warmup_steps,
            anneal_steps=config.routing_anneal_steps,
            noise_type=config.routing_noise_type,
            initial_noise=config.routing_initial_noise,
            final_noise=config.routing_final_noise,
            balance_weight=config.routing_balance_weight,
        )

        # Optional: NJT (Noetic Judgment Transistor) circuits
        self.njt = None
        self.use_njt = config.use_njt and NJT_AVAILABLE
        if self.use_njt and NJTLayer is not None:
            njt_config = NJTConfig(
                dim=config.hidden_dim,
                num_transistors=config.njt_num_transistors,
                initial_gain=config.njt_initial_gain,
                initial_threshold=config.njt_initial_threshold,
                gate_sharpness=config.njt_gate_sharpness,
                use_hysteresis=config.njt_use_hysteresis,
                hysteresis_gap=config.njt_hysteresis_gap,
                use_rhythm=config.njt_use_rhythm,
            )
            self.njt = NJTLayer(njt_config)

        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Linear(self.dim * 4, self.dim),
            nn.Dropout(config.dropout)
        )

        self.attn = GeneralFractalAttention(self.dim, config.num_scales, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        return_njt_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict], Optional[Dict]]:
        """
        Forward pass through noetic block.

        Args:
            x: Input tensor [batch, seq, dim]
            attention_mask: Optional attention mask
            step: Training step for stable routing temperature annealing
            return_njt_trace: Whether to return NJT trace info

        Returns:
            x: Output tensor
            r_weights: Routing weights
            aux_loss: Auxiliary loss from stable routing (or None)
            metrics: Stability metrics (or None)
            njt_trace: NJT trace info (or None)
        """
        njt_trace = None

        # 1. Routing
        # We calculate routing weights for the trace
        r_weights, aux_loss, metrics = self.router(self.norm1(x), step=step)

        # 2. Optional NJT processing (amplify/dampen signals based on context)
        if self.njt is not None:
            # NJT modulates the signal before FFN
            # Uses routing context as bias signal
            x_njt, njt_trace = self.njt(
                self.norm1(x),
                context=x,  # Use current state as context
                return_trace=return_njt_trace
            )
            x = x + x_njt  # Residual connection

        # 3. FFN Path
        h = self.ffn(self.norm1(x))
        x = x + h

        # 4. Attention
        attn_out = self.attn(self.norm2(x), attention_mask)
        x = x + attn_out

        return x, r_weights, aux_loss, metrics, njt_trace

class TKSGeneralLM(nn.Module):
    def __init__(self, config: TKSGeneralConfig):
        super().__init__()
        self.config = config
        
        self.embedding = GeneralNoeticEmbedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, config.hidden_dim)) # Simple learned pos
        
        self.blocks = nn.ModuleList([
            GeneralNoeticBlock(config) for _ in range(config.num_layers)
        ])
        
        # Canonical Components (Projected)
        # We need to extract 40-dim state for Attractor/RPM
        self.to_canonical = nn.Linear(config.hidden_dim, CANONICAL_DIM)
        self.from_canonical = nn.Linear(CANONICAL_DIM, config.hidden_dim)
        
        if config.use_attractor:
            self.attractor = AttractorComputationLayer(
                dim=CANONICAL_DIM, 
                contraction_factor=config.contraction_factor,
                max_iterations=config.max_attractor_iter
            )
            
        if config.use_rpm:
            self.rpm = RPMGatingMechanism(dim=CANONICAL_DIM)
            
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_full_trace=False,
        step: Optional[int] = None,
    ):
        """
        Forward pass through TKS General LM.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Optional attention mask
            return_full_trace: If True, return full trace info
            step: Training step for stable routing temperature annealing

        Returns:
            Dictionary containing:
            - logits: Output logits
            - routing_aux_loss: Total auxiliary loss from stable routing (if enabled)
            - trace: Full trace info (if return_full_trace=True)
        """
        b, s = input_ids.shape
        x = self.embedding(input_ids)
        x = x + self.pos_emb[:, :s, :]

        trace_routing = []
        trace_njt = []
        total_aux_loss = 0.0
        routing_metrics = {}

        for i, block in enumerate(self.blocks):
            x, weights, aux_loss, metrics, njt_trace = block(
                x, attention_mask, step=step, return_njt_trace=return_full_trace
            )
            if return_full_trace:
                trace_routing.append(weights)
                if njt_trace is not None:
                    trace_njt.append(njt_trace)

            # Accumulate auxiliary loss from stable routing
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

            # Store last layer's metrics
            if metrics is not None:
                routing_metrics = metrics

        x = self.final_norm(x)

        # TKS Canonical Loop (Attractor/RPM)
        # Project to 40-dim canonical space
        canon_x = self.to_canonical(x)

        canon_trace = {}
        if self.config.use_attractor:
            att_out = self.attractor(canon_x, return_trajectory=return_full_trace)
            canon_x = att_out["attractor"]
            if return_full_trace:
                canon_trace["attractor"] = att_out

        if self.config.use_rpm:
            # Note: RPM typically needs a goal state.
            # For general LM, we might skip or use self as goal.
            # Using simple gating for now.
            rpm_out = self.rpm(canon_x)
            canon_x = rpm_out["gated_output"]
            if return_full_trace:
                canon_trace["rpm"] = rpm_out

        # Project back residual (soft constraint)
        # We allow the general model to be influenced by the canonical result
        x = x + self.from_canonical(canon_x)

        logits = self.head(x)

        out = {"logits": logits}

        # Include routing auxiliary loss for training
        if isinstance(total_aux_loss, torch.Tensor):
            out["routing_aux_loss"] = total_aux_loss
        elif total_aux_loss != 0.0:
            out["routing_aux_loss"] = torch.tensor(total_aux_loss, device=logits.device)

        # Include routing metrics for monitoring
        if routing_metrics:
            out["routing_metrics"] = routing_metrics

        if return_full_trace:
            out["trace"] = {
                "routing": trace_routing,
                "canonical": canon_trace,
            }
            # Include NJT trace if available
            if trace_njt:
                out["trace"]["njt"] = trace_njt

        return out

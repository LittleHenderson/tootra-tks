"""
TKS-LLM Core Components v6 - Reasoning-Capable Architecture

Goals:
- Integrate Recurrent NJT with Prerequisite State Vector and Recursion Stack
- Enable RPM-based multi-step reasoning via Nested Memory
- Maintain v5 foundation (General-Capable scale, Fractal Attention, MoE Routing)
- Supports Chain-of-Thought reasoning for TKS equations and logic

v6 Updates:
- Replaced flat NJTLayer with TKSNJTv6Block (from tks_features.njt_v6_recurrent)
- Added explicit support for 12 Prerequisites and 4 Gradation stages
- Added Recursion Stack for sub-goal processing
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use components from previous versions
from tks_llm_core import NOETIC_DIM, NUM_WORLDS
from tks_llm_core_v4 import (
    NoeticProcessor,
    AttractorComputationLayer,
    RPMGatingMechanism,
    _build_standardized_trace,
    TRACE_SCHEMA_VERSION
)
from tks_llm_core_v5 import (
    GeneralNoeticEmbedding,
    GeneralNoeticRouter,
    GeneralFractalAttention,
    ROUTING_STABILITY_AVAILABLE,
    StableRouting,
    CANONICAL_DIM
)

# NJT v6 Recurrent Components
from tks_features.njt_v6_recurrent import (
    TKSNJTv6Block,
    NJTv6Config,
    PREREQUISITE_NAMES,
    GRADATION_ABSENT,
    GRADATION_STRONG
)

@dataclass
class TKSGeneralConfig:
    vocab_size: int = 16384
    hidden_dim: int = 384
    num_layers: int = 12
    num_scales: int = 4
    dropout: float = 0.2  # Higher dropout for better generalization
    use_attractor: bool = True
    contraction_factor: float = 0.5
    max_attractor_iter: int = 15
    use_rpm: bool = True
    
    # Operator Core
    use_operator_core: bool = False
    operator_equation_dim: int = 384
    
    # Stable Routing
    use_stable_routing: bool = True
    routing_initial_temp: float = 2.0
    routing_final_temp: float = 0.1
    routing_warmup_steps: int = 1000
    routing_anneal_steps: int = 10000
    routing_noise_type: str = "gumbel"
    routing_initial_noise: float = 0.5
    routing_final_noise: float = 0.0
    routing_balance_weight: float = 0.01
    
    # NJT v6 (Noetic Judgment Transistor) circuits
    use_njt: bool = True  # Enabled by default in v6
    njt_num_transistors: int = 10
    njt_max_iterations: int = 12
    njt_use_nested_memory: bool = True
    njt_max_recursion_depth: int = 4
    njt_initial_gain: float = 2.0
    njt_initial_threshold: float = 0.5
    njt_gate_sharpness: float = 10.0
    njt_sufficiency_threshold: float = 0.7

class GeneralNoeticBlockv6(nn.Module):
    """v6 Block with Recurrent NJT integration."""
    def __init__(self, config: TKSGeneralConfig):
        super().__init__()
        self.dim = config.hidden_dim
        self.config = config
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)

        # Router: MoE selection
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

        # NJT v6 Block: Recurrent Reasoning Engine
        self.njt = None
        if config.use_njt:
            njt_cfg = NJTv6Config(
                dim=config.hidden_dim,
                num_transistors=config.njt_num_transistors,
                max_iterations=config.njt_max_iterations,
                use_nested_memory=config.njt_use_nested_memory,
                max_recursion_depth=config.njt_max_recursion_depth,
                initial_gain=config.njt_initial_gain,
                initial_threshold=config.njt_initial_threshold,
                gate_sharpness=config.njt_gate_sharpness,
                sufficiency_threshold=config.njt_sufficiency_threshold
            )
            self.njt = TKSNJTv6Block(dim=config.hidden_dim, config=njt_cfg)

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
        # 1. Routing
        r_weights, aux_loss, metrics = self.router(self.norm1(x), step=step)

        # 2. NJT v6 Processing (Recursive Reasoning)
        njt_trace = None
        if self.njt is not None:
            # Note: v6 block handles pooling and expansion internally
            x_njt, njt_trace = self.njt(x, return_trace=return_njt_trace)
            x = x + x_njt  # Residual connection

        # 3. FFN
        h = self.ffn(self.norm1(x))
        x = x + h

        # 4. Attention
        attn_out = self.attn(self.norm2(x), attention_mask)
        x = x + attn_out

        return x, r_weights, aux_loss, metrics, njt_trace

class TKSGeneralLMv6(nn.Module):
    """Reasoning-Capable TKS LM with Recurrent NJT v6."""
    def __init__(self, config: TKSGeneralConfig):
        super().__init__()
        self.config = config
        
        self.embedding = GeneralNoeticEmbedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, config.hidden_dim))
        
        self.blocks = nn.ModuleList([
            GeneralNoeticBlockv6(config) for _ in range(config.num_layers)
        ])
        
        # Canonical Projection
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

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
            if metrics is not None:
                routing_metrics = metrics

        x = self.final_norm(x)

        # Canonical Pass
        canon_x = self.to_canonical(x)
        canon_trace = {}
        if self.config.use_attractor:
            att_out = self.attractor(canon_x, return_trajectory=return_full_trace)
            canon_x = att_out["attractor"]
            if return_full_trace:
                canon_trace["attractor"] = att_out

        if self.config.use_rpm:
            rpm_out = self.rpm(canon_x)
            canon_x = rpm_out["gated_output"]
            if return_full_trace:
                canon_trace["rpm"] = rpm_out

        x = x + self.from_canonical(canon_x)
        logits = self.head(x)

        out = {"logits": logits}
        if isinstance(total_aux_loss, torch.Tensor):
            out["routing_aux_loss"] = total_aux_loss
        if routing_metrics:
            out["routing_metrics"] = routing_metrics

        if return_full_trace:
            out["trace"] = {
                "routing": trace_routing,
                "canonical": canon_trace,
            }
            if trace_njt:
                out["trace"]["njt"] = trace_njt

        return out

if __name__ == "__main__":
    # Smoke test
    config = TKSGeneralConfig(num_layers=2)
    model = TKSGeneralLMv6(config)
    input_ids = torch.randint(0, 100, (1, 10))
    output = model(input_ids, return_full_trace=True)
    print(f"v6 Model Output shape: {output['logits'].shape}")
    print(f"Trace available: {'njt' in output['trace']}")
    print("v6 Core successfully initialized.")

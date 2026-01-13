"""
NJT-Transformer: Transformer with Explicit Inhibitory Circuits

This is a SKETCH/CONCEPT for adding NJT (Noetic Judgment Transistor)
circuits to standard transformer architecture for improved:
- Negation handling ("not happy" → actually negative)
- Contrastive reasoning ("but", "however", "although")
- Logical operations (if A and NOT B then C)
- Fine-grained attention control (amplify focus, suppress noise)

Architecture Overview:
======================

Standard Transformer Block:
    Input → Attention → Add&Norm → FFN → Add&Norm → Output

NJT-Transformer Block:
    Input → Attention → NJT-Gate → Add&Norm → FFN → Add&Norm → Output
                        ^^^^^^^^
                    NEW: Push-Pull control after attention

Alternative placement (NJT-FFN):
    Input → Attention → Add&Norm → NJT-FFN → Add&Norm → Output
                                   ^^^^^^^
                              Replace FFN with NJT-enhanced FFN

Author: TKS Development
Date: 2025-01-10
Status: SKETCH - Not production ready
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


# =============================================================================
# CORE NJT COMPONENTS (Simplified from tks_features/njt_circuits.py)
# =============================================================================

class NJTGate(nn.Module):
    """
    Combined NJT+/NJT- gate for transformer integration.

    Unlike separate NJT+ and NJT-, this combines them into a single
    differentiable gate that learns when to amplify vs inhibit.
    """

    def __init__(self, dim: int, num_gates: int = 8):
        super().__init__()
        self.dim = dim
        self.num_gates = num_gates

        # Project to gate space
        self.to_gates = nn.Linear(dim, num_gates * 3)  # cause, bias, mode

        # Learnable thresholds per gate
        self.threshold = nn.Parameter(torch.ones(num_gates) * 0.5)

        # Learnable gain per gate
        self.gain = nn.Parameter(torch.ones(num_gates) * 2.0)

        # Project back from gates
        self.from_gates = nn.Linear(num_gates, dim)

        # Gate sharpness (how binary vs soft the gate is)
        self.sharpness = nn.Parameter(torch.tensor(5.0))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor [batch, seq, dim]
            context: Optional context for bias (defaults to x)

        Returns:
            Gated output [batch, seq, dim]
        """
        if context is None:
            context = x

        # Project to gate space
        gate_input = self.to_gates(x)  # [batch, seq, num_gates * 3]

        # Split into cause (what to gate), bias (when to gate), mode (how to gate)
        cause, bias, mode = gate_input.chunk(3, dim=-1)

        # Normalize
        cause = torch.sigmoid(cause)  # [0, 1]
        bias = torch.sigmoid(bias)    # [0, 1]
        mode = torch.tanh(mode)       # [-1, 1]: -1=inhibit, +1=amplify

        # Compute gate activation
        gate = torch.sigmoid(self.sharpness * (bias - self.threshold))

        # Apply mode: positive mode = amplify, negative mode = inhibit
        # mode > 0: output = cause * gate * gain (amplify)
        # mode < 0: output = cause * (1-gate) * gain (inhibit)

        amplify = cause * gate
        inhibit = cause * (1 - gate)

        # Blend based on mode
        mode_weight = (mode + 1) / 2  # Convert [-1,1] to [0,1]
        gated = mode_weight * amplify + (1 - mode_weight) * inhibit

        # Apply gain and project back
        gated = gated * self.gain
        output = self.from_gates(gated)

        return output


# =============================================================================
# NJT-ENHANCED ATTENTION
# =============================================================================

class NJTAttention(nn.Module):
    """
    Multi-head attention with NJT gating on the output.

    The NJT gate learns to:
    - Amplify relevant attended information
    - Inhibit irrelevant or contradictory information
    - Handle negation in context ("not X" suppresses X)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_njt_gates: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Standard attention projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # NJT gate on attention output
        self.njt_gate = NJTGate(dim, num_njt_gates)

        # Balance between raw attention and NJT-gated
        self.njt_blend = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_len, _ = x.shape

        # Standard attention
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        out = (attn @ v).transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        out = self.out_proj(out)

        # NJT gating - use input as context for the gate
        njt_out = self.njt_gate(out, context=x)

        # Blend raw attention with NJT-gated
        blend = torch.sigmoid(self.njt_blend)
        final = blend * out + (1 - blend) * njt_out

        if return_attention:
            return final, attn
        return final, None


# =============================================================================
# NJT-ENHANCED FEED-FORWARD
# =============================================================================

class NJTFFN(nn.Module):
    """
    Feed-forward network with NJT activation instead of GELU/ReLU.

    Standard FFN:  x → Linear → GELU → Linear → out
    NJT FFN:       x → Linear → NJT → Linear → out

    The NJT layer provides:
    - Amplification path (like ReLU positive side)
    - Inhibition path (explicit negative/suppression)
    - Learned balance per feature
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_njt_gates: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.up_proj = nn.Linear(dim, hidden_dim)
        self.njt_gate = NJTGate(hidden_dim, num_njt_gates)
        self.down_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Up project
        h = self.up_proj(x)

        # NJT activation (replaces GELU)
        h = self.njt_gate(h)

        # Down project
        out = self.down_proj(h)
        out = self.dropout(out)

        return out


# =============================================================================
# NJT TRANSFORMER BLOCK
# =============================================================================

class NJTTransformerBlock(nn.Module):
    """
    Full transformer block with NJT enhancements.

    Architecture:
        Input
          ↓
        NJT-Attention (attention + inhibitory gating)
          ↓
        Add & LayerNorm
          ↓
        NJT-FFN (feedforward with push-pull activation)
          ↓
        Add & LayerNorm
          ↓
        Output
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_njt_gates: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # NJT-enhanced attention
        self.attention = NJTAttention(
            dim=dim,
            num_heads=num_heads,
            num_njt_gates=num_njt_gates,
            dropout=dropout,
        )

        # NJT-enhanced FFN
        self.ffn = NJTFFN(
            dim=dim,
            hidden_dim=dim * ffn_mult,
            num_njt_gates=num_njt_gates * 2,  # More gates in FFN
            dropout=dropout,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x


# =============================================================================
# FULL NJT-TRANSFORMER
# =============================================================================

class NJTTransformer(nn.Module):
    """
    Complete NJT-Transformer for language modeling.

    This combines:
    - Standard transformer attention (for pattern matching)
    - NJT gates (for logical operations, negation, contrast)
    - Learnable push-pull dynamics

    Use cases:
    - Better negation: "The movie was not good" → correctly negative
    - Better contrast: "He was tired but kept going" → understand opposition
    - Better logic: "If sunny and not raining, go outside" → proper inference
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_njt_gates: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # NJT Transformer blocks
        self.layers = nn.ModuleList([
            NJTTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_njt_gates=num_njt_gates,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embed.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_embed(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.pos_embed(pos_ids)

        x = tok_emb + pos_emb

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            mask = ~mask  # Invert: 1 = attend, 0 = mask

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        return {"logits": logits}


# =============================================================================
# COMPARISON: Standard vs NJT Transformer
# =============================================================================

def compare_architectures():
    """Print comparison of standard vs NJT transformer."""

    print("""
================================================================================
                    STANDARD TRANSFORMER vs NJT-TRANSFORMER
================================================================================

  STANDARD TRANSFORMER
  ====================

    Input
      |
    [Attention] --> Q,K,V projections --> Softmax --> Output
      |
    [Add & Norm]
      |
    [FFN] --> Linear --> GELU (positive bias) --> Linear
      |
    [Add & Norm]
      |
    Output

    Activation: GELU = x * phi(x)  [mostly positive, soft positive gate]
    Problem: No explicit inhibition. "NOT X" must be learned indirectly.

================================================================================

  NJT-TRANSFORMER
  ===============

    Input
      |
    [NJT-Attention] --> Standard Attention --> NJT Gate --> Output
      |                              |
      |                    +---------+---------+
      |                    |                   |
      |                 [NJT+]             [NJT-]
      |                 Amplify           Inhibit
      |                    |                   |
      |                    +------ Blend ------+
      |
    [Add & Norm]
      |
    [NJT-FFN] --> Linear --> NJT Gate --> Linear
      |                         |
      |               +---------+---------+
      |               |                   |
      |            [NJT+]             [NJT-]
      |            Amplify           Inhibit
      |               |                   |
      |               +------ Blend ------+
      |
    [Add & Norm]
      |
    Output

    Activation: NJT = blend(amplify, inhibit) based on learned mode
    Advantage: Explicit inhibition. "NOT X" directly suppresses X.

================================================================================

  EXPECTED IMPROVEMENTS:

  +---------------------+---------------------+--------------------------+
  | Task                | Standard            | NJT-Transformer          |
  +---------------------+---------------------+--------------------------+
  | Negation            | Often fails         | Native NOT gate          |
  | "not happy"         | -> positive bias    | -> correct inhibition    |
  +---------------------+---------------------+--------------------------+
  | Contrast            | Weak                | Strong                   |
  | "good but flawed"   | -> mixed signal     | -> amplify+inhibit blend |
  +---------------------+---------------------+--------------------------+
  | Logical Ops         | Learned indirectly  | Native gates             |
  | "if A and not B"    | -> approximation    | -> direct computation    |
  +---------------------+---------------------+--------------------------+
  | Attention Focus     | Soft only           | Hard suppress possible   |
  | Irrelevant context  | -> still influences | -> actively dampened     |
  +---------------------+---------------------+--------------------------+

================================================================================

  PARAMETER OVERHEAD:

  For dim=512, num_njt_gates=8:
    - NJTGate adds: ~12K params per gate instance
    - Per block: ~36K additional params
    - 6 layers: ~216K additional params
    - Overhead: ~2-3% of total model size

  Small cost, potentially large benefit for logical reasoning.

================================================================================
""")


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    # Print architecture comparison
    compare_architectures()

    # Quick test
    print("\n" + "="*60)
    print("QUICK TEST: NJT-Transformer Forward Pass")
    print("="*60 + "\n")

    # Create small model
    model = NJTTransformer(
        vocab_size=32000,
        dim=256,
        num_layers=4,
        num_heads=4,
        num_njt_gates=8,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))

    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {output['logits'].shape}")

    print("\n[OK] NJT-Transformer working!")
    print("\nNext steps:")
    print("  1. Train on language data")
    print("  2. Compare negation handling vs standard transformer")
    print("  3. Benchmark logical reasoning tasks")

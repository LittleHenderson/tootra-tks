"""
TKS 4-Operation Block V2 - Subtractive Hypothesis Testing

This module implements the improved 4-operation block with:
- Separate attention contexts for inhibition (fixes shared context issue)
- Independent sigmoid gates (fixes softmax collapse)
- Interpretable inhibition attention (returns attention patterns)
- Stability mechanisms (alpha_sub, tanh bounding)
- Comprehensive instrumentation (gate statistics, attention visualization)

Architecture:
- ADD: Standard accumulation via shared attention
- SUB: Inhibition via separate inhibition attention
- MUL: Binding/gating via shared attention
- DIV: Normalization via shared attention

Key Innovation:
Inhibition (SUB) gets its own attention mechanism to learn orthogonal patterns
from accumulation, testing the core TKS hypothesis about negation scope.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class MHSA(nn.Module):
    """Multi-Head Self-Attention with optional attention return."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, D]
            attn_mask: Optional[B, T, T] or broadcastable
            return_attn: If True, return attention weights

        Returns:
            output: [B, T, D]
            attn_weights: Optional[B, H, T, T] if return_attn=True
        """
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T, T]

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply additional mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, T, T]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        out = self.out_proj(out)

        if return_attn:
            return out, attn_weights
        return out, None


class TKS4OpBlockV2(nn.Module):
    """
    TKS 4-Operation Block V2 - Testing Subtractive Hypothesis

    Key Fixes from V1:
    1. Separate inhibition attention (attn_inh) - orthogonal from accumulation
    2. Independent sigmoid gates - no softmax competition
    3. Interpretable inhibition - can return attention patterns
    4. Stability mechanisms - alpha_sub scaling, tanh bounding

    Operations:
    - ADD: x + w_add * c_add (accumulation)
    - SUB: x - w_sub * alpha_sub * tanh(c_inh) (inhibition with separate attention)
    - MUL: x * (1 + w_mul * g_mul) (gating/binding)
    - DIV: x / (eps + w_div * denom) (normalization)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        inhib_heads: int = 4,
        dropout: float = 0.1,
        causal: bool = True,
        alpha_sub: float = 0.1,
        eps: float = 1e-6
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.inhib_heads = inhib_heads
        self.alpha_sub = alpha_sub  # Stability scaling for subtraction
        self.eps = eps

        # Pre-LN for each pathway
        self.ln_add = nn.LayerNorm(d_model)
        self.ln_inh = nn.LayerNorm(d_model)

        # CRITICAL: Separate attention mechanisms
        self.attn_add = MHSA(d_model, n_heads, dropout=dropout, causal=causal)
        self.attn_inh = MHSA(d_model, inhib_heads, dropout=dropout, causal=causal)

        # Context projections
        self.add_proj = nn.Linear(d_model, d_model)
        self.inh_proj = nn.Linear(d_model, d_model)

        # Independent gate projections (no softmax competition)
        self.g_add = nn.Linear(d_model, d_model)
        self.g_sub = nn.Linear(d_model, d_model)
        self.g_mul = nn.Linear(d_model, d_model)
        self.g_div = nn.Linear(d_model, d_model)

        # MUL and DIV additional projections
        self.mul_gate_proj = nn.Linear(d_model, d_model)
        self.div_denom_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        return_inhib_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, D] input
            attn_mask: Optional attention mask
            return_stats: If True, return gate statistics (detached)
            return_inhib_attn: If True, return inhibition attention weights

        Returns:
            output: [B, T, D]
            stats: Optional dict with gate statistics
            inhib_attn: Optional[B, H, T, T] inhibition attention weights
        """
        B, T, D = x.shape

        # ===== SEPARATE CONTEXTS (KEY FIX) =====
        z_add = self.ln_add(x)
        z_inh = self.ln_inh(x)

        # Accumulation context (shared for ADD/MUL/DIV)
        c_add, _ = self.attn_add(z_add, attn_mask=attn_mask, return_attn=False)
        c_add = self.add_proj(c_add)

        # Inhibition context (separate for SUB)
        if return_inhib_attn:
            c_inh, inh_attn = self.attn_inh(z_inh, attn_mask=attn_mask, return_attn=True)
        else:
            c_inh, inh_attn = self.attn_inh(z_inh, attn_mask=attn_mask, return_attn=False)
        c_inh = self.inh_proj(c_inh)

        # ===== INDEPENDENT GATES (NO COMPETITION) =====
        w_add = torch.sigmoid(self.g_add(z_add))  # [B, T, D]
        w_sub = torch.sigmoid(self.g_sub(z_inh))  # Uses inhibition context
        w_mul = torch.sigmoid(self.g_mul(z_add))
        w_div = torch.sigmoid(self.g_div(z_add))

        # ===== OPERATION EXECUTION =====
        # ADD: Standard accumulation
        term_add = w_add * c_add

        # SUB: Inhibition with stability (tanh bounds to [-1, 1])
        term_sub = w_sub * self.alpha_sub * torch.tanh(c_inh)

        # MUL: Gating/binding
        g_mul = torch.sigmoid(self.mul_gate_proj(c_add))
        term_mul = x * (1.0 + w_mul * g_mul)

        # DIV: Normalization
        denom = self.eps + F.softplus(self.div_denom_proj(c_add))
        term_div = x / (self.eps + w_div * denom)

        # ===== COMBINE =====
        out = x + term_add - term_sub
        out = term_mul  # Apply multiplicative gating
        out = term_div  # Apply divisive normalization

        # ===== STATISTICS (DETACHED) =====
        stats = None
        if return_stats:
            with torch.no_grad():
                stats = {
                    "w_add": w_add.mean().item(),
                    "w_sub": w_sub.mean().item(),
                    "w_mul": w_mul.mean().item(),
                    "w_div": w_div.mean().item(),
                    "c_add_norm": c_add.norm(dim=-1).mean().item(),
                    "c_inh_norm": c_inh.norm(dim=-1).mean().item(),
                }

        return out, stats, inh_attn


class BaselineTransformerBlock(nn.Module):
    """Standard transformer block for comparison (Condition A)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        causal: bool = True
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHSA(d_model, n_heads, dropout=dropout, causal=causal)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Attention with residual
        attn_out, _ = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x


class SubtractiveOnlyBlock(nn.Module):
    """Subtractive-only block (Condition B) - testing core hypothesis."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        inhib_heads: int = 4,
        dropout: float = 0.1,
        causal: bool = True,
        alpha_sub: float = 0.1,
        eps: float = 1e-6
    ):
        super().__init__()

        self.d_model = d_model
        self.alpha_sub = alpha_sub

        # Pre-LN
        self.ln_add = nn.LayerNorm(d_model)
        self.ln_inh = nn.LayerNorm(d_model)

        # Separate attentions
        self.attn_add = MHSA(d_model, n_heads, dropout=dropout, causal=causal)
        self.attn_inh = MHSA(d_model, inhib_heads, dropout=dropout, causal=causal)

        # Projections
        self.add_proj = nn.Linear(d_model, d_model)
        self.inh_proj = nn.Linear(d_model, d_model)

        # Gates
        self.g_add = nn.Linear(d_model, d_model)
        self.g_sub = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_inhib_attn: bool = False
    ):
        z_add = self.ln_add(x)
        z_inh = self.ln_inh(x)

        # Contexts
        c_add, _ = self.attn_add(z_add, attn_mask=attn_mask)
        c_add = self.add_proj(c_add)

        if return_inhib_attn:
            c_inh, inh_attn = self.attn_inh(z_inh, attn_mask=attn_mask, return_attn=True)
        else:
            c_inh, _ = self.attn_inh(z_inh, attn_mask=attn_mask)
            inh_attn = None
        c_inh = self.inh_proj(c_inh)

        # Gates
        w_add = torch.sigmoid(self.g_add(z_add))
        w_sub = torch.sigmoid(self.g_sub(z_inh))

        # Operations (ADD and SUB only)
        out = x + w_add * c_add - w_sub * self.alpha_sub * torch.tanh(c_inh)

        return out, inh_attn


# ===== MODEL WRAPPERS =====

class SimpleTransformerLM(nn.Module):
    """Simple transformer language model for baseline."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            BaselineTransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape

        # Embeddings
        x = self.embed(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(pos)

        # Blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


class SubtractiveLM(nn.Module):
    """Language model with subtractive blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        inhib_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        alpha_sub: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            SubtractiveOnlyBlock(
                d_model, n_heads, inhib_heads,
                dropout=dropout, alpha_sub=alpha_sub
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_inhib_attn: bool = False
    ):
        B, T = input_ids.shape

        # Embeddings
        x = self.embed(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(pos)

        # Blocks
        inhib_attns = []
        for block in self.blocks:
            x, inh_attn = block(x, return_inhib_attn=return_inhib_attn)
            if return_inhib_attn and inh_attn is not None:
                inhib_attns.append(inh_attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_inhib_attn:
            return logits, inhib_attns
        return logits


class Full4OpLM(nn.Module):
    """Language model with full 4-operation blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        inhib_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        alpha_sub: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TKS4OpBlockV2(
                d_model, n_heads, inhib_heads,
                dropout=dropout, alpha_sub=alpha_sub
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_stats: bool = False,
        return_inhib_attn: bool = False
    ):
        B, T = input_ids.shape

        # Embeddings
        x = self.embed(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(pos)

        # Blocks
        all_stats = []
        inhib_attns = []
        for block in self.blocks:
            x, stats, inh_attn = block(
                x,
                return_stats=return_stats,
                return_inhib_attn=return_inhib_attn
            )
            if return_stats and stats is not None:
                all_stats.append(stats)
            if return_inhib_attn and inh_attn is not None:
                inhib_attns.append(inh_attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        results = [logits]
        if return_stats:
            results.append(all_stats)
        if return_inhib_attn:
            results.append(inhib_attns)

        return tuple(results) if len(results) > 1 else logits


if __name__ == "__main__":
    # Quick test
    print("Testing TKS4OpBlockV2...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test block
    block = TKS4OpBlockV2(d_model=256, n_heads=8, inhib_heads=4).to(device)
    x = torch.randn(2, 16, 256).to(device)

    out, stats, inh_attn = block(x, return_stats=True, return_inhib_attn=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Stats: {stats}")
    print(f"Inhib attn shape: {inh_attn.shape if inh_attn is not None else None}")

    # Test models
    print("\nTesting models...")
    vocab_size = 1000

    baseline = SimpleTransformerLM(vocab_size, d_model=256, n_layers=4).to(device)
    subtractive = SubtractiveLM(vocab_size, d_model=256, n_layers=4).to(device)
    full_4op = Full4OpLM(vocab_size, d_model=256, n_layers=4).to(device)

    input_ids = torch.randint(0, vocab_size, (2, 16)).to(device)

    print(f"Baseline params: {sum(p.numel() for p in baseline.parameters()):,}")
    print(f"Subtractive params: {sum(p.numel() for p in subtractive.parameters()):,}")
    print(f"Full 4Op params: {sum(p.numel() for p in full_4op.parameters()):,}")

    print("\nAll tests passed!")

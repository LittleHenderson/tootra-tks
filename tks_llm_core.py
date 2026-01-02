"""
TKS-LLM Core Components — Minimal PyTorch Prototype

This module implements the foundational neural components for TKS-LLM,
a novel architecture based on the TOOTRA Kabbalistic System.

Components:
    1. NoeticEmbeddingLayer: Maps tokens to 40-dim noetic space
    2. NoeticProcessor: Applies one of 10 noetic transforms (ν₀–ν₉)
    3. FractalAttentionMechanism: Multi-scale self-attention mixing

Canonical References:
    - TKS_Symbol_Sense_Table_v1.0.md (40-element definitions)
    - TKS_LLM_Canonical_Validation_v1.0.md (validated mapping)

40-Dimensional Noetic Space Structure:
    The 40 dimensions correspond to the 40 TKS Elements (10 Noetics × 4 Worlds):

    World A (Spiritual): indices 0-9   → Elements A0-A9
    World B (Mental):    indices 10-19 → Elements B0-B9
    World C (Emotional): indices 20-29 → Elements C0-C9
    World D (Physical):  indices 30-39 → Elements D0-D9

    Index formula: index(Xn) = world_offset(X) + n
    where world_offset = {A:0, B:10, C:20, D:30}

The 10 Noetics (per world):
    ν₀ IDEA      - Pure potential, unity
    ν₁ MIND      - Consciousness, awareness
    ν₂ POSITIVE  - Attraction, expansion
    ν₃ NEGATIVE  - Repulsion, contraction
    ν₄ VIBRATION - Intensity, oscillation
    ν₅ FEMALE    - Receptivity, integration
    ν₆ MALE      - Projection, differentiation
    ν₇ RHYTHM    - Cycles, periodicity
    ν₈ CAUSE     - Origin, above
    ν₉ EFFECT    - Result, below

Author: TKS-LLM Development Team
Date: 2025-12-11
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from tks_rules.noetics import NOETIC_NAMES
from tks_rules.worlds import WORLD_OFFSETS
NOETIC_DIM = 10
NUM_WORLDS = 4
TOTAL_DIM = 40

class NoeticEmbeddingLayer(nn.Module):
    """
    Maps token indices to 40-dimensional noetic embedding space.

    The 40-dim output represents the 40 TKS Elements structured as:
        - A-world (Spiritual): dims 0–9
        - B-world (Mental):    dims 10–19
        - C-world (Emotional): dims 20–29
        - D-world (Physical):  dims 30–39

    Each world slice undergoes a learnable linear transform initialized
    near-identity to preserve structure during early training.

    Args:
        vocab_size: Number of tokens in vocabulary
        hidden_dim: Intermediate embedding dimension before noetic projection

    References:
        TKS_LLM_Canonical_Validation_v1.0.md Section 8.2
        TKS_Symbol_Sense_Table_v1.0.md
    """

    def __init__(self, vocab_size: int=1000, hidden_dim: int=128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.noetic_projection = nn.Linear(hidden_dim, TOTAL_DIM)
        self.world_A = nn.Linear(NOETIC_DIM, NOETIC_DIM)
        self.world_B = nn.Linear(NOETIC_DIM, NOETIC_DIM)
        self.world_C = nn.Linear(NOETIC_DIM, NOETIC_DIM)
        self.world_D = nn.Linear(NOETIC_DIM, NOETIC_DIM)
        for layer in [self.world_A, self.world_B, self.world_C, self.world_D]:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: torch.LongTensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: tokens → noetic embeddings.

        Args:
            tokens: [batch, seq] token indices

        Returns:
            Dict with keys:
                'noetic': [batch, seq, 40] full noetic embedding
                'A': [batch, seq, 10] Spiritual world
                'B': [batch, seq, 10] Mental world
                'C': [batch, seq, 10] Emotional world
                'D': [batch, seq, 10] Physical world
        """
        h = self.token_embedding(tokens)
        noetic = self.noetic_projection(h)
        A = noetic[..., 0:10]
        B = noetic[..., 10:20]
        C = noetic[..., 20:30]
        D = noetic[..., 30:40]
        A = self.world_A(A)
        B = self.world_B(B)
        C = self.world_C(C)
        D = self.world_D(D)
        noetic_out = torch.cat([A, B, C, D], dim=-1)
        return {'noetic': noetic_out, 'A': A, 'B': B, 'C': C, 'D': D}

class NoeticProcessor(nn.Module):
    """
    Applies one of 10 noetic transforms (ν₀ through ν₉) to input vectors.

    Each noetic is implemented as a learnable linear layer + nonlinearity.
    The transforms are initialized as follows:
        - ν₀ (IDEA): Near-identity (preserves input)
        - ν₁–ν₉: Small random perturbations from identity

    This minimal version applies a single noetic per forward pass.
    Future versions will support weighted combinations and enforce
    involution constraints (e.g., ν₂ ∘ ν₃ ≈ I).

    Args:
        dim: Input/output dimension (default 40 for full noetic space)

    References:
        TKS_LLM_Canonical_Validation_v1.0.md Section 5
    """

    def __init__(self, dim: int=40) -> None:
        super().__init__()
        self.dim = dim
        self.noetics = nn.ModuleList([nn.Linear(dim, dim) for _ in range(10)])
        self._init_noetics()

    def _init_noetics(self) -> None:
        """Initialize noetic matrices with appropriate structure."""
        for k, layer in enumerate(self.noetics):
            if k == 0:
                nn.init.eye_(layer.weight)
                layer.weight.data *= 0.99
                layer.weight.data += 0.01 * torch.randn_like(layer.weight)
            else:
                nn.init.eye_(layer.weight)
                layer.weight.data += 0.1 * torch.randn_like(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, noetic_idx: int) -> torch.Tensor:
        """
        Apply a single noetic transform.

        Args:
            x: [batch, seq, dim] input tensor
            noetic_idx: Which noetic to apply (0–9)

        Returns:
            [batch, seq, dim] transformed tensor
        """
        if not 0 <= noetic_idx <= 9:
            raise ValueError(f'noetic_idx must be 0–9, got {noetic_idx}')
        out = self.noetics[noetic_idx](x)
        out = F.gelu(out)
        return out

    def get_noetic_name(self, idx: int) -> str:
        """Get canonical name for noetic index."""
        return NOETIC_NAMES[idx]

class FractalAttentionMechanism(nn.Module):
    """
    Multi-scale self-attention mixing inspired by fractal self-similarity.

    Operates at multiple scales by downsampling the sequence, then
    computes cross-scale similarities to mix information back to
    the original resolution.

    Process:
        1. Create downsampled versions of input at each scale
        2. Project each scale to the same dimension
        3. Compute similarity between base sequence and each scale
        4. Softmax over scales to get mixing weights
        5. Blend scales back to output

    This is a minimal prototype; future versions will incorporate
    learnable fractal dimension and TKS-specific scale relationships.

    Args:
        dim: Input/output dimension
        num_scales: Number of resolution scales (including base)

    References:
        TKS_LLM_Architecture_v1.0.md Section on Fractal Attention
    """

    def __init__(self, dim: int=40, num_scales: int=3) -> None:
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.scale_projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_scales)])
        self.query_proj = nn.Linear(dim, dim)
        self.key_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_scales)])
        self.output_proj = nn.Linear(dim, dim)

    def _downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsample sequence by averaging adjacent tokens.

        Args:
            x: [batch, seq, dim]
            factor: Downsampling factor (1 = no change)

        Returns:
            [batch, seq // factor, dim] (or [batch, 1, dim] if factor > seq)
        """
        if factor == 1:
            return x
        batch, seq, dim = x.shape
        new_seq = max(1, seq // factor)
        x_t = x.transpose(1, 2)
        x_pooled = F.adaptive_avg_pool1d(x_t, new_seq)
        return x_pooled.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fractal attention mixing.

        Args:
            x: [batch, seq, dim] input tensor

        Returns:
            [batch, seq, dim] output tensor with multi-scale information
        """
        batch, seq, dim = x.shape
        scales: List[torch.Tensor] = []
        for s in range(self.num_scales):
            factor = 2 ** s
            x_scaled = self._downsample(x, factor)
            x_proj = self.scale_projections[s](x_scaled)
            scales.append(x_proj)
        query = self.query_proj(x)
        scale_outputs: List[torch.Tensor] = []
        scale_weights: List[torch.Tensor] = []
        for s, x_scale in enumerate(scales):
            keys = self.key_projs[s](x_scale)
            attn_scores = torch.matmul(query, keys.transpose(-2, -1))
            attn_scores = attn_scores / dim ** 0.5
            attn_weights = F.softmax(attn_scores, dim=-1)
            attended = torch.matmul(attn_weights, x_scale)
            scale_outputs.append(attended)
            mean_attn = attn_weights.mean(dim=(1, 2), keepdim=True)
            scale_weights.append(mean_attn)
        scale_weights_tensor = torch.cat(scale_weights, dim=-1)
        mix_weights = F.softmax(scale_weights_tensor, dim=-1)
        output = torch.zeros_like(x)
        for s, scale_out in enumerate(scale_outputs):
            w = mix_weights[..., s:s + 1]
            output = output + w * scale_out
        output = self.output_proj(output)
        return output

def run_noetic_pipeline_example():
    """
    High-level helper: given a sequence length, run the 3-layer TKS core
    (Embedding → NoeticProcessor → FractalAttention) and return the tensors.
    This is what other code (or Claude) can import and call.
    """
    batch_size = 1
    seq_len = 8
    tokens = torch.randint(0, 100, (batch_size, seq_len))
    embedding_layer = NoeticEmbeddingLayer(vocab_size=100)
    processor = NoeticProcessor()
    fractal = FractalAttentionMechanism(dim=40)
    emb = embedding_layer(tokens)['noetic']
    proc = processor(emb, noetic_idx=1)
    frac = fractal(proc)
    return {'tokens': tokens, 'noetic': emb, 'processed': proc, 'fractal': frac}
if __name__ == '__main__':
    out = run_noetic_pipeline_example()
    print('=== TKS core pipeline demo ===')
    for name, tensor in out.items():
        print(f'{name:9s}', tensor.shape)
"""
TKS Noetic Fractal Coherence System

A comprehensive coherence analysis and enforcement module that uses lacunary series
mathematics and fractal dimension analysis to detect and prevent incoherent outputs.

Core Principle:
    If the model's internal representations can be mapped to valid Noetic Fractals,
    the output is semantically coherent. Gibberish fails to map to NF space.

Key Components:
    1. LacunaryCoherenceAnalyzer - Detects gibberish via lacunary gap analysis
    2. FractalDimensionTracker - Monitors attention coherence via Hausdorff dimension
    3. NoeticFractalEncoder - Maps phrase embeddings to NF space (X:Y:Z notation)
    4. CoherenceGate - Blocks incoherent representations before output
    5. NoeticFractalDecoder - Projects NF coordinates back to language embedding space

Mathematical Foundations:
    - Lacunary Series: sum(a_n * z^(lambda^n)) where lambda > 1
        * Rapid gap growth in embeddings indicates incoherent token transitions
        * Used to compute "coherence index" of token sequences

    - Fractal Dimension: D = lim(log(N(epsilon)) / log(1/epsilon)) as epsilon -> 0
        * Attention patterns have characteristic dimensions for coherent vs gibberish
        * Sudden dimension collapse indicates hallucination/confabulation

    - Noetic Fractals: X:Y:Z = nested consciousness operators
        * X = outer noetic (0-9), Y = inner noetic, Z = innermost
        * Valid NFs have semantic meaning; invalid combinations = gibberish
        * 1000 possible 3-deep fractals (10^3), but only ~700 are semantically coherent

Coherence Flow:
    Input tokens -> Embeddings -> LacunaryAnalyzer -> NF Encoder -> CoherenceGate
                                                                         |
    Output <- NF Decoder <- Constrained Generation <--- BLOCKED if incoherent

Author: TKS Research Pipeline
Date: 2026-01-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math
import json
from pathlib import Path


# =============================================================================
# CONSTANTS AND NOETIC DEFINITIONS
# =============================================================================

# The 10 Noetic operators (0-9) with their core meanings
NOETIC_MEANINGS = {
    0: "Idea",       # Pure potential, identity, the content itself
    1: "Mind",       # Awareness, attention, consciousness
    2: "Positive",   # Attraction, approach, desire
    3: "Negative",   # Aversion, rejection, withdrawal
    4: "Vibration",  # Intensity, resonance, depth
    5: "Female",     # Receptive, pattern, belief, conditioning
    6: "Male",       # Projective, structure, composition
    7: "Rhythm",     # Timing, cycle, schedule, repetition
    8: "Above",      # Cause, trigger, source, inner
    9: "Below",      # Effect, result, outcome, outer
}

# Dual pairs: i + j = 9 (involution)
NOETIC_DUAL_PAIRS = [(0, 9), (1, 8), (2, 7), (3, 6), (4, 5)]

# Self-dual noetics (MVR - Mind, Vibration, Rhythm)
SELF_DUAL_NOETICS = {1, 4, 7}

# Incoherence patterns: these NF combinations are semantically unstable
# (based on TKS theory - contradictory nestings)
INCOHERENT_PATTERNS = {
    (2, 3, 2),  # Attraction within Aversion within Attraction = oscillation trap
    (3, 2, 3),  # Aversion within Attraction within Aversion = oscillation trap
    (8, 9, 8),  # Cause within Effect within Cause = causal loop
    (9, 8, 9),  # Effect within Cause within Effect = causal loop
    (0, 0, 0),  # Pure identity nesting = semantic void
}

# Coherent anchor patterns (strong semantic grounding)
COHERENT_ANCHORS = {
    (1, 4, 7): "MVR: Mind-Vibration-Rhythm - the self-regulating triad",
    (5, 1, 3): "Self-concept blind spots - therapeutically valid",
    (8, 5, 9): "Cause of conditioning leading to effect - narrative structure",
    (2, 6, 1): "Attraction to structure containing awareness - learning pattern",
}


@dataclass
class CoherenceScore:
    """Comprehensive coherence assessment."""
    lacunary_index: float        # 0=continuous, 1=maximally sparse
    fractal_dimension: float     # Expected ~1.5 for coherent, <0.5 = collapse
    nf_validity: float          # 0=invalid NF, 1=valid NF mapping
    semantic_stability: float    # 0=unstable, 1=stable
    overall: float              # Weighted combination

    # Diagnostic info
    detected_nf: Optional[Tuple[int, int, int]] = None
    collapse_detected: bool = False
    gibberish_detected: bool = False

    def is_coherent(self, threshold: float = 0.5) -> bool:
        """Check if score indicates coherent output."""
        return self.overall >= threshold and not self.gibberish_detected


# =============================================================================
# LACUNARY COHERENCE ANALYZER
# =============================================================================

class LacunaryCoherenceAnalyzer(nn.Module):
    """
    Analyzes token sequences for coherence using lacunary series properties.

    Lacunary Series: sum(a_n * z^(lambda^n)) where lambda > 1

    Key Insight: Coherent text has smooth semantic transitions. Gibberish has
    "lacunary gaps" - sudden jumps in embedding space that indicate broken
    semantic chains.

    Algorithm:
        1. Compute embedding differences between consecutive tokens
        2. Measure gap growth rate (ratio of successive differences)
        3. Lacunary index = mean growth rate - 1 (clamped to [0, 1])
        4. High index (>0.5) indicates incoherent transitions
    """

    def __init__(
        self,
        embed_dim: int = 384,
        noetic_dim: int = 40,
        gap_threshold: float = 2.0,    # Gap ratio threshold for lacunary detection
        window_size: int = 5,          # Sliding window for gap analysis
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.noetic_dim = noetic_dim
        self.gap_threshold = gap_threshold
        self.window_size = window_size

        # Learned embedding-to-semantic projection
        self.semantic_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, noetic_dim),
        )

        # Gap significance predictor (learns which gaps matter)
        self.gap_importance = nn.Sequential(
            nn.Linear(noetic_dim * 2, noetic_dim),
            nn.GELU(),
            nn.Linear(noetic_dim, 1),
            nn.Sigmoid(),
        )

        # Lacunary pattern detector (learned)
        self.lacunary_detector = nn.Sequential(
            nn.Linear(window_size, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def compute_semantic_gaps(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic gaps between consecutive embeddings.

        Args:
            embeddings: [batch, seq, embed_dim]

        Returns:
            gaps: [batch, seq-1] - magnitude of semantic jumps
        """
        # Project to semantic space
        semantic = self.semantic_proj(embeddings)  # [batch, seq, noetic_dim]

        # Compute differences
        diffs = semantic[:, 1:] - semantic[:, :-1]  # [batch, seq-1, noetic_dim]

        # Weighted gap magnitude (learned importance)
        pairs = torch.cat([semantic[:, :-1], semantic[:, 1:]], dim=-1)
        importance = self.gap_importance(pairs).squeeze(-1)  # [batch, seq-1]

        # Gap magnitudes weighted by importance
        gaps = diffs.norm(dim=-1) * importance  # [batch, seq-1]

        return gaps

    def compute_gap_ratios(self, gaps: torch.Tensor) -> torch.Tensor:
        """
        Compute growth ratios between consecutive gaps.

        Lacunary series have rapidly growing gaps (ratio >> 1).
        Coherent text has stable gaps (ratio ~ 1).

        Args:
            gaps: [batch, seq-1]

        Returns:
            ratios: [batch, seq-2]
        """
        # Avoid division by zero
        gaps_safe = gaps.clamp(min=1e-8)
        ratios = gaps_safe[:, 1:] / gaps_safe[:, :-1]
        return ratios

    def compute_lacunary_index(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute lacunary index indicating coherence level.

        Args:
            embeddings: [batch, seq, embed_dim]

        Returns:
            index: [batch] - 0=coherent, 1=incoherent
            diagnostics: Dict with gap analysis details
        """
        batch, seq, _ = embeddings.shape

        if seq < 3:
            # Too short to analyze
            return torch.zeros(batch, device=embeddings.device), {}

        # Compute gaps and ratios
        gaps = self.compute_semantic_gaps(embeddings)  # [batch, seq-1]
        ratios = self.compute_gap_ratios(gaps)  # [batch, seq-2]

        # Sliding window lacunary detection
        if ratios.shape[1] >= self.window_size:
            # Create windows
            windows = ratios.unfold(1, self.window_size, 1)  # [batch, num_windows, window_size]
            lacunary_scores = self.lacunary_detector(windows).squeeze(-1)  # [batch, num_windows]
            index = lacunary_scores.max(dim=-1).values  # [batch] - max lacunarity in sequence
        else:
            # Short sequence: use simple ratio-based detection
            high_ratios = (ratios > self.gap_threshold).float()
            index = high_ratios.mean(dim=-1)

        diagnostics = {
            'mean_gap': gaps.mean(dim=-1),
            'max_gap': gaps.max(dim=-1).values,
            'mean_ratio': ratios.mean(dim=-1),
            'max_ratio': ratios.max(dim=-1).values,
            'high_ratio_count': (ratios > self.gap_threshold).sum(dim=-1),
        }

        return index, diagnostics

    def forward(
        self,
        embeddings: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Analyze embeddings for lacunary (incoherent) patterns.

        Args:
            embeddings: [batch, seq, embed_dim]
            return_diagnostics: Whether to return detailed analysis

        Returns:
            lacunary_index: [batch] - coherence score (0=coherent, 1=gibberish)
            diagnostics: Optional dict with analysis details
        """
        index, diag = self.compute_lacunary_index(embeddings)

        if return_diagnostics:
            return index, diag
        return index, None


# =============================================================================
# FRACTAL DIMENSION TRACKER
# =============================================================================

class FractalDimensionTracker(nn.Module):
    """
    Monitors attention pattern complexity using fractal dimension estimation.

    Coherent attention has characteristic fractal dimension (~1.2-1.8).
    Collapsing dimension (<0.5) indicates hallucination/mode collapse.
    Exploding dimension (>2.5) indicates chaotic/random attention.

    Method: Box-counting dimension estimation on attention matrices.
        D = lim(log(N(epsilon)) / log(1/epsilon)) as epsilon -> 0

    Where N(epsilon) = number of boxes of size epsilon needed to cover pattern.
    """

    def __init__(
        self,
        num_scales: int = 5,
        min_dimension: float = 0.5,
        max_dimension: float = 2.5,
        target_dimension: float = 1.5,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.target_dimension = target_dimension

        # Learnable scale weights for dimension estimation
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # Dimension classifier (maps estimated D to coherence)
        self.dimension_coherence = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def box_count(self, attention: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Count number of non-empty boxes at given scale.

        Args:
            attention: [batch, seq, seq] attention weights
            scale: Box size (power of 2)

        Returns:
            count: [batch] number of occupied boxes
        """
        batch, seq, _ = attention.shape

        # Threshold attention (binarize)
        binary = (attention > 0.1).float()

        # Downsample by max-pooling
        if scale > 1 and seq >= scale:
            pooled = F.max_pool2d(
                binary.unsqueeze(1),  # Add channel dim
                kernel_size=scale,
                stride=scale,
            ).squeeze(1)
        else:
            pooled = binary

        # Count non-zero boxes
        count = (pooled > 0).float().sum(dim=(1, 2))

        return count

    def estimate_dimension(self, attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate fractal dimension using box-counting.

        Args:
            attention: [batch, seq, seq]

        Returns:
            dimension: [batch] estimated fractal dimension
            r_squared: [batch] fit quality (higher = more reliable)
        """
        batch, seq, _ = attention.shape
        device = attention.device

        # Scales: 1, 2, 4, 8, 16...
        scales = [2 ** i for i in range(self.num_scales) if 2 ** i < seq]
        if len(scales) < 2:
            # Too small for reliable estimation
            return (
                torch.full((batch,), self.target_dimension, device=device),
                torch.zeros(batch, device=device)
            )

        # Compute box counts at each scale
        log_counts = []
        log_scales = []

        for scale in scales:
            count = self.box_count(attention, scale)
            # Avoid log(0)
            log_counts.append(torch.log(count.clamp(min=1)))
            log_scales.append(math.log(1 / scale))

        log_counts = torch.stack(log_counts, dim=1)  # [batch, num_scales]
        log_scales = torch.tensor(log_scales, device=device)  # [num_scales]

        # Linear regression: log(N) = D * log(1/epsilon) + c
        # Using weighted least squares
        weights = F.softmax(self.scale_weights[:len(scales)], dim=0)

        # Weighted means
        mean_x = (log_scales * weights).sum()
        mean_y = (log_counts * weights.unsqueeze(0)).sum(dim=1)

        # Weighted covariance
        dx = log_scales - mean_x
        dy = log_counts - mean_y.unsqueeze(1)

        cov_xy = (dx * dy * weights.unsqueeze(0)).sum(dim=1)
        var_x = (dx ** 2 * weights).sum()

        # Slope = dimension
        dimension = cov_xy / (var_x + 1e-8)

        # R-squared for fit quality
        y_pred = dimension.unsqueeze(1) * log_scales + (mean_y - dimension * mean_x).unsqueeze(1)
        ss_res = ((log_counts - y_pred) ** 2 * weights.unsqueeze(0)).sum(dim=1)
        ss_tot = (dy ** 2 * weights.unsqueeze(0)).sum(dim=1)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)

        return dimension.clamp(0, 3), r_squared.clamp(0, 1)

    def forward(
        self,
        attention: torch.Tensor,
        return_dimension: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Analyze attention patterns for coherence via fractal dimension.

        Args:
            attention: [batch, seq, seq] or [batch, heads, seq, seq]
            return_dimension: Whether to return raw dimension estimate

        Returns:
            coherence: [batch] - 0=collapse/chaos, 1=healthy dimension
            dimension: Optional [batch] - raw fractal dimension
        """
        # Handle multi-head attention
        if attention.dim() == 4:
            # Average over heads
            attention = attention.mean(dim=1)

        # Estimate dimension
        dimension, r_squared = self.estimate_dimension(attention)

        # Map dimension to coherence score
        # Optimal dimension is ~1.5, penalty for deviation
        deviation = torch.abs(dimension - self.target_dimension)

        # Penalize extreme dimensions more
        collapse = (dimension < self.min_dimension).float()
        chaos = (dimension > self.max_dimension).float()

        # Coherence score
        base_coherence = self.dimension_coherence(dimension.unsqueeze(-1)).squeeze(-1)
        coherence = base_coherence * (1 - collapse * 0.5) * (1 - chaos * 0.3) * r_squared

        if return_dimension:
            return coherence, dimension
        return coherence, None


# =============================================================================
# NOETIC FRACTAL ENCODER
# =============================================================================

class NoeticFractalEncoder(nn.Module):
    """
    Maps phrase embeddings to Noetic Fractal space (X:Y:Z coordinates).

    Core Insight: Coherent phrases naturally cluster around valid NF combinations.
    Gibberish maps to invalid/incoherent NF regions.

    Architecture:
        1. Project embedding to 3 x 10 logits (outer, middle, inner noetic)
        2. Use Gumbel-Softmax for differentiable NF selection
        3. Validate against known coherent/incoherent patterns
        4. Return NF coordinates with validity score
    """

    def __init__(
        self,
        embed_dim: int = 384,
        noetic_dim: int = 40,
        num_noetics: int = 10,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.noetic_dim = noetic_dim
        self.num_noetics = num_noetics
        self.temperature = temperature

        # Hierarchical NF encoding
        # Level 1 (X): outer noetic
        self.outer_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_noetics),
        )

        # Level 2 (Y): middle noetic (conditioned on X)
        self.middle_encoder = nn.Sequential(
            nn.Linear(embed_dim + num_noetics, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_noetics),
        )

        # Level 3 (Z): inner noetic (conditioned on X:Y)
        self.inner_encoder = nn.Sequential(
            nn.Linear(embed_dim + num_noetics * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_noetics),
        )

        # Validity predictor (learned from training data)
        self.validity_head = nn.Sequential(
            nn.Linear(num_noetics * 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Embedding for each NF combination (10^3 = 1000 possible)
        self.nf_embeddings = nn.Parameter(torch.randn(10, 10, 10, noetic_dim) * 0.02)

        # Register incoherent patterns as buffer
        incoherent_tensor = torch.zeros(10, 10, 10)
        for pattern in INCOHERENT_PATTERNS:
            incoherent_tensor[pattern] = 1.0
        self.register_buffer('incoherent_mask', incoherent_tensor)

    def encode(
        self,
        embeddings: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode embeddings to NF coordinates.

        Args:
            embeddings: [batch, embed_dim] or [batch, seq, embed_dim]
            hard: If True, return hard (discrete) NF selection

        Returns:
            x_probs: [batch, 10] outer noetic probabilities
            y_probs: [batch, 10] middle noetic probabilities
            z_probs: [batch, 10] inner noetic probabilities
            validity: [batch] NF validity score
        """
        # Handle sequence input
        if embeddings.dim() == 3:
            # Pool over sequence
            embeddings = embeddings.mean(dim=1)

        batch = embeddings.shape[0]

        # Encode outer noetic (X)
        x_logits = self.outer_encoder(embeddings)
        if hard:
            x_probs = F.gumbel_softmax(x_logits, tau=self.temperature, hard=True)
        else:
            x_probs = F.softmax(x_logits / self.temperature, dim=-1)

        # Encode middle noetic (Y) conditioned on X
        y_input = torch.cat([embeddings, x_probs], dim=-1)
        y_logits = self.middle_encoder(y_input)
        if hard:
            y_probs = F.gumbel_softmax(y_logits, tau=self.temperature, hard=True)
        else:
            y_probs = F.softmax(y_logits / self.temperature, dim=-1)

        # Encode inner noetic (Z) conditioned on X:Y
        z_input = torch.cat([embeddings, x_probs, y_probs], dim=-1)
        z_logits = self.inner_encoder(z_input)
        if hard:
            z_probs = F.gumbel_softmax(z_logits, tau=self.temperature, hard=True)
        else:
            z_probs = F.softmax(z_logits / self.temperature, dim=-1)

        # Predict validity
        nf_concat = torch.cat([x_probs, y_probs, z_probs], dim=-1)
        validity = self.validity_head(nf_concat).squeeze(-1)

        # Penalize incoherent patterns
        # Compute probability mass on incoherent combinations
        # P(X=i, Y=j, Z=k) = x_probs[i] * y_probs[j] * z_probs[k]
        prob_cube = x_probs.unsqueeze(-1).unsqueeze(-1) * \
                    y_probs.unsqueeze(-2).unsqueeze(-1) * \
                    z_probs.unsqueeze(-2).unsqueeze(-2)  # [batch, 10, 10, 10]

        incoherent_mass = (prob_cube * self.incoherent_mask).sum(dim=(1, 2, 3))
        validity = validity * (1 - incoherent_mass)

        return x_probs, y_probs, z_probs, validity

    def get_nf_indices(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get discrete NF indices (argmax selection).

        Args:
            embeddings: [batch, embed_dim]

        Returns:
            x_idx, y_idx, z_idx: [batch] discrete noetic indices
        """
        x_probs, y_probs, z_probs, _ = self.encode(embeddings, hard=True)

        x_idx = x_probs.argmax(dim=-1)
        y_idx = y_probs.argmax(dim=-1)
        z_idx = z_probs.argmax(dim=-1)

        return x_idx, y_idx, z_idx

    def get_nf_embedding(
        self,
        embeddings: torch.Tensor,
        soft: bool = True,
    ) -> torch.Tensor:
        """
        Get NF-projected embedding in noetic space.

        Args:
            embeddings: [batch, embed_dim]
            soft: If True, use soft weighted sum of NF embeddings

        Returns:
            nf_embed: [batch, noetic_dim]
        """
        x_probs, y_probs, z_probs, _ = self.encode(embeddings, hard=not soft)

        if soft:
            # Weighted sum over all NF combinations
            # [batch, 10, 10, 10] @ [10, 10, 10, noetic_dim]
            prob_cube = x_probs.unsqueeze(-1).unsqueeze(-1) * \
                        y_probs.unsqueeze(-2).unsqueeze(-1) * \
                        z_probs.unsqueeze(-2).unsqueeze(-2)

            nf_embed = torch.einsum('bijk,ijkd->bd', prob_cube, self.nf_embeddings)
        else:
            # Hard selection
            x_idx, y_idx, z_idx = self.get_nf_indices(embeddings)
            nf_embed = self.nf_embeddings[x_idx, y_idx, z_idx]

        return nf_embed

    def forward(
        self,
        embeddings: torch.Tensor,
        return_nf: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Encode embeddings and return NF validity score.

        Args:
            embeddings: [batch, embed_dim] or [batch, seq, embed_dim]
            return_nf: If True, return NF indices

        Returns:
            nf_embed: [batch, noetic_dim] - NF-projected embedding
            validity: [batch] - NF validity score
            nf_indices: Optional (x, y, z) discrete indices
        """
        x_probs, y_probs, z_probs, validity = self.encode(embeddings)
        nf_embed = self.get_nf_embedding(embeddings, soft=True)

        if return_nf:
            x_idx = x_probs.argmax(dim=-1)
            y_idx = y_probs.argmax(dim=-1)
            z_idx = z_probs.argmax(dim=-1)
            return nf_embed, validity, (x_idx, y_idx, z_idx)

        return nf_embed, validity, None


# =============================================================================
# NOETIC FRACTAL DECODER
# =============================================================================

class NoeticFractalDecoder(nn.Module):
    """
    Decodes Noetic Fractal coordinates back to language embedding space.

    This enables the model to "think in NFs" and output coherent language.

    Architecture:
        1. Look up NF embedding from coordinates
        2. Apply noetic-specific transforms
        3. Project to language embedding space
        4. Add semantic structure from NF meaning
    """

    def __init__(
        self,
        noetic_dim: int = 40,
        embed_dim: int = 384,
        num_noetics: int = 10,
    ):
        super().__init__()
        self.noetic_dim = noetic_dim
        self.embed_dim = embed_dim
        self.num_noetics = num_noetics

        # NF to noetic space projection
        self.nf_to_noetic = nn.Sequential(
            nn.Linear(num_noetics * 3, noetic_dim * 2),
            nn.GELU(),
            nn.Linear(noetic_dim * 2, noetic_dim),
        )

        # Noetic to language projection
        self.noetic_to_embed = nn.Sequential(
            nn.Linear(noetic_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Semantic structure injection (based on NF meaning)
        self.semantic_structure = nn.ParameterDict({
            str(k): nn.Parameter(torch.randn(embed_dim) * 0.01)
            for k in range(num_noetics)
        })

        # Layer norm for output stability
        self.output_norm = nn.LayerNorm(embed_dim)

    def decode(
        self,
        x_probs: torch.Tensor,
        y_probs: torch.Tensor,
        z_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode NF probabilities to language embedding.

        Args:
            x_probs: [batch, 10] outer noetic probs
            y_probs: [batch, 10] middle noetic probs
            z_probs: [batch, 10] inner noetic probs

        Returns:
            embed: [batch, embed_dim] language embedding
        """
        # Concatenate NF representation
        nf_repr = torch.cat([x_probs, y_probs, z_probs], dim=-1)  # [batch, 30]

        # Project to noetic space
        noetic_repr = self.nf_to_noetic(nf_repr)  # [batch, noetic_dim]

        # Project to language space
        embed = self.noetic_to_embed(noetic_repr)  # [batch, embed_dim]

        # Add semantic structure from dominant noetics
        x_dominant = x_probs.argmax(dim=-1)
        y_dominant = y_probs.argmax(dim=-1)
        z_dominant = z_probs.argmax(dim=-1)

        # Weighted semantic injection
        for i in range(embed.shape[0]):
            x_struct = self.semantic_structure[str(x_dominant[i].item())]
            y_struct = self.semantic_structure[str(y_dominant[i].item())]
            z_struct = self.semantic_structure[str(z_dominant[i].item())]

            # Hierarchical weighting: outer has most influence
            embed[i] = embed[i] + 0.3 * x_struct + 0.2 * y_struct + 0.1 * z_struct

        return self.output_norm(embed)

    def decode_from_indices(
        self,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
        z_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode from discrete NF indices.

        Args:
            x_idx, y_idx, z_idx: [batch] integer indices

        Returns:
            embed: [batch, embed_dim]
        """
        batch = x_idx.shape[0]
        device = x_idx.device

        # One-hot encode
        x_probs = F.one_hot(x_idx, self.num_noetics).float()
        y_probs = F.one_hot(y_idx, self.num_noetics).float()
        z_probs = F.one_hot(z_idx, self.num_noetics).float()

        return self.decode(x_probs, y_probs, z_probs)

    def forward(
        self,
        nf_input: torch.Tensor,
        is_indices: bool = False,
    ) -> torch.Tensor:
        """
        Decode NF to language embedding.

        Args:
            nf_input: Either [batch, 30] (probs) or [batch, 3] (indices)
            is_indices: Whether input is discrete indices

        Returns:
            embed: [batch, embed_dim]
        """
        if is_indices:
            x_idx = nf_input[:, 0].long()
            y_idx = nf_input[:, 1].long()
            z_idx = nf_input[:, 2].long()
            return self.decode_from_indices(x_idx, y_idx, z_idx)
        else:
            x_probs = nf_input[:, :10]
            y_probs = nf_input[:, 10:20]
            z_probs = nf_input[:, 20:30]
            return self.decode(x_probs, y_probs, z_probs)


# =============================================================================
# COHERENCE GATE
# =============================================================================

class CoherenceGate(nn.Module):
    """
    Gates model outputs based on coherence analysis.

    This is the key component that prevents gibberish:
        1. Analyze candidate output with LacunaryAnalyzer
        2. Check NF validity with NoeticFractalEncoder
        3. Monitor fractal dimension of attention
        4. Compute overall coherence score
        5. Block or constrain output if incoherent

    Modes:
        - BLOCK: Completely block incoherent outputs (return zeros)
        - CONSTRAIN: Project incoherent outputs toward coherent NF space
        - WARN: Pass through with warning flag
    """

    def __init__(
        self,
        embed_dim: int = 384,
        noetic_dim: int = 40,
        coherence_threshold: float = 0.5,
        mode: str = 'constrain',  # 'block', 'constrain', 'warn'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.noetic_dim = noetic_dim
        self.coherence_threshold = coherence_threshold
        self.mode = mode

        # Component analyzers
        self.lacunary_analyzer = LacunaryCoherenceAnalyzer(embed_dim, noetic_dim)
        self.nf_encoder = NoeticFractalEncoder(embed_dim, noetic_dim)
        self.nf_decoder = NoeticFractalDecoder(noetic_dim, embed_dim)
        self.fractal_tracker = FractalDimensionTracker()

        # Coherence score aggregator
        self.score_aggregator = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Coherence projection (for constrain mode)
        self.coherence_proj = nn.Sequential(
            nn.Linear(embed_dim + noetic_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Output layer norm
        self.output_norm = nn.LayerNorm(embed_dim)

    def compute_coherence(
        self,
        embeddings: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
    ) -> CoherenceScore:
        """
        Compute comprehensive coherence score.

        Args:
            embeddings: [batch, seq, embed_dim]
            attention: Optional [batch, seq, seq] attention weights

        Returns:
            CoherenceScore with all diagnostic information
        """
        batch = embeddings.shape[0]
        device = embeddings.device

        # 1. Lacunary analysis
        lacunary_idx, lacunary_diag = self.lacunary_analyzer(embeddings, return_diagnostics=True)

        # 2. NF validity
        pooled = embeddings.mean(dim=1)  # [batch, embed_dim]
        nf_embed, nf_validity, nf_indices = self.nf_encoder(pooled, return_nf=True)

        # 3. Fractal dimension (if attention provided)
        if attention is not None:
            frac_coherence, frac_dim = self.fractal_tracker(attention, return_dimension=True)
        else:
            frac_coherence = torch.ones(batch, device=device)
            frac_dim = torch.full((batch,), 1.5, device=device)

        # 4. Semantic stability (NF-based)
        # Check if detected NF is in coherent anchor patterns
        stability_scores = []
        detected_nfs = []
        for i in range(batch):
            nf = (nf_indices[0][i].item(), nf_indices[1][i].item(), nf_indices[2][i].item())
            detected_nfs.append(nf)

            if nf in COHERENT_ANCHORS:
                stability_scores.append(1.0)
            elif nf in INCOHERENT_PATTERNS:
                stability_scores.append(0.0)
            else:
                # Neutral pattern
                stability_scores.append(0.7)

        semantic_stability = torch.tensor(stability_scores, device=device)

        # 5. Aggregate scores
        score_inputs = torch.stack([
            1 - lacunary_idx,  # Invert: low lacunary = good
            nf_validity,
            frac_coherence,
            semantic_stability,
        ], dim=-1)  # [batch, 4]

        overall = self.score_aggregator(score_inputs).squeeze(-1)

        # Detect specific issues
        collapse_detected = (frac_dim < 0.5).any().item() if attention is not None else False
        gibberish_detected = (lacunary_idx > 0.7).any().item()

        # Return first batch item's NF for simplicity
        detected_nf = detected_nfs[0] if detected_nfs else None

        return CoherenceScore(
            lacunary_index=lacunary_idx.mean().item(),
            fractal_dimension=frac_dim.mean().item() if attention is not None else 1.5,
            nf_validity=nf_validity.mean().item(),
            semantic_stability=semantic_stability.mean().item(),
            overall=overall.mean().item(),
            detected_nf=detected_nf,
            collapse_detected=collapse_detected,
            gibberish_detected=gibberish_detected,
        )

    def constrain_to_coherent(
        self,
        embeddings: torch.Tensor,
        nf_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project embeddings toward coherent NF space.

        Args:
            embeddings: [batch, seq, embed_dim]
            nf_embed: [batch, noetic_dim] coherent NF embedding

        Returns:
            constrained: [batch, seq, embed_dim] coherence-constrained embeddings
        """
        batch, seq, dim = embeddings.shape

        # Expand NF embedding to sequence
        nf_expanded = nf_embed.unsqueeze(1).expand(-1, seq, -1)  # [batch, seq, noetic_dim]

        # Combine with original embeddings
        combined = torch.cat([embeddings, nf_expanded], dim=-1)  # [batch, seq, embed_dim + noetic_dim]

        # Project to coherent space
        constrained = self.coherence_proj(combined)

        return self.output_norm(constrained)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
        return_score: bool = False,
    ) -> Tuple[torch.Tensor, Optional[CoherenceScore]]:
        """
        Apply coherence gating to embeddings.

        Args:
            embeddings: [batch, seq, embed_dim]
            attention: Optional attention weights for dimension tracking
            return_score: Whether to return detailed coherence score

        Returns:
            output: [batch, seq, embed_dim] gated embeddings
            score: Optional CoherenceScore
        """
        # Compute coherence
        score = self.compute_coherence(embeddings, attention)

        # Get NF embedding for potential constraint
        pooled = embeddings.mean(dim=1)
        nf_embed, _, _ = self.nf_encoder(pooled)

        if score.is_coherent(self.coherence_threshold):
            # Coherent: pass through
            output = embeddings
        else:
            # Incoherent: apply mode-specific handling
            if self.mode == 'block':
                # Zero out incoherent outputs
                output = torch.zeros_like(embeddings)
            elif self.mode == 'constrain':
                # Project toward coherent NF space
                output = self.constrain_to_coherent(embeddings, nf_embed)
            else:  # 'warn'
                # Pass through (warning in score)
                output = embeddings

        if return_score:
            return output, score
        return output, None


# =============================================================================
# COMPLETE NOETIC FRACTAL COHERENCE SYSTEM
# =============================================================================

class NoeticFractalCoherenceSystem(nn.Module):
    """
    Complete coherence system integrating all components.

    This module wraps the entire coherence pipeline and provides:
        1. Pre-generation coherence check on input
        2. During-generation attention monitoring
        3. Post-generation coherence gate on output
        4. NF-based representation for training

    Usage:
        system = NoeticFractalCoherenceSystem(...)

        # Check input coherence
        input_score = system.check_input(input_embeddings)

        # During generation, monitor attention
        system.update_attention_history(attention)

        # Gate output
        output, score = system.gate_output(output_embeddings)

        # Get NF representation for training
        nf_repr = system.get_nf_representation(embeddings)
    """

    def __init__(
        self,
        embed_dim: int = 384,
        noetic_dim: int = 40,
        coherence_threshold: float = 0.5,
        gate_mode: str = 'constrain',
        history_length: int = 10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.noetic_dim = noetic_dim
        self.coherence_threshold = coherence_threshold

        # Core components
        self.coherence_gate = CoherenceGate(
            embed_dim, noetic_dim, coherence_threshold, gate_mode
        )

        # Attention history for dimension tracking
        self.history_length = history_length
        self.attention_history: List[torch.Tensor] = []

        # NF representation for output
        self.nf_output_proj = nn.Linear(noetic_dim, embed_dim)

        # Coherence loss computation
        self.coherence_loss_weight = nn.Parameter(torch.tensor(1.0))

    def check_input(self, embeddings: torch.Tensor) -> CoherenceScore:
        """Check coherence of input embeddings."""
        return self.coherence_gate.compute_coherence(embeddings)

    def update_attention_history(self, attention: torch.Tensor):
        """Track attention patterns over generation."""
        self.attention_history.append(attention.detach())
        if len(self.attention_history) > self.history_length:
            self.attention_history.pop(0)

    def get_attention_trend(self) -> Optional[torch.Tensor]:
        """Get aggregated attention for dimension tracking."""
        if not self.attention_history:
            return None
        return torch.stack(self.attention_history).mean(dim=0)

    def gate_output(
        self,
        embeddings: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, CoherenceScore]:
        """Gate output embeddings based on coherence."""
        if attention is None:
            attention = self.get_attention_trend()
        return self.coherence_gate(embeddings, attention, return_score=True)

    def get_nf_representation(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int], float]:
        """
        Get Noetic Fractal representation of embeddings.

        Args:
            embeddings: [batch, embed_dim] or [batch, seq, embed_dim]

        Returns:
            nf_embed: [batch, noetic_dim]
            nf_coords: (x, y, z) tuple of dominant noetics
            validity: NF validity score
        """
        nf_encoder = self.coherence_gate.nf_encoder

        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)

        nf_embed, validity, (x, y, z) = nf_encoder(embeddings, return_nf=True)

        # Get dominant NF coordinates
        coords = (x[0].item(), y[0].item(), z[0].item())

        return nf_embed, coords, validity[0].item()

    def compute_coherence_loss(
        self,
        embeddings: torch.Tensor,
        target_validity: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute loss to encourage coherent outputs.

        Args:
            embeddings: Model output embeddings
            target_validity: Target NF validity (1.0 = fully coherent)

        Returns:
            loss: Scalar coherence loss
        """
        score = self.coherence_gate.compute_coherence(embeddings)

        # Loss components
        lacunary_loss = score.lacunary_index  # Penalize high lacunarity
        validity_loss = (target_validity - score.nf_validity) ** 2
        stability_loss = (1 - score.semantic_stability) ** 2

        total_loss = self.coherence_loss_weight * (
            0.3 * lacunary_loss +
            0.4 * validity_loss +
            0.3 * stability_loss
        )

        return total_loss

    def nf_to_text_hint(self, nf_coords: Tuple[int, int, int]) -> str:
        """
        Convert NF coordinates to human-readable meaning.

        Args:
            nf_coords: (x, y, z) noetic indices

        Returns:
            meaning: Human-readable NF meaning
        """
        x, y, z = nf_coords

        x_name = NOETIC_MEANINGS[x]
        y_name = NOETIC_MEANINGS[y]
        z_name = NOETIC_MEANINGS[z]

        # Format: "X containing Y containing Z"
        return f"{x_name} containing {y_name} containing {z_name} ({x}:{y}:{z})"

    def forward(
        self,
        embeddings: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process embeddings through complete coherence system.

        Args:
            embeddings: [batch, seq, embed_dim]
            attention: Optional attention weights
            return_all: If True, return all intermediate outputs

        Returns:
            Dict with 'output', 'coherence_score', 'nf_coords', etc.
        """
        # Update attention history if provided
        if attention is not None:
            self.update_attention_history(attention)

        # Gate output
        output, score = self.gate_output(embeddings, attention)

        # Get NF representation
        nf_embed, nf_coords, validity = self.get_nf_representation(embeddings)

        result = {
            'output': output,
            'coherence_score': torch.tensor(score.overall),
            'is_coherent': torch.tensor(score.is_coherent(self.coherence_threshold)),
            'nf_coords': torch.tensor(nf_coords),
            'nf_validity': torch.tensor(validity),
        }

        if return_all:
            result['nf_embed'] = nf_embed
            result['lacunary_index'] = torch.tensor(score.lacunary_index)
            result['fractal_dimension'] = torch.tensor(score.fractal_dimension)
            result['semantic_stability'] = torch.tensor(score.semantic_stability)
            result['detected_nf'] = nf_coords
            result['gibberish_detected'] = score.gibberish_detected
            result['collapse_detected'] = score.collapse_detected

        return result

    def reset_history(self):
        """Clear attention history."""
        self.attention_history.clear()


# =============================================================================
# TRAINING DATA GENERATOR FOR NF PATTERNS
# =============================================================================

class NoeticFractalTrainingGenerator:
    """
    Generates training data that maps phrases to Noetic Fractals.

    This creates (phrase, NF) pairs for teaching the model to "speak in NFs"
    that come out as coherent language.
    """

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phrase templates for each noetic
        self.templates = {
            0: ["pure {concept}", "the essence of {concept}", "{concept} itself"],
            1: ["awareness of {concept}", "noticing {concept}", "conscious of {concept}"],
            2: ["drawn to {concept}", "wanting {concept}", "attracted by {concept}"],
            3: ["rejecting {concept}", "averse to {concept}", "pulling away from {concept}"],
            4: ["deeply feeling {concept}", "resonating with {concept}", "intense {concept}"],
            5: ["conditioned by {concept}", "patterned around {concept}", "belief in {concept}"],
            6: ["structuring {concept}", "composing {concept}", "building {concept}"],
            7: ["cycling through {concept}", "rhythm of {concept}", "timing of {concept}"],
            8: ["caused by {concept}", "triggered by {concept}", "sourced from {concept}"],
            9: ["resulting in {concept}", "effect of {concept}", "outcome of {concept}"],
        }

        # Concept words for variation
        self.concepts = [
            "change", "self", "others", "reality", "truth", "growth",
            "fear", "love", "power", "wisdom", "connection", "meaning",
        ]

    def generate_nf_phrase(self, x: int, y: int, z: int) -> str:
        """Generate a phrase for given NF coordinates."""
        import random

        # Get templates for each level
        x_template = random.choice(self.templates[x])
        y_template = random.choice(self.templates[y])
        z_template = random.choice(self.templates[z])

        # Pick concept
        concept = random.choice(self.concepts)

        # Build nested phrase (inside-out)
        z_phrase = z_template.format(concept=concept)
        y_phrase = y_template.format(concept=z_phrase)
        x_phrase = x_template.format(concept=y_phrase)

        return x_phrase

    def generate_training_pairs(
        self,
        num_pairs: int = 1000,
        include_incoherent: bool = True,
    ) -> List[Dict]:
        """
        Generate training pairs for NF learning.

        Args:
            num_pairs: Number of pairs to generate
            include_incoherent: Include incoherent patterns (for contrast learning)

        Returns:
            List of {phrase, nf, validity, meaning} dicts
        """
        import random

        pairs = []

        for _ in range(num_pairs):
            # Random NF coordinates
            x = random.randint(0, 9)
            y = random.randint(0, 9)
            z = random.randint(0, 9)

            nf = (x, y, z)
            phrase = self.generate_nf_phrase(x, y, z)

            # Determine validity
            if nf in INCOHERENT_PATTERNS:
                validity = 0.0
                meaning = "INCOHERENT"
            elif nf in COHERENT_ANCHORS:
                validity = 1.0
                meaning = COHERENT_ANCHORS[nf]
            else:
                validity = 0.7
                meaning = f"{NOETIC_MEANINGS[x]} within {NOETIC_MEANINGS[y]} within {NOETIC_MEANINGS[z]}"

            if not include_incoherent and validity == 0.0:
                continue

            pairs.append({
                "phrase": phrase,
                "nf": f"{x}:{y}:{z}",
                "nf_indices": [x, y, z],
                "validity": validity,
                "meaning": meaning,
            })

        return pairs

    def save_training_data(
        self,
        num_pairs: int = 2000,
        filename: str = "nf_coherence_training.jsonl",
    ):
        """Save training data to JSONL file."""
        pairs = self.generate_training_pairs(num_pairs)

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')

        print(f"Generated {len(pairs)} NF training pairs -> {output_path}")

        # Statistics
        valid = sum(1 for p in pairs if p['validity'] > 0.5)
        print(f"  Coherent: {valid}, Incoherent: {len(pairs) - valid}")


# =============================================================================
# DEMO AND TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("NOETIC FRACTAL COHERENCE SYSTEM - Demo")
    print("=" * 80)

    # Create system
    system = NoeticFractalCoherenceSystem(
        embed_dim=384,
        noetic_dim=40,
        coherence_threshold=0.5,
        gate_mode='constrain',
    )

    print(f"\nSystem parameters: {sum(p.numel() for p in system.parameters()):,}")

    # Test with random embeddings
    batch, seq = 4, 16
    embeddings = torch.randn(batch, seq, 384)
    attention = torch.softmax(torch.randn(batch, seq, seq), dim=-1)

    # Process through system
    result = system(embeddings, attention, return_all=True)

    print(f"\nInput shape: {embeddings.shape}")
    print(f"Output shape: {result['output'].shape}")
    print(f"\nCoherence Analysis:")
    print(f"  Overall score: {result['coherence_score'].item():.3f}")
    print(f"  Is coherent: {result['is_coherent'].item()}")
    print(f"  NF coordinates: {result['detected_nf']}")
    print(f"  NF validity: {result['nf_validity'].item():.3f}")
    print(f"  Lacunary index: {result['lacunary_index'].item():.3f}")
    print(f"  Fractal dimension: {result['fractal_dimension'].item():.3f}")
    print(f"  Semantic stability: {result['semantic_stability'].item():.3f}")
    print(f"  Gibberish detected: {result['gibberish_detected']}")
    print(f"  Collapse detected: {result['collapse_detected']}")

    # Get NF meaning
    nf_meaning = system.nf_to_text_hint(result['detected_nf'])
    print(f"\nDetected NF meaning: {nf_meaning}")

    # Test training data generator
    print("\n" + "-" * 40)
    print("Training Data Generator Demo")
    print("-" * 40)

    generator = NoeticFractalTrainingGenerator(output_dir="data")
    pairs = generator.generate_training_pairs(num_pairs=5)

    print("\nSample NF training pairs:")
    for pair in pairs:
        print(f"\n  Phrase: {pair['phrase'][:60]}...")
        print(f"  NF: {pair['nf']}")
        print(f"  Validity: {pair['validity']}")
        print(f"  Meaning: {pair['meaning']}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)

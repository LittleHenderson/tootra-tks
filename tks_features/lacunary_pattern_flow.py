"""
Lacunary Pattern Flow System

A comprehensive pattern tracking, categorization, and feedback system that monitors
the flow of semantic patterns through the entire input→processing→output pipeline.

Core Architecture:
    INPUT → LacunaryTracker → PatternCategorizer → CoherenceBank
                                     ↓
    OUTPUT ← PatternFeedback ← DepthController ← PatternRetriever

Key Components:
    1. LacunaryPatternTracker - Monitors patterns at every layer
    2. PatternCategorizer - Classifies patterns into coherence categories
    3. CoherencePatternBank - Stores high-coherence patterns with metadata
    4. PatternFeedbackLoop - Uses stored patterns to guide generation
    5. AdaptiveDepthController - Adjusts processing depth based on coherence
    6. CoherenceMemory - Persistent cross-session pattern storage

The Flow:
    1. Input enters → Lacunary analysis at embedding layer
    2. Each transformer layer → Pattern tracked and categorized
    3. High-coherence patterns → Saved to CoherencePatternBank
    4. Low-coherence detection → Triggers depth increase via DPS
    5. Output generation → Guided by retrieved coherent patterns
    6. Post-generation → Best patterns persisted for future use

Mathematical Foundation (Lacunary Flow):
    At each layer l, compute:
        - Gap vector: G_l = ||h_l - h_{l-1}||
        - Flow coherence: FC_l = 1 - (G_l / G_max)
        - Cumulative lacunarity: CL = Σ(λ^l * G_l) where λ < 1 (decay)

    Pattern quality metric:
        Q = FC * NF_validity * (1 - lacunary_index)

Author: TKS Research Pipeline
Date: 2026-01-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import json
import math
import hashlib
from pathlib import Path
import time


# =============================================================================
# PATTERN DATA STRUCTURES
# =============================================================================

@dataclass
class PatternSignature:
    """Unique signature for a coherent pattern."""
    nf_coords: Tuple[int, int, int]      # Noetic Fractal coordinates
    embedding_hash: str                   # Hash of normalized embedding
    layer_origin: int                     # Which layer produced this
    quality_score: float                  # Overall quality metric
    category: str                         # Pattern category
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            'nf_coords': list(self.nf_coords),
            'embedding_hash': self.embedding_hash,
            'layer_origin': self.layer_origin,
            'quality_score': self.quality_score,
            'category': self.category,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PatternSignature':
        return cls(
            nf_coords=tuple(data['nf_coords']),
            embedding_hash=data['embedding_hash'],
            layer_origin=data['layer_origin'],
            quality_score=data['quality_score'],
            category=data['category'],
            timestamp=data.get('timestamp', time.time()),
        )


@dataclass
class LayerPattern:
    """Pattern captured at a specific layer."""
    layer_idx: int
    hidden_state: torch.Tensor           # [batch, seq, dim]
    gap_from_previous: torch.Tensor      # [batch, seq]
    flow_coherence: float
    nf_coords: Optional[Tuple[int, int, int]] = None
    category: Optional[str] = None


@dataclass
class FlowTrace:
    """Complete trace of pattern flow through model."""
    input_coherence: float
    layer_patterns: List[LayerPattern] = field(default_factory=list)
    cumulative_lacunarity: float = 0.0
    depth_adjustments: List[int] = field(default_factory=list)
    retrieved_patterns: List[PatternSignature] = field(default_factory=list)
    output_coherence: float = 0.0
    quality_score: float = 0.0

    def add_layer(self, pattern: LayerPattern):
        self.layer_patterns.append(pattern)

    def get_coherence_trajectory(self) -> List[float]:
        return [p.flow_coherence for p in self.layer_patterns]


# =============================================================================
# CANONICAL TKS IMPORTS
# =============================================================================

# Import canonical TKS definitions
from tks_rules.noetics import (
    NOETICS, NOETIC_NAMES, INVOLUTION_PAIRS, SELF_DUAL_NOETICS,
    MVR_FRACTAL, get_complement, validate_noetic
)
from tks_rules.worlds import (
    WORLDS, WORLD_LETTERS, WORLD_DOMAINS, WORLD_OFFSETS,
    WORLD_CASCADE_ORDER, world_distance
)
from tks_rules.foundations import (
    FOUNDATIONS, FOUNDATION_NAMES, SUB_FOUNDATIONS, SUB_FOUNDATION_CODES,
    validate_foundation, validate_subfoundation
)


# =============================================================================
# TKS PATTERN CATEGORIZATION SYSTEM
# =============================================================================

@dataclass
class TKSPatternCategory:
    """
    Canonical TKS pattern category using:
    - Dominant Noetic (0-9)
    - World Domain (A-D)
    - Foundation Principle (1-7)
    - Sub-Foundation Code (1a-7d)
    """
    noetic: int                    # Dominant noetic (0-9)
    world: str                     # World domain (A, B, C, D)
    foundation: int                # Foundation principle (1-7)
    subfoundation: str             # Sub-foundation code (1a-7d)
    is_mvr: bool                   # Is MVR Fractal pattern
    is_involution: bool            # Contains involution pair
    quality_boost: float           # Quality multiplier

    @property
    def noetic_name(self) -> str:
        return NOETIC_NAMES.get(self.noetic, "Unknown")

    @property
    def world_name(self) -> str:
        return WORLDS[self.world].name if self.world in WORLDS else "Unknown"

    @property
    def foundation_name(self) -> str:
        return FOUNDATION_NAMES.get(self.foundation, "Unknown")

    @property
    def subfoundation_name(self) -> str:
        sf = SUB_FOUNDATIONS.get(self.subfoundation)
        return sf.name if sf else "Unknown"

    def to_string(self) -> str:
        """Generate canonical TKS category string."""
        mvr_tag = "[MVR]" if self.is_mvr else ""
        inv_tag = "[INV]" if self.is_involution else ""
        return f"nu_{self.noetic}/{self.world}/{self.subfoundation}{mvr_tag}{inv_tag}"


class PatternCategory:
    """
    Canonical TKS Pattern Categorization System.

    Uses the TKS canonical structures:
    - 10 Noetics (nu_0 through nu_9) for consciousness operator classification
    - 4 Worlds (A, B, C, D) for domain level classification
    - 7 Foundations for functional principle classification
    - 28 Sub-Foundations (1a-7d) for detailed categorization
    - MVR Fractal (1, 4, 7) detection for stabilizing patterns
    - Involution pairs for complement/oscillation detection
    """

    # Quality boosts for noetics (some are more stable than others)
    NOETIC_QUALITY = {
        0: 0.85,   # Idea - abstract, needs grounding
        1: 1.15,   # Mind - stabilizing (MVR component)
        2: 1.0,    # Positive - attraction
        3: 0.9,    # Negative - repulsion (can destabilize)
        4: 1.15,   # Vibration - resonance (MVR component)
        5: 1.0,    # Female - receptivity
        6: 1.0,    # Male - structure
        7: 1.15,   # Rhythm - timing (MVR component)
        8: 1.05,   # Above - cause/trigger
        9: 0.95,   # Below - effect (reactive)
    }

    # Quality boosts for worlds (cascade order matters)
    WORLD_QUALITY = {
        "A": 1.1,   # Spiritual - archetypal coherence
        "B": 1.15,  # Mental - intellectual clarity
        "C": 0.95,  # Emotional - can oscillate
        "D": 1.0,   # Physical - grounded
    }

    # Quality boosts for foundations
    FOUNDATION_QUALITY = {
        1: 1.2,    # Association - coherence and linking (highest)
        2: 1.15,   # Wisdom - truth-orientation
        3: 1.05,   # Life - vitality
        4: 1.0,    # Companionship - relational
        5: 1.1,    # Power - influence
        6: 1.0,    # Wealth - material stability
        7: 1.05,   # Continuation - propagation
    }

    # MVR Fractal bonus
    MVR_BONUS = 1.25

    # Involution penalty (oscillating patterns less stable)
    INVOLUTION_PENALTY = 0.85

    @classmethod
    def _detect_dominant_noetic(cls, nf_coords: Tuple[int, int, int]) -> int:
        """Detect the dominant noetic in X:Y:Z coordinates."""
        x, y, z = nf_coords
        counts = {}
        for n in [x, y, z]:
            counts[n] = counts.get(n, 0) + 1
        return max(counts, key=counts.get)

    @classmethod
    def _detect_world(cls, nf_coords: Tuple[int, int, int]) -> str:
        """
        Map NF coordinates to World domain.

        Uses the average noetic value to determine world:
        - High (7-9): A (Spiritual/Archetypal)
        - Mid-high (5-6): B (Mental/Intellectual)
        - Mid-low (3-4): C (Emotional/Formative)
        - Low (0-2): D (Physical/Material)
        """
        avg = sum(nf_coords) / 3.0
        if avg >= 7:
            return "A"
        elif avg >= 5:
            return "B"
        elif avg >= 3:
            return "C"
        else:
            return "D"

    @classmethod
    def _detect_foundation(cls, nf_coords: Tuple[int, int, int], dominant: int) -> int:
        """
        Map pattern to Foundation principle based on dominant noetic.

        Foundation mapping:
        - nu_0 (Idea) -> F1 (Association) - linking ideas
        - nu_1 (Mind) -> F2 (Wisdom) - awareness/discernment
        - nu_2 (Positive) -> F4 (Companionship) - attraction/bonding
        - nu_3 (Negative) -> F5 (Power) - aversion/influence
        - nu_4 (Vibration) -> F3 (Life) - resonance/vitality
        - nu_5 (Female) -> F6 (Wealth) - receptivity/resources
        - nu_6 (Male) -> F7 (Continuation) - structure/propagation
        - nu_7 (Rhythm) -> F3 (Life) - timing/cycles
        - nu_8 (Above) -> F5 (Power) - cause/trigger
        - nu_9 (Below) -> F4 (Companionship) - effect/relationship
        """
        foundation_map = {
            0: 1,  # Idea -> Association
            1: 2,  # Mind -> Wisdom
            2: 4,  # Positive -> Companionship
            3: 5,  # Negative -> Power
            4: 3,  # Vibration -> Life
            5: 6,  # Female -> Wealth
            6: 7,  # Male -> Continuation
            7: 3,  # Rhythm -> Life
            8: 5,  # Above -> Power
            9: 4,  # Below -> Companionship
        }
        return foundation_map.get(dominant, 1)

    @classmethod
    def _is_mvr_pattern(cls, nf_coords: Tuple[int, int, int]) -> bool:
        """Check if pattern contains MVR Fractal (1, 4, 7)."""
        coord_set = set(nf_coords)
        mvr_set = set(MVR_FRACTAL)
        # Full MVR or at least 2 of 3 MVR components
        return len(coord_set & mvr_set) >= 2

    @classmethod
    def _contains_involution(cls, nf_coords: Tuple[int, int, int]) -> bool:
        """Check if pattern contains involution pairs."""
        x, y, z = nf_coords
        for pair in INVOLUTION_PAIRS:
            if set(pair) <= {x, y, z}:
                return True
        # Special causal pair
        if 8 in nf_coords and 9 in nf_coords:
            return True
        return False

    @classmethod
    def categorize(cls, nf_coords: Tuple[int, int, int]) -> Tuple[str, float]:
        """
        Categorize a Noetic Fractal pattern using canonical TKS system.

        Args:
            nf_coords: (X, Y, Z) NF coordinates

        Returns:
            category: TKS category string (nu_X/World/SubF[flags])
            quality_boost: Multiplicative quality adjustment
        """
        category = cls.categorize_full(nf_coords)
        return category.to_string(), category.quality_boost

    @classmethod
    def categorize_full(cls, nf_coords: Tuple[int, int, int]) -> TKSPatternCategory:
        """
        Full TKS categorization with all details.

        Args:
            nf_coords: (X, Y, Z) NF coordinates

        Returns:
            TKSPatternCategory with full details
        """
        # Detect components
        dominant = cls._detect_dominant_noetic(nf_coords)
        world = cls._detect_world(nf_coords)
        foundation = cls._detect_foundation(nf_coords, dominant)
        is_mvr = cls._is_mvr_pattern(nf_coords)
        is_involution = cls._contains_involution(nf_coords)

        # Build sub-foundation code
        world_lower = world.lower()
        subfoundation = f"{foundation}{world_lower}"

        # Calculate quality boost
        quality = 1.0
        quality *= cls.NOETIC_QUALITY.get(dominant, 1.0)
        quality *= cls.WORLD_QUALITY.get(world, 1.0)
        quality *= cls.FOUNDATION_QUALITY.get(foundation, 1.0)

        if is_mvr:
            quality *= cls.MVR_BONUS
        if is_involution:
            quality *= cls.INVOLUTION_PENALTY

        return TKSPatternCategory(
            noetic=dominant,
            world=world,
            foundation=foundation,
            subfoundation=subfoundation,
            is_mvr=is_mvr,
            is_involution=is_involution,
            quality_boost=quality,
        )

    @classmethod
    def get_category_description(cls, category: TKSPatternCategory) -> str:
        """Generate human-readable description of category."""
        noetic_def = NOETICS.get(category.noetic)
        world_def = WORLDS.get(category.world)
        foundation_def = FOUNDATIONS.get(category.foundation)

        parts = []

        if noetic_def:
            parts.append(f"Noetic: {noetic_def.name} ({noetic_def.role})")

        if world_def:
            parts.append(f"World: {world_def.name} ({world_def.domain})")

        if foundation_def:
            parts.append(f"Foundation: {foundation_def.name} ({foundation_def.principle})")

        if category.is_mvr:
            parts.append("Contains MVR Fractal (stabilizing)")

        if category.is_involution:
            parts.append("Contains involution pair (oscillating)")

        parts.append(f"Quality boost: {category.quality_boost:.3f}")

        return "\n".join(parts)

    @classmethod
    def get_all_subfoundations(cls) -> List[str]:
        """Get all 28 sub-foundation codes."""
        return SUB_FOUNDATION_CODES.copy()

    @classmethod
    def get_foundation_patterns(cls, foundation: int) -> List[str]:
        """Get all sub-foundations for a given foundation."""
        return [f"{foundation}{w}" for w in ["a", "b", "c", "d"]]

    @classmethod
    def get_world_patterns(cls, world: str) -> List[str]:
        """Get all sub-foundations for a given world."""
        w = world.lower()
        return [f"{f}{w}" for f in range(1, 8)]


# =============================================================================
# CANONICAL TKS LACUNARITY (from TKS Formal Mathematical Manual v7.4)
# =============================================================================

@dataclass
class TextureTuple:
    """
    Texture of a Noetic Fractal (Definition 6.38).

    Tex(Φω) = (dimH(AΦ), L(AΦ), Ξ(Φω))

    Where:
        - dimH = fractal dimension (Hausdorff)
        - L = prefactor lacunarity
        - Ξ = noetic complexity (Shannon entropy)
    """
    dimension: float           # Fractal dimension
    lacunarity: float          # Prefactor lacunarity L
    complexity: float          # Noetic complexity Ξ (Shannon entropy)
    is_coherent: bool = True   # Coherent if lacunarity close to 1

    def coherence_score(self) -> float:
        """Compute overall coherence from texture tuple."""
        # Lacunarity = 1 is perfectly coherent, higher = more gaps
        lacunarity_score = 1.0 / max(self.lacunarity, 1.0)
        # Higher complexity generally means more coherent (not repetitive)
        complexity_score = min(self.complexity / math.log(10), 1.0)  # Normalize to max entropy
        # Dimension should be moderate (not collapsed)
        dim_score = 1.0 - abs(self.dimension - 1.5) / 1.5
        return (lacunarity_score + complexity_score + dim_score) / 3.0


class CanonicalLacunarity:
    """
    Canonical Lacunarity calculations from TKS Formal Mathematical Manual v7.4.

    Implements:
        - Definition 6.31: Lacunarity Λε(A) = E[μ²ε] / E[με]² = 1 + Var(με) / E[με]²
        - Definition 6.33: Gliding-box lacunarity
        - Definition 6.39: Noetic complexity Ξ(Φω) = H(πΦω)
        - Theorem 6.41: Lacunarity-Noetic correspondence
    """

    # Stabilizing fractals (Corollary 2.23)
    STABILIZING_FRACTALS = {
        (1, 4, 7): "MVR Fractal (Mind-Vibration-Rhythm)",
        (1, 1, 1): "Deep Mind Fractal",
        (4, 4, 4): "Pure Vibration (Resonance)",
        (7, 7, 7): "Pure Rhythm (Timing)",
    }

    # Eigenfractals achieve maximum lacunarity (Corollary 6.43)
    # ⟨k:k:k:...⟩ω = deterministic, no entropy
    EIGENFRACTALS = {k: f"Eigenfractal_{k}" for k in range(10)}

    @staticmethod
    def compute_lacunarity(
        embeddings: torch.Tensor,
        scale: float = 1.0,
    ) -> float:
        """
        Compute lacunarity using the canonical formula (Definition 6.31).

        Λε(A) = E[μ²ε] / E[με]² = 1 + Var(με) / E[με]²

        Where με(x) = local mass at scale ε.

        Args:
            embeddings: [batch, seq, dim] or [seq, dim] tensor
            scale: Scale factor for neighborhood size

        Returns:
            Lacunarity value (1.0 = homogeneous/coherent, >1 = gappy/incoherent)
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=0)  # Average over batch

        seq_len, dim = embeddings.shape

        if seq_len < 2:
            return 1.0  # Trivially uniform

        # Compute pairwise distances (local mass proxy)
        # Using L2 distance to estimate local density
        dists = torch.cdist(embeddings.unsqueeze(0), embeddings.unsqueeze(0)).squeeze(0)

        # Scale-dependent neighborhood: count neighbors within radius
        radius = scale * dists.mean()
        local_mass = (dists <= radius).float().sum(dim=-1)  # [seq_len]

        # Compute lacunarity: Λ = 1 + Var(μ) / E[μ]²
        mean_mass = local_mass.mean()
        var_mass = local_mass.var()

        if mean_mass < 1e-8:
            return float('inf')  # Degenerate case

        lacunarity = 1.0 + (var_mass / (mean_mass ** 2)).item()
        return lacunarity

    @staticmethod
    def compute_gliding_box_lacunarity(
        embeddings: torch.Tensor,
        box_sizes: List[float] = None,
    ) -> Dict[str, float]:
        """
        Compute gliding-box lacunarity (Definition 6.33).

        Λr(A) = (Σx∈I |A ∩ Br(x)|² / |I|) / (Σx∈I |A ∩ Br(x)| / |I|)²

        Args:
            embeddings: [seq, dim] tensor
            box_sizes: List of relative box sizes to test

        Returns:
            Dict with lacunarity at each scale and average
        """
        if box_sizes is None:
            box_sizes = [0.1, 0.25, 0.5, 1.0, 2.0]

        results = {}
        lacunarities = []

        for scale in box_sizes:
            lac = CanonicalLacunarity.compute_lacunarity(embeddings, scale)
            results[f"scale_{scale}"] = lac
            if not math.isinf(lac):
                lacunarities.append(lac)

        results["mean"] = sum(lacunarities) / len(lacunarities) if lacunarities else float('inf')
        results["min"] = min(lacunarities) if lacunarities else float('inf')
        results["max"] = max(lacunarities) if lacunarities else float('inf')

        return results

    @staticmethod
    def compute_noetic_complexity(nf_sequence: List[Tuple[int, int, int]]) -> float:
        """
        Compute noetic complexity Ξ (Definition 6.39).

        Ξ(Φω) = H(πΦω)

        Where πΦω(k) is the empirical distribution of noetics:
            πΦω(k) = |{n < N | Φω(n) = k}| / N

        Higher complexity = more diverse noetics = generally more coherent.
        Lower complexity = repetitive = potentially incoherent (eigenfractal-like).

        Args:
            nf_sequence: List of (X, Y, Z) Noetic Fractal coordinates

        Returns:
            Shannon entropy of the noetic distribution
        """
        if not nf_sequence:
            return 0.0

        # Count all noetics across all positions
        noetic_counts = {k: 0 for k in range(10)}
        total = 0

        for x, y, z in nf_sequence:
            for noetic in [x, y, z]:
                if 0 <= noetic <= 9:
                    noetic_counts[noetic] += 1
                    total += 1

        if total == 0:
            return 0.0

        # Compute Shannon entropy
        entropy = 0.0
        for k, count in noetic_counts.items():
            if count > 0:
                p = count / total
                entropy -= p * math.log(p)

        return entropy

    @staticmethod
    def compute_clustering_coefficient(nf_sequence: List[Tuple[int, int, int]]) -> float:
        """
        Compute clustering coefficient γ (Theorem 6.41 D).

        γ(Φω) = max_k max_m |{n ∈ [m, m+N) | Φω(n) = k}| / N

        Measures the maximum run length of any noetic.
        High clustering implies high lacunarity (gibberish).

        Args:
            nf_sequence: List of NF coordinates

        Returns:
            Clustering coefficient (0 = no clustering, 1 = all same)
        """
        if not nf_sequence:
            return 0.0

        # Flatten to sequence of noetics
        noetics = []
        for x, y, z in nf_sequence:
            noetics.extend([x, y, z])

        if len(noetics) < 2:
            return 0.0

        # Find maximum run length for any noetic
        max_run = 1
        current_run = 1
        for i in range(1, len(noetics)):
            if noetics[i] == noetics[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        # Normalize by sequence length
        return max_run / len(noetics)

    @staticmethod
    def is_stabilizing_fractal(nf_coords: Tuple[int, int, int]) -> Tuple[bool, str]:
        """
        Check if NF is a stabilizing fractal (Corollary 2.23).

        Stabilizing fractals converge to fixed points:
            ∃ X* ∈ I : lim_{n→∞} Φⁿ(X) = X*

        Args:
            nf_coords: (X, Y, Z) Noetic Fractal coordinates

        Returns:
            (is_stabilizing, description)
        """
        # Check exact match
        if nf_coords in CanonicalLacunarity.STABILIZING_FRACTALS:
            return True, CanonicalLacunarity.STABILIZING_FRACTALS[nf_coords]

        # Check if eigenfractal (all same noetic)
        x, y, z = nf_coords
        if x == y == z:
            return True, f"Eigenfractal ⟨{x}:{x}:{x}⟩ (converges to fixed point)"

        # Check for partial MVR (at least 2 of Mind-Vibration-Rhythm)
        mvr_set = {1, 4, 7}
        coord_set = {x, y, z}
        mvr_overlap = len(mvr_set & coord_set)
        if mvr_overlap >= 2:
            return True, f"Partial MVR ({mvr_overlap}/3 components)"

        return False, "Non-stabilizing"

    @staticmethod
    def compute_texture(
        embeddings: torch.Tensor,
        nf_sequence: List[Tuple[int, int, int]],
        dimension: float = 1.5,
    ) -> TextureTuple:
        """
        Compute full texture tuple (Definition 6.38).

        Tex(Φω) = (dimH(AΦ), L(AΦ), Ξ(Φω))

        Args:
            embeddings: Token embeddings for lacunarity computation
            nf_sequence: Sequence of NF coordinates
            dimension: Pre-computed fractal dimension (if available)

        Returns:
            TextureTuple with all components
        """
        # Compute lacunarity
        lacunarity = CanonicalLacunarity.compute_lacunarity(embeddings)

        # Compute noetic complexity
        complexity = CanonicalLacunarity.compute_noetic_complexity(nf_sequence)

        # Determine coherence
        # Lacunarity close to 1 = coherent
        # High complexity = diverse = coherent
        # Clustering coefficient low = coherent
        clustering = CanonicalLacunarity.compute_clustering_coefficient(nf_sequence)

        is_coherent = (
            lacunarity < 2.0 and  # Not too gappy
            complexity > 1.0 and  # Not too repetitive
            clustering < 0.5      # Not too clustered
        )

        return TextureTuple(
            dimension=dimension,
            lacunarity=lacunarity,
            complexity=complexity,
            is_coherent=is_coherent,
        )

    @staticmethod
    def entropy_lacunarity_relationship(
        noetic_complexity: float,
        probability_entropy: float,
        lyapunov_exponent: float = 1.0,
        base_lacunarity: float = 1.0,
    ) -> float:
        """
        Compute lacunarity from entropy relationship (Theorem 6.41 A).

        L(AΦS) = exp((Ξ(Φω) - H(p)) / λ) · L₀

        Where:
            - Ξ(Φω) = noetic complexity
            - H(p) = probability entropy
            - λ = Lyapunov exponent
            - L₀ = base lacunarity constant

        Args:
            noetic_complexity: Ξ from noetic sequence
            probability_entropy: H(p) from probability weights
            lyapunov_exponent: λ (stability measure)
            base_lacunarity: L₀ (baseline)

        Returns:
            Predicted lacunarity
        """
        if lyapunov_exponent == 0:
            return base_lacunarity

        exponent = (noetic_complexity - probability_entropy) / lyapunov_exponent
        return math.exp(exponent) * base_lacunarity

    @staticmethod
    def clustering_lacunarity_bound(
        clustering_coefficient: float,
        constant: float = 1.0,
    ) -> float:
        """
        Compute lacunarity lower bound from clustering (Theorem 6.41 D).

        L(AΦS) ≥ 1 + c · γ(Φω)²

        Args:
            clustering_coefficient: γ from compute_clustering_coefficient
            constant: c (depends on contraction ratios)

        Returns:
            Lower bound on lacunarity
        """
        return 1.0 + constant * (clustering_coefficient ** 2)


# =============================================================================
# LACUNARY PATTERN TRACKER
# =============================================================================

class LacunaryPatternTracker(nn.Module):
    """
    Tracks semantic patterns as they flow through model layers.

    At each layer:
        1. Compute gap from previous layer
        2. Calculate flow coherence
        3. Extract NF coordinates
        4. Update cumulative lacunarity

    The tracker maintains a decay-weighted sum of gaps:
        CL = Σ(λ^l * G_l)

    Where λ < 1 ensures recent layers matter more.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        noetic_dim: int = 40,
        num_layers: int = 12,
        decay_factor: float = 0.9,
        coherence_threshold: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.noetic_dim = noetic_dim
        self.num_layers = num_layers
        self.decay_factor = decay_factor
        self.coherence_threshold = coherence_threshold

        # Per-layer projection to noetic space
        self.layer_projs = nn.ModuleList([
            nn.Linear(hidden_dim, noetic_dim) for _ in range(num_layers)
        ])

        # NF encoder (shared across layers)
        from tks_features.noetic_fractal_coherence import NoeticFractalEncoder
        self.nf_encoder = NoeticFractalEncoder(
            embed_dim=noetic_dim,
            noetic_dim=noetic_dim,
        )

        # Gap normalization (learned per layer)
        self.gap_norms = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_layers)
        ])

        # Flow coherence predictor
        self.flow_predictor = nn.Sequential(
            nn.Linear(noetic_dim * 2, noetic_dim),
            nn.GELU(),
            nn.Linear(noetic_dim, 1),
            nn.Sigmoid(),
        )

        # Current trace
        self.current_trace: Optional[FlowTrace] = None

    def start_trace(self, input_coherence: float = 1.0):
        """Start a new flow trace."""
        self.current_trace = FlowTrace(input_coherence=input_coherence)

    def track_layer(
        self,
        layer_idx: int,
        hidden_state: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> LayerPattern:
        """
        Track pattern at a specific layer.

        Args:
            layer_idx: Layer index
            hidden_state: [batch, seq, hidden_dim]
            prev_hidden: Previous layer's hidden state

        Returns:
            LayerPattern with analysis
        """
        batch, seq, dim = hidden_state.shape
        device = hidden_state.device

        # Project to noetic space
        noetic_repr = self.layer_projs[layer_idx](hidden_state)  # [batch, seq, noetic_dim]

        # Compute gap from previous layer
        if prev_hidden is not None:
            prev_noetic = self.layer_projs[max(0, layer_idx - 1)](prev_hidden)
            gap = (noetic_repr - prev_noetic).norm(dim=-1)  # [batch, seq]
            gap = gap / (self.gap_norms[layer_idx] + 1e-8)  # Normalize
        else:
            gap = torch.zeros(batch, seq, device=device)

        # Compute flow coherence
        if prev_hidden is not None:
            flow_input = torch.cat([
                noetic_repr.mean(dim=1),
                prev_noetic.mean(dim=1)
            ], dim=-1)  # [batch, noetic_dim * 2]
            flow_coherence = self.flow_predictor(flow_input).mean().item()
        else:
            flow_coherence = 1.0  # First layer is coherent by definition

        # Get NF coordinates
        pooled = noetic_repr.mean(dim=1)  # [batch, noetic_dim]
        _, _, (x, y, z) = self.nf_encoder(pooled, return_nf=True)
        nf_coords = (x[0].item(), y[0].item(), z[0].item())

        # Categorize
        category, _ = PatternCategory.categorize(nf_coords)

        pattern = LayerPattern(
            layer_idx=layer_idx,
            hidden_state=hidden_state.detach(),
            gap_from_previous=gap.detach(),
            flow_coherence=flow_coherence,
            nf_coords=nf_coords,
            category=category,
        )

        # Update trace
        if self.current_trace is not None:
            self.current_trace.add_layer(pattern)
            # Update cumulative lacunarity
            weight = self.decay_factor ** (self.num_layers - layer_idx)
            self.current_trace.cumulative_lacunarity += weight * gap.mean().item()

        return pattern

    def get_trace(self) -> Optional[FlowTrace]:
        """Get current flow trace."""
        return self.current_trace

    def compute_pattern_quality(
        self,
        pattern: LayerPattern,
        nf_validity: float,
    ) -> float:
        """
        Compute overall quality score for a pattern.

        Q = flow_coherence * nf_validity * category_boost * (1 - gap_penalty)
        """
        _, category_boost = PatternCategory.categorize(pattern.nf_coords)

        gap_penalty = pattern.gap_from_previous.mean().item()
        gap_penalty = min(gap_penalty / 2.0, 0.5)  # Cap at 0.5

        quality = (
            pattern.flow_coherence *
            nf_validity *
            category_boost *
            (1 - gap_penalty)
        )

        return max(0.0, min(1.0, quality))


# =============================================================================
# COHERENCE PATTERN BANK
# =============================================================================

class CoherencePatternBank(nn.Module):
    """
    Stores and retrieves high-coherence patterns for guiding generation.

    Features:
        - LRU cache for recent patterns
        - Quality-based eviction
        - Category-indexed retrieval
        - Embedding similarity search
    """

    def __init__(
        self,
        noetic_dim: int = 40,
        max_patterns: int = 10000,
        min_quality: float = 0.6,
        embedding_cache_size: int = 1000,
    ):
        super().__init__()

        self.noetic_dim = noetic_dim
        self.max_patterns = max_patterns
        self.min_quality = min_quality

        # Pattern storage: signature -> embedding
        self.patterns: OrderedDict[str, Tuple[PatternSignature, torch.Tensor]] = OrderedDict()

        # Category index for fast retrieval
        self.category_index: Dict[str, List[str]] = {}

        # Embedding cache for similarity search
        self.embedding_cache_size = embedding_cache_size
        self.register_buffer('embedding_cache', torch.zeros(embedding_cache_size, noetic_dim))
        self.register_buffer('cache_qualities', torch.zeros(embedding_cache_size))
        self.cache_signatures: List[Optional[str]] = [None] * embedding_cache_size
        self.cache_ptr = 0

        # Statistics
        self.total_added = 0
        self.total_retrieved = 0
        self.total_evicted = 0

    def _compute_hash(self, embedding: torch.Tensor) -> str:
        """Compute hash of normalized embedding."""
        normalized = F.normalize(embedding.flatten(), dim=0)
        # Quantize for stable hashing
        quantized = (normalized * 1000).int().cpu().numpy().tobytes()
        return hashlib.md5(quantized).hexdigest()[:16]

    def add_pattern(
        self,
        embedding: torch.Tensor,
        nf_coords: Tuple[int, int, int],
        layer_origin: int,
        quality_score: float,
    ) -> Optional[PatternSignature]:
        """
        Add a pattern to the bank if it meets quality threshold.

        Args:
            embedding: [noetic_dim] pattern embedding
            nf_coords: NF coordinates
            layer_origin: Source layer
            quality_score: Pattern quality

        Returns:
            PatternSignature if added, None otherwise
        """
        if quality_score < self.min_quality:
            return None

        # Compute signature
        emb_hash = self._compute_hash(embedding)
        category, _ = PatternCategory.categorize(nf_coords)

        signature = PatternSignature(
            nf_coords=nf_coords,
            embedding_hash=emb_hash,
            layer_origin=layer_origin,
            quality_score=quality_score,
            category=category,
        )

        # Check if already exists (update if higher quality)
        if emb_hash in self.patterns:
            existing_sig, existing_emb = self.patterns[emb_hash]
            if quality_score <= existing_sig.quality_score:
                return None  # Don't update with lower quality
            # Remove from category index
            if existing_sig.category in self.category_index:
                self.category_index[existing_sig.category] = [
                    h for h in self.category_index[existing_sig.category] if h != emb_hash
                ]

        # Evict if at capacity
        if len(self.patterns) >= self.max_patterns:
            self._evict_lowest_quality()

        # Add pattern
        self.patterns[emb_hash] = (signature, embedding.detach().clone())

        # Update category index
        if category not in self.category_index:
            self.category_index[category] = []
        self.category_index[category].append(emb_hash)

        # Update embedding cache
        self._update_cache(embedding, quality_score, emb_hash)

        self.total_added += 1
        return signature

    def _evict_lowest_quality(self):
        """Evict the lowest quality pattern."""
        if not self.patterns:
            return

        # Find lowest quality
        lowest_hash = min(
            self.patterns.keys(),
            key=lambda h: self.patterns[h][0].quality_score
        )

        sig, _ = self.patterns.pop(lowest_hash)

        # Remove from category index
        if sig.category in self.category_index:
            self.category_index[sig.category] = [
                h for h in self.category_index[sig.category] if h != lowest_hash
            ]

        self.total_evicted += 1

    def _update_cache(self, embedding: torch.Tensor, quality: float, sig_hash: str):
        """Update the embedding cache for fast retrieval."""
        # Find slot (either empty or lowest quality)
        if self.cache_ptr < self.embedding_cache_size:
            slot = self.cache_ptr
            self.cache_ptr += 1
        else:
            # Replace lowest quality
            slot = self.cache_qualities.argmin().item()
            if self.cache_qualities[slot] >= quality:
                return  # Don't replace with lower quality

        self.embedding_cache[slot] = embedding.detach()
        self.cache_qualities[slot] = quality
        self.cache_signatures[slot] = sig_hash

    def retrieve_by_category(
        self,
        category: str,
        top_k: int = 5,
    ) -> List[Tuple[PatternSignature, torch.Tensor]]:
        """Retrieve top patterns by category."""
        if category not in self.category_index:
            return []

        hashes = self.category_index[category]
        patterns = [self.patterns[h] for h in hashes if h in self.patterns]

        # Sort by quality
        patterns.sort(key=lambda p: p[0].quality_score, reverse=True)

        self.total_retrieved += len(patterns[:top_k])
        return patterns[:top_k]

    def retrieve_similar(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Tuple[PatternSignature, torch.Tensor, float]]:
        """
        Retrieve patterns similar to query embedding.

        Uses cosine similarity on the embedding cache.
        """
        # Normalize query
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)

        # Compute similarities with cache
        cache_norm = F.normalize(self.embedding_cache[:self.cache_ptr], dim=-1)
        similarities = torch.mm(query_norm, cache_norm.t()).squeeze(0)

        # Get top-k above threshold
        mask = similarities >= min_similarity
        valid_sims = similarities * mask.float() + (~mask).float() * -1

        top_indices = valid_sims.topk(min(top_k, self.cache_ptr)).indices

        results = []
        for idx in top_indices:
            idx = idx.item()
            if similarities[idx] >= min_similarity and self.cache_signatures[idx] is not None:
                sig_hash = self.cache_signatures[idx]
                if sig_hash in self.patterns:
                    sig, emb = self.patterns[sig_hash]
                    results.append((sig, emb, similarities[idx].item()))

        self.total_retrieved += len(results)
        return results

    def retrieve_best(self, top_k: int = 10) -> List[Tuple[PatternSignature, torch.Tensor]]:
        """Retrieve the highest quality patterns overall."""
        all_patterns = list(self.patterns.values())
        all_patterns.sort(key=lambda p: p[0].quality_score, reverse=True)

        self.total_retrieved += len(all_patterns[:top_k])
        return all_patterns[:top_k]

    def get_stats(self) -> Dict:
        """Get bank statistics."""
        category_counts = {
            cat: len(hashes) for cat, hashes in self.category_index.items()
        }

        return {
            'total_patterns': len(self.patterns),
            'total_added': self.total_added,
            'total_retrieved': self.total_retrieved,
            'total_evicted': self.total_evicted,
            'category_counts': category_counts,
            'cache_utilization': self.cache_ptr / self.embedding_cache_size,
        }


# =============================================================================
# PATTERN FEEDBACK LOOP
# =============================================================================

class PatternFeedbackLoop(nn.Module):
    """
    Uses stored coherent patterns to guide generation.

    Feedback Mechanism:
        1. Query pattern bank for relevant patterns
        2. Compute guidance signal from retrieved patterns
        3. Blend guidance with current representation
        4. Adjust depth based on coherence gap
    """

    def __init__(
        self,
        noetic_dim: int = 40,
        hidden_dim: int = 384,
        num_retrieved: int = 5,
        blend_strength: float = 0.3,
    ):
        super().__init__()

        self.noetic_dim = noetic_dim
        self.hidden_dim = hidden_dim
        self.num_retrieved = num_retrieved
        self.blend_strength = blend_strength

        # Pattern bank
        self.pattern_bank = CoherencePatternBank(noetic_dim)

        # Guidance projector
        self.guidance_proj = nn.Sequential(
            nn.Linear(noetic_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Blend controller (learns optimal blending)
        self.blend_controller = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Depth adjustment predictor
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim + noetic_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [decrease, maintain, increase]
        )

    def compute_guidance(
        self,
        current_repr: torch.Tensor,
        current_nf: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, List[PatternSignature]]:
        """
        Compute guidance signal from pattern bank.

        Args:
            current_repr: [batch, hidden_dim] current representation
            current_nf: Optional current NF coordinates

        Returns:
            guidance: [batch, hidden_dim] guidance signal
            retrieved: List of retrieved patterns
        """
        batch = current_repr.shape[0]
        device = current_repr.device

        # Retrieve similar patterns
        retrieved_patterns = self.pattern_bank.retrieve_similar(
            current_repr[0, :self.noetic_dim] if current_repr.shape[-1] > self.noetic_dim else current_repr[0],
            top_k=self.num_retrieved,
        )

        if not retrieved_patterns:
            # Fall back to best patterns
            best = self.pattern_bank.retrieve_best(self.num_retrieved)
            retrieved_patterns = [(s, e, 1.0) for s, e in best]

        if not retrieved_patterns:
            # No patterns available
            return torch.zeros(batch, self.hidden_dim, device=device), []

        # Aggregate guidance from retrieved patterns
        total_weight = 0.0
        guidance = torch.zeros(self.noetic_dim, device=device)

        retrieved_sigs = []
        for sig, emb, sim in retrieved_patterns:
            weight = sig.quality_score * sim
            guidance = guidance + weight * emb.to(device)
            total_weight += weight
            retrieved_sigs.append(sig)

        if total_weight > 0:
            guidance = guidance / total_weight

        # Project to hidden dim
        guidance = self.guidance_proj(guidance.unsqueeze(0))  # [1, hidden_dim]
        guidance = guidance.expand(batch, -1)  # [batch, hidden_dim]

        return guidance, retrieved_sigs

    def apply_feedback(
        self,
        hidden_state: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend guidance with current hidden state.

        Args:
            hidden_state: [batch, seq, hidden_dim]
            guidance: [batch, hidden_dim]

        Returns:
            blended: [batch, seq, hidden_dim]
        """
        batch, seq, dim = hidden_state.shape

        # Expand guidance to sequence
        guidance_expanded = guidance.unsqueeze(1).expand(-1, seq, -1)

        # Compute adaptive blend weight
        blend_input = torch.cat([
            hidden_state.mean(dim=1),
            guidance
        ], dim=-1)  # [batch, hidden_dim * 2]

        blend_weight = self.blend_controller(blend_input)  # [batch, 1]
        blend_weight = blend_weight.unsqueeze(1) * self.blend_strength  # [batch, 1, 1]

        # Blend
        blended = hidden_state + blend_weight * (guidance_expanded - hidden_state)

        return blended

    def predict_depth_adjustment(
        self,
        hidden_state: torch.Tensor,
        current_coherence: float,
        target_coherence: float = 0.7,
    ) -> int:
        """
        Predict whether to adjust processing depth.

        Returns:
            -1: Decrease depth (already coherent)
             0: Maintain depth
            +1: Increase depth (needs more processing)
        """
        # Compute coherence gap
        coherence_gap = target_coherence - current_coherence

        # Get NF representation
        pooled = hidden_state.mean(dim=(0, 1))[:self.noetic_dim]

        # Predict adjustment
        pred_input = torch.cat([
            hidden_state.mean(dim=(0, 1)),
            pooled
        ], dim=-1).unsqueeze(0)

        logits = self.depth_predictor(pred_input)
        adjustment = logits.argmax(dim=-1).item() - 1  # Map [0,1,2] to [-1,0,1]

        # Override with coherence-based heuristic
        if coherence_gap > 0.3:
            adjustment = max(adjustment, 1)  # Force increase
        elif coherence_gap < -0.2:
            adjustment = min(adjustment, -1)  # Allow decrease

        return adjustment

    def forward(
        self,
        hidden_state: torch.Tensor,
        current_nf: Optional[Tuple[int, int, int]] = None,
        current_coherence: float = 0.5,
    ) -> Tuple[torch.Tensor, int, List[PatternSignature]]:
        """
        Apply pattern feedback to hidden state.

        Args:
            hidden_state: [batch, seq, hidden_dim]
            current_nf: Current NF coordinates
            current_coherence: Current coherence score

        Returns:
            output: Feedback-adjusted hidden state
            depth_adj: Suggested depth adjustment
            retrieved: List of retrieved pattern signatures
        """
        # Compute guidance
        guidance, retrieved = self.compute_guidance(hidden_state.mean(dim=1), current_nf)

        # Apply feedback
        output = self.apply_feedback(hidden_state, guidance)

        # Predict depth adjustment
        depth_adj = self.predict_depth_adjustment(hidden_state, current_coherence)

        return output, depth_adj, retrieved


# =============================================================================
# COHERENCE MEMORY (Persistent Storage)
# =============================================================================

class CoherenceMemory:
    """
    Persistent storage for coherent patterns across sessions.

    Saves patterns to disk and loads them on initialization.
    """

    def __init__(
        self,
        storage_path: str = "checkpoints/coherence_memory",
        max_stored: int = 50000,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_stored = max_stored

        self.patterns_file = self.storage_path / "patterns.jsonl"
        self.embeddings_file = self.storage_path / "embeddings.pt"
        self.metadata_file = self.storage_path / "metadata.json"

    def save(self, pattern_bank: CoherencePatternBank):
        """Save pattern bank to disk."""
        # Save signatures
        with open(self.patterns_file, 'w') as f:
            for sig, _ in pattern_bank.patterns.values():
                f.write(json.dumps(sig.to_dict()) + '\n')

        # Save embeddings
        embeddings = {
            hash_: emb for hash_, (_, emb) in pattern_bank.patterns.items()
        }
        torch.save(embeddings, self.embeddings_file)

        # Save metadata
        metadata = {
            'total_patterns': len(pattern_bank.patterns),
            'stats': pattern_bank.get_stats(),
            'saved_at': time.time(),
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(pattern_bank.patterns)} patterns to {self.storage_path}")

    def load(self, pattern_bank: CoherencePatternBank) -> int:
        """Load patterns into pattern bank."""
        if not self.patterns_file.exists():
            return 0

        # Load signatures
        signatures = {}
        with open(self.patterns_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                sig = PatternSignature.from_dict(data)
                signatures[sig.embedding_hash] = sig

        # Load embeddings
        if self.embeddings_file.exists():
            embeddings = torch.load(self.embeddings_file)
        else:
            return 0

        # Add to pattern bank
        loaded = 0
        for hash_, sig in signatures.items():
            if hash_ in embeddings:
                emb = embeddings[hash_]
                pattern_bank.patterns[hash_] = (sig, emb)

                # Update category index
                if sig.category not in pattern_bank.category_index:
                    pattern_bank.category_index[sig.category] = []
                pattern_bank.category_index[sig.category].append(hash_)

                loaded += 1

        print(f"Loaded {loaded} patterns from {self.storage_path}")
        return loaded


# =============================================================================
# INTEGRATED LACUNARY FLOW SYSTEM
# =============================================================================

class LacunaryFlowSystem(nn.Module):
    """
    Complete integrated system for lacunary pattern flow.

    Combines all components:
        - Pattern tracking through layers
        - Quality assessment and categorization
        - Pattern storage and retrieval
        - Feedback guidance
        - Depth adjustment
        - Persistent memory
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        noetic_dim: int = 40,
        num_layers: int = 12,
        storage_path: str = "checkpoints/coherence_memory",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.noetic_dim = noetic_dim
        self.num_layers = num_layers

        # Core components
        self.tracker = LacunaryPatternTracker(
            hidden_dim=hidden_dim,
            noetic_dim=noetic_dim,
            num_layers=num_layers,
        )

        self.feedback = PatternFeedbackLoop(
            noetic_dim=noetic_dim,
            hidden_dim=hidden_dim,
        )

        # Persistent memory
        self.memory = CoherenceMemory(storage_path)

        # Load saved patterns
        self._load_patterns()

        # Integration with DPS
        self.dps_integration_enabled = True
        self.depth_adjustment_history: List[int] = []

    def _load_patterns(self):
        """Load patterns from persistent storage."""
        self.memory.load(self.feedback.pattern_bank)

    def save_patterns(self):
        """Save patterns to persistent storage."""
        self.memory.save(self.feedback.pattern_bank)

    def start_generation(self, input_coherence: float = 1.0):
        """Start tracking a new generation."""
        self.tracker.start_trace(input_coherence)
        self.depth_adjustment_history.clear()

    def process_layer(
        self,
        layer_idx: int,
        hidden_state: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        apply_feedback: bool = True,
    ) -> Tuple[torch.Tensor, LayerPattern, int]:
        """
        Process a layer through the lacunary flow system.

        Args:
            layer_idx: Current layer index
            hidden_state: [batch, seq, hidden_dim]
            prev_hidden: Previous layer's hidden state
            apply_feedback: Whether to apply pattern feedback

        Returns:
            output: Processed hidden state
            pattern: Extracted layer pattern
            depth_adj: Suggested depth adjustment
        """
        # Track pattern
        pattern = self.tracker.track_layer(layer_idx, hidden_state, prev_hidden)

        # Compute quality
        _, nf_validity, _ = self.tracker.nf_encoder(
            hidden_state.mean(dim=1)[:, :self.noetic_dim] if hidden_state.shape[-1] > self.noetic_dim else hidden_state.mean(dim=1),
            return_nf=True
        )
        quality = self.tracker.compute_pattern_quality(pattern, nf_validity[0].item())

        # Save high-quality patterns
        if quality >= 0.6:
            noetic_repr = self.tracker.layer_projs[layer_idx](hidden_state).mean(dim=(0, 1))
            self.feedback.pattern_bank.add_pattern(
                embedding=noetic_repr,
                nf_coords=pattern.nf_coords,
                layer_origin=layer_idx,
                quality_score=quality,
            )

        # Apply feedback
        depth_adj = 0
        if apply_feedback and layer_idx > 0:
            output, depth_adj, retrieved = self.feedback(
                hidden_state,
                pattern.nf_coords,
                pattern.flow_coherence,
            )
            self.depth_adjustment_history.append(depth_adj)

            # Update trace with retrieved patterns
            if self.tracker.current_trace is not None:
                for sig in retrieved:
                    self.tracker.current_trace.retrieved_patterns.append(sig)
        else:
            output = hidden_state

        return output, pattern, depth_adj

    def get_depth_recommendation(self) -> int:
        """Get overall depth recommendation based on history."""
        if not self.depth_adjustment_history:
            return 0

        # Majority vote
        total = sum(self.depth_adjustment_history)
        if total > len(self.depth_adjustment_history) * 0.3:
            return 1  # Increase
        elif total < -len(self.depth_adjustment_history) * 0.3:
            return -1  # Decrease
        return 0  # Maintain

    def finalize_generation(
        self,
        output_coherence: float,
        save_best: bool = True,
    ) -> FlowTrace:
        """
        Finalize generation and return trace.

        Args:
            output_coherence: Final output coherence score
            save_best: Whether to save best patterns

        Returns:
            Complete flow trace
        """
        trace = self.tracker.get_trace()
        if trace is not None:
            trace.output_coherence = output_coherence

            # Compute overall quality
            coherence_trajectory = trace.get_coherence_trajectory()
            if coherence_trajectory:
                avg_coherence = sum(coherence_trajectory) / len(coherence_trajectory)
                trace.quality_score = avg_coherence * output_coherence * (1 - trace.cumulative_lacunarity / self.num_layers)

        # Save patterns if generation was successful
        if save_best and output_coherence > 0.7:
            self.save_patterns()

        return trace

    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            'pattern_bank': self.feedback.pattern_bank.get_stats(),
            'depth_adjustments': {
                'history': self.depth_adjustment_history,
                'recommendation': self.get_depth_recommendation(),
            },
            'trace': self.tracker.get_trace().__dict__ if self.tracker.get_trace() else None,
        }

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        input_coherence: float = 1.0,
        output_coherence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Process complete forward pass through lacunary flow.

        Args:
            hidden_states: List of hidden states from each layer
            input_coherence: Input coherence score
            output_coherence: Output coherence score

        Returns:
            Dict with processed outputs and trace
        """
        self.start_generation(input_coherence)

        processed_states = []
        patterns = []
        depth_adjs = []

        prev_hidden = None
        for layer_idx, hidden in enumerate(hidden_states):
            output, pattern, depth_adj = self.process_layer(
                layer_idx, hidden, prev_hidden, apply_feedback=True
            )
            processed_states.append(output)
            patterns.append(pattern)
            depth_adjs.append(depth_adj)
            prev_hidden = hidden

        trace = self.finalize_generation(output_coherence)

        return {
            'processed_states': processed_states,
            'patterns': patterns,
            'depth_adjustments': depth_adjs,
            'trace': trace,
            'depth_recommendation': self.get_depth_recommendation(),
            'stats': self.get_stats(),
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("LACUNARY PATTERN FLOW SYSTEM - Demo")
    print("Canonical TKS Categorization (10 Noetics, 4 Worlds, 7 Foundations, 28 Sub-Foundations)")
    print("=" * 80)

    # Demo TKS categorization first
    print("\n" + "-" * 40)
    print("TKS Pattern Categorization Demo")
    print("-" * 40)

    test_patterns = [
        (1, 4, 7),   # Full MVR Fractal
        (1, 1, 4),   # Mind-dominant with MVR
        (2, 3, 5),   # Contains involution (2,7 pair partial)
        (8, 9, 0),   # Causal pair + Idea
        (5, 5, 6),   # Female-dominant
        (0, 0, 0),   # Pure Idea
        (7, 7, 7),   # Pure Rhythm
        (3, 6, 9),   # Involution (3,6) pair
    ]

    print("\nCanonical TKS Pattern Categories:")
    for nf in test_patterns:
        cat = PatternCategory.categorize_full(nf)
        mvr_str = " [MVR]" if cat.is_mvr else ""
        inv_str = " [INV]" if cat.is_involution else ""
        print(f"  {nf} -> nu_{cat.noetic}({cat.noetic_name})/{cat.world}({cat.world_name})/{cat.subfoundation}({cat.subfoundation_name}){mvr_str}{inv_str} Q={cat.quality_boost:.3f}")

    print(f"\nAll 28 Sub-Foundations: {PatternCategory.get_all_subfoundations()}")

    # Create system
    print("\n" + "-" * 40)
    print("Lacunary Flow System Demo")
    print("-" * 40)

    system = LacunaryFlowSystem(
        hidden_dim=384,
        noetic_dim=40,
        num_layers=12,
        storage_path="checkpoints/coherence_memory_demo",
    )

    print(f"\nSystem parameters: {sum(p.numel() for p in system.parameters()):,}")

    # Simulate layer hidden states
    batch, seq = 4, 16
    hidden_states = [
        torch.randn(batch, seq, 384) for _ in range(12)
    ]

    # Process through system
    result = system(hidden_states, input_coherence=0.8, output_coherence=0.7)

    print(f"\nProcessed {len(result['processed_states'])} layers")
    print(f"Depth adjustments: {result['depth_adjustments']}")
    print(f"Depth recommendation: {result['depth_recommendation']}")

    # Show trace
    trace = result['trace']
    if trace:
        print(f"\nFlow Trace:")
        print(f"  Input coherence: {trace.input_coherence:.3f}")
        print(f"  Output coherence: {trace.output_coherence:.3f}")
        print(f"  Cumulative lacunarity: {trace.cumulative_lacunarity:.3f}")
        print(f"  Quality score: {trace.quality_score:.3f}")
        print(f"  Coherence trajectory: {[f'{c:.2f}' for c in trace.get_coherence_trajectory()]}")

    # Show stats
    stats = result['stats']
    print(f"\nPattern Bank Stats:")
    for key, value in stats['pattern_bank'].items():
        if key != 'category_counts':
            print(f"  {key}: {value}")

    # Show patterns by TKS category
    print(f"\nPatterns by TKS Category (nu_X/World/SubF):")
    for cat, count in sorted(stats['pattern_bank'].get('category_counts', {}).items()):
        print(f"  {cat}: {count}")

    # Test retrieval
    print("\n" + "-" * 40)
    print("Pattern Retrieval Test")
    print("-" * 40)

    # Retrieve best patterns
    best = system.feedback.pattern_bank.retrieve_best(5)
    print(f"\nTop 5 patterns with TKS categorization:")
    for sig, emb in best:
        cat = PatternCategory.categorize_full(sig.nf_coords)
        print(f"  NF: {sig.nf_coords}")
        print(f"    Category: {sig.category}")
        print(f"    Noetic: {cat.noetic_name}, World: {cat.world_name}, Foundation: {cat.foundation_name}")
        print(f"    Quality: {sig.quality_score:.3f}, MVR: {cat.is_mvr}, Involution: {cat.is_involution}")

    # Show MVR fractal patterns
    print("\n" + "-" * 40)
    print("MVR Fractal Patterns (Stabilizing)")
    print("-" * 40)
    print(f"MVR Fractal: {MVR_FRACTAL} (Mind, Vibration, Rhythm)")
    print("Patterns containing 2+ MVR components get {:.0f}% quality boost".format((PatternCategory.MVR_BONUS - 1) * 100))

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)

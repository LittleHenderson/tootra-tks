"""
DPS Gating Layer Interface - Depth Permission System

This module defines the INTERFACE for the Depth Permission System (DPS) gating layer.
DPS is a LEARNED module (trainable nn.Module) that controls computational depth
based on validated novelty.

IMPORTANT: This is an INTERFACE file. Implementation is handled by agent-be.

Key Insight from strg-llm-stk:
    DPS has ~3,000+ training examples and should be implemented as a neural module
    with learnable parameters, conceptually similar to:
    - Adaptive Computation Time (ACT) from Graves
    - Universal Transformers with adaptive depth
    - PonderNet

Novelty Weight Formula:
    NW = V x I x (1 - S)
    Where:
        V = validity (verifier confidence x consistency x evidence)
        I = impact (reward gain, cost reduction, uncertainty drop)
        S = similarity to existing knowledge

Classification Thresholds:
    HEAVY  (NW >= 0.70): Immediate unlock, depth burst eligible
    COUNT  (0.45 <= NW < 0.70): Token accumulation toward unlock
    NOCOUNT (NW < 0.45): No progress, logged as "interesting"

Backward Compatibility:
    All features are optional via config flags. When use_dps_gating=False,
    v4 works exactly as before.

Author: ARCH agent
Date: 2026-01-05
Schema Version: 1.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Protocol, runtime_checkable

import torch
import torch.nn as nn


# =============================================================================
# CONSTANTS
# =============================================================================

# Classification thresholds (adjusted for untrained networks)
# Original spec: HEAVY=0.70, COUNT=0.45
# With untrained V/I/S networks outputting ~0.5, NW ≈ 0.25
# Lower thresholds to allow depth unlocking during initial training
DEFAULT_HEAVY_THRESHOLD = 0.35  # Was 0.70
DEFAULT_COUNT_THRESHOLD = 0.20  # Was 0.45
DEFAULT_TOKENS_FOR_UNLOCK = 3   # Was 5 (faster accumulation)
DEFAULT_COOLDOWN_EPISODES = 10
DEFAULT_MAX_DEPTH = 5
DEFAULT_INITIAL_P_MAX = 2


class NoveltyClass(Enum):
    """Classification of novelty weight."""
    HEAVY = "HEAVY"       # NW >= 0.70 - immediate unlock eligible
    COUNT = "COUNT"       # 0.45 <= NW < 0.70 - token accumulation
    NOCOUNT = "NOCOUNT"   # NW < 0.45 - no progress


class DPSMode(Enum):
    """Operational mode affecting DPS behavior."""
    NORMAL = "Normal"           # Standard operation, unlocks enabled
    HIGH_STAKES = "High-Stakes" # Unlocks disabled by default
    CRITICAL = "Critical"       # No unlocks, no deeper recursion


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DPSConfig:
    """
    Configuration for Depth Permission System.

    All thresholds and caps are tunable. Defaults match the strg-llm-stk spec
    recommended starting configuration.

    Attributes:
        n_tokens_threshold: Number of COUNT novelties needed for unlock
        count_threshold: NW threshold for COUNT classification (>= this)
        heavy_threshold: NW threshold for HEAVY classification (>= this)
        cooldown_k: Episodes to wait after an unlock before next unlock
        unlock_high_stakes: Whether to allow unlocks in High-Stakes mode
        unlock_critical: Whether to allow unlocks in Critical mode (should be False)
        max_depth: Hard cap on p_max (prevents runaway depth)
        initial_p_max: Starting value for p_max

        # Neural architecture config
        hidden_dim: Hidden dimension for novelty computation networks
        validity_dim: Dimension for validity feature extraction
        impact_dim: Dimension for impact feature extraction
        similarity_method: Method for computing similarity ("embedding", "ast", "hybrid")
        use_adaptive_iteration: If True, DPS controls NoeticBlock iteration count
    """
    # Threshold configuration (from spec)
    n_tokens_threshold: int = DEFAULT_TOKENS_FOR_UNLOCK
    count_threshold: float = DEFAULT_COUNT_THRESHOLD
    heavy_threshold: float = DEFAULT_HEAVY_THRESHOLD
    cooldown_k: int = DEFAULT_COOLDOWN_EPISODES
    unlock_high_stakes: bool = False  # Disabled by default per spec
    unlock_critical: bool = False     # Always disabled per spec
    max_depth: int = DEFAULT_MAX_DEPTH
    initial_p_max: int = DEFAULT_INITIAL_P_MAX

    # Neural architecture config
    hidden_dim: int = 64
    validity_dim: int = 32
    impact_dim: int = 32
    similarity_method: str = "hybrid"  # "embedding", "ast", "hybrid"
    use_adaptive_iteration: bool = True  # Controls whether DPS affects iteration count

    # Training config
    dropout: float = 0.1


# =============================================================================
# STATE
# =============================================================================

@dataclass
class DPSState:
    """
    Runtime state for Depth Permission System.

    This state persists across episodes and tracks the agent's "earned depth"
    progression. It should be serialized/deserialized for persistence.

    Attributes:
        p_max: Current allowed maximum depth (earned through novelty)
        tokens: Accumulated COUNT novelty tokens toward next unlock
        cooldown: Remaining episodes before next unlock is allowed
        mode: Current operational mode (Normal/High-Stakes/Critical)
        last_unlock_episode: Episode ID of last unlock (for audit)
        total_unlocks: Total number of unlocks (for statistics)
        novelty_history: Recent novelty weights for analysis (bounded list)
    """
    p_max: int = DEFAULT_INITIAL_P_MAX
    tokens: int = 0
    cooldown: int = 0
    mode: DPSMode = DPSMode.NORMAL
    last_unlock_episode: Optional[str] = None
    total_unlocks: int = 0
    novelty_history: List[float] = field(default_factory=list)

    def can_unlock(self, config: DPSConfig) -> bool:
        """Check if unlock is currently allowed based on state and config."""
        if self.cooldown > 0:
            return False
        if self.mode == DPSMode.CRITICAL and not config.unlock_critical:
            return False
        if self.mode == DPSMode.HIGH_STAKES and not config.unlock_high_stakes:
            return False
        if self.p_max >= config.max_depth:
            return False
        return True

    def to_dict(self) -> Dict:
        """Serialize state for persistence."""
        return {
            "p_max": self.p_max,
            "tokens": self.tokens,
            "cooldown": self.cooldown,
            "mode": self.mode.value,
            "last_unlock_episode": self.last_unlock_episode,
            "total_unlocks": self.total_unlocks,
            "novelty_history": self.novelty_history[-100:],  # Keep last 100
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DPSState":
        """Deserialize state from persistence."""
        return cls(
            p_max=data.get("p_max", DEFAULT_INITIAL_P_MAX),
            tokens=data.get("tokens", 0),
            cooldown=data.get("cooldown", 0),
            mode=DPSMode(data.get("mode", "Normal")),
            last_unlock_episode=data.get("last_unlock_episode"),
            total_unlocks=data.get("total_unlocks", 0),
            novelty_history=data.get("novelty_history", []),
        )


# =============================================================================
# NOVELTY COMPONENTS (for NW = V x I x (1-S))
# =============================================================================

@dataclass
class ValidityScore:
    """
    Validity component of novelty weight.

    V = confidence x consistency x evidence_ok

    In High-Stakes/Critical, if evidence is missing, V = 0 (HARD_FAIL).
    """
    confidence: float      # Verifier confidence [0, 1]
    consistency: float     # Canon consistency score [0, 1]
    evidence_ok: float     # Evidence binding quality [0, 1]

    @property
    def value(self) -> float:
        """Compute V = confidence x consistency x evidence_ok"""
        return self.confidence * self.consistency * self.evidence_ok


@dataclass
class ImpactScore:
    """
    Impact component of novelty weight.

    I = clip01(a*delta_reward + b*delta_uncertainty + c*delta_coverage + d*delta_cost)

    All deltas are improvements (positive = better).
    """
    delta_reward: float      # Improvement in reward
    delta_uncertainty: float # Reduction in uncertainty (positive = reduced)
    delta_coverage: float    # Increase in coverage
    delta_cost: float        # Cost savings (positive = cheaper)

    # Mixing weights (tunable)
    weight_reward: float = 0.4
    weight_uncertainty: float = 0.3
    weight_coverage: float = 0.2
    weight_cost: float = 0.1

    @property
    def value(self) -> float:
        """Compute normalized impact score."""
        raw = (
            self.weight_reward * self.delta_reward +
            self.weight_uncertainty * self.delta_uncertainty +
            self.weight_coverage * self.delta_coverage +
            self.weight_cost * self.delta_cost
        )
        return max(0.0, min(1.0, raw))


@dataclass
class SimilarityScore:
    """
    Similarity component of novelty weight.

    S = max(S_embed, S_ast) where:
        S_embed = cosine similarity to stored packet embeddings
        S_ast = AST/tree edit distance similarity for TKS expressions

    Lower similarity = higher novelty contribution (1 - S).
    """
    s_embed: float  # Embedding similarity [0, 1]
    s_ast: float    # AST/structural similarity [0, 1]

    @property
    def value(self) -> float:
        """Compute S = max(S_embed, S_ast)"""
        return max(self.s_embed, self.s_ast)


@dataclass
class NoveltyWeight:
    """
    Complete novelty weight computation.

    NW = V x I x (1 - S)
    """
    validity: ValidityScore
    impact: ImpactScore
    similarity: SimilarityScore

    @property
    def value(self) -> float:
        """Compute NW = V x I x (1 - S)"""
        return self.validity.value * self.impact.value * (1.0 - self.similarity.value)

    def classify(self, config: DPSConfig) -> NoveltyClass:
        """Classify this novelty weight."""
        nw = self.value
        if nw >= config.heavy_threshold:
            return NoveltyClass.HEAVY
        elif nw >= config.count_threshold:
            return NoveltyClass.COUNT
        else:
            return NoveltyClass.NOCOUNT


# =============================================================================
# DPS GATING LAYER INTERFACE
# =============================================================================

@runtime_checkable
class DPSGatingProtocol(Protocol):
    """
    Protocol defining the DPS gating layer interface.

    Any implementation must satisfy this protocol for integration with v4.
    """

    def forward(
        self,
        x: torch.Tensor,
        state: DPSState,
        memory_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DPSState, Dict]:
        """
        Forward pass through DPS gating.

        Args:
            x: Input tensor [batch, seq, dim] - current thought state
            state: Current DPS state (will be updated)
            memory_embeddings: Optional [num_memories, dim] stored packet embeddings
                for similarity computation

        Returns:
            output: Gated/transformed tensor [batch, seq, dim]
            new_state: Updated DPS state
            trace: Dict containing:
                - novelty_weight: float (NW value)
                - novelty_class: str ("HEAVY", "COUNT", "NOCOUNT")
                - validity: ValidityScore
                - impact: ImpactScore
                - similarity: SimilarityScore
                - depth_allowed: int (p_max from state)
                - unlock_occurred: bool
                - tokens_accumulated: int
        """
        ...

    def compute_novelty(
        self,
        x: torch.Tensor,
        memory_embeddings: Optional[torch.Tensor] = None,
    ) -> NoveltyWeight:
        """
        Compute novelty weight for input without updating state.

        This is useful for evaluation/analysis without side effects.
        """
        ...


class DPSGatingLayer(nn.Module):
    """
    Depth Permission System gating layer.

    This is a TRAINABLE neural module that computes novelty weight:
        NW = V x I x (1 - S)

    And classifies thoughts into:
        HEAVY (NW >= 0.70): Immediate unlock eligible
        COUNT (0.45 <= NW < 0.70): Token accumulation
        NOCOUNT (NW < 0.45): No progress

    The layer can operate in two modes:
        1. Output gating: Simply gate/transform the output based on novelty
        2. Adaptive iteration: Control how many NoeticBlock iterations run

    Integration with v4:
        When use_dps_gating=True in TKSNoeticLMConfig, this layer is inserted
        after RPMGating and before final output projection.

    Architecture:
        - ValidityNetwork: Extracts validity features from verifier signals
        - ImpactNetwork: Computes impact from reward/cost/uncertainty signals
        - SimilarityNetwork: Computes similarity to stored memory embeddings
        - GatingNetwork: Produces final gated output

    Trace Schema:
        When return_trace=True, adds to trace dict:
        {
            "dps": {
                "novelty_weight": float,
                "novelty_class": str,
                "validity": {"confidence": float, "consistency": float, "evidence_ok": float},
                "impact": {"delta_reward": float, "delta_uncertainty": float, ...},
                "similarity": {"s_embed": float, "s_ast": float},
                "depth_allowed": int,
                "unlock_occurred": bool,
                "tokens": int
            }
        }
    """

    def __init__(self, config: DPSConfig, noetic_dim: int = 40) -> None:
        """
        Initialize DPS gating layer.

        Args:
            config: DPS configuration
            noetic_dim: Dimension of noetic space (default 40 for TKS)
        """
        super().__init__()
        self.config = config
        self.noetic_dim = noetic_dim
        hidden_dim = config.hidden_dim
        validity_dim = config.validity_dim
        impact_dim = config.impact_dim
        dropout = config.dropout

        # Validity network: noetic_dim -> validity_dim -> 3 (confidence, consistency, evidence_ok)
        # When verifier_signals is provided, it uses those; otherwise derives from x
        self.validity_net = nn.Sequential(
            nn.Linear(noetic_dim, validity_dim),
            nn.LayerNorm(validity_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(validity_dim, validity_dim),
            nn.GELU(),
            nn.Linear(validity_dim, 3),  # confidence, consistency, evidence_ok
            nn.Sigmoid(),  # Outputs in [0, 1]
        )

        # Impact network: noetic_dim + 4 context signals -> impact_dim -> 4 (deltas)
        # Context: [reward, uncertainty, coverage, cost] from reward_signals
        self.impact_net = nn.Sequential(
            nn.Linear(noetic_dim + 4, impact_dim),
            nn.LayerNorm(impact_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(impact_dim, impact_dim),
            nn.GELU(),
            nn.Linear(impact_dim, 4),  # delta_reward, delta_uncertainty, delta_coverage, delta_cost
            nn.Sigmoid(),  # Outputs in [0, 1]
        )

        # Similarity network: computes S_embed and S_ast from comparing x to memory
        # Uses noetic_dim x 2 (x pooled + memory comparison) -> 2
        self.similarity_embed_net = nn.Sequential(
            nn.Linear(noetic_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # S_embed in [0, 1]
        )

        # AST similarity: structural comparison (simplified as learned projection)
        self.similarity_ast_net = nn.Sequential(
            nn.Linear(noetic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # S_ast in [0, 1]
        )

        # Gating network: noetic_dim + 9 novelty features -> noetic_dim
        # Features: [V, I, S, NW, 3 validity components, 2 similarity components]
        self.gating_net = nn.Sequential(
            nn.Linear(noetic_dim + 9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, noetic_dim),
        )

        # Learnable gate for blending gated output with original
        self.gate_weight = nn.Parameter(torch.tensor(0.1))

        # Learnable threshold adjustments (for fine-tuning thresholds)
        # [count_bias, heavy_bias]
        self.threshold_bias = nn.Parameter(torch.zeros(2))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_validity(
        self,
        x: torch.Tensor,
        verifier_signals: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ValidityScore:
        """
        Compute validity score from input and optional verifier signals.

        Args:
            x: Input tensor [batch, seq, dim]
            verifier_signals: Optional dict with 'confidence', 'consistency', 'evidence_ok'

        Returns:
            ValidityScore with confidence, consistency, evidence_ok
        """
        # Pool sequence to get [batch, dim]
        x_pooled = x.mean(dim=1)

        if verifier_signals is not None:
            # Use provided signals if available
            confidence = verifier_signals.get('confidence', None)
            consistency = verifier_signals.get('consistency', None)
            evidence_ok = verifier_signals.get('evidence_ok', None)

            if confidence is not None and consistency is not None and evidence_ok is not None:
                return ValidityScore(
                    confidence=confidence.mean().item(),
                    consistency=consistency.mean().item(),
                    evidence_ok=evidence_ok.mean().item(),
                )

        # Derive from x using validity network
        validity_out = self.validity_net(x_pooled)  # [batch, 3]
        validity_mean = validity_out.mean(dim=0)  # [3]

        return ValidityScore(
            confidence=validity_mean[0].item(),
            consistency=validity_mean[1].item(),
            evidence_ok=validity_mean[2].item(),
        )

    def _compute_impact(
        self,
        x: torch.Tensor,
        reward_signals: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ImpactScore:
        """
        Compute impact score from input and optional reward signals.

        Args:
            x: Input tensor [batch, seq, dim]
            reward_signals: Optional dict with reward/cost/uncertainty signals

        Returns:
            ImpactScore with delta_reward, delta_uncertainty, delta_coverage, delta_cost
        """
        x_pooled = x.mean(dim=1)  # [batch, dim]
        batch_size = x_pooled.shape[0]
        device = x_pooled.device

        # Prepare context signals
        if reward_signals is not None:
            context = torch.stack([
                reward_signals.get('reward', torch.zeros(batch_size, device=device)),
                reward_signals.get('uncertainty', torch.zeros(batch_size, device=device)),
                reward_signals.get('coverage', torch.zeros(batch_size, device=device)),
                reward_signals.get('cost', torch.zeros(batch_size, device=device)),
            ], dim=-1)  # [batch, 4]
        else:
            # Default context: neutral values
            context = torch.full((batch_size, 4), 0.5, device=device)

        # Concatenate and compute impact
        impact_input = torch.cat([x_pooled, context], dim=-1)  # [batch, dim + 4]
        impact_out = self.impact_net(impact_input)  # [batch, 4]
        impact_mean = impact_out.mean(dim=0)  # [4]

        return ImpactScore(
            delta_reward=impact_mean[0].item(),
            delta_uncertainty=impact_mean[1].item(),
            delta_coverage=impact_mean[2].item(),
            delta_cost=impact_mean[3].item(),
        )

    def _compute_similarity(
        self,
        x: torch.Tensor,
        memory_embeddings: Optional[torch.Tensor] = None,
    ) -> SimilarityScore:
        """
        Compute similarity score comparing x to stored memory embeddings.

        Args:
            x: Input tensor [batch, seq, dim]
            memory_embeddings: Optional [num_memories, dim] stored embeddings

        Returns:
            SimilarityScore with s_embed and s_ast
        """
        x_pooled = x.mean(dim=1)  # [batch, dim]

        if memory_embeddings is None or memory_embeddings.shape[0] == 0:
            # No memories: low similarity (high novelty)
            s_ast_out = self.similarity_ast_net(x_pooled).mean().item()
            return SimilarityScore(s_embed=0.0, s_ast=s_ast_out)

        # Compute embedding similarity: max cosine similarity to any memory
        x_norm = torch.nn.functional.normalize(x_pooled, p=2, dim=-1)  # [batch, dim]
        mem_norm = torch.nn.functional.normalize(memory_embeddings, p=2, dim=-1)  # [num_mem, dim]

        # Cosine similarity: [batch, num_memories]
        cos_sim = torch.matmul(x_norm, mem_norm.t())
        max_cos_sim = cos_sim.max(dim=-1)[0].mean()  # Average across batch

        # Get most similar memory for learned comparison
        best_mem_idx = cos_sim.max(dim=-1)[1]  # [batch]
        best_mem = memory_embeddings[best_mem_idx]  # [batch, dim]

        # Learned embedding similarity
        combined = torch.cat([x_pooled, best_mem], dim=-1)  # [batch, dim * 2]
        s_embed = self.similarity_embed_net(combined).mean().item()

        # Blend learned and cosine similarity
        s_embed_final = 0.5 * s_embed + 0.5 * max_cos_sim.item()

        # AST similarity (structural)
        s_ast = self.similarity_ast_net(x_pooled).mean().item()

        return SimilarityScore(s_embed=s_embed_final, s_ast=s_ast)

    def compute_novelty(
        self,
        x: torch.Tensor,
        memory_embeddings: Optional[torch.Tensor] = None,
        verifier_signals: Optional[Dict[str, torch.Tensor]] = None,
        reward_signals: Optional[Dict[str, torch.Tensor]] = None,
    ) -> NoveltyWeight:
        """
        Compute novelty weight for input without updating state.

        Args:
            x: Input tensor [batch, seq, dim]
            memory_embeddings: Optional stored packet embeddings
            verifier_signals: Optional verifier outputs
            reward_signals: Optional reward/cost signals

        Returns:
            NoveltyWeight dataclass with V, I, S components
        """
        validity = self._compute_validity(x, verifier_signals)
        impact = self._compute_impact(x, reward_signals)
        similarity = self._compute_similarity(x, memory_embeddings)

        return NoveltyWeight(
            validity=validity,
            impact=impact,
            similarity=similarity,
        )

    def compute_novelty_batch(
        self,
        x: torch.Tensor,
        memory_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute novelty weight for batch training (returns per-sample tensors).

        Unlike compute_novelty() which returns aggregated scalars for inference,
        this method returns per-sample tensors for batch loss computation.

        Args:
            x: Input tensor [batch, seq, dim] or [batch, dim]
            memory_embeddings: Optional [num_memories, dim] stored embeddings

        Returns:
            Dict with per-sample tensors:
            - 'nw': Novelty weight [batch]
            - 'v': Validity [batch]
            - 'i': Impact [batch]
            - 's': Similarity [batch]
            - 'class_logits': Classification logits [batch, 3]
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, dim]

        batch_size = x.shape[0]
        x_pooled = x.mean(dim=1)  # [batch, dim]

        # Validity: per-sample
        validity_out = self.validity_net(x_pooled)  # [batch, 3]
        v = validity_out.mean(dim=-1)  # [batch]
        v = torch.sigmoid(v)  # Normalize to [0, 1]

        # Impact: per-sample
        impact_input = torch.cat([
            x_pooled,
            torch.zeros(batch_size, 4, device=x.device)  # Placeholder signals
        ], dim=-1)
        impact_out = self.impact_net(impact_input)  # [batch, 4]
        i = impact_out.mean(dim=-1)  # [batch]
        i = torch.sigmoid(i)  # Normalize to [0, 1]

        # Similarity: per-sample
        if memory_embeddings is None or memory_embeddings.shape[0] == 0:
            s = torch.zeros(batch_size, device=x.device)
        else:
            x_norm = torch.nn.functional.normalize(x_pooled, p=2, dim=-1)
            mem_norm = torch.nn.functional.normalize(memory_embeddings, p=2, dim=-1)
            cos_sim = torch.matmul(x_norm, mem_norm.t())  # [batch, num_mem]
            s = cos_sim.max(dim=-1)[0]  # [batch] - max similarity

        # Compute NW = V * I * (1 - S)
        nw = v * i * (1.0 - s)

        # Compute class logits for HEAVY/COUNT/NOCOUNT
        # Use learned thresholds as class boundaries
        count_thresh = self.config.count_threshold + self.threshold_bias[0]
        heavy_thresh = self.config.heavy_threshold + self.threshold_bias[1]

        # Create soft class logits based on NW distance from thresholds
        # Class 0: HEAVY (nw >= heavy_thresh)
        # Class 1: COUNT (count_thresh <= nw < heavy_thresh)
        # Class 2: NOCOUNT (nw < count_thresh)
        logit_heavy = (nw - heavy_thresh) * 10  # Higher if NW > heavy_thresh
        logit_count = -torch.abs(nw - (count_thresh + heavy_thresh) / 2) * 10 + 2  # Peak at midpoint
        logit_nocount = (count_thresh - nw) * 10  # Higher if NW < count_thresh

        class_logits = torch.stack([logit_heavy, logit_count, logit_nocount], dim=-1)

        return {
            'nw': nw,
            'v': v,
            'i': i,
            's': s,
            'class_logits': class_logits,
        }

    def update_state(
        self,
        state: DPSState,
        novelty: NoveltyWeight,
        episode_id: str,
    ) -> Tuple[DPSState, bool]:
        """
        Update DPS state based on novelty classification.

        This implements the token accumulation and unlock logic:
        - HEAVY: Immediate unlock if allowed
        - COUNT: Add token, check for threshold unlock
        - NOCOUNT: No state change

        Args:
            state: Current DPS state
            novelty: Computed novelty weight
            episode_id: Current episode ID for audit

        Returns:
            new_state: Updated state (new object, immutable)
            unlock_occurred: Whether an unlock happened
        """
        # Apply learned threshold biases
        adjusted_count_threshold = self.config.count_threshold + self.threshold_bias[0].item()
        adjusted_heavy_threshold = self.config.heavy_threshold + self.threshold_bias[1].item()

        # Clamp thresholds to valid range
        adjusted_count_threshold = max(0.1, min(0.9, adjusted_count_threshold))
        adjusted_heavy_threshold = max(adjusted_count_threshold + 0.1, min(0.95, adjusted_heavy_threshold))

        nw = novelty.value
        unlock_occurred = False

        # Create new state (immutable update)
        new_p_max = state.p_max
        new_tokens = state.tokens
        new_cooldown = max(0, state.cooldown - 1)  # Decrement cooldown
        new_total_unlocks = state.total_unlocks
        new_last_unlock = state.last_unlock_episode
        new_history = state.novelty_history[-99:] + [nw]  # Keep last 100

        # Classify novelty
        if nw >= adjusted_heavy_threshold:
            # HEAVY: Immediate unlock if allowed
            if state.can_unlock(self.config):
                new_p_max = min(state.p_max + 1, self.config.max_depth)
                new_cooldown = self.config.cooldown_k
                new_tokens = 0  # Reset tokens after unlock
                new_total_unlocks = state.total_unlocks + 1
                new_last_unlock = episode_id
                unlock_occurred = True
        elif nw >= adjusted_count_threshold:
            # COUNT: Accumulate token
            new_tokens = state.tokens + 1
            # Check if tokens reached threshold for unlock
            if new_tokens >= self.config.n_tokens_threshold and state.can_unlock(self.config):
                new_p_max = min(state.p_max + 1, self.config.max_depth)
                new_cooldown = self.config.cooldown_k
                new_tokens = 0  # Reset tokens after unlock
                new_total_unlocks = state.total_unlocks + 1
                new_last_unlock = episode_id
                unlock_occurred = True
        # NOCOUNT: No state change (tokens preserved)

        new_state = DPSState(
            p_max=new_p_max,
            tokens=new_tokens,
            cooldown=new_cooldown,
            mode=state.mode,
            last_unlock_episode=new_last_unlock,
            total_unlocks=new_total_unlocks,
            novelty_history=new_history,
        )

        return new_state, unlock_occurred

    def forward(
        self,
        x: torch.Tensor,
        state: DPSState,
        memory_embeddings: Optional[torch.Tensor] = None,
        verifier_signals: Optional[Dict[str, torch.Tensor]] = None,
        reward_signals: Optional[Dict[str, torch.Tensor]] = None,
        episode_id: str = "unknown",
    ) -> Tuple[torch.Tensor, DPSState, Dict]:
        """
        Forward pass through DPS gating.

        Args:
            x: Input tensor [batch, seq, dim] - current thought state
            state: Current DPS state (will be updated and returned)
            memory_embeddings: Optional [num_memories, dim] stored packet embeddings
            verifier_signals: Optional dict with verifier outputs for validity
            reward_signals: Optional dict with reward/cost signals for impact
            episode_id: Current episode ID for audit trail

        Returns:
            output: Gated/transformed tensor [batch, seq, dim]
            new_state: Updated DPS state (immutable update - new object)
            trace: Dict with DPS trace information
        """
        batch_size, seq_len, dim = x.shape
        device = x.device

        # Compute novelty weight
        novelty = self.compute_novelty(x, memory_embeddings, verifier_signals, reward_signals)
        novelty_class = novelty.classify(self.config)

        # Update state
        new_state, unlock_occurred = self.update_state(state, novelty, episode_id)

        # Prepare novelty features for gating
        # [V, I, S, NW, confidence, consistency, evidence_ok, s_embed, s_ast]
        novelty_features = torch.tensor([
            novelty.validity.value,
            novelty.impact.value,
            novelty.similarity.value,
            novelty.value,
            novelty.validity.confidence,
            novelty.validity.consistency,
            novelty.validity.evidence_ok,
            novelty.similarity.s_embed,
            novelty.similarity.s_ast,
        ], device=device, dtype=x.dtype)

        # Expand features to match sequence: [batch, seq, 9]
        novelty_features = novelty_features.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Concatenate with input and apply gating network
        gating_input = torch.cat([x, novelty_features], dim=-1)  # [batch, seq, dim + 9]
        gated = self.gating_net(gating_input)  # [batch, seq, dim]

        # Apply learnable gate to blend gated output with original
        gate = torch.sigmoid(self.gate_weight)
        output = x + gate * gated

        # Build trace
        trace = {
            "novelty_weight": novelty.value,
            "novelty_class": novelty_class.value,
            "validity": {
                "confidence": novelty.validity.confidence,
                "consistency": novelty.validity.consistency,
                "evidence_ok": novelty.validity.evidence_ok,
            },
            "impact": {
                "delta_reward": novelty.impact.delta_reward,
                "delta_uncertainty": novelty.impact.delta_uncertainty,
                "delta_coverage": novelty.impact.delta_coverage,
                "delta_cost": novelty.impact.delta_cost,
            },
            "similarity": {
                "s_embed": novelty.similarity.s_embed,
                "s_ast": novelty.similarity.s_ast,
            },
            "depth_allowed": new_state.p_max,
            "unlock_occurred": unlock_occurred,
            "tokens": new_state.tokens,
            "iteration_budget": self.get_iteration_budget(new_state) if self.config.use_adaptive_iteration else None,
        }

        return output, new_state, trace

    def get_iteration_budget(self, state: DPSState) -> int:
        """
        Get the number of NoeticBlock iterations allowed based on state.

        When use_adaptive_iteration=True, this controls how many times
        the NoeticBlock loop runs.

        Args:
            state: Current DPS state

        Returns:
            Number of iterations allowed (based on p_max)
        """
        return state.p_max


# =============================================================================
# ADAPTIVE ITERATION CONTROLLER (Option A from spec)
# =============================================================================

class AdaptiveIterationController:
    """
    Controller for adaptive computation based on DPS.

    This implements "Option A" from the agent-arch spec where DPS controls
    how many NoeticBlock iterations run, not just gates the final output.

    Usage in v4 forward pass:
        controller = AdaptiveIterationController(dps_layer, state)
        for iteration in range(controller.max_iterations):
            x = self.noetic_block(x)
            if controller.should_stop(x, iteration):
                break
            if controller.should_extend(x):
                controller.extend_budget()
    """

    def __init__(
        self,
        dps_layer: DPSGatingLayer,
        state: DPSState,
        min_iterations: int = 1,
    ) -> None:
        """
        Initialize adaptive iteration controller.

        Args:
            dps_layer: The DPS gating layer for novelty computation
            state: Current DPS state
            min_iterations: Minimum iterations before early stopping allowed
        """
        self.dps_layer = dps_layer
        self.state = state
        self.min_iterations = min_iterations
        self._budget = state.p_max
        self._iteration = 0

    @property
    def max_iterations(self) -> int:
        """Current maximum iterations allowed."""
        return self._budget

    def should_stop(
        self,
        x: torch.Tensor,
        iteration: int,
        novelty_threshold: float = DEFAULT_HEAVY_THRESHOLD,
    ) -> bool:
        """
        Check if iteration should stop early.

        Stops if:
        1. Iteration count exceeds budget
        2. Novelty weight is very high (converged to stable novel representation)

        Args:
            x: Current state after iteration
            iteration: Current iteration number
            novelty_threshold: Threshold for early stopping

        Returns:
            True if should stop, False to continue
        """
        self._iteration = iteration

        # Always stop if we've exceeded the budget
        if iteration >= self._budget:
            return True

        # Don't stop early before minimum iterations
        if iteration < self.min_iterations:
            return False

        # Compute novelty to check for convergence
        with torch.no_grad():
            novelty = self.dps_layer.compute_novelty(x)

        # Stop early if novelty is very high (stable novel representation)
        # This means the thought has "settled" into something valuable
        if novelty.value >= novelty_threshold:
            return True

        return False

    def should_extend(
        self,
        x: torch.Tensor,
        extension_threshold: float = DEFAULT_HEAVY_THRESHOLD,
    ) -> bool:
        """
        Check if computation budget should be extended.

        If novelty is HEAVY, may earn the right to think deeper.
        Only extends if state allows unlock and we haven't hit max_depth.

        Args:
            x: Current state
            extension_threshold: NW threshold for extension eligibility

        Returns:
            True if extension is earned, False otherwise
        """
        # Check if extension is even possible
        if self._budget >= self.dps_layer.config.max_depth:
            return False

        if not self.state.can_unlock(self.dps_layer.config):
            return False

        # Compute novelty to check for extension eligibility
        with torch.no_grad():
            novelty = self.dps_layer.compute_novelty(x)

        # Extend if novelty is HEAVY
        return novelty.value >= extension_threshold

    def extend_budget(self, amount: int = 1) -> None:
        """
        Extend the iteration budget.

        Called when should_extend() returns True.

        Args:
            amount: Number of additional iterations to allow
        """
        max_allowed = self.dps_layer.config.max_depth
        self._budget = min(self._budget + amount, max_allowed)


# =============================================================================
# TRACE SCHEMA ADDITIONS
# =============================================================================

DPS_TRACE_SCHEMA = {
    "dps": {
        "novelty_weight": "float - NW = V x I x (1-S)",
        "novelty_class": "str - HEAVY | COUNT | NOCOUNT",
        "validity": {
            "confidence": "float [0,1] - verifier confidence",
            "consistency": "float [0,1] - canon consistency",
            "evidence_ok": "float [0,1] - evidence binding quality",
        },
        "impact": {
            "delta_reward": "float - reward improvement",
            "delta_uncertainty": "float - uncertainty reduction",
            "delta_coverage": "float - coverage increase",
            "delta_cost": "float - cost savings",
        },
        "similarity": {
            "s_embed": "float [0,1] - embedding similarity",
            "s_ast": "float [0,1] - AST similarity",
        },
        "depth_allowed": "int - current p_max",
        "unlock_occurred": "bool - whether unlock happened this pass",
        "tokens": "int - accumulated tokens toward next unlock",
        "iteration_budget": "int - allowed iterations (if adaptive)",
    }
}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_HEAVY_THRESHOLD",
    "DEFAULT_COUNT_THRESHOLD",
    "DEFAULT_TOKENS_FOR_UNLOCK",
    "DEFAULT_COOLDOWN_EPISODES",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_INITIAL_P_MAX",
    # Enums
    "NoveltyClass",
    "DPSMode",
    # Config and State
    "DPSConfig",
    "DPSState",
    # Score components
    "ValidityScore",
    "ImpactScore",
    "SimilarityScore",
    "NoveltyWeight",
    # Main interface
    "DPSGatingProtocol",
    "DPSGatingLayer",
    "AdaptiveIterationController",
    # Trace schema
    "DPS_TRACE_SCHEMA",
]

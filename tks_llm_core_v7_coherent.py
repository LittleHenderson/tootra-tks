"""
TKS-LLM Core Components v7-Coherent - Earned Depth with Coherence Integration

This module extends v7 with full coherence and noetic fractal integration:
1. All v6-Coherent features (LacunaryTracker, NF Encoder, CoherenceGate)
2. DPS (Depth Permission System) with coherence-aware gating
3. Coherence-driven depth adjustment
4. Pattern memory for novelty detection

Architecture:
    Input -> LacunaryTracker.start()
         -> [v7 Blocks with DPS + NJT]
              -> pattern = tracker.track_layer()
              -> coherence = compute_coherence()
              -> IF coherence LOW: increase DPS.p_max
              -> feedback.apply(pattern)
         -> NoeticFractalEncoder.encode()
         -> CoherenceGate.check()
         -> Output (or BLOCK if incoherent)

Key Innovation: Coherence-Driven Earned Depth
    - Low coherence signals need for deeper processing
    - High coherence allows shallower, faster processing
    - DPS.p_max adjusted based on coherence trajectory
    - Novel + coherent thoughts unlock maximum depth

Author: TKS Development
Date: 2026-01-13
Status: v7-Coherent Architecture
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# v6-Coherent base
from tks_llm_core_v6_coherent import (
    TKSCoherentConfig,
    CoherenceGate,
    GeneralNoeticBlockv6Coherent,
)

# v7 components
from tks_llm_core_v7 import (
    TKSGeneralConfigV7,
    GeneralNoeticBlockv7,
    DPSIntegratedRecursionController,
)

# Core components
from tks_llm_core_v5 import (
    GeneralNoeticEmbedding,
    CANONICAL_DIM
)
from tks_llm_core_v4 import (
    AttractorComputationLayer,
    RPMGatingMechanism,
)

# DPS components
from tks_features.dps_gating import (
    DPSConfig,
    DPSState,
    DPSGatingLayer,
    NoveltyClass,
    MemoryBank,
    DEFAULT_INITIAL_P_MAX,
    DEFAULT_MAX_DEPTH
)

# Coherence components
from tks_features.lacunary_pattern_flow import (
    LacunaryPatternTracker,
    PatternFeedbackLoop,
    LacunaryFlowSystem,
    FlowTrace,
    LayerPattern,
    PatternCategory,
    CanonicalLacunarity,
    TextureTuple,
)

# Noetic Fractal components
from tks_features.noetic_fractal_coherence import (
    NoeticFractalEncoder,
    LacunaryCoherenceAnalyzer,
    FractalDimensionTracker,
    CoherenceScore,
)

# Regulator integration
try:
    from tks_features.regulator_integration import (
        IntegratedRegulatorLayer,
        RegulatorConfig,
    )
    REGULATOR_AVAILABLE = True
except ImportError:
    REGULATOR_AVAILABLE = False


@dataclass
class TKSCoherentConfigV7(TKSCoherentConfig):
    """v7-Coherent Config with DPS + Coherence integration."""

    # DPS Configuration (from v7)
    use_dps_gating: bool = True
    dps_initial_p_max: int = 4  # Start at reasonable depth (was 2)
    dps_max_depth: int = 8  # Hard cap at 8
    dps_heavy_threshold: float = 0.70
    dps_count_threshold: float = 0.45
    dps_tokens_for_unlock: int = 5
    dps_cooldown_episodes: int = 10
    dps_hidden_dim: int = 64

    # v7 specific
    earned_depth_mode: bool = True
    novelty_boost_on_heavy: bool = True
    use_impact_only_nw: bool = False

    # Memory bank
    use_memory_bank: bool = True
    memory_bank_size: int = 1000
    memory_use_decay: bool = True
    memory_decay_factor: float = 0.995

    # Regulator integration
    use_regulators: bool = True
    regulator_gap_threshold: float = 0.3
    regulator_on_every_block: bool = False

    # Coherence-Driven Depth (NEW)
    coherence_drives_depth: bool = True
    coherence_depth_boost_threshold: float = 0.3  # Below this coherence, boost depth
    coherence_depth_reduce_threshold: float = 0.8  # Above this, allow shallower
    max_coherence_depth_boost: int = 2  # Max extra depth from coherence


class CoherenceAwareDPSController:
    """
    Controller that bridges DPS, RecursionStack, AND Coherence.

    Extends DPSIntegratedRecursionController with coherence awareness:
    - Low coherence can boost allowed depth (need more processing)
    - High coherence can reduce depth (already good enough)
    - Coherence trajectory affects burst activation
    """

    def __init__(
        self,
        dps_state: DPSState,
        hard_cap: int = 8,
        coherence_boost_threshold: float = 0.3,
        coherence_reduce_threshold: float = 0.8,
        max_coherence_boost: int = 2,
    ):
        self.dps_state = dps_state
        self.hard_cap = hard_cap
        self.coherence_boost_threshold = coherence_boost_threshold
        self.coherence_reduce_threshold = coherence_reduce_threshold
        self.max_coherence_boost = max_coherence_boost

        # Depth burst (from novelty)
        self._novelty_burst_active = False
        self._novelty_burst_remaining = 0

        # Coherence-driven adjustment
        self._coherence_boost = 0
        self._coherence_history: List[float] = []

    @property
    def allowed_depth(self) -> int:
        """Current maximum allowed recursion depth."""
        base = self.dps_state.p_max

        # Add novelty burst
        if self._novelty_burst_active and self._novelty_burst_remaining > 0:
            base += 1

        # Add coherence boost
        base += self._coherence_boost

        return min(base, self.hard_cap)

    def update_coherence(self, coherence_score: float):
        """Update depth based on coherence."""
        self._coherence_history.append(coherence_score)

        # Keep last 5 coherence scores
        if len(self._coherence_history) > 5:
            self._coherence_history.pop(0)

        avg_coherence = sum(self._coherence_history) / len(self._coherence_history)

        if avg_coherence < self.coherence_boost_threshold:
            # Low coherence: boost depth
            self._coherence_boost = min(
                self._coherence_boost + 1,
                self.max_coherence_boost
            )
        elif avg_coherence > self.coherence_reduce_threshold:
            # High coherence: reduce depth
            self._coherence_boost = max(self._coherence_boost - 1, 0)

    def activate_novelty_burst(self, duration: int = 1):
        """Temporarily allow extra depth from novelty."""
        self._novelty_burst_active = True
        self._novelty_burst_remaining = duration

    def tick_burst(self):
        """Decrement burst counter."""
        if self._novelty_burst_active:
            self._novelty_burst_remaining -= 1
            if self._novelty_burst_remaining <= 0:
                self._novelty_burst_active = False

    def can_push(self, current_depth: int) -> bool:
        """Check if we can push deeper."""
        return current_depth < self.allowed_depth

    def get_status(self) -> Dict:
        """Get controller status."""
        return {
            "dps_p_max": self.dps_state.p_max,
            "novelty_burst": self._novelty_burst_active,
            "coherence_boost": self._coherence_boost,
            "allowed_depth": self.allowed_depth,
            "avg_coherence": sum(self._coherence_history) / len(self._coherence_history) if self._coherence_history else 0.5,
        }


class GeneralNoeticBlockv7Coherent(nn.Module):
    """
    v7 Block with integrated DPS + Coherence.

    Combines:
    - v6 block with NJT/RecursionStack
    - DPS gating for novelty-based depth
    - Coherence tracking and feedback
    - Coherence-aware depth adjustment
    """

    def __init__(self, config: TKSCoherentConfigV7, block_idx: int = 0):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.dim = config.hidden_dim

        # Core v6-Coherent block
        self.coherent_block = GeneralNoeticBlockv6Coherent(config, block_idx)

        # DPS integration (every 3rd block)
        self.has_dps = config.use_dps_gating and (block_idx % 3 == 0)
        if self.has_dps:
            dps_config = DPSConfig(
                n_tokens_threshold=config.dps_tokens_for_unlock,
                count_threshold=config.dps_count_threshold,
                heavy_threshold=config.dps_heavy_threshold,
                cooldown_k=config.dps_cooldown_episodes,
                max_depth=config.dps_max_depth,
                initial_p_max=config.dps_initial_p_max,
                hidden_dim=config.dps_hidden_dim,
                use_impact_only_nw=config.use_impact_only_nw,
            )
            self.dps_layer = DPSGatingLayer(dps_config, noetic_dim=config.hidden_dim)
        else:
            self.dps_layer = None

        # Regulator integration (every 2nd block)
        self.has_regulator = (
            config.use_regulators and
            REGULATOR_AVAILABLE and
            (config.regulator_on_every_block or block_idx % 2 == 0)
        )
        if self.has_regulator:
            reg_config = RegulatorConfig(
                gap_threshold=config.regulator_gap_threshold,
                hidden_dim=config.dps_hidden_dim,
            )
            self.regulator = IntegratedRegulatorLayer(
                noetic_dim=min(config.hidden_dim, 40),
                config=reg_config,
            )
        else:
            self.regulator = None

        # Local coherence analyzer (for block-level coherence)
        self.local_coherence = LacunaryCoherenceAnalyzer(
            embed_dim=config.hidden_dim,
            noetic_dim=CANONICAL_DIM,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        prev_hidden: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        dps_state: Optional[DPSState] = None,
        depth_controller: Optional[CoherenceAwareDPSController] = None,
        memory_embeddings: Optional[torch.Tensor] = None,
        episode_id: str = "unknown",
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict], Optional[LayerPattern], Optional[DPSState]]:
        """
        Forward pass with DPS + Coherence.

        Args:
            x: Input [batch, seq, dim]
            attention_mask: Optional mask
            step: Training step
            prev_hidden: Previous layer hidden
            guidance: Pattern feedback
            dps_state: Current DPS state
            depth_controller: Coherence-aware depth controller
            memory_embeddings: For novelty detection
            episode_id: For DPS audit
            return_trace: Whether to return trace

        Returns:
            output: Processed tensor
            aux_loss: Combined auxiliary loss
            trace: Optional trace dict
            pattern: Optional LayerPattern
            new_dps_state: Updated DPS state
        """
        # Override max_depth if controller provided
        if depth_controller is not None:
            v6_block = self.coherent_block.v6_block
            if v6_block.njt is not None:
                original_max = v6_block.njt.njt_v6.recursion_stack.max_depth
                v6_block.njt.njt_v6.recursion_stack.max_depth = depth_controller.allowed_depth

        # Run coherent block
        out, aux_loss, trace, pattern = self.coherent_block(
            x, attention_mask, step,
            prev_hidden=prev_hidden,
            guidance=guidance,
            return_pattern=return_trace,
        )

        # Restore max_depth
        if depth_controller is not None:
            v6_block = self.coherent_block.v6_block
            if v6_block.njt is not None:
                v6_block.njt.njt_v6.recursion_stack.max_depth = original_max

        # Compute local coherence
        local_lac, _ = self.local_coherence(out.unsqueeze(0) if out.dim() == 2 else out)
        local_coherence = 1.0 - local_lac.mean().item()

        # Update depth controller with coherence
        if depth_controller is not None and self.config.coherence_drives_depth:
            depth_controller.update_coherence(local_coherence)

        # DPS gating
        new_dps_state = dps_state
        dps_trace = None

        if self.has_dps and self.dps_layer is not None and dps_state is not None:
            out, new_dps_state, dps_trace = self.dps_layer(
                out,
                dps_state,
                memory_embeddings=memory_embeddings,
                episode_id=episode_id,
            )

            # Activate depth burst on HEAVY novelty
            if dps_trace.get("unlock_occurred", False) and depth_controller is not None:
                if self.config.novelty_boost_on_heavy:
                    depth_controller.activate_novelty_burst(duration=1)

        # Apply regulators
        regulator_trace = None
        if self.has_regulator and self.regulator is not None:
            noetic_dim = min(out.shape[-1], 40)
            out_noetic = out[..., :noetic_dim]

            out_regulated, gap_info = self.regulator(out_noetic, return_gap_info=return_trace)

            out = out.clone()
            out[..., :noetic_dim] = out_regulated

            if gap_info is not None:
                reg_loss = self.regulator.compute_regulator_loss(gap_info)
                if isinstance(aux_loss, torch.Tensor):
                    aux_loss = aux_loss + reg_loss * 0.1
                else:
                    aux_loss = reg_loss * 0.1

            if return_trace and gap_info is not None:
                regulator_trace = gap_info

        # Build combined trace
        if return_trace:
            trace = trace or {}
            trace.update({
                "local_coherence": local_coherence,
                "dps": dps_trace,
                "regulator": regulator_trace,
                "depth_status": depth_controller.get_status() if depth_controller else None,
            })

        return out, aux_loss, trace, pattern, new_dps_state


class TKSGeneralLMv7Coherent(nn.Module):
    """
    Coherence-Integrated TKS LM v7 with Earned Depth.

    Full integration of:
    - Coherence tracking and gating (from v6-Coherent)
    - DPS novelty-based depth control (from v7)
    - Coherence-driven depth adjustment (NEW)
    - Pattern memory for both coherence and novelty

    Architecture:
        Input -> Embedding
             -> init DPS state
             -> init depth controller (coherence-aware)
             -> start_trace()
             -> [12 x v7-Coherent Blocks]
                  -> track_layer()
                  -> compute_coherence()
                  -> update depth_controller
                  -> DPS gating (novelty)
                  -> apply_feedback()
             -> Canonical Projection
             -> Attractor + RPM
             -> CoherenceGate (final check)
             -> LM Head
             -> finalize_trace()
        Output

    Depth Control Flow:
        1. DPS.p_max starts at initial value (2)
        2. HEAVY novelty boosts p_max (+1)
        3. COUNT novelty accumulates toward boost
        4. LOW coherence adds temporary boost (need more thinking)
        5. HIGH coherence allows reduction (already good)
        6. Hard cap prevents infinite recursion
    """

    def __init__(self, config: TKSCoherentConfigV7):
        super().__init__()
        self.config = config

        # Embedding
        self.embedding = GeneralNoeticEmbedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2048, config.hidden_dim))

        # v7-Coherent blocks
        self.blocks = nn.ModuleList([
            GeneralNoeticBlockv7Coherent(config, block_idx=i)
            for i in range(config.num_layers)
        ])

        # Canonical projection
        self.to_canonical = nn.Linear(config.hidden_dim, CANONICAL_DIM)
        self.from_canonical = nn.Linear(CANONICAL_DIM, config.hidden_dim)

        # Attractor and RPM
        if config.use_attractor:
            self.attractor = AttractorComputationLayer(
                dim=CANONICAL_DIM,
                contraction_factor=config.contraction_factor,
                max_iterations=config.max_attractor_iter
            )
        else:
            self.attractor = None

        if config.use_rpm:
            self.rpm = RPMGatingMechanism(dim=CANONICAL_DIM)
        else:
            self.rpm = None

        # Full lacunary flow system
        if config.use_coherence_tracking:
            self.flow_system = LacunaryFlowSystem(
                hidden_dim=config.hidden_dim,
                noetic_dim=CANONICAL_DIM,
                num_layers=config.num_layers,
            )
        else:
            self.flow_system = None

        # Coherence gate
        if config.coherence_gate_enabled:
            self.coherence_gate = CoherenceGate(config)
        else:
            self.coherence_gate = None

        # Memory bank for novelty detection
        if config.use_memory_bank:
            self.memory_bank = MemoryBank(
                capacity=config.memory_bank_size,
                dim=config.hidden_dim,
            )
        else:
            self.memory_bank = None

        # Output
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def _init_dps_state(self, batch_size: int, device: torch.device) -> DPSState:
        """Initialize DPS state for a batch."""
        return DPSState(
            p_max=self.config.dps_initial_p_max,
            tokens=0,
            cooldown=0,
            total_unlocks=0,
            novelty_history=[],
        )

    def _init_depth_controller(self, dps_state: DPSState) -> CoherenceAwareDPSController:
        """Initialize coherence-aware depth controller."""
        return CoherenceAwareDPSController(
            dps_state=dps_state,
            hard_cap=self.config.dps_max_depth,
            coherence_boost_threshold=self.config.coherence_depth_boost_threshold,
            coherence_reduce_threshold=self.config.coherence_depth_reduce_threshold,
            max_coherence_boost=self.config.max_coherence_depth_boost,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_full_trace: bool = False,
        step: Optional[int] = None,
        episode_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Forward pass with DPS + Coherence integration.

        Args:
            input_ids: [batch, seq] input tokens
            attention_mask: Optional mask
            return_full_trace: Whether to return detailed trace
            step: Training step
            episode_id: Episode ID for DPS audit

        Returns:
            Dict with:
                - logits: Output logits
                - coherence_score: Final coherence
                - dps_state: Final DPS state
                - depth_status: Depth controller status
                - routing_aux_loss: Combined aux loss
                - trace: Detailed trace (if requested)
        """
        b, s = input_ids.shape
        device = input_ids.device

        # Embed
        x = self.embedding(input_ids)
        x = x + self.pos_emb[:, :s, :]

        # Initialize DPS and depth controller
        dps_state = self._init_dps_state(b, device)
        depth_controller = self._init_depth_controller(dps_state)

        # Get memory embeddings for novelty detection
        memory_embeddings = None
        if self.memory_bank is not None:
            memory_embeddings = self.memory_bank.get_memory()

        # Start flow tracking
        if self.flow_system is not None:
            self.flow_system.start_generation(input_coherence=1.0)

        # Process through blocks
        trace_layers = []
        patterns = []
        total_aux_loss = 0.0
        prev_hidden = None

        for i, block in enumerate(self.blocks):
            # Get guidance from flow system
            guidance = None
            if self.flow_system is not None and i > 0:
                current_nf = patterns[-1].nf_coords if patterns else None
                current_coherence = patterns[-1].flow_coherence if patterns else 0.5
                guidance, _ = self.flow_system.feedback.compute_guidance(
                    x.mean(dim=1), current_nf
                )

            # Forward through block
            x, aux_loss, trace, pattern, dps_state = block(
                x, attention_mask, step,
                prev_hidden=prev_hidden,
                guidance=guidance,
                dps_state=dps_state,
                depth_controller=depth_controller,
                memory_embeddings=memory_embeddings,
                episode_id=episode_id,
                return_trace=return_full_trace,
            )

            if aux_loss is not None:
                if isinstance(total_aux_loss, torch.Tensor):
                    total_aux_loss = total_aux_loss + aux_loss
                else:
                    total_aux_loss = aux_loss

            if pattern is not None:
                patterns.append(pattern)

                # Process through flow system
                if self.flow_system is not None:
                    self.flow_system.process_layer(i, x, prev_hidden, apply_feedback=False)

            if return_full_trace:
                trace_layers.append(trace)

            # Tick depth burst
            depth_controller.tick_burst()
            prev_hidden = x.detach()

        # Final normalization
        x = self.final_norm(x)

        # Canonical projection
        canon_x = self.to_canonical(x)
        canon_trace = {}

        if self.attractor is not None:
            att_out = self.attractor(canon_x, return_trajectory=return_full_trace)
            canon_x = att_out["attractor"]
            if return_full_trace:
                canon_trace["attractor"] = att_out

        if self.rpm is not None:
            rpm_out = self.rpm(canon_x)
            canon_x = rpm_out["gated_output"]
            if return_full_trace:
                canon_trace["rpm"] = rpm_out

        # Project back
        x = x + self.from_canonical(canon_x)

        # Coherence gate
        coherence_score = None
        if self.coherence_gate is not None:
            x, coherence_score = self.coherence_gate(x, return_score=True)

        # Output logits
        logits = self.head(x)

        # Update memory bank with this thought
        if self.memory_bank is not None:
            thought_embedding = x.mean(dim=1).detach()  # [batch, dim]
            for b_idx in range(b):
                self.memory_bank.add(thought_embedding[b_idx])

        # Finalize flow trace
        flow_trace = None
        if self.flow_system is not None:
            output_coherence = coherence_score.overall if coherence_score else 0.5
            flow_trace = self.flow_system.finalize_generation(output_coherence)

        # Build output
        out = {
            "logits": logits,
            "coherence_score": coherence_score,
            "dps_state": dps_state,
            "depth_status": depth_controller.get_status(),
        }

        if isinstance(total_aux_loss, torch.Tensor):
            out["routing_aux_loss"] = total_aux_loss

        if return_full_trace:
            out["trace"] = {
                "layers": trace_layers,
                "patterns": patterns,
                "canonical": canon_trace,
                "flow_trace": flow_trace,
                "memory_bank_size": self.memory_bank.size if self.memory_bank else 0,
            }

        return out

    def generate_coherent(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        coherence_threshold: float = 0.5,
        max_retries: int = 3,
        temperature: float = 1.0,
        episode_id: str = "generation",
    ) -> Tuple[torch.Tensor, List[CoherenceScore], Dict]:
        """
        Generate tokens with coherence + novelty awareness.

        Args:
            input_ids: Starting tokens
            max_new_tokens: Maximum tokens to generate
            coherence_threshold: Minimum coherence for acceptance
            max_retries: Retries for low-coherence tokens
            temperature: Sampling temperature
            episode_id: Episode ID for tracking

        Returns:
            generated_ids: Full sequence
            coherence_history: Coherence at each step
            depth_history: Depth status at each step
        """
        self.eval()
        device = input_ids.device

        generated = input_ids.clone()
        coherence_history = []
        depth_history = []

        with torch.no_grad():
            for t in range(max_new_tokens):
                # Forward pass
                outputs = self(
                    generated,
                    return_full_trace=False,
                    episode_id=f"{episode_id}_t{t}"
                )
                logits = outputs["logits"][:, -1, :] / temperature
                coherence = outputs["coherence_score"]
                depth_status = outputs["depth_status"]

                # Sample token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Retry if low coherence
                retries = 0
                while coherence is not None and coherence.overall < coherence_threshold and retries < max_retries:
                    next_token = torch.multinomial(probs, num_samples=1)
                    test_seq = torch.cat([generated, next_token], dim=1)
                    test_out = self(test_seq, return_full_trace=False)
                    coherence = test_out["coherence_score"]
                    retries += 1

                # Append
                generated = torch.cat([generated, next_token], dim=1)
                if coherence is not None:
                    coherence_history.append(coherence)
                depth_history.append(depth_status)

                # Check EOS
                if next_token.item() == 2:
                    break

        return generated, coherence_history, depth_history


def test_v7_coherent():
    """Test the v7-Coherent model."""
    print("=" * 70)
    print("TKS v7-COHERENT MODEL TEST")
    print("(Earned Depth + Coherence Integration)")
    print("=" * 70)

    # Create config
    config = TKSCoherentConfigV7(
        vocab_size=16384,
        hidden_dim=384,
        num_layers=6,  # Fewer layers for testing
        use_coherence_tracking=True,
        coherence_gate_enabled=True,
        use_dps_gating=True,
        use_memory_bank=True,
        coherence_drives_depth=True,
    )

    # Create model
    model = TKSGeneralLMv7Coherent(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    print("\n1. Forward Pass Test")
    print("-" * 40)
    output = model(input_ids, return_full_trace=True, episode_id="test_001")

    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Coherence: {output['coherence_score']}")
    print(f"   DPS state: p_max={output['dps_state'].p_max}")
    print(f"   Depth status: {output['depth_status']}")

    # Check trace
    if 'trace' in output:
        trace = output['trace']
        print(f"   Patterns tracked: {len(trace['patterns'])}")
        print(f"   Memory bank size: {trace['memory_bank_size']}")

        for i, layer_trace in enumerate(trace['layers'][:3]):
            if layer_trace:
                print(f"     Layer {i}: coherence={layer_trace.get('local_coherence', 'N/A'):.3f}")

    print("\n2. Coherence-Driven Depth Test")
    print("-" * 40)
    depth_status = output['depth_status']
    print(f"   DPS p_max: {depth_status['dps_p_max']}")
    print(f"   Coherence boost: {depth_status['coherence_boost']}")
    print(f"   Novelty burst: {depth_status['novelty_burst']}")
    print(f"   Allowed depth: {depth_status['allowed_depth']}")
    print(f"   Avg coherence: {depth_status['avg_coherence']:.3f}")

    print("\n3. Memory Bank Test")
    print("-" * 40)
    if model.memory_bank is not None:
        print(f"   Memory bank size: {model.memory_bank.size}")
        print(f"   Memory bank capacity: {model.memory_bank.capacity}")

    print("\n4. DPS State Test")
    print("-" * 40)
    dps = output['dps_state']
    print(f"   p_max: {dps.p_max}")
    print(f"   tokens: {dps.tokens}")
    print(f"   total_unlocks: {dps.total_unlocks}")

    print("\n[OK] v7-Coherent model functional!")
    return model


if __name__ == "__main__":
    test_v7_coherent()

"""
TKS-LLM Core Components v6-Coherent - Reasoning with Coherence Integration

This module extends v6 with full integration of:
1. LacunaryPatternTracker - Monitors pattern flow through layers
2. NoeticFractalEncoder - Maps embeddings to X:Y:Z coordinates
3. CoherenceGate - Blocks gibberish before output
4. PatternFeedbackLoop - Uses stored patterns to guide generation

Architecture:
    Input -> LacunaryTracker.start()
         -> [v6 Blocks with NJT]
              -> pattern = tracker.track_layer()
              -> feedback.apply(pattern)
         -> NoeticFractalEncoder.encode()
         -> CoherenceGate.check()
         -> Output (or BLOCK if incoherent)

The coherence system provides:
- Real-time gibberish detection via lacunarity analysis
- Semantic validation via NF coordinate mapping
- Pattern memory for consistency across generation
- Depth adjustment signals when coherence drops

Author: TKS Development
Date: 2026-01-13
Status: v6-Coherent Architecture
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use v6 components
from tks_llm_core_v6 import (
    TKSGeneralConfig,
    GeneralNoeticBlockv6,
    TKSGeneralLMv6,
    CANONICAL_DIM
)

from tks_llm_core import NOETIC_DIM, NUM_WORLDS
from tks_llm_core_v4 import (
    AttractorComputationLayer,
    RPMGatingMechanism,
)
from tks_llm_core_v5 import (
    GeneralNoeticEmbedding,
    CANONICAL_DIM
)

# Coherence components
from tks_features.lacunary_pattern_flow import (
    LacunaryPatternTracker,
    PatternFeedbackLoop,
    CoherencePatternBank,
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
    COHERENT_ANCHORS,
    INCOHERENT_PATTERNS,
)


@dataclass
class TKSCoherentConfig(TKSGeneralConfig):
    """v6-Coherent Config with coherence system integration."""

    # Coherence System
    use_coherence_tracking: bool = True
    coherence_threshold: float = 0.5  # Below this = gibberish
    coherence_gate_enabled: bool = True  # Block incoherent outputs

    # Lacunary Analysis
    lacunary_gap_threshold: float = 2.0
    lacunary_window_size: int = 5
    lacunary_decay_factor: float = 0.9

    # Noetic Fractal
    nf_temperature: float = 1.0
    nf_hard_selection: bool = False  # Use soft NF during training

    # Pattern Feedback
    use_pattern_feedback: bool = True
    feedback_blend_strength: float = 0.3
    pattern_bank_size: int = 10000
    pattern_min_quality: float = 0.6

    # Depth Adjustment
    coherence_depth_adjustment: bool = True
    depth_increase_threshold: float = 0.3  # Coherence gap to trigger depth increase
    depth_decrease_threshold: float = 0.8  # Coherence level to allow depth decrease


class CoherenceGate(nn.Module):
    """
    Gates output based on coherence analysis.

    Blocks or attenuates outputs that fail coherence checks:
    1. High lacunarity index (gappy embeddings)
    2. Invalid NF coordinates (incoherent patterns)
    3. Collapsed fractal dimension (hallucination)
    """

    def __init__(self, config: TKSCoherentConfig):
        super().__init__()
        self.config = config

        # Coherence components
        self.lacunary_analyzer = LacunaryCoherenceAnalyzer(
            embed_dim=config.hidden_dim,
            noetic_dim=CANONICAL_DIM,
            gap_threshold=config.lacunary_gap_threshold,
            window_size=config.lacunary_window_size,
        )

        self.nf_encoder = NoeticFractalEncoder(
            embed_dim=config.hidden_dim,
            noetic_dim=CANONICAL_DIM,
            temperature=config.nf_temperature,
        )

        self.fractal_tracker = FractalDimensionTracker(
            num_scales=4,
            min_dimension=0.5,
            max_dimension=2.5,
            target_dimension=1.5,
        )

        # Gate computation
        self.gate_proj = nn.Sequential(
            nn.Linear(4, 32),  # 4 coherence signals
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Fallback projection (for when output is gated)
        self.fallback_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def compute_coherence_score(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> CoherenceScore:
        """
        Compute comprehensive coherence score.

        Args:
            hidden_states: [batch, seq, dim]
            attention_weights: Optional [batch, heads, seq, seq]

        Returns:
            CoherenceScore with all components
        """
        batch = hidden_states.shape[0]
        device = hidden_states.device

        # 1. Lacunary index (embedding gap analysis)
        lacunary_idx, lac_diag = self.lacunary_analyzer(hidden_states, return_diagnostics=True)

        # 2. NF validity (semantic coherence)
        x_probs, y_probs, z_probs, nf_validity = self.nf_encoder.encode(hidden_states)

        # Get discrete NF for diagnostics
        x_idx = x_probs.argmax(dim=-1)
        y_idx = y_probs.argmax(dim=-1)
        z_idx = z_probs.argmax(dim=-1)

        # 3. Fractal dimension (attention coherence)
        if attention_weights is not None:
            frac_coherence, frac_dim = self.fractal_tracker(attention_weights, return_dimension=True)
        else:
            frac_coherence = torch.ones(batch, device=device)
            frac_dim = torch.full((batch,), 1.5, device=device)

        # 4. Semantic stability (check for incoherent NF patterns)
        stability = torch.ones(batch, device=device)
        collapse_detected = torch.zeros(batch, dtype=torch.bool, device=device)
        gibberish_detected = torch.zeros(batch, dtype=torch.bool, device=device)

        for b in range(batch):
            nf = (x_idx[b].item(), y_idx[b].item(), z_idx[b].item())
            if nf in INCOHERENT_PATTERNS:
                stability[b] = 0.3
                gibberish_detected[b] = True
            elif nf in COHERENT_ANCHORS:
                stability[b] = 1.2  # Boost for known good patterns

            # Check for dimension collapse
            if frac_dim[b] < 0.5:
                collapse_detected[b] = True

        # Compute overall score
        # Weight: lacunary (30%), NF validity (30%), fractal (20%), stability (20%)
        overall = (
            0.3 * (1 - lacunary_idx) +
            0.3 * nf_validity +
            0.2 * frac_coherence +
            0.2 * stability.clamp(0, 1)
        )

        return CoherenceScore(
            lacunary_index=lacunary_idx.mean().item(),
            fractal_dimension=frac_dim.mean().item(),
            nf_validity=nf_validity.mean().item(),
            semantic_stability=stability.mean().item(),
            overall=overall.mean().item(),
            detected_nf=(x_idx[0].item(), y_idx[0].item(), z_idx[0].item()),
            collapse_detected=collapse_detected.any().item(),
            gibberish_detected=gibberish_detected.any().item(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        return_score: bool = False,
    ) -> Tuple[torch.Tensor, Optional[CoherenceScore]]:
        """
        Apply coherence gate to hidden states.

        Args:
            hidden_states: [batch, seq, dim]
            attention_weights: Optional attention weights
            return_score: Whether to return coherence score

        Returns:
            gated_output: Coherence-gated hidden states
            score: Optional CoherenceScore
        """
        if not self.config.coherence_gate_enabled:
            return hidden_states, None

        # Compute coherence
        score = self.compute_coherence_score(hidden_states, attention_weights)

        # Compute gate value
        batch = hidden_states.shape[0]
        device = hidden_states.device

        gate_input = torch.tensor([
            1 - score.lacunary_index,
            score.nf_validity,
            score.fractal_dimension / 2.0,  # Normalize to ~0-1
            score.semantic_stability,
        ], device=device).unsqueeze(0).expand(batch, -1)

        gate = self.gate_proj(gate_input)  # [batch, 1]
        gate = gate.unsqueeze(1)  # [batch, 1, 1] for broadcasting

        # Apply gate
        if score.overall < self.config.coherence_threshold:
            # Low coherence: blend with fallback
            fallback = self.fallback_proj(hidden_states)
            output = gate * hidden_states + (1 - gate) * fallback
        else:
            # High coherence: pass through with slight attenuation
            output = hidden_states * (0.9 + 0.1 * gate)

        if return_score:
            return output, score
        return output, None


class GeneralNoeticBlockv6Coherent(nn.Module):
    """
    v6 Block with integrated coherence tracking.

    Extends GeneralNoeticBlockv6 with:
    - Pattern tracking at layer output
    - Feedback from stored patterns
    - Coherence-based depth adjustment signals
    """

    def __init__(self, config: TKSCoherentConfig, block_idx: int = 0):
        super().__init__()
        self.config = config
        self.block_idx = block_idx

        # Core v6 block
        self.v6_block = GeneralNoeticBlockv6(config)

        # Pattern tracking projection (to noetic dim)
        self.pattern_proj = nn.Linear(config.hidden_dim, CANONICAL_DIM)

        # Feedback integration
        if config.use_pattern_feedback:
            self.feedback_gate = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.feedback_gate = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        prev_hidden: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        return_pattern: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict], Optional[LayerPattern]]:
        """
        Forward pass with coherence tracking.

        Args:
            x: Input [batch, seq, dim]
            attention_mask: Optional mask
            step: Training step
            prev_hidden: Previous layer hidden for gap computation
            guidance: Pattern feedback guidance
            return_pattern: Whether to return layer pattern

        Returns:
            output: Processed tensor
            aux_loss: Routing auxiliary loss
            trace: Optional trace dict
            pattern: Optional LayerPattern
        """
        # Apply feedback if available
        if guidance is not None and self.feedback_gate is not None:
            guidance_expanded = guidance.unsqueeze(1).expand(-1, x.shape[1], -1)
            blend_input = torch.cat([x.mean(dim=1), guidance], dim=-1)
            blend_weight = self.feedback_gate(blend_input).unsqueeze(1)
            x = x + blend_weight * self.config.feedback_blend_strength * (guidance_expanded - x)

        # Run v6 block
        out, r_weights, aux_loss, metrics, njt_trace = self.v6_block(
            x, attention_mask, step, return_njt_trace=return_pattern
        )

        # Compute pattern if requested
        pattern = None
        if return_pattern:
            # Project to noetic space
            noetic_repr = self.pattern_proj(out)  # [batch, seq, noetic_dim]

            # Compute gap from previous
            if prev_hidden is not None:
                prev_noetic = self.pattern_proj(prev_hidden)
                gap = (noetic_repr - prev_noetic).norm(dim=-1)
            else:
                gap = torch.zeros(out.shape[0], out.shape[1], device=out.device)

            # Compute flow coherence (simple version)
            flow_coherence = 1.0 - gap.mean().item() / 10.0
            flow_coherence = max(0.0, min(1.0, flow_coherence))

            # Get NF coordinates
            pooled = noetic_repr.mean(dim=1)
            x_probs, y_probs, z_probs, validity = NoeticFractalEncoder(
                embed_dim=CANONICAL_DIM,
                noetic_dim=CANONICAL_DIM,
            ).to(out.device).encode(pooled)

            x_idx = x_probs.argmax(dim=-1)[0].item()
            y_idx = y_probs.argmax(dim=-1)[0].item()
            z_idx = z_probs.argmax(dim=-1)[0].item()

            category, _ = PatternCategory.categorize((x_idx, y_idx, z_idx))

            pattern = LayerPattern(
                layer_idx=self.block_idx,
                hidden_state=out.detach(),
                gap_from_previous=gap.detach(),
                flow_coherence=flow_coherence,
                nf_coords=(x_idx, y_idx, z_idx),
                category=category,
            )

        trace = {
            'routing_weights': r_weights,
            'metrics': metrics,
            'njt_trace': njt_trace,
        } if return_pattern else None

        return out, aux_loss, trace, pattern


class TKSGeneralLMv6Coherent(nn.Module):
    """
    Coherence-Integrated TKS LM v6.

    Full integration of coherence and noetic fractal systems:
    - LacunaryPatternTracker monitors all layers
    - PatternFeedbackLoop guides generation
    - CoherenceGate validates output
    - Pattern bank stores high-quality patterns

    Architecture:
        Input -> Embedding
             -> start_trace()
             -> [12 x CoherentBlock]
                  -> track_layer()
                  -> apply_feedback()
             -> Canonical Projection
             -> Attractor + RPM
             -> CoherenceGate
             -> LM Head
             -> finalize_trace()
        Output
    """

    def __init__(self, config: TKSCoherentConfig):
        super().__init__()
        self.config = config

        # Embedding layers
        self.embedding = GeneralNoeticEmbedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, config.hidden_dim))

        # Coherent blocks
        self.blocks = nn.ModuleList([
            GeneralNoeticBlockv6Coherent(config, block_idx=i)
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

        # Coherence system
        if config.use_coherence_tracking:
            self.pattern_tracker = LacunaryPatternTracker(
                hidden_dim=config.hidden_dim,
                noetic_dim=CANONICAL_DIM,
                num_layers=config.num_layers,
                decay_factor=config.lacunary_decay_factor,
            )

            self.pattern_feedback = PatternFeedbackLoop(
                noetic_dim=CANONICAL_DIM,
                hidden_dim=config.hidden_dim,
                blend_strength=config.feedback_blend_strength,
            )
        else:
            self.pattern_tracker = None
            self.pattern_feedback = None

        # Coherence gate
        if config.coherence_gate_enabled:
            self.coherence_gate = CoherenceGate(config)
        else:
            self.coherence_gate = None

        # Output layers
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_full_trace: bool = False,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with coherence integration.

        Args:
            input_ids: [batch, seq] input token IDs
            attention_mask: Optional attention mask
            return_full_trace: Whether to return detailed trace
            step: Training step for schedulers

        Returns:
            Dict with:
                - logits: Output logits
                - coherence_score: Overall coherence
                - routing_aux_loss: Routing loss
                - trace: Detailed trace (if requested)
                - depth_adjustment: Suggested depth change
        """
        b, s = input_ids.shape
        device = input_ids.device

        # Embed
        x = self.embedding(input_ids)
        x = x + self.pos_emb[:, :s, :]

        # Start coherence tracking
        if self.pattern_tracker is not None:
            self.pattern_tracker.start_trace(input_coherence=1.0)

        # Process through blocks
        trace_layers = []
        patterns = []
        total_aux_loss = 0.0
        prev_hidden = None

        for i, block in enumerate(self.blocks):
            # Get guidance from pattern feedback
            guidance = None
            if self.pattern_feedback is not None and i > 0:
                current_nf = patterns[-1].nf_coords if patterns else None
                current_coherence = patterns[-1].flow_coherence if patterns else 0.5
                guidance, _ = self.pattern_feedback.compute_guidance(
                    x.mean(dim=1), current_nf
                )

            # Forward through block
            x, aux_loss, trace, pattern = block(
                x, attention_mask, step,
                prev_hidden=prev_hidden,
                guidance=guidance,
                return_pattern=return_full_trace or self.config.use_coherence_tracking,
            )

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

            if pattern is not None:
                patterns.append(pattern)

                # Track in pattern system
                if self.pattern_tracker is not None:
                    tracked = self.pattern_tracker.track_layer(i, x, prev_hidden)

                    # Save high-quality patterns
                    if pattern.flow_coherence > self.config.pattern_min_quality:
                        noetic_repr = self.pattern_tracker.layer_projs[i](x).mean(dim=(0, 1))
                        self.pattern_feedback.pattern_bank.add_pattern(
                            embedding=noetic_repr,
                            nf_coords=pattern.nf_coords,
                            layer_origin=i,
                            quality_score=pattern.flow_coherence,
                        )

            if return_full_trace:
                trace_layers.append(trace)

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

        # Project back and add residual
        x = x + self.from_canonical(canon_x)

        # Apply coherence gate
        coherence_score = None
        if self.coherence_gate is not None:
            x, coherence_score = self.coherence_gate(x, return_score=True)

        # Output logits
        logits = self.head(x)

        # Compute depth adjustment
        depth_adjustment = 0
        if coherence_score is not None and self.config.coherence_depth_adjustment:
            if coherence_score.overall < self.config.depth_increase_threshold:
                depth_adjustment = 1  # Need more processing
            elif coherence_score.overall > self.config.depth_decrease_threshold:
                depth_adjustment = -1  # Can reduce processing

        # Finalize trace
        flow_trace = None
        if self.pattern_tracker is not None:
            output_coherence = coherence_score.overall if coherence_score else 0.5
            flow_trace = self.pattern_tracker.current_trace
            if flow_trace is not None:
                flow_trace.output_coherence = output_coherence

        # Build output
        out = {
            "logits": logits,
            "coherence_score": coherence_score,
            "depth_adjustment": depth_adjustment,
        }

        if isinstance(total_aux_loss, torch.Tensor):
            out["routing_aux_loss"] = total_aux_loss

        if return_full_trace:
            out["trace"] = {
                "layers": trace_layers,
                "patterns": patterns,
                "canonical": canon_trace,
                "flow_trace": flow_trace,
            }

        return out

    def generate_coherent(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        coherence_threshold: float = 0.5,
        max_retries: int = 3,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, List[CoherenceScore]]:
        """
        Generate tokens with coherence validation.

        Args:
            input_ids: Starting tokens
            max_new_tokens: Maximum tokens to generate
            coherence_threshold: Minimum coherence for acceptance
            max_retries: Retries for low-coherence tokens
            temperature: Sampling temperature

        Returns:
            generated_ids: Full sequence with generated tokens
            coherence_history: Coherence scores for each step
        """
        self.eval()
        device = input_ids.device

        generated = input_ids.clone()
        coherence_history = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self(generated, return_full_trace=False)
                logits = outputs["logits"][:, -1, :] / temperature
                coherence = outputs["coherence_score"]

                # Sample token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check coherence and retry if needed
                retries = 0
                while coherence is not None and coherence.overall < coherence_threshold and retries < max_retries:
                    # Try different token
                    next_token = torch.multinomial(probs, num_samples=1)
                    test_seq = torch.cat([generated, next_token], dim=1)
                    test_out = self(test_seq, return_full_trace=False)
                    coherence = test_out["coherence_score"]
                    retries += 1

                # Append token
                generated = torch.cat([generated, next_token], dim=1)
                if coherence is not None:
                    coherence_history.append(coherence)

                # Check for EOS (assuming token 2 is EOS)
                if next_token.item() == 2:
                    break

        return generated, coherence_history


def test_v6_coherent():
    """Test the coherence-integrated v6 model."""
    print("=" * 70)
    print("TKS v6-COHERENT MODEL TEST")
    print("=" * 70)

    # Create config
    config = TKSCoherentConfig(
        vocab_size=16384,
        hidden_dim=384,
        num_layers=4,  # Fewer layers for testing
        use_coherence_tracking=True,
        coherence_gate_enabled=True,
        use_pattern_feedback=True,
    )

    # Create model
    model = TKSGeneralLMv6Coherent(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    print("\n1. Forward Pass Test")
    print("-" * 40)
    output = model(input_ids, return_full_trace=True)

    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Coherence score: {output['coherence_score']}")
    print(f"   Depth adjustment: {output['depth_adjustment']}")

    # Check trace
    if 'trace' in output:
        trace = output['trace']
        print(f"   Patterns tracked: {len(trace['patterns'])}")
        for i, p in enumerate(trace['patterns']):
            print(f"     Layer {i}: NF={p.nf_coords}, coherence={p.flow_coherence:.3f}")

    print("\n2. Coherence Gate Test")
    print("-" * 40)
    if output['coherence_score'] is not None:
        score = output['coherence_score']
        print(f"   Lacunary index: {score.lacunary_index:.3f}")
        print(f"   NF validity: {score.nf_validity:.3f}")
        print(f"   Fractal dimension: {score.fractal_dimension:.3f}")
        print(f"   Semantic stability: {score.semantic_stability:.3f}")
        print(f"   Overall: {score.overall:.3f}")
        print(f"   Detected NF: {score.detected_nf}")
        print(f"   Gibberish: {score.gibberish_detected}")
        print(f"   Collapse: {score.collapse_detected}")

    print("\n3. Pattern Bank Test")
    print("-" * 40)
    if model.pattern_feedback is not None:
        stats = model.pattern_feedback.pattern_bank.get_stats()
        print(f"   Total patterns: {stats['total_patterns']}")
        print(f"   Total added: {stats['total_added']}")

    print("\n[OK] v6-Coherent model functional!")
    return model


if __name__ == "__main__":
    test_v6_coherent()

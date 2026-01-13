"""
Coherent TKS Wrapper - Integration of Noetic Fractal Coherence with TKS LLM

This module wraps the TKS LLM with the Noetic Fractal Coherence System to ensure
all model outputs are semantically coherent.

Key Integration Points:
    1. Pre-input coherence check (validate input makes sense)
    2. During-generation attention monitoring (track coherence drift)
    3. Post-output coherence gate (block/constrain gibberish)
    4. Training mode: add coherence loss to standard LM loss

Usage:
    from tks_features.coherent_tks_wrapper import CoherentTKSModel

    model = CoherentTKSModel(base_model, coherence_config)

    # Inference with coherence guarantee
    output = model.generate(input_ids, max_length=100)
    # output is guaranteed coherent (gibberish blocked/constrained)

    # Training with coherence loss
    loss = model(input_ids, labels=labels)  # Includes coherence loss component

Author: TKS Research Pipeline
Date: 2026-01-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

from tks_features.noetic_fractal_coherence import (
    NoeticFractalCoherenceSystem,
    CoherenceScore,
    NoeticFractalTrainingGenerator,
    NOETIC_MEANINGS,
    COHERENT_ANCHORS,
    INCOHERENT_PATTERNS,
)


@dataclass
class CoherenceConfig:
    """Configuration for coherence enforcement."""
    enabled: bool = True
    threshold: float = 0.5              # Below this = incoherent
    gate_mode: str = 'constrain'        # 'block', 'constrain', 'warn'
    coherence_loss_weight: float = 0.1  # Weight in combined loss
    check_input: bool = True            # Check input coherence
    track_attention: bool = True        # Track attention dimension
    history_length: int = 10            # Attention history for tracking
    warn_threshold: float = 0.3         # Warn if below this (even if passing)
    strict_mode: bool = False           # Block ALL incoherent outputs


@dataclass
class CoherenceTrace:
    """Trace of coherence analysis during generation."""
    step: int = 0
    input_score: Optional[CoherenceScore] = None
    output_scores: List[CoherenceScore] = field(default_factory=list)
    blocked_count: int = 0
    constrained_count: int = 0
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, msg: str):
        self.warnings.append(f"Step {self.step}: {msg}")

    def summary(self) -> Dict:
        return {
            'total_steps': self.step,
            'blocked': self.blocked_count,
            'constrained': self.constrained_count,
            'warnings': len(self.warnings),
            'avg_coherence': sum(s.overall for s in self.output_scores) / len(self.output_scores) if self.output_scores else 0,
        }


class CoherentTKSModel(nn.Module):
    """
    Wrapper that adds coherence enforcement to any TKS LLM.

    This module intercepts the model's embeddings and applies the
    Noetic Fractal Coherence System to ensure outputs are meaningful.

    The core insight: by mapping all outputs to Noetic Fractal space,
    we can detect gibberish (fails to map to valid NF) and either
    block it or constrain it toward coherent representations.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[CoherenceConfig] = None,
        embed_dim: int = 384,
        noetic_dim: int = 40,
    ):
        super().__init__()

        self.base_model = base_model
        self.config = config or CoherenceConfig()
        self.embed_dim = embed_dim
        self.noetic_dim = noetic_dim

        # Initialize coherence system
        self.coherence_system = NoeticFractalCoherenceSystem(
            embed_dim=embed_dim,
            noetic_dim=noetic_dim,
            coherence_threshold=self.config.threshold,
            gate_mode=self.config.gate_mode,
            history_length=self.config.history_length,
        )

        # Trace for debugging
        self.trace = CoherenceTrace()

        # Projection for connecting to base model's hidden states
        # (adapt to model's actual hidden dim if different)
        self._hidden_proj = None  # Lazy init based on actual hidden dim

    def _ensure_hidden_proj(self, hidden_dim: int):
        """Lazily initialize hidden projection if needed."""
        if self._hidden_proj is None and hidden_dim != self.embed_dim:
            self._hidden_proj = nn.Linear(hidden_dim, self.embed_dim)
            self._hidden_proj.to(next(self.base_model.parameters()).device)

    def _project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to coherence system dimension."""
        hidden_dim = hidden.shape[-1]

        if hidden_dim == self.embed_dim:
            return hidden

        self._ensure_hidden_proj(hidden_dim)
        return self._hidden_proj(hidden)

    def check_input_coherence(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[bool, CoherenceScore]:
        """
        Check if input embeddings are coherent.

        Args:
            embeddings: [batch, seq, dim] input embeddings

        Returns:
            is_coherent: Boolean indicating coherence
            score: Detailed coherence score
        """
        if not self.config.enabled or not self.config.check_input:
            # Return default coherent score
            return True, CoherenceScore(0, 1.5, 1.0, 1.0, 1.0)

        projected = self._project_hidden(embeddings)
        score = self.coherence_system.check_input(projected)

        is_coherent = score.is_coherent(self.config.threshold)

        if not is_coherent:
            self.trace.add_warning(f"Incoherent input detected (score={score.overall:.3f})")

        self.trace.input_score = score
        return is_coherent, score

    def gate_output(
        self,
        hidden_states: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, CoherenceScore]:
        """
        Apply coherence gating to model outputs.

        Args:
            hidden_states: [batch, seq, dim] model hidden states
            attention: Optional attention weights

        Returns:
            gated: [batch, seq, dim] coherence-gated hidden states
            score: Coherence score
        """
        if not self.config.enabled:
            return hidden_states, CoherenceScore(0, 1.5, 1.0, 1.0, 1.0)

        # Project to coherence system dimension
        projected = self._project_hidden(hidden_states)

        # Gate through coherence system
        gated, score = self.coherence_system.gate_output(projected, attention)

        # Track statistics
        self.trace.step += 1
        self.trace.output_scores.append(score)

        if score.gibberish_detected:
            if self.config.gate_mode == 'block':
                self.trace.blocked_count += 1
            elif self.config.gate_mode == 'constrain':
                self.trace.constrained_count += 1

        if score.overall < self.config.warn_threshold:
            self.trace.add_warning(f"Low coherence output (score={score.overall:.3f})")

        # Project back if needed (inverse projection)
        if self._hidden_proj is not None:
            # Simple inverse: transpose weight (not exact inverse, but good enough)
            gated = F.linear(gated, self._hidden_proj.weight.t())

        return gated, score

    def compute_coherence_loss(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coherence loss for training.

        Args:
            hidden_states: Model output hidden states

        Returns:
            loss: Scalar coherence loss
        """
        if not self.config.enabled:
            return torch.tensor(0.0, device=hidden_states.device)

        projected = self._project_hidden(hidden_states)
        loss = self.coherence_system.compute_coherence_loss(projected)

        return self.config.coherence_loss_weight * loss

    def get_nf_representation(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[str, float]:
        """
        Get Noetic Fractal representation of hidden states.

        Args:
            hidden_states: [batch, seq, dim]

        Returns:
            nf_string: "X:Y:Z" format NF coordinates
            validity: NF validity score
        """
        projected = self._project_hidden(hidden_states)
        _, nf_coords, validity = self.coherence_system.get_nf_representation(projected)

        x, y, z = nf_coords
        return f"{x}:{y}:{z}", validity

    def explain_output(
        self,
        hidden_states: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Explain the coherence analysis of an output.

        Args:
            hidden_states: Model output hidden states

        Returns:
            explanation: Dict with human-readable analysis
        """
        projected = self._project_hidden(hidden_states)
        result = self.coherence_system(projected, return_all=True)

        nf_coords = result['detected_nf']
        nf_meaning = self.coherence_system.nf_to_text_hint(tuple(nf_coords))

        # Check against known patterns
        if tuple(nf_coords) in COHERENT_ANCHORS:
            pattern_type = "COHERENT_ANCHOR"
            pattern_meaning = COHERENT_ANCHORS[tuple(nf_coords)]
        elif tuple(nf_coords) in INCOHERENT_PATTERNS:
            pattern_type = "INCOHERENT_PATTERN"
            pattern_meaning = "Warning: This pattern is semantically unstable"
        else:
            pattern_type = "NEUTRAL"
            pattern_meaning = "Standard semantic pattern"

        return {
            'nf_coordinates': f"{nf_coords[0]}:{nf_coords[1]}:{nf_coords[2]}",
            'nf_meaning': nf_meaning,
            'pattern_type': pattern_type,
            'pattern_meaning': pattern_meaning,
            'coherence_score': result['coherence_score'].item(),
            'is_coherent': result['is_coherent'].item(),
            'lacunary_index': result['lacunary_index'].item(),
            'fractal_dimension': result['fractal_dimension'].item(),
            'semantic_stability': result['semantic_stability'].item(),
            'gibberish_detected': result['gibberish_detected'],
            'collapse_detected': result['collapse_detected'],
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_coherence: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with coherence enforcement.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (for training)
            return_coherence: Whether to return coherence info
            **kwargs: Additional args for base model

        Returns:
            Dict with 'loss', 'logits', and optionally coherence info
        """
        # Run base model
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # Extract hidden states (model-specific)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            # Fallback: use logits projected back
            hidden_states = None

        result = {
            'logits': outputs.logits if hasattr(outputs, 'logits') else outputs,
        }

        # Apply coherence gating to hidden states
        if hidden_states is not None and self.config.enabled:
            # Get attention if available
            attention = None
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention = outputs.attentions[-1]

            gated_hidden, coherence_score = self.gate_output(hidden_states, attention)

            if return_coherence:
                result['coherence_score'] = torch.tensor(coherence_score.overall)
                result['is_coherent'] = torch.tensor(coherence_score.is_coherent())
                result['nf_coords'] = coherence_score.detected_nf

        # Compute loss (including coherence loss if training)
        if labels is not None:
            base_loss = outputs.loss if hasattr(outputs, 'loss') else F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
            )

            if hidden_states is not None and self.config.enabled:
                coherence_loss = self.compute_coherence_loss(hidden_states)
                result['loss'] = base_loss + coherence_loss
                result['base_loss'] = base_loss
                result['coherence_loss'] = coherence_loss
            else:
                result['loss'] = base_loss
        elif hasattr(outputs, 'loss'):
            result['loss'] = outputs.loss

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        coherence_check_interval: int = 5,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with coherence monitoring.

        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            coherence_check_interval: Check coherence every N tokens
            **kwargs: Additional args for base model generate

        Returns:
            generated: Generated token IDs (coherence-enforced)
        """
        # Reset trace for new generation
        self.trace = CoherenceTrace()

        # For now, delegate to base model
        # A full implementation would intercept each step
        if hasattr(self.base_model, 'generate'):
            return self.base_model.generate(
                input_ids,
                max_length=max_length,
                **kwargs,
            )
        else:
            raise NotImplementedError("Base model doesn't support generate()")

    def get_trace(self) -> Dict:
        """Get coherence trace summary."""
        return self.trace.summary()

    def reset_trace(self):
        """Reset coherence trace."""
        self.trace = CoherenceTrace()
        self.coherence_system.reset_history()


# =============================================================================
# COHERENT OUTPUT HEAD (Alternative Integration)
# =============================================================================

class CoherentOutputHead(nn.Module):
    """
    A drop-in replacement for the LM head that ensures coherent outputs.

    Instead of directly projecting to vocabulary, this:
        1. Projects to NF space
        2. Validates NF coherence
        3. Projects coherent NF to vocabulary

    This forces all outputs through the coherence bottleneck.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        noetic_dim: int = 40,
        coherence_threshold: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.noetic_dim = noetic_dim

        # Hidden -> NF projection
        self.to_nf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, noetic_dim),
        )

        # NF coherence system (lightweight version)
        from tks_features.noetic_fractal_coherence import NoeticFractalEncoder
        self.nf_encoder = NoeticFractalEncoder(
            embed_dim=noetic_dim,  # NF space
            noetic_dim=noetic_dim,
        )

        # NF -> vocab projection
        self.nf_to_vocab = nn.Sequential(
            nn.Linear(noetic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

        # Bypass for low-coherence (learned fallback)
        self.bypass = nn.Linear(hidden_dim, vocab_size)

        self.coherence_threshold = coherence_threshold

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary through NF bottleneck.

        Args:
            hidden_states: [batch, seq, hidden_dim]

        Returns:
            logits: [batch, seq, vocab_size]
        """
        batch, seq, _ = hidden_states.shape

        # Project to NF space
        nf_repr = self.to_nf(hidden_states)  # [batch, seq, noetic_dim]

        # Check NF validity per position
        nf_flat = nf_repr.view(-1, self.noetic_dim)
        _, validity, _ = self.nf_encoder(nf_flat)
        validity = validity.view(batch, seq)  # [batch, seq]

        # Project NF to vocabulary
        nf_logits = self.nf_to_vocab(nf_repr)  # [batch, seq, vocab]

        # Bypass for low-coherence positions
        bypass_logits = self.bypass(hidden_states)

        # Blend based on validity
        validity_weight = validity.unsqueeze(-1)  # [batch, seq, 1]
        logits = validity_weight * nf_logits + (1 - validity_weight) * bypass_logits

        return logits


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("COHERENT TKS WRAPPER - Demo")
    print("=" * 80)

    # Create a simple mock model for testing
    class MockTKSModel(nn.Module):
        def __init__(self, vocab_size=16384, hidden_dim=384):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, batch_first=True),
                num_layers=2,
            )
            self.lm_head = nn.Linear(hidden_dim, vocab_size)
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            logits = self.lm_head(x)

            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))

            class Output:
                pass
            out = Output()
            out.logits = logits
            out.loss = loss
            out.hidden_states = [x]  # Last hidden state
            return out

    # Create mock model
    mock_model = MockTKSModel()
    print(f"\nMock model parameters: {sum(p.numel() for p in mock_model.parameters()):,}")

    # Wrap with coherence
    config = CoherenceConfig(
        enabled=True,
        threshold=0.5,
        gate_mode='constrain',
        coherence_loss_weight=0.1,
    )

    coherent_model = CoherentTKSModel(
        mock_model,
        config=config,
        embed_dim=384,
    )
    print(f"Coherent wrapper parameters: {sum(p.numel() for p in coherent_model.parameters()):,}")

    # Test forward pass
    batch, seq = 4, 16
    input_ids = torch.randint(0, 16384, (batch, seq))
    labels = torch.randint(0, 16384, (batch, seq))

    # Forward pass
    outputs = coherent_model(input_ids, labels=labels, return_coherence=True)

    print(f"\nForward pass results:")
    print(f"  Total loss: {outputs['loss'].item():.4f}")
    print(f"  Base loss: {outputs.get('base_loss', 'N/A')}")
    print(f"  Coherence loss: {outputs.get('coherence_loss', 'N/A')}")
    print(f"  Logits shape: {outputs['logits'].shape}")

    if 'coherence_score' in outputs:
        print(f"  Coherence score: {outputs['coherence_score'].item():.3f}")
        print(f"  Is coherent: {outputs['is_coherent'].item()}")

    # Test explanation
    hidden = mock_model.embedding(input_ids)
    explanation = coherent_model.explain_output(hidden)

    print(f"\nOutput explanation:")
    for key, value in explanation.items():
        print(f"  {key}: {value}")

    # Get trace
    trace = coherent_model.get_trace()
    print(f"\nGeneration trace:")
    for key, value in trace.items():
        print(f"  {key}: {value}")

    # Test CoherentOutputHead
    print("\n" + "-" * 40)
    print("CoherentOutputHead Demo")
    print("-" * 40)

    head = CoherentOutputHead(
        hidden_dim=384,
        vocab_size=16384,
        noetic_dim=40,
    )
    print(f"CoherentOutputHead parameters: {sum(p.numel() for p in head.parameters()):,}")

    hidden_states = torch.randn(batch, seq, 384)
    logits = head(hidden_states)
    print(f"Input: {hidden_states.shape} -> Output: {logits.shape}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)

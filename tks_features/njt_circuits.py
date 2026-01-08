"""
NJT (Noetic Judgment Transistor) Circuits for TKS LLM

Implements consciousness amplifier/dampener circuits based on transistor model:
- NJT+ (Excitatory): Amplifies signals when bias exceeds threshold
- NJT- (Inhibitory): Dampens signals when bias exceeds threshold

Mathematical Specification:
    Gate function:  g(B) = sigmoid(k * (p_B_modified - threshold))
    where p_B_modified = p_B + alpha2 * N2 - alpha3 * N3

    NJT+ output: p_E = clamp(beta * p_C * g(B), 0, 1)
    NJT- output: p_E = clamp(beta * p_C * (1 - g(B)), 0, 1)

Badges:
    +N2: Positive bias offset (lowers threshold)
    -N3: Negative bias offset (raises threshold)
    xN4: Gain multiplier (amplification factor)
    N5:  Hysteresis memory (different on/off thresholds)
    N6:  Delivery gate (output forwarding)
    N7:  Rhythm control (timing/frequency)
    N8->N9: Trigger-response pairing

Author: TKS Development
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NJTConfig:
    """Configuration for NJT circuits."""
    dim: int = 384
    num_transistors: int = 10
    initial_gain: float = 2.0
    initial_threshold: float = 0.5
    gate_sharpness: float = 10.0  # k parameter in sigmoid
    n2_weight: float = 0.2  # +N2 bias offset
    n3_weight: float = 0.2  # -N3 bias offset
    use_hysteresis: bool = True
    hysteresis_gap: float = 0.3  # Gap between on/off thresholds
    use_rhythm: bool = False
    rhythm_period: int = 10


class NoeticGate(nn.Module):
    """
    Gate function for NJT circuits.

    g(B) = sigmoid(k * (p_B_modified - threshold))
    where p_B_modified = p_B + alpha2 * N2 - alpha3 * N3
    """

    def __init__(self, num_gates: int, config: NJTConfig):
        super().__init__()
        self.config = config

        # Learnable thresholds per gate
        self.threshold = nn.Parameter(
            torch.ones(num_gates) * config.initial_threshold
        )

        # Badge modifiers (learnable)
        self.n2_modifier = nn.Parameter(torch.zeros(num_gates))  # +N2
        self.n3_modifier = nn.Parameter(torch.zeros(num_gates))  # -N3

    def forward(self, bias: torch.Tensor) -> torch.Tensor:
        """
        Compute gate activation.

        Args:
            bias: Bias signal [batch, seq, num_gates] or [batch, num_gates]

        Returns:
            Gate activation in [0, 1]
        """
        # Apply badge modifiers
        # p_B_modified = p_B + alpha2 * N2 - alpha3 * N3
        modified_bias = (
            bias
            + self.config.n2_weight * torch.sigmoid(self.n2_modifier)
            - self.config.n3_weight * torch.sigmoid(self.n3_modifier)
        )

        # Gate function: g(B) = sigmoid(k * (p_B_modified - threshold))
        gate = torch.sigmoid(
            self.config.gate_sharpness * (modified_bias - self.threshold)
        )

        return gate


class NJTPlus(nn.Module):
    """
    NJT+ (Positive/Excitatory Transistor)

    Amplifies the input signal when bias exceeds threshold.

    Output = clamp(beta * input * g(bias), 0, 1)

    Circuit diagram:
        C ──[p_C]──┤ NJT+ ├──[p_E]──→ E
                   │      │
             Bias ─┴──────┘
    """

    def __init__(self, dim: int, num_transistors: int, config: NJTConfig):
        super().__init__()
        self.dim = dim
        self.num_transistors = num_transistors
        self.config = config

        # Input projection (Cause -> internal)
        self.cause_proj = nn.Linear(dim, num_transistors)

        # Bias projection
        self.bias_proj = nn.Linear(dim, num_transistors)

        # Gate function
        self.gate = NoeticGate(num_transistors, config)

        # Gain (beta) - learnable per transistor (N4 badge)
        self.gain = nn.Parameter(
            torch.ones(num_transistors) * config.initial_gain
        )

        # Output projection (internal -> Effect)
        self.effect_proj = nn.Linear(num_transistors, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_internals: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through NJT+ transistor bank.

        Args:
            x: Input signal (Cause) [batch, seq, dim]
            context: Context for bias (defaults to x if None)
            return_internals: Whether to return internal states

        Returns:
            output: Amplified signal [batch, seq, dim]
            internals: Dict of internal states (if return_internals)
        """
        if context is None:
            context = x

        # Project to transistor space
        cause = torch.sigmoid(self.cause_proj(x))  # [batch, seq, num_transistors]
        bias = torch.sigmoid(self.bias_proj(context))  # [batch, seq, num_transistors]

        # Compute gate activation
        gate = self.gate(bias)  # [batch, seq, num_transistors]

        # NJT+ formula: output = beta * input * g(bias)
        internal_out = self.gain * cause * gate

        # Clamp to [0, 1]
        internal_out = torch.clamp(internal_out, 0, 1)

        # Project back to full dimension
        output = self.effect_proj(internal_out)

        if return_internals:
            internals = {
                'cause': cause,
                'bias': bias,
                'gate': gate,
                'gain': self.gain,
                'internal_output': internal_out,
            }
            return output, internals

        return output, None


class NJTMinus(nn.Module):
    """
    NJT- (Negative/Inhibitory Transistor)

    Dampens the input signal when bias exceeds threshold.

    Output = clamp(beta * input * (1 - g(bias)), 0, 1)

    Circuit diagram:
        C ──[p_C]──┤ NJT- ├──[p_E]──→ E (dampened)
                   │      │
             Bias ─┴──────┘ (brake)
    """

    def __init__(self, dim: int, num_transistors: int, config: NJTConfig):
        super().__init__()
        self.dim = dim
        self.num_transistors = num_transistors
        self.config = config

        # Input projection (Cause -> internal)
        self.cause_proj = nn.Linear(dim, num_transistors)

        # Bias projection (brake signal)
        self.bias_proj = nn.Linear(dim, num_transistors)

        # Gate function
        self.gate = NoeticGate(num_transistors, config)

        # Gain (beta)
        self.gain = nn.Parameter(
            torch.ones(num_transistors) * config.initial_gain
        )

        # Output projection
        self.effect_proj = nn.Linear(num_transistors, dim)

    def forward(
        self,
        x: torch.Tensor,
        brake_signal: Optional[torch.Tensor] = None,
        return_internals: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through NJT- transistor bank.

        Args:
            x: Input signal (Cause) [batch, seq, dim]
            brake_signal: Signal to apply braking (defaults to x if None)
            return_internals: Whether to return internal states

        Returns:
            output: Dampened signal [batch, seq, dim]
            internals: Dict of internal states (if return_internals)
        """
        if brake_signal is None:
            brake_signal = x

        # Project to transistor space
        cause = torch.sigmoid(self.cause_proj(x))
        bias = torch.sigmoid(self.bias_proj(brake_signal))

        # Compute gate activation
        gate = self.gate(bias)

        # NJT- formula: output = beta * input * (1 - g(bias))
        # Higher gate = more braking = less output
        internal_out = self.gain * cause * (1 - gate)

        # Clamp to [0, 1]
        internal_out = torch.clamp(internal_out, 0, 1)

        # Project back to full dimension
        output = self.effect_proj(internal_out)

        if return_internals:
            internals = {
                'cause': cause,
                'bias': bias,
                'gate': gate,
                'brake_strength': 1 - gate,
                'gain': self.gain,
                'internal_output': internal_out,
            }
            return output, internals

        return output, None


class HysteresisMemory(nn.Module):
    """
    N5 Badge - Hysteresis Memory

    Creates "sticky" states with different thresholds for activation vs deactivation.
    Once a circuit activates, it takes more effort to deactivate it (and vice versa).

    This models habit formation and commitment to reasoning paths.

    activate_threshold > deactivate_threshold creates positive hysteresis:
    - Hard to start, but once started, hard to stop
    """

    def __init__(self, num_units: int, gap: float = 0.3):
        super().__init__()
        self.num_units = num_units

        # Base threshold
        self.base_threshold = nn.Parameter(torch.ones(num_units) * 0.5)

        # Gap determines hysteresis strength
        self.gap = nn.Parameter(torch.ones(num_units) * gap)

        # State tracking (not a parameter - runtime state)
        self.register_buffer('state', torch.zeros(num_units))
        self.register_buffer('initialized', torch.tensor(False))

    @property
    def activate_threshold(self) -> torch.Tensor:
        """Threshold to turn ON (higher = harder to start)."""
        return self.base_threshold + self.gap / 2

    @property
    def deactivate_threshold(self) -> torch.Tensor:
        """Threshold to turn OFF (lower = harder to stop)."""
        return self.base_threshold - self.gap / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hysteresis to input signal.

        Args:
            x: Input signal [batch, seq, num_units] or [batch, num_units]

        Returns:
            Hysteresis-modulated signal
        """
        # Get thresholds
        act_thresh = self.activate_threshold
        deact_thresh = self.deactivate_threshold

        # Determine new state based on hysteresis
        # If currently OFF: need to exceed activate_threshold to turn ON
        # If currently ON: need to drop below deactivate_threshold to turn OFF

        if not self.initialized:
            # First pass - use activate threshold
            new_state = (x > act_thresh).float()
            self.initialized.fill_(True)
        else:
            # Apply hysteresis
            should_activate = (x > act_thresh).float()
            should_stay_on = (x > deact_thresh).float()

            # New state: activate if above act_thresh, OR stay on if above deact_thresh
            new_state = torch.max(
                should_activate,
                self.state * should_stay_on
            )

        # Update state (detached to avoid backprop through state)
        self.state = new_state.detach().mean(dim=tuple(range(new_state.dim() - 1)))

        # Modulate input by state
        return x * new_state

    def reset_state(self):
        """Reset hysteresis state."""
        self.state.zero_()
        self.initialized.fill_(False)


class DifferentialPair(nn.Module):
    """
    Differential Pair - Two competing NJT+ circuits

    Models decision-making between competing options.
    When options are balanced, small biases tip the balance dramatically.

    Circuit diagram:
              ┌──┤ NJT+ ├──→ Choice A
    Input ────┤
              └──┤ NJT+ ├──→ Choice B

    Winner = argmax(beta * g(B))
    """

    def __init__(self, dim: int, num_options: int, config: NJTConfig):
        super().__init__()
        self.dim = dim
        self.num_options = num_options
        self.config = config

        # One NJT+ per option
        self.njt_bank = nn.ModuleList([
            NJTPlus(dim, 1, config) for _ in range(num_options)
        ])

        # Shared attention/resource budget
        self.attention_budget = nn.Parameter(torch.ones(1))

        # Temperature for softmax competition
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        option_biases: Optional[torch.Tensor] = None,
        return_internals: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Compete options and select winner.

        Args:
            x: Input signal [batch, seq, dim]
            option_biases: Per-option bias adjustments [batch, seq, num_options]
            return_internals: Whether to return competition details

        Returns:
            winner_output: Output from winning circuit
            winner_weights: Softmax weights showing which option won
            internals: Competition details (if return_internals)
        """
        activations = []
        outputs = []

        for i, njt in enumerate(self.njt_bank):
            # Apply option-specific bias if provided
            if option_biases is not None:
                bias = x + option_biases[..., i:i+1].expand_as(x)
            else:
                bias = x

            out, _ = njt(x, bias)
            outputs.append(out)

            # Activation strength = mean of output
            activation = out.abs().mean(dim=-1, keepdim=True)
            activations.append(activation)

        # Stack activations [batch, seq, num_options]
        activations = torch.cat(activations, dim=-1)

        # Winner-takes-all via softmax
        winner_weights = F.softmax(
            activations / self.temperature,
            dim=-1
        )

        # Weighted combination of outputs
        outputs = torch.stack(outputs, dim=-1)  # [batch, seq, dim, num_options]
        winner_output = (outputs * winner_weights.unsqueeze(-2)).sum(dim=-1)

        if return_internals:
            internals = {
                'activations': activations,
                'winner_weights': winner_weights,
                'temperature': self.temperature,
                'individual_outputs': outputs,
            }
            return winner_output, winner_weights, internals

        return winner_output, winner_weights, None


class RhythmOscillator(nn.Module):
    """
    N7 Badge - Rhythm/Oscillator Circuit

    Creates self-sustaining rhythmic patterns for flow states.

    Circuit:
        ┌──────────────[Feedback]──────────────┐
        ↓                                       ↑
    Start──┤ NJT+ ├──[Delay]──[NOT]──[Clamp]───┘
           │  N7  │
           └──────┘
    """

    def __init__(self, dim: int, config: NJTConfig):
        super().__init__()
        self.dim = dim
        self.config = config
        self.period = config.rhythm_period

        # Core NJT+ for amplification
        self.njt = NJTPlus(dim, config.num_transistors, config)

        # Delay implemented as GRU
        self.delay = nn.GRU(dim, dim, batch_first=True)

        # Phase modulation
        self.phase_embed = nn.Embedding(config.rhythm_period, dim)

        # Output clamp
        self.clamp = nn.Hardtanh(0, 1)

    def forward(
        self,
        seed: torch.Tensor,
        num_steps: int = 10,
        return_sequence: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate rhythmic output sequence.

        Args:
            seed: Initial signal [batch, dim]
            num_steps: Number of oscillation steps
            return_sequence: Whether to return full sequence

        Returns:
            output: Final or full sequence output
            phases: Phase indices used (if return_sequence)
        """
        batch_size = seed.shape[0]
        outputs = []
        hidden = None
        x = seed.unsqueeze(1)  # [batch, 1, dim]

        for step in range(num_steps):
            # Get phase embedding
            phase_idx = torch.tensor([step % self.period], device=seed.device)
            phase = self.phase_embed(phase_idx).unsqueeze(0).expand(batch_size, 1, -1)

            # NJT+ amplification with phase as bias
            x_amp, _ = self.njt(x, phase)

            # Delay (GRU step)
            x_delayed, hidden = self.delay(x_amp, hidden)

            # Feedback with inversion (NOT) and clamp
            x = self.clamp(1 - x_delayed)  # Invert for oscillation

            outputs.append(x)

        if return_sequence:
            return torch.cat(outputs, dim=1), torch.arange(num_steps) % self.period

        return outputs[-1].squeeze(1), None


class NJTLayer(nn.Module):
    """
    Complete NJT Layer combining excitatory and inhibitory circuits.

    This is the main integration point for adding NJT to transformer blocks.

    Architecture:
        Input ──┬── NJT+ (amplify) ──┬── Combine ── Output
                │                    │
                └── NJT- (inhibit) ──┘

    Optional components:
        - Hysteresis memory (N5)
        - Differential pairs for decisions
        - Rhythm oscillator (N7)
    """

    def __init__(self, config: NJTConfig):
        super().__init__()
        self.config = config

        # Core transistor banks
        self.excitatory = NJTPlus(config.dim, config.num_transistors, config)
        self.inhibitory = NJTMinus(config.dim, config.num_transistors, config)

        # Balance between excitatory and inhibitory
        self.exc_inh_balance = nn.Parameter(torch.tensor(0.5))

        # Optional: Hysteresis memory (N5)
        self.hysteresis = None
        if config.use_hysteresis:
            self.hysteresis = HysteresisMemory(
                config.num_transistors,
                config.hysteresis_gap
            )

        # Optional: Rhythm oscillator (N7)
        self.oscillator = None
        if config.use_rhythm:
            self.oscillator = RhythmOscillator(config.dim, config)

        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # Output projection
        self.output_proj = nn.Linear(config.dim, config.dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        brake_signal: Optional[torch.Tensor] = None,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through NJT layer.

        Args:
            x: Input signal [batch, seq, dim]
            context: Context for excitatory bias (defaults to x)
            brake_signal: Signal for inhibitory brake (defaults to x)
            return_trace: Whether to return full trace for debugging

        Returns:
            output: Processed signal [batch, seq, dim]
            trace: Dict of internal states (if return_trace)
        """
        trace = {} if return_trace else None

        # Excitatory path (amplification)
        exc_out, exc_internals = self.excitatory(
            x, context, return_internals=return_trace
        )

        # Inhibitory path (dampening)
        inh_out, inh_internals = self.inhibitory(
            x, brake_signal, return_internals=return_trace
        )

        # Balance excitation and inhibition
        balance = torch.sigmoid(self.exc_inh_balance)
        combined = balance * exc_out + (1 - balance) * inh_out

        # Apply hysteresis if enabled
        if self.hysteresis is not None:
            # Project to transistor space for hysteresis
            hyst_input = combined.mean(dim=-1)  # [batch, seq]
            # Expand to match transistor count
            hyst_input = hyst_input.unsqueeze(-1).expand(-1, -1, self.config.num_transistors)
            hyst_out = self.hysteresis(hyst_input)
            # Modulate combined output
            hyst_scale = hyst_out.mean(dim=-1, keepdim=True)
            combined = combined * (0.5 + 0.5 * hyst_scale)

        # Output projection with residual
        output = self.output_proj(combined)
        output = output + self.residual_weight * x

        if return_trace:
            trace = {
                'excitatory': exc_internals,
                'inhibitory': inh_internals,
                'exc_inh_balance': balance,
                'combined_pre_hysteresis': combined,
                'output': output,
            }
            if self.hysteresis is not None:
                trace['hysteresis_state'] = self.hysteresis.state.clone()

        return output, trace

    def reset_hysteresis(self):
        """Reset hysteresis memory state."""
        if self.hysteresis is not None:
            self.hysteresis.reset_state()


# Convenience function to create NJT layer with defaults
def create_njt_layer(
    dim: int = 384,
    num_transistors: int = 10,
    use_hysteresis: bool = True,
    use_rhythm: bool = False,
    **kwargs
) -> NJTLayer:
    """Create an NJT layer with sensible defaults."""
    config = NJTConfig(
        dim=dim,
        num_transistors=num_transistors,
        use_hysteresis=use_hysteresis,
        use_rhythm=use_rhythm,
        **kwargs
    )
    return NJTLayer(config)

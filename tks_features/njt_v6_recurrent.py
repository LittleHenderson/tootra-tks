"""
NJT v6 Recurrent - Prerequisite-Aware Recurrent NJT for RPM Processing

This module extends the base NJT system to support:
1. 12-slot Prerequisite State Vector (D×W×P × 4 Worlds)
2. Scalar Gradation (0.0-1.0) for each prerequisite
3. Noetic 9 (Effect) as halting signal
4. Recurrent processing for arbitrary-depth RPM chains

The 12 Prerequisites:
    Spiritual (A): 1D, 2W, 3P
    Mental (B):    4D, 5W, 6P
    Emotional (C): 7D, 8W, 9P
    Physical (D):  10D, 11W, 12P

Each prerequisite has 4 gradation stages:
    DESIRE (Kings):   0.0=Absent, 0.3=Beginning, 0.7=Weakening, 1.0=Strong
    WISDOM (Jacks):   0.0=None, 0.3=Learning, 0.7=Knowing, 1.0=Mastering
    POWER (Queens):   0.0=None, 0.3=Birth, 0.7=Adolescent, 1.0=Mature

Author: TKS Development
Date: 2026-01-10
Status: v6 Architecture for RPM-based reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math


# =============================================================================
# CONSTANTS: TKS PREREQUISITE STRUCTURE
# =============================================================================

# The 12 Prerequisites indexed by position
PREREQUISITE_NAMES = [
    "Spiritual_Desire",   # 0 (1D)
    "Spiritual_Wisdom",   # 1 (2W)
    "Spiritual_Power",    # 2 (3P)
    "Mental_Desire",      # 3 (4D)
    "Mental_Wisdom",      # 4 (5W)
    "Mental_Power",       # 5 (6P)
    "Emotional_Desire",   # 6 (7D)
    "Emotional_Wisdom",   # 7 (8W)
    "Emotional_Power",    # 8 (9P)
    "Physical_Desire",    # 9 (10D)
    "Physical_Wisdom",    # 10 (11W)
    "Physical_Power",     # 11 (12P)
]

# World indices
WORLD_SPIRITUAL = 0  # A
WORLD_MENTAL = 1     # B
WORLD_EMOTIONAL = 2  # C
WORLD_PHYSICAL = 3   # D

# Prerequisite type indices
TYPE_DESIRE = 0  # D
TYPE_WISDOM = 1  # W
TYPE_POWER = 2   # P

# Gradation levels (scalar values)
GRADATION_ABSENT = 0.0      # Night / None
GRADATION_BEGINNING = 0.33  # Sunrise / Learning / Birth
GRADATION_WEAKENING = 0.67  # Sunset / Knowing / Adolescent
GRADATION_STRONG = 1.0      # Noon / Mastering / Mature


@dataclass
class NJTv6Config:
    """Configuration for NJT v6 Recurrent system."""
    dim: int = 384
    num_transistors: int = 10
    num_prerequisites: int = 12
    num_worlds: int = 4
    num_prereq_types: int = 3  # D, W, P

    # Recurrent parameters
    max_iterations: int = 12  # Max RPM depth (one per prerequisite)
    use_adaptive_halt: bool = True
    halt_threshold: float = 0.9  # Confidence to stop

    # Gate parameters
    initial_gain: float = 2.0
    initial_threshold: float = 0.5
    gate_sharpness: float = 10.0

    # Gradation parameters
    gradation_levels: int = 4
    use_soft_gradation: bool = True  # Continuous vs discrete

    # Nested Memory (Recursion Stack) parameters
    use_nested_memory: bool = True  # Enable recursion stack
    max_recursion_depth: int = 4  # How deep can we nest sub-goals
    sufficiency_threshold: float = 0.6  # What value counts as "satisfied"


# =============================================================================
# PREREQUISITE STATE VECTOR
# =============================================================================

class PrerequisiteStateVector(nn.Module):
    """
    12-slot state vector representing D×W×P across 4 Worlds.

    Each slot holds a scalar gradation value [0.0, 1.0]:
        0.0 = Absent (Night/None)
        0.33 = Beginning (Sunrise/Learning/Birth)
        0.67 = Weakening (Sunset/Knowing/Adolescent)
        1.0 = Strong (Noon/Mastering/Mature)

    Shape: [batch, 12] or [batch, seq, 12]

    Layout:
        [0-2]:  Spiritual D, W, P
        [3-5]:  Mental D, W, P
        [6-8]:  Emotional D, W, P
        [9-11]: Physical D, W, P
    """

    def __init__(self, config: NJTv6Config):
        super().__init__()
        self.config = config
        self.num_prereqs = config.num_prerequisites

        # Learnable initial state (default to mid-level)
        self.initial_state = nn.Parameter(
            torch.ones(config.num_prerequisites) * 0.5
        )

        # Thresholds for each prerequisite (what level is "sufficient"?)
        self.sufficiency_threshold = nn.Parameter(
            torch.ones(config.num_prerequisites) * 0.7
        )

        # World-specific weighting (some worlds matter more for certain goals)
        self.world_weights = nn.Parameter(
            torch.ones(config.num_worlds)
        )

        # Type-specific weighting (D vs W vs P priority)
        self.type_weights = nn.Parameter(
            torch.tensor([1.0, 1.0, 1.0])  # D, W, P
        )

    def get_prerequisite(
        self,
        state: torch.Tensor,
        world: int,
        prereq_type: int
    ) -> torch.Tensor:
        """
        Get a specific prerequisite value from state.

        Args:
            state: [batch, 12] or [batch, seq, 12]
            world: 0=A (Spiritual), 1=B (Mental), 2=C (Emotional), 3=D (Physical)
            prereq_type: 0=Desire, 1=Wisdom, 2=Power

        Returns:
            Scalar gradation value [batch] or [batch, seq]
        """
        idx = world * 3 + prereq_type
        return state[..., idx]

    def set_prerequisite(
        self,
        state: torch.Tensor,
        world: int,
        prereq_type: int,
        value: torch.Tensor
    ) -> torch.Tensor:
        """Set a specific prerequisite value in state."""
        idx = world * 3 + prereq_type
        new_state = state.clone()
        new_state[..., idx] = value
        return new_state

    def get_world_state(self, state: torch.Tensor, world: int) -> torch.Tensor:
        """Get D, W, P triple for a specific world."""
        start_idx = world * 3
        return state[..., start_idx:start_idx + 3]

    def is_satisfied(self, state: torch.Tensor) -> torch.Tensor:
        """
        Check if all prerequisites meet their sufficiency thresholds.

        Returns:
            Boolean tensor [batch] or [batch, seq]
        """
        return (state >= self.sufficiency_threshold).all(dim=-1)

    def find_first_block(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the first unsatisfied prerequisite (RPM block point).

        Returns:
            block_idx: Index of first blocked prerequisite [batch]
            block_value: Current value of blocked prerequisite [batch]
        """
        # Check which are below threshold
        unsatisfied = state < self.sufficiency_threshold

        # Find first True in each batch item
        # If all satisfied, return -1
        has_block = unsatisfied.any(dim=-1)

        # Get index of first block (or 0 if none)
        block_idx = unsatisfied.float().argmax(dim=-1)
        block_idx = torch.where(has_block, block_idx, torch.tensor(-1, device=state.device))

        # Get the value at that index
        batch_size = state.shape[0]
        batch_indices = torch.arange(batch_size, device=state.device)
        block_value = torch.where(
            has_block,
            state[batch_indices, block_idx.clamp(min=0)],
            torch.ones(batch_size, device=state.device)
        )

        return block_idx, block_value

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize state vector for a batch."""
        return self.initial_state.unsqueeze(0).expand(batch_size, -1).clone()


# =============================================================================
# GRADATION ENCODER/DECODER
# =============================================================================

class GradationCodec(nn.Module):
    """
    Encode/decode between embedding space and scalar gradation values.

    Gradation values (0.0-1.0) map to the 4 stages:
        - Desire: Absent → Beginning → Weakening → Strong
        - Wisdom: None → Learning → Knowing → Mastering
        - Power: None → Birth → Adolescent → Mature
    """

    def __init__(self, config: NJTv6Config):
        super().__init__()
        self.config = config

        # Encode: embedding dim -> 12 gradation values
        self.encoder = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, config.num_prerequisites),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Decode: 12 gradation values -> embedding dim
        self.decoder = nn.Sequential(
            nn.Linear(config.num_prerequisites, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, config.dim)
        )

        # Gradation level embeddings (4 discrete levels per prereq type)
        self.desire_levels = nn.Embedding(4, config.dim // 4)  # Kings
        self.wisdom_levels = nn.Embedding(4, config.dim // 4)  # Jacks
        self.power_levels = nn.Embedding(4, config.dim // 4)   # Queens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert embedding to 12 gradation values.

        Args:
            x: [batch, seq, dim] or [batch, dim]

        Returns:
            gradations: [batch, seq, 12] or [batch, 12]
        """
        return self.encoder(x)

    def decode(self, gradations: torch.Tensor) -> torch.Tensor:
        """
        Convert 12 gradation values back to embedding.

        Args:
            gradations: [batch, seq, 12] or [batch, 12]

        Returns:
            x: [batch, seq, dim] or [batch, dim]
        """
        return self.decoder(gradations)

    def quantize_gradation(self, value: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous [0, 1] to discrete level index [0, 1, 2, 3].

        0.0-0.25 -> 0 (Absent/None)
        0.25-0.5 -> 1 (Beginning/Learning/Birth)
        0.5-0.75 -> 2 (Weakening/Knowing/Adolescent)
        0.75-1.0 -> 3 (Strong/Mastering/Mature)
        """
        return (value * 3.99).long().clamp(0, 3)

    def get_level_embedding(
        self,
        gradation: torch.Tensor,
        prereq_type: int
    ) -> torch.Tensor:
        """Get the level embedding for a gradation value."""
        level_idx = self.quantize_gradation(gradation)

        if prereq_type == TYPE_DESIRE:
            return self.desire_levels(level_idx)
        elif prereq_type == TYPE_WISDOM:
            return self.wisdom_levels(level_idx)
        else:  # TYPE_POWER
            return self.power_levels(level_idx)


# =============================================================================
# NOETIC 9 (EFFECT) HALTING MECHANISM
# =============================================================================

class Noetic9Halt(nn.Module):
    """
    Noetic 9 (Effect) - The halting/completion signal.

    In TKS, Noetic 9 represents the final Effect of a causal chain.
    Here it signals when the RPM recursion is complete:
    - All prerequisites are satisfied
    - The chain has reached its terminal state

    Outputs a scalar "done" probability [0, 1].
    """

    def __init__(self, config: NJTv6Config):
        super().__init__()
        self.config = config
        self.halt_threshold = config.halt_threshold

        # Project from state + hidden to halt probability
        self.halt_proj = nn.Sequential(
            nn.Linear(config.num_prerequisites + config.dim, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, 1),
            nn.Sigmoid()
        )

        # Learnable "effect" detection per prerequisite
        self.prereq_effect_weights = nn.Parameter(
            torch.ones(config.num_prerequisites)
        )

    def forward(
        self,
        prereq_state: torch.Tensor,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute halting probability (Noetic 9 activation).

        Args:
            prereq_state: [batch, 12] prerequisite state vector
            hidden: [batch, dim] current hidden state

        Returns:
            halt_prob: [batch, 1] probability of halting
            effect_strength: [batch, 1] weighted effect measure
        """
        # Weighted sum of prerequisite satisfaction
        weighted_state = prereq_state * F.softmax(self.prereq_effect_weights, dim=0)
        effect_strength = weighted_state.sum(dim=-1, keepdim=True)

        # Combine state and hidden for halt decision
        combined = torch.cat([prereq_state, hidden], dim=-1)
        halt_prob = self.halt_proj(combined)

        return halt_prob, effect_strength

    def should_halt(self, halt_prob: torch.Tensor) -> torch.Tensor:
        """Check if we should halt (during inference)."""
        return halt_prob > self.halt_threshold


# =============================================================================
# RECURSION STACK (Nested Memory)
# =============================================================================

class RecursionStack(nn.Module):
    """
    Differentiable Stack for Nested Memory in RPM Processing.

    When the model hits a BLOCK (unsatisfied prerequisite), it:
    1. PUSH: Saves current state to stack
    2. Creates new context for sub-goal
    3. Processes sub-goal recursively
    4. POP: Restores parent state when sub-goal resolved

    This enables true recursive reasoning:
        Level 0: Goal="Open Shop" -> Block at P12 (no money)
        Level 1: Sub-Goal="Get Money" -> Block at P11 (no knowledge)
        Level 2: Sub-Goal="Learn Finance" -> Satisfied
        Level 1: P11 resolved -> P12 now possible
        Level 0: P12 resolved -> Goal achieved

    Isomorphic to:
    - Call stacks in programming
    - Noetic Fractals (5 -> 5.1 -> 5.1.3)
    - Tree of Knowledge recursion
    """

    def __init__(self, config: NJTv6Config, max_depth: int = 4):
        super().__init__()
        self.config = config
        self.max_depth = max_depth

        # Stack dimensions
        # Each stack frame stores: prereq_state (12) + hidden (dim) + block_idx (1)
        self.frame_size = config.num_prerequisites + config.dim + 1

        # Stack memory buffer (differentiable)
        # Shape: [batch, max_depth, frame_size]
        self.register_buffer(
            'stack_memory',
            torch.zeros(1, max_depth, self.frame_size)
        )

        # Stack pointer (which level are we at?)
        self.register_buffer('stack_ptr', torch.zeros(1, dtype=torch.long))

        # Learnable frame encoding/decoding
        self.frame_encoder = nn.Linear(self.frame_size, config.dim)
        self.frame_decoder = nn.Linear(config.dim, self.frame_size)

        # Context projection for sub-goal creation
        self.subgoal_proj = nn.Sequential(
            nn.Linear(config.dim + config.num_prerequisites, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim)
        )

    def reset(self, batch_size: int, device: torch.device):
        """Reset stack for new batch."""
        self.stack_memory = torch.zeros(
            batch_size, self.max_depth, self.frame_size,
            device=device
        )
        self.stack_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)

    def push(
        self,
        prereq_state: torch.Tensor,
        hidden: torch.Tensor,
        block_idx: torch.Tensor
    ) -> bool:
        """
        Push current state onto stack when hitting a block.

        Args:
            prereq_state: [batch, 12] current prerequisite state
            hidden: [batch, dim] current hidden state
            block_idx: [batch, 1] index of blocked prerequisite

        Returns:
            success: Whether push succeeded (not at max depth)
        """
        batch_size = prereq_state.shape[0]
        device = prereq_state.device

        # Check if we can push
        can_push = self.stack_ptr < self.max_depth - 1

        if not can_push.any():
            return False

        # Create frame: [prereq_state, hidden, block_idx] - detached
        with torch.no_grad():
            frame = torch.cat([
                prereq_state.detach(),
                hidden.detach(),
                block_idx.float().unsqueeze(-1).detach() if block_idx.dim() == 1 else block_idx.float().detach()
            ], dim=-1)

            # Push to stack at current pointer position
            for b in range(batch_size):
                if can_push[b]:
                    ptr = self.stack_ptr[b].item()
                    self.stack_memory[b, ptr] = frame[b]
                    self.stack_ptr[b] += 1

        return True

    def pop(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Pop state from stack when sub-goal is resolved.

        Returns:
            prereq_state: [batch, 12] restored prerequisite state
            hidden: [batch, dim] restored hidden state
            block_idx: [batch] which prerequisite was blocked
            Returns (None, None, None) if stack is empty
        """
        with torch.no_grad():
            batch_size = self.stack_memory.shape[0]

            # Check if we can pop
            can_pop = self.stack_ptr > 0

            if not can_pop.any():
                return None, None, None

            prereq_states = []
            hiddens = []
            block_idxs = []

            for b in range(batch_size):
                if can_pop[b]:
                    self.stack_ptr[b] -= 1
                    ptr = self.stack_ptr[b].item()
                    frame = self.stack_memory[b, ptr]

                    # Decode frame
                    prereq_state = frame[:self.config.num_prerequisites]
                    hidden = frame[self.config.num_prerequisites:self.config.num_prerequisites + self.config.dim]
                    block_idx = frame[-1]

                    prereq_states.append(prereq_state)
                    hiddens.append(hidden)
                    block_idxs.append(block_idx)
                else:
                    # Return zeros for items that can't pop
                    prereq_states.append(torch.zeros(self.config.num_prerequisites, device=self.stack_memory.device))
                    hiddens.append(torch.zeros(self.config.dim, device=self.stack_memory.device))
                    block_idxs.append(torch.tensor(0.0, device=self.stack_memory.device))

            return (
                torch.stack(prereq_states),
                torch.stack(hiddens),
                torch.stack(block_idxs).long()
            )

    def peek(self) -> Tuple[Optional[torch.Tensor], int]:
        """
        Look at top of stack without popping.

        Returns:
            top_frame: The frame at top of stack (or None if empty)
            depth: Current stack depth
        """
        batch_size = self.stack_memory.shape[0]
        depth = self.stack_ptr[0].item()  # Assume same depth for batch

        if depth == 0:
            return None, 0

        top_frames = []
        for b in range(batch_size):
            ptr = max(0, self.stack_ptr[b].item() - 1)
            top_frames.append(self.stack_memory[b, ptr])

        return torch.stack(top_frames), depth

    def get_depth(self) -> torch.Tensor:
        """Get current recursion depth for each batch item."""
        return self.stack_ptr.clone()

    def create_subgoal_context(
        self,
        prereq_state: torch.Tensor,
        hidden: torch.Tensor,
        block_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Create context embedding for sub-goal processing.

        When we hit a block, we need to create a new context
        focused on resolving that specific prerequisite.

        Args:
            prereq_state: Current state
            hidden: Current hidden
            block_idx: Which prerequisite is blocked

        Returns:
            subgoal_context: [batch, dim] new context for sub-goal
        """
        # Combine state and hidden
        combined = torch.cat([prereq_state, hidden], dim=-1)
        subgoal_context = self.subgoal_proj(combined)

        return subgoal_context


class PushPopGate(nn.Module):
    """
    Noetic Gate that decides PUSH (drill down) or POP (return up).

    PUSH condition (NJT- style - inhibition detected):
    - Current prerequisite is blocked (below threshold)
    - Sub-goal processing is needed

    POP condition (NJT+ style - satisfaction detected):
    - Sub-goal has been resolved
    - Can return to parent context

    This gate is the "decision maker" for recursion control.
    """

    def __init__(self, config: NJTv6Config):
        super().__init__()
        self.config = config

        # Push decision (detect block -> need to recurse)
        self.push_gate = nn.Sequential(
            nn.Linear(config.num_prerequisites + config.dim, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, 1),
            nn.Sigmoid()
        )

        # Pop decision (detect resolution -> can return)
        self.pop_gate = nn.Sequential(
            nn.Linear(config.num_prerequisites + config.dim + 1, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, 1),
            nn.Sigmoid()
        )

        # Thresholds (start low to encourage exploration, model will learn)
        self.push_threshold = nn.Parameter(torch.tensor(0.3))  # Was 0.6 - too high for untrained gates
        self.pop_threshold = nn.Parameter(torch.tensor(0.5))   # Was 0.7

    def should_push(
        self,
        prereq_state: torch.Tensor,
        hidden: torch.Tensor,
        block_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide if we should PUSH (start sub-goal processing).

        Args:
            prereq_state: [batch, 12]
            hidden: [batch, dim]
            block_idx: [batch] index of potential block

        Returns:
            should_push: [batch] boolean tensor
            push_prob: [batch, 1] probability
        """
        combined = torch.cat([prereq_state, hidden], dim=-1)
        push_prob = self.push_gate(combined)

        # Also check if there actually is a block
        batch_size = prereq_state.shape[0]
        has_block = block_idx >= 0

        should_push = (push_prob.squeeze(-1) > self.push_threshold) & has_block

        return should_push, push_prob

    def should_pop(
        self,
        prereq_state: torch.Tensor,
        hidden: torch.Tensor,
        original_block_idx: torch.Tensor,
        stack_depth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide if we should POP (return from sub-goal).

        Args:
            prereq_state: [batch, 12] current state
            hidden: [batch, dim]
            original_block_idx: [batch] what was blocked when we pushed
            stack_depth: [batch] current recursion depth

        Returns:
            should_pop: [batch] boolean tensor
            pop_prob: [batch, 1] probability
        """
        # Check if the originally blocked prerequisite is now satisfied
        batch_size = prereq_state.shape[0]

        # Get value of originally blocked prereq
        block_values = torch.zeros(batch_size, 1, device=prereq_state.device)
        for b in range(batch_size):
            idx = original_block_idx[b].item()
            if 0 <= idx < self.config.num_prerequisites:
                block_values[b] = prereq_state[b, idx]

        combined = torch.cat([prereq_state, hidden, block_values], dim=-1)
        pop_prob = self.pop_gate(combined)

        # Can only pop if we're not at root (depth > 0)
        can_pop = stack_depth > 0

        should_pop = (pop_prob.squeeze(-1) > self.pop_threshold) & can_pop

        return should_pop, pop_prob


# =============================================================================
# NJT V6 RECURRENT CELL
# =============================================================================

class NJTv6RecurrentCell(nn.Module):
    """
    Single recurrent step of NJT v6 processing.

    Each step:
    1. Takes current input + prerequisite state + hidden state
    2. Updates prerequisite state based on NJT gating
    3. Produces output + new hidden state + halt probability

    This is the "brain" that processes one level of RPM recursion.
    """

    def __init__(self, config: NJTv6Config):
        super().__init__()
        self.config = config

        # Input projection (combines input, prereq state, hidden)
        input_dim = config.dim + config.num_prerequisites + config.dim
        self.input_proj = nn.Linear(input_dim, config.dim)

        # NJT+ (excitatory) for amplifying prerequisites
        self.njt_plus = nn.Sequential(
            nn.Linear(config.dim, config.num_transistors),
            nn.Sigmoid(),
        )
        self.njt_plus_gate = nn.Linear(config.dim, config.num_transistors)
        self.njt_plus_gain = nn.Parameter(torch.ones(config.num_transistors) * config.initial_gain)

        # NJT- (inhibitory) for blocking prerequisites
        self.njt_minus = nn.Sequential(
            nn.Linear(config.dim, config.num_transistors),
            nn.Sigmoid(),
        )
        self.njt_minus_gate = nn.Linear(config.dim, config.num_transistors)
        self.njt_minus_gain = nn.Parameter(torch.ones(config.num_transistors) * config.initial_gain)

        # Prerequisite update projections
        self.prereq_delta_proj = nn.Linear(config.num_transistors * 2, config.num_prerequisites)

        # Hidden state update (GRU-style)
        self.hidden_update = nn.GRUCell(config.dim, config.dim)

        # Output projection
        self.output_proj = nn.Linear(config.dim, config.dim)

        # Thresholds
        self.threshold = nn.Parameter(torch.ones(config.num_transistors) * config.initial_threshold)
        self.sharpness = config.gate_sharpness

    def forward(
        self,
        x: torch.Tensor,
        prereq_state: torch.Tensor,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Single recurrent step.

        Args:
            x: Input [batch, dim]
            prereq_state: Current prerequisite state [batch, 12]
            hidden: Previous hidden state [batch, dim]

        Returns:
            output: Output embedding [batch, dim]
            new_prereq_state: Updated prerequisites [batch, 12]
            new_hidden: New hidden state [batch, dim]
            info: Debug information
        """
        batch_size = x.shape[0]

        # Combine inputs
        combined = torch.cat([x, prereq_state, hidden], dim=-1)
        h = self.input_proj(combined)

        # NJT+ path (amplification)
        plus_cause = self.njt_plus(h)
        plus_bias = torch.sigmoid(self.njt_plus_gate(h))
        plus_gate = torch.sigmoid(self.sharpness * (plus_bias - self.threshold))
        plus_out = self.njt_plus_gain * plus_cause * plus_gate
        plus_out = torch.clamp(plus_out, 0, 1)

        # NJT- path (inhibition)
        minus_cause = self.njt_minus(h)
        minus_bias = torch.sigmoid(self.njt_minus_gate(h))
        minus_gate = torch.sigmoid(self.sharpness * (minus_bias - self.threshold))
        minus_out = self.njt_minus_gain * minus_cause * (1 - minus_gate)
        minus_out = torch.clamp(minus_out, 0, 1)

        # Compute prerequisite delta from NJT outputs
        njt_combined = torch.cat([plus_out, minus_out], dim=-1)
        prereq_delta = torch.tanh(self.prereq_delta_proj(njt_combined)) * 0.1  # Small updates

        # Update prerequisite state (clamped to [0, 1])
        new_prereq_state = torch.clamp(prereq_state + prereq_delta, 0, 1)

        # Update hidden state
        new_hidden = self.hidden_update(h, hidden)

        # Output
        output = self.output_proj(new_hidden)

        info = {
            'plus_gate': plus_gate,
            'minus_gate': minus_gate,
            'prereq_delta': prereq_delta,
        }

        return output, new_prereq_state, new_hidden, info


# =============================================================================
# FULL NJT V6 RECURRENT SYSTEM
# =============================================================================

class NJTv6Recurrent(nn.Module):
    """
    Complete NJT v6 Recurrent System for RPM-based reasoning.

    Now with NESTED MEMORY (RecursionStack) for true recursive processing:

    Processes input through multiple recurrent steps until:
    - All prerequisites are satisfied (Noetic 9 fires)
    - Maximum iterations reached

    Each iteration:
    1. Process input through NJT cell
    2. Update prerequisite state
    3. Check for BLOCK (unsatisfied prerequisite)
    4. If BLOCKED: PUSH state and create sub-goal context
    5. If sub-goal RESOLVED: POP and return to parent
    6. Check halting condition (Noetic 9)

    This implements the RPM algorithm in neural form:
    - Scan prerequisites (D->W->P per world)
    - Find block
    - PUSH to stack, recurse on block (sub-goal)
    - When resolved, POP and ascend
    - Final EFFECT when all satisfied
    """

    def __init__(self, config: NJTv6Config):
        super().__init__()
        self.config = config

        # Core Components
        self.prereq_state = PrerequisiteStateVector(config)
        self.gradation_codec = GradationCodec(config)
        self.noetic9_halt = Noetic9Halt(config)
        self.recurrent_cell = NJTv6RecurrentCell(config)

        # Nested Memory Components (NEW)
        if config.use_nested_memory:
            self.recursion_stack = RecursionStack(config, config.max_recursion_depth)
            self.push_pop_gate = PushPopGate(config)
        else:
            self.recursion_stack = None
            self.push_pop_gate = None

        # Iteration embedding (positional encoding for depth)
        self.iter_embed = nn.Embedding(config.max_iterations, config.dim)

        # Depth embedding (which recursion level are we at?)
        self.depth_embed = nn.Embedding(config.max_recursion_depth + 1, config.dim)

        # Output processing
        self.output_norm = nn.LayerNorm(config.dim)
        self.final_proj = nn.Linear(config.dim + config.num_prerequisites, config.dim)

    def forward(
        self,
        x: torch.Tensor,
        initial_prereq_state: Optional[torch.Tensor] = None,
        max_iters: Optional[int] = None,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Process input through recurrent NJT system with NESTED MEMORY.

        Args:
            x: Input embedding [batch, dim]
            initial_prereq_state: Starting prerequisites [batch, 12] (optional)
            max_iters: Maximum iterations (defaults to config)
            return_trace: Return full processing trace

        Returns:
            output: Final output [batch, dim]
            final_prereq_state: Final prerequisite state [batch, 12]
            trace: Processing trace (if return_trace)
        """
        batch_size = x.shape[0]
        device = x.device

        if max_iters is None:
            max_iters = self.config.max_iterations

        # Initialize states
        if initial_prereq_state is None:
            prereq_state = self.prereq_state(batch_size, device)
        else:
            prereq_state = initial_prereq_state

        hidden = torch.zeros(batch_size, self.config.dim, device=device)

        # Initialize recursion stack if using nested memory
        if self.recursion_stack is not None:
            self.recursion_stack.reset(batch_size, device)
            current_block_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        # Tracking
        trace = [] if return_trace else None
        outputs = []
        halt_probs = []
        prereq_history = [prereq_state.clone()]
        stack_history = [] if return_trace else None

        # Recurrent processing with nested memory
        for i in range(max_iters):
            # Get current recursion depth
            if self.recursion_stack is not None:
                depth = self.recursion_stack.get_depth()
                depth_emb = self.depth_embed(depth.clamp(max=self.config.max_recursion_depth))
            else:
                depth = torch.zeros(batch_size, dtype=torch.long, device=device)
                depth_emb = torch.zeros(batch_size, self.config.dim, device=device)

            # Add iteration + depth embeddings
            iter_emb = self.iter_embed(
                torch.tensor([i % self.config.max_iterations], device=device)
            ).expand(batch_size, -1)
            x_iter = x + iter_emb + depth_emb

            # Process through cell
            out, prereq_state, hidden, cell_info = self.recurrent_cell(
                x_iter, prereq_state, hidden
            )
            outputs.append(out)
            prereq_history.append(prereq_state.clone())

            # === NESTED MEMORY LOGIC ===
            if self.recursion_stack is not None and self.push_pop_gate is not None:
                # Find first block in current state
                block_idx, block_val = self.prereq_state.find_first_block(prereq_state)

                # Check if we should PUSH (block detected, need sub-goal)
                should_push, push_prob = self.push_pop_gate.should_push(
                    prereq_state, hidden, block_idx
                )

                # Check if we should POP (sub-goal resolved)
                should_pop, pop_prob = self.push_pop_gate.should_pop(
                    prereq_state, hidden, current_block_idx, depth
                )

                # Execute PUSH for batch items that need it
                if should_push.any() and not should_pop.any():
                    # Create masks for selective update
                    push_mask = (should_push & (depth < self.config.max_recursion_depth)).unsqueeze(-1)

                    # Store CLONED+DETACHED state to stack (breaks gradient to stack)
                    for b in range(batch_size):
                        if should_push[b] and depth[b] < self.config.max_recursion_depth:
                            self.recursion_stack.push(
                                prereq_state[b:b+1].clone().detach(),
                                hidden[b:b+1].clone().detach(),
                                block_idx[b:b+1].clone().detach()
                            )
                    # Update block index out-of-place
                    current_block_idx = torch.where(should_push, block_idx, current_block_idx)

                    # Create sub-goal context (detached inputs to avoid graph issues)
                    subgoal_ctx = self.recursion_stack.create_subgoal_context(
                        prereq_state.detach(), hidden.detach(), block_idx.unsqueeze(-1)
                    )

                    # Apply update out-of-place using mask
                    hidden = torch.where(push_mask, hidden + subgoal_ctx * 0.5, hidden)

                # Execute POP for batch items that resolved sub-goal
                elif should_pop.any():
                    pop_mask = should_pop.unsqueeze(-1)

                    # Pop from stack (detached - stack contents don't need gradients)
                    for b in range(batch_size):
                        if should_pop[b]:
                            # Just pop to update stack pointer, we blend with current state
                            _ = self.recursion_stack.pop()
                            current_block_idx[b] = -1

                    # Blend: keep 70% of current progress, 30% boost from "resolution"
                    # This is gradient-safe as we only use current state
                    blended_prereq = prereq_state + 0.1  # Small boost for resolved items
                    blended_prereq = torch.clamp(blended_prereq, 0, 1)
                    blended_hidden = hidden  # Keep current hidden

                    prereq_state = torch.where(pop_mask, blended_prereq, prereq_state)
                    hidden = torch.where(pop_mask, blended_hidden, hidden)

                if return_trace and stack_history is not None:
                    stack_history.append({
                        'depth': depth.tolist(),
                        'block_idx': block_idx.tolist(),
                        'push_prob': push_prob.mean().item(),
                        'pop_prob': pop_prob.mean().item(),
                        'did_push': should_push.tolist(),
                        'did_pop': should_pop.tolist(),
                    })

            # Check halting (Noetic 9)
            halt_prob, effect_strength = self.noetic9_halt(prereq_state, hidden)
            halt_probs.append(halt_prob)

            if return_trace:
                trace.append({
                    'iteration': i,
                    'prereq_state': prereq_state.clone(),
                    'halt_prob': halt_prob.mean().item(),
                    'effect_strength': effect_strength.mean().item(),
                    'plus_gate_mean': cell_info['plus_gate'].mean().item(),
                    'minus_gate_mean': cell_info['minus_gate'].mean().item(),
                    'recursion_depth': depth[0].item() if self.recursion_stack else 0,
                })

            # Early stopping during inference (only if stack is empty)
            if not self.training:
                stack_empty = True
                if self.recursion_stack is not None:
                    stack_empty = (self.recursion_stack.get_depth() == 0).all()

                if self.noetic9_halt.should_halt(halt_prob).all() and stack_empty:
                    break

        # Combine outputs weighted by halt probabilities (Adaptive Computation Time)
        if len(outputs) > 1 and self.config.use_adaptive_halt:
            halt_stack = torch.cat(halt_probs, dim=-1)  # [batch, num_iters]
            halt_weights = F.softmax(halt_stack, dim=-1)

            out_stack = torch.stack(outputs, dim=-1)  # [batch, dim, num_iters]
            final_out = (out_stack * halt_weights.unsqueeze(1)).sum(dim=-1)
        else:
            final_out = outputs[-1]

        # Final processing
        final_out = self.output_norm(final_out)
        combined = torch.cat([final_out, prereq_state], dim=-1)
        output = self.final_proj(combined)

        if return_trace:
            trace_dict = {
                'iterations': trace,
                'num_iterations': len(outputs),
                'prereq_history': prereq_history,
                'halt_probs': [h.mean().item() for h in halt_probs],
                'stack_history': stack_history,
                'max_depth_reached': max(t.get('recursion_depth', 0) for t in trace) if trace else 0,
            }
            return output, prereq_state, trace_dict

        return output, prereq_state, None

    def diagnose_prerequisites(
        self,
        prereq_state: torch.Tensor
    ) -> Dict[str, any]:
        """
        Diagnose the current prerequisite state (RPM-style).

        Returns human-readable analysis of which prerequisites
        are satisfied, blocked, or weak.
        """
        threshold = self.prereq_state.sufficiency_threshold

        diagnosis = {
            'satisfied': [],
            'blocked': [],
            'weak': [],  # Between 0.3 and threshold
        }

        for i, name in enumerate(PREREQUISITE_NAMES):
            val = prereq_state[0, i].item()
            thresh = threshold[i].item()

            if val >= thresh:
                diagnosis['satisfied'].append((name, val))
            elif val < 0.3:
                diagnosis['blocked'].append((name, val))
            else:
                diagnosis['weak'].append((name, val))

        # Find first block (RPM style)
        block_idx, block_val = self.prereq_state.find_first_block(prereq_state)
        if block_idx[0].item() >= 0:
            diagnosis['first_block'] = PREREQUISITE_NAMES[block_idx[0].item()]
        else:
            diagnosis['first_block'] = None

        return diagnosis


# =============================================================================
# INTEGRATION LAYER FOR TKS LLM
# =============================================================================

class TKSNJTv6Block(nn.Module):
    """
    Drop-in replacement for NJTLayer that uses v6 recurrent system.

    Use this in TKSGeneralLM to enable RPM-based reasoning:

    Before (v5):
        Attention -> NJTLayer -> FFN

    After (v6):
        Attention -> TKSNJTv6Block -> FFN
    """

    def __init__(self, dim: int, config: Optional[NJTv6Config] = None):
        super().__init__()

        if config is None:
            config = NJTv6Config(dim=dim)
        else:
            config.dim = dim

        self.config = config
        self.njt_v6 = NJTv6Recurrent(config)

        # Handle sequence dimension
        self.pool_seq = nn.AdaptiveAvgPool1d(1)
        self.expand_seq = nn.Linear(dim, dim)

        # Residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through v6 NJT block.

        Args:
            x: Input [batch, seq, dim]
            return_trace: Return processing trace

        Returns:
            output: Processed [batch, seq, dim]
            trace: Optional trace info
        """
        batch, seq, dim = x.shape

        # Pool sequence to single vector for recurrent processing
        x_pooled = self.pool_seq(x.transpose(1, 2)).squeeze(-1)  # [batch, dim]

        # Process through v6 recurrent
        out, prereq_state, trace = self.njt_v6(x_pooled, return_trace=return_trace)

        # Expand back to sequence
        out_expanded = self.expand_seq(out).unsqueeze(1).expand(-1, seq, -1)

        # Residual and norm
        output = self.norm(x + self.residual_weight * out_expanded)

        if return_trace:
            trace['prereq_state'] = prereq_state
            return output, trace

        return output, None


# =============================================================================
# TEST
# =============================================================================

def test_njt_v6():
    """Test NJT v6 recurrent system with nested memory."""
    print("=" * 70)
    print("NJT V6 RECURRENT - PREREQUISITE-AWARE REASONING")
    print("WITH NESTED MEMORY (RecursionStack)")
    print("=" * 70)

    config = NJTv6Config(
        dim=256,
        max_iterations=12,
        use_nested_memory=True,
        max_recursion_depth=4
    )
    model = NJTv6Recurrent(config)

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, config.dim)

    print("\n1. FORWARD PASS WITH NESTED MEMORY")
    print("-" * 40)
    output, final_prereq, trace = model(x, return_trace=True)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Final Prerequisites: {final_prereq.shape}")
    print(f"   Iterations used: {trace['num_iterations']}")
    print(f"   Max recursion depth: {trace['max_depth_reached']}")

    print("\n2. PREREQUISITE STATE ANALYSIS")
    print("-" * 40)
    diagnosis = model.diagnose_prerequisites(final_prereq)
    print(f"   Satisfied: {len(diagnosis['satisfied'])}")
    print(f"   Weak: {len(diagnosis['weak'])}")
    print(f"   Blocked: {len(diagnosis['blocked'])}")
    print(f"   First Block: {diagnosis['first_block']}")

    print("\n3. ITERATION TRACE (with recursion depth)")
    print("-" * 40)
    for step in trace['iterations'][:6]:  # First 6 iterations
        depth = step.get('recursion_depth', 0)
        indent = "  " * depth
        print(f"   {indent}Iter {step['iteration']}: depth={depth}, "
              f"halt={step['halt_prob']:.3f}, effect={step['effect_strength']:.3f}")

    print("\n4. STACK HISTORY (Push/Pop events)")
    print("-" * 40)
    if trace.get('stack_history'):
        for i, sh in enumerate(trace['stack_history'][:4]):
            if any(sh['did_push']) or any(sh['did_pop']):
                action = "PUSH" if any(sh['did_push']) else "POP"
                print(f"   Step {i}: {action} at depth {sh['depth'][0]}, "
                      f"block_idx={sh['block_idx'][0]}")
    else:
        print("   (No push/pop events in this run)")

    print("\n5. PARAMETER COUNT")
    print("-" * 40)
    total_params = sum(p.numel() for p in model.parameters())
    stack_params = sum(p.numel() for p in model.recursion_stack.parameters()) if model.recursion_stack else 0
    gate_params = sum(p.numel() for p in model.push_pop_gate.parameters()) if model.push_pop_gate else 0
    print(f"   Total: {total_params:,} parameters")
    print(f"   RecursionStack: {stack_params:,} parameters")
    print(f"   PushPopGate: {gate_params:,} parameters")

    print("\n6. INTEGRATION BLOCK TEST")
    print("-" * 40)
    block = TKSNJTv6Block(dim=256)
    x_seq = torch.randn(2, 16, 256)  # [batch, seq, dim]
    out_seq, _ = block(x_seq)
    print(f"   Input:  {x_seq.shape}")
    print(f"   Output: {out_seq.shape}")

    print("\n[OK] NJT v6 Recurrent with Nested Memory working!")

    print("\n" + "=" * 70)
    print("THE 12 PREREQUISITES STRUCTURE")
    print("=" * 70)
    print("""
    +---------------+-----------------+-----------------+-----------------+
    | World         | Desire (D)      | Wisdom (W)      | Power (P)       |
    +---------------+-----------------+-----------------+-----------------+
    | Spiritual (A) | [0] 1D          | [1] 2W          | [2] 3P          |
    | Mental (B)    | [3] 4D          | [4] 5W          | [5] 6P          |
    | Emotional (C) | [6] 7D          | [7] 8W          | [8] 9P          |
    | Physical (D)  | [9] 10D         | [10] 11W        | [11] 12P        |
    +---------------+-----------------+-----------------+-----------------+

    Each slot holds a scalar gradation [0.0 - 1.0]:
        0.0  = Absent/Night/None
        0.33 = Beginning/Sunrise/Learning/Birth
        0.67 = Weakening/Sunset/Knowing/Adolescent
        1.0  = Strong/Noon/Mastering/Mature
    """)


if __name__ == "__main__":
    test_njt_v6()

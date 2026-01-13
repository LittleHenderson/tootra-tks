"""
Nested NJT System - Hierarchical Gate Architecture for Deep Nesting

Problem: Current NJT handles single operations well but fails on chains:
  - "A -> B" works
  - "A -> B -> C -> D" collapses

Solution: NJT gates that feed into other NJT gates, creating a
processing stack that can handle arbitrary depth.

Three Approaches:
1. NJT Stack - Fixed depth, gates feed forward
2. NJT Recurrent - Same gate processes iteratively (like RNN)
3. NJT Fractal - Self-similar structure at multiple scales

Author: TKS Development
Date: 2025-01-10
Status: SKETCH - Conceptual design for v6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


# =============================================================================
# CORE NJT GATE (Base unit)
# =============================================================================

class NJTGateUnit(nn.Module):
    """
    Single NJT gate unit - the atomic building block.

    Can operate as NJT+ (amplify) or NJT- (inhibit) based on learned mode.
    Outputs both the processed signal AND its gate state for chaining.
    """

    def __init__(self, dim: int, num_transistors: int = 10):
        super().__init__()
        self.dim = dim
        self.num_transistors = num_transistors

        # Projections
        self.cause_proj = nn.Linear(dim, num_transistors)
        self.bias_proj = nn.Linear(dim, num_transistors)
        self.mode_proj = nn.Linear(dim, num_transistors)  # Decides +/-

        # Learnable parameters
        self.threshold = nn.Parameter(torch.ones(num_transistors) * 0.5)
        self.gain = nn.Parameter(torch.ones(num_transistors) * 2.0)
        self.sharpness = nn.Parameter(torch.tensor(10.0))

        # Output projection
        self.out_proj = nn.Linear(num_transistors, dim)

        # Gate state projection (for chaining)
        self.state_proj = nn.Linear(num_transistors, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: Input signal [batch, seq, dim]
            context: Bias context (defaults to x)
            prev_state: Previous gate state for chaining [batch, seq, dim]

        Returns:
            output: Processed signal
            state: Gate state for next level
            info: Debug info
        """
        if context is None:
            context = x

        # If we have previous state, incorporate it
        if prev_state is not None:
            context = context + prev_state

        # Project to transistor space
        cause = torch.sigmoid(self.cause_proj(x))
        bias = torch.sigmoid(self.bias_proj(context))
        mode = torch.tanh(self.mode_proj(context))  # [-1, 1]

        # Gate activation
        gate = torch.sigmoid(self.sharpness * (bias - self.threshold))

        # NJT+ path (amplify when gate open)
        njt_plus = cause * gate * self.gain

        # NJT- path (inhibit when gate open)
        njt_minus = cause * (1 - gate) * self.gain

        # Blend based on mode
        mode_weight = (mode + 1) / 2  # [0, 1]
        transistor_out = mode_weight * njt_plus + (1 - mode_weight) * njt_minus
        transistor_out = torch.clamp(transistor_out, 0, 1)

        # Output signal
        output = self.out_proj(transistor_out)

        # Gate state for chaining (carries forward what this gate "decided")
        state = self.state_proj(transistor_out * gate)  # State weighted by gate activation

        info = {
            'gate': gate,
            'mode': mode,
            'transistor_out': transistor_out,
        }

        return output, state, info


# =============================================================================
# APPROACH 1: NJT STACK (Fixed Depth Chain)
# =============================================================================

class NJTStack(nn.Module):
    """
    Stack of NJT gates where each gate's state feeds into the next.

    Architecture:
        Input
          |
        [NJT Gate 1] --state-->
          |                    |
        [NJT Gate 2] <---------+--state-->
          |                              |
        [NJT Gate 3] <-------------------+--state-->
          |                                        |
        [NJT Gate 4] <-----------------------------+
          |
        Output

    Each gate receives:
    1. The evolving signal (residual connection)
    2. The accumulated state from ALL previous gates

    This allows processing chains like A -> B -> C -> D
    where each gate handles one step of the chain.
    """

    def __init__(
        self,
        dim: int,
        num_gates: int = 4,  # Depth of stack
        num_transistors: int = 10
    ):
        super().__init__()
        self.dim = dim
        self.num_gates = num_gates

        # Stack of NJT gates
        self.gates = nn.ModuleList([
            NJTGateUnit(dim, num_transistors)
            for _ in range(num_gates)
        ])

        # State accumulator (combines states from multiple gates)
        self.state_combiner = nn.Linear(dim * num_gates, dim)

        # Layer norms between gates
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_gates)
        ])

        # Residual weights (learn how much each gate contributes)
        self.gate_weights = nn.Parameter(torch.ones(num_gates) / num_gates)

    def forward(
        self,
        x: torch.Tensor,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process input through the NJT stack.

        For input "A -> B -> C -> D":
          Gate 1: Processes A -> B
          Gate 2: Uses Gate 1's state to process B -> C
          Gate 3: Uses Gate 1+2's state to process C -> D
          Gate 4: Finalizes and outputs D
        """
        batch, seq, _ = x.shape

        states = []
        outputs = []
        trace = [] if return_trace else None

        current = x
        accumulated_state = None

        for i, (gate, norm) in enumerate(zip(self.gates, self.norms)):
            # Process through gate
            out, state, info = gate(current, prev_state=accumulated_state)

            # Normalize and residual
            out = norm(out + current)

            # Accumulate state
            states.append(state)
            outputs.append(out)

            # Combine all states so far for next gate
            if len(states) > 1:
                # Concatenate and project
                stacked = torch.cat(states, dim=-1)
                # Pad if needed
                if stacked.shape[-1] < self.dim * self.num_gates:
                    padding = torch.zeros(
                        batch, seq, self.dim * self.num_gates - stacked.shape[-1],
                        device=x.device
                    )
                    stacked = torch.cat([stacked, padding], dim=-1)
                accumulated_state = self.state_combiner(stacked)
            else:
                accumulated_state = state

            current = out

            if return_trace:
                trace.append({
                    'gate_idx': i,
                    'gate_activation': info['gate'].mean().item(),
                    'mode': info['mode'].mean().item(),
                })

        # Weighted combination of all gate outputs
        weights = F.softmax(self.gate_weights, dim=0)
        final = sum(w * o for w, o in zip(weights, outputs))

        if return_trace:
            return final, {'stack_trace': trace, 'gate_weights': weights.tolist()}
        return final, None


# =============================================================================
# APPROACH 2: NJT RECURRENT (Same Gate, Iterative Processing)
# =============================================================================

class NJTRecurrent(nn.Module):
    """
    Single NJT gate that processes iteratively, like an RNN.

    Architecture:
        Input
          |
          v
        [NJT Gate] <--+
          |          |
          +--state---+
          |
          v (after N iterations)
        Output

    The same gate processes the input multiple times,
    with its state feeding back into itself.

    Advantage: Can handle ARBITRARY depth with fixed parameters.
    The model learns when to "stop" via a halting mechanism.
    """

    def __init__(
        self,
        dim: int,
        num_transistors: int = 10,
        max_iterations: int = 8,
        use_adaptive_halt: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.max_iterations = max_iterations
        self.use_adaptive_halt = use_adaptive_halt

        # Single recurrent NJT gate
        self.gate = NJTGateUnit(dim, num_transistors)

        # Iteration embedding (like positional encoding for depth)
        self.iter_embed = nn.Embedding(max_iterations, dim)

        # Halting mechanism (learns when to stop)
        if use_adaptive_halt:
            self.halt_proj = nn.Linear(dim, 1)

        # Output projection
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        num_iterations: Optional[int] = None,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process input iteratively through the same NJT gate.

        For "A -> B -> C -> D":
          Iter 1: A -> B (state carries "processed A")
          Iter 2: B -> C (state carries "processed A, B")
          Iter 3: C -> D (state carries "processed A, B, C")
          Iter 4: Output D
        """
        batch, seq, _ = x.shape
        device = x.device

        if num_iterations is None:
            num_iterations = self.max_iterations

        trace = [] if return_trace else None

        current = x
        state = None
        halting_probs = []
        outputs = []

        for i in range(num_iterations):
            # Add iteration embedding
            iter_emb = self.iter_embed(
                torch.tensor([i], device=device)
            ).unsqueeze(0).expand(batch, seq, -1)

            current_with_iter = current + iter_emb

            # Process through gate
            out, new_state, info = self.gate(current_with_iter, prev_state=state)

            # Residual
            out = out + current
            outputs.append(out)

            # Update state
            state = new_state
            current = out

            # Adaptive halting
            if self.use_adaptive_halt:
                halt_prob = torch.sigmoid(self.halt_proj(out)).squeeze(-1)
                halting_probs.append(halt_prob)

                # If mean halt prob > 0.9, could stop early (during inference)
                # For training, we continue all iterations

            if return_trace:
                trace.append({
                    'iteration': i,
                    'gate_activation': info['gate'].mean().item(),
                    'halt_prob': halt_prob.mean().item() if self.use_adaptive_halt else None,
                })

        # Final output
        final = self.out_norm(current)

        # If using adaptive halt, weight outputs by halting probs
        if self.use_adaptive_halt and len(halting_probs) > 1:
            # Compute "remainder" halting distribution
            # (like Adaptive Computation Time from Graves)
            halt_stack = torch.stack(halting_probs, dim=-1)  # [batch, seq, num_iter]
            halt_weights = F.softmax(halt_stack, dim=-1)

            # Weighted sum of outputs
            out_stack = torch.stack(outputs, dim=-1)  # [batch, seq, dim, num_iter]
            final = (out_stack * halt_weights.unsqueeze(-2)).sum(dim=-1)

        if return_trace:
            return final, {
                'recurrent_trace': trace,
                'num_iterations': num_iterations,
            }
        return final, None


# =============================================================================
# APPROACH 3: NJT FRACTAL (Self-Similar at Multiple Scales)
# =============================================================================

class NJTFractal(nn.Module):
    """
    Fractal NJT structure - same pattern at multiple scales.

    Architecture (3 levels):

    Level 1 (Fine):     [NJT] [NJT] [NJT] [NJT] [NJT] [NJT] [NJT] [NJT]
                           \   /       \   /       \   /       \   /
    Level 2 (Medium):      [NJT]       [NJT]       [NJT]       [NJT]
                              \         /             \         /
    Level 3 (Coarse):          [NJT]                   [NJT]
                                  \                     /
    Output:                        [   Final NJT    ]

    Each level:
    - Processes pairs of inputs from level below
    - Carries state upward
    - Same gate structure at every level (fractal)

    This mirrors the Noetic Fractal concept in TKS:
    - Same rules apply at every scale
    - Output of one level is input to next
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 3,
        num_transistors: int = 10
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        # One NJT gate per level (fractal = same structure)
        # But different learned weights per level
        self.level_gates = nn.ModuleList([
            NJTGateUnit(dim, num_transistors)
            for _ in range(num_levels)
        ])

        # Pair combiner (combines two inputs for next level)
        self.pair_combine = nn.Linear(dim * 2, dim)

        # Level norms
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_levels)
        ])

        # Final projection
        self.final_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process input through fractal structure.

        For sequence of length 8:
          Level 1: Process positions [0,1], [2,3], [4,5], [6,7]
          Level 2: Combine pairs -> [0-1,2-3], [4-5,6-7]
          Level 3: Combine pairs -> [0-3, 4-7]
          Final: Combine -> [0-7]

        This naturally handles A -> B -> C -> D by:
          - Processing each operation at fine level
          - Combining results hierarchically
        """
        batch, seq, _ = x.shape
        trace = [] if return_trace else None

        current = x  # [batch, seq, dim]

        for level, (gate, norm) in enumerate(zip(self.level_gates, self.level_norms)):
            batch, cur_len, _ = current.shape

            # If odd length, pad
            if cur_len % 2 == 1:
                padding = torch.zeros(batch, 1, self.dim, device=x.device)
                current = torch.cat([current, padding], dim=1)
                cur_len += 1

            # Process pairs
            pairs = current.view(batch, cur_len // 2, 2, self.dim)

            # Combine each pair
            combined = self.pair_combine(
                pairs.view(batch, cur_len // 2, self.dim * 2)
            )  # [batch, cur_len//2, dim]

            # Process through NJT gate
            processed, state, info = gate(combined)

            # Normalize with residual
            current = norm(processed + combined)

            if return_trace:
                trace.append({
                    'level': level,
                    'input_len': cur_len,
                    'output_len': cur_len // 2,
                    'gate_activation': info['gate'].mean().item(),
                })

        # Final projection
        # Current should be [batch, 1 or small, dim]
        final = self.final_proj(current.mean(dim=1, keepdim=True))

        # Expand back to original sequence length for compatibility
        final = final.expand(-1, seq, -1)

        if return_trace:
            return final, {'fractal_trace': trace}
        return final, None


# =============================================================================
# COMBINED: NJT NESTED SYSTEM (All three approaches)
# =============================================================================

class NestedNJTSystem(nn.Module):
    """
    Complete Nested NJT System combining all three approaches.

    The system learns to route inputs to the appropriate approach:
    - Simple operations -> NJT Stack (efficient)
    - Deep chains -> NJT Recurrent (arbitrary depth)
    - Hierarchical -> NJT Fractal (scale-invariant)

    A router (also NJT-based!) decides which path to use.
    """

    def __init__(
        self,
        dim: int,
        num_transistors: int = 10,
        stack_depth: int = 4,
        recurrent_iters: int = 8,
        fractal_levels: int = 3
    ):
        super().__init__()
        self.dim = dim

        # Three approaches
        self.stack = NJTStack(dim, stack_depth, num_transistors)
        self.recurrent = NJTRecurrent(dim, num_transistors, recurrent_iters)
        self.fractal = NJTFractal(dim, fractal_levels, num_transistors)

        # Router (decides which approach to use)
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 3),  # 3 approaches
        )

        # Complexity detector (estimates nesting depth)
        self.complexity_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        force_approach: Optional[str] = None,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process input through nested NJT system.

        Args:
            x: Input [batch, seq, dim]
            force_approach: Force specific approach ('stack', 'recurrent', 'fractal')
            return_trace: Return debug info
        """
        batch, seq, _ = x.shape

        # Detect complexity
        complexity = self.complexity_detector(x.mean(dim=1))  # [batch, 1]

        # Route decision
        route_logits = self.router(x.mean(dim=1))  # [batch, 3]
        route_weights = F.softmax(route_logits, dim=-1)

        if force_approach:
            approaches = {'stack': 0, 'recurrent': 1, 'fractal': 2}
            route_weights = torch.zeros_like(route_weights)
            route_weights[:, approaches[force_approach]] = 1.0

        # Process through each approach
        stack_out, stack_trace = self.stack(x, return_trace)
        recurrent_out, recurrent_trace = self.recurrent(x, return_trace=return_trace)
        fractal_out, fractal_trace = self.fractal(x, return_trace)

        # Weighted combination
        outputs = torch.stack([stack_out, recurrent_out, fractal_out], dim=-1)
        final = (outputs * route_weights.unsqueeze(1).unsqueeze(1)).sum(dim=-1)

        if return_trace:
            return final, {
                'complexity': complexity.mean().item(),
                'route_weights': route_weights.mean(dim=0).tolist(),
                'stack': stack_trace,
                'recurrent': recurrent_trace,
                'fractal': fractal_trace,
            }
        return final, None


# =============================================================================
# INTEGRATION WITH TKS MODEL
# =============================================================================

class TKSNestedNJTBlock(nn.Module):
    """
    Drop-in replacement for standard NJT layer that uses nested structure.

    Use this in TKSGeneralLM to handle deep nesting:

    Before (v5):
        Attention -> NJTLayer -> FFN

    After (v6):
        Attention -> TKSNestedNJTBlock -> FFN
    """

    def __init__(self, dim: int, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}

        self.nested_njt = NestedNJTSystem(
            dim=dim,
            num_transistors=config.get('num_transistors', 10),
            stack_depth=config.get('stack_depth', 4),
            recurrent_iters=config.get('recurrent_iters', 8),
            fractal_levels=config.get('fractal_levels', 3),
        )

        self.norm = nn.LayerNorm(dim)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        # Nested NJT processing
        out, trace = self.nested_njt(x, return_trace=return_trace)

        # Residual connection
        final = self.norm(x + self.residual_weight * out)

        return final, trace


# =============================================================================
# TEST
# =============================================================================

def test_nested_njt():
    """Test all nested NJT approaches."""
    print("=" * 70)
    print("NESTED NJT SYSTEM - Testing All Approaches")
    print("=" * 70)

    dim = 256
    batch = 2
    seq = 16

    x = torch.randn(batch, seq, dim)

    # Test each approach
    print("\n1. NJT STACK (Fixed depth chain)")
    print("-" * 40)
    stack = NJTStack(dim, num_gates=4)
    out, trace = stack(x, return_trace=True)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Gate weights: {trace['gate_weights']}")

    print("\n2. NJT RECURRENT (Same gate, iterative)")
    print("-" * 40)
    recurrent = NJTRecurrent(dim, max_iterations=6)
    out, trace = recurrent(x, return_trace=True)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Iterations: {trace['num_iterations']}")

    print("\n3. NJT FRACTAL (Hierarchical, scale-invariant)")
    print("-" * 40)
    fractal = NJTFractal(dim, num_levels=3)
    out, trace = fractal(x, return_trace=True)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    for level_info in trace['fractal_trace']:
        print(f"   Level {level_info['level']}: {level_info['input_len']} -> {level_info['output_len']}")

    print("\n4. COMBINED SYSTEM (Router selects approach)")
    print("-" * 40)
    system = NestedNJTSystem(dim)
    out, trace = system(x, return_trace=True)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Detected complexity: {trace['complexity']:.3f}")
    print(f"   Route weights: Stack={trace['route_weights'][0]:.2f}, "
          f"Recurrent={trace['route_weights'][1]:.2f}, "
          f"Fractal={trace['route_weights'][2]:.2f}")

    # Parameter counts
    print("\n" + "=" * 70)
    print("PARAMETER COUNTS")
    print("=" * 70)
    print(f"   NJT Stack:     {sum(p.numel() for p in stack.parameters()):,}")
    print(f"   NJT Recurrent: {sum(p.numel() for p in recurrent.parameters()):,}")
    print(f"   NJT Fractal:   {sum(p.numel() for p in fractal.parameters()):,}")
    print(f"   Full System:   {sum(p.numel() for p in system.parameters()):,}")

    print("\n[OK] All Nested NJT approaches working!")

    print("\n" + "=" * 70)
    print("HOW THIS SOLVES THE SYNTACTIC NESTING PROBLEM")
    print("=" * 70)
    print("""
    Current v5 Problem:
        Input:  "A -> B -> C -> D"
        Output: "A3" (collapsed - lost the chain)

    With Nested NJT:

    1. STACK approach:
        Gate 1: Processes A -> B, outputs B, carries state "did A->B"
        Gate 2: Sees B + state, processes B -> C, outputs C
        Gate 3: Sees C + state, processes C -> D, outputs D
        Gate 4: Finalizes D with full context

    2. RECURRENT approach:
        Iter 1: Process A, state = "have A"
        Iter 2: Process B given state, state = "have A->B"
        Iter 3: Process C given state, state = "have A->B->C"
        Iter 4: Process D given state, output final D
        (Halting mechanism learns WHEN chain is complete)

    3. FRACTAL approach:
        Level 1: [A->B] [C->D] (process pairs)
        Level 2: [(A->B) -> (C->D)] (combine results)
        Final: Complete chain with hierarchy preserved

    The key insight: State propagation allows the model to
    "remember" earlier operations while processing later ones.
    Current flat NJT has no memory across operations.
    """)


if __name__ == "__main__":
    test_nested_njt()

"""
TKS-LLM Core Components v7 - Earned Depth Architecture

Goals:
- Integrate DPS (Depth Permission System) with RecursionStack
- Dynamic max_recursion_depth based on validated novelty
- Model earns deeper recursion through genuine insights
- Novelty Weight: NW = V × I × (1 - S)

v7 Updates:
- Added DPSGatingLayer for novelty-gated depth control
- RecursionStack depth is now controlled by DPS.p_max
- HEAVY novelty (NW >= 0.70) unlocks deeper thinking
- COUNT novelty accumulates toward unlock
- Maintains full v6 reasoning capabilities

Architecture:
    Input -> [v6 Blocks with NJT] -> DPS Gate -> Output
                     ^                    |
                     |                    v
              RecursionStack <---- DPS.p_max (dynamic depth limit)

The "Earned Depth" principle:
    - Start shallow (p_max=2)
    - Novel thoughts earn deeper recursion
    - Non-novel repetition stays shallow
    - Prevents runaway self-recursion
    - Rewards genuine insight
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use v6 components
from tks_llm_core_v6 import (
    TKSGeneralConfig as TKSGeneralConfigV6,
    GeneralNoeticBlockv6,
    GeneralNoeticEmbedding,
    GeneralFractalAttention,
    NoeticProcessor,
    AttractorComputationLayer,
    RPMGatingMechanism,
    _build_standardized_trace,
    TRACE_SCHEMA_VERSION
)

# DPS components
from tks_features.dps_gating import (
    DPSConfig,
    DPSState,
    DPSGatingLayer,
    NoveltyClass,
    DEFAULT_INITIAL_P_MAX,
    DEFAULT_MAX_DEPTH
)

# NJT v6 components
from tks_features.njt_v6_recurrent import (
    TKSNJTv6Block,
    NJTv6Config,
    PREREQUISITE_NAMES
)

# Regulator integration (FM, ACBE, MVR)
try:
    from tks_features.regulator_integration import (
        IntegratedRegulatorLayer,
        RegulatorConfig,
    )
    REGULATOR_AVAILABLE = True
except ImportError:
    REGULATOR_AVAILABLE = False
    IntegratedRegulatorLayer = None
    RegulatorConfig = None


@dataclass
class TKSGeneralConfigV7(TKSGeneralConfigV6):
    """v7 Config - extends v6 with DPS integration."""

    # DPS Configuration
    use_dps_gating: bool = True
    dps_initial_p_max: int = DEFAULT_INITIAL_P_MAX  # Start shallow
    dps_max_depth: int = DEFAULT_MAX_DEPTH  # Hard cap
    dps_heavy_threshold: float = 0.70  # Instant unlock
    dps_count_threshold: float = 0.45  # Token accumulation
    dps_tokens_for_unlock: int = 5  # COUNT tokens needed
    dps_cooldown_episodes: int = 10  # Anti-mania
    dps_hidden_dim: int = 64

    # v7 specific
    earned_depth_mode: bool = True  # Enable earned depth
    novelty_boost_on_heavy: bool = True  # Depth burst on HEAVY novelty

    # Regulator integration (FM/ACBE/MVR)
    use_regulators: bool = True  # Enable prerequisite regulators
    regulator_gap_threshold: float = 0.3  # Below this, trigger regulator
    regulator_on_every_block: bool = False  # Apply on every block or just specific ones


class DPSIntegratedRecursionController:
    """
    Controller that bridges DPS and RecursionStack.

    This class manages the dynamic depth limit based on DPS state.
    The RecursionStack buffer stays at a fixed max size, but this
    controller limits actual recursion to DPS.p_max.
    """

    def __init__(self, dps_state: DPSState, hard_cap: int = 5):
        self.dps_state = dps_state
        self.hard_cap = hard_cap
        self._depth_burst_active = False
        self._burst_remaining = 0

    @property
    def allowed_depth(self) -> int:
        """Current maximum allowed recursion depth."""
        base = self.dps_state.p_max
        if self._depth_burst_active and self._burst_remaining > 0:
            return min(base + 1, self.hard_cap)
        return min(base, self.hard_cap)

    def activate_depth_burst(self, duration: int = 1):
        """Temporarily allow one extra level of recursion."""
        self._depth_burst_active = True
        self._burst_remaining = duration

    def tick_burst(self):
        """Decrement burst counter after each use."""
        if self._depth_burst_active:
            self._burst_remaining -= 1
            if self._burst_remaining <= 0:
                self._depth_burst_active = False

    def can_push(self, current_depth: int) -> bool:
        """Check if we can push to a deeper level."""
        return current_depth < self.allowed_depth


class GeneralNoeticBlockv7(nn.Module):
    """
    v7 Block with DPS-controlled recursion depth.

    Extends v6 block with:
    - Novelty computation at each block
    - Dynamic depth control via DPS
    - Depth burst on HEAVY novelty
    """

    def __init__(self, config: TKSGeneralConfigV7, block_idx: int = 0):
        super().__init__()
        self.dim = config.hidden_dim
        self.config = config
        self.block_idx = block_idx

        # Inherit v6 block structure
        self.v6_block = GeneralNoeticBlockv6(config)

        # DPS integration (only on specific blocks, e.g., every 3rd)
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
            )
            self.dps_layer = DPSGatingLayer(dps_config, noetic_dim=config.hidden_dim)
        else:
            self.dps_layer = None

        # Regulator integration (FM/ACBE/MVR for D/W/P gaps)
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
                noetic_dim=min(config.hidden_dim, 40),  # Regulators work on 40-dim noetic space
                config=reg_config,
            )
        else:
            self.regulator = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        dps_state: Optional[DPSState] = None,
        depth_controller: Optional[DPSIntegratedRecursionController] = None,
        return_trace: bool = False,
        episode_id: str = "unknown",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict], Optional[DPSState]]:
        """
        Forward pass with DPS-controlled depth.

        Args:
            x: Input tensor [batch, seq, dim]
            attention_mask: Optional attention mask
            step: Training step for routing scheduler
            dps_state: Current DPS state (for novelty tracking)
            depth_controller: Controller managing recursion depth
            return_trace: Whether to return detailed trace
            episode_id: Episode ID for DPS audit

        Returns:
            output: Processed tensor
            aux_loss: Routing auxiliary loss
            trace: Optional trace dict
            new_dps_state: Updated DPS state (if DPS active)
        """
        # 1. Run v6 block (includes NJT with RecursionStack)
        # Temporarily override max_depth if controller provided
        if depth_controller is not None and self.v6_block.njt is not None:
            # Store original and set dynamic limit
            original_max = self.v6_block.njt.njt_v6.recursion_stack.max_depth
            self.v6_block.njt.njt_v6.recursion_stack.max_depth = depth_controller.allowed_depth

        out, aux_loss, _, njt_trace, router_metrics = self.v6_block(
            x, attention_mask, step, return_njt_trace=return_trace
        )

        # Restore original max_depth
        if depth_controller is not None and self.v6_block.njt is not None:
            self.v6_block.njt.njt_v6.recursion_stack.max_depth = original_max

        # 2. DPS gating (if active on this block)
        new_dps_state = dps_state
        dps_trace = None

        if self.has_dps and self.dps_layer is not None and dps_state is not None:
            out, new_dps_state, dps_trace = self.dps_layer(
                out,
                dps_state,
                memory_embeddings=None,  # Could add memory bank here
                episode_id=episode_id,
            )

            # Check for depth burst on HEAVY novelty
            if dps_trace.get("unlock_occurred", False) and depth_controller is not None:
                if self.config.novelty_boost_on_heavy:
                    depth_controller.activate_depth_burst(duration=1)

        # 3. Apply regulators (FM/ACBE/MVR) for D/W/P gaps
        regulator_trace = None
        if self.has_regulator and self.regulator is not None:
            # Project to noetic space if needed (regulators work on 40-dim)
            noetic_dim = min(out.shape[-1], 40)
            out_noetic = out[..., :noetic_dim]  # Take first 40 dims

            # Apply regulators
            out_regulated, gap_info = self.regulator(out_noetic, return_gap_info=return_trace)

            # Merge back (replace first 40 dims with regulated output)
            out = out.clone()
            out[..., :noetic_dim] = out_regulated

            # Add regulator loss to aux_loss
            if gap_info is not None:
                reg_loss = self.regulator.compute_regulator_loss(gap_info)
                if isinstance(aux_loss, torch.Tensor):
                    aux_loss = aux_loss + reg_loss * 0.1
                else:
                    aux_loss = reg_loss * 0.1

            if return_trace and gap_info is not None:
                regulator_trace = {
                    "desire_score": gap_info['desire_score'].mean().item(),
                    "wisdom_score": gap_info['wisdom_score'].mean().item(),
                    "power_score": gap_info['power_score'].mean().item(),
                    "desire_gap": gap_info['desire_gap'].mean().item(),
                    "wisdom_gap": gap_info['wisdom_gap'].mean().item(),
                    "power_gap": gap_info['power_gap'].mean().item(),
                }

        # 4. Build combined trace
        trace = None
        if return_trace:
            trace = {
                "block_idx": self.block_idx,
                "njt": njt_trace,
                "dps": dps_trace,
                "regulator": regulator_trace,
                "router": router_metrics,
                "allowed_depth": depth_controller.allowed_depth if depth_controller else None,
            }

        return out, aux_loss, trace, new_dps_state


class TKSGeneralLMv7(nn.Module):
    """
    TKS General Language Model v7 - Earned Depth Architecture.

    This model integrates DPS with the v6 reasoning engine:
    - RecursionStack depth controlled by DPS.p_max
    - Novel thoughts earn deeper recursion
    - Maintains full v6 capabilities
    """

    def __init__(self, config: TKSGeneralConfigV7):
        super().__init__()
        self.config = config

        # Token embedding (from v5/v6)
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(2048, config.hidden_dim)

        # v7 Blocks with DPS integration
        self.blocks = nn.ModuleList([
            GeneralNoeticBlockv7(config, block_idx=i)
            for i in range(config.num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_emb.weight

        # DPS state (persistent across forward passes during inference)
        self._dps_state = DPSState(
            p_max=config.dps_initial_p_max,
            tokens=0,
            cooldown=0,
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    @property
    def dps_state(self) -> DPSState:
        """Current DPS state."""
        return self._dps_state

    def reset_dps_state(self):
        """Reset DPS state to initial values."""
        self._dps_state = DPSState(
            p_max=self.config.dps_initial_p_max,
            tokens=0,
            cooldown=0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        return_full_trace: bool = False,
        episode_id: str = "unknown",
        external_dps_state: Optional[DPSState] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with earned depth control.

        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: Optional mask [batch, seq]
            step: Training step
            return_full_trace: Return detailed execution trace
            episode_id: Episode ID for DPS audit
            external_dps_state: Override internal DPS state

        Returns:
            Dict with:
                - logits: [batch, seq, vocab]
                - routing_aux_loss: Scalar
                - trace: Optional detailed trace
                - dps_state: Updated DPS state
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Use external or internal DPS state
        dps_state = external_dps_state if external_dps_state is not None else self._dps_state

        # Create depth controller
        depth_controller = DPSIntegratedRecursionController(
            dps_state,
            hard_cap=self.config.dps_max_depth
        )

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Process through v7 blocks
        total_aux_loss = 0.0
        traces = []

        for block in self.blocks:
            x, aux_loss, trace, dps_state = block(
                x,
                attention_mask=attention_mask,
                step=step,
                dps_state=dps_state,
                depth_controller=depth_controller,
                return_trace=return_full_trace,
                episode_id=episode_id,
            )
            # Ensure aux_loss is scalar before accumulating
            if aux_loss is not None:
                if isinstance(aux_loss, torch.Tensor):
                    if aux_loss.numel() > 1:
                        aux_loss = aux_loss.mean()
                    elif aux_loss.numel() == 1:
                        aux_loss = aux_loss.squeeze()
                total_aux_loss = total_aux_loss + aux_loss
            if trace is not None:
                traces.append(trace)

            # Tick depth burst counter
            depth_controller.tick_burst()

        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)

        # Update internal state
        self._dps_state = dps_state

        # Build output
        output = {
            "logits": logits,
            "routing_aux_loss": total_aux_loss,
            "dps_state": dps_state,
            "allowed_depth": depth_controller.allowed_depth,
        }

        if return_full_trace:
            output["trace"] = {
                "blocks": traces,
                "dps": {
                    "p_max": dps_state.p_max,
                    "tokens": dps_state.tokens,
                    "cooldown": dps_state.cooldown,
                    "total_unlocks": dps_state.total_unlocks,
                },
                "schema_version": TRACE_SCHEMA_VERSION,
            }

        return output

    def get_depth_stats(self) -> Dict[str, int]:
        """Get current depth permission stats."""
        return {
            "p_max": self._dps_state.p_max,
            "tokens": self._dps_state.tokens,
            "cooldown": self._dps_state.cooldown,
            "total_unlocks": self._dps_state.total_unlocks,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_v7_from_v6(
    v6_checkpoint_path: str,
    config: Optional[TKSGeneralConfigV7] = None,
    device: str = "cuda",
) -> TKSGeneralLMv7:
    """
    Create v7 model initialized from v6 checkpoint.

    Args:
        v6_checkpoint_path: Path to v6 checkpoint
        config: Optional v7 config (uses defaults if None)
        device: Device to load to

    Returns:
        TKSGeneralLMv7 with v6 weights migrated
    """
    import torch
    from pathlib import Path

    # Load v6 checkpoint
    v6_state = torch.load(v6_checkpoint_path, map_location=device, weights_only=False)

    # Infer config from checkpoint if not provided
    if config is None:
        # Get vocab size from embedding
        vocab_size = v6_state.get("token_emb.weight", v6_state.get("lm_head.weight")).shape[0]
        config = TKSGeneralConfigV7(vocab_size=vocab_size)

    # Create v7 model
    model = TKSGeneralLMv7(config)

    # Migrate v6 weights
    # The v7 model wraps v6 blocks, so we need to remap keys
    v7_state = model.state_dict()

    migrated = 0
    for key, value in v6_state.items():
        # Skip stack buffers (they're reset each forward)
        if "stack_memory" in key or "stack_ptr" in key:
            continue

        # Direct match
        if key in v7_state and v7_state[key].shape == value.shape:
            v7_state[key] = value
            migrated += 1
            continue

        # Try remapping: blocks.N.X -> blocks.N.v6_block.X
        if key.startswith("blocks."):
            parts = key.split(".", 2)
            if len(parts) >= 3:
                new_key = f"blocks.{parts[1]}.v6_block.{parts[2]}"
                if new_key in v7_state and v7_state[new_key].shape == value.shape:
                    v7_state[new_key] = value
                    migrated += 1

    model.load_state_dict(v7_state, strict=False)
    print(f"Migrated {migrated} weights from v6 to v7")

    # Reset DPS thresholds to v7 defaults
    for block in model.blocks:
        if hasattr(block, 'dps_layer') and block.dps_layer is not None:
            # Ensure thresholds are fresh
            pass  # DPS layer initializes with config defaults

    return model.to(device)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TKSGeneralConfigV7",
    "TKSGeneralLMv7",
    "GeneralNoeticBlockv7",
    "DPSIntegratedRecursionController",
    "create_v7_from_v6",
]

"""
TKS Cost Metrics

Computes various costs associated with an agent episode for reward calculation.
Includes tool call costs, latency, token usage, and uncertainty costs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time


class ToolType(Enum):
    """Categories of tools with different cost weights."""
    READ = "read"           # File reading, web fetching - low cost
    SEARCH = "search"       # Grep, glob, search - low cost
    WRITE = "write"         # File writing, editing - medium cost
    EXECUTE = "execute"     # Shell commands, code execution - high cost
    API = "api"             # External API calls - high cost
    LLM = "llm"             # LLM inference calls - very high cost
    UNKNOWN = "unknown"     # Unknown tools - medium cost


@dataclass
class ToolCall:
    """Represents a single tool call in an episode."""
    tool_name: str
    tool_type: ToolType
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrace:
    """Complete trace of an episode for cost computation."""
    tool_calls: List[ToolCall] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    uncertainty_initial: float = 1.0
    uncertainty_final: float = 0.0
    budget_limit: Optional[float] = None

    # Additional metadata
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    model_id: str = ""


@dataclass
class CostBreakdown:
    """Detailed breakdown of all costs in an episode."""
    tool_calls_cost: float
    latency_cost: float
    token_cost: float
    uncertainty_cost: float
    total_cost: float

    # Per-type breakdown
    tool_type_costs: Dict[str, float] = field(default_factory=dict)

    # Raw metrics (not normalized)
    raw_metrics: Dict[str, float] = field(default_factory=dict)

    # Budget information
    budget_used_fraction: float = 0.0
    budget_exceeded: bool = False


class CostMetrics:
    """
    Computes cost metrics for agent episodes.

    All costs are normalized to a 0-1 scale where:
    - 0.0 = no cost (ideal)
    - 1.0 = maximum acceptable cost
    - >1.0 = cost exceeded acceptable limits
    """

    # Default tool type weights (relative cost per call)
    DEFAULT_TOOL_WEIGHTS: Dict[ToolType, float] = {
        ToolType.READ: 0.05,
        ToolType.SEARCH: 0.05,
        ToolType.WRITE: 0.15,
        ToolType.EXECUTE: 0.25,
        ToolType.API: 0.30,
        ToolType.LLM: 0.50,
        ToolType.UNKNOWN: 0.10,
    }

    # Default baselines for normalization
    DEFAULT_BASELINES = {
        "max_tool_calls": 50,           # Maximum expected tool calls
        "max_latency_ms": 120_000,      # 2 minutes max
        "max_tokens": 50_000,           # Maximum token budget
        "token_cost_per_1k": 0.01,      # Relative cost per 1k tokens
    }

    def __init__(
        self,
        tool_weights: Optional[Dict[ToolType, float]] = None,
        baselines: Optional[Dict[str, float]] = None,
        latency_penalty_factor: float = 1.0,
        token_penalty_factor: float = 1.0
    ):
        """
        Initialize CostMetrics calculator.

        Args:
            tool_weights: Custom weights for each tool type
            baselines: Custom baseline values for normalization
            latency_penalty_factor: Multiplier for latency cost
            token_penalty_factor: Multiplier for token cost
        """
        self.tool_weights = tool_weights or self.DEFAULT_TOOL_WEIGHTS.copy()
        self.baselines = baselines or self.DEFAULT_BASELINES.copy()
        self.latency_penalty_factor = latency_penalty_factor
        self.token_penalty_factor = token_penalty_factor

    def compute_cost(self, episode_trace: EpisodeTrace) -> CostBreakdown:
        """
        Compute all costs for an episode.

        Args:
            episode_trace: Complete trace of the episode

        Returns:
            CostBreakdown with all cost components
        """
        # Compute individual cost components
        tool_calls_cost, tool_type_costs = self._compute_tool_calls_cost(
            episode_trace.tool_calls
        )
        latency_cost = self._compute_latency_cost(episode_trace)
        token_cost = self._compute_token_cost(episode_trace)
        uncertainty_cost = self._compute_uncertainty_cost(episode_trace)

        # Total cost (weighted sum)
        total_cost = (
            tool_calls_cost * 0.25 +
            latency_cost * 0.25 +
            token_cost * 0.30 +
            uncertainty_cost * 0.20
        )

        # Budget tracking
        budget_used_fraction = 0.0
        budget_exceeded = False
        if episode_trace.budget_limit and episode_trace.budget_limit > 0:
            actual_cost = self._compute_raw_cost(episode_trace)
            budget_used_fraction = actual_cost / episode_trace.budget_limit
            budget_exceeded = budget_used_fraction > 1.0

        # Raw metrics for debugging/analysis
        raw_metrics = {
            "tool_call_count": len(episode_trace.tool_calls),
            "duration_ms": episode_trace.total_duration_ms,
            "tokens_in": episode_trace.total_tokens_in,
            "tokens_out": episode_trace.total_tokens_out,
            "total_tokens": episode_trace.total_tokens_in + episode_trace.total_tokens_out,
            "uncertainty_initial": episode_trace.uncertainty_initial,
            "uncertainty_final": episode_trace.uncertainty_final,
        }

        return CostBreakdown(
            tool_calls_cost=tool_calls_cost,
            latency_cost=latency_cost,
            token_cost=token_cost,
            uncertainty_cost=uncertainty_cost,
            total_cost=total_cost,
            tool_type_costs=tool_type_costs,
            raw_metrics=raw_metrics,
            budget_used_fraction=budget_used_fraction,
            budget_exceeded=budget_exceeded
        )

    def _compute_tool_calls_cost(
        self,
        tool_calls: List[ToolCall]
    ) -> tuple[float, Dict[str, float]]:
        """
        Compute cost of tool calls based on type and count.

        Returns:
            Tuple of (normalized total cost, per-type cost breakdown)
        """
        if not tool_calls:
            return 0.0, {}

        type_costs: Dict[str, float] = {}
        total_weighted_cost = 0.0

        for call in tool_calls:
            weight = self.tool_weights.get(call.tool_type, 0.1)

            # Failed calls have additional cost
            if not call.success:
                weight *= 1.5

            total_weighted_cost += weight

            # Track per-type
            type_name = call.tool_type.value
            type_costs[type_name] = type_costs.get(type_name, 0.0) + weight

        # Normalize by maximum expected
        max_cost = self.baselines["max_tool_calls"] * 0.15  # Average weight
        normalized_cost = min(1.5, total_weighted_cost / max_cost)

        return normalized_cost, type_costs

    def _compute_latency_cost(self, episode_trace: EpisodeTrace) -> float:
        """
        Compute cost based on time taken.

        Longer execution time = higher cost.
        """
        duration_ms = episode_trace.total_duration_ms

        # If we have start/end times, use those
        if episode_trace.start_time and episode_trace.end_time:
            duration_ms = (episode_trace.end_time - episode_trace.start_time) * 1000

        # Normalize against maximum expected latency
        max_latency = self.baselines["max_latency_ms"]
        normalized_cost = (duration_ms / max_latency) * self.latency_penalty_factor

        return min(1.5, normalized_cost)  # Cap at 1.5 for extreme cases

    def _compute_token_cost(self, episode_trace: EpisodeTrace) -> float:
        """
        Compute cost based on token usage.

        More tokens = higher cost.
        """
        total_tokens = (
            episode_trace.total_tokens_in +
            episode_trace.total_tokens_out
        )

        # Output tokens typically cost more
        weighted_tokens = (
            episode_trace.total_tokens_in +
            episode_trace.total_tokens_out * 3  # Output is 3x cost
        )

        # Normalize against maximum
        max_tokens = self.baselines["max_tokens"]
        normalized_cost = (weighted_tokens / max_tokens) * self.token_penalty_factor

        return min(1.5, normalized_cost)

    def _compute_uncertainty_cost(self, episode_trace: EpisodeTrace) -> float:
        """
        Compute cost based on remaining uncertainty.

        High remaining uncertainty = higher cost.
        """
        # Remaining uncertainty directly translates to cost
        # 0.0 final uncertainty = 0.0 cost
        # 1.0 final uncertainty = 1.0 cost
        return episode_trace.uncertainty_final

    def _compute_raw_cost(self, episode_trace: EpisodeTrace) -> float:
        """
        Compute raw (non-normalized) cost for budget tracking.

        This could represent actual monetary cost, compute units, etc.
        """
        # Simple model: cost based on tokens + tool calls + time
        token_cost = (
            (episode_trace.total_tokens_in + episode_trace.total_tokens_out)
            * self.baselines["token_cost_per_1k"] / 1000
        )

        tool_cost = sum(
            self.tool_weights.get(call.tool_type, 0.1)
            for call in episode_trace.tool_calls
        )

        time_cost = episode_trace.total_duration_ms / 1000 * 0.001  # $0.001/sec

        return token_cost + tool_cost + time_cost

    @staticmethod
    def classify_tool(tool_name: str) -> ToolType:
        """
        Classify a tool by name into a ToolType category.

        Args:
            tool_name: Name of the tool

        Returns:
            Appropriate ToolType classification
        """
        tool_name_lower = tool_name.lower()

        # Read operations
        if any(x in tool_name_lower for x in ["read", "get", "fetch", "load", "view"]):
            return ToolType.READ

        # Search operations
        if any(x in tool_name_lower for x in ["search", "grep", "glob", "find", "query"]):
            return ToolType.SEARCH

        # Write operations
        if any(x in tool_name_lower for x in ["write", "edit", "update", "create", "save"]):
            return ToolType.WRITE

        # Execute operations
        if any(x in tool_name_lower for x in ["bash", "execute", "run", "shell", "command"]):
            return ToolType.EXECUTE

        # API operations
        if any(x in tool_name_lower for x in ["api", "http", "request", "web"]):
            return ToolType.API

        # LLM operations
        if any(x in tool_name_lower for x in ["llm", "chat", "complete", "generate", "model"]):
            return ToolType.LLM

        return ToolType.UNKNOWN


def create_tool_call(
    tool_name: str,
    duration_ms: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    success: bool = True,
    **metadata
) -> ToolCall:
    """
    Convenience factory for creating ToolCall objects.

    Args:
        tool_name: Name of the tool
        duration_ms: How long the call took
        tokens_in: Input tokens used
        tokens_out: Output tokens generated
        success: Whether the call succeeded
        **metadata: Additional metadata

    Returns:
        ToolCall object with classified tool type
    """
    return ToolCall(
        tool_name=tool_name,
        tool_type=CostMetrics.classify_tool(tool_name),
        duration_ms=duration_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        success=success,
        metadata=metadata
    )

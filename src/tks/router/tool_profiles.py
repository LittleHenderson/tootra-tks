"""
TKS Tool Capability Profiles

Defines profiles for each tool that specify:
- Capability thresholds required to use the tool
- Reversibility classification (reversible, partial, irreversible)
- Mode constraints (which operational modes allow the tool)
- Evidence requirements

These profiles are used by the Router to enforce governance rails
before routing RPM prerequisites to tools.

Author: ARCH agent
Date: 2026-01-05
Schema Version: 1.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from ..config import OperationalMode


class ToolReversibility(Enum):
    """Classification of how reversible a tool's actions are."""
    REVERSIBLE = "reversible"           # Can fully undo (e.g., memory_read)
    PARTIAL = "partial"                 # Can partially undo (e.g., file_write with backup)
    IRREVERSIBLE = "irreversible"       # Cannot undo (e.g., file_delete, network send)


@dataclass(frozen=True)
class ToolProfile:
    """
    Profile defining a tool's capabilities and constraints.

    Attributes:
        name: Unique tool identifier
        capability_threshold: Minimum capability score [0, 1] required to use this tool
        reversibility: How reversible the tool's actions are
        mode_constraints: Set of operational modes where tool is allowed
        evidence_required: Whether tool calls must be bound to evidence
        description: Human-readable description of what the tool does
        requires_network: Whether tool needs network access
        requires_filesystem: Whether tool needs filesystem access
        requires_code_execution: Whether tool executes arbitrary code
        sandbox_required: Whether tool must run in sandbox environment
        max_calls_per_episode: Maximum times tool can be called in one episode (0 = no limit)
        cost_estimate: Estimated resource cost per invocation
    """
    name: str
    capability_threshold: float = 0.0
    reversibility: ToolReversibility = ToolReversibility.REVERSIBLE
    mode_constraints: frozenset = field(default_factory=lambda: frozenset({
        OperationalMode.NORMAL,
        OperationalMode.HIGH_STAKES,
        OperationalMode.CRITICAL,
    }))
    evidence_required: bool = False
    description: str = ""
    requires_network: bool = False
    requires_filesystem: bool = False
    requires_code_execution: bool = False
    sandbox_required: bool = False
    max_calls_per_episode: int = 0  # 0 = no specific limit (use global budget)
    cost_estimate: float = 0.0

    def is_allowed_in_mode(self, mode: OperationalMode) -> bool:
        """Check if tool is allowed in the given operational mode."""
        return mode in self.mode_constraints

    def meets_capability(self, capability_score: float) -> bool:
        """Check if the provided capability score meets the threshold."""
        return capability_score >= self.capability_threshold


# =============================================================================
# TOOL REGISTRY - Default Tool Profiles
# =============================================================================

TOOL_REGISTRY: Dict[str, ToolProfile] = {
    # ---------------------------------------------------------------------
    # MEMORY TOOLS - Read/write to agent's working memory
    # ---------------------------------------------------------------------
    "memory_read": ToolProfile(
        name="memory_read",
        capability_threshold=0.0,  # Always allowed
        reversibility=ToolReversibility.REVERSIBLE,
        mode_constraints=frozenset({
            OperationalMode.NORMAL,
            OperationalMode.HIGH_STAKES,
            OperationalMode.CRITICAL,
        }),
        evidence_required=False,
        description="Read from agent's working memory or context store",
        requires_network=False,
        requires_filesystem=False,
        requires_code_execution=False,
        sandbox_required=False,
        max_calls_per_episode=0,  # No limit - reads are cheap
        cost_estimate=0.01,
    ),

    "memory_write": ToolProfile(
        name="memory_write",
        capability_threshold=0.1,  # Low threshold
        reversibility=ToolReversibility.PARTIAL,  # Can overwrite but history may be lost
        mode_constraints=frozenset({
            OperationalMode.NORMAL,
            OperationalMode.HIGH_STAKES,
        }),  # Not in Critical - no state changes
        evidence_required=False,
        description="Write to agent's working memory or context store",
        requires_network=False,
        requires_filesystem=False,
        requires_code_execution=False,
        sandbox_required=False,
        max_calls_per_episode=50,
        cost_estimate=0.02,
    ),

    # ---------------------------------------------------------------------
    # SEARCH & RETRIEVAL TOOLS
    # ---------------------------------------------------------------------
    "search": ToolProfile(
        name="search",
        capability_threshold=0.1,
        reversibility=ToolReversibility.REVERSIBLE,
        mode_constraints=frozenset({
            OperationalMode.NORMAL,
            OperationalMode.HIGH_STAKES,
            OperationalMode.CRITICAL,
        }),
        evidence_required=False,
        description="Search through indexed content (knowledge base, documents)",
        requires_network=False,  # Could be True for web search
        requires_filesystem=False,
        requires_code_execution=False,
        sandbox_required=False,
        max_calls_per_episode=0,
        cost_estimate=0.05,
    ),

    "summarize": ToolProfile(
        name="summarize",
        capability_threshold=0.2,
        reversibility=ToolReversibility.REVERSIBLE,
        mode_constraints=frozenset({
            OperationalMode.NORMAL,
            OperationalMode.HIGH_STAKES,
            OperationalMode.CRITICAL,
        }),
        evidence_required=False,
        description="Summarize content or generate synthesis of multiple sources",
        requires_network=False,
        requires_filesystem=False,
        requires_code_execution=False,
        sandbox_required=False,
        max_calls_per_episode=0,
        cost_estimate=0.1,
    ),

    # ---------------------------------------------------------------------
    # CODE EXECUTION TOOLS
    # ---------------------------------------------------------------------
    "code_execute": ToolProfile(
        name="code_execute",
        capability_threshold=0.5,  # Higher threshold - risky
        reversibility=ToolReversibility.PARTIAL,  # Side effects may not be undone
        mode_constraints=frozenset({
            OperationalMode.NORMAL,
            OperationalMode.HIGH_STAKES,
        }),  # Not in Critical
        evidence_required=True,  # Must justify code execution
        description="Execute code in sandboxed environment",
        requires_network=False,  # Sandbox restricts network
        requires_filesystem=False,  # Sandbox restricts FS
        requires_code_execution=True,
        sandbox_required=True,  # MUST run sandboxed
        max_calls_per_episode=10,
        cost_estimate=0.5,
    ),

    # ---------------------------------------------------------------------
    # FILE SYSTEM TOOLS
    # ---------------------------------------------------------------------
    "file_read": ToolProfile(
        name="file_read",
        capability_threshold=0.2,
        reversibility=ToolReversibility.REVERSIBLE,
        mode_constraints=frozenset({
            OperationalMode.NORMAL,
            OperationalMode.HIGH_STAKES,
            OperationalMode.CRITICAL,
        }),
        evidence_required=False,
        description="Read content from files within allowed paths",
        requires_network=False,
        requires_filesystem=True,
        requires_code_execution=False,
        sandbox_required=False,
        max_calls_per_episode=0,
        cost_estimate=0.02,
    ),

    "file_write": ToolProfile(
        name="file_write",
        capability_threshold=0.4,
        reversibility=ToolReversibility.PARTIAL,  # Can overwrite but may lose data
        mode_constraints=frozenset({
            OperationalMode.NORMAL,
            OperationalMode.HIGH_STAKES,
        }),  # Not in Critical
        evidence_required=True,  # Must justify writes
        description="Write content to files within allowed paths",
        requires_network=False,
        requires_filesystem=True,
        requires_code_execution=False,
        sandbox_required=False,
        max_calls_per_episode=20,
        cost_estimate=0.1,
    ),

    "file_delete": ToolProfile(
        name="file_delete",
        capability_threshold=0.7,  # High threshold - destructive
        reversibility=ToolReversibility.IRREVERSIBLE,
        mode_constraints=frozenset({
            OperationalMode.NORMAL,
        }),  # Only in Normal, requires explicit clearance
        evidence_required=True,  # Must justify deletion
        description="Delete files within allowed paths (IRREVERSIBLE)",
        requires_network=False,
        requires_filesystem=True,
        requires_code_execution=False,
        sandbox_required=False,
        max_calls_per_episode=5,
        cost_estimate=0.2,
    ),
}


# =============================================================================
# TOOL CATEGORIES
# =============================================================================

TOOL_CATEGORIES: Dict[str, List[str]] = {
    "memory": ["memory_read", "memory_write"],
    "retrieval": ["search", "summarize"],
    "execution": ["code_execute"],
    "filesystem": ["file_read", "file_write", "file_delete"],
}


def get_tools_by_category(category: str) -> List[ToolProfile]:
    """Get all tool profiles in a category."""
    tool_names = TOOL_CATEGORIES.get(category, [])
    return [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]


def get_tools_for_mode(mode: OperationalMode) -> List[ToolProfile]:
    """Get all tools allowed in a specific operational mode."""
    return [
        profile for profile in TOOL_REGISTRY.values()
        if profile.is_allowed_in_mode(mode)
    ]


def get_safe_tools() -> List[ToolProfile]:
    """Get all tools that are reversible and allowed in all modes."""
    return [
        profile for profile in TOOL_REGISTRY.values()
        if profile.reversibility == ToolReversibility.REVERSIBLE
        and len(profile.mode_constraints) == 3  # Allowed in all modes
    ]


def get_risky_tools() -> List[ToolProfile]:
    """Get all tools classified as partial or irreversible."""
    return [
        profile for profile in TOOL_REGISTRY.values()
        if profile.reversibility in (ToolReversibility.PARTIAL, ToolReversibility.IRREVERSIBLE)
    ]


def register_tool(profile: ToolProfile) -> None:
    """Register a new tool profile in the registry."""
    if profile.name in TOOL_REGISTRY:
        raise ValueError(f"Tool '{profile.name}' already registered")
    TOOL_REGISTRY[profile.name] = profile


def unregister_tool(name: str) -> Optional[ToolProfile]:
    """Remove a tool profile from the registry."""
    return TOOL_REGISTRY.pop(name, None)


# =============================================================================
# BUDGET CALCULATIONS
# =============================================================================

# Budget limits by operational mode
TOOL_CALL_BUDGETS: Dict[OperationalMode, int] = {
    OperationalMode.NORMAL: 12,
    OperationalMode.HIGH_STAKES: 8,
    OperationalMode.CRITICAL: 0,  # No tool calls in Critical until verified
}

# RPM depth caps by mode
RPM_DEPTH_CAPS: Dict[OperationalMode, int] = {
    OperationalMode.NORMAL: 3,
    OperationalMode.HIGH_STAKES: 2,
    OperationalMode.CRITICAL: 1,
}

# RPM breadth cap (same for all modes)
RPM_BREADTH_CAP: int = 5


def get_budget_for_mode(mode: OperationalMode) -> int:
    """Get maximum tool calls allowed for a mode."""
    return TOOL_CALL_BUDGETS.get(mode, 0)


def get_depth_cap_for_mode(mode: OperationalMode) -> int:
    """Get RPM depth cap for a mode."""
    return RPM_DEPTH_CAPS.get(mode, 1)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ToolReversibility",
    "OperationalMode",
    # Main types
    "ToolProfile",
    "TOOL_REGISTRY",
    # Categories
    "TOOL_CATEGORIES",
    "get_tools_by_category",
    "get_tools_for_mode",
    "get_safe_tools",
    "get_risky_tools",
    # Registration
    "register_tool",
    "unregister_tool",
    # Budgets
    "TOOL_CALL_BUDGETS",
    "RPM_DEPTH_CAPS",
    "RPM_BREADTH_CAP",
    "get_budget_for_mode",
    "get_depth_cap_for_mode",
]

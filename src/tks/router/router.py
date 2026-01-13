"""
TKS Router Module

Routes RPM (Desire/Wisdom/Power) prerequisites to appropriate tools.
The router:
1. Receives prerequisites from the RPM decomposition
2. Checks governance rails BEFORE routing
3. Enforces budgets (max_tool_calls, max_depth, max_breadth)
4. Returns a RouterDecision with tool selection and parameters

Flow: input -> RPM -> router -> tools -> evidence -> output

Author: ARCH agent
Date: 2026-01-05
Schema Version: 1.0
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .tool_profiles import (
    TOOL_REGISTRY,
    ToolProfile,
    ToolReversibility,
    OperationalMode,
    TOOL_CALL_BUDGETS,
    RPM_DEPTH_CAPS,
    RPM_BREADTH_CAP,
    get_budget_for_mode,
    get_depth_cap_for_mode,
)


# =============================================================================
# PREREQUISITE TYPES
# =============================================================================

class PrerequisiteType(Enum):
    """Types of RPM prerequisites that can be routed."""
    KNOWLEDGE = "knowledge"           # Need to retrieve/read information
    COMPUTATION = "computation"       # Need to execute/calculate
    STATE_UPDATE = "state_update"     # Need to modify memory/state
    VERIFICATION = "verification"     # Need to verify/validate
    SYNTHESIS = "synthesis"           # Need to combine/summarize
    EXTERNAL_ACTION = "external"      # Need to affect external world


@dataclass
class Prerequisite:
    """
    An RPM prerequisite that needs to be satisfied.

    Prerequisites are generated during RPM decomposition and
    represent sub-tasks that must be completed.
    """
    prereq_id: str
    prereq_type: PrerequisiteType
    description: str
    required_inputs: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1 = highest priority
    parent_node_id: Optional[str] = None
    depth: int = 0
    evidence_required: bool = False
    estimated_complexity: float = 0.5  # [0, 1]


# =============================================================================
# ROUTING CONTEXT
# =============================================================================

@dataclass
class RoutingContext:
    """
    Context for routing decisions.

    Contains state needed to make routing decisions including
    current operational mode, budget consumption, and governance info.
    """
    episode_id: str
    current_mode: OperationalMode = OperationalMode.NORMAL
    tool_calls_made: int = 0
    current_depth: int = 0
    current_breadth: int = 0
    uncertainty: float = 0.0
    stakes: float = 0.0
    alignment: float = 1.0
    has_evidence: bool = True
    clearance_tokens: List[str] = field(default_factory=list)
    allowed_tools: Optional[Set[str]] = None
    allowed_paths: Optional[Set[str]] = None
    network_enabled: bool = False
    sandbox_enabled: bool = True
    tool_history: List[str] = field(default_factory=list)

    def get_budget_remaining(self) -> int:
        """Get remaining tool call budget."""
        max_budget = get_budget_for_mode(self.current_mode)
        return max(0, max_budget - self.tool_calls_made)

    def can_make_tool_call(self) -> bool:
        """Check if budget allows another tool call."""
        return self.get_budget_remaining() > 0

    def depth_allowed(self) -> bool:
        """Check if current depth is within limits."""
        max_depth = get_depth_cap_for_mode(self.current_mode)
        return self.current_depth <= max_depth

    def breadth_allowed(self) -> bool:
        """Check if current breadth is within limits."""
        return self.current_breadth <= RPM_BREADTH_CAP


# =============================================================================
# GOVERNANCE CHECK RESULT
# =============================================================================

class GovernanceDecision(Enum):
    """Result of governance rail check."""
    ALLOW = "ALLOW"
    PAUSE = "PAUSE"
    BLOCK = "BLOCK"


@dataclass
class GovernanceCheckResult:
    """
    Result from checking governance rails.
    """
    decision: GovernanceDecision
    reason: str
    mode: OperationalMode
    required_actions: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    clearance_required: bool = False


# =============================================================================
# ROUTER DECISION
# =============================================================================

@dataclass
class RouterDecision:
    """
    Decision output from the router.

    Contains everything needed to execute a tool call.
    """
    decision_id: str
    tool_name: str
    action_params: Dict[str, Any]
    evidence_requirements: List[str]
    budget_remaining: int
    governance_result: GovernanceCheckResult
    prerequisite: Prerequisite
    routed_at: datetime = field(default_factory=datetime.now)
    estimated_duration_ms: int = 100
    fallback_tool: Optional[str] = None

    @property
    def is_allowed(self) -> bool:
        """Check if this decision is allowed to proceed."""
        return self.governance_result.decision == GovernanceDecision.ALLOW

    @property
    def needs_verification(self) -> bool:
        """Check if decision needs verification before proceeding."""
        return self.governance_result.decision == GovernanceDecision.PAUSE

    @property
    def is_blocked(self) -> bool:
        """Check if decision is blocked by governance."""
        return self.governance_result.decision == GovernanceDecision.BLOCK

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision for logging."""
        return {
            "decision_id": self.decision_id,
            "tool_name": self.tool_name,
            "action_params": self.action_params,
            "evidence_requirements": self.evidence_requirements,
            "budget_remaining": self.budget_remaining,
            "governance_decision": self.governance_result.decision.value,
            "governance_reason": self.governance_result.reason,
            "prerequisite_id": self.prerequisite.prereq_id,
            "routed_at": self.routed_at.isoformat(),
        }


# =============================================================================
# TOOL SELECTION STRATEGY
# =============================================================================

# Mapping from prerequisite types to candidate tools (in priority order)
PREREQ_TO_TOOLS: Dict[PrerequisiteType, List[str]] = {
    PrerequisiteType.KNOWLEDGE: ["memory_read", "search", "file_read"],
    PrerequisiteType.COMPUTATION: ["code_execute", "summarize"],
    PrerequisiteType.STATE_UPDATE: ["memory_write", "file_write"],
    PrerequisiteType.VERIFICATION: ["memory_read", "search"],
    PrerequisiteType.SYNTHESIS: ["summarize", "memory_write"],
    PrerequisiteType.EXTERNAL_ACTION: ["file_write", "file_delete", "code_execute"],
}


def select_tool_for_prerequisite(
    prereq: Prerequisite,
    context: RoutingContext,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Select the best tool for a prerequisite.

    Returns:
        Tuple of (primary_tool, fallback_tool) or (None, None) if no suitable tool.
    """
    candidates = PREREQ_TO_TOOLS.get(prereq.prereq_type, [])

    primary = None
    fallback = None

    for tool_name in candidates:
        profile = TOOL_REGISTRY.get(tool_name)
        if profile is None:
            continue

        # Check if tool is allowed in current mode
        if not profile.is_allowed_in_mode(context.current_mode):
            continue

        # Check if tool is in allowlist (if specified)
        if context.allowed_tools is not None:
            if tool_name not in context.allowed_tools:
                continue

        # Check capability threshold
        # Use complexity as proxy for capability requirement
        if prereq.estimated_complexity < profile.capability_threshold:
            continue

        # First valid candidate is primary
        if primary is None:
            primary = tool_name
        elif fallback is None:
            fallback = tool_name
            break  # We have both primary and fallback

    return primary, fallback


# =============================================================================
# GOVERNANCE RAIL CHECKS
# =============================================================================

def check_governance_rails(
    tool_name: str,
    prereq: Prerequisite,
    context: RoutingContext,
) -> GovernanceCheckResult:
    """
    Check governance rails before routing to a tool.

    This is a simplified version - the full implementation would
    integrate with tks_features/governance_rails.py.
    """
    profile = TOOL_REGISTRY.get(tool_name)

    if profile is None:
        return GovernanceCheckResult(
            decision=GovernanceDecision.BLOCK,
            reason=f"Unknown tool: {tool_name}",
            mode=context.current_mode,
            required_actions=["Use a registered tool"],
        )

    # Check 1: Mode constraint
    if not profile.is_allowed_in_mode(context.current_mode):
        return GovernanceCheckResult(
            decision=GovernanceDecision.BLOCK,
            reason=f"Tool '{tool_name}' not allowed in {context.current_mode.value} mode",
            mode=context.current_mode,
            required_actions=["Change mode or use different tool"],
        )

    # Check 2: Budget constraint
    if not context.can_make_tool_call():
        return GovernanceCheckResult(
            decision=GovernanceDecision.BLOCK,
            reason=f"Tool call budget exhausted (mode: {context.current_mode.value})",
            mode=context.current_mode,
            required_actions=["Stop tool calls", "Summarize progress"],
        )

    # Check 3: Depth constraint
    if not context.depth_allowed():
        return GovernanceCheckResult(
            decision=GovernanceDecision.BLOCK,
            reason=f"RPM depth limit exceeded (current: {context.current_depth})",
            mode=context.current_mode,
            required_actions=["Reduce depth", "ASK for missing info"],
        )

    # Check 4: Breadth constraint
    if not context.breadth_allowed():
        return GovernanceCheckResult(
            decision=GovernanceDecision.BLOCK,
            reason=f"RPM breadth limit exceeded (current: {context.current_breadth})",
            mode=context.current_mode,
            required_actions=["Prioritize prerequisites"],
        )

    # Check 5: Evidence requirement
    if profile.evidence_required and not context.has_evidence:
        return GovernanceCheckResult(
            decision=GovernanceDecision.PAUSE,
            reason=f"Tool '{tool_name}' requires evidence binding",
            mode=context.current_mode,
            required_actions=["Bind evidence"],
            missing_evidence=["action_justification"],
        )

    # Check 6: Irreversibility
    if profile.reversibility == ToolReversibility.IRREVERSIBLE:
        if context.current_mode != OperationalMode.NORMAL:
            return GovernanceCheckResult(
                decision=GovernanceDecision.BLOCK,
                reason=f"Irreversible tool '{tool_name}' blocked outside Normal mode",
                mode=context.current_mode,
                required_actions=["Switch to Normal mode", "Obtain clearance"],
                clearance_required=True,
            )
        # Even in Normal, require clearance for irreversible
        if not context.clearance_tokens:
            return GovernanceCheckResult(
                decision=GovernanceDecision.PAUSE,
                reason=f"Irreversible tool '{tool_name}' requires clearance",
                mode=context.current_mode,
                required_actions=["Obtain clearance token"],
                clearance_required=True,
            )

    # Check 7: Partial reversibility in High-Stakes
    if profile.reversibility == ToolReversibility.PARTIAL:
        if context.current_mode == OperationalMode.HIGH_STAKES:
            return GovernanceCheckResult(
                decision=GovernanceDecision.PAUSE,
                reason=f"Partial-reversible tool '{tool_name}' requires verification in High-Stakes",
                mode=context.current_mode,
                required_actions=["Verify action", "Consider alternatives"],
            )

    # Check 8: Sandbox requirement
    if profile.sandbox_required and not context.sandbox_enabled:
        return GovernanceCheckResult(
            decision=GovernanceDecision.BLOCK,
            reason=f"Tool '{tool_name}' requires sandbox but sandbox is disabled",
            mode=context.current_mode,
            required_actions=["Enable sandbox"],
        )

    # Check 9: Network requirement
    if profile.requires_network and not context.network_enabled:
        return GovernanceCheckResult(
            decision=GovernanceDecision.BLOCK,
            reason=f"Tool '{tool_name}' requires network but network is disabled",
            mode=context.current_mode,
            required_actions=["Enable network", "Use offline alternative"],
        )

    # Check 10: Filesystem path allowlist
    if profile.requires_filesystem and context.allowed_paths is not None:
        target_path = prereq.constraints.get("target_path", "")
        if target_path:
            path_allowed = any(
                target_path.startswith(allowed)
                for allowed in context.allowed_paths
            )
            if not path_allowed:
                return GovernanceCheckResult(
                    decision=GovernanceDecision.BLOCK,
                    reason=f"Path '{target_path}' not in allowed paths",
                    mode=context.current_mode,
                    required_actions=["Use allowed path"],
                )

    # All checks passed
    return GovernanceCheckResult(
        decision=GovernanceDecision.ALLOW,
        reason="All governance checks passed",
        mode=context.current_mode,
    )


# =============================================================================
# ROUTER CLASS
# =============================================================================

class Router:
    """
    TKS Router - Routes RPM prerequisites to appropriate tools.

    The router is responsible for:
    1. Selecting the best tool for each prerequisite
    2. Checking governance rails before routing
    3. Enforcing budgets (tool calls, depth, breadth)
    4. Generating router decisions with evidence requirements

    Usage:
        router = Router()
        decision = router.route(prerequisite, context)
        if decision.is_allowed:
            # Execute tool
            ...
        elif decision.needs_verification:
            # Get user verification
            ...
        else:
            # Blocked - handle gracefully
            ...
    """

    def __init__(
        self,
        custom_tool_registry: Optional[Dict[str, ToolProfile]] = None,
        custom_prereq_mapping: Optional[Dict[PrerequisiteType, List[str]]] = None,
    ) -> None:
        """
        Initialize the router.

        Args:
            custom_tool_registry: Optional custom tool registry to use
            custom_prereq_mapping: Optional custom prerequisite-to-tool mapping
        """
        self.tool_registry = custom_tool_registry or TOOL_REGISTRY
        self.prereq_mapping = custom_prereq_mapping or PREREQ_TO_TOOLS
        self._decision_counter = 0

    def _generate_decision_id(self) -> str:
        """Generate a unique decision ID."""
        self._decision_counter += 1
        timestamp = int(time.time() * 1000)
        return f"rd_{timestamp}_{self._decision_counter:06d}"

    def _build_action_params(
        self,
        tool_name: str,
        prereq: Prerequisite,
        context: RoutingContext,
    ) -> Dict[str, Any]:
        """Build action parameters for a tool call."""
        params = {
            "prereq_id": prereq.prereq_id,
            "prereq_type": prereq.prereq_type.value,
            "description": prereq.description,
            **prereq.required_inputs,
        }

        # Add constraints as params where relevant
        for key, value in prereq.constraints.items():
            if key not in params:
                params[key] = value

        # Add context info
        params["episode_id"] = context.episode_id
        params["mode"] = context.current_mode.value

        return params

    def _determine_evidence_requirements(
        self,
        tool_name: str,
        prereq: Prerequisite,
    ) -> List[str]:
        """Determine what evidence is required for the tool call."""
        requirements = []

        profile = self.tool_registry.get(tool_name)
        if profile is None:
            return requirements

        if profile.evidence_required:
            requirements.append("action_justification")

        if prereq.evidence_required:
            requirements.append("prerequisite_evidence")

        if profile.reversibility == ToolReversibility.IRREVERSIBLE:
            requirements.append("irreversibility_acknowledgment")

        if profile.requires_code_execution:
            requirements.append("code_safety_verification")

        return requirements

    def route(
        self,
        prerequisite: Prerequisite,
        context: RoutingContext,
        governance_override: Optional[GovernanceCheckResult] = None,
    ) -> RouterDecision:
        """
        Route a prerequisite to an appropriate tool.

        This is the main entry point for routing decisions.

        Args:
            prerequisite: The prerequisite to route
            context: Current routing context
            governance_override: Optional override for governance (for testing)

        Returns:
            RouterDecision with tool selection and governance result
        """
        decision_id = self._generate_decision_id()

        # Step 1: Select tool
        primary_tool, fallback_tool = select_tool_for_prerequisite(
            prerequisite, context
        )

        if primary_tool is None:
            # No suitable tool found
            return RouterDecision(
                decision_id=decision_id,
                tool_name="none",
                action_params={},
                evidence_requirements=[],
                budget_remaining=context.get_budget_remaining(),
                governance_result=GovernanceCheckResult(
                    decision=GovernanceDecision.BLOCK,
                    reason=f"No suitable tool for prerequisite type: {prerequisite.prereq_type.value}",
                    mode=context.current_mode,
                    required_actions=["Check tool availability", "Modify prerequisite"],
                ),
                prerequisite=prerequisite,
                fallback_tool=fallback_tool,
            )

        # Step 2: Check governance rails
        if governance_override is not None:
            governance_result = governance_override
        else:
            governance_result = check_governance_rails(
                primary_tool, prerequisite, context
            )

        # Step 3: If blocked, try fallback
        if governance_result.decision == GovernanceDecision.BLOCK and fallback_tool:
            fallback_governance = check_governance_rails(
                fallback_tool, prerequisite, context
            )
            if fallback_governance.decision != GovernanceDecision.BLOCK:
                # Use fallback
                primary_tool = fallback_tool
                governance_result = fallback_governance
                fallback_tool = None

        # Step 4: Build action params
        action_params = self._build_action_params(
            primary_tool, prerequisite, context
        )

        # Step 5: Determine evidence requirements
        evidence_requirements = self._determine_evidence_requirements(
            primary_tool, prerequisite
        )

        # Step 6: Calculate remaining budget (accounting for this call if allowed)
        budget_remaining = context.get_budget_remaining()
        if governance_result.decision == GovernanceDecision.ALLOW:
            budget_remaining -= 1

        # Step 7: Build and return decision
        return RouterDecision(
            decision_id=decision_id,
            tool_name=primary_tool,
            action_params=action_params,
            evidence_requirements=evidence_requirements,
            budget_remaining=budget_remaining,
            governance_result=governance_result,
            prerequisite=prerequisite,
            fallback_tool=fallback_tool,
        )

    def route_batch(
        self,
        prerequisites: List[Prerequisite],
        context: RoutingContext,
    ) -> List[RouterDecision]:
        """
        Route multiple prerequisites in batch.

        Prerequisites are routed in priority order, and context
        is updated after each routing decision.

        Args:
            prerequisites: List of prerequisites to route
            context: Initial routing context

        Returns:
            List of RouterDecisions in priority order
        """
        # Sort by priority (1 = highest)
        sorted_prereqs = sorted(prerequisites, key=lambda p: p.priority)

        decisions = []
        current_context = context

        for prereq in sorted_prereqs:
            decision = self.route(prereq, current_context)
            decisions.append(decision)

            # Update context for next routing
            if decision.is_allowed:
                current_context.tool_calls_made += 1
                current_context.tool_history.append(decision.tool_name)

        return decisions

    def can_route(self, context: RoutingContext) -> bool:
        """Check if any routing is possible given current context."""
        return (
            context.can_make_tool_call() and
            context.depth_allowed() and
            context.breadth_allowed()
        )

    def get_available_tools(self, context: RoutingContext) -> List[str]:
        """Get list of tools available in current context."""
        available = []
        for name, profile in self.tool_registry.items():
            if profile.is_allowed_in_mode(context.current_mode):
                if context.allowed_tools is None or name in context.allowed_tools:
                    available.append(name)
        return available


# =============================================================================
# ROUTER EVENTS FOR TRACING
# =============================================================================

@dataclass
class RouterEvent:
    """Event logged for each routing decision."""
    event_id: str
    timestamp: datetime
    episode_id: str
    prerequisite_id: str
    decision_id: str
    tool_selected: str
    governance_decision: str
    budget_remaining: int
    depth: int
    breadth: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "episode_id": self.episode_id,
            "prerequisite_id": self.prerequisite_id,
            "decision_id": self.decision_id,
            "tool_selected": self.tool_selected,
            "governance_decision": self.governance_decision,
            "budget_remaining": self.budget_remaining,
            "depth": self.depth,
            "breadth": self.breadth,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "PrerequisiteType",
    "Prerequisite",
    "RoutingContext",
    "GovernanceDecision",
    "GovernanceCheckResult",
    "RouterDecision",
    # Main class
    "Router",
    # Functions
    "select_tool_for_prerequisite",
    "check_governance_rails",
    # Mappings
    "PREREQ_TO_TOOLS",
    # Events
    "RouterEvent",
]

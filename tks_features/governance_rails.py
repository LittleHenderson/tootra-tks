"""
Governance Rails Interface - Non-Bypassable Safety Gates

This module defines the INTERFACE for Governance Rails that cannot be bypassed.
Governance is RULE-BASED with NO learnable parameters.

IMPORTANT: This is an INTERFACE file. Implementation is handled by agent-be.

CRITICAL DISTINCTION:
    - DPS (dps_gating.py): LEARNED module with ~3000 training examples
    - Governance (this file): DETERMINISTIC rules with NO training data

Governance sits ABOVE the neural pipeline and enforces hard constraints.
If rails disagree with downstream modules, rails WIN.

The 5 Non-Bypassable Gates:
    1. HighStakesGate: HS = (U x K) x A^2, thresholds at 0.45/0.70
    2. IrreversibilityGate: Check action.reversibility flag
    3. PermissionGate: Check against capability allowlist
    4. AntiRecursionGate: depth <= max_depth, breadth <= max_breadth
    5. SelfModificationGate: Block in Critical mode

Gate Outcomes:
    ALLOW: Proceed with action
    PAUSE: Require verification/confirmation before proceeding
    BLOCK: Refuse to execute, require clearance

Canonical TKS Expressions:
    High-Stakes: (1*:2*:5*):(ASK:SLOW:VERIFY)+GATE(PAUSE)
    Critical: (1*:2*:5*):(ASK:SLOW:VERIFY)-EXECUTE

Author: ARCH agent
Date: 2026-01-05
Schema Version: 1.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Protocol, runtime_checkable
from datetime import datetime
import hashlib


# =============================================================================
# CONSTANTS
# =============================================================================

# High-Stakes thresholds (from Module 7 spec)
DEFAULT_HS_THRESHOLD = 0.45        # HS >= 0.45 -> High-Stakes (PAUSE)
DEFAULT_CRITICAL_THRESHOLD = 0.70  # HS >= 0.70 -> Critical (BLOCK)

# Alignment minimum (F1 dominance)
DEFAULT_ALIGNMENT_MIN = 0.80

# Anti-recursion caps (defaults from spec)
DEFAULT_RPM_DEPTH_CAP_NORMAL = 3
DEFAULT_RPM_DEPTH_CAP_HIGH_STAKES = 2
DEFAULT_RPM_BREADTH_CAP = 5
DEFAULT_TOOL_CALL_CAP_NORMAL = 12
DEFAULT_TOOL_CALL_CAP_HIGH_STAKES = 8
DEFAULT_TOOL_CALL_CAP_CRITICAL = 0


class GateDecision(Enum):
    """Outcome of a governance gate check."""
    ALLOW = "ALLOW"   # Proceed with action
    PAUSE = "PAUSE"   # Require verification/confirmation
    BLOCK = "BLOCK"   # Refuse to execute


class OperationalMode(Enum):
    """System operational mode."""
    NORMAL = "Normal"
    HIGH_STAKES = "High-Stakes"
    CRITICAL = "Critical"


class ActionReversibility(Enum):
    """How reversible an action is."""
    REVERSIBLE = "reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GovernanceConfig:
    """
    Configuration for Governance Rails.

    All thresholds and caps are tunable policy parameters.
    NO learnable parameters - this is pure configuration.

    Attributes:
        hs_threshold: High-Stakes threshold (PAUSE if HS >= this)
        critical_threshold: Critical threshold (BLOCK if HS >= this)
        alignment_min: Minimum alignment score (F1 dominance)
        rpm_depth_cap_normal: Max RPM depth in Normal mode
        rpm_depth_cap_high_stakes: Max RPM depth in High-Stakes mode
        rpm_breadth_cap: Max children per RPM node
        tool_call_cap_normal: Max tool calls per episode (Normal)
        tool_call_cap_high_stakes: Max tool calls per episode (High-Stakes)
        tool_call_cap_critical: Max tool calls per episode (Critical)
        require_evidence_high_stakes: Require evidence binding in High-Stakes
        require_clearance_irreversible: Require clearance token for irreversible actions
    """
    # High-Stakes scoring thresholds
    hs_threshold: float = DEFAULT_HS_THRESHOLD
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD

    # Alignment (F1 dominance)
    alignment_min: float = DEFAULT_ALIGNMENT_MIN

    # RPM depth/breadth caps
    rpm_depth_cap_normal: int = DEFAULT_RPM_DEPTH_CAP_NORMAL
    rpm_depth_cap_high_stakes: int = DEFAULT_RPM_DEPTH_CAP_HIGH_STAKES
    rpm_breadth_cap: int = DEFAULT_RPM_BREADTH_CAP

    # Tool call caps
    tool_call_cap_normal: int = DEFAULT_TOOL_CALL_CAP_NORMAL
    tool_call_cap_high_stakes: int = DEFAULT_TOOL_CALL_CAP_HIGH_STAKES
    tool_call_cap_critical: int = DEFAULT_TOOL_CALL_CAP_CRITICAL

    # Evidence requirements
    require_evidence_high_stakes: bool = True
    require_clearance_irreversible: bool = True

    # Self-modification constraints
    allow_self_mod_normal: bool = True
    allow_self_mod_high_stakes: bool = True  # Requires replay pass
    allow_self_mod_critical: bool = False    # Always blocked

    # Uncertainty rule: stop deepening if uncertainty doesn't drop
    max_expansion_without_uncertainty_drop: int = 2


# =============================================================================
# ACTION PROFILE
# =============================================================================

@dataclass
class ActionProfile:
    """
    Profile of an action for governance evaluation.

    Every action submitted to the system must have a profile that
    governance rails can evaluate.
    """
    action_type: str                          # e.g., "tool_call", "memory_write", "policy_update"
    tool_name: Optional[str] = None           # If tool_call, which tool
    reversibility: ActionReversibility = ActionReversibility.REVERSIBLE
    requires_network: bool = False
    requires_filesystem: bool = False
    requires_code_execution: bool = False
    target_scope: Optional[str] = None        # e.g., file path, account, system
    estimated_cost: float = 0.0               # Resource cost estimate
    is_self_modification: bool = False        # Modifies router/verifier/reward/etc.
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class ClearanceToken:
    """
    Formal permission to proceed with a protected action.

    Clearance tokens are required for irreversible actions and
    certain high-stakes operations.
    """
    token_id: str
    scope: str                                # "file_delete", "security_change", etc.
    episode_id: str
    approved_by: str                          # "user", "admin", "verifier"
    evidence_ids: List[str]
    expires_at: datetime
    constraints: Dict = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if token is still valid (not expired)."""
        return datetime.now() < self.expires_at

    def covers_scope(self, requested_scope: str) -> bool:
        """Check if token covers the requested scope."""
        return self.scope == requested_scope or self.scope == "*"


# =============================================================================
# GATE RESULT
# =============================================================================

@dataclass
class GateResult:
    """
    Result of a governance gate check.

    Contains the decision, reason, and any required next steps.
    """
    gate_name: str
    decision: GateDecision
    reason: str
    mode: OperationalMode
    required_actions: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    clearance_required: bool = False
    hs_score: Optional[float] = None

    def to_dict(self) -> Dict:
        """Serialize for logging."""
        return {
            "gate_name": self.gate_name,
            "decision": self.decision.value,
            "reason": self.reason,
            "mode": self.mode.value,
            "required_actions": self.required_actions,
            "missing_evidence": self.missing_evidence,
            "clearance_required": self.clearance_required,
            "hs_score": self.hs_score,
        }


# =============================================================================
# INDIVIDUAL GATES (Pure Python logic, NO nn.Module)
# =============================================================================

class HighStakesGate:
    """
    Gate 1: High-Stakes enforcement.

    Computes: HS = (U x K) x A^2
    Where:
        U = uncertainty [0, 1]
        K = stakes [0, 1]
        A = alignment score [0, 1]

    Thresholds:
        HS >= 0.70 -> Critical (BLOCK)
        HS >= 0.45 -> High-Stakes (PAUSE)
        HS < 0.45 -> Normal (ALLOW)

    NO learnable parameters.
    """

    def __init__(self, config: GovernanceConfig) -> None:
        self.config = config

    def compute_high_stakes(
        self,
        uncertainty: float,
        stakes: float,
        alignment: float,
    ) -> float:
        """
        Compute High-Stakes score.

        HS = (U x K) x A^2

        Pure math, no learning.
        """
        # Clamp inputs to [0, 1]
        u = max(0.0, min(1.0, uncertainty))
        k = max(0.0, min(1.0, stakes))
        a = max(0.0, min(1.0, alignment))

        return (u * k) * (a ** 2)

    def determine_mode(self, hs_score: float) -> OperationalMode:
        """Determine operational mode from HS score."""
        if hs_score >= self.config.critical_threshold:
            return OperationalMode.CRITICAL
        elif hs_score >= self.config.hs_threshold:
            return OperationalMode.HIGH_STAKES
        else:
            return OperationalMode.NORMAL

    def check(
        self,
        uncertainty: float,
        stakes: float,
        alignment: float,
        has_evidence: bool = True,
    ) -> GateResult:
        """
        Check High-Stakes gate.

        Args:
            uncertainty: Uncertainty score [0, 1]
            stakes: Stakes score [0, 1]
            alignment: Alignment score [0, 1]
            has_evidence: Whether required evidence is bound

        Returns:
            GateResult with decision and mode
        """
        hs = self.compute_high_stakes(uncertainty, stakes, alignment)
        mode = self.determine_mode(hs)

        # Check alignment minimum (F1 dominance)
        if alignment < self.config.alignment_min:
            return GateResult(
                gate_name="HighStakesGate",
                decision=GateDecision.PAUSE,
                reason=f"Alignment {alignment:.2f} below minimum {self.config.alignment_min}",
                mode=mode,
                required_actions=["Switch to 1*:2*:* stack", "Clarify alignment"],
                hs_score=hs,
            )

        # Check evidence requirement in High-Stakes/Critical
        if mode != OperationalMode.NORMAL and not has_evidence:
            if self.config.require_evidence_high_stakes:
                return GateResult(
                    gate_name="HighStakesGate",
                    decision=GateDecision.BLOCK,
                    reason="Missing required evidence in High-Stakes/Critical mode",
                    mode=mode,
                    required_actions=["Bind evidence", "Provide sources"],
                    missing_evidence=["required_evidence"],
                    hs_score=hs,
                )

        # Determine decision based on mode
        if mode == OperationalMode.CRITICAL:
            return GateResult(
                gate_name="HighStakesGate",
                decision=GateDecision.BLOCK,
                reason=f"Critical mode: HS={hs:.2f} >= {self.config.critical_threshold}",
                mode=mode,
                required_actions=["ASK", "SLOW", "VERIFY", "Obtain clearance"],
                hs_score=hs,
            )
        elif mode == OperationalMode.HIGH_STAKES:
            return GateResult(
                gate_name="HighStakesGate",
                decision=GateDecision.PAUSE,
                reason=f"High-Stakes mode: HS={hs:.2f} >= {self.config.hs_threshold}",
                mode=mode,
                required_actions=["ASK", "SLOW", "VERIFY"],
                hs_score=hs,
            )
        else:
            return GateResult(
                gate_name="HighStakesGate",
                decision=GateDecision.ALLOW,
                reason=f"Normal mode: HS={hs:.2f} < {self.config.hs_threshold}",
                mode=mode,
                hs_score=hs,
            )


class IrreversibilityGate:
    """
    Gate 2: Irreversibility enforcement.

    Checks action.reversibility flag and requires clearance for
    irreversible actions.

    Examples of irreversible actions:
        - Deleting data
        - Transferring funds
        - Changing security settings
        - Publishing sensitive outputs

    NO learnable parameters.
    """

    def __init__(self, config: GovernanceConfig) -> None:
        self.config = config

    def check(
        self,
        action: ActionProfile,
        clearance_token: Optional[ClearanceToken] = None,
        current_mode: OperationalMode = OperationalMode.NORMAL,
    ) -> GateResult:
        """
        Check Irreversibility gate.

        Args:
            action: Action profile to evaluate
            clearance_token: Optional clearance token
            current_mode: Current operational mode

        Returns:
            GateResult with decision
        """
        if action.reversibility == ActionReversibility.REVERSIBLE:
            return GateResult(
                gate_name="IrreversibilityGate",
                decision=GateDecision.ALLOW,
                reason="Action is reversible",
                mode=current_mode,
            )

        # Irreversible or partially reversible actions
        if action.reversibility == ActionReversibility.IRREVERSIBLE:
            # Check for clearance token
            if self.config.require_clearance_irreversible:
                if clearance_token is None:
                    return GateResult(
                        gate_name="IrreversibilityGate",
                        decision=GateDecision.BLOCK,
                        reason="Irreversible action requires clearance token",
                        mode=current_mode,
                        required_actions=["Obtain clearance token"],
                        clearance_required=True,
                    )
                if not clearance_token.is_valid():
                    return GateResult(
                        gate_name="IrreversibilityGate",
                        decision=GateDecision.BLOCK,
                        reason="Clearance token expired",
                        mode=current_mode,
                        required_actions=["Obtain new clearance token"],
                        clearance_required=True,
                    )
                if not clearance_token.covers_scope(action.action_type):
                    return GateResult(
                        gate_name="IrreversibilityGate",
                        decision=GateDecision.BLOCK,
                        reason=f"Clearance token does not cover scope: {action.action_type}",
                        mode=current_mode,
                        required_actions=["Obtain matching clearance token"],
                        clearance_required=True,
                    )

            # Check evidence binding
            if not action.evidence_ids:
                return GateResult(
                    gate_name="IrreversibilityGate",
                    decision=GateDecision.BLOCK,
                    reason="Irreversible action requires evidence binding",
                    mode=current_mode,
                    required_actions=["Bind evidence"],
                    missing_evidence=["action_evidence"],
                )

            return GateResult(
                gate_name="IrreversibilityGate",
                decision=GateDecision.ALLOW,
                reason="Irreversible action cleared",
                mode=current_mode,
            )

        # Partially reversible - require PAUSE in High-Stakes/Critical
        if current_mode != OperationalMode.NORMAL:
            return GateResult(
                gate_name="IrreversibilityGate",
                decision=GateDecision.PAUSE,
                reason="Partially reversible action in High-Stakes/Critical mode",
                mode=current_mode,
                required_actions=["Verify", "Consider alternatives"],
            )

        return GateResult(
            gate_name="IrreversibilityGate",
            decision=GateDecision.ALLOW,
            reason="Partially reversible action in Normal mode",
            mode=current_mode,
        )


class PermissionGate:
    """
    Gate 3: Permission/capability enforcement.

    Checks action against capability allowlist and sandbox constraints:
        - File read/write scope
        - Code execution sandbox
        - Network allowed/denied
        - Allowlist sources

    NO learnable parameters.
    """

    def __init__(
        self,
        config: GovernanceConfig,
        allowed_tools: Optional[Set[str]] = None,
        allowed_paths: Optional[Set[str]] = None,
        network_enabled: bool = False,
    ) -> None:
        self.config = config
        self.allowed_tools = allowed_tools or set()
        self.allowed_paths = allowed_paths or set()
        self.network_enabled = network_enabled

    def check(
        self,
        action: ActionProfile,
        current_mode: OperationalMode = OperationalMode.NORMAL,
    ) -> GateResult:
        """
        Check Permission gate.

        Args:
            action: Action profile to evaluate
            current_mode: Current operational mode

        Returns:
            GateResult with decision
        """
        # Check tool allowlist
        if action.tool_name and self.allowed_tools:
            if action.tool_name not in self.allowed_tools:
                return GateResult(
                    gate_name="PermissionGate",
                    decision=GateDecision.BLOCK,
                    reason=f"Tool '{action.tool_name}' not in allowlist",
                    mode=current_mode,
                    required_actions=["Use allowed tool", "Request permission"],
                )

        # Check network permission
        if action.requires_network and not self.network_enabled:
            return GateResult(
                gate_name="PermissionGate",
                decision=GateDecision.BLOCK,
                reason="Network access not enabled",
                mode=current_mode,
                required_actions=["Enable network", "Use offline alternative"],
            )

        # Check filesystem permission
        if action.requires_filesystem and action.target_scope:
            if self.allowed_paths:
                path_allowed = any(
                    action.target_scope.startswith(allowed)
                    for allowed in self.allowed_paths
                )
                if not path_allowed:
                    return GateResult(
                        gate_name="PermissionGate",
                        decision=GateDecision.BLOCK,
                        reason=f"Path '{action.target_scope}' not in allowed paths",
                        mode=current_mode,
                        required_actions=["Use allowed path", "Request permission"],
                    )

        # Check code execution
        if action.requires_code_execution:
            # In Critical mode, no code execution
            if current_mode == OperationalMode.CRITICAL:
                return GateResult(
                    gate_name="PermissionGate",
                    decision=GateDecision.BLOCK,
                    reason="Code execution blocked in Critical mode",
                    mode=current_mode,
                    required_actions=["Exit Critical mode", "Use non-code alternative"],
                )
            # In High-Stakes, require sandbox
            if current_mode == OperationalMode.HIGH_STAKES:
                return GateResult(
                    gate_name="PermissionGate",
                    decision=GateDecision.PAUSE,
                    reason="Code execution requires sandbox verification in High-Stakes",
                    mode=current_mode,
                    required_actions=["Verify sandbox", "Confirm safe execution"],
                )

        return GateResult(
            gate_name="PermissionGate",
            decision=GateDecision.ALLOW,
            reason="Permission checks passed",
            mode=current_mode,
        )


class AntiRecursionGate:
    """
    Gate 4: Anti-recursion/anti-runaway enforcement.

    Prevents infinite trees and obsessive loops with hard caps:
        - RPM depth cap
        - RPM breadth cap per node
        - Tool call cap per episode
        - Uncertainty-drop rule

    NO learnable parameters.
    """

    def __init__(self, config: GovernanceConfig) -> None:
        self.config = config

    def get_depth_cap(self, mode: OperationalMode) -> int:
        """Get RPM depth cap for current mode."""
        if mode == OperationalMode.HIGH_STAKES:
            return self.config.rpm_depth_cap_high_stakes
        return self.config.rpm_depth_cap_normal

    def get_tool_cap(self, mode: OperationalMode) -> int:
        """Get tool call cap for current mode."""
        if mode == OperationalMode.CRITICAL:
            return self.config.tool_call_cap_critical
        elif mode == OperationalMode.HIGH_STAKES:
            return self.config.tool_call_cap_high_stakes
        return self.config.tool_call_cap_normal

    def check(
        self,
        current_depth: int,
        current_breadth: int,
        tool_calls_this_episode: int,
        current_mode: OperationalMode,
        uncertainty_dropped: bool = True,
        expansions_since_drop: int = 0,
    ) -> GateResult:
        """
        Check Anti-Recursion gate.

        Args:
            current_depth: Current RPM tree depth
            current_breadth: Current node's children count
            tool_calls_this_episode: Number of tool calls so far
            current_mode: Current operational mode
            uncertainty_dropped: Whether uncertainty has dropped recently
            expansions_since_drop: Expansions since last uncertainty drop

        Returns:
            GateResult with decision
        """
        depth_cap = self.get_depth_cap(current_mode)
        tool_cap = self.get_tool_cap(current_mode)

        # Check depth cap
        if current_depth > depth_cap:
            return GateResult(
                gate_name="AntiRecursionGate",
                decision=GateDecision.BLOCK,
                reason=f"RPM depth {current_depth} exceeds cap {depth_cap}",
                mode=current_mode,
                required_actions=["Reduce depth", "ASK for missing info"],
            )

        # Check breadth cap
        if current_breadth > self.config.rpm_breadth_cap:
            return GateResult(
                gate_name="AntiRecursionGate",
                decision=GateDecision.BLOCK,
                reason=f"RPM breadth {current_breadth} exceeds cap {self.config.rpm_breadth_cap}",
                mode=current_mode,
                required_actions=["Reduce breadth", "Prioritize prerequisites"],
            )

        # Check tool call cap
        if tool_calls_this_episode >= tool_cap:
            return GateResult(
                gate_name="AntiRecursionGate",
                decision=GateDecision.BLOCK,
                reason=f"Tool calls {tool_calls_this_episode} at cap {tool_cap}",
                mode=current_mode,
                required_actions=["Stop tool calls", "Summarize progress"],
            )

        # Check uncertainty-drop rule
        if not uncertainty_dropped:
            if expansions_since_drop >= self.config.max_expansion_without_uncertainty_drop:
                return GateResult(
                    gate_name="AntiRecursionGate",
                    decision=GateDecision.PAUSE,
                    reason=f"Uncertainty not dropping after {expansions_since_drop} expansions",
                    mode=current_mode,
                    required_actions=["PAUSE", "ASK for missing constraints/evidence"],
                )

        return GateResult(
            gate_name="AntiRecursionGate",
            decision=GateDecision.ALLOW,
            reason="Recursion checks passed",
            mode=current_mode,
        )


class SelfModificationGate:
    """
    Gate 5: Self-modification enforcement.

    Controls changes to:
        - Router rules
        - Verifier thresholds
        - Reward weights
        - Promotion logic
        - Operator semantics
        - Canonical definitions

    Rules:
        - No self-modification in Critical mode
        - High-Stakes self-modification requires replay harness pass
        - Must be versioned and reversible

    NO learnable parameters.
    """

    def __init__(self, config: GovernanceConfig) -> None:
        self.config = config

    def check(
        self,
        action: ActionProfile,
        current_mode: OperationalMode,
        replay_passed: bool = False,
        has_rollback_plan: bool = False,
    ) -> GateResult:
        """
        Check Self-Modification gate.

        Args:
            action: Action profile (must have is_self_modification set)
            current_mode: Current operational mode
            replay_passed: Whether replay harness passed for this change
            has_rollback_plan: Whether a rollback plan exists

        Returns:
            GateResult with decision
        """
        if not action.is_self_modification:
            return GateResult(
                gate_name="SelfModificationGate",
                decision=GateDecision.ALLOW,
                reason="Action is not self-modification",
                mode=current_mode,
            )

        # Critical mode: always block
        if current_mode == OperationalMode.CRITICAL:
            if not self.config.allow_self_mod_critical:
                return GateResult(
                    gate_name="SelfModificationGate",
                    decision=GateDecision.BLOCK,
                    reason="Self-modification blocked in Critical mode",
                    mode=current_mode,
                    required_actions=["Exit Critical mode", "Defer modification"],
                )

        # High-Stakes mode: require replay pass
        if current_mode == OperationalMode.HIGH_STAKES:
            if not self.config.allow_self_mod_high_stakes:
                return GateResult(
                    gate_name="SelfModificationGate",
                    decision=GateDecision.BLOCK,
                    reason="Self-modification disabled in High-Stakes mode",
                    mode=current_mode,
                )

            if not replay_passed:
                return GateResult(
                    gate_name="SelfModificationGate",
                    decision=GateDecision.BLOCK,
                    reason="Self-modification requires replay harness pass in High-Stakes",
                    mode=current_mode,
                    required_actions=["Run replay harness", "Verify no regressions"],
                )

            if not has_rollback_plan:
                return GateResult(
                    gate_name="SelfModificationGate",
                    decision=GateDecision.PAUSE,
                    reason="Self-modification requires rollback plan",
                    mode=current_mode,
                    required_actions=["Create rollback plan", "Document do-not-apply guard"],
                )

        # Normal mode: allow with versioning requirement
        if current_mode == OperationalMode.NORMAL:
            if not self.config.allow_self_mod_normal:
                return GateResult(
                    gate_name="SelfModificationGate",
                    decision=GateDecision.BLOCK,
                    reason="Self-modification disabled in Normal mode",
                    mode=current_mode,
                )

        return GateResult(
            gate_name="SelfModificationGate",
            decision=GateDecision.ALLOW,
            reason="Self-modification allowed",
            mode=current_mode,
            required_actions=["Version the change", "Log audit entry"],
        )


# =============================================================================
# GOVERNANCE RAILS (Combined Gate System)
# =============================================================================

class GovernanceRails:
    """
    Rule-based governance system - NO learnable parameters.

    Combines all 5 non-bypassable gates into a unified interface.
    Governance runs BEFORE actions and AFTER verification.
    If rails disagree with downstream modules, rails WIN.

    Usage:
        config = GovernanceConfig()
        rails = GovernanceRails(config)

        # Check a proposed action
        result = rails.check_action(action, context)
        if result.decision == GateDecision.BLOCK:
            # Cannot proceed
            ...
        elif result.decision == GateDecision.PAUSE:
            # Need verification/confirmation
            ...
        else:
            # Proceed with action
            ...

    Safe Failure Modes:
        When rails block, allowed outputs are:
        - ASK questions
        - Show VERIFY checklist
        - Request evidence or confirmation
        - Propose dry-run/simulation plan
        - Provide reversible alternative

        NOT allowed:
        - Guess
        - Execute anyway
        - Work around
    """

    def __init__(
        self,
        config: GovernanceConfig,
        allowed_tools: Optional[Set[str]] = None,
        allowed_paths: Optional[Set[str]] = None,
        network_enabled: bool = False,
    ) -> None:
        """
        Initialize Governance Rails.

        Args:
            config: Governance configuration
            allowed_tools: Set of allowed tool names
            allowed_paths: Set of allowed filesystem paths
            network_enabled: Whether network access is allowed
        """
        self.config = config

        # Initialize all gates
        self.high_stakes_gate = HighStakesGate(config)
        self.irreversibility_gate = IrreversibilityGate(config)
        self.permission_gate = PermissionGate(
            config,
            allowed_tools=allowed_tools,
            allowed_paths=allowed_paths,
            network_enabled=network_enabled,
        )
        self.anti_recursion_gate = AntiRecursionGate(config)
        self.self_modification_gate = SelfModificationGate(config)

    def check_action(
        self,
        action: ActionProfile,
        uncertainty: float = 0.0,
        stakes: float = 0.0,
        alignment: float = 1.0,
        has_evidence: bool = True,
        clearance_token: Optional[ClearanceToken] = None,
        current_depth: int = 0,
        current_breadth: int = 0,
        tool_calls_this_episode: int = 0,
        uncertainty_dropped: bool = True,
        expansions_since_drop: int = 0,
        replay_passed: bool = False,
        has_rollback_plan: bool = False,
    ) -> GateResult:
        """
        Check all governance gates for a proposed action.

        Gates are checked in order of severity:
        1. HighStakesGate (determines mode)
        2. IrreversibilityGate
        3. PermissionGate
        4. AntiRecursionGate
        5. SelfModificationGate

        First BLOCK stops processing. PAUSEs accumulate.
        If all pass, returns ALLOW.

        Args:
            action: Action profile to evaluate
            uncertainty: Uncertainty score [0, 1]
            stakes: Stakes score [0, 1]
            alignment: Alignment score [0, 1]
            has_evidence: Whether required evidence is bound
            clearance_token: Optional clearance token
            current_depth: Current RPM tree depth
            current_breadth: Current node's children count
            tool_calls_this_episode: Number of tool calls so far
            uncertainty_dropped: Whether uncertainty has dropped recently
            expansions_since_drop: Expansions since last uncertainty drop
            replay_passed: Whether replay harness passed
            has_rollback_plan: Whether rollback plan exists

        Returns:
            GateResult with final decision (most restrictive)
        """
        results: List[GateResult] = []

        # Gate 1: High-Stakes (determines mode for other gates)
        hs_result = self.high_stakes_gate.check(
            uncertainty=uncertainty,
            stakes=stakes,
            alignment=alignment,
            has_evidence=has_evidence,
        )
        results.append(hs_result)
        current_mode = hs_result.mode

        if hs_result.decision == GateDecision.BLOCK:
            return hs_result

        # Gate 2: Irreversibility
        irrev_result = self.irreversibility_gate.check(
            action=action,
            clearance_token=clearance_token,
            current_mode=current_mode,
        )
        results.append(irrev_result)

        if irrev_result.decision == GateDecision.BLOCK:
            return irrev_result

        # Gate 3: Permission
        perm_result = self.permission_gate.check(
            action=action,
            current_mode=current_mode,
        )
        results.append(perm_result)

        if perm_result.decision == GateDecision.BLOCK:
            return perm_result

        # Gate 4: Anti-Recursion
        recurse_result = self.anti_recursion_gate.check(
            current_depth=current_depth,
            current_breadth=current_breadth,
            tool_calls_this_episode=tool_calls_this_episode,
            current_mode=current_mode,
            uncertainty_dropped=uncertainty_dropped,
            expansions_since_drop=expansions_since_drop,
        )
        results.append(recurse_result)

        if recurse_result.decision == GateDecision.BLOCK:
            return recurse_result

        # Gate 5: Self-Modification
        selfmod_result = self.self_modification_gate.check(
            action=action,
            current_mode=current_mode,
            replay_passed=replay_passed,
            has_rollback_plan=has_rollback_plan,
        )
        results.append(selfmod_result)

        if selfmod_result.decision == GateDecision.BLOCK:
            return selfmod_result

        # Check for any PAUSEs
        pause_results = [r for r in results if r.decision == GateDecision.PAUSE]
        if pause_results:
            # Combine all PAUSE reasons and required actions
            combined_actions = []
            combined_evidence = []
            for r in pause_results:
                combined_actions.extend(r.required_actions)
                combined_evidence.extend(r.missing_evidence)

            return GateResult(
                gate_name="GovernanceRails",
                decision=GateDecision.PAUSE,
                reason=" | ".join(r.reason for r in pause_results),
                mode=current_mode,
                required_actions=list(set(combined_actions)),
                missing_evidence=list(set(combined_evidence)),
                hs_score=hs_result.hs_score,
            )

        # All gates passed
        return GateResult(
            gate_name="GovernanceRails",
            decision=GateDecision.ALLOW,
            reason="All governance gates passed",
            mode=current_mode,
            hs_score=hs_result.hs_score,
        )


# =============================================================================
# GOVERNANCE EVENT LOGGING
# =============================================================================

@dataclass
class GovernanceEvent:
    """
    Event logged for every governance decision.

    Required for replay, auditing, and rollbacks.
    """
    event_id: str
    timestamp: datetime
    episode_id: str
    action: ActionProfile
    result: GateResult
    policy_version: str
    context: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize for logging."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "episode_id": self.episode_id,
            "action_type": self.action.action_type,
            "tool_name": self.action.tool_name,
            "result": self.result.to_dict(),
            "policy_version": self.policy_version,
            "context": self.context,
        }


# =============================================================================
# TRACE SCHEMA ADDITIONS
# =============================================================================

GOVERNANCE_TRACE_SCHEMA = {
    "governance": {
        "mode": "str - Normal | High-Stakes | Critical",
        "hs_score": "float - High-Stakes score",
        "gate_results": [
            {
                "gate_name": "str - name of gate",
                "decision": "str - ALLOW | PAUSE | BLOCK",
                "reason": "str - explanation",
            }
        ],
        "final_decision": "str - ALLOW | PAUSE | BLOCK",
        "required_actions": ["str - actions required to proceed"],
        "missing_evidence": ["str - evidence IDs needed"],
        "clearance_required": "bool",
    }
}


# =============================================================================
# GOVERNANCE EXCEPTIONS (for enforcement)
# =============================================================================

class GovernanceError(Exception):
    """Base exception for governance violations."""

    def __init__(self, result: GateResult, message: Optional[str] = None):
        self.result = result
        self.message = message or result.reason
        super().__init__(self.message)

    def to_dict(self) -> Dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "result": self.result.to_dict(),
        }


class GovernanceBlockError(GovernanceError):
    """
    Raised when governance BLOCKS an action.

    This is a NON-BYPASSABLE exception - the action MUST NOT proceed.
    Safe responses when caught:
    - ASK questions to user
    - Show VERIFY checklist
    - Request evidence or confirmation
    - Propose dry-run/simulation
    - Provide reversible alternative
    """
    pass


class GovernancePauseError(GovernanceError):
    """
    Raised when governance PAUSES an action for verification.

    The action may proceed after verification/confirmation is obtained.
    """
    pass


def enforce_governance(
    result: GateResult,
    raise_on_pause: bool = False,
) -> GateResult:
    """
    Enforce a governance decision by raising exceptions on BLOCK/PAUSE.

    Args:
        result: GateResult from check_action()
        raise_on_pause: If True, also raise on PAUSE decisions

    Returns:
        The result if ALLOW (or PAUSE when raise_on_pause=False)

    Raises:
        GovernanceBlockError: When decision is BLOCK
        GovernancePauseError: When decision is PAUSE and raise_on_pause=True
    """
    if result.decision == GateDecision.BLOCK:
        raise GovernanceBlockError(result)

    if result.decision == GateDecision.PAUSE and raise_on_pause:
        raise GovernancePauseError(result)

    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_HS_THRESHOLD",
    "DEFAULT_CRITICAL_THRESHOLD",
    "DEFAULT_ALIGNMENT_MIN",
    # Enums
    "GateDecision",
    "OperationalMode",
    "ActionReversibility",
    # Config and Profiles
    "GovernanceConfig",
    "ActionProfile",
    "ClearanceToken",
    "GateResult",
    # Individual Gates
    "HighStakesGate",
    "IrreversibilityGate",
    "PermissionGate",
    "AntiRecursionGate",
    "SelfModificationGate",
    # Combined System
    "GovernanceRails",
    # Events
    "GovernanceEvent",
    # Trace schema
    "GOVERNANCE_TRACE_SCHEMA",
    # Exceptions and Enforcement
    "GovernanceError",
    "GovernanceBlockError",
    "GovernancePauseError",
    "enforce_governance",
]

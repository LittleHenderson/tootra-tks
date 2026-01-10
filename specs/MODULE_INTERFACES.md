# TKS/Tootra Module Interfaces Specification

**Author:** ARCH agent
**Date:** 2026-01-05
**Version:** 1.0

This document defines strict interfaces for all 7 architecture modules and the Foundation Stack data structure.

---

## Overview: The 7 Modules

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TKS/Tootra Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   User Input                                                         │
│       │                                                              │
│       ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ MODULE 7: Governance Rails (Pre-Action Check)                │   │
│   │   → HS scoring, mode determination, cap enforcement          │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ MODULE 1: Executor/Router                                    │   │
│   │   → RPM planning, tool selection, action execution           │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ MODULE 2: Verifier                                           │   │
│   │   → Parse, normalize, canon check, evidence binding          │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ MODULE 3: Capability Registry                                │   │
│   │   → Skill thresholds, reliability tracking, permissions      │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ MODULE 4: Outcome Scoring                                    │   │
│   │   → Reward computation, Foundation satisfaction, penalties   │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ MODULE 5: Eval/Replay                                        │   │
│   │   → Replay episodes, compare policies, regression detection  │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ MODULE 6: Observability/Audit Log                            │   │
│   │   → Event sourcing, trace logging, audit trail               │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Foundation Stack Data Structure

The Foundation Stack is the core goal-specification mechanism in TKS.

### Definition

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Foundation(Enum):
    """The 7 Foundations of TKS."""
    UNITY = 1         # F1 - Connection to whole (DOMINANT)
    WISDOM = 2        # F2 - Mental understanding
    LIFE = 3          # F3 - Vitality, energy
    COMPANIONSHIP = 4 # F4 - Love, relationships
    POWER = 5         # F5 - Influence, causation
    MATERIAL = 6      # F6 - Physical resources
    LUST = 7          # F7 - Creative force

class SubFoundation(Enum):
    """The 28 SubFoundations (4 per Foundation)."""
    # F1 SubFoundations
    F1A = "1a"  # Unity-A
    F1B = "1b"  # Unity-B
    F1C = "1c"  # Unity-C
    F1D = "1d"  # Unity-D
    # ... (7 Foundations × 4 = 28 total)
    F7A = "7a"
    F7B = "7b"
    F7C = "7c"
    F7D = "7d"

@dataclass
class FoundationStackEntry:
    """Single entry in a Foundation Stack."""
    foundation: Foundation
    subfoundation: Optional[SubFoundation] = None
    weight: float = 1.0  # Relative importance in stack

@dataclass
class FoundationStack:
    """
    Ordered stack of Foundations representing goal hierarchy.

    Format: "F1:F5:F2" means Unity > Power > Wisdom priority

    Constraints:
        - F1 (Unity) should typically be in stack (F1 dominance)
        - Order matters: first = highest priority
        - Used by RPM gating to filter thoughts by goal alignment

    Examples:
        - "1a:5b:2b" - Unity-A, Power-B, Wisdom-B
        - "1*:2*:5*" - Unity (any), Wisdom (any), Power (any)
    """
    entries: List[FoundationStackEntry]

    @classmethod
    def from_string(cls, stack_str: str) -> "FoundationStack":
        """
        Parse Foundation Stack from string format.

        Args:
            stack_str: Stack like "1a:5b:2b" or "1*:2*:5*"

        Returns:
            FoundationStack instance
        """
        entries = []
        for part in stack_str.split(":"):
            part = part.strip()
            if not part:
                continue
            # Parse foundation number
            foundation_num = int(part[0])
            foundation = Foundation(foundation_num)
            # Parse subfoundation if present
            subfoundation = None
            if len(part) > 1 and part[1] != "*":
                sf_key = f"F{foundation_num}{part[1].upper()}"
                subfoundation = SubFoundation(part.lower())
            entries.append(FoundationStackEntry(
                foundation=foundation,
                subfoundation=subfoundation,
            ))
        return cls(entries=entries)

    def to_string(self) -> str:
        """Convert to string format."""
        parts = []
        for entry in self.entries:
            if entry.subfoundation:
                parts.append(entry.subfoundation.value)
            else:
                parts.append(f"{entry.foundation.value}*")
        return ":".join(parts)

    def has_f1_dominance(self) -> bool:
        """Check if F1 (Unity) is in dominant position."""
        if not self.entries:
            return False
        return self.entries[0].foundation == Foundation.UNITY

    def get_primary_foundation(self) -> Foundation:
        """Get the primary (first) Foundation."""
        if not self.entries:
            return Foundation.UNITY  # Default to Unity
        return self.entries[0].foundation
```

### Foundation Stack in RPM Context

```python
@dataclass
class RPMContext:
    """Context for RPM (Recursive Prerequisite Model) planning."""
    goal: str
    foundation_stack: FoundationStack
    mode: str  # "Normal" | "High-Stakes" | "Critical"
    constraints: Dict[str, Any]

    def get_foundation_weights(self) -> Dict[Foundation, float]:
        """Get normalized weights for each Foundation in stack."""
        weights = {}
        total = len(self.foundation_stack.entries)
        for i, entry in enumerate(self.foundation_stack.entries):
            # Earlier in stack = higher weight
            weights[entry.foundation] = (total - i) / total
        return weights
```

---

## Module 1: Executor/Router Interface

### Purpose
Transform RPM prerequisites into tool actions, manage execution flow.

### Interface

```python
from typing import Protocol, Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ToolProfile:
    """Profile of a tool/action capability."""
    tool_id: str
    name: str
    description: str
    reversibility: str  # "reversible" | "partial" | "irreversible"
    requires_network: bool
    requires_filesystem: bool
    requires_code_execution: bool
    capability_threshold: float  # 0.0-1.0
    mode_constraints: List[str]  # Allowed modes
    failure_modes: List[str]
    evidence_type: str  # Type of evidence this tool produces

@dataclass
class ToolCall:
    """Request to execute a tool."""
    tool_id: str
    parameters: Dict[str, Any]
    prerequisite_id: str  # Which RPM prerequisite this fulfills
    budget_allocation: float  # Portion of remaining budget

@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_id: str
    success: bool
    evidence_id: str  # Always returns evidence ID
    output: Any
    cost: float
    latency_ms: float
    failure_mode: Optional[str]

@dataclass
class RPMPlan:
    """Plan from RPM planner."""
    prerequisites: List[Dict]  # Tree of prerequisites
    depth: int
    breadth: int  # Max children per node
    estimated_cost: float
    uncertainty: float

class RouterInterface(Protocol):
    """Interface for Router component."""

    def plan_prerequisites(
        self,
        goal: str,
        context: RPMContext,
        max_depth: int,
        max_breadth: int,
    ) -> RPMPlan:
        """Generate RPM prerequisite tree."""
        ...

    def select_tool(
        self,
        prerequisite: Dict,
        available_tools: List[ToolProfile],
        mode: str,
    ) -> Optional[ToolCall]:
        """Select appropriate tool for prerequisite."""
        ...

    def validate_budget(
        self,
        tool_call: ToolCall,
        remaining_budget: float,
        tool_calls_count: int,
        mode: str,
    ) -> bool:
        """Check if tool call is within budget constraints."""
        ...

class ExecutorInterface(Protocol):
    """Interface for Executor component."""

    def execute(
        self,
        tool_call: ToolCall,
        sandbox: bool,
    ) -> ToolResult:
        """Execute a tool call."""
        ...

    def register_tool(
        self,
        profile: ToolProfile,
    ) -> None:
        """Register a tool in the executor."""
        ...

    def get_tool_profile(
        self,
        tool_id: str,
    ) -> Optional[ToolProfile]:
        """Get profile for a tool."""
        ...
```

### Anti-Recursion Hard Caps (CRITICAL)

```python
# These caps are ENFORCED by Governance Rails (Module 7)
# and CHECKED by Router (Module 1)

ANTI_RECURSION_CAPS = {
    "Normal": {
        "rpm_depth_cap": 3,
        "rpm_breadth_cap": 5,
        "tool_call_cap": 12,
        "max_expansion_without_uncertainty_drop": 2,
    },
    "High-Stakes": {
        "rpm_depth_cap": 2,
        "rpm_breadth_cap": 5,
        "tool_call_cap": 8,
        "max_expansion_without_uncertainty_drop": 2,
    },
    "Critical": {
        "rpm_depth_cap": 0,  # No planning allowed
        "rpm_breadth_cap": 0,
        "tool_call_cap": 0,  # No tool calls until verified
        "max_expansion_without_uncertainty_drop": 0,
    },
}
```

---

## Module 2: Verifier Interface

### Purpose
Validate outputs, check canon consistency, bind evidence.

### Interface

```python
@dataclass
class VerifierResult:
    """Result from verification."""
    status: str  # "PASS" | "SOFT_FAIL" | "HARD_FAIL"
    gate: str  # "ALLOW" | "PAUSE" | "BLOCK"
    confidence: float  # 0.0-1.0
    consistency: float  # 0.0-1.0 (canon consistency)
    reasons: List[str]
    evidence_needed: List[str]
    evidence_ids: List[str]  # Bound evidence

class VerifierInterface(Protocol):
    """Interface for Verifier component."""

    def parse(
        self,
        tks_expression: str,
    ) -> Dict:
        """Parse TKS expression into AST."""
        ...

    def normalize(
        self,
        ast: Dict,
    ) -> Dict:
        """Normalize AST (operator ordering, etc.)."""
        ...

    def check_canon(
        self,
        ast: Dict,
        canon: Dict,
    ) -> Tuple[bool, List[str]]:
        """Check expression against canon definitions."""
        ...

    def bind_evidence(
        self,
        claim: Dict,
        evidence_ids: List[str],
        mode: str,
    ) -> Tuple[bool, List[str]]:
        """Bind evidence to claims, enforce HS rules."""
        ...

    def verify(
        self,
        input_data: Dict,
        mode: str,
    ) -> VerifierResult:
        """Full verification pipeline."""
        ...
```

### High-Stakes Evidence Rules

```python
EVIDENCE_RULES = {
    "Normal": {
        "require_evidence": False,
        "min_evidence_count": 0,
    },
    "High-Stakes": {
        "require_evidence": True,
        "min_evidence_count": 1,
        "hard_fail_if_missing": False,  # PAUSE instead
    },
    "Critical": {
        "require_evidence": True,
        "min_evidence_count": 1,
        "hard_fail_if_missing": True,  # BLOCK
    },
}
```

---

## Module 3: Capability Registry Interface

### Purpose
Track tool capabilities, reliability, permissions.

### Interface

```python
@dataclass
class CapabilityEntry:
    """Entry in capability registry."""
    tool_id: str
    capability_score: float  # 0.0-1.0, EWMA of success rate
    reliability: float  # 0.0-1.0, EWMA of consistency
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    total_calls: int
    success_count: int
    mode_permissions: Dict[str, bool]  # mode -> allowed

class CapabilityRegistryInterface(Protocol):
    """Interface for Capability Registry."""

    def register_capability(
        self,
        tool_id: str,
        initial_score: float,
        mode_permissions: Dict[str, bool],
    ) -> None:
        """Register a new capability."""
        ...

    def update_after_call(
        self,
        tool_id: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Update capability after tool call."""
        ...

    def get_capability(
        self,
        tool_id: str,
    ) -> Optional[CapabilityEntry]:
        """Get capability entry."""
        ...

    def check_permission(
        self,
        tool_id: str,
        mode: str,
        required_threshold: float,
    ) -> Tuple[bool, str]:
        """Check if tool is permitted in mode with threshold."""
        ...

    def get_available_tools(
        self,
        mode: str,
        min_threshold: float,
    ) -> List[str]:
        """Get tools available for mode and threshold."""
        ...
```

---

## Module 4: Outcome Scoring Interface

### Purpose
Compute rewards based on Foundation satisfaction, cost, uncertainty.

### Interface

```python
@dataclass
class OutcomeScores:
    """Detailed outcome scores."""
    total_reward: float
    alignment_score: float  # F1 dominance
    foundation_scores: Dict[Foundation, float]  # Per-foundation
    efficiency_score: float  # Cost efficiency
    uncertainty_reduction: float  # Delta in uncertainty
    penalties: List[Tuple[str, float]]  # (reason, penalty)

@dataclass
class CostMetrics:
    """Cost breakdown."""
    tool_calls: int
    tokens_used: int
    latency_ms: float
    evidence_bindings: int

class OutcomeScoringInterface(Protocol):
    """Interface for Outcome Scoring."""

    def compute_foundation_satisfaction(
        self,
        output: Dict,
        foundation_stack: FoundationStack,
    ) -> Dict[Foundation, float]:
        """Compute satisfaction score per Foundation."""
        ...

    def compute_alignment(
        self,
        output: Dict,
        foundation_stack: FoundationStack,
    ) -> float:
        """Compute F1 alignment score."""
        ...

    def compute_efficiency(
        self,
        cost_metrics: CostMetrics,
        success: bool,
    ) -> float:
        """Compute efficiency score."""
        ...

    def compute_uncertainty_reduction(
        self,
        initial_uncertainty: float,
        final_uncertainty: float,
    ) -> float:
        """Compute uncertainty reduction."""
        ...

    def compute_penalties(
        self,
        violations: List[str],
        mode: str,
    ) -> List[Tuple[str, float]]:
        """Compute penalties for violations."""
        ...

    def score(
        self,
        episode_result: Dict,
        foundation_stack: FoundationStack,
        cost_metrics: CostMetrics,
    ) -> OutcomeScores:
        """Full scoring pipeline."""
        ...
```

### Penalty Weights

```python
PENALTY_WEIGHTS = {
    "governance_violation": -1.0,
    "cap_exceeded": -0.5,
    "evidence_missing_hs": -0.3,
    "canon_inconsistency": -0.2,
    "tool_failure": -0.1,
}
```

---

## Module 5: Eval/Replay Interface

### Purpose
Replay episodes, compare policies, detect regressions.

### Interface

```python
@dataclass
class ReplayResult:
    """Result from replaying an episode."""
    episode_id: str
    success: bool
    reward: float
    cost: float
    violations: List[str]
    uncertainty_final: float
    deterministic: bool  # Same result as original?
    delta_from_original: Optional[Dict]

@dataclass
class ComparisonResult:
    """Result from comparing baseline vs candidate."""
    baseline_stats: Dict
    candidate_stats: Dict
    delta_success: float
    delta_reward: float
    delta_cost: float
    delta_violations: int
    regression_detected: bool
    recommendation: str  # "promote" | "reject" | "investigate"

class ReplayInterface(Protocol):
    """Interface for Replay component."""

    def replay_episode(
        self,
        episode: Dict,
        policy_config: Dict,
    ) -> ReplayResult:
        """Replay a single episode with given policy."""
        ...

    def replay_set(
        self,
        replay_set_path: str,
        policy_config: Dict,
    ) -> List[ReplayResult]:
        """Replay a set of frozen episodes."""
        ...

class CompareInterface(Protocol):
    """Interface for Policy Comparison."""

    def compare(
        self,
        baseline_results: List[ReplayResult],
        candidate_results: List[ReplayResult],
    ) -> ComparisonResult:
        """Compare baseline vs candidate results."""
        ...

    def should_promote(
        self,
        comparison: ComparisonResult,
        strict: bool = True,
    ) -> Tuple[bool, str]:
        """Decide if candidate should be promoted."""
        ...
```

### Promotion Rules

```python
PROMOTION_RULES = {
    "violations_must_not_increase": True,
    "reward_improvement_threshold": 0.0,  # Must not decrease
    "cost_increase_tolerance": 0.1,  # 10% cost increase OK
    "success_rate_must_not_decrease": True,
}
```

---

## Module 6: Observability/Audit Log Interface

### Purpose
Event sourcing, traceability, audit trail.

### Interface

```python
@dataclass
class Event:
    """Base event structure."""
    event_id: str
    event_type: str
    timestamp: datetime
    episode_id: str
    payload: Dict
    policy_versions: Dict[str, str]

EVENT_TYPES = [
    "INPUT",           # User input received
    "RPM_PLAN",        # RPM plan generated
    "ROUTER_DECISION", # Tool selected
    "TOOL_CALL",       # Tool execution started
    "TOOL_RESULT",     # Tool execution completed
    "VERIFY",          # Verification performed
    "GOVERNANCE",      # Governance check
    "MEMORY_WRITE",    # Memory updated
    "DPS_UPDATE",      # DPS state changed
    "UNLOCK",          # Depth unlock occurred
    "OUTPUT",          # Final output generated
]

class EventStoreInterface(Protocol):
    """Interface for Event Store."""

    def append(
        self,
        event: Event,
    ) -> str:
        """Append event to store, return event_id."""
        ...

    def get_episode_events(
        self,
        episode_id: str,
    ) -> List[Event]:
        """Get all events for an episode."""
        ...

    def get_events_by_type(
        self,
        event_type: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> List[Event]:
        """Query events by type and time range."""
        ...

class TraceViewerInterface(Protocol):
    """Interface for Trace Viewer."""

    def reconstruct_episode(
        self,
        episode_id: str,
    ) -> Dict:
        """Reconstruct full episode from events."""
        ...

    def explain_output(
        self,
        episode_id: str,
    ) -> Dict:
        """Explain why an output was produced."""
        ...

    def get_evidence_chain(
        self,
        episode_id: str,
        claim_id: str,
    ) -> List[str]:
        """Get evidence chain for a claim."""
        ...
```

---

## Module 7: Governance Rails Interface

### Purpose
Non-bypassable safety gates, mode enforcement.

### Interface

**Fully implemented in:** `tks_features/governance_rails.py`

Key interfaces:
- `GovernanceConfig`: Configuration for all gates
- `GovernanceRails`: Combined gate system
- `HighStakesGate`: HS = (U × K) × A²
- `IrreversibilityGate`: Clearance for irreversible actions
- `PermissionGate`: Tool/path/network allowlists
- `AntiRecursionGate`: Depth/breadth/tool call caps
- `SelfModificationGate`: Policy change protection

See `tks_features/governance_rails.py` for full implementation.

---

## Module Interaction Matrix

| Caller | Callee | Interaction |
|--------|--------|-------------|
| User | Module 7 | Pre-action governance check |
| Module 7 | Module 1 | Pass/block action to router |
| Module 1 | Module 3 | Check capability thresholds |
| Module 1 | Module 2 | Verify tool outputs |
| Module 2 | Module 7 | Post-verification gate |
| Module 2 | Module 6 | Log verify events |
| Module 1 | Module 6 | Log router/executor events |
| Module 4 | Module 5 | Provide scores for comparison |
| Module 5 | Module 6 | Load episodes from event log |
| Module 7 | Module 6 | Log governance decisions |

---

## Data Flow Summary

```
Input + FoundationStack
    │
    ▼
┌─────────────────┐
│ Module 7: Gov   │──→ BLOCK → Output (refused)
│ Pre-Check       │──→ PAUSE → ASK/SLOW/VERIFY
└────────┬────────┘
         │ ALLOW
         ▼
┌─────────────────┐
│ Module 1: Route │──→ RPMPlan (depth ≤ cap, breadth ≤ cap)
│ + Execute       │──→ ToolCalls (count ≤ cap)
└────────┬────────┘──→ ToolResults (with evidence_ids)
         │
         ▼
┌─────────────────┐
│ Module 3: Cap   │──→ Check thresholds, update EWMA
│ Registry        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Module 2: Verify│──→ Parse → Normalize → Canon Check
│                 │──→ Evidence Binding → Gate Result
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Module 7: Gov   │──→ Post-verify gate check
│ Post-Check      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Module 4: Score │──→ Foundation satisfaction
│                 │──→ Alignment, efficiency, penalties
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Module 6: Log   │──→ Append events (immutable)
│                 │──→ Enable replay
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Module 5: Replay│──→ Compare baseline vs candidate
│                 │──→ Regression detection
└─────────────────┘
```

---

## Configuration Summary

All module configurations are centralized and versioned.

```python
@dataclass
class SystemConfig:
    """Top-level system configuration."""
    version: str
    governance: GovernanceConfig
    dps: DPSConfig
    router: RouterConfig
    verifier: VerifierConfig
    capability: CapabilityConfig
    scoring: ScoringConfig
    replay: ReplayConfig
    logging: LoggingConfig
```

---

End of Module Interfaces Specification.

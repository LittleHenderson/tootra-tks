"""
TKS RPM Planning Module - Recursive Prerequisite Management

This module implements the RPM (Recursive Prerequisite Management) planning
system that decomposes goals into prerequisite trees.

Key concepts:
    - Prerequisites are organized as nested trees (depth ≤ max_depth)
    - Regulators (FM, ACBE, MVR) are triggered when gaps are detected
    - The DWP (Desire-Wisdom-Power) product determines which foundation to use

Integration:
    - Uses regulators from src/tks/regulators
    - Respects governance caps on depth/breadth
    - Outputs plans for the Router to execute
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid

# Import regulators from sibling module
try:
    from ..regulators.regulators import TKSNode, TKSRegulators, RPMPlanner as RegulatorPlanner
except ImportError:
    # Fallback for direct execution
    TKSNode = None
    TKSRegulators = None
    RegulatorPlanner = None


class GapType(Enum):
    """Types of gaps that can be detected in prerequisite planning."""
    WISDOM = "wisdom"           # Need more understanding → FM regulator
    CAPABILITY = "capability"   # Need tools/skills → ACBE regulator
    MOTIVATION = "motivation"   # Low desire → MVR regulator
    EVIDENCE = "evidence"       # Need verification data
    CONSTRAINT = "constraint"   # Need constraint satisfaction


class PrerequisiteStatus(Enum):
    """Status of a prerequisite node."""
    PENDING = "pending"         # Not yet evaluated
    SATISFIED = "satisfied"     # Condition met
    UNSATISFIED = "unsatisfied" # Gap detected
    BLOCKED = "blocked"         # Governance blocked
    EXECUTING = "executing"     # Currently being resolved


@dataclass
class PrerequisiteNode:
    """A node in the RPM prerequisite tree."""
    id: str
    expression: str
    foundation_stack: str
    status: PrerequisiteStatus = PrerequisiteStatus.PENDING
    gap_type: Optional[GapType] = None
    depth: int = 0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize for logging/transport."""
        return {
            "id": self.id,
            "expression": self.expression,
            "foundation_stack": self.foundation_stack,
            "status": self.status.value,
            "gap_type": self.gap_type.value if self.gap_type else None,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children": self.children,
            "meta": self.meta,
        }


@dataclass
class RPMPlan:
    """Complete RPM plan for an episode."""
    goal: str
    root_id: str
    nodes: Dict[str, PrerequisiteNode] = field(default_factory=dict)
    max_depth: int = 3
    max_breadth: int = 5
    mode: str = "Normal"

    def add_node(self, node: PrerequisiteNode) -> None:
        """Add a node to the plan."""
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children:
                parent.children.append(node.id)

    def get_pending_nodes(self) -> List[PrerequisiteNode]:
        """Get all pending (ready to execute) nodes."""
        return [n for n in self.nodes.values() if n.status == PrerequisiteStatus.PENDING]

    def get_leaf_nodes(self) -> List[PrerequisiteNode]:
        """Get all leaf nodes (no children)."""
        return [n for n in self.nodes.values() if not n.children]

    def to_dict(self) -> Dict:
        """Serialize the plan."""
        return {
            "goal": self.goal,
            "root_id": self.root_id,
            "max_depth": self.max_depth,
            "max_breadth": self.max_breadth,
            "mode": self.mode,
            "prerequisites": [n.to_dict() for n in self.nodes.values()],
            "node_count": len(self.nodes),
        }


class RPMNestingLogic:
    """
    Core RPM nesting logic implementing TKS prerequisite decomposition.

    Enforces:
        - Nesting normalization: A:B:C == A:(B:C)
        - Depth caps based on mode
        - Breadth caps per node
        - Regulator triggering based on gap type
    """

    def __init__(
        self,
        max_depth_normal: int = 3,
        max_depth_hs: int = 2,
        max_breadth: int = 5,
    ):
        self.max_depth_normal = max_depth_normal
        self.max_depth_hs = max_depth_hs
        self.max_breadth = max_breadth

        # Initialize regulator planner if available
        self.regulator_planner = RegulatorPlanner() if RegulatorPlanner else None

    def get_max_depth(self, mode: str) -> int:
        """Get max depth based on operational mode."""
        if mode in ("High-Stakes", "HIGH_STAKES"):
            return self.max_depth_hs
        elif mode in ("Critical", "CRITICAL"):
            return 1  # Minimal depth in Critical
        return self.max_depth_normal

    def normalize_expression(self, expression: str) -> str:
        """
        Normalize TKS expression to canonical nesting.

        Rule: A:B:C normalizes to A:(B:C) - right-associative nesting.
        """
        parts = expression.split(":")
        if len(parts) <= 2:
            return expression

        # Build right-associative nesting
        result = parts[-1]
        for i in range(len(parts) - 2, -1, -1):
            result = f"{parts[i]}:({result})"
        return result

    def detect_gap_type(self, node: PrerequisiteNode) -> Optional[GapType]:
        """
        Analyze a node to detect what type of gap exists.

        Uses foundation stack and meta hints to classify.
        """
        stack = node.foundation_stack.lower()
        meta = node.meta or {}

        # Check explicit hints first
        if meta.get("gap_type"):
            try:
                return GapType(meta["gap_type"])
            except ValueError:
                pass

        # Infer from foundation stack
        if "f2" in stack or "wisdom" in str(meta).lower():
            return GapType.WISDOM
        if "f5" in stack or "capability" in str(meta).lower():
            return GapType.CAPABILITY
        if "low_desire" in str(node.status).lower():
            return GapType.MOTIVATION
        if "evidence" in str(meta).lower():
            return GapType.EVIDENCE

        # Default to wisdom gap
        return GapType.WISDOM

    def trigger_regulator(
        self,
        node: PrerequisiteNode,
        gap_type: GapType,
    ) -> List[PrerequisiteNode]:
        """
        Trigger appropriate regulator to generate sub-prerequisites.

        Regulators:
            - FM (Foundation Modulator): Wisdom gaps → Female + Male
            - ACBE (Above/Below Cause/Effect): Capability gaps → Above + Below
            - MVR (Mind/Vibration/Rhythm): Motivation gaps → Desire builder
        """
        if not self.regulator_planner:
            # Fallback if regulators not available
            return self._simple_expand(node, gap_type)

        # Convert to TKS node format
        tks_node = TKSNode(
            expression=node.expression,
            foundation_stack=node.foundation_stack,
            status="unsatisfied" if gap_type != GapType.MOTIVATION else "low_desire",
            meta={"gap_type": gap_type.value}
        )

        # Get regulator output
        sub_nodes = self.regulator_planner.expand_gap(tks_node)

        # Convert back to PrerequisiteNode format
        result = []
        for sub in sub_nodes:
            new_node = PrerequisiteNode(
                id=f"prereq_{uuid.uuid4().hex[:8]}",
                expression=sub.expression,
                foundation_stack=sub.foundation_stack,
                status=PrerequisiteStatus.PENDING,
                depth=node.depth + 1,
                parent_id=node.id,
                meta=sub.meta or {},
            )
            result.append(new_node)

        return result

    def _simple_expand(
        self,
        node: PrerequisiteNode,
        gap_type: GapType,
    ) -> List[PrerequisiteNode]:
        """Simple fallback expansion when regulators unavailable."""
        base_id = f"prereq_{uuid.uuid4().hex[:8]}"

        if gap_type == GapType.WISDOM:
            return [
                PrerequisiteNode(
                    id=f"{base_id}_auth",
                    expression=f"({node.expression})^5",  # Female - authority
                    foundation_stack="F2:F4:F1",
                    depth=node.depth + 1,
                    parent_id=node.id,
                    meta={"regulator": "FM", "type": "Female/Authority"},
                ),
                PrerequisiteNode(
                    id=f"{base_id}_rel",
                    expression=f"({node.expression})^6",  # Male - related ideas
                    foundation_stack="F2:F5:F1",
                    depth=node.depth + 1,
                    parent_id=node.id,
                    meta={"regulator": "FM", "type": "Male/Superset"},
                ),
            ]
        elif gap_type == GapType.CAPABILITY:
            return [
                PrerequisiteNode(
                    id=f"{base_id}_cause",
                    expression=f"({node.expression})^8",  # Above - cause
                    foundation_stack="F5:F6:F1",
                    depth=node.depth + 1,
                    parent_id=node.id,
                    meta={"regulator": "ACBE", "type": "Above/Cause"},
                ),
                PrerequisiteNode(
                    id=f"{base_id}_effect",
                    expression=f"({node.expression})^9",  # Below - effect
                    foundation_stack="F5:F2:F1",
                    depth=node.depth + 1,
                    parent_id=node.id,
                    meta={"regulator": "ACBE", "type": "Below/Effect"},
                ),
            ]
        elif gap_type == GapType.MOTIVATION:
            return [
                PrerequisiteNode(
                    id=f"{base_id}_mvr",
                    expression=f"((({node.expression})^7)^4)^1",  # MVR chain
                    foundation_stack="F1:F3:F1",
                    depth=node.depth + 1,
                    parent_id=node.id,
                    meta={"regulator": "MVR", "type": "Desire_Builder"},
                ),
            ]
        else:
            # Evidence/Constraint - simple single node
            return [
                PrerequisiteNode(
                    id=f"{base_id}_resolve",
                    expression=f"RESOLVE({node.expression})",
                    foundation_stack=node.foundation_stack,
                    depth=node.depth + 1,
                    parent_id=node.id,
                    meta={"gap_type": gap_type.value},
                ),
            ]

    def expand_plan(
        self,
        plan: RPMPlan,
        node: PrerequisiteNode,
    ) -> List[PrerequisiteNode]:
        """
        Expand a single node in the plan, respecting caps.

        Returns list of new nodes added (empty if at cap or satisfied).
        """
        max_depth = self.get_max_depth(plan.mode)

        # Check depth cap
        if node.depth >= max_depth:
            node.meta["capped"] = True
            node.meta["cap_reason"] = "depth_limit"
            return []

        # Check breadth cap
        if len(node.children) >= self.max_breadth:
            node.meta["capped"] = True
            node.meta["cap_reason"] = "breadth_limit"
            return []

        # Detect gap type
        gap_type = self.detect_gap_type(node)
        if not gap_type:
            return []

        node.gap_type = gap_type

        # Trigger regulator
        new_nodes = self.trigger_regulator(node, gap_type)

        # Respect breadth cap
        available_slots = self.max_breadth - len(node.children)
        new_nodes = new_nodes[:available_slots]

        # Add to plan
        for new_node in new_nodes:
            plan.add_node(new_node)

        return new_nodes


def create_plan(
    goal: str,
    foundation_stack: str = "1a",
    mode: str = "Normal",
    max_depth: int = 3,
    max_breadth: int = 5,
) -> RPMPlan:
    """
    Create an initial RPM plan from a goal.

    Args:
        goal: The goal to decompose
        foundation_stack: Initial foundation stack (e.g., "1a:5b")
        mode: Operational mode (Normal, High-Stakes, Critical)
        max_depth: Maximum tree depth
        max_breadth: Maximum children per node

    Returns:
        RPMPlan with root node
    """
    root_id = f"prereq_{uuid.uuid4().hex[:8]}"
    root = PrerequisiteNode(
        id=root_id,
        expression=goal,
        foundation_stack=foundation_stack,
        status=PrerequisiteStatus.PENDING,
        depth=0,
        meta={"is_root": True},
    )

    plan = RPMPlan(
        goal=goal,
        root_id=root_id,
        max_depth=max_depth,
        max_breadth=max_breadth,
        mode=mode,
    )
    plan.add_node(root)

    return plan


# Convenience exports
__all__ = [
    "GapType",
    "PrerequisiteStatus",
    "PrerequisiteNode",
    "RPMPlan",
    "RPMNestingLogic",
    "create_plan",
]

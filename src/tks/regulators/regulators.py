"""
TKS Regulators (FM, ACBE, MVR) - RPM Prerequisite Resolvers

These routines are triggered when an RPM expansion hits an unsatisfied 
prerequisite (Gap). They generate the necessary sub-desires to bridge 
the gap.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TKSNode:
    """A single node in an RPM cascade."""
    expression: str
    foundation_stack: str  # e.g., "1a:5b:2b"
    status: str = "unsatisfied" # satisfied, unsatisfied, low_desire
    meta: Dict = None


class TKSRegulators:
    """
    Sub-routines that apply specific Noetic logic to resolve Gaps.
    """

    @staticmethod
    def fm_wisdom_regulator(target_idea: str) -> List[TKSNode]:
        """
        FM (Foundation Modulator) - Triggered by Wisdom gaps.
        Uses Noetics 5 (Female) and 6 (Male).
        """
        # Female: Who/What is the authority/reservoir?
        node_female = TKSNode(
            expression=f"({target_idea})^5",
            foundation_stack="F2:F4:F1", # Wisdom through Trust constrained by Alignment
            meta={"regulator": "FM", "type": "Female/Authority"}
        )
        # Male: What are the related ideas/supersets?
        node_male = TKSNode(
            expression=f"({target_idea})^6",
            foundation_stack="F2:F5:F1", # Wisdom through Power/Control
            meta={"regulator": "FM", "type": "Male/Superset"}
        )
        return [node_female, node_male]

    @staticmethod
    def acbe_capability_regulator(required_capability: str) -> List[TKSNode]:
        """
        ACBE (Above/Cause, Below/Effect) - Triggered by Capability gaps.
        Uses Noetics 8 (Above) and 9 (Below).
        """
        # Above: What causes this capability?
        node_above = TKSNode(
            expression=f"({required_capability})^8",
            foundation_stack="F5:F6:F1", # Power through Material/Tools
            meta={"regulator": "ACBE", "type": "Above/Cause"}
        )
        # Below: What are the effects of this capability?
        node_below = TKSNode(
            expression=f"({required_capability})^9",
            foundation_stack="F5:F2:F1", # Power through Wisdom
            meta={"regulator": "ACBE", "type": "Below/Effect"}
        )
        return [node_above, node_below]

    @staticmethod
    def mvr_desire_regulator(idea_with_no_desire: str) -> List[TKSNode]:
        """
        MVR (Mind, Vibration, Rhythm) - Triggered by Low Motivation.
        Uses Noetics 1, 4, and 7.
        """
        # A single nested expression representing the intensity loop
        # Mind(Vibration(Rhythm(Idea)))
        node_mvr = TKSNode(
            expression=f"(({idea_with_no_desire})^7)^4)^1",
            foundation_stack="F1:F3:F1", # Alignment through Life/Stability
            meta={"regulator": "MVR", "type": "Desire_Builder"}
        )
        return [node_mvr]


class RPMPlanner:
    def __init__(self):
        self.regulators = TKSRegulators()

    def expand_gap(self, node: TKSNode) -> List[TKSNode]:
        """
        Examines a node and decides which regulator to trigger.
        """
        if node.status == "low_desire":
            return self.regulators.mvr_desire_regulator(node.expression)
        
        # In a real implementation, we would use a classifier or 
        # the Foundation/Acquisition type to decide between FM and ACBE.
        # For now, we'll use a simple keyword check in metadata.
        reg_type = node.meta.get("gap_type") if node.meta else "wisdom"
        
        if reg_type == "capability":
            return self.regulators.acbe_capability_regulator(node.expression)
        else:
            return self.regulators.fm_wisdom_regulator(node.expression)


# --- Example Usage ---
if __name__ == "__main__":
    planner = RPMPlanner()
    
    # Example: A missing capability "Quantum Inference"
    gap_node = TKSNode(
        expression="Quantum_Inference", 
        foundation_stack="5d", 
        meta={"gap_type": "capability"}
    )
    
    sub_nodes = planner.expand_gap(gap_node)
    print(f"Expanding Gap: {gap_node.expression}")
    for n in sub_nodes:
        print(f"  -> Generated: {n.expression} | Stack: {n.foundation_stack} | Type: {n.meta['type']}")

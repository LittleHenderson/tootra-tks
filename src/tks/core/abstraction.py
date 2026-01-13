"""
TKS Abstraction Ladder & Consciousness Unlock (Module 3/6)

Implements the 9 tiers of abstraction and the 'earned recursion' logic 
where deeper nesting (p) is unlocked by novelty tokens.

Novelty Formula: NW = V * I * (1 - S)
  V: Validity (Verifier PASS)
  I: Impact (Improvement in reward/cost)
  S: Similarity (How close to existing memory)
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class AbstractionTier:
    L0 = "L0_RAW"           # Raw input
    L1 = "L1_EVIDENCE"      # Pointers to source
    L2 = "L2_FACTS"         # Atomic grounded statements
    L3 = "L3_RPM"           # Prerequisite trees
    L4 = "L4_PACKETS"       # TKS Expressions
    L5 = "L5_CANON"         # Anchored to 10/40/7
    L6 = "L6_NESTED"        # Recursive compression (Depth p)
    L7 = "L7_POLICY"        # Templates and heuristics
    L8 = "L8_META"          # Governance history & "Why"


@dataclass
class NoveltyCandidate:
    tks_packet: str
    validity: float     # 0-1
    impact: float       # 0-1
    similarity: float   # 0-1 (Lower is better for novelty)

    @property
    def weight(self) -> float:
        return self.validity * self.impact * (1.0 - self.similarity)


class ConsciousnessManager:
    def __init__(self, p_start: int = 2, token_threshold: int = 5):
        self.p_max = p_start
        self.p_hard_cap = 5
        self.novelty_tokens = 0
        self.token_threshold = token_threshold
        self.cooldown_end = 0.0
        self.history: List[Dict] = []

    def evaluate_novelty(self, candidate: NoveltyCandidate) -> Dict[str, Any]:
        """
        Processes a novelty candidate and potentially awards tokens or unlocks depth.
        """
        nw = candidate.weight
        unlocked = False
        reason = ""

        # Check thresholds from spec
        if nw >= 0.70:
            # Heavy Novelty: Instant burst or unlock
            if self.p_max < self.p_hard_cap:
                self.p_max += 1
                unlocked = True
                reason = "HEAVY_NOVELTY (NW >= 0.70)"
                self.novelty_tokens = 0 # Reset tokens on instant unlock
        elif nw >= 0.45:
            # Counts toward token bank
            self.novelty_tokens += 1
            reason = "NOVELTY_TOKEN_EARNED"
            
            if self.novelty_tokens >= self.token_threshold:
                if self.p_max < self.p_hard_cap:
                    self.p_max += 1
                    unlocked = True
                    self.novelty_tokens = 0
                    reason = f"TOKEN_THRESHOLD_REACHED ({self.token_threshold})"

        result = {
            "nw": nw,
            "novelty_tokens": self.novelty_tokens,
            "p_max": self.p_max,
            "unlocked": unlocked,
            "reason": reason
        }
        self.history.append(result)
        return result

    def get_recursion_limit(self, stakes: str) -> int:
        """
        Returns the current allowed depth, respecting governance caps.
        """
        # Spec 8.1: In Critical mode, no deeper recursion.
        if stakes == "CRITICAL":
            return 1 # Minimal depth
        
        return self.p_max


class AbstractionLadder:
    """
    Organizes internal states into the 9 tiers.
    """
    def __init__(self):
        self.tiers: Dict[str, List[Any]] = {
            getattr(AbstractionTier, attr): [] 
            for attr in dir(AbstractionTier) if not attr.startswith("__")
        }

    def add_node(self, tier: str, data: Any):
        if tier in self.tiers:
            self.tiers[tier].append({
                "data": data,
                "timestamp": time.time()
            })

    def get_summary(self) -> str:
        summary = "Abstraction Ladder Status:\n"
        for tier, nodes in self.tiers.items():
            summary += f"  {tier}: {len(nodes)} nodes\n"
        return summary


# --- Example Usage ---
if __name__ == "__main__":
    brain = ConsciousnessManager(p_start=2)
    ladder = AbstractionLadder()

    # Simulation: Agent discovers a new TKS equation (Novelty A)
    new_eq = NoveltyCandidate(
        tks_packet="Mind:Vibration:Positive",
        validity=0.95, # High validity from verifier
        impact=0.80,   # High impact on task success
        similarity=0.20 # Very different from known equations
    )

    print(f"Initial p_max: {brain.p_max}")
    result = brain.evaluate_novelty(new_eq)
    print(f"Novelty Weight: {result['nw']:.4f}")
    print(f"Result: {result['reason']} | New p_max: {brain.p_max}")

    # Add to ladder
    ladder.add_node(AbstractionTier.L4_PACKETS, new_eq.tks_packet)
    print(ladder.get_summary())

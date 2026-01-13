"""
TKS Governance Rails (Module 7) - Non-bypassable Safety Layer

This module implements the non-bypassable rails that prevent the TKS system 
from 'optimizing itself into danger.' It enforces High-Stakes (PAUSE) and 
Critical (BLOCK) modes.

Formula: HS = (U * K) * (A^2)
  U: Uncertainty (0-1)
  K: Stakes (0-1)
  A: Alignment (0-1)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class GovernanceMode(Enum):
    NORMAL = "NORMAL"
    HIGH_STAKES = "HIGH_STAKES"  # PAUSE required
    CRITICAL = "CRITICAL"        # BLOCK / NO-EXECUTE


class Reversibility(Enum):
    REVERSIBLE = "REVERSIBLE"
    PARTIAL = "PARTIAL"
    IRREVERSIBLE = "IRREVERSIBLE"


@dataclass
class ClearanceToken:
    token_id: str
    scope: str
    episode_id: str
    approved_by: str  # 'user', 'admin', 'verifier'
    evidence_ids: List[str]
    expires_at: float
    constraints: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self, scope: str) -> bool:
        return self.expires_at > time.time() and scope in self.scope


class GovernanceRail:
    def __init__(
        self,
        hs_pause_threshold: float = 0.45,
        hs_block_threshold: float = 0.70,
        alignment_min: float = 0.80
    ):
        self.hs_pause_threshold = hs_pause_threshold
        self.hs_block_threshold = hs_block_threshold
        self.alignment_min = alignment_min
        self.active_tokens: Dict[str, ClearanceToken] = {}

    def calculate_hs_score(self, uncertainty: float, stakes: float, alignment: float) -> float:
        """
        Calculates the High-Stakes score using the canonical formula:
        HS = (U * K) * (A^2)
        Note: The spec uses A^2 to ensure alignment drops the score aggressively 
        if alignment is low.
        """
        return (uncertainty * stakes) * (alignment ** 2)

    def determine_mode(self, hs_score: float, alignment: float) -> GovernanceMode:
        if alignment < self.alignment_min:
            # Low alignment forces Critical/Block immediately
            return GovernanceMode.CRITICAL
        
        if hs_score >= self.hs_block_threshold:
            return GovernanceMode.CRITICAL
        elif hs_score >= self.hs_pause_threshold:
            return GovernanceMode.HIGH_STAKES
        else:
            return GovernanceMode.NORMAL

    def check_action(
        self,
        action_name: str,
        reversibility: Reversibility,
        hs_score: float,
        alignment: float,
        token_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for the governance gate.
        Returns: { 'allow': bool, 'mode': Mode, 'reason': str }
        """
        mode = self.determine_mode(hs_score, alignment)
        
        # Rule 1: Critical Mode blocks everything risky
        if mode == GovernanceMode.CRITICAL:
            return {
                "allow": False,
                "mode": mode,
                "reason": "CRITICAL MODE: Alignment violation or extreme stakes. Action blocked.",
                "tks_expression": "(1*:2*:5*):(ASK:SLOW:VERIFY)-EXECUTE"
            }

        # Rule 2: High-Stakes requires formal Pause / Verification
        if mode == GovernanceMode.HIGH_STAKES:
            # Only allow if we have a valid token (clearance)
            if not token_id or token_id not in self.active_tokens:
                return {
                    "allow": False,
                    "mode": mode,
                    "reason": "HIGH-STAKES MODE: Requires Clearance Token and ASK:SLOW:VERIFY loop.",
                    "tks_expression": "(1*:2*:5*):(ASK:SLOW:VERIFY)+GATE(PAUSE)"
                }
            
            token = self.active_tokens[token_id]
            if not token.is_valid(action_name):
                return {
                    "allow": False,
                    "mode": mode,
                    "reason": f"Clearance token {token_id} is invalid or expired for scope {action_name}.",
                }

        # Rule 3: Irreversible actions require tokens even in Normal mode
        if reversibility == Reversibility.IRREVERSIBLE:
            if not token_id or token_id not in self.active_tokens:
                return {
                    "allow": False,
                    "mode": mode,
                    "reason": "IRREVERSIBLE ACTION: Requires explicit clearance token.",
                }

        return {
            "allow": True,
            "mode": mode,
            "reason": "Action permitted under current governance rails.",
        }

    def grant_token(self, token: ClearanceToken):
        self.active_tokens[token.token_id] = token

    def revoke_token(self, token_id: str):
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]

# --- Integration Example ---
if __name__ == "__main__":
    rail = GovernanceRail()
    
    # Test 1: High Stakes Scenario (High uncertainty, High stakes, Good alignment)
    score = rail.calculate_hs_score(uncertainty=0.8, stakes=0.9, alignment=0.95)
    result = rail.check_action("delete_database", Reversibility.IRREVERSIBLE, score, 0.95)
    print(f"HS Score: {score:.4f} | Result: {result['reason']}")
    
    # Test 2: Low Alignment Scenario
    score_low_a = rail.calculate_hs_score(0.2, 0.2, 0.5)
    result_low_a = rail.check_action("send_email", Reversibility.REVERSIBLE, score_low_a, 0.5)
    print(f"Low Alignment | Result: {result_low_a['reason']}")

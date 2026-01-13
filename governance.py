"""Legacy shim for governance module. Prefer src.tks.governance.governance."""

from src.tks.governance.governance import (
    GovernanceMode,
    Reversibility,
    ClearanceToken,
    GovernanceRail,
)

__all__ = [
    "GovernanceMode",
    "Reversibility",
    "ClearanceToken",
    "GovernanceRail",
]

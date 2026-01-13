"""Legacy shim for regulators module. Prefer src.tks.regulators.regulators."""

from src.tks.regulators.regulators import TKSNode, TKSRegulators, RPMPlanner

__all__ = [
    "TKSNode",
    "TKSRegulators",
    "RPMPlanner",
]

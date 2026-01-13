"""Verifier exports and minimal verification logic."""

from typing import Any, Dict, List

from ..config import OperationalMode, VerifierConfig
from .normalize import TKSNormalizer


class Verifier:
    """Minimal verifier using the TKS normalizer and basic evidence checks."""

    def __init__(self, config: VerifierConfig) -> None:
        self.config = config
        self.normalizer = TKSNormalizer()

    def verify(
        self,
        tool_results: List[Dict[str, Any]],
        context: Dict[str, Any],
        mode: OperationalMode,
    ) -> Dict[str, Any]:
        evidence_ok = any(
            isinstance(r, dict) and r.get("evidence_id") for r in tool_results
        )
        mode_name = mode.name if hasattr(mode, "name") else str(mode)
        if mode_name in {"HIGH_STAKES", "CRITICAL"} and self.config.evidence_required_in_hs:
            if not evidence_ok:
                return {
                    "status": "FAIL",
                    "gate": "BLOCK",
                    "reason": "Evidence required in elevated mode",
                }

        expressions = []
        for result in tool_results:
            if not isinstance(result, dict):
                continue
            for key in ("tks_expression", "expression", "equation", "output"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    expressions.append(value.strip())

        if not expressions:
            return {"status": "PASS", "gate": "ALLOW", "normalized": []}

        errors = []
        normalized = []
        for expr in expressions:
            norm = self.normalizer.normalize(expr)
            if norm.is_valid:
                normalized.append(norm.normalized)
            else:
                errors.append({"expression": expr, "error": norm.error})

        if errors:
            gate = "BLOCK" if self.config.strict_mode else "ALLOW"
            return {
                "status": "FAIL",
                "gate": gate,
                "errors": errors,
            }

        return {"status": "PASS", "gate": "ALLOW", "normalized": normalized}


__all__ = ["Verifier"]

"""Minimal executor stub for tool calls with RTTA-C reasoning support."""
from typing import Any, Dict, Optional
from uuid import uuid4
import re


class Executor:
    """Executes tool calls using a minimal placeholder implementation."""

    def __init__(self) -> None:
        self._reasoner = None

    def _load_reasoner(self):
        if self._reasoner is not None:
            return self._reasoner
        try:
            from tks_features.rtta_canonical import RTTACanonical
        except Exception:
            return None
        self._reasoner = RTTACanonical()
        return self._reasoner

    def _extract_question(self, params: Dict[str, Any]) -> Optional[str]:
        for key in ("question", "goal", "prompt", "task"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        description = params.get("description")
        if isinstance(description, str) and description.strip():
            match = re.match(
                r"(?i)\s*(gather info for|compute|solve|answer|task)\s*[:\-]?\s*(.+)",
                description.strip(),
            )
            if match:
                return match.group(2).strip()
            return description.strip()
        return None

    def execute(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        question = self._extract_question(params)
        if question:
            reasoner = self._load_reasoner()
            if reasoner:
                try:
                    result = reasoner.reason(question)
                    answer = result.get("answer")
                    derived = (result.get("evidence") or {}).get("derived", {})
                    payload = {
                        "answer": answer,
                        "problem_type": derived.get("problem_type"),
                        "derived": derived,
                        "iterations": result.get("iterations"),
                        "reasoner": "rtta_canonical",
                    }
                    evidence_id = params.get("evidence_id")
                    if evidence_id is None and answer and answer not in {
                        "Unable to determine",
                        "Cannot determine (unrecognized problem type)",
                    }:
                        evidence_id = f"ev_{uuid4().hex[:8]}"
                    return {
                        "tool_name": tool_name,
                        "params": params,
                        "status": "ok",
                        "output": payload,
                        "evidence_id": evidence_id,
                    }
                except Exception as exc:
                    return {
                        "tool_name": tool_name,
                        "params": params,
                        "status": "error",
                        "output": {"error": str(exc), "reasoner": "rtta_canonical"},
                        "evidence_id": params.get("evidence_id"),
                    }
        return {
            "tool_name": tool_name,
            "params": params,
            "status": "ok",
            "output": None,
            "evidence_id": params.get("evidence_id"),
        }

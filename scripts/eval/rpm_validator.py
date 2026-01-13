"""
RPM path validation helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Set

from tks.verifier.normalize import TKSNormalizer
from tks.verifier.parser import FoundationStack, WorldNoetic

_PATH_RE = re.compile(
    r"[A-D](?:10|[1-9])(?:\s*:\s*[A-D](?:10|[1-9]))+",
    re.IGNORECASE,
)


@dataclass
class RPMPathValidationResult:
    text: str
    candidate: str
    normalized: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    stack_len: int = 0


class RPMPathValidator:
    def __init__(self, *, min_len: int = 2, max_len: int = 3):
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self._normalizer = TKSNormalizer()

    def extract_candidate(self, text: str) -> Optional[str]:
        if not text:
            return None
        matches = [m.group(0) for m in _PATH_RE.finditer(text)]
        if not matches:
            return None
        valid = []
        for candidate in matches:
            parts = [p for p in re.split(r"\s*:\s*", candidate) if p]
            if self.min_len <= len(parts) <= self.max_len:
                valid.append(candidate)
        pool = valid or matches
        return max(pool, key=len)

    @staticmethod
    def _strip_wrapping_parens(candidate: str) -> str:
        text = candidate.strip()
        if not (text.startswith("(") and text.endswith(")")):
            return candidate
        inner = text[1:-1].strip()
        if inner.count("(") or inner.count(")"):
            return candidate
        return inner

    def validate(self, text: str, *, allow_extract: bool = True) -> RPMPathValidationResult:
        raw = text or ""
        candidate = raw.strip()
        if allow_extract:
            extracted = self.extract_candidate(raw)
            if extracted:
                candidate = extracted
        candidate = self._strip_wrapping_parens(candidate)

        if not candidate:
            return RPMPathValidationResult(
                text=raw,
                candidate="",
                normalized="",
                is_valid=False,
                errors=["no rpm path candidate found"],
            )

        norm = self._normalizer.normalize(candidate)
        if not norm.is_valid:
            return RPMPathValidationResult(
                text=raw,
                candidate=candidate,
                normalized=norm.normalized,
                is_valid=False,
                errors=[norm.error or "normalization failed"],
            )

        if not isinstance(norm.ast, FoundationStack):
            return RPMPathValidationResult(
                text=raw,
                candidate=candidate,
                normalized=norm.normalized,
                is_valid=False,
                errors=["expected colon stack"],
            )

        stack_len = len(norm.ast.elements)
        errors: List[str] = []
        if stack_len < self.min_len or stack_len > self.max_len:
            errors.append(
                f"stack length {stack_len} outside [{self.min_len}, {self.max_len}]"
            )

        for element in norm.ast.elements:
            if not isinstance(element, WorldNoetic):
                errors.append("invalid element type")
                break
            if element.world not in {"A", "B", "C", "D"}:
                errors.append(f"invalid world {element.world}")
            if not 0 <= element.noetic <= 10:
                errors.append(f"invalid noetic {element.noetic}")

        return RPMPathValidationResult(
            text=raw,
            candidate=candidate,
            normalized=norm.normalized,
            is_valid=len(errors) == 0,
            errors=errors,
            stack_len=stack_len,
        )

    def is_valid(self, text: str, *, allow_extract: bool = True) -> bool:
        return self.validate(text, allow_extract=allow_extract).is_valid


def build_rpm_token_filter(tokenizer, *, extra_allowed: str = "") -> Set[int]:
    allowed_chars = set("ABCD0123456789: ")
    if extra_allowed:
        allowed_chars.update(extra_allowed)

    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = getattr(tokenizer, "vocab", None)
        if vocab is None:
            vocab = getattr(tokenizer, "token_to_id", None)
        if vocab is None:
            return set()

    allowed_ids: Set[int] = set()
    for _token, tid in vocab.items():
        decoded = ""
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            if hasattr(tokenizer, "id_to_token"):
                decoded = tokenizer.id_to_token.get(int(tid), "")
        if not decoded:
            continue
        if all(char in allowed_chars for char in decoded):
            allowed_ids.add(int(tid))

    return allowed_ids

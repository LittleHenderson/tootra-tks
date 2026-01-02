#!/usr/bin/env python3
"""
Shared generation metrics for Story↔Equation decoding experiments.

Focus metrics for the "0% exact match" issue:
- Exact match (normalized)
- Element F1 (multiset)
- Operator F1 (multiset)
- String similarity
- Canonical/syntax validity (CanonicalValidator)
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple

from teacher.validator import CanonicalValidator


_ELEMENT_RE = re.compile(r'[ABCD](?:10|[1-9])(?:\^[1-9])?(?:_d[1-7])?')

# Order matters: match multi-char operators first.
_OP_RE = re.compile(r'(\+T|\-T|\*T|/T|->|<-|\+|\-|o)')


def normalize_equation(eq: str) -> str:
    """
    Normalize equations for exact-match comparison.

    - Strip leading/trailing whitespace
    - Remove ALL whitespace
    - Uppercase
    """
    if eq is None:
        return ""
    return re.sub(r'\s+', '', eq.strip()).upper()


def extract_elements(eq: str) -> List[str]:
    if not eq:
        return []
    return _ELEMENT_RE.findall(eq.upper())


def extract_operators(eq: str) -> List[str]:
    if not eq:
        return []
    return _OP_RE.findall(eq)


def _multiset_f1(pred: Iterable[str], gold: Iterable[str]) -> float:
    pred_c = Counter(pred)
    gold_c = Counter(gold)
    tp = sum(min(pred_c[k], gold_c[k]) for k in pred_c.keys() | gold_c.keys())
    pred_n = sum(pred_c.values())
    gold_n = sum(gold_c.values())
    if pred_n == 0 and gold_n == 0:
        return 1.0
    if pred_n == 0 or gold_n == 0:
        return 0.0
    precision = tp / pred_n
    recall = tp / gold_n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").strip(), (b or "").strip()).ratio()


class CanonValidator:
    def __init__(self):
        self._validator = CanonicalValidator(strict_mode=True)

    def is_valid(self, equation: str) -> bool:
        try:
            return bool(self._validator.validate(equation).is_valid)
        except Exception:
            return False


def compute_generation_metrics(
    predictions: List[str],
    targets: List[str],
    *,
    validator: Optional[CanonValidator] = None,
) -> Dict:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have same length")
    if validator is None:
        validator = CanonValidator()

    exact = 0
    element_f1s: List[float] = []
    op_f1s: List[float] = []
    sims: List[float] = []
    valid = 0

    for pred, tgt in zip(predictions, targets):
        if normalize_equation(pred) == normalize_equation(tgt):
            exact += 1

        element_f1s.append(_multiset_f1(extract_elements(pred), extract_elements(tgt)))
        op_f1s.append(_multiset_f1(extract_operators(pred), extract_operators(tgt)))
        sims.append(string_similarity(pred, tgt))
        valid += 1 if validator.is_valid(pred) else 0

    n = max(len(predictions), 1)
    return {
        "exact_match_rate": exact / n,
        "exact_match_count": exact,
        "total_examples": len(predictions),
        "avg_element_f1": sum(element_f1s) / n,
        "avg_operator_f1": sum(op_f1s) / n,
        "avg_string_similarity": sum(sims) / n,
        "syntax_valid_rate": valid / n,
    }


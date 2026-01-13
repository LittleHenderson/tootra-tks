"""
Shared evaluation helpers for grounding text and TKS expression metrics.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tks.verifier.normalize import normalize_tks
from src.tks.verifier.parser import (  # noqa: E402
    AtomicExpression,
    BinaryExpression,
    FoundationStack,
    GroupExpression,
    TKSExpression,
)


@dataclass(frozen=True)
class TksMatch:
    syntax_valid: bool
    structure_match: bool
    id_exact: bool
    full_exact: bool
    pred_normalized: str = ""
    gold_normalized: str = ""
    pred_canonical: str = ""
    gold_canonical: str = ""


def normalize_text(text: str) -> str:
    """Normalize grounding text for comparison."""
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(":", " : ").replace("/", " / ").replace(",", " , ")
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def partial_text_match(pred: str, gold: str, *, window: int = 20) -> bool:
    """Loose substring match for grounding text."""
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return False
    prefix_len = min(window, len(pred_norm), len(gold_norm))
    return pred_norm[:prefix_len] in gold_norm or gold_norm[:prefix_len] in pred_norm


def _shape_signature(expr: TKSExpression) -> Tuple:
    """Return an AST-only signature (ignore element identities)."""
    if isinstance(expr, AtomicExpression):
        return ("atom",)
    if isinstance(expr, GroupExpression):
        return ("group", _shape_signature(expr.inner))
    if isinstance(expr, FoundationStack):
        return ("stack", len(expr.elements))
    if isinstance(expr, BinaryExpression):
        return ("binary", _shape_signature(expr.left), _shape_signature(expr.right))
    return ("unknown", type(expr).__name__)


def compare_tks_expressions(pred: str, gold: str) -> TksMatch:
    """Compare two TKS expressions using the canonical normalizer."""
    pred_text = pred or ""
    gold_text = gold or ""
    pred_result = normalize_tks(pred_text)
    gold_result = normalize_tks(gold_text)

    if not pred_result.is_valid or not gold_result.is_valid:
        return TksMatch(
            syntax_valid=bool(pred_result.is_valid),
            structure_match=False,
            id_exact=False,
            full_exact=False,
            pred_normalized=pred_result.normalized,
            gold_normalized=gold_result.normalized,
            pred_canonical=pred_result.canonical_form,
            gold_canonical=gold_result.canonical_form,
        )

    pred_shape = _shape_signature(pred_result.ast)
    gold_shape = _shape_signature(gold_result.ast)

    return TksMatch(
        syntax_valid=True,
        structure_match=pred_shape == gold_shape,
        id_exact=pred_result.canonical_form == gold_result.canonical_form,
        full_exact=pred_result.normalized == gold_result.normalized,
        pred_normalized=pred_result.normalized,
        gold_normalized=gold_result.normalized,
        pred_canonical=pred_result.canonical_form,
        gold_canonical=gold_result.canonical_form,
    )


def compute_tks_metrics(predictions: Iterable[str], targets: Iterable[str]) -> Dict[str, float]:
    """Aggregate TKS comparison metrics over a dataset."""
    pred_list = list(predictions)
    target_list = list(targets)
    if len(pred_list) != len(target_list):
        raise ValueError("predictions and targets must have same length")

    totals = {
        "syntax_valid": 0,
        "structure_match": 0,
        "id_exact": 0,
        "full_exact": 0,
    }
    for pred, gold in zip(pred_list, target_list):
        result = compare_tks_expressions(pred, gold)
        totals["syntax_valid"] += int(result.syntax_valid)
        totals["structure_match"] += int(result.structure_match)
        totals["id_exact"] += int(result.id_exact)
        totals["full_exact"] += int(result.full_exact)

    n = max(len(pred_list), 1)
    return {
        "syntax_valid_rate": totals["syntax_valid"] / n,
        "structure_match_rate": totals["structure_match"] / n,
        "id_exact_rate": totals["id_exact"] / n,
        "full_exact_rate": totals["full_exact"] / n,
        "total_examples": len(pred_list),
    }

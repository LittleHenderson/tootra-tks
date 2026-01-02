#!/usr/bin/env python3
"""
Audit the repo's custom Seq2SeqTokenizer for Story↔Equation tasks.

This tokenizer is currently character-level; this audit quantifies how much
it splits semantic units like:
- Elements: A1..D10 (and extended A1^3_d5)
- Operators: +T, -T, ->, <-, *T, /T

Output is JSON with findings + recommendations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import eval_story_equation as ese  # noqa: E402


TKS_ELEMENTS = [f"{w}{n}" for w in "ABCD" for n in range(1, 11)]
TKS_OPERATORS = ["+", "-", "o", "+T", "-T", "*T", "/T", "->", "<-"]

TEST_EQUATIONS = [
    "D10 o C1 + A3 -T B7^2_d4",
    "A1 + B2 - C3",
    "D9 o D8 o D7",
    "C10^4 - D1_d7",
    "A1 +T B2 -> C3 /T D4",
]

_ELEM_RE = re.compile(r"[ABCD](?:10|[1-9])(?:\^[1-9])?(?:_d[1-7])?")
_OP_RE = re.compile(r"(\+T|\-T|\*T|/T|->|<-|\+|\-|o)")


def _trim_pad(ids: List[int], pad_id: int = 0) -> List[int]:
    try:
        pad_at = ids.index(pad_id)
    except ValueError:
        pad_at = len(ids)
    return ids[:pad_at]


def count_semantic_units(equation: str) -> int:
    elements = _ELEM_RE.findall(equation)
    ops = _OP_RE.findall(equation)
    # Count numeric modifiers (sense/foundation) as part of element, not separate units.
    return len(elements) + len(ops)


def audit(tokenizer: ese.Seq2SeqTokenizer) -> Dict:
    pad, bos, eos = 0, 2, 3

    element_as_single = 0
    split_elements = []
    element_analysis = {}

    for element in TKS_ELEMENTS:
        ids = tokenizer.tokenize(element, add_bos=True, add_eos=True)
        ids = _trim_pad(ids, pad_id=pad)
        core = [i for i in ids if i not in {bos, eos}]
        is_single = len(core) == 1
        if is_single:
            element_as_single += 1
        else:
            split_elements.append({"element": element, "tokens": [tokenizer.id_to_token.get(i, "?") for i in core]})
        element_analysis[element] = {"token_count": len(core), "is_single_token": is_single}

    operator_as_single = 0
    split_ops = []
    operator_analysis = {}
    for op in TKS_OPERATORS:
        ids = tokenizer.tokenize(op, add_bos=True, add_eos=True)
        ids = _trim_pad(ids, pad_id=pad)
        core = [i for i in ids if i not in {bos, eos}]
        is_single = len(core) == 1
        if is_single:
            operator_as_single += 1
        else:
            split_ops.append({"operator": op, "tokens": [tokenizer.id_to_token.get(i, "?") for i in core]})
        operator_analysis[op] = {"token_count": len(core), "is_single_token": is_single}

    equation_analysis = []
    for eq in TEST_EQUATIONS:
        ids = tokenizer.tokenize(eq, add_bos=True, add_eos=True)
        ids = _trim_pad(ids, pad_id=pad)
        core = [i for i in ids if i not in {bos, eos}]
        expected = count_semantic_units(eq)
        equation_analysis.append(
            {
                "equation": eq,
                "token_count": len(core),
                "expected_units": expected,
                "expansion_ratio": (len(core) / expected) if expected else None,
            }
        )

    avg_expansion = (
        sum(a["expansion_ratio"] for a in equation_analysis if a["expansion_ratio"] is not None) / len(equation_analysis)
        if equation_analysis
        else None
    )

    issues = []
    if split_elements:
        issues.append({"type": "split_elements", "count": len(split_elements), "examples": split_elements[:10]})
    if split_ops:
        issues.append({"type": "split_operators", "count": len(split_ops), "examples": split_ops[:10]})

    recommendations = []
    if element_as_single < len(TKS_ELEMENTS):
        recommendations.append(
            {
                "priority": "HIGH",
                "action": "implement_mixed_tokenizer_and_retrain",
                "reason": f"{len(TKS_ELEMENTS) - element_as_single} / {len(TKS_ELEMENTS)} elements are split into characters",
                "details": "Current Seq2SeqTokenizer.tokenize() is character-level; element/operator tokens exist in vocab but are unused.",
            }
        )
    if avg_expansion and avg_expansion > 2.0:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "action": "reduce_sequence_length",
                "reason": f"Average equation expansion ratio is {avg_expansion:.2f}",
                "details": "Shorter sequences improve decoding stability and exact match.",
            }
        )

    severity = "NONE"
    if element_as_single == 0:
        severity = "HIGH"
    elif element_as_single < len(TKS_ELEMENTS) * 0.8:
        severity = "MEDIUM"

    return {
        "vocab_size": tokenizer.actual_vocab_size,
        "max_length": tokenizer.max_length,
        "findings": {
            "elements_as_single_token": element_as_single,
            "total_elements_tested": len(TKS_ELEMENTS),
            "operators_as_single_token": operator_as_single,
            "total_operators_tested": len(TKS_OPERATORS),
            "average_expansion_ratio": avg_expansion,
            "split_element_examples": split_elements[:10],
            "split_operator_examples": split_ops[:10],
        },
        "issues_found": issues,
        "recommendations": recommendations,
        "severity": severity,
        "element_analysis": element_analysis,
        "operator_analysis": operator_analysis,
        "equation_analysis": equation_analysis,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit custom StoryEQ tokenizer")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json from a checkpoint dir")
    parser.add_argument("--tokenizer-mode", choices=["char", "mixed"], default=None, help="Override tokenizer mode")
    parser.add_argument("--output", required=True, help="Output JSON path for audit report")
    args = parser.parse_args()

    tok = ese.Seq2SeqTokenizer.from_json(Path(args.tokenizer))
    if args.tokenizer_mode:
        tok.mode = args.tokenizer_mode.lower()
        tok._rebuild_multi_token_index()
    report = audit(tok)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved audit report to: {out}")
    print(f"Severity: {report['severity']}")
    print(f"Elements as single token: {report['findings']['elements_as_single_token']}/{report['findings']['total_elements_tested']}")
    print(f"Operators as single token: {report['findings']['operators_as_single_token']}/{report['findings']['total_operators_tested']}")
    print(f"Avg expansion ratio: {report['findings']['average_expansion_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

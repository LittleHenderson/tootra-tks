#!/usr/bin/env python3
"""
Create an RPM-specific holdout set with novel prompt templates.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple


RPM_INPUT_PATTERNS = [
    re.compile(r"^Goal:\s*(.+)$", re.IGNORECASE),
    re.compile(r"^RPM Cascade for\s*(.+)$", re.IGNORECASE),
    re.compile(r"^Deep Plan:\s*I want\s*'([^']+)'$", re.IGNORECASE),
    re.compile(r"^Deep Plan:\s*I want\s*(.+)$", re.IGNORECASE),
]

RPM_HOLDOUT_TEMPLATES = [
    "Construct an RPM stack for {target} (format A1:B2:C3).",
    "Provide the RPM cascade path for {target}.",
    "Give a colon-separated RPM sequence for {target}.",
    "Resolve the RPM path for {target} using A/B/C/D elements.",
    "Return RPM chain for {target} as a colon stack.",
    "What is the RPM path (A#:B#:C#) for {target}?",
]


def load_jsonl(path: Path) -> List[dict]:
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def extract_target(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.strip()
    for pattern in RPM_INPUT_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            return match.group(1).strip().rstrip(".")
    return None


def build_holdout(
    items: List[dict],
    *,
    rng: random.Random,
    count: int,
) -> List[dict]:
    pool: List[Tuple[str, str]] = []
    for item in items:
        if item.get("task_type") != "rpm":
            continue
        target = extract_target(str(item.get("input", "")))
        if not target:
            continue
        output = str(item.get("output", "")).strip()
        if not output:
            continue
        pool.append((target, output))

    rng.shuffle(pool)
    seen_prompts = set()
    holdout = []

    for target, output in pool:
        templates = list(RPM_HOLDOUT_TEMPLATES)
        rng.shuffle(templates)
        for template in templates:
            prompt = template.format(target=target)
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            holdout.append(
                {
                    "input": prompt,
                    "output": output,
                    "task_type": "rpm_holdout",
                    "source": "holdout_rpm_v5",
                }
            )
            if len(holdout) >= count:
                return holdout

    return holdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RPM holdout dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="output/final_reconciled_mix.jsonl",
        help="Source dataset for RPM targets",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of holdout items to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/holdout_rpm_v5.jsonl",
        help="Holdout output path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    items = load_jsonl(input_path)
    if not items:
        raise SystemExit(f"No items found in {input_path}")

    rng = random.Random(args.seed)
    holdout = build_holdout(items, rng=rng, count=args.count)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in holdout:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"Generated {len(holdout)} RPM holdout items -> {output_path}")


if __name__ == "__main__":
    main()

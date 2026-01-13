"""
Generate a cleaned repair dataset to break looping attractors.
Filters outputs that contain "Trace:" or repeated punctuation patterns.
"""
import argparse
import json
import os
import random
import re
from typing import Iterable, Optional


BAD_PATTERNS = [
    re.compile(r"Trace:\s*", re.IGNORECASE),
    re.compile(r"Input:\s*", re.IGNORECASE),
    re.compile(r"Output:\s*", re.IGNORECASE),
    re.compile(r"(:\s*){3,}"),
    re.compile(r"(->\s*){3,}"),
    re.compile(r"(\b\w+\b)(\s+\1){3,}", re.IGNORECASE),
]


def is_bad(text: str) -> bool:
    if not text:
        return True
    if len(text) < 10:
        return True
    for pattern in BAD_PATTERNS:
        if pattern.search(text):
            return True
    return False


def iter_records(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_output(record: dict) -> Optional[str]:
    for key in ("output", "response", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_dataset(input_path: str, output_path: str, max_samples: int, seed: int) -> None:
    rng = random.Random(seed)
    seen = set()
    rows = []

    for record in iter_records(input_path):
        output = extract_output(record)
        if output is None or is_bad(output):
            continue
        if output in seen:
            continue
        seen.add(output)
        rows.append({"output": output})
        if len(rows) >= max_samples:
            break

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/v7_nl_bridge_training.jsonl")
    parser.add_argument("--output", default="data/v7_repair_training.jsonl")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build_dataset(args.input, args.output, args.max_samples, args.seed)

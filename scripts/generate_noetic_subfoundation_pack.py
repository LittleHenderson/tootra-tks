#!/usr/bin/env python3
"""
Generate a dedicated noetic/subfoundation classification pack.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def make_task(task_type: str, input_text: str, output_text: str) -> Dict:
    return {
        "input": input_text,
        "output": output_text,
        "task_type": task_type,
        "source": "noetic_subfoundation_pack_v1",
    }


def pick_distractors(rng: random.Random, pool: List[str], exclude: set, k: int) -> List[str]:
    candidates = [p for p in pool if p not in exclude]
    rng.shuffle(candidates)
    return candidates[:k]


NOETIC_CONFUSIONS = {
    "^0": ["^1"],
    "^1": ["^0", "^2"],
    "^2": ["^1", "^3"],
    "^3": ["^2", "^4"],
    "^5": ["^6"],
    "^6": ["^5", "^7"],
    "^7": ["^6"],
    "^8": ["^9"],
    "^9": ["^8"],
}


NOETIC_TEMPLATES = [
    "Select the noetic ID for: {name}: {meaning}",
    "Which noetic matches: {meaning}?",
    "Noetic ID for '{name}'?",
    "Which noetic matches: {name}?",
]

NOETIC_MCQ_TEMPLATES = [
    "Choose the noetic ID for: {meaning}\nOptions: {options}",
    "Pick the noetic for: {name}: {meaning}\nOptions: {options}",
]

SUBFOUNDATION_TEMPLATES = [
    "Select the sub-foundation ID for: {name}",
    "Which sub-foundation matches: {name}?",
    "Sub-foundation ID for '{name}'?",
]

SUBFOUNDATION_MCQ_TEMPLATES = [
    "Choose the sub-foundation ID for: {name}\nOptions: {options}",
]


def generate_noetic_items(
    rng: random.Random,
    noetics: List[Dict],
    multiplier: int,
) -> List[Dict]:
    items: List[Dict] = []
    all_ids = [n["id"] for n in noetics]
    multiplier = max(int(multiplier), 1)
    for _ in range(multiplier):
        for n in noetics:
            for template in NOETIC_TEMPLATES:
                items.append(
                    make_task(
                        "noetic_classify",
                        template.format(name=n["name"], meaning=n["meaning"]),
                        n["id"],
                    )
                )
            for template in NOETIC_MCQ_TEMPLATES:
                distractors = []
                for candidate in NOETIC_CONFUSIONS.get(n["id"], []):
                    distractors.append(candidate)
                distractors.extend(
                    pick_distractors(rng, all_ids, {n["id"]} | set(distractors), 3)
                )
                options = [n["id"]] + distractors[:3]
                rng.shuffle(options)
                items.append(
                    make_task(
                        "noetic_classify",
                        template.format(
                            name=n["name"],
                            meaning=n["meaning"],
                            options=", ".join(options),
                        ),
                        n["id"],
                    )
                )
    return items


def generate_subfoundation_items(
    rng: random.Random,
    subfoundations: List[Dict],
    multiplier: int,
) -> List[Dict]:
    items: List[Dict] = []
    multiplier = max(int(multiplier), 1)
    by_foundation: Dict[str, List[str]] = defaultdict(list)
    by_world: Dict[str, List[str]] = defaultdict(list)
    for sf in subfoundations:
        by_foundation[sf["foundation_id"]].append(sf["id"])
        by_world[sf["world"]].append(sf["id"])

    all_ids = [sf["id"] for sf in subfoundations]
    for _ in range(multiplier):
        for sf in subfoundations:
            for template in SUBFOUNDATION_TEMPLATES:
                items.append(
                    make_task(
                        "subfoundation_classify",
                        template.format(name=sf["name"]),
                        sf["id"],
                    )
                )
            for template in SUBFOUNDATION_MCQ_TEMPLATES:
                distractors = []
                distractors.extend(
                    pick_distractors(
                        rng,
                        by_foundation[sf["foundation_id"]],
                        {sf["id"]},
                        1,
                    )
                )
                distractors.extend(
                    pick_distractors(rng, by_world[sf["world"]], {sf["id"]}, 1)
                )
                distractors.extend(
                    pick_distractors(
                        rng,
                        all_ids,
                        {sf["id"]} | set(distractors),
                        3,
                    )
                )
                options = [sf["id"]] + distractors[:3]
                rng.shuffle(options)
                items.append(
                    make_task(
                        "subfoundation_classify",
                        template.format(name=sf["name"], options=", ".join(options)),
                        sf["id"],
                    )
                )
    return items


def merge_with_mix(
    mix_items: List[Dict],
    pack_items: List[Dict],
) -> Dict[str, int | List[Dict]]:
    output: List[Dict] = list(mix_items)
    outputs_by_input: Dict[str, set] = defaultdict(set)
    for item in mix_items:
        input_key = normalize_text(item.get("input", ""))
        output_key = normalize_text(item.get("output", ""))
        if input_key:
            outputs_by_input[input_key].add(output_key)

    added = 0
    conflicts = 0
    for item in pack_items:
        input_key = normalize_text(item.get("input", ""))
        output_key = normalize_text(item.get("output", ""))
        if not input_key:
            continue
        if input_key in outputs_by_input and output_key not in outputs_by_input[input_key]:
            conflicts += 1
            continue
        output.append(item)
        outputs_by_input[input_key].add(output_key)
        added += 1

    return {"items": output, "conflicts": conflicts, "added": added}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noetic/subfoundation pack")
    parser.add_argument("--canon-dir", default="canon/normalized", help="Canon directory")
    parser.add_argument(
        "--output",
        default="output/noetic_subfoundation_pack_v1.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--noetic-multiplier",
        type=int,
        default=1,
        help="Repeat noetic items",
    )
    parser.add_argument(
        "--subfoundation-multiplier",
        type=int,
        default=1,
        help="Repeat subfoundation items",
    )
    parser.add_argument(
        "--mix-input",
        default="",
        help="Optional mix input JSONL to merge with pack",
    )
    parser.add_argument(
        "--mix-output",
        default="",
        help="Optional mix output JSONL path",
    )
    args = parser.parse_args()

    canon_dir = Path(args.canon_dir)
    noetics = load_json(canon_dir / "canon_noetics_10.json")
    subfoundations = load_json(canon_dir / "canon_subfoundations_28.json")

    rng = random.Random(args.seed)
    pack_items: List[Dict] = []
    pack_items.extend(
        generate_noetic_items(rng, noetics, multiplier=args.noetic_multiplier)
    )
    pack_items.extend(
        generate_subfoundation_items(
            rng,
            subfoundations,
            multiplier=args.subfoundation_multiplier,
        )
    )
    rng.shuffle(pack_items)
    output_path = Path(args.output)
    write_jsonl(output_path, pack_items)
    print(f"Wrote pack: {output_path} ({len(pack_items)} items)")

    counts = Counter(item["task_type"] for item in pack_items)
    for key, value in counts.items():
        print(f"  {key}: {value}")

    if args.mix_input and args.mix_output:
        mix_items = load_jsonl(Path(args.mix_input))
        merged = merge_with_mix(mix_items, pack_items)
        mix_output = Path(args.mix_output)
        write_jsonl(mix_output, merged["items"])
        print(f"Wrote merged mix: {mix_output} ({len(merged['items'])} items)")
        print(f"  added_from_pack: {merged['added']}")
        print(f"  conflicts_dropped: {merged['conflicts']}")


if __name__ == "__main__":
    main()

"""
Create a v5 holdout set with novel templates and paraphrases.
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from teacher.validator import CanonicalValidator  # noqa: E402

RULES_DIR = ROOT_DIR / "tks_rules"


def _load_rule_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_noetics = _load_rule_module(RULES_DIR / "noetics.py", "tks_rules_noetics")
_operators = _load_rule_module(RULES_DIR / "operators.py", "tks_rules_operators")
_worlds = _load_rule_module(RULES_DIR / "worlds.py", "tks_rules_worlds")

SENSES = _noetics.SENSES
OPERATORS = _operators.OPERATORS
WORLD_LETTERS = _worlds.WORLD_LETTERS

# Canonical element notation uses noetic numbers 1-10 (A1..D10)
NOETIC_NUMBERS = list(range(1, 11))
VALID_SENSES = SENSES
VALID_FOUNDATIONS = ["_d1", "_d2", "_d3", "_d4", "_d5", "_d6", "_d7"]

# Use the 9-operator subset used across this repo's training data (exclude * and /)
CANONICAL_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]

ELEMENT_PATTERN = re.compile(r"^([ABCD])(10|[1-9])(\^[1-9])?(_d[1-7])?$")
ELEMENT_SPLIT = re.compile(r"^([ABCD])(10|[1-9])(?:\^([1-9]))?(?:(_d[1-7]))?$")

OPERATOR_PHRASES = {
    "+": ["aligns with", "pairs with", "joins forces with"],
    "-": ["pulls away from", "separates from", "cancels against"],
    "+T": ["builds alongside", "accumulates with", "grows together with"],
    "-T": ["erodes from", "diminishes alongside", "drains from"],
    "*T": ["intensifies through", "amplifies with", "strengthens by"],
    "/T": ["tensions against", "dilutes through", "checks against"],
    "->": ["leads into", "drives toward", "results in"],
    "<-": ["arises from", "draws from", "is triggered by"],
    "o": ["follows from", "composes after", "runs after"],
}

OPENINGS = [
    "Track this relation:",
    "Map this sequence:",
    "In this chain,",
    "Observe the linkage:",
    "Follow the path:",
]

CLOSINGS = [
    ".",
    " to complete the chain.",
    " as the sequence resolves.",
    " in this working.",
]

CONNECTORS = [
    "; then",
    "; next",
    "; afterward",
    "; in turn",
]

SINGLE_TEMPLATES = [
    "Focus on {elem} alone.",
    "Only {elem} is active here.",
    "{elem} stands by itself in this case.",
]

QUESTION_TEMPLATES = [
    "How does {chain}?",
    "What sequence captures {chain}?",
    "Which chain best matches {chain}?",
]

INSTRUCTION_TEMPLATES = [
    "Trace the chain: {chain}.",
    "Describe the flow: {chain}.",
    "Outline the steps: {chain}.",
]


def generate_element(rng: random.Random, use_extended: bool = False) -> str:
    """Generate a canonical element."""
    world = rng.choice(WORLD_LETTERS)
    noetic = rng.choice(NOETIC_NUMBERS)
    element = f"{world}{noetic}"
    if use_extended and rng.random() < 0.4:
        element += f"^{rng.choice(VALID_SENSES)}"
    if use_extended and rng.random() < 0.3:
        element += rng.choice(VALID_FOUNDATIONS)
    return element


def generate_equation(rng: random.Random, length: int) -> Tuple[str, List[str], List[str]]:
    """Generate a canonical equation."""
    use_extended = rng.random() < 0.5
    elements = [generate_element(rng, use_extended) for _ in range(length)]
    operators = [rng.choice(CANONICAL_OPERATORS) for _ in range(length - 1)]
    parts = [elements[0]]
    for i, op in enumerate(operators):
        parts.append(op)
        parts.append(elements[i + 1])
    equation = " ".join(parts)
    return equation, elements, operators


def validate_element(element: str) -> bool:
    """Validate element against canon."""
    return bool(ELEMENT_PATTERN.match(element))


def _element_phrase(element: str, rng: random.Random) -> str:
    match = ELEMENT_SPLIT.match(element)
    if not match:
        return element
    world, noetic, sense, foundation = match.groups()
    base_variants = [
        f"{world}{noetic}",
        f"element {world}{noetic}",
        f"World {world} noetic {noetic}",
        f"W{world}-{noetic}",
        f"world {world} / noetic {noetic}",
    ]
    phrase = rng.choice(base_variants)
    if sense:
        phrase += rng.choice(
            [
                f" with sense {sense}",
                f" (sense {sense})",
                f" at sense {sense}",
            ]
        )
    if foundation:
        phrase += rng.choice(
            [
                f" grounded {foundation}",
                f" with foundation {foundation}",
                f" anchored {foundation}",
            ]
        )
    return phrase


def _join_segments(segments: List[str], rng: random.Random) -> str:
    text = segments[0]
    for seg in segments[1:]:
        text += f"{rng.choice(CONNECTORS)} {seg}"
    return text


def build_story(elements: List[str], operators: List[str], rng: random.Random) -> Tuple[str, str]:
    """Generate a novel paraphrased story for a chain."""
    if not elements:
        return "Empty sequence.", "empty"
    if len(elements) == 1:
        elem = _element_phrase(elements[0], rng)
        return rng.choice(SINGLE_TEMPLATES).format(elem=elem), "single"

    elem_phrases = [_element_phrase(e, rng) for e in elements]
    segments = []
    for i, op in enumerate(operators):
        op_phrase = rng.choice(OPERATOR_PHRASES.get(op, ["relates to"]))
        segments.append(f"{elem_phrases[i]} {op_phrase} {elem_phrases[i + 1]}")

    chain = _join_segments(segments, rng)

    style = rng.choice(["narrative", "instruction", "question"])
    if style == "question":
        return rng.choice(QUESTION_TEMPLATES).format(chain=chain), style
    if style == "instruction":
        return rng.choice(INSTRUCTION_TEMPLATES).format(chain=chain), style

    story = f"{rng.choice(OPENINGS)} {chain}{rng.choice(CLOSINGS)}"
    return story, style


def create_holdout_v5(count: int, seed: int) -> List[Dict]:
    """Create v5 holdout set with new templates/paraphrases."""
    rng = random.Random(seed)
    validator = CanonicalValidator(strict_mode=True)
    items: List[Dict] = []
    seen_equations = set()

    target_short = int(count * 0.4)
    target_mid = int(count * 0.35)
    target_long = count - target_short - target_mid
    print(f"Generating v5 holdout: {target_short} short, {target_mid} mid, {target_long} long")

    def add_items(length_range: Tuple[int, int], target: int) -> int:
        added = 0
        attempts = 0
        max_attempts = target * 30
        while added < target and attempts < max_attempts:
            attempts += 1
            length = rng.randint(length_range[0], length_range[1])
            equation, elements, operators = generate_equation(rng, length)
            if equation in seen_equations:
                continue
            if not all(validate_element(e) for e in elements):
                continue
            try:
                if not validator.validate(equation).is_valid:
                    continue
            except Exception:
                continue

            story, style = build_story(elements, operators, rng)
            pair_id = f"holdout_v5_{len(items):06d}"
            item = {
                "story": story,
                "equation": equation,
                "expr_elements": elements,
                "expr_ops": operators,
                "format": style,
                "chain_length": length,
                "pair_id": pair_id,
                "aug_type": "holdout_v5",
            }
            items.append(item)
            seen_equations.add(equation)
            added += 1
        return added

    short_added = add_items((2, 3), target_short)
    mid_added = add_items((4, 4), target_mid)
    long_added = add_items((5, 6), target_long)
    print(f"Generated: {short_added} short, {mid_added} mid, {long_added} long")
    print(f"Total: {len(items)} items")
    rng.shuffle(items)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Create v5 holdout dataset")
    parser.add_argument("--count", type=int, default=200, help="Number of items")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/holdout_v5.jsonl", help="Output file")
    args = parser.parse_args()

    items = create_holdout_v5(args.count, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"\nSaved {len(items)} items to: {output_path}")
    format_dist: Dict[str, int] = {}
    length_dist: Dict[int, int] = {}
    for item in items:
        fmt = item.get("format", "unknown")
        format_dist[fmt] = format_dist.get(fmt, 0) + 1
        length = item.get("chain_length", 0)
        length_dist[length] = length_dist.get(length, 0) + 1

    print("\nFormat distribution:")
    for fmt, cnt in sorted(format_dist.items()):
        print(f"  {fmt}: {cnt} ({cnt / len(items) * 100:.1f}%)")

    print("\nChain length distribution:")
    for length, cnt in sorted(length_dist.items()):
        print(f"  {length} elements: {cnt} ({cnt / len(items) * 100:.1f}%)")

    print("\nSample items:")
    for i, item in enumerate(items[:3]):
        print(f"\nItem {i + 1} ({item.get('format')}, {item.get('chain_length')} elements):")
        print(f"  Story: {item['story'][:80]}...")
        print(f"  Equation: {item['equation']}")


if __name__ == "__main__":
    main()

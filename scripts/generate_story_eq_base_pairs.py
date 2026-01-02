#!/usr/bin/env python3
"""
Generate *base* story↔equation pairs (one JSON object per pair).

This is the scalable format for large datasets (100k–600k+) and is directly
consumed by `scripts/train_story_equation.py` and `scripts/eval_story_equation.py`.

Output format (JSONL; one base pair per line):
  {
    "pair_id": "pair_0000000",
    "story": "...",          # narrative ONLY (does not include equation text)
    "equation": "A1 + B2",
    "expr_elements": ["A1", "B2"],
    "expr_ops": ["+"],
    "canon_score": 1.0
  }

Notes:
- Enforces uniqueness by equation (and optionally story) via 64-bit digests.
- Canonical by construction (worlds A-D, noetics 1-10, senses ^1-^9, foundations _d1-_d7,
  and the 9 training operators + - +T -T -> <- *T /T o).
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_rules.foundations import FOUNDATIONS
from tks_rules.noetics import NOETICS, SENSES
from tks_rules.operators import OPERATORS
from tks_rules.worlds import WORLDS, WORLD_LETTERS


TRAINING_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]
FOUNDATION_NUMBERS = list(FOUNDATIONS.keys())
NOETIC_NUMBERS = list(range(1, 11))


def _digest64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def _noetic_name(noetic_number: int) -> str:
    # Element notation uses 1-10, but canonical noetics are 0-9 (0 == Idea).
    if noetic_number == 10:
        return NOETICS[0].name
    return NOETICS[noetic_number].name


def _foundation_name(f_idx: int) -> str:
    # Prefer metaphysical_name for the story.
    return FOUNDATIONS[f_idx].metaphysical_name


def generate_element(
    rng: random.Random,
    *,
    world: Optional[str] = None,
    noetic: Optional[int] = None,
    sense_prob: float = 0.25,
    foundation_prob: float = 0.20,
) -> Tuple[str, str, int, Optional[int], Optional[int]]:
    world = world or rng.choice(WORLD_LETTERS)
    noetic = noetic or rng.choice(NOETIC_NUMBERS)
    sense = rng.choice(SENSES) if rng.random() < sense_prob else None
    foundation = rng.choice(FOUNDATION_NUMBERS) if rng.random() < foundation_prob else None

    elem = f"{world}{noetic}"
    if sense is not None:
        elem += f"^{sense}"
    if foundation is not None:
        elem += f"_d{foundation}"
    return (elem, world, noetic, sense, foundation)


def element_phrase(world: str, noetic: int, sense: Optional[int], foundation: Optional[int], rng: random.Random) -> str:
    world_def = WORLDS[world]
    noetic_name = _noetic_name(noetic)

    base_templates = [
        f"{world_def.domain} {noetic_name}",
        f"{noetic_name} in the {world_def.domain.lower()} domain",
        f"{noetic_name} within {world_def.name}",
    ]
    phrase = rng.choice(base_templates)

    if sense is not None:
        phrase += rng.choice([f" (sense {sense})", f" at sense {sense}"])
    if foundation is not None:
        fname = _foundation_name(foundation)
        phrase += rng.choice([f" grounded in {fname}", f" anchored in {fname}"])

    return phrase


OPERATOR_PHRASES: Dict[str, List[str]] = {
    "+": ["combines with", "joins", "unites with", "merges with"],
    "-": ["separates from", "removes", "subtracts", "divides from"],
    "+T": ["accumulates with", "adds over time to", "builds with", "reinforces with"],
    "-T": ["diminishes from", "reduces over time from", "releases from", "strips away from"],
    "->": ["flows to", "becomes", "transforms into", "leads to"],
    "<-": ["receives from", "is sourced by", "draws from", "is caused by"],
    "*T": ["amplifies with", "intensifies through", "resonates with", "magnifies via"],
    "/T": ["attenuates through", "conflicts with", "dampens with", "moderates via"],
    "o": ["then composes with", "then cycles with", "then applies to", "then passes through"],
}


def operator_phrase(op: str, rng: random.Random) -> str:
    return rng.choice(OPERATOR_PHRASES.get(op, [op]))


def generate_equation(rng: random.Random, *, chain_length: int) -> Tuple[str, List[str], List[str], List[Tuple]]:
    elements_meta = [generate_element(rng) for _ in range(chain_length)]
    expr_elements = [e[0] for e in elements_meta]
    expr_ops = [rng.choice(TRAINING_OPERATORS) for _ in range(chain_length - 1)]

    parts: List[str] = [expr_elements[0]]
    for i, op in enumerate(expr_ops):
        parts.append(op)
        parts.append(expr_elements[i + 1])
    equation = " ".join(parts)
    return (equation, expr_elements, expr_ops, elements_meta)


def generate_story(elements_meta: List[Tuple], expr_ops: List[str], rng: random.Random) -> str:
    openings = [
        "In TKS,",
        "This working shows",
        "The pattern indicates",
        "The working describes",
        "A TKS sequence:",
    ]
    closings = [
        ".",
        ", completing the circuit.",
        ", stabilizing the working.",
        ", producing the intended effect.",
    ]

    phrases = []
    for elem, world, noetic, sense, foundation in elements_meta:
        phrases.append(element_phrase(world, noetic, sense, foundation, rng))

    story_parts: List[str] = [rng.choice(openings), phrases[0]]
    for i, op in enumerate(expr_ops):
        story_parts.append(operator_phrase(op, rng))
        story_parts.append(phrases[i + 1])

    return " ".join(story_parts) + rng.choice(closings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate base story-equation pairs (JSONL)")
    parser.add_argument("--count", type=int, default=100000, help="Number of base pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/story_eq_pairs_base.jsonl", help="Output JSONL path")
    parser.add_argument("--progress-every", type=int, default=10000, help="Progress print interval (0=off)")
    parser.add_argument("--max-attempts-mult", type=int, default=12, help="Stop after count*mult attempts")
    parser.add_argument("--dedup-story", action="store_true", help="Also enforce unique story text")
    parser.add_argument("--min-len", type=int, default=2, help="Min chain length")
    parser.add_argument("--max-len", type=int, default=6, help="Max chain length")
    args = parser.parse_args()

    if args.min_len < 2 or args.max_len < args.min_len:
        raise SystemExit("Invalid --min-len/--max-len (must be >=2 and max>=min)")

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_eq = set()
    seen_story = set() if args.dedup_story else None

    attempts = 0
    max_attempts = args.count * max(args.max_attempts_mult, 1)
    created = 0

    print(f"Generating {args.count:,} base pairs (seed={args.seed})")
    print(f"Output: {output_path}")
    print(f"Dedup: equation={'ON'} story={'ON' if args.dedup_story else 'OFF'}")
    print(f"Chain length: {args.min_len}-{args.max_len}")
    print(f"Operators: {', '.join(TRAINING_OPERATORS)}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        while created < args.count and attempts < max_attempts:
            attempts += 1

            chain_length = rng.choices(
                population=list(range(args.min_len, args.max_len + 1)),
                weights=[0.22, 0.34, 0.24, 0.14, 0.06][: (args.max_len - args.min_len + 1)],
            )[0]

            equation, expr_elements, expr_ops, elements_meta = generate_equation(rng, chain_length=chain_length)
            eq_key = _digest64(equation)
            if eq_key in seen_eq:
                continue

            story = generate_story(elements_meta, expr_ops, rng)
            if seen_story is not None:
                story_key = _digest64(story)
                if story_key in seen_story:
                    continue
                seen_story.add(story_key)

            seen_eq.add(eq_key)
            pair = {
                "pair_id": f"pair_{created:07d}",
                "story": story,
                "equation": equation,
                "expr_elements": expr_elements,
                "expr_ops": expr_ops,
                "canon_score": 1.0,
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            created += 1

            if args.progress_every and created % args.progress_every == 0:
                print(f"Generated {created:,} pairs... (attempts={attempts:,})")

    print(f"\nGenerated {created:,} unique pairs in {attempts:,} attempts")
    if created < args.count:
        print(f"[warn] Stopped early; requested {args.count:,} pairs")


if __name__ == "__main__":
    main()


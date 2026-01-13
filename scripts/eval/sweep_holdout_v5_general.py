#!/usr/bin/env python3
"""
Sweep holdout_v5 evaluation across a range of checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "scripts" / "eval"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_holdout_v5_general as evaler  # noqa: E402


def pick_best(results: List[Dict]) -> Dict:
    if not results:
        return {}
    return sorted(
        results,
        key=lambda r: (
            r["rates"]["full_exact_rate"],
            r["rates"]["structure_match_rate"],
            r["rates"]["syntax_valid_rate"],
        ),
        reverse=True,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep holdout_v5 checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/v5_reconciled",
        help="Directory containing epoch_*.pt checkpoints",
    )
    parser.add_argument("--start", type=int, default=20, help="Start epoch")
    parser.add_argument("--end", type=int, default=40, help="End epoch")
    parser.add_argument("--step", type=int, default=1, help="Epoch step")
    parser.add_argument(
        "--data",
        default="data/holdout_v5.jsonl",
        help="Holdout JSONL input",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer_v5.json",
        help="Tokenizer JSON path",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional sample size (0 = full set)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="Enable constrained decoding",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    items = evaler.load_jsonl(data_path)
    if not items:
        raise SystemExit(f"No records found in {data_path}")

    rng = random.Random(args.seed)
    if args.sample_size and args.sample_size < len(items):
        items = rng.sample(items, k=args.sample_size)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    allowed_ids = evaler.build_equation_token_filter(tokenizer) if args.constrained else None

    checkpoint_dir = Path(args.checkpoint_dir)
    results: List[Dict] = []

    for epoch in range(args.start, args.end + 1, args.step):
        ckpt_path = checkpoint_dir / f"epoch_{epoch}.pt"
        if not ckpt_path.exists():
            continue
        model = evaler.load_model(str(ckpt_path), tokenizer)
        totals = evaler.evaluate(
            model,
            tokenizer,
            items,
            max_new_tokens=args.max_new_tokens,
            allowed_token_ids=allowed_ids,
        )
        rates = evaler.summarize(totals)
        entry = {
            "epoch": epoch,
            "checkpoint": str(ckpt_path),
            "totals": totals,
            "rates": rates,
        }
        results.append(entry)
        print(
            f"epoch {epoch}: "
            f"syntax={rates['syntax_valid_rate']:.2%} "
            f"struct={rates['structure_match_rate']:.2%} "
            f"id={rates['id_exact_rate']:.2%} "
            f"full={rates['full_exact_rate']:.2%}"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = pick_best(results)
    if best:
        print(
            "\nBest checkpoint: "
            f"epoch {best['epoch']} "
            f"(full={best['rates']['full_exact_rate']:.2%}, "
            f"struct={best['rates']['structure_match_rate']:.2%}, "
            f"syntax={best['rates']['syntax_valid_rate']:.2%})"
        )
    else:
        print("\nNo checkpoints found in the specified range.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "checkpoint_dir": str(checkpoint_dir),
                "data": str(data_path),
                "tokenizer": args.tokenizer,
                "max_new_tokens": args.max_new_tokens,
                "sample_size": len(items),
                "seed": args.seed,
                "constrained": args.constrained,
                "range": [args.start, args.end, args.step],
            },
            "results": results,
            "best": best,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved sweep report: {output_path}")


if __name__ == "__main__":
    main()

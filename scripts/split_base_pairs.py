#!/usr/bin/env python3
"""
Stream-split story↔equation *base pair* JSONL into train/holdout JSONL.

This keeps each base pair intact (no expansion). Use this for very large datasets
(100k–600k+) without loading everything into RAM.

Input line format:
  {"pair_id": "...", "story": "...", "equation": "...", ...}

Output line format: identical JSON objects (base pairs).
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Tuple


def _split_bucket(pair_id: str, *, seed: int) -> float:
    """
    Deterministic "random" bucket in [0,1) from (seed, pair_id).
    """
    key = f"{seed}|{pair_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


def split_base_pairs(
    input_path: Path,
    train_out: Path,
    holdout_out: Path,
    *,
    holdout_ratio: float,
    seed: int,
    progress_every: int,
) -> Tuple[int, int, int]:
    train_out.parent.mkdir(parents=True, exist_ok=True)
    holdout_out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    train_n = 0
    hold_n = 0

    with (
        open(input_path, "r", encoding="utf-8") as f_in,
        open(train_out, "w", encoding="utf-8") as f_train,
        open(holdout_out, "w", encoding="utf-8") as f_hold,
    ):
        for line in f_in:
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            pair_id = obj.get("pair_id", f"pair_{total:07d}")
            bucket = _split_bucket(pair_id, seed=seed)
            out_line = json.dumps(obj, ensure_ascii=False) + "\n"

            if bucket < holdout_ratio:
                f_hold.write(out_line)
                hold_n += 1
            else:
                f_train.write(out_line)
                train_n += 1

            if progress_every and total % progress_every == 0:
                print(f"Processed {total:,} pairs... train={train_n:,} holdout={hold_n:,}")

    return total, train_n, hold_n


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-split base pairs into train/holdout JSONL")
    parser.add_argument("--input", required=True, help="Input base-pair JSONL")
    parser.add_argument("--train-output", required=True, help="Train base-pair JSONL output")
    parser.add_argument("--holdout-output", required=True, help="Holdout base-pair JSONL output")
    parser.add_argument("--holdout-ratio", type=float, default=0.1, help="Holdout ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic split")
    parser.add_argument("--progress-every", type=int, default=50000, help="Progress print interval (0=off)")
    args = parser.parse_args()

    input_path = Path(args.input)
    train_out = Path(args.train_output)
    holdout_out = Path(args.holdout_output)

    print("=" * 70)
    print("BASE PAIR SPLIT")
    print("=" * 70)
    print(f"Input:   {input_path}")
    print(f"Train:   {train_out}")
    print(f"Holdout: {holdout_out}")
    print(f"Holdout ratio: {args.holdout_ratio}")
    print(f"Seed: {args.seed}\n")

    total, train_n, hold_n = split_base_pairs(
        input_path,
        train_out,
        holdout_out,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
        progress_every=args.progress_every,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Total:   {total:,}")
    print(f"Train:   {train_n:,} ({train_n / max(total, 1) * 100:.1f}%)")
    print(f"Holdout: {hold_n:,} ({hold_n / max(total, 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()


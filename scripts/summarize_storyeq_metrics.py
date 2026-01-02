#!/usr/bin/env python3
"""
Summarize `training_metrics.json` produced by `scripts/train_story_equation.py`.

Prints:
- Best epoch by eval token loss
- Train/eval perplexity trends
- Overfitting signals (generalization gap)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt(x: Optional[float], *, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return f"{x:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize story↔equation training metrics")
    parser.add_argument("--metrics", required=True, help="Path to training_metrics.json")
    parser.add_argument("--last", type=int, default=40, help="Show last N epochs (default: 40)")
    parser.add_argument("--by-direction", action="store_true", help="Also print per-direction tables when available")
    parser.add_argument("--show-seq-exact", action="store_true", help="Include teacher-forced sequence exact in tables")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Min delta for overfitting warnings")
    parser.add_argument("--patience", type=int, default=5, help="Epochs of rising eval loss to warn (default: 5)")
    args = parser.parse_args()

    path = Path(args.metrics)
    data = json.loads(path.read_text(encoding="utf-8"))
    epochs: List[Dict[str, Any]] = data.get("epoch_metrics", []) or []
    if not epochs:
        raise SystemExit("No epoch_metrics found (need a newer training_metrics.json).")

    best_epoch = None
    best_eval_token_loss = None
    for e in epochs:
        ev = e.get("eval", {})
        ev_loss = ev.get("token_loss")
        if ev_loss is None:
            continue
        if best_eval_token_loss is None or ev_loss < best_eval_token_loss:
            best_eval_token_loss = ev_loss
            best_epoch = int(e.get("epoch", 0))

    print("=" * 80)
    print(f"Metrics: {path}")
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch} (eval_token_loss={_fmt(best_eval_token_loss)})")
    print("=" * 80)

    start = max(0, len(epochs) - max(args.last, 1))
    view = epochs[start:]

    cols = ["epoch", "time_s", "lr", "train_ppl", "eval_ppl", "train_acc", "eval_acc", "gap_ppl"]
    if args.show_seq_exact:
        cols.extend(["train_seq", "eval_seq"])
    print("  ".join(f"{c:>9}" for c in cols))
    for e in view:
        tr = e.get("train", {})
        ev = e.get("eval", {})
        epoch = int(e.get("epoch", 0))
        tsec = e.get("time_sec")
        lr = e.get("lr")
        tr_ppl = tr.get("perplexity")
        ev_ppl = ev.get("perplexity")
        tr_acc = tr.get("token_accuracy")
        ev_acc = ev.get("token_accuracy")
        tr_seq = tr.get("sequence_exact")
        ev_seq = ev.get("sequence_exact")
        gap = (ev_ppl - tr_ppl) if (isinstance(ev_ppl, (int, float)) and isinstance(tr_ppl, (int, float))) else None

        row = [
            f"{epoch:9d}",
            f"{_fmt(tsec, nd=1):>9}",
            f"{_fmt(lr, nd=6):>9}",
            f"{_fmt(tr_ppl, nd=2):>9}",
            f"{_fmt(ev_ppl, nd=2):>9}",
            f"{_fmt(tr_acc, nd=3):>9}",
            f"{_fmt(ev_acc, nd=3):>9}",
            f"{_fmt(gap, nd=2):>9}",
        ]
        if args.show_seq_exact:
            row.extend([f"{_fmt(tr_seq, nd=3):>9}", f"{_fmt(ev_seq, nd=3):>9}"])
        print("  ".join(row))

    if args.by_direction:
        # Gather directions present.
        directions = set()
        for e in epochs:
            for side in ("train", "eval"):
                by_dir = (e.get(side) or {}).get("by_direction") or {}
                directions.update(by_dir.keys())

        for direction in sorted(directions):
            best_d_epoch = None
            best_d_loss = None
            for e in epochs:
                ev = (e.get("eval") or {}).get("by_direction") or {}
                if direction not in ev:
                    continue
                loss = ev[direction].get("token_loss")
                if loss is None:
                    continue
                if best_d_loss is None or loss < best_d_loss:
                    best_d_loss = loss
                    best_d_epoch = int(e.get("epoch", 0))

            print("\n" + "-" * 80)
            print(f"Direction: {direction}")
            if best_d_epoch is not None:
                print(f"Best epoch ({direction}): {best_d_epoch} (eval_token_loss={_fmt(best_d_loss)})")
            print("-" * 80)

            d_cols = ["epoch", "train_ppl", "eval_ppl", "train_acc", "eval_acc"]
            if args.show_seq_exact:
                d_cols.extend(["train_seq", "eval_seq"])
            print("  ".join(f"{c:>9}" for c in d_cols))

            for e in view:
                epoch = int(e.get("epoch", 0))
                tr = ((e.get("train") or {}).get("by_direction") or {}).get(direction, {}) or {}
                ev = ((e.get("eval") or {}).get("by_direction") or {}).get(direction, {}) or {}
                tr_ppl = tr.get("perplexity")
                ev_ppl = ev.get("perplexity")
                tr_acc = tr.get("token_accuracy")
                ev_acc = ev.get("token_accuracy")
                tr_seq = tr.get("sequence_exact")
                ev_seq = ev.get("sequence_exact")

                d_row = [
                    f"{epoch:9d}",
                    f"{_fmt(tr_ppl, nd=2):>9}",
                    f"{_fmt(ev_ppl, nd=2):>9}",
                    f"{_fmt(tr_acc, nd=3):>9}",
                    f"{_fmt(ev_acc, nd=3):>9}",
                ]
                if args.show_seq_exact:
                    d_row.extend([f"{_fmt(tr_seq, nd=3):>9}", f"{_fmt(ev_seq, nd=3):>9}"])
                print("  ".join(d_row))

    # Overfitting heuristic: eval token loss rising for patience epochs after best
    if best_epoch is not None and best_eval_token_loss is not None:
        patience = max(args.patience, 1)
        min_delta = args.min_delta

        post = [e for e in epochs if int(e.get("epoch", 0)) > best_epoch]
        rises = 0
        for e in post:
            ev_loss = (e.get("eval") or {}).get("token_loss")
            if ev_loss is None:
                continue
            if ev_loss > best_eval_token_loss + min_delta:
                rises += 1
            else:
                rises = 0
            if rises >= patience:
                print("\n[warn] Potential overfitting:")
                print(
                    f"  Eval token loss stayed above best by >{min_delta} "
                    f"for {patience} consecutive epochs after epoch {best_epoch}."
                )
                break


if __name__ == "__main__":
    main()

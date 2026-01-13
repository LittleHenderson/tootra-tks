#!/usr/bin/env python3
"""
Evaluate the v5 general LM on holdout_v5.jsonl (story -> equation).
Reports syntax/structure/ID/full metrics using canonical normalization.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "scripts" / "eval"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from tks_llm_core_v5 import TKSGeneralConfig, TKSGeneralLM  # noqa: E402
from metrics import compare_tks_expressions  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_model(checkpoint_path: str, tokenizer: Tokenizer) -> TKSGeneralLM:
    config = TKSGeneralConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=384,
        num_layers=12,
        num_scales=4,
    )
    model = TKSGeneralLM(config).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def build_equation_token_filter(tokenizer: Tokenizer) -> Optional[Set[int]]:
    allowed_chars = set("ABCD0123456789^_dD+-*/o<>():. T")
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None

    allowed_ids: Set[int] = set()
    for _token, tid in vocab.items():
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            continue
        if not decoded:
            continue
        if all(char in allowed_chars for char in decoded):
            allowed_ids.add(int(tid))

    return allowed_ids or None


def generate(
    model: TKSGeneralLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    allowed_token_ids: Optional[Set[int]] = None,
) -> str:
    sep_id = tokenizer.token_to_id("<SEP>")
    eos_id = tokenizer.token_to_id("<EOS>")
    bos_id = tokenizer.token_to_id("<BOS>")
    pad_id = tokenizer.token_to_id("<PAD>")

    enc = tokenizer.encode(prompt + " <SEP> ")
    input_ids = [bos_id] + enc.ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    allowed_ids = None
    if allowed_token_ids:
        allowed_set = set(allowed_token_ids)
        allowed_set.update({eos_id})
        allowed_ids = torch.tensor(
            sorted(allowed_set),
            device=input_tensor.device,
            dtype=torch.long,
        )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_tensor)
            logits = out["logits"][:, -1, :]
            if allowed_ids is not None:
                mask = torch.full_like(logits, -float("inf"))
                mask[:, allowed_ids] = 0.0
                logits = logits + mask
            if pad_id is not None:
                logits[:, pad_id] = -1e9
            next_token = logits.argmax(dim=-1).item()
            if next_token == eos_id or (pad_id is not None and next_token == pad_id):
                break
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], device=DEVICE)],
                dim=1,
            )

    all_ids = input_tensor[0].tolist()
    try:
        sep_pos = all_ids.index(sep_id)
        output_ids = all_ids[sep_pos + 1 :]
    except ValueError:
        output_ids = all_ids[len(input_ids) :]

    return tokenizer.decode(output_ids).strip()


def evaluate(
    model: TKSGeneralLM,
    tokenizer: Tokenizer,
    items: List[Dict],
    *,
    max_new_tokens: int,
    allowed_token_ids: Optional[Set[int]] = None,
) -> Dict[str, int]:
    totals = {
        "syntax_valid": 0,
        "structure_match": 0,
        "id_exact": 0,
        "full_exact": 0,
        "total": 0,
    }
    for item in items:
        prompt = str(item.get("story", ""))
        expected = str(item.get("equation", ""))
        predicted = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            allowed_token_ids=allowed_token_ids,
        )
        metrics = compare_tks_expressions(predicted, expected)
        totals["syntax_valid"] += int(metrics.syntax_valid)
        totals["structure_match"] += int(metrics.structure_match)
        totals["id_exact"] += int(metrics.id_exact)
        totals["full_exact"] += int(metrics.full_exact)
        totals["total"] += 1
    return totals


def summarize(totals: Dict[str, int]) -> Dict[str, float]:
    total = max(totals.get("total", 0), 1)
    return {
        "syntax_valid_rate": totals["syntax_valid"] / total,
        "structure_match_rate": totals["structure_match"] / total,
        "id_exact_rate": totals["id_exact"] / total,
        "full_exact_rate": totals["full_exact"] / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v5 general holdout")
    parser.add_argument(
        "--data",
        default="data/holdout_v5.jsonl",
        help="Holdout JSONL input",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v5_reconciled/epoch_60.pt",
        help="Model checkpoint",
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
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--constrained-only",
        action="store_true",
        help="Run only constrained decoding",
    )
    parser.add_argument(
        "--unconstrained-only",
        action="store_true",
        help="Run only unconstrained decoding",
    )
    args = parser.parse_args()

    if args.constrained_only and args.unconstrained_only:
        raise SystemExit("Choose at most one of --constrained-only or --unconstrained-only.")

    data_path = Path(args.data)
    items = load_jsonl(data_path)
    if not items:
        raise SystemExit(f"No records found in {data_path}")

    rng = random.Random(args.seed)
    if args.sample_size and args.sample_size < len(items):
        items = rng.sample(items, k=args.sample_size)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint, tokenizer)
    allowed_ids = build_equation_token_filter(tokenizer)

    results: Dict[str, Dict[str, int]] = {}
    run_unconstrained = not args.constrained_only
    run_constrained = not args.unconstrained_only

    if run_unconstrained:
        totals = evaluate(
            model,
            tokenizer,
            items,
            max_new_tokens=args.max_new_tokens,
            allowed_token_ids=None,
        )
        results["unconstrained"] = totals
        rates = summarize(totals)
        print(f"Evaluated {totals['total']} records on {DEVICE}")
        print(
            "Unconstrained rates: "
            f"syntax_valid={rates['syntax_valid_rate']:.2%}, "
            f"structure_match={rates['structure_match_rate']:.2%}, "
            f"id_exact={rates['id_exact_rate']:.2%}, "
            f"full_exact={rates['full_exact_rate']:.2%}"
        )

    if run_constrained:
        totals = evaluate(
            model,
            tokenizer,
            items,
            max_new_tokens=args.max_new_tokens,
            allowed_token_ids=allowed_ids,
        )
        results["constrained"] = totals
        rates = summarize(totals)
        print(
            "Constrained rates: "
            f"syntax_valid={rates['syntax_valid_rate']:.2%}, "
            f"structure_match={rates['structure_match_rate']:.2%}, "
            f"id_exact={rates['id_exact_rate']:.2%}, "
            f"full_exact={rates['full_exact_rate']:.2%}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "data": str(data_path),
                "checkpoint": args.checkpoint,
                "tokenizer": args.tokenizer,
                "max_new_tokens": args.max_new_tokens,
                "sample_size": len(items),
                "seed": args.seed,
            },
            "results": results,
            "rates": {key: summarize(val) for key, val in results.items()},
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()

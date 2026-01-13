#!/usr/bin/env python3
"""
Evaluate RPM holdout v5 with canonical TKS metrics and RPM path validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tks_llm_core_v5 import TKSGeneralConfig, TKSGeneralLM  # noqa: E402
from scripts.eval.metrics import compare_tks_expressions  # noqa: E402
from scripts.eval.rpm_validator import (  # noqa: E402
    RPMPathValidator,
    build_rpm_token_filter,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path: Path) -> List[Dict]:
    items = []
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


def generate(
    model: TKSGeneralLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    allowed_token_ids: List[int] | None = None,
    require_colons: int = 0,
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

    colon_count = 0
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
            if require_colons and eos_id is not None and colon_count < require_colons:
                logits[:, eos_id] = -1e9

            next_token = logits.argmax(dim=-1).item()
            if next_token == eos_id or (pad_id is not None and next_token == pad_id):
                break

            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], device=DEVICE)], dim=1
            )
            if require_colons:
                try:
                    decoded = tokenizer.decode([next_token])
                except Exception:
                    decoded = ""
                if decoded:
                    colon_count += decoded.count(":")

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
    records: List[Dict],
    *,
    rpm_validator: RPMPathValidator,
    max_new_tokens: int,
    require_colons: int,
    sample_count: int,
) -> Tuple[Dict, List[Dict]]:
    allowed_ids = build_rpm_token_filter(tokenizer)
    if not allowed_ids:
        allowed_ids = None

    totals = {
        "syntax_valid": 0,
        "structure_match": 0,
        "id_exact": 0,
        "full_exact": 0,
        "rpm_valid": 0,
        "rpm_expected_valid": 0,
    }
    error_counts: Dict[str, int] = {}
    samples: List[Dict] = []

    for idx, record in enumerate(records):
        prompt = str(record.get("input", ""))
        expected = str(record.get("output", ""))
        predicted = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            allowed_token_ids=allowed_ids,
            require_colons=require_colons,
        )

        rpm_result = rpm_validator.validate(predicted, allow_extract=True)
        pred_eval = rpm_result.candidate or predicted
        metrics = compare_tks_expressions(pred_eval, expected)

        totals["syntax_valid"] += int(metrics.syntax_valid)
        totals["structure_match"] += int(metrics.structure_match)
        totals["id_exact"] += int(metrics.id_exact)
        totals["full_exact"] += int(metrics.full_exact)
        totals["rpm_valid"] += int(rpm_result.is_valid)

        expected_check = rpm_validator.validate(expected, allow_extract=False)
        totals["rpm_expected_valid"] += int(expected_check.is_valid)

        for error in rpm_result.errors:
            error_counts[error] = error_counts.get(error, 0) + 1

        if len(samples) < sample_count:
            samples.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "expected": expected,
                    "predicted": predicted,
                    "candidate": rpm_result.candidate,
                    "rpm_valid": rpm_result.is_valid,
                    "rpm_errors": rpm_result.errors,
                    "syntax_valid": metrics.syntax_valid,
                    "structure_match": metrics.structure_match,
                    "id_exact": metrics.id_exact,
                    "full_exact": metrics.full_exact,
                }
            )

    n = max(len(records), 1)
    summary = {
        "total_examples": len(records),
        "syntax_valid_rate": totals["syntax_valid"] / n,
        "structure_match_rate": totals["structure_match"] / n,
        "id_exact_rate": totals["id_exact"] / n,
        "full_exact_rate": totals["full_exact"] / n,
        "rpm_valid_rate": totals["rpm_valid"] / n,
        "rpm_expected_valid_rate": totals["rpm_expected_valid"] / n,
        "counts": totals,
        "rpm_error_counts": error_counts,
    }
    return summary, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RPM holdout v5")
    parser.add_argument(
        "--data",
        default="data/holdout_rpm_v5.jsonl",
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
        "--output",
        default="output/holdout_eval_v5.json",
        help="Output JSON report",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=3,
        help="Minimum RPM stack length",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=3,
        help="Maximum RPM stack length",
    )
    parser.add_argument(
        "--require-colons",
        type=int,
        default=2,
        help="Minimum number of ':' before EOS allowed",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Sample count to include in report",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_path = Path(args.data)
    records = load_jsonl(data_path)
    if not records:
        raise SystemExit(f"No records found in {data_path}")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint, tokenizer)
    rpm_validator = RPMPathValidator(min_len=args.min_len, max_len=args.max_len)

    summary, samples = evaluate(
        model,
        tokenizer,
        records,
        rpm_validator=rpm_validator,
        max_new_tokens=args.max_new_tokens,
        require_colons=args.require_colons,
        sample_count=args.sample_count,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": DEVICE,
        "config": {
            "data": str(data_path),
            "checkpoint": args.checkpoint,
            "tokenizer": args.tokenizer,
            "max_new_tokens": args.max_new_tokens,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "require_colons": args.require_colons,
            "seed": args.seed,
        },
        "summary": summary,
        "samples": samples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluated {summary['total_examples']} records")
    print(
        "Rates: "
        f"syntax_valid={summary['syntax_valid_rate']:.2%}, "
        f"structure_match={summary['structure_match_rate']:.2%}, "
        f"id_exact={summary['id_exact_rate']:.2%}, "
        f"full_exact={summary['full_exact_rate']:.2%}, "
        f"rpm_valid={summary['rpm_valid_rate']:.2%}"
    )
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()

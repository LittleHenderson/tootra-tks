#!/usr/bin/env python3
"""
RPM holdout evaluation with constrained decoding.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tokenizers import Tokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from tks_llm_core_v5 import TKSGeneralLM, TKSGeneralConfig
from eval.rpm_validator import RPMPathValidator, build_rpm_token_filter


def load_model(checkpoint_path: str, tokenizer: Tokenizer, device: torch.device):
    config = TKSGeneralConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=384,
        num_layers=12,
        num_scales=4,
    )
    model = TKSGeneralLM(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 60,
    allowed_token_ids: Optional[set[int]] = None,
) -> str:
    sep_id = tokenizer.token_to_id("<SEP>")
    eos_id = tokenizer.token_to_id("<EOS>")
    bos_id = tokenizer.token_to_id("<BOS>")
    if sep_id is None or eos_id is None or bos_id is None:
        raise SystemExit("Tokenizer missing required <BOS>/<EOS>/<SEP> tokens.")

    enc = tokenizer.encode(prompt + " <SEP>")
    input_ids = [bos_id] + enc.ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    allowed_ids = None
    if allowed_token_ids:
        allowed_set = set(allowed_token_ids)
        allowed_set.update({eos_id, 0})
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

            next_token = logits.argmax(dim=-1).item()
            if next_token == eos_id or next_token == 0:
                break
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], device=device)],
                dim=1,
            )

    all_ids = input_tensor[0].tolist()
    try:
        sep_pos = all_ids.index(sep_id)
        output_ids = all_ids[sep_pos + 1 :]
    except ValueError:
        output_ids = all_ids[len(input_ids) :]
    return tokenizer.decode(output_ids).strip()


def load_holdout(path: Path) -> List[Dict]:
    entries: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RPM holdout validity")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v5_reconciled/epoch_60.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer_v5.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--data",
        default="datasets/jsonl/tks_holdout_v1.jsonl",
        help="Path to holdout JSONL",
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint, tokenizer, device)

    data_path = Path(args.data)
    entries = load_holdout(data_path)
    rpm_entries = [e for e in entries if "rpm" in str(e.get("type", "")).lower()]

    if args.max_samples and args.max_samples < len(rpm_entries):
        rng = random.Random(args.seed)
        rpm_entries = rng.sample(rpm_entries, k=args.max_samples)

    rpm_validator = RPMPathValidator(min_len=2, max_len=3)
    rpm_allowed_ids = build_rpm_token_filter(tokenizer)
    if not rpm_allowed_ids:
        rpm_allowed_ids = None

    total = 0
    exact = 0
    valid = 0
    invalid_targets = 0

    for entry in rpm_entries:
        prompt = entry.get("input", "")
        target = entry.get("output", entry.get("target", ""))
        pred = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            allowed_token_ids=rpm_allowed_ids,
        )
        pred_result = rpm_validator.validate(pred, allow_extract=False)
        tgt_result = rpm_validator.validate(target, allow_extract=True)
        if not tgt_result.is_valid:
            invalid_targets += 1
            continue

        total += 1
        if pred_result.is_valid:
            valid += 1
        if pred_result.is_valid and pred_result.normalized == tgt_result.normalized:
            exact += 1

    valid_rate = (valid / total) if total else 0.0
    exact_rate = (exact / total) if total else 0.0

    print("=" * 70)
    print("RPM HOLDOUT EVAL")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {data_path}")
    print(f"RPM samples: {total}")
    if invalid_targets:
        print(f"Skipped invalid targets: {invalid_targets}")
    print(f"RPM validity: {valid}/{total} ({valid_rate:.2%})")
    print(f"RPM exact:    {exact}/{total} ({exact_rate:.2%})")
    print("=" * 70)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "data": str(data_path),
            "rpm_samples": total,
            "rpm_valid": valid,
            "rpm_exact": exact,
            "rpm_valid_rate": valid_rate,
            "rpm_exact_rate": exact_rate,
            "invalid_targets": invalid_targets,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

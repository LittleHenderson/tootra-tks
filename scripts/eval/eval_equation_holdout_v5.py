#!/usr/bin/env python3
"""
Evaluate constrained vs unconstrained decoding on holdout equations.

Computes syntax/structure/ID/full metrics using the canonical normalizer.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_story_equation as ese  # noqa: E402
from beam_search_decoder import BeamSearchDecoder, TKSCharEquationConstraint  # noqa: E402
from eval.metrics import compare_tks_expressions  # noqa: E402


ELEMENT_RE = re.compile(r"[ABCD](?:10|[0-9])(?:\^(?:10|[0-9]))?(?:_d[1-7])?")


def build_story_to_eq_prompt(story: str) -> str:
    return (
        "Given this TKS narrative:\n\n"
        f"{story}\n\n"
        "Translate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 0-9)."
    )


def is_equation(text: str) -> bool:
    return bool(ELEMENT_RE.search(text or ""))


def load_pairs(path: Path) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "story" in obj and "equation" in obj:
                story = obj.get("story", "")
                equation = obj.get("equation", "")
                pair_id = obj.get("pair_id", "")
                pairs.append((pair_id, build_story_to_eq_prompt(story), equation))
                continue
            if "input" in obj and "target" in obj and is_equation(obj.get("target", "")):
                pair_id = obj.get("metadata", {}).get("pair_id", "")
                pairs.append((pair_id, obj.get("input", ""), obj.get("target", "")))
    return pairs


def load_seq2seq_model(checkpoint_path: Path, tokenizer: ese.Seq2SeqTokenizer, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {}) or {}
    else:
        state_dict = checkpoint
        config = {}

    model_type = config.get("model_type")
    state_keys = list(state_dict.keys()) if hasattr(state_dict, "keys") else []

    if not model_type:
        if any(k.startswith("encoder_embedding.") for k in state_keys):
            model_type = "transformer_seq2seq"
        elif any(k.startswith("encoder.") for k in state_keys) and any(k.startswith("decoder.") for k in state_keys):
            model_type = "lstm_seq2seq"

    if model_type == "transformer_seq2seq":
        model = ese.Seq2SeqTransformer(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            nhead=int(config.get("nhead", 8)),
            max_length=int(config.get("max_length", tokenizer.max_length)),
        )
    elif model_type == "lstm_seq2seq":
        model = ese.SimpleLSTMSeq2Seq(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=int(config.get("hidden_dim", 128)),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_type={model_type!r}")

    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model_type, model


def decode_prompts(
    prompts: List[str],
    model: torch.nn.Module,
    tokenizer: ese.Seq2SeqTokenizer,
    constraint: TKSCharEquationConstraint | None,
    *,
    beam_width: int,
    max_new_tokens: int,
    length_penalty: float,
) -> List[str]:
    decoder = BeamSearchDecoder(
        model,
        tokenizer,
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        early_stopping=True,
        n_best=1,
        constraint=constraint,
    )
    preds: List[str] = []
    for prompt in prompts:
        preds.append(decoder.decode_text(prompt).strip())
    return preds


def compute_counts(preds: List[str], targets: List[str]) -> Dict[str, int]:
    totals = {
        "syntax_valid": 0,
        "structure_match": 0,
        "id_exact": 0,
        "full_exact": 0,
        "total": 0,
    }
    for pred, gold in zip(preds, targets):
        result = compare_tks_expressions(pred, gold)
        totals["syntax_valid"] += int(result.syntax_valid)
        totals["structure_match"] += int(result.structure_match)
        totals["id_exact"] += int(result.id_exact)
        totals["full_exact"] += int(result.full_exact)
        totals["total"] += 1
    return totals


def summarize_counts(totals: Dict[str, int]) -> Dict[str, float]:
    total = max(totals.get("total", 0), 1)
    return {
        "syntax_valid_rate": totals["syntax_valid"] / total,
        "structure_match_rate": totals["structure_match"] / total,
        "id_exact_rate": totals["id_exact"] / total,
        "full_exact_rate": totals["full_exact"] / total,
    }


def print_summary(label: str, totals: Dict[str, int]) -> None:
    rates = summarize_counts(totals)
    total = max(totals.get("total", 0), 1)
    print(f"{label}:")
    print(f"  Syntax valid:    {totals['syntax_valid']}/{total} ({rates['syntax_valid_rate']:.2%})")
    print(f"  Structure match: {totals['structure_match']}/{total} ({rates['structure_match_rate']:.2%})")
    print(f"  ID exact:        {totals['id_exact']}/{total} ({rates['id_exact_rate']:.2%})")
    print(f"  Full exact:      {totals['full_exact']}/{total} ({rates['full_exact_rate']:.2%})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Holdout equation eval with constrained decoding")
    parser.add_argument("--model", required=True, help="Path to seq2seq checkpoint (.pt)")
    parser.add_argument("--data", default="data/holdout_v5.jsonl", help="Holdout JSONL path")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer.json")
    parser.add_argument("--tokenizer-mode", type=str, default=None, choices=["char", "mixed"])
    parser.add_argument("--constrained-only", action="store_true", help="Run only constrained decode")
    parser.add_argument("--unconstrained-only", action="store_true", help="Run only unconstrained decode")
    args = parser.parse_args()

    if args.constrained_only and args.unconstrained_only:
        raise SystemExit("Choose at most one of --constrained-only or --unconstrained-only.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False. Run `python scripts/check_cuda.py`.")

    data_path = Path(args.data)
    pairs = load_pairs(data_path)
    if not pairs:
        raise SystemExit(f"No equation pairs found in {data_path}")

    rng = random.Random(args.seed)
    if args.sample_size and args.sample_size < len(pairs):
        pairs = rng.sample(pairs, k=args.sample_size)

    prompts = [prompt for _pid, prompt, _eq in pairs]
    targets = [eq for _pid, _prompt, eq in pairs]

    model_path = Path(args.model)
    tok_path = Path(args.tokenizer) if args.tokenizer else (model_path.parent / "tokenizer.json")
    tokenizer = ese.Seq2SeqTokenizer.from_json(tok_path) if tok_path.exists() else ese.Seq2SeqTokenizer(max_length=256)
    if args.tokenizer_mode:
        tokenizer.mode = args.tokenizer_mode.lower()
        tokenizer._rebuild_multi_token_index()

    model_type, model = load_seq2seq_model(model_path, tokenizer, device)

    print("=" * 70)
    print("HOLDOUT EQUATION EVAL (CONSTRAINED VS UNCONSTRAINED)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Tokenizer: {tok_path if tok_path.exists() else 'default'} (mode={tokenizer.mode})")
    print(f"Data: {data_path}")
    print(f"Samples: {len(prompts)}")
    print(f"Beam width: {args.beam_width}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Length penalty: {args.length_penalty}")
    print(f"Device: {device}\n")

    results: Dict[str, Dict] = {}

    run_unconstrained = not args.constrained_only
    run_constrained = not args.unconstrained_only

    if run_unconstrained:
        t0 = time.monotonic()
        preds = decode_prompts(
            prompts,
            model,
            tokenizer,
            None,
            beam_width=args.beam_width,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
        )
        totals = compute_counts(preds, targets)
        totals["seconds"] = time.monotonic() - t0
        results["unconstrained"] = totals
        print_summary("Unconstrained", totals)
        print(f"  Time: {totals['seconds']:.1f}s\n")

    if run_constrained:
        if tokenizer.mode != "char":
            raise SystemExit(
                "--constrained requires char tokenizer mode. Rerun with --tokenizer-mode char."
            )
        constraint = TKSCharEquationConstraint(tokenizer)
        t0 = time.monotonic()
        preds = decode_prompts(
            prompts,
            model,
            tokenizer,
            constraint,
            beam_width=args.beam_width,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
        )
        totals = compute_counts(preds, targets)
        totals["seconds"] = time.monotonic() - t0
        results["constrained"] = totals
        print_summary("Constrained", totals)
        print(f"  Time: {totals['seconds']:.1f}s\n")

    if run_constrained and run_unconstrained:
        uncon = results["unconstrained"]
        cons = results["constrained"]
        print("Delta (constrained - unconstrained):")
        for key in ("syntax_valid", "structure_match", "id_exact", "full_exact"):
            delta = cons[key] - uncon[key]
            print(f"  {key}: {delta:+d}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "model": str(model_path),
                "data": str(data_path),
                "beam_width": args.beam_width,
                "max_new_tokens": args.max_new_tokens,
                "length_penalty": args.length_penalty,
                "sample_size": len(prompts),
                "tokenizer_mode": tokenizer.mode,
            },
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

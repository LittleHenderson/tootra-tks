#!/usr/bin/env python3
"""
Run beam search decoding experiments for Story→Equation generation.

Evaluates exact match, element/operator F1, similarity, and canonical validity.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import eval_story_equation as ese  # noqa: E402
from beam_search_decoder import BeamSearchDecoder, TKSCharEquationConstraint  # noqa: E402
from storyeq_generation_metrics import CanonValidator, compute_generation_metrics  # noqa: E402


def build_story_to_eq_prompt(story: str) -> str:
    return (
        "Given this TKS narrative:\n\n"
        f"{story}\n\n"
        "Translate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 1-10)."
    )


def load_base_pairs(path: Path) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            story = obj.get("story", "")
            equation = obj.get("equation", "")
            pair_id = obj.get("pair_id", "")
            pairs.append((pair_id, story, equation))
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


def generate_comparison_table(results: Dict[str, Dict], output_path: Path) -> None:
    lines = [
        "# Beam Search Comparison Results",
        "",
        "| Beam Width | Exact Match | Element F1 | Operator F1 | String Sim | Syntax Valid | avg_s/ex |",
        "|------------|-------------|------------|-------------|------------|--------------|----------|",
    ]

    def beam_key_sort(k: str) -> int:
        try:
            return int(k.split("_", 1)[1])
        except Exception:
            return 999999

    for key in sorted(results.keys(), key=beam_key_sort):
        r = results[key]
        lines.append(
            f"| {r['beam_width']:^10} | {r['exact_match_rate']:^11.2%} | "
            f"{r['avg_element_f1']:^10.4f} | {r['avg_operator_f1']:^11.4f} | "
            f"{r['avg_string_similarity']:^10.4f} | {r['syntax_valid_rate']:^12.2%} | "
            f"{r.get('avg_seconds_per_example', 0.0):^8.3f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run beam search experiments for Story→Equation decoding")
    parser.add_argument("--model", required=True, help="Path to seq2seq checkpoint (.pt)")
    parser.add_argument("--data", required=True, help="Path to base-pair holdout JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--beams", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer.json (defaults to model sibling)")
    parser.add_argument("--tokenizer-mode", type=str, default=None, choices=["char", "mixed"], help="Override tokenizer mode")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--constrained", action="store_true", help="Apply strict TKS equation grammar constraints")
    parser.add_argument("--save-examples", type=int, default=25, help="Save N example predictions per beam")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False. Run `python scripts/check_cuda.py`.")

    model_path = Path(args.model)
    tok_path = Path(args.tokenizer) if args.tokenizer else (model_path.parent / "tokenizer.json")
    tokenizer = ese.Seq2SeqTokenizer.from_json(tok_path) if tok_path.exists() else ese.Seq2SeqTokenizer(max_length=256)
    if args.tokenizer_mode:
        tokenizer.mode = args.tokenizer_mode.lower()
        tokenizer._rebuild_multi_token_index()

    model_type, model = load_seq2seq_model(model_path, tokenizer, device)

    data_path = Path(args.data)
    pairs = load_base_pairs(data_path)
    rng = random.Random(args.seed)
    if args.sample_size and args.sample_size < len(pairs):
        pairs = rng.sample(pairs, k=args.sample_size)

    prompts = [build_story_to_eq_prompt(story) for _pid, story, _eq in pairs]
    targets = [eq for _pid, _story, eq in pairs]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BEAM SEARCH EXPERIMENTS (STORY -> EQUATION)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Tokenizer: {tok_path if tok_path.exists() else 'default'} (vocab={tokenizer.actual_vocab_size}, max_len={tokenizer.max_length})")
    print(f"Data: {data_path}")
    print(f"Samples: {len(prompts)}")
    print(f"Beams: {args.beams}")
    print(f"Constrained: {args.constrained}")
    print(f"Device: {device}\n")

    validator = CanonValidator()
    if args.constrained and getattr(tokenizer, "mode", "char") != "char":
        raise SystemExit(
            "--constrained currently supports only the char tokenizer; "
            "rerun without --constrained or implement token-level constraints."
        )
    constraint = TKSCharEquationConstraint(tokenizer) if args.constrained else None

    results: Dict[str, Dict] = {}

    for beam_width in args.beams:
        print(f"\n{'=' * 60}")
        print(f"beam_width = {beam_width}")
        print(f"{'=' * 60}")

        decoder = BeamSearchDecoder(
            model,
            tokenizer,
            beam_width=beam_width,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
            early_stopping=True,
            n_best=1,
            constraint=constraint,
        )

        preds: List[str] = []
        t0 = time.monotonic()
        for i, prompt in enumerate(prompts):
            pred = decoder.decode_text(prompt).strip()
            preds.append(pred)
            if (i + 1) % 50 == 0 or (i + 1) == len(prompts):
                dt = time.monotonic() - t0
                print(f"  decoded {i+1}/{len(prompts)}  ({dt:.1f}s)")

        dt = time.monotonic() - t0
        avg_s = dt / max(len(prompts), 1)

        metrics = compute_generation_metrics(preds, targets, validator=validator)
        metrics["beam_width"] = int(beam_width)
        metrics["sample_size"] = len(prompts)
        metrics["max_new_tokens"] = args.max_new_tokens
        metrics["length_penalty"] = args.length_penalty
        metrics["constrained"] = bool(args.constrained)
        metrics["avg_seconds_per_example"] = avg_s

        results[f"beam_{beam_width}"] = metrics

        print(f"Exact Match: {metrics['exact_match_rate']:.2%} ({metrics['exact_match_count']}/{metrics['total_examples']})")
        print(f"Element F1:  {metrics['avg_element_f1']:.4f}")
        print(f"Operator F1: {metrics['avg_operator_f1']:.4f}")
        print(f"Similarity:  {metrics['avg_string_similarity']:.4f}")
        print(f"Syntax valid:{metrics['syntax_valid_rate']:.2%}")
        print(f"Speed:      {avg_s:.3f}s/example")

        # Save metrics + a few examples
        (out_dir / f"beam_{beam_width}_results.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if args.save_examples:
            ex_n = min(args.save_examples, len(prompts))
            examples = []
            for (pair_id, _story, tgt), pred, prompt in zip(pairs[:ex_n], preds[:ex_n], prompts[:ex_n]):
                examples.append(
                    {
                        "pair_id": pair_id,
                        "prompt": prompt,
                        "target": tgt,
                        "prediction": pred,
                    }
                )
            (out_dir / f"beam_{beam_width}_examples.json").write_text(
                json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    summary = {
        "experiment": "beam_search_comparison",
        "model": str(model_path),
        "data": str(data_path),
        "device": str(device),
        "beams": args.beams,
        "sample_size": len(prompts),
        "constrained": bool(args.constrained),
        "results": results,
    }
    (out_dir / "beam_search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    generate_comparison_table(results, out_dir / "beam_comparison.md")
    print(f"\nSaved: {out_dir / 'beam_search_summary.json'}")
    print(f"Saved: {out_dir / 'beam_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

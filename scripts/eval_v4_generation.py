#!/usr/bin/env python3
"""
Generation Evaluation for TKSNoeticLM v4

Runs autoregressive generation on held-out data and computes:
- Exact match rate (normalized)
- Element F1 (TKS elements A1-D10)
- Operator F1 (+, -, ->, etc.)
- String similarity (edit distance)
- Syntax validity rate (canonical validator)

Usage:
    python scripts/eval_v4_generation.py \
        --checkpoint output/combined_model_v5_noetic/best_model.pt \
        --data data/combined_all_v5.jsonl \
        --output output/reports/generation_eval \
        --max-new-tokens 128 \
        --num-samples 500

Author: Claude Code
Date: 2026-01-01
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.storyeq_generation_metrics import (
    CanonValidator,
    compute_generation_metrics,
    normalize_equation,
    extract_elements,
    extract_operators,
    string_similarity,
)

# Import tokenizer from train_cuda (same one used in training)
from scripts.train_cuda import SimpleTokenizer

from teacher.validator import CanonicalValidator


# ==============================================================================
# TOKENIZERS AND FORMATTING
# ==============================================================================

class ByteTokenizer:
    """Byte-level tokenizer with special tokens (matches train_tks_llm_v4)."""

    def __init__(self):
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.sep_token = "<SEP>"
        self.unk_token = "<UNK>"
        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.unk_token,
        ]
        self.special_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.byte_offset = len(self.special_tokens)
        self.vocab_size = self.byte_offset + 256

    @classmethod
    def from_json(cls, path: Path) -> "ByteTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != "byte":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')}")
        return cls()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.special_to_id[self.bos_token])
        for b in text.encode("utf-8", errors="replace"):
            tokens.append(self.byte_offset + b)
        if add_eos:
            tokens.append(self.special_to_id[self.eos_token])
        return tokens

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for tid in ids:
            if tid < self.byte_offset:
                continue
            out.append(tid - self.byte_offset)
        return out.decode("utf-8", errors="replace")


class TokenizerAdapter:
    """Unifies byte and char tokenizers for evaluation."""

    def __init__(self, tokenizer: Any, kind: str, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.kind = kind
        self.max_length = max_length
        if kind == "byte":
            self.pad_id = tokenizer.special_to_id["<PAD>"]
            self.bos_id = tokenizer.special_to_id["<BOS>"]
            self.eos_id = tokenizer.special_to_id["<EOS>"]
            self.unk_id = tokenizer.special_to_id["<UNK>"]
            self.id_to_token = None
        else:
            self.pad_id = tokenizer.token_to_id.get("<PAD>", 0)
            self.bos_id = tokenizer.token_to_id.get("<BOS>", 2)
            self.eos_id = tokenizer.token_to_id.get("<EOS>", 3)
            self.unk_id = tokenizer.token_to_id.get("<UNK>", 1)
            self.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        self.stop_ids = [self.eos_id]

    def encode_prompt(self, text: str) -> List[int]:
        if self.kind == "byte":
            return self.tokenizer.encode(text, add_bos=True, add_eos=False)
        ids = self.tokenizer.tokenize(text)
        return [t for t in ids if t not in (self.pad_id, self.eos_id)]

    def encode_target(self, text: str) -> List[int]:
        if self.kind == "byte":
            return self.tokenizer.encode(text, add_bos=False, add_eos=False)
        ids = self.tokenizer.tokenize(text)
        return [t for t in ids if t not in (self.pad_id, self.bos_id, self.eos_id)]

    def decode(self, ids: List[int]) -> str:
        if self.kind == "byte":
            return self.tokenizer.decode(ids)
        return "".join(self.id_to_token.get(t, "") for t in ids if t not in (self.pad_id, self.bos_id, self.eos_id))


EQ_TASK_TYPES = {
    "language_to_fractal",
    "nl_to_fractal",
    "nl_to_equation",
    "story_to_equation",
    "text_to_equation",
    "language-to-fractal",
}

NL_TASK_TYPES = {
    "fractal_to_language",
    "equation_to_story",
    "fractal_to_nl",
    "fractal-to-language",
}


def load_json_if_exists(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint_assets(checkpoint_path: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    if checkpoint_path.is_dir():
        model_path = checkpoint_path / "model.pt"
        if not model_path.exists():
            candidates = sorted(checkpoint_path.glob("*.pt"))
            if candidates:
                model_path = candidates[0]
        config_path = checkpoint_path / "config.json"
        tokenizer_path = checkpoint_path / "tokenizer.json"
        return (
            model_path,
            config_path if config_path.exists() else None,
            tokenizer_path if tokenizer_path.exists() else None,
        )
    config_path = checkpoint_path.parent / "config.json"
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    return (
        checkpoint_path,
        config_path if config_path.exists() else None,
        tokenizer_path if tokenizer_path.exists() else None,
    )


def choose_prompt_format(fmt: str, tokenizer_kind: str, sample_entry: Optional[Dict]) -> str:
    if fmt != "auto":
        return fmt
    if sample_entry:
        sample_input = sample_entry.get("input", "")
        if "<PROMPT>" in sample_input and "<ANSWER>" in sample_input:
            return "raw"
    if tokenizer_kind == "byte":
        return "tagged"
    return "arrow"


def build_prompt(entry: Dict, prompt_format: str, separator: str) -> str:
    inp = entry.get("input", "")
    if prompt_format == "arrow":
        return f"{inp}{separator}"
    if prompt_format == "tagged":
        return f"<PROMPT>\n{inp}\n\n<ANSWER>\n"
    if prompt_format == "raw":
        return inp
    raise ValueError(f"Unknown prompt_format: {prompt_format}")


def is_equation_task(task_type: str, target: str) -> bool:
    key = (task_type or "").lower()
    if key in EQ_TASK_TYPES:
        return True
    if key in NL_TASK_TYPES:
        return False
    elems = extract_elements(target)
    ops = extract_operators(target)
    return bool(elems) and (bool(ops) or len(elems) > 1)


def parse_length_buckets(spec: Optional[str]) -> List[Tuple[str, int, int]]:
    if not spec:
        spec = "0-32,33-64,65-128,129-1000000"
    buckets = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid bucket spec: {part}")
        lo_s, hi_s = part.split("-", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        buckets.append((f"{lo}-{hi}", lo, hi))
    return buckets


def bucket_for_length(length: int, buckets: List[Tuple[str, int, int]]) -> Optional[str]:
    for name, lo, hi in buckets:
        if lo <= length <= hi:
            return name
    return None


def compute_mixed_metrics(
    predictions: List[str],
    targets: List[str],
    eq_flags: List[bool],
    validator: CanonValidator,
) -> Dict[str, Any]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have same length")
    total = len(predictions)
    if total == 0:
        return {
            "exact_match_rate": 0.0,
            "exact_match_count": 0,
            "total_examples": 0,
            "avg_string_similarity": 0.0,
            "avg_element_f1": None,
            "avg_operator_f1": None,
            "syntax_valid_rate": None,
            "equation_examples": 0,
        }

    exact = 0
    sims: List[float] = []
    for pred, tgt in zip(predictions, targets):
        if normalize_equation(pred) == normalize_equation(tgt):
            exact += 1
        sims.append(string_similarity(pred, tgt))

    eq_preds = [p for p, is_eq in zip(predictions, eq_flags) if is_eq]
    eq_targets = [t for t, is_eq in zip(targets, eq_flags) if is_eq]
    eq_metrics = None
    if eq_preds:
        eq_metrics = compute_generation_metrics(eq_preds, eq_targets, validator=validator)

    return {
        "exact_match_rate": exact / max(total, 1),
        "exact_match_count": exact,
        "total_examples": total,
        "avg_string_similarity": sum(sims) / max(total, 1),
        "avg_element_f1": eq_metrics["avg_element_f1"] if eq_metrics else None,
        "avg_operator_f1": eq_metrics["avg_operator_f1"] if eq_metrics else None,
        "syntax_valid_rate": eq_metrics["syntax_valid_rate"] if eq_metrics else None,
        "equation_examples": len(eq_preds),
    }


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_v4_model(checkpoint_path: str, device: torch.device, config_path: Optional[Path] = None):
    """Load TKSNoeticLM v4 model from checkpoint."""
    from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict: Dict[str, Any] = {}
    if config_path:
        config_dict.update(load_json_if_exists(config_path))

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if isinstance(checkpoint.get('config'), dict):
            config_dict.update(checkpoint['config'])
    else:
        state_dict = checkpoint

    # Infer vocab_size from embedding weight shape if not in config
    vocab_size = config_dict.get('vocab_size', 512)
    if 'embedding.embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.embedding.weight'].shape[0]
    elif 'output_projection.bias' in state_dict:
        vocab_size = state_dict['output_projection.bias'].shape[0]

    # Build config from checkpoint (filter unknown keys)
    if config_dict:
        allowed = set(TKSNoeticLMConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in config_dict.items() if k in allowed}
        config = TKSNoeticLMConfig(**filtered)
    else:
        config = TKSNoeticLMConfig()
    config.vocab_size = vocab_size

    model = TKSNoeticLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, config


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_eval_data(
    data_path: str,
    num_samples: Optional[int] = None,
    seed: int = 42,
    split: Optional[str] = None,
) -> List[Dict]:
    """Load evaluation data from JSONL."""
    entries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Only include seq2seq format entries
                if 'input' in entry and 'target' in entry:
                    if split:
                        split_key = None
                        for key in ("split", "dataset_split", "data_split", "partition", "subset"):
                            if key in entry:
                                split_key = key
                                break
                        if split_key is None:
                            continue
                        if str(entry.get(split_key, "")).lower() != split.lower():
                            continue
                    entries.append(entry)

    if num_samples and num_samples < len(entries):
        random.seed(seed)
        entries = random.sample(entries, num_samples)

    return entries


# ==============================================================================
# GENERATION
# ==============================================================================

def greedy_generate_with_nll(
    model,
    prompt_ids: List[int],
    target_ids: List[int],
    max_new_tokens: int,
    device: torch.device,
    stop_ids: Optional[List[int]] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    """Greedy generation while scoring gold tokens under free-running context."""
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: List[int] = []
    nlls: List[float] = []

    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            if max_seq_len and tokens.shape[1] > max_seq_len:
                tokens = tokens[:, -max_seq_len:]

            out = model(tokens)
            logits = out["logits"][:, -1, :]

            if step < len(target_ids):
                log_probs = torch.log_softmax(logits, dim=-1)
                gold = torch.tensor([target_ids[step]], device=device)
                nll = -log_probs.gather(1, gold.view(-1, 1)).item()
                nlls.append(nll)

            next_token = logits.argmax(dim=-1, keepdim=True)
            if stop_ids and next_token.item() in stop_ids:
                break

            tokens = torch.cat([tokens, next_token], dim=1)
            generated.append(next_token.item())

    return generated, nlls


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_generation(
    model,
    tokenizer: TokenizerAdapter,
    eval_data: List[Dict],
    max_new_tokens: int = 128,
    device: torch.device = torch.device("cpu"),
    prompt_format: str = "arrow",
    separator: str = " => ",
    length_buckets: Optional[List[Tuple[str, int, int]]] = None,
    verbose: bool = False,
) -> Dict:
    """
    Run generation evaluation on held-out data.

    Returns dict with overall, per-task, and length-bucket metrics.
    """
    validator = CanonValidator()
    detailed_validator = CanonicalValidator(strict_mode=True)

    task_predictions = defaultdict(list)
    task_targets = defaultdict(list)
    task_eq_flags = defaultdict(list)
    all_predictions: List[str] = []
    all_targets: List[str] = []
    all_eq_flags: List[bool] = []

    examples = []

    bucket_store = {}
    if length_buckets:
        for name, _, _ in length_buckets:
            bucket_store[name] = {"preds": [], "targets": [], "eq_flags": []}

    nll_sum = 0.0
    nll_tokens = 0
    target_tokens_total = 0
    eq_nll_sum = 0.0
    eq_nll_tokens = 0
    eq_target_tokens_total = 0

    start_time = time.time()
    model_config = getattr(model, "config", None)
    max_seq_len = getattr(model_config, "max_seq_len", None)

    for i, entry in enumerate(eval_data):
        prompt = build_prompt(entry, prompt_format, separator)
        target = entry["target"]
        task_type = entry.get("task_type", entry.get("direction", "unknown"))
        eq_flag = is_equation_task(task_type, target)

        prompt_ids = tokenizer.encode_prompt(prompt)
        target_ids = tokenizer.encode_target(target)

        generated_ids, nlls = greedy_generate_with_nll(
            model,
            prompt_ids,
            target_ids,
            max_new_tokens=max_new_tokens,
            device=device,
            stop_ids=tokenizer.stop_ids,
            max_seq_len=max_seq_len,
        )
        generated = tokenizer.decode(generated_ids)

        all_predictions.append(generated)
        all_targets.append(target)
        all_eq_flags.append(eq_flag)
        task_predictions[task_type].append(generated)
        task_targets[task_type].append(target)
        task_eq_flags[task_type].append(eq_flag)

        nll_sum += sum(nlls)
        nll_tokens += len(nlls)
        target_tokens_total += len(target_ids)
        if eq_flag:
            eq_nll_sum += sum(nlls)
            eq_nll_tokens += len(nlls)
            eq_target_tokens_total += len(target_ids)

        if length_buckets:
            bucket_name = bucket_for_length(len(target_ids), length_buckets)
            if bucket_name:
                bucket_store[bucket_name]["preds"].append(generated)
                bucket_store[bucket_name]["targets"].append(target)
                bucket_store[bucket_name]["eq_flags"].append(eq_flag)

        is_exact = normalize_equation(generated) == normalize_equation(target)
        similarity = string_similarity(generated, target)
        syntax_valid = None
        validator_error = None
        if eq_flag:
            result = detailed_validator.validate(generated)
            syntax_valid = result.is_valid
            if not result.is_valid and result.issues:
                validator_error = "; ".join(issue.message for issue in result.issues[:3])

        examples.append({
            "idx": i,
            "task_type": task_type,
            "prompt": entry.get("input", "")[:100] + ("..." if len(entry.get("input", "")) > 100 else ""),
            "target": target,
            "generated": generated,
            "exact_match": is_exact,
            "string_similarity": similarity,
            "syntax_valid": syntax_valid,
            "validator_error": validator_error,
        })

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {i+1}/{len(eval_data)} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time

    overall_metrics = compute_mixed_metrics(all_predictions, all_targets, all_eq_flags, validator)
    overall_metrics["eval_time_seconds"] = elapsed
    overall_metrics["samples_per_second"] = len(eval_data) / max(elapsed, 1e-6)

    eq_predictions = [p for p, is_eq in zip(all_predictions, all_eq_flags) if is_eq]
    eq_targets = [t for t, is_eq in zip(all_targets, all_eq_flags) if is_eq]
    equation_only_metrics = (
        compute_generation_metrics(eq_predictions, eq_targets, validator=validator)
        if eq_predictions
        else None
    )

    per_task_metrics = {}
    for task_type in task_predictions:
        per_task_metrics[task_type] = compute_mixed_metrics(
            task_predictions[task_type],
            task_targets[task_type],
            task_eq_flags[task_type],
            validator,
        )

    length_bucket_metrics = {}
    if length_buckets:
        for name, _, _ in length_buckets:
            bucket = bucket_store.get(name, {"preds": [], "targets": [], "eq_flags": []})
            length_bucket_metrics[name] = compute_mixed_metrics(
                bucket["preds"],
                bucket["targets"],
                bucket["eq_flags"],
                validator,
            )

    fr_avg_nll = (nll_sum / nll_tokens) if nll_tokens else None
    fr_perplexity = math.exp(fr_avg_nll) if fr_avg_nll is not None else None
    fr_coverage = (nll_tokens / target_tokens_total) if target_tokens_total else None

    eq_fr_avg_nll = (eq_nll_sum / eq_nll_tokens) if eq_nll_tokens else None
    eq_fr_perplexity = math.exp(eq_fr_avg_nll) if eq_fr_avg_nll is not None else None
    eq_fr_coverage = (eq_nll_tokens / eq_target_tokens_total) if eq_target_tokens_total else None

    failures = [e for e in examples if not e["exact_match"]]
    failures_by_type = defaultdict(list)
    for f in failures:
        failures_by_type[f["task_type"]].append(f)

    def failure_key(item: Dict[str, Any]) -> Tuple[int, float]:
        valid = item.get("syntax_valid")
        invalid_rank = 0 if valid is False else 1
        return (invalid_rank, item.get("string_similarity", 1.0))

    top_failures = {}
    for task_type, items in failures_by_type.items():
        top_failures[task_type] = sorted(items, key=failure_key)[:5]

    return {
        "overall": overall_metrics,
        "overall_equation": equation_only_metrics,
        "per_task": per_task_metrics,
        "length_buckets": length_bucket_metrics,
        "free_running_xent": {
            "avg_nll": fr_avg_nll,
            "perplexity": fr_perplexity,
            "token_coverage": fr_coverage,
            "total_tokens": nll_tokens,
        },
        "free_running_xent_equation": {
            "avg_nll": eq_fr_avg_nll,
            "perplexity": eq_fr_perplexity,
            "token_coverage": eq_fr_coverage,
            "total_tokens": eq_nll_tokens,
        },
        "examples": examples[:20],
        "top_failures": top_failures,
        "metadata": {
            "num_samples": len(eval_data),
            "max_new_tokens": max_new_tokens,
            "eval_time": elapsed,
            "prompt_format": prompt_format,
            "separator": separator,
            "tokenizer_kind": tokenizer.kind,
        },
    }


# ==============================================================================
# REPORTING
# ==============================================================================

def generate_markdown_report(results: Dict, output_path: Path) -> str:
    """Generate markdown report from evaluation results."""
    def fmt_pct(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value*100:.2f}%"

    def fmt_float(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        "# TKSNoeticLM v4 Generation Evaluation",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Samples:** {results['metadata']['num_samples']}",
        f"**Max New Tokens:** {results['metadata']['max_new_tokens']}",
        f"**Eval Time:** {results['metadata']['eval_time']:.1f}s",
        f"**Prompt Format:** {results['metadata'].get('prompt_format', 'unknown')}",
        f"**Tokenizer:** {results['metadata'].get('tokenizer_kind', 'unknown')}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Exact Match | **{fmt_pct(results['overall']['exact_match_rate'])}** |",
        f"| Element F1 | {fmt_pct(results['overall']['avg_element_f1'])} |",
        f"| Operator F1 | {fmt_pct(results['overall']['avg_operator_f1'])} |",
        f"| String Similarity | {fmt_pct(results['overall']['avg_string_similarity'])} |",
        f"| Syntax Valid (equation outputs) | {fmt_pct(results['overall']['syntax_valid_rate'])} |",
        "",
    ]

    if results.get("overall_equation"):
        eq = results["overall_equation"]
        lines.extend([
            "## Equation-Only Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Exact Match | **{fmt_pct(eq['exact_match_rate'])}** |",
            f"| Element F1 | {fmt_pct(eq['avg_element_f1'])} |",
            f"| Operator F1 | {fmt_pct(eq['avg_operator_f1'])} |",
            f"| String Similarity | {fmt_pct(eq['avg_string_similarity'])} |",
            f"| Syntax Valid | {fmt_pct(eq['syntax_valid_rate'])} |",
            "",
        ])

    fr = results.get("free_running_xent", {})
    lines.extend([
        "## Free-Running Cross-Entropy",
        "",
        f"- **Avg NLL:** {fmt_float(fr.get('avg_nll'))}",
        f"- **Perplexity:** {fmt_float(fr.get('perplexity'))}",
        f"- **Token Coverage:** {fmt_pct(fr.get('token_coverage'))}",
        "",
    ])

    lines.extend([
        "## Per-Task Metrics",
        "",
    ])

    for task_type, metrics in results['per_task'].items():
        lines.extend([
            f"### {task_type}",
            "",
            f"- **Exact Match:** {fmt_pct(metrics['exact_match_rate'])} ({metrics['exact_match_count']}/{metrics['total_examples']})",
            f"- **Element F1:** {fmt_pct(metrics['avg_element_f1'])}",
            f"- **Operator F1:** {fmt_pct(metrics['avg_operator_f1'])}",
            f"- **String Similarity:** {fmt_pct(metrics['avg_string_similarity'])}",
            f"- **Syntax Valid:** {fmt_pct(metrics['syntax_valid_rate'])}",
            "",
        ])

    if results.get("length_buckets"):
        lines.extend([
            "## Length Buckets",
            "",
            "| Bucket | Examples | Exact Match | String Similarity | Element F1 | Operator F1 | Syntax Valid |",
            "|--------|----------|-------------|-------------------|------------|-------------|--------------|",
        ])
        for bucket_name, metrics in results["length_buckets"].items():
            lines.append(
                f"| {bucket_name} | {metrics['total_examples']} | "
                f"{fmt_pct(metrics['exact_match_rate'])} | "
                f"{fmt_pct(metrics['avg_string_similarity'])} | "
                f"{fmt_pct(metrics['avg_element_f1'])} | "
                f"{fmt_pct(metrics['avg_operator_f1'])} | "
                f"{fmt_pct(metrics['syntax_valid_rate'])} |"
            )
        lines.append("")

    # Top failures
    lines.extend([
        "## Top Failures",
        "",
    ])

    for task_type, failures in results['top_failures'].items():
        if failures:
            lines.append(f"### {task_type}")
            lines.append("")
            for i, f in enumerate(failures[:3]):
                lines.extend([
                    f"**Failure {i+1}:**",
                    f"- Prompt: `{f['prompt']}`",
                    f"- Target: `{f['target']}`",
                    f"- Generated: `{f['generated']}`",
                    f"- Syntax Valid: {f['syntax_valid']}",
                    f"- Validator Error: {f.get('validator_error')}",
                    "",
                ])

    report = '\n'.join(lines)

    # Write to file
    md_path = output_path.with_suffix('.md')
    md_path.write_text(report, encoding='utf-8')

    return report


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generation Evaluation for TKSNoeticLM v4")
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', required=True, help='Path to evaluation data JSONL')
    parser.add_argument('--output', default='output/reports/generation_eval', help='Output path prefix')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='Max tokens to generate')
    parser.add_argument('--max-new-chars', type=int, default=None, help='Alias for --max-new-tokens')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples (None=all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--split', type=str, default=None, help='Filter by split field (eval/test)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prompt-format', default='auto', choices=['auto', 'arrow', 'tagged', 'raw'])
    parser.add_argument('--separator', default=' => ', help='Separator for arrow format')
    parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer.json (optional)')
    parser.add_argument('--tokenizer-type', default='auto', choices=['auto', 'byte', 'simple'])
    parser.add_argument('--length-buckets', type=str, default=None, help='Bucket spec, e.g. 0-32,33-64,65-128,129-1000000')
    parser.add_argument('--no-length-buckets', action='store_true', help='Disable length bucket metrics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    max_new_tokens = args.max_new_chars if args.max_new_chars is not None else args.max_new_tokens
    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TKSNoeticLM v4 Generation Evaluation")
    print("=" * 70)

    # Load model
    ckpt_path = Path(args.checkpoint)
    model_path, config_path, tokenizer_path = resolve_checkpoint_assets(ckpt_path)
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)

    print(f"\nLoading model from {model_path}...")
    model, config = load_v4_model(str(model_path), device, config_path=config_path)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create tokenizer
    tokenizer_kind = args.tokenizer_type
    tokenizer_meta = load_json_if_exists(tokenizer_path) if tokenizer_path else {}
    if tokenizer_kind == "auto":
        tokenizer_kind = "byte" if tokenizer_meta.get("type") == "byte" else "simple"

    if tokenizer_kind == "byte":
        if tokenizer_path:
            tok = ByteTokenizer.from_json(tokenizer_path)
        else:
            tok = ByteTokenizer()
        tokenizer = TokenizerAdapter(tok, "byte", max_length=getattr(config, "max_seq_len", None))
        print(f"  Tokenizer: byte ({tok.vocab_size} tokens)")
        if config.vocab_size != tok.vocab_size:
            print(f"  Warning: tokenizer vocab {tok.vocab_size} != model vocab {config.vocab_size}")
    else:
        tok = SimpleTokenizer(vocab_size=1000, max_length=config.max_seq_len)
        tokenizer = TokenizerAdapter(tok, "simple", max_length=config.max_seq_len)
        print(f"  Tokenizer: simple ({tok.actual_vocab_size} tokens)")
        if config.vocab_size != tok.actual_vocab_size:
            print(f"  Warning: tokenizer vocab {tok.actual_vocab_size} != model vocab {config.vocab_size}")

    # Load data
    print(f"\nLoading eval data from {args.data}...")
    eval_data = load_eval_data(args.data, num_samples=args.num_samples, seed=args.seed, split=args.split)
    print(f"  Loaded {len(eval_data)} samples")

    # Task type distribution
    task_dist = defaultdict(int)
    for e in eval_data:
        task_dist[e.get('task_type', e.get('direction', 'unknown'))] += 1
    for task, count in task_dist.items():
        print(f"    {task}: {count}")

    prompt_format = choose_prompt_format(args.prompt_format, tokenizer.kind, eval_data[0] if eval_data else None)
    length_buckets = None if args.no_length_buckets else parse_length_buckets(args.length_buckets)

    # Run evaluation
    print(f"\nRunning generation evaluation (max_new_tokens={max_new_tokens})...")
    results = evaluate_generation(
        model, tokenizer, eval_data,
        max_new_tokens=max_new_tokens,
        device=device,
        prompt_format=prompt_format,
        separator=args.separator,
        length_buckets=length_buckets,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def fmt_pct(value: Optional[float]) -> str:
        return "  N/A" if value is None else f"{value*100:6.2f}%"

    overall = results['overall']
    print(f"\nOverall ({results['metadata']['num_samples']} samples):")
    print(f"  Exact Match:      {fmt_pct(overall['exact_match_rate'])} ({overall['exact_match_count']}/{overall['total_examples']})")
    print(f"  Element F1:       {fmt_pct(overall['avg_element_f1'])}")
    print(f"  Operator F1:      {fmt_pct(overall['avg_operator_f1'])}")
    print(f"  String Similarity:{fmt_pct(overall['avg_string_similarity'])}")
    print(f"  Syntax Valid:     {fmt_pct(overall['syntax_valid_rate'])}")

    if results.get("overall_equation"):
        eq = results["overall_equation"]
        print("\nEquation-Only:")
        print(f"  Exact Match:      {fmt_pct(eq['exact_match_rate'])} ({eq['exact_match_count']}/{eq['total_examples']})")
        print(f"  Element F1:       {fmt_pct(eq['avg_element_f1'])}")
        print(f"  Operator F1:      {fmt_pct(eq['avg_operator_f1'])}")
        print(f"  String Similarity:{fmt_pct(eq['avg_string_similarity'])}")
        print(f"  Syntax Valid:     {fmt_pct(eq['syntax_valid_rate'])}")

    fr = results.get("free_running_xent", {})
    if fr:
        print("\nFree-Running Cross-Entropy:")
        avg_nll = fr.get("avg_nll")
        ppl = fr.get("perplexity")
        coverage = fr.get("token_coverage")
        print(f"  Avg NLL:          {avg_nll:.4f}" if avg_nll is not None else "  Avg NLL:          N/A")
        print(f"  Perplexity:       {ppl:.4f}" if ppl is not None else "  Perplexity:       N/A")
        print(f"  Token Coverage:   {coverage*100:6.2f}%" if coverage is not None else "  Token Coverage:   N/A")

    print(f"\nPer-Task:")
    for task_type, metrics in results['per_task'].items():
        print(f"\n  {task_type}:")
        print(f"    Exact Match: {fmt_pct(metrics['exact_match_rate'])} ({metrics['exact_match_count']}/{metrics['total_examples']})")
        print(f"    Element F1:  {fmt_pct(metrics['avg_element_f1'])}")
        print(f"    Syntax Valid:{fmt_pct(metrics['syntax_valid_rate'])}")

    # Save results
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Generate markdown report
    report = generate_markdown_report(results, output_path)
    print(f"Report saved to: {output_path.with_suffix('.md')}")

    # Print top failures
    print("\n" + "-" * 70)
    print("TOP FAILURES (for debugging)")
    print("-" * 70)
    for task_type, failures in results['top_failures'].items():
        if failures:
            print(f"\n{task_type}:")
            for f in failures[:2]:
                print(f"  Target:    {f['target'][:60]}...")
                print(f"  Generated: {f['generated'][:60]}...")
                if f.get("validator_error"):
                    print(f"  Validator: {f['validator_error']}")
                print()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())

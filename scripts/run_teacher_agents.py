#!/usr/bin/env python3
"""
Fan-out teacher generation (WSL/Linux friendly).

Generates TKS training data by running each provider independently across
chunked equation inputs, then combines outputs into a single JSONL.

This mirrors scripts/run_teacher_agents.ps1 without PowerShell dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher import MultiLLMTeacher, TeacherConfig, TKSEquation, TaskType


@dataclass
class ChunkData:
    index: int
    equations: List[TKSEquation]
    source: Optional[Path] = None


def parse_providers(specs: List[str]) -> List[Dict[str, str]]:
    providers = []
    for spec in specs:
        parts = spec.split(":", 1)
        if len(parts) == 2:
            providers.append({"type": parts[0], "model": parts[1]})
        else:
            providers.append({"type": parts[0], "model": "default"})
    return providers


def load_keys_from_config(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.getenv(key):
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if not line.startswith(key):
                continue
            parts = line.split("=", 1)
            if len(parts) != 2:
                continue
            raw = parts[1].strip()
            if raw.startswith(("\"", "'")) and raw.endswith(("\"", "'")):
                os.environ[key] = raw[1:-1]
            break


def parse_equation_string(equation_str: str) -> List[str]:
    elements = re.findall(r"[ABCD]\d{1,2}(?:\^\d)?(?:_d[1-7])?", equation_str.upper())
    return elements


def load_equations(path: Path, limit: Optional[int] = None) -> List[TKSEquation]:
    equations: List[TKSEquation] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(equations) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "elements" in entry and entry["elements"]:
                equations.append(TKSEquation(elements=entry["elements"]))
                continue
            if "equation" in entry:
                elements = parse_equation_string(entry["equation"])
                if elements:
                    equations.append(TKSEquation(elements=elements))
    return equations


def parse_chunk_index(path: Path) -> Optional[int]:
    match = re.match(r"equations_(\d+)\.jsonl$", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_chunk_files(
    chunk_dir: Path,
    chunk_indices: Optional[List[int]] = None,
    chunk_start: Optional[int] = None,
    chunk_end: Optional[int] = None,
) -> List[Path]:
    files: List[Path] = []
    if not chunk_dir.exists():
        return files
    indexed = []
    for path in chunk_dir.iterdir():
        if not path.is_file():
            continue
        idx = parse_chunk_index(path)
        if idx is None:
            continue
        indexed.append((idx, path))
    indexed.sort(key=lambda item: item[0])

    if chunk_indices:
        index_set = set(chunk_indices)
        indexed = [item for item in indexed if item[0] in index_set]
    elif chunk_start is not None or chunk_end is not None:
        start = chunk_start if chunk_start is not None else 0
        end = chunk_end if chunk_end is not None else (indexed[-1][0] if indexed else -1)
        indexed = [item for item in indexed if start <= item[0] <= end]

    for _, path in indexed:
        files.append(path)
    return files


def split_chunks(items: List[TKSEquation], chunks: int) -> List[List[TKSEquation]]:
    if chunks <= 1:
        return [items]
    size = math.ceil(len(items) / chunks)
    return [items[i * size : (i + 1) * size] for i in range(chunks)]


def parse_tasks(task_list: Optional[List[str]]) -> Optional[List[TaskType]]:
    if not task_list:
        return None
    task_map = {
        "e2i": TaskType.E2I,
        "i2e": TaskType.I2E,
        "s2e": TaskType.S2E,
        "e2rpm": TaskType.E2RPM,
        "e2f": TaskType.E2F,
        "full": TaskType.FULL,
    }
    tasks = []
    for t in task_list:
        key = t.lower()
        if key in task_map:
            tasks.append(task_map[key])
    return tasks or None


def write_examples(path: Path, examples) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def combine_jsonl(paths: List[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        total += 1
    return total


def append_jsonl(target: Path, sources: List[Path]) -> int:
    total = 0
    with target.open("a", encoding="utf-8") as out:
        for src in sources:
            if not src.exists():
                continue
            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        total += 1
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-provider teacher generation (WSL/Linux).")
    parser.add_argument("--data", default="data/equations.jsonl", help="Input equations JSONL")
    parser.add_argument("--output-dir", default="output", help="Base output directory")
    parser.add_argument("--chunks", type=int, default=5, help="Number of chunks")
    parser.add_argument("--chunk-dir", default=None, help="Directory of pre-split chunks (equations_<idx>.jsonl)")
    parser.add_argument("--chunk-indices", nargs="+", type=int, default=None, help="Specific chunk indices to process")
    parser.add_argument("--chunk-start", type=int, default=None, help="Start chunk index (inclusive)")
    parser.add_argument("--chunk-end", type=int, default=None, help="End chunk index (inclusive)")
    parser.add_argument("--providers", nargs="+", default=None, help="Providers (type:model)")
    parser.add_argument("--min-canon", type=float, default=0.8, help="Minimum canon score")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence score")
    parser.add_argument("--tasks", nargs="+", help="Task types (e2i, i2e, s2e, e2rpm, e2f, full)")
    parser.add_argument("--limit", type=int, default=None, help="Limit equations processed")
    parser.add_argument("--append-fractals", action="store_true", help="Append NF teacher dataset")
    parser.add_argument("--fractals-path", default="data/noetic_fractals_complete_teacher.jsonl", help="NF teacher JSONL")
    parser.add_argument("--keys-from-config", default=None, help="Optional config file with API keys")
    parser.add_argument("--skip-combine", action="store_true", help="Skip combining outputs into teacher_all.jsonl")
    args = parser.parse_args()

    if (args.chunk_indices or args.chunk_start is not None or args.chunk_end is not None) and not args.chunk_dir:
        print("Chunk selection options require --chunk-dir.")
        return 1

    data_path = Path(args.data)
    if not args.chunk_dir and not data_path.exists():
        print(f"Input file not found: {data_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.keys_from_config:
        load_keys_from_config(Path(args.keys_from_config))

    providers = args.providers or [
        "gemini:gemini-2.5-pro",
        "anthropic:claude-opus-4-5",
    ]

    missing = []
    for spec in providers:
        ptype = spec.split(":", 1)[0]
        if ptype == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if ptype == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            missing.append("GEMINI_API_KEY")
        if ptype == "openai" and not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
    if missing:
        unique = sorted(set(missing))
        print("WARNING: Missing API keys:", ", ".join(unique))
        print("Set env vars or pass --keys-from-config <path> before running large jobs.")

    chunk_data: List[ChunkData] = []
    total_equations = 0
    if args.chunk_dir:
        chunk_dir = Path(args.chunk_dir)
        chunk_files = load_chunk_files(
            chunk_dir,
            chunk_indices=args.chunk_indices,
            chunk_start=args.chunk_start,
            chunk_end=args.chunk_end,
        )
        if not chunk_files:
            print(f"No chunk files found in: {chunk_dir}")
            return 1
        for path in chunk_files:
            idx = parse_chunk_index(path)
            if idx is None:
                continue
            equations = load_equations(path, limit=args.limit)
            if not equations:
                print(f"Skipping empty chunk file: {path}")
                continue
            chunk_data.append(ChunkData(index=idx, equations=equations, source=path))
            total_equations += len(equations)
    else:
        equations = load_equations(data_path, limit=args.limit)
        if not equations:
            print("No equations loaded; check input format.")
            return 1
        chunks = split_chunks(equations, args.chunks)
        for idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            chunk_data.append(ChunkData(index=idx, equations=chunk))
            total_equations += len(chunk)

    if not chunk_data:
        print("No equations loaded after chunk filtering.")
        return 1

    tasks = parse_tasks(args.tasks)

    print("=" * 70)
    print("TKS MULTI-PROVIDER TEACHER GENERATION (PYTHON)")
    print("=" * 70)
    if args.chunk_dir:
        print(f"Input chunks: {args.chunk_dir} ({len(chunk_data)} chunks, {total_equations} equations)")
    else:
        print(f"Input: {data_path} ({total_equations} equations)")
    print(f"Chunks: {len(chunk_data)}")
    print(f"Providers: {', '.join(providers)}")
    print(f"Min canon: {args.min_canon}")
    print(f"Min confidence: {args.min_confidence}")

    output_files: List[Path] = []
    for prov_spec in providers:
        prov_type = prov_spec.split(":", 1)[0]
        prov_config = parse_providers([prov_spec])[0]

        for counter, chunk in enumerate(chunk_data, start=1):
            out_path = output_dir / f"teacher_{prov_type}_{chunk.index}.jsonl"
            source = f" ({chunk.source.name})" if chunk.source else ""
            print(f"\n[{prov_type}] chunk {counter}/{len(chunk_data)} (id {chunk.index}){source} -> {out_path}")

            config = TeacherConfig(
                providers=[prov_config],
                min_canon_score=args.min_canon,
                min_confidence_score=args.min_confidence,
                strict_validation=True,
            )
            teacher = MultiLLMTeacher(config)
            examples = teacher.generate_training_data(chunk.equations, task_types=tasks, show_progress=True)

            for ex in examples:
                ex.metadata["provider"] = prov_type
                ex.metadata["provider_model"] = prov_config.get("model", "default")
                ex.metadata["chunk"] = chunk.index

            count = write_examples(out_path, examples)
            output_files.append(out_path)
            print(f"  wrote {count} examples")

    if args.skip_combine:
        if args.append_fractals:
            print("Skipping append-fractals because --skip-combine is set.")
        return 0

    combined_path = output_dir / "teacher_all.jsonl"
    total = combine_jsonl(output_files, combined_path)
    print(f"\nCombined: {combined_path} ({total} entries)")

    if args.append_fractals:
        fractals_path = Path(args.fractals_path)
        if fractals_path.exists():
            added = append_jsonl(combined_path, [fractals_path])
            print(f"Appended NF dataset: {fractals_path} (+{added} entries)")
        else:
            print(f"NF dataset not found: {fractals_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

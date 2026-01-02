#!/usr/bin/env python3
"""
Supervisor for multi-worker teacher generation.

Splits equation seeds into chunks, launches worker processes, verifies outputs,
then merges results into a single JSONL.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def parse_chunk_index(path: Path) -> Optional[int]:
    match = re.match(r"equations_(\d+)\.jsonl$", path.name)
    if not match:
        return None
    return int(match.group(1))


def list_chunk_files(chunk_dir: Path) -> List[Path]:
    indexed: List[Tuple[int, Path]] = []
    for path in chunk_dir.iterdir():
        if not path.is_file():
            continue
        idx = parse_chunk_index(path)
        if idx is None:
            continue
        indexed.append((idx, path))
    indexed.sort(key=lambda item: item[0])
    return [path for _, path in indexed]


def count_non_empty_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def split_jsonl(input_path: Path, chunk_dir: Path, chunks: int) -> Tuple[List[Path], int]:
    total_lines = count_non_empty_lines(input_path)
    if total_lines == 0:
        return [], 0

    chunk_count = min(chunks, total_lines)
    base = total_lines // chunk_count
    extra = total_lines % chunk_count
    sizes = [base + (1 if i < extra else 0) for i in range(chunk_count)]

    chunk_files: List[Path] = []
    chunk_idx = 0
    remaining = sizes[0]
    out_path = chunk_dir / f"equations_{chunk_idx}.jsonl"
    out = out_path.open("w", encoding="utf-8")
    chunk_files.append(out_path)

    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                out.write(line)
                remaining -= 1
                if remaining == 0:
                    out.close()
                    chunk_idx += 1
                    if chunk_idx >= chunk_count:
                        break
                    remaining = sizes[chunk_idx]
                    out_path = chunk_dir / f"equations_{chunk_idx}.jsonl"
                    out = out_path.open("w", encoding="utf-8")
                    chunk_files.append(out_path)
    finally:
        if not out.closed:
            out.close()

    return chunk_files, total_lines


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


def provider_tags(providers: List[str]) -> List[str]:
    tags = []
    for spec in providers:
        tags.append(spec.split(":", 1)[0])
    return tags


def build_worker_ranges(chunk_indices: List[int], workers: int) -> List[Tuple[int, int, int]]:
    ranges: List[Tuple[int, int, int]] = []
    if not chunk_indices:
        return ranges
    per_worker = math.ceil(len(chunk_indices) / workers)
    for worker_id in range(workers):
        start_pos = worker_id * per_worker
        end_pos = min((worker_id + 1) * per_worker, len(chunk_indices)) - 1
        if start_pos > end_pos:
            continue
        start_idx = chunk_indices[start_pos]
        end_idx = chunk_indices[end_pos]
        ranges.append((worker_id, start_idx, end_idx))
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervisor for multi-worker teacher generation.")
    parser.add_argument("--data", default="data/equations.jsonl", help="Input equations JSONL")
    parser.add_argument("--output-dir", default="output", help="Base output directory for teacher outputs")
    parser.add_argument("--chunk-dir", default=None, help="Directory for chunk files (default: <output-dir>/chunks)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--chunks", type=int, default=None, help="Total chunks (default: workers * chunks-per-worker)")
    parser.add_argument("--chunks-per-worker", type=int, default=10, help="Chunks per worker when --chunks is not set")
    parser.add_argument("--providers", nargs="+", default=None, help="Providers (type:model)")
    parser.add_argument("--min-canon", type=float, default=0.8, help="Minimum canon score")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence score")
    parser.add_argument("--tasks", nargs="+", help="Task types (e2i, i2e, s2e, e2rpm, e2f, full)")
    parser.add_argument("--limit", type=int, default=None, help="Limit equations per chunk (debug)")
    parser.add_argument("--append-fractals", action="store_true", help="Append NF teacher dataset")
    parser.add_argument("--fractals-path", default="data/noetic_fractals_complete_teacher.jsonl", help="NF teacher JSONL")
    parser.add_argument("--keys-from-config", default=None, help="Optional config file with API keys")
    parser.add_argument("--python", dest="python_bin", default=None, help="Python executable for workers")
    parser.add_argument("--log-dir", default=None, help="Directory for worker logs (default: <output-dir>/logs)")
    parser.add_argument("--reuse-chunks", action="store_true", help="Reuse existing chunk files")
    parser.add_argument("--overwrite-chunks", action="store_true", help="Overwrite existing chunk files")
    parser.add_argument("--dry-run", action="store_true", help="Print worker commands and exit")
    args = parser.parse_args()

    if args.reuse_chunks and args.overwrite_chunks:
        print("Use only one of --reuse-chunks or --overwrite-chunks.")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Input file not found: {data_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else (output_dir / "chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    existing_chunks = list_chunk_files(chunk_dir)
    chunk_files: List[Path] = []
    total_lines = 0

    if existing_chunks and not args.reuse_chunks and not args.overwrite_chunks:
        print(f"Chunk files already exist in: {chunk_dir}")
        print("Use --reuse-chunks to reuse or --overwrite-chunks to regenerate.")
        return 1

    if args.reuse_chunks:
        chunk_files = existing_chunks
        if not chunk_files:
            print(f"No chunk files found to reuse in: {chunk_dir}")
            return 1
    else:
        chunks = args.chunks if args.chunks is not None else (args.workers * args.chunks_per_worker)
        chunk_files, total_lines = split_jsonl(data_path, chunk_dir, chunks)
        if not chunk_files:
            print("No equations found to split.")
            return 1

    chunk_indices = [parse_chunk_index(path) for path in chunk_files]
    chunk_indices = sorted([idx for idx in chunk_indices if idx is not None])

    workers = max(1, args.workers)
    ranges = build_worker_ranges(chunk_indices, workers)
    if not ranges:
        print("No worker ranges available.")
        return 1

    providers = args.providers or [
        "gemini:gemini-2.5-pro",
        "anthropic:claude-opus-4-5",
    ]

    python_bin = args.python_bin or sys.executable
    worker_script = Path(__file__).parent / "run_teacher_agents.py"
    log_dir = Path(args.log_dir) if args.log_dir else (output_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TKS TEACHER SUPERVISOR")
    print("=" * 70)
    if total_lines:
        print(f"Input: {data_path} ({total_lines} equations)")
    else:
        print(f"Input: {data_path}")
    print(f"Chunk dir: {chunk_dir}")
    print(f"Chunks: {len(chunk_indices)}")
    print(f"Workers: {len(ranges)}")
    print(f"Providers: {', '.join(providers)}")
    print(f"Output dir: {output_dir}")
    print("")

    commands = []
    for worker_id, start_idx, end_idx in ranges:
        cmd = [
            python_bin,
            str(worker_script),
            "--chunk-dir",
            str(chunk_dir),
            "--chunk-start",
            str(start_idx),
            "--chunk-end",
            str(end_idx),
            "--output-dir",
            str(output_dir),
            "--min-canon",
            str(args.min_canon),
            "--min-confidence",
            str(args.min_confidence),
            "--skip-combine",
        ]
        if args.providers:
            cmd.extend(["--providers", *providers])
        if args.tasks:
            cmd.extend(["--tasks", *args.tasks])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.keys_from_config:
            cmd.extend(["--keys-from-config", args.keys_from_config])
        commands.append((worker_id, start_idx, end_idx, cmd))

    if args.dry_run:
        for worker_id, start_idx, end_idx, cmd in commands:
            print(f"Worker {worker_id}: chunks {start_idx}-{end_idx}")
            print("  " + " ".join(cmd))
        return 0

    procs = []
    for worker_id, start_idx, end_idx, cmd in commands:
        log_path = log_dir / f"worker_{worker_id}.log"
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(f"Command: {' '.join(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((worker_id, proc, log_file, log_path, start_idx, end_idx))
        print(f"Started worker {worker_id} (chunks {start_idx}-{end_idx}) -> {log_path}")

    failed = False
    for worker_id, proc, log_file, log_path, start_idx, end_idx in procs:
        code = proc.wait()
        log_file.close()
        if code != 0:
            failed = True
            print(f"Worker {worker_id} failed (chunks {start_idx}-{end_idx}). See {log_path}")

    provider_list = provider_tags(providers)
    expected = len(chunk_indices) * len(provider_list)
    output_files: List[Path] = []
    missing = []
    for tag in provider_list:
        for idx in chunk_indices:
            path = output_dir / f"teacher_{tag}_{idx}.jsonl"
            if path.exists():
                output_files.append(path)
            else:
                missing.append(str(path))

    print("")
    print("Supervisor check:")
    print(f"  Expected output files: {expected}")
    print(f"  Found output files: {len(output_files)}")
    if missing:
        print(f"  Missing outputs: {len(missing)}")
        failed = True
    else:
        print("  Missing outputs: 0")

    combined_path = output_dir / "teacher_all.jsonl"
    total = combine_jsonl(output_files, combined_path)
    print(f"Combined: {combined_path} ({total} entries)")

    if args.append_fractals:
        fractals_path = Path(args.fractals_path)
        if fractals_path.exists():
            added = append_jsonl(combined_path, [fractals_path])
            print(f"Appended NF dataset: {fractals_path} (+{added} entries)")
        else:
            print(f"NF dataset not found: {fractals_path}")

    if missing:
        print("Missing outputs (first 10):")
        for path in missing[:10]:
            print(f"  {path}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

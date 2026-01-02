#!/usr/bin/env python3
"""
Resume missing teacher outputs by re-running only the missing chunks.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_PROVIDERS = [
    "gemini:gemini-2.5-pro",
    "anthropic:claude-opus-4-5",
]


def parse_chunk_index(path: Path) -> Optional[int]:
    match = re.match(r"equations_(\d+)\.jsonl$", path.name)
    if not match:
        return None
    return int(match.group(1))


def list_chunk_indices(chunk_dir: Path) -> List[int]:
    indices = []
    if not chunk_dir.exists():
        return indices
    for path in chunk_dir.iterdir():
        if not path.is_file():
            continue
        idx = parse_chunk_index(path)
        if idx is None:
            continue
        indices.append(idx)
    return sorted(indices)


def provider_tag(spec: str) -> str:
    return spec.split(":", 1)[0]


def chunked(items: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_missing(
    output_dir: Path,
    chunk_indices: List[int],
    providers: List[str],
) -> Dict[str, List[int]]:
    missing: Dict[str, List[int]] = {}
    for spec in providers:
        tag = provider_tag(spec)
        missing[tag] = []
        for idx in chunk_indices:
            path = output_dir / f"teacher_{tag}_{idx}.jsonl"
            if not path.exists():
                missing[tag].append(idx)
    return missing


def run_batches(
    commands: List[Tuple[str, List[int], List[str]]],
    max_parallel: int,
    log_dir: Path,
    dry_run: bool,
) -> int:
    if dry_run:
        for label, indices, cmd in commands:
            print(f"{label}: chunks {indices[0]}-{indices[-1]} ({len(indices)} chunks)")
            print("  " + " ".join(cmd))
        return 0

    failed = False
    for i in range(0, len(commands), max_parallel):
        batch = commands[i : i + max_parallel]
        procs = []
        for label, indices, cmd in batch:
            log_path = log_dir / f"resume_{label}.log"
            log_file = log_path.open("w", encoding="utf-8")
            log_file.write(f"Command: {' '.join(cmd)}\n\n")
            log_file.flush()
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((label, proc, log_file, log_path))
            print(f"Started resume {label} -> {log_path}")

        for label, proc, log_file, log_path in procs:
            code = proc.wait()
            log_file.close()
            if code != 0:
                failed = True
                print(f"Resume batch failed: {label}. See {log_path}")

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume missing teacher outputs.")
    parser.add_argument("--output-dir", default="output", help="Output directory used by supervisor/workers")
    parser.add_argument("--chunk-dir", default=None, help="Chunk directory (default: <output-dir>/chunks)")
    parser.add_argument("--providers", nargs="+", default=None, help="Providers (type:model)")
    parser.add_argument("--min-canon", type=float, default=0.8, help="Minimum canon score")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence score")
    parser.add_argument("--tasks", nargs="+", help="Task types (e2i, i2e, s2e, e2rpm, e2f, full)")
    parser.add_argument("--limit", type=int, default=None, help="Limit equations per chunk (debug)")
    parser.add_argument("--keys-from-config", default=None, help="Optional config file with API keys")
    parser.add_argument("--python", dest="python_bin", default=None, help="Python executable for workers")
    parser.add_argument("--log-dir", default=None, help="Directory for resume logs (default: <output-dir>/logs)")
    parser.add_argument("--batch-size", type=int, default=10, help="Missing chunk batch size per resume call")
    parser.add_argument("--max-parallel", type=int, default=1, help="Parallel resume processes")
    parser.add_argument("--dry-run", action="store_true", help="Print resume commands and exit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else (output_dir / "chunks")
    log_dir = Path(args.log_dir) if args.log_dir else (output_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    chunk_indices = list_chunk_indices(chunk_dir)
    if not chunk_indices:
        print(f"No chunk files found in: {chunk_dir}")
        return 1

    providers = args.providers or DEFAULT_PROVIDERS
    missing = build_missing(output_dir, chunk_indices, providers)

    if all(not items for items in missing.values()):
        print("No missing outputs detected.")
        return 0

    python_bin = args.python_bin or sys.executable
    worker_script = Path(__file__).parent / "run_teacher_agents.py"

    commands: List[Tuple[str, List[int], List[str]]] = []
    for spec in providers:
        tag = provider_tag(spec)
        indices = missing.get(tag, [])
        if not indices:
            continue
        for batch_num, batch in enumerate(chunked(indices, args.batch_size), start=1):
            label = f"{tag}_batch{batch_num:03d}"
            cmd = [
                python_bin,
                str(worker_script),
                "--chunk-dir",
                str(chunk_dir),
                "--chunk-indices",
                *[str(idx) for idx in batch],
                "--output-dir",
                str(output_dir),
                "--min-canon",
                str(args.min_canon),
                "--min-confidence",
                str(args.min_confidence),
                "--skip-combine",
                "--providers",
                spec,
            ]
            if args.tasks:
                cmd.extend(["--tasks", *args.tasks])
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            if args.keys_from_config:
                cmd.extend(["--keys-from-config", args.keys_from_config])
            commands.append((label, batch, cmd))

    if not commands:
        print("No resume commands to run.")
        return 0

    return run_batches(commands, max(1, args.max_parallel), log_dir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

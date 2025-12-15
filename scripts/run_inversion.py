#!/usr/bin/env python3
"""
TKS Total Inversion Engine CLI
Usage:
  python scripts/run_inversion.py "B5,D3,C3,D5" --ops "+T,->,-T" --mode Opposite
  python scripts/run_inversion.py "B5,D3,C3,D5" --ops "+T,->,-T" --mode ReverseCausal
  python scripts/run_inversion.py "B5,D3,C3,D5" --ops "+T,->,-T" --mode ParallelAnalogue --from-world B --to-world D
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inversion.engine import total_inversion, DialConfig, TargetProfile, InversionMode


def parse_args():
    p = argparse.ArgumentParser(description="Run TKS Total Inversion Engine.")
    p.add_argument("elements", help="Comma-separated elements, e.g., B5,D3,C3,D5")
    p.add_argument("--ops", help="Comma-separated ops between elements, e.g., +T,->,-T", default="")
    p.add_argument("--mode", default="Opposite", help="Inversion mode name")
    p.add_argument("--from-world", dest="from_world", help="Source world for Parallel/Domain remaps")
    p.add_argument("--to-world", dest="to_world", help="Target world for Parallel/Domain remaps")
    p.add_argument("--intensity", default="soft", choices=["soft", "medium", "hard"])
    p.add_argument("--scope", default="equation", choices=["local", "term", "chain", "equation", "scenario"])
    return p.parse_args()


def main():
    args = parse_args()
    elements = [e.strip() for e in args.elements.split(",") if e.strip()]
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    target = TargetProfile(
        enable=bool(args.from_world and args.to_world),
        from_world=args.from_world,
        to_world=args.to_world,
    )
    dial = DialConfig(
        mode=InversionMode(args.mode),
        intensity=args.intensity,
        scope=args.scope,
        target_profile=target,
    )
    result = total_inversion(
        elements=elements,
        ops=ops,
        mode=dial.mode,
        dial=dial,
    )
    print("=== INPUT ===")
    print("Elements:", elements)
    print("Ops:", ops)
    print("\n=== OUTPUT ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

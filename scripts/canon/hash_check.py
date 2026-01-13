#!/usr/bin/env python3
"""
Canon Hash Check Script - Verify integrity of canon files.

Usage:
    python scripts/canon/hash_check.py [--canon canon/normalized]

This script:
1. Reads canon_index.json for stored file hashes
2. Computes current SHA-256 hashes of all referenced files
3. Reports any mismatches between stored and computed hashes
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_canon_index(canon_dir: Path) -> Dict:
    """Load the canon_index.json file."""
    index_path = canon_dir / "canon_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Canon index not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_hashes(index: Dict, verbose: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """
    Check all file hashes against stored values.

    Returns:
        Tuple of (matches, mismatches, missing)
    """
    matches = []
    mismatches = []
    missing = []

    # Check normalized files
    for filename, info in index.get("normalized", {}).items():
        filepath = PROJECT_ROOT / info["path"]
        stored_hash = info["hash"]

        if not filepath.exists():
            missing.append(f"normalized/{filename}")
            continue

        computed_hash = compute_file_hash(filepath)

        if computed_hash == stored_hash:
            matches.append(f"normalized/{filename}")
            if verbose:
                print(f"  [MATCH] {filename}")
        else:
            mismatches.append({
                "file": f"normalized/{filename}",
                "stored": stored_hash,
                "computed": computed_hash
            })
            if verbose:
                print(f"  [MISMATCH] {filename}")
                print(f"    Stored:   {stored_hash[:16]}...")
                print(f"    Computed: {computed_hash[:16]}...")

    # Check gold files
    for filename, info in index.get("gold", {}).items():
        filepath = PROJECT_ROOT / info["path"]
        stored_hash = info["hash"]

        if not filepath.exists():
            missing.append(f"gold/{filename}")
            continue

        computed_hash = compute_file_hash(filepath)

        if computed_hash == stored_hash:
            matches.append(f"gold/{filename}")
            if verbose:
                print(f"  [MATCH] {filename}")
        else:
            mismatches.append({
                "file": f"gold/{filename}",
                "stored": stored_hash,
                "computed": computed_hash
            })
            if verbose:
                print(f"  [MISMATCH] {filename}")
                print(f"    Stored:   {stored_hash[:16]}...")
                print(f"    Computed: {computed_hash[:16]}...")

    return matches, mismatches, missing


def main():
    parser = argparse.ArgumentParser(
        description="Verify integrity of TKS canon files by checking hashes"
    )
    parser.add_argument(
        "--canon", dest="canon_dir",
        default="canon/normalized",
        help="Directory containing canon_index.json"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output for each file"
    )
    parser.add_argument(
        "--json", dest="output_json", action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    canon_dir = PROJECT_ROOT / args.canon_dir

    if not args.output_json:
        print("=" * 60)
        print("TKS Canon Hash Checker")
        print("=" * 60)
        print(f"Canon dir: {canon_dir}")
        print()

    # Load index
    try:
        index = load_canon_index(canon_dir)
    except FileNotFoundError as e:
        if args.output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"[ERROR] {e}")
        sys.exit(1)

    if not args.output_json:
        print("Checking file hashes...")
        if args.verbose:
            print()

    matches, mismatches, missing = check_hashes(index, verbose=args.verbose)

    # Output results
    if args.output_json:
        result = {
            "status": "pass" if not mismatches and not missing else "fail",
            "matches": len(matches),
            "mismatches": mismatches,
            "missing": missing
        }
        print(json.dumps(result, indent=2))
    else:
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"  Matched:    {len(matches)} file(s)")
        print(f"  Mismatched: {len(mismatches)} file(s)")
        print(f"  Missing:    {len(missing)} file(s)")

        if mismatches:
            print("\nMismatched files:")
            for m in mismatches:
                print(f"  - {m['file']}")
                print(f"      Stored:   {m['stored']}")
                print(f"      Computed: {m['computed']}")

        if missing:
            print("\nMissing files:")
            for m in missing:
                print(f"  - {m}")

        print()
        if mismatches or missing:
            print("[FAILED] Hash verification failed.")
            print("  Run 'python scripts/canon/build_canon.py' to update hashes.")
            sys.exit(1)
        else:
            print("[SUCCESS] All hashes match.")
            sys.exit(0)


if __name__ == "__main__":
    main()

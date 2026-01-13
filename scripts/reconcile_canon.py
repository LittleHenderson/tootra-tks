#!/usr/bin/env python3
"""
Canon Reconciliation Script
============================

Reconciles all training data with canonical JSON files.
Fixes grounding_id_to_def entries to match canon/*.json definitions.

Usage:
    python scripts/reconcile_canon.py --input output/rebalanced_mix.jsonl --output output/reconciled_mix.jsonl
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


class CanonLoader:
    """Loads and indexes all canonical TKS definitions."""

    def __init__(self, canon_dir: str = "canon/normalized"):
        self.canon_dir = Path(canon_dir)

        # Canonical lookups
        self.elements = {}      # A0, B1, C2, D3, etc.
        self.noetics = {}       # ^0, ^1, ^2, etc.
        self.subfoundations = {}  # 1a, 1b, 2c, etc.
        self.foundations = {}   # 1, 2, 3, etc.
        self.acquisitions = {}  # W1, W2, etc.
        self.worlds = {}        # A, B, C, D
        self.operators = {}     # +, -, +T, -T, etc.

        self._load_all()

    def _load_all(self):
        """Load all canonical files."""
        print("Loading canonical definitions...")

        # Elements (40)
        with open(self.canon_dir / "canon_elements_40.json", encoding="utf-8") as f:
            for elem in json.load(f):
                self.elements[elem["symbol"]] = {
                    "name": elem["name"],
                    "definition": elem["definition"],
                    "world": elem["world"],
                    "short_name": elem["name"].split(" ", 1)[-1] if " " in elem["name"] else elem["name"]
                }
        print(f"  Elements: {len(self.elements)}")

        # Noetics (10)
        with open(self.canon_dir / "canon_noetics_10.json", encoding="utf-8") as f:
            for noetic in json.load(f):
                self.noetics[noetic["id"]] = {
                    "name": noetic["name"],
                    "meaning": noetic["meaning"]
                }
        print(f"  Noetics: {len(self.noetics)}")

        # Subfoundations (28)
        with open(self.canon_dir / "canon_subfoundations_28.json", encoding="utf-8") as f:
            for sf in json.load(f):
                self.subfoundations[sf["id"]] = {
                    "name": sf["name"],
                    "foundation_id": sf["foundation_id"],
                    "world": sf["world"]
                }
        print(f"  Subfoundations: {len(self.subfoundations)}")

        # Foundations (7)
        with open(self.canon_dir / "canon_foundations_7.json", encoding="utf-8") as f:
            for found in json.load(f):
                self.foundations[found["id"]] = {
                    "name": found["name"],
                    "sub_foundations": found["sub_foundations"]
                }
        print(f"  Foundations: {len(self.foundations)}")

        # Acquisitions (22)
        with open(self.canon_dir / "canon_acquisitions_22.json", encoding="utf-8") as f:
            for acq in json.load(f):
                # Create W-prefixed ID for lookups
                self.acquisitions[f"W{acq['id']}"] = {
                    "name": acq["name"],
                    "type": acq["type"],
                    "foundation_id": acq["foundation_id"]
                }
        print(f"  Acquisitions: {len(self.acquisitions)}")

        # Worlds (4)
        with open(self.canon_dir / "canon_worlds_4.json", encoding="utf-8") as f:
            for world in json.load(f):
                self.worlds[world["id"]] = {
                    "name": world["name"],
                    "description": world.get("description", "")
                }
        print(f"  Worlds: {len(self.worlds)}")

        # Operators
        with open(self.canon_dir / "canon_operators.json", encoding="utf-8") as f:
            ops_data = json.load(f)
            ops = ops_data.get("operators", {})
            for symbol, op in ops.items():
                self.operators[symbol] = {
                    "name": op["name"],
                    "semantic": op.get("semantic", ""),
                    "description": op.get("description", "")
                }
        print(f"  Operators: {len(self.operators)}")

    def get_canonical_definition(self, entity_type: str, entity_id: str) -> Optional[str]:
        """Get the canonical definition for an entity."""

        if entity_type == "Element":
            if entity_id in self.elements:
                elem = self.elements[entity_id]
                return elem["definition"]

        elif entity_type == "Noetic":
            if entity_id in self.noetics:
                noetic = self.noetics[entity_id]
                return f"{noetic['name']}: {noetic['meaning']}"

        elif entity_type == "Sub-Foundation":
            if entity_id in self.subfoundations:
                sf = self.subfoundations[entity_id]
                return sf["name"]

        elif entity_type == "Foundation":
            if entity_id in self.foundations:
                found = self.foundations[entity_id]
                return found["name"]

        elif entity_type == "Acquisition":
            if entity_id in self.acquisitions:
                acq = self.acquisitions[entity_id]
                return f"{acq['name']} ({acq['type']})"

        elif entity_type == "World":
            if entity_id in self.worlds:
                world = self.worlds[entity_id]
                return f"{world['name']}: {world['description']}"

        return None


class TrainingDataReconciler:
    """Reconciles training data with canonical definitions."""

    def __init__(self, canon: CanonLoader):
        self.canon = canon
        self.stats = defaultdict(int)

    def parse_grounding_input(self, input_text: str) -> Optional[tuple]:
        """Parse a grounding input to extract entity type and ID."""

        # Pattern: "Define TKS <Type> <ID>"
        patterns = [
            (r"Define TKS Element ([ABCD]\d+)", "Element"),
            (r"Define TKS Noetic (\^?\d+)", "Noetic"),
            (r"Define TKS Sub-Foundation (\d+[abcd])", "Sub-Foundation"),
            (r"Define TKS Foundation (\d+)", "Foundation"),
            (r"Define TKS Acquisition (W?\d+)", "Acquisition"),
            (r"Define TKS World ([ABCD])", "World"),
        ]

        for pattern, entity_type in patterns:
            match = re.search(pattern, input_text)
            if match:
                entity_id = match.group(1)
                # Normalize noetic ID
                if entity_type == "Noetic" and not entity_id.startswith("^"):
                    entity_id = f"^{entity_id}"
                return (entity_type, entity_id)

        return None

    def reconcile_sample(self, sample: Dict) -> Dict:
        """Reconcile a single training sample with canon."""

        task_type = sample.get("task_type", "")

        # Only reconcile grounding_id_to_def tasks
        if task_type != "grounding_id_to_def":
            return sample

        parsed = self.parse_grounding_input(sample["input"])
        if not parsed:
            self.stats["parse_failed"] += 1
            return sample

        entity_type, entity_id = parsed
        canonical_def = self.canon.get_canonical_definition(entity_type, entity_id)

        if canonical_def is None:
            self.stats["canon_not_found"] += 1
            return sample

        # Check if update needed
        old_output = sample["output"]
        if old_output != canonical_def:
            self.stats["updated"] += 1
            self.stats[f"updated_{entity_type}"] += 1

            # Create updated sample
            updated = sample.copy()
            updated["output"] = canonical_def
            updated["_reconciled"] = True
            updated["_old_output"] = old_output
            return updated
        else:
            self.stats["already_correct"] += 1
            return sample

    def reconcile_file(self, input_path: str, output_path: str) -> Dict:
        """Reconcile an entire training file."""

        print(f"\nReconciling: {input_path}")

        samples = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        print(f"  Loaded {len(samples)} samples")

        # Reconcile each sample
        reconciled = []
        for sample in samples:
            reconciled.append(self.reconcile_sample(sample))

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in reconciled:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"  Saved to: {output_path}")

        return dict(self.stats)


def generate_canon_grounding_data(canon: CanonLoader, output_path: str):
    """Generate fresh grounding data directly from canon."""

    print(f"\nGenerating canonical grounding data...")

    samples = []

    # Elements
    for elem_id, elem in canon.elements.items():
        samples.append({
            "input": f"Define TKS Element {elem_id}",
            "output": elem["definition"],
            "task_type": "grounding_id_to_def",
            "source": "canon_reconciled"
        })
        # Also add name lookup
        samples.append({
            "input": f"What is {elem_id} in TKS?",
            "output": f"{elem['short_name']}: {elem['definition']}",
            "task_type": "grounding_id_to_def",
            "source": "canon_reconciled"
        })

    # Noetics
    for noetic_id, noetic in canon.noetics.items():
        samples.append({
            "input": f"Define TKS Noetic {noetic_id}",
            "output": f"{noetic['name']}: {noetic['meaning']}",
            "task_type": "grounding_id_to_def",
            "source": "canon_reconciled"
        })

    # Subfoundations
    for sf_id, sf in canon.subfoundations.items():
        samples.append({
            "input": f"Define TKS Sub-Foundation {sf_id}",
            "output": sf["name"],
            "task_type": "grounding_id_to_def",
            "source": "canon_reconciled"
        })

    # Foundations
    for found_id, found in canon.foundations.items():
        samples.append({
            "input": f"Define TKS Foundation {found_id}",
            "output": found["name"],
            "task_type": "grounding_id_to_def",
            "source": "canon_reconciled"
        })

    # Acquisitions
    for acq_id, acq in canon.acquisitions.items():
        samples.append({
            "input": f"Define TKS Acquisition {acq_id}",
            "output": f"{acq['name']} ({acq['type']})",
            "task_type": "grounding_id_to_def",
            "source": "canon_reconciled"
        })

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  Generated {len(samples)} canonical grounding samples")
    print(f"  Saved to: {output_path}")

    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Reconcile training data with TKS canon")
    parser.add_argument("--input", type=str, default="output/rebalanced_mix.jsonl",
                        help="Input training file")
    parser.add_argument("--output", type=str, default="output/reconciled_mix.jsonl",
                        help="Output reconciled file")
    parser.add_argument("--canon-dir", type=str, default="canon/normalized",
                        help="Directory containing canon JSON files")
    parser.add_argument("--generate-grounding", action="store_true",
                        help="Generate fresh grounding data from canon")
    parser.add_argument("--grounding-output", type=str, default="output/canon_grounding.jsonl",
                        help="Output file for generated grounding data")
    args = parser.parse_args()

    print("=" * 60)
    print("CANON RECONCILIATION")
    print("=" * 60)

    # Load canon
    canon = CanonLoader(args.canon_dir)

    # Reconcile training data
    reconciler = TrainingDataReconciler(canon)
    stats = reconciler.reconcile_file(args.input, args.output)

    print("\n" + "=" * 60)
    print("RECONCILIATION STATS")
    print("=" * 60)
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")

    # Optionally generate fresh grounding data
    if args.generate_grounding:
        print("\n" + "=" * 60)
        print("GENERATING CANONICAL GROUNDING DATA")
        print("=" * 60)
        generate_canon_grounding_data(canon, args.grounding_output)

    print("\n" + "=" * 60)
    print("RECONCILIATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Schema Validation for TKS Rumination Specs

Validates sample_ledger.json against failure_ledger.schema.json
"""

import json
import os
from pathlib import Path

def validate_schemas():
    """Validate sample ledger against schema."""
    specs_dir = Path(__file__).parent

    # Load schema
    schema_path = specs_dir / "failure_ledger.schema.json"
    with open(schema_path) as f:
        schema = json.load(f)

    # Load sample
    sample_path = specs_dir / "sample_ledger.json"
    with open(sample_path) as f:
        sample = json.load(f)

    print("=" * 50)
    print("TKS Schema Validation")
    print("=" * 50)

    # Basic structure validation (without jsonschema library)
    errors = []

    # Check required top-level fields
    required = ["version", "threshold", "created_at", "updated_at", "records", "index"]
    for field in required:
        if field not in sample:
            errors.append(f"Missing required field: {field}")

    # Check version
    if sample.get("version") != "1.0":
        errors.append(f"Invalid version: {sample.get('version')}")

    # Check threshold range
    threshold = sample.get("threshold", 0)
    if not (0 <= threshold <= 1):
        errors.append(f"Threshold out of range: {threshold}")

    # Check index structure
    index = sample.get("index", {})
    for key in ["unsolved", "solved", "retired"]:
        if key not in index:
            errors.append(f"Missing index key: {key}")
        elif not isinstance(index[key], list):
            errors.append(f"Index[{key}] must be a list")

    # Validate records
    records = sample.get("records", {})
    valid_slots = ["1D", "2W", "3P", "4D", "5W", "6P", "7D", "8W", "9P", "10D", "11W", "12P"]
    valid_statuses = ["unsolved", "solved", "retired"]

    for rid, record in records.items():
        prefix = f"records[{rid}]"

        # Check required record fields
        rec_required = ["id", "question_text", "expected", "model_answer", "status",
                        "first_unmet_slot", "impact_score", "desire_level", "gist"]
        for field in rec_required:
            if field not in record:
                errors.append(f"{prefix}: Missing {field}")

        # Check slot validity
        slot = record.get("first_unmet_slot")
        if slot and slot not in valid_slots:
            errors.append(f"{prefix}: Invalid slot {slot}")

        # Check status validity
        status = record.get("status")
        if status and status not in valid_statuses:
            errors.append(f"{prefix}: Invalid status {status}")

        # Check score ranges
        for score_field in ["impact_score", "desire_level", "error_severity", "novelty_gap", "recency"]:
            val = record.get(score_field)
            if val is not None and not (0 <= val <= 1):
                errors.append(f"{prefix}: {score_field} out of range: {val}")

        # Check gist structure
        gist = record.get("gist", {})
        if "type" not in gist:
            errors.append(f"{prefix}.gist: Missing type")
        if "signature" not in gist:
            errors.append(f"{prefix}.gist: Missing signature")
        if "slot_block" not in gist:
            errors.append(f"{prefix}.gist: Missing slot_block")
        elif gist.get("slot_block") not in valid_slots:
            errors.append(f"{prefix}.gist: Invalid slot_block {gist.get('slot_block')}")

        # Check hypotheses
        for i, hyp in enumerate(record.get("hypotheses", [])):
            if "label" not in hyp:
                errors.append(f"{prefix}.hypotheses[{i}]: Missing label")
            if hyp.get("status") not in ["proposed", "validated", "rejected"]:
                errors.append(f"{prefix}.hypotheses[{i}]: Invalid status {hyp.get('status')}")

    # Cross-check index with records
    all_ids = set(records.keys())
    indexed_ids = set(index.get("unsolved", []) + index.get("solved", []) + index.get("retired", []))

    if all_ids != indexed_ids:
        missing = all_ids - indexed_ids
        extra = indexed_ids - all_ids
        if missing:
            errors.append(f"Records not in index: {missing}")
        if extra:
            errors.append(f"Index references missing records: {extra}")

    # Report results
    if errors:
        print(f"\nFAILED with {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"\nPASSED")
        print(f"  Records: {len(records)}")
        print(f"  Unsolved: {len(index.get('unsolved', []))}")
        print(f"  Solved: {len(index.get('solved', []))}")
        print(f"  Retired: {len(index.get('retired', []))}")
        return True


if __name__ == "__main__":
    import sys
    success = validate_schemas()
    sys.exit(0 if success else 1)

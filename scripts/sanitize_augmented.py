#!/usr/bin/env python3
"""
TKS Data Quality / Sanitizer Script - Phase 6

Scans augmented JSONL files for data quality issues and provides options to clean them.

Detects:
- Duplicate entries (by id and/or content hash)
- Conflicts (invalid operators, non-canonical elements)
- Missing required fields

Options:
- --drop-invalid: Remove invalid entries from output
- --flag-only: Report issues without removing entries
- --output: Specify output file for cleaned data

Pipeline Integration Points:
1. POST-TEACHER GENERATION (Recommended):
   Run after teacher generation, before augmentation
   Use: --flag-only (catch issues early)

2. POST-AUGMENTATION (Critical):
   Run after all augmentation (inversion, anti-attractor, etc.)
   Use: --drop-invalid (final quality gate before training)

See docs/DATA_SANITIZER_GUIDE.md for full pipeline integration details.

Usage:
    # Scan and report only
    python scripts/sanitize_augmented.py --input data/augmented.jsonl --flag-only

    # Clean and save
    python scripts/sanitize_augmented.py --input data/augmented.jsonl --output data/clean.jsonl --drop-invalid

    # Scan with detailed report
    python scripts/sanitize_augmented.py --input data/augmented.jsonl --report report.json

Author: TKS-LLM Agent 2
Date: 2025-12-14
Version: 1.0.0
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative.constants import (
    WORLD_LETTERS,
    ALLOWED_OPS,
    is_valid_world,
    is_valid_noetic,
    is_valid_foundation,
    is_valid_operator
)

# =============================================================================
# CANONICAL CONSTRAINTS (from guardrails)
# =============================================================================

CANONICAL_WORLDS = {"A", "B", "C", "D"}
CANONICAL_NOETICS = set(range(1, 11))  # 1-10
CANONICAL_FOUNDATIONS = set(range(1, 8))  # 1-7
CANONICAL_OPS = ALLOWED_OPS  # {"+", "-", "+T", "-T", "->", "<-", "*T", "/T", "o"}

# Noetic involutions and self-duals
NOETIC_INVOLUTIONS = {2: 3, 3: 2, 5: 6, 6: 5, 8: 9, 9: 8}
NOETIC_SELF_DUALS = {1, 4, 7, 10}

REQUIRED_FIELDS = {"id", "story", "expr", "aug_type", "validator_pass"}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SanitizationIssue:
    """Represents a data quality issue found during sanitization."""
    entry_id: str
    issue_type: str
    description: str
    severity: str = "error"  # "error", "warning", "info"
    field: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SanitizationReport:
    """Report of sanitization process."""
    total_entries: int = 0
    clean_entries: int = 0
    duplicate_entries: int = 0
    invalid_operators: int = 0
    invalid_worlds: int = 0
    invalid_noetics: int = 0
    missing_fields: int = 0
    structural_errors: int = 0

    issues: List[SanitizationIssue] = field(default_factory=list)
    duplicates_by_id: Dict[str, int] = field(default_factory=dict)
    duplicates_by_hash: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def add_issue(self, issue: SanitizationIssue):
        """Add an issue to the report."""
        self.issues.append(issue)

        # Update counters
        if issue.issue_type == "duplicate_id":
            self.duplicate_entries += 1
        elif issue.issue_type == "invalid_operator":
            self.invalid_operators += 1
        elif issue.issue_type == "invalid_world":
            self.invalid_worlds += 1
        elif issue.issue_type == "invalid_noetic":
            self.invalid_noetics += 1
        elif issue.issue_type == "missing_field":
            self.missing_fields += 1
        elif issue.issue_type == "structural_error":
            self.structural_errors += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "summary": {
                "total_entries": self.total_entries,
                "clean_entries": self.clean_entries,
                "duplicate_entries": self.duplicate_entries,
                "invalid_operators": self.invalid_operators,
                "invalid_worlds": self.invalid_worlds,
                "invalid_noetics": self.invalid_noetics,
                "missing_fields": self.missing_fields,
                "structural_errors": self.structural_errors,
                "pass_rate": self.clean_entries / self.total_entries if self.total_entries > 0 else 0.0,
            },
            "issues": [issue.to_dict() for issue in self.issues],
            "duplicates": {
                "by_id": self.duplicates_by_id,
                "by_hash": {k: v for k, v in self.duplicates_by_hash.items() if len(v) > 1}
            }
        }

    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "="*70)
        print("TKS DATA SANITIZATION REPORT")
        print("="*70)
        print(f"\nTotal entries scanned:      {self.total_entries}")
        print(f"Clean entries:              {self.clean_entries}")
        print(f"Entries with issues:        {self.total_entries - self.clean_entries}")

        if self.total_entries > 0:
            pass_rate = (self.clean_entries / self.total_entries) * 100
            print(f"Pass rate:                  {pass_rate:.1f}%")

        print("\n" + "-"*70)
        print("ISSUES BREAKDOWN:")
        print("-"*70)
        print(f"  Duplicate entries (by id):    {self.duplicate_entries}")
        print(f"  Duplicate content (by hash):  {len([v for v in self.duplicates_by_hash.values() if len(v) > 1])}")
        print(f"  Invalid operators:            {self.invalid_operators}")
        print(f"  Invalid worlds:               {self.invalid_worlds}")
        print(f"  Invalid noetics:              {self.invalid_noetics}")
        print(f"  Missing required fields:      {self.missing_fields}")
        print(f"  Structural errors:            {self.structural_errors}")

        if self.issues:
            print("\n" + "-"*70)
            print(f"DETAILED ISSUES (showing first 20 of {len(self.issues)}):")
            print("-"*70)
            for issue in self.issues[:20]:
                severity_marker = "[ERROR]" if issue.severity == "error" else "[WARN]"
                field_info = f" (field: {issue.field})" if issue.field else ""
                print(f"{severity_marker} {issue.entry_id}: {issue.description}{field_info}")

        print("\n" + "="*70 + "\n")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def compute_content_hash(entry: Dict[str, Any]) -> str:
    """Compute a hash of entry content (excluding id and metadata)."""
    # Create a canonical representation for hashing
    content_fields = ["story", "expr", "aug_type"]
    content = {k: entry.get(k, "") for k in content_fields if k in entry}
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()


def validate_required_fields(entry: Dict[str, Any]) -> List[SanitizationIssue]:
    """Check that all required fields are present."""
    issues = []
    entry_id = entry.get("id", "<missing_id>")

    for field in REQUIRED_FIELDS:
        if field not in entry or entry[field] is None or entry[field] == "":
            issues.append(SanitizationIssue(
                entry_id=entry_id,
                issue_type="missing_field",
                description=f"Missing required field: {field}",
                severity="error",
                field=field
            ))

    return issues


def validate_operators(entry: Dict[str, Any]) -> List[SanitizationIssue]:
    """Validate that all operators are canonical."""
    issues = []
    entry_id = entry.get("id", "<missing_id>")

    ops = entry.get("expr_ops", [])
    if not isinstance(ops, list):
        issues.append(SanitizationIssue(
            entry_id=entry_id,
            issue_type="structural_error",
            description="expr_ops must be a list",
            severity="error",
            field="expr_ops"
        ))
        return issues

    for i, op in enumerate(ops):
        if not is_valid_operator(op):
            issues.append(SanitizationIssue(
                entry_id=entry_id,
                issue_type="invalid_operator",
                description=f"Invalid operator '{op}' at position {i} (must be in {CANONICAL_OPS})",
                severity="error",
                field="expr_ops"
            ))

    return issues


def validate_elements(entry: Dict[str, Any]) -> List[SanitizationIssue]:
    """Validate that all elements use canonical worlds and noetics."""
    issues = []
    entry_id = entry.get("id", "<missing_id>")

    elements = entry.get("expr_elements", [])
    if not isinstance(elements, list):
        issues.append(SanitizationIssue(
            entry_id=entry_id,
            issue_type="structural_error",
            description="expr_elements must be a list",
            severity="error",
            field="expr_elements"
        ))
        return issues

    for i, element in enumerate(elements):
        element_str = str(element).strip()

        if not element_str:
            issues.append(SanitizationIssue(
                entry_id=entry_id,
                issue_type="structural_error",
                description=f"Empty element at position {i}",
                severity="error",
                field="expr_elements"
            ))
            continue

        # Parse element (e.g., "D5", "B2.1", "A10")
        # Strip sense notation if present
        base_element = element_str.split(".")[0].split("^")[0].split("_")[0]

        if len(base_element) < 2:
            issues.append(SanitizationIssue(
                entry_id=entry_id,
                issue_type="structural_error",
                description=f"Invalid element format '{element_str}' at position {i}",
                severity="error",
                field="expr_elements"
            ))
            continue

        # Extract world and noetic
        world = base_element[0].upper()
        noetic_str = base_element[1:]

        # Validate world
        if not is_valid_world(world):
            issues.append(SanitizationIssue(
                entry_id=entry_id,
                issue_type="invalid_world",
                description=f"Invalid world '{world}' in element '{element_str}' at position {i} (must be A/B/C/D)",
                severity="error",
                field="expr_elements"
            ))

        # Validate noetic
        try:
            noetic = int(noetic_str)
            if not is_valid_noetic(noetic):
                issues.append(SanitizationIssue(
                    entry_id=entry_id,
                    issue_type="invalid_noetic",
                    description=f"Invalid noetic '{noetic}' in element '{element_str}' at position {i} (must be 1-10)",
                    severity="error",
                    field="expr_elements"
                ))
        except ValueError:
            issues.append(SanitizationIssue(
                entry_id=entry_id,
                issue_type="invalid_noetic",
                description=f"Invalid noetic '{noetic_str}' in element '{element_str}' at position {i} (must be integer)",
                severity="error",
                field="expr_elements"
            ))

    return issues


def validate_structure(entry: Dict[str, Any]) -> List[SanitizationIssue]:
    """Validate structural consistency of expressions."""
    issues = []
    entry_id = entry.get("id", "<missing_id>")

    elements = entry.get("expr_elements", [])
    ops = entry.get("expr_ops", [])

    if not isinstance(elements, list) or not isinstance(ops, list):
        return issues  # Already reported in other validators

    # Check that number of operators is correct
    if len(elements) > 1:
        expected_ops = len(elements) - 1
        if len(ops) != expected_ops:
            issues.append(SanitizationIssue(
                entry_id=entry_id,
                issue_type="structural_error",
                description=f"Structural inconsistency: {len(elements)} elements require {expected_ops} operators, got {len(ops)}",
                severity="error",
                field="expr_ops"
            ))
    elif len(elements) == 1 and len(ops) > 0:
        issues.append(SanitizationIssue(
            entry_id=entry_id,
            issue_type="structural_error",
            description=f"Single element should have no operators, got {len(ops)}",
            severity="error",
            field="expr_ops"
        ))

    return issues


def validate_entry(entry: Dict[str, Any]) -> Tuple[bool, List[SanitizationIssue]]:
    """
    Validate a single entry for all data quality issues.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check required fields
    issues.extend(validate_required_fields(entry))

    # Only validate content if required fields are present
    if not any(issue.issue_type == "missing_field" for issue in issues):
        # Validate operators
        issues.extend(validate_operators(entry))

        # Validate elements
        issues.extend(validate_elements(entry))

        # Validate structure
        issues.extend(validate_structure(entry))

    is_valid = len(issues) == 0
    return is_valid, issues


# =============================================================================
# SANITIZATION FUNCTIONS
# =============================================================================

def scan_jsonl(input_file: Path) -> Tuple[List[Dict[str, Any]], SanitizationReport]:
    """
    Scan a JSONL file for data quality issues.

    Returns:
        Tuple of (entries, report)
    """
    entries = []
    report = SanitizationReport()

    seen_ids = {}
    content_hashes = defaultdict(list)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                # Create a placeholder entry for reporting
                placeholder_id = f"<line_{line_num}>"
                report.add_issue(SanitizationIssue(
                    entry_id=placeholder_id,
                    issue_type="structural_error",
                    description=f"JSON decode error: {e}",
                    severity="error"
                ))
                report.total_entries += 1
                continue

            entries.append(entry)
            report.total_entries += 1

            entry_id = entry.get("id", f"<line_{line_num}>")

            # Check for duplicate IDs
            if entry_id in seen_ids:
                report.add_issue(SanitizationIssue(
                    entry_id=entry_id,
                    issue_type="duplicate_id",
                    description=f"Duplicate ID (previously seen at line {seen_ids[entry_id]})",
                    severity="error",
                    field="id"
                ))
                report.duplicates_by_id[entry_id] = report.duplicates_by_id.get(entry_id, 1) + 1
            else:
                seen_ids[entry_id] = line_num

            # Check for duplicate content (by hash)
            content_hash = compute_content_hash(entry)
            content_hashes[content_hash].append(entry_id)

            # Validate entry
            is_valid, issues = validate_entry(entry)

            for issue in issues:
                report.add_issue(issue)

            if is_valid and entry_id not in report.duplicates_by_id:
                report.clean_entries += 1

    # Store duplicate hashes
    report.duplicates_by_hash = content_hashes

    return entries, report


def clean_entries(
    entries: List[Dict[str, Any]],
    report: SanitizationReport,
    drop_invalid: bool = False
) -> List[Dict[str, Any]]:
    """
    Clean entries based on sanitization report.

    Args:
        entries: List of entries
        report: Sanitization report
        drop_invalid: If True, remove invalid entries; if False, keep all

    Returns:
        List of cleaned entries
    """
    if not drop_invalid:
        return entries

    # Build set of IDs with issues
    invalid_ids = set()
    for issue in report.issues:
        if issue.severity == "error":
            invalid_ids.add(issue.entry_id)

    # Filter out invalid entries
    cleaned = []
    for entry in entries:
        entry_id = entry.get("id", "")
        if entry_id not in invalid_ids:
            cleaned.append(entry)

    return cleaned


def write_jsonl(entries: List[Dict[str, Any]], output_file: Path):
    """Write entries to JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TKS Data Quality Sanitizer - Scan and clean augmented JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan and report only
  python scripts/sanitize_augmented.py --input data/augmented.jsonl --flag-only

  # Clean and save to new file
  python scripts/sanitize_augmented.py --input data/augmented.jsonl --output data/clean.jsonl --drop-invalid

  # Generate detailed JSON report
  python scripts/sanitize_augmented.py --input data/augmented.jsonl --report report.json --flag-only
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file to scan"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file for cleaned data (optional)"
    )

    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Remove invalid entries from output (requires --output)"
    )

    parser.add_argument(
        "--flag-only",
        action="store_true",
        help="Report issues without removing entries"
    )

    parser.add_argument(
        "--report",
        type=str,
        help="Save detailed report to JSON file (optional)"
    )

    args = parser.parse_args()

    # Validate arguments
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if args.drop_invalid and not args.output:
        print("Error: --drop-invalid requires --output", file=sys.stderr)
        return 1

    # Scan file
    print(f"Scanning {input_path}...")
    entries, report = scan_jsonl(input_path)

    # Print summary
    report.print_summary()

    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Detailed report saved to: {report_path}")

    # Clean and save if requested
    if args.output:
        output_path = Path(args.output)

        if args.flag_only:
            # Just copy all entries
            cleaned = entries
            print(f"\nWriting all {len(cleaned)} entries to {output_path} (flag-only mode)")
        else:
            # Clean entries
            cleaned = clean_entries(entries, report, drop_invalid=args.drop_invalid)

            if args.drop_invalid:
                removed = len(entries) - len(cleaned)
                print(f"\nRemoved {removed} invalid entries")
                print(f"Writing {len(cleaned)} clean entries to {output_path}")
            else:
                print(f"\nWriting all {len(cleaned)} entries to {output_path}")

        write_jsonl(cleaned, output_path)
        print(f"Output saved to: {output_path}")

    # Return exit code based on data quality
    if report.total_entries == report.clean_entries:
        print("\n All entries are clean!")
        return 0
    else:
        issues_count = report.total_entries - report.clean_entries
        print(f"\n Found {issues_count} entries with issues")
        return 0 if args.flag_only else 1


if __name__ == "__main__":
    sys.exit(main())

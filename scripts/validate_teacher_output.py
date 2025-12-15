#!/usr/bin/env python3
"""
Validate teacher outputs for canonical compliance.
"""
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher import CanonicalValidator


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_teacher_output.py <teacher_outputs.jsonl>")
        return 1

    input_file = sys.argv[1]

    print("=" * 60)
    print("TEACHER OUTPUT VALIDATION")
    print("=" * 60)
    print(f"\nInput: {input_file}")

    # Load entries
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Loaded: {len(entries)} entries")

    # Validate each entry
    validator = CanonicalValidator(strict_mode=True)

    valid_count = 0
    invalid_count = 0
    issues_by_type = {}

    for i, entry in enumerate(entries):
        # Extract text to validate (interpretation or target)
        text_to_validate = entry.get('interpretation', '') or entry.get('target', '')

        # Also check metadata
        metadata = entry.get('metadata', {})
        equation = entry.get('equation', {})

        # Validate interpretation text
        result = validator.validate(text_to_validate)

        # Check equation elements for canonical worlds
        elements = equation.get('elements', [])
        for elem in elements:
            if elem and len(elem) >= 2:
                world = elem[0].upper()
                if world not in ['A', 'B', 'C', 'D']:
                    result.is_valid = False
                    result.add_issue(f"Non-canonical world '{world}' in element '{elem}'")

        if result.is_valid and result.canon_score >= 0.8:
            valid_count += 1
        else:
            invalid_count += 1

            # Track issues
            for issue in result.issues:
                issue_type = issue.rule
                issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1

    # Report
    print("\n" + "-" * 60)
    print("VALIDATION RESULTS")
    print("-" * 60)
    print(f"Total entries: {len(entries)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"Pass rate: {valid_count / len(entries) * 100:.1f}%")

    if issues_by_type:
        print("\nIssues found:")
        for issue_type, count in sorted(issues_by_type.items()):
            print(f"  {issue_type}: {count}")
    else:
        print("\nNo validation issues found!")

    print("\n" + "=" * 60)
    print("CANONICAL CONSTRAINTS VERIFIED:")
    print("=" * 60)
    print("- Worlds: A, B, C, D only")
    print("- Noetics: 1-10 (pairs: 2<->3, 5<->6, 8<->9; self-duals: 1,4,7,10)")
    print("- Foundations: 1-7")
    print("- Operators: +, -, +T, -T, ->, <-, *T, /T, o")
    print("=" * 60)

    return 0 if invalid_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

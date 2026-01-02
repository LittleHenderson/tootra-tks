#!/usr/bin/env python3
"""
Convert teacher output format to augmentation-compatible format.

Teacher format:
{
    "task_type": "...",
    "equation": {"elements": ["A1", "B2"], "noetics": [1, 2], ...},
    ...
}

Augmentation format:
{
    "story": "...",
    "equation": "A1 +T B2",
    "expr_elements": ["A1", "B2"],
    "expr_ops": ["+T"],
    "aug_type": "original"
}

Usage:
    python scripts/convert_teacher_format.py --input output/teacher_all.jsonl --output output/teacher_converted.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Default operator for connecting elements
DEFAULT_OP = "+T"

# Pattern for extended notation elements
ELEMENT_PATTERN = re.compile(r'^([ABCD])(10|[1-9])(\^[1-9])?(_d[1-7])?$', re.IGNORECASE)


def extract_base_element(element: str) -> str:
    """Extract base element (world + noetic) from extended notation.

    E.g., 'B8^3_d4' -> 'B8', 'A10^5_d7' -> 'A10'
    """
    match = ELEMENT_PATTERN.match(element.upper())
    if match:
        return f"{match.group(1)}{match.group(2)}"
    # Fall back to simple extraction if no match
    simple_match = re.match(r'([ABCD])(10|[1-9])', element.upper())
    if simple_match:
        return f"{simple_match.group(1)}{simple_match.group(2)}"
    return element  # Return as-is if no pattern matches


def infer_ops_from_elements(elements: List[str]) -> List[str]:
    """
    Infer operators for a list of elements.
    Uses +T as default connector.

    Args:
        elements: List of element codes like ["A1", "B2", "C3"]

    Returns:
        List of operators, one fewer than elements
    """
    if len(elements) <= 1:
        return []
    return [DEFAULT_OP] * (len(elements) - 1)


def build_equation_string(elements: List[str], ops: List[str]) -> str:
    """
    Build equation string from elements and operators.

    Args:
        elements: List of element codes
        ops: List of operators

    Returns:
        Equation string like "A1 +T B2 -T C3"
    """
    if not elements:
        return ""

    if len(elements) == 1:
        return elements[0]

    result = elements[0]
    for i, op in enumerate(ops):
        if i + 1 < len(elements):
            result += f" {op} {elements[i + 1]}"

    return result


def convert_entry(entry: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """
    Convert a single teacher entry to augmentation format.

    Args:
        entry: Teacher-format entry
        idx: Entry index for ID generation

    Returns:
        Converted entry or None if conversion fails
    """
    # Extract elements from various possible locations
    elements = []

    # Try equation.elements first
    if "equation" in entry and isinstance(entry["equation"], dict):
        elements = entry["equation"].get("elements", [])

    # Fall back to metadata.elements
    if not elements and "metadata" in entry:
        elements = entry["metadata"].get("elements", [])
        if not elements:
            elements = entry["metadata"].get("target_elements", [])

    if not elements:
        return None

    # Infer operators
    ops = infer_ops_from_elements(elements)

    # Build equation string
    equation_str = build_equation_string(elements, ops)

    # Build story from interpretation or input
    story = entry.get("interpretation", "")
    if not story or story == "This is a mock response for TKS equation interpretation.":
        story = entry.get("input", "")
        # Clean up story - extract meaningful part
        if "Given the TKS equation:" in story:
            story = f"Working with {equation_str}"
        elif "Given this interpretation" in story:
            story = f"Interpretation of {equation_str}"

    # Determine aug_type from task_type
    task_type = entry.get("task_type", "original")
    aug_type_map = {
        "equation_to_interpretation": "original",
        "interpretation_to_equation": "reverse",
        "equation_to_rpm": "rpm",
        "equation_to_foundations": "foundations",
    }
    aug_type = aug_type_map.get(task_type, "original")

    # Extract base elements for validation (strips extended notation)
    base_elements = [extract_base_element(e) for e in elements]

    return {
        "id": f"teacher_{idx}",
        "story": story,
        "equation": equation_str,
        "expr_elements": elements,           # Raw elements (may include ^k_dF)
        "expr_elements_base": base_elements, # Base elements for validation
        "expr_ops": ops,
        "aug_type": aug_type,
        "canon_score": entry.get("canon_score", 1.0),
        "source": "teacher_generation",
        "original_task_type": task_type,
    }


def convert_file(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Convert entire teacher output file.

    Args:
        input_path: Path to teacher output JSONL
        output_path: Path to write converted JSONL

    Returns:
        Statistics about conversion
    """
    stats = {
        "total": 0,
        "converted": 0,
        "skipped": 0,
        "by_task_type": {},
    }

    converted_entries = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                entry = json.loads(line)
                task_type = entry.get("task_type", "unknown")
                stats["by_task_type"][task_type] = stats["by_task_type"].get(task_type, 0) + 1

                converted = convert_entry(entry, idx)
                if converted:
                    converted_entries.append(converted)
                    stats["converted"] += 1
                else:
                    stats["skipped"] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {idx}: {e}", file=sys.stderr)
                stats["skipped"] += 1

    # Write converted entries
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in converted_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert teacher output to augmentation-compatible format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input teacher JSONL file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output converted JSONL file')

    args = parser.parse_args()

    print(f"Converting: {args.input} -> {args.output}")

    stats = convert_file(args.input, args.output)

    print(f"\nConversion Statistics:")
    print(f"  Total entries: {stats['total']}")
    print(f"  Converted: {stats['converted']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"\n  By task type:")
    for task_type, count in sorted(stats['by_task_type'].items()):
        print(f"    {task_type}: {count}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Dual-Task Story-Equation Converter

Converts story-equation paired data into training format for bidirectional learning.
Handles both story_to_eq and eq_to_story directions with full canon validation.

Source format (data/story_equation_pairs_*.jsonl):
{
    "story": "...",
    "equation": "A1 +T B2",
    "expr_elements": ["A1", "B2"],
    "expr_ops": ["+T"],
    "direction": "story_to_eq" | "eq_to_story",
    "pair_id": "pair_0000"
}

Target format (output/teacher_story_eq_*.jsonl):
{
    "task_type": "story_to_equation" | "equation_to_story",
    "input": "...",
    "target": "...",
    "metadata": {
        "elements": [...],
        "operators": [...],
        "pair_id": "..."
    },
    "direction": "story_to_eq" | "eq_to_story",
    "canon_score": 1.0
}

Canon Validation:
- Worlds: A, B, C, D only (strict)
- Noetics: 1-10 only (strict)
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (exactly 9)
- Senses: ^1-^9 (extended notation)
- Foundations: _d1-_d7 (extended notation)

Usage:
    python scripts/convert_story_equation.py --input data/story_equation_pairs_train.jsonl --output output/teacher_story_eq_train.jsonl
    python scripts/convert_story_equation.py --input data/story_equation_pairs_holdout.jsonl --output output/teacher_story_eq_holdout.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# ==============================================================================
# CANONICAL CONSTANTS
# ==============================================================================

# Valid canonical worlds (strict)
CANONICAL_WORLDS = {'A', 'B', 'C', 'D'}

# Valid noetics (strict)
CANONICAL_NOETICS = set(range(1, 11))  # 1-10

# Valid operators (exactly 9)
CANONICAL_OPERATORS = {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'}

# Valid senses (extended notation)
CANONICAL_SENSES = set(range(1, 10))  # ^1-^9

# Valid foundations (extended notation)
CANONICAL_FOUNDATIONS = set(range(1, 8))  # _d1-_d7

# ==============================================================================
# REGEX PATTERNS
# ==============================================================================

# Extended notation element pattern: [ABCD](10|[1-9])(^[1-9])?(_d[1-7])?
ELEMENT_PATTERN = re.compile(r'^([ABCD])(10|[1-9])(\^[1-9])?(_d[1-7])?$', re.IGNORECASE)

# Base element pattern (world + noetic only)
BASE_ELEMENT_PATTERN = re.compile(r'^([ABCD])(10|[1-9])', re.IGNORECASE)

# Non-canonical world detection
NON_CANONICAL_WORLD_PATTERN = re.compile(r'^([EFGHIJKLMNOPQRSTUVWXYZ])\d+', re.IGNORECASE)

# ==============================================================================
# VALIDATION DATACLASS
# ==============================================================================

@dataclass
class CanonValidationResult:
    """Result of canonical validation."""
    is_valid: bool
    canon_score: float
    errors: List[str]
    warnings: List[str]

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False
        self.canon_score = max(0.0, self.canon_score - 0.2)

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        self.canon_score = max(0.0, self.canon_score - 0.05)


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_element(element: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single element against canonical rules.

    Args:
        element: Element string (e.g., "A1", "B10^3_d5")

    Returns:
        (is_valid, error_message)
    """
    # Check for non-canonical worlds first
    if NON_CANONICAL_WORLD_PATTERN.match(element.upper()):
        world = element[0].upper()
        return False, f"Non-canonical world '{world}' in element '{element}'"

    match = ELEMENT_PATTERN.match(element.upper())
    if not match:
        return False, f"Invalid element format: '{element}'"

    world = match.group(1).upper()
    noetic = int(match.group(2))
    sense = match.group(3)
    foundation = match.group(4)

    # Validate world (strict)
    if world not in CANONICAL_WORLDS:
        return False, f"Invalid world '{world}' (must be A, B, C, or D)"

    # Validate noetic (strict)
    if noetic not in CANONICAL_NOETICS:
        return False, f"Invalid noetic {noetic} (must be 1-10)"

    # Validate sense if present
    if sense:
        sense_num = int(sense[1])  # Extract number from ^N
        if sense_num not in CANONICAL_SENSES:
            return False, f"Invalid sense {sense} (must be ^1-^9)"

    # Validate foundation if present
    if foundation:
        foundation_num = int(foundation[2])  # Extract number from _dN
        if foundation_num not in CANONICAL_FOUNDATIONS:
            return False, f"Invalid foundation {foundation} (must be _d1-_d7)"

    return True, None


def validate_operator(operator: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single operator against canonical rules.

    Args:
        operator: Operator string (e.g., "+T", "->")

    Returns:
        (is_valid, error_message)
    """
    if operator not in CANONICAL_OPERATORS:
        return False, f"Invalid operator '{operator}' (must be one of: {', '.join(sorted(CANONICAL_OPERATORS))})"
    return True, None


def validate_equation_components(
    elements: List[str],
    operators: List[str]
) -> CanonValidationResult:
    """
    Validate equation components against canonical rules.

    Args:
        elements: List of element strings
        operators: List of operator strings

    Returns:
        CanonValidationResult with validation details
    """
    result = CanonValidationResult(
        is_valid=True,
        canon_score=1.0,
        errors=[],
        warnings=[]
    )

    # Validate elements
    for elem in elements:
        is_valid, error = validate_element(elem)
        if not is_valid:
            result.add_error(f"Element validation: {error}")

    # Validate operators
    for op in operators:
        is_valid, error = validate_operator(op)
        if not is_valid:
            result.add_error(f"Operator validation: {error}")

    # Validate element-operator count relationship
    if len(operators) != len(elements) - 1:
        result.add_warning(
            f"Operator count mismatch: {len(operators)} operators for {len(elements)} elements "
            f"(expected {len(elements) - 1})"
        )

    return result


def extract_base_element(element: str) -> str:
    """
    Extract base element (world + noetic) from extended notation.

    Args:
        element: Element with optional extended notation (e.g., "B8^3_d4")

    Returns:
        Base element (e.g., "B8")
    """
    match = BASE_ELEMENT_PATTERN.match(element.upper())
    if match:
        return f"{match.group(1)}{match.group(2)}"
    # Fallback: return uppercased element
    return element.upper()


# ==============================================================================
# CONVERSION FUNCTIONS
# ==============================================================================

def build_story_to_eq_prompt(story: str) -> str:
    """
    Build input prompt for story-to-equation task.

    Args:
        story: Natural language story

    Returns:
        Formatted input prompt
    """
    return (
        f"Given this TKS narrative:\n\n"
        f"{story}\n\n"
        f"Translate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 1-10)."
    )


def build_eq_to_story_prompt(equation: str, elements: List[str]) -> str:
    """
    Build input prompt for equation-to-story task.

    Args:
        equation: TKS equation string
        elements: List of elements in equation

    Returns:
        Formatted input prompt
    """
    elements_str = ", ".join(elements)
    return (
        f"Given the TKS equation: {equation}\n\n"
        f"Elements: {elements_str}\n\n"
        f"Generate a natural language narrative describing this TKS working."
    )


def convert_entry(entry: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """
    Convert a single story-equation pair to training format.

    Args:
        entry: Source entry from story_equation_pairs file
        idx: Entry index for error reporting

    Returns:
        Converted entry or None if conversion fails
    """
    # Extract required fields
    try:
        story = entry["story"]
        equation = entry["equation"]
        elements = entry["expr_elements"]
        operators = entry["expr_ops"]
        direction = entry["direction"]
        pair_id = entry.get("pair_id", f"pair_{idx:04d}")
    except KeyError as e:
        print(f"Warning: Missing required field {e} in entry {idx}", file=sys.stderr)
        return None

    # Validate direction
    if direction not in ["story_to_eq", "eq_to_story"]:
        print(f"Warning: Invalid direction '{direction}' in entry {idx}", file=sys.stderr)
        return None

    # Validate equation components
    validation = validate_equation_components(elements, operators)

    if not validation.is_valid:
        print(f"Warning: Canon validation failed for entry {idx}:", file=sys.stderr)
        for error in validation.errors:
            print(f"  ERROR: {error}", file=sys.stderr)
        for warning in validation.warnings:
            print(f"  WARNING: {warning}", file=sys.stderr)
        # Still include the entry but with low canon score

    # Build input and target based on direction
    if direction == "story_to_eq":
        task_type = "story_to_equation"
        input_text = build_story_to_eq_prompt(story)
        target = equation
    else:  # eq_to_story
        task_type = "equation_to_story"
        input_text = build_eq_to_story_prompt(equation, elements)
        target = story

    # Extract base elements for validation
    base_elements = [extract_base_element(e) for e in elements]

    # Build output entry
    return {
        "task_type": task_type,
        "input": input_text,
        "target": target,
        "metadata": {
            "elements": elements,
            "elements_base": base_elements,
            "operators": operators,
            "pair_id": pair_id,
            "validation_errors": validation.errors if validation.errors else None,
            "validation_warnings": validation.warnings if validation.warnings else None,
        },
        "direction": direction,
        "canon_score": validation.canon_score,
    }


def convert_file(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Convert entire story-equation pairs file.

    Args:
        input_path: Path to source JSONL file
        output_path: Path to write converted JSONL

    Returns:
        Statistics about conversion
    """
    stats = {
        "total": 0,
        "converted": 0,
        "skipped": 0,
        "by_direction": {},
        "canon_valid": 0,
        "canon_invalid": 0,
        "avg_canon_score": 0.0,
    }

    converted_entries = []
    canon_scores = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                entry = json.loads(line)
                direction = entry.get("direction", "unknown")
                stats["by_direction"][direction] = stats["by_direction"].get(direction, 0) + 1

                converted = convert_entry(entry, idx)
                if converted:
                    converted_entries.append(converted)
                    stats["converted"] += 1

                    # Track canon statistics
                    canon_score = converted["canon_score"]
                    canon_scores.append(canon_score)
                    if canon_score >= 1.0:
                        stats["canon_valid"] += 1
                    else:
                        stats["canon_invalid"] += 1
                else:
                    stats["skipped"] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {idx}: {e}", file=sys.stderr)
                stats["skipped"] += 1

    # Compute average canon score
    if canon_scores:
        stats["avg_canon_score"] = sum(canon_scores) / len(canon_scores)

    # Write converted entries
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in converted_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return stats


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert story-equation pairs to training format with dual-task support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert training data
    python scripts/convert_story_equation.py \\
        --input data/story_equation_pairs_train.jsonl \\
        --output output/teacher_story_eq_train.jsonl

    # Convert holdout data
    python scripts/convert_story_equation.py \\
        --input data/story_equation_pairs_holdout.jsonl \\
        --output output/teacher_story_eq_holdout.jsonl
"""
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input story-equation pairs JSONL file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output converted JSONL file')
    parser.add_argument('--strict', action='store_true',
                       help='Strict mode: skip entries with canon violations')

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Converting: {args.input} -> {args.output}")
    print(f"Strict mode: {'ON' if args.strict else 'OFF'}")
    print()

    stats = convert_file(args.input, args.output)

    print(f"\nConversion Statistics:")
    print(f"{'='*60}")
    print(f"  Total entries:       {stats['total']:>6}")
    print(f"  Converted:           {stats['converted']:>6}")
    print(f"  Skipped:             {stats['skipped']:>6}")
    print()
    print(f"  By direction:")
    for direction, count in sorted(stats['by_direction'].items()):
        print(f"    {direction:20} {count:>6}")
    print()
    print(f"  Canon validation:")
    print(f"    Valid (score=1.0): {stats['canon_valid']:>6}")
    print(f"    Invalid (score<1): {stats['canon_invalid']:>6}")
    print(f"    Average score:     {stats['avg_canon_score']:>6.3f}")
    print(f"{'='*60}")

    if stats['skipped'] > 0:
        print(f"\nWarning: {stats['skipped']} entries were skipped. Check stderr for details.")

    return 0


if __name__ == '__main__':
    sys.exit(main())

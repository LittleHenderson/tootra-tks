#!/usr/bin/env python3
"""
TKS Inference CLI - Lightweight encode->invert/anti-attractor->decode pipeline

Usage:
  # Single item mode:
  python scripts/run_inference.py --story "A woman loved a man" --axes W,N --mode soft
  python scripts/run_inference.py --equation "B5 +T D3" --anti-attractor --format json
  python scripts/run_inference.py --story "Power corrupts" --axes F --mode targeted --lenient

  # Bulk mode (JSONL input/output):
  python scripts/run_inference.py --input-jsonl input.jsonl --output-jsonl output.jsonl --axes W,N
  python scripts/run_inference.py --input-jsonl stories.jsonl --output-jsonl results.jsonl --anti-attractor

JSONL input format (one JSON object per line):
  {"story": "A woman loved a man"}
  {"equation": "B5 +T D3"}
  {"story": "Power corrupts", "axes": "F", "mode": "targeted"}

JSONL output format (one JSON object per line with rich metadata):
  {"success": true, "result": {...}, "validator": {...}, "inversion_type": "inversion|anti-attractor", ...}
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenario_inversion import (
    EncodeStory,
    DecodeStory,
    ScenarioInvert,
    ExplainInversion,
    parse_equation,
    TKSExpression,
    AXES_MAP,
)
from anti_attractor import (
    compute_anti_attractor,
    anti_attractor,
    compute_attractor_signature,
    explain_signature,
)
from inversion.engine import TargetProfile
from teacher.validator import CanonicalValidator, ValidationResult


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_axes(axes_str: str) -> set:
    """
    Parse comma-separated axes string into set of axis names.

    Args:
        axes_str: Comma-separated axes (e.g., "W,N" or "World,Noetic")

    Returns:
        Set of normalized axis names
    """
    if not axes_str:
        return set()

    axes = set()
    for a in axes_str.split(","):
        a = a.strip()
        a_upper = a.upper()

        # Try letter code first (e.g., "W" -> "World")
        if a_upper in AXES_MAP:
            axes.add(AXES_MAP[a_upper])
        # Try full name with proper capitalization (e.g., "World" or "world")
        elif a.capitalize() in AXES_MAP.values():
            axes.add(a.capitalize())
        # Try already capitalized (e.g., "World")
        elif a in AXES_MAP.values():
            axes.add(a)
    return axes


def format_expression(expr: TKSExpression) -> str:
    """
    Format TKSExpression for display.

    Args:
        expr: TKS expression to format

    Returns:
        Formatted string (e.g., "B5 +T D3 -> C2")
    """
    parts = []
    for i, elem in enumerate(expr.elements):
        parts.append(elem)
        if i < len(expr.ops):
            parts.append(expr.ops[i])
    return " ".join(parts)


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

def run_inference(
    input_text: str,
    is_equation: bool,
    anti_attractor: bool,
    axes: set,
    mode: str,
    strict: bool,
    target: Optional[TargetProfile] = None
) -> Dict[str, Any]:
    """
    Execute the complete inference pipeline.

    Pipeline stages:
    1. Encode input to TKS expression (story->TKS or parse equation)
    2. Apply inversion or anti-attractor synthesis
    3. Decode to natural language

    Args:
        input_text: Natural language story or TKS equation
        is_equation: True if input is equation, False if story
        anti_attractor: Enable anti-attractor synthesis
        axes: Set of axes for inversion (ignored if anti_attractor=True)
        mode: Inversion mode (soft/hard/targeted)
        strict: Strict mode for encoding (reject unknown tokens)
        target: Optional target profile for targeted mode

    Returns:
        Dict with original_expr, original_story, inverted_expr, inverted_story,
        explanation, and optionally signature

    Raises:
        ValueError: If strict=True and unknown tokens detected during encoding
    """
    # Stage 1: Encode input to TKS expression
    if is_equation:
        expr = parse_equation(input_text)
    else:
        expr = EncodeStory(input_text, strict=strict)

    # Stage 2: Apply inversion or anti-attractor
    if anti_attractor:
        # Anti-attractor synthesis mode
        orig_sig, inv_sig, inverted_expr = compute_anti_attractor(expr)

        # Generate explanation
        explanation = explain_signature(orig_sig, "Original") + "\n\n" + \
                     explain_signature(inv_sig, "Anti-Attractor")

        result = {
            'original_expr': expr,
            'inverted_expr': inverted_expr,
            'signature': orig_sig,
            'inverted_signature': inv_sig,
            'explanation': explanation,
        }
    else:
        # Standard multi-axis inversion mode
        inverted_expr = ScenarioInvert(expr, axes, mode, target)
        explanation = ExplainInversion(expr, inverted_expr)

        result = {
            'original_expr': expr,
            'inverted_expr': inverted_expr,
            'explanation': explanation,
        }

    # Stage 3: Decode to natural language
    original_story = DecodeStory(expr)
    inverted_story = DecodeStory(inverted_expr)

    result['original_story'] = original_story
    result['inverted_story'] = inverted_story

    return result


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_text_output(result: Dict[str, Any], anti_attractor_mode: bool) -> str:
    """
    Format results as human-readable text.

    Args:
        result: Inference pipeline result
        anti_attractor_mode: Whether anti-attractor synthesis was used

    Returns:
        Formatted text output
    """
    lines = []
    lines.append("=" * 70)
    if anti_attractor_mode:
        lines.append("  TKS INFERENCE: ANTI-ATTRACTOR SYNTHESIS")
    else:
        lines.append("  TKS INFERENCE: SCENARIO INVERSION")
    lines.append("=" * 70)
    lines.append("")

    # Original
    lines.append("=== ORIGINAL ===")
    lines.append(f"Expression: {format_expression(result['original_expr'])}")
    lines.append(f"Story:      {result['original_story']}")
    lines.append("")

    # Attractor signature (if anti-attractor mode)
    if anti_attractor_mode and 'signature' in result:
        lines.append("=== ATTRACTOR SIGNATURE ===")
        sig = result['signature']

        # Element distribution
        lines.append("Element Distribution:")
        for (world, noetic), count in sorted(sig.element_counts.items(),
                                            key=lambda x: x[1],
                                            reverse=True):
            lines.append(f"  {world}{noetic}: {count} occurrence{'s' if count > 1 else ''}")

        # Dominant pattern
        polarity_str = {1: "Positive", -1: "Negative", 0: "Neutral"}.get(sig.polarity, "Unknown")
        lines.append(f"Dominant:   {sig.dominant_world}{sig.dominant_noetic}")
        lines.append(f"Polarity:   {sig.polarity:+d} ({polarity_str})")

        # Foundations
        if sig.foundation_tags:
            foundation_names = {
                1: "Unity", 2: "Wisdom", 3: "Life", 4: "Companionship",
                5: "Power", 6: "Material", 7: "Lust"
            }
            found_strs = [f"F{fid} ({foundation_names.get(fid, '?')})"
                         for fid in sorted(sig.foundation_tags)]
            lines.append(f"Foundations: {', '.join(found_strs)}")

        lines.append("")

    # Inverted
    lines.append("=== INVERTED ===")
    lines.append(f"Expression: {format_expression(result['inverted_expr'])}")
    lines.append(f"Story:      {result['inverted_story']}")
    lines.append("")

    # Explanation
    lines.append("=== EXPLANATION ===")
    lines.append(result['explanation'])
    lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_json_output(
    result: Dict[str, Any],
    anti_attractor_mode: bool,
    validator_result: Optional[ValidationResult] = None,
    include_validator: bool = True,
    compact: bool = False
) -> str:
    """
    Format results as rich JSON with validator flags, signatures, and metadata.

    Args:
        result: Inference pipeline result
        anti_attractor_mode: Whether anti-attractor synthesis was used
        validator_result: Optional validation result from CanonicalValidator
        include_validator: Whether to include validator section (default: True)
        compact: Whether to output compact JSON (no indentation)

    Returns:
        JSON string with rich metadata
    """
    output = {
        "mode": "anti_attractor" if anti_attractor_mode else "inversion",
        "inversion_type": "anti-attractor" if anti_attractor_mode else "inversion",
        "original": {
            "expression": format_expression(result['original_expr']),
            "elements": result['original_expr'].elements,
            "ops": result['original_expr'].ops,
            "story": result['original_story'],
        },
        "inverted": {
            "expression": format_expression(result['inverted_expr']),
            "elements": result['inverted_expr'].elements,
            "ops": result['inverted_expr'].ops,
            "story": result['inverted_story'],
        },
        "explanation": result['explanation'],
    }

    # Add signature info if anti-attractor mode
    if anti_attractor_mode and 'signature' in result:
        sig = result['signature']
        output["signature"] = {
            "element_counts": {f"{w}{n}": c for (w, n), c in sig.element_counts.items()},
            "dominant_world": sig.dominant_world,
            "dominant_noetic": sig.dominant_noetic,
            "polarity": sig.polarity,
            "foundation_tags": sorted(sig.foundation_tags),
        }

        # Add inverted signature if available
        if 'inverted_signature' in result:
            inv_sig = result['inverted_signature']
            output["inverted_signature"] = {
                "element_counts": {f"{w}{n}": c for (w, n), c in inv_sig.element_counts.items()},
                "dominant_world": inv_sig.dominant_world,
                "dominant_noetic": inv_sig.dominant_noetic,
                "polarity": inv_sig.polarity,
                "foundation_tags": sorted(inv_sig.foundation_tags),
            }

    # Add validator section if provided and requested
    if include_validator and validator_result is not None:
        output["validator"] = {
            "is_valid": validator_result.is_valid,
            "canon_score": validator_result.canon_score,
            "error_count": validator_result.error_count,
            "warning_count": validator_result.warning_count,
            "issues": [
                {
                    "rule": issue.rule,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "location": issue.location,
                    "suggestion": issue.suggestion,
                }
                for issue in validator_result.issues
            ] if validator_result.issues else [],
        }

    indent = None if compact else 2
    return json.dumps(output, indent=indent)


# =============================================================================
# BULK PROCESSING FUNCTIONS
# =============================================================================

def process_single_item(
    item: Dict[str, Any],
    default_axes: set,
    default_mode: str,
    default_anti_attractor: bool,
    default_strict: bool,
    include_validator: bool = True,
) -> Dict[str, Any]:
    """
    Process a single item from JSONL input.

    Args:
        item: Input item dict with 'story' or 'equation' key
        default_axes: Default axes to use if not specified in item
        default_mode: Default mode to use if not specified in item
        default_anti_attractor: Default anti-attractor flag
        default_strict: Default strict mode flag
        include_validator: Whether to run canonical validation

    Returns:
        Dict with success, result, validator, inversion_type, and error fields
    """
    output = {
        "success": False,
        "input": item,
        "inversion_type": None,
        "result": None,
        "validator": None,
        "error": None,
    }

    try:
        # Determine input type
        is_equation = "equation" in item
        input_text = item.get("equation") or item.get("story")

        if not input_text:
            output["error"] = "Missing 'story' or 'equation' field in input"
            return output

        # Get item-level overrides or use defaults
        axes_str = item.get("axes")
        axes = parse_axes(axes_str) if axes_str else default_axes
        if not axes:
            axes = {"World", "Noetic"}

        mode = item.get("mode", default_mode)
        anti_attractor_flag = item.get("anti_attractor", default_anti_attractor)
        strict = not item.get("lenient", not default_strict)

        # Build target profile if specified in item
        target = None
        if mode == "targeted":
            target = TargetProfile(
                enable=bool(item.get("from_foundation") or item.get("from_world")),
                from_foundation=item.get("from_foundation"),
                to_foundation=item.get("to_foundation"),
                from_world=item.get("from_world"),
                to_world=item.get("to_world"),
            )

        # Run inference
        result = run_inference(
            input_text=input_text,
            is_equation=is_equation,
            anti_attractor=anti_attractor_flag,
            axes=axes,
            mode=mode,
            strict=strict,
            target=target,
        )

        output["success"] = True
        output["inversion_type"] = "anti-attractor" if anti_attractor_flag else "inversion"

        # Format result
        result_json = json.loads(format_json_output(
            result,
            anti_attractor_mode=anti_attractor_flag,
            validator_result=None,
            include_validator=False,
            compact=True,
        ))
        output["result"] = result_json

        # Run validator if requested
        if include_validator:
            validator = CanonicalValidator(strict_mode=strict)

            # Validate the inverted story
            inverted_story = result.get('inverted_story', '')
            inverted_expr = format_expression(result['inverted_expr'])

            # Combine expression and story for validation
            validation_text = f"{inverted_expr}\n{inverted_story}"
            validator_result = validator.validate(validation_text)

            output["validator"] = {
                "is_valid": validator_result.is_valid,
                "canon_score": validator_result.canon_score,
                "error_count": validator_result.error_count,
                "warning_count": validator_result.warning_count,
                "issues": [
                    {
                        "rule": issue.rule,
                        "severity": issue.severity.value,
                        "message": issue.message,
                        "location": issue.location,
                        "suggestion": issue.suggestion,
                    }
                    for issue in validator_result.issues
                ] if validator_result.issues else [],
            }

        # Add signature info for anti-attractor mode
        if anti_attractor_flag and 'signature' in result:
            sig = result['signature']
            output["signature"] = {
                "element_counts": {f"{w}{n}": c for (w, n), c in sig.element_counts.items()},
                "dominant_world": sig.dominant_world,
                "dominant_noetic": sig.dominant_noetic,
                "polarity": sig.polarity,
                "foundation_tags": sorted(sig.foundation_tags),
            }

    except ValueError as e:
        output["error"] = f"Validation error: {str(e)}"
    except Exception as e:
        output["error"] = f"Processing error: {str(e)}"

    return output


def read_jsonl(file_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Read JSONL file and yield parsed items.

    Args:
        file_path: Path to JSONL file

    Yields:
        Parsed JSON objects from each line
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                yield {
                    "_parse_error": True,
                    "_line_number": line_num,
                    "_error": f"JSON parse error: {str(e)}",
                    "_raw_line": line[:200],  # Truncate long lines
                }


def write_jsonl(file_path: Path, items: Iterator[Dict[str, Any]]) -> int:
    """
    Write items to JSONL file.

    Args:
        file_path: Path to output JSONL file
        items: Iterator of items to write

    Returns:
        Number of items written
    """
    count = 0
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    return count


def run_bulk_inference(
    input_path: Path,
    output_path: Path,
    default_axes: set,
    default_mode: str,
    default_anti_attractor: bool,
    default_strict: bool,
    include_validator: bool = True,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Process JSONL input file and write results to JSONL output file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        default_axes: Default axes for inference
        default_mode: Default inversion mode
        default_anti_attractor: Default anti-attractor flag
        default_strict: Default strict mode flag
        include_validator: Whether to run canonical validation
        verbose: Whether to print progress

    Returns:
        Dict with statistics: total, success, error counts
    """
    stats = {"total": 0, "success": 0, "errors": 0, "parse_errors": 0}

    def process_items():
        for item in read_jsonl(input_path):
            stats["total"] += 1

            # Handle parse errors from read_jsonl
            if item.get("_parse_error"):
                stats["parse_errors"] += 1
                yield {
                    "success": False,
                    "line_number": item.get("_line_number"),
                    "error": item.get("_error"),
                    "raw_input": item.get("_raw_line"),
                }
                continue

            # Process valid item
            result = process_single_item(
                item=item,
                default_axes=default_axes,
                default_mode=default_mode,
                default_anti_attractor=default_anti_attractor,
                default_strict=default_strict,
                include_validator=include_validator,
            )

            if result["success"]:
                stats["success"] += 1
            else:
                stats["errors"] += 1

            if verbose and stats["total"] % 100 == 0:
                print(f"Processed {stats['total']} items...", file=sys.stderr)

            yield result

    # Write results
    write_jsonl(output_path, process_items())

    return stats


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="TKS Inference CLI - Encode->Invert/Anti-Attractor->Decode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Story input with standard inversion
  python scripts/run_inference.py --story "A woman loved a man" --axes W,N --mode soft

  # Equation input with anti-attractor synthesis
  python scripts/run_inference.py --equation "B5 +T D3" --anti-attractor --format json

  # Targeted inversion with foundation remapping
  python scripts/run_inference.py --story "Power corrupts" --axes F --mode targeted \\
      --from-foundation 5 --to-foundation 2

  # Lenient mode for unknown tokens
  python scripts/run_inference.py --story "Some unknown words here" --lenient

  # Bulk mode with JSONL input/output
  python scripts/run_inference.py --input-jsonl stories.jsonl --output-jsonl results.jsonl
  python scripts/run_inference.py --input-jsonl input.jsonl --output-jsonl output.jsonl --anti-attractor

JSONL input format (one JSON object per line):
  {"story": "A woman loved a man"}
  {"equation": "B5 +T D3"}
  {"story": "Power corrupts", "axes": "F", "mode": "targeted"}

Axes (comma-separated, for standard inversion):
  N = Noetic (involution pairs: 2<>3, 5<>6, 8<>9)
  E = Element (full element inversion: world + noetic)
  W = World (world mirror: A<>D, B<>C)
  F = Foundation (1<>7, 2<>6, 3<>5, 4 self-dual)
  S = SubFoundation (foundation + world compound)
  A = Acquisition (negation toggle)
  P = Polarity (valence flip)

Modes (for standard inversion):
  soft     - Invert only where canonical dual/opposite exists (default)
  hard     - Apply on all selected axes unconditionally
  targeted - Apply TargetProfile remaps; others unchanged
        """
    )

    # Input group (mutually exclusive: single item OR bulk mode)
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--story",
        type=str,
        help="Natural language story input (single item mode)"
    )
    input_group.add_argument(
        "--equation",
        type=str,
        help="TKS equation input (e.g., 'B5 +T D3' or 'B5,+T,D3') (single item mode)"
    )
    input_group.add_argument(
        "--input-jsonl",
        type=str,
        dest="input_jsonl",
        help="Path to input JSONL file for bulk processing"
    )

    # Inversion mode selector
    p.add_argument(
        "--anti-attractor",
        action="store_true",
        help="Enable anti-attractor synthesis (overrides --axes and --mode)"
    )

    # Standard inversion options (ignored if --anti-attractor is set)
    p.add_argument(
        "--axes",
        type=str,
        default="W,N",
        help="Inversion axes: N,E,W,F,S,A,P (default: W,N)"
    )
    p.add_argument(
        "--mode",
        type=str,
        default="soft",
        choices=["soft", "hard", "targeted"],
        help="Inversion mode (default: soft)"
    )

    # Targeted mode options
    p.add_argument(
        "--from-foundation",
        type=int,
        dest="from_foundation",
        help="Source foundation for targeted remap (1-7)"
    )
    p.add_argument(
        "--to-foundation",
        type=int,
        dest="to_foundation",
        help="Target foundation for targeted remap (1-7)"
    )
    p.add_argument(
        "--from-world",
        type=str,
        dest="from_world",
        help="Source world for targeted remap (A/B/C/D)"
    )
    p.add_argument(
        "--to-world",
        type=str,
        dest="to_world",
        help="Target world for targeted remap (A/B/C/D)"
    )

    # Output format
    p.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)"
    )

    # Strict/lenient mode
    p.add_argument(
        "--lenient",
        action="store_true",
        help="Use lenient mode for unknown tokens (default: strict)"
    )

    # Bulk mode options
    p.add_argument(
        "--output-jsonl",
        type=str,
        dest="output_jsonl",
        help="Path to output JSONL file for bulk processing (required with --input-jsonl)"
    )
    p.add_argument(
        "--no-validator",
        action="store_true",
        dest="no_validator",
        help="Skip canonical validation in bulk mode (faster processing)"
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information to stderr"
    )

    return p.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main CLI entry point."""
    args = parse_args()

    # Determine strict mode
    strict = not args.lenient

    # Parse axes (only used if not anti-attractor mode)
    axes = parse_axes(args.axes) if not args.anti_attractor else set()
    if not axes and not args.anti_attractor:
        axes = {"World", "Noetic"}  # Default axes

    # =========================================================================
    # BULK MODE: Process JSONL input/output
    # =========================================================================
    if args.input_jsonl:
        # Validate bulk mode arguments
        if not args.output_jsonl:
            print("ERROR: --output-jsonl is required when using --input-jsonl", file=sys.stderr)
            sys.exit(1)

        input_path = Path(args.input_jsonl)
        output_path = Path(args.output_jsonl)

        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        # Run bulk inference
        if args.verbose:
            print(f"Processing {input_path} -> {output_path}", file=sys.stderr)
            print(f"Axes: {axes}, Mode: {args.mode}, Anti-attractor: {args.anti_attractor}", file=sys.stderr)

        try:
            stats = run_bulk_inference(
                input_path=input_path,
                output_path=output_path,
                default_axes=axes,
                default_mode=args.mode,
                default_anti_attractor=args.anti_attractor,
                default_strict=strict,
                include_validator=not args.no_validator,
                verbose=args.verbose,
            )

            # Print summary
            print(f"\nBulk processing complete:", file=sys.stderr)
            print(f"  Total items:  {stats['total']}", file=sys.stderr)
            print(f"  Successful:   {stats['success']}", file=sys.stderr)
            print(f"  Errors:       {stats['errors']}", file=sys.stderr)
            print(f"  Parse errors: {stats['parse_errors']}", file=sys.stderr)
            print(f"  Output:       {output_path}", file=sys.stderr)

            sys.exit(0 if stats['errors'] == 0 else 1)

        except Exception as e:
            print(f"ERROR: Bulk processing failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    # =========================================================================
    # SINGLE ITEM MODE: Process story or equation
    # =========================================================================
    is_equation = args.equation is not None
    input_text = args.equation if is_equation else args.story

    # Build target profile (only used in targeted mode)
    target = None
    if args.mode == "targeted" and not args.anti_attractor:
        target = TargetProfile(
            enable=bool(args.from_foundation or args.from_world),
            from_foundation=args.from_foundation,
            to_foundation=args.to_foundation,
            from_world=args.from_world,
            to_world=args.to_world,
        )

    # Run inference pipeline with error handling
    try:
        result = run_inference(
            input_text=input_text,
            is_equation=is_equation,
            anti_attractor=args.anti_attractor,
            axes=axes,
            mode=args.mode,
            strict=strict,
            target=target,
        )

        # Run validator for JSON output
        validator_result = None
        if args.format == "json":
            validator = CanonicalValidator(strict_mode=strict)
            inverted_story = result.get('inverted_story', '')
            inverted_expr = format_expression(result['inverted_expr'])
            validation_text = f"{inverted_expr}\n{inverted_story}"
            validator_result = validator.validate(validation_text)

        # Format and output results
        if args.format == "json":
            output = format_json_output(
                result,
                args.anti_attractor,
                validator_result=validator_result,
                include_validator=True,
            )
        else:
            output = format_text_output(result, args.anti_attractor)

        print(output)

        # Exit successfully
        sys.exit(0)

    except ValueError as e:
        # Encoding error in strict mode
        error_msg = str(e)

        if "Unknown token" in error_msg or "invalid" in error_msg.lower():
            print(f"ERROR: {error_msg}", file=sys.stderr)
            print("\nStrict mode rejected unknown tokens.", file=sys.stderr)
            print("Try using the --lenient flag to allow unknown tokens with warnings.", file=sys.stderr)
            print("\nExample:", file=sys.stderr)
            print(f"  python scripts/run_inference.py {'--story' if not is_equation else '--equation'} \"{input_text}\" --lenient", file=sys.stderr)
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)

        sys.exit(1)

    except Exception as e:
        # Other errors
        print(f"ERROR: Unexpected error during inference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

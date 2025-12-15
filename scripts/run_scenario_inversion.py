#!/usr/bin/env python3
"""
TKS Scenario Inversion Knob CLI

Usage:
  python scripts/run_scenario_inversion.py --story "She loved him" --axes W,N --mode soft
  python scripts/run_scenario_inversion.py --equation "B5,+T,D3" --axes E --mode hard
  python scripts/run_scenario_inversion.py --story "Power corrupts" --axes F,A --mode targeted --from-foundation 5 --to-foundation 2
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenario_inversion import (
    InvertStory,
    ScenarioInvert,
    ExplainInversion,
    parse_equation,
    DecodeStory,
    AXES_MAP,
)
from inversion.engine import TargetProfile
from anti_attractor import anti_attractor, compute_attractor_signature


def parse_axes(axes_str: str) -> set:
    """Parse comma-separated axes string into set of axis names."""
    if not axes_str:
        return set()

    axes = set()
    for a in axes_str.upper().split(","):
        a = a.strip()
        if a in AXES_MAP:
            axes.add(AXES_MAP[a])
        elif a in AXES_MAP.values():
            axes.add(a)
    return axes


def format_expression(expr) -> str:
    """Format TKSExpression for display."""
    parts = []
    for i, elem in enumerate(expr.elements):
        parts.append(elem)
        if i < len(expr.ops):
            parts.append(expr.ops[i])
    return " ".join(parts)


def output_text(result: dict, original_input: str, input_type: str, anti_attractor_mode: bool = False):
    """Output results in text format."""
    print("=" * 60)
    if anti_attractor_mode:
        print("  TKS ANTI-ATTRACTOR SYNTHESIS")
    else:
        print("  TKS SCENARIO INVERSION KNOB")
    print("=" * 60)
    print()

    print("=== ORIGINAL ===")
    if input_type == "story":
        print(f"Story: {original_input}")
    print(f"Equation: {format_expression(result['expr_original'])}")
    print()

    if anti_attractor_mode:
        print("=== ATTRACTOR SIGNATURE ===")
        if 'signature' in result:
            sig = result['signature']
            print(f"Element counts: {dict(sig.element_counts)}")
            print(f"Dominant world: {sig.dominant_world}")
            print(f"Dominant noetic: N{sig.dominant_noetic}")
            print(f"Polarity: {sig.polarity} ({'positive' if sig.polarity > 0 else 'negative' if sig.polarity < 0 else 'neutral'})")
            print(f"Foundation tags: {sorted(sig.foundation_tags)}")
            print()

    print("=== INVERTED ===")
    print(f"Equation: {format_expression(result['expr_inverted'])}")
    print(f"Story: {result['story_inverted']}")
    print()

    if not anti_attractor_mode:
        print("=== EXPLANATION ===")
        explanation = ExplainInversion(result["expr_original"], result["expr_inverted"])
        print(explanation)
        print()
    print("=" * 60)


def output_json(result: dict, original_input: str, input_type: str):
    """Output results in JSON format."""
    output = {
        "input_type": input_type,
        "original_input": original_input,
        "expr_original": {
            "elements": result["expr_original"].elements,
            "ops": result["expr_original"].ops,
        },
        "expr_inverted": {
            "elements": result["expr_inverted"].elements,
            "ops": result["expr_inverted"].ops,
        },
        "story_inverted": result["story_inverted"],
        "explanation": ExplainInversion(result["expr_original"], result["expr_inverted"]),
    }
    print(json.dumps(output, indent=2))


def parse_args():
    p = argparse.ArgumentParser(
        description="TKS Scenario Inversion Knob CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_scenario_inversion.py --story "She loved him" --axes W,N --mode soft
  python scripts/run_scenario_inversion.py --equation "B5,+T,D3" --axes E --mode hard
  python scripts/run_scenario_inversion.py --story "Power corrupts" --axes F --mode targeted --from-foundation 5 --to-foundation 2

Axes (comma-separated):
  N = Noetic (involution pairs: 2<->3, 5<->6, 8<->9)
  E = Element (full element inversion: world + noetic)
  W = World (world mirror: A<->D, B<->C)
  F = Foundation (1<->7, 2<->6, 3<->5, 4 self-dual)
  S = SubFoundation (foundation + world compound)
  A = Acquisition (negation toggle)
  P = Polarity (valence flip)

Modes:
  soft     - Invert only where canonical dual/opposite exists
  hard     - Apply on all selected axes unconditionally
  targeted - Apply TargetProfile remaps; others unchanged
        """
    )

    # Input group (mutually exclusive)
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--story",
        help="Natural language story input"
    )
    input_group.add_argument(
        "--equation",
        help="TKS equation input (e.g., 'B5,+T,D3' or 'B5 +T D3')"
    )

    # Inversion configuration
    p.add_argument(
        "--axes",
        default="E",
        help="Comma-separated axes: N,E,W,F,S,A,P (default: E)"
    )
    p.add_argument(
        "--mode",
        default="soft",
        choices=["soft", "hard", "targeted"],
        help="Inversion mode (default: soft)"
    )

    # Target profile for targeted mode
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
        dest="from_world",
        help="Source world for targeted remap (A/B/C/D)"
    )
    p.add_argument(
        "--to-world",
        dest="to_world",
        help="Target world for targeted remap (A/B/C/D)"
    )

    # Output format
    p.add_argument(
        "--format",
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)"
    )

    # Strict mode (default) with lenient opt-out
    p.add_argument(
        "--lenient",
        action="store_true",
        help="Allow unknown tokens with warnings (default: strict mode rejects unknown tokens)"
    )

    # Anti-attractor mode
    p.add_argument(
        "--anti-attractor",
        action="store_true",
        dest="anti_attractor",
        help="Generate counter-scenario using anti-attractor synthesis (ignores --axes and --mode)"
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Determine strict mode: strict by default, unless --lenient is specified
    strict = not args.lenient

    # Process input based on anti-attractor mode
    if args.anti_attractor:
        # Anti-attractor synthesis mode
        if args.story:
            from scenario_inversion import EncodeStory
            input_type = "story"
            original_input = args.story
            expr_original = EncodeStory(args.story, strict=strict)
        else:
            input_type = "equation"
            original_input = args.equation
            expr_original = parse_equation(args.equation)

        # Compute signature and generate counter-scenario
        signature = compute_attractor_signature(expr_original)
        expr_inverted = anti_attractor(expr_original)
        story_inverted = DecodeStory(expr_inverted)

        result = {
            "expr_original": expr_original,
            "expr_inverted": expr_inverted,
            "story_inverted": story_inverted,
            "signature": signature,
        }

        # Output
        if args.format == "json":
            output_json(result, original_input, input_type)
        else:
            output_text(result, original_input, input_type, anti_attractor_mode=True)

    else:
        # Standard inversion mode
        # Parse axes
        axes = parse_axes(args.axes)
        if not axes:
            axes = {"Element"}  # Default

        # Build target profile
        target = TargetProfile(
            enable=bool(args.from_foundation or args.from_world),
            from_foundation=args.from_foundation,
            to_foundation=args.to_foundation,
            from_world=args.from_world,
            to_world=args.to_world,
        )

        # Process input
        if args.story:
            input_type = "story"
            original_input = args.story
            result = InvertStory(args.story, axes, args.mode, target, strict=strict)
        else:
            input_type = "equation"
            original_input = args.equation
            expr_original = parse_equation(args.equation)
            expr_inverted = ScenarioInvert(expr_original, axes, args.mode, target)
            story_inverted = DecodeStory(expr_inverted)
            result = {
                "expr_original": expr_original,
                "expr_inverted": expr_inverted,
                "story_inverted": story_inverted,
            }

        # Output
        if args.format == "json":
            output_json(result, original_input, input_type)
        else:
            output_text(result, original_input, input_type)


if __name__ == "__main__":
    main()

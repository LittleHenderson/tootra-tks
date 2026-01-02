"""
Inversion Integration Module

Integrates the TKS inversion engine with the TKS wrapper to provide
inversion capabilities for TKSAnalysisResult objects.

This module allows you to:
1. Invert a complete TKS analysis result
2. Generate all possible inversion types
3. Format inverted equations in readable ways

Example:
    from tks_wrapper import TKSWrapper
    from tks_features import invert_analysis, get_all_inversions

    wrapper = TKSWrapper(provider="mock")
    result = wrapper.analyze("I want to start a business")

    # Get opposite inversion
    inverted = invert_analysis(result, mode=InversionMode.Opposite)
    print(inverted.equation)

    # Get all inversion types
    all_inversions = get_all_inversions(result)
    for inv in all_inversions:
        print(f"{inv.mode}: {inv.equation}")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from inversion engine
from inversion.engine import (
    InversionMode,
    DialConfig,
    TargetProfile,
    total_inversion,
    invert_elements,
    invert_foundations,
    invert_acquisitions
)

# Import from TKS wrapper
from tks_wrapper import (
    TKSAnalysisResult,
    TKSEquationParsed,
    TKSElement,
    RPMAnalysis
)


@dataclass
class InvertedAnalysisResult:
    """Result of inverting a TKS analysis."""

    # Original analysis
    original_equation: str
    original_interpretation: str

    # Inverted analysis
    inverted_equation: str
    inverted_elements: List[str]
    inverted_interpretation: str

    # Inversion metadata
    mode: str
    dial_config: DialConfig

    # Optional foundation/acquisition inversions
    inverted_foundations: Optional[List[Tuple[int, str]]] = None
    inverted_acquisitions: Optional[List[str]] = None

    def __str__(self) -> str:
        """Return a formatted string representation."""
        return format_inverted_equation(
            self.original_equation,
            self.inverted_equation,
            self.mode
        )

    def summary(self) -> str:
        """Return a detailed summary of the inversion."""
        lines = [
            "=" * 60,
            f"TKS INVERSION ANALYSIS - {self.mode}",
            "=" * 60,
            "",
            "ORIGINAL EQUATION:",
            f"  {self.original_equation}",
            "",
            "INVERTED EQUATION:",
            f"  {self.inverted_equation}",
            "",
            "INVERTED ELEMENTS:",
        ]

        for elem in self.inverted_elements:
            lines.append(f"  - {elem}")

        if self.inverted_foundations:
            lines.append("")
            lines.append("INVERTED FOUNDATIONS:")
            for fid, world in self.inverted_foundations:
                lines.append(f"  - Foundation {fid} in World {world}")

        if self.inverted_acquisitions:
            lines.append("")
            lines.append("INVERTED ACQUISITIONS:")
            for acq in self.inverted_acquisitions:
                lines.append(f"  - {acq}")

        lines.append("")
        lines.append("INTERPRETATION:")
        lines.append(f"  {self.inverted_interpretation}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def _parse_equation_to_elements(equation: TKSEquationParsed) -> Tuple[List[str], List[str]]:
    """
    Parse TKSEquationParsed into element strings and operators.

    Args:
        equation: Parsed TKS equation

    Returns:
        Tuple of (element_strings, operators)
    """
    element_strings = [elem.code for elem in equation.elements]
    operators = equation.operators
    return element_strings, operators


def _rebuild_equation_string(elements: List[str], operators: List[str]) -> str:
    """
    Rebuild equation string from elements and operators.

    Args:
        elements: List of element codes (e.g., ["A1", "B2", "C3"])
        operators: List of operators (e.g., ["+T", "-T"])

    Returns:
        Formatted equation string (e.g., "A1 +T B2 -T C3")
    """
    if not elements:
        return ""

    if len(elements) == 1:
        return elements[0]

    parts = [elements[0]]
    for i, op in enumerate(operators):
        if i + 1 < len(elements):
            parts.append(op)
            parts.append(elements[i + 1])

    return " ".join(parts)


def _generate_inverted_interpretation(
    original_result: TKSAnalysisResult,
    inverted_elements: List[str],
    mode: InversionMode
) -> str:
    """
    Generate interpretation text for inverted equation.

    Args:
        original_result: Original TKS analysis result
        inverted_elements: List of inverted element codes
        mode: Inversion mode used

    Returns:
        Interpretation string
    """
    mode_descriptions = {
        InversionMode.Opposite: "opposite polarity",
        InversionMode.Dual: "dual complementary",
        InversionMode.Mirror: "mirror reflection",
        InversionMode.ReverseCausal: "reversed causality",
        InversionMode.ParallelAnalogue: "parallel analogue",
        InversionMode.FoundationFlip: "foundation inversion",
        InversionMode.Polarity: "polarity shift",
        InversionMode.CounterPole: "counterpole transformation"
    }

    mode_desc = mode_descriptions.get(mode, "transformation")

    interpretation = (
        f"This is the {mode_desc} of the original situation. "
        f"The inverted equation represents an alternative perspective where "
        f"the fundamental dynamics are transformed. "
    )

    if mode == InversionMode.Opposite:
        interpretation += (
            "Elements have been inverted to their opposite polarities "
            "(Positive<->Negative, Male<->Female, Cause<->Effect), and worlds "
            "have been flipped (A<->D, B<->C). This reveals the complementary "
            "dynamics that balance the original situation."
        )
    elif mode == InversionMode.Mirror:
        interpretation += (
            "The causal chain has been reversed, showing the situation "
            "from the opposite temporal or causal direction."
        )
    elif mode == InversionMode.ReverseCausal:
        interpretation += (
            "Both the causal direction and temporal flow have been reversed, "
            "revealing what would need to happen for the opposite outcome."
        )

    return interpretation


def invert_analysis(
    tks_result: TKSAnalysisResult,
    mode: InversionMode = InversionMode.Opposite,
    dial_config: Optional[DialConfig] = None,
    target_profile: Optional[TargetProfile] = None
) -> InvertedAnalysisResult:
    """
    Invert a TKS analysis result using the specified inversion mode.

    This function takes a complete TKSAnalysisResult and applies the
    inversion engine to produce an inverted version of the analysis.

    Args:
        tks_result: Original TKS analysis result to invert
        mode: Inversion mode to apply (default: Opposite)
        dial_config: Optional dial configuration for fine-grained control
        target_profile: Optional target profile for directed inversions

    Returns:
        InvertedAnalysisResult containing the inverted analysis

    Example:
        >>> wrapper = TKSWrapper(provider="mock")
        >>> result = wrapper.analyze("I want success")
        >>> inverted = invert_analysis(result, mode=InversionMode.Opposite)
        >>> print(inverted.inverted_equation)
        D3 -T A9  # (inverted from B2 +T D8)
    """
    # Create dial config if not provided
    if dial_config is None:
        dial_config = DialConfig(mode=mode)
        if target_profile:
            dial_config.target_profile = target_profile

    # Parse the original equation into components
    element_strings, operators = _parse_equation_to_elements(tks_result.equation)

    # Perform the inversion using the engine
    inversion_result = total_inversion(
        elements=element_strings,
        foundations=None,  # Could extract from tks_result.foundations if needed
        acquisitions=None,  # Could extract if present
        ops=operators,
        mode=mode,
        dial=dial_config
    )

    # Extract inverted components
    inverted_elements = inversion_result["elements"]
    inverted_ops = inversion_result["ops"]
    inverted_foundations = inversion_result.get("foundations")
    inverted_acquisitions = inversion_result.get("acquisitions")

    # Rebuild the inverted equation string
    inverted_equation_str = _rebuild_equation_string(inverted_elements, inverted_ops)

    # Generate interpretation
    inverted_interpretation = _generate_inverted_interpretation(
        tks_result,
        inverted_elements,
        mode
    )

    # Create and return the result
    return InvertedAnalysisResult(
        original_equation=tks_result.equation.raw,
        original_interpretation=tks_result.interpretation,
        inverted_equation=inverted_equation_str,
        inverted_elements=inverted_elements,
        inverted_interpretation=inverted_interpretation,
        mode=mode.value,
        dial_config=dial_config,
        inverted_foundations=inverted_foundations,
        inverted_acquisitions=inverted_acquisitions
    )


def get_all_inversions(
    tks_result: TKSAnalysisResult,
    modes: Optional[List[InversionMode]] = None
) -> List[InvertedAnalysisResult]:
    """
    Generate all inversion types for a TKS analysis result.

    This function applies multiple inversion modes to the same analysis,
    allowing you to explore different transformational perspectives.

    Args:
        tks_result: Original TKS analysis result
        modes: Optional list of specific modes to generate. If None,
               generates all major inversion types.

    Returns:
        List of InvertedAnalysisResult objects, one for each mode

    Example:
        >>> wrapper = TKSWrapper(provider="mock")
        >>> result = wrapper.analyze("I'm torn between options")
        >>> inversions = get_all_inversions(result)
        >>> for inv in inversions:
        ...     print(f"{inv.mode}: {inv.inverted_equation}")
        Opposite: D3 -T A9
        Dual: C4 +T B8
        Mirror: A9 -T D3
        ...
    """
    if modes is None:
        # Default to the most commonly used inversion modes
        modes = [
            InversionMode.Opposite,
            InversionMode.Dual,
            InversionMode.Mirror,
            InversionMode.ReverseCausal,
            InversionMode.CounterPole,
            InversionMode.ParallelAnalogue,
            InversionMode.FoundationFlip,
            InversionMode.Polarity
        ]

    inversions = []
    for mode in modes:
        try:
            inverted = invert_analysis(tks_result, mode=mode)
            inversions.append(inverted)
        except Exception as e:
            # Log error but continue with other modes
            print(f"Warning: Failed to generate {mode.value} inversion: {e}")
            continue

    return inversions


def format_inverted_equation(
    original: str,
    inverted: str,
    mode: str,
    include_arrows: bool = True
) -> str:
    """
    Format an inverted equation for display.

    Creates a nicely formatted string showing the original equation,
    the transformation arrow, and the inverted result.

    Args:
        original: Original equation string
        inverted: Inverted equation string
        mode: Inversion mode name
        include_arrows: Whether to include transformation arrows

    Returns:
        Formatted string

    Example:
        >>> formatted = format_inverted_equation(
        ...     "B2 +T D8",
        ...     "C3 -T A9",
        ...     "Opposite"
        ... )
        >>> print(formatted)
        B2 +T D8  ---[Opposite]--->  C3 -T A9
    """
    if include_arrows:
        return f"{original}  ---[{mode}]--->  {inverted}"
    else:
        return f"{mode}:\n  Original: {original}\n  Inverted: {inverted}"


def compare_inversions(
    tks_result: TKSAnalysisResult,
    modes: Optional[List[InversionMode]] = None
) -> str:
    """
    Generate a comparison table of multiple inversions.

    Args:
        tks_result: Original TKS analysis result
        modes: Optional list of modes to compare

    Returns:
        Formatted comparison string

    Example:
        >>> wrapper = TKSWrapper(provider="mock")
        >>> result = wrapper.analyze("I want to achieve my goals")
        >>> comparison = compare_inversions(result)
        >>> print(comparison)
    """
    inversions = get_all_inversions(tks_result, modes)

    lines = [
        "=" * 80,
        "TKS INVERSION COMPARISON",
        "=" * 80,
        "",
        f"ORIGINAL: {tks_result.equation.raw}",
        "",
        "INVERSIONS:",
        "-" * 80
    ]

    for inv in inversions:
        lines.append(f"{inv.mode:20s} | {inv.inverted_equation}")

    lines.append("-" * 80)
    lines.append("")

    return "\n".join(lines)


# Convenience function for quick inversion
def quick_invert(equation_str: str, mode: InversionMode = InversionMode.Opposite) -> str:
    """
    Quickly invert an equation string without full TKS analysis.

    Args:
        equation_str: Equation string like "B2 +T D8"
        mode: Inversion mode

    Returns:
        Inverted equation string

    Example:
        >>> quick_invert("B2 +T D8", InversionMode.Opposite)
        'C3 -T A9'
    """
    # Parse equation string manually
    import re

    element_pattern = r'([ABCD])(\d{1,2})'
    operator_pattern = r'(\+T|-T|\+|-|->|<-|\*T|/T|o)'

    elements = re.findall(element_pattern, equation_str)
    operators = re.findall(operator_pattern, equation_str)

    element_strings = [f"{w}{n}" for w, n in elements]

    # Apply inversion
    dial = DialConfig(mode=mode)
    result = total_inversion(
        elements=element_strings,
        ops=operators,
        mode=mode,
        dial=dial
    )

    return _rebuild_equation_string(result["elements"], result["ops"])


if __name__ == "__main__":
    # Demo usage
    print("TKS Inversion Integration Module")
    print("=" * 60)
    print()
    print("This module integrates the inversion engine with TKS wrapper.")
    print()
    print("Example usage:")
    print()
    print("  from tks_wrapper import TKSWrapper")
    print("  from tks_features import invert_analysis, get_all_inversions")
    print()
    print("  wrapper = TKSWrapper(provider='mock')")
    print("  result = wrapper.analyze('I want success')")
    print()
    print("  # Get opposite inversion")
    print("  inverted = invert_analysis(result)")
    print("  print(inverted.inverted_equation)")
    print()
    print("  # Get all inversions")
    print("  all_inv = get_all_inversions(result)")
    print("  for inv in all_inv:")
    print("      print(f'{inv.mode}: {inv.inverted_equation}')")
    print()

    # Quick demo
    print("Quick demo with equation string:")
    demo_eq = "B2 +T D8 -T C3"
    print(f"  Original: {demo_eq}")

    inverted_opposite = quick_invert(demo_eq, InversionMode.Opposite)
    print(f"  Opposite: {inverted_opposite}")

    inverted_mirror = quick_invert(demo_eq, InversionMode.Mirror)
    print(f"  Mirror:   {inverted_mirror}")

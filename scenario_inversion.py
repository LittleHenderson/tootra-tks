"""
TKS Scenario Inversion Knob - Helper Module
Exposes: ScenarioInvert, InvertStory, ExplainInversion

Wraps inversion/engine.py with axis/mode configuration from the spec.
Uses narrative module for real EncodeStory/DecodeStory.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field

from inversion.engine import (
    total_inversion,
    DialConfig,
    InversionMode,
    TargetProfile,
    WORLD_OPP,
    NOETIC_OPPOSITE,
    FOUNDATION_OPP,
)

# Import real encoder/decoder from narrative module
from narrative import (
    EncodeStory as _encode_story,
    DecodeStory as _decode_story,
    TKSExpression as NarrativeTKSExpression,
)


# Axis enumeration matching spec
AXES_MAP = {
    "N": "Noetic",
    "E": "Element",
    "W": "World",
    "F": "Foundation",
    "S": "SubFoundation",
    "A": "Acquisition",
    "P": "Polarity",
}

# Mode mapping
MODE_MAP = {
    "soft": InversionMode.Opposite,      # Soft uses standard inversion
    "hard": InversionMode.Opposite,      # Hard applies more aggressively
    "targeted": InversionMode.ParallelAnalogue,  # Targeted uses profile remaps
}


@dataclass
class TKSExpression:
    """Parsed TKS expression structure (legacy compatibility wrapper)."""
    elements: List[str] = field(default_factory=list)
    ops: List[str] = field(default_factory=list)
    foundations: List[Tuple[int, Optional[str]]] = field(default_factory=list)
    acquisitions: List[str] = field(default_factory=list)
    raw: str = ""

    @classmethod
    def from_narrative(cls, expr: NarrativeTKSExpression) -> "TKSExpression":
        """Convert from narrative module TKSExpression."""
        return cls(
            elements=list(expr.elements),
            ops=list(expr.ops),
            foundations=list(expr.foundations) if expr.foundations else [],
            acquisitions=list(expr.acquisitions) if expr.acquisitions else [],
            raw=expr.raw,
        )

    def to_narrative(self) -> NarrativeTKSExpression:
        """Convert to narrative module TKSExpression."""
        return NarrativeTKSExpression(
            elements=list(self.elements),
            ops=list(self.ops),
            foundations=list(self.foundations),
            acquisitions=list(self.acquisitions),
            raw=self.raw,
        )


# =============================================================================
# Narrative Encode/Decode (Real Implementation from narrative module)
# =============================================================================

def EncodeStory(story: str, strict: bool = True) -> TKSExpression:
    """
    Encode natural language story to TKS expression.

    Uses the real encoder from narrative module per Narrative Semantics Rulebook v1.0.

    Args:
        story: Natural language story text
        strict: If True (default), raise ValueError when unknown tokens are detected.
                If False, warn/skip unknown tokens.
                CLI default: strict=True (use --lenient to disable)

    Returns:
        TKSExpression

    Raises:
        ValueError: If strict=True and unknown tokens are detected
    """
    narrative_expr = _encode_story(story, strict=strict)
    return TKSExpression.from_narrative(narrative_expr)


def DecodeStory(expr: TKSExpression) -> str:
    """
    Decode TKS expression to natural language story.

    Uses the real decoder from narrative module per Narrative Semantics Rulebook v1.0.
    """
    narrative_expr = expr.to_narrative()
    return _decode_story(narrative_expr)


# =============================================================================
# Core API Functions
# =============================================================================

def build_dial_config(
    axes: Set[str],
    mode: str,
    target: Optional[TargetProfile] = None
) -> DialConfig:
    """Build DialConfig from axes set and mode."""
    # Normalize axes to full names
    has_noetic = "Noetic" in axes or "N" in axes
    has_element = "Element" in axes or "E" in axes
    has_world = "World" in axes or "W" in axes
    has_foundation = "Foundation" in axes or "F" in axes
    has_subfoundation = "SubFoundation" in axes or "S" in axes
    has_acquisition = "Acquisition" in axes or "A" in axes
    has_polarity = "Polarity" in axes or "P" in axes

    # Determine InversionMode based on axes and mode
    inv_mode = InversionMode.Opposite  # Default

    # World-only (without Element or Noetic): use DomainPermutation for world-only inversion
    if has_world and not has_element and not has_noetic:
        inv_mode = InversionMode.DomainPermutation

    # For targeted mode with profile
    if mode.lower() == "targeted" and target and target.enable:
        if target.from_world or target.to_world:
            inv_mode = InversionMode.ParallelAnalogue
        elif target.from_foundation or target.to_foundation:
            inv_mode = InversionMode.FoundationFlip

    # Build axes dict - element processing needed if any element-level axis selected
    element_processing = has_element or has_world or has_noetic or has_polarity

    axes_dict = {
        "noetic": has_noetic or has_polarity,  # Polarity implies noetic processing
        "element": element_processing,
        "world": has_world,
        "foundation": has_foundation,
        "subFoundation": has_subfoundation,
        "acquisition": has_acquisition,
        "causal": True,  # Always enable causal chain processing
        "scalarValence": has_polarity,
    }

    return DialConfig(
        mode=inv_mode,
        axes=axes_dict,
        intensity="hard" if mode.lower() == "hard" else "soft",
        target_profile=target or TargetProfile(),
    )


def ScenarioInvert(
    expr: TKSExpression,
    axes: Set[str],
    mode: str,
    target: Optional[TargetProfile] = None
) -> TKSExpression:
    """
    Apply multi-axis inversion to TKS expression.

    Args:
        expr: TKS expression to invert
        axes: Set of axis names or letters (N,E,W,F,S,A,P)
        mode: Inversion mode (soft, hard, targeted)
        target: Optional target profile for targeted mode

    Returns:
        Inverted TKS expression
    """
    dial = build_dial_config(axes, mode, target)

    # Convert foundations: None -> "" for engine compatibility
    # Engine expects List[Tuple[int, str]], narrative uses List[Tuple[int, Optional[str]]]
    engine_foundations = [(fid, subfound or "") for fid, subfound in expr.foundations]

    result = total_inversion(
        elements=expr.elements,
        foundations=engine_foundations,
        acquisitions=expr.acquisitions,
        ops=expr.ops,
        mode=dial.mode,
        dial=dial,
    )

    # Convert foundations back: "" -> None for narrative compatibility
    result_foundations = [
        (fid, subfound if subfound else None)
        for fid, subfound in result.get("foundations", [])
    ]

    return TKSExpression(
        elements=result["elements"],
        ops=result["ops"],
        foundations=result_foundations,
        acquisitions=result.get("acquisitions", []),
        raw=expr.raw,
    )


def InvertStory(
    story: str,
    axes: Set[str],
    mode: str,
    target: Optional[TargetProfile] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Full pipeline: story -> inverted story.

    Args:
        story: Natural language story
        axes: Set of axis names or letters
        mode: Inversion mode
        target: Optional target profile
        strict: If True (default), raise ValueError when unknown tokens are detected.
                If False, warn/skip unknown tokens.
                CLI default: strict=True (use --lenient to disable)

    Returns:
        Dict with expr_original, expr_inverted, story_inverted

    Raises:
        ValueError: If strict=True and unknown tokens are detected during encoding
    """
    expr_original = EncodeStory(story, strict=strict)
    expr_inverted = ScenarioInvert(expr_original, axes, mode, target)
    story_inverted = DecodeStory(expr_inverted)

    return {
        "expr_original": expr_original,
        "expr_inverted": expr_inverted,
        "story_inverted": story_inverted,
    }


def ExplainInversion(
    expr_original: TKSExpression,
    expr_inverted: TKSExpression
) -> str:
    """
    Generate human-readable explanation of inversion changes.

    Args:
        expr_original: Original TKS expression
        expr_inverted: Inverted TKS expression

    Returns:
        Explanation string
    """
    changes = []

    # Compare elements
    for i, (orig, inv) in enumerate(zip(expr_original.elements, expr_inverted.elements)):
        if orig != inv:
            orig_world, orig_noetic = orig[0], orig[1:]
            inv_world, inv_noetic = inv[0], inv[1:]

            parts = []
            if orig_world != inv_world:
                parts.append(f"World: {orig_world}->{inv_world}")
            if orig_noetic != inv_noetic:
                parts.append(f"Noetic: N{orig_noetic}->N{inv_noetic}")

            changes.append(f"Element {i+1}: {orig} -> {inv} ({', '.join(parts)})")

    # Compare operators
    for i, (orig, inv) in enumerate(zip(expr_original.ops, expr_inverted.ops)):
        if orig != inv:
            changes.append(f"Operator {i+1}: {orig} -> {inv}")

    if not changes:
        return "No changes detected (expression may be self-dual on selected axes)."

    return "INVERSION CHANGES:\n" + "\n".join(f"  {c}" for c in changes)


def parse_equation(equation: str) -> TKSExpression:
    """
    Parse equation string into TKSExpression.

    Accepts formats:
        - "B5,+T,D3,-T,C8" (comma-separated)
        - "B5 +T D3 -T C8" (space-separated)
        - "B5 -> D3 +T C8" (with causal arrows)

    Uses the parser from narrative module.
    """
    from narrative import parse_equation as _parse_equation
    narrative_expr = _parse_equation(equation)
    return TKSExpression.from_narrative(narrative_expr)


def AntiAttractorInvert(
    expr: TKSExpression,
    return_signature: bool = False
) -> Dict[str, Any]:
    """
    Generate anti-attractor counter-scenario for a TKS expression.

    This function applies anti-attractor synthesis, which:
    1. Analyzes the attractor signature (element frequencies, polarity, foundations)
    2. Inverts the signature to create an anti-attractor profile
    3. Synthesizes a counter-scenario expression from the inverted signature

    Args:
        expr: TKS expression to invert
        return_signature: If True, include attractor signature in result

    Returns:
        Dict with expr_inverted and optionally signature

    Example:
        >>> expr = parse_equation("B2 -> C2 +T D2")
        >>> result = AntiAttractorInvert(expr, return_signature=True)
        >>> print(result['expr_inverted'].elements)
        ['C3', 'B3', 'A3']
    """
    from anti_attractor import anti_attractor, compute_attractor_signature

    result = {"expr_inverted": anti_attractor(expr)}

    if return_signature:
        result["signature"] = compute_attractor_signature(expr)

    return result

"""
TKS Canonical Syntax and Expression Rules Module

Defines:
- Symbol position rules (superscripts for noetics, subscripts for foundations)
- Valid expression patterns
- Fractal notation
- Expression parsing utilities
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import re


# =============================================================================
# SYMBOL POSITION RULES
# =============================================================================

@dataclass
class SymbolPositionRules:
    """Rules for symbol positioning in TKS expressions."""

    # Superscripts: Noetic modifications
    # Format: Element^noetic or Foundation^noetic
    # Examples: A2^4, 3d^1
    SUPERSCRIPT_USE = "noetics"
    SUPERSCRIPT_DESCRIPTION = "Noetic modification of an Idea/Element/Foundation"

    # Subscripts: Foundation/Sub-foundation contextualization
    # Format: Element_subfoundation
    # Examples: A2_{3c}, B5_{1a}
    SUBSCRIPT_USE = "foundations"
    SUBSCRIPT_DESCRIPTION = "Foundation/sub-foundation filtering or contextualization"

    # Combined notation allowed
    # Format: Element_{subfoundation}^{noetic}
    # Example: A2_{3c}^{4}
    COMBINED_ALLOWED = True


# =============================================================================
# THREE DISTINCT NAMESPACES (Never Conflate!)
# =============================================================================

NAMESPACES: Dict[str, Dict[str, any]] = {
    "elements": {
        "format": "Letter + Digit",
        "pattern": r"[ABCD](10|[1-9])",
        "examples": ["A1", "B2", "C5", "D10"],
        "count": 40,
        "role": "World-level modes (10 Noetics x 4 Worlds)"
    },
    "subfoundations": {
        "format": "Digit + Lowercase Letter",
        "pattern": r"[1-7][abcd]",
        "examples": ["1a", "3c", "7d"],
        "count": 28,
        "role": "Life-domain structures (7 Foundations x 4 Worlds)"
    },
    "acquisitions": {
        "format": "Letter + Digit or A22",
        "pattern": r"[DWP][1-7]|A22",
        "examples": ["D3", "W5", "P6", "A22"],
        "count": 22,
        "role": "Desire/Wisdom/Power states + Meta-Desire"
    }
}


# =============================================================================
# VALID EXPRESSION PATTERNS
# =============================================================================

# Basic element pattern
ELEMENT_PATTERN = re.compile(r'^([ABCD])(10|[1-9])$')

# Sub-foundation pattern
SUBFOUNDATION_PATTERN = re.compile(r'^([1-7])([abcd])$')

# Element with noetic superscript: A2^4
ELEMENT_WITH_NOETIC = re.compile(r'^([ABCD])(10|[1-9])\^([0-9])$')

# Foundation with noetic superscript: 3d^1
FOUNDATION_WITH_NOETIC = re.compile(r'^([1-7])([abcd])\^([0-9])$')

# Element with foundation subscript: A2_{3c} or A2_3c
ELEMENT_WITH_FOUNDATION = re.compile(r'^([ABCD])(10|[1-9])_\{?([1-7][abcd])\}?$')

# Full expression with both: A2_{3c}^{4} or A2_3c^4
FULL_EXPRESSION = re.compile(r'^([ABCD])(10|[1-9])_\{?([1-7][abcd])\}?\^\{?([0-9])\}?$')

# Operator pattern (for parsing equations)
OPERATOR_PATTERN = re.compile(r'(\+T|-T|\*T|/T|->|<-|\+|-|\*|/|o)')


# =============================================================================
# INVALID PATTERNS (Canonical Prohibitions)
# =============================================================================

INVALID_PATTERNS: List[str] = [
    "^noetic + Foundation",      # Noetic cannot directly add to Foundation
    "noetic - Element",          # Noetic cannot directly subtract from Element
    "Element + noetic",          # Element cannot directly add noetic
    "Foundation / noetic",       # Foundation cannot directly divide by noetic
]

# Noetics MODIFY; they do not ADD directly
GOLDEN_RULE = "Noetics MODIFY other values; they do not participate in direct arithmetic with Elements or Foundations."


# =============================================================================
# FRACTAL NOTATION
# =============================================================================

@dataclass
class FractalNotation:
    """Fractal expression notation rules."""

    # Unicode format
    unicode_format: str = "⟨n₁:n₂:…:nₖ⟩"

    # ASCII fallback
    ascii_format: str = "<<n1:n2:...:nk>>"

    # Application order
    application_order: str = "right_to_left"  # Innermost first

    # Example: ⟨1:4:7⟩(X) = ν₁(ν₄(ν₇(X)))
    # The MVR fractal applies Rhythm first, then Vibration, then Mind


# Fractal pattern for parsing
FRACTAL_PATTERN_UNICODE = re.compile(r'⟨([\d:]+)⟩')
FRACTAL_PATTERN_ASCII = re.compile(r'<<([\d:]+)>>')


def parse_fractal(fractal_str: str) -> Optional[List[int]]:
    """
    Parse a fractal notation string into list of noetics.

    Supports both Unicode (⟨1:4:7⟩) and ASCII (<<1:4:7>>) formats.
    Returns noetics in left-to-right order (outermost first).
    """
    # Try Unicode first
    match = FRACTAL_PATTERN_UNICODE.search(fractal_str)
    if match:
        return [int(n) for n in match.group(1).split(":")]

    # Try ASCII
    match = FRACTAL_PATTERN_ASCII.search(fractal_str)
    if match:
        return [int(n) for n in match.group(1).split(":")]

    return None


def format_fractal(noetics: List[int], use_unicode: bool = True) -> str:
    """
    Format a list of noetics as fractal notation.

    Args:
        noetics: List of noetic indices (outermost first)
        use_unicode: Use Unicode brackets if True

    Returns:
        Formatted fractal string
    """
    inner = ":".join(str(n) for n in noetics)
    if use_unicode:
        return f"⟨{inner}⟩"
    return f"<<{inner}>>"


# =============================================================================
# EXPRESSION PARSING UTILITIES
# =============================================================================

@dataclass
class ParsedExpression:
    """Result of parsing a TKS expression."""
    raw: str
    elements: List[str]
    operators: List[str]
    noetic_modifiers: Dict[str, int]    # element -> noetic
    foundation_filters: Dict[str, str]   # element -> subfoundation
    fractals: List[List[int]]
    valid: bool
    errors: List[str]


def parse_simple_equation(equation: str) -> ParsedExpression:
    """
    Parse a simple TKS equation into components.

    Example: "A2 +T B3 -T C4" -> elements=['A2','B3','C4'], operators=['+T','-T']
    """
    errors = []
    elements = []
    operators = []
    noetic_mods = {}
    foundation_filters = {}
    fractals = []

    # Normalize whitespace
    equation = equation.strip()

    # Split by operators
    parts = OPERATOR_PATTERN.split(equation)

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Check if it's an operator
        if OPERATOR_PATTERN.fullmatch(part):
            operators.append(part)
            continue

        # Try to parse as element (possibly with modifiers)

        # Check for full expression (element + foundation + noetic)
        match = FULL_EXPRESSION.match(part)
        if match:
            elem = f"{match.group(1)}{match.group(2)}"
            elements.append(elem)
            foundation_filters[elem] = match.group(3)
            noetic_mods[elem] = int(match.group(4))
            continue

        # Check for element with foundation only
        match = ELEMENT_WITH_FOUNDATION.match(part)
        if match:
            elem = f"{match.group(1)}{match.group(2)}"
            elements.append(elem)
            foundation_filters[elem] = match.group(3)
            continue

        # Check for element with noetic only
        match = ELEMENT_WITH_NOETIC.match(part)
        if match:
            elem = f"{match.group(1)}{match.group(2)}"
            elements.append(elem)
            noetic_mods[elem] = int(match.group(3))
            continue

        # Check for plain element
        match = ELEMENT_PATTERN.match(part)
        if match:
            elements.append(part)
            continue

        # Check for fractal
        fractal = parse_fractal(part)
        if fractal:
            fractals.append(fractal)
            continue

        # Unknown part
        errors.append(f"Unknown expression part: {part}")

    # Validate operator count
    expected_ops = len(elements) - 1 if elements else 0
    if len(operators) != expected_ops and elements:
        errors.append(f"Expected {expected_ops} operators, got {len(operators)}")

    return ParsedExpression(
        raw=equation,
        elements=elements,
        operators=operators,
        noetic_modifiers=noetic_mods,
        foundation_filters=foundation_filters,
        fractals=fractals,
        valid=len(errors) == 0,
        errors=errors
    )


def validate_expression_syntax(equation: str) -> Tuple[bool, List[str]]:
    """
    Validate the syntax of a TKS expression.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    parsed = parse_simple_equation(equation)
    return (parsed.valid, parsed.errors)


__all__ = [
    "SymbolPositionRules",
    "NAMESPACES",
    "ELEMENT_PATTERN",
    "SUBFOUNDATION_PATTERN",
    "ELEMENT_WITH_NOETIC",
    "FOUNDATION_WITH_NOETIC",
    "ELEMENT_WITH_FOUNDATION",
    "FULL_EXPRESSION",
    "OPERATOR_PATTERN",
    "INVALID_PATTERNS",
    "GOLDEN_RULE",
    "FractalNotation",
    "FRACTAL_PATTERN_UNICODE",
    "FRACTAL_PATTERN_ASCII",
    "ParsedExpression",
    "parse_fractal",
    "format_fractal",
    "parse_simple_equation",
    "validate_expression_syntax",
]

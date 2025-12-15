"""
TKS Narrative Semantics - Decoder

Implements DecodeStory: TKS expression -> Natural language story
Following TKS_Narrative_Semantics_Rulebook_v1.0

Decoding Algorithm:
1. Expression parsing
2. Element sense lookup
3. Operator-to-grammar mapping
4. World-to-layer mapping
5. Domain/codomain-to-flow
6. Tree-to-sentence assembly
7. Narrative smoothing
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .constants import (
    WORLDS,
    NOETIC_NAMES,
    FOUNDATIONS,
    ELEMENT_DEFAULTS,
    SENSE_LABELS,
    OPERATORS,
    SUBFOUND_MAP,
    get_subfound_label,
)
from .types import (
    ElementRef,
    TKSExpression,
    DecodeResult,
)


# =============================================================================
# OPERATOR TEMPLATES
# =============================================================================

OPERATOR_TEMPLATES: Dict[str, Tuple[str, str]] = {
    # (full_template, short_connector)
    # TOOTRA operators (from rulebook)
    "+T": ("{left} together with {right}", "and"),
    "-T": ("{left} without {right}", "without"),
    "*T": ("{left} intensified by {right}", "intensified by"),    # ×_T
    "/T": ("{left} in conflict with {right}", "in conflict with"), # /_T
    # Composition operators
    "o": ("First {left}, then {right}", "then"),  # ∘ (sequential)
    "->": ("{left} leads to {right}", "causes"),  # → (causal)
    "<-": ("{right} leads to {left}", "caused by"),
    # Basic operators
    "+": ("{left} and {right}", "and"),
    "-": ("{left} minus {right}", "minus"),
}


# =============================================================================
# ELEMENT LABELING
# =============================================================================

def get_element_label(elem: str, sense: Optional[int] = None) -> str:
    """
    Get the natural language label for an element.

    Args:
        elem: Element code (e.g., "D5")
        sense: Optional sense index

    Returns:
        Human-readable label
    """
    # Normalize element code
    if len(elem) < 2:
        return f"element {elem}"

    world = elem[0].upper()
    try:
        noetic = int(elem[1:])
    except ValueError:
        return f"element {elem}"

    # Try full sense label first (e.g., "D5.1", "B5.2")
    if sense is not None:
        key = f"{world}{noetic}.{sense}"
        if key in SENSE_LABELS:
            return SENSE_LABELS[key]

    # Try default element label
    base_key = f"{world}{noetic}"
    if base_key in ELEMENT_DEFAULTS:
        return ELEMENT_DEFAULTS[base_key][0]

    # Construct generic label
    world_name = WORLDS.get(world, world).lower()
    noetic_name = NOETIC_NAMES.get(noetic, str(noetic)).lower()

    # Special cases for common patterns
    if noetic in (5, 6):
        if world == "D":
            return "a woman" if noetic == 5 else "a man"
        return f"{world_name} {noetic_name}"
    elif noetic in (2, 3):
        if world == "C":
            return "joy" if noetic == 2 else "fear"
        elif world == "B":
            return "positive belief" if noetic == 2 else "limiting belief"
        return f"{world_name} {'positive' if noetic == 2 else 'negative'}"

    return f"{world_name} {noetic_name}"


def get_element_article(label: str) -> str:
    """
    Get the appropriate article for an element label.

    Returns 'a', 'an', 'the', or empty string as appropriate.
    """
    # Labels that don't need articles
    no_article = {
        "joy", "fear", "anger", "love", "hatred",
        "health", "illness", "wealth", "money",
        "clear thinking", "positive belief", "limiting belief",
    }

    label_lower = label.lower()

    if label_lower in no_article:
        return ""

    # Already has article
    if label_lower.startswith(("a ", "an ", "the ")):
        return ""

    # Vowel start -> 'an'
    if label_lower[0] in "aeiou":
        return "an "

    return "a "


# =============================================================================
# WORLD LAYER DESCRIPTIONS
# =============================================================================

WORLD_LAYER_INTRO: Dict[str, str] = {
    "A": "At the spiritual level",
    "B": "In their mind",
    "C": "Emotionally",
    "D": "Physically",
}

WORLD_LAYER_VERB: Dict[str, str] = {
    "A": "experiences spiritually",
    "B": "thinks about",
    "C": "feels",
    "D": "does",
}


def get_foundation_context(fid: int, world: Optional[str] = None) -> str:
    """
    Get contextual description for a foundation with optional world.

    Args:
        fid: Foundation ID (1-7)
        world: Optional world letter (A/B/C/D)

    Returns:
        Contextual phrase like "in emotional relationship context"
    """
    if world is None:
        # Just foundation name
        foundation_name = FOUNDATIONS.get(fid, f"foundation {fid}").lower()
        return f"in {foundation_name} context"

    # Get sub-foundation label
    subfound_label = get_subfound_label(fid, world)
    if subfound_label:
        return f"in {subfound_label.lower()} context"

    # Fallback to foundation + world
    foundation_name = FOUNDATIONS.get(fid, f"foundation {fid}").lower()
    world_name = WORLDS.get(world, world).lower()
    return f"in {world_name} {foundation_name} context"


# =============================================================================
# SENTENCE BUILDERS
# =============================================================================

def build_element_phrase(
    elem: str,
    sense: Optional[int] = None,
    with_article: bool = True
) -> str:
    """
    Build a natural language phrase for an element.

    Args:
        elem: Element code
        sense: Optional sense index
        with_article: Whether to include article

    Returns:
        Natural language phrase
    """
    label = get_element_label(elem, sense)

    if with_article:
        article = get_element_article(label)
        return f"{article}{label}"

    return label


def build_binary_phrase(
    left: str,
    op: str,
    right: str,
    left_sense: Optional[int] = None,
    right_sense: Optional[int] = None
) -> str:
    """
    Build a phrase for a binary operation.

    Args:
        left: Left element code
        op: Operator
        right: Right element code
        left_sense: Optional left sense
        right_sense: Optional right sense

    Returns:
        Natural language phrase
    """
    left_phrase = build_element_phrase(left, left_sense)
    right_phrase = build_element_phrase(right, right_sense, with_article=False)

    template, connector = OPERATOR_TEMPLATES.get(op, ("{left} with {right}", "with"))

    return template.format(left=left_phrase, right=right_phrase)


# =============================================================================
# MAIN DECODER
# =============================================================================

def DecodeStory(expr: TKSExpression) -> str:
    """
    Decode a TKS expression into a natural language story.

    This is the main entry point for story decoding.

    Args:
        expr: TKS expression to decode

    Returns:
        Natural language story string
    """
    result = decode_story_full(expr)
    return result.story


def decode_story_full(expr: TKSExpression) -> DecodeResult:
    """
    Full decoding with diagnostics.

    Args:
        expr: TKS expression to decode

    Returns:
        DecodeResult with story and diagnostic info
    """
    warnings = []

    if not expr.elements:
        return DecodeResult(
            story="",
            success=False,
            warnings=["Empty expression"]
        )

    # Get element senses if available
    senses = []
    for i, elem in enumerate(expr.elements):
        if i < len(expr.element_refs) and expr.element_refs[i].sense:
            senses.append(expr.element_refs[i].sense)
        else:
            senses.append(None)

    # Get foundation context if available
    foundation_context = None
    if expr.foundations:
        # Use the first foundation tag (most expressions have one main context)
        fid, world = expr.foundations[0]
        foundation_context = get_foundation_context(fid, world)

    # Single element
    if len(expr.elements) == 1:
        phrase = build_element_phrase(expr.elements[0], senses[0])
        story = f"There is {phrase}"
        if foundation_context:
            story += f" {foundation_context}"
        story += "."
        return DecodeResult(story=story, success=True)

    # Two elements
    if len(expr.elements) == 2:
        op = expr.ops[0] if expr.ops else "+T"
        story = build_binary_phrase(
            expr.elements[0], op, expr.elements[1],
            senses[0], senses[1]
        )
        # Capitalize
        story = story[0].upper() + story[1:]
        # Add foundation context if available
        if foundation_context:
            story += f", {foundation_context}"
        story += "."
        return DecodeResult(story=story, success=True)

    # Multiple elements - build complex narrative
    sentences = []
    current_phrase = build_element_phrase(expr.elements[0], senses[0])

    for i in range(len(expr.elements) - 1):
        op = expr.ops[i] if i < len(expr.ops) else "+T"
        next_elem = expr.elements[i + 1]
        next_sense = senses[i + 1] if i + 1 < len(senses) else None
        next_phrase = build_element_phrase(next_elem, next_sense, with_article=False)

        if op in ("->", "<-"):
            # Causal - create new sentence
            sentences.append(f"{current_phrase.capitalize()}")
            if op == "->":
                current_phrase = f"This leads to {next_phrase}"
            else:
                current_phrase = f"{next_phrase} causes this"
        else:
            # Combination - extend phrase
            _, connector = OPERATOR_TEMPLATES.get(op, (None, "and"))
            current_phrase = f"{current_phrase} {connector} {next_phrase}"

    # Add final phrase
    sentences.append(f"{current_phrase.capitalize()}")

    # Join sentences
    if len(sentences) == 1:
        story = sentences[0]
        # Add foundation context for single-sentence complex narratives
        if foundation_context:
            story += f", {foundation_context}"
        story += "."
    else:
        story = ". ".join(sentences)
        # Add foundation context as final sentence for multi-sentence narratives
        if foundation_context:
            story += f", {foundation_context}"
        story += "."

    # Apply smoothing
    story = smooth_narrative(story)

    return DecodeResult(story=story, success=True, warnings=warnings)


# =============================================================================
# NARRATIVE SMOOTHING
# =============================================================================

def smooth_narrative(text: str) -> str:
    """
    Apply narrative smoothing rules to make text more natural.

    Args:
        text: Raw narrative text

    Returns:
        Smoothed narrative
    """
    # Fix double periods
    text = text.replace("..", ".")

    # Fix capitalization after periods
    sentences = text.split(". ")
    smoothed = []
    for s in sentences:
        s = s.strip()
        if s:
            s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
            smoothed.append(s)

    text = ". ".join(smoothed)

    # Ensure single period at end
    text = text.rstrip(".")
    text += "."

    # Fix "a a" -> "a"
    text = text.replace("a a ", "a ")
    text = text.replace("an an ", "an ")

    # Fix article issues
    text = text.replace(" a fear", " fear")
    text = text.replace(" a joy", " joy")
    text = text.replace(" a anger", " anger")
    text = text.replace(" a love", " love")

    return text


# =============================================================================
# EXPRESSION FORMATTER
# =============================================================================

def format_expression(expr: TKSExpression) -> str:
    """
    Format a TKS expression as a canonical string.

    Args:
        expr: TKS expression

    Returns:
        Formatted string like "B5 +T D3 -> C2"
    """
    if not expr.elements:
        return ""

    parts = []
    for i, elem in enumerate(expr.elements):
        # Include sense if available
        if i < len(expr.element_refs) and expr.element_refs[i].sense:
            parts.append(f"{elem}.{expr.element_refs[i].sense}")
        else:
            parts.append(elem)

        if i < len(expr.ops):
            parts.append(expr.ops[i])

    return " ".join(parts)

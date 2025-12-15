# TKS Narrative Semantics Module
# Implements EncodeStory/DecodeStory per Narrative Semantics Rulebook v1.0

from .types import (
    ElementRef,
    FoundationTag,
    AcquisitionTag,
    TKSToken,
    TKSExpression,
    EncodeResult,
    DecodeResult,
)

from .encoder import (
    EncodeStory,
    encode_story_full,
    parse_equation,
)

from .decoder import (
    DecodeStory,
    decode_story_full,
    format_expression,
)

from .constants import (
    WORLDS,
    WORLD_LETTERS,
    NOETIC_NAMES,
    NOETIC_INVOLUTIONS,
    NOETIC_SELF_DUALS,
    FOUNDATIONS,
    FOUNDATION_OPPOSITES,
    OPERATORS,
    OPERATOR_TOKENS,
    LEXICON,
    ELEMENT_DEFAULTS,
    SENSE_LABELS,
    is_valid_world,
    is_valid_noetic,
    is_valid_foundation,
    is_valid_operator,
    validate_element,
)

__all__ = [
    # Types
    "ElementRef",
    "FoundationTag",
    "AcquisitionTag",
    "TKSToken",
    "TKSExpression",
    "EncodeResult",
    "DecodeResult",
    # Encoder
    "EncodeStory",
    "encode_story_full",
    "parse_equation",
    # Decoder
    "DecodeStory",
    "decode_story_full",
    "format_expression",
    # Constants
    "WORLDS",
    "WORLD_LETTERS",
    "NOETIC_NAMES",
    "NOETIC_INVOLUTIONS",
    "NOETIC_SELF_DUALS",
    "FOUNDATIONS",
    "FOUNDATION_OPPOSITES",
    "OPERATORS",
    "OPERATOR_TOKENS",
    "LEXICON",
    "ELEMENT_DEFAULTS",
    "SENSE_LABELS",
    # Validators
    "is_valid_world",
    "is_valid_noetic",
    "is_valid_foundation",
    "is_valid_operator",
    "validate_element",
]

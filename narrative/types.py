"""
TKS Narrative Semantics - Type Definitions

Dataclasses for representing TKS expressions, elements, and encoding results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from .constants import (
    is_valid_world,
    is_valid_noetic,
    is_valid_foundation,
    is_valid_operator,
    ELEMENT_DEFAULTS,
    SENSE_LABELS,
    NOETIC_NAMES,
    WORLDS,
)


@dataclass
class ElementRef:
    """
    A TKS element reference: World + Noetic with optional sense and foundation.

    Canonical forms:
    - Basic: WN (e.g., "D5")
    - With sense (dot): WN.S (e.g., "D5.1")
    - With sense (caret): WN^S (e.g., "B8^5")
    - With foundation: WN_Fw (e.g., "B8_d5" = foundation 5 in world D)
    - Full extended: WN^S_Fw (e.g., "B8^5_d5")

    Where:
    - W: World letter (A/B/C/D)
    - N: Noetic digit (1-10)
    - S: Optional sense index
    - Fw: Foundation suffix (e.g., _d5 = foundation 5 in world D)
    """
    world: str          # A/B/C/D
    noetic: int         # 1-10
    sense: Optional[int] = None  # Sense index if specified
    foundation: Optional[int] = None  # Foundation ID (1-7)
    subfoundation: Optional[str] = None  # World for foundation context (A/B/C/D)

    def __post_init__(self):
        """Validate element reference."""
        if not is_valid_world(self.world):
            raise ValueError(f"Invalid world: {self.world} (must be A/B/C/D)")
        if not is_valid_noetic(self.noetic):
            raise ValueError(f"Invalid noetic: {self.noetic} (must be 1-10)")
        if self.foundation is not None and not is_valid_foundation(self.foundation):
            raise ValueError(f"Invalid foundation: {self.foundation} (must be 1-7)")
        if self.subfoundation is not None and not is_valid_world(self.subfoundation):
            raise ValueError(f"Invalid subfoundation world: {self.subfoundation} (must be A/B/C/D)")

    @property
    def code(self) -> str:
        """Return element code without sense (e.g., 'D5')."""
        return f"{self.world}{self.noetic}"

    @property
    def full_code(self) -> str:
        """Return element code with sense and/or foundation if present.

        Examples:
        - 'D5.1' (dot notation for sense)
        - 'B8^5' (caret notation for sense)
        - 'B8_d5' (foundation suffix)
        - 'B8^5_d5' (full extended)
        """
        code = f"{self.world}{self.noetic}"

        # Add sense using caret notation if present
        if self.sense is not None:
            code += f"^{self.sense}"

        # Add foundation suffix if present
        if self.foundation is not None and self.subfoundation is not None:
            code += f"_{self.subfoundation.lower()}{self.foundation}"

        return code

    @property
    def label(self) -> str:
        """Return human-readable label for this element."""
        # Try full sense label first
        if self.sense is not None:
            key = f"{self.code}.{self.sense}"
            if key in SENSE_LABELS:
                return SENSE_LABELS[key]
        # Fall back to default label
        if self.code in ELEMENT_DEFAULTS:
            return ELEMENT_DEFAULTS[self.code][0]
        # Generic label
        world_name = WORLDS.get(self.world, self.world)
        noetic_name = NOETIC_NAMES.get(self.noetic, str(self.noetic))
        return f"{world_name.lower()} {noetic_name.lower()}"

    @classmethod
    def from_string(cls, s: str) -> "ElementRef":
        """
        Parse element from string with extended syntax support.

        Supported formats:
        - Basic: 'D5', 'B8'
        - Dot sense: 'D5.1', 'B8.5'
        - Caret sense: 'D5^1', 'B8^5'
        - Foundation: 'B8_d5' (foundation 5 in world D)
        - Full extended: 'B8^5_d5'

        Args:
            s: Element string

        Returns:
            ElementRef instance
        """
        import re

        s = s.strip()
        if not s:
            raise ValueError("Empty element string")

        # Parse foundation suffix if present: _Fw (e.g., _d5, _a3)
        foundation = None
        subfoundation = None
        if "_" in s:
            main, found_suffix = s.rsplit("_", 1)
            # Parse foundation suffix: should be [a-d][1-7]
            match = re.match(r'^([a-dA-D])([1-7])$', found_suffix)
            if match:
                subfoundation = match.group(1).upper()
                foundation = int(match.group(2))
                s = main
            else:
                raise ValueError(f"Invalid foundation suffix: _{found_suffix} (expected format: _w# where w=A-D, #=1-7)")

        # Parse sense if present (dot or caret notation)
        sense = None
        if "." in s:
            # Dot notation: D5.1
            main, sense_str = s.split(".", 1)
            sense = int(sense_str)
            s = main
        elif "^" in s:
            # Caret notation: B8^5
            main, sense_str = s.split("^", 1)
            sense = int(sense_str)
            s = main

        # Parse world and noetic
        world = s[0].upper()
        noetic_str = s[1:]

        # Handle noetic = 10
        noetic = int(noetic_str)

        return cls(
            world=world,
            noetic=noetic,
            sense=sense,
            foundation=foundation,
            subfoundation=subfoundation
        )

    def __str__(self) -> str:
        return self.full_code

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.code == other or self.full_code == other
        if isinstance(other, ElementRef):
            return self.world == other.world and self.noetic == other.noetic
        return False

    def __hash__(self) -> int:
        return hash((self.world, self.noetic))


@dataclass
class FoundationTag:
    """
    A foundation tag: F1-F7 with optional axis.

    Used to annotate elements or expressions with foundation context.
    """
    foundation: int     # 1-7
    axis: Optional[str] = None  # Optional axis label

    def __post_init__(self):
        if not is_valid_foundation(self.foundation):
            raise ValueError(f"Invalid foundation: {self.foundation} (must be 1-7)")

    @property
    def code(self) -> str:
        """Return foundation code (e.g., 'F3')."""
        return f"F{self.foundation}"

    def __str__(self) -> str:
        if self.axis:
            return f"{self.code}:{self.axis}"
        return self.code


@dataclass
class AcquisitionTag:
    """
    Acquisition marker for element acquisition state.

    Represents whether an element is acquired (+) or negated (-).
    """
    element: str        # Element code
    acquired: bool      # True = acquired (+), False = negated (-)

    @property
    def code(self) -> str:
        """Return acquisition code (e.g., '+D5' or '-D5')."""
        sign = "+" if self.acquired else "-"
        return f"{sign}{self.element}"

    def __str__(self) -> str:
        return self.code


@dataclass
class TKSToken:
    """
    A token in a TKS expression - either an element or operator.
    """
    value: str          # Element code or operator
    is_operator: bool   # True if operator, False if element
    element_ref: Optional[ElementRef] = None  # Parsed element if not operator
    source_span: Optional[Tuple[int, int]] = None  # Character span in source

    @classmethod
    def element(cls, code: str, span: Optional[Tuple[int, int]] = None) -> "TKSToken":
        """Create an element token."""
        ref = ElementRef.from_string(code)
        return cls(value=ref.code, is_operator=False, element_ref=ref, source_span=span)

    @classmethod
    def operator(cls, op: str, span: Optional[Tuple[int, int]] = None) -> "TKSToken":
        """Create an operator token."""
        if not is_valid_operator(op):
            raise ValueError(f"Invalid operator: {op}")
        return cls(value=op, is_operator=True, source_span=span)

    def __str__(self) -> str:
        return self.value


@dataclass
class TKSExpression:
    """
    A TKS expression: sequence of elements connected by operators.

    Canonical form: E1 op1 E2 op2 E3 ...
    Example: "D5 +T C2.3 -> D6"

    This is the main type used throughout the system.
    """
    elements: List[str] = field(default_factory=list)  # Element codes
    ops: List[str] = field(default_factory=list)       # Operators between elements
    foundations: List[Tuple[int, Optional[str]]] = field(default_factory=list)  # Foundation tags
    acquisitions: List[str] = field(default_factory=list)  # Acquisition markers
    raw: str = ""                                       # Original source text

    # Extended attributes for rich encoding
    tokens: List[TKSToken] = field(default_factory=list)  # Full token sequence
    element_refs: List[ElementRef] = field(default_factory=list)  # Parsed elements
    confidence: float = 1.0                             # Encoding confidence (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self):
        """Build element_refs from elements if not provided."""
        if self.elements and not self.element_refs:
            for elem in self.elements:
                try:
                    self.element_refs.append(ElementRef.from_string(elem))
                except (ValueError, IndexError):
                    pass  # Skip invalid elements

    @property
    def canonical(self) -> str:
        """Return canonical string representation."""
        if not self.elements:
            return ""

        parts = []
        for i, elem in enumerate(self.elements):
            parts.append(elem)
            if i < len(self.ops):
                parts.append(self.ops[i])
        return " ".join(parts)

    @property
    def element_count(self) -> int:
        """Return number of elements."""
        return len(self.elements)

    @property
    def is_causal_chain(self) -> bool:
        """Check if expression contains causal operators."""
        return any(op in ("->", "<-") for op in self.ops)

    def get_element(self, index: int) -> Optional[ElementRef]:
        """Get element reference at index."""
        if 0 <= index < len(self.element_refs):
            return self.element_refs[index]
        return None

    def copy(self) -> "TKSExpression":
        """Create a deep copy of this expression."""
        return TKSExpression(
            elements=self.elements.copy(),
            ops=self.ops.copy(),
            foundations=self.foundations.copy(),
            acquisitions=self.acquisitions.copy(),
            raw=self.raw,
            tokens=[t for t in self.tokens],
            element_refs=[ElementRef(e.world, e.noetic, e.sense) for e in self.element_refs],
            confidence=self.confidence,
            metadata=self.metadata.copy(),
        )

    def __str__(self) -> str:
        return self.canonical

    def __len__(self) -> int:
        return len(self.elements)

    def __bool__(self) -> bool:
        return bool(self.elements)


@dataclass
class EncodeResult:
    """
    Result from encoding a story to TKS expression.

    Contains the expression plus diagnostic information.
    """
    expression: TKSExpression
    success: bool = True
    warnings: List[str] = field(default_factory=list)
    unknown_words: List[str] = field(default_factory=list)

    @property
    def canonical(self) -> str:
        """Return canonical expression string."""
        return self.expression.canonical


@dataclass
class DecodeResult:
    """
    Result from decoding a TKS expression to natural language.

    Contains the story plus diagnostic information.
    """
    story: str
    success: bool = True
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.story

"""
TKS Canonical Operators Definition Module

Defines all TKS operators as per v7.4 canonical documents:
- 4 Tootra Arithmetic Operators (+, -, *, /)
- 5 Relational/Sequential Operators (+T, -T, *T, /T, o)
- 2 Causal Operators (->, <-)
- Composition operator (o)
"""

from typing import Dict, Set, List, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorDefinition:
    """Immutable definition of a single operator."""
    symbol: str           # The operator symbol
    name: str             # Full name
    semantics: str        # Meaning/semantics
    category: str         # Category (arithmetic, relational, causal, etc.)
    latex: str            # LaTeX representation
    commutative: bool     # Is the operation commutative?
    associative: bool     # Is the operation associative?


# =============================================================================
# CANONICAL OPERATORS
# =============================================================================

OPERATORS: Dict[str, OperatorDefinition] = {
    # Tootra Arithmetic Operators
    "+": OperatorDefinition(
        symbol="+",
        name="Tootra Addition",
        semantics="Combining, joining, associating Ideas - 'and'",
        category="arithmetic",
        latex=r"\oplus_T",
        commutative=True,
        associative=True
    ),
    "-": OperatorDefinition(
        symbol="-",
        name="Tootra Subtraction",
        semantics="Removing, separating, eliminating - 'minus'",
        category="arithmetic",
        latex=r"\ominus_T",
        commutative=False,
        associative=False
    ),
    "*": OperatorDefinition(
        symbol="*",
        name="Tootra Multiplication",
        semantics="Amplifying, increasing, expanding",
        category="arithmetic",
        latex=r"\otimes_T",
        commutative=True,
        associative=True
    ),
    "/": OperatorDefinition(
        symbol="/",
        name="Tootra Division",
        semantics="Diminishing, apportioning, resolving",
        category="arithmetic",
        latex=r"\oslash_T",
        commutative=False,
        associative=False
    ),

    # Relational Operators (with T suffix for Tootra-specific)
    "+T": OperatorDefinition(
        symbol="+T",
        name="Together With",
        semantics="Together with, accumulates with - associative union",
        category="relational",
        latex=r"\oplus_T",
        commutative=True,
        associative=True
    ),
    "-T": OperatorDefinition(
        symbol="-T",
        name="Without",
        semantics="Without, diminishes from - removal",
        category="relational",
        latex=r"\ominus_T",
        commutative=False,
        associative=False
    ),
    "*T": OperatorDefinition(
        symbol="*T",
        name="Intensified By",
        semantics="Intensified by, amplifies through - multiplicative",
        category="relational",
        latex=r"\otimes_T",
        commutative=True,
        associative=True
    ),
    "/T": OperatorDefinition(
        symbol="/T",
        name="In Conflict With",
        semantics="In conflict with, attenuates through - divisive",
        category="relational",
        latex=r"\oslash_T",
        commutative=False,
        associative=False
    ),

    # Causal Operators
    "->": OperatorDefinition(
        symbol="->",
        name="Causes",
        semantics="Causes, leads to, transforms into",
        category="causal",
        latex=r"\rightarrow",
        commutative=False,
        associative=False
    ),
    "<-": OperatorDefinition(
        symbol="<-",
        name="Caused By",
        semantics="Caused by, receives from, follows from",
        category="causal",
        latex=r"\leftarrow",
        commutative=False,
        associative=False
    ),

    # Composition Operator
    "o": OperatorDefinition(
        symbol="o",
        name="Then/Composition",
        semantics="Then, composes with - sequential application (right to left)",
        category="composition",
        latex=r"\circ",
        commutative=False,
        associative=True  # (f o g) o h = f o (g o h)
    ),
}


# =============================================================================
# OPERATOR SETS
# =============================================================================

# All valid operator symbols
OPERATOR_SYMBOLS: Set[str] = set(OPERATORS.keys())

# Also accept these as valid (aliases)
ALLOWED_OPS: Set[str] = {"+T", "-T", "*T", "/T", "o", "->", "<-", "+", "-", "*", "/"}

# Operators by category
ARITHMETIC_OPS: Set[str] = {"+", "-", "*", "/"}
RELATIONAL_OPS: Set[str] = {"+T", "-T", "*T", "/T"}
CAUSAL_OPS: Set[str] = {"->", "<-"}
COMPOSITION_OPS: Set[str] = {"o"}

# Commutative operators
COMMUTATIVE_OPS: Set[str] = {op for op, defn in OPERATORS.items() if defn.commutative}

# Associative operators
ASSOCIATIVE_OPS: Set[str] = {op for op, defn in OPERATORS.items() if defn.associative}


# =============================================================================
# OPERATOR PRECEDENCE
# =============================================================================

# Higher number = higher precedence (evaluated first)
OPERATOR_PRECEDENCE: Dict[str, int] = {
    "o": 5,      # Composition (highest - function application)
    "*T": 4,     # Multiplicative
    "/T": 4,
    "*": 4,
    "/": 4,
    "+T": 3,     # Additive
    "-T": 3,
    "+": 3,
    "-": 3,
    "->": 2,     # Causal (lowest)
    "<-": 2,
}


# =============================================================================
# OPERATOR VALIDATION AND HELPERS
# =============================================================================

def validate_operator(op: str) -> bool:
    """Check if a string is a valid operator."""
    return op in ALLOWED_OPS


def get_operator_precedence(op: str) -> int:
    """Get the precedence of an operator (higher = evaluated first)."""
    return OPERATOR_PRECEDENCE.get(op, 0)


def is_commutative(op: str) -> bool:
    """Check if an operator is commutative."""
    return op in COMMUTATIVE_OPS


def is_associative(op: str) -> bool:
    """Check if an operator is associative."""
    return op in ASSOCIATIVE_OPS


def get_operator_semantics(op: str) -> str:
    """Get the semantic meaning of an operator."""
    if op in OPERATORS:
        return OPERATORS[op].semantics
    return "Unknown operator"


# =============================================================================
# OPERATOR SEMANTIC MAPPINGS (Alternative Descriptions)
# =============================================================================

OPERATOR_SEMANTICS_ALT: Dict[str, str] = {
    "+": "and",
    "-": "minus",
    "+T": "together with",
    "-T": "without",
    "*T": "intensified by",
    "/T": "in conflict with",
    "o": "then",
    "->": "causes",
    "<-": "caused by",
}


__all__ = [
    "OperatorDefinition",
    "OPERATORS",
    "OPERATOR_SYMBOLS",
    "ALLOWED_OPS",
    "ARITHMETIC_OPS",
    "RELATIONAL_OPS",
    "CAUSAL_OPS",
    "COMPOSITION_OPS",
    "COMMUTATIVE_OPS",
    "ASSOCIATIVE_OPS",
    "OPERATOR_PRECEDENCE",
    "OPERATOR_SEMANTICS_ALT",
    "validate_operator",
    "get_operator_precedence",
    "is_commutative",
    "is_associative",
    "get_operator_semantics",
]

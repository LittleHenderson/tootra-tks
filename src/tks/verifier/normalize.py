"""
TKS Expression Normalizer - Module 2: Truth Plumbing

Normalizes TKS expressions to canonical form.
Handles whitespace, operator standardization, and PEMDAS-aware ordering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .parser import (
    AtomicExpression,
    BinaryExpression,
    FoundationStack,
    GroupExpression,
    Operator,
    TKSExpression,
    WorldNoetic,
    parse_tks,
)


@dataclass
class NormalizationResult:
    """Result of normalizing a TKS expression."""
    original: str
    normalized: str
    canonical_form: str
    ast: TKSExpression
    changes_made: list[str]
    is_valid: bool
    error: Optional[str] = None


class TKSNormalizer:
    """
    Normalizes TKS expressions to canonical form.

    Normalization includes:
    1. Whitespace normalization (single spaces between tokens)
    2. Operator standardization (+T, -T, *T, /T format)
    3. Case normalization (uppercase world codes)
    4. Canonical ordering for symmetric operators
    """

    # Operator normalization mappings
    OPERATOR_CANONICAL = {
        "+": "+T",
        "+T": "+T",
        "-": "-T",
        "-T": "-T",
        "*": "*T",
        "*T": "*T",
        "x": "*T",
        "/": "/T",
        "/T": "/T",
        "div": "/T",
        "o": "o",
        "->": "->",
        "<-": "<-",
        ":": ":",
    }

    def __init__(self):
        self.changes: list[str] = []

    def normalize(self, expression: str) -> NormalizationResult:
        """
        Normalize a TKS expression to canonical form.

        Args:
            expression: Raw TKS expression string

        Returns:
            NormalizationResult with normalized form and metadata
        """
        self.changes = []
        original = expression

        try:
            # Step 1: Pre-processing (whitespace, case)
            preprocessed = self._preprocess(expression)

            # Step 2: Operator standardization
            standardized = self._standardize_operators(preprocessed)

            # Step 3: Parse to AST
            ast = parse_tks(standardized)

            # Step 4: Canonical ordering
            canonical_ast = self._canonical_order(ast)

            # Step 5: Render canonical form
            canonical_form = self._render_canonical(canonical_ast)
            normalized = self._render_normalized(canonical_ast)

            return NormalizationResult(
                original=original,
                normalized=normalized,
                canonical_form=canonical_form,
                ast=canonical_ast,
                changes_made=self.changes,
                is_valid=True,
            )

        except Exception as e:
            return NormalizationResult(
                original=original,
                normalized=original,
                canonical_form=original,
                ast=AtomicExpression(WorldNoetic("A", 1)),  # Placeholder
                changes_made=self.changes,
                is_valid=False,
                error=str(e),
            )

    def _preprocess(self, expression: str) -> str:
        """
        Pre-process expression: normalize whitespace and case.
        """
        original = expression

        # Normalize multiple whitespace to single spaces
        result = re.sub(r"\s+", " ", expression.strip())
        if result != original.strip():
            self.changes.append("Normalized whitespace")

        # Uppercase world codes (preserve operator case temporarily)
        def uppercase_worlds(match: re.Match) -> str:
            return match.group(0).upper()

        result_upper = re.sub(r"[A-Da-d]\d{1,2}", uppercase_worlds, result)
        if result_upper != result:
            self.changes.append("Uppercased world codes")
            result = result_upper

        return result

    def _standardize_operators(self, expression: str) -> str:
        """
        Standardize operators to canonical T-suffixed form.
        """
        result = expression

        # Replace operator variants with canonical forms
        # Order matters: longer patterns first
        replacements = [
            (r"div", "/T"),
            (r"\+T", "+T"),
            (r"-T", "-T"),
            (r"\*T", "*T"),
            (r"/T", "/T"),
            (r"->", "->"),
            (r"<-", "<-"),
            # Non-T versions (but not within world-noetic like A1)
            (r"(?<![A-Da-d0-9])\+(?!T)", "+T"),
            (r"(?<![A-Da-d0-9<>])-(?!T|>)", "-T"),
            (r"(?<![A-Da-d0-9])\*(?!T)", "*T"),
            (r"(?<![A-Da-d0-9])x(?![0-9])", "*T"),
            (r"(?<![A-Da-d0-9])/(?!T)", "/T"),
        ]

        for pattern, replacement in replacements:
            new_result = re.sub(pattern, replacement, result)
            if new_result != result:
                self.changes.append(f"Standardized operator: {pattern} -> {replacement}")
                result = new_result

        return result

    def _canonical_order(self, expr: TKSExpression) -> TKSExpression:
        """
        Apply canonical ordering to the AST.

        For symmetric operators (+T, *T), order operands consistently.
        World codes are ordered alphabetically, then by noetic index.
        """
        if isinstance(expr, AtomicExpression):
            return expr

        if isinstance(expr, GroupExpression):
            return GroupExpression(inner=self._canonical_order(expr.inner))

        if isinstance(expr, FoundationStack):
            # Foundation stacks maintain their order (semantically significant)
            return expr

        if isinstance(expr, BinaryExpression):
            left = self._canonical_order(expr.left)
            right = self._canonical_order(expr.right)

            # For symmetric operators, order operands canonically
            if expr.operator.is_symmetric:
                left_key = self._expression_sort_key(left)
                right_key = self._expression_sort_key(right)

                if left_key > right_key:
                    self.changes.append(
                        f"Reordered symmetric expression: {right} {expr.operator.value} {left}"
                    )
                    left, right = right, left

            return BinaryExpression(left=left, operator=expr.operator, right=right)

        return expr

    def _expression_sort_key(self, expr: TKSExpression) -> tuple:
        """
        Generate a sort key for an expression for canonical ordering.
        """
        if isinstance(expr, AtomicExpression):
            return (
                0,
                expr.element.world,
                expr.element.noetic,
                expr.element.sense or 0,
                expr.element.foundation or 0,
            )
        if isinstance(expr, GroupExpression):
            inner_key = self._expression_sort_key(expr.inner)
            return (1, *inner_key)
        if isinstance(expr, BinaryExpression):
            left_key = self._expression_sort_key(expr.left)
            right_key = self._expression_sort_key(expr.right)
            return (2, expr.operator.precedence, left_key, right_key)
        if isinstance(expr, FoundationStack):
            element_keys = [
                (e.world, e.noetic, e.sense or 0, e.foundation or 0)
                for e in expr.elements
            ]
            return (3, *element_keys)
        return (99,)

    def _render_canonical(self, expr: TKSExpression) -> str:
        """
        Render expression in canonical form (with explicit parentheses).
        """
        if isinstance(expr, AtomicExpression):
            return str(expr.element)

        if isinstance(expr, GroupExpression):
            return f"({self._render_canonical(expr.inner)})"

        if isinstance(expr, BinaryExpression):
            left = self._render_canonical(expr.left)
            right = self._render_canonical(expr.right)
            return f"({left} {expr.operator.value} {right})"

        if isinstance(expr, FoundationStack):
            return ":".join(str(e) for e in expr.elements)

        return str(expr)

    def _render_normalized(self, expr: TKSExpression) -> str:
        """
        Render expression in normalized form (minimal parentheses).
        """
        return self._render_with_precedence(expr, 0)

    def _render_with_precedence(self, expr: TKSExpression, parent_prec: int) -> str:
        """
        Render expression respecting operator precedence.
        """
        if isinstance(expr, AtomicExpression):
            return str(expr.element)

        if isinstance(expr, GroupExpression):
            return f"({self._render_with_precedence(expr.inner, 0)})"

        if isinstance(expr, BinaryExpression):
            op_prec = expr.operator.precedence
            left = self._render_with_precedence(expr.left, op_prec)
            right = self._render_with_precedence(expr.right, op_prec)
            result = f"{left} {expr.operator.value} {right}"

            # Add parens if needed for precedence
            if op_prec < parent_prec:
                result = f"({result})"

            return result

        if isinstance(expr, FoundationStack):
            return ":".join(str(e) for e in expr.elements)

        return str(expr)


def normalize_tks(expression: str) -> NormalizationResult:
    """
    Normalize a TKS expression to canonical form.

    Args:
        expression: Raw TKS expression string

    Returns:
        NormalizationResult with normalized form and metadata
    """
    normalizer = TKSNormalizer()
    return normalizer.normalize(expression)


def quick_normalize(expression: str) -> str:
    """
    Quick normalization returning just the normalized string.

    Args:
        expression: Raw TKS expression string

    Returns:
        Normalized expression string
    """
    result = normalize_tks(expression)
    return result.normalized if result.is_valid else expression


def are_equivalent(expr1: str, expr2: str) -> bool:
    """
    Check if two TKS expressions are semantically equivalent.

    Args:
        expr1: First TKS expression
        expr2: Second TKS expression

    Returns:
        True if expressions normalize to the same canonical form
    """
    result1 = normalize_tks(expr1)
    result2 = normalize_tks(expr2)

    if not result1.is_valid or not result2.is_valid:
        return False

    return result1.canonical_form == result2.canonical_form

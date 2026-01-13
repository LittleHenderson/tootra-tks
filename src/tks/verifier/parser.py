"""
TKS Expression Parser - Module 2: Truth Plumbing

Parses TKS expressions into Abstract Syntax Trees (AST).
Handles world codes, noetic indices, and TKS operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


class TokenType(Enum):
    """Token types for TKS expression lexing."""
    WORLD_NOETIC = "WORLD_NOETIC"  # e.g., A1, B5, C10
    OPERATOR = "OPERATOR"  # +T, -T, *T, /T, :
    LPAREN = "LPAREN"  # (
    RPAREN = "RPAREN"  # )
    EOF = "EOF"


class Operator(Enum):
    """TKS Operators with their semantics."""
    ASSOCIATION = "+T"  # combine/link/coexist/support
    DISASSOCIATION = "-T"  # sever/remove/exclude influence
    INTENSIFICATION = "*T"  # amplify/fuse/strengthen
    OPPOSITION = "/T"  # contrast/tension/dialectic
    COMPOSITION = "o"  # sequential composition
    CAUSES = "->"  # causal forward
    CAUSED_BY = "<-"  # causal reverse
    FOUNDATION_STACK = ":"  # colon-separated foundation stack

    @classmethod
    def from_string(cls, s: str) -> Optional["Operator"]:
        """Parse operator from string representation."""
        mapping = {
            "+T": cls.ASSOCIATION,
            "+": cls.ASSOCIATION,
            "-T": cls.DISASSOCIATION,
            "-": cls.DISASSOCIATION,
            "*T": cls.INTENSIFICATION,
            "*": cls.INTENSIFICATION,
            "x": cls.INTENSIFICATION,
            "/T": cls.OPPOSITION,
            "/": cls.OPPOSITION,
            "div": cls.OPPOSITION,
            "o": cls.COMPOSITION,
            "->": cls.CAUSES,
            "<-": cls.CAUSED_BY,
            ":": cls.FOUNDATION_STACK,
        }
        return mapping.get(s)

    @property
    def precedence(self) -> int:
        """Return operator precedence (higher = binds tighter)."""
        precedence_map = {
            Operator.COMPOSITION: 5,  # o highest
            Operator.INTENSIFICATION: 4,  # * next
            Operator.OPPOSITION: 4,  # / next
            Operator.ASSOCIATION: 3,  # + lower
            Operator.DISASSOCIATION: 3,  # - same as +
            Operator.CAUSES: 2,  # causal lower
            Operator.CAUSED_BY: 2,  # causal lower
            Operator.FOUNDATION_STACK: 0,  # : lowest
        }
        return precedence_map.get(self, 0)

    @property
    def is_symmetric(self) -> bool:
        """Return whether the operator is symmetric."""
        return self in {Operator.ASSOCIATION, Operator.INTENSIFICATION}


@dataclass
class Token:
    """Lexical token from TKS expression."""
    type: TokenType
    value: str
    position: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', pos={self.position})"


@dataclass
class WorldNoetic:
    """Represents a world-noetic element like A1, B5, C10 with optional suffixes."""
    world: str  # A, B, C, D
    noetic: int  # 0-10
    sense: Optional[int] = None  # ^1-^9
    foundation: Optional[int] = None  # _d1-_d7

    def __post_init__(self) -> None:
        if self.world not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid world code: {self.world}. Must be A, B, C, or D.")
        if not 0 <= self.noetic <= 10:
            raise ValueError(f"Invalid noetic index: {self.noetic}. Must be 0-10.")
        if self.sense is not None and not 1 <= self.sense <= 9:
            raise ValueError(f"Invalid sense index: {self.sense}. Must be 1-9.")
        if self.foundation is not None and not 1 <= self.foundation <= 7:
            raise ValueError(
                f"Invalid foundation index: {self.foundation}. Must be 1-7."
            )

    def __str__(self) -> str:
        base = f"{self.world}{self.noetic}"
        if self.sense is not None:
            base += f"^{self.sense}"
        if self.foundation is not None:
            base += f"_d{self.foundation}"
        return base

    def __repr__(self) -> str:
        return (
            "WorldNoetic("
            f"'{self.world}', {self.noetic}, sense={self.sense}, foundation={self.foundation})"
        )

    @classmethod
    def from_string(cls, s: str) -> "WorldNoetic":
        """Parse WorldNoetic from string like 'A1', 'B10^3', 'C2_d4'."""
        match = re.match(
            r"^([A-Da-d])(10|[0-9])(?:\^([1-9]))?(?:_[dD]([1-7]))?$",
            s.strip(),
        )
        if not match:
            raise ValueError(f"Invalid world-noetic format: {s}")
        world = match.group(1).upper()
        noetic = int(match.group(2))
        sense = int(match.group(3)) if match.group(3) else None
        foundation = int(match.group(4)) if match.group(4) else None
        return cls(world=world, noetic=noetic, sense=sense, foundation=foundation)


@dataclass
class TKSExpression:
    """Abstract Syntax Tree node for TKS expressions."""
    pass


@dataclass
class AtomicExpression(TKSExpression):
    """Leaf node: a single world-noetic element."""
    element: WorldNoetic

    def __str__(self) -> str:
        return str(self.element)


@dataclass
class BinaryExpression(TKSExpression):
    """Binary operation: left op right."""
    left: TKSExpression
    operator: Operator
    right: TKSExpression

    def __str__(self) -> str:
        left_str = str(self.left)
        right_str = str(self.right)
        op_str = self.operator.value
        return f"({left_str} {op_str} {right_str})"


@dataclass
class FoundationStack(TKSExpression):
    """Foundation stack: colon-separated sequence like F1:F5:F2."""
    elements: list[WorldNoetic] = field(default_factory=list)

    def __str__(self) -> str:
        return ":".join(str(e) for e in self.elements)


@dataclass
class GroupExpression(TKSExpression):
    """Parenthesized group expression."""
    inner: TKSExpression

    def __str__(self) -> str:
        return f"({self.inner})"


@dataclass
class ParseError(Exception):
    """Error during TKS expression parsing."""
    message: str
    position: int
    token: Optional[Token] = None

    def __str__(self) -> str:
        pos_info = f" at position {self.position}" if self.position >= 0 else ""
        token_info = f" (token: {self.token})" if self.token else ""
        return f"ParseError: {self.message}{pos_info}{token_info}"


class TKSLexer:
    """Lexical analyzer for TKS expressions."""

    # Regex patterns for tokens
    WORLD_NOETIC_PATTERN = re.compile(
        r"[A-Da-d](?:10|[0-9])(?:\^[1-9])?(?:_[dD][1-7])?"
    )
    OPERATOR_PATTERN = re.compile(r"->|<-|\+T?|-T?|\*T?|/T?|div|x|o|:")

    def __init__(self, expression: str):
        self.expression = expression
        self.position = 0
        self.tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        """Tokenize the entire expression."""
        self.tokens = []
        self.position = 0

        while self.position < len(self.expression):
            self._skip_whitespace()
            if self.position >= len(self.expression):
                break

            token = self._next_token()
            if token:
                self.tokens.append(token)

        self.tokens.append(Token(TokenType.EOF, "", self.position))
        return self.tokens

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.position < len(self.expression) and self.expression[self.position].isspace():
            self.position += 1

    def _next_token(self) -> Optional[Token]:
        """Extract the next token from the expression."""
        if self.position >= len(self.expression):
            return None

        char = self.expression[self.position]

        # Check for parentheses
        if char == "(":
            token = Token(TokenType.LPAREN, "(", self.position)
            self.position += 1
            return token

        if char == ")":
            token = Token(TokenType.RPAREN, ")", self.position)
            self.position += 1
            return token

        # Try to match world-noetic (must check before operator due to 'x')
        remaining = self.expression[self.position:]
        wn_match = self.WORLD_NOETIC_PATTERN.match(remaining)
        if wn_match:
            value = wn_match.group().upper()
            token = Token(TokenType.WORLD_NOETIC, value, self.position)
            self.position += len(wn_match.group())
            return token

        # Try to match operator
        op_match = self.OPERATOR_PATTERN.match(remaining)
        if op_match:
            value = op_match.group()
            token = Token(TokenType.OPERATOR, value, self.position)
            self.position += len(value)
            return token

        # Unknown character
        raise ParseError(f"Unexpected character: '{char}'", self.position)


class TKSParser:
    """
    Recursive descent parser for TKS expressions.

    Grammar:
        expression  -> foundation_stack | causal_expr
        foundation_stack -> element (":" element)*
        causal_expr -> additive (("->" | "<-") additive)*
        additive    -> multiplicative (("+T" | "-T") multiplicative)*
        multiplicative -> composition (("*T" | "/T") composition)*
        composition -> factor ("o" factor)*
        factor      -> element | "(" expression ")"
        element     -> WORLD_NOETIC
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.position = 0

    @property
    def current_token(self) -> Token:
        """Get current token."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return Token(TokenType.EOF, "", -1)

    def advance(self) -> Token:
        """Move to next token and return the previous one."""
        token = self.current_token
        self.position += 1
        return token

    def expect(self, token_type: TokenType, value: Optional[str] = None) -> Token:
        """Expect a specific token type/value or raise error."""
        token = self.current_token
        if token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {token.type.name}",
                token.position,
                token
            )
        if value is not None and token.value != value:
            raise ParseError(
                f"Expected '{value}', got '{token.value}'",
                token.position,
                token
            )
        return self.advance()

    def parse(self) -> TKSExpression:
        """Parse the token stream into an AST."""
        if not self.tokens or self.tokens[0].type == TokenType.EOF:
            raise ParseError("Empty expression", 0)

        # Check if this is a foundation stack (contains ':' operator)
        has_colon = any(t.type == TokenType.OPERATOR and t.value == ":" for t in self.tokens)
        has_other_ops = any(
            t.type == TokenType.OPERATOR and t.value != ":"
            for t in self.tokens
        )

        if has_colon and not has_other_ops:
            result = self._parse_foundation_stack()
        else:
            result = self._parse_expression()

        # Ensure we consumed all tokens
        if self.current_token.type != TokenType.EOF:
            raise ParseError(
                f"Unexpected token after expression: {self.current_token.value}",
                self.current_token.position,
                self.current_token
            )

        return result

    def _parse_foundation_stack(self) -> FoundationStack:
        """Parse a foundation stack like F1:F5:F2."""
        elements = []

        # First element
        token = self.expect(TokenType.WORLD_NOETIC)
        elements.append(WorldNoetic.from_string(token.value))

        # Additional elements separated by ':'
        while self.current_token.type == TokenType.OPERATOR and self.current_token.value == ":":
            self.advance()  # consume ':'
            token = self.expect(TokenType.WORLD_NOETIC)
            elements.append(WorldNoetic.from_string(token.value))

        return FoundationStack(elements=elements)

    def _parse_expression(self) -> TKSExpression:
        """Parse a full expression (lowest precedence operators)."""
        return self._parse_causal()

    def _parse_causal(self) -> TKSExpression:
        """Parse causal operators (->, <-)."""
        left = self._parse_additive()

        while self.current_token.type == TokenType.OPERATOR:
            op_str = self.current_token.value
            op = Operator.from_string(op_str)
            if op not in {Operator.CAUSES, Operator.CAUSED_BY}:
                break
            self.advance()
            right = self._parse_additive()
            left = BinaryExpression(left=left, operator=op, right=right)

        return left

    def _parse_additive(self) -> TKSExpression:
        """Parse additive operators (+T, -T)."""
        left = self._parse_multiplicative()

        while self.current_token.type == TokenType.OPERATOR:
            op_str = self.current_token.value
            op = Operator.from_string(op_str)
            if op not in {Operator.ASSOCIATION, Operator.DISASSOCIATION}:
                break
            self.advance()
            right = self._parse_multiplicative()
            left = BinaryExpression(left=left, operator=op, right=right)

        return left

    def _parse_multiplicative(self) -> TKSExpression:
        """Parse multiplicative operators (*T, /T)."""
        left = self._parse_composition()

        while self.current_token.type == TokenType.OPERATOR:
            op_str = self.current_token.value
            op = Operator.from_string(op_str)
            if op not in {Operator.INTENSIFICATION, Operator.OPPOSITION}:
                break
            self.advance()
            right = self._parse_composition()
            left = BinaryExpression(left=left, operator=op, right=right)

        return left

    def _parse_composition(self) -> TKSExpression:
        """Parse composition operator (o)."""
        left = self._parse_factor()

        while self.current_token.type == TokenType.OPERATOR:
            op_str = self.current_token.value
            op = Operator.from_string(op_str)
            if op != Operator.COMPOSITION:
                break
            self.advance()
            right = self._parse_factor()
            left = BinaryExpression(left=left, operator=op, right=right)

        return left

    def _parse_factor(self) -> TKSExpression:
        """Parse a factor (element or parenthesized expression)."""
        token = self.current_token

        if token.type == TokenType.LPAREN:
            self.advance()  # consume '('
            expr = self._parse_expression()
            self.expect(TokenType.RPAREN)
            return GroupExpression(inner=expr)

        if token.type == TokenType.WORLD_NOETIC:
            self.advance()
            element = WorldNoetic.from_string(token.value)
            return AtomicExpression(element=element)

        raise ParseError(
            f"Expected world-noetic or '(', got {token.type.name}",
            token.position,
            token
        )


def parse_tks(expression: str) -> TKSExpression:
    """
    Parse a TKS expression string into an AST.

    Args:
        expression: TKS expression like "A1 +T B2 -T C3" or "A1:B5:C2"

    Returns:
        TKSExpression AST node

    Raises:
        ParseError: If the expression is invalid
    """
    lexer = TKSLexer(expression)
    tokens = lexer.tokenize()
    parser = TKSParser(tokens)
    return parser.parse()


def extract_elements(expr: TKSExpression) -> list[WorldNoetic]:
    """
    Extract all world-noetic elements from an expression.

    Args:
        expr: The TKS expression AST

    Returns:
        List of all WorldNoetic elements in the expression
    """
    elements: list[WorldNoetic] = []

    def visit(node: TKSExpression) -> None:
        if isinstance(node, AtomicExpression):
            elements.append(node.element)
        elif isinstance(node, BinaryExpression):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, FoundationStack):
            elements.extend(node.elements)
        elif isinstance(node, GroupExpression):
            visit(node.inner)

    visit(expr)
    return elements


def extract_operators(expr: TKSExpression) -> list[Operator]:
    """
    Extract all operators from an expression.

    Args:
        expr: The TKS expression AST

    Returns:
        List of all Operators in the expression
    """
    operators: list[Operator] = []

    def visit(node: TKSExpression) -> None:
        if isinstance(node, BinaryExpression):
            visit(node.left)
            operators.append(node.operator)
            visit(node.right)
        elif isinstance(node, GroupExpression):
            visit(node.inner)
        elif isinstance(node, FoundationStack) and len(node.elements) > 1:
            # Foundation stack has implicit ':' operators
            for _ in range(len(node.elements) - 1):
                operators.append(Operator.FOUNDATION_STACK)

    visit(expr)
    return operators

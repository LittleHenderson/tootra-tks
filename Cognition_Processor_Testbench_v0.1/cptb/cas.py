"""CAS state primitives used by the Cognition Processor Testbench.

This module implements the lexical subset needed by the v0.1 testbench.  It is
not a claim to be the full canonical CAS compiler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Iterable


class Domain(str, Enum):
    P = "P"
    C = "C"
    E = "E"
    O = "O"

    @property
    def rank(self) -> int:
        """Higher numbers govern lower numbers: P > C > E > O."""
        return {Domain.P: 3, Domain.C: 2, Domain.E: 1, Domain.O: 0}[self]


NOETIC_NAMES: dict[int, str] = {
    0: "Idea",
    1: "Mind",
    2: "Positive",
    3: "Negative",
    4: "Vibration",
    5: "Female",
    6: "Male",
    7: "Rhythm",
    8: "Above/Cause",
    9: "Below/Effect",
}

INVOLUTION: dict[int, int] = {2: 3, 3: 2, 5: 6, 6: 5, 8: 9, 9: 8}
VALID_REGISTERS = {"T", "A", "f", "E"}

_STATE_RE = re.compile(
    r"^\^(?P<noetic>[0-9])"
    r"(?:\^(?P<domain>[PCEO]))?"
    r"(?:_\{F(?P<drive>[1-7])\})?"
    r"(?:\|(?P<register>T|A|f|E|⟨[TAfE](?:,[TAfE])*⟩))?$"
)


@dataclass(frozen=True)
class CASState:
    noetic: int
    domain: Domain | None = None
    drive: int | None = None
    registers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.noetic not in NOETIC_NAMES:
            raise ValueError(f"Noetic must be 0..9, got {self.noetic}")
        if self.drive is not None and not 1 <= self.drive <= 7:
            raise ValueError(f"Drive must be F1..F7, got F{self.drive}")
        if not set(self.registers).issubset(VALID_REGISTERS):
            raise ValueError(f"Invalid register set: {self.registers}")
        if len(set(self.registers)) != len(self.registers):
            raise ValueError(f"Duplicate register in set: {self.registers}")

    @classmethod
    def parse(cls, literal: str) -> "CASState":
        literal = literal.strip()
        match = _STATE_RE.fullmatch(literal)
        if not match:
            raise ValueError(f"Malformed CAS state literal: {literal!r}")
        register_raw = match.group("register")
        if register_raw is None:
            registers: tuple[str, ...] = ()
        elif register_raw.startswith("⟨"):
            registers = tuple(register_raw[1:-1].split(","))
        else:
            registers = (register_raw,)
        domain_raw = match.group("domain")
        return cls(
            noetic=int(match.group("noetic")),
            domain=Domain(domain_raw) if domain_raw else None,
            drive=int(match.group("drive")) if match.group("drive") else None,
            registers=registers,
        )

    @property
    def name(self) -> str:
        return NOETIC_NAMES[self.noetic]

    def canonical(self) -> str:
        text = f"^{self.noetic}"
        if self.domain is not None:
            text += f"^{self.domain.value}"
        if self.drive is not None:
            text += f"_{{F{self.drive}}}"
        if self.registers:
            if len(self.registers) == 1:
                text += f"|{self.registers[0]}"
            else:
                text += "|⟨" + ",".join(self.registers) + "⟩"
        return text

    def involution(self) -> "CASState":
        return CASState(
            noetic=INVOLUTION.get(self.noetic, self.noetic),
            domain=self.domain,
            drive=self.drive,
            registers=self.registers,
        )

    def with_noetic(self, noetic: int) -> "CASState":
        return CASState(noetic, self.domain, self.drive, self.registers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "literal": self.canonical(),
            "noetic": self.noetic,
            "name": self.name,
            "domain": self.domain.value if self.domain else None,
            "drive": f"F{self.drive}" if self.drive else None,
            "registers": list(self.registers),
        }

    def __str__(self) -> str:
        return self.canonical()


@dataclass(frozen=True)
class Signal:
    state: CASState | None
    activation: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def clamped(self, lower: float = 0.0, upper: float = 1.0) -> "Signal":
        return Signal(
            state=self.state,
            activation=max(lower, min(upper, self.activation)),
            source=self.source,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict() if self.state else None,
            "activation": round(float(self.activation), 9),
            "source": self.source,
            "metadata": self.metadata,
        }


def fisher_rao_distance(p: Iterable[float], q: Iterable[float]) -> float:
    """Fisher-Rao geodesic distance on a probability simplex.

    The testbench uses this for optional memory/address checks. Inputs are
    normalized defensively and clipped against floating-point drift.
    """
    p_list = [max(0.0, float(x)) for x in p]
    q_list = [max(0.0, float(x)) for x in q]
    if len(p_list) != len(q_list) or not p_list:
        raise ValueError("Fisher-Rao inputs must be non-empty and equal length")
    p_sum, q_sum = sum(p_list), sum(q_list)
    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("Fisher-Rao inputs must have positive mass")
    coefficient = sum(
        math.sqrt((px / p_sum) * (qx / q_sum)) for px, qx in zip(p_list, q_list)
    )
    coefficient = max(-1.0, min(1.0, coefficient))
    return 2.0 * math.acos(coefficient)

"""Eight-cell CMB-derived register bank for the v0.1 microcore."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .cas import CASState, Signal


@dataclass
class MemoryCell:
    name: str
    state: CASState
    capacity: float
    leakage: float
    charge: float = 0.0
    ever_written: bool = False

    def decay(self, ticks: int = 1) -> None:
        for _ in range(max(0, ticks)):
            self.charge *= 1.0 - self.leakage
        if self.charge < 1e-12:
            self.charge = 0.0

    def write(self, activation: float, impression: float = 1.0) -> None:
        # Capacity shapes how quickly a cell fills; the public charge remains 0..1.
        fill_rate = min(0.5, 25.0 / max(self.capacity, 1.0))
        increment = max(0.0, activation) * max(0.0, impression) * fill_rate
        self.charge = min(1.0, self.charge + increment * (1.0 - self.charge))
        self.ever_written = True

    def refresh(self, amount: float) -> None:
        if self.ever_written:
            self.charge = min(1.0, self.charge + max(0.0, amount) * (1.0 - self.charge))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.to_dict(),
            "capacity": self.capacity,
            "leakage": self.leakage,
            "charge": round(self.charge, 9),
            "ever_written": self.ever_written,
            "lacunarity": not self.ever_written,
        }


class MemoryBank:
    """A deliberately small register file, not the full 40-cell CMB."""

    def __init__(self) -> None:
        definitions = [
            ("P3", "^3^P_{F7}|f", 200.0, 0.001),
            ("P5", "^5^P_{F7}|T", 220.0, 0.0005),
            ("C6", "^6^C_{F6}|A", 110.0, 0.02),
            ("C8", "^8^C_{F2}|A", 60.0, 0.05),
            ("E3", "^3^E_{F6}|A", 100.0, 0.025),
            ("E6", "^6^E_{F6}|A", 110.0, 0.02),
            ("O9", "^9^O_{F5}|E", 80.0, 0.04),
            ("C7", "^7^C_{F7}|A", 130.0, 0.015),
        ]
        self.cells: dict[str, MemoryCell] = {
            name: MemoryCell(name, CASState.parse(literal), capacity, leakage)
            for name, literal, capacity, leakage in definitions
        }
        # The constitution begins loaded; this is the register-file analogue of ROM/firmware.
        self.cells["P5"].charge = 0.92
        self.cells["P5"].ever_written = True

    def decay(self, ticks: int = 1) -> None:
        for cell in self.cells.values():
            cell.decay(ticks)

    def write_signals(self, signals: Iterable[Signal]) -> list[str]:
        written: list[str] = []
        for signal in signals:
            if signal.state is None or signal.state.domain is None:
                continue
            for cell in self.cells.values():
                if cell.state.noetic == signal.state.noetic and cell.state.domain == signal.state.domain:
                    cell.write(signal.activation)
                    written.append(cell.name)
                    break
        return written

    def refresh_durable(self, rhythm_activation: float) -> None:
        self.cells["P3"].refresh(0.08 * rhythm_activation)
        self.cells["P5"].refresh(0.10 * rhythm_activation)
        self.cells["C6"].refresh(0.04 * rhythm_activation)
        self.cells["E6"].refresh(0.04 * rhythm_activation)
        # C8 intentionally receives no refresh: passing associations may fade.

    def charge_for(self, name: str) -> float:
        return self.cells[name].charge

    def snapshot(self) -> dict[str, Any]:
        return {name: cell.to_dict() for name, cell in self.cells.items()}

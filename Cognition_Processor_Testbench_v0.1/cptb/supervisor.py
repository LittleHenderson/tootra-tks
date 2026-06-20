"""CCB-6-inspired Cognitive Power & Safety Supervisor (CPSS)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .cas import Domain


@dataclass
class SupervisorConfig:
    principled_voltage: float = 5.0
    vitality_nominal: float = 12.0
    brownout_floor: float = 2.5
    severe_brownout_floor: float = 1.25
    thermal_warn: float = 0.65
    thermal_critical: float = 0.80
    thermal_emergency: float = 0.95
    watchdog_timeout_ticks: int = 2
    channel_refuse_threshold: float = 0.50

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FuseState:
    blown: bool = False
    reason: str | None = None

    def trip(self, reason: str) -> None:
        if not self.blown:
            self.blown = True
            self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {"blown": self.blown, "reason": self.reason}


@dataclass(frozen=True)
class PowerUpReading:
    requested_order: tuple[str, ...]
    enabled: dict[str, bool]
    blocked_attempts: tuple[str, ...]
    pgood: dict[str, bool]

    @property
    def en_o(self) -> bool:
        return bool(self.enabled.get("O", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_order": list(self.requested_order),
            "enabled": dict(self.enabled),
            "blocked_attempts": list(self.blocked_attempts),
            "pgood": dict(self.pgood),
            "EN_O": self.en_o,
        }


@dataclass(frozen=True)
class ProtectionReading:
    vitality: float
    thermal_load: float
    derate: float
    force_rest: bool
    status_bus: tuple[str, ...]
    emergency: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "vitality": self.vitality,
            "thermal_load": self.thermal_load,
            "derate": self.derate,
            "force_rest": self.force_rest,
            "status_bus": list(self.status_bus),
            "emergency": self.emergency,
        }


@dataclass(frozen=True)
class VoteReading:
    channels: dict[str, bool | None]
    safe_vote: bool
    odd_channel: str | None
    masked_channels: tuple[str, ...]
    fail_closed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "SAFE_VOTE": self.safe_vote,
            "odd_channel": self.odd_channel,
            "masked_channels": list(self.masked_channels),
            "fail_closed": self.fail_closed,
        }


class CPSSSupervisor:
    """Power sequencing, protection, TMR voting, watchdog, and output cutoff."""

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        self.config = config or SupervisorConfig()
        self.master_fuse = FuseState()
        self._missed_watchdog_ticks = 0
        self.last_power: PowerUpReading | None = None
        self.last_protection: ProtectionReading | None = None
        self.last_vote: VoteReading | None = None

    def power_up(
        self,
        requested_order: Iterable[str] = ("P", "C", "E", "O"),
        pgood: dict[str, bool] | None = None,
    ) -> PowerUpReading:
        pgood_map = {domain.value: True for domain in Domain}
        if pgood:
            pgood_map.update({str(key): bool(value) for key, value in pgood.items()})
        enabled = {domain.value: False for domain in Domain}
        blocked: list[str] = []

        for domain in requested_order:
            domain = str(domain)
            if domain not in enabled:
                blocked.append(f"UNKNOWN:{domain}")
                continue
            prerequisites = {
                "P": (),
                "C": ("P",),
                "E": ("P", "C"),
                "O": ("P", "C", "E"),
            }[domain]
            if pgood_map.get(domain, False) and all(enabled[p] and pgood_map.get(p, False) for p in prerequisites):
                enabled[domain] = True
            else:
                blocked.append(domain)

        reading = PowerUpReading(tuple(str(x) for x in requested_order), enabled, tuple(blocked), pgood_map)
        self.last_power = reading
        return reading

    def protect(self, vitality: float, thermal_load: float) -> ProtectionReading:
        status: list[str] = []
        derate = 1.0
        force_rest = False
        emergency: str | None = None

        if vitality < self.config.brownout_floor:
            status.append("BO_ALERT")
            derate *= 0.50
            force_rest = True
            if vitality < self.config.severe_brownout_floor:
                emergency = "SEVERE_BROWNOUT"
                self.master_fuse.trip(emergency)

        if thermal_load >= self.config.thermal_warn:
            status.append("T_WARN")
            derate *= 0.75
        if thermal_load >= self.config.thermal_critical:
            status.append("T_CRIT")
            force_rest = True
        if thermal_load >= self.config.thermal_emergency:
            status.append("T_EMERG")
            emergency = "THERMAL_EMERGENCY"
            self.master_fuse.trip(emergency)

        reading = ProtectionReading(
            vitality=float(vitality),
            thermal_load=float(thermal_load),
            derate=derate,
            force_rest=force_rest,
            status_bus=tuple(status),
            emergency=emergency,
        )
        self.last_protection = reading
        return reading

    def vote(self, channels: dict[str, bool | None]) -> VoteReading:
        if len(channels) != 3:
            raise ValueError("CPSS TMR voter requires exactly three named channels")
        active = {name: value for name, value in channels.items() if value is not None}
        masked = tuple(name for name, value in channels.items() if value is None)
        allow_count = sum(value is True for value in active.values())
        refuse_count = sum(value is False for value in active.values())
        fail_closed = False

        if allow_count >= 2:
            safe_vote = True
        elif refuse_count >= 2:
            safe_vote = False
        else:
            # A tie, two failed channels, or ambiguous liveness fails closed.
            safe_vote = False
            fail_closed = True

        odd_channel: str | None = None
        if len(active) == 3 and (allow_count == 1 or refuse_count == 1):
            minority_value = allow_count == 1
            odd_channel = next(name for name, value in active.items() if value is minority_value)

        reading = VoteReading(dict(channels), safe_vote, odd_channel, masked, fail_closed)
        self.last_vote = reading
        return reading

    def watchdog_tick(self, *, kick: bool) -> None:
        if kick:
            self._missed_watchdog_ticks = 0
        else:
            self._missed_watchdog_ticks += 1
            if self._missed_watchdog_ticks >= self.config.watchdog_timeout_ticks:
                self.master_fuse.trip("WDT_TRIP")

    def route_output(self, candidate: str, refusal: str, vote: VoteReading | None = None) -> str:
        vote = vote or self.last_vote
        if self.master_fuse.blown:
            return "MASTER_CUTOFF"
        if self.last_power is None or not self.last_power.en_o:
            return "HELD_RESET"
        if self.last_protection and self.last_protection.force_rest:
            return "REST"
        if vote is None:
            return refusal
        return candidate if vote.safe_vote else refusal

    def probes(self, result: str | None = None) -> dict[str, Any]:
        return {
            "TP1": {"V_Principled": self.config.principled_voltage},
            "TP2": {"PGOOD": self.last_power.pgood if self.last_power else None},
            "TP3": {"EN_O": self.last_power.en_o if self.last_power else False},
            "TP4": self.last_protection.to_dict() if self.last_protection else None,
            "TP5": self.last_vote.to_dict() if self.last_vote else None,
            "TP6": {"result": result, "master_fuse": self.master_fuse.to_dict()},
        }

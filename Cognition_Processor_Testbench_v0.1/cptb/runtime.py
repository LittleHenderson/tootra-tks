"""Clocked reference runtime for the Cognition Processor Testbench."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .cas import Signal
from .ccdl import CCDLParser, CircuitSpec, Diagnostic
from .components import CJTDevice, ComponentReading
from .memory import MemoryBank
from .scenarios import Scenario
from .seam import SemanticContract, SemanticSeamBench, SeamMeasurement
from .supervisor import CPSSSupervisor, SupervisorConfig, PowerUpReading, ProtectionReading, VoteReading


@dataclass(frozen=True)
class TraceEvent:
    tick: int
    phase: str
    event: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tick": self.tick, "phase": self.phase, "event": self.event, "data": self.data}


@dataclass(frozen=True)
class RunResult:
    run_id: str
    scenario: dict[str, Any]
    circuit_name: str
    circuit_digest: str
    output: str
    route: str
    expected_route: str
    passed: bool
    power: PowerUpReading
    protection: ProtectionReading
    vote: VoteReading
    master_fuse: dict[str, Any]
    probes: dict[str, Any]
    signals: dict[str, Any]
    component_readings: dict[str, Any]
    seam: tuple[SeamMeasurement, ...]
    memory: dict[str, Any]
    trace: tuple[TraceEvent, ...]
    diagnostics: tuple[Diagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "circuit_name": self.circuit_name,
            "circuit_digest": self.circuit_digest,
            "output": self.output,
            "route": self.route,
            "expected_route": self.expected_route,
            "passed": self.passed,
            "power": self.power.to_dict(),
            "protection": self.protection.to_dict(),
            "vote": self.vote.to_dict(),
            "master_fuse": self.master_fuse,
            "probes": self.probes,
            "signals": self.signals,
            "component_readings": self.component_readings,
            "seam": [item.to_dict() for item in self.seam],
            "memory": self.memory,
            "trace": [event.to_dict() for event in self.trace],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class CognitionProcessor:
    """Small deterministic microcore under a CPSS supervisory envelope."""

    def __init__(
        self,
        circuit: CircuitSpec,
        contract: SemanticContract,
        *,
        supervisor_config: SupervisorConfig | None = None,
        memory: MemoryBank | None = None,
    ) -> None:
        self.circuit = circuit
        self.contract = contract
        self.supervisor = CPSSSupervisor(supervisor_config)
        self.memory = memory or MemoryBank()
        errors = [d for d in circuit.validate() if d.severity == "error"]
        if errors:
            messages = "; ".join(f"{d.code} {d.message}" for d in errors)
            raise ValueError(f"Circuit failed DRC: {messages}")

    @classmethod
    def from_files(
        cls,
        ccdl_path: str | Path,
        contract_path: str | Path,
        *,
        supervisor_config: SupervisorConfig | None = None,
    ) -> "CognitionProcessor":
        return cls(
            CCDLParser.parse_file(ccdl_path),
            SemanticContract.load(contract_path),
            supervisor_config=supervisor_config,
        )

    def run(
        self,
        scenario: Scenario,
        *,
        depth: int = 3,
        faults: dict[str, str] | None = None,
        startup_order: tuple[str, ...] = ("P", "C", "E", "O"),
        pgood: dict[str, bool] | None = None,
        watchdog_missed_ticks: int = 0,
        reflection_probes: dict[str, Any] | None = None,
    ) -> RunResult:
        faults = dict(faults or {})
        reflection_probes = dict(reflection_probes or {})
        trace: list[TraceEvent] = []
        tick = 0

        def record(phase: str, event: str, data: dict[str, Any]) -> None:
            nonlocal tick
            trace.append(TraceEvent(tick, phase, event, _jsonable(data)))
            tick += 1

        record("FETCH", "input_loaded", scenario.to_dict())

        power = self.supervisor.power_up(startup_order, pgood)
        record("POWER", "domain_sequence", power.to_dict())

        protection = self.supervisor.protect(scenario.vitality, scenario.thermal_load)
        record("PROTECTION", "resource_check", protection.to_dict())

        signals: dict[str, Signal] = {
            name: Signal(None, activation, f"input:{name}")
            for name, activation in scenario.inputs().items()
        }
        signals["constitution"] = Signal(
            self.circuit.nodes["constitution"],
            self.memory.charge_for("P5"),
            "memory:P5",
        )
        signals["working_model"] = Signal(self.circuit.nodes["working_model"], 0.90, "register:working_model")
        signals["experience_e"] = Signal(self.circuit.nodes["experience_e"], 0.85, "register:experience_e")
        record(
            "BIAS",
            "bias_loaded",
            {name: signals[name].to_dict() for name in ("constitution", "working_model", "experience_e")},
        )

        readings: dict[str, ComponentReading] = {}
        for identifier in self.circuit.topological_order():
            spec = self.circuit.cjts[identifier]
            cause = signals.get(spec.cause)
            if cause is None:
                cause = Signal(self.circuit.ref_state(spec.cause), 0.0, f"unseeded:{spec.cause}")
            bias = signals.get(spec.bias)
            bias_activation = bias.activation if bias is not None else 0.0
            device = CJTDevice(spec, self.circuit.ref_state(spec.effect))
            reading = device.transfer(
                cause,
                bias_activation,
                derate=protection.derate,
                fault=faults.get(identifier),
            )
            readings[identifier] = reading
            signals[spec.effect] = reading.effect
            record("EXECUTE", identifier, reading.to_dict())

        threshold = self.supervisor.config.channel_refuse_threshold
        channel_map = {
            "A": "ch_a_guard",
            "B": "ch_b_guard",
            "C": "ch_c_guard",
        }
        channel_verdicts: dict[str, bool | None] = {}
        for channel, component in channel_map.items():
            reading = readings[component]
            if reading.fault == "silent":
                channel_verdicts[channel] = None
            elif reading.region == "breakdown":
                channel_verdicts[channel] = False
            else:
                channel_verdicts[channel] = reading.effect.activation < threshold
        vote = self.supervisor.vote(channel_verdicts)
        record("SAFETY", "tmr_vote", vote.to_dict())

        if watchdog_missed_ticks > 0:
            for _ in range(watchdog_missed_ticks):
                self.supervisor.watchdog_tick(kick=False)
                record("WATCHDOG", "missed_kick", self.supervisor.master_fuse.to_dict())
        else:
            self.supervisor.watchdog_tick(kick=True)
            record("WATCHDOG", "kick", self.supervisor.master_fuse.to_dict())

        seam_bench = SemanticSeamBench(self.circuit, self.contract)
        seam = tuple(seam_bench.measure(scenario.behavioral_seed, range(1, depth + 1)))
        record("SEAM", "dual_interpreter", {"measurements": [item.to_dict() for item in seam]})

        self.memory.decay(1)
        memory_signals = [
            signals[name]
            for name in ("expanded", "decomposed", "evaluated", "candidate", "a_guard", "memory_rehearsal")
            if name in signals
        ]
        written = self.memory.write_signals(memory_signals)
        rhythm = signals.get("memory_rehearsal", Signal(None, 0.0, "none")).activation
        self.memory.refresh_durable(rhythm)
        record("MEMORY", "register_update", {"written_cells": written, "snapshot": self.memory.snapshot()})

        output = self.supervisor.route_output(scenario.candidate, scenario.refusal, vote)
        route = _route_from_output(output, scenario)
        passed = route == scenario.expected_route
        record("COMMIT", "output_routed", {"output": output, "route": route, "expected": scenario.expected_route, "passed": passed})

        final_seam = seam[-1]
        probes = self.supervisor.probes(output)
        probes.update(
            {
                "TP7": {"structural_result": round(final_seam.structural_value, 9)},
                "TP8": {"behavioral_result": round(final_seam.behavioral_value, 9)},
                "TP9": {"seam_divergence_delta": round(final_seam.delta, 9)},
                "TP10": {"proposed_patch": reflection_probes.get("proposed_patch")},
                "TP11": {"verification": reflection_probes.get("verification")},
                "TP12": {"commit_event": reflection_probes.get("commit_event")},
            }
        )

        diagnostics = tuple(self.circuit.validate())
        circuit_digest = hashlib.sha256(self.circuit.to_ccdl().encode("utf-8")).hexdigest()
        payload = {
            "scenario": scenario.to_dict(),
            "circuit_digest": circuit_digest,
            "output": output,
            "route": route,
            "power": power.to_dict(),
            "protection": protection.to_dict(),
            "vote": vote.to_dict(),
            "fuse": self.supervisor.master_fuse.to_dict(),
            "probes": probes,
            "signals": {name: signal.to_dict() for name, signal in sorted(signals.items())},
            "readings": {name: reading.to_dict() for name, reading in readings.items()},
            "seam": [item.to_dict() for item in seam],
            "memory": self.memory.snapshot(),
            "trace": [event.to_dict() for event in trace],
        }
        run_id = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

        return RunResult(
            run_id=run_id,
            scenario=scenario.to_dict(),
            circuit_name=self.circuit.name,
            circuit_digest=circuit_digest,
            output=output,
            route=route,
            expected_route=scenario.expected_route,
            passed=passed,
            power=power,
            protection=protection,
            vote=vote,
            master_fuse=self.supervisor.master_fuse.to_dict(),
            probes=probes,
            signals={name: signal.to_dict() for name, signal in sorted(signals.items())},
            component_readings={name: reading.to_dict() for name, reading in readings.items()},
            seam=seam,
            memory=self.memory.snapshot(),
            trace=tuple(trace),
            diagnostics=diagnostics,
        )


def _route_from_output(output: str, scenario: Scenario) -> str:
    if output == scenario.candidate:
        return "ALLOW"
    if output == scenario.refusal:
        return "REFUSE"
    return output


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value

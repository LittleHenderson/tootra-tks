"""Dual-interpreter seam layer.

The structural interpreter follows the declared CCDL graph.  The behavioral
interpreter is a deterministic fixture contract representing the task-level
behavior expected from that graph.  It is intentionally not an LLM benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .ccdl import CircuitSpec


@dataclass(frozen=True)
class ContractStage:
    component: str
    expected_noetic: int
    expected_beta: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class SemanticContract:
    name: str
    description: str
    pipeline: tuple[ContractStage, ...]
    categorical_penalty_per_depth: float
    acceptance_max_delta: float
    required_depth: int

    @classmethod
    def load(cls, path: str | Path) -> "SemanticContract":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            description=data["description"],
            pipeline=tuple(ContractStage(**item) for item in data["pipeline"]),
            categorical_penalty_per_depth=float(data["categorical_penalty_per_depth"]),
            acceptance_max_delta=float(data["acceptance"]["max_delta"]),
            required_depth=int(data["acceptance"]["required_depth"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "pipeline": [stage.to_dict() for stage in self.pipeline],
            "categorical_penalty_per_depth": self.categorical_penalty_per_depth,
            "acceptance": {
                "max_delta": self.acceptance_max_delta,
                "required_depth": self.required_depth,
            },
        }


@dataclass(frozen=True)
class StageObservation:
    depth: int
    component: str
    input_value: float
    beta: float
    noetic: int
    output_value: float
    expected_beta: float
    expected_noetic: int
    beta_residual: float
    type_mismatch: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "component": self.component,
            "input": round(self.input_value, 9),
            "actual_beta": self.beta,
            "actual_noetic": self.noetic,
            "output": round(self.output_value, 9),
            "expected_beta": self.expected_beta,
            "expected_noetic": self.expected_noetic,
            "beta_residual": round(self.beta_residual, 9),
            "type_mismatch": self.type_mismatch,
        }


@dataclass(frozen=True)
class SeamMeasurement:
    depth: int
    structural_value: float
    behavioral_value: float
    numeric_delta: float
    categorical_penalty: float
    delta: float
    observations: tuple[StageObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "structural_value": round(self.structural_value, 9),
            "behavioral_value": round(self.behavioral_value, 9),
            "numeric_delta": round(self.numeric_delta, 9),
            "categorical_penalty": round(self.categorical_penalty, 9),
            "delta": round(self.delta, 9),
            "observations": [obs.to_dict() for obs in self.observations],
        }


class SemanticSeamBench:
    def __init__(self, circuit: CircuitSpec, contract: SemanticContract):
        self.circuit = circuit
        self.contract = contract

    @staticmethod
    def _operator(noetic: int, value: float) -> float:
        value = max(0.0, min(1.0, value))
        if noetic == 6:  # Male / analytical decomposition
            return value
        if noetic == 3:  # Negative / aversion introduces a categorical bias
            return min(1.0, 0.15 + 0.85 * value)
        return value

    def measure(self, seed: float, depths: range | list[int] | tuple[int, ...]) -> list[SeamMeasurement]:
        requested = sorted(set(int(depth) for depth in depths))
        if not requested or requested[0] < 1:
            raise ValueError("Depths must contain positive integers")
        max_depth = max(requested)
        structural = float(seed)
        behavioral = float(seed)
        mismatch_count = 0
        all_observations: list[StageObservation] = []
        measurements: list[SeamMeasurement] = []

        for depth in range(1, max_depth + 1):
            depth_observations: list[StageObservation] = []
            for stage in self.contract.pipeline:
                cjt = self.circuit.cjts[stage.component]
                state = self.circuit.ref_state(cjt.effect)
                if state is None:
                    raise ValueError(f"Contract component {stage.component!r} has untyped effect {cjt.effect!r}")

                before = structural
                structural_raw = min(1.0, max(0.0, structural * cjt.beta))
                structural = self._operator(state.noetic, structural_raw)

                behavioral_raw = min(1.0, max(0.0, behavioral * stage.expected_beta))
                behavioral = self._operator(stage.expected_noetic, behavioral_raw)

                mismatch = state.noetic != stage.expected_noetic
                if mismatch:
                    mismatch_count += 1
                observation = StageObservation(
                    depth=depth,
                    component=stage.component,
                    input_value=before,
                    beta=cjt.beta,
                    noetic=state.noetic,
                    output_value=structural,
                    expected_beta=stage.expected_beta,
                    expected_noetic=stage.expected_noetic,
                    beta_residual=abs(cjt.beta - stage.expected_beta),
                    type_mismatch=mismatch,
                )
                depth_observations.append(observation)
                all_observations.append(observation)

            if depth in requested:
                numeric_delta = abs(structural - behavioral)
                categorical_penalty = mismatch_count * self.contract.categorical_penalty_per_depth
                measurements.append(
                    SeamMeasurement(
                        depth=depth,
                        structural_value=structural,
                        behavioral_value=behavioral,
                        numeric_delta=numeric_delta,
                        categorical_penalty=categorical_penalty,
                        delta=numeric_delta + categorical_penalty,
                        observations=tuple(depth_observations),
                    )
                )
        return measurements

    def contract_residuals(self) -> list[dict[str, Any]]:
        residuals: list[dict[str, Any]] = []
        for stage in self.contract.pipeline:
            cjt = self.circuit.cjts[stage.component]
            state = self.circuit.ref_state(cjt.effect)
            if state is None:
                continue
            residuals.append(
                {
                    "component": stage.component,
                    "effect_node": cjt.effect,
                    "actual_noetic": state.noetic,
                    "expected_noetic": stage.expected_noetic,
                    "type_mismatch": state.noetic != stage.expected_noetic,
                    "actual_beta": cjt.beta,
                    "expected_beta": stage.expected_beta,
                    "beta_residual": abs(cjt.beta - stage.expected_beta),
                }
            )
        return residuals

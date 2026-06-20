"""Executable component semantics for the testbench's clean layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cas import CASState, Signal
from .ccdl import CJTSpec


@dataclass(frozen=True)
class ComponentReading:
    component: str
    cause: Signal
    bias_activation: float
    effect: Signal
    region: str
    delivered_gain: float
    stress: float
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    fault: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "cause": self.cause.to_dict(),
            "bias_activation": round(self.bias_activation, 9),
            "effect": self.effect.to_dict(),
            "region": self.region,
            "delivered_gain": round(self.delivered_gain, 9),
            "stress": round(self.stress, 9),
            "diagnostics": list(self.diagnostics),
            "fault": self.fault,
        }


class CJTDevice:
    """Reference transfer function for one CJT.

    v0.1 makes one explicit simplifying assumption: while the bias is above its
    cutoff floor, it gates the transition but does not continuously rescale β.
    This preserves the book's active-region statement Δeffect ≈ β·Δcause at
    constant bias.  gm/rπ dynamics are outside this prototype.
    """

    def __init__(self, spec: CJTSpec, effect_state: CASState | None):
        self.spec = spec
        self.effect_state = effect_state

    def transfer(
        self,
        cause: Signal,
        bias_activation: float,
        *,
        derate: float = 1.0,
        fault: str | None = None,
    ) -> ComponentReading:
        bias_activation = max(0.0, float(bias_activation))
        derate = max(0.0, float(derate))
        diagnostics: list[str] = []

        if fault == "silent":
            effect = Signal(self.effect_state, 0.0, self.spec.identifier, {"fault": fault})
            return ComponentReading(
                component=self.spec.identifier,
                cause=cause,
                bias_activation=bias_activation,
                effect=effect,
                region="cutoff",
                delivered_gain=0.0,
                stress=0.0,
                diagnostics=("E-CJT-003 CjtCutoff", "FAULT-SILENT"),
                fault=fault,
            )

        if bias_activation < self.spec.cutoff and fault != "short":
            effect = Signal(self.effect_state, 0.0, self.spec.identifier)
            return ComponentReading(
                component=self.spec.identifier,
                cause=cause,
                bias_activation=bias_activation,
                effect=effect,
                region="cutoff",
                delivered_gain=0.0,
                stress=0.0,
                diagnostics=("E-CJT-003 CjtCutoff",),
                fault=fault,
            )

        effective_beta = self.spec.beta * derate
        if fault == "short":
            effective_beta = max(1.0, effective_beta) * 1.25
            diagnostics.append("W-CJT-006 CjtShortCircuit")
        stress = max(0.0, cause.activation) * effective_beta
        breakdown_threshold = (
            self.spec.breakdown
            if self.spec.breakdown is not None
            else max(1.05, self.spec.sat + 0.25)
        )

        effect_state = self.effect_state
        if fault == "inverted" and effect_state is not None:
            effect_state = effect_state.involution()
            diagnostics.append("FAULT-INVERTED")

        if fault == "stuck_saturation":
            activation = self.spec.sat
            region = "saturation"
            diagnostics.append("W-CJT-004 CjtSaturation")
        elif stress >= breakdown_threshold:
            activation = min(1.0, max(self.spec.sat, stress - breakdown_threshold + self.spec.sat))
            region = "breakdown"
            if effect_state is not None:
                effect_state = effect_state.involution()
            diagnostics.append("CJT-BREAKDOWN INVOLUTION-FLIP")
        elif stress >= self.spec.sat:
            activation = self.spec.sat
            region = "saturation"
            diagnostics.append("W-CJT-004 CjtSaturation")
        else:
            activation = stress
            region = "active"

        delivered_gain = activation / cause.activation if cause.activation > 0 else 0.0
        effect = Signal(
            state=effect_state,
            activation=activation,
            source=self.spec.identifier,
            metadata={"declared_region": self.spec.region},
        )
        return ComponentReading(
            component=self.spec.identifier,
            cause=cause,
            bias_activation=bias_activation,
            effect=effect,
            region=region,
            delivered_gain=delivered_gain,
            stress=stress,
            diagnostics=tuple(diagnostics),
            fault=fault,
        )

"""A deliberately small, executable CCDL subset.

Supported declarations:
  circuit, input, output, node, cjt, feedback, probe

The source book specifies CCDL as a declarative validation language rather than
an executable runtime.  This module parses the subset needed by the testbench,
performs transparent DRC checks, and supplies a graph to the reference runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from pathlib import Path
import re
from typing import Any, Iterable

from .cas import CASState


@dataclass(frozen=True)
class Diagnostic:
    code: str
    name: str
    severity: str
    message: str
    subject: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "subject": self.subject,
        }


@dataclass
class CJTSpec:
    identifier: str
    cause: str
    bias: str
    effect: str
    beta: float = 1.0
    sat: float = 0.95
    cutoff: float = 0.1
    region: str = "active"
    gm: float | None = None
    r_pi: float | None = None
    breakdown: float | None = None

    def parameters(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "β": self.beta,
            "sat": self.sat,
            "cutoff": self.cutoff,
            "region": self.region,
        }
        if self.gm is not None:
            values["gm"] = self.gm
        if self.r_pi is not None:
            values["rπ"] = self.r_pi
        if self.breakdown is not None:
            values["breakdown"] = self.breakdown
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "cause": self.cause,
            "bias": self.bias,
            "effect": self.effect,
            "parameters": self.parameters(),
        }


@dataclass
class FeedbackSpec:
    source: str
    target: str
    beta: float = 0.2

    @property
    def identifier(self) -> str:
        return f"{self.source}->{self.target}"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "β": self.beta}


@dataclass
class ProbeSpec:
    identifier: str
    ref: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.identifier, "ref": self.ref}


@dataclass
class CircuitSpec:
    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    nodes: dict[str, CASState] = field(default_factory=dict)
    cjts: dict[str, CJTSpec] = field(default_factory=dict)
    feedback: list[FeedbackSpec] = field(default_factory=list)
    probes: dict[str, ProbeSpec] = field(default_factory=dict)
    parse_diagnostics: list[Diagnostic] = field(default_factory=list)

    def clone(self) -> "CircuitSpec":
        return copy.deepcopy(self)

    @property
    def refs(self) -> set[str]:
        return set(self.inputs) | set(self.outputs) | set(self.nodes)

    def ref_state(self, ref: str) -> CASState | None:
        if ref in self.nodes:
            return self.nodes[ref]
        if ref.startswith("^"):
            return CASState.parse(ref)
        return None

    def validate(self, cascade_warning_depth: int = 6) -> list[Diagnostic]:
        diagnostics = list(self.parse_diagnostics)

        if not self.inputs:
            diagnostics.append(Diagnostic("E-CCDL-001", "MissingInput", "error", "Circuit has no input terminal", self.name))
        if not self.outputs:
            diagnostics.append(Diagnostic("E-CCDL-002", "MissingOutput", "error", "Circuit has no output terminal", self.name))
        if not self.cjts:
            diagnostics.append(Diagnostic("E-CCDL-003", "MissingComponent", "error", "Circuit has no CJT component", self.name))

        known_refs = self.refs
        for cjt in self.cjts.values():
            for role, ref in (("cause", cjt.cause), ("bias", cjt.bias), ("effect", cjt.effect)):
                if ref not in known_refs and not ref.startswith("^"):
                    diagnostics.append(
                        Diagnostic("E-CCDL-004", "UnknownReference", "error", f"Unknown {role} reference {ref!r}", cjt.identifier)
                    )

            if not cjt.bias:
                diagnostics.append(Diagnostic("E-CJT-001", "MissingBias", "error", "CJT has no bias terminal", cjt.identifier))

            if not math.isfinite(cjt.beta) or cjt.beta < 0:
                diagnostics.append(Diagnostic("E-CJT-005", "GainDivergence", "error", "β must be finite and non-negative", cjt.identifier))
            if not 0 < cjt.sat <= 1:
                diagnostics.append(Diagnostic("E-CCDL-005", "InvalidSaturation", "error", "sat must be in (0, 1]", cjt.identifier))
            if not 0 <= cjt.cutoff < 1:
                diagnostics.append(Diagnostic("E-CCDL-006", "InvalidCutoff", "error", "cutoff must be in [0, 1)", cjt.identifier))
            if cjt.cutoff >= cjt.sat:
                diagnostics.append(Diagnostic("E-CCDL-007", "EmptyActiveRegion", "error", "cutoff must be below sat", cjt.identifier))
            if cjt.region not in {"cutoff", "active", "saturation", "breakdown"}:
                diagnostics.append(Diagnostic("E-CCDL-008", "InvalidRegion", "error", f"Unknown region {cjt.region!r}", cjt.identifier))

            cause_state = self.ref_state(cjt.cause)
            bias_state = self.ref_state(cjt.bias)
            effect_state = self.ref_state(cjt.effect)
            if bias_state is not None and bias_state.noetic != 5:
                diagnostics.append(
                    Diagnostic(
                        "W-CJT-009",
                        "NonFemaleBias",
                        "warning",
                        "Bias is legal in this subset but is not a ^5 Female substrate",
                        cjt.identifier,
                    )
                )
            if cause_state and cause_state.domain and bias_state and bias_state.domain:
                if bias_state.domain.rank < cause_state.domain.rank:
                    diagnostics.append(
                        Diagnostic(
                            "E-CJT-008",
                            "CjtCrossDomainBias",
                            "error",
                            f"Bias domain {bias_state.domain.value} is below cause domain {cause_state.domain.value}",
                            cjt.identifier,
                        )
                    )
            if cause_state and effect_state and cause_state == effect_state:
                diagnostics.append(
                    Diagnostic("W-CJT-006", "CjtShortCircuit", "warning", "Cause and effect collapse to the same typed state", cjt.identifier)
                )

        for fb in self.feedback:
            if fb.source not in known_refs or fb.target not in known_refs:
                diagnostics.append(Diagnostic("E-CCDL-004", "UnknownReference", "error", "Feedback references unknown node", fb.identifier))
            if not math.isfinite(fb.beta) or fb.beta < 0 or fb.beta >= 1:
                diagnostics.append(
                    Diagnostic("E-CJT-005", "GainDivergence", "error", "Feedback round-trip β must be finite and below 1", fb.identifier)
                )
            source_state, target_state = self.ref_state(fb.source), self.ref_state(fb.target)
            if source_state and target_state and source_state.domain and target_state.domain:
                is_upward = source_state.domain.rank < target_state.domain.rank
                if is_upward and fb.beta >= 1:
                    diagnostics.append(
                        Diagnostic("E-CJT-008", "CjtCrossDomainBias", "error", "Upward feedback must be explicit and low gain", fb.identifier)
                    )

        try:
            order = self.topological_order()
            depth = self.longest_cascade_depth(order)
            if depth > cascade_warning_depth:
                diagnostics.append(
                    Diagnostic(
                        "W-CJT-007",
                        "CjtCascadeDepth",
                        "warning",
                        f"Longest CJT cascade is {depth}, above the review threshold {cascade_warning_depth}",
                        self.name,
                    )
                )
        except ValueError as exc:
            diagnostics.append(Diagnostic("E-RUN-001", "CombinationalCycle", "error", str(exc), self.name))

        return diagnostics

    def hard_errors(self) -> list[Diagnostic]:
        return [d for d in self.validate() if d.severity == "error"]

    def topological_order(self) -> list[str]:
        """Return CJT identifiers in feed-forward order; declared feedback is ignored."""
        writers: dict[str, str] = {}
        for cjt in self.cjts.values():
            if cjt.effect in writers:
                raise ValueError(f"Multiple CJTs write node {cjt.effect!r}")
            writers[cjt.effect] = cjt.identifier

        incoming: dict[str, set[str]] = {identifier: set() for identifier in self.cjts}
        outgoing: dict[str, set[str]] = {identifier: set() for identifier in self.cjts}
        for cjt in self.cjts.values():
            parent = writers.get(cjt.cause)
            if parent and parent != cjt.identifier:
                incoming[cjt.identifier].add(parent)
                outgoing[parent].add(cjt.identifier)

        ready = [identifier for identifier, parents in incoming.items() if not parents]
        ready.sort(key=lambda x: list(self.cjts).index(x))
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for child in sorted(outgoing[current], key=lambda x: list(self.cjts).index(x)):
                incoming[child].discard(current)
                if not incoming[child] and child not in ready and child not in order:
                    ready.append(child)
        if len(order) != len(self.cjts):
            cyclic = sorted(set(self.cjts) - set(order))
            raise ValueError(f"Undeclared combinational cycle among CJTs: {cyclic}")
        return order

    def longest_cascade_depth(self, order: Iterable[str] | None = None) -> int:
        order_list = list(order if order is not None else self.topological_order())
        writers = {cjt.effect: cjt.identifier for cjt in self.cjts.values()}
        depth: dict[str, int] = {}
        for identifier in order_list:
            cjt = self.cjts[identifier]
            parent = writers.get(cjt.cause)
            depth[identifier] = 1 + (depth.get(parent, 0) if parent else 0)
        return max(depth.values(), default=0)

    def set_cjt_param(self, identifier: str, key: str, value: Any) -> None:
        cjt = self.cjts[identifier]
        normalized = {"β": "beta", "beta": "beta", "rπ": "r_pi", "r_pi": "r_pi"}.get(key, key)
        if not hasattr(cjt, normalized):
            raise KeyError(f"Unknown CJT parameter {key!r}")
        setattr(cjt, normalized, value)

    def set_node_state(self, identifier: str, state: CASState | str) -> None:
        if identifier not in self.nodes:
            raise KeyError(f"Unknown node {identifier!r}")
        self.nodes[identifier] = CASState.parse(state) if isinstance(state, str) else state

    def set_feedback_beta(self, source: str, target: str, value: float) -> None:
        for fb in self.feedback:
            if fb.source == source and fb.target == target:
                fb.beta = float(value)
                return
        raise KeyError(f"Unknown feedback {source}->{target}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "nodes": {name: state.to_dict() for name, state in self.nodes.items()},
            "cjts": [cjt.to_dict() for cjt in self.cjts.values()],
            "feedback": [fb.to_dict() for fb in self.feedback],
            "probes": [probe.to_dict() for probe in self.probes.values()],
            "diagnostics": [d.to_dict() for d in self.validate()],
        }

    def to_ccdl(self) -> str:
        lines = [f"circuit {self.name} {{"]
        if self.inputs:
            lines.append("  input: " + ", ".join(self.inputs) + ";")
        if self.outputs:
            lines.append("  output: " + ", ".join(self.outputs) + ";")
        lines.append("")
        for name, state in self.nodes.items():
            lines.append(f"  node {name}: {state.canonical()};")
        lines.append("")
        for cjt in self.cjts.values():
            params = (
                f"β={_format_number(cjt.beta)}, sat={_format_number(cjt.sat)}, "
                f"cutoff={_format_number(cjt.cutoff)}, region={cjt.region}"
            )
            if cjt.breakdown is not None:
                params += f", breakdown={_format_number(cjt.breakdown)}"
            lines.append(
                f"  cjt {cjt.identifier}: CJT< {cjt.cause} | {cjt.bias} => {cjt.effect} > [{params}];"
            )
        if self.feedback:
            lines.append("")
            for fb in self.feedback:
                lines.append(f"  feedback {fb.source} -> {fb.target} [β={_format_number(fb.beta)}];")
        if self.probes:
            lines.append("")
            for probe in self.probes.values():
                lines.append(f"  probe {probe.identifier}: {probe.ref};")
        lines.append("}")
        return "\n".join(lines) + "\n"


class CCDLParser:
    _circuit_re = re.compile(r"\bcircuit\s+([A-Za-z][A-Za-z0-9_]*)\s*\{(.*)\}\s*$", re.S)
    _identifier_re = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    @classmethod
    def parse_file(cls, path: str | Path) -> CircuitSpec:
        return cls.parse(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def parse(cls, source: str) -> CircuitSpec:
        stripped = re.sub(r"//[^\n]*", "", source)
        match = cls._circuit_re.search(stripped.strip())
        if not match:
            return CircuitSpec(
                name="invalid",
                parse_diagnostics=[Diagnostic("E-CCDL-PARSE", "ParseError", "error", "Expected circuit <name> { ... }")],
            )
        circuit = CircuitSpec(name=match.group(1))
        body = match.group(2)
        statements = [piece.strip() for piece in body.split(";") if piece.strip()]
        for statement in statements:
            try:
                cls._parse_statement(circuit, statement)
            except (ValueError, KeyError) as exc:
                circuit.parse_diagnostics.append(
                    Diagnostic("E-CCDL-PARSE", "ParseError", "error", str(exc), statement[:80])
                )
        return circuit

    @classmethod
    def _parse_statement(cls, circuit: CircuitSpec, statement: str) -> None:
        if statement.startswith("input:") or statement.startswith("output:"):
            kind, values = statement.split(":", 1)
            identifiers = [item.strip() for item in values.split(",") if item.strip()]
            for identifier in identifiers:
                cls._require_identifier(identifier)
            target = circuit.inputs if kind.strip() == "input" else circuit.outputs
            target.extend(identifiers)
            return

        node_match = re.fullmatch(r"node\s+([A-Za-z][A-Za-z0-9_]*)\s*:\s*(\^\S+)", statement, re.S)
        if node_match:
            identifier, literal = node_match.groups()
            if identifier in circuit.nodes:
                raise ValueError(f"Duplicate node {identifier!r}")
            circuit.nodes[identifier] = CASState.parse(literal.strip())
            return

        cjt_match = re.fullmatch(
            r"cjt\s+([A-Za-z][A-Za-z0-9_]*)\s*:\s*CJT<\s*(.*?)\s+\|\s+(.*?)\s+=>\s+(.*?)\s*>\s*(?:\[(.*)\])?",
            statement,
            re.S,
        )
        if cjt_match:
            identifier, cause, bias, effect, param_text = cjt_match.groups()
            if identifier in circuit.cjts:
                raise ValueError(f"Duplicate CJT {identifier!r}")
            params = _parse_params(param_text or "")
            circuit.cjts[identifier] = CJTSpec(
                identifier=identifier,
                cause=cause.strip(),
                bias=bias.strip(),
                effect=effect.strip(),
                beta=float(params.get("β", params.get("beta", 1.0))),
                sat=float(params.get("sat", 0.95)),
                cutoff=float(params.get("cutoff", 0.1)),
                region=str(params.get("region", "active")),
                gm=_optional_float(params.get("gm")),
                r_pi=_optional_float(params.get("rπ", params.get("r_pi"))),
                breakdown=_optional_float(params.get("breakdown")),
            )
            return

        feedback_match = re.fullmatch(
            r"feedback\s+(\S+)\s*->\s*(\S+)\s*(?:\[(.*)\])?", statement, re.S
        )
        if feedback_match:
            source, target, param_text = feedback_match.groups()
            params = _parse_params(param_text or "")
            circuit.feedback.append(FeedbackSpec(source.strip(), target.strip(), float(params.get("β", params.get("beta", 0.2)))))
            return

        probe_match = re.fullmatch(r"probe\s+([A-Za-z][A-Za-z0-9_]*)\s*:\s*(\S+)", statement, re.S)
        if probe_match:
            identifier, ref = probe_match.groups()
            circuit.probes[identifier] = ProbeSpec(identifier, ref.strip())
            return

        raise ValueError(f"Unsupported or malformed statement: {statement!r}")

    @classmethod
    def _require_identifier(cls, identifier: str) -> None:
        if not cls._identifier_re.fullmatch(identifier):
            raise ValueError(f"Malformed identifier: {identifier!r}")


def _parse_params(text: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if not text.strip():
        return params
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Malformed parameter {part!r}")
        key, raw = [piece.strip() for piece in part.split("=", 1)]
        if raw in {"cutoff", "active", "saturation", "breakdown"}:
            value: Any = raw
        else:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        params[key] = value
    return params


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _format_number(value: float) -> str:
    text = f"{float(value):.9g}"
    return text

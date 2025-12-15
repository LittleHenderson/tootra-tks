from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

WORLDS = ["A", "B", "C", "D"]
WORLD_OPP = {"A": "D", "D": "A", "B": "C", "C": "B"}

NOETIC_OPPOSITE: Dict[int, int] = {
    0: 0,   # N1/Mind - self (observer)
    1: 2,   # N2/Positive ↔ N3/Negative
    2: 1,   # N3/Negative ↔ N2/Positive
    3: 3,   # N4/Vibration - self (fundamental)
    4: 5,   # N5/Female ↔ N6/Male
    5: 4,   # N6/Male ↔ N5/Female
    6: 6,   # N7/Rhythm - self (fundamental)
    7: 8,   # N8/Cause ↔ N9/Effect
    8: 7,   # N9/Effect ↔ N8/Cause
    9: 9,   # N10/Idea - self (synthesis)
}
NOETIC_DUAL: Dict[int, int] = {
    0: 0, 1: 4, 4: 1, 2: 8, 3: 9, 8: 2, 9: 3, 5: 7, 7: 5, 6: 4,
}
NOETIC_COUNTERPOLE = NOETIC_OPPOSITE

# Foundations: Unity, Wisdom, Life, Companionship, Power, Material, Lust
FOUNDATION_OPP = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}


class InversionMode(str, enum.Enum):
    Opposite = "Opposite"
    Dual = "Dual"
    CounterPole = "CounterPole"
    Mirror = "Mirror"
    ReverseCausal = "ReverseCausal"
    ParallelAnalogue = "ParallelAnalogue"
    NoeticComplement = "NoeticComplement"
    AcquisitionPolarity = "AcquisitionPolarity"
    FoundationFlip = "FoundationFlip"
    SubFoundationReversal = "SubFoundationReversal"
    DomainPermutation = "DomainPermutation"
    TemporalInversion = "TemporalInversion"
    ScalarInversion = "ScalarInversion"
    StructuralPermutation = "StructuralPermutation"
    ContextFrame = "ContextFrame"
    Motivational = "Motivational"
    Polarity = "Polarity"
    CausalDensity = "CausalDensity"
    Attention = "Attention"
    Value = "Value"
    DesireInhibition = "DesireInhibition"
    Expectation = "Expectation"
    Attractor = "Attractor"
    Stability = "Stability"
    Entropy = "Entropy"
    Constraint = "Constraint"
    Boundary = "Boundary"
    SemanticParity = "SemanticParity"
    AgentRole = "AgentRole"


@dataclass
class TargetProfile:
    enable: bool = False
    from_world: Optional[str] = None
    to_world: Optional[str] = None
    from_foundation: Optional[int] = None
    to_foundation: Optional[int] = None


@dataclass
class DialConfig:
    mode: InversionMode = InversionMode.Opposite
    axes: Dict[str, bool] = field(default_factory=dict)
    intensity: str = "soft"
    scope: str = "equation"
    direction: str = "bidirectional"
    target_profile: TargetProfile = field(default_factory=TargetProfile)

    def __post_init__(self):
        if not self.axes:
            self.axes = {
                "noetic": True,
                "element": True,
                "world": True,
                "foundation": True,
                "subFoundation": True,
                "acquisition": True,
                "causal": True,
                "narrativeRole": False,
                "scalarValence": True,
            }


def _invert_noetic(noetic_idx: int, mode: InversionMode) -> int:
    if mode in {InversionMode.Opposite, InversionMode.Polarity, InversionMode.NoeticComplement}:
        return NOETIC_OPPOSITE.get(noetic_idx, noetic_idx)
    if mode == InversionMode.Dual:
        return NOETIC_DUAL.get(noetic_idx, noetic_idx)
    if mode == InversionMode.CounterPole:
        return NOETIC_COUNTERPOLE.get(noetic_idx, noetic_idx)
    return noetic_idx


def _invert_world(world: str, mode: InversionMode, target: TargetProfile) -> str:
    if mode in {InversionMode.Opposite, InversionMode.DomainPermutation}:
        if target.enable and target.from_world and target.to_world and world == target.from_world:
            return target.to_world
        return WORLD_OPP.get(world, world)
    if mode == InversionMode.ParallelAnalogue and target.enable:
        if target.from_world and target.to_world and world == target.from_world:
            return target.to_world
    return world


def _invert_element(elem: str, mode: InversionMode, target: TargetProfile) -> str:
    world = elem[0]
    noetic = int(elem[1:]) - 1
    new_noetic = _invert_noetic(noetic, mode)
    new_world = _invert_world(world, mode, target)
    return f"{new_world}{new_noetic + 1}"


def _invert_foundation(fid: int, mode: InversionMode, target: TargetProfile) -> int:
    if mode in {InversionMode.Opposite, InversionMode.FoundationFlip, InversionMode.Polarity}:
        if target.enable and target.from_foundation and target.to_foundation and fid == target.from_foundation:
            return target.to_foundation
        return FOUNDATION_OPP.get(fid, fid)
    return fid


def _invert_subfoundation(fid: int, world: str, mode: InversionMode, target: TargetProfile) -> Tuple[int, str]:
    new_f = _invert_foundation(fid, mode, target)
    new_w = _invert_world(world, mode, target) if mode in {
        InversionMode.Opposite, InversionMode.SubFoundationReversal, InversionMode.DomainPermutation
    } else world
    return new_f, new_w


def _invert_acquisition(label: str, mode: InversionMode) -> str:
    if mode in {InversionMode.Opposite, InversionMode.AcquisitionPolarity, InversionMode.Polarity}:
        return label[1:] if label.startswith("¬") else f"¬{label}"
    return label


def invert_elements(elements: List[str], mode: InversionMode, dial: DialConfig) -> List[str]:
    if not dial.axes.get("element", True):
        return elements
    return [_invert_element(e, mode, dial.target_profile) for e in elements]


def invert_foundations(foundations: List[Tuple[int, str]], mode: InversionMode, dial: DialConfig) -> List[Tuple[int, str]]:
    if not dial.axes.get("foundation", True) and not dial.axes.get("subFoundation", True):
        return foundations
    out = []
    for fid, w in foundations:
        if dial.axes.get("foundation", True):
            fid = _invert_foundation(fid, mode, dial.target_profile)
        if dial.axes.get("subFoundation", True):
            fid, w = _invert_subfoundation(fid, w, mode, dial.target_profile)
        out.append((fid, w))
    return out


def invert_acquisitions(acqs: List[str], mode: InversionMode, dial: DialConfig) -> List[str]:
    if not dial.axes.get("acquisition", True):
        return acqs
    return [_invert_acquisition(a, mode) for a in acqs]


def invert_causal_chain(chain: List[str], ops: List[str], mode: InversionMode, dial: DialConfig) -> Tuple[List[str], List[str]]:
    if not dial.axes.get("causal", True):
        return chain, ops
    if mode == InversionMode.Mirror:
        return list(reversed(chain)), list(reversed(ops))
    if mode == InversionMode.ReverseCausal:
        rev_chain = list(reversed(chain))
        flip = {"->": "<-", "<-": "->", "+T": "-T", "-T": "+T", "→": "←", "←": "→"}
        rev_ops = [flip.get(op, op) for op in reversed(ops)]
        return rev_chain, rev_ops
    if mode in {InversionMode.Opposite, InversionMode.StructuralPermutation, InversionMode.TemporalInversion}:
        flip = {"+T": "-T", "-T": "+T", "→": "←", "←": "→", "->": "<-", "<-": "->"}
        return chain, [flip.get(op, op) for op in ops]
    return chain, ops


def total_inversion(
    elements: List[str],
    foundations: Optional[List[Tuple[int, str]]] = None,
    acquisitions: Optional[List[str]] = None,
    ops: Optional[List[str]] = None,
    mode: InversionMode = InversionMode.Opposite,
    dial: Optional[DialConfig] = None,
) -> Dict[str, object]:
    if dial is None:
        dial = DialConfig(mode=mode)
    inv_elements = invert_elements(elements, mode, dial)
    inv_foundations = invert_foundations(foundations or [], mode, dial)
    inv_acqs = invert_acquisitions(acquisitions or [], mode, dial)
    inv_chain, inv_ops = invert_causal_chain(inv_elements, ops or [], mode, dial)
    return {
        "mode": mode.value,
        "elements": inv_elements,
        "foundations": inv_foundations,
        "acquisitions": inv_acqs,
        "chain": inv_chain,
        "ops": inv_ops,
    }

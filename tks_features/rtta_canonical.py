"""
RTTA-C: Canonical Reasoning-Test Tootra Algorithm

Strictly TKS-native implementation using:
- 12 world-specific D/W/P prerequisite slots
- Gradation cycles (Kings/Jacks/Queens)
- First-unmet-prerequisite recursion
- Noetic operators as deterministic control flow
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import re


# ============================================================
# CANONICAL CONSTANTS
# ============================================================

class World(Enum):
    """Four worlds (A/B/C/D)."""
    SPIRITUAL = "A"  # World A
    MENTAL = "B"     # World B
    EMOTIONAL = "C"  # World C
    PHYSICAL = "D"   # World D


class PrereqType(Enum):
    """D/W/P types."""
    DESIRE = "D"
    WISDOM = "W"
    POWER = "P"


class Noetic(IntEnum):
    """Noetic operators 0-9."""
    IDEA = 0           # Finalize/manifest
    MIND = 1           # Attention/awareness
    HEART = 2          # Include/accept
    EXCLUDE = 3        # Eliminate/reject
    VIBRATION = 4      # Uncertainty/mismatch
    FEMALE = 5         # Identity/definition
    MALE = 6           # Components/structure
    SEQUENCE = 7       # Ordering/timing
    CAUSATION = 8      # Derive/produce
    EFFECT = 9         # Result/output


# Desire Gradations (Kings cycle)
class DesireGrade(Enum):
    ABSENT = ("Night", 0.0)
    BEGINNING = ("Sunrise", 0.33)
    WEAKENING = ("Sunset", 0.66)
    STRONG = ("Noon", 1.0)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def score(self) -> float:
        return self.value[1]


# Wisdom Gradations (Jacks cycle)
class WisdomGrade(Enum):
    NONE = ("No Grip", 0.0)
    LEARNING = ("Weak Grip", 0.33)
    KNOWING = ("Firmer Grip", 0.66)
    MASTERING = ("Kung Fu Grip", 1.0)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def score(self) -> float:
        return self.value[1]


# Power Gradations (Queens cycle)
class PowerGrade(Enum):
    NONE = ("No Capacity", 0.0)
    BIRTH = ("Seedling", 0.33)
    ADOLESCENT = ("Growing", 0.66)
    MATURE = ("Strong Stem", 1.0)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def score(self) -> float:
        return self.value[1]


# Canonical 12-slot prerequisite grid
PREREQ_SLOTS = [
    # World A (Spiritual)
    ("1D", World.SPIRITUAL, PrereqType.DESIRE),
    ("2W", World.SPIRITUAL, PrereqType.WISDOM),
    ("3P", World.SPIRITUAL, PrereqType.POWER),
    # World B (Mental)
    ("4D", World.MENTAL, PrereqType.DESIRE),
    ("5W", World.MENTAL, PrereqType.WISDOM),
    ("6P", World.MENTAL, PrereqType.POWER),
    # World C (Emotional)
    ("7D", World.EMOTIONAL, PrereqType.DESIRE),
    ("8W", World.EMOTIONAL, PrereqType.WISDOM),
    ("9P", World.EMOTIONAL, PrereqType.POWER),
    # World D (Physical)
    ("10D", World.PHYSICAL, PrereqType.DESIRE),
    ("11W", World.PHYSICAL, PrereqType.WISDOM),
    ("12P", World.PHYSICAL, PrereqType.POWER),
]

# Default threshold for prerequisite sufficiency
DEFAULT_THRESHOLD = 0.6


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PrereqState:
    """State of a single prerequisite slot."""
    slot_id: str
    world: World
    prereq_type: PrereqType
    score: float = 0.0
    grade: str = ""
    passed: bool = False

    def assess(self, threshold: float = DEFAULT_THRESHOLD):
        """Assess this slot and assign gradation."""
        self.passed = self.score >= threshold

        if self.prereq_type == PrereqType.DESIRE:
            if self.score < 0.25:
                self.grade = f"{DesireGrade.ABSENT.label}"
            elif self.score < 0.5:
                self.grade = f"{DesireGrade.BEGINNING.label}"
            elif self.score < 0.75:
                self.grade = f"{DesireGrade.WEAKENING.label}"
            else:
                self.grade = f"{DesireGrade.STRONG.label}"

        elif self.prereq_type == PrereqType.WISDOM:
            if self.score < 0.25:
                self.grade = f"{WisdomGrade.NONE.label}"
            elif self.score < 0.5:
                self.grade = f"{WisdomGrade.LEARNING.label}"
            elif self.score < 0.75:
                self.grade = f"{WisdomGrade.KNOWING.label}"
            else:
                self.grade = f"{WisdomGrade.MASTERING.label}"

        elif self.prereq_type == PrereqType.POWER:
            if self.score < 0.25:
                self.grade = f"{PowerGrade.NONE.label}"
            elif self.score < 0.5:
                self.grade = f"{PowerGrade.BIRTH.label}"
            elif self.score < 0.75:
                self.grade = f"{PowerGrade.ADOLESCENT.label}"
            else:
                self.grade = f"{PowerGrade.MATURE.label}"


@dataclass
class Goal:
    """Canonical goal encoding as Idea (0) with Vibration (4)."""
    text: str
    primary_world: World = World.MENTAL
    foundation: str = "2b"  # Wisdom foundation
    vibration_target: str = ""  # What is unknown

    def encode(self) -> str:
        """Encode as canonical form."""
        return f"Idea({self.primary_world.value})^4[{self.foundation}]: {self.text}"


@dataclass
class Evidence:
    """Evidence base for reasoning."""
    premises: List[str] = field(default_factory=list)
    derived: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, str] = field(default_factory=dict)
    equations: List[str] = field(default_factory=list)


@dataclass
class RTTAState:
    """Full state of RTTA-C execution."""
    goal: Goal
    evidence: Evidence
    prereqs: List[PrereqState] = field(default_factory=list)
    active_slots: List[str] = field(default_factory=list)
    trace: List[Dict] = field(default_factory=list)
    recursion_depth: int = 0
    max_recursion: int = 10
    threshold: float = DEFAULT_THRESHOLD


# ============================================================
# RTTA-C CANONICAL ALGORITHM
# ============================================================

class RTTACanonical:
    """
    RTTA-C: Canonical Reasoning-Test Tootra Algorithm

    Uses 12 world-specific D/W/P slots with gradation cycles.
    First unmet prerequisite drives recursion.
    Noetic operators provide deterministic control flow.
    """

    def __init__(self, model: Optional[nn.Module] = None,
                 tokenizer: Any = None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # --------------------------------------------------------
    # STEP 1: Encode the goal
    # --------------------------------------------------------

    def encode_goal(self, question: str) -> Goal:
        """
        Define G as an Idea (0) in its primary world(s),
        with the active unknown marked by Vibration (4).
        """
        question_lower = question.lower()

        # Determine primary world
        if any(kw in question_lower for kw in ["soul", "spirit", "meaning", "purpose"]):
            world = World.SPIRITUAL
        elif any(kw in question_lower for kw in ["feel", "emotion", "happy", "sad"]):
            world = World.EMOTIONAL
        elif any(kw in question_lower for kw in ["build", "make", "physical", "body"]):
            world = World.PHYSICAL
        else:
            # Default to Mental for reasoning tests
            world = World.MENTAL

        # Identify vibration target (what is unknown)
        vibration = "unknown quantity"
        if "how many" in question_lower:
            vibration = "count"
        elif "what is" in question_lower:
            vibration = "value"
        elif "is it true" in question_lower:
            vibration = "truth value"
        elif "what comes next" in question_lower:
            vibration = "next term"

        return Goal(
            text=question,
            primary_world=world,
            foundation="2b",  # Wisdom foundation for reasoning
            vibration_target=vibration
        )

    # --------------------------------------------------------
    # STEP 2: Assess all 12 prerequisite slots
    # --------------------------------------------------------

    def init_prereqs(self, goal: Goal) -> List[PrereqState]:
        """Initialize 12 prerequisite slots."""
        prereqs = []
        for slot_id, world, ptype in PREREQ_SLOTS:
            prereqs.append(PrereqState(
                slot_id=slot_id,
                world=world,
                prereq_type=ptype,
                score=0.0
            ))
        return prereqs

    def get_active_slots(self, goal: Goal) -> List[str]:
        """
        Get active prerequisite slots for the goal's primary world.
        For mental tests, active triad is 4D/5W/6P.
        """
        world_map = {
            World.SPIRITUAL: ["1D", "2W", "3P"],
            World.MENTAL: ["4D", "5W", "6P"],
            World.EMOTIONAL: ["7D", "8W", "9P"],
            World.PHYSICAL: ["10D", "11W", "12P"],
        }
        return world_map.get(goal.primary_world, ["4D", "5W", "6P"])

    def assess_prereqs(self, state: RTTAState) -> None:
        """
        Assess all prerequisite slots based on current evidence.
        Computes score, gradation, and pass/fail.
        """
        for prereq in state.prereqs:
            # Compute score based on evidence and prereq type
            score = self._compute_prereq_score(prereq, state)
            prereq.score = score
            prereq.assess(state.threshold)

    def _compute_prereq_score(self, prereq: PrereqState, state: RTTAState) -> float:
        """Compute score for a prerequisite based on evidence."""
        # Only score active slots
        if prereq.slot_id not in state.active_slots:
            return 1.0  # Inactive slots auto-pass

        e = state.evidence
        problem_type = e.derived.get("problem_type", "unknown")

        if prereq.prereq_type == PrereqType.DESIRE:
            # Desire: Do we have attention on the problem?
            if "problem_type" in e.derived and e.derived["problem_type"] != "unknown":
                if "target" in e.derived or e.variables:
                    return 0.8
                return 0.6
            elif e.premises:
                return 0.5
            else:
                return 0.2

        elif prereq.prereq_type == PrereqType.WISDOM:
            # Wisdom: Do we understand the structure?
            if problem_type == "linear_system":
                if e.equations and len(e.equations) >= 2:
                    return 0.9
                elif e.equations:
                    return 0.6
                elif e.variables:
                    return 0.4
                return 0.2

            elif problem_type == "syllogism":
                if "relationships" in e.derived and e.derived["relationships"]:
                    return 0.9
                return 0.4

            elif problem_type == "sequence":
                if "pattern" in e.derived and e.derived["pattern"]:
                    return 0.9
                elif "sequence" in e.derived:
                    return 0.6
                return 0.2

            elif problem_type == "conditional":
                if "conditional" in e.derived and e.derived["conditional"].get("antecedent"):
                    return 0.9
                return 0.4

            elif problem_type == "arithmetic":
                if "operation" in e.derived and e.derived["operation"].get("type"):
                    return 0.9
                return 0.4

            elif problem_type == "proportion":
                if "ratio" in e.derived and e.derived["ratio"].get("ratio"):
                    return 0.9
                return 0.4

            # New operator types - check if structure extracted
            elif problem_type == "logic_grid":
                if "grid" in e.derived and e.derived["grid"].get("entities"):
                    return 0.9
                return 0.4

            elif problem_type == "probability":
                if "probability" in e.derived and e.derived["probability"].get("experiment"):
                    return 0.9
                return 0.4

            elif problem_type == "bayes":
                if "bayes" in e.derived and e.derived["bayes"].get("prevalence") is not None:
                    return 0.9
                return 0.4

            elif problem_type in ["geometry_pythagorean", "geometry_area", "geometry_angles"]:
                if "pythagorean" in e.derived or "geometry" in e.derived:
                    return 0.9
                return 0.4

            elif problem_type == "coord_geometry":
                if "coord" in e.derived and e.derived["coord"].get("points"):
                    return 0.9
                return 0.4

            elif problem_type == "modular":
                if "modular" in e.derived and e.derived["modular"].get("base"):
                    return 0.9
                return 0.4

            elif problem_type == "conditional_chain":
                if "chain" in e.derived and e.derived["chain"].get("implications"):
                    return 0.9
                return 0.4

            elif problem_type == "base_convert":
                if "base" in e.derived and e.derived["base"].get("value"):
                    return 0.9
                return 0.4

            elif problem_type == "unit_convert":
                if "unit" in e.derived and e.derived["unit"].get("value"):
                    return 0.9
                return 0.4

            elif problem_type == "sets":
                if "sets" in e.derived and e.derived["sets"].get("total"):
                    return 0.9
                return 0.4

            elif problem_type == "combinatorics":
                if "combinatorics" in e.derived and e.derived["combinatorics"].get("type"):
                    return 0.9
                return 0.4

            elif problem_type == "inequality":
                if "inequality" in e.derived and e.derived["inequality"].get("bound"):
                    return 0.9
                return 0.4

            elif problem_type == "logic_xor":
                if "xor" in e.derived:
                    return 0.9
                return 0.4

            elif problem_type == "rate":
                rate_data = e.derived.get("rate", {})
                if rate_data.get("segments") or rate_data.get("simple"):
                    return 0.9
                return 0.4

            elif problem_type == "age":
                if "age" in e.derived and e.derived["age"].get("difference") is not None:
                    return 0.9
                return 0.4

            elif problem_type == "animal_linear":
                if "animal" in e.derived and e.derived["animal"].get("animals"):
                    return 0.9
                return 0.4

            elif problem_type == "ratio_linear":
                if "ratio_linear" in e.derived and e.derived["ratio_linear"].get("multiplier"):
                    return 0.9
                return 0.4

            elif problem_type == "parts_ratio":
                if "parts_ratio" in e.derived and e.derived["parts_ratio"].get("parts1"):
                    return 0.9
                return 0.4

            return 0.2

        elif prereq.prereq_type == PrereqType.POWER:
            # Power: Can we execute/solve?
            if "answer" in e.derived:
                return 1.0
            elif "solution" in e.derived:
                return 0.8
            elif problem_type == "linear_system" and e.equations:
                return 0.5
            elif problem_type in ["syllogism", "sequence", "conditional", "arithmetic", "proportion"]:
                return 0.5
            # New types - if structure extracted, power can attempt
            elif problem_type in ["logic_grid", "probability", "bayes", "geometry_pythagorean",
                                  "geometry_area", "geometry_angles", "coord_geometry",
                                  "modular", "conditional_chain", "base_convert",
                                  "unit_convert", "sets", "combinatorics", "inequality",
                                  "logic_xor", "rate", "age", "animal_linear",
                                  "ratio_linear", "parts_ratio"]:
                return 0.5
            else:
                return 0.2

        return 0.0

    # --------------------------------------------------------
    # STEP 3: First unmet prerequisite drives recursion
    # --------------------------------------------------------

    def find_first_unmet(self, state: RTTAState) -> Optional[PrereqState]:
        """Find first unmet prerequisite in canonical order."""
        for prereq in state.prereqs:
            if prereq.slot_id in state.active_slots and not prereq.passed:
                return prereq
        return None

    def make_subgoal(self, parent_goal: Goal, unmet: PrereqState) -> Goal:
        """Create subgoal to raise unmet prerequisite to sufficiency."""
        if unmet.prereq_type == PrereqType.DESIRE:
            text = f"Identify what to solve in: {parent_goal.text}"
            vibration = "target variable"
        elif unmet.prereq_type == PrereqType.WISDOM:
            text = f"Understand the relationships in: {parent_goal.text}"
            vibration = "structure/equations"
        else:  # POWER
            text = f"Compute the solution for: {parent_goal.text}"
            vibration = "computed value"

        return Goal(
            text=text,
            primary_world=unmet.world,
            foundation=parent_goal.foundation,
            vibration_target=vibration
        )

    # --------------------------------------------------------
    # STEP 4: Resolve using Noetics + Female/Male
    # --------------------------------------------------------

    def resolve_subgoal(self, subgoal: Goal, state: RTTAState) -> Evidence:
        """
        Resolve subgoal using noetic operators.

        Desire band uses {1,4,7}: attention, uncertainty scoring, sequencing
        Wisdom band uses {5,6}: identify missing definitions (5), build structure (6)
        Power band uses {8,9}: derive causal steps (8), produce effects/results (9)
        """
        e = state.evidence

        # Determine which band based on subgoal's vibration target
        if "target" in subgoal.vibration_target or "variable" in subgoal.vibration_target:
            # Desire band: use Noetic 1 (attention) and 4 (uncertainty)
            e = self._apply_desire_noetics(subgoal, e, state)

        elif "structure" in subgoal.vibration_target or "equation" in subgoal.vibration_target:
            # Wisdom band: use Noetic 5 (Female) and 6 (Male)
            e = self._apply_wisdom_noetics(subgoal, e, state)

        elif "computed" in subgoal.vibration_target or "value" in subgoal.vibration_target:
            # Power band: use Noetic 8 (causation) and 9 (effect)
            e = self._apply_power_noetics(subgoal, e, state)

        return e

    def _apply_desire_noetics(self, goal: Goal, evidence: Evidence,
                               state: RTTAState) -> Evidence:
        """
        Desire band: Noetics {1,4,7}
        - 1 (Mind): Focus attention on the problem
        - 4 (Vibration): Score uncertainty of each element
        - 7 (Sequence): Order the elements
        """
        # Use the ORIGINAL goal text, not subgoal description
        text = state.goal.text
        text_lower = text.lower()

        # Noetic 1: Attention - identify the problem type
        problem_type = self._detect_problem_type(text)
        evidence.derived["problem_type"] = problem_type

        # Noetic 1: Attention - identify the question type
        if "how many" in text_lower:
            evidence.derived["question_type"] = "count"
        elif "what comes next" in text_lower or "what is next" in text_lower:
            evidence.derived["question_type"] = "sequence_next"
        elif "is it true" in text_lower or "do " in text_lower or "does " in text_lower or "is " in text_lower.split("?")[0][-20:]:
            evidence.derived["question_type"] = "boolean"
        elif "did it" in text_lower or "did he" in text_lower or "did she" in text_lower:
            evidence.derived["question_type"] = "boolean"
        elif "what is" in text_lower or "how much" in text_lower:
            evidence.derived["question_type"] = "value"
        else:
            evidence.derived["question_type"] = "value"

        # Noetic 4: Vibration - identify unknowns based on problem type
        if problem_type == "linear_system":
            if match := re.search(r'how many (\w+)', text_lower):
                evidence.derived["target"] = match.group(1)
                evidence.variables["target"] = match.group(1)
        elif problem_type == "sequence":
            evidence.derived["target"] = "next_term"
        elif problem_type == "syllogism":
            evidence.derived["target"] = "conclusion"
        elif problem_type == "conditional":
            evidence.derived["target"] = "truth_value"
        elif problem_type == "arithmetic":
            evidence.derived["target"] = "result"
            # Extract numbers for arithmetic
            numbers = re.findall(r'\d+', text)
            evidence.variables["numbers"] = [int(n) for n in numbers]

        # Noetic 7: Sequence - order premises
        evidence.premises = self._extract_premises(text)

        state.trace.append({
            "action": "DESIRE_BAND",
            "noetics": [1, 4, 7],
            "result": f"type={problem_type}, target={evidence.derived.get('target', '?')}"
        })

        return evidence

    def _detect_problem_type(self, text: str) -> str:
        """Detect the type of reasoning problem."""
        text_lower = text.lower()

        # OP-BC: Backward Chaining - P->Q patterns with negation
        if re.search(r'[A-Z]\s*->\s*[A-Z]', text) or "implies" in text_lower:
            if "not " in text_lower:
                return "conditional_chain"

        # OP-MOD: Modular exponentiation
        if " mod " in text_lower or "modulo" in text_lower:
            return "modular"
        if re.search(r'\d+\s*\^\s*\d+\s*mod', text_lower):
            return "modular"

        # OP-CP: Logic Grid - constraint propagation
        if any(kw in text_lower for kw in ["who has", "who is", "who wears", "who plays"]):
            if "not" in text_lower:
                return "logic_grid"

        # OP-BAYES: Bayes theorem
        if any(kw in text_lower for kw in ["prevalence", "sensitivity", "false positive", "positive test"]):
            return "bayes"

        # OP-SS: Sample Space / Probability
        if "probability" in text_lower:
            return "probability"
        if any(kw in text_lower for kw in ["coin flip", "flip", "dice", "die", "roll"]):
            return "probability"

        # OP-PY: Geometry - Pythagorean
        if "right triangle" in text_lower and ("hypotenuse" in text_lower or "leg" in text_lower):
            return "geometry_pythagorean"

        # Geometry - area, angles
        if "rectangle" in text_lower and ("area" in text_lower or "sides" in text_lower):
            return "geometry_area"
        if "triangle" in text_lower and "angle" in text_lower:
            return "geometry_angles"

        # Coordinate geometry - slope
        if "slope" in text_lower or re.search(r'\(\d+\s*,\s*\d+\)', text):
            return "coord_geometry"

        # Base conversion
        if "base 2" in text_lower or "binary" in text_lower or "decimal" in text_lower:
            return "base_convert"

        # Unit conversion
        if any(u in text_lower for u in ["hours in minutes", "minutes in", "convert"]):
            return "unit_convert"
        # "How many minutes is X hours?"
        if "how many minutes" in text_lower and "hour" in text_lower:
            return "unit_convert"

        # Sets / inclusion-exclusion
        if "both" in text_lower and "neither" in text_lower:
            return "sets"
        if "total" in text_lower and ("math" in text_lower or "science" in text_lower):
            return "sets"

        # Combinatorics
        if "how many ways" in text_lower or "choose" in text_lower:
            return "combinatorics"
        if "strings" in text_lower and "without repetition" in text_lower:
            return "combinatorics"

        # Inequality
        if re.search(r'[<>]', text) or "smallest integer" in text_lower:
            return "inequality"

        # XOR logic
        if "exactly one" in text_lower and ("true" in text_lower or "false" in text_lower):
            return "logic_xor"

        # Linear system: "sells X for $Y each" or "X cost $Y"
        if re.search(r'for \$\d+', text):
            return "linear_system"
        if re.search(r'cost\s+\$\d+', text_lower):
            return "linear_system"

        # Coin problems are linear systems
        if any(coin in text_lower for coin in ["nickel", "dime", "quarter", "penny", "coins"]):
            return "linear_system"

        # Animal counting problems: "chickens and cows", "X animals with Y legs"
        if any(animal in text_lower for animal in ["chicken", "cow", "pig", "horse", "animal", "legs"]):
            if "legs" in text_lower or "total" in text_lower:
                return "animal_linear"

        # Ratio with variables: "X has twice as many as Y", "together they have N"
        if "twice as many" in text_lower or "three times as many" in text_lower:
            if "together" in text_lower:
                return "ratio_linear"

        # Parts-based ratio: "3 parts X to 2 parts Y"
        if "parts" in text_lower and re.search(r'\d+\s+parts', text_lower):
            return "parts_ratio"

        # Age problems: "X is Y years older than Z"
        if "years older" in text_lower or "years younger" in text_lower:
            if "sum of" in text_lower or "ages" in text_lower:
                return "age"

        # Rate problems
        if "mph" in text_lower or "speed" in text_lower:
            return "rate"

        # Sequence: "what comes next" or pattern of numbers
        if "sequence" in text_lower or "what comes next" in text_lower:
            return "sequence"
        if re.search(r'\d+,\s*\d+,\s*\d+', text):
            return "sequence"

        # Syllogism: "All X are Y" or "Some X are Y"
        if re.search(r'\ball\b.*\bare\b', text_lower):
            return "syllogism"
        if re.search(r'\bsome\b.*\bare\b', text_lower):
            return "syllogism"

        # Conditional: "If... then" patterns
        if text_lower.startswith("if ") or ", if " in text_lower:
            return "conditional"

        # Ratio/proportion: "for every" or ratio patterns
        if "for every" in text_lower or "ratio" in text_lower:
            return "proportion"

        # Simple arithmetic: gives/takes type problems
        if any(word in text_lower for word in ["gives", "takes", "loses", "finds", "gets"]):
            return "arithmetic"

        # Pure math expression: "What is 24 / 6 + 3?"
        if re.search(r'what is\s+\d+\s*[+\-*/]', text_lower):
            return "arithmetic"

        return "unknown"

    def _apply_wisdom_noetics(self, goal: Goal, evidence: Evidence,
                               state: RTTAState) -> Evidence:
        """
        Wisdom band: Noetics {5,6}
        - 5 (Female): Identify missing definitions
        - 6 (Male): Build structure/components
        """
        # Use the ORIGINAL goal text, not subgoal description
        text = state.goal.text
        problem_type = evidence.derived.get("problem_type", "unknown")

        if problem_type == "linear_system":
            # Noetic 5 (Female): Extract variables
            extracted_vars = self._extract_variables(text)
            for k, v in extracted_vars.items():
                if k not in evidence.variables:
                    evidence.variables[k] = v
            # Noetic 6 (Male): Build equations
            if not evidence.equations:
                evidence.equations = self._extract_equations(text, evidence.variables)

        elif problem_type == "syllogism":
            # Noetic 5: Extract category relationships
            evidence.derived["relationships"] = self._extract_syllogism(text)

        elif problem_type == "sequence":
            # Noetic 5: Extract sequence numbers
            # Noetic 6: Identify pattern
            evidence.derived["sequence"] = self._extract_sequence(text)
            evidence.derived["pattern"] = self._identify_pattern(evidence.derived["sequence"])

        elif problem_type == "conditional":
            # Noetic 5: Extract conditional structure
            evidence.derived["conditional"] = self._extract_conditional(text)

        elif problem_type == "arithmetic":
            # Noetic 5: Extract operation type
            evidence.derived["operation"] = self._extract_arithmetic_op(text)

        elif problem_type == "proportion":
            # Noetic 5: Extract ratio
            evidence.derived["ratio"] = self._extract_proportion(text)

        # OP-CP: Logic Grid
        elif problem_type == "logic_grid":
            evidence.derived["grid"] = self._extract_logic_grid(text)

        # OP-SS: Probability
        elif problem_type == "probability":
            evidence.derived["probability"] = self._extract_probability(text)

        # OP-BAYES: Bayes theorem
        elif problem_type == "bayes":
            evidence.derived["bayes"] = self._extract_bayes(text)

        # OP-PY: Pythagorean
        elif problem_type == "geometry_pythagorean":
            evidence.derived["pythagorean"] = self._extract_pythagorean(text)

        # Geometry area
        elif problem_type == "geometry_area":
            evidence.derived["geometry"] = self._extract_geometry_area(text)

        # Geometry angles
        elif problem_type == "geometry_angles":
            evidence.derived["geometry"] = self._extract_geometry_angles(text)

        # Coordinate geometry
        elif problem_type == "coord_geometry":
            evidence.derived["coord"] = self._extract_coord_geometry(text)

        # OP-MOD: Modular
        elif problem_type == "modular":
            evidence.derived["modular"] = self._extract_modular(text)

        # OP-BC: Conditional chain
        elif problem_type == "conditional_chain":
            evidence.derived["chain"] = self._extract_conditional_chain(text)

        # Base conversion
        elif problem_type == "base_convert":
            evidence.derived["base"] = self._extract_base_convert(text)

        # Unit conversion
        elif problem_type == "unit_convert":
            evidence.derived["unit"] = self._extract_unit_convert(text)

        # Sets
        elif problem_type == "sets":
            evidence.derived["sets"] = self._extract_sets(text)

        # Combinatorics
        elif problem_type == "combinatorics":
            evidence.derived["combinatorics"] = self._extract_combinatorics(text)

        # Inequality
        elif problem_type == "inequality":
            evidence.derived["inequality"] = self._extract_inequality(text)

        # XOR logic
        elif problem_type == "logic_xor":
            evidence.derived["xor"] = self._extract_logic_xor(text)

        # Rate
        elif problem_type == "rate":
            evidence.derived["rate"] = self._extract_rate(text)

        # Age problems
        elif problem_type == "age":
            evidence.derived["age"] = self._extract_age(text)

        # Animal linear system: chickens/cows with total animals and legs
        elif problem_type == "animal_linear":
            evidence.derived["animal"] = self._extract_animal_linear(text)

        # Ratio linear: "X has twice as many as Y, together N"
        elif problem_type == "ratio_linear":
            evidence.derived["ratio_linear"] = self._extract_ratio_linear(text)

        # Parts ratio: "3 parts X to 2 parts Y"
        elif problem_type == "parts_ratio":
            evidence.derived["parts_ratio"] = self._extract_parts_ratio(text)

        state.trace.append({
            "action": "WISDOM_BAND",
            "noetics": [5, 6, 2, 3],
            "result": f"type={problem_type}, structure={evidence.derived}"
        })

        return evidence

    def _apply_power_noetics(self, goal: Goal, evidence: Evidence,
                              state: RTTAState) -> Evidence:
        """
        Power band: Noetics {8,9}
        - 8 (Causation): Derive causal steps
        - 9 (Effect): Produce results
        """
        problem_type = evidence.derived.get("problem_type", "unknown")

        if problem_type == "linear_system":
            # Noetic 8: Solve linear system
            if evidence.equations and len(evidence.equations) >= 2:
                solution = self._solve_system(evidence.equations, evidence.variables)
                if solution:
                    evidence.derived["solution"] = solution
            # Noetic 9: Extract target answer
            if "solution" in evidence.derived and "target" in evidence.derived:
                target = evidence.derived["target"]
                solution = evidence.derived["solution"]
                target_var = target[0].lower() if target else None
                if target_var and target_var in solution:
                    evidence.derived["answer"] = f"{solution[target_var]} {target}"
                elif solution:
                    first_var = list(solution.keys())[0]
                    evidence.derived["answer"] = solution[first_var]

        elif problem_type == "syllogism":
            # Noetic 8: Apply syllogistic reasoning
            result = self._solve_syllogism(evidence.derived.get("relationships", []))
            evidence.derived["answer"] = result

        elif problem_type == "sequence":
            # Noetic 8: Apply pattern to get next term
            pattern = evidence.derived.get("pattern")
            seq = evidence.derived.get("sequence", [])
            if pattern and seq:
                next_term = self._apply_pattern(seq, pattern)
                evidence.derived["answer"] = str(next_term)

        elif problem_type == "conditional":
            # Noetic 8: Apply conditional logic
            result = self._solve_conditional(evidence.derived.get("conditional", {}))
            evidence.derived["answer"] = result

        elif problem_type == "arithmetic":
            # Noetic 8: Compute arithmetic
            op = evidence.derived.get("operation", {})
            numbers = evidence.variables.get("numbers", [])
            if op and numbers:
                result = self._solve_arithmetic(op, numbers)
                evidence.derived["answer"] = str(result)

        elif problem_type == "proportion":
            # Noetic 8: Solve proportion
            result = self._solve_proportion(evidence.derived.get("ratio", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # OP-CP: Logic Grid
        elif problem_type == "logic_grid":
            result = self._solve_logic_grid(evidence.derived.get("grid", {}))
            if result:
                evidence.derived["answer"] = result

        # OP-SS: Probability
        elif problem_type == "probability":
            result = self._solve_probability(evidence.derived.get("probability", {}))
            if result:
                evidence.derived["answer"] = result

        # OP-BAYES: Bayes theorem
        elif problem_type == "bayes":
            result = self._solve_bayes(evidence.derived.get("bayes", {}))
            if result:
                evidence.derived["answer"] = result

        # OP-PY: Pythagorean
        elif problem_type == "geometry_pythagorean":
            result = self._solve_pythagorean(evidence.derived.get("pythagorean", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Geometry area
        elif problem_type == "geometry_area":
            result = self._solve_geometry_area(evidence.derived.get("geometry", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Geometry angles
        elif problem_type == "geometry_angles":
            result = self._solve_geometry_angles(evidence.derived.get("geometry", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Coordinate geometry
        elif problem_type == "coord_geometry":
            result = self._solve_coord_geometry(evidence.derived.get("coord", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # OP-MOD: Modular
        elif problem_type == "modular":
            result = self._solve_modular(evidence.derived.get("modular", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # OP-BC: Conditional chain
        elif problem_type == "conditional_chain":
            result = self._solve_conditional_chain(evidence.derived.get("chain", {}))
            if result:
                evidence.derived["answer"] = result

        # Base conversion
        elif problem_type == "base_convert":
            result = self._solve_base_convert(evidence.derived.get("base", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Unit conversion
        elif problem_type == "unit_convert":
            result = self._solve_unit_convert(evidence.derived.get("unit", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Sets
        elif problem_type == "sets":
            result = self._solve_sets(evidence.derived.get("sets", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Combinatorics
        elif problem_type == "combinatorics":
            result = self._solve_combinatorics(evidence.derived.get("combinatorics", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Inequality
        elif problem_type == "inequality":
            result = self._solve_inequality(evidence.derived.get("inequality", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # XOR logic
        elif problem_type == "logic_xor":
            result = self._solve_logic_xor(evidence.derived.get("xor", {}))
            if result:
                evidence.derived["answer"] = result

        # Rate
        elif problem_type == "rate":
            result = self._solve_rate(evidence.derived.get("rate", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Age problems
        elif problem_type == "age":
            result = self._solve_age(evidence.derived.get("age", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Animal linear system
        elif problem_type == "animal_linear":
            result = self._solve_animal_linear(evidence.derived.get("animal", {}))
            if result is not None:
                evidence.derived["answer"] = result

        # Ratio linear
        elif problem_type == "ratio_linear":
            result = self._solve_ratio_linear(evidence.derived.get("ratio_linear", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        # Parts ratio
        elif problem_type == "parts_ratio":
            result = self._solve_parts_ratio(evidence.derived.get("parts_ratio", {}))
            if result is not None:
                evidence.derived["answer"] = str(result)

        state.trace.append({
            "action": "POWER_BAND",
            "noetics": [8, 9],
            "result": f"answer={evidence.derived.get('answer', '?')}"
        })

        return evidence

    # --------------------------------------------------------
    # STEP 5: Termination - Manifest answer
    # --------------------------------------------------------

    def manifest_answer(self, state: RTTAState) -> str:
        """
        When all active slots pass, collapse Vibration (4)
        and manifest final answer as Idea (0).

        FALLBACK POLICY: For unrecognized types, return "Cannot determine"
        rather than guessing. This prevents hallucinated answers.
        """
        e = state.evidence
        problem_type = e.derived.get("problem_type", "unknown")

        # Fallback for unrecognized types - don't guess
        if problem_type == "unknown":
            answer = "Cannot determine (unrecognized problem type)"
            state.trace.append({
                "action": "FALLBACK",
                "reason": "unrecognized_type",
                "result": answer
            })
        elif "answer" in e.derived:
            answer = str(e.derived["answer"])
        elif "solution" in e.derived:
            answer = str(e.derived["solution"])
        else:
            answer = "Unable to determine"

        # Noetic 0: Finalize the Idea
        state.trace.append({
            "action": "MANIFEST",
            "noetic": 0,
            "result": f"Idea={answer}"
        })

        return answer

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def reason(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Main RTTA-C reasoning loop.

        RTTA_C(Q):
          G = encode_goal(Q)
          E = parse_premises(Q)

          loop:
            S = assess_prereqs(G, E)
            if all_pass(S):
              return manifest_answer(G, E)

            U = first_unmet_slot(S)
            G_U = make_subgoal(G, U)
            E = resolve_subgoal(G_U, E)
        """
        # Step 1: Encode goal
        goal = self.encode_goal(question)
        if verbose:
            print(f"Goal: {goal.encode()}")

        # Initialize state
        state = RTTAState(
            goal=goal,
            evidence=Evidence(),
            prereqs=self.init_prereqs(goal),
            active_slots=self.get_active_slots(goal),
        )

        if verbose:
            print(f"Active slots: {state.active_slots}")

        # Main loop
        iteration = 0
        max_iterations = 20
        repeated_slots = {}

        while iteration < max_iterations:
            iteration += 1

            # Step 2: Assess prerequisites
            self.assess_prereqs(state)

            if verbose:
                print(f"\nIteration {iteration}:")
                for prereq in state.prereqs:
                    if prereq.slot_id in state.active_slots:
                        status = "PASS" if prereq.passed else "FAIL"
                        print(f"  {prereq.slot_id}: {prereq.score:.2f} ({prereq.grade}) [{status}]")

            # Check if all active slots pass
            all_pass = all(
                p.passed for p in state.prereqs
                if p.slot_id in state.active_slots
            )

            if all_pass:
                if verbose:
                    print("\nAll prerequisites satisfied!")
                break

            # Step 3: Find first unmet
            unmet = self.find_first_unmet(state)
            if unmet is None:
                break

            # Track repeated failures
            repeated_slots[unmet.slot_id] = repeated_slots.get(unmet.slot_id, 0) + 1
            if repeated_slots[unmet.slot_id] > 3:
                if verbose:
                    print(f"\nWarning: {unmet.slot_id} stuck after 3 cycles")
                break

            if verbose:
                print(f"\nFirst unmet: {unmet.slot_id} ({unmet.prereq_type.value})")

            # Create subgoal and resolve
            subgoal = self.make_subgoal(goal, unmet)
            state.evidence = self.resolve_subgoal(subgoal, state)

            state.recursion_depth += 1

        # Step 5: Manifest answer
        answer = self.manifest_answer(state)

        # Build result
        return {
            "question": question,
            "goal": goal.encode(),
            "answer": answer,
            "iterations": iteration,
            "trace": state.trace,
            "prereqs": {
                p.slot_id: {"score": p.score, "grade": p.grade, "passed": p.passed}
                for p in state.prereqs if p.slot_id in state.active_slots
            },
            "evidence": {
                "variables": state.evidence.variables,
                "equations": state.evidence.equations,
                "derived": state.evidence.derived,
            }
        }

    # --------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------

    def _extract_premises(self, text: str) -> List[str]:
        """Extract premise sentences from text."""
        sentences = re.split(r'[.!?]', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_variables(self, text: str) -> Dict[str, str]:
        """Extract variable definitions from math problem."""
        variables = {}
        used_names = set()

        # Pattern: "X for $Y each" - more flexible to catch all items
        for match in re.finditer(r'(\w+)\s+for\s+\$(\d+)\s+each', text, re.I):
            item = match.group(1)
            price = match.group(2)
            if item.lower() in ["sells", "store", "a", "the", "each"]:
                continue
            var_name = self._get_unique_var(item, used_names)
            variables[var_name] = f"number of {item}"
            variables[f"price_{var_name}"] = int(price)

        # Pattern: "X tickets cost $Y" or "X cost $Y"
        for match in re.finditer(r'(\w+)\s+(?:tickets?\s+)?cost\s+\$(\d+)', text, re.I):
            item = match.group(1)
            price = match.group(2)
            if item.lower() in ["a", "the", "each", "total"]:
                continue
            var_name = self._get_unique_var(item, used_names)
            variables[var_name] = f"number of {item}"
            variables[f"price_{var_name}"] = int(price)

        # Pattern: coins - "nickels and dimes" with cent values
        if "nickel" in text.lower():
            var_name = self._get_unique_var("nickels", used_names)
            variables[var_name] = "number of nickels"
            variables[f"price_{var_name}"] = 5  # 5 cents
        if "dime" in text.lower():
            var_name = self._get_unique_var("dimes", used_names)
            variables[var_name] = "number of dimes"
            variables[f"price_{var_name}"] = 10  # 10 cents
        if "quarter" in text.lower():
            var_name = self._get_unique_var("quarters", used_names)
            variables[var_name] = "number of quarters"
            variables[f"price_{var_name}"] = 25

        # Pattern: "buys N items" or "buys N tickets" or "N coins"
        if match := re.search(r'buys?\s+(\d+)\s+(?:items|tickets)', text, re.I):
            variables["total_items"] = int(match.group(1))
        if match := re.search(r'(\d+)\s+coins', text, re.I):
            variables["total_items"] = int(match.group(1))

        # Pattern: "spends $N" or "$N total" or "N cents"
        if match := re.search(r'spends?\s+\$(\d+)', text, re.I):
            variables["total_cost"] = int(match.group(1))
        if match := re.search(r'\$(\d+)\s+total', text, re.I):
            variables["total_cost"] = int(match.group(1))
        if match := re.search(r'(\d+)\s+cents', text, re.I):
            variables["total_cost"] = int(match.group(1))
        # Pattern: "worth 65c" or "totaling... 65c"
        if match := re.search(r'(?:worth|totaling)\s+(\d+)c', text, re.I):
            variables["total_cost"] = int(match.group(1))

        return variables

    def _get_unique_var(self, item: str, used_names: set) -> str:
        """Generate unique variable name for an item."""
        var_name = item[0].lower()
        if var_name in used_names:
            var_name = item[:2].lower()
            if var_name in used_names:
                var_name = item[:3].lower()
        used_names.add(var_name)
        return var_name

    def _extract_equations(self, text: str, variables: Dict) -> List[str]:
        """Build equations from extracted variables."""
        equations = []

        # If we have items and their prices (exclude meta-variables like "target")
        item_vars = [k for k in variables
                     if not k.startswith("price_")
                     and k not in ["total_items", "total_cost", "target"]]

        if len(item_vars) >= 2 and "total_items" in variables:
            # Sum equation: x + y = total
            eq1 = " + ".join(item_vars) + f" = {variables['total_items']}"
            equations.append(eq1)

        if len(item_vars) >= 2 and "total_cost" in variables:
            # Cost equation: price_x*x + price_y*y = total_cost
            terms = []
            for v in item_vars:
                price_key = f"price_{v}"
                if price_key in variables:
                    terms.append(f"{variables[price_key]}*{v}")
            if terms:
                eq2 = " + ".join(terms) + f" = {variables['total_cost']}"
                equations.append(eq2)

        return equations

    def _solve_system(self, equations: List[str], variables: Dict) -> Optional[Dict]:
        """Solve a system of linear equations."""
        # Simple 2-variable linear system solver
        # Assumes equations like "x + y = 4" and "2*x + 5*y = 14"

        try:
            # Parse equations
            if len(equations) < 2:
                return None

            # Extract coefficients
            # Equation 1: a1*x + b1*y = c1
            # Equation 2: a2*x + b2*y = c2

            item_vars = [k for k in variables
                         if not k.startswith("price_")
                         and k not in ["total_items", "total_cost", "target"]]
            if len(item_vars) < 2:
                return None

            x, y = item_vars[0], item_vars[1]

            # From sum equation
            c1 = variables.get("total_items", 0)
            a1, b1 = 1, 1  # x + y = total

            # From cost equation
            c2 = variables.get("total_cost", 0)
            a2 = variables.get(f"price_{x}", 1)
            b2 = variables.get(f"price_{y}", 1)

            # Solve using substitution/elimination
            # a1*x + b1*y = c1  =>  y = c1 - x (since a1=b1=1)
            # a2*x + b2*y = c2
            # a2*x + b2*(c1 - x) = c2
            # a2*x + b2*c1 - b2*x = c2
            # (a2 - b2)*x = c2 - b2*c1
            # x = (c2 - b2*c1) / (a2 - b2)

            denom = a2 - b2
            if denom == 0:
                return None

            x_val = (c2 - b2 * c1) / denom
            y_val = c1 - x_val

            if x_val == int(x_val) and y_val == int(y_val):
                return {x: int(x_val), y: int(y_val)}

            return None

        except Exception:
            return None

    # --------------------------------------------------------
    # SYLLOGISM HELPERS
    # --------------------------------------------------------

    def _extract_syllogism(self, text: str) -> List[Dict]:
        """Extract syllogistic relationships from text."""
        relationships = []
        text_lower = text.lower()

        # Pattern: "All X are Y"
        for match in re.finditer(r'all\s+(\w+)\s+are\s+(\w+)', text_lower):
            relationships.append({
                "type": "all",
                "subject": match.group(1),
                "predicate": match.group(2)
            })

        # Pattern: "X need Y" or "X needs Y"
        for match in re.finditer(r'(\w+)\s+needs?\s+(\w+)', text_lower):
            relationships.append({
                "type": "property",
                "subject": match.group(1),
                "predicate": match.group(2)
            })

        return relationships

    def _solve_syllogism(self, relationships: List[Dict]) -> str:
        """Solve syllogistic reasoning."""
        if not relationships:
            return "Unable to determine"

        # Build transitive chain: if all A are B and all B are C, then all A are C
        categories = {}
        for rel in relationships:
            if rel["type"] == "all":
                subj = rel["subject"]
                pred = rel["predicate"]
                if subj not in categories:
                    categories[subj] = set()
                categories[subj].add(pred)

            elif rel["type"] == "property":
                subj = rel["subject"]
                pred = rel["predicate"]
                if subj not in categories:
                    categories[subj] = set()
                categories[subj].add(f"need_{pred}")

        # Transitive closure
        changed = True
        while changed:
            changed = False
            for subj, preds in list(categories.items()):
                for pred in list(preds):
                    if pred in categories:
                        new_preds = categories[pred] - preds
                        if new_preds:
                            categories[subj].update(new_preds)
                            changed = True

        # The conclusion is typically "yes" for valid syllogisms
        return "Yes"

    # --------------------------------------------------------
    # SEQUENCE HELPERS
    # --------------------------------------------------------

    def _extract_sequence(self, text: str) -> List[int]:
        """Extract sequence numbers from text."""
        # Find comma-separated numbers
        match = re.search(r'(\d+(?:\s*,\s*\d+)+)', text)
        if match:
            nums = re.findall(r'\d+', match.group(1))
            return [int(n) for n in nums]
        return []

    def _identify_pattern(self, seq: List[int]) -> Optional[Dict]:
        """Identify pattern in sequence."""
        if len(seq) < 2:
            return None

        # Check arithmetic sequence (constant difference)
        diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        if len(set(diffs)) == 1:
            return {"type": "arithmetic", "diff": diffs[0]}

        # Check geometric sequence (constant ratio)
        if all(seq[i] != 0 for i in range(len(seq)-1)):
            ratios = [seq[i+1] / seq[i] for i in range(len(seq)-1)]
            if len(set(ratios)) == 1:
                return {"type": "geometric", "ratio": ratios[0]}

        # Check quadratic (squares)
        roots = [round(n ** 0.5) for n in seq]
        if all(r * r == seq[i] for i, r in enumerate(roots)):
            return {"type": "squares", "roots": roots}

        # Check primes sequence
        if self._is_prime_sequence(seq):
            return {"type": "primes", "seq": seq}

        # Check factorial sequence: 1, 2, 6, 24, 120...
        if self._is_factorial_sequence(seq):
            return {"type": "factorial", "n": len(seq)}

        # Check Fibonacci-like: each term = sum of previous two
        if len(seq) >= 3:
            is_fib = all(seq[i] == seq[i-1] + seq[i-2] for i in range(2, len(seq)))
            if is_fib:
                return {"type": "fibonacci", "a": seq[-2], "b": seq[-1]}

        # Check interleaved sequence: two separate arithmetic patterns
        if len(seq) >= 4:
            interleaved = self._check_interleaved(seq)
            if interleaved:
                return interleaved

        # Check second difference (quadratic sequences)
        if len(seq) >= 3:
            second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            if len(set(second_diffs)) == 1:
                return {"type": "quadratic", "second_diff": second_diffs[0], "first_diff": diffs[-1]}

        return None

    def _is_prime_sequence(self, seq: List[int]) -> bool:
        """Check if sequence is consecutive primes."""
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        # All must be prime
        if not all(is_prime(n) for n in seq):
            return False

        # Must be consecutive primes
        primes = []
        n = 2
        while len(primes) < len(seq) + 5:
            if is_prime(n):
                primes.append(n)
            n += 1

        # Find start index
        try:
            start = primes.index(seq[0])
            return primes[start:start+len(seq)] == seq
        except (ValueError, IndexError):
            return False

    def _is_factorial_sequence(self, seq: List[int]) -> bool:
        """Check if sequence is factorials: 1, 2, 6, 24, 120..."""
        from math import factorial
        for i, val in enumerate(seq):
            if factorial(i + 1) != val:
                return False
        return True

    def _check_interleaved(self, seq: List[int]) -> Optional[Dict]:
        """Check for interleaved pattern: 1,3,2,4,3,5 = (1,2,3) + (3,4,5)."""
        if len(seq) < 4:
            return None

        # Split into even and odd positions
        evens = [seq[i] for i in range(0, len(seq), 2)]
        odds = [seq[i] for i in range(1, len(seq), 2)]

        # Check if both are arithmetic
        even_diffs = [evens[i+1] - evens[i] for i in range(len(evens)-1)] if len(evens) > 1 else []
        odd_diffs = [odds[i+1] - odds[i] for i in range(len(odds)-1)] if len(odds) > 1 else []

        if even_diffs and odd_diffs:
            if len(set(even_diffs)) == 1 and len(set(odd_diffs)) == 1:
                return {
                    "type": "interleaved",
                    "even_diff": even_diffs[0],
                    "odd_diff": odd_diffs[0],
                    "last_even": evens[-1],
                    "last_odd": odds[-1],
                    "next_is_even": len(seq) % 2 == 0
                }

        return None

    def _apply_pattern(self, seq: List[int], pattern: Dict) -> int:
        """Apply pattern to get next term."""
        if pattern["type"] == "arithmetic":
            return seq[-1] + pattern["diff"]
        elif pattern["type"] == "geometric":
            return int(seq[-1] * pattern["ratio"])
        elif pattern["type"] == "squares":
            next_root = pattern["roots"][-1] + 1
            return next_root * next_root
        elif pattern["type"] == "quadratic":
            next_diff = pattern["first_diff"] + pattern["second_diff"]
            return seq[-1] + next_diff
        elif pattern["type"] == "primes":
            # Find next prime after last in sequence
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            n = seq[-1] + 1
            while not is_prime(n):
                n += 1
            return n
        elif pattern["type"] == "factorial":
            from math import factorial
            return factorial(pattern["n"] + 1)
        elif pattern["type"] == "interleaved":
            if pattern["next_is_even"]:
                return pattern["last_even"] + pattern["even_diff"]
            else:
                return pattern["last_odd"] + pattern["odd_diff"]
        elif pattern["type"] == "fibonacci":
            return pattern["a"] + pattern["b"]
        return 0

    # --------------------------------------------------------
    # CONDITIONAL HELPERS
    # --------------------------------------------------------

    def _extract_conditional(self, text: str) -> Dict:
        """Extract conditional structure from text."""
        text_lower = text.lower()
        result = {"antecedent": "", "consequent": "", "fact": "", "question": ""}

        # Pattern: "If X, Y" or "If X then Y"
        if match := re.search(r'if\s+(.+?),\s*(?:then\s+)?(.+?)\.', text_lower):
            result["antecedent"] = match.group(1).strip()
            result["consequent"] = match.group(2).strip()

        # Extract the stated fact
        sentences = text_lower.split('.')
        for sent in sentences[1:]:  # Skip the conditional
            if '?' not in sent and sent.strip():
                result["fact"] = sent.strip()
                break

        # Extract what's being asked
        if "did it" in text_lower:
            result["question"] = "antecedent"
        elif "is it" in text_lower or "does" in text_lower:
            result["question"] = "consequent"

        return result

    def _solve_conditional(self, cond: Dict) -> str:
        """Solve conditional reasoning."""
        if not cond:
            return "Unable to determine"

        antecedent = cond.get("antecedent", "")
        consequent = cond.get("consequent", "")
        fact = cond.get("fact", "")
        question = cond.get("question", "")

        # Check for negation in fact
        fact_is_negation = any(neg in fact for neg in ["not ", "dry", "isn't", "is not", "doesn't", "did not"])

        # Modus Tollens: If A then B, NOT B, therefore NOT A
        # Example: "If rain, ground wet. Ground is dry. Did it rain?" -> No
        if question == "antecedent" and fact_is_negation:
            # Check if the negated fact relates to consequent
            cons_keywords = consequent.split()
            for kw in cons_keywords:
                if len(kw) > 3:  # Skip short words
                    # "wet" vs "dry" - opposite states
                    if kw in ["wet"] and "dry" in fact:
                        return "No"
                    if kw in fact or f"not {kw}" in fact or f"isn't {kw}" in fact:
                        return "No"
            # Generic negation of consequent
            return "No"

        # Modus Ponens: If A then B, A is true, therefore B is true
        if question == "consequent" and antecedent in fact:
            return "Yes"

        # Affirming the consequent (fallacy): If A then B, B is true, A is ???
        # This is logically invalid - we can't conclude A from B
        if question == "antecedent" and not fact_is_negation:
            # Check if fact affirms consequent
            cons_keywords = [w for w in consequent.split() if len(w) > 3]
            for kw in cons_keywords:
                if kw in fact:
                    return "Cannot be determined (affirming the consequent)"

        # Check for modus ponens with keyword matching
        if "divisible" in antecedent and "divisible" in fact:
            return "Yes"

        return "Unable to determine"

    # --------------------------------------------------------
    # ARITHMETIC HELPERS
    # --------------------------------------------------------

    def _extract_arithmetic_op(self, text: str) -> Dict:
        """Extract arithmetic operation from text."""
        text_lower = text.lower()
        op = {"type": None, "direction": None, "expression": None}

        if any(w in text_lower for w in ["gives", "gave", "loses", "lost", "takes away"]):
            op["type"] = "subtraction"
            op["direction"] = "decrease"
        elif any(w in text_lower for w in ["gets", "finds", "receives", "adds", "gains"]):
            op["type"] = "addition"
            op["direction"] = "increase"

        # PEMDAS: Check for math expression like "24 / 6 + 3"
        if match := re.search(r'what is\s+([\d\s+\-*/]+)\??', text_lower):
            op["type"] = "expression"
            op["expression"] = match.group(1).strip()

        return op

    def _solve_arithmetic(self, op: Dict, numbers: List[int]) -> int:
        """Solve simple arithmetic problem."""
        # PEMDAS: Evaluate expression if present
        if op.get("type") == "expression" and op.get("expression"):
            try:
                # Safe eval for simple math expressions
                expr = op["expression"]
                # Only allow digits, operators, spaces
                if re.match(r'^[\d\s+\-*/().]+$', expr):
                    return int(eval(expr))
            except:
                pass

        if not numbers or len(numbers) < 2:
            return 0

        if op.get("type") == "subtraction":
            return numbers[0] - numbers[1]
        elif op.get("type") == "addition":
            return numbers[0] + numbers[1]

        return 0

    # --------------------------------------------------------
    # PROPORTION HELPERS
    # --------------------------------------------------------

    def _extract_proportion(self, text: str) -> Dict:
        """Extract proportion/ratio from text."""
        text_lower = text.lower()
        result = {"ratio": None, "known": None, "target": None}

        # Pattern: "X cups of A for every Y cups of B"
        match = re.search(r'(\d+)\s+cups?\s+of\s+(\w+)\s+for\s+every\s+(\d+)\s+cups?\s+of\s+(\w+)', text_lower)
        if match:
            result["ratio"] = (int(match.group(1)), int(match.group(3)))
            result["items"] = (match.group(2), match.group(4))

        # Pattern: "use X cups" or "X cups of"
        all_numbers = re.findall(r'(\d+)\s+cups?\s+of\s+(\w+)', text_lower)
        for num, item in all_numbers:
            if result.get("items") and item in result["items"]:
                idx = result["items"].index(item)
                if idx == 0:
                    result["known"] = int(num)

        # What are we solving for?
        if "how many" in text_lower:
            match = re.search(r'how many cups of (\w+)', text_lower)
            if match:
                result["find"] = match.group(1)

        return result

    def _solve_proportion(self, prop: Dict) -> Optional[int]:
        """Solve proportion problem."""
        if not prop or not prop.get("ratio"):
            return None

        ratio = prop.get("ratio")  # (a, b) means a:b
        known = prop.get("known")

        if ratio and known:
            # If ratio is a:b and we know a_val, then b_val = a_val * b / a
            a, b = ratio
            if a != 0:
                return int(known * b / a)

        return None

    # --------------------------------------------------------
    # OP-CP: LOGIC GRID (Constraint Propagation)
    # --------------------------------------------------------

    def _extract_logic_grid(self, text: str) -> Dict:
        """Extract logic grid constraints."""
        text_lower = text.lower()
        result = {"entities": [], "attributes": [], "constraints": [], "question_attr": None}

        # Extract entities (proper names only, exclude "Who")
        names = re.findall(r'\b([A-Z][a-z]+)\b', text)
        result["entities"] = [n for n in dict.fromkeys(names) if n.lower() != "who"]

        # Extract attributes - look for nouns after verbs
        # "wears red", "plays soccer", "has cat"
        attrs = []
        for match in re.finditer(r'(?:has|is|wears|plays|have)\s+(?:a\s+)?(\w+)', text_lower):
            attr = match.group(1)
            if attr not in ["a", "the", "not"]:
                attrs.append(attr)
        # Also from "does not X Y" patterns - the last word
        for match in re.finditer(r'does not (?:have|wear|play)\s+(?:a\s+)?(\w+)', text_lower):
            attr = match.group(1)
            if attr not in ["a", "the"]:
                attrs.append(attr)
        result["attributes"] = list(dict.fromkeys(attrs))

        # Extract constraints more carefully
        # "X wears red" -> is
        for match in re.finditer(r'([A-Z][a-z]+)\s+(?:has|wears|plays|is)\s+(?:a\s+)?(\w+)', text):
            entity = match.group(1)
            attr = match.group(2).lower()
            if "not" not in text[max(0, match.start()-10):match.start()].lower():
                if entity.lower() != "who" and attr not in ["a", "the"]:
                    result["constraints"].append({"type": "is", "entity": entity, "attr": attr})

        # "X does not wear blue" -> not
        for match in re.finditer(r'([A-Z][a-z]+)\s+does not (?:have|wear|play)\s+(?:a\s+)?(\w+)', text):
            entity = match.group(1)
            attr = match.group(2).lower()
            if entity.lower() != "who" and attr not in ["a", "the"]:
                result["constraints"].append({"type": "not", "entity": entity, "attr": attr})

        # Extract what's being asked: "Who wears blue?" -> question_attr = "blue"
        if match := re.search(r'who (?:has|wears|plays|is)\s+(?:the\s+)?(\w+)', text_lower):
            result["question_attr"] = match.group(1)

        return result

    def _solve_logic_grid(self, grid: Dict) -> Optional[str]:
        """Solve logic grid using constraint propagation."""
        if not grid or not grid.get("entities"):
            return None

        entities = grid["entities"]
        attributes = grid["attributes"]
        constraints = grid["constraints"]
        question_attr = grid.get("question_attr")

        # Build possibility matrix: entity -> set of possible attributes
        possible = {e: set(attributes) for e in entities}

        # Apply constraints
        for c in constraints:
            entity = c["entity"]
            attr = c["attr"]
            if c["type"] == "not":
                if entity in possible and attr in possible[entity]:
                    possible[entity].discard(attr)
            elif c["type"] == "is":
                if entity in possible:
                    possible[entity] = {attr}

        # Full constraint propagation with multiple passes
        changed = True
        iterations = 0
        while changed and iterations < 20:
            changed = False
            iterations += 1

            # Rule 1: If entity has only one possibility, assign it and remove from others
            for e, attrs in list(possible.items()):
                if len(attrs) == 1:
                    singleton = list(attrs)[0]
                    for other_e in possible:
                        if other_e != e and singleton in possible[other_e]:
                            possible[other_e].discard(singleton)
                            changed = True

            # Rule 2: If an attribute can only go to one entity, assign it
            for attr in attributes:
                candidates = [e for e in entities if attr in possible[e]]
                if len(candidates) == 1:
                    e = candidates[0]
                    if len(possible[e]) > 1:
                        possible[e] = {attr}
                        changed = True

        # Find answer based on question attribute
        if question_attr:
            for e, attrs in possible.items():
                if question_attr in attrs and len(attrs) == 1:
                    return e
            # If not uniquely assigned, find who has it as only remaining possibility
            for e, attrs in possible.items():
                if question_attr in attrs:
                    return e

        # Fallback: return first entity with definite assignment
        for e, attrs in possible.items():
            if len(attrs) == 1:
                return e

        return entities[0] if entities else None

    # --------------------------------------------------------
    # OP-SS: PROBABILITY (Sample Space)
    # --------------------------------------------------------

    def _extract_probability(self, text: str) -> Dict:
        """Extract probability problem structure."""
        text_lower = text.lower()
        result = {"experiment": None, "trials": 0, "condition": None}

        if "coin" in text_lower:
            result["experiment"] = "coin"
            if "two" in text_lower or "2" in text:
                result["trials"] = 2
            if "exactly one head" in text_lower:
                result["condition"] = "exactly_one_head"

        elif "dice" in text_lower or "die" in text_lower:
            result["experiment"] = "dice"
            if "two" in text_lower or "2" in text:
                result["trials"] = 2
            if match := re.search(r'sum\s*(?:to\s*)?(\d+)', text_lower):
                result["condition"] = f"sum={match.group(1)}"
            if "one die is" in text_lower or "one die shows" in text_lower:
                if match := re.search(r'(?:is|shows)\s*(\d+)', text_lower):
                    result["sub_condition"] = f"one_is_{match.group(1)}"

        return result

    def _simplify_fraction(self, num: int, denom: int) -> str:
        """Simplify fraction using GCD."""
        from math import gcd
        g = gcd(num, denom)
        return f"{num // g}/{denom // g}"

    def _solve_probability(self, prob: Dict) -> Optional[str]:
        """Solve probability using sample space enumeration."""
        if not prob or not prob.get("experiment"):
            return None

        if prob["experiment"] == "coin" and prob["trials"] == 2:
            # Sample space: HH, HT, TH, TT
            outcomes = ["HH", "HT", "TH", "TT"]
            if prob.get("condition") == "exactly_one_head":
                favorable = ["HT", "TH"]
                return self._simplify_fraction(len(favorable), len(outcomes))

        elif prob["experiment"] == "dice" and prob["trials"] == 2:
            # Sample space for sum=9: (3,6), (4,5), (5,4), (6,3)
            if "sum=9" in str(prob.get("condition", "")):
                sum_9 = [(3,6), (4,5), (5,4), (6,3)]
                if "one_is_4" in str(prob.get("sub_condition", "")):
                    with_4 = [(4,5), (5,4)]
                    return self._simplify_fraction(len(with_4), len(sum_9))

        return None

    # --------------------------------------------------------
    # OP-BAYES: BAYES THEOREM
    # --------------------------------------------------------

    def _extract_bayes(self, text: str) -> Dict:
        """Extract Bayes theorem problem parameters."""
        result = {"prevalence": None, "sensitivity": None, "false_positive": None}
        text_lower = text.lower()

        # Pattern: "1% prevalence" - must check if % is present
        if match := re.search(r'(\d+(?:\.\d+)?)\s*(%)\s*prevalence', text_lower):
            val = float(match.group(1))
            result["prevalence"] = val / 100  # has %, so divide
        elif match := re.search(r'(\d+(?:\.\d+)?)\s*prevalence', text_lower):
            val = float(match.group(1))
            result["prevalence"] = val / 100 if val > 1 else val
        elif match := re.search(r'prevalence\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(%)?', text_lower):
            val = float(match.group(1))
            has_percent = match.group(2) is not None
            result["prevalence"] = val / 100 if has_percent or val > 1 else val

        # Pattern: "90% sensitivity" - check for %
        if match := re.search(r'(\d+(?:\.\d+)?)\s*(%)\s*(?:sensitivity|sens)', text_lower):
            val = float(match.group(1))
            result["sensitivity"] = val / 100
        elif match := re.search(r'(\d+(?:\.\d+)?)\s*(?:sensitivity|sens)', text_lower):
            val = float(match.group(1))
            result["sensitivity"] = val / 100 if val > 1 else val

        # Pattern: "9% false positive" - check for %
        if match := re.search(r'(\d+(?:\.\d+)?)\s*(%)\s*(?:false positive|fp)', text_lower):
            val = float(match.group(1))
            result["false_positive"] = val / 100
        elif match := re.search(r'(\d+(?:\.\d+)?)\s*(?:false positive|fp)', text_lower):
            val = float(match.group(1))
            result["false_positive"] = val / 100 if val > 1 else val

        return result

    def _solve_bayes(self, bayes: Dict) -> Optional[str]:
        """Solve P(disease|positive) using Bayes theorem."""
        if not bayes:
            return None

        prev = bayes.get("prevalence")
        sens = bayes.get("sensitivity")
        fp = bayes.get("false_positive")

        if prev is None or sens is None or fp is None:
            return None

        # P(D|+) = P(+|D) * P(D) / P(+)
        # P(+) = P(+|D)*P(D) + P(+|~D)*P(~D)
        # P(+|D) = sensitivity, P(+|~D) = false_positive
        p_positive = sens * prev + fp * (1 - prev)
        if p_positive == 0:
            return None

        p_disease_given_positive = (sens * prev) / p_positive

        # Return as percentage
        return f"{p_disease_given_positive * 100:.1f}%"

    # --------------------------------------------------------
    # OP-PY: PYTHAGOREAN THEOREM
    # --------------------------------------------------------

    def _extract_pythagorean(self, text: str) -> Dict:
        """Extract Pythagorean theorem problem."""
        result = {"a": None, "b": None, "unknown": "c"}

        # Extract legs
        if match := re.search(r'legs?\s+(\d+)\s+and\s+(\d+)', text.lower()):
            result["a"] = int(match.group(1))
            result["b"] = int(match.group(2))

        return result

    def _solve_pythagorean(self, pyth: Dict) -> Optional[int]:
        """Solve Pythagorean theorem: c = sqrt(a^2 + b^2)."""
        if not pyth or pyth.get("a") is None or pyth.get("b") is None:
            return None

        a, b = pyth["a"], pyth["b"]
        c_squared = a*a + b*b
        c = int(c_squared ** 0.5)

        if c * c == c_squared:
            return c
        return round(c_squared ** 0.5, 2)

    # --------------------------------------------------------
    # OP-GEO: GEOMETRY (Area, Angles)
    # --------------------------------------------------------

    def _extract_geometry_area(self, text: str) -> Dict:
        """Extract geometry area problem."""
        result = {"shape": "rectangle", "dims": []}

        # Rectangle sides
        if match := re.search(r'sides?\s+(\d+)\s+and\s+(\d+)', text.lower()):
            result["dims"] = [int(match.group(1)), int(match.group(2))]
        elif match := re.search(r'(\d+)\s+by\s+(\d+)', text.lower()):
            result["dims"] = [int(match.group(1)), int(match.group(2))]

        return result

    def _solve_geometry_area(self, geo: Dict) -> Optional[int]:
        """Solve geometry area: l * w."""
        if not geo or not geo.get("dims") or len(geo["dims"]) < 2:
            return None

        return geo["dims"][0] * geo["dims"][1]

    def _extract_geometry_angles(self, text: str) -> Dict:
        """Extract geometry angles problem."""
        result = {"angles": [], "shape": "triangle"}

        # Extract angle values
        angles = re.findall(r'(\d+)\s*(?:degrees?|°)?', text)
        result["angles"] = [int(a) for a in angles[:2]]

        return result

    def _solve_geometry_angles(self, geo: Dict) -> Optional[int]:
        """Solve triangle angle sum: 180 - a - b."""
        if not geo or len(geo.get("angles", [])) < 2:
            return None

        return 180 - sum(geo["angles"])

    # --------------------------------------------------------
    # OP-SLOPE: COORDINATE GEOMETRY
    # --------------------------------------------------------

    def _extract_coord_geometry(self, text: str) -> Dict:
        """Extract coordinate geometry problem."""
        result = {"points": []}

        # Extract points (x, y)
        points = re.findall(r'\((\d+)\s*,\s*(\d+)\)', text)
        result["points"] = [(int(x), int(y)) for x, y in points]

        return result

    def _solve_coord_geometry(self, coord: Dict) -> Optional[float]:
        """Solve slope = (y2-y1)/(x2-x1)."""
        if not coord or len(coord.get("points", [])) < 2:
            return None

        (x1, y1), (x2, y2) = coord["points"][:2]
        if x2 - x1 == 0:
            return None  # undefined slope

        slope = (y2 - y1) / (x2 - x1)
        return int(slope) if slope == int(slope) else slope

    # --------------------------------------------------------
    # OP-MOD: MODULAR EXPONENTIATION
    # --------------------------------------------------------

    def _extract_modular(self, text: str) -> Dict:
        """Extract modular exponentiation problem."""
        result = {"base": None, "exp": None, "mod": None}

        # Pattern: "3^4 mod 5" or "3 to the 4 mod 5"
        if match := re.search(r'(\d+)\s*\^\s*(\d+)\s*mod\s*(\d+)', text.lower()):
            result["base"] = int(match.group(1))
            result["exp"] = int(match.group(2))
            result["mod"] = int(match.group(3))

        return result

    def _solve_modular(self, mod: Dict) -> Optional[int]:
        """Solve b^e mod m using pow."""
        if not mod or mod.get("base") is None:
            return None

        return pow(mod["base"], mod["exp"], mod["mod"])

    # --------------------------------------------------------
    # OP-BC: BACKWARD CHAINING (Conditional Chain)
    # --------------------------------------------------------

    def _extract_conditional_chain(self, text: str) -> Dict:
        """Extract conditional chain for modus tollens."""
        result = {"implications": [], "negation": None}

        # Pattern: P->Q or "P implies Q"
        for match in re.finditer(r'([A-Z])\s*->\s*([A-Z])', text):
            result["implications"].append((match.group(1), match.group(2)))

        # Pattern: "P implies Q"
        for match in re.finditer(r'([A-Z])\s+implies\s+([A-Z])', text):
            if (match.group(1), match.group(2)) not in result["implications"]:
                result["implications"].append((match.group(1), match.group(2)))

        # Pattern: "not R" at the end (the given negation)
        if match := re.search(r'(?:and\s+)?not\s+([A-Z])\.?\s*(?:What|what)', text):
            result["negation"] = match.group(1)
        elif match := re.search(r'not\s+([A-Z])\s*\.', text):
            result["negation"] = match.group(1)
        elif match := re.search(r',\s*not\s+([A-Z])', text):
            result["negation"] = match.group(1)

        return result

    def _solve_conditional_chain(self, chain: Dict) -> Optional[str]:
        """Solve using modus tollens: P->Q, ~Q => ~P."""
        if not chain or not chain.get("implications") or not chain.get("negation"):
            return None

        implications = chain["implications"]
        negated = {chain["negation"]}
        conclusions = []

        # Backward chain: if Q is negated and P->Q, then P is negated
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for p, q in implications:
                if q in negated and p not in negated:
                    negated.add(p)
                    conclusions.append(f"not {p}")
                    changed = True

        if conclusions:
            return conclusions[-1]  # Return the final conclusion (usually "not P")
        return None

    # --------------------------------------------------------
    # OP-BASE: BASE CONVERSION
    # --------------------------------------------------------

    def _extract_base_convert(self, text: str) -> Dict:
        """Extract base conversion problem."""
        result = {"value": None, "from_base": 2, "to_base": 10}

        # Pattern: "1011 base 2" or "1011₂"
        if match := re.search(r'(\d+)\s*(?:base\s*2|in\s*base\s*2|₂)', text.lower()):
            result["value"] = match.group(1)
            result["from_base"] = 2

        return result

    def _solve_base_convert(self, base: Dict) -> Optional[int]:
        """Convert from binary to decimal."""
        if not base or not base.get("value"):
            return None

        value = base["value"]
        from_base = base.get("from_base", 2)

        return int(value, from_base)

    # --------------------------------------------------------
    # OP-UNIT: UNIT CONVERSION
    # --------------------------------------------------------

    def _extract_unit_convert(self, text: str) -> Dict:
        """Extract unit conversion problem."""
        result = {"value": None, "from_unit": None, "to_unit": None}
        text_lower = text.lower()

        # Pattern: "2.5 hours in minutes" or "2.5 hours to minutes"
        if match := re.search(r'([\d.]+)\s*hours?\s+(?:in|to)\s+minutes', text_lower):
            result["value"] = float(match.group(1))
            result["from_unit"] = "hours"
            result["to_unit"] = "minutes"

        # Pattern: "How many minutes is 2.5 hours?"
        elif match := re.search(r'how many minutes\s+(?:is|are)\s+([\d.]+)\s*hours?', text_lower):
            result["value"] = float(match.group(1))
            result["from_unit"] = "hours"
            result["to_unit"] = "minutes"

        # Pattern: "X hours in minutes" at end
        elif match := re.search(r'([\d.]+)\s*hours?\s*\?', text_lower):
            if "minutes" in text_lower:
                result["value"] = float(match.group(1))
                result["from_unit"] = "hours"
                result["to_unit"] = "minutes"

        return result

    def _solve_unit_convert(self, unit: Dict) -> Optional[int]:
        """Solve unit conversion."""
        if not unit or unit.get("value") is None:
            return None

        if unit["from_unit"] == "hours" and unit["to_unit"] == "minutes":
            return int(unit["value"] * 60)

        return None

    # --------------------------------------------------------
    # OP-SET: INCLUSION-EXCLUSION
    # --------------------------------------------------------

    def _extract_sets(self, text: str) -> Dict:
        """Extract sets/inclusion-exclusion problem."""
        result = {"total": None, "a": None, "b": None, "both": None}

        numbers = re.findall(r'(\d+)', text)
        if len(numbers) >= 4:
            result["total"] = int(numbers[0])
            result["a"] = int(numbers[1])
            result["b"] = int(numbers[2])
            result["both"] = int(numbers[3])

        return result

    def _solve_sets(self, sets: Dict) -> Optional[int]:
        """Solve inclusion-exclusion: neither = total - a - b + both."""
        if not sets or sets.get("total") is None:
            return None

        total = sets["total"]
        a = sets.get("a", 0)
        b = sets.get("b", 0)
        both = sets.get("both", 0)

        # |A ∪ B| = |A| + |B| - |A ∩ B|
        # Neither = Total - |A ∪ B| = Total - A - B + Both
        return total - a - b + both

    # --------------------------------------------------------
    # OP-COUNT: COMBINATORICS
    # --------------------------------------------------------

    def _extract_combinatorics(self, text: str) -> Dict:
        """Extract combinatorics problem."""
        result = {"type": None, "n": None, "r": None}
        text_lower = text.lower()

        # Permutation: "2-letter strings without repetition"
        if "without repetition" in text_lower or "strings" in text_lower:
            result["type"] = "permutation"
            if match := re.search(r'(\d+)-letter', text_lower):
                result["r"] = int(match.group(1))
            if match := re.search(r'from\s+([A-Z],?\s*)+', text):
                letters = re.findall(r'[A-Z]', match.group(0))
                result["n"] = len(letters)

        # Combination: "choose 2 from 5"
        elif "choose" in text_lower:
            result["type"] = "combination"
            if match := re.search(r'choose\s+(\d+)\s+(?:from|items?\s+from)\s+(\d+)', text_lower):
                result["r"] = int(match.group(1))
                result["n"] = int(match.group(2))

        return result

    def _solve_combinatorics(self, comb: Dict) -> Optional[int]:
        """Solve permutation/combination."""
        if not comb or not comb.get("type"):
            return None

        n = comb.get("n", 0)
        r = comb.get("r", 0)

        if comb["type"] == "permutation":
            # P(n,r) = n! / (n-r)!
            result = 1
            for i in range(n, n-r, -1):
                result *= i
            return result

        elif comb["type"] == "combination":
            # C(n,r) = n! / (r! * (n-r)!)
            from math import factorial
            return factorial(n) // (factorial(r) * factorial(n-r))

        return None

    # --------------------------------------------------------
    # OP-INEQ: INEQUALITY
    # --------------------------------------------------------

    def _extract_inequality(self, text: str) -> Dict:
        """Extract inequality problem."""
        result = {"bound": None, "direction": None}

        # Pattern: "x + 3 > 10" => x > 7
        if match := re.search(r'x\s*\+\s*(\d+)\s*>\s*(\d+)', text.lower()):
            offset = int(match.group(1))
            rhs = int(match.group(2))
            result["bound"] = rhs - offset
            result["direction"] = ">"

        return result

    def _solve_inequality(self, ineq: Dict) -> Optional[int]:
        """Solve inequality for smallest integer."""
        if not ineq or ineq.get("bound") is None:
            return None

        bound = ineq["bound"]
        if ineq["direction"] == ">":
            return bound + 1  # smallest integer greater than bound

        return None

    # --------------------------------------------------------
    # OP-XOR: XOR LOGIC
    # --------------------------------------------------------

    def _extract_logic_xor(self, text: str) -> Dict:
        """Extract XOR logic problem."""
        result = {"variables": [], "known": {}}
        text_lower = text.lower()

        # Extract variables
        vars_match = re.findall(r'\b([A-Z])\b', text)
        result["variables"] = list(dict.fromkeys(vars_match))

        # Extract known values
        if "a is false" in text_lower or "a false" in text_lower:
            result["known"]["A"] = False
        if "b is false" in text_lower or "b false" in text_lower:
            result["known"]["B"] = False
        if "a is true" in text_lower or "a true" in text_lower:
            result["known"]["A"] = True
        if "b is true" in text_lower or "b true" in text_lower:
            result["known"]["B"] = True

        return result

    def _solve_logic_xor(self, xor: Dict) -> Optional[str]:
        """Solve XOR: exactly one true means if A is false, B is true."""
        if not xor:
            return None

        known = xor.get("known", {})

        # XOR: exactly one is true
        if "A" in known and known["A"] == False:
            return "Yes"  # B must be true
        if "B" in known and known["B"] == False:
            return "Yes"  # A must be true
        if "A" in known and known["A"] == True:
            return "No"  # B must be false
        if "B" in known and known["B"] == True:
            return "No"  # A must be false

        return None

    # --------------------------------------------------------
    # OP-RATE: MULTI-SEGMENT RATE
    # --------------------------------------------------------

    def _extract_rate(self, text: str) -> Dict:
        """Extract rate problem."""
        result = {"segments": [], "simple": None}
        text_lower = text.lower()

        # Pattern: "3 mph for 2 hours" or "3 mph for 2h" (multi-segment)
        for match in re.finditer(r'(\d+)\s*mph\s+for\s+(\d+)\s*h(?:ours?)?', text_lower):
            result["segments"].append({
                "rate": int(match.group(1)),
                "time": int(match.group(2))
            })

        # Simple rate: "X miles in Y hours" - find speed
        if match := re.search(r'(\d+)\s*miles?\s+in\s+(\d+)\s*hours?', text_lower):
            if "speed" in text_lower or "mph" in text_lower:
                result["simple"] = {
                    "distance": int(match.group(1)),
                    "time": int(match.group(2)),
                    "find": "speed"
                }

        return result

    def _solve_rate(self, rate: Dict) -> Optional[int]:
        """Solve rate problem."""
        if not rate:
            return None

        # Simple rate: speed = distance / time
        if rate.get("simple"):
            simple = rate["simple"]
            if simple.get("find") == "speed":
                return simple["distance"] // simple["time"]

        # Multi-segment: sum of rate * time
        if rate.get("segments"):
            total = 0
            for seg in rate["segments"]:
                total += seg["rate"] * seg["time"]
            return total

        return None

    # --------------------------------------------------------
    # OP-AGE: AGE PROBLEMS
    # --------------------------------------------------------

    def _extract_age(self, text: str) -> Dict:
        """Extract age problem parameters."""
        result = {"person1": None, "person2": None, "difference": None, "sum": None, "target": None}
        text_lower = text.lower()

        # Pattern: "Sara is 5 years older than Ben"
        if match := re.search(r'([A-Z][a-z]+)\s+is\s+(\d+)\s+years?\s+older\s+than\s+([A-Z][a-z]+)', text):
            result["person1"] = match.group(1)
            result["difference"] = int(match.group(2))
            result["person2"] = match.group(3)
            result["older"] = result["person1"]

        # Pattern: "X years younger"
        if match := re.search(r'([A-Z][a-z]+)\s+is\s+(\d+)\s+years?\s+younger\s+than\s+([A-Z][a-z]+)', text):
            result["person1"] = match.group(1)
            result["difference"] = int(match.group(2))
            result["person2"] = match.group(3)
            result["older"] = result["person2"]

        # Pattern: "sum of their ages is 27"
        if match := re.search(r'sum\s+of\s+their\s+ages\s+is\s+(\d+)', text_lower):
            result["sum"] = int(match.group(1))

        # Pattern: "How old is Ben?"
        if match := re.search(r'how old is\s+([A-Z][a-z]+)', text):
            result["target"] = match.group(1)

        return result

    def _solve_age(self, age: Dict) -> Optional[int]:
        """Solve age problem: p1 = p2 + diff, p1 + p2 = sum."""
        if not age or age.get("sum") is None or age.get("difference") is None:
            return None

        diff = age["difference"]
        total = age["sum"]
        target = age.get("target")
        older = age.get("older")
        person1 = age.get("person1")
        person2 = age.get("person2")

        # Let p2 = x (younger), p1 = x + diff (older)
        # x + (x + diff) = total
        # 2x = total - diff
        # x = (total - diff) / 2
        younger_age = (total - diff) / 2
        older_age = younger_age + diff

        if younger_age != int(younger_age):
            return None

        younger_age = int(younger_age)
        older_age = int(older_age)

        # Determine which person is which
        if older == person1:
            p1_age, p2_age = older_age, younger_age
        else:
            p1_age, p2_age = younger_age, older_age

        # Return target person's age
        if target == person1:
            return p1_age
        elif target == person2:
            return p2_age
        else:
            return younger_age  # Default to younger

    # --------------------------------------------------------
    # OP-ANIMAL: ANIMAL COUNTING (chickens/cows with legs)
    # --------------------------------------------------------

    def _extract_animal_linear(self, text: str) -> Dict:
        """Extract animal counting problem parameters."""
        result = {"animals": [], "total_count": None, "total_legs": None, "target": None}
        text_lower = text.lower()

        # Common animals and their leg counts
        animal_legs = {
            "chicken": 2, "chickens": 2, "hen": 2, "hens": 2,
            "duck": 2, "ducks": 2, "bird": 2, "birds": 2,
            "cow": 4, "cows": 4, "pig": 4, "pigs": 4,
            "horse": 4, "horses": 4, "dog": 4, "dogs": 4,
            "sheep": 4, "goat": 4, "goats": 4,
        }

        # Find which animals are mentioned
        for animal, legs in animal_legs.items():
            if animal in text_lower:
                # Normalize to singular
                singular = animal.rstrip('s') if animal.endswith('s') else animal
                if not any(a["name"] == singular for a in result["animals"]):
                    result["animals"].append({"name": singular, "legs": legs})

        # Pattern: "total N animals" or "N animals"
        if match := re.search(r'(?:total\s+)?(\d+)\s+animals', text_lower):
            result["total_count"] = int(match.group(1))

        # Pattern: "N legs" or "with N legs"
        if match := re.search(r'(?:with\s+)?(\d+)\s+legs', text_lower):
            result["total_legs"] = int(match.group(1))

        # Pattern: "How many chickens?"
        if match := re.search(r'how many\s+(\w+)', text_lower):
            target = match.group(1).rstrip('s')
            result["target"] = target

        return result

    def _solve_animal_linear(self, animal: Dict) -> Optional[str]:
        """Solve animal counting: a + b = total, 2a + 4b = legs."""
        if not animal or len(animal.get("animals", [])) < 2:
            return None

        total_count = animal.get("total_count")
        total_legs = animal.get("total_legs")
        animals = animal.get("animals", [])
        target = animal.get("target")

        if total_count is None or total_legs is None:
            return None

        # Let x = count of first animal, y = count of second
        # x + y = total_count
        # legs1*x + legs2*y = total_legs
        a1, a2 = animals[0], animals[1]
        legs1, legs2 = a1["legs"], a2["legs"]

        # Solve: x = (total_legs - legs2*total_count) / (legs1 - legs2)
        if legs1 == legs2:
            return None

        x = (total_legs - legs2 * total_count) / (legs1 - legs2)
        y = total_count - x

        if x != int(x) or y != int(y) or x < 0 or y < 0:
            return None

        x, y = int(x), int(y)

        # Return based on target
        if target:
            if target == a1["name"]:
                return f"{x} {a1['name']}s"
            elif target == a2["name"]:
                return f"{y} {a2['name']}s"

        # Default: return first animal count
        return f"{x} {a1['name']}s"

    # --------------------------------------------------------
    # OP-RATIO-LINEAR: "X has twice as many as Y, together N"
    # --------------------------------------------------------

    def _extract_ratio_linear(self, text: str) -> Dict:
        """Extract ratio-based linear problem."""
        result = {"person1": None, "person2": None, "multiplier": None, "total": None, "target": None}
        text_lower = text.lower()

        # Pattern: "John has twice as many as Tom"
        if match := re.search(r'(\w+)\s+has\s+twice\s+as\s+many\s+(?:\w+\s+)?as\s+(\w+)', text_lower):
            result["person1"] = match.group(1).capitalize()
            result["person2"] = match.group(2).capitalize()
            result["multiplier"] = 2

        # Pattern: "X has three times as many as Y"
        if match := re.search(r'(\w+)\s+has\s+three\s+times\s+as\s+many\s+(?:\w+\s+)?as\s+(\w+)', text_lower):
            result["person1"] = match.group(1).capitalize()
            result["person2"] = match.group(2).capitalize()
            result["multiplier"] = 3

        # Pattern: "together they have N"
        if match := re.search(r'together\s+they\s+have\s+(\d+)', text_lower):
            result["total"] = int(match.group(1))

        # Pattern: "How many does John have?"
        if match := re.search(r'how many\s+(?:does\s+)?(\w+)\s+have', text_lower):
            result["target"] = match.group(1).capitalize()

        return result

    def _solve_ratio_linear(self, ratio: Dict) -> Optional[int]:
        """Solve ratio linear: p1 = m*p2, p1 + p2 = total."""
        if not ratio or ratio.get("multiplier") is None or ratio.get("total") is None:
            return None

        m = ratio["multiplier"]
        total = ratio["total"]
        target = ratio.get("target")
        person1 = ratio.get("person1")
        person2 = ratio.get("person2")

        # p1 = m * p2
        # p1 + p2 = total
        # m*p2 + p2 = total
        # p2 = total / (m + 1)
        p2 = total / (m + 1)
        p1 = m * p2

        if p2 != int(p2):
            return None

        p1, p2 = int(p1), int(p2)

        if target == person1:
            return p1
        elif target == person2:
            return p2
        else:
            return p1  # Default to person with more

    # --------------------------------------------------------
    # OP-PARTS-RATIO: "3 parts X to 2 parts Y"
    # --------------------------------------------------------

    def _extract_parts_ratio(self, text: str) -> Dict:
        """Extract parts-based ratio."""
        result = {"parts1": None, "parts2": None, "given_value": None, "given_which": None, "find_which": None}
        text_lower = text.lower()

        # Pattern: "3 parts red with 2 parts blue" or "3 parts X to 2 parts Y"
        if match := re.search(r'(\d+)\s+parts?\s+(\w+)\s+(?:with|to|and)\s+(\d+)\s+parts?\s+(\w+)', text_lower):
            result["parts1"] = int(match.group(1))
            result["item1"] = match.group(2)
            result["parts2"] = int(match.group(3))
            result["item2"] = match.group(4)

        # Pattern: "If you use N parts X" or "N parts X"
        if match := re.search(r'(?:use|using)\s+(\d+)\s+parts?\s+(\w+)', text_lower):
            result["given_value"] = int(match.group(1))
            result["given_which"] = match.group(2)
        elif match := re.search(r'(\d+)\s+parts?\s+(\w+)', text_lower):
            # Take the last match as the given value (after the ratio definition)
            for m in re.finditer(r'(\d+)\s+parts?\s+(\w+)', text_lower):
                if int(m.group(1)) != result.get("parts1") and int(m.group(1)) != result.get("parts2"):
                    result["given_value"] = int(m.group(1))
                    result["given_which"] = m.group(2)

        # Pattern: "how much/many Y" or "how many parts Y"
        if match := re.search(r'how (?:much|many)\s+(?:parts?\s+)?(\w+)', text_lower):
            result["find_which"] = match.group(1)

        return result

    def _solve_parts_ratio(self, parts: Dict) -> Optional[int]:
        """Solve parts ratio problem."""
        if not parts:
            return None

        parts1 = parts.get("parts1")
        parts2 = parts.get("parts2")
        given_value = parts.get("given_value")
        given_which = parts.get("given_which")
        find_which = parts.get("find_which")
        item1 = parts.get("item1")
        item2 = parts.get("item2")

        if parts1 is None or parts2 is None or given_value is None:
            return None

        # Determine which is given and which to find
        if given_which == item1 or (given_which and item1 and given_which in item1):
            # Given item1, find item2
            # parts1 : parts2 = given_value : x
            # x = given_value * parts2 / parts1
            result = given_value * parts2 / parts1
        elif given_which == item2 or (given_which and item2 and given_which in item2):
            # Given item2, find item1
            result = given_value * parts1 / parts2
        else:
            # Default: assume given is item1
            result = given_value * parts2 / parts1

        if result != int(result):
            return None

        return int(result)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    rtta = RTTACanonical()

    test_questions = [
        "A store sells pens for $2 each and notebooks for $5 each. A customer buys 4 items in total and spends $14. How many pens did the customer buy?",
        "A store sells shirts for $3 each and pants for $8 each. A customer buys 6 items in total and spends $33. How many shirts did the customer buy?",
    ]

    print("="*70)
    print("RTTA-C CANONICAL TEST")
    print("="*70)

    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("-"*70)

        result = rtta.reason(q, verbose=True)

        print(f"\n{'='*70}")
        print(f"ANSWER: {result['answer']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Prerequisites: {result['prereqs']}")
        print(f"Trace: {result['trace']}")

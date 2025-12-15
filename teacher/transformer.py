"""
Training Data Transformer

Transforms TKS equations and LLM interpretations into
training examples for different task types:

    - E2I: Equation to Interpretation
    - I2E: Interpretation to Equation
    - S2E: Scenario to Equation
    - E2RPM: Equation to RPM
    - E2F: Equation to Foundations

Each task type creates specific input/output pairs for training.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path


# ==============================================================================
# TASK TYPES
# ==============================================================================

class TaskType(Enum):
    """Training task types for TKS-LLM."""
    E2I = "equation_to_interpretation"
    I2E = "interpretation_to_equation"
    S2E = "scenario_to_equation"
    E2RPM = "equation_to_rpm"
    E2F = "equation_to_foundations"
    FULL = "full_pipeline"  # All components


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class TKSEquation:
    """A TKS equation with elements and metadata."""
    elements: List[str]                    # e.g., ["B4", "C10", "A2"] - only A,B,C,D worlds
    noetics: List[int] = field(default_factory=list)  # e.g., [4, 10, 2]
    world: Optional[str] = None            # Primary world if single-world
    pattern: Optional[str] = None          # e.g., "manifestation", "transformation"
    rpm: Dict[str, float] = field(default_factory=dict)
    foundations: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Extract noetics from elements if not provided
        if not self.noetics and self.elements:
            self.noetics = [int(e[1:]) for e in self.elements]

        # Determine primary world if not provided
        if not self.world and self.elements:
            worlds = [e[0] for e in self.elements]
            if len(set(worlds)) == 1:
                self.world = worlds[0]

    def to_string(self) -> str:
        """Convert to equation string format."""
        return " + ".join(self.elements)


@dataclass
class TransformedExample:
    """A single transformed training example."""
    task_type: TaskType
    input_text: str
    target_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source information
    equation: Optional[TKSEquation] = None
    interpretation: Optional[str] = None
    teacher_response: Optional[str] = None

    # Quality scores
    canon_score: float = 1.0
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_type": self.task_type.value,
            "input": self.input_text,
            "target": self.target_text,
            "metadata": self.metadata,
            "equation": asdict(self.equation) if self.equation else None,
            "interpretation": self.interpretation,
            "canon_score": self.canon_score,
            "confidence_score": self.confidence_score
        }


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

E2I_TEMPLATE = """Given the TKS equation: {equation}

Elements present:
{element_list}

Provide an interpretation of this equation's meaning and effects.
Focus on the noetic interactions and RPM implications."""

I2E_TEMPLATE = """Given this interpretation of a TKS working:

{interpretation}

What elements would construct this equation?
Provide the elements in format like: B4, C10, A2 (only A,B,C,D worlds are valid)"""

S2E_TEMPLATE = """Scenario: {scenario}

What TKS equation would address this scenario?
Provide the elements and explain the noetic reasoning."""

E2RPM_TEMPLATE = """Given the TKS equation: {equation}

Compute the RPM (Desire/Wisdom/Power) distribution.
- Desire comes from N2 (Positive) and N3 (Negative)
- Wisdom comes from N1, N4, N5, N6, N7
- Power comes from N8 (Cause) and N9 (Effect)"""

E2F_TEMPLATE = """Given the TKS equation: {equation}

Identify the 7 Foundations present:
- Unity
- Wisdom
- Life
- Companionship
- Power
- Material
- Lust"""


# ==============================================================================
# TRAINING DATA TRANSFORMER
# ==============================================================================

class TrainingDataTransformer:
    """
    Transforms TKS equations and interpretations into training data.

    Takes:
        - TKS equations
        - LLM teacher interpretations
        - Validation scores

    Produces:
        - Task-specific training examples
        - Filtered by quality thresholds
    """

    def __init__(
        self,
        min_canon_score: float = 0.8,
        min_confidence_score: float = 0.6,
        include_metadata: bool = True
    ):
        """
        Initialize transformer.

        Args:
            min_canon_score: Minimum canon score to include example
            min_confidence_score: Minimum confidence to include example
            include_metadata: Whether to include metadata in output
        """
        self.min_canon_score = min_canon_score
        self.min_confidence_score = min_confidence_score
        self.include_metadata = include_metadata

        # Noetic canonical names
        self.noetic_names = {
            1: "Mind", 2: "Positive", 3: "Negative", 4: "Vibration",
            5: "Female", 6: "Male", 7: "Rhythm", 8: "Cause",
            9: "Effect", 10: "Idea"
        }

        # Canonical worlds: A, B, C, D only
        self.world_names = {
            'A': "Spiritual", 'B': "Mental", 'C': "Emotional", 'D': "Physical"
        }

        # RPM groupings
        self.rpm_noetics = {
            "desire": {2, 3},
            "wisdom": {1, 4, 5, 6, 7},
            "power": {8, 9}
        }

        # Foundation mappings (canonical)
        self.noetic_to_foundations = {
            10: ["Unity", "Material"],
            1: ["Wisdom"],
            4: ["Wisdom", "Life", "Material"],
            6: ["Wisdom", "Power", "Lust"],
            7: ["Wisdom", "Lust"],
            2: ["Companionship"],
            5: ["Companionship", "Lust"],
            8: ["Power"],
            9: ["Material"],
        }

    def transform(
        self,
        equation: TKSEquation,
        interpretation: str,
        canon_score: float = 1.0,
        confidence_score: float = 1.0,
        task_types: Optional[List[TaskType]] = None
    ) -> List[TransformedExample]:
        """
        Transform equation and interpretation into training examples.

        Args:
            equation: TKS equation object
            interpretation: LLM-generated interpretation
            canon_score: Canonical validation score
            confidence_score: Overall confidence score
            task_types: Which task types to generate (None = all)

        Returns:
            List of transformed training examples
        """
        # Check quality thresholds
        if canon_score < self.min_canon_score:
            return []
        if confidence_score < self.min_confidence_score:
            return []

        if task_types is None:
            task_types = [TaskType.E2I, TaskType.I2E, TaskType.E2RPM, TaskType.E2F]

        examples = []

        for task_type in task_types:
            example = self._transform_for_task(
                equation, interpretation, task_type,
                canon_score, confidence_score
            )
            if example:
                examples.append(example)

        return examples

    def _transform_for_task(
        self,
        equation: TKSEquation,
        interpretation: str,
        task_type: TaskType,
        canon_score: float,
        confidence_score: float
    ) -> Optional[TransformedExample]:
        """Transform for a specific task type."""
        if task_type == TaskType.E2I:
            return self._create_e2i_example(equation, interpretation, canon_score, confidence_score)
        elif task_type == TaskType.I2E:
            return self._create_i2e_example(equation, interpretation, canon_score, confidence_score)
        elif task_type == TaskType.E2RPM:
            return self._create_e2rpm_example(equation, interpretation, canon_score, confidence_score)
        elif task_type == TaskType.E2F:
            return self._create_e2f_example(equation, interpretation, canon_score, confidence_score)
        elif task_type == TaskType.S2E:
            return self._create_s2e_example(equation, interpretation, canon_score, confidence_score)
        elif task_type == TaskType.FULL:
            return self._create_full_example(equation, interpretation, canon_score, confidence_score)
        return None

    def _create_e2i_example(
        self,
        equation: TKSEquation,
        interpretation: str,
        canon_score: float,
        confidence_score: float
    ) -> TransformedExample:
        """Create Equation-to-Interpretation example."""
        # Build element list description
        element_list = []
        for elem in equation.elements:
            world = elem[0]
            noetic = int(elem[1:])
            element_list.append(
                f"- {elem}: {self.world_names.get(world, world)}-{self.noetic_names.get(noetic, noetic)}"
            )

        input_text = E2I_TEMPLATE.format(
            equation=equation.to_string(),
            element_list="\n".join(element_list)
        )

        return TransformedExample(
            task_type=TaskType.E2I,
            input_text=input_text,
            target_text=interpretation,
            equation=equation,
            interpretation=interpretation,
            canon_score=canon_score,
            confidence_score=confidence_score,
            metadata={
                "elements": equation.elements,
                "noetics": equation.noetics,
                "world": equation.world,
                "pattern": equation.pattern
            }
        )

    def _create_i2e_example(
        self,
        equation: TKSEquation,
        interpretation: str,
        canon_score: float,
        confidence_score: float
    ) -> TransformedExample:
        """Create Interpretation-to-Equation example."""
        input_text = I2E_TEMPLATE.format(interpretation=interpretation)
        target_text = equation.to_string()

        return TransformedExample(
            task_type=TaskType.I2E,
            input_text=input_text,
            target_text=target_text,
            equation=equation,
            interpretation=interpretation,
            canon_score=canon_score,
            confidence_score=confidence_score,
            metadata={
                "target_elements": equation.elements,
                "target_noetics": equation.noetics
            }
        )

    def _create_e2rpm_example(
        self,
        equation: TKSEquation,
        interpretation: str,
        canon_score: float,
        confidence_score: float
    ) -> TransformedExample:
        """Create Equation-to-RPM example."""
        input_text = E2RPM_TEMPLATE.format(equation=equation.to_string())

        # Compute RPM from noetics
        rpm = self._compute_rpm(equation.noetics)
        target_text = self._format_rpm_response(rpm, equation.noetics)

        return TransformedExample(
            task_type=TaskType.E2RPM,
            input_text=input_text,
            target_text=target_text,
            equation=equation,
            interpretation=interpretation,
            canon_score=canon_score,
            confidence_score=confidence_score,
            metadata={
                "target_rpm": rpm,
                "noetics": equation.noetics
            }
        )

    def _create_e2f_example(
        self,
        equation: TKSEquation,
        interpretation: str,
        canon_score: float,
        confidence_score: float
    ) -> TransformedExample:
        """Create Equation-to-Foundations example."""
        input_text = E2F_TEMPLATE.format(equation=equation.to_string())

        # Compute foundations from noetics
        foundations = self._compute_foundations(equation.noetics)
        target_text = self._format_foundations_response(foundations, equation.noetics)

        return TransformedExample(
            task_type=TaskType.E2F,
            input_text=input_text,
            target_text=target_text,
            equation=equation,
            interpretation=interpretation,
            canon_score=canon_score,
            confidence_score=confidence_score,
            metadata={
                "target_foundations": foundations,
                "noetics": equation.noetics
            }
        )

    def _create_s2e_example(
        self,
        equation: TKSEquation,
        interpretation: str,
        canon_score: float,
        confidence_score: float
    ) -> Optional[TransformedExample]:
        """Create Scenario-to-Equation example."""
        # Extract scenario from interpretation if possible
        scenario = self._extract_scenario(interpretation)
        if not scenario:
            return None

        input_text = S2E_TEMPLATE.format(scenario=scenario)
        target_text = f"Equation: {equation.to_string()}\n\nReasoning: {interpretation}"

        return TransformedExample(
            task_type=TaskType.S2E,
            input_text=input_text,
            target_text=target_text,
            equation=equation,
            interpretation=interpretation,
            canon_score=canon_score,
            confidence_score=confidence_score,
            metadata={
                "scenario": scenario,
                "target_elements": equation.elements
            }
        )

    def _create_full_example(
        self,
        equation: TKSEquation,
        interpretation: str,
        canon_score: float,
        confidence_score: float
    ) -> TransformedExample:
        """Create full pipeline example with all components."""
        rpm = self._compute_rpm(equation.noetics)
        foundations = self._compute_foundations(equation.noetics)

        input_text = f"Analyze the TKS equation: {equation.to_string()}"

        target_parts = [
            f"Equation: {equation.to_string()}",
            f"\nElements: {', '.join(equation.elements)}",
            f"\nRPM Distribution:",
            f"- Desire: {rpm['desire']:.2%}",
            f"- Wisdom: {rpm['wisdom']:.2%}",
            f"- Power: {rpm['power']:.2%}",
            f"\nFoundations Present: {', '.join(foundations)}",
            f"\nInterpretation:\n{interpretation}"
        ]

        return TransformedExample(
            task_type=TaskType.FULL,
            input_text=input_text,
            target_text="\n".join(target_parts),
            equation=equation,
            interpretation=interpretation,
            canon_score=canon_score,
            confidence_score=confidence_score,
            metadata={
                "rpm": rpm,
                "foundations": foundations,
                "elements": equation.elements
            }
        )

    def _compute_rpm(self, noetics: List[int]) -> Dict[str, float]:
        """Compute RPM distribution from noetics."""
        if not noetics:
            return {"desire": 0.0, "wisdom": 0.0, "power": 0.0}

        desire_count = sum(1 for n in noetics if n in self.rpm_noetics["desire"])
        wisdom_count = sum(1 for n in noetics if n in self.rpm_noetics["wisdom"])
        power_count = sum(1 for n in noetics if n in self.rpm_noetics["power"])

        total = len(noetics)
        return {
            "desire": desire_count / total,
            "wisdom": wisdom_count / total,
            "power": power_count / total
        }

    def _compute_foundations(self, noetics: List[int]) -> List[str]:
        """Compute foundations present from noetics."""
        foundations = set()
        for n in noetics:
            if n in self.noetic_to_foundations:
                foundations.update(self.noetic_to_foundations[n])
        return sorted(foundations)

    def _format_rpm_response(self, rpm: Dict[str, float], noetics: List[int]) -> str:
        """Format RPM as response text."""
        lines = ["RPM Distribution:"]
        lines.append(f"- Desire: {rpm['desire']:.1%} (from N2, N3)")
        lines.append(f"- Wisdom: {rpm['wisdom']:.1%} (from N1, N4, N5, N6, N7)")
        lines.append(f"- Power: {rpm['power']:.1%} (from N8, N9)")

        dominant = max(rpm, key=rpm.get)
        lines.append(f"\nDominant component: {dominant.title()}")

        return "\n".join(lines)

    def _format_foundations_response(self, foundations: List[str], noetics: List[int]) -> str:
        """Format foundations as response text."""
        lines = ["Foundations Present:"]
        for f in foundations:
            lines.append(f"- {f}")

        if not foundations:
            lines.append("- None (N10/Idea contains all foundations)")

        return "\n".join(lines)

    def _extract_scenario(self, interpretation: str) -> Optional[str]:
        """Extract a scenario description from interpretation."""
        # Look for goal/purpose statements
        patterns = [
            r"(?:to|for|in order to)\s+(.+?)(?:\.|,|$)",
            r"(?:goal|purpose|intent)(?:ion)?(?:\s+is)?\s*:?\s*(.+?)(?:\.|$)",
            r"(?:working for|equation for)\s+(.+?)(?:\.|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, interpretation, re.I)
            if match:
                scenario = match.group(1).strip()
                if len(scenario) > 10:  # Minimum length
                    return scenario

        # Fallback: use first sentence if substantial
        sentences = interpretation.split('.')
        if sentences and len(sentences[0]) > 20:
            return sentences[0].strip()

        return None

    def transform_batch(
        self,
        equations: List[TKSEquation],
        interpretations: List[str],
        canon_scores: Optional[List[float]] = None,
        confidence_scores: Optional[List[float]] = None,
        task_types: Optional[List[TaskType]] = None
    ) -> List[TransformedExample]:
        """
        Transform a batch of equations and interpretations.

        Args:
            equations: List of TKS equations
            interpretations: Corresponding interpretations
            canon_scores: Optional list of canon scores
            confidence_scores: Optional list of confidence scores
            task_types: Task types to generate

        Returns:
            List of all transformed examples
        """
        if len(equations) != len(interpretations):
            raise ValueError("Equations and interpretations must have same length")

        if canon_scores is None:
            canon_scores = [1.0] * len(equations)
        if confidence_scores is None:
            confidence_scores = [1.0] * len(equations)

        all_examples = []
        for eq, interp, cs, conf in zip(equations, interpretations, canon_scores, confidence_scores):
            examples = self.transform(eq, interp, cs, conf, task_types)
            all_examples.extend(examples)

        return all_examples

    def save_examples(
        self,
        examples: List[TransformedExample],
        output_path: Union[str, Path],
        format: str = "jsonl"
    ):
        """
        Save transformed examples to file.

        Args:
            examples: List of transformed examples
            output_path: Output file path
            format: "jsonl" or "json"
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example.to_dict()) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([e.to_dict() for e in examples], f, indent=2)

    def load_examples(
        self,
        input_path: Union[str, Path]
    ) -> List[TransformedExample]:
        """
        Load transformed examples from file.

        Args:
            input_path: Input file path (.jsonl or .json)

        Returns:
            List of transformed examples
        """
        input_path = Path(input_path)

        examples = []
        if input_path.suffix == ".jsonl":
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(self._dict_to_example(data))
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                for data in data_list:
                    examples.append(self._dict_to_example(data))

        return examples

    def _dict_to_example(self, data: Dict) -> TransformedExample:
        """Convert dictionary to TransformedExample."""
        equation = None
        if data.get("equation"):
            eq_data = data["equation"]
            equation = TKSEquation(
                elements=eq_data.get("elements", []),
                noetics=eq_data.get("noetics", []),
                world=eq_data.get("world"),
                pattern=eq_data.get("pattern"),
                rpm=eq_data.get("rpm", {}),
                foundations=eq_data.get("foundations", [])
            )

        return TransformedExample(
            task_type=TaskType(data["task_type"]),
            input_text=data["input"],
            target_text=data["target"],
            metadata=data.get("metadata", {}),
            equation=equation,
            interpretation=data.get("interpretation"),
            canon_score=data.get("canon_score", 1.0),
            confidence_score=data.get("confidence_score", 1.0)
        )

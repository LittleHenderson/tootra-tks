"""
RTTA (Reasoning-Test Tootra Algorithm) Inference Scaffolding

Wraps a base language model with structured reasoning:
- Goal encoding
- RPM prerequisite tracking
- Curiosity-driven step selection
- Answer synthesis
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import math


class World(Enum):
    """Tootra worlds."""
    PHYSICAL = "physical"
    EMOTIONAL = "emotional"
    MENTAL = "mental"
    SPIRITUAL = "spiritual"


class Noetic(Enum):
    """Noetic operators 1-10."""
    MIND = 1          # Awareness/consciousness
    HEART = 2         # Feeling/connection
    DESIRE = 3        # Want/attraction
    VIBRATION = 4     # Uncertainty/mismatch
    FEMALE = 5        # Identity/form
    MALE = 6          # Components/parts
    WISDOM = 7        # Understanding
    POWER = 8         # Execution/force
    FOUNDATION = 9    # Grounding
    MANIFESTATION = 10  # Realization


@dataclass
class Goal:
    """RTTA Goal encoding."""
    text: str
    worlds: List[World] = field(default_factory=lambda: [World.MENTAL])
    noetics: List[Noetic] = field(default_factory=lambda: [Noetic.MIND, Noetic.VIBRATION])
    foundation: str = "wisdom"

    def encode(self) -> str:
        """Encode goal as string for model input."""
        worlds_str = "/".join(w.value for w in self.worlds)
        noetics_str = "+".join(str(n.value) for n in self.noetics)
        return f"G = ({worlds_str})^{noetics_str}_{self.foundation}: {self.text}"


@dataclass
class Prerequisite:
    """Single prerequisite in RPM tree."""
    id: str
    name: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    satisfied: bool = False
    evidence: str = ""

    def is_available(self, satisfied_prereqs: set) -> bool:
        """Check if this prereq is available (all dependencies satisfied)."""
        return all(dep in satisfied_prereqs for dep in self.depends_on)


@dataclass
class ReasoningState:
    """State during RTTA reasoning loop."""
    acquired: set = field(default_factory=set)
    evidence: Dict[str, str] = field(default_factory=dict)
    desire: float = 0.9
    wisdom: float = 0.8
    power: float = 0.8
    steps_taken: List[Dict] = field(default_factory=list)
    max_steps: int = 10


class RTTAScaffold:
    """
    RTTA inference scaffolding that wraps a base model.

    Uses structured reasoning to guide the model through:
    1. Goal encoding
    2. RPM prerequisite decomposition
    3. Curiosity-driven step selection
    4. Answer synthesis
    """

    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Standard RPM templates for different problem types
        self.rpm_templates = {
            "math": self._math_rpm_template(),
            "logic": self._logic_rpm_template(),
            "sequence": self._sequence_rpm_template(),
            "general": self._general_rpm_template(),
        }

    def _math_rpm_template(self) -> List[Prerequisite]:
        """RPM template for math problems."""
        return [
            Prerequisite("P1", "Variable Identification",
                        "Identify unknowns and knowns", []),
            Prerequisite("P2", "Equation Construction",
                        "Build equations from problem text", ["P1"]),
            Prerequisite("P3", "Solve System",
                        "Solve equations for unknowns", ["P2"]),
            Prerequisite("P4", "Sanity Check",
                        "Verify solution validity", ["P3"]),
            Prerequisite("P5", "Answer Extraction",
                        "Format final answer", ["P4"]),
        ]

    def _logic_rpm_template(self) -> List[Prerequisite]:
        """RPM template for logic problems."""
        return [
            Prerequisite("P1", "Parse Premises",
                        "Extract logical statements", []),
            Prerequisite("P2", "Apply Rules",
                        "Apply inference rules", ["P1"]),
            Prerequisite("P3", "Derive Conclusion",
                        "Determine if conclusion follows", ["P2"]),
        ]

    def _sequence_rpm_template(self) -> List[Prerequisite]:
        """RPM template for sequence problems."""
        return [
            Prerequisite("P1", "Observe Pattern",
                        "Identify differences/ratios between terms", []),
            Prerequisite("P2", "Hypothesize Rule",
                        "Form hypothesis about sequence rule", ["P1"]),
            Prerequisite("P3", "Verify Rule",
                        "Check if rule fits all given terms", ["P2"]),
            Prerequisite("P4", "Apply Rule",
                        "Use rule to predict next term", ["P3"]),
        ]

    def _general_rpm_template(self) -> List[Prerequisite]:
        """Generic RPM template."""
        return [
            Prerequisite("P1", "Understand Question",
                        "Parse what is being asked", []),
            Prerequisite("P2", "Gather Information",
                        "Identify relevant facts", ["P1"]),
            Prerequisite("P3", "Reason",
                        "Apply reasoning to derive answer", ["P2"]),
            Prerequisite("P4", "Conclude",
                        "State final answer", ["P3"]),
        ]

    def classify_problem(self, question: str) -> str:
        """Classify problem type based on question text."""
        question_lower = question.lower()

        if any(kw in question_lower for kw in ["$", "cost", "price", "spend", "buy", "sell", "total"]):
            return "math"
        if any(kw in question_lower for kw in ["all", "some", "if", "then", "therefore", "follows"]):
            return "logic"
        if any(kw in question_lower for kw in ["sequence", "next", "pattern", "comes after"]):
            return "sequence"
        if re.search(r'\d+\s*[+\-*/]\s*\d+', question):
            return "math"
        if re.search(r'\d+,\s*\d+,\s*\d+', question):
            return "sequence"

        return "general"

    def encode_goal(self, question: str, problem_type: str) -> Goal:
        """Encode question as RTTA goal."""
        if problem_type == "math":
            return Goal(
                text=question,
                worlds=[World.MENTAL, World.EMOTIONAL],
                noetics=[Noetic.MIND, Noetic.VIBRATION],
                foundation="wisdom"
            )
        elif problem_type == "logic":
            return Goal(
                text=question,
                worlds=[World.MENTAL],
                noetics=[Noetic.MIND, Noetic.VIBRATION, Noetic.WISDOM],
                foundation="wisdom"
            )
        else:
            return Goal(
                text=question,
                worlds=[World.MENTAL],
                noetics=[Noetic.MIND, Noetic.VIBRATION],
                foundation="wisdom"
            )

    def compute_curiosity(self, prereq: Prerequisite, state: ReasoningState,
                          all_prereqs: List[Prerequisite]) -> float:
        """
        Compute curiosity score for a prerequisite.

        Curiosity = f(Vibration, InfoGain, Relevance) * Wisdom * Desire
        """
        # Vibration: how much uncertainty does this prereq carry?
        # Higher for earlier prereqs that haven't been resolved
        prereq_idx = next((i for i, p in enumerate(all_prereqs) if p.id == prereq.id), 0)
        vibration = 1.0 - (prereq_idx / len(all_prereqs)) * 0.5

        # InfoGain: how much will resolving this reduce uncertainty?
        # Higher for prereqs with more dependents
        dependents = sum(1 for p in all_prereqs if prereq.id in p.depends_on)
        info_gain = 0.5 + (dependents / len(all_prereqs)) * 0.5

        # Relevance: how directly does this move us to the goal?
        # Higher for prereqs closer to the end
        relevance = 0.5 + (prereq_idx / len(all_prereqs)) * 0.5

        # Combine with D/W/P
        curiosity = (vibration * 0.4 + info_gain * 0.3 + relevance * 0.3)
        curiosity *= state.wisdom * state.desire

        return curiosity

    def select_next_prereq(self, prereqs: List[Prerequisite],
                           state: ReasoningState) -> Optional[Prerequisite]:
        """Select next prerequisite to resolve based on curiosity."""
        available = [p for p in prereqs if not p.satisfied and p.is_available(state.acquired)]

        if not available:
            return None

        # Compute curiosity for each available prereq
        scored = [(p, self.compute_curiosity(p, state, prereqs)) for p in available]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0]

    def resolve_prereq(self, prereq: Prerequisite, goal: Goal,
                       state: ReasoningState, prereqs: List[Prerequisite]) -> Tuple[bool, str]:
        """
        Resolve a prerequisite using the base model.

        Returns (success, evidence).
        """
        # Build prompt for the model
        prompt = self._build_resolve_prompt(prereq, goal, state, prereqs)

        # Generate response from model
        response = self._generate(prompt, max_tokens=100)

        # Parse response for evidence
        evidence = self._extract_evidence(response, prereq)

        # Determine success (simplified: if we got non-empty evidence)
        success = len(evidence.strip()) > 0

        return success, evidence

    def _build_resolve_prompt(self, prereq: Prerequisite, goal: Goal,
                              state: ReasoningState, prereqs: List[Prerequisite]) -> str:
        """Build prompt for resolving a prerequisite."""
        lines = [
            f"Question: {goal.text}",
            "",
            "Current state:",
        ]

        for p in prereqs:
            status = "DONE" if p.satisfied else "pending"
            if p.satisfied and p.id in state.evidence:
                lines.append(f"  {p.id} ({p.name}): {status} - {state.evidence[p.id]}")
            else:
                lines.append(f"  {p.id} ({p.name}): {status}")

        lines.extend([
            "",
            f"Now resolving {prereq.id}: {prereq.name}",
            f"Task: {prereq.description}",
            "",
            "Result:"
        ])

        return "\n".join(lines)

    def _generate(self, prompt: str, max_tokens: int = 100,
                  temperature: float = 0.7) -> str:
        """Generate text from the base model."""
        # Encode prompt
        if hasattr(self.tokenizer, 'encode'):
            enc = self.tokenizer.encode(prompt)
            input_ids = enc.ids if hasattr(enc, 'ids') else enc
        else:
            input_ids = [self.tokenizer.get(c, 1) for c in prompt]

        # Add BOS if available
        bos_id = None
        if hasattr(self.tokenizer, 'token_to_id'):
            bos_id = self.tokenizer.token_to_id("<BOS>")
        if bos_id is not None:
            input_ids = [bos_id] + input_ids

        input_tensor = torch.tensor([input_ids], device=self.device)

        # Generate
        generated_ids = []
        current = input_tensor

        eos_id = None
        if hasattr(self.tokenizer, 'token_to_id'):
            eos_id = self.tokenizer.token_to_id("<EOS>")

        with torch.no_grad():
            for _ in range(max_tokens):
                output = self.model(current)
                logits = output['logits'][0, -1, :] / temperature

                # Top-k sampling
                top_k = 40
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_values[-1]
                logits[logits < threshold] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated_ids.append(next_token.item())

                if eos_id is not None and next_token.item() == eos_id:
                    break

                current = torch.cat([current, next_token.unsqueeze(0)], dim=1)

        # Decode
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(generated_ids)
        else:
            idx_to_token = {v: k for k, v in self.tokenizer.items()}
            return "".join(idx_to_token.get(i, "?") for i in generated_ids)

    def _extract_evidence(self, response: str, prereq: Prerequisite) -> str:
        """Extract evidence from model response."""
        # Clean up response
        response = response.strip()

        # Remove common noise
        response = re.sub(r'^(Result:|Answer:|Output:)\s*', '', response)

        # Take first meaningful line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            return lines[0][:200]  # Limit length

        return response[:200] if response else f"Completed {prereq.name}"

    def synthesize_answer(self, goal: Goal, state: ReasoningState,
                          prereqs: List[Prerequisite]) -> str:
        """Synthesize final answer from resolved prereqs."""
        # Build synthesis prompt
        lines = [
            f"Question: {goal.text}",
            "",
            "Reasoning completed:",
        ]

        for p in prereqs:
            if p.satisfied and p.id in state.evidence:
                lines.append(f"  {p.name}: {state.evidence[p.id]}")

        lines.extend([
            "",
            "Final Answer:"
        ])

        prompt = "\n".join(lines)

        # Generate final answer
        answer = self._generate(prompt, max_tokens=50, temperature=0.5)

        return answer.strip()

    def reason(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Main RTTA reasoning loop.

        Args:
            question: The question to answer
            verbose: Whether to print reasoning steps

        Returns:
            Dict with answer, explanation, and reasoning trace
        """
        # Classify problem type
        problem_type = self.classify_problem(question)
        if verbose:
            print(f"Problem type: {problem_type}")

        # Encode goal
        goal = self.encode_goal(question, problem_type)
        if verbose:
            print(f"Goal: {goal.encode()}")

        # Get RPM template
        prereqs = [Prerequisite(p.id, p.name, p.description, list(p.depends_on))
                   for p in self.rpm_templates[problem_type]]

        # Initialize state
        state = ReasoningState()

        if verbose:
            print("\nRPM Prerequisites:")
            for p in prereqs:
                deps = f" <- {p.depends_on}" if p.depends_on else ""
                print(f"  {p.id}: {p.name}{deps}")

        # Curiosity loop
        if verbose:
            print("\nReasoning loop:")

        while len(state.acquired) < len(prereqs) and len(state.steps_taken) < state.max_steps:
            # Select next prereq
            next_prereq = self.select_next_prereq(prereqs, state)

            if next_prereq is None:
                break

            # Compute curiosity scores for logging
            available = [p for p in prereqs if not p.satisfied and p.is_available(state.acquired)]
            curiosity_scores = {p.id: self.compute_curiosity(p, state, prereqs) for p in available}

            if verbose:
                print(f"\n  Step {len(state.steps_taken) + 1}: {next_prereq.id} ({next_prereq.name})")
                print(f"    Curiosity: {curiosity_scores.get(next_prereq.id, 0):.3f}")

            # Resolve prereq
            success, evidence = self.resolve_prereq(next_prereq, goal, state, prereqs)

            if success:
                next_prereq.satisfied = True
                state.acquired.add(next_prereq.id)
                state.evidence[next_prereq.id] = evidence

                if verbose:
                    print(f"    Evidence: {evidence[:80]}...")

            # Record step
            state.steps_taken.append({
                "prereq": next_prereq.id,
                "curiosity": curiosity_scores,
                "success": success,
                "evidence": evidence
            })

        # Synthesize answer
        if verbose:
            print("\nSynthesizing answer...")

        answer = self.synthesize_answer(goal, state, prereqs)

        # Build explanation
        explanation_parts = []
        for p in prereqs:
            if p.satisfied and p.id in state.evidence:
                explanation_parts.append(f"{p.name}: {state.evidence[p.id]}")

        return {
            "question": question,
            "problem_type": problem_type,
            "goal": goal.encode(),
            "answer": answer,
            "explanation": " | ".join(explanation_parts),
            "steps": state.steps_taken,
            "prereqs_satisfied": len(state.acquired),
            "prereqs_total": len(prereqs)
        }


# ============================================================
# STANDALONE RTTA (without model, for testing)
# ============================================================

class StandaloneRTTA:
    """
    RTTA that uses rules instead of a model.
    Useful for testing the scaffolding logic.
    """

    def __init__(self):
        self.rpm_templates = {
            "math": [
                ("P1", "Variable Identification", []),
                ("P2", "Equation Construction", ["P1"]),
                ("P3", "Solve System", ["P2"]),
                ("P4", "Sanity Check", ["P3"]),
                ("P5", "Answer Extraction", ["P4"]),
            ],
            "logic": [
                ("P1", "Parse Premises", []),
                ("P2", "Apply Rules", ["P1"]),
                ("P3", "Derive Conclusion", ["P2"]),
            ],
        }

    def solve_math(self, question: str) -> Dict:
        """Solve a math word problem using rules."""
        # Simple pattern matching for linear system problems
        import re

        # Extract prices
        prices = re.findall(r'\$(\d+)', question)
        prices = [int(p) for p in prices]

        # Extract totals
        totals = re.findall(r'(\d+)\s+items', question)
        total_items = int(totals[0]) if totals else None

        cost_match = re.search(r'spends?\s+\$(\d+)', question)
        total_cost = int(cost_match.group(1)) if cost_match else None

        if len(prices) >= 2 and total_items and total_cost:
            p1, p2 = prices[0], prices[1]
            # p + n = total_items
            # p1*p + p2*n = total_cost
            # Solve: p = (total_cost - p2*total_items) / (p1 - p2)
            if p1 != p2:
                p = (total_cost - p2 * total_items) / (p1 - p2)
                if p == int(p) and 0 <= p <= total_items:
                    return {"answer": str(int(p)), "method": "linear_system"}

        return {"answer": "Unable to solve", "method": "failed"}

    def reason(self, question: str) -> Dict:
        """Main reasoning entry point."""
        question_lower = question.lower()

        if "$" in question and "items" in question_lower:
            result = self.solve_math(question)
            return {
                "question": question,
                "answer": result["answer"],
                "method": result["method"]
            }

        return {
            "question": question,
            "answer": "Unknown problem type",
            "method": "unsupported"
        }


if __name__ == "__main__":
    # Test standalone RTTA
    rtta = StandaloneRTTA()

    test_questions = [
        "A store sells pens for $2 each and notebooks for $5 each. A customer buys 4 items in total and spends $14. How many pens did the customer buy?",
        "A store sells shirts for $3 each and pants for $8 each. A customer buys 6 items in total and spends $33. How many shirts did the customer buy?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = rtta.reason(q)
        print(f"A: {result['answer']}")
        print(f"Method: {result['method']}")

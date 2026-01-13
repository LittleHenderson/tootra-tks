"""
RTTA (Reasoning-Test Tootra Algorithm) Training Data Generator

Generates diverse reasoning problems in RTTA format:
- G: Goal encoding (worlds, noetics, foundation)
- RPM: Prerequisite tree
- Loop: Curiosity-driven steps
- Answer: Final synthesized result
"""

import json
import random
import os
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict

@dataclass
class RTTAGoal:
    """Tootra goal encoding."""
    worlds: List[str]  # mental, emotional, physical, spiritual
    noetics: List[str]  # mind, vibration, etc.
    foundation: str  # desire, wisdom, power
    text: str  # actual question

@dataclass
class RTTAPrereq:
    """Single prerequisite in RPM tree."""
    id: str
    name: str
    description: str
    depends_on: List[str]

@dataclass
class RTTAStep:
    """Single step in the curiosity loop."""
    available: List[str]
    chosen: str
    curiosity_scores: Dict[str, float]
    reasoning: str
    evidence_delta: str
    acquired_after: List[str]

@dataclass
class RTTAExample:
    """Complete RTTA training example."""
    question: str
    goal: RTTAGoal
    rpm: List[RTTAPrereq]
    desire: float
    wisdom: float
    power: float
    steps: List[RTTAStep]
    final_answer: str
    explanation: str


# ============================================================
# PROBLEM GENERATORS
# ============================================================

class MathWordProblemGenerator:
    """Generates math word problems with RTTA structure."""

    TEMPLATES = [
        {
            "type": "linear_system",
            "template": "A store sells {item1} for ${price1} each and {item2} for ${price2} each. A customer buys {total_items} items in total and spends ${total_cost}. How many {item1} did the customer buy?",
            "solve": lambda p1, p2, total, cost: (cost - p2 * total) // (p1 - p2) if (p1 - p2) != 0 else 0
        },
        {
            "type": "percentage",
            "template": "A shirt originally costs ${original}. It is on sale for {discount}% off. What is the sale price?",
            "solve": lambda orig, disc: orig * (100 - disc) / 100
        },
        {
            "type": "distance_time",
            "template": "A car travels at {speed} mph for {time} hours. How many miles does it travel?",
            "solve": lambda s, t: s * t
        },
        {
            "type": "ratio",
            "template": "The ratio of boys to girls in a class is {r1}:{r2}. If there are {total} students total, how many boys are there?",
            "solve": lambda r1, r2, total: (r1 * total) // (r1 + r2)
        },
        {
            "type": "age",
            "template": "{name1} is {diff} years older than {name2}. In {years} years, the sum of their ages will be {future_sum}. How old is {name2} now?",
            "solve": lambda diff, years, future_sum: (future_sum - 2 * years - diff) // 2
        },
    ]

    ITEMS = [("pens", "notebooks"), ("apples", "oranges"), ("books", "magazines"),
             ("shirts", "pants"), ("cookies", "brownies")]
    NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

    def generate(self) -> Tuple[str, str, Dict]:
        """Generate a math word problem with solution."""
        template = random.choice(self.TEMPLATES)

        if template["type"] == "linear_system":
            item1, item2 = random.choice(self.ITEMS)
            price1 = random.randint(2, 5)
            price2 = random.randint(6, 10)
            # Ensure valid solution exists
            answer = random.randint(1, 5)
            other = random.randint(1, 5)
            total_items = answer + other
            total_cost = price1 * answer + price2 * other

            question = template["template"].format(
                item1=item1, item2=item2, price1=price1, price2=price2,
                total_items=total_items, total_cost=total_cost
            )
            return question, str(answer), {
                "type": "linear_system",
                "variables": {"p": f"number of {item1}", "n": f"number of {item2}"},
                "equations": [f"p + n = {total_items}", f"{price1}p + {price2}n = {total_cost}"],
                "solution": {"p": answer, "n": other}
            }

        elif template["type"] == "percentage":
            original = random.randint(20, 100)
            discount = random.choice([10, 15, 20, 25, 30, 50])
            answer = original * (100 - discount) / 100

            question = template["template"].format(original=original, discount=discount)
            return question, f"${answer:.2f}", {
                "type": "percentage",
                "original": original,
                "discount": discount,
                "calculation": f"{original} * (100 - {discount}) / 100 = {answer}"
            }

        elif template["type"] == "distance_time":
            speed = random.choice([30, 40, 50, 60, 65, 70])
            time = random.randint(2, 6)
            answer = speed * time

            question = template["template"].format(speed=speed, time=time)
            return question, f"{answer} miles", {
                "type": "distance_time",
                "formula": "distance = speed * time",
                "calculation": f"{speed} * {time} = {answer}"
            }

        elif template["type"] == "ratio":
            r1 = random.randint(2, 5)
            r2 = random.randint(2, 5)
            multiplier = random.randint(3, 8)
            total = (r1 + r2) * multiplier
            answer = r1 * multiplier

            question = template["template"].format(r1=r1, r2=r2, total=total)
            return question, str(answer), {
                "type": "ratio",
                "ratio": f"{r1}:{r2}",
                "total": total,
                "calculation": f"boys = {r1} * {total} / ({r1} + {r2}) = {answer}"
            }

        elif template["type"] == "age":
            name1, name2 = random.sample(self.NAMES, 2)
            age2_now = random.randint(10, 30)
            diff = random.randint(3, 10)
            years = random.randint(5, 15)
            age1_now = age2_now + diff
            future_sum = (age1_now + years) + (age2_now + years)

            question = template["template"].format(
                name1=name1, name2=name2, diff=diff, years=years, future_sum=future_sum
            )
            return question, str(age2_now), {
                "type": "age",
                "variables": {name1: f"x + {diff}", name2: "x"},
                "equation": f"(x + {diff} + {years}) + (x + {years}) = {future_sum}",
                "solution": age2_now
            }

        return "", "", {}


class LogicProblemGenerator:
    """Generates logic problems with RTTA structure."""

    def generate(self) -> Tuple[str, str, Dict]:
        """Generate a logic problem."""
        problem_type = random.choice(["syllogism", "conditional", "sequence", "odd_one_out"])

        if problem_type == "syllogism":
            return self._syllogism()
        elif problem_type == "conditional":
            return self._conditional()
        elif problem_type == "sequence":
            return self._sequence()
        else:
            return self._odd_one_out()

    def _syllogism(self) -> Tuple[str, str, Dict]:
        """Generate a syllogism problem."""
        categories = [
            ("mammals", "animals", "living things"),
            ("dogs", "mammals", "animals"),
            ("squares", "rectangles", "shapes"),
            ("roses", "flowers", "plants"),
            ("students", "people", "beings"),
        ]

        A, B, C = random.choice(categories)

        # Valid syllogism: All A are B, All B are C -> All A are C
        question = f"All {A} are {B}. All {B} are {C}. Is it true that all {A} are {C}?"
        answer = "Yes"

        return question, answer, {
            "type": "syllogism",
            "premises": [f"All {A} are {B}", f"All {B} are {C}"],
            "conclusion": f"All {A} are {C}",
            "valid": True,
            "reasoning": "Transitive property of subset relations"
        }

    def _conditional(self) -> Tuple[str, str, Dict]:
        """Generate a conditional logic problem."""
        scenarios = [
            ("it rains", "the ground gets wet", "the ground is wet", "Did it rain?", "Not necessarily"),
            ("you study", "you pass the test", "you didn't study", "Will you pass?", "Not necessarily"),
            ("the light is on", "someone is home", "no one is home", "Is the light on?", "No"),
        ]

        P, Q, given, question_part, answer = random.choice(scenarios)

        question = f"If {P}, then {Q}. Given: {given}. {question_part}"

        return question, answer, {
            "type": "conditional",
            "premise": f"If {P} then {Q}",
            "given": given,
            "reasoning": "Affirming the consequent is a fallacy; denying the antecedent gives uncertain result"
        }

    def _sequence(self) -> Tuple[str, str, Dict]:
        """Generate a number sequence problem."""
        seq_type = random.choice(["arithmetic", "geometric", "fibonacci_like", "squares"])

        if seq_type == "arithmetic":
            start = random.randint(1, 10)
            diff = random.randint(2, 5)
            seq = [start + i * diff for i in range(5)]
            answer = seq[-1] + diff
            pattern = f"arithmetic sequence with difference {diff}"

        elif seq_type == "geometric":
            start = random.randint(1, 3)
            ratio = random.randint(2, 3)
            seq = [start * (ratio ** i) for i in range(5)]
            answer = seq[-1] * ratio
            pattern = f"geometric sequence with ratio {ratio}"

        elif seq_type == "fibonacci_like":
            a, b = random.randint(1, 3), random.randint(1, 3)
            seq = [a, b]
            for _ in range(4):
                seq.append(seq[-1] + seq[-2])
            answer = seq[-1] + seq[-2]
            seq = seq[:5]
            pattern = "each term is sum of two preceding terms"

        else:  # squares
            start = random.randint(1, 4)
            seq = [(start + i) ** 2 for i in range(5)]
            answer = (start + 5) ** 2
            pattern = "sequence of consecutive squares"

        question = f"What comes next in the sequence: {', '.join(map(str, seq))}, ?"

        return question, str(answer), {
            "type": "sequence",
            "sequence": seq,
            "pattern": pattern,
            "next_value": answer
        }

    def _odd_one_out(self) -> Tuple[str, str, Dict]:
        """Generate an odd-one-out problem."""
        groups = [
            (["dog", "cat", "elephant", "car", "lion"], "car", "not an animal"),
            (["2", "4", "6", "9", "10"], "9", "not even"),
            (["apple", "banana", "carrot", "orange", "grape"], "carrot", "not a fruit"),
            (["circle", "square", "triangle", "red", "pentagon"], "red", "not a shape"),
        ]

        items, answer, reason = random.choice(groups)
        random.shuffle(items)

        question = f"Which one doesn't belong: {', '.join(items)}?"

        return question, answer, {
            "type": "odd_one_out",
            "items": items,
            "odd_one": answer,
            "reason": reason
        }


# ============================================================
# RTTA FORMATTER
# ============================================================

class RTTAFormatter:
    """Formats problems into full RTTA structure."""

    def format_math_problem(self, question: str, answer: str, details: Dict) -> RTTAExample:
        """Format a math problem into RTTA structure."""

        # Goal encoding
        goal = RTTAGoal(
            worlds=["mental", "emotional"],
            noetics=["mind", "vibration"],
            foundation="wisdom",
            text=question
        )

        # RPM prerequisites
        rpm = [
            RTTAPrereq("P1", "Variable Identification",
                      "Identify unknowns and knowns from the problem", []),
            RTTAPrereq("P2", "Equation Construction",
                      "Build mathematical equations from the text", ["P1"]),
            RTTAPrereq("P3", "Solve System",
                      "Solve the equations to find unknowns", ["P2"]),
            RTTAPrereq("P4", "Sanity Check",
                      "Verify solution fits original constraints", ["P3"]),
            RTTAPrereq("P5", "Answer Extraction",
                      "Express final answer in required format", ["P4"]),
        ]

        # Build steps based on problem type
        steps = self._build_math_steps(details)

        # Explanation
        explanation = self._build_math_explanation(details, answer)

        return RTTAExample(
            question=question,
            goal=goal,
            rpm=rpm,
            desire=0.9,
            wisdom=0.8,
            power=0.8,
            steps=steps,
            final_answer=answer,
            explanation=explanation
        )

    def format_logic_problem(self, question: str, answer: str, details: Dict) -> RTTAExample:
        """Format a logic problem into RTTA structure."""

        goal = RTTAGoal(
            worlds=["mental"],
            noetics=["mind", "vibration"],
            foundation="wisdom",
            text=question
        )

        # Simpler RPM for logic
        rpm = [
            RTTAPrereq("P1", "Parse Premises",
                      "Identify given statements and their logical form", []),
            RTTAPrereq("P2", "Apply Rules",
                      "Apply logical inference rules", ["P1"]),
            RTTAPrereq("P3", "Derive Conclusion",
                      "Determine if target statement follows", ["P2"]),
        ]

        steps = self._build_logic_steps(details)
        explanation = self._build_logic_explanation(details, answer)

        return RTTAExample(
            question=question,
            goal=goal,
            rpm=rpm,
            desire=0.85,
            wisdom=0.9,
            power=0.7,
            steps=steps,
            final_answer=answer,
            explanation=explanation
        )

    def _build_math_steps(self, details: Dict) -> List[RTTAStep]:
        """Build curiosity loop steps for math problem."""
        steps = []

        # Step 1: Variable identification
        if "variables" in details:
            var_str = ", ".join(f"{k} = {v}" for k, v in details["variables"].items())
        else:
            var_str = "identified relevant quantities"

        steps.append(RTTAStep(
            available=["P1"],
            chosen="P1",
            curiosity_scores={"P1": 0.95},
            reasoning="High vibration - don't know what we're solving for yet",
            evidence_delta=f"Variables defined: {var_str}",
            acquired_after=["P1"]
        ))

        # Step 2: Equation construction
        if "equations" in details:
            eq_str = "; ".join(details["equations"])
        elif "formula" in details:
            eq_str = details["formula"]
        else:
            eq_str = "formulated relationship"

        steps.append(RTTAStep(
            available=["P2"],
            chosen="P2",
            curiosity_scores={"P2": 0.9},
            reasoning="Need formal equations to proceed",
            evidence_delta=f"Equations: {eq_str}",
            acquired_after=["P1", "P2"]
        ))

        # Step 3: Solve
        if "calculation" in details:
            calc_str = details["calculation"]
        elif "solution" in details:
            calc_str = f"Solution: {details['solution']}"
        else:
            calc_str = "computed result"

        steps.append(RTTAStep(
            available=["P3"],
            chosen="P3",
            curiosity_scores={"P3": 0.85},
            reasoning="Equations ready, now solve",
            evidence_delta=calc_str,
            acquired_after=["P1", "P2", "P3"]
        ))

        # Step 4: Check
        steps.append(RTTAStep(
            available=["P4"],
            chosen="P4",
            curiosity_scores={"P4": 0.6},
            reasoning="Verify solution makes sense",
            evidence_delta="Solution verified: values are positive and satisfy constraints",
            acquired_after=["P1", "P2", "P3", "P4"]
        ))

        # Step 5: Extract
        steps.append(RTTAStep(
            available=["P5"],
            chosen="P5",
            curiosity_scores={"P5": 0.4},
            reasoning="Format final answer",
            evidence_delta="Answer formatted for output",
            acquired_after=["P1", "P2", "P3", "P4", "P5"]
        ))

        return steps

    def _build_logic_steps(self, details: Dict) -> List[RTTAStep]:
        """Build curiosity loop steps for logic problem."""
        steps = []

        # Step 1: Parse
        if "premises" in details:
            premise_str = "; ".join(details["premises"])
        elif "premise" in details:
            premise_str = details["premise"]
        else:
            premise_str = "identified logical structure"

        steps.append(RTTAStep(
            available=["P1"],
            chosen="P1",
            curiosity_scores={"P1": 0.9},
            reasoning="Must understand the logical form first",
            evidence_delta=f"Premises: {premise_str}",
            acquired_after=["P1"]
        ))

        # Step 2: Apply rules
        if "reasoning" in details:
            rule_str = details["reasoning"]
        elif "pattern" in details:
            rule_str = f"Pattern identified: {details['pattern']}"
        else:
            rule_str = "applied inference rules"

        steps.append(RTTAStep(
            available=["P2"],
            chosen="P2",
            curiosity_scores={"P2": 0.85},
            reasoning="Apply logical inference",
            evidence_delta=rule_str,
            acquired_after=["P1", "P2"]
        ))

        # Step 3: Conclude
        steps.append(RTTAStep(
            available=["P3"],
            chosen="P3",
            curiosity_scores={"P3": 0.7},
            reasoning="Derive final conclusion",
            evidence_delta="Conclusion derived from premises",
            acquired_after=["P1", "P2", "P3"]
        ))

        return steps

    def _build_math_explanation(self, details: Dict, answer: str) -> str:
        """Build explanation for math problem."""
        parts = []

        if "variables" in details:
            parts.append(f"Let {', '.join(f'{k} = {v}' for k, v in details['variables'].items())}.")

        if "equations" in details:
            parts.append(f"From the problem: {' and '.join(details['equations'])}.")

        if "calculation" in details:
            parts.append(f"Solving: {details['calculation']}.")

        parts.append(f"Therefore, the answer is {answer}.")

        return " ".join(parts)

    def _build_logic_explanation(self, details: Dict, answer: str) -> str:
        """Build explanation for logic problem."""
        parts = []

        if "premises" in details:
            parts.append(f"Given: {' and '.join(details['premises'])}.")

        if "reasoning" in details:
            parts.append(f"By {details['reasoning']}.")

        if "reason" in details:
            parts.append(f"The answer is {answer} because it is {details['reason']}.")
        else:
            parts.append(f"Therefore, the answer is {answer}.")

        return " ".join(parts)


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_rtta_dataset(n_examples: int = 1000, output_path: str = "data/rtta_training.jsonl"):
    """Generate RTTA training dataset."""

    math_gen = MathWordProblemGenerator()
    logic_gen = LogicProblemGenerator()
    formatter = RTTAFormatter()

    examples = []

    for i in range(n_examples):
        # Alternate between math and logic
        if i % 2 == 0:
            question, answer, details = math_gen.generate()
            if question:
                example = formatter.format_math_problem(question, answer, details)
        else:
            question, answer, details = logic_gen.generate()
            if question:
                example = formatter.format_logic_problem(question, answer, details)

        if question:
            # Convert to serializable format
            ex_dict = {
                "question": example.question,
                "goal": asdict(example.goal),
                "rpm": [asdict(p) for p in example.rpm],
                "desire": example.desire,
                "wisdom": example.wisdom,
                "power": example.power,
                "steps": [asdict(s) for s in example.steps],
                "final_answer": example.final_answer,
                "explanation": example.explanation
            }
            examples.append(ex_dict)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_examples} examples")

    # Also create a simpler input/output format for training
    simple_examples = []
    for ex in examples:
        # Format as structured reasoning trace
        trace = format_as_training_text(ex)
        simple_examples.append({
            "input": ex["question"],
            "output": trace
        })

    # Save both formats
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Full RTTA format
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    # Simple training format
    simple_path = output_path.replace('.jsonl', '_simple.jsonl')
    with open(simple_path, 'w', encoding='utf-8') as f:
        for ex in simple_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved {len(examples)} examples to {output_path}")
    print(f"Saved simple format to {simple_path}")

    return examples


def format_as_training_text(ex: Dict) -> str:
    """Format RTTA example as training text."""
    lines = []

    # Goal
    g = ex["goal"]
    lines.append(f"Goal: Resolve {g['foundation']} question in {'/'.join(g['worlds'])} domain")
    lines.append("")

    # RPM
    lines.append("Prerequisites:")
    for p in ex["rpm"]:
        deps = f" (requires: {', '.join(p['depends_on'])})" if p['depends_on'] else ""
        lines.append(f"  {p['id']}: {p['name']}{deps}")
    lines.append("")

    # Curiosity loop
    lines.append("Reasoning steps:")
    for i, step in enumerate(ex["steps"], 1):
        lines.append(f"  Step {i}: Choose {step['chosen']} (curiosity={step['curiosity_scores'].get(step['chosen'], 0):.2f})")
        lines.append(f"    Reasoning: {step['reasoning']}")
        lines.append(f"    Result: {step['evidence_delta']}")
    lines.append("")

    # Answer
    lines.append(f"Final Answer: {ex['final_answer']}")
    lines.append(f"Explanation: {ex['explanation']}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Number of examples")
    parser.add_argument("--output", default="data/rtta_training.jsonl", help="Output path")
    args = parser.parse_args()

    generate_rtta_dataset(args.n, args.output)

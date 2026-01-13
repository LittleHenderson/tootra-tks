"""
TKS v7 Diversified Data Generator - Project MAJIK (v7.4 Definitive)
Generates diversified samples without "Trace:" / "Output:" scaffolding.
"""
import argparse
import json
import os
import random
from typing import Dict

# =============================================================================
# ATOMIC COMPONENTS
# =============================================================================
WORLDS = ["A", "B", "C", "D"]
NOETICS = ["nu0", "nu1", "nu2", "nu3", "nu4", "nu5", "nu6", "nu7", "nu8", "nu9"]
ELEMENTS = [f"{w}{i}" for w in WORLDS for i in range(10)]
FOUNDATIONS = [
    "Association",
    "Wisdom",
    "Life",
    "Companionship",
    "Power",
    "Wealth",
    "Continuation",
]
PREREQS = [
    "D1", "W1", "P1",
    "D2", "W2", "P2",
    "D3", "W3", "P3",
    "D4", "W4", "P4",
    "D5", "W5", "P5",
    "D6", "W6", "P6",
    "D7", "W7", "P7",
]

NOUNS = [
    "Attractor",
    "Fractal",
    "Consciousness",
    "Flow",
    "System",
    "Engine",
    "Processor",
    "Window",
    "Resonance",
    "Cascade",
]
VERBS = [
    "analyze",
    "manifest",
    "calculate",
    "derive",
    "structure",
    "regulate",
    "observe",
    "project",
]

# =============================================================================
# PROCEDURAL GENERATORS
# =============================================================================
def gen_oop() -> Dict[str, str]:
    name = random.choice(NOUNS) + random.choice(NOUNS)
    field = random.choice(["value", "state", "energy", "index", "intensity"])
    method = random.choice(VERBS)
    target = random.choice(ELEMENTS)

    code = (
        f"blueprint {name} {{\n"
        "  specifics {\n"
        f"    {field}: Ordinal;\n"
        "  }\n"
        "  actions {\n"
        f"    {method}(identity): Unit = {random.choice(NOETICS)}({target});\n"
        "  }\n"
        "}"
    )
    return {
        "input": f"Implement a TKS blueprint named {name} that uses {method} "
                 f"to affect {target}.",
        "trace": f"RECURSE -> Define blueprint {name}. Sub-goal: Set internal "
                 f"composition. RESOLVE.",
        "output": code,
    }


def gen_dsl() -> Dict[str, str]:
    bound = random.choice(["omega", "ord(alpha)", "limit", "omega+1"])
    start = random.choice(ELEMENTS)
    step_nu = random.choice(NOETICS)

    code = (
        f"transfinite loop i < {bound}\n"
        f"from {start}\n"
        f"step (x -> {step_nu}(x))\n"
        "limit (f -> stabilize(f))"
    )
    return {
        "input": f"Write a transfinite loop bounded by {bound} starting from "
                 f"{start}.",
        "trace": f"RECURSE -> Identify transfinite boundary. Sub-goal: Define "
                 f"step function {step_nu}. RESOLVE.",
        "output": code,
    }


def gen_rpm() -> Dict[str, str]:
    goal = f"{random.choice(VERBS)} {random.choice(FOUNDATIONS)}"
    block = random.choice(PREREQS)

    return {
        "input": f"Analyze the RPM requirements for the goal: {goal}.",
        "trace": f"RECURSE -> Scan prerequisites for {goal}. BLOCK DETECTED: "
                 f"{block} is Jack of Clubs. Sub-goal: Acquire {block}. RESOLVE.",
        "output": (
            f"The goal '{goal}' requires acquisition of {block} before full "
            "manifestation can occur in D-World."
        ),
    }


def build_entry(sample: Dict[str, str], category: str) -> Dict[str, str]:
    input_text = sample["input"]
    if category in {"OOP", "DSL"}:
        input_text = f"{input_text} Respond with TKS DSL only."
    else:
        input_text = f"{input_text} Respond with a concise statement."

    return {
        "input": input_text,
        "output": sample["output"],
        "trace": sample["trace"],
        "category": category,
    }


def generate_dataset(num_samples: int, output_path: str) -> None:
    print(f"Generating {num_samples} diversified TKS samples...")
    data = []

    for _ in range(num_samples):
        choice = random.random()
        if choice < 0.33:
            sample = gen_oop()
            category = "OOP"
        elif choice < 0.66:
            sample = gen_dsl()
            category = "DSL"
        else:
            sample = gen_rpm()
            category = "RPM"

        data.append(build_entry(sample, category))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--output", default="data/v7_diversified_training.jsonl")
    args = parser.parse_args()
    generate_dataset(args.num_samples, args.output)

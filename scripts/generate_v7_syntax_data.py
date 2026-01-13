"""
Generate nested-operator DSL samples for syntax-focused finetuning.
Avoids "Trace:" / "Output:" scaffolding to prevent looping attractors.
"""
import argparse
import json
import os
import random
from typing import Dict

WORLDS = ["A", "B", "C", "D"]
ELEMENTS = [f"{w}{i}" for w in WORLDS for i in range(1, 11)]
OPERATORS = ["+T", "-T", "*T", "/T", "->", "<-", "o"]
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


def rand_elem() -> str:
    return random.choice(ELEMENTS)


def build_expr(depth: int) -> str:
    if depth <= 0:
        return rand_elem()
    left = build_expr(depth - 1)
    right = build_expr(depth - 1)
    op = random.choice(OPERATORS)
    return f"({left} {op} {right})"


def nested_expr(min_depth: int = 2, max_depth: int = 4) -> str:
    depth = random.randint(min_depth, max_depth)
    return build_expr(depth)


def gen_reasoning() -> Dict[str, str]:
    intent = f"{random.choice(VERBS)} {random.choice(NOUNS)}"
    prompt = f"Goal: {intent} <SEP> Reasoning:"
    expr = nested_expr()
    return {"input": prompt, "output": expr}


def gen_blueprint() -> Dict[str, str]:
    name = random.choice(NOUNS) + random.choice(NOUNS)
    method = random.choice(VERBS)
    expr = nested_expr()
    code = (
        f"blueprint {name} {{\n"
        "  specifics {\n"
        "    seed: Ordinal;\n"
        "  }\n"
        "  actions {\n"
        f"    {method}(identity): Unit = {expr};\n"
        "  }\n"
        "}"
    )
    prompt = f"Write a TKS DSL blueprint for {name}. Respond with TKS DSL only."
    return {"input": prompt, "output": code}


def gen_loop() -> Dict[str, str]:
    bound = random.choice(["omega", "ord(alpha)", "limit", "omega+1"])
    start = rand_elem()
    expr = nested_expr()
    code = (
        f"transfinite loop i < {bound}\n"
        f"from {start}\n"
        f"step (x -> {expr})\n"
        "limit (f -> stabilize(f))"
    )
    prompt = (
        f"Create a transfinite loop bounded by {bound} starting from {start}. "
        "Respond with TKS DSL only."
    )
    return {"input": prompt, "output": code}


def gen_chain_statement() -> Dict[str, str]:
    expr = nested_expr()
    prompt = "Compose a nested operator chain with at least three operators."
    return {"input": prompt, "output": expr}


def generate_dataset(num_samples: int, output_path: str) -> None:
    print(f"Generating {num_samples} syntax samples...")
    rows = []
    for _ in range(num_samples):
        choice = random.random()
        if choice < 0.4:
            rows.append(gen_reasoning())
        elif choice < 0.7:
            rows.append(gen_blueprint())
        elif choice < 0.9:
            rows.append(gen_loop())
        else:
            rows.append(gen_chain_statement())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--output", default="data/v7_syntax_training.jsonl")
    args = parser.parse_args()
    generate_dataset(args.num_samples, args.output)

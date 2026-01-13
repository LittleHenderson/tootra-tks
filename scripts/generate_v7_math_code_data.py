"""
TKS v7 Math, Code, DSL & OOP Data Generator - Project MAJIK (v7.4 Definitive)

Generates 10,000 samples covering:
1. Standard Math & Python Code
2. TKS v7.4 Extended DSL Grammar & VM Instructions
3. TKS Hybrid OOP (Blueprint, Specifics, Details, Actions)
4. Type System & Algebraic Effect Handlers
"""

import json
import random
import os
from typing import List, Dict

# =============================================================================
# DSL & OOP SPECIFICATIONS (From Handoff & Docs)
# =============================================================================

TKS_DSL_KEYWORDS = [
    "omega", "ord", "limit", "succ", "transfinite",  # Transfinite
    "effect", "handle", "with", "resume", "perform", "handler", "op", # Effects
    "measure", "superpose", "entangle", "qstate", "amplitude", # Quantum
    "blueprint", "plan", "specifics", "description", "details", "actions", "identity", "repeat", "mut" # OOP
]

TKS_VM_OPS = [
    "PUSH_ELEMENT", "PUSH_FOUNDATION", "ELEMENT_WING", "FOUNDATION_LEVEL",
    "APPLY_NOETIC", "NOETIC_COMPOSE", "MAKE_FRACTAL", "ACBE_DESCEND",
    "INSTALL_HANDLER", "PERFORM_EFFECT", "TRANS_LOOP_INIT", "ORD_ADD",
    "MAKE_KET", "SUPERPOSE", "MEASURE", "ENTANGLE"
]

OOP_EXAMPLES = [
    {
        "input": "Define a TKS Counter class using hybrid OOP syntax.",
        "trace": "RECURSE -> Use 'blueprint' for class definition. Sub-goal: Define fields in 'specifics'. CHECK 11W(Grammar) = 0.9. Sub-goal: Define methods in 'actions' with 'identity'. RESOLVE: blueprint Counter { specifics { value: Ordinal; } actions { inc(identity): Counter = repeat Counter { value: succ(identity.value) }; } }",
        "output": "```tks\nblueprint Counter {\n  specifics {\n    value: Ordinal;\n  }\n  actions {\n    inc(identity): Counter = repeat Counter { value: succ(identity.value) };\n  }\n}```",
        "category": "OOP"
    },
    {
        "input": "How do you call a constructor in TKS Hybrid OOP?",
        "trace": "RECURSE -> Identify constructor keyword. RESOLVE: The 'repeat' keyword is used followed by the class name and braced field initializers.",
        "output": "In TKS Hybrid OOP, constructors use the 'repeat' keyword: `repeat ClassName { field1: value1, field2: value2 }`.",
        "category": "OOP"
    }
]

DSL_EXAMPLES = [
    {
        "input": "Write a TKS transfinite loop that computes the omega-limit of a noetic application.",
        "trace": "RECURSE -> Use 'transfinite loop' syntax. Sub-goal: Set loop bound to 'omega'. CHECK 11W(Transfinite) = 0.9. RESOLVE: transfinite loop i < omega from A1 step f limit g",
        "output": "```tks\ntransfinite loop i < omega\nfrom A1\nstep (x -> nu3(x))\nlimit (f -> limit_noetic(f))```",
        "category": "DSL"
    },
    {
        "input": "Define an algebraic effect for logging in TKS.",
        "trace": "RECURSE -> Use 'effect' keyword. Sub-goal: Define operations with types. RESOLVE: effect Log { op log(msg: String): Unit; }",
        "output": "```tks\neffect Log {\n  op log(msg: String): Unit;\n}```",
        "category": "DSL"
    }
]

VM_EXAMPLES = [
    {
        "input": "Translate 'nu2(A3)' into TKS VM Bytecode.",
        "trace": "RECURSE -> Identify elements. Sub-goal: PUSH_ELEMENT A 3. Sub-goal: PUSH_NOETIC 2. Sub-goal: APPLY_NOETIC. RESOLVE: PUSH_ELEMENT A 3; PUSH_NOETIC 2; APPLY_NOETIC",
        "output": "Bytecode:\n0x40 PUSH_ELEMENT A 3\n0x50 PUSH_NOETIC 2\n0x51 APPLY_NOETIC",
        "category": "VM"
    }
]

MATH_LOGIC = [
    {
        "input": "What is the Hausdorff Dimension of an omega-eigenfractal with 3 maps and ratio 1/2?",
        "trace": "RECURSE -> Apply dim_H formula. Sub-goal: dim_H = log N / log(1/r). N=3, r=0.5. RESOLVE: log(3)/log(2) approx 1.585.",
        "output": "The Hausdorff Dimension is log(3)/log(2) ≈ 1.585.",
        "category": "Math"
    }
]

# =============================================================================
# GENERATOR
# =============================================================================

def generate_data(num_samples=10000):
    print(f"Generating {num_samples} Math, Code, DSL & OOP samples...")
    data = []
    
    # Probabilities for categories
    categories = ["OOP", "DSL", "VM", "Math", "Python"]
    weights = [0.25, 0.25, 0.15, 0.15, 0.20]
    
    # Base Python/Math templates for filler
    py_templates = [
        ("Write a Python function to calculate factorial.", "def factorial(n):\n    return 1 if n < 2 else n * factorial(n-1)"),
        ("Write a Python class for a Point(x,y).", "class Point:\n    def __init__(self, x, y): self.x = x; self.y = y")
    ]

    for i in range(num_samples):
        cat = random.choices(categories, weights=weights)[0]
        
        if cat == "OOP":
            base = random.choice(OOP_EXAMPLES)
        elif cat == "DSL":
            base = random.choice(DSL_EXAMPLES)
        elif cat == "VM":
            base = random.choice(VM_EXAMPLES)
        elif cat == "Math":
            base = random.choice(MATH_LOGIC)
        else: # Python
            t = random.choice(py_templates)
            base = {
                "input": t[0],
                "trace": "RECURSE -> Standard Algorithm -> CHECK 12P(Coding) = 0.9. RESOLVE.",
                "output": t[1],
                "category": "Python"
            }
            
        sample = {
            "input": base["input"],
            "output": f"Step 1: Analyzing {base['category']} task...\nTrace: {base['trace']}\nOutput:\n{base['output']}",
            "tks_notation": base['trace'],
            "category": base['category']
        }
        data.append(sample)
        
    os.makedirs("data", exist_ok=True)
    out_path = "data/v7_math_code_training.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(data)} samples to {out_path}")

if __name__ == "__main__":
    generate_data()
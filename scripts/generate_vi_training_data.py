#!/usr/bin/env python3
"""
Generate Training Data for V (Validity) and I (Impact) Networks

This creates labeled data to train the DPS novelty components so that:
- V outputs HIGH for valid/correct reasoning, LOW for invalid
- I outputs HIGH for impactful/breakthrough thoughts, LOW for trivial

Training Data Sources:
1. Valid reasoning traces from TKS training data (V=HIGH)
2. Corrupted/invalid traces (V=LOW)
3. Key insight moments from reasoning chains (I=HIGH)
4. Repetitive/trivial steps (I=LOW)
"""

import sys
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Seed for reproducibility
random.seed(42)

# =============================================================================
# VALIDITY TRAINING DATA
# =============================================================================

# Patterns that indicate VALID reasoning (V should be HIGH)
VALID_PATTERNS = [
    # Correct TKS notation
    "CHECK {node} = {value} -> PASS",
    "RESOLVE {node} = {value} -> RESOLVED",
    "RECURSE -> Sub-goal: {subgoal}",
    "BLOCK at {node}. Sub-goal needed: {need}",
    # Correct logical structures
    "Step {n}: {action} because {reason}",
    "Therefore, {conclusion}",
    "Given {premise}, we can infer {inference}",
    # Correct mathematical operations
    "Let x = {value}, then f(x) = {result}",
    "Since {a} > {b}, we have {conclusion}",
]

# Patterns that indicate INVALID reasoning (V should be LOW)
INVALID_PATTERNS = [
    # Contradictions
    "CHECK {node} = {value} -> PASS ... CHECK {node} = {other} -> PASS",
    "Therefore {a} and therefore not {a}",
    # Circular reasoning
    "Because {x} is true, {x} must be true",
    "We know {a} because {b}, and {b} because {a}",
    # Non-sequiturs
    "Step 1: {action}. Step 2: [unrelated action]. Conclusion: {unrelated}",
    # Missing evidence
    "It is obvious that {claim}",
    "Everyone knows {claim}",
    # Invalid TKS
    "CHECK INVALID_NODE -> ???",
    "BLOCK at nowhere",
]

# High-impact patterns (I should be HIGH)
HIGH_IMPACT_PATTERNS = [
    # Breakthroughs
    "KEY INSIGHT: {insight}",
    "This reveals that {discovery}",
    "The critical observation is {observation}",
    # Problem resolution
    "RESOLVE {node} -> Goal achieved",
    "Solution found: {solution}",
    # Efficiency gains
    "Simplifies to {simplified}",
    "Can be reduced to {reduced}",
]

# Low-impact patterns (I should be LOW)
LOW_IMPACT_PATTERNS = [
    # Repetition
    "As mentioned before, {repeated}",
    "Again, {repeated}",
    # Trivial observations
    "Note that {obvious}",
    "Clearly, {trivial}",
    # No progress
    "BLOCK at {node} -> BLOCK at {node} -> BLOCK at {node}",
    "Checking... still checking... still checking...",
]


def generate_validity_examples(n: int = 1000) -> List[Dict]:
    """Generate labeled validity training examples."""
    examples = []

    # TKS reasoning nodes
    nodes = [
        "1D (Spiritual Desire)", "2W (Spiritual Wisdom)", "3P (Spiritual Power)",
        "4D (Emotional Desire)", "5W (Mental Wisdom)", "6P (Mental Power)",
        "7D (Emotional Desire)", "8W (Emotional Wisdom)", "9P (Emotional Power)",
        "10D (Physical Desire)", "11W (Physical Wisdom)", "12P (Physical Power)",
    ]

    subgoals = [
        "Develop spiritual discipline", "Build emotional resilience",
        "Acquire knowledge", "Strengthen willpower", "Cultivate wisdom",
        "Understand the problem domain", "Gather resources",
    ]

    # Generate VALID examples (V = HIGH)
    for _ in range(n // 2):
        node = random.choice(nodes)
        value = round(random.uniform(0.5, 0.95), 2)
        subgoal = random.choice(subgoals)

        # Build valid reasoning trace
        trace_parts = [
            f"Goal: {random.choice(['achieve wisdom', 'gain power', 'fulfill desire'])}",
            f"Step 1: CHECK {node} = {value} [STRONG] -> PASS",
        ]

        # Add more valid steps
        if random.random() > 0.5:
            node2 = random.choice([n for n in nodes if n != node])
            trace_parts.append(f"Step 2: CHECK {node2} = {round(random.uniform(0.3, 0.7), 2)} -> PASS")

        if random.random() > 0.3:
            trace_parts.append(f"Step 3: RECURSE -> Sub-goal: {subgoal}")
            trace_parts.append(f"Step 4: RESOLVE {node} = {round(value + 0.1, 2)} -> RESOLVED")

        trace = " | ".join(trace_parts)

        examples.append({
            "text": trace,
            "valid": True,
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "consistency": round(random.uniform(0.75, 0.98), 2),
            "evidence_ok": round(random.uniform(0.8, 1.0), 2),
        })

    # Generate INVALID examples (V = LOW)
    for _ in range(n // 2):
        node = random.choice(nodes)

        # Create invalid reasoning
        invalid_type = random.choice(["contradiction", "circular", "missing", "nonsense"])

        if invalid_type == "contradiction":
            trace = f"CHECK {node} = 0.8 -> PASS | CHECK {node} = 0.2 -> PASS | Contradiction!"
        elif invalid_type == "circular":
            trace = f"Because {node} is strong, {node} must be strong. QED."
        elif invalid_type == "missing":
            trace = f"Obviously {node} is sufficient. No evidence needed."
        else:
            trace = f"BLORK at FLARB -> ZZZZ -> INVALID SYNTAX {random.randint(0,99)}"

        examples.append({
            "text": trace,
            "valid": False,
            "confidence": round(random.uniform(0.05, 0.35), 2),
            "consistency": round(random.uniform(0.1, 0.4), 2),
            "evidence_ok": round(random.uniform(0.0, 0.3), 2),
        })

    random.shuffle(examples)
    return examples


def generate_impact_examples(n: int = 1000) -> List[Dict]:
    """Generate labeled impact training examples."""
    examples = []

    insights = [
        "the recursive structure mirrors the goal hierarchy",
        "blockage at Power requires Wisdom sub-goal",
        "the pattern repeats across all four worlds",
        "desire must precede power in the causal chain",
    ]

    trivial = [
        "checking the next node",
        "continuing the process",
        "still working on it",
        "more of the same",
    ]

    # HIGH IMPACT examples
    for _ in range(n // 2):
        insight = random.choice(insights)

        trace = random.choice([
            f"KEY INSIGHT: {insight} | This changes the approach entirely.",
            f"BREAKTHROUGH: After 5 recursive attempts, discovered {insight}",
            f"RESOLVE 2W -> Goal ACHIEVED. {insight}",
            f"Simplification found: {insight} reduces problem to base case.",
        ])

        examples.append({
            "text": trace,
            "is_breakthrough": True,
            "delta_reward": round(random.uniform(0.6, 0.95), 2),
            "delta_uncertainty": round(random.uniform(0.5, 0.9), 2),
            "delta_coverage": round(random.uniform(0.4, 0.8), 2),
            "delta_cost": round(random.uniform(0.3, 0.7), 2),
        })

    # LOW IMPACT examples
    for _ in range(n // 2):
        triv = random.choice(trivial)

        trace = random.choice([
            f"Step N: {triv}. Step N+1: {triv}. Step N+2: {triv}.",
            f"As mentioned before, {triv}. Repeating: {triv}.",
            f"BLOCK -> BLOCK -> BLOCK (no progress)",
            f"Clearly, {triv}. Obviously, {triv}.",
        ])

        examples.append({
            "text": trace,
            "is_breakthrough": False,
            "delta_reward": round(random.uniform(0.0, 0.25), 2),
            "delta_uncertainty": round(random.uniform(0.0, 0.2), 2),
            "delta_coverage": round(random.uniform(0.0, 0.15), 2),
            "delta_cost": round(random.uniform(0.0, 0.2), 2),
        })

    random.shuffle(examples)
    return examples


def generate_depth_benchmark(n: int = 200) -> List[Dict]:
    """
    Generate benchmark to verify depth correlates with reasoning quality.

    Contains problems with known "required depth" and expected quality.
    """
    examples = []

    # SHALLOW problems (depth 1-2 should suffice)
    shallow_problems = [
        ("What is 2+2?", 1, "4"),
        ("Is the sky blue?", 1, "yes"),
        ("Name a color", 1, "blue"),
        ("What comes after Monday?", 1, "Tuesday"),
    ]

    # MEDIUM problems (depth 3-4 needed)
    medium_problems = [
        ("If A implies B, and B implies C, does A imply C?", 3, "yes, by transitivity"),
        ("What is the derivative of x^2?", 3, "2x"),
        ("Find the pattern: 2, 4, 8, 16, ?", 3, "32 (powers of 2)"),
    ]

    # DEEP problems (depth 5+ needed)
    deep_problems = [
        ("Prove that sqrt(2) is irrational", 6, "proof by contradiction"),
        ("Explain recursive goal decomposition in TKS", 5, "hierarchical structure"),
        ("How does earned depth relate to genuine insight?", 7, "novelty must be validated"),
    ]

    # Generate examples
    for problem, depth, answer in shallow_problems:
        for _ in range(n // 12):
            examples.append({
                "problem": problem,
                "required_depth": depth,
                "expected_answer": answer,
                "category": "shallow",
            })

    for problem, depth, answer in medium_problems:
        for _ in range(n // 9):
            examples.append({
                "problem": problem,
                "required_depth": depth,
                "expected_answer": answer,
                "category": "medium",
            })

    for problem, depth, answer in deep_problems:
        for _ in range(n // 9):
            examples.append({
                "problem": problem,
                "required_depth": depth,
                "expected_answer": answer,
                "category": "deep",
            })

    random.shuffle(examples)
    return examples


def main():
    output_dir = Path("data/dps_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating V (Validity) training data...")
    validity_data = generate_validity_examples(2000)
    with open(output_dir / "validity_training.jsonl", "w") as f:
        for ex in validity_data:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(validity_data)} examples to validity_training.jsonl")

    print("\nGenerating I (Impact) training data...")
    impact_data = generate_impact_examples(2000)
    with open(output_dir / "impact_training.jsonl", "w") as f:
        for ex in impact_data:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(impact_data)} examples to impact_training.jsonl")

    print("\nGenerating depth verification benchmark...")
    benchmark_data = generate_depth_benchmark(300)
    with open(output_dir / "depth_benchmark.jsonl", "w") as f:
        for ex in benchmark_data:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(benchmark_data)} examples to depth_benchmark.jsonl")

    # Summary stats
    valid_count = sum(1 for x in validity_data if x["valid"])
    breakthrough_count = sum(1 for x in impact_data if x["is_breakthrough"])

    print("\n" + "="*50)
    print("DATA GENERATION COMPLETE")
    print("="*50)
    print(f"Validity: {valid_count} valid, {len(validity_data) - valid_count} invalid")
    print(f"Impact: {breakthrough_count} breakthroughs, {len(impact_data) - breakthrough_count} trivial")
    print(f"Benchmark: {len(benchmark_data)} problems across 3 depth categories")


if __name__ == "__main__":
    main()

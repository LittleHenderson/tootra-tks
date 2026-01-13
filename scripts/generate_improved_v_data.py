#!/usr/bin/env python3
"""
Generate IMPROVED V (Validity) Training Data

The previous data was too pattern-based. This version creates:
1. Structurally valid reasoning with logical flow
2. Structurally invalid reasoning with clear flaws
3. More diverse patterns so the network learns VALIDITY not just keywords

Key improvements:
- Valid examples have consistent node values and logical flow
- Invalid examples have SPECIFIC flaws (contradictions, circular, gaps)
- Mix of TKS notation and natural language reasoning
- Explicit markers of validity/invalidity in structure
"""

import sys
import os
import json
import random
from pathlib import Path
from typing import List, Dict

random.seed(42)

# TKS nodes with their meanings
NODES = {
    "1D": ("Spiritual Desire", "divine"),
    "2W": ("Spiritual Wisdom", "understanding"),
    "3P": ("Spiritual Power", "discipline"),
    "4D": ("Emotional Desire", "passion"),
    "5W": ("Mental Wisdom", "knowledge"),
    "6P": ("Mental Power", "focus"),
    "7D": ("Emotional Desire", "motivation"),
    "8W": ("Emotional Wisdom", "empathy"),
    "9P": ("Emotional Power", "courage"),
    "10D": ("Physical Desire", "ambition"),
    "11W": ("Physical Wisdom", "practical knowledge"),
    "12P": ("Physical Power", "action"),
}

def generate_valid_tks_trace() -> str:
    """Generate structurally valid TKS reasoning trace."""
    node_keys = list(NODES.keys())

    # Pick a goal and consistent path
    goal_node = random.choice(node_keys)
    goal_name, goal_concept = NODES[goal_node]

    # Generate consistent values that make sense
    base_value = round(random.uniform(0.6, 0.95), 2)

    templates = [
        # Template 1: Clean pass-through
        f"Goal: Achieve {goal_concept}\n"
        f"Step 1: CHECK {goal_node} ({goal_name}) = {base_value} [STRONG] -> PASS\n"
        f"Step 2: Sufficient capacity confirmed\n"
        f"Step 3: RESOLVE {goal_node} -> GOAL ACHIEVED\n"
        f"Conclusion: {goal_concept.capitalize()} attained through valid path",

        # Template 2: Recurse and resolve
        f"Goal: Develop {goal_concept}\n"
        f"Step 1: CHECK {goal_node} ({goal_name}) = {round(base_value * 0.4, 2)} [INSUFFICIENT]\n"
        f"Step 2: BLOCK at {goal_node} - need more capacity\n"
        f"Step 3: RECURSE -> Sub-goal: Build foundation\n"
        f"Step 4: Foundation built, returning\n"
        f"Step 5: CHECK {goal_node} = {base_value} [STRONG] -> PASS\n"
        f"Step 6: RESOLVE {goal_node} -> SUCCESS",

        # Template 3: Multi-node consistent
        f"Reasoning chain for {goal_concept}:\n"
        f"Premise 1: {goal_node} requires {goal_name}\n"
        f"Premise 2: Current {goal_node} = {base_value}\n"
        f"Premise 3: Threshold for PASS is 0.5\n"
        f"Since {base_value} > 0.5, {goal_node} is SUFFICIENT\n"
        f"Therefore: Can proceed with {goal_concept}\n"
        f"QED: Valid reasoning confirmed",

        # Template 4: Evidence-backed
        f"Claim: {goal_concept} is achievable\n"
        f"Evidence 1: {goal_node} measured at {base_value}\n"
        f"Evidence 2: Historical success rate at this level: 87%\n"
        f"Evidence 3: No blocking conditions detected\n"
        f"Inference: All prerequisites met\n"
        f"Conclusion: Claim VALIDATED with evidence",

        # Template 5: Logical chain
        f"If {goal_node} >= 0.5 then PROCEED\n"
        f"Measured: {goal_node} = {base_value}\n"
        f"Check: {base_value} >= 0.5 is TRUE\n"
        f"Therefore: PROCEED is valid\n"
        f"Action: Execute {goal_concept} plan",
    ]

    return random.choice(templates)


def generate_invalid_tks_trace() -> str:
    """Generate structurally INVALID reasoning with specific flaws."""
    node_keys = list(NODES.keys())
    node = random.choice(node_keys)
    name, concept = NODES[node]

    flaw_type = random.choice([
        "contradiction", "circular", "non_sequitur",
        "missing_evidence", "invalid_inference", "value_mismatch"
    ])

    if flaw_type == "contradiction":
        # Same node, contradictory values
        v1 = round(random.uniform(0.7, 0.9), 2)
        v2 = round(random.uniform(0.1, 0.3), 2)
        return (
            f"Analysis of {concept}:\n"
            f"Step 1: CHECK {node} = {v1} [STRONG] -> PASS\n"
            f"Step 2: CHECK {node} = {v2} [WEAK] -> FAIL\n"
            f"CONTRADICTION: {node} cannot be both {v1} and {v2}\n"
            f"Conclusion: {concept} achieved\n"
            f"ERROR: Invalid - contradictory measurements"
        )

    elif flaw_type == "circular":
        return (
            f"Proof of {concept}:\n"
            f"Statement A: {node} is strong because {concept} is present\n"
            f"Statement B: {concept} is present because {node} is strong\n"
            f"Therefore: {concept} is proven\n"
            f"ERROR: Circular reasoning - A proves B proves A"
        )

    elif flaw_type == "non_sequitur":
        other_node = random.choice([n for n in node_keys if n != node])
        other_name, other_concept = NODES[other_node]
        return (
            f"Goal: Achieve {concept}\n"
            f"Step 1: CHECK {other_node} ({other_name}) = 0.8 -> PASS\n"
            f"Conclusion: Therefore {concept} is achieved\n"
            f"ERROR: Non-sequitur - {other_node} does not imply {node}"
        )

    elif flaw_type == "missing_evidence":
        return (
            f"Claim: {concept} is definitely achievable\n"
            f"Reasoning: It is obvious that {node} is sufficient\n"
            f"Proof: Everyone knows this to be true\n"
            f"Conclusion: {concept} confirmed\n"
            f"ERROR: No evidence provided - appeals to obviousness"
        )

    elif flaw_type == "invalid_inference":
        v = round(random.uniform(0.2, 0.4), 2)
        return (
            f"Goal: {concept}\n"
            f"Measurement: {node} = {v} [INSUFFICIENT]\n"
            f"Required: {node} >= 0.5\n"
            f"Check: {v} < 0.5 is TRUE\n"
            f"Conclusion: PROCEED anyway\n"
            f"ERROR: Invalid inference - conclusion contradicts premises"
        )

    else:  # value_mismatch
        v1 = round(random.uniform(0.3, 0.4), 2)
        v2 = round(random.uniform(0.8, 0.9), 2)
        return (
            f"Step 1: Initial {node} = {v1}\n"
            f"Step 2: No action taken\n"
            f"Step 3: Final {node} = {v2}\n"
            f"Claim: Improvement achieved\n"
            f"ERROR: Value jumped from {v1} to {v2} with no cause"
        )


def generate_valid_natural_reasoning() -> str:
    """Generate valid natural language reasoning."""
    templates = [
        # Modus ponens
        "If it rains, the ground gets wet.\n"
        "It is raining.\n"
        "Therefore, the ground is wet.\n"
        "VALID: Modus ponens correctly applied",

        # Syllogism
        "All mammals are warm-blooded.\n"
        "Dogs are mammals.\n"
        "Therefore, dogs are warm-blooded.\n"
        "VALID: Categorical syllogism",

        # Evidence-based
        "Observation: Temperature measured at 100C\n"
        "Principle: Water boils at 100C at sea level\n"
        "Location: Sea level confirmed\n"
        "Conclusion: Water is boiling\n"
        "VALID: Conclusion follows from evidence",

        # Mathematical
        "Given: x = 5\n"
        "Given: y = x + 3\n"
        "Therefore: y = 5 + 3 = 8\n"
        "VALID: Arithmetic correctly applied",

        # Conditional chain
        "If A then B. If B then C. A is true.\n"
        "Step 1: A is true (given)\n"
        "Step 2: Therefore B is true (from A->B)\n"
        "Step 3: Therefore C is true (from B->C)\n"
        "VALID: Hypothetical syllogism",
    ]
    return random.choice(templates)


def generate_invalid_natural_reasoning() -> str:
    """Generate invalid natural language reasoning."""
    templates = [
        # Affirming consequent
        "If it rains, the ground gets wet.\n"
        "The ground is wet.\n"
        "Therefore, it rained.\n"
        "INVALID: Affirming the consequent fallacy",

        # Hasty generalization
        "I met two rude people from City X.\n"
        "Therefore, everyone from City X is rude.\n"
        "INVALID: Hasty generalization from small sample",

        # False dichotomy
        "Either you support policy X or you hate the country.\n"
        "You don't support policy X.\n"
        "Therefore, you hate the country.\n"
        "INVALID: False dichotomy - other options exist",

        # Ad hominem
        "Person A claims the theory is correct.\n"
        "Person A has bad fashion sense.\n"
        "Therefore, the theory is incorrect.\n"
        "INVALID: Ad hominem - irrelevant to truth of claim",

        # Post hoc
        "Event A happened before Event B.\n"
        "Therefore, A caused B.\n"
        "INVALID: Post hoc ergo propter hoc fallacy",

        # Contradiction
        "Statement 1: X is always true.\n"
        "Statement 2: X is sometimes false.\n"
        "Conclusion: Both statements are correct.\n"
        "INVALID: Direct contradiction",
    ]
    return random.choice(templates)


def generate_dataset(n: int = 3000) -> List[Dict]:
    """Generate balanced validity dataset."""
    examples = []

    # 40% valid TKS traces
    for _ in range(int(n * 0.4)):
        text = generate_valid_tks_trace()
        examples.append({
            "text": text,
            "valid": True,
            "confidence": round(random.uniform(0.75, 0.95), 2),
            "consistency": round(random.uniform(0.8, 0.98), 2),
            "evidence_ok": round(random.uniform(0.85, 1.0), 2),
        })

    # 40% invalid TKS traces
    for _ in range(int(n * 0.4)):
        text = generate_invalid_tks_trace()
        examples.append({
            "text": text,
            "valid": False,
            "confidence": round(random.uniform(0.1, 0.35), 2),
            "consistency": round(random.uniform(0.05, 0.3), 2),
            "evidence_ok": round(random.uniform(0.0, 0.25), 2),
        })

    # 10% valid natural reasoning
    for _ in range(int(n * 0.1)):
        text = generate_valid_natural_reasoning()
        examples.append({
            "text": text,
            "valid": True,
            "confidence": round(random.uniform(0.8, 0.95), 2),
            "consistency": round(random.uniform(0.85, 0.98), 2),
            "evidence_ok": round(random.uniform(0.9, 1.0), 2),
        })

    # 10% invalid natural reasoning
    for _ in range(int(n * 0.1)):
        text = generate_invalid_natural_reasoning()
        examples.append({
            "text": text,
            "valid": False,
            "confidence": round(random.uniform(0.15, 0.4), 2),
            "consistency": round(random.uniform(0.1, 0.35), 2),
            "evidence_ok": round(random.uniform(0.05, 0.3), 2),
        })

    random.shuffle(examples)
    return examples


def main():
    output_dir = Path("data/dps_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating IMPROVED V (Validity) training data...")
    data = generate_dataset(4000)

    output_path = output_dir / "validity_training_v2.jsonl"
    with open(output_path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

    # Stats
    valid_count = sum(1 for x in data if x["valid"])
    invalid_count = len(data) - valid_count

    avg_valid_conf = sum(x["confidence"] for x in data if x["valid"]) / valid_count
    avg_invalid_conf = sum(x["confidence"] for x in data if not x["valid"]) / invalid_count

    print(f"\nSaved {len(data)} examples to {output_path}")
    print(f"  Valid: {valid_count} (avg conf: {avg_valid_conf:.2f})")
    print(f"  Invalid: {invalid_count} (avg conf: {avg_invalid_conf:.2f})")
    print(f"  Label separation: {avg_valid_conf - avg_invalid_conf:.2f}")

    # Show samples
    print("\n--- SAMPLE VALID ---")
    valid_sample = next(x for x in data if x["valid"])
    print(valid_sample["text"][:200] + "...")

    print("\n--- SAMPLE INVALID ---")
    invalid_sample = next(x for x in data if not x["valid"])
    print(invalid_sample["text"][:200] + "...")


if __name__ == "__main__":
    main()

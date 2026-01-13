"""
TKS v7 Integrated Reasoning Data Generator - Project MAJIK (v7.4 Definitive)

Generates 10,000 samples of "Chain-of-Thought" style training data.
Format:
Input: [Complex TKS Task]
Output: Trace: [Concise Logic Steps] -> RESOLVE: [Final Coherent Answer]

This bridges the gap between deep recursion and natural language fluency.
"""

import json
import random
import os

TASKS = [
    {
        "prompt": "Design a Noetic Fractal for anxiety reduction.",
        "trace": "RECURSE -> Identify anxiety (C3). Sub-goal: Counter with C2/A1. Pattern: Awareness (1) -> Vibration (4) -> Release (9).",
        "answer": "To reduce anxiety (C3), use the fractal ⟨1:4:9⟩. Mind (1) observes the feeling, Vibration (4) harmonizes the intensity, and Below (9) grounds the energy into physical relaxation."
    },
    {
        "input": "Explain how the ACBE functor maps to the 40 Elements.",
        "trace": "RECURSE -> ACBE descends A->C->B->E (D). Sub-goal: Map Worlds. A=Spiritual, B=Mental, C=Emotional, D=Physical. Elements are intersections.",
        "answer": "The ACBE functor maps the flow of creation through the 4 Worlds: Atziluth (A) to Yetzirah (C) to Briah (B) to Assiah (D). This generates the 40 Elements by applying the 10 Noetics within each World layer."
    },
    {
        "input": "Derive the spectral dimension of a Sierpinski fractal.",
        "trace": "RECURSE -> Use d_S formula: 2*log(N)/log(1/rho). N=3, rho=1/5 (energy scaling). Sub-goal: Compute.",
        "answer": "For a Sierpinski fractal with N=3 self-similar copies and energy scaling factor ρ=1/5, the spectral dimension is d_S = 2*log(3)/log(5) ≈ 1.365."
    },
    {
        "input": "Write a TKS DSL blueprint for a 'WealthAccumulator'.",
        "trace": "RECURSE -> Class structure. Sub-goal: Fields (resources). Sub-goal: Methods (invest). Use 'blueprint' syntax.",
        "answer": "```tks\nblueprint WealthAccumulator {\n  specifics {\n    assets: List[Resource];\n    rate: Float;\n  }\n  actions {\n    invest(identity): Unit = perform acquire(identity.assets);\n  }\n}```"
    },
    {
        "input": "Analyze the 'Social Weather' of a viral meme.",
        "trace": "RECURSE -> Meme components. Sub-goal: Resonance (4). Sub-goal: Spread (7). Pattern: ⟨2:4:7⟩.",
        "answer": "A viral meme typically exhibits a ⟨2:4:7⟩ fractal pattern. Positive Affinity (2) hooks attention, High Vibration (4) creates emotional impact, and Rhythm (7) ensures rapid propagation through the social graph."
    }
]

def generate_data(num_samples=10000):
    print(f"Generating {num_samples} Integrated Reasoning samples...")
    data = []
    
    for i in range(num_samples):
        base = random.choice(TASKS)
        
        # Add slight variations to prompt to prevent exact memorization
        prompt_variations = [
            base.get("input", base.get("prompt")),
            f"Task: {base.get('input', base.get('prompt'))}",
            f"TKS Query: {base.get('input', base.get('prompt'))}",
            f"Please {base.get('input', base.get('prompt')).lower()}"
        ]
        
        prompt = random.choice(prompt_variations)
        
        entry = {
            "input": prompt,
            "output": f"Trace: {base['trace']} -> RESOLVE:\n{base['answer']}",
            "category": "Integrated"
        }
        data.append(entry)
        
    os.makedirs("data", exist_ok=True)
    out_path = "data/v7_integrated_training.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(data)} samples to {out_path}")

if __name__ == "__main__":
    generate_data()

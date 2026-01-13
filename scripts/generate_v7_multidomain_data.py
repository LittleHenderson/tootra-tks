"""
TKS v7 Multi-Domain Data Generator - Project MAJIK (v7.4 Definitive)

Includes:
- Psychology (Weaponized Therapy/Profiling)
- Social Engineering (Social Weather Systems)
- Cultural Engineering (Overton/Hyperstition)
- Computer Science, Business, History, Science
"""

import json
import random
import os
from typing import List, Dict

# =============================================================================
# DEFINITIVE TKS DOMAIN DATA (From Manuscripts)
# =============================================================================

TKS_DOMAINS = {
    "Psychology": [
        {
            "input": "Perform a psychological profile extraction for a subject showing high aversion to risk.",
            "trace": "RECURSE -> Sub-goal: Map text to Noetics. CHECK 1W(Mind) = 0.9. RECURSE -> Sub-goal: Identify dominant Negative (3) affinity. CHECK 3P(Aversion) = 0.9. RECURSE -> Sub-goal: Calculate Lacunarity. RESOLVE: Subject shows high lacunarity (Lambda > 1.8) in P9 (Effect) regions. Profile Φ = ⟨3:5:7⟩.",
            "output": "The subject displays a ⟨3:5:7⟩ (Negative-Female-Rhythm) fractal pattern. High lacunarity indicates a vulnerability in the power cycle, specifically regarding physical implementation of plans."
        },
        {
            "input": "Define 'Weaponized Therapy' in the TKS framework.",
            "trace": "RECURSE -> Access Cognitive Framework. Sub-goal: Define intervention fractal. RESOLVE: Weaponized Therapy is the engineering of a counter-fractal Φ_manip = Φ_destab ∘ Φ_guide to shift a subject's psychological attractor.",
            "output": "Weaponized Therapy is the application of TKS fractals to dismantle limiting internal structures (Noetic 5) and install new behavioral attractors via the Law of Association (F1)."
        }
    ],
    "Social Engineering": [
        {
            "input": "Model a social meme cascade for a new cultural concept.",
            "trace": "RECURSE -> Define viral signature. CHECK 2W(Affinity) = 0.8. RECURSE -> Apply dρ/dt cascade equation. RESOLVE: Signature ⟨2:4:7⟩ (Positive-Vibration-Rhythm) ensures high propagation. Critical mass θ_crit = 0.15.",
            "output": "The social weather system predicts a cascade if the meme achieves θ_crit of 15% adoption using a ⟨2:4:7⟩ resonance pattern, triggering a self-sustaining adoption cycle."
        },
        {
            "input": "How is the Overton Window shifted using TKS?",
            "trace": "RECURSE -> Define window O(t). Sub-goal: Apply transfinite sequence Φ_shift. RESOLVE: By iterating a transfinite sequence of Ideas x_α where the distance d(x_α, x_{α+1}) is below the normalization threshold, the window O shifts toward the target.",
            "output": "Overton shifting is achieved via a transfinite fractal sequence that incrementally moves the acceptance threshold (θ_norm) through subtle, repeated noetic updates."
        }
    ],
    "Hyperstition": [
        {
            "input": "Explain the mechanics of a hyperstition.",
            "trace": "RECURSE -> Sub-goal: Define self-fulfillment condition. CHECK 8W(Causality) = 0.9. RESOLVE: A hyperstition is an Idea-structure where M(B_t) > M(B_{t-1}) ⟹ B_{t+1} > B_t. The story makes itself real.",
            "output": "Hyperstitions are 'self-writing stories' that function as feedback loops: the more an Idea (10) is processed as real (1), the more concrete its manifestation (9) becomes in D-World."
        }
    ],
    "Computer Science": [
        {
            "input": "How does TKS solve the AI interpretability problem?",
            "trace": "RECURSE -> Sub-goal: Map decision path to Noetic Algebra. RESOLVE: Every decision path π admits an interpretation Interpret(π) = ⟨ν_k1, ..., ν_kn, A_Φ⟩, exposing the exact noetic operators that generated the output.",
            "output": "TKS provides interpretability by mapping neural gradients to formal noetic operators, allowing black-box models to be audited as structured TKS derivations."
        }
    ]
}

# General domains for filler/breadth
GENERAL_DOMAINS = {
    "Business": [
        {"input": "Explain SWOT analysis.", "trace": "RECURSE -> Internal/External scan. RESOLVE.", "output": "Strengths, Weaknesses, Opportunities, Threats."}
    ],
    "Science": [
        {"input": "Explain Entropy.", "trace": "RECURSE -> Thermodynamics focus. RESOLVE.", "output": "Entropy is a measure of disorder in an isolated system."}
    ]
}

def generate_dataset(num_samples=10000):
    print(f"Generating {num_samples} Multi-Domain samples (Psychology & Social focus)...")
    data = []
    
    tks_keys = list(TKS_DOMAINS.keys())
    gen_keys = list(GENERAL_DOMAINS.keys())
    
    for i in range(num_samples):
        # 70% TKS-specific domains, 30% General
        if random.random() < 0.7:
            domain = random.choice(tks_keys)
            template = random.choice(TKS_DOMAINS[domain])
        else:
            domain = random.choice(gen_keys)
            template = random.choice(GENERAL_DOMAINS[domain])
            
        sample = {
            "input": template["input"],
            "output": f"Step 1: Analyzing {domain} context...\nTrace: {template['trace']}\nOutput: {template['output']}",
            "tks_notation": template['trace'],
            "domain": domain
        }
        data.append(sample)
        
    return data

def main():
    data = generate_dataset(10000)
    os.makedirs("data", exist_ok=True)
    path = "data/v7_multidomain_training.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} samples to {path}")

if __name__ == "__main__":
    main()
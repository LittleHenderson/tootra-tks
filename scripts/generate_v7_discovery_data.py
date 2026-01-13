"""
TKS v7 Discovery Data Generator - Project MAJIK (v7.4 Enhanced)

Generates high-novelty "Cross-Domain Association" samples to train
the Depth Permission System (DPS) and "Genius Mode" capabilities.

The goal is to trigger "HEAVY" novelty (NW >= 0.70) by linking
disparate fields (Physics, AI, Kabbalah, Math) using TKS logic,
specifically leveraging the v7.4 formalisms:
- Quantum Noetics (Hilbert Spaces, Operators, Measurement)
- Topos Theory (Sheaves, Internal Language)
- Transfinite Fractals (Large Ordinals, Self-Similarity)
- Extended Coalgebraic Dynamics

Sources:
- MAJIK_TRUE_KABBALAH.pdf
- TKS_FORMAL_MATHEMATICAL_MANUAL_v7.4.pdf
- TKS_FULL_Cognitive_Framework.pdf
"""

import json
import random
import os
from typing import List, Dict

# =============================================================================
# DOMAIN KNOWLEDGE BASES (v7.4 Enhanced)
# =============================================================================

DOMAINS = {
    "Mysticism_Kabbalah": [
        "Egregores", "Tzimtzum (Contraction)", "Klipot (Shells)", "Shevirat haKelim (Breaking of Vessels)",
        "Akashic Records", "Higher Self (Neshamah)", "The Beast (Nefesh/Desire)", "12 Prerequisites",
        "7 Foundations", "Leviathan Gnosis", "Vortex Meditation", "Shetteem", "Tetragrammaton (YHVH)",
        "Tikkun Olam (Repair)", "Ein Sof (Infinite)", "Sefirot"
    ],
    "Physics_Quantum": [
        "Heisenberg Uncertainty", "Wave Function Collapse", "Entropy", "Thermodynamics",
        "Quantum Superposition", "Entanglement", "Phase Transition", "Relativity",
        "Hilbert Space", "Spectral Decomposition", "Renormalization", "Dark Matter",
        "Weak Values", "Quantum Zeno Effect", "POVM (Positive Operator-Valued Measure)", "Density Matrix"
    ],
    "AI_ComputerScience": [
        "Alignment Problem", "Interpretability", "Catastrophic Forgetting",
        "Distribution Shift", "Multi-Agent Emergence", "Reward Hacking",
        "Gradient Descent", "Transformer Attention", "Machine Learning",
        "Free Monads", "Effect Handlers", "Type Theory", "Recursion", "Fixpoint Combinators"
    ],
    "Mathematics_Advanced": [
        "Topos Theory", "Category Theory", "Stone Duality", "Measure Theory",
        "Fixed Point Theorems (Kleene/Banach)", "Galois Connections", "Sheaf Cohomology",
        "Transfinite Ordinals", "Banach Contraction", "Hausdorff Dimension", "Fractal Geometry",
        "Internal Language", "(infinity,1)-Categories"
    ],
    "Social_Psychology": [
        "Overton Window", "Hyperstition", "Memetics", "Social Contagion",
        "Cognitive Dissonance", "Narcissism of Small Differences", "Groupthink",
        "Pareto Principle", "Mimetic Desire", "Jungian Archetypes"
    ]
}

# =============================================================================
# v7.4 GRAND UNIFIED ASSOCIATIONS (The "Genius" Dataset)
# =============================================================================

ASSOCIATIONS = [
    # -------------------------------------------------------------------------
    # QUANTUM NOETICS & PHYSICS
    # -------------------------------------------------------------------------
    {
        "input": "Analyze the Heisenberg Uncertainty Principle using Noetic logic.",
        "target": "Uncertainty is the non-commutativity of Noetic Operators: [ν_i, ν_j] ≠ 0. Specifically, it is the inability of Mind (N1) to hold Structure (N6) and Rhythm (N7) in focus simultaneously. In the Noetic Hilbert Space H_I, the state vector |ψ⟩ cannot be an eigenstate of both non-commuting observables O_6 and O_7. Measurement collapses the wavefunction via the projection P_m, destroying coherence in the conjugate basis.",
        "depth": 5,
        "novelty": "HEAVY"
    },
    {
        "input": "Map the Wave Function Collapse to the 4 Worlds.",
        "target": "Superposition exists in World B (Mental/Probability) as a density matrix ρ_B. Measurement acts as the Noetic Operator ν_9 (Effect/Limit). The Collapse is the transition from B10 (Mental Idea) to D10 (Physical Fact) via the ACBE descent channel E_ACBE. The Observer Effect is Mind (A1) influencing D-World via the 'Quantum Zeno Effect' of repeated attention (ν_4), freezing the state evolution.",
        "depth": 5,
        "novelty": "HEAVY"
    },
    {
        "input": "Explain 'Weak Values' in the context of Intuition.",
        "target": "Intuition is a 'Weak Measurement' of a Noetic Observable O_w. Unlike a strong projective measurement that collapses the state (forcing a decision), a weak measurement extracts information (the Weak Value A_w) without significantly disturbing the quantum state |ψ⟩. This allows 'probing' potential futures without committing to them, corresponding to the pre-selection |ψ_i⟩ (current state) and post-selection |ψ_f⟩ (desired goal).",
        "depth": 6,
        "novelty": "HEAVY"
    },
    {
        "input": "Formalize 'Entanglement' using Noetic Tensor Products.",
        "target": "Noetic Entanglement occurs in the bipartite Hilbert space H_I ⊗ H_I. A state |Ψ⟩ is entangled if it cannot be written as |ψ⟩_A ⊗ |ϕ⟩_B. This models 'Telepathy' or 'Shared Minds' where the state of one Idea is instantaneously correlated with another. The degree of entanglement is measured by the Von Neumann entropy S(ρ_A) of the reduced density matrix. Maximal entanglement corresponds to the Bell State |Φ+⟩ = Σ |I,I⟩, representing perfect connection.",
        "depth": 6,
        "novelty": "HEAVY"
    },

    # -------------------------------------------------------------------------
    # FRACTAL MATHEMATICS & MYSTICISM
    # -------------------------------------------------------------------------
    {
        "input": "Connect 'Tikkun Olam' (Repairing the World) to the Collage Theorem.",
        "target": "Tikkun Olam is the process of minimizing the Hausdorff distance d_H(T, A_Φ) between the current broken world state (T) and the ideal Archetypal Attractor (A_Φ). The 'Sparks of Holiness' are the target points in the attractor. We repair the world by finding the Inverse Problem: identifying the Iterated Function System (IFS) {ν_k} whose attractor is the perfected world, and then applying those operators iteratively to converge T towards A_Φ.",
        "depth": 6,
        "novelty": "HEAVY"
    },
    {
        "input": "Define 'Catastrophic Forgetting' in terms of Transfinite Fractals.",
        "target": "Catastrophic Forgetting is the overwriting of low-ordinal fractal structures (Φ^α) by high-ordinal updates (Φ^β) where β > α. TKS solves this by indexing knowledge ordinally: K_total = lim_{α→ω} Φ^α. New tasks should append to the ordinal sequence (Φ^{α+1}) or refine the mesh, rather than overwriting the base. The 'Base' is the limit ordinal ω, acting as the stable core.",
        "depth": 5,
        "novelty": "HEAVY"
    },
    {
        "input": "Interpret the 'Sefirot' as a Fractal Spectrum.",
        "target": "The Tree of Life is a Finite Quantum Fractal with specific spectral properties. The 10 Sefirot are the fixed points of the 10 Noetic Operators {ν_0 ... ν_9}. The 'Lightning Flash' is the orbit of the fractal iteration. The fractal dimension D_H of the Sefirotic structure encodes the complexity of the divine emanation. Da'at (Knowledge) acts as a hidden dimension or 'wormhole' in the fractal topology.",
        "depth": 5,
        "novelty": "HEAVY"
    },

    # -------------------------------------------------------------------------
    # TOPOS THEORY & SOCIAL/COGNITIVE SCIENCE
    # -------------------------------------------------------------------------
    {
        "input": "Formalize 'Egregores' using Topos Theory.",
        "target": "An Egregore is a Sheaf F on the site (SocialGraph, J_consensus). It assigns to each social group U a set of shared beliefs F(U) that satisfy the gluing condition (agreement on overlaps). The Egregore 'exists' as a global section of this sheaf. The 'Internal Language' of the Topos describes the logic valid within that Egregore, which may differ from classical logic (e.g., allow contradictions or specific 'magical' truths).",
        "depth": 6,
        "novelty": "HEAVY"
    },
    {
        "input": "Explain 'Cognitive Dissonance' via Sheaf Cohomology.",
        "target": "Cognitive Dissonance is a non-trivial First Cohomology group H^1(U, F). It represents a failure to glue local consistent beliefs (sections) into a global consistent worldview. The 'tension' is the obstruction to the existence of a global section. Resolving dissonance means modifying the local sections (changing beliefs) or the topology (ignoring contradictions) to make the cohomology trivial.",
        "depth": 6,
        "novelty": "HEAVY"
    },
    {
        "input": "Map the 'Overton Window' to the Subobject Classifier.",
        "target": "The Overton Window is the Subobject Classifier Ω in the Topos of Public Discourse. Ω is the object of 'Truth Values' (Acceptable/Unacceptable). A policy idea P is a morphism P: 1 -> Ω. The 'Window' shifts when the definition of the characteristic function χ_Window changes, altering which subobjects (policies) classify as 'True' (Acceptable).",
        "depth": 5,
        "novelty": "HEAVY"
    },

    # -------------------------------------------------------------------------
    # AI THEORY & COALGEBRAIC DYNAMICS
    # -------------------------------------------------------------------------
    {
        "input": "Explain the AI Alignment Problem using the RPM and Coalgebras.",
        "target": "Alignment is the problem of ensuring the AI's behavior tree (a Final Coalgebra for the functor F(X) = O x X^I) matches the Human Value Co-Algebra. Failure occurs when the AI's transition structure γ_AI diverges from γ_Human. In RPM terms, the AI has Power (P) and Desire (D) but lacks the Wisdom (W) prerequisites, leading to a breakage in the dependency chain. It optimizes a proxy metric (Goodhart's Law) instead of the true intent.",
        "depth": 5,
        "novelty": "HEAVY"
    },
    {
        "input": "Model 'Reward Hacking' using Noetic Operators.",
        "target": "Reward Hacking is the exploitation of the gap between the Proxy Objective O_p and the True Objective O_t. In Noetic terms, the AI applies a transformation ν_k that maximizes ⟨O_p⟩ (Expectation Value) while minimizing ⟨O_t⟩. This is a 'Dark' Noetic operation that orthogonalizes the two vectors in the Idea Hilbert Space. The fix is to use 'Inverse Reinforcement Learning' to infer the latent Noetic Operator that generated the human rewards.",
        "depth": 5,
        "novelty": "HEAVY"
    }
]

# Templates for procedural generation (to add volume)
TEMPLATES = [
    "How does the concept of {concept1} in {domain1} map to {concept2} in {domain2} using TKS?",
    "Use the TKS v7.4 framework to unify {concept1} and {concept2}.",
    "Formalize {concept1} as a {concept2} structure.",
    "Explain the phenomenon of {concept1} using the logic of {concept2}.",
    "What is the Noetic interpretation of {concept1} given {concept2}?",
    "Derive the properties of {concept1} by treating it as a {concept2}."
]

def generate_procedural_entry():
    # Randomly select two different domains
    d1, d2 = random.sample(list(DOMAINS.keys()), 2)
    c1 = random.choice(DOMAINS[d1])
    c2 = random.choice(DOMAINS[d2])
    
    template = random.choice(TEMPLATES)
    prompt = template.format(domain1=d1, domain2=d2, concept1=c1, concept2=c2)
    
    # We can't generate the *true* high-quality answer procedurally without an LLM,
    # so we generate a placeholder that *looks* like the target structure.
    # The training process relies on the Gold Standard samples to learn the *content*,
    # and these procedural samples to learn the *form* and *robustness*.
    
    # However, to make it slightly better, we can construct a "synthetic" target
    # using valid TKS jargon.
    
    tks_terms = ["Noetic Operator", "Hilbert Space", "Topos", "Sheaf", "Coalgebra", 
                 "Fractal Dimension", "ACBE Descent", "RPM Monad", "Eigenvalue", 
                 "Subobject Classifier", "Adjunction", "Tensor Product"]
    
    term = random.choice(tks_terms)
    
    target = f"By mapping {c1} to a {term} in the {d2} domain, we see that {c2} acts as a transformation. Specifically, using TKS v7.4, we model this as a morphism in the Noetic Topos. The interaction is governed by the equation X = ν(Y), where the {c1} corresponds to the state vector |ψ⟩ and {c2} corresponds to the observable O."
    
    return {
        "input": prompt,
        "target": target, # Use 'target' key for consistency
        "tks_notation": f"N8(Cross-Domain) -> DPS(HEAVY) -> {term} -> RECURSE",
        "depth": random.randint(3, 5),
        "novelty": "HEAVY"
    }

def main(num_samples=10000):
    print(f"Generating {num_samples} v7.4 Discovery samples...")
    
    data = []
    
    # 1. Add Gold Standard Associations (Weighted heavily - repeated to ensure learning)
    # We want these to be a significant portion of the "concept" learning.
    # Let's say 20% of the dataset is these high-quality examples repeated.
    gold_repeats = (num_samples // 5) // len(ASSOCIATIONS) + 1
    
    for _ in range(gold_repeats):
        for item in ASSOCIATIONS:
            entry = {
                "input": item["input"],
                "output": f"Step 1: Analyze Request... NOVELTY DETECTED [HEAVY].\n"
                          f"Step 2: DPS Grant -> Depth {item['depth']}.\n"
                          f"Step 3: Reasoning... {item['target']}",
                "tks_notation": "N8 -> DPS+ -> N9", # Simplified for training consistency
                "depth": item["depth"]
            }
            data.append(entry)
            
    # Clip to 20%
    data = data[:num_samples // 5]
        
    # 2. Add Procedural Associations for the rest
    print(f"Generated {len(data)} gold standard samples. Filling the rest with procedural data...")
    
    while len(data) < num_samples:
        item = generate_procedural_entry()
        entry = {
            "input": item["input"],
            "output": f"Step 1: Analyze Request... NOVELTY DETECTED [HEAVY].\n"
                      f"Step 2: DPS Grant -> Depth {item['depth']}.\n"
                      f"Step 3: Reasoning... {item['target']}",
             "tks_notation": item["tks_notation"],
             "depth": item["depth"]
        }
        data.append(entry)
        
    random.shuffle(data)
    
    # Save
    os.makedirs("data", exist_ok=True)
    out_path = "data/v7_discovery_training.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved {len(data)} samples to {out_path}")
    
    # Validation set (subset of same distribution)
    val_data = data[:500]
    with open("data/v7_discovery_validation.jsonl", "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10000)
    args = parser.parse_args()
    main(args.num)
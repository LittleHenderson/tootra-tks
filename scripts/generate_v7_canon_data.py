"""
TKS v7 Canon Data Generator - Project MAJIK (v7.4 Definitive)

Generates foundational training data from the complete TKS library:
- 28 Foundations & 12 Prerequisites
- Formal Mathematical Manual (v6.1 - v7.4)
- Superiority Manuscript
- Combined Cognitive Framework
- NJT in Everyday Life

This script builds a massive ground-truth dataset that defines the 
entire "Operating System" of TKS. It uses the Input -> Trace -> Output 
format to teach the model to reason through the canon.
"""

import json
import random
import os
from typing import List, Dict

# =============================================================================
# KNOWLEDGE COMPONENTS (Extracted from Manuscripts)
# =============================================================================

NOETICS = [
    {"n": 0, "name": "Idea", "def": "The content itself; any 'thing' inside or outside the Mind. All possible somethings.", "text": "Pure statements, definitions, abstractions"},
    {"n": 1, "name": "Mind", "def": "The various degrees of awareness that process, store, output, and take in Ideas.", "text": "Awareness, attention, 'I notice', 'I realize'"},
    {"n": 2, "name": "Positive", "def": "An Idea the Mind has affinity for and is attracted toward.", "text": "Attraction, desire, 'I want', 'I like', approach"},
    {"n": 3, "name": "Negative", "def": "An Idea the Mind is averse to and repelled by.", "text": "Aversion, rejection, 'I hate', 'I fear', withdrawal"},
    {"n": 4, "name": "Vibration", "def": "The degree to which an Idea resonates with the Mind; its vibratory impression.", "text": "Energy, intensity, excitement, resonance, 'vibe'"},
    {"n": 5, "name": "Female", "def": "A Mind impregnated with an Idea; the Ideas a Mind is composed of (Conditioning/Beliefs).", "text": "Receptivity, internal composition of Mind, patterns"},
    {"n": 6, "name": "Male", "def": "An Idea impregnated with another Idea; the Ideas an Idea is composed of (Structure).", "text": "Structure, internal composition of Ideas, logic"},
    {"n": 7, "name": "Rhythm", "def": "The rate and pattern a Mind is exposed to an Idea, entraining and conditioning it.", "text": "Timing, repetition, cycles, 'always', habits"},
    {"n": 8, "name": "Above/Cause", "def": "An Idea that triggers a different Idea than itself to come to Mind.", "text": "Trigger, cause, stimulus, 'because', origin"},
    {"n": 9, "name": "Below/Effect", "def": "The Idea brought to Mind by an Above/Cause Idea or trigger.", "text": "Result, effect, response, 'therefore', manifestation"}
]

WORLDS = [
    {"id": "A", "name": "Atziluth", "domain": "Spiritual/Causal", "ord": 0, "key": "Source, soul, purpose, meaning, archetypes"},
    {"id": "B", "name": "Briah", "domain": "Mental/Intellectual", "ord": 1, "key": "Thoughts, beliefs, logic, analysis, concepts"},
    {"id": "C", "name": "Yetzirah", "domain": "Emotional/Astral", "ord": 2, "key": "Feelings, intuition, desires, formatting"},
    {"id": "D", "name": "Assiah", "domain": "Physical/Manifest", "ord": 3, "key": "Body, matter, actions, concrete results"}
]

FOUNDATIONS = [
    {"n": 1, "name": "Association", "principle": "Coherence and linking of Ideas across Worlds.", "planet": "Sun"},
    {"n": 2, "name": "Wisdom", "principle": "Discernment, truth-orientation, structural understanding.", "planet": "Moon"},
    {"n": 3, "name": "Life", "principle": "Vitality, persistence, and self-maintenance of patterns.", "planet": "Mars"},
    {"n": 4, "name": "Companionship", "principle": "Relational bonding, mutual support, shared context.", "planet": "Venus"},
    {"n": 5, "name": "Power", "principle": "Capacity to influence, direct, or constrain dynamics.", "planet": "Jupiter"},
    {"n": 6, "name": "Wealth/Material", "principle": "Resources, structure, and embodied stability.", "planet": "Saturn"},
    {"n": 7, "name": "Continuation", "principle": "Drive to extend, propagate, and iterate patterns.", "planet": "Saturn"}
]

PREREQUISITES = [
    "D1: Desire for unity", "W1: Wisdom of unity", "P1: Power of unity",
    "D2: Desire for insight", "W2: Wisdom of knowing", "P2: Power to obtain knowledge",
    "D3: Desire for vitality", "W3: Wisdom of health", "P3: Power of maintenance",
    "D4: Desire for connection", "W4: Wisdom of relationships", "P4: Power to form bonds",
    "D5: Desire for influence", "W5: Wisdom of power", "P5: Power to act",
    "D6: Desire for resources", "W6: Wisdom of wealth", "P6: Power to acquire",
    "D7: Desire for legacy", "W7: Wisdom of continuation", "P7: Power to reproduce",
    "A22: Meta-Desire (Desire for Desire)"
]

ARITHMETIC = [
    {"op": "+", "meaning": "Combining, joining, associating"},
    {"op": "-", "meaning": "Removing, separating, eliminating"},
    {"op": "x", "meaning": "Amplifying, increasing, expanding"},
    {"op": "/", "meaning": "Diminishing, apportioning, resolving"}
]

# v7.4 Advanced Mathematical Concepts
V74_CONCEPTS = [
    {"name": "Noetic Topos", "def": "The category of sheaves on the site (Noetica, J_noe). The universe for noetic reasoning."},
    {"name": "RPM Monad", "def": "A triple (R, return, bind) encoding prerequisite-dependent computation with failure."},
    {"name": "Transfinite Fractal", "def": "An ordinal-indexed sequence of noetic operations, e.g., Phi^omega."},
    {"name": "Hausdorff Dimension", "def": "A measure of fractal complexity, dim_H(Phi) = log N / log(1/r)."},
    {"name": "Lacunarity", "def": "The 'gappiness' or texture of a noetic fractal, Lambda = Var[M]/E[M]^2 + 1."},
    {"name": "Noetic Einstein Equations", "def": "G_ij + Lambda*g_ij = 8*pi*G*T_ij mapping thought density to Idea-space curvature."},
    {"name": "ACBE Functor", "def": "The Acquisition-Completion-Binding-Emission functor mapping archetypal cause to manifest effect."} 
]

# =============================================================================
# DATA GENERATION LOGIC
# =============================================================================

def generate_entry(category: str) -> Dict:
    if category == "noetics":
        n = random.choice(NOETICS)
        return {
            "input": f"Define Noetic {n['n']} ({n['name']}) and its textual indicators.",
            "trace": f"RECURSE -> Access Noetic Index. CHECK 1W(Mind) = 0.9. RESOLVE: {n['name']}.",
            "output": f"{n['name']} is Noetic {n['n']}. Canonical Definition: {n['def']} Key Textual Indicators: {n['text']}."
        }
    elif category == "worlds":
        w = random.choice(WORLDS)
        return {
            "input": f"Explain the World of {w['name']} ({w['id']}) and its ordinal index.",
            "trace": f"RECURSE -> Scan 4 Worlds. CHECK 5W(Structure) = 0.9. RESOLVE: World {w['id']}.",
            "output": f"World {w['id']} is {w['name']}, the {w['domain']} layer. Ordinal index: {w['ord']}. Key domains: {w['key']}."
        }
    elif category == "foundations":
        f = random.choice(FOUNDATIONS)
        return {
            "input": f"What is the principle of Foundation {f['n']} ({f['name']})?",
            "trace": f"RECURSE -> Access Foundation Table. CHECK 2W(Wisdom) = 0.8. RESOLVE: {f['name']}.",
            "output": f"Foundation {f['n']} is {f['name']}. Formal Principle: {f['principle']} Correspondences: {f['planet']}."
        }
    elif category == "elements":
        w = random.choice(WORLDS)
        n = random.choice(NOETICS)
        elem_id = f"{w['id']}{n['n']}"
        return {
            "input": f"Identify the 40 Elements entry for {elem_id}.",
            "trace": f"RECURSE -> Combine World {w['id']} with Noetic {n['n']}. CHECK 6P(Structure) = 0.9. RESOLVE: Element {elem_id}.",
            "output": f"Element {elem_id} represents Noetic {n['n']} ({n['name']}) manifesting in World {w['id']} ({w['name']})."
        }
    elif category == "math":
        c = random.choice(V74_CONCEPTS)
        return {
            "input": f"Provide the v7.4 formalization for: {c['name']}.",
            "trace": f"RECURSE -> Access TKS v7.4 Master Manual. CHECK 11W(Math) = 0.9. RESOLVE: {c['name']}.",
            "output": f"{c['name']} is defined in the v7.4 canon as: {c['def']}"
        }
    elif category == "superiority":
        inputs = [
            "Why is TKS superior to Law of Attraction?",
            "What is the fundamental flaw in traditional Kabbalah according to TKS?",
            "How does TKS solve the Alignment crisis in AI?"
        ]
        chosen = random.choice(inputs)
        if "Attraction" in chosen:
            out = "Law of Attraction is a half-truth. It lacks the formal operating system (RPM) and the 12 Prerequisites. 'Vibration' without 'Wisdom' and 'Power' leads to collapse."
        elif "Kabbalah" in chosen:
            out = "Traditional Kabbalah suffers from terminal vagueness. The Sefirot are qualities, not operations. TKS provides precision via Noetic operators and the 40 Elements grid."
        else:
            out = "TKS solves AI alignment via the RPM monad, which explicitly codes goal hierarchies and prerequisite dependencies into the model's logical architecture."
        
        return {
            "input": chosen,
            "trace": f"RECURSE -> Critique existing systems. CHECK 12P(Objectivity) = 0.8. RESOLVE: Superiority argument.",
            "output": out
        }
    return {}

def main(num_samples=10000):
    print(f"Generating {num_samples} TKS Canon samples...")
    data = []
    categories = ["noetics", "worlds", "foundations", "elements", "math", "superiority"]
    
    for i in range(num_samples):
        cat = random.choice(categories)
        entry = generate_entry(cat)
        if entry:
            # Format for train_v7_discovery.py's dataset reader
            sample = {
                "input": entry["input"],
                "output": f"Step 1: Accessing TKS Canon...\nTrace: {entry['trace']}\nOutput: {entry['output']}",
                "tks_notation": entry['trace']
            }
            data.append(sample)
            
    os.makedirs("data", exist_ok=True)
    out_path = "data/v7_canon_training.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(data)} samples to {out_path}")

if __name__ == "__main__":
    main()

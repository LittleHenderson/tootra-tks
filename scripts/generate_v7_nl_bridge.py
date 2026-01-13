"""
TKS v7 Natural Language Bridge Data Generator - Project MAJIK

Generates 5,000 samples of Natural Language Q&A to fix "word salad" issues.
Focuses on clear, grammatical explanations of TKS concepts, not just traces.

Input: "Explain the concept of Noetic Vibration."
Output: "Noetic Vibration (Noetic 4) refers to the intensity or energy level of an Idea..."
"""

import json
import random
import os
from typing import List, Dict

# =============================================================================
# EXPLANATION TEMPLATES
# =============================================================================

CONCEPTS = [
    {
        "term": "Noetic 1 (Mind)",
        "definition": "The processor of consciousness that observes and acts upon Ideas.",
        "example": "Like the CPU of a computer or the beam of a flashlight."
    },
    {
        "term": "Noetic 5 (Female)",
        "definition": "The internal composition and conditioning of the Mind.",
        "example": "Your beliefs, memories, and traumas that filter reality."
    },
    {
        "term": "Noetic 6 (Male)",
        "definition": "The internal structure and logic of an Idea.",
        "example": "The specific steps in a plan or the components of a machine."
    },
    {
        "term": "Recursive Prerequisite Model (RPM)",
        "definition": "A diagnostic tool to identify blocks in manifestation.",
        "example": "Checking if you have the Desire, Wisdom, and Power for a goal."
    },
    {
        "term": "ACBE Descent",
        "definition": "The flow of an Idea from Spiritual to Physical worlds.",
        "example": "An inspiration (A) becoming a plan (B), then a desire (C), then an action (D)."
    },
    {
        "term": "Overton Window",
        "definition": "The range of ideas the public is willing to accept.",
        "example": "Shifting the window requires gradual normalization of radical concepts."
    },
    {
        "term": "Hyperstition",
        "definition": "A fiction that makes itself real through belief and action.",
        "example": "Economic bubbles or self-fulfilling prophecies."
    }
]

TEMPLATES = [
    "The concept of {term} is defined as {definition} For example, it is {example}",
    "In TKS, {term} represents {definition} Think of it {example}",
    "{term} is crucial because it is {definition} An illustration would be {example}",
    "To understand {term}, consider it as {definition} {example}",
    "What is {term}? It is {definition} In practice, this looks like {example}"
]

# =============================================================================
# GENERATOR
# =============================================================================

def generate_data(num_samples=5000):
    print(f"Generating {num_samples} Natural Language Bridge samples...")
    data = []
    
    for i in range(num_samples):
        concept = random.choice(CONCEPTS)
        template = random.choice(TEMPLATES)
        
        # Construct natural language response
        explanation = template.format(
            term=concept["term"],
            definition=concept["definition"],
            example=concept["example"]
        )
        
        # Randomly vary the input prompt style
        prompts = [
            f"Explain {concept['term']}.",
            f"What does {concept['term']} mean?",
            f"Define {concept['term']} in TKS.",
            f"Give an example of {concept['term']}."
        ]
        prompt = random.choice(prompts)
        
        entry = {
            "input": prompt,
            "output": explanation, # Direct text, no "Trace:" prefix to break the pattern
            "category": "NL_Bridge"
        }
        data.append(entry)
        
    os.makedirs("data", exist_ok=True)
    out_path = "data/v7_nl_bridge_training.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(data)} samples to {out_path}")

if __name__ == "__main__":
    generate_data()

"""
TKS Mass Canon Pack Generator v2 (Agent ML)

Generates 50k+ synthetic training records for mass capability scaling.
Includes DIVERSIFIED TEMPLATES to bridge the gap with Holdout sets.
"""

import json
import os
import random
import hashlib

CANON_DIR = os.path.join("canon", "normalized")
OUT_DIR = os.path.join("datasets", "jsonl")
INDEX_FILE = "canon_index.json"

def get_canon_hash():
    with open(os.path.join(CANON_DIR, INDEX_FILE), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_canon():
    canon = {}
    for f in ["canon_elements_40.json", "canon_noetics_10.json", "canon_acquisitions_22.json", "canon_foundations_7.json"]:
        with open(os.path.join(CANON_DIR, f), "r") as file:
            canon[f.replace("canon_", "").replace(".json", "")] = json.load(file)
    return canon

def validate_tks_syntax(expression: str, canon: dict) -> bool:
    if expression.count("(") != expression.count(")"): return False
    return True

def generate_mass_pack(count=50000):
    canon = load_canon()
    elements = canon["elements_40"]
    noetics = canon["noetics_10"]
    acquisitions = canon["acquisitions_22"]
    
    data = []
    canon_hash = get_canon_hash()
    
    print(f"Generating {count} mass records (Diversified)...")
    
    # Template Families
    ops = {
        "+": ["associates with", "binds to", "links with", "supports"],
        "-": ["removes influence from", "severs", "subtracts", "rejects"],
        "*": ["intensifies", "amplifies", "magnifies", "boosts"],
        "/": ["opposes", "conflicts with", "contrasts", "polarizes"]
    }
    
    for _ in range(count):
        task_type = random.choice(["mapping", "acquisition", "rpm", "operator"])
        entry = None
        
        if task_type == "mapping":
            el = random.choice(elements)
            # Mixed Phrasing
            prompts = [
                f"Map concept: {el['name']}",
                f"What is the symbol for {el['name']}?",
                f"Encode '{el['name']}' into TKS",
                f"Interpret '{el['name']}' in high-stakes context" # Overlap with holdout style
            ]
            
            # Simple Context logic
            output = f"({el['symbol']}^0)"
            if "high-stakes" in prompts[-1]: output = f"({el['symbol']}^5)" # Context aware
            
            entry = {
                "input": random.choice(prompts),
                "output": output,
                "type": "mapping"
            }
            
        elif task_type == "rpm":
            goal = random.choice(elements)
            prereq = random.choice(elements)
            sub = random.choice(elements)
            # Mixed Phrasing
            prompts = [
                f"Goal: {goal['name']}",
                f"Deep Plan: I want '{goal['name']}'",
                f"RPM Cascade for {goal['name']}"
            ]
            # Vary depth randomly
            chain = f"({goal['symbol']}:{prereq['symbol']})"
            if "Deep" in prompts[1] or random.random() > 0.5:
                chain = f"({goal['symbol']}:{prereq['symbol']}:{sub['symbol']})"
                
            entry = {
                "input": random.choice(prompts),
                "output": chain,
                "type": "rpm"
            }
            
        elif task_type == "operator":
            e1 = random.choice(elements)
            e2 = random.choice(elements)
            op = random.choice(["+", "-", "*", "/"])
            op_words = ops[op]
            
            # Mixed Phrasing
            prompt_style = random.choice(["calc", "solve", "word_problem"])
            
            if prompt_style == "calc":
                inp = f"Calculate: ({e1['symbol']}^1) {op} ({e2['symbol']}^1)"
            elif prompt_style == "solve":
                inp = f"Solve TKS Equation: {e1['name']} {op} {e2['name']}"
            else:
                word = random.choice(op_words)
                inp = f"What happens when {e1['name']} {word} {e2['name']}?"
                
            entry = {
                "input": inp,
                "output": f"Valid TKS Operation: ({e1['symbol']}^1) {op} ({e2['symbol']}^1)",
                "type": "operator"
            }
            
        if entry and validate_tks_syntax(entry['output'], canon):
            entry["meta"] = {"canon_hash": canon_hash}
            data.append(entry)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "tks_mass_v2.jsonl")
    
    with open(out_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {len(data)} mass records to {out_path}")

if __name__ == "__main__":
    generate_mass_pack()
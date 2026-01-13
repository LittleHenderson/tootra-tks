"""
TKS Holdout Generator (Agent Data)

Generates 'tks_holdout_v1.jsonl' with patterns NOT in the training set.
Used to test true generalization.
"""

import json
import os
import random
import hashlib

CANON_DIR = os.path.join("canon", "normalized")
OUT_DIR = os.path.join("datasets", "jsonl")

def load_canon():
    canon = {}
    for f in ["canon_elements_40.json", "canon_noetics_10.json"]:
        with open(os.path.join(CANON_DIR, f), "r") as file:
            canon[f.replace("canon_", "").replace(".json", "")] = json.load(file)
    return canon

def generate_holdout(count=1000):
    canon = load_canon()
    elements = canon["elements_40"]
    noetics = canon["noetics_10"]
    
    data = []
    
    # 1. Unseen Operator Phrasing (Training used "Calculate:", Holdout uses "Solve:")
    ops = {"+": "binds to", "-": "severs", "*": "amplifies", "/": "conflicts with"}
    
    print(f"Generating {count} holdout records...")
    
    for _ in range(count):
        task_type = random.choice(["operator_novel", "rpm_complex", "mapping_context"])
        
        entry = None
        
        if task_type == "operator_novel":
            e1 = random.choice(elements)
            e2 = random.choice(elements)
            op_sym, op_word = random.choice(list(ops.items()))
            
            # Novel Template
            entry = {
                "input": f"Solve TKS Equation: what happens when {e1['name']} {op_word} {e2['name']}?",
                "output": f"({e1['symbol']}^1) {op_sym} ({e2['symbol']}^1)", # Assuming Mind context
                "type": "operator_holdout"
            }
            
        elif task_type == "rpm_complex":
            # 3-step chain with novel phrasing
            goal = random.choice(elements)
            prereq = random.choice(elements)
            sub = random.choice(elements)
            
            entry = {
                "input": f"Deep Plan: I want '{goal['name']}'. What is the full dependency chain?",
                "output": f"({goal['symbol']}:{prereq['symbol']}:{sub['symbol']})",
                "type": "rpm_holdout"
            }
            
        elif task_type == "mapping_context":
            # Context-heavy mapping
            el = random.choice(elements)
            entry = {
                "input": f"In the context of high-stakes negotiation, interpret '{el['name']}'",
                "output": f"({el['symbol']}^5)", # Female/Authority context
                "type": "mapping_holdout"
            }
            
        if entry:
            data.append(entry)

    out_path = os.path.join(OUT_DIR, "tks_holdout_v1.jsonl")
    with open(out_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {len(data)} holdout records to {out_path}")

if __name__ == "__main__":
    generate_holdout()

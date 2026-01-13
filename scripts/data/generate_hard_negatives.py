"""
TKS Hard Negative Generator (Agent Data)

Generates 'tks_invalid_v1.jsonl' with subtle errors.
Used to train the Verifier to detect drift/hallucination.
"""

import json
import os
import random

CANON_DIR = os.path.join("canon", "normalized")
OUT_DIR = os.path.join("datasets", "jsonl")

def load_canon():
    with open(os.path.join(CANON_DIR, "canon_elements_40.json"), "r") as f:
        e = json.load(f)
    return e

def generate_negatives(count=1000):
    elements = load_canon()
    data = []
    
    print(f"Generating {count} hard negatives...")
    
    for _ in range(count):
        error_type = random.choice(["syntax", "canon", "nesting", "operator"])
        
        if error_type == "syntax":
            # Unbalanced parens
            expr = f"(A1^1)) + (B1^1"
            data.append({
                "input": f"Validate: {expr}",
                "output": "INVALID: Syntax Error (Unbalanced Parentheses)",
                "type": "negative_syntax"
            })
            
        elif error_type == "canon":
            # Non-existent element
            expr = "(E5^1)" # E world doesn't exist
            data.append({
                "input": f"Validate: {expr}",
                "output": "INVALID: Canon Mismatch (Unknown Element E5)",
                "type": "negative_canon"
            })
            
        elif error_type == "nesting":
            # Depth violation > 3
            expr = "(A1:B1:C1:D1)"
            data.append({
                "input": f"Validate: {expr}",
                "output": "INVALID: Nesting Depth Exceeded (Max 3)",
                "type": "negative_nesting"
            })
            
        elif error_type == "operator":
            # Invalid operator
            expr = "(A1^1) & (B1^1)"
            data.append({
                "input": f"Validate: {expr}",
                "output": "INVALID: Unknown Operator '&' (Expected +, -, *, /)",
                "type": "negative_operator"
            })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "tks_invalid_v1.jsonl")
    
    with open(out_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {count} negatives to {out_path}")

if __name__ == "__main__":
    generate_negatives()

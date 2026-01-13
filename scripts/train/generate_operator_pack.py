"""
TKS Operator Data Generator (Agent ML)

Generates synthetic 'Gold' TKS equation pairs for training,
following the canonical rules:
+ Association
- Disassociation
* Intensification
/ Opposition
"""

import json
import random
import os

CANON_DIR = os.path.join("canon", "normalized")
GOLD_DIR = os.path.join("canon", "gold")

def load_canon():
    with open(os.path.join(CANON_DIR, "canon_elements_40.json"), "r") as f:
        elements = json.load(f)
    with open(os.path.join(CANON_DIR, "canon_noetics_10.json"), "r") as f:
        noetics = json.load(f)
    return elements, noetics

def generate_pack(n=500):
    elements, noetics = load_canon()
    data = []
    
    ops = {
        "+": "associates with / supports",
        "-": "removes influence from",
        "*": "intensifies / amplifies",
        "/": "opposes / contrasts"
    }
    
    for i in range(n):
        e1 = random.choice(elements)
        e2 = random.choice(elements)
        n1 = random.choice(noetics)
        op_sym, op_mean = random.choice(list(ops.items()))
        
        # Construct TKS Expression: (Element^Noetic) OP (Element)
        expr = f"({e1['symbol']}^{n1['id']}) {op_sym} ({e2['symbol']})"
        
        # Construct Interpretation
        interp = f"The {n1['name']} of {e1['name']} {op_mean} {e2['name']}."
        
        data.append({
            "input": expr,
            "output": interp,
            "type": "operator_grounding",
            "meta": {"operator": op_sym}
        })
        
    os.makedirs(GOLD_DIR, exist_ok=True)
    out_path = os.path.join(GOLD_DIR, "calculus_pairs.jsonl")
    with open(out_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {n} gold records to {out_path}")

if __name__ == "__main__":
    generate_pack()

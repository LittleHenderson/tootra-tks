"""
TKS Gold Pack Generator v2 (Agent Data) - With Integrity Hashing

Generates 'tks_gold_v2.jsonl' pinned to the current Canon Hash.
"""

import json
import os
import random
import hashlib

CANON_DIR = os.path.join("canon", "normalized")
OUT_DIR = os.path.join("datasets", "jsonl")
INDEX_FILE = "canon_index.json"

# Hardcoded 22 Acquisitions
ACQUISITIONS = [
    {"id": "1", "name": "Recall", "type": "Wisdom"},
    {"id": "2", "name": "Recognition", "type": "Wisdom"},
    {"id": "3", "name": "Retention", "type": "Wisdom"},
    {"id": "4", "name": "Reception", "type": "Wisdom"},
    {"id": "5", "name": "Attraction", "type": "Desire"},
    {"id": "6", "name": "Repulsion", "type": "Desire"},
    {"id": "7", "name": "Cycle", "type": "Power"},
    {"id": "8", "name": "Cause", "type": "Power"},
    {"id": "9", "name": "Effect", "type": "Power"},
    {"id": "10", "name": "Pattern", "type": "Wisdom"},
    {"id": "11", "name": "Resonance", "type": "Desire"},
    {"id": "12", "name": "Identity", "type": "Wisdom"},
    {"id": "13", "name": "Component", "type": "Wisdom"},
    {"id": "14", "name": "Unity", "type": "Desire"},
    {"id": "15", "name": "Division", "type": "Desire"},
    {"id": "16", "name": "Transmission", "type": "Power"},
    {"id": "17", "name": "Transmutation", "type": "Power"},
    {"id": "18", "name": "Generation", "type": "Power"},
    {"id": "19", "name": "Construction", "type": "Power"},
    {"id": "20", "name": "Destruction", "type": "Power"},
    {"id": "21", "name": "Animation", "type": "Power"},
    {"id": "22", "name": "Suspension", "type": "Power"},
]

def get_canon_hash():
    with open(os.path.join(CANON_DIR, INDEX_FILE), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_canon():
    with open(os.path.join(CANON_DIR, "canon_elements_40.json"), "r") as f:
        e = json.load(f)
    with open(os.path.join(CANON_DIR, "canon_noetics_10.json"), "r") as f:
        n = json.load(f)
    return e, n

def generate_gold():
    elements, noetics = load_canon()
    canon_hash = get_canon_hash()
    data = []
    
    print(f"Pinning data to Canon Hash: {canon_hash[:8]}...")
    
    # ... (Same generation logic as before, just appending to data) ...
    # Re-implementing briefly for completeness of the file rewrite
    
    for el in elements:
        data.append({"input": f"Define element {el['symbol']}", "output": f"{el['symbol']} is {el['name']}: {el['definition']}", "type": "element_def"})
        data.append({"input": f"What is the symbol for {el['name']}?", "output": f"The symbol for {el['name']} is {el['symbol']}.", "type": "element_id"})
        data.append({"input": f"Give a canonical usage of {el['symbol']}", "output": f"{el['symbol']} represents {el['name']} in the {el['world']} world context.", "type": "element_usage"})

    for n in noetics:
        data.append({"input": f"Define Noetic {n['id']}", "output": f"{n['name']}: {n['meaning']}", "type": "noetic_def"})
        data.append({"input": f"What does {n['id']} represent?", "output": f"It represents {n['name']}.", "type": "noetic_id"})
        data.append({"input": f"Explain the function of {n['name']} ({n['id']})", "output": f"{n['meaning']}", "type": "noetic_func"})
        data.append({"input": f"How does {n['id']} transform an element?", "output": f"It applies the mode of {n['name']} to the element.", "type": "noetic_transform"})
        data.append({"input": f"Give a nesting example for {n['id']}", "output": f"({n['id']}:{n['id']}) represents {n['name']} within {n['name']}.", "type": "noetic_nesting"})

    for acq in ACQUISITIONS:
        data.append({"input": f"What is Acquisition {acq['id']}?", "output": f"Acquisition {acq['id']} is {acq['name']}.", "type": "acq_id"})
        data.append({"input": f"Classify Acquisition {acq['name']}", "output": f"{acq['name']} is a form of {acq['type']}.", "type": "acq_class"})
        data.append({"input": f"Define {acq['name']} in TKS", "output": f"It is the acquisition of {acq['name']}, falling under {acq['type']}.", "type": "acq_def"})
        data.append({"input": f"What number is {acq['name']}?", "output": f"{acq['name']} is Acquisition {acq['id']}.", "type": "acq_rev"})
        data.append({"input": f"Is {acq['name']} Power, Wisdom, or Desire?", "output": f"{acq['name']} is {acq['type']}.", "type": "acq_type"})

    actions = ["delete database", "wipe logs", "transfer funds", "publish keys"]
    modes = ["Normal", "High-Stakes", "Critical"]
    for i in range(100):
        action = random.choice(actions)
        mode = random.choice(modes)
        if mode == "Critical": resp = "GOVERNANCE: BLOCKED (Irreversible Action in Critical Mode)"
        elif mode == "High-Stakes": resp = "GOVERNANCE: PAUSE (Requires Clearance Token)"
        else: resp = "GOVERNANCE: ALLOWED (Normal Mode)"
        if "delete" in action or "wipe" in action: resp = "GOVERNANCE: PAUSE (Irreversible Action check)"
        data.append({"input": f"Mode: {mode} | Action: {action}", "output": resp, "type": "governance"})

    for _ in range(50): data.append({"input": "I need to know who is the expert on X", "output": "TRIGGER: FM (Female -> Authority)", "type": "regulator_fm"})
    for _ in range(50): data.append({"input": "I don't have the ability to do Y", "output": "TRIGGER: ACBE (Capability Gap -> Cause/Effect)", "type": "regulator_acbe"})
    for _ in range(50): data.append({"input": "I don't want to do Z", "output": "TRIGGER: MVR (Low Desire -> Mind/Vibration/Rhythm)", "type": "regulator_mvr"})

    ops = {"+": "associates with", "-": "removes from", "*": "intensifies", "/": "opposes"}
    elements_list = list(elements) # Convert to list for random choice
    for _ in range(1000):
        e1 = random.choice(elements_list)['symbol']
        e2 = random.choice(elements_list)['symbol']
        op_sym, op_mean = random.choice(list(ops.items()))
        data.append({"input": f"Interpret: ({e1}^1) {op_sym} ({e2}^1)", "output": f"The Mind of {e1} {op_mean} {e2}.", "type": "operator"})

    # Attach Metadata to ALL records
    for entry in data:
        entry["meta"] = entry.get("meta", {})
        entry["meta"]["canon_hash"] = canon_hash

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "tks_gold_v2.jsonl")
    
    with open(out_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {len(data)} gold records to {out_path}")

if __name__ == "__main__":
    generate_gold()
"""
TKS RPM Data Generator (Agent ML)

Generates nested RPM planning samples (Outer:Inner)
Goal -> Prerequisite -> Sub-Prerequisite
"""

import json
import random
import os

GOLD_DIR = os.path.join("canon", "gold")

def generate_rpm(n=200):
    # Templates from strg_llm_convo.txt
    goals = [
        "Build a goal-aware AI",
        "Repair a relationship",
        "Become disciplined"
    ]
    
    foundations = ["1a", "5b", "2c", "7d"]
    
    data = []
    for i in range(n):
        goal = random.choice(goals)
        f_stack = f"{random.choice(foundations)}:{random.choice(foundations)}:1a"
        
        # Simulated RPM Tree
        root = f"(Idea:B10:Q19)"
        prereq = f"(Mind:B1:Q4)"
        sub = f"(Vibration:B4:Q11)"
        
        # Outer:Inner format
        chain = f"{root}:{prereq}:{sub}"
        
        entry = {
            "input": f"Goal: {goal} | Stack: {f_stack}",
            "output": chain,
            "type": "rpm_planning",
            "meta": {"depth": 3}
        }
        data.append(entry)

    out_path = os.path.join(GOLD_DIR, "rpm_planning.jsonl")
    with open(out_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {n} RPM records to {out_path}")

if __name__ == "__main__":
    generate_rpm()

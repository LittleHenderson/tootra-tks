"""
TKS Dataset Coverage Report (Agent Data)

Analyzes the distribution of records across Gold and Mass packs.
"""

import json
import os
from collections import Counter

DATA_DIR = os.path.join("datasets", "jsonl")

def analyze_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {filename}")
        return

    print(f"\n--- Analyzing {filename} ---")
    types = Counter()
    total = 0
    
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                types[rec.get("type", "unknown")] += 1
                total += 1
            except:
                pass
                
    print(f"Total Records: {total}")
    print("Distribution:")
    for t, count in types.most_common():
        print(f"  - {t}: {count} ({count/total*100:.1f}%)")

def main():
    analyze_file("tks_gold_v2.jsonl")
    analyze_file("tks_mass_v1.jsonl")

if __name__ == "__main__":
    main()


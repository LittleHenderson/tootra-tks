"""
TKS Data Leakage Check (Agent Eval)

Verifies that Train and Holdout sets are disjoint.
"""

import json
import os

DATA_DIR = os.path.join("datasets", "jsonl")
TRAIN_FILE = "tks_mass_v1.jsonl"
TEST_FILE = "tks_holdout_v1.jsonl"

def load_hashes(filename):
    hashes = set()
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return hashes
        
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            # Hash the input prompt to check for exact duplicates
            h = hash(entry['input'].strip().lower())
            hashes.add(h)
    return hashes

def check_overlap():
    print(f"Checking overlap between {TRAIN_FILE} and {TEST_FILE}...")
    
    train_hashes = load_hashes(TRAIN_FILE)
    test_hashes = load_hashes(TEST_FILE)
    
    overlap = train_hashes.intersection(test_hashes)
    
    print(f"Train Size: {len(train_hashes)}")
    print(f"Test Size:  {len(test_hashes)}")
    print(f"Overlap:    {len(overlap)}")
    
    if len(overlap) == 0:
        print("\n✅ PASS: No leakage detected.")
    else:
        print(f"\n❌ FAIL: {len(overlap)} records leaked from Train to Test.")
        print("Example leak:", list(overlap)[0])

if __name__ == "__main__":
    check_overlap()

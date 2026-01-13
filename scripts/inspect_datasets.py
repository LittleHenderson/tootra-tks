import json
import os
from collections import Counter

files = [
    "datasets/jsonl/tks_gold_v2.jsonl",
    "datasets/jsonl/tks_mass_v2.jsonl",
    "output/teacher_story_eq_train.jsonl",
    "output/teacher_grounding_curriculum.jsonl",
    "datasets/jsonl/tks_operator_nl2tks.jsonl"
]

for fpath in files:
    if not os.path.exists(fpath):
        print(f"Skipping {fpath} (not found)")
        continue
        
    print(f"\nScanning {fpath}...")
    counts = Counter()
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > 5000: break # Sample first 5000
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    tt = item.get('task_type') or item.get('type') or 'unknown'
                    counts[tt] += 1
                except:
                    pass
        for k, v in counts.most_common():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error reading {fpath}: {e}")

"""
Create Rebalanced Dataset for TKS Training.

Mix:
- 40% Mapping (story_to_eq, mapping, rpm, operator_nl2tks)
- 30% Grounding (teacher_grounding_curriculum)
- 30% Operators/Regulators (tks_gold_v2)

Total target: ~5000 samples.
"""

import json
import random
import os

OUTPUT_FILE = "output/rebalanced_mix.jsonl"
TARGET_TOTAL = 5000

SOURCES = [
    {
        "path": "output/teacher_story_eq_train.jsonl",
        "types": ["story_to_equation"],
        "category": "mapping",
        "weight": 1.0 # Take all
    },
    {
        "path": "datasets/jsonl/tks_operator_nl2tks.jsonl",
        "types": ["operator_nl2tks"],
        "category": "mapping", 
        "target_count": 800
    },
    {
        "path": "datasets/jsonl/tks_mass_v2.jsonl",
        "types": ["mapping", "rpm"],
        "category": "mapping",
        "target_count": 860
    },
    {
        "path": "output/teacher_grounding_curriculum.jsonl",
        "types": None, # All
        "category": "grounding",
        "target_count": 1500 # Will need repeating
    },
    {
        "path": "datasets/jsonl/tks_gold_v2.jsonl",
        "types": ["operator", "regulator_fm", "regulator_acbe", "regulator_mvr", "governance"],
        "category": "operator",
        "target_count": 1500
    }
]

def load_and_filter(source_config):
    path = source_config["path"]
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        return []
        
    valid_types = set(source_config["types"]) if source_config.get("types") else None
    
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                tt = item.get('task_type') or item.get('type')
                if valid_types is None or tt in valid_types:
                    # Normalize to input/output
                    inp = item.get('input') or item.get('prompt')
                    out = item.get('output') or item.get('target') or item.get('completion')
                    
                    if inp and out:
                        new_item = {
                            'input': inp,
                            'output': out,
                            'task_type': tt,
                            'source': path
                        }
                        items.append(new_item)
            except:
                pass
                
    return items

def main():
    final_dataset = []
    
    for config in SOURCES:
        print(f"Processing {config['path']}...")
        items = load_and_filter(config)
        print(f"  Found {len(items)} matching items.")
        
        target = config.get("target_count")
        if target:
            if len(items) == 0:
                continue
            elif len(items) >= target:
                selected = random.sample(items, target)
            else:
                # Repeat to fill
                selected = items * (target // len(items))
                rem = target % len(items)
                selected.extend(random.sample(items, rem))
        else:
            selected = items # Take all
            
        print(f"  Selected {len(selected)} items.")
        final_dataset.extend(selected)
        
    random.shuffle(final_dataset)
    
    print(f"\nTotal dataset size: {len(final_dataset)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item) + '\n')
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

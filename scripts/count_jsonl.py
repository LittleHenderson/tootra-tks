import json
from collections import Counter
import sys

def count_types(path):
    print(f"Counts for {path}:")
    counts = Counter()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                tt = item.get('task_type') or item.get('type') or 'unknown'
                counts[tt] += 1
        for k, v in counts.most_common():
            print(f"  {k}: {v}")
    except FileNotFoundError:
        print("  File not found.")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        count_types(p)

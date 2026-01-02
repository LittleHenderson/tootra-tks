"""Check for content overlap between train and holdout sets."""
import json

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train = load("output/teacher_story_eq_train.jsonl")
hold = load("output/teacher_story_eq_holdout.jsonl")

# Check keys in data
print("Train sample keys:", list(train[0].keys()) if train else "empty")
print("Holdout sample keys:", list(hold[0].keys()) if hold else "empty")

# Get story/equation or input/target based on format
def get_pair(r):
    if "story" in r and "equation" in r:
        return (r["story"], r["equation"])
    elif "input" in r and "target" in r:
        return (r["input"], r["target"])
    return (str(r), "")

train_pairs = {get_pair(r) for r in train}
hold_pairs = {get_pair(r) for r in hold}

print(f"Train unique pairs: {len(train_pairs)}")
print(f"Hold unique pairs: {len(hold_pairs)}")
overlap = train_pairs & hold_pairs
print(f"Overlap count: {len(overlap)}")

if overlap:
    print("\nOverlapping samples found!")
    for p in list(overlap)[:3]:
        print(f"  Input: {p[0][:80]}...")
        print(f"  Target: {p[1][:80]}...")
else:
    print("\nNo overlap - sets are disjoint")

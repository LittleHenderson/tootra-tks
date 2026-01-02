"""
Convert teacher format to quick_train format for MVR training
"""
import json

# Read teacher format
input_file = 'output/teacher_story_eq_train.jsonl'
output_file = 'output/teacher_mvr_converted.jsonl'

lines = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            lines.append(json.loads(line))

print(f"Loaded {len(lines)} samples from {input_file}")

# Convert to quick_train format
converted = []
for line in lines:
    # Combine input and target into story field
    story = f"{line['input']}\n{line['target']}"
    converted.append({
        'story': story,
        'aug_type': 'mvr_foundations'
    })

print(f"Converted {len(converted)} samples")

# Save
with open(output_file, 'w', encoding='utf-8') as f:
    for item in converted:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Saved to {output_file}")
print(f"\nFirst sample:")
print(json.dumps(converted[0], indent=2)[:300])

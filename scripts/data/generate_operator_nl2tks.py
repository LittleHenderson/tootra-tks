"""
Generate NL -> TKS operator training data.

Converts natural language operator descriptions to TKS expressions,
matching the holdout format.

Patterns:
- "binds to" -> +
- "conflicts with" -> /
- "transforms into" -> *T
- "separates from" -> -
"""

import json
import random
import os

# TKS Elements (40 canonical elements)
WORLDS = ['A', 'B', 'C', 'D']
NOETICS = list(range(0, 10))

# Element descriptions (simplified)
ELEMENT_NAMES = {
    'A0': 'Spiritual Concept', 'A1': 'Divine Spark', 'A2': 'Spiritual Alignment',
    'A3': 'Cosmic Truth', 'A4': 'Spiritual Frequency', 'A5': 'Divine Receptivity',
    'A6': 'Divine Will', 'A7': 'Spiritual Pattern', 'A8': 'Divine Cause',
    'A9': 'Spiritual Effect',
    'B0': 'Thought Form', 'B1': 'Mental Energy', 'B2': 'Positive Belief',
    'B3': 'Mental Duality', 'B4': 'Intellectual Intensity', 'B5': 'Mental Receptivity',
    'B6': 'Logical Thinking', 'B7': 'Thought Pattern', 'B8': 'Thought Trigger',
    'B9': 'Cognitive Response',
    'C0': 'Emotional Concept', 'C1': 'Feeling Spark', 'C2': 'Joy/Happiness',
    'C3': 'Emotional Duality', 'C4': 'Emotional Intensity', 'C5': 'Emotional Receptivity',
    'C6': 'Love/Desire', 'C7': 'Mood Cycle', 'C8': 'Emotional Trigger',
    'C9': 'Emotional Response',
    'D0': 'Physical Template', 'D1': 'Body Awareness', 'D2': 'Physical Alignment',
    'D3': 'Material Duality', 'D4': 'Physical Energy', 'D5': 'Physical Woman',
    'D6': 'Physical Man', 'D7': 'Body Rhythm', 'D8': 'Physical Trigger',
    'D9': 'Physical Response',
}

# Operator templates (NL -> operator symbol)
OPERATOR_TEMPLATES = [
    # + (binding/addition)
    {
        'op': '+',
        'templates': [
            "what happens when {e1} binds to {e2}",
            "combine {e1} with {e2}",
            "{e1} joining with {e2}",
            "the union of {e1} and {e2}",
        ]
    },
    # - (separation/subtraction)
    {
        'op': '-',
        'templates': [
            "{e1} separates from {e2}",
            "remove {e2} from {e1}",
            "{e1} without {e2}",
            "the difference between {e1} and {e2}",
        ]
    },
    # / (division/conflict)
    {
        'op': '/',
        'templates': [
            "{e1} conflicts with {e2}",
            "{e1} divided by {e2}",
            "the tension between {e1} and {e2}",
            "{e1} opposing {e2}",
        ]
    },
    # *T (transformation)
    {
        'op': '*T',
        'templates': [
            "{e1} transforms into {e2}",
            "{e1} becoming {e2}",
            "the metamorphosis of {e1} to {e2}",
        ]
    },
]

# Question prefixes
PREFIXES = [
    "Solve TKS Equation:",
    "TKS Operation:",
    "Calculate TKS:",
    "Express in TKS:",
]


def get_element_full_name(elem_code):
    """Get full element name like 'B9.1 Cognitive Response'"""
    return f"{elem_code}.1 {ELEMENT_NAMES.get(elem_code, 'Unknown')}"


def generate_operator_pairs(n_samples=10000):
    """Generate NL -> TKS operator training pairs."""
    pairs = []

    for _ in range(n_samples):
        # Pick random operator
        op_config = random.choice(OPERATOR_TEMPLATES)
        op_symbol = op_config['op']
        template = random.choice(op_config['templates'])

        # Pick two random elements
        w1, w2 = random.choice(WORLDS), random.choice(WORLDS)
        n1, n2 = random.choice(NOETICS), random.choice(NOETICS)
        elem1 = f"{w1}{n1}"
        elem2 = f"{w2}{n2}"

        # Create full names
        e1_full = get_element_full_name(elem1)
        e2_full = get_element_full_name(elem2)

        # Fill template
        question = template.format(e1=e1_full, e2=e2_full)

        # Add prefix
        prefix = random.choice(PREFIXES)
        full_input = f"{prefix} {question}"

        # Create TKS output (using ^1 for Mind noetic as default)
        # Match holdout format: (X^noetic) op (Y^noetic)
        noetic_sense = random.choice([1, 2, 3, 4, 5])  # Vary noetic sense
        output = f"({elem1}^{noetic_sense}) {op_symbol} ({elem2}^{noetic_sense})"

        pairs.append({
            'input': full_input,
            'output': output,
            'type': 'operator_nl2tks',
        })

    return pairs


def main():
    output_dir = "datasets/jsonl"
    os.makedirs(output_dir, exist_ok=True)

    # Generate training data
    print("Generating NL->TKS operator training data...")
    pairs = generate_operator_pairs(n_samples=15000)

    output_path = os.path.join(output_dir, "tks_operator_nl2tks.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"Generated {len(pairs)} operator pairs -> {output_path}")

    # Show samples
    print("\nSamples:")
    for p in pairs[:5]:
        print(f"  Input:  {p['input']}")
        print(f"  Output: {p['output']}")
        print()


if __name__ == "__main__":
    main()

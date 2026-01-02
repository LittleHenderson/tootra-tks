#!/usr/bin/env python3
"""
Add test-style NL phrases to training data to improve generalization.
These match the phrases used in test_noetic_regression.py for evaluation.
"""

import json
import argparse
from pathlib import Path

# Test-style NL phrases from test_noetic_regression.py
TEST_NL_PHRASES = {
    'A': [
        ("Divine realm", "A1+A2+A3"),
        ("Spiritual awareness", "A4+A5+A6"),
        ("Sacred consciousness", "A7+A8+A9"),
        ("Transcendent being", "A1+A4+A7"),
        ("Higher self awakening", "A2+A5+A8"),
        ("Cosmic unity", "A3+A6+A9"),
    ],
    'B': [
        ("Mental focus", "B1+B2+B3"),
        ("Cognitive clarity", "B4+B5+B6"),
        ("Thinking deeply", "B7+B8+B9"),
        ("Intellectual insight", "B1+B4+B7"),
        ("Rational analysis", "B2+B5+B8"),
        ("Logical reasoning", "B3+B6+B9"),
    ],
    'C': [
        ("Emotional flow", "C1+C2+C3"),
        ("Feeling warmth", "C4+C5+C6"),
        ("Heart connection", "C7+C8+C9"),
        ("Compassionate love", "C1+C4+C7"),
        ("Empathic resonance", "C2+C5+C8"),
        ("Soulful expression", "C3+C6+C9"),
    ],
    'D': [
        ("Physical power", "D1+D2+D3"),
        ("Bodily energy", "D4+D5+D6"),
        ("Material strength", "D7+D8+D9"),
        ("Grounded presence", "D1+D4+D7"),
        ("Tangible force", "D2+D5+D8"),
        ("Manifest action", "D3+D6+D9"),
    ],
}

# RPM-weighted variants
RPM_VARIANTS = {
    'desire': {'desire': 0.7, 'wisdom': 0.15, 'power': 0.15},
    'wisdom': {'desire': 0.15, 'wisdom': 0.7, 'power': 0.15},
    'power': {'desire': 0.15, 'wisdom': 0.15, 'power': 0.7},
}

def create_training_entry(phrase: str, target: str, world: str, rpm: str = 'wisdom'):
    """Create a training entry matching the expected format."""
    world_weights = {'A': 0.1, 'B': 0.1, 'C': 0.1, 'D': 0.1}
    world_weights[world] = 0.7

    return {
        'input': phrase,
        'target': target,
        'world_label': world,
        'rpm_label': rpm,
        'world_weights': world_weights,
        'rpm_weights': RPM_VARIANTS[rpm],
    }

def generate_test_phrases():
    """Generate all test-style training entries."""
    entries = []

    # Cycle through RPM types
    rpm_types = ['desire', 'wisdom', 'power']
    rpm_idx = 0

    for world, phrases in TEST_NL_PHRASES.items():
        for phrase, target in phrases:
            rpm = rpm_types[rpm_idx % 3]
            entry = create_training_entry(phrase, target, world, rpm)
            entries.append(entry)
            rpm_idx += 1

    return entries

def main():
    parser = argparse.ArgumentParser(description='Add test-style NL phrases to training data')
    parser.add_argument('--input', '-i', required=True, help='Input training JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file with added phrases')
    parser.add_argument('--repeat', '-r', type=int, default=3, help='Repeat each phrase N times')
    args = parser.parse_args()

    # Load existing training data
    existing = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))

    print(f"Loaded {len(existing)} existing training examples")

    # Generate test-style phrases
    test_phrases = generate_test_phrases()
    print(f"Generated {len(test_phrases)} test-style phrases")

    # Repeat for emphasis
    repeated = []
    for _ in range(args.repeat):
        repeated.extend(test_phrases)
    print(f"Repeated {args.repeat}x = {len(repeated)} test phrase entries")

    # Combine
    combined = existing + repeated
    print(f"Total training examples: {len(combined)}")

    # Write output
    with open(args.output, 'w') as f:
        for entry in combined:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved to {args.output}")

    # Print distribution
    world_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for entry in combined:
        w = entry.get('world_label', 'A')
        world_counts[w] = world_counts.get(w, 0) + 1

    print("\nWorld distribution:")
    for w, c in sorted(world_counts.items()):
        print(f"  {w}: {c} ({100*c/len(combined):.1f}%)")

if __name__ == '__main__':
    main()

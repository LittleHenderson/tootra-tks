#!/usr/bin/env python3
"""Show detailed traces for selected questions."""

import sys
sys.path.insert(0, '.')

from tks_features.rtta_canonical import RTTACanonical

rtta = RTTACanonical()

# 5 PASSED examples
passed = [
    ("05", "linear_system", "A person has nickels and dimes totaling 8 coins worth 65 cents. How many dimes?", "5"),
    ("11", "syllogism", "All cats are mammals. All mammals are animals. Are cats animals?", "Yes"),
    ("14", "conditional", "If it is raining, the ground is wet. The ground is wet. Did it rain?", "Cannot determine"),
    ("16", "sequence", "What comes next in the sequence: 1, 4, 9, 16, ?", "25"),
    ("18", "sequence", "What comes next in the sequence: 1, 2, 4, 7, 11, ?", "16"),
]

# 5 FAILED examples (expected failures - need new operators)
failed = [
    ("21", "logic_grid", "Ann does not have a cat. Ben does not have a dog. Cal has a fish. Who has the dog?", "Ann"),
    ("27", "probability", "You flip two coins. What is the probability of getting exactly one head?", "1/2"),
    ("31", "geometry", "A right triangle has legs 3 and 4. What is the hypotenuse?", "5"),
    ("37", "modular", "What is 3^4 mod 5?", "1"),
    ("39", "conditional", "Given P implies Q, Q implies R, and not R. What follows?", "not P"),
]

print("=" * 80)
print("5 PASSED EXAMPLES")
print("=" * 80)

for test_id, test_type, question, expected in passed:
    print(f"\n{'='*80}")
    print(f"[{test_id}] {test_type.upper()}")
    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print("-" * 80)

    result = rtta.reason(question, verbose=True)

    print(f"\nANSWER: {result['answer']}")
    print(f"Trace: {result['trace']}")

print("\n" + "=" * 80)
print("5 FAILED EXAMPLES (need new operators)")
print("=" * 80)

for test_id, test_type, question, expected in failed:
    print(f"\n{'='*80}")
    print(f"[{test_id}] {test_type.upper()}")
    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print("-" * 80)

    result = rtta.reason(question, verbose=True)

    print(f"\nANSWER: {result['answer']}")
    print(f"Missing: {test_type} operator not implemented")
    print(f"Trace: {result['trace']}")

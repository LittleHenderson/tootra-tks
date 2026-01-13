#!/usr/bin/env python3
"""Test RTTA-C with diverse reasoning questions."""

import sys
sys.path.insert(0, '.')

from tks_features.rtta_canonical import RTTACanonical

rtta = RTTACanonical()

test_cases = [
    # Math word problems
    ("A store sells apples for $1 each and oranges for $2 each. A customer buys 10 items in total and spends $15. How many apples did the customer buy?", "5 apples"),
    ("A bakery sells cookies for $2 each and cakes for $10 each. A customer buys 7 items in total and spends $38. How many cookies did the customer buy?", "4 cookies"),

    # Logic / Syllogisms
    ("All dogs are mammals. All mammals are animals. Is it true that all dogs are animals?", "Yes"),
    ("All roses are flowers. All flowers need water. Do roses need water?", "Yes"),

    # Sequences
    ("What comes next in the sequence: 2, 4, 6, 8, ?", "10"),
    ("What comes next in the sequence: 1, 4, 9, 16, ?", "25"),

    # Conditional reasoning
    ("If it is raining, the ground is wet. The ground is wet. Did it rain?", "Maybe"),
    ("If a number is divisible by 4, it is even. 12 is divisible by 4. Is 12 even?", "Yes"),

    # Proportion/ratio
    ("A recipe uses 2 cups of flour for every 3 cups of sugar. If you use 6 cups of flour, how many cups of sugar do you need?", "9"),

    # Simple arithmetic word problems
    ("Tom has 15 marbles. He gives 7 to his friend. How many marbles does Tom have now?", "8"),
]

print("=" * 70)
print("RTTA-C CANONICAL REASONING TEST")
print("=" * 70)

correct = 0
total = len(test_cases)

for question, expected in test_cases:
    print(f"\n{'='*70}")
    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print("-" * 70)

    result = rtta.reason(question, verbose=False)
    answer = result['answer']

    print(f"RTTA Answer: {answer}")
    print(f"Iterations: {result['iterations']}")
    print(f"Prereqs: {result['prereqs']}")

    # Check if answer matches (loose matching)
    answer_lower = str(answer).lower()
    expected_lower = expected.lower()

    # Extract just the number/word for comparison
    match = False
    if expected_lower in answer_lower:
        match = True
    elif answer_lower in expected_lower:
        match = True
    # Special case: "maybe" and "cannot be determined" are equivalent
    if expected_lower == "maybe" and "cannot be determined" in answer_lower:
        match = True
    # Check numeric match
    try:
        ans_num = int(''.join(c for c in str(answer) if c.isdigit()))
        exp_num = int(''.join(c for c in expected if c.isdigit()))
        if ans_num == exp_num:
            match = True
    except:
        pass

    status = "PASS" if match else "FAIL"
    print(f"Status: {status}")

    if match:
        correct += 1

print(f"\n{'='*70}")
print(f"SUMMARY: {correct}/{total} correct ({100*correct/total:.1f}%)")
print("=" * 70)

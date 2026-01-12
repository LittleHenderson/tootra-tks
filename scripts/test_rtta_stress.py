#!/usr/bin/env python3
"""RTTA-C Stress Test - 40 adversarial questions with trace saving."""

import sys
import json
import os
from datetime import datetime

sys.path.insert(0, '.')

from tks_features.rtta_canonical import RTTACanonical

rtta = RTTACanonical()

# Format: (ID, Type, Question, Expected, Slot_tag)
# Slot_tag: "pass" = should work, "5W" = likely fails at Mental Wisdom, etc.
stress_tests = [
    # Linear systems (should pass)
    ("01", "linear_system", "A store sells apples for $1 each and oranges for $2 each. A customer buys 10 items and spends $15. How many apples?", "5", "pass"),
    ("02", "linear_system", "A bakery sells cookies for $2 each and cakes for $10 each. A customer buys 7 items and spends $38. How many cookies?", "4", "pass"),
    ("03", "linear_system", "A store sells pencils for $1 each and pens for $3 each. A customer buys 6 items and spends $14. How many pencils?", "2", "pass"),
    ("04", "linear_system", "Adult tickets cost $8 and child tickets cost $4. A group buys 7 tickets for $44 total. How many adult tickets?", "4", "pass"),
    ("05", "linear_system", "A person has nickels and dimes totaling 8 coins worth 65 cents. How many dimes?", "5", "pass"),

    # Arithmetic
    ("06", "arithmetic", "Tom has 15 marbles. He gives 7 to his friend. How many marbles does Tom have now?", "8", "pass"),
    ("07", "arithmetic", "What is 24 / 6 + 3?", "7", "5W"),

    # Proportion
    ("08", "proportion", "A recipe uses 2 cups of flour for every 3 cups of sugar. If you use 6 cups of flour, how many cups of sugar do you need?", "9", "pass"),

    # Rate (new type)
    ("09", "rate", "A car travels 180 miles in 3 hours. What is the speed in mph?", "60", "5W"),

    # Age problems (new type)
    ("10", "age", "Sara is 5 years older than Ben. The sum of their ages is 27. How old is Ben?", "11", "5W"),

    # Syllogisms
    ("11", "syllogism", "All cats are mammals. All mammals are animals. Are cats animals?", "Yes", "pass"),
    ("12", "syllogism", "Some A are B. All B are C. Are some A also C?", "Yes", "5W"),

    # Conditionals
    ("13", "conditional", "If a number is divisible by 6, then it is divisible by 3. 18 is divisible by 6. Is 18 divisible by 3?", "Yes", "pass"),
    ("14", "conditional", "If it is raining, the ground is wet. The ground is wet. Did it rain?", "Cannot determine", "pass"),

    # Sequences - easy
    ("15", "sequence", "What comes next in the sequence: 2, 4, 6, 8, ?", "10", "pass"),
    ("16", "sequence", "What comes next in the sequence: 1, 4, 9, 16, ?", "25", "pass"),

    # Sequences - hard (expected to fail)
    ("17", "sequence", "What comes next in the sequence: 1, 3, 2, 4, 3, 5, ?", "4", "5W"),
    ("18", "sequence", "What comes next in the sequence: 1, 2, 4, 7, 11, ?", "16", "5W"),
    ("19", "sequence", "What comes next in the sequence: 2, 3, 5, 7, 11, ?", "13", "5W"),
    ("20", "sequence", "What comes next in the sequence: 1, 2, 6, 24, ?", "120", "5W"),

    # Questions 21-40
    ("21", "logic_grid", "Ann does not have a cat. Ben does not have a dog. Cal has a fish. Who has the dog?", "Ann", "6P"),
    ("22", "logic_grid", "Alice wears red. Bob does not wear blue. Carol does not wear red. Who wears blue?", "Carol", "6P"),
    ("23", "logic_grid", "Dan plays soccer. Eli does not play tennis. Fay does not play soccer. Who plays tennis?", "Fay", "6P"),
    ("24", "logic_xor", "Exactly one of A and B is true. A is false. Is B true?", "Yes", "pass"),
    ("25", "combinatorics", "How many 2-letter strings can be made from A, B, C without repetition?", "6", "5W"),
    ("26", "combinatorics", "How many ways can you choose 2 items from 5?", "10", "5W"),
    ("27", "probability", "You flip two coins. What is the probability of getting exactly one head?", "1/2", "5W"),
    ("28", "probability", "Two dice are rolled and sum to 9. What is the probability that one die shows 4?", "1/2", "5W"),
    ("29", "bayes", "A disease has 1% prevalence, 90% sensitivity, and 9% false positive rate. What is P(disease|positive)?", "9.2%", "5W"),
    ("30", "geometry", "A rectangle has sides 8 and 3. What is its area?", "24", "5W"),
    ("31", "geometry", "A right triangle has legs 3 and 4. What is the hypotenuse?", "5", "5W"),
    ("32", "geometry", "A triangle has angles 30 and 50 degrees. What is the third angle?", "100", "5W"),
    ("33", "coord_geom", "What is the slope between points (1,2) and (5,10)?", "2", "5W"),
    ("34", "unit_convert", "How many minutes is 2.5 hours?", "150", "11W"),
    ("35", "rate", "You walk 3 mph for 2 hours, then 4 mph for 1 hour. Total distance?", "10", "pass"),
    ("36", "inequality", "If x + 3 > 10, what is the smallest integer x?", "8", "5W"),
    ("37", "modular", "What is 3^4 mod 5?", "1", "5W"),
    ("38", "base_convert", "What is 1011 in base 2 converted to decimal?", "11", "5W"),
    ("39", "conditional", "Given P implies Q, Q implies R, and not R. What follows?", "not P", "6P"),
    ("40", "sets", "30 students total, 18 take math, 12 take science, 5 take both. How many take neither?", "5", "5W"),
]

print("=" * 80)
print("RTTA-C STRESS TEST")
print("=" * 80)

results = {"pass": 0, "fail": 0, "expected_fail": 0}
details = []

for test_id, test_type, question, expected, slot_tag in stress_tests:
    result = rtta.reason(question, verbose=False)
    answer = str(result['answer'])

    # Check match
    answer_lower = answer.lower()
    expected_lower = expected.lower()

    match = False
    if expected_lower in answer_lower or answer_lower in expected_lower:
        match = True
    if expected_lower == "cannot determine" and "cannot" in answer_lower:
        match = True
    try:
        ans_num = int(''.join(c for c in answer if c.isdigit() or c == '-'))
        exp_num = int(''.join(c for c in expected if c.isdigit() or c == '-'))
        if ans_num == exp_num:
            match = True
    except:
        pass

    # Determine status
    if match:
        status = "PASS"
        results["pass"] += 1
    elif slot_tag != "pass":
        status = f"EXPECTED_FAIL ({slot_tag})"
        results["expected_fail"] += 1
    else:
        status = "FAIL"
        results["fail"] += 1

    # Get first unmet slot
    prereqs = result.get('prereqs', {})
    first_unmet = "none"
    for slot in ['4D', '5W', '6P']:
        if slot in prereqs and not prereqs[slot].get('passed', True):
            first_unmet = slot
            break

    details.append({
        "id": test_id,
        "type": test_type,
        "expected": expected,
        "got": answer[:30],
        "status": status,
        "iterations": result['iterations'],
        "first_unmet": first_unmet if not match else "-"
    })

    print(f"{test_id} | {test_type:14} | {status:20} | Expected: {expected:10} | Got: {answer[:25]}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  PASS:          {results['pass']}/{len(stress_tests)}")
print(f"  FAIL:          {results['fail']}/{len(stress_tests)}")
print(f"  EXPECTED_FAIL: {results['expected_fail']}/{len(stress_tests)}")
print(f"  Accuracy:      {100*results['pass']/len(stress_tests):.1f}%")
print(f"  Baseline:      {100*(results['pass']+results['expected_fail'])/len(stress_tests):.1f}% (including expected fails)")

# Show failures
print()
print("FAILURES (unexpected):")
for d in details:
    if d['status'] == 'FAIL':
        print(f"  {d['id']}: {d['type']} - Expected '{d['expected']}', got '{d['got']}'")

print()
print("EXPECTED FAILURES (need new operators):")
for d in details:
    if 'EXPECTED_FAIL' in d['status']:
        print(f"  {d['id']}: {d['type']} - {d['status']}")

# Save traces for regression tracking
trace_dir = "logs"
os.makedirs(trace_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trace_file = os.path.join(trace_dir, f"rtta_traces_stress_{timestamp}.json")

with open(trace_file, 'w') as f:
    json.dump({
        "timestamp": timestamp,
        "suite": "stress_test",
        "total": len(stress_tests),
        "passed": results['pass'],
        "failed": results['fail'],
        "expected_fail": results['expected_fail'],
        "accuracy": results['pass'] / len(stress_tests),
        "details": details
    }, f, indent=2)

print(f"\nTraces saved to: {trace_file}")

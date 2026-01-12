#!/usr/bin/env python3
"""RTTA-C Hard Mode Test Suite - 60 adversarial questions with mixed phrasing."""

import sys
import json
import os
from datetime import datetime

sys.path.insert(0, '.')

from tks_features.rtta_canonical import RTTACanonical

rtta = RTTACanonical()

# Hard mode: adversarial phrasing, edge cases, mixed facts
# Format: (ID, Type, Question, Expected)
hard_tests = [
    # === LINEAR SYSTEMS - tricky phrasing ===
    ("H01", "linear_system", "Pens cost $3, pencils cost $1. I bought some of each, 5 total, spent $11. Pens purchased?", "3"),
    ("H02", "linear_system", "At the fair, rides are $4 and games are $2. With $20 I did 7 activities. How many rides?", "3"),
    ("H03", "linear_system", "Mixing nickels and quarters: 12 coins, $1.80 total. Quarter count?", "5"),
    ("H04", "linear_system", "Books=$5, magazines=$2. Bought 8 items for $31. Books?", "5"),
    ("H05", "linear_system", "Large pizzas $15, small $8. Ordered 6 pizzas for $76. Large pizzas ordered?", "4"),

    # === ARITHMETIC - PEMDAS edge cases ===
    ("H06", "arithmetic", "What is 100 / 10 / 2?", "5"),
    ("H07", "arithmetic", "What is 2 + 3 * 4?", "14"),
    ("H08", "arithmetic", "What is 20 - 4 * 3?", "8"),
    ("H09", "arithmetic", "What is 8 + 16 / 4?", "12"),
    ("H10", "arithmetic", "What is 50 - 30 + 10?", "30"),

    # === SEQUENCES - harder patterns ===
    ("H11", "sequence", "Next in: 3, 6, 12, 24, ?", "48"),
    ("H12", "sequence", "Continue: 5, 10, 20, 40, ?", "80"),
    ("H13", "sequence", "Pattern: 1, 1, 2, 3, 5, 8, ?", "13"),  # Fibonacci
    ("H14", "sequence", "What follows: 2, 6, 12, 20, ?", "30"),  # n*(n+1)
    ("H15", "sequence", "Next: 0, 1, 1, 2, 3, 5, ?", "8"),  # Fibonacci starting 0

    # === SYLLOGISMS - varied phrasing ===
    ("H16", "syllogism", "Every bird can fly. Sparrows are birds. Can sparrows fly?", "Yes"),
    ("H17", "syllogism", "All squares are rectangles. All rectangles have 4 sides. Do squares have 4 sides?", "Yes"),
    ("H18", "syllogism", "Some students play sports. All who play sports exercise. Do some students exercise?", "Yes"),
    ("H19", "syllogism", "All prime numbers greater than 2 are odd. 7 is prime and greater than 2. Is 7 odd?", "Yes"),
    ("H20", "syllogism", "All metals conduct electricity. Copper is a metal. Does copper conduct electricity?", "Yes"),

    # === CONDITIONALS - tricky logic ===
    ("H21", "conditional", "When it snows, schools close. Schools are closed. Did it snow?", "Cannot determine"),
    ("H22", "conditional", "If x > 5 then x > 3. x = 7. Is x > 3?", "Yes"),
    ("H23", "conditional", "Wet grass implies it rained or sprinklers ran. Grass is wet. Did it definitely rain?", "Cannot determine"),
    ("H24", "conditional", "If the alarm rings, wake up. The alarm rang. Should you wake up?", "Yes"),
    ("H25", "conditional", "Passing requires 60+ points. You got 75. Did you pass?", "Yes"),

    # === LOGIC GRIDS - harder elimination ===
    ("H26", "logic_grid", "Tom has neither the cat nor the bird. Sue has the cat. Who has the bird?", "Tom"),
    ("H27", "logic_grid", "Red is not worn by Amy. Blue is not worn by Ben. Amy wears blue. Who wears red?", "Ben"),
    ("H28", "logic_grid", "Math is taught by Mr. X. Science is not taught by Ms. Y. Ms. Y teaches art. Who teaches science?", "Mr"),

    # === PROBABILITY - varied scenarios ===
    ("H29", "probability", "Roll one die. What is the probability of getting an even number?", "1/2"),
    ("H30", "probability", "Flip three coins. Probability of all heads?", "1/8"),
    ("H31", "probability", "Draw from a deck. Probability of a heart?", "1/4"),

    # === GEOMETRY - mixed shapes ===
    ("H32", "geometry_area", "Square with side 7. Area?", "49"),
    ("H33", "geometry_angles", "Isoceles triangle: two angles are 70 degrees each. Third angle?", "40"),
    ("H34", "geometry_pythagorean", "Right triangle: one leg is 5, hypotenuse is 13. Other leg?", "12"),
    ("H35", "geometry_area", "Rectangle: length 12, width 5. Area?", "60"),

    # === COORDINATE GEOMETRY ===
    ("H36", "coord_geometry", "Find slope between (0,0) and (4,8).", "2"),
    ("H37", "coord_geometry", "Slope of line through (2,3) and (6,11)?", "2"),
    ("H38", "coord_geometry", "Points (1,1) and (4,7). Slope?", "2"),

    # === MODULAR ARITHMETIC ===
    ("H39", "modular", "What is 7^3 mod 10?", "3"),
    ("H40", "modular", "Calculate 2^10 mod 7.", "2"),
    ("H41", "modular", "Find 5^4 mod 13.", "1"),

    # === BASE CONVERSION ===
    ("H42", "base_convert", "Convert 1111 from base 2 to decimal.", "15"),
    ("H43", "base_convert", "Binary 10101 to decimal?", "21"),
    ("H44", "base_convert", "What is 11001 in base 2 as a decimal number?", "25"),

    # === UNIT CONVERSION ===
    ("H45", "unit_convert", "Convert 3 hours to minutes.", "180"),
    ("H46", "unit_convert", "How many minutes in 1.5 hours?", "90"),
    ("H47", "unit_convert", "Express 4.25 hours in minutes.", "255"),

    # === RATE PROBLEMS ===
    ("H48", "rate", "Car goes 240 miles in 4 hours. Speed in mph?", "60"),
    ("H49", "rate", "Bike at 15 mph for 2 hours, then 10 mph for 3 hours. Total distance?", "60"),
    ("H50", "rate", "Train travels 300 miles at 60 mph. Time in hours?", "5"),

    # === AGE PROBLEMS ===
    ("H51", "age", "Mom is 25 years older than daughter. Together they are 45. Daughter's age?", "10"),
    ("H52", "age", "Brother is 3 years younger than sister. Sum of ages is 17. Sister's age?", "10"),
    ("H53", "age", "Father is 30 years older than son. Their ages sum to 50. How old is the son?", "10"),

    # === SETS / INCLUSION-EXCLUSION ===
    ("H54", "sets", "Class of 40: 25 like math, 20 like science, 10 like both. How many like neither?", "5"),
    ("H55", "sets", "50 people: 30 drink coffee, 25 drink tea, 15 drink both. Neither?", "10"),

    # === COMBINATORICS ===
    ("H56", "combinatorics", "Permutations of 3 items from 4?", "24"),
    ("H57", "combinatorics", "How many ways to select 3 from 6?", "20"),
    ("H58", "combinatorics", "3-letter arrangements from A,B,C,D without repetition?", "24"),

    # === INEQUALITY ===
    ("H59", "inequality", "If 2x + 1 > 9, smallest integer x?", "5"),
    ("H60", "inequality", "For x - 5 > 3, what is the minimum integer x?", "9"),
]

def run_hard_suite(save_traces=True):
    """Run hard mode suite and optionally save traces."""
    print("=" * 80)
    print("RTTA-C HARD MODE TEST SUITE")
    print("=" * 80)

    results = {"pass": 0, "fail": 0}
    all_traces = []

    for test_id, test_type, question, expected in hard_tests:
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
            # Extract numbers for comparison
            ans_num = int(''.join(c for c in answer if c.isdigit() or c == '-')[:10])
            exp_num = int(''.join(c for c in expected if c.isdigit() or c == '-')[:10])
            if ans_num == exp_num:
                match = True
        except:
            pass
        # Fraction match
        if '/' in expected and '/' in answer:
            if expected == answer:
                match = True

        status = "PASS" if match else "FAIL"
        results["pass" if match else "fail"] += 1

        # Save trace
        trace_entry = {
            "id": test_id,
            "type": test_type,
            "question": question,
            "expected": expected,
            "got": answer,
            "status": status,
            "iterations": result['iterations'],
            "trace": result['trace'],
            "prereqs": result['prereqs']
        }
        all_traces.append(trace_entry)

        print(f"{test_id} | {test_type:18} | {status:4} | Expected: {expected:15} | Got: {answer[:25]}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  PASS: {results['pass']}/{len(hard_tests)}")
    print(f"  FAIL: {results['fail']}/{len(hard_tests)}")
    print(f"  Accuracy: {100*results['pass']/len(hard_tests):.1f}%")

    # Save traces
    if save_traces:
        trace_dir = "logs"
        os.makedirs(trace_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_file = os.path.join(trace_dir, f"rtta_traces_hard_{timestamp}.json")

        with open(trace_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "suite": "hard_mode",
                "total": len(hard_tests),
                "passed": results['pass'],
                "failed": results['fail'],
                "accuracy": results['pass'] / len(hard_tests),
                "traces": all_traces
            }, f, indent=2)

        print(f"\n  Traces saved to: {trace_file}")

    # Show failures
    if results['fail'] > 0:
        print()
        print("FAILURES:")
        for t in all_traces:
            if t['status'] == 'FAIL':
                print(f"  {t['id']}: {t['type']}")
                print(f"       Q: {t['question'][:60]}...")
                print(f"       Expected: {t['expected']}, Got: {t['got']}")

    return results, all_traces


if __name__ == "__main__":
    run_hard_suite(save_traces=True)

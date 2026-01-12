#!/usr/bin/env python3
"""
RTTA-C + RTTA-R Integration Test

Runs hard mode suite and logs all failures to the rumination system,
then runs rumination cycles to analyze patterns.
"""

import sys
sys.path.insert(0, '.')

from tks_features.rtta_canonical import RTTACanonical
from tks_features.rtta_rumination import RTTARumination

# Import hard tests
from scripts.test_rtta_hard import hard_tests

def run_with_rumination():
    """Run hard mode with rumination logging."""
    # Create RTTA-C
    rtta = RTTACanonical()

    # Create RTTA-R
    rumination = RTTARumination(rtta.reason)

    print("=" * 70)
    print("RTTA-C + RTTA-R INTEGRATION TEST")
    print("=" * 70)

    results = {"pass": 0, "fail": 0}

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
            ans_num = int(''.join(c for c in answer if c.isdigit() or c == '-')[:10])
            exp_num = int(''.join(c for c in expected if c.isdigit() or c == '-')[:10])
            if ans_num == exp_num:
                match = True
        except:
            pass
        if '/' in expected and '/' in answer and expected == answer:
            match = True

        if match:
            results["pass"] += 1
            status = "PASS"
        else:
            results["fail"] += 1
            status = "FAIL"
            # Log to rumination
            record = rumination.log_failure(test_id, question, expected, answer, result)
            status = f"FAIL (impact={record.impact_score:.2f})"

        print(f"{test_id} | {test_type:18} | {status}")

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  PASS: {results['pass']}/{len(hard_tests)}")
    print(f"  FAIL: {results['fail']}/{len(hard_tests)}")
    print(f"  Accuracy: {100*results['pass']/len(hard_tests):.1f}%")

    print()
    print("=" * 70)
    print("RUMINATION ANALYSIS")
    print("=" * 70)
    print(rumination.report())

    # Run rumination cycles
    print()
    print("Running 20 rumination cycles...")
    rumination.ruminate_now(budget=20)

    print()
    print("Post-rumination report:")
    print(rumination.report())

    # Show failure patterns by type
    print()
    print("=" * 70)
    print("FAILURE PATTERNS BY TYPE")
    print("=" * 70)

    type_counts = {}
    slot_counts = {}

    for f in rumination.ledger.failures.values():
        t = f.gist.type
        s = f.first_unmet_slot
        type_counts[t] = type_counts.get(t, 0) + 1
        slot_counts[s] = slot_counts.get(s, 0) + 1

    print("\nBy problem type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20}: {count}")

    print("\nBy blocked slot:")
    for s, count in sorted(slot_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {count}")

    # Show proposed operators
    print()
    print("=" * 70)
    print("PROPOSED OPERATORS")
    print("=" * 70)
    for name, op in rumination.operator_bank.proposed.items():
        print(f"  {name}")
        print(f"    trigger: {op.trigger_pattern}")
        print(f"    confidence: {op.confidence:.2f}")

    return rumination


if __name__ == "__main__":
    run_with_rumination()

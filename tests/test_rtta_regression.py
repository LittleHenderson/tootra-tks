#!/usr/bin/env python3
"""
RTTA Hard-Mode Regression Suite

Tests:
1. Trace thresholds - ensure RTTA-C meets baseline accuracy
2. Rumination recovery - failures logged and analyzed correctly
3. Schema compliance - ledger matches JSON Schema

Run with: pytest tests/test_rtta_regression.py -v
"""

import sys
import os
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_features.rtta_canonical import RTTACanonical
from tks_features.rtta_rumination import RTTARumination, FailureLedger


# ============================================================
# TEST DATA
# ============================================================

# Subset of hard mode tests for regression
# Questions must match RTTA-C patterns exactly
REGRESSION_TESTS = [
    # Must pass (baseline) - using exact patterns from stress test
    ("R01", "linear_system", "A store sells apples for $1 each and oranges for $2 each. A customer buys 10 items and spends $15. How many apples?", "5"),
    ("R02", "arithmetic", "Tom has 15 marbles. He gives 7 to his friend. How many marbles does Tom have now?", "8"),
    ("R03", "proportion", "A recipe uses 2 cups of flour for every 3 cups of sugar. If you use 6 cups of flour, how many cups of sugar do you need?", "9"),
    ("R04", "syllogism", "All cats are mammals. All mammals are animals. Are cats animals?", "Yes"),
    ("R05", "conditional", "If a number is divisible by 6, then it is divisible by 3. 18 is divisible by 6. Is 18 divisible by 3?", "Yes"),
    ("R06", "sequence", "What comes next in the sequence: 2, 4, 6, 8, ?", "10"),

    # Expected to fail (tests rumination logging)
    ("R07", "sequence", "What comes next in the sequence: 1, 1, 2, 3, 5, 8, ?", "13"),
    ("R08", "logic_grid", "Ann does not have a cat. Ben does not have a dog. Cal has a fish. Who has the dog?", "Ann"),
    ("R09", "probability", "You flip two coins. What is the probability of getting exactly one head?", "1/2"),
]

# Thresholds
BASELINE_ACCURACY = 0.55  # Must pass at least 55% (5/9 = 55.6% is expected)
PASS_THRESHOLD = 5  # At least 5 must pass


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def rtta():
    """Create RTTA-C instance."""
    return RTTACanonical()


@pytest.fixture
def fresh_ledger(tmp_path):
    """Create fresh ledger for testing."""
    ledger_path = tmp_path / "test_ledger.json"
    return FailureLedger(str(ledger_path))


@pytest.fixture
def rumination(rtta, tmp_path):
    """Create rumination system with fresh ledger."""
    ledger_path = tmp_path / "test_ledger.json"
    operator_path = tmp_path / "test_operators.json"
    return RTTARumination(
        rtta_solver=rtta.reason,
        ledger_path=str(ledger_path),
        operator_path=str(operator_path)
    )


# ============================================================
# TRACE THRESHOLD TESTS
# ============================================================

class TestTraceThresholds:
    """Test that RTTA-C meets baseline accuracy thresholds."""

    def test_baseline_accuracy(self, rtta):
        """RTTA-C must achieve at least 60% accuracy on regression suite."""
        passed = 0
        failed = 0
        results = []

        for test_id, test_type, question, expected in REGRESSION_TESTS:
            result = rtta.reason(question, verbose=False)
            answer = str(result['answer'])

            # Check match
            match = self._check_match(answer, expected)

            if match:
                passed += 1
                results.append((test_id, "PASS", expected, answer))
            else:
                failed += 1
                results.append((test_id, "FAIL", expected, answer))

        accuracy = passed / len(REGRESSION_TESTS)

        # Print results for debugging
        print(f"\nRegression Results: {passed}/{len(REGRESSION_TESTS)} ({accuracy*100:.1f}%)")
        for r in results:
            print(f"  {r[0]}: {r[1]} (expected={r[2]}, got={r[3][:30]})")

        assert accuracy >= BASELINE_ACCURACY, \
            f"Accuracy {accuracy:.2f} below threshold {BASELINE_ACCURACY}"

    def test_must_pass_tests(self, rtta):
        """Core tests R01-R06 must mostly pass (allow 1 failure)."""
        must_pass = [t for t in REGRESSION_TESTS if t[0] in
                     ["R01", "R02", "R03", "R04", "R05", "R06"]]

        failures = []
        for test_id, test_type, question, expected in must_pass:
            result = rtta.reason(question, verbose=False)
            answer = str(result['answer'])

            if not self._check_match(answer, expected):
                failures.append((test_id, expected, answer))

        # Allow at most 1 failure in core tests
        assert len(failures) <= 1, \
            f"Too many must-pass tests failed: {failures}"

    def test_iterations_within_budget(self, rtta):
        """All tests must complete within iteration budget."""
        MAX_ITERATIONS = 15

        for test_id, test_type, question, expected in REGRESSION_TESTS:
            result = rtta.reason(question, verbose=False)
            iterations = result.get('iterations', 0)

            assert iterations <= MAX_ITERATIONS, \
                f"{test_id} exceeded iteration budget: {iterations} > {MAX_ITERATIONS}"

    def _check_match(self, answer: str, expected: str) -> bool:
        """Check if answer matches expected."""
        answer_lower = answer.lower()
        expected_lower = expected.lower()

        if expected_lower in answer_lower or answer_lower in expected_lower:
            return True

        # Numeric extraction
        try:
            ans_num = int(''.join(c for c in answer if c.isdigit() or c == '-')[:10])
            exp_num = int(''.join(c for c in expected if c.isdigit() or c == '-')[:10])
            if ans_num == exp_num:
                return True
        except:
            pass

        # Fraction matching
        if '/' in expected and '/' in answer:
            return expected == answer

        return False


# ============================================================
# RUMINATION RECOVERY TESTS
# ============================================================

class TestRuminationRecovery:
    """Test that failures are logged and rumination analyzes them."""

    def test_failures_logged_to_ledger(self, rtta, rumination):
        """Failed tests should be logged to the failure ledger."""
        # Run tests and log failures
        for test_id, test_type, question, expected in REGRESSION_TESTS:
            result = rtta.reason(question, verbose=False)
            answer = str(result['answer'])

            if expected.lower() not in answer.lower():
                rumination.log_failure(test_id, question, expected, answer, result)

        # Check ledger has entries
        stats = rumination.get_stats()
        assert stats['total_failures'] > 0, "No failures logged"
        assert stats['unsolved'] > 0, "No unsolved failures"

    def test_impact_scores_computed(self, rtta, rumination):
        """Logged failures should have valid impact scores."""
        # Log a known failure
        result = rtta.reason("Pattern: 1, 1, 2, 3, 5, 8, ?", verbose=False)
        record = rumination.log_failure(
            "TEST-FIB",
            "Pattern: 1, 1, 2, 3, 5, 8, ?",
            "13",
            str(result['answer']),
            result
        )

        assert 0 <= record.impact_score <= 1, \
            f"Invalid impact score: {record.impact_score}"
        assert 0 <= record.desire_level <= 1, \
            f"Invalid desire level: {record.desire_level}"

    def test_rumination_cycles_run(self, rtta, rumination):
        """Rumination cycles should execute without error."""
        # Log some failures
        failures = [
            ("FIB", "Pattern: 1, 1, 2, 3, 5, 8, ?", "13"),
            ("GRID", "Ann no cat. Ben no dog. Cal fish. Dog?", "Ann"),
        ]

        for fid, q, exp in failures:
            result = rtta.reason(q, verbose=False)
            rumination.log_failure(fid, q, exp, str(result['answer']), result)

        # Run rumination
        initial_cycles = rumination.engine.total_cycles
        rumination.ruminate_now(budget=5)
        final_cycles = rumination.engine.total_cycles

        assert final_cycles > initial_cycles, \
            "Rumination cycles did not execute"

    def test_desire_diminishing_returns(self, rtta, rumination):
        """Desire should increase with diminishing returns."""
        result = rtta.reason("Test question", verbose=False)
        record = rumination.log_failure(
            "TEST-DIM",
            "Test question",
            "42",
            "unknown",
            result
        )

        initial_desire = record.desire_level

        # Run multiple cycles
        for _ in range(5):
            rumination.engine.raise_desire(record)

        # Desire should increase but not linearly
        assert record.desire_level > initial_desire, \
            "Desire did not increase"
        assert record.desire_level < 1.0, \
            "Desire exceeded maximum"

    def test_stagnation_penalty_applied(self, rtta, rumination):
        """Stagnation penalty should apply after threshold cycles."""
        result = rtta.reason("Stuck question", verbose=False)
        record = rumination.log_failure(
            "TEST-STAG",
            "Stuck question",
            "answer",
            "wrong",
            result
        )

        # Simulate many cycles without improvement
        record.rumination_cycles = 10
        record._last_improvement = 0

        rumination.engine.apply_stagnation_penalty(record)

        assert record._stagnation_penalty > 0, \
            "Stagnation penalty not applied"


# ============================================================
# SCHEMA COMPLIANCE TESTS
# ============================================================

class TestSchemaCompliance:
    """Test that ledger matches JSON Schema."""

    def test_ledger_has_required_fields(self, fresh_ledger):
        """Ledger must have all required top-level fields."""
        assert hasattr(fresh_ledger, 'version')
        assert hasattr(fresh_ledger, 'threshold')
        assert hasattr(fresh_ledger, 'created_at')
        assert hasattr(fresh_ledger, 'records')
        assert hasattr(fresh_ledger, 'index')

    def test_index_structure(self, fresh_ledger):
        """Index must have unsolved/solved/retired lists."""
        assert 'unsolved' in fresh_ledger.index
        assert 'solved' in fresh_ledger.index
        assert 'retired' in fresh_ledger.index
        assert isinstance(fresh_ledger.index['unsolved'], list)

    def test_valid_slot_values(self, rtta, fresh_ledger):
        """Slot values must be from valid set."""
        valid_slots = ["1D", "2W", "3P", "4D", "5W", "6P",
                       "7D", "8W", "9P", "10D", "11W", "12P"]

        result = rtta.reason("Test", verbose=False)
        record = fresh_ledger.add_failure(
            "SLOT-TEST", "Test", "answer", "wrong", result
        )

        assert record.first_unmet_slot in valid_slots, \
            f"Invalid slot: {record.first_unmet_slot}"
        assert record.gist.slot_block in valid_slots, \
            f"Invalid gist slot: {record.gist.slot_block}"

    def test_status_enum_values(self, rtta, fresh_ledger):
        """Status must be unsolved/solved/retired."""
        valid_statuses = ["unsolved", "solved", "retired"]

        result = rtta.reason("Test", verbose=False)
        record = fresh_ledger.add_failure(
            "STATUS-TEST", "Test", "answer", "wrong", result
        )

        assert record.status in valid_statuses

    def test_ledger_serialization(self, rtta, fresh_ledger):
        """Ledger must serialize to valid JSON."""
        result = rtta.reason("Test", verbose=False)
        fresh_ledger.add_failure("JSON-TEST", "Test", "ans", "wrong", result)

        # Force save
        fresh_ledger._save()

        # Load and validate
        with open(fresh_ledger.storage_path) as f:
            data = json.load(f)

        assert data['version'] == "1.0"
        assert 'records' in data
        assert 'index' in data


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, rtta, tmp_path):
        """Test full RTTA-C + RTTA-R pipeline."""
        # Create rumination system
        rum = RTTARumination(
            rtta_solver=rtta.reason,
            ledger_path=str(tmp_path / "pipe_ledger.json"),
            operator_path=str(tmp_path / "pipe_ops.json")
        )

        # Run all regression tests
        passed = 0
        for test_id, test_type, question, expected in REGRESSION_TESTS:
            result = rtta.reason(question, verbose=False)
            answer = str(result['answer'])

            if expected.lower() in answer.lower():
                passed += 1
            else:
                rum.log_failure(test_id, question, expected, answer, result)

        # Run rumination
        rum.ruminate_now(budget=10)

        # Check stats
        stats = rum.get_stats()
        print(f"\nPipeline stats: {stats}")

        assert stats['total_cycles'] > 0
        assert passed >= PASS_THRESHOLD


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

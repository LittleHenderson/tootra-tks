#!/usr/bin/env python3
"""
Test Integrated Runner - Verifies all systems work together

Tests:
1. EpisodeRunner initializes with all systems
2. RTTA-C, RTTA-R, Lacunary, Coherence all active
3. Episode runs with coherence gating
4. Output includes all metrics
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tks.config import TKSConfig, OperationalMode
from src.tks.runner import (
    EpisodeRunner,
    EpisodeInput,
    RUMINATION_AVAILABLE,
    LACUNARY_AVAILABLE,
    COHERENCE_AVAILABLE,
)


def test_system_availability():
    """Test all systems are available."""
    print("=" * 60)
    print("SYSTEM AVAILABILITY CHECK")
    print("=" * 60)

    systems = {
        "RTTA-R Rumination": RUMINATION_AVAILABLE,
        "Lacunary Pattern Flow": LACUNARY_AVAILABLE,
        "Coherence Gating": COHERENCE_AVAILABLE,
    }

    all_available = True
    for name, available in systems.items():
        status = "OK" if available else "MISSING"
        print(f"  {name}: {status}")
        if not available:
            all_available = False

    print("-" * 60)
    if all_available:
        print("ALL SYSTEMS AVAILABLE")
    else:
        print("WARNING: Some systems missing")

    return all_available


def test_runner_initialization():
    """Test runner initializes with all systems."""
    print("\n" + "=" * 60)
    print("RUNNER INITIALIZATION TEST")
    print("=" * 60)

    # Create minimal config
    config = TKSConfig()

    # Create runner with all systems enabled
    runner = EpisodeRunner(
        config=config,
        enable_rumination=True,
        enable_lacunary=True,
        enable_coherence=True,
    )

    # Check each system
    systems = {
        "RTTA-C": runner.rtta is not None,
        "RTTA-R Rumination": runner.rumination is not None,
        "Lacunary Flow": runner.lacunary is not None,
        "Coherence System": runner.coherence_system is not None,
        "DPS Gating": runner.dps is not None,
        "Governance Rails": runner.governance is not None,
    }

    for name, initialized in systems.items():
        status = "INIT" if initialized else "SKIP"
        print(f"  {name}: {status}")

    return runner


def test_episode_with_coherence(runner: EpisodeRunner):
    """Test episode runs with coherence metrics."""
    print("\n" + "=" * 60)
    print("EPISODE WITH COHERENCE TEST")
    print("=" * 60)

    # Create test input
    input = EpisodeInput(
        goal="What is 2 + 2?",
        context={"uncertainty": 0.1, "stakes": 0.1, "alignment": 0.95},
    )

    # Run episode
    result = runner.run_episode(input)

    print(f"  Episode ID: {result.episode_id}")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration_ms:.1f}ms")
    print(f"  Mode: {result.mode.value}")
    print(f"  Violations: {result.violations}")

    # Coherence metrics
    print("\n  COHERENCE METRICS:")
    print(f"    Coherence Score: {result.coherence_score}")
    print(f"    Lacunarity: {result.lacunarity}")
    print(f"    NF Coords: {result.nf_coords}")
    print(f"    Pattern Category: {result.pattern_category}")
    print(f"    Coherence Gated: {result.coherence_gated}")

    # Check trace has coherence stage
    coherence_stage = None
    for stage in result.trace.get("stages", []):
        if stage.get("stage") == "COHERENCE_CHECK":
            coherence_stage = stage
            break

    if coherence_stage:
        print("\n  COHERENCE STAGE IN TRACE: YES")
    else:
        print("\n  COHERENCE STAGE IN TRACE: NO")

    return result


def test_rtta_integration(runner: EpisodeRunner):
    """Test RTTA-C works for reasoning."""
    print("\n" + "=" * 60)
    print("RTTA-C INTEGRATION TEST")
    print("=" * 60)

    if runner.rtta is None:
        print("  RTTA-C not available, skipping")
        return

    # Test questions
    questions = [
        ("2 + 2 = ?", "4"),
        ("If all cats are mammals and all mammals are animals, are cats animals?", "Yes"),
        ("What comes next: 2, 4, 6, 8, ?", "10"),
    ]

    for q, expected in questions:
        result = runner.rtta.reason(q, verbose=False)
        answer = str(result.get('answer', ''))
        match = expected.lower() in answer.lower()
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] Q: {q[:40]}...")
        print(f"         Expected: {expected}, Got: {answer[:30]}")


def test_rumination_stats(runner: EpisodeRunner):
    """Test rumination system stats."""
    print("\n" + "=" * 60)
    print("RUMINATION STATS TEST")
    print("=" * 60)

    stats = runner.get_rumination_stats()
    if stats is None:
        print("  Rumination not available")
        return

    print(f"  Total Failures: {stats.get('total_failures', 0)}")
    print(f"  Total Cycles: {stats.get('total_cycles', 0)}")
    print(f"  Unsolved: {stats.get('unsolved', 0)}")
    print(f"  Solved: {stats.get('solved', 0)}")


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("TKS INTEGRATED RUNNER TEST SUITE")
    print("Testing: RTTA-C + RTTA-R + Lacunary + Coherence")
    print("=" * 60 + "\n")

    # 1. Check availability
    all_available = test_system_availability()

    # 2. Initialize runner
    runner = test_runner_initialization()

    # 3. Run episode with coherence
    result = test_episode_with_coherence(runner)

    # 4. Test RTTA integration
    test_rtta_integration(runner)

    # 5. Check rumination stats
    test_rumination_stats(runner)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    checks = [
        ("All systems available", all_available),
        ("Runner initialized", runner is not None),
        ("Episode completed", result is not None and result.episode_id is not None),
        ("Coherence computed", result.coherence_score is not None),
        ("RTTA-C active", runner.rtta is not None),
    ]

    passed = sum(1 for _, v in checks if v)
    total = len(checks)

    for name, passed_check in checks:
        status = "PASS" if passed_check else "FAIL"
        print(f"  [{status}] {name}")

    print("-" * 60)
    print(f"RESULT: {passed}/{total} checks passed")

    if passed == total:
        print("\nALL SYSTEMS INTEGRATED AND WORKING!")
    else:
        print("\nSome systems need attention.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

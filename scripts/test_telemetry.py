#!/usr/bin/env python3
"""
Test script for TKS Telemetry/Monitoring system

This script demonstrates the full telemetry pipeline:
1. Generate sample augmentation metrics
2. Save to JSON and CSV
3. Generate plots from metrics
4. Verify all components work together

Author: TKS-LLM Training Integration Team
Date: 2025-12-14
Version: 1.0.0
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from augmentation_metrics import AugmentationLogger
    METRICS_AVAILABLE = True
except ImportError:
    print("Error: Could not import augmentation_metrics module")
    METRICS_AVAILABLE = False
    sys.exit(1)


def generate_sample_metrics():
    """Generate sample augmentation metrics for testing."""
    print("=" * 70)
    print("TELEMETRY TEST - Generating Sample Metrics")
    print("=" * 70)

    # Create logger
    logger = AugmentationLogger()
    logger.store_entries = True  # Store entries for inspection

    print("\nGenerating sample augmentation entries...")

    # Sample entries with different augmentation types
    sample_entries = [
        # Original entries
        {
            "id": "entry_001",
            "story": "A teacher helps a student grow",
            "expr": "B2 -> D5",
            "expr_elements": ["B2", "D5"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "source_id": "entry_001",
            "validator_pass": True,
        },
        {
            "id": "entry_002",
            "story": "A parent nurtures independence",
            "expr": "B3 -> D7",
            "expr_elements": ["B3", "D7"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "source_id": "entry_002",
            "validator_pass": True,
        },
        {
            "id": "entry_003",
            "story": "A mentor guides discovery",
            "expr": "B1 + C4 -> D6",
            "expr_elements": ["B1", "C4", "D6"],
            "expr_ops": ["+", "->"],
            "aug_type": "original",
            "source_id": "entry_003",
            "validator_pass": True,
        },

        # Inverted entries (W axis)
        {
            "id": "entry_001_inv_W",
            "story": "A student helps a teacher grow",
            "expr": "D2 -> B5",
            "expr_elements": ["D2", "B5"],
            "expr_ops": ["->"],
            "aug_type": "inversion",
            "source_id": "entry_001",
            "axes": ["W"],
            "mode": "soft",
            "validator_pass": True,
        },
        {
            "id": "entry_002_inv_W",
            "story": "Independence nurtures a parent",
            "expr": "D3 -> B7",
            "expr_elements": ["D3", "B7"],
            "expr_ops": ["->"],
            "aug_type": "inversion",
            "source_id": "entry_002",
            "axes": ["W"],
            "mode": "soft",
            "validator_pass": True,
        },

        # Inverted entries (N axis)
        {
            "id": "entry_001_inv_N",
            "story": "A teacher hinders a student's decline",
            "expr": "B8 -> D1",
            "expr_elements": ["B8", "D1"],
            "expr_ops": ["->"],
            "aug_type": "inversion",
            "source_id": "entry_001",
            "axes": ["N"],
            "mode": "soft",
            "validator_pass": True,
        },

        # Anti-attractor entries
        {
            "id": "entry_001_anti",
            "story": "A critic blocks a student's progress",
            "expr": "A9 -T D5",
            "expr_elements": ["A9", "D5"],
            "expr_ops": ["-T"],
            "aug_type": "anti_attractor",
            "source_id": "entry_001",
            "validator_pass": True,
        },
        {
            "id": "entry_002_anti",
            "story": "A controller prevents independence",
            "expr": "A8 -T D7",
            "expr_elements": ["A8", "D7"],
            "expr_ops": ["-T"],
            "aug_type": "anti_attractor",
            "source_id": "entry_002",
            "validator_pass": True,
        },

        # Some failed validation entries
        {
            "id": "entry_004_inv_W",
            "story": "[INVERSION FAILED]",
            "expr": None,
            "expr_elements": [],
            "expr_ops": [],
            "aug_type": "inversion",
            "source_id": "entry_004",
            "axes": ["W"],
            "mode": "soft",
            "validator_pass": False,
            "validation_error": "encoding_failed"
        },
    ]

    # Log all entries
    for entry in sample_entries:
        logger.log_entry(entry)

    print(f"Logged {len(sample_entries)} sample entries")

    return logger


def test_metrics_persistence(logger, output_dir):
    """Test saving metrics to JSON and CSV formats."""
    print("\n" + "=" * 70)
    print("TEST 1: Metrics Persistence")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Save to JSON (single object)
    print("\n[Test 1.1] Saving to JSON (single object)...")
    json_path = output_dir / "test_metrics.json"
    logger.save(str(json_path))
    assert json_path.exists(), "JSON file not created"
    print(f"  [PASS] Saved to {json_path}")

    # Verify JSON content
    with json_path.open("r") as f:
        data = json.load(f)
        assert "timestamp" in data, "Missing timestamp"
        assert "augmentation" in data, "Missing augmentation stats"
        assert "validation" in data, "Missing validation stats"
        print(f"  [PASS] JSON structure verified")

    # Test 2: Save to CSV
    print("\n[Test 1.2] Saving to CSV...")
    csv_path = output_dir / "test_metrics.csv"
    logger.save_to_csv(str(csv_path), append=False)
    assert csv_path.exists(), "CSV file not created"
    print(f"  [PASS] Saved to {csv_path}")

    # Verify CSV content
    with csv_path.open("r") as f:
        lines = f.readlines()
        assert len(lines) >= 2, "CSV should have header + data row"
        assert "timestamp" in lines[0], "Missing timestamp column"
        print(f"  [PASS] CSV structure verified ({len(lines)} lines)")

    # Test 3: Append to CSV
    print("\n[Test 1.3] Testing CSV append mode...")
    logger.save_to_csv(str(csv_path), append=True)
    with csv_path.open("r") as f:
        lines = f.readlines()
        assert len(lines) == 3, "CSV should have header + 2 data rows"
        print(f"  [PASS] CSV append verified ({len(lines)} lines)")

    # Test 4: Save to JSON array (trend tracking)
    print("\n[Test 1.4] Saving to JSON array (trend tracking)...")
    trend_path = output_dir / "test_trends_new.json"  # Use unique name to avoid conflicts
    # Remove if exists
    if trend_path.exists():
        trend_path.unlink()

    logger.save_to_json(str(trend_path), append=True)
    logger.save_to_json(str(trend_path), append=True)  # Append second time

    with trend_path.open("r") as f:
        data = json.load(f)
        assert isinstance(data, list), "Trend file should be JSON array"
        assert len(data) == 2, f"Should have 2 entries, got {len(data)}"
        print(f"  [PASS] Trend tracking verified ({len(data)} entries)")

    print("\n[PASS] All persistence tests passed!")
    return json_path, csv_path, trend_path


def test_multi_epoch_metrics(output_dir):
    """Test metrics tracking across multiple epochs."""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Epoch Metrics")
    print("=" * 70)

    output_dir = Path(output_dir)

    # Simulate 5 epochs of training
    csv_path = output_dir / "epoch_metrics_new.csv"  # Use unique name
    json_path = output_dir / "epoch_trends_new.json"  # Use unique name

    # Clean up existing files
    if csv_path.exists():
        csv_path.unlink()
    if json_path.exists():
        json_path.unlink()

    print("\nSimulating 5 epochs of training...")
    for epoch in range(1, 6):
        logger = AugmentationLogger()

        # Generate metrics for this epoch
        num_entries = 100 + epoch * 20  # Increasing dataset

        for i in range(num_entries):
            entry_type = ["original", "inversion", "anti_attractor"][i % 3]
            validator_pass = (i % 10) != 0  # 90% pass rate

            entry = {
                "id": f"epoch{epoch}_entry{i}",
                "expr_elements": ["B2", "D5"],
                "expr_ops": ["->"],
                "aug_type": entry_type,
                "validator_pass": validator_pass,
            }

            if entry_type == "inversion":
                entry["axes"] = ["W"]
                entry["mode"] = "soft"

            logger.log_entry(entry)

        # Save metrics
        logger.save_to_csv(str(csv_path), append=True)
        logger.save_to_json(str(json_path), append=True)

        summary = logger.get_summary()
        print(f"  Epoch {epoch}: {summary['augmentation']['total_count']} entries, "
              f"pass_rate={summary['validation']['pass_rate']:.2%}")

    # Verify multi-epoch files
    with csv_path.open("r") as f:
        lines = f.readlines()
        assert len(lines) == 6, f"CSV should have 6 lines (header + 5 epochs), got {len(lines)}"

    with json_path.open("r") as f:
        data = json.load(f)
        assert len(data) == 5, f"JSON should have 5 entries, got {len(data)}"

    print(f"\n[PASS] Multi-epoch metrics verified")
    return csv_path, json_path


def test_plotting_integration(metrics_files, output_dir):
    """Test plotting from metrics files."""
    print("\n" + "=" * 70)
    print("TEST 3: Plotting Integration")
    print("=" * 70)

    import subprocess

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Test plotting from JSON
    print("\n[Test 3.1] Plotting from JSON metrics...")
    json_file, csv_file, _ = metrics_files

    result = subprocess.run([
        sys.executable,
        "scripts/plot_metrics.py",
        "--input", str(json_file),
        "--output-dir", str(plots_dir / "json"),
        "--plot-type", "all"
    ], capture_output=True, text=True, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"  [FAIL] Plotting failed: {result.stderr}")
        return False

    # Check generated plots
    json_plots_dir = plots_dir / "json"
    expected_plots = [
        "loss_curve.png",
        "augmentation_distribution.png",
        "validation_rates.png",
        "world_noetic_distribution.png",
        "augmentation_ratios.png"
    ]

    for plot_name in expected_plots:
        plot_path = json_plots_dir / plot_name
        if not plot_path.exists():
            print(f"  [FAIL] Expected plot not generated: {plot_name}")
            return False

    print(f"  [PASS] All JSON plots generated ({len(expected_plots)} files)")

    # Test plotting from CSV
    print("\n[Test 3.2] Plotting from CSV metrics...")
    result = subprocess.run([
        sys.executable,
        "scripts/plot_metrics.py",
        "--input", str(csv_file),
        "--output-dir", str(plots_dir / "csv"),
        "--plot-type", "validation"
    ], capture_output=True, text=True, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"  [FAIL] CSV plotting failed: {result.stderr}")
        return False

    csv_plot = plots_dir / "csv" / "validation_rates.png"
    assert csv_plot.exists(), "CSV validation plot not generated"
    print(f"  [PASS] CSV plot generated")

    print("\n[PASS] All plotting tests passed!")
    return True


def test_integration_summary(logger):
    """Print integration summary and verify metrics."""
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)

    summary = logger.get_summary()

    print("\nAugmentation Statistics:")
    print(f"  Original:        {summary['augmentation']['original_count']:>6,}")
    print(f"  Inversions:      {summary['augmentation']['inversion_count']:>6,}")
    print(f"  Anti-attractors: {summary['augmentation']['anti_attractor_count']:>6,}")
    print(f"  Total:           {summary['augmentation']['total_count']:>6,}")
    print(f"  Aug ratio:       {summary['augmentation']['augmentation_ratio']:>8.2f}x")

    print("\nValidation Statistics:")
    print(f"  Total:     {summary['validation']['total']:>6,}")
    print(f"  Passed:    {summary['validation']['passed']:>6,}")
    print(f"  Failed:    {summary['validation']['failed']:>6,}")
    print(f"  Pass rate: {summary['validation']['pass_rate']:>7.2%}")

    print("\nDistribution Statistics:")
    dist = summary['distribution']
    if dist['world_counts']:
        print("  Worlds:", ", ".join(f"{w}={c}" for w, c in sorted(dist['world_counts'].items())))
    if dist['noetic_counts']:
        print("  Noetics:", ", ".join(f"{n}={c}" for n, c in sorted(dist['noetic_counts'].items())[:5]))

    # Verify expected values
    assert summary['augmentation']['total_count'] == 9, "Expected 9 total entries"
    assert summary['validation']['total'] == 9, "Expected 9 validated entries"
    assert summary['validation']['passed'] == 8, "Expected 8 passed validations"

    print("\n[PASS] Integration summary verified!")


def main():
    """Run all telemetry tests."""
    print("\n" + "=" * 80)
    print(" " * 20 + "TKS TELEMETRY INTEGRATION TEST")
    print("=" * 80)

    if not METRICS_AVAILABLE:
        print("\n[FAIL] augmentation_metrics module not available")
        return 1

    output_dir = Path("output/telemetry_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Test 1: Generate sample metrics
        logger = generate_sample_metrics()

        # Test 2: Persistence (JSON, CSV, trends)
        metrics_files = test_metrics_persistence(logger, output_dir)

        # Test 3: Multi-epoch tracking
        epoch_files = test_multi_epoch_metrics(output_dir)

        # Test 4: Plotting integration
        test_plotting_integration(metrics_files, output_dir)

        # Test 5: Integration summary
        test_integration_summary(logger)

        # Final summary
        print("\n" + "=" * 80)
        print(" " * 25 + "ALL TESTS PASSED!")
        print("=" * 80)
        print(f"\nTest outputs saved to: {output_dir.absolute()}")
        print("\nGenerated files:")
        print("  - test_metrics.json (single-run metrics)")
        print("  - test_metrics.csv (CSV format for plotting)")
        print("  - test_trends.json (multi-run trend tracking)")
        print("  - epoch_metrics.csv (multi-epoch CSV)")
        print("  - epoch_trends.json (multi-epoch JSON array)")
        print("  - plots/json/ (plots from JSON metrics)")
        print("  - plots/csv/ (plots from CSV metrics)")

        print("\nUsage Examples:")
        print("  # Generate plots from JSON")
        print(f"  python scripts/plot_metrics.py --input {output_dir}/test_metrics.json --output-dir {output_dir}/plots --plot-type all")
        print("\n  # Generate plots from CSV")
        print(f"  python scripts/plot_metrics.py --input {output_dir}/epoch_metrics.csv --output-dir {output_dir}/plots --plot-type validation")

        print("\n" + "=" * 80)

        return 0

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

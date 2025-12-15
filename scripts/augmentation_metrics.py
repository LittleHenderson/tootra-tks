"""
TKS Augmentation Metrics Logger

Lightweight logging module for tracking augmentation and canonical violations
during training/eval. No heavy dependencies (pandas/wandb) required.

Usage:
    logger = AugmentationLogger()
    logger.log_entry(entry)
    logger.log_batch(entries)
    summary = logger.get_summary()
    logger.save("metrics.json")

Features:
    - Track augmentation counts by type (original/inversion/anti_attractor)
    - Monitor validation pass/fail rates
    - Record distribution of operators, noetics, worlds
    - In-memory accumulation with JSON export
    - Print formatted summaries

Author: TKS-LLM Training Integration Team
Date: 2025-12-14
Version: 1.0.0
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ValidationStats:
    """Validation statistics for canonical compliance."""
    total: int = 0
    passed: int = 0
    failed: int = 0

    # Component validity
    world_valid: int = 0
    noetic_valid: int = 0
    operator_valid: int = 0
    structural_valid: int = 0
    foundation_valid: int = 0

    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=dict)

    def pass_rate(self) -> float:
        """Calculate overall pass rate."""
        return self.passed / self.total if self.total > 0 else 0.0

    def world_validity_rate(self) -> float:
        """Calculate world validity rate."""
        return self.world_valid / self.total if self.total > 0 else 0.0

    def noetic_validity_rate(self) -> float:
        """Calculate noetic validity rate."""
        return self.noetic_valid / self.total if self.total > 0 else 0.0

    def operator_validity_rate(self) -> float:
        """Calculate operator validity rate."""
        return self.operator_valid / self.total if self.total > 0 else 0.0

    def structural_validity_rate(self) -> float:
        """Calculate structural validity rate."""
        return self.structural_valid / self.total if self.total > 0 else 0.0

    def foundation_validity_rate(self) -> float:
        """Calculate foundation validity rate."""
        return self.foundation_valid / self.total if self.total > 0 else 0.0


@dataclass
class DistributionStats:
    """Distribution statistics for TKS elements."""
    # Element distributions
    world_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    noetic_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    operator_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    foundation_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Polarity tracking
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

    def world_distribution(self) -> Dict[str, float]:
        """Calculate world distribution as percentages."""
        total = sum(self.world_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.world_counts.items()}

    def noetic_distribution(self) -> Dict[int, float]:
        """Calculate noetic distribution as percentages."""
        total = sum(self.noetic_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.noetic_counts.items()}

    def operator_distribution(self) -> Dict[str, float]:
        """Calculate operator distribution as percentages."""
        total = sum(self.operator_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.operator_counts.items()}

    def polarity_distribution(self) -> Dict[str, float]:
        """Calculate polarity distribution as percentages."""
        total = self.positive_count + self.negative_count + self.neutral_count
        if total == 0:
            return {}
        return {
            "positive": self.positive_count / total,
            "negative": self.negative_count / total,
            "neutral": self.neutral_count / total
        }


@dataclass
class AugmentationStats:
    """Augmentation statistics."""
    original_count: int = 0
    inversion_count: int = 0
    anti_attractor_count: int = 0

    # Axis usage tracking
    axes_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Mode tracking
    mode_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def total_count(self) -> int:
        """Total number of entries."""
        return self.original_count + self.inversion_count + self.anti_attractor_count

    def augmentation_ratio(self) -> float:
        """Ratio of augmented to original."""
        if self.original_count == 0:
            return 0.0
        return (self.inversion_count + self.anti_attractor_count) / self.original_count

    def inversion_ratio(self) -> float:
        """Ratio of inversions to original."""
        if self.original_count == 0:
            return 0.0
        return self.inversion_count / self.original_count

    def anti_attractor_ratio(self) -> float:
        """Ratio of anti-attractors to original."""
        if self.original_count == 0:
            return 0.0
        return self.anti_attractor_count / self.original_count


# ==============================================================================
# AUGMENTATION LOGGER CLASS
# ==============================================================================

class AugmentationLogger:
    """
    Lightweight logger for augmentation and validation metrics.

    Usage:
        logger = AugmentationLogger()

        # Log individual entries
        logger.log_entry(entry)

        # Log batch of entries
        logger.log_batch(entries)

        # Get summary statistics
        summary = logger.get_summary()

        # Print formatted summary
        logger.print_summary()

        # Save to JSON
        logger.save("metrics.json")

        # Reset metrics
        logger.reset()
    """

    def __init__(self):
        """Initialize logger with empty metrics."""
        self.validation = ValidationStats()
        self.distribution = DistributionStats()
        self.augmentation = AugmentationStats()

        self.start_time = datetime.now()
        self.end_time = None

        # Raw entry storage (optional, for debugging)
        self.entries: List[Dict[str, Any]] = []
        self.store_entries = False  # Set to True to store all entries

    def log_entry(self, entry: Dict[str, Any]) -> None:
        """
        Log a single entry.

        Args:
            entry: Dictionary containing entry data with fields:
                - aug_type: "original", "inversion", or "anti_attractor"
                - validator_pass: boolean
                - expr_elements: list of element codes (e.g., ["B2", "D5"])
                - expr_ops: list of operators
                - axes: list of axes (for inversions)
                - mode: inversion mode (for inversions)
        """
        if self.store_entries:
            self.entries.append(entry)

        # Track augmentation type
        aug_type = entry.get("aug_type", "original")
        if aug_type == "original":
            self.augmentation.original_count += 1
        elif aug_type == "inversion":
            self.augmentation.inversion_count += 1

            # Track axes usage
            axes = entry.get("axes", [])
            for axis in axes:
                self.augmentation.axes_usage[axis] += 1

            # Track mode usage
            mode = entry.get("mode", "unknown")
            self.augmentation.mode_counts[mode] += 1

        elif aug_type == "anti_attractor":
            self.augmentation.anti_attractor_count += 1

        # Track validation
        self.validation.total += 1
        validator_pass = entry.get("validator_pass", False)

        if validator_pass:
            self.validation.passed += 1
        else:
            self.validation.failed += 1

            # Track error if present
            error = entry.get("validation_error", "unknown")
            self.validation.error_counts[error] = self.validation.error_counts.get(error, 0) + 1

        # Track component validity
        expr_elements = entry.get("expr_elements", [])
        expr_ops = entry.get("expr_ops", [])

        if expr_elements:
            # World validity
            try:
                worlds_valid = all(elem[0] in "ABCD" for elem in expr_elements if len(elem) > 0)
                if worlds_valid:
                    self.validation.world_valid += 1
            except (IndexError, TypeError):
                pass

            # Noetic validity
            try:
                noetics_valid = all(
                    1 <= int(elem[1:]) <= 10
                    for elem in expr_elements
                    if len(elem) > 1
                )
                if noetics_valid:
                    self.validation.noetic_valid += 1
            except (ValueError, IndexError, TypeError):
                pass

            # Track distributions
            for elem in expr_elements:
                if len(elem) >= 2:
                    world = elem[0]
                    self.distribution.world_counts[world] += 1

                    try:
                        noetic = int(elem[1:].split('.')[0])  # Handle sense notation
                        self.distribution.noetic_counts[noetic] += 1
                    except ValueError:
                        pass

        # Operator validity and distribution
        if expr_ops:
            # Define allowed operators
            ALLOWED_OPS = {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'}

            try:
                ops_valid = all(op in ALLOWED_OPS for op in expr_ops)
                if ops_valid:
                    self.validation.operator_valid += 1
            except TypeError:
                pass

            for op in expr_ops:
                self.distribution.operator_counts[op] += 1

        # Structural validity
        if expr_elements and expr_ops:
            if len(expr_ops) == len(expr_elements) - 1:
                self.validation.structural_valid += 1

        # Foundation tracking (if present)
        foundations = entry.get("foundations", [])
        if foundations:
            self.validation.foundation_valid += 1
            for foundation in foundations:
                if isinstance(foundation, tuple):
                    fid, _ = foundation
                else:
                    fid = foundation
                self.distribution.foundation_counts[fid] += 1

    def log_batch(self, entries: List[Dict[str, Any]]) -> None:
        """
        Log a batch of entries.

        Args:
            entries: List of entry dictionaries
        """
        for entry in entries:
            self.log_entry(entry)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary containing all metrics in a structured format
        """
        # Update end time
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        return {
            "timestamp": self.end_time.isoformat(),
            "duration_seconds": duration,

            "augmentation": {
                "original_count": self.augmentation.original_count,
                "inversion_count": self.augmentation.inversion_count,
                "anti_attractor_count": self.augmentation.anti_attractor_count,
                "total_count": self.augmentation.total_count(),
                "augmentation_ratio": self.augmentation.augmentation_ratio(),
                "inversion_ratio": self.augmentation.inversion_ratio(),
                "anti_attractor_ratio": self.augmentation.anti_attractor_ratio(),
                "axes_usage": dict(self.augmentation.axes_usage),
                "mode_counts": dict(self.augmentation.mode_counts),
            },

            "validation": {
                "total": self.validation.total,
                "passed": self.validation.passed,
                "failed": self.validation.failed,
                "pass_rate": self.validation.pass_rate(),
                "world_validity_rate": self.validation.world_validity_rate(),
                "noetic_validity_rate": self.validation.noetic_validity_rate(),
                "operator_validity_rate": self.validation.operator_validity_rate(),
                "structural_validity_rate": self.validation.structural_validity_rate(),
                "foundation_validity_rate": self.validation.foundation_validity_rate(),
                "error_counts": dict(self.validation.error_counts),
            },

            "distribution": {
                "world_counts": dict(self.distribution.world_counts),
                "world_distribution": self.distribution.world_distribution(),
                "noetic_counts": {str(k): v for k, v in self.distribution.noetic_counts.items()},
                "noetic_distribution": {str(k): v for k, v in self.distribution.noetic_distribution().items()},
                "operator_counts": dict(self.distribution.operator_counts),
                "operator_distribution": self.distribution.operator_distribution(),
                "foundation_counts": {str(k): v for k, v in self.distribution.foundation_counts.items()},
                "polarity": {
                    "positive": self.distribution.positive_count,
                    "negative": self.distribution.negative_count,
                    "neutral": self.distribution.neutral_count,
                    "distribution": self.distribution.polarity_distribution(),
                },
            },
        }

    def print_summary(self, detailed: bool = False) -> None:
        """
        Print formatted summary to console.

        Args:
            detailed: If True, print detailed distributions
        """
        summary = self.get_summary()

        print("=" * 70)
        print("TKS AUGMENTATION METRICS SUMMARY")
        print("=" * 70)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print()

        # Augmentation statistics
        aug = summary["augmentation"]
        print("AUGMENTATION STATISTICS")
        print("-" * 70)
        print(f"  Original entries:       {aug['original_count']:>8,}")
        print(f"  Inverted entries:       {aug['inversion_count']:>8,}")
        print(f"  Anti-attractor entries: {aug['anti_attractor_count']:>8,}")
        print(f"  Total entries:          {aug['total_count']:>8,}")
        print()
        print(f"  Augmentation ratio:     {aug['augmentation_ratio']:>8.2f}x")
        print(f"  Inversion ratio:        {aug['inversion_ratio']:>8.2f}x")
        print(f"  Anti-attractor ratio:   {aug['anti_attractor_ratio']:>8.2f}x")

        if aug['axes_usage']:
            print()
            print("  Axes usage:")
            for axis, count in sorted(aug['axes_usage'].items()):
                print(f"    {axis}: {count:>6,}")

        if aug['mode_counts']:
            print()
            print("  Mode usage:")
            for mode, count in sorted(aug['mode_counts'].items()):
                print(f"    {mode}: {count:>6,}")

        print()

        # Validation statistics
        val = summary["validation"]
        print("VALIDATION STATISTICS")
        print("-" * 70)
        print(f"  Total validated:        {val['total']:>8,}")
        print(f"  Passed:                 {val['passed']:>8,}")
        print(f"  Failed:                 {val['failed']:>8,}")
        print(f"  Pass rate:              {val['pass_rate']:>8.2%}")
        print()
        print(f"  World validity:         {val['world_validity_rate']:>8.2%}")
        print(f"  Noetic validity:        {val['noetic_validity_rate']:>8.2%}")
        print(f"  Operator validity:      {val['operator_validity_rate']:>8.2%}")
        print(f"  Structural validity:    {val['structural_validity_rate']:>8.2%}")
        print(f"  Foundation validity:    {val['foundation_validity_rate']:>8.2%}")

        if val['error_counts'] and detailed:
            print()
            print("  Error counts:")
            for error, count in sorted(val['error_counts'].items(), key=lambda x: -x[1])[:10]:
                print(f"    {error[:50]}: {count}")

        print()

        # Distribution statistics
        if detailed:
            dist = summary["distribution"]

            print("DISTRIBUTION STATISTICS")
            print("-" * 70)

            if dist['world_distribution']:
                print("  World distribution:")
                for world, pct in sorted(dist['world_distribution'].items()):
                    count = dist['world_counts'][world]
                    print(f"    {world}: {count:>6,} ({pct:>6.2%})")
                print()

            if dist['noetic_distribution']:
                print("  Noetic distribution:")
                for noetic, pct in sorted(dist['noetic_distribution'].items(), key=lambda x: int(x[0])):
                    count = dist['noetic_counts'][noetic]
                    print(f"    {noetic:>2}: {count:>6,} ({pct:>6.2%})")
                print()

            if dist['operator_distribution']:
                print("  Operator distribution:")
                for op, pct in sorted(dist['operator_distribution'].items(), key=lambda x: -x[1]):
                    count = dist['operator_counts'][op]
                    print(f"    {op:>3}: {count:>6,} ({pct:>6.2%})")
                print()

        print("=" * 70)

    def save(self, filepath: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        summary = self.get_summary()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def save_to_json(self, filepath: str, append: bool = False) -> None:
        """
        Save metrics to JSON file with optional append mode for trend tracking.

        Args:
            filepath: Path to output JSON file
            append: If True, append to existing file as JSON array

        Note:
            When append=True, the file will be a JSON array where each entry
            is a complete metrics summary with timestamps for trend analysis.
        """
        summary = self.get_summary()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if append and filepath.exists():
            # Load existing data
            try:
                with filepath.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except (json.JSONDecodeError, ValueError):
                existing_data = []

            # Append new summary
            existing_data.append(summary)

            # Save back
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
        else:
            # Save as single object or start new array
            data = [summary] if append else summary
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def save_to_csv(self, filepath: str, append: bool = True) -> None:
        """
        Save metrics to CSV file with optional append mode for trend tracking.

        Args:
            filepath: Path to output CSV file
            append: If True, append to existing file (default: True)

        CSV Format:
            timestamp, duration_seconds,
            original_count, inversion_count, anti_attractor_count, total_count,
            augmentation_ratio, inversion_ratio, anti_attractor_ratio,
            validation_total, validation_passed, validation_failed, pass_rate,
            world_validity_rate, noetic_validity_rate, operator_validity_rate,
            structural_validity_rate, foundation_validity_rate
        """
        import csv

        summary = self.get_summary()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV headers
        headers = [
            "timestamp", "duration_seconds",
            # Augmentation metrics
            "original_count", "inversion_count", "anti_attractor_count", "total_count",
            "augmentation_ratio", "inversion_ratio", "anti_attractor_ratio",
            # Validation metrics
            "validation_total", "validation_passed", "validation_failed", "pass_rate",
            "world_validity_rate", "noetic_validity_rate", "operator_validity_rate",
            "structural_validity_rate", "foundation_validity_rate"
        ]

        # Extract values
        aug = summary["augmentation"]
        val = summary["validation"]

        row = [
            summary["timestamp"],
            summary["duration_seconds"],
            # Augmentation
            aug["original_count"],
            aug["inversion_count"],
            aug["anti_attractor_count"],
            aug["total_count"],
            aug["augmentation_ratio"],
            aug["inversion_ratio"],
            aug["anti_attractor_ratio"],
            # Validation
            val["total"],
            val["passed"],
            val["failed"],
            val["pass_rate"],
            val["world_validity_rate"],
            val["noetic_validity_rate"],
            val["operator_validity_rate"],
            val["structural_validity_rate"],
            val["foundation_validity_rate"]
        ]

        # Check if file exists and has content
        file_exists = filepath.exists() and filepath.stat().st_size > 0

        # Write CSV
        with filepath.open("a" if (append and file_exists) else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header if new file or not appending
            if not (append and file_exists):
                writer.writerow(headers)

            # Write data row
            writer.writerow(row)

    def reset(self) -> None:
        """Reset all metrics."""
        self.validation = ValidationStats()
        self.distribution = DistributionStats()
        self.augmentation = AugmentationStats()
        self.start_time = datetime.now()
        self.end_time = None
        self.entries = []


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_batch_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics for a batch of entries.

    Args:
        entries: List of entry dictionaries

    Returns:
        Dictionary with batch statistics
    """
    logger = AugmentationLogger()
    logger.log_batch(entries)
    return logger.get_summary()


def track_epoch_stats(
    epoch: int,
    entries: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Track and optionally save epoch statistics.

    Args:
        epoch: Epoch number
        entries: List of entries from this epoch
        output_dir: Optional directory to save epoch metrics

    Returns:
        Dictionary with epoch statistics
    """
    logger = AugmentationLogger()
    logger.log_batch(entries)
    summary = logger.get_summary()

    summary["epoch"] = epoch

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"epoch_{epoch:03d}_metrics.json"
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def compare_metrics(
    baseline_summary: Dict[str, Any],
    augmented_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare baseline and augmented metrics.

    Args:
        baseline_summary: Summary from baseline logger
        augmented_summary: Summary from augmented logger

    Returns:
        Dictionary with comparison metrics
    """
    baseline_val = baseline_summary["validation"]
    augmented_val = augmented_summary["validation"]

    return {
        "pass_rate_improvement": augmented_val["pass_rate"] - baseline_val["pass_rate"],
        "world_validity_improvement": augmented_val["world_validity_rate"] - baseline_val["world_validity_rate"],
        "noetic_validity_improvement": augmented_val["noetic_validity_rate"] - baseline_val["noetic_validity_rate"],
        "operator_validity_improvement": augmented_val["operator_validity_rate"] - baseline_val["operator_validity_rate"],
        "structural_validity_improvement": augmented_val["structural_validity_rate"] - baseline_val["structural_validity_rate"],
        "augmentation_ratio": augmented_summary["augmentation"]["augmentation_ratio"],
    }

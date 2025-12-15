#!/usr/bin/env python3
"""
TKS Metrics Plotting Script

Generates visualization plots from augmentation/training metrics files.
Supports both JSON and CSV input formats with minimal dependencies.

Features:
    - Loss curve plotting (training progress over epochs)
    - Augmentation distribution (original/inversion/anti-attractor pie chart)
    - Augmentation counts bar chart (grouped bars over epochs)
    - Validation rate trends over time (pass rate, world/noetic/operator validity)
    - World and noetic distribution analysis (bar charts)
    - Augmentation ratios over time (line chart)
    - Combined dashboard (2x2 grid with loss, validation, augmentation, summary)

Usage:
    # Generate all plots from JSON metrics
    python scripts/plot_metrics.py --input output/metrics.json --output-dir output/plots --plot-type all

    # Generate specific plot from CSV metrics
    python scripts/plot_metrics.py --input output/metrics.csv --output-dir output/plots --plot-type loss

    # Generate distribution plots only
    python scripts/plot_metrics.py --input output/metrics.json --output-dir output/plots --plot-type distribution

    # Generate combined dashboard (training + augmentation overview)
    python scripts/plot_metrics.py --input output/training_metrics.json --output-dir output/plots --plot-type dashboard

    # Generate augmentation counts bar chart
    python scripts/plot_metrics.py --input output/metrics.json --output-dir output/plots --plot-type counts-bar

Plot Types:
    - loss: Training loss curve over epochs
    - distribution: Augmentation type distribution (pie chart)
    - validation: Validation pass rates over time (multi-line)
    - world-noetic: World (A/B/C/D) and noetic (1-10) distribution (bar charts)
    - ratios: Augmentation ratios over time (line chart)
    - counts-bar: Augmentation counts by type over epochs (grouped bar chart)
    - dashboard: Combined 2x2 dashboard with loss, validation, augmentation, summary
    - all: Generate all available plots including dashboard

Dependencies:
    - matplotlib (only required dependency)

Author: TKS-LLM Training Integration Team
Date: 2025-12-14
Version: 1.1.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Error: matplotlib is required but not installed.")
    print("Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False
    sys.exit(1)


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_metrics_json(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load metrics from JSON file.

    Args:
        filepath: Path to JSON metrics file

    Returns:
        List of metric dictionaries (one per epoch/run)

    Supports two formats:
        1. Single object: {"timestamp": ..., "augmentation": {...}, ...}
        2. Array of objects: [{...}, {...}, ...]
    """
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize to list
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected JSON format: {type(data)}")


def load_metrics_csv(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load metrics from CSV file.

    Args:
        filepath: Path to CSV metrics file

    Returns:
        List of metric dictionaries (one per row)

    Expected CSV format:
        timestamp, duration_seconds, original_count, inversion_count, ...
    """
    import csv

    metrics = []
    with filepath.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            metric = {}
            for key, value in row.items():
                # Try to convert to float/int
                try:
                    if "." in value:
                        metric[key] = float(value)
                    else:
                        metric[key] = int(value)
                except (ValueError, AttributeError):
                    metric[key] = value

            metrics.append(metric)

    return metrics


def load_metrics(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load metrics from JSON or CSV file (auto-detect format).

    Args:
        filepath: Path to metrics file

    Returns:
        List of metric dictionaries
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".json":
        return load_metrics_json(filepath)
    elif suffix == ".csv":
        return load_metrics_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix} (expected .json or .csv)")


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_loss_curve(metrics: List[Dict[str, Any]], output_path: Path, title: str = "Training Loss Over Time") -> None:
    """
    Plot training loss over time/epochs.

    Args:
        metrics: List of metric dictionaries with 'loss' or epoch loss data
        output_path: Path to save PNG output
        title: Plot title

    Expected metric format:
        - For training metrics: {"epoch": 1, "loss": 0.5, ...}
        - For CSV: {"timestamp": "...", "validation_total": N, ...}
          (uses pass_rate as proxy if no loss available)
        - For training summary: {"loss": {"epoch_losses": [[1, 0.5], [2, 0.4]]}}

    Note:
        If 'loss' field is not available, will use validation pass rate as proxy.
    """
    # Extract loss data
    epochs = []
    losses = []

    for i, metric in enumerate(metrics):
        # Try different loss field names
        loss = None

        # Check for training summary format with epoch_losses list
        if "loss" in metric and isinstance(metric["loss"], dict):
            if "epoch_losses" in metric["loss"]:
                # Format: {"loss": {"epoch_losses": [[epoch, loss], ...]}}
                for epoch_loss_pair in metric["loss"]["epoch_losses"]:
                    epochs.append(epoch_loss_pair[0])
                    losses.append(epoch_loss_pair[1])
                continue  # Skip to next metric

        # Check for explicit loss field
        if "loss" in metric and isinstance(metric["loss"], (int, float)):
            loss = metric["loss"]
        elif "avg_loss" in metric:
            loss = metric["avg_loss"]
        elif "epoch_loss" in metric:
            loss = metric["epoch_loss"]
        # Use validation pass rate as proxy (invert to make it loss-like)
        elif "pass_rate" in metric:
            loss = 1.0 - metric["pass_rate"]
        elif "validation" in metric and "pass_rate" in metric["validation"]:
            loss = 1.0 - metric["validation"]["pass_rate"]

        if loss is not None:
            # Determine epoch number
            if "epoch" in metric:
                epoch = metric["epoch"]
            else:
                epoch = i + 1

            epochs.append(epoch)
            losses.append(loss)

    if not losses:
        print(f"Warning: No loss data found in metrics, skipping loss plot")
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved loss curve to: {output_path}")


def plot_augmentation_distribution(metrics: List[Dict[str, Any]], output_path: Path, title: str = "Augmentation Type Distribution") -> None:
    """
    Plot pie/bar chart of augmentation type distribution.

    Args:
        metrics: List of metric dictionaries with augmentation counts
        output_path: Path to save PNG output
        title: Plot title

    Shows distribution of:
        - Original entries
        - Inversion entries
        - Anti-attractor entries
    """
    # Aggregate counts across all metrics (use last/latest if multiple)
    metric = metrics[-1] if metrics else {}

    # Extract counts
    if "augmentation" in metric:
        aug = metric["augmentation"]
        original_count = aug.get("original_count", 0)
        inversion_count = aug.get("inversion_count", 0)
        anti_attractor_count = aug.get("anti_attractor_count", 0)
    else:
        # CSV format
        original_count = metric.get("original_count", 0)
        inversion_count = metric.get("inversion_count", 0)
        anti_attractor_count = metric.get("anti_attractor_count", 0)

    # Prepare data
    labels = ['Original', 'Inversion', 'Anti-Attractor']
    sizes = [original_count, inversion_count, anti_attractor_count]
    colors = ['#A23B72', '#F18F01', '#C73E1D']
    explode = (0.05, 0.05, 0.05)  # Slight separation

    # Filter out zero values
    filtered_data = [(label, size, color, exp) for label, size, color, exp in zip(labels, sizes, colors, explode) if size > 0]

    if not filtered_data:
        print(f"Warning: No augmentation data found, skipping distribution plot")
        return

    labels, sizes, colors, explode = zip(*filtered_data)

    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 11})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')  # Equal aspect ratio ensures circular pie

    # Add legend with counts
    legend_labels = [f"{label}: {size:,}" for label, size in zip(labels, sizes)]
    plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved augmentation distribution to: {output_path}")


def plot_validation_rates(metrics: List[Dict[str, Any]], output_path: Path, title: str = "Validation Pass Rates Over Time") -> None:
    """
    Plot validator pass rates over epochs/time.

    Args:
        metrics: List of metric dictionaries with validation data
        output_path: Path to save PNG output
        title: Plot title

    Plots multiple validation metrics:
        - Overall pass rate
        - World validity rate
        - Noetic validity rate
        - Operator validity rate
    """
    # Extract validation data
    epochs = []
    pass_rates = []
    world_rates = []
    noetic_rates = []
    operator_rates = []

    for i, metric in enumerate(metrics):
        # Determine epoch number
        if "epoch" in metric:
            epoch = metric["epoch"]
        else:
            epoch = i + 1

        # Extract validation rates
        if "validation" in metric:
            val = metric["validation"]
            pass_rate = val.get("pass_rate", 0)
            world_rate = val.get("world_validity_rate", 0)
            noetic_rate = val.get("noetic_validity_rate", 0)
            operator_rate = val.get("operator_validity_rate", 0)
        else:
            # CSV format
            pass_rate = metric.get("pass_rate", 0)
            world_rate = metric.get("world_validity_rate", 0)
            noetic_rate = metric.get("noetic_validity_rate", 0)
            operator_rate = metric.get("operator_validity_rate", 0)

        epochs.append(epoch)
        pass_rates.append(pass_rate)
        world_rates.append(world_rate)
        noetic_rates.append(noetic_rate)
        operator_rates.append(operator_rate)

    if not pass_rates:
        print(f"Warning: No validation data found, skipping validation plot")
        return

    # Create plot
    plt.figure(figsize=(12, 7))

    plt.plot(epochs, pass_rates, marker='o', linewidth=2, markersize=6,
             label='Overall Pass Rate', color='#2E86AB')
    plt.plot(epochs, world_rates, marker='s', linewidth=2, markersize=5,
             label='World Validity', color='#A23B72', linestyle='--')
    plt.plot(epochs, noetic_rates, marker='^', linewidth=2, markersize=5,
             label='Noetic Validity', color='#F18F01', linestyle='--')
    plt.plot(epochs, operator_rates, marker='d', linewidth=2, markersize=5,
             label='Operator Validity', color='#C73E1D', linestyle='--')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-0.05, 1.05)  # 0-100% range with padding

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved validation rates to: {output_path}")


def plot_world_noetic_distribution(metrics: List[Dict[str, Any]], output_path: Path, title: str = "World and Noetic Distribution") -> None:
    """
    Plot distribution of worlds and noetics in data.

    Args:
        metrics: List of metric dictionaries with distribution data
        output_path: Path to save PNG output
        title: Plot title

    Creates two subplots:
        1. Bar chart of world distribution (A/B/C/D)
        2. Bar chart of noetic distribution (1-10)
    """
    # Use latest metric
    metric = metrics[-1] if metrics else {}

    # Extract distribution data
    if "distribution" in metric:
        dist = metric["distribution"]
        world_counts = dist.get("world_counts", {})
        noetic_counts = dist.get("noetic_counts", {})
    else:
        print(f"Warning: No distribution data found in metrics, skipping distribution plot")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # World distribution
    if world_counts:
        worlds = sorted(world_counts.keys())
        counts = [world_counts[w] for w in worlds]

        ax1.bar(worlds, counts, color='#2E86AB', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('World', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('World Distribution (A/B/C/D)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add count labels on bars
        for i, (world, count) in enumerate(zip(worlds, counts)):
            ax1.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=9)

    # Noetic distribution
    if noetic_counts:
        # Convert string keys to integers for sorting
        noetic_items = [(int(k) if isinstance(k, str) else k, v) for k, v in noetic_counts.items()]
        noetic_items.sort()
        noetics = [str(n) for n, _ in noetic_items]
        counts = [c for _, c in noetic_items]

        ax2.bar(noetics, counts, color='#F18F01', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Noetic Index', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Noetic Distribution (1-10)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add count labels on bars (only for taller bars to avoid clutter)
        max_count = max(counts) if counts else 0
        for i, (noetic, count) in enumerate(zip(noetics, counts)):
            if count > max_count * 0.1:  # Only label significant bars
                ax2.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=8)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved world/noetic distribution to: {output_path}")


def plot_augmentation_ratios(metrics: List[Dict[str, Any]], output_path: Path, title: str = "Augmentation Ratios Over Time") -> None:
    """
    Plot augmentation ratios over epochs/time.

    Args:
        metrics: List of metric dictionaries with augmentation ratio data
        output_path: Path to save PNG output
        title: Plot title

    Plots:
        - Total augmentation ratio
        - Inversion ratio
        - Anti-attractor ratio
    """
    epochs = []
    total_ratios = []
    inversion_ratios = []
    anti_ratios = []

    for i, metric in enumerate(metrics):
        # Determine epoch number
        if "epoch" in metric:
            epoch = metric["epoch"]
        else:
            epoch = i + 1

        # Extract ratios
        if "augmentation" in metric:
            aug = metric["augmentation"]
            total_ratio = aug.get("augmentation_ratio", 0)
            inversion_ratio = aug.get("inversion_ratio", 0)
            anti_ratio = aug.get("anti_attractor_ratio", 0)
        else:
            # CSV format
            total_ratio = metric.get("augmentation_ratio", 0)
            inversion_ratio = metric.get("inversion_ratio", 0)
            anti_ratio = metric.get("anti_attractor_ratio", 0)

        epochs.append(epoch)
        total_ratios.append(total_ratio)
        inversion_ratios.append(inversion_ratio)
        anti_ratios.append(anti_ratio)

    if not total_ratios:
        print(f"Warning: No augmentation ratio data found, skipping ratio plot")
        return

    # Create plot
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, total_ratios, marker='o', linewidth=2, markersize=6,
             label='Total Augmentation', color='#2E86AB')
    plt.plot(epochs, inversion_ratios, marker='s', linewidth=2, markersize=5,
             label='Inversion', color='#A23B72', linestyle='--')
    plt.plot(epochs, anti_ratios, marker='^', linewidth=2, markersize=5,
             label='Anti-Attractor', color='#C73E1D', linestyle='--')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Ratio (augmented / original)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved augmentation ratios to: {output_path}")


def plot_augmentation_counts_bar(metrics: List[Dict[str, Any]], output_path: Path, title: str = "Augmentation Counts by Type Over Time") -> None:
    """
    Plot augmentation counts by type as a grouped bar chart over epochs.

    Args:
        metrics: List of metric dictionaries with augmentation count data
        output_path: Path to save PNG output
        title: Plot title

    Shows stacked or grouped bars for:
        - Original count
        - Inversion count
        - Anti-attractor count

    This complements the pie chart by showing trends over time.
    """
    epochs = []
    original_counts = []
    inversion_counts = []
    anti_counts = []

    for i, metric in enumerate(metrics):
        # Determine epoch number
        if "epoch" in metric:
            epoch = metric["epoch"]
        else:
            epoch = i + 1

        # Extract counts
        if "augmentation" in metric:
            aug = metric["augmentation"]
            original = aug.get("original_count", 0)
            inversion = aug.get("inversion_count", 0)
            anti = aug.get("anti_attractor_count", 0)
        else:
            # CSV format
            original = metric.get("original_count", 0)
            inversion = metric.get("inversion_count", 0)
            anti = metric.get("anti_attractor_count", 0)

        epochs.append(epoch)
        original_counts.append(original)
        inversion_counts.append(inversion)
        anti_counts.append(anti)

    if not original_counts and not inversion_counts and not anti_counts:
        print(f"Warning: No augmentation count data found, skipping counts bar plot")
        return

    # Create grouped bar chart
    import numpy as np
    x = np.arange(len(epochs))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, original_counts, width, label='Original', color='#A23B72', alpha=0.8)
    plt.bar(x, inversion_counts, width, label='Inversion', color='#F18F01', alpha=0.8)
    plt.bar(x + width, anti_counts, width, label='Anti-Attractor', color='#C73E1D', alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, [str(e) for e in epochs])
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved augmentation counts bar chart to: {output_path}")


def plot_combined_dashboard(metrics: List[Dict[str, Any]], output_path: Path, title: str = "TKS Training + Augmentation Dashboard") -> None:
    """
    Plot a combined dashboard with training and augmentation metrics.

    Creates a 2x2 grid showing:
        - Top-left: Loss vs. Epoch
        - Top-right: Validator Pass Rate Over Time
        - Bottom-left: Augmentation Counts by Type (stacked area)
        - Bottom-right: Summary Statistics Table

    Args:
        metrics: List of metric dictionaries (from training_metrics.json or epoch files)
        output_path: Path to save PNG output
        title: Dashboard title

    This dashboard provides a single-view overview of training progress,
    validation quality, and augmentation balance.
    """
    # Extract data
    epochs = []
    losses = []
    pass_rates = []
    world_rates = []
    noetic_rates = []
    operator_rates = []
    original_counts = []
    inversion_counts = []
    anti_counts = []

    for i, metric in enumerate(metrics):
        # Determine epoch
        if "epoch" in metric:
            epoch = metric["epoch"]
        else:
            epoch = i + 1
        epochs.append(epoch)

        # Extract loss
        loss = None
        if "loss" in metric and isinstance(metric["loss"], dict):
            if "epoch_losses" in metric["loss"]:
                # Training summary format: extract for all epochs
                for ep, ep_loss in metric["loss"]["epoch_losses"]:
                    if ep == epoch:
                        loss = ep_loss
                        break
                if loss is None and metric["loss"]["epoch_losses"]:
                    # Use last loss if epoch not found
                    loss = metric["loss"]["epoch_losses"][-1][1]
            elif "final_loss" in metric["loss"]:
                loss = metric["loss"]["final_loss"]
        elif "loss" in metric and isinstance(metric["loss"], (int, float)):
            loss = metric["loss"]
        elif "avg_loss" in metric:
            loss = metric["avg_loss"]
        elif "pass_rate" in metric:
            loss = 1.0 - metric["pass_rate"]
        elif "validation" in metric and "pass_rate" in metric["validation"]:
            loss = 1.0 - metric["validation"]["pass_rate"]

        losses.append(loss if loss is not None else 0)

        # Extract validation rates
        if "validation" in metric:
            val = metric["validation"]
            pass_rates.append(val.get("pass_rate", 0))
            world_rates.append(val.get("world_validity_rate", 0))
            noetic_rates.append(val.get("noetic_validity_rate", 0))
            operator_rates.append(val.get("operator_validity_rate", 0))
        else:
            pass_rates.append(metric.get("pass_rate", 0))
            world_rates.append(metric.get("world_validity_rate", 0))
            noetic_rates.append(metric.get("noetic_validity_rate", 0))
            operator_rates.append(metric.get("operator_validity_rate", 0))

        # Extract augmentation counts
        if "augmentation" in metric:
            aug = metric["augmentation"]
            original_counts.append(aug.get("original_count", 0))
            inversion_counts.append(aug.get("inversion_count", 0))
            anti_counts.append(aug.get("anti_attractor_count", 0))
        else:
            original_counts.append(metric.get("original_count", 0))
            inversion_counts.append(metric.get("inversion_count", 0))
            anti_counts.append(metric.get("anti_attractor_count", 0))

    # Handle training summary with epoch_losses array
    if len(metrics) == 1 and "loss" in metrics[0] and isinstance(metrics[0]["loss"], dict):
        loss_data = metrics[0]["loss"]
        if "epoch_losses" in loss_data:
            epochs = [ep for ep, _ in loss_data["epoch_losses"]]
            losses = [l for _, l in loss_data["epoch_losses"]]
            # Replicate other data for each epoch if needed
            if len(pass_rates) < len(epochs):
                val = metrics[0].get("validation", {})
                pr = val.get("pass_rate", 0)
                pass_rates = [pr] * len(epochs)
                world_rates = [val.get("world_validity_rate", 0)] * len(epochs)
                noetic_rates = [val.get("noetic_validity_rate", 0)] * len(epochs)
                operator_rates = [val.get("operator_validity_rate", 0)] * len(epochs)
                aug = metrics[0].get("augmentation", {})
                original_counts = [aug.get("original_count", 0)] * len(epochs)
                inversion_counts = [aug.get("inversion_count", 0)] * len(epochs)
                anti_counts = [aug.get("anti_attractor_count", 0)] * len(epochs)

    if not epochs:
        print(f"Warning: No data found for dashboard, skipping")
        return

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Top-left: Loss vs Epoch
    ax1 = axes[0, 0]
    if any(l > 0 for l in losses):
        ax1.plot(epochs, losses, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Loss vs. Epoch', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
    else:
        ax1.text(0.5, 0.5, 'No loss data available', ha='center', va='center',
                 fontsize=12, color='gray', transform=ax1.transAxes)
        ax1.set_title('Loss vs. Epoch', fontsize=12, fontweight='bold')

    # Top-right: Validation Pass Rates
    ax2 = axes[0, 1]
    if any(p > 0 for p in pass_rates):
        ax2.plot(epochs, pass_rates, marker='o', linewidth=2, markersize=6,
                 label='Overall Pass Rate', color='#2E86AB')
        if any(w > 0 for w in world_rates):
            ax2.plot(epochs, world_rates, marker='s', linewidth=1.5, markersize=4,
                     label='World Validity', color='#A23B72', linestyle='--', alpha=0.7)
        if any(n > 0 for n in noetic_rates):
            ax2.plot(epochs, noetic_rates, marker='^', linewidth=1.5, markersize=4,
                     label='Noetic Validity', color='#F18F01', linestyle='--', alpha=0.7)
        if any(o > 0 for o in operator_rates):
            ax2.plot(epochs, operator_rates, marker='d', linewidth=1.5, markersize=4,
                     label='Operator Validity', color='#C73E1D', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Validation Rate', fontsize=11)
        ax2.set_title('Validator Pass Rate Over Time', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(-0.05, 1.05)
    else:
        ax2.text(0.5, 0.5, 'No validation data available', ha='center', va='center',
                 fontsize=12, color='gray', transform=ax2.transAxes)
        ax2.set_title('Validator Pass Rate Over Time', fontsize=12, fontweight='bold')

    # Bottom-left: Augmentation Counts (stacked area)
    ax3 = axes[1, 0]
    has_aug_data = any(o > 0 for o in original_counts) or any(i > 0 for i in inversion_counts) or any(a > 0 for a in anti_counts)
    if has_aug_data:
        import numpy as np
        ax3.stackplot(epochs,
                      original_counts,
                      inversion_counts,
                      anti_counts,
                      labels=['Original', 'Inversion', 'Anti-Attractor'],
                      colors=['#A23B72', '#F18F01', '#C73E1D'],
                      alpha=0.8)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Augmentation Counts by Type', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, 'No augmentation data available', ha='center', va='center',
                 fontsize=12, color='gray', transform=ax3.transAxes)
        ax3.set_title('Augmentation Counts by Type', fontsize=12, fontweight='bold')

    # Bottom-right: Summary Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute summary stats
    total_original = sum(original_counts)
    total_inversion = sum(inversion_counts)
    total_anti = sum(anti_counts)
    total_samples = total_original + total_inversion + total_anti

    final_loss = losses[-1] if losses else 0
    final_pass_rate = pass_rates[-1] if pass_rates else 0
    avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0

    aug_ratio = (total_inversion + total_anti) / total_original if total_original > 0 else 0

    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['Total Epochs', f'{len(epochs)}'],
        ['Total Samples', f'{total_samples:,}'],
        ['Original Count', f'{total_original:,}'],
        ['Inversion Count', f'{total_inversion:,}'],
        ['Anti-Attractor Count', f'{total_anti:,}'],
        ['Augmentation Ratio', f'{aug_ratio:.2f}x'],
        ['Final Loss', f'{final_loss:.4f}'],
        ['Final Pass Rate', f'{final_pass_rate:.2%}'],
        ['Avg Pass Rate', f'{avg_pass_rate:.2%}'],
    ]

    table = ax4.table(
        cellText=table_data,
        loc='center',
        cellLoc='left',
        colWidths=[0.5, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(2):
            table[(i, j)].set_facecolor(color)

    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined dashboard to: {output_path}")


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TKS Metrics Plotting Script - Generate visualization plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all plots from JSON metrics
    python scripts/plot_metrics.py --input output/metrics.json --output-dir output/plots --plot-type all

    # Generate loss curve from CSV
    python scripts/plot_metrics.py --input output/metrics.csv --output-dir output/plots --plot-type loss

    # Generate specific plots
    python scripts/plot_metrics.py --input output/metrics.json --output-dir output/plots --plot-type distribution
    python scripts/plot_metrics.py --input output/metrics.json --output-dir output/plots --plot-type validation

    # Generate combined dashboard (training + augmentation overview)
    python scripts/plot_metrics.py --input output/training_metrics.json --output-dir output/plots --plot-type dashboard

    # Generate augmentation counts bar chart
    python scripts/plot_metrics.py --input output/metrics.json --output-dir output/plots --plot-type counts-bar

Plot types:
    - loss: Training loss curve over epochs
    - distribution: Augmentation type distribution (pie chart)
    - validation: Validation pass rates over time
    - world-noetic: World and noetic distribution (bar charts)
    - ratios: Augmentation ratios over time
    - counts-bar: Augmentation counts by type (grouped bar chart)
    - dashboard: Combined 2x2 dashboard (loss, validation, augmentation, summary)
    - all: Generate all available plots including dashboard
        """
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to metrics file (JSON or CSV)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save PNG plot outputs"
    )

    parser.add_argument(
        "--plot-type",
        choices=["loss", "distribution", "validation", "world-noetic", "ratios", "counts-bar", "dashboard", "all"],
        default="all",
        help="Which plots to generate (default: all)"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames (default: none)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required but not installed.")
        return 1

    print("=" * 70)
    print("TKS METRICS PLOTTING")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Type:   {args.plot_type}")
    print()

    # Load metrics
    try:
        metrics = load_metrics(args.input)
        print(f"Loaded {len(metrics)} metric entries from {args.input}")
        print()
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return 1

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots based on type
    prefix = f"{args.prefix}_" if args.prefix else ""

    try:
        if args.plot_type in ["loss", "all"]:
            output_path = args.output_dir / f"{prefix}loss_curve.png"
            plot_loss_curve(metrics, output_path)

        if args.plot_type in ["distribution", "all"]:
            output_path = args.output_dir / f"{prefix}augmentation_distribution.png"
            plot_augmentation_distribution(metrics, output_path)

        if args.plot_type in ["validation", "all"]:
            output_path = args.output_dir / f"{prefix}validation_rates.png"
            plot_validation_rates(metrics, output_path)

        if args.plot_type in ["world-noetic", "all"]:
            output_path = args.output_dir / f"{prefix}world_noetic_distribution.png"
            plot_world_noetic_distribution(metrics, output_path)

        if args.plot_type in ["ratios", "all"]:
            output_path = args.output_dir / f"{prefix}augmentation_ratios.png"
            plot_augmentation_ratios(metrics, output_path)

        if args.plot_type in ["counts-bar", "all"]:
            output_path = args.output_dir / f"{prefix}augmentation_counts_bar.png"
            plot_augmentation_counts_bar(metrics, output_path)

        if args.plot_type in ["dashboard", "all"]:
            output_path = args.output_dir / f"{prefix}combined_dashboard.png"
            plot_combined_dashboard(metrics, output_path)

        print()
        print("=" * 70)
        print("PLOTTING COMPLETE")
        print("=" * 70)
        print(f"Plots saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

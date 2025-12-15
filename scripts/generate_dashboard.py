#!/usr/bin/env python3
"""
TKS Phase 6 Monitoring Dashboard Generator

Generates comprehensive monitoring dashboards from TKS training and augmentation metrics.
Produces both individual plots and an integrated HTML dashboard with embedded images.

Features:
    - Loss vs. epoch curves (training and eval)
    - Validator pass-rate over time (overall and component-level)
    - Augmentation counts by type (teacher/inversion/anti-attractor)
    - Anti-attractor usage tracking
    - HTML dashboard with all plots embedded
    - Support for both JSON and CSV metrics

Canon Guardrails:
    - Worlds: A/B/C/D
    - Noetics: 1-10 (pairs 2-3, 5-6, 8-9; self-duals 1,4,7,10)
    - Foundations: 1-7, Sub-foundations: 7x4=28
    - ALLOWED_OPS: +, -, +T, -T, ->, <-, *T, /T, o (9 total)

Usage:
    # Generate dashboard from training metrics
    python scripts/generate_dashboard.py --input output/models/metrics/training_metrics.json --output-dir output/dashboard

    # Generate from augmentation metrics
    python scripts/generate_dashboard.py --input output/teacher_augmented.metrics.json --output-dir output/dashboard

    # Generate from multiple epoch files
    python scripts/generate_dashboard.py --input output/models/metrics --output-dir output/dashboard --multi-epoch

    # Generate from CSV
    python scripts/generate_dashboard.py --input output/models/metrics/training_metrics_epochs.csv --output-dir output/dashboard

Dependencies:
    - matplotlib (required)

Author: TKS-LLM Agent 4 - Monitoring Dashboard
Date: 2025-12-14
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
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


def load_multi_epoch_metrics(directory: Path) -> List[Dict[str, Any]]:
    """
    Load metrics from multiple epoch files in a directory.

    Args:
        directory: Path to directory containing epoch_NNN_metrics.json files

    Returns:
        List of metric dictionaries sorted by epoch
    """
    epoch_files = sorted(directory.glob("epoch_*_metrics.json"))

    if not epoch_files:
        raise FileNotFoundError(f"No epoch metric files found in {directory}")

    metrics = []
    for epoch_file in epoch_files:
        with epoch_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            metrics.append(data)

    return metrics


def load_metrics(filepath: Path, multi_epoch: bool = False) -> List[Dict[str, Any]]:
    """
    Load metrics from JSON, CSV file, or directory (auto-detect format).

    Args:
        filepath: Path to metrics file or directory
        multi_epoch: If True and filepath is directory, load all epoch files

    Returns:
        List of metric dictionaries
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file/directory not found: {filepath}")

    # Handle directory input for multi-epoch
    if filepath.is_dir():
        if multi_epoch:
            return load_multi_epoch_metrics(filepath)
        else:
            # Look for training_metrics.json in directory
            training_metrics = filepath / "training_metrics.json"
            if training_metrics.exists():
                return load_metrics_json(training_metrics)
            else:
                raise FileNotFoundError(f"No training_metrics.json found in {filepath}")

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

def plot_loss_curves(metrics: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Plot training and evaluation loss curves.

    Args:
        metrics: List of metric dictionaries with loss data
        output_path: Path to save PNG output
    """
    epochs = []
    train_losses = []
    eval_losses = []

    for i, metric in enumerate(metrics):
        # Determine epoch
        if "epoch" in metric:
            epoch = metric["epoch"]
        else:
            epoch = i + 1

        # Extract training loss
        train_loss = None
        if "loss" in metric and isinstance(metric["loss"], dict):
            if "epoch_losses" in metric["loss"]:
                # Training summary format
                for ep_data in metric["loss"]["epoch_losses"]:
                    if isinstance(ep_data, dict) and ep_data.get("epoch") == epoch:
                        train_loss = ep_data.get("loss")
                        break
                    elif isinstance(ep_data, list) and len(ep_data) == 2:
                        if ep_data[0] == epoch:
                            train_loss = ep_data[1]
                            break
            elif "final_loss" in metric["loss"]:
                train_loss = metric["loss"]["final_loss"]
        elif "loss" in metric and isinstance(metric["loss"], (int, float)):
            train_loss = metric["loss"]

        # Extract eval loss
        eval_loss = None
        if "eval_results" in metric:
            for eval_result in metric["eval_results"]:
                if eval_result.get("epoch") == epoch:
                    eval_loss = eval_result.get("loss")
                    break

        if train_loss is not None or eval_loss is not None:
            epochs.append(epoch)
            train_losses.append(train_loss if train_loss is not None else 0)
            eval_losses.append(eval_loss if eval_loss is not None else 0)

    # Handle single metric with epoch_losses array
    if len(metrics) == 1 and "loss" in metrics[0] and isinstance(metrics[0]["loss"], dict):
        loss_data = metrics[0]["loss"]
        if "epoch_losses" in loss_data:
            epochs = []
            train_losses = []
            for ep_data in loss_data["epoch_losses"]:
                if isinstance(ep_data, dict):
                    epochs.append(ep_data.get("epoch", len(epochs) + 1))
                    train_losses.append(ep_data.get("loss", 0))
                elif isinstance(ep_data, list) and len(ep_data) == 2:
                    epochs.append(ep_data[0])
                    train_losses.append(ep_data[1])

        # Handle eval losses
        if "eval_losses" in loss_data:
            eval_losses = []
            for ep_data in loss_data["eval_losses"]:
                if isinstance(ep_data, dict):
                    eval_losses.append(ep_data.get("loss", 0))
                elif isinstance(ep_data, list) and len(ep_data) == 2:
                    eval_losses.append(ep_data[1])

        # Handle eval_results
        if "eval_results" in metrics[0]:
            eval_losses = []
            for eval_result in metrics[0]["eval_results"]:
                eval_losses.append(eval_result.get("loss", 0))

    if not epochs:
        print(f"Warning: No loss data found in metrics, skipping loss plot")
        return

    # Create plot
    plt.figure(figsize=(12, 7))

    if any(l > 0 for l in train_losses):
        plt.plot(epochs, train_losses, marker='o', linewidth=2, markersize=7,
                label='Training Loss', color='#2E86AB')

    if any(l > 0 for l in eval_losses):
        plt.plot(epochs, eval_losses, marker='s', linewidth=2, markersize=6,
                label='Eval Loss', color='#F18F01', linestyle='--')

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title('Training and Evaluation Loss Over Time', fontsize=15, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved loss curves to: {output_path}")


def plot_validation_rates(metrics: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Plot validator pass rates over time.

    Args:
        metrics: List of metric dictionaries with validation data
        output_path: Path to save PNG output
    """
    epochs = []
    pass_rates = []
    world_rates = []
    noetic_rates = []
    operator_rates = []

    for i, metric in enumerate(metrics):
        # Determine epoch
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
            pass_rate = metric.get("pass_rate", 0)
            world_rate = metric.get("world_validity_rate", 0)
            noetic_rate = metric.get("noetic_validity_rate", 0)
            operator_rate = metric.get("operator_validity_rate", 0)

        epochs.append(epoch)
        pass_rates.append(pass_rate)
        world_rates.append(world_rate)
        noetic_rates.append(noetic_rate)
        operator_rates.append(operator_rate)

    if not pass_rates and not world_rates:
        print(f"Warning: No validation data found, skipping validation plot")
        return

    # Create plot
    plt.figure(figsize=(12, 7))

    if any(p > 0 for p in pass_rates):
        plt.plot(epochs, pass_rates, marker='o', linewidth=2.5, markersize=7,
                label='Overall Pass Rate', color='#2E86AB')

    if any(w > 0 for w in world_rates):
        plt.plot(epochs, world_rates, marker='s', linewidth=2, markersize=6,
                label='World Validity', color='#A23B72', linestyle='--', alpha=0.8)

    if any(n > 0 for n in noetic_rates):
        plt.plot(epochs, noetic_rates, marker='^', linewidth=2, markersize=6,
                label='Noetic Validity', color='#F18F01', linestyle='--', alpha=0.8)

    if any(o > 0 for o in operator_rates):
        plt.plot(epochs, operator_rates, marker='d', linewidth=2, markersize=6,
                label='Operator Validity', color='#C73E1D', linestyle='--', alpha=0.8)

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Validation Rate', fontsize=13)
    plt.title('Validator Pass Rates Over Time', fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved validation rates to: {output_path}")


def plot_augmentation_counts(metrics: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Plot augmentation counts by type over time.

    Args:
        metrics: List of metric dictionaries with augmentation data
        output_path: Path to save PNG output
    """
    epochs = []
    original_counts = []
    inversion_counts = []
    anti_counts = []

    for i, metric in enumerate(metrics):
        # Determine epoch
        if "epoch" in metric:
            epoch = metric["epoch"]
        else:
            epoch = i + 1

        # Extract augmentation counts
        if "augmentation" in metric:
            aug = metric["augmentation"]
            # Handle different structures
            if "distribution" in aug:
                dist = aug["distribution"]
                original = dist.get("original", 0)
                inversion = dist.get("inversion", 0)
                anti = dist.get("anti_attractor", 0)
            else:
                original = aug.get("original_count", 0)
                inversion = aug.get("inversion_count", 0) or aug.get("inverted_count", 0)
                anti = aug.get("anti_attractor_count", 0)
        else:
            original = metric.get("original_count", 0)
            inversion = metric.get("inversion_count", 0) or metric.get("inverted_count", 0)
            anti = metric.get("anti_attractor_count", 0)

        epochs.append(epoch)
        original_counts.append(original)
        inversion_counts.append(inversion)
        anti_counts.append(anti)

    if not any(original_counts) and not any(inversion_counts) and not any(anti_counts):
        print(f"Warning: No augmentation count data found, skipping augmentation plot")
        return

    # Create grouped bar chart
    x = np.arange(len(epochs))
    width = 0.25

    plt.figure(figsize=(12, 7))

    plt.bar(x - width, original_counts, width, label='Original (Teacher)',
            color='#A23B72', alpha=0.85, edgecolor='black', linewidth=0.5)
    plt.bar(x, inversion_counts, width, label='Inversion',
            color='#F18F01', alpha=0.85, edgecolor='black', linewidth=0.5)
    plt.bar(x + width, anti_counts, width, label='Anti-Attractor',
            color='#C73E1D', alpha=0.85, edgecolor='black', linewidth=0.5)

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.title('Augmentation Counts by Type Over Time', fontsize=15, fontweight='bold')
    plt.xticks(x, [str(e) for e in epochs])
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved augmentation counts to: {output_path}")


def plot_anti_attractor_usage(metrics: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Plot anti-attractor usage count over time.

    Args:
        metrics: List of metric dictionaries with anti-attractor data
        output_path: Path to save PNG output
    """
    epochs = []
    anti_counts = []
    anti_ratios = []

    for i, metric in enumerate(metrics):
        # Determine epoch
        if "epoch" in metric:
            epoch = metric["epoch"]
        else:
            epoch = i + 1

        # Extract anti-attractor data
        anti_count = 0
        anti_ratio = 0

        if "augmentation" in metric:
            aug = metric["augmentation"]
            if "distribution" in aug:
                anti_count = aug["distribution"].get("anti_attractor", 0)
            else:
                anti_count = aug.get("anti_attractor_count", 0)
            anti_ratio = aug.get("anti_attractor_ratio", 0)
        else:
            anti_count = metric.get("anti_attractor_count", 0)
            anti_ratio = metric.get("anti_attractor_ratio", 0)

        epochs.append(epoch)
        anti_counts.append(anti_count)
        anti_ratios.append(anti_ratio)

    if not any(anti_counts):
        print(f"Warning: No anti-attractor data found, skipping anti-attractor plot")
        return

    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = '#C73E1D'
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Anti-Attractor Count', color=color, fontsize=13)
    ax1.bar(epochs, anti_counts, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add ratio line if available
    if any(anti_ratios):
        ax2 = ax1.twinx()
        color = '#2E86AB'
        ax2.set_ylabel('Anti-Attractor Ratio (vs Original)', color=color, fontsize=13)
        ax2.plot(epochs, anti_ratios, color=color, marker='o', linewidth=2.5,
                markersize=7, label='Ratio')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Anti-Attractor Usage Over Time', fontsize=15, fontweight='bold')
    fig.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved anti-attractor usage to: {output_path}")


def plot_comprehensive_dashboard(metrics: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Plot comprehensive 2x2 dashboard with key metrics.

    Creates a 2x2 grid showing:
        - Top-left: Loss vs. Epoch
        - Top-right: Validator Pass Rate
        - Bottom-left: Augmentation Counts
        - Bottom-right: Summary Statistics

    Args:
        metrics: List of metric dictionaries
        output_path: Path to save PNG output
    """
    # Extract all data
    epochs = []
    train_losses = []
    eval_losses = []
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
        train_loss = None
        if "loss" in metric and isinstance(metric["loss"], dict):
            if "epoch_losses" in metric["loss"]:
                for ep_data in metric["loss"]["epoch_losses"]:
                    if isinstance(ep_data, dict) and ep_data.get("epoch") == epoch:
                        train_loss = ep_data.get("loss")
                        break
                    elif isinstance(ep_data, list) and len(ep_data) == 2:
                        if ep_data[0] == epoch:
                            train_loss = ep_data[1]
                            break
            elif "final_loss" in metric["loss"]:
                train_loss = metric["loss"]["final_loss"]
        elif "loss" in metric and isinstance(metric["loss"], (int, float)):
            train_loss = metric["loss"]

        train_losses.append(train_loss if train_loss is not None else 0)

        # Extract eval loss
        eval_loss = None
        if "eval_results" in metric:
            for eval_result in metric["eval_results"]:
                if eval_result.get("epoch") == epoch:
                    eval_loss = eval_result.get("loss")
                    break
        eval_losses.append(eval_loss if eval_loss is not None else 0)

        # Extract validation rates
        if "validation" in metric:
            val = metric["validation"]
            pass_rates.append(val.get("pass_rate", 0))
            world_rates.append(val.get("world_validity_rate", 0))
            noetic_rates.append(val.get("noetic_validity_rate", 0))
            operator_rates.append(val.get("operator_validity_rate", 0))
        else:
            pass_rates.append(0)
            world_rates.append(0)
            noetic_rates.append(0)
            operator_rates.append(0)

        # Extract augmentation counts
        if "augmentation" in metric:
            aug = metric["augmentation"]
            if "distribution" in aug:
                dist = aug["distribution"]
                original_counts.append(dist.get("original", 0))
                inversion_counts.append(dist.get("inversion", 0))
                anti_counts.append(dist.get("anti_attractor", 0))
            else:
                original_counts.append(aug.get("original_count", 0))
                inversion_counts.append(aug.get("inversion_count", 0) or aug.get("inverted_count", 0))
                anti_counts.append(aug.get("anti_attractor_count", 0))
        else:
            original_counts.append(0)
            inversion_counts.append(0)
            anti_counts.append(0)

    # Handle single metric with epoch_losses array
    if len(metrics) == 1 and "loss" in metrics[0] and isinstance(metrics[0]["loss"], dict):
        loss_data = metrics[0]["loss"]
        if "epoch_losses" in loss_data:
            epochs = []
            train_losses = []
            for ep_data in loss_data["epoch_losses"]:
                if isinstance(ep_data, dict):
                    epochs.append(ep_data.get("epoch", len(epochs) + 1))
                    train_losses.append(ep_data.get("loss", 0))
                elif isinstance(ep_data, list) and len(ep_data) == 2:
                    epochs.append(ep_data[0])
                    train_losses.append(ep_data[1])

            # Replicate other data
            if len(pass_rates) < len(epochs):
                val = metrics[0].get("validation", {})
                pr = val.get("pass_rate", 0)
                pass_rates = [pr] * len(epochs)
                world_rates = [val.get("world_validity_rate", 0)] * len(epochs)
                noetic_rates = [val.get("noetic_validity_rate", 0)] * len(epochs)
                operator_rates = [val.get("operator_validity_rate", 0)] * len(epochs)

                aug = metrics[0].get("augmentation", {})
                if "distribution" in aug:
                    dist = aug["distribution"]
                    oc = dist.get("original", 0)
                    ic = dist.get("inversion", 0)
                    ac = dist.get("anti_attractor", 0)
                else:
                    oc = aug.get("original_count", 0)
                    ic = aug.get("inversion_count", 0) or aug.get("inverted_count", 0)
                    ac = aug.get("anti_attractor_count", 0)

                original_counts = [oc] * len(epochs)
                inversion_counts = [ic] * len(epochs)
                anti_counts = [ac] * len(epochs)

    if not epochs:
        print(f"Warning: No data found for dashboard, skipping")
        return

    # Create 2x2 dashboard
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top-left: Loss vs Epoch
    ax1 = fig.add_subplot(gs[0, 0])
    if any(l > 0 for l in train_losses):
        ax1.plot(epochs, train_losses, marker='o', linewidth=2, markersize=6,
                color='#2E86AB', label='Training Loss')
    if any(l > 0 for l in eval_losses):
        ax1.plot(epochs, eval_losses, marker='s', linewidth=2, markersize=5,
                color='#F18F01', linestyle='--', label='Eval Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss vs. Epoch', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    if any(l > 0 for l in eval_losses):
        ax1.legend(fontsize=9)

    # Top-right: Validator Pass Rates
    ax2 = fig.add_subplot(gs[0, 1])
    if any(p > 0 for p in pass_rates):
        ax2.plot(epochs, pass_rates, marker='o', linewidth=2.5, markersize=6,
                label='Overall', color='#2E86AB')
        if any(w > 0 for w in world_rates):
            ax2.plot(epochs, world_rates, marker='s', linewidth=1.5, markersize=4,
                    label='World', color='#A23B72', linestyle='--', alpha=0.7)
        if any(n > 0 for n in noetic_rates):
            ax2.plot(epochs, noetic_rates, marker='^', linewidth=1.5, markersize=4,
                    label='Noetic', color='#F18F01', linestyle='--', alpha=0.7)
        if any(o > 0 for o in operator_rates):
            ax2.plot(epochs, operator_rates, marker='d', linewidth=1.5, markersize=4,
                    label='Operator', color='#C73E1D', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Validation Rate', fontsize=11)
        ax2.set_title('Validator Pass Rate Over Time', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='lower right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(-0.05, 1.05)
    else:
        ax2.text(0.5, 0.5, 'No validation data', ha='center', va='center',
                fontsize=11, color='gray', transform=ax2.transAxes)
        ax2.set_title('Validator Pass Rate Over Time', fontsize=12, fontweight='bold')

    # Bottom-left: Augmentation Counts (stacked area)
    ax3 = fig.add_subplot(gs[1, 0])
    has_aug = any(o > 0 for o in original_counts) or any(i > 0 for i in inversion_counts) or any(a > 0 for a in anti_counts)
    if has_aug:
        ax3.stackplot(epochs,
                     original_counts,
                     inversion_counts,
                     anti_counts,
                     labels=['Original (Teacher)', 'Inversion', 'Anti-Attractor'],
                     colors=['#A23B72', '#F18F01', '#C73E1D'],
                     alpha=0.8)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Augmentation Counts by Type', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='upper left')
        ax3.grid(True, alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, 'No augmentation data', ha='center', va='center',
                fontsize=11, color='gray', transform=ax3.transAxes)
        ax3.set_title('Augmentation Counts by Type', fontsize=12, fontweight='bold')

    # Bottom-right: Summary Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Compute summary
    total_original = sum(original_counts)
    total_inversion = sum(inversion_counts)
    total_anti = sum(anti_counts)
    total_samples = total_original + total_inversion + total_anti

    final_loss = train_losses[-1] if train_losses else 0
    final_eval_loss = eval_losses[-1] if eval_losses else 0
    final_pass_rate = pass_rates[-1] if pass_rates else 0
    avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0

    aug_ratio = (total_inversion + total_anti) / total_original if total_original > 0 else 0
    anti_ratio = total_anti / total_original if total_original > 0 else 0

    # Create table
    table_data = [
        ['Metric', 'Value'],
        ['Total Epochs', f'{len(epochs)}'],
        ['Total Samples', f'{total_samples:,}'],
        ['Original (Teacher)', f'{total_original:,}'],
        ['Inversion Count', f'{total_inversion:,}'],
        ['Anti-Attractor Count', f'{total_anti:,}'],
        ['Augmentation Ratio', f'{aug_ratio:.2f}x'],
        ['Anti-Attractor Ratio', f'{anti_ratio:.2f}x'],
        ['Final Train Loss', f'{final_loss:.4f}'],
        ['Final Eval Loss', f'{final_eval_loss:.4f}'],
        ['Final Pass Rate', f'{final_pass_rate:.2%}'],
        ['Avg Pass Rate', f'{avg_pass_rate:.2%}'],
    ]

    table = ax4.table(
        cellText=table_data,
        loc='center',
        cellLoc='left',
        colWidths=[0.55, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(2):
            table[(i, j)].set_facecolor(color)

    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('TKS Phase 6 Monitoring Dashboard', fontsize=16, fontweight='bold', y=0.98)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved comprehensive dashboard to: {output_path}")


# ==============================================================================
# HTML DASHBOARD GENERATION
# ==============================================================================

def generate_html_dashboard(output_dir: Path, metrics: List[Dict[str, Any]]) -> None:
    """
    Generate HTML dashboard with embedded plots.

    Args:
        output_dir: Directory containing PNG plots
        metrics: Metrics data for summary
    """
    # Compute summary stats
    total_original = 0
    total_inversion = 0
    total_anti = 0

    for metric in metrics:
        if "augmentation" in metric:
            aug = metric["augmentation"]
            if "distribution" in aug:
                dist = aug["distribution"]
                total_original += dist.get("original", 0)
                total_inversion += dist.get("inversion", 0)
                total_anti += dist.get("anti_attractor", 0)
            else:
                total_original += aug.get("original_count", 0)
                total_inversion += aug.get("inversion_count", 0) or aug.get("inverted_count", 0)
                total_anti += aug.get("anti_attractor_count", 0)

    total_samples = total_original + total_inversion + total_anti
    aug_ratio = (total_inversion + total_anti) / total_original if total_original > 0 else 0

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TKS Phase 6 Monitoring Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E86AB;
            text-align: center;
            margin-bottom: 10px;
            font-size: 32px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .summary {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 30px;
            border-left: 4px solid #2E86AB;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #2E86AB;
            font-size: 20px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            background-color: white;
            padding: 12px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .summary-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2E86AB;
            margin-top: 5px;
        }}
        .plot-section {{
            margin-bottom: 40px;
        }}
        .plot-section h2 {{
            color: #2E86AB;
            margin-bottom: 15px;
            font-size: 22px;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 10px;
        }}
        .plot-container {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
        .canon-note {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .canon-note h3 {{
            margin-top: 0;
            color: #856404;
            font-size: 16px;
        }}
        .canon-note ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .canon-note li {{
            color: #856404;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>TKS Phase 6 Monitoring Dashboard</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

        <div class="canon-note">
            <h3>Canon Guardrails</h3>
            <ul>
                <li><strong>Worlds:</strong> A/B/C/D</li>
                <li><strong>Noetics:</strong> 1-10 (pairs 2-3, 5-6, 8-9; self-duals 1,4,7,10)</li>
                <li><strong>Foundations:</strong> 1-7, Sub-foundations: 7x4=28</li>
                <li><strong>ALLOWED_OPS:</strong> +, -, +T, -T, ->, &lt;-, *T, /T, o (9 total)</li>
            </ul>
        </div>

        <div class="summary">
            <h2>Quick Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Total Samples</div>
                    <div class="summary-value">{total_samples:,}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Original (Teacher)</div>
                    <div class="summary-value">{total_original:,}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Inversion</div>
                    <div class="summary-value">{total_inversion:,}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Anti-Attractor</div>
                    <div class="summary-value">{total_anti:,}</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Augmentation Ratio</div>
                    <div class="summary-value">{aug_ratio:.2f}x</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Epochs</div>
                    <div class="summary-value">{len(metrics)}</div>
                </div>
            </div>
        </div>

        <div class="plot-section">
            <h2>Comprehensive Dashboard</h2>
            <div class="plot-container">
                <img src="comprehensive_dashboard.png" alt="Comprehensive Dashboard">
            </div>
        </div>

        <div class="plot-section">
            <h2>Training Progress</h2>
            <div class="plot-container">
                <img src="loss_curves.png" alt="Loss Curves">
            </div>
        </div>

        <div class="plot-section">
            <h2>Validation Metrics</h2>
            <div class="plot-container">
                <img src="validation_rates.png" alt="Validation Rates">
            </div>
        </div>

        <div class="plot-section">
            <h2>Augmentation Analysis</h2>
            <div class="plot-container">
                <img src="augmentation_counts.png" alt="Augmentation Counts">
            </div>
            <div class="plot-container">
                <img src="anti_attractor_usage.png" alt="Anti-Attractor Usage">
            </div>
        </div>

        <div class="footer">
            TKS Phase 6 Monitoring Dashboard | Generated by Agent 4 | {datetime.now().strftime('%Y-%m-%d')}
        </div>
    </div>
</body>
</html>
"""

    # Save HTML
    html_path = output_dir / "dashboard.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Saved HTML dashboard to: {html_path}")


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TKS Phase 6 Monitoring Dashboard Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate dashboard from training metrics
    python scripts/generate_dashboard.py --input output/models/metrics/training_metrics.json --output-dir output/dashboard

    # Generate from augmentation metrics
    python scripts/generate_dashboard.py --input output/teacher_augmented.metrics.json --output-dir output/dashboard

    # Generate from multiple epoch files
    python scripts/generate_dashboard.py --input output/models/metrics --output-dir output/dashboard --multi-epoch

    # Generate from CSV
    python scripts/generate_dashboard.py --input output/models/metrics/training_metrics_epochs.csv --output-dir output/dashboard

    # Skip HTML generation
    python scripts/generate_dashboard.py --input output/metrics.json --output-dir output/dashboard --no-html
        """
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to metrics file or directory (JSON/CSV)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save dashboard outputs"
    )

    parser.add_argument(
        "--multi-epoch",
        action="store_true",
        help="Load all epoch_NNN_metrics.json files from input directory"
    )

    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML dashboard generation (only generate PNGs)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required but not installed.")
        return 1

    print("=" * 70)
    print("TKS PHASE 6 MONITORING DASHBOARD")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Multi-epoch: {args.multi_epoch}")
    print()

    # Load metrics
    try:
        metrics = load_metrics(args.input, multi_epoch=args.multi_epoch)
        print(f"Loaded {len(metrics)} metric entries from {args.input}")
        print()
    except Exception as e:
        print(f"Error loading metrics: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    try:
        print("Generating plots...")
        print()

        # 1. Loss curves
        loss_path = args.output_dir / "loss_curves.png"
        plot_loss_curves(metrics, loss_path)

        # 2. Validation rates
        validation_path = args.output_dir / "validation_rates.png"
        plot_validation_rates(metrics, validation_path)

        # 3. Augmentation counts
        aug_path = args.output_dir / "augmentation_counts.png"
        plot_augmentation_counts(metrics, aug_path)

        # 4. Anti-attractor usage
        anti_path = args.output_dir / "anti_attractor_usage.png"
        plot_anti_attractor_usage(metrics, anti_path)

        # 5. Comprehensive dashboard
        dashboard_path = args.output_dir / "comprehensive_dashboard.png"
        plot_comprehensive_dashboard(metrics, dashboard_path)

        print()
        print("=" * 70)
        print("PLOTS GENERATED")
        print("=" * 70)
        print(f"Output directory: {args.output_dir}")
        print()

        # Generate HTML dashboard
        if not args.no_html:
            print("Generating HTML dashboard...")
            generate_html_dashboard(args.output_dir, metrics)
            print()
            print("=" * 70)
            print("DASHBOARD COMPLETE")
            print("=" * 70)
            print(f"View dashboard: {args.output_dir / 'dashboard.html'}")
        else:
            print("=" * 70)
            print("DASHBOARD COMPLETE (PNGs only)")
            print("=" * 70)

    except Exception as e:
        print(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

# TKS Telemetry & Monitoring Guide

Complete guide to the TKS telemetry and monitoring system for tracking augmentation and training metrics.

**Author:** TKS-LLM Training Integration Team
**Date:** 2025-12-14
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Metrics Collection](#metrics-collection)
4. [Persistence Formats](#persistence-formats)
5. [Visualization](#visualization)
6. [Integration](#integration)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)

---

## Overview

The TKS telemetry system provides comprehensive tracking and visualization of:

- **Augmentation metrics**: Counts, ratios, and distributions of augmentation types
- **Validation metrics**: Pass rates, component validity, error tracking
- **Distribution metrics**: World, noetic, operator, and foundation distributions
- **Training metrics**: Loss curves, batch statistics, epoch progress
- **Temporal tracking**: Timestamps, durations, trend analysis

### Key Features

✅ **Lightweight dependencies** - Only matplotlib required for visualization
✅ **Multiple output formats** - JSON, CSV, and JSON arrays for trend tracking
✅ **Auto-wiring** - Integrated into augmentation and training pipelines
✅ **Flexible plotting** - Support for various chart types and configurations
✅ **Append mode** - CSV append for continuous metric tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TKS Telemetry System                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Data Sources    │      │  Metrics Logger  │      │  Persistence     │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│ • Augmentation   │─────▶│ • AugmentationL  │─────▶│ • JSON (single)  │
│   Pipeline       │      │   ogger          │      │ • JSON (array)   │
│ • Training Loop  │      │ • ValidationStats│      │ • CSV (append)   │
│ • Validation     │      │ • DistributionSt │      └──────────────────┘
│   Results        │      │   ats            │               │
└──────────────────┘      │ • AugmentationSt │               │
                          │   ats            │               │
                          └──────────────────┘               ▼
                                                    ┌──────────────────┐
                                                    │  Visualization   │
                                                    ├──────────────────┤
                                                    │ • plot_metrics.py│
                                                    │ • Loss curves    │
                                                    │ • Distributions  │
                                                    │ • Validation     │
                                                    │ • Bar/Pie charts │
                                                    └──────────────────┘
```

### Core Components

1. **`augmentation_metrics.py`** - Core metrics logging module
   - `AugmentationLogger` class
   - `ValidationStats`, `DistributionStats`, `AugmentationStats` dataclasses
   - JSON/CSV persistence methods

2. **`plot_metrics.py`** - Visualization module
   - Plot generation functions
   - Multi-format input support (JSON/CSV)
   - CLI interface

3. **Integration points**:
   - `generate_augmented_data.py` - Augmentation metrics
   - `train_with_augmented.py` - Training metrics
   - Custom scripts via `AugmentationLogger` API

---

## Metrics Collection

### AugmentationLogger API

```python
from augmentation_metrics import AugmentationLogger

# Initialize logger
logger = AugmentationLogger()

# Log individual entry
entry = {
    "id": "entry_001",
    "expr_elements": ["B2", "D5"],
    "expr_ops": ["->"],
    "aug_type": "original",  # or "inversion" or "anti_attractor"
    "validator_pass": True,
    "axes": ["W"],  # for inversions
    "mode": "soft"  # for inversions
}
logger.log_entry(entry)

# Log batch of entries
logger.log_batch(entries)

# Get summary statistics
summary = logger.get_summary()

# Print formatted summary
logger.print_summary(detailed=True)

# Reset metrics
logger.reset()
```

### Tracked Metrics

#### Augmentation Metrics
- **Counts**: Original, inversion, anti-attractor entries
- **Ratios**: Augmentation ratio, inversion ratio, anti-attractor ratio
- **Axes usage**: Distribution of inversion axes (W, N, F, etc.)
- **Modes**: Distribution of inversion modes (soft, hard, targeted)

#### Validation Metrics
- **Overall**: Total, passed, failed, pass rate
- **Component validity**: World, noetic, operator, structural, foundation
- **Error tracking**: Counts and types of validation errors

#### Distribution Metrics
- **Worlds**: Counts and percentages for A, B, C, D
- **Noetics**: Counts and percentages for 1-10
- **Operators**: Usage of TKS operators (+, -, ->, etc.)
- **Foundations**: Distribution of foundation IDs
- **Polarity**: Positive, negative, neutral counts

---

## Persistence Formats

### 1. JSON (Single Object)

**File**: `metrics.json`
**Use**: Single-run snapshot
**Method**: `logger.save(filepath)`

```json
{
  "timestamp": "2025-12-14T10:00:00.000000",
  "duration_seconds": 45.2,
  "augmentation": {
    "original_count": 100,
    "inversion_count": 150,
    "anti_attractor_count": 50,
    "total_count": 300,
    "augmentation_ratio": 2.0,
    "axes_usage": {"W": 75, "N": 75},
    "mode_counts": {"soft": 120, "hard": 30}
  },
  "validation": {
    "total": 300,
    "passed": 270,
    "pass_rate": 0.9,
    "world_validity_rate": 0.95
  },
  "distribution": {
    "world_counts": {"A": 80, "B": 75, "C": 70, "D": 75},
    "noetic_counts": {"1": 30, "2": 35, ...}
  }
}
```

### 2. JSON Array (Trend Tracking)

**File**: `trends.json`
**Use**: Multi-run trend analysis
**Method**: `logger.save_to_json(filepath, append=True)`

```json
[
  {
    "timestamp": "2025-12-14T10:00:00.000000",
    "augmentation": {...},
    "validation": {...}
  },
  {
    "timestamp": "2025-12-14T11:00:00.000000",
    "augmentation": {...},
    "validation": {...}
  }
]
```

### 3. CSV (Tabular)

**File**: `metrics.csv`
**Use**: Time-series analysis, easy plotting
**Method**: `logger.save_to_csv(filepath, append=True)`

```csv
timestamp,duration_seconds,original_count,inversion_count,anti_attractor_count,total_count,augmentation_ratio,validation_total,validation_passed,pass_rate,world_validity_rate,noetic_validity_rate
2025-12-14T10:00:00,45.2,100,150,50,300,2.0,300,270,0.9,0.95,0.92
2025-12-14T11:00:00,52.1,100,150,50,300,2.0,300,275,0.92,0.96,0.93
```

**Features**:
- Automatic header management
- Append mode for continuous tracking
- Compatible with pandas, Excel, plotting tools

---

## Visualization

### plot_metrics.py CLI

```bash
# Generate all plots from JSON
python scripts/plot_metrics.py \
  --input output/metrics.json \
  --output-dir output/plots \
  --plot-type all

# Generate specific plot from CSV
python scripts/plot_metrics.py \
  --input output/metrics.csv \
  --output-dir output/plots \
  --plot-type loss

# Generate with custom prefix
python scripts/plot_metrics.py \
  --input output/metrics.json \
  --output-dir output/plots \
  --plot-type all \
  --prefix experiment1
```

### Plot Types

#### 1. Loss Curve (`--plot-type loss`)
- **File**: `loss_curve.png`
- **Shows**: Training loss over epochs/time
- **X-axis**: Epoch number
- **Y-axis**: Loss value

#### 2. Augmentation Distribution (`--plot-type distribution`)
- **File**: `augmentation_distribution.png`
- **Shows**: Pie chart of augmentation types
- **Segments**: Original, Inversion, Anti-attractor
- **Legend**: Counts and percentages

#### 3. Validation Rates (`--plot-type validation`)
- **File**: `validation_rates.png`
- **Shows**: Multiple validation metrics over time
- **Lines**: Overall pass rate, world validity, noetic validity, operator validity
- **X-axis**: Epoch number
- **Y-axis**: Validation rate (0-1)

#### 4. World/Noetic Distribution (`--plot-type world-noetic`)
- **File**: `world_noetic_distribution.png`
- **Shows**: Two bar charts side-by-side
- **Left**: World distribution (A/B/C/D)
- **Right**: Noetic distribution (1-10)

#### 5. Augmentation Ratios (`--plot-type ratios`)
- **File**: `augmentation_ratios.png`
- **Shows**: Augmentation ratios over time
- **Lines**: Total augmentation, inversion, anti-attractor ratios
- **X-axis**: Epoch number
- **Y-axis**: Ratio (augmented/original)

#### 6. Augmentation Counts Bar Chart (`--plot-type counts-bar`)
- **File**: `augmentation_counts_bar.png`
- **Shows**: Grouped bar chart of augmentation counts by type over epochs
- **Bars**: Original (magenta), Inversion (orange), Anti-Attractor (red)
- **X-axis**: Epoch number
- **Y-axis**: Count
- **Use**: Track how augmentation volumes change across epochs

#### 7. Combined Dashboard (`--plot-type dashboard`)
- **File**: `combined_dashboard.png`
- **Shows**: 2x2 grid combining training and augmentation metrics
- **Panels**:
  - Top-left: Loss vs. Epoch (line chart)
  - Top-right: Validator Pass Rate Over Time (multi-line: overall, world, noetic, operator)
  - Bottom-left: Augmentation Counts by Type (stacked area chart)
  - Bottom-right: Summary Statistics Table (totals, ratios, final metrics)
- **Use**: Single-view overview of training progress, validation quality, and augmentation balance
- **Interpretation**:
  - Loss should decrease over epochs (model learning)
  - Pass rates should increase (better canonical compliance)
  - Augmentation mix shows data balance (original vs. synthetic)
  - Summary table provides quick reference for key metrics

---

## Integration

### 1. Augmentation Pipeline Integration

The `generate_augmented_data.py` script automatically logs metrics:

```python
# In generate_augmented_data.py
from augmentation_metrics import AugmentationLogger

# Initialize logger
logger = AugmentationLogger()

# During augmentation
for entry in entries:
    # ... generate augmentations ...
    logger.log_entry(original_entry)
    logger.log_entry(inverted_entry)
    logger.log_entry(anti_entry)

# Save metrics (auto-wired at lines 887-908)
if config.save_metrics:
    # JSON snapshot
    logger_metrics_path = output_corpus.with_suffix(".detailed_metrics.json")
    logger.save(str(logger_metrics_path))

    # CSV for plotting
    csv_metrics_path = output_corpus.with_suffix(".metrics.csv")
    logger.save_to_csv(str(csv_metrics_path), append=False)

    # Trend tracking
    trend_metrics_path = output_corpus.parent / "augmentation_trends.json"
    logger.save_to_json(str(trend_metrics_path), append=True)

    # Print summary
    logger.print_summary(detailed=True)
```

### 2. Training Pipeline Integration

The `train_with_augmented.py` script tracks training metrics:

```python
# In train_with_augmented.py
from augmentation_metrics import AugmentationLogger, track_epoch_stats

# Per-epoch tracking
epoch_logger = AugmentationLogger()

for epoch in range(num_epochs):
    for batch in batches:
        # ... training step ...
        epoch_logger.log_batch(batch_entries)

    # Save epoch metrics
    epoch_stats = track_epoch_stats(
        epoch=epoch,
        entries=epoch_entries,
        output_dir=metrics_dir
    )

    # Reset for next epoch
    epoch_logger.reset()
```

### 3. Custom Integration

For custom scripts or experiments:

```python
from augmentation_metrics import AugmentationLogger

# Initialize
logger = AugmentationLogger()
logger.store_entries = True  # Optional: store entries for debugging

# During processing
for entry in process_data():
    logger.log_entry({
        "expr_elements": entry.elements,
        "expr_ops": entry.ops,
        "aug_type": entry.type,
        "validator_pass": validate(entry)
    })

# Save results
logger.save("output/experiment_metrics.json")
logger.save_to_csv("output/experiment_metrics.csv")
logger.print_summary(detailed=True)
```

---

## Usage Examples

### Example 1: Basic Augmentation Workflow

```bash
# Step 1: Generate augmented data (metrics auto-saved)
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/augmented.jsonl \
  --axes W N \
  --use-anti-attractor \
  --save-metrics

# Output files:
#   - data/augmented.jsonl
#   - data/augmented.detailed_metrics.json
#   - data/augmented.metrics.csv
#   - data/augmentation_trends.json (appended)

# Step 2: Generate plots
python scripts/plot_metrics.py \
  --input data/augmented.detailed_metrics.json \
  --output-dir output/plots/augmentation \
  --plot-type all

# Output plots:
#   - output/plots/augmentation/loss_curve.png
#   - output/plots/augmentation/augmentation_distribution.png
#   - output/plots/augmentation/validation_rates.png
#   - output/plots/augmentation/world_noetic_distribution.png
#   - output/plots/augmentation/augmentation_ratios.png
```

### Example 2: Training with Metrics

```bash
# Train model (metrics auto-saved)
python scripts/train_with_augmented.py \
  --data data/augmented.jsonl \
  --epochs 10 \
  --batch-size 32 \
  --output-dir output/models

# Output files:
#   - output/models/metrics/training_metrics.json
#   - output/models/metrics/training_metrics_epochs.csv
#   - output/models/metrics/training_metrics_steps.csv
#   - output/models/metrics/epoch_001_metrics.json (per epoch)

# Generate training plots
python scripts/plot_metrics.py \
  --input output/models/metrics/training_metrics_epochs.csv \
  --output-dir output/plots/training \
  --plot-type loss
```

### Example 3: Multi-Run Comparison

```bash
# Run 1
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/run1_augmented.jsonl \
  --axes W N

# Run 2
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/run2_augmented.jsonl \
  --axes W N F

# Run 3
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/run3_augmented.jsonl \
  --axes W N F \
  --use-anti-attractor

# Trend file now contains all 3 runs:
#   - data/augmentation_trends.json

# Plot trends
python scripts/plot_metrics.py \
  --input data/augmentation_trends.json \
  --output-dir output/plots/comparison \
  --plot-type validation
```

### Example 4: CSV-Based Analysis

```python
# Python script for custom analysis
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV metrics
df = pd.read_csv("output/augmented.metrics.csv")

# Compute statistics
print(f"Average pass rate: {df['pass_rate'].mean():.2%}")
print(f"Max augmentation ratio: {df['augmentation_ratio'].max():.2f}x")

# Custom plot
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['pass_rate'], marker='o')
plt.xlabel('Timestamp')
plt.ylabel('Pass Rate')
plt.title('Validation Pass Rate Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/custom_plot.png', dpi=300)
```

---

## Best Practices

### 1. Metric Collection

✅ **DO**: Log all entries during augmentation
✅ **DO**: Use batch logging for performance
✅ **DO**: Reset logger between epochs
❌ **DON'T**: Log partial/incomplete entries
❌ **DON'T**: Skip validation status tracking

### 2. Persistence

✅ **DO**: Use CSV for time-series tracking
✅ **DO**: Use JSON arrays for multi-run trends
✅ **DO**: Enable append mode for continuous monitoring
❌ **DON'T**: Overwrite existing trend files
❌ **DON'T**: Mix formats within same analysis

### 3. Visualization

✅ **DO**: Generate plots after each major run
✅ **DO**: Use consistent output directories
✅ **DO**: Include timestamps in plot filenames
❌ **DON'T**: Generate plots without checking data quality
❌ **DON'T**: Use low DPI for publication plots

### 4. Debugging

✅ **DO**: Set `logger.store_entries = True` for debugging
✅ **DO**: Use `logger.print_summary(detailed=True)` for inspection
✅ **DO**: Check validation error counts
❌ **DON'T**: Store entries in production (memory overhead)
❌ **DON'T**: Ignore validation failures

### 5. Performance

✅ **DO**: Use batch logging when possible
✅ **DO**: Limit CSV appends to end of epoch
✅ **DO**: Generate plots asynchronously if needed
❌ **DON'T**: Log every single training step to CSV
❌ **DON'T**: Generate plots during training loop

---

## File Structure

```
output/
├── augmented.jsonl                    # Augmented data
├── augmented.detailed_metrics.json    # Full metrics (single run)
├── augmented.metrics.csv              # CSV metrics (time-series)
├── augmentation_trends.json           # Multi-run trends (JSON array)
│
├── models/
│   └── metrics/
│       ├── training_metrics.json      # Training summary
│       ├── training_metrics_epochs.csv
│       ├── training_metrics_steps.csv
│       ├── epoch_001_metrics.json     # Per-epoch details
│       ├── epoch_002_metrics.json
│       └── epoch_003_metrics.json
│
└── plots/
    ├── augmentation/
    │   ├── loss_curve.png
    │   ├── augmentation_distribution.png
    │   ├── validation_rates.png
    │   ├── world_noetic_distribution.png
    │   ├── augmentation_ratios.png
    │   ├── augmentation_counts_bar.png
    │   └── combined_dashboard.png
    │
    └── training/
        ├── loss_curve.png
        ├── validation_rates.png
        ├── augmentation_ratios.png
        └── combined_dashboard.png
```

---

## Interpreting the Plots

### Loss Curve Interpretation
- **Decreasing trend**: Model is learning (good)
- **Flat trend**: Learning has plateaued; consider adjusting learning rate
- **Oscillating**: Learning rate may be too high or data is noisy
- **Increasing**: Model is diverging; reduce learning rate or check data quality

### Validation Pass Rate Interpretation
- **Overall Pass Rate > 90%**: Excellent canonical compliance
- **Overall Pass Rate 70-90%**: Good compliance, room for improvement
- **Overall Pass Rate < 70%**: Review augmentation quality and validator rules
- **Component rates (world/noetic/operator)**: Help diagnose specific issues
  - Low world validity: Check world labels (A/B/C/D)
  - Low noetic validity: Check noetic indices (1-10)
  - Low operator validity: Check ALLOWED_OPS compliance

### Augmentation Distribution Interpretation
- **Balanced distribution**: Original, Inversion, Anti-Attractor roughly equal
- **Heavy original bias**: May need more augmentation
- **Heavy augmentation bias**: May be over-augmenting; check quality
- **Recommended ratio**: 1.0-2.0x augmentation (synthetic:original)

### Combined Dashboard Quick Reference
The dashboard provides at-a-glance status:
1. **Loss panel**: Should trend downward
2. **Validation panel**: All lines should trend upward toward 1.0
3. **Augmentation panel**: Stacked area shows data balance
4. **Summary table**: Key metrics for quick assessment

---

## Troubleshooting

### Issue: "matplotlib is required but not installed"

**Solution**:
```bash
pip install matplotlib
```

### Issue: "CSV file not created"

**Cause**: File path doesn't exist or permissions issue
**Solution**:
```python
# Check parent directory exists
from pathlib import Path
csv_path = Path("output/metrics.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)
logger.save_to_csv(str(csv_path))
```

### Issue: "No loss data found in metrics"

**Cause**: Metrics file doesn't contain `loss` or `pass_rate` fields
**Solution**: Use validation pass rate as proxy:
```python
# In plot_metrics.py, this is automatic:
# If 'loss' not found, uses 1.0 - pass_rate
```

### Issue: "Plots are blurry"

**Cause**: Default DPI is 300
**Solution**: Increase DPI in `plot_metrics.py`:
```python
plt.savefig(output_path, dpi=600, bbox_inches='tight')
```

---

## API Reference

### AugmentationLogger

```python
class AugmentationLogger:
    def __init__(self):
        """Initialize logger with empty metrics."""

    def log_entry(self, entry: Dict[str, Any]) -> None:
        """Log a single entry."""

    def log_batch(self, entries: List[Dict[str, Any]]) -> None:
        """Log a batch of entries."""

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""

    def print_summary(self, detailed: bool = False) -> None:
        """Print formatted summary to console."""

    def save(self, filepath: str) -> None:
        """Save metrics to JSON file (single object)."""

    def save_to_json(self, filepath: str, append: bool = False) -> None:
        """Save to JSON with optional append (JSON array)."""

    def save_to_csv(self, filepath: str, append: bool = True) -> None:
        """Save to CSV with optional append."""

    def reset(self) -> None:
        """Reset all metrics."""
```

### Helper Functions

```python
def compute_batch_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics for a batch of entries."""

def track_epoch_stats(
    epoch: int,
    entries: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Track and optionally save epoch statistics."""

def compare_metrics(
    baseline_summary: Dict[str, Any],
    augmented_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare baseline and augmented metrics."""
```

---

## Changelog

### Version 1.0.0 (2025-12-14)
- Initial release
- CSV persistence with append mode
- JSON array persistence for trend tracking
- Complete plotting suite (5 plot types)
- Integration with augmentation and training pipelines
- Comprehensive test suite

---

## Future Enhancements

### Planned Features
- [ ] Real-time plotting during training
- [ ] Web-based dashboard (optional)
- [ ] Metric comparison UI
- [ ] Anomaly detection in validation rates
- [ ] Automated report generation
- [ ] Integration with tensorboard/wandb (optional)

### Community Contributions
Contributions welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## Support

For issues, questions, or feature requests:
- **Documentation**: See [docs/](../docs/)
- **Examples**: See [examples/](../examples/)
- **Tests**: Run `python scripts/test_telemetry.py`

---

**End of TKS Telemetry & Monitoring Guide**

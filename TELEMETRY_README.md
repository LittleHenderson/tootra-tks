# TKS Telemetry & Monitoring System

**Complete metrics tracking and visualization for TKS augmentation and training**

[![Status](https://img.shields.io/badge/status-production-green.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencies-matplotlib-orange.svg)]()

---

## Overview

The TKS Telemetry System provides comprehensive tracking and visualization of augmentation and training metrics with minimal dependencies. It's fully integrated into the TKS pipeline and supports multiple output formats for flexible analysis.

### Key Features

✅ **Automatic Integration** - Wired into `generate_augmented_data.py` and `train_with_augmented.py`
✅ **Multiple Formats** - JSON, CSV, and JSON arrays for different use cases
✅ **Rich Visualizations** - 5+ plot types via matplotlib
✅ **Trend Tracking** - CSV append mode for continuous monitoring
✅ **Minimal Dependencies** - Only matplotlib required
✅ **Comprehensive Metrics** - Augmentation, validation, and distribution stats
✅ **Production Ready** - Tested and documented

---

## Quick Start

### Installation

```bash
pip install matplotlib
```

### Generate Augmented Data with Metrics

```bash
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/augmented.jsonl \
  --axes W N \
  --use-anti-attractor \
  --save-metrics
```

**Outputs**:
- `data/augmented.jsonl` - Augmented training data
- `data/augmented.detailed_metrics.json` - Full metrics snapshot
- `data/augmented.metrics.csv` - CSV format for plotting
- `data/augmentation_trends.json` - Multi-run trend tracking

### Generate Plots

```bash
python scripts/plot_metrics.py \
  --input data/augmented.detailed_metrics.json \
  --output-dir output/plots \
  --plot-type all
```

**Outputs**:
- `loss_curve.png` - Training loss over time
- `augmentation_distribution.png` - Pie chart of augmentation types
- `validation_rates.png` - Validation metrics over epochs
- `world_noetic_distribution.png` - World/noetic bar charts
- `augmentation_ratios.png` - Augmentation ratios over time

### Train with Metrics

```bash
python scripts/train_with_augmented.py \
  --data data/augmented.jsonl \
  --epochs 10 \
  --batch-size 32 \
  --output-dir output/models
```

**Outputs**:
- `output/models/metrics/training_metrics.json` - Training summary
- `output/models/metrics/training_metrics_epochs.csv` - Per-epoch CSV
- `output/models/metrics/epoch_NNN_metrics.json` - Per-epoch details

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  TKS Telemetry Pipeline                    │
└────────────────────────────────────────────────────────────┘

Data Source                Metrics Logger              Persistence
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ Augmentation │──────▶   │ Augmentation │──────▶   │ JSON Single  │
│   Pipeline   │          │    Logger    │          │ JSON Array   │
└──────────────┘          │              │          │ CSV Append   │
                          │ • Validation │          └──────────────┘
┌──────────────┐          │ • Augment    │                  │
│   Training   │──────▶   │ • Distribut  │                  │
│     Loop     │          │              │                  ▼
└──────────────┘          └──────────────┘          ┌──────────────┐
                                                    │ Visualization│
                                                    │ • Loss       │
                                                    │ • Validation │
                                                    │ • Distribut  │
                                                    └──────────────┘
```

---

## Components

### 1. Core Metrics Module

**File**: `scripts/augmentation_metrics.py`

- `AugmentationLogger` - Main logging class
- `ValidationStats` - Validation metrics
- `DistributionStats` - Distribution tracking
- `AugmentationStats` - Augmentation metrics

### 2. Visualization Module

**File**: `scripts/plot_metrics.py`

- `plot_loss_curve()` - Loss over time
- `plot_augmentation_distribution()` - Pie chart
- `plot_validation_rates()` - Validation metrics
- `plot_world_noetic_distribution()` - Bar charts
- `plot_augmentation_ratios()` - Ratios over time

### 3. Integration Points

**Augmentation**: `scripts/generate_augmented_data.py` (lines 887-908)
**Training**: `scripts/train_with_augmented.py` (lines 966-1088)

---

## Metrics Tracked

### Augmentation Metrics
- **Counts**: Original, inversion, anti-attractor entries
- **Ratios**: Augmentation ratio, inversion ratio, anti-attractor ratio
- **Axes**: Distribution of inversion axes (W, N, F, S, A, P, E)
- **Modes**: Distribution of inversion modes (soft, hard, targeted)

### Validation Metrics
- **Overall**: Total validated, passed, failed, pass rate
- **Component Validity**: World, noetic, operator, structural, foundation validity rates
- **Error Tracking**: Counts and types of validation errors

### Distribution Metrics
- **Worlds**: Counts and percentages for A, B, C, D
- **Noetics**: Counts and percentages for 1-10
- **Operators**: Usage of TKS operators (+, -, ->, +T, -T, etc.)
- **Foundations**: Distribution of foundation IDs (1-7)

---

## File Formats

### JSON (Single Object)
```json
{
  "timestamp": "2025-12-14T10:00:00.000000",
  "duration_seconds": 45.2,
  "augmentation": {
    "original_count": 100,
    "inversion_count": 150,
    "anti_attractor_count": 50,
    "augmentation_ratio": 2.0
  },
  "validation": {
    "total": 300,
    "passed": 270,
    "pass_rate": 0.9
  }
}
```

### CSV (Time-Series)
```csv
timestamp,original_count,inversion_count,pass_rate
2025-12-14T10:00:00,100,150,0.90
2025-12-14T11:00:00,100,150,0.92
```

### JSON Array (Trends)
```json
[
  {"timestamp": "2025-12-14T10:00:00", ...},
  {"timestamp": "2025-12-14T11:00:00", ...}
]
```

---

## Usage Examples

### Example 1: Basic Workflow

```bash
# Generate augmented data
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/augmented.jsonl \
  --save-metrics

# Generate plots
python scripts/plot_metrics.py \
  --input data/augmented.detailed_metrics.json \
  --output-dir output/plots \
  --plot-type all
```

### Example 2: Training Workflow

```bash
# Train model
python scripts/train_with_augmented.py \
  --data data/augmented.jsonl \
  --epochs 10 \
  --output-dir output/models

# Plot training progress
python scripts/plot_metrics.py \
  --input output/models/metrics/training_metrics_epochs.csv \
  --output-dir output/plots/training \
  --plot-type loss
```

### Example 3: Custom Metrics (Python)

```python
from augmentation_metrics import AugmentationLogger

# Initialize logger
logger = AugmentationLogger()

# Log entries
for entry in your_data:
    logger.log_entry({
        "expr_elements": ["B2", "D5"],
        "expr_ops": ["->"],
        "aug_type": "original",
        "validator_pass": True
    })

# Save and display
logger.save("output/metrics.json")
logger.save_to_csv("output/metrics.csv")
logger.print_summary(detailed=True)
```

---

## Testing

### Run Comprehensive Test Suite

```bash
python scripts/test_telemetry.py
```

**Tests**:
1. Metrics persistence (JSON/CSV)
2. CSV append mode
3. Multi-epoch tracking
4. Plotting integration
5. End-to-end validation

**Output**: `output/telemetry_test/`

---

## Documentation

| Document | Description |
|----------|-------------|
| [TELEMETRY_GUIDE.md](docs/TELEMETRY_GUIDE.md) | Complete guide (35+ pages) |
| [TELEMETRY_QUICKSTART.md](docs/TELEMETRY_QUICKSTART.md) | 5-minute quick start |
| This file | Overview and reference |

---

## API Reference

### AugmentationLogger Class

```python
class AugmentationLogger:
    def __init__(self):
        """Initialize logger with empty metrics."""

    def log_entry(self, entry: Dict[str, Any]) -> None:
        """Log a single entry with augmentation/validation info."""

    def log_batch(self, entries: List[Dict[str, Any]]) -> None:
        """Log a batch of entries."""

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics as dictionary."""

    def print_summary(self, detailed: bool = False) -> None:
        """Print formatted summary to console."""

    def save(self, filepath: str) -> None:
        """Save to JSON file (single object)."""

    def save_to_json(self, filepath: str, append: bool = False) -> None:
        """Save to JSON with optional append (creates array)."""

    def save_to_csv(self, filepath: str, append: bool = True) -> None:
        """Save to CSV with optional append (default: True)."""

    def reset(self) -> None:
        """Reset all metrics to zero."""
```

### Helper Functions

```python
def compute_batch_stats(entries: List[Dict]) -> Dict[str, Any]:
    """Compute statistics for a batch of entries."""

def track_epoch_stats(
    epoch: int,
    entries: List[Dict],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Track and optionally save epoch statistics."""

def compare_metrics(
    baseline_summary: Dict,
    augmented_summary: Dict
) -> Dict[str, Any]:
    """Compare baseline and augmented metrics."""
```

---

## Plot Gallery

### Loss Curve
![Loss Curve](docs/images/loss_curve_example.png)
*Training loss over epochs*

### Augmentation Distribution
![Augmentation Distribution](docs/images/aug_dist_example.png)
*Pie chart showing augmentation types*

### Validation Rates
![Validation Rates](docs/images/validation_rates_example.png)
*Multiple validation metrics over time*

### World/Noetic Distribution
![World/Noetic Distribution](docs/images/world_noetic_example.png)
*Bar charts for world and noetic distributions*

---

## Troubleshooting

### Issue: "matplotlib is required but not installed"
```bash
pip install matplotlib
```

### Issue: CSV file not created
```python
# Ensure parent directory exists
from pathlib import Path
csv_path = Path("output/metrics.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)
```

### Issue: No loss data found
The system automatically uses validation pass rate as a proxy if no explicit loss field is found.

---

## Performance

- **Logging overhead**: < 1ms per entry
- **CSV append**: O(1) operation
- **Plot generation**: ~2-5 seconds for all plots
- **Memory**: ~1MB per 10,000 entries (without entry storage)

---

## Dependencies

- **Required**: matplotlib (for plotting)
- **Optional**: pandas (for advanced CSV analysis)
- **Built-in**: json, csv, pathlib, datetime, collections

---

## Changelog

### Version 1.0.0 (2025-12-14)
- ✅ Initial release
- ✅ CSV persistence with append mode
- ✅ JSON array persistence for trend tracking
- ✅ 5 plot types (loss, distribution, validation, world-noetic, ratios)
- ✅ Full integration with augmentation pipeline
- ✅ Full integration with training pipeline
- ✅ Comprehensive test suite
- ✅ Complete documentation

---

## Future Enhancements

- [ ] Real-time plotting during training
- [ ] Web-based dashboard (optional)
- [ ] Integration with tensorboard/wandb (optional)
- [ ] Anomaly detection
- [ ] Automated report generation

---

## Contributing

Contributions welcome! Please:
1. Read [TELEMETRY_GUIDE.md](docs/TELEMETRY_GUIDE.md)
2. Run tests: `python scripts/test_telemetry.py`
3. Follow existing code style
4. Update documentation

---

## License

Part of the TKS-LLM Training Integration System.

---

## Support

- **Documentation**: [docs/TELEMETRY_GUIDE.md](docs/TELEMETRY_GUIDE.md)
- **Quick Start**: [docs/TELEMETRY_QUICKSTART.md](docs/TELEMETRY_QUICKSTART.md)
- **Tests**: `python scripts/test_telemetry.py`

---

**TKS Telemetry System v1.0.0** - Production Ready ✨

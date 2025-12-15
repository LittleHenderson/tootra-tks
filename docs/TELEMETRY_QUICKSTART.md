# TKS Telemetry Quick Start Guide

**5-minute guide to get started with TKS metrics and plotting**

---

## Installation

```bash
# Only dependency: matplotlib
pip install matplotlib
```

---

## Quick Examples

### 1. Generate Augmented Data with Metrics

```bash
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/augmented.jsonl \
  --axes W N \
  --use-anti-attractor \
  --save-metrics
```

**Output**:
- `data/augmented.jsonl` - Augmented data
- `data/augmented.detailed_metrics.json` - Full metrics
- `data/augmented.metrics.csv` - CSV format
- `data/augmentation_trends.json` - Trend tracking

---

### 2. Generate Plots

```bash
# All plots from JSON
python scripts/plot_metrics.py \
  --input data/augmented.detailed_metrics.json \
  --output-dir output/plots \
  --plot-type all

# Specific plot from CSV
python scripts/plot_metrics.py \
  --input data/augmented.metrics.csv \
  --output-dir output/plots \
  --plot-type validation
```

**Output Plots**:
- `loss_curve.png` - Training loss over time
- `augmentation_distribution.png` - Pie chart of aug types
- `validation_rates.png` - Validation metrics over time
- `world_noetic_distribution.png` - Distribution bar charts
- `augmentation_ratios.png` - Augmentation ratios over time

---

### 3. Train with Metrics

```bash
python scripts/train_with_augmented.py \
  --data data/augmented.jsonl \
  --epochs 10 \
  --batch-size 32 \
  --output-dir output/models
```

**Output**:
- `output/models/metrics/training_metrics.json` - Training summary
- `output/models/metrics/training_metrics_epochs.csv` - Per-epoch CSV
- `output/models/metrics/epoch_NNN_metrics.json` - Per-epoch details

---

### 4. Custom Logging (Python)

```python
from augmentation_metrics import AugmentationLogger

# Initialize
logger = AugmentationLogger()

# Log entries
for entry in your_data:
    logger.log_entry({
        "expr_elements": ["B2", "D5"],
        "expr_ops": ["->"],
        "aug_type": "original",
        "validator_pass": True
    })

# Save
logger.save("output/metrics.json")
logger.save_to_csv("output/metrics.csv")
logger.print_summary(detailed=True)
```

---

## Plot Types

| Flag | Description | Output File |
|------|-------------|-------------|
| `loss` | Loss curve over epochs | `loss_curve.png` |
| `distribution` | Augmentation type pie chart | `augmentation_distribution.png` |
| `validation` | Validation rates over time | `validation_rates.png` |
| `world-noetic` | World/noetic bar charts | `world_noetic_distribution.png` |
| `ratios` | Augmentation ratios | `augmentation_ratios.png` |
| `all` | All of the above | All files |

---

## File Formats

### JSON (Single Object)
```json
{
  "timestamp": "2025-12-14T10:00:00",
  "augmentation": {...},
  "validation": {...},
  "distribution": {...}
}
```

### JSON Array (Trends)
```json
[
  {"timestamp": "...", "augmentation": {...}},
  {"timestamp": "...", "augmentation": {...}}
]
```

### CSV (Time-Series)
```csv
timestamp,original_count,inversion_count,pass_rate,...
2025-12-14T10:00:00,100,150,0.90,...
2025-12-14T11:00:00,100,150,0.92,...
```

---

## Common Commands

```bash
# Test telemetry system
python scripts/test_telemetry.py

# Generate all plots from JSON
python scripts/plot_metrics.py --input metrics.json --output-dir plots --plot-type all

# Generate loss plot from CSV
python scripts/plot_metrics.py --input metrics.csv --output-dir plots --plot-type loss

# Generate plots with prefix
python scripts/plot_metrics.py --input metrics.json --output-dir plots --plot-type all --prefix exp1
```

---

## Metrics Tracked

### Augmentation
- Original count, inversion count, anti-attractor count
- Augmentation ratio (augmented/original)
- Axes usage (W, N, F, etc.)
- Mode distribution (soft, hard, targeted)

### Validation
- Total validated, passed, failed
- Pass rate (%)
- World/noetic/operator/structural validity rates
- Error counts by type

### Distribution
- World counts (A, B, C, D)
- Noetic counts (1-10)
- Operator usage (+, -, ->, etc.)
- Foundation distribution

---

## Troubleshooting

**Issue**: "matplotlib not installed"
```bash
pip install matplotlib
```

**Issue**: "File not found"
```bash
# Check file path and create parent directories
mkdir -p output/plots
```

**Issue**: "No plots generated"
```bash
# Check for errors in output
python scripts/plot_metrics.py --input metrics.json --output-dir plots --plot-type all
# Look for error messages
```

---

## Next Steps

📖 Read full guide: [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md)
🧪 Run tests: `python scripts/test_telemetry.py`
📊 Explore examples: Check `output/plots/` after running

---

**Quick Start Complete!** 🎉

# TKS Data Augmentation Pipeline - README

## Overview

This directory contains the complete TKS data augmentation pipeline for generating training data for TKS-LLM fine-tuning. The pipeline applies semantic transformations to TKS scenarios while maintaining strict canonical compliance.

## Documentation Files

### 📘 [AUGMENTATION_PIPELINE_SPEC.md](AUGMENTATION_PIPELINE_SPEC.md)
**Complete design specification** - Start here for comprehensive understanding
- Detailed input/output formats
- Step-by-step processing pipeline (8 steps)
- Default configurations and guardrails
- Canonical constraints reference
- Error handling strategies
- Usage examples for all scenarios
- Implementation roadmap (Phase 1/2/3)

### 📋 [AUGMENTATION_QUICK_REFERENCE.md](AUGMENTATION_QUICK_REFERENCE.md)
**Quick reference guide** - Use for day-to-day operations
- At-a-glance format summaries
- Common usage patterns
- CLI flags reference table
- Metadata fields explanation
- Error handling quick lookup

### 📊 [AUGMENTATION_PIPELINE_DIAGRAM.txt](AUGMENTATION_PIPELINE_DIAGRAM.txt)
**Visual pipeline flow** - Use for understanding data flow
- ASCII diagram of complete pipeline
- Augmentation ratio breakdowns
- Validation flow (strict vs lenient)
- Error recovery strategy
- Typical augmentation ratios

### 🔧 [generate_augmented_data.py](generate_augmented_data.py)
**Implementation script** - The executable pipeline
- Full implementation with real TKS modules
- CLI interface with argparse
- JSONL I/O handling
- Comprehensive metrics tracking

## Quick Start

### Basic Usage (Default Settings)

Generate augmented data with 3 inversions per scenario:

```bash
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented.jsonl
```

**Generates:**
- Original scenarios (unchanged)
- World inversions (W)
- Noetic inversions (N)
- Combined inversions (W+N)
- Metrics JSON

### Full Augmentation

Include anti-attractors for maximum diversity:

```bash
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented_full.jsonl \
    --axes W N F \
    --use-anti-attractor
```

**Generates:**
- 3 inversions (W, N, F individually)
- 1 anti-attractor counter-scenario
- 4x augmentation ratio

## Input Format

Your input JSONL file should have one JSON object per line:

```json
{"id": "story_001", "story": "A spiritual teacher causes mental growth in students"}
{"id": "eq_001", "equation": "A5 -> B2 +T D6"}
```

**Required:** Either `story` (natural language) OR `equation` (TKS notation)
**Optional:** `id` (auto-generated if missing), `metadata` (preserved in output)

## Output Format

The output JSONL file contains original + augmented scenarios:

```json
{
  "id": "story_001_inv_WN_001",
  "story": "A physical student prevents emotional stagnation in teachers",
  "equation": "D6 <- C3 -T A5",
  "aug_type": "inversion",
  "source_id": "story_001",
  "validator_pass": true,
  "transform_metadata": {
    "axes": ["W", "N"],
    "mode": "soft",
    "explanation": "World + Noetic inversion"
  }
}
```

Plus a metrics JSON file: `<output>.metrics.json`

## Canonical Constraints

The pipeline enforces strict TKS canonical compliance:

| Constraint | Valid Values | Count |
|------------|--------------|-------|
| **Worlds** | A, B, C, D | 4 |
| **Noetics** | 1-10 | 10 |
| **Foundations** | 1-7 | 7 |
| **Operators** | +T, -T, *T, /T, o, ->, <-, +, - | 9 |

**Any violation triggers validation failure** (skip in strict mode, mark in lenient mode)

## Common Workflows

### 1. Production Training Data

Generate high-quality augmented corpus for fine-tuning:

```bash
python scripts/generate_augmented_data.py \
    --input data/training/validated_corpus.jsonl \
    --output data/training/final_augmented.jsonl \
    --axes W N \
    --mode soft \
    --min-pass-rate 0.95 \
    --verbose
```

### 2. Exploratory Development

Develop corpus with lenient validation:

```bash
python scripts/generate_augmented_data.py \
    --input data/experimental/raw_stories.jsonl \
    --output data/experimental/augmented.jsonl \
    --lenient \
    --use-anti-attractor
```

### 3. Custom Axes Combinations

Generate specific axis inversions:

```bash
python scripts/generate_augmented_data.py \
    --input data/pilot/stories.jsonl \
    --output data/pilot/augmented_custom.jsonl \
    --axes W N E F
```

## Dependencies

### Required TKS Modules
- `scenario_inversion.py` - InvertStory, EncodeStory, DecodeStory
- `anti_attractor.py` - AntiAttractorInvert, compute_attractor_signature
- `narrative/encoder.py` - Story encoding with lexicon
- `narrative/constants.py` - Canonical mappings and validators

### Python Standard Library
- `argparse`, `json`, `pathlib`, `dataclasses`, `datetime`

## Implementation Status

### ✅ Phase 1: Complete (Current)
- Design specification
- Data structures
- Function signatures
- CLI interface
- Documentation

### 🚧 Phase 2: Ready for Implementation
- Core pipeline implementation
- Real API integration
- Validation logic
- Metrics computation

### 🔮 Phase 3: Future Enhancements
- Parallel processing
- Progress bars
- Checkpoint/resume
- Deduplication
- Quality scoring

## Error Handling

The pipeline is designed for robustness:

- **Malformed input**: Skip with warning
- **Encoding failures**: Skip scenario, log error
- **Inversion failures**: Skip that axis, continue with others
- **Validation failures**: Skip (strict) or mark (lenient)
- **Zero augmentations**: Include original only

**Result: Pipeline never halts on errors, always produces maximum valid output**

## Metrics Tracking

Every run generates comprehensive metrics:

```json
{
  "original_count": 100,
  "inverted_count": 300,
  "anti_attractor_count": 100,
  "augmentation_ratio": 4.0,
  "validator_pass_rate": 0.99,
  "world_validity": 1.0,
  "noetic_validity": 0.98,
  "duration_seconds": 342.5
}
```

Use these metrics to:
- Verify augmentation quality
- Track validation pass rates
- Optimize configuration
- Report corpus statistics

## Typical Augmentation Ratios

| Configuration | Ratio | Description |
|---------------|-------|-------------|
| Default (W, N, W+N) | 3.0x | Balanced coverage |
| + Anti-attractor | 4.0x | Recommended for training |
| All axes (W, N, E, F) | 4.0x | Comprehensive |
| Full (all + anti) | 5.0x | Maximum diversity |

**Recommendation**: Start with 3.0x-4.0x for balanced training data

## Getting Help

1. **Quick lookup**: See [AUGMENTATION_QUICK_REFERENCE.md](AUGMENTATION_QUICK_REFERENCE.md)
2. **Detailed spec**: See [AUGMENTATION_PIPELINE_SPEC.md](AUGMENTATION_PIPELINE_SPEC.md)
3. **Visual guide**: See [AUGMENTATION_PIPELINE_DIAGRAM.txt](AUGMENTATION_PIPELINE_DIAGRAM.txt)
4. **CLI help**: Run `python scripts/generate_augmented_data.py --help`

## Examples

See the full specification for detailed examples including:
- Basic augmentation
- Full augmentation with anti-attractors
- Custom axes combinations
- Lenient mode for exploration
- Production mode with strict validation

## Author & Version

- **Author**: TKS-LLM Training Integration Team
- **Date**: 2025-12-14
- **Version**: 1.0.0
- **Status**: Specification Complete - Ready for Phase 2 Implementation

## License

Part of the TKS (TOOTRA Kabbalistic System) project.

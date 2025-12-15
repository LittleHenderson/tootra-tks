# TKS (TOOTRA Knowledge System)

![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yaml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A587%25-brightgreen)
![Release](https://img.shields.io/badge/release-v0.2.4-brightgreen)
[![Changelog](https://img.shields.io/badge/changelog-CHANGELOG.md-blue)](CHANGELOG.md)
[![Release Notes](https://img.shields.io/badge/release_notes-RELEASE.md-blue)](RELEASE.md)

A scenario inversion and analysis toolkit for the TOOTRA Knowledge System (TKS), enabling transformations across multiple dimensions of narrative structure.

## Quickstart

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd everthing-tootra-tks

# Install dependencies (if any)
pip install -r requirements.txt  # Python dependencies
```

### Basic Usage

**Note:** Strict mode is now the default. Use `--lenient` for permissive mode that allows unknown tokens with warnings.

```bash
# Story inversion across Element axis (soft mode)
python scripts/run_scenario_inversion.py --story "A woman loved a man" --axes E --mode soft

# Equation inversion across Noetic and World axes (hard mode)
python scripts/run_scenario_inversion.py --equation "B5 +T D3" --axes N,W --mode hard

# Anti-attractor synthesis (generates counter-scenario)
python scripts/run_scenario_inversion.py --equation "C3 -> D5" --anti-attractor
```

### Axes Overview

- **N** = Noetic (involution pairs: 2<->3, 5<->6, 8<->9)
- **E** = Element (full element inversion: world + noetic)
- **W** = World (world mirror: A<->D, B<->C)
- **F** = Foundation (1<->7, 2<->6, 3<->5, 4 self-dual)
- **S** = SubFoundation (foundation + world compound)
- **A** = Acquisition (negation toggle)
- **P** = Polarity (valence flip)

### Running Tests

```bash
# Run all tests
python -m pytest tests -v

# Run specific test suite
python -m pytest tests/test_scenario_inversion_cli.py -v
```

## Recipes & Workflows

Complete end-to-end workflow recipes for common use cases - see **[docs/RECIPES.md](docs/RECIPES.md)** for detailed step-by-step guides:

1. **Complete Training Pipeline** - Augment data, train model, evaluate, and run inference
2. **Anti-Attractor Counter-Scenario Generation** - Generate narrative oppositions for balanced training
3. **Bulk Inference Processing** - Process large batches of scenarios via JSONL input/output

Quick example workflows:
```bash
# Full pipeline: augment -> train -> evaluate -> infer
python scripts/generate_augmented_data.py --input data/stories.jsonl --output data/augmented.jsonl --axes W N --use-anti-attractor
python scripts/train_with_augmented.py --data data/augmented.jsonl --epochs 10 --filter-validated
python scripts/evaluate_model.py --checkpoint output/models/best_model.pt --data data/augmented.jsonl
python scripts/run_inference.py --story "New scenario" --axes W,N --format json

# Bulk inference on JSONL files
python scripts/run_inference.py --input-jsonl input.jsonl --output-jsonl results.jsonl --verbose
```

## HTTP API Server

Serve TKS inference via HTTP REST API:

```bash
# Start server on default port (8000)
python scripts/serve_inference.py

# Start server on custom port
python scripts/serve_inference.py --port 8080

# Start server in lenient mode
python scripts/serve_inference.py --lenient

# Test endpoints with curl
curl http://localhost:8000/health
curl -X POST http://localhost:8000/encode -H "Content-Type: application/json" -d '{"story": "A woman loved a man"}'
curl -X POST http://localhost:8000/invert -H "Content-Type: application/json" -d '{"story": "A woman loved a man", "axes": ["W", "N"]}'
curl -X POST http://localhost:8000/anti-attractor -H "Content-Type: application/json" -d '{"story": "Power corrupts"}'
```

**Available Endpoints:**
- `GET /health` - Server health check with canonical configuration
- `POST /encode` - Encode natural language story to TKS expression
- `POST /invert` - Perform scenario inversion with validator flags
- `POST /anti-attractor` - Anti-attractor synthesis with signature analysis

See **[docs/INFERENCE_API.md](docs/INFERENCE_API.md)** for complete API documentation, examples, and integration guides.

## Full Documentation

See `docs/SCENARIO_INVERSION.md` for:
- Complete CLI reference (`scripts/run_scenario_inversion.py`)
- Detailed axis and mode explanations
- Anti-attractor synthesis (`--anti-attractor` flag) for generating counter-scenarios
- Advanced examples for story and equation inputs
- Canon guardrails (worlds A/B/C/D; fixed noetics; fixed foundations; involution pairs)

### Operators
- Allowed operators: `+`, `-`, `+T`, `-T`, `->`, `<-`, `*T` (intensify), `/T` (conflict), `o` (sequence)
- Strict validation rejects unknown operators; templates are defined for all allowed ops
- **Strict mode is the default**: Unknown tokens/operators are rejected with helpful error messages
- Use `--lenient` to allow unknown tokens with warnings (permissive mode)

## Data Augmentation & Training

Generate augmented training data with inversions and anti-attractors:
```bash
python scripts/generate_augmented_data.py \
  --input data/stories.jsonl \
  --output data/augmented.jsonl \
  --axes W,N,F \
  --use-anti-attractor \
  --validate
```

Train models using augmented data:
```bash
python scripts/train_with_augmented.py \
  --data data/augmented.jsonl \
  --epochs 10 \
  --batch-size 16 \
  --filter-validated \
  --output-dir output/models
```

Evaluate trained models:
```bash
python scripts/evaluate_model.py \
  --checkpoint output/models/best_model.pt \
  --data data/augmented.jsonl \
  --output eval_report.json
```

Canonical validation: `scripts/canonical_validator.py` enforces worlds A/B/C/D, noetics 1-10, foundations 1-7, ops in {+, -, +T, -T, ->, <-, *T, /T, o}.

## Inference & Bulk Processing

Run inference on single scenarios or process large batches via JSONL:

```bash
# Single scenario inference
python scripts/run_inference.py \
  --story "A teacher inspires a student" \
  --axes W,N \
  --format json

# Bulk inference (JSONL input/output)
python scripts/run_inference.py \
  --input-jsonl input_scenarios.jsonl \
  --output-jsonl results.jsonl \
  --axes W,N \
  --verbose

# Anti-attractor bulk processing
python scripts/run_inference.py \
  --input-jsonl stories.jsonl \
  --output-jsonl anti_attractors.jsonl \
  --anti-attractor \
  --verbose
```

**JSONL Input Format:**
```jsonl
{"story": "A teacher inspires a student"}
{"equation": "B5 -> D3"}
{"story": "Custom config", "axes": "F", "mode": "targeted"}
```

**JSONL Output Format:**
```jsonl
{
  "success": true,
  "inversion_type": "inversion",
  "result": {
    "original": {"expression": "B5 -> D3", "story": "..."},
    "inverted": {"expression": "C6 -> A2", "story": "..."}
  },
  "validator": {"is_valid": true, "canon_score": 1.0}
}
```

## Releases and Versioning

- **Changelog**: See [CHANGELOG.md](CHANGELOG.md) for complete version history
- **Release Notes**: See [RELEASE.md](RELEASE.md) for current release highlights
- **GitHub Releases**: Each tagged release (v*) includes CHANGELOG.md and RELEASE.md as downloadable assets

### Generating Release Notes

```bash
# Extract notes for a specific version
python scripts/generate_release_notes.py --version 0.2.2

# Extract notes for the latest version
python scripts/generate_release_notes.py --latest

# Save to file
python scripts/generate_release_notes.py --latest --output release_notes.md
```

## CI/CD

- **CI Pipeline**: Runs on push/PR to main/master branches
  - Python 3.10 and 3.11 matrix testing
  - Coverage threshold: 87% (enforced; core modules 75-94%)
  - Fuzz testing with 95% pass rate threshold (warning below)
- **Release Pipeline**: Triggers on v* tags
  - Full test suite execution
  - Automatic GitHub release creation with changelog attachments

## Canon Guardrails

| Property | Constraint |
|----------|------------|
| Worlds | A, B, C, D only |
| Noetics | 1-10 (pairs 2<->3, 5<->6, 8<->9; self-duals 1,4,7,10) |
| Foundations | 1-7 only |
| Sub-foundations | 7x4=28 combinations |
| Operators | +, -, +T, -T, ->, <-, *T, /T, o (9 total) |

# Changelog

All notable changes to the TKS (TOOTRA Knowledge System) project will be documented in this file.

## [Unreleased]

### Added - Parser Extension (Task A)
- **Extended Parser Syntax Support**
  - Caret sense notation: `B8^5` (canonical form)
  - Foundation suffix notation: `B8_d5` (foundation 5 in world D)
  - Full extended syntax: `B8^5_d5` (sense + foundation)
  - Canon validation enforcing: worlds A-D, noetics 1-10, foundations 1-7
  - 34 comprehensive test cases in `tests/test_parser_extended.py`
  - Demo script: `examples/demo_extended_parser.py`
  - Complete documentation: `PARSER_EXTENSION_SUMMARY.md`

### Changed
- `ElementRef` class now supports `foundation` and `subfoundation` attributes
- `ElementRef.from_string()` parser extended to handle new syntax formats
- `ElementRef.full_code` property now uses caret notation for sense (was dot notation)
- `parse_equation()` now uses `ElementRef.from_string()` for extended parsing

### Fixed
- Updated `test_element_ref_creation` to reflect caret notation in `full_code`

### Compatibility
- 100% backward compatible with existing syntax
- Dot notation (`.`) still supported for sense, converted to caret (`^`) in output
- Total test suite: 226/226 tests passing (including new suites)

## [v0.2.0] - 2025-12-14

### Added
- Parser extension (sense/foundation suffixes) with validation and demo (`PARSER_EXTENSION_SUMMARY.md`).
- Augmentation pipeline (`scripts/generate_augmented_data.py`): inversion + anti-attractor generation, validation, metrics.
- Canonical validator (`scripts/canonical_validator.py`) and training hook stub (`scripts/train_with_augmented.py`).
- CI fuzz retained; docs updated; release notes and quickstart refreshed.

### Features
- Strict mode default; `--lenient` opt-out.
- Anti-attractor operator heuristics: `*T` for same polarity, `/T` for opposite polarity.
- Canon guardrails: ALLOWED_OPS=9; worlds A/B/C/D; fixed noetics/foundations; SUBFOUND_MAP=28.

## [v0.2.1] - 2025-12-14

### Added
- Training integration Phase 2: functional `load_augmented_corpus`, `prepare_training_batch`, smoke tests, and sample augmented data.
- Monitoring/logging: `scripts/augmentation_metrics.py` (AugmentationLogger), integrated into augmentation and training stubs.
- Release workflow: `.github/workflows/release.yaml`; coverage integration and badge in README.
- Parser UX: improved error messages with extended syntax hints; docs updated.
- Additional docs: `PHASE2_IMPLEMENTATION_SUMMARY.md`, `TRAINING_QUICKSTART.md`.

### Tests/CI
- Total tests: 235/235 passing; fuzz retained.
- CI: release workflow triggers on v* tags; coverage reported.

### Canon
- ALLOWED_OPS=9 unchanged.
- SUBFOUND_MAP=28; worlds A/B/C/D; foundations 1–7; noetics 1–10.

## [v0.2.2] - 2025-12-14

### Added
- Full training loop with DummyTKSModel, augmented data loader, and metrics logging.
- Inference CLI: `scripts/run_inference.py` with 24 tests.
- Metrics plotting: `scripts/plot_metrics.py` (5 plot types).
- Sense coverage: SENSE_RULES expanded 21 → 208 entries.

### Tests/CI
- Total tests: 298 passing; fuzz expanded to 144 tests; coverage ~85%.
- CI: release workflow remains; coverage badge in README.

### Canon
- ALLOWED_OPS=9 unchanged; worlds A/B/C/D; foundations 1–7; noetics 1–10; SUBFOUND_MAP=28; involution pairs N2↔N3, N5↔N6, N8↔N9; self-duals N1, N4, N7, N10.

## [v0.2.3] - 2025-12-14

### Added/Updated
- Training Phase 3: evaluation script, smoke tests, Phase 3 docs; `test_train_with_augmented.py` aligned to new API (TKSTokenizer, TKSAugmentedDataset, TrainingMetricsLogger).
- Sense/Parser QA: 13 new regression tests for extended syntax; lexicon refinements.
- Inference UX: bulk mode verified; 3 new smoke tests for `run_inference.py`.
- Monitoring/Plots: plot functionality verified; dashboard docs added.
- CI/Release: coverage threshold set to 85%; README badges polished.

### Tests/CI
- Total tests: 377 passing; fuzz retained/expanded; coverage ≥85%.
- CI: release workflow intact; coverage badge in README.

### Canon
- ALLOWED_OPS=9 unchanged; worlds A/B/C/D; foundations 1–7; noetics 1–10; SUBFOUND_MAP=28; involution pairs (2↔3), (5↔6), (8↔9); self-duals {1,4,7,10}.

## [v0.2.4] - 2025-12-14

### Added/Updated
- Teacher-driven pipeline: generated 60 teacher examples from 15 equations, validated and augmented (3x inversion/anti-attractor).
- Training run: real model with augmented teacher data; loss reduction 4.17 → 3.24 over 2 epochs.
- Evaluation: 100% canonical validity on eval set (smoke-level accuracy ~1% expected).
- New scripts: `scripts/validate_teacher_output.py`, `scripts/quick_train.py`.
- New data/artifacts: `data/equations.jsonl`, `output/teacher_outputs.jsonl`, `output/teacher_augmented.jsonl`, `output/phase5_models/final_model.pt`, training/eval reports.
- Docs: `docs/RECIPES.md` updated (Recipe 4: Teacher→Train Flow); PHASE5 summaries/guides added.

### Tests/CI
- Total tests: 432 passing; fuzz retained; coverage threshold 87% in CI/release workflows.

### Canon
- ALLOWED_OPS=9 unchanged; worlds A/B/C/D; foundations 1–7; noetics 1–10; SUBFOUND_MAP=28; involution pairs (2↔3), (5↔6), (8↔9); self-duals {1,4,7,10}.


## [v0.1.0] - 2025-12-14

### Added
- Scenario inversion CLI with multi-axis support (N/E/W/F/S/A/P) and three modes (soft/hard/targeted)
- Anti-attractor synthesis for generating counter-scenarios by inverting attractor signatures
- Strict validation mode (default) with helpful error messages for unknown tokens/operators
- Comprehensive test suite: 159 tests with deterministic fuzz suite (56 runs) integrated into CI
- Full documentation: README quickstart, `docs/SCENARIO_INVERSION.md`, and `docs/TRAINING_INTEGRATION_PLAN.md`

### Features
- Strict mode default: unknown tokens/operators rejected with suggestions; use `--lenient` for permissive mode
- Anti-attractor operator heuristics: `*T` for same polarity, `/T` for opposite polarity
- Canon guardrails: 9 allowed operators, worlds A/B/C/D, fixed noetics/foundations, SUBFOUND_MAP=28
- CI/CD: pytest coverage across Python 3.10/3.11 with deterministic fuzz testing

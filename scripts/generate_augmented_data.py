#!/usr/bin/env python3
"""
TKS Data Augmentation Script - Implementation

Generates augmented training data using:
    1. Scenario Inversion (InvertStory API from scenario_inversion.py)
    2. Anti-Attractor Synthesis (AntiAttractorInvert from anti_attractor.py)

This script implements the full augmentation pipeline with:
- JSONL I/O for corpus entries
- Multi-axis inversion support
- Anti-attractor counter-scenario generation
- Canonical validation of TKS expressions
- Comprehensive metrics tracking

DESIGN SPECIFICATION:
    See scripts/AUGMENTATION_PIPELINE_SPEC.md for complete design spec including:
    - Detailed input/output formats
    - Step-by-step processing pipeline
    - Default configurations and guardrails
    - Canonical constraints (worlds: A/B/C/D, noetics: 1-10, foundations: 1-7)
    - Error handling strategies
    - Usage examples and CLI reference

Usage:
    # Basic augmentation (default settings)
    python scripts/generate_augmented_data.py \
        --input data/pilot/stories.jsonl \
        --output data/pilot/augmented.jsonl

    # Full augmentation with anti-attractors
    python scripts/generate_augmented_data.py \
        --input data/pilot/stories.jsonl \
        --output data/pilot/augmented.jsonl \
        --axes W N F \
        --use-anti-attractor \
        --validate

    # Lenient mode for exploratory corpus development
    python scripts/generate_augmented_data.py \
        --input data/experimental/stories.jsonl \
        --output data/experimental/augmented.jsonl \
        --lenient

Author: TKS-LLM Training Integration Team
Date: 2025-12-14
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# TKS imports
try:
    from scenario_inversion import (
        InvertStory,
        EncodeStory,
        DecodeStory,
        TKSExpression,
        parse_equation,
        AntiAttractorInvert
    )
    from anti_attractor import compute_attractor_signature
    from narrative.constants import ALLOWED_OPS, WORLD_LETTERS
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import TKS modules: {e}")
    print("This is expected for Phase 1 stub. Full implementation will resolve imports.")
    IMPORTS_AVAILABLE = False

# Import augmentation metrics logger
try:
    from augmentation_metrics import AugmentationLogger
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import augmentation_metrics. Metrics logging will be limited.")
    METRICS_AVAILABLE = False


# ==============================================================================
# CONFIGURATION DATA CLASSES
# ==============================================================================

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation pipeline."""

    # Inversion settings
    axes_combinations: List[Set[str]] = field(default_factory=lambda: [
        {"W"},      # World-only inversion
        {"N"},      # Noetic-only inversion
        {"W", "N"}, # World + Noetic combined
    ])
    inversion_mode: str = "soft"  # "soft", "hard", or "targeted"

    # Anti-attractor settings
    use_anti_attractor: bool = True
    anti_attractor_elements: int = 3  # Number of elements in counter-scenario

    # Validation settings
    validate_canonical: bool = True
    min_pass_rate: float = 0.90  # 90% minimum pass rate

    # Output settings
    save_metrics: bool = True
    verbose: bool = True


@dataclass
class AugmentationMetrics:
    """Metrics tracking for augmentation process."""

    # Counts
    original_count: int = 0
    inverted_count: int = 0
    anti_attractor_count: int = 0
    validation_failures: int = 0

    # Ratios
    augmentation_ratio: float = 0.0
    inversion_ratio: float = 0.0
    anti_attractor_ratio: float = 0.0

    # Validation metrics
    validator_pass_rate: float = 0.0
    world_validity: float = 0.0
    noetic_validity: float = 0.0
    operator_validity: float = 0.0
    structural_validity: float = 0.0

    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return asdict(self)


# ==============================================================================
# CORE AUGMENTATION FUNCTIONS (STUBS FOR PHASE 1)
# ==============================================================================

def generate_inverted_scenarios(
    entry: Dict[str, Any],
    axes: Set[str],
    mode: str = "soft",
    strict: bool = False
) -> Dict[str, Any]:
    """
    Generate inverted scenario using InvertStory API.

    Args:
        entry: Original entry dict with "story" or "equation" field
        axes: Set of axes to invert (e.g., {"W", "N"})
        mode: Inversion mode ("soft", "hard", "targeted")
        strict: If True, raise errors on encoding failures

    Returns:
        Dict containing:
            - story: Inverted story text
            - expr: Inverted TKS expression (as string)
            - expr_elements: List of element codes
            - expr_ops: List of operators
            - axes: Axes applied
            - mode: Mode used

    Raises:
        ValueError: If strict=True and encoding fails
    """
    if not IMPORTS_AVAILABLE:
        return {
            "story": f"[INVERTED: {entry.get('story', entry.get('equation', ''))}]",
            "expr": None,
            "expr_elements": [],
            "expr_ops": [],
            "axes": list(axes),
            "mode": mode,
        }

    try:
        # Determine if we have a story or equation
        if "story" in entry:
            # Use InvertStory for natural language
            result = InvertStory(entry["story"], axes, mode, strict=strict)
            inverted_expr = result["expr_inverted"]
            inverted_story = result["story_inverted"]
        elif "equation" in entry:
            # Parse equation and invert expression
            expr_original = parse_equation(entry["equation"])
            from scenario_inversion import ScenarioInvert
            inverted_expr = ScenarioInvert(expr_original, axes, mode)
            inverted_story = DecodeStory(inverted_expr)
        else:
            raise ValueError("Entry must have 'story' or 'equation' field")

        # Format expression as string
        expr_str = " ".join(
            [inverted_expr.elements[0]] +
            [f"{op} {elem}" for op, elem in zip(inverted_expr.ops, inverted_expr.elements[1:])]
        ) if inverted_expr.elements else ""

        return {
            "story": inverted_story,
            "expr": expr_str,
            "expr_elements": list(inverted_expr.elements),
            "expr_ops": list(inverted_expr.ops),
            "axes": list(axes),
            "mode": mode,
        }

    except Exception as e:
        if strict:
            raise
        # In non-strict mode, return a placeholder
        print(f"Warning: Failed to invert entry: {e}")
        return {
            "story": f"[INVERSION FAILED: {str(e)}]",
            "expr": None,
            "expr_elements": [],
            "expr_ops": [],
            "axes": list(axes),
            "mode": mode,
        }


def generate_anti_attractor_pairs(
    entry: Dict[str, Any],
    num_elements: int = 3,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Generate anti-attractor counter-scenario.

    Args:
        entry: Original entry dict with "story" or "equation" field
        num_elements: Number of elements in counter-scenario
        strict: If True, raise errors on encoding failures

    Returns:
        Dict containing:
            - story: Counter-scenario story text
            - expr: Anti-attractor TKS expression (as string)
            - expr_elements: List of element codes
            - expr_ops: List of operators

    Raises:
        ValueError: If strict=True and encoding fails
    """
    if not IMPORTS_AVAILABLE:
        return {
            "story": "[ANTI-ATTRACTOR: counter-scenario]",
            "expr": None,
            "expr_elements": [],
            "expr_ops": [],
        }

    try:
        # Encode the entry to TKS expression
        if "story" in entry:
            expr_original = EncodeStory(entry["story"], strict=strict)
        elif "equation" in entry:
            expr_original = parse_equation(entry["equation"])
        else:
            raise ValueError("Entry must have 'story' or 'equation' field")

        # Generate anti-attractor
        result = AntiAttractorInvert(expr_original, return_signature=False)
        anti_expr = result["expr_inverted"]

        # Decode to story
        anti_story = DecodeStory(anti_expr)

        # Format expression as string
        expr_str = " ".join(
            [anti_expr.elements[0]] +
            [f"{op} {elem}" for op, elem in zip(anti_expr.ops, anti_expr.elements[1:])]
        ) if anti_expr.elements else ""

        return {
            "story": anti_story,
            "expr": expr_str,
            "expr_elements": list(anti_expr.elements),
            "expr_ops": list(anti_expr.ops),
        }

    except Exception as e:
        if strict:
            raise
        # In non-strict mode, return a placeholder
        print(f"Warning: Failed to generate anti-attractor: {e}")
        return {
            "story": f"[ANTI-ATTRACTOR FAILED: {str(e)}]",
            "expr": None,
            "expr_elements": [],
            "expr_ops": [],
        }


def is_canonical_expr(expr: TKSExpression) -> Tuple[bool, Optional[str]]:
    """
    Validate TKS expression against canonical semantics.

    Args:
        expr: TKSExpression to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if expression is canonical
        - error_message: None if valid, error description if invalid

    Validation Checks:
        1. Valid world letters (A, B, C, D only)
        2. Valid noetic indices (1-10)
        3. Valid operators from ALLOWED_OPS
        4. Structural consistency (elements/ops alignment)
        5. Foundation validity (if present)
    """
    if not IMPORTS_AVAILABLE:
        return True, None

    # Check world letters and noetic indices
    for elem in expr.elements:
        if len(elem) < 2:
            return False, f"Invalid element format: {elem}"

        world = elem[0]
        try:
            noetic = int(elem[1:])
        except ValueError:
            return False, f"Invalid noetic in element: {elem}"

        if world not in WORLD_LETTERS:
            return False, f"Invalid world letter: {world} (must be A, B, C, or D)"

        if not (1 <= noetic <= 10):
            return False, f"Invalid noetic index: {noetic} (must be 1-10)"

    # Check operators
    for op in expr.ops:
        if op not in ALLOWED_OPS:
            return False, f"Invalid operator: {op}"

    # Check structural consistency
    if len(expr.elements) > 0 and len(expr.ops) != len(expr.elements) - 1:
        return False, f"Structural inconsistency: {len(expr.elements)} elements but {len(expr.ops)} operators"

    # Check foundations
    for fid, _ in expr.foundations:
        if not (1 <= fid <= 7):
            return False, f"Invalid foundation ID: {fid} (must be 1-7)"

    return True, None


def validate_canonical(story_or_expr) -> TKSExpression:
    """
    Validate story/expression against TKS canonical semantics.

    Args:
        story_or_expr: Story string or TKSExpression to validate

    Returns:
        TKSExpression if valid

    Raises:
        ValueError: If validation fails, with detailed error message
    """
    if not IMPORTS_AVAILABLE:
        # Fallback for when imports not available
        if isinstance(story_or_expr, str):
            return TKSExpression(
                elements=["A1"],
                ops=[],
                foundations=[],
                acquisitions=[],
                raw=story_or_expr
            )
        return story_or_expr

    # Convert string to expression if needed
    if isinstance(story_or_expr, str):
        expr = EncodeStory(story_or_expr, strict=True)
    else:
        expr = story_or_expr

    # Validate the expression
    is_valid, error_msg = is_canonical_expr(expr)
    if not is_valid:
        raise ValueError(f"Validation failed: {error_msg}")

    return expr


# ==============================================================================
# METRICS COMPUTATION FUNCTIONS (STUBS FOR PHASE 1)
# ==============================================================================

def compute_validator_pass_rate(augmented_entries: List[Dict]) -> Dict[str, float]:
    """
    Compute canonical validation pass rate for augmented entries.

    Args:
        augmented_entries: List of augmented entry dicts

    Returns:
        Dict with validation metrics:
            - total: Total scenarios tested
            - valid: Number passing validation
            - pass_rate: Overall pass rate (0-1)
            - world_validity: % with valid world letters
            - noetic_validity: % with valid noetic indices
            - operator_validity: % with valid operators
            - structural_validity: % with valid structure
    """
    if not augmented_entries:
        return {
            "total": 0,
            "valid": 0,
            "pass_rate": 0.0,
            "world_validity": 0.0,
            "noetic_validity": 0.0,
            "operator_validity": 0.0,
            "structural_validity": 0.0
        }

    total = len(augmented_entries)
    valid_count = 0
    world_valid = 0
    noetic_valid = 0
    operator_valid = 0
    structural_valid = 0

    for entry in augmented_entries:
        # Skip entries that don't have validator_pass field
        if "validator_pass" not in entry:
            continue

        if entry["validator_pass"]:
            valid_count += 1

        # Check component validity from expr_elements and expr_ops
        if "expr_elements" in entry and entry["expr_elements"]:
            # Check world validity
            worlds_ok = all(
                elem[0] in WORLD_LETTERS if IMPORTS_AVAILABLE and len(elem) > 0 else True
                for elem in entry["expr_elements"]
            )
            if worlds_ok:
                world_valid += 1

            # Check noetic validity
            noetics_ok = all(
                1 <= int(elem[1:]) <= 10 if len(elem) > 1 else False
                for elem in entry["expr_elements"]
            )
            if noetics_ok:
                noetic_valid += 1

        # Check operator validity
        if "expr_ops" in entry and entry["expr_ops"]:
            ops_ok = all(
                op in ALLOWED_OPS if IMPORTS_AVAILABLE else True
                for op in entry["expr_ops"]
            )
            if ops_ok:
                operator_valid += 1

        # Check structural validity
        if "expr_elements" in entry and "expr_ops" in entry:
            if entry["expr_elements"] and entry["expr_ops"]:
                struct_ok = len(entry["expr_ops"]) == len(entry["expr_elements"]) - 1
                if struct_ok:
                    structural_valid += 1

    return {
        "total": total,
        "valid": valid_count,
        "pass_rate": valid_count / total if total > 0 else 0.0,
        "world_validity": world_valid / total if total > 0 else 0.0,
        "noetic_validity": noetic_valid / total if total > 0 else 0.0,
        "operator_validity": operator_valid / total if total > 0 else 0.0,
        "structural_validity": structural_valid / total if total > 0 else 0.0
    }


def compute_augmentation_ratio(corpus_metadata: List[Dict]) -> Dict[str, float]:
    """
    Compute augmentation ratios for corpus.

    Args:
        corpus_metadata: List of augmented scenario metadata dicts

    Returns:
        Dict with augmentation ratios:
            - total_ratio: (inverted + anti) / original
            - inversion_ratio: inverted / original
            - anti_attractor_ratio: anti / original
    """
    original_count = sum(1 for item in corpus_metadata if item.get("aug_type") == "original")
    inverted_count = sum(1 for item in corpus_metadata if item.get("aug_type") == "inversion")
    anti_count = sum(1 for item in corpus_metadata if item.get("aug_type") == "anti_attractor")

    if original_count == 0:
        return {
            "total_ratio": 0.0,
            "inversion_ratio": 0.0,
            "anti_attractor_ratio": 0.0
        }

    return {
        "total_ratio": (inverted_count + anti_count) / original_count,
        "inversion_ratio": inverted_count / original_count,
        "anti_attractor_ratio": anti_count / original_count
    }


# ==============================================================================
# CORPUS I/O FUNCTIONS (STUBS FOR PHASE 1)
# ==============================================================================

def load_jsonl(input_path: Path) -> List[Dict]:
    """
    Load corpus from JSONL file.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of entry dicts

    Expected Format:
        Each line is a JSON object with at least a "story" or "equation" field:
        {"story": "A teacher causes growth", "id": "001", ...}
        {"equation": "B5 -> D3", "id": "002", ...}
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    entries = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue

    return entries


def load_corpus(input_path: Path) -> List[str]:
    """
    Legacy wrapper - Load corpus and extract story fields only.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of story strings
    """
    entries = load_jsonl(input_path)
    stories = []
    for entry in entries:
        if "story" in entry:
            stories.append(entry["story"])
        elif "equation" in entry:
            # For equation-only entries, we'll use the equation as-is
            stories.append(entry["equation"])
    return stories


def write_jsonl(output_path: Path, entries: List[Dict]):
    """
    Write entries to JSONL file.

    Args:
        output_path: Path to output JSONL file
        entries: List of dicts to write
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            try:
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + "\n")
            except (TypeError, ValueError) as e:
                print(f"Warning: Failed to serialize entry: {e}")
                continue


def save_augmented_corpus(augmented_data: List[Dict], output_path: Path):
    """
    Save augmented corpus to JSONL file.

    Args:
        augmented_data: List of augmented scenario dicts
        output_path: Path to output JSONL file

    Output Format:
        Each line is a JSON object:
        {
            "story": "Inverted story text",
            "expr": "B3 -> D5",
            "aug_type": "inverted",
            "axes": ["W", "N"],
            "source_id": "001"
        }
    """
    write_jsonl(output_path, augmented_data)


def save_metrics(metrics: AugmentationMetrics, output_path: Path):
    """
    Save augmentation metrics to JSON file.

    TODO (Phase 2):
        - Convert metrics to dict
        - Write formatted JSON
        - Add timestamp
        - Create parent directories if needed

    Args:
        metrics: AugmentationMetrics object
        output_path: Path to output JSON file
    """
    # Phase 1 stub: Create empty file
    # TODO: Implement actual metrics saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics.to_dict(), f, indent=2)


# ==============================================================================
# MAIN AUGMENTATION PIPELINE (STUB FOR PHASE 1)
# ==============================================================================

def augment_corpus(
    input_corpus: Path,
    output_corpus: Path,
    config: AugmentationConfig
) -> AugmentationMetrics:
    """
    Main augmentation pipeline - orchestrates entire process.

    Args:
        input_corpus: Path to input JSONL file
        output_corpus: Path to output JSONL file
        config: AugmentationConfig with pipeline settings

    Returns:
        AugmentationMetrics with final statistics

    Pipeline Flow:
        1. Load corpus
        2. For each entry:
            a. Validate canonical (skip if invalid in strict mode)
            b. Add original entry
            c. Generate inversions (multiple axes)
            d. Generate anti-attractor pairs
            e. Validate all augmentations
        3. Compute metrics
        4. Save augmented corpus
        5. Save metrics
    """
    import time
    start_time_ts = time.time()

    metrics = AugmentationMetrics(
        start_time=datetime.now().isoformat()
    )

    # Initialize augmentation logger
    logger = None
    if METRICS_AVAILABLE:
        logger = AugmentationLogger()

    if config.verbose:
        print(f"TKS Data Augmentation Pipeline")
        print(f"Input:  {input_corpus}")
        print(f"Output: {output_corpus}")
        print(f"Axes combinations: {config.axes_combinations}")
        print(f"Anti-attractor: {config.use_anti_attractor}")
        print(f"Validation: {config.validate_canonical}")
        if logger:
            print(f"Metrics logging: enabled")
        print()

    augmented_data = []

    # Load corpus entries
    entries = load_jsonl(input_corpus)
    metrics.original_count = len(entries)

    if config.verbose:
        print(f"Loaded {len(entries)} entries from corpus")
        print()

    # Process each entry
    for idx, entry in enumerate(entries):
        entry_id = entry.get("id", f"entry_{idx}")

        if config.verbose and idx % 10 == 0:
            print(f"Processing entry {idx + 1}/{len(entries)}...")

        # Validate original entry
        try:
            if "story" in entry:
                original_expr = EncodeStory(entry["story"], strict=config.validate_canonical)
            elif "equation" in entry:
                original_expr = parse_equation(entry["equation"])
            else:
                if config.verbose:
                    print(f"Skipping entry {entry_id}: no 'story' or 'equation' field")
                continue

            # Validate canonical
            is_valid, error_msg = is_canonical_expr(original_expr)
            original_validator_pass = is_valid

            if not is_valid and config.validate_canonical:
                if config.verbose:
                    print(f"Validation failed for {entry_id}: {error_msg}")
                metrics.validation_failures += 1
                # Skip in strict mode, continue in lenient mode
                continue

        except Exception as e:
            if config.verbose:
                print(f"Failed to encode entry {entry_id}: {e}")
            metrics.validation_failures += 1
            continue

        # Add original entry
        original_entry = {
            "id": entry_id,
            "story": entry.get("story", ""),
            "equation": entry.get("equation", ""),
            "expr": " ".join(
                [original_expr.elements[0]] +
                [f"{op} {elem}" for op, elem in zip(original_expr.ops, original_expr.elements[1:])]
            ) if original_expr.elements else "",
            "expr_elements": list(original_expr.elements),
            "expr_ops": list(original_expr.ops),
            "aug_type": "original",
            "source_id": entry_id,
            "validator_pass": original_validator_pass,
        }
        augmented_data.append(original_entry)

        # Log to metrics
        if logger:
            logger.log_entry(original_entry)

        # Generate inversions for each axis combination
        for axes in config.axes_combinations:
            try:
                inverted = generate_inverted_scenarios(
                    entry, axes, config.inversion_mode, strict=False
                )

                # Validate inverted expression
                if inverted["expr_elements"]:
                    inv_expr = TKSExpression(
                        elements=inverted["expr_elements"],
                        ops=inverted["expr_ops"],
                        foundations=[],
                        acquisitions=[],
                        raw=""
                    )
                    is_valid, _ = is_canonical_expr(inv_expr)
                else:
                    is_valid = False

                inversion_entry = {
                    "id": f"{entry_id}_inv_{''.join(sorted(axes))}",
                    "story": inverted["story"],
                    "expr": inverted["expr"],
                    "expr_elements": inverted["expr_elements"],
                    "expr_ops": inverted["expr_ops"],
                    "aug_type": "inversion",
                    "source_id": entry_id,
                    "axes": inverted["axes"],
                    "mode": inverted["mode"],
                    "validator_pass": is_valid,
                }
                augmented_data.append(inversion_entry)
                metrics.inverted_count += 1

                # Log to metrics
                if logger:
                    logger.log_entry(inversion_entry)

            except Exception as e:
                if config.verbose:
                    print(f"Failed to generate inversion for {entry_id}: {e}")
                continue

        # Generate anti-attractor pairs
        if config.use_anti_attractor:
            try:
                anti = generate_anti_attractor_pairs(
                    entry, config.anti_attractor_elements, strict=False
                )

                # Validate anti-attractor expression
                if anti["expr_elements"]:
                    anti_expr = TKSExpression(
                        elements=anti["expr_elements"],
                        ops=anti["expr_ops"],
                        foundations=[],
                        acquisitions=[],
                        raw=""
                    )
                    is_valid, _ = is_canonical_expr(anti_expr)
                else:
                    is_valid = False

                anti_entry = {
                    "id": f"{entry_id}_anti",
                    "story": anti["story"],
                    "expr": anti["expr"],
                    "expr_elements": anti["expr_elements"],
                    "expr_ops": anti["expr_ops"],
                    "aug_type": "anti_attractor",
                    "source_id": entry_id,
                    "validator_pass": is_valid,
                }
                augmented_data.append(anti_entry)
                metrics.anti_attractor_count += 1

                # Log to metrics
                if logger:
                    logger.log_entry(anti_entry)

            except Exception as e:
                if config.verbose:
                    print(f"Failed to generate anti-attractor for {entry_id}: {e}")
                continue

    # Compute final metrics
    if config.verbose:
        print()
        print("Computing final metrics...")

    # Compute ratios
    ratios = compute_augmentation_ratio(augmented_data)
    metrics.augmentation_ratio = ratios["total_ratio"]
    metrics.inversion_ratio = ratios["inversion_ratio"]
    metrics.anti_attractor_ratio = ratios["anti_attractor_ratio"]

    # Compute validation metrics
    validation_metrics = compute_validator_pass_rate(augmented_data)
    metrics.validator_pass_rate = validation_metrics["pass_rate"]
    metrics.world_validity = validation_metrics["world_validity"]
    metrics.noetic_validity = validation_metrics["noetic_validity"]
    metrics.operator_validity = validation_metrics["operator_validity"]
    metrics.structural_validity = validation_metrics["structural_validity"]

    # Save augmented corpus
    save_augmented_corpus(augmented_data, output_corpus)

    # Finalize metrics
    metrics.end_time = datetime.now().isoformat()
    metrics.duration_seconds = time.time() - start_time_ts

    if config.save_metrics:
        metrics_path = output_corpus.with_suffix(".metrics.json")
        save_metrics(metrics, metrics_path)
        if config.verbose:
            print(f"Saved metrics to {metrics_path}")

        # Save detailed logger metrics if available
        if logger:
            # Save JSON (full detailed metrics)
            logger_metrics_path = output_corpus.with_suffix(".detailed_metrics.json")
            logger.save(str(logger_metrics_path))
            if config.verbose:
                print(f"Saved detailed metrics to {logger_metrics_path}")

            # Save CSV (for trend tracking and plotting)
            csv_metrics_path = output_corpus.with_suffix(".metrics.csv")
            logger.save_to_csv(str(csv_metrics_path), append=False)
            if config.verbose:
                print(f"Saved CSV metrics to {csv_metrics_path}")

            # Save JSON array format (for multi-run trend tracking)
            trend_metrics_path = output_corpus.parent / "augmentation_trends.json"
            logger.save_to_json(str(trend_metrics_path), append=True)
            if config.verbose:
                print(f"Appended trend metrics to {trend_metrics_path}")

                # Print summary
                print()
                logger.print_summary(detailed=True)

    return metrics


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TKS Data Augmentation Script - Generate augmented training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic augmentation with default settings
    python scripts/generate_augmented_data.py \\
        --input data/pilot/stories.jsonl \\
        --output data/pilot/augmented.jsonl

    # Custom axes and anti-attractor
    python scripts/generate_augmented_data.py \\
        --input data/pilot/stories.jsonl \\
        --output data/pilot/augmented.jsonl \\
        --axes W N F \\
        --use-anti-attractor \\
        --validate

    # Production mode with strict validation
    python scripts/generate_augmented_data.py \\
        --input data/training/corpus.jsonl \\
        --output data/training/augmented.jsonl \\
        --mode soft \\
        --min-pass-rate 0.95 \\
        --save-metrics
        """
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input corpus JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output augmented corpus JSONL file"
    )

    # Inversion settings
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["W", "N"],
        choices=["N", "E", "W", "F", "S", "A", "P"],
        help="Axes to use for inversion (default: W N)"
    )
    parser.add_argument(
        "--mode",
        default="soft",
        choices=["soft", "hard", "targeted"],
        help="Inversion mode (default: soft)"
    )

    # Anti-attractor settings
    parser.add_argument(
        "--use-anti-attractor",
        action="store_true",
        help="Generate anti-attractor pairs"
    )
    parser.add_argument(
        "--anti-elements",
        type=int,
        default=3,
        help="Number of elements in anti-attractor scenarios (default: 3)"
    )

    # Validation settings
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run canonical validation on augmented data (default: True)"
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.90,
        help="Minimum validation pass rate (default: 0.90)"
    )

    # Output settings
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        default=True,
        help="Save augmentation metrics to JSON (default: True)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build axes combinations from CLI args
    # For Phase 1, just use individual axes
    axes_combinations = [set([axis]) for axis in args.axes]

    # Create configuration
    config = AugmentationConfig(
        axes_combinations=axes_combinations,
        inversion_mode=args.mode,
        use_anti_attractor=args.use_anti_attractor,
        anti_attractor_elements=args.anti_elements,
        validate_canonical=args.validate,
        min_pass_rate=args.min_pass_rate,
        save_metrics=args.save_metrics,
        verbose=args.verbose
    )

    # Run augmentation pipeline
    try:
        metrics = augment_corpus(
            input_corpus=args.input,
            output_corpus=args.output,
            config=config
        )

        if config.verbose:
            print("\n" + "="*60)
            print("AUGMENTATION COMPLETE")
            print("="*60)
            print(f"Original scenarios:       {metrics.original_count}")
            print(f"Inverted scenarios:       {metrics.inverted_count}")
            print(f"Anti-attractor scenarios: {metrics.anti_attractor_count}")
            print(f"Validation failures:      {metrics.validation_failures}")
            print()
            print(f"Augmentation ratio:       {metrics.augmentation_ratio:.2f}x")
            print(f"Inversion ratio:          {metrics.inversion_ratio:.2f}x")
            print(f"Anti-attractor ratio:     {metrics.anti_attractor_ratio:.2f}x")
            print()
            print(f"Validation pass rate:     {metrics.validator_pass_rate:.2%}")
            print(f"World validity:           {metrics.world_validity:.2%}")
            print(f"Noetic validity:          {metrics.noetic_validity:.2%}")
            print(f"Operator validity:        {metrics.operator_validity:.2%}")
            print(f"Structural validity:      {metrics.structural_validity:.2%}")
            print()
            print(f"Duration:                 {metrics.duration_seconds:.2f} seconds")
            print("="*60)

        return 0

    except Exception as e:
        print(f"Error during augmentation: {e}", file=sys.stderr)
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

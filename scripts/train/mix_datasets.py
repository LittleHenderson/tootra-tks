#!/usr/bin/env python3
"""
Mix Datasets — Combine and shuffle training data for TKS-LLM v4

This script combines multiple JSONL training datasets with configurable
mixing ratios, shuffle, and stratification options.

Features:
- Mix multiple JSONL files with custom ratios
- Stratified sampling by task type
- Deduplication by content hash
- Train/val/test split generation
- Statistics and validation

Usage:
    python scripts/train/mix_datasets.py --config configs/mix_config.yaml
    python scripts/train/mix_datasets.py --auto  # Auto-detect and mix all packs

Datasets supported:
- Operator training packs (2250 records)
- RPM/regulator packs (800 records)
- Consciousness unlock packs (1000 records)
- Equation corpus (600K records)

Author: ML Agent
Date: 2026-01-05
"""

import argparse
import json
import hashlib
import logging
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DatasetSpec:
    """Specification for a single dataset."""
    path: str
    weight: float = 1.0
    max_samples: Optional[int] = None
    task_types: Optional[List[str]] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = Path(self.path).stem


@dataclass
class MixConfig:
    """Configuration for dataset mixing."""
    # Input datasets
    datasets: List[DatasetSpec] = field(default_factory=list)

    # Output
    output_dir: str = "data/mixed"
    output_prefix: str = "tks_mixed"

    # Splitting
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    # Processing
    shuffle: bool = True
    seed: int = 42
    deduplicate: bool = True
    stratify_by_type: bool = True

    # Limits
    max_total_samples: Optional[int] = None

    @classmethod
    def auto_detect(cls) -> "MixConfig":
        """Auto-detect available datasets and create config."""
        datasets = []

        # Standard dataset paths
        paths = [
            ("strg-llm-stk/tks_operator_training_pack_2000.jsonl", 1.0, "operator_main"),
            ("strg-llm-stk/tks_operator_training_pack_250.jsonl", 1.0, "operator_extra"),
            ("strg-llm-stk/tks_rpm_fm_acbe_mvr_training_pack_800.jsonl", 1.0, "rpm"),
            ("strg-llm-stk/tks_consciousness_unlock_HS_enabled_pack_1000.jsonl", 1.0, "dps"),
        ]

        for path, weight, name in paths:
            full_path = PROJECT_ROOT / path
            if full_path.exists():
                datasets.append(DatasetSpec(path=path, weight=weight, name=name))
                logger.info(f"Found dataset: {path}")

        return cls(datasets=datasets)


# =============================================================================
# DATASET MIXER
# =============================================================================

class DatasetMixer:
    """Mixes multiple JSONL datasets into combined training files."""

    def __init__(self, config: MixConfig):
        self.config = config
        random.seed(config.seed)

        self.output_dir = PROJECT_ROOT / config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_samples: List[Dict] = []
        self.stats: Dict[str, Any] = defaultdict(int)
        self.seen_hashes: set = set()

    def load_dataset(self, spec: DatasetSpec) -> List[Dict]:
        """Load a single JSONL dataset."""
        samples = []
        path = PROJECT_ROOT / spec.path

        if not path.exists():
            logger.warning(f"Dataset not found: {path}")
            return samples

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())

                    # Filter by task type if specified
                    if spec.task_types:
                        record_type = record.get('type', '')
                        if record_type not in spec.task_types:
                            continue

                    # Add source metadata
                    record['_source'] = spec.name
                    record['_source_line'] = line_num

                    samples.append(record)

                    if spec.max_samples and len(samples) >= spec.max_samples:
                        break

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON error at {spec.path}:{line_num}: {e}")

        logger.info(f"Loaded {len(samples)} samples from {spec.name}")
        return samples

    def content_hash(self, record: Dict) -> str:
        """Generate hash of record content for deduplication."""
        # Hash based on messages content
        messages = record.get('messages', [])
        content = ""
        for msg in messages:
            content += msg.get('role', '') + msg.get('content', '')
        return hashlib.md5(content.encode()).hexdigest()

    def deduplicate(self, samples: List[Dict]) -> List[Dict]:
        """Remove duplicate samples based on content hash."""
        unique = []
        for sample in samples:
            h = self.content_hash(sample)
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                unique.append(sample)
            else:
                self.stats['duplicates_removed'] += 1
        return unique

    def apply_weights(self, samples: List[Dict], weight: float) -> List[Dict]:
        """Apply sampling weight (oversample or undersample)."""
        if weight == 1.0:
            return samples
        elif weight > 1.0:
            # Oversample
            n_extra = int(len(samples) * (weight - 1.0))
            extra = random.choices(samples, k=n_extra)
            return samples + extra
        else:
            # Undersample
            n_keep = int(len(samples) * weight)
            return random.sample(samples, min(n_keep, len(samples)))

    def mix(self) -> None:
        """Load and mix all datasets."""
        logger.info("=" * 60)
        logger.info("MIXING DATASETS")
        logger.info("=" * 60)

        # Load each dataset
        for spec in self.config.datasets:
            samples = self.load_dataset(spec)

            # Apply weight
            if spec.weight != 1.0:
                samples = self.apply_weights(samples, spec.weight)
                logger.info(f"  After weighting ({spec.weight}x): {len(samples)} samples")

            # Track stats
            self.stats[f'{spec.name}_count'] = len(samples)

            self.all_samples.extend(samples)

        logger.info(f"Total samples before dedup: {len(self.all_samples)}")

        # Deduplicate
        if self.config.deduplicate:
            self.all_samples = self.deduplicate(self.all_samples)
            logger.info(f"After deduplication: {len(self.all_samples)}")
            logger.info(f"  Duplicates removed: {self.stats['duplicates_removed']}")

        # Apply max limit
        if self.config.max_total_samples:
            self.all_samples = self.all_samples[:self.config.max_total_samples]
            logger.info(f"After max limit: {len(self.all_samples)}")

        # Shuffle
        if self.config.shuffle:
            random.shuffle(self.all_samples)
            logger.info("Shuffled samples")

        self.stats['total_mixed'] = len(self.all_samples)

    def split_and_save(self) -> Dict[str, Path]:
        """Split into train/val/test and save to files."""
        n_total = len(self.all_samples)

        # Calculate split sizes
        n_test = int(n_total * self.config.test_ratio)
        n_val = int(n_total * self.config.val_ratio)
        n_train = n_total - n_val - n_test

        # Split
        train_samples = self.all_samples[:n_train]
        val_samples = self.all_samples[n_train:n_train + n_val]
        test_samples = self.all_samples[n_train + n_val:]

        logger.info(f"\nSplit sizes:")
        logger.info(f"  Train: {len(train_samples)}")
        logger.info(f"  Val: {len(val_samples)}")
        logger.info(f"  Test: {len(test_samples)}")

        # Save
        output_files = {}

        for name, samples in [
            ('train', train_samples),
            ('val', val_samples),
            ('test', test_samples),
        ]:
            path = self.output_dir / f"{self.config.output_prefix}_{name}.jsonl"
            self._save_jsonl(samples, path)
            output_files[name] = path

            # Track stats
            self.stats[f'{name}_count'] = len(samples)

        # Save full mixed dataset
        full_path = self.output_dir / f"{self.config.output_prefix}_full.jsonl"
        self._save_jsonl(self.all_samples, full_path)
        output_files['full'] = full_path

        # Save statistics
        stats_path = self.output_dir / f"{self.config.output_prefix}_stats.json"
        self._save_stats(stats_path)
        output_files['stats'] = stats_path

        return output_files

    def _save_jsonl(self, samples: List[Dict], path: Path) -> None:
        """Save samples to JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Remove internal metadata
                clean = {k: v for k, v in sample.items() if not k.startswith('_')}
                f.write(json.dumps(clean, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(samples)} samples to {path}")

    def _save_stats(self, path: Path) -> None:
        """Save mixing statistics."""
        # Compute task type distribution
        type_counts = defaultdict(int)
        for sample in self.all_samples:
            type_counts[sample.get('type', 'unknown')] += 1

        self.stats['task_type_distribution'] = dict(type_counts)

        # Source distribution
        source_counts = defaultdict(int)
        for sample in self.all_samples:
            source_counts[sample.get('_source', 'unknown')] += 1
        self.stats['source_distribution'] = dict(source_counts)

        with open(path, 'w') as f:
            json.dump(dict(self.stats), f, indent=2)
        logger.info(f"Saved stats to {path}")

    def get_stratified_split(
        self,
        samples: List[Dict],
        ratio: float,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Stratified split by task type."""
        by_type = defaultdict(list)
        for sample in samples:
            by_type[sample.get('type', 'unknown')].append(sample)

        split_a, split_b = [], []
        for task_type, type_samples in by_type.items():
            n_a = int(len(type_samples) * ratio)
            random.shuffle(type_samples)
            split_a.extend(type_samples[:n_a])
            split_b.extend(type_samples[n_a:])

        return split_a, split_b


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Mix TKS training datasets")
    parser.add_argument("--config", type=str, help="Path to mix config YAML")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect and mix all available datasets")
    parser.add_argument("--output-dir", type=str, default="data/mixed",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle output")
    args = parser.parse_args()

    # Create config
    if args.auto:
        config = MixConfig.auto_detect()
    elif args.config:
        import yaml
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = MixConfig(**config_dict)
    else:
        # Default: auto-detect
        config = MixConfig.auto_detect()

    # Apply CLI overrides
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.shuffle = not args.no_shuffle

    # Mix datasets
    mixer = DatasetMixer(config)
    mixer.mix()

    # Split and save
    output_files = mixer.split_and_save()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MIXING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total mixed: {mixer.stats['total_mixed']}")
    logger.info(f"Train: {mixer.stats.get('train_count', 0)}")
    logger.info(f"Val: {mixer.stats.get('val_count', 0)}")
    logger.info(f"Test: {mixer.stats.get('test_count', 0)}")
    logger.info(f"\nOutput files:")
    for name, path in output_files.items():
        logger.info(f"  {name}: {path}")


if __name__ == "__main__":
    main()

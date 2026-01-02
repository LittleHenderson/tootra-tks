"""
Noetic Coherence Metrics Logger for Training Integration

This module provides functions to compute and log noetic coherence metrics
during training. It integrates with the training loop to track:
- World Separation (target world accuracy)
- RPM Differentiation (D/W/P variance)
- Attractor Convergence (convergence rate)
- Noetic Consistency (similar/dissimilar separation)

Usage in training loop:
    from training.noetic_metrics import NoeticMetricsLogger

    logger = NoeticMetricsLogger(model, tokenizer, device)

    for epoch in range(num_epochs):
        # ... training code ...

        # Log metrics at end of epoch
        metrics = logger.log_epoch(epoch)

        # Check for collapse
        if logger.detect_collapse():
            print("WARNING: Noetic collapse detected!")

        # Save checkpoint only if metrics are good
        if metrics['all_passed']:
            save_checkpoint(model, f"epoch_{epoch}_passed.pt")

References:
    - tests/test_noetic_regression.py (thresholds and metric computation)
    - scripts/compare_checkpoints.py (checkpoint comparison)
    - TKS_LLM_Noetic_Mathematics_v1.0.md

Author: TKS CI/Evaluation Specialist (Agent C)
Date: 2025-12-22
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# THRESHOLDS (imported from test module for consistency)
# =============================================================================

@dataclass
class NoeticThresholds:
    """Pass/fail thresholds for noetic coherence metrics."""
    WORLD_SEPARATION_THRESHOLD: float = 0.60
    RPM_VARIANCE_THRESHOLD: float = 0.01
    ATTRACTOR_CONVERGENCE_THRESHOLD: float = 0.80
    NOETIC_CONSISTENCY_THRESHOLD: float = 0.05

    # Early warning thresholds (detect problems before failure)
    WORLD_SEPARATION_WARNING: float = 0.50
    RPM_VARIANCE_WARNING: float = 0.005
    ATTRACTOR_CONVERGENCE_WARNING: float = 0.70
    NOETIC_CONSISTENCY_WARNING: float = 0.02


THRESHOLDS = NoeticThresholds()


# =============================================================================
# TEST DATA (subset for fast in-training evaluation)
# =============================================================================

# Minimal test set for fast evaluation during training
FAST_SIMILAR_PAIRS = [
    ("Mental focus and clarity", "Clear thinking and concentration"),
    ("Emotional warmth and love", "Feeling affection and care"),
]

FAST_DISSIMILAR_PAIRS = [
    ("Mental focus and clarity", "Physical energy and power"),
    ("1:7:9", "Fear and anxiety"),
]

FAST_WORLD_TEXTS = {
    'A': ["A1 + A2 + A3"],
    'B': ["B4 + B5 + B6"],
    'C': ["C7 + C8 + C9"],
    'D': ["D10 + D1 + D2"],
}

FAST_RPM_TEXTS = {
    'desire': ["I want power and control"],
    'wisdom': ["Understanding the balance"],
    'power': ["Cause and effect in action"],
}

FAST_ATTRACTOR_TEXTS = [
    "1:7:9",
    "A1 + B2 + C3 + D4",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_noetic_state(model, tokenizer, text: str, device) -> Dict:
    """Extract noetic state for input text."""
    input_ids = tokenizer.tokenize(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_tensor)

    gated_output = output['gated_output'][0]
    rpm_gate = output['rpm_gate'][0]
    attractor_converged = output['attractor_converged']

    content_len = min(len(text) + 2, gated_output.shape[0])
    noetic_state = gated_output[:content_len].mean(dim=0).cpu().numpy()
    rpm_mean = rpm_gate[:content_len].mean().item()

    world_states = {
        'A': noetic_state[0:10],
        'B': noetic_state[10:20],
        'C': noetic_state[20:30],
        'D': noetic_state[30:40],
    }

    return {
        'noetic_state': noetic_state,
        'world_states': world_states,
        'rpm_gate': rpm_mean,
        'attractor_converged': bool(attractor_converged),
    }


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# METRICS LOGGER CLASS
# =============================================================================

@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    timestamp: float
    world_separation: float
    rpm_variance: float
    attractor_convergence: float
    noetic_consistency: float
    world_separation_passed: bool
    rpm_variance_passed: bool
    attractor_convergence_passed: bool
    noetic_consistency_passed: bool
    all_passed: bool
    any_warning: bool


class NoeticMetricsLogger:
    """
    Logger for tracking noetic coherence metrics during training.

    Usage:
        logger = NoeticMetricsLogger(model, tokenizer, device)

        for epoch in range(num_epochs):
            train_epoch(model, data)
            metrics = logger.log_epoch(epoch)

            if logger.detect_collapse():
                handle_collapse()

        logger.save_history("training_metrics.json")
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        fast_mode: bool = True,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the metrics logger.

        Args:
            model: TKSLLMCorePipeline model
            tokenizer: Tokenizer instance
            device: torch.device
            fast_mode: Use reduced test set for faster evaluation
            log_dir: Directory to save metrics (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.fast_mode = fast_mode
        self.log_dir = Path(log_dir) if log_dir else None

        self.history: List[EpochMetrics] = []
        self.collapse_detected = False
        self.best_metrics: Optional[EpochMetrics] = None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(self) -> Dict:
        """
        Compute all noetic coherence metrics.

        Returns:
            Dict with all metric values and pass/fail status
        """
        self.model.eval()

        # Select test data based on mode
        if self.fast_mode:
            similar_pairs = FAST_SIMILAR_PAIRS
            dissimilar_pairs = FAST_DISSIMILAR_PAIRS
            world_texts = FAST_WORLD_TEXTS
            rpm_texts = FAST_RPM_TEXTS
            attractor_texts = FAST_ATTRACTOR_TEXTS
        else:
            # Import full test data from test module
            from tests.test_noetic_regression import (
                SIMILAR_PAIRS,
                DISSIMILAR_PAIRS,
                WORLD_TEXTS,
                RPM_TEST_CASES,
                ATTRACTOR_TEXTS,
            )
            similar_pairs = SIMILAR_PAIRS
            dissimilar_pairs = DISSIMILAR_PAIRS
            world_texts = WORLD_TEXTS
            rpm_texts = RPM_TEST_CASES
            attractor_texts = ATTRACTOR_TEXTS

        # 1. Noetic Consistency
        similar_scores = []
        for text1, text2 in similar_pairs:
            state1 = get_noetic_state(self.model, self.tokenizer, text1, self.device)
            state2 = get_noetic_state(self.model, self.tokenizer, text2, self.device)
            sim = cosine_similarity(state1['noetic_state'], state2['noetic_state'])
            similar_scores.append(sim)

        dissimilar_scores = []
        for text1, text2 in dissimilar_pairs:
            state1 = get_noetic_state(self.model, self.tokenizer, text1, self.device)
            state2 = get_noetic_state(self.model, self.tokenizer, text2, self.device)
            sim = cosine_similarity(state1['noetic_state'], state2['noetic_state'])
            dissimilar_scores.append(sim)

        noetic_consistency = np.mean(similar_scores) - np.mean(dissimilar_scores)

        # 2. World Separation
        correct_count = 0
        total_count = 0
        for target_world, texts in world_texts.items():
            for text in texts:
                state = get_noetic_state(self.model, self.tokenizer, text, self.device)
                world_activations = {
                    w: float(np.linalg.norm(state['world_states'][w]))
                    for w in 'ABCD'
                }
                max_world = max(world_activations, key=world_activations.get)
                if max_world == target_world:
                    correct_count += 1
                total_count += 1

        world_separation = correct_count / total_count if total_count > 0 else 0.0

        # 3. RPM Differentiation
        category_means = {}
        for category, texts in rpm_texts.items():
            scores = []
            for text in texts:
                state = get_noetic_state(self.model, self.tokenizer, text, self.device)
                scores.append(state['rpm_gate'])
            category_means[category] = float(np.mean(scores))

        rpm_variance = float(np.var(list(category_means.values())))

        # 4. Attractor Convergence
        converged_count = 0
        for text in attractor_texts:
            state = get_noetic_state(self.model, self.tokenizer, text, self.device)
            if state['attractor_converged']:
                converged_count += 1

        attractor_convergence = converged_count / len(attractor_texts) if attractor_texts else 0.0

        # Check thresholds
        world_separation_passed = world_separation >= THRESHOLDS.WORLD_SEPARATION_THRESHOLD
        rpm_variance_passed = rpm_variance > THRESHOLDS.RPM_VARIANCE_THRESHOLD
        attractor_convergence_passed = attractor_convergence >= THRESHOLDS.ATTRACTOR_CONVERGENCE_THRESHOLD
        noetic_consistency_passed = noetic_consistency > THRESHOLDS.NOETIC_CONSISTENCY_THRESHOLD

        # Check warnings
        world_separation_warning = world_separation < THRESHOLDS.WORLD_SEPARATION_WARNING
        rpm_variance_warning = rpm_variance < THRESHOLDS.RPM_VARIANCE_WARNING
        attractor_convergence_warning = attractor_convergence < THRESHOLDS.ATTRACTOR_CONVERGENCE_WARNING
        noetic_consistency_warning = noetic_consistency < THRESHOLDS.NOETIC_CONSISTENCY_WARNING

        return {
            'world_separation': float(world_separation),
            'rpm_variance': float(rpm_variance),
            'attractor_convergence': float(attractor_convergence),
            'noetic_consistency': float(noetic_consistency),
            'world_separation_passed': world_separation_passed,
            'rpm_variance_passed': rpm_variance_passed,
            'attractor_convergence_passed': attractor_convergence_passed,
            'noetic_consistency_passed': noetic_consistency_passed,
            'all_passed': all([
                world_separation_passed,
                rpm_variance_passed,
                attractor_convergence_passed,
                noetic_consistency_passed,
            ]),
            'any_warning': any([
                world_separation_warning,
                rpm_variance_warning,
                attractor_convergence_warning,
                noetic_consistency_warning,
            ]),
        }

    def log_epoch(self, epoch: int, print_metrics: bool = True) -> EpochMetrics:
        """
        Compute and log metrics for an epoch.

        Args:
            epoch: Current epoch number
            print_metrics: Whether to print metrics to console

        Returns:
            EpochMetrics dataclass with all metrics
        """
        metrics = self.compute_metrics()

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            timestamp=time.time(),
            world_separation=metrics['world_separation'],
            rpm_variance=metrics['rpm_variance'],
            attractor_convergence=metrics['attractor_convergence'],
            noetic_consistency=metrics['noetic_consistency'],
            world_separation_passed=metrics['world_separation_passed'],
            rpm_variance_passed=metrics['rpm_variance_passed'],
            attractor_convergence_passed=metrics['attractor_convergence_passed'],
            noetic_consistency_passed=metrics['noetic_consistency_passed'],
            all_passed=metrics['all_passed'],
            any_warning=metrics['any_warning'],
        )

        self.history.append(epoch_metrics)

        # Update best metrics
        if self.best_metrics is None or (epoch_metrics.all_passed and not self.best_metrics.all_passed):
            self.best_metrics = epoch_metrics
        elif epoch_metrics.all_passed and self.best_metrics.all_passed:
            # Compare by sum of normalized metrics
            current_score = (
                epoch_metrics.world_separation +
                epoch_metrics.rpm_variance * 100 +  # Scale up
                epoch_metrics.attractor_convergence +
                epoch_metrics.noetic_consistency * 10  # Scale up
            )
            best_score = (
                self.best_metrics.world_separation +
                self.best_metrics.rpm_variance * 100 +
                self.best_metrics.attractor_convergence +
                self.best_metrics.noetic_consistency * 10
            )
            if current_score > best_score:
                self.best_metrics = epoch_metrics

        if print_metrics:
            self._print_epoch_metrics(epoch_metrics)

        # Save to log dir if specified
        if self.log_dir:
            self._save_epoch_metrics(epoch_metrics)

        return epoch_metrics

    def _print_epoch_metrics(self, metrics: EpochMetrics):
        """Print epoch metrics to console."""
        status = "PASS" if metrics.all_passed else "FAIL"
        warning = " [WARNING]" if metrics.any_warning else ""

        print(f"\n[Epoch {metrics.epoch}] Noetic Metrics [{status}]{warning}")
        print(f"  World Separation:     {metrics.world_separation:.1%} "
              f"({'PASS' if metrics.world_separation_passed else 'FAIL'})")
        print(f"  RPM Variance:         {metrics.rpm_variance:.6f} "
              f"({'PASS' if metrics.rpm_variance_passed else 'FAIL'})")
        print(f"  Attractor Convergence: {metrics.attractor_convergence:.0%} "
              f"({'PASS' if metrics.attractor_convergence_passed else 'FAIL'})")
        print(f"  Noetic Consistency:   {metrics.noetic_consistency:.4f} "
              f"({'PASS' if metrics.noetic_consistency_passed else 'FAIL'})")

    def _save_epoch_metrics(self, metrics: EpochMetrics):
        """Save epoch metrics to log directory."""
        if not self.log_dir:
            return

        # Append to history file
        history_file = self.log_dir / "noetic_metrics_history.jsonl"
        with open(history_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

        # Save latest metrics
        latest_file = self.log_dir / "noetic_metrics_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

    def detect_collapse(self, lookback: int = 3) -> bool:
        """
        Detect if model has collapsed based on recent metric history.

        Collapse is detected if:
        - All recent epochs fail all thresholds
        - Metrics are trending downward

        Args:
            lookback: Number of recent epochs to check

        Returns:
            True if collapse detected
        """
        if len(self.history) < lookback:
            return False

        recent = self.history[-lookback:]

        # Check if all recent epochs fail
        all_fail = all(not m.all_passed for m in recent)

        # Check for downward trend
        if len(recent) >= 2:
            first = recent[0]
            last = recent[-1]

            trend_down = (
                last.world_separation < first.world_separation and
                last.noetic_consistency < first.noetic_consistency
            )
        else:
            trend_down = False

        self.collapse_detected = all_fail and trend_down
        return self.collapse_detected

    def get_summary(self) -> Dict:
        """Get summary of all logged metrics."""
        if not self.history:
            return {'error': 'No metrics logged yet'}

        return {
            'total_epochs': len(self.history),
            'passing_epochs': sum(1 for m in self.history if m.all_passed),
            'failing_epochs': sum(1 for m in self.history if not m.all_passed),
            'warning_epochs': sum(1 for m in self.history if m.any_warning),
            'collapse_detected': self.collapse_detected,
            'best_epoch': self.best_metrics.epoch if self.best_metrics else None,
            'latest': asdict(self.history[-1]) if self.history else None,
            'best': asdict(self.best_metrics) if self.best_metrics else None,
        }

    def save_history(self, output_path: str):
        """Save full metrics history to JSON file."""
        data = {
            'history': [asdict(m) for m in self.history],
            'summary': self.get_summary(),
            'thresholds': asdict(THRESHOLDS),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Metrics history saved to: {output_path}")


# =============================================================================
# TRAINING CALLBACK INTEGRATION
# =============================================================================

class NoeticMetricsCallback:
    """
    Callback for integration with training frameworks.

    Example with custom training loop:
        callback = NoeticMetricsCallback(model, tokenizer, device)

        for epoch in range(num_epochs):
            train_epoch(...)
            callback.on_epoch_end(epoch)

            if callback.should_stop():
                print("Early stopping due to noetic collapse")
                break
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        early_stop_on_collapse: bool = True,
        collapse_patience: int = 3,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize callback.

        Args:
            model: TKSLLMCorePipeline model
            tokenizer: Tokenizer instance
            device: torch.device
            early_stop_on_collapse: Stop training if collapse detected
            collapse_patience: Number of failing epochs before stopping
            log_dir: Directory to save metrics
        """
        self.logger = NoeticMetricsLogger(
            model, tokenizer, device,
            fast_mode=True,
            log_dir=log_dir
        )
        self.early_stop_on_collapse = early_stop_on_collapse
        self.collapse_patience = collapse_patience
        self._should_stop = False

    def on_epoch_end(self, epoch: int) -> EpochMetrics:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number

        Returns:
            EpochMetrics for this epoch
        """
        metrics = self.logger.log_epoch(epoch)

        # Check for collapse
        if self.early_stop_on_collapse:
            if self.logger.detect_collapse(lookback=self.collapse_patience):
                print(f"\nWARNING: Noetic collapse detected after {self.collapse_patience} failing epochs!")
                print("Consider reducing learning rate or reverting to earlier checkpoint.")
                self._should_stop = True

        return metrics

    def should_stop(self) -> bool:
        """Check if training should stop due to collapse."""
        return self._should_stop

    def get_best_checkpoint_info(self) -> Optional[Dict]:
        """Get info about the best checkpoint."""
        if self.logger.best_metrics:
            return {
                'epoch': self.logger.best_metrics.epoch,
                'all_passed': self.logger.best_metrics.all_passed,
            }
        return None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_training_integration():
    """
    Example showing how to integrate noetic metrics into a training loop.

    This is documentation - not meant to be run directly.
    """
    example_code = '''
    # In your training script:
    from training.noetic_metrics import NoeticMetricsCallback

    # Initialize callback
    callback = NoeticMetricsCallback(
        model=model,
        tokenizer=tokenizer,
        device=device,
        early_stop_on_collapse=True,
        collapse_patience=3,
        log_dir="output/training_run/metrics"
    )

    # Training loop
    for epoch in range(num_epochs):
        # Training step
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()

        # Evaluate noetic coherence at end of epoch
        metrics = callback.on_epoch_end(epoch)

        # Save checkpoint if metrics pass
        if metrics.all_passed:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'noetic_metrics': {
                    'world_separation': metrics.world_separation,
                    'rpm_variance': metrics.rpm_variance,
                    'attractor_convergence': metrics.attractor_convergence,
                    'noetic_consistency': metrics.noetic_consistency,
                }
            }, f"checkpoint_epoch_{epoch}_passed.pt")

        # Early stopping on collapse
        if callback.should_stop():
            print("Training stopped due to noetic collapse")
            break

    # Save metrics history
    callback.logger.save_history("output/training_run/noetic_metrics.json")
    '''
    print(example_code)


if __name__ == "__main__":
    print("Noetic Metrics Logger - Training Integration Module")
    print("=" * 60)
    print("\nExample usage:")
    example_training_integration()

#!/usr/bin/env python3
"""
TKS Training Pipeline Smoke Test - Phase 3

This script provides comprehensive end-to-end smoke testing for the TKS training
pipeline. It validates that all components work together correctly:

1. Data loading from augmented JSONL
2. TKSTokenizer tokenization
3. TKSAugmentedDataset creation
4. TKSLLMCorePipeline model initialization
5. Forward pass through all 5 layers
6. TKSLoss computation (all components)
7. Backward pass and optimizer step
8. Evaluation on held-out data
9. Checkpoint save/load cycle
10. Metrics logging

Usage:
    # Run full smoke test
    python smoke_test_training.py

    # Run with custom data
    python smoke_test_training.py --data output/sample_augmented.jsonl

    # Run with verbose output
    python smoke_test_training.py --verbose

Exit Codes:
    0: All tests passed
    1: One or more tests failed

Author: TKS-LLM Training Integration Team
Date: 2025-12-14
Version: 3.0.0 (Phase 3 - Complete Pipeline Smoke Test)
"""

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TKS components
try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    from training.losses import TKSLoss, TKSLossConfig
    from training.trainer import TrainingConfig, TKSTrainer
    REAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Import error: {e}")
    REAL_MODEL_AVAILABLE = False
    TOTAL_DIM = 40

from train_with_augmented import (
    TKSTokenizer,
    TKSAugmentedDataset,
    TrainingMetricsLogger,
    evaluate_model,
    PAD_TOKEN,
)


# ==============================================================================
# TEST UTILITIES
# ==============================================================================

class TestResult:
    """Container for individual test results."""

    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details

    def __str__(self):
        status = "[PASS]" if self.passed else "[FAIL]"
        msg = f" - {self.message}" if self.message else ""
        return f"{status} {self.name}{msg}"


class SmokeTestRunner:
    """Runs smoke tests and collects results."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def log(self, message: str):
        """Log message if verbose mode."""
        if self.verbose:
            print(f"  {message}")

    def add_result(self, result: TestResult):
        """Add test result."""
        self.results.append(result)
        print(result)

    def run_test(self, name: str, test_fn):
        """Run a test function and record result."""
        try:
            result = test_fn()
            if result is True:
                self.add_result(TestResult(name, True))
            elif isinstance(result, tuple):
                self.add_result(TestResult(name, result[0], result[1]))
            else:
                self.add_result(TestResult(name, True, str(result) if result else ""))
            return True
        except Exception as e:
            self.add_result(TestResult(name, False, str(e)))
            if self.verbose:
                traceback.print_exc()
            return False

    def summary(self) -> Tuple[int, int]:
        """Return (passed, total) counts."""
        passed = sum(1 for r in self.results if r.passed)
        return passed, len(self.results)


# ==============================================================================
# SMOKE TESTS
# ==============================================================================

def create_sample_data(output_path: Path) -> None:
    """Create sample augmented JSONL data for testing."""
    sample_entries = [
        {
            "id": "test_001",
            "story": "A spiritual teacher causes enlightenment",
            "expr": "A5 -> D2",
            "expr_elements": ["A5", "D2"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": True,
        },
        {
            "id": "test_002",
            "story": "Mental clarity creates physical action",
            "expr": "B2 -> D5",
            "expr_elements": ["B2", "D5"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": True,
        },
        {
            "id": "test_003",
            "story": "Emotional wisdom transforms into growth",
            "expr": "C2 +T A1",
            "expr_elements": ["C2", "A1"],
            "expr_ops": ["+T"],
            "aug_type": "inversion",
            "validator_pass": True,
        },
        {
            "id": "test_004",
            "story": "Physical force opposes mental resistance",
            "expr": "D8 -T B3",
            "expr_elements": ["D8", "B3"],
            "expr_ops": ["-T"],
            "aug_type": "anti_attractor",
            "validator_pass": True,
        },
        {
            "id": "test_005",
            "story": "Spiritual unity attracts emotional harmony",
            "expr": "A1 +T C5",
            "expr_elements": ["A1", "C5"],
            "expr_ops": ["+T"],
            "aug_type": "original",
            "validator_pass": False,
        },
    ]

    # Duplicate to create enough data for batches
    expanded_entries = []
    for i in range(20):
        for entry in sample_entries:
            new_entry = entry.copy()
            new_entry['id'] = f"{entry['id']}_{i}"
            expanded_entries.append(new_entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in expanded_entries:
            f.write(json.dumps(entry) + '\n')


def run_smoke_tests(data_path: str, verbose: bool = False) -> bool:
    """
    Run complete smoke test suite.

    Args:
        data_path: Path to augmented JSONL data
        verbose: Enable verbose output

    Returns:
        True if all tests passed
    """
    runner = SmokeTestRunner(verbose=verbose)
    device = runner.device

    print("=" * 70)
    print("TKS TRAINING PIPELINE SMOKE TEST")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Data: {data_path}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Real model available: {REAL_MODEL_AVAILABLE}")
    print()

    # Shared state for tests
    state = {}

    # Test 1: Data file existence
    def test_data_exists():
        path = Path(data_path)
        if not path.exists():
            return False, f"Data file not found: {data_path}"
        lines = path.read_text(encoding='utf-8').strip().split('\n')
        state['num_lines'] = len(lines)
        return True, f"Found {len(lines)} entries"

    runner.run_test("Data file exists", test_data_exists)

    # Test 2: Tokenizer initialization
    def test_tokenizer():
        tokenizer = TKSTokenizer(vocab_size=1000, max_length=64)
        state['tokenizer'] = tokenizer
        runner.log(f"Vocab size: {tokenizer.actual_vocab_size}")

        # Test tokenization
        test_text = "A5 -> D2"
        tokens = tokenizer.tokenize(test_text)
        decoded = tokenizer.decode(tokens)
        runner.log(f"Tokenize '{test_text}' -> {len(tokens)} tokens")

        if len(tokens) != 64:  # Should be padded to max_length
            return False, f"Expected 64 tokens, got {len(tokens)}"
        return True

    runner.run_test("Tokenizer initialization", test_tokenizer)

    # Test 3: Dataset loading
    def test_dataset():
        dataset = TKSAugmentedDataset(
            data_path=data_path,
            tokenizer=state['tokenizer'],
            max_length=64,
            filter_validated=False,
        )
        state['dataset'] = dataset
        runner.log(f"Loaded {len(dataset)} entries")

        if len(dataset) == 0:
            return False, "Dataset is empty"

        # Test single item
        item = dataset[0]
        required_keys = ['input_ids', 'targets', 'attention_mask', 'aug_type']
        for key in required_keys:
            if key not in item:
                return False, f"Missing key: {key}"

        return True, f"{len(dataset)} entries loaded"

    runner.run_test("Dataset loading", test_dataset)

    # Test 4: DataLoader creation
    def test_dataloader():
        dataset = state['dataset']
        train_size = max(len(dataset) - 10, int(len(dataset) * 0.8))
        eval_size = len(dataset) - train_size

        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
        state['train_dataset'] = train_dataset
        state['eval_dataset'] = eval_dataset

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

        state['train_loader'] = train_loader
        state['eval_loader'] = eval_loader

        # Get a batch
        batch = next(iter(train_loader))
        state['batch'] = batch

        runner.log(f"Batch shape: {batch['input_ids'].shape}")

        if batch['input_ids'].shape[0] != 4:
            return False, f"Expected batch size 4, got {batch['input_ids'].shape[0]}"

        return True

    runner.run_test("DataLoader creation", test_dataloader)

    # Test 5: Model initialization
    def test_model_init():
        tokenizer = state['tokenizer']

        if REAL_MODEL_AVAILABLE:
            model = TKSLLMCorePipeline(
                vocab_size=tokenizer.actual_vocab_size,
                hidden_dim=64,
                noetic_dim=TOTAL_DIM,
            ).to(device)
            state['model_type'] = 'TKSLLMCorePipeline'
        else:
            model = nn.Sequential(
                nn.Embedding(tokenizer.actual_vocab_size, 64),
                nn.Linear(64, tokenizer.actual_vocab_size),
            ).to(device)
            state['model_type'] = 'FallbackModel'

        state['model'] = model

        num_params = sum(p.numel() for p in model.parameters())
        runner.log(f"Model: {state['model_type']}")
        runner.log(f"Parameters: {num_params:,}")

        return True, f"{state['model_type']} ({num_params:,} params)"

    runner.run_test("Model initialization", test_model_init)

    # Test 6: Forward pass
    def test_forward():
        model = state['model']
        batch = state['batch']

        input_ids = batch['input_ids'].to(device)

        if REAL_MODEL_AVAILABLE and isinstance(model, TKSLLMCorePipeline):
            outputs = model(tokens=input_ids, return_full_trace=True)
            logits = outputs['logits']
            state['outputs'] = outputs

            # Check output keys
            expected_keys = ['logits', 'noetic_embeddings', 'attractor_results', 'rpm_results']
            for key in expected_keys:
                if key not in outputs:
                    runner.log(f"Missing output key: {key}")
        else:
            logits = model(input_ids)
            state['outputs'] = {'logits': logits}

        state['logits'] = logits

        runner.log(f"Logits shape: {logits.shape}")

        if logits.shape[:2] != input_ids.shape:
            return False, f"Shape mismatch: {logits.shape} vs {input_ids.shape}"

        return True

    runner.run_test("Forward pass", test_forward)

    # Test 7: Loss computation
    def test_loss():
        model = state['model']
        batch = state['batch']
        outputs = state['outputs']
        logits = state['logits']

        targets = batch['targets'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        if REAL_MODEL_AVAILABLE and isinstance(model, TKSLLMCorePipeline):
            loss_config = TKSLossConfig()
            loss_fn = TKSLoss(loss_config)

            loss_dict = loss_fn(
                pipeline_output=outputs,
                targets=targets,
                pipeline=model,
                mask=attention_mask,
                compute_all=True,
            )
            loss = loss_dict['total']
            state['loss_fn'] = loss_fn
            state['loss_dict'] = loss_dict

            runner.log(f"Total loss: {loss.item():.4f}")
            for key, val in loss_dict.items():
                if key != 'total' and isinstance(val, torch.Tensor):
                    runner.log(f"  {key}: {val.item():.4f}")
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            state['loss_fn'] = loss_fn
            runner.log(f"Loss: {loss.item():.4f}")

        state['loss'] = loss

        if torch.isnan(loss):
            return False, "Loss is NaN"
        if loss.item() <= 0:
            return False, "Loss should be positive"

        return True, f"loss={loss.item():.4f}"

    runner.run_test("Loss computation", test_loss)

    # Test 8: Backward pass
    def test_backward():
        model = state['model']
        loss = state['loss']

        optimizer = AdamW(model.parameters(), lr=1e-3)
        state['optimizer'] = optimizer

        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break

        if not has_grad:
            return False, "No gradients computed"

        optimizer.step()
        return True

    runner.run_test("Backward pass", test_backward)

    # Test 9: Evaluation function
    def test_evaluation():
        model = state['model']
        eval_loader = state['eval_loader']
        loss_fn = state['loss_fn']

        eval_results = evaluate_model(model, eval_loader, loss_fn, device)

        runner.log(f"Eval loss: {eval_results['loss']:.4f}")
        runner.log(f"Eval accuracy: {eval_results['accuracy']:.4f}")

        if 'loss' not in eval_results:
            return False, "Missing loss in eval results"
        if 'accuracy' not in eval_results:
            return False, "Missing accuracy in eval results"

        return True, f"loss={eval_results['loss']:.4f}, acc={eval_results['accuracy']:.4f}"

    runner.run_test("Evaluation function", test_evaluation)

    # Test 10: Checkpoint save/load
    def test_checkpoint():
        model = state['model']

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

            # Save
            torch.save(model.state_dict(), checkpoint_path)
            runner.log(f"Saved checkpoint: {checkpoint_path}")

            # Verify file exists
            if not checkpoint_path.exists():
                return False, "Checkpoint file not created"

            # Load into new model
            if REAL_MODEL_AVAILABLE and isinstance(model, TKSLLMCorePipeline):
                new_model = TKSLLMCorePipeline(
                    vocab_size=state['tokenizer'].actual_vocab_size,
                    hidden_dim=64,
                    noetic_dim=TOTAL_DIM,
                ).to(device)
            else:
                new_model = nn.Sequential(
                    nn.Embedding(state['tokenizer'].actual_vocab_size, 64),
                    nn.Linear(64, state['tokenizer'].actual_vocab_size),
                ).to(device)

            new_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            runner.log("Loaded checkpoint into new model")

            # Verify outputs match
            batch = state['batch']
            input_ids = batch['input_ids'].to(device)

            model.eval()
            new_model.eval()

            with torch.no_grad():
                if REAL_MODEL_AVAILABLE and isinstance(model, TKSLLMCorePipeline):
                    out1 = model(tokens=input_ids, return_full_trace=True)['logits']
                    out2 = new_model(tokens=input_ids, return_full_trace=True)['logits']
                else:
                    out1 = model(input_ids)
                    out2 = new_model(input_ids)

            if not torch.allclose(out1, out2, atol=1e-5):
                return False, "Outputs differ after checkpoint load"

        return True

    runner.run_test("Checkpoint save/load", test_checkpoint)

    # Test 11: Metrics logger
    def test_metrics_logger():
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = TrainingMetricsLogger(output_dir=Path(tmpdir))

            # Log some data
            metrics.log_step(1, 0, 1.5, 4)
            metrics.log_step(1, 1, 1.4, 4)
            metrics.log_epoch(1, 1.45, [{'aug_type': 'original', 'validator_pass': True}])
            metrics.log_eval(1, {'loss': 1.3, 'accuracy': 0.5})

            # Get summary
            summary = metrics.get_summary()

            if summary['total_steps'] != 2:
                return False, f"Expected 2 steps, got {summary['total_steps']}"

            # Save and verify
            metrics.save()
            metrics_file = Path(tmpdir) / "training_metrics.json"

            if not metrics_file.exists():
                return False, "Metrics file not saved"

            runner.log("Metrics logged and saved successfully")

        return True

    runner.run_test("Metrics logger", test_metrics_logger)

    # Summary
    print()
    print("=" * 70)
    passed, total = runner.summary()
    if passed == total:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"FAILED: {total - passed}/{total} tests failed")
    print("=" * 70)

    return passed == total


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TKS Training Pipeline Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--data', type=str, default=None,
                       help='Path to augmented JSONL data (default: auto-generate)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Use provided data or create temporary test data
    if args.data:
        data_path = args.data
    else:
        # Check for sample data
        default_path = Path(__file__).parent.parent / "output" / "sample_augmented.jsonl"
        if default_path.exists():
            data_path = str(default_path)
        else:
            # Create temporary test data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                temp_path = f.name
            create_sample_data(Path(temp_path))
            data_path = temp_path
            print(f"Created temporary test data: {data_path}")

    # Run tests
    success = run_smoke_tests(data_path, verbose=args.verbose)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

"""
TKS Training/Eval Flow Smoke Test - Phase 4

This smoke test verifies that the end-to-end training and evaluation pipeline works correctly.

Test Coverage:
    1. Data loading (augmented JSONL)
    2. Model initialization (TKSLLMCorePipeline)
    3. Training loop (1-2 epochs, small batch)
    4. Loss computation (TKSLoss multi-component)
    5. Evaluation metrics (accuracy, loss, canonical validation)
    6. Checkpoint saving/loading

Author: TKS-LLM Training Integration Team
Date: 2025-12-14
Version: 1.0.0
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

# Import training components
try:
    from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM
    from training.losses import TKSLoss, TKSLossConfig
    from training.trainer import TrainingConfig, TKSTrainer
    from scripts.train_with_augmented import (
        TKSTokenizer,
        TKSAugmentedDataset,
        evaluate_model,
    )
    from scripts.evaluate_model import CanonicalValidator
    REAL_MODEL_AVAILABLE = True
except ImportError as e:
    REAL_MODEL_AVAILABLE = False
    pytest.skip(f"Real model not available: {e}", allow_module_level=True)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_augmented_data(tmp_path):
    """Create a small sample augmented JSONL file for testing."""
    data_file = tmp_path / "sample_augmented.jsonl"

    # Create sample entries
    entries = [
        {
            "story": "A spiritual teacher causes positive change",
            "expr": "A1 -> B2",
            "expr_elements": ["A1", "B2"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": True,
        },
        {
            "story": "Physical student effects negative resistance",
            "expr": "D1 <- C3",
            "expr_elements": ["D1", "C3"],
            "expr_ops": ["<-"],
            "aug_type": "inversion",
            "validator_pass": True,
        },
        {
            "story": "Mental unity transforms emotional duality",
            "expr": "B1 *T C2",
            "expr_elements": ["B1", "C2"],
            "expr_ops": ["*T"],
            "aug_type": "original",
            "validator_pass": True,
        },
        {
            "story": "Creative power flows through material wisdom",
            "expr": "A5 o D6",
            "expr_elements": ["A5", "D6"],
            "expr_ops": ["o"],
            "aug_type": "anti_attractor",
            "validator_pass": False,
        },
    ] * 5  # Repeat to get 20 entries

    with data_file.open('w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    return data_file


@pytest.fixture
def tokenizer():
    """Create a simple tokenizer for testing."""
    return TKSTokenizer(vocab_size=200, max_length=64)


@pytest.fixture
def small_model(tokenizer):
    """Create a small TKS model for testing."""
    return TKSLLMCorePipeline(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=32,
        noetic_dim=TOTAL_DIM,
        num_scales=2,
        max_attractor_iter=5,
        contraction_factor=0.5,
    )


# ==============================================================================
# TESTS
# ==============================================================================

def test_data_loading(sample_augmented_data, tokenizer):
    """Test that augmented data can be loaded correctly."""
    dataset = TKSAugmentedDataset(
        data_path=str(sample_augmented_data),
        tokenizer=tokenizer,
        max_length=64,
        filter_validated=False,
        use_expr=False,
    )

    assert len(dataset) == 20, "Dataset should have 20 entries"

    # Check first entry
    sample = dataset[0]
    assert 'input_ids' in sample
    assert 'targets' in sample
    assert 'attention_mask' in sample
    assert 'aug_type' in sample
    assert 'validator_pass' in sample

    assert sample['input_ids'].shape == (64,)
    assert sample['aug_type'] == 'original'
    assert sample['validator_pass'] == True


def test_model_initialization(small_model):
    """Test that model can be initialized correctly."""
    assert small_model is not None

    # Count parameters
    num_params = sum(p.numel() for p in small_model.parameters())
    assert num_params > 0, "Model should have parameters"

    # Test forward pass
    batch_size = 2
    seq_len = 8
    vocab_size = 200

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = small_model(tokens, return_full_trace=True)

    assert 'logits' in output
    assert 'gated_output' in output
    assert output['logits'].shape == (batch_size, seq_len, vocab_size)


def test_loss_computation(small_model):
    """Test that TKSLoss can compute loss correctly."""
    batch_size = 2
    seq_len = 8
    vocab_size = 200

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = small_model(tokens, return_full_trace=True)

    loss_config = TKSLossConfig(
        lambda_task=1.0,
        lambda_rpm=0.1,
        lambda_attractor=0.1,
    )
    loss_fn = TKSLoss(loss_config)

    loss_dict = loss_fn(
        pipeline_output=output,
        targets=targets,
        pipeline=small_model,
        compute_all=True,
    )

    assert 'total' in loss_dict
    assert 'task' in loss_dict
    assert loss_dict['total'].item() > 0
    assert not torch.isnan(loss_dict['total'])


def test_training_smoke(sample_augmented_data, tokenizer, small_model, tmp_path):
    """Smoke test for training loop (1 epoch, small dataset)."""
    # Create dataset
    dataset = TKSAugmentedDataset(
        data_path=str(sample_augmented_data),
        tokenizer=tokenizer,
        max_length=64,
        filter_validated=False,
        use_expr=False,
    )

    # Split into train/eval
    train_size = 16
    eval_size = 4
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + eval_size))

    # Create training config
    config = TrainingConfig(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=32,
        noetic_dim=TOTAL_DIM,
        epochs=1,  # Just 1 epoch for smoke test
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        log_every=5,
        eval_every=10,
        checkpoint_every=100,
        save_best=False,
        seed=42,
    )

    # Create trainer
    trainer = TKSTrainer(small_model, config)

    # Run training
    state = trainer.train(train_dataset, eval_dataset)

    # Verify training completed
    assert state.global_step > 0, "Should have completed at least one step"
    assert len(state.loss_history) > 0, "Should have loss history"
    assert state.loss_history[0] > 0, "Initial loss should be positive"

    # Verify checkpoint was created
    final_checkpoint = tmp_path / "checkpoints" / "final.pt"
    assert final_checkpoint.exists(), "Final checkpoint should be saved"


def test_evaluation_smoke(sample_augmented_data, tokenizer, small_model):
    """Smoke test for evaluation (compute metrics on small dataset)."""
    # Create dataset
    dataset = TKSAugmentedDataset(
        data_path=str(sample_augmented_data),
        tokenizer=tokenizer,
        max_length=64,
        filter_validated=False,
        use_expr=False,
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    # Create loss function
    loss_fn = TKSLoss(TKSLossConfig())

    # Run evaluation
    device = torch.device('cpu')
    small_model.to(device)

    results = evaluate_model(small_model, dataloader, loss_fn, device)

    # Verify results
    assert 'loss' in results
    assert 'accuracy' in results
    assert 'perplexity' in results
    assert results['loss'] > 0
    assert 0 <= results['accuracy'] <= 1
    assert results['perplexity'] > 0


def test_canonical_validation_smoke(sample_augmented_data, tokenizer):
    """Smoke test for canonical validation."""
    validator = CanonicalValidator()

    # Test element validation
    result = validator.validate_element("A1")
    assert result['valid'] == True
    assert result['world_valid'] == True
    assert result['noetic_valid'] == True

    # Test invalid element
    result = validator.validate_element("X99")
    assert result['valid'] == False

    # Test operator validation
    assert validator.validate_operator("+") == True
    assert validator.validate_operator("->") == True
    assert validator.validate_operator("invalid") == False

    # Test expression validation
    result = validator.validate_expression(["A1", "B2"], ["->"])
    assert result['valid'] == True
    assert result['world_validity_rate'] == 1.0
    assert result['noetic_validity_rate'] == 1.0
    assert result['operator_validity_rate'] == 1.0


def test_checkpoint_loading(sample_augmented_data, tokenizer, small_model, tmp_path):
    """Test that checkpoints can be saved and loaded correctly."""
    # Create a simple dataset
    dataset = TKSAugmentedDataset(
        data_path=str(sample_augmented_data),
        tokenizer=tokenizer,
        max_length=64,
        filter_validated=False,
        use_expr=False,
    )

    train_dataset = torch.utils.data.Subset(dataset, range(16))

    # Train for 1 epoch and save checkpoint
    config = TrainingConfig(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=32,
        noetic_dim=TOTAL_DIM,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        log_every=5,
        seed=42,
    )

    trainer = TKSTrainer(small_model, config)
    state = trainer.train(train_dataset)

    # Verify checkpoint exists
    checkpoint_path = tmp_path / "checkpoints" / "final.pt"
    assert checkpoint_path.exists()

    # Load checkpoint into new model (must match small_model's architecture)
    new_model = TKSLLMCorePipeline(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=32,
        noetic_dim=TOTAL_DIM,
        num_scales=2,  # Must match small_model fixture
        max_attractor_iter=5,
        contraction_factor=0.5,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_model.load_state_dict(checkpoint['model'])

    # Verify models have same weights
    for (name1, param1), (name2, param2) in zip(
        small_model.named_parameters(),
        new_model.named_parameters()
    ):
        assert name1 == name2
        assert torch.allclose(param1, param2, atol=1e-6)


# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

def test_end_to_end_integration(sample_augmented_data, tokenizer, tmp_path):
    """
    Complete end-to-end integration test:
    1. Load data
    2. Create model
    3. Train for 2 epochs
    4. Evaluate
    5. Save/load checkpoint
    6. Evaluate again
    """
    # 1. Load data
    dataset = TKSAugmentedDataset(
        data_path=str(sample_augmented_data),
        tokenizer=tokenizer,
        max_length=64,
        filter_validated=False,
        use_expr=False,
    )

    train_size = 16
    eval_size = 4
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + eval_size))

    # 2. Create model
    model = TKSLLMCorePipeline(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=32,
        noetic_dim=TOTAL_DIM,
        num_scales=2,
        max_attractor_iter=5,
    )

    # 3. Train for 2 epochs
    config = TrainingConfig(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=32,
        noetic_dim=TOTAL_DIM,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        log_every=5,
        eval_every=10,
        save_best=True,
        seed=42,
    )

    trainer = TKSTrainer(model, config)
    state = trainer.train(train_dataset, eval_dataset)

    # Check that training completed (epoch may be 1-indexed or 0-indexed depending on trainer)
    # With 16 samples and batch_size=4, 2 epochs = 8 steps
    assert state.global_step >= 8, "Should complete 2 epochs worth of steps"
    assert len(state.loss_history) > 0

    # 4. Evaluate
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4)
    loss_fn = TKSLoss(TKSLossConfig())
    device = torch.device('cpu')

    results_1 = evaluate_model(model, eval_loader, loss_fn, device)
    assert results_1['accuracy'] >= 0

    # 5. Save/load checkpoint
    checkpoint_path = tmp_path / "checkpoints" / "final.pt"
    assert checkpoint_path.exists()

    new_model = TKSLLMCorePipeline(
        vocab_size=tokenizer.actual_vocab_size,
        hidden_dim=32,
        noetic_dim=TOTAL_DIM,
        num_scales=2,  # Must match original model
        max_attractor_iter=5,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_model.load_state_dict(checkpoint['model'])

    # 6. Evaluate again with loaded model
    new_model.to(device)
    results_2 = evaluate_model(new_model, eval_loader, loss_fn, device)

    # Results should be identical
    assert abs(results_1['loss'] - results_2['loss']) < 1e-5
    assert abs(results_1['accuracy'] - results_2['accuracy']) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

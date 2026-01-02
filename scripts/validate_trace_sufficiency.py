#!/usr/bin/env python3
"""
Validate Trace Sufficiency for Traceable TKS-Transformer

This script tests whether the model's trace is sufficient to predict its outputs.
This is the critical validation for the traceable transformer approach.

The experiment:
1. Load a trained TKS v4 model
2. Run inference with return_full_trace=True
3. Extract trace features (fixed-size vector per token)
4. Train a small "trace-only" head to predict logits from trace features
5. Measure how well trace predicts actual outputs

Metrics:
- Top-1 Match Rate: Does argmax(trace_pred) == argmax(true_logits)?
- Top-5 Match Rate: Is true argmax in top-5 of trace predictions?
- KL Divergence: Distribution similarity
- Rank Correlation: Do trace predictions rank tokens similarly?

Success Criteria:
- Top-1 match > 50% -> trace captures dominant prediction
- Top-5 match > 80% -> trace captures likely candidates
- KL div < 2.0 -> distributions are similar

Usage:
    # Quick test with synthetic data (no model needed)
    python scripts/validate_trace_sufficiency.py --quick

    # Full validation with trained model
    python scripts/validate_trace_sufficiency.py \
        --model-path output/v4_joint_finetuned.pt \
        --data-jsonl data/equation_embeddings/splits/val.jsonl \
        --epochs 20

    # With synthetic data but real model structure
    python scripts/validate_trace_sufficiency.py \
        --model-path output/v4_joint_finetuned.pt \
        --synthetic-samples 500 \
        --epochs 15

Author: TKS-Quantum Validation Agent
Date: 2025-12-23
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig, TOTAL_DIM
from tks_llm_core_v2 import FOUNDATION_NAMES


# =============================================================================
# CONSTANTS
# =============================================================================

# Trace feature dimensions (from TKS v4 model trace output)
NOETIC_WEIGHT_DIM = 10       # Noetic router weights per block
RPM_GATE_DIM = 1             # RPM gate scalar
DWP_SCORES_DIM = 21          # 7 foundations x 3 (D, W, P)
ATTRACTOR_META_DIM = 3       # converged (bool), iterations, final_delta

# Total trace feature dimension per position
# = noetic_weights * num_blocks + rpm_gate + dwp_scores + attractor_meta + hidden_state
def compute_trace_dim(num_blocks: int, include_hidden: bool = True) -> int:
    """Compute total trace feature dimension."""
    base = NOETIC_WEIGHT_DIM * num_blocks + RPM_GATE_DIM + DWP_SCORES_DIM + ATTRACTOR_META_DIM
    if include_hidden:
        base += TOTAL_DIM  # 40-dim noetic state
    return base


# =============================================================================
# TRACE CONSISTENCY HEAD
# =============================================================================

class TraceConsistencyHead(nn.Module):
    """
    Small neural network that predicts logits from trace features alone.

    If this head can accurately predict the model's outputs, then the trace
    is sufficient for interpretability - we can understand WHY the model
    made its prediction by examining the trace.

    Architecture:
        trace_features -> Linear -> ReLU -> Linear -> ReLU -> Linear -> logits

    Args:
        trace_dim: Input trace feature dimension
        hidden_dim: Hidden layer dimension
        vocab_size: Output vocabulary size
    """

    def __init__(
        self,
        trace_dim: int,
        hidden_dim: int = 128,
        vocab_size: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.trace_dim = trace_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.net = nn.Sequential(
            nn.Linear(trace_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, trace_features: torch.Tensor) -> torch.Tensor:
        """
        Predict logits from trace features.

        Args:
            trace_features: [batch, seq, trace_dim] or [batch * seq, trace_dim]

        Returns:
            Predicted logits [batch, seq, vocab_size] or [batch * seq, vocab_size]
        """
        return self.net(trace_features)


# =============================================================================
# TRACE FEATURE EXTRACTION
# =============================================================================

def extract_trace_features(
    trace: Dict[str, torch.Tensor],
    num_blocks: int,
    include_hidden: bool = True,
    gated_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Extract fixed-size feature vector from model trace.

    The trace contains:
    - embedding: [batch, seq, 40] initial noetic embedding
    - worlds: dict of world embeddings
    - blocks: list of per-block traces with noetic_weights, attn_weights
    - attractor: [batch, seq, 40] converged attractor state
    - attractor_converged: bool
    - attractor_iterations: int
    - rpm_gate: [batch, seq] gate values
    - dwp_scores: [batch, seq, 7, 3] D/W/P scores per foundation

    Args:
        trace: Model trace dictionary from return_full_trace=True
        num_blocks: Number of transformer blocks
        include_hidden: Whether to include hidden state in features
        gated_output: Fallback hidden state if not in trace

    Returns:
        [batch, seq, trace_dim] tensor of trace features
    """
    # Get hidden state - try multiple keys
    hidden_state = None
    for key in ['attractor', 'embedding', 'gated_output']:
        if key in trace and isinstance(trace[key], torch.Tensor):
            hidden_state = trace[key]
            break

    if hidden_state is None and gated_output is not None:
        hidden_state = gated_output

    if hidden_state is None:
        raise ValueError("No valid hidden state found in trace")

    batch_size = hidden_state.shape[0]
    seq_len = hidden_state.shape[1]
    device = hidden_state.device

    features = []

    # 1. Noetic weights from each block [num_blocks * 10]
    if 'blocks' in trace:
        for block_trace in trace['blocks']:
            if 'noetic_weights' in block_trace:
                # [batch, seq, 10] -> append
                nw = block_trace['noetic_weights']
                if nw.dim() == 2:
                    # If only [batch, 10], expand to seq
                    nw = nw.unsqueeze(1).expand(-1, seq_len, -1)
                features.append(nw)
            else:
                # Placeholder if no weights
                features.append(torch.zeros(batch_size, seq_len, NOETIC_WEIGHT_DIM, device=device))
    else:
        # No block traces - use zeros
        for _ in range(num_blocks):
            features.append(torch.zeros(batch_size, seq_len, NOETIC_WEIGHT_DIM, device=device))

    # 2. RPM gate [1]
    if 'rpm_gate' in trace:
        rpm_gate = trace['rpm_gate']
        if rpm_gate.dim() == 2:
            features.append(rpm_gate.unsqueeze(-1))
        else:
            features.append(rpm_gate)
    else:
        features.append(torch.zeros(batch_size, seq_len, 1, device=device))

    # 3. DWP scores [21 = 7 * 3]
    if 'dwp_scores' in trace:
        dwp = trace['dwp_scores']  # [batch, seq, 7, 3]
        dwp_flat = dwp.view(batch_size, seq_len, -1)  # [batch, seq, 21]
        features.append(dwp_flat)
    else:
        features.append(torch.zeros(batch_size, seq_len, DWP_SCORES_DIM, device=device))

    # 4. Attractor metadata [3]
    attractor_meta = torch.zeros(batch_size, seq_len, ATTRACTOR_META_DIM, device=device)

    if 'attractor_converged' in trace:
        converged = float(trace['attractor_converged'])
        attractor_meta[:, :, 0] = converged

    if 'attractor_iterations' in trace:
        iterations = float(trace['attractor_iterations'])
        attractor_meta[:, :, 1] = iterations / 15.0  # Normalize by max iterations

    features.append(attractor_meta)

    # 5. Hidden state (attractor output) [40]
    if include_hidden:
        features.append(hidden_state)

    # Concatenate all features
    trace_features = torch.cat(features, dim=-1)

    return trace_features


def extract_trace_features_simple(
    model_output: Dict[str, torch.Tensor],
    num_blocks: int,
) -> torch.Tensor:
    """
    Simplified trace extraction that works even without full trace.

    Uses gated_output and rpm_gate which are always available.

    Args:
        model_output: Model forward pass output
        num_blocks: Number of blocks (for padding)

    Returns:
        [batch, seq, trace_dim] trace features
    """
    gated = model_output['gated_output']  # [batch, seq, 40]
    batch_size, seq_len, _ = gated.shape
    device = gated.device

    features = []

    # Pad noetic weights with zeros (not available without trace)
    for _ in range(num_blocks):
        features.append(torch.zeros(batch_size, seq_len, NOETIC_WEIGHT_DIM, device=device))

    # RPM gate
    if 'rpm_gate' in model_output:
        rpm_gate = model_output['rpm_gate'].unsqueeze(-1)
        features.append(rpm_gate)
    else:
        features.append(torch.zeros(batch_size, seq_len, 1, device=device))

    # DWP scores placeholder
    features.append(torch.zeros(batch_size, seq_len, DWP_SCORES_DIM, device=device))

    # Attractor metadata
    attractor_meta = torch.zeros(batch_size, seq_len, ATTRACTOR_META_DIM, device=device)
    if 'attractor_converged' in model_output:
        attractor_meta[:, :, 0] = float(model_output['attractor_converged'])
    features.append(attractor_meta)

    # Hidden state
    features.append(gated)

    return torch.cat(features, dim=-1)


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_traces(
    model: TKSNoeticLM,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference and collect traces + true logits.

    Args:
        model: TKS model to evaluate
        dataloader: DataLoader yielding batches with 'input_ids'
        device: Device to run on
        max_batches: Maximum number of batches to process

    Returns:
        Tuple of (trace_features, true_logits)
        - trace_features: [total_tokens, trace_dim]
        - true_logits: [total_tokens, vocab_size]
    """
    model.eval()

    all_trace_features = []
    all_logits = []

    num_blocks = len(model.blocks)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            # Get input tokens
            if isinstance(batch, dict):
                tokens = batch['input_ids'].to(device)
            else:
                tokens = batch[0].to(device)

            # Run forward with full trace
            output = model(tokens, return_full_trace=True)

            # Extract trace features
            if 'trace' in output and output['trace']:
                trace_features = extract_trace_features(
                    output['trace'],
                    num_blocks,
                    include_hidden=True,
                    gated_output=output.get('gated_output'),
                )
            else:
                trace_features = extract_trace_features_simple(
                    output,
                    num_blocks,
                )

            # Get logits
            logits = output['logits']  # [batch, seq, vocab]

            # Flatten batch and sequence dimensions
            batch_size, seq_len, vocab_size = logits.shape
            trace_dim = trace_features.shape[-1]

            trace_features = trace_features.view(-1, trace_dim)
            logits = logits.view(-1, vocab_size)

            all_trace_features.append(trace_features.cpu())
            all_logits.append(logits.cpu())

    # Concatenate all batches
    trace_features = torch.cat(all_trace_features, dim=0)
    true_logits = torch.cat(all_logits, dim=0)

    return trace_features, true_logits


def generate_synthetic_data(
    model: TKSNoeticLM,
    num_samples: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data by running model on random inputs.

    Args:
        model: TKS model
        num_samples: Number of sequences to generate
        seq_len: Sequence length
        device: Device to run on

    Returns:
        Tuple of (trace_features, true_logits)
    """
    model.eval()

    # Create random input tokens
    vocab_size = model.vocab_size
    tokens = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)

    # Create simple dataloader
    dataset = TensorDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=32)

    return collect_traces(model, dataloader, device)


# =============================================================================
# TRAINING
# =============================================================================

def train_trace_head(
    trace_features: torch.Tensor,
    true_logits: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> TraceConsistencyHead:
    """
    Train the trace-only prediction head.

    Args:
        trace_features: [N, trace_dim] trace features
        true_logits: [N, vocab_size] target logits
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        device: Device for training
        verbose: Print progress

    Returns:
        Trained TraceConsistencyHead
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trace_dim = trace_features.shape[-1]
    vocab_size = true_logits.shape[-1]

    # Create model
    head = TraceConsistencyHead(
        trace_dim=trace_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        dropout=dropout,
    ).to(device)

    # Move data to device
    trace_features = trace_features.to(device)
    true_logits = true_logits.to(device)

    # Create dataloader
    dataset = TensorDataset(trace_features, true_logits)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    # Training loop
    head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for trace_batch, logits_batch in dataloader:
            optimizer.zero_grad()

            # Forward pass
            pred_logits = head(trace_batch)

            # KL divergence loss (treating true logits as target distribution)
            log_pred = F.log_softmax(pred_logits, dim=-1)
            target_probs = F.softmax(logits_batch, dim=-1)
            loss = F.kl_div(log_pred, target_probs, reduction='batchmean')

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    head.eval()
    return head


# =============================================================================
# EVALUATION METRICS
# =============================================================================

@dataclass
class SufficiencyMetrics:
    """Metrics for trace sufficiency evaluation."""
    top1_match_rate: float
    top5_match_rate: float
    kl_divergence: float
    rank_correlation: float
    num_samples: int

    def passes_criteria(self) -> Dict[str, bool]:
        """Check if metrics pass success criteria."""
        return {
            'top1': self.top1_match_rate > 0.50,
            'top5': self.top5_match_rate > 0.80,
            'kl_div': self.kl_divergence < 2.0,
        }

    def overall_pass(self) -> bool:
        """Check if all criteria pass."""
        return all(self.passes_criteria().values())


def compute_rank_correlation(pred_logits: torch.Tensor, true_logits: torch.Tensor) -> float:
    """
    Compute Spearman rank correlation between predicted and true logit rankings.

    Args:
        pred_logits: [N, vocab_size] predicted logits
        true_logits: [N, vocab_size] true logits

    Returns:
        Average Spearman correlation coefficient
    """
    # Get rankings
    pred_ranks = pred_logits.argsort(dim=-1, descending=True).argsort(dim=-1).float()
    true_ranks = true_logits.argsort(dim=-1, descending=True).argsort(dim=-1).float()

    # Compute correlation per sample
    # Spearman rho = 1 - 6 * sum(d^2) / (n * (n^2 - 1))
    n = pred_logits.shape[-1]
    d_squared = (pred_ranks - true_ranks) ** 2
    rho = 1 - 6 * d_squared.sum(dim=-1) / (n * (n**2 - 1))

    return rho.mean().item()


def evaluate_sufficiency(
    trace_head: TraceConsistencyHead,
    trace_features: torch.Tensor,
    true_logits: torch.Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> SufficiencyMetrics:
    """
    Compute all sufficiency metrics.

    Args:
        trace_head: Trained trace prediction head
        trace_features: [N, trace_dim] trace features
        true_logits: [N, vocab_size] true logits
        batch_size: Batch size for evaluation
        device: Device for computation

    Returns:
        SufficiencyMetrics dataclass with all metrics
    """
    if device is None:
        device = next(trace_head.parameters()).device

    trace_head.eval()

    # Move to device
    trace_features = trace_features.to(device)
    true_logits = true_logits.to(device)

    # Create dataloader
    dataset = TensorDataset(trace_features, true_logits)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_top1_correct = 0
    total_top5_correct = 0
    total_kl_div = 0.0
    total_rank_corr = 0.0
    total_samples = 0

    with torch.no_grad():
        for trace_batch, logits_batch in dataloader:
            batch_size_actual = trace_batch.shape[0]

            # Predict
            pred_logits = trace_head(trace_batch)

            # Top-1 match
            pred_top1 = pred_logits.argmax(dim=-1)
            true_top1 = logits_batch.argmax(dim=-1)
            top1_correct = (pred_top1 == true_top1).sum().item()

            # Top-5 match
            pred_top5 = pred_logits.topk(5, dim=-1).indices
            true_top1_expanded = true_top1.unsqueeze(-1).expand(-1, 5)
            top5_correct = (pred_top5 == true_top1_expanded).any(dim=-1).sum().item()

            # KL divergence
            log_pred = F.log_softmax(pred_logits, dim=-1)
            target_probs = F.softmax(logits_batch, dim=-1)
            kl_div = F.kl_div(log_pred, target_probs, reduction='batchmean').item()

            # Rank correlation
            rank_corr = compute_rank_correlation(pred_logits, logits_batch)

            # Accumulate
            total_top1_correct += top1_correct
            total_top5_correct += top5_correct
            total_kl_div += kl_div * batch_size_actual
            total_rank_corr += rank_corr * batch_size_actual
            total_samples += batch_size_actual

    return SufficiencyMetrics(
        top1_match_rate=total_top1_correct / total_samples,
        top5_match_rate=total_top5_correct / total_samples,
        kl_divergence=total_kl_div / total_samples,
        rank_correlation=total_rank_corr / total_samples,
        num_samples=total_samples,
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_report(
    metrics: SufficiencyMetrics,
    model_path: str,
    data_source: str,
    trace_dim: int,
) -> None:
    """Print formatted report of trace sufficiency validation."""
    print()
    print("=" * 70)
    print("TRACE SUFFICIENCY VALIDATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data: {metrics.num_samples} samples from {data_source}")
    print(f"Trace feature dim: {trace_dim}")
    print()

    # Results
    print("RESULTS")
    print("-" * 70)

    passes = metrics.passes_criteria()

    # Top-1 Match Rate
    status = "PASS > 50%" if passes['top1'] else "FAIL <= 50%"
    print(f"Top-1 Match Rate:  {metrics.top1_match_rate * 100:5.1f}%  [{status}]")

    # Top-5 Match Rate
    status = "PASS > 80%" if passes['top5'] else "FAIL <= 80%"
    print(f"Top-5 Match Rate:  {metrics.top5_match_rate * 100:5.1f}%  [{status}]")

    # KL Divergence
    status = "PASS < 2.0" if passes['kl_div'] else "FAIL >= 2.0"
    print(f"KL Divergence:     {metrics.kl_divergence:5.2f}   [{status}]")

    # Rank Correlation
    quality = "EXCELLENT" if metrics.rank_correlation > 0.8 else \
              "GOOD" if metrics.rank_correlation > 0.6 else \
              "FAIR" if metrics.rank_correlation > 0.4 else "POOR"
    print(f"Rank Correlation:  {metrics.rank_correlation:5.2f}   [{quality}]")

    print()

    # Verdict
    if metrics.overall_pass():
        print("VERDICT: TRACE IS SUFFICIENT")
        print("The traceable transformer approach is viable.")
    else:
        print("VERDICT: TRACE IS INSUFFICIENT")
        print("The trace does not fully capture model behavior.")
        print("Consider:")
        print("  - Adding more trace signals (attention patterns, intermediate states)")
        print("  - Training longer or with more data")
        print("  - Reviewing model architecture for hidden computations")

    print("=" * 70)


def save_results(
    metrics: SufficiencyMetrics,
    model_path: str,
    data_source: str,
    trace_dim: int,
    output_path: Path,
) -> None:
    """Save results to JSON file."""
    results = {
        'model_path': model_path,
        'data_source': data_source,
        'trace_dim': trace_dim,
        'num_samples': metrics.num_samples,
        'metrics': {
            'top1_match_rate': metrics.top1_match_rate,
            'top5_match_rate': metrics.top5_match_rate,
            'kl_divergence': metrics.kl_divergence,
            'rank_correlation': metrics.rank_correlation,
        },
        'criteria_pass': metrics.passes_criteria(),
        'overall_pass': metrics.overall_pass(),
        'verdict': 'SUFFICIENT' if metrics.overall_pass() else 'INSUFFICIENT',
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# QUICK TEST MODE
# =============================================================================

def run_quick_test(device: torch.device) -> SufficiencyMetrics:
    """
    Run quick test with synthetic model and data.

    This mode creates a small model and synthetic data to verify
    the validation pipeline works correctly.
    """
    print("\n--- QUICK TEST MODE ---")
    print("Creating synthetic model and data...")

    # Create small model
    config = TKSNoeticLMConfig(
        vocab_size=256,
        max_seq_len=32,
        num_layers=2,
        num_scales=2,
        dropout=0.0,
        use_attractor=True,
        use_stable_attractor=True,
        use_rpm=True,
    )
    model = TKSNoeticLM(config).to(device)

    # Generate synthetic data
    num_samples = 200
    seq_len = 16

    print(f"Generating {num_samples} synthetic sequences...")
    trace_features, true_logits = generate_synthetic_data(
        model, num_samples, seq_len, device
    )

    trace_dim = trace_features.shape[-1]
    print(f"Trace feature dimension: {trace_dim}")
    print(f"Total tokens: {trace_features.shape[0]}")

    # Split train/test
    n_train = int(0.8 * trace_features.shape[0])
    train_features = trace_features[:n_train]
    train_logits = true_logits[:n_train]
    test_features = trace_features[n_train:]
    test_logits = true_logits[n_train:]

    print(f"\nTraining trace-only head...")
    head = train_trace_head(
        train_features,
        train_logits,
        epochs=15,
        batch_size=32,
        lr=1e-3,
        hidden_dim=64,
        device=device,
        verbose=True,
    )

    print(f"\nEvaluating sufficiency...")
    metrics = evaluate_sufficiency(
        head,
        test_features,
        test_logits,
        device=device,
    )

    return metrics, trace_dim


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate trace sufficiency for traceable TKS-Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (no model needed)
    python scripts/validate_trace_sufficiency.py --quick

    # Full validation with model
    python scripts/validate_trace_sufficiency.py \\
        --model-path output/v4_joint_finetuned.pt \\
        --data-jsonl data/equation_embeddings/splits/val.jsonl

    # Synthetic data with real model
    python scripts/validate_trace_sufficiency.py \\
        --model-path output/v4_joint_finetuned.pt \\
        --synthetic-samples 500
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with synthetic model and data'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained TKS v4 model checkpoint'
    )

    parser.add_argument(
        '--data-jsonl',
        type=str,
        help='Path to validation data JSONL file'
    )

    parser.add_argument(
        '--synthetic-samples',
        type=int,
        default=0,
        help='Number of synthetic samples to generate (if no --data-jsonl)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs for trace head (default: 20)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training/evaluation (default: 64)'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Hidden dimension for trace head (default: 128)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/trace_sufficiency_results.json',
        help='Path to save results JSON'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto/cpu/cuda)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Quick test mode
    if args.quick:
        metrics, trace_dim = run_quick_test(device)
        print_report(
            metrics,
            model_path="[synthetic model]",
            data_source="[synthetic data]",
            trace_dim=trace_dim,
        )
        return

    # Full validation mode
    if not args.model_path:
        print("ERROR: --model-path is required for full validation")
        print("Use --quick for a quick test without a trained model")
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            config = TKSNoeticLMConfig(**config)
    else:
        # Default config
        config = TKSNoeticLMConfig()

    model = TKSNoeticLM(config).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model loaded: {config.vocab_size} vocab, {config.num_layers} layers")

    # Get data
    if args.data_jsonl:
        data_path = Path(args.data_jsonl)
        if not data_path.exists():
            print(f"ERROR: Data file not found: {data_path}")
            sys.exit(1)

        print(f"\nLoading data from: {data_path}")

        # Load JSONL data
        examples = []
        with open(data_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))

        # Create tokens from examples (assuming 'input_ids' or 'elements' field)
        tokens_list = []
        for ex in examples:
            if 'input_ids' in ex:
                tokens_list.append(torch.tensor(ex['input_ids']))
            elif 'elements' in ex:
                # Convert element names to token IDs (simplified)
                tokens = [hash(e) % config.vocab_size for e in ex['elements']]
                tokens_list.append(torch.tensor(tokens[:config.max_seq_len]))

        # Pad to uniform length
        max_len = min(max(len(t) for t in tokens_list), config.max_seq_len)
        tokens_padded = []
        for t in tokens_list:
            if len(t) < max_len:
                t = F.pad(t, (0, max_len - len(t)), value=0)
            else:
                t = t[:max_len]
            tokens_padded.append(t)

        tokens = torch.stack(tokens_padded).to(device)
        dataset = TensorDataset(tokens)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)

        data_source = f"{len(examples)} samples from {data_path.name}"

        print(f"Collecting traces from {len(examples)} examples...")
        trace_features, true_logits = collect_traces(model, dataloader, device)

    elif args.synthetic_samples > 0:
        print(f"\nGenerating {args.synthetic_samples} synthetic samples...")
        trace_features, true_logits = generate_synthetic_data(
            model,
            args.synthetic_samples,
            config.max_seq_len // 4,  # Shorter sequences for efficiency
            device,
        )
        data_source = f"{args.synthetic_samples} synthetic sequences"

    else:
        print("ERROR: Either --data-jsonl or --synthetic-samples is required")
        sys.exit(1)

    trace_dim = trace_features.shape[-1]
    total_tokens = trace_features.shape[0]
    print(f"Trace feature dimension: {trace_dim}")
    print(f"Total tokens: {total_tokens}")

    # Split train/test
    n_train = int(0.8 * total_tokens)
    train_features = trace_features[:n_train]
    train_logits = true_logits[:n_train]
    test_features = trace_features[n_train:]
    test_logits = true_logits[n_train:]

    print(f"\nTraining trace-only head ({n_train} train, {total_tokens - n_train} test)...")
    head = train_trace_head(
        train_features,
        train_logits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        hidden_dim=args.hidden_dim,
        device=device,
        verbose=True,
    )

    print(f"\nEvaluating sufficiency...")
    metrics = evaluate_sufficiency(
        head,
        test_features,
        test_logits,
        batch_size=args.batch_size,
        device=device,
    )

    # Print report
    print_report(
        metrics,
        model_path=str(model_path),
        data_source=data_source,
        trace_dim=trace_dim,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(
        metrics,
        model_path=str(model_path),
        data_source=data_source,
        trace_dim=trace_dim,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

"""
Train TKS NL→Equation Retriever

Trains a lightweight linear retriever using contrastive learning (InfoNCE/NT-Xent).

Dataset:
    - nl_corpus.pt: Contains NL embeddings and paired equation indices
    - equation_corpus.pt: Contains equation embeddings and metadata

Training:
    - Contrastive loss: positive pairs (NL ↔ equation), negatives (other equations in batch)
    - Batch size: 128
    - Learning rate: 1e-3 with cosine decay
    - Epochs: 50
    - Temperature: 0.07

Evaluation:
    - Recall@1, Recall@5, Recall@10
    - Triplet accuracy (exact match of left_idx, right_idx, operator_idx)

Output:
    - Checkpoint saved to: output/nl_retriever.pt
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tks_nl_retriever import TKSNLRetriever, InfoNCELoss


class NLEquationDataset(Dataset):
    """
    Dataset for NL→Equation retrieval training.

    Expected data format:
        nl_corpus.pt: {
            'embeddings': Tensor of shape (n, nl_dim),
            'equation_indices': List[int] or Tensor of length n,
            'texts': List[str] (optional)
        }
        equation_corpus.pt: {
            'embeddings': Tensor of shape (m, eq_dim),
            'metadata': List[Dict] with keys 'left_idx', 'right_idx', 'operator_idx',
            'equations': List[str] (optional)
        }
    """

    def __init__(self, nl_corpus_path: str, equation_corpus_path: str):
        self.nl_data = torch.load(nl_corpus_path)
        self.eq_data = torch.load(equation_corpus_path)

        self.nl_embeddings = self.nl_data['embeddings']
        self.equation_indices = self.nl_data.get('equation_indices', [])

        if isinstance(self.equation_indices, torch.Tensor):
            self.equation_indices = self.equation_indices.tolist()

        self.eq_embeddings = self.eq_data['embeddings']
        self.eq_metadata = self.eq_data.get('metadata', [])

        # Validate sizes
        assert len(self.nl_embeddings) == len(self.equation_indices), \
            f"NL embeddings ({len(self.nl_embeddings)}) != equation indices ({len(self.equation_indices)})"

        print(f"Loaded {len(self.nl_embeddings)} NL samples")
        print(f"Loaded {len(self.eq_embeddings)} equation embeddings")

    def __len__(self) -> int:
        return len(self.nl_embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            nl_embedding: NL embedding tensor
            eq_embedding: Corresponding equation embedding
            eq_idx: Index of equation in corpus
        """
        nl_emb = self.nl_embeddings[idx]
        eq_idx = self.equation_indices[idx]
        eq_emb = self.eq_embeddings[eq_idx]

        return nl_emb, eq_emb, eq_idx


def compute_recall_at_k(
    retriever: TKSNLRetriever,
    nl_embeddings: torch.Tensor,
    equation_corpus: torch.Tensor,
    ground_truth_indices: List[int],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Recall@K metrics.

    Args:
        retriever: Trained retriever model
        nl_embeddings: NL embeddings to query
        equation_corpus: Full equation corpus
        ground_truth_indices: True equation indices for each NL embedding
        k_values: List of K values to compute recall for

    Returns:
        Dictionary mapping 'recall@K' to recall value
    """
    retriever.eval()
    results = {}

    with torch.no_grad():
        # Get top-max(k) predictions for all samples
        max_k = max(k_values)
        top_indices, _ = retriever.retrieve(nl_embeddings, equation_corpus, top_k=max_k)

        # top_indices: (num_samples, max_k)
        ground_truth = torch.tensor(ground_truth_indices, device=top_indices.device).unsqueeze(1)

        for k in k_values:
            # Check if ground truth is in top-k
            top_k_indices = top_indices[:, :k]
            matches = (top_k_indices == ground_truth).any(dim=1)
            recall = matches.float().mean().item()
            results[f'recall@{k}'] = recall

    return results


def compute_triplet_accuracy(
    retriever: TKSNLRetriever,
    nl_embeddings: torch.Tensor,
    equation_corpus: torch.Tensor,
    corpus_metadata: List[Dict],
    ground_truth_indices: List[int]
) -> Dict[str, float]:
    """
    Compute triplet accuracy (exact match of left_idx, right_idx, operator_idx).

    Args:
        retriever: Trained retriever model
        nl_embeddings: NL embeddings to query
        equation_corpus: Full equation corpus
        corpus_metadata: Metadata for each equation in corpus
        ground_truth_indices: True equation indices for each NL embedding

    Returns:
        Dictionary with accuracy metrics
    """
    retriever.eval()

    with torch.no_grad():
        # Retrieve triplets for all samples
        triplets = retriever.batch_retrieve_triplets(
            nl_embeddings,
            equation_corpus,
            corpus_metadata
        )

    # Compare with ground truth
    exact_matches = 0
    left_matches = 0
    right_matches = 0
    op_matches = 0
    valid_samples = 0

    for i, triplet in enumerate(triplets):
        if triplet is None:
            continue

        gt_idx = ground_truth_indices[i]
        if gt_idx >= len(corpus_metadata):
            continue

        gt_meta = corpus_metadata[gt_idx]
        valid_samples += 1

        # Check individual components
        if triplet['left_idx'] == gt_meta.get('left_idx', -1):
            left_matches += 1
        if triplet['right_idx'] == gt_meta.get('right_idx', -1):
            right_matches += 1
        if triplet['operator_idx'] == gt_meta.get('operator_idx', -1):
            op_matches += 1

        # Check exact match
        if (triplet['left_idx'] == gt_meta.get('left_idx', -1) and
            triplet['right_idx'] == gt_meta.get('right_idx', -1) and
            triplet['operator_idx'] == gt_meta.get('operator_idx', -1)):
            exact_matches += 1

    if valid_samples == 0:
        return {
            'triplet_accuracy': 0.0,
            'left_accuracy': 0.0,
            'right_accuracy': 0.0,
            'operator_accuracy': 0.0
        }

    return {
        'triplet_accuracy': exact_matches / valid_samples,
        'left_accuracy': left_matches / valid_samples,
        'right_accuracy': right_matches / valid_samples,
        'operator_accuracy': op_matches / valid_samples
    }


def train_epoch(
    retriever: TKSNLRetriever,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: InfoNCELoss,
    device: str,
    epoch: int
) -> float:
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    retriever.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for nl_emb, eq_emb, _ in pbar:
        nl_emb = nl_emb.to(device)
        eq_emb = eq_emb.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Project NL embeddings
        projected_nl = retriever(nl_emb)

        # Compute contrastive loss
        loss = criterion(projected_nl, eq_emb)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(
    retriever: TKSNLRetriever,
    dataset: NLEquationDataset,
    device: str,
    batch_size: int = 256
) -> Dict[str, float]:
    """
    Evaluate retriever on full dataset.

    Returns:
        Dictionary with all evaluation metrics
    """
    retriever.eval()

    # Prepare data
    nl_embeddings = dataset.nl_embeddings.to(device)
    equation_corpus = dataset.eq_embeddings.to(device)
    ground_truth_indices = dataset.equation_indices
    corpus_metadata = dataset.eq_metadata

    # Compute recall metrics
    recall_metrics = compute_recall_at_k(
        retriever,
        nl_embeddings,
        equation_corpus,
        ground_truth_indices,
        k_values=[1, 5, 10]
    )

    # Compute triplet accuracy
    triplet_metrics = compute_triplet_accuracy(
        retriever,
        nl_embeddings,
        equation_corpus,
        corpus_metadata,
        ground_truth_indices
    )

    # Combine metrics
    metrics = {**recall_metrics, **triplet_metrics}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train TKS NL→Equation Retriever")
    parser.add_argument(
        '--nl_corpus',
        type=str,
        default='output/nl_corpus.pt',
        help='Path to NL corpus file'
    )
    parser.add_argument(
        '--equation_corpus',
        type=str,
        default='output/equation_corpus.pt',
        help='Path to equation corpus file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output directory for checkpoint'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help='Temperature for InfoNCE loss'
    )
    parser.add_argument(
        '--nl_dim',
        type=int,
        default=384,
        help='NL embedding dimension'
    )
    parser.add_argument(
        '--eq_dim',
        type=int,
        default=384,
        help='Equation embedding dimension'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden layer dimension'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.1,
        help='Test split ratio'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("TKS NL→Equation Retriever Training")
    print("=" * 80)
    print(f"NL corpus: {args.nl_corpus}")
    print(f"Equation corpus: {args.equation_corpus}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = NLEquationDataset(args.nl_corpus, args.equation_corpus)

    # Split into train/val/test
    total_size = len(full_dataset)
    test_size = int(total_size * args.test_split)
    val_size = int(total_size * args.val_split)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Initialize model
    print("\nInitializing model...")
    retriever = TKSNLRetriever(
        nl_dim=args.nl_dim,
        eq_dim=args.eq_dim,
        hidden_dim=args.hidden_dim,
        use_hidden=True,
        dropout=args.dropout
    ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in retriever.parameters())}")

    # Initialize loss and optimizer
    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr, weight_decay=1e-5)

    # Cosine learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Training loop
    print("\nStarting training...")
    best_val_recall = 0.0
    best_epoch = 0
    training_history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(
            retriever,
            train_loader,
            optimizer,
            criterion,
            args.device,
            epoch
        )

        print(f"Train loss: {train_loss:.4f}")

        # Validate every 5 epochs or on last epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            print("\nValidating...")

            # Create temporary dataset for validation
            val_nl_embeddings = []
            val_eq_indices = []
            for idx in val_dataset.indices:
                nl_emb, _, eq_idx = full_dataset[idx]
                val_nl_embeddings.append(nl_emb)
                val_eq_indices.append(eq_idx)

            val_nl_embeddings = torch.stack(val_nl_embeddings)

            # Compute metrics
            val_recall = compute_recall_at_k(
                retriever,
                val_nl_embeddings.to(args.device),
                full_dataset.eq_embeddings.to(args.device),
                val_eq_indices,
                k_values=[1, 5, 10]
            )

            val_triplet = compute_triplet_accuracy(
                retriever,
                val_nl_embeddings.to(args.device),
                full_dataset.eq_embeddings.to(args.device),
                full_dataset.eq_metadata,
                val_eq_indices
            )

            print(f"Val Recall@1: {val_recall['recall@1']:.4f}")
            print(f"Val Recall@5: {val_recall['recall@5']:.4f}")
            print(f"Val Recall@10: {val_recall['recall@10']:.4f}")
            print(f"Val Triplet Accuracy: {val_triplet['triplet_accuracy']:.4f}")

            # Save best model
            if val_recall['recall@1'] > best_val_recall:
                best_val_recall = val_recall['recall@1']
                best_epoch = epoch

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': retriever.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_metrics': {**val_recall, **val_triplet},
                    'config': {
                        'nl_dim': args.nl_dim,
                        'eq_dim': args.eq_dim,
                        'hidden_dim': args.hidden_dim,
                        'use_hidden': True,
                        'dropout': args.dropout,
                        'temperature': args.temperature,
                        'lr': args.lr,
                        'batch_size': args.batch_size
                    }
                }

                checkpoint_path = os.path.join(args.output_dir, 'nl_retriever.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"\nSaved best model to {checkpoint_path}")

            # Track history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_recall@1': val_recall['recall@1'],
                'val_recall@5': val_recall['recall@5'],
                'val_recall@10': val_recall['recall@10'],
                'val_triplet_accuracy': val_triplet['triplet_accuracy']
            })

        # Step scheduler
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation Recall@1: {best_val_recall:.4f} at epoch {best_epoch}")
    print("=" * 80)

    # Final evaluation on test set
    print("\nEvaluating on test set...")

    # Load best model
    checkpoint_path = os.path.join(args.output_dir, 'nl_retriever.pt')
    checkpoint = torch.load(checkpoint_path)
    retriever.load_state_dict(checkpoint['model_state_dict'])

    # Prepare test data
    test_nl_embeddings = []
    test_eq_indices = []
    for idx in test_dataset.indices:
        nl_emb, _, eq_idx = full_dataset[idx]
        test_nl_embeddings.append(nl_emb)
        test_eq_indices.append(eq_idx)

    test_nl_embeddings = torch.stack(test_nl_embeddings)

    # Compute test metrics
    test_recall = compute_recall_at_k(
        retriever,
        test_nl_embeddings.to(args.device),
        full_dataset.eq_embeddings.to(args.device),
        test_eq_indices,
        k_values=[1, 5, 10]
    )

    test_triplet = compute_triplet_accuracy(
        retriever,
        test_nl_embeddings.to(args.device),
        full_dataset.eq_embeddings.to(args.device),
        full_dataset.eq_metadata,
        test_eq_indices
    )

    print("\nTest Results:")
    print("-" * 80)
    print(f"Recall@1: {test_recall['recall@1']:.4f}")
    print(f"Recall@5: {test_recall['recall@5']:.4f}")
    print(f"Recall@10: {test_recall['recall@10']:.4f}")
    print(f"Triplet Accuracy: {test_triplet['triplet_accuracy']:.4f}")
    print(f"  Left Accuracy: {test_triplet['left_accuracy']:.4f}")
    print(f"  Right Accuracy: {test_triplet['right_accuracy']:.4f}")
    print(f"  Operator Accuracy: {test_triplet['operator_accuracy']:.4f}")
    print("-" * 80)

    # Save test results
    test_results = {
        'test_metrics': {**test_recall, **test_triplet},
        'best_epoch': best_epoch,
        'best_val_recall@1': best_val_recall,
        'training_history': training_history
    }

    results_path = os.path.join(args.output_dir, 'nl_retriever_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Model checkpoint saved to {checkpoint_path}")


if __name__ == '__main__':
    main()

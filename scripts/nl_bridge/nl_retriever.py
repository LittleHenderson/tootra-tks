#!/usr/bin/env python3
"""
Agent B: Linear NL Retriever

Pure linear NL->equation retriever using contrastive learning.
Maps MiniLM embeddings (384D) to equation corpus via cosine similarity.

Constraints:
- PURE LINEAR (no transformers, no MLPs, no hidden layers)
- Uses same MiniLM model as corpus embeddings
- Contrastive loss with source weighting

Usage:
    python scripts/nl_bridge/nl_retriever.py train
    python scripts/nl_bridge/nl_retriever.py eval
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nl_bridge"
CORPUS_PATH = PROJECT_ROOT / "data" / "equation_embeddings" / "equation_corpus.pt"
MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_CONFIG = {
    'embedding_dim': 384,
    'top_k': 5,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'epochs': 50,
    'patience': 10,
    'temperature_init': 0.07,
    'confidence_threshold': 0.5,
}


# =============================================================================
# LINEAR NL RETRIEVER (PURE LINEAR - NO HIDDEN LAYERS)
# =============================================================================

class LinearNLRetriever(nn.Module):
    """
    Pure linear NL->equation retriever.

    Architecture:
        NL embedding (384D) -> Linear projection (384D) -> cosine sim -> top-k

    This is the ONLY learnable component. No MLPs, no attention, no FFN.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        corpus_path: str = None,
        top_k: int = 5,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.top_k = top_k

        # PURE LINEAR: Single projection matrix, no bias for cosine similarity
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Initialize close to identity for stable start
        nn.init.eye_(self.projection.weight)

        # Learnable temperature for contrastive loss
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(temperature_init))
        )

        # Load corpus embeddings (frozen, not trained)
        self.corpus_loaded = False
        if corpus_path and os.path.exists(corpus_path):
            self._load_corpus(corpus_path)

    def _load_corpus(self, corpus_path: str) -> None:
        """Load and register corpus embeddings as buffers."""
        corpus = torch.load(corpus_path, weights_only=False)

        self.register_buffer('corpus_embeddings', corpus['embeddings'])
        self.register_buffer('corpus_operators', corpus['operators'].long())
        self.register_buffer('corpus_left', corpus['left_indices'].long())
        self.register_buffer('corpus_right', corpus['right_indices'].long())
        self.register_buffer('source_weights', corpus['source_weights'])

        self.corpus_loaded = True
        print(f"Loaded corpus: {corpus['embeddings'].shape[0]} equations, {corpus['embeddings'].shape[1]}D")

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    @property
    def corpus_size(self) -> int:
        return self.corpus_embeddings.shape[0] if self.corpus_loaded else 0

    def forward(
        self,
        nl_embeddings: torch.Tensor,  # [batch, 384]
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve equation triplets from NL embeddings.

        Args:
            nl_embeddings: [batch, 384] from MiniLM encoder

        Returns:
            {
                'left_idx': [batch] LongTensor,
                'right_idx': [batch] LongTensor,
                'operator_idx': [batch] LongTensor,
                'confidence': [batch] FloatTensor (0-1),
                'top_k_indices': [batch, k] LongTensor,
                'top_k_scores': [batch, k] FloatTensor,
            }
        """
        if not self.corpus_loaded:
            raise RuntimeError("Corpus not loaded. Call _load_corpus first.")

        batch_size = nl_embeddings.shape[0]

        # Project and normalize (PURE LINEAR)
        projected = self.projection(nl_embeddings)
        projected = F.normalize(projected, dim=-1)

        # Normalize corpus
        corpus_normed = F.normalize(self.corpus_embeddings, dim=-1)

        # Cosine similarity (scaled by temperature)
        similarity = torch.mm(projected, corpus_normed.T) / self.temperature

        # Top-k retrieval
        top_k_scores, top_k_indices = similarity.topk(self.top_k, dim=-1)

        # Best match (top-1)
        best_idx = top_k_indices[:, 0]

        # Confidence: softmax over top-k, take top-1 probability
        confidence = F.softmax(top_k_scores, dim=-1)[:, 0]

        return {
            'left_idx': self.corpus_left[best_idx],
            'right_idx': self.corpus_right[best_idx],
            'operator_idx': self.corpus_operators[best_idx],
            'confidence': confidence,
            'top_k_indices': top_k_indices,
            'top_k_scores': top_k_scores,
            'best_idx': best_idx,
        }

    def compute_similarity(
        self,
        nl_embeddings: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity to specific corpus entries."""
        projected = self.projection(nl_embeddings)
        projected = F.normalize(projected, dim=-1)

        targets = F.normalize(self.corpus_embeddings[target_indices], dim=-1)
        return (projected * targets).sum(dim=-1) / self.temperature

    def contrastive_loss(
        self,
        nl_embeddings: torch.Tensor,   # [batch, 384]
        target_indices: torch.Tensor,  # [batch] corpus indices
        source_weights: Optional[torch.Tensor] = None,  # [batch] weights
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss with optional source weighting.

        Higher weight for original corpus examples (1.0) vs generated (0.5).
        """
        batch_size = nl_embeddings.shape[0]

        # Project and normalize
        projected = self.projection(nl_embeddings)
        projected = F.normalize(projected, dim=-1)

        # Get target embeddings
        targets = F.normalize(self.corpus_embeddings[target_indices], dim=-1)

        # Positive similarity
        pos_sim = (projected * targets).sum(dim=-1) / self.temperature

        # All-corpus negatives
        corpus_normed = F.normalize(self.corpus_embeddings, dim=-1)
        all_sim = torch.mm(projected, corpus_normed.T) / self.temperature

        # InfoNCE: log(exp(pos) / sum(exp(all)))
        log_softmax = pos_sim - torch.logsumexp(all_sim, dim=-1)

        # Apply source weighting
        if source_weights is not None:
            loss = -(log_softmax * source_weights).mean()
        else:
            loss = -log_softmax.mean()

        return loss

    def get_config(self) -> Dict:
        """Return model configuration."""
        return {
            'embedding_dim': self.embedding_dim,
            'top_k': self.top_k,
            'temperature': self.temperature.item(),
            'corpus_size': self.corpus_size,
        }


# =============================================================================
# DATASET
# =============================================================================

class NLBridgeDataset(Dataset):
    """Dataset for NL->equation training."""

    def __init__(
        self,
        jsonl_path: Path,
        encoder_name: str = 'all-MiniLM-L6-v2',
        cache_embeddings: bool = True,
    ):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.encoder_name = encoder_name
        self.cache_embeddings = cache_embeddings
        self._embeddings_cache = None

        # Load encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(encoder_name)
            self.encoder.eval()
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        # Pre-compute embeddings if caching
        if cache_embeddings:
            self._precompute_embeddings()

    def _precompute_embeddings(self) -> None:
        """Pre-compute all NL embeddings."""
        print(f"Pre-computing {len(self.samples)} embeddings...")
        texts = [s['nl_text'] for s in self.samples]

        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
        self._embeddings_cache = embeddings.cpu()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if self._embeddings_cache is not None:
            embedding = self._embeddings_cache[idx]
        else:
            with torch.no_grad():
                embedding = self.encoder.encode(
                    sample['nl_text'],
                    convert_to_tensor=True,
                )

        return {
            'embedding': embedding,
            'equation_idx': torch.tensor(sample['equation_idx'], dtype=torch.long),
            'left_idx': torch.tensor(sample['left_idx'], dtype=torch.long),
            'right_idx': torch.tensor(sample['right_idx'], dtype=torch.long),
            'operator_idx': torch.tensor(sample['operator_idx'], dtype=torch.long),
            'source_weight': torch.tensor(sample['source_weight'], dtype=torch.float),
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_retriever(
    config: Dict = None,
    verbose: bool = True,
) -> LinearNLRetriever:
    """Train the NL retriever."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_ds = NLBridgeDataset(DATA_DIR / "nl_train.jsonl")
    val_ds = NLBridgeDataset(DATA_DIR / "nl_val.jsonl")

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Initialize model
    model = LinearNLRetriever(
        embedding_dim=config['embedding_dim'],
        corpus_path=str(CORPUS_PATH),
        top_k=config['top_k'],
        temperature_init=config['temperature_init'],
    ).to(device)

    # Optimizer (only projection matrix + temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
    )

    # Training loop
    best_recall = 0.0
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not verbose):
            embeddings = batch['embedding'].to(device)
            targets = batch['equation_idx'].to(device)
            weights = batch['source_weight'].to(device)

            optimizer.zero_grad()
            loss = model.contrastive_loss(embeddings, targets, weights)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        correct_at_1 = 0
        correct_at_5 = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding'].to(device)
                targets = batch['equation_idx'].to(device)

                results = model(embeddings)
                best_idx = results['best_idx']
                top_k = results['top_k_indices']

                correct_at_1 += (best_idx == targets).sum().item()
                correct_at_5 += (top_k == targets.unsqueeze(1)).any(dim=1).sum().item()
                total += len(targets)

        recall_1 = correct_at_1 / total
        recall_5 = correct_at_5 / total

        if verbose:
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, R@1={recall_1:.3f}, R@5={recall_5:.3f}, temp={model.temperature.item():.4f}")

        # Early stopping
        if recall_5 > best_recall:
            best_recall = recall_5
            patience_counter = 0
            # Save best model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "nl_retriever.pt")
            with open(MODEL_DIR / "nl_retriever_config.json", 'w') as f:
                json.dump(model.get_config(), f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(MODEL_DIR / "nl_retriever.pt", weights_only=True))

    print(f"\nBest validation Recall@5: {best_recall:.3f}")
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_retriever(
    model: LinearNLRetriever = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate the NL retriever on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = LinearNLRetriever(
            corpus_path=str(CORPUS_PATH),
        )
        model.load_state_dict(
            torch.load(MODEL_DIR / "nl_retriever.pt", weights_only=True)
        )
    model = model.to(device)
    model.eval()

    test_ds = NLBridgeDataset(DATA_DIR / "nl_test.jsonl")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print(f"Evaluating on {len(test_ds)} test samples...")

    correct_at = {1: 0, 5: 0, 10: 0}
    total = 0
    confidences = []

    with torch.no_grad():
        for batch in tqdm(test_loader, disable=not verbose):
            embeddings = batch['embedding'].to(device)
            targets = batch['equation_idx'].to(device)

            results = model(embeddings)
            top_k = results['top_k_indices']
            conf = results['confidence']

            confidences.extend(conf.cpu().tolist())

            for k in [1, 5, 10]:
                if k <= top_k.shape[1]:
                    correct_at[k] += (top_k[:, :k] == targets.unsqueeze(1)).any(dim=1).sum().item()

            total += len(targets)

    metrics = {
        'recall_at_1': correct_at[1] / total,
        'recall_at_5': correct_at[5] / total,
        'recall_at_10': correct_at.get(10, 0) / total if 10 <= model.top_k else None,
        'mean_confidence': sum(confidences) / len(confidences),
        'test_samples': total,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Recall@1:  {metrics['recall_at_1']:.3f}")
        print(f"Recall@5:  {metrics['recall_at_5']:.3f}")
        if metrics['recall_at_10']:
            print(f"Recall@10: {metrics['recall_at_10']:.3f}")
        print(f"Mean confidence: {metrics['mean_confidence']:.3f}")

    return metrics


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NL Retriever Training/Evaluation")
    parser.add_argument('command', choices=['train', 'eval', 'test'],
                       help="Command to run")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    if args.command == 'train':
        config = DEFAULT_CONFIG.copy()
        config['epochs'] = args.epochs
        config['batch_size'] = args.batch_size
        config['learning_rate'] = args.lr

        model = train_retriever(config)
        evaluate_retriever(model)

    elif args.command in ('eval', 'test'):
        evaluate_retriever()


if __name__ == "__main__":
    main()

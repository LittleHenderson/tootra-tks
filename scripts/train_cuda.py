"""
CUDA-Optimized TKS Training Script

This script provides GPU-optimized training with:
- Automatic mixed precision (FP16/BF16)
- Multi-GPU support via DataParallel or DistributedDataParallel
- Memory optimization with gradient checkpointing
- Efficient data loading with pinned memory
- CUDA memory management

Usage:
    # Single GPU
    python scripts/train_cuda.py --data output/teacher_augmented.jsonl --epochs 10

    # Multi-GPU (DataParallel)
    python scripts/train_cuda.py --data output/teacher_augmented.jsonl --multi-gpu

    # Distributed (for multi-node or multi-GPU)
    torchrun --nproc_per_node=2 scripts/train_cuda.py --data output/teacher_augmented.jsonl --distributed
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TKS LLM Core Pipeline (v2 - 92k params)
try:
    from tks_llm_core_v2 import (
        TKSLLMCorePipeline,
        StableAttractorLayer,
        AttractorComputationLayer,
        attractor_convergence_loss,
        compute_lipschitz_penalty,
    )
    TKS_PIPELINE_AVAILABLE = True
except ImportError:
    TKS_PIPELINE_AVAILABLE = False
    logger.warning("TKSLLMCorePipeline not available. Only 'simple' model type will work.")

# Import TKS Noetic LM v4 (136k params) - Full decoder-only noetic model
try:
    from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
    TKS_NOETIC_V4_AVAILABLE = True
except ImportError:
    TKS_NOETIC_V4_AVAILABLE = False
    logger.warning("TKSNoeticLM v4 not available.")

# Import World/RPM auxiliary losses
try:
    from training.losses import (
        WorldClassificationLoss, RPMDifferentiationLoss, WorldRPMSupervisedLoss,
        NoeticContrastiveLoss, WorldOrthogonalityLoss,
        WorldUsageKLLoss, FocalWorldLoss, LearnedWorldHead, WorldClassificationLossV2,
        HybridWorldHead, WorldCenterLoss, WorldClassificationLossV3,
        RPMDifferentiationLossV2
    )
    AUXILIARY_LOSSES_AVAILABLE = True
except ImportError:
    AUXILIARY_LOSSES_AVAILABLE = False
    logger.warning("Auxiliary losses not available. World/RPM supervision disabled.")


def check_cuda_setup() -> Dict[str, Any]:
    """Check CUDA setup and return device info."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
        "bf16_supported": False,
        "tf32_supported": False,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            })

            # Check for BF16 support (Ampere+, compute >= 8.0)
            if props.major >= 8:
                info["bf16_supported"] = True
                info["tf32_supported"] = True

    return info


# NaN/Inf guard utilities for training stability
class NaNGuard:
    """
    NaN/Inf detection and logging for training stability.

    Tracks problematic batches and optionally dumps them to JSONL for analysis.
    """

    def __init__(self, dump_path: Optional[str] = None, max_dumps: int = 100):
        """
        Args:
            dump_path: Path to JSONL file for dumping problematic batches
            max_dumps: Maximum number of batches to dump
        """
        self.dump_path = dump_path
        self.max_dumps = max_dumps
        self.nan_count = 0
        self.inf_count = 0
        self.skipped_batches = []

        if dump_path:
            Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
            # Clear existing file
            with open(dump_path, 'w') as f:
                pass

    def check_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if tensor contains NaN or Inf. Returns True if problematic."""
        if tensor is None:
            return False
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan:
            self.nan_count += 1
        if has_inf:
            self.inf_count += 1
        return has_nan or has_inf

    def check_loss(self, loss: torch.Tensor, batch_idx: int, epoch: int) -> bool:
        """
        Check if loss is NaN or Inf.

        Returns:
            True if loss is problematic (should skip batch)
        """
        if self.check_tensor(loss, "loss"):
            self.skipped_batches.append({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'loss_value': loss.item() if not torch.isnan(loss) else 'nan'
            })
            return True
        return False

    def dump_batch(self, batch: Dict[str, Any], batch_idx: int, epoch: int, loss_components: Dict[str, float]):
        """Dump problematic batch to JSONL for analysis."""
        if self.dump_path is None:
            return
        if len(self.skipped_batches) > self.max_dumps:
            return

        dump_entry = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'loss_components': loss_components,
        }

        # Convert tensors to lists for JSON serialization
        if 'input_ids' in batch:
            dump_entry['input_ids'] = batch['input_ids'][:2].cpu().tolist()  # First 2 samples
        if 'targets' in batch:
            dump_entry['targets'] = batch['targets'][:2].cpu().tolist()
        if 'world_label' in batch:
            dump_entry['world_label'] = batch['world_label'][:2].cpu().tolist()
        if 'rpm_label' in batch:
            dump_entry['rpm_label'] = batch['rpm_label'][:2].cpu().tolist()

        try:
            with open(self.dump_path, 'a') as f:
                f.write(json.dumps(dump_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to dump batch: {e}")

    def log_summary(self):
        """Log summary of NaN/Inf detections."""
        if self.nan_count > 0 or self.inf_count > 0:
            logger.warning(f"NaN/Inf Summary: {self.nan_count} NaN, {self.inf_count} Inf detected")
            logger.warning(f"Skipped {len(self.skipped_batches)} batches")
            if self.dump_path:
                logger.info(f"Problematic batches dumped to: {self.dump_path}")


def run_per_epoch_eval(
    model_path: str,
    output_dir: str,
    epoch: int,
    test_script: str = 'tests/test_noetic_regression.py'
) -> Dict[str, Any]:
    """
    Run equation + NL evaluation and return metrics.

    Returns dict with:
        - epoch: int
        - equation_metrics: dict with world_separation, rpm_variance, attractor_convergence, noetic_consistency
        - nl_metrics: dict with same keys
    """
    import subprocess

    metrics_dir = os.path.join(output_dir, 'per_epoch_metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    result = {'epoch': epoch, 'equation_metrics': {}, 'nl_metrics': {}}

    for test_style in ['equation', 'nl']:
        metrics_path = os.path.join(metrics_dir, f'epoch_{epoch:03d}_{test_style}.json')

        cmd = [
            'python3', test_script,
            '--model', model_path,
            '--test-style', test_style,
            '--output', metrics_path
        ]

        try:
            # Run evaluation subprocess with timeout
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})

                    extracted = {
                        'world_separation': metrics.get('world_separation', {}).get('correct_rate'),
                        'rpm_variance': metrics.get('rpm_differentiation', {}).get('variance'),
                        'attractor_convergence': metrics.get('attractor_convergence', {}).get('convergence_rate'),
                        'noetic_consistency': metrics.get('noetic_consistency', {}).get('separation'),
                    }
                    result[f'{test_style}_metrics'] = extracted
            else:
                logger.warning(f"Eval metrics not saved: {metrics_path}")

        except subprocess.TimeoutExpired:
            logger.warning(f"Eval timeout for test_style={test_style}")
        except Exception as e:
            logger.warning(f"Eval error for test_style={test_style}: {e}")

    # Append to epoch history JSONL
    history_path = os.path.join(output_dir, 'epoch_eval_history.jsonl')
    with open(history_path, 'a') as f:
        f.write(json.dumps(result) + '\n')

    return result


def log_epoch_eval_metrics(metrics: Dict[str, Any]):
    """Log per-epoch eval metrics in a compact format."""
    epoch = metrics.get('epoch', 0)
    eq = metrics.get('equation_metrics', {})
    nl = metrics.get('nl_metrics', {})

    def fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"

    logger.info(f"  Per-Epoch Eval (epoch {epoch}):")
    logger.info(f"    Equation: world_sep={fmt(eq.get('world_separation'))} | "
                f"rpm_var={fmt(eq.get('rpm_variance'))} | "
                f"attr_conv={fmt(eq.get('attractor_convergence'))} | "
                f"noetic={fmt(eq.get('noetic_consistency'))}")
    logger.info(f"    NL:       world_sep={fmt(nl.get('world_separation'))} | "
                f"rpm_var={fmt(nl.get('rpm_variance'))} | "
                f"attr_conv={fmt(nl.get('attractor_convergence'))} | "
                f"noetic={fmt(nl.get('noetic_consistency'))}")


class SimpleTokenizer:
    """Simple tokenizer for TKS data."""

    def __init__(self, vocab_size=1000, max_length=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}

        next_id = 4
        # TKS elements
        for world in ['A', 'B', 'C', 'D']:
            for noetic in range(1, 11):
                self.token_to_id[f"{world}{noetic}"] = next_id
                next_id += 1

        # Operators
        for op in ['+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o']:
            self.token_to_id[op] = next_id
            next_id += 1

        # Characters (including *, /, <, > for TKS operators, and common text chars)
        for c in ' .,!?;:\'"()-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789^_*/<>=':
            if c not in self.token_to_id:
                self.token_to_id[c] = next_id
                next_id += 1

        self.actual_vocab_size = next_id

    def tokenize(self, text: str) -> list:
        tokens = [2]  # BOS
        for c in text[:self.max_length - 2]:
            tokens.append(self.token_to_id.get(c, 1))
        tokens.append(3)  # EOS
        while len(tokens) < self.max_length:
            tokens.append(0)
        return tokens[:self.max_length]


class TKSDataset(Dataset):
    """TKS training dataset supporting both LM and seq2seq formats."""

    def __init__(self, data_path: str, tokenizer: SimpleTokenizer, seq2seq: bool = False):
        self.tokenizer = tokenizer
        self.entries = []
        self.seq2seq = seq2seq
        self.sep_token = " => "  # Separator between input and target

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.entries.append(entry)

        # Auto-detect seq2seq format if not explicitly set
        if self.entries and not seq2seq:
            first = self.entries[0]
            if 'input' in first and 'target' in first:
                self.seq2seq = True
                print(f"  Auto-detected seq2seq format in {data_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        if self.seq2seq:
            # Seq2seq format: input => target
            inp = entry.get('input', '')
            tgt = entry.get('target', '')
            # Concatenate: input => target
            full_text = inp + self.sep_token + tgt
            input_ids = self.tokenizer.tokenize(full_text)

            # Target is next-token prediction
            targets = input_ids[1:] + [0]

            # Find where target tokens start by counting actual characters
            # (before padding is applied by tokenizer)
            inp_char_len = len(inp) + len(self.sep_token)

            # Create loss mask: 0 for input+sep tokens, 1 for target tokens
            # targets[i] corresponds to string character i (after BOS offset)
            # inp+sep spans characters 0 to inp_char_len-1, target starts at inp_char_len
            loss_mask = []
            for i in range(len(targets)):
                # We want loss only for predicting target characters (index >= inp_char_len)
                if i >= inp_char_len:
                    loss_mask.append(1.0)
                else:
                    loss_mask.append(0.0)

            # Ensure mask matches targets length
            while len(loss_mask) < len(targets):
                loss_mask.append(0.0)
            loss_mask = loss_mask[:len(targets)]

            result = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'targets': torch.tensor(targets, dtype=torch.long),
                'loss_mask': torch.tensor(loss_mask, dtype=torch.float),
            }

            # Include world/RPM labels if present in data
            if 'world_label' in entry:
                world_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                result['world_label'] = torch.tensor(world_map.get(entry['world_label'], 0), dtype=torch.long)
            if 'rpm_label' in entry:
                rpm_map = {'desire': 0, 'wisdom': 1, 'power': 2}
                result['rpm_label'] = torch.tensor(rpm_map.get(entry['rpm_label'], 0), dtype=torch.long)

            return result
        else:
            # LM format: story only (next-token prediction)
            story = entry.get('story', entry.get('text', ''))
            input_ids = self.tokenizer.tokenize(story)
            targets = input_ids[1:] + [0]

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'targets': torch.tensor(targets, dtype=torch.long),
            }


class TransformerModel(nn.Module):
    """Transformer model optimized for GPU training."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(512, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=True  # CUDA optimization
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        device = x.device

        # Embeddings
        tok_emb = self.embedding(x)
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device, dtype=x.dtype
        )

        # Transformer
        if self.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.transformer, x, mask,
                use_reentrant=False
            )
        else:
            x = self.transformer(x, mask)

        return self.output(x)


class CUDATrainer:
    """GPU-optimized trainer for TKS models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: Dict[str, Any]
    ):
        self.config = config
        self.device = self._setup_device()

        # Move model to device
        self.model = model.to(self.device)

        # Multi-GPU setup
        if config.get('multi_gpu') and torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Distributed setup
        if config.get('distributed'):
            self._setup_distributed()

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Scheduler
        total_steps = len(train_loader) * config['epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # Mixed precision
        self.use_amp = config.get('mixed_precision', True) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Determine precision type
        self.amp_dtype = torch.float16
        if config.get('use_bf16') and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            logger.info("Using BF16 mixed precision")
        elif self.use_amp:
            logger.info("Using FP16 mixed precision")

        # TKS Pipeline mode
        self.use_tks_pipeline = config.get('use_tks_pipeline', False)
        if self.use_tks_pipeline:
            logger.info("Using TKSLLMCorePipeline mode (dict output)")

        # World/RPM auxiliary losses
        self.use_auxiliary_losses = config.get('use_auxiliary_losses', False)
        self.world_loss_weight = config.get('world_loss_weight', 0.1)
        self.rpm_loss_weight = config.get('rpm_loss_weight', 0.1)
        self.use_world_loss_v2 = config.get('use_world_loss_v2', False)
        self.use_world_loss_v3 = config.get('use_world_loss_v3', False)

        if self.use_auxiliary_losses and AUXILIARY_LOSSES_AVAILABLE:
            if self.use_world_loss_v3:
                # V3: Full hybrid head + center loss (recommended for 4-way separation)
                self.world_loss_fn = WorldClassificationLossV3(
                    margin=0.5,
                    temperature=1.0,
                    use_focal=True,
                    focal_gamma=config.get('focal_gamma', 2.0),
                    auto_balance=True,
                    use_usage_reg=True,
                    usage_weight=config.get('usage_reg_weight', 0.5),
                    use_hybrid_head=True,
                    hybrid_alpha=config.get('hybrid_alpha', 0.5),
                    consistency_weight=config.get('consistency_weight', 0.1),
                    learned_head_hidden=config.get('learned_head_hidden', 8),
                    use_center_loss=True,
                    center_weight=config.get('center_loss_weight', 0.3),
                    center_mode=config.get('center_mode', 'contrastive'),
                ).to(self.device)  # Move to device (needed for learned head)
                logger.info(f"WorldClassificationLossV3 enabled: hybrid_alpha={config.get('hybrid_alpha', 0.5)}, "
                           f"center_weight={config.get('center_loss_weight', 0.3)}, "
                           f"center_mode={config.get('center_mode', 'contrastive')}")
            elif self.use_world_loss_v2:
                # Enhanced V2 with focal loss, usage regularizer, optional learned head
                self.world_loss_fn = WorldClassificationLossV2(
                    margin=0.5,
                    temperature=1.0,
                    use_focal=True,
                    focal_gamma=config.get('focal_gamma', 2.0),
                    auto_balance=True,
                    use_usage_reg=True,
                    usage_weight=config.get('usage_reg_weight', 0.5),
                    use_learned_head=config.get('use_learned_world_head', False),
                    learned_head_hidden=config.get('learned_head_hidden', 0),
                ).to(self.device)  # Move to device (needed for learned head)
                logger.info(f"WorldClassificationLossV2 enabled: focal_gamma={config.get('focal_gamma', 2.0)}, "
                           f"usage_weight={config.get('usage_reg_weight', 0.5)}, "
                           f"learned_head={config.get('use_learned_world_head', False)}")
            else:
                # Original world loss
                self.world_loss_fn = WorldClassificationLoss(margin=0.5, temperature=1.0)

            # RPM differentiation loss
            self.use_rpm_loss_v2 = config.get('use_rpm_loss_v2', False)
            if self.use_rpm_loss_v2:
                self.rpm_loss_fn = RPMDifferentiationLossV2(
                    contrastive_margin=config.get('rpm_contrastive_margin', 0.5),
                    spread_target_variance=config.get('rpm_spread_variance', 0.15),
                    focal_gamma=config.get('focal_gamma', 2.0),  # Reuse focal_gamma
                    orthogonality_weight=config.get('rpm_orthogonality_weight', 0.3),
                )
                logger.info(f"RPMDifferentiationLossV2 enabled: margin={config.get('rpm_contrastive_margin', 0.5)}, "
                           f"spread_var={config.get('rpm_spread_variance', 0.15)}, "
                           f"orth_weight={config.get('rpm_orthogonality_weight', 0.3)}")
            else:
                self.rpm_loss_fn = RPMDifferentiationLoss(
                    entropy_weight=0.1,
                    contrastive_margin=0.2,
                    calibration_weight=0.5
                )
            logger.info(f"Auxiliary losses enabled: world_weight={self.world_loss_weight}, rpm_weight={self.rpm_loss_weight}")
        else:
            self.world_loss_fn = None
            self.rpm_loss_fn = None
            if not config.get('use_auxiliary_losses', False):
                logger.info("Auxiliary losses disabled by configuration")

        # Contrastive noetic-consistency loss
        self.contrastive_loss_weight = config.get('contrastive_loss_weight', 0.0)
        if self.contrastive_loss_weight > 0 and AUXILIARY_LOSSES_AVAILABLE:
            self.contrastive_loss_fn = NoeticContrastiveLoss(
                margin=config.get('contrastive_margin', 0.5),
                temperature=config.get('contrastive_temperature', 0.1),
                use_infonce=config.get('contrastive_use_infonce', True)
            )
            logger.info(f"Contrastive loss enabled: weight={self.contrastive_loss_weight}")
        else:
            self.contrastive_loss_fn = None

        # World orthogonality loss (fights mode collapse)
        self.orthogonality_loss_weight = config.get('orthogonality_loss_weight', 0.0)
        if self.orthogonality_loss_weight > 0 and AUXILIARY_LOSSES_AVAILABLE:
            self.orthogonality_loss_fn = WorldOrthogonalityLoss(
                orthogonality_weight=1.0,
                use_cosine=True,
                target_off_diagonal=0.0
            )
            logger.info(f"Orthogonality loss enabled: weight={self.orthogonality_loss_weight}")
        else:
            self.orthogonality_loss_fn = None

        # Attractor convergence loss settings
        self.use_stable_attractor = config.get('use_stable_attractor', True)
        self.attractor_loss_weight = config.get('attractor_loss_weight', 0.05)
        if self.attractor_loss_weight > 0 and TKS_PIPELINE_AVAILABLE:
            logger.info(f"Attractor convergence loss enabled: weight={self.attractor_loss_weight}")

        # Metrics
        self.metrics = {
            'train_losses': [],
            'eval_losses': [],
            'learning_rates': [],
            'gpu_memory_used': [],
            # Auxiliary loss metrics
            'world_losses': [],
            'rpm_losses': [],
            'world_accuracy': [],
            # Contrastive loss metrics
            'contrastive_losses': [],
            # Orthogonality loss metrics
            'orthogonality_losses': [],
            'confusion_matrices': [],  # Per-epoch confusion matrices
            'per_world_energy': [],    # Per-world energy stats
            # Usage regularizer metrics (V2)
            'usage_kl_losses': [],
            'predicted_world_usage': [],  # Per-epoch predicted p(world)
            # Center loss metrics (V3)
            'center_losses': [],
            'center_distances': [],  # Per-epoch world center distances
            # Attractor convergence metrics
            'attractor_losses': [],
            'attractor_convergence_rate': [],
            'attractor_avg_iterations': [],
            'attractor_avg_final_delta': [],
            'attractor_lipschitz_estimate': [],
        }

        # NaN/Inf guard for training stability
        nan_dump_path = None
        if config.get('dump_nan_batches', True):
            nan_dump_path = os.path.join(config.get('output_dir', 'output'), 'nan_batches.jsonl')
        self.nan_guard = NaNGuard(
            dump_path=nan_dump_path,
            max_dumps=config.get('max_nan_dumps', 100)
        )
        logger.info(f"NaN guard enabled, dump path: {nan_dump_path}")

    def _get_logits(self, model_output):
        """Extract logits from model output (tensor or dict)."""
        if self.use_tks_pipeline and isinstance(model_output, dict):
            return model_output['logits']
        return model_output

    def _setup_device(self) -> torch.device:
        """Setup compute device with optimizations."""
        if torch.cuda.is_available():
            device = torch.device('cuda')

            # Enable TF32 for Ampere GPUs (faster, minimal precision loss)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 for matmul operations")

            # Optimize cuDNN
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark mode")

            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device('cpu')

    def _setup_distributed(self):
        """Setup distributed training."""
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)

        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        logger.info(f"Initialized DDP on rank {local_rank}")

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch with attractor convergence tracking."""
        self.model.train()
        total_loss = 0.0
        total_world_loss = 0.0
        total_rpm_loss = 0.0
        total_world_accuracy = 0.0
        total_contrastive_loss = 0.0
        total_attractor_loss = 0.0
        total_orthogonality_loss = 0.0
        total_usage_kl_loss = 0.0
        total_center_loss = 0.0
        last_center_distances = {}
        num_batches = 0
        num_aux_batches = 0
        num_contrastive_batches = 0
        num_orthogonality_batches = 0
        epoch_predicted_usage = torch.zeros(4)  # Accumulate predicted p(world)
        # Confusion matrix accumulator (4x4: actual vs predicted)
        epoch_confusion_matrix = torch.zeros(4, 4, dtype=torch.long)
        epoch_world_energy_sums = torch.zeros(4)
        epoch_world_energy_counts = torch.zeros(4)

        # Attractor convergence tracking for the epoch
        epoch_convergence_count = 0
        epoch_total_iterations = 0
        epoch_total_final_delta = 0.0
        num_attractor_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                # Use return_full_trace=True to get attractor metrics and dwp_scores
                # return_tensor_deltas is only available in TKSLLMCorePipeline (v2), not TKSNoeticLM (v4)
                needs_trace = (
                    self.use_tks_pipeline and
                    (self.attractor_loss_weight > 0 or self.use_auxiliary_losses)
                )
                if needs_trace:
                    # Check if model is v4 (TKSNoeticLM) or v2 (TKSLLMCorePipeline)
                    is_v4_model = hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size')
                    if is_v4_model:
                        # TKSNoeticLM v4 - only supports return_full_trace
                        output = self.model(input_ids, return_full_trace=True)
                    else:
                        # TKSLLMCorePipeline v2 - supports both
                        output = self.model(
                            input_ids,
                            return_full_trace=True,
                            return_tensor_deltas=True
                        )
                else:
                    output = self.model(input_ids)
                logits = self._get_logits(output)

                # Track attractor convergence metrics if available
                if self.use_tks_pipeline and isinstance(output, dict):
                    if 'attractor_converged' in output:
                        num_attractor_samples += 1
                        if output['attractor_converged']:
                            epoch_convergence_count += 1
                        trace = output.get('trace', {})
                        if trace and 'attractor_iterations' in trace:
                            epoch_total_iterations += trace['attractor_iterations']

                # Apply loss mask if present (seq2seq: only predict target tokens)
                if 'loss_mask' in batch:
                    loss_mask = batch['loss_mask'].to(self.device, non_blocking=True)
                    # Per-token loss without reduction
                    per_token_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        reduction='none'
                    )
                    # Apply mask and compute mean over masked tokens
                    masked_loss = per_token_loss * loss_mask.view(-1)
                    main_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
                else:
                    main_loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )

                # Compute auxiliary losses if enabled and model outputs dict
                world_loss_val = 0.0
                rpm_loss_val = 0.0
                world_acc_val = 0.0

                if (self.use_auxiliary_losses and
                    self.use_tks_pipeline and
                    isinstance(output, dict) and
                    self.world_loss_fn is not None):

                    # Extract noetic embedding from trace or gated_output
                    trace = output.get('trace', {})
                    noetic_embedding = trace.get('embedding') if trace else None
                    dwp_scores = trace.get('dwp_scores') if trace else None
                    gated_output = output.get('gated_output')

                    # Use gated_output as noetic embedding fallback
                    if noetic_embedding is None and gated_output is not None:
                        noetic_embedding = gated_output

                    if noetic_embedding is not None:
                        # Use ground truth labels if available, otherwise use pseudo-labels
                        if 'world_label' in batch:
                            # Ground truth labels from dataset
                            world_labels = batch['world_label'].to(self.device)
                        else:
                            # Fallback: infer pseudo-labels from model output (self-supervised)
                            with torch.no_grad():
                                world_energies = self.world_loss_fn.compute_world_energies(noetic_embedding)
                                # Handle V3 returning tuple (logits, hybrid_out) when using hybrid head
                                if isinstance(world_energies, tuple):
                                    world_energies = world_energies[0]
                                world_labels = world_energies.argmax(dim=-1)

                        # Compute world classification loss in float32 to prevent NaN
                        # (focal loss + center loss can overflow in FP16/BF16)
                        with torch.cuda.amp.autocast(enabled=False):
                            noetic_embedding_f32 = noetic_embedding.float()
                            world_losses = self.world_loss_fn(
                                noetic_embedding=noetic_embedding_f32,
                                world_labels=world_labels,
                                world_weights=None,
                            )
                        world_loss_val = world_losses['total'].item()
                        world_acc_val = world_losses['accuracy'].item()

                        # Extract usage_kl if using V2/V3
                        if (self.use_world_loss_v2 or self.use_world_loss_v3) and 'usage_kl' in world_losses:
                            usage_kl_val = world_losses['usage_kl'].item()
                            total_usage_kl_loss += usage_kl_val
                            if 'predicted_usage' in world_losses:
                                epoch_predicted_usage += world_losses['predicted_usage'].cpu()

                        # Extract center_loss if using V3
                        if self.use_world_loss_v3 and 'center_loss' in world_losses:
                            center_loss_val = world_losses['center_loss'].item()
                            total_center_loss += center_loss_val
                            if 'center_distances' in world_losses:
                                last_center_distances = world_losses['center_distances']

                        # Apply world loss with weight
                        main_loss = main_loss + self.world_loss_weight * world_losses['total']

                    # Compute RPM differentiation loss if dwp_scores available
                    if dwp_scores is not None and self.rpm_loss_fn is not None:
                        # Use ground truth RPM labels if available
                        rpm_labels = batch.get('rpm_label')
                        if rpm_labels is not None:
                            rpm_labels = rpm_labels.to(self.device)

                        # Compute RPM loss in float32 to prevent NaN
                        with torch.cuda.amp.autocast(enabled=False):
                            dwp_scores_f32 = dwp_scores.float()
                            rpm_losses = self.rpm_loss_fn(
                                dwp_scores=dwp_scores_f32,
                                rpm_labels=rpm_labels,
                                rpm_weights=None,
                            )
                        rpm_loss_val = rpm_losses['total'].item()

                        # Apply RPM loss with weight
                        main_loss = main_loss + self.rpm_loss_weight * rpm_losses['total']

                    num_aux_batches += 1
                    total_world_loss += world_loss_val
                    total_rpm_loss += rpm_loss_val
                    total_world_accuracy += world_acc_val

                    # Update confusion matrix (ground truth vs predicted)
                    with torch.no_grad():
                        world_energies_result = self.world_loss_fn.compute_world_energies(noetic_embedding)
                        # V3 returns (logits, hybrid_out) tuple; V1/V2 return just tensor
                        if isinstance(world_energies_result, tuple):
                            world_energies = world_energies_result[0]
                        else:
                            world_energies = world_energies_result
                        predicted_worlds = world_energies.argmax(dim=-1)  # [batch]
                        for actual, pred in zip(world_labels.cpu(), predicted_worlds.cpu()):
                            epoch_confusion_matrix[actual.item(), pred.item()] += 1
                        # Per-world energy stats
                        for i in range(4):
                            mask = (world_labels == i)
                            if mask.any():
                                epoch_world_energy_sums[i] += world_energies[mask, i].sum().cpu()
                                epoch_world_energy_counts[i] += mask.sum().cpu()

                # Compute orthogonality loss if enabled (fights mode collapse)
                orthogonality_loss_val = 0.0
                if (self.orthogonality_loss_weight > 0 and
                    self.use_tks_pipeline and
                    isinstance(output, dict) and
                    self.orthogonality_loss_fn is not None):

                    trace = output.get('trace')
                    noetic_embedding = trace.get('embedding') if trace else None
                    gated_output = output.get('gated_output')
                    if noetic_embedding is None and gated_output is not None:
                        noetic_embedding = gated_output

                    if noetic_embedding is not None:
                        # Use ground truth labels if available
                        orth_labels = batch.get('world_label')
                        if orth_labels is not None:
                            orth_labels = orth_labels.to(self.device)

                        orth_result = self.orthogonality_loss_fn(
                            noetic_embedding=noetic_embedding,
                            world_labels=orth_labels
                        )
                        orthogonality_loss_val = orth_result['total'].item()

                        # Apply orthogonality loss with weight
                        main_loss = main_loss + self.orthogonality_loss_weight * orth_result['total']

                        num_orthogonality_batches += 1
                        total_orthogonality_loss += orthogonality_loss_val

                # Compute contrastive noetic-consistency loss if enabled
                contrastive_loss_val = 0.0
                if (self.contrastive_loss_weight > 0 and
                    self.use_tks_pipeline and
                    isinstance(output, dict) and
                    self.contrastive_loss_fn is not None):

                    # Use gated_output as noetic states (40D per sample)
                    gated_output = output.get('gated_output')

                    if gated_output is not None:
                        # Generate pseudo-labels from world energies
                        with torch.no_grad():
                            # Use world_loss_fn to compute world energies
                            if self.world_loss_fn is not None:
                                world_energies = self.world_loss_fn.compute_world_energies(gated_output)
                                # Handle V3 returning tuple (logits, hybrid_out) when using hybrid head
                                if isinstance(world_energies, tuple):
                                    world_energies = world_energies[0]
                                pseudo_labels = world_energies.argmax(dim=-1)  # [batch]
                            else:
                                # Fallback: infer labels from which world has highest energy
                                # World slices: A=0:10, B=10:20, C=20:30, D=30:40
                                if gated_output.dim() == 3:
                                    gated_avg = gated_output.mean(dim=1)  # [batch, 40]
                                else:
                                    gated_avg = gated_output
                                world_energies_local = torch.stack([
                                    gated_avg[:, 0:10].norm(dim=-1),
                                    gated_avg[:, 10:20].norm(dim=-1),
                                    gated_avg[:, 20:30].norm(dim=-1),
                                    gated_avg[:, 30:40].norm(dim=-1),
                                ], dim=-1)  # [batch, 4]
                                pseudo_labels = world_energies_local.argmax(dim=-1)  # [batch]

                        # Compute contrastive loss in float32 to prevent NaN
                        # (InfoNCE with temperature scaling can overflow in FP16/BF16)
                        with torch.cuda.amp.autocast(enabled=False):
                            gated_output_f32 = gated_output.float()
                            contrastive_loss = self.contrastive_loss_fn(
                                noetic_states=gated_output_f32,
                                labels=pseudo_labels,
                            )
                        contrastive_loss_val = contrastive_loss.item()

                        # Apply contrastive loss with weight
                        main_loss = main_loss + self.contrastive_loss_weight * contrastive_loss

                        num_contrastive_batches += 1
                        total_contrastive_loss += contrastive_loss_val

                # Compute attractor convergence loss if enabled
                attractor_loss_val = 0.0
                if (self.use_tks_pipeline and self.attractor_loss_weight > 0 and
                    TKS_PIPELINE_AVAILABLE and isinstance(output, dict)):
                    # Use convergence_loss from model output for direct backprop
                    if 'convergence_loss' in output:
                        conv_loss = output['convergence_loss']
                        attractor_loss_val = conv_loss.item()
                        main_loss = main_loss + self.attractor_loss_weight * conv_loss
                        total_attractor_loss += attractor_loss_val
                    else:
                        # Fallback to Lipschitz penalty if convergence_loss not available
                        base_model = self.model.module if hasattr(self.model, 'module') else self.model
                        if hasattr(base_model, 'attractor'):
                            attractor_layer = base_model.attractor
                            if isinstance(attractor_layer, StableAttractorLayer):
                                lip_penalty = compute_lipschitz_penalty(
                                    attractor_layer,
                                    num_samples=20,
                                    target_lipschitz=0.9
                                )
                                attractor_loss_val = lip_penalty.item()
                                main_loss = main_loss + self.attractor_loss_weight * lip_penalty
                                total_attractor_loss += attractor_loss_val

                loss = main_loss

            # NaN/Inf guard - skip batch if loss is problematic
            if self.nan_guard.check_loss(loss, batch_idx, epoch):
                # Build loss components for debugging
                loss_components = {
                    'main_loss': main_loss.item() if not torch.isnan(main_loss) else 'nan',
                    'world_loss': world_loss_val,
                    'rpm_loss': rpm_loss_val,
                    'contrastive_loss': contrastive_loss_val,
                    'orthogonality_loss': orthogonality_loss_val,
                    'attractor_loss': attractor_loss_val,
                }
                # Dump batch for analysis
                self.nan_guard.dump_batch(batch, batch_idx, epoch, loss_components)
                logger.warning(f"NaN/Inf detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                logger.warning(f"  Loss components: {loss_components}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 20 == 0:
                lr = self.scheduler.get_last_lr()[0]
                mem_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                aux_info = ""
                if self.use_auxiliary_losses and num_aux_batches > 0:
                    aux_info = f" | World: {world_loss_val:.4f} | RPM: {rpm_loss_val:.4f}"
                if self.contrastive_loss_weight > 0 and contrastive_loss_val > 0:
                    aux_info += f" | Contr: {contrastive_loss_val:.4f}"
                if self.orthogonality_loss_weight > 0 and orthogonality_loss_val > 0:
                    aux_info += f" | Orth: {orthogonality_loss_val:.4f}"
                if self.attractor_loss_weight > 0 and attractor_loss_val > 0:
                    aux_info += f" | Attr: {attractor_loss_val:.4f}"
                logger.info(
                    f"Epoch {epoch+1} | Step {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f}{aux_info} | LR: {lr:.2e} | GPU Mem: {mem_used:.2f}GB"
                )

        # Store auxiliary loss metrics
        if num_aux_batches > 0:
            self.metrics['world_losses'].append(total_world_loss / num_aux_batches)
            self.metrics['rpm_losses'].append(total_rpm_loss / num_aux_batches)
            self.metrics['world_accuracy'].append(total_world_accuracy / num_aux_batches)

        # Store contrastive loss metrics
        if num_contrastive_batches > 0:
            self.metrics['contrastive_losses'].append(total_contrastive_loss / num_contrastive_batches)

        # Store orthogonality loss metrics and confusion matrix
        if num_orthogonality_batches > 0:
            self.metrics['orthogonality_losses'].append(total_orthogonality_loss / num_orthogonality_batches)

        # Store usage KL metrics (V2/V3)
        if (self.use_world_loss_v2 or self.use_world_loss_v3) and num_aux_batches > 0:
            self.metrics['usage_kl_losses'].append(total_usage_kl_loss / num_aux_batches)
            # Normalize predicted usage
            if epoch_predicted_usage.sum() > 0:
                pred_usage_norm = epoch_predicted_usage / epoch_predicted_usage.sum()
                self.metrics['predicted_world_usage'].append(
                    {w: float(pred_usage_norm[i]) for i, w in enumerate(['A', 'B', 'C', 'D'])}
                )
                logger.info(f"  World Usage: A={pred_usage_norm[0]:.2%} B={pred_usage_norm[1]:.2%} "
                           f"C={pred_usage_norm[2]:.2%} D={pred_usage_norm[3]:.2%} | Usage KL: {total_usage_kl_loss / num_aux_batches:.4f}")

        # Store center loss metrics (V3)
        if self.use_world_loss_v3 and num_aux_batches > 0:
            self.metrics['center_losses'].append(total_center_loss / num_aux_batches)
            if last_center_distances:
                self.metrics['center_distances'].append(last_center_distances)
                logger.info(f"  Center Distances: " + " | ".join([f"{k}={v:.3f}" for k, v in last_center_distances.items()]))

        if epoch_confusion_matrix.sum() > 0:
            # Convert confusion matrix to nested list for JSON serialization
            self.metrics['confusion_matrices'].append(epoch_confusion_matrix.tolist())
            # Per-world energy averages
            per_world_energy = {}
            for i, world_name in enumerate(['A', 'B', 'C', 'D']):
                if epoch_world_energy_counts[i] > 0:
                    per_world_energy[world_name] = float(epoch_world_energy_sums[i] / epoch_world_energy_counts[i])
                else:
                    per_world_energy[world_name] = 0.0
            self.metrics['per_world_energy'].append(per_world_energy)
            # Log confusion matrix summary
            logger.info(f"  Confusion Matrix (A/B/C/D predictions per ground truth):")
            for i, world_name in enumerate(['A', 'B', 'C', 'D']):
                row = epoch_confusion_matrix[i].tolist()
                total = sum(row)
                acc = row[i] / total * 100 if total > 0 else 0
                logger.info(f"    {world_name}: {row} -> {acc:.1f}% correct")

        # Store attractor convergence metrics
        if num_attractor_samples > 0:
            convergence_rate = epoch_convergence_count / num_attractor_samples * 100
            avg_iterations = epoch_total_iterations / num_attractor_samples
            avg_attractor_loss = total_attractor_loss / num_batches

            self.metrics['attractor_losses'].append(avg_attractor_loss)
            self.metrics['attractor_convergence_rate'].append(convergence_rate)
            self.metrics['attractor_avg_iterations'].append(avg_iterations)

            # Log attractor convergence summary for the epoch
            logger.info(f"  Attractor: {convergence_rate:.1f}% converged, avg {avg_iterations:.1f} iters, loss {avg_attractor_loss:.4f}")

        # Log NaN/Inf summary for the epoch
        self.nan_guard.log_summary()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate model."""
        if self.eval_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                output = self.model(input_ids)
                logits = self._get_logits(output)

                # Apply loss mask if present (seq2seq)
                if 'loss_mask' in batch:
                    loss_mask = batch['loss_mask'].to(self.device, non_blocking=True)
                    per_token_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        reduction='none'
                    )
                    masked_loss = per_token_loss * loss_mask.view(-1)
                    loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
                else:
                    loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self, epochs: int, output_dir: str, per_epoch_eval: bool = False, eval_every: int = 1, eval_test_script: str = 'tests/test_noetic_regression.py') -> Dict[str, Any]:
        """Full training loop with optional early stopping and per-epoch evaluation.

        Args:
            epochs: Number of epochs to train
            output_dir: Directory to save outputs
            per_epoch_eval: If True, run equation + NL evaluation after each epoch
            eval_every: Run eval every N epochs (default: 1)
            eval_test_script: Path to evaluation test script
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        best_eval_loss = float('inf')
        start_time = time.time()

        # Early stopping setup
        patience = self.config.get('early_stopping_patience', 0)
        epochs_without_improvement = 0

        # Per-epoch eval setup
        if per_epoch_eval:
            logger.info(f"Per-epoch evaluation enabled (every {eval_every} epochs)")
            # Clear previous eval history
            history_path = output_path / 'epoch_eval_history.jsonl'
            if history_path.exists():
                history_path.unlink()

        logger.info("=" * 70)
        logger.info("CUDA-OPTIMIZED TKS TRAINING")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp} ({self.amp_dtype})")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if patience > 0:
            logger.info(f"Early Stopping: enabled (patience={patience})")

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.metrics['train_losses'].append(train_loss)
            self.metrics['learning_rates'].append(self.scheduler.get_last_lr()[0])

            # Evaluate
            eval_loss = self.evaluate()
            self.metrics['eval_losses'].append(eval_loss)

            # GPU memory
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                self.metrics['gpu_memory_used'].append(mem_gb)
                torch.cuda.reset_peak_memory_stats()

            epoch_time = time.time() - epoch_start

            logger.info("-" * 70)
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Log auxiliary losses if available
            if self.use_auxiliary_losses and len(self.metrics['world_losses']) > epoch:
                world_loss = self.metrics['world_losses'][epoch]
                rpm_loss = self.metrics['rpm_losses'][epoch]
                world_acc = self.metrics['world_accuracy'][epoch]
                logger.info(
                    f"  Auxiliary: World Loss: {world_loss:.4f} | "
                    f"RPM Loss: {rpm_loss:.4f} | World Acc: {world_acc:.2%}"
                )

            # Log contrastive loss if available
            if self.contrastive_loss_weight > 0 and len(self.metrics['contrastive_losses']) > epoch:
                contrastive_loss = self.metrics['contrastive_losses'][epoch]
                logger.info(f"  Contrastive: {contrastive_loss:.4f}")

            # Save best model and check for early stopping
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                epochs_without_improvement = 0
                self._save_checkpoint(output_path / "best_model.pt", epoch)
                logger.info(f"  -> New best model saved!")
            else:
                epochs_without_improvement += 1
                if patience > 0:
                    logger.info(f"  -> No improvement ({epochs_without_improvement}/{patience})")

            # Early stopping check
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(output_path / f"checkpoint_epoch_{epoch+1}.pt", epoch)

            # Per-epoch evaluation (equation + NL metrics)
            if per_epoch_eval and (epoch + 1) % eval_every == 0:
                # Use best model for eval
                model_path = str(output_path / "best_model.pt")
                if os.path.exists(model_path):
                    eval_metrics = run_per_epoch_eval(
                        model_path=model_path,
                        output_dir=str(output_path),
                        epoch=epoch + 1,
                        test_script=eval_test_script
                    )
                    log_epoch_eval_metrics(eval_metrics)

        # Final save
        self._save_checkpoint(output_path / "final_model.pt", epochs - 1)

        # Save metrics
        total_time = time.time() - start_time
        self.metrics['total_time_seconds'] = total_time
        self.metrics['samples_per_second'] = (
            len(self.train_loader.dataset) * epochs / total_time
        )

        with open(output_path / "training_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best eval loss: {best_eval_loss:.4f}")
        logger.info(f"Throughput: {self.metrics['samples_per_second']:.1f} samples/sec")
        logger.info("=" * 70)

        return self.metrics

    def _save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        model_state = self.model.module.state_dict() \
            if hasattr(self.model, 'module') else self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }, path)


def main():
    parser = argparse.ArgumentParser(description="CUDA-Optimized TKS Training")
    parser.add_argument('--data', required=True, help='Path to training data JSONL')
    parser.add_argument('--output-dir', default='output/cuda_models', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (loads model weights only)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--multi-gpu', action='store_true', help='Use DataParallel')
    parser.add_argument('--distributed', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--use-bf16', action='store_true', help='Use BF16 instead of FP16')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--model-type', choices=['simple', 'tks_pipeline', 'noetic_v4'], default='noetic_v4',
                        help='Model type: noetic_v4 (136k params, recommended), tks_pipeline (92k params), simple (deprecated)')
    parser.add_argument('--early-stopping', type=int, default=0, metavar='PATIENCE',
                        help='Stop training if eval loss does not improve for N epochs (0=disabled)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for regularization')

    # World/RPM auxiliary loss arguments
    parser.add_argument('--world-loss-weight', type=float, default=0.1,
                        help='Weight for world classification auxiliary loss (default: 0.1)')
    parser.add_argument('--rpm-loss-weight', type=float, default=0.1,
                        help='Weight for RPM differentiation auxiliary loss (default: 0.1)')
    parser.add_argument('--disable-auxiliary-losses', action='store_true',
                        help='Disable World/RPM auxiliary losses even if available')

    # Stable attractor arguments
    parser.add_argument('--use-stable-attractor', action='store_true', default=True,
                        help='Use StableAttractorLayer with guaranteed convergence (default: True)')
    parser.add_argument('--no-stable-attractor', action='store_false', dest='use_stable_attractor',
                        help='Use legacy AttractorComputationLayer (may have convergence issues)')
    parser.add_argument('--attractor-loss-weight', type=float, default=0.05,
                        help='Weight for attractor convergence auxiliary loss (default: 0.05)')
    parser.add_argument('--max-attractor-iter', type=int, default=None,
                        help='Max attractor iterations (default: num_layers). Higher = more convergence time.')

    # Contrastive noetic-consistency loss arguments
    parser.add_argument('--contrastive-loss-weight', type=float, default=0.0,
                        help='Weight for contrastive noetic-consistency loss (default: 0.0, disabled)')
    parser.add_argument('--contrastive-margin', type=float, default=0.5,
                        help='Margin for contrastive loss (default: 0.5)')
    parser.add_argument('--contrastive-temperature', type=float, default=0.1,
                        help='Temperature for InfoNCE contrastive loss (default: 0.1)')
    parser.add_argument('--contrastive-use-infonce', action='store_true', default=True,
                        help='Use InfoNCE-style contrastive loss (default: True)')
    parser.add_argument('--contrastive-use-margin', action='store_false', dest='contrastive_use_infonce',
                        help='Use simple margin-based contrastive loss instead of InfoNCE')

    # World orthogonality loss arguments (fights mode collapse)
    parser.add_argument('--orthogonality-loss-weight', type=float, default=0.0,
                        help='Weight for world orthogonality loss (default: 0.0, disabled). Fights mode collapse.')

    # Enhanced world loss arguments (V2)
    parser.add_argument('--use-world-loss-v2', action='store_true',
                        help='Use enhanced WorldClassificationLossV2 with focal loss and usage regularizer')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (default: 2.0, higher = more focus on hard examples)')
    parser.add_argument('--usage-reg-weight', type=float, default=0.5,
                        help='Weight for world-usage KL regularizer (default: 0.5)')
    parser.add_argument('--use-learned-world-head', action='store_true',
                        help='Use learned projections instead of L2 norm for world scoring')
    parser.add_argument('--learned-head-hidden', type=int, default=0,
                        help='Hidden dim for learned world head (0 = direct 10->1)')

    # Full world loss V3 (hybrid head + center loss)
    parser.add_argument('--use-world-loss-v3', action='store_true',
                        help='Use WorldClassificationLossV3 with hybrid head + center loss (recommended for 4-way separation)')
    parser.add_argument('--hybrid-alpha', type=float, default=0.5,
                        help='Blend factor for hybrid head (0=energy, 1=learned, default: 0.5)')
    parser.add_argument('--consistency-weight', type=float, default=0.1,
                        help='Weight for hybrid head consistency loss (default: 0.1)')
    parser.add_argument('--center-loss-weight', type=float, default=0.3,
                        help='Weight for world center loss (default: 0.3)')
    parser.add_argument('--center-mode', type=str, default='contrastive', choices=['center', 'contrastive'],
                        help='Center loss mode: center (pull only) or contrastive (pull+push)')

    # Enhanced RPM loss arguments (V2)
    parser.add_argument('--use-rpm-loss-v2', action='store_true',
                        help='Use enhanced RPMDifferentiationLossV2 with focal+contrastive+orthogonality')
    parser.add_argument('--rpm-contrastive-margin', type=float, default=0.5,
                        help='Contrastive margin for RPM V2 (default: 0.5)')
    parser.add_argument('--rpm-spread-variance', type=float, default=0.15,
                        help='Target variance for D/W/P spread loss (default: 0.15)')
    parser.add_argument('--rpm-orthogonality-weight', type=float, default=0.3,
                        help='Weight for D/W/P orthogonality loss (default: 0.3)')

    # Presets for reproducible configurations
    parser.add_argument('--world-loss-preset', type=str, default=None,
                        choices=['v3_strict', 'v3_balanced', 'v2_focal', 'full_separation', 'rpm_finetune', 'rpm_finetune_nl'],
                        help='Apply a named preset configuration for world loss')

    # Per-epoch evaluation hooks for tracking equation vs NL metrics
    parser.add_argument('--per-epoch-eval', action='store_true',
                        help='Run equation + NL evaluation after each epoch')
    parser.add_argument('--eval-every', type=int, default=1,
                        help='Run eval every N epochs (default: 1)')
    parser.add_argument('--eval-test-script', type=str, default='tests/test_noetic_regression.py',
                        help='Path to evaluation test script')

    args = parser.parse_args()

    # Apply world loss presets (override individual args)
    if args.world_loss_preset == 'v3_strict':
        # Winning config for full 4-way separation (93.75% accuracy)
        args.use_world_loss_v3 = True
        args.world_loss_weight = 3.0
        args.focal_gamma = 5.0
        args.usage_reg_weight = 0.5
        args.hybrid_alpha = 0.6
        args.center_loss_weight = 0.7
        args.center_mode = 'center'
        args.learned_head_hidden = 8
        logger.info("Applied preset 'v3_strict': focal_gamma=5.0, hybrid_alpha=0.6, center_weight=0.7")
    elif args.world_loss_preset == 'v3_balanced':
        # More balanced config, good for larger datasets
        args.use_world_loss_v3 = True
        args.world_loss_weight = 2.0
        args.focal_gamma = 3.0
        args.usage_reg_weight = 0.5
        args.hybrid_alpha = 0.5
        args.center_loss_weight = 0.5
        args.center_mode = 'contrastive'
        args.learned_head_hidden = 8
        logger.info("Applied preset 'v3_balanced': focal_gamma=3.0, hybrid_alpha=0.5, center_weight=0.5")
    elif args.world_loss_preset == 'v2_focal':
        # V2 with strong focal loss (simpler, no center loss)
        args.use_world_loss_v2 = True
        args.world_loss_weight = 2.0
        args.focal_gamma = 3.0
        args.usage_reg_weight = 0.5
        args.use_learned_world_head = True
        args.learned_head_hidden = 8
        logger.info("Applied preset 'v2_focal': V2 with focal_gamma=3.0, learned head")
    elif args.world_loss_preset == 'full_separation':
        # Full world + RPM separation (combines v3_strict + rpm_v2)
        # World loss V3 strict
        args.use_world_loss_v3 = True
        args.world_loss_weight = 3.0
        args.focal_gamma = 5.0
        args.usage_reg_weight = 0.5
        args.hybrid_alpha = 0.6
        args.center_loss_weight = 0.7
        args.center_mode = 'center'
        args.learned_head_hidden = 8
        # RPM loss V2
        args.use_rpm_loss_v2 = True
        args.rpm_loss_weight = 1.0  # Higher RPM weight
        args.rpm_contrastive_margin = 0.5
        args.rpm_spread_variance = 0.15
        args.rpm_orthogonality_weight = 0.3
        logger.info("Applied preset 'full_separation': world_v3_strict + rpm_v2 (world=3.0, rpm=1.0)")
    elif args.world_loss_preset == 'rpm_finetune':
        # For fine-tuning from a world-trained checkpoint: maintain world separation while training RPM
        # IMPORTANT: Use equation-only data (--nl-ratio 0.0) for best noetic consistency.
        # NL data causes inverted consistency due to prompt mismatch with test equations.
        # Uses lower world weight to maintain (not reinforce) separation, plus RPM V2
        args.use_world_loss_v3 = True
        args.world_loss_weight = 1.5  # Lower than training but still maintains separation
        args.focal_gamma = 3.0  # Less aggressive focal
        args.usage_reg_weight = 0.3
        args.hybrid_alpha = 0.6
        args.center_loss_weight = 0.5  # Lower center loss for maintenance
        args.center_mode = 'center'
        args.learned_head_hidden = 8
        # RPM loss V2 with moderate weight
        args.use_rpm_loss_v2 = True
        args.rpm_loss_weight = 0.3  # Moderate RPM to avoid overpowering world
        args.rpm_contrastive_margin = 0.5
        args.rpm_spread_variance = 0.15
        args.rpm_orthogonality_weight = 0.3
        # Stable attractor settings for convergence
        args.use_stable_attractor = True
        args.attractor_loss_weight = 0.1  # Higher than default to encourage convergence
        args.max_attractor_iter = 10  # More iterations for stable convergence
        # Contrastive loss for noetic consistency (requires contrastive data pairs)
        args.contrastive_loss_weight = 0.1  # Higher weight for better separation
        args.contrastive_margin = 0.5
        args.contrastive_temperature = 0.1
        args.contrastive_use_infonce = True
        logger.info("Applied preset 'rpm_finetune': world_v3(1.5) + rpm_v2(0.3) + attractor(0.1) + contrastive(0.1)")
        logger.info("  NOTE: Use equation-only data (generate_world_rpm_labels.py --nl-ratio 0.0) for best consistency")
    elif args.world_loss_preset == 'rpm_finetune_nl':
        # For NL generalization after equation-only training
        # Use this preset AFTER rpm_finetune to generalize to natural language
        # NOTE: World/RPM/contrastive losses are computed in float32 to prevent NaN (see train_epoch)
        args.use_world_loss_v3 = True
        args.world_loss_weight = 1.0  # Lower world weight for generalization
        args.focal_gamma = 3.5  # Higher gamma (3-4) reduces NaN risk on NL data
        args.usage_reg_weight = 0.2
        args.hybrid_alpha = 0.6
        args.center_loss_weight = 0.3  # Keep ≤ 0.5 for NL stability
        args.center_mode = 'center'
        args.learned_head_hidden = 8
        # RPM loss V2
        args.use_rpm_loss_v2 = True
        args.rpm_loss_weight = 0.2  # Lower RPM for generalization
        args.rpm_contrastive_margin = 0.3
        args.rpm_spread_variance = 0.1
        args.rpm_orthogonality_weight = 0.2
        # Stable attractor (CRITICAL: keeps attractor convergence > 0%)
        args.use_stable_attractor = True
        args.attractor_loss_weight = 0.1  # Increased to maintain attractor dynamics
        args.max_attractor_iter = 10
        # Lower contrastive for NL flexibility
        args.contrastive_loss_weight = 0.05
        args.contrastive_margin = 0.3
        args.contrastive_temperature = 0.15  # Slightly higher temp for numerical stability
        args.contrastive_use_infonce = True
        logger.info("Applied preset 'rpm_finetune_nl': world_v3(1.0) + rpm_v2(0.2) + attractor(0.1) for stable NL training")
        logger.info("  Float32 losses + NaN guards enabled for numerical stability")

    # Default max_attractor_iter to num_layers if not specified
    if args.max_attractor_iter is None:
        args.max_attractor_iter = args.num_layers

    # Check CUDA setup
    cuda_info = check_cuda_setup()
    logger.info("CUDA Setup Info:")
    logger.info(f"  CUDA Available: {cuda_info['cuda_available']}")
    if cuda_info['cuda_available']:
        logger.info(f"  CUDA Version: {cuda_info['cuda_version']}")
        logger.info(f"  Device Count: {cuda_info['device_count']}")
        for dev in cuda_info['devices']:
            logger.info(f"  Device {dev['index']}: {dev['name']} ({dev['total_memory_gb']:.1f}GB)")
        logger.info(f"  BF16 Supported: {cuda_info['bf16_supported']}")

    # Load data
    tokenizer = SimpleTokenizer(max_length=256)
    dataset = TKSDataset(args.data, tokenizer)

    # Split
    eval_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    logger.info(f"Dataset: {len(dataset)} total, {train_size} train, {eval_size} eval")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Model creation based on type
    use_tks_pipeline = False

    if args.model_type == 'noetic_v4':
        # TKSNoeticLM v4 - Full decoder-only noetic model (136k params)
        if not TKS_NOETIC_V4_AVAILABLE:
            logger.error("TKSNoeticLM v4 not available. Install tks_llm_core_v4.")
            sys.exit(1)

        logger.info("Creating TKSNoeticLM v4 model (Full noetic decoder-only LM)")
        attractor_type = "StableAttractorLayer" if args.use_stable_attractor else "AttractorComputationLayer (legacy)"
        logger.info(f"  Attractor type: {attractor_type}")
        logger.info(f"  Num layers: {args.num_layers}")
        logger.info(f"  Max attractor iterations: {args.max_attractor_iter}")
        logger.info(f"  Attractor loss weight: {args.attractor_loss_weight}")

        config_v4 = TKSNoeticLMConfig(
            vocab_size=tokenizer.actual_vocab_size,
            max_seq_len=tokenizer.max_length,
            num_layers=args.num_layers,
            num_scales=3,
            dropout=args.dropout,
            use_attractor=True,
            use_stable_attractor=args.use_stable_attractor,
            contraction_factor=0.5,
            max_attractor_iter=args.max_attractor_iter,
            use_rpm=True,
            tie_embeddings=False,
        )
        model = TKSNoeticLM(config_v4)
        use_tks_pipeline = True  # Enable trace features

    elif args.model_type == 'tks_pipeline':
        # TKSLLMCorePipeline v2 - Simpler pipeline (92k params)
        if not TKS_PIPELINE_AVAILABLE:
            logger.error("TKSLLMCorePipeline not available. Install tks_llm_core_v2.")
            sys.exit(1)

        logger.info("Creating TKSLLMCorePipeline model (v2 - 92k params)")
        attractor_type = "StableAttractorLayer" if args.use_stable_attractor else "AttractorComputationLayer (legacy)"
        logger.info(f"  Attractor type: {attractor_type}")
        logger.info(f"  Max attractor iterations: {args.max_attractor_iter}")
        logger.info(f"  Attractor loss weight: {args.attractor_loss_weight}")
        model = TKSLLMCorePipeline(
            vocab_size=tokenizer.actual_vocab_size,
            hidden_dim=args.hidden_dim,
            noetic_dim=40,  # 4 worlds × 10 noetics
            num_scales=3,
            max_attractor_iter=args.max_attractor_iter,
            contraction_factor=0.5,
            use_stable_attractor=args.use_stable_attractor,
        )
        use_tks_pipeline = True
    else:
        logger.error(f"Unsupported model type: {args.model_type}. Use noetic_v4 or tks_pipeline.")
        sys.exit(1)

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Checkpoint loaded successfully")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Config
    config = {
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'multi_gpu': args.multi_gpu,
        'distributed': args.distributed,
        'mixed_precision': not args.no_amp,
        'use_bf16': args.use_bf16,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'use_tks_pipeline': use_tks_pipeline,
        'early_stopping_patience': args.early_stopping,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        # World/RPM auxiliary loss config
        'world_loss_weight': args.world_loss_weight,
        'rpm_loss_weight': args.rpm_loss_weight,
        'use_auxiliary_losses': AUXILIARY_LOSSES_AVAILABLE and not args.disable_auxiliary_losses,
        # Stable attractor config
        'use_stable_attractor': args.use_stable_attractor,
        'attractor_loss_weight': args.attractor_loss_weight,
        'max_attractor_iter': args.max_attractor_iter,
        # Contrastive noetic-consistency loss config
        'contrastive_loss_weight': args.contrastive_loss_weight,
        'contrastive_margin': args.contrastive_margin,
        'contrastive_temperature': args.contrastive_temperature,
        'contrastive_use_infonce': args.contrastive_use_infonce,
        # World orthogonality loss config
        'orthogonality_loss_weight': args.orthogonality_loss_weight,
        # Enhanced world loss V2 config
        'use_world_loss_v2': args.use_world_loss_v2,
        'focal_gamma': args.focal_gamma,
        'usage_reg_weight': args.usage_reg_weight,
        'use_learned_world_head': args.use_learned_world_head,
        'learned_head_hidden': args.learned_head_hidden,
        # World loss V3 config (hybrid head + center loss)
        'use_world_loss_v3': args.use_world_loss_v3,
        'hybrid_alpha': args.hybrid_alpha,
        'consistency_weight': args.consistency_weight,
        'center_loss_weight': args.center_loss_weight,
        'center_mode': args.center_mode,
        # RPM loss V2 config
        'use_rpm_loss_v2': args.use_rpm_loss_v2,
        'rpm_contrastive_margin': args.rpm_contrastive_margin,
        'rpm_spread_variance': args.rpm_spread_variance,
        'rpm_orthogonality_weight': args.rpm_orthogonality_weight,
    }

    # Train
    trainer = CUDATrainer(model, train_loader, eval_loader, config)
    metrics = trainer.train(
        args.epochs,
        args.output_dir,
        per_epoch_eval=args.per_epoch_eval,
        eval_every=args.eval_every,
        eval_test_script=args.eval_test_script
    )

    return metrics


if __name__ == '__main__':
    main()

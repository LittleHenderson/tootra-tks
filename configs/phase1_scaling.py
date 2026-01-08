"""
Phase 1 Scaling Configuration: 10-50M Parameter Models
=======================================================

Goal: Scale TKS Noetic LM from 105K to 10-50M parameters while:
1. Preserving TKS as "native language" (special tokens)
2. Adding natural language capability (BPE tokenizer)
3. Maintaining Noetic/Attractor/RPM architecture
4. Enabling mixed TKS + NL training

Architecture Philosophy:
- TKS tokens are ATOMIC (never split by BPE)
- 4-world structure (A/B/C/D) preserved in embedding space
- Attractors provide "deliberate reasoning" capability
- RPM gating controls information flow between cognitive modes
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math


# =============================================================================
# TOKENIZER CONFIGURATION
# =============================================================================

@dataclass
class TKSHybridTokenizerConfig:
    """Hybrid tokenizer: BPE for NL + atomic TKS special tokens."""

    # Base vocabulary (BPE for natural language)
    base_vocab_size: int = 8000  # Small BPE vocab for NL

    # TKS Special Tokens (NEVER split by BPE)
    # These are added on top of base vocab
    tks_special_tokens: Dict[str, List[str]] = field(default_factory=lambda: {
        # Control tokens
        "control": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"],

        # Sephirot tokens (40 total: A0-A9, B0-B9, C0-C9, D0-D9)
        "sephirot": [f"{w}{i}" for w in "ABCD" for i in range(10)],

        # World markers
        "world": ["<WORLD_A>", "<WORLD_B>", "<WORLD_C>", "<WORLD_D>"],

        # TKS Operators (atomic)
        "operators": [
            "<OP_ADD>",      # +
            "<OP_SUB>",      # -
            "<OP_TRANS_F>",  # +T (transform forward)
            "<OP_TRANS_R>",  # -T (transform reverse)
            "<OP_MUL_T>",    # *T (multiply transform)
            "<OP_DIV_T>",    # /T (divide transform)
            "<OP_FLOW_F>",   # -> (flow forward)
            "<OP_FLOW_R>",   # <- (flow reverse)
        ],

        # TKS Structural tokens
        "structural": [
            "<PAREN_L>",     # (
            "<PAREN_R>",     # )
            "<NOETIC>",      # ^ (noetic level marker)
            "<PATH>",        # : (path/connection marker)
        ],

        # Noetic levels (0-9)
        "noetic_levels": [f"<N{i}>" for i in range(10)],

        # RPM markers
        "rpm": ["<RPM_DESIRE>", "<RPM_WISDOM>", "<RPM_POWER>"],
    })

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary = BPE + all TKS special tokens."""
        tks_count = sum(len(v) for v in self.tks_special_tokens.values())
        return self.base_vocab_size + tks_count

    @property
    def tks_token_count(self) -> int:
        return sum(len(v) for v in self.tks_special_tokens.values())


# =============================================================================
# MODEL CONFIGURATIONS (10M, 25M, 50M targets)
# =============================================================================

@dataclass
class Phase1ModelConfig:
    """Base config for Phase 1 scaled models."""

    # Model identification
    name: str = "tks-noetic-phase1"
    version: str = "1.0"

    # Tokenizer
    vocab_size: int = 8096  # BPE base + TKS special tokens

    # Core transformer dimensions
    d_model: int = 256      # Hidden dimension
    n_heads: int = 8        # Attention heads
    n_layers: int = 8       # Transformer layers
    d_ff: int = 1024        # FFN intermediate dimension (4x d_model)

    # Sequence
    max_seq_len: int = 512  # Context window

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # TKS-specific (preserved from original)
    num_scales: int = 3              # Multi-scale attention
    use_attractor: bool = True       # Attractor dynamics
    use_stable_attractor: bool = True
    contraction_factor: float = 0.5
    max_attractor_iter: int = 15
    use_rpm: bool = True             # RPM gating

    # Noetic basis (scaled)
    use_noetic_basis: bool = True
    noetic_basis_residual: bool = True
    noetic_basis_residual_budget: float = 0.1

    # New for Phase 1: World-aware embeddings
    use_world_embeddings: bool = True  # Separate embeddings per world
    world_embedding_dim: int = 64      # Additional world-specific dims

    # Training
    tie_embeddings: bool = False  # Don't tie for now (different input/output needs)

    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        # Embeddings
        embed_params = self.vocab_size * self.d_model
        pos_params = self.max_seq_len * self.d_model

        # Per layer (attention + FFN + norms)
        # Attention: 4 * d_model^2 (Q, K, V, O projections)
        # FFN: 2 * d_model * d_ff
        # Norms: 2 * d_model (2 layer norms)
        attn_params = 4 * self.d_model * self.d_model
        ffn_params = 2 * self.d_model * self.d_ff
        norm_params = 4 * self.d_model  # 2 norms with weight + bias
        layer_params = attn_params + ffn_params + norm_params

        # Output projection
        output_params = self.d_model * self.vocab_size

        # TKS-specific modules (rough estimate)
        attractor_params = self.d_model * self.d_model // 2 if self.use_attractor else 0
        rpm_params = self.d_model * 64 if self.use_rpm else 0
        world_params = 4 * self.world_embedding_dim * self.d_model if self.use_world_embeddings else 0

        total = (
            embed_params + pos_params +
            self.n_layers * layer_params +
            output_params +
            attractor_params + rpm_params + world_params
        )

        return total


# Pre-defined configurations for different scales
CONFIG_10M = Phase1ModelConfig(
    name="tks-noetic-10m",
    vocab_size=8096,
    d_model=256,
    n_heads=8,
    n_layers=8,
    d_ff=1024,
    max_seq_len=512,
)

CONFIG_25M = Phase1ModelConfig(
    name="tks-noetic-25m",
    vocab_size=8096,
    d_model=384,
    n_heads=8,
    n_layers=10,
    d_ff=1536,
    max_seq_len=1024,
)

CONFIG_50M = Phase1ModelConfig(
    name="tks-noetic-50m",
    vocab_size=8096,
    d_model=512,
    n_heads=8,
    n_layers=12,
    d_ff=2048,
    max_seq_len=1024,
)

CONFIG_100M = Phase1ModelConfig(
    name="tks-noetic-100m",
    vocab_size=16000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=2048,
)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class Phase1TrainingConfig:
    """Training configuration for Phase 1 models."""

    # Basic training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 128  # batch_size * grad_accum

    # Learning rate
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # Training duration
    max_steps: int = 100000
    eval_every: int = 1000
    save_every: int = 5000

    # Optimization
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 or float16

    # Data
    train_data_mix: Dict[str, float] = field(default_factory=lambda: {
        "tks_notation": 0.4,      # Pure TKS (current data)
        "tks_explanations": 0.3,  # TKS with NL explanations
        "nl_with_tks": 0.2,       # NL text mentioning TKS
        "general_nl": 0.1,        # General English (bootstrap)
    })

    # TKS-specific training
    tks_loss_weight: float = 1.5  # Weight TKS tokens higher
    world_consistency_loss: bool = True  # Encourage consistent world usage
    attractor_convergence_loss: bool = True


# =============================================================================
# DATA FORMAT SPECIFICATION
# =============================================================================

"""
Phase 1 Training Data Formats:

1. TKS Notation (current format):
   {"input": "Goal: A0.1 Divine Blueprint", "output": "(A0:B2)", "type": "tks_notation"}

2. TKS with Explanation:
   {
     "input": "Explain the TKS expression (A0:B2)",
     "output": "This expression represents the Divine Blueprint (A0) connected to Positive Belief (B2). It shows the pathway from spiritual source to mental reception.",
     "type": "tks_explanation"
   }

3. NL with embedded TKS:
   {
     "text": "The concept of divine will can be expressed in TKS as (A6^0). When we want to show transformation, we use operators like +T for forward transformation.",
     "type": "nl_with_tks"
   }

4. Dialogue format:
   {
     "messages": [
       {"role": "user", "content": "How do I express combining joy with creativity in TKS?"},
       {"role": "assistant", "content": "In TKS, joy is C2 (Emotional) and creativity might be B3 (Cognitive). To combine them: (C2^2) + (B3^2). The ^2 indicates interaction level."}
     ],
     "type": "dialogue"
   }

5. Reasoning chain:
   {
     "input": "Break down the goal of achieving physical health through positive thinking",
     "output": "Let me analyze this step by step:\n1. Physical health = D2 (Physical world)\n2. Positive thinking = B2 (Cognitive world)\n3. The pathway: (B2) -> (D2)\n4. Full expression: (B2:D2) or with transformation: (B2^3) +T (D2^3)",
     "type": "reasoning"
   }
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_config_comparison():
    """Print comparison of all Phase 1 configurations."""
    configs = [CONFIG_10M, CONFIG_25M, CONFIG_50M, CONFIG_100M]

    print("=" * 80)
    print("PHASE 1 MODEL CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Config':<20} {'Params':>12} {'d_model':>10} {'Layers':>8} {'Heads':>8} {'d_ff':>8} {'SeqLen':>8}")
    print("-" * 80)

    for cfg in configs:
        params = cfg.estimate_params()
        params_str = f"{params/1e6:.1f}M"
        print(f"{cfg.name:<20} {params_str:>12} {cfg.d_model:>10} {cfg.n_layers:>8} {cfg.n_heads:>8} {cfg.d_ff:>8} {cfg.max_seq_len:>8}")

    print("=" * 80)


def get_config_by_size(target_params_millions: float) -> Phase1ModelConfig:
    """Get the closest config to target parameter count."""
    configs = [CONFIG_10M, CONFIG_25M, CONFIG_50M, CONFIG_100M]
    target = target_params_millions * 1e6

    closest = min(configs, key=lambda c: abs(c.estimate_params() - target))
    return closest


if __name__ == "__main__":
    print_config_comparison()

    # Show tokenizer config
    tok_cfg = TKSHybridTokenizerConfig()
    print(f"\nTokenizer: {tok_cfg.base_vocab_size} BPE + {tok_cfg.tks_token_count} TKS = {tok_cfg.total_vocab_size} total")

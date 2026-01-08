"""
TKS LLM v5 Recommended Configuration

Provides factory functions for creating properly configured v5 models.

Usage:
    from configs.v5_recommended import get_v5_config, create_v5_model

    # Get config
    config = get_v5_config()

    # Or create model directly
    model = create_v5_model()
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tks_llm_core_v5 import TKSGeneralConfig, TKSGeneralLM


def get_v5_config(
    # Size variants
    size: str = "base",  # "small", "base", "large"
    # Optional overrides
    use_stable_routing: bool = True,
    use_attractor: bool = True,
    use_rpm: bool = True,
    use_dps: bool = False,
    use_governance: bool = False,
    # NJT (Noetic Judgment Transistor) options
    use_njt: bool = False,
    njt_num_transistors: int = 10,
    njt_use_hysteresis: bool = True,
    njt_use_rhythm: bool = False,
    **kwargs,
) -> TKSGeneralConfig:
    """
    Get recommended v5 configuration.

    Args:
        size: Model size variant ("small", "base", "large")
        use_stable_routing: Enable stable routing (recommended)
        use_attractor: Enable attractor computation
        use_rpm: Enable RPM gating
        use_dps: Enable DPS layer (requires trained DPS)
        use_governance: Enable governance rails
        use_njt: Enable NJT (Noetic Judgment Transistor) circuits
        njt_num_transistors: Number of transistors per NJT bank
        njt_use_hysteresis: Enable N5 hysteresis memory
        njt_use_rhythm: Enable N7 rhythm oscillator
        **kwargs: Additional config overrides

    Returns:
        TKSGeneralConfig instance
    """
    # Size presets
    size_configs = {
        "small": {
            "hidden_dim": 256,
            "num_layers": 6,
            "vocab_size": 8192,
        },
        "base": {
            "hidden_dim": 384,
            "num_layers": 12,
            "vocab_size": 16384,
        },
        "large": {
            "hidden_dim": 512,
            "num_layers": 16,
            "vocab_size": 32768,
        },
    }

    if size not in size_configs:
        raise ValueError(f"Unknown size: {size}. Choose from: {list(size_configs.keys())}")

    base_config = size_configs[size]

    # Build config
    config = TKSGeneralConfig(
        # Size-dependent
        hidden_dim=base_config["hidden_dim"],
        num_layers=base_config["num_layers"],
        vocab_size=base_config["vocab_size"],

        # Fixed
        num_scales=4,
        dropout=0.1,

        # Canonical components
        use_attractor=use_attractor,
        contraction_factor=0.5,
        max_attractor_iter=15,
        use_rpm=use_rpm,

        # Operator core (off by default)
        use_operator_core=False,
        operator_equation_dim=384,

        # Stable routing (RECOMMENDED)
        use_stable_routing=use_stable_routing,
        routing_initial_temp=2.0,
        routing_final_temp=0.1,
        routing_warmup_steps=1000,
        routing_anneal_steps=10000,
        routing_noise_type="gumbel",
        routing_initial_noise=0.5,
        routing_final_noise=0.0,
        routing_balance_weight=0.01,

        # NJT (Noetic Judgment Transistor) circuits
        use_njt=use_njt,
        njt_num_transistors=njt_num_transistors,
        njt_use_hysteresis=njt_use_hysteresis,
        njt_use_rhythm=njt_use_rhythm,
        njt_initial_gain=2.0,
        njt_initial_threshold=0.5,
        njt_gate_sharpness=10.0,
        njt_hysteresis_gap=0.3,
    )

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def create_v5_model(
    size: str = "base",
    **kwargs,
) -> TKSGeneralLM:
    """
    Create a v5 model with recommended configuration.

    Args:
        size: Model size variant
        **kwargs: Config overrides

    Returns:
        TKSGeneralLM instance
    """
    config = get_v5_config(size=size, **kwargs)
    return TKSGeneralLM(config)


def get_conservative_config() -> TKSGeneralConfig:
    """
    Get a conservative config for initial training.

    Disables experimental features, focuses on core components.
    """
    return get_v5_config(
        size="base",
        use_stable_routing=True,
        use_attractor=True,
        use_rpm=True,
        use_dps=False,
        use_governance=False,
    )


def get_full_config() -> TKSGeneralConfig:
    """
    Get full config with all features enabled.

    Only use after DPS layer is trained and tested.
    """
    return get_v5_config(
        size="base",
        use_stable_routing=True,
        use_attractor=True,
        use_rpm=True,
        use_dps=True,
        use_governance=True,
    )


# Quick reference
CONFIG_PRESETS = {
    "conservative": get_conservative_config,
    "full": get_full_config,
}


if __name__ == "__main__":
    # Print config info
    print("TKS LLM v5 Recommended Configurations")
    print("=" * 60)

    for size in ["small", "base", "large"]:
        config = get_v5_config(size=size)
        model = TKSGeneralLM(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n{size.upper()} ({num_params/1e6:.1f}M params):")
        print(f"  hidden_dim: {config.hidden_dim}")
        print(f"  num_layers: {config.num_layers}")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  use_stable_routing: {config.use_stable_routing}")
        print(f"  use_attractor: {config.use_attractor}")
        print(f"  use_rpm: {config.use_rpm}")

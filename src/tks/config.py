"""
TKS Runtime Configuration

Central configuration for all TKS runtime modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path


class OperationalMode(Enum):
    """System operational mode."""
    NORMAL = "Normal"
    HIGH_STAKES = "High-Stakes"
    CRITICAL = "Critical"


@dataclass
class VerifierConfig:
    """Configuration for the Verifier module."""
    canon_path: Path = field(default_factory=lambda: Path("data/canon"))
    strict_mode: bool = True
    allow_unknown_operators: bool = False
    evidence_required_in_hs: bool = True
    evidence_required_in_critical: bool = True


@dataclass
class RouterConfig:
    """Configuration for the Router module."""
    max_tool_calls_normal: int = 12
    max_tool_calls_hs: int = 8
    max_tool_calls_critical: int = 0  # Must be verified first
    max_rpm_depth_normal: int = 3
    max_rpm_depth_hs: int = 2
    max_rpm_breadth: int = 5


@dataclass
class GovernanceConfig:
    """Configuration for Governance Rails."""
    hs_threshold: float = 0.45
    critical_threshold: float = 0.70
    alignment_min: float = 0.80
    enable_all_gates: bool = True


@dataclass
class DPSConfig:
    """Configuration for Depth Permission System."""
    initial_p_max: int = 2
    max_depth: int = 5
    count_threshold: float = 0.45
    heavy_threshold: float = 0.70
    tokens_for_unlock: int = 5
    cooldown_k: int = 10
    unlock_in_hs: bool = False
    unlock_in_critical: bool = False


@dataclass
class ScoringConfig:
    """Configuration for Outcome Scoring."""
    # Foundation weights (F1 dominant)
    foundation_weights: Dict[int, float] = field(default_factory=lambda: {
        1: 0.30,  # Highest Desire (alignment) - DOMINANT
        2: 0.15,  # Wisdom
        3: 0.10,  # Patience
        4: 0.10,  # Strength
        5: 0.15,  # Efficiency (but constrained by F1)
        6: 0.10,  # Beauty
        7: 0.10,  # Stability
    })
    cost_weight: float = 0.1
    uncertainty_bonus_weight: float = 0.2
    violation_penalty: float = 10.0


@dataclass
class ReplayConfig:
    """Configuration for Replay Harness."""
    replay_sets_path: Path = field(default_factory=lambda: Path("replay_sets"))
    tolerance: float = 0.01  # Output match tolerance
    reject_on_violation_increase: bool = True


@dataclass
class MemoryConfig:
    """Configuration for Memory Onion."""
    max_compression_depth: int = 5
    default_reconstruction_precision: int = 2
    lossiness_threshold: float = 0.3


@dataclass
class RuminationConfig:
    """Configuration for Rumination Loop."""
    episodes_per_cycle: int = 10
    max_alternatives: int = 3
    novelty_heavy_threshold: float = 0.70
    novelty_count_threshold: float = 0.45
    ewma_alpha: float = 0.1


@dataclass
class LoggingConfig:
    """Configuration for Event Store."""
    logs_path: Path = field(default_factory=lambda: Path("logs"))
    enable_audit: bool = True
    retention_days: int = 90


@dataclass
class TKSConfig:
    """
    Master configuration for the TKS runtime system.

    All module configs are nested here for single-point configuration.
    """
    # Module configs
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    dps: DPSConfig = field(default_factory=DPSConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rumination: RuminationConfig = field(default_factory=RuminationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global settings
    mode: OperationalMode = OperationalMode.NORMAL
    debug: bool = False
    dry_run: bool = False  # If True, don't execute tools

    @classmethod
    def from_yaml(cls, path: Path) -> "TKSConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        import yaml
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

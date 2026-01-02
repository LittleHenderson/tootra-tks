"""
TKS Features Module - Extended capabilities for the TKS wrapper.

Features:
    - inversion_integration: Inversion engine for flipping scenarios
    - dual_compare: Compare results from multiple providers
    - conversation_memory: Multi-turn dialogue support
    - session_logger: Track and archive analyses
    - scenario_generator: Generate ideal equations for goals
    - provider_fallback: Automatic failover between providers
    - routing_stability: Mitigations for routing collapse in TKS-Transformer
    - world_bridge: Audited cross-world mixing with limited capacity
    - residual_stream: Two-stream noetic architecture for over-constraint mitigation

Usage:
    from tks_features import (
        TKSInversionIntegration,
        DualProviderComparison,
        TKSConversationMemory,
        TKSSessionLogger,
        TKSScenarioGenerator,
        TKSProviderFallback,
        TKSFallbackWrapper,
        # Routing stability components
        RoutingTemperatureScheduler,
        LoadBalancingLoss,
        NoisyRouting,
        RoutingEntropyMonitor,
        StableRouting,
        StableRoutingConfig,
        create_stable_routing,
        # Cross-world bridge components
        WorldBridge,
        FoundationBridge,
        BridgeTraceRecord,
        BridgeConstraintLoss,
        TKSWorldBridgeLayer,
        # Residual stream components (over-constraint mitigation)
        TwoStreamNoetic,
        ConstraintScheduler,
        ResidualDistillationLoss,
        TraceableTwoStreamNoetic,
    )
"""

from .inversion_integration import TKSInversionIntegration
from .dual_compare import DualProviderComparison
from .conversation_memory import TKSConversationMemory
from .session_logger import TKSSessionLogger
from .scenario_generator import TKSScenarioGenerator
from .provider_fallback import TKSProviderFallback, TKSFallbackWrapper
from .routing_stability import (
    RoutingTemperatureScheduler,
    LoadBalancingLoss,
    NoisyRouting,
    RoutingEntropyMonitor,
    StableRouting,
    StableRoutingConfig,
    create_stable_routing,
)
from .world_bridge import (
    WorldBridge,
    FoundationBridge,
    LowRankBridge,
    BridgeTraceRecord,
    bridge_to_trace,
    BridgeConstraintLoss,
    TKSWorldBridgeLayer,
    WORLD_PAIRS,
)
from .residual_stream import (
    TwoStreamNoetic,
    ConstraintScheduler,
    ResidualDistillationLoss,
    TraceableTwoStreamNoetic,
)
from .noetic_basis import (
    NoeticBasisSet,
    NoeticBasisLinear,
    NoeticSpectralConstraint,
    NoeticBasisAttention,
    convert_linear_to_noetic_basis,
)

__all__ = [
    "TKSInversionIntegration",
    "DualProviderComparison",
    "TKSConversationMemory",
    "TKSSessionLogger",
    "TKSScenarioGenerator",
    "TKSProviderFallback",
    "TKSFallbackWrapper",
    # Routing stability
    "RoutingTemperatureScheduler",
    "LoadBalancingLoss",
    "NoisyRouting",
    "RoutingEntropyMonitor",
    "StableRouting",
    "StableRoutingConfig",
    "create_stable_routing",
    # Cross-world bridge
    "WorldBridge",
    "FoundationBridge",
    "LowRankBridge",
    "BridgeTraceRecord",
    "bridge_to_trace",
    "BridgeConstraintLoss",
    "TKSWorldBridgeLayer",
    "WORLD_PAIRS",
    # Residual stream (over-constraint mitigation)
    "TwoStreamNoetic",
    "ConstraintScheduler",
    "ResidualDistillationLoss",
    "TraceableTwoStreamNoetic",
    # Noetic basis parameterization (Phase 3)
    "NoeticBasisSet",
    "NoeticBasisLinear",
    "NoeticSpectralConstraint",
    "NoeticBasisAttention",
    "convert_linear_to_noetic_basis",
]

__version__ = "1.4.0"

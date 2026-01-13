"""
TKS Runtime System - The Strong AI Foundation

This package provides the complete runtime system for the TKS/Tootra agent,
connecting training artifacts to a functioning agent loop.

Modules:
    verifier    - Truth plumbing: parse → normalize → canon check → evidence → gate
    router      - Route prerequisites to appropriate tools
    executor    - Execute tool calls with evidence tracking
    logs        - Event-sourced observability and audit trail
    replay      - Deterministic replay and policy comparison
    scoring     - 7-Foundations reward function
    capability  - Self-knowledge and permission checks
    memory      - Nested packet compression/reconstruction
    rumination  - Subconscious self-improvement loop

The main entry point is the EpisodeRunner which orchestrates:
    Input → RPM → Router → Execute → Verify → Memory → Score → Log

Example:
    from tks import EpisodeRunner, TKSConfig

    runner = EpisodeRunner(TKSConfig())
    result = runner.run_episode(goal="...", context={...})
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
def get_episode_runner():
    from .runner import EpisodeRunner
    return EpisodeRunner

def get_config():
    from .config import TKSConfig
    return TKSConfig

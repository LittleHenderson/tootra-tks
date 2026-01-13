"""Legacy shim for TKSStrategicAgent. Prefer src.tks.core.tks_agent."""

from src.tks.core.tks_agent import TKSStrategicAgent

__all__ = ["TKSStrategicAgent"]


if __name__ == "__main__":
    agent = TKSStrategicAgent()
    result = agent.run_episode("How do I build a goal-aware AI?")
    print(result)

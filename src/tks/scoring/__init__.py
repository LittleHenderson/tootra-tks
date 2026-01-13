"""Scoring exports and reward wrapper."""

from typing import Any, Dict

from .foundations_reward import ActionTrace, FoundationsReward, GoalState


class RewardFunction:
    """Wrapper around FoundationsReward used by the EpisodeRunner."""

    def __init__(self, config) -> None:
        self._reward = FoundationsReward(weights=config.foundation_weights)

    def compute_reward(
        self,
        trace: Dict[str, Any],
        goal: Dict[str, Any],
        config,
    ) -> Dict[str, Any]:
        stages = trace.get("stages", [])
        verify_stage = next((s for s in stages if s.get("stage") == "VERIFY"), None)
        gate = None
        if verify_stage:
            gate = verify_stage.get("result", {}).get("gate")

        achieved = gate is not None and gate != "BLOCK"
        goal_state = GoalState(
            goal_description=goal.get("goal", ""),
            achieved=achieved,
            partial_completion=1.0 if achieved else 0.0,
            goal_drift_detected=False,
        )
        action_trace = ActionTrace(actions=stages)
        result = self._reward.compute_full_result(action_trace, goal_state)
        return {
            "total": result.weighted_total,
            "foundations": result.scores,
            "alignment_satisfied": result.alignment_satisfied,
            "details": result.details,
        }


__all__ = [
    "RewardFunction",
    "FoundationsReward",
    "ActionTrace",
    "GoalState",
]

"""
TKS Foundations Reward Signal

Implements the 7 Foundations reward scoring based on TKS/Tootra philosophy.
Each foundation represents a core principle that guides agent behavior.

The 7 Foundations:
    F1: Highest Desire - Alignment to goal (DOMINANT - must constrain all others)
    F2: Wisdom/Understanding - Knowledge and insight quality
    F3: Patience/Endurance - Persistence and thoroughness
    F4: Strength/Courage - Decisiveness and conviction
    F5: Efficiency/Control - Resource optimization (constrained by F1)
    F6: Beauty/Harmony - Elegance and coherence
    F7: Stability/Kingdom - Consistency and reliability
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import IntEnum
import math


class Foundation(IntEnum):
    """The 7 Foundations of TKS philosophy."""
    HIGHEST_DESIRE = 1      # Alignment to goal
    WISDOM = 2              # Understanding
    PATIENCE = 3            # Endurance
    STRENGTH = 4            # Courage
    EFFICIENCY = 5          # Control (constrained by F1)
    BEAUTY = 6              # Harmony
    STABILITY = 7           # Kingdom


@dataclass
class ActionTrace:
    """Represents a trace of actions taken during an episode."""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    evidence_collected: List[Dict[str, Any]] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    resources_used: Dict[str, float] = field(default_factory=dict)
    time_elapsed: float = 0.0
    retries: int = 0
    backtrack_count: int = 0

    # Quality indicators
    coherence_score: float = 0.0
    consistency_checks_passed: int = 0
    consistency_checks_total: int = 0


@dataclass
class GoalState:
    """Represents the goal and current state for alignment scoring."""
    goal_description: str = ""
    goal_constraints: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    priority_level: str = "normal"  # low, normal, high, critical
    success_criteria: List[str] = field(default_factory=list)

    # Outcome
    achieved: bool = False
    partial_completion: float = 0.0
    goal_drift_detected: bool = False


@dataclass
class FoundationsScoreResult:
    """Result containing scores for all 7 foundations."""
    scores: Dict[int, float]  # Foundation enum value -> score (0.0 to 1.0)
    weighted_total: float
    alignment_satisfied: bool  # F1 check
    details: Dict[int, str] = field(default_factory=dict)  # Explanations

    def __post_init__(self):
        if not self.details:
            self.details = {}


class FoundationsReward:
    """
    Computes satisfaction scores for each of the 7 Foundations.

    F1 (Highest Desire) is always dominant - it constrains all other foundations.
    If F1 is not satisfied, the agent has failed regardless of other scores.
    """

    # Default weights for each foundation (sum to 1.0)
    # F1 has highest weight as it's the dominant constraint
    DEFAULT_WEIGHTS: Dict[int, float] = {
        Foundation.HIGHEST_DESIRE: 0.30,  # Dominant
        Foundation.WISDOM: 0.15,
        Foundation.PATIENCE: 0.10,
        Foundation.STRENGTH: 0.10,
        Foundation.EFFICIENCY: 0.15,      # Important but constrained by F1
        Foundation.BEAUTY: 0.10,
        Foundation.STABILITY: 0.10,
    }

    # Minimum F1 threshold - below this, overall reward is severely penalized
    F1_MINIMUM_THRESHOLD: float = 0.5

    def __init__(
        self,
        weights: Optional[Dict[int, float]] = None,
        custom_scorers: Optional[Dict[int, Callable]] = None,
        f1_threshold: float = 0.5
    ):
        """
        Initialize the FoundationsReward calculator.

        Args:
            weights: Custom weights for each foundation (should sum to 1.0)
            custom_scorers: Custom scoring functions for specific foundations
            f1_threshold: Minimum F1 score required for valid outcome
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

        self.custom_scorers = custom_scorers or {}
        self.f1_threshold = f1_threshold

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0 (within tolerance)."""
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-3):
            raise ValueError(f"Foundation weights must sum to 1.0, got {total}")

        # Ensure all foundations have weights
        for f in Foundation:
            if f.value not in self.weights:
                raise ValueError(f"Missing weight for foundation {f.name}")

    def foundations_score(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> Dict[int, float]:
        """
        Compute satisfaction score for each of the 7 Foundations.

        Args:
            action_trace: The trace of actions taken during the episode
            goal_state: The goal and current state information

        Returns:
            Dictionary mapping foundation enum value to score (0.0 to 1.0)
        """
        scores = {}

        # F1: Highest Desire (Alignment to Goal)
        scores[Foundation.HIGHEST_DESIRE] = self._score_f1_alignment(
            action_trace, goal_state
        )

        # F2: Wisdom/Understanding
        scores[Foundation.WISDOM] = self._score_f2_wisdom(
            action_trace, goal_state
        )

        # F3: Patience/Endurance
        scores[Foundation.PATIENCE] = self._score_f3_patience(
            action_trace, goal_state
        )

        # F4: Strength/Courage
        scores[Foundation.STRENGTH] = self._score_f4_strength(
            action_trace, goal_state
        )

        # F5: Efficiency/Control (constrained by F1)
        raw_f5 = self._score_f5_efficiency(action_trace, goal_state)
        # F5 is constrained by F1 - efficiency without alignment is worthless
        f1_score = scores[Foundation.HIGHEST_DESIRE]
        scores[Foundation.EFFICIENCY] = raw_f5 * min(1.0, f1_score / self.f1_threshold)

        # F6: Beauty/Harmony
        scores[Foundation.BEAUTY] = self._score_f6_beauty(
            action_trace, goal_state
        )

        # F7: Stability/Kingdom
        scores[Foundation.STABILITY] = self._score_f7_stability(
            action_trace, goal_state
        )

        # Apply custom scorers if provided
        for foundation_value, scorer in self.custom_scorers.items():
            scores[foundation_value] = scorer(action_trace, goal_state)

        return scores

    def compute_full_result(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> FoundationsScoreResult:
        """
        Compute full foundation scores with weighted total and details.

        Args:
            action_trace: The trace of actions taken
            goal_state: Goal and state information

        Returns:
            Complete FoundationsScoreResult with all scoring information
        """
        scores = self.foundations_score(action_trace, goal_state)

        # Compute weighted total
        weighted_total = sum(
            scores[f] * self.weights[f]
            for f in Foundation
        )

        # Check if F1 (alignment) is satisfied
        f1_satisfied = scores[Foundation.HIGHEST_DESIRE] >= self.f1_threshold

        # If F1 not satisfied, apply severe penalty to total
        if not f1_satisfied:
            weighted_total *= 0.3  # 70% penalty for misalignment

        # Generate details
        details = self._generate_details(scores, action_trace, goal_state)

        return FoundationsScoreResult(
            scores=scores,
            weighted_total=weighted_total,
            alignment_satisfied=f1_satisfied,
            details=details
        )

    def _score_f1_alignment(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> float:
        """
        Score F1: Highest Desire - Alignment to Goal.

        Measures how well the actions align with the stated goal.
        """
        score = 0.0
        factors = 0

        # Factor 1: Goal achievement
        if goal_state.achieved:
            score += 1.0
        else:
            score += goal_state.partial_completion
        factors += 1

        # Factor 2: Goal drift (deviation from original goal)
        if not goal_state.goal_drift_detected:
            score += 1.0
        else:
            score += 0.3  # Significant penalty for drift
        factors += 1

        # Factor 3: Success criteria met
        if goal_state.success_criteria:
            # This would be evaluated by checking each criterion
            # For now, use partial completion as proxy
            score += goal_state.partial_completion
            factors += 1

        # Factor 4: Constraints respected
        if goal_state.goal_constraints:
            # Assume constraints are respected unless explicitly violated
            # In real implementation, this would check constraint satisfaction
            constraint_score = 1.0 - (0.2 * len([
                c for c in goal_state.goal_constraints
                if c.startswith("VIOLATED:")
            ]))
            score += max(0.0, constraint_score)
            factors += 1

        return min(1.0, score / max(1, factors))

    def _score_f2_wisdom(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> float:
        """
        Score F2: Wisdom/Understanding.

        Measures quality of knowledge, insight, and reasoning.
        """
        score = 0.0
        factors = 0

        # Factor 1: Evidence collection quality
        if goal_state.required_evidence:
            collected = len(action_trace.evidence_collected)
            required = len(goal_state.required_evidence)
            if required > 0:
                evidence_ratio = min(1.0, collected / required)
                score += evidence_ratio
                factors += 1

        # Factor 2: Reasoning depth
        reasoning_count = len(action_trace.reasoning_steps)
        if reasoning_count > 0:
            # Reward thoughtful reasoning, but diminishing returns
            reasoning_score = min(1.0, math.log(reasoning_count + 1) / math.log(10))
            score += reasoning_score
            factors += 1

        # Factor 3: Decision quality (based on coherence)
        if action_trace.coherence_score > 0:
            score += action_trace.coherence_score
            factors += 1

        return score / max(1, factors) if factors > 0 else 0.5  # Default to neutral

    def _score_f3_patience(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> float:
        """
        Score F3: Patience/Endurance.

        Measures persistence, thoroughness, and willingness to work through difficulty.
        """
        score = 0.0
        factors = 0

        # Factor 1: Retry behavior (persistence in face of difficulty)
        if action_trace.retries > 0:
            # Retries show persistence, but too many might indicate issues
            retry_score = min(1.0, 0.5 + (action_trace.retries * 0.1))
            if action_trace.retries > 5:
                retry_score = max(0.3, retry_score - 0.1 * (action_trace.retries - 5))
            score += retry_score
            factors += 1
        else:
            # No retries needed - could be good or might indicate giving up
            score += 0.7 if goal_state.achieved else 0.4
            factors += 1

        # Factor 2: Thoroughness (actions taken relative to complexity)
        action_count = len(action_trace.actions)
        if action_count > 0:
            # More actions generally means more thorough (with limits)
            thoroughness = min(1.0, action_count / 10.0)
            score += thoroughness
            factors += 1

        # Factor 3: Didn't give up prematurely
        if goal_state.achieved or goal_state.partial_completion > 0.8:
            score += 1.0
        elif goal_state.partial_completion > 0.5:
            score += 0.7
        else:
            score += goal_state.partial_completion
        factors += 1

        return score / max(1, factors)

    def _score_f4_strength(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> float:
        """
        Score F4: Strength/Courage.

        Measures decisiveness, conviction, and willingness to act.
        """
        score = 0.0
        factors = 0

        # Factor 1: Decisions made (not paralyzed by analysis)
        decision_count = len(action_trace.decisions_made)
        if decision_count > 0:
            decision_score = min(1.0, decision_count / 5.0)
            score += decision_score
            factors += 1

        # Factor 2: Actions taken (willingness to act)
        action_count = len(action_trace.actions)
        if action_count > 0:
            action_score = min(1.0, action_count / 8.0)
            score += action_score
            factors += 1

        # Factor 3: Low backtracking (committed to decisions)
        if action_trace.backtrack_count == 0:
            score += 1.0
        else:
            # Some backtracking is okay, but too much shows indecision
            backtrack_penalty = min(0.8, action_trace.backtrack_count * 0.15)
            score += 1.0 - backtrack_penalty
        factors += 1

        # Factor 4: Handled high priority appropriately
        if goal_state.priority_level in ("high", "critical"):
            # For critical tasks, we expect decisive action
            if goal_state.achieved:
                score += 1.0
            else:
                score += 0.3
            factors += 1

        return score / max(1, factors) if factors > 0 else 0.5

    def _score_f5_efficiency(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> float:
        """
        Score F5: Efficiency/Control.

        Measures resource optimization and effective control.
        Note: This score is later constrained by F1 (alignment).
        """
        score = 0.0
        factors = 0

        # Factor 1: Resource usage efficiency
        resources = action_trace.resources_used
        if resources:
            # Lower resource usage is better (normalized)
            token_usage = resources.get("tokens", 0)
            if token_usage > 0:
                # Assume 10000 tokens is baseline
                token_efficiency = max(0.0, min(1.0, 1.0 - (token_usage / 20000)))
                score += token_efficiency
                factors += 1

            tool_calls = resources.get("tool_calls", 0)
            if tool_calls > 0:
                # Assume 20 tool calls is baseline
                tool_efficiency = max(0.0, min(1.0, 1.0 - (tool_calls / 40)))
                score += tool_efficiency
                factors += 1

        # Factor 2: Time efficiency
        if action_trace.time_elapsed > 0:
            # Assume 60 seconds is baseline for normal tasks
            time_efficiency = max(0.0, min(1.0, 1.0 - (action_trace.time_elapsed / 120)))
            score += time_efficiency
            factors += 1

        # Factor 3: Minimal redundant actions
        action_count = len(action_trace.actions)
        retry_count = action_trace.retries
        if action_count > 0:
            redundancy = retry_count / action_count
            redundancy_score = max(0.0, 1.0 - redundancy)
            score += redundancy_score
            factors += 1

        # Factor 4: Goal achieved efficiently
        if goal_state.achieved:
            score += 1.0
            factors += 1

        return score / max(1, factors) if factors > 0 else 0.5

    def _score_f6_beauty(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> float:
        """
        Score F6: Beauty/Harmony.

        Measures elegance, coherence, and aesthetic quality of solution.
        """
        score = 0.0
        factors = 0

        # Factor 1: Coherence score (directly measures harmony)
        if action_trace.coherence_score > 0:
            score += action_trace.coherence_score
            factors += 1

        # Factor 2: Clean execution (no excessive retries or backtracking)
        noise_ratio = (action_trace.retries + action_trace.backtrack_count)
        action_count = max(1, len(action_trace.actions))
        cleanliness = max(0.0, 1.0 - (noise_ratio / action_count))
        score += cleanliness
        factors += 1

        # Factor 3: Consistency in approach
        if action_trace.consistency_checks_total > 0:
            consistency = (
                action_trace.consistency_checks_passed /
                action_trace.consistency_checks_total
            )
            score += consistency
            factors += 1

        return score / max(1, factors) if factors > 0 else 0.5

    def _score_f7_stability(
        self,
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> float:
        """
        Score F7: Stability/Kingdom.

        Measures consistency, reliability, and sustainable outcomes.
        """
        score = 0.0
        factors = 0

        # Factor 1: Consistency checks passed
        if action_trace.consistency_checks_total > 0:
            stability = (
                action_trace.consistency_checks_passed /
                action_trace.consistency_checks_total
            )
            score += stability
            factors += 1

        # Factor 2: No goal drift (stability of purpose)
        if not goal_state.goal_drift_detected:
            score += 1.0
        else:
            score += 0.4
        factors += 1

        # Factor 3: Low backtracking (stable decisions)
        if action_trace.backtrack_count == 0:
            score += 1.0
        elif action_trace.backtrack_count <= 2:
            score += 0.7
        else:
            score += max(0.2, 1.0 - (action_trace.backtrack_count * 0.15))
        factors += 1

        # Factor 4: Final state achieved and stable
        if goal_state.achieved:
            score += 1.0
        else:
            # Partial completion with stability
            score += goal_state.partial_completion * 0.8
        factors += 1

        return score / max(1, factors)

    def _generate_details(
        self,
        scores: Dict[int, float],
        action_trace: ActionTrace,
        goal_state: GoalState
    ) -> Dict[int, str]:
        """Generate human-readable explanations for each foundation score."""
        details = {}

        foundation_names = {
            Foundation.HIGHEST_DESIRE: "Highest Desire (Alignment)",
            Foundation.WISDOM: "Wisdom/Understanding",
            Foundation.PATIENCE: "Patience/Endurance",
            Foundation.STRENGTH: "Strength/Courage",
            Foundation.EFFICIENCY: "Efficiency/Control",
            Foundation.BEAUTY: "Beauty/Harmony",
            Foundation.STABILITY: "Stability/Kingdom",
        }

        for f in Foundation:
            score = scores[f]
            name = foundation_names[f]

            if score >= 0.9:
                level = "Excellent"
            elif score >= 0.7:
                level = "Good"
            elif score >= 0.5:
                level = "Satisfactory"
            elif score >= 0.3:
                level = "Needs Improvement"
            else:
                level = "Critical"

            details[f] = f"{name}: {level} ({score:.2f})"

        return details

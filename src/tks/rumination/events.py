"""
TKS Rumination Events - Event schemas for the rumination loop.

This module defines the event types emitted by the rumination system,
integrating with the TKS event store (Module 6) for observability.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Import from event store module
try:
    from ..logs.event_schema import TKSEvent, EventType, generate_event_id
except ImportError:
    # Fallback for standalone usage
    def generate_event_id() -> str:
        return f"evt_{uuid.uuid4().hex[:12].upper()}"

    class EventType(Enum):
        NOVELTY_CANDIDATE = "NOVELTY_CANDIDATE"
        DEPTH_UNLOCK = "DEPTH_UNLOCK"
        RUMINATION_CYCLE = "RUMINATION_CYCLE"


class RuminationEventType(Enum):
    """Event types specific to the rumination system."""
    NOVELTY_CANDIDATE = "NOVELTY_CANDIDATE"
    DEPTH_UNLOCK = "DEPTH_UNLOCK"
    RUMINATION_CYCLE = "RUMINATION_CYCLE"
    EPISODE_SELECTED = "EPISODE_SELECTED"
    ALTERNATIVE_GENERATED = "ALTERNATIVE_GENERATED"
    SHADOW_POLICY_TESTED = "SHADOW_POLICY_TESTED"
    PROMOTION_ATTEMPTED = "PROMOTION_ATTEMPTED"


def generate_rumination_event_id() -> str:
    """Generate a unique rumination event ID."""
    unique_part = uuid.uuid4().hex[:12].upper()
    return f"rum_{unique_part}"


@dataclass
class NoveltyScore:
    """
    Novelty scoring result using the DPS formula: NW = V * I * (1 - S)

    Attributes:
        validity: V - Verifier confidence * consistency * evidence_ok
        impact: I - Delta reward + uncertainty + coverage + cost impact
        similarity: S - Similarity to existing knowledge (0-1, higher = more similar)
        novelty_weight: Computed NW score
        breakdown: Detailed component breakdown
    """
    validity: float
    impact: float
    similarity: float
    novelty_weight: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    # Thresholds from DPS spec
    HEAVY_THRESHOLD: float = 0.70
    COUNT_THRESHOLD: float = 0.45

    @property
    def qualifies_for_heavy(self) -> bool:
        """Check if NW qualifies for heavy/significant consideration."""
        return self.novelty_weight >= self.HEAVY_THRESHOLD

    @property
    def qualifies_for_count(self) -> bool:
        """Check if NW qualifies for counting toward depth unlock."""
        return self.novelty_weight >= self.COUNT_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validity": self.validity,
            "impact": self.impact,
            "similarity": self.similarity,
            "novelty_weight": self.novelty_weight,
            "breakdown": self.breakdown,
            "qualifies_for_heavy": self.qualifies_for_heavy,
            "qualifies_for_count": self.qualifies_for_count,
        }


@dataclass
class NoveltyCandidateEvent:
    """
    Event emitted when a novelty candidate is identified.

    Represents a potential improvement discovered during rumination
    that may lead to policy updates or depth unlocks.
    """
    event_id: str
    timestamp: str
    episode_id: str
    candidate_id: str
    novelty_score: NoveltyScore
    source_type: str  # "alternative", "observation", "synthesis"
    description: str
    related_episode_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        episode_id: str,
        novelty_score: NoveltyScore,
        source_type: str,
        description: str,
        related_episode_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "NoveltyCandidateEvent":
        """Factory method to create a new NoveltyCandidate event."""
        return cls(
            event_id=generate_rumination_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            episode_id=episode_id,
            candidate_id=f"cnd_{uuid.uuid4().hex[:8]}",
            novelty_score=novelty_score,
            source_type=source_type,
            description=description,
            related_episode_ids=related_episode_ids or [],
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": RuminationEventType.NOVELTY_CANDIDATE.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "candidate_id": self.candidate_id,
            "novelty_score": self.novelty_score.to_dict(),
            "source_type": self.source_type,
            "description": self.description,
            "related_episode_ids": self.related_episode_ids,
            "metadata": self.metadata,
        }


@dataclass
class DepthUnlockEvent:
    """
    Event emitted when a depth level is unlocked.

    Depth unlocks are earned through validated novelty contributions
    that expand the agent's capability ceiling (p_max).
    """
    event_id: str
    timestamp: str
    previous_depth: int
    new_depth: int
    p_max_before: float
    p_max_after: float
    trigger_candidates: List[str]  # candidate_ids that triggered unlock
    cumulative_novelty: float
    unlock_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        previous_depth: int,
        new_depth: int,
        p_max_before: float,
        p_max_after: float,
        trigger_candidates: List[str],
        cumulative_novelty: float,
        unlock_reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DepthUnlockEvent":
        """Factory method to create a new DepthUnlock event."""
        return cls(
            event_id=generate_rumination_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            previous_depth=previous_depth,
            new_depth=new_depth,
            p_max_before=p_max_before,
            p_max_after=p_max_after,
            trigger_candidates=trigger_candidates,
            cumulative_novelty=cumulative_novelty,
            unlock_reason=unlock_reason,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": RuminationEventType.DEPTH_UNLOCK.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "previous_depth": self.previous_depth,
            "new_depth": self.new_depth,
            "p_max_before": self.p_max_before,
            "p_max_after": self.p_max_after,
            "trigger_candidates": self.trigger_candidates,
            "cumulative_novelty": self.cumulative_novelty,
            "unlock_reason": self.unlock_reason,
            "metadata": self.metadata,
        }


@dataclass
class RuminationCycleEvent:
    """
    Event emitted at the completion of a rumination cycle.

    Captures the full cycle metrics for observability and analysis.
    """
    event_id: str
    timestamp: str
    cycle_id: str
    duration_ms: int
    episodes_processed: int
    alternatives_generated: int
    alternatives_tested: int
    promotions_attempted: int
    promotions_successful: int
    depth_unlocks: int
    novelty_candidates_found: int
    budget_used: int
    budget_limit: int
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        cycle_id: str,
        duration_ms: int,
        episodes_processed: int,
        alternatives_generated: int,
        alternatives_tested: int,
        promotions_attempted: int,
        promotions_successful: int,
        depth_unlocks: int,
        novelty_candidates_found: int,
        budget_used: int,
        budget_limit: int,
        errors: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RuminationCycleEvent":
        """Factory method to create a new RuminationCycle event."""
        return cls(
            event_id=generate_rumination_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle_id=cycle_id,
            duration_ms=duration_ms,
            episodes_processed=episodes_processed,
            alternatives_generated=alternatives_generated,
            alternatives_tested=alternatives_tested,
            promotions_attempted=promotions_attempted,
            promotions_successful=promotions_successful,
            depth_unlocks=depth_unlocks,
            novelty_candidates_found=novelty_candidates_found,
            budget_used=budget_used,
            budget_limit=budget_limit,
            errors=errors or [],
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": RuminationEventType.RUMINATION_CYCLE.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
            "duration_ms": self.duration_ms,
            "episodes_processed": self.episodes_processed,
            "alternatives_generated": self.alternatives_generated,
            "alternatives_tested": self.alternatives_tested,
            "promotions_attempted": self.promotions_attempted,
            "promotions_successful": self.promotions_successful,
            "depth_unlocks": self.depth_unlocks,
            "novelty_candidates_found": self.novelty_candidates_found,
            "budget_used": self.budget_used,
            "budget_limit": self.budget_limit,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class EpisodeSelectedEvent:
    """Event emitted when an episode is selected for rumination."""
    event_id: str
    timestamp: str
    episode_id: str
    selection_reason: str  # "low_reward", "high_uncertainty", "near_miss", "novel"
    priority_score: float
    metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        episode_id: str,
        selection_reason: str,
        priority_score: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> "EpisodeSelectedEvent":
        """Factory method to create a new EpisodeSelected event."""
        return cls(
            event_id=generate_rumination_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            episode_id=episode_id,
            selection_reason=selection_reason,
            priority_score=priority_score,
            metrics=metrics or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": RuminationEventType.EPISODE_SELECTED.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "selection_reason": self.selection_reason,
            "priority_score": self.priority_score,
            "metrics": self.metrics,
        }


@dataclass
class AlternativeGeneratedEvent:
    """Event emitted when an alternative approach is generated."""
    event_id: str
    timestamp: str
    episode_id: str
    alternative_id: str
    strategy: str
    expected_improvement: float
    is_safe: bool
    safety_reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        episode_id: str,
        alternative_id: str,
        strategy: str,
        expected_improvement: float,
        is_safe: bool,
        safety_reason: Optional[str] = None,
    ) -> "AlternativeGeneratedEvent":
        """Factory method to create a new AlternativeGenerated event."""
        return cls(
            event_id=generate_rumination_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            episode_id=episode_id,
            alternative_id=alternative_id,
            strategy=strategy,
            expected_improvement=expected_improvement,
            is_safe=is_safe,
            safety_reason=safety_reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": RuminationEventType.ALTERNATIVE_GENERATED.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "alternative_id": self.alternative_id,
            "strategy": self.strategy,
            "expected_improvement": self.expected_improvement,
            "is_safe": self.is_safe,
            "safety_reason": self.safety_reason,
        }


@dataclass
class ShadowPolicyTestedEvent:
    """Event emitted when a shadow policy is tested via replay."""
    event_id: str
    timestamp: str
    shadow_policy_id: str
    replay_episode_ids: List[str]
    passed: bool
    regression_detected: bool
    violation_increase: bool
    reward_delta: float
    cost_delta: float
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        shadow_policy_id: str,
        replay_episode_ids: List[str],
        passed: bool,
        regression_detected: bool,
        violation_increase: bool,
        reward_delta: float,
        cost_delta: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> "ShadowPolicyTestedEvent":
        """Factory method to create a new ShadowPolicyTested event."""
        return cls(
            event_id=generate_rumination_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            shadow_policy_id=shadow_policy_id,
            replay_episode_ids=replay_episode_ids,
            passed=passed,
            regression_detected=regression_detected,
            violation_increase=violation_increase,
            reward_delta=reward_delta,
            cost_delta=cost_delta,
            details=details or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": RuminationEventType.SHADOW_POLICY_TESTED.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "shadow_policy_id": self.shadow_policy_id,
            "replay_episode_ids": self.replay_episode_ids,
            "passed": self.passed,
            "regression_detected": self.regression_detected,
            "violation_increase": self.violation_increase,
            "reward_delta": self.reward_delta,
            "cost_delta": self.cost_delta,
            "details": self.details,
        }


@dataclass
class PromotionAttemptedEvent:
    """Event emitted when promotion from shadow to prime is attempted."""
    event_id: str
    timestamp: str
    shadow_policy_id: str
    promoted: bool
    rejection_reason: Optional[str] = None
    improvement_metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        shadow_policy_id: str,
        promoted: bool,
        rejection_reason: Optional[str] = None,
        improvement_metrics: Optional[Dict[str, float]] = None,
    ) -> "PromotionAttemptedEvent":
        """Factory method to create a new PromotionAttempted event."""
        return cls(
            event_id=generate_rumination_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            shadow_policy_id=shadow_policy_id,
            promoted=promoted,
            rejection_reason=rejection_reason,
            improvement_metrics=improvement_metrics or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": RuminationEventType.PROMOTION_ATTEMPTED.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "shadow_policy_id": self.shadow_policy_id,
            "promoted": self.promoted,
            "rejection_reason": self.rejection_reason,
            "improvement_metrics": self.improvement_metrics,
        }


class RuminationEventEmitter:
    """
    Event emitter for rumination events.

    Integrates with the TKS event store for persistence and observability.
    """

    def __init__(self, event_store: Optional[Any] = None):
        """
        Initialize the event emitter.

        Args:
            event_store: Optional event store for persistence.
                        If None, events are collected in memory.
        """
        self._event_store = event_store
        self._pending_events: List[Dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        """
        Emit an event to the store.

        Args:
            event: Event object with to_dict() method
        """
        event_dict = event.to_dict()

        if self._event_store is not None:
            try:
                self._event_store.append(event_dict)
            except Exception:
                # Fallback to pending queue if store fails
                self._pending_events.append(event_dict)
        else:
            self._pending_events.append(event_dict)

    def flush_pending(self) -> List[Dict[str, Any]]:
        """
        Get and clear pending events.

        Returns:
            List of pending event dictionaries
        """
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    @property
    def pending_count(self) -> int:
        """Number of pending events."""
        return len(self._pending_events)

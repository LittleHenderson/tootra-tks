"""
Frozen Episode Serialization for TKS Replay Harness.

This module provides the FrozenEpisode dataclass for capturing and serializing
agent episodes for deterministic replay. Frozen episodes contain all necessary
data to replay an agent's behavior without external dependencies.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ToolResult:
    """Represents a deterministic tool result for replay.

    Tool results are captured during live execution and stored for replay,
    ensuring that the same tool invocation always produces the same result.
    """
    tool_name: str
    tool_input: Dict[str, Any]
    result: Any
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "result": self.result,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolResult:
        """Create ToolResult from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            tool_input=data["tool_input"],
            result=data["result"],
            success=data.get("success", True),
            error_message=data.get("error_message"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )


@dataclass
class EpisodeStep:
    """Represents a single step in an episode.

    Each step captures the agent's action and the resulting tool execution,
    enabling step-by-step replay verification.
    """
    step_index: int
    action: str
    action_input: Dict[str, Any]
    tool_result: Optional[ToolResult] = None
    agent_reasoning: Optional[str] = None
    timestamp_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_index": self.step_index,
            "action": self.action,
            "action_input": self.action_input,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "agent_reasoning": self.agent_reasoning,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodeStep:
        """Create EpisodeStep from dictionary."""
        tool_result = None
        if data.get("tool_result"):
            tool_result = ToolResult.from_dict(data["tool_result"])

        return cls(
            step_index=data["step_index"],
            action=data["action"],
            action_input=data["action_input"],
            tool_result=tool_result,
            agent_reasoning=data.get("agent_reasoning"),
            timestamp_ms=data.get("timestamp_ms", 0.0),
        )


@dataclass
class EpisodeMetadata:
    """Metadata about a frozen episode.

    Captures context about when and how the episode was recorded,
    useful for debugging and audit trails.
    """
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "live"  # "live", "synthetic", "replay"
    policy_version: str = "unknown"
    agent_version: str = "unknown"
    environment: str = "production"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    duration_ms: float = 0.0
    total_cost: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "created_at": self.created_at,
            "source": self.source,
            "policy_version": self.policy_version,
            "agent_version": self.agent_version,
            "environment": self.environment,
            "tags": self.tags,
            "notes": self.notes,
            "duration_ms": self.duration_ms,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodeMetadata:
        """Create EpisodeMetadata from dictionary."""
        return cls(
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            source=data.get("source", "live"),
            policy_version=data.get("policy_version", "unknown"),
            agent_version=data.get("agent_version", "unknown"),
            environment=data.get("environment", "production"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            duration_ms=data.get("duration_ms", 0.0),
            total_cost=data.get("total_cost", 0.0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class FrozenEpisode:
    """A frozen episode for deterministic replay.

    FrozenEpisode captures all necessary data to replay an agent's behavior:
    - The original input that triggered the episode
    - The expected output from the original execution
    - All tool results (stubbed for deterministic replay)
    - Metadata for tracking and debugging

    Frozen episodes enable:
    - Regression testing: Verify new policies don't break existing behavior
    - Policy comparison: Compare baseline vs candidate on identical scenarios
    - Debugging: Reproduce exact conditions that led to issues

    Example:
        >>> episode = FrozenEpisode(
        ...     episode_id="ep_001",
        ...     input={"query": "What's the weather?"},
        ...     expected_output={"response": "It's sunny."},
        ...     tool_results=[ToolResult("weather_api", {"city": "NYC"}, {"temp": 72})]
        ... )
        >>> episode.save("replay_sets/weather_test.json")
    """
    episode_id: str
    input: Dict[str, Any]
    expected_output: Dict[str, Any]
    tool_results: List[ToolResult] = field(default_factory=list)
    steps: List[EpisodeStep] = field(default_factory=list)
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)

    # Scoring expectations for validation
    expected_reward: float = 0.0
    expected_success: bool = True
    violation_count: int = 0
    uncertainty_score: float = 0.0

    def __post_init__(self):
        """Validate episode data after initialization."""
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        if self.input is None:
            raise ValueError("input cannot be None")

    @property
    def content_hash(self) -> str:
        """Compute a hash of the episode content for integrity verification."""
        content = json.dumps({
            "episode_id": self.episode_id,
            "input": self.input,
            "expected_output": self.expected_output,
            "tool_results": [tr.to_dict() for tr in self.tool_results],
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def step_count(self) -> int:
        """Return the number of steps in the episode."""
        return len(self.steps)

    @property
    def tool_count(self) -> int:
        """Return the number of tool invocations."""
        return len(self.tool_results)

    def get_tool_result(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[ToolResult]:
        """Retrieve a specific tool result by name and input.

        This is used during replay to stub tool calls with deterministic results.

        Args:
            tool_name: The name of the tool
            tool_input: The input parameters for the tool call

        Returns:
            The matching ToolResult, or None if not found
        """
        input_str = json.dumps(tool_input, sort_keys=True)
        for result in self.tool_results:
            if result.tool_name == tool_name:
                result_input_str = json.dumps(result.tool_input, sort_keys=True)
                if result_input_str == input_str:
                    return result
        return None

    def get_tool_results_by_name(self, tool_name: str) -> List[ToolResult]:
        """Get all tool results for a specific tool."""
        return [tr for tr in self.tool_results if tr.tool_name == tool_name]

    def add_step(self, action: str, action_input: Dict[str, Any],
                 tool_result: Optional[ToolResult] = None,
                 agent_reasoning: Optional[str] = None) -> EpisodeStep:
        """Add a new step to the episode.

        Args:
            action: The action taken by the agent
            action_input: Input parameters for the action
            tool_result: Optional result from tool execution
            agent_reasoning: Optional reasoning from the agent

        Returns:
            The created EpisodeStep
        """
        step = EpisodeStep(
            step_index=len(self.steps),
            action=action,
            action_input=action_input,
            tool_result=tool_result,
            agent_reasoning=agent_reasoning,
        )
        self.steps.append(step)

        # Also add to tool_results if present
        if tool_result:
            self.tool_results.append(tool_result)

        return step

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "input": self.input,
            "expected_output": self.expected_output,
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata.to_dict(),
            "expected_reward": self.expected_reward,
            "expected_success": self.expected_success,
            "violation_count": self.violation_count,
            "uncertainty_score": self.uncertainty_score,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FrozenEpisode:
        """Create FrozenEpisode from dictionary.

        Args:
            data: Dictionary containing episode data

        Returns:
            A FrozenEpisode instance

        Raises:
            ValueError: If required fields are missing
        """
        if "episode_id" not in data:
            raise ValueError("Missing required field: episode_id")
        if "input" not in data:
            raise ValueError("Missing required field: input")

        tool_results = [
            ToolResult.from_dict(tr) for tr in data.get("tool_results", [])
        ]

        steps = [
            EpisodeStep.from_dict(step) for step in data.get("steps", [])
        ]

        metadata = EpisodeMetadata.from_dict(data.get("metadata", {}))

        return cls(
            episode_id=data["episode_id"],
            input=data["input"],
            expected_output=data.get("expected_output", {}),
            tool_results=tool_results,
            steps=steps,
            metadata=metadata,
            expected_reward=data.get("expected_reward", 0.0),
            expected_success=data.get("expected_success", True),
            violation_count=data.get("violation_count", 0),
            uncertainty_score=data.get("uncertainty_score", 0.0),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the frozen episode to a JSON file.

        Args:
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> FrozenEpisode:
        """Load a frozen episode from a JSON file.

        Args:
            path: File path to load from

        Returns:
            A FrozenEpisode instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_live_episode(cls,
                          episode_id: str,
                          input_data: Dict[str, Any],
                          output_data: Dict[str, Any],
                          tool_results: List[ToolResult],
                          steps: Optional[List[EpisodeStep]] = None,
                          reward: float = 0.0,
                          success: bool = True,
                          violations: int = 0,
                          uncertainty: float = 0.0,
                          policy_version: str = "unknown",
                          agent_version: str = "unknown",
                          environment: str = "production",
                          tags: Optional[List[str]] = None,
                          notes: str = "") -> FrozenEpisode:
        """Create a frozen episode from a live episode execution.

        This is the primary factory method for creating frozen episodes
        from production or test runs.

        Args:
            episode_id: Unique identifier for the episode
            input_data: The input that triggered the episode
            output_data: The output produced by the agent
            tool_results: List of tool results from the execution
            steps: Optional list of execution steps
            reward: The reward score from the episode
            success: Whether the episode was successful
            violations: Number of policy violations
            uncertainty: Uncertainty score
            policy_version: Version of the policy used
            agent_version: Version of the agent
            environment: Environment where the episode ran
            tags: Optional tags for categorization
            notes: Optional notes about the episode

        Returns:
            A FrozenEpisode ready for serialization and replay
        """
        metadata = EpisodeMetadata(
            source="live",
            policy_version=policy_version,
            agent_version=agent_version,
            environment=environment,
            tags=tags or [],
            notes=notes,
        )

        return cls(
            episode_id=episode_id,
            input=input_data,
            expected_output=output_data,
            tool_results=tool_results,
            steps=steps or [],
            metadata=metadata,
            expected_reward=reward,
            expected_success=success,
            violation_count=violations,
            uncertainty_score=uncertainty,
        )


@dataclass
class ReplaySet:
    """A collection of frozen episodes for batch replay and comparison.

    ReplaySets group related episodes for policy testing and validation.
    They can be loaded from a single JSON file or assembled programmatically.
    """
    name: str
    description: str = ""
    episodes: List[FrozenEpisode] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    def __getitem__(self, idx: int) -> FrozenEpisode:
        return self.episodes[idx]

    def add_episode(self, episode: FrozenEpisode) -> None:
        """Add an episode to the set."""
        self.episodes.append(episode)

    def get_episode(self, episode_id: str) -> Optional[FrozenEpisode]:
        """Retrieve an episode by ID."""
        for ep in self.episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def filter_by_tag(self, tag: str) -> List[FrozenEpisode]:
        """Get all episodes with a specific tag."""
        return [ep for ep in self.episodes if tag in ep.metadata.tags]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "episodes": [ep.to_dict() for ep in self.episodes],
            "created_at": self.created_at,
            "version": self.version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReplaySet:
        """Create ReplaySet from dictionary."""
        episodes = [
            FrozenEpisode.from_dict(ep) for ep in data.get("episodes", [])
        ]

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            episodes=episodes,
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the replay set to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ReplaySet:
        """Load a replay set from a JSON file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

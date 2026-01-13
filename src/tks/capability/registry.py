"""
TKS Capability Registry - Tool capability definitions and registry.

This module provides the CapabilityRegistry class for managing tool capabilities,
including skill thresholds, mode constraints, reversibility, and risk levels.
The registry enables the TKS agent to maintain self-knowledge about its reliability
and constraints for each tool/action.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from ..config import OperationalMode


class Reversibility(Enum):
    """
    Enumeration of action reversibility levels.

    Determines whether the effects of an action can be undone:
    - REVERSIBLE: Action can be completely undone (e.g., read operations)
    - PARTIAL: Action can be partially undone with some residual effects
    - IRREVERSIBLE: Action cannot be undone (e.g., destructive writes, sends)
    """
    REVERSIBLE = "reversible"
    PARTIAL = "partial"
    IRREVERSIBLE = "irreversible"


class RiskLevel(Enum):
    """
    Enumeration of action risk levels.

    Risk levels affect governance requirements and threshold adjustments:
    - LOW: Minimal risk, standard approval (e.g., read operations)
    - MEDIUM: Moderate risk, enhanced verification (e.g., local writes)
    - HIGH: Significant risk, requires oversight (e.g., external API calls)
    - CRITICAL: Maximum risk, requires explicit approval (e.g., production changes)
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolCapability:
    """
    Defines the capability profile for a single tool/action.

    Attributes:
        tool_name: Unique identifier for the tool
        capability_threshold: Minimum skill level required (0.0-1.0)
        mode_constraints: List of modes where this tool is allowed
        reversibility: Whether the action can be undone
        evidence_required: Whether verification evidence is needed
        risk_level: Risk classification for governance decisions
        description: Human-readable description of the tool
        category: Tool category for grouping (e.g., "file_io", "network")
        cooldown_seconds: Minimum time between successive calls (0 = no limit)
        max_concurrent: Maximum concurrent instances (0 = unlimited)
        requires_confirmation: Whether explicit user confirmation is needed
        metadata: Additional tool-specific metadata
    """
    tool_name: str
    capability_threshold: float = 0.5
    mode_constraints: List[str] = field(default_factory=lambda: [
        OperationalMode.NORMAL.value,
        OperationalMode.HIGH_STAKES.value,
        OperationalMode.CRITICAL.value,
    ])
    reversibility: Reversibility = Reversibility.REVERSIBLE
    evidence_required: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    description: str = ""
    category: str = "general"
    cooldown_seconds: float = 0.0
    max_concurrent: int = 0
    requires_confirmation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate capability parameters after initialization."""
        if not 0.0 <= self.capability_threshold <= 1.0:
            raise ValueError(
                f"capability_threshold must be between 0.0 and 1.0, got {self.capability_threshold}"
            )
        if self.cooldown_seconds < 0:
            raise ValueError(
                f"cooldown_seconds must be non-negative, got {self.cooldown_seconds}"
            )
        if self.max_concurrent < 0:
            raise ValueError(
                f"max_concurrent must be non-negative, got {self.max_concurrent}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "capability_threshold": self.capability_threshold,
            "mode_constraints": self.mode_constraints,
            "reversibility": self.reversibility.value,
            "evidence_required": self.evidence_required,
            "risk_level": self.risk_level.value,
            "description": self.description,
            "category": self.category,
            "cooldown_seconds": self.cooldown_seconds,
            "max_concurrent": self.max_concurrent,
            "requires_confirmation": self.requires_confirmation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCapability:
        """Create a ToolCapability from a dictionary."""
        return cls(
            tool_name=data["tool_name"],
            capability_threshold=data.get("capability_threshold", 0.5),
            mode_constraints=data.get("mode_constraints", ["normal", "high_stakes", "critical", "sandbox"]),
            reversibility=Reversibility(data.get("reversibility", "reversible")),
            evidence_required=data.get("evidence_required", False),
            risk_level=RiskLevel(data.get("risk_level", "low")),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            cooldown_seconds=data.get("cooldown_seconds", 0.0),
            max_concurrent=data.get("max_concurrent", 0),
            requires_confirmation=data.get("requires_confirmation", False),
            metadata=data.get("metadata", {}),
        )


class CapabilityRegistry:
    """
    Singleton registry for managing tool capabilities.

    The CapabilityRegistry maintains the authoritative source of truth for
    tool capabilities across the TKS system. It provides:
    - Tool capability registration and lookup
    - Permission checking based on skill level and mode
    - Thread-safe operations for concurrent access

    Usage:
        registry = CapabilityRegistry.get_instance()
        registry.register(ToolCapability(tool_name="read_file", ...))
        capability = registry.get_capability("read_file")
        allowed = registry.check_allowed("read_file", skill_level=0.7, mode="normal")
    """

    _instance: Optional[CapabilityRegistry] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> CapabilityRegistry:
        """Ensure singleton pattern - only one registry instance exists."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry with empty capability store."""
        if self._initialized:
            return

        self._capabilities: Dict[str, ToolCapability] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._capability_lock = threading.RLock()
        self._initialized = True

        # Register default capabilities
        self._register_defaults()

    @classmethod
    def get_instance(cls) -> CapabilityRegistry:
        """
        Get the singleton instance of CapabilityRegistry.

        Returns:
            CapabilityRegistry: The singleton registry instance
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (primarily for testing).

        Warning: This will clear all registered capabilities.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._capabilities.clear()
                cls._instance._categories.clear()
                cls._instance._initialized = False
            cls._instance = None

    def _register_defaults(self) -> None:
        """Register default tool capabilities for common operations."""
        defaults = [
            # Read operations - low risk, reversible
            ToolCapability(
                tool_name="read_file",
                capability_threshold=0.2,
                reversibility=Reversibility.REVERSIBLE,
                risk_level=RiskLevel.LOW,
                evidence_required=False,
                description="Read file contents from filesystem",
                category="file_io",
            ),
            ToolCapability(
                tool_name="list_directory",
                capability_threshold=0.2,
                reversibility=Reversibility.REVERSIBLE,
                risk_level=RiskLevel.LOW,
                evidence_required=False,
                description="List directory contents",
                category="file_io",
            ),
            # Write operations - medium risk, partially reversible
            ToolCapability(
                tool_name="write_file",
                capability_threshold=0.5,
                reversibility=Reversibility.PARTIAL,
                risk_level=RiskLevel.MEDIUM,
                evidence_required=True,
                description="Write or modify file contents",
                category="file_io",
            ),
            ToolCapability(
                tool_name="create_file",
                capability_threshold=0.4,
                reversibility=Reversibility.REVERSIBLE,
                risk_level=RiskLevel.MEDIUM,
                evidence_required=True,
                description="Create a new file",
                category="file_io",
            ),
            # Delete operations - high risk, irreversible
            ToolCapability(
                tool_name="delete_file",
                capability_threshold=0.7,
                reversibility=Reversibility.IRREVERSIBLE,
                risk_level=RiskLevel.HIGH,
                evidence_required=True,
                requires_confirmation=True,
                description="Delete a file from filesystem",
                category="file_io",
            ),
            # Network operations
            ToolCapability(
                tool_name="http_get",
                capability_threshold=0.3,
                reversibility=Reversibility.REVERSIBLE,
                risk_level=RiskLevel.LOW,
                evidence_required=False,
                description="Make HTTP GET request",
                category="network",
            ),
            ToolCapability(
                tool_name="http_post",
                capability_threshold=0.5,
                reversibility=Reversibility.IRREVERSIBLE,
                risk_level=RiskLevel.MEDIUM,
                evidence_required=True,
                description="Make HTTP POST request",
                category="network",
            ),
            # Shell/execution operations - high risk
            ToolCapability(
                tool_name="execute_command",
                capability_threshold=0.6,
                mode_constraints=["normal", "sandbox"],
                reversibility=Reversibility.IRREVERSIBLE,
                risk_level=RiskLevel.HIGH,
                evidence_required=True,
                requires_confirmation=True,
                description="Execute shell command",
                category="execution",
            ),
            # Critical operations
            ToolCapability(
                tool_name="deploy_production",
                capability_threshold=0.9,
                mode_constraints=["normal"],
                reversibility=Reversibility.PARTIAL,
                risk_level=RiskLevel.CRITICAL,
                evidence_required=True,
                requires_confirmation=True,
                cooldown_seconds=300.0,
                description="Deploy to production environment",
                category="deployment",
            ),
        ]

        for capability in defaults:
            self.register(capability, overwrite=False)

    def register(
        self,
        capability: ToolCapability,
        overwrite: bool = True
    ) -> bool:
        """
        Register a tool capability in the registry.

        Args:
            capability: The ToolCapability to register
            overwrite: Whether to overwrite existing capability (default True)

        Returns:
            bool: True if registration succeeded, False if exists and overwrite=False
        """
        with self._capability_lock:
            if capability.tool_name in self._capabilities and not overwrite:
                return False

            self._capabilities[capability.tool_name] = capability

            # Update category index
            if capability.category not in self._categories:
                self._categories[capability.category] = set()
            self._categories[capability.category].add(capability.tool_name)

            return True

    def unregister(self, tool_name: str) -> bool:
        """
        Remove a tool capability from the registry.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            bool: True if tool was found and removed, False otherwise
        """
        with self._capability_lock:
            if tool_name not in self._capabilities:
                return False

            capability = self._capabilities.pop(tool_name)

            # Update category index
            if capability.category in self._categories:
                self._categories[capability.category].discard(tool_name)
                if not self._categories[capability.category]:
                    del self._categories[capability.category]

            return True

    def get_capability(self, tool_name: str) -> Optional[ToolCapability]:
        """
        Retrieve the capability definition for a tool.

        Args:
            tool_name: Name of the tool to look up

        Returns:
            Optional[ToolCapability]: The capability if found, None otherwise
        """
        with self._capability_lock:
            return self._capabilities.get(tool_name)

    def get_all_capabilities(self) -> Dict[str, ToolCapability]:
        """
        Get all registered capabilities.

        Returns:
            Dict[str, ToolCapability]: Copy of all registered capabilities
        """
        with self._capability_lock:
            return dict(self._capabilities)

    def get_by_category(self, category: str) -> List[ToolCapability]:
        """
        Get all capabilities in a specific category.

        Args:
            category: The category to filter by

        Returns:
            List[ToolCapability]: List of capabilities in the category
        """
        with self._capability_lock:
            tool_names = self._categories.get(category, set())
            return [self._capabilities[name] for name in tool_names]

    def get_categories(self) -> List[str]:
        """
        Get all registered categories.

        Returns:
            List[str]: List of category names
        """
        with self._capability_lock:
            return list(self._categories.keys())

    def check_allowed(
        self,
        tool_name: str,
        current_skill: float,
        mode: OperationalMode | str,
        effective_threshold: Optional[float] = None
    ) -> bool:
        """
        Check if a tool action is allowed given current conditions.

        This is the primary permission check that considers:
        - Current skill level vs capability threshold
        - Mode constraints

        Args:
            tool_name: Name of the tool to check
            current_skill: Current skill level for this tool (0.0-1.0)
            mode: Current operation mode (OperationalMode or string)
            effective_threshold: Optional override threshold (from ThresholdManager)

        Returns:
            bool: True if the action is allowed, False otherwise

        Note:
            This is a basic check. For full permission evaluation including
            governance, use PermissionChecker.check_permission().
        """
        capability = self.get_capability(tool_name)
        if capability is None:
            # Unknown tool - deny by default
            return False

        # Check mode constraints
        mode_value = self._normalize_mode(mode)
        if mode_value.lower() not in [m.lower() for m in capability.mode_constraints]:
            return False

        # Use effective threshold if provided, otherwise use base threshold
        threshold = effective_threshold if effective_threshold is not None else capability.capability_threshold

        # Check skill level against threshold
        if current_skill < threshold:
            return False

        return True

    @staticmethod
    def _normalize_mode(mode: OperationalMode | str) -> str:
        if isinstance(mode, OperationalMode):
            return mode.value
        return str(mode)

    def get_tools_by_risk(self, risk_level: RiskLevel) -> List[ToolCapability]:
        """
        Get all tools at a specific risk level.

        Args:
            risk_level: The risk level to filter by

        Returns:
            List[ToolCapability]: List of capabilities at that risk level
        """
        with self._capability_lock:
            return [
                cap for cap in self._capabilities.values()
                if cap.risk_level == risk_level
            ]

    def get_tools_requiring_confirmation(self) -> List[ToolCapability]:
        """
        Get all tools that require explicit confirmation.

        Returns:
            List[ToolCapability]: List of capabilities requiring confirmation
        """
        with self._capability_lock:
            return [
                cap for cap in self._capabilities.values()
                if cap.requires_confirmation
            ]

    def get_irreversible_tools(self) -> List[ToolCapability]:
        """
        Get all tools with irreversible effects.

        Returns:
            List[ToolCapability]: List of irreversible capabilities
        """
        with self._capability_lock:
            return [
                cap for cap in self._capabilities.values()
                if cap.reversibility == Reversibility.IRREVERSIBLE
            ]

    def __len__(self) -> int:
        """Return the number of registered capabilities."""
        with self._capability_lock:
            return len(self._capabilities)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        with self._capability_lock:
            return tool_name in self._capabilities

    def __repr__(self) -> str:
        return f"CapabilityRegistry(tools={len(self)}, categories={len(self._categories)})"

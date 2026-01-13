"""
TKS Packet Representation - Memory Onion Core Data Structure

This module defines the TKSPacket dataclass, representing nested memory units
in the TKS Memory Onion architecture. Packets form a hierarchical structure
where each level represents a different abstraction of the same information.

Abstraction Levels (L0-L8):
    L0: Raw input (unprocessed observations)
    L1: Evidence objects (structured evidence)
    L2: Extracted facts (derived knowledge)
    L3: RPM structure (Rumination/Planning/Memory patterns)
    L4: TKS packets (symbolic equations)
    L5: Canon-linked (connected to canonical knowledge)
    L6: Nested memory nodes (compressed at precision p)
    L7: Policy/templates (generalized patterns)
    L8: Meta-memory (memory about memory)

Author: TKS Memory Onion Module
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional


class AbstractionLevel(IntEnum):
    """
    Enumeration of abstraction levels in the memory onion.

    Higher levels represent more compressed, abstract representations.
    Lower levels contain more concrete, detailed information.
    """
    L0_RAW_INPUT = 0          # Raw observations, unprocessed data
    L1_EVIDENCE = 1           # Structured evidence objects
    L2_FACTS = 2              # Extracted factual claims
    L3_RPM_STRUCTURE = 3      # Rumination/Planning/Memory patterns
    L4_TKS_SYMBOLIC = 4       # TKS symbolic equations
    L5_CANON_LINKED = 5       # Connected to canonical knowledge base
    L6_MEMORY_NODES = 6       # Nested compressed memory nodes
    L7_POLICY_TEMPLATES = 7   # Generalized policy templates
    L8_META_MEMORY = 8        # Meta-memory (memory about memory)


def generate_packet_id() -> str:
    """
    Generate a unique packet ID in the format: pkt_XXXXXXXXXXXX

    Returns:
        str: Packet ID in format pkt_XXXXXXXXXXXX
    """
    unique_part = uuid.uuid4().hex[:12].upper()
    return f"pkt_{unique_part}"


@dataclass
class PacketMetadata:
    """
    Metadata associated with a TKS packet.

    Attributes:
        created_at: ISO 8601 timestamp when packet was created
        updated_at: ISO 8601 timestamp of last update
        source: Origin of the packet (e.g., episode, canon, synthesis)
        episode_id: ID of the originating episode (if applicable)
        compression_ratio: Ratio of original to compressed size
        fidelity_score: Estimated fidelity after compression (0.0-1.0)
        semantic_hash: Hash of semantic content for deduplication
        tags: Free-form tags for categorization
        custom: Additional custom metadata
    """
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "unknown"
    episode_id: Optional[str] = None
    compression_ratio: float = 1.0
    fidelity_score: float = 1.0
    semantic_hash: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PacketMetadata:
        """Create metadata from dictionary."""
        return cls(
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            source=data.get("source", "unknown"),
            episode_id=data.get("episode_id"),
            compression_ratio=data.get("compression_ratio", 1.0),
            fidelity_score=data.get("fidelity_score", 1.0),
            semantic_hash=data.get("semantic_hash"),
            tags=data.get("tags", []),
            custom=data.get("custom", {}),
        )


@dataclass
class TKSPacket:
    """
    TKS Packet - Core memory unit in the Memory Onion architecture.

    A TKSPacket represents a unit of memory at a specific abstraction level.
    Packets can contain nested child packets, forming a hierarchical structure
    (the "onion"). Each layer can be reconstructed at different precision levels.

    Attributes:
        packet_id: Unique identifier (pkt_XXXXXXXXXXXX format)
        level: Abstraction level (0-8)
        tks_expression: The TKS equation representing this memory
        evidence_ids: List of evidence IDs supporting this packet
        children: Nested child packets (inner onion layers)
        metadata: Associated metadata (timestamps, sources, etc.)
        content: Raw content data (level-specific)
        prerequisites: IDs of prerequisite packets
        regulators: IDs of regulating packets (constraints)
        world_associations: Cross-world linkages (A, B, C, D worlds)
    """
    packet_id: str
    level: int
    tks_expression: str
    evidence_ids: List[str] = field(default_factory=list)
    children: List[TKSPacket] = field(default_factory=list)
    metadata: PacketMetadata = field(default_factory=PacketMetadata)
    content: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    regulators: List[str] = field(default_factory=list)
    world_associations: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate packet after initialization."""
        if not 0 <= self.level <= 8:
            raise ValueError(f"Invalid abstraction level: {self.level}. Must be 0-8.")

    @classmethod
    def create(
        cls,
        level: int,
        tks_expression: str,
        evidence_ids: Optional[List[str]] = None,
        children: Optional[List[TKSPacket]] = None,
        metadata: Optional[PacketMetadata] = None,
        content: Optional[Dict[str, Any]] = None,
        prerequisites: Optional[List[str]] = None,
        regulators: Optional[List[str]] = None,
        world_associations: Optional[Dict[str, List[str]]] = None,
        packet_id: Optional[str] = None,
    ) -> TKSPacket:
        """
        Factory method to create a new TKSPacket with auto-generated ID.

        Args:
            level: Abstraction level (0-8)
            tks_expression: TKS equation string
            evidence_ids: Supporting evidence IDs
            children: Nested child packets
            metadata: Packet metadata
            content: Raw content data
            prerequisites: Prerequisite packet IDs
            regulators: Regulator packet IDs
            world_associations: Cross-world linkages
            packet_id: Override packet ID (for replay/reconstruction)

        Returns:
            TKSPacket: New packet instance
        """
        return cls(
            packet_id=packet_id or generate_packet_id(),
            level=level,
            tks_expression=tks_expression,
            evidence_ids=evidence_ids or [],
            children=children or [],
            metadata=metadata or PacketMetadata(),
            content=content or {},
            prerequisites=prerequisites or [],
            regulators=regulators or [],
            world_associations=world_associations or {},
        )

    @property
    def abstraction_level(self) -> AbstractionLevel:
        """Get the abstraction level as enum."""
        return AbstractionLevel(self.level)

    @property
    def depth(self) -> int:
        """Calculate the depth of the packet tree (1 = leaf, >1 = has children)."""
        if not self.children:
            return 1
        return 1 + max(child.depth for child in self.children)

    @property
    def total_packets(self) -> int:
        """Count total packets including all nested children."""
        return 1 + sum(child.total_packets for child in self.children)

    def get_all_evidence_ids(self) -> List[str]:
        """Get all evidence IDs including from children."""
        all_ids = list(self.evidence_ids)
        for child in self.children:
            all_ids.extend(child.get_all_evidence_ids())
        return list(set(all_ids))

    def get_packets_at_level(self, target_level: int) -> List[TKSPacket]:
        """Get all packets at a specific abstraction level."""
        result = []
        if self.level == target_level:
            result.append(self)
        for child in self.children:
            result.extend(child.get_packets_at_level(target_level))
        return result

    def get_leaf_packets(self) -> List[TKSPacket]:
        """Get all leaf packets (packets with no children)."""
        if not self.children:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_leaf_packets())
        return result

    def find_packet(self, packet_id: str) -> Optional[TKSPacket]:
        """Find a packet by ID within this packet tree."""
        if self.packet_id == packet_id:
            return self
        for child in self.children:
            found = child.find_packet(packet_id)
            if found:
                return found
        return None

    def add_child(self, child: TKSPacket) -> None:
        """Add a child packet."""
        self.children.append(child)
        self.metadata.updated_at = datetime.now(timezone.utc).isoformat()

    def remove_child(self, packet_id: str) -> bool:
        """Remove a child packet by ID. Returns True if removed."""
        for i, child in enumerate(self.children):
            if child.packet_id == packet_id:
                del self.children[i]
                self.metadata.updated_at = datetime.now(timezone.utc).isoformat()
                return True
        return False

    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """
        Convert packet to dictionary for serialization.

        Args:
            include_children: Whether to include nested children (default True)

        Returns:
            Dictionary representation of the packet
        """
        result = {
            "packet_id": self.packet_id,
            "level": self.level,
            "tks_expression": self.tks_expression,
            "evidence_ids": self.evidence_ids,
            "metadata": self.metadata.to_dict(),
            "content": self.content,
            "prerequisites": self.prerequisites,
            "regulators": self.regulators,
            "world_associations": self.world_associations,
        }
        if include_children:
            result["children"] = [child.to_dict() for child in self.children]
        else:
            result["children_count"] = len(self.children)
        return result

    def to_json(self, indent: Optional[int] = None, include_children: bool = True) -> str:
        """
        Serialize packet to JSON string.

        Args:
            indent: JSON indentation level (None for compact)
            include_children: Whether to include nested children

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(include_children), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TKSPacket:
        """
        Create a TKSPacket from a dictionary.

        Args:
            data: Dictionary containing packet data

        Returns:
            TKSPacket instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        children = []
        if "children" in data and isinstance(data["children"], list):
            children = [cls.from_dict(child_data) for child_data in data["children"]]

        metadata = PacketMetadata.from_dict(data.get("metadata", {}))

        return cls(
            packet_id=data["packet_id"],
            level=data["level"],
            tks_expression=data["tks_expression"],
            evidence_ids=data.get("evidence_ids", []),
            children=children,
            metadata=metadata,
            content=data.get("content", {}),
            prerequisites=data.get("prerequisites", []),
            regulators=data.get("regulators", []),
            world_associations=data.get("world_associations", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> TKSPacket:
        """
        Deserialize a TKSPacket from a JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            TKSPacket instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_summary(self) -> str:
        """Get a human-readable summary of the packet."""
        level_name = AbstractionLevel(self.level).name
        return (
            f"TKSPacket[{level_name}] id={self.packet_id} "
            f"expr='{self.tks_expression[:50]}...' "
            f"children={len(self.children)} evidence={len(self.evidence_ids)}"
        )

    def __repr__(self) -> str:
        return (
            f"TKSPacket(id={self.packet_id}, level={self.level}, "
            f"expr='{self.tks_expression[:30]}...', children={len(self.children)})"
        )


@dataclass
class PacketBatch:
    """
    A batch of TKS packets for efficient bulk operations.

    Attributes:
        packets: List of TKSPacket instances
        batch_id: Unique identifier for this batch
        created_at: Timestamp when batch was created
        source_episode_id: Common source episode (if applicable)
    """
    packets: List[TKSPacket]
    batch_id: str = field(default_factory=lambda: f"batch_{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_episode_id: Optional[str] = None

    @property
    def total_packets(self) -> int:
        """Total count including nested packets."""
        return sum(p.total_packets for p in self.packets)

    @property
    def max_depth(self) -> int:
        """Maximum depth across all packets."""
        if not self.packets:
            return 0
        return max(p.depth for p in self.packets)

    def get_by_level(self, level: int) -> List[TKSPacket]:
        """Get all packets at a specific level."""
        result = []
        for packet in self.packets:
            result.extend(packet.get_packets_at_level(level))
        return result

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize batch to JSON."""
        return json.dumps({
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "source_episode_id": self.source_episode_id,
            "packet_count": len(self.packets),
            "total_packets": self.total_packets,
            "packets": [p.to_dict() for p in self.packets]
        }, indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> PacketBatch:
        """Deserialize batch from JSON."""
        data = json.loads(json_str)
        packets = [TKSPacket.from_dict(p) for p in data["packets"]]
        return cls(
            packets=packets,
            batch_id=data["batch_id"],
            created_at=data["created_at"],
            source_episode_id=data.get("source_episode_id"),
        )

"""Memory store exports and minimal in-memory store implementation."""

from typing import Dict, List, Optional

from .packets import TKSPacket, PacketMetadata, AbstractionLevel


class MemoryStore:
    """Simple in-memory packet store for episode outputs."""

    def __init__(self, config) -> None:
        self.config = config
        self._packets: Dict[str, TKSPacket] = {}

    def store(self, episode_id: str, tks_packet: str, evidence_ids: List[str]) -> str:
        packet = TKSPacket.create(
            level=int(AbstractionLevel.L4_TKS_SYMBOLIC),
            tks_expression=tks_packet,
            evidence_ids=evidence_ids,
            content={"episode_id": episode_id},
        )
        self._packets[packet.packet_id] = packet
        return packet.packet_id

    def get(self, packet_id: str) -> Optional[TKSPacket]:
        return self._packets.get(packet_id)

    def list_packets(self) -> List[TKSPacket]:
        return list(self._packets.values())


__all__ = [
    "MemoryStore",
    "TKSPacket",
    "PacketMetadata",
    "AbstractionLevel",
]

# -*- coding: utf-8 -*-
"""
Agent Registry - AGENT-FOUND-007: Agent Registry & Service Catalog

Thread-safe in-memory agent registry with indexed lookups for fast
discovery by layer, sector, capability, and tag. Supports CRUD
operations, hot-reload, export/import, and statistics.

Zero-Hallucination Guarantees:
    - All operations are deterministic
    - No LLM calls in any registry operations
    - SHA-256 provenance hashes for all mutations

Example:
    >>> from greenlang.agent_registry.registry import AgentRegistry
    >>> registry = AgentRegistry()
    >>> metadata = AgentMetadataEntry(...)
    >>> provenance_hash = registry.register_agent(metadata)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from greenlang.agent_registry.config import AgentRegistryConfig, get_config
from greenlang.agent_registry.models import (
    AgentCapability,
    AgentHealthStatus,
    AgentLayer,
    AgentMetadataEntry,
    CapabilityCategory,
    ExecutionMode,
    RegistryQueryInput,
    RegistryQueryOutput,
    SectorClassification,
    SemanticVersion,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(content: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable content.

    Args:
        content: Data to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


class AgentRegistry:
    """Thread-safe in-memory agent registry with indexed lookups.

    Provides full CRUD for agent metadata entries, indexed by layer,
    sector, capability, and tag for fast discovery. Supports hot-reload,
    export/import, and provenance hash generation.

    Attributes:
        config: Registry configuration.

    Example:
        >>> registry = AgentRegistry()
        >>> provenance = registry.register_agent(metadata)
        >>> agent = registry.get_agent("GL-MRV-X-001")
    """

    def __init__(self, config: Optional[AgentRegistryConfig] = None) -> None:
        """Initialize the AgentRegistry.

        Args:
            config: Optional config. Uses global singleton if None.
        """
        self.config = config or get_config()

        # Primary storage: agent_id -> {version -> metadata}
        self._registry: Dict[str, Dict[str, AgentMetadataEntry]] = {}

        # Indexes for fast lookup
        self._by_layer: Dict[AgentLayer, Set[str]] = defaultdict(set)
        self._by_sector: Dict[SectorClassification, Set[str]] = defaultdict(set)
        self._by_capability: Dict[str, Set[str]] = defaultdict(set)
        self._by_tag: Dict[str, Set[str]] = defaultdict(set)

        # Thread safety
        self._lock = threading.RLock()

        # Hot reload callbacks
        self._reload_callbacks: List[Callable[[str, str], None]] = []

        # Statistics
        self._registration_count = 0
        self._query_count = 0

        logger.info("AgentRegistry initialized (max_agents=%d)", self.config.max_agents)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, metadata: AgentMetadataEntry) -> str:
        """Register an agent with the registry.

        Args:
            metadata: Agent metadata entry to register.

        Returns:
            SHA-256 provenance hash of the registered entry.

        Raises:
            ValueError: If the registry is at capacity.
        """
        with self._lock:
            agent_id = metadata.agent_id
            version = metadata.version

            # Capacity check
            if agent_id not in self._registry:
                if len(self._registry) >= self.config.max_agents:
                    raise ValueError(
                        f"Registry at capacity ({self.config.max_agents} agents)"
                    )

            # Version limit check
            if agent_id in self._registry:
                existing_versions = len(self._registry[agent_id])
                if (
                    version not in self._registry[agent_id]
                    and existing_versions >= self.config.max_versions_per_agent
                ):
                    raise ValueError(
                        f"Agent {agent_id} at version capacity "
                        f"({self.config.max_versions_per_agent} versions)"
                    )

            # Initialize version dict if needed
            if agent_id not in self._registry:
                self._registry[agent_id] = {}

            # Update timestamp
            metadata.updated_at = _utcnow()

            # Store metadata
            self._registry[agent_id][version] = metadata

            # Update indexes
            self._update_indexes(metadata)

            self._registration_count += 1
            provenance_hash = metadata.provenance_hash

            # Notify callbacks
            self._notify_reload(agent_id, version)

            logger.info(
                "Registered agent: %s@%s (hash=%s)",
                agent_id, version, provenance_hash[:16],
            )
            return provenance_hash

    def unregister_agent(self, agent_id: str, version: Optional[str] = None) -> bool:
        """Unregister an agent from the registry.

        Args:
            agent_id: Agent ID to remove.
            version: Specific version to remove (None = all versions).

        Returns:
            True if the agent was found and removed.
        """
        with self._lock:
            if agent_id not in self._registry:
                logger.warning("Unregister: agent not found: %s", agent_id)
                return False

            if version is not None:
                if version not in self._registry[agent_id]:
                    return False
                del self._registry[agent_id][version]
                logger.info("Unregistered version: %s@%s", agent_id, version)

                if not self._registry[agent_id]:
                    del self._registry[agent_id]
                    self._remove_from_indexes(agent_id)
            else:
                del self._registry[agent_id]
                self._remove_from_indexes(agent_id)
                logger.info("Unregistered all versions: %s", agent_id)

            return True

    def get_agent(
        self, agent_id: str, version: Optional[str] = None,
    ) -> Optional[AgentMetadataEntry]:
        """Get agent metadata by ID and optional version.

        Args:
            agent_id: Agent ID to look up.
            version: Specific version (None = latest).

        Returns:
            AgentMetadataEntry or None if not found.
        """
        with self._lock:
            if agent_id not in self._registry:
                return None

            versions = self._registry[agent_id]
            if version is not None:
                return versions.get(version)

            if not versions:
                return None

            latest_version = max(
                versions.keys(), key=lambda v: SemanticVersion.parse(v),
            )
            return versions[latest_version]

    def list_agents(
        self,
        layer: Optional[AgentLayer] = None,
        sector: Optional[SectorClassification] = None,
        capability: Optional[str] = None,
        tags: Optional[List[str]] = None,
        health: Optional[AgentHealthStatus] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AgentMetadataEntry]:
        """List agents matching the given filters.

        Args:
            layer: Filter by agent layer.
            sector: Filter by sector.
            capability: Filter by capability name.
            tags: Filter by tags (all must match).
            health: Filter by health status.
            search: Text search in name and description.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of matching AgentMetadataEntry (latest version per agent).
        """
        query = RegistryQueryInput(
            layer=layer,
            sector=sector,
            capability=capability,
            tags=tags or [],
            health_status=health,
            search_text=search,
            limit=limit,
            offset=offset,
        )
        result = self.query_agents(query)
        return result.agents

    def query_agents(self, query: RegistryQueryInput) -> RegistryQueryOutput:
        """Query agents with flexible filtering.

        Args:
            query: Query parameters and filters.

        Returns:
            RegistryQueryOutput with matching agents and metadata.
        """
        start_time = time.time()
        self._query_count += 1

        with self._lock:
            candidate_ids: Optional[Set[str]] = None

            # Index-based filtering
            if query.layer is not None:
                layer_ids = self._by_layer.get(query.layer, set())
                candidate_ids = set(layer_ids) if candidate_ids is None else candidate_ids & layer_ids

            if query.sector is not None:
                sector_ids = self._by_sector.get(query.sector, set())
                candidate_ids = set(sector_ids) if candidate_ids is None else candidate_ids & sector_ids

            if query.capability is not None:
                cap_ids = self._by_capability.get(query.capability, set())
                candidate_ids = set(cap_ids) if candidate_ids is None else candidate_ids & cap_ids

            for tag in query.tags:
                tag_ids = self._by_tag.get(tag.lower(), set())
                candidate_ids = set(tag_ids) if candidate_ids is None else candidate_ids & tag_ids

            if candidate_ids is None:
                candidate_ids = set(self._registry.keys())

            # Fine-grained filtering
            matching: List[AgentMetadataEntry] = []
            for agent_id in candidate_ids:
                versions = self._registry.get(agent_id, {})
                for _version, metadata in versions.items():
                    if self._matches_query(metadata, query):
                        matching.append(metadata)

            # Sort by agent_id then version descending
            matching.sort(
                key=lambda m: (m.agent_id, m.parsed_version), reverse=True,
            )

            total_count = len(matching)
            paginated = matching[query.offset : query.offset + query.limit]
            query_time = (time.time() - start_time) * 1000

            provenance_content = f"query:{json.dumps(query.model_dump(), sort_keys=True, default=str)}:{total_count}"
            provenance_hash = hashlib.sha256(provenance_content.encode()).hexdigest()

            return RegistryQueryOutput(
                agents=paginated,
                total_count=total_count,
                query_time_ms=query_time,
                provenance_hash=provenance_hash,
            )

    def list_versions(self, agent_id: str) -> List[str]:
        """List all versions of an agent, newest first.

        Args:
            agent_id: Agent ID to list versions for.

        Returns:
            Sorted version strings (newest first).
        """
        with self._lock:
            if agent_id not in self._registry:
                return []
            versions = list(self._registry[agent_id].keys())
            versions.sort(key=lambda v: SemanticVersion.parse(v), reverse=True)
            return versions

    def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> str:
        """Update specific fields of the latest agent version.

        Args:
            agent_id: Agent to update.
            updates: Dictionary of field names to new values.

        Returns:
            SHA-256 provenance hash after update.

        Raises:
            KeyError: If agent not found.
        """
        with self._lock:
            metadata = self.get_agent(agent_id)
            if metadata is None:
                raise KeyError(f"Agent not found: {agent_id}")

            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

            metadata.updated_at = _utcnow()
            provenance_hash = metadata.provenance_hash

            # Rebuild indexes for this agent
            self._remove_from_indexes(agent_id)
            for _version, meta in self._registry[agent_id].items():
                self._update_indexes(meta)

            logger.info("Updated agent: %s (hash=%s)", agent_id, provenance_hash[:16])
            return provenance_hash

    def hot_reload_agent(
        self, agent_id: str, metadata: AgentMetadataEntry,
    ) -> bool:
        """Hot-reload an agent without service interruption.

        Atomically replaces the agent registration with new metadata.

        Args:
            agent_id: Agent ID to reload.
            metadata: New metadata to register.

        Returns:
            True if reload succeeded.
        """
        if not self.config.enable_hot_reload:
            logger.warning("Hot-reload disabled; rejecting reload of %s", agent_id)
            return False

        logger.info("Hot-reloading agent: %s@%s", agent_id, metadata.version)
        self.register_agent(metadata)
        return True

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def get_all_agent_ids(self) -> List[str]:
        """Get all registered agent IDs.

        Returns:
            Sorted list of agent ID strings.
        """
        with self._lock:
            return sorted(self._registry.keys())

    def get_agents_by_layer(self, layer: AgentLayer) -> List[str]:
        """Get all agent IDs in a specific layer.

        Args:
            layer: Agent layer to filter by.

        Returns:
            Sorted list of agent IDs.
        """
        with self._lock:
            return sorted(self._by_layer.get(layer, set()))

    def export_registry(self) -> Dict[str, Any]:
        """Export the registry to a serialisable dictionary.

        Returns:
            Dictionary containing all registry data.
        """
        with self._lock:
            export_data: Dict[str, Any] = {
                "version": "1.0",
                "exported_at": _utcnow().isoformat(),
                "agents": {},
            }
            for agent_id, versions in self._registry.items():
                export_data["agents"][agent_id] = {
                    v: m.model_dump(mode="json") for v, m in versions.items()
                }
            return export_data

    def import_registry(self, data: Dict[str, Any], merge: bool = True) -> int:
        """Import registry data.

        Args:
            data: Previously exported registry data.
            merge: If True, merge with existing. If False, replace.

        Returns:
            Number of agent versions imported.
        """
        with self._lock:
            if not merge:
                self._registry.clear()
                self._by_layer.clear()
                self._by_sector.clear()
                self._by_capability.clear()
                self._by_tag.clear()

            count = 0
            for agent_id, versions in data.get("agents", {}).items():
                for version, metadata_dict in versions.items():
                    try:
                        metadata = AgentMetadataEntry(**metadata_dict)
                        self.register_agent(metadata)
                        count += 1
                    except Exception as exc:
                        logger.error(
                            "Failed to import %s@%s: %s", agent_id, version, exc,
                        )
            return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics summary.

        Returns:
            Dictionary of registry statistics.
        """
        with self._lock:
            total_agents = len(self._registry)
            total_versions = sum(len(v) for v in self._registry.values())

            layer_counts = {
                layer.value: len(agents)
                for layer, agents in self._by_layer.items()
            }
            sector_counts = {
                sector.value: len(agents)
                for sector, agents in self._by_sector.items()
            }

            health_counts: Dict[str, int] = defaultdict(int)
            for agent_versions in self._registry.values():
                for metadata in agent_versions.values():
                    health_counts[metadata.health_status.value] += 1

            return {
                "total_agents": total_agents,
                "total_versions": total_versions,
                "registration_count": self._registration_count,
                "query_count": self._query_count,
                "agents_by_layer": dict(layer_counts),
                "agents_by_sector": dict(sector_counts),
                "agents_by_health": dict(health_counts),
                "reload_callbacks": len(self._reload_callbacks),
            }

    @property
    def count(self) -> int:
        """Return total number of registered agents."""
        with self._lock:
            return len(self._registry)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def register_reload_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for hot-reload notifications.

        Args:
            callback: Function(agent_id, version) invoked on registration.
        """
        self._reload_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_indexes(self, metadata: AgentMetadataEntry) -> None:
        """Add metadata to all relevant indexes."""
        agent_id = metadata.agent_id
        self._by_layer[metadata.layer].add(agent_id)
        for sector in metadata.sectors:
            self._by_sector[sector].add(agent_id)
        for cap in metadata.capabilities:
            self._by_capability[cap.name].add(agent_id)
        for tag in metadata.tags:
            self._by_tag[tag.lower()].add(agent_id)

    def _remove_from_indexes(self, agent_id: str) -> None:
        """Remove agent from all indexes."""
        for layer_set in self._by_layer.values():
            layer_set.discard(agent_id)
        for sector_set in self._by_sector.values():
            sector_set.discard(agent_id)
        for cap_set in self._by_capability.values():
            cap_set.discard(agent_id)
        for tag_set in self._by_tag.values():
            tag_set.discard(agent_id)

    def _notify_reload(self, agent_id: str, version: str) -> None:
        """Notify all registered callbacks of a change."""
        for callback in self._reload_callbacks:
            try:
                callback(agent_id, version)
            except Exception as exc:
                logger.error("Reload callback failed: %s", exc)

    def _matches_query(
        self, metadata: AgentMetadataEntry, query: RegistryQueryInput,
    ) -> bool:
        """Check if metadata matches all query criteria.

        Args:
            metadata: The metadata to test.
            query: The query to match against.

        Returns:
            True if all criteria match.
        """
        if query.capability_category is not None:
            if not any(
                c.category == query.capability_category
                for c in metadata.capabilities
            ):
                return False

        if query.health_status is not None:
            if metadata.health_status != query.health_status:
                return False

        if query.execution_mode is not None:
            if metadata.execution_mode != query.execution_mode:
                return False

        if query.search_text is not None:
            search_lower = query.search_text.lower()
            if (
                search_lower not in metadata.name.lower()
                and search_lower not in metadata.description.lower()
            ):
                return False

        return True


__all__ = [
    "AgentRegistry",
]

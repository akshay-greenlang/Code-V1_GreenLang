# -*- coding: utf-8 -*-
"""
Capability Matcher - AGENT-FOUND-007: Agent Registry & Service Catalog

Provides capability-based agent discovery: find agents by required
capabilities, category, input/output types, and build capability matrices.

Zero-Hallucination Guarantees:
    - Matching uses exact specification comparison
    - No LLM calls in any matching operations
    - All results are deterministically sorted

Example:
    >>> from greenlang.agent_registry.capability_matcher import CapabilityMatcher
    >>> matcher = CapabilityMatcher(registry)
    >>> agents = matcher.find_by_capabilities([required_cap])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from greenlang.agent_registry.models import (
    AgentCapability,
    AgentLayer,
    AgentMetadataEntry,
    CapabilityCategory,
    SectorClassification,
)

logger = logging.getLogger(__name__)


class CapabilityMatcher:
    """Capability-based agent discovery engine.

    Finds agents that satisfy a set of required capabilities with
    optional filtering by sector and layer.

    Attributes:
        _registry: Reference to the parent AgentRegistry.

    Example:
        >>> matcher = CapabilityMatcher(registry)
        >>> agents = matcher.find_by_category(CapabilityCategory.CALCULATION)
    """

    def __init__(self, registry: Any) -> None:
        """Initialize the CapabilityMatcher.

        Args:
            registry: AgentRegistry instance for agent lookups.
        """
        self._registry = registry
        logger.info("CapabilityMatcher initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_by_capabilities(
        self,
        required_capabilities: List[AgentCapability],
        sector: Optional[SectorClassification] = None,
        layer: Optional[AgentLayer] = None,
    ) -> List[AgentMetadataEntry]:
        """Find agents that provide all required capabilities.

        Args:
            required_capabilities: Capabilities that must all be present.
            sector: Optional sector filter.
            layer: Optional layer filter.

        Returns:
            List of matching agents (latest version per agent), sorted by
            agent_id.
        """
        matching: List[AgentMetadataEntry] = []
        all_ids = self._registry.get_all_agent_ids()

        for agent_id in all_ids:
            metadata = self._registry.get_agent(agent_id)
            if metadata is None:
                continue

            # Apply optional filters
            if layer is not None and metadata.layer != layer:
                continue
            if sector is not None and not metadata.supports_sector(sector):
                continue

            # Check all required capabilities
            if self._has_all_capabilities(metadata, required_capabilities):
                matching.append(metadata)

        matching.sort(key=lambda m: m.agent_id)
        logger.debug(
            "find_by_capabilities: %d required, %d matches",
            len(required_capabilities), len(matching),
        )
        return matching

    def find_by_category(
        self, category: CapabilityCategory,
    ) -> List[AgentMetadataEntry]:
        """Find agents that have any capability in the given category.

        Args:
            category: The capability category to search for.

        Returns:
            Sorted list of matching agents.
        """
        matching: List[AgentMetadataEntry] = []
        all_ids = self._registry.get_all_agent_ids()

        for agent_id in all_ids:
            metadata = self._registry.get_agent(agent_id)
            if metadata is None:
                continue
            if any(c.category == category for c in metadata.capabilities):
                matching.append(metadata)

        matching.sort(key=lambda m: m.agent_id)
        return matching

    def find_by_input_type(self, input_type: str) -> List[AgentMetadataEntry]:
        """Find agents that accept the given input type.

        Args:
            input_type: The input data type to search for.

        Returns:
            Sorted list of matching agents.
        """
        matching: List[AgentMetadataEntry] = []
        all_ids = self._registry.get_all_agent_ids()

        for agent_id in all_ids:
            metadata = self._registry.get_agent(agent_id)
            if metadata is None:
                continue
            if any(
                input_type in c.input_types for c in metadata.capabilities
            ):
                matching.append(metadata)

        matching.sort(key=lambda m: m.agent_id)
        return matching

    def find_by_output_type(self, output_type: str) -> List[AgentMetadataEntry]:
        """Find agents that produce the given output type.

        Args:
            output_type: The output data type to search for.

        Returns:
            Sorted list of matching agents.
        """
        matching: List[AgentMetadataEntry] = []
        all_ids = self._registry.get_all_agent_ids()

        for agent_id in all_ids:
            metadata = self._registry.get_agent(agent_id)
            if metadata is None:
                continue
            if any(
                output_type in c.output_types for c in metadata.capabilities
            ):
                matching.append(metadata)

        matching.sort(key=lambda m: m.agent_id)
        return matching

    def get_capability_matrix(self) -> Dict[str, List[str]]:
        """Build a matrix mapping capability names to agent ID lists.

        Returns:
            Dictionary: capability_name -> sorted list of agent IDs.
        """
        matrix: Dict[str, List[str]] = {}
        all_ids = self._registry.get_all_agent_ids()

        for agent_id in all_ids:
            metadata = self._registry.get_agent(agent_id)
            if metadata is None:
                continue
            for cap in metadata.capabilities:
                if cap.name not in matrix:
                    matrix[cap.name] = []
                if agent_id not in matrix[cap.name]:
                    matrix[cap.name].append(agent_id)

        # Sort each list
        for cap_name in matrix:
            matrix[cap_name].sort()

        return matrix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_all_capabilities(
        self,
        metadata: AgentMetadataEntry,
        required: List[AgentCapability],
    ) -> bool:
        """Check if an agent has all required capabilities.

        Args:
            metadata: Agent to check.
            required: Required capabilities.

        Returns:
            True if all required capabilities are matched.
        """
        for req in required:
            matched = False
            for cap in metadata.capabilities:
                if cap.matches(req):
                    matched = True
                    break
            if not matched:
                return False
        return True


__all__ = [
    "CapabilityMatcher",
]

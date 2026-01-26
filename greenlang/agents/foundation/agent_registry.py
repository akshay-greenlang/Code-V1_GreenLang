# -*- coding: utf-8 -*-
"""
GL-FOUND-X-010: Agent Registry & Versioning Agent
==================================================

A comprehensive catalog of all available agents in the GreenLang Climate OS.
This agent maintains agent metadata, versions, capabilities, dependencies,
and supports runtime hot-reload for dynamic agent registration.

Capabilities:
    - Agent Registration with full metadata, version, and capabilities
    - Agent Discovery by type, capability, layer, sector, and variants
    - Version Management with semantic versioning and compatibility tracking
    - Capability Matching for finding agents that meet required capabilities
    - Dependency Resolution for multi-agent pipeline construction
    - Hot Reload Support for runtime agent registration without restart
    - Health Status Tracking for agent availability monitoring

Zero-Hallucination Guarantees:
    - All registry operations are deterministic
    - Capability matching uses exact specification matching
    - Version resolution follows semantic versioning rules
    - No LLM calls in any registry operations

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Taxonomy Enums
# =============================================================================

class AgentLayer(str, Enum):
    """
    11-layer agent taxonomy for GreenLang Climate OS.

    Each layer represents a specific domain of climate intelligence capabilities.
    Layers are organized from foundational infrastructure to specialized tooling.
    """
    FOUNDATION = "foundation"       # Core infrastructure (GL-FOUND-X-XXX)
    DATA = "data"                   # Data ingestion and transformation (GL-DATA-X-XXX)
    MRV = "mrv"                     # Measurement, Reporting, Verification (GL-MRV-X-XXX)
    PLANNING = "planning"           # Decarbonization planning (GL-PLAN-X-XXX)
    RISK = "risk"                   # Climate risk assessment (GL-RISK-X-XXX)
    FINANCE = "finance"             # Green finance and carbon markets (GL-FIN-X-XXX)
    PROCUREMENT = "procurement"     # Sustainable procurement (GL-PROC-X-XXX)
    POLICY = "policy"               # Regulatory compliance (GL-POL-X-XXX)
    REPORTING = "reporting"         # Disclosure and reporting (GL-REP-X-XXX)
    OPERATIONS = "operations"       # Operational optimization (GL-OPS-X-XXX)
    DEVTOOLS = "devtools"           # Developer tools and testing (GL-DEV-X-XXX)

    @property
    def prefix(self) -> str:
        """Get the agent ID prefix for this layer."""
        prefixes = {
            AgentLayer.FOUNDATION: "GL-FOUND",
            AgentLayer.DATA: "GL-DATA",
            AgentLayer.MRV: "GL-MRV",
            AgentLayer.PLANNING: "GL-PLAN",
            AgentLayer.RISK: "GL-RISK",
            AgentLayer.FINANCE: "GL-FIN",
            AgentLayer.PROCUREMENT: "GL-PROC",
            AgentLayer.POLICY: "GL-POL",
            AgentLayer.REPORTING: "GL-REP",
            AgentLayer.OPERATIONS: "GL-OPS",
            AgentLayer.DEVTOOLS: "GL-DEV",
        }
        return prefixes[self]

    @property
    def description(self) -> str:
        """Get human-readable description of this layer."""
        descriptions = {
            AgentLayer.FOUNDATION: "Core infrastructure agents for orchestration, validation, and system services",
            AgentLayer.DATA: "Data ingestion, transformation, and quality assurance agents",
            AgentLayer.MRV: "Measurement, Reporting, and Verification agents for emissions accounting",
            AgentLayer.PLANNING: "Decarbonization pathway planning and scenario modeling agents",
            AgentLayer.RISK: "Climate risk assessment, TCFD alignment, and scenario analysis agents",
            AgentLayer.FINANCE: "Green finance, carbon markets, and climate investment agents",
            AgentLayer.PROCUREMENT: "Sustainable procurement and supplier engagement agents",
            AgentLayer.POLICY: "Regulatory compliance and policy alignment agents",
            AgentLayer.REPORTING: "Disclosure, reporting, and stakeholder communication agents",
            AgentLayer.OPERATIONS: "Operational optimization and efficiency improvement agents",
            AgentLayer.DEVTOOLS: "Developer tools, testing frameworks, and debugging agents",
        }
        return descriptions[self]


class SectorClassification(str, Enum):
    """
    10-sector classification for agent specialization.

    Aligned with GICS sectors and climate disclosure frameworks.
    """
    ENERGY = "energy"                       # Oil, gas, renewables
    UTILITIES = "utilities"                 # Power generation, distribution
    MATERIALS = "materials"                 # Chemicals, metals, mining
    INDUSTRIALS = "industrials"             # Manufacturing, construction
    CONSUMER_DISCRETIONARY = "consumer_discretionary"  # Automotive, retail
    CONSUMER_STAPLES = "consumer_staples"   # Food, beverage, agriculture
    HEALTHCARE = "healthcare"               # Pharmaceuticals, medical
    FINANCIALS = "financials"               # Banking, insurance, investment
    REAL_ESTATE = "real_estate"             # Buildings, property
    INFORMATION_TECHNOLOGY = "information_technology"  # Data centers, electronics

    @property
    def description(self) -> str:
        """Get human-readable description of this sector."""
        descriptions = {
            SectorClassification.ENERGY: "Oil & gas, renewables, and energy production",
            SectorClassification.UTILITIES: "Power generation, transmission, and distribution",
            SectorClassification.MATERIALS: "Chemicals, metals, mining, and raw materials",
            SectorClassification.INDUSTRIALS: "Manufacturing, construction, and industrial processes",
            SectorClassification.CONSUMER_DISCRETIONARY: "Automotive, retail, and consumer goods",
            SectorClassification.CONSUMER_STAPLES: "Food, beverage, and agriculture",
            SectorClassification.HEALTHCARE: "Pharmaceuticals, medical devices, and healthcare",
            SectorClassification.FINANCIALS: "Banking, insurance, and investment services",
            SectorClassification.REAL_ESTATE: "Buildings, property, and real estate management",
            SectorClassification.INFORMATION_TECHNOLOGY: "Data centers, electronics, and IT services",
        }
        return descriptions[self]


class AgentHealthStatus(str, Enum):
    """Health status of an agent in the registry."""
    HEALTHY = "healthy"             # Agent is operational
    DEGRADED = "degraded"           # Agent has reduced functionality
    UNHEALTHY = "unhealthy"         # Agent is not operational
    UNKNOWN = "unknown"             # Health status not checked
    DISABLED = "disabled"           # Agent explicitly disabled


class CapabilityCategory(str, Enum):
    """Categories of agent capabilities."""
    CALCULATION = "calculation"     # Numeric calculations
    TRANSFORMATION = "transformation"  # Data transformation
    VALIDATION = "validation"       # Data validation
    AGGREGATION = "aggregation"     # Data aggregation
    CLASSIFICATION = "classification"  # Classification/categorization
    GENERATION = "generation"       # Content generation
    INTEGRATION = "integration"     # External system integration
    ORCHESTRATION = "orchestration"  # Pipeline orchestration
    ANALYSIS = "analysis"           # Data analysis
    REPORTING = "reporting"         # Report generation


# =============================================================================
# Pydantic Models for Agent Metadata
# =============================================================================

class SemanticVersion(BaseModel):
    """Semantic version representation with comparison support."""
    major: int = Field(ge=0, description="Major version (breaking changes)")
    minor: int = Field(ge=0, description="Minor version (new features)")
    patch: int = Field(ge=0, description="Patch version (bug fixes)")
    prerelease: Optional[str] = Field(None, description="Prerelease identifier")
    build: Optional[str] = Field(None, description="Build metadata")

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse a version string into SemanticVersion."""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$'
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid semantic version: {version_str}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5)
        )

    def __str__(self) -> str:
        """Convert to version string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        """Compare versions for sorting."""
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        # Handle prerelease comparison
        if self.prerelease and not other.prerelease:
            return True  # Prerelease is less than release
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        return False

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Check if this version is compatible with another (same major version)."""
        return self.major == other.major


class AgentCapability(BaseModel):
    """Definition of a specific agent capability."""
    name: str = Field(..., description="Capability name (unique identifier)")
    category: CapabilityCategory = Field(..., description="Capability category")
    description: str = Field(..., description="Human-readable description")
    input_types: List[str] = Field(default_factory=list, description="Supported input data types")
    output_types: List[str] = Field(default_factory=list, description="Output data types")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")

    def matches(self, required: "AgentCapability") -> bool:
        """Check if this capability matches a required capability."""
        if self.name != required.name:
            return False
        if self.category != required.category:
            return False
        # Check that all required input types are supported
        for input_type in required.input_types:
            if input_type not in self.input_types:
                return False
        return True


class AgentVariant(BaseModel):
    """Agent variant for geographic or fuel-type specialization."""
    variant_type: str = Field(..., description="Type of variant (geography, fuel_type, protocol, etc.)")
    variant_value: str = Field(..., description="Specific variant value")
    description: Optional[str] = Field(None, description="Description of the variant")

    @property
    def key(self) -> str:
        """Get unique key for this variant."""
        return f"{self.variant_type}:{self.variant_value}"


class AgentDependency(BaseModel):
    """Definition of an agent dependency."""
    agent_id: str = Field(..., description="Dependent agent ID")
    version_constraint: str = Field(default="*", description="Version constraint (e.g., '>=1.0.0', '^2.0.0')")
    optional: bool = Field(default=False, description="Whether dependency is optional")
    reason: Optional[str] = Field(None, description="Why this dependency is needed")

    def version_satisfies(self, version: SemanticVersion) -> bool:
        """Check if a version satisfies this dependency constraint."""
        if self.version_constraint == "*":
            return True

        # Parse constraint (simplified - full semver spec is more complex)
        constraint = self.version_constraint

        if constraint.startswith(">="):
            min_version = SemanticVersion.parse(constraint[2:])
            return not version < min_version
        elif constraint.startswith("^"):
            # Caret: compatible with major version
            target = SemanticVersion.parse(constraint[1:])
            return version.is_compatible_with(target) and not version < target
        elif constraint.startswith("~"):
            # Tilde: compatible with minor version
            target = SemanticVersion.parse(constraint[1:])
            return (version.major == target.major and
                    version.minor == target.minor and
                    version.patch >= target.patch)
        elif constraint.startswith("="):
            target = SemanticVersion.parse(constraint[1:])
            return str(version) == str(target)
        else:
            # Exact match
            target = SemanticVersion.parse(constraint)
            return str(version) == str(target)


class AgentMetadataEntry(BaseModel):
    """
    Complete metadata entry for a registered agent.

    This contains all information needed to discover, instantiate,
    and integrate an agent into a pipeline.
    """
    # Identity
    agent_id: str = Field(..., description="Unique agent identifier (e.g., GL-MRV-X-001)")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(..., description="Detailed description of agent purpose")

    # Versioning
    version: str = Field(..., description="Semantic version string")

    # Taxonomy
    layer: AgentLayer = Field(..., description="Agent layer in 11-layer taxonomy")
    sectors: List[SectorClassification] = Field(default_factory=list, description="Applicable sectors")

    # Capabilities
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")

    # Variants
    variants: List[AgentVariant] = Field(default_factory=list, description="Agent variants")

    # Dependencies
    dependencies: List[AgentDependency] = Field(default_factory=list, description="Required dependencies")

    # Runtime
    agent_class: Optional[str] = Field(None, description="Fully qualified class name")
    module_path: Optional[str] = Field(None, description="Module path for import")

    # Health
    health_status: AgentHealthStatus = Field(default=AgentHealthStatus.UNKNOWN, description="Current health status")
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")

    # Metadata
    author: Optional[str] = Field(None, description="Agent author")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")

    # Audit
    registered_at: datetime = Field(default_factory=DeterministicClock.now, description="Registration timestamp")
    updated_at: datetime = Field(default_factory=DeterministicClock.now, description="Last update timestamp")

    @field_validator('agent_id')
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent ID format."""
        pattern = r'^GL-[A-Z]+-[A-Z]-\d{3}$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid agent ID format: {v}. Expected format: GL-LAYER-X-NNN")
        return v

    @property
    def parsed_version(self) -> SemanticVersion:
        """Get parsed semantic version."""
        return SemanticVersion.parse(self.version)

    @property
    def provenance_hash(self) -> str:
        """Generate provenance hash for this metadata entry."""
        content = f"{self.agent_id}:{self.version}:{self.name}:{len(self.capabilities)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability."""
        return any(c.name == capability_name for c in self.capabilities)

    def has_variant(self, variant_type: str, variant_value: Optional[str] = None) -> bool:
        """Check if agent has a specific variant."""
        for v in self.variants:
            if v.variant_type == variant_type:
                if variant_value is None or v.variant_value == variant_value:
                    return True
        return False

    def supports_sector(self, sector: SectorClassification) -> bool:
        """Check if agent supports a specific sector."""
        return len(self.sectors) == 0 or sector in self.sectors


class RegistryQueryInput(BaseModel):
    """Input for registry query operations."""
    # Filter criteria
    layer: Optional[AgentLayer] = Field(None, description="Filter by layer")
    sector: Optional[SectorClassification] = Field(None, description="Filter by sector")
    capability: Optional[str] = Field(None, description="Filter by capability name")
    capability_category: Optional[CapabilityCategory] = Field(None, description="Filter by capability category")
    variant_type: Optional[str] = Field(None, description="Filter by variant type")
    variant_value: Optional[str] = Field(None, description="Filter by variant value")
    tags: List[str] = Field(default_factory=list, description="Filter by tags (all must match)")
    health_status: Optional[AgentHealthStatus] = Field(None, description="Filter by health status")

    # Version constraints
    min_version: Optional[str] = Field(None, description="Minimum version constraint")
    max_version: Optional[str] = Field(None, description="Maximum version constraint")

    # Search
    search_text: Optional[str] = Field(None, description="Text search in name and description")

    # Pagination
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class RegistryQueryOutput(BaseModel):
    """Output from registry query operations."""
    agents: List[AgentMetadataEntry] = Field(default_factory=list, description="Matching agents")
    total_count: int = Field(default=0, description="Total matching count (before pagination)")
    query_time_ms: float = Field(default=0.0, description="Query execution time")
    provenance_hash: str = Field(default="", description="Hash for audit trail")


class DependencyResolutionInput(BaseModel):
    """Input for dependency resolution."""
    agent_ids: List[str] = Field(..., description="Agent IDs to resolve dependencies for")
    include_optional: bool = Field(default=False, description="Include optional dependencies")
    fail_on_missing: bool = Field(default=True, description="Fail if dependency is missing")


class DependencyResolutionOutput(BaseModel):
    """Output from dependency resolution."""
    resolved_order: List[str] = Field(default_factory=list, description="Topologically sorted agent IDs")
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict, description="Dependency graph")
    missing_dependencies: List[str] = Field(default_factory=list, description="Missing dependency IDs")
    circular_dependencies: List[List[str]] = Field(default_factory=list, description="Detected cycles")
    resolution_time_ms: float = Field(default=0.0, description="Resolution time")
    success: bool = Field(default=True, description="Whether resolution succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


# =============================================================================
# Versioned Agent Registry Implementation
# =============================================================================

class VersionedAgentRegistry(BaseAgent):
    """
    GL-FOUND-X-010: Agent Registry & Versioning Agent

    A comprehensive catalog of all available agents in the GreenLang Climate OS.
    Provides agent registration, discovery, version management, capability matching,
    and dependency resolution with support for hot-reload.

    Zero-Hallucination Guarantees:
        - All registry operations are deterministic
        - Capability matching uses exact specification matching
        - Version resolution follows semantic versioning rules
        - No LLM calls in any registry operations

    Thread Safety:
        - Uses read-write locks for concurrent access
        - Supports hot-reload without service interruption

    Usage:
        registry = VersionedAgentRegistry()

        # Register an agent
        registry.register_agent(metadata)

        # Discover agents by capability
        result = registry.query_agents(RegistryQueryInput(capability="emissions_calculation"))

        # Resolve dependencies
        resolution = registry.resolve_dependencies(DependencyResolutionInput(
            agent_ids=["GL-MRV-X-001", "GL-MRV-X-002"]
        ))
    """

    AGENT_ID = "GL-FOUND-X-010"
    AGENT_NAME = "Agent Registry & Versioning Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Versioned Agent Registry."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Comprehensive catalog of all available agents in GreenLang Climate OS",
                version=self.VERSION,
                parameters={
                    "enable_hot_reload": True,
                    "health_check_interval_seconds": 60,
                    "cache_ttl_seconds": 300,
                }
            )
        super().__init__(config)

        # Primary storage: agent_id -> {version -> metadata}
        self._registry: Dict[str, Dict[str, AgentMetadataEntry]] = {}

        # Indexes for fast lookup
        self._by_layer: Dict[AgentLayer, Set[str]] = defaultdict(set)
        self._by_sector: Dict[SectorClassification, Set[str]] = defaultdict(set)
        self._by_capability: Dict[str, Set[str]] = defaultdict(set)
        self._by_tag: Dict[str, Set[str]] = defaultdict(set)

        # Agent class registry for instantiation
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Hot reload callbacks
        self._reload_callbacks: List[Callable[[str, str], None]] = []

        # Statistics
        self._registration_count = 0
        self._query_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute registry operations based on input data.

        Supported operations:
            - register: Register a new agent
            - unregister: Remove an agent
            - query: Query agents with filters
            - resolve_dependencies: Resolve agent dependencies
            - get_agent: Get specific agent metadata
            - health_check: Check agent health

        Args:
            input_data: Dictionary with 'operation' and operation-specific data

        Returns:
            AgentResult with operation results
        """
        start_time = time.time()
        operation = input_data.get("operation", "query")

        try:
            if operation == "register":
                metadata = AgentMetadataEntry(**input_data.get("metadata", {}))
                agent_class = input_data.get("agent_class")
                result = self.register_agent(metadata, agent_class)
                return AgentResult(
                    success=True,
                    data={"registered": result, "agent_id": metadata.agent_id},
                    metadata={"operation": "register", "duration_ms": (time.time() - start_time) * 1000}
                )

            elif operation == "unregister":
                agent_id = input_data.get("agent_id")
                version = input_data.get("version")
                result = self.unregister_agent(agent_id, version)
                return AgentResult(
                    success=result,
                    data={"unregistered": result},
                    metadata={"operation": "unregister", "duration_ms": (time.time() - start_time) * 1000}
                )

            elif operation == "query":
                query_input = RegistryQueryInput(**input_data.get("query", {}))
                result = self.query_agents(query_input)
                return AgentResult(
                    success=True,
                    data=result.model_dump(),
                    metadata={"operation": "query", "duration_ms": (time.time() - start_time) * 1000}
                )

            elif operation == "resolve_dependencies":
                resolution_input = DependencyResolutionInput(**input_data.get("resolution", {}))
                result = self.resolve_dependencies(resolution_input)
                return AgentResult(
                    success=result.success,
                    data=result.model_dump(),
                    error=result.error,
                    metadata={"operation": "resolve_dependencies", "duration_ms": (time.time() - start_time) * 1000}
                )

            elif operation == "get_agent":
                agent_id = input_data.get("agent_id")
                version = input_data.get("version")
                metadata = self.get_agent(agent_id, version)
                return AgentResult(
                    success=metadata is not None,
                    data={"metadata": metadata.model_dump() if metadata else None},
                    error=None if metadata else f"Agent not found: {agent_id}",
                    metadata={"operation": "get_agent", "duration_ms": (time.time() - start_time) * 1000}
                )

            elif operation == "health_check":
                agent_id = input_data.get("agent_id")
                status = self.check_agent_health(agent_id)
                return AgentResult(
                    success=True,
                    data={"health_status": status.value if status else "unknown"},
                    metadata={"operation": "health_check", "duration_ms": (time.time() - start_time) * 1000}
                )

            elif operation == "list_versions":
                agent_id = input_data.get("agent_id")
                versions = self.list_versions(agent_id)
                return AgentResult(
                    success=True,
                    data={"versions": versions},
                    metadata={"operation": "list_versions", "duration_ms": (time.time() - start_time) * 1000}
                )

            elif operation == "get_statistics":
                stats = self.get_statistics()
                return AgentResult(
                    success=True,
                    data=stats,
                    metadata={"operation": "get_statistics", "duration_ms": (time.time() - start_time) * 1000}
                )

            else:
                return AgentResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                    metadata={"operation": operation, "duration_ms": (time.time() - start_time) * 1000}
                )

        except Exception as e:
            self.logger.error(f"Registry operation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"operation": operation, "duration_ms": (time.time() - start_time) * 1000}
            )

    # =========================================================================
    # Agent Registration
    # =========================================================================

    def register_agent(
        self,
        metadata: AgentMetadataEntry,
        agent_class: Optional[Type[BaseAgent]] = None
    ) -> bool:
        """
        Register an agent with the registry.

        Args:
            metadata: Agent metadata entry
            agent_class: Optional agent class for instantiation

        Returns:
            True if registration succeeded

        Raises:
            ValueError: If metadata is invalid
        """
        with self._lock:
            agent_id = metadata.agent_id
            version = metadata.version

            # Initialize version dict if needed
            if agent_id not in self._registry:
                self._registry[agent_id] = {}

            # Check for duplicate version
            if version in self._registry[agent_id]:
                self.logger.warning(f"Overwriting existing registration: {agent_id}@{version}")

            # Update timestamp
            metadata.updated_at = DeterministicClock.now()

            # Store metadata
            self._registry[agent_id][version] = metadata

            # Update indexes
            self._by_layer[metadata.layer].add(agent_id)

            for sector in metadata.sectors:
                self._by_sector[sector].add(agent_id)

            for capability in metadata.capabilities:
                self._by_capability[capability.name].add(agent_id)

            for tag in metadata.tags:
                self._by_tag[tag.lower()].add(agent_id)

            # Store agent class if provided
            if agent_class is not None:
                class_key = f"{agent_id}@{version}"
                self._agent_classes[class_key] = agent_class

            self._registration_count += 1

            # Notify hot-reload callbacks
            self._notify_reload(agent_id, version)

            self.logger.info(f"Registered agent: {agent_id}@{version}")
            return True

    def unregister_agent(
        self,
        agent_id: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Unregister an agent from the registry.

        Args:
            agent_id: Agent ID to unregister
            version: Specific version to unregister (None = all versions)

        Returns:
            True if unregistration succeeded
        """
        with self._lock:
            if agent_id not in self._registry:
                self.logger.warning(f"Agent not found: {agent_id}")
                return False

            if version is not None:
                # Remove specific version
                if version in self._registry[agent_id]:
                    del self._registry[agent_id][version]
                    class_key = f"{agent_id}@{version}"
                    if class_key in self._agent_classes:
                        del self._agent_classes[class_key]
                    self.logger.info(f"Unregistered agent version: {agent_id}@{version}")
                else:
                    return False

                # Remove agent entirely if no versions left
                if not self._registry[agent_id]:
                    del self._registry[agent_id]
                    self._remove_from_indexes(agent_id)
            else:
                # Remove all versions
                for v in list(self._registry[agent_id].keys()):
                    class_key = f"{agent_id}@{v}"
                    if class_key in self._agent_classes:
                        del self._agent_classes[class_key]

                del self._registry[agent_id]
                self._remove_from_indexes(agent_id)
                self.logger.info(f"Unregistered all versions of agent: {agent_id}")

            return True

    def _remove_from_indexes(self, agent_id: str) -> None:
        """Remove agent from all indexes."""
        for layer_set in self._by_layer.values():
            layer_set.discard(agent_id)

        for sector_set in self._by_sector.values():
            sector_set.discard(agent_id)

        for capability_set in self._by_capability.values():
            capability_set.discard(agent_id)

        for tag_set in self._by_tag.values():
            tag_set.discard(agent_id)

    # =========================================================================
    # Agent Discovery
    # =========================================================================

    def query_agents(self, query: RegistryQueryInput) -> RegistryQueryOutput:
        """
        Query agents with flexible filtering.

        Args:
            query: Query parameters and filters

        Returns:
            RegistryQueryOutput with matching agents
        """
        start_time = time.time()
        self._query_count += 1

        with self._lock:
            # Start with all agents or filtered set
            candidate_ids: Optional[Set[str]] = None

            # Apply index-based filters first for efficiency
            if query.layer is not None:
                layer_ids = self._by_layer.get(query.layer, set())
                candidate_ids = layer_ids if candidate_ids is None else candidate_ids & layer_ids

            if query.sector is not None:
                sector_ids = self._by_sector.get(query.sector, set())
                candidate_ids = sector_ids if candidate_ids is None else candidate_ids & sector_ids

            if query.capability is not None:
                capability_ids = self._by_capability.get(query.capability, set())
                candidate_ids = capability_ids if candidate_ids is None else candidate_ids & capability_ids

            for tag in query.tags:
                tag_ids = self._by_tag.get(tag.lower(), set())
                candidate_ids = tag_ids if candidate_ids is None else candidate_ids & tag_ids

            # Default to all agents if no index filters applied
            if candidate_ids is None:
                candidate_ids = set(self._registry.keys())

            # Apply remaining filters
            matching_agents: List[AgentMetadataEntry] = []

            for agent_id in candidate_ids:
                versions = self._registry.get(agent_id, {})

                for version, metadata in versions.items():
                    if self._matches_query(metadata, query):
                        matching_agents.append(metadata)

            # Sort by agent_id and version (latest first)
            matching_agents.sort(
                key=lambda m: (m.agent_id, m.parsed_version),
                reverse=True
            )

            total_count = len(matching_agents)

            # Apply pagination
            paginated = matching_agents[query.offset:query.offset + query.limit]

            query_time = (time.time() - start_time) * 1000

            # Generate provenance hash
            provenance_content = f"query:{json.dumps(query.model_dump(), sort_keys=True)}:{total_count}"
            provenance_hash = hashlib.sha256(provenance_content.encode()).hexdigest()[:16]

            return RegistryQueryOutput(
                agents=paginated,
                total_count=total_count,
                query_time_ms=query_time,
                provenance_hash=provenance_hash
            )

    def _matches_query(self, metadata: AgentMetadataEntry, query: RegistryQueryInput) -> bool:
        """Check if metadata matches all query criteria."""
        # Capability category filter
        if query.capability_category is not None:
            if not any(c.category == query.capability_category for c in metadata.capabilities):
                return False

        # Variant filters
        if query.variant_type is not None:
            if not metadata.has_variant(query.variant_type, query.variant_value):
                return False

        # Health status filter
        if query.health_status is not None:
            if metadata.health_status != query.health_status:
                return False

        # Version constraints
        if query.min_version is not None:
            min_v = SemanticVersion.parse(query.min_version)
            if metadata.parsed_version < min_v:
                return False

        if query.max_version is not None:
            max_v = SemanticVersion.parse(query.max_version)
            if not metadata.parsed_version < max_v and str(metadata.parsed_version) != str(max_v):
                return False

        # Text search
        if query.search_text is not None:
            search_lower = query.search_text.lower()
            if (search_lower not in metadata.name.lower() and
                search_lower not in metadata.description.lower()):
                return False

        return True

    def get_agent(
        self,
        agent_id: str,
        version: Optional[str] = None
    ) -> Optional[AgentMetadataEntry]:
        """
        Get agent metadata by ID and optional version.

        Args:
            agent_id: Agent ID
            version: Specific version (None = latest)

        Returns:
            AgentMetadataEntry or None if not found
        """
        with self._lock:
            if agent_id not in self._registry:
                return None

            versions = self._registry[agent_id]

            if version is not None:
                return versions.get(version)

            # Return latest version
            if not versions:
                return None

            latest_version = max(versions.keys(), key=lambda v: SemanticVersion.parse(v))
            return versions[latest_version]

    def list_versions(self, agent_id: str) -> List[str]:
        """
        List all versions of an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of version strings, sorted newest first
        """
        with self._lock:
            if agent_id not in self._registry:
                return []

            versions = list(self._registry[agent_id].keys())
            versions.sort(key=lambda v: SemanticVersion.parse(v), reverse=True)
            return versions

    def get_agent_class(
        self,
        agent_id: str,
        version: Optional[str] = None
    ) -> Optional[Type[BaseAgent]]:
        """
        Get the agent class for instantiation.

        Args:
            agent_id: Agent ID
            version: Specific version (None = latest)

        Returns:
            Agent class or None if not found
        """
        with self._lock:
            if version is None:
                versions = self.list_versions(agent_id)
                if not versions:
                    return None
                version = versions[0]

            class_key = f"{agent_id}@{version}"
            return self._agent_classes.get(class_key)

    # =========================================================================
    # Capability Matching
    # =========================================================================

    def find_agents_by_capabilities(
        self,
        required_capabilities: List[AgentCapability],
        sector: Optional[SectorClassification] = None,
        layer: Optional[AgentLayer] = None,
        require_healthy: bool = False
    ) -> List[AgentMetadataEntry]:
        """
        Find agents that provide all required capabilities.

        Args:
            required_capabilities: List of required capabilities
            sector: Optional sector filter
            layer: Optional layer filter
            require_healthy: If True, only return agents with HEALTHY status

        Returns:
            List of matching agents
        """
        query = RegistryQueryInput(
            sector=sector,
            layer=layer,
            health_status=AgentHealthStatus.HEALTHY if require_healthy else None
        )

        result = self.query_agents(query)

        matching = []
        for agent in result.agents:
            if self._agent_has_capabilities(agent, required_capabilities):
                matching.append(agent)

        return matching

    def _agent_has_capabilities(
        self,
        agent: AgentMetadataEntry,
        required: List[AgentCapability]
    ) -> bool:
        """Check if agent has all required capabilities."""
        for req in required:
            has_match = False
            for cap in agent.capabilities:
                if cap.matches(req):
                    has_match = True
                    break
            if not has_match:
                return False
        return True

    # =========================================================================
    # Dependency Resolution
    # =========================================================================

    def resolve_dependencies(
        self,
        input_data: DependencyResolutionInput
    ) -> DependencyResolutionOutput:
        """
        Resolve dependencies for a set of agents.

        Performs topological sort to determine execution order.
        Detects circular dependencies and missing dependencies.

        Args:
            input_data: Resolution parameters

        Returns:
            DependencyResolutionOutput with resolved order
        """
        start_time = time.time()

        with self._lock:
            # Build dependency graph
            graph: Dict[str, List[str]] = {}
            all_deps: Set[str] = set()

            # Queue of agents to process
            to_process = list(input_data.agent_ids)
            processed: Set[str] = set()

            while to_process:
                agent_id = to_process.pop(0)

                if agent_id in processed:
                    continue

                processed.add(agent_id)

                metadata = self.get_agent(agent_id)
                if metadata is None:
                    continue

                deps = []
                for dep in metadata.dependencies:
                    if dep.optional and not input_data.include_optional:
                        continue

                    deps.append(dep.agent_id)
                    all_deps.add(dep.agent_id)

                    if dep.agent_id not in processed:
                        to_process.append(dep.agent_id)

                graph[agent_id] = deps

            # Check for missing dependencies
            missing = []
            for dep_id in all_deps:
                if self.get_agent(dep_id) is None:
                    missing.append(dep_id)

            if missing and input_data.fail_on_missing:
                return DependencyResolutionOutput(
                    dependency_graph=graph,
                    missing_dependencies=missing,
                    resolution_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=f"Missing dependencies: {missing}"
                )

            # Detect circular dependencies
            circular = self._detect_cycles(graph)
            if circular:
                return DependencyResolutionOutput(
                    dependency_graph=graph,
                    circular_dependencies=circular,
                    resolution_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=f"Circular dependencies detected: {circular}"
                )

            # Topological sort
            resolved_order = self._topological_sort(graph)

            return DependencyResolutionOutput(
                resolved_order=resolved_order,
                dependency_graph=graph,
                missing_dependencies=missing,
                resolution_time_ms=(time.time() - start_time) * 1000,
                success=True
            )

    def _detect_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect cycles in dependency graph using DFS."""
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on dependency graph."""
        # Calculate in-degrees
        in_degree: Dict[str, int] = defaultdict(int)
        for node in graph:
            in_degree.setdefault(node, 0)
            for dep in graph.get(node, []):
                in_degree[node] += 1

        # Start with nodes that have no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree of nodes that depend on this one
            for other_node, deps in graph.items():
                if node in deps:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)

        return result

    # =========================================================================
    # Health Tracking
    # =========================================================================

    def check_agent_health(self, agent_id: str) -> Optional[AgentHealthStatus]:
        """
        Check and update health status of an agent.

        Args:
            agent_id: Agent ID to check

        Returns:
            Current health status or None if agent not found
        """
        with self._lock:
            metadata = self.get_agent(agent_id)
            if metadata is None:
                return None

            # Get agent class and try to instantiate
            agent_class = self.get_agent_class(agent_id)

            if agent_class is None:
                metadata.health_status = AgentHealthStatus.UNKNOWN
            else:
                try:
                    # Basic instantiation check
                    _ = agent_class()
                    metadata.health_status = AgentHealthStatus.HEALTHY
                except Exception as e:
                    self.logger.warning(f"Health check failed for {agent_id}: {e}")
                    metadata.health_status = AgentHealthStatus.UNHEALTHY

            metadata.last_health_check = DeterministicClock.now()

            return metadata.health_status

    def set_agent_health(
        self,
        agent_id: str,
        status: AgentHealthStatus,
        version: Optional[str] = None
    ) -> bool:
        """
        Manually set health status for an agent.

        Args:
            agent_id: Agent ID
            status: New health status
            version: Specific version (None = all versions)

        Returns:
            True if status was updated
        """
        with self._lock:
            if agent_id not in self._registry:
                return False

            if version is not None:
                metadata = self._registry[agent_id].get(version)
                if metadata:
                    metadata.health_status = status
                    metadata.last_health_check = DeterministicClock.now()
                    return True
                return False

            # Update all versions
            for metadata in self._registry[agent_id].values():
                metadata.health_status = status
                metadata.last_health_check = DeterministicClock.now()

            return True

    # =========================================================================
    # Hot Reload Support
    # =========================================================================

    def register_reload_callback(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """
        Register a callback for hot-reload notifications.

        Args:
            callback: Function(agent_id, version) to call on reload
        """
        self._reload_callbacks.append(callback)

    def unregister_reload_callback(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """
        Unregister a hot-reload callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)

    def _notify_reload(self, agent_id: str, version: str) -> None:
        """Notify all reload callbacks of a registration."""
        for callback in self._reload_callbacks:
            try:
                callback(agent_id, version)
            except Exception as e:
                self.logger.error(f"Reload callback failed: {e}")

    def hot_reload_agent(
        self,
        agent_id: str,
        metadata: AgentMetadataEntry,
        agent_class: Optional[Type[BaseAgent]] = None
    ) -> bool:
        """
        Hot-reload an agent without service interruption.

        This atomically replaces the agent registration.

        Args:
            agent_id: Agent ID to reload
            metadata: New metadata
            agent_class: New agent class

        Returns:
            True if reload succeeded
        """
        self.logger.info(f"Hot-reloading agent: {agent_id}@{metadata.version}")
        return self.register_agent(metadata, agent_class)

    # =========================================================================
    # Statistics and Metrics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary of registry statistics
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

            health_counts = defaultdict(int)
            for agent_versions in self._registry.values():
                for metadata in agent_versions.values():
                    health_counts[metadata.health_status.value] += 1

            return {
                "total_agents": total_agents,
                "total_versions": total_versions,
                "registration_count": self._registration_count,
                "query_count": self._query_count,
                "agents_by_layer": layer_counts,
                "agents_by_sector": sector_counts,
                "agents_by_health": dict(health_counts),
                "registered_classes": len(self._agent_classes),
                "reload_callbacks": len(self._reload_callbacks),
            }

    def get_all_agent_ids(self) -> List[str]:
        """
        Get all registered agent IDs.

        Returns:
            List of agent IDs
        """
        with self._lock:
            return list(self._registry.keys())

    def get_agents_by_layer(self, layer: AgentLayer) -> List[str]:
        """
        Get all agent IDs in a specific layer.

        Args:
            layer: Agent layer

        Returns:
            List of agent IDs in the layer
        """
        with self._lock:
            return list(self._by_layer.get(layer, set()))

    # =========================================================================
    # Export and Import
    # =========================================================================

    def export_registry(self) -> Dict[str, Any]:
        """
        Export the registry to a serializable format.

        Returns:
            Dictionary representation of the registry
        """
        with self._lock:
            export_data = {
                "version": "1.0",
                "exported_at": DeterministicClock.now().isoformat(),
                "agents": {}
            }

            for agent_id, versions in self._registry.items():
                export_data["agents"][agent_id] = {
                    v: m.model_dump() for v, m in versions.items()
                }

            return export_data

    def import_registry(
        self,
        data: Dict[str, Any],
        merge: bool = True
    ) -> int:
        """
        Import registry data.

        Args:
            data: Exported registry data
            merge: If True, merge with existing; if False, replace

        Returns:
            Number of agents imported
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
                    except Exception as e:
                        self.logger.error(f"Failed to import {agent_id}@{version}: {e}")

            return count


# =============================================================================
# Factory Functions
# =============================================================================

def create_agent_registry(config: Optional[AgentConfig] = None) -> VersionedAgentRegistry:
    """
    Create a new VersionedAgentRegistry instance.

    Args:
        config: Optional configuration

    Returns:
        Configured VersionedAgentRegistry instance
    """
    return VersionedAgentRegistry(config)


def create_agent_metadata(
    agent_id: str,
    name: str,
    description: str,
    version: str,
    layer: AgentLayer,
    capabilities: Optional[List[Dict[str, Any]]] = None,
    sectors: Optional[List[SectorClassification]] = None,
    variants: Optional[List[Dict[str, Any]]] = None,
    dependencies: Optional[List[Dict[str, Any]]] = None,
    tags: Optional[List[str]] = None,
    **kwargs
) -> AgentMetadataEntry:
    """
    Factory function to create agent metadata.

    Args:
        agent_id: Unique agent identifier
        name: Human-readable name
        description: Agent description
        version: Semantic version string
        layer: Agent layer
        capabilities: List of capability dictionaries
        sectors: List of applicable sectors
        variants: List of variant dictionaries
        dependencies: List of dependency dictionaries
        tags: Searchable tags
        **kwargs: Additional metadata fields

    Returns:
        AgentMetadataEntry instance
    """
    return AgentMetadataEntry(
        agent_id=agent_id,
        name=name,
        description=description,
        version=version,
        layer=layer,
        capabilities=[AgentCapability(**c) for c in (capabilities or [])],
        sectors=sectors or [],
        variants=[AgentVariant(**v) for v in (variants or [])],
        dependencies=[AgentDependency(**d) for d in (dependencies or [])],
        tags=tags or [],
        **kwargs
    )

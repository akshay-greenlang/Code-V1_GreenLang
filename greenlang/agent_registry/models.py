# -*- coding: utf-8 -*-
"""
Agent Registry Models - AGENT-FOUND-007: Agent Registry & Service Catalog

Re-exports and extends the Layer 1 enums and Pydantic models from
``greenlang.agents.foundation.agent_registry`` with additional SDK-level
models for health checking, service catalog entries, and audit trails.

Key additions over Layer 1:
    - RegistryChangeType enum for audit trail categorisation
    - HealthCheckResult model for structured health probe results
    - ServiceCatalogEntry model for rich catalog entries
    - RegistryAuditEntry model for change-log auditing

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger: Any  # forward reference -- assigned after import
import logging

logger = logging.getLogger(__name__)


# ===================================================================
# Utility helpers
# ===================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# ===================================================================
# Re-exported Layer 1 enums
# ===================================================================


class AgentLayer(str, Enum):
    """11-layer agent taxonomy for GreenLang Climate OS."""

    FOUNDATION = "foundation"
    DATA = "data"
    MRV = "mrv"
    PLANNING = "planning"
    RISK = "risk"
    FINANCE = "finance"
    PROCUREMENT = "procurement"
    POLICY = "policy"
    REPORTING = "reporting"
    OPERATIONS = "operations"
    DEVTOOLS = "devtools"

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
    """10-sector classification for agent specialisation."""

    ENERGY = "energy"
    UTILITIES = "utilities"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    REAL_ESTATE = "real_estate"
    INFORMATION_TECHNOLOGY = "information_technology"

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

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class ExecutionMode(str, Enum):
    """Agent execution mode for GLIP v1 compatibility."""

    GLIP_V1 = "glip_v1"
    LEGACY_HTTP = "legacy_http"
    HYBRID = "hybrid"


class IdempotencySupport(str, Enum):
    """Level of idempotency support for an agent."""

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class CapabilityCategory(str, Enum):
    """Categories of agent capabilities."""

    CALCULATION = "calculation"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    INTEGRATION = "integration"
    ORCHESTRATION = "orchestration"
    ANALYSIS = "analysis"
    REPORTING = "reporting"


# ===================================================================
# SDK-level enum addition
# ===================================================================


class RegistryChangeType(str, Enum):
    """Types of changes that can be audited in the registry."""

    REGISTER = "register"
    UNREGISTER = "unregister"
    UPDATE = "update"
    RELOAD = "reload"
    HEALTH_UPDATE = "health_update"
    MIGRATE = "migrate"


# ===================================================================
# Pydantic models -- Layer 1 compatible
# ===================================================================


class ResourceProfile(BaseModel):
    """Resource requirements for GLIP v1 K8s Job execution."""

    cpu_request: str = Field(default="100m", description="CPU request (e.g., '100m', '1')")
    cpu_limit: str = Field(default="1", description="CPU limit (e.g., '1', '4')")
    memory_request: str = Field(default="256Mi", description="Memory request (e.g., '256Mi', '1Gi')")
    memory_limit: str = Field(default="1Gi", description="Memory limit (e.g., '1Gi', '4Gi')")
    gpu_count: int = Field(default=0, ge=0, description="Number of GPUs required")
    gpu_type: Optional[str] = Field(None, description="GPU type (e.g., 'nvidia-tesla-t4')")
    ephemeral_storage_request: str = Field(default="1Gi", description="Ephemeral storage request")
    ephemeral_storage_limit: str = Field(default="10Gi", description="Ephemeral storage limit")
    timeout_seconds: int = Field(default=3600, ge=60, le=86400, description="Max execution time (60s-24h)")


class ContainerSpec(BaseModel):
    """Container specification for GLIP v1 agent execution."""

    image: str = Field(..., description="Docker image (e.g., 'greenlang/gl-mrv-x-001:1.0.0')")
    image_pull_policy: str = Field(default="IfNotPresent", description="K8s image pull policy")
    image_pull_secrets: List[str] = Field(default_factory=list, description="Image pull secret names")
    entrypoint: Optional[List[str]] = Field(None, description="Container entrypoint override")
    command: Optional[List[str]] = Field(None, description="Container command override")
    working_dir: Optional[str] = Field(None, description="Working directory in container")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Additional environment variables")
    volume_mounts: List[Dict[str, str]] = Field(default_factory=list, description="Volume mount specifications")

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Validate Docker image format."""
        if not v or "/" not in v:
            raise ValueError(f"Invalid Docker image format: {v}. Expected format: registry/name:tag")
        return v


class LegacyHttpConfig(BaseModel):
    """Legacy HTTP endpoint configuration for backward compatibility."""

    endpoint: str = Field(..., description="HTTP endpoint URL")
    method: str = Field(default="POST", description="HTTP method")
    auth_type: Optional[str] = Field(None, description="Auth type: 'bearer', 'api_key', 'basic'")
    auth_secret_name: Optional[str] = Field(None, description="K8s secret name for auth credentials")
    timeout_seconds: int = Field(default=300, ge=10, le=3600, description="HTTP timeout")
    retry_count: int = Field(default=3, ge=0, le=10, description="Retry count on failure")
    health_check_path: Optional[str] = Field(None, description="Health check endpoint path")


class SemanticVersion(BaseModel):
    """Semantic version representation with comparison support."""

    major: int = Field(ge=0, description="Major version (breaking changes)")
    minor: int = Field(ge=0, description="Minor version (new features)")
    patch: int = Field(ge=0, description="Patch version (bug fixes)")
    prerelease: Optional[str] = Field(None, description="Prerelease identifier")
    build: Optional[str] = Field(None, description="Build metadata")

    @classmethod
    def parse(cls, version_str: str) -> SemanticVersion:
        """Parse a version string into SemanticVersion.

        Args:
            version_str: SemVer string like '1.2.3', '1.0.0-beta', '1.0.0+build'.

        Returns:
            Parsed SemanticVersion.

        Raises:
            ValueError: If the string is not valid semver.
        """
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$"
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid semantic version: {version_str}")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    def __str__(self) -> str:
        """Convert to version string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: SemanticVersion) -> bool:
        """Compare versions for sorting."""
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        return False

    def __le__(self, other: SemanticVersion) -> bool:
        """Less-than-or-equal comparison."""
        return self == other or self < other

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch, self.prerelease) == (
            other.major, other.minor, other.patch, other.prerelease,
        )

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """Check if this version is compatible with another (same major)."""
        return self.major == other.major


class AgentCapability(BaseModel):
    """Definition of a specific agent capability."""

    name: str = Field(..., description="Capability name (unique identifier)")
    category: CapabilityCategory = Field(..., description="Capability category")
    description: str = Field(..., description="Human-readable description")
    input_types: List[str] = Field(default_factory=list, description="Supported input data types")
    output_types: List[str] = Field(default_factory=list, description="Output data types")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")

    def matches(self, required: AgentCapability) -> bool:
        """Check if this capability matches a required capability.

        Args:
            required: The required capability to match against.

        Returns:
            True if name and category match, and all required input types
            are present.
        """
        if self.name != required.name:
            return False
        if self.category != required.category:
            return False
        for input_type in required.input_types:
            if input_type not in self.input_types:
                return False
        return True


class AgentVariant(BaseModel):
    """Agent variant for geographic or fuel-type specialisation."""

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
        """Check if a version satisfies this dependency constraint.

        Args:
            version: The version to test.

        Returns:
            True if version satisfies the constraint.
        """
        if self.version_constraint == "*":
            return True

        constraint = self.version_constraint
        if constraint.startswith(">="):
            min_version = SemanticVersion.parse(constraint[2:])
            return not version < min_version
        elif constraint.startswith("^"):
            target = SemanticVersion.parse(constraint[1:])
            return version.is_compatible_with(target) and not version < target
        elif constraint.startswith("~"):
            target = SemanticVersion.parse(constraint[1:])
            return (
                version.major == target.major
                and version.minor == target.minor
                and version.patch >= target.patch
            )
        elif constraint.startswith("="):
            target = SemanticVersion.parse(constraint[1:])
            return str(version) == str(target)
        else:
            target = SemanticVersion.parse(constraint)
            return str(version) == str(target)


class AgentMetadataEntry(BaseModel):
    """Complete metadata entry for a registered agent.

    Contains all information needed to discover, instantiate, and
    integrate an agent into a pipeline.
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
    health_status: AgentHealthStatus = Field(
        default=AgentHealthStatus.UNKNOWN, description="Current health status",
    )
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")

    # Metadata
    author: Optional[str] = Field(None, description="Agent author")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")

    # Audit
    registered_at: datetime = Field(default_factory=_utcnow, description="Registration timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Last update timestamp")

    # GLIP v1 extensions
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.LEGACY_HTTP,
        description="Execution mode: GLIP v1 (K8s Jobs) or Legacy HTTP",
    )
    idempotency_support: IdempotencySupport = Field(
        default=IdempotencySupport.NONE,
        description="Level of idempotency support for retry safety",
    )
    resource_profile: Optional[ResourceProfile] = Field(
        None, description="Resource requirements for GLIP v1 K8s Job execution",
    )
    container_spec: Optional[ContainerSpec] = Field(
        None, description="Container specification for GLIP v1 execution",
    )
    legacy_http_config: Optional[LegacyHttpConfig] = Field(
        None, description="Legacy HTTP endpoint configuration",
    )
    glip_version: Optional[str] = Field(
        None, description="GLIP protocol version supported",
    )
    supports_checkpointing: bool = Field(
        default=False, description="Whether agent supports checkpoint/resume",
    )
    deterministic: bool = Field(
        default=True, description="Whether agent produces deterministic outputs",
    )
    max_concurrent_runs: Optional[int] = Field(
        None, ge=1, description="Maximum concurrent runs allowed",
    )

    @property
    def parsed_version(self) -> SemanticVersion:
        """Get parsed semantic version."""
        return SemanticVersion.parse(self.version)

    @property
    def provenance_hash(self) -> str:
        """Generate provenance hash for this metadata entry."""
        content = json.dumps(
            {
                "agent_id": self.agent_id,
                "version": self.version,
                "name": self.name,
                "capabilities_count": len(self.capabilities),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability.

        Args:
            capability_name: The capability name to look for.

        Returns:
            True if the agent has the named capability.
        """
        return any(c.name == capability_name for c in self.capabilities)

    def has_variant(self, variant_type: str, variant_value: Optional[str] = None) -> bool:
        """Check if agent has a specific variant.

        Args:
            variant_type: The variant type to look for.
            variant_value: Optional specific value to match.

        Returns:
            True if the agent has the matching variant.
        """
        for v in self.variants:
            if v.variant_type == variant_type:
                if variant_value is None or v.variant_value == variant_value:
                    return True
        return False

    def supports_sector(self, sector: SectorClassification) -> bool:
        """Check if agent supports a specific sector.

        Args:
            sector: The sector to test.

        Returns:
            True if supported (empty sectors list means all sectors).
        """
        return len(self.sectors) == 0 or sector in self.sectors

    @property
    def is_glip_compatible(self) -> bool:
        """Check if agent supports GLIP v1 execution."""
        return self.execution_mode in (ExecutionMode.GLIP_V1, ExecutionMode.HYBRID)


# ===================================================================
# Query / resolution models
# ===================================================================


class RegistryQueryInput(BaseModel):
    """Input for registry query operations."""

    layer: Optional[AgentLayer] = Field(None, description="Filter by layer")
    sector: Optional[SectorClassification] = Field(None, description="Filter by sector")
    capability: Optional[str] = Field(None, description="Filter by capability name")
    capability_category: Optional[CapabilityCategory] = Field(None, description="Filter by capability category")
    tags: List[str] = Field(default_factory=list, description="Filter by tags (all must match)")
    health_status: Optional[AgentHealthStatus] = Field(None, description="Filter by health status")
    search_text: Optional[str] = Field(None, description="Text search in name and description")
    execution_mode: Optional[ExecutionMode] = Field(None, description="Filter by execution mode")
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


# ===================================================================
# SDK-level additions
# ===================================================================


class HealthCheckResult(BaseModel):
    """Structured result from a health check probe.

    Attributes:
        agent_id: The checked agent ID.
        version: The checked version (or 'latest').
        status: The resulting health status.
        response_time_ms: Probe response time in milliseconds.
        checked_at: When the probe was executed.
        error: Error message if the probe failed.
        details: Additional probe details.
    """

    agent_id: str = Field(..., description="Checked agent ID")
    version: str = Field(default="latest", description="Checked version")
    status: AgentHealthStatus = Field(..., description="Resulting health status")
    response_time_ms: float = Field(default=0.0, description="Probe response time in ms")
    checked_at: datetime = Field(default_factory=_utcnow, description="Probe timestamp")
    error: Optional[str] = Field(None, description="Error message if probe failed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional probe details")


class ServiceCatalogEntry(BaseModel):
    """Rich service catalog entry combining metadata with runtime state.

    Extends AgentMetadataEntry with health summary and dependency info
    for use in service catalog UIs and documentation generators.

    Attributes:
        agent_id: Agent identifier.
        name: Agent name.
        description: Agent description.
        version: Current version.
        layer: Agent layer.
        sectors: Applicable sectors.
        capability_names: List of capability names.
        tags: Tags.
        health_status: Current health.
        dependency_ids: List of dependency agent IDs.
        registered_at: Registration timestamp.
        documentation_url: Link to documentation.
    """

    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    version: str = Field(..., description="Current version")
    layer: AgentLayer = Field(..., description="Agent layer")
    sectors: List[SectorClassification] = Field(default_factory=list, description="Applicable sectors")
    capability_names: List[str] = Field(default_factory=list, description="Capability names")
    tags: List[str] = Field(default_factory=list, description="Tags")
    health_status: AgentHealthStatus = Field(
        default=AgentHealthStatus.UNKNOWN, description="Current health",
    )
    dependency_ids: List[str] = Field(default_factory=list, description="Dependency agent IDs")
    registered_at: datetime = Field(default_factory=_utcnow, description="Registration timestamp")
    documentation_url: Optional[str] = Field(None, description="Link to docs")

    @classmethod
    def from_metadata(cls, metadata: AgentMetadataEntry) -> ServiceCatalogEntry:
        """Build a catalog entry from an AgentMetadataEntry.

        Args:
            metadata: The agent metadata to convert.

        Returns:
            Populated ServiceCatalogEntry.
        """
        return cls(
            agent_id=metadata.agent_id,
            name=metadata.name,
            description=metadata.description,
            version=metadata.version,
            layer=metadata.layer,
            sectors=metadata.sectors,
            capability_names=[c.name for c in metadata.capabilities],
            tags=metadata.tags,
            health_status=metadata.health_status,
            dependency_ids=[d.agent_id for d in metadata.dependencies],
            registered_at=metadata.registered_at,
            documentation_url=metadata.documentation_url,
        )


class RegistryAuditEntry(BaseModel):
    """Audit entry for a registry change.

    Attributes:
        entry_id: Unique audit entry ID.
        change_type: Type of change.
        agent_id: Affected agent ID.
        version: Affected version.
        user_id: User who made the change.
        timestamp: When the change occurred.
        data_hash: SHA-256 hash of the data at this point.
        details: Additional context about the change.
    """

    entry_id: str = Field(default_factory=_new_uuid, description="Audit entry ID")
    change_type: RegistryChangeType = Field(..., description="Type of change")
    agent_id: str = Field(..., description="Affected agent ID")
    version: str = Field(default="", description="Affected version")
    user_id: str = Field(default="system", description="User who made the change")
    timestamp: datetime = Field(default_factory=_utcnow, description="Change timestamp")
    data_hash: str = Field(default="", description="SHA-256 hash of data at this point")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


__all__ = [
    # Enums
    "AgentLayer",
    "SectorClassification",
    "AgentHealthStatus",
    "ExecutionMode",
    "IdempotencySupport",
    "CapabilityCategory",
    "RegistryChangeType",
    # Layer 1 models
    "ResourceProfile",
    "ContainerSpec",
    "LegacyHttpConfig",
    "SemanticVersion",
    "AgentCapability",
    "AgentVariant",
    "AgentDependency",
    "AgentMetadataEntry",
    "RegistryQueryInput",
    "RegistryQueryOutput",
    "DependencyResolutionInput",
    "DependencyResolutionOutput",
    # SDK additions
    "HealthCheckResult",
    "ServiceCatalogEntry",
    "RegistryAuditEntry",
]

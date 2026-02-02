"""
Pydantic Models for GreenLang Agent Registry API

This module defines the request/response models for the Agent Registry REST API.
All models use Pydantic v2 for validation and serialization.

Models:
- AgentMetadata: Core agent information
- AgentVersion: Version-specific data
- PublishRequest: Request to publish a new agent version
- PublishResponse: Response from publish operation
- PromoteRequest: Request to promote agent state
- ListAgentsResponse: Paginated list of agents
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


class LifecycleState(str, Enum):
    """Valid lifecycle states for agent versions."""

    DRAFT = "draft"
    EXPERIMENTAL = "experimental"
    CERTIFIED = "certified"
    DEPRECATED = "deprecated"


# =============================================================================
# Core Data Models
# =============================================================================


class SemanticVersion(BaseModel):
    """Parsed semantic version components."""

    model_config = ConfigDict(extra="forbid")

    major: int = Field(..., ge=0, description="Major version (breaking changes)")
    minor: int = Field(..., ge=0, description="Minor version (new features)")
    patch: int = Field(..., ge=0, description="Patch version (bug fixes)")
    prerelease: Optional[str] = Field(None, description="Pre-release identifier (e.g., 'alpha', 'beta')")
    build: Optional[str] = Field(None, description="Build metadata")

    def to_string(self) -> str:
        """Convert to version string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version


class AgentCapability(BaseModel):
    """Describes a single capability of an agent."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., min_length=1, max_length=255, description="Capability name")
    description: Optional[str] = Field(None, description="Capability description")
    input_schema: Optional[str] = Field(None, description="JSON Schema reference for input")
    output_schema: Optional[str] = Field(None, description="JSON Schema reference for output")


class RuntimeRequirements(BaseModel):
    """Runtime resource requirements for an agent."""

    model_config = ConfigDict(extra="allow")

    cpu_request: Optional[str] = Field(None, description="CPU request (e.g., '500m')")
    cpu_limit: Optional[str] = Field(None, description="CPU limit (e.g., '2000m')")
    memory_request: Optional[str] = Field(None, description="Memory request (e.g., '512Mi')")
    memory_limit: Optional[str] = Field(None, description="Memory limit (e.g., '2Gi')")
    gpu_required: bool = Field(False, description="Whether GPU is required")
    python_version: Optional[str] = Field(None, description="Required Python version")
    dependencies: Optional[List[str]] = Field(None, description="Required Python packages")
    services: Optional[List[Dict[str, Any]]] = Field(None, description="Required services")
    llm_providers: Optional[List[Dict[str, Any]]] = Field(None, description="Required LLM providers")


class AgentMetadata(BaseModel):
    """Core agent metadata independent of version."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "agent_id": "gl-cbam-calculator-v2",
                "name": "CBAM Carbon Calculator",
                "description": "Calculates embedded carbon for CBAM shipments",
                "domain": "sustainability.cbam",
                "type": "calculator",
                "category": "regulatory_compliance",
                "tags": ["cbam", "carbon", "eu-regulation"],
                "team": "greenlang/cbam-team",
                "tenant_id": "customer-abc-123",
            }
        },
    )

    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        pattern=r"^[a-z][a-z0-9-]*$",
        description="Unique agent identifier (lowercase, alphanumeric, hyphens)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable agent name",
    )
    description: Optional[str] = Field(None, description="Detailed agent description")
    domain: Optional[str] = Field(
        None,
        max_length=100,
        description="Classification domain (e.g., 'sustainability.cbam')",
    )
    type: Optional[str] = Field(
        None,
        max_length=50,
        description="Agent type (e.g., 'calculator', 'validator')",
    )
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Agent category (e.g., 'regulatory_compliance')",
    )
    tags: Optional[List[str]] = Field(None, description="Classification tags")
    created_by: Optional[str] = Field(None, max_length=255, description="Creator identifier")
    team: Optional[str] = Field(None, max_length=255, description="Owning team")
    tenant_id: Optional[str] = Field(None, max_length=255, description="Tenant identifier")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class AgentVersion(BaseModel):
    """Version-specific agent data."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "version_id": "gl-cbam-calculator-v2:2.3.1",
                "agent_id": "gl-cbam-calculator-v2",
                "version": "2.3.1",
                "lifecycle_state": "certified",
                "container_image": "gcr.io/greenlang/cbam-calculator:2.3.1",
            }
        },
    )

    version_id: str = Field(..., description="Unique version identifier (agent_id:version)")
    agent_id: str = Field(..., description="Parent agent identifier")
    version: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$",
        description="Semantic version string",
    )
    semantic_version: Optional[SemanticVersion] = Field(None, description="Parsed version")
    lifecycle_state: LifecycleState = Field(
        LifecycleState.DRAFT,
        description="Current lifecycle state",
    )
    container_image: Optional[str] = Field(None, description="Docker image reference")
    image_digest: Optional[str] = Field(None, description="SHA256 image digest")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    runtime_requirements: Optional[RuntimeRequirements] = Field(
        None, description="Runtime requirements"
    )
    capabilities: Optional[List[AgentCapability]] = Field(
        None, description="Agent capabilities"
    )
    created_at: Optional[datetime] = Field(None, description="Version creation timestamp")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    deprecated_at: Optional[datetime] = Field(None, description="Deprecation timestamp")


# =============================================================================
# Request Models
# =============================================================================


class PublishRequest(BaseModel):
    """Request to publish a new agent or agent version."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "agent_id": "gl-cbam-calculator-v2",
                "name": "CBAM Carbon Calculator",
                "description": "Calculates embedded carbon for CBAM shipments",
                "version": "2.3.1",
                "domain": "sustainability.cbam",
                "type": "calculator",
                "container_image": "gcr.io/greenlang/cbam-calculator:2.3.1",
                "team": "greenlang/cbam-team",
                "tenant_id": "customer-abc-123",
            }
        },
    )

    # Agent metadata
    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        pattern=r"^[a-z][a-z0-9-]*$",
        description="Unique agent identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable agent name",
    )
    description: Optional[str] = Field(None, description="Agent description")
    domain: Optional[str] = Field(None, max_length=100, description="Classification domain")
    type: Optional[str] = Field(None, max_length=50, description="Agent type")
    category: Optional[str] = Field(None, max_length=100, description="Agent category")
    tags: Optional[List[str]] = Field(None, description="Classification tags")
    team: Optional[str] = Field(None, max_length=255, description="Owning team")
    tenant_id: Optional[str] = Field(None, max_length=255, description="Tenant identifier")

    # Version metadata
    version: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$",
        description="Semantic version string",
    )
    container_image: Optional[str] = Field(None, max_length=500, description="Docker image")
    image_digest: Optional[str] = Field(None, max_length=100, description="Image digest")
    runtime_requirements: Optional[RuntimeRequirements] = Field(
        None, description="Runtime requirements"
    )
    capabilities: Optional[List[AgentCapability]] = Field(
        None, description="Agent capabilities"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate and normalize version string."""
        # Basic semver validation already done by pattern
        return v.strip()

    def parse_semantic_version(self) -> SemanticVersion:
        """Parse version string into SemanticVersion object."""
        import re

        # Parse semver pattern
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, self.version)
        if not match:
            raise ValueError(f"Invalid semantic version: {self.version}")

        return SemanticVersion(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )


class PromoteRequest(BaseModel):
    """Request to promote an agent version to the next lifecycle state."""

    model_config = ConfigDict(extra="forbid")

    target_state: LifecycleState = Field(
        ...,
        description="Target lifecycle state",
    )
    reason: Optional[str] = Field(
        None,
        max_length=1000,
        description="Reason for promotion",
    )
    promoted_by: Optional[str] = Field(
        None,
        max_length=255,
        description="User performing the promotion",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional promotion metadata",
    )


class ListAgentsQuery(BaseModel):
    """Query parameters for listing agents."""

    model_config = ConfigDict(extra="forbid")

    domain: Optional[str] = Field(None, description="Filter by domain")
    type: Optional[str] = Field(None, description="Filter by type")
    tenant_id: Optional[str] = Field(None, description="Filter by tenant")
    lifecycle_state: Optional[LifecycleState] = Field(None, description="Filter by state")
    search: Optional[str] = Field(None, description="Search in name/description")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")


# =============================================================================
# Response Models
# =============================================================================


class PublishResponse(BaseModel):
    """Response from agent publish operation."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(..., description="Whether publish succeeded")
    agent_id: str = Field(..., description="Published agent identifier")
    version_id: str = Field(..., description="Published version identifier")
    version: str = Field(..., description="Published version string")
    lifecycle_state: LifecycleState = Field(..., description="Initial lifecycle state")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Creation timestamp")


class PromoteResponse(BaseModel):
    """Response from agent promotion operation."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(..., description="Whether promotion succeeded")
    version_id: str = Field(..., description="Promoted version identifier")
    from_state: LifecycleState = Field(..., description="Previous state")
    to_state: LifecycleState = Field(..., description="New state")
    message: str = Field(..., description="Status message")
    transitioned_at: datetime = Field(..., description="Transition timestamp")


class AgentDetail(BaseModel):
    """Detailed agent information including versions."""

    model_config = ConfigDict(extra="forbid")

    agent: AgentMetadata = Field(..., description="Agent metadata")
    versions: List[AgentVersion] = Field(..., description="All versions")
    latest_version: Optional[AgentVersion] = Field(None, description="Latest version")
    version_count: int = Field(..., description="Total version count")


class ListAgentsResponse(BaseModel):
    """Paginated list of agents response."""

    model_config = ConfigDict(extra="forbid")

    agents: List[AgentMetadata] = Field(..., description="List of agents")
    total: int = Field(..., description="Total number of agents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class ErrorResponse(BaseModel):
    """Standard error response."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    request_id: Optional[str] = Field(None, description="Request tracking ID")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Overall health status")
    database: str = Field(..., description="Database connection status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Check timestamp")
    pool: Optional[Dict[str, Any]] = Field(None, description="Connection pool metrics")

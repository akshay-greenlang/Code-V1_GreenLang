"""
Agent Registry Pydantic Models

This module defines Pydantic models for the Agent Registry API including:
- AgentRecord: Core agent metadata
- AgentVersion: Version management with changelog
- Request/Response models for API endpoints
- Validation and serialization logic

All models follow GreenLang's zero-hallucination principle with strict
validation and provenance tracking via SHA-256 checksums.

Example:
    >>> from backend.registry.models import AgentRecord, AgentStatus
    >>> agent = AgentRecord(
    ...     name="carbon-emissions-agent",
    ...     version="1.0.0",
    ...     category="emissions",
    ...     author="greenlang-team"
    ... )
"""

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator


class AgentStatus(str, Enum):
    """
    Agent lifecycle status.

    States represent the publishing workflow:
    - draft: Initial state, under development
    - published: Publicly available and certified
    - deprecated: Marked for removal, still functional
    """

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


class CertificationStatus(BaseModel):
    """
    Agent certification status tracking.

    Tracks regulatory compliance certifications with timestamps
    and certifier information for audit trails.

    Attributes:
        framework: Regulatory framework name (CBAM, CSRD, etc.)
        certified: Whether agent is certified for this framework
        certified_at: Timestamp when certification was granted
        certified_by: User or system that granted certification
        expiry_date: Optional certification expiry date
        notes: Additional certification notes
    """

    framework: str = Field(..., description="Regulatory framework name")
    certified: bool = Field(False, description="Whether certified")
    certified_at: Optional[datetime] = Field(None, description="Certification timestamp")
    certified_by: Optional[str] = Field(None, description="Certifier identifier")
    expiry_date: Optional[datetime] = Field(None, description="Certification expiry")
    notes: Optional[str] = Field(None, description="Certification notes")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }


class AgentRecord(BaseModel):
    """
    Core agent record model.

    Represents a registered agent in the registry with all metadata,
    configuration, and audit information. Uses SHA-256 checksums for
    provenance tracking.

    Attributes:
        id: Unique agent identifier (UUID)
        name: Human-readable agent name
        version: Semantic version string (X.Y.Z)
        description: Detailed agent description
        category: Agent category (emissions, cbam, csrd, etc.)
        pack_yaml: Full pack.yaml configuration as JSON
        generated_code: Generated agent code artifacts as JSON
        checksum: SHA-256 hash of agent contents for integrity
        status: Lifecycle status (draft, published, deprecated)
        author: Agent author or owner
        created_at: Creation timestamp
        updated_at: Last update timestamp
        downloads: Total download count
        certification_status: Per-framework certification status

    Example:
        >>> agent = AgentRecord(
        ...     name="ghg-calculator",
        ...     version="1.0.0",
        ...     category="emissions",
        ...     author="greenlang"
        ... )
        >>> print(agent.checksum)
        'sha256:abc123...'
    """

    id: UUID = Field(default_factory=uuid4, description="Unique agent identifier")
    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Agent name (alphanumeric, hyphens, underscores)",
    )
    version: str = Field(
        ...,
        description="Semantic version (X.Y.Z)",
    )
    description: str = Field(
        "",
        max_length=2000,
        description="Detailed agent description",
    )
    category: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Agent category",
    )
    pack_yaml: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full pack.yaml configuration",
    )
    generated_code: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generated code artifacts",
    )
    checksum: str = Field(
        "",
        description="SHA-256 checksum for integrity verification",
    )
    status: AgentStatus = Field(
        AgentStatus.DRAFT,
        description="Lifecycle status",
    )
    author: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Agent author or owner",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp",
    )
    downloads: int = Field(
        0,
        ge=0,
        description="Total download count",
    )
    certification_status: List[CertificationStatus] = Field(
        default_factory=list,
        description="Per-framework certification status",
    )

    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    regulatory_frameworks: List[str] = Field(
        default_factory=list,
        description="Applicable regulatory frameworks",
    )
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    repository_url: Optional[str] = Field(None, description="Source repository URL")
    license: str = Field("Apache-2.0", description="License identifier")

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """
        Validate agent name format.

        Names must be alphanumeric with hyphens and underscores only.
        """
        pattern = r"^[a-zA-Z][a-zA-Z0-9_-]*$"
        if not re.match(pattern, v):
            raise ValueError(
                "Name must start with a letter and contain only "
                "alphanumeric characters, hyphens, and underscores"
            )
        return v.lower()

    @validator("version")
    def validate_version(cls, v: str) -> str:
        """
        Validate semantic version format.

        Supports X.Y.Z with optional pre-release and build metadata.
        """
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"
        if not re.match(pattern, v):
            raise ValueError(
                "Version must follow semantic versioning (X.Y.Z) "
                "with optional pre-release and build metadata"
            )
        return v

    @validator("category")
    def validate_category(cls, v: str) -> str:
        """Normalize category to lowercase."""
        return v.lower().strip()

    @root_validator(pre=False)
    def compute_checksum(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute SHA-256 checksum if not provided.

        Checksum is computed from name, version, pack_yaml, and generated_code
        to ensure integrity verification.
        """
        if not values.get("checksum"):
            content = (
                f"{values.get('name', '')}"
                f"{values.get('version', '')}"
                f"{str(values.get('pack_yaml', {}))}"
                f"{str(values.get('generated_code', {}))}"
            )
            hash_value = hashlib.sha256(content.encode("utf-8")).hexdigest()
            values["checksum"] = f"sha256:{hash_value}"
        return values

    def increment_downloads(self) -> "AgentRecord":
        """Increment download count and return updated record."""
        return self.copy(update={"downloads": self.downloads + 1})

    def deprecate(self) -> "AgentRecord":
        """Mark agent as deprecated."""
        return self.copy(
            update={
                "status": AgentStatus.DEPRECATED,
                "updated_at": datetime.utcnow(),
            }
        )

    def publish(self) -> "AgentRecord":
        """Publish agent (transition from draft to published)."""
        if self.status == AgentStatus.DEPRECATED:
            raise ValueError("Cannot publish a deprecated agent")
        return self.copy(
            update={
                "status": AgentStatus.PUBLISHED,
                "updated_at": datetime.utcnow(),
            }
        )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class AgentVersion(BaseModel):
    """
    Agent version record.

    Tracks individual versions of an agent with changelog,
    breaking change indicators, and release notes.

    Attributes:
        id: Version record identifier
        agent_id: Reference to parent agent
        version: Semantic version string
        changelog: Version changelog (markdown supported)
        breaking_changes: Whether this version has breaking changes
        release_notes: Detailed release notes
        artifact_path: Path or URL to version artifacts
        checksum: SHA-256 checksum of version artifacts
        created_at: Version creation timestamp
        published_at: When version was published (if applicable)
        deprecated_at: When version was deprecated (if applicable)

    Example:
        >>> version = AgentVersion(
        ...     agent_id=agent.id,
        ...     version="2.0.0",
        ...     changelog="Major update with new APIs",
        ...     breaking_changes=True
        ... )
    """

    id: UUID = Field(default_factory=uuid4, description="Version record ID")
    agent_id: UUID = Field(..., description="Parent agent ID")
    version: str = Field(..., description="Semantic version")
    changelog: str = Field("", description="Version changelog")
    breaking_changes: bool = Field(False, description="Has breaking changes")
    release_notes: str = Field("", description="Detailed release notes")
    artifact_path: Optional[str] = Field(None, description="Artifact storage path")
    checksum: str = Field("", description="Artifact checksum")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp",
    )
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    deprecated_at: Optional[datetime] = Field(None, description="Deprecation timestamp")
    is_latest: bool = Field(False, description="Is this the latest version")
    downloads: int = Field(0, ge=0, description="Version-specific download count")

    # Metadata
    min_runtime_version: Optional[str] = Field(
        None,
        description="Minimum GreenLang runtime version",
    )
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Agent dependencies with version constraints",
    )

    @validator("version")
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"
        if not re.match(pattern, v):
            raise ValueError("Version must follow semantic versioning (X.Y.Z)")
        return v

    def compare_version(self, other: "AgentVersion") -> int:
        """
        Compare versions for sorting.

        Returns:
            -1 if self < other, 0 if equal, 1 if self > other
        """
        self_parts = [int(x) for x in self.version.split("-")[0].split(".")]
        other_parts = [int(x) for x in other.version.split("-")[0].split(".")]

        for s, o in zip(self_parts, other_parts):
            if s < o:
                return -1
            if s > o:
                return 1
        return 0

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v),
        }


# Request/Response Models for API


class AgentCreateRequest(BaseModel):
    """
    Request model for creating a new agent.

    Example:
        POST /agents
        {
            "name": "carbon-calculator",
            "version": "1.0.0",
            "description": "Carbon emissions calculator agent",
            "category": "emissions",
            "author": "greenlang-team"
        }
    """

    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Agent name",
    )
    version: str = Field(
        "1.0.0",
        description="Initial version",
    )
    description: str = Field(
        "",
        max_length=2000,
        description="Agent description",
    )
    category: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Agent category",
    )
    author: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Agent author",
    )
    pack_yaml: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pack configuration",
    )
    generated_code: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generated code",
    )
    tags: List[str] = Field(default_factory=list, description="Tags")
    regulatory_frameworks: List[str] = Field(
        default_factory=list,
        description="Applicable frameworks",
    )
    documentation_url: Optional[str] = Field(None, description="Docs URL")
    repository_url: Optional[str] = Field(None, description="Repo URL")
    license: str = Field("Apache-2.0", description="License")

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """Validate name format."""
        pattern = r"^[a-zA-Z][a-zA-Z0-9_-]*$"
        if not re.match(pattern, v):
            raise ValueError("Invalid name format")
        return v.lower()


class AgentUpdateRequest(BaseModel):
    """
    Request model for updating an agent.

    Only provided fields are updated (partial update).
    """

    description: Optional[str] = Field(None, max_length=2000)
    pack_yaml: Optional[Dict[str, Any]] = None
    generated_code: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    regulatory_frameworks: Optional[List[str]] = None
    documentation_url: Optional[str] = None
    repository_url: Optional[str] = None


class AgentResponse(BaseModel):
    """
    Response model for agent endpoints.

    Includes all agent data plus computed fields.
    """

    id: UUID
    name: str
    version: str
    description: str
    category: str
    status: AgentStatus
    author: str
    checksum: str
    created_at: datetime
    updated_at: datetime
    downloads: int
    tags: List[str]
    regulatory_frameworks: List[str]
    certification_status: List[CertificationStatus]
    documentation_url: Optional[str]
    repository_url: Optional[str]
    license: str

    # Computed fields
    version_count: int = Field(0, description="Total version count")
    latest_version: Optional[str] = Field(None, description="Latest version string")

    class Config:
        """Pydantic configuration."""

        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class AgentListResponse(BaseModel):
    """
    Paginated response for listing agents.
    """

    data: List[AgentResponse]
    meta: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total": 0,
            "limit": 20,
            "offset": 0,
            "has_more": False,
        }
    )


class AgentSearchRequest(BaseModel):
    """
    Request model for searching agents.
    """

    query: str = Field(..., min_length=1, description="Search query")
    category: Optional[str] = Field(None, description="Filter by category")
    status: Optional[AgentStatus] = Field(None, description="Filter by status")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    regulatory_frameworks: Optional[List[str]] = Field(
        None,
        description="Filter by frameworks",
    )
    author: Optional[str] = Field(None, description="Filter by author")
    limit: int = Field(20, ge=1, le=100, description="Max results")
    offset: int = Field(0, ge=0, description="Pagination offset")
    sort_by: str = Field("downloads", description="Sort field")
    sort_order: str = Field("desc", description="Sort order")


class VersionCreateRequest(BaseModel):
    """
    Request model for creating a new version.
    """

    version: str = Field(..., description="Semantic version")
    changelog: str = Field("", description="Version changelog")
    breaking_changes: bool = Field(False, description="Has breaking changes")
    release_notes: str = Field("", description="Detailed release notes")
    pack_yaml: Optional[Dict[str, Any]] = Field(None, description="Updated pack config")
    generated_code: Optional[Dict[str, Any]] = Field(None, description="Updated code")

    @validator("version")
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"
        if not re.match(pattern, v):
            raise ValueError("Version must follow semantic versioning (X.Y.Z)")
        return v


class VersionResponse(BaseModel):
    """
    Response model for version endpoints.
    """

    id: UUID
    agent_id: UUID
    version: str
    changelog: str
    breaking_changes: bool
    release_notes: str
    artifact_path: Optional[str]
    checksum: str
    created_at: datetime
    published_at: Optional[datetime]
    deprecated_at: Optional[datetime]
    is_latest: bool
    downloads: int

    class Config:
        """Pydantic configuration."""

        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v),
        }


class PublishRequest(BaseModel):
    """
    Request model for publishing an agent version.
    """

    version: str = Field(..., description="Version to publish")
    release_notes: Optional[str] = Field(None, description="Release notes")
    certifications: Optional[List[str]] = Field(
        None,
        description="Frameworks to certify",
    )


class PublishResponse(BaseModel):
    """
    Response model for publish operation.
    """

    success: bool
    agent_id: UUID
    version: str
    published_at: datetime
    artifact_url: Optional[str]
    checksum: str

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

"""
Agent Registry SQLAlchemy ORM Models

This module defines SQLAlchemy ORM models for the Agent Registry with:
- AgentRecordDB: Main agent table with full metadata
- AgentVersionDB: Version management table
- Proper relationships, indexes, and constraints

All models support PostgreSQL with Row-Level Security (RLS) for
multi-tenant isolation.

Example:
    >>> from backend.registry.db_models import AgentRecordDB, AgentVersionDB
    >>> from sqlalchemy.ext.asyncio import AsyncSession
    >>>
    >>> async with session.begin():
    ...     agent = AgentRecordDB(name="test-agent", ...)
    ...     session.add(agent)
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property

# Import base from the existing db module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.base import Base


class AgentRecordDB(Base):
    """
    SQLAlchemy ORM model for agent records.

    This table stores all registered agents with their metadata,
    configuration, and audit information. Supports multi-tenant
    isolation via Row-Level Security.

    Attributes:
        id: Primary key (UUID)
        name: Unique agent name
        version: Current version string
        description: Agent description
        category: Agent category for classification
        pack_yaml: Full pack.yaml configuration (JSONB)
        generated_code: Generated code artifacts (JSONB)
        checksum: SHA-256 integrity checksum
        status: Lifecycle status (draft/published/deprecated)
        author: Agent author or owner
        created_at: Creation timestamp
        updated_at: Last modification timestamp
        downloads: Total download count
        certification_status: Per-framework certifications (JSONB)

    Indexes:
        - ix_agent_records_name: Unique name lookup
        - ix_agent_records_category: Category filtering
        - ix_agent_records_status: Status filtering
        - ix_agent_records_author: Author filtering
        - ix_agent_records_tags: GIN index for tag search
        - ix_agent_records_search: Full-text search
    """

    __tablename__ = "agent_records"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique agent identifier",
    )

    # Core fields
    name = Column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        comment="Agent name (unique, lowercase)",
    )
    version = Column(
        String(50),
        nullable=False,
        default="1.0.0",
        comment="Current semantic version",
    )
    description = Column(
        Text,
        nullable=False,
        default="",
        comment="Agent description",
    )
    category = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Agent category",
    )

    # Configuration storage (JSONB for PostgreSQL)
    pack_yaml = Column(
        JSONB,
        nullable=False,
        default={},
        comment="Full pack.yaml configuration",
    )
    generated_code = Column(
        JSONB,
        nullable=False,
        default={},
        comment="Generated code artifacts",
    )

    # Integrity and status
    checksum = Column(
        String(128),
        nullable=False,
        default="",
        comment="SHA-256 checksum",
    )
    status = Column(
        String(20),
        nullable=False,
        default="draft",
        index=True,
        comment="Lifecycle status (draft/published/deprecated)",
    )

    # Ownership
    author = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Agent author or owner",
    )
    tenant_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Tenant ID for multi-tenancy (RLS)",
    )

    # Timestamps
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="Creation timestamp",
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="Last update timestamp",
    )

    # Metrics
    downloads = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total download count",
    )

    # Certification and compliance
    certification_status = Column(
        JSONB,
        nullable=False,
        default=[],
        comment="Per-framework certification status",
    )

    # Tags and frameworks (PostgreSQL arrays)
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=[],
        comment="Searchable tags",
    )
    regulatory_frameworks = Column(
        ARRAY(String),
        nullable=False,
        default=[],
        comment="Applicable regulatory frameworks",
    )

    # Additional metadata
    documentation_url = Column(
        String(500),
        nullable=True,
        comment="Documentation URL",
    )
    repository_url = Column(
        String(500),
        nullable=True,
        comment="Source repository URL",
    )
    license = Column(
        String(50),
        nullable=False,
        default="Apache-2.0",
        comment="License identifier",
    )

    # Relationships
    versions = relationship(
        "AgentVersionDB",
        back_populates="agent",
        cascade="all, delete-orphan",
        order_by="AgentVersionDB.created_at.desc()",
    )

    # Table constraints and indexes
    __table_args__ = (
        # Check constraints
        CheckConstraint(
            "status IN ('draft', 'published', 'deprecated')",
            name="ck_agent_records_status",
        ),
        CheckConstraint(
            "downloads >= 0",
            name="ck_agent_records_downloads_positive",
        ),
        # Composite indexes
        Index("ix_agent_records_category_status", "category", "status"),
        Index("ix_agent_records_author_status", "author", "status"),
        Index("ix_agent_records_tenant_status", "tenant_id", "status"),
        # GIN indexes for array fields (PostgreSQL)
        Index("ix_agent_records_tags", "tags", postgresql_using="gin"),
        Index(
            "ix_agent_records_frameworks",
            "regulatory_frameworks",
            postgresql_using="gin",
        ),
        # Table comment
        {"comment": "Registered agents in the Agent Registry"},
    )

    @validates("name")
    def validate_name(self, key: str, value: str) -> str:
        """Validate and normalize agent name."""
        if not value or len(value) < 3:
            raise ValueError("Name must be at least 3 characters")
        return value.lower().strip()

    @validates("status")
    def validate_status(self, key: str, value: str) -> str:
        """Validate status value."""
        valid_statuses = {"draft", "published", "deprecated"}
        if value not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return value

    @hybrid_property
    def is_published(self) -> bool:
        """Check if agent is published."""
        return self.status == "published"

    @hybrid_property
    def version_count(self) -> int:
        """Get total version count."""
        return len(self.versions) if self.versions else 0

    def __repr__(self) -> str:
        """String representation."""
        return f"<AgentRecordDB(name={self.name}, version={self.version}, status={self.status})>"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for API responses.

        Returns:
            Dictionary representation of agent record.
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "checksum": self.checksum,
            "status": self.status,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "downloads": self.downloads,
            "tags": self.tags or [],
            "regulatory_frameworks": self.regulatory_frameworks or [],
            "certification_status": self.certification_status or [],
            "documentation_url": self.documentation_url,
            "repository_url": self.repository_url,
            "license": self.license,
            "version_count": len(self.versions) if self.versions else 0,
        }


class AgentVersionDB(Base):
    """
    SQLAlchemy ORM model for agent versions.

    Tracks individual versions of agents with changelogs,
    breaking change indicators, and artifact references.

    Attributes:
        id: Version record identifier
        agent_id: Reference to parent agent
        version: Semantic version string
        changelog: Version changelog (markdown)
        breaking_changes: Whether version has breaking changes
        release_notes: Detailed release notes
        artifact_path: Storage path for version artifacts
        checksum: SHA-256 checksum of artifacts
        is_latest: Whether this is the latest version
        downloads: Version-specific download count

    Indexes:
        - ix_agent_versions_agent: Agent foreign key
        - ix_agent_versions_version: Version lookup
        - ix_agent_versions_latest: Latest version filtering
    """

    __tablename__ = "agent_versions"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Version record identifier",
    )

    # Foreign key to agent
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent agent ID",
    )

    # Version info
    version = Column(
        String(50),
        nullable=False,
        comment="Semantic version string",
    )
    changelog = Column(
        Text,
        nullable=False,
        default="",
        comment="Version changelog (markdown)",
    )
    breaking_changes = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Has breaking changes",
    )
    release_notes = Column(
        Text,
        nullable=False,
        default="",
        comment="Detailed release notes",
    )

    # Artifact storage
    artifact_path = Column(
        String(500),
        nullable=True,
        comment="Path or URL to version artifacts",
    )
    checksum = Column(
        String(128),
        nullable=False,
        default="",
        comment="SHA-256 checksum of artifacts",
    )

    # Status flags
    is_latest = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="Is this the latest version",
    )

    # Timestamps
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="Version creation timestamp",
    )
    published_at = Column(
        DateTime,
        nullable=True,
        comment="When version was published",
    )
    deprecated_at = Column(
        DateTime,
        nullable=True,
        comment="When version was deprecated",
    )

    # Metrics
    downloads = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Version-specific download count",
    )

    # Dependencies and compatibility
    min_runtime_version = Column(
        String(50),
        nullable=True,
        comment="Minimum GreenLang runtime version",
    )
    dependencies = Column(
        JSONB,
        nullable=False,
        default={},
        comment="Agent dependencies with version constraints",
    )

    # Pack configuration snapshot for this version
    pack_yaml_snapshot = Column(
        JSONB,
        nullable=True,
        comment="Pack.yaml at time of version creation",
    )
    generated_code_snapshot = Column(
        JSONB,
        nullable=True,
        comment="Generated code at time of version creation",
    )

    # Relationships
    agent = relationship(
        "AgentRecordDB",
        back_populates="versions",
    )

    # Table constraints and indexes
    __table_args__ = (
        # Unique constraint: agent + version
        UniqueConstraint(
            "agent_id",
            "version",
            name="uq_agent_versions_agent_version",
        ),
        # Check constraints
        CheckConstraint(
            "downloads >= 0",
            name="ck_agent_versions_downloads_positive",
        ),
        # Composite indexes
        Index("ix_agent_versions_agent_latest", "agent_id", "is_latest"),
        Index("ix_agent_versions_agent_created", "agent_id", "created_at"),
        # Table comment
        {"comment": "Agent version history and artifacts"},
    )

    @validates("version")
    def validate_version(self, key: str, value: str) -> str:
        """Validate semantic version format."""
        import re
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"
        if not re.match(pattern, value):
            raise ValueError("Version must follow semantic versioning (X.Y.Z)")
        return value

    @hybrid_property
    def is_deprecated(self) -> bool:
        """Check if version is deprecated."""
        return self.deprecated_at is not None

    def __repr__(self) -> str:
        """String representation."""
        return f"<AgentVersionDB(version={self.version}, is_latest={self.is_latest})>"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for API responses.

        Returns:
            Dictionary representation of version record.
        """
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "version": self.version,
            "changelog": self.changelog,
            "breaking_changes": self.breaking_changes,
            "release_notes": self.release_notes,
            "artifact_path": self.artifact_path,
            "checksum": self.checksum,
            "is_latest": self.is_latest,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "downloads": self.downloads,
            "min_runtime_version": self.min_runtime_version,
            "dependencies": self.dependencies or {},
        }


# Additional indexes for full-text search (to be created in migration)
# CREATE INDEX ix_agent_records_search ON agent_records
# USING gin(to_tsvector('english', name || ' ' || description));

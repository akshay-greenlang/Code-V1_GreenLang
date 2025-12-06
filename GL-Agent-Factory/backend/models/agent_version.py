"""
Agent Version SQLAlchemy Model

This module defines the AgentVersion model for version management.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from db.base import Base


class AgentVersion(Base):
    """
    Agent Version model.

    Represents a specific version of an agent with its artifact
    and metadata.

    Attributes:
        id: Primary key
        agent_id: Foreign key to agent
        version: Semantic version (X.Y.Z)
        artifact_path: S3 path to artifact
        checksum: SHA-256 checksum of artifact
        changelog: Version changelog
        is_latest: Whether this is the latest version
    """

    __tablename__ = "agent_versions"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Foreign key to agent
    agent_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id"),
        nullable=False,
        index=True,
    )

    # Version info
    version = Column(
        String(50),
        nullable=False,
        comment="Semantic version",
    )
    artifact_path = Column(
        String(500),
        nullable=True,
        comment="S3 path to artifact",
    )
    checksum = Column(
        String(64),
        nullable=True,
        comment="SHA-256 checksum",
    )
    changelog = Column(
        Text,
        nullable=True,
        comment="Version changelog",
    )
    is_latest = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Is this the latest version",
    )

    # Timestamps
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    deprecated_at = Column(
        DateTime,
        nullable=True,
        comment="When version was deprecated",
    )
    sunset_date = Column(
        DateTime,
        nullable=True,
        comment="When version will be removed",
    )

    # Relationships
    agent = relationship(
        "Agent",
        back_populates="versions",
    )

    # Indexes and constraints
    __table_args__ = (
        Index("ix_agent_versions_agent_version", "agent_uuid", "version", unique=True),
        Index("ix_agent_versions_latest", "agent_uuid", "is_latest"),
        {"comment": "Agent versions"},
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<AgentVersion(version={self.version}, is_latest={self.is_latest})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_uuid),
            "version": self.version,
            "artifact_path": self.artifact_path,
            "checksum": self.checksum,
            "changelog": self.changelog,
            "is_latest": self.is_latest,
            "created_at": self.created_at.isoformat(),
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
        }

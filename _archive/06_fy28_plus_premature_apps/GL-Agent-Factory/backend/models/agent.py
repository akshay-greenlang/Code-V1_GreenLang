"""
Agent SQLAlchemy Model

This module defines the Agent database model with Row-Level Security support.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    Float,
    Boolean,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship
import uuid

from db.base import Base


class AgentState:
    """Agent lifecycle states."""

    DRAFT = "DRAFT"
    EXPERIMENTAL = "EXPERIMENTAL"
    CERTIFIED = "CERTIFIED"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


class Agent(Base):
    """
    Agent model.

    Represents a registered agent in the Agent Factory with its
    metadata, lifecycle state, and configuration.

    Attributes:
        id: Primary key (UUID)
        agent_id: Unique agent identifier (category/name format)
        name: Human-readable name
        description: Agent description
        category: Agent category (emissions, cbam, csrd, etc.)
        state: Lifecycle state (DRAFT, CERTIFIED, etc.)
        tenant_id: Owning tenant (for RLS)
        tags: Searchable tags
        regulatory_frameworks: Applicable regulatory frameworks
        spec: Full agent specification (JSON)
    """

    __tablename__ = "agents"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Primary key",
    )

    # Agent identifier (category/name format)
    agent_id = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique agent identifier",
    )

    # Basic info
    name = Column(
        String(255),
        nullable=False,
        comment="Human-readable name",
    )
    description = Column(
        Text,
        nullable=True,
        comment="Agent description",
    )
    category = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Agent category",
    )

    # Lifecycle state
    state = Column(
        String(50),
        nullable=False,
        default=AgentState.DRAFT,
        index=True,
        comment="Lifecycle state",
    )

    # Multi-tenancy
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id"),
        nullable=False,
        index=True,
        comment="Owning tenant",
    )

    # Tags and frameworks
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
        comment="Applicable frameworks",
    )

    # Configuration
    entrypoint = Column(
        String(500),
        nullable=False,
        comment="Python entrypoint",
    )
    deterministic = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Is agent deterministic",
    )

    # Full specification (JSON)
    spec = Column(
        JSON,
        nullable=True,
        comment="Full agent specification",
    )

    # Metrics
    invocation_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total invocations",
    )
    success_rate = Column(
        Float,
        nullable=False,
        default=1.0,
        comment="Success rate (0-1)",
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

    # Relationships
    versions = relationship(
        "AgentVersion",
        back_populates="agent",
        cascade="all, delete-orphan",
    )
    executions = relationship(
        "Execution",
        back_populates="agent",
    )
    tenant = relationship(
        "Tenant",
        back_populates="agents",
    )

    # Indexes
    __table_args__ = (
        Index("ix_agents_tenant_category", "tenant_id", "category"),
        Index("ix_agents_tenant_state", "tenant_id", "state"),
        Index("ix_agents_tags", "tags", postgresql_using="gin"),
        {"comment": "Registered agents"},
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<Agent(agent_id={self.agent_id}, state={self.state})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "state": self.state,
            "tenant_id": str(self.tenant_id),
            "tags": self.tags,
            "regulatory_frameworks": self.regulatory_frameworks,
            "invocation_count": self.invocation_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

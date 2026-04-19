"""
Execution SQLAlchemy Model

This module defines the Execution model for tracking agent executions.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from db.base import Base


class ExecutionStatus:
    """Execution status values."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


class Execution(Base):
    """
    Execution model.

    Represents an agent execution with its input, output, and metrics.

    Attributes:
        id: Primary key
        execution_id: Unique execution identifier
        agent_id: Foreign key to agent
        status: Execution status
        input_hash: SHA-256 hash of input
        output_hash: SHA-256 hash of output
        provenance_hash: Full provenance chain hash
    """

    __tablename__ = "executions"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Execution identifier
    execution_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    # Foreign key to agent
    agent_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id"),
        nullable=False,
        index=True,
    )

    # Multi-tenancy
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id"),
        nullable=False,
        index=True,
    )

    # User who initiated
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )

    # Status
    status = Column(
        String(50),
        nullable=False,
        default=ExecutionStatus.PENDING,
        index=True,
    )
    progress = Column(
        Integer,
        nullable=True,
        comment="Progress percentage (0-100)",
    )
    error_message = Column(
        Text,
        nullable=True,
    )

    # Input/Output
    input_data = Column(
        JSON,
        nullable=True,
        comment="Input data",
    )
    output_data = Column(
        JSON,
        nullable=True,
        comment="Output data",
    )

    # Provenance
    input_hash = Column(
        String(64),
        nullable=True,
        comment="SHA-256 of input",
    )
    output_hash = Column(
        String(64),
        nullable=True,
        comment="SHA-256 of output",
    )
    provenance_hash = Column(
        String(64),
        nullable=True,
        comment="Full provenance chain hash",
    )

    # Metrics
    duration_ms = Column(
        Float,
        nullable=True,
        comment="Execution duration in ms",
    )
    llm_tokens_input = Column(
        Integer,
        default=0,
    )
    llm_tokens_output = Column(
        Integer,
        default=0,
    )

    # Cost
    compute_cost_usd = Column(
        Float,
        default=0,
    )
    llm_cost_usd = Column(
        Float,
        default=0,
    )
    total_cost_usd = Column(
        Float,
        default=0,
    )

    # Timestamps
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )
    started_at = Column(
        DateTime,
        nullable=True,
    )
    completed_at = Column(
        DateTime,
        nullable=True,
    )

    # Version used
    version_used = Column(
        String(50),
        nullable=True,
    )

    # Relationships
    agent = relationship(
        "Agent",
        back_populates="executions",
    )

    # Indexes
    __table_args__ = (
        Index("ix_executions_agent_created", "agent_uuid", "created_at"),
        Index("ix_executions_tenant_status", "tenant_id", "status"),
        {"comment": "Agent executions"},
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<Execution(id={self.execution_id}, status={self.status})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "execution_id": self.execution_id,
            "agent_id": str(self.agent_uuid),
            "status": self.status,
            "progress": self.progress,
            "provenance_hash": self.provenance_hash,
            "duration_ms": self.duration_ms,
            "total_cost_usd": self.total_cost_usd,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

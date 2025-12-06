"""
Audit Log SQLAlchemy Model

This module defines the AuditLog model for compliance tracking.
"""

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
import uuid

from db.base import Base


class AuditLog(Base):
    """
    Audit Log model.

    Provides tamper-evident audit logging for regulatory compliance.

    Attributes:
        id: Primary key
        actor: User or system that performed the action
        action: Action performed (CREATE, UPDATE, DELETE, EXECUTE, etc.)
        resource_type: Type of resource (AGENT, EXECUTION, TENANT, etc.)
        resource_id: ID of the resource
        context: Additional context (JSON)
    """

    __tablename__ = "audit_logs"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Who
    actor = Column(
        String(255),
        nullable=False,
        index=True,
        comment="User or system ID",
    )
    actor_type = Column(
        String(50),
        nullable=False,
        default="user",
        comment="Actor type (user, system, api_key)",
    )

    # What
    action = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Action performed",
    )
    resource_type = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Resource type",
    )
    resource_id = Column(
        String(255),
        nullable=False,
        comment="Resource identifier",
    )

    # Context
    context = Column(
        JSON,
        nullable=True,
        comment="Additional context",
    )
    old_value = Column(
        JSON,
        nullable=True,
        comment="Previous value (for updates)",
    )
    new_value = Column(
        JSON,
        nullable=True,
        comment="New value (for updates)",
    )

    # Multi-tenancy
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id"),
        nullable=False,
        index=True,
    )

    # Request metadata
    ip_address = Column(
        String(50),
        nullable=True,
    )
    user_agent = Column(
        Text,
        nullable=True,
    )
    correlation_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    # Timestamp
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    # Hash chain for tamper evidence
    previous_hash = Column(
        String(64),
        nullable=True,
        comment="Hash of previous entry",
    )
    entry_hash = Column(
        String(64),
        nullable=True,
        comment="Hash of this entry",
    )

    # Indexes
    __table_args__ = (
        Index("ix_audit_logs_tenant_time", "tenant_id", "created_at"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        {"comment": "Audit logs"},
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<AuditLog(action={self.action}, resource={self.resource_type}/{self.resource_id})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "actor": self.actor,
            "actor_type": self.actor_type,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "context": self.context,
            "tenant_id": str(self.tenant_id),
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
        }

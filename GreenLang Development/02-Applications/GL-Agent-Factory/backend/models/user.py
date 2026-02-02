"""
User SQLAlchemy Model

This module defines the User model.
"""

from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import Column, DateTime, ForeignKey, String, Boolean
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship
import uuid

from db.base import Base


class User(Base):
    """
    User model.

    Represents a user in the system.

    Attributes:
        id: Primary key
        email: User email
        tenant_id: Foreign key to tenant
        roles: List of role names
    """

    __tablename__ = "users"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # User identifier
    user_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    email = Column(
        String(255),
        unique=True,
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

    # Roles
    roles = Column(
        ARRAY(String),
        nullable=False,
        default=["viewer"],
        comment="User roles",
    )

    # Status
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
    )

    # Timestamps
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    last_login_at = Column(
        DateTime,
        nullable=True,
    )

    # Relationships
    tenant = relationship(
        "Tenant",
        back_populates="users",
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<User(email={self.email}, roles={self.roles})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "email": self.email,
            "tenant_id": str(self.tenant_id),
            "roles": self.roles,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return "admin" in self.roles

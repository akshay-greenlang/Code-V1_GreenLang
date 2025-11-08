"""
Database models for Phase 4 Authentication and Authorization

Supports:
- RBAC (Role-Based Access Control)
- ABAC (Attribute-Based Access Control)
- SSO (SAML, OAuth, LDAP)
- Multi-tenancy
- Audit logging
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    JSON,
)
from sqlalchemy.orm import relationship

from greenlang.db.base import Base


class User(Base):
    """User model with multi-tenancy support"""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    username = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=True)  # Nullable for SSO-only users

    # Profile
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    display_name = Column(String(255), nullable=True)

    # Status
    active = Column(Boolean, default=True, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    locked = Column(Boolean, default=False, nullable=False)
    locked_at = Column(DateTime, nullable=True)
    locked_reason = Column(Text, nullable=True)

    # MFA
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(255), nullable=True)
    mfa_backup_codes = Column(Text, nullable=True)  # JSON array

    # SSO
    sso_provider = Column(String(50), nullable=True)  # saml, oauth, ldap
    sso_provider_id = Column(String(36), nullable=True)
    sso_external_id = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, nullable=True)

    # Metadata
    metadata = Column(JSON, nullable=True)

    # Relationships
    roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "username", name="uq_user_tenant_username"),
        UniqueConstraint("tenant_id", "email", name="uq_user_tenant_email"),
        Index("idx_user_email", "email"),
        Index("idx_user_sso", "sso_provider", "sso_external_id"),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, tenant_id={self.tenant_id})>"


class Role(Base):
    """Role model for RBAC"""

    __tablename__ = "roles"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=True, index=True)  # Null for system roles
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Role hierarchy
    parent_role_id = Column(String(36), ForeignKey("roles.id"), nullable=True)

    # Role type
    is_system_role = Column(Boolean, default=False, nullable=False)
    is_default_role = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Metadata for ABAC
    metadata = Column(JSON, nullable=True)

    # Relationships
    parent_role = relationship("Role", remote_side=[id], backref="child_roles")
    permissions = relationship("Permission", back_populates="role", cascade="all, delete-orphan")
    user_roles = relationship("UserRole", back_populates="role")

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_role_tenant_name"),
        Index("idx_role_parent", "parent_role_id"),
    )

    def __repr__(self):
        return f"<Role(id={self.id}, name={self.name}, tenant_id={self.tenant_id})>"


class Permission(Base):
    """Permission model for fine-grained access control"""

    __tablename__ = "permissions"

    id = Column(String(36), primary_key=True)
    role_id = Column(String(36), ForeignKey("roles.id"), nullable=False)

    # Permission definition
    resource_type = Column(String(255), nullable=False)  # pipeline, pack, dataset, etc.
    resource_id = Column(String(255), nullable=True)  # Specific resource or pattern
    action = Column(String(255), nullable=False)  # create, read, update, delete, execute, etc.

    # Scope
    scope = Column(String(255), nullable=True)  # tenant:123, namespace:prod, etc.

    # ABAC conditions
    conditions = Column(JSON, nullable=True)  # JSON object with conditions

    # Effect
    effect = Column(String(10), default="allow", nullable=False)  # allow or deny

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    role = relationship("Role", back_populates="permissions")

    # Indexes
    __table_args__ = (
        Index("idx_permission_role", "role_id"),
        Index("idx_permission_resource", "resource_type", "resource_id"),
    )

    def __repr__(self):
        return f"<Permission(id={self.id}, resource_type={self.resource_type}, action={self.action})>"


class UserRole(Base):
    """User-Role association with expiry support"""

    __tablename__ = "user_roles"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    role_id = Column(String(36), ForeignKey("roles.id"), nullable=False)

    # Temporal validity
    assigned_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    assigned_by = Column(String(36), nullable=True)

    # Relationships
    user = relationship("User", back_populates="roles")
    role = relationship("Role", back_populates="user_roles")

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="uq_user_role"),
        Index("idx_user_role_user", "user_id"),
        Index("idx_user_role_role", "role_id"),
    )

    def __repr__(self):
        return f"<UserRole(user_id={self.user_id}, role_id={self.role_id})>"


class Session(Base):
    """Session model for tracking user sessions"""

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    tenant_id = Column(String(36), nullable=False, index=True)

    # Session data
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), nullable=True)

    # Session metadata
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    device_id = Column(String(255), nullable=True)

    # Validity
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    last_activity_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Status
    active = Column(Boolean, default=True, nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    revoke_reason = Column(String(255), nullable=True)

    # Redis key (for distributed sessions)
    redis_key = Column(String(255), nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    # Indexes
    __table_args__ = (
        Index("idx_session_user", "user_id"),
        Index("idx_session_expiry", "expires_at"),
        Index("idx_session_active", "active", "revoked"),
    )

    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id})>"


class APIKey(Base):
    """API Key model for programmatic access"""

    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    tenant_id = Column(String(36), nullable=False, index=True)

    # Key data
    key_id = Column(String(32), unique=True, nullable=False, index=True)
    key_hash = Column(String(255), nullable=False)  # Hashed secret
    key_prefix = Column(String(10), nullable=False)  # For display (e.g., "glk_abc...")

    # Metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Permissions
    scopes = Column(JSON, nullable=True)  # List of scopes

    # Validity
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    last_rotated_at = Column(DateTime, nullable=True)

    # Status
    active = Column(Boolean, default=True, nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    # Usage tracking
    use_count = Column(Integer, default=0, nullable=False)
    rate_limit = Column(Integer, nullable=True)  # Requests per hour

    # Restrictions
    allowed_ips = Column(JSON, nullable=True)  # List of allowed IP addresses
    allowed_origins = Column(JSON, nullable=True)  # List of allowed origins

    # Relationships
    user = relationship("User", back_populates="api_keys")

    # Indexes
    __table_args__ = (
        Index("idx_apikey_user", "user_id"),
        Index("idx_apikey_active", "active", "revoked"),
    )

    def __repr__(self):
        return f"<APIKey(id={self.id}, key_prefix={self.key_prefix})>"


class AuditLog(Base):
    """Audit log for tracking all authentication and authorization events"""

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)

    # Event details
    event_type = Column(String(255), nullable=False)  # login, logout, permission_check, etc.
    event_category = Column(String(50), nullable=False)  # auth, authz, admin, data
    resource_type = Column(String(255), nullable=True)
    resource_id = Column(String(255), nullable=True)
    action = Column(String(255), nullable=True)

    # Result
    result = Column(String(50), nullable=False)  # success, failure, denied
    reason = Column(Text, nullable=True)

    # Context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(36), nullable=True)

    # Additional data
    metadata = Column(JSON, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    # Indexes
    __table_args__ = (
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_event", "event_type", "event_category"),
        Index("idx_audit_result", "result"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_timestamp", "timestamp"),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, event_type={self.event_type}, result={self.result})>"


class SAMLProvider(Base):
    """SAML Identity Provider configuration"""

    __tablename__ = "saml_providers"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False, index=True)

    # Provider details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    entity_id = Column(String(255), nullable=False)

    # SAML configuration
    sso_url = Column(String(255), nullable=False)
    slo_url = Column(String(255), nullable=True)
    x509_cert = Column(Text, nullable=False)
    metadata_url = Column(String(255), nullable=True)

    # Attribute mapping
    attribute_mapping = Column(JSON, nullable=True)  # Map SAML attributes to user fields

    # Status
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Metadata
    metadata = Column(JSON, nullable=True)

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "entity_id", name="uq_saml_tenant_entity"),
    )

    def __repr__(self):
        return f"<SAMLProvider(id={self.id}, name={self.name})>"


class OAuthProvider(Base):
    """OAuth 2.0 Provider configuration"""

    __tablename__ = "oauth_providers"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False, index=True)

    # Provider details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    provider_type = Column(String(50), nullable=False)  # google, github, azure, custom

    # OAuth configuration
    client_id = Column(String(255), nullable=False)
    client_secret = Column(String(255), nullable=False)  # Encrypted
    authorization_url = Column(String(255), nullable=False)
    token_url = Column(String(255), nullable=False)
    userinfo_url = Column(String(255), nullable=True)
    scopes = Column(JSON, nullable=True)  # List of scopes

    # Attribute mapping
    attribute_mapping = Column(JSON, nullable=True)

    # Status
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Metadata
    metadata = Column(JSON, nullable=True)

    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "provider_type", "name", name="uq_oauth_tenant_provider"),
    )

    def __repr__(self):
        return f"<OAuthProvider(id={self.id}, name={self.name}, provider_type={self.provider_type})>"


class LDAPConfig(Base):
    """LDAP/Active Directory configuration"""

    __tablename__ = "ldap_configs"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False, index=True, unique=True)

    # LDAP server details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    server_url = Column(String(255), nullable=False)
    port = Column(Integer, default=389, nullable=False)
    use_ssl = Column(Boolean, default=False, nullable=False)
    use_tls = Column(Boolean, default=False, nullable=False)

    # Authentication
    bind_dn = Column(String(255), nullable=True)
    bind_password = Column(String(255), nullable=True)  # Encrypted

    # Search configuration
    base_dn = Column(String(255), nullable=False)
    user_search_base = Column(String(255), nullable=True)
    user_search_filter = Column(String(255), nullable=True)
    group_search_base = Column(String(255), nullable=True)
    group_search_filter = Column(String(255), nullable=True)

    # Attribute mapping
    attribute_mapping = Column(JSON, nullable=True)
    group_role_mapping = Column(JSON, nullable=True)  # Map LDAP groups to roles

    # Status
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Metadata
    metadata = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<LDAPConfig(id={self.id}, name={self.name}, server_url={self.server_url})>"

"""
SQLAlchemy database models for GreenLang Agent Foundation.

All tables for agent memory, tenancy, users, audit logs, and LLM tracking.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, JSON, Text,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
import enum

Base = declarative_base()


# ============================================================================
# ENUMS
# ============================================================================

class AgentStatus(str, enum.Enum):
    """Agent status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class MemoryType(str, enum.Enum):
    """Memory type enumeration."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"


class TenantStatus(str, enum.Enum):
    """Tenant status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CANCELLED = "cancelled"


class UserRole(str, enum.Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API = "api"


class AuditAction(str, enum.Enum):
    """Audit action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"


# ============================================================================
# TENANT MANAGEMENT
# ============================================================================

class Tenant(Base):
    """
    Multi-tenant organization table.

    Each tenant represents an isolated customer environment.
    """
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    status = Column(SQLEnum(TenantStatus), nullable=False, default=TenantStatus.TRIAL)

    # Subscription
    plan_type = Column(String(50), nullable=False, default="trial")  # trial, basic, pro, enterprise
    max_agents = Column(Integer, nullable=False, default=10)
    max_users = Column(Integer, nullable=False, default=5)
    max_api_calls_per_month = Column(Integer, nullable=False, default=10000)

    # Metadata
    metadata = Column(JSONB, nullable=True)
    settings = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_tenant_status", "status"),
        Index("idx_tenant_created", "created_at"),
    )


# ============================================================================
# USER MANAGEMENT
# ============================================================================

class User(Base):
    """
    User accounts with role-based access control.
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    api_key_hash = Column(String(255), nullable=True, unique=True, index=True)

    # Profile
    full_name = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.USER)
    is_active = Column(Boolean, nullable=False, default=True)
    is_verified = Column(Boolean, nullable=False, default=False)

    # Metadata
    metadata = Column(JSONB, nullable=True)
    preferences = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_user_tenant", "tenant_id", "email"),
        Index("idx_user_role", "role"),
    )


# ============================================================================
# AGENT MEMORY TABLES
# ============================================================================

class AgentMemory(Base):
    """
    Agent memory storage for short-term and long-term memory.

    Stores all agent memories with tenant isolation and provenance tracking.
    """
    __tablename__ = "agent_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Agent identification
    agent_id = Column(String(255), nullable=False, index=True)
    agent_type = Column(String(100), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)

    # Memory details
    memory_type = Column(SQLEnum(MemoryType), nullable=False, index=True)
    key = Column(String(255), nullable=False, index=True)
    value = Column(JSONB, nullable=False)

    # Metadata
    importance = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)

    # Provenance
    provenance_hash = Column(String(64), nullable=False, index=True)
    source = Column(String(255), nullable=True)

    # TTL and expiration
    ttl_seconds = Column(Integer, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_memory_agent_session", "tenant_id", "agent_id", "session_id"),
        Index("idx_memory_type", "memory_type", "created_at"),
        Index("idx_memory_key", "tenant_id", "key"),
        Index("idx_memory_expires", "expires_at"),
        UniqueConstraint("tenant_id", "agent_id", "session_id", "memory_type", "key", name="uq_memory_key"),
    )


class AgentState(Base):
    """
    Agent state snapshots for recovery and debugging.
    """
    __tablename__ = "agent_state"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Agent identification
    agent_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)

    # State
    state = Column(JSONB, nullable=False)
    status = Column(SQLEnum(AgentStatus), nullable=False, default=AgentStatus.ACTIVE)

    # Provenance
    provenance_hash = Column(String(64), nullable=False)
    checkpoint_reason = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_state_agent", "tenant_id", "agent_id", "created_at"),
        Index("idx_state_session", "session_id"),
    )


# ============================================================================
# AUDIT LOGS
# ============================================================================

class AuditLog(Base):
    """
    Comprehensive audit trail for all system actions.

    Tracks all CRUD operations, agent executions, and API calls for compliance.
    """
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Actor
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    agent_id = Column(String(255), nullable=True, index=True)
    api_key_id = Column(String(255), nullable=True)

    # Action
    action = Column(SQLEnum(AuditAction), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(255), nullable=False, index=True)

    # Details
    description = Column(Text, nullable=True)
    changes = Column(JSONB, nullable=True)  # Before/after values
    metadata = Column(JSONB, nullable=True)

    # Context
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(255), nullable=True, index=True)

    # Status
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_audit_tenant_created", "tenant_id", "created_at"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_user", "user_id", "created_at"),
        Index("idx_audit_action", "action", "created_at"),
    )


# ============================================================================
# LLM API TRACKING
# ============================================================================

class LLMAPICall(Base):
    """
    Track all LLM API calls for cost monitoring and compliance.

    Records every LLM invocation with complete provenance and performance metrics.
    """
    __tablename__ = "llm_api_calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Agent context
    agent_id = Column(String(255), nullable=False, index=True)
    agent_type = Column(String(100), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)

    # LLM details
    provider = Column(String(50), nullable=False, index=True)  # openai, anthropic, etc.
    model = Column(String(100), nullable=False, index=True)
    purpose = Column(String(255), nullable=False)  # classification, summarization, etc.

    # Request
    prompt = Column(Text, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    max_tokens = Column(Integer, nullable=True)
    temperature = Column(Float, nullable=True)
    other_params = Column(JSONB, nullable=True)

    # Response
    response = Column(Text, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=False)

    # Performance
    latency_ms = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)

    # Cost tracking
    cost_usd = Column(Float, nullable=True)

    # Provenance
    provenance_hash = Column(String(64), nullable=False, index=True)
    request_id = Column(String(255), nullable=True, index=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_llm_tenant_created", "tenant_id", "created_at"),
        Index("idx_llm_agent", "agent_id", "created_at"),
        Index("idx_llm_provider_model", "provider", "model", "created_at"),
        Index("idx_llm_cost", "tenant_id", "created_at", "cost_usd"),
    )


# ============================================================================
# MATERIALIZED VIEWS FOR CACHING
# ============================================================================

class AgentMemorySummary(Base):
    """
    Materialized view for agent memory aggregations.

    This is a cache layer (L4) for expensive queries.
    Refreshed periodically by background jobs.
    """
    __tablename__ = "agent_memory_summary_mv"

    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True)
    agent_id = Column(String(255), primary_key=True)
    memory_type = Column(SQLEnum(MemoryType), primary_key=True)

    # Aggregations
    total_memories = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(Integer, nullable=False, default=0)
    avg_importance = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_summary_tenant", "tenant_id"),
    )


class LLMCostSummary(Base):
    """
    Materialized view for LLM cost aggregations.

    Daily rollups of LLM costs by tenant, provider, and model.
    """
    __tablename__ = "llm_cost_summary_mv"

    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True)
    date = Column(DateTime(timezone=True), primary_key=True)
    provider = Column(String(50), primary_key=True)
    model = Column(String(100), primary_key=True)

    # Aggregations
    total_calls = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    total_cost_usd = Column(Float, nullable=False, default=0.0)
    avg_latency_ms = Column(Float, nullable=False, default=0.0)
    success_rate = Column(Float, nullable=False, default=1.0)

    last_updated = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_cost_summary_tenant_date", "tenant_id", "date"),
    )

"""
SQLAlchemy Database Models for GreenLang Agent Registry

This module defines the 7 core database tables for the Agent Registry:
- agents: Core agent metadata
- agent_versions: Versioned agent releases with lifecycle state
- evaluation_results: Agent evaluation and certification data
- state_transitions: Audit trail of lifecycle state changes
- usage_metrics: Agent usage analytics and performance metrics
- audit_logs: Comprehensive audit logging for all operations
- governance_policies: Tenant-specific governance rules

All models use SQLAlchemy 2.0 async patterns with asyncpg driver.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String,
    Text,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, INET
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map = {
        Dict[str, Any]: JSONB,
        dict: JSONB,
    }


class Agent(Base):
    """
    Agent metadata table - core registry of all agents.

    Stores the base information about an agent independent of version.
    Each agent can have multiple versions tracked in agent_versions.

    Attributes:
        agent_id: Unique identifier for the agent (e.g., 'gl-cbam-calculator-v2')
        name: Human-readable name
        description: Detailed description of agent capabilities
        domain: Classification domain (e.g., 'sustainability.cbam')
        type: Agent type (e.g., 'calculator', 'validator')
        created_by: User/service that created the agent
        team: Owning team identifier
        tenant_id: Multi-tenant isolation key
    """

    __tablename__ = "agents"

    agent_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    domain: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tags: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    team: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    versions: Mapped[List["AgentVersion"]] = relationship(
        "AgentVersion",
        back_populates="agent",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_agents_tenant", "tenant_id"),
        Index("idx_agents_domain", "domain"),
        Index("idx_agents_type", "type"),
        Index("idx_agents_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Agent(agent_id='{self.agent_id}', name='{self.name}')>"


class AgentVersion(Base):
    """
    Agent version table - tracks immutable agent releases.

    Each version represents a specific, immutable release of an agent.
    Versions progress through lifecycle states: draft -> experimental -> certified -> deprecated.

    Attributes:
        version_id: Unique identifier for this version (agent_id:version)
        agent_id: Reference to parent agent
        version: Semantic version string (e.g., '2.3.1')
        semantic_version: Parsed version components as JSONB
        lifecycle_state: Current state (draft, experimental, certified, deprecated)
        container_image: Docker image reference
        image_digest: SHA256 digest for image verification
        metadata: Additional version-specific metadata
        runtime_requirements: CPU, memory, dependencies specification
    """

    __tablename__ = "agent_versions"

    version_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    agent_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("agents.agent_id", ondelete="CASCADE"),
        nullable=False
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    semantic_version: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    lifecycle_state: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="draft"
    )
    container_image: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    image_digest: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    runtime_requirements: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    capabilities: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    deprecated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="versions")
    evaluation_results: Mapped[List["EvaluationResult"]] = relationship(
        "EvaluationResult",
        back_populates="version",
        cascade="all, delete-orphan"
    )
    state_transitions: Mapped[List["StateTransition"]] = relationship(
        "StateTransition",
        back_populates="version",
        cascade="all, delete-orphan"
    )
    usage_metrics: Mapped[List["UsageMetric"]] = relationship(
        "UsageMetric",
        back_populates="version",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("agent_id", "version", name="uq_agent_version"),
        Index("idx_versions_agent", "agent_id"),
        Index("idx_versions_state", "lifecycle_state"),
        Index("idx_versions_agent_version", "agent_id", "version"),
        Index("idx_versions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AgentVersion(version_id='{self.version_id}', state='{self.lifecycle_state}')>"


class EvaluationResult(Base):
    """
    Evaluation results table - stores agent evaluation and certification data.

    Captures comprehensive evaluation metrics including performance,
    quality, compliance checks, and test results for each agent version.

    Attributes:
        evaluation_id: Unique identifier for evaluation run
        version_id: Reference to evaluated agent version
        evaluated_at: Timestamp of evaluation
        evaluator_version: Version of the evaluation framework used
        performance_metrics: Latency, throughput, error rates
        quality_metrics: Accuracy, precision, recall, F1
        compliance_checks: Security, license, dependency audit results
        test_results: Unit, integration, e2e test outcomes
        certification_status: Certification level and expiry
    """

    __tablename__ = "evaluation_results"

    evaluation_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    version_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("agent_versions.version_id", ondelete="CASCADE"),
        nullable=False
    )
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    evaluator_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    performance_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    quality_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    compliance_checks: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    test_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    certification_status: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    version: Mapped["AgentVersion"] = relationship(
        "AgentVersion",
        back_populates="evaluation_results"
    )

    __table_args__ = (
        Index("idx_eval_version", "version_id"),
        Index("idx_eval_evaluated_at", "evaluated_at"),
    )

    def __repr__(self) -> str:
        return f"<EvaluationResult(evaluation_id='{self.evaluation_id}')>"


class StateTransition(Base):
    """
    State transitions table - audit trail of lifecycle state changes.

    Records every state transition for agent versions, providing
    complete traceability of the promotion/deprecation lifecycle.

    Attributes:
        transition_id: Auto-increment primary key
        version_id: Reference to agent version
        from_state: Previous lifecycle state
        to_state: New lifecycle state
        transitioned_at: Timestamp of transition
        transitioned_by: User/service that triggered transition
        reason: Human-readable reason for transition
        metadata: Additional context (promotion criteria, etc.)
    """

    __tablename__ = "state_transitions"

    transition_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("agent_versions.version_id", ondelete="CASCADE"),
        nullable=False
    )
    from_state: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    to_state: Mapped[str] = mapped_column(String(50), nullable=False)
    transitioned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    transitioned_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Relationships
    version: Mapped["AgentVersion"] = relationship(
        "AgentVersion",
        back_populates="state_transitions"
    )

    __table_args__ = (
        Index("idx_transitions_version", "version_id"),
        Index("idx_transitions_at", "transitioned_at"),
    )

    def __repr__(self) -> str:
        return f"<StateTransition(id={self.transition_id}, {self.from_state}->{self.to_state})>"


class UsageMetric(Base):
    """
    Usage metrics table - agent usage analytics and performance tracking.

    Stores time-series usage data for monitoring agent health,
    performance trends, and capacity planning.

    Attributes:
        metric_id: Auto-increment primary key
        version_id: Reference to agent version
        tenant_id: Tenant that generated the metrics
        timestamp: Time bucket for the metrics
        request_count: Number of requests in the time bucket
        error_count: Number of errors in the time bucket
        latency_p50_ms: 50th percentile latency
        latency_p95_ms: 95th percentile latency
        latency_p99_ms: 99th percentile latency
    """

    __tablename__ = "usage_metrics"

    metric_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("agent_versions.version_id", ondelete="CASCADE"),
        nullable=False
    )
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    request_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_p50_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_p95_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_p99_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Relationships
    version: Mapped["AgentVersion"] = relationship(
        "AgentVersion",
        back_populates="usage_metrics"
    )

    __table_args__ = (
        Index("idx_usage_version_time", "version_id", "timestamp"),
        Index("idx_usage_tenant_time", "tenant_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<UsageMetric(id={self.metric_id}, version='{self.version_id}')>"


class AuditLog(Base):
    """
    Audit logs table - comprehensive audit trail for all registry operations.

    Records all significant actions performed on the registry including
    agent publish, update, promote, deprecate, and access attempts.

    Attributes:
        log_id: Auto-increment primary key
        version_id: Reference to agent version (if applicable)
        action: Action performed (publish, promote, deprecate, etc.)
        performed_by: User/service that performed the action
        tenant_id: Tenant context
        timestamp: When the action occurred
        details: Additional action-specific details
        ip_address: Client IP address
        user_agent: Client user agent string
    """

    __tablename__ = "audit_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    performed_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(INET, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    __table_args__ = (
        Index("idx_audit_version", "version_id"),
        Index("idx_audit_agent", "agent_id"),
        Index("idx_audit_tenant", "tenant_id"),
        Index("idx_audit_timestamp", "timestamp"),
        Index("idx_audit_action", "action"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.log_id}, action='{self.action}')>"


class GovernancePolicy(Base):
    """
    Governance policies table - tenant-specific governance rules.

    Stores policies that govern agent usage, promotion criteria,
    access control, and compliance requirements per tenant.

    Attributes:
        policy_id: Unique identifier for the policy
        tenant_id: Tenant this policy applies to (NULL for global)
        policy_type: Type of policy (promotion, access, compliance, etc.)
        policy_rules: JSONB structure containing the policy rules
        active: Whether this policy is currently enforced
        created_by: User/service that created the policy
    """

    __tablename__ = "governance_policies"

    policy_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    policy_type: Mapped[str] = mapped_column(String(50), nullable=False)
    policy_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    policy_rules: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    priority: Mapped[Optional[int]] = mapped_column(Integer, default=100, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    __table_args__ = (
        Index("idx_policy_tenant", "tenant_id"),
        Index("idx_policy_type", "policy_type"),
        Index("idx_policy_active", "active"),
    )

    def __repr__(self) -> str:
        return f"<GovernancePolicy(policy_id='{self.policy_id}', type='{self.policy_type}')>"


# Lifecycle state constants
class LifecycleState:
    """Valid lifecycle states for agent versions."""

    DRAFT = "draft"
    EXPERIMENTAL = "experimental"
    CERTIFIED = "certified"
    DEPRECATED = "deprecated"

    # Valid state transitions
    VALID_TRANSITIONS = {
        DRAFT: [EXPERIMENTAL],
        EXPERIMENTAL: [CERTIFIED, DEPRECATED],
        CERTIFIED: [DEPRECATED],
        DEPRECATED: [],
    }

    @classmethod
    def can_transition(cls, from_state: str, to_state: str) -> bool:
        """Check if a state transition is valid."""
        valid_targets = cls.VALID_TRANSITIONS.get(from_state, [])
        return to_state in valid_targets

    @classmethod
    def all_states(cls) -> list:
        """Return all valid lifecycle states."""
        return [cls.DRAFT, cls.EXPERIMENTAL, cls.CERTIFIED, cls.DEPRECATED]

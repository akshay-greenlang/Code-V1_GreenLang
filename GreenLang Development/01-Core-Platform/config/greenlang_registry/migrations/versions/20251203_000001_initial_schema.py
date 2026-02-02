"""Initial schema with 7 core tables for Agent Registry.

Revision ID: 20251203_000001
Revises:
Create Date: 2025-12-03 00:00:01

This migration creates the initial database schema for the GreenLang Agent Registry:
- agents: Core agent metadata
- agent_versions: Versioned agent releases
- evaluation_results: Agent evaluation data
- state_transitions: Lifecycle state audit trail
- usage_metrics: Usage analytics
- audit_logs: Comprehensive audit logging
- governance_policies: Tenant governance rules
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20251203_000001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all initial tables for Agent Registry."""

    # ==========================================================================
    # Table 1: agents - Core agent metadata
    # ==========================================================================
    op.create_table(
        "agents",
        sa.Column("agent_id", sa.String(255), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("domain", sa.String(100), nullable=True),
        sa.Column("type", sa.String(50), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("team", sa.String(255), nullable=True),
        sa.Column("tenant_id", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_index("idx_agents_tenant", "agents", ["tenant_id"])
    op.create_index("idx_agents_domain", "agents", ["domain"])
    op.create_index("idx_agents_type", "agents", ["type"])
    op.create_index("idx_agents_created_at", "agents", ["created_at"])

    # ==========================================================================
    # Table 2: agent_versions - Versioned agent releases
    # ==========================================================================
    op.create_table(
        "agent_versions",
        sa.Column("version_id", sa.String(255), primary_key=True),
        sa.Column(
            "agent_id",
            sa.String(255),
            sa.ForeignKey("agents.agent_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column(
            "semantic_version",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("lifecycle_state", sa.String(50), nullable=False, default="draft"),
        sa.Column("container_image", sa.String(500), nullable=True),
        sa.Column("image_digest", sa.String(100), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "runtime_requirements",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deprecated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("agent_id", "version", name="uq_agent_version"),
    )

    op.create_index("idx_versions_agent", "agent_versions", ["agent_id"])
    op.create_index("idx_versions_state", "agent_versions", ["lifecycle_state"])
    op.create_index(
        "idx_versions_agent_version", "agent_versions", ["agent_id", "version"]
    )
    op.create_index("idx_versions_created_at", "agent_versions", ["created_at"])

    # ==========================================================================
    # Table 3: evaluation_results - Agent evaluation and certification
    # ==========================================================================
    op.create_table(
        "evaluation_results",
        sa.Column("evaluation_id", sa.String(255), primary_key=True),
        sa.Column(
            "version_id",
            sa.String(255),
            sa.ForeignKey("agent_versions.version_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("evaluator_version", sa.String(50), nullable=True),
        sa.Column(
            "performance_metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "quality_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "compliance_checks",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "test_results", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "certification_status",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_index("idx_eval_version", "evaluation_results", ["version_id"])
    op.create_index("idx_eval_evaluated_at", "evaluation_results", ["evaluated_at"])

    # ==========================================================================
    # Table 4: state_transitions - Lifecycle state audit trail
    # ==========================================================================
    op.create_table(
        "state_transitions",
        sa.Column(
            "transition_id", sa.Integer, primary_key=True, autoincrement=True
        ),
        sa.Column(
            "version_id",
            sa.String(255),
            sa.ForeignKey("agent_versions.version_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("from_state", sa.String(50), nullable=True),
        sa.Column("to_state", sa.String(50), nullable=False),
        sa.Column(
            "transitioned_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("transitioned_by", sa.String(255), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_index("idx_transitions_version", "state_transitions", ["version_id"])
    op.create_index("idx_transitions_at", "state_transitions", ["transitioned_at"])

    # ==========================================================================
    # Table 5: usage_metrics - Usage analytics and performance tracking
    # ==========================================================================
    op.create_table(
        "usage_metrics",
        sa.Column("metric_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "version_id",
            sa.String(255),
            sa.ForeignKey("agent_versions.version_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(255), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("request_count", sa.Integer, nullable=True),
        sa.Column("error_count", sa.Integer, nullable=True),
        sa.Column("latency_p50_ms", sa.Integer, nullable=True),
        sa.Column("latency_p95_ms", sa.Integer, nullable=True),
        sa.Column("latency_p99_ms", sa.Integer, nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_index(
        "idx_usage_version_time", "usage_metrics", ["version_id", "timestamp"]
    )
    op.create_index(
        "idx_usage_tenant_time", "usage_metrics", ["tenant_id", "timestamp"]
    )

    # ==========================================================================
    # Table 6: audit_logs - Comprehensive audit logging
    # ==========================================================================
    op.create_table(
        "audit_logs",
        sa.Column("log_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("version_id", sa.String(255), nullable=True),
        sa.Column("agent_id", sa.String(255), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("performed_by", sa.String(255), nullable=True),
        sa.Column("tenant_id", sa.String(255), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", postgresql.INET, nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(50), nullable=True),
    )

    op.create_index("idx_audit_version", "audit_logs", ["version_id"])
    op.create_index("idx_audit_agent", "audit_logs", ["agent_id"])
    op.create_index("idx_audit_tenant", "audit_logs", ["tenant_id"])
    op.create_index("idx_audit_timestamp", "audit_logs", ["timestamp"])
    op.create_index("idx_audit_action", "audit_logs", ["action"])

    # ==========================================================================
    # Table 7: governance_policies - Tenant governance rules
    # ==========================================================================
    op.create_table(
        "governance_policies",
        sa.Column("policy_id", sa.String(255), primary_key=True),
        sa.Column("tenant_id", sa.String(255), nullable=True),
        sa.Column("policy_type", sa.String(50), nullable=False),
        sa.Column("policy_name", sa.String(255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "policy_rules", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("active", sa.Boolean, default=True, nullable=False),
        sa.Column("priority", sa.Integer, default=100, nullable=True),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_index("idx_policy_tenant", "governance_policies", ["tenant_id"])
    op.create_index("idx_policy_type", "governance_policies", ["policy_type"])
    op.create_index("idx_policy_active", "governance_policies", ["active"])


def downgrade() -> None:
    """Drop all tables in reverse order."""
    # Drop tables with foreign keys first
    op.drop_table("governance_policies")
    op.drop_table("audit_logs")
    op.drop_table("usage_metrics")
    op.drop_table("state_transitions")
    op.drop_table("evaluation_results")
    op.drop_table("agent_versions")
    op.drop_table("agents")

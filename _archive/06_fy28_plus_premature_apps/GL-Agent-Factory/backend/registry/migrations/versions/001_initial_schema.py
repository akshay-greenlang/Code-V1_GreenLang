"""
Initial Schema Creation

Creates the core tables for the Agent Registry:
- agent_records: Main agent table with metadata
- agent_versions: Version history table

Revision ID: 001_initial
Revises: None
Create Date: 2024-12-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create initial Agent Registry schema.

    Creates:
    - agent_records table with all metadata columns
    - agent_versions table with version tracking
    - Foreign key relationships
    - Check constraints
    """
    # Create agent_records table
    op.create_table(
        "agent_records",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique agent identifier",
        ),
        # Core fields
        sa.Column(
            "name",
            sa.String(100),
            nullable=False,
            unique=True,
            comment="Agent name (unique, lowercase)",
        ),
        sa.Column(
            "version",
            sa.String(50),
            nullable=False,
            server_default="1.0.0",
            comment="Current semantic version",
        ),
        sa.Column(
            "description",
            sa.Text,
            nullable=False,
            server_default="",
            comment="Agent description",
        ),
        sa.Column(
            "category",
            sa.String(50),
            nullable=False,
            comment="Agent category",
        ),
        # Configuration storage (JSONB)
        sa.Column(
            "pack_yaml",
            postgresql.JSONB,
            nullable=False,
            server_default="{}",
            comment="Full pack.yaml configuration",
        ),
        sa.Column(
            "generated_code",
            postgresql.JSONB,
            nullable=False,
            server_default="{}",
            comment="Generated code artifacts",
        ),
        # Integrity and status
        sa.Column(
            "checksum",
            sa.String(128),
            nullable=False,
            server_default="",
            comment="SHA-256 checksum",
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="draft",
            comment="Lifecycle status (draft/published/deprecated)",
        ),
        # Ownership
        sa.Column(
            "author",
            sa.String(100),
            nullable=False,
            comment="Agent author or owner",
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="Tenant ID for multi-tenancy (RLS)",
        ),
        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Creation timestamp",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Last update timestamp",
        ),
        # Metrics
        sa.Column(
            "downloads",
            sa.Integer,
            nullable=False,
            server_default="0",
            comment="Total download count",
        ),
        # Certification and compliance (JSONB array)
        sa.Column(
            "certification_status",
            postgresql.JSONB,
            nullable=False,
            server_default="[]",
            comment="Per-framework certification status",
        ),
        # Tags and frameworks (PostgreSQL arrays)
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.String),
            nullable=False,
            server_default="{}",
            comment="Searchable tags",
        ),
        sa.Column(
            "regulatory_frameworks",
            postgresql.ARRAY(sa.String),
            nullable=False,
            server_default="{}",
            comment="Applicable regulatory frameworks",
        ),
        # Additional metadata
        sa.Column(
            "documentation_url",
            sa.String(500),
            nullable=True,
            comment="Documentation URL",
        ),
        sa.Column(
            "repository_url",
            sa.String(500),
            nullable=True,
            comment="Source repository URL",
        ),
        sa.Column(
            "license",
            sa.String(50),
            nullable=False,
            server_default="Apache-2.0",
            comment="License identifier",
        ),
        # Check constraints
        sa.CheckConstraint(
            "status IN ('draft', 'published', 'deprecated')",
            name="ck_agent_records_status",
        ),
        sa.CheckConstraint(
            "downloads >= 0",
            name="ck_agent_records_downloads_positive",
        ),
        # Table comment
        comment="Registered agents in the Agent Registry",
    )

    # Create agent_versions table
    op.create_table(
        "agent_versions",
        # Primary key
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            comment="Version record identifier",
        ),
        # Foreign key to agent
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_records.id", ondelete="CASCADE"),
            nullable=False,
            comment="Parent agent ID",
        ),
        # Version info
        sa.Column(
            "version",
            sa.String(50),
            nullable=False,
            comment="Semantic version string",
        ),
        sa.Column(
            "changelog",
            sa.Text,
            nullable=False,
            server_default="",
            comment="Version changelog (markdown)",
        ),
        sa.Column(
            "breaking_changes",
            sa.Boolean,
            nullable=False,
            server_default="false",
            comment="Has breaking changes",
        ),
        sa.Column(
            "release_notes",
            sa.Text,
            nullable=False,
            server_default="",
            comment="Detailed release notes",
        ),
        # Artifact storage
        sa.Column(
            "artifact_path",
            sa.String(500),
            nullable=True,
            comment="Path or URL to version artifacts",
        ),
        sa.Column(
            "checksum",
            sa.String(128),
            nullable=False,
            server_default="",
            comment="SHA-256 checksum of artifacts",
        ),
        # Status flags
        sa.Column(
            "is_latest",
            sa.Boolean,
            nullable=False,
            server_default="false",
            comment="Is this the latest version",
        ),
        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Version creation timestamp",
        ),
        sa.Column(
            "published_at",
            sa.DateTime,
            nullable=True,
            comment="When version was published",
        ),
        sa.Column(
            "deprecated_at",
            sa.DateTime,
            nullable=True,
            comment="When version was deprecated",
        ),
        # Metrics
        sa.Column(
            "downloads",
            sa.Integer,
            nullable=False,
            server_default="0",
            comment="Version-specific download count",
        ),
        # Dependencies and compatibility
        sa.Column(
            "min_runtime_version",
            sa.String(50),
            nullable=True,
            comment="Minimum GreenLang runtime version",
        ),
        sa.Column(
            "dependencies",
            postgresql.JSONB,
            nullable=False,
            server_default="{}",
            comment="Agent dependencies with version constraints",
        ),
        # Pack configuration snapshot
        sa.Column(
            "pack_yaml_snapshot",
            postgresql.JSONB,
            nullable=True,
            comment="Pack.yaml at time of version creation",
        ),
        sa.Column(
            "generated_code_snapshot",
            postgresql.JSONB,
            nullable=True,
            comment="Generated code at time of version creation",
        ),
        # Unique constraint: agent + version
        sa.UniqueConstraint(
            "agent_id",
            "version",
            name="uq_agent_versions_agent_version",
        ),
        # Check constraint
        sa.CheckConstraint(
            "downloads >= 0",
            name="ck_agent_versions_downloads_positive",
        ),
        # Table comment
        comment="Agent version history and artifacts",
    )

    # Create update timestamp trigger function
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    # Create trigger for agent_records
    op.execute("""
        CREATE TRIGGER update_agent_records_updated_at
        BEFORE UPDATE ON agent_records
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    """
    Drop Agent Registry schema.

    Removes:
    - Triggers
    - Functions
    - Tables
    """
    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS update_agent_records_updated_at ON agent_records;")

    # Drop function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")

    # Drop tables
    op.drop_table("agent_versions")
    op.drop_table("agent_records")

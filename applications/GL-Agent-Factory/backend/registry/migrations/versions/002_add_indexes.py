"""
Add Search and Performance Indexes

Creates indexes for efficient querying:
- B-tree indexes for common lookups
- GIN indexes for array and JSONB fields
- Full-text search index
- Composite indexes for common query patterns

Revision ID: 002_add_indexes
Revises: 001_initial
Create Date: 2024-12-09
"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers
revision = "002_add_indexes"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create indexes for Agent Registry tables.

    Indexes created:
    - B-tree indexes for category, status, author, tenant
    - GIN indexes for tags and regulatory_frameworks arrays
    - GIN index for full-text search on name + description
    - Composite indexes for common query patterns
    """
    # ==========================================================================
    # agent_records indexes
    # ==========================================================================

    # Single column B-tree indexes (some already created by model)
    op.create_index(
        "ix_agent_records_name",
        "agent_records",
        ["name"],
        unique=True,
    )
    op.create_index(
        "ix_agent_records_category",
        "agent_records",
        ["category"],
    )
    op.create_index(
        "ix_agent_records_status",
        "agent_records",
        ["status"],
    )
    op.create_index(
        "ix_agent_records_author",
        "agent_records",
        ["author"],
    )
    op.create_index(
        "ix_agent_records_tenant_id",
        "agent_records",
        ["tenant_id"],
    )
    op.create_index(
        "ix_agent_records_created_at",
        "agent_records",
        ["created_at"],
    )
    op.create_index(
        "ix_agent_records_downloads",
        "agent_records",
        ["downloads"],
    )

    # Composite indexes for common query patterns
    op.create_index(
        "ix_agent_records_category_status",
        "agent_records",
        ["category", "status"],
    )
    op.create_index(
        "ix_agent_records_author_status",
        "agent_records",
        ["author", "status"],
    )
    op.create_index(
        "ix_agent_records_tenant_status",
        "agent_records",
        ["tenant_id", "status"],
    )
    op.create_index(
        "ix_agent_records_tenant_category",
        "agent_records",
        ["tenant_id", "category"],
    )
    op.create_index(
        "ix_agent_records_status_downloads",
        "agent_records",
        ["status", "downloads"],
    )

    # GIN indexes for array fields (PostgreSQL specific)
    op.execute("""
        CREATE INDEX ix_agent_records_tags
        ON agent_records USING gin(tags);
    """)
    op.execute("""
        CREATE INDEX ix_agent_records_frameworks
        ON agent_records USING gin(regulatory_frameworks);
    """)

    # Full-text search index on name and description
    op.execute("""
        CREATE INDEX ix_agent_records_search
        ON agent_records USING gin(
            to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, ''))
        );
    """)

    # GIN index for JSONB pack_yaml (for querying specific fields)
    op.execute("""
        CREATE INDEX ix_agent_records_pack_yaml
        ON agent_records USING gin(pack_yaml jsonb_path_ops);
    """)

    # ==========================================================================
    # agent_versions indexes
    # ==========================================================================

    # Single column indexes
    op.create_index(
        "ix_agent_versions_agent_id",
        "agent_versions",
        ["agent_id"],
    )
    op.create_index(
        "ix_agent_versions_version",
        "agent_versions",
        ["version"],
    )
    op.create_index(
        "ix_agent_versions_is_latest",
        "agent_versions",
        ["is_latest"],
    )
    op.create_index(
        "ix_agent_versions_created_at",
        "agent_versions",
        ["created_at"],
    )
    op.create_index(
        "ix_agent_versions_published_at",
        "agent_versions",
        ["published_at"],
    )

    # Composite indexes
    op.create_index(
        "ix_agent_versions_agent_latest",
        "agent_versions",
        ["agent_id", "is_latest"],
    )
    op.create_index(
        "ix_agent_versions_agent_created",
        "agent_versions",
        ["agent_id", "created_at"],
    )
    op.create_index(
        "ix_agent_versions_agent_version",
        "agent_versions",
        ["agent_id", "version"],
        unique=True,
    )

    # Partial index for latest versions only (efficient for "get latest" queries)
    op.execute("""
        CREATE INDEX ix_agent_versions_latest_only
        ON agent_versions (agent_id)
        WHERE is_latest = true;
    """)

    # Partial index for published versions only
    op.execute("""
        CREATE INDEX ix_agent_versions_published_only
        ON agent_versions (agent_id, version)
        WHERE published_at IS NOT NULL AND deprecated_at IS NULL;
    """)


def downgrade() -> None:
    """
    Remove all indexes.

    Note: This preserves the base tables and data.
    """
    # agent_versions indexes
    op.execute("DROP INDEX IF EXISTS ix_agent_versions_published_only;")
    op.execute("DROP INDEX IF EXISTS ix_agent_versions_latest_only;")
    op.drop_index("ix_agent_versions_agent_version", "agent_versions")
    op.drop_index("ix_agent_versions_agent_created", "agent_versions")
    op.drop_index("ix_agent_versions_agent_latest", "agent_versions")
    op.drop_index("ix_agent_versions_published_at", "agent_versions")
    op.drop_index("ix_agent_versions_created_at", "agent_versions")
    op.drop_index("ix_agent_versions_is_latest", "agent_versions")
    op.drop_index("ix_agent_versions_version", "agent_versions")
    op.drop_index("ix_agent_versions_agent_id", "agent_versions")

    # agent_records indexes
    op.execute("DROP INDEX IF EXISTS ix_agent_records_pack_yaml;")
    op.execute("DROP INDEX IF EXISTS ix_agent_records_search;")
    op.execute("DROP INDEX IF EXISTS ix_agent_records_frameworks;")
    op.execute("DROP INDEX IF EXISTS ix_agent_records_tags;")
    op.drop_index("ix_agent_records_status_downloads", "agent_records")
    op.drop_index("ix_agent_records_tenant_category", "agent_records")
    op.drop_index("ix_agent_records_tenant_status", "agent_records")
    op.drop_index("ix_agent_records_author_status", "agent_records")
    op.drop_index("ix_agent_records_category_status", "agent_records")
    op.drop_index("ix_agent_records_downloads", "agent_records")
    op.drop_index("ix_agent_records_created_at", "agent_records")
    op.drop_index("ix_agent_records_tenant_id", "agent_records")
    op.drop_index("ix_agent_records_author", "agent_records")
    op.drop_index("ix_agent_records_status", "agent_records")
    op.drop_index("ix_agent_records_category", "agent_records")
    op.drop_index("ix_agent_records_name", "agent_records")

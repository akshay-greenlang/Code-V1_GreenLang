"""
Add Row-Level Security for Multi-Tenancy

Implements PostgreSQL Row-Level Security (RLS) for tenant isolation:
- Enables RLS on agent_records and agent_versions
- Creates policies for CRUD operations
- Sets up tenant context functions

Revision ID: 003_add_rls
Revises: 002_add_indexes
Create Date: 2024-12-09
"""

from alembic import op

# Revision identifiers
revision = "003_add_rls"
down_revision = "002_add_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Enable Row-Level Security for multi-tenant isolation.

    Creates:
    - Tenant context setting functions
    - RLS policies for SELECT, INSERT, UPDATE, DELETE
    - Cascade policies for version table
    """
    # ==========================================================================
    # Create tenant context functions
    # ==========================================================================

    # Function to set current tenant
    op.execute("""
        CREATE OR REPLACE FUNCTION set_current_tenant(tenant uuid)
        RETURNS void AS $$
        BEGIN
            PERFORM set_config('app.current_tenant', tenant::text, false);
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)

    # Function to get current tenant
    op.execute("""
        CREATE OR REPLACE FUNCTION get_current_tenant()
        RETURNS uuid AS $$
        BEGIN
            RETURN NULLIF(current_setting('app.current_tenant', true), '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                RETURN NULL;
        END;
        $$ LANGUAGE plpgsql STABLE SECURITY DEFINER;
    """)

    # Function to check if user is admin (bypasses RLS)
    op.execute("""
        CREATE OR REPLACE FUNCTION is_admin_user()
        RETURNS boolean AS $$
        BEGIN
            RETURN COALESCE(
                NULLIF(current_setting('app.is_admin', true), '')::boolean,
                false
            );
        EXCEPTION
            WHEN OTHERS THEN
                RETURN false;
        END;
        $$ LANGUAGE plpgsql STABLE SECURITY DEFINER;
    """)

    # ==========================================================================
    # Enable RLS on agent_records
    # ==========================================================================

    op.execute("ALTER TABLE agent_records ENABLE ROW LEVEL SECURITY;")

    # Policy for SELECT - tenant can see own records or admin sees all
    op.execute("""
        CREATE POLICY agent_records_select_policy ON agent_records
        FOR SELECT
        USING (
            is_admin_user()
            OR tenant_id = get_current_tenant()
            OR tenant_id IS NULL  -- Public agents
        );
    """)

    # Policy for INSERT - tenant can only insert own records
    op.execute("""
        CREATE POLICY agent_records_insert_policy ON agent_records
        FOR INSERT
        WITH CHECK (
            is_admin_user()
            OR tenant_id = get_current_tenant()
        );
    """)

    # Policy for UPDATE - tenant can only update own records
    op.execute("""
        CREATE POLICY agent_records_update_policy ON agent_records
        FOR UPDATE
        USING (
            is_admin_user()
            OR tenant_id = get_current_tenant()
        )
        WITH CHECK (
            is_admin_user()
            OR tenant_id = get_current_tenant()
        );
    """)

    # Policy for DELETE - tenant can only delete own records
    op.execute("""
        CREATE POLICY agent_records_delete_policy ON agent_records
        FOR DELETE
        USING (
            is_admin_user()
            OR tenant_id = get_current_tenant()
        );
    """)

    # ==========================================================================
    # Enable RLS on agent_versions
    # ==========================================================================

    op.execute("ALTER TABLE agent_versions ENABLE ROW LEVEL SECURITY;")

    # Policy for SELECT - inherits from parent agent
    op.execute("""
        CREATE POLICY agent_versions_select_policy ON agent_versions
        FOR SELECT
        USING (
            is_admin_user()
            OR EXISTS (
                SELECT 1 FROM agent_records ar
                WHERE ar.id = agent_versions.agent_id
                AND (
                    ar.tenant_id = get_current_tenant()
                    OR ar.tenant_id IS NULL
                )
            )
        );
    """)

    # Policy for INSERT - inherits from parent agent
    op.execute("""
        CREATE POLICY agent_versions_insert_policy ON agent_versions
        FOR INSERT
        WITH CHECK (
            is_admin_user()
            OR EXISTS (
                SELECT 1 FROM agent_records ar
                WHERE ar.id = agent_versions.agent_id
                AND ar.tenant_id = get_current_tenant()
            )
        );
    """)

    # Policy for UPDATE - inherits from parent agent
    op.execute("""
        CREATE POLICY agent_versions_update_policy ON agent_versions
        FOR UPDATE
        USING (
            is_admin_user()
            OR EXISTS (
                SELECT 1 FROM agent_records ar
                WHERE ar.id = agent_versions.agent_id
                AND ar.tenant_id = get_current_tenant()
            )
        )
        WITH CHECK (
            is_admin_user()
            OR EXISTS (
                SELECT 1 FROM agent_records ar
                WHERE ar.id = agent_versions.agent_id
                AND ar.tenant_id = get_current_tenant()
            )
        );
    """)

    # Policy for DELETE - inherits from parent agent
    op.execute("""
        CREATE POLICY agent_versions_delete_policy ON agent_versions
        FOR DELETE
        USING (
            is_admin_user()
            OR EXISTS (
                SELECT 1 FROM agent_records ar
                WHERE ar.id = agent_versions.agent_id
                AND ar.tenant_id = get_current_tenant()
            )
        );
    """)

    # ==========================================================================
    # Create helper view for public agents
    # ==========================================================================

    op.execute("""
        CREATE OR REPLACE VIEW public_agents AS
        SELECT
            ar.id,
            ar.name,
            ar.version,
            ar.description,
            ar.category,
            ar.status,
            ar.author,
            ar.downloads,
            ar.tags,
            ar.regulatory_frameworks,
            ar.created_at,
            ar.updated_at
        FROM agent_records ar
        WHERE ar.status = 'published'
        AND (ar.tenant_id IS NULL OR EXISTS (
            SELECT 1 FROM agent_records ar2
            WHERE ar2.id = ar.id
            -- Add logic for shared agents
        ));
    """)


def downgrade() -> None:
    """
    Remove Row-Level Security.

    Disables RLS and drops all policies and functions.
    """
    # Drop view
    op.execute("DROP VIEW IF EXISTS public_agents;")

    # Drop agent_versions policies
    op.execute("DROP POLICY IF EXISTS agent_versions_delete_policy ON agent_versions;")
    op.execute("DROP POLICY IF EXISTS agent_versions_update_policy ON agent_versions;")
    op.execute("DROP POLICY IF EXISTS agent_versions_insert_policy ON agent_versions;")
    op.execute("DROP POLICY IF EXISTS agent_versions_select_policy ON agent_versions;")

    # Disable RLS on agent_versions
    op.execute("ALTER TABLE agent_versions DISABLE ROW LEVEL SECURITY;")

    # Drop agent_records policies
    op.execute("DROP POLICY IF EXISTS agent_records_delete_policy ON agent_records;")
    op.execute("DROP POLICY IF EXISTS agent_records_update_policy ON agent_records;")
    op.execute("DROP POLICY IF EXISTS agent_records_insert_policy ON agent_records;")
    op.execute("DROP POLICY IF EXISTS agent_records_select_policy ON agent_records;")

    # Disable RLS on agent_records
    op.execute("ALTER TABLE agent_records DISABLE ROW LEVEL SECURITY;")

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS is_admin_user();")
    op.execute("DROP FUNCTION IF EXISTS get_current_tenant();")
    op.execute("DROP FUNCTION IF EXISTS set_current_tenant(uuid);")

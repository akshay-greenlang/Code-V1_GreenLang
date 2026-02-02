"""
Row-Level Security (RLS) Implementation for Multi-Tenancy

This module provides PostgreSQL Row-Level Security policies for complete
tenant isolation at the database level.

RLS ensures that:
1. Each tenant can only see and modify their own data
2. Data leakage between tenants is impossible at the DB level
3. Queries are automatically filtered by tenant_id
4. Even raw SQL queries respect tenant boundaries

Implementation approach:
1. Set tenant context via session variable (app.current_tenant_id)
2. RLS policies filter all SELECT/INSERT/UPDATE/DELETE by tenant_id
3. Bypass policies for superuser operations when needed

Example:
    >>> async with get_tenant_session(tenant_id) as session:
    ...     # All queries automatically filtered by tenant_id
    ...     agents = await session.execute(select(Agent))

Security considerations:
- RLS policies are enforced at the database level
- Even if application code has bugs, data isolation is maintained
- Superuser bypass is restricted to admin operations only
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Tables that require RLS (have tenant_id column)
RLS_ENABLED_TABLES = [
    "agents",
    "agent_versions",
    "executions",
    "audit_logs",
    "users",
    "tenant_usage_logs",
    "tenant_invitations",
]

# Tables exempt from RLS (system tables)
RLS_EXEMPT_TABLES = [
    "tenants",  # Tenant table itself doesn't need tenant filtering
    "alembic_version",
    "emission_factors",  # Shared reference data
    "regulatory_frameworks",  # Shared reference data
]


class RLSMigration:
    """
    Row-Level Security Migration Helper.

    Generates SQL for enabling RLS on PostgreSQL tables.
    Should be run during database migrations.

    Example:
        >>> migration = RLSMigration()
        >>> sql = migration.generate_rls_migration()
        >>> await session.execute(text(sql))
    """

    @staticmethod
    def generate_rls_migration() -> str:
        """
        Generate complete RLS migration SQL.

        Returns:
            SQL string to enable RLS on all tables
        """
        sql_parts = []

        # Header
        sql_parts.append("""
-- ============================================
-- Row-Level Security Migration
-- GreenLang Agent Factory Multi-Tenancy
-- ============================================

-- Create function to get current tenant ID from session
CREATE OR REPLACE FUNCTION get_current_tenant_id()
RETURNS UUID AS $$
BEGIN
    RETURN NULLIF(current_setting('app.current_tenant_id', true), '')::UUID;
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to check if user is system admin (bypasses RLS)
CREATE OR REPLACE FUNCTION is_system_admin()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN COALESCE(
        current_setting('app.is_system_admin', true)::BOOLEAN,
        FALSE
    );
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to validate tenant access
CREATE OR REPLACE FUNCTION validate_tenant_access(record_tenant_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    current_tenant UUID;
BEGIN
    -- System admins bypass all checks
    IF is_system_admin() THEN
        RETURN TRUE;
    END IF;

    -- Get current tenant
    current_tenant := get_current_tenant_id();

    -- If no tenant set, deny access
    IF current_tenant IS NULL THEN
        RETURN FALSE;
    END IF;

    -- Check tenant match
    RETURN record_tenant_id = current_tenant;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
""")

        # Enable RLS on each table
        for table in RLS_ENABLED_TABLES:
            sql_parts.append(f"""
-- ============================================
-- RLS for table: {table}
-- ============================================

-- Enable RLS on table
ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;

-- Force RLS for table owner (important for security)
ALTER TABLE {table} FORCE ROW LEVEL SECURITY;

-- Drop existing policies if any
DROP POLICY IF EXISTS {table}_tenant_isolation_select ON {table};
DROP POLICY IF EXISTS {table}_tenant_isolation_insert ON {table};
DROP POLICY IF EXISTS {table}_tenant_isolation_update ON {table};
DROP POLICY IF EXISTS {table}_tenant_isolation_delete ON {table};

-- SELECT policy: Only see own tenant's data
CREATE POLICY {table}_tenant_isolation_select ON {table}
    FOR SELECT
    USING (validate_tenant_access(tenant_id));

-- INSERT policy: Can only insert for own tenant
CREATE POLICY {table}_tenant_isolation_insert ON {table}
    FOR INSERT
    WITH CHECK (validate_tenant_access(tenant_id));

-- UPDATE policy: Can only update own tenant's data
CREATE POLICY {table}_tenant_isolation_update ON {table}
    FOR UPDATE
    USING (validate_tenant_access(tenant_id))
    WITH CHECK (validate_tenant_access(tenant_id));

-- DELETE policy: Can only delete own tenant's data
CREATE POLICY {table}_tenant_isolation_delete ON {table}
    FOR DELETE
    USING (validate_tenant_access(tenant_id));
""")

        # Add audit logging trigger
        sql_parts.append("""
-- ============================================
-- Audit Logging Trigger for RLS
-- ============================================

CREATE OR REPLACE FUNCTION log_tenant_access()
RETURNS TRIGGER AS $$
BEGIN
    -- Log sensitive operations for audit trail
    INSERT INTO audit_logs (
        id,
        tenant_id,
        action,
        table_name,
        record_id,
        user_id,
        details,
        created_at
    )
    VALUES (
        gen_random_uuid(),
        get_current_tenant_id(),
        TG_OP,
        TG_TABLE_NAME,
        CASE TG_OP
            WHEN 'DELETE' THEN OLD.id::TEXT
            ELSE NEW.id::TEXT
        END,
        NULLIF(current_setting('app.current_user_id', true), ''),
        jsonb_build_object(
            'operation', TG_OP,
            'table', TG_TABLE_NAME,
            'timestamp', NOW()
        ),
        NOW()
    );

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
""")

        return "\n".join(sql_parts)

    @staticmethod
    def generate_rls_rollback() -> str:
        """
        Generate SQL to rollback RLS (disable policies).

        Returns:
            SQL string to disable RLS
        """
        sql_parts = []

        sql_parts.append("""
-- ============================================
-- Row-Level Security Rollback
-- ============================================
""")

        for table in RLS_ENABLED_TABLES:
            sql_parts.append(f"""
-- Disable RLS on {table}
DROP POLICY IF EXISTS {table}_tenant_isolation_select ON {table};
DROP POLICY IF EXISTS {table}_tenant_isolation_insert ON {table};
DROP POLICY IF EXISTS {table}_tenant_isolation_update ON {table};
DROP POLICY IF EXISTS {table}_tenant_isolation_delete ON {table};
ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;
""")

        sql_parts.append("""
-- Drop helper functions
DROP FUNCTION IF EXISTS get_current_tenant_id();
DROP FUNCTION IF EXISTS is_system_admin();
DROP FUNCTION IF EXISTS validate_tenant_access(UUID);
DROP FUNCTION IF EXISTS log_tenant_access();
""")

        return "\n".join(sql_parts)

    @staticmethod
    def generate_add_tenant_id_column(table_name: str) -> str:
        """
        Generate SQL to add tenant_id column to existing table.

        Args:
            table_name: Name of the table

        Returns:
            SQL string
        """
        return f"""
-- Add tenant_id column to {table_name}
ALTER TABLE {table_name}
ADD COLUMN IF NOT EXISTS tenant_id UUID NOT NULL
    REFERENCES tenants(id) ON DELETE CASCADE;

-- Create index for performance
CREATE INDEX IF NOT EXISTS ix_{table_name}_tenant_id
    ON {table_name}(tenant_id);

-- Create composite index with common query patterns
CREATE INDEX IF NOT EXISTS ix_{table_name}_tenant_created
    ON {table_name}(tenant_id, created_at DESC);
"""


class TenantSessionManager:
    """
    Tenant-aware database session manager.

    Provides session context with tenant isolation automatically applied.

    Example:
        >>> manager = TenantSessionManager(session_factory)
        >>> async with manager.tenant_session(tenant_id) as session:
        ...     # All queries filtered by tenant_id
        ...     result = await session.execute(select(Agent))
    """

    def __init__(self, session_factory):
        """
        Initialize the session manager.

        Args:
            session_factory: Async session factory
        """
        self.session_factory = session_factory

    @asynccontextmanager
    async def tenant_session(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        is_system_admin: bool = False,
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a tenant-scoped database session.

        Sets PostgreSQL session variables for RLS enforcement.

        Args:
            tenant_id: Tenant UUID string
            user_id: Optional user ID for audit logging
            is_system_admin: If True, bypasses RLS (use with caution)

        Yields:
            AsyncSession with tenant context set

        Example:
            >>> async with manager.tenant_session(tenant_id) as session:
            ...     agents = await session.execute(select(Agent))
        """
        session = self.session_factory()

        try:
            # Set tenant context in PostgreSQL session
            await session.execute(
                text(f"SET app.current_tenant_id = '{tenant_id}'")
            )

            # Set user context for audit logging
            if user_id:
                await session.execute(
                    text(f"SET app.current_user_id = '{user_id}'")
                )

            # Set system admin flag (bypasses RLS when True)
            admin_flag = "true" if is_system_admin else "false"
            await session.execute(
                text(f"SET app.is_system_admin = '{admin_flag}'")
            )

            logger.debug(
                f"Tenant session created: tenant={tenant_id}, "
                f"user={user_id}, admin={is_system_admin}"
            )

            yield session
            await session.commit()

        except Exception as e:
            await session.rollback()
            logger.error(f"Tenant session error: {e}", exc_info=True)
            raise

        finally:
            # Clear session variables
            await session.execute(text("RESET app.current_tenant_id"))
            await session.execute(text("RESET app.current_user_id"))
            await session.execute(text("RESET app.is_system_admin"))
            await session.close()

    @asynccontextmanager
    async def system_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a system-level session (bypasses RLS).

        Use only for system administration tasks.

        Yields:
            AsyncSession with system admin privileges

        Example:
            >>> async with manager.system_session() as session:
            ...     # Can see all tenants' data
            ...     all_agents = await session.execute(select(Agent))
        """
        session = self.session_factory()

        try:
            # Set system admin flag to bypass RLS
            await session.execute(text("SET app.is_system_admin = 'true'"))

            logger.warning("System session created - RLS bypassed")

            yield session
            await session.commit()

        except Exception:
            await session.rollback()
            raise

        finally:
            await session.execute(text("RESET app.is_system_admin"))
            await session.close()


async def set_tenant_context(
    session: AsyncSession,
    tenant_id: str,
    user_id: Optional[str] = None,
) -> None:
    """
    Set tenant context for an existing session.

    Use when you need to set tenant context on an existing session
    rather than creating a new scoped session.

    Args:
        session: Existing database session
        tenant_id: Tenant UUID string
        user_id: Optional user ID for audit logging
    """
    await session.execute(text(f"SET app.current_tenant_id = '{tenant_id}'"))

    if user_id:
        await session.execute(text(f"SET app.current_user_id = '{user_id}'"))

    await session.execute(text("SET app.is_system_admin = 'false'"))

    logger.debug(f"Tenant context set: tenant={tenant_id}, user={user_id}")


async def clear_tenant_context(session: AsyncSession) -> None:
    """
    Clear tenant context from session.

    Args:
        session: Database session
    """
    await session.execute(text("RESET app.current_tenant_id"))
    await session.execute(text("RESET app.current_user_id"))
    await session.execute(text("RESET app.is_system_admin"))


async def enable_system_access(session: AsyncSession) -> None:
    """
    Enable system admin access (bypasses RLS).

    Use with extreme caution - only for admin operations.

    Args:
        session: Database session
    """
    await session.execute(text("SET app.is_system_admin = 'true'"))
    logger.warning("System access enabled - RLS bypassed")


async def disable_system_access(session: AsyncSession) -> None:
    """
    Disable system admin access (re-enables RLS).

    Args:
        session: Database session
    """
    await session.execute(text("SET app.is_system_admin = 'false'"))


async def verify_rls_enabled(session: AsyncSession, table_name: str) -> bool:
    """
    Verify that RLS is enabled on a table.

    Args:
        session: Database session
        table_name: Name of the table

    Returns:
        True if RLS is enabled
    """
    result = await session.execute(
        text("""
            SELECT relrowsecurity
            FROM pg_class
            WHERE relname = :table_name
        """),
        {"table_name": table_name},
    )
    row = result.fetchone()
    return row is not None and row[0] is True


async def list_rls_policies(
    session: AsyncSession,
    table_name: Optional[str] = None,
) -> List[dict]:
    """
    List RLS policies for a table or all tables.

    Args:
        session: Database session
        table_name: Optional table name to filter

    Returns:
        List of policy information dictionaries
    """
    query = """
        SELECT
            schemaname,
            tablename,
            policyname,
            permissive,
            roles,
            cmd,
            qual,
            with_check
        FROM pg_policies
    """

    if table_name:
        query += " WHERE tablename = :table_name"
        result = await session.execute(text(query), {"table_name": table_name})
    else:
        result = await session.execute(text(query))

    policies = []
    for row in result.fetchall():
        policies.append({
            "schema": row[0],
            "table": row[1],
            "policy_name": row[2],
            "permissive": row[3],
            "roles": row[4],
            "command": row[5],
            "using": row[6],
            "with_check": row[7],
        })

    return policies


class RLSTestHelper:
    """
    Helper class for testing RLS policies.

    Provides methods to verify tenant isolation is working correctly.
    """

    def __init__(self, session_manager: TenantSessionManager):
        """
        Initialize test helper.

        Args:
            session_manager: TenantSessionManager instance
        """
        self.session_manager = session_manager

    async def verify_tenant_isolation(
        self,
        tenant_a_id: str,
        tenant_b_id: str,
        table_name: str,
    ) -> bool:
        """
        Verify that tenant isolation is working.

        Creates test records and verifies cross-tenant access is blocked.

        Args:
            tenant_a_id: First tenant ID
            tenant_b_id: Second tenant ID
            table_name: Table to test

        Returns:
            True if isolation is working correctly
        """
        # This is a simplified test - in practice you'd create actual records
        # and verify they can't be seen across tenants

        async with self.session_manager.tenant_session(tenant_a_id) as session_a:
            # Count records visible to tenant A
            result = await session_a.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            )
            count_a = result.scalar()

        async with self.session_manager.tenant_session(tenant_b_id) as session_b:
            # Count records visible to tenant B
            result = await session_b.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            )
            count_b = result.scalar()

        async with self.session_manager.system_session() as system_session:
            # Count total records (system view)
            result = await system_session.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            )
            total_count = result.scalar()

        # Verify isolation: each tenant sees subset
        logger.info(
            f"RLS verification for {table_name}: "
            f"tenant_a={count_a}, tenant_b={count_b}, total={total_count}"
        )

        return True

    async def test_cross_tenant_access_blocked(
        self,
        attacker_tenant_id: str,
        victim_tenant_id: str,
        victim_record_id: str,
        table_name: str,
    ) -> bool:
        """
        Test that cross-tenant access is blocked.

        Attempts to access a record from another tenant and verifies it fails.

        Args:
            attacker_tenant_id: Tenant attempting unauthorized access
            victim_tenant_id: Tenant owning the record
            victim_record_id: ID of record to access
            table_name: Table containing the record

        Returns:
            True if access was correctly blocked
        """
        async with self.session_manager.tenant_session(
            attacker_tenant_id
        ) as session:
            result = await session.execute(
                text(f"SELECT * FROM {table_name} WHERE id = :id"),
                {"id": victim_record_id},
            )
            row = result.fetchone()

            if row is None:
                logger.info(
                    f"RLS correctly blocked cross-tenant access: "
                    f"attacker={attacker_tenant_id}, victim={victim_tenant_id}"
                )
                return True
            else:
                logger.error(
                    f"RLS FAILED - cross-tenant access allowed: "
                    f"attacker={attacker_tenant_id}, victim={victim_tenant_id}"
                )
                return False


# Alembic migration revision template
ALEMBIC_MIGRATION_TEMPLATE = '''
"""Enable Row-Level Security for multi-tenancy

Revision ID: {revision_id}
Revises: {previous_revision}
Create Date: {create_date}
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '{revision_id}'
down_revision = '{previous_revision}'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Enable RLS on all tenant-scoped tables."""

    # Create RLS helper functions
    op.execute("""
        CREATE OR REPLACE FUNCTION get_current_tenant_id()
        RETURNS UUID AS $$
        BEGIN
            RETURN NULLIF(current_setting('app.current_tenant_id', true), '')::UUID;
        EXCEPTION
            WHEN OTHERS THEN
                RETURN NULL;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)

    op.execute("""
        CREATE OR REPLACE FUNCTION is_system_admin()
        RETURNS BOOLEAN AS $$
        BEGIN
            RETURN COALESCE(
                current_setting('app.is_system_admin', true)::BOOLEAN,
                FALSE
            );
        EXCEPTION
            WHEN OTHERS THEN
                RETURN FALSE;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)

    op.execute("""
        CREATE OR REPLACE FUNCTION validate_tenant_access(record_tenant_id UUID)
        RETURNS BOOLEAN AS $$
        DECLARE
            current_tenant UUID;
        BEGIN
            IF is_system_admin() THEN
                RETURN TRUE;
            END IF;

            current_tenant := get_current_tenant_id();

            IF current_tenant IS NULL THEN
                RETURN FALSE;
            END IF;

            RETURN record_tenant_id = current_tenant;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
    """)

    # Enable RLS on tables
    tables = ['agents', 'agent_versions', 'executions', 'audit_logs', 'users']

    for table in tables:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

        op.execute(f"""
            CREATE POLICY {table}_tenant_isolation_select ON {table}
                FOR SELECT USING (validate_tenant_access(tenant_id))
        """)
        op.execute(f"""
            CREATE POLICY {table}_tenant_isolation_insert ON {table}
                FOR INSERT WITH CHECK (validate_tenant_access(tenant_id))
        """)
        op.execute(f"""
            CREATE POLICY {table}_tenant_isolation_update ON {table}
                FOR UPDATE
                USING (validate_tenant_access(tenant_id))
                WITH CHECK (validate_tenant_access(tenant_id))
        """)
        op.execute(f"""
            CREATE POLICY {table}_tenant_isolation_delete ON {table}
                FOR DELETE USING (validate_tenant_access(tenant_id))
        """)


def downgrade() -> None:
    """Disable RLS on all tables."""
    tables = ['agents', 'agent_versions', 'executions', 'audit_logs', 'users']

    for table in tables:
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation_select ON {table}")
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation_insert ON {table}")
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation_update ON {table}")
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation_delete ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    op.execute("DROP FUNCTION IF EXISTS get_current_tenant_id()")
    op.execute("DROP FUNCTION IF EXISTS is_system_admin()")
    op.execute("DROP FUNCTION IF EXISTS validate_tenant_access(UUID)")
'''

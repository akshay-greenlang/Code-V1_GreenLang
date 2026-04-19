"""Initial schema creation for GreenLang Auth PostgreSQL backend

Revision ID: 001
Revises:
Create Date: 2025-11-21 12:00:00.000000

This migration creates the initial database schema for the GreenLang authentication
system including permissions, roles, policies, and audit logging tables with proper
indexes and constraints for production use.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""

    # Create permissions table
    op.create_table('permissions',
        sa.Column('permission_id', sa.String(64), nullable=False),
        sa.Column('resource', sa.String(255), nullable=False),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('effect', sa.String(10), nullable=False, server_default='allow'),
        sa.Column('scope', sa.String(255), nullable=True),
        sa.Column('conditions', postgresql.JSONB, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('provenance_hash', sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint('permission_id'),
        sa.CheckConstraint("effect IN ('allow', 'deny')", name='check_effect')
    )

    # Create indexes for permissions
    op.create_index('idx_resource_action', 'permissions', ['resource', 'action'])
    op.create_index('idx_effect_scope', 'permissions', ['effect', 'scope'])
    op.create_index('idx_created_at', 'permissions', ['created_at'])
    op.create_index('idx_provenance', 'permissions', ['provenance_hash'])
    op.create_index('idx_permission_resource', 'permissions', ['resource'])
    op.create_index('idx_permission_action', 'permissions', ['action'])
    op.create_index('idx_permission_scope', 'permissions', ['scope'])

    # Create roles table
    op.create_table('roles',
        sa.Column('role_id', sa.String(64), nullable=False),
        sa.Column('role_name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('permissions', postgresql.JSONB, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('is_system_role', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='0'),
        sa.PrimaryKeyConstraint('role_id'),
        sa.UniqueConstraint('role_name', name='uq_role_name')
    )

    # Create indexes for roles
    op.create_index('idx_role_name', 'roles', ['role_name'])
    op.create_index('idx_role_priority', 'roles', ['priority'])
    op.create_index('idx_system_role', 'roles', ['is_system_role'])

    # Create policies table
    op.create_table('policies',
        sa.Column('policy_id', sa.String(64), nullable=False),
        sa.Column('policy_name', sa.String(100), nullable=False),
        sa.Column('policy_type', sa.String(50), nullable=False),
        sa.Column('rules', postgresql.JSONB, nullable=False),
        sa.Column('conditions', postgresql.JSONB, nullable=True),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, onupdate=sa.func.now()),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('policy_id'),
        sa.UniqueConstraint('policy_name', name='uq_policy_name')
    )

    # Create indexes for policies
    op.create_index('idx_policy_name', 'policies', ['policy_name'])
    op.create_index('idx_policy_type', 'policies', ['policy_type'])
    op.create_index('idx_policy_enabled', 'policies', ['enabled'])
    op.create_index('idx_policy_priority', 'policies', ['priority'])
    op.create_index('idx_policy_expires', 'policies', ['expires_at'])

    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('log_id', sa.String(64), nullable=False, server_default=sa.text('gen_random_uuid()::text')),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('resource', sa.String(255), nullable=True),
        sa.Column('action', sa.String(100), nullable=True),
        sa.Column('result', sa.String(20), nullable=False),
        sa.Column('details', postgresql.JSONB, nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('correlation_id', sa.String(64), nullable=True),
        sa.Column('provenance_hash', sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint('log_id'),
        sa.CheckConstraint("result IN ('success', 'failure', 'error')", name='check_result')
    )

    # Create indexes for audit_logs
    op.create_index('idx_audit_timestamp', 'audit_logs', ['timestamp'])
    op.create_index('idx_audit_user', 'audit_logs', ['user_id'])
    op.create_index('idx_audit_event_type', 'audit_logs', ['event_type'])
    op.create_index('idx_audit_session', 'audit_logs', ['session_id'])
    op.create_index('idx_audit_correlation', 'audit_logs', ['correlation_id'])
    op.create_index('idx_audit_result', 'audit_logs', ['result'])

    # Create role_permissions association table
    op.create_table('role_permissions',
        sa.Column('role_id', sa.String(64), nullable=False),
        sa.Column('permission_id', sa.String(64), nullable=False),
        sa.Column('granted_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('granted_by', sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(['permission_id'], ['permissions.permission_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['role_id'], ['roles.role_id'], ondelete='CASCADE'),
        sa.UniqueConstraint('role_id', 'permission_id', name='uq_role_permission')
    )

    # Create user_roles association table
    op.create_table('user_roles',
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('role_id', sa.String(64), nullable=False),
        sa.Column('assigned_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('assigned_by', sa.String(255), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['role_id'], ['roles.role_id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'role_id', name='uq_user_role')
    )

    # Create indexes for association tables
    op.create_index('idx_user_roles_user', 'user_roles', ['user_id'])
    op.create_index('idx_user_roles_expires', 'user_roles', ['expires_at'])

    # Create default system roles
    op.execute("""
        INSERT INTO roles (role_id, role_name, description, is_system_role, priority, created_by)
        VALUES
            ('sys-admin', 'System Administrator', 'Full system access', true, 100, 'system'),
            ('sys-operator', 'System Operator', 'Operational access', true, 80, 'system'),
            ('sys-viewer', 'System Viewer', 'Read-only access', true, 50, 'system'),
            ('sys-auditor', 'System Auditor', 'Audit log access', true, 60, 'system');
    """)

    # Create default admin permissions
    op.execute("""
        INSERT INTO permissions (permission_id, resource, action, effect, created_by)
        VALUES
            ('perm-admin-all', '*', '*', 'allow', 'system'),
            ('perm-operator-exec', 'agent:*', 'execute', 'allow', 'system'),
            ('perm-operator-read', '*', 'read', 'allow', 'system'),
            ('perm-viewer-read', '*', 'read', 'allow', 'system'),
            ('perm-auditor-logs', 'audit:*', '*', 'allow', 'system');
    """)

    # Assign permissions to system roles
    op.execute("""
        INSERT INTO role_permissions (role_id, permission_id, granted_by)
        VALUES
            ('sys-admin', 'perm-admin-all', 'system'),
            ('sys-operator', 'perm-operator-exec', 'system'),
            ('sys-operator', 'perm-operator-read', 'system'),
            ('sys-viewer', 'perm-viewer-read', 'system'),
            ('sys-auditor', 'perm-auditor-logs', 'system');
    """)


def downgrade() -> None:
    """Drop all tables in reverse order."""

    # Drop association tables first
    op.drop_table('user_roles')
    op.drop_table('role_permissions')

    # Drop main tables
    op.drop_table('audit_logs')
    op.drop_table('policies')
    op.drop_table('roles')
    op.drop_table('permissions')
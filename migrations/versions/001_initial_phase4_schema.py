"""Initial Phase 4 schema - Authentication and Authorization

Revision ID: 001_phase4_init
Revises:
Create Date: 2025-11-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_phase4_init'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('username', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=True),
        sa.Column('first_name', sa.String(255), nullable=True),
        sa.Column('last_name', sa.String(255), nullable=True),
        sa.Column('display_name', sa.String(255), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('email_verified', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('locked', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('locked_at', sa.DateTime(), nullable=True),
        sa.Column('locked_reason', sa.Text(), nullable=True),
        sa.Column('mfa_enabled', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('mfa_secret', sa.String(255), nullable=True),
        sa.Column('mfa_backup_codes', sa.Text(), nullable=True),
        sa.Column('sso_provider', sa.String(50), nullable=True),
        sa.Column('sso_provider_id', sa.String(36), nullable=True),
        sa.Column('sso_external_id', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
        sa.Column('password_changed_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'username', name='uq_user_tenant_username'),
        sa.UniqueConstraint('tenant_id', 'email', name='uq_user_tenant_email')
    )
    op.create_index('idx_user_email', 'users', ['email'])
    op.create_index('idx_user_sso', 'users', ['sso_provider', 'sso_external_id'])
    op.create_index('ix_users_tenant_id', 'users', ['tenant_id'])

    # Create roles table
    op.create_table(
        'roles',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('parent_role_id', sa.String(36), nullable=True),
        sa.Column('is_system_role', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('is_default_role', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['parent_role_id'], ['roles.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'name', name='uq_role_tenant_name')
    )
    op.create_index('idx_role_parent', 'roles', ['parent_role_id'])
    op.create_index('ix_roles_tenant_id', 'roles', ['tenant_id'])

    # Create permissions table
    op.create_table(
        'permissions',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('role_id', sa.String(36), nullable=False),
        sa.Column('resource_type', sa.String(255), nullable=False),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('action', sa.String(255), nullable=False),
        sa.Column('scope', sa.String(255), nullable=True),
        sa.Column('conditions', sa.JSON(), nullable=True),
        sa.Column('effect', sa.String(10), nullable=False, server_default='allow'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_permission_role', 'permissions', ['role_id'])
    op.create_index('idx_permission_resource', 'permissions', ['resource_type', 'resource_id'])

    # Create user_roles table
    op.create_table(
        'user_roles',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('role_id', sa.String(36), nullable=False),
        sa.Column('assigned_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('assigned_by', sa.String(36), nullable=True),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'role_id', name='uq_user_role')
    )
    op.create_index('idx_user_role_role', 'user_roles', ['role_id'])
    op.create_index('idx_user_role_user', 'user_roles', ['user_id'])

    # Create sessions table
    op.create_table(
        'sessions',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('session_token', sa.String(255), nullable=False),
        sa.Column('refresh_token', sa.String(255), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('device_id', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(), nullable=False),
        sa.Column('active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('revoked', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('revoke_reason', sa.String(255), nullable=True),
        sa.Column('redis_key', sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_session_active', 'sessions', ['active', 'revoked'])
    op.create_index('idx_session_expiry', 'sessions', ['expires_at'])
    op.create_index('idx_session_user', 'sessions', ['user_id'])
    op.create_index('ix_sessions_session_token', 'sessions', ['session_token'], unique=True)
    op.create_index('ix_sessions_tenant_id', 'sessions', ['tenant_id'])

    # Create api_keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=True),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('key_id', sa.String(32), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('key_prefix', sa.String(10), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('scopes', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('last_rotated_at', sa.DateTime(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('revoked', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('use_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rate_limit', sa.Integer(), nullable=True),
        sa.Column('allowed_ips', sa.JSON(), nullable=True),
        sa.Column('allowed_origins', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_apikey_active', 'api_keys', ['active', 'revoked'])
    op.create_index('idx_apikey_user', 'api_keys', ['user_id'])
    op.create_index('ix_api_keys_key_id', 'api_keys', ['key_id'], unique=True)
    op.create_index('ix_api_keys_tenant_id', 'api_keys', ['tenant_id'])

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=True),
        sa.Column('event_type', sa.String(255), nullable=False),
        sa.Column('event_category', sa.String(50), nullable=False),
        sa.Column('resource_type', sa.String(255), nullable=True),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('action', sa.String(255), nullable=True),
        sa.Column('result', sa.String(50), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(36), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_event', 'audit_logs', ['event_type', 'event_category'])
    op.create_index('idx_audit_resource', 'audit_logs', ['resource_type', 'resource_id'])
    op.create_index('idx_audit_result', 'audit_logs', ['result'])
    op.create_index('idx_audit_timestamp', 'audit_logs', ['timestamp'])
    op.create_index('idx_audit_user', 'audit_logs', ['user_id'])
    op.create_index('ix_audit_logs_tenant_id', 'audit_logs', ['tenant_id'])

    # Create saml_providers table
    op.create_table(
        'saml_providers',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('entity_id', sa.String(255), nullable=False),
        sa.Column('sso_url', sa.String(255), nullable=False),
        sa.Column('slo_url', sa.String(255), nullable=True),
        sa.Column('x509_cert', sa.Text(), nullable=False),
        sa.Column('metadata_url', sa.String(255), nullable=True),
        sa.Column('attribute_mapping', sa.JSON(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'entity_id', name='uq_saml_tenant_entity')
    )
    op.create_index('ix_saml_providers_tenant_id', 'saml_providers', ['tenant_id'])

    # Create oauth_providers table
    op.create_table(
        'oauth_providers',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('provider_type', sa.String(50), nullable=False),
        sa.Column('client_id', sa.String(255), nullable=False),
        sa.Column('client_secret', sa.String(255), nullable=False),
        sa.Column('authorization_url', sa.String(255), nullable=False),
        sa.Column('token_url', sa.String(255), nullable=False),
        sa.Column('userinfo_url', sa.String(255), nullable=True),
        sa.Column('scopes', sa.JSON(), nullable=True),
        sa.Column('attribute_mapping', sa.JSON(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'provider_type', 'name', name='uq_oauth_tenant_provider')
    )
    op.create_index('ix_oauth_providers_tenant_id', 'oauth_providers', ['tenant_id'])

    # Create ldap_configs table
    op.create_table(
        'ldap_configs',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('server_url', sa.String(255), nullable=False),
        sa.Column('port', sa.Integer(), nullable=False, server_default='389'),
        sa.Column('use_ssl', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('use_tls', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('bind_dn', sa.String(255), nullable=True),
        sa.Column('bind_password', sa.String(255), nullable=True),
        sa.Column('base_dn', sa.String(255), nullable=False),
        sa.Column('user_search_base', sa.String(255), nullable=True),
        sa.Column('user_search_filter', sa.String(255), nullable=True),
        sa.Column('group_search_base', sa.String(255), nullable=True),
        sa.Column('group_search_filter', sa.String(255), nullable=True),
        sa.Column('attribute_mapping', sa.JSON(), nullable=True),
        sa.Column('group_role_mapping', sa.JSON(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ldap_configs_tenant_id', 'ldap_configs', ['tenant_id'], unique=True)


def downgrade() -> None:
    op.drop_table('ldap_configs')
    op.drop_table('oauth_providers')
    op.drop_table('saml_providers')
    op.drop_table('audit_logs')
    op.drop_table('api_keys')
    op.drop_table('sessions')
    op.drop_table('user_roles')
    op.drop_table('permissions')
    op.drop_table('roles')
    op.drop_table('users')

"""Initial schema - Create all GL-Agent-Factory tables

Revision ID: 20241208_130000_001
Revises: None
Create Date: 2024-12-08 13:00:00 UTC

Creates the complete database schema for GL-Agent-Factory:
- tenants: Multi-tenant organization management
- users: User accounts with tenant association
- agents: Registered AI agents
- agent_versions: Agent version history
- executions: Agent execution records
- audit_logs: Compliance audit trail
- tenant_usage_logs: Usage tracking for billing
- tenant_invitations: User invitation management

Features:
- UUID primary keys for security
- JSONB columns for flexible metadata
- GIN indexes for JSONB query performance
- Foreign key constraints with proper cascades
- Composite indexes for common query patterns
- PostgreSQL ENUM types for status fields
- Timestamps with timezone awareness
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers, used by Alembic.
revision: str = '20241208_130000_001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create all GL-Agent-Factory database tables.

    Tables are created in dependency order:
    1. tenants (no dependencies)
    2. users (depends on tenants)
    3. agents (depends on tenants)
    4. agent_versions (depends on agents)
    5. executions (depends on agents, tenants, users)
    6. audit_logs (depends on tenants)
    7. tenant_usage_logs (depends on tenants)
    8. tenant_invitations (depends on tenants)
    """

    # ========================================
    # Create ENUM types
    # ========================================

    # Subscription tier enum
    subscription_tier_enum = postgresql.ENUM(
        'free', 'pro', 'enterprise',
        name='subscription_tier',
        create_type=True
    )
    subscription_tier_enum.create(op.get_bind(), checkfirst=True)

    # Tenant status enum
    tenant_status_enum = postgresql.ENUM(
        'pending', 'active', 'suspended', 'deactivated',
        name='tenant_status',
        create_type=True
    )
    tenant_status_enum.create(op.get_bind(), checkfirst=True)

    # Agent state enum
    agent_state_enum = postgresql.ENUM(
        'DRAFT', 'EXPERIMENTAL', 'CERTIFIED', 'DEPRECATED', 'RETIRED',
        name='agent_state',
        create_type=True
    )
    agent_state_enum.create(op.get_bind(), checkfirst=True)

    # Execution status enum
    execution_status_enum = postgresql.ENUM(
        'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT',
        name='execution_status',
        create_type=True
    )
    execution_status_enum.create(op.get_bind(), checkfirst=True)

    # ========================================
    # Create tenants table
    # ========================================
    op.create_table(
        'tenants',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()'),
                  comment='Primary key (UUID)'),

        # Tenant identifier
        sa.Column('tenant_id', sa.String(100), nullable=False, unique=True,
                  comment='External tenant identifier (e.g., t-acme-corp)'),

        # Basic information
        sa.Column('name', sa.String(255), nullable=False,
                  comment='Organization name'),
        sa.Column('slug', sa.String(100), nullable=False, unique=True,
                  comment='URL-safe slug for subdomains'),
        sa.Column('domain', sa.String(255), nullable=True, unique=True,
                  comment='Custom domain (e.g., acme.greenlang.io)'),

        # Status and tier
        sa.Column('status', sa.Enum('pending', 'active', 'suspended', 'deactivated',
                                     name='tenant_status', create_type=False),
                  nullable=False, server_default='pending',
                  comment='Account status'),
        sa.Column('subscription_tier', sa.Enum('free', 'pro', 'enterprise',
                                                name='subscription_tier', create_type=False),
                  nullable=False, server_default='free',
                  comment='Subscription tier'),

        # Activation tracking
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true',
                  comment='Quick active check (derived from status)'),
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When tenant was activated'),
        sa.Column('suspended_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When tenant was suspended (if applicable)'),
        sa.Column('suspension_reason', sa.Text(), nullable=True,
                  comment='Reason for suspension'),

        # Settings (tenant-specific configuration)
        sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default='{}',
                  comment='Tenant-specific settings'),

        # Quotas and usage
        sa.Column('quotas', postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default='{}',
                  comment='Resource quotas (merged with tier defaults)'),
        sa.Column('current_usage', postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default='{}',
                  comment='Current resource usage'),
        sa.Column('usage_reset_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When monthly usage counters were last reset'),

        # Feature flags
        sa.Column('feature_flags', postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default='{}',
                  comment='Feature flag overrides (merged with tier defaults)'),

        # Billing information
        sa.Column('billing_info', postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default='{}',
                  comment='Billing and payment information'),
        sa.Column('billing_email', sa.String(255), nullable=True,
                  comment='Billing email address'),
        sa.Column('stripe_customer_id', sa.String(255), nullable=True, unique=True,
                  comment='Stripe customer ID for billing'),
        sa.Column('stripe_subscription_id', sa.String(255), nullable=True, unique=True,
                  comment='Stripe subscription ID'),

        # Trial tracking
        sa.Column('trial_ends_at', sa.DateTime(timezone=True), nullable=True,
                  comment='Trial period end date'),
        sa.Column('is_trial', sa.Boolean(), nullable=False, server_default='false',
                  comment='Whether tenant is in trial period'),

        # Contact information
        sa.Column('primary_contact_name', sa.String(255), nullable=True,
                  comment='Primary contact name'),
        sa.Column('primary_contact_email', sa.String(255), nullable=True,
                  comment='Primary contact email'),

        # Compliance and legal
        sa.Column('data_residency_region', sa.String(50), nullable=True,
                  server_default='us-east-1',
                  comment='Data residency region for compliance'),
        sa.Column('accepted_terms_version', sa.String(50), nullable=True,
                  comment='Version of terms of service accepted'),
        sa.Column('accepted_terms_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When terms were accepted'),
        sa.Column('dpa_signed', sa.Boolean(), nullable=False, server_default='false',
                  comment='Data Processing Agreement signed'),

        # Metadata
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default='{}',
                  comment='Additional metadata'),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default='[]',
                  comment='Tags for categorization'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Creation timestamp'),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Last update timestamp'),

        comment='Multi-tenant organizations'
    )

    # Tenants indexes
    op.create_index('ix_tenants_tenant_id', 'tenants', ['tenant_id'])
    op.create_index('ix_tenants_slug', 'tenants', ['slug'])
    op.create_index('ix_tenants_domain', 'tenants', ['domain'])
    op.create_index('ix_tenants_status', 'tenants', ['status'])
    op.create_index('ix_tenants_subscription_tier', 'tenants', ['subscription_tier'])
    op.create_index('ix_tenants_stripe_customer_id', 'tenants', ['stripe_customer_id'])
    op.create_index('ix_tenants_status_tier', 'tenants', ['status', 'subscription_tier'])
    op.create_index('ix_tenants_created_at', 'tenants', ['created_at'])
    op.create_index('ix_tenants_settings', 'tenants', ['settings'],
                    postgresql_using='gin')
    op.create_index('ix_tenants_feature_flags', 'tenants', ['feature_flags'],
                    postgresql_using='gin')

    # ========================================
    # Create users table
    # ========================================
    op.create_table(
        'users',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),

        # User identifier
        sa.Column('user_id', sa.String(100), nullable=False, unique=True,
                  comment='External user identifier'),
        sa.Column('email', sa.String(255), nullable=False, unique=True,
                  comment='User email address'),

        # Multi-tenancy
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False,
                  comment='Associated tenant'),

        # Roles
        sa.Column('roles', postgresql.ARRAY(sa.String()),
                  nullable=False, server_default='{"viewer"}',
                  comment='User roles'),

        # Status
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true',
                  comment='Whether user is active'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Creation timestamp'),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True,
                  comment='Last login timestamp'),

        comment='System users'
    )

    # Users indexes
    op.create_index('ix_users_user_id', 'users', ['user_id'])
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_tenant_id', 'users', ['tenant_id'])
    op.create_index('ix_users_tenant_active', 'users', ['tenant_id', 'is_active'])

    # ========================================
    # Create agents table
    # ========================================
    op.create_table(
        'agents',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()'),
                  comment='Primary key'),

        # Agent identifier
        sa.Column('agent_id', sa.String(255), nullable=False, unique=True,
                  comment='Unique agent identifier'),

        # Basic info
        sa.Column('name', sa.String(255), nullable=False,
                  comment='Human-readable name'),
        sa.Column('description', sa.Text(), nullable=True,
                  comment='Agent description'),
        sa.Column('category', sa.String(100), nullable=False,
                  comment='Agent category'),

        # Lifecycle state - using string for flexibility
        sa.Column('state', sa.String(50), nullable=False, server_default='DRAFT',
                  comment='Lifecycle state (DRAFT, EXPERIMENTAL, CERTIFIED, DEPRECATED, RETIRED)'),

        # Multi-tenancy
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False,
                  comment='Owning tenant'),

        # Tags and frameworks
        sa.Column('tags', postgresql.ARRAY(sa.String()),
                  nullable=False, server_default='{}',
                  comment='Searchable tags'),
        sa.Column('regulatory_frameworks', postgresql.ARRAY(sa.String()),
                  nullable=False, server_default='{}',
                  comment='Applicable frameworks'),

        # Configuration
        sa.Column('entrypoint', sa.String(500), nullable=False,
                  comment='Python entrypoint'),
        sa.Column('deterministic', sa.Boolean(), nullable=False, server_default='true',
                  comment='Is agent deterministic'),

        # Full specification (JSON)
        sa.Column('spec', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Full agent specification'),

        # Metrics
        sa.Column('invocation_count', sa.Integer(), nullable=False, server_default='0',
                  comment='Total invocations'),
        sa.Column('success_rate', sa.Float(), nullable=False, server_default='1.0',
                  comment='Success rate (0-1)'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Creation timestamp'),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Last update timestamp'),

        comment='Registered agents'
    )

    # Agents indexes
    op.create_index('ix_agents_agent_id', 'agents', ['agent_id'])
    op.create_index('ix_agents_category', 'agents', ['category'])
    op.create_index('ix_agents_state', 'agents', ['state'])
    op.create_index('ix_agents_tenant_id', 'agents', ['tenant_id'])
    op.create_index('ix_agents_tenant_category', 'agents', ['tenant_id', 'category'])
    op.create_index('ix_agents_tenant_state', 'agents', ['tenant_id', 'state'])
    op.create_index('ix_agents_tags', 'agents', ['tags'], postgresql_using='gin')
    op.create_index('ix_agents_spec', 'agents', ['spec'], postgresql_using='gin')

    # ========================================
    # Create agent_versions table
    # ========================================
    op.create_table(
        'agent_versions',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),

        # Foreign key to agent
        sa.Column('agent_uuid', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('agents.id', ondelete='CASCADE'),
                  nullable=False,
                  comment='Associated agent'),

        # Multi-tenancy (denormalized for RLS performance)
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False,
                  comment='Owning tenant (denormalized)'),

        # Version info
        sa.Column('version', sa.String(50), nullable=False,
                  comment='Semantic version'),
        sa.Column('artifact_path', sa.String(500), nullable=True,
                  comment='S3 path to artifact'),
        sa.Column('checksum', sa.String(64), nullable=True,
                  comment='SHA-256 checksum'),
        sa.Column('changelog', sa.Text(), nullable=True,
                  comment='Version changelog'),
        sa.Column('is_latest', sa.Boolean(), nullable=False, server_default='false',
                  comment='Is this the latest version'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Creation timestamp'),
        sa.Column('deprecated_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When version was deprecated'),
        sa.Column('sunset_date', sa.DateTime(timezone=True), nullable=True,
                  comment='When version will be removed'),

        comment='Agent versions'
    )

    # Agent versions indexes
    op.create_index('ix_agent_versions_agent_uuid', 'agent_versions', ['agent_uuid'])
    op.create_index('ix_agent_versions_tenant_id', 'agent_versions', ['tenant_id'])
    op.create_index('ix_agent_versions_agent_version', 'agent_versions',
                    ['agent_uuid', 'version'], unique=True)
    op.create_index('ix_agent_versions_latest', 'agent_versions',
                    ['agent_uuid', 'is_latest'])

    # ========================================
    # Create executions table
    # ========================================
    op.create_table(
        'executions',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),

        # Execution identifier
        sa.Column('execution_id', sa.String(100), nullable=False, unique=True,
                  comment='Unique execution identifier'),

        # Foreign key to agent
        sa.Column('agent_uuid', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('agents.id', ondelete='SET NULL'),
                  nullable=True,
                  comment='Associated agent'),

        # Multi-tenancy
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False,
                  comment='Owning tenant'),

        # User who initiated
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='SET NULL'),
                  nullable=True,
                  comment='User who initiated'),

        # Status - using string for flexibility
        sa.Column('status', sa.String(50), nullable=False, server_default='PENDING',
                  comment='Execution status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT)'),
        sa.Column('progress', sa.Integer(), nullable=True,
                  comment='Progress percentage (0-100)'),
        sa.Column('error_message', sa.Text(), nullable=True,
                  comment='Error message if failed'),

        # Input/Output
        sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Input data'),
        sa.Column('output_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Output data'),

        # Provenance
        sa.Column('input_hash', sa.String(64), nullable=True,
                  comment='SHA-256 of input'),
        sa.Column('output_hash', sa.String(64), nullable=True,
                  comment='SHA-256 of output'),
        sa.Column('provenance_hash', sa.String(64), nullable=True,
                  comment='Full provenance chain hash'),

        # Metrics
        sa.Column('duration_ms', sa.Float(), nullable=True,
                  comment='Execution duration in ms'),
        sa.Column('llm_tokens_input', sa.Integer(), nullable=False, server_default='0',
                  comment='LLM input tokens'),
        sa.Column('llm_tokens_output', sa.Integer(), nullable=False, server_default='0',
                  comment='LLM output tokens'),

        # Cost
        sa.Column('compute_cost_usd', sa.Float(), nullable=False, server_default='0',
                  comment='Compute cost in USD'),
        sa.Column('llm_cost_usd', sa.Float(), nullable=False, server_default='0',
                  comment='LLM cost in USD'),
        sa.Column('total_cost_usd', sa.Float(), nullable=False, server_default='0',
                  comment='Total cost in USD'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Creation timestamp'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When execution started'),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When execution completed'),

        # Version used
        sa.Column('version_used', sa.String(50), nullable=True,
                  comment='Agent version used'),

        comment='Agent executions'
    )

    # Executions indexes
    op.create_index('ix_executions_execution_id', 'executions', ['execution_id'])
    op.create_index('ix_executions_agent_uuid', 'executions', ['agent_uuid'])
    op.create_index('ix_executions_tenant_id', 'executions', ['tenant_id'])
    op.create_index('ix_executions_user_id', 'executions', ['user_id'])
    op.create_index('ix_executions_status', 'executions', ['status'])
    op.create_index('ix_executions_created_at', 'executions', ['created_at'])
    op.create_index('ix_executions_agent_created', 'executions',
                    ['agent_uuid', 'created_at'])
    op.create_index('ix_executions_tenant_status', 'executions',
                    ['tenant_id', 'status'])
    op.create_index('ix_executions_tenant_created', 'executions',
                    ['tenant_id', 'created_at'])
    op.create_index('ix_executions_input_data', 'executions', ['input_data'],
                    postgresql_using='gin')
    op.create_index('ix_executions_output_data', 'executions', ['output_data'],
                    postgresql_using='gin')

    # ========================================
    # Create audit_logs table
    # ========================================
    op.create_table(
        'audit_logs',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),

        # Who
        sa.Column('actor', sa.String(255), nullable=False,
                  comment='User or system ID'),
        sa.Column('actor_type', sa.String(50), nullable=False, server_default='user',
                  comment='Actor type (user, system, api_key)'),

        # What
        sa.Column('action', sa.String(100), nullable=False,
                  comment='Action performed'),
        sa.Column('resource_type', sa.String(100), nullable=False,
                  comment='Resource type'),
        sa.Column('resource_id', sa.String(255), nullable=False,
                  comment='Resource identifier'),

        # Context
        sa.Column('context', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Additional context'),
        sa.Column('old_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Previous value (for updates)'),
        sa.Column('new_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='New value (for updates)'),

        # Multi-tenancy
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False,
                  comment='Associated tenant'),

        # Request metadata
        sa.Column('ip_address', sa.String(50), nullable=True,
                  comment='Client IP address'),
        sa.Column('user_agent', sa.Text(), nullable=True,
                  comment='Client user agent'),
        sa.Column('correlation_id', sa.String(100), nullable=True,
                  comment='Request correlation ID'),

        # Timestamp
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Event timestamp'),

        # Hash chain for tamper evidence
        sa.Column('previous_hash', sa.String(64), nullable=True,
                  comment='Hash of previous entry'),
        sa.Column('entry_hash', sa.String(64), nullable=True,
                  comment='Hash of this entry'),

        comment='Audit logs'
    )

    # Audit logs indexes
    op.create_index('ix_audit_logs_actor', 'audit_logs', ['actor'])
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_logs_resource_type', 'audit_logs', ['resource_type'])
    op.create_index('ix_audit_logs_tenant_id', 'audit_logs', ['tenant_id'])
    op.create_index('ix_audit_logs_correlation_id', 'audit_logs', ['correlation_id'])
    op.create_index('ix_audit_logs_created_at', 'audit_logs', ['created_at'])
    op.create_index('ix_audit_logs_tenant_time', 'audit_logs',
                    ['tenant_id', 'created_at'])
    op.create_index('ix_audit_logs_resource', 'audit_logs',
                    ['resource_type', 'resource_id'])
    op.create_index('ix_audit_logs_context', 'audit_logs', ['context'],
                    postgresql_using='gin')

    # ========================================
    # Create tenant_usage_logs table
    # ========================================
    op.create_table(
        'tenant_usage_logs',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),

        # Foreign key to tenant
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False,
                  comment='Associated tenant'),

        # Metric info
        sa.Column('metric_name', sa.String(100), nullable=False,
                  comment='Name of the usage metric'),
        sa.Column('metric_value', sa.BigInteger(), nullable=False,
                  comment='Metric value'),

        # Period
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False,
                  comment='Period start timestamp'),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False,
                  comment='Period end timestamp'),

        # Metadata
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default='{}',
                  comment='Additional metadata'),

        # Timestamp
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='When recorded'),

        comment='Tenant usage history'
    )

    # Tenant usage logs indexes
    op.create_index('ix_usage_logs_tenant_id', 'tenant_usage_logs', ['tenant_id'])
    op.create_index('ix_usage_logs_metric_name', 'tenant_usage_logs', ['metric_name'])
    op.create_index('ix_usage_logs_period_start', 'tenant_usage_logs', ['period_start'])
    op.create_index('ix_usage_logs_tenant_metric', 'tenant_usage_logs',
                    ['tenant_id', 'metric_name'])
    op.create_index('ix_usage_logs_period', 'tenant_usage_logs',
                    ['tenant_id', 'period_start', 'period_end'])

    # ========================================
    # Create tenant_invitations table
    # ========================================
    op.create_table(
        'tenant_invitations',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),

        # Foreign key to tenant
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('tenants.id', ondelete='CASCADE'),
                  nullable=False,
                  comment='Associated tenant'),

        # Invitation details
        sa.Column('email', sa.String(255), nullable=False,
                  comment='Invited email address'),
        sa.Column('role', sa.String(50), nullable=False, server_default='viewer',
                  comment='Role to assign'),
        sa.Column('token', sa.String(255), nullable=False, unique=True,
                  comment='Invitation token'),
        sa.Column('invited_by', postgresql.UUID(as_uuid=True), nullable=False,
                  comment='User who sent invitation'),

        # Timestamps
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False,
                  comment='Token expiration'),
        sa.Column('accepted_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When invitation was accepted'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()'),
                  comment='Creation timestamp'),

        comment='Pending user invitations'
    )

    # Tenant invitations indexes
    op.create_index('ix_invitations_tenant_id', 'tenant_invitations', ['tenant_id'])
    op.create_index('ix_invitations_email', 'tenant_invitations', ['email'])
    op.create_index('ix_invitations_token', 'tenant_invitations', ['token'])
    op.create_index('ix_invitations_tenant_email', 'tenant_invitations',
                    ['tenant_id', 'email'])

    # ========================================
    # Create updated_at trigger function
    # ========================================
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Apply updated_at trigger to tenants
    op.execute("""
        CREATE TRIGGER update_tenants_updated_at
            BEFORE UPDATE ON tenants
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)

    # Apply updated_at trigger to agents
    op.execute("""
        CREATE TRIGGER update_agents_updated_at
            BEFORE UPDATE ON agents
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    """
    Drop all GL-Agent-Factory database tables.

    Tables are dropped in reverse dependency order.
    """

    # Drop triggers first
    op.execute("DROP TRIGGER IF EXISTS update_agents_updated_at ON agents")
    op.execute("DROP TRIGGER IF EXISTS update_tenants_updated_at ON tenants")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop tables in reverse dependency order
    op.drop_table('tenant_invitations')
    op.drop_table('tenant_usage_logs')
    op.drop_table('audit_logs')
    op.drop_table('executions')
    op.drop_table('agent_versions')
    op.drop_table('agents')
    op.drop_table('users')
    op.drop_table('tenants')

    # Drop ENUM types
    op.execute("DROP TYPE IF EXISTS execution_status")
    op.execute("DROP TYPE IF EXISTS agent_state")
    op.execute("DROP TYPE IF EXISTS tenant_status")
    op.execute("DROP TYPE IF EXISTS subscription_tier")

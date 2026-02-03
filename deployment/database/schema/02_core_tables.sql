-- =============================================================================
-- GreenLang Climate OS - Core Application Tables
-- =============================================================================
-- File: 02_core_tables.sql
-- Description: Core application tables for organizations, users, projects,
--              and API key management in the public schema.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Organizations Table
-- -----------------------------------------------------------------------------
-- Represents customer organizations using the GreenLang platform.
-- Each organization can have multiple users and projects.
CREATE TABLE IF NOT EXISTS public.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Organization identification
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,

    -- Organization type and tier
    organization_type VARCHAR(50) NOT NULL DEFAULT 'enterprise',
    subscription_tier VARCHAR(50) NOT NULL DEFAULT 'standard',

    -- Settings stored as JSONB for flexibility
    -- Example: {"timezone": "UTC", "currency": "USD", "fiscal_year_start": "01-01"}
    settings JSONB NOT NULL DEFAULT '{}',

    -- Feature flags for organization-specific features
    feature_flags JSONB NOT NULL DEFAULT '{}',

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT organizations_name_length CHECK (LENGTH(name) >= 2),
    CONSTRAINT organizations_slug_format CHECK (slug ~ '^[a-z0-9-]+$'),
    CONSTRAINT organizations_type_valid CHECK (organization_type IN ('enterprise', 'small_business', 'government', 'nonprofit', 'educational')),
    CONSTRAINT organizations_tier_valid CHECK (subscription_tier IN ('free', 'standard', 'professional', 'enterprise'))
);

-- Index for slug lookups (used in URL routing)
CREATE INDEX IF NOT EXISTS idx_organizations_slug ON public.organizations(slug);

-- Index for active organizations
CREATE INDEX IF NOT EXISTS idx_organizations_active ON public.organizations(is_active) WHERE is_active = true;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER organizations_updated_at
    BEFORE UPDATE ON public.organizations
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

COMMENT ON TABLE public.organizations IS 'Customer organizations using the GreenLang Climate OS platform';
COMMENT ON COLUMN public.organizations.settings IS 'Organization-specific settings as JSONB (timezone, currency, fiscal year, etc.)';
COMMENT ON COLUMN public.organizations.feature_flags IS 'Feature flags for controlling organization-specific functionality';

-- -----------------------------------------------------------------------------
-- Users Table
-- -----------------------------------------------------------------------------
-- User accounts within organizations. Supports multiple roles and MFA.
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Organization relationship
    org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,

    -- User identification
    email VARCHAR(255) NOT NULL,
    email_verified BOOLEAN NOT NULL DEFAULT false,

    -- Authentication (password hash stored using pgcrypto)
    password_hash VARCHAR(255),

    -- Multi-factor authentication
    mfa_enabled BOOLEAN NOT NULL DEFAULT false,
    mfa_secret VARCHAR(255),

    -- User profile
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    display_name VARCHAR(200),
    avatar_url VARCHAR(500),

    -- Role and permissions
    -- Roles: admin, manager, analyst, viewer, api_only
    role VARCHAR(50) NOT NULL DEFAULT 'viewer',

    -- Additional permissions as JSONB
    permissions JSONB NOT NULL DEFAULT '[]',

    -- User preferences
    preferences JSONB NOT NULL DEFAULT '{}',

    -- Status and security
    is_active BOOLEAN NOT NULL DEFAULT true,
    last_login_at TIMESTAMPTZ,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TIMESTAMPTZ,

    -- Password reset
    password_reset_token VARCHAR(255),
    password_reset_expires TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT users_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT users_role_valid CHECK (role IN ('admin', 'manager', 'analyst', 'viewer', 'api_only')),
    CONSTRAINT users_org_email_unique UNIQUE (org_id, email)
);

-- Index for email lookups (authentication)
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);

-- Index for organization membership queries
CREATE INDEX IF NOT EXISTS idx_users_org_id ON public.users(org_id);

-- Index for active users
CREATE INDEX IF NOT EXISTS idx_users_active ON public.users(is_active) WHERE is_active = true;

-- Partial index for locked accounts (security monitoring)
CREATE INDEX IF NOT EXISTS idx_users_locked ON public.users(locked_until) WHERE locked_until IS NOT NULL;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

COMMENT ON TABLE public.users IS 'User accounts within organizations with role-based access control';
COMMENT ON COLUMN public.users.role IS 'User role: admin, manager, analyst, viewer, api_only';
COMMENT ON COLUMN public.users.permissions IS 'Additional granular permissions as JSONB array';
COMMENT ON COLUMN public.users.preferences IS 'User preferences (UI settings, notifications, etc.)';

-- -----------------------------------------------------------------------------
-- Projects Table
-- -----------------------------------------------------------------------------
-- Projects organize emission sources and calculations within an organization.
CREATE TABLE IF NOT EXISTS public.projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Organization relationship
    org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,

    -- Project identification
    name VARCHAR(255) NOT NULL,
    description TEXT,
    code VARCHAR(50),

    -- Project type
    -- Types: facility, supply_chain, product, portfolio, custom
    project_type VARCHAR(50) NOT NULL DEFAULT 'facility',

    -- Project configuration as JSONB
    -- Example: {"boundaries": {...}, "reporting_period": {...}, "methodology": "ghg_protocol"}
    config JSONB NOT NULL DEFAULT '{}',

    -- Emission boundaries (organizational, operational, equity share)
    boundary_type VARCHAR(50) NOT NULL DEFAULT 'operational',

    -- Project status
    status VARCHAR(50) NOT NULL DEFAULT 'active',

    -- Project team (user IDs with roles)
    team_members JSONB NOT NULL DEFAULT '[]',

    -- Tags for categorization
    tags VARCHAR(100)[] DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT projects_name_length CHECK (LENGTH(name) >= 2),
    CONSTRAINT projects_type_valid CHECK (project_type IN ('facility', 'supply_chain', 'product', 'portfolio', 'custom')),
    CONSTRAINT projects_boundary_valid CHECK (boundary_type IN ('operational', 'financial', 'equity_share')),
    CONSTRAINT projects_status_valid CHECK (status IN ('draft', 'active', 'archived', 'completed')),
    CONSTRAINT projects_org_code_unique UNIQUE (org_id, code)
);

-- Index for organization projects
CREATE INDEX IF NOT EXISTS idx_projects_org_id ON public.projects(org_id);

-- Index for project status queries
CREATE INDEX IF NOT EXISTS idx_projects_status ON public.projects(status);

-- Index for project type queries
CREATE INDEX IF NOT EXISTS idx_projects_type ON public.projects(project_type);

-- GIN index for tags array search
CREATE INDEX IF NOT EXISTS idx_projects_tags ON public.projects USING GIN(tags);

-- GIN index for JSONB config search
CREATE INDEX IF NOT EXISTS idx_projects_config ON public.projects USING GIN(config jsonb_path_ops);

CREATE TRIGGER projects_updated_at
    BEFORE UPDATE ON public.projects
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

COMMENT ON TABLE public.projects IS 'Projects organize emission sources and calculations within organizations';
COMMENT ON COLUMN public.projects.config IS 'Project configuration including boundaries, reporting periods, and methodology';
COMMENT ON COLUMN public.projects.boundary_type IS 'GHG Protocol boundary approach: operational, financial, or equity_share';

-- -----------------------------------------------------------------------------
-- API Keys Table
-- -----------------------------------------------------------------------------
-- API keys for programmatic access to the GreenLang API.
CREATE TABLE IF NOT EXISTS public.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- User who created the key
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,

    -- Key identification
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Key hash (never store plaintext keys)
    -- The actual key is shown once on creation, only the hash is stored
    key_hash VARCHAR(255) NOT NULL,

    -- Key prefix for identification (first 8 chars of key)
    key_prefix VARCHAR(10) NOT NULL,

    -- Scopes (permissions) for this key
    -- Example: ["emissions:read", "emissions:write", "reports:read"]
    scopes VARCHAR(100)[] NOT NULL DEFAULT '{}',

    -- Rate limiting
    rate_limit_per_minute INTEGER NOT NULL DEFAULT 100,
    rate_limit_per_day INTEGER NOT NULL DEFAULT 10000,

    -- IP restrictions (optional)
    allowed_ips INET[] DEFAULT NULL,

    -- Expiration
    expires_at TIMESTAMPTZ,

    -- Usage tracking
    last_used_at TIMESTAMPTZ,
    total_requests BIGINT NOT NULL DEFAULT 0,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,
    revoked_at TIMESTAMPTZ,
    revoked_reason VARCHAR(255),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT api_keys_name_length CHECK (LENGTH(name) >= 2),
    CONSTRAINT api_keys_hash_unique UNIQUE (key_hash)
);

-- Index for key prefix lookups (fast key verification)
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON public.api_keys(key_prefix);

-- Index for user's keys
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON public.api_keys(user_id);

-- Index for active keys
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON public.api_keys(is_active) WHERE is_active = true;

-- Index for expiring keys (for cleanup jobs)
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON public.api_keys(expires_at) WHERE expires_at IS NOT NULL;

-- GIN index for scopes search
CREATE INDEX IF NOT EXISTS idx_api_keys_scopes ON public.api_keys USING GIN(scopes);

CREATE TRIGGER api_keys_updated_at
    BEFORE UPDATE ON public.api_keys
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

COMMENT ON TABLE public.api_keys IS 'API keys for programmatic access with scoped permissions';
COMMENT ON COLUMN public.api_keys.key_hash IS 'SHA-256 hash of the API key (plaintext never stored)';
COMMENT ON COLUMN public.api_keys.key_prefix IS 'First 8 characters of key for identification without exposing full key';
COMMENT ON COLUMN public.api_keys.scopes IS 'Permission scopes granted to this key';

-- -----------------------------------------------------------------------------
-- Helper function to verify API key
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.verify_api_key(p_key_prefix VARCHAR, p_key_hash VARCHAR)
RETURNS TABLE (
    key_id UUID,
    user_id UUID,
    org_id UUID,
    scopes VARCHAR[],
    is_valid BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ak.id,
        ak.user_id,
        u.org_id,
        ak.scopes,
        (
            ak.is_active = true
            AND ak.key_hash = p_key_hash
            AND (ak.expires_at IS NULL OR ak.expires_at > NOW())
            AND ak.revoked_at IS NULL
            AND u.is_active = true
        ) AS is_valid
    FROM public.api_keys ak
    JOIN public.users u ON u.id = ak.user_id
    WHERE ak.key_prefix = p_key_prefix;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION public.verify_api_key IS 'Verify API key and return associated user/org information';

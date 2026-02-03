-- =============================================================================
-- V001: GreenLang Initial Schema
-- =============================================================================
-- Description: Creates core tables for organizations, users, and projects
--              with audit triggers, indexes, and RLS policies.
-- Author: GreenLang Data Integration Team
-- Created: 2024
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema Setup
-- -----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS public;
CREATE SCHEMA IF NOT EXISTS metrics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO public, metrics, audit;

-- -----------------------------------------------------------------------------
-- Extensions
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- -----------------------------------------------------------------------------
-- Custom Types
-- -----------------------------------------------------------------------------

-- Organization status
CREATE TYPE organization_status AS ENUM (
    'active',
    'suspended',
    'pending_verification',
    'deactivated'
);

-- User status
CREATE TYPE user_status AS ENUM (
    'active',
    'inactive',
    'pending_invitation',
    'locked',
    'deactivated'
);

-- Project status
CREATE TYPE project_status AS ENUM (
    'draft',
    'active',
    'paused',
    'completed',
    'archived'
);

-- User role within organization
CREATE TYPE user_role AS ENUM (
    'owner',
    'admin',
    'editor',
    'viewer',
    'auditor'
);

-- -----------------------------------------------------------------------------
-- Core Tables: Organizations
-- -----------------------------------------------------------------------------

CREATE TABLE public.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Basic Information
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,

    -- Legal Entity Information
    legal_name VARCHAR(500),
    tax_id VARCHAR(50),
    registration_number VARCHAR(100),

    -- Contact Information
    primary_email VARCHAR(255) NOT NULL,
    phone VARCHAR(50),
    website VARCHAR(500),

    -- Address
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_province VARCHAR(100),
    postal_code VARCHAR(20),
    country_code CHAR(2),  -- ISO 3166-1 alpha-2

    -- Industry Classification
    industry_code VARCHAR(20),  -- NAICS or SIC code
    industry_name VARCHAR(255),

    -- Settings (JSONB for flexibility)
    settings JSONB DEFAULT '{}'::jsonb,

    -- Subscription and Billing
    subscription_tier VARCHAR(50) DEFAULT 'free',
    billing_email VARCHAR(255),

    -- Status and Verification
    status organization_status DEFAULT 'pending_verification',
    verified_at TIMESTAMPTZ,
    verified_by UUID,

    -- Audit Fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by UUID,
    deleted_at TIMESTAMPTZ,
    deleted_by UUID,

    -- Version for optimistic locking
    version INTEGER NOT NULL DEFAULT 1,

    -- Constraints
    CONSTRAINT organizations_email_check CHECK (primary_email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT organizations_country_check CHECK (country_code IS NULL OR LENGTH(country_code) = 2)
);

-- Organization metadata for extensibility
CREATE TABLE public.organization_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    key VARCHAR(100) NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT organization_metadata_unique UNIQUE (organization_id, key)
);

-- -----------------------------------------------------------------------------
-- Core Tables: Users
-- -----------------------------------------------------------------------------

CREATE TABLE public.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Authentication
    email VARCHAR(255) NOT NULL UNIQUE,
    email_verified BOOLEAN DEFAULT FALSE,
    email_verified_at TIMESTAMPTZ,
    password_hash VARCHAR(255),  -- bcrypt or argon2 hash

    -- Profile Information
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    display_name VARCHAR(200),
    avatar_url VARCHAR(500),
    phone VARCHAR(50),
    phone_verified BOOLEAN DEFAULT FALSE,

    -- Preferences (JSONB for flexibility)
    preferences JSONB DEFAULT '{
        "timezone": "UTC",
        "locale": "en-US",
        "date_format": "YYYY-MM-DD",
        "notifications": {
            "email": true,
            "push": true,
            "sms": false
        }
    }'::jsonb,

    -- Security
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    last_login_at TIMESTAMPTZ,
    last_login_ip INET,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMPTZ,

    -- Status
    status user_status DEFAULT 'pending_invitation',

    -- Terms and Privacy
    terms_accepted_at TIMESTAMPTZ,
    privacy_accepted_at TIMESTAMPTZ,

    -- Audit Fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    -- Version for optimistic locking
    version INTEGER NOT NULL DEFAULT 1,

    -- Constraints
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- User sessions for tracking
CREATE TABLE public.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,

    -- Session Information
    token_hash VARCHAR(255) NOT NULL,  -- Hashed session token
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,

    -- Validity
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,
    revoked_reason VARCHAR(100)
);

-- -----------------------------------------------------------------------------
-- Core Tables: Organization Memberships
-- -----------------------------------------------------------------------------

CREATE TABLE public.organization_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,

    -- Role and Permissions
    role user_role NOT NULL DEFAULT 'viewer',
    custom_permissions JSONB DEFAULT '{}'::jsonb,

    -- Invitation
    invited_by UUID REFERENCES public.users(id),
    invited_at TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,

    -- Status
    is_primary_org BOOLEAN DEFAULT FALSE,

    -- Audit Fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT organization_members_unique UNIQUE (organization_id, user_id)
);

-- -----------------------------------------------------------------------------
-- Core Tables: Projects
-- -----------------------------------------------------------------------------

CREATE TABLE public.projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,

    -- Basic Information
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL,
    description TEXT,

    -- Project Configuration
    settings JSONB DEFAULT '{
        "emissions_framework": "GHG_PROTOCOL",
        "reporting_period": "calendar_year",
        "base_year": 2020,
        "currency": "USD"
    }'::jsonb,

    -- Boundaries
    boundaries JSONB DEFAULT '{
        "organizational": "operational_control",
        "scopes": ["scope1", "scope2", "scope3"]
    }'::jsonb,

    -- Status
    status project_status DEFAULT 'draft',

    -- Timeline
    start_date DATE,
    end_date DATE,

    -- Audit Fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID REFERENCES public.users(id),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by UUID REFERENCES public.users(id),
    deleted_at TIMESTAMPTZ,
    deleted_by UUID REFERENCES public.users(id),

    -- Version for optimistic locking
    version INTEGER NOT NULL DEFAULT 1,

    -- Constraints
    CONSTRAINT projects_slug_org_unique UNIQUE (organization_id, slug),
    CONSTRAINT projects_dates_check CHECK (end_date IS NULL OR start_date IS NULL OR end_date >= start_date)
);

-- Project members (additional access beyond org membership)
CREATE TABLE public.project_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,

    -- Role override for this project
    role user_role NOT NULL DEFAULT 'viewer',

    -- Audit Fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID REFERENCES public.users(id),

    CONSTRAINT project_members_unique UNIQUE (project_id, user_id)
);

-- -----------------------------------------------------------------------------
-- Audit Infrastructure
-- -----------------------------------------------------------------------------

-- Generic audit log table (will be converted to hypertable later)
CREATE TABLE audit.audit_log (
    id UUID DEFAULT uuid_generate_v4(),

    -- What changed
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    operation VARCHAR(10) NOT NULL,  -- INSERT, UPDATE, DELETE

    -- Change details
    old_data JSONB,
    new_data JSONB,
    changed_fields TEXT[],

    -- Who and when
    performed_by UUID,
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Context
    ip_address INET,
    user_agent TEXT,
    request_id UUID,

    -- Primary key for hypertable
    PRIMARY KEY (performed_at, id)
);

-- -----------------------------------------------------------------------------
-- Audit Trigger Function
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION audit.log_changes()
RETURNS TRIGGER AS $$
DECLARE
    old_data JSONB;
    new_data JSONB;
    changed_fields TEXT[];
    key TEXT;
BEGIN
    -- Determine operation type and set data
    IF TG_OP = 'DELETE' THEN
        old_data := to_jsonb(OLD);
        new_data := NULL;
        changed_fields := ARRAY(SELECT jsonb_object_keys(old_data));
    ELSIF TG_OP = 'INSERT' THEN
        old_data := NULL;
        new_data := to_jsonb(NEW);
        changed_fields := ARRAY(SELECT jsonb_object_keys(new_data));
    ELSIF TG_OP = 'UPDATE' THEN
        old_data := to_jsonb(OLD);
        new_data := to_jsonb(NEW);

        -- Find changed fields
        changed_fields := ARRAY(
            SELECT key
            FROM jsonb_each(old_data)
            WHERE old_data->key IS DISTINCT FROM new_data->key
        );

        -- Skip if no actual changes
        IF array_length(changed_fields, 1) IS NULL THEN
            RETURN NEW;
        END IF;
    END IF;

    -- Insert audit record
    INSERT INTO audit.audit_log (
        table_name,
        record_id,
        operation,
        old_data,
        new_data,
        changed_fields,
        performed_by,
        performed_at,
        request_id
    ) VALUES (
        TG_TABLE_SCHEMA || '.' || TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        TG_OP,
        old_data,
        new_data,
        changed_fields,
        NULLIF(current_setting('app.current_user_id', true), '')::UUID,
        NOW(),
        NULLIF(current_setting('app.request_id', true), '')::UUID
    );

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- -----------------------------------------------------------------------------
-- Apply Audit Triggers
-- -----------------------------------------------------------------------------

-- Organizations audit trigger
CREATE TRIGGER organizations_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON public.organizations
    FOR EACH ROW EXECUTE FUNCTION audit.log_changes();

-- Users audit trigger
CREATE TRIGGER users_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON public.users
    FOR EACH ROW EXECUTE FUNCTION audit.log_changes();

-- Projects audit trigger
CREATE TRIGGER projects_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON public.projects
    FOR EACH ROW EXECUTE FUNCTION audit.log_changes();

-- Organization members audit trigger
CREATE TRIGGER organization_members_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON public.organization_members
    FOR EACH ROW EXECUTE FUNCTION audit.log_changes();

-- -----------------------------------------------------------------------------
-- Update Timestamp Trigger Function
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.version = COALESCE(OLD.version, 0) + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables with updated_at
CREATE TRIGGER organizations_update_timestamp
    BEFORE UPDATE ON public.organizations
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER users_update_timestamp
    BEFORE UPDATE ON public.users
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER projects_update_timestamp
    BEFORE UPDATE ON public.projects
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- -----------------------------------------------------------------------------
-- Indexes
-- -----------------------------------------------------------------------------

-- Organizations indexes
CREATE INDEX idx_organizations_slug ON public.organizations(slug);
CREATE INDEX idx_organizations_status ON public.organizations(status) WHERE deleted_at IS NULL;
CREATE INDEX idx_organizations_country ON public.organizations(country_code) WHERE deleted_at IS NULL;
CREATE INDEX idx_organizations_industry ON public.organizations(industry_code) WHERE deleted_at IS NULL;
CREATE INDEX idx_organizations_created_at ON public.organizations(created_at);
CREATE INDEX idx_organizations_name_trgm ON public.organizations USING gin(name gin_trgm_ops);
CREATE INDEX idx_organizations_settings ON public.organizations USING gin(settings jsonb_path_ops);

-- Users indexes
CREATE INDEX idx_users_email ON public.users(email);
CREATE INDEX idx_users_status ON public.users(status) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_last_login ON public.users(last_login_at);
CREATE INDEX idx_users_name_search ON public.users USING gin(
    (first_name || ' ' || last_name) gin_trgm_ops
);

-- User sessions indexes
CREATE INDEX idx_user_sessions_user_id ON public.user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires ON public.user_sessions(expires_at)
    WHERE revoked_at IS NULL;
CREATE INDEX idx_user_sessions_token ON public.user_sessions(token_hash);

-- Organization members indexes
CREATE INDEX idx_org_members_org_id ON public.organization_members(organization_id);
CREATE INDEX idx_org_members_user_id ON public.organization_members(user_id);
CREATE INDEX idx_org_members_role ON public.organization_members(role);

-- Projects indexes
CREATE INDEX idx_projects_org_id ON public.projects(organization_id);
CREATE INDEX idx_projects_status ON public.projects(status) WHERE deleted_at IS NULL;
CREATE INDEX idx_projects_slug ON public.projects(slug);
CREATE INDEX idx_projects_dates ON public.projects(start_date, end_date);
CREATE INDEX idx_projects_settings ON public.projects USING gin(settings jsonb_path_ops);

-- Audit log indexes (before hypertable conversion)
CREATE INDEX idx_audit_log_table ON audit.audit_log(table_name, performed_at DESC);
CREATE INDEX idx_audit_log_record ON audit.audit_log(record_id, performed_at DESC);
CREATE INDEX idx_audit_log_user ON audit.audit_log(performed_by, performed_at DESC);

-- -----------------------------------------------------------------------------
-- Row Level Security (RLS) Policies
-- -----------------------------------------------------------------------------

-- Enable RLS on all tables
ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;

-- Organizations: Users can only see orgs they belong to
CREATE POLICY organizations_select_policy ON public.organizations
    FOR SELECT
    USING (
        id IN (
            SELECT organization_id
            FROM public.organization_members
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
        )
    );

-- Organizations: Only owners/admins can update
CREATE POLICY organizations_update_policy ON public.organizations
    FOR UPDATE
    USING (
        id IN (
            SELECT organization_id
            FROM public.organization_members
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
              AND role IN ('owner', 'admin')
        )
    );

-- Projects: Users can see projects in their orgs
CREATE POLICY projects_select_policy ON public.projects
    FOR SELECT
    USING (
        organization_id IN (
            SELECT organization_id
            FROM public.organization_members
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
        )
    );

-- Bypass RLS for service accounts
CREATE ROLE greenlang_service;
ALTER TABLE public.organizations FORCE ROW LEVEL SECURITY;
CREATE POLICY service_account_bypass ON public.organizations
    FOR ALL
    TO greenlang_service
    USING (true)
    WITH CHECK (true);

-- -----------------------------------------------------------------------------
-- Helper Functions
-- -----------------------------------------------------------------------------

-- Function to check user's role in an organization
CREATE OR REPLACE FUNCTION public.user_org_role(
    p_user_id UUID,
    p_organization_id UUID
) RETURNS user_role AS $$
    SELECT role
    FROM public.organization_members
    WHERE user_id = p_user_id
      AND organization_id = p_organization_id;
$$ LANGUAGE sql STABLE;

-- Function to check if user has permission
CREATE OR REPLACE FUNCTION public.user_has_permission(
    p_user_id UUID,
    p_organization_id UUID,
    p_permission VARCHAR(100)
) RETURNS BOOLEAN AS $$
DECLARE
    v_role user_role;
    v_custom_permissions JSONB;
BEGIN
    SELECT role, custom_permissions
    INTO v_role, v_custom_permissions
    FROM public.organization_members
    WHERE user_id = p_user_id
      AND organization_id = p_organization_id;

    IF v_role IS NULL THEN
        RETURN FALSE;
    END IF;

    -- Check custom permissions first
    IF v_custom_permissions ? p_permission THEN
        RETURN (v_custom_permissions->>p_permission)::BOOLEAN;
    END IF;

    -- Default permissions by role
    RETURN CASE v_role
        WHEN 'owner' THEN TRUE
        WHEN 'admin' THEN p_permission NOT IN ('delete_organization', 'transfer_ownership')
        WHEN 'editor' THEN p_permission IN ('read', 'write', 'create_project')
        WHEN 'viewer' THEN p_permission = 'read'
        WHEN 'auditor' THEN p_permission IN ('read', 'view_audit_log')
        ELSE FALSE
    END;
END;
$$ LANGUAGE plpgsql STABLE;

-- -----------------------------------------------------------------------------
-- Initial Data (if needed)
-- -----------------------------------------------------------------------------

-- This section intentionally left empty for migrations
-- Seed data should be handled separately

-- -----------------------------------------------------------------------------
-- Comments for Documentation
-- -----------------------------------------------------------------------------

COMMENT ON TABLE public.organizations IS 'Core organization/tenant table for multi-tenancy';
COMMENT ON TABLE public.users IS 'User accounts with authentication and profile data';
COMMENT ON TABLE public.organization_members IS 'Many-to-many relationship between users and organizations with roles';
COMMENT ON TABLE public.projects IS 'Emissions tracking projects within organizations';
COMMENT ON TABLE audit.audit_log IS 'Comprehensive audit trail for all data changes';

COMMENT ON FUNCTION audit.log_changes() IS 'Trigger function to automatically log all data changes to audit table';
COMMENT ON FUNCTION public.user_has_permission(UUID, UUID, VARCHAR) IS 'Check if a user has a specific permission in an organization';

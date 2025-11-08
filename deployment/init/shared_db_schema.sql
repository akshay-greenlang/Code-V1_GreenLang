-- ============================================================================
-- GreenLang Platform - Shared Database Schema
-- ============================================================================
-- Purpose: Initialize shared database with app-specific schemas and tables
-- Version: 1.0.0
-- Last Updated: 2025-11-08
-- ============================================================================

-- Set timezone
SET timezone = 'UTC';

-- ============================================================================
-- CREATE SCHEMAS FOR EACH APPLICATION
-- ============================================================================

-- CBAM Application Schema
CREATE SCHEMA IF NOT EXISTS cbam;

-- CSRD Application Schema
CREATE SCHEMA IF NOT EXISTS csrd;

-- VCCI Application Schema
CREATE SCHEMA IF NOT EXISTS vcci;

-- Shared/Common Schema
CREATE SCHEMA IF NOT EXISTS shared;

-- ============================================================================
-- SHARED TABLES (public schema - accessible by all apps)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Organizations (Multi-tenant support)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    industry VARCHAR(100),
    country_code VARCHAR(3),
    timezone VARCHAR(50) DEFAULT 'UTC',
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_orgs_slug ON public.organizations(slug) WHERE deleted_at IS NULL;
CREATE INDEX idx_orgs_active ON public.organizations(is_active) WHERE deleted_at IS NULL;

-- ----------------------------------------------------------------------------
-- Users (Shared authentication)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    display_name VARCHAR(200),
    org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_users_email ON public.users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_org ON public.users(org_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_active ON public.users(is_active) WHERE deleted_at IS NULL;

-- ----------------------------------------------------------------------------
-- User App Roles (Per-app permissions)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.user_app_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    app_name VARCHAR(50) NOT NULL CHECK (app_name IN ('cbam', 'csrd', 'vcci')),
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'analyst', 'viewer', 'editor', 'auditor')),
    permissions JSONB DEFAULT '[]',
    granted_by UUID REFERENCES public.users(id),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, app_name, role)
);

CREATE INDEX idx_user_app_roles_user ON public.user_app_roles(user_id);
CREATE INDEX idx_user_app_roles_app ON public.user_app_roles(app_name);
CREATE INDEX idx_user_app_roles_active ON public.user_app_roles(is_active);

-- ----------------------------------------------------------------------------
-- Cross-App Sync Events (Message queue fallback/audit)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.cross_app_sync (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_app VARCHAR(50) NOT NULL CHECK (source_app IN ('cbam', 'csrd', 'vcci')),
    target_app VARCHAR(50) NOT NULL CHECK (target_app IN ('cbam', 'csrd', 'vcci', 'all')),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    payload JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sync_source ON public.cross_app_sync(source_app);
CREATE INDEX idx_sync_target ON public.cross_app_sync(target_app);
CREATE INDEX idx_sync_status ON public.cross_app_sync(status);
CREATE INDEX idx_sync_created ON public.cross_app_sync(created_at DESC);
CREATE INDEX idx_sync_event_type ON public.cross_app_sync(event_type);

-- ----------------------------------------------------------------------------
-- API Keys (For inter-app authentication)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    app_name VARCHAR(50) NOT NULL CHECK (app_name IN ('cbam', 'csrd', 'vcci')),
    scopes TEXT[] DEFAULT '{}',
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_by UUID REFERENCES public.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP WITH TIME ZONE,
    revoked_by UUID REFERENCES public.users(id)
);

CREATE INDEX idx_api_keys_org ON public.api_keys(org_id);
CREATE INDEX idx_api_keys_app ON public.api_keys(app_name);
CREATE INDEX idx_api_keys_active ON public.api_keys(is_active);

-- ----------------------------------------------------------------------------
-- Audit Log (Cross-app activity tracking)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES public.users(id),
    org_id UUID REFERENCES public.organizations(id),
    app_name VARCHAR(50) NOT NULL CHECK (app_name IN ('cbam', 'csrd', 'vcci')),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user ON public.audit_log(user_id);
CREATE INDEX idx_audit_org ON public.audit_log(org_id);
CREATE INDEX idx_audit_app ON public.audit_log(app_name);
CREATE INDEX idx_audit_created ON public.audit_log(created_at DESC);

-- ============================================================================
-- CBAM SCHEMA TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS cbam.import_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES public.organizations(id),
    user_id UUID NOT NULL REFERENCES public.users(id),
    file_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    total_records INTEGER DEFAULT 0,
    processed_records INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_cbam_import_org ON cbam.import_sessions(org_id);
CREATE INDEX idx_cbam_import_status ON cbam.import_sessions(status);

-- ============================================================================
-- CSRD SCHEMA TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS csrd.reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES public.organizations(id),
    user_id UUID NOT NULL REFERENCES public.users(id),
    report_type VARCHAR(100) NOT NULL,
    reporting_period VARCHAR(50),
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_csrd_reports_org ON csrd.reports(org_id);
CREATE INDEX idx_csrd_reports_status ON csrd.reports(status);

-- ============================================================================
-- VCCI SCHEMA TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS vcci.emissions_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES public.organizations(id),
    user_id UUID NOT NULL REFERENCES public.users(id),
    calculation_type VARCHAR(100) NOT NULL,
    scope VARCHAR(20),
    total_emissions_kg DECIMAL(20, 4),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_vcci_emissions_org ON vcci.emissions_calculations(org_id);
CREATE INDEX idx_vcci_emissions_status ON vcci.emissions_calculations(status);

-- ============================================================================
-- INTEGRATION VIEWS (Cross-App Data Access)
-- ============================================================================

-- View: Combined Emissions Data (VCCI → CSRD)
CREATE OR REPLACE VIEW shared.vw_emissions_for_csrd AS
SELECT
    e.id,
    e.org_id,
    o.name AS org_name,
    e.user_id,
    u.email AS user_email,
    e.calculation_type,
    e.scope,
    e.total_emissions_kg,
    e.status,
    e.created_at,
    e.completed_at
FROM vcci.emissions_calculations e
JOIN public.organizations o ON e.org_id = o.id
JOIN public.users u ON e.user_id = u.id
WHERE e.status = 'completed';

-- View: Active Organizations with App Access
CREATE OR REPLACE VIEW shared.vw_org_app_access AS
SELECT
    o.id AS org_id,
    o.name AS org_name,
    o.slug,
    COUNT(DISTINCT CASE WHEN uar.app_name = 'cbam' THEN uar.user_id END) AS cbam_users,
    COUNT(DISTINCT CASE WHEN uar.app_name = 'csrd' THEN uar.user_id END) AS csrd_users,
    COUNT(DISTINCT CASE WHEN uar.app_name = 'vcci' THEN uar.user_id END) AS vcci_users,
    o.created_at
FROM public.organizations o
LEFT JOIN public.users u ON o.id = u.org_id
LEFT JOIN public.user_app_roles uar ON u.id = uar.user_id
WHERE o.is_active = TRUE AND o.deleted_at IS NULL
GROUP BY o.id, o.name, o.slug, o.created_at;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to tables with updated_at
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON public.organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_app_roles_updated_at BEFORE UPDATE ON public.user_app_roles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cross_app_sync_updated_at BEFORE UPDATE ON public.cross_app_sync
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SEED DATA (Development/Testing)
-- ============================================================================

-- Create default organization
INSERT INTO public.organizations (id, name, slug, description, is_active)
VALUES
    ('00000000-0000-0000-0000-000000000001', 'GreenLang Demo Corp', 'demo-corp', 'Demo organization for testing', TRUE)
ON CONFLICT (id) DO NOTHING;

-- Create admin user (password: admin123)
INSERT INTO public.users (id, email, password_hash, first_name, last_name, org_id, is_superuser, is_active)
VALUES
    ('00000000-0000-0000-0000-000000000001',
     'admin@greenlang.com',
     '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5oPkqKZpZHdHi', -- admin123
     'Admin',
     'User',
     '00000000-0000-0000-0000-000000000001',
     TRUE,
     TRUE)
ON CONFLICT (email) DO NOTHING;

-- Grant admin user access to all apps
INSERT INTO public.user_app_roles (user_id, app_name, role, is_active)
VALUES
    ('00000000-0000-0000-0000-000000000001', 'cbam', 'admin', TRUE),
    ('00000000-0000-0000-0000-000000000001', 'csrd', 'admin', TRUE),
    ('00000000-0000-0000-0000-000000000001', 'vcci', 'admin', TRUE)
ON CONFLICT (user_id, app_name, role) DO NOTHING;

-- ============================================================================
-- GRANTS (Application User Permissions)
-- ============================================================================

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO PUBLIC;
GRANT USAGE ON SCHEMA cbam TO PUBLIC;
GRANT USAGE ON SCHEMA csrd TO PUBLIC;
GRANT USAGE ON SCHEMA vcci TO PUBLIC;
GRANT USAGE ON SCHEMA shared TO PUBLIC;

-- Grant table access
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA cbam TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA csrd TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA vcci TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA shared TO PUBLIC;

-- Grant sequence access
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA cbam TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA csrd TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA vcci TO PUBLIC;

-- ============================================================================
-- COMPLETE
-- ============================================================================

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE '✓ GreenLang Platform Database Initialized';
    RAISE NOTICE '  - Schemas: public, cbam, csrd, vcci, shared';
    RAISE NOTICE '  - Tables: organizations, users, user_app_roles, cross_app_sync, api_keys, audit_log';
    RAISE NOTICE '  - Default Org: demo-corp (ID: 00000000-0000-0000-0000-000000000001)';
    RAISE NOTICE '  - Admin User: admin@greenlang.com (password: admin123)';
END $$;

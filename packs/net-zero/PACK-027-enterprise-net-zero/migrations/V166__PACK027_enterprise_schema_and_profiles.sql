-- =============================================================================
-- V166: PACK-027 Enterprise Net Zero - Schema & Enterprise Profiles
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    001 of 015
-- Date:         March 2026
--
-- Creates the pack027_enterprise_net_zero schema and the enterprise_profiles
-- table for large enterprise net-zero program management. Captures corporate
-- metadata, sector classification, boundary approach, multi-entity hierarchy
-- roots, regulatory jurisdictions, and ERP integration status.
--
-- Tables (1):
--   1. pack027_enterprise_net_zero.gl_enterprise_profiles
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V165__PACK026_audit_trails_views.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack027_enterprise_net_zero;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_enterprise_profiles
-- =============================================================================
-- Enterprise organization profiles for net-zero program management with sector
-- classification, GHG Protocol boundary approach, multi-entity hierarchy,
-- regulatory jurisdiction tracking, and ERP integration metadata.

CREATE TABLE pack027_enterprise_net_zero.gl_enterprise_profiles (
    company_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    name                        VARCHAR(500)    NOT NULL,
    legal_name                  VARCHAR(500),
    sector                      VARCHAR(100)    NOT NULL,
    industry_nace               VARCHAR(10),
    industry_sic                VARCHAR(10),
    industry_gics               VARCHAR(10),
    employees                   INTEGER         NOT NULL,
    revenue_usd                 DECIMAL(18,2),
    revenue_currency            VARCHAR(3)      DEFAULT 'USD',
    -- Organizational boundary
    boundary_approach           VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    consolidation_method        VARCHAR(30)     DEFAULT 'FULL',
    -- Hierarchy
    parent_id                   UUID            REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE SET NULL,
    ownership_pct               DECIMAL(6,2)    DEFAULT 100.00,
    is_group_parent             BOOLEAN         DEFAULT FALSE,
    entity_count                INTEGER         DEFAULT 1,
    -- Geography
    hq_country                  VARCHAR(3)      NOT NULL,
    hq_region                   VARCHAR(100),
    operating_countries         TEXT[]          DEFAULT '{}',
    -- Regulatory scope
    regulatory_jurisdictions    TEXT[]          DEFAULT '{}',
    sec_filer_status            VARCHAR(30),
    csrd_scope                  BOOLEAN         DEFAULT FALSE,
    sb253_scope                 BOOLEAN         DEFAULT FALSE,
    -- ERP integration
    primary_erp                 VARCHAR(50),
    erp_connected               BOOLEAN         DEFAULT FALSE,
    -- Data quality
    data_quality_target         VARCHAR(10)     DEFAULT 'FINANCIAL_GRADE',
    assurance_level             VARCHAR(30)     DEFAULT 'LIMITED',
    -- Fiscal
    fiscal_year_end             VARCHAR(10),
    base_year                   INTEGER,
    -- Contact
    cso_name                    VARCHAR(255),
    cso_email                   VARCHAR(255),
    sustainability_team_size    INTEGER,
    -- Status
    profile_status              VARCHAR(30)     DEFAULT 'active',
    onboarding_phase            VARCHAR(30)     DEFAULT 'PHASE_1',
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_ep_boundary_approach CHECK (
        boundary_approach IN ('FINANCIAL_CONTROL', 'OPERATIONAL_CONTROL', 'EQUITY_SHARE')
    ),
    CONSTRAINT chk_p027_ep_consolidation_method CHECK (
        consolidation_method IS NULL OR consolidation_method IN ('FULL', 'PROPORTIONAL', 'EQUITY')
    ),
    CONSTRAINT chk_p027_ep_employees CHECK (
        employees >= 1
    ),
    CONSTRAINT chk_p027_ep_revenue CHECK (
        revenue_usd IS NULL OR revenue_usd >= 0
    ),
    CONSTRAINT chk_p027_ep_ownership CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_p027_ep_country_len CHECK (
        LENGTH(hq_country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p027_ep_sec_filer CHECK (
        sec_filer_status IS NULL OR sec_filer_status IN (
            'LARGE_ACCELERATED', 'ACCELERATED', 'NON_ACCELERATED', 'SMALLER_REPORTING', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p027_ep_primary_erp CHECK (
        primary_erp IS NULL OR primary_erp IN (
            'SAP_S4HANA', 'SAP_ECC', 'ORACLE_ERP_CLOUD', 'ORACLE_EBUSINESS',
            'WORKDAY', 'MICROSOFT_D365', 'INFOR', 'NETSUITE', 'OTHER', 'NONE'
        )
    ),
    CONSTRAINT chk_p027_ep_data_quality_target CHECK (
        data_quality_target IN ('FINANCIAL_GRADE', 'ACTIVITY_BASED', 'HYBRID', 'SPEND_BASED')
    ),
    CONSTRAINT chk_p027_ep_assurance_level CHECK (
        assurance_level IN ('NONE', 'LIMITED', 'REASONABLE', 'BOTH')
    ),
    CONSTRAINT chk_p027_ep_profile_status CHECK (
        profile_status IN ('active', 'inactive', 'draft', 'onboarding', 'suspended')
    ),
    CONSTRAINT chk_p027_ep_onboarding_phase CHECK (
        onboarding_phase IN ('PHASE_1', 'PHASE_2', 'PHASE_3', 'PHASE_4', 'COMPLETED')
    ),
    CONSTRAINT chk_p027_ep_base_year CHECK (
        base_year IS NULL OR (base_year >= 2015 AND base_year <= 2100)
    ),
    CONSTRAINT chk_p027_ep_entity_count CHECK (
        entity_count IS NULL OR entity_count >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_ep_tenant             ON pack027_enterprise_net_zero.gl_enterprise_profiles(tenant_id);
CREATE INDEX idx_p027_ep_sector             ON pack027_enterprise_net_zero.gl_enterprise_profiles(sector);
CREATE INDEX idx_p027_ep_nace              ON pack027_enterprise_net_zero.gl_enterprise_profiles(industry_nace);
CREATE INDEX idx_p027_ep_gics              ON pack027_enterprise_net_zero.gl_enterprise_profiles(industry_gics);
CREATE INDEX idx_p027_ep_boundary          ON pack027_enterprise_net_zero.gl_enterprise_profiles(boundary_approach);
CREATE INDEX idx_p027_ep_parent            ON pack027_enterprise_net_zero.gl_enterprise_profiles(parent_id);
CREATE INDEX idx_p027_ep_group_parent      ON pack027_enterprise_net_zero.gl_enterprise_profiles(is_group_parent) WHERE is_group_parent = TRUE;
CREATE INDEX idx_p027_ep_country           ON pack027_enterprise_net_zero.gl_enterprise_profiles(hq_country);
CREATE INDEX idx_p027_ep_countries         ON pack027_enterprise_net_zero.gl_enterprise_profiles USING GIN(operating_countries);
CREATE INDEX idx_p027_ep_jurisdictions     ON pack027_enterprise_net_zero.gl_enterprise_profiles USING GIN(regulatory_jurisdictions);
CREATE INDEX idx_p027_ep_sec_filer         ON pack027_enterprise_net_zero.gl_enterprise_profiles(sec_filer_status);
CREATE INDEX idx_p027_ep_csrd              ON pack027_enterprise_net_zero.gl_enterprise_profiles(csrd_scope) WHERE csrd_scope = TRUE;
CREATE INDEX idx_p027_ep_erp               ON pack027_enterprise_net_zero.gl_enterprise_profiles(primary_erp);
CREATE INDEX idx_p027_ep_erp_connected     ON pack027_enterprise_net_zero.gl_enterprise_profiles(erp_connected) WHERE erp_connected = TRUE;
CREATE INDEX idx_p027_ep_assurance         ON pack027_enterprise_net_zero.gl_enterprise_profiles(assurance_level);
CREATE INDEX idx_p027_ep_status            ON pack027_enterprise_net_zero.gl_enterprise_profiles(profile_status);
CREATE INDEX idx_p027_ep_onboarding        ON pack027_enterprise_net_zero.gl_enterprise_profiles(onboarding_phase);
CREATE INDEX idx_p027_ep_created           ON pack027_enterprise_net_zero.gl_enterprise_profiles(created_at DESC);
CREATE INDEX idx_p027_ep_metadata          ON pack027_enterprise_net_zero.gl_enterprise_profiles USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_enterprise_profiles_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_enterprise_profiles
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_enterprise_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_ep_tenant_isolation
    ON pack027_enterprise_net_zero.gl_enterprise_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_ep_service_bypass
    ON pack027_enterprise_net_zero.gl_enterprise_profiles
    TO greenlang_service
    USING (TRUE)
    WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack027_enterprise_net_zero TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_enterprise_profiles TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack027_enterprise_net_zero.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack027_enterprise_net_zero IS
    'PACK-027 Enterprise Net Zero Pack - Financial-grade GHG accounting, SBTi Corporate Standard, multi-entity consolidation, and external assurance readiness for large enterprises.';

COMMENT ON TABLE pack027_enterprise_net_zero.gl_enterprise_profiles IS
    'Enterprise corporate profiles for net-zero program management with sector classification, GHG Protocol boundary approach, multi-entity hierarchy, and regulatory jurisdiction tracking.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.company_id IS
    'Unique identifier for the enterprise profile.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.name IS
    'Trading or common name of the enterprise.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.legal_name IS
    'Registered legal entity name.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.sector IS
    'Primary sector classification for SBTi SDA pathway matching.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.industry_nace IS
    'NACE Rev.2 industry classification code.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.employees IS
    'Number of full-time equivalent employees across all entities.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.boundary_approach IS
    'GHG Protocol organizational boundary approach: FINANCIAL_CONTROL, OPERATIONAL_CONTROL, or EQUITY_SHARE.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.parent_id IS
    'Reference to parent entity for group hierarchy (NULL for group parent).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.ownership_pct IS
    'Ownership percentage held by parent entity (100% for wholly owned).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.regulatory_jurisdictions IS
    'Array of regulatory frameworks applicable to this enterprise (SEC, CSRD, SB253, etc.).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.primary_erp IS
    'Primary ERP system (SAP_S4HANA, ORACLE_ERP_CLOUD, WORKDAY, etc.).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.data_quality_target IS
    'Target data quality level: FINANCIAL_GRADE (+/-3%), ACTIVITY_BASED (+/-10%), HYBRID, SPEND_BASED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_profiles.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';

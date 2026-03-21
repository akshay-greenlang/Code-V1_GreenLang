-- =============================================================================
-- V158: PACK-026 SME Net Zero - Schema & SME Profiles
-- =============================================================================
-- Pack:         PACK-026 (SME Net Zero Pack)
-- Migration:    001 of 008
-- Date:         March 2026
--
-- Creates the pack026_sme_net_zero schema and the sme_profiles table for
-- small and medium enterprise net-zero journey tracking. Captures entity
-- metadata, size classification, data quality tier, accounting integration,
-- and certification pathway preferences.
--
-- Tables (1):
--   1. pack026_sme_net_zero.sme_profiles
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V157__pack025_race_to_zero_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack026_sme_net_zero;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack026_sme_net_zero.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack026_sme_net_zero.sme_profiles
-- =============================================================================
-- SME organization profiles for net-zero journey tracking with size tier
-- classification, NACE sector codes, data quality tiers, and accounting
-- software integration preferences.

CREATE TABLE pack026_sme_net_zero.sme_profiles (
    sme_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(500)    NOT NULL,
    industry_nace           VARCHAR(10)     NOT NULL,
    industry_description    VARCHAR(255),
    size_tier               VARCHAR(10)     NOT NULL,
    employee_count          INTEGER,
    revenue_eur             DECIMAL(18,2),
    country                 VARCHAR(3)      NOT NULL,
    region                  VARCHAR(100),
    postcode                VARCHAR(20),
    data_quality_tier       VARCHAR(10)     NOT NULL DEFAULT 'BRONZE',
    accounting_software     VARCHAR(30),
    certification_pathway   VARCHAR(50),
    contact_name            VARCHAR(255),
    contact_email           VARCHAR(255),
    website                 VARCHAR(500),
    fiscal_year_end         VARCHAR(10),
    profile_status          VARCHAR(30)     DEFAULT 'active',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_sme_size_tier CHECK (
        size_tier IN ('MICRO', 'SMALL', 'MEDIUM')
    ),
    CONSTRAINT chk_p026_sme_data_quality CHECK (
        data_quality_tier IN ('BRONZE', 'SILVER', 'GOLD')
    ),
    CONSTRAINT chk_p026_sme_accounting CHECK (
        accounting_software IS NULL OR accounting_software IN (
            'XERO', 'QUICKBOOKS', 'SAGE', 'FREEAGENT', 'FRESHBOOKS',
            'WAVE', 'ZOHO', 'KASHFLOW', 'OTHER', 'NONE'
        )
    ),
    CONSTRAINT chk_p026_sme_cert_pathway CHECK (
        certification_pathway IS NULL OR certification_pathway IN (
            'SME_CLIMATE_HUB', 'B_CORP', 'ISO14001', 'CARBON_TRUST',
            'CLIMATE_ACTIVE', 'PLANET_MARK', 'TOITU', 'NONE'
        )
    ),
    CONSTRAINT chk_p026_sme_profile_status CHECK (
        profile_status IN ('active', 'inactive', 'draft', 'suspended')
    ),
    CONSTRAINT chk_p026_sme_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p026_sme_employee_count CHECK (
        employee_count IS NULL OR (employee_count >= 0 AND employee_count <= 250)
    ),
    CONSTRAINT chk_p026_sme_revenue CHECK (
        revenue_eur IS NULL OR revenue_eur >= 0
    ),
    CONSTRAINT chk_p026_sme_size_employee_consistency CHECK (
        CASE
            WHEN size_tier = 'MICRO' THEN employee_count IS NULL OR employee_count <= 10
            WHEN size_tier = 'SMALL' THEN employee_count IS NULL OR employee_count <= 50
            WHEN size_tier = 'MEDIUM' THEN employee_count IS NULL OR employee_count <= 250
            ELSE TRUE
        END
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_sme_tenant          ON pack026_sme_net_zero.sme_profiles(tenant_id);
CREATE INDEX idx_p026_sme_industry_nace   ON pack026_sme_net_zero.sme_profiles(industry_nace);
CREATE INDEX idx_p026_sme_size_tier       ON pack026_sme_net_zero.sme_profiles(size_tier);
CREATE INDEX idx_p026_sme_country         ON pack026_sme_net_zero.sme_profiles(country);
CREATE INDEX idx_p026_sme_region          ON pack026_sme_net_zero.sme_profiles(country, region);
CREATE INDEX idx_p026_sme_data_quality    ON pack026_sme_net_zero.sme_profiles(data_quality_tier);
CREATE INDEX idx_p026_sme_accounting      ON pack026_sme_net_zero.sme_profiles(accounting_software);
CREATE INDEX idx_p026_sme_cert_pathway    ON pack026_sme_net_zero.sme_profiles(certification_pathway);
CREATE INDEX idx_p026_sme_status          ON pack026_sme_net_zero.sme_profiles(profile_status);
CREATE INDEX idx_p026_sme_created         ON pack026_sme_net_zero.sme_profiles(created_at DESC);
CREATE INDEX idx_p026_sme_metadata        ON pack026_sme_net_zero.sme_profiles USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p026_sme_profiles_updated
    BEFORE UPDATE ON pack026_sme_net_zero.sme_profiles
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack026_sme_net_zero.sme_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY p026_sme_profiles_tenant_isolation
    ON pack026_sme_net_zero.sme_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_sme_profiles_service_bypass
    ON pack026_sme_net_zero.sme_profiles
    TO greenlang_service
    USING (TRUE)
    WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack026_sme_net_zero TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.sme_profiles TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack026_sme_net_zero.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack026_sme_net_zero IS
    'PACK-026 SME Net Zero Pack - Net-zero journey management for small and medium enterprises.';

COMMENT ON TABLE pack026_sme_net_zero.sme_profiles IS
    'SME organization profiles for net-zero journey tracking with size tier classification, NACE sector codes, data quality tiers, and accounting integration.';

COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.sme_id IS
    'Unique identifier for the SME profile.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.name IS
    'Legal or registered name of the SME.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.industry_nace IS
    'NACE Rev.2 industry classification code.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.size_tier IS
    'EU SME size classification: MICRO (<10 employees), SMALL (<50), MEDIUM (<250).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.employee_count IS
    'Number of full-time equivalent employees.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.revenue_eur IS
    'Annual revenue in EUR.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.country IS
    'ISO 3166-1 alpha-2/3 country code.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.region IS
    'Sub-national region or state.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.postcode IS
    'Postal/ZIP code for regional grant matching.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.data_quality_tier IS
    'Data quality tier: BRONZE (estimates), SILVER (spend-based), GOLD (activity-based).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.accounting_software IS
    'Connected accounting software (XERO, QUICKBOOKS, SAGE, etc.).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.certification_pathway IS
    'Preferred certification pathway (SME_CLIMATE_HUB, B_CORP, ISO14001, etc.).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_profiles.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';

-- =============================================================================
-- V148: PACK-025 Race to Zero - Schema & Organization Profiles
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    001 of 010
-- Date:         March 2026
--
-- Creates the pack025_race_to_zero schema and the organization_profiles table
-- for Race to Zero campaign participants. Tracks entity metadata, sector NACE
-- codes, actor types, baseline emissions across all scopes, and governance.
--
-- Tables (1):
--   1. pack025_race_to_zero.organization_profiles
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V147__pack024_carbon_neutral_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack025_race_to_zero;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack025_race_to_zero.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack025_race_to_zero.organization_profiles
-- =============================================================================
-- Organization profiles for Race to Zero campaign participants including
-- sector classification, actor type, baseline emissions, and governance.

CREATE TABLE pack025_race_to_zero.organization_profiles (
    org_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(500)    NOT NULL,
    sector_nace             VARCHAR(10)     NOT NULL,
    sector_description      VARCHAR(255),
    actor_type              VARCHAR(50)     NOT NULL,
    country                 VARCHAR(3)      NOT NULL,
    region                  VARCHAR(100),
    employee_count          INTEGER,
    revenue_usd             DECIMAL(18,2),
    baseline_year           INTEGER         NOT NULL,
    baseline_emissions_s1   DECIMAL(18,4)   NOT NULL,
    baseline_emissions_s2   DECIMAL(18,4)   NOT NULL,
    baseline_emissions_s3   DECIMAL(18,4),
    total_baseline_tco2e    DECIMAL(18,4)   GENERATED ALWAYS AS (
        baseline_emissions_s1 + baseline_emissions_s2 + COALESCE(baseline_emissions_s3, 0)
    ) STORED,
    reporting_boundary      VARCHAR(50)     DEFAULT 'OPERATIONAL_CONTROL',
    fiscal_year_end         VARCHAR(10),
    governance_endorsement  BOOLEAN         DEFAULT FALSE,
    board_approval_date     DATE,
    public_commitment       BOOLEAN         DEFAULT FALSE,
    profile_status          VARCHAR(30)     DEFAULT 'draft',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_org_actor_type CHECK (
        actor_type IN ('CORPORATE', 'FINANCIAL_INSTITUTION', 'CITY',
                       'REGION', 'SME', 'UNIVERSITY', 'HEALTHCARE',
                       'HIGH_EMITTER', 'SERVICE_SECTOR', 'MANUFACTURING')
    ),
    CONSTRAINT chk_p025_org_profile_status CHECK (
        profile_status IN ('draft', 'active', 'suspended', 'withdrawn')
    ),
    CONSTRAINT chk_p025_org_baseline_year CHECK (
        baseline_year >= 2000 AND baseline_year <= 2100
    ),
    CONSTRAINT chk_p025_org_emissions_non_neg CHECK (
        baseline_emissions_s1 >= 0 AND baseline_emissions_s2 >= 0
        AND (baseline_emissions_s3 IS NULL OR baseline_emissions_s3 >= 0)
    ),
    CONSTRAINT chk_p025_org_employee_count CHECK (
        employee_count IS NULL OR employee_count >= 0
    ),
    CONSTRAINT chk_p025_org_revenue CHECK (
        revenue_usd IS NULL OR revenue_usd >= 0
    ),
    CONSTRAINT chk_p025_org_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p025_org_boundary CHECK (
        reporting_boundary IN ('OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_org_tenant       ON pack025_race_to_zero.organization_profiles(tenant_id);
CREATE INDEX idx_p025_org_sector_nace  ON pack025_race_to_zero.organization_profiles(sector_nace);
CREATE INDEX idx_p025_org_actor_type   ON pack025_race_to_zero.organization_profiles(actor_type);
CREATE INDEX idx_p025_org_country      ON pack025_race_to_zero.organization_profiles(country);
CREATE INDEX idx_p025_org_baseline_yr  ON pack025_race_to_zero.organization_profiles(baseline_year);
CREATE INDEX idx_p025_org_status       ON pack025_race_to_zero.organization_profiles(profile_status);
CREATE INDEX idx_p025_org_created      ON pack025_race_to_zero.organization_profiles(created_at DESC);
CREATE INDEX idx_p025_org_metadata     ON pack025_race_to_zero.organization_profiles USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_org_profiles_updated
    BEFORE UPDATE ON pack025_race_to_zero.organization_profiles
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.organization_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_org_profiles_tenant_isolation
    ON pack025_race_to_zero.organization_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_org_profiles_service_bypass
    ON pack025_race_to_zero.organization_profiles
    TO greenlang_service
    USING (TRUE)
    WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack025_race_to_zero TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.organization_profiles TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack025_race_to_zero.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack025_race_to_zero IS
    'PACK-025 Race to Zero Pack - UN Race to Zero campaign compliance and credibility management.';

COMMENT ON TABLE pack025_race_to_zero.organization_profiles IS
    'Organization profiles for Race to Zero campaign participants with sector NACE codes, actor types, baseline emissions, and governance metadata.';

COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.org_id IS
    'Unique identifier for the organization profile.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.name IS
    'Legal or registered name of the organization.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.sector_nace IS
    'NACE Rev.2 sector classification code.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.actor_type IS
    'Race to Zero actor type (CORPORATE, CITY, REGION, SME, etc.).';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.country IS
    'ISO 3166-1 alpha-2/3 country code.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.baseline_year IS
    'Emissions baseline reference year.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.baseline_emissions_s1 IS
    'Scope 1 baseline emissions in tCO2e.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.baseline_emissions_s2 IS
    'Scope 2 baseline emissions in tCO2e.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.baseline_emissions_s3 IS
    'Scope 3 baseline emissions in tCO2e (nullable for SMEs).';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.total_baseline_tco2e IS
    'Generated total baseline emissions across all scopes.';
COMMENT ON COLUMN pack025_race_to_zero.organization_profiles.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';

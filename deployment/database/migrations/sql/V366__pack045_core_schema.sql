-- =============================================================================
-- V366: PACK-045 Base Year Management Pack - Core Schema
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_base_year schema and foundational tables for base year
-- management per GHG Protocol Corporate Standard Chapter 5. Tracks base year
-- definitions, selection criteria scoring, and organisation-level configuration
-- for recalculation policies and threshold settings.
--
-- Tables (3):
--   1. ghg_base_year.gl_by_base_years
--   2. ghg_base_year.gl_by_selection_criteria
--   3. ghg_base_year.gl_by_configuration
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V365__pack044_views_indexes_seed.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ghg_base_year;

SET search_path TO ghg_base_year, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_base_year.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_base_years
-- =============================================================================
-- Core base year record for an organisation. Defines the chosen base year,
-- its type (fixed, rolling, regulatory), status, total emissions, and the
-- provenance hash linking to the underlying inventory data. Each organisation
-- may have multiple base year records to track changes over time, but only
-- one may be ACTIVE at any given time.

CREATE TABLE ghg_base_year.gl_by_base_years (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    base_year                   INTEGER         NOT NULL,
    base_year_type              VARCHAR(30)     NOT NULL DEFAULT 'FIXED',
    rolling_window_years        INTEGER,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    established_date            DATE            NOT NULL DEFAULT CURRENT_DATE,
    total_tco2e                 NUMERIC(14,3),
    scope1_tco2e                NUMERIC(14,3),
    scope2_location_tco2e       NUMERIC(14,3),
    scope2_market_tco2e         NUMERIC(14,3),
    scope3_tco2e                NUMERIC(14,3),
    gwp_version                 VARCHAR(10)     DEFAULT 'AR5',
    consolidation_approach      VARCHAR(30)     DEFAULT 'OPERATIONAL_CONTROL',
    methodology_description     TEXT,
    verification_status         VARCHAR(30)     DEFAULT 'UNVERIFIED',
    verified_by                 VARCHAR(255),
    verified_date               DATE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p045_by_year CHECK (
        base_year >= 1990 AND base_year <= 2100
    ),
    CONSTRAINT chk_p045_by_type CHECK (
        base_year_type IN ('FIXED', 'ROLLING_AVERAGE', 'REGULATORY')
    ),
    CONSTRAINT chk_p045_by_status CHECK (
        status IN ('DRAFT', 'UNDER_REVIEW', 'ACTIVE', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p045_by_rolling_window CHECK (
        rolling_window_years IS NULL OR (rolling_window_years >= 2 AND rolling_window_years <= 10)
    ),
    CONSTRAINT chk_p045_by_total CHECK (
        total_tco2e IS NULL OR total_tco2e >= 0
    ),
    CONSTRAINT chk_p045_by_scope1 CHECK (
        scope1_tco2e IS NULL OR scope1_tco2e >= 0
    ),
    CONSTRAINT chk_p045_by_scope2l CHECK (
        scope2_location_tco2e IS NULL OR scope2_location_tco2e >= 0
    ),
    CONSTRAINT chk_p045_by_scope2m CHECK (
        scope2_market_tco2e IS NULL OR scope2_market_tco2e >= 0
    ),
    CONSTRAINT chk_p045_by_scope3 CHECK (
        scope3_tco2e IS NULL OR scope3_tco2e >= 0
    ),
    CONSTRAINT chk_p045_by_gwp CHECK (
        gwp_version IN ('SAR', 'TAR', 'AR4', 'AR5', 'AR6')
    ),
    CONSTRAINT chk_p045_by_consolidation CHECK (
        consolidation_approach IN (
            'EQUITY_SHARE', 'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL'
        )
    ),
    CONSTRAINT chk_p045_by_verification CHECK (
        verification_status IN ('UNVERIFIED', 'LIMITED_ASSURANCE', 'REASONABLE_ASSURANCE')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_by_tenant          ON ghg_base_year.gl_by_base_years(tenant_id);
CREATE INDEX idx_p045_by_org             ON ghg_base_year.gl_by_base_years(org_id);
CREATE INDEX idx_p045_by_year            ON ghg_base_year.gl_by_base_years(base_year);
CREATE INDEX idx_p045_by_type            ON ghg_base_year.gl_by_base_years(base_year_type);
CREATE INDEX idx_p045_by_status          ON ghg_base_year.gl_by_base_years(status);
CREATE INDEX idx_p045_by_established     ON ghg_base_year.gl_by_base_years(established_date);
CREATE INDEX idx_p045_by_provenance      ON ghg_base_year.gl_by_base_years(provenance_hash);
CREATE INDEX idx_p045_by_created         ON ghg_base_year.gl_by_base_years(created_at DESC);
CREATE INDEX idx_p045_by_metadata        ON ghg_base_year.gl_by_base_years USING GIN(metadata);

-- Composite: org + active base year (only one active per org)
CREATE UNIQUE INDEX idx_p045_by_org_active ON ghg_base_year.gl_by_base_years(org_id)
    WHERE status = 'ACTIVE';

-- Composite: tenant + org for multi-tenant queries
CREATE INDEX idx_p045_by_tenant_org      ON ghg_base_year.gl_by_base_years(tenant_id, org_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_by_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_base_years
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_base_year.gl_by_selection_criteria
-- =============================================================================
-- Scoring criteria used to evaluate and justify the selected base year. Each
-- criterion (data completeness, data quality, operational stability, etc.) is
-- scored and weighted. The weighted sum determines the overall suitability of
-- a candidate year as the base year per GHG Protocol guidance.

CREATE TABLE ghg_base_year.gl_by_selection_criteria (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    base_year_id                UUID            NOT NULL REFERENCES ghg_base_year.gl_by_base_years(id) ON DELETE CASCADE,
    criterion                   VARCHAR(60)     NOT NULL,
    candidate_year              INTEGER         NOT NULL,
    score                       NUMERIC(5,2)    NOT NULL,
    weight                      NUMERIC(5,3)    NOT NULL,
    weighted_score              NUMERIC(8,4)    GENERATED ALWAYS AS (score * weight) STORED,
    max_score                   NUMERIC(5,2)    NOT NULL DEFAULT 100.00,
    evidence_description        TEXT,
    data_source                 VARCHAR(255),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_sc_criterion CHECK (
        criterion IN (
            'DATA_COMPLETENESS', 'DATA_QUALITY', 'OPERATIONAL_STABILITY',
            'REGULATORY_ALIGNMENT', 'WEATHER_NORMALITY', 'PORTFOLIO_STABILITY',
            'PRODUCTION_STABILITY', 'VERIFICATION_STATUS', 'METHODOLOGY_CONSISTENCY',
            'STAKEHOLDER_ACCEPTANCE'
        )
    ),
    CONSTRAINT chk_p045_sc_year CHECK (
        candidate_year >= 1990 AND candidate_year <= 2100
    ),
    CONSTRAINT chk_p045_sc_score CHECK (
        score >= 0 AND score <= max_score
    ),
    CONSTRAINT chk_p045_sc_weight CHECK (
        weight >= 0 AND weight <= 1.0
    ),
    CONSTRAINT chk_p045_sc_max_score CHECK (
        max_score > 0
    ),
    CONSTRAINT uq_p045_sc_by_criterion_year UNIQUE (base_year_id, criterion, candidate_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_sc_tenant          ON ghg_base_year.gl_by_selection_criteria(tenant_id);
CREATE INDEX idx_p045_sc_base_year       ON ghg_base_year.gl_by_selection_criteria(base_year_id);
CREATE INDEX idx_p045_sc_criterion       ON ghg_base_year.gl_by_selection_criteria(criterion);
CREATE INDEX idx_p045_sc_candidate       ON ghg_base_year.gl_by_selection_criteria(candidate_year);
CREATE INDEX idx_p045_sc_created         ON ghg_base_year.gl_by_selection_criteria(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_sc_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_selection_criteria
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_base_year.gl_by_configuration
-- =============================================================================
-- Organisation-level configuration for base year management. Stores the
-- recalculation policy, significance thresholds, trigger rules, target
-- tracking preferences, and reporting settings as structured JSON. One
-- active configuration per organisation; versioned via config_version.

CREATE TABLE ghg_base_year.gl_by_configuration (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_version              INTEGER         NOT NULL DEFAULT 1,
    preset_name                 VARCHAR(60),
    policy_json                 JSONB           NOT NULL DEFAULT '{}',
    config_json                 JSONB           NOT NULL DEFAULT '{}',
    thresholds_json             JSONB           NOT NULL DEFAULT '{}',
    target_tracking_json        JSONB           DEFAULT '{}',
    reporting_json              JSONB           DEFAULT '{}',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    effective_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    superseded_date             DATE,
    approved_by                 VARCHAR(255),
    approved_date               DATE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p045_cfg_version CHECK (
        config_version >= 1
    ),
    CONSTRAINT chk_p045_cfg_dates CHECK (
        superseded_date IS NULL OR superseded_date >= effective_date
    ),
    CONSTRAINT uq_p045_cfg_org_version UNIQUE (org_id, config_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_cfg_tenant         ON ghg_base_year.gl_by_configuration(tenant_id);
CREATE INDEX idx_p045_cfg_org            ON ghg_base_year.gl_by_configuration(org_id);
CREATE INDEX idx_p045_cfg_preset         ON ghg_base_year.gl_by_configuration(preset_name);
CREATE INDEX idx_p045_cfg_active         ON ghg_base_year.gl_by_configuration(is_active) WHERE is_active = true;
CREATE INDEX idx_p045_cfg_effective      ON ghg_base_year.gl_by_configuration(effective_date);
CREATE INDEX idx_p045_cfg_created        ON ghg_base_year.gl_by_configuration(created_at DESC);
CREATE INDEX idx_p045_cfg_policy         ON ghg_base_year.gl_by_configuration USING GIN(policy_json);
CREATE INDEX idx_p045_cfg_config         ON ghg_base_year.gl_by_configuration USING GIN(config_json);
CREATE INDEX idx_p045_cfg_thresholds     ON ghg_base_year.gl_by_configuration USING GIN(thresholds_json);

-- Composite: org + active config (only one active per org)
CREATE UNIQUE INDEX idx_p045_cfg_org_active ON ghg_base_year.gl_by_configuration(org_id)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_cfg_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_configuration
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_base_years ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_base_year.gl_by_selection_criteria ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_base_year.gl_by_configuration ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_by_tenant_isolation
    ON ghg_base_year.gl_by_base_years
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_by_service_bypass
    ON ghg_base_year.gl_by_base_years
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p045_sc_tenant_isolation
    ON ghg_base_year.gl_by_selection_criteria
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_sc_service_bypass
    ON ghg_base_year.gl_by_selection_criteria
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p045_cfg_tenant_isolation
    ON ghg_base_year.gl_by_configuration
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_cfg_service_bypass
    ON ghg_base_year.gl_by_configuration
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA ghg_base_year TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_base_years TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_selection_criteria TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_configuration TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_base_year.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_base_year IS
    'PACK-045 Base Year Management Pack - Complete base year lifecycle including selection, recalculation policy, trigger detection, significance assessment, adjustment packages, time series consistency, target tracking, and audit trail.';

COMMENT ON TABLE ghg_base_year.gl_by_base_years IS
    'Core base year records per organisation with type (fixed/rolling/regulatory), total emissions, scope breakdown, and verification status per GHG Protocol Chapter 5.';
COMMENT ON TABLE ghg_base_year.gl_by_selection_criteria IS
    'Weighted scoring criteria used to evaluate and justify candidate base years for selection.';
COMMENT ON TABLE ghg_base_year.gl_by_configuration IS
    'Organisation-level configuration for recalculation policy, significance thresholds, and reporting preferences. Versioned with one active config per org.';

COMMENT ON COLUMN ghg_base_year.gl_by_base_years.provenance_hash IS 'SHA-256 hash linking base year to underlying inventory data for audit provenance.';
COMMENT ON COLUMN ghg_base_year.gl_by_base_years.base_year_type IS 'FIXED (single year), ROLLING_AVERAGE (multi-year average), or REGULATORY (externally mandated).';
COMMENT ON COLUMN ghg_base_year.gl_by_base_years.gwp_version IS 'IPCC Global Warming Potential version used: SAR, TAR, AR4, AR5, AR6.';
COMMENT ON COLUMN ghg_base_year.gl_by_selection_criteria.weighted_score IS 'Auto-calculated: score * weight. Used for ranking candidate years.';
COMMENT ON COLUMN ghg_base_year.gl_by_configuration.policy_json IS 'Recalculation policy configuration including trigger rules and approval workflow.';
COMMENT ON COLUMN ghg_base_year.gl_by_configuration.thresholds_json IS 'Significance and de minimis thresholds for recalculation decisions.';

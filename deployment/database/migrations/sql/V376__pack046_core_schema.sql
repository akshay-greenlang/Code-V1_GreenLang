-- =============================================================================
-- V376: PACK-046 Intensity Metrics Pack - Core Schema
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_intensity schema and foundational tables for intensity
-- metrics management. Tracks organisation-level configurations (sector,
-- scope inclusion, consolidation approach) and reporting periods for which
-- intensity calculations are performed. Implements multi-framework
-- intensity reporting per GHG Protocol, ESRS E1, CDP, SBTi SDA, and
-- sector-specific pathways.
--
-- Tables (2):
--   1. ghg_intensity.gl_im_configurations
--   2. ghg_intensity.gl_im_reporting_periods
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V375__pack045_views_indexes_seed.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ghg_intensity;

SET search_path TO ghg_intensity, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_intensity.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_configurations
-- =============================================================================
-- Pack-level configuration per organisation. Defines the sector context,
-- scope inclusion boundaries, consolidation approach, base currency, and
-- decimal precision for intensity calculations. Each organisation may have
-- multiple configurations (e.g., group-level vs subsidiary), but each
-- config_name must be unique within the organisation.

CREATE TABLE ghg_intensity.gl_im_configurations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_name                 VARCHAR(255)    NOT NULL,
    sector                      VARCHAR(100)    NOT NULL,
    scope_inclusion             VARCHAR(50)     NOT NULL DEFAULT 'SCOPE_1_2_LOCATION',
    consolidation_approach      VARCHAR(50)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    base_currency               VARCHAR(3)      NOT NULL DEFAULT 'EUR',
    decimal_places              INTEGER         NOT NULL DEFAULT 6,
    preset_name                 VARCHAR(100),
    config_json                 JSONB           NOT NULL DEFAULT '{}',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p046_cfg_sector CHECK (
        sector IN (
            'MULTI_SECTOR', 'MANUFACTURING', 'REAL_ESTATE', 'POWER_GENERATION',
            'TRANSPORT', 'FREIGHT_TRANSPORT', 'PASSENGER_TRANSPORT',
            'CEMENT', 'STEEL', 'ALUMINIUM', 'PULP_PAPER', 'CHEMICALS',
            'FOOD', 'BEVERAGE', 'MINING', 'OIL_GAS', 'UTILITIES',
            'BANKING', 'INSURANCE', 'ASSET_MANAGEMENT', 'HEALTHCARE',
            'HOSPITALITY', 'RETAIL', 'DATA_CENTER', 'ICT', 'AGRICULTURE',
            'FORESTRY', 'SERVICES', 'OFFICE', 'COMMERCIAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p046_cfg_scope CHECK (
        scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p046_cfg_consolidation CHECK (
        consolidation_approach IN (
            'EQUITY_SHARE', 'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL'
        )
    ),
    CONSTRAINT chk_p046_cfg_currency CHECK (
        LENGTH(base_currency) = 3
    ),
    CONSTRAINT chk_p046_cfg_decimal CHECK (
        decimal_places >= 0 AND decimal_places <= 10
    ),
    CONSTRAINT uq_p046_cfg_org_name UNIQUE (org_id, config_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_cfg_tenant           ON ghg_intensity.gl_im_configurations(tenant_id);
CREATE INDEX idx_p046_cfg_org              ON ghg_intensity.gl_im_configurations(org_id);
CREATE INDEX idx_p046_cfg_sector           ON ghg_intensity.gl_im_configurations(sector);
CREATE INDEX idx_p046_cfg_scope            ON ghg_intensity.gl_im_configurations(scope_inclusion);
CREATE INDEX idx_p046_cfg_preset           ON ghg_intensity.gl_im_configurations(preset_name);
CREATE INDEX idx_p046_cfg_active           ON ghg_intensity.gl_im_configurations(is_active) WHERE is_active = true;
CREATE INDEX idx_p046_cfg_created          ON ghg_intensity.gl_im_configurations(created_at DESC);
CREATE INDEX idx_p046_cfg_config_json      ON ghg_intensity.gl_im_configurations USING GIN(config_json);
CREATE INDEX idx_p046_cfg_metadata         ON ghg_intensity.gl_im_configurations USING GIN(metadata);

-- Composite: tenant + org for multi-tenant queries
CREATE INDEX idx_p046_cfg_tenant_org       ON ghg_intensity.gl_im_configurations(tenant_id, org_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_cfg_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_configurations
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_reporting_periods
-- =============================================================================
-- Reporting periods for which intensity metrics are calculated. Each period
-- is linked to a configuration and has a label (e.g., 'FY2025', 'Q1-2026'),
-- start/end dates, and a status tracking the calculation lifecycle. A period
-- may be flagged as base year for target tracking and benchmarking.

CREATE TABLE ghg_intensity.gl_im_reporting_periods (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    period_label                VARCHAR(50)     NOT NULL,
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    is_base_year                BOOLEAN         NOT NULL DEFAULT false,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_rp_dates CHECK (
        period_end > period_start
    ),
    CONSTRAINT chk_p046_rp_status CHECK (
        status IN ('DRAFT', 'CALCULATED', 'VERIFIED', 'PUBLISHED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p046_rp_config_label UNIQUE (config_id, period_label)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_rp_tenant            ON ghg_intensity.gl_im_reporting_periods(tenant_id);
CREATE INDEX idx_p046_rp_org               ON ghg_intensity.gl_im_reporting_periods(org_id);
CREATE INDEX idx_p046_rp_config            ON ghg_intensity.gl_im_reporting_periods(config_id);
CREATE INDEX idx_p046_rp_label             ON ghg_intensity.gl_im_reporting_periods(period_label);
CREATE INDEX idx_p046_rp_status            ON ghg_intensity.gl_im_reporting_periods(status);
CREATE INDEX idx_p046_rp_base_year         ON ghg_intensity.gl_im_reporting_periods(is_base_year) WHERE is_base_year = true;
CREATE INDEX idx_p046_rp_dates             ON ghg_intensity.gl_im_reporting_periods(period_start, period_end);
CREATE INDEX idx_p046_rp_created           ON ghg_intensity.gl_im_reporting_periods(created_at DESC);

-- Composite: org + config for filtered queries
CREATE INDEX idx_p046_rp_org_config        ON ghg_intensity.gl_im_reporting_periods(org_id, config_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_rp_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_reporting_periods
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_reporting_periods ENABLE ROW LEVEL SECURITY;

CREATE POLICY p046_cfg_tenant_isolation
    ON ghg_intensity.gl_im_configurations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_cfg_service_bypass
    ON ghg_intensity.gl_im_configurations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p046_rp_tenant_isolation
    ON ghg_intensity.gl_im_reporting_periods
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_rp_service_bypass
    ON ghg_intensity.gl_im_reporting_periods
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA ghg_intensity TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_configurations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_reporting_periods TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_intensity.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_intensity IS
    'PACK-046 Intensity Metrics Pack - Complete intensity metrics lifecycle including denominator registry, calculations, LMDI decomposition, benchmarking, SBTi SDA targets, scenario analysis, uncertainty quantification, and multi-framework disclosure.';

COMMENT ON TABLE ghg_intensity.gl_im_configurations IS
    'Organisation-level intensity metrics configuration defining sector context, scope inclusion boundaries, consolidation approach, and calculation precision.';
COMMENT ON TABLE ghg_intensity.gl_im_reporting_periods IS
    'Reporting periods for intensity calculations with lifecycle status tracking and base year flagging.';

COMMENT ON COLUMN ghg_intensity.gl_im_configurations.sector IS 'Sector classification determining applicable denominators and SBTi SDA pathway.';
COMMENT ON COLUMN ghg_intensity.gl_im_configurations.scope_inclusion IS 'Which emission scopes to include in intensity numerator: SCOPE_1_ONLY through ALL_SCOPES.';
COMMENT ON COLUMN ghg_intensity.gl_im_configurations.consolidation_approach IS 'GHG Protocol consolidation: EQUITY_SHARE, OPERATIONAL_CONTROL, or FINANCIAL_CONTROL.';
COMMENT ON COLUMN ghg_intensity.gl_im_configurations.config_json IS 'Extended configuration as JSON including sector-specific settings, custom thresholds, and integration parameters.';
COMMENT ON COLUMN ghg_intensity.gl_im_reporting_periods.period_label IS 'Human-readable period identifier, e.g. FY2025, Q1-2026, CY2025.';
COMMENT ON COLUMN ghg_intensity.gl_im_reporting_periods.is_base_year IS 'Flag indicating this period serves as the base year for target tracking and trend analysis.';

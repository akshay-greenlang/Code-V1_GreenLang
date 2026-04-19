-- =============================================================================
-- V168: PACK-027 Enterprise Net Zero - Comprehensive Baselines
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    003 of 015
-- Date:         March 2026
--
-- Enterprise GHG baselines with financial-grade detail across all scopes:
-- Scope 1 (8 sub-categories), Scope 2 (location + market), and all 15
-- Scope 3 categories with per-category data quality scoring and confidence
-- intervals. Supports multi-entity per-year baselines with base year
-- recalculation tracking.
--
-- Tables (1):
--   1. pack027_enterprise_net_zero.gl_enterprise_baselines
--
-- Previous: V167__PACK027_multi_entity_hierarchy.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_enterprise_baselines
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_enterprise_baselines (
    baseline_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Reporting period
    reporting_year              INTEGER         NOT NULL,
    is_base_year                BOOLEAN         DEFAULT FALSE,
    base_year_version           INTEGER         DEFAULT 1,
    recalculation_reason        TEXT,
    -- Scope 1 emissions (8 sub-categories per MRV agents)
    scope1_stationary_tco2e     DECIMAL(18,4)   DEFAULT 0,
    scope1_refrigerant_tco2e    DECIMAL(18,4)   DEFAULT 0,
    scope1_mobile_tco2e         DECIMAL(18,4)   DEFAULT 0,
    scope1_process_tco2e        DECIMAL(18,4)   DEFAULT 0,
    scope1_fugitive_tco2e       DECIMAL(18,4)   DEFAULT 0,
    scope1_land_use_tco2e       DECIMAL(18,4)   DEFAULT 0,
    scope1_waste_tco2e          DECIMAL(18,4)   DEFAULT 0,
    scope1_agricultural_tco2e   DECIMAL(18,4)   DEFAULT 0,
    scope1_total_tco2e          DECIMAL(18,4)   GENERATED ALWAYS AS (
        COALESCE(scope1_stationary_tco2e, 0) + COALESCE(scope1_refrigerant_tco2e, 0) +
        COALESCE(scope1_mobile_tco2e, 0) + COALESCE(scope1_process_tco2e, 0) +
        COALESCE(scope1_fugitive_tco2e, 0) + COALESCE(scope1_land_use_tco2e, 0) +
        COALESCE(scope1_waste_tco2e, 0) + COALESCE(scope1_agricultural_tco2e, 0)
    ) STORED,
    -- Scope 2 emissions (dual reporting)
    scope2_location_tco2e       DECIMAL(18,4)   NOT NULL DEFAULT 0,
    scope2_market_tco2e         DECIMAL(18,4)   NOT NULL DEFAULT 0,
    scope2_steam_heat_tco2e     DECIMAL(18,4)   DEFAULT 0,
    scope2_cooling_tco2e        DECIMAL(18,4)   DEFAULT 0,
    -- Scope 3 emissions (all 15 categories)
    scope3_cat1_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat2_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat3_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat4_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat5_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat6_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat7_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat8_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat9_tco2e           DECIMAL(18,4)   DEFAULT 0,
    scope3_cat10_tco2e          DECIMAL(18,4)   DEFAULT 0,
    scope3_cat11_tco2e          DECIMAL(18,4)   DEFAULT 0,
    scope3_cat12_tco2e          DECIMAL(18,4)   DEFAULT 0,
    scope3_cat13_tco2e          DECIMAL(18,4)   DEFAULT 0,
    scope3_cat14_tco2e          DECIMAL(18,4)   DEFAULT 0,
    scope3_cat15_tco2e          DECIMAL(18,4)   DEFAULT 0,
    scope3_total_tco2e          DECIMAL(18,4)   GENERATED ALWAYS AS (
        COALESCE(scope3_cat1_tco2e, 0) + COALESCE(scope3_cat2_tco2e, 0) +
        COALESCE(scope3_cat3_tco2e, 0) + COALESCE(scope3_cat4_tco2e, 0) +
        COALESCE(scope3_cat5_tco2e, 0) + COALESCE(scope3_cat6_tco2e, 0) +
        COALESCE(scope3_cat7_tco2e, 0) + COALESCE(scope3_cat8_tco2e, 0) +
        COALESCE(scope3_cat9_tco2e, 0) + COALESCE(scope3_cat10_tco2e, 0) +
        COALESCE(scope3_cat11_tco2e, 0) + COALESCE(scope3_cat12_tco2e, 0) +
        COALESCE(scope3_cat13_tco2e, 0) + COALESCE(scope3_cat14_tco2e, 0) +
        COALESCE(scope3_cat15_tco2e, 0)
    ) STORED,
    -- Data quality scoring (1-5 scale per GHG Protocol hierarchy)
    data_quality_score          DECIMAL(5,2),
    dq_scope1                   DECIMAL(3,1),
    dq_scope2                   DECIMAL(3,1),
    dq_scope3_overall           DECIMAL(3,1),
    dq_by_category              JSONB           DEFAULT '{}',
    -- Confidence intervals
    confidence_level_pct        DECIMAL(5,2)    DEFAULT 95.00,
    scope1_ci_lower_tco2e       DECIMAL(18,4),
    scope1_ci_upper_tco2e       DECIMAL(18,4),
    scope2_ci_lower_tco2e       DECIMAL(18,4),
    scope2_ci_upper_tco2e       DECIMAL(18,4),
    scope3_ci_lower_tco2e       DECIMAL(18,4),
    scope3_ci_upper_tco2e       DECIMAL(18,4),
    -- Intensity metrics
    intensity_per_employee      DECIMAL(18,6),
    intensity_per_revenue       DECIMAL(18,8),
    intensity_per_unit          DECIMAL(18,8),
    intensity_unit_type         VARCHAR(50),
    -- Consolidation
    consolidation_approach      VARCHAR(30),
    entity_count_included       INTEGER,
    entities_excluded           TEXT[],
    exclusion_justification     TEXT,
    intercompany_eliminated     BOOLEAN         DEFAULT FALSE,
    -- Materiality assessment
    materiality_assessment      JSONB           DEFAULT '{}',
    categories_excluded         TEXT[]          DEFAULT '{}',
    exclusion_total_pct         DECIMAL(6,2),
    -- Verification
    verification_status         VARCHAR(30)     DEFAULT 'unverified',
    verified_by                 VARCHAR(255),
    verified_date               DATE,
    assurance_level             VARCHAR(30),
    assurance_provider          VARCHAR(255),
    -- Metadata
    calculation_methodology     JSONB           DEFAULT '{}',
    emission_factors_used       JSONB           DEFAULT '{}',
    warnings                    TEXT[]          DEFAULT '{}',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_bl_reporting_year CHECK (
        reporting_year >= 2015 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p027_bl_scope1_non_neg CHECK (
        COALESCE(scope1_stationary_tco2e, 0) >= 0 AND
        COALESCE(scope1_refrigerant_tco2e, 0) >= 0 AND
        COALESCE(scope1_mobile_tco2e, 0) >= 0 AND
        COALESCE(scope1_process_tco2e, 0) >= 0 AND
        COALESCE(scope1_fugitive_tco2e, 0) >= 0 AND
        COALESCE(scope1_land_use_tco2e, 0) >= 0 AND
        COALESCE(scope1_waste_tco2e, 0) >= 0 AND
        COALESCE(scope1_agricultural_tco2e, 0) >= 0
    ),
    CONSTRAINT chk_p027_bl_scope2_non_neg CHECK (
        scope2_location_tco2e >= 0 AND scope2_market_tco2e >= 0
    ),
    CONSTRAINT chk_p027_bl_scope3_non_neg CHECK (
        COALESCE(scope3_cat1_tco2e, 0) >= 0 AND COALESCE(scope3_cat2_tco2e, 0) >= 0 AND
        COALESCE(scope3_cat3_tco2e, 0) >= 0 AND COALESCE(scope3_cat4_tco2e, 0) >= 0 AND
        COALESCE(scope3_cat5_tco2e, 0) >= 0 AND COALESCE(scope3_cat6_tco2e, 0) >= 0 AND
        COALESCE(scope3_cat7_tco2e, 0) >= 0 AND COALESCE(scope3_cat8_tco2e, 0) >= 0 AND
        COALESCE(scope3_cat9_tco2e, 0) >= 0 AND COALESCE(scope3_cat10_tco2e, 0) >= 0 AND
        COALESCE(scope3_cat11_tco2e, 0) >= 0 AND COALESCE(scope3_cat12_tco2e, 0) >= 0 AND
        COALESCE(scope3_cat13_tco2e, 0) >= 0 AND COALESCE(scope3_cat14_tco2e, 0) >= 0 AND
        COALESCE(scope3_cat15_tco2e, 0) >= 0
    ),
    CONSTRAINT chk_p027_bl_dq_score CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p027_bl_dq_scope CHECK (
        (dq_scope1 IS NULL OR (dq_scope1 >= 1.0 AND dq_scope1 <= 5.0)) AND
        (dq_scope2 IS NULL OR (dq_scope2 >= 1.0 AND dq_scope2 <= 5.0)) AND
        (dq_scope3_overall IS NULL OR (dq_scope3_overall >= 1.0 AND dq_scope3_overall <= 5.0))
    ),
    CONSTRAINT chk_p027_bl_confidence CHECK (
        confidence_level_pct IS NULL OR (confidence_level_pct >= 50 AND confidence_level_pct <= 99.99)
    ),
    CONSTRAINT chk_p027_bl_exclusion_pct CHECK (
        exclusion_total_pct IS NULL OR (exclusion_total_pct >= 0 AND exclusion_total_pct <= 5.0)
    ),
    CONSTRAINT chk_p027_bl_verification CHECK (
        verification_status IN ('unverified', 'self_verified', 'limited_assurance', 'reasonable_assurance', 'pending')
    ),
    CONSTRAINT chk_p027_bl_assurance_level CHECK (
        assurance_level IS NULL OR assurance_level IN ('LIMITED', 'REASONABLE')
    ),
    CONSTRAINT chk_p027_bl_consolidation CHECK (
        consolidation_approach IS NULL OR consolidation_approach IN (
            'FINANCIAL_CONTROL', 'OPERATIONAL_CONTROL', 'EQUITY_SHARE'
        )
    ),
    CONSTRAINT uq_p027_bl_company_year_version UNIQUE (company_id, reporting_year, base_year_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_bl_company            ON pack027_enterprise_net_zero.gl_enterprise_baselines(company_id);
CREATE INDEX idx_p027_bl_tenant             ON pack027_enterprise_net_zero.gl_enterprise_baselines(tenant_id);
CREATE INDEX idx_p027_bl_year               ON pack027_enterprise_net_zero.gl_enterprise_baselines(reporting_year);
CREATE INDEX idx_p027_bl_company_year       ON pack027_enterprise_net_zero.gl_enterprise_baselines(company_id, reporting_year);
CREATE INDEX idx_p027_bl_base_year          ON pack027_enterprise_net_zero.gl_enterprise_baselines(is_base_year) WHERE is_base_year = TRUE;
CREATE INDEX idx_p027_bl_dq_score           ON pack027_enterprise_net_zero.gl_enterprise_baselines(data_quality_score);
CREATE INDEX idx_p027_bl_verification       ON pack027_enterprise_net_zero.gl_enterprise_baselines(verification_status);
CREATE INDEX idx_p027_bl_assurance          ON pack027_enterprise_net_zero.gl_enterprise_baselines(assurance_level);
CREATE INDEX idx_p027_bl_consolidation      ON pack027_enterprise_net_zero.gl_enterprise_baselines(consolidation_approach);
CREATE INDEX idx_p027_bl_created            ON pack027_enterprise_net_zero.gl_enterprise_baselines(created_at DESC);
CREATE INDEX idx_p027_bl_dq_cat             ON pack027_enterprise_net_zero.gl_enterprise_baselines USING GIN(dq_by_category);
CREATE INDEX idx_p027_bl_materiality        ON pack027_enterprise_net_zero.gl_enterprise_baselines USING GIN(materiality_assessment);
CREATE INDEX idx_p027_bl_methodology        ON pack027_enterprise_net_zero.gl_enterprise_baselines USING GIN(calculation_methodology);
CREATE INDEX idx_p027_bl_metadata           ON pack027_enterprise_net_zero.gl_enterprise_baselines USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_baselines_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_enterprise_baselines
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_enterprise_baselines ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_bl_tenant_isolation
    ON pack027_enterprise_net_zero.gl_enterprise_baselines
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_bl_service_bypass
    ON pack027_enterprise_net_zero.gl_enterprise_baselines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_enterprise_baselines TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_enterprise_baselines IS
    'Enterprise GHG baselines with financial-grade detail: Scope 1 (8 sub-categories), Scope 2 (location + market), all 15 Scope 3 categories, per-category data quality scoring, and confidence intervals.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.baseline_id IS 'Unique baseline identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.reporting_year IS 'Emissions reporting year.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.is_base_year IS 'Whether this record represents the GHG Protocol base year.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.base_year_version IS 'Base year version for tracking recalculations (increments on recalc).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.scope2_location_tco2e IS 'Scope 2 location-based emissions in tCO2e (grid average factors).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.scope2_market_tco2e IS 'Scope 2 market-based emissions in tCO2e (contractual instruments).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.data_quality_score IS 'Weighted average data quality score (0-100) across all scopes.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.dq_by_category IS 'JSONB data quality scores per Scope 3 category (1.0-5.0 scale).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.exclusion_total_pct IS 'Total excluded Scope 3 categories as % of anticipated total (must be <=5%).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_enterprise_baselines.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

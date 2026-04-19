-- =============================================================================
-- V199: PACK-029 Interim Targets Pack - Actual Performance
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    004 of 015
-- Date:         March 2026
--
-- Actual emissions performance data with quarterly granularity, scope and
-- category breakdown, data quality tiers, verification status, and linkage
-- to MRV agent calculation results.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_actual_performance
--
-- TimescaleDB hypertable for time-series query performance.
-- Previous: V198__PACK029_quarterly_milestones.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_actual_performance
-- =============================================================================
-- Actual emissions performance records with quarterly granularity, GHG scope
-- and category breakdowns, data quality tiers (1-5), verification status,
-- and references to MRV agent calculation results.

CREATE TABLE pack029_interim_targets.gl_actual_performance (
    performance_id              UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    -- Time dimension
    year                        INTEGER         NOT NULL,
    quarter                     VARCHAR(2),
    period_start_date           DATE,
    period_end_date             DATE,
    reporting_period            VARCHAR(10)     DEFAULT 'QUARTERLY',
    -- Scope and category
    scope                       VARCHAR(20)     NOT NULL,
    category                    VARCHAR(60),
    category_number             INTEGER,
    sub_category                VARCHAR(100),
    -- Actual emissions
    actual_emissions_tco2e      DECIMAL(18,4)   NOT NULL,
    co2_emissions_tonnes        DECIMAL(18,4),
    ch4_emissions_tco2e         DECIMAL(18,4),
    n2o_emissions_tco2e         DECIMAL(18,4),
    fgas_emissions_tco2e        DECIMAL(18,4),
    biogenic_emissions_tco2e    DECIMAL(18,4),
    -- Intensity metrics
    actual_intensity_value      DECIMAL(18,8),
    actual_intensity_unit       VARCHAR(80),
    activity_data_value         DECIMAL(18,4),
    activity_data_unit          VARCHAR(50),
    -- Data quality
    data_quality_tier           INTEGER         NOT NULL DEFAULT 3,
    data_quality_score          DECIMAL(5,2),
    uncertainty_lower_pct       DECIMAL(6,2),
    uncertainty_upper_pct       DECIMAL(6,2),
    data_completeness_pct       DECIMAL(5,2)    DEFAULT 100.00,
    -- Data source
    data_source                 VARCHAR(200),
    calculation_method          VARCHAR(50),
    emission_factor_source      VARCHAR(100),
    emission_factor_version     VARCHAR(30),
    -- Verification
    verification_status         VARCHAR(30)     NOT NULL DEFAULT 'UNVERIFIED',
    verified_by                 VARCHAR(255),
    verification_date           DATE,
    verification_standard       VARCHAR(50),
    assurance_level             VARCHAR(20),
    -- MRV linkage
    mrv_calculation_id          UUID,
    mrv_agent_id                VARCHAR(50),
    mrv_run_timestamp           TIMESTAMPTZ,
    -- Reporting
    reported_date               DATE,
    reported_by                 VARCHAR(255),
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    is_restated                 BOOLEAN         DEFAULT FALSE,
    restated_reason             TEXT,
    prior_value_tco2e           DECIMAL(18,4),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Primary key for hypertable
    PRIMARY KEY (performance_id, year),
    -- Constraints
    CONSTRAINT chk_p029_perf_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_perf_quarter CHECK (
        quarter IS NULL OR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    ),
    CONSTRAINT chk_p029_perf_reporting_period CHECK (
        reporting_period IN ('ANNUAL', 'SEMI_ANNUAL', 'QUARTERLY', 'MONTHLY')
    ),
    CONSTRAINT chk_p029_perf_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p029_perf_actual_emissions CHECK (
        actual_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_perf_data_quality_tier CHECK (
        data_quality_tier >= 1 AND data_quality_tier <= 5
    ),
    CONSTRAINT chk_p029_perf_data_quality_score CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p029_perf_completeness CHECK (
        data_completeness_pct >= 0 AND data_completeness_pct <= 100
    ),
    CONSTRAINT chk_p029_perf_verification CHECK (
        verification_status IN ('UNVERIFIED', 'SELF_DECLARED', 'INTERNALLY_VERIFIED',
                                'THIRD_PARTY_LIMITED', 'THIRD_PARTY_REASONABLE', 'SBTI_VALIDATED')
    ),
    CONSTRAINT chk_p029_perf_assurance CHECK (
        assurance_level IS NULL OR assurance_level IN ('NONE', 'LIMITED', 'REASONABLE')
    ),
    CONSTRAINT chk_p029_perf_calc_method CHECK (
        calculation_method IS NULL OR calculation_method IN (
            'MEASURED', 'CALCULATED', 'SPEND_BASED', 'ACTIVITY_BASED',
            'SUPPLIER_SPECIFIC', 'AVERAGE_DATA', 'HYBRID', 'PROXY', 'DEFAULT'
        )
    ),
    CONSTRAINT chk_p029_perf_category_number CHECK (
        category_number IS NULL OR (category_number >= 1 AND category_number <= 15)
    )
);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable
-- ---------------------------------------------------------------------------
SELECT create_hypertable(
    'pack029_interim_targets.gl_actual_performance',
    'year',
    chunk_time_interval => 5,
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_perf_tenant           ON pack029_interim_targets.gl_actual_performance(tenant_id);
CREATE INDEX idx_p029_perf_org              ON pack029_interim_targets.gl_actual_performance(organization_id);
CREATE INDEX idx_p029_perf_target           ON pack029_interim_targets.gl_actual_performance(target_id);
CREATE INDEX idx_p029_perf_org_year_qtr     ON pack029_interim_targets.gl_actual_performance(organization_id, year, quarter);
CREATE INDEX idx_p029_perf_org_year_scope   ON pack029_interim_targets.gl_actual_performance(organization_id, year, quarter, scope);
CREATE INDEX idx_p029_perf_scope            ON pack029_interim_targets.gl_actual_performance(scope);
CREATE INDEX idx_p029_perf_category         ON pack029_interim_targets.gl_actual_performance(category);
CREATE INDEX idx_p029_perf_verification     ON pack029_interim_targets.gl_actual_performance(verification_status);
CREATE INDEX idx_p029_perf_unverified       ON pack029_interim_targets.gl_actual_performance(organization_id, year) WHERE verification_status = 'UNVERIFIED';
CREATE INDEX idx_p029_perf_data_quality     ON pack029_interim_targets.gl_actual_performance(data_quality_tier);
CREATE INDEX idx_p029_perf_low_quality      ON pack029_interim_targets.gl_actual_performance(organization_id, data_quality_tier) WHERE data_quality_tier >= 4;
CREATE INDEX idx_p029_perf_mrv_calc         ON pack029_interim_targets.gl_actual_performance(mrv_calculation_id) WHERE mrv_calculation_id IS NOT NULL;
CREATE INDEX idx_p029_perf_mrv_agent        ON pack029_interim_targets.gl_actual_performance(mrv_agent_id) WHERE mrv_agent_id IS NOT NULL;
CREATE INDEX idx_p029_perf_reported_date    ON pack029_interim_targets.gl_actual_performance(reported_date DESC);
CREATE INDEX idx_p029_perf_restated         ON pack029_interim_targets.gl_actual_performance(organization_id, is_restated) WHERE is_restated = TRUE;
CREATE INDEX idx_p029_perf_active           ON pack029_interim_targets.gl_actual_performance(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_perf_created          ON pack029_interim_targets.gl_actual_performance(created_at DESC);
CREATE INDEX idx_p029_perf_metadata         ON pack029_interim_targets.gl_actual_performance USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_actual_performance_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_actual_performance
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_actual_performance ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_perf_tenant_isolation
    ON pack029_interim_targets.gl_actual_performance
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_perf_service_bypass
    ON pack029_interim_targets.gl_actual_performance
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_actual_performance TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_actual_performance IS
    'Actual emissions performance records with quarterly granularity, GHG scope/category breakdowns, data quality tiers (1-5), verification status, and MRV agent calculation linkage for target progress tracking.';

COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.performance_id IS 'Unique actual performance record identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.organization_id IS 'Reference to the organization reporting performance.';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.year IS 'Reporting year for this performance record.';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.quarter IS 'Reporting quarter: Q1, Q2, Q3, Q4 (NULL for annual records).';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.scope IS 'GHG Protocol scope: SCOPE_1, SCOPE_2, SCOPE_3, SCOPE_1_2, SCOPE_1_2_3.';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.category IS 'Emissions category (e.g., Purchased Goods & Services for Scope 3 Cat 1).';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.actual_emissions_tco2e IS 'Actual reported emissions in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.data_quality_tier IS 'Data quality tier (1=highest/measured, 5=lowest/estimated).';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.verification_status IS 'Verification level: UNVERIFIED through SBTI_VALIDATED.';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.mrv_calculation_id IS 'Reference to MRV agent calculation that produced this data.';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.is_restated IS 'Whether this record is a restatement of previously reported data.';
COMMENT ON COLUMN pack029_interim_targets.gl_actual_performance.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

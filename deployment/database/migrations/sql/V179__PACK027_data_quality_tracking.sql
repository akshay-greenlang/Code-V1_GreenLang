-- =============================================================================
-- V179: PACK-027 Enterprise Net Zero - Data Quality Tracking
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    014 of 015
-- Date:         March 2026
--
-- Data quality scoring and improvement tracking with per-category per-entity
-- completeness, accuracy, and timeliness metrics. Supports the GHG Protocol
-- data quality hierarchy (1-5) and financial-grade (+/-3%) accuracy targets.
--
-- Tables (1):
--   1. pack027_enterprise_net_zero.gl_data_quality_scores
--
-- Previous: V178__PACK027_board_reporting.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_data_quality_scores
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_data_quality_scores (
    score_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Period
    reporting_year              INTEGER         NOT NULL,
    assessment_date             DATE            NOT NULL DEFAULT CURRENT_DATE,
    -- Category
    category                    VARCHAR(50)     NOT NULL,
    sub_category                VARCHAR(100),
    entity_ref                  UUID,
    entity_name                 VARCHAR(500),
    -- GHG Protocol DQ hierarchy (1=best, 5=worst)
    ghg_dq_level                INTEGER,
    -- Completeness
    completeness_pct            DECIMAL(6,2)    NOT NULL,
    data_points_expected        INTEGER,
    data_points_received        INTEGER,
    data_points_missing         INTEGER,
    missing_data_handling       VARCHAR(30),
    -- Accuracy
    accuracy_pct                DECIMAL(6,2)    NOT NULL,
    accuracy_method             VARCHAR(30),
    variance_from_benchmark_pct DECIMAL(8,2),
    cross_validation_pass       BOOLEAN,
    -- Timeliness
    timeliness_score            DECIMAL(6,2)    NOT NULL,
    avg_data_age_days           INTEGER,
    max_data_age_days           INTEGER,
    real_time_pct               DECIMAL(6,2),
    -- Source quality
    primary_data_pct            DECIMAL(6,2),
    secondary_data_pct          DECIMAL(6,2),
    estimated_data_pct          DECIMAL(6,2),
    source_breakdown            JSONB           DEFAULT '{}',
    -- Emission factor quality
    ef_quality_score            DECIMAL(5,2),
    ef_source                   VARCHAR(100),
    ef_vintage_year             INTEGER,
    ef_geographic_relevance     VARCHAR(20),
    ef_temporal_relevance       VARCHAR(20),
    -- Improvement tracking
    previous_year_dq_level      INTEGER,
    improvement_from_prior      DECIMAL(6,2),
    improvement_target          DECIMAL(6,2),
    improvement_plan            TEXT,
    improvement_actions         JSONB           DEFAULT '{}',
    -- Weighted score
    weighted_dq_score           DECIMAL(5,2),
    weight_in_total             DECIMAL(6,4),
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_dq_reporting_year CHECK (
        reporting_year >= 2015 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p027_dq_category CHECK (
        category IN ('SCOPE_1_STATIONARY', 'SCOPE_1_MOBILE', 'SCOPE_1_PROCESS',
                      'SCOPE_1_FUGITIVE', 'SCOPE_1_REFRIGERANT', 'SCOPE_1_LAND_USE',
                      'SCOPE_1_WASTE', 'SCOPE_1_AGRICULTURAL',
                      'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_2_STEAM', 'SCOPE_2_COOLING',
                      'SCOPE_3_CAT1', 'SCOPE_3_CAT2', 'SCOPE_3_CAT3', 'SCOPE_3_CAT4',
                      'SCOPE_3_CAT5', 'SCOPE_3_CAT6', 'SCOPE_3_CAT7', 'SCOPE_3_CAT8',
                      'SCOPE_3_CAT9', 'SCOPE_3_CAT10', 'SCOPE_3_CAT11', 'SCOPE_3_CAT12',
                      'SCOPE_3_CAT13', 'SCOPE_3_CAT14', 'SCOPE_3_CAT15',
                      'OVERALL', 'CONSOLIDATION')
    ),
    CONSTRAINT chk_p027_dq_ghg_level CHECK (
        ghg_dq_level IS NULL OR (ghg_dq_level >= 1 AND ghg_dq_level <= 5)
    ),
    CONSTRAINT chk_p027_dq_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p027_dq_accuracy CHECK (
        accuracy_pct >= 0 AND accuracy_pct <= 100
    ),
    CONSTRAINT chk_p027_dq_timeliness CHECK (
        timeliness_score >= 0 AND timeliness_score <= 100
    ),
    CONSTRAINT chk_p027_dq_missing_handling CHECK (
        missing_data_handling IS NULL OR missing_data_handling IN (
            'ZERO_FILL', 'ESTIMATION', 'PROXY', 'INTERPOLATION', 'EXTRAPOLATION',
            'INDUSTRY_AVERAGE', 'EXCLUDED', 'CARRY_FORWARD'
        )
    ),
    CONSTRAINT chk_p027_dq_accuracy_method CHECK (
        accuracy_method IS NULL OR accuracy_method IN (
            'CROSS_VALIDATION', 'BENCHMARK_COMPARISON', 'TREND_ANALYSIS',
            'EXPERT_REVIEW', 'AUTOMATED_CHECK', 'AUDIT_VERIFIED'
        )
    ),
    CONSTRAINT chk_p027_dq_ef_geo_relevance CHECK (
        ef_geographic_relevance IS NULL OR ef_geographic_relevance IN (
            'SITE_SPECIFIC', 'COUNTRY', 'REGIONAL', 'CONTINENTAL', 'GLOBAL'
        )
    ),
    CONSTRAINT chk_p027_dq_ef_temporal CHECK (
        ef_temporal_relevance IS NULL OR ef_temporal_relevance IN (
            'CURRENT_YEAR', 'WITHIN_3_YEARS', 'WITHIN_5_YEARS', 'OLDER_THAN_5_YEARS'
        )
    ),
    CONSTRAINT chk_p027_dq_pct_sum CHECK (
        (primary_data_pct IS NULL OR secondary_data_pct IS NULL OR estimated_data_pct IS NULL)
        OR (primary_data_pct + secondary_data_pct + estimated_data_pct BETWEEN 99.0 AND 101.0)
    ),
    CONSTRAINT chk_p027_dq_prev_level CHECK (
        previous_year_dq_level IS NULL OR (previous_year_dq_level >= 1 AND previous_year_dq_level <= 5)
    ),
    CONSTRAINT uq_p027_dq_company_year_category_entity UNIQUE (company_id, reporting_year, category, entity_ref)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_dq_company            ON pack027_enterprise_net_zero.gl_data_quality_scores(company_id);
CREATE INDEX idx_p027_dq_tenant             ON pack027_enterprise_net_zero.gl_data_quality_scores(tenant_id);
CREATE INDEX idx_p027_dq_year               ON pack027_enterprise_net_zero.gl_data_quality_scores(reporting_year);
CREATE INDEX idx_p027_dq_category           ON pack027_enterprise_net_zero.gl_data_quality_scores(category);
CREATE INDEX idx_p027_dq_company_year       ON pack027_enterprise_net_zero.gl_data_quality_scores(company_id, reporting_year);
CREATE INDEX idx_p027_dq_ghg_level          ON pack027_enterprise_net_zero.gl_data_quality_scores(ghg_dq_level);
CREATE INDEX idx_p027_dq_completeness       ON pack027_enterprise_net_zero.gl_data_quality_scores(completeness_pct);
CREATE INDEX idx_p027_dq_accuracy           ON pack027_enterprise_net_zero.gl_data_quality_scores(accuracy_pct);
CREATE INDEX idx_p027_dq_timeliness         ON pack027_enterprise_net_zero.gl_data_quality_scores(timeliness_score);
CREATE INDEX idx_p027_dq_entity             ON pack027_enterprise_net_zero.gl_data_quality_scores(entity_ref);
CREATE INDEX idx_p027_dq_weighted           ON pack027_enterprise_net_zero.gl_data_quality_scores(weighted_dq_score);
CREATE INDEX idx_p027_dq_assessment         ON pack027_enterprise_net_zero.gl_data_quality_scores(assessment_date DESC);
CREATE INDEX idx_p027_dq_ef_source          ON pack027_enterprise_net_zero.gl_data_quality_scores(ef_source);
CREATE INDEX idx_p027_dq_improvement        ON pack027_enterprise_net_zero.gl_data_quality_scores USING GIN(improvement_actions);
CREATE INDEX idx_p027_dq_source             ON pack027_enterprise_net_zero.gl_data_quality_scores USING GIN(source_breakdown);
CREATE INDEX idx_p027_dq_created            ON pack027_enterprise_net_zero.gl_data_quality_scores(created_at DESC);
CREATE INDEX idx_p027_dq_metadata           ON pack027_enterprise_net_zero.gl_data_quality_scores USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_data_quality_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_data_quality_scores
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_data_quality_scores ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_dq_tenant_isolation
    ON pack027_enterprise_net_zero.gl_data_quality_scores
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_dq_service_bypass
    ON pack027_enterprise_net_zero.gl_data_quality_scores
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_data_quality_scores TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_data_quality_scores IS
    'Data quality scoring and improvement tracking per GHG Protocol hierarchy with completeness, accuracy, timeliness, and emission factor quality per category per entity.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.score_id IS 'Unique data quality score record identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.category IS 'Emission category being scored (SCOPE_1_STATIONARY, SCOPE_3_CAT1, etc.).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.ghg_dq_level IS 'GHG Protocol data quality hierarchy level: 1 (best/supplier-specific) to 5 (worst/proxy).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.completeness_pct IS 'Data completeness as percentage (0-100%).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.accuracy_pct IS 'Data accuracy as percentage (0-100%).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.timeliness_score IS 'Data timeliness score (0-100).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.primary_data_pct IS 'Percentage of data from primary sources (direct measurement/supplier-specific).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.improvement_from_prior IS 'DQ level improvement from previous reporting year.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_data_quality_scores.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

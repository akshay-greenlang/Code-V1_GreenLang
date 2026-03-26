-- =============================================================================
-- V342: PACK-042 Scope 3 Starter Pack - Data Quality Assessment & Tracking
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates data quality assessment and tracking tables implementing the
-- GHG Protocol Scope 3 data quality framework. Supports the five Data
-- Quality Indicators (DQIs): technological representativeness, temporal
-- correlation, geographical correlation, completeness, and reliability.
-- Calculates a weighted Data Quality Rating (DQR) per category and
-- tracks improvement actions and historical trends.
--
-- Tables (4):
--   1. ghg_accounting_scope3.quality_assessments
--   2. ghg_accounting_scope3.category_quality_scores
--   3. ghg_accounting_scope3.quality_improvements
--   4. ghg_accounting_scope3.quality_trends
--
-- Also includes: indexes, RLS, comments.
-- Previous: V341__pack042_supplier_engagement.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.quality_assessments
-- =============================================================================
-- Top-level data quality assessment for a Scope 3 inventory. Produces an
-- overall DQR (Data Quality Rating) score on a 1-5 scale where 1 is best.
-- Follows the PEFCR/PEF methodology for data quality scoring.

CREATE TABLE ghg_accounting_scope3.quality_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    -- Assessment header
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessment_version          INTEGER         NOT NULL DEFAULT 1,
    assessor                    VARCHAR(255),
    methodology                 VARCHAR(50)     NOT NULL DEFAULT 'GHG_PROTOCOL_DQI',
    -- Overall results
    overall_dqr                 DECIMAL(3,1)    NOT NULL DEFAULT 3.0,
    overall_rating              VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    categories_assessed         INTEGER         NOT NULL DEFAULT 0,
    -- DQI averages across all categories
    avg_technological_score     DECIMAL(3,1),
    avg_temporal_score          DECIMAL(3,1),
    avg_geographical_score      DECIMAL(3,1),
    avg_completeness_score      DECIMAL(3,1),
    avg_reliability_score       DECIMAL(3,1),
    -- Primary data statistics
    pct_primary_data            DECIMAL(5,2)    DEFAULT 0,
    pct_secondary_data          DECIMAL(5,2)    DEFAULT 0,
    pct_estimated_data          DECIMAL(5,2)    DEFAULT 0,
    -- Improvement summary
    total_improvements_needed   INTEGER         DEFAULT 0,
    critical_improvements       INTEGER         DEFAULT 0,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_qa_dqr CHECK (
        overall_dqr >= 1.0 AND overall_dqr <= 5.0
    ),
    CONSTRAINT chk_p042_qa_rating CHECK (
        overall_rating IN ('VERY_HIGH', 'HIGH', 'MODERATE', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p042_qa_methodology CHECK (
        methodology IN ('GHG_PROTOCOL_DQI', 'PEF_DQR', 'ISO_14044', 'CUSTOM')
    ),
    CONSTRAINT chk_p042_qa_version CHECK (
        assessment_version >= 1
    ),
    CONSTRAINT chk_p042_qa_categories CHECK (
        categories_assessed >= 0 AND categories_assessed <= 15
    ),
    CONSTRAINT chk_p042_qa_tech CHECK (
        avg_technological_score IS NULL OR (avg_technological_score >= 1.0 AND avg_technological_score <= 5.0)
    ),
    CONSTRAINT chk_p042_qa_temporal CHECK (
        avg_temporal_score IS NULL OR (avg_temporal_score >= 1.0 AND avg_temporal_score <= 5.0)
    ),
    CONSTRAINT chk_p042_qa_geo CHECK (
        avg_geographical_score IS NULL OR (avg_geographical_score >= 1.0 AND avg_geographical_score <= 5.0)
    ),
    CONSTRAINT chk_p042_qa_completeness CHECK (
        avg_completeness_score IS NULL OR (avg_completeness_score >= 1.0 AND avg_completeness_score <= 5.0)
    ),
    CONSTRAINT chk_p042_qa_reliability CHECK (
        avg_reliability_score IS NULL OR (avg_reliability_score >= 1.0 AND avg_reliability_score <= 5.0)
    ),
    CONSTRAINT chk_p042_qa_primary_pct CHECK (
        pct_primary_data IS NULL OR (pct_primary_data >= 0 AND pct_primary_data <= 100)
    ),
    CONSTRAINT chk_p042_qa_secondary_pct CHECK (
        pct_secondary_data IS NULL OR (pct_secondary_data >= 0 AND pct_secondary_data <= 100)
    ),
    CONSTRAINT chk_p042_qa_estimated_pct CHECK (
        pct_estimated_data IS NULL OR (pct_estimated_data >= 0 AND pct_estimated_data <= 100)
    ),
    CONSTRAINT chk_p042_qa_improvements CHECK (
        total_improvements_needed IS NULL OR total_improvements_needed >= 0
    ),
    CONSTRAINT chk_p042_qa_critical CHECK (
        critical_improvements IS NULL OR critical_improvements >= 0
    ),
    CONSTRAINT chk_p042_qa_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'REVIEWED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p042_qa_inventory_version UNIQUE (inventory_id, assessment_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_qa_tenant             ON ghg_accounting_scope3.quality_assessments(tenant_id);
CREATE INDEX idx_p042_qa_inventory          ON ghg_accounting_scope3.quality_assessments(inventory_id);
CREATE INDEX idx_p042_qa_date               ON ghg_accounting_scope3.quality_assessments(assessment_date DESC);
CREATE INDEX idx_p042_qa_dqr                ON ghg_accounting_scope3.quality_assessments(overall_dqr);
CREATE INDEX idx_p042_qa_rating             ON ghg_accounting_scope3.quality_assessments(overall_rating);
CREATE INDEX idx_p042_qa_status             ON ghg_accounting_scope3.quality_assessments(status);
CREATE INDEX idx_p042_qa_created            ON ghg_accounting_scope3.quality_assessments(created_at DESC);

-- Composite: inventory + latest assessment
CREATE INDEX idx_p042_qa_inv_latest         ON ghg_accounting_scope3.quality_assessments(inventory_id, assessment_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_qa_updated
    BEFORE UPDATE ON ghg_accounting_scope3.quality_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.category_quality_scores
-- =============================================================================
-- Per-category DQI scores within a quality assessment. Evaluates each of
-- the five Data Quality Indicators (technological, temporal, geographical,
-- completeness, reliability) on a 1-5 scale and calculates a weighted DQR.
-- Lower scores indicate higher data quality (1 = best, 5 = worst).

CREATE TABLE ghg_accounting_scope3.category_quality_scores (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3.quality_assessments(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- DQI scores (1 = best, 5 = worst)
    technological_score         DECIMAL(3,1)    NOT NULL DEFAULT 3.0,
    temporal_score              DECIMAL(3,1)    NOT NULL DEFAULT 3.0,
    geographical_score          DECIMAL(3,1)    NOT NULL DEFAULT 3.0,
    completeness_score          DECIMAL(3,1)    NOT NULL DEFAULT 3.0,
    reliability_score           DECIMAL(3,1)    NOT NULL DEFAULT 3.0,
    -- DQI weights (sum should = 1.0)
    technological_weight        DECIMAL(3,2)    NOT NULL DEFAULT 0.20,
    temporal_weight             DECIMAL(3,2)    NOT NULL DEFAULT 0.20,
    geographical_weight         DECIMAL(3,2)    NOT NULL DEFAULT 0.20,
    completeness_weight         DECIMAL(3,2)    NOT NULL DEFAULT 0.20,
    reliability_weight          DECIMAL(3,2)    NOT NULL DEFAULT 0.20,
    -- Weighted DQR
    weighted_dqr                DECIMAL(3,1)    NOT NULL DEFAULT 3.0,
    quality_tier                VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    -- Data source summary
    primary_data_pct            DECIMAL(5,2)    DEFAULT 0,
    emission_factor_source      VARCHAR(100),
    emission_factor_age_years   INTEGER,
    geographic_match            VARCHAR(30),
    -- Justification
    technological_justification TEXT,
    temporal_justification      TEXT,
    geographical_justification  TEXT,
    completeness_justification  TEXT,
    reliability_justification   TEXT,
    -- Metadata
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_cqs_tech CHECK (technological_score >= 1.0 AND technological_score <= 5.0),
    CONSTRAINT chk_p042_cqs_temp CHECK (temporal_score >= 1.0 AND temporal_score <= 5.0),
    CONSTRAINT chk_p042_cqs_geo CHECK (geographical_score >= 1.0 AND geographical_score <= 5.0),
    CONSTRAINT chk_p042_cqs_comp CHECK (completeness_score >= 1.0 AND completeness_score <= 5.0),
    CONSTRAINT chk_p042_cqs_rel CHECK (reliability_score >= 1.0 AND reliability_score <= 5.0),
    CONSTRAINT chk_p042_cqs_dqr CHECK (weighted_dqr >= 1.0 AND weighted_dqr <= 5.0),
    CONSTRAINT chk_p042_cqs_tier CHECK (
        quality_tier IN ('VERY_HIGH', 'HIGH', 'MODERATE', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p042_cqs_primary CHECK (
        primary_data_pct IS NULL OR (primary_data_pct >= 0 AND primary_data_pct <= 100)
    ),
    CONSTRAINT chk_p042_cqs_geo_match CHECK (
        geographic_match IS NULL OR geographic_match IN (
            'EXACT', 'COUNTRY', 'REGION', 'CONTINENT', 'GLOBAL'
        )
    ),
    CONSTRAINT chk_p042_cqs_ef_age CHECK (
        emission_factor_age_years IS NULL OR emission_factor_age_years >= 0
    ),
    CONSTRAINT uq_p042_cqs_assessment_category UNIQUE (assessment_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_cqs_tenant            ON ghg_accounting_scope3.category_quality_scores(tenant_id);
CREATE INDEX idx_p042_cqs_assessment        ON ghg_accounting_scope3.category_quality_scores(assessment_id);
CREATE INDEX idx_p042_cqs_category          ON ghg_accounting_scope3.category_quality_scores(category);
CREATE INDEX idx_p042_cqs_dqr               ON ghg_accounting_scope3.category_quality_scores(weighted_dqr);
CREATE INDEX idx_p042_cqs_tier              ON ghg_accounting_scope3.category_quality_scores(quality_tier);
CREATE INDEX idx_p042_cqs_created           ON ghg_accounting_scope3.category_quality_scores(created_at DESC);

-- Composite: assessment + DQR for ranked quality view
CREATE INDEX idx_p042_cqs_assessment_dqr    ON ghg_accounting_scope3.category_quality_scores(assessment_id, weighted_dqr);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_cqs_updated
    BEFORE UPDATE ON ghg_accounting_scope3.category_quality_scores
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.quality_improvements
-- =============================================================================
-- Data quality improvement actions tied to specific categories. Each action
-- describes what needs to change (e.g., switch from EEIO to supplier data),
-- the effort and impact levels, and tracks implementation status.

CREATE TABLE ghg_accounting_scope3.quality_improvements (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3.quality_assessments(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Improvement details
    action_description          TEXT            NOT NULL,
    target_dqi                  VARCHAR(20)     NOT NULL,
    current_score               DECIMAL(3,1)    NOT NULL,
    target_score                DECIMAL(3,1)    NOT NULL,
    -- Classification
    effort_level                VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    impact_level                VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    priority                    INTEGER         NOT NULL DEFAULT 3,
    -- Implementation
    status                      VARCHAR(30)     NOT NULL DEFAULT 'IDENTIFIED',
    assigned_to                 VARCHAR(255),
    target_date                 DATE,
    completed_date              DATE,
    actual_score                DECIMAL(3,1),
    -- Cost
    estimated_cost              NUMERIC(18,2),
    cost_currency               VARCHAR(3)      DEFAULT 'USD',
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_qi_target_dqi CHECK (
        target_dqi IN (
            'TECHNOLOGICAL', 'TEMPORAL', 'GEOGRAPHICAL',
            'COMPLETENESS', 'RELIABILITY', 'OVERALL'
        )
    ),
    CONSTRAINT chk_p042_qi_current CHECK (current_score >= 1.0 AND current_score <= 5.0),
    CONSTRAINT chk_p042_qi_target CHECK (target_score >= 1.0 AND target_score <= 5.0),
    CONSTRAINT chk_p042_qi_target_better CHECK (target_score <= current_score),
    CONSTRAINT chk_p042_qi_actual CHECK (
        actual_score IS NULL OR (actual_score >= 1.0 AND actual_score <= 5.0)
    ),
    CONSTRAINT chk_p042_qi_effort CHECK (
        effort_level IN ('VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p042_qi_impact CHECK (
        impact_level IN ('VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p042_qi_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p042_qi_status CHECK (
        status IN (
            'IDENTIFIED', 'PLANNED', 'IN_PROGRESS', 'COMPLETED',
            'DEFERRED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p042_qi_cost CHECK (
        estimated_cost IS NULL OR estimated_cost >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_qi_tenant             ON ghg_accounting_scope3.quality_improvements(tenant_id);
CREATE INDEX idx_p042_qi_assessment         ON ghg_accounting_scope3.quality_improvements(assessment_id);
CREATE INDEX idx_p042_qi_category           ON ghg_accounting_scope3.quality_improvements(category);
CREATE INDEX idx_p042_qi_dqi                ON ghg_accounting_scope3.quality_improvements(target_dqi);
CREATE INDEX idx_p042_qi_priority           ON ghg_accounting_scope3.quality_improvements(priority);
CREATE INDEX idx_p042_qi_effort             ON ghg_accounting_scope3.quality_improvements(effort_level);
CREATE INDEX idx_p042_qi_impact             ON ghg_accounting_scope3.quality_improvements(impact_level);
CREATE INDEX idx_p042_qi_status             ON ghg_accounting_scope3.quality_improvements(status);
CREATE INDEX idx_p042_qi_target_date        ON ghg_accounting_scope3.quality_improvements(target_date);
CREATE INDEX idx_p042_qi_created            ON ghg_accounting_scope3.quality_improvements(created_at DESC);

-- Composite: open improvements by priority
CREATE INDEX idx_p042_qi_open_priority      ON ghg_accounting_scope3.quality_improvements(assessment_id, priority)
    WHERE status IN ('IDENTIFIED', 'PLANNED', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_qi_updated
    BEFORE UPDATE ON ghg_accounting_scope3.quality_improvements
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.quality_trends
-- =============================================================================
-- Historical DQR tracking per category across reporting periods. Enables
-- visualization of data quality improvement over time and supports
-- year-over-year quality comparison for continuous improvement programs.

CREATE TABLE ghg_accounting_scope3.quality_trends (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Period
    period                      VARCHAR(20)     NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    -- Scores
    dqr_score                   DECIMAL(3,1)    NOT NULL,
    quality_tier                VARCHAR(20)     NOT NULL,
    -- DQI breakdown
    technological_score         DECIMAL(3,1),
    temporal_score              DECIMAL(3,1),
    geographical_score          DECIMAL(3,1),
    completeness_score          DECIMAL(3,1),
    reliability_score           DECIMAL(3,1),
    -- Data source
    primary_data_pct            DECIMAL(5,2)    DEFAULT 0,
    methodology_tier            ghg_accounting_scope3.methodology_tier_type,
    -- Change tracking
    dqr_change_from_prior       DECIMAL(3,1),
    change_direction            VARCHAR(10),
    -- Metadata
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_qt_dqr CHECK (
        dqr_score >= 1.0 AND dqr_score <= 5.0
    ),
    CONSTRAINT chk_p042_qt_tier CHECK (
        quality_tier IN ('VERY_HIGH', 'HIGH', 'MODERATE', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p042_qt_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p042_qt_primary CHECK (
        primary_data_pct IS NULL OR (primary_data_pct >= 0 AND primary_data_pct <= 100)
    ),
    CONSTRAINT chk_p042_qt_change CHECK (
        change_direction IS NULL OR change_direction IN ('IMPROVED', 'STABLE', 'DECLINED')
    ),
    CONSTRAINT uq_p042_qt_inv_category_period UNIQUE (inventory_id, category, period)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_qt_tenant             ON ghg_accounting_scope3.quality_trends(tenant_id);
CREATE INDEX idx_p042_qt_inventory          ON ghg_accounting_scope3.quality_trends(inventory_id);
CREATE INDEX idx_p042_qt_category           ON ghg_accounting_scope3.quality_trends(category);
CREATE INDEX idx_p042_qt_year               ON ghg_accounting_scope3.quality_trends(reporting_year);
CREATE INDEX idx_p042_qt_dqr                ON ghg_accounting_scope3.quality_trends(dqr_score);
CREATE INDEX idx_p042_qt_tier               ON ghg_accounting_scope3.quality_trends(quality_tier);
CREATE INDEX idx_p042_qt_created            ON ghg_accounting_scope3.quality_trends(created_at DESC);

-- Composite: category + year for trend analysis
CREATE INDEX idx_p042_qt_cat_year           ON ghg_accounting_scope3.quality_trends(tenant_id, category, reporting_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_qt_updated
    BEFORE UPDATE ON ghg_accounting_scope3.quality_trends
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3.quality_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.category_quality_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.quality_improvements ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.quality_trends ENABLE ROW LEVEL SECURITY;

CREATE POLICY p042_qa_tenant_isolation ON ghg_accounting_scope3.quality_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_qa_service_bypass ON ghg_accounting_scope3.quality_assessments
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_cqs_tenant_isolation ON ghg_accounting_scope3.category_quality_scores
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_cqs_service_bypass ON ghg_accounting_scope3.category_quality_scores
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_qi_tenant_isolation ON ghg_accounting_scope3.quality_improvements
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_qi_service_bypass ON ghg_accounting_scope3.quality_improvements
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_qt_tenant_isolation ON ghg_accounting_scope3.quality_trends
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_qt_service_bypass ON ghg_accounting_scope3.quality_trends
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.quality_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.category_quality_scores TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.quality_improvements TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.quality_trends TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3.quality_assessments IS
    'Top-level data quality assessment for a Scope 3 inventory with overall DQR score and summary statistics across all categories.';
COMMENT ON TABLE ghg_accounting_scope3.category_quality_scores IS
    'Per-category DQI scores (technological, temporal, geographical, completeness, reliability) on 1-5 scale with weighted DQR calculation.';
COMMENT ON TABLE ghg_accounting_scope3.quality_improvements IS
    'Data quality improvement actions tied to specific categories and DQIs with effort, impact, priority, and implementation tracking.';
COMMENT ON TABLE ghg_accounting_scope3.quality_trends IS
    'Historical DQR tracking per category across reporting periods for continuous improvement visualization.';

COMMENT ON COLUMN ghg_accounting_scope3.quality_assessments.overall_dqr IS 'Overall Data Quality Rating on 1-5 scale (1 = highest quality). Weighted average of all category DQRs.';
COMMENT ON COLUMN ghg_accounting_scope3.category_quality_scores.technological_score IS 'Technological representativeness (1-5): how well the emission factors represent the actual technology used.';
COMMENT ON COLUMN ghg_accounting_scope3.category_quality_scores.temporal_score IS 'Temporal correlation (1-5): how recent the emission factors and activity data are relative to the reporting period.';
COMMENT ON COLUMN ghg_accounting_scope3.category_quality_scores.geographical_score IS 'Geographical correlation (1-5): how well the factors match the geography where activities occur.';
COMMENT ON COLUMN ghg_accounting_scope3.category_quality_scores.completeness_score IS 'Data completeness (1-5): percentage of relevant activity data captured versus total estimated.';
COMMENT ON COLUMN ghg_accounting_scope3.category_quality_scores.reliability_score IS 'Data reliability (1-5): verification status and trustworthiness of the data source.';
COMMENT ON COLUMN ghg_accounting_scope3.category_quality_scores.weighted_dqr IS 'Weighted Data Quality Rating: sum(DQI_score * DQI_weight) across all five indicators.';

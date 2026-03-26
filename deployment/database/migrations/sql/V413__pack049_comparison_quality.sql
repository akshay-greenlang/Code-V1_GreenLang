-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V413 - Comparison & Quality
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates comparison and quality management tables for site-level
-- performance analysis. Site KPIs track intensity metrics and rankings;
-- quality scores assess data completeness, accuracy, and consistency
-- across multiple dimensions.
--
-- Tables (4):
--   1. ghg_multisite.gl_ms_site_kpis
--   2. ghg_multisite.gl_ms_site_rankings
--   3. ghg_multisite.gl_ms_quality_scores
--   4. ghg_multisite.gl_ms_quality_dimensions
--
-- Also includes: indexes, RLS, comments.
-- Previous: V412__pack049_allocation.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_site_kpis
-- =============================================================================
-- Key performance indicators computed per site per period. KPIs include
-- emission intensity ratios (tCO2e per unit of activity) and absolute
-- metrics with year-on-year change tracking.

CREATE TABLE ghg_multisite.gl_ms_site_kpis (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    kpi_type                    VARCHAR(50)     NOT NULL,
    kpi_name                    VARCHAR(255)    NOT NULL,
    numerator_value             NUMERIC(20,6)   NOT NULL,
    numerator_unit              VARCHAR(50)     NOT NULL DEFAULT 'tCO2e',
    denominator_value           NUMERIC(20,6),
    denominator_unit            VARCHAR(50),
    kpi_value                   NUMERIC(20,6)   NOT NULL,
    kpi_unit                    VARCHAR(100)    NOT NULL,
    prior_period_value          NUMERIC(20,6),
    yoy_change_pct              NUMERIC(10,4),
    yoy_change_absolute         NUMERIC(20,6),
    target_value                NUMERIC(20,6),
    target_gap_pct              NUMERIC(10,4),
    is_below_target             BOOLEAN,
    scope_coverage              VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_1_2',
    data_quality_tier           INTEGER,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_kpi_type CHECK (
        kpi_type IN (
            'INTENSITY_PER_M2', 'INTENSITY_PER_FTE', 'INTENSITY_PER_REVENUE',
            'INTENSITY_PER_UNIT', 'INTENSITY_PER_MWH', 'INTENSITY_PER_HOUR',
            'ABSOLUTE_EMISSIONS', 'ENERGY_CONSUMPTION'
        )
    ),
    CONSTRAINT chk_p049_kpi_scope CHECK (
        scope_coverage IN (
            'SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2',
            'SCOPE_1_2_3', 'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p049_kpi_tier CHECK (
        data_quality_tier IS NULL OR (data_quality_tier >= 1 AND data_quality_tier <= 5)
    ),
    CONSTRAINT chk_p049_kpi_num CHECK (numerator_value >= 0),
    CONSTRAINT chk_p049_kpi_denom CHECK (
        denominator_value IS NULL OR denominator_value > 0
    ),
    CONSTRAINT uq_p049_kpi_site_period_type UNIQUE (site_id, period_id, kpi_type, scope_coverage)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_kpi_tenant         ON ghg_multisite.gl_ms_site_kpis(tenant_id);
CREATE INDEX idx_p049_kpi_config         ON ghg_multisite.gl_ms_site_kpis(config_id);
CREATE INDEX idx_p049_kpi_period         ON ghg_multisite.gl_ms_site_kpis(period_id);
CREATE INDEX idx_p049_kpi_site           ON ghg_multisite.gl_ms_site_kpis(site_id);
CREATE INDEX idx_p049_kpi_type           ON ghg_multisite.gl_ms_site_kpis(kpi_type);
CREATE INDEX idx_p049_kpi_scope          ON ghg_multisite.gl_ms_site_kpis(scope_coverage);
CREATE INDEX idx_p049_kpi_value          ON ghg_multisite.gl_ms_site_kpis(kpi_value);
CREATE INDEX idx_p049_kpi_target         ON ghg_multisite.gl_ms_site_kpis(site_id)
    WHERE target_value IS NOT NULL;
CREATE INDEX idx_p049_kpi_below_target   ON ghg_multisite.gl_ms_site_kpis(site_id)
    WHERE is_below_target = false;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_site_kpis ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_kpi_tenant_isolation ON ghg_multisite.gl_ms_site_kpis
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_site_rankings
-- =============================================================================
-- Site rankings within peer groups and overall portfolio. Rankings are
-- computed per KPI type and can be filtered by facility type, region,
-- or custom group.

CREATE TABLE ghg_multisite.gl_ms_site_rankings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    kpi_id                      UUID            REFERENCES ghg_multisite.gl_ms_site_kpis(id) ON DELETE SET NULL,
    ranking_scope               VARCHAR(30)     NOT NULL DEFAULT 'PORTFOLIO',
    group_id                    UUID            REFERENCES ghg_multisite.gl_ms_site_groups(id) ON DELETE SET NULL,
    kpi_type                    VARCHAR(50)     NOT NULL,
    kpi_value                   NUMERIC(20,6)   NOT NULL,
    rank_position               INTEGER         NOT NULL,
    total_in_group              INTEGER         NOT NULL,
    percentile                  NUMERIC(10,4)   NOT NULL,
    quartile                    INTEGER         NOT NULL,
    is_best_in_class            BOOLEAN         NOT NULL DEFAULT false,
    is_worst_in_class           BOOLEAN         NOT NULL DEFAULT false,
    gap_to_best_pct             NUMERIC(10,4),
    gap_to_median_pct           NUMERIC(10,4),
    prior_rank_position         INTEGER,
    rank_change                 INTEGER,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_rank_scope CHECK (
        ranking_scope IN (
            'PORTFOLIO', 'PEER_GROUP', 'FACILITY_TYPE', 'REGION',
            'BUSINESS_UNIT', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p049_rank_kpi_type CHECK (
        kpi_type IN (
            'INTENSITY_PER_M2', 'INTENSITY_PER_FTE', 'INTENSITY_PER_REVENUE',
            'INTENSITY_PER_UNIT', 'INTENSITY_PER_MWH', 'INTENSITY_PER_HOUR',
            'ABSOLUTE_EMISSIONS', 'ENERGY_CONSUMPTION'
        )
    ),
    CONSTRAINT chk_p049_rank_pos CHECK (rank_position >= 1),
    CONSTRAINT chk_p049_rank_total CHECK (total_in_group >= 1),
    CONSTRAINT chk_p049_rank_pos_valid CHECK (rank_position <= total_in_group),
    CONSTRAINT chk_p049_rank_pctile CHECK (
        percentile >= 0 AND percentile <= 100
    ),
    CONSTRAINT chk_p049_rank_quartile CHECK (
        quartile >= 1 AND quartile <= 4
    ),
    CONSTRAINT uq_p049_rank_site_period_scope UNIQUE (
        site_id, period_id, kpi_type, ranking_scope, group_id
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_rank_tenant        ON ghg_multisite.gl_ms_site_rankings(tenant_id);
CREATE INDEX idx_p049_rank_config        ON ghg_multisite.gl_ms_site_rankings(config_id);
CREATE INDEX idx_p049_rank_period        ON ghg_multisite.gl_ms_site_rankings(period_id);
CREATE INDEX idx_p049_rank_site          ON ghg_multisite.gl_ms_site_rankings(site_id);
CREATE INDEX idx_p049_rank_kpi           ON ghg_multisite.gl_ms_site_rankings(kpi_id)
    WHERE kpi_id IS NOT NULL;
CREATE INDEX idx_p049_rank_scope         ON ghg_multisite.gl_ms_site_rankings(ranking_scope);
CREATE INDEX idx_p049_rank_group         ON ghg_multisite.gl_ms_site_rankings(group_id)
    WHERE group_id IS NOT NULL;
CREATE INDEX idx_p049_rank_type          ON ghg_multisite.gl_ms_site_rankings(kpi_type);
CREATE INDEX idx_p049_rank_best          ON ghg_multisite.gl_ms_site_rankings(period_id, kpi_type)
    WHERE is_best_in_class = true;
CREATE INDEX idx_p049_rank_worst         ON ghg_multisite.gl_ms_site_rankings(period_id, kpi_type)
    WHERE is_worst_in_class = true;
CREATE INDEX idx_p049_rank_position      ON ghg_multisite.gl_ms_site_rankings(rank_position);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_site_rankings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_rank_tenant_isolation ON ghg_multisite.gl_ms_site_rankings
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_multisite.gl_ms_quality_scores
-- =============================================================================
-- Composite data quality scores per site per period. Aggregates multiple
-- quality dimensions into an overall score with PCAF-equivalent tier.

CREATE TABLE ghg_multisite.gl_ms_quality_scores (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    overall_score               NUMERIC(10,4)   NOT NULL,
    pcaf_equivalent_tier        INTEGER,
    completeness_score          NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    accuracy_score              NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    consistency_score           NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    timeliness_score            NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    transparency_score          NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    reliability_score           NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    dimension_weights           JSONB           NOT NULL DEFAULT '{}',
    issues_found                INTEGER         NOT NULL DEFAULT 0,
    critical_issues             INTEGER         NOT NULL DEFAULT 0,
    improvement_potential_pct   NUMERIC(10,4),
    prior_period_score          NUMERIC(10,4),
    score_change                NUMERIC(10,4),
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessed_by                 VARCHAR(100),
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_qs_overall CHECK (overall_score >= 0 AND overall_score <= 100),
    CONSTRAINT chk_p049_qs_pcaf CHECK (
        pcaf_equivalent_tier IS NULL OR (pcaf_equivalent_tier >= 1 AND pcaf_equivalent_tier <= 5)
    ),
    CONSTRAINT chk_p049_qs_completeness CHECK (completeness_score >= 0 AND completeness_score <= 100),
    CONSTRAINT chk_p049_qs_accuracy CHECK (accuracy_score >= 0 AND accuracy_score <= 100),
    CONSTRAINT chk_p049_qs_consistency CHECK (consistency_score >= 0 AND consistency_score <= 100),
    CONSTRAINT chk_p049_qs_timeliness CHECK (timeliness_score >= 0 AND timeliness_score <= 100),
    CONSTRAINT chk_p049_qs_transparency CHECK (transparency_score >= 0 AND transparency_score <= 100),
    CONSTRAINT chk_p049_qs_reliability CHECK (reliability_score >= 0 AND reliability_score <= 100),
    CONSTRAINT chk_p049_qs_issues CHECK (issues_found >= 0),
    CONSTRAINT chk_p049_qs_critical CHECK (critical_issues >= 0),
    CONSTRAINT chk_p049_qs_improvement CHECK (
        improvement_potential_pct IS NULL OR (improvement_potential_pct >= 0 AND improvement_potential_pct <= 100)
    ),
    CONSTRAINT uq_p049_qs_site_period UNIQUE (site_id, period_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_qs_tenant          ON ghg_multisite.gl_ms_quality_scores(tenant_id);
CREATE INDEX idx_p049_qs_config          ON ghg_multisite.gl_ms_quality_scores(config_id);
CREATE INDEX idx_p049_qs_period          ON ghg_multisite.gl_ms_quality_scores(period_id);
CREATE INDEX idx_p049_qs_site            ON ghg_multisite.gl_ms_quality_scores(site_id);
CREATE INDEX idx_p049_qs_overall         ON ghg_multisite.gl_ms_quality_scores(overall_score);
CREATE INDEX idx_p049_qs_pcaf            ON ghg_multisite.gl_ms_quality_scores(pcaf_equivalent_tier)
    WHERE pcaf_equivalent_tier IS NOT NULL;
CREATE INDEX idx_p049_qs_critical        ON ghg_multisite.gl_ms_quality_scores(site_id)
    WHERE critical_issues > 0;
CREATE INDEX idx_p049_qs_low_quality     ON ghg_multisite.gl_ms_quality_scores(period_id, overall_score)
    WHERE overall_score < 50;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_quality_scores ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_qs_tenant_isolation ON ghg_multisite.gl_ms_quality_scores
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 4: ghg_multisite.gl_ms_quality_dimensions
-- =============================================================================
-- Detailed quality dimension breakdowns per site quality assessment.
-- Each dimension has specific indicators, thresholds, and improvement
-- recommendations.

CREATE TABLE ghg_multisite.gl_ms_quality_dimensions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    quality_score_id            UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_quality_scores(id) ON DELETE CASCADE,
    dimension_name              VARCHAR(50)     NOT NULL,
    dimension_score             NUMERIC(10,4)   NOT NULL,
    weight                      NUMERIC(10,4)   NOT NULL DEFAULT 1.0000,
    weighted_score              NUMERIC(10,4)   NOT NULL,
    indicators_total            INTEGER         NOT NULL DEFAULT 0,
    indicators_passed           INTEGER         NOT NULL DEFAULT 0,
    indicators_failed           INTEGER         NOT NULL DEFAULT 0,
    pass_rate_pct               NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    threshold_min               NUMERIC(10,4)   NOT NULL DEFAULT 50.0000,
    is_above_threshold          BOOLEAN         NOT NULL DEFAULT true,
    findings                    JSONB           DEFAULT '[]',
    recommendations             JSONB           DEFAULT '[]',
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_qd_name CHECK (
        dimension_name IN (
            'COMPLETENESS', 'ACCURACY', 'CONSISTENCY',
            'TIMELINESS', 'TRANSPARENCY', 'RELIABILITY'
        )
    ),
    CONSTRAINT chk_p049_qd_score CHECK (dimension_score >= 0 AND dimension_score <= 100),
    CONSTRAINT chk_p049_qd_weight CHECK (weight > 0 AND weight <= 10),
    CONSTRAINT chk_p049_qd_weighted CHECK (weighted_score >= 0 AND weighted_score <= 1000),
    CONSTRAINT chk_p049_qd_total CHECK (indicators_total >= 0),
    CONSTRAINT chk_p049_qd_passed CHECK (indicators_passed >= 0),
    CONSTRAINT chk_p049_qd_failed CHECK (indicators_failed >= 0),
    CONSTRAINT chk_p049_qd_pass_rate CHECK (pass_rate_pct >= 0 AND pass_rate_pct <= 100),
    CONSTRAINT chk_p049_qd_threshold CHECK (threshold_min >= 0 AND threshold_min <= 100),
    CONSTRAINT uq_p049_qd_score_dim UNIQUE (quality_score_id, dimension_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_qd_tenant          ON ghg_multisite.gl_ms_quality_dimensions(tenant_id);
CREATE INDEX idx_p049_qd_score           ON ghg_multisite.gl_ms_quality_dimensions(quality_score_id);
CREATE INDEX idx_p049_qd_name            ON ghg_multisite.gl_ms_quality_dimensions(dimension_name);
CREATE INDEX idx_p049_qd_below           ON ghg_multisite.gl_ms_quality_dimensions(quality_score_id)
    WHERE is_above_threshold = false;
CREATE INDEX idx_p049_qd_findings        ON ghg_multisite.gl_ms_quality_dimensions USING gin(findings);
CREATE INDEX idx_p049_qd_recs            ON ghg_multisite.gl_ms_quality_dimensions USING gin(recommendations);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_quality_dimensions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_qd_tenant_isolation ON ghg_multisite.gl_ms_quality_dimensions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_multisite.gl_ms_site_kpis IS
    'PACK-049: Site KPIs (8 types) with intensity/absolute metrics, YoY change, and target tracking.';
COMMENT ON TABLE ghg_multisite.gl_ms_site_rankings IS
    'PACK-049: Site rankings (6 scopes) with percentile, quartile, best/worst flags, and gap analysis.';
COMMENT ON TABLE ghg_multisite.gl_ms_quality_scores IS
    'PACK-049: Composite quality scores (6 dimensions) with PCAF tier equivalent and issue counts.';
COMMENT ON TABLE ghg_multisite.gl_ms_quality_dimensions IS
    'PACK-049: Quality dimension detail (6 dimensions) with indicators, pass rates, findings, and recommendations.';

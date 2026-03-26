-- =============================================================================
-- V330: PACK-041 Scope 1-2 Complete Pack - Uncertainty Analysis
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates uncertainty analysis tables implementing GHG Protocol Chapter 6
-- (Managing Inventory Quality) and IPCC 2006 Guidelines Volume 1 Chapter 3
-- (Uncertainties). Supports both error propagation (analytical) and Monte
-- Carlo simulation methods. Tracks uncertainty by source category, identifies
-- improvement opportunities, and quantifies the contribution of each source
-- to total inventory uncertainty.
--
-- Tables (3):
--   1. ghg_scope12.uncertainty_analyses
--   2. ghg_scope12.uncertainty_sources
--   3. ghg_scope12.uncertainty_improvements
--
-- Also includes: indexes, RLS, comments.
-- Previous: V329__pack041_scope2_consolidation.sql
-- =============================================================================

SET search_path TO ghg_scope12, public;

-- =============================================================================
-- Table 1: ghg_scope12.uncertainty_analyses
-- =============================================================================
-- Top-level uncertainty analysis for an inventory (Scope 1 or Scope 2).
-- Supports IPCC Approach 1 (error propagation / analytical) and Approach 2
-- (Monte Carlo simulation). Stores confidence intervals, statistical
-- parameters, and the overall uncertainty assessment at inventory level.

CREATE TABLE ghg_scope12.uncertainty_analyses (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    inventory_type              VARCHAR(10)     NOT NULL DEFAULT 'SCOPE_1',
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    reporting_year              INTEGER         NOT NULL,
    method                      VARCHAR(30)     NOT NULL DEFAULT 'ANALYTICAL',
    confidence_level            DECIMAL(3,2)    NOT NULL DEFAULT 0.95,
    -- Analytical results (IPCC Approach 1 - error propagation)
    analytical_total_tco2e      DECIMAL(15,3),
    analytical_lower            DECIMAL(15,3),
    analytical_upper            DECIMAL(15,3),
    analytical_uncertainty_pct  DECIMAL(8,4),
    analytical_half_width       DECIMAL(15,3),
    -- Monte Carlo results (IPCC Approach 2)
    mc_iterations               INTEGER,
    mc_seed                     INTEGER,
    mc_mean                     DECIMAL(15,3),
    mc_median                   DECIMAL(15,3),
    mc_mode                     DECIMAL(15,3),
    mc_std_dev                  DECIMAL(15,3),
    mc_skewness                 DECIMAL(8,4),
    mc_kurtosis                 DECIMAL(8,4),
    mc_p2_5                     DECIMAL(15,3),
    mc_p5                       DECIMAL(15,3),
    mc_p10                      DECIMAL(15,3),
    mc_p25                      DECIMAL(15,3),
    mc_p75                      DECIMAL(15,3),
    mc_p90                      DECIMAL(15,3),
    mc_p95                      DECIMAL(15,3),
    mc_p97_5                    DECIMAL(15,3),
    mc_coefficient_of_var       DECIMAL(8,4),
    mc_convergence_achieved     BOOLEAN,
    mc_convergence_iteration    INTEGER,
    -- Summary
    recommended_method          VARCHAR(30),
    overall_uncertainty_pct     DECIMAL(8,4),
    overall_rating              VARCHAR(20),
    -- Metadata
    analysis_notes              TEXT,
    methodology_reference       VARCHAR(500),
    software_version            VARCHAR(50),
    analyst                     VARCHAR(255),
    reviewed_by                 VARCHAR(255),
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_ua_inv_type CHECK (
        inventory_type IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_1_2')
    ),
    CONSTRAINT chk_p041_ua_method CHECK (
        method IN ('ANALYTICAL', 'MONTE_CARLO', 'BOTH', 'EXPERT_JUDGMENT')
    ),
    CONSTRAINT chk_p041_ua_confidence CHECK (
        confidence_level >= 0.50 AND confidence_level <= 0.99
    ),
    CONSTRAINT chk_p041_ua_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p041_ua_mc_iterations CHECK (
        mc_iterations IS NULL OR (mc_iterations >= 100 AND mc_iterations <= 10000000)
    ),
    CONSTRAINT chk_p041_ua_analytical_bounds CHECK (
        analytical_lower IS NULL OR analytical_upper IS NULL OR
        analytical_lower <= analytical_upper
    ),
    CONSTRAINT chk_p041_ua_mc_percentiles CHECK (
        mc_p2_5 IS NULL OR mc_p97_5 IS NULL OR mc_p2_5 <= mc_p97_5
    ),
    CONSTRAINT chk_p041_ua_analytical_pct CHECK (
        analytical_uncertainty_pct IS NULL OR (analytical_uncertainty_pct >= 0 AND analytical_uncertainty_pct <= 500)
    ),
    CONSTRAINT chk_p041_ua_overall_pct CHECK (
        overall_uncertainty_pct IS NULL OR (overall_uncertainty_pct >= 0 AND overall_uncertainty_pct <= 500)
    ),
    CONSTRAINT chk_p041_ua_overall_rating CHECK (
        overall_rating IS NULL OR overall_rating IN (
            'VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH'
        )
    ),
    CONSTRAINT chk_p041_ua_recommended CHECK (
        recommended_method IS NULL OR recommended_method IN (
            'ANALYTICAL', 'MONTE_CARLO', 'BOTH'
        )
    ),
    CONSTRAINT uq_p041_ua_inv_method UNIQUE (inventory_id, inventory_type, method)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_ua_tenant             ON ghg_scope12.uncertainty_analyses(tenant_id);
CREATE INDEX idx_p041_ua_inventory          ON ghg_scope12.uncertainty_analyses(inventory_id);
CREATE INDEX idx_p041_ua_inv_type           ON ghg_scope12.uncertainty_analyses(inventory_type);
CREATE INDEX idx_p041_ua_org               ON ghg_scope12.uncertainty_analyses(organization_id);
CREATE INDEX idx_p041_ua_year              ON ghg_scope12.uncertainty_analyses(reporting_year);
CREATE INDEX idx_p041_ua_method            ON ghg_scope12.uncertainty_analyses(method);
CREATE INDEX idx_p041_ua_rating            ON ghg_scope12.uncertainty_analyses(overall_rating);
CREATE INDEX idx_p041_ua_created           ON ghg_scope12.uncertainty_analyses(created_at DESC);
CREATE INDEX idx_p041_ua_metadata          ON ghg_scope12.uncertainty_analyses USING GIN(metadata);

-- Composite: org + year for trending
CREATE INDEX idx_p041_ua_org_year          ON ghg_scope12.uncertainty_analyses(organization_id, reporting_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_ua_updated
    BEFORE UPDATE ON ghg_scope12.uncertainty_analyses
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.uncertainty_sources
-- =============================================================================
-- Per-source-category uncertainty breakdown within an analysis. Decomposes
-- total uncertainty into activity data uncertainty and emission factor
-- uncertainty for each source. Calculates each source's contribution to
-- total inventory uncertainty using sensitivity analysis (partial derivatives
-- for analytical, correlation analysis for Monte Carlo).

CREATE TABLE ghg_scope12.uncertainty_sources (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_scope12.uncertainty_analyses(id) ON DELETE CASCADE,
    source_category             VARCHAR(60)     NOT NULL,
    source_sub_category         VARCHAR(60),
    facility_id                 UUID            REFERENCES ghg_scope12.facilities(id) ON DELETE SET NULL,
    -- Emissions
    emissions_tco2e             DECIMAL(12,3)   NOT NULL DEFAULT 0,
    emissions_pct_of_total      DECIMAL(8,4),
    -- Activity data uncertainty
    activity_data_source        VARCHAR(100),
    activity_uncertainty_pct    DECIMAL(8,4)    NOT NULL DEFAULT 0,
    activity_distribution       VARCHAR(20)     DEFAULT 'NORMAL',
    activity_lower_bound        DECIMAL(18,6),
    activity_upper_bound        DECIMAL(18,6),
    -- Emission factor uncertainty
    ef_source                   VARCHAR(100),
    ef_uncertainty_pct          DECIMAL(8,4)    NOT NULL DEFAULT 0,
    ef_distribution             VARCHAR(20)     DEFAULT 'NORMAL',
    ef_lower_bound              DECIMAL(12,6),
    ef_upper_bound              DECIMAL(12,6),
    -- Combined uncertainty
    combined_uncertainty_pct    DECIMAL(8,4)    NOT NULL DEFAULT 0,
    combined_lower_tco2e        DECIMAL(12,3),
    combined_upper_tco2e        DECIMAL(12,3),
    -- Contribution analysis
    contribution_to_total_pct   DECIMAL(8,4),
    sensitivity_coefficient     DECIMAL(10,6),
    rank_by_contribution        INTEGER,
    -- Quality
    data_quality_indicator      VARCHAR(20),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_us_emissions CHECK (
        emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p041_us_emissions_pct CHECK (
        emissions_pct_of_total IS NULL OR (emissions_pct_of_total >= 0 AND emissions_pct_of_total <= 100)
    ),
    CONSTRAINT chk_p041_us_activity_unc CHECK (
        activity_uncertainty_pct >= 0 AND activity_uncertainty_pct <= 500
    ),
    CONSTRAINT chk_p041_us_ef_unc CHECK (
        ef_uncertainty_pct >= 0 AND ef_uncertainty_pct <= 500
    ),
    CONSTRAINT chk_p041_us_combined_unc CHECK (
        combined_uncertainty_pct >= 0 AND combined_uncertainty_pct <= 500
    ),
    CONSTRAINT chk_p041_us_contribution CHECK (
        contribution_to_total_pct IS NULL OR (contribution_to_total_pct >= 0 AND contribution_to_total_pct <= 100)
    ),
    CONSTRAINT chk_p041_us_activity_dist CHECK (
        activity_distribution IS NULL OR activity_distribution IN (
            'NORMAL', 'LOGNORMAL', 'UNIFORM', 'TRIANGULAR', 'BETA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_us_ef_dist CHECK (
        ef_distribution IS NULL OR ef_distribution IN (
            'NORMAL', 'LOGNORMAL', 'UNIFORM', 'TRIANGULAR', 'BETA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_us_quality_ind CHECK (
        data_quality_indicator IS NULL OR data_quality_indicator IN (
            'HIGH', 'MEDIUM', 'LOW', 'ESTIMATED', 'DEFAULT'
        )
    ),
    CONSTRAINT chk_p041_us_rank CHECK (
        rank_by_contribution IS NULL OR rank_by_contribution >= 1
    ),
    CONSTRAINT uq_p041_us_analysis_source UNIQUE (analysis_id, source_category, source_sub_category, facility_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_us_tenant             ON ghg_scope12.uncertainty_sources(tenant_id);
CREATE INDEX idx_p041_us_analysis           ON ghg_scope12.uncertainty_sources(analysis_id);
CREATE INDEX idx_p041_us_category           ON ghg_scope12.uncertainty_sources(source_category);
CREATE INDEX idx_p041_us_facility           ON ghg_scope12.uncertainty_sources(facility_id);
CREATE INDEX idx_p041_us_contribution       ON ghg_scope12.uncertainty_sources(contribution_to_total_pct DESC);
CREATE INDEX idx_p041_us_rank              ON ghg_scope12.uncertainty_sources(rank_by_contribution);
CREATE INDEX idx_p041_us_combined           ON ghg_scope12.uncertainty_sources(combined_uncertainty_pct DESC);
CREATE INDEX idx_p041_us_created            ON ghg_scope12.uncertainty_sources(created_at DESC);

-- Composite: analysis + rank for top contributors
CREATE INDEX idx_p041_us_analysis_rank      ON ghg_scope12.uncertainty_sources(analysis_id, rank_by_contribution)
    WHERE rank_by_contribution IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_us_updated
    BEFORE UPDATE ON ghg_scope12.uncertainty_sources
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.uncertainty_improvements
-- =============================================================================
-- Recommended actions to reduce uncertainty for specific source categories.
-- Prioritized by impact (reduction in total uncertainty) and feasibility.
-- Tracks current vs. target uncertainty levels and links to methodology
-- tier improvements.

CREATE TABLE ghg_scope12.uncertainty_improvements (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_scope12.uncertainty_analyses(id) ON DELETE CASCADE,
    source_category             VARCHAR(60)     NOT NULL,
    source_sub_category         VARCHAR(60),
    facility_id                 UUID            REFERENCES ghg_scope12.facilities(id) ON DELETE SET NULL,
    -- Current state
    current_uncertainty_pct     DECIMAL(8,4)    NOT NULL,
    current_methodology_tier    VARCHAR(20),
    current_data_source         VARCHAR(100),
    -- Target state
    target_uncertainty_pct      DECIMAL(8,4)    NOT NULL,
    target_methodology_tier     VARCHAR(20),
    target_data_source          VARCHAR(100),
    -- Improvement
    recommended_action          TEXT            NOT NULL,
    action_category             VARCHAR(30)     NOT NULL DEFAULT 'DATA_IMPROVEMENT',
    expected_reduction_pct      DECIMAL(8,4),
    impact_on_total_unc_pct     DECIMAL(8,4),
    -- Implementation
    estimated_cost              DECIMAL(18,2),
    cost_currency               VARCHAR(3)      DEFAULT 'USD',
    implementation_timeframe    VARCHAR(30),
    responsible_party           VARCHAR(255),
    priority                    INTEGER         NOT NULL DEFAULT 3,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PROPOSED',
    -- Tracking
    implemented_at              TIMESTAMPTZ,
    actual_reduction_pct        DECIMAL(8,4),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_ui_current_unc CHECK (
        current_uncertainty_pct >= 0 AND current_uncertainty_pct <= 500
    ),
    CONSTRAINT chk_p041_ui_target_unc CHECK (
        target_uncertainty_pct >= 0 AND target_uncertainty_pct <= 500
    ),
    CONSTRAINT chk_p041_ui_target_lower CHECK (
        target_uncertainty_pct <= current_uncertainty_pct
    ),
    CONSTRAINT chk_p041_ui_action_category CHECK (
        action_category IN (
            'DATA_IMPROVEMENT', 'METHODOLOGY_UPGRADE', 'MEASUREMENT_INSTALL',
            'SUPPLIER_ENGAGEMENT', 'PROCESS_IMPROVEMENT', 'VERIFICATION',
            'TRAINING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_ui_timeframe CHECK (
        implementation_timeframe IS NULL OR implementation_timeframe IN (
            'IMMEDIATE', 'SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM', 'ONGOING'
        )
    ),
    CONSTRAINT chk_p041_ui_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p041_ui_status CHECK (
        status IN ('PROPOSED', 'APPROVED', 'IN_PROGRESS', 'COMPLETED', 'DEFERRED', 'REJECTED')
    ),
    CONSTRAINT chk_p041_ui_cost CHECK (
        estimated_cost IS NULL OR estimated_cost >= 0
    ),
    CONSTRAINT chk_p041_ui_expected_reduction CHECK (
        expected_reduction_pct IS NULL OR (expected_reduction_pct >= 0 AND expected_reduction_pct <= 100)
    ),
    CONSTRAINT chk_p041_ui_actual_reduction CHECK (
        actual_reduction_pct IS NULL OR (actual_reduction_pct >= 0 AND actual_reduction_pct <= 100)
    ),
    CONSTRAINT chk_p041_ui_tier CHECK (
        current_methodology_tier IS NULL OR current_methodology_tier IN (
            'TIER_1', 'TIER_2', 'TIER_3', 'DIRECT_MEASUREMENT', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p041_ui_target_tier CHECK (
        target_methodology_tier IS NULL OR target_methodology_tier IN (
            'TIER_1', 'TIER_2', 'TIER_3', 'DIRECT_MEASUREMENT', 'HYBRID'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_ui_tenant             ON ghg_scope12.uncertainty_improvements(tenant_id);
CREATE INDEX idx_p041_ui_analysis           ON ghg_scope12.uncertainty_improvements(analysis_id);
CREATE INDEX idx_p041_ui_category           ON ghg_scope12.uncertainty_improvements(source_category);
CREATE INDEX idx_p041_ui_facility           ON ghg_scope12.uncertainty_improvements(facility_id);
CREATE INDEX idx_p041_ui_priority           ON ghg_scope12.uncertainty_improvements(priority);
CREATE INDEX idx_p041_ui_status             ON ghg_scope12.uncertainty_improvements(status);
CREATE INDEX idx_p041_ui_action_cat         ON ghg_scope12.uncertainty_improvements(action_category);
CREATE INDEX idx_p041_ui_impact             ON ghg_scope12.uncertainty_improvements(impact_on_total_unc_pct DESC);
CREATE INDEX idx_p041_ui_created            ON ghg_scope12.uncertainty_improvements(created_at DESC);

-- Composite: analysis + open items by priority
CREATE INDEX idx_p041_ui_analysis_open      ON ghg_scope12.uncertainty_improvements(analysis_id, priority)
    WHERE status IN ('PROPOSED', 'APPROVED', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_ui_updated
    BEFORE UPDATE ON ghg_scope12.uncertainty_improvements
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.uncertainty_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.uncertainty_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.uncertainty_improvements ENABLE ROW LEVEL SECURITY;

CREATE POLICY p041_ua_tenant_isolation
    ON ghg_scope12.uncertainty_analyses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_ua_service_bypass
    ON ghg_scope12.uncertainty_analyses
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_us_tenant_isolation
    ON ghg_scope12.uncertainty_sources
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_us_service_bypass
    ON ghg_scope12.uncertainty_sources
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_ui_tenant_isolation
    ON ghg_scope12.uncertainty_improvements
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_ui_service_bypass
    ON ghg_scope12.uncertainty_improvements
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.uncertainty_analyses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.uncertainty_sources TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.uncertainty_improvements TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_scope12.uncertainty_analyses IS
    'Inventory-level uncertainty analysis supporting IPCC Approach 1 (analytical/error propagation) and Approach 2 (Monte Carlo simulation) with confidence intervals.';
COMMENT ON TABLE ghg_scope12.uncertainty_sources IS
    'Per-source-category uncertainty decomposition with activity data and emission factor uncertainty, probability distributions, and contribution to total.';
COMMENT ON TABLE ghg_scope12.uncertainty_improvements IS
    'Prioritized improvement recommendations to reduce uncertainty by source category with cost, timeframe, and implementation tracking.';

COMMENT ON COLUMN ghg_scope12.uncertainty_analyses.method IS 'Analysis method: ANALYTICAL (IPCC Approach 1), MONTE_CARLO (IPCC Approach 2), BOTH, or EXPERT_JUDGMENT.';
COMMENT ON COLUMN ghg_scope12.uncertainty_analyses.confidence_level IS 'Statistical confidence level (0.50-0.99, typically 0.95 for 95% CI).';
COMMENT ON COLUMN ghg_scope12.uncertainty_analyses.mc_p2_5 IS '2.5th percentile from Monte Carlo (lower bound of 95% CI).';
COMMENT ON COLUMN ghg_scope12.uncertainty_analyses.mc_p97_5 IS '97.5th percentile from Monte Carlo (upper bound of 95% CI).';
COMMENT ON COLUMN ghg_scope12.uncertainty_analyses.mc_coefficient_of_var IS 'Coefficient of variation (std_dev / mean) from Monte Carlo distribution.';
COMMENT ON COLUMN ghg_scope12.uncertainty_analyses.overall_rating IS 'Qualitative uncertainty rating: VERY_LOW (<5%), LOW (5-15%), MODERATE (15-30%), HIGH (30-50%), VERY_HIGH (>50%).';

COMMENT ON COLUMN ghg_scope12.uncertainty_sources.activity_distribution IS 'Probability distribution for activity data: NORMAL, LOGNORMAL, UNIFORM, TRIANGULAR, BETA.';
COMMENT ON COLUMN ghg_scope12.uncertainty_sources.ef_distribution IS 'Probability distribution for emission factor: NORMAL, LOGNORMAL, UNIFORM, TRIANGULAR, BETA.';
COMMENT ON COLUMN ghg_scope12.uncertainty_sources.contribution_to_total_pct IS 'Percentage contribution of this source to total inventory uncertainty (sum across all sources = 100%).';
COMMENT ON COLUMN ghg_scope12.uncertainty_sources.sensitivity_coefficient IS 'Sensitivity coefficient (partial derivative) indicating how much total uncertainty changes per unit change in this source.';

COMMENT ON COLUMN ghg_scope12.uncertainty_improvements.priority IS 'Priority 1 (highest) to 5 (lowest) based on impact and feasibility.';
COMMENT ON COLUMN ghg_scope12.uncertainty_improvements.impact_on_total_unc_pct IS 'Expected reduction in total inventory uncertainty (percentage points) if this improvement is implemented.';

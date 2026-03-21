-- =============================================================================
-- V188: PACK-028 Sector Pathway Pack - Scenario Comparisons
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    008 of 015
-- Date:         March 2026
--
-- Multi-scenario pathway comparisons (1.5C/WB2C/2C/APS/STEPS) with
-- side-by-side results, investment deltas, technology adoption timeline
-- differences, risk-return analysis, and optimal pathway recommendations.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_scenario_comparisons
--
-- Previous: V187__PACK028_sector_benchmarks.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_scenario_comparisons
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_scenario_comparisons (
    comparison_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    classification_id           UUID            REFERENCES pack028_sector_pathway.gl_sector_classifications(classification_id) ON DELETE SET NULL,
    -- Sector context
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    -- Comparison definition
    comparison_name             VARCHAR(255)    NOT NULL,
    comparison_type             VARCHAR(30)     NOT NULL DEFAULT 'FULL_SCENARIO',
    analysis_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    -- Scenarios compared
    scenarios_compared          TEXT[]          NOT NULL,
    primary_scenario            VARCHAR(30)     NOT NULL DEFAULT 'NZE_1_5C',
    reference_scenario          VARCHAR(30)     DEFAULT 'STEPS',
    scenario_count              INTEGER         NOT NULL DEFAULT 2,
    -- Pathway references
    pathway_ids                 UUID[]          DEFAULT '{}',
    -- Per-scenario results (JSONB array of scenario objects)
    scenario_results            JSONB           NOT NULL DEFAULT '[]',
    -- Intensity comparisons
    intensity_comparison_2030   JSONB           DEFAULT '{}',
    intensity_comparison_2040   JSONB           DEFAULT '{}',
    intensity_comparison_2050   JSONB           DEFAULT '{}',
    -- Emissions comparisons
    emissions_comparison_2030   JSONB           DEFAULT '{}',
    emissions_comparison_2040   JSONB           DEFAULT '{}',
    emissions_comparison_2050   JSONB           DEFAULT '{}',
    cumulative_emissions_delta  JSONB           DEFAULT '{}',
    -- Investment analysis
    investment_comparison       JSONB           DEFAULT '{}',
    total_investment_by_scenario JSONB          DEFAULT '{}',
    incremental_investment_vs_bau JSONB         DEFAULT '{}',
    investment_delta_1_5c_vs_2c DECIMAL(18,2),
    investment_delta_nze_vs_aps DECIMAL(18,2),
    -- Technology adoption differences
    technology_timeline_comparison JSONB        DEFAULT '{}',
    technology_acceleration_needed JSONB        DEFAULT '{}',
    key_technology_deltas       JSONB           DEFAULT '[]',
    -- Risk-return analysis
    risk_return_matrix          JSONB           DEFAULT '{}',
    risk_by_scenario            JSONB           DEFAULT '{}',
    stranded_asset_risk_by_scenario JSONB       DEFAULT '{}',
    carbon_price_risk_by_scenario JSONB         DEFAULT '{}',
    regulatory_risk_by_scenario JSONB           DEFAULT '{}',
    -- Optimal pathway recommendation
    recommended_scenario        VARCHAR(30),
    recommendation_rationale    TEXT,
    recommendation_confidence   DECIMAL(5,2),
    recommendation_factors      JSONB           DEFAULT '[]',
    -- Sensitivity analysis
    sensitivity_results         JSONB           DEFAULT '{}',
    key_decision_points         JSONB           DEFAULT '[]',
    scenario_switching_triggers JSONB           DEFAULT '[]',
    -- Carbon budget analysis
    carbon_budget_remaining     JSONB           DEFAULT '{}',
    carbon_budget_exhaustion_year JSONB         DEFAULT '{}',
    -- Cost of delay analysis
    cost_of_1yr_delay_usd       DECIMAL(18,2),
    cost_of_5yr_delay_usd       DECIMAL(18,2),
    delay_emissions_impact_tco2e DECIMAL(18,4),
    -- Summary metrics
    ambition_gap_matrix         JSONB           DEFAULT '{}',
    feasibility_scores          JSONB           DEFAULT '{}',
    cost_effectiveness_ranking  JSONB           DEFAULT '[]',
    -- Status
    comparison_status           VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    -- Metadata
    version                     INTEGER         DEFAULT 1,
    is_active                   BOOLEAN         DEFAULT TRUE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_scc_comparison_type CHECK (
        comparison_type IN ('FULL_SCENARIO', 'PATHWAY_ONLY', 'INVESTMENT_FOCUSED',
                            'TECHNOLOGY_FOCUSED', 'RISK_FOCUSED', 'QUICK_COMPARE')
    ),
    CONSTRAINT chk_p028_scc_primary_scenario CHECK (
        primary_scenario IN ('NZE_1_5C', 'WB2C', '2C', 'APS', 'STEPS', 'NDC_ALIGNED', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_scc_reference_scenario CHECK (
        reference_scenario IS NULL OR reference_scenario IN (
            'NZE_1_5C', 'WB2C', '2C', 'APS', 'STEPS', 'NDC_ALIGNED', 'BAU', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p028_scc_recommended CHECK (
        recommended_scenario IS NULL OR recommended_scenario IN (
            'NZE_1_5C', 'WB2C', '2C', 'APS', 'STEPS', 'NDC_ALIGNED', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p028_scc_scenario_count CHECK (
        scenario_count >= 2 AND scenario_count <= 10
    ),
    CONSTRAINT chk_p028_scc_confidence CHECK (
        recommendation_confidence IS NULL OR (recommendation_confidence >= 0 AND recommendation_confidence <= 100)
    ),
    CONSTRAINT chk_p028_scc_status CHECK (
        comparison_status IN ('DRAFT', 'IN_PROGRESS', 'COMPLETED', 'APPROVED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p028_scc_version CHECK (
        version >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_scc_tenant            ON pack028_sector_pathway.gl_scenario_comparisons(tenant_id);
CREATE INDEX idx_p028_scc_company           ON pack028_sector_pathway.gl_scenario_comparisons(company_id);
CREATE INDEX idx_p028_scc_classification    ON pack028_sector_pathway.gl_scenario_comparisons(classification_id);
CREATE INDEX idx_p028_scc_sector            ON pack028_sector_pathway.gl_scenario_comparisons(sector_code);
CREATE INDEX idx_p028_scc_comparison_type   ON pack028_sector_pathway.gl_scenario_comparisons(comparison_type);
CREATE INDEX idx_p028_scc_primary           ON pack028_sector_pathway.gl_scenario_comparisons(primary_scenario);
CREATE INDEX idx_p028_scc_recommended       ON pack028_sector_pathway.gl_scenario_comparisons(recommended_scenario);
CREATE INDEX idx_p028_scc_status            ON pack028_sector_pathway.gl_scenario_comparisons(comparison_status);
CREATE INDEX idx_p028_scc_active            ON pack028_sector_pathway.gl_scenario_comparisons(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_scc_date              ON pack028_sector_pathway.gl_scenario_comparisons(analysis_date DESC);
CREATE INDEX idx_p028_scc_company_sector    ON pack028_sector_pathway.gl_scenario_comparisons(company_id, sector_code);
CREATE INDEX idx_p028_scc_scenarios         ON pack028_sector_pathway.gl_scenario_comparisons USING GIN(scenarios_compared);
CREATE INDEX idx_p028_scc_pathway_ids       ON pack028_sector_pathway.gl_scenario_comparisons USING GIN(pathway_ids);
CREATE INDEX idx_p028_scc_results           ON pack028_sector_pathway.gl_scenario_comparisons USING GIN(scenario_results);
CREATE INDEX idx_p028_scc_risk_return       ON pack028_sector_pathway.gl_scenario_comparisons USING GIN(risk_return_matrix);
CREATE INDEX idx_p028_scc_created           ON pack028_sector_pathway.gl_scenario_comparisons(created_at DESC);
CREATE INDEX idx_p028_scc_metadata          ON pack028_sector_pathway.gl_scenario_comparisons USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_scenario_comparisons_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_scenario_comparisons
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_scenario_comparisons ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_scc_tenant_isolation
    ON pack028_sector_pathway.gl_scenario_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_scc_service_bypass
    ON pack028_sector_pathway.gl_scenario_comparisons
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_scenario_comparisons TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_scenario_comparisons IS
    'Multi-scenario pathway comparisons (NZE/WB2C/2C/APS/STEPS) with investment deltas, technology timeline differences, risk-return analysis, and optimal pathway recommendations.';

COMMENT ON COLUMN pack028_sector_pathway.gl_scenario_comparisons.comparison_id IS 'Unique scenario comparison identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_scenario_comparisons.scenarios_compared IS 'Array of scenario codes included in this comparison.';
COMMENT ON COLUMN pack028_sector_pathway.gl_scenario_comparisons.scenario_results IS 'JSONB array of per-scenario results with intensity trajectories, emissions, and investments.';
COMMENT ON COLUMN pack028_sector_pathway.gl_scenario_comparisons.recommended_scenario IS 'Recommended optimal scenario based on risk-return analysis.';
COMMENT ON COLUMN pack028_sector_pathway.gl_scenario_comparisons.cost_of_1yr_delay_usd IS 'Additional cost (USD) incurred by delaying pathway action by 1 year.';
COMMENT ON COLUMN pack028_sector_pathway.gl_scenario_comparisons.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

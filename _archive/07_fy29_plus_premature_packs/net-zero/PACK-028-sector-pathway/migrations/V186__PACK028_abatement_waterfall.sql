-- =============================================================================
-- V186: PACK-028 Sector Pathway Pack - Abatement Waterfall
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    006 of 015
-- Date:         March 2026
--
-- Sector-specific abatement lever definitions with waterfall calculations,
-- cost curves, lever interdependencies, implementation sequencing, and
-- cumulative abatement tracking for sector pathway achievement.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_sector_abatement_levers
--
-- Previous: V185__PACK028_technology_roadmaps.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_sector_abatement_levers
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_sector_abatement_levers (
    lever_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    pathway_id                  UUID            REFERENCES pack028_sector_pathway.gl_sector_pathways(pathway_id) ON DELETE SET NULL,
    roadmap_id                  UUID            REFERENCES pack028_sector_pathway.gl_technology_roadmaps(roadmap_id) ON DELETE SET NULL,
    -- Sector context
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    -- Lever definition
    lever_name                  VARCHAR(255)    NOT NULL,
    lever_code                  VARCHAR(60)     NOT NULL,
    lever_category              VARCHAR(50)     NOT NULL,
    lever_sub_category          VARCHAR(80),
    lever_description           TEXT,
    -- Waterfall position
    waterfall_order             INTEGER         NOT NULL DEFAULT 1,
    waterfall_group             VARCHAR(50),
    is_baseline                 BOOLEAN         DEFAULT FALSE,
    is_residual                 BOOLEAN         DEFAULT FALSE,
    -- Abatement potential
    abatement_tco2e_annual      DECIMAL(18,4),
    abatement_tco2e_cumulative  DECIMAL(18,4),
    abatement_pct_of_total      DECIMAL(6,2),
    abatement_intensity_impact  DECIMAL(18,8),
    abatement_confidence        VARCHAR(20)     DEFAULT 'MEDIUM',
    -- Cost analysis
    cost_per_tco2e              DECIMAL(12,2),
    cost_per_tco2e_low          DECIMAL(12,2),
    cost_per_tco2e_high         DECIMAL(12,2),
    cost_currency               VARCHAR(3)      DEFAULT 'USD',
    total_cost_usd              DECIMAL(18,2),
    is_negative_cost            BOOLEAN         DEFAULT FALSE,
    net_present_value_usd       DECIMAL(18,2),
    -- Implementation timeline
    implementation_start_year   INTEGER,
    implementation_end_year     INTEGER,
    ramp_up_years               INTEGER         DEFAULT 2,
    full_effect_year            INTEGER,
    annual_abatement_schedule   JSONB           DEFAULT '{}',
    -- Annual cost schedule
    annual_cost_schedule        JSONB           DEFAULT '{}',
    capex_required_usd          DECIMAL(18,2),
    annual_opex_usd             DECIMAL(18,2),
    -- Dependencies and sequencing
    prerequisite_levers         TEXT[]          DEFAULT '{}',
    dependent_levers            TEXT[]          DEFAULT '{}',
    complementary_levers        TEXT[]          DEFAULT '{}',
    conflicting_levers          TEXT[]          DEFAULT '{}',
    dependency_type             VARCHAR(20),
    -- Lever characteristics
    lever_type                  VARCHAR(30)     NOT NULL DEFAULT 'TECHNICAL',
    reversibility               VARCHAR(20)     DEFAULT 'PERMANENT',
    scalability                 VARCHAR(20)     DEFAULT 'SCALABLE',
    maturity                    VARCHAR(20)     DEFAULT 'PROVEN',
    -- Uncertainty ranges
    abatement_low_tco2e         DECIMAL(18,4),
    abatement_high_tco2e        DECIMAL(18,4),
    probability_of_delivery     DECIMAL(5,2),
    -- Progress tracking
    implementation_status       VARCHAR(20)     NOT NULL DEFAULT 'PLANNED',
    actual_abatement_tco2e      DECIMAL(18,4),
    actual_vs_planned_pct       DECIMAL(8,2),
    last_progress_date          DATE,
    progress_notes              TEXT,
    -- Sector-specific parameters
    sector_parameters           JSONB           DEFAULT '{}',
    -- Metadata
    priority                    INTEGER         DEFAULT 3,
    is_active                   BOOLEAN         DEFAULT TRUE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_al_lever_category CHECK (
        lever_category IN (
            'RENEWABLE_PROCUREMENT', 'FUEL_SWITCHING', 'ELECTRIFICATION',
            'ENERGY_EFFICIENCY', 'PROCESS_EFFICIENCY', 'CCS_CCUS',
            'HYDROGEN', 'CIRCULAR_ECONOMY', 'FLEET_TRANSITION',
            'BUILDING_RETROFIT', 'SUSTAINABLE_FUELS', 'NATURE_BASED',
            'DEMAND_REDUCTION', 'SUPPLY_CHAIN', 'OFFSET_REMOVAL',
            'BEHAVIORAL_CHANGE', 'DIGITALIZATION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p028_al_lever_type CHECK (
        lever_type IN ('TECHNICAL', 'OPERATIONAL', 'BEHAVIORAL', 'MARKET_BASED',
                       'NATURE_BASED', 'POLICY_DRIVEN', 'HYBRID')
    ),
    CONSTRAINT chk_p028_al_reversibility CHECK (
        reversibility IN ('PERMANENT', 'LONG_TERM', 'MEDIUM_TERM', 'SHORT_TERM', 'REVERSIBLE')
    ),
    CONSTRAINT chk_p028_al_scalability CHECK (
        scalability IN ('HIGHLY_SCALABLE', 'SCALABLE', 'MODERATELY_SCALABLE',
                       'LIMITED_SCALABILITY', 'NOT_SCALABLE')
    ),
    CONSTRAINT chk_p028_al_maturity CHECK (
        maturity IN ('PROVEN', 'COMMERCIALLY_AVAILABLE', 'DEMONSTRATION',
                    'PILOT', 'PROTOTYPE', 'CONCEPT', 'RESEARCH')
    ),
    CONSTRAINT chk_p028_al_confidence CHECK (
        abatement_confidence IN ('VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p028_al_dependency_type CHECK (
        dependency_type IS NULL OR dependency_type IN (
            'SEQUENTIAL', 'PARALLEL', 'CONDITIONAL', 'INDEPENDENT'
        )
    ),
    CONSTRAINT chk_p028_al_implementation CHECK (
        implementation_status IN ('PLANNED', 'APPROVED', 'IN_PROGRESS', 'SCALING',
                                  'COMPLETED', 'ON_HOLD', 'CANCELLED')
    ),
    CONSTRAINT chk_p028_al_abatement_pct CHECK (
        abatement_pct_of_total IS NULL OR (abatement_pct_of_total >= 0 AND abatement_pct_of_total <= 100)
    ),
    CONSTRAINT chk_p028_al_probability CHECK (
        probability_of_delivery IS NULL OR (probability_of_delivery >= 0 AND probability_of_delivery <= 100)
    ),
    CONSTRAINT chk_p028_al_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p028_al_waterfall_order CHECK (
        waterfall_order >= 0
    ),
    CONSTRAINT chk_p028_al_impl_years CHECK (
        implementation_start_year IS NULL OR (implementation_start_year >= 2020 AND implementation_start_year <= 2060)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_al_tenant             ON pack028_sector_pathway.gl_sector_abatement_levers(tenant_id);
CREATE INDEX idx_p028_al_company            ON pack028_sector_pathway.gl_sector_abatement_levers(company_id);
CREATE INDEX idx_p028_al_pathway            ON pack028_sector_pathway.gl_sector_abatement_levers(pathway_id);
CREATE INDEX idx_p028_al_roadmap            ON pack028_sector_pathway.gl_sector_abatement_levers(roadmap_id);
CREATE INDEX idx_p028_al_sector             ON pack028_sector_pathway.gl_sector_abatement_levers(sector_code);
CREATE INDEX idx_p028_al_lever_name         ON pack028_sector_pathway.gl_sector_abatement_levers(lever_name);
CREATE INDEX idx_p028_al_lever_code         ON pack028_sector_pathway.gl_sector_abatement_levers(lever_code);
CREATE INDEX idx_p028_al_lever_category     ON pack028_sector_pathway.gl_sector_abatement_levers(lever_category);
CREATE INDEX idx_p028_al_waterfall_order    ON pack028_sector_pathway.gl_sector_abatement_levers(company_id, pathway_id, waterfall_order);
CREATE INDEX idx_p028_al_cost_per_tco2e     ON pack028_sector_pathway.gl_sector_abatement_levers(cost_per_tco2e);
CREATE INDEX idx_p028_al_negative_cost      ON pack028_sector_pathway.gl_sector_abatement_levers(is_negative_cost) WHERE is_negative_cost = TRUE;
CREATE INDEX idx_p028_al_impl_status        ON pack028_sector_pathway.gl_sector_abatement_levers(implementation_status);
CREATE INDEX idx_p028_al_lever_type         ON pack028_sector_pathway.gl_sector_abatement_levers(lever_type);
CREATE INDEX idx_p028_al_maturity           ON pack028_sector_pathway.gl_sector_abatement_levers(maturity);
CREATE INDEX idx_p028_al_priority           ON pack028_sector_pathway.gl_sector_abatement_levers(priority);
CREATE INDEX idx_p028_al_active             ON pack028_sector_pathway.gl_sector_abatement_levers(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_al_company_category   ON pack028_sector_pathway.gl_sector_abatement_levers(company_id, lever_category);
CREATE INDEX idx_p028_al_abatement_desc     ON pack028_sector_pathway.gl_sector_abatement_levers(abatement_tco2e_annual DESC NULLS LAST);
CREATE INDEX idx_p028_al_prereq_levers      ON pack028_sector_pathway.gl_sector_abatement_levers USING GIN(prerequisite_levers);
CREATE INDEX idx_p028_al_annual_sched       ON pack028_sector_pathway.gl_sector_abatement_levers USING GIN(annual_abatement_schedule);
CREATE INDEX idx_p028_al_sector_params      ON pack028_sector_pathway.gl_sector_abatement_levers USING GIN(sector_parameters);
CREATE INDEX idx_p028_al_created            ON pack028_sector_pathway.gl_sector_abatement_levers(created_at DESC);
CREATE INDEX idx_p028_al_metadata           ON pack028_sector_pathway.gl_sector_abatement_levers USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_abatement_levers_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_abatement_levers
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_sector_abatement_levers ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_al_tenant_isolation
    ON pack028_sector_pathway.gl_sector_abatement_levers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_al_service_bypass
    ON pack028_sector_pathway.gl_sector_abatement_levers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_abatement_levers TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_sector_abatement_levers IS
    'Sector-specific abatement levers with waterfall ordering, cost curves, interdependencies, implementation tracking, and cumulative abatement calculations for pathway achievement.';

COMMENT ON COLUMN pack028_sector_pathway.gl_sector_abatement_levers.lever_id IS 'Unique abatement lever identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_abatement_levers.waterfall_order IS 'Position in the abatement waterfall chart (ascending order).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_abatement_levers.lever_category IS 'Abatement lever category: RENEWABLE_PROCUREMENT, FUEL_SWITCHING, CCS_CCUS, HYDROGEN, etc.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_abatement_levers.abatement_tco2e_annual IS 'Annual emission reduction potential in tCO2e at full deployment.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_abatement_levers.cost_per_tco2e IS 'Marginal abatement cost in USD per tCO2e (negative = net savings).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_abatement_levers.prerequisite_levers IS 'Array of lever codes that must be deployed before this lever.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_abatement_levers.probability_of_delivery IS 'Probability (0-100%) of delivering planned abatement.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_abatement_levers.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

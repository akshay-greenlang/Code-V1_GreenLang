-- =============================================================================
-- V185: PACK-028 Sector Pathway Pack - Technology Roadmaps
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    005 of 015
-- Date:         March 2026
--
-- Technology transition roadmaps with technology adoption schedules,
-- IEA milestone tracking, TRL assessment, CapEx phasing, dependency
-- mapping, and sector-specific technology pathways.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_technology_roadmaps
--
-- Previous: V184__PACK028_convergence_analysis.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_technology_roadmaps
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_technology_roadmaps (
    roadmap_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    pathway_id                  UUID            REFERENCES pack028_sector_pathway.gl_sector_pathways(pathway_id) ON DELETE SET NULL,
    classification_id           UUID            REFERENCES pack028_sector_pathway.gl_sector_classifications(classification_id) ON DELETE SET NULL,
    -- Sector context
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    -- Technology definition
    technology_name             VARCHAR(255)    NOT NULL,
    technology_code             VARCHAR(60)     NOT NULL,
    technology_category         VARCHAR(50)     NOT NULL,
    technology_sub_category     VARCHAR(80),
    technology_description      TEXT,
    -- Technology readiness
    current_trl                 INTEGER         NOT NULL DEFAULT 1,
    target_trl                  INTEGER         NOT NULL DEFAULT 9,
    trl_assessment_date         DATE,
    commercial_readiness_year   INTEGER,
    -- Adoption schedule
    adoption_start_year         INTEGER         NOT NULL,
    adoption_ramp_year          INTEGER,
    adoption_maturity_year      INTEGER,
    adoption_end_year           INTEGER,
    -- Adoption curve (S-curve parameters)
    adoption_model              VARCHAR(30)     DEFAULT 'S_CURVE',
    current_penetration_pct     DECIMAL(6,2)    DEFAULT 0.00,
    target_penetration_2030_pct DECIMAL(6,2),
    target_penetration_2040_pct DECIMAL(6,2),
    target_penetration_2050_pct DECIMAL(6,2),
    max_penetration_pct         DECIMAL(6,2)    DEFAULT 100.00,
    s_curve_midpoint_year       INTEGER,
    s_curve_growth_rate         DECIMAL(8,6),
    annual_adoption_schedule    JSONB           DEFAULT '{}',
    -- Emission reduction potential
    abatement_potential_tco2e   DECIMAL(18,4),
    abatement_intensity_impact  DECIMAL(18,8),
    abatement_share_pct         DECIMAL(6,2),
    marginal_abatement_cost     DECIMAL(12,2),
    mac_unit                    VARCHAR(20)     DEFAULT 'USD/tCO2e',
    -- CapEx requirements
    total_capex_usd             DECIMAL(18,2),
    capex_per_unit              DECIMAL(14,2),
    capex_unit                  VARCHAR(50),
    annual_capex_schedule       JSONB           DEFAULT '{}',
    capex_phasing               JSONB           DEFAULT '{}',
    -- OpEx impact
    annual_opex_change_usd      DECIMAL(18,2),
    opex_change_pct             DECIMAL(8,2),
    opex_savings_start_year     INTEGER,
    total_lifetime_savings_usd  DECIMAL(18,2),
    payback_period_years        DECIMAL(6,2),
    irr_pct                     DECIMAL(8,2),
    -- IEA milestones
    iea_milestone_ref           VARCHAR(100),
    iea_milestone_year          INTEGER,
    iea_milestone_description   TEXT,
    iea_milestone_status        VARCHAR(20),
    iea_chapter                 VARCHAR(100),
    -- Dependencies
    dependency_technologies     TEXT[]          DEFAULT '{}',
    dependency_infrastructure   TEXT[]          DEFAULT '{}',
    dependency_policy           TEXT[]          DEFAULT '{}',
    dependency_supply_chain     TEXT[]          DEFAULT '{}',
    dependencies_met            BOOLEAN         DEFAULT FALSE,
    blocking_dependencies       JSONB           DEFAULT '[]',
    -- Risk assessment
    technology_risk             VARCHAR(20)     DEFAULT 'MEDIUM',
    cost_uncertainty_pct        DECIMAL(6,2),
    performance_uncertainty_pct DECIMAL(6,2),
    supply_chain_risk           VARCHAR(20),
    policy_dependency_risk      VARCHAR(20),
    risk_factors                JSONB           DEFAULT '[]',
    -- Regional availability
    region                      VARCHAR(30)     DEFAULT 'GLOBAL',
    regional_availability       JSONB           DEFAULT '{}',
    -- Implementation status
    implementation_status       VARCHAR(20)     NOT NULL DEFAULT 'PLANNED',
    pilot_started               BOOLEAN         DEFAULT FALSE,
    pilot_start_date            DATE,
    pilot_results               JSONB           DEFAULT '{}',
    full_deployment_date        DATE,
    -- Metadata
    priority                    INTEGER         DEFAULT 3,
    is_active                   BOOLEAN         DEFAULT TRUE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_tr_technology_category CHECK (
        technology_category IN (
            'RENEWABLE_ENERGY', 'ENERGY_STORAGE', 'HYDROGEN', 'CCS_CCUS',
            'ELECTRIFICATION', 'FUEL_SWITCHING', 'ENERGY_EFFICIENCY',
            'PROCESS_INNOVATION', 'CIRCULAR_ECONOMY', 'DIGITALIZATION',
            'FLEET_TRANSITION', 'BUILDING_RETROFIT', 'SUSTAINABLE_FUELS',
            'NUCLEAR', 'GRID_INFRASTRUCTURE', 'HEAT_PUMPS', 'BIOMASS',
            'CARBON_REMOVAL', 'NATURE_BASED', 'OTHER'
        )
    ),
    CONSTRAINT chk_p028_tr_trl CHECK (
        current_trl >= 1 AND current_trl <= 9 AND target_trl >= 1 AND target_trl <= 9
    ),
    CONSTRAINT chk_p028_tr_trl_progress CHECK (
        target_trl >= current_trl
    ),
    CONSTRAINT chk_p028_tr_adoption_model CHECK (
        adoption_model IN ('S_CURVE', 'LINEAR', 'EXPONENTIAL', 'STEPPED', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_tr_penetration CHECK (
        current_penetration_pct >= 0 AND current_penetration_pct <= 100 AND
        max_penetration_pct >= 0 AND max_penetration_pct <= 100
    ),
    CONSTRAINT chk_p028_tr_technology_risk CHECK (
        technology_risk IN ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p028_tr_supply_chain_risk CHECK (
        supply_chain_risk IS NULL OR supply_chain_risk IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p028_tr_policy_risk CHECK (
        policy_dependency_risk IS NULL OR policy_dependency_risk IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p028_tr_iea_milestone_status CHECK (
        iea_milestone_status IS NULL OR iea_milestone_status IN (
            'ON_TRACK', 'BEHIND', 'WELL_BEHIND', 'ACHIEVED', 'NOT_STARTED', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p028_tr_implementation CHECK (
        implementation_status IN ('PLANNED', 'EVALUATING', 'PILOT', 'SCALING',
                                  'DEPLOYED', 'MATURE', 'DECOMMISSIONING', 'CANCELLED')
    ),
    CONSTRAINT chk_p028_tr_region CHECK (
        region IN ('GLOBAL', 'OECD', 'NON_OECD', 'EU', 'NORTH_AMERICA', 'ASIA_PACIFIC',
                   'LATIN_AMERICA', 'AFRICA', 'MIDDLE_EAST', 'CHINA', 'INDIA', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_tr_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p028_tr_adoption_start CHECK (
        adoption_start_year >= 2020 AND adoption_start_year <= 2060
    ),
    CONSTRAINT chk_p028_tr_payback CHECK (
        payback_period_years IS NULL OR payback_period_years >= 0
    ),
    CONSTRAINT chk_p028_tr_cost_uncertainty CHECK (
        cost_uncertainty_pct IS NULL OR (cost_uncertainty_pct >= 0 AND cost_uncertainty_pct <= 200)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_tr_tenant             ON pack028_sector_pathway.gl_technology_roadmaps(tenant_id);
CREATE INDEX idx_p028_tr_company            ON pack028_sector_pathway.gl_technology_roadmaps(company_id);
CREATE INDEX idx_p028_tr_pathway            ON pack028_sector_pathway.gl_technology_roadmaps(pathway_id);
CREATE INDEX idx_p028_tr_classification     ON pack028_sector_pathway.gl_technology_roadmaps(classification_id);
CREATE INDEX idx_p028_tr_sector             ON pack028_sector_pathway.gl_technology_roadmaps(sector_code);
CREATE INDEX idx_p028_tr_tech_name          ON pack028_sector_pathway.gl_technology_roadmaps(technology_name);
CREATE INDEX idx_p028_tr_tech_code          ON pack028_sector_pathway.gl_technology_roadmaps(technology_code);
CREATE INDEX idx_p028_tr_tech_category      ON pack028_sector_pathway.gl_technology_roadmaps(technology_category);
CREATE INDEX idx_p028_tr_trl                ON pack028_sector_pathway.gl_technology_roadmaps(current_trl);
CREATE INDEX idx_p028_tr_adoption_start     ON pack028_sector_pathway.gl_technology_roadmaps(adoption_start_year);
CREATE INDEX idx_p028_tr_impl_status        ON pack028_sector_pathway.gl_technology_roadmaps(implementation_status);
CREATE INDEX idx_p028_tr_tech_risk          ON pack028_sector_pathway.gl_technology_roadmaps(technology_risk);
CREATE INDEX idx_p028_tr_iea_milestone      ON pack028_sector_pathway.gl_technology_roadmaps(iea_milestone_status);
CREATE INDEX idx_p028_tr_priority           ON pack028_sector_pathway.gl_technology_roadmaps(priority);
CREATE INDEX idx_p028_tr_active             ON pack028_sector_pathway.gl_technology_roadmaps(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_tr_company_sector     ON pack028_sector_pathway.gl_technology_roadmaps(company_id, sector_code);
CREATE INDEX idx_p028_tr_company_category   ON pack028_sector_pathway.gl_technology_roadmaps(company_id, technology_category);
CREATE INDEX idx_p028_tr_mac_desc           ON pack028_sector_pathway.gl_technology_roadmaps(marginal_abatement_cost) WHERE marginal_abatement_cost IS NOT NULL;
CREATE INDEX idx_p028_tr_pilot              ON pack028_sector_pathway.gl_technology_roadmaps(pilot_started) WHERE pilot_started = TRUE;
CREATE INDEX idx_p028_tr_region             ON pack028_sector_pathway.gl_technology_roadmaps(region);
CREATE INDEX idx_p028_tr_deps_met           ON pack028_sector_pathway.gl_technology_roadmaps(dependencies_met) WHERE dependencies_met = FALSE;
CREATE INDEX idx_p028_tr_created            ON pack028_sector_pathway.gl_technology_roadmaps(created_at DESC);
CREATE INDEX idx_p028_tr_capex_schedule     ON pack028_sector_pathway.gl_technology_roadmaps USING GIN(annual_capex_schedule);
CREATE INDEX idx_p028_tr_adoption_sched     ON pack028_sector_pathway.gl_technology_roadmaps USING GIN(annual_adoption_schedule);
CREATE INDEX idx_p028_tr_dep_techs          ON pack028_sector_pathway.gl_technology_roadmaps USING GIN(dependency_technologies);
CREATE INDEX idx_p028_tr_risk_factors       ON pack028_sector_pathway.gl_technology_roadmaps USING GIN(risk_factors);
CREATE INDEX idx_p028_tr_metadata           ON pack028_sector_pathway.gl_technology_roadmaps USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_technology_roadmaps_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_technology_roadmaps
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_technology_roadmaps ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_tr_tenant_isolation
    ON pack028_sector_pathway.gl_technology_roadmaps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_tr_service_bypass
    ON pack028_sector_pathway.gl_technology_roadmaps
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_technology_roadmaps TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_technology_roadmaps IS
    'Technology transition roadmaps with adoption schedules, IEA milestone tracking, TRL assessment, CapEx phasing, dependency mapping, and risk analysis for sector-specific technology pathways.';

COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.roadmap_id IS 'Unique technology roadmap entry identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.technology_category IS 'Technology category: RENEWABLE_ENERGY, HYDROGEN, CCS_CCUS, ELECTRIFICATION, etc.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.current_trl IS 'Current Technology Readiness Level (1-9 scale).';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.adoption_model IS 'Technology adoption curve model: S_CURVE, LINEAR, EXPONENTIAL, STEPPED.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.marginal_abatement_cost IS 'Cost per tCO2e abated (marginal abatement cost).';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.iea_milestone_ref IS 'Reference to IEA NZE 2050 technology milestone identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.dependency_technologies IS 'Array of prerequisite technologies that must be deployed first.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.implementation_status IS 'Current implementation status: PLANNED, PILOT, SCALING, DEPLOYED, MATURE.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_roadmaps.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

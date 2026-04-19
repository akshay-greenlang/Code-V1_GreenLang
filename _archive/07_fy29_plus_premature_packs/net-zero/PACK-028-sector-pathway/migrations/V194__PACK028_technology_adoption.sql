-- =============================================================================
-- V194: PACK-028 Sector Pathway Pack - Technology Adoption Tracking
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    014 of 015
-- Date:         March 2026
--
-- Technology adoption tracking with TRL progression, deployment status,
-- cost tracking, performance measurement, and milestone compliance
-- for sector-specific technology transitions.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_technology_adoption_tracking
--
-- Previous: V193__PACK028_multi_scenario_modeling.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_technology_adoption_tracking
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_technology_adoption_tracking (
    tracking_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    roadmap_id                  UUID            REFERENCES pack028_sector_pathway.gl_technology_roadmaps(roadmap_id) ON DELETE SET NULL,
    technology_catalog_id       UUID            REFERENCES pack028_sector_pathway.gl_sector_technology_catalog(technology_id) ON DELETE SET NULL,
    -- Sector context
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    -- Technology reference
    technology_name             VARCHAR(255)    NOT NULL,
    technology_code             VARCHAR(60)     NOT NULL,
    technology_category         VARCHAR(50)     NOT NULL,
    -- Reporting period
    reporting_year              INTEGER         NOT NULL,
    reporting_quarter           INTEGER,
    reporting_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    -- TRL progression
    current_trl                 INTEGER         NOT NULL,
    previous_trl                INTEGER,
    trl_change                  INTEGER         GENERATED ALWAYS AS (current_trl - COALESCE(previous_trl, current_trl)) STORED,
    target_trl                  INTEGER         NOT NULL DEFAULT 9,
    trl_target_date             DATE,
    trl_on_track                BOOLEAN,
    -- Deployment status
    deployment_status           VARCHAR(30)     NOT NULL DEFAULT 'PLANNED',
    deployment_phase            VARCHAR(30),
    deployment_start_date       DATE,
    deployment_completion_date  DATE,
    deployment_progress_pct     DECIMAL(5,2)    DEFAULT 0.00,
    -- Penetration / adoption rate
    current_penetration_pct     DECIMAL(6,2)    DEFAULT 0.00,
    planned_penetration_pct     DECIMAL(6,2),
    penetration_gap_pct         DECIMAL(6,2),
    adoption_rate_pct_per_year  DECIMAL(6,2),
    -- Capacity / volume
    installed_capacity          DECIMAL(18,4),
    capacity_unit               VARCHAR(50),
    planned_capacity            DECIMAL(18,4),
    capacity_utilization_pct    DECIMAL(6,2),
    -- Emission reduction achieved
    actual_abatement_tco2e      DECIMAL(18,4),
    planned_abatement_tco2e     DECIMAL(18,4),
    abatement_variance_pct      DECIMAL(8,2),
    intensity_impact            DECIMAL(18,8),
    -- Cost tracking
    actual_capex_usd            DECIMAL(18,2),
    planned_capex_usd           DECIMAL(18,2),
    capex_variance_pct          DECIMAL(8,2),
    actual_opex_usd             DECIMAL(18,2),
    planned_opex_usd            DECIMAL(18,2),
    opex_variance_pct           DECIMAL(8,2),
    total_cost_to_date_usd      DECIMAL(18,2),
    remaining_budget_usd        DECIMAL(18,2),
    actual_cost_per_tco2e       DECIMAL(12,2),
    -- Performance metrics
    energy_efficiency_gain_pct  DECIMAL(6,2),
    availability_pct            DECIMAL(6,2),
    reliability_score           DECIMAL(5,2),
    performance_vs_baseline_pct DECIMAL(8,2),
    -- IEA milestone alignment
    iea_milestone_ref           VARCHAR(100),
    iea_milestone_status        VARCHAR(20),
    iea_gap_description         TEXT,
    -- Risk tracking
    risk_level                  VARCHAR(20)     DEFAULT 'MEDIUM',
    risk_factors                JSONB           DEFAULT '[]',
    blockers                    JSONB           DEFAULT '[]',
    mitigation_actions          JSONB           DEFAULT '[]',
    -- Dependencies status
    dependencies_status         JSONB           DEFAULT '{}',
    blocking_dependencies       TEXT[]          DEFAULT '{}',
    all_dependencies_met        BOOLEAN         DEFAULT FALSE,
    -- Lessons learned
    lessons_learned             TEXT,
    success_factors             TEXT[],
    challenges                  TEXT[],
    -- Next steps
    next_milestone              VARCHAR(255),
    next_milestone_date         DATE,
    next_actions                JSONB           DEFAULT '[]',
    -- Verification
    verification_status         VARCHAR(20)     DEFAULT 'UNVERIFIED',
    verified_by                 VARCHAR(255),
    verification_date           DATE,
    -- Metadata
    is_active                   BOOLEAN         DEFAULT TRUE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_tat_category CHECK (
        technology_category IN (
            'RENEWABLE_ENERGY', 'ENERGY_STORAGE', 'HYDROGEN', 'CCS_CCUS',
            'ELECTRIFICATION', 'FUEL_SWITCHING', 'ENERGY_EFFICIENCY',
            'PROCESS_INNOVATION', 'CIRCULAR_ECONOMY', 'DIGITALIZATION',
            'FLEET_TRANSITION', 'BUILDING_RETROFIT', 'SUSTAINABLE_FUELS',
            'NUCLEAR', 'GRID_INFRASTRUCTURE', 'HEAT_PUMPS', 'BIOMASS',
            'CARBON_REMOVAL', 'NATURE_BASED', 'OTHER'
        )
    ),
    CONSTRAINT chk_p028_tat_trl CHECK (
        current_trl >= 1 AND current_trl <= 9 AND target_trl >= 1 AND target_trl <= 9
    ),
    CONSTRAINT chk_p028_tat_deployment_status CHECK (
        deployment_status IN ('PLANNED', 'PROCUREMENT', 'INSTALLATION', 'COMMISSIONING',
                              'PILOT', 'SCALING', 'OPERATIONAL', 'OPTIMIZING',
                              'MATURE', 'DECOMMISSIONING', 'CANCELLED', 'ON_HOLD')
    ),
    CONSTRAINT chk_p028_tat_deployment_phase CHECK (
        deployment_phase IS NULL OR deployment_phase IN (
            'FEASIBILITY', 'DESIGN', 'ENGINEERING', 'PROCUREMENT',
            'CONSTRUCTION', 'TESTING', 'COMMISSIONING', 'OPERATION'
        )
    ),
    CONSTRAINT chk_p028_tat_progress CHECK (
        deployment_progress_pct >= 0 AND deployment_progress_pct <= 100
    ),
    CONSTRAINT chk_p028_tat_penetration CHECK (
        current_penetration_pct >= 0 AND current_penetration_pct <= 100
    ),
    CONSTRAINT chk_p028_tat_iea_status CHECK (
        iea_milestone_status IS NULL OR iea_milestone_status IN (
            'ON_TRACK', 'BEHIND', 'WELL_BEHIND', 'ACHIEVED', 'NOT_STARTED', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p028_tat_risk CHECK (
        risk_level IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL')
    ),
    CONSTRAINT chk_p028_tat_verification CHECK (
        verification_status IN ('UNVERIFIED', 'INTERNALLY_VERIFIED',
                                'THIRD_PARTY_LIMITED', 'THIRD_PARTY_REASONABLE')
    ),
    CONSTRAINT chk_p028_tat_reporting_year CHECK (
        reporting_year >= 2020 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p028_tat_quarter CHECK (
        reporting_quarter IS NULL OR (reporting_quarter >= 1 AND reporting_quarter <= 4)
    ),
    CONSTRAINT chk_p028_tat_reliability CHECK (
        reliability_score IS NULL OR (reliability_score >= 0 AND reliability_score <= 100)
    ),
    CONSTRAINT chk_p028_tat_availability CHECK (
        availability_pct IS NULL OR (availability_pct >= 0 AND availability_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_tat_tenant            ON pack028_sector_pathway.gl_technology_adoption_tracking(tenant_id);
CREATE INDEX idx_p028_tat_company           ON pack028_sector_pathway.gl_technology_adoption_tracking(company_id);
CREATE INDEX idx_p028_tat_roadmap           ON pack028_sector_pathway.gl_technology_adoption_tracking(roadmap_id);
CREATE INDEX idx_p028_tat_catalog           ON pack028_sector_pathway.gl_technology_adoption_tracking(technology_catalog_id);
CREATE INDEX idx_p028_tat_sector            ON pack028_sector_pathway.gl_technology_adoption_tracking(sector_code);
CREATE INDEX idx_p028_tat_tech_code         ON pack028_sector_pathway.gl_technology_adoption_tracking(technology_code);
CREATE INDEX idx_p028_tat_category          ON pack028_sector_pathway.gl_technology_adoption_tracking(technology_category);
CREATE INDEX idx_p028_tat_year              ON pack028_sector_pathway.gl_technology_adoption_tracking(reporting_year);
CREATE INDEX idx_p028_tat_year_quarter      ON pack028_sector_pathway.gl_technology_adoption_tracking(reporting_year, reporting_quarter);
CREATE INDEX idx_p028_tat_trl               ON pack028_sector_pathway.gl_technology_adoption_tracking(current_trl);
CREATE INDEX idx_p028_tat_deploy_status     ON pack028_sector_pathway.gl_technology_adoption_tracking(deployment_status);
CREATE INDEX idx_p028_tat_risk              ON pack028_sector_pathway.gl_technology_adoption_tracking(risk_level);
CREATE INDEX idx_p028_tat_iea_status        ON pack028_sector_pathway.gl_technology_adoption_tracking(iea_milestone_status);
CREATE INDEX idx_p028_tat_active            ON pack028_sector_pathway.gl_technology_adoption_tracking(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_tat_company_tech      ON pack028_sector_pathway.gl_technology_adoption_tracking(company_id, technology_code, reporting_year);
CREATE INDEX idx_p028_tat_company_sector    ON pack028_sector_pathway.gl_technology_adoption_tracking(company_id, sector_code, reporting_year);
CREATE INDEX idx_p028_tat_behind            ON pack028_sector_pathway.gl_technology_adoption_tracking(iea_milestone_status) WHERE iea_milestone_status IN ('BEHIND', 'WELL_BEHIND');
CREATE INDEX idx_p028_tat_critical_risk     ON pack028_sector_pathway.gl_technology_adoption_tracking(risk_level) WHERE risk_level IN ('CRITICAL', 'HIGH');
CREATE INDEX idx_p028_tat_deps_unmet        ON pack028_sector_pathway.gl_technology_adoption_tracking(all_dependencies_met) WHERE all_dependencies_met = FALSE;
CREATE INDEX idx_p028_tat_next_milestone    ON pack028_sector_pathway.gl_technology_adoption_tracking(next_milestone_date) WHERE next_milestone_date IS NOT NULL;
CREATE INDEX idx_p028_tat_created           ON pack028_sector_pathway.gl_technology_adoption_tracking(created_at DESC);
CREATE INDEX idx_p028_tat_risk_factors      ON pack028_sector_pathway.gl_technology_adoption_tracking USING GIN(risk_factors);
CREATE INDEX idx_p028_tat_blockers          ON pack028_sector_pathway.gl_technology_adoption_tracking USING GIN(blockers);
CREATE INDEX idx_p028_tat_deps_status       ON pack028_sector_pathway.gl_technology_adoption_tracking USING GIN(dependencies_status);
CREATE INDEX idx_p028_tat_metadata          ON pack028_sector_pathway.gl_technology_adoption_tracking USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_tech_adoption_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_technology_adoption_tracking
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_technology_adoption_tracking ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_tat_tenant_isolation
    ON pack028_sector_pathway.gl_technology_adoption_tracking
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_tat_service_bypass
    ON pack028_sector_pathway.gl_technology_adoption_tracking
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_technology_adoption_tracking TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_technology_adoption_tracking IS
    'Technology adoption tracking with TRL progression, deployment status, cost tracking, performance measurement, emission reduction verification, and IEA milestone compliance.';

COMMENT ON COLUMN pack028_sector_pathway.gl_technology_adoption_tracking.tracking_id IS 'Unique tracking record identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_adoption_tracking.current_trl IS 'Current Technology Readiness Level (1-9).';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_adoption_tracking.trl_change IS 'TRL level change from previous period (generated column).';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_adoption_tracking.deployment_status IS 'Current deployment lifecycle stage: PLANNED through OPERATIONAL.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_adoption_tracking.actual_abatement_tco2e IS 'Actual emission reduction achieved by this technology in reporting period.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_adoption_tracking.actual_cost_per_tco2e IS 'Realized cost per tCO2e abated.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_adoption_tracking.iea_milestone_status IS 'Alignment with IEA NZE milestone: ON_TRACK, BEHIND, WELL_BEHIND, ACHIEVED.';
COMMENT ON COLUMN pack028_sector_pathway.gl_technology_adoption_tracking.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

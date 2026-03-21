-- =============================================================================
-- V203: PACK-029 Interim Targets Pack - Initiative Schedule
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    008 of 015
-- Date:         March 2026
--
-- Initiative scheduling for emission reduction projects with planned vs
-- actual timelines, technology readiness levels, budget tracking, critical
-- path identification, and linkage to corrective actions.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_initiative_schedule
--
-- Previous: V202__PACK029_progress_alerts.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_initiative_schedule
-- =============================================================================
-- Initiative schedule records with planned and actual start/completion dates,
-- technology readiness levels, planned reduction quantities, budget tracking,
-- critical path identification, and dependency management.

CREATE TABLE pack029_interim_targets.gl_initiative_schedule (
    schedule_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    -- Initiative identification
    initiative_id               UUID,
    initiative_name             VARCHAR(200)    NOT NULL,
    initiative_code             VARCHAR(30),
    initiative_category         VARCHAR(60),
    initiative_description      TEXT,
    -- Scope and impact
    scope                       VARCHAR(20)     NOT NULL,
    planned_reduction_tco2e_per_year DECIMAL(18,4) NOT NULL,
    actual_reduction_tco2e_per_year  DECIMAL(18,4),
    cumulative_reduction_tco2e  DECIMAL(18,4),
    reduction_ramp_up_months    INTEGER         DEFAULT 0,
    full_impact_year            INTEGER,
    -- Planned timeline
    planned_start_year          INTEGER         NOT NULL,
    planned_start_quarter       VARCHAR(2),
    planned_start_date          DATE,
    planned_completion_year     INTEGER,
    planned_completion_quarter  VARCHAR(2),
    planned_completion_date     DATE,
    planned_duration_months     INTEGER,
    -- Actual timeline
    actual_start_year           INTEGER,
    actual_start_quarter        VARCHAR(2),
    actual_start_date           DATE,
    actual_completion_year      INTEGER,
    actual_completion_quarter   VARCHAR(2),
    actual_completion_date      DATE,
    actual_duration_months      INTEGER,
    -- Schedule variance
    schedule_variance_months    INTEGER,
    schedule_status             VARCHAR(20)     DEFAULT 'ON_SCHEDULE',
    -- Budget
    budget_usd                  DECIMAL(18,2),
    actual_spend_usd            DECIMAL(18,2),
    budget_variance_usd         DECIMAL(18,2),
    budget_variance_pct         DECIMAL(8,2),
    budget_status               VARCHAR(20)     DEFAULT 'ON_BUDGET',
    -- Technology
    technology_trl              INTEGER,
    technology_type             VARCHAR(100),
    technology_provider         VARCHAR(200),
    pilot_required              BOOLEAN         DEFAULT FALSE,
    pilot_completed             BOOLEAN         DEFAULT FALSE,
    -- Critical path and dependencies
    critical_path               BOOLEAN         DEFAULT FALSE,
    dependency_ids              UUID[]          DEFAULT '{}',
    dependencies_met            BOOLEAN         DEFAULT TRUE,
    blocking_issues             JSONB           DEFAULT '[]',
    -- Resource requirements
    fte_required                DECIMAL(6,2),
    external_resources          BOOLEAN         DEFAULT FALSE,
    permits_required            BOOLEAN         DEFAULT FALSE,
    permits_obtained            BOOLEAN         DEFAULT FALSE,
    -- Risk
    risk_level                  VARCHAR(10)     DEFAULT 'MEDIUM',
    risk_factors                JSONB           DEFAULT '[]',
    mitigation_plan             TEXT,
    -- Progress
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PLANNED',
    progress_pct                DECIMAL(5,2)    DEFAULT 0,
    last_progress_update        TIMESTAMPTZ,
    -- Priority
    priority                    INTEGER         DEFAULT 3,
    sequence_order              INTEGER,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_is_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_is_planned_start_year CHECK (
        planned_start_year >= 2000 AND planned_start_year <= 2100
    ),
    CONSTRAINT chk_p029_is_quarters CHECK (
        (planned_start_quarter IS NULL OR planned_start_quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))
        AND (planned_completion_quarter IS NULL OR planned_completion_quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))
        AND (actual_start_quarter IS NULL OR actual_start_quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))
        AND (actual_completion_quarter IS NULL OR actual_completion_quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))
    ),
    CONSTRAINT chk_p029_is_planned_reduction CHECK (
        planned_reduction_tco2e_per_year >= 0
    ),
    CONSTRAINT chk_p029_is_trl CHECK (
        technology_trl IS NULL OR (technology_trl >= 1 AND technology_trl <= 9)
    ),
    CONSTRAINT chk_p029_is_risk_level CHECK (
        risk_level IN ('HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p029_is_status CHECK (
        status IN ('PLANNED', 'APPROVED', 'PROCUREMENT', 'IN_PROGRESS', 'PILOT',
                   'SCALING', 'COMPLETED', 'ON_HOLD', 'CANCELLED', 'DEFERRED')
    ),
    CONSTRAINT chk_p029_is_schedule_status CHECK (
        schedule_status IN ('ON_SCHEDULE', 'AHEAD', 'DELAYED', 'SIGNIFICANTLY_DELAYED', 'NOT_STARTED')
    ),
    CONSTRAINT chk_p029_is_budget_status CHECK (
        budget_status IN ('ON_BUDGET', 'UNDER_BUDGET', 'OVER_BUDGET', 'SIGNIFICANTLY_OVER', 'NOT_ALLOCATED')
    ),
    CONSTRAINT chk_p029_is_progress CHECK (
        progress_pct >= 0 AND progress_pct <= 100
    ),
    CONSTRAINT chk_p029_is_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p029_is_initiative_category CHECK (
        initiative_category IS NULL OR initiative_category IN (
            'ENERGY_EFFICIENCY', 'RENEWABLE_ENERGY', 'FUEL_SWITCHING', 'ELECTRIFICATION',
            'PROCESS_OPTIMIZATION', 'SUPPLY_CHAIN', 'CARBON_CAPTURE', 'OFFSETS',
            'BEHAVIORAL_CHANGE', 'TECHNOLOGY_UPGRADE', 'CIRCULAR_ECONOMY',
            'FLEET_TRANSITION', 'BUILDING_RETROFIT', 'DIGITAL_OPTIMIZATION', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_is_tenant             ON pack029_interim_targets.gl_initiative_schedule(tenant_id);
CREATE INDEX idx_p029_is_org                ON pack029_interim_targets.gl_initiative_schedule(organization_id);
CREATE INDEX idx_p029_is_target             ON pack029_interim_targets.gl_initiative_schedule(target_id);
CREATE INDEX idx_p029_is_initiative         ON pack029_interim_targets.gl_initiative_schedule(initiative_id) WHERE initiative_id IS NOT NULL;
CREATE INDEX idx_p029_is_org_start_year     ON pack029_interim_targets.gl_initiative_schedule(organization_id, planned_start_year);
CREATE INDEX idx_p029_is_org_status         ON pack029_interim_targets.gl_initiative_schedule(organization_id, status);
CREATE INDEX idx_p029_is_critical_path      ON pack029_interim_targets.gl_initiative_schedule(organization_id, planned_start_year) WHERE critical_path = TRUE;
CREATE INDEX idx_p029_is_status             ON pack029_interim_targets.gl_initiative_schedule(status);
CREATE INDEX idx_p029_is_in_progress        ON pack029_interim_targets.gl_initiative_schedule(organization_id) WHERE status = 'IN_PROGRESS';
CREATE INDEX idx_p029_is_delayed            ON pack029_interim_targets.gl_initiative_schedule(organization_id) WHERE schedule_status IN ('DELAYED', 'SIGNIFICANTLY_DELAYED');
CREATE INDEX idx_p029_is_over_budget        ON pack029_interim_targets.gl_initiative_schedule(organization_id) WHERE budget_status IN ('OVER_BUDGET', 'SIGNIFICANTLY_OVER');
CREATE INDEX idx_p029_is_risk               ON pack029_interim_targets.gl_initiative_schedule(risk_level, organization_id);
CREATE INDEX idx_p029_is_trl                ON pack029_interim_targets.gl_initiative_schedule(technology_trl);
CREATE INDEX idx_p029_is_category           ON pack029_interim_targets.gl_initiative_schedule(initiative_category);
CREATE INDEX idx_p029_is_priority           ON pack029_interim_targets.gl_initiative_schedule(priority, organization_id);
CREATE INDEX idx_p029_is_reduction_desc     ON pack029_interim_targets.gl_initiative_schedule(planned_reduction_tco2e_per_year DESC);
CREATE INDEX idx_p029_is_deps_unmet         ON pack029_interim_targets.gl_initiative_schedule(organization_id) WHERE dependencies_met = FALSE;
CREATE INDEX idx_p029_is_created            ON pack029_interim_targets.gl_initiative_schedule(created_at DESC);
CREATE INDEX idx_p029_is_dependency_ids     ON pack029_interim_targets.gl_initiative_schedule USING GIN(dependency_ids);
CREATE INDEX idx_p029_is_blocking_issues    ON pack029_interim_targets.gl_initiative_schedule USING GIN(blocking_issues);
CREATE INDEX idx_p029_is_risk_factors       ON pack029_interim_targets.gl_initiative_schedule USING GIN(risk_factors);
CREATE INDEX idx_p029_is_metadata           ON pack029_interim_targets.gl_initiative_schedule USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_initiative_schedule_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_initiative_schedule
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_initiative_schedule ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_is_tenant_isolation
    ON pack029_interim_targets.gl_initiative_schedule
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_is_service_bypass
    ON pack029_interim_targets.gl_initiative_schedule
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_initiative_schedule TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_initiative_schedule IS
    'Initiative scheduling for emission reduction projects with planned vs actual timelines, technology readiness levels, budget tracking, critical path identification, and dependency management for interim target achievement.';

COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.schedule_id IS 'Unique initiative schedule identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.organization_id IS 'Reference to the organization implementing this initiative.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.initiative_name IS 'Name of the reduction initiative.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.planned_reduction_tco2e_per_year IS 'Planned annual reduction in tonnes CO2 equivalent when fully operational.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.planned_start_year IS 'Planned start year for this initiative.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.budget_usd IS 'Total budget allocation in USD.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.technology_trl IS 'Technology Readiness Level (1-9 scale).';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.critical_path IS 'Whether this initiative is on the critical path for target achievement.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.status IS 'Initiative status: PLANNED through COMPLETED/CANCELLED.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.schedule_status IS 'Schedule adherence: ON_SCHEDULE, AHEAD, DELAYED, SIGNIFICANTLY_DELAYED.';
COMMENT ON COLUMN pack029_interim_targets.gl_initiative_schedule.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

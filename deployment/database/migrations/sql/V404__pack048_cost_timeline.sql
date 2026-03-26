-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V404 - Cost Estimation & Timeline Management
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for assurance cost estimation, engagement timeline
-- milestone tracking, and resource planning. Cost estimates include
-- base cost with multipliers for assurance level, jurisdiction, first-time
-- premium, and Scope 3 complexity. Timeline milestones track planned vs
-- actual dates with phase attribution and dependency management. Resource
-- plans allocate FTE hours and costs by role and phase.
--
-- Tables (3):
--   1. ghg_assurance.gl_ap_cost_estimates
--   2. ghg_assurance.gl_ap_timeline_milestones
--   3. ghg_assurance.gl_ap_resource_plans
--
-- Also includes: indexes, RLS, comments.
-- Previous: V403__pack048_regulatory.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_cost_estimates
-- =============================================================================
-- Cost estimation for assurance engagements. Base cost is adjusted by
-- multipliers for assurance level (reasonable costs 1.5-2x limited),
-- jurisdiction uplift (e.g., US SEC adds compliance premium), first-time
-- premium (initial engagements typically +30-50%), and Scope 3 complexity
-- (upstream/downstream analysis adds effort). Internal FTE hours estimate
-- the organisation's own preparation effort.

CREATE TABLE ghg_assurance.gl_ap_cost_estimates (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    engagement_id               UUID            REFERENCES ghg_assurance.gl_ap_engagements(id) ON DELETE SET NULL,
    base_cost                   NUMERIC(14,2)   NOT NULL,
    assurance_level_multiplier  NUMERIC(4,2)    NOT NULL DEFAULT 1.00,
    jurisdiction_uplift         NUMERIC(4,2)    NOT NULL DEFAULT 1.00,
    first_time_premium          NUMERIC(4,2)    NOT NULL DEFAULT 1.00,
    scope3_complexity           NUMERIC(4,2)    NOT NULL DEFAULT 1.00,
    total_estimated_cost        NUMERIC(14,2)   NOT NULL,
    currency                    TEXT            NOT NULL DEFAULT 'EUR',
    internal_fte_hours          NUMERIC(8,2),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    estimated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_ce_base_cost CHECK (
        base_cost >= 0
    ),
    CONSTRAINT chk_p048_ce_level_mult CHECK (
        assurance_level_multiplier >= 0.50 AND assurance_level_multiplier <= 5.00
    ),
    CONSTRAINT chk_p048_ce_jur_uplift CHECK (
        jurisdiction_uplift >= 0.50 AND jurisdiction_uplift <= 5.00
    ),
    CONSTRAINT chk_p048_ce_ftp CHECK (
        first_time_premium >= 0.50 AND first_time_premium <= 5.00
    ),
    CONSTRAINT chk_p048_ce_scope3 CHECK (
        scope3_complexity >= 0.50 AND scope3_complexity <= 5.00
    ),
    CONSTRAINT chk_p048_ce_total CHECK (
        total_estimated_cost >= 0
    ),
    CONSTRAINT chk_p048_ce_currency CHECK (
        LENGTH(currency) = 3
    ),
    CONSTRAINT chk_p048_ce_fte CHECK (
        internal_fte_hours IS NULL OR internal_fte_hours >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_ce_tenant            ON ghg_assurance.gl_ap_cost_estimates(tenant_id);
CREATE INDEX idx_p048_ce_config            ON ghg_assurance.gl_ap_cost_estimates(config_id);
CREATE INDEX idx_p048_ce_engagement        ON ghg_assurance.gl_ap_cost_estimates(engagement_id);
CREATE INDEX idx_p048_ce_total_cost        ON ghg_assurance.gl_ap_cost_estimates(total_estimated_cost);
CREATE INDEX idx_p048_ce_currency          ON ghg_assurance.gl_ap_cost_estimates(currency);
CREATE INDEX idx_p048_ce_estimated         ON ghg_assurance.gl_ap_cost_estimates(estimated_at DESC);
CREATE INDEX idx_p048_ce_created           ON ghg_assurance.gl_ap_cost_estimates(created_at DESC);
CREATE INDEX idx_p048_ce_provenance        ON ghg_assurance.gl_ap_cost_estimates(provenance_hash);
CREATE INDEX idx_p048_ce_metadata          ON ghg_assurance.gl_ap_cost_estimates USING GIN(metadata);

-- Composite: tenant + config for scoped listing
CREATE INDEX idx_p048_ce_tenant_config     ON ghg_assurance.gl_ap_cost_estimates(tenant_id, config_id);

-- Composite: config + estimated_at for time series
CREATE INDEX idx_p048_ce_config_date       ON ghg_assurance.gl_ap_cost_estimates(config_id, estimated_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_ce_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_cost_estimates
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_timeline_milestones
-- =============================================================================
-- Engagement milestone tracking with planned vs actual dates, phase
-- attribution, status lifecycle, dependencies, and notes. Milestones
-- span the full assurance lifecycle from PLANNING through CLOSEOUT.

CREATE TABLE ghg_assurance.gl_ap_timeline_milestones (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    engagement_id               UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_engagements(id) ON DELETE CASCADE,
    milestone_name              VARCHAR(255)    NOT NULL,
    phase                       VARCHAR(30)     NOT NULL,
    planned_date                DATE            NOT NULL,
    actual_date                 DATE,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    dependencies                JSONB           DEFAULT '[]',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_tm_phase CHECK (
        phase IN (
            'PLANNING', 'RISK_ASSESSMENT', 'FIELDWORK',
            'REPORTING', 'CLOSEOUT'
        )
    ),
    CONSTRAINT chk_p048_tm_status CHECK (
        status IN (
            'PENDING', 'IN_PROGRESS', 'COMPLETED',
            'DELAYED', 'CANCELLED'
        )
    ),
    CONSTRAINT uq_p048_tm_eng_name UNIQUE (engagement_id, milestone_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_tm_tenant            ON ghg_assurance.gl_ap_timeline_milestones(tenant_id);
CREATE INDEX idx_p048_tm_engagement        ON ghg_assurance.gl_ap_timeline_milestones(engagement_id);
CREATE INDEX idx_p048_tm_phase             ON ghg_assurance.gl_ap_timeline_milestones(phase);
CREATE INDEX idx_p048_tm_planned           ON ghg_assurance.gl_ap_timeline_milestones(planned_date);
CREATE INDEX idx_p048_tm_actual            ON ghg_assurance.gl_ap_timeline_milestones(actual_date);
CREATE INDEX idx_p048_tm_status            ON ghg_assurance.gl_ap_timeline_milestones(status);
CREATE INDEX idx_p048_tm_created           ON ghg_assurance.gl_ap_timeline_milestones(created_at DESC);
CREATE INDEX idx_p048_tm_deps              ON ghg_assurance.gl_ap_timeline_milestones USING GIN(dependencies);
CREATE INDEX idx_p048_tm_metadata          ON ghg_assurance.gl_ap_timeline_milestones USING GIN(metadata);

-- Composite: engagement + phase for phase-grouped display
CREATE INDEX idx_p048_tm_eng_phase         ON ghg_assurance.gl_ap_timeline_milestones(engagement_id, phase);

-- Composite: engagement + status for progress tracking
CREATE INDEX idx_p048_tm_eng_status        ON ghg_assurance.gl_ap_timeline_milestones(engagement_id, status);

-- Composite: engagement + planned_date for timeline view
CREATE INDEX idx_p048_tm_eng_planned       ON ghg_assurance.gl_ap_timeline_milestones(engagement_id, planned_date ASC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_tm_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_timeline_milestones
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_assurance.gl_ap_resource_plans
-- =============================================================================
-- Resource allocation per engagement. Tracks FTE hours, hourly rates,
-- total costs by role, phase assignment, and notes. Used for both
-- internal team planning and verifier resource estimation.

CREATE TABLE ghg_assurance.gl_ap_resource_plans (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    engagement_id               UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_engagements(id) ON DELETE CASCADE,
    role_name                   VARCHAR(100)    NOT NULL,
    fte_hours                   NUMERIC(8,2)    NOT NULL,
    hourly_rate                 NUMERIC(8,2),
    total_cost                  NUMERIC(14,2),
    phase                       VARCHAR(30),
    assigned_to                 UUID,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_rp_fte CHECK (
        fte_hours >= 0
    ),
    CONSTRAINT chk_p048_rp_rate CHECK (
        hourly_rate IS NULL OR hourly_rate >= 0
    ),
    CONSTRAINT chk_p048_rp_cost CHECK (
        total_cost IS NULL OR total_cost >= 0
    ),
    CONSTRAINT chk_p048_rp_phase CHECK (
        phase IS NULL OR phase IN (
            'PLANNING', 'RISK_ASSESSMENT', 'FIELDWORK',
            'REPORTING', 'CLOSEOUT'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_rp_tenant            ON ghg_assurance.gl_ap_resource_plans(tenant_id);
CREATE INDEX idx_p048_rp_engagement        ON ghg_assurance.gl_ap_resource_plans(engagement_id);
CREATE INDEX idx_p048_rp_role              ON ghg_assurance.gl_ap_resource_plans(role_name);
CREATE INDEX idx_p048_rp_phase             ON ghg_assurance.gl_ap_resource_plans(phase);
CREATE INDEX idx_p048_rp_assigned          ON ghg_assurance.gl_ap_resource_plans(assigned_to);
CREATE INDEX idx_p048_rp_created           ON ghg_assurance.gl_ap_resource_plans(created_at DESC);
CREATE INDEX idx_p048_rp_metadata          ON ghg_assurance.gl_ap_resource_plans USING GIN(metadata);

-- Composite: engagement + role for role-based summary
CREATE INDEX idx_p048_rp_eng_role          ON ghg_assurance.gl_ap_resource_plans(engagement_id, role_name);

-- Composite: engagement + phase for phase-based budgeting
CREATE INDEX idx_p048_rp_eng_phase         ON ghg_assurance.gl_ap_resource_plans(engagement_id, phase);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_rp_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_resource_plans
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_cost_estimates ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_timeline_milestones ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_resource_plans ENABLE ROW LEVEL SECURITY;

CREATE POLICY p048_ce_tenant_isolation
    ON ghg_assurance.gl_ap_cost_estimates
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_ce_service_bypass
    ON ghg_assurance.gl_ap_cost_estimates
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_tm_tenant_isolation
    ON ghg_assurance.gl_ap_timeline_milestones
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_tm_service_bypass
    ON ghg_assurance.gl_ap_timeline_milestones
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_rp_tenant_isolation
    ON ghg_assurance.gl_ap_resource_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_rp_service_bypass
    ON ghg_assurance.gl_ap_resource_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_cost_estimates TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_timeline_milestones TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_resource_plans TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_assurance.gl_ap_cost_estimates IS
    'Assurance cost estimation with base cost and multipliers for level, jurisdiction, first-time, and Scope 3 complexity.';
COMMENT ON TABLE ghg_assurance.gl_ap_timeline_milestones IS
    'Engagement milestones with planned vs actual dates, phase attribution, status lifecycle, and dependencies.';
COMMENT ON TABLE ghg_assurance.gl_ap_resource_plans IS
    'Resource allocation by role and phase with FTE hours, rates, and cost tracking.';

COMMENT ON COLUMN ghg_assurance.gl_ap_cost_estimates.base_cost IS 'Base assurance cost before multipliers, in specified currency.';
COMMENT ON COLUMN ghg_assurance.gl_ap_cost_estimates.assurance_level_multiplier IS 'Multiplier for assurance level: 1.0 for limited, 1.5-2.0 for reasonable.';
COMMENT ON COLUMN ghg_assurance.gl_ap_cost_estimates.jurisdiction_uplift IS 'Jurisdiction-specific cost uplift: 1.0 baseline, >1.0 for complex regulatory environments.';
COMMENT ON COLUMN ghg_assurance.gl_ap_cost_estimates.first_time_premium IS 'First-time engagement premium: typically 1.3-1.5 for initial engagements, 1.0 for recurring.';
COMMENT ON COLUMN ghg_assurance.gl_ap_cost_estimates.scope3_complexity IS 'Scope 3 complexity multiplier: 1.0 for Scope 1-2 only, 1.3-2.0 for full Scope 3 coverage.';
COMMENT ON COLUMN ghg_assurance.gl_ap_cost_estimates.total_estimated_cost IS 'Total estimated cost = base_cost * level_multiplier * jurisdiction_uplift * first_time_premium * scope3_complexity.';
COMMENT ON COLUMN ghg_assurance.gl_ap_cost_estimates.internal_fte_hours IS 'Estimated internal FTE hours for organisation preparation effort.';
COMMENT ON COLUMN ghg_assurance.gl_ap_timeline_milestones.phase IS 'Engagement phase: PLANNING, RISK_ASSESSMENT, FIELDWORK, REPORTING, CLOSEOUT.';
COMMENT ON COLUMN ghg_assurance.gl_ap_timeline_milestones.status IS 'PENDING (not started), IN_PROGRESS (underway), COMPLETED (done), DELAYED (behind schedule), CANCELLED.';
COMMENT ON COLUMN ghg_assurance.gl_ap_timeline_milestones.dependencies IS 'JSON array of milestone IDs that must complete before this one can start.';
COMMENT ON COLUMN ghg_assurance.gl_ap_resource_plans.fte_hours IS 'Full-time equivalent hours allocated for this role in the specified phase.';
COMMENT ON COLUMN ghg_assurance.gl_ap_resource_plans.total_cost IS 'Total cost = fte_hours * hourly_rate. May be manually overridden.';

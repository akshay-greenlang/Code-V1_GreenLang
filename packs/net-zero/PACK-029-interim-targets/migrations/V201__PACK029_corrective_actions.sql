-- =============================================================================
-- V201: PACK-029 Interim Targets Pack - Corrective Actions
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    006 of 015
-- Date:         March 2026
--
-- Corrective action plans for closing gap-to-target with initiative linkage,
-- expected reduction quantification, cost tracking, risk assessment,
-- and deployment scheduling.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_corrective_actions
--
-- Previous: V200__PACK029_variance_analysis.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_corrective_actions
-- =============================================================================
-- Corrective action records linking gap-to-target shortfalls with specific
-- reduction initiatives, expected abatement, deployment timeline, cost
-- estimates, risk levels, and status tracking.

CREATE TABLE pack029_interim_targets.gl_corrective_actions (
    action_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    variance_id                 UUID            REFERENCES pack029_interim_targets.gl_variance_analysis(variance_id) ON DELETE SET NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    -- Time context
    year                        INTEGER         NOT NULL,
    quarter                     VARCHAR(2),
    -- Gap identification
    gap_to_target_tco2e         DECIMAL(18,4)   NOT NULL,
    gap_to_target_pct           DECIMAL(8,4),
    gap_scope                   VARCHAR(20)     NOT NULL,
    -- Initiative details
    initiative_id               UUID,
    initiative_name             VARCHAR(200)    NOT NULL,
    initiative_category         VARCHAR(60),
    initiative_description      TEXT,
    -- Expected reduction
    expected_reduction_tco2e    DECIMAL(18,4)   NOT NULL,
    expected_reduction_pct      DECIMAL(8,4),
    reduction_confidence        VARCHAR(20)     DEFAULT 'MEDIUM',
    reduction_lower_bound_tco2e DECIMAL(18,4),
    reduction_upper_bound_tco2e DECIMAL(18,4),
    -- Deployment timeline
    deployment_year             INTEGER         NOT NULL,
    deployment_quarter          VARCHAR(2),
    planned_start_date          DATE,
    planned_end_date            DATE,
    actual_start_date           DATE,
    actual_end_date             DATE,
    time_to_impact_months       INTEGER,
    -- Cost
    cost_usd                    DECIMAL(18,2),
    cost_per_tco2e              DECIMAL(12,2),
    budget_allocated            BOOLEAN         DEFAULT FALSE,
    budget_source               VARCHAR(100),
    roi_pct                     DECIMAL(8,2),
    payback_period_years        DECIMAL(6,2),
    -- Risk assessment
    risk_level                  VARCHAR(10)     NOT NULL DEFAULT 'MEDIUM',
    risk_factors                JSONB           DEFAULT '[]',
    dependency_on_external      BOOLEAN         DEFAULT FALSE,
    technology_readiness        INTEGER,
    -- Responsibility
    owner_name                  VARCHAR(255),
    owner_department            VARCHAR(100),
    sponsor_name                VARCHAR(255),
    -- Progress tracking
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PLANNED',
    progress_pct                DECIMAL(5,2)    DEFAULT 0,
    actual_reduction_tco2e      DECIMAL(18,4),
    actual_cost_usd             DECIMAL(18,2),
    -- Priority and ordering
    priority                    INTEGER         DEFAULT 3,
    sequence_order              INTEGER,
    critical_path               BOOLEAN         DEFAULT FALSE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_ca_scope CHECK (
        gap_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_ca_quarter CHECK (
        quarter IS NULL OR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    ),
    CONSTRAINT chk_p029_ca_deploy_quarter CHECK (
        deployment_quarter IS NULL OR deployment_quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    ),
    CONSTRAINT chk_p029_ca_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p029_ca_deploy_year CHECK (
        deployment_year >= 2000 AND deployment_year <= 2100
    ),
    CONSTRAINT chk_p029_ca_gap CHECK (
        gap_to_target_tco2e >= 0
    ),
    CONSTRAINT chk_p029_ca_expected_reduction CHECK (
        expected_reduction_tco2e >= 0
    ),
    CONSTRAINT chk_p029_ca_risk_level CHECK (
        risk_level IN ('HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p029_ca_status CHECK (
        status IN ('PLANNED', 'APPROVED', 'IN_PROGRESS', 'ON_HOLD', 'COMPLETED',
                   'CANCELLED', 'DEFERRED')
    ),
    CONSTRAINT chk_p029_ca_progress CHECK (
        progress_pct >= 0 AND progress_pct <= 100
    ),
    CONSTRAINT chk_p029_ca_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p029_ca_reduction_confidence CHECK (
        reduction_confidence IN ('HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p029_ca_technology_readiness CHECK (
        technology_readiness IS NULL OR (technology_readiness >= 1 AND technology_readiness <= 9)
    ),
    CONSTRAINT chk_p029_ca_initiative_category CHECK (
        initiative_category IS NULL OR initiative_category IN (
            'ENERGY_EFFICIENCY', 'RENEWABLE_ENERGY', 'FUEL_SWITCHING', 'ELECTRIFICATION',
            'PROCESS_OPTIMIZATION', 'SUPPLY_CHAIN', 'CARBON_CAPTURE', 'OFFSETS',
            'BEHAVIORAL_CHANGE', 'TECHNOLOGY_UPGRADE', 'CIRCULAR_ECONOMY', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_ca_tenant             ON pack029_interim_targets.gl_corrective_actions(tenant_id);
CREATE INDEX idx_p029_ca_org                ON pack029_interim_targets.gl_corrective_actions(organization_id);
CREATE INDEX idx_p029_ca_variance           ON pack029_interim_targets.gl_corrective_actions(variance_id);
CREATE INDEX idx_p029_ca_target             ON pack029_interim_targets.gl_corrective_actions(target_id);
CREATE INDEX idx_p029_ca_initiative         ON pack029_interim_targets.gl_corrective_actions(initiative_id) WHERE initiative_id IS NOT NULL;
CREATE INDEX idx_p029_ca_org_deploy_year    ON pack029_interim_targets.gl_corrective_actions(organization_id, deployment_year);
CREATE INDEX idx_p029_ca_org_year           ON pack029_interim_targets.gl_corrective_actions(organization_id, year);
CREATE INDEX idx_p029_ca_status             ON pack029_interim_targets.gl_corrective_actions(status);
CREATE INDEX idx_p029_ca_in_progress        ON pack029_interim_targets.gl_corrective_actions(organization_id) WHERE status = 'IN_PROGRESS';
CREATE INDEX idx_p029_ca_planned            ON pack029_interim_targets.gl_corrective_actions(organization_id, deployment_year) WHERE status = 'PLANNED';
CREATE INDEX idx_p029_ca_risk               ON pack029_interim_targets.gl_corrective_actions(risk_level, organization_id);
CREATE INDEX idx_p029_ca_high_risk          ON pack029_interim_targets.gl_corrective_actions(organization_id) WHERE risk_level = 'HIGH';
CREATE INDEX idx_p029_ca_critical_path      ON pack029_interim_targets.gl_corrective_actions(organization_id, deployment_year) WHERE critical_path = TRUE;
CREATE INDEX idx_p029_ca_priority           ON pack029_interim_targets.gl_corrective_actions(priority, organization_id);
CREATE INDEX idx_p029_ca_category           ON pack029_interim_targets.gl_corrective_actions(initiative_category);
CREATE INDEX idx_p029_ca_cost_per_tco2e     ON pack029_interim_targets.gl_corrective_actions(cost_per_tco2e ASC NULLS LAST);
CREATE INDEX idx_p029_ca_expected_red_desc  ON pack029_interim_targets.gl_corrective_actions(expected_reduction_tco2e DESC);
CREATE INDEX idx_p029_ca_created            ON pack029_interim_targets.gl_corrective_actions(created_at DESC);
CREATE INDEX idx_p029_ca_risk_factors       ON pack029_interim_targets.gl_corrective_actions USING GIN(risk_factors);
CREATE INDEX idx_p029_ca_metadata           ON pack029_interim_targets.gl_corrective_actions USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_corrective_actions_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_corrective_actions
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_corrective_actions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_ca_tenant_isolation
    ON pack029_interim_targets.gl_corrective_actions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_ca_service_bypass
    ON pack029_interim_targets.gl_corrective_actions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_corrective_actions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_corrective_actions IS
    'Corrective action plans linking gap-to-target shortfalls with reduction initiatives, expected abatement, deployment timeline, cost estimates, risk levels, and status tracking for closing interim target gaps.';

COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.action_id IS 'Unique corrective action identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.organization_id IS 'Reference to the organization implementing this action.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.gap_to_target_tco2e IS 'Total gap to target that this action addresses in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.initiative_name IS 'Name of the reduction initiative or corrective action.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.expected_reduction_tco2e IS 'Expected emission reduction from this action in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.deployment_year IS 'Year when this initiative is planned for deployment.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.cost_usd IS 'Total estimated cost in USD.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.cost_per_tco2e IS 'Marginal abatement cost in USD per tonne CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.risk_level IS 'Risk level: HIGH, MEDIUM, LOW.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.status IS 'Action status: PLANNED, APPROVED, IN_PROGRESS, ON_HOLD, COMPLETED, CANCELLED, DEFERRED.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.critical_path IS 'Whether this action is on the critical path for target achievement.';
COMMENT ON COLUMN pack029_interim_targets.gl_corrective_actions.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

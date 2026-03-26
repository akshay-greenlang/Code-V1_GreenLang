-- =============================================================================
-- V363: PACK-044 GHG Inventory Management - Gap Analysis Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Gap analysis and maturity assessment tables for GHG inventory programs.
-- Assesses the organisation's inventory management maturity against a
-- 5-level model (Initial, Repeatable, Defined, Managed, Optimising).
-- Identifies specific data gaps with severity and prioritisation.
-- Generates improvement roadmaps with phased action plans.
--
-- Tables (3):
--   1. ghg_inventory.gl_inv_gap_assessments
--   2. ghg_inventory.gl_inv_data_gaps
--   3. ghg_inventory.gl_inv_improvement_roadmaps
--
-- Previous: V362__pack044_consolidation.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_gap_assessments
-- =============================================================================
-- A periodic assessment of the organisation's GHG inventory management
-- maturity and completeness. Evaluates multiple dimensions (data coverage,
-- methodology, QA/QC, governance, reporting) against a maturity model.
-- Produces an overall maturity score and identifies priority areas.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_gap_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    assessment_name             VARCHAR(300)    NOT NULL,
    assessment_type             VARCHAR(30)     NOT NULL DEFAULT 'COMPREHENSIVE',
    maturity_model              VARCHAR(50)     NOT NULL DEFAULT 'GHG_PROTOCOL_5_LEVEL',
    assessed_by_user_id         UUID,
    assessed_by_name            VARCHAR(255),
    assessment_date             DATE            NOT NULL DEFAULT CURRENT_DATE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'IN_PROGRESS',
    -- Dimension scores (1-5 maturity levels)
    data_coverage_score         NUMERIC(3,1),
    methodology_score           NUMERIC(3,1),
    qaqc_score                  NUMERIC(3,1),
    governance_score            NUMERIC(3,1),
    reporting_score             NUMERIC(3,1),
    verification_score          NUMERIC(3,1),
    systems_score               NUMERIC(3,1),
    stakeholder_score           NUMERIC(3,1),
    overall_maturity_score      NUMERIC(3,1),
    overall_maturity_level      VARCHAR(30),
    previous_overall_score      NUMERIC(3,1),
    score_change                NUMERIC(3,1),
    total_gaps_identified       INTEGER         NOT NULL DEFAULT 0,
    critical_gaps               INTEGER         NOT NULL DEFAULT 0,
    high_gaps                   INTEGER         NOT NULL DEFAULT 0,
    medium_gaps                 INTEGER         NOT NULL DEFAULT 0,
    low_gaps                    INTEGER         NOT NULL DEFAULT 0,
    executive_summary           TEXT,
    recommendations             TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ga_type CHECK (
        assessment_type IN ('COMPREHENSIVE', 'FOCUSED', 'QUICK_SCAN', 'VERIFICATION_READINESS', 'ANNUAL_REVIEW')
    ),
    CONSTRAINT chk_p044_ga_model CHECK (
        maturity_model IN (
            'GHG_PROTOCOL_5_LEVEL', 'ISO_14064_MATURITY', 'CDSB_MATURITY',
            'CUSTOM_3_LEVEL', 'CUSTOM_5_LEVEL'
        )
    ),
    CONSTRAINT chk_p044_ga_status CHECK (
        status IN ('PLANNED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED')
    ),
    CONSTRAINT chk_p044_ga_maturity_level CHECK (
        overall_maturity_level IS NULL OR overall_maturity_level IN (
            'INITIAL', 'REPEATABLE', 'DEFINED', 'MANAGED', 'OPTIMISING'
        )
    ),
    CONSTRAINT chk_p044_ga_scores CHECK (
        (data_coverage_score IS NULL OR (data_coverage_score >= 1 AND data_coverage_score <= 5)) AND
        (methodology_score IS NULL OR (methodology_score >= 1 AND methodology_score <= 5)) AND
        (qaqc_score IS NULL OR (qaqc_score >= 1 AND qaqc_score <= 5)) AND
        (governance_score IS NULL OR (governance_score >= 1 AND governance_score <= 5)) AND
        (reporting_score IS NULL OR (reporting_score >= 1 AND reporting_score <= 5)) AND
        (verification_score IS NULL OR (verification_score >= 1 AND verification_score <= 5)) AND
        (systems_score IS NULL OR (systems_score >= 1 AND systems_score <= 5)) AND
        (stakeholder_score IS NULL OR (stakeholder_score >= 1 AND stakeholder_score <= 5)) AND
        (overall_maturity_score IS NULL OR (overall_maturity_score >= 1 AND overall_maturity_score <= 5))
    ),
    CONSTRAINT chk_p044_ga_gaps CHECK (
        total_gaps_identified >= 0 AND critical_gaps >= 0 AND
        high_gaps >= 0 AND medium_gaps >= 0 AND low_gaps >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ga_tenant          ON ghg_inventory.gl_inv_gap_assessments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ga_period          ON ghg_inventory.gl_inv_gap_assessments(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_ga_type            ON ghg_inventory.gl_inv_gap_assessments(assessment_type);
CREATE INDEX IF NOT EXISTS idx_p044_ga_status          ON ghg_inventory.gl_inv_gap_assessments(status);
CREATE INDEX IF NOT EXISTS idx_p044_ga_maturity        ON ghg_inventory.gl_inv_gap_assessments(overall_maturity_level);
CREATE INDEX IF NOT EXISTS idx_p044_ga_date            ON ghg_inventory.gl_inv_gap_assessments(assessment_date DESC);
CREATE INDEX IF NOT EXISTS idx_p044_ga_created         ON ghg_inventory.gl_inv_gap_assessments(created_at DESC);

-- Composite: period + completed assessments
CREATE INDEX IF NOT EXISTS idx_p044_ga_period_done     ON ghg_inventory.gl_inv_gap_assessments(period_id, assessment_date DESC)
    WHERE status = 'COMPLETED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ga_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_gap_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_data_gaps
-- =============================================================================
-- Individual data gaps identified during a gap assessment. Each gap describes
-- a specific deficiency in the inventory (missing data, low quality, missing
-- methodology, etc.) with severity, affected scope, and prioritisation.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_data_gaps (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_gap_assessments(id) ON DELETE CASCADE,
    gap_code                    VARCHAR(50)     NOT NULL,
    gap_title                   VARCHAR(300)    NOT NULL,
    gap_description             TEXT            NOT NULL,
    gap_category                VARCHAR(50)     NOT NULL,
    dimension                   VARCHAR(30)     NOT NULL,
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    affected_scope              VARCHAR(10),
    affected_category           VARCHAR(60),
    facility_id                 UUID,
    impact_description          TEXT,
    estimated_emissions_gap_tco2e NUMERIC(12,3),
    current_state               TEXT,
    desired_state               TEXT,
    priority_rank               INTEGER,
    effort_estimate             VARCHAR(20),
    timeline_estimate           VARCHAR(30),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'OPEN',
    assigned_to_user_id         UUID,
    assigned_to_name            VARCHAR(255),
    resolved_at                 TIMESTAMPTZ,
    resolution_notes            TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_dg_category CHECK (
        gap_category IN (
            'MISSING_DATA', 'LOW_QUALITY_DATA', 'MISSING_METHODOLOGY',
            'INCOMPLETE_BOUNDARY', 'MISSING_EMISSION_FACTOR', 'MISSING_QC_PROCEDURE',
            'GOVERNANCE_GAP', 'REPORTING_GAP', 'VERIFICATION_GAP',
            'SYSTEM_GAP', 'PROCESS_GAP', 'COMPETENCY_GAP', 'OTHER'
        )
    ),
    CONSTRAINT chk_p044_dg_dimension CHECK (
        dimension IN (
            'DATA_COVERAGE', 'METHODOLOGY', 'QAQC', 'GOVERNANCE',
            'REPORTING', 'VERIFICATION', 'SYSTEMS', 'STAKEHOLDER'
        )
    ),
    CONSTRAINT chk_p044_dg_severity CHECK (
        severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p044_dg_scope CHECK (
        affected_scope IS NULL OR affected_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ALL')
    ),
    CONSTRAINT chk_p044_dg_effort CHECK (
        effort_estimate IS NULL OR effort_estimate IN ('MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p044_dg_timeline CHECK (
        timeline_estimate IS NULL OR timeline_estimate IN (
            'IMMEDIATE', 'WITHIN_1_MONTH', 'WITHIN_3_MONTHS',
            'WITHIN_6_MONTHS', 'WITHIN_1_YEAR', 'BEYOND_1_YEAR'
        )
    ),
    CONSTRAINT chk_p044_dg_status CHECK (
        status IN ('OPEN', 'IN_PROGRESS', 'RESOLVED', 'ACCEPTED_RISK', 'DEFERRED', 'CLOSED')
    ),
    CONSTRAINT chk_p044_dg_rank CHECK (
        priority_rank IS NULL OR priority_rank >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_dg_tenant          ON ghg_inventory.gl_inv_data_gaps(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_dg_assessment      ON ghg_inventory.gl_inv_data_gaps(assessment_id);
CREATE INDEX IF NOT EXISTS idx_p044_dg_code            ON ghg_inventory.gl_inv_data_gaps(gap_code);
CREATE INDEX IF NOT EXISTS idx_p044_dg_category        ON ghg_inventory.gl_inv_data_gaps(gap_category);
CREATE INDEX IF NOT EXISTS idx_p044_dg_dimension       ON ghg_inventory.gl_inv_data_gaps(dimension);
CREATE INDEX IF NOT EXISTS idx_p044_dg_severity        ON ghg_inventory.gl_inv_data_gaps(severity);
CREATE INDEX IF NOT EXISTS idx_p044_dg_scope           ON ghg_inventory.gl_inv_data_gaps(affected_scope);
CREATE INDEX IF NOT EXISTS idx_p044_dg_facility        ON ghg_inventory.gl_inv_data_gaps(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_dg_status          ON ghg_inventory.gl_inv_data_gaps(status);
CREATE INDEX IF NOT EXISTS idx_p044_dg_priority        ON ghg_inventory.gl_inv_data_gaps(priority_rank);
CREATE INDEX IF NOT EXISTS idx_p044_dg_created         ON ghg_inventory.gl_inv_data_gaps(created_at DESC);

-- Composite: assessment + open gaps
CREATE INDEX IF NOT EXISTS idx_p044_dg_assess_open     ON ghg_inventory.gl_inv_data_gaps(assessment_id, severity)
    WHERE status IN ('OPEN', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_dg_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_data_gaps
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_improvement_roadmaps
-- =============================================================================
-- Phased improvement plans generated from gap assessments. Roadmaps define
-- a multi-phase strategy to close identified gaps and advance the
-- organisation's inventory management maturity level.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_improvement_roadmaps (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_gap_assessments(id) ON DELETE CASCADE,
    roadmap_name                VARCHAR(300)    NOT NULL,
    phase                       VARCHAR(30)     NOT NULL,
    phase_order                 INTEGER         NOT NULL DEFAULT 1,
    phase_title                 VARCHAR(200)    NOT NULL,
    phase_description           TEXT,
    target_maturity_level       VARCHAR(30),
    start_date                  DATE,
    end_date                    DATE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PLANNED',
    total_actions               INTEGER         NOT NULL DEFAULT 0,
    completed_actions           INTEGER         NOT NULL DEFAULT 0,
    completion_pct              NUMERIC(5,2)    DEFAULT 0.00,
    estimated_cost              NUMERIC(14,2),
    cost_currency               VARCHAR(3)      DEFAULT 'USD',
    resource_requirements       TEXT,
    dependencies                TEXT,
    risks                       TEXT,
    success_criteria             TEXT,
    owner_user_id               UUID,
    owner_name                  VARCHAR(255),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ir_phase CHECK (
        phase IN ('PHASE_1', 'PHASE_2', 'PHASE_3', 'PHASE_4', 'QUICK_WIN', 'FOUNDATION', 'ADVANCED')
    ),
    CONSTRAINT chk_p044_ir_order CHECK (
        phase_order >= 1 AND phase_order <= 20
    ),
    CONSTRAINT chk_p044_ir_maturity CHECK (
        target_maturity_level IS NULL OR target_maturity_level IN (
            'INITIAL', 'REPEATABLE', 'DEFINED', 'MANAGED', 'OPTIMISING'
        )
    ),
    CONSTRAINT chk_p044_ir_status CHECK (
        status IN ('PLANNED', 'IN_PROGRESS', 'COMPLETED', 'ON_HOLD', 'CANCELLED')
    ),
    CONSTRAINT chk_p044_ir_dates CHECK (
        start_date IS NULL OR end_date IS NULL OR start_date <= end_date
    ),
    CONSTRAINT chk_p044_ir_actions CHECK (
        total_actions >= 0 AND completed_actions >= 0 AND completed_actions <= total_actions
    ),
    CONSTRAINT chk_p044_ir_completion CHECK (
        completion_pct IS NULL OR (completion_pct >= 0 AND completion_pct <= 100)
    ),
    CONSTRAINT chk_p044_ir_cost CHECK (
        estimated_cost IS NULL OR estimated_cost >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ir_tenant          ON ghg_inventory.gl_inv_improvement_roadmaps(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ir_assessment      ON ghg_inventory.gl_inv_improvement_roadmaps(assessment_id);
CREATE INDEX IF NOT EXISTS idx_p044_ir_phase           ON ghg_inventory.gl_inv_improvement_roadmaps(phase);
CREATE INDEX IF NOT EXISTS idx_p044_ir_status          ON ghg_inventory.gl_inv_improvement_roadmaps(status);
CREATE INDEX IF NOT EXISTS idx_p044_ir_maturity        ON ghg_inventory.gl_inv_improvement_roadmaps(target_maturity_level);
CREATE INDEX IF NOT EXISTS idx_p044_ir_owner           ON ghg_inventory.gl_inv_improvement_roadmaps(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_ir_dates           ON ghg_inventory.gl_inv_improvement_roadmaps(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_p044_ir_created         ON ghg_inventory.gl_inv_improvement_roadmaps(created_at DESC);

-- Composite: assessment + active phases
CREATE INDEX IF NOT EXISTS idx_p044_ir_assess_active   ON ghg_inventory.gl_inv_improvement_roadmaps(assessment_id, phase_order)
    WHERE status IN ('PLANNED', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ir_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_improvement_roadmaps
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_gap_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_data_gaps ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_improvement_roadmaps ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_ga_tenant_isolation
    ON ghg_inventory.gl_inv_gap_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ga_service_bypass
    ON ghg_inventory.gl_inv_gap_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_dg_tenant_isolation
    ON ghg_inventory.gl_inv_data_gaps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_dg_service_bypass
    ON ghg_inventory.gl_inv_data_gaps
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_ir_tenant_isolation
    ON ghg_inventory.gl_inv_improvement_roadmaps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ir_service_bypass
    ON ghg_inventory.gl_inv_improvement_roadmaps
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_gap_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_data_gaps TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_improvement_roadmaps TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_gap_assessments IS
    'Periodic assessments of GHG inventory management maturity across eight dimensions with overall scoring.';
COMMENT ON TABLE ghg_inventory.gl_inv_data_gaps IS
    'Individual data gaps identified during assessments with severity, affected scope, and prioritisation.';
COMMENT ON TABLE ghg_inventory.gl_inv_improvement_roadmaps IS
    'Phased improvement plans to close identified gaps and advance inventory management maturity.';

COMMENT ON COLUMN ghg_inventory.gl_inv_gap_assessments.maturity_model IS 'Maturity framework: GHG_PROTOCOL_5_LEVEL, ISO_14064_MATURITY, CDSB_MATURITY, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_gap_assessments.overall_maturity_level IS 'Current maturity level: INITIAL (1), REPEATABLE (2), DEFINED (3), MANAGED (4), OPTIMISING (5).';
COMMENT ON COLUMN ghg_inventory.gl_inv_data_gaps.gap_category IS 'Category: MISSING_DATA, LOW_QUALITY_DATA, MISSING_METHODOLOGY, GOVERNANCE_GAP, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_data_gaps.dimension IS 'Maturity dimension: DATA_COVERAGE, METHODOLOGY, QAQC, GOVERNANCE, REPORTING, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_data_gaps.effort_estimate IS 'Estimated effort to close the gap: MINIMAL, LOW, MEDIUM, HIGH, VERY_HIGH.';
COMMENT ON COLUMN ghg_inventory.gl_inv_improvement_roadmaps.target_maturity_level IS 'Target maturity level upon completion of this phase.';

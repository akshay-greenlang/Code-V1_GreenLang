-- =============================================================================
-- V370: PACK-045 Base Year Management Pack - Significance Assessment
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the significance assessment table that evaluates whether a detected
-- trigger event exceeds the recalculation policy threshold. Each assessment
-- calculates the emission impact as a percentage of the base year total and
-- compares it against the configured significance threshold. Supports multiple
-- assessment methods (percentage, absolute, cumulative) and sensitivity
-- analysis for borderline cases.
--
-- Tables (1):
--   1. ghg_base_year.gl_by_significance_assessments
--
-- Also includes: indexes, RLS, comments.
-- Previous: V369__pack045_triggers.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_significance_assessments
-- =============================================================================
-- Evaluates whether a trigger event's emission impact exceeds the defined
-- significance threshold. The assessment compares the impact tCO2e against
-- the base year total to calculate a significance percentage. Multiple
-- methods may be applied (percentage of total, absolute threshold, cumulative
-- check). Sensitivity analysis captures alternative scenarios for borderline
-- cases.

CREATE TABLE ghg_base_year.gl_by_significance_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    trigger_id                  UUID            NOT NULL REFERENCES ghg_base_year.gl_by_triggers(id) ON DELETE CASCADE,
    policy_id                   UUID            REFERENCES ghg_base_year.gl_by_policies(id) ON DELETE SET NULL,
    assessment_date             DATE            NOT NULL DEFAULT CURRENT_DATE,
    assessed_by                 VARCHAR(255),
    method                      VARCHAR(30)     NOT NULL DEFAULT 'PERCENTAGE',
    impact_tco2e                NUMERIC(14,3)   NOT NULL,
    base_year_total_tco2e       NUMERIC(14,3)   NOT NULL,
    significance_pct            NUMERIC(8,4)    GENERATED ALWAYS AS (
        CASE WHEN base_year_total_tco2e > 0
             THEN (impact_tco2e / base_year_total_tco2e) * 100
             ELSE 0
        END
    ) STORED,
    threshold_pct               NUMERIC(5,2)    NOT NULL,
    threshold_tco2e             NUMERIC(12,3),
    outcome                     VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    outcome_rationale           TEXT,
    cumulative_impact_tco2e     NUMERIC(14,3),
    cumulative_pct              NUMERIC(8,4),
    cumulative_threshold_pct    NUMERIC(5,2),
    cumulative_outcome          VARCHAR(30),
    scope_breakdown_json        JSONB           DEFAULT '{}',
    sensitivity_json            JSONB           DEFAULT '{}',
    alternative_scenarios_json  JSONB           DEFAULT '{}',
    peer_review_status          VARCHAR(30),
    peer_reviewer               VARCHAR(255),
    peer_review_date            DATE,
    peer_review_notes           TEXT,
    evidence_refs               TEXT[],
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_sa_method CHECK (
        method IN ('PERCENTAGE', 'ABSOLUTE', 'CUMULATIVE', 'COMBINED', 'EXPERT_JUDGEMENT')
    ),
    CONSTRAINT chk_p045_sa_impact CHECK (
        impact_tco2e >= 0
    ),
    CONSTRAINT chk_p045_sa_base_total CHECK (
        base_year_total_tco2e >= 0
    ),
    CONSTRAINT chk_p045_sa_threshold CHECK (
        threshold_pct > 0 AND threshold_pct <= 100
    ),
    CONSTRAINT chk_p045_sa_outcome CHECK (
        outcome IN (
            'PENDING', 'SIGNIFICANT', 'NOT_SIGNIFICANT', 'BORDERLINE',
            'DE_MINIMIS', 'REQUIRES_EXPERT_REVIEW', 'DEFERRED'
        )
    ),
    CONSTRAINT chk_p045_sa_cumulative_outcome CHECK (
        cumulative_outcome IS NULL OR cumulative_outcome IN (
            'SIGNIFICANT', 'NOT_SIGNIFICANT', 'BORDERLINE', 'NOT_ASSESSED'
        )
    ),
    CONSTRAINT chk_p045_sa_peer_review CHECK (
        peer_review_status IS NULL OR peer_review_status IN (
            'NOT_REQUIRED', 'PENDING', 'IN_PROGRESS', 'APPROVED', 'REJECTED'
        )
    ),
    CONSTRAINT chk_p045_sa_cumulative_impact CHECK (
        cumulative_impact_tco2e IS NULL OR cumulative_impact_tco2e >= 0
    ),
    CONSTRAINT chk_p045_sa_cumulative_pct CHECK (
        cumulative_pct IS NULL OR (cumulative_pct >= 0 AND cumulative_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_sa_tenant          ON ghg_base_year.gl_by_significance_assessments(tenant_id);
CREATE INDEX idx_p045_sa_trigger         ON ghg_base_year.gl_by_significance_assessments(trigger_id);
CREATE INDEX idx_p045_sa_policy          ON ghg_base_year.gl_by_significance_assessments(policy_id);
CREATE INDEX idx_p045_sa_method          ON ghg_base_year.gl_by_significance_assessments(method);
CREATE INDEX idx_p045_sa_outcome         ON ghg_base_year.gl_by_significance_assessments(outcome);
CREATE INDEX idx_p045_sa_date            ON ghg_base_year.gl_by_significance_assessments(assessment_date);
CREATE INDEX idx_p045_sa_peer_review     ON ghg_base_year.gl_by_significance_assessments(peer_review_status);
CREATE INDEX idx_p045_sa_created         ON ghg_base_year.gl_by_significance_assessments(created_at DESC);
CREATE INDEX idx_p045_sa_sensitivity     ON ghg_base_year.gl_by_significance_assessments USING GIN(sensitivity_json);
CREATE INDEX idx_p045_sa_metadata        ON ghg_base_year.gl_by_significance_assessments USING GIN(metadata);

-- Composite: trigger + outcome for workflow queries
CREATE INDEX idx_p045_sa_trigger_outcome ON ghg_base_year.gl_by_significance_assessments(trigger_id, outcome);

-- Composite: significant outcomes pending action
CREATE INDEX idx_p045_sa_significant     ON ghg_base_year.gl_by_significance_assessments(outcome)
    WHERE outcome IN ('SIGNIFICANT', 'BORDERLINE', 'REQUIRES_EXPERT_REVIEW');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_sa_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_significance_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_significance_assessments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_sa_tenant_isolation
    ON ghg_base_year.gl_by_significance_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_sa_service_bypass
    ON ghg_base_year.gl_by_significance_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_significance_assessments TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_base_year.gl_by_significance_assessments IS
    'Significance assessments evaluating whether trigger events exceed recalculation thresholds, with sensitivity analysis for borderline cases.';

COMMENT ON COLUMN ghg_base_year.gl_by_significance_assessments.significance_pct IS 'Auto-calculated: (impact_tco2e / base_year_total_tco2e) * 100. Compared against threshold_pct.';
COMMENT ON COLUMN ghg_base_year.gl_by_significance_assessments.threshold_pct IS 'Policy threshold percentage against which significance_pct is compared.';
COMMENT ON COLUMN ghg_base_year.gl_by_significance_assessments.outcome IS 'Assessment result: SIGNIFICANT (recalculate), NOT_SIGNIFICANT (no action), BORDERLINE (review), DE_MINIMIS (below absolute threshold).';
COMMENT ON COLUMN ghg_base_year.gl_by_significance_assessments.sensitivity_json IS 'Alternative calculations showing impact under different assumptions for borderline assessments.';
COMMENT ON COLUMN ghg_base_year.gl_by_significance_assessments.cumulative_impact_tco2e IS 'Sum of all non-significant triggers since last recalculation, compared against cumulative threshold.';

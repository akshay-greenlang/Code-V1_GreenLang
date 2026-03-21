-- =============================================================================
-- V149: PACK-025 Race to Zero - Pledges & Starting Line Assessments
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    002 of 010
-- Date:         March 2026
--
-- Pledge commitment records with eligibility scoring and quality rating,
-- plus Starting Line Criteria assessments across the four pillars
-- (Pledge, Plan, Proceed, Publish) with gap analysis and remediation plans.
--
-- Tables (2):
--   1. pack025_race_to_zero.pledges
--   2. pack025_race_to_zero.starting_line_assessments
--
-- Previous: V148__pack025_race_to_zero_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.pledges
-- =============================================================================
-- Race to Zero pledge commitments with target years, eligibility scoring,
-- and quality assessment.

CREATE TABLE pack025_race_to_zero.pledges (
    pledge_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    pledge_date             DATE            NOT NULL,
    target_year_interim     INTEGER         NOT NULL DEFAULT 2030,
    target_year_longterm    INTEGER         NOT NULL DEFAULT 2050,
    net_zero_definition     TEXT,
    scope_coverage          VARCHAR(30)     NOT NULL DEFAULT 'ALL_SCOPES',
    status                  VARCHAR(30)     NOT NULL DEFAULT 'draft',
    eligibility_score       DECIMAL(6,2),
    quality_rating          VARCHAR(30),
    criterion_results       JSONB           DEFAULT '{}',
    pledge_statement        TEXT,
    governance_approved     BOOLEAN         DEFAULT FALSE,
    governance_approval_date DATE,
    public_disclosure       BOOLEAN         DEFAULT FALSE,
    public_disclosure_url   TEXT,
    commitment_letter_url   TEXT,
    partner_initiative      VARCHAR(100),
    submission_date         DATE,
    confirmation_date       DATE,
    warnings                TEXT[]          DEFAULT '{}',
    errors                  TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_pledge_status CHECK (
        status IN ('draft', 'submitted', 'confirmed', 'active', 'suspended', 'withdrawn')
    ),
    CONSTRAINT chk_p025_pledge_quality CHECK (
        quality_rating IS NULL OR quality_rating IN ('STRONG', 'ADEQUATE', 'WEAK', 'INELIGIBLE')
    ),
    CONSTRAINT chk_p025_pledge_scope CHECK (
        scope_coverage IN ('ALL_SCOPES', 'SCOPE_1_2', 'SCOPE_1_2_MATERIAL_3')
    ),
    CONSTRAINT chk_p025_pledge_interim_year CHECK (
        target_year_interim >= 2025 AND target_year_interim <= 2040
    ),
    CONSTRAINT chk_p025_pledge_longterm_year CHECK (
        target_year_longterm >= 2040 AND target_year_longterm <= 2060
    ),
    CONSTRAINT chk_p025_pledge_year_order CHECK (
        target_year_interim < target_year_longterm
    ),
    CONSTRAINT chk_p025_pledge_eligibility_range CHECK (
        eligibility_score IS NULL OR (eligibility_score >= 0 AND eligibility_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_pledges_org       ON pack025_race_to_zero.pledges(org_id);
CREATE INDEX idx_p025_pledges_tenant    ON pack025_race_to_zero.pledges(tenant_id);
CREATE INDEX idx_p025_pledges_date      ON pack025_race_to_zero.pledges(pledge_date);
CREATE INDEX idx_p025_pledges_status    ON pack025_race_to_zero.pledges(status);
CREATE INDEX idx_p025_pledges_quality   ON pack025_race_to_zero.pledges(quality_rating);
CREATE INDEX idx_p025_pledges_interim   ON pack025_race_to_zero.pledges(target_year_interim);
CREATE INDEX idx_p025_pledges_longterm  ON pack025_race_to_zero.pledges(target_year_longterm);
CREATE INDEX idx_p025_pledges_partner   ON pack025_race_to_zero.pledges(partner_initiative);
CREATE INDEX idx_p025_pledges_created   ON pack025_race_to_zero.pledges(created_at DESC);
CREATE INDEX idx_p025_pledges_criteria  ON pack025_race_to_zero.pledges USING GIN(criterion_results);
CREATE INDEX idx_p025_pledges_metadata  ON pack025_race_to_zero.pledges USING GIN(metadata);

-- =============================================================================
-- Table 2: pack025_race_to_zero.starting_line_assessments
-- =============================================================================
-- Starting Line Criteria compliance assessment across the four pillars:
-- Pledge, Plan, Proceed, Publish. Includes gap analysis and remediation.

CREATE TABLE pack025_race_to_zero.starting_line_assessments (
    assessment_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    pledge_id               UUID            NOT NULL REFERENCES pack025_race_to_zero.pledges(pledge_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    pledge_compliance       DECIMAL(6,2)    NOT NULL DEFAULT 0,
    plan_compliance         DECIMAL(6,2)    NOT NULL DEFAULT 0,
    proceed_compliance      DECIMAL(6,2)    NOT NULL DEFAULT 0,
    publish_compliance      DECIMAL(6,2)    NOT NULL DEFAULT 0,
    overall_compliance      DECIMAL(6,2)    GENERATED ALWAYS AS (
        (pledge_compliance + plan_compliance + proceed_compliance + publish_compliance) / 4.0
    ) STORED,
    overall_status          VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    criteria_passed         INTEGER         DEFAULT 0,
    criteria_failed         INTEGER         DEFAULT 0,
    criteria_total          INTEGER         DEFAULT 20,
    gaps_json               JSONB           NOT NULL DEFAULT '[]',
    remediation_plan        TEXT,
    remediation_deadline    DATE,
    evidence_collected      JSONB           DEFAULT '{}',
    assessor_id             UUID,
    assessor_notes          TEXT,
    warnings                TEXT[]          DEFAULT '{}',
    errors                  TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_sl_status CHECK (
        overall_status IN ('COMPLIANT', 'NON_COMPLIANT', 'PARTIALLY_COMPLIANT', 'PENDING')
    ),
    CONSTRAINT chk_p025_sl_pledge_comp CHECK (
        pledge_compliance >= 0 AND pledge_compliance <= 100
    ),
    CONSTRAINT chk_p025_sl_plan_comp CHECK (
        plan_compliance >= 0 AND plan_compliance <= 100
    ),
    CONSTRAINT chk_p025_sl_proceed_comp CHECK (
        proceed_compliance >= 0 AND proceed_compliance <= 100
    ),
    CONSTRAINT chk_p025_sl_publish_comp CHECK (
        publish_compliance >= 0 AND publish_compliance <= 100
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_sl_org            ON pack025_race_to_zero.starting_line_assessments(org_id);
CREATE INDEX idx_p025_sl_pledge         ON pack025_race_to_zero.starting_line_assessments(pledge_id);
CREATE INDEX idx_p025_sl_tenant         ON pack025_race_to_zero.starting_line_assessments(tenant_id);
CREATE INDEX idx_p025_sl_date           ON pack025_race_to_zero.starting_line_assessments(assessment_date);
CREATE INDEX idx_p025_sl_status         ON pack025_race_to_zero.starting_line_assessments(overall_status);
CREATE INDEX idx_p025_sl_created        ON pack025_race_to_zero.starting_line_assessments(created_at DESC);
CREATE INDEX idx_p025_sl_gaps           ON pack025_race_to_zero.starting_line_assessments USING GIN(gaps_json);
CREATE INDEX idx_p025_sl_evidence       ON pack025_race_to_zero.starting_line_assessments USING GIN(evidence_collected);
CREATE INDEX idx_p025_sl_metadata       ON pack025_race_to_zero.starting_line_assessments USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_pledges_updated
    BEFORE UPDATE ON pack025_race_to_zero.pledges
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

CREATE TRIGGER trg_p025_sl_assessments_updated
    BEFORE UPDATE ON pack025_race_to_zero.starting_line_assessments
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.pledges ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.starting_line_assessments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_pledges_tenant_isolation
    ON pack025_race_to_zero.pledges
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_pledges_service_bypass
    ON pack025_race_to_zero.pledges
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_sl_assess_tenant_isolation
    ON pack025_race_to_zero.starting_line_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_sl_assess_service_bypass
    ON pack025_race_to_zero.starting_line_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.pledges TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.starting_line_assessments TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.pledges IS
    'Race to Zero pledge commitments with interim/long-term target years, eligibility scoring, and quality ratings.';
COMMENT ON TABLE pack025_race_to_zero.starting_line_assessments IS
    'Starting Line Criteria compliance across Pledge/Plan/Proceed/Publish pillars with gap analysis and remediation tracking.';

COMMENT ON COLUMN pack025_race_to_zero.pledges.pledge_id IS 'Unique pledge identifier.';
COMMENT ON COLUMN pack025_race_to_zero.pledges.eligibility_score IS 'Overall eligibility score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.pledges.quality_rating IS 'Pledge quality rating: STRONG, ADEQUATE, WEAK, INELIGIBLE.';
COMMENT ON COLUMN pack025_race_to_zero.starting_line_assessments.pledge_compliance IS 'Pledge pillar compliance score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.starting_line_assessments.plan_compliance IS 'Plan pillar compliance score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.starting_line_assessments.proceed_compliance IS 'Proceed pillar compliance score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.starting_line_assessments.publish_compliance IS 'Publish pillar compliance score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.starting_line_assessments.gaps_json IS 'JSONB array of identified compliance gaps with severity, criterion_id, and description.';
COMMENT ON COLUMN pack025_race_to_zero.starting_line_assessments.remediation_plan IS 'Text description of the remediation plan to close identified gaps.';

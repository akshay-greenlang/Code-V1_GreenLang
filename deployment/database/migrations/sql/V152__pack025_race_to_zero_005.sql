-- =============================================================================
-- V152: PACK-025 Race to Zero - Annual Reports & Progress Tracking
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    005 of 010
-- Date:         March 2026
--
-- Annual emission reporting with scope-level actuals, variance analysis,
-- verification tracking. Progress tracking with reduction gaps, credibility
-- scoring, and recommendations.
--
-- Tables (2):
--   1. pack025_race_to_zero.annual_reports
--   2. pack025_race_to_zero.progress_tracking
--
-- Previous: V151__pack025_race_to_zero_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.annual_reports
-- =============================================================================
-- Annual emission reports with scope-level actuals, target comparison,
-- variance analysis, verification status, and third-party assurance.

CREATE TABLE pack025_race_to_zero.annual_reports (
    report_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    pledge_id               UUID            NOT NULL REFERENCES pack025_race_to_zero.pledges(pledge_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    reporting_year          INTEGER         NOT NULL,
    report_date             DATE,
    -- Actual emissions
    actual_emissions_s1     DECIMAL(18,4),
    actual_emissions_s2     DECIMAL(18,4),
    actual_emissions_s3     DECIMAL(18,4),
    total_actual_tco2e      DECIMAL(18,4)   GENERATED ALWAYS AS (
        COALESCE(actual_emissions_s1, 0) + COALESCE(actual_emissions_s2, 0) + COALESCE(actual_emissions_s3, 0)
    ) STORED,
    -- Target comparison
    target_emissions        DECIMAL(18,4),
    variance_pct            DECIMAL(8,3),
    yoy_change_pct          DECIMAL(8,3),
    cumulative_reduction_pct DECIMAL(8,3),
    -- On-track assessment
    on_track_status         VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    on_track_score          DECIMAL(6,2),
    trajectory_analysis     JSONB           DEFAULT '{}',
    -- Verification
    verification_status     VARCHAR(30)     DEFAULT 'unverified',
    verification_date       DATE,
    verifier_name           VARCHAR(255),
    verifier_accreditation  VARCHAR(255),
    assurance_level         VARCHAR(30),
    verification_statement_url TEXT,
    -- Reporting
    report_url              TEXT,
    submission_channels     TEXT[]          DEFAULT '{}',
    report_status           VARCHAR(30)     DEFAULT 'draft',
    warnings                TEXT[]          DEFAULT '{}',
    errors                  TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_ar_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p025_ar_on_track CHECK (
        on_track_status IN ('ON_TRACK', 'SLIGHTLY_OFF', 'SIGNIFICANTLY_OFF', 'REVERSED', 'PENDING')
    ),
    CONSTRAINT chk_p025_ar_verification CHECK (
        verification_status IN ('unverified', 'limited', 'reasonable', 'failed')
    ),
    CONSTRAINT chk_p025_ar_assurance CHECK (
        assurance_level IS NULL OR assurance_level IN ('LIMITED', 'REASONABLE', 'NONE')
    ),
    CONSTRAINT chk_p025_ar_report_status CHECK (
        report_status IN ('draft', 'submitted', 'verified', 'published', 'archived')
    ),
    CONSTRAINT chk_p025_ar_emissions_non_neg CHECK (
        (actual_emissions_s1 IS NULL OR actual_emissions_s1 >= 0) AND
        (actual_emissions_s2 IS NULL OR actual_emissions_s2 >= 0) AND
        (actual_emissions_s3 IS NULL OR actual_emissions_s3 >= 0)
    ),
    CONSTRAINT chk_p025_ar_target_non_neg CHECK (
        target_emissions IS NULL OR target_emissions >= 0
    ),
    CONSTRAINT uq_p025_ar_org_year UNIQUE (org_id, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes for annual_reports
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_ar_org             ON pack025_race_to_zero.annual_reports(org_id);
CREATE INDEX idx_p025_ar_pledge          ON pack025_race_to_zero.annual_reports(pledge_id);
CREATE INDEX idx_p025_ar_tenant          ON pack025_race_to_zero.annual_reports(tenant_id);
CREATE INDEX idx_p025_ar_year            ON pack025_race_to_zero.annual_reports(reporting_year);
CREATE INDEX idx_p025_ar_org_year        ON pack025_race_to_zero.annual_reports(org_id, reporting_year);
CREATE INDEX idx_p025_ar_on_track        ON pack025_race_to_zero.annual_reports(on_track_status);
CREATE INDEX idx_p025_ar_verification    ON pack025_race_to_zero.annual_reports(verification_status);
CREATE INDEX idx_p025_ar_status          ON pack025_race_to_zero.annual_reports(report_status);
CREATE INDEX idx_p025_ar_verifier        ON pack025_race_to_zero.annual_reports(verifier_name);
CREATE INDEX idx_p025_ar_created         ON pack025_race_to_zero.annual_reports(created_at DESC);
CREATE INDEX idx_p025_ar_trajectory      ON pack025_race_to_zero.annual_reports USING GIN(trajectory_analysis);
CREATE INDEX idx_p025_ar_metadata        ON pack025_race_to_zero.annual_reports USING GIN(metadata);

-- =============================================================================
-- Table 2: pack025_race_to_zero.progress_tracking
-- =============================================================================
-- Year-level progress tracking with reduction gap analysis, credibility
-- scoring, and automated recommendations.

CREATE TABLE pack025_race_to_zero.progress_tracking (
    tracking_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    report_id               UUID            REFERENCES pack025_race_to_zero.annual_reports(report_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    year                    INTEGER         NOT NULL,
    -- Reduction analysis
    reduction_achieved_pct  DECIMAL(8,3)    NOT NULL DEFAULT 0,
    reduction_required_pct  DECIMAL(8,3)    NOT NULL DEFAULT 0,
    gap_tco2e               DECIMAL(18,4),
    gap_pct                 DECIMAL(8,3),
    -- Scoring
    credibility_score       DECIMAL(6,2),
    trajectory_score        DECIMAL(6,2),
    action_implementation_score DECIMAL(6,2),
    data_quality_score      DECIMAL(6,2),
    -- Recommendations
    recommendations         TEXT,
    priority_actions        JSONB           DEFAULT '[]',
    risk_factors            JSONB           DEFAULT '[]',
    improvement_areas       TEXT[]          DEFAULT '{}',
    -- Status
    overall_assessment      VARCHAR(30)     DEFAULT 'PENDING',
    review_date             DATE,
    reviewer_id             UUID,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_pt_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p025_pt_assessment CHECK (
        overall_assessment IN ('ON_TRACK', 'NEEDS_ACCELERATION', 'AT_RISK', 'OFF_TRACK', 'PENDING')
    ),
    CONSTRAINT chk_p025_pt_credibility CHECK (
        credibility_score IS NULL OR (credibility_score >= 0 AND credibility_score <= 100)
    ),
    CONSTRAINT chk_p025_pt_trajectory CHECK (
        trajectory_score IS NULL OR (trajectory_score >= 0 AND trajectory_score <= 100)
    ),
    CONSTRAINT chk_p025_pt_action_impl CHECK (
        action_implementation_score IS NULL OR (action_implementation_score >= 0 AND action_implementation_score <= 100)
    ),
    CONSTRAINT chk_p025_pt_data_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT uq_p025_pt_org_year UNIQUE (org_id, year)
);

-- ---------------------------------------------------------------------------
-- Indexes for progress_tracking
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_pt_org             ON pack025_race_to_zero.progress_tracking(org_id);
CREATE INDEX idx_p025_pt_report          ON pack025_race_to_zero.progress_tracking(report_id);
CREATE INDEX idx_p025_pt_tenant          ON pack025_race_to_zero.progress_tracking(tenant_id);
CREATE INDEX idx_p025_pt_year            ON pack025_race_to_zero.progress_tracking(year);
CREATE INDEX idx_p025_pt_org_year        ON pack025_race_to_zero.progress_tracking(org_id, year);
CREATE INDEX idx_p025_pt_assessment      ON pack025_race_to_zero.progress_tracking(overall_assessment);
CREATE INDEX idx_p025_pt_credibility     ON pack025_race_to_zero.progress_tracking(credibility_score);
CREATE INDEX idx_p025_pt_created         ON pack025_race_to_zero.progress_tracking(created_at DESC);
CREATE INDEX idx_p025_pt_priority        ON pack025_race_to_zero.progress_tracking USING GIN(priority_actions);
CREATE INDEX idx_p025_pt_risks           ON pack025_race_to_zero.progress_tracking USING GIN(risk_factors);
CREATE INDEX idx_p025_pt_metadata        ON pack025_race_to_zero.progress_tracking USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_annual_reports_updated
    BEFORE UPDATE ON pack025_race_to_zero.annual_reports
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

CREATE TRIGGER trg_p025_progress_tracking_updated
    BEFORE UPDATE ON pack025_race_to_zero.progress_tracking
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.annual_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.progress_tracking ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_ar_tenant_isolation
    ON pack025_race_to_zero.annual_reports
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_ar_service_bypass
    ON pack025_race_to_zero.annual_reports
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_pt_tenant_isolation
    ON pack025_race_to_zero.progress_tracking
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_pt_service_bypass
    ON pack025_race_to_zero.progress_tracking
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.annual_reports TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.progress_tracking TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.annual_reports IS
    'Annual emission reports with scope-level actuals, target variance, verification status, and third-party assurance tracking.';
COMMENT ON TABLE pack025_race_to_zero.progress_tracking IS
    'Year-level progress tracking with reduction gap analysis, credibility scoring, and automated improvement recommendations.';

COMMENT ON COLUMN pack025_race_to_zero.annual_reports.report_id IS 'Unique annual report identifier.';
COMMENT ON COLUMN pack025_race_to_zero.annual_reports.variance_pct IS 'Percentage variance from target emissions.';
COMMENT ON COLUMN pack025_race_to_zero.annual_reports.on_track_status IS 'Overall on-track assessment: ON_TRACK, SLIGHTLY_OFF, SIGNIFICANTLY_OFF, REVERSED, PENDING.';
COMMENT ON COLUMN pack025_race_to_zero.annual_reports.verifier_name IS 'Name of third-party verification body.';
COMMENT ON COLUMN pack025_race_to_zero.progress_tracking.tracking_id IS 'Unique progress tracking identifier.';
COMMENT ON COLUMN pack025_race_to_zero.progress_tracking.gap_tco2e IS 'Gap between required and achieved reduction in tCO2e.';
COMMENT ON COLUMN pack025_race_to_zero.progress_tracking.credibility_score IS 'Overall credibility score (0-100).';
COMMENT ON COLUMN pack025_race_to_zero.progress_tracking.recommendations IS 'Text-based improvement recommendations.';

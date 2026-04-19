-- =============================================================================
-- V205: PACK-029 Interim Targets Pack - Reporting Periods
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    010 of 015
-- Date:         March 2026
--
-- Reporting period management with fiscal/calendar year alignment, framework
-- submission tracking (SBTi/CDP/TCFD), assurance levels, and deadline
-- management for multi-framework interim target reporting.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_reporting_periods
--
-- Previous: V204__PACK029_carbon_budget_allocation.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_reporting_periods
-- =============================================================================
-- Reporting period definitions with fiscal/calendar year alignment, framework
-- submission tracking, assurance provider management, and deadline monitoring
-- for SBTi, CDP, TCFD, and other framework reporting requirements.

CREATE TABLE pack029_interim_targets.gl_reporting_periods (
    period_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    -- Reporting period
    reporting_year              INTEGER         NOT NULL,
    reporting_period_type       VARCHAR(20)     DEFAULT 'ANNUAL',
    period_start_date           DATE,
    period_end_date             DATE,
    -- Fiscal year alignment
    fiscal_year_start_month     INTEGER         DEFAULT 1,
    fiscal_year_end_month       INTEGER         DEFAULT 12,
    calendar_year_aligned       BOOLEAN         DEFAULT TRUE,
    fiscal_year_label           VARCHAR(20),
    -- Deadlines
    reporting_deadline          DATE,
    internal_deadline           DATE,
    data_collection_deadline    DATE,
    review_deadline             DATE,
    -- Submission tracking
    submission_date             DATE,
    submission_status           VARCHAR(20)     DEFAULT 'NOT_SUBMITTED',
    submitted_by                VARCHAR(255),
    -- Framework alignment
    frameworks                  TEXT[]          DEFAULT '{}',
    primary_framework           VARCHAR(20),
    sbti_reporting              BOOLEAN         DEFAULT FALSE,
    cdp_reporting               BOOLEAN         DEFAULT FALSE,
    tcfd_reporting              BOOLEAN         DEFAULT FALSE,
    csrd_reporting              BOOLEAN         DEFAULT FALSE,
    iso14064_reporting          BOOLEAN         DEFAULT FALSE,
    -- Assurance
    assurance_level             VARCHAR(20)     DEFAULT 'NONE',
    assurance_provider          VARCHAR(200),
    assurance_standard          VARCHAR(50),
    assurance_scope             TEXT,
    assurance_opinion           VARCHAR(30),
    assurance_date              DATE,
    assurance_report_url        VARCHAR(500),
    -- Data completeness
    data_completeness_pct       DECIMAL(5,2)    DEFAULT 0,
    scope1_complete             BOOLEAN         DEFAULT FALSE,
    scope2_complete             BOOLEAN         DEFAULT FALSE,
    scope3_complete             BOOLEAN         DEFAULT FALSE,
    categories_reported         INTEGER         DEFAULT 0,
    -- Quality review
    internal_review_completed   BOOLEAN         DEFAULT FALSE,
    internal_reviewer           VARCHAR(255),
    internal_review_date        DATE,
    quality_score               DECIMAL(5,2),
    -- Publication
    published                   BOOLEAN         DEFAULT FALSE,
    published_date              DATE,
    publication_url             VARCHAR(500),
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    is_final                    BOOLEAN         DEFAULT FALSE,
    is_restated                 BOOLEAN         DEFAULT FALSE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_rp_reporting_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p029_rp_period_type CHECK (
        reporting_period_type IN ('ANNUAL', 'SEMI_ANNUAL', 'QUARTERLY', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_rp_fiscal_months CHECK (
        fiscal_year_start_month >= 1 AND fiscal_year_start_month <= 12
        AND fiscal_year_end_month >= 1 AND fiscal_year_end_month <= 12
    ),
    CONSTRAINT chk_p029_rp_submission_status CHECK (
        submission_status IN ('NOT_SUBMITTED', 'IN_PREPARATION', 'UNDER_REVIEW',
                              'SUBMITTED', 'ACCEPTED', 'REJECTED', 'RESUBMISSION_REQUIRED')
    ),
    CONSTRAINT chk_p029_rp_primary_framework CHECK (
        primary_framework IS NULL OR primary_framework IN (
            'SBTI', 'CDP', 'TCFD', 'CSRD', 'ISO14064', 'GRI', 'SASB', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p029_rp_assurance_level CHECK (
        assurance_level IN ('NONE', 'LIMITED', 'REASONABLE')
    ),
    CONSTRAINT chk_p029_rp_assurance_opinion CHECK (
        assurance_opinion IS NULL OR assurance_opinion IN (
            'UNMODIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER', 'PENDING'
        )
    ),
    CONSTRAINT chk_p029_rp_assurance_standard CHECK (
        assurance_standard IS NULL OR assurance_standard IN (
            'ISAE3000', 'ISAE3410', 'ISO14064_3', 'AA1000AS', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p029_rp_completeness CHECK (
        data_completeness_pct >= 0 AND data_completeness_pct <= 100
    ),
    CONSTRAINT chk_p029_rp_quality_score CHECK (
        quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 100)
    ),
    CONSTRAINT chk_p029_rp_categories CHECK (
        categories_reported >= 0 AND categories_reported <= 15
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_rp_tenant             ON pack029_interim_targets.gl_reporting_periods(tenant_id);
CREATE INDEX idx_p029_rp_org                ON pack029_interim_targets.gl_reporting_periods(organization_id);
CREATE INDEX idx_p029_rp_org_year           ON pack029_interim_targets.gl_reporting_periods(organization_id, reporting_year);
CREATE INDEX idx_p029_rp_submission_date    ON pack029_interim_targets.gl_reporting_periods(submission_date DESC);
CREATE INDEX idx_p029_rp_submission_status  ON pack029_interim_targets.gl_reporting_periods(submission_status);
CREATE INDEX idx_p029_rp_not_submitted      ON pack029_interim_targets.gl_reporting_periods(organization_id, reporting_deadline) WHERE submission_status = 'NOT_SUBMITTED';
CREATE INDEX idx_p029_rp_deadline           ON pack029_interim_targets.gl_reporting_periods(reporting_deadline) WHERE submission_status NOT IN ('SUBMITTED', 'ACCEPTED');
CREATE INDEX idx_p029_rp_assurance          ON pack029_interim_targets.gl_reporting_periods(assurance_level);
CREATE INDEX idx_p029_rp_unassured          ON pack029_interim_targets.gl_reporting_periods(organization_id, reporting_year) WHERE assurance_level = 'NONE';
CREATE INDEX idx_p029_rp_sbti               ON pack029_interim_targets.gl_reporting_periods(organization_id) WHERE sbti_reporting = TRUE;
CREATE INDEX idx_p029_rp_cdp                ON pack029_interim_targets.gl_reporting_periods(organization_id) WHERE cdp_reporting = TRUE;
CREATE INDEX idx_p029_rp_active             ON pack029_interim_targets.gl_reporting_periods(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_rp_final              ON pack029_interim_targets.gl_reporting_periods(organization_id, reporting_year) WHERE is_final = TRUE;
CREATE INDEX idx_p029_rp_created            ON pack029_interim_targets.gl_reporting_periods(created_at DESC);
CREATE INDEX idx_p029_rp_frameworks         ON pack029_interim_targets.gl_reporting_periods USING GIN(frameworks);
CREATE INDEX idx_p029_rp_metadata           ON pack029_interim_targets.gl_reporting_periods USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_reporting_periods_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_reporting_periods
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_reporting_periods ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_rp_tenant_isolation
    ON pack029_interim_targets.gl_reporting_periods
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_rp_service_bypass
    ON pack029_interim_targets.gl_reporting_periods
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_reporting_periods TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_reporting_periods IS
    'Reporting period management with fiscal/calendar year alignment, multi-framework submission tracking (SBTi/CDP/TCFD/CSRD), assurance levels, data completeness, and deadline management for interim target reporting.';

COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.period_id IS 'Unique reporting period identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.organization_id IS 'Reference to the reporting organization.';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.reporting_year IS 'The reporting year this period covers.';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.fiscal_year_start_month IS 'Month when the fiscal year starts (1-12).';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.calendar_year_aligned IS 'Whether the fiscal year aligns with calendar year (Jan-Dec).';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.reporting_deadline IS 'External reporting deadline date.';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.submission_date IS 'Actual submission date.';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.frameworks IS 'Array of framework identifiers this report covers.';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.assurance_level IS 'Assurance level: NONE, LIMITED, REASONABLE.';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.assurance_provider IS 'Name of the third-party assurance provider.';
COMMENT ON COLUMN pack029_interim_targets.gl_reporting_periods.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

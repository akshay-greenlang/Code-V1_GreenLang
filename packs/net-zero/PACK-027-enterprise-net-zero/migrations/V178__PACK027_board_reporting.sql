-- =============================================================================
-- V178: PACK-027 Enterprise Net Zero - Board Reporting
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    013 of 015
-- Date:         March 2026
--
-- Board-level climate reporting with quarterly emissions summaries, target
-- progress tracking, risk highlights, recommended actions, and governance
-- decision tracking. Designed for non-specialist board consumption.
--
-- Tables (1):
--   1. pack027_enterprise_net_zero.gl_board_reports
--
-- Previous: V177__PACK027_assurance_records.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_board_reports
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_board_reports (
    report_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Reporting period
    reporting_quarter           VARCHAR(10)     NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    report_date                 DATE            NOT NULL DEFAULT CURRENT_DATE,
    -- Emissions summary
    emissions_summary           JSONB           NOT NULL DEFAULT '{}',
    total_tco2e_ytd             DECIMAL(18,4),
    scope1_tco2e_ytd            DECIMAL(18,4),
    scope2_tco2e_ytd            DECIMAL(18,4),
    scope3_tco2e_ytd            DECIMAL(18,4),
    yoy_change_pct              DECIMAL(8,2),
    vs_baseline_change_pct      DECIMAL(8,2),
    -- Target progress
    target_progress             JSONB           DEFAULT '{}',
    near_term_on_track          VARCHAR(20),
    long_term_on_track          VARCHAR(20),
    sbti_status_summary         VARCHAR(100),
    -- Carbon financial impact
    carbon_cost_ytd_usd         DECIMAL(18,2),
    carbon_cost_vs_budget_pct   DECIMAL(8,2),
    cbam_exposure_ytd_usd       DECIMAL(18,2),
    -- Risk summary
    risk_summary                JSONB           DEFAULT '{}',
    critical_risks_count        INTEGER         DEFAULT 0,
    high_risks_count            INTEGER         DEFAULT 0,
    new_risks_this_quarter      INTEGER         DEFAULT 0,
    mitigated_risks_this_quarter INTEGER        DEFAULT 0,
    -- Supply chain
    supplier_engagement_pct     DECIMAL(6,2),
    suppliers_with_sbti_pct     DECIMAL(6,2),
    scope3_data_quality_score   DECIMAL(3,1),
    -- Regulatory status
    regulatory_status           JSONB           DEFAULT '{}',
    upcoming_deadlines          JSONB           DEFAULT '{}',
    compliance_issues_count     INTEGER         DEFAULT 0,
    -- Assurance status
    assurance_status_summary    VARCHAR(100),
    -- Board actions
    board_actions               JSONB           DEFAULT '{}',
    decisions_required          JSONB           DEFAULT '{}',
    previous_actions_status     JSONB           DEFAULT '{}',
    -- Key achievements
    key_achievements            TEXT[]          DEFAULT '{}',
    key_challenges              TEXT[]          DEFAULT '{}',
    -- Executive commentary
    cso_commentary              TEXT,
    cfo_commentary              TEXT,
    -- Generation metadata
    generated_at                TIMESTAMPTZ,
    generation_time_seconds     DECIMAL(10,2),
    data_freshness_hours        DECIMAL(8,1),
    -- Approval
    status                      VARCHAR(30)     DEFAULT 'DRAFT',
    approved_by                 VARCHAR(255),
    approved_date               DATE,
    presented_date              DATE,
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_br_quarter CHECK (
        reporting_quarter IN ('Q1', 'Q2', 'Q3', 'Q4', 'H1', 'H2', 'FY')
    ),
    CONSTRAINT chk_p027_br_reporting_year CHECK (
        reporting_year >= 2020 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p027_br_on_track CHECK (
        (near_term_on_track IS NULL OR near_term_on_track IN ('ON_TRACK', 'AT_RISK', 'OFF_TRACK', 'ACHIEVED', 'NOT_SET')) AND
        (long_term_on_track IS NULL OR long_term_on_track IN ('ON_TRACK', 'AT_RISK', 'OFF_TRACK', 'ACHIEVED', 'NOT_SET'))
    ),
    CONSTRAINT chk_p027_br_status CHECK (
        status IN ('DRAFT', 'REVIEW', 'APPROVED', 'PRESENTED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p027_br_company_quarter_year UNIQUE (company_id, reporting_quarter, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_br_company            ON pack027_enterprise_net_zero.gl_board_reports(company_id);
CREATE INDEX idx_p027_br_tenant             ON pack027_enterprise_net_zero.gl_board_reports(tenant_id);
CREATE INDEX idx_p027_br_quarter            ON pack027_enterprise_net_zero.gl_board_reports(reporting_quarter);
CREATE INDEX idx_p027_br_year               ON pack027_enterprise_net_zero.gl_board_reports(reporting_year);
CREATE INDEX idx_p027_br_year_quarter       ON pack027_enterprise_net_zero.gl_board_reports(reporting_year, reporting_quarter);
CREATE INDEX idx_p027_br_report_date        ON pack027_enterprise_net_zero.gl_board_reports(report_date DESC);
CREATE INDEX idx_p027_br_status             ON pack027_enterprise_net_zero.gl_board_reports(status);
CREATE INDEX idx_p027_br_near_term          ON pack027_enterprise_net_zero.gl_board_reports(near_term_on_track);
CREATE INDEX idx_p027_br_critical_risks     ON pack027_enterprise_net_zero.gl_board_reports(critical_risks_count) WHERE critical_risks_count > 0;
CREATE INDEX idx_p027_br_compliance         ON pack027_enterprise_net_zero.gl_board_reports(compliance_issues_count) WHERE compliance_issues_count > 0;
CREATE INDEX idx_p027_br_created            ON pack027_enterprise_net_zero.gl_board_reports(created_at DESC);
CREATE INDEX idx_p027_br_emissions          ON pack027_enterprise_net_zero.gl_board_reports USING GIN(emissions_summary);
CREATE INDEX idx_p027_br_target             ON pack027_enterprise_net_zero.gl_board_reports USING GIN(target_progress);
CREATE INDEX idx_p027_br_risk               ON pack027_enterprise_net_zero.gl_board_reports USING GIN(risk_summary);
CREATE INDEX idx_p027_br_actions            ON pack027_enterprise_net_zero.gl_board_reports USING GIN(board_actions);
CREATE INDEX idx_p027_br_metadata           ON pack027_enterprise_net_zero.gl_board_reports USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_board_reports_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_board_reports
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_board_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_br_tenant_isolation
    ON pack027_enterprise_net_zero.gl_board_reports
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_br_service_bypass
    ON pack027_enterprise_net_zero.gl_board_reports
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_board_reports TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_board_reports IS
    'Board-level quarterly climate reports with emissions summaries, target progress, risk highlights, regulatory status, and governance decision tracking.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_board_reports.report_id IS 'Unique board report identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_board_reports.reporting_quarter IS 'Reporting period: Q1, Q2, Q3, Q4, H1, H2, or FY.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_board_reports.emissions_summary IS 'JSONB consolidated emissions summary for the period.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_board_reports.target_progress IS 'JSONB target progress tracking (near-term, long-term, net-zero).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_board_reports.risk_summary IS 'JSONB climate risk summary for board consumption.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_board_reports.board_actions IS 'JSONB record of board actions and decisions on climate matters.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_board_reports.near_term_on_track IS 'Near-term target tracking: ON_TRACK, AT_RISK, OFF_TRACK, ACHIEVED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_board_reports.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

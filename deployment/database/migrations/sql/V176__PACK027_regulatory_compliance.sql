-- =============================================================================
-- V176: PACK-027 Enterprise Net Zero - Regulatory Compliance
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    011 of 015
-- Date:         March 2026
--
-- Multi-framework regulatory filing management (SEC, CSRD, CDP, TCFD, SB253,
-- ISO 14064, ISSB S2) with submission tracking, compliance gap analysis, and
-- remediation planning. Supports 8+ simultaneous regulatory frameworks.
--
-- Tables (2):
--   1. pack027_enterprise_net_zero.gl_regulatory_filings
--   2. pack027_enterprise_net_zero.gl_compliance_gaps
--
-- Previous: V175__PACK027_risk_assessments.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_regulatory_filings
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_regulatory_filings (
    filing_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Framework
    framework                   VARCHAR(30)     NOT NULL,
    framework_version           VARCHAR(30),
    sub_framework               VARCHAR(50),
    -- Filing period
    filing_year                 INTEGER         NOT NULL,
    reporting_period_start      DATE,
    reporting_period_end        DATE,
    -- Filing details
    filing_type                 VARCHAR(30)     DEFAULT 'ANNUAL',
    filing_reference            VARCHAR(255),
    file_path                   TEXT,
    file_hash                   VARCHAR(64),
    -- Status and dates
    status                      VARCHAR(30)     NOT NULL DEFAULT 'NOT_STARTED',
    preparation_start_date      DATE,
    internal_review_date        DATE,
    submission_date             DATE,
    acknowledgement_date        DATE,
    publication_date            DATE,
    deadline                    DATE,
    -- Assurance
    assurance_required          BOOLEAN         DEFAULT FALSE,
    assurance_level             VARCHAR(30),
    assurance_provider          VARCHAR(255),
    assurance_status            VARCHAR(30),
    assurance_opinion           VARCHAR(30),
    -- Content coverage
    scope1_included             BOOLEAN         DEFAULT TRUE,
    scope2_included             BOOLEAN         DEFAULT TRUE,
    scope3_included             BOOLEAN         DEFAULT FALSE,
    scope3_categories_included  TEXT[]          DEFAULT '{}',
    transition_plan_included    BOOLEAN         DEFAULT FALSE,
    risk_assessment_included    BOOLEAN         DEFAULT FALSE,
    targets_included            BOOLEAN         DEFAULT FALSE,
    -- Quality
    completeness_score          DECIMAL(5,2),
    accuracy_score              DECIMAL(5,2),
    consistency_score           DECIMAL(5,2),
    overall_quality_score       DECIMAL(5,2),
    -- Review
    reviewer_name               VARCHAR(255),
    reviewer_comments           TEXT,
    revision_count              INTEGER         DEFAULT 0,
    -- Regulatory response
    regulator_feedback          TEXT,
    deficiency_noted            BOOLEAN         DEFAULT FALSE,
    remediation_required        BOOLEAN         DEFAULT FALSE,
    remediation_deadline        DATE,
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_rf_framework CHECK (
        framework IN ('SEC_CLIMATE', 'CSRD_ESRS_E1', 'CDP_CLIMATE', 'TCFD', 'ISSB_S2',
                       'SB253', 'SB261', 'ISO14064', 'GHG_PROTOCOL', 'SBTI',
                       'EU_TAXONOMY', 'SFDR', 'OTHER')
    ),
    CONSTRAINT chk_p027_rf_filing_year CHECK (
        filing_year >= 2020 AND filing_year <= 2100
    ),
    CONSTRAINT chk_p027_rf_filing_type CHECK (
        filing_type IN ('ANNUAL', 'INTERIM', 'QUARTERLY', 'BIENNIAL', 'AD_HOC', 'RESTATEMENT')
    ),
    CONSTRAINT chk_p027_rf_status CHECK (
        status IN ('NOT_STARTED', 'DATA_COLLECTION', 'CALCULATION', 'REVIEW',
                    'ASSURANCE', 'FINAL_REVIEW', 'SUBMITTED', 'ACCEPTED',
                    'PUBLISHED', 'REJECTED', 'REVISION_REQUIRED', 'WITHDRAWN')
    ),
    CONSTRAINT chk_p027_rf_assurance_level CHECK (
        assurance_level IS NULL OR assurance_level IN ('LIMITED', 'REASONABLE', 'NONE')
    ),
    CONSTRAINT chk_p027_rf_assurance_status CHECK (
        assurance_status IS NULL OR assurance_status IN (
            'NOT_STARTED', 'PLANNING', 'FIELDWORK', 'REPORTING', 'COMPLETED'
        )
    ),
    CONSTRAINT chk_p027_rf_assurance_opinion CHECK (
        assurance_opinion IS NULL OR assurance_opinion IN (
            'UNMODIFIED', 'MODIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER'
        )
    ),
    CONSTRAINT chk_p027_rf_quality_scores CHECK (
        (completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 100)) AND
        (accuracy_score IS NULL OR (accuracy_score >= 0 AND accuracy_score <= 100)) AND
        (consistency_score IS NULL OR (consistency_score >= 0 AND consistency_score <= 100)) AND
        (overall_quality_score IS NULL OR (overall_quality_score >= 0 AND overall_quality_score <= 100))
    ),
    CONSTRAINT uq_p027_rf_company_framework_year UNIQUE (company_id, framework, filing_year, filing_type)
);

-- =============================================================================
-- Table 2: pack027_enterprise_net_zero.gl_compliance_gaps
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_compliance_gaps (
    gap_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    filing_id                   UUID            REFERENCES pack027_enterprise_net_zero.gl_regulatory_filings(filing_id) ON DELETE SET NULL,
    -- Framework and requirement
    framework                   VARCHAR(30)     NOT NULL,
    requirement                 VARCHAR(255)    NOT NULL,
    requirement_reference       VARCHAR(100),
    requirement_category        VARCHAR(50),
    -- Gap assessment
    gap_description             TEXT            NOT NULL,
    gap_severity                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    gap_type                    VARCHAR(30)     NOT NULL,
    -- Current state
    current_state               TEXT,
    required_state              TEXT,
    -- Remediation
    remediation_plan            TEXT,
    remediation_owner           VARCHAR(255),
    remediation_priority        VARCHAR(20)     DEFAULT 'MEDIUM',
    remediation_effort          VARCHAR(30),
    estimated_cost_usd          DECIMAL(18,2),
    estimated_hours             DECIMAL(8,1),
    -- Timeline
    identified_date             DATE            NOT NULL DEFAULT CURRENT_DATE,
    target_resolution_date      DATE,
    actual_resolution_date      DATE,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'OPEN',
    resolution_evidence         TEXT,
    verified_resolved           BOOLEAN         DEFAULT FALSE,
    verified_by                 VARCHAR(255),
    -- Risk
    non_compliance_risk         VARCHAR(30),
    potential_penalty_usd       DECIMAL(18,2),
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_cg_framework CHECK (
        framework IN ('SEC_CLIMATE', 'CSRD_ESRS_E1', 'CDP_CLIMATE', 'TCFD', 'ISSB_S2',
                       'SB253', 'SB261', 'ISO14064', 'GHG_PROTOCOL', 'SBTI',
                       'EU_TAXONOMY', 'SFDR', 'OTHER')
    ),
    CONSTRAINT chk_p027_cg_severity CHECK (
        gap_severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p027_cg_type CHECK (
        gap_type IN ('DATA_AVAILABILITY', 'METHODOLOGY', 'PROCESS', 'TECHNOLOGY',
                      'GOVERNANCE', 'DOCUMENTATION', 'ASSURANCE', 'DISCLOSURE',
                      'COVERAGE', 'TIMELINESS', 'QUALITY')
    ),
    CONSTRAINT chk_p027_cg_priority CHECK (
        remediation_priority IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p027_cg_effort CHECK (
        remediation_effort IS NULL OR remediation_effort IN (
            'TRIVIAL', 'SMALL', 'MEDIUM', 'LARGE', 'EXTRA_LARGE'
        )
    ),
    CONSTRAINT chk_p027_cg_status CHECK (
        status IN ('OPEN', 'IN_PROGRESS', 'RESOLVED', 'ACCEPTED_RISK', 'DEFERRED', 'CLOSED')
    ),
    CONSTRAINT chk_p027_cg_non_compliance CHECK (
        non_compliance_risk IS NULL OR non_compliance_risk IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p027_cg_dates CHECK (
        target_resolution_date IS NULL OR target_resolution_date >= identified_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for gl_regulatory_filings
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_rf_company            ON pack027_enterprise_net_zero.gl_regulatory_filings(company_id);
CREATE INDEX idx_p027_rf_tenant             ON pack027_enterprise_net_zero.gl_regulatory_filings(tenant_id);
CREATE INDEX idx_p027_rf_framework          ON pack027_enterprise_net_zero.gl_regulatory_filings(framework);
CREATE INDEX idx_p027_rf_year               ON pack027_enterprise_net_zero.gl_regulatory_filings(filing_year);
CREATE INDEX idx_p027_rf_status             ON pack027_enterprise_net_zero.gl_regulatory_filings(status);
CREATE INDEX idx_p027_rf_deadline           ON pack027_enterprise_net_zero.gl_regulatory_filings(deadline);
CREATE INDEX idx_p027_rf_submission         ON pack027_enterprise_net_zero.gl_regulatory_filings(submission_date);
CREATE INDEX idx_p027_rf_assurance          ON pack027_enterprise_net_zero.gl_regulatory_filings(assurance_required) WHERE assurance_required = TRUE;
CREATE INDEX idx_p027_rf_assurance_status   ON pack027_enterprise_net_zero.gl_regulatory_filings(assurance_status);
CREATE INDEX idx_p027_rf_deficiency         ON pack027_enterprise_net_zero.gl_regulatory_filings(deficiency_noted) WHERE deficiency_noted = TRUE;
CREATE INDEX idx_p027_rf_quality            ON pack027_enterprise_net_zero.gl_regulatory_filings(overall_quality_score);
CREATE INDEX idx_p027_rf_company_framework  ON pack027_enterprise_net_zero.gl_regulatory_filings(company_id, framework);
CREATE INDEX idx_p027_rf_created            ON pack027_enterprise_net_zero.gl_regulatory_filings(created_at DESC);
CREATE INDEX idx_p027_rf_metadata           ON pack027_enterprise_net_zero.gl_regulatory_filings USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Indexes for gl_compliance_gaps
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_cg_company            ON pack027_enterprise_net_zero.gl_compliance_gaps(company_id);
CREATE INDEX idx_p027_cg_tenant             ON pack027_enterprise_net_zero.gl_compliance_gaps(tenant_id);
CREATE INDEX idx_p027_cg_filing             ON pack027_enterprise_net_zero.gl_compliance_gaps(filing_id);
CREATE INDEX idx_p027_cg_framework          ON pack027_enterprise_net_zero.gl_compliance_gaps(framework);
CREATE INDEX idx_p027_cg_severity           ON pack027_enterprise_net_zero.gl_compliance_gaps(gap_severity);
CREATE INDEX idx_p027_cg_type               ON pack027_enterprise_net_zero.gl_compliance_gaps(gap_type);
CREATE INDEX idx_p027_cg_status             ON pack027_enterprise_net_zero.gl_compliance_gaps(status);
CREATE INDEX idx_p027_cg_priority           ON pack027_enterprise_net_zero.gl_compliance_gaps(remediation_priority);
CREATE INDEX idx_p027_cg_owner              ON pack027_enterprise_net_zero.gl_compliance_gaps(remediation_owner);
CREATE INDEX idx_p027_cg_target_date        ON pack027_enterprise_net_zero.gl_compliance_gaps(target_resolution_date);
CREATE INDEX idx_p027_cg_risk               ON pack027_enterprise_net_zero.gl_compliance_gaps(non_compliance_risk);
CREATE INDEX idx_p027_cg_open               ON pack027_enterprise_net_zero.gl_compliance_gaps(status) WHERE status IN ('OPEN', 'IN_PROGRESS');
CREATE INDEX idx_p027_cg_created            ON pack027_enterprise_net_zero.gl_compliance_gaps(created_at DESC);
CREATE INDEX idx_p027_cg_metadata           ON pack027_enterprise_net_zero.gl_compliance_gaps USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_regulatory_filings_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_regulatory_filings
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p027_compliance_gaps_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_compliance_gaps
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_regulatory_filings ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack027_enterprise_net_zero.gl_compliance_gaps ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_rf_tenant_isolation
    ON pack027_enterprise_net_zero.gl_regulatory_filings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_rf_service_bypass
    ON pack027_enterprise_net_zero.gl_regulatory_filings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p027_cg_tenant_isolation
    ON pack027_enterprise_net_zero.gl_compliance_gaps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_cg_service_bypass
    ON pack027_enterprise_net_zero.gl_compliance_gaps
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_regulatory_filings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_compliance_gaps TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_regulatory_filings IS
    'Multi-framework regulatory filing management (SEC, CSRD, CDP, TCFD, SB253, ISO 14064, ISSB S2) with submission tracking and assurance status.';
COMMENT ON TABLE pack027_enterprise_net_zero.gl_compliance_gaps IS
    'Compliance gap analysis with requirement-level gap identification, severity assessment, remediation planning, and resolution tracking.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_regulatory_filings.filing_id IS 'Unique regulatory filing identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_regulatory_filings.framework IS 'Regulatory framework: SEC_CLIMATE, CSRD_ESRS_E1, CDP_CLIMATE, TCFD, ISSB_S2, SB253, ISO14064, etc.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_regulatory_filings.status IS 'Filing pipeline status from NOT_STARTED through PUBLISHED or REJECTED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_regulatory_filings.file_path IS 'Storage path to the filed document.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_compliance_gaps.gap_id IS 'Unique compliance gap identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_compliance_gaps.requirement IS 'Specific regulatory requirement where gap exists.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_compliance_gaps.gap_description IS 'Description of the compliance gap.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_compliance_gaps.remediation_plan IS 'Planned actions to close the compliance gap.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_compliance_gaps.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

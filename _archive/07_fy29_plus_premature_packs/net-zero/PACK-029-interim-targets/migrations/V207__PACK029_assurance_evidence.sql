-- =============================================================================
-- V207: PACK-029 Interim Targets Pack - Assurance Evidence
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    012 of 015
-- Date:         March 2026
--
-- Assurance evidence and workpaper tracking for ISO 14064-3 compliance with
-- evidence type classification, document management, completeness scoring,
-- and assurance provider review status.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_assurance_evidence
--
-- Previous: V206__PACK029_validation_results.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_assurance_evidence
-- =============================================================================
-- Assurance evidence records with evidence type classification, document
-- URL management, evidence tier (primary/secondary/tertiary), completeness
-- tracking, and assurance provider review status for ISO 14064-3 workpapers.

CREATE TABLE pack029_interim_targets.gl_assurance_evidence (
    evidence_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    period_id                   UUID            REFERENCES pack029_interim_targets.gl_reporting_periods(period_id) ON DELETE SET NULL,
    -- Reporting context
    reporting_year              INTEGER         NOT NULL,
    scope                       VARCHAR(20),
    category                    VARCHAR(60),
    -- Evidence classification
    evidence_type               VARCHAR(40)     NOT NULL,
    evidence_subtype            VARCHAR(60),
    evidence_title              VARCHAR(300)    NOT NULL,
    evidence_description        TEXT,
    -- Document management
    document_url                VARCHAR(500),
    document_format             VARCHAR(20),
    document_size_bytes         BIGINT,
    document_hash               VARCHAR(64),
    document_version            VARCHAR(20),
    -- Evidence tier
    evidence_tier               VARCHAR(20)     NOT NULL DEFAULT 'SECONDARY',
    -- Completeness
    completeness_pct            DECIMAL(5,2)    DEFAULT 0,
    completeness_gaps           JSONB           DEFAULT '[]',
    -- Data linkage
    data_source_id              UUID,
    data_source_type            VARCHAR(50),
    mrv_agent_id                VARCHAR(50),
    calculation_reference       VARCHAR(200),
    -- Assurance review
    assurance_provider_reviewed BOOLEAN         DEFAULT FALSE,
    review_date                 DATE,
    review_outcome              VARCHAR(20),
    reviewer_name               VARCHAR(255),
    reviewer_comments           TEXT,
    -- Quality assessment
    reliability_score           DECIMAL(5,2),
    relevance_score             DECIMAL(5,2),
    sufficiency_score           DECIMAL(5,2),
    overall_quality_score       DECIMAL(5,2),
    -- Retention
    retention_policy            VARCHAR(30)     DEFAULT 'STANDARD',
    retention_until             DATE,
    archived                    BOOLEAN         DEFAULT FALSE,
    archived_at                 TIMESTAMPTZ,
    -- Upload tracking
    uploaded_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    uploaded_by                 VARCHAR(255),
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_ae_evidence_type CHECK (
        evidence_type IN (
            'CALCULATION_SHEET', 'DATA_SOURCE', 'METHODOLOGY', 'VARIANCE_EXPLANATION',
            'EMISSION_FACTOR', 'ACTIVITY_DATA', 'SUPPLIER_DATA', 'METER_DATA',
            'INVOICE', 'AUDIT_REPORT', 'THIRD_PARTY_VERIFICATION', 'BOARD_APPROVAL',
            'TARGET_DOCUMENTATION', 'BOUNDARY_DEFINITION', 'RESTATEMENT_JUSTIFICATION',
            'METHODOLOGY_CHANGE', 'UNCERTAINTY_ANALYSIS', 'QUALITY_CONTROL',
            'INTERNAL_AUDIT', 'SUPPORTING_NARRATIVE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p029_ae_evidence_tier CHECK (
        evidence_tier IN ('PRIMARY', 'SECONDARY', 'TERTIARY')
    ),
    CONSTRAINT chk_p029_ae_scope CHECK (
        scope IS NULL OR scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_ae_reporting_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p029_ae_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p029_ae_review_outcome CHECK (
        review_outcome IS NULL OR review_outcome IN (
            'ACCEPTED', 'ACCEPTED_WITH_COMMENTS', 'REJECTED', 'FURTHER_INFO_REQUIRED', 'PENDING'
        )
    ),
    CONSTRAINT chk_p029_ae_quality_scores CHECK (
        (reliability_score IS NULL OR (reliability_score >= 0 AND reliability_score <= 100))
        AND (relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 100))
        AND (sufficiency_score IS NULL OR (sufficiency_score >= 0 AND sufficiency_score <= 100))
        AND (overall_quality_score IS NULL OR (overall_quality_score >= 0 AND overall_quality_score <= 100))
    ),
    CONSTRAINT chk_p029_ae_retention_policy CHECK (
        retention_policy IN ('STANDARD', 'EXTENDED', 'PERMANENT', 'REGULATORY', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_ae_document_format CHECK (
        document_format IS NULL OR document_format IN (
            'PDF', 'XLSX', 'CSV', 'DOCX', 'JSON', 'XML', 'PNG', 'JPG', 'ZIP', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_ae_tenant             ON pack029_interim_targets.gl_assurance_evidence(tenant_id);
CREATE INDEX idx_p029_ae_org                ON pack029_interim_targets.gl_assurance_evidence(organization_id);
CREATE INDEX idx_p029_ae_target             ON pack029_interim_targets.gl_assurance_evidence(target_id);
CREATE INDEX idx_p029_ae_period             ON pack029_interim_targets.gl_assurance_evidence(period_id);
CREATE INDEX idx_p029_ae_org_year           ON pack029_interim_targets.gl_assurance_evidence(organization_id, reporting_year);
CREATE INDEX idx_p029_ae_evidence_type      ON pack029_interim_targets.gl_assurance_evidence(evidence_type);
CREATE INDEX idx_p029_ae_evidence_tier      ON pack029_interim_targets.gl_assurance_evidence(evidence_tier);
CREATE INDEX idx_p029_ae_completeness       ON pack029_interim_targets.gl_assurance_evidence(completeness_pct);
CREATE INDEX idx_p029_ae_incomplete         ON pack029_interim_targets.gl_assurance_evidence(organization_id, reporting_year) WHERE completeness_pct < 100;
CREATE INDEX idx_p029_ae_unreviewed         ON pack029_interim_targets.gl_assurance_evidence(organization_id) WHERE assurance_provider_reviewed = FALSE;
CREATE INDEX idx_p029_ae_reviewed           ON pack029_interim_targets.gl_assurance_evidence(review_date DESC) WHERE assurance_provider_reviewed = TRUE;
CREATE INDEX idx_p029_ae_review_outcome     ON pack029_interim_targets.gl_assurance_evidence(review_outcome);
CREATE INDEX idx_p029_ae_rejected           ON pack029_interim_targets.gl_assurance_evidence(organization_id) WHERE review_outcome = 'REJECTED';
CREATE INDEX idx_p029_ae_mrv_agent          ON pack029_interim_targets.gl_assurance_evidence(mrv_agent_id) WHERE mrv_agent_id IS NOT NULL;
CREATE INDEX idx_p029_ae_active             ON pack029_interim_targets.gl_assurance_evidence(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_ae_uploaded           ON pack029_interim_targets.gl_assurance_evidence(uploaded_date DESC);
CREATE INDEX idx_p029_ae_created            ON pack029_interim_targets.gl_assurance_evidence(created_at DESC);
CREATE INDEX idx_p029_ae_completeness_gaps  ON pack029_interim_targets.gl_assurance_evidence USING GIN(completeness_gaps);
CREATE INDEX idx_p029_ae_metadata           ON pack029_interim_targets.gl_assurance_evidence USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_assurance_evidence_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_assurance_evidence
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_assurance_evidence ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_ae_tenant_isolation
    ON pack029_interim_targets.gl_assurance_evidence
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_ae_service_bypass
    ON pack029_interim_targets.gl_assurance_evidence
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_assurance_evidence TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_assurance_evidence IS
    'Assurance evidence and ISO 14064-3 workpaper tracking with evidence type classification, document management, completeness scoring, quality assessment, and assurance provider review status.';

COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.evidence_id IS 'Unique assurance evidence identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.organization_id IS 'Reference to the organization this evidence belongs to.';
COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.evidence_type IS 'Evidence type: CALCULATION_SHEET, DATA_SOURCE, METHODOLOGY, VARIANCE_EXPLANATION, etc.';
COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.document_url IS 'URL or path to the evidence document.';
COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.evidence_tier IS 'Evidence tier: PRIMARY (direct measurement), SECONDARY (calculated), TERTIARY (estimated).';
COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.completeness_pct IS 'Completeness percentage of the evidence documentation (0-100).';
COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.assurance_provider_reviewed IS 'Whether this evidence has been reviewed by the assurance provider.';
COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.overall_quality_score IS 'Overall quality score combining reliability, relevance, and sufficiency.';
COMMENT ON COLUMN pack029_interim_targets.gl_assurance_evidence.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

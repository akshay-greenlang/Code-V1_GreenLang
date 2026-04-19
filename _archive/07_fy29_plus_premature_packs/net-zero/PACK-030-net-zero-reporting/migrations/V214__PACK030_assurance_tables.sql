-- =============================================================================
-- V214: PACK-030 Net Zero Reporting Pack - Assurance Evidence Tables
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    004 of 015
-- Date:         March 2026
--
-- Assurance evidence packaging for ISAE 3410/3000 audit readiness with
-- evidence type classification, SHA-256 checksums, completeness scoring,
-- and control matrix generation support.
--
-- Tables (1):
--   1. pack030_nz_reporting.gl_nz_assurance_evidence
--
-- Previous: V213__PACK030_narrative_tables.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack030_nz_reporting.gl_nz_assurance_evidence
-- =============================================================================
-- Assurance evidence records for ISAE 3410/3000 audit bundles with evidence
-- type classification, document management, SHA-256 checksums, completeness
-- scoring, quality assessment, and auditor review tracking.

CREATE TABLE pack030_nz_reporting.gl_nz_assurance_evidence (
    evidence_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    report_id                   UUID            NOT NULL REFERENCES pack030_nz_reporting.gl_nz_reports(report_id) ON DELETE CASCADE,
    -- Evidence classification
    evidence_type               VARCHAR(100)    NOT NULL,
    evidence_subtype            VARCHAR(100),
    evidence_title              VARCHAR(500)    NOT NULL,
    evidence_description        TEXT,
    -- Document management
    file_path                   VARCHAR(500),
    file_name                   VARCHAR(255),
    file_size_bytes             BIGINT,
    mime_type                   VARCHAR(100),
    -- Integrity
    checksum                    CHAR(64)        NOT NULL,
    checksum_algorithm          VARCHAR(20)     NOT NULL DEFAULT 'SHA-256',
    -- Assurance standard
    assurance_standard          VARCHAR(50),
    assurance_level             VARCHAR(30),
    -- Evidence tier
    evidence_tier               VARCHAR(20)     NOT NULL DEFAULT 'SECONDARY',
    -- Completeness
    completeness_pct            DECIMAL(5,2)    DEFAULT 0,
    completeness_gaps           JSONB           NOT NULL DEFAULT '[]',
    -- Source linkage
    source_metric_ids           JSONB           NOT NULL DEFAULT '[]',
    source_system               VARCHAR(100),
    source_pack                 VARCHAR(50),
    calculation_reference       VARCHAR(200),
    -- Lineage
    data_lineage_summary        JSONB           NOT NULL DEFAULT '{}',
    -- Methodology
    methodology_documented      BOOLEAN         NOT NULL DEFAULT FALSE,
    methodology_reference       VARCHAR(500),
    -- Control matrix
    control_objective           VARCHAR(200),
    control_activity            VARCHAR(200),
    control_evidence_adequate   BOOLEAN,
    -- Auditor review
    auditor_reviewed            BOOLEAN         NOT NULL DEFAULT FALSE,
    auditor_name                VARCHAR(255),
    auditor_firm                VARCHAR(200),
    review_date                 DATE,
    review_outcome              VARCHAR(30),
    review_comments             TEXT,
    -- Quality
    reliability_score           DECIMAL(5,2),
    relevance_score             DECIMAL(5,2),
    sufficiency_score           DECIMAL(5,2),
    overall_quality_score       DECIMAL(5,2),
    -- Retention
    retention_policy            VARCHAR(30)     NOT NULL DEFAULT 'STANDARD',
    retention_until             DATE,
    archived                    BOOLEAN         NOT NULL DEFAULT FALSE,
    archived_at                 TIMESTAMPTZ,
    -- Bundle
    bundle_id                   UUID,
    bundle_order                INTEGER,
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_ae_evidence_type CHECK (
        evidence_type IN (
            'PROVENANCE_HASH', 'DATA_LINEAGE', 'METHODOLOGY_DOC', 'CONTROL_MATRIX',
            'CALCULATION_SHEET', 'DATA_SOURCE', 'EMISSION_FACTOR', 'ACTIVITY_DATA',
            'SUPPLIER_DATA', 'METER_DATA', 'INVOICE', 'AUDIT_REPORT',
            'THIRD_PARTY_VERIFICATION', 'BOARD_APPROVAL', 'MANAGEMENT_ASSERTION',
            'BOUNDARY_DEFINITION', 'RESTATEMENT_JUSTIFICATION', 'METHODOLOGY_CHANGE',
            'UNCERTAINTY_ANALYSIS', 'QUALITY_CONTROL', 'INTERNAL_AUDIT',
            'CROSS_FRAMEWORK_RECONCILIATION', 'NARRATIVE_CONSISTENCY_CHECK', 'OTHER'
        )
    ),
    CONSTRAINT chk_p030_ae_evidence_tier CHECK (
        evidence_tier IN ('PRIMARY', 'SECONDARY', 'TERTIARY')
    ),
    CONSTRAINT chk_p030_ae_assurance_standard CHECK (
        assurance_standard IS NULL OR assurance_standard IN (
            'ISAE_3410', 'ISAE_3000', 'AA1000AS', 'ISO_14064_3', 'SOC_2', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p030_ae_assurance_level CHECK (
        assurance_level IS NULL OR assurance_level IN (
            'LIMITED', 'REASONABLE', 'HIGH', 'MODERATE', 'NONE'
        )
    ),
    CONSTRAINT chk_p030_ae_review_outcome CHECK (
        review_outcome IS NULL OR review_outcome IN (
            'ACCEPTED', 'ACCEPTED_WITH_COMMENTS', 'REJECTED', 'FURTHER_INFO_REQUIRED', 'PENDING'
        )
    ),
    CONSTRAINT chk_p030_ae_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p030_ae_quality_scores CHECK (
        (reliability_score IS NULL OR (reliability_score >= 0 AND reliability_score <= 100))
        AND (relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 100))
        AND (sufficiency_score IS NULL OR (sufficiency_score >= 0 AND sufficiency_score <= 100))
        AND (overall_quality_score IS NULL OR (overall_quality_score >= 0 AND overall_quality_score <= 100))
    ),
    CONSTRAINT chk_p030_ae_retention_policy CHECK (
        retention_policy IN ('STANDARD', 'EXTENDED', 'PERMANENT', 'REGULATORY', 'CUSTOM')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_assurance_evidence
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_ae_tenant               ON pack030_nz_reporting.gl_nz_assurance_evidence(tenant_id);
CREATE INDEX idx_p030_ae_org                  ON pack030_nz_reporting.gl_nz_assurance_evidence(organization_id);
CREATE INDEX idx_p030_ae_report               ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id);
CREATE INDEX idx_p030_ae_report_type          ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id, evidence_type);
CREATE INDEX idx_p030_ae_evidence_type        ON pack030_nz_reporting.gl_nz_assurance_evidence(evidence_type);
CREATE INDEX idx_p030_ae_evidence_tier        ON pack030_nz_reporting.gl_nz_assurance_evidence(evidence_tier);
CREATE INDEX idx_p030_ae_checksum             ON pack030_nz_reporting.gl_nz_assurance_evidence(checksum);
CREATE INDEX idx_p030_ae_assurance_std        ON pack030_nz_reporting.gl_nz_assurance_evidence(assurance_standard);
CREATE INDEX idx_p030_ae_completeness         ON pack030_nz_reporting.gl_nz_assurance_evidence(completeness_pct);
CREATE INDEX idx_p030_ae_incomplete           ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id) WHERE completeness_pct < 100;
CREATE INDEX idx_p030_ae_unreviewed           ON pack030_nz_reporting.gl_nz_assurance_evidence(organization_id) WHERE auditor_reviewed = FALSE AND is_active = TRUE;
CREATE INDEX idx_p030_ae_reviewed             ON pack030_nz_reporting.gl_nz_assurance_evidence(review_date DESC) WHERE auditor_reviewed = TRUE;
CREATE INDEX idx_p030_ae_review_outcome       ON pack030_nz_reporting.gl_nz_assurance_evidence(review_outcome);
CREATE INDEX idx_p030_ae_rejected             ON pack030_nz_reporting.gl_nz_assurance_evidence(organization_id) WHERE review_outcome = 'REJECTED';
CREATE INDEX idx_p030_ae_bundle               ON pack030_nz_reporting.gl_nz_assurance_evidence(bundle_id, bundle_order);
CREATE INDEX idx_p030_ae_source_pack          ON pack030_nz_reporting.gl_nz_assurance_evidence(source_pack);
CREATE INDEX idx_p030_ae_active               ON pack030_nz_reporting.gl_nz_assurance_evidence(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_ae_retention_due        ON pack030_nz_reporting.gl_nz_assurance_evidence(retention_until) WHERE archived = FALSE AND retention_until IS NOT NULL;
CREATE INDEX idx_p030_ae_created              ON pack030_nz_reporting.gl_nz_assurance_evidence(created_at DESC);
CREATE INDEX idx_p030_ae_source_metrics       ON pack030_nz_reporting.gl_nz_assurance_evidence USING GIN(source_metric_ids);
CREATE INDEX idx_p030_ae_completeness_gaps    ON pack030_nz_reporting.gl_nz_assurance_evidence USING GIN(completeness_gaps);
CREATE INDEX idx_p030_ae_metadata             ON pack030_nz_reporting.gl_nz_assurance_evidence USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_assurance_evidence
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_assurance_evidence_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_assurance_evidence
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_assurance_evidence
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_assurance_evidence ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_ae_tenant_isolation
    ON pack030_nz_reporting.gl_nz_assurance_evidence
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_ae_service_bypass
    ON pack030_nz_reporting.gl_nz_assurance_evidence
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_assurance_evidence TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack030_nz_reporting.gl_nz_assurance_evidence IS
    'Assurance evidence records for ISAE 3410/3000 audit bundles with evidence type classification, SHA-256 checksums, completeness scoring, control matrix mapping, quality assessment, auditor review tracking, and retention management.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_assurance_evidence.evidence_id IS 'Unique evidence record identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_assurance_evidence.evidence_type IS 'Evidence classification: PROVENANCE_HASH, DATA_LINEAGE, METHODOLOGY_DOC, CONTROL_MATRIX, etc.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_assurance_evidence.checksum IS 'SHA-256 checksum for evidence document integrity verification.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_assurance_evidence.evidence_tier IS 'Evidence tier: PRIMARY (direct measurement), SECONDARY (calculated), TERTIARY (estimated).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_assurance_evidence.completeness_pct IS 'Evidence documentation completeness percentage (0-100).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_assurance_evidence.assurance_standard IS 'Target assurance standard: ISAE_3410, ISAE_3000, AA1000AS, ISO_14064_3, SOC_2.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_assurance_evidence.control_objective IS 'ISAE 3410 control objective this evidence supports.';

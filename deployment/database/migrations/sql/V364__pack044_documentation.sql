-- =============================================================================
-- V364: PACK-044 GHG Inventory Management - Documentation Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Documentation and evidence management tables for GHG inventory audit
-- trails. Methodology documents capture the quantification approaches used.
-- Assumptions register tracks all assumptions with justification and
-- sensitivity analysis. Evidence records link raw supporting documents
-- (invoices, meter readings, certificates) to specific inventory data points.
--
-- Tables (3):
--   1. ghg_inventory.gl_inv_methodology_docs
--   2. ghg_inventory.gl_inv_assumptions
--   3. ghg_inventory.gl_inv_evidence_records
--
-- Previous: V363__pack044_gap_analysis.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_methodology_docs
-- =============================================================================
-- Methodology documentation for the GHG inventory. Records the quantification
-- methodology for each emission source including calculation approach,
-- emission factor selection rationale, data sources, and quality procedures.
-- Required for verification readiness per GHG Protocol Chapter 8 and
-- ISO 14064-1 clause 6.3.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_methodology_docs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    doc_code                    VARCHAR(50)     NOT NULL,
    doc_title                   VARCHAR(300)    NOT NULL,
    doc_type                    VARCHAR(30)     NOT NULL DEFAULT 'QUANTIFICATION',
    scope                       VARCHAR(10),
    source_category             VARCHAR(60),
    version                     INTEGER         NOT NULL DEFAULT 1,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    content                     TEXT,
    calculation_approach        VARCHAR(50),
    emission_factor_source      VARCHAR(200),
    emission_factor_rationale   TEXT,
    data_sources                TEXT,
    data_collection_procedure   TEXT,
    qaqc_procedure              TEXT,
    uncertainty_approach        TEXT,
    limitations                 TEXT,
    author_user_id              UUID,
    author_name                 VARCHAR(255),
    reviewed_by_name            VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    approved_by_name            VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    file_id                     UUID,
    file_url                    TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_md_type CHECK (
        doc_type IN (
            'QUANTIFICATION', 'MONITORING_PLAN', 'QC_PROCEDURE',
            'BOUNDARY_DESCRIPTION', 'BASE_YEAR_POLICY', 'RESTATEMENT_POLICY',
            'UNCERTAINTY_ANALYSIS', 'VERIFICATION_PLAN', 'GENERAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p044_md_scope CHECK (
        scope IS NULL OR scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ALL')
    ),
    CONSTRAINT chk_p044_md_version CHECK (
        version >= 1
    ),
    CONSTRAINT chk_p044_md_status CHECK (
        status IN ('DRAFT', 'UNDER_REVIEW', 'APPROVED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p044_md_approach CHECK (
        calculation_approach IS NULL OR calculation_approach IN (
            'CALCULATION_BASED', 'DIRECT_MEASUREMENT', 'MASS_BALANCE',
            'SPEND_BASED', 'DISTANCE_BASED', 'ASSET_BASED',
            'AVERAGE_DATA', 'SUPPLIER_SPECIFIC', 'HYBRID', 'OTHER'
        )
    ),
    CONSTRAINT uq_p044_md_period_code_version UNIQUE (period_id, doc_code, version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_md_tenant          ON ghg_inventory.gl_inv_methodology_docs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_md_period          ON ghg_inventory.gl_inv_methodology_docs(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_md_code            ON ghg_inventory.gl_inv_methodology_docs(doc_code);
CREATE INDEX IF NOT EXISTS idx_p044_md_type            ON ghg_inventory.gl_inv_methodology_docs(doc_type);
CREATE INDEX IF NOT EXISTS idx_p044_md_scope           ON ghg_inventory.gl_inv_methodology_docs(scope);
CREATE INDEX IF NOT EXISTS idx_p044_md_category        ON ghg_inventory.gl_inv_methodology_docs(source_category);
CREATE INDEX IF NOT EXISTS idx_p044_md_status          ON ghg_inventory.gl_inv_methodology_docs(status);
CREATE INDEX IF NOT EXISTS idx_p044_md_created         ON ghg_inventory.gl_inv_methodology_docs(created_at DESC);

-- Composite: period + approved docs
CREATE INDEX IF NOT EXISTS idx_p044_md_period_approved ON ghg_inventory.gl_inv_methodology_docs(period_id, doc_type)
    WHERE status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_md_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_methodology_docs
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_assumptions
-- =============================================================================
-- Register of all assumptions made during inventory preparation. Each
-- assumption is documented with its justification, sensitivity to the
-- overall result, and the conditions under which it should be reviewed.
-- Critical for transparency, verification, and year-on-year consistency.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_assumptions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    assumption_code             VARCHAR(50)     NOT NULL,
    assumption_title            VARCHAR(300)    NOT NULL,
    assumption_description      TEXT            NOT NULL,
    assumption_category         VARCHAR(50)     NOT NULL,
    scope                       VARCHAR(10),
    source_category             VARCHAR(60),
    facility_id                 UUID,
    justification               TEXT            NOT NULL,
    data_source                 VARCHAR(200),
    sensitivity                 VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    impact_if_wrong_tco2e       NUMERIC(12,3),
    impact_if_wrong_pct         NUMERIC(8,3),
    review_trigger              TEXT,
    valid_from                  DATE,
    valid_to                    DATE,
    is_recurring                BOOLEAN         NOT NULL DEFAULT true,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    approved_by_name            VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_as_category CHECK (
        assumption_category IN (
            'EMISSION_FACTOR', 'ACTIVITY_DATA', 'BOUNDARY', 'METHODOLOGY',
            'ESTIMATION', 'ALLOCATION', 'GWP', 'OPERATIONAL',
            'FINANCIAL', 'STRUCTURAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p044_as_scope CHECK (
        scope IS NULL OR scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ALL')
    ),
    CONSTRAINT chk_p044_as_sensitivity CHECK (
        sensitivity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE')
    ),
    CONSTRAINT chk_p044_as_status CHECK (
        status IN ('ACTIVE', 'UNDER_REVIEW', 'SUPERSEDED', 'RETIRED', 'REJECTED')
    ),
    CONSTRAINT chk_p044_as_dates CHECK (
        valid_from IS NULL OR valid_to IS NULL OR valid_from <= valid_to
    ),
    CONSTRAINT uq_p044_as_period_code UNIQUE (period_id, assumption_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_as_tenant          ON ghg_inventory.gl_inv_assumptions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_as_period          ON ghg_inventory.gl_inv_assumptions(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_as_code            ON ghg_inventory.gl_inv_assumptions(assumption_code);
CREATE INDEX IF NOT EXISTS idx_p044_as_category        ON ghg_inventory.gl_inv_assumptions(assumption_category);
CREATE INDEX IF NOT EXISTS idx_p044_as_scope           ON ghg_inventory.gl_inv_assumptions(scope);
CREATE INDEX IF NOT EXISTS idx_p044_as_facility        ON ghg_inventory.gl_inv_assumptions(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_as_sensitivity     ON ghg_inventory.gl_inv_assumptions(sensitivity);
CREATE INDEX IF NOT EXISTS idx_p044_as_status          ON ghg_inventory.gl_inv_assumptions(status);
CREATE INDEX IF NOT EXISTS idx_p044_as_created         ON ghg_inventory.gl_inv_assumptions(created_at DESC);

-- Composite: period + active high-sensitivity assumptions
CREATE INDEX IF NOT EXISTS idx_p044_as_period_critical ON ghg_inventory.gl_inv_assumptions(period_id, sensitivity)
    WHERE status = 'ACTIVE' AND sensitivity IN ('CRITICAL', 'HIGH');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_as_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_assumptions
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_evidence_records
-- =============================================================================
-- Evidence and supporting document records linked to inventory data. Tracks
-- invoices, meter readings, energy certificates, fleet logs, ERP extracts,
-- and other source documents that substantiate reported activity data.
-- Essential for third-party verification and audit trail completeness.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_evidence_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    evidence_code               VARCHAR(50)     NOT NULL,
    evidence_title              VARCHAR(300)    NOT NULL,
    evidence_type               VARCHAR(30)     NOT NULL,
    source_description          TEXT,
    facility_id                 UUID,
    source_category             VARCHAR(60),
    submission_id               UUID,
    linked_entity_type          VARCHAR(50),
    linked_entity_id            UUID,
    file_id                     UUID,
    file_name                   VARCHAR(500),
    file_type                   VARCHAR(50),
    file_size_bytes             BIGINT,
    file_url                    TEXT,
    file_hash                   VARCHAR(64),
    covers_period_start         DATE,
    covers_period_end           DATE,
    data_provider               VARCHAR(255),
    is_third_party              BOOLEAN         NOT NULL DEFAULT false,
    verification_status         VARCHAR(30)     DEFAULT 'NOT_VERIFIED',
    verified_by_name            VARCHAR(255),
    verified_at                 TIMESTAMPTZ,
    confidentiality_level       VARCHAR(20)     DEFAULT 'INTERNAL',
    retention_until             DATE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ev_type CHECK (
        evidence_type IN (
            'INVOICE', 'METER_READING', 'UTILITY_BILL', 'FUEL_RECEIPT',
            'ENERGY_CERTIFICATE', 'ERP_EXTRACT', 'FLEET_LOG', 'TELEMATICS_EXPORT',
            'SUPPLIER_REPORT', 'CALIBRATION_CERTIFICATE', 'LABORATORY_ANALYSIS',
            'PHOTOGRAPH', 'CALCULATION_SPREADSHEET', 'METHODOLOGY_DOCUMENT',
            'CORRESPONDENCE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p044_ev_linked_type CHECK (
        linked_entity_type IS NULL OR linked_entity_type IN (
            'SUBMISSION', 'EMISSION_RECORD', 'ASSUMPTION', 'METHODOLOGY',
            'CHANGE_REQUEST', 'QUALITY_ISSUE', 'REVIEW_COMMENT'
        )
    ),
    CONSTRAINT chk_p044_ev_verification CHECK (
        verification_status IS NULL OR verification_status IN (
            'NOT_VERIFIED', 'VERIFIED', 'PARTIALLY_VERIFIED', 'REJECTED', 'PENDING'
        )
    ),
    CONSTRAINT chk_p044_ev_confidentiality CHECK (
        confidentiality_level IS NULL OR confidentiality_level IN (
            'PUBLIC', 'INTERNAL', 'CONFIDENTIAL', 'RESTRICTED'
        )
    ),
    CONSTRAINT chk_p044_ev_file_size CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    ),
    CONSTRAINT chk_p044_ev_period_dates CHECK (
        covers_period_start IS NULL OR covers_period_end IS NULL OR
        covers_period_start <= covers_period_end
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ev_tenant          ON ghg_inventory.gl_inv_evidence_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ev_period          ON ghg_inventory.gl_inv_evidence_records(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_ev_code            ON ghg_inventory.gl_inv_evidence_records(evidence_code);
CREATE INDEX IF NOT EXISTS idx_p044_ev_type            ON ghg_inventory.gl_inv_evidence_records(evidence_type);
CREATE INDEX IF NOT EXISTS idx_p044_ev_facility        ON ghg_inventory.gl_inv_evidence_records(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_ev_category        ON ghg_inventory.gl_inv_evidence_records(source_category);
CREATE INDEX IF NOT EXISTS idx_p044_ev_submission      ON ghg_inventory.gl_inv_evidence_records(submission_id);
CREATE INDEX IF NOT EXISTS idx_p044_ev_linked          ON ghg_inventory.gl_inv_evidence_records(linked_entity_type, linked_entity_id);
CREATE INDEX IF NOT EXISTS idx_p044_ev_verification    ON ghg_inventory.gl_inv_evidence_records(verification_status);
CREATE INDEX IF NOT EXISTS idx_p044_ev_file_hash       ON ghg_inventory.gl_inv_evidence_records(file_hash);
CREATE INDEX IF NOT EXISTS idx_p044_ev_created         ON ghg_inventory.gl_inv_evidence_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_p044_ev_metadata        ON ghg_inventory.gl_inv_evidence_records USING GIN(metadata);

-- Composite: period + evidence by type
CREATE INDEX IF NOT EXISTS idx_p044_ev_period_type     ON ghg_inventory.gl_inv_evidence_records(period_id, evidence_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ev_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_evidence_records
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_methodology_docs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_assumptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_evidence_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_md_tenant_isolation
    ON ghg_inventory.gl_inv_methodology_docs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_md_service_bypass
    ON ghg_inventory.gl_inv_methodology_docs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_as_tenant_isolation
    ON ghg_inventory.gl_inv_assumptions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_as_service_bypass
    ON ghg_inventory.gl_inv_assumptions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_ev_tenant_isolation
    ON ghg_inventory.gl_inv_evidence_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ev_service_bypass
    ON ghg_inventory.gl_inv_evidence_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_methodology_docs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_assumptions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_evidence_records TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_methodology_docs IS
    'Methodology documentation for GHG inventory quantification approaches per GHG Protocol Chapter 8 and ISO 14064-1.';
COMMENT ON TABLE ghg_inventory.gl_inv_assumptions IS
    'Register of all assumptions made during inventory preparation with justification and sensitivity analysis.';
COMMENT ON TABLE ghg_inventory.gl_inv_evidence_records IS
    'Evidence and supporting documents linked to inventory data for verification and audit trail completeness.';

COMMENT ON COLUMN ghg_inventory.gl_inv_methodology_docs.doc_type IS 'Document type: QUANTIFICATION, MONITORING_PLAN, QC_PROCEDURE, BOUNDARY_DESCRIPTION, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_methodology_docs.calculation_approach IS 'Calculation approach: CALCULATION_BASED, DIRECT_MEASUREMENT, MASS_BALANCE, SPEND_BASED, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_assumptions.sensitivity IS 'How sensitive the overall inventory result is to this assumption: CRITICAL, HIGH, MEDIUM, LOW, NEGLIGIBLE.';
COMMENT ON COLUMN ghg_inventory.gl_inv_assumptions.review_trigger IS 'Conditions under which this assumption should be reviewed and potentially updated.';
COMMENT ON COLUMN ghg_inventory.gl_inv_evidence_records.evidence_type IS 'Type of evidence: INVOICE, METER_READING, UTILITY_BILL, ENERGY_CERTIFICATE, ERP_EXTRACT, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_evidence_records.file_hash IS 'SHA-256 hash of the evidence file for integrity verification.';
COMMENT ON COLUMN ghg_inventory.gl_inv_evidence_records.confidentiality_level IS 'Data classification: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED.';

-- =============================================================================
-- V334: PACK-041 Scope 1-2 Complete Pack - Reporting & Audit Trail
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates reporting, verification, and audit trail tables. Report metadata
-- tracks generated GHG reports (CDP, CSRD, custom). Verification packages
-- bundle all materials needed for third-party verification. The audit trail
-- captures all data changes across the ghg_scope12 schema for complete
-- traceability. The audit trail table is configured as a TimescaleDB
-- hypertable for efficient time-series queries over the append-only log.
--
-- Tables (3):
--   1. ghg_scope12.report_metadata
--   2. ghg_scope12.verification_packages
--   3. ghg_scope12.audit_trail_entries
--
-- Also includes: TimescaleDB hypertable, indexes, RLS, comments.
-- Previous: V333__pack041_compliance.sql
-- =============================================================================

SET search_path TO ghg_scope12, public;

-- =============================================================================
-- Table 1: ghg_scope12.report_metadata
-- =============================================================================
-- Metadata for generated GHG inventory reports. Tracks report type (CDP
-- questionnaire, CSRD disclosure, SEC filing, custom report), format,
-- generation parameters, and provenance hash for ensuring report integrity.
-- Reports are stored in object storage (S3); this table holds the metadata.

CREATE TABLE ghg_scope12.report_metadata (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    -- Report identification
    report_name                 VARCHAR(500)    NOT NULL,
    report_type                 VARCHAR(50)     NOT NULL,
    report_sub_type             VARCHAR(50),
    reporting_year              INTEGER         NOT NULL,
    -- Content scope
    scope_coverage              VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_1_2',
    scope1_inventory_id         UUID            REFERENCES ghg_scope12.scope1_inventories(id) ON DELETE SET NULL,
    scope2_inventory_id         UUID            REFERENCES ghg_scope12.scope2_inventories(id) ON DELETE SET NULL,
    compliance_assessment_id    UUID            REFERENCES ghg_scope12.compliance_assessments(id) ON DELETE SET NULL,
    -- Format
    format                      VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    template_id                 VARCHAR(100),
    template_version            VARCHAR(20),
    language                    VARCHAR(10)     NOT NULL DEFAULT 'en',
    -- Generation
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    generated_by                VARCHAR(255),
    generation_duration_ms      INTEGER,
    generation_parameters       JSONB           DEFAULT '{}',
    -- File storage
    file_path                   TEXT,
    file_name                   VARCHAR(500),
    file_size_bytes             BIGINT,
    file_checksum_sha256        VARCHAR(64),
    storage_bucket              VARCHAR(255),
    -- Version management
    version                     INTEGER         NOT NULL DEFAULT 1,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    previous_version_id         UUID            REFERENCES ghg_scope12.report_metadata(id) ON DELETE SET NULL,
    -- Distribution
    distribution_list           TEXT[],
    published_at                TIMESTAMPTZ,
    submitted_to                VARCHAR(255),
    submission_date             DATE,
    submission_reference        VARCHAR(200),
    -- Workflow
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    reviewed_by                 VARCHAR(255),
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_rm_report_type CHECK (
        report_type IN (
            'CDP_QUESTIONNAIRE', 'CSRD_ESRS_E1', 'SEC_FILING', 'TCFD_REPORT',
            'ISO_14064_REPORT', 'GHG_PROTOCOL_REPORT', 'ANNUAL_SUSTAINABILITY',
            'BOARD_SUMMARY', 'EXECUTIVE_DASHBOARD', 'FACILITY_REPORT',
            'VERIFICATION_READY', 'REGULATORY_FILING', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_rm_scope CHECK (
        scope_coverage IN (
            'SCOPE_1', 'SCOPE_2', 'SCOPE_1_2', 'SCOPE_1_2_3', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_rm_format CHECK (
        format IN ('PDF', 'HTML', 'XLSX', 'CSV', 'JSON', 'XML', 'XBRL', 'MARKDOWN', 'DOCX')
    ),
    CONSTRAINT chk_p041_rm_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p041_rm_version CHECK (
        version >= 1
    ),
    CONSTRAINT chk_p041_rm_file_size CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    ),
    CONSTRAINT chk_p041_rm_status CHECK (
        status IN (
            'DRAFT', 'GENERATING', 'GENERATED', 'REVIEWED', 'APPROVED',
            'PUBLISHED', 'SUBMITTED', 'ARCHIVED', 'SUPERSEDED', 'ERROR'
        )
    ),
    CONSTRAINT chk_p041_rm_duration CHECK (
        generation_duration_ms IS NULL OR generation_duration_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_rm_tenant             ON ghg_scope12.report_metadata(tenant_id);
CREATE INDEX idx_p041_rm_org               ON ghg_scope12.report_metadata(organization_id);
CREATE INDEX idx_p041_rm_report_type       ON ghg_scope12.report_metadata(report_type);
CREATE INDEX idx_p041_rm_year              ON ghg_scope12.report_metadata(reporting_year);
CREATE INDEX idx_p041_rm_format            ON ghg_scope12.report_metadata(format);
CREATE INDEX idx_p041_rm_status            ON ghg_scope12.report_metadata(status);
CREATE INDEX idx_p041_rm_current           ON ghg_scope12.report_metadata(is_current) WHERE is_current = true;
CREATE INDEX idx_p041_rm_generated         ON ghg_scope12.report_metadata(generated_at DESC);
CREATE INDEX idx_p041_rm_submitted         ON ghg_scope12.report_metadata(submission_date);
CREATE INDEX idx_p041_rm_created           ON ghg_scope12.report_metadata(created_at DESC);
CREATE INDEX idx_p041_rm_metadata          ON ghg_scope12.report_metadata USING GIN(metadata);
CREATE INDEX idx_p041_rm_params            ON ghg_scope12.report_metadata USING GIN(generation_parameters);

-- Composite: org + type + year + current for latest report
CREATE INDEX idx_p041_rm_org_type_year     ON ghg_scope12.report_metadata(organization_id, report_type, reporting_year DESC)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_rm_updated
    BEFORE UPDATE ON ghg_scope12.report_metadata
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.verification_packages
-- =============================================================================
-- Bundles all materials needed for third-party verification of the GHG
-- inventory. Includes verifier details, verification standard (ISO 14064-3,
-- ISAE 3410), assurance level (limited or reasonable), and the verification
-- statement details. Tracks the verification lifecycle from preparation
-- through statement issuance.

CREATE TABLE ghg_scope12.verification_packages (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    reporting_year              INTEGER         NOT NULL,
    -- Scope
    scope_coverage              VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_1_2',
    scope1_inventory_id         UUID            REFERENCES ghg_scope12.scope1_inventories(id) ON DELETE SET NULL,
    scope2_inventory_id         UUID            REFERENCES ghg_scope12.scope2_inventories(id) ON DELETE SET NULL,
    -- Verifier
    verifier_name               VARCHAR(255),
    verifier_organization       VARCHAR(255),
    verifier_accreditation      VARCHAR(200),
    lead_verifier               VARCHAR(255),
    -- Verification details
    verification_standard       VARCHAR(50)     NOT NULL DEFAULT 'ISO_14064_3',
    assurance_level             VARCHAR(20)     NOT NULL DEFAULT 'LIMITED',
    materiality_threshold_pct   DECIMAL(5,2)    DEFAULT 5.00,
    materiality_threshold_tco2e DECIMAL(12,3),
    -- Schedule
    engagement_start_date       DATE,
    site_visit_dates            DATE[],
    draft_statement_date        DATE,
    final_statement_date        DATE,
    verification_date           DATE,
    -- Results
    verification_opinion        VARCHAR(30),
    qualified                   BOOLEAN,
    qualification_details       TEXT,
    findings_count              INTEGER         DEFAULT 0,
    corrective_actions_count    INTEGER         DEFAULT 0,
    corrective_actions_resolved INTEGER         DEFAULT 0,
    -- Documents
    verification_statement_path TEXT,
    verification_report_path    TEXT,
    evidence_package_path       TEXT,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    cost                        DECIMAL(18,2),
    cost_currency               VARCHAR(3)      DEFAULT 'USD',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_vp_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p041_vp_scope CHECK (
        scope_coverage IN (
            'SCOPE_1', 'SCOPE_2', 'SCOPE_1_2', 'SCOPE_1_2_3', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_vp_standard CHECK (
        verification_standard IN (
            'ISO_14064_3', 'ISAE_3410', 'ISAE_3000', 'AA1000AS',
            'CARB', 'EU_ETS_AVR', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_vp_assurance CHECK (
        assurance_level IN ('LIMITED', 'REASONABLE', 'COMBINED')
    ),
    CONSTRAINT chk_p041_vp_opinion CHECK (
        verification_opinion IS NULL OR verification_opinion IN (
            'UNQUALIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER'
        )
    ),
    CONSTRAINT chk_p041_vp_status CHECK (
        status IN (
            'DRAFT', 'PREPARATION', 'ENGAGEMENT', 'SITE_VISIT',
            'REVIEW', 'DRAFT_STATEMENT', 'FINAL_STATEMENT',
            'COMPLETED', 'CANCELLED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p041_vp_materiality_pct CHECK (
        materiality_threshold_pct IS NULL OR (materiality_threshold_pct > 0 AND materiality_threshold_pct <= 20)
    ),
    CONSTRAINT chk_p041_vp_findings CHECK (
        findings_count IS NULL OR findings_count >= 0
    ),
    CONSTRAINT chk_p041_vp_corrective CHECK (
        corrective_actions_count IS NULL OR corrective_actions_count >= 0
    ),
    CONSTRAINT chk_p041_vp_resolved CHECK (
        corrective_actions_resolved IS NULL OR corrective_actions_count IS NULL OR
        corrective_actions_resolved <= corrective_actions_count
    ),
    CONSTRAINT chk_p041_vp_cost CHECK (
        cost IS NULL OR cost >= 0
    ),
    CONSTRAINT uq_p041_vp_org_year_scope UNIQUE (organization_id, reporting_year, scope_coverage)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_vp_tenant             ON ghg_scope12.verification_packages(tenant_id);
CREATE INDEX idx_p041_vp_org               ON ghg_scope12.verification_packages(organization_id);
CREATE INDEX idx_p041_vp_year              ON ghg_scope12.verification_packages(reporting_year);
CREATE INDEX idx_p041_vp_standard          ON ghg_scope12.verification_packages(verification_standard);
CREATE INDEX idx_p041_vp_assurance         ON ghg_scope12.verification_packages(assurance_level);
CREATE INDEX idx_p041_vp_verifier          ON ghg_scope12.verification_packages(verifier_name);
CREATE INDEX idx_p041_vp_opinion           ON ghg_scope12.verification_packages(verification_opinion);
CREATE INDEX idx_p041_vp_status            ON ghg_scope12.verification_packages(status);
CREATE INDEX idx_p041_vp_created           ON ghg_scope12.verification_packages(created_at DESC);
CREATE INDEX idx_p041_vp_metadata          ON ghg_scope12.verification_packages USING GIN(metadata);

-- Composite: org + year for history
CREATE INDEX idx_p041_vp_org_year          ON ghg_scope12.verification_packages(organization_id, reporting_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_vp_updated
    BEFORE UPDATE ON ghg_scope12.verification_packages
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.audit_trail_entries
-- =============================================================================
-- Append-only audit trail capturing all data changes across the ghg_scope12
-- schema. Every INSERT, UPDATE, DELETE operation on inventory-related tables
-- is recorded with old/new values, the performing user, and a provenance
-- hash. Configured as a TimescaleDB hypertable for efficient time-range
-- queries over potentially millions of audit records.

CREATE TABLE ghg_scope12.audit_trail_entries (
    id                          UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- What changed
    entity_type                 VARCHAR(100)    NOT NULL,
    entity_id                   UUID            NOT NULL,
    entity_name                 VARCHAR(500),
    -- Change details
    action                      VARCHAR(20)     NOT NULL,
    change_category             VARCHAR(30),
    old_value                   JSONB,
    new_value                   JSONB,
    changed_fields              TEXT[],
    change_summary              TEXT,
    -- Context
    organization_id             UUID,
    facility_id                 UUID,
    reporting_year              INTEGER,
    scope                       VARCHAR(10),
    -- Who/when
    performed_by                VARCHAR(255)    NOT NULL,
    performed_by_id             UUID,
    performed_by_role           VARCHAR(50),
    performed_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- System context
    source_system               VARCHAR(100)    DEFAULT 'GREENLANG',
    source_agent                VARCHAR(100),
    api_endpoint                VARCHAR(500),
    session_id                  VARCHAR(100),
    ip_address                  VARCHAR(45),
    -- Provenance
    provenance_hash             VARCHAR(64)     NOT NULL,
    parent_hash                 VARCHAR(64),
    -- Constraints
    CONSTRAINT chk_p041_at_action CHECK (
        action IN (
            'CREATE', 'UPDATE', 'DELETE', 'APPROVE', 'REJECT',
            'FINALIZE', 'RESTATE', 'RECALCULATE', 'SUBMIT',
            'VERIFY', 'ARCHIVE', 'RESTORE', 'IMPORT', 'EXPORT'
        )
    ),
    CONSTRAINT chk_p041_at_change_cat CHECK (
        change_category IS NULL OR change_category IN (
            'DATA_ENTRY', 'CALCULATION', 'BOUNDARY_CHANGE', 'FACTOR_UPDATE',
            'METHODOLOGY_CHANGE', 'RECALCULATION', 'CORRECTION',
            'VERIFICATION', 'APPROVAL', 'REPORTING', 'SYSTEM', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_at_scope CHECK (
        scope IS NULL OR scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_1_2')
    )
);

-- Set performed_at as primary key component for hypertable
-- Note: UUID id alone is not sufficient for hypertable partitioning
ALTER TABLE ghg_scope12.audit_trail_entries
    ADD PRIMARY KEY (id, performed_at);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable
-- ---------------------------------------------------------------------------
-- Convert to hypertable partitioned by performed_at with monthly chunks
-- Wrapped in DO block to handle cases where TimescaleDB is not installed
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'ghg_scope12.audit_trail_entries',
            'performed_at',
            chunk_time_interval => INTERVAL '1 month',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'TimescaleDB hypertable created for audit_trail_entries';
    ELSE
        RAISE NOTICE 'TimescaleDB not available - audit_trail_entries created as regular table';
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_at_tenant             ON ghg_scope12.audit_trail_entries(tenant_id, performed_at DESC);
CREATE INDEX idx_p041_at_entity_type        ON ghg_scope12.audit_trail_entries(entity_type, performed_at DESC);
CREATE INDEX idx_p041_at_entity_id          ON ghg_scope12.audit_trail_entries(entity_id, performed_at DESC);
CREATE INDEX idx_p041_at_action             ON ghg_scope12.audit_trail_entries(action, performed_at DESC);
CREATE INDEX idx_p041_at_change_cat         ON ghg_scope12.audit_trail_entries(change_category, performed_at DESC);
CREATE INDEX idx_p041_at_org               ON ghg_scope12.audit_trail_entries(organization_id, performed_at DESC);
CREATE INDEX idx_p041_at_facility           ON ghg_scope12.audit_trail_entries(facility_id, performed_at DESC);
CREATE INDEX idx_p041_at_performed_by       ON ghg_scope12.audit_trail_entries(performed_by, performed_at DESC);
CREATE INDEX idx_p041_at_year              ON ghg_scope12.audit_trail_entries(reporting_year, performed_at DESC);
CREATE INDEX idx_p041_at_scope             ON ghg_scope12.audit_trail_entries(scope, performed_at DESC);
CREATE INDEX idx_p041_at_source_agent       ON ghg_scope12.audit_trail_entries(source_agent, performed_at DESC);
CREATE INDEX idx_p041_at_provenance         ON ghg_scope12.audit_trail_entries(provenance_hash);
CREATE INDEX idx_p041_at_parent             ON ghg_scope12.audit_trail_entries(parent_hash);
CREATE INDEX idx_p041_at_old_value          ON ghg_scope12.audit_trail_entries USING GIN(old_value);
CREATE INDEX idx_p041_at_new_value          ON ghg_scope12.audit_trail_entries USING GIN(new_value);
CREATE INDEX idx_p041_at_fields             ON ghg_scope12.audit_trail_entries USING GIN(changed_fields);

-- Composite: tenant + entity + time for entity history
CREATE INDEX idx_p041_at_tenant_entity      ON ghg_scope12.audit_trail_entries(tenant_id, entity_type, entity_id, performed_at DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.report_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.verification_packages ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.audit_trail_entries ENABLE ROW LEVEL SECURITY;

CREATE POLICY p041_rm_tenant_isolation
    ON ghg_scope12.report_metadata
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_rm_service_bypass
    ON ghg_scope12.report_metadata
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_vp_tenant_isolation
    ON ghg_scope12.verification_packages
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_vp_service_bypass
    ON ghg_scope12.verification_packages
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_at_tenant_isolation
    ON ghg_scope12.audit_trail_entries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_at_service_bypass
    ON ghg_scope12.audit_trail_entries
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.report_metadata TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.verification_packages TO PUBLIC;
GRANT SELECT, INSERT ON ghg_scope12.audit_trail_entries TO PUBLIC;  -- No UPDATE/DELETE on audit trail
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.audit_trail_entries TO greenlang_service;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_scope12.report_metadata IS
    'Metadata for generated GHG reports (CDP, CSRD, SEC, custom) with format, generation parameters, file storage reference, and provenance hash.';
COMMENT ON TABLE ghg_scope12.verification_packages IS
    'Third-party verification engagement tracking with verifier details, standard (ISO 14064-3, ISAE 3410), assurance level, and verification statement.';
COMMENT ON TABLE ghg_scope12.audit_trail_entries IS
    'Append-only audit trail for all ghg_scope12 data changes. TimescaleDB hypertable partitioned by performed_at for efficient time-range queries.';

COMMENT ON COLUMN ghg_scope12.report_metadata.report_type IS 'Report type: CDP_QUESTIONNAIRE, CSRD_ESRS_E1, SEC_FILING, TCFD_REPORT, GHG_PROTOCOL_REPORT, etc.';
COMMENT ON COLUMN ghg_scope12.report_metadata.format IS 'Output format: PDF, HTML, XLSX, CSV, JSON, XML, XBRL, MARKDOWN, DOCX.';
COMMENT ON COLUMN ghg_scope12.report_metadata.provenance_hash IS 'SHA-256 hash of report content for integrity verification.';
COMMENT ON COLUMN ghg_scope12.report_metadata.file_checksum_sha256 IS 'SHA-256 checksum of the generated file for tamper detection.';

COMMENT ON COLUMN ghg_scope12.verification_packages.verification_standard IS 'Verification standard: ISO_14064_3, ISAE_3410, ISAE_3000, AA1000AS, CARB, EU_ETS_AVR.';
COMMENT ON COLUMN ghg_scope12.verification_packages.assurance_level IS 'Assurance level: LIMITED (negative assurance) or REASONABLE (positive assurance).';
COMMENT ON COLUMN ghg_scope12.verification_packages.verification_opinion IS 'Verifier opinion: UNQUALIFIED (clean), QUALIFIED (with exceptions), ADVERSE, DISCLAIMER.';
COMMENT ON COLUMN ghg_scope12.verification_packages.materiality_threshold_pct IS 'Materiality threshold as percentage of total emissions (typically 5%).';

COMMENT ON COLUMN ghg_scope12.audit_trail_entries.entity_type IS 'Table name of the modified entity (e.g., scope1_inventories, emission_factor_registry).';
COMMENT ON COLUMN ghg_scope12.audit_trail_entries.action IS 'Action type: CREATE, UPDATE, DELETE, APPROVE, REJECT, FINALIZE, RESTATE, RECALCULATE, etc.';
COMMENT ON COLUMN ghg_scope12.audit_trail_entries.provenance_hash IS 'SHA-256 hash of this audit entry for chain-of-custody integrity.';
COMMENT ON COLUMN ghg_scope12.audit_trail_entries.parent_hash IS 'Hash of the previous audit entry for the same entity (forms a hash chain).';

-- =============================================================================
-- V344: PACK-042 Scope 3 Starter Pack - Compliance Mapping & Reporting
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates compliance mapping, reporting, verification, and audit trail
-- tables for Scope 3 disclosures. Supports multi-framework compliance
-- assessment (GHG Protocol, CSRD, CDP, SBTi, TCFD, SEC), individual
-- requirement checking, gap analysis, report generation tracking,
-- verification package management, and a TimescaleDB-backed audit trail
-- for complete traceability of all Scope 3 data changes.
--
-- Tables (7):
--   1. ghg_accounting_scope3.compliance_assessments
--   2. ghg_accounting_scope3.framework_results
--   3. ghg_accounting_scope3.requirement_checks
--   4. ghg_accounting_scope3.compliance_gaps
--   5. ghg_accounting_scope3.report_metadata
--   6. ghg_accounting_scope3.verification_packages
--   7. ghg_accounting_scope3.scope3_audit_trail (hypertable)
--
-- Also includes: TimescaleDB hypertable, indexes, RLS, comments.
-- Previous: V343__pack042_uncertainty.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.compliance_assessments
-- =============================================================================
-- Top-level compliance assessment for a Scope 3 inventory. Evaluates the
-- inventory against multiple reporting frameworks simultaneously and
-- produces an overall compliance score.

CREATE TABLE ghg_accounting_scope3.compliance_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    -- Assessment details
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessment_version          INTEGER         NOT NULL DEFAULT 1,
    assessor                    VARCHAR(255),
    -- Overall results
    overall_compliance_score    DECIMAL(5,2)    NOT NULL DEFAULT 0,
    frameworks_assessed         INTEGER         NOT NULL DEFAULT 0,
    frameworks_compliant        INTEGER         NOT NULL DEFAULT 0,
    total_requirements          INTEGER         NOT NULL DEFAULT 0,
    met_requirements            INTEGER         NOT NULL DEFAULT 0,
    total_gaps                  INTEGER         NOT NULL DEFAULT 0,
    critical_gaps               INTEGER         NOT NULL DEFAULT 0,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_ca_score CHECK (
        overall_compliance_score >= 0 AND overall_compliance_score <= 100
    ),
    CONSTRAINT chk_p042_ca_frameworks CHECK (
        frameworks_assessed >= 0 AND frameworks_compliant >= 0 AND
        frameworks_compliant <= frameworks_assessed
    ),
    CONSTRAINT chk_p042_ca_requirements CHECK (
        total_requirements >= 0 AND met_requirements >= 0 AND
        met_requirements <= total_requirements
    ),
    CONSTRAINT chk_p042_ca_gaps CHECK (
        total_gaps >= 0 AND critical_gaps >= 0 AND
        critical_gaps <= total_gaps
    ),
    CONSTRAINT chk_p042_ca_version CHECK (
        assessment_version >= 1
    ),
    CONSTRAINT chk_p042_ca_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'REVIEWED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p042_ca_inventory_version UNIQUE (inventory_id, assessment_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ca_tenant             ON ghg_accounting_scope3.compliance_assessments(tenant_id);
CREATE INDEX idx_p042_ca_inventory          ON ghg_accounting_scope3.compliance_assessments(inventory_id);
CREATE INDEX idx_p042_ca_date               ON ghg_accounting_scope3.compliance_assessments(assessment_date DESC);
CREATE INDEX idx_p042_ca_score              ON ghg_accounting_scope3.compliance_assessments(overall_compliance_score);
CREATE INDEX idx_p042_ca_status             ON ghg_accounting_scope3.compliance_assessments(status);
CREATE INDEX idx_p042_ca_created            ON ghg_accounting_scope3.compliance_assessments(created_at DESC);

-- Composite: inventory + latest
CREATE INDEX idx_p042_ca_inv_latest         ON ghg_accounting_scope3.compliance_assessments(inventory_id, assessment_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ca_updated
    BEFORE UPDATE ON ghg_accounting_scope3.compliance_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.framework_results
-- =============================================================================
-- Per-framework compliance scores within an assessment. Each record
-- evaluates the inventory against a specific framework (GHG Protocol,
-- CSRD ESRS E1, CDP, SBTi, TCFD, SEC Climate).

CREATE TABLE ghg_accounting_scope3.framework_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3.compliance_assessments(id) ON DELETE CASCADE,
    -- Framework
    framework_type              VARCHAR(50)     NOT NULL,
    framework_version           VARCHAR(20),
    -- Scores
    compliance_score            DECIMAL(5,2)    NOT NULL DEFAULT 0,
    total_requirements          INTEGER         NOT NULL DEFAULT 0,
    met_requirements            INTEGER         NOT NULL DEFAULT 0,
    partially_met               INTEGER         DEFAULT 0,
    not_met                     INTEGER         DEFAULT 0,
    not_applicable              INTEGER         DEFAULT 0,
    gap_count                   INTEGER         NOT NULL DEFAULT 0,
    -- Classification
    compliance_level            VARCHAR(30)     NOT NULL DEFAULT 'PARTIAL',
    disclosure_readiness_pct    DECIMAL(5,2)    DEFAULT 0,
    -- Timeline
    reporting_deadline          DATE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_fr_framework CHECK (
        framework_type IN (
            'GHG_PROTOCOL_SCOPE3', 'CSRD_ESRS_E1', 'CDP_CLIMATE',
            'SBTI', 'TCFD', 'SEC_CLIMATE', 'ISO_14064_1',
            'CDP_SUPPLY_CHAIN', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p042_fr_score CHECK (
        compliance_score >= 0 AND compliance_score <= 100
    ),
    CONSTRAINT chk_p042_fr_requirements CHECK (
        total_requirements >= 0 AND met_requirements >= 0 AND
        met_requirements <= total_requirements
    ),
    CONSTRAINT chk_p042_fr_partial CHECK (
        partially_met IS NULL OR partially_met >= 0
    ),
    CONSTRAINT chk_p042_fr_not_met CHECK (
        not_met IS NULL OR not_met >= 0
    ),
    CONSTRAINT chk_p042_fr_na CHECK (
        not_applicable IS NULL OR not_applicable >= 0
    ),
    CONSTRAINT chk_p042_fr_gaps CHECK (
        gap_count >= 0
    ),
    CONSTRAINT chk_p042_fr_level CHECK (
        compliance_level IN (
            'FULLY_COMPLIANT', 'SUBSTANTIALLY_COMPLIANT',
            'PARTIAL', 'MINIMAL', 'NON_COMPLIANT'
        )
    ),
    CONSTRAINT chk_p042_fr_readiness CHECK (
        disclosure_readiness_pct IS NULL OR (disclosure_readiness_pct >= 0 AND disclosure_readiness_pct <= 100)
    ),
    CONSTRAINT uq_p042_fr_assessment_framework UNIQUE (assessment_id, framework_type)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_fr_tenant             ON ghg_accounting_scope3.framework_results(tenant_id);
CREATE INDEX idx_p042_fr_assessment         ON ghg_accounting_scope3.framework_results(assessment_id);
CREATE INDEX idx_p042_fr_framework          ON ghg_accounting_scope3.framework_results(framework_type);
CREATE INDEX idx_p042_fr_score              ON ghg_accounting_scope3.framework_results(compliance_score);
CREATE INDEX idx_p042_fr_level              ON ghg_accounting_scope3.framework_results(compliance_level);
CREATE INDEX idx_p042_fr_deadline           ON ghg_accounting_scope3.framework_results(reporting_deadline);
CREATE INDEX idx_p042_fr_created            ON ghg_accounting_scope3.framework_results(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_fr_updated
    BEFORE UPDATE ON ghg_accounting_scope3.framework_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.requirement_checks
-- =============================================================================
-- Individual requirement-level results within a framework assessment.
-- Each record evaluates one specific disclosure requirement.

CREATE TABLE ghg_accounting_scope3.requirement_checks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    framework_result_id         UUID            NOT NULL REFERENCES ghg_accounting_scope3.framework_results(id) ON DELETE CASCADE,
    -- Requirement
    requirement_id              VARCHAR(50)     NOT NULL,
    requirement_description     TEXT            NOT NULL,
    requirement_section         VARCHAR(100),
    requirement_type            VARCHAR(20)     NOT NULL DEFAULT 'MANDATORY',
    -- Result
    status                      VARCHAR(20)     NOT NULL DEFAULT 'NOT_ASSESSED',
    evidence_ref                TEXT,
    evidence_quality            VARCHAR(20),
    -- Notes
    notes                       TEXT,
    remediation_suggestion      TEXT,
    -- Metadata
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_rc_type CHECK (
        requirement_type IN ('MANDATORY', 'RECOMMENDED', 'OPTIONAL', 'CONDITIONAL')
    ),
    CONSTRAINT chk_p042_rc_status CHECK (
        status IN ('MET', 'PARTIALLY_MET', 'NOT_MET', 'NOT_APPLICABLE', 'NOT_ASSESSED')
    ),
    CONSTRAINT chk_p042_rc_evidence_quality CHECK (
        evidence_quality IS NULL OR evidence_quality IN (
            'STRONG', 'MODERATE', 'WEAK', 'MISSING'
        )
    ),
    CONSTRAINT uq_p042_rc_result_req UNIQUE (framework_result_id, requirement_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_rc_tenant             ON ghg_accounting_scope3.requirement_checks(tenant_id);
CREATE INDEX idx_p042_rc_result             ON ghg_accounting_scope3.requirement_checks(framework_result_id);
CREATE INDEX idx_p042_rc_req_id             ON ghg_accounting_scope3.requirement_checks(requirement_id);
CREATE INDEX idx_p042_rc_status             ON ghg_accounting_scope3.requirement_checks(status);
CREATE INDEX idx_p042_rc_type               ON ghg_accounting_scope3.requirement_checks(requirement_type);
CREATE INDEX idx_p042_rc_created            ON ghg_accounting_scope3.requirement_checks(created_at DESC);

-- Composite: result + unmet for gap identification
CREATE INDEX idx_p042_rc_result_unmet       ON ghg_accounting_scope3.requirement_checks(framework_result_id, requirement_type)
    WHERE status IN ('NOT_MET', 'PARTIALLY_MET');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_rc_updated
    BEFORE UPDATE ON ghg_accounting_scope3.requirement_checks
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.compliance_gaps
-- =============================================================================
-- Detailed gap records for unmet or partially met requirements.
-- Includes gap description, priority, estimated effort, and action plan.

CREATE TABLE ghg_accounting_scope3.compliance_gaps (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    framework_result_id         UUID            NOT NULL REFERENCES ghg_accounting_scope3.framework_results(id) ON DELETE CASCADE,
    requirement_id              VARCHAR(50)     NOT NULL,
    -- Gap details
    gap_description             TEXT            NOT NULL,
    gap_severity                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    -- Prioritization
    priority                    INTEGER         NOT NULL DEFAULT 3,
    effort_estimate             VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    -- Action plan
    action_plan                 TEXT,
    assigned_to                 VARCHAR(255),
    target_date                 DATE,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'OPEN',
    resolved_at                 TIMESTAMPTZ,
    resolved_by                 VARCHAR(255),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_cg_severity CHECK (
        gap_severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFORMATIONAL')
    ),
    CONSTRAINT chk_p042_cg_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p042_cg_effort CHECK (
        effort_estimate IN ('VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p042_cg_status CHECK (
        status IN ('OPEN', 'IN_PROGRESS', 'RESOLVED', 'ACCEPTED', 'DEFERRED', 'WONT_FIX')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_cg_tenant             ON ghg_accounting_scope3.compliance_gaps(tenant_id);
CREATE INDEX idx_p042_cg_result             ON ghg_accounting_scope3.compliance_gaps(framework_result_id);
CREATE INDEX idx_p042_cg_req_id             ON ghg_accounting_scope3.compliance_gaps(requirement_id);
CREATE INDEX idx_p042_cg_severity           ON ghg_accounting_scope3.compliance_gaps(gap_severity);
CREATE INDEX idx_p042_cg_priority           ON ghg_accounting_scope3.compliance_gaps(priority);
CREATE INDEX idx_p042_cg_status             ON ghg_accounting_scope3.compliance_gaps(status);
CREATE INDEX idx_p042_cg_target_date        ON ghg_accounting_scope3.compliance_gaps(target_date);
CREATE INDEX idx_p042_cg_created            ON ghg_accounting_scope3.compliance_gaps(created_at DESC);

-- Composite: open gaps by severity and priority
CREATE INDEX idx_p042_cg_open_priority      ON ghg_accounting_scope3.compliance_gaps(gap_severity, priority, target_date)
    WHERE status IN ('OPEN', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_cg_updated
    BEFORE UPDATE ON ghg_accounting_scope3.compliance_gaps
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_accounting_scope3.report_metadata
-- =============================================================================
-- Metadata for generated Scope 3 reports. Tracks report type, format,
-- generation parameters, file storage reference, and provenance hash.

CREATE TABLE ghg_accounting_scope3.report_metadata (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    -- Report details
    report_type                 VARCHAR(50)     NOT NULL,
    report_name                 VARCHAR(500)    NOT NULL,
    format                      VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    reporting_year              INTEGER         NOT NULL,
    -- Generation
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    generated_by                VARCHAR(255),
    generation_duration_ms      INTEGER,
    generation_parameters       JSONB           DEFAULT '{}',
    -- File
    file_path                   TEXT,
    file_name                   VARCHAR(500),
    file_size_bytes             BIGINT,
    file_checksum_sha256        VARCHAR(64),
    -- Version
    version                     INTEGER         NOT NULL DEFAULT 1,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'GENERATED',
    -- Provenance
    provenance_hash             VARCHAR(64)     NOT NULL,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_rm_type CHECK (
        report_type IN (
            'SCOPE3_INVENTORY', 'HOTSPOT_ANALYSIS', 'SUPPLIER_ENGAGEMENT',
            'DATA_QUALITY', 'UNCERTAINTY', 'COMPLIANCE', 'EXECUTIVE_SUMMARY',
            'CDP_RESPONSE', 'CSRD_DISCLOSURE', 'VERIFICATION_READY',
            'YEAR_OVER_YEAR', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p042_rm_format CHECK (
        format IN ('PDF', 'HTML', 'XLSX', 'CSV', 'JSON', 'XML', 'XBRL', 'MARKDOWN', 'DOCX')
    ),
    CONSTRAINT chk_p042_rm_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p042_rm_version CHECK (version >= 1),
    CONSTRAINT chk_p042_rm_file_size CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),
    CONSTRAINT chk_p042_rm_duration CHECK (generation_duration_ms IS NULL OR generation_duration_ms >= 0),
    CONSTRAINT chk_p042_rm_status CHECK (
        status IN ('GENERATING', 'GENERATED', 'REVIEWED', 'APPROVED', 'PUBLISHED', 'ARCHIVED', 'ERROR')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_rm_tenant             ON ghg_accounting_scope3.report_metadata(tenant_id);
CREATE INDEX idx_p042_rm_inventory          ON ghg_accounting_scope3.report_metadata(inventory_id);
CREATE INDEX idx_p042_rm_type               ON ghg_accounting_scope3.report_metadata(report_type);
CREATE INDEX idx_p042_rm_format             ON ghg_accounting_scope3.report_metadata(format);
CREATE INDEX idx_p042_rm_year               ON ghg_accounting_scope3.report_metadata(reporting_year);
CREATE INDEX idx_p042_rm_status             ON ghg_accounting_scope3.report_metadata(status);
CREATE INDEX idx_p042_rm_current            ON ghg_accounting_scope3.report_metadata(is_current) WHERE is_current = true;
CREATE INDEX idx_p042_rm_generated          ON ghg_accounting_scope3.report_metadata(generated_at DESC);
CREATE INDEX idx_p042_rm_created            ON ghg_accounting_scope3.report_metadata(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_rm_updated
    BEFORE UPDATE ON ghg_accounting_scope3.report_metadata
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 6: ghg_accounting_scope3.verification_packages
-- =============================================================================
-- Verification bundle for third-party assurance of Scope 3 inventory.

CREATE TABLE ghg_accounting_scope3.verification_packages (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    -- Details
    package_date                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    verifier_name               VARCHAR(255),
    verifier_organization       VARCHAR(255),
    assurance_level             VARCHAR(20)     NOT NULL DEFAULT 'LIMITED',
    verification_standard       VARCHAR(50)     DEFAULT 'ISO_14064_3',
    -- Scope
    categories_in_scope         ghg_accounting_scope3.scope3_category_type[],
    total_tco2e_in_scope        DECIMAL(15,3),
    materiality_threshold_pct   DECIMAL(5,2)    DEFAULT 5.00,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PREPARATION',
    verification_opinion        VARCHAR(30),
    findings_count              INTEGER         DEFAULT 0,
    -- Documents
    statement_path              TEXT,
    report_path                 TEXT,
    evidence_path               TEXT,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_vp_assurance CHECK (
        assurance_level IN ('LIMITED', 'REASONABLE', 'COMBINED')
    ),
    CONSTRAINT chk_p042_vp_standard CHECK (
        verification_standard IS NULL OR verification_standard IN (
            'ISO_14064_3', 'ISAE_3410', 'ISAE_3000', 'AA1000AS', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p042_vp_opinion CHECK (
        verification_opinion IS NULL OR verification_opinion IN (
            'UNQUALIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER'
        )
    ),
    CONSTRAINT chk_p042_vp_status CHECK (
        status IN (
            'PREPARATION', 'ENGAGEMENT', 'IN_PROGRESS', 'DRAFT_STATEMENT',
            'FINAL_STATEMENT', 'COMPLETED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p042_vp_findings CHECK (
        findings_count IS NULL OR findings_count >= 0
    ),
    CONSTRAINT chk_p042_vp_materiality CHECK (
        materiality_threshold_pct IS NULL OR (materiality_threshold_pct > 0 AND materiality_threshold_pct <= 20)
    ),
    CONSTRAINT chk_p042_vp_tco2e CHECK (
        total_tco2e_in_scope IS NULL OR total_tco2e_in_scope >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_vp_tenant             ON ghg_accounting_scope3.verification_packages(tenant_id);
CREATE INDEX idx_p042_vp_inventory          ON ghg_accounting_scope3.verification_packages(inventory_id);
CREATE INDEX idx_p042_vp_date               ON ghg_accounting_scope3.verification_packages(package_date DESC);
CREATE INDEX idx_p042_vp_status             ON ghg_accounting_scope3.verification_packages(status);
CREATE INDEX idx_p042_vp_verifier           ON ghg_accounting_scope3.verification_packages(verifier_name);
CREATE INDEX idx_p042_vp_assurance          ON ghg_accounting_scope3.verification_packages(assurance_level);
CREATE INDEX idx_p042_vp_opinion            ON ghg_accounting_scope3.verification_packages(verification_opinion);
CREATE INDEX idx_p042_vp_created            ON ghg_accounting_scope3.verification_packages(created_at DESC);
CREATE INDEX idx_p042_vp_categories         ON ghg_accounting_scope3.verification_packages USING GIN(categories_in_scope);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_vp_updated
    BEFORE UPDATE ON ghg_accounting_scope3.verification_packages
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 7: ghg_accounting_scope3.scope3_audit_trail
-- =============================================================================
-- Append-only audit trail for all data changes across the ghg_accounting_scope3
-- schema. Configured as a TimescaleDB hypertable for efficient time-range
-- queries. Records every INSERT, UPDATE, DELETE, and workflow action.

CREATE TABLE ghg_accounting_scope3.scope3_audit_trail (
    id                          UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- What changed
    event_type                  VARCHAR(30)     NOT NULL,
    entity_type                 VARCHAR(100)    NOT NULL,
    entity_id                   UUID            NOT NULL,
    -- Context
    inventory_id                UUID,
    category                    ghg_accounting_scope3.scope3_category_type,
    -- Change data
    event_data                  JSONB           NOT NULL DEFAULT '{}',
    old_values                  JSONB,
    new_values                  JSONB,
    changed_fields              TEXT[],
    change_summary              TEXT,
    -- Who/when
    user_id                     UUID,
    user_name                   VARCHAR(255),
    user_role                   VARCHAR(50),
    timestamp                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- System context
    source_agent                VARCHAR(100),
    source_system               VARCHAR(100)    DEFAULT 'GREENLANG',
    session_id                  VARCHAR(100),
    ip_address                  VARCHAR(45),
    -- Provenance
    provenance_hash             VARCHAR(64)     NOT NULL,
    parent_hash                 VARCHAR(64),
    -- Constraints
    CONSTRAINT chk_p042_at_event_type CHECK (
        event_type IN (
            'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'RECALCULATE',
            'CLASSIFY', 'SCREEN', 'ASSESS', 'VERIFY', 'APPROVE',
            'REJECT', 'SUBMIT', 'PUBLISH', 'ARCHIVE', 'IMPORT',
            'EXPORT', 'RECONCILE', 'ENGAGE', 'RESPOND', 'SYSTEM'
        )
    )
);

-- Set timestamp as primary key component for hypertable partitioning
ALTER TABLE ghg_accounting_scope3.scope3_audit_trail
    ADD PRIMARY KEY (id, timestamp);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'ghg_accounting_scope3.scope3_audit_trail',
            'timestamp',
            chunk_time_interval => INTERVAL '1 month',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'TimescaleDB hypertable created for scope3_audit_trail';
    ELSE
        RAISE NOTICE 'TimescaleDB not available - scope3_audit_trail created as regular table';
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_at_tenant             ON ghg_accounting_scope3.scope3_audit_trail(tenant_id, timestamp DESC);
CREATE INDEX idx_p042_at_event_type         ON ghg_accounting_scope3.scope3_audit_trail(event_type, timestamp DESC);
CREATE INDEX idx_p042_at_entity_type        ON ghg_accounting_scope3.scope3_audit_trail(entity_type, timestamp DESC);
CREATE INDEX idx_p042_at_entity_id          ON ghg_accounting_scope3.scope3_audit_trail(entity_id, timestamp DESC);
CREATE INDEX idx_p042_at_inventory          ON ghg_accounting_scope3.scope3_audit_trail(inventory_id, timestamp DESC);
CREATE INDEX idx_p042_at_category           ON ghg_accounting_scope3.scope3_audit_trail(category, timestamp DESC);
CREATE INDEX idx_p042_at_user_id            ON ghg_accounting_scope3.scope3_audit_trail(user_id, timestamp DESC);
CREATE INDEX idx_p042_at_source_agent       ON ghg_accounting_scope3.scope3_audit_trail(source_agent, timestamp DESC);
CREATE INDEX idx_p042_at_provenance         ON ghg_accounting_scope3.scope3_audit_trail(provenance_hash);
CREATE INDEX idx_p042_at_parent             ON ghg_accounting_scope3.scope3_audit_trail(parent_hash);
CREATE INDEX idx_p042_at_event_data         ON ghg_accounting_scope3.scope3_audit_trail USING GIN(event_data);
CREATE INDEX idx_p042_at_changed_fields     ON ghg_accounting_scope3.scope3_audit_trail USING GIN(changed_fields);

-- Composite: tenant + entity + time for entity history
CREATE INDEX idx_p042_at_tenant_entity      ON ghg_accounting_scope3.scope3_audit_trail(tenant_id, entity_type, entity_id, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3.compliance_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.framework_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.requirement_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.compliance_gaps ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.report_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.verification_packages ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.scope3_audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p042_ca_tenant_isolation ON ghg_accounting_scope3.compliance_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_ca_service_bypass ON ghg_accounting_scope3.compliance_assessments
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_fr_tenant_isolation ON ghg_accounting_scope3.framework_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_fr_service_bypass ON ghg_accounting_scope3.framework_results
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_rc_tenant_isolation ON ghg_accounting_scope3.requirement_checks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_rc_service_bypass ON ghg_accounting_scope3.requirement_checks
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_cg_tenant_isolation ON ghg_accounting_scope3.compliance_gaps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_cg_service_bypass ON ghg_accounting_scope3.compliance_gaps
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_rm_tenant_isolation ON ghg_accounting_scope3.report_metadata
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_rm_service_bypass ON ghg_accounting_scope3.report_metadata
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_vp_tenant_isolation ON ghg_accounting_scope3.verification_packages
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_vp_service_bypass ON ghg_accounting_scope3.verification_packages
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_at_tenant_isolation ON ghg_accounting_scope3.scope3_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_at_service_bypass ON ghg_accounting_scope3.scope3_audit_trail
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.compliance_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.framework_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.requirement_checks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.compliance_gaps TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.report_metadata TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.verification_packages TO PUBLIC;
GRANT SELECT, INSERT ON ghg_accounting_scope3.scope3_audit_trail TO PUBLIC;  -- No UPDATE/DELETE on audit trail
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.scope3_audit_trail TO greenlang_service;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3.compliance_assessments IS
    'Top-level compliance assessment evaluating a Scope 3 inventory against multiple reporting frameworks with overall compliance score.';
COMMENT ON TABLE ghg_accounting_scope3.framework_results IS
    'Per-framework compliance scores (GHG Protocol Scope 3, CSRD, CDP, SBTi, TCFD, SEC) with requirement counts and disclosure readiness.';
COMMENT ON TABLE ghg_accounting_scope3.requirement_checks IS
    'Individual requirement-level compliance results with status (MET, PARTIALLY_MET, NOT_MET) and evidence references.';
COMMENT ON TABLE ghg_accounting_scope3.compliance_gaps IS
    'Detailed gap records for unmet requirements with severity, priority, effort estimate, and remediation action plan.';
COMMENT ON TABLE ghg_accounting_scope3.report_metadata IS
    'Metadata for generated Scope 3 reports (inventory, hotspot, compliance, etc.) with file storage and provenance tracking.';
COMMENT ON TABLE ghg_accounting_scope3.verification_packages IS
    'Third-party verification engagement packages with verifier details, assurance level, scope, and verification opinion.';
COMMENT ON TABLE ghg_accounting_scope3.scope3_audit_trail IS
    'Append-only audit trail (TimescaleDB hypertable) for all ghg_accounting_scope3 data changes with provenance hash chain.';

COMMENT ON COLUMN ghg_accounting_scope3.scope3_audit_trail.event_type IS 'Action type: CREATE, UPDATE, DELETE, CALCULATE, CLASSIFY, SCREEN, ASSESS, VERIFY, APPROVE, etc.';
COMMENT ON COLUMN ghg_accounting_scope3.scope3_audit_trail.provenance_hash IS 'SHA-256 hash of this audit entry for chain-of-custody integrity.';
COMMENT ON COLUMN ghg_accounting_scope3.scope3_audit_trail.parent_hash IS 'Hash of the previous audit entry for the same entity (forms a hash chain).';

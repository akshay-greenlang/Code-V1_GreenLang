-- =============================================================================
-- V145: PACK-024-carbon-neutral-008: Verification Packages
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for comprehensive verification packages supporting
-- carbon neutral substantiation with documentation bundles, audit evidence,
-- and verification readiness tracking.
--
-- EXTENDS:
--   V144: Claims Substantiation
--
-- These tables organize verification evidence into coherent packages
-- for streamlined third-party audits and regulatory inspections.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_verification_packages         - Package bundles
--   2. pack024_carbon_neutral.pack024_package_documentation         - Package contents
--   3. pack024_carbon_neutral.pack024_package_review_findings       - Review results
--   4. pack024_carbon_neutral.pack024_package_audit_trail           - Audit tracking
--
-- Also includes: 50+ indexes, update triggers, security grants, and comments.
-- Previous: V144__pack024_carbon_neutral_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_verification_packages
-- =============================================================================
-- Verification package bundles with readiness and submission status.

CREATE TABLE pack024_carbon_neutral.pack024_verification_packages (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    package_name            VARCHAR(500)    NOT NULL,
    package_code            VARCHAR(50),
    package_version         VARCHAR(20)     DEFAULT '1.0',
    package_type            VARCHAR(100)    NOT NULL,
    package_scope           VARCHAR(255),
    coverage_period_start   DATE            NOT NULL,
    coverage_period_end     DATE            NOT NULL,
    creation_date           DATE            NOT NULL,
    completion_date         DATE,
    intended_use            VARCHAR(255),
    target_verifier_type    VARCHAR(100),
    target_regulatory_body  VARCHAR(100),
    certification_target    VARCHAR(100),
    total_document_count    INTEGER         DEFAULT 0,
    critical_document_count INTEGER         DEFAULT 0,
    supporting_document_count INTEGER       DEFAULT 0,
    appendix_count          INTEGER         DEFAULT 0,
    total_page_estimate     INTEGER,
    organization_structure  VARCHAR(100),
    organization_approach   VARCHAR(50),
    executive_summary_included BOOLEAN      DEFAULT TRUE,
    table_of_contents_prepared BOOLEAN      DEFAULT TRUE,
    index_prepared          BOOLEAN         DEFAULT FALSE,
    cross_reference_guide   BOOLEAN         DEFAULT FALSE,
    package_completeness    DECIMAL(6,2),
    critical_docs_complete  BOOLEAN         DEFAULT FALSE,
    gap_analysis_performed  BOOLEAN         DEFAULT FALSE,
    identified_gaps         TEXT[],
    gap_remediation_plan    TEXT,
    gap_remediation_deadline DATE,
    quality_control_passed  BOOLEAN         DEFAULT FALSE,
    quality_check_date      DATE,
    quality_checker_name    VARCHAR(255),
    quality_issues_identified TEXT[],
    quality_issue_resolution TEXT,
    readiness_assessment    VARCHAR(30),
    readiness_score         DECIMAL(5,2),
    readiness_assessment_date DATE,
    assessed_by             VARCHAR(255),
    readiness_criteria      JSONB           DEFAULT '{}',
    verification_readiness  BOOLEAN         DEFAULT FALSE,
    internal_review_completed BOOLEAN       DEFAULT FALSE,
    internal_reviewer       VARCHAR(255),
    internal_review_date    DATE,
    internal_approval       BOOLEAN         DEFAULT FALSE,
    internal_approval_comments TEXT,
    external_review_requested BOOLEAN       DEFAULT FALSE,
    external_reviewer       VARCHAR(255),
    external_review_date    DATE,
    external_approval       BOOLEAN         DEFAULT FALSE,
    package_status          VARCHAR(30)     DEFAULT 'in_preparation',
    submission_requested    BOOLEAN         DEFAULT FALSE,
    submission_date         DATE,
    submission_status       VARCHAR(30),
    submission_reference    VARCHAR(100),
    submission_method       VARCHAR(50),
    receiver_name           VARCHAR(255),
    receiver_organization   VARCHAR(255),
    delivery_confirmation   BOOLEAN         DEFAULT FALSE,
    delivery_date           DATE,
    package_access_controlled BOOLEAN       DEFAULT TRUE,
    access_controls         VARCHAR(100),
    confidentiality_mark    VARCHAR(50),
    retention_requirement   VARCHAR(100),
    archival_plan           TEXT,
    version_history         JSONB           DEFAULT '{}',
    last_updated_date       DATE,
    prepared_by             VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_pkg_type CHECK (
        package_type IN ('EMISSIONS_SUBSTANTIATION', 'CREDIT_VERIFICATION', 'BASELINE_ESTABLISHMENT',
                        'METHODOLOGY_DOCUMENTATION', 'DATA_QUALITY_PACKAGE', 'GOVERNANCE_PACKAGE',
                        'COMPREHENSIVE_DOSSIER', 'AUDIT_PREPARATION', 'OTHER')
    ),
    CONSTRAINT chk_pack024_pkg_readiness CHECK (
        readiness_assessment IN ('NOT_READY', 'PARTIALLY_READY', 'READY', 'READY_WITH_CAVEATS')
    ),
    CONSTRAINT chk_pack024_pkg_status CHECK (
        package_status IN ('IN_PREPARATION', 'REVIEW', 'APPROVED', 'SUBMITTED', 'ACCEPTED', 'REJECTED')
    ),
    CONSTRAINT chk_pack024_pkg_completeness CHECK (
        package_completeness >= 0 AND package_completeness <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_pkg_org ON pack024_carbon_neutral.pack024_verification_packages(org_id);
CREATE INDEX idx_pack024_pkg_tenant ON pack024_carbon_neutral.pack024_verification_packages(tenant_id);
CREATE INDEX idx_pack024_pkg_creation_date ON pack024_carbon_neutral.pack024_verification_packages(creation_date DESC);
CREATE INDEX idx_pack024_pkg_type ON pack024_carbon_neutral.pack024_verification_packages(package_type);
CREATE INDEX idx_pack024_pkg_status ON pack024_carbon_neutral.pack024_verification_packages(package_status);
CREATE INDEX idx_pack024_pkg_readiness ON pack024_carbon_neutral.pack024_verification_packages(readiness_assessment);
CREATE INDEX idx_pack024_pkg_readiness_score ON pack024_carbon_neutral.pack024_verification_packages(readiness_score DESC);
CREATE INDEX idx_pack024_pkg_completeness ON pack024_carbon_neutral.pack024_verification_packages(package_completeness DESC);
CREATE INDEX idx_pack024_pkg_qc_passed ON pack024_carbon_neutral.pack024_verification_packages(quality_control_passed);
CREATE INDEX idx_pack024_pkg_verification_ready ON pack024_carbon_neutral.pack024_verification_packages(verification_readiness);
CREATE INDEX idx_pack024_pkg_internal_approval ON pack024_carbon_neutral.pack024_verification_packages(internal_approval);
CREATE INDEX idx_pack024_pkg_submission_date ON pack024_carbon_neutral.pack024_verification_packages(submission_date);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_pkg_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_verification_packages
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_package_documentation
-- =============================================================================
-- Individual documents within verification packages with status tracking.

CREATE TABLE pack024_carbon_neutral.pack024_package_documentation (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_package_id UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_verification_packages(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    document_title          VARCHAR(500)    NOT NULL,
    document_code           VARCHAR(50),
    document_type           VARCHAR(100)    NOT NULL,
    document_category       VARCHAR(100),
    document_subcategory    VARCHAR(100),
    document_description    TEXT,
    version_number          VARCHAR(20),
    version_date            DATE,
    prepared_by             VARCHAR(255),
    reviewed_by             VARCHAR(255),
    approved_by             VARCHAR(255),
    approval_date           DATE,
    document_status         VARCHAR(30)     DEFAULT 'draft',
    content_complete        BOOLEAN         DEFAULT FALSE,
    content_accuracy_verified BOOLEAN       DEFAULT FALSE,
    page_count              INTEGER,
    attachment_count        INTEGER,
    file_format             VARCHAR(50),
    file_size_mb            DECIMAL(10,2),
    file_path               VARCHAR(500),
    file_hash               VARCHAR(255),
    document_language       VARCHAR(10),
    document_confidentiality VARCHAR(50),
    required_for_verification BOOLEAN       DEFAULT TRUE,
    verification_relevance  DECIMAL(5,2),
    evidence_strength       VARCHAR(30),
    critical_importance     BOOLEAN         DEFAULT FALSE,
    gap_item_reference      VARCHAR(100),
    supports_claim          BOOLEAN         DEFAULT FALSE,
    claim_reference         VARCHAR(100),
    regulatory_requirement  VARCHAR(255),
    standard_requirement    VARCHAR(255),
    methodological_basis    VARCHAR(255),
    primary_document        BOOLEAN         DEFAULT FALSE,
    supporting_document_references VARCHAR(100)[],
    cross_references        VARCHAR(100)[],
    appendix_items          TEXT[],
    data_source             VARCHAR(255),
    data_reliability        VARCHAR(30),
    evidence_collection_date DATE,
    evidence_currency       VARCHAR(50),
    limitations_noted       TEXT,
    assumptions_documented  BOOLEAN         DEFAULT TRUE,
    calculation_verification BOOLEAN        DEFAULT FALSE,
    calculation_verified_by VARCHAR(255),
    calculation_verification_date DATE,
    document_review_notes   TEXT,
    acceptance_status       VARCHAR(30)     DEFAULT 'pending_review',
    acceptance_date         DATE,
    acceptance_comments     TEXT,
    revision_requests       TEXT[],
    revision_status         VARCHAR(30),
    resubmission_required   BOOLEAN         DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_doc_type CHECK (
        document_type IN ('EMISSIONS_REPORT', 'CALCULATION_SPREADSHEET', 'AUDIT_REPORT', 'VERIFICATION_STATEMENT',
                         'POLICY_DOCUMENT', 'PROCEDURE_DOCUMENTATION', 'DATA_QUALITY_ASSESSMENT', 'METHODOLOGY_DOCUMENT',
                         'ORGANIZATIONAL_CHART', 'BOARD_MINUTES', 'SUSTAINABILITY_REPORT', 'OTHER')
    ),
    CONSTRAINT chk_pack024_doc_status CHECK (
        document_status IN ('DRAFT', 'REVIEW', 'APPROVED', 'FINALIZED', 'ARCHIVED')
    ),
    CONSTRAINT chk_pack024_doc_relevance CHECK (
        verification_relevance >= 0 AND verification_relevance <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_doc_pkg_id ON pack024_carbon_neutral.pack024_package_documentation(verification_package_id);
CREATE INDEX idx_pack024_doc_org ON pack024_carbon_neutral.pack024_package_documentation(org_id);
CREATE INDEX idx_pack024_doc_tenant ON pack024_carbon_neutral.pack024_package_documentation(tenant_id);
CREATE INDEX idx_pack024_doc_title ON pack024_carbon_neutral.pack024_package_documentation(document_title);
CREATE INDEX idx_pack024_doc_type ON pack024_carbon_neutral.pack024_package_documentation(document_type);
CREATE INDEX idx_pack024_doc_category ON pack024_carbon_neutral.pack024_package_documentation(document_category);
CREATE INDEX idx_pack024_doc_status ON pack024_carbon_neutral.pack024_package_documentation(document_status);
CREATE INDEX idx_pack024_doc_acceptance ON pack024_carbon_neutral.pack024_package_documentation(acceptance_status);
CREATE INDEX idx_pack024_doc_required ON pack024_carbon_neutral.pack024_package_documentation(required_for_verification);
CREATE INDEX idx_pack024_doc_critical ON pack024_carbon_neutral.pack024_package_documentation(critical_importance);
CREATE INDEX idx_pack024_doc_primary ON pack024_carbon_neutral.pack024_package_documentation(primary_document);
CREATE INDEX idx_pack024_doc_relevance ON pack024_carbon_neutral.pack024_package_documentation(verification_relevance DESC);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_doc_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_package_documentation
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_package_review_findings
-- =============================================================================
-- Review findings and improvement recommendations for verification packages.

CREATE TABLE pack024_carbon_neutral.pack024_package_review_findings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_package_id UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_verification_packages(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    review_date             DATE            NOT NULL,
    finding_type            VARCHAR(50)     NOT NULL,
    finding_severity        VARCHAR(30)     NOT NULL,
    finding_category        VARCHAR(100),
    finding_description     TEXT            NOT NULL,
    affected_area           VARCHAR(255),
    affected_document_id    UUID,
    root_cause              TEXT,
    root_cause_analysis     TEXT,
    impact_assessment       TEXT,
    business_impact         VARCHAR(30),
    compliance_impact       VARCHAR(30),
    remediation_required    BOOLEAN         DEFAULT FALSE,
    remediation_action      TEXT,
    remediation_timeline    VARCHAR(100),
    remediation_deadline    DATE,
    remediation_owner       VARCHAR(255),
    remediation_status      VARCHAR(30),
    remediation_completion_date DATE,
    remediation_evidence    TEXT[],
    closure_approval        VARCHAR(255),
    closure_date            DATE,
    finding_criticality     VARCHAR(30),
    verification_impact     VARCHAR(30),
    acceptance_status       VARCHAR(30)     DEFAULT 'open',
    acceptance_date         DATE,
    accepted_by             VARCHAR(255),
    acceptance_rationale    TEXT,
    reviewer_name           VARCHAR(255),
    reviewer_organization   VARCHAR(255),
    review_phase            VARCHAR(50),
    related_findings        UUID[],
    preventive_measures     TEXT[],
    similar_finding_history VARCHAR(255),
    follow_up_required      BOOLEAN         DEFAULT FALSE,
    follow_up_date          DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_finding_type CHECK (
        finding_type IN ('OBSERVATION', 'EXCEPTION', 'DEFICIENCY', 'RECOMMENDATION', 'COMPLIANCE_ISSUE')
    ),
    CONSTRAINT chk_pack024_finding_severity CHECK (
        finding_severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_pack024_finding_acceptance CHECK (
        acceptance_status IN ('OPEN', 'ACCEPTED', 'REJECTED', 'CLOSED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_finding_pkg_id ON pack024_carbon_neutral.pack024_package_review_findings(verification_package_id);
CREATE INDEX idx_pack024_finding_org ON pack024_carbon_neutral.pack024_package_review_findings(org_id);
CREATE INDEX idx_pack024_finding_tenant ON pack024_carbon_neutral.pack024_package_review_findings(tenant_id);
CREATE INDEX idx_pack024_finding_date ON pack024_carbon_neutral.pack024_package_review_findings(review_date DESC);
CREATE INDEX idx_pack024_finding_type ON pack024_carbon_neutral.pack024_package_review_findings(finding_type);
CREATE INDEX idx_pack024_finding_severity ON pack024_carbon_neutral.pack024_package_review_findings(finding_severity);
CREATE INDEX idx_pack024_finding_category ON pack024_carbon_neutral.pack024_package_review_findings(finding_category);
CREATE INDEX idx_pack024_finding_status ON pack024_carbon_neutral.pack024_package_review_findings(acceptance_status);
CREATE INDEX idx_pack024_finding_remediation ON pack024_carbon_neutral.pack024_package_review_findings(remediation_status);
CREATE INDEX idx_pack024_finding_deadline ON pack024_carbon_neutral.pack024_package_review_findings(remediation_deadline);
CREATE INDEX idx_pack024_finding_criticality ON pack024_carbon_neutral.pack024_package_review_findings(finding_criticality);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_finding_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_package_review_findings
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_package_audit_trail
-- =============================================================================
-- Comprehensive audit trail for verification package activities.

CREATE TABLE pack024_carbon_neutral.pack024_package_audit_trail (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_package_id UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_verification_packages(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    event_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    event_type              VARCHAR(100)    NOT NULL,
    event_action            VARCHAR(255)    NOT NULL,
    event_description       TEXT,
    actor_name              VARCHAR(255)    NOT NULL,
    actor_role              VARCHAR(100),
    actor_department        VARCHAR(100),
    affected_entity_type    VARCHAR(100),
    affected_entity_id      VARCHAR(100),
    affected_entity_name    VARCHAR(255),
    old_value               TEXT,
    new_value               TEXT,
    change_rationale        TEXT,
    approval_status_change  VARCHAR(30),
    approval_by             VARCHAR(255),
    approval_date           TIMESTAMPTZ,
    approval_comments       TEXT,
    system_source           VARCHAR(100),
    ip_address              VARCHAR(45),
    session_id              VARCHAR(255),
    change_impact           VARCHAR(255),
    rollback_possible       BOOLEAN         DEFAULT FALSE,
    rollback_action         TEXT,
    notification_sent       BOOLEAN         DEFAULT FALSE,
    notification_recipients TEXT[],
    audit_status            VARCHAR(30),
    audit_notes             TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_audit_type CHECK (
        event_type IN ('CREATION', 'MODIFICATION', 'APPROVAL', 'REJECTION', 'SUBMISSION',
                      'REVIEW_START', 'REVIEW_COMPLETE', 'FINDING_ADDED', 'REMEDIATION_STATUS_CHANGE',
                      'DOCUMENT_ADDED', 'DOCUMENT_REMOVED', 'VERSION_UPDATE', 'ACCESS_GRANTED', 'DELETION')
    )
);

-- Indexes
CREATE INDEX idx_pack024_audit_pkg_id ON pack024_carbon_neutral.pack024_package_audit_trail(verification_package_id);
CREATE INDEX idx_pack024_audit_org ON pack024_carbon_neutral.pack024_package_audit_trail(org_id);
CREATE INDEX idx_pack024_audit_tenant ON pack024_carbon_neutral.pack024_package_audit_trail(tenant_id);
CREATE INDEX idx_pack024_audit_event_date ON pack024_carbon_neutral.pack024_package_audit_trail(event_date DESC);
CREATE INDEX idx_pack024_audit_event_type ON pack024_carbon_neutral.pack024_package_audit_trail(event_type);
CREATE INDEX idx_pack024_audit_actor ON pack024_carbon_neutral.pack024_package_audit_trail(actor_name);
CREATE INDEX idx_pack024_audit_affected_entity ON pack024_carbon_neutral.pack024_package_audit_trail(affected_entity_type, affected_entity_id);
CREATE INDEX idx_pack024_audit_approval_status ON pack024_carbon_neutral.pack024_package_audit_trail(approval_status_change);

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack024_carbon_neutral TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack024_carbon_neutral TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack024_carbon_neutral TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack024_carbon_neutral.pack024_verification_packages IS
'Verification package bundles with readiness assessment and submission status for carbon neutral substantiation.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_package_documentation IS
'Individual documents within verification packages with content completion, acceptance, and revision tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_package_review_findings IS
'Review findings and improvement recommendations for verification packages with remediation action tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_package_audit_trail IS
'Comprehensive audit trail for verification package activities including modifications, approvals, and access events.';

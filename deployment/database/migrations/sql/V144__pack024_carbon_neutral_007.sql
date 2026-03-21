-- =============================================================================
-- V144: PACK-024-carbon-neutral-007: Claims Substantiation
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for carbon neutral claims substantiation with evidence
-- tracking, verification requirements, disclosure documentation, and
-- integrity assurance for carbon neutral marketing and reporting claims.
--
-- EXTENDS:
--   V143: Neutralization Balance Reconciliation
--
-- These tables support substantiation of carbon neutral claims with
-- comprehensive evidence management and compliance verification.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_claim_substantiation         - Claims registry
--   2. pack024_carbon_neutral.pack024_claim_evidence               - Evidence documentation
--   3. pack024_carbon_neutral.pack024_claim_verification           - Verification audits
--   4. pack024_carbon_neutral.pack024_claim_disclosure             - Public disclosure
--
-- Also includes: 50+ indexes, update triggers, security grants, and comments.
-- Previous: V143__pack024_carbon_neutral_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_claim_substantiation
-- =============================================================================
-- Carbon neutral claims registry with substantiation requirements.

CREATE TABLE pack024_carbon_neutral.pack024_claim_substantiation (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    claim_date              DATE            NOT NULL,
    claim_type              VARCHAR(100)    NOT NULL,
    claim_statement         TEXT            NOT NULL,
    claim_scope             VARCHAR(100),
    claim_period_start      DATE,
    claim_period_end        DATE,
    covered_scopes          TEXT[],
    geographical_scope      TEXT[],
    product_service_scope   TEXT,
    claim_specificity       VARCHAR(50),
    claim_quantification    DECIMAL(18,4),
    quantification_unit     VARCHAR(30),
    claim_timeline          VARCHAR(100),
    claim_timeframe_type    VARCHAR(50),
    baseline_reference      VARCHAR(255),
    comparable_baseline     VARCHAR(255),
    third_party_claim       BOOLEAN         DEFAULT FALSE,
    third_party_name        VARCHAR(255),
    third_party_relationship VARCHAR(100),
    marketing_usage         BOOLEAN         DEFAULT FALSE,
    marketing_context       TEXT,
    regulatory_filing       BOOLEAN         DEFAULT FALSE,
    regulatory_body         VARCHAR(100),
    filing_date             DATE,
    stakeholder_communication BOOLEAN       DEFAULT FALSE,
    stakeholder_audience    TEXT[],
    claim_eligibility_met   BOOLEAN         DEFAULT TRUE,
    eligibility_criteria    JSONB           DEFAULT '{}',
    substantiation_basis    VARCHAR(255),
    substantiation_complete BOOLEAN         DEFAULT FALSE,
    substantiation_date     DATE,
    substantiation_methodology VARCHAR(255),
    supporting_evidence_count INTEGER,
    required_evidence_count INTEGER,
    evidence_gaps           TEXT[],
    gap_remediation_plan    TEXT,
    gap_remediation_deadline DATE,
    interim_substantiation  BOOLEAN         DEFAULT FALSE,
    interim_basis           TEXT,
    interim_completion_date DATE,
    full_substantiation_expected_date DATE,
    claim_confidence_level  DECIMAL(5,2),
    materiality_threshold   DECIMAL(6,2),
    materiality_reached     BOOLEAN         DEFAULT FALSE,
    claim_status            VARCHAR(30)     DEFAULT 'proposed',
    approval_status         VARCHAR(30)     DEFAULT 'pending',
    approved_by             VARCHAR(255),
    approval_date           DATE,
    approval_notes          TEXT,
    rejected_reason         TEXT,
    resubmission_plan       TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_claim_type CHECK (
        claim_type IN ('CARBON_NEUTRAL', 'NET_ZERO', 'CARBON_NEGATIVE', 'CLIMATE_POSITIVE',
                       'EMISSION_REDUCTION', 'OFFSET_RETIREMENT', 'OTHER')
    ),
    CONSTRAINT chk_pack024_claim_status CHECK (
        claim_status IN ('PROPOSED', 'UNDER_SUBSTANTIATION', 'SUBSTANTIATED', 'REJECTED', 'WITHDRAWN')
    ),
    CONSTRAINT chk_pack024_claim_approval CHECK (
        approval_status IN ('PENDING', 'APPROVED', 'REJECTED', 'CONDITIONAL')
    )
);

-- Indexes
CREATE INDEX idx_pack024_claim_org ON pack024_carbon_neutral.pack024_claim_substantiation(org_id);
CREATE INDEX idx_pack024_claim_tenant ON pack024_carbon_neutral.pack024_claim_substantiation(tenant_id);
CREATE INDEX idx_pack024_claim_date ON pack024_carbon_neutral.pack024_claim_substantiation(claim_date DESC);
CREATE INDEX idx_pack024_claim_type ON pack024_carbon_neutral.pack024_claim_substantiation(claim_type);
CREATE INDEX idx_pack024_claim_status ON pack024_carbon_neutral.pack024_claim_substantiation(claim_status);
CREATE INDEX idx_pack024_claim_approval ON pack024_carbon_neutral.pack024_claim_substantiation(approval_status);
CREATE INDEX idx_pack024_claim_scope ON pack024_carbon_neutral.pack024_claim_substantiation(claim_scope);
CREATE INDEX idx_pack024_claim_substantiation_complete ON pack024_carbon_neutral.pack024_claim_substantiation(substantiation_complete);
CREATE INDEX idx_pack024_claim_marketing ON pack024_carbon_neutral.pack024_claim_substantiation(marketing_usage);
CREATE INDEX idx_pack024_claim_confidence ON pack024_carbon_neutral.pack024_claim_substantiation(claim_confidence_level DESC);
CREATE INDEX idx_pack024_claim_evidence_gaps ON pack024_carbon_neutral.pack024_claim_substantiation(evidence_gaps);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_claim_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_claim_substantiation
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_claim_evidence
-- =============================================================================
-- Evidence documentation for claim substantiation.

CREATE TABLE pack024_carbon_neutral.pack024_claim_evidence (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_substantiation_id UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_claim_substantiation(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    evidence_date           DATE            NOT NULL,
    evidence_type           VARCHAR(100)    NOT NULL,
    evidence_category       VARCHAR(100),
    evidence_title          VARCHAR(500),
    evidence_description    TEXT,
    evidence_content        TEXT,
    supporting_documents    TEXT[],
    document_references     VARCHAR(255)[],
    file_path               VARCHAR(500),
    file_format             VARCHAR(50),
    file_hash               VARCHAR(255),
    data_source             VARCHAR(255),
    data_source_credibility VARCHAR(30),
    data_collection_date    DATE,
    data_collection_method  VARCHAR(255),
    data_quality_assessment VARCHAR(30),
    relevance_to_claim      DECIMAL(5,2),
    relevance_explanation   TEXT,
    completeness_assessment DECIMAL(6,2),
    geographic_coverage     TEXT[],
    temporal_coverage_start DATE,
    temporal_coverage_end   DATE,
    temporal_relevance      VARCHAR(50),
    third_party_validation  BOOLEAN         DEFAULT FALSE,
    validation_source       VARCHAR(255),
    validation_date         DATE,
    validator_credentials   VARCHAR(255),
    evidence_chain_of_custody BOOLEAN       DEFAULT FALSE,
    chain_custodian_records TEXT[],
    authenticity_verified   BOOLEAN         DEFAULT FALSE,
    authenticity_method     VARCHAR(100),
    evidence_integrity_check BOOLEAN        DEFAULT FALSE,
    integrity_check_result  VARCHAR(50),
    evidence_status         VARCHAR(30)     DEFAULT 'submitted',
    acceptance_status       VARCHAR(30)     DEFAULT 'pending_review',
    acceptance_date         DATE,
    accepted_by             VARCHAR(255),
    acceptance_notes        TEXT,
    rejection_reason        TEXT,
    resubmission_plan       TEXT,
    evidence_weight         DECIMAL(5,2),
    evidence_criticality    VARCHAR(30),
    compliance_reference    TEXT[],
    precedent_references    VARCHAR(255)[],
    conflicting_evidence    UUID[],
    conflict_resolution     TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_evidence_type CHECK (
        evidence_type IN ('EMISSIONS_INVENTORY', 'CREDIT_RETIREMENT', 'AUDIT_REPORT', 'CALCULATION_METHODOLOGY',
                         'DATA_QUALITY_REPORT', 'VERIFICATION_STATEMENT', 'OFFSET_DOCUMENTATION', 'POLICY_DOCUMENT',
                         'BASELINE_ESTABLISHMENT', 'ASSUMPTIONS_DOCUMENTATION', 'OTHER')
    ),
    CONSTRAINT chk_pack024_evidence_status CHECK (
        evidence_status IN ('SUBMITTED', 'ACCEPTED', 'REJECTED', 'REVISED', 'WITHDRAWN')
    ),
    CONSTRAINT chk_pack024_evidence_relevance CHECK (
        relevance_to_claim >= 0 AND relevance_to_claim <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_evid_claim_id ON pack024_carbon_neutral.pack024_claim_evidence(claim_substantiation_id);
CREATE INDEX idx_pack024_evid_org ON pack024_carbon_neutral.pack024_claim_evidence(org_id);
CREATE INDEX idx_pack024_evid_tenant ON pack024_carbon_neutral.pack024_claim_evidence(tenant_id);
CREATE INDEX idx_pack024_evid_date ON pack024_carbon_neutral.pack024_claim_evidence(evidence_date DESC);
CREATE INDEX idx_pack024_evid_type ON pack024_carbon_neutral.pack024_claim_evidence(evidence_type);
CREATE INDEX idx_pack024_evid_category ON pack024_carbon_neutral.pack024_claim_evidence(evidence_category);
CREATE INDEX idx_pack024_evid_status ON pack024_carbon_neutral.pack024_claim_evidence(evidence_status);
CREATE INDEX idx_pack024_evid_acceptance ON pack024_carbon_neutral.pack024_claim_evidence(acceptance_status);
CREATE INDEX idx_pack024_evid_relevance ON pack024_carbon_neutral.pack024_claim_evidence(relevance_to_claim DESC);
CREATE INDEX idx_pack024_evid_validated ON pack024_carbon_neutral.pack024_claim_evidence(third_party_validation);
CREATE INDEX idx_pack024_evid_authenticity ON pack024_carbon_neutral.pack024_claim_evidence(authenticity_verified);
CREATE INDEX idx_pack024_evid_weight ON pack024_carbon_neutral.pack024_claim_evidence(evidence_weight DESC);
CREATE INDEX idx_pack024_evid_criticality ON pack024_carbon_neutral.pack024_claim_evidence(evidence_criticality);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_evid_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_claim_evidence
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_claim_verification
-- =============================================================================
-- Verification audits for claim substantiation.

CREATE TABLE pack024_carbon_neutral.pack024_claim_verification (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_substantiation_id UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_claim_substantiation(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    verification_date       DATE            NOT NULL,
    verification_type       VARCHAR(50)     NOT NULL,
    verification_scope      VARCHAR(255),
    verification_period_start DATE,
    verification_period_end DATE,
    verifier_name           VARCHAR(255),
    verifier_organization   VARCHAR(255),
    verifier_credentials    VARCHAR(255),
    verifier_independence   BOOLEAN         DEFAULT TRUE,
    verifier_experience     VARCHAR(255),
    verification_methodology VARCHAR(255),
    methodology_standard    VARCHAR(100),
    assurance_level         VARCHAR(50),
    sampling_approach       VARCHAR(100),
    sample_size             INTEGER,
    sample_percentage       DECIMAL(6,2),
    review_procedures       TEXT[],
    testing_procedures      TEXT[],
    recalculation_performed BOOLEAN         DEFAULT FALSE,
    recalculation_results   JSONB           DEFAULT '{}',
    analytical_procedures   TEXT[],
    substantive_procedures  TEXT[],
    findings_summary        TEXT,
    findings_detail         JSONB           DEFAULT '{}',
    exceptions_identified   BOOLEAN         DEFAULT FALSE,
    exceptions_count        INTEGER,
    exception_details       TEXT[],
    exception_significance  VARCHAR(30),
    exception_resolution    TEXT,
    management_response     TEXT,
    corrective_actions      TEXT[],
    verification_conclusion VARCHAR(500),
    overall_opinion         VARCHAR(30),
    qualified_opinion       BOOLEAN         DEFAULT FALSE,
    qualification_basis     TEXT,
    verification_confidence DECIMAL(5,2),
    limitations_noted       TEXT[],
    risk_assessment         VARCHAR(255),
    internal_control_review BOOLEAN         DEFAULT FALSE,
    control_deficiencies    TEXT[],
    compliance_assessment   BOOLEAN         DEFAULT FALSE,
    compliance_findings     TEXT[],
    verification_status     VARCHAR(30),
    report_issued           BOOLEAN         DEFAULT FALSE,
    report_date             DATE,
    report_reference        VARCHAR(100),
    report_distribution     TEXT[],
    public_disclosure       BOOLEAN         DEFAULT FALSE,
    approval_status         VARCHAR(30),
    approved_by             VARCHAR(255),
    approval_date           DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_ver_type CHECK (
        verification_type IN ('INDEPENDENT_VERIFICATION', 'INTERNAL_AUDIT', 'EXTERNAL_AUDIT',
                             'PEER_REVIEW', 'REGULATORY_INSPECTION', 'THIRD_PARTY_ASSURANCE')
    ),
    CONSTRAINT chk_pack024_ver_assurance CHECK (
        assurance_level IN ('REASONABLE', 'LIMITED', 'AGREED_UPON', 'COMPILATION')
    ),
    CONSTRAINT chk_pack024_ver_opinion CHECK (
        overall_opinion IN ('UNQUALIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER')
    )
);

-- Indexes
CREATE INDEX idx_pack024_ver_claim_id ON pack024_carbon_neutral.pack024_claim_verification(claim_substantiation_id);
CREATE INDEX idx_pack024_ver_org ON pack024_carbon_neutral.pack024_claim_verification(org_id);
CREATE INDEX idx_pack024_ver_tenant ON pack024_carbon_neutral.pack024_claim_verification(tenant_id);
CREATE INDEX idx_pack024_ver_date ON pack024_carbon_neutral.pack024_claim_verification(verification_date DESC);
CREATE INDEX idx_pack024_ver_type ON pack024_carbon_neutral.pack024_claim_verification(verification_type);
CREATE INDEX idx_pack024_ver_scope ON pack024_carbon_neutral.pack024_claim_verification(verification_scope);
CREATE INDEX idx_pack024_ver_verifier ON pack024_carbon_neutral.pack024_claim_verification(verifier_organization);
CREATE INDEX idx_pack024_ver_assurance ON pack024_carbon_neutral.pack024_claim_verification(assurance_level);
CREATE INDEX idx_pack024_ver_opinion ON pack024_carbon_neutral.pack024_claim_verification(overall_opinion);
CREATE INDEX idx_pack024_ver_exceptions ON pack024_carbon_neutral.pack024_claim_verification(exceptions_identified);
CREATE INDEX idx_pack024_ver_qualified ON pack024_carbon_neutral.pack024_claim_verification(qualified_opinion);
CREATE INDEX idx_pack024_ver_report_issued ON pack024_carbon_neutral.pack024_claim_verification(report_issued);
CREATE INDEX idx_pack024_ver_confidence ON pack024_carbon_neutral.pack024_claim_verification(verification_confidence DESC);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_ver_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_claim_verification
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_claim_disclosure
-- =============================================================================
-- Public disclosure documentation and communication of carbon neutral claims.

CREATE TABLE pack024_carbon_neutral.pack024_claim_disclosure (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_substantiation_id UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_claim_substantiation(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    disclosure_date         DATE            NOT NULL,
    disclosure_type         VARCHAR(50)     NOT NULL,
    disclosure_medium       VARCHAR(100)[],
    disclosure_audience     TEXT[],
    disclosure_statement    TEXT,
    claim_wording           VARCHAR(1000),
    claim_context           TEXT,
    supporting_information  TEXT,
    methodology_explanation VARCHAR(500),
    limitation_disclaimers  TEXT[],
    verification_references TEXT[],
    certification_references TEXT[],
    external_links          VARCHAR(500)[],
    website_integration     BOOLEAN         DEFAULT FALSE,
    website_url             VARCHAR(500),
    report_integration      BOOLEAN         DEFAULT FALSE,
    report_name             VARCHAR(255),
    report_page_reference   VARCHAR(50),
    social_media_disclosure BOOLEAN         DEFAULT FALSE,
    social_media_platforms  VARCHAR(50)[],
    social_media_reach      INTEGER,
    stakeholder_communication_plan BOOLEAN   DEFAULT FALSE,
    communication_channels  TEXT[],
    key_messaging           TEXT[],
    faqs_prepared           BOOLEAN         DEFAULT FALSE,
    faq_topics              TEXT[],
    internal_communication  BOOLEAN         DEFAULT FALSE,
    employee_communication  BOOLEAN         DEFAULT FALSE,
    investor_communication  BOOLEAN         DEFAULT FALSE,
    customer_communication  BOOLEAN         DEFAULT FALSE,
    supplier_communication  BOOLEAN         DEFAULT FALSE,
    regulatory_reporting    BOOLEAN         DEFAULT FALSE,
    regulatory_body         VARCHAR(100),
    regulatory_filing_date  DATE,
    regulatory_filing_reference VARCHAR(100),
    disclosure_approval     BOOLEAN         DEFAULT FALSE,
    approved_by             VARCHAR(255),
    approval_date           DATE,
    approval_notes          TEXT,
    pre_disclosure_review   BOOLEAN         DEFAULT FALSE,
    reviewer_name           VARCHAR(255),
    review_date             DATE,
    review_approval         BOOLEAN         DEFAULT FALSE,
    legal_review_completed  BOOLEAN         DEFAULT FALSE,
    legal_reviewer          VARCHAR(255),
    legal_review_date       DATE,
    legal_clearance         BOOLEAN         DEFAULT FALSE,
    compliance_review       BOOLEAN         DEFAULT FALSE,
    compliance_reviewer     VARCHAR(255),
    compliance_clearance    BOOLEAN         DEFAULT FALSE,
    disclosure_status       VARCHAR(30)     DEFAULT 'draft',
    publication_date        DATE,
    publication_status      VARCHAR(30),
    public_availability     BOOLEAN         DEFAULT FALSE,
    access_controls         VARCHAR(50),
    retention_period        INTEGER,
    archival_plan           TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_disc_type CHECK (
        disclosure_type IN ('ANNUAL_REPORT', 'SUSTAINABILITY_REPORT', 'CSR_REPORT', 'WEBSITE_STATEMENT',
                           'MARKETING_MATERIAL', 'PRODUCT_LABEL', 'REGULATORY_FILING', 'INVESTOR_PRESENTATION',
                           'EMPLOYEE_COMMUNICATION', 'CUSTOMER_NOTIFICATION', 'OTHER')
    ),
    CONSTRAINT chk_pack024_disc_status CHECK (
        disclosure_status IN ('DRAFT', 'REVIEW', 'APPROVED', 'PUBLISHED', 'ARCHIVED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_disc_claim_id ON pack024_carbon_neutral.pack024_claim_disclosure(claim_substantiation_id);
CREATE INDEX idx_pack024_disc_org ON pack024_carbon_neutral.pack024_claim_disclosure(org_id);
CREATE INDEX idx_pack024_disc_tenant ON pack024_carbon_neutral.pack024_claim_disclosure(tenant_id);
CREATE INDEX idx_pack024_disc_date ON pack024_carbon_neutral.pack024_claim_disclosure(disclosure_date DESC);
CREATE INDEX idx_pack024_disc_type ON pack024_carbon_neutral.pack024_claim_disclosure(disclosure_type);
CREATE INDEX idx_pack024_disc_status ON pack024_carbon_neutral.pack024_claim_disclosure(disclosure_status);
CREATE INDEX idx_pack024_disc_public ON pack024_carbon_neutral.pack024_claim_disclosure(public_availability);
CREATE INDEX idx_pack024_disc_website ON pack024_carbon_neutral.pack024_claim_disclosure(website_integration);
CREATE INDEX idx_pack024_disc_regulatory ON pack024_carbon_neutral.pack024_claim_disclosure(regulatory_reporting);
CREATE INDEX idx_pack024_disc_approval ON pack024_carbon_neutral.pack024_claim_disclosure(disclosure_approval);
CREATE INDEX idx_pack024_disc_legal_clearance ON pack024_carbon_neutral.pack024_claim_disclosure(legal_clearance);
CREATE INDEX idx_pack024_disc_publication ON pack024_carbon_neutral.pack024_claim_disclosure(publication_date);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_disc_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_claim_disclosure
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack024_carbon_neutral TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack024_carbon_neutral TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack024_carbon_neutral TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack024_carbon_neutral.pack024_claim_substantiation IS
'Carbon neutral claims registry with substantiation requirements, evidence tracking, and approval workflow.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_claim_evidence IS
'Evidence documentation for claim substantiation with source credibility, validation, and acceptance status.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_claim_verification IS
'Verification audits for claim substantiation with verifier independence, findings, and opinion issuance.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_claim_disclosure IS
'Public disclosure documentation and communication of carbon neutral claims with approval, review, and publication tracking.';

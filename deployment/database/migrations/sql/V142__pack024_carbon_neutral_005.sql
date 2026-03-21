-- =============================================================================
-- V142: PACK-024-carbon-neutral-005: Registry Retirements
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for carbon credit retirement processing and tracking with
-- registry integration, retirement statements, and compliance documentation.
--
-- EXTENDS:
--   V141: Portfolio Optimization Results
--
-- These tables provide the retirement execution and compliance framework
-- for offset retirement with full audit trail and documentation management.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_retirement_records          - Retirement transactions
--   2. pack024_carbon_neutral.pack024_retirement_statements       - Formal statements
--   3. pack024_carbon_neutral.pack024_registry_submissions        - Registry uploads
--   4. pack024_carbon_neutral.pack024_retirement_certificates     - Certificates
--
-- Also includes: 45+ indexes, update triggers, security grants, and comments.
-- Previous: V141__pack024_carbon_neutral_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_retirement_records
-- =============================================================================
-- Carbon credit retirement records with registry integration and tracking.

CREATE TABLE pack024_carbon_neutral.pack024_retirement_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    credit_inventory_id     UUID            REFERENCES pack024_carbon_neutral.pack024_credit_inventory(id),
    retirement_date         DATE            NOT NULL,
    retirement_type         VARCHAR(50)     NOT NULL,
    retirement_reason       VARCHAR(100),
    retirement_basis        VARCHAR(255),
    units_retired           DECIMAL(18,2)   NOT NULL,
    unit_type               VARCHAR(30)     DEFAULT 'tCO2e',
    retirement_quantity     DECIMAL(18,2)   NOT NULL,
    covered_footprint_id    UUID,
    covered_footprint_year  INTEGER,
    covered_emissions       DECIMAL(18,4),
    coverage_percentage     DECIMAL(6,2),
    retiring_party_name     VARCHAR(255)    NOT NULL,
    retiring_party_address  VARCHAR(500),
    retiring_party_country  VARCHAR(3),
    first_owner_name        VARCHAR(255),
    ownership_chain         TEXT[],
    verification_status     VARCHAR(30),
    third_party_verified    BOOLEAN         DEFAULT FALSE,
    verifier_organization   VARCHAR(255),
    verification_date       DATE,
    assurance_level         VARCHAR(30),
    permanent_retirement    BOOLEAN         DEFAULT TRUE,
    co_benefits_retired     TEXT[],
    credit_preservation_method VARCHAR(100),
    credit_preservation_evidence TEXT,
    double_counting_prevention BOOLEAN      DEFAULT TRUE,
    double_counting_check_results TEXT,
    registry_name           VARCHAR(100),
    registry_account_id     VARCHAR(100),
    serial_numbers          TEXT[],
    batch_verification_code VARCHAR(100),
    retirement_status       VARCHAR(30)     DEFAULT 'initiated',
    submission_date         DATE,
    registry_confirmation_date DATE,
    registry_confirmation_number VARCHAR(100),
    permanent_record_indicator BOOLEAN      DEFAULT FALSE,
    retirement_statement_prepared BOOLEAN   DEFAULT FALSE,
    statement_distribution_date DATE,
    statement_recipients    TEXT[],
    public_disclosure       BOOLEAN         DEFAULT FALSE,
    public_disclosure_date  DATE,
    disclosure_url          VARCHAR(500),
    audit_trail             JSONB           DEFAULT '{}',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_ret_type CHECK (
        retirement_type IN ('VOLUNTARY', 'COMPLIANCE', 'VOLUNTARY_CORPORATE', 'COMPLIANCE_REGULATORY',
                           'RETIREMENT_OF_EXCESS', 'EARLY_RETIREMENT', 'TRANSFER_FOR_RETIREMENT')
    ),
    CONSTRAINT chk_pack024_ret_units_non_neg CHECK (
        units_retired > 0 AND retirement_quantity > 0
    ),
    CONSTRAINT chk_pack024_ret_status CHECK (
        retirement_status IN ('INITIATED', 'SUBMITTED', 'CONFIRMED', 'FAILED', 'CANCELLED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_ret_org ON pack024_carbon_neutral.pack024_retirement_records(org_id);
CREATE INDEX idx_pack024_ret_tenant ON pack024_carbon_neutral.pack024_retirement_records(tenant_id);
CREATE INDEX idx_pack024_ret_inventory_id ON pack024_carbon_neutral.pack024_retirement_records(credit_inventory_id);
CREATE INDEX idx_pack024_ret_date ON pack024_carbon_neutral.pack024_retirement_records(retirement_date DESC);
CREATE INDEX idx_pack024_ret_type ON pack024_carbon_neutral.pack024_retirement_records(retirement_type);
CREATE INDEX idx_pack024_ret_status ON pack024_carbon_neutral.pack024_retirement_records(retirement_status);
CREATE INDEX idx_pack024_ret_retiring_party ON pack024_carbon_neutral.pack024_retirement_records(retiring_party_name);
CREATE INDEX idx_pack024_ret_registry_name ON pack024_carbon_neutral.pack024_retirement_records(registry_name);
CREATE INDEX idx_pack024_ret_verified ON pack024_carbon_neutral.pack024_retirement_records(third_party_verified);
CREATE INDEX idx_pack024_ret_permanent ON pack024_carbon_neutral.pack024_retirement_records(permanent_retirement);
CREATE INDEX idx_pack024_ret_registry_confirm ON pack024_carbon_neutral.pack024_retirement_records(registry_confirmation_date);
CREATE INDEX idx_pack024_ret_public_disclosure ON pack024_carbon_neutral.pack024_retirement_records(public_disclosure);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_ret_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_retirement_records
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_retirement_statements
-- =============================================================================
-- Formal retirement statements with certification and public communication.

CREATE TABLE pack024_carbon_neutral.pack024_retirement_statements (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    retirement_record_id    UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_retirement_records(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    statement_date          DATE            NOT NULL,
    statement_type          VARCHAR(50)     NOT NULL,
    statement_version       VARCHAR(20)     DEFAULT '1.0',
    statement_title         VARCHAR(500),
    statement_language      VARCHAR(10)     DEFAULT 'en',
    statement_prepared_by   VARCHAR(255),
    statement_approved_by   VARCHAR(255),
    statement_approved_date DATE,
    executive_summary       TEXT,
    detailed_description    TEXT,
    methodology_section     TEXT,
    findings_section        TEXT,
    conclusions_section     TEXT,
    credit_details_section  JSONB           DEFAULT '{}',
    retirement_justification TEXT,
    carbon_footprint_covered DECIMAL(18,4),
    baseline_year           INTEGER,
    reporting_year          INTEGER,
    contribution_to_targets TEXT,
    alignment_with_strategy TEXT,
    co_benefits_section     TEXT,
    sdg_contribution        TEXT[],
    social_impact_section   TEXT,
    environmental_impact_section TEXT,
    governance_section      TEXT,
    stakeholder_engagement  TEXT,
    third_party_assurance   BOOLEAN         DEFAULT FALSE,
    assurance_provider      VARCHAR(255),
    assurance_type          VARCHAR(50),
    assurance_statement     TEXT,
    assurance_date          DATE,
    assurance_opinion       VARCHAR(100),
    certification_obtained  BOOLEAN         DEFAULT FALSE,
    certification_type      VARCHAR(100),
    certification_body      VARCHAR(255),
    certification_date      DATE,
    certification_reference VARCHAR(100),
    verification_statement  TEXT,
    verification_evidence   TEXT[],
    limitations_and_assumptions TEXT,
    comparison_with_targets JSONB           DEFAULT '{}',
    future_commitments      TEXT[],
    supporting_documentation TEXT[],
    public_availability     BOOLEAN         DEFAULT FALSE,
    publication_date        DATE,
    publication_channel     VARCHAR(100),
    publication_url         VARCHAR(500),
    internal_distribution   TEXT[],
    external_distribution   TEXT[],
    statement_status        VARCHAR(30)     DEFAULT 'draft',
    approval_status         VARCHAR(30),
    publication_status      VARCHAR(30),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_stmt_type CHECK (
        statement_type IN ('CARBON_NEUTRAL_CLAIM', 'EMISSION_REDUCTION_SUMMARY',
                          'OFFSET_VERIFICATION', 'CORPORATE_COMMITMENT', 'OTHER')
    ),
    CONSTRAINT chk_pack024_stmt_status CHECK (
        statement_status IN ('DRAFT', 'REVIEW', 'APPROVED', 'PUBLISHED', 'ARCHIVED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_stmt_ret_id ON pack024_carbon_neutral.pack024_retirement_statements(retirement_record_id);
CREATE INDEX idx_pack024_stmt_org ON pack024_carbon_neutral.pack024_retirement_statements(org_id);
CREATE INDEX idx_pack024_stmt_tenant ON pack024_carbon_neutral.pack024_retirement_statements(tenant_id);
CREATE INDEX idx_pack024_stmt_date ON pack024_carbon_neutral.pack024_retirement_statements(statement_date DESC);
CREATE INDEX idx_pack024_stmt_type ON pack024_carbon_neutral.pack024_retirement_statements(statement_type);
CREATE INDEX idx_pack024_stmt_status ON pack024_carbon_neutral.pack024_retirement_statements(statement_status);
CREATE INDEX idx_pack024_stmt_approved ON pack024_carbon_neutral.pack024_retirement_statements(statement_approved_date);
CREATE INDEX idx_pack024_stmt_assurance ON pack024_carbon_neutral.pack024_retirement_statements(third_party_assurance);
CREATE INDEX idx_pack024_stmt_certified ON pack024_carbon_neutral.pack024_retirement_statements(certification_obtained);
CREATE INDEX idx_pack024_stmt_public ON pack024_carbon_neutral.pack024_retirement_statements(public_availability);
CREATE INDEX idx_pack024_stmt_publication_date ON pack024_carbon_neutral.pack024_retirement_statements(publication_date);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_stmt_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_retirement_statements
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_registry_submissions
-- =============================================================================
-- Registry submission tracking with upload status and confirmation.

CREATE TABLE pack024_carbon_neutral.pack024_registry_submissions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    retirement_record_id    UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_retirement_records(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    submission_date         DATE            NOT NULL,
    submission_type         VARCHAR(50)     NOT NULL,
    registry_name           VARCHAR(100)    NOT NULL,
    registry_url            VARCHAR(500),
    account_holder_name     VARCHAR(255),
    account_id              VARCHAR(100),
    submission_method       VARCHAR(50),
    submission_format       VARCHAR(50),
    file_name               VARCHAR(500),
    file_size_bytes         BIGINT,
    file_checksum           VARCHAR(255),
    submission_reference    VARCHAR(100),
    batch_number            VARCHAR(100),
    serial_number_range     VARCHAR(100),
    units_submitted         DECIMAL(18,2),
    submission_status       VARCHAR(30)     DEFAULT 'pending_confirmation',
    submission_error        TEXT,
    retry_count             INTEGER         DEFAULT 0,
    max_retries             INTEGER         DEFAULT 3,
    estimated_processing_time INTERVAL,
    confirmation_received   BOOLEAN         DEFAULT FALSE,
    confirmation_date       DATE,
    confirmation_number     VARCHAR(100),
    confirmation_reference  VARCHAR(100),
    confirmation_details    JSONB           DEFAULT '{}',
    rejection_reason        TEXT,
    rejection_date          DATE,
    resubmission_required   BOOLEAN         DEFAULT FALSE,
    resubmission_plan       TEXT,
    permanent_record_created BOOLEAN        DEFAULT FALSE,
    permanent_record_id     VARCHAR(100),
    registry_serials_assigned TEXT[],
    compliance_verification BOOLEAN         DEFAULT FALSE,
    compliance_notes        TEXT,
    audit_trail             JSONB           DEFAULT '{}',
    submitted_by            VARCHAR(255),
    contact_email           VARCHAR(255),
    contact_phone           VARCHAR(20),
    registry_contact_person VARCHAR(255),
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_sub_type CHECK (
        submission_type IN ('INITIAL_SUBMISSION', 'AMENDMENT', 'CORRECTION', 'RESUBMISSION')
    ),
    CONSTRAINT chk_pack024_sub_status CHECK (
        submission_status IN ('PENDING_CONFIRMATION', 'CONFIRMED', 'REJECTED', 'CANCELLED', 'ARCHIVED')
    ),
    CONSTRAINT chk_pack024_sub_retries CHECK (
        retry_count >= 0 AND retry_count <= max_retries
    )
);

-- Indexes
CREATE INDEX idx_pack024_sub_ret_id ON pack024_carbon_neutral.pack024_registry_submissions(retirement_record_id);
CREATE INDEX idx_pack024_sub_org ON pack024_carbon_neutral.pack024_registry_submissions(org_id);
CREATE INDEX idx_pack024_sub_tenant ON pack024_carbon_neutral.pack024_registry_submissions(tenant_id);
CREATE INDEX idx_pack024_sub_date ON pack024_carbon_neutral.pack024_registry_submissions(submission_date DESC);
CREATE INDEX idx_pack024_sub_type ON pack024_carbon_neutral.pack024_registry_submissions(submission_type);
CREATE INDEX idx_pack024_sub_registry ON pack024_carbon_neutral.pack024_registry_submissions(registry_name);
CREATE INDEX idx_pack024_sub_status ON pack024_carbon_neutral.pack024_registry_submissions(submission_status);
CREATE INDEX idx_pack024_sub_confirmation ON pack024_carbon_neutral.pack024_registry_submissions(confirmation_received);
CREATE INDEX idx_pack024_sub_confirmation_date ON pack024_carbon_neutral.pack024_registry_submissions(confirmation_date);
CREATE INDEX idx_pack024_sub_resubmission ON pack024_carbon_neutral.pack024_registry_submissions(resubmission_required);
CREATE INDEX idx_pack024_sub_permanent_record ON pack024_carbon_neutral.pack024_registry_submissions(permanent_record_created);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_sub_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_registry_submissions
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_retirement_certificates
-- =============================================================================
-- Digital retirement certificates with authenticity verification.

CREATE TABLE pack024_carbon_neutral.pack024_retirement_certificates (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    retirement_record_id    UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_retirement_records(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    certificate_date        DATE            NOT NULL,
    certificate_number      VARCHAR(100)    NOT NULL UNIQUE,
    certificate_version     VARCHAR(20)     DEFAULT '1.0',
    issued_by               VARCHAR(255),
    issued_by_authority     VARCHAR(255),
    issue_date              DATE,
    valid_from_date         DATE,
    expiry_date             DATE,
    permanent_certificate   BOOLEAN         DEFAULT TRUE,
    retiring_entity_name    VARCHAR(255)    NOT NULL,
    retiring_entity_id      VARCHAR(100),
    retiring_entity_country VARCHAR(3),
    certificate_title       VARCHAR(500),
    certificate_description TEXT,
    units_certified         DECIMAL(18,2)   NOT NULL,
    certification_scope     TEXT,
    methodology_description TEXT,
    verification_standard   VARCHAR(100),
    standard_version        VARCHAR(20),
    corresponding_footprint_id UUID,
    corresponding_footprint_year INTEGER,
    corresponding_emissions DECIMAL(18,4),
    co_benefits_certified   TEXT[],
    sdg_alignment           TEXT[],
    impact_claims           TEXT[],
    certificate_format      VARCHAR(50),
    digital_format          BOOLEAN         DEFAULT TRUE,
    certificate_url         VARCHAR(500),
    certificate_hash        VARCHAR(255),
    blockchain_registered   BOOLEAN         DEFAULT FALSE,
    blockchain_address      VARCHAR(255),
    blockchain_transaction_id VARCHAR(255),
    qr_code_generated       BOOLEAN         DEFAULT FALSE,
    qr_code_url             VARCHAR(500),
    hologram_security       BOOLEAN         DEFAULT FALSE,
    hologram_serial         VARCHAR(100),
    authenticity_features   JSONB           DEFAULT '{}',
    certificate_status      VARCHAR(30)     DEFAULT 'issued',
    validity_status         VARCHAR(30),
    revocation_status       BOOLEAN         DEFAULT FALSE,
    revocation_date         DATE,
    revocation_reason       TEXT,
    public_registry         BOOLEAN         DEFAULT FALSE,
    public_registry_url     VARCHAR(500),
    verification_method     VARCHAR(100),
    verification_successful BOOLEAN         DEFAULT FALSE,
    verification_date       DATE,
    verification_notes      TEXT,
    recipient_contact       VARCHAR(255),
    delivery_method         VARCHAR(100),
    delivery_date           DATE,
    display_authorization   BOOLEAN         DEFAULT TRUE,
    additional_metadata     JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_cert_status CHECK (
        certificate_status IN ('ISSUED', 'ACTIVE', 'EXPIRED', 'REVOKED', 'ARCHIVED')
    ),
    CONSTRAINT chk_pack024_cert_units_non_neg CHECK (
        units_certified > 0
    )
);

-- Indexes
CREATE INDEX idx_pack024_cert_ret_id ON pack024_carbon_neutral.pack024_retirement_certificates(retirement_record_id);
CREATE INDEX idx_pack024_cert_org ON pack024_carbon_neutral.pack024_retirement_certificates(org_id);
CREATE INDEX idx_pack024_cert_tenant ON pack024_carbon_neutral.pack024_retirement_certificates(tenant_id);
CREATE INDEX idx_pack024_cert_number ON pack024_carbon_neutral.pack024_retirement_certificates(certificate_number);
CREATE INDEX idx_pack024_cert_date ON pack024_carbon_neutral.pack024_retirement_certificates(certificate_date DESC);
CREATE INDEX idx_pack024_cert_issued_by ON pack024_carbon_neutral.pack024_retirement_certificates(issued_by);
CREATE INDEX idx_pack024_cert_entity ON pack024_carbon_neutral.pack024_retirement_certificates(retiring_entity_name);
CREATE INDEX idx_pack024_cert_status ON pack024_carbon_neutral.pack024_retirement_certificates(certificate_status);
CREATE INDEX idx_pack024_cert_validity ON pack024_carbon_neutral.pack024_retirement_certificates(validity_status);
CREATE INDEX idx_pack024_cert_revocation ON pack024_carbon_neutral.pack024_retirement_certificates(revocation_status);
CREATE INDEX idx_pack024_cert_blockchain ON pack024_carbon_neutral.pack024_retirement_certificates(blockchain_registered);
CREATE INDEX idx_pack024_cert_public ON pack024_carbon_neutral.pack024_retirement_certificates(public_registry);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_cert_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_retirement_certificates
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

COMMENT ON TABLE pack024_carbon_neutral.pack024_retirement_records IS
'Carbon credit retirement records with registry integration, verification status, and permanent retirement tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_retirement_statements IS
'Formal retirement statements with certification, public communication, third-party assurance, and publication tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_registry_submissions IS
'Registry submission tracking with upload status, confirmation receipts, and retry management for registry processing.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_retirement_certificates IS
'Digital retirement certificates with authenticity verification, blockchain integration, and public registry publication.';

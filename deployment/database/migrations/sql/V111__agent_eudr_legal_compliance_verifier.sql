-- ============================================================================
-- V111: AGENT-EUDR-023 Legal Compliance Verifier Agent
-- ============================================================================
-- Creates tables for country legal frameworks registry, compliance documents,
-- certification records, red flag alerts, compliance assessments, audit reports,
-- legal requirements, compliance reports, country compliance mappings, batch
-- operations, legal opinions, verification logs, red flag events, compliance
-- history, and audit logs.
--
-- Schema: eudr_legal_compliance (15 tables)
-- Tables: 15 (11 regular + 4 hypertables)
-- Hypertables: gl_eudr_lcv_verification_log (30d chunks),
--              gl_eudr_lcv_red_flag_events (30d chunks),
--              gl_eudr_lcv_compliance_history (30d chunks),
--              gl_eudr_lcv_audit_log (30d chunks)
-- Continuous Aggregates: 2 (monthly_compliance_summary + weekly_red_flag_summary)
-- Retention Policies: 4 (5 years per EUDR Article 31)
-- Indexes: ~160 (B-tree, GIN, partial)
-- GIN indexes: ~20 (on JSONB columns)
-- Partial indexes: ~10 (for active/filtered records)
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V111: Creating AGENT-EUDR-023 Legal Compliance Verifier tables...';


-- ============================================================================
-- 1. gl_eudr_lcv_legal_frameworks — Country legal frameworks registry
-- ============================================================================
RAISE NOTICE 'V111 [1/15]: Creating gl_eudr_lcv_legal_frameworks...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_legal_frameworks (
    framework_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    legislation_category        VARCHAR(50)     NOT NULL,
        -- One of 8 categories from EUDR Article 2(40)
    framework_name              VARCHAR(500)    NOT NULL,
        -- Official name of the legislative or regulatory framework
    legal_citation              TEXT,
        -- Formal legal citation (e.g., "Law No. 12.651/2012 - Forest Code")
    effective_date              DATE,
        -- Date the framework entered into force
    last_updated                DATE,
        -- Date the framework was last amended or reviewed
    verification_source         VARCHAR(200),
        -- Organization or institution providing framework data
    source_url                  TEXT,
        -- URL to the official legal text or registry
    reliability_score           NUMERIC(3,2)    CHECK (reliability_score >= 0 AND reliability_score <= 1),
        -- Source reliability rating (0.0 = unverified, 1.0 = official gazette)
    metadata                    JSONB           DEFAULT '{}',
        -- Additional attributes (implementing_body, penalties, amendments, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_fw_category CHECK (legislation_category IN (
        'forest_protection', 'land_use_rights', 'environmental_protection',
        'indigenous_rights', 'anti_corruption', 'trade_customs',
        'labor_rights', 'tax_fiscal'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_country ON gl_eudr_lcv_legal_frameworks (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_category ON gl_eudr_lcv_legal_frameworks (legislation_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_name ON gl_eudr_lcv_legal_frameworks (framework_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_effective ON gl_eudr_lcv_legal_frameworks (effective_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_last_updated ON gl_eudr_lcv_legal_frameworks (last_updated DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_source ON gl_eudr_lcv_legal_frameworks (verification_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_reliability ON gl_eudr_lcv_legal_frameworks (reliability_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_tenant ON gl_eudr_lcv_legal_frameworks (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_created ON gl_eudr_lcv_legal_frameworks (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_country_cat ON gl_eudr_lcv_legal_frameworks (country_code, legislation_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high-reliability frameworks
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_high_rel ON gl_eudr_lcv_legal_frameworks (country_code, legislation_category, effective_date DESC)
        WHERE reliability_score >= 0.80;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_fw_metadata ON gl_eudr_lcv_legal_frameworks USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_legal_frameworks IS 'Registry of country-specific legal frameworks across 8 EUDR Article 2(40) legislation categories with provenance and reliability scoring';
COMMENT ON COLUMN gl_eudr_lcv_legal_frameworks.legislation_category IS 'EUDR Article 2(40) categories: forest_protection, land_use_rights, environmental_protection, indigenous_rights, anti_corruption, trade_customs, labor_rights, tax_fiscal';
COMMENT ON COLUMN gl_eudr_lcv_legal_frameworks.reliability_score IS 'Source reliability: 0.0 = unverified/informal, 1.0 = official gazette or authoritative government source';


-- ============================================================================
-- 2. gl_eudr_lcv_compliance_documents — Verification documents
-- ============================================================================
RAISE NOTICE 'V111 [2/15]: Creating gl_eudr_lcv_compliance_documents...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_compliance_documents (
    document_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
        -- Supplier submitting the compliance document
    plot_id                     UUID,
        -- Production plot this document relates to (nullable for supplier-level docs)
    document_type               VARCHAR(50)     NOT NULL,
        -- Classification of the compliance document
    issuing_authority           VARCHAR(500),
        -- Authority or organization that issued the document
    issue_date                  DATE,
        -- Date the document was issued
    expiry_date                 DATE,
        -- Date the document expires (nullable for perpetual documents)
    validity_status             VARCHAR(30)     NOT NULL DEFAULT 'pending_review',
        -- Current validity assessment
    verification_date           TIMESTAMPTZ,
        -- When the document was last verified
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for document integrity verification
    document_url                TEXT,
        -- URL or storage path to the document file
    metadata                    JSONB           DEFAULT '{}',
        -- Additional document attributes (version, language, certifier, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_doc_type CHECK (document_type IN (
        'land_title', 'forest_permit', 'environmental_license',
        'export_permit', 'tax_clearance', 'labor_compliance_cert',
        'indigenous_consent', 'anti_corruption_declaration',
        'customs_declaration', 'phytosanitary_certificate',
        'origin_certificate', 'transport_permit'
    )),
    CONSTRAINT chk_lcv_doc_validity CHECK (validity_status IN (
        'valid', 'expired', 'revoked', 'suspended', 'pending_review',
        'rejected', 'under_investigation'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_supplier ON gl_eudr_lcv_compliance_documents (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_plot ON gl_eudr_lcv_compliance_documents (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_type ON gl_eudr_lcv_compliance_documents (document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_authority ON gl_eudr_lcv_compliance_documents (issuing_authority);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_issue ON gl_eudr_lcv_compliance_documents (issue_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_expiry ON gl_eudr_lcv_compliance_documents (expiry_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_validity ON gl_eudr_lcv_compliance_documents (validity_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_verification ON gl_eudr_lcv_compliance_documents (verification_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_provenance ON gl_eudr_lcv_compliance_documents (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_tenant ON gl_eudr_lcv_compliance_documents (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_created ON gl_eudr_lcv_compliance_documents (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_supplier_type ON gl_eudr_lcv_compliance_documents (supplier_id, document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for valid documents
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_active ON gl_eudr_lcv_compliance_documents (supplier_id, document_type, expiry_date DESC)
        WHERE validity_status = 'valid';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for expiring soon (within 90 days)
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_expiring ON gl_eudr_lcv_compliance_documents (expiry_date, supplier_id)
        WHERE validity_status = 'valid' AND expiry_date IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_doc_metadata ON gl_eudr_lcv_compliance_documents USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_compliance_documents IS 'Verification document registry for 12 compliance document types with validity tracking, provenance hashing, and authority verification';
COMMENT ON COLUMN gl_eudr_lcv_compliance_documents.document_type IS '12 document types: land_title, forest_permit, environmental_license, export_permit, tax_clearance, labor_compliance_cert, indigenous_consent, anti_corruption_declaration, customs_declaration, phytosanitary_certificate, origin_certificate, transport_permit';
COMMENT ON COLUMN gl_eudr_lcv_compliance_documents.validity_status IS 'Validity states: valid (verified and current), expired, revoked (issuer withdrawn), suspended (under review), pending_review, rejected, under_investigation';


-- ============================================================================
-- 3. gl_eudr_lcv_certification_records — Certification validation
-- ============================================================================
RAISE NOTICE 'V111 [3/15]: Creating gl_eudr_lcv_certification_records...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_certification_records (
    certification_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
        -- Certified supplier
    certification_scheme        VARCHAR(50)     NOT NULL,
        -- Certification scheme identifier
    certificate_number          VARCHAR(100)    NOT NULL,
        -- Unique certificate number assigned by the certifying body
    issue_date                  DATE            NOT NULL,
        -- Date the certificate was issued
    expiry_date                 DATE,
        -- Date the certificate expires
    scope                       TEXT,
        -- Description of the certification scope (products, geographies, etc.)
    eudr_equivalence            BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the scheme has been assessed as EUDR-equivalent
    validation_status           VARCHAR(30)     NOT NULL DEFAULT 'pending',
        -- Current validation status
    validation_date             TIMESTAMPTZ,
        -- When the certificate was last validated
    certifying_body             VARCHAR(300),
        -- Name of the accredited certifying body
    audit_report_url            TEXT,
        -- URL to the audit report supporting the certification
    metadata                    JSONB           DEFAULT '{}',
        -- Additional attributes (audit_date, nonconformities, scope_changes, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_cert_scheme CHECK (certification_scheme IN (
        'FSC', 'PEFC', 'RSPO', 'RA', 'ISCC',
        'SAN', 'UTZ', 'GLOBALG.A.P', 'BONSUCRO', 'ASI'
    )),
    CONSTRAINT chk_lcv_cert_status CHECK (validation_status IN (
        'valid', 'expired', 'suspended', 'withdrawn', 'pending',
        'under_review', 'rejected'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_supplier ON gl_eudr_lcv_certification_records (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_scheme ON gl_eudr_lcv_certification_records (certification_scheme);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_number ON gl_eudr_lcv_certification_records (certificate_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_issue ON gl_eudr_lcv_certification_records (issue_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_expiry ON gl_eudr_lcv_certification_records (expiry_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_eudr_eq ON gl_eudr_lcv_certification_records (eudr_equivalence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_status ON gl_eudr_lcv_certification_records (validation_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_validation ON gl_eudr_lcv_certification_records (validation_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_body ON gl_eudr_lcv_certification_records (certifying_body);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_tenant ON gl_eudr_lcv_certification_records (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_created ON gl_eudr_lcv_certification_records (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_supplier_scheme ON gl_eudr_lcv_certification_records (supplier_id, certification_scheme);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for valid EUDR-equivalent certificates
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_eudr_valid ON gl_eudr_lcv_certification_records (supplier_id, certification_scheme, expiry_date DESC)
        WHERE validation_status = 'valid' AND eudr_equivalence = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_cert_metadata ON gl_eudr_lcv_certification_records USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_certification_records IS 'Certification validation records for FSC, PEFC, RSPO, RA, ISCC, and other recognized schemes with EUDR equivalence assessment';
COMMENT ON COLUMN gl_eudr_lcv_certification_records.certification_scheme IS 'Recognized schemes: FSC (Forest Stewardship Council), PEFC (Programme for Endorsement of Forest Certification), RSPO (Roundtable on Sustainable Palm Oil), RA (Rainforest Alliance), ISCC (International Sustainability and Carbon Certification)';
COMMENT ON COLUMN gl_eudr_lcv_certification_records.eudr_equivalence IS 'Whether this certification scheme has been assessed as providing EUDR-equivalent assurance per Article 10(1)';


-- ============================================================================
-- 4. gl_eudr_lcv_red_flag_alerts — Red flag detections
-- ============================================================================
RAISE NOTICE 'V111 [4/15]: Creating gl_eudr_lcv_red_flag_alerts...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_red_flag_alerts (
    flag_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
        -- Supplier associated with the red flag
    plot_id                     UUID,
        -- Production plot (nullable for supplier-level flags)
    red_flag_category           VARCHAR(50)     NOT NULL,
        -- High-level red flag category
    red_flag_type               VARCHAR(100)    NOT NULL,
        -- Specific red flag type within the category
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Severity classification
    severity_score              NUMERIC(5,2)    CHECK (severity_score >= 0 AND severity_score <= 100),
        -- Numeric severity score for ranking (0-100)
    evidence                    JSONB           DEFAULT '[]',
        -- [{ "source": "...", "type": "document_mismatch", "date": "...", "url": "..." }, ...]
    detected_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
        -- Date the red flag was detected
    status                      VARCHAR(30)     NOT NULL DEFAULT 'open',
        -- Current red flag handling status
    suppressed                  BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the alert has been suppressed (acknowledged but not acted upon)
    suppressed_reason           TEXT,
        -- Justification for suppression (required if suppressed=true)
    resolved_at                 TIMESTAMPTZ,
        -- Timestamp when the red flag was resolved
    resolution_notes            TEXT,
        -- Description of the resolution or corrective action
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_rf_category CHECK (red_flag_category IN (
        'document_irregularity', 'certification_gap', 'legal_non_compliance',
        'sanctions_match', 'corruption_indicator', 'deforestation_link'
    )),
    CONSTRAINT chk_lcv_rf_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low', 'informational'
    )),
    CONSTRAINT chk_lcv_rf_status CHECK (status IN (
        'open', 'investigating', 'confirmed', 'remediation_in_progress',
        'resolved', 'dismissed', 'escalated', 'suppressed'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_supplier ON gl_eudr_lcv_red_flag_alerts (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_plot ON gl_eudr_lcv_red_flag_alerts (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_category ON gl_eudr_lcv_red_flag_alerts (red_flag_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_type ON gl_eudr_lcv_red_flag_alerts (red_flag_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_severity ON gl_eudr_lcv_red_flag_alerts (severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_score ON gl_eudr_lcv_red_flag_alerts (severity_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_detected ON gl_eudr_lcv_red_flag_alerts (detected_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_status ON gl_eudr_lcv_red_flag_alerts (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_suppressed ON gl_eudr_lcv_red_flag_alerts (suppressed);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_tenant ON gl_eudr_lcv_red_flag_alerts (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_created ON gl_eudr_lcv_red_flag_alerts (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_supplier_cat ON gl_eudr_lcv_red_flag_alerts (supplier_id, red_flag_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_sev_status ON gl_eudr_lcv_red_flag_alerts (severity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unresolved red flags
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_unresolved ON gl_eudr_lcv_red_flag_alerts (severity, severity_score DESC)
        WHERE status IN ('open', 'investigating', 'confirmed', 'remediation_in_progress', 'escalated');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rf_evidence ON gl_eudr_lcv_red_flag_alerts USING GIN (evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_red_flag_alerts IS 'Red flag detection registry across 6 categories with severity scoring, evidence tracking, suppression management, and resolution workflow';
COMMENT ON COLUMN gl_eudr_lcv_red_flag_alerts.red_flag_category IS '6 categories: document_irregularity, certification_gap, legal_non_compliance, sanctions_match, corruption_indicator, deforestation_link';
COMMENT ON COLUMN gl_eudr_lcv_red_flag_alerts.severity_score IS 'Numeric severity score (0-100) derived from flag type, evidence strength, recurrence, and supplier history';


-- ============================================================================
-- 5. gl_eudr_lcv_compliance_assessments — Full assessments
-- ============================================================================
RAISE NOTICE 'V111 [5/15]: Creating gl_eudr_lcv_compliance_assessments...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_compliance_assessments (
    assessment_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
        -- Assessed supplier
    assessed_date               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the assessment
    overall_status              VARCHAR(30)     NOT NULL,
        -- Overall compliance determination
    overall_score               NUMERIC(5,2)    CHECK (overall_score >= 0 AND overall_score <= 100),
        -- Composite compliance score across all 8 categories (0-100)
    category_results            JSONB           NOT NULL DEFAULT '{}',
        -- { "forest_protection": {"status": "compliant", "score": 95, "gaps": []},
        --   "land_use_rights": {"status": "partial", "score": 70, "gaps": ["missing deed"]},
        --   ... for all 8 categories }
    red_flags_count             INTEGER         NOT NULL DEFAULT 0 CHECK (red_flags_count >= 0),
        -- Total number of red flags detected in this assessment
    gaps_identified             JSONB           DEFAULT '[]',
        -- [{ "category": "...", "gap": "...", "severity": "high", "recommendation": "..." }, ...]
    assessor                    VARCHAR(200),
        -- User or system agent performing the assessment
    assessment_method           VARCHAR(50)     NOT NULL DEFAULT 'automated',
        -- Method of assessment
    notes                       TEXT,
        -- Assessment notes and observations
    metadata                    JSONB           DEFAULT '{}',
        -- Additional assessment context (data_sources, confidence_levels, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_asmt_status CHECK (overall_status IN (
        'compliant', 'non_compliant', 'partially_compliant',
        'under_review', 'remediation_required', 'insufficient_data'
    )),
    CONSTRAINT chk_lcv_asmt_method CHECK (assessment_method IN (
        'automated', 'manual_review', 'third_party_audit',
        'hybrid', 'self_declaration'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_supplier ON gl_eudr_lcv_compliance_assessments (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_assessed ON gl_eudr_lcv_compliance_assessments (assessed_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_status ON gl_eudr_lcv_compliance_assessments (overall_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_score ON gl_eudr_lcv_compliance_assessments (overall_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_red_flags ON gl_eudr_lcv_compliance_assessments (red_flags_count DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_assessor ON gl_eudr_lcv_compliance_assessments (assessor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_method ON gl_eudr_lcv_compliance_assessments (assessment_method);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_tenant ON gl_eudr_lcv_compliance_assessments (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_created ON gl_eudr_lcv_compliance_assessments (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_supplier_date ON gl_eudr_lcv_compliance_assessments (supplier_id, assessed_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for non-compliant assessments requiring action
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_non_compliant ON gl_eudr_lcv_compliance_assessments (supplier_id, overall_score)
        WHERE overall_status IN ('non_compliant', 'remediation_required');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_cat_results ON gl_eudr_lcv_compliance_assessments USING GIN (category_results);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_gaps ON gl_eudr_lcv_compliance_assessments USING GIN (gaps_identified);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_asmt_metadata ON gl_eudr_lcv_compliance_assessments USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_compliance_assessments IS 'Full compliance assessments across 8 EUDR Article 2(40) categories with composite scoring, gap analysis, and red flag counts';
COMMENT ON COLUMN gl_eudr_lcv_compliance_assessments.category_results IS 'Per-category results for all 8 legislation categories with individual status, score, and identified gaps';
COMMENT ON COLUMN gl_eudr_lcv_compliance_assessments.overall_score IS 'Weighted composite score (0-100): <60 non_compliant, 60-79 partially_compliant, >=80 compliant';


-- ============================================================================
-- 6. gl_eudr_lcv_audit_reports — Third-party audit reports
-- ============================================================================
RAISE NOTICE 'V111 [6/15]: Creating gl_eudr_lcv_audit_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_audit_reports (
    audit_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
        -- Audited supplier
    audit_firm                  VARCHAR(300)    NOT NULL,
        -- Name of the audit firm or organization
    auditor_name                VARCHAR(200),
        -- Lead auditor name
    audit_type                  VARCHAR(50)     NOT NULL,
        -- Type of audit performed
    audit_date                  DATE            NOT NULL,
        -- Date the audit was conducted
    audit_scope                 TEXT,
        -- Description of the audit scope
    findings                    JSONB           DEFAULT '[]',
        -- [{ "finding_id": "F001", "category": "...", "severity": "major", "description": "...", "recommendation": "..." }, ...]
    nonconformities_major       INTEGER         NOT NULL DEFAULT 0 CHECK (nonconformities_major >= 0),
        -- Count of major nonconformities
    nonconformities_minor       INTEGER         NOT NULL DEFAULT 0 CHECK (nonconformities_minor >= 0),
        -- Count of minor nonconformities
    corrective_action_deadline  DATE,
        -- Deadline for corrective action closure
    audit_result                VARCHAR(30)     NOT NULL,
        -- Overall audit result
    report_url                  TEXT,
        -- URL to the audit report document
    metadata                    JSONB           DEFAULT '{}',
        -- Additional audit context (accreditation, standards_used, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_audit_type CHECK (audit_type IN (
        'initial_certification', 'surveillance', 'recertification',
        'special_investigation', 'desk_review', 'field_audit'
    )),
    CONSTRAINT chk_lcv_audit_result CHECK (audit_result IN (
        'pass', 'conditional_pass', 'fail', 'pending_corrective_action',
        'suspended', 'withdrawn'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_supplier ON gl_eudr_lcv_audit_reports (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_firm ON gl_eudr_lcv_audit_reports (audit_firm);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_type ON gl_eudr_lcv_audit_reports (audit_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_date ON gl_eudr_lcv_audit_reports (audit_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_nc_major ON gl_eudr_lcv_audit_reports (nonconformities_major DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_deadline ON gl_eudr_lcv_audit_reports (corrective_action_deadline);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_result ON gl_eudr_lcv_audit_reports (audit_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_tenant ON gl_eudr_lcv_audit_reports (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_created ON gl_eudr_lcv_audit_reports (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_supplier_date ON gl_eudr_lcv_audit_reports (supplier_id, audit_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for audits with outstanding corrective actions
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_pending ON gl_eudr_lcv_audit_reports (corrective_action_deadline, supplier_id)
        WHERE audit_result IN ('conditional_pass', 'pending_corrective_action');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_findings ON gl_eudr_lcv_audit_reports USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_audit_metadata ON gl_eudr_lcv_audit_reports USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_audit_reports IS 'Third-party audit reports with findings, nonconformity counts, corrective action tracking, and audit result classification';
COMMENT ON COLUMN gl_eudr_lcv_audit_reports.audit_result IS 'Audit outcome: pass, conditional_pass (minor NCs), fail (major NCs), pending_corrective_action, suspended, withdrawn';


-- ============================================================================
-- 7. gl_eudr_lcv_legal_requirements — Country-specific requirements
-- ============================================================================
RAISE NOTICE 'V111 [7/15]: Creating gl_eudr_lcv_legal_requirements...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_legal_requirements (
    requirement_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    framework_id                UUID            NOT NULL REFERENCES gl_eudr_lcv_legal_frameworks(framework_id),
        -- Parent legal framework
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    requirement_code            VARCHAR(50)     NOT NULL,
        -- Unique requirement identifier within the framework
    requirement_title           VARCHAR(500)    NOT NULL,
        -- Short title of the requirement
    requirement_description     TEXT,
        -- Full description of the legal requirement
    legislation_category        VARCHAR(50)     NOT NULL,
        -- Category this requirement falls under
    mandatory                   BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether compliance is mandatory or advisory
    enforcement_status          VARCHAR(50)     NOT NULL DEFAULT 'active',
        -- Current enforcement status
    penalty_description         TEXT,
        -- Description of penalties for non-compliance
    metadata                    JSONB           DEFAULT '{}',
        -- Additional context (exceptions, thresholds, commodities_affected, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_req_category CHECK (legislation_category IN (
        'forest_protection', 'land_use_rights', 'environmental_protection',
        'indigenous_rights', 'anti_corruption', 'trade_customs',
        'labor_rights', 'tax_fiscal'
    )),
    CONSTRAINT chk_lcv_req_enforcement CHECK (enforcement_status IN (
        'active', 'suspended', 'repealed', 'pending_enactment',
        'under_amendment', 'transitional'
    )),
    CONSTRAINT uq_lcv_req_code UNIQUE (framework_id, requirement_code)
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_framework ON gl_eudr_lcv_legal_requirements (framework_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_country ON gl_eudr_lcv_legal_requirements (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_code ON gl_eudr_lcv_legal_requirements (requirement_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_category ON gl_eudr_lcv_legal_requirements (legislation_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_mandatory ON gl_eudr_lcv_legal_requirements (mandatory);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_enforcement ON gl_eudr_lcv_legal_requirements (enforcement_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_tenant ON gl_eudr_lcv_legal_requirements (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_created ON gl_eudr_lcv_legal_requirements (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_country_cat ON gl_eudr_lcv_legal_requirements (country_code, legislation_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active mandatory requirements
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_active_mandatory ON gl_eudr_lcv_legal_requirements (country_code, legislation_category)
        WHERE mandatory = TRUE AND enforcement_status = 'active';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_req_metadata ON gl_eudr_lcv_legal_requirements USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_legal_requirements IS 'Granular legal requirements derived from country frameworks with enforcement status, mandatory classification, and penalty information';
COMMENT ON COLUMN gl_eudr_lcv_legal_requirements.enforcement_status IS 'Enforcement: active (currently enforced), suspended (temporarily halted), repealed, pending_enactment, under_amendment, transitional';


-- ============================================================================
-- 8. gl_eudr_lcv_compliance_reports — Generated compliance reports
-- ============================================================================
RAISE NOTICE 'V111 [8/15]: Creating gl_eudr_lcv_compliance_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_compliance_reports (
    report_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id               UUID            REFERENCES gl_eudr_lcv_compliance_assessments(assessment_id),
        -- Parent assessment (nullable for standalone reports)
    supplier_id                 UUID            NOT NULL,
        -- Supplier covered by the report
    report_type                 VARCHAR(50)     NOT NULL,
        -- Type of compliance report
    compliance_status           VARCHAR(30)     NOT NULL,
        -- Overall compliance determination
    executive_summary           TEXT,
        -- High-level summary of findings
    findings                    JSONB           DEFAULT '[]',
        -- [{ "finding": "...", "category": "...", "severity": "high", "reference": "EUDR Art.3" }, ...]
    recommendations             JSONB           DEFAULT '[]',
        -- [{ "action": "...", "priority": "urgent", "deadline": "2026-06-01" }, ...]
    report_format               VARCHAR(20)     NOT NULL DEFAULT 'json',
        -- Output format of the report
    report_url                  TEXT,
        -- URL or storage path for the generated report
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Report generation timestamp
    tenant_id                   UUID            NOT NULL,

    CONSTRAINT chk_lcv_rpt_type CHECK (report_type IN (
        'due_diligence_statement', 'annual_compliance_review',
        'incident_report', 'remediation_plan', 'regulatory_submission',
        'board_summary', 'supplier_scorecard'
    )),
    CONSTRAINT chk_lcv_rpt_compliance CHECK (compliance_status IN (
        'compliant', 'non_compliant', 'partially_compliant',
        'under_review', 'remediation_required'
    )),
    CONSTRAINT chk_lcv_rpt_format CHECK (report_format IN (
        'json', 'pdf', 'html', 'csv', 'xml'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_assessment ON gl_eudr_lcv_compliance_reports (assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_supplier ON gl_eudr_lcv_compliance_reports (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_type ON gl_eudr_lcv_compliance_reports (report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_compliance ON gl_eudr_lcv_compliance_reports (compliance_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_format ON gl_eudr_lcv_compliance_reports (report_format);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_generated ON gl_eudr_lcv_compliance_reports (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_tenant ON gl_eudr_lcv_compliance_reports (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_supplier_date ON gl_eudr_lcv_compliance_reports (supplier_id, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for non-compliant reports
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_non_compliant ON gl_eudr_lcv_compliance_reports (generated_at DESC)
        WHERE compliance_status IN ('non_compliant', 'remediation_required');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_findings ON gl_eudr_lcv_compliance_reports USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rpt_recommendations ON gl_eudr_lcv_compliance_reports USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_compliance_reports IS 'Generated legal compliance reports with findings, recommendations, and compliance determination for due diligence statements and regulatory submissions';
COMMENT ON COLUMN gl_eudr_lcv_compliance_reports.report_type IS '7 report types: due_diligence_statement, annual_compliance_review, incident_report, remediation_plan, regulatory_submission, board_summary, supplier_scorecard';


-- ============================================================================
-- 9. gl_eudr_lcv_country_compliance_mappings — Country rule sets
-- ============================================================================
RAISE NOTICE 'V111 [9/15]: Creating gl_eudr_lcv_country_compliance_mappings...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_country_compliance_mappings (
    mapping_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    commodity_type              VARCHAR(50)     NOT NULL,
        -- EUDR commodity type
    legislation_category        VARCHAR(50)     NOT NULL,
        -- Legislation category
    required_documents          JSONB           NOT NULL DEFAULT '[]',
        -- [{ "document_type": "land_title", "mandatory": true, "alternatives": [] }, ...]
    required_certifications     JSONB           DEFAULT '[]',
        -- [{ "scheme": "FSC", "required": false, "eudr_equivalent": true }, ...]
    compliance_rules            JSONB           NOT NULL DEFAULT '{}',
        -- { "max_gap_days": 30, "min_documents": 3, "auto_fail_conditions": [...] }
    risk_level                  VARCHAR(20)     NOT NULL DEFAULT 'standard',
        -- Country-commodity risk classification
    effective_from              DATE            NOT NULL DEFAULT CURRENT_DATE,
        -- Date these mappings become effective
    effective_to                DATE,
        -- Date these mappings expire (null for current)
    metadata                    JSONB           DEFAULT '{}',
        -- Additional context (regulatory_body, update_notes, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_ccm_commodity CHECK (commodity_type IN (
        'palm_oil', 'soy', 'coffee', 'cocoa', 'rubber',
        'timber', 'cattle', 'derived_products'
    )),
    CONSTRAINT chk_lcv_ccm_category CHECK (legislation_category IN (
        'forest_protection', 'land_use_rights', 'environmental_protection',
        'indigenous_rights', 'anti_corruption', 'trade_customs',
        'labor_rights', 'tax_fiscal'
    )),
    CONSTRAINT chk_lcv_ccm_risk CHECK (risk_level IN (
        'high', 'standard', 'low'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_country ON gl_eudr_lcv_country_compliance_mappings (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_commodity ON gl_eudr_lcv_country_compliance_mappings (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_category ON gl_eudr_lcv_country_compliance_mappings (legislation_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_risk ON gl_eudr_lcv_country_compliance_mappings (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_effective ON gl_eudr_lcv_country_compliance_mappings (effective_from DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_tenant ON gl_eudr_lcv_country_compliance_mappings (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_created ON gl_eudr_lcv_country_compliance_mappings (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_country_commodity ON gl_eudr_lcv_country_compliance_mappings (country_code, commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_country_cat ON gl_eudr_lcv_country_compliance_mappings (country_code, legislation_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for current active mappings
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_current ON gl_eudr_lcv_country_compliance_mappings (country_code, commodity_type, legislation_category)
        WHERE effective_to IS NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_req_docs ON gl_eudr_lcv_country_compliance_mappings USING GIN (required_documents);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_req_certs ON gl_eudr_lcv_country_compliance_mappings USING GIN (required_certifications);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_rules ON gl_eudr_lcv_country_compliance_mappings USING GIN (compliance_rules);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ccm_metadata ON gl_eudr_lcv_country_compliance_mappings USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_country_compliance_mappings IS 'Country-commodity-category compliance rule sets defining required documents, certifications, and compliance rules with EU risk benchmarking';
COMMENT ON COLUMN gl_eudr_lcv_country_compliance_mappings.risk_level IS 'EU benchmarking risk level per EUDR Article 29: high (enhanced due diligence), standard, low (simplified)';


-- ============================================================================
-- 10. gl_eudr_lcv_batch_operations — Batch processing tracking
-- ============================================================================
RAISE NOTICE 'V111 [10/15]: Creating gl_eudr_lcv_batch_operations...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_batch_operations (
    batch_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_type              VARCHAR(50)     NOT NULL,
        -- Type of batch operation
    status                      VARCHAR(30)     NOT NULL DEFAULT 'pending',
        -- Current batch status
    total_items                 INTEGER         NOT NULL DEFAULT 0 CHECK (total_items >= 0),
        -- Total items in the batch
    processed_items             INTEGER         NOT NULL DEFAULT 0 CHECK (processed_items >= 0),
        -- Items successfully processed
    failed_items                INTEGER         NOT NULL DEFAULT 0 CHECK (failed_items >= 0),
        -- Items that failed processing
    error_details               JSONB           DEFAULT '[]',
        -- [{ "item_id": "...", "error": "...", "timestamp": "..." }, ...]
    started_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Batch start timestamp
    completed_at                TIMESTAMPTZ,
        -- Batch completion timestamp
    initiated_by                VARCHAR(200),
        -- User or system that initiated the batch
    metadata                    JSONB           DEFAULT '{}',
        -- Additional batch context (parameters, filters, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_batch_op_type CHECK (operation_type IN (
        'bulk_assessment', 'document_verification', 'certification_check',
        'red_flag_scan', 'framework_update', 'report_generation',
        'data_import', 'data_export'
    )),
    CONSTRAINT chk_lcv_batch_status CHECK (status IN (
        'pending', 'running', 'completed', 'failed',
        'cancelled', 'partially_completed'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_op_type ON gl_eudr_lcv_batch_operations (operation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_status ON gl_eudr_lcv_batch_operations (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_started ON gl_eudr_lcv_batch_operations (started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_completed ON gl_eudr_lcv_batch_operations (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_initiated ON gl_eudr_lcv_batch_operations (initiated_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_tenant ON gl_eudr_lcv_batch_operations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_created ON gl_eudr_lcv_batch_operations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active batches
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_active ON gl_eudr_lcv_batch_operations (operation_type, started_at DESC)
        WHERE status IN ('pending', 'running');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_errors ON gl_eudr_lcv_batch_operations USING GIN (error_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_batch_metadata ON gl_eudr_lcv_batch_operations USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_batch_operations IS 'Batch processing tracker for bulk assessments, document verifications, red flag scans, and other bulk operations with progress and error tracking';
COMMENT ON COLUMN gl_eudr_lcv_batch_operations.operation_type IS '8 batch types: bulk_assessment, document_verification, certification_check, red_flag_scan, framework_update, report_generation, data_import, data_export';


-- ============================================================================
-- 11. gl_eudr_lcv_legal_opinions — Legal opinion storage
-- ============================================================================
RAISE NOTICE 'V111 [11/15]: Creating gl_eudr_lcv_legal_opinions...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_legal_opinions (
    opinion_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    framework_id                UUID            REFERENCES gl_eudr_lcv_legal_frameworks(framework_id),
        -- Legal framework this opinion relates to (nullable for general opinions)
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    opinion_type                VARCHAR(50)     NOT NULL,
        -- Classification of the legal opinion
    subject                     VARCHAR(500)    NOT NULL,
        -- Brief subject line of the opinion
    opinion_text                TEXT            NOT NULL,
        -- Full text of the legal opinion
    legal_counsel               VARCHAR(300)    NOT NULL,
        -- Name of the legal counsel or firm providing the opinion
    opinion_date                DATE            NOT NULL,
        -- Date the opinion was issued
    validity_period_months      INTEGER         CHECK (validity_period_months > 0),
        -- Number of months the opinion remains valid
    confidentiality_level       VARCHAR(20)     NOT NULL DEFAULT 'internal',
        -- Access classification
    applicable_commodities      JSONB           DEFAULT '[]',
        -- ["palm_oil", "timber", "cocoa", ...]
    metadata                    JSONB           DEFAULT '{}',
        -- Additional context (case_references, disclaimers, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_lcv_opin_type CHECK (opinion_type IN (
        'regulatory_interpretation', 'compliance_guidance',
        'risk_assessment', 'enforcement_advisory',
        'dispute_resolution', 'legislative_update'
    )),
    CONSTRAINT chk_lcv_opin_confidentiality CHECK (confidentiality_level IN (
        'public', 'internal', 'confidential', 'privileged'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_framework ON gl_eudr_lcv_legal_opinions (framework_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_country ON gl_eudr_lcv_legal_opinions (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_type ON gl_eudr_lcv_legal_opinions (opinion_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_counsel ON gl_eudr_lcv_legal_opinions (legal_counsel);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_date ON gl_eudr_lcv_legal_opinions (opinion_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_confidentiality ON gl_eudr_lcv_legal_opinions (confidentiality_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_tenant ON gl_eudr_lcv_legal_opinions (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_created ON gl_eudr_lcv_legal_opinions (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_country_type ON gl_eudr_lcv_legal_opinions (country_code, opinion_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_commodities ON gl_eudr_lcv_legal_opinions USING GIN (applicable_commodities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_opin_metadata ON gl_eudr_lcv_legal_opinions USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_legal_opinions IS 'Legal opinion storage for regulatory interpretations, compliance guidance, and enforcement advisories with confidentiality classification';
COMMENT ON COLUMN gl_eudr_lcv_legal_opinions.opinion_type IS '6 types: regulatory_interpretation, compliance_guidance, risk_assessment, enforcement_advisory, dispute_resolution, legislative_update';


-- ============================================================================
-- 12. gl_eudr_lcv_verification_log — Verification audit trail (hypertable)
-- ============================================================================
RAISE NOTICE 'V111 [12/15]: Creating gl_eudr_lcv_verification_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_verification_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    verified_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the verification event (partition key)
    document_id                 UUID,
        -- Reference to the document being verified (nullable for non-document verifications)
    certification_id            UUID,
        -- Reference to the certification being validated (nullable)
    supplier_id                 UUID            NOT NULL,
        -- Supplier being verified
    verification_type           VARCHAR(50)     NOT NULL,
        -- Type of verification performed
    previous_status             VARCHAR(50),
        -- Status before this verification
    new_status                  VARCHAR(50)     NOT NULL,
        -- Status after this verification
    verifier                    VARCHAR(200)    NOT NULL,
        -- User or system agent performing the verification
    verification_method         VARCHAR(50),
        -- Method used for verification
    confidence                  NUMERIC(3,2)    CHECK (confidence >= 0 AND confidence <= 1),
        -- Verification confidence score (0.0 to 1.0)
    notes                       TEXT,
        -- Notes accompanying the verification
    evidence                    JSONB           DEFAULT '{}',
        -- Supporting evidence for the verification decision
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for audit trail integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, verified_at),

    CONSTRAINT chk_lcv_vl_type CHECK (verification_type IN (
        'document_authenticity', 'document_validity', 'certification_check',
        'authority_confirmation', 'cross_reference', 'expiry_check',
        'red_flag_screening', 'sanctions_screening', 'periodic_revalidation'
    )),
    CONSTRAINT chk_lcv_vl_method CHECK (verification_method IS NULL OR verification_method IN (
        'manual_review', 'automated_check', 'api_validation',
        'third_party_verification', 'blockchain_verification',
        'registry_lookup', 'field_verification'
    ))
);

SELECT create_hypertable(
    'gl_eudr_lcv_verification_log',
    'verified_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_document ON gl_eudr_lcv_verification_log (document_id, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_certification ON gl_eudr_lcv_verification_log (certification_id, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_supplier ON gl_eudr_lcv_verification_log (supplier_id, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_type ON gl_eudr_lcv_verification_log (verification_type, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_new_status ON gl_eudr_lcv_verification_log (new_status, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_verifier ON gl_eudr_lcv_verification_log (verifier, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_method ON gl_eudr_lcv_verification_log (verification_method, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_confidence ON gl_eudr_lcv_verification_log (confidence DESC, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_provenance ON gl_eudr_lcv_verification_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_tenant ON gl_eudr_lcv_verification_log (tenant_id, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_vl_evidence ON gl_eudr_lcv_verification_log USING GIN (evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_verification_log IS 'Immutable audit log for all document and certification verification events with verification method, confidence, and provenance tracking';
COMMENT ON COLUMN gl_eudr_lcv_verification_log.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- 13. gl_eudr_lcv_red_flag_events — Red flag event stream (hypertable)
-- ============================================================================
RAISE NOTICE 'V111 [13/15]: Creating gl_eudr_lcv_red_flag_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_red_flag_events (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    event_timestamp             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)
    flag_id                     UUID            NOT NULL,
        -- Reference to the parent red flag alert
    event_type                  VARCHAR(50)     NOT NULL,
        -- Type of red flag lifecycle event
    actor                       VARCHAR(200)    NOT NULL,
        -- User or system agent performing the action
    previous_status             VARCHAR(50),
        -- Status before this event
    new_status                  VARCHAR(50)     NOT NULL,
        -- Status after this event
    details                     JSONB           DEFAULT '{}',
        -- { "changed_fields": [...], "reason": "...", "attachments": [...] }
    ip_address                  INET,
        -- Source IP address of the actor
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for event integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, event_timestamp),

    CONSTRAINT chk_lcv_rfe_event_type CHECK (event_type IN (
        'detected', 'acknowledged', 'investigation_started',
        'evidence_added', 'severity_updated', 'escalated',
        'remediation_started', 'remediation_completed', 'resolved',
        'dismissed', 'reopened', 'suppressed', 'unsuppressed',
        'comment_added'
    ))
);

SELECT create_hypertable(
    'gl_eudr_lcv_red_flag_events',
    'event_timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rfe_flag ON gl_eudr_lcv_red_flag_events (flag_id, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rfe_type ON gl_eudr_lcv_red_flag_events (event_type, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rfe_actor ON gl_eudr_lcv_red_flag_events (actor, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rfe_new_status ON gl_eudr_lcv_red_flag_events (new_status, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rfe_provenance ON gl_eudr_lcv_red_flag_events (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rfe_tenant ON gl_eudr_lcv_red_flag_events (tenant_id, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_rfe_details ON gl_eudr_lcv_red_flag_events USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_red_flag_events IS 'Immutable event stream for red flag lifecycle tracking with actor, status transitions, suppression/unsuppression events, and provenance hashing';
COMMENT ON COLUMN gl_eudr_lcv_red_flag_events.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- 14. gl_eudr_lcv_compliance_history — Compliance score history (hypertable)
-- ============================================================================
RAISE NOTICE 'V111 [14/15]: Creating gl_eudr_lcv_compliance_history (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_compliance_history (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Assessment timestamp (partition key)
    supplier_id                 UUID            NOT NULL,
        -- Assessed supplier
    country_code                CHAR(2)         NOT NULL,
        -- Country context for the assessment
    overall_score               NUMERIC(5,2)    CHECK (overall_score >= 0 AND overall_score <= 100),
        -- Composite compliance score (0-100)
    overall_status              VARCHAR(30)     NOT NULL,
        -- Compliance determination
    category_scores             JSONB           DEFAULT '{}',
        -- { "forest_protection": 95, "land_use_rights": 70, ... }
    red_flags_count             INTEGER         NOT NULL DEFAULT 0 CHECK (red_flags_count >= 0),
        -- Red flags detected at this point in time
    documents_verified          INTEGER         NOT NULL DEFAULT 0 CHECK (documents_verified >= 0),
        -- Number of documents verified
    certifications_valid        INTEGER         NOT NULL DEFAULT 0 CHECK (certifications_valid >= 0),
        -- Number of valid certifications
    assessment_method           VARCHAR(50),
        -- Assessment method used
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for history integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, assessed_at),

    CONSTRAINT chk_lcv_ch_status CHECK (overall_status IN (
        'compliant', 'non_compliant', 'partially_compliant',
        'under_review', 'remediation_required', 'insufficient_data'
    ))
);

SELECT create_hypertable(
    'gl_eudr_lcv_compliance_history',
    'assessed_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_supplier ON gl_eudr_lcv_compliance_history (supplier_id, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_country ON gl_eudr_lcv_compliance_history (country_code, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_score ON gl_eudr_lcv_compliance_history (overall_score DESC, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_status ON gl_eudr_lcv_compliance_history (overall_status, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_red_flags ON gl_eudr_lcv_compliance_history (red_flags_count DESC, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_method ON gl_eudr_lcv_compliance_history (assessment_method, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_provenance ON gl_eudr_lcv_compliance_history (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_tenant ON gl_eudr_lcv_compliance_history (tenant_id, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_ch_cat_scores ON gl_eudr_lcv_compliance_history USING GIN (category_scores);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_compliance_history IS 'Time-series compliance score history per supplier with per-category breakdowns, red flag counts, and document/certification statistics';
COMMENT ON COLUMN gl_eudr_lcv_compliance_history.provenance_hash IS 'SHA-256 hash for history integrity and audit trail verification';


-- ============================================================================
-- 15. gl_eudr_lcv_audit_log — General audit log (hypertable)
-- ============================================================================
RAISE NOTICE 'V111 [15/15]: Creating gl_eudr_lcv_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_lcv_audit_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    timestamp                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)
    action                      VARCHAR(50)     NOT NULL,
        -- Action performed
    entity_type                 VARCHAR(50)     NOT NULL,
        -- Type of entity affected
    entity_id                   UUID            NOT NULL,
        -- ID of the affected entity
    actor                       VARCHAR(200)    NOT NULL,
        -- User or system agent performing the action
    actor_role                  VARCHAR(50),
        -- Role of the actor
    previous_state              JSONB           DEFAULT '{}',
        -- State before the action
    new_state                   JSONB           DEFAULT '{}',
        -- State after the action
    change_summary              TEXT,
        -- Human-readable summary of what changed
    ip_address                  INET,
        -- Source IP address
    user_agent                  TEXT,
        -- Client user agent string
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for audit integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, timestamp),

    CONSTRAINT chk_lcv_al_action CHECK (action IN (
        'create', 'update', 'delete', 'archive',
        'verify', 'reject', 'approve', 'escalate',
        'suppress', 'restore', 'export', 'import',
        'login', 'logout', 'access', 'permission_change'
    )),
    CONSTRAINT chk_lcv_al_entity CHECK (entity_type IN (
        'legal_framework', 'compliance_document', 'certification_record',
        'red_flag_alert', 'compliance_assessment', 'audit_report',
        'legal_requirement', 'compliance_report', 'country_mapping',
        'batch_operation', 'legal_opinion', 'user', 'system'
    ))
);

SELECT create_hypertable(
    'gl_eudr_lcv_audit_log',
    'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_action ON gl_eudr_lcv_audit_log (action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_entity_type ON gl_eudr_lcv_audit_log (entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_entity_id ON gl_eudr_lcv_audit_log (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_actor ON gl_eudr_lcv_audit_log (actor, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_actor_role ON gl_eudr_lcv_audit_log (actor_role, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_ip ON gl_eudr_lcv_audit_log (ip_address, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_provenance ON gl_eudr_lcv_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_tenant ON gl_eudr_lcv_audit_log (tenant_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_entity_action ON gl_eudr_lcv_audit_log (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_prev_state ON gl_eudr_lcv_audit_log USING GIN (previous_state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_lcv_al_new_state ON gl_eudr_lcv_audit_log USING GIN (new_state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_lcv_audit_log IS 'General immutable audit log for all entity lifecycle events in the Legal Compliance Verifier with full state tracking and provenance';
COMMENT ON COLUMN gl_eudr_lcv_audit_log.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Monthly compliance summary by country and category
RAISE NOTICE 'V111: Creating continuous aggregate: monthly_compliance_summary...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_lcv_monthly_compliance_summary
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('30 days', assessed_at)     AS month,
        tenant_id,
        country_code,
        overall_status,
        COUNT(*)                                AS assessment_count,
        AVG(overall_score)                      AS avg_score,
        MIN(overall_score)                      AS min_score,
        MAX(overall_score)                      AS max_score,
        SUM(red_flags_count)                    AS total_red_flags,
        SUM(documents_verified)                 AS total_docs_verified,
        SUM(certifications_valid)               AS total_certs_valid,
        COUNT(DISTINCT supplier_id)             AS unique_suppliers
    FROM gl_eudr_lcv_compliance_history
    GROUP BY month, tenant_id, country_code, overall_status;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_lcv_monthly_compliance_summary',
        start_offset => INTERVAL '60 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_lcv_monthly_compliance_summary IS 'Monthly rollup of compliance assessments by country and status with score statistics, red flag totals, and supplier counts';


-- Weekly red flag summary by category and severity
RAISE NOTICE 'V111: Creating continuous aggregate: weekly_red_flag_summary...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_lcv_weekly_red_flag_summary
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('7 days', event_timestamp)  AS week,
        tenant_id,
        event_type,
        new_status,
        COUNT(*)                                AS event_count,
        COUNT(DISTINCT flag_id)                 AS unique_flags
    FROM gl_eudr_lcv_red_flag_events
    GROUP BY week, tenant_id, event_type, new_status;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_lcv_weekly_red_flag_summary',
        start_offset => INTERVAL '14 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_lcv_weekly_red_flag_summary IS 'Weekly rollup of red flag events by event type and status with event counts and unique flag tallies';


-- ============================================================================
-- RETENTION POLICIES (5 years per EUDR Article 31)
-- ============================================================================

RAISE NOTICE 'V111: Creating retention policies (5 years per EUDR Article 31)...';

-- 5 years for verification logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_lcv_verification_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for red flag events
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_lcv_red_flag_events', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for compliance history
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_lcv_compliance_history', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for audit log
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_lcv_audit_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- GRANTS — greenlang_app role
-- ============================================================================

RAISE NOTICE 'V111: Granting permissions to greenlang_app...';

-- Regular tables
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_legal_frameworks TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_compliance_documents TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_certification_records TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_red_flag_alerts TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_compliance_assessments TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_audit_reports TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_legal_requirements TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_compliance_reports TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_country_compliance_mappings TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_batch_operations TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_lcv_legal_opinions TO greenlang_app;

-- Hypertables (append-only for audit integrity)
GRANT SELECT, INSERT ON gl_eudr_lcv_verification_log TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_lcv_red_flag_events TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_lcv_compliance_history TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_lcv_audit_log TO greenlang_app;

-- Continuous aggregates
GRANT SELECT ON gl_eudr_lcv_monthly_compliance_summary TO greenlang_app;
GRANT SELECT ON gl_eudr_lcv_weekly_red_flag_summary TO greenlang_app;

-- Read-only role (conditional)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT SELECT ON gl_eudr_lcv_legal_frameworks TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_compliance_documents TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_certification_records TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_red_flag_alerts TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_compliance_assessments TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_audit_reports TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_legal_requirements TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_compliance_reports TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_country_compliance_mappings TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_batch_operations TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_legal_opinions TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_verification_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_red_flag_events TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_compliance_history TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_audit_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_monthly_compliance_summary TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_lcv_weekly_red_flag_summary TO greenlang_readonly;
    END IF;
END
$$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V111: AGENT-EUDR-023 Legal Compliance Verifier tables created successfully!';
RAISE NOTICE 'V111: Created 15 tables (11 regular + 4 hypertables), 2 continuous aggregates, ~160 indexes';
RAISE NOTICE 'V111: ~20 GIN indexes on JSONB columns, ~10 partial indexes for active/filtered records';
RAISE NOTICE 'V111: Retention policies: 5y on all 4 hypertables per EUDR Article 31';
RAISE NOTICE 'V111: Grants applied for greenlang_app and greenlang_readonly roles';

COMMIT;

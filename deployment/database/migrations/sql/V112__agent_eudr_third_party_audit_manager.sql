-- ============================================================================
-- V112: AGENT-EUDR-024 Third-Party Audit Manager Agent
-- ============================================================================
-- Creates tables for the Third-Party Audit Manager agent which provides
-- end-to-end management of the third-party audit lifecycle for EUDR
-- compliance verification including risk-based audit planning, auditor
-- qualification tracking (ISO/IEC 17065/17021-1), audit execution with
-- EUDR-specific checklists, non-conformance detection and classification,
-- corrective action request (CAR) lifecycle management, certification
-- scheme integration (FSC/PEFC/RSPO/RA/ISCC), ISO 19011:2018 compliant
-- report generation, and competent authority liaison workflows.
--
-- Schema: eudr_third_party_audit (17 tables)
-- Tables: 17 (13 regular + 4 hypertables)
-- Hypertables: gl_eudr_tam_audit_schedule (90d chunks, time: scheduled_at),
--              gl_eudr_tam_nc_trend_log (30d chunks, time: recorded_at),
--              gl_eudr_tam_car_sla_log (30d chunks, time: changed_at),
--              gl_eudr_tam_audit_trail (30d chunks, time: recorded_at)
-- Continuous Aggregates: 2 (gl_eudr_tam_audit_schedule_daily,
--                           gl_eudr_tam_nc_trends_monthly)
-- Retention Policies: 4 (5 years per EUDR Article 31)
-- Compression Policies: 4 (on all hypertables)
-- Indexes: ~180 (B-tree, GIN for JSONB, partial indexes)
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V112: Creating AGENT-EUDR-024 Third-Party Audit Manager tables...';


-- ============================================================================
-- 1. gl_eudr_tam_audits — Core audit records
-- ============================================================================
RAISE NOTICE 'V112 [1/17]: Creating gl_eudr_tam_audits...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_audits (
    audit_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator owning this audit
    supplier_id                 UUID            NOT NULL,
        -- Supplier being audited
    audit_type                  VARCHAR(30)     NOT NULL DEFAULT 'full',
        -- Audit scope classification
    modality                    VARCHAR(30)     NOT NULL DEFAULT 'on_site',
        -- Audit delivery modality
    certification_scheme        VARCHAR(50),
        -- Associated certification scheme (nullable for EUDR-only audits)
    eudr_articles               JSONB           DEFAULT '[]',
        -- EUDR articles in audit scope (e.g. ["Art.3","Art.9","Art.10"])
    planned_date                DATE            NOT NULL,
        -- Scheduled audit date
    actual_start_date           DATE,
        -- Actual start date of audit fieldwork
    actual_end_date             DATE,
        -- Actual completion date of audit fieldwork
    lead_auditor_id             UUID,
        -- Assigned lead auditor (FK reference to auditors table)
    audit_team                  JSONB           DEFAULT '[]',
        -- Audit team member IDs and roles
    status                      VARCHAR(30)     NOT NULL DEFAULT 'planned',
        -- Current audit lifecycle status
    priority_score              NUMERIC(5,2)    DEFAULT 0.0,
        -- Risk-based audit priority score (0-100)
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code of audited site
    commodity                   VARCHAR(50)     NOT NULL,
        -- Primary commodity under audit scope
    site_ids                    JSONB           DEFAULT '[]',
        -- Identifiers of audited production/processing sites
    checklist_completion        NUMERIC(5,2)    DEFAULT 0.0,
        -- Percentage of audit checklist criteria assessed (0-100)
    findings_count              JSONB           DEFAULT '{"critical":0,"major":0,"minor":0,"observation":0}',
        -- Count of findings by severity classification
    evidence_count              INTEGER         DEFAULT 0,
        -- Total number of evidence items collected
    report_id                   UUID,
        -- Generated audit report ID (nullable until report generated)
    trigger_reason              VARCHAR(200),
        -- For unscheduled audits: reason for trigger (deforestation alert, cert suspension, etc.)
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for audit record integrity verification
    metadata                    JSONB           DEFAULT '{}',
        -- Additional audit attributes (notes, tags, cost, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_audit_type CHECK (audit_type IN (
        'full', 'targeted', 'surveillance', 'unscheduled'
    )),
    CONSTRAINT chk_tam_audit_modality CHECK (modality IN (
        'on_site', 'remote', 'hybrid', 'unannounced'
    )),
    CONSTRAINT chk_tam_audit_status CHECK (status IN (
        'planned', 'auditor_assigned', 'in_preparation', 'in_progress',
        'fieldwork_complete', 'report_drafting', 'report_issued',
        'car_follow_up', 'closed', 'cancelled'
    )),
    CONSTRAINT chk_tam_audit_scheme CHECK (certification_scheme IS NULL OR certification_scheme IN (
        'fsc', 'pefc', 'rspo', 'rainforest_alliance', 'iscc'
    )),
    CONSTRAINT chk_tam_audit_priority CHECK (priority_score >= 0 AND priority_score <= 100),
    CONSTRAINT chk_tam_audit_checklist CHECK (checklist_completion >= 0 AND checklist_completion <= 100)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_operator ON gl_eudr_tam_audits (operator_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_supplier ON gl_eudr_tam_audits (supplier_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_status ON gl_eudr_tam_audits (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_planned ON gl_eudr_tam_audits (planned_date DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_country ON gl_eudr_tam_audits (country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_commodity ON gl_eudr_tam_audits (commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_scheme ON gl_eudr_tam_audits (certification_scheme); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_type ON gl_eudr_tam_audits (audit_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_modality ON gl_eudr_tam_audits (modality); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_lead ON gl_eudr_tam_audits (lead_auditor_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_priority ON gl_eudr_tam_audits (priority_score DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_provenance ON gl_eudr_tam_audits (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_tenant ON gl_eudr_tam_audits (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_created ON gl_eudr_tam_audits (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_op_status ON gl_eudr_tam_audits (operator_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_sup_planned ON gl_eudr_tam_audits (supplier_id, planned_date DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_country_scheme ON gl_eudr_tam_audits (country_code, certification_scheme); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active audits (not closed or cancelled)
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_aud_active ON gl_eudr_tam_audits (operator_id, status, priority_score DESC)
        WHERE status NOT IN ('closed', 'cancelled');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_articles ON gl_eudr_tam_audits USING GIN (eudr_articles); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_team ON gl_eudr_tam_audits USING GIN (audit_team); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_sites ON gl_eudr_tam_audits USING GIN (site_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_findings ON gl_eudr_tam_audits USING GIN (findings_count); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_aud_metadata ON gl_eudr_tam_audits USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_audits IS 'Core audit records for the Third-Party Audit Manager agent managing the full audit lifecycle from planning through closure with risk-based prioritization and multi-scheme support';
COMMENT ON COLUMN gl_eudr_tam_audits.audit_type IS 'Audit scope: full (all EUDR + scheme criteria), targeted (specific risk areas), surveillance (maintenance), unscheduled (event-triggered)';
COMMENT ON COLUMN gl_eudr_tam_audits.modality IS 'Delivery modality: on_site (physical), remote (desktop), hybrid (remote + on-site), unannounced (no advance notice)';
COMMENT ON COLUMN gl_eudr_tam_audits.priority_score IS 'Risk-based priority score (0-100) calculated from country risk, supplier risk, NC history, certification gap, and deforestation alerts';


-- ============================================================================
-- 2. gl_eudr_tam_auditors — Auditor registry
-- ============================================================================
RAISE NOTICE 'V112 [2/17]: Creating gl_eudr_tam_auditors...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_auditors (
    auditor_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name                   VARCHAR(500)    NOT NULL,
        -- Auditor full legal name
    organization                VARCHAR(500)    NOT NULL,
        -- Employing certification body or audit firm
    accreditation_body          VARCHAR(200),
        -- IAF MLA signatory accreditation body
    accreditation_status        VARCHAR(30)     NOT NULL DEFAULT 'active',
        -- Current accreditation status
    accreditation_expiry        DATE,
        -- Date accreditation expires
    accreditation_scope         JSONB           DEFAULT '[]',
        -- Accreditation scope details
    commodity_competencies      JSONB           DEFAULT '[]',
        -- EUDR commodities qualified for audit (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)
    scheme_qualifications       JSONB           DEFAULT '[]',
        -- Scheme-specific qualifications (FSC Lead Auditor, RSPO Lead Assessor, etc.)
    country_expertise           JSONB           DEFAULT '[]',
        -- Countries of expertise as ISO 3166-1 alpha-2 codes
    languages                   JSONB           DEFAULT '[]',
        -- Language proficiencies as ISO 639-1 codes
    conflict_of_interest        JSONB           DEFAULT '[]',
        -- Conflict of interest declarations with dates and entities
    audit_count                 INTEGER         DEFAULT 0,
        -- Total number of audits conducted
    performance_rating          NUMERIC(5,2)    DEFAULT 0.0,
        -- Performance rating (0-100) based on finding accuracy and client feedback
    cpd_hours                   INTEGER         DEFAULT 0,
        -- Continuing professional development hours completed
    cpd_compliant               BOOLEAN         DEFAULT TRUE,
        -- Whether CPD requirements are met per accreditation body rules
    contact_email               VARCHAR(500),
        -- Encrypted contact email (AES-256 via SEC-003)
    metadata                    JSONB           DEFAULT '{}',
        -- Additional auditor attributes (certifications, training, availability)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_auditor_status CHECK (accreditation_status IN (
        'active', 'suspended', 'withdrawn', 'expired'
    )),
    CONSTRAINT chk_tam_auditor_rating CHECK (performance_rating >= 0 AND performance_rating <= 100),
    CONSTRAINT chk_tam_auditor_cpd CHECK (cpd_hours >= 0),
    CONSTRAINT chk_tam_auditor_count CHECK (audit_count >= 0)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_name ON gl_eudr_tam_auditors (full_name); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_org ON gl_eudr_tam_auditors (organization); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_accred_status ON gl_eudr_tam_auditors (accreditation_status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_accred_expiry ON gl_eudr_tam_auditors (accreditation_expiry); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_rating ON gl_eudr_tam_auditors (performance_rating DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_cpd ON gl_eudr_tam_auditors (cpd_compliant); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_tenant ON gl_eudr_tam_auditors (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_created ON gl_eudr_tam_auditors (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_status_rating ON gl_eudr_tam_auditors (accreditation_status, performance_rating DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active auditors available for assignment
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_adr_available ON gl_eudr_tam_auditors (performance_rating DESC, accreditation_expiry)
        WHERE accreditation_status = 'active' AND cpd_compliant = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_commodities ON gl_eudr_tam_auditors USING GIN (commodity_competencies); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_schemes ON gl_eudr_tam_auditors USING GIN (scheme_qualifications); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_countries ON gl_eudr_tam_auditors USING GIN (country_expertise); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_languages ON gl_eudr_tam_auditors USING GIN (languages); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_coi ON gl_eudr_tam_auditors USING GIN (conflict_of_interest); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_adr_metadata ON gl_eudr_tam_auditors USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_auditors IS 'Auditor registry with ISO/IEC 17065 and ISO/IEC 17021-1 competence tracking for qualification-based assignment to EUDR audits';
COMMENT ON COLUMN gl_eudr_tam_auditors.accreditation_status IS 'Accreditation status: active (current), suspended (temporarily halted), withdrawn (revoked), expired (lapsed)';
COMMENT ON COLUMN gl_eudr_tam_auditors.performance_rating IS 'Composite performance rating (0-100) from finding accuracy, CAR closure rate, duration efficiency, and client feedback';


-- ============================================================================
-- 3. gl_eudr_tam_audit_checklists — Audit checklists
-- ============================================================================
RAISE NOTICE 'V112 [3/17]: Creating gl_eudr_tam_audit_checklists...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_audit_checklists (
    checklist_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                    UUID            NOT NULL REFERENCES gl_eudr_tam_audits(audit_id),
        -- Parent audit
    checklist_type              VARCHAR(50)     NOT NULL,
        -- Type of checklist (eudr, fsc, pefc, rspo, rainforest_alliance, iscc)
    checklist_version           VARCHAR(20)     NOT NULL,
        -- Version of the checklist criteria (e.g. "2026.1")
    criteria                    JSONB           NOT NULL DEFAULT '[]',
        -- Array of audit criteria with results
    completion_percentage       NUMERIC(5,2)    DEFAULT 0.0,
        -- Overall checklist completion (0-100)
    total_criteria              INTEGER         DEFAULT 0,
        -- Total number of criteria in checklist
    passed_criteria             INTEGER         DEFAULT 0,
        -- Criteria assessed as pass
    failed_criteria             INTEGER         DEFAULT 0,
        -- Criteria assessed as fail
    na_criteria                 INTEGER         DEFAULT 0,
        -- Criteria assessed as not applicable
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_cl_type CHECK (checklist_type IN (
        'eudr', 'fsc', 'pefc', 'rspo', 'rainforest_alliance', 'iscc', 'combined'
    )),
    CONSTRAINT chk_tam_cl_completion CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    CONSTRAINT chk_tam_cl_counts CHECK (passed_criteria >= 0 AND failed_criteria >= 0 AND na_criteria >= 0 AND total_criteria >= 0)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_cl_audit ON gl_eudr_tam_audit_checklists (audit_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cl_type ON gl_eudr_tam_audit_checklists (checklist_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cl_version ON gl_eudr_tam_audit_checklists (checklist_version); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cl_completion ON gl_eudr_tam_audit_checklists (completion_percentage DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cl_tenant ON gl_eudr_tam_audit_checklists (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cl_created ON gl_eudr_tam_audit_checklists (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cl_audit_type ON gl_eudr_tam_audit_checklists (audit_id, checklist_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for incomplete checklists
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_cl_incomplete ON gl_eudr_tam_audit_checklists (audit_id, completion_percentage)
        WHERE completion_percentage < 100;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cl_criteria ON gl_eudr_tam_audit_checklists USING GIN (criteria); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_audit_checklists IS 'Audit checklists for EUDR-specific and certification scheme criteria with real-time completion tracking and pass/fail/NA assessment';
COMMENT ON COLUMN gl_eudr_tam_audit_checklists.checklist_type IS 'Checklist types: eudr (EUDR articles), fsc/pefc/rspo/rainforest_alliance/iscc (scheme-specific), combined (unified view)';


-- ============================================================================
-- 4. gl_eudr_tam_audit_evidence — Evidence items
-- ============================================================================
RAISE NOTICE 'V112 [4/17]: Creating gl_eudr_tam_audit_evidence...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_audit_evidence (
    evidence_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                    UUID            NOT NULL REFERENCES gl_eudr_tam_audits(audit_id),
        -- Parent audit
    evidence_type               VARCHAR(50)     NOT NULL,
        -- Classification of evidence item
    file_name                   VARCHAR(500),
        -- Original file name
    file_path                   VARCHAR(1000),
        -- S3 path (s3://gl-eudr-tam-evidence/{operator_id}/{audit_id}/{evidence_id})
    file_size_bytes             BIGINT,
        -- File size in bytes
    mime_type                   VARCHAR(100),
        -- MIME type of the uploaded file
    description                 TEXT,
        -- Evidence description and context
    tags                        JSONB           DEFAULT '[]',
        -- Categorization tags for search and filtering
    location_latitude           DOUBLE PRECISION,
        -- GPS latitude of evidence capture location
    location_longitude          DOUBLE PRECISION,
        -- GPS longitude of evidence capture location
    captured_date               TIMESTAMPTZ,
        -- Date and time the evidence was captured in the field
    sha256_hash                 VARCHAR(64)     NOT NULL,
        -- SHA-256 hash of file content for integrity verification
    uploaded_by                 VARCHAR(100),
        -- User or auditor who uploaded the evidence
    metadata                    JSONB           DEFAULT '{}',
        -- Additional evidence attributes (device, settings, chain_of_custody)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_ev_type CHECK (evidence_type IN (
        'permit', 'certificate', 'photo', 'gps_record',
        'interview_transcript', 'lab_result', 'document_scan', 'other'
    ))
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_audit ON gl_eudr_tam_audit_evidence (audit_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_type ON gl_eudr_tam_audit_evidence (evidence_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_hash ON gl_eudr_tam_audit_evidence (sha256_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_captured ON gl_eudr_tam_audit_evidence (captured_date DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_uploaded_by ON gl_eudr_tam_audit_evidence (uploaded_by); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_tenant ON gl_eudr_tam_audit_evidence (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_created ON gl_eudr_tam_audit_evidence (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_audit_type ON gl_eudr_tam_audit_evidence (audit_id, evidence_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_tags ON gl_eudr_tam_audit_evidence USING GIN (tags); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ev_metadata ON gl_eudr_tam_audit_evidence USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_audit_evidence IS 'Evidence items collected during audit execution with SHA-256 integrity hashing, GPS location, and type classification for EUDR compliance verification';
COMMENT ON COLUMN gl_eudr_tam_audit_evidence.evidence_type IS 'Evidence types: permit, certificate, photo, gps_record, interview_transcript, lab_result, document_scan, other';
COMMENT ON COLUMN gl_eudr_tam_audit_evidence.sha256_hash IS 'SHA-256 hash of file content computed on upload for integrity verification and tamper detection';
COMMENT ON COLUMN gl_eudr_tam_audit_evidence.file_path IS 'S3 storage path pattern: s3://gl-eudr-tam-evidence/{operator_id}/{audit_id}/{evidence_id}. AES-256-GCM encrypted at rest via SEC-003';


-- ============================================================================
-- 5. gl_eudr_tam_non_conformances — Non-conformance records
-- ============================================================================
RAISE NOTICE 'V112 [5/17]: Creating gl_eudr_tam_non_conformances...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_non_conformances (
    nc_id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                    UUID            NOT NULL REFERENCES gl_eudr_tam_audits(audit_id),
        -- Parent audit where NC was detected
    finding_statement           TEXT            NOT NULL,
        -- Description of the non-conformance finding
    objective_evidence          TEXT            NOT NULL,
        -- Objective evidence supporting the finding
    severity                    VARCHAR(20)     NOT NULL,
        -- NC severity classification per ISO 19011 and scheme conventions
    eudr_article                VARCHAR(20),
        -- Mapped EUDR article reference (e.g. "Art.9", "Art.10")
    scheme_clause               VARCHAR(100),
        -- Mapped certification scheme clause (e.g. "FSC P6 C6.1")
    article_2_40_category       VARCHAR(50),
        -- EUDR Article 2(40) legislation category (1-8)
    root_cause_analysis         JSONB,
        -- Root cause analysis output (5 Whys or Ishikawa structure)
    root_cause_method           VARCHAR(30),
        -- RCA method used
    risk_impact_score           NUMERIC(5,2)    DEFAULT 0.0,
        -- NC risk impact score (0-100)
    status                      VARCHAR(30)     NOT NULL DEFAULT 'open',
        -- NC lifecycle status
    car_id                      UUID,
        -- Linked corrective action request ID
    evidence_ids                JSONB           DEFAULT '[]',
        -- Linked evidence item IDs
    classification_rules_applied JSONB          DEFAULT '[]',
        -- Rule IDs applied during severity classification for audit trail
    disputed                    BOOLEAN         DEFAULT FALSE,
        -- Whether the NC has been disputed by the auditee
    dispute_rationale           TEXT,
        -- Justification for dispute (if disputed=true)
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for NC record integrity
    supplier_id                 UUID            NOT NULL,
        -- Supplier associated with this NC (denormalized for analytics)
    country_code                CHAR(2),
        -- Country where NC was detected (denormalized for analytics)
    commodity                   VARCHAR(50),
        -- Commodity associated with NC (denormalized for analytics)
    tenant_id                   UUID            NOT NULL,
    detected_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    resolved_at                 TIMESTAMPTZ,

    CONSTRAINT chk_tam_nc_severity CHECK (severity IN (
        'critical', 'major', 'minor', 'observation'
    )),
    CONSTRAINT chk_tam_nc_rca_method CHECK (root_cause_method IS NULL OR root_cause_method IN (
        'five_whys', 'ishikawa', 'direct'
    )),
    CONSTRAINT chk_tam_nc_status CHECK (status IN (
        'open', 'acknowledged', 'car_issued', 'cap_submitted',
        'in_progress', 'verification_pending', 'closed', 'escalated', 'disputed'
    )),
    CONSTRAINT chk_tam_nc_impact CHECK (risk_impact_score >= 0 AND risk_impact_score <= 100)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_audit ON gl_eudr_tam_non_conformances (audit_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_severity ON gl_eudr_tam_non_conformances (severity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_status ON gl_eudr_tam_non_conformances (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_eudr_art ON gl_eudr_tam_non_conformances (eudr_article); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_scheme_cl ON gl_eudr_tam_non_conformances (scheme_clause); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_art240 ON gl_eudr_tam_non_conformances (article_2_40_category); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_car ON gl_eudr_tam_non_conformances (car_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_supplier ON gl_eudr_tam_non_conformances (supplier_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_country ON gl_eudr_tam_non_conformances (country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_commodity ON gl_eudr_tam_non_conformances (commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_impact ON gl_eudr_tam_non_conformances (risk_impact_score DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_provenance ON gl_eudr_tam_non_conformances (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_tenant ON gl_eudr_tam_non_conformances (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_detected ON gl_eudr_tam_non_conformances (detected_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_audit_sev ON gl_eudr_tam_non_conformances (audit_id, severity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_status_sev ON gl_eudr_tam_non_conformances (status, severity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_sup_sev ON gl_eudr_tam_non_conformances (supplier_id, severity, detected_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for open NCs requiring action
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_nc_open ON gl_eudr_tam_non_conformances (severity, risk_impact_score DESC)
        WHERE status IN ('open', 'acknowledged', 'car_issued', 'cap_submitted', 'in_progress', 'escalated');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for disputed NCs
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_nc_disputed ON gl_eudr_tam_non_conformances (audit_id, severity)
        WHERE disputed = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_evidence ON gl_eudr_tam_non_conformances USING GIN (evidence_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_rules ON gl_eudr_tam_non_conformances USING GIN (classification_rules_applied); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_nc_rca ON gl_eudr_tam_non_conformances USING GIN (root_cause_analysis); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_non_conformances IS 'Non-conformance records with ISO 19011 severity classification (critical/major/minor/observation), EUDR article mapping, root cause analysis, and deterministic risk impact scoring';
COMMENT ON COLUMN gl_eudr_tam_non_conformances.severity IS 'Severity per ISO 19011: critical (30-day deadline), major (90-day), minor (365-day), observation (no CAR required)';
COMMENT ON COLUMN gl_eudr_tam_non_conformances.classification_rules_applied IS 'Deterministic classification rule IDs (CR-001, MJ-001, MN-001, etc.) applied during severity determination for audit trail';


-- ============================================================================
-- 6. gl_eudr_tam_corrective_action_requests — CARs
-- ============================================================================
RAISE NOTICE 'V112 [6/17]: Creating gl_eudr_tam_corrective_action_requests...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_corrective_action_requests (
    car_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    nc_ids                      JSONB           NOT NULL DEFAULT '[]',
        -- Linked non-conformance IDs (supports grouping)
    audit_id                    UUID            NOT NULL REFERENCES gl_eudr_tam_audits(audit_id),
        -- Originating audit
    supplier_id                 UUID            NOT NULL,
        -- Supplier required to take corrective action
    severity                    VARCHAR(20)     NOT NULL,
        -- Highest severity of linked NCs (determines SLA deadline)
    sla_deadline                TIMESTAMPTZ     NOT NULL,
        -- Calculated SLA deadline based on severity (Critical: 30d, Major: 90d, Minor: 365d)
    sla_status                  VARCHAR(20)     DEFAULT 'on_track',
        -- Current SLA compliance status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'issued',
        -- CAR lifecycle status
    issued_by                   VARCHAR(100)    NOT NULL,
        -- Auditor or authority who issued the CAR
    issued_at                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of CAR issuance
    acknowledged_at             TIMESTAMPTZ,
        -- Timestamp of auditee acknowledgment
    rca_submitted_at            TIMESTAMPTZ,
        -- Timestamp of root cause analysis submission
    cap_submitted_at            TIMESTAMPTZ,
        -- Timestamp of corrective action plan submission
    cap_approved_at             TIMESTAMPTZ,
        -- Timestamp of CAP approval by lead auditor
    evidence_submitted_at       TIMESTAMPTZ,
        -- Timestamp of implementation evidence submission
    verified_at                 TIMESTAMPTZ,
        -- Timestamp of effectiveness verification
    closed_at                   TIMESTAMPTZ,
        -- Timestamp of formal closure
    corrective_action_plan      JSONB,
        -- Structured CAP: actions, responsibilities, timelines
    verification_outcome        VARCHAR(30),
        -- Verification result
    verification_evidence_ids   JSONB           DEFAULT '[]',
        -- Evidence IDs supporting verification
    escalation_level            INTEGER         DEFAULT 0,
        -- Current escalation level (0-4)
    escalation_history          JSONB           DEFAULT '[]',
        -- History of escalation events with timestamps
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for CAR record integrity
    metadata                    JSONB           DEFAULT '{}',
        -- Additional CAR attributes (notes, attachments, cost)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_car_severity CHECK (severity IN ('critical', 'major', 'minor')),
    CONSTRAINT chk_tam_car_sla_status CHECK (sla_status IN (
        'on_track', 'warning', 'critical', 'overdue'
    )),
    CONSTRAINT chk_tam_car_status CHECK (status IN (
        'issued', 'acknowledged', 'rca_submitted', 'cap_submitted', 'cap_approved',
        'in_progress', 'evidence_submitted', 'verification_pending',
        'closed', 'rejected', 'overdue', 'escalated'
    )),
    CONSTRAINT chk_tam_car_verification CHECK (verification_outcome IS NULL OR verification_outcome IN (
        'effective', 'not_effective'
    )),
    CONSTRAINT chk_tam_car_escalation CHECK (escalation_level >= 0 AND escalation_level <= 4)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_audit ON gl_eudr_tam_corrective_action_requests (audit_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_supplier ON gl_eudr_tam_corrective_action_requests (supplier_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_severity ON gl_eudr_tam_corrective_action_requests (severity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_sla_status ON gl_eudr_tam_corrective_action_requests (sla_status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_status ON gl_eudr_tam_corrective_action_requests (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_sla_deadline ON gl_eudr_tam_corrective_action_requests (sla_deadline); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_issued_at ON gl_eudr_tam_corrective_action_requests (issued_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_issued_by ON gl_eudr_tam_corrective_action_requests (issued_by); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_escalation ON gl_eudr_tam_corrective_action_requests (escalation_level); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_provenance ON gl_eudr_tam_corrective_action_requests (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_tenant ON gl_eudr_tam_corrective_action_requests (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_created ON gl_eudr_tam_corrective_action_requests (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_sup_status ON gl_eudr_tam_corrective_action_requests (supplier_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_sla_sev ON gl_eudr_tam_corrective_action_requests (sla_status, severity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_status_deadline ON gl_eudr_tam_corrective_action_requests (status, sla_deadline); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for overdue and escalated CARs
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_car_overdue ON gl_eudr_tam_corrective_action_requests (severity, sla_deadline, escalation_level)
        WHERE sla_status IN ('critical', 'overdue') OR status IN ('overdue', 'escalated');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for open CARs requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_car_open ON gl_eudr_tam_corrective_action_requests (supplier_id, severity, sla_deadline)
        WHERE status NOT IN ('closed', 'rejected');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_nc_ids ON gl_eudr_tam_corrective_action_requests USING GIN (nc_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_cap ON gl_eudr_tam_corrective_action_requests USING GIN (corrective_action_plan); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_ver_ev ON gl_eudr_tam_corrective_action_requests USING GIN (verification_evidence_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_esc_hist ON gl_eudr_tam_corrective_action_requests USING GIN (escalation_history); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_car_metadata ON gl_eudr_tam_corrective_action_requests USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_corrective_action_requests IS 'Corrective action requests with severity-based SLA deadlines (Critical: 30d, Major: 90d, Minor: 365d), 4-stage escalation, and full lifecycle tracking from issuance through verified closure';
COMMENT ON COLUMN gl_eudr_tam_corrective_action_requests.sla_status IS 'SLA status: on_track (<75% elapsed), warning (75-90% elapsed), critical (>90% elapsed), overdue (deadline exceeded)';
COMMENT ON COLUMN gl_eudr_tam_corrective_action_requests.escalation_level IS 'Escalation levels 0-4: 0=none, 1=75% SLA (notify auditee), 2=90% SLA (manager escalation), 3=100% SLA (head of compliance), 4=30+ days overdue (cert suspension recommendation)';


-- ============================================================================
-- 7. gl_eudr_tam_certificate_records — Certification scheme certificates
-- ============================================================================
RAISE NOTICE 'V112 [7/17]: Creating gl_eudr_tam_certificate_records...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_certificate_records (
    certificate_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
        -- Certified supplier
    scheme                      VARCHAR(50)     NOT NULL,
        -- Certification scheme identifier
    certificate_number          VARCHAR(200)    NOT NULL,
        -- Unique certificate number from the certifying body
    status                      VARCHAR(30)     NOT NULL DEFAULT 'active',
        -- Current certificate status
    scope                       VARCHAR(200),
        -- Certification scope (FM, CoC, SFM, etc.)
    supply_chain_model          VARCHAR(30),
        -- Supply chain model (IP, SG, MB for RSPO)
    issue_date                  DATE,
        -- Certificate issue date
    expiry_date                 DATE,
        -- Certificate expiry date
    certified_products          JSONB           DEFAULT '[]',
        -- Products covered by certification
    certified_sites             JSONB           DEFAULT '[]',
        -- Sites covered by certification
    certification_body          VARCHAR(500),
        -- Accredited certification body name
    last_audit_date             DATE,
        -- Date of most recent scheme audit
    next_audit_date             DATE,
        -- Scheduled date for next scheme audit
    eudr_coverage_matrix        JSONB           DEFAULT '{}',
        -- EUDR coverage assessment (Art.3: FULL, Art.9: PARTIAL, etc.)
    metadata                    JSONB           DEFAULT '{}',
        -- Additional certificate attributes
    synced_at                   TIMESTAMPTZ,
        -- Last synchronization with scheme database
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_cert_scheme CHECK (scheme IN (
        'fsc', 'pefc', 'rspo', 'rainforest_alliance', 'iscc'
    )),
    CONSTRAINT chk_tam_cert_status CHECK (status IN (
        'active', 'suspended', 'terminated', 'expired'
    )),
    CONSTRAINT uq_tam_cert_scheme_number UNIQUE (scheme, certificate_number)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_supplier ON gl_eudr_tam_certificate_records (supplier_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_scheme ON gl_eudr_tam_certificate_records (scheme); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_number ON gl_eudr_tam_certificate_records (certificate_number); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_status ON gl_eudr_tam_certificate_records (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_expiry ON gl_eudr_tam_certificate_records (expiry_date); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_body ON gl_eudr_tam_certificate_records (certification_body); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_next_audit ON gl_eudr_tam_certificate_records (next_audit_date); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_synced ON gl_eudr_tam_certificate_records (synced_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_tenant ON gl_eudr_tam_certificate_records (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_created ON gl_eudr_tam_certificate_records (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_sup_scheme ON gl_eudr_tam_certificate_records (supplier_id, scheme); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_scheme_status ON gl_eudr_tam_certificate_records (scheme, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active certificates
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_cert_active ON gl_eudr_tam_certificate_records (supplier_id, scheme, expiry_date DESC)
        WHERE status = 'active';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for expiring certificates (within 90 days)
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_cert_expiring ON gl_eudr_tam_certificate_records (expiry_date, supplier_id)
        WHERE status = 'active' AND expiry_date IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_products ON gl_eudr_tam_certificate_records USING GIN (certified_products); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_sites ON gl_eudr_tam_certificate_records USING GIN (certified_sites); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_coverage ON gl_eudr_tam_certificate_records USING GIN (eudr_coverage_matrix); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_cert_metadata ON gl_eudr_tam_certificate_records USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_certificate_records IS 'Certification scheme certificate registry for FSC, PEFC, RSPO, Rainforest Alliance, and ISCC with EUDR coverage matrix and status synchronization';
COMMENT ON COLUMN gl_eudr_tam_certificate_records.scheme IS 'Certification schemes: fsc (Forest Stewardship Council), pefc (Programme for Endorsement of Forest Certification), rspo (Roundtable on Sustainable Palm Oil), rainforest_alliance, iscc (International Sustainability and Carbon Certification)';
COMMENT ON COLUMN gl_eudr_tam_certificate_records.status IS 'Certificate status: active (current and valid), suspended (temporarily halted by CB), terminated (withdrawn by CB), expired (past expiry date)';
COMMENT ON COLUMN gl_eudr_tam_certificate_records.eudr_coverage_matrix IS 'EUDR article coverage assessment per scheme: {"Art.3": "FULL", "Art.9": "PARTIAL", "Art.10": "FULL", ...}. Values: FULL, PARTIAL, NONE';
COMMENT ON COLUMN gl_eudr_tam_certificate_records.supply_chain_model IS 'Supply chain model for RSPO: IP (Identity Preserved), SG (Segregated), MB (Mass Balance)';


-- ============================================================================
-- 8. gl_eudr_tam_audit_reports — Generated audit reports
-- ============================================================================
RAISE NOTICE 'V112 [8/17]: Creating gl_eudr_tam_audit_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_audit_reports (
    report_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                    UUID            NOT NULL REFERENCES gl_eudr_tam_audits(audit_id),
        -- Parent audit
    report_type                 VARCHAR(50)     NOT NULL DEFAULT 'iso_19011',
        -- Report format standard
    report_version              INTEGER         DEFAULT 1,
        -- Version number (incremented for amendments)
    language                    VARCHAR(5)      DEFAULT 'en',
        -- Report language
    format                      VARCHAR(10)     NOT NULL DEFAULT 'pdf',
        -- Output format
    file_path                   VARCHAR(1000),
        -- S3 path to generated report file
    file_size_bytes             BIGINT,
        -- Report file size in bytes
    sha256_hash                 VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for report integrity and tamper detection
    sections                    JSONB           DEFAULT '{}',
        -- Report section metadata (objectives, scope, criteria, findings, conclusions)
    finding_count               JSONB           DEFAULT '{}',
        -- Summary of findings by severity
    evidence_package_path       VARCHAR(1000),
        -- S3 path to accompanying evidence package archive
    is_amended                  BOOLEAN         DEFAULT FALSE,
        -- Whether this is an amended report
    amendment_rationale         TEXT,
        -- Reason for amendment (if amended)
    previous_version_id         UUID,
        -- Reference to previous version (if amended)
    generated_by                VARCHAR(100),
        -- User or agent that generated the report
    tenant_id                   UUID            NOT NULL,
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_rpt_language CHECK (language IN ('en', 'fr', 'de', 'es', 'pt')),
    CONSTRAINT chk_tam_rpt_format CHECK (format IN ('pdf', 'json', 'html', 'xlsx', 'xml')),
    CONSTRAINT chk_tam_rpt_version CHECK (report_version >= 1)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_audit ON gl_eudr_tam_audit_reports (audit_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_type ON gl_eudr_tam_audit_reports (report_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_language ON gl_eudr_tam_audit_reports (language); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_format ON gl_eudr_tam_audit_reports (format); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_hash ON gl_eudr_tam_audit_reports (sha256_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_generated_by ON gl_eudr_tam_audit_reports (generated_by); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_tenant ON gl_eudr_tam_audit_reports (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_generated ON gl_eudr_tam_audit_reports (generated_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_audit_type ON gl_eudr_tam_audit_reports (audit_id, report_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_sections ON gl_eudr_tam_audit_reports USING GIN (sections); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_rpt_findings ON gl_eudr_tam_audit_reports USING GIN (finding_count); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_audit_reports IS 'ISO 19011:2018 Clause 6.6 compliant audit reports in 5 formats (PDF/JSON/HTML/XLSX/XML) and 5 languages with SHA-256 integrity hashing and amendment tracking';
COMMENT ON COLUMN gl_eudr_tam_audit_reports.sha256_hash IS 'SHA-256 hash of complete report content for integrity verification and tamper detection';
COMMENT ON COLUMN gl_eudr_tam_audit_reports.language IS 'Report languages: en (English), fr (French), de (German), es (Spanish), pt (Portuguese)';
COMMENT ON COLUMN gl_eudr_tam_audit_reports.format IS 'Output formats: pdf (primary), json (machine-readable), html (web display), xlsx (data analysis), xml (regulatory submission)';
COMMENT ON COLUMN gl_eudr_tam_audit_reports.sections IS 'ISO 19011 Clause 6.6 report sections: objectives, scope, criteria, client, team, dates, findings, conclusions, distribution';


-- ============================================================================
-- 9. gl_eudr_tam_competent_authority_interactions — Authority interactions
-- ============================================================================
RAISE NOTICE 'V112 [9/17]: Creating gl_eudr_tam_competent_authority_interactions...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_competent_authority_interactions (
    interaction_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator receiving the authority request
    authority_name              VARCHAR(500)    NOT NULL,
        -- Name of the competent authority
    member_state                CHAR(2)         NOT NULL,
        -- EU Member State (ISO 3166-1 alpha-2)
    interaction_type            VARCHAR(50)     NOT NULL,
        -- Type of regulatory interaction
    received_date               TIMESTAMPTZ     NOT NULL,
        -- Date the authority communication was received
    response_deadline           TIMESTAMPTZ     NOT NULL,
        -- Deadline for response submission
    response_sla_status         VARCHAR(20)     DEFAULT 'on_track',
        -- Response SLA compliance status
    internal_tasks              JSONB           DEFAULT '[]',
        -- Internal response preparation tasks and assignments
    evidence_package_id         UUID,
        -- Generated evidence package ID (nullable until assembled)
    response_submitted_at       TIMESTAMPTZ,
        -- Timestamp of response submission to authority
    authority_decision          TEXT,
        -- Authority decision or outcome following response
    enforcement_measures        JSONB           DEFAULT '[]',
        -- Enforcement measures issued by the authority
    status                      VARCHAR(30)     NOT NULL DEFAULT 'open',
        -- Interaction lifecycle status
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for interaction record integrity
    metadata                    JSONB           DEFAULT '{}',
        -- Additional interaction context
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_ai_type CHECK (interaction_type IN (
        'document_request', 'inspection_notification', 'unannounced_inspection',
        'corrective_action_order', 'interim_measure', 'definitive_measure',
        'information_request'
    )),
    CONSTRAINT chk_tam_ai_sla CHECK (response_sla_status IN (
        'on_track', 'warning', 'critical', 'overdue'
    )),
    CONSTRAINT chk_tam_ai_status CHECK (status IN (
        'open', 'in_progress', 'responded', 'closed'
    ))
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_operator ON gl_eudr_tam_competent_authority_interactions (operator_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_state ON gl_eudr_tam_competent_authority_interactions (member_state); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_type ON gl_eudr_tam_competent_authority_interactions (interaction_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_status ON gl_eudr_tam_competent_authority_interactions (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_sla ON gl_eudr_tam_competent_authority_interactions (response_sla_status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_received ON gl_eudr_tam_competent_authority_interactions (received_date DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_deadline ON gl_eudr_tam_competent_authority_interactions (response_deadline); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_provenance ON gl_eudr_tam_competent_authority_interactions (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_tenant ON gl_eudr_tam_competent_authority_interactions (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_created ON gl_eudr_tam_competent_authority_interactions (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_op_status ON gl_eudr_tam_competent_authority_interactions (operator_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_status_deadline ON gl_eudr_tam_competent_authority_interactions (status, response_deadline); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for open authority interactions requiring response
DO $$ BEGIN
    CREATE INDEX idx_eudr_tam_ai_open ON gl_eudr_tam_competent_authority_interactions (response_deadline, response_sla_status)
        WHERE status IN ('open', 'in_progress');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_tasks ON gl_eudr_tam_competent_authority_interactions USING GIN (internal_tasks); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_enforcement ON gl_eudr_tam_competent_authority_interactions USING GIN (enforcement_measures); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ai_metadata ON gl_eudr_tam_competent_authority_interactions USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_competent_authority_interactions IS 'Competent authority interaction registry for 27 EU Member State authorities under EUDR Articles 14-16 with response SLA tracking, evidence assembly, and enforcement measure monitoring';
COMMENT ON COLUMN gl_eudr_tam_competent_authority_interactions.interaction_type IS 'Interaction types per EUDR Articles 14-20: document_request (Art.15), inspection_notification (Art.15), unannounced_inspection (Art.15(2)), corrective_action_order (Art.18), interim_measure (Art.19), definitive_measure (Art.20), information_request (Art.21)';
COMMENT ON COLUMN gl_eudr_tam_competent_authority_interactions.response_sla_status IS 'Response SLA status: on_track (<75% deadline elapsed), warning (75-90%), critical (>90%), overdue (deadline exceeded)';
COMMENT ON COLUMN gl_eudr_tam_competent_authority_interactions.enforcement_measures IS 'Enforcement measures issued by authority: fines, market_access_suspension, goods_confiscation, public_naming per EUDR Articles 22-23';


-- ============================================================================
-- 10. gl_eudr_tam_competent_authorities — Authority reference data
-- ============================================================================
RAISE NOTICE 'V112 [10/17]: Creating gl_eudr_tam_competent_authorities...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_competent_authorities (
    authority_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    member_state                CHAR(2)         NOT NULL UNIQUE,
        -- EU Member State ISO 3166-1 alpha-2 code
    authority_name              VARCHAR(500)    NOT NULL,
        -- Official authority name
    legal_basis                 TEXT,
        -- National implementing legislation reference
    inspection_focus            JSONB           DEFAULT '[]',
        -- Priority inspection areas (commodities, supply chains, document types)
    contact_details             JSONB           DEFAULT '{}',
        -- Authority contact information
    default_response_days       INTEGER         DEFAULT 30,
        -- Default response timeline in business days
    active                      BOOLEAN         DEFAULT TRUE,
        -- Whether the authority profile is currently active
    metadata                    JSONB           DEFAULT '{}',
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_ca_response_days CHECK (default_response_days > 0 AND default_response_days <= 365)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_ca_state ON gl_eudr_tam_competent_authorities (member_state); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ca_active ON gl_eudr_tam_competent_authorities (active); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ca_tenant ON gl_eudr_tam_competent_authorities (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ca_focus ON gl_eudr_tam_competent_authorities USING GIN (inspection_focus); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ca_contact ON gl_eudr_tam_competent_authorities USING GIN (contact_details); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_competent_authorities IS 'Reference data for 27 EU Member State competent authorities designated under EUDR Article 14 with inspection focus areas and default response timelines';
COMMENT ON COLUMN gl_eudr_tam_competent_authorities.member_state IS 'EU Member State ISO 3166-1 alpha-2 code (e.g., DE=Germany, FR=France, NL=Netherlands, BE=Belgium, IT=Italy, ES=Spain)';
COMMENT ON COLUMN gl_eudr_tam_competent_authorities.default_response_days IS 'Default response deadline in business days (typically 30 days for document requests per Article 15)';


-- ============================================================================
-- 11. gl_eudr_tam_scheme_nc_mappings — NC taxonomy mapping reference
-- ============================================================================
RAISE NOTICE 'V112 [11/17]: Creating gl_eudr_tam_scheme_nc_mappings...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_scheme_nc_mappings (
    mapping_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scheme                      VARCHAR(50)     NOT NULL,
        -- Certification scheme
    scheme_nc_level             VARCHAR(50)     NOT NULL,
        -- Scheme-specific NC level (e.g. "Major NC", "Critical", "Improvement Need")
    unified_nc_level            VARCHAR(20)     NOT NULL,
        -- GreenLang unified NC level
    sla_days                    INTEGER         NOT NULL,
        -- SLA deadline in days for this severity
    description                 TEXT,
        -- Description of the mapping rationale
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_ncm_scheme CHECK (scheme IN (
        'fsc', 'pefc', 'rspo', 'rainforest_alliance', 'iscc'
    )),
    CONSTRAINT chk_tam_ncm_unified CHECK (unified_nc_level IN (
        'critical', 'major', 'minor', 'observation'
    )),
    CONSTRAINT chk_tam_ncm_sla CHECK (sla_days > 0),
    CONSTRAINT uq_tam_ncm_scheme_level UNIQUE (scheme, scheme_nc_level)
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncm_scheme ON gl_eudr_tam_scheme_nc_mappings (scheme); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncm_unified ON gl_eudr_tam_scheme_nc_mappings (unified_nc_level); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncm_scheme_level ON gl_eudr_tam_scheme_nc_mappings (scheme, scheme_nc_level); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncm_tenant ON gl_eudr_tam_scheme_nc_mappings (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_scheme_nc_mappings IS 'Deterministic NC taxonomy mapping from 5 certification scheme NC levels to unified GreenLang NC severity (critical/major/minor/observation) with SLA days';
COMMENT ON COLUMN gl_eudr_tam_scheme_nc_mappings.scheme_nc_level IS 'Scheme-specific NC level (e.g., FSC: Major/Minor/OFI; RSPO: Major NC/Minor NC/Observation; RA: Critical/Major/Minor/Improvement Need; ISCC: Major NC/Minor NC/Observation)';
COMMENT ON COLUMN gl_eudr_tam_scheme_nc_mappings.sla_days IS 'SLA deadline in days: critical=30, major=90, minor=365, observation=0 (no CAR required)';


-- ============================================================================
-- 12. gl_eudr_tam_audit_team_assignments — Audit team member assignments
-- ============================================================================
RAISE NOTICE 'V112 [12/17]: Creating gl_eudr_tam_audit_team_assignments...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_audit_team_assignments (
    assignment_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                    UUID            NOT NULL REFERENCES gl_eudr_tam_audits(audit_id),
        -- Audit this assignment belongs to
    auditor_id                  UUID            NOT NULL REFERENCES gl_eudr_tam_auditors(auditor_id),
        -- Assigned auditor
    role                        VARCHAR(50)     NOT NULL,
        -- Team member role in the audit
    match_score                 NUMERIC(5,2),
        -- Competence match score from auditor matching algorithm (0-100)
    assigned_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Assignment timestamp
    tenant_id                   UUID            NOT NULL,

    CONSTRAINT chk_tam_ata_role CHECK (role IN (
        'lead_auditor', 'co_auditor', 'technical_expert', 'observer', 'trainee'
    )),
    CONSTRAINT chk_tam_ata_match CHECK (match_score IS NULL OR (match_score >= 0 AND match_score <= 100))
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_ata_audit ON gl_eudr_tam_audit_team_assignments (audit_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ata_auditor ON gl_eudr_tam_audit_team_assignments (auditor_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ata_role ON gl_eudr_tam_audit_team_assignments (role); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ata_tenant ON gl_eudr_tam_audit_team_assignments (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ata_assigned ON gl_eudr_tam_audit_team_assignments (assigned_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ata_audit_role ON gl_eudr_tam_audit_team_assignments (audit_id, role); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ata_auditor_audit ON gl_eudr_tam_audit_team_assignments (auditor_id, audit_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_audit_team_assignments IS 'Audit team member assignments linking auditors to audits with role classification and competence match score';


-- ============================================================================
-- 13. gl_eudr_tam_stakeholder_interviews — Interview records
-- ============================================================================
RAISE NOTICE 'V112 [13/17]: Creating gl_eudr_tam_stakeholder_interviews...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_stakeholder_interviews (
    interview_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                    UUID            NOT NULL REFERENCES gl_eudr_tam_audits(audit_id),
        -- Parent audit
    interview_type              VARCHAR(50)     NOT NULL,
        -- Type of stakeholder interview
    interviewee_role            VARCHAR(200),
        -- Role or position of the interviewee
    scheduled_date              TIMESTAMPTZ,
        -- Planned interview date and time
    conducted_date              TIMESTAMPTZ,
        -- Actual interview date and time
    template_id                 VARCHAR(50),
        -- Structured interview template identifier
    outcome_summary             TEXT,
        -- Summary of key interview outcomes
    evidence_ids                JSONB           DEFAULT '[]',
        -- Linked evidence item IDs (transcripts, recordings)
    metadata                    JSONB           DEFAULT '{}',
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_tam_si_type CHECK (interview_type IN (
        'community', 'worker', 'management', 'government', 'other'
    ))
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_si_audit ON gl_eudr_tam_stakeholder_interviews (audit_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_si_type ON gl_eudr_tam_stakeholder_interviews (interview_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_si_scheduled ON gl_eudr_tam_stakeholder_interviews (scheduled_date DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_si_conducted ON gl_eudr_tam_stakeholder_interviews (conducted_date DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_si_tenant ON gl_eudr_tam_stakeholder_interviews (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_si_audit_type ON gl_eudr_tam_stakeholder_interviews (audit_id, interview_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_si_evidence ON gl_eudr_tam_stakeholder_interviews USING GIN (evidence_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_stakeholder_interviews IS 'Stakeholder interview records from audit fieldwork (community, worker, management, government) with structured templates and outcome tracking';


-- ============================================================================
-- 14. gl_eudr_tam_audit_schedule — Audit planning (hypertable)
-- ============================================================================
RAISE NOTICE 'V112 [14/17]: Creating gl_eudr_tam_audit_schedule (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_audit_schedule (
    schedule_id                 UUID            DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator owning the audit schedule
    supplier_id                 UUID            NOT NULL,
        -- Supplier to be audited
    planned_quarter             VARCHAR(7)      NOT NULL,
        -- Planned quarter (e.g. "2026-Q2")
    audit_type                  VARCHAR(30)     NOT NULL,
        -- Planned audit type
    modality                    VARCHAR(30)     NOT NULL,
        -- Planned audit modality
    priority_score              NUMERIC(5,2),
        -- Risk-based priority score at scheduling time
    risk_factors                JSONB           DEFAULT '{}',
        -- Risk factor breakdown used in priority calculation
    assigned_auditor_id         UUID,
        -- Pre-assigned auditor (nullable until assigned)
    certification_scheme        VARCHAR(50),
        -- Associated certification scheme
    status                      VARCHAR(30)     DEFAULT 'planned',
        -- Schedule item status
    linked_audit_id             UUID,
        -- Linked audit record once created
    country_code                CHAR(2),
        -- Country for scheduled audit
    commodity                   VARCHAR(50),
        -- Commodity for scheduled audit
    scheduled_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of schedule creation (partition key)
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (schedule_id, scheduled_at),

    CONSTRAINT chk_tam_sched_type CHECK (audit_type IN ('full', 'targeted', 'surveillance', 'unscheduled')),
    CONSTRAINT chk_tam_sched_modality CHECK (modality IN ('on_site', 'remote', 'hybrid', 'unannounced')),
    CONSTRAINT chk_tam_sched_status CHECK (status IN ('planned', 'confirmed', 'in_progress', 'completed', 'cancelled', 'rescheduled'))
);

SELECT create_hypertable(
    'gl_eudr_tam_audit_schedule',
    'scheduled_at',
    chunk_time_interval => INTERVAL '90 days',
    if_not_exists => TRUE
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_operator ON gl_eudr_tam_audit_schedule (operator_id, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_supplier ON gl_eudr_tam_audit_schedule (supplier_id, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_quarter ON gl_eudr_tam_audit_schedule (planned_quarter, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_type ON gl_eudr_tam_audit_schedule (audit_type, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_status ON gl_eudr_tam_audit_schedule (status, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_priority ON gl_eudr_tam_audit_schedule (priority_score DESC, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_auditor ON gl_eudr_tam_audit_schedule (assigned_auditor_id, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_scheme ON gl_eudr_tam_audit_schedule (certification_scheme, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_country ON gl_eudr_tam_audit_schedule (country_code, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_commodity ON gl_eudr_tam_audit_schedule (commodity, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_tenant ON gl_eudr_tam_audit_schedule (tenant_id, scheduled_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_sched_risk ON gl_eudr_tam_audit_schedule USING GIN (risk_factors); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_audit_schedule IS 'Time-series audit schedule hypertable (90-day chunks) for risk-based audit planning with priority scoring and quarterly planning cycles';


-- ============================================================================
-- 15. gl_eudr_tam_nc_trend_log — NC trend tracking (hypertable)
-- ============================================================================
RAISE NOTICE 'V112 [15/17]: Creating gl_eudr_tam_nc_trend_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_nc_trend_log (
    log_id                      UUID            DEFAULT gen_random_uuid(),
    audit_id                    UUID            NOT NULL,
        -- Source audit
    supplier_id                 UUID            NOT NULL,
        -- Supplier associated with NC
    nc_id                       UUID            NOT NULL,
        -- Non-conformance record
    severity                    VARCHAR(20)     NOT NULL,
        -- NC severity at detection time
    eudr_article                VARCHAR(20),
        -- Mapped EUDR article
    scheme_clause               VARCHAR(100),
        -- Mapped certification scheme clause
    country_code                CHAR(2),
        -- Country where NC occurred
    commodity                   VARCHAR(50),
        -- Commodity associated with NC
    root_cause_category         VARCHAR(100),
        -- Root cause category from RCA
    risk_impact_score           NUMERIC(5,2),
        -- NC risk impact score at detection
    recorded_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (log_id, recorded_at),

    CONSTRAINT chk_tam_ncl_severity CHECK (severity IN ('critical', 'major', 'minor', 'observation'))
);

SELECT create_hypertable(
    'gl_eudr_tam_nc_trend_log',
    'recorded_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_audit ON gl_eudr_tam_nc_trend_log (audit_id, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_supplier ON gl_eudr_tam_nc_trend_log (supplier_id, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_severity ON gl_eudr_tam_nc_trend_log (severity, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_country ON gl_eudr_tam_nc_trend_log (country_code, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_commodity ON gl_eudr_tam_nc_trend_log (commodity, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_eudr_art ON gl_eudr_tam_nc_trend_log (eudr_article, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_root_cause ON gl_eudr_tam_nc_trend_log (root_cause_category, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_impact ON gl_eudr_tam_nc_trend_log (risk_impact_score DESC, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_tenant ON gl_eudr_tam_nc_trend_log (tenant_id, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_sev_country ON gl_eudr_tam_nc_trend_log (severity, country_code, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_ncl_sev_commodity ON gl_eudr_tam_nc_trend_log (severity, commodity, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_nc_trend_log IS 'Time-series NC trend log hypertable (30-day chunks) for non-conformance pattern analysis by severity, country, commodity, and root cause category';


-- ============================================================================
-- 16. gl_eudr_tam_car_sla_log — CAR SLA tracking (hypertable)
-- ============================================================================
RAISE NOTICE 'V112 [16/17]: Creating gl_eudr_tam_car_sla_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_car_sla_log (
    log_id                      UUID            DEFAULT gen_random_uuid(),
    car_id                      UUID            NOT NULL,
        -- Associated CAR
    previous_status             VARCHAR(30),
        -- CAR status before change
    new_status                  VARCHAR(30),
        -- CAR status after change
    sla_remaining_days          INTEGER,
        -- Days remaining until SLA deadline at time of change
    escalation_level            INTEGER,
        -- Current escalation level at time of change
    severity                    VARCHAR(20),
        -- CAR severity (denormalized for aggregation)
    supplier_id                 UUID,
        -- Supplier (denormalized for aggregation)
    changed_by                  VARCHAR(100),
        -- User or system that triggered the change
    changed_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of change (partition key)
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (log_id, changed_at)
);

SELECT create_hypertable(
    'gl_eudr_tam_car_sla_log',
    'changed_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_csl_car ON gl_eudr_tam_car_sla_log (car_id, changed_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_csl_new_status ON gl_eudr_tam_car_sla_log (new_status, changed_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_csl_severity ON gl_eudr_tam_car_sla_log (severity, changed_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_csl_supplier ON gl_eudr_tam_car_sla_log (supplier_id, changed_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_csl_sla_days ON gl_eudr_tam_car_sla_log (sla_remaining_days, changed_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_csl_escalation ON gl_eudr_tam_car_sla_log (escalation_level, changed_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_csl_changed_by ON gl_eudr_tam_car_sla_log (changed_by, changed_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_csl_tenant ON gl_eudr_tam_car_sla_log (tenant_id, changed_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_car_sla_log IS 'Time-series CAR SLA tracking hypertable (30-day chunks) recording all CAR status transitions with SLA remaining days and escalation level for compliance monitoring';


-- ============================================================================
-- 17. gl_eudr_tam_audit_trail — Immutable audit trail (hypertable)
-- ============================================================================
RAISE NOTICE 'V112 [17/17]: Creating gl_eudr_tam_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_tam_audit_trail (
    trail_id                    UUID            DEFAULT gen_random_uuid(),
    entity_type                 VARCHAR(50)     NOT NULL,
        -- Type of entity affected
    entity_id                   UUID            NOT NULL,
        -- ID of the affected entity
    action                      VARCHAR(50)     NOT NULL,
        -- Action performed
    before_value                JSONB,
        -- State before the action
    after_value                 JSONB,
        -- State after the action
    actor                       VARCHAR(100)    NOT NULL,
        -- User or system agent performing the action
    actor_role                  VARCHAR(50),
        -- Role of the actor
    ip_address                  VARCHAR(45),
        -- Source IP address
    change_summary              TEXT,
        -- Human-readable summary of what changed
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for audit integrity chain
    recorded_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (trail_id, recorded_at),

    CONSTRAINT chk_tam_at_action CHECK (action IN (
        'create', 'update', 'delete', 'archive',
        'verify', 'reject', 'approve', 'escalate',
        'issue', 'acknowledge', 'close', 'reopen',
        'assign', 'reassign', 'suspend', 'restore',
        'generate', 'export', 'import', 'sync'
    )),
    CONSTRAINT chk_tam_at_entity CHECK (entity_type IN (
        'audit', 'auditor', 'checklist', 'evidence', 'non_conformance',
        'corrective_action_request', 'certificate', 'report',
        'authority_interaction', 'competent_authority', 'scheme_mapping',
        'team_assignment', 'interview', 'schedule', 'system'
    ))
);

SELECT create_hypertable(
    'gl_eudr_tam_audit_trail',
    'recorded_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_entity_type ON gl_eudr_tam_audit_trail (entity_type, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_entity_id ON gl_eudr_tam_audit_trail (entity_id, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_action ON gl_eudr_tam_audit_trail (action, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_actor ON gl_eudr_tam_audit_trail (actor, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_actor_role ON gl_eudr_tam_audit_trail (actor_role, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_provenance ON gl_eudr_tam_audit_trail (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_tenant ON gl_eudr_tam_audit_trail (tenant_id, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_entity_action ON gl_eudr_tam_audit_trail (entity_type, action, recorded_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_before ON gl_eudr_tam_audit_trail USING GIN (before_value); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_tam_at_after ON gl_eudr_tam_audit_trail USING GIN (after_value); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_tam_audit_trail IS 'Immutable audit trail hypertable (30-day chunks) recording all entity lifecycle events with before/after state, actor identification, and SHA-256 provenance hashing per EUDR Article 31';
COMMENT ON COLUMN gl_eudr_tam_audit_trail.action IS 'Actions: create, update, delete, archive, verify, reject, approve, escalate, issue, acknowledge, close, reopen, assign, reassign, suspend, restore, generate, export, import, sync';
COMMENT ON COLUMN gl_eudr_tam_audit_trail.entity_type IS 'Entity types: audit, auditor, checklist, evidence, non_conformance, corrective_action_request, certificate, report, authority_interaction, competent_authority, scheme_mapping, team_assignment, interview, schedule, system';
COMMENT ON COLUMN gl_eudr_tam_audit_trail.provenance_hash IS 'SHA-256 hash linking to provenance chain for immutability verification';


-- ============================================================================
-- SEED DATA: NC taxonomy mappings for 5 certification schemes
-- ============================================================================

RAISE NOTICE 'V112: Seeding NC taxonomy mappings for 5 certification schemes...';

-- FSC NC taxonomy mapping
INSERT INTO gl_eudr_tam_scheme_nc_mappings (scheme, scheme_nc_level, unified_nc_level, sla_days, description, tenant_id)
VALUES
    ('fsc', 'Major', 'major', 90, 'FSC Major NC maps to GreenLang major (90-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('fsc', 'Minor', 'minor', 365, 'FSC Minor NC maps to GreenLang minor (365-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('fsc', 'Observation', 'observation', 0, 'FSC Observation maps to GreenLang observation (no CAR required)', '00000000-0000-0000-0000-000000000000')
ON CONFLICT (scheme, scheme_nc_level) DO NOTHING;

-- PEFC NC taxonomy mapping
INSERT INTO gl_eudr_tam_scheme_nc_mappings (scheme, scheme_nc_level, unified_nc_level, sla_days, description, tenant_id)
VALUES
    ('pefc', 'Major NC', 'major', 90, 'PEFC Major NC maps to GreenLang major (90-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('pefc', 'Minor NC', 'minor', 365, 'PEFC Minor NC maps to GreenLang minor (365-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('pefc', 'Observation', 'observation', 0, 'PEFC Observation maps to GreenLang observation (no CAR required)', '00000000-0000-0000-0000-000000000000')
ON CONFLICT (scheme, scheme_nc_level) DO NOTHING;

-- RSPO NC taxonomy mapping
INSERT INTO gl_eudr_tam_scheme_nc_mappings (scheme, scheme_nc_level, unified_nc_level, sla_days, description, tenant_id)
VALUES
    ('rspo', 'Major NC', 'major', 90, 'RSPO Major NC maps to GreenLang major (90-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('rspo', 'Minor NC', 'minor', 365, 'RSPO Minor NC maps to GreenLang minor (365-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('rspo', 'Observation', 'observation', 0, 'RSPO Observation maps to GreenLang observation (no CAR required)', '00000000-0000-0000-0000-000000000000')
ON CONFLICT (scheme, scheme_nc_level) DO NOTHING;

-- Rainforest Alliance NC taxonomy mapping
INSERT INTO gl_eudr_tam_scheme_nc_mappings (scheme, scheme_nc_level, unified_nc_level, sla_days, description, tenant_id)
VALUES
    ('rainforest_alliance', 'Critical', 'critical', 30, 'RA Critical NC maps to GreenLang critical (30-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('rainforest_alliance', 'Major', 'major', 90, 'RA Major NC maps to GreenLang major (90-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('rainforest_alliance', 'Minor', 'minor', 365, 'RA Minor NC maps to GreenLang minor (365-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('rainforest_alliance', 'Improvement Need', 'observation', 0, 'RA Improvement Need maps to GreenLang observation (no CAR required)', '00000000-0000-0000-0000-000000000000')
ON CONFLICT (scheme, scheme_nc_level) DO NOTHING;

-- ISCC NC taxonomy mapping
INSERT INTO gl_eudr_tam_scheme_nc_mappings (scheme, scheme_nc_level, unified_nc_level, sla_days, description, tenant_id)
VALUES
    ('iscc', 'Major NC', 'major', 90, 'ISCC Major NC maps to GreenLang major (90-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('iscc', 'Minor NC', 'minor', 365, 'ISCC Minor NC maps to GreenLang minor (365-day CAR deadline)', '00000000-0000-0000-0000-000000000000'),
    ('iscc', 'Observation', 'observation', 0, 'ISCC Observation maps to GreenLang observation (no CAR required)', '00000000-0000-0000-0000-000000000000')
ON CONFLICT (scheme, scheme_nc_level) DO NOTHING;


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Daily audit schedule summary
RAISE NOTICE 'V112: Creating continuous aggregate: gl_eudr_tam_audit_schedule_daily...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_tam_audit_schedule_daily
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', scheduled_at)       AS day,
        tenant_id,
        operator_id,
        country_code,
        commodity,
        certification_scheme,
        audit_type,
        status,
        COUNT(*)                                  AS schedule_count,
        AVG(priority_score)                       AS avg_priority,
        MAX(priority_score)                       AS max_priority,
        COUNT(DISTINCT supplier_id)               AS unique_suppliers
    FROM gl_eudr_tam_audit_schedule
    GROUP BY day, tenant_id, operator_id, country_code, commodity, certification_scheme, audit_type, status;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_tam_audit_schedule_daily',
        start_offset => INTERVAL '7 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_tam_audit_schedule_daily IS 'Daily rollup of audit schedule entries by operator, country, commodity, scheme, type, and status with priority statistics';


-- Monthly NC trends summary
RAISE NOTICE 'V112: Creating continuous aggregate: gl_eudr_tam_nc_trends_monthly...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_tam_nc_trends_monthly
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('30 days', recorded_at)      AS month,
        tenant_id,
        country_code,
        commodity,
        severity,
        eudr_article,
        root_cause_category,
        COUNT(*)                                  AS nc_count,
        AVG(risk_impact_score)                    AS avg_impact,
        MAX(risk_impact_score)                    AS max_impact,
        COUNT(DISTINCT supplier_id)               AS unique_suppliers,
        COUNT(DISTINCT audit_id)                  AS unique_audits
    FROM gl_eudr_tam_nc_trend_log
    GROUP BY month, tenant_id, country_code, commodity, severity, eudr_article, root_cause_category;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_tam_nc_trends_monthly',
        start_offset => INTERVAL '60 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_tam_nc_trends_monthly IS 'Monthly rollup of NC trends by country, commodity, severity, EUDR article, and root cause category with impact statistics and supplier/audit counts';


-- ============================================================================
-- RETENTION POLICIES (5 years per EUDR Article 31)
-- ============================================================================

RAISE NOTICE 'V112: Creating retention policies (5 years per EUDR Article 31)...';

-- 5 years for audit schedule
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_tam_audit_schedule', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for NC trend log
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_tam_nc_trend_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for CAR SLA log
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_tam_car_sla_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for audit trail
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_tam_audit_trail', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- COMPRESSION POLICIES
-- ============================================================================

RAISE NOTICE 'V112: Creating compression policies on hypertables...';

-- Compression on audit schedule (after 90 days)
DO $$ BEGIN
    ALTER TABLE gl_eudr_tam_audit_schedule SET (timescaledb.compress);
    SELECT add_compression_policy('gl_eudr_tam_audit_schedule', INTERVAL '90 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compression on NC trend log (after 60 days)
DO $$ BEGIN
    ALTER TABLE gl_eudr_tam_nc_trend_log SET (timescaledb.compress);
    SELECT add_compression_policy('gl_eudr_tam_nc_trend_log', INTERVAL '60 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compression on CAR SLA log (after 60 days)
DO $$ BEGIN
    ALTER TABLE gl_eudr_tam_car_sla_log SET (timescaledb.compress);
    SELECT add_compression_policy('gl_eudr_tam_car_sla_log', INTERVAL '60 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compression on audit trail (after 60 days)
DO $$ BEGIN
    ALTER TABLE gl_eudr_tam_audit_trail SET (timescaledb.compress);
    SELECT add_compression_policy('gl_eudr_tam_audit_trail', INTERVAL '60 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- GRANTS — greenlang_app role
-- ============================================================================

RAISE NOTICE 'V112: Granting permissions to greenlang_app...';

-- Regular tables (full CRUD)
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_audits TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_auditors TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_audit_checklists TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_audit_evidence TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_non_conformances TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_corrective_action_requests TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_certificate_records TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_audit_reports TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_competent_authority_interactions TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_competent_authorities TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_scheme_nc_mappings TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_audit_team_assignments TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_tam_stakeholder_interviews TO greenlang_app;

-- Hypertables (append-only for audit integrity)
GRANT SELECT, INSERT ON gl_eudr_tam_audit_schedule TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_tam_nc_trend_log TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_tam_car_sla_log TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_tam_audit_trail TO greenlang_app;

-- Continuous aggregates (read-only)
GRANT SELECT ON gl_eudr_tam_audit_schedule_daily TO greenlang_app;
GRANT SELECT ON gl_eudr_tam_nc_trends_monthly TO greenlang_app;

-- Read-only role (conditional)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT SELECT ON gl_eudr_tam_audits TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_auditors TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_audit_checklists TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_audit_evidence TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_non_conformances TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_corrective_action_requests TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_certificate_records TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_audit_reports TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_competent_authority_interactions TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_competent_authorities TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_scheme_nc_mappings TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_audit_team_assignments TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_stakeholder_interviews TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_audit_schedule TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_nc_trend_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_car_sla_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_audit_trail TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_audit_schedule_daily TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_tam_nc_trends_monthly TO greenlang_readonly;
    END IF;
END
$$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V112: AGENT-EUDR-024 Third-Party Audit Manager tables created successfully!';
RAISE NOTICE 'V112: Created 17 tables (13 regular + 4 hypertables), 2 continuous aggregates, ~180 indexes';
RAISE NOTICE 'V112: ~30 GIN indexes on JSONB columns, ~10 partial indexes for filtered records';
RAISE NOTICE 'V112: Retention policies: 5y on all 4 hypertables per EUDR Article 31';
RAISE NOTICE 'V112: Compression policies: 60-90d on all 4 hypertables';
RAISE NOTICE 'V112: Grants applied for greenlang_app and greenlang_readonly roles';

COMMIT;

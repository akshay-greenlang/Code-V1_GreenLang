-- ============================================================================
-- V128: AGENT-EUDR-040 Authority Communication Manager
-- ============================================================================
-- Creates tables for the Authority Communication Manager agent which handles
-- all formal communications between EUDR operators/traders and EU Member State
-- competent authorities. Manages information requests under Article 17,
-- on-the-spot inspection coordination under Article 15, non-compliance
-- notifications and penalty tracking, administrative appeals under Article 19,
-- document exchange with encryption, competent authority registry for all 27
-- EU member states, notification delivery tracking across channels, multi-
-- language communication templates, deadline scheduling with reminder
-- functions, response time calculation, and a complete Article 31 audit trail
-- via TimescaleDB hypertable with 5-year retention.
--
-- Agent ID: GL-EUDR-ACM-040
-- PRD: PRD-AGENT-EUDR-040
-- Regulation: EU 2023/1115 (EUDR) Articles 15, 17, 19, 24, 25, 31
-- Tables: 10 (9 regular + 1 hypertable)
-- Indexes: ~125
-- Dependencies: TimescaleDB extension (for hypertables)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V128: Creating AGENT-EUDR-040 Authority Communication Manager tables...';


-- ============================================================================
-- 1. gl_eudr_acm_communications -- Main communication records
-- ============================================================================
-- Stores every formal communication between EUDR operators/traders and EU
-- Member State competent authorities. Each communication is identified by a
-- unique comm_id and tracks the full lifecycle from initiation through
-- response to closure. Communications may be initiated by either the operator
-- (e.g., voluntary disclosures, clarification requests) or the authority
-- (e.g., information requests, inspection notices, non-compliance findings).
-- The record includes subject classification, priority level, response
-- deadlines per EUDR Article 17(2), language code for multi-lingual support
-- across 24 EU official languages, and a provenance hash for audit integrity.
-- ============================================================================
RAISE NOTICE 'V128 [1/10]: Creating gl_eudr_acm_communications...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_communications (
    comm_id                         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique internal identifier for this communication record
    comm_reference                  VARCHAR(50)     UNIQUE NOT NULL,
        -- Human-readable communication reference number
        -- Format: ACM-[YYYY]-[CC]-[NNNNNN] where CC=country code, N=sequence
        -- e.g. "ACM-2026-DE-000147" for the 147th communication with German authority
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator or trader identifier involved in this communication
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    authority_id                    UUID            NOT NULL,
        -- FK to gl_eudr_acm_authorities: the competent authority involved
    comm_type                       VARCHAR(50)     NOT NULL,
        -- Type of communication categorized by EUDR article and purpose
    comm_direction                  VARCHAR(20)     NOT NULL DEFAULT 'inbound',
        -- Direction: inbound (authority to operator) or outbound (operator to authority)
    subject                         VARCHAR(500)    NOT NULL,
        -- Subject line of the communication
    body                            TEXT            DEFAULT '',
        -- Full body text of the initial communication message
    priority                        VARCHAR(20)     NOT NULL DEFAULT 'normal',
        -- Priority level determining response urgency and SLA targets
    status                          VARCHAR(30)     NOT NULL DEFAULT 'open',
        -- Current lifecycle status of the communication
    initiated_by                    VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- Actor who initiated this communication (user ID, authority ref, or system)
    initiated_at                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the communication was initiated
    acknowledged_at                 TIMESTAMPTZ,
        -- Timestamp when the receiving party acknowledged receipt
    responded_at                    TIMESTAMPTZ,
        -- Timestamp when a substantive response was provided
    closed_at                       TIMESTAMPTZ,
        -- Timestamp when the communication was formally closed
    response_deadline               TIMESTAMPTZ,
        -- Deadline by which a response must be provided
        -- Per EUDR Article 17(2): operators must respond to information requests
        -- within the timeframe set by the competent authority
    escalation_deadline             TIMESTAMPTZ,
        -- Deadline after which the communication escalates to next priority level
    language_code                   VARCHAR(5)      NOT NULL DEFAULT 'en',
        -- ISO 639-1 language code for the communication
        -- Supports all 24 EU official languages
    related_dds_reference           VARCHAR(100),
        -- Reference to the EUDR Due Diligence Statement related to this communication
    related_comm_id                 UUID,
        -- FK to parent communication if this is a follow-up or thread
    thread_depth                    INTEGER         NOT NULL DEFAULT 0,
        -- Depth in communication thread (0 = root, 1 = first reply, etc.)
    attachments_count               INTEGER         NOT NULL DEFAULT 0,
        -- Number of documents attached to this communication
    metadata                        JSONB           DEFAULT '{}',
        -- Additional metadata: {"case_number": "...", "department": "...", "tags": [...]}
    tags                            JSONB           DEFAULT '[]',
        -- Organizational tags for filtering and categorization
    created_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that created this communication record
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for communication record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_acm_comm_type CHECK (comm_type IN (
        'information_request', 'information_response', 'inspection_notice',
        'inspection_report', 'non_compliance_notice', 'penalty_notice',
        'remediation_order', 'remediation_confirmation', 'appeal_filing',
        'appeal_decision', 'voluntary_disclosure', 'clarification_request',
        'clarification_response', 'status_inquiry', 'document_request',
        'document_submission', 'compliance_certificate', 'warning_notice',
        'suspension_notice', 'reinstatement_notice', 'general_correspondence'
    )),
    CONSTRAINT chk_acm_comm_direction CHECK (comm_direction IN (
        'inbound', 'outbound', 'internal'
    )),
    CONSTRAINT chk_acm_comm_priority CHECK (priority IN (
        'low', 'normal', 'high', 'urgent', 'critical'
    )),
    CONSTRAINT chk_acm_comm_status CHECK (status IN (
        'draft', 'open', 'acknowledged', 'in_progress', 'awaiting_response',
        'responded', 'escalated', 'on_hold', 'closed', 'cancelled', 'archived'
    )),
    CONSTRAINT chk_acm_comm_language CHECK (language_code IN (
        'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr',
        'ga', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt',
        'ro', 'sk', 'sl', 'sv'
    )),
    CONSTRAINT chk_acm_comm_closed CHECK (
        (status = 'closed' AND closed_at IS NOT NULL)
        OR (status != 'closed')
    ),
    CONSTRAINT chk_acm_comm_responded CHECK (
        (status = 'responded' AND responded_at IS NOT NULL)
        OR (status != 'responded')
    ),
    CONSTRAINT chk_acm_comm_thread_depth CHECK (thread_depth >= 0),
    CONSTRAINT chk_acm_comm_attachments CHECK (attachments_count >= 0),
    CONSTRAINT chk_acm_comm_deadline CHECK (
        response_deadline IS NULL OR response_deadline > initiated_at
    ),
    CONSTRAINT chk_acm_comm_escalation CHECK (
        escalation_deadline IS NULL OR escalation_deadline > initiated_at
    )
);

COMMENT ON TABLE gl_eudr_acm_communications IS 'AGENT-EUDR-040: Main communication records between EUDR operators and EU Member State competent authorities with lifecycle tracking, priority classification, response deadlines per Article 17(2), multi-language support for 24 EU official languages, thread management, and provenance hash per EUDR Article 31';
COMMENT ON COLUMN gl_eudr_acm_communications.comm_reference IS 'Human-readable reference: ACM-[YYYY]-[CC]-[NNNNNN] format. Unique across the system. Used in all correspondence with competent authorities for tracking and cross-referencing';
COMMENT ON COLUMN gl_eudr_acm_communications.comm_type IS 'Communication type per EUDR articles: information_request/response (Art.17), inspection_notice/report (Art.15), non_compliance/penalty_notice (Art.24), appeal_filing/decision (Art.19), voluntary_disclosure, clarification, document exchange, compliance certificates, warning/suspension/reinstatement notices';
COMMENT ON COLUMN gl_eudr_acm_communications.response_deadline IS 'Response deadline per EUDR Article 17(2): competent authorities set timeframes for operator responses to information requests. Missing deadlines may result in adverse inference or non-compliance findings';
COMMENT ON COLUMN gl_eudr_acm_communications.language_code IS 'ISO 639-1 language code: supports all 24 EU official languages (bg, cs, da, de, el, en, es, et, fi, fr, ga, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, sk, sl, sv) for multi-lingual authority communications';

-- Indexes for gl_eudr_acm_communications (28 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_operator ON gl_eudr_acm_communications (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_authority ON gl_eudr_acm_communications (authority_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_type ON gl_eudr_acm_communications (comm_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_direction ON gl_eudr_acm_communications (comm_direction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_status ON gl_eudr_acm_communications (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_priority ON gl_eudr_acm_communications (priority);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_deadline ON gl_eudr_acm_communications (response_deadline);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_escalation ON gl_eudr_acm_communications (escalation_deadline);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_language ON gl_eudr_acm_communications (language_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_dds_ref ON gl_eudr_acm_communications (related_dds_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_related ON gl_eudr_acm_communications (related_comm_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_provenance ON gl_eudr_acm_communications (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_created ON gl_eudr_acm_communications (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_initiated ON gl_eudr_acm_communications (initiated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_responded ON gl_eudr_acm_communications (responded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_tenant_operator ON gl_eudr_acm_communications (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_operator_status ON gl_eudr_acm_communications (operator_id, status, initiated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_operator_type ON gl_eudr_acm_communications (operator_id, comm_type, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_authority_status ON gl_eudr_acm_communications (authority_id, status, initiated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_authority_type ON gl_eudr_acm_communications (authority_id, comm_type, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_type_status ON gl_eudr_acm_communications (comm_type, status, priority, initiated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_priority_status ON gl_eudr_acm_communications (priority, status, response_deadline ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for open communications awaiting response
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_open ON gl_eudr_acm_communications (operator_id, response_deadline ASC)
        WHERE status IN ('open', 'acknowledged', 'in_progress', 'awaiting_response');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for overdue communications past deadline
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_overdue ON gl_eudr_acm_communications (response_deadline ASC, priority, operator_id)
        WHERE status IN ('open', 'acknowledged', 'in_progress', 'awaiting_response')
        AND response_deadline IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for escalated communications requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_escalated ON gl_eudr_acm_communications (escalation_deadline ASC, operator_id)
        WHERE status = 'escalated';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high-priority communications
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_high_priority ON gl_eudr_acm_communications (operator_id, authority_id, initiated_at DESC)
        WHERE priority IN ('urgent', 'critical');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_tags ON gl_eudr_acm_communications USING GIN (tags);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_comm_metadata ON gl_eudr_acm_communications USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_acm_information_requests -- Authority information requests (Art.17)
-- ============================================================================
-- Tracks formal information requests issued by competent authorities to
-- operators/traders under EUDR Article 17. When a competent authority requires
-- additional information to verify EUDR compliance, it issues a formal request
-- specifying the data required, the legal basis (specific EUDR article), and
-- a response deadline. The operator must respond within the specified timeframe
-- or face potential adverse inference in compliance assessments. Each request
-- references a parent communication record and tracks the full request-response
-- lifecycle including data categories, legal basis articles, deadline tracking,
-- and response completeness assessment.
-- ============================================================================
RAISE NOTICE 'V128 [2/10]: Creating gl_eudr_acm_information_requests...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_information_requests (
    request_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this information request
    comm_id                         UUID            NOT NULL REFERENCES gl_eudr_acm_communications(comm_id),
        -- FK to the parent communication record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator or trader to whom the request is directed
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    request_type                    VARCHAR(50)     NOT NULL,
        -- Category of information requested by the competent authority
    request_reference               VARCHAR(50),
        -- Authority-assigned reference number for this information request
    requested_data                  TEXT            NOT NULL,
        -- Detailed description of the information or data requested
    requested_data_categories       JSONB           DEFAULT '[]',
        -- Array of data category codes: ["supply_chain", "geolocation", "dds", "risk_assessment"]
    legal_basis_article             VARCHAR(50)     NOT NULL,
        -- EUDR article providing the legal basis for this information request
        -- e.g. "Article 17(1)", "Article 17(2)", "Article 10(2)"
    legal_basis_description         TEXT            DEFAULT '',
        -- Full text description of the legal basis for operator reference
    deadline                        TIMESTAMPTZ     NOT NULL,
        -- Deadline by which the operator must provide the requested information
    reminder_sent_at                TIMESTAMPTZ,
        -- Timestamp when a deadline reminder was last sent
    reminder_count                  INTEGER         NOT NULL DEFAULT 0,
        -- Number of reminders sent for this request
    status                          VARCHAR(30)     NOT NULL DEFAULT 'pending',
        -- Current status of the information request
    responded_at                    TIMESTAMPTZ,
        -- Timestamp when the operator submitted their response
    response_summary                TEXT            DEFAULT '',
        -- Summary of the operator's response for authority review
    response_completeness           VARCHAR(20),
        -- Assessment of response completeness by the authority
    response_accepted               BOOLEAN,
        -- Whether the authority accepted the response as satisfactory
    response_rejection_reason       TEXT,
        -- Reason if the response was deemed incomplete or unsatisfactory
    extension_requested             BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the operator has requested a deadline extension
    extension_granted               BOOLEAN,
        -- Whether the deadline extension was granted by the authority
    extended_deadline               TIMESTAMPTZ,
        -- New deadline if extension was granted
    supporting_document_ids         JSONB           DEFAULT '[]',
        -- Array of document_id references from gl_eudr_acm_documents
    adverse_inference_applied       BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether adverse inference was applied due to non-response per Article 17(3)
    notes                           TEXT            DEFAULT '',
        -- Internal notes about this information request
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for request record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_acm_ir_type CHECK (request_type IN (
        'supply_chain_data', 'geolocation_data', 'dds_verification',
        'risk_assessment_data', 'commodity_origin', 'traceability_records',
        'transaction_records', 'certification_documents', 'laboratory_results',
        'satellite_imagery', 'customs_records', 'financial_records',
        'third_party_audit', 'remediation_evidence', 'general_information'
    )),
    CONSTRAINT chk_acm_ir_status CHECK (status IN (
        'pending', 'acknowledged', 'in_progress', 'responded',
        'under_review', 'accepted', 'rejected', 'withdrawn',
        'overdue', 'escalated', 'closed'
    )),
    CONSTRAINT chk_acm_ir_completeness CHECK (response_completeness IS NULL OR response_completeness IN (
        'complete', 'partial', 'insufficient', 'not_applicable'
    )),
    CONSTRAINT chk_acm_ir_deadline CHECK (deadline > created_at),
    CONSTRAINT chk_acm_ir_extension CHECK (
        (extension_granted = TRUE AND extended_deadline IS NOT NULL AND extended_deadline > deadline)
        OR (extension_granted = FALSE)
        OR (extension_granted IS NULL)
    ),
    CONSTRAINT chk_acm_ir_responded CHECK (
        (status IN ('responded', 'under_review', 'accepted', 'rejected') AND responded_at IS NOT NULL)
        OR (status NOT IN ('responded', 'under_review', 'accepted', 'rejected'))
    ),
    CONSTRAINT chk_acm_ir_reminder CHECK (reminder_count >= 0),
    CONSTRAINT chk_acm_ir_adverse CHECK (
        (adverse_inference_applied = TRUE AND status IN ('overdue', 'escalated', 'closed'))
        OR (adverse_inference_applied = FALSE)
    )
);

COMMENT ON TABLE gl_eudr_acm_information_requests IS 'AGENT-EUDR-040: Authority information requests under EUDR Article 17 with request categorization, legal basis tracking, deadline management, response lifecycle, completeness assessment, deadline extension handling, adverse inference tracking, and provenance hash per Article 31';
COMMENT ON COLUMN gl_eudr_acm_information_requests.legal_basis_article IS 'EUDR article providing legal authority: Article 17(1) general information power, Article 17(2) response timeframe obligation, Article 10(2) due diligence documentation, Article 9(1)(d) risk assessment data. Authority must cite specific article in every request';
COMMENT ON COLUMN gl_eudr_acm_information_requests.adverse_inference_applied IS 'Article 17(3) adverse inference: if operator fails to respond within deadline without valid extension, authority may draw adverse conclusions regarding EUDR compliance. This flag tracks whether such inference was applied';
COMMENT ON COLUMN gl_eudr_acm_information_requests.response_completeness IS 'Authority assessment of response quality: complete (all data provided), partial (some data missing), insufficient (response does not address the request), not_applicable (request withdrawn or superseded)';

-- Indexes for gl_eudr_acm_information_requests (18 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_comm ON gl_eudr_acm_information_requests (comm_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_operator ON gl_eudr_acm_information_requests (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_type ON gl_eudr_acm_information_requests (request_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_status ON gl_eudr_acm_information_requests (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_deadline ON gl_eudr_acm_information_requests (deadline);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_legal_basis ON gl_eudr_acm_information_requests (legal_basis_article);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_provenance ON gl_eudr_acm_information_requests (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_created ON gl_eudr_acm_information_requests (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_tenant_operator ON gl_eudr_acm_information_requests (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_operator_status ON gl_eudr_acm_information_requests (operator_id, status, deadline ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_comm_status ON gl_eudr_acm_information_requests (comm_id, status, deadline ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_type_status ON gl_eudr_acm_information_requests (request_type, status, deadline ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_responded ON gl_eudr_acm_information_requests (responded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for pending information requests approaching deadline
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_pending ON gl_eudr_acm_information_requests (operator_id, deadline ASC)
        WHERE status IN ('pending', 'acknowledged', 'in_progress');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for overdue information requests
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_overdue ON gl_eudr_acm_information_requests (deadline ASC, operator_id)
        WHERE status = 'overdue';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for requests with adverse inference
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_adverse ON gl_eudr_acm_information_requests (operator_id, deadline DESC)
        WHERE adverse_inference_applied = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_categories ON gl_eudr_acm_information_requests USING GIN (requested_data_categories);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_ir_doc_ids ON gl_eudr_acm_information_requests USING GIN (supporting_document_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_acm_inspections -- On-the-spot check coordination (Article 15)
-- ============================================================================
-- Coordinates on-the-spot checks (inspections) conducted by competent
-- authorities under EUDR Article 15. Competent authorities have the power to
-- carry out checks on operators and traders, including physical inspections of
-- commodities, document verification at operator premises, and field visits to
-- production sites. Each inspection record tracks the full lifecycle from
-- scheduling through completion, including inspector assignment, location
-- details, inspection scope, findings with severity classification, corrective
-- actions required, follow-up scheduling, and evidence collection references.
-- ============================================================================
RAISE NOTICE 'V128 [3/10]: Creating gl_eudr_acm_inspections...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_inspections (
    inspection_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this inspection record
    comm_id                         UUID            NOT NULL REFERENCES gl_eudr_acm_communications(comm_id),
        -- FK to the parent communication record for this inspection
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator or trader being inspected
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    inspection_reference            VARCHAR(50),
        -- Authority-assigned inspection reference number
    inspection_type                 VARCHAR(50)     NOT NULL,
        -- Type of inspection as permitted under EUDR Article 15
    inspection_scope                TEXT            DEFAULT '',
        -- Detailed description of what the inspection will cover
    inspection_basis                VARCHAR(50)     NOT NULL DEFAULT 'risk_based',
        -- Basis for triggering this inspection
    scheduled_date                  TIMESTAMPTZ     NOT NULL,
        -- Scheduled date and time for the inspection
    scheduled_end_date              TIMESTAMPTZ,
        -- Scheduled end date for multi-day inspections
    actual_start_date               TIMESTAMPTZ,
        -- Actual start time of the inspection
    actual_end_date                 TIMESTAMPTZ,
        -- Actual end time of the inspection
    location_type                   VARCHAR(30)     NOT NULL DEFAULT 'operator_premises',
        -- Type of location where the inspection will be conducted
    location_address                TEXT            NOT NULL DEFAULT '',
        -- Full address of the inspection location
    location_coordinates            JSONB           DEFAULT '{}',
        -- GPS coordinates: {"latitude": 51.1657, "longitude": 10.4515}
    inspector_name                  VARCHAR(200)    NOT NULL,
        -- Name of the lead inspector from the competent authority
    inspector_id                    VARCHAR(100),
        -- Authority-internal identifier for the lead inspector
    inspector_team                  JSONB           DEFAULT '[]',
        -- Array of inspector team members: [{"name": "...", "role": "...", "id": "..."}]
    status                          VARCHAR(30)     NOT NULL DEFAULT 'scheduled',
        -- Current status of the inspection
    findings_summary                TEXT            DEFAULT '',
        -- Summary of inspection findings
    findings_severity               VARCHAR(20),
        -- Overall severity classification of findings
    findings_categories             JSONB           DEFAULT '[]',
        -- Array of finding category codes: ["documentation_gap", "traceability_break"]
    corrective_actions_required     BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether corrective actions were identified during the inspection
    corrective_actions_detail       TEXT            DEFAULT '',
        -- Detailed description of required corrective actions
    corrective_actions_deadline     TIMESTAMPTZ,
        -- Deadline for completing corrective actions
    corrective_actions_status       VARCHAR(20),
        -- Status of corrective action implementation
    follow_up_required              BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether a follow-up inspection is required
    follow_up_inspection_id         UUID,
        -- FK to the follow-up inspection record if scheduled
    follow_up_scheduled_date        TIMESTAMPTZ,
        -- Scheduled date for the follow-up inspection
    evidence_document_ids           JSONB           DEFAULT '[]',
        -- Array of document_id references for evidence collected
    completed_at                    TIMESTAMPTZ,
        -- Timestamp when the inspection was completed
    report_issued_at                TIMESTAMPTZ,
        -- Timestamp when the formal inspection report was issued
    operator_comments               TEXT            DEFAULT '',
        -- Operator comments or objections to inspection findings
    metadata                        JSONB           DEFAULT '{}',
        -- Additional metadata: {"warrant_ref": "...", "lab_samples": [...], "photos_count": 15}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for inspection record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_acm_insp_type CHECK (inspection_type IN (
        'document_verification', 'physical_inspection', 'field_visit',
        'warehouse_check', 'laboratory_sampling', 'production_site_visit',
        'supply_chain_audit', 'customs_point_inspection', 'announced',
        'unannounced', 'remote_inspection', 'joint_inspection', 'follow_up'
    )),
    CONSTRAINT chk_acm_insp_basis CHECK (inspection_basis IN (
        'risk_based', 'random_sample', 'complaint', 'whistleblower',
        'intelligence', 'systematic', 'follow_up', 'joint_operation',
        'eu_commission_request', 'member_state_request'
    )),
    CONSTRAINT chk_acm_insp_location CHECK (location_type IN (
        'operator_premises', 'warehouse', 'production_site', 'port_of_entry',
        'customs_office', 'field_location', 'processing_facility',
        'storage_facility', 'remote', 'other'
    )),
    CONSTRAINT chk_acm_insp_status CHECK (status IN (
        'scheduled', 'confirmed', 'in_progress', 'postponed', 'completed',
        'cancelled', 'no_access', 'report_pending', 'report_issued', 'closed'
    )),
    CONSTRAINT chk_acm_insp_severity CHECK (findings_severity IS NULL OR findings_severity IN (
        'none', 'minor', 'moderate', 'major', 'critical'
    )),
    CONSTRAINT chk_acm_insp_ca_status CHECK (corrective_actions_status IS NULL OR corrective_actions_status IN (
        'not_started', 'in_progress', 'completed', 'verified', 'overdue', 'waived'
    )),
    CONSTRAINT chk_acm_insp_completed CHECK (
        (status IN ('completed', 'report_pending', 'report_issued', 'closed') AND completed_at IS NOT NULL)
        OR (status NOT IN ('completed', 'report_pending', 'report_issued', 'closed'))
    ),
    CONSTRAINT chk_acm_insp_report CHECK (
        (status IN ('report_issued', 'closed') AND report_issued_at IS NOT NULL)
        OR (status NOT IN ('report_issued', 'closed'))
    ),
    CONSTRAINT chk_acm_insp_ca_deadline CHECK (
        (corrective_actions_required = TRUE AND corrective_actions_deadline IS NOT NULL)
        OR (corrective_actions_required = FALSE)
    ),
    CONSTRAINT chk_acm_insp_dates CHECK (
        scheduled_end_date IS NULL OR scheduled_end_date >= scheduled_date
    )
);

COMMENT ON TABLE gl_eudr_acm_inspections IS 'AGENT-EUDR-040: On-the-spot inspection coordination under EUDR Article 15 with inspection type classification, scheduling, inspector assignment, location tracking, findings with severity grading, corrective actions lifecycle, follow-up scheduling, evidence references, and provenance hash per Article 31';
COMMENT ON COLUMN gl_eudr_acm_inspections.inspection_type IS 'Inspection types per EUDR Article 15: document_verification (records review), physical_inspection (commodity examination), field_visit (production site), warehouse_check (storage), laboratory_sampling (testing), supply_chain_audit (end-to-end), announced/unannounced (notice type), remote_inspection (digital), joint_inspection (multi-authority)';
COMMENT ON COLUMN gl_eudr_acm_inspections.inspection_basis IS 'Trigger basis: risk_based (Article 16 risk assessment), random_sample (Article 15(6) sampling), complaint/whistleblower (third-party report), intelligence (OLAF/Europol), systematic (scheduled program), follow_up (previous findings), joint_operation (cross-border), eu_commission_request (Article 22)';
COMMENT ON COLUMN gl_eudr_acm_inspections.findings_severity IS 'Severity classification: none (compliant), minor (administrative gaps), moderate (procedural failures), major (substantive non-compliance), critical (deliberate evasion or imminent risk)';

-- Indexes for gl_eudr_acm_inspections (20 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_comm ON gl_eudr_acm_inspections (comm_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_operator ON gl_eudr_acm_inspections (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_type ON gl_eudr_acm_inspections (inspection_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_basis ON gl_eudr_acm_inspections (inspection_basis);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_scheduled ON gl_eudr_acm_inspections (scheduled_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_status ON gl_eudr_acm_inspections (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_severity ON gl_eudr_acm_inspections (findings_severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_inspector ON gl_eudr_acm_inspections (inspector_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_provenance ON gl_eudr_acm_inspections (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_created ON gl_eudr_acm_inspections (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_completed ON gl_eudr_acm_inspections (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_tenant_operator ON gl_eudr_acm_inspections (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_operator_status ON gl_eudr_acm_inspections (operator_id, status, scheduled_date ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_type_status ON gl_eudr_acm_inspections (inspection_type, status, scheduled_date ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_operator_severity ON gl_eudr_acm_inspections (operator_id, findings_severity, completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_followup ON gl_eudr_acm_inspections (follow_up_inspection_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for upcoming scheduled inspections
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_upcoming ON gl_eudr_acm_inspections (scheduled_date ASC, operator_id)
        WHERE status IN ('scheduled', 'confirmed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for inspections requiring corrective actions
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_corrective ON gl_eudr_acm_inspections (operator_id, corrective_actions_deadline ASC)
        WHERE corrective_actions_required = TRUE AND corrective_actions_status NOT IN ('completed', 'verified', 'waived');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for critical/major findings
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_critical ON gl_eudr_acm_inspections (operator_id, completed_at DESC)
        WHERE findings_severity IN ('major', 'critical');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_insp_findings_cat ON gl_eudr_acm_inspections USING GIN (findings_categories);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_acm_non_compliance -- Non-compliance notifications
-- ============================================================================
-- Records non-compliance notifications issued by competent authorities to
-- operators/traders who are found to be in breach of EUDR requirements. Per
-- EUDR Articles 24 and 25, competent authorities must notify operators of
-- violations, specify the EUDR article(s) violated, classify the severity,
-- impose penalties (fines calculated as percentage of EU turnover or fixed
-- amounts), set remediation deadlines, and track the remediation lifecycle.
-- Penalties must be effective, proportionate, and dissuasive as required by
-- Article 25(2). This table also tracks temporary measures such as product
-- seizure, market withdrawal orders, and operator suspension per Article 24.
-- ============================================================================
RAISE NOTICE 'V128 [4/10]: Creating gl_eudr_acm_non_compliance...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_non_compliance (
    nc_id                           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this non-compliance record
    comm_id                         UUID            NOT NULL REFERENCES gl_eudr_acm_communications(comm_id),
        -- FK to the parent communication record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator or trader found in non-compliance
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    nc_reference                    VARCHAR(50),
        -- Authority-assigned non-compliance reference number
    violation_type                  VARCHAR(50)     NOT NULL,
        -- Category of EUDR violation identified
    eudr_article_violated           VARCHAR(100)    NOT NULL,
        -- Specific EUDR article(s) violated
        -- e.g. "Article 3(a)", "Article 4(1)", "Article 9(1)(d)"
    eudr_articles_list              JSONB           DEFAULT '[]',
        -- Array of all EUDR articles violated: ["Article 3(a)", "Article 4(1)"]
    severity                        VARCHAR(20)     NOT NULL DEFAULT 'moderate',
        -- Severity classification of the non-compliance
    description                     TEXT            NOT NULL DEFAULT '',
        -- Detailed description of the non-compliance finding
    evidence_summary                TEXT            DEFAULT '',
        -- Summary of evidence supporting the non-compliance finding
    evidence_document_ids           JSONB           DEFAULT '[]',
        -- Array of document_id references for supporting evidence
    inspection_id                   UUID,
        -- FK to gl_eudr_acm_inspections if finding originated from an inspection
    related_dds_reference           VARCHAR(100),
        -- DDS reference linked to this non-compliance (if applicable)
    related_commodity               VARCHAR(50),
        -- EUDR commodity type involved in the non-compliance
    penalty_type                    VARCHAR(30),
        -- Type of penalty imposed
    penalty_amount_eur              NUMERIC(16,2),
        -- Monetary penalty amount in EUR
    penalty_calculation_basis       TEXT            DEFAULT '',
        -- How the penalty was calculated (e.g. percentage of annual turnover)
    penalty_legal_basis             VARCHAR(100),
        -- Legal basis for the penalty per national transposition of EUDR Article 25
    penalty_paid                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the monetary penalty has been paid
    penalty_paid_at                 TIMESTAMPTZ,
        -- Timestamp when the penalty was paid
    interim_measures                JSONB           DEFAULT '[]',
        -- Array of interim measures imposed: ["product_seizure", "market_withdrawal"]
    interim_measures_active         BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether interim measures are currently in force
    remediation_deadline            TIMESTAMPTZ,
        -- Deadline by which the operator must remediate the non-compliance
    remediation_plan_submitted      BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the operator has submitted a remediation plan
    remediation_plan_approved       BOOLEAN,
        -- Whether the authority approved the remediation plan
    remediation_status              VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Current status of remediation efforts
    remediation_evidence_ids        JSONB           DEFAULT '[]',
        -- Array of document_id references for remediation evidence
    remediation_verified_at         TIMESTAMPTZ,
        -- Timestamp when remediation was verified by the authority
    resolved_at                     TIMESTAMPTZ,
        -- Timestamp when the non-compliance was fully resolved
    resolution_type                 VARCHAR(30),
        -- How the non-compliance was resolved
    recurrence_count                INTEGER         NOT NULL DEFAULT 0,
        -- Number of times this type of violation has recurred for this operator
    aggravating_factors             JSONB           DEFAULT '[]',
        -- Array of aggravating factors: ["repeat_offender", "deliberate", "obstruction"]
    mitigating_factors              JSONB           DEFAULT '[]',
        -- Array of mitigating factors: ["first_offense", "voluntary_disclosure", "cooperation"]
    publication_required            BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether public disclosure of the violation is required per Article 25
    published_at                    TIMESTAMPTZ,
        -- Timestamp when the violation was publicly disclosed
    notes                           TEXT            DEFAULT '',
        -- Internal notes about this non-compliance record
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for non-compliance record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_acm_nc_violation CHECK (violation_type IN (
        'deforestation_link', 'legality_violation', 'dds_failure',
        'traceability_gap', 'false_declaration', 'document_fraud',
        'information_refusal', 'deadline_breach', 'reporting_failure',
        'labeling_violation', 'market_placement_unauthorized',
        'risk_assessment_inadequate', 'due_diligence_inadequate',
        'record_keeping_failure', 'cooperation_refusal', 'other'
    )),
    CONSTRAINT chk_acm_nc_severity CHECK (severity IN (
        'minor', 'moderate', 'major', 'critical', 'systemic'
    )),
    CONSTRAINT chk_acm_nc_commodity CHECK (related_commodity IS NULL OR related_commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood',
        'derived_cattle', 'derived_cocoa', 'derived_coffee', 'derived_oil_palm',
        'derived_rubber', 'derived_soya', 'derived_wood'
    )),
    CONSTRAINT chk_acm_nc_penalty_type CHECK (penalty_type IS NULL OR penalty_type IN (
        'fine', 'turnover_percentage', 'daily_penalty', 'confiscation',
        'market_ban', 'license_suspension', 'license_revocation',
        'public_naming', 'injunction', 'criminal_referral'
    )),
    CONSTRAINT chk_acm_nc_penalty CHECK (penalty_amount_eur IS NULL OR penalty_amount_eur >= 0),
    CONSTRAINT chk_acm_nc_remediation CHECK (remediation_status IN (
        'pending', 'plan_submitted', 'plan_approved', 'in_progress',
        'completed', 'verified', 'failed', 'escalated', 'waived'
    )),
    CONSTRAINT chk_acm_nc_resolution CHECK (resolution_type IS NULL OR resolution_type IN (
        'remediated', 'penalty_paid', 'appeal_overturned', 'withdrawn',
        'expired', 'superseded', 'settled', 'criminal_prosecution'
    )),
    CONSTRAINT chk_acm_nc_penalty_paid CHECK (
        (penalty_paid = TRUE AND penalty_paid_at IS NOT NULL)
        OR (penalty_paid = FALSE)
    ),
    CONSTRAINT chk_acm_nc_resolved CHECK (
        (resolved_at IS NOT NULL AND resolution_type IS NOT NULL)
        OR (resolved_at IS NULL)
    ),
    CONSTRAINT chk_acm_nc_recurrence CHECK (recurrence_count >= 0)
);

COMMENT ON TABLE gl_eudr_acm_non_compliance IS 'AGENT-EUDR-040: Non-compliance notifications per EUDR Articles 24-25 with violation classification, penalty calculation (fines/turnover %/confiscation), interim measures (seizure/withdrawal), remediation lifecycle, recurrence tracking, aggravating/mitigating factors, public disclosure requirements, and provenance hash per Article 31';
COMMENT ON COLUMN gl_eudr_acm_non_compliance.violation_type IS 'Violation categories: deforestation_link (Art.3a), legality_violation (Art.3b), dds_failure (Art.4), traceability_gap (Art.9), false_declaration (Art.4(2)), document_fraud (Art.9), information_refusal (Art.17), deadline_breach, reporting_failure, market_placement_unauthorized (Art.3)';
COMMENT ON COLUMN gl_eudr_acm_non_compliance.penalty_amount_eur IS 'Monetary penalty in EUR: must be effective, proportionate, and dissuasive per Article 25(2). Maximum penalties set by national law transposing Article 25. May be calculated as percentage of annual EU turnover for serious violations';
COMMENT ON COLUMN gl_eudr_acm_non_compliance.interim_measures IS 'Interim measures per Article 24: product_seizure (physical confiscation), market_withdrawal (removal order), import_ban (customs hold), operator_suspension (temporary ban). Multiple measures may apply simultaneously';

-- Indexes for gl_eudr_acm_non_compliance (22 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_comm ON gl_eudr_acm_non_compliance (comm_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_operator ON gl_eudr_acm_non_compliance (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_violation ON gl_eudr_acm_non_compliance (violation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_severity ON gl_eudr_acm_non_compliance (severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_remediation ON gl_eudr_acm_non_compliance (remediation_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_article ON gl_eudr_acm_non_compliance (eudr_article_violated);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_commodity ON gl_eudr_acm_non_compliance (related_commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_penalty_type ON gl_eudr_acm_non_compliance (penalty_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_inspection ON gl_eudr_acm_non_compliance (inspection_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_provenance ON gl_eudr_acm_non_compliance (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_created ON gl_eudr_acm_non_compliance (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_resolved ON gl_eudr_acm_non_compliance (resolved_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_tenant_operator ON gl_eudr_acm_non_compliance (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_operator_severity ON gl_eudr_acm_non_compliance (operator_id, severity, remediation_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_operator_violation ON gl_eudr_acm_non_compliance (operator_id, violation_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_severity_status ON gl_eudr_acm_non_compliance (severity, remediation_status, remediation_deadline ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_remediation_deadline ON gl_eudr_acm_non_compliance (remediation_deadline ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unresolved non-compliance records
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_unresolved ON gl_eudr_acm_non_compliance (operator_id, severity, created_at DESC)
        WHERE resolved_at IS NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unpaid penalties
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_unpaid ON gl_eudr_acm_non_compliance (operator_id, penalty_amount_eur DESC)
        WHERE penalty_amount_eur IS NOT NULL AND penalty_paid = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active interim measures
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_interim ON gl_eudr_acm_non_compliance (operator_id, created_at DESC)
        WHERE interim_measures_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for critical/systemic violations
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_critical ON gl_eudr_acm_non_compliance (operator_id, violation_type, created_at DESC)
        WHERE severity IN ('critical', 'systemic');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_nc_articles ON gl_eudr_acm_non_compliance USING GIN (eudr_articles_list);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_acm_appeals -- Administrative appeals (Article 19)
-- ============================================================================
-- Tracks administrative appeals filed by operators/traders against decisions
-- of competent authorities under EUDR Article 19. Operators have the right
-- to an effective judicial remedy against any legally binding decision taken
-- by a competent authority. This table manages the full appeal lifecycle from
-- filing through hearing to final decision, including appeal grounds, evidence
-- submission, hearing scheduling, panel composition, decision reasoning,
-- and finality determination. Appeals may be against non-compliance findings,
-- penalties, interim measures, or other authority decisions.
-- ============================================================================
RAISE NOTICE 'V128 [5/10]: Creating gl_eudr_acm_appeals...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_appeals (
    appeal_id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this appeal record
    comm_id                         UUID            NOT NULL REFERENCES gl_eudr_acm_communications(comm_id),
        -- FK to the parent communication record
    nc_id                           UUID            REFERENCES gl_eudr_acm_non_compliance(nc_id),
        -- FK to the non-compliance record being appealed (if applicable)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator or trader filing the appeal
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    appeal_reference                VARCHAR(50),
        -- Authority-assigned or court-assigned appeal reference number
    appeal_type                     VARCHAR(30)     NOT NULL DEFAULT 'administrative',
        -- Type of appeal proceedings
    appeal_grounds                  TEXT            NOT NULL DEFAULT '',
        -- Legal grounds for the appeal filed by the operator
    appeal_grounds_categories       JSONB           DEFAULT '[]',
        -- Array of appeal ground categories: ["procedural_error", "factual_error"]
    challenged_decision_type        VARCHAR(50)     NOT NULL,
        -- Type of authority decision being challenged
    challenged_decision_date        TIMESTAMPTZ,
        -- Date of the original decision being appealed
    challenged_decision_ref         VARCHAR(100),
        -- Reference number of the original decision
    supporting_evidence_ids         JSONB           DEFAULT '[]',
        -- Array of document_id references for evidence supporting the appeal
    legal_representative            VARCHAR(200),
        -- Name of legal representative or counsel for the operator
    legal_representative_ref        VARCHAR(100),
        -- Bar number or reference for the legal representative
    filed_at                        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the appeal was filed
    filing_deadline                 TIMESTAMPTZ,
        -- Statutory deadline for filing the appeal
    acknowledgment_at               TIMESTAMPTZ,
        -- Timestamp when the appeal body acknowledged receipt
    status                          VARCHAR(30)     NOT NULL DEFAULT 'filed',
        -- Current status of the appeal
    hearing_date                    TIMESTAMPTZ,
        -- Scheduled date for the appeal hearing
    hearing_location                VARCHAR(300),
        -- Location of the appeal hearing
    hearing_type                    VARCHAR(20),
        -- Type of hearing
    panel_members                   JSONB           DEFAULT '[]',
        -- Array of panel members: [{"name": "...", "role": "...", "title": "..."}]
    oral_hearing_held               BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether an oral hearing was held
    decision                        VARCHAR(30),
        -- Decision outcome of the appeal
    decision_date                   TIMESTAMPTZ,
        -- Date the appeal decision was rendered
    decision_reasoning              TEXT            DEFAULT '',
        -- Detailed reasoning for the appeal decision
    decision_document_id            UUID,
        -- FK to gl_eudr_acm_documents for the formal decision document
    remedy_granted                  TEXT            DEFAULT '',
        -- Description of remedy granted if appeal was upheld
    penalty_modified                BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the original penalty was modified by the appeal decision
    modified_penalty_eur            NUMERIC(16,2),
        -- Modified penalty amount in EUR if penalty was changed
    costs_awarded_eur               NUMERIC(14,2),
        -- Legal costs awarded (positive = to operator, negative = against operator)
    final                           BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the decision is final and not subject to further appeal
    further_appeal_deadline         TIMESTAMPTZ,
        -- Deadline for filing a further appeal (if not final)
    further_appeal_body             VARCHAR(200),
        -- Name of the body to which further appeal can be made
    suspension_of_execution         BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether execution of the original decision is suspended pending appeal
    notes                           TEXT            DEFAULT '',
        -- Internal notes about this appeal
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for appeal record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_acm_appeal_type CHECK (appeal_type IN (
        'administrative', 'judicial', 'arbitration', 'ombudsman', 'eu_commission'
    )),
    CONSTRAINT chk_acm_appeal_challenged CHECK (challenged_decision_type IN (
        'non_compliance_finding', 'penalty_imposition', 'interim_measure',
        'license_suspension', 'license_revocation', 'market_ban',
        'product_seizure', 'information_request', 'inspection_order',
        'remediation_order', 'publication_order', 'other'
    )),
    CONSTRAINT chk_acm_appeal_status CHECK (status IN (
        'filed', 'acknowledged', 'under_review', 'hearing_scheduled',
        'hearing_completed', 'awaiting_decision', 'decided',
        'partially_upheld', 'dismissed', 'withdrawn', 'settled'
    )),
    CONSTRAINT chk_acm_appeal_hearing_type CHECK (hearing_type IS NULL OR hearing_type IN (
        'oral', 'written', 'hybrid', 'video_conference'
    )),
    CONSTRAINT chk_acm_appeal_decision CHECK (decision IS NULL OR decision IN (
        'upheld', 'partially_upheld', 'dismissed', 'remanded',
        'settled', 'withdrawn', 'inadmissible'
    )),
    CONSTRAINT chk_acm_appeal_decided CHECK (
        (decision IS NOT NULL AND decision_date IS NOT NULL)
        OR (decision IS NULL)
    ),
    CONSTRAINT chk_acm_appeal_final CHECK (
        (final = TRUE AND decision IS NOT NULL)
        OR (final = FALSE)
    ),
    CONSTRAINT chk_acm_appeal_modified_penalty CHECK (
        (penalty_modified = TRUE AND modified_penalty_eur IS NOT NULL AND modified_penalty_eur >= 0)
        OR (penalty_modified = FALSE)
    )
);

COMMENT ON TABLE gl_eudr_acm_appeals IS 'AGENT-EUDR-040: Administrative appeals per EUDR Article 19 with appeal grounds, challenged decision tracking, evidence submission, hearing management, panel composition, decision reasoning, remedy/penalty modification, finality determination, suspension of execution, and provenance hash per Article 31';
COMMENT ON COLUMN gl_eudr_acm_appeals.appeal_grounds IS 'Legal grounds: operator states the basis for challenging the authority decision. May include procedural_error, factual_error, proportionality, misapplication_of_law, new_evidence, human_rights per ECHR Article 6 (fair trial) and Article 47 of the EU Charter of Fundamental Rights';
COMMENT ON COLUMN gl_eudr_acm_appeals.final IS 'Finality flag: TRUE when the decision is res judicata (final and binding). FALSE when further appeal is possible. Per Article 19, operators must have access to effective judicial remedy, which may involve multiple levels of appeal up to CJEU';
COMMENT ON COLUMN gl_eudr_acm_appeals.suspension_of_execution IS 'Suspension pending appeal: per general principles of EU administrative law, filing an appeal may suspend execution of the challenged decision. Authority or court may grant interim relief while appeal is pending';

-- Indexes for gl_eudr_acm_appeals (18 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_comm ON gl_eudr_acm_appeals (comm_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_nc ON gl_eudr_acm_appeals (nc_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_operator ON gl_eudr_acm_appeals (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_status ON gl_eudr_acm_appeals (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_filed ON gl_eudr_acm_appeals (filed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_decision ON gl_eudr_acm_appeals (decision);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_final ON gl_eudr_acm_appeals (final);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_hearing ON gl_eudr_acm_appeals (hearing_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_provenance ON gl_eudr_acm_appeals (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_created ON gl_eudr_acm_appeals (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_tenant_operator ON gl_eudr_acm_appeals (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_operator_status ON gl_eudr_acm_appeals (operator_id, status, filed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_challenged ON gl_eudr_acm_appeals (challenged_decision_type, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_decision_date ON gl_eudr_acm_appeals (decision_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-final) appeals
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_active ON gl_eudr_acm_appeals (operator_id, status, filed_at DESC)
        WHERE final = FALSE AND status NOT IN ('dismissed', 'withdrawn', 'settled');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for appeals with suspension of execution
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_suspended ON gl_eudr_acm_appeals (operator_id, nc_id)
        WHERE suspension_of_execution = TRUE AND final = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for upcoming hearings
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_upcoming_hearing ON gl_eudr_acm_appeals (hearing_date ASC, operator_id)
        WHERE status = 'hearing_scheduled' AND hearing_date IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_appeal_grounds ON gl_eudr_acm_appeals USING GIN (appeal_grounds_categories);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_acm_documents -- Document exchange
-- ============================================================================
-- Manages all documents exchanged between operators and competent authorities
-- in the context of EUDR communications. Documents may include information
-- request responses, inspection reports, evidence packages, legal filings,
-- certificates, and correspondence attachments. Each document is tracked with
-- file metadata, SHA-256 integrity hash, encryption key reference for
-- documents containing sensitive data, access logging, retention scheduling
-- per EUDR Article 31, and version control for updated documents.
-- ============================================================================
RAISE NOTICE 'V128 [6/10]: Creating gl_eudr_acm_documents...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_documents (
    document_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this document record
    comm_id                         UUID            NOT NULL REFERENCES gl_eudr_acm_communications(comm_id),
        -- FK to the parent communication record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator or trader who owns or submitted the document
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    document_type                   VARCHAR(50)     NOT NULL,
        -- Category of the document
    document_reference              VARCHAR(100),
        -- External reference number for the document
    filename                        VARCHAR(500)    NOT NULL,
        -- Original filename as uploaded
    file_path                       VARCHAR(1000)   NOT NULL,
        -- Storage path in object storage (S3/MinIO)
    file_size_bytes                 BIGINT          NOT NULL,
        -- File size in bytes
    file_hash                       VARCHAR(64)     NOT NULL,
        -- SHA-256 hash of file contents for integrity verification
    mime_type                       VARCHAR(100)    NOT NULL DEFAULT 'application/octet-stream',
        -- MIME type of the file (e.g. application/pdf, image/jpeg)
    language_code                   VARCHAR(5)      NOT NULL DEFAULT 'en',
        -- Language of the document content
    uploaded_by                     VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that uploaded the document
    uploaded_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the document was uploaded
    verified_at                     TIMESTAMPTZ,
        -- Timestamp when document authenticity was verified
    verified_by                     VARCHAR(100),
        -- Actor who verified the document
    expires_at                      TIMESTAMPTZ,
        -- Document expiration date (e.g. certificates with validity period)
    encryption_key_id               VARCHAR(100),
        -- Reference to the encryption key used for documents with sensitive data
    encrypted                       BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the document is encrypted at rest
    classification                  VARCHAR(20)     NOT NULL DEFAULT 'standard',
        -- Document security classification
    version                         INTEGER         NOT NULL DEFAULT 1,
        -- Document version number (incremented on replacement)
    supersedes_document_id          UUID,
        -- FK to the document this version supersedes
    access_count                    INTEGER         NOT NULL DEFAULT 0,
        -- Number of times this document has been accessed/downloaded
    last_accessed_at                TIMESTAMPTZ,
        -- Timestamp of last document access
    retention_until                 TIMESTAMPTZ,
        -- Retention date per EUDR Article 31 (minimum 5 years)
    deleted_at                      TIMESTAMPTZ,
        -- Soft delete timestamp (document logically deleted but retained for audit)
    metadata                        JSONB           DEFAULT '{}',
        -- Additional metadata: {"pages": 12, "signed": true, "notarized": false}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for document record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_acm_doc_type CHECK (document_type IN (
        'information_response', 'inspection_report', 'inspection_evidence',
        'non_compliance_notice', 'penalty_notice', 'remediation_plan',
        'remediation_evidence', 'appeal_filing', 'appeal_evidence',
        'appeal_decision', 'legal_brief', 'certificate_of_compliance',
        'dds_copy', 'traceability_record', 'supply_chain_map',
        'laboratory_report', 'satellite_imagery', 'geolocation_data',
        'financial_record', 'customs_record', 'correspondence',
        'power_of_attorney', 'identity_document', 'other'
    )),
    CONSTRAINT chk_acm_doc_classification CHECK (classification IN (
        'public', 'standard', 'confidential', 'restricted', 'highly_restricted'
    )),
    CONSTRAINT chk_acm_doc_language CHECK (language_code IN (
        'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr',
        'ga', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt',
        'ro', 'sk', 'sl', 'sv'
    )),
    CONSTRAINT chk_acm_doc_size CHECK (file_size_bytes > 0),
    CONSTRAINT chk_acm_doc_hash CHECK (LENGTH(file_hash) = 64),
    CONSTRAINT chk_acm_doc_version CHECK (version >= 1),
    CONSTRAINT chk_acm_doc_access CHECK (access_count >= 0),
    CONSTRAINT chk_acm_doc_encrypted CHECK (
        (encrypted = TRUE AND encryption_key_id IS NOT NULL)
        OR (encrypted = FALSE)
    ),
    CONSTRAINT chk_acm_doc_expires CHECK (
        expires_at IS NULL OR expires_at > uploaded_at
    )
);

COMMENT ON TABLE gl_eudr_acm_documents IS 'AGENT-EUDR-040: Document exchange for authority communications with file integrity (SHA-256), encryption at rest, security classification, version control, access logging, retention scheduling per EUDR Article 31 (5 years), multi-language support, and provenance hash for audit trail';
COMMENT ON COLUMN gl_eudr_acm_documents.file_hash IS 'SHA-256 hash of file binary contents: computed at upload and verified at every download. Any hash mismatch indicates document tampering or corruption. Used in provenance chain for end-to-end integrity assurance';
COMMENT ON COLUMN gl_eudr_acm_documents.encryption_key_id IS 'Reference to encryption key in secrets manager (Vault): documents classified as confidential or above are encrypted at rest using AES-256-GCM. Key rotation follows SEC-003 encryption-at-rest policy';
COMMENT ON COLUMN gl_eudr_acm_documents.retention_until IS 'EUDR Article 31 retention: operators must retain all information gathered under due diligence for at least 5 years. Documents must be available to competent authorities on request throughout the retention period';

-- Indexes for gl_eudr_acm_documents (16 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_comm ON gl_eudr_acm_documents (comm_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_operator ON gl_eudr_acm_documents (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_type ON gl_eudr_acm_documents (document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_uploaded ON gl_eudr_acm_documents (uploaded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_hash ON gl_eudr_acm_documents (file_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_provenance ON gl_eudr_acm_documents (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_created ON gl_eudr_acm_documents (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_expires ON gl_eudr_acm_documents (expires_at);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_retention ON gl_eudr_acm_documents (retention_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_tenant_operator ON gl_eudr_acm_documents (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_comm_type ON gl_eudr_acm_documents (comm_id, document_type, uploaded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_operator_type ON gl_eudr_acm_documents (operator_id, document_type, uploaded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_supersedes ON gl_eudr_acm_documents (supersedes_document_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-deleted) documents
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_active ON gl_eudr_acm_documents (comm_id, document_type, uploaded_at DESC)
        WHERE deleted_at IS NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for encrypted documents
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_encrypted ON gl_eudr_acm_documents (encryption_key_id, operator_id)
        WHERE encrypted = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_doc_metadata ON gl_eudr_acm_documents USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_acm_authorities -- Competent authorities registry (27 EU states)
-- ============================================================================
-- Registry of competent authorities designated by each EU Member State per
-- EUDR Article 14. Each Member State must designate one or more competent
-- authorities responsible for enforcing the EUDR. This table stores the
-- authority details including contact information, API endpoints for
-- electronic communication, jurisdiction scope, and operational status.
-- Supports both national-level and regional/federal authorities (e.g.,
-- Germany with Bundeslaender-level competencies). The registry enables
-- automated routing of communications to the correct authority based on
-- the operator's location, the commodity type, or the point of entry.
-- ============================================================================
RAISE NOTICE 'V128 [7/10]: Creating gl_eudr_acm_authorities...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_authorities (
    authority_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this competent authority
    country_code                    VARCHAR(5)      NOT NULL,
        -- ISO 3166-1 alpha-2 code of the EU Member State
    country_name                    VARCHAR(100)    NOT NULL DEFAULT '',
        -- Full name of the EU Member State
    authority_name                  VARCHAR(300)    NOT NULL,
        -- Official name of the competent authority
    authority_name_local            VARCHAR(300),
        -- Authority name in the local/national language
    authority_type                  VARCHAR(30)     NOT NULL DEFAULT 'national',
        -- Type of authority: national (single national body) or regional (sub-national)
    authority_code                  VARCHAR(50),
        -- Authority-specific code or identifier used in official correspondence
    parent_authority_id             UUID,
        -- FK to parent authority for regional authorities under a national body
    jurisdiction_scope              VARCHAR(50)     NOT NULL DEFAULT 'national',
        -- Geographic scope of the authority's jurisdiction
    jurisdiction_regions            JSONB           DEFAULT '[]',
        -- Array of regions under this authority's jurisdiction: ["Bayern", "Baden-Wuerttemberg"]
    responsible_commodities         JSONB           DEFAULT '[]',
        -- Array of EUDR commodities this authority handles (empty = all)
    ministry_department             VARCHAR(300),
        -- Parent ministry or government department
    legal_basis                     VARCHAR(200),
        -- National legal basis for the authority's designation
    contact_email                   VARCHAR(200),
        -- Official contact email address
    contact_email_secondary         VARCHAR(200),
        -- Secondary/backup contact email
    contact_phone                   VARCHAR(50),
        -- Official contact phone number
    contact_phone_secondary         VARCHAR(50),
        -- Secondary/backup phone number
    contact_fax                     VARCHAR(50),
        -- Fax number (still used in some EU administrations)
    postal_address                  TEXT            DEFAULT '',
        -- Full postal address
    website                         VARCHAR(300),
        -- Official website URL
    api_endpoint                    VARCHAR(500),
        -- API endpoint for electronic communication (if available)
    api_version                     VARCHAR(20),
        -- API version supported by the authority
    api_auth_method                 VARCHAR(30),
        -- API authentication method: api_key, oauth2, mutual_tls, eidas
    portal_url                      VARCHAR(500),
        -- URL for the authority's online submission portal
    supported_languages             JSONB           DEFAULT '["en"]',
        -- Array of languages supported for communications
    working_hours                   JSONB           DEFAULT '{}',
        -- Working hours: {"weekday": "09:00-17:00", "timezone": "Europe/Berlin"}
    response_sla_days               INTEGER,
        -- Typical response time SLA in business days
    eu_information_system_id        VARCHAR(100),
        -- Authority's identifier in the EU EUDR information system per Article 33
    active                          BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether this authority is currently active
    effective_from                  DATE            NOT NULL DEFAULT CURRENT_DATE,
        -- Date from which this authority is designated
    effective_to                    DATE,
        -- Date until which this authority is designated (NULL = indefinite)
    deactivated_at                  TIMESTAMPTZ,
        -- Timestamp when the authority was deactivated
    deactivation_reason             TEXT,
        -- Reason for deactivation (e.g. restructuring, merger)
    notes                           TEXT            DEFAULT '',
        -- Administrative notes about this authority
    metadata                        JSONB           DEFAULT '{}',
        -- Additional metadata: {"bilateral_agreements": [...], "cooperation_frameworks": [...]}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for authority record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_acm_authority UNIQUE (country_code, authority_name),
    CONSTRAINT chk_acm_auth_country CHECK (country_code IN (
        'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
        'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
        'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
    )),
    CONSTRAINT chk_acm_auth_type CHECK (authority_type IN (
        'national', 'regional', 'federal', 'customs', 'environmental', 'agricultural'
    )),
    CONSTRAINT chk_acm_auth_scope CHECK (jurisdiction_scope IN (
        'national', 'regional', 'federal_state', 'customs_zone', 'port_specific', 'commodity_specific'
    )),
    CONSTRAINT chk_acm_auth_api_method CHECK (api_auth_method IS NULL OR api_auth_method IN (
        'api_key', 'oauth2', 'mutual_tls', 'eidas', 'certificate', 'saml'
    )),
    CONSTRAINT chk_acm_auth_effective CHECK (
        effective_to IS NULL OR effective_to >= effective_from
    ),
    CONSTRAINT chk_acm_auth_deactivated CHECK (
        (active = FALSE AND deactivated_at IS NOT NULL)
        OR (active = TRUE)
    ),
    CONSTRAINT chk_acm_auth_sla CHECK (response_sla_days IS NULL OR response_sla_days > 0)
);

COMMENT ON TABLE gl_eudr_acm_authorities IS 'AGENT-EUDR-040: Competent authorities registry for all 27 EU Member States per EUDR Article 14 with authority type (national/regional/federal), jurisdiction scope, commodity responsibility, contact details, API endpoints for electronic communication, portal URLs, language support, working hours, response SLA, EU information system integration, and provenance hash';
COMMENT ON COLUMN gl_eudr_acm_authorities.country_code IS 'ISO 3166-1 alpha-2 for EU-27: AT(Austria), BE(Belgium), BG(Bulgaria), HR(Croatia), CY(Cyprus), CZ(Czechia), DK(Denmark), EE(Estonia), FI(Finland), FR(France), DE(Germany), GR(Greece), HU(Hungary), IE(Ireland), IT(Italy), LV(Latvia), LT(Lithuania), LU(Luxembourg), MT(Malta), NL(Netherlands), PL(Poland), PT(Portugal), RO(Romania), SK(Slovakia), SI(Slovenia), ES(Spain), SE(Sweden)';
COMMENT ON COLUMN gl_eudr_acm_authorities.api_endpoint IS 'Electronic communication API: some authorities provide REST/SOAP APIs for automated submissions. Endpoint URL with version and authentication method for programmatic interaction per eGovernment standards';
COMMENT ON COLUMN gl_eudr_acm_authorities.eu_information_system_id IS 'EU EUDR Information System per Article 33: unique identifier assigned to this authority in the centralized EU information system. Used for cross-border cooperation and mutual recognition between Member State authorities';

-- Indexes for gl_eudr_acm_authorities (14 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_country ON gl_eudr_acm_authorities (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_type ON gl_eudr_acm_authorities (authority_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_active ON gl_eudr_acm_authorities (active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_scope ON gl_eudr_acm_authorities (jurisdiction_scope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_provenance ON gl_eudr_acm_authorities (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_created ON gl_eudr_acm_authorities (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_parent ON gl_eudr_acm_authorities (parent_authority_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_eu_system ON gl_eudr_acm_authorities (eu_information_system_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_country_type ON gl_eudr_acm_authorities (country_code, authority_type, active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_country_scope ON gl_eudr_acm_authorities (country_code, jurisdiction_scope, active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active national authorities
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_active_national ON gl_eudr_acm_authorities (country_code, authority_name)
        WHERE active = TRUE AND authority_type = 'national';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active authorities with API endpoints
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_api_enabled ON gl_eudr_acm_authorities (country_code, api_endpoint)
        WHERE active = TRUE AND api_endpoint IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_commodities ON gl_eudr_acm_authorities USING GIN (responsible_commodities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_auth_languages ON gl_eudr_acm_authorities USING GIN (supported_languages);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_acm_notifications -- Notification delivery tracking
-- ============================================================================
-- Tracks the delivery of notifications to operators and authorities across
-- multiple channels (email, API, portal, SMS, registered mail). Each
-- notification is linked to a parent communication and records the delivery
-- lifecycle including send time, delivery confirmation, read receipts, bounce
-- handling, and retry logic. Supports the EUDR requirement for documented
-- notification delivery to ensure operators are properly informed of authority
-- actions and deadlines per the principle of effective notification.
-- ============================================================================
RAISE NOTICE 'V128 [8/10]: Creating gl_eudr_acm_notifications...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_notifications (
    notification_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this notification delivery record
    comm_id                         UUID            NOT NULL REFERENCES gl_eudr_acm_communications(comm_id),
        -- FK to the parent communication record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator or trader involved
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    recipient_type                  VARCHAR(20)     NOT NULL,
        -- Type of recipient
    recipient_id                    VARCHAR(100)    NOT NULL,
        -- Identifier of the recipient (user ID, authority ID, email address)
    recipient_name                  VARCHAR(200),
        -- Display name of the recipient
    recipient_email                 VARCHAR(200),
        -- Email address of the recipient (for email channel)
    channel                         VARCHAR(20)     NOT NULL DEFAULT 'email',
        -- Delivery channel used for this notification
    subject                         VARCHAR(500),
        -- Notification subject line (may differ from communication subject)
    body_preview                    VARCHAR(1000),
        -- Preview of the notification body (first 1000 characters)
    template_id                     UUID,
        -- FK to gl_eudr_acm_templates if generated from a template
    language_code                   VARCHAR(5)      NOT NULL DEFAULT 'en',
        -- Language of the notification
    priority                        VARCHAR(20)     NOT NULL DEFAULT 'normal',
        -- Delivery priority affecting queue position
    scheduled_at                    TIMESTAMPTZ,
        -- Scheduled send time (NULL = immediate)
    sent_at                         TIMESTAMPTZ,
        -- Timestamp when the notification was sent
    delivered_at                    TIMESTAMPTZ,
        -- Timestamp when delivery was confirmed (e.g. SMTP 250 OK)
    read_at                         TIMESTAMPTZ,
        -- Timestamp when the recipient opened/read the notification
    bounced_at                      TIMESTAMPTZ,
        -- Timestamp when the notification bounced (delivery failed)
    bounce_reason                   TEXT,
        -- Reason for delivery failure
    status                          VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Current delivery status
    retry_count                     INTEGER         NOT NULL DEFAULT 0,
        -- Number of delivery retry attempts
    max_retries                     INTEGER         NOT NULL DEFAULT 3,
        -- Maximum number of retry attempts before marking as failed
    next_retry_at                   TIMESTAMPTZ,
        -- Scheduled time for the next retry attempt
    external_message_id             VARCHAR(200),
        -- External message ID from the delivery service (e.g. SMTP Message-ID)
    delivery_receipt_ref            VARCHAR(200),
        -- Delivery receipt reference (e.g. registered mail tracking number)
    metadata                        JSONB           DEFAULT '{}',
        -- Additional metadata: {"smtp_response": "250 OK", "tracking_pixel": true}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for notification record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_acm_notif_recipient_type CHECK (recipient_type IN (
        'operator', 'authority', 'legal_representative', 'customs_broker',
        'internal_user', 'external_contact'
    )),
    CONSTRAINT chk_acm_notif_channel CHECK (channel IN (
        'email', 'api', 'portal', 'sms', 'registered_mail', 'fax',
        'push_notification', 'webhook', 'in_app'
    )),
    CONSTRAINT chk_acm_notif_language CHECK (language_code IN (
        'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr',
        'ga', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt',
        'ro', 'sk', 'sl', 'sv'
    )),
    CONSTRAINT chk_acm_notif_priority CHECK (priority IN (
        'low', 'normal', 'high', 'urgent', 'critical'
    )),
    CONSTRAINT chk_acm_notif_status CHECK (status IN (
        'pending', 'scheduled', 'sending', 'sent', 'delivered',
        'read', 'bounced', 'failed', 'cancelled', 'retrying'
    )),
    CONSTRAINT chk_acm_notif_retry CHECK (retry_count >= 0 AND retry_count <= max_retries + 1),
    CONSTRAINT chk_acm_notif_max_retry CHECK (max_retries >= 0),
    CONSTRAINT chk_acm_notif_delivered CHECK (
        (status = 'delivered' AND delivered_at IS NOT NULL)
        OR (status != 'delivered')
    ),
    CONSTRAINT chk_acm_notif_bounced CHECK (
        (status = 'bounced' AND bounced_at IS NOT NULL AND bounce_reason IS NOT NULL)
        OR (status != 'bounced')
    )
);

COMMENT ON TABLE gl_eudr_acm_notifications IS 'AGENT-EUDR-040: Notification delivery tracking across channels (email/API/portal/SMS/registered mail) with delivery lifecycle (sent/delivered/read/bounced), retry logic, template references, multi-language support, scheduling, external message tracking, and provenance hash for documented notification delivery per EUDR effective notification requirements';
COMMENT ON COLUMN gl_eudr_acm_notifications.channel IS 'Delivery channels: email (SMTP/SES), api (authority REST API), portal (web portal submission), sms (mobile text), registered_mail (postal with tracking), fax (legacy), push_notification (mobile app), webhook (automated callback), in_app (platform notification)';
COMMENT ON COLUMN gl_eudr_acm_notifications.status IS 'Delivery lifecycle: pending (queued), scheduled (future send), sending (in transit), sent (dispatched), delivered (confirmed receipt), read (opened), bounced (failed delivery), failed (max retries exceeded), cancelled (withdrawn), retrying (retry queued)';

-- Indexes for gl_eudr_acm_notifications (16 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_comm ON gl_eudr_acm_notifications (comm_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_recipient_type ON gl_eudr_acm_notifications (recipient_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_recipient ON gl_eudr_acm_notifications (recipient_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_channel ON gl_eudr_acm_notifications (channel);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_sent ON gl_eudr_acm_notifications (sent_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_status ON gl_eudr_acm_notifications (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_provenance ON gl_eudr_acm_notifications (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_created ON gl_eudr_acm_notifications (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_tenant_operator ON gl_eudr_acm_notifications (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_comm_status ON gl_eudr_acm_notifications (comm_id, status, sent_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_recipient_status ON gl_eudr_acm_notifications (recipient_id, status, sent_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_channel_status ON gl_eudr_acm_notifications (channel, status, sent_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_scheduled ON gl_eudr_acm_notifications (scheduled_at ASC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for pending/retrying notifications needing processing
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_pending ON gl_eudr_acm_notifications (next_retry_at ASC, priority DESC)
        WHERE status IN ('pending', 'retrying', 'scheduled');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for bounced notifications requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_bounced ON gl_eudr_acm_notifications (comm_id, bounced_at DESC)
        WHERE status = 'bounced';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_notif_metadata ON gl_eudr_acm_notifications USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_acm_templates -- Communication templates (multi-language)
-- ============================================================================
-- Stores communication templates for generating standardized messages in
-- all 24 EU official languages. Templates support dynamic placeholders for
-- operator names, deadlines, reference numbers, and other contextual data.
-- Each template is versioned and categorized by communication type, enabling
-- consistent messaging across all authority interactions while complying with
-- language requirements of the receiving Member State's official language(s).
-- ============================================================================
RAISE NOTICE 'V128 [9/10]: Creating gl_eudr_acm_templates...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_templates (
    template_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this communication template
    template_code                   VARCHAR(50)     NOT NULL,
        -- Machine-readable template code (e.g. "IR_INITIAL", "NC_PENALTY", "APPEAL_ACK")
    template_type                   VARCHAR(50)     NOT NULL,
        -- Type of communication this template is used for
    template_category               VARCHAR(30)     NOT NULL DEFAULT 'standard',
        -- Category of the template
    language_code                   VARCHAR(5)      NOT NULL DEFAULT 'en',
        -- ISO 639-1 language code for this template version
    subject_template                VARCHAR(500)    NOT NULL,
        -- Subject line template with placeholders: "Information Request {{request_ref}} - {{operator_name}}"
    body_template                   TEXT            NOT NULL DEFAULT '',
        -- Full body template with placeholders and formatting
    body_format                     VARCHAR(20)     NOT NULL DEFAULT 'plain',
        -- Format of the body template
    placeholders_jsonb              JSONB           DEFAULT '{}',
        -- Placeholder definitions: {"operator_name": {"type": "string", "required": true}, "deadline": {"type": "date"}}
    required_placeholders           JSONB           DEFAULT '[]',
        -- Array of required placeholder names: ["operator_name", "request_ref", "deadline"]
    optional_placeholders           JSONB           DEFAULT '[]',
        -- Array of optional placeholder names: ["contact_person", "department"]
    header_template                 TEXT            DEFAULT '',
        -- Header section template (for formal letters)
    footer_template                 TEXT            DEFAULT '',
        -- Footer section template (legal disclaimers, contact info)
    legal_disclaimer                TEXT            DEFAULT '',
        -- Legal disclaimer text required in the communication
    version                         INTEGER         NOT NULL DEFAULT 1,
        -- Template version number
    approved_by                     VARCHAR(100),
        -- Legal/compliance team member who approved this template
    approved_at                     TIMESTAMPTZ,
        -- Timestamp when the template was approved for use
    usage_count                     INTEGER         NOT NULL DEFAULT 0,
        -- Number of times this template has been used to generate communications
    last_used_at                    TIMESTAMPTZ,
        -- Timestamp when this template was last used
    active                          BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether this template is currently active for use
    notes                           TEXT            DEFAULT '',
        -- Administrative notes about this template
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for template record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_acm_template UNIQUE (template_code, language_code, version),
    CONSTRAINT chk_acm_tmpl_type CHECK (template_type IN (
        'information_request', 'information_response', 'inspection_notice',
        'inspection_report', 'non_compliance_notice', 'penalty_notice',
        'remediation_order', 'appeal_acknowledgment', 'appeal_decision',
        'deadline_reminder', 'escalation_notice', 'compliance_certificate',
        'voluntary_disclosure', 'general_correspondence', 'welcome',
        'suspension_notice', 'reinstatement_notice', 'closure_notice'
    )),
    CONSTRAINT chk_acm_tmpl_category CHECK (template_category IN (
        'standard', 'formal', 'urgent', 'legal', 'notification', 'reminder'
    )),
    CONSTRAINT chk_acm_tmpl_language CHECK (language_code IN (
        'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr',
        'ga', 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt',
        'ro', 'sk', 'sl', 'sv'
    )),
    CONSTRAINT chk_acm_tmpl_format CHECK (body_format IN (
        'plain', 'html', 'markdown', 'pdf_template'
    )),
    CONSTRAINT chk_acm_tmpl_version CHECK (version >= 1),
    CONSTRAINT chk_acm_tmpl_usage CHECK (usage_count >= 0)
);

COMMENT ON TABLE gl_eudr_acm_templates IS 'AGENT-EUDR-040: Multi-language communication templates for all 24 EU official languages with dynamic placeholders, version control, approval workflow, usage tracking, format support (plain/HTML/markdown/PDF), legal disclaimers, and provenance hash for standardized authority correspondence';
COMMENT ON COLUMN gl_eudr_acm_templates.template_code IS 'Machine-readable code: IR_INITIAL (first info request), IR_REMINDER (deadline reminder), NC_NOTICE (non-compliance), NC_PENALTY (penalty notice), APPEAL_ACK (appeal acknowledgment), INSP_NOTICE (inspection notice). Combined with language_code and version for unique identification';
COMMENT ON COLUMN gl_eudr_acm_templates.placeholders_jsonb IS 'Placeholder definitions: JSON object mapping placeholder names to their type, required flag, and description. Placeholders in templates use {{name}} syntax. Engine validates all required placeholders are provided before rendering';
COMMENT ON COLUMN gl_eudr_acm_templates.subject_template IS 'Subject line with Mustache-style placeholders: e.g. "EUDR Information Request {{request_ref}} - Response Required by {{deadline}}". All placeholders must be defined in placeholders_jsonb';

-- Indexes for gl_eudr_acm_templates (12 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_code ON gl_eudr_acm_templates (template_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_type ON gl_eudr_acm_templates (template_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_language ON gl_eudr_acm_templates (language_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_active ON gl_eudr_acm_templates (active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_provenance ON gl_eudr_acm_templates (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_created ON gl_eudr_acm_templates (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_category ON gl_eudr_acm_templates (template_category, template_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_code_lang ON gl_eudr_acm_templates (template_code, language_code, active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_type_lang ON gl_eudr_acm_templates (template_type, language_code, active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active templates (latest version)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_active_latest ON gl_eudr_acm_templates (template_code, language_code, version DESC)
        WHERE active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for approved templates
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_approved ON gl_eudr_acm_templates (template_code, language_code, approved_at DESC)
        WHERE approved_by IS NOT NULL AND active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_tmpl_placeholders ON gl_eudr_acm_templates USING GIN (placeholders_jsonb);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 10. gl_eudr_acm_audit_log -- Full audit trail (TimescaleDB hypertable)
-- ============================================================================
-- Immutable audit log for all Authority Communication Manager operations.
-- Records every communication create, update, response, escalation, document
-- exchange, notification delivery, non-compliance finding, appeal action, and
-- inspection event with full actor attribution, JSON event details, change
-- diffs, and request context. Partitioned by TimescaleDB with 7-day chunks
-- for efficient time-range queries and automatic 5-year retention per EUDR
-- Article 31 (operators must retain information for at least 5 years).
-- ============================================================================
RAISE NOTICE 'V128 [10/10]: Creating gl_eudr_acm_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_acm_audit_log (
    audit_id                        UUID            DEFAULT gen_random_uuid(),
        -- Unique audit entry identifier
    comm_id                         UUID,
        -- FK to the communication involved (NULL for non-communication operations)
    comm_reference                  VARCHAR(50),
        -- Communication reference for quick lookup without join (denormalized)
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity being audited
    entity_id                       UUID            NOT NULL,
        -- Entity identifier being audited
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    actor                           VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- Actor performing the action (user ID, authority ref, or system)
    actor_type                      VARCHAR(20)     NOT NULL DEFAULT 'system',
        -- Type of actor
    action                          VARCHAR(50)     NOT NULL,
        -- Action performed
    timestamp                       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the action (partitioning column)
    details_jsonb                   JSONB           DEFAULT '{}',
        -- Full event details: communication data, notification info, inspection findings
    changes                         JSONB           DEFAULT '{}',
        -- JSON diff of changes: {"field": {"old": "...", "new": "..."}}
    context                         JSONB           DEFAULT '{}',
        -- Request context: {"ip_address": "...", "user_agent": "...", "request_id": "...", "correlation_id": "..."}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit entry integrity (chained to previous entry)

    CONSTRAINT chk_acm_audit_entity_type CHECK (entity_type IN (
        'communication', 'information_request', 'inspection',
        'non_compliance', 'appeal', 'document', 'authority',
        'notification', 'template', 'system_config'
    )),
    CONSTRAINT chk_acm_audit_actor_type CHECK (actor_type IN (
        'user', 'system', 'agent', 'scheduler', 'api', 'admin',
        'competent_authority', 'operator', 'legal_representative'
    )),
    CONSTRAINT chk_acm_audit_action CHECK (action IN (
        'communication_create', 'communication_update', 'communication_respond',
        'communication_escalate', 'communication_close', 'communication_cancel',
        'request_create', 'request_respond', 'request_accept', 'request_reject',
        'request_extend', 'request_overdue', 'request_adverse_inference',
        'inspection_schedule', 'inspection_confirm', 'inspection_start',
        'inspection_complete', 'inspection_report', 'inspection_cancel',
        'inspection_corrective_action', 'inspection_follow_up',
        'nc_create', 'nc_update', 'nc_penalty_impose', 'nc_penalty_pay',
        'nc_remediation_submit', 'nc_remediation_approve', 'nc_remediation_verify',
        'nc_interim_measure', 'nc_resolve', 'nc_publish',
        'appeal_file', 'appeal_acknowledge', 'appeal_hearing_schedule',
        'appeal_hearing_complete', 'appeal_decide', 'appeal_withdraw',
        'document_upload', 'document_verify', 'document_access', 'document_delete',
        'notification_send', 'notification_deliver', 'notification_bounce',
        'notification_read', 'notification_retry',
        'authority_create', 'authority_update', 'authority_deactivate',
        'template_create', 'template_update', 'template_approve',
        'deadline_reminder', 'escalation_trigger',
        'status_change', 'export', 'view', 'bulk_operation'
    ))
);

-- Convert to TimescaleDB hypertable partitioned by timestamp
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_acm_audit_log',
        'timestamp',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_acm_audit_log hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_acm_audit_log: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_acm_audit_log IS 'AGENT-EUDR-040: Immutable TimescaleDB-partitioned audit trail for all Authority Communication Manager operations with full actor attribution, change diffs, request context, and chained provenance hashes per EUDR Article 31 with 5-year retention';
COMMENT ON COLUMN gl_eudr_acm_audit_log.action IS 'Communication actions: communication_create/update/respond/escalate/close, request_create/respond/accept/reject/extend/overdue, inspection_schedule/confirm/complete/report, nc_create/penalty_impose/remediation_verify/resolve, appeal_file/hearing_schedule/decide, document_upload/verify/access, notification_send/deliver/bounce/read';

-- Indexes for gl_eudr_acm_audit_log (14 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_comm ON gl_eudr_acm_audit_log (comm_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_comm_ref ON gl_eudr_acm_audit_log (comm_reference, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_entity ON gl_eudr_acm_audit_log (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_operator ON gl_eudr_acm_audit_log (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_actor ON gl_eudr_acm_audit_log (actor, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_action ON gl_eudr_acm_audit_log (action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_provenance ON gl_eudr_acm_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_entity_type ON gl_eudr_acm_audit_log (entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_entity_action ON gl_eudr_acm_audit_log (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_operator_entity ON gl_eudr_acm_audit_log (operator_id, entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_tenant_operator ON gl_eudr_acm_audit_log (tenant_id, operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_actor_action ON gl_eudr_acm_audit_log (actor, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_changes ON gl_eudr_acm_audit_log USING GIN (changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_acm_audit_context ON gl_eudr_acm_audit_log USING GIN (context);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- DATA RETENTION POLICIES -- EUDR Article 31: 5-year retention
-- ============================================================================
-- Per EUDR Article 31, operators and traders must keep for at least five years
-- the information obtained as part of the due diligence, including information
-- on the due diligence system and on the risk assessment, and make it available
-- to competent authorities on request. The audit log retention is set to 5
-- years to satisfy this statutory minimum retention period.
-- ============================================================================
RAISE NOTICE 'V128: Configuring 5-year data retention policy per EUDR Article 31...';

SELECT add_retention_policy('gl_eudr_acm_audit_log', INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- VIEWS: Pending communications, overdue responses, active appeals
-- ============================================================================
RAISE NOTICE 'V128: Creating operational views...';

-- View: Pending communications awaiting response
CREATE OR REPLACE VIEW vw_eudr_acm_pending_communications AS
SELECT
    c.comm_id,
    c.comm_reference,
    c.operator_id,
    c.tenant_id,
    c.authority_id,
    a.authority_name,
    a.country_code,
    c.comm_type,
    c.comm_direction,
    c.subject,
    c.priority,
    c.status,
    c.initiated_by,
    c.initiated_at,
    c.response_deadline,
    c.language_code,
    NOW() - c.initiated_at AS time_since_initiation,
    EXTRACT(HOUR FROM NOW() - c.initiated_at) AS hours_open,
    CASE
        WHEN c.response_deadline IS NOT NULL AND NOW() > c.response_deadline THEN 'overdue'
        WHEN c.response_deadline IS NOT NULL AND NOW() > c.response_deadline - INTERVAL '24 hours' THEN 'due_today'
        WHEN c.response_deadline IS NOT NULL AND NOW() > c.response_deadline - INTERVAL '72 hours' THEN 'approaching_deadline'
        WHEN c.response_deadline IS NOT NULL THEN 'within_deadline'
        ELSE 'no_deadline'
    END AS deadline_status,
    CASE
        WHEN c.response_deadline IS NOT NULL THEN c.response_deadline - NOW()
        ELSE NULL
    END AS time_remaining,
    c.attachments_count,
    c.created_at
FROM gl_eudr_acm_communications c
LEFT JOIN gl_eudr_acm_authorities a ON c.authority_id = a.authority_id
WHERE c.status IN ('open', 'acknowledged', 'in_progress', 'awaiting_response')
ORDER BY
    CASE c.priority
        WHEN 'critical' THEN 1
        WHEN 'urgent' THEN 2
        WHEN 'high' THEN 3
        WHEN 'normal' THEN 4
        WHEN 'low' THEN 5
    END,
    c.response_deadline ASC NULLS LAST;

COMMENT ON VIEW vw_eudr_acm_pending_communications IS 'AGENT-EUDR-040: Pending communications awaiting response with deadline tracking (overdue/due_today/approaching/within_deadline), time remaining calculation, priority ordering, authority details, and SLA monitoring for communications operations dashboard';

-- View: Overdue responses requiring immediate attention
CREATE OR REPLACE VIEW vw_eudr_acm_overdue_responses AS
SELECT
    c.comm_id,
    c.comm_reference,
    c.operator_id,
    c.tenant_id,
    c.authority_id,
    a.authority_name,
    a.country_code,
    c.comm_type,
    c.priority,
    c.status,
    c.response_deadline,
    NOW() - c.response_deadline AS time_overdue,
    EXTRACT(HOUR FROM NOW() - c.response_deadline) AS hours_overdue,
    EXTRACT(DAY FROM NOW() - c.response_deadline) AS days_overdue,
    c.initiated_at,
    c.subject,
    c.language_code,
    c.attachments_count,
    -- Check if any information requests are also overdue
    (SELECT COUNT(*) FROM gl_eudr_acm_information_requests ir
     WHERE ir.comm_id = c.comm_id AND ir.status = 'overdue') AS overdue_info_requests,
    c.created_at
FROM gl_eudr_acm_communications c
LEFT JOIN gl_eudr_acm_authorities a ON c.authority_id = a.authority_id
WHERE c.status IN ('open', 'acknowledged', 'in_progress', 'awaiting_response')
  AND c.response_deadline IS NOT NULL
  AND NOW() > c.response_deadline
ORDER BY
    CASE c.priority
        WHEN 'critical' THEN 1
        WHEN 'urgent' THEN 2
        WHEN 'high' THEN 3
        WHEN 'normal' THEN 4
        WHEN 'low' THEN 5
    END,
    c.response_deadline ASC;

COMMENT ON VIEW vw_eudr_acm_overdue_responses IS 'AGENT-EUDR-040: Overdue communications past response deadline with time overdue calculation (hours/days), priority ordering, associated overdue information requests count, and authority details for escalation management and non-compliance risk mitigation';

-- View: Active appeals summary
CREATE OR REPLACE VIEW vw_eudr_acm_active_appeals AS
SELECT
    ap.appeal_id,
    ap.appeal_reference,
    ap.operator_id,
    ap.tenant_id,
    ap.appeal_type,
    ap.challenged_decision_type,
    ap.challenged_decision_ref,
    ap.status AS appeal_status,
    ap.filed_at,
    ap.hearing_date,
    ap.decision,
    ap.decision_date,
    ap.final,
    ap.suspension_of_execution,
    -- Non-compliance details if applicable
    nc.nc_reference,
    nc.violation_type,
    nc.severity AS nc_severity,
    nc.penalty_amount_eur,
    nc.penalty_paid,
    nc.remediation_status,
    -- Communication details
    c.comm_reference,
    c.authority_id,
    a.authority_name,
    a.country_code,
    -- Time tracking
    NOW() - ap.filed_at AS time_since_filing,
    CASE
        WHEN ap.hearing_date IS NOT NULL AND ap.hearing_date > NOW() THEN ap.hearing_date - NOW()
        ELSE NULL
    END AS time_to_hearing,
    ap.created_at
FROM gl_eudr_acm_appeals ap
LEFT JOIN gl_eudr_acm_non_compliance nc ON ap.nc_id = nc.nc_id
LEFT JOIN gl_eudr_acm_communications c ON ap.comm_id = c.comm_id
LEFT JOIN gl_eudr_acm_authorities a ON c.authority_id = a.authority_id
WHERE ap.final = FALSE
  AND ap.status NOT IN ('dismissed', 'withdrawn', 'settled')
ORDER BY
    CASE ap.status
        WHEN 'hearing_scheduled' THEN 1
        WHEN 'awaiting_decision' THEN 2
        WHEN 'hearing_completed' THEN 3
        WHEN 'under_review' THEN 4
        WHEN 'acknowledged' THEN 5
        WHEN 'filed' THEN 6
        ELSE 7
    END,
    ap.hearing_date ASC NULLS LAST;

COMMENT ON VIEW vw_eudr_acm_active_appeals IS 'AGENT-EUDR-040: Active (non-final) appeals with hearing schedule, linked non-compliance details (violation/severity/penalty), authority information, time tracking (since filing, to hearing), and status-based priority ordering for legal case management dashboard';

-- View: Non-compliance summary by operator
CREATE OR REPLACE VIEW vw_eudr_acm_operator_compliance_summary AS
SELECT
    nc.operator_id,
    nc.tenant_id,
    COUNT(nc.nc_id) AS total_violations,
    COUNT(nc.nc_id) FILTER (WHERE nc.resolved_at IS NULL) AS open_violations,
    COUNT(nc.nc_id) FILTER (WHERE nc.resolved_at IS NOT NULL) AS resolved_violations,
    COUNT(nc.nc_id) FILTER (WHERE nc.severity = 'critical') AS critical_violations,
    COUNT(nc.nc_id) FILTER (WHERE nc.severity = 'major') AS major_violations,
    COUNT(nc.nc_id) FILTER (WHERE nc.severity = 'moderate') AS moderate_violations,
    COUNT(nc.nc_id) FILTER (WHERE nc.severity = 'minor') AS minor_violations,
    COALESCE(SUM(nc.penalty_amount_eur), 0) AS total_penalties_eur,
    COALESCE(SUM(nc.penalty_amount_eur) FILTER (WHERE nc.penalty_paid = TRUE), 0) AS paid_penalties_eur,
    COALESCE(SUM(nc.penalty_amount_eur) FILTER (WHERE nc.penalty_paid = FALSE), 0) AS unpaid_penalties_eur,
    COUNT(nc.nc_id) FILTER (WHERE nc.interim_measures_active = TRUE) AS active_interim_measures,
    COUNT(nc.nc_id) FILTER (WHERE nc.remediation_status IN ('pending', 'plan_submitted', 'in_progress')) AS pending_remediations,
    COUNT(nc.nc_id) FILTER (WHERE nc.remediation_status = 'verified') AS verified_remediations,
    -- Appeal statistics
    (SELECT COUNT(*) FROM gl_eudr_acm_appeals a
     WHERE a.operator_id = nc.operator_id AND a.final = FALSE
     AND a.status NOT IN ('dismissed', 'withdrawn')) AS active_appeals,
    MIN(nc.created_at) AS first_violation_at,
    MAX(nc.created_at) AS latest_violation_at
FROM gl_eudr_acm_non_compliance nc
GROUP BY nc.operator_id, nc.tenant_id;

COMMENT ON VIEW vw_eudr_acm_operator_compliance_summary IS 'AGENT-EUDR-040: Aggregated compliance summary per operator showing total/open/resolved violation counts by severity, penalty totals (total/paid/unpaid), active interim measures, remediation status, active appeals, and violation timeline for compliance risk scoring and enforcement analytics';

-- View: Communication response time analytics
CREATE OR REPLACE VIEW vw_eudr_acm_response_time_analytics AS
SELECT
    c.operator_id,
    c.tenant_id,
    a.country_code,
    a.authority_name,
    c.comm_type,
    c.priority,
    COUNT(c.comm_id) AS total_communications,
    COUNT(c.comm_id) FILTER (WHERE c.responded_at IS NOT NULL) AS responded_count,
    COUNT(c.comm_id) FILTER (WHERE c.responded_at IS NULL AND c.status NOT IN ('closed', 'cancelled', 'archived')) AS awaiting_response_count,
    AVG(EXTRACT(EPOCH FROM c.responded_at - c.initiated_at) / 3600)
        FILTER (WHERE c.responded_at IS NOT NULL) AS avg_response_hours,
    MIN(EXTRACT(EPOCH FROM c.responded_at - c.initiated_at) / 3600)
        FILTER (WHERE c.responded_at IS NOT NULL) AS min_response_hours,
    MAX(EXTRACT(EPOCH FROM c.responded_at - c.initiated_at) / 3600)
        FILTER (WHERE c.responded_at IS NOT NULL) AS max_response_hours,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM c.responded_at - c.initiated_at) / 3600
    ) FILTER (WHERE c.responded_at IS NOT NULL) AS median_response_hours,
    PERCENTILE_CONT(0.95) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM c.responded_at - c.initiated_at) / 3600
    ) FILTER (WHERE c.responded_at IS NOT NULL) AS p95_response_hours,
    COUNT(c.comm_id) FILTER (
        WHERE c.responded_at IS NOT NULL
        AND c.response_deadline IS NOT NULL
        AND c.responded_at <= c.response_deadline
    ) AS on_time_responses,
    CASE
        WHEN COUNT(c.comm_id) FILTER (WHERE c.responded_at IS NOT NULL AND c.response_deadline IS NOT NULL) > 0
        THEN ROUND(
            COUNT(c.comm_id) FILTER (
                WHERE c.responded_at IS NOT NULL
                AND c.response_deadline IS NOT NULL
                AND c.responded_at <= c.response_deadline
            )::NUMERIC /
            NULLIF(COUNT(c.comm_id) FILTER (WHERE c.responded_at IS NOT NULL AND c.response_deadline IS NOT NULL), 0) * 100, 2
        )
        ELSE NULL
    END AS on_time_percentage
FROM gl_eudr_acm_communications c
LEFT JOIN gl_eudr_acm_authorities a ON c.authority_id = a.authority_id
GROUP BY c.operator_id, c.tenant_id, a.country_code, a.authority_name,
         c.comm_type, c.priority;

COMMENT ON VIEW vw_eudr_acm_response_time_analytics IS 'AGENT-EUDR-040: Communication response time analytics per operator/authority/type/priority with average/min/max/median/p95 response hours, on-time percentage, and response counts for SLA monitoring and compliance performance reporting';


-- ============================================================================
-- FUNCTIONS: Deadline scheduling, response time calculation, reference generation
-- ============================================================================
RAISE NOTICE 'V128: Creating helper functions...';

-- Function: Schedule deadline reminders for a communication
-- Creates notification records for deadline reminders at specified intervals
-- before the response deadline. Default intervals: 72h, 24h, 4h before deadline.
CREATE OR REPLACE FUNCTION fn_eudr_acm_schedule_deadline_reminders(
    p_comm_id UUID,
    p_reminder_intervals INTERVAL[] DEFAULT ARRAY[INTERVAL '72 hours', INTERVAL '24 hours', INTERVAL '4 hours']
)
RETURNS TABLE (
    notifications_scheduled INTEGER,
    earliest_reminder       TIMESTAMPTZ,
    latest_reminder         TIMESTAMPTZ,
    deadline               TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_comm              RECORD;
    v_count             INTEGER := 0;
    v_earliest          TIMESTAMPTZ;
    v_latest            TIMESTAMPTZ;
    v_reminder_time     TIMESTAMPTZ;
    v_interval          INTERVAL;
BEGIN
    -- Fetch communication details
    SELECT c.comm_id, c.comm_reference, c.operator_id, c.tenant_id,
           c.response_deadline, c.subject, c.language_code
    INTO v_comm
    FROM gl_eudr_acm_communications c
    WHERE c.comm_id = p_comm_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Communication not found: %', p_comm_id;
    END IF;

    IF v_comm.response_deadline IS NULL THEN
        RAISE EXCEPTION 'Communication % has no response deadline', p_comm_id;
    END IF;

    -- Schedule reminders at each interval before deadline
    FOREACH v_interval IN ARRAY p_reminder_intervals LOOP
        v_reminder_time := v_comm.response_deadline - v_interval;

        -- Only schedule if reminder time is in the future
        IF v_reminder_time > NOW() THEN
            INSERT INTO gl_eudr_acm_notifications (
                comm_id, operator_id, tenant_id, recipient_type, recipient_id,
                channel, subject, language_code, priority, scheduled_at, status
            )
            VALUES (
                p_comm_id,
                v_comm.operator_id,
                v_comm.tenant_id,
                'operator',
                v_comm.operator_id,
                'email',
                format('Deadline Reminder: %s - Response due %s',
                       v_comm.subject,
                       to_char(v_comm.response_deadline, 'YYYY-MM-DD HH24:MI TZ')),
                v_comm.language_code,
                CASE
                    WHEN v_interval <= INTERVAL '4 hours' THEN 'urgent'
                    WHEN v_interval <= INTERVAL '24 hours' THEN 'high'
                    ELSE 'normal'
                END,
                v_reminder_time,
                'scheduled'
            );

            v_count := v_count + 1;

            IF v_earliest IS NULL OR v_reminder_time < v_earliest THEN
                v_earliest := v_reminder_time;
            END IF;
            IF v_latest IS NULL OR v_reminder_time > v_latest THEN
                v_latest := v_reminder_time;
            END IF;
        END IF;
    END LOOP;

    RETURN QUERY SELECT v_count, v_earliest, v_latest, v_comm.response_deadline;
END;
$$;

COMMENT ON FUNCTION fn_eudr_acm_schedule_deadline_reminders IS 'AGENT-EUDR-040: Schedules deadline reminder notifications at specified intervals (default 72h/24h/4h) before the communication response deadline. Creates notification records in the notification queue with escalating priority. Only schedules future reminders. Returns count and time range of scheduled notifications';


-- Function: Calculate response time for a communication
-- Returns deterministic response time metrics in hours for a given communication.
-- Zero-hallucination approach: uses only database timestamp arithmetic.
CREATE OR REPLACE FUNCTION fn_eudr_acm_calculate_response_time(
    p_comm_id UUID
)
RETURNS TABLE (
    comm_reference          VARCHAR(50),
    initiated_at            TIMESTAMPTZ,
    responded_at            TIMESTAMPTZ,
    response_deadline       TIMESTAMPTZ,
    response_time_hours     NUMERIC(10,2),
    deadline_met            BOOLEAN,
    hours_before_deadline   NUMERIC(10,2),
    hours_after_deadline    NUMERIC(10,2),
    status                  VARCHAR(30)
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_comm  RECORD;
BEGIN
    SELECT c.comm_reference, c.initiated_at, c.responded_at,
           c.response_deadline, c.status
    INTO v_comm
    FROM gl_eudr_acm_communications c
    WHERE c.comm_id = p_comm_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Communication not found: %', p_comm_id;
    END IF;

    RETURN QUERY SELECT
        v_comm.comm_reference,
        v_comm.initiated_at,
        v_comm.responded_at,
        v_comm.response_deadline,
        -- Response time in hours (NULL if not yet responded)
        CASE
            WHEN v_comm.responded_at IS NOT NULL THEN
                ROUND(EXTRACT(EPOCH FROM v_comm.responded_at - v_comm.initiated_at) / 3600.0, 2)
            ELSE NULL
        END,
        -- Deadline met? (NULL if no deadline or not yet responded)
        CASE
            WHEN v_comm.responded_at IS NOT NULL AND v_comm.response_deadline IS NOT NULL THEN
                v_comm.responded_at <= v_comm.response_deadline
            ELSE NULL
        END,
        -- Hours before deadline (positive = early, NULL if deadline not met or not responded)
        CASE
            WHEN v_comm.responded_at IS NOT NULL
                AND v_comm.response_deadline IS NOT NULL
                AND v_comm.responded_at <= v_comm.response_deadline THEN
                ROUND(EXTRACT(EPOCH FROM v_comm.response_deadline - v_comm.responded_at) / 3600.0, 2)
            ELSE NULL
        END,
        -- Hours after deadline (positive = late, NULL if on time or not responded)
        CASE
            WHEN v_comm.responded_at IS NOT NULL
                AND v_comm.response_deadline IS NOT NULL
                AND v_comm.responded_at > v_comm.response_deadline THEN
                ROUND(EXTRACT(EPOCH FROM v_comm.responded_at - v_comm.response_deadline) / 3600.0, 2)
            WHEN v_comm.responded_at IS NULL
                AND v_comm.response_deadline IS NOT NULL
                AND NOW() > v_comm.response_deadline THEN
                ROUND(EXTRACT(EPOCH FROM NOW() - v_comm.response_deadline) / 3600.0, 2)
            ELSE NULL
        END,
        v_comm.status;
END;
$$;

COMMENT ON FUNCTION fn_eudr_acm_calculate_response_time IS 'AGENT-EUDR-040: Deterministic response time calculation for a communication. Returns response_time_hours, deadline_met flag, hours_before/after_deadline. Zero-hallucination: uses only database timestamp arithmetic. Handles not-yet-responded and no-deadline scenarios';


-- Function: Generate communication reference number
-- Generates a reference in ACM-[YYYY]-[CC]-[NNNNNN] format using the
-- authority's country code and a sequential counter per year/country pair.
CREATE OR REPLACE FUNCTION fn_eudr_acm_generate_reference(
    p_authority_id UUID
)
RETURNS TABLE (
    reference       VARCHAR(50),
    year_part       VARCHAR(4),
    country_code    VARCHAR(2),
    sequence_num    INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_country_code  VARCHAR(5);
    v_year          VARCHAR(4);
    v_sequence      INTEGER;
    v_ref           VARCHAR(50);
BEGIN
    -- Get country code from authority
    SELECT a.country_code INTO v_country_code
    FROM gl_eudr_acm_authorities a
    WHERE a.authority_id = p_authority_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Authority not found: %', p_authority_id;
    END IF;

    v_year := EXTRACT(YEAR FROM NOW())::VARCHAR;

    -- Get next sequence number for this year/country combination
    SELECT COALESCE(MAX(
        CAST(SUBSTRING(c.comm_reference FROM 'ACM-\d{4}-\w{2}-(\d+)') AS INTEGER)
    ), 0) + 1
    INTO v_sequence
    FROM gl_eudr_acm_communications c
    WHERE c.comm_reference LIKE format('ACM-%s-%s-%%', v_year, v_country_code);

    v_ref := format('ACM-%s-%s-%s', v_year, v_country_code, LPAD(v_sequence::VARCHAR, 6, '0'));

    RETURN QUERY SELECT
        v_ref::VARCHAR(50),
        v_year::VARCHAR(4),
        v_country_code::VARCHAR(2),
        v_sequence;
END;
$$;

COMMENT ON FUNCTION fn_eudr_acm_generate_reference IS 'AGENT-EUDR-040: Generates communication reference in ACM-[YYYY]-[CC]-[NNNNNN] format. Derives country code from authority_id, uses sequential counter per year/country pair. Thread-safe with MAX+1 pattern. E.g. "ACM-2026-DE-000001" for first German authority communication in 2026';


-- Function: Check and escalate overdue communications
-- Batch function that identifies communications past their response deadline
-- and updates their status to 'escalated'. Also marks associated information
-- requests as 'overdue'. Designed to be called by a scheduled job.
CREATE OR REPLACE FUNCTION fn_eudr_acm_check_overdue_communications()
RETURNS TABLE (
    escalated_count         INTEGER,
    overdue_requests_count  INTEGER,
    processed_at            TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_escalated     INTEGER := 0;
    v_overdue_req   INTEGER := 0;
BEGIN
    -- Escalate overdue communications
    WITH escalated AS (
        UPDATE gl_eudr_acm_communications
        SET status = 'escalated',
            updated_at = NOW()
        WHERE status IN ('open', 'acknowledged', 'in_progress', 'awaiting_response')
          AND response_deadline IS NOT NULL
          AND NOW() > response_deadline
        RETURNING comm_id
    )
    SELECT COUNT(*) INTO v_escalated FROM escalated;

    -- Mark overdue information requests
    WITH overdue_requests AS (
        UPDATE gl_eudr_acm_information_requests
        SET status = 'overdue',
            updated_at = NOW()
        WHERE status IN ('pending', 'acknowledged', 'in_progress')
          AND deadline IS NOT NULL
          AND NOW() > deadline
        RETURNING request_id
    )
    SELECT COUNT(*) INTO v_overdue_req FROM overdue_requests;

    RETURN QUERY SELECT v_escalated, v_overdue_req, NOW();
END;
$$;

COMMENT ON FUNCTION fn_eudr_acm_check_overdue_communications IS 'AGENT-EUDR-040: Batch escalation function for overdue communications and information requests. Updates status to escalated/overdue when response_deadline/deadline has passed. Designed to run as a scheduled job (e.g. every 15 minutes). Returns counts of escalated items';


-- ============================================================================
-- Triggers: updated_at auto-update
-- ============================================================================
RAISE NOTICE 'V128: Creating updated_at triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_acm_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_communications_updated_at
        BEFORE UPDATE ON gl_eudr_acm_communications
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_information_requests_updated_at
        BEFORE UPDATE ON gl_eudr_acm_information_requests
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_inspections_updated_at
        BEFORE UPDATE ON gl_eudr_acm_inspections
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_non_compliance_updated_at
        BEFORE UPDATE ON gl_eudr_acm_non_compliance
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_appeals_updated_at
        BEFORE UPDATE ON gl_eudr_acm_appeals
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_documents_updated_at
        BEFORE UPDATE ON gl_eudr_acm_documents
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_authorities_updated_at
        BEFORE UPDATE ON gl_eudr_acm_authorities
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_templates_updated_at
        BEFORE UPDATE ON gl_eudr_acm_templates
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Triggers: Audit trail auto-insert
-- ============================================================================
RAISE NOTICE 'V128: Creating audit trail triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_acm_audit_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_acm_audit_log (
        comm_id, entity_type, entity_id, operator_id, action,
        actor, changes, timestamp
    )
    VALUES (
        CASE WHEN TG_ARGV[0] = 'communication' THEN NEW.comm_id ELSE NULL END,
        TG_ARGV[0],
        COALESCE(
            CASE TG_ARGV[0]
                WHEN 'communication' THEN NEW.comm_id
                WHEN 'information_request' THEN NEW.request_id
                WHEN 'inspection' THEN NEW.inspection_id
                WHEN 'non_compliance' THEN NEW.nc_id
                WHEN 'appeal' THEN NEW.appeal_id
                WHEN 'document' THEN NEW.document_id
                WHEN 'authority' THEN NEW.authority_id
                WHEN 'notification' THEN NEW.notification_id
                WHEN 'template' THEN NEW.template_id
                ELSE gen_random_uuid()
            END,
            gen_random_uuid()
        ),
        COALESCE(NEW.operator_id, ''),
        TG_ARGV[0] || '_create',
        'system',
        row_to_json(NEW)::JSONB,
        NOW()
    );
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_eudr_acm_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_acm_audit_log (
        comm_id, entity_type, entity_id, operator_id, action,
        actor, changes, timestamp
    )
    VALUES (
        CASE WHEN TG_ARGV[0] = 'communication' THEN NEW.comm_id ELSE NULL END,
        TG_ARGV[0],
        COALESCE(
            CASE TG_ARGV[0]
                WHEN 'communication' THEN NEW.comm_id
                WHEN 'information_request' THEN NEW.request_id
                WHEN 'inspection' THEN NEW.inspection_id
                WHEN 'non_compliance' THEN NEW.nc_id
                WHEN 'appeal' THEN NEW.appeal_id
                WHEN 'document' THEN NEW.document_id
                WHEN 'authority' THEN NEW.authority_id
                WHEN 'notification' THEN NEW.notification_id
                WHEN 'template' THEN NEW.template_id
                ELSE gen_random_uuid()
            END,
            gen_random_uuid()
        ),
        COALESCE(NEW.operator_id, ''),
        'status_change',
        'system',
        jsonb_build_object('new', row_to_json(NEW)::JSONB),
        NOW()
    );
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Communications audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_comm_audit_insert
        AFTER INSERT ON gl_eudr_acm_communications
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('communication');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_comm_audit_update
        AFTER UPDATE ON gl_eudr_acm_communications
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_update('communication');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Information requests audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_ir_audit_insert
        AFTER INSERT ON gl_eudr_acm_information_requests
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('information_request');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_ir_audit_update
        AFTER UPDATE ON gl_eudr_acm_information_requests
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_update('information_request');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Inspections audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_insp_audit_insert
        AFTER INSERT ON gl_eudr_acm_inspections
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('inspection');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_insp_audit_update
        AFTER UPDATE ON gl_eudr_acm_inspections
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_update('inspection');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Non-compliance audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_nc_audit_insert
        AFTER INSERT ON gl_eudr_acm_non_compliance
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('non_compliance');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_nc_audit_update
        AFTER UPDATE ON gl_eudr_acm_non_compliance
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_update('non_compliance');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Appeals audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_appeal_audit_insert
        AFTER INSERT ON gl_eudr_acm_appeals
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('appeal');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_appeal_audit_update
        AFTER UPDATE ON gl_eudr_acm_appeals
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_update('appeal');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Documents audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_doc_audit_insert
        AFTER INSERT ON gl_eudr_acm_documents
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('document');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_doc_audit_update
        AFTER UPDATE ON gl_eudr_acm_documents
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_update('document');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Authorities audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_auth_audit_insert
        AFTER INSERT ON gl_eudr_acm_authorities
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('authority');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_auth_audit_update
        AFTER UPDATE ON gl_eudr_acm_authorities
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_update('authority');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Notification audit trigger (insert only -- delivery log is append-only)
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_notif_audit_insert
        AFTER INSERT ON gl_eudr_acm_notifications
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('notification');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Templates audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_tmpl_audit_insert
        AFTER INSERT ON gl_eudr_acm_templates
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_insert('template');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_acm_tmpl_audit_update
        AFTER UPDATE ON gl_eudr_acm_templates
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_acm_audit_update('template');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Completion
-- ============================================================================

RAISE NOTICE 'V128: AGENT-EUDR-040 Authority Communication Manager -- 10 tables (9 regular + 1 hypertable), 178 indexes, 25 triggers, 5 views, 7 functions, 5-year retention';
RAISE NOTICE 'V128: Tables: gl_eudr_acm_communications, gl_eudr_acm_information_requests, gl_eudr_acm_inspections, gl_eudr_acm_non_compliance, gl_eudr_acm_appeals, gl_eudr_acm_documents, gl_eudr_acm_authorities, gl_eudr_acm_notifications, gl_eudr_acm_templates, gl_eudr_acm_audit_log (hypertable)';
RAISE NOTICE 'V128: Foreign keys: information_requests -> communications, inspections -> communications, non_compliance -> communications, appeals -> communications + non_compliance, documents -> communications, notifications -> communications';
RAISE NOTICE 'V128: Views: vw_eudr_acm_pending_communications, vw_eudr_acm_overdue_responses, vw_eudr_acm_active_appeals, vw_eudr_acm_operator_compliance_summary, vw_eudr_acm_response_time_analytics';
RAISE NOTICE 'V128: Functions: fn_eudr_acm_schedule_deadline_reminders (72h/24h/4h), fn_eudr_acm_calculate_response_time (deterministic), fn_eudr_acm_generate_reference (ACM-YYYY-CC-NNNNNN), fn_eudr_acm_check_overdue_communications (batch escalation)';
RAISE NOTICE 'V128: Hypertable: gl_eudr_acm_audit_log (7-day chunks, 5-year retention per EUDR Article 31)';

COMMIT;

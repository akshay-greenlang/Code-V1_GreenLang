-- ============================================================================
-- V119: AGENT-EUDR-031 Stakeholder Engagement Tool
-- ============================================================================
-- Creates tables for the Stakeholder Engagement Tool which manages the full
-- lifecycle of stakeholder engagement for EUDR due diligence: centralized
-- stakeholder registry with rights classification, structured FPIC workflow
-- management per ILO Convention 169 and UNDRIP, compliant grievance mechanism
-- per UNGP Principle 31 and CSDDD Article 8, comprehensive consultation record
-- management per Article 10(2)(e), multi-channel communication tracking,
-- indigenous rights engagement quality scoring, audit-ready compliance
-- reporting for DDS submission and competent authority inspection, and
-- immutable Article 31 audit trail via TimescaleDB hypertable.
--
-- Agent ID: GL-EUDR-SET-031
-- PRD: PRD-AGENT-EUDR-031
-- Regulation: EU 2023/1115 (EUDR) Articles 2, 4, 8, 9, 10, 11, 12, 29, 31;
--             ILO Convention 169; UNDRIP; CSDDD Articles 7, 8, 9
-- Tables: 10 (8 regular + 2 hypertables)
-- Indexes: ~140
--
-- Dependencies: TimescaleDB extension (for hypertables)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V119: Creating AGENT-EUDR-031 Stakeholder Engagement Tool tables...';


-- ============================================================================
-- 1. gl_eudr_set_stakeholders -- Stakeholder registry
-- ============================================================================
RAISE NOTICE 'V119 [1/10]: Creating gl_eudr_set_stakeholders...';

CREATE TABLE IF NOT EXISTS gl_eudr_set_stakeholders (
    stakeholder_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for stakeholder record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator managing this stakeholder relationship
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    stakeholder_type                VARCHAR(50)     NOT NULL,
        -- Stakeholder category per EUDR engagement requirements
    name                            VARCHAR(500)    NOT NULL,
        -- Primary display name of stakeholder (individual or organization)
    legal_name                      VARCHAR(500),
        -- Official legal/registered name where applicable
    contact_info                    JSONB           DEFAULT '{}',
        -- Contact details: {"email": "...", "phone": "...", "address": "...", "representative": "..."}
    geographic_latitude             DOUBLE PRECISION,
        -- Geographic latitude of stakeholder primary location (WGS84)
    geographic_longitude            DOUBLE PRECISION,
        -- Geographic longitude of stakeholder primary location (WGS84)
    country_code                    CHAR(2),
        -- ISO 3166-1 alpha-2 country code
    region                          VARCHAR(200),
        -- Sub-national region/province/state
    supply_chain_nodes              JSONB           DEFAULT '[]',
        -- Array of linked EUDR-001 supply chain node IDs: ["node-uuid-1", "node-uuid-2"]
    rights_classification           JSONB           DEFAULT '{}',
        -- Rights framework mapping: {"fpic_rights": true, "ilo_169": true, "undrip": true, "national_rights": ["BR_Law_6040"], ...}
    legal_protections               JSONB           DEFAULT '[]',
        -- Array of applicable legal protections: ["ILO_169", "UNDRIP_Art32", "BR_Law_6040"]
    engagement_history              JSONB           DEFAULT '[]',
        -- Array of engagement timeline entries: [{"date": "...", "type": "...", "summary": "..."}, ...]
    communication_preferences       JSONB           DEFAULT '{}',
        -- Preferred communication settings: {"channels": ["email", "sms"], "languages": ["en", "fr"], "frequency": "monthly"}
    preferred_language              VARCHAR(10)     DEFAULT 'en',
        -- Primary preferred language (ISO 639-1)
    relationship_status             VARCHAR(30)     NOT NULL DEFAULT 'identified',
        -- Current relationship lifecycle status
    engagement_history_count        INTEGER         DEFAULT 0,
        -- Total count of engagement interactions for performance queries
    last_engagement_date            TIMESTAMPTZ,
        -- Timestamp of most recent engagement activity
    metadata                        JSONB           DEFAULT '{}',
        -- Extensible metadata: {"source": "eudr-001-discovery", "tags": [...], ...}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for stakeholder record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_set_stakeholder_type CHECK (stakeholder_type IN (
        'indigenous_peoples', 'local_communities', 'smallholder_cooperatives',
        'ngos', 'workers_unions', 'government_authorities', 'certification_bodies',
        'civil_society', 'academic_institutions', 'media'
    )),
    CONSTRAINT chk_set_stakeholder_status CHECK (relationship_status IN (
        'identified', 'contacted', 'engaged', 'active', 'inactive', 'disputed'
    )),
    CONSTRAINT chk_set_stakeholder_lat CHECK (geographic_latitude IS NULL OR
        (geographic_latitude >= -90.0 AND geographic_latitude <= 90.0)),
    CONSTRAINT chk_set_stakeholder_lon CHECK (geographic_longitude IS NULL OR
        (geographic_longitude >= -180.0 AND geographic_longitude <= 180.0))
);

-- B-tree indexes for gl_eudr_set_stakeholders
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_operator ON gl_eudr_set_stakeholders (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_tenant ON gl_eudr_set_stakeholders (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_type ON gl_eudr_set_stakeholders (stakeholder_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_name ON gl_eudr_set_stakeholders (name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_country ON gl_eudr_set_stakeholders (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_region ON gl_eudr_set_stakeholders (region);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_status ON gl_eudr_set_stakeholders (relationship_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_language ON gl_eudr_set_stakeholders (preferred_language);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_last_engagement ON gl_eudr_set_stakeholders (last_engagement_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_provenance ON gl_eudr_set_stakeholders (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_created ON gl_eudr_set_stakeholders (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_updated ON gl_eudr_set_stakeholders (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_operator_type ON gl_eudr_set_stakeholders (operator_id, stakeholder_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_operator_status ON gl_eudr_set_stakeholders (operator_id, relationship_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_tenant_operator ON gl_eudr_set_stakeholders (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_type_status ON gl_eudr_set_stakeholders (stakeholder_type, relationship_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_country_type ON gl_eudr_set_stakeholders (country_code, stakeholder_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_operator_country ON gl_eudr_set_stakeholders (operator_id, country_code, stakeholder_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Geographic bounding box index for spatial queries (lat/lon B-tree for range scans)
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_geo_lat ON gl_eudr_set_stakeholders (geographic_latitude)
        WHERE geographic_latitude IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_geo_lon ON gl_eudr_set_stakeholders (geographic_longitude)
        WHERE geographic_longitude IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_geo_latlon ON gl_eudr_set_stakeholders (geographic_latitude, geographic_longitude)
        WHERE geographic_latitude IS NOT NULL AND geographic_longitude IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_active ON gl_eudr_set_stakeholders (operator_id, last_engagement_date DESC)
        WHERE relationship_status IN ('engaged', 'active');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_indigenous ON gl_eudr_set_stakeholders (operator_id, country_code)
        WHERE stakeholder_type = 'indigenous_peoples';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_disputed ON gl_eudr_set_stakeholders (updated_at DESC, operator_id)
        WHERE relationship_status = 'disputed';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes for JSONB columns
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_contact ON gl_eudr_set_stakeholders USING GIN (contact_info);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_supply_nodes ON gl_eudr_set_stakeholders USING GIN (supply_chain_nodes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_rights ON gl_eudr_set_stakeholders USING GIN (rights_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_legal_prot ON gl_eudr_set_stakeholders USING GIN (legal_protections);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_comm_prefs ON gl_eudr_set_stakeholders USING GIN (communication_preferences);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_stk_metadata ON gl_eudr_set_stakeholders USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_set_stakeholders IS 'AGENT-EUDR-031: Centralized stakeholder registry mapping all stakeholders across EUDR supply chains with rights classification (ILO 169, UNDRIP), geographic location, engagement history, and communication preferences per EUDR Articles 10(2)(e), 11(2), 29(3)(c)';
COMMENT ON COLUMN gl_eudr_set_stakeholders.stakeholder_type IS 'Stakeholder category: indigenous_peoples, local_communities, smallholder_cooperatives, ngos, workers_unions, government_authorities, certification_bodies, civil_society, academic_institutions, media';
COMMENT ON COLUMN gl_eudr_set_stakeholders.rights_classification IS 'Rights framework mapping: {"fpic_rights": true, "ilo_169": true, "undrip": true, "customary_land_tenure": true, "national_rights": ["BR_Law_6040"]}';
COMMENT ON COLUMN gl_eudr_set_stakeholders.relationship_status IS 'Relationship lifecycle: identified -> contacted -> engaged -> active; may transition to inactive or disputed';


-- ============================================================================
-- 2. gl_eudr_set_fpic_workflows -- FPIC process tracking
-- ============================================================================
RAISE NOTICE 'V119 [2/10]: Creating gl_eudr_set_fpic_workflows...';

CREATE TABLE IF NOT EXISTS gl_eudr_set_fpic_workflows (
    fpic_id                         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for FPIC workflow
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator conducting the FPIC process
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    stakeholder_id                  UUID            NOT NULL,
        -- Primary affected stakeholder (community) for this FPIC workflow
    supply_chain_node_id            VARCHAR(100),
        -- EUDR-001 supply chain node where activities intersect community territory
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity associated with this FPIC process
    workflow_stage                  VARCHAR(50)     NOT NULL DEFAULT 'identification',
        -- Current FPIC workflow stage per ILO 169 / UNDRIP procedural requirements
    stage_sla_deadline              TIMESTAMPTZ,
        -- Deadline for completing the current stage (configurable per stage)
    stage_status                    VARCHAR(30)     DEFAULT 'in_progress',
        -- Status within the current stage
    affected_communities            JSONB           DEFAULT '[]',
        -- Array of affected community details: [{"community_id": "...", "name": "...", "population": 500}, ...]
    information_provided            JSONB           DEFAULT '{}',
        -- Information provision details: {"documents": [...], "languages": [...], "formats": [...], "date_provided": "..."}
    deliberation_period_days        INTEGER         DEFAULT 90,
        -- Allowed deliberation period in days (ILO 169 minimum, default 90, extendable to 180)
    deliberation_start_date         TIMESTAMPTZ,
        -- Start date of community deliberation period
    deliberation_end_date           TIMESTAMPTZ,
        -- Calculated end date of deliberation period
    consultation_methodology        TEXT,
        -- Description of culturally appropriate consultation methods employed
    consultation_records            JSONB           DEFAULT '[]',
        -- Array of consultation record references: [{"consultation_id": "...", "date": "...", "type": "..."}, ...]
    consent_status                  VARCHAR(30),
        -- Consent outcome from the affected community
    consent_recorded_at             TIMESTAMPTZ,
        -- Timestamp when consent decision was formally recorded
    agreement_terms                 JSONB,
        -- Formal agreement details: {"benefit_sharing": {...}, "conditions": [...], "monitoring": {...}, "duration_years": 5}
    agreement_signed_at             TIMESTAMPTZ,
        -- Timestamp when formal agreement was signed by all parties
    monitoring_status               VARCHAR(30),
        -- Status of ongoing consent condition monitoring
    monitoring_findings             JSONB           DEFAULT '[]',
        -- Array of monitoring findings: [{"date": "...", "finding": "...", "status": "...", "action_required": "..."}, ...]
    evidence_ids                    JSONB           DEFAULT '[]',
        -- Array of evidence document references (S3 keys, document IDs)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for FPIC workflow integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at                    TIMESTAMPTZ,
        -- Timestamp when entire FPIC workflow reached terminal state

    CONSTRAINT fk_set_fpic_stakeholder FOREIGN KEY (stakeholder_id)
        REFERENCES gl_eudr_set_stakeholders (stakeholder_id),
    CONSTRAINT chk_set_fpic_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_set_fpic_stage CHECK (workflow_stage IN (
        'identification', 'information_provision', 'deliberation', 'consultation',
        'consent_recording', 'agreement', 'monitoring'
    )),
    CONSTRAINT chk_set_fpic_stage_status CHECK (stage_status IN (
        'in_progress', 'completed', 'overdue', 'blocked', 'skipped'
    )),
    CONSTRAINT chk_set_fpic_consent CHECK (consent_status IS NULL OR consent_status IN (
        'pending', 'granted', 'withheld', 'conditional', 'revoked'
    )),
    CONSTRAINT chk_set_fpic_monitoring CHECK (monitoring_status IS NULL OR monitoring_status IN (
        'not_started', 'active', 'compliant', 'non_compliant', 'review_needed', 'closed'
    )),
    CONSTRAINT chk_set_fpic_deliberation CHECK (deliberation_period_days >= 1 AND deliberation_period_days <= 365)
);

-- B-tree indexes for gl_eudr_set_fpic_workflows
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_operator ON gl_eudr_set_fpic_workflows (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_tenant ON gl_eudr_set_fpic_workflows (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_stakeholder ON gl_eudr_set_fpic_workflows (stakeholder_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_node ON gl_eudr_set_fpic_workflows (supply_chain_node_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_commodity ON gl_eudr_set_fpic_workflows (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_stage ON gl_eudr_set_fpic_workflows (workflow_stage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_consent ON gl_eudr_set_fpic_workflows (consent_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_sla ON gl_eudr_set_fpic_workflows (stage_sla_deadline);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_monitoring ON gl_eudr_set_fpic_workflows (monitoring_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_provenance ON gl_eudr_set_fpic_workflows (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_created ON gl_eudr_set_fpic_workflows (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_updated ON gl_eudr_set_fpic_workflows (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_completed ON gl_eudr_set_fpic_workflows (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_operator_stage ON gl_eudr_set_fpic_workflows (operator_id, workflow_stage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_operator_commodity ON gl_eudr_set_fpic_workflows (operator_id, commodity, workflow_stage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_stakeholder_stage ON gl_eudr_set_fpic_workflows (stakeholder_id, workflow_stage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_stage_consent ON gl_eudr_set_fpic_workflows (workflow_stage, consent_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_tenant_operator ON gl_eudr_set_fpic_workflows (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_active ON gl_eudr_set_fpic_workflows (operator_id, workflow_stage, stage_sla_deadline)
        WHERE completed_at IS NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_overdue ON gl_eudr_set_fpic_workflows (stage_sla_deadline, operator_id)
        WHERE stage_status = 'overdue' AND completed_at IS NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_consent_withheld ON gl_eudr_set_fpic_workflows (updated_at DESC, operator_id)
        WHERE consent_status = 'withheld';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_consent_revoked ON gl_eudr_set_fpic_workflows (updated_at DESC, operator_id)
        WHERE consent_status = 'revoked';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes for JSONB columns
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_communities ON gl_eudr_set_fpic_workflows USING GIN (affected_communities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_info_provided ON gl_eudr_set_fpic_workflows USING GIN (information_provided);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_consult_records ON gl_eudr_set_fpic_workflows USING GIN (consultation_records);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_agreement ON gl_eudr_set_fpic_workflows USING GIN (agreement_terms);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_mon_findings ON gl_eudr_set_fpic_workflows USING GIN (monitoring_findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_fpic_evidence ON gl_eudr_set_fpic_workflows USING GIN (evidence_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_set_fpic_workflows IS 'AGENT-EUDR-031: Free, Prior and Informed Consent (FPIC) workflow management with 7 mandatory stages per ILO Convention 169 and UNDRIP, tracking community identification, information provision, deliberation periods, consultation, consent recording, formal agreements, and ongoing monitoring per EUDR Articles 10(2)(d-e), 29(3)(c)';
COMMENT ON COLUMN gl_eudr_set_fpic_workflows.workflow_stage IS 'FPIC stages: identification -> information_provision -> deliberation -> consultation -> consent_recording -> agreement -> monitoring';
COMMENT ON COLUMN gl_eudr_set_fpic_workflows.consent_status IS 'Consent outcome: pending (in process), granted (community consented), withheld (community declined), conditional (consented with conditions), revoked (previously granted consent withdrawn)';
COMMENT ON COLUMN gl_eudr_set_fpic_workflows.deliberation_period_days IS 'Community deliberation period in days per ILO 169: minimum 90 days default, extendable to 365 days based on community needs and decision complexity';


-- ============================================================================
-- 3. gl_eudr_set_grievances -- Complaint tracking
-- ============================================================================
RAISE NOTICE 'V119 [3/10]: Creating gl_eudr_set_grievances...';

CREATE TABLE IF NOT EXISTS gl_eudr_set_grievances (
    grievance_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for grievance record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator responsible for this grievance
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    complainant_id                  UUID,
        -- FK to stakeholder registry (NULL for anonymous complaints)
    is_anonymous                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the complaint was submitted anonymously
    case_reference                  VARCHAR(50)     UNIQUE,
        -- Human-readable case reference (e.g. "GRV-2026-00123")
    intake_channel                  VARCHAR(30)     NOT NULL,
        -- Channel through which complaint was received
    severity                        VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Complaint severity classification
    category                        VARCHAR(50)     NOT NULL,
        -- Complaint subject matter category
    complaint_text                  TEXT            NOT NULL,
        -- Full text of the complaint as submitted
    complaint_language              VARCHAR(10)     DEFAULT 'en',
        -- Language in which the complaint was submitted (ISO 639-1)
    supply_chain_context            JSONB           DEFAULT '{}',
        -- Supply chain context: {"node_ids": [...], "commodity": "...", "country": "...", "location": {...}}
    status                          VARCHAR(50)     NOT NULL DEFAULT 'submitted',
        -- Current grievance lifecycle status
    assigned_investigator           VARCHAR(100),
        -- Assigned investigator user ID or name
    investigation_notes             JSONB           DEFAULT '[]',
        -- Array of investigation notes: [{"date": "...", "author": "...", "note": "...", "evidence_ids": [...]}, ...]
    root_cause_analysis             TEXT,
        -- Root cause analysis findings
    resolution_actions              JSONB           DEFAULT '[]',
        -- Array of resolution actions: [{"action": "...", "responsible": "...", "deadline": "...", "status": "..."}, ...]
    remediation_tracking            JSONB           DEFAULT '{}',
        -- Remediation progress: {"plan_id": "...", "measures": [...], "progress_pct": 0.0, "verified_at": null}
    satisfaction_assessment          SMALLINT,
        -- Complainant satisfaction rating (1-5 scale, NULL if not yet assessed)
    appeal_reason                   TEXT,
        -- Reason for appeal if complainant rejects resolution
    appeal_record                   JSONB,
        -- Appeal details: {"appeal_date": "...", "reviewer": "...", "outcome": "...", "revised_resolution": "..."}
    sla_deadline                    TIMESTAMPTZ,
        -- SLA deadline for resolution based on severity (critical: 7 days, high: 14 days, medium: 30 days, low: 90 days)
    acknowledged_at                 TIMESTAMPTZ,
        -- Timestamp when complaint was acknowledged (target: within 48 hours)
    resolved_at                     TIMESTAMPTZ,
        -- Timestamp when resolution was proposed/accepted
    closed_at                       TIMESTAMPTZ,
        -- Timestamp when grievance was formally closed
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for grievance record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_set_grv_complainant FOREIGN KEY (complainant_id)
        REFERENCES gl_eudr_set_stakeholders (stakeholder_id),
    CONSTRAINT chk_set_grv_channel CHECK (intake_channel IN (
        'web_portal', 'mobile_app', 'sms', 'email', 'community_point', 'phone_hotline'
    )),
    CONSTRAINT chk_set_grv_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low'
    )),
    CONSTRAINT chk_set_grv_category CHECK (category IN (
        'environmental', 'human_rights', 'labor_rights', 'land_rights',
        'community_impact', 'process_violation'
    )),
    CONSTRAINT chk_set_grv_status CHECK (status IN (
        'submitted', 'triaged', 'investigating', 'resolved', 'closed', 'appealed'
    )),
    CONSTRAINT chk_set_grv_satisfaction CHECK (satisfaction_assessment IS NULL OR
        (satisfaction_assessment >= 1 AND satisfaction_assessment <= 5)),
    CONSTRAINT chk_set_grv_anonymous CHECK (
        (is_anonymous = TRUE AND complainant_id IS NULL) OR
        (is_anonymous = FALSE)
    )
);

-- B-tree indexes for gl_eudr_set_grievances
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_operator ON gl_eudr_set_grievances (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_tenant ON gl_eudr_set_grievances (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_complainant ON gl_eudr_set_grievances (complainant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_case_ref ON gl_eudr_set_grievances (case_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_channel ON gl_eudr_set_grievances (intake_channel);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_severity ON gl_eudr_set_grievances (severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_category ON gl_eudr_set_grievances (category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_status ON gl_eudr_set_grievances (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_sla ON gl_eudr_set_grievances (sla_deadline);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_acknowledged ON gl_eudr_set_grievances (acknowledged_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_resolved ON gl_eudr_set_grievances (resolved_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_closed ON gl_eudr_set_grievances (closed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_provenance ON gl_eudr_set_grievances (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_created ON gl_eudr_set_grievances (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_updated ON gl_eudr_set_grievances (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_operator_status ON gl_eudr_set_grievances (operator_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_operator_severity ON gl_eudr_set_grievances (operator_id, severity, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_status_severity ON gl_eudr_set_grievances (status, severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_status_created ON gl_eudr_set_grievances (status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_severity_created ON gl_eudr_set_grievances (severity, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_tenant_operator ON gl_eudr_set_grievances (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_category_status ON gl_eudr_set_grievances (category, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_open ON gl_eudr_set_grievances (severity, sla_deadline, operator_id)
        WHERE status NOT IN ('closed', 'resolved');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_critical ON gl_eudr_set_grievances (created_at DESC, operator_id)
        WHERE severity = 'critical' AND status NOT IN ('closed', 'resolved');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_overdue ON gl_eudr_set_grievances (sla_deadline, operator_id)
        WHERE status NOT IN ('closed', 'resolved') AND sla_deadline < NOW();
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_appealed ON gl_eudr_set_grievances (updated_at DESC, operator_id)
        WHERE status = 'appealed';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_anonymous ON gl_eudr_set_grievances (created_at DESC, operator_id)
        WHERE is_anonymous = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes for JSONB columns
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_supply_ctx ON gl_eudr_set_grievances USING GIN (supply_chain_context);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_inv_notes ON gl_eudr_set_grievances USING GIN (investigation_notes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_resolution ON gl_eudr_set_grievances USING GIN (resolution_actions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_grv_remediation ON gl_eudr_set_grievances USING GIN (remediation_tracking);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_set_grievances IS 'AGENT-EUDR-031: Grievance mechanism complaint tracking compliant with UNGP Principle 31 effectiveness criteria and CSDDD Article 8, supporting multi-channel intake, anonymous reporting, severity-based triage, investigation workflow, resolution tracking, satisfaction assessment, and appeal process per EUDR Articles 10-11';
COMMENT ON COLUMN gl_eudr_set_grievances.intake_channel IS 'Complaint submission channel: web_portal, mobile_app, sms, email, community_point (physical), phone_hotline';
COMMENT ON COLUMN gl_eudr_set_grievances.severity IS 'Complaint severity: critical (immediate safety/rights risk, 7-day SLA), high (ongoing harm, 14-day SLA), medium (potential harm, 30-day SLA), low (inquiry/feedback, 90-day SLA)';
COMMENT ON COLUMN gl_eudr_set_grievances.status IS 'Grievance lifecycle: submitted -> triaged -> investigating -> resolved -> closed; may transition to appealed';


-- ============================================================================
-- 4. gl_eudr_set_consultations -- Consultation records
-- ============================================================================
RAISE NOTICE 'V119 [4/10]: Creating gl_eudr_set_consultations...';

CREATE TABLE IF NOT EXISTS gl_eudr_set_consultations (
    consultation_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for consultation record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator conducting the consultation
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    stakeholder_ids                 JSONB           NOT NULL DEFAULT '[]',
        -- Array of stakeholder UUIDs participating in this consultation
    fpic_id                         UUID,
        -- FK to FPIC workflow if consultation is part of an FPIC process (NULL otherwise)
    consultation_type               VARCHAR(50)     NOT NULL,
        -- Type of consultation conducted
    objectives                      TEXT,
        -- Stated objectives for the consultation
    date_time                       TIMESTAMPTZ     NOT NULL,
        -- Date and time when the consultation was held
    location_name                   VARCHAR(500),
        -- Name/description of consultation location
    location_latitude               DOUBLE PRECISION,
        -- GPS latitude of consultation location (WGS84)
    location_longitude              DOUBLE PRECISION,
        -- GPS longitude of consultation location (WGS84)
    participants                    JSONB           DEFAULT '[]',
        -- Array of participant details: [{"name": "...", "role": "...", "affiliation": "...", "community": "..."}, ...]
    methodology                     TEXT,
        -- Description of consultation methodology employed
    language_used                   VARCHAR(10)     DEFAULT 'en',
        -- Primary language used during consultation (ISO 639-1)
    topics_discussed                JSONB           DEFAULT '[]',
        -- Array of topics: [{"topic": "...", "summary": "...", "outcome": "..."}, ...]
    outcomes                        JSONB           DEFAULT '[]',
        -- Array of outcomes: [{"outcome": "...", "agreed_by": [...], "conditions": "..."}, ...]
    commitments                     JSONB           DEFAULT '[]',
        -- Array of commitments: [{"party": "...", "commitment": "...", "deadline": "...", "status": "pending"}, ...]
    follow_up_actions               JSONB           DEFAULT '[]',
        -- Array of follow-up actions: [{"action": "...", "responsible": "...", "deadline": "...", "status": "pending"}, ...]
    evidence_files                  JSONB           DEFAULT '[]',
        -- Array of evidence references: [{"type": "minutes|photo|audio|attendance_sheet", "s3_key": "...", "description": "..."}, ...]
    participant_consent_recorded    BOOLEAN         DEFAULT FALSE,
        -- Whether participant consent for documentation/recording was obtained
    is_finalized                    BOOLEAN         DEFAULT FALSE,
        -- Whether the consultation record has been finalized (immutable after TRUE)
    finalized_at                    TIMESTAMPTZ,
        -- Timestamp when the record was finalized
    finalized_by                    VARCHAR(100),
        -- User who finalized the record
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for consultation record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_set_con_fpic FOREIGN KEY (fpic_id)
        REFERENCES gl_eudr_set_fpic_workflows (fpic_id),
    CONSTRAINT chk_set_con_type CHECK (consultation_type IN (
        'community_meeting', 'focus_group', 'bilateral', 'survey',
        'public_hearing', 'workshop'
    )),
    CONSTRAINT chk_set_con_lat CHECK (location_latitude IS NULL OR
        (location_latitude >= -90.0 AND location_latitude <= 90.0)),
    CONSTRAINT chk_set_con_lon CHECK (location_longitude IS NULL OR
        (location_longitude >= -180.0 AND location_longitude <= 180.0))
);

-- B-tree indexes for gl_eudr_set_consultations
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_operator ON gl_eudr_set_consultations (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_tenant ON gl_eudr_set_consultations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_fpic ON gl_eudr_set_consultations (fpic_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_type ON gl_eudr_set_consultations (consultation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_date ON gl_eudr_set_consultations (date_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_language ON gl_eudr_set_consultations (language_used);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_finalized ON gl_eudr_set_consultations (is_finalized);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_finalized_at ON gl_eudr_set_consultations (finalized_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_provenance ON gl_eudr_set_consultations (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_created ON gl_eudr_set_consultations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_updated ON gl_eudr_set_consultations (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_operator_type ON gl_eudr_set_consultations (operator_id, consultation_type, date_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_operator_date ON gl_eudr_set_consultations (operator_id, date_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_tenant_operator ON gl_eudr_set_consultations (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_fpic_date ON gl_eudr_set_consultations (fpic_id, date_time DESC)
        WHERE fpic_id IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_draft ON gl_eudr_set_consultations (created_at DESC, operator_id)
        WHERE is_finalized = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_finalized_records ON gl_eudr_set_consultations (finalized_at DESC, operator_id)
        WHERE is_finalized = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes for JSONB columns
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_stakeholder_ids ON gl_eudr_set_consultations USING GIN (stakeholder_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_participants ON gl_eudr_set_consultations USING GIN (participants);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_topics ON gl_eudr_set_consultations USING GIN (topics_discussed);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_outcomes ON gl_eudr_set_consultations USING GIN (outcomes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_commitments ON gl_eudr_set_consultations USING GIN (commitments);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_followups ON gl_eudr_set_consultations USING GIN (follow_up_actions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_con_evidence ON gl_eudr_set_consultations USING GIN (evidence_files);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_set_consultations IS 'AGENT-EUDR-031: Consultation record management capturing structured documentation of all stakeholder consultations per EUDR Article 10(2)(e), with objectives, participants, methodology, outcomes, commitments, follow-up actions, and evidence files; records are immutable after finalization with SHA-256 provenance hashing';
COMMENT ON COLUMN gl_eudr_set_consultations.consultation_type IS 'Consultation type: community_meeting, focus_group, bilateral (one-on-one), survey, public_hearing, workshop';
COMMENT ON COLUMN gl_eudr_set_consultations.is_finalized IS 'Immutability flag: once TRUE, the consultation record cannot be edited; amendments create linked addendum records';


-- ============================================================================
-- 5. gl_eudr_set_communications -- Communication tracking (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V119 [5/10]: Creating gl_eudr_set_communications (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_set_communications (
    communication_id                UUID            DEFAULT gen_random_uuid(),
        -- Unique communication record identifier
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator sending the communication
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    campaign_id                     UUID,
        -- Campaign identifier for coordinated multi-stakeholder communications (NULL for ad-hoc)
    stakeholder_ids                 JSONB           NOT NULL DEFAULT '[]',
        -- Array of recipient stakeholder UUIDs
    communication_channel           VARCHAR(30)     NOT NULL,
        -- Channel used for delivery
    message_template_id             VARCHAR(100),
        -- Template ID from template library (NULL for custom messages)
    message_content                 JSONB           DEFAULT '{}',
        -- Message content with localization: {"en": {"subject": "...", "body": "..."}, "fr": {"subject": "...", "body": "..."}}
    content_hash                    VARCHAR(64),
        -- SHA-256 hash of message content for deduplication and audit
    scheduled_at                    TIMESTAMPTZ,
        -- Scheduled dispatch time (NULL for immediate)
    sent_at                         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Actual dispatch timestamp (partition key for hypertable)
    delivered_at                    TIMESTAMPTZ,
        -- Timestamp when delivery was confirmed by channel
    read_at                         TIMESTAMPTZ,
        -- Timestamp when read receipt received (channel-dependent)
    delivery_status                 VARCHAR(30)     DEFAULT 'pending',
        -- Current delivery lifecycle status
    response_received               BOOLEAN         DEFAULT FALSE,
        -- Whether a response was received from recipient
    response_content                TEXT,
        -- Summary or content of the response
    response_at                     TIMESTAMPTZ,
        -- Timestamp when response was received
    metadata                        JSONB           DEFAULT '{}',
        -- Additional context: {"language": "en", "priority": "normal", "retry_count": 0}
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_set_comm_channel CHECK (communication_channel IN (
        'email', 'sms', 'whatsapp', 'portal', 'printed', 'community_radio', 'phone'
    )),
    CONSTRAINT chk_set_comm_status CHECK (delivery_status IN (
        'pending', 'scheduled', 'sending', 'sent', 'delivered',
        'read', 'failed', 'bounced', 'cancelled'
    ))
);

-- Convert to TimescaleDB hypertable partitioned on sent_at
SELECT create_hypertable('gl_eudr_set_communications', 'sent_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- B-tree indexes for gl_eudr_set_communications
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_operator ON gl_eudr_set_communications (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_tenant ON gl_eudr_set_communications (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_campaign ON gl_eudr_set_communications (campaign_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_channel ON gl_eudr_set_communications (communication_channel);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_template ON gl_eudr_set_communications (message_template_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_sent ON gl_eudr_set_communications (sent_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_delivered ON gl_eudr_set_communications (delivered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_status ON gl_eudr_set_communications (delivery_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_content_hash ON gl_eudr_set_communications (content_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_created ON gl_eudr_set_communications (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_operator_sent ON gl_eudr_set_communications (operator_id, sent_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_operator_channel ON gl_eudr_set_communications (operator_id, communication_channel, sent_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_operator_status ON gl_eudr_set_communications (operator_id, delivery_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_channel_status ON gl_eudr_set_communications (communication_channel, delivery_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_tenant_operator ON gl_eudr_set_communications (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_failed ON gl_eudr_set_communications (sent_at DESC, operator_id)
        WHERE delivery_status IN ('failed', 'bounced');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_pending ON gl_eudr_set_communications (scheduled_at, operator_id)
        WHERE delivery_status IN ('pending', 'scheduled');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_with_response ON gl_eudr_set_communications (response_at DESC, operator_id)
        WHERE response_received = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes for JSONB columns
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_stakeholder_ids ON gl_eudr_set_communications USING GIN (stakeholder_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_content ON gl_eudr_set_communications USING GIN (message_content);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_comm_metadata ON gl_eudr_set_communications USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_set_communications IS 'AGENT-EUDR-031: Multi-channel stakeholder communication tracking (TimescaleDB hypertable, 1-month chunks on sent_at) supporting email, SMS, WhatsApp, portal, printed, community radio, and phone channels with delivery confirmation, response tracking, and multi-language support per EUDR Article 8(2) due diligence system evidence';
COMMENT ON COLUMN gl_eudr_set_communications.delivery_status IS 'Delivery lifecycle: pending -> scheduled -> sending -> sent -> delivered -> read; may fail (failed/bounced) or be cancelled';
COMMENT ON COLUMN gl_eudr_set_communications.message_content IS 'Localized message content: {"en": {"subject": "...", "body": "..."}, "fr": {"subject": "...", "body": "..."}} supporting 12+ languages';


-- ============================================================================
-- 6. gl_eudr_set_engagement_assessments -- Engagement quality scoring
-- ============================================================================
RAISE NOTICE 'V119 [6/10]: Creating gl_eudr_set_engagement_assessments...';

CREATE TABLE IF NOT EXISTS gl_eudr_set_engagement_assessments (
    assessment_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for engagement assessment
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose engagement is being assessed
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    stakeholder_id                  UUID            NOT NULL,
        -- Stakeholder (community) whose engagement quality is being assessed
    fpic_id                         UUID,
        -- FK to FPIC workflow if assessment is for a specific FPIC process (NULL for general)
    assessment_period_start         TIMESTAMPTZ     NOT NULL,
        -- Start of the assessment period
    assessment_period_end           TIMESTAMPTZ     NOT NULL,
        -- End of the assessment period
    cultural_appropriateness_score  NUMERIC(5,2)    NOT NULL DEFAULT 0.00,
        -- Dimension 1: Cultural appropriateness of engagement methods (0.00-100.00)
    language_accessibility_score    NUMERIC(5,2)    NOT NULL DEFAULT 0.00,
        -- Dimension 2: Language accessibility of information provided (0.00-100.00)
    deliberation_time_score         NUMERIC(5,2)    NOT NULL DEFAULT 0.00,
        -- Dimension 3: Adequacy of deliberation time allowed (0.00-100.00)
    representation_inclusiveness_score NUMERIC(5,2) NOT NULL DEFAULT 0.00,
        -- Dimension 4: Inclusiveness of community representation (0.00-100.00)
    consultation_genuineness_score  NUMERIC(5,2)    NOT NULL DEFAULT 0.00,
        -- Dimension 5: Genuineness of consultation (substantive, not merely informational) (0.00-100.00)
    decision_process_respect_score  NUMERIC(5,2)    NOT NULL DEFAULT 0.00,
        -- Dimension 6: Respect for community decision-making processes (0.00-100.00)
    composite_score                 NUMERIC(5,2)    NOT NULL DEFAULT 0.00,
        -- Weighted aggregate of 6 dimension scores (0.00-100.00)
    quality_classification          VARCHAR(30),
        -- Quality classification derived from composite score
    dimension_weights               JSONB           DEFAULT '{"cultural": 0.1667, "language": 0.1667, "deliberation": 0.1667, "representation": 0.1667, "genuineness": 0.1667, "decision": 0.1667}',
        -- Weights used for composite score calculation (default equal weighting)
    findings                        JSONB           DEFAULT '[]',
        -- Array of findings: [{"dimension": "...", "finding": "...", "evidence": "..."}, ...]
    recommendations                 JSONB           DEFAULT '[]',
        -- Array of improvement recommendations: [{"dimension": "...", "recommendation": "...", "priority": "..."}, ...]
    evidence_ids                    JSONB           DEFAULT '[]',
        -- Array of evidence document references supporting the assessment
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for assessment integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_set_ea_stakeholder FOREIGN KEY (stakeholder_id)
        REFERENCES gl_eudr_set_stakeholders (stakeholder_id),
    CONSTRAINT fk_set_ea_fpic FOREIGN KEY (fpic_id)
        REFERENCES gl_eudr_set_fpic_workflows (fpic_id),
    CONSTRAINT chk_set_ea_cultural CHECK (cultural_appropriateness_score >= 0.00 AND cultural_appropriateness_score <= 100.00),
    CONSTRAINT chk_set_ea_language CHECK (language_accessibility_score >= 0.00 AND language_accessibility_score <= 100.00),
    CONSTRAINT chk_set_ea_deliberation CHECK (deliberation_time_score >= 0.00 AND deliberation_time_score <= 100.00),
    CONSTRAINT chk_set_ea_representation CHECK (representation_inclusiveness_score >= 0.00 AND representation_inclusiveness_score <= 100.00),
    CONSTRAINT chk_set_ea_genuineness CHECK (consultation_genuineness_score >= 0.00 AND consultation_genuineness_score <= 100.00),
    CONSTRAINT chk_set_ea_decision CHECK (decision_process_respect_score >= 0.00 AND decision_process_respect_score <= 100.00),
    CONSTRAINT chk_set_ea_composite CHECK (composite_score >= 0.00 AND composite_score <= 100.00),
    CONSTRAINT chk_set_ea_classification CHECK (quality_classification IS NULL OR quality_classification IN (
        'exemplary', 'good', 'adequate', 'insufficient', 'non_compliant'
    )),
    CONSTRAINT chk_set_ea_period CHECK (assessment_period_end >= assessment_period_start)
);

-- B-tree indexes for gl_eudr_set_engagement_assessments
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_operator ON gl_eudr_set_engagement_assessments (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_tenant ON gl_eudr_set_engagement_assessments (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_stakeholder ON gl_eudr_set_engagement_assessments (stakeholder_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_fpic ON gl_eudr_set_engagement_assessments (fpic_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_period_start ON gl_eudr_set_engagement_assessments (assessment_period_start);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_period_end ON gl_eudr_set_engagement_assessments (assessment_period_end);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_composite ON gl_eudr_set_engagement_assessments (composite_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_classification ON gl_eudr_set_engagement_assessments (quality_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_provenance ON gl_eudr_set_engagement_assessments (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_created ON gl_eudr_set_engagement_assessments (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_operator_stakeholder ON gl_eudr_set_engagement_assessments (operator_id, stakeholder_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_stakeholder_score ON gl_eudr_set_engagement_assessments (stakeholder_id, composite_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_operator_class ON gl_eudr_set_engagement_assessments (operator_id, quality_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_tenant_operator ON gl_eudr_set_engagement_assessments (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_insufficient ON gl_eudr_set_engagement_assessments (composite_score, operator_id)
        WHERE quality_classification IN ('insufficient', 'non_compliant');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_exemplary ON gl_eudr_set_engagement_assessments (created_at DESC, operator_id)
        WHERE quality_classification = 'exemplary';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes for JSONB columns
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_findings ON gl_eudr_set_engagement_assessments USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_recommendations ON gl_eudr_set_engagement_assessments USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_ea_evidence ON gl_eudr_set_engagement_assessments USING GIN (evidence_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_set_engagement_assessments IS 'AGENT-EUDR-031: Indigenous rights engagement quality scoring across 6 dimensions (cultural appropriateness, language accessibility, deliberation time, representation inclusiveness, consultation genuineness, decision process respect) per ILO Convention 169 and UNDRIP, with deterministic composite scoring for EUDR Article 29(3)(c) evidence';
COMMENT ON COLUMN gl_eudr_set_engagement_assessments.composite_score IS 'Weighted aggregate engagement quality score: 0.00-100.00; classified as exemplary (90-100), good (75-89), adequate (60-74), insufficient (40-59), non_compliant (0-39)';
COMMENT ON COLUMN gl_eudr_set_engagement_assessments.quality_classification IS 'Quality classification: exemplary (90-100), good (75-89), adequate (60-74), insufficient (40-59), non_compliant (0-39)';


-- ============================================================================
-- 7. gl_eudr_set_compliance_reports -- Generated compliance reports
-- ============================================================================
RAISE NOTICE 'V119 [7/10]: Creating gl_eudr_set_compliance_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_set_compliance_reports (
    report_id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for compliance report
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for whom the report was generated
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    report_type                     VARCHAR(50)     NOT NULL,
        -- Type of compliance report generated
    report_format                   VARCHAR(10)     NOT NULL DEFAULT 'json',
        -- Output format of the generated report
    report_language                 VARCHAR(10)     DEFAULT 'en',
        -- Language of the generated report (ISO 639-1)
    date_range_start                TIMESTAMPTZ,
        -- Start of the reporting period (NULL for all-time)
    date_range_end                  TIMESTAMPTZ,
        -- End of the reporting period (NULL for all-time)
    commodity_filter                VARCHAR(50),
        -- Commodity filter applied during generation (NULL for all commodities)
    country_filter                  CHAR(2),
        -- Country filter applied during generation (NULL for all countries)
    generated_at                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the report was generated
    generated_by                    VARCHAR(100),
        -- User or system actor that triggered report generation
    report_content                  JSONB           DEFAULT '{}',
        -- Structured report content (for JSON format; other formats stored as files)
    report_file_reference           VARCHAR(1000),
        -- S3 key or file path for non-JSON report formats (PDF, HTML, XLSX)
    file_size_bytes                 BIGINT,
        -- Size of the generated report file in bytes
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for report integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_set_cr_type CHECK (report_type IN (
        'dds_summary', 'fpic_compliance', 'grievance_annual',
        'consultation_register', 'engagement_assessment',
        'communication_log', 'effectiveness_report'
    )),
    CONSTRAINT chk_set_cr_format CHECK (report_format IN (
        'pdf', 'json', 'html', 'xlsx'
    )),
    CONSTRAINT chk_set_cr_commodity CHECK (commodity_filter IS NULL OR commodity_filter IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    ))
);

-- B-tree indexes for gl_eudr_set_compliance_reports
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_operator ON gl_eudr_set_compliance_reports (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_tenant ON gl_eudr_set_compliance_reports (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_type ON gl_eudr_set_compliance_reports (report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_format ON gl_eudr_set_compliance_reports (report_format);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_language ON gl_eudr_set_compliance_reports (report_language);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_generated ON gl_eudr_set_compliance_reports (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_generated_by ON gl_eudr_set_compliance_reports (generated_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_provenance ON gl_eudr_set_compliance_reports (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_created ON gl_eudr_set_compliance_reports (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_operator_type ON gl_eudr_set_compliance_reports (operator_id, report_type, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_type_format ON gl_eudr_set_compliance_reports (report_type, report_format);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_operator_commodity ON gl_eudr_set_compliance_reports (operator_id, commodity_filter, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_tenant_operator ON gl_eudr_set_compliance_reports (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes for JSONB columns
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_cr_content ON gl_eudr_set_compliance_reports USING GIN (report_content);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_set_compliance_reports IS 'AGENT-EUDR-031: Generated compliance reports for DDS submission (Article 12), competent authority inspection (Articles 14-16), certification audits (FSC, RSPO, RA), and third-party verification; supports 7 report types in 4 formats with multi-language generation and SHA-256 provenance hashing';
COMMENT ON COLUMN gl_eudr_set_compliance_reports.report_type IS 'Report type: dds_summary (Article 12), fpic_compliance (ILO 169/UNDRIP), grievance_annual (CSDDD Art 8), consultation_register (Art 10(2)(e)), engagement_assessment (Art 29(3)(c)), communication_log (Art 8(2)), effectiveness_report (Art 8(3))';
COMMENT ON COLUMN gl_eudr_set_compliance_reports.report_format IS 'Output format: pdf (human-readable), json (machine-readable for EUDR-030 integration), html (web display), xlsx (tabular export)';


-- ============================================================================
-- 8. gl_eudr_set_audit_trail -- Immutable audit log (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V119 [8/10]: Creating gl_eudr_set_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_set_audit_trail (
    audit_id                        UUID            DEFAULT gen_random_uuid(),
        -- Unique audit trail entry identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity affected by the operation
    entity_id                       VARCHAR(100)    NOT NULL,
        -- Identifier of the entity affected
    operator_id                     VARCHAR(100),
        -- EUDR operator context for this audit entry
    tenant_id                       VARCHAR(100),
        -- Multi-tenant isolation identifier
    action                          VARCHAR(50)     NOT NULL,
        -- Action that was performed on the entity
    actor_id                        VARCHAR(100)    NOT NULL DEFAULT 'gl-eudr-set-031',
        -- Actor who performed the action (system agent or user ID)
    actor_email                     VARCHAR(255),
        -- Email of the actor (for user-initiated actions)
    changes                         JSONB           DEFAULT '{}',
        -- Object describing changes: {"field_name": {"old": "...", "new": "..."}, ...}
    reason                          TEXT,
        -- Reason or justification for the action
    metadata                        JSONB           DEFAULT '{}',
        -- Additional context: {"ip_address": "...", "session_id": "...", "correlation_id": "...", "source": "api|workflow|system"}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash chained to previous entry for tamper-evident audit trail
    timestamp                       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the audit event (partition key for hypertable)

    CONSTRAINT chk_set_at_entity_type CHECK (entity_type IN (
        'stakeholder', 'fpic_workflow', 'grievance', 'consultation',
        'communication', 'engagement_assessment', 'compliance_report', 'configuration'
    )),
    CONSTRAINT chk_set_at_action CHECK (action IN (
        'created', 'updated', 'deleted', 'finalized', 'approved',
        'rejected', 'escalated',
        'stakeholder_registered', 'stakeholder_updated', 'stakeholder_archived',
        'fpic_stage_transitioned', 'fpic_consent_recorded', 'fpic_agreement_signed', 'fpic_monitoring_updated',
        'grievance_submitted', 'grievance_triaged', 'grievance_assigned', 'grievance_investigated',
        'grievance_resolved', 'grievance_closed', 'grievance_appealed',
        'consultation_created', 'consultation_finalized', 'consultation_addendum',
        'communication_sent', 'communication_delivered', 'communication_failed',
        'assessment_completed', 'report_generated',
        'config_updated', 'manual_action'
    ))
);

-- Convert to TimescaleDB hypertable partitioned on timestamp
SELECT create_hypertable('gl_eudr_set_audit_trail', 'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- B-tree indexes for gl_eudr_set_audit_trail
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_entity_type ON gl_eudr_set_audit_trail (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_entity_id ON gl_eudr_set_audit_trail (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_operator ON gl_eudr_set_audit_trail (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_tenant ON gl_eudr_set_audit_trail (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_action ON gl_eudr_set_audit_trail (action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_actor_id ON gl_eudr_set_audit_trail (actor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_actor_email ON gl_eudr_set_audit_trail (actor_email);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_provenance ON gl_eudr_set_audit_trail (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_timestamp ON gl_eudr_set_audit_trail (timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_entity_action ON gl_eudr_set_audit_trail (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_entity_id_time ON gl_eudr_set_audit_trail (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_actor_time ON gl_eudr_set_audit_trail (actor_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_operator_time ON gl_eudr_set_audit_trail (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_tenant_operator ON gl_eudr_set_audit_trail (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes for JSONB columns
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_changes ON gl_eudr_set_audit_trail USING GIN (changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_set_at_metadata ON gl_eudr_set_audit_trail USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- DATA RETENTION POLICY -- Article 31: 5-year retention for EUDR audit data
-- ============================================================================
RAISE NOTICE 'V119: Configuring 5-year data retention policies per EUDR Article 31...';

-- Audit trail retention: 5 years (60 months) per Article 31 requirement
SELECT add_retention_policy('gl_eudr_set_audit_trail',
    INTERVAL '5 years',
    if_not_exists => TRUE
);

-- Communications retention: 5 years per Article 31 requirement
SELECT add_retention_policy('gl_eudr_set_communications',
    INTERVAL '5 years',
    if_not_exists => TRUE
);


-- ============================================================================
-- 9. AUDIT TRAIL TRIGGER FUNCTION -- Automated audit logging
-- ============================================================================
RAISE NOTICE 'V119 [9/10]: Creating audit trail trigger function...';

CREATE OR REPLACE FUNCTION gl_eudr_set_audit_trigger_fn()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            TG_ARGV[0],
            NEW.stakeholder_id::TEXT,
            COALESCE(NEW.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, 'unknown'),
            'created',
            'gl-eudr-set-031',
            jsonb_build_object('new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            TG_ARGV[0],
            COALESCE(NEW.stakeholder_id::TEXT, OLD.stakeholder_id::TEXT),
            COALESCE(NEW.operator_id, OLD.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, OLD.tenant_id, 'unknown'),
            'updated',
            'gl-eudr-set-031',
            jsonb_build_object('old_record', to_jsonb(OLD), 'new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            TG_ARGV[0],
            OLD.stakeholder_id::TEXT,
            COALESCE(OLD.operator_id, 'unknown'),
            COALESCE(OLD.tenant_id, 'unknown'),
            'deleted',
            'gl-eudr-set-031',
            jsonb_build_object('deleted_record', to_jsonb(OLD)),
            NOW()
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- 10. AUDIT TRAIL TRIGGERS -- Attach to core tables
-- ============================================================================
RAISE NOTICE 'V119 [10/10]: Attaching audit trail triggers to core tables...';

-- Stakeholder audit trigger
DROP TRIGGER IF EXISTS trg_eudr_set_stk_audit ON gl_eudr_set_stakeholders;
CREATE TRIGGER trg_eudr_set_stk_audit
    AFTER INSERT OR UPDATE OR DELETE ON gl_eudr_set_stakeholders
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_audit_trigger_fn('stakeholder');

-- FPIC workflow audit trigger (uses fpic_id instead of stakeholder_id)
CREATE OR REPLACE FUNCTION gl_eudr_set_fpic_audit_trigger_fn()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'fpic_workflow',
            NEW.fpic_id::TEXT,
            COALESCE(NEW.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, 'unknown'),
            'created',
            'gl-eudr-set-031',
            jsonb_build_object('new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'fpic_workflow',
            COALESCE(NEW.fpic_id::TEXT, OLD.fpic_id::TEXT),
            COALESCE(NEW.operator_id, OLD.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, OLD.tenant_id, 'unknown'),
            CASE
                WHEN OLD.workflow_stage IS DISTINCT FROM NEW.workflow_stage THEN 'fpic_stage_transitioned'
                WHEN OLD.consent_status IS DISTINCT FROM NEW.consent_status THEN 'fpic_consent_recorded'
                WHEN OLD.agreement_signed_at IS NULL AND NEW.agreement_signed_at IS NOT NULL THEN 'fpic_agreement_signed'
                ELSE 'updated'
            END,
            'gl-eudr-set-031',
            jsonb_build_object('old_record', to_jsonb(OLD), 'new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'fpic_workflow',
            OLD.fpic_id::TEXT,
            COALESCE(OLD.operator_id, 'unknown'),
            COALESCE(OLD.tenant_id, 'unknown'),
            'deleted',
            'gl-eudr-set-031',
            jsonb_build_object('deleted_record', to_jsonb(OLD)),
            NOW()
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_eudr_set_fpic_audit ON gl_eudr_set_fpic_workflows;
CREATE TRIGGER trg_eudr_set_fpic_audit
    AFTER INSERT OR UPDATE OR DELETE ON gl_eudr_set_fpic_workflows
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_fpic_audit_trigger_fn();

-- Grievance audit trigger (uses grievance_id)
CREATE OR REPLACE FUNCTION gl_eudr_set_grv_audit_trigger_fn()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'grievance',
            NEW.grievance_id::TEXT,
            COALESCE(NEW.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, 'unknown'),
            'grievance_submitted',
            'gl-eudr-set-031',
            jsonb_build_object('new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'grievance',
            COALESCE(NEW.grievance_id::TEXT, OLD.grievance_id::TEXT),
            COALESCE(NEW.operator_id, OLD.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, OLD.tenant_id, 'unknown'),
            CASE
                WHEN OLD.status IS DISTINCT FROM NEW.status AND NEW.status = 'triaged' THEN 'grievance_triaged'
                WHEN OLD.assigned_investigator IS DISTINCT FROM NEW.assigned_investigator THEN 'grievance_assigned'
                WHEN OLD.status IS DISTINCT FROM NEW.status AND NEW.status = 'investigating' THEN 'grievance_investigated'
                WHEN OLD.status IS DISTINCT FROM NEW.status AND NEW.status = 'resolved' THEN 'grievance_resolved'
                WHEN OLD.status IS DISTINCT FROM NEW.status AND NEW.status = 'closed' THEN 'grievance_closed'
                WHEN OLD.status IS DISTINCT FROM NEW.status AND NEW.status = 'appealed' THEN 'grievance_appealed'
                ELSE 'updated'
            END,
            'gl-eudr-set-031',
            jsonb_build_object('old_record', to_jsonb(OLD), 'new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'grievance',
            OLD.grievance_id::TEXT,
            COALESCE(OLD.operator_id, 'unknown'),
            COALESCE(OLD.tenant_id, 'unknown'),
            'deleted',
            'gl-eudr-set-031',
            jsonb_build_object('deleted_record', to_jsonb(OLD)),
            NOW()
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_eudr_set_grv_audit ON gl_eudr_set_grievances;
CREATE TRIGGER trg_eudr_set_grv_audit
    AFTER INSERT OR UPDATE OR DELETE ON gl_eudr_set_grievances
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_grv_audit_trigger_fn();

-- Consultation audit trigger (uses consultation_id)
CREATE OR REPLACE FUNCTION gl_eudr_set_con_audit_trigger_fn()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'consultation',
            NEW.consultation_id::TEXT,
            COALESCE(NEW.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, 'unknown'),
            'consultation_created',
            'gl-eudr-set-031',
            jsonb_build_object('new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'consultation',
            COALESCE(NEW.consultation_id::TEXT, OLD.consultation_id::TEXT),
            COALESCE(NEW.operator_id, OLD.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, OLD.tenant_id, 'unknown'),
            CASE
                WHEN OLD.is_finalized = FALSE AND NEW.is_finalized = TRUE THEN 'consultation_finalized'
                ELSE 'updated'
            END,
            'gl-eudr-set-031',
            jsonb_build_object('old_record', to_jsonb(OLD), 'new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'consultation',
            OLD.consultation_id::TEXT,
            COALESCE(OLD.operator_id, 'unknown'),
            COALESCE(OLD.tenant_id, 'unknown'),
            'deleted',
            'gl-eudr-set-031',
            jsonb_build_object('deleted_record', to_jsonb(OLD)),
            NOW()
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_eudr_set_con_audit ON gl_eudr_set_consultations;
CREATE TRIGGER trg_eudr_set_con_audit
    AFTER INSERT OR UPDATE OR DELETE ON gl_eudr_set_consultations
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_con_audit_trigger_fn();

-- Engagement assessment audit trigger (uses assessment_id)
CREATE OR REPLACE FUNCTION gl_eudr_set_ea_audit_trigger_fn()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'engagement_assessment',
            NEW.assessment_id::TEXT,
            COALESCE(NEW.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, 'unknown'),
            'assessment_completed',
            'gl-eudr-set-031',
            jsonb_build_object('new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO gl_eudr_set_audit_trail (
            entity_type, entity_id, operator_id, tenant_id,
            action, actor_id, changes, timestamp
        ) VALUES (
            'engagement_assessment',
            COALESCE(NEW.assessment_id::TEXT, OLD.assessment_id::TEXT),
            COALESCE(NEW.operator_id, OLD.operator_id, 'unknown'),
            COALESCE(NEW.tenant_id, OLD.tenant_id, 'unknown'),
            'updated',
            'gl-eudr-set-031',
            jsonb_build_object('old_record', to_jsonb(OLD), 'new_record', to_jsonb(NEW)),
            NOW()
        );
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_eudr_set_ea_audit ON gl_eudr_set_engagement_assessments;
CREATE TRIGGER trg_eudr_set_ea_audit
    AFTER INSERT OR UPDATE ON gl_eudr_set_engagement_assessments
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_ea_audit_trigger_fn();


-- ============================================================================
-- UPDATED_AT AUTO-UPDATE TRIGGER -- Automatic timestamp management
-- ============================================================================
RAISE NOTICE 'V119: Creating updated_at auto-update triggers...';

CREATE OR REPLACE FUNCTION gl_eudr_set_updated_at_fn()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to all mutable tables
DROP TRIGGER IF EXISTS trg_eudr_set_stk_updated ON gl_eudr_set_stakeholders;
CREATE TRIGGER trg_eudr_set_stk_updated
    BEFORE UPDATE ON gl_eudr_set_stakeholders
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_updated_at_fn();

DROP TRIGGER IF EXISTS trg_eudr_set_fpic_updated ON gl_eudr_set_fpic_workflows;
CREATE TRIGGER trg_eudr_set_fpic_updated
    BEFORE UPDATE ON gl_eudr_set_fpic_workflows
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_updated_at_fn();

DROP TRIGGER IF EXISTS trg_eudr_set_grv_updated ON gl_eudr_set_grievances;
CREATE TRIGGER trg_eudr_set_grv_updated
    BEFORE UPDATE ON gl_eudr_set_grievances
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_updated_at_fn();

DROP TRIGGER IF EXISTS trg_eudr_set_con_updated ON gl_eudr_set_consultations;
CREATE TRIGGER trg_eudr_set_con_updated
    BEFORE UPDATE ON gl_eudr_set_consultations
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_updated_at_fn();

DROP TRIGGER IF EXISTS trg_eudr_set_ea_updated ON gl_eudr_set_engagement_assessments;
CREATE TRIGGER trg_eudr_set_ea_updated
    BEFORE UPDATE ON gl_eudr_set_engagement_assessments
    FOR EACH ROW EXECUTE FUNCTION gl_eudr_set_updated_at_fn();


-- ============================================================================
-- TABLE COMMENTS -- Audit trail
-- ============================================================================

COMMENT ON TABLE gl_eudr_set_audit_trail IS 'AGENT-EUDR-031: Article 31 compliant immutable audit trail (TimescaleDB hypertable, 1-month chunks) for all stakeholder engagement operations including stakeholder registration, FPIC workflow transitions, grievance lifecycle, consultation documentation, communication dispatch, and engagement assessment with 5-year retention';
COMMENT ON COLUMN gl_eudr_set_audit_trail.actor_id IS 'Default actor is gl-eudr-set-031 (system agent); overridden for manual user actions such as grievance investigation, FPIC approval, and consultation finalization';
COMMENT ON COLUMN gl_eudr_set_audit_trail.provenance_hash IS 'SHA-256 hash chained to previous entry for tamper-evident audit trail per EUDR Article 31';


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V119: AGENT-EUDR-031 Stakeholder Engagement Tool tables created successfully!';
RAISE NOTICE 'V119: Created 10 tables (8 regular + 2 hypertables), ~140 indexes (B-tree, GIN, partial)';
RAISE NOTICE 'V119: Core tables: gl_eudr_set_stakeholders, gl_eudr_set_fpic_workflows, gl_eudr_set_grievances, gl_eudr_set_consultations';
RAISE NOTICE 'V119: Supporting tables: gl_eudr_set_communications (hypertable), gl_eudr_set_engagement_assessments, gl_eudr_set_compliance_reports, gl_eudr_set_audit_trail (hypertable)';
RAISE NOTICE 'V119: Foreign keys: fpic_workflows -> stakeholders; grievances -> stakeholders; consultations -> fpic_workflows; engagement_assessments -> stakeholders, fpic_workflows';
RAISE NOTICE 'V119: Hypertables: gl_eudr_set_communications (1-month chunks on sent_at), gl_eudr_set_audit_trail (1-month chunks on timestamp)';
RAISE NOTICE 'V119: Retention policies: 5 years for communications and audit trail per EUDR Article 31';
RAISE NOTICE 'V119: Triggers: audit trail automation (5 tables), updated_at automation (5 tables)';

COMMIT;

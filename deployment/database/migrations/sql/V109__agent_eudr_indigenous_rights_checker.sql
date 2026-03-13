-- ============================================================================
-- V109: AGENT-EUDR-021 Indigenous Rights Checker Agent
-- ============================================================================
-- Creates tables for indigenous territory registry, FPIC documentation,
-- land rights overlap analysis, community consultations, rights violations,
-- community registry, FPIC workflows, consultation scoring, territory data
-- sources, country FPIC requirements, compliance reports, territory-plot
-- associations, FPIC verification audit logs, overlap analysis audit logs,
-- and rights violation event streams.
--
-- Schema: eudr_indigenous_rights (15 tables)
-- Tables: 15 (12 regular + 3 hypertables)
-- Hypertables: gl_eudr_irc_fpic_verification_log (30d chunks),
--              gl_eudr_irc_overlap_analysis_log (30d chunks),
--              gl_eudr_irc_rights_violation_events (30d chunks)
-- Continuous Aggregates: 2 (daily_violation_counts + monthly_fpic_compliance)
-- Retention Policies: 3 (5 years per EUDR Article 31)
-- Indexes: ~131 (B-tree, GIST, GIN, partial)
-- GIST spatial indexes: 3 (on geometry columns)
-- GIN indexes: 16 (on JSONB columns)
-- Partial indexes: 7 (for active/filtered records)
--
-- Dependencies: TimescaleDB extension (V002), PostGIS extension
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V109: Creating AGENT-EUDR-021 Indigenous Rights Checker tables...';


-- ============================================================================
-- 1. gl_eudr_irc_indigenous_territories — Master territory registry
-- ============================================================================
RAISE NOTICE 'V109 [1/15]: Creating gl_eudr_irc_indigenous_territories...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_indigenous_territories (
    territory_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_name              VARCHAR(500)    NOT NULL,
        -- Official or commonly recognized territory name
    indigenous_group            VARCHAR(500)    NOT NULL,
        -- Name of the indigenous people or community
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    geometry                    GEOMETRY(MULTIPOLYGON, 4326),
        -- PostGIS multipolygon in WGS-84 (SRID 4326)
    area_hectares               NUMERIC(12,2)   CHECK (area_hectares >= 0),
        -- Total territory area in hectares
    legal_status                VARCHAR(50)     NOT NULL DEFAULT 'unrecognized',
        -- Current legal recognition status
    recognition_date            DATE,
        -- Date when territory was formally recognized
    data_source                 VARCHAR(100),
        -- Origin dataset identifier (e.g., 'RAISG', 'LandMark', 'FUNAI')
    data_source_url             TEXT,
        -- URL to the original data source or record
    confidence_score            NUMERIC(3,2)    CHECK (confidence_score >= 0 AND confidence_score <= 1),
        -- Data quality confidence (0.0 = low, 1.0 = high)
    metadata                    JSONB           DEFAULT '{}',
        -- Additional territory attributes (languages, treaties, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_irc_terr_legal_status CHECK (legal_status IN (
        'recognized', 'titled', 'pending', 'claimed', 'unrecognized',
        'disputed', 'reserved', 'protected'
    ))
);

-- GIST spatial index on geometry
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_geometry ON gl_eudr_irc_indigenous_territories USING GIST (geometry);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_name ON gl_eudr_irc_indigenous_territories (territory_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_group ON gl_eudr_irc_indigenous_territories (indigenous_group);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_country ON gl_eudr_irc_indigenous_territories (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_legal ON gl_eudr_irc_indigenous_territories (legal_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_area ON gl_eudr_irc_indigenous_territories (area_hectares);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_confidence ON gl_eudr_irc_indigenous_territories (confidence_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_source ON gl_eudr_irc_indigenous_territories (data_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_tenant ON gl_eudr_irc_indigenous_territories (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_created ON gl_eudr_irc_indigenous_territories (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_country_legal ON gl_eudr_irc_indigenous_territories (country_code, legal_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for recognized/titled territories
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_recognized ON gl_eudr_irc_indigenous_territories (country_code, area_hectares DESC)
        WHERE legal_status IN ('recognized', 'titled', 'protected');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_terr_metadata ON gl_eudr_irc_indigenous_territories USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_indigenous_territories IS 'Master registry of indigenous and traditional community territories with PostGIS geometry, legal status, and data provenance';
COMMENT ON COLUMN gl_eudr_irc_indigenous_territories.geometry IS 'PostGIS MULTIPOLYGON in WGS-84 (SRID 4326) representing territory boundaries';
COMMENT ON COLUMN gl_eudr_irc_indigenous_territories.confidence_score IS 'Data quality confidence: 0.0 = low certainty, 1.0 = high certainty (survey-grade)';
COMMENT ON COLUMN gl_eudr_irc_indigenous_territories.legal_status IS 'Legal recognition: recognized (state-acknowledged), titled (formal land title), pending (application in process), claimed (community assertion), unrecognized, disputed, reserved (government reserve), protected (conservation area)';


-- ============================================================================
-- 2. gl_eudr_irc_fpic_documents — FPIC documentation registry
-- ============================================================================
RAISE NOTICE 'V109 [2/15]: Creating gl_eudr_irc_fpic_documents...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_fpic_documents (
    document_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Reference to the production plot requiring FPIC
    territory_id                UUID            NOT NULL REFERENCES gl_eudr_irc_indigenous_territories(territory_id),
        -- Territory from which consent is sought
    document_type               VARCHAR(50)     NOT NULL,
        -- Type of FPIC documentation
    document_date               DATE            NOT NULL,
        -- Date the document was issued or signed
    issuing_authority           VARCHAR(500),
        -- Organization or body that issued the document
    consent_status              VARCHAR(50)     NOT NULL DEFAULT 'pending',
        -- Current consent determination
    document_url                TEXT,
        -- URL or storage path to the document
    verification_status         VARCHAR(50)     NOT NULL DEFAULT 'unverified',
        -- Document verification state
    verification_date           TIMESTAMPTZ,
        -- When the document was last verified
    fpic_score                  NUMERIC(5,2)    CHECK (fpic_score >= 0 AND fpic_score <= 100),
        -- Composite FPIC compliance score (0-100)
    score_breakdown             JSONB           DEFAULT '{}',
        -- { "completeness": 85, "timeliness": 90, "authenticity": 95, "good_faith": 80 }
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for document integrity verification
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_irc_fpic_doc_type CHECK (document_type IN (
        'consent_agreement', 'consultation_record', 'community_resolution',
        'government_certification', 'ngo_attestation', 'impact_assessment',
        'benefit_sharing_agreement', 'withdrawal_notice', 'amendment'
    )),
    CONSTRAINT chk_irc_fpic_consent CHECK (consent_status IN (
        'granted', 'denied', 'conditional', 'pending', 'withdrawn', 'expired'
    )),
    CONSTRAINT chk_irc_fpic_verification CHECK (verification_status IN (
        'verified', 'unverified', 'rejected', 'expired', 'under_review'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_plot ON gl_eudr_irc_fpic_documents (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_territory ON gl_eudr_irc_fpic_documents (territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_type ON gl_eudr_irc_fpic_documents (document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_consent ON gl_eudr_irc_fpic_documents (consent_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_verification ON gl_eudr_irc_fpic_documents (verification_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_score ON gl_eudr_irc_fpic_documents (fpic_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_provenance ON gl_eudr_irc_fpic_documents (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_tenant ON gl_eudr_irc_fpic_documents (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_created ON gl_eudr_irc_fpic_documents (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_plot_territory ON gl_eudr_irc_fpic_documents (plot_id, territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active consent documents
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_active_consent ON gl_eudr_irc_fpic_documents (plot_id, fpic_score DESC)
        WHERE consent_status = 'granted' AND verification_status = 'verified';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fpic_score_brkdwn ON gl_eudr_irc_fpic_documents USING GIN (score_breakdown);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_fpic_documents IS 'Free, Prior and Informed Consent (FPIC) documentation registry linking production plots to indigenous territory consent status';
COMMENT ON COLUMN gl_eudr_irc_fpic_documents.fpic_score IS 'Composite FPIC compliance score (0-100) computed from completeness, timeliness, authenticity, and good-faith components';
COMMENT ON COLUMN gl_eudr_irc_fpic_documents.consent_status IS 'Consent determination: granted, denied, conditional (with stipulations), pending (in process), withdrawn (revoked), expired';


-- ============================================================================
-- 3. gl_eudr_irc_land_rights_overlaps — Plot-territory overlap results
-- ============================================================================
RAISE NOTICE 'V109 [3/15]: Creating gl_eudr_irc_land_rights_overlaps...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_land_rights_overlaps (
    overlap_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot being analyzed
    territory_id                UUID            NOT NULL REFERENCES gl_eudr_irc_indigenous_territories(territory_id),
        -- Indigenous territory that overlaps with the plot
    overlap_type                VARCHAR(50)     NOT NULL,
        -- Classification of the overlap relationship
    overlap_geometry            GEOMETRY(POLYGON, 4326),
        -- PostGIS polygon representing the actual intersection area (WGS-84)
    overlap_area_hectares       NUMERIC(12,4)   CHECK (overlap_area_hectares >= 0),
        -- Area of spatial intersection in hectares
    overlap_percentage          NUMERIC(5,2)    CHECK (overlap_percentage >= 0 AND overlap_percentage <= 100),
        -- Percentage of the plot area that overlaps with the territory
    risk_level                  VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Risk classification based on overlap characteristics
    risk_score                  NUMERIC(5,2)    CHECK (risk_score >= 0 AND risk_score <= 100),
        -- Numeric risk score (0-100) for ranking and prioritization
    detected_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the overlap was first detected
    resolved                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the overlap has been resolved (FPIC obtained, etc.)
    resolved_at                 TIMESTAMPTZ,
        -- Timestamp when the overlap was resolved
    tenant_id                   UUID            NOT NULL,

    CONSTRAINT chk_irc_overlap_type CHECK (overlap_type IN (
        'full_containment', 'partial_overlap', 'boundary_adjacent',
        'buffer_zone', 'corridor', 'enclave'
    )),
    CONSTRAINT chk_irc_overlap_risk CHECK (risk_level IN (
        'critical', 'high', 'medium', 'low', 'informational'
    ))
);

-- GIST spatial index on overlap geometry
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_geometry ON gl_eudr_irc_land_rights_overlaps USING GIST (overlap_geometry);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_plot ON gl_eudr_irc_land_rights_overlaps (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_territory ON gl_eudr_irc_land_rights_overlaps (territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_type ON gl_eudr_irc_land_rights_overlaps (overlap_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_risk ON gl_eudr_irc_land_rights_overlaps (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_score ON gl_eudr_irc_land_rights_overlaps (risk_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_detected ON gl_eudr_irc_land_rights_overlaps (detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_resolved ON gl_eudr_irc_land_rights_overlaps (resolved);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_tenant ON gl_eudr_irc_land_rights_overlaps (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_plot_territory ON gl_eudr_irc_land_rights_overlaps (plot_id, territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unresolved overlaps requiring action
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ovlp_unresolved ON gl_eudr_irc_land_rights_overlaps (risk_level, risk_score DESC)
        WHERE resolved = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_land_rights_overlaps IS 'Spatial overlap analysis results between production plots and indigenous territories with risk scoring and resolution tracking';
COMMENT ON COLUMN gl_eudr_irc_land_rights_overlaps.overlap_type IS 'Overlap classification: full_containment (plot entirely within territory), partial_overlap, boundary_adjacent (within buffer), buffer_zone, corridor (transit route), enclave (territory within plot)';
COMMENT ON COLUMN gl_eudr_irc_land_rights_overlaps.risk_score IS 'Numeric risk score (0-100) derived from overlap percentage, territory legal status, and FPIC availability';


-- ============================================================================
-- 4. gl_eudr_irc_community_consultations — Consultation activity tracking
-- ============================================================================
RAISE NOTICE 'V109 [4/15]: Creating gl_eudr_irc_community_consultations...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_community_consultations (
    consultation_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot associated with the consultation
    community_id                UUID,
        -- Reference to the community being consulted (nullable for initial consultations)
    consultation_type           VARCHAR(50)     NOT NULL,
        -- Type of consultation activity
    consultation_date           DATE            NOT NULL,
        -- Date the consultation took place
    participants                JSONB           DEFAULT '[]',
        -- [{ "name": "...", "role": "elder", "organization": "..." }, ...]
    good_faith_score            NUMERIC(5,2)    CHECK (good_faith_score >= 0 AND good_faith_score <= 100),
        -- Assessment of consultation quality and good faith (0-100)
    outcome                     VARCHAR(100),
        -- Consultation outcome summary
    documentation_url           TEXT,
        -- URL or storage path for consultation records
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_irc_consult_type CHECK (consultation_type IN (
        'initial_engagement', 'information_sharing', 'impact_discussion',
        'consent_negotiation', 'benefit_sharing', 'follow_up',
        'grievance_hearing', 'periodic_review'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_consult_plot ON gl_eudr_irc_community_consultations (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_consult_community ON gl_eudr_irc_community_consultations (community_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_consult_type ON gl_eudr_irc_community_consultations (consultation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_consult_date ON gl_eudr_irc_community_consultations (consultation_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_consult_gf_score ON gl_eudr_irc_community_consultations (good_faith_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_consult_tenant ON gl_eudr_irc_community_consultations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_consult_created ON gl_eudr_irc_community_consultations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_consult_participants ON gl_eudr_irc_community_consultations USING GIN (participants);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_community_consultations IS 'Tracking of community consultation activities including engagement type, participants, good-faith scoring, and outcomes';
COMMENT ON COLUMN gl_eudr_irc_community_consultations.good_faith_score IS 'Good faith assessment (0-100) evaluating transparency, inclusivity, timing, language accessibility, and community agency';


-- ============================================================================
-- 5. gl_eudr_irc_rights_violations — Detected violations
-- ============================================================================
RAISE NOTICE 'V109 [5/15]: Creating gl_eudr_irc_rights_violations...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_rights_violations (
    violation_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot where the violation occurred
    territory_id                UUID            NOT NULL REFERENCES gl_eudr_irc_indigenous_territories(territory_id),
        -- Affected indigenous territory
    violation_type              VARCHAR(100)    NOT NULL,
        -- Classification of the rights violation
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Severity classification
    severity_score              NUMERIC(5,2)    CHECK (severity_score >= 0 AND severity_score <= 100),
        -- Numeric severity score for ranking (0-100)
    description                 TEXT,
        -- Detailed description of the violation
    evidence                    JSONB           DEFAULT '[]',
        -- [{ "type": "satellite_imagery", "url": "...", "date": "..." }, ...]
    detected_date               DATE            NOT NULL,
        -- Date the violation was detected
    status                      VARCHAR(50)     NOT NULL DEFAULT 'open',
        -- Current violation handling status
    resolution                  TEXT,
        -- Description of the resolution or remediation applied
    resolved_at                 TIMESTAMPTZ,
        -- Timestamp when the violation was resolved
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_irc_viol_type CHECK (violation_type IN (
        'unauthorized_access', 'fpic_not_obtained', 'consent_violated',
        'land_encroachment', 'resource_extraction', 'environmental_damage',
        'cultural_site_desecration', 'forced_displacement',
        'inadequate_consultation', 'benefit_sharing_breach'
    )),
    CONSTRAINT chk_irc_viol_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low', 'informational'
    )),
    CONSTRAINT chk_irc_viol_status CHECK (status IN (
        'open', 'investigating', 'confirmed', 'remediation_in_progress',
        'resolved', 'dismissed', 'escalated'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_plot ON gl_eudr_irc_rights_violations (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_territory ON gl_eudr_irc_rights_violations (territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_type ON gl_eudr_irc_rights_violations (violation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_severity ON gl_eudr_irc_rights_violations (severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_score ON gl_eudr_irc_rights_violations (severity_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_status ON gl_eudr_irc_rights_violations (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_detected ON gl_eudr_irc_rights_violations (detected_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_tenant ON gl_eudr_irc_rights_violations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_created ON gl_eudr_irc_rights_violations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_sev_status ON gl_eudr_irc_rights_violations (severity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for open/active violations
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_active ON gl_eudr_irc_rights_violations (severity, severity_score DESC)
        WHERE status IN ('open', 'investigating', 'confirmed', 'remediation_in_progress', 'escalated');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_viol_evidence ON gl_eudr_irc_rights_violations USING GIN (evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_rights_violations IS 'Detected indigenous rights violations linked to production plots and territories with severity scoring, evidence, and resolution tracking';
COMMENT ON COLUMN gl_eudr_irc_rights_violations.violation_type IS 'Violation classification per ILO Convention 169 and UNDRIP articles';
COMMENT ON COLUMN gl_eudr_irc_rights_violations.severity_score IS 'Numeric severity score (0-100) derived from violation type, affected population, area impact, and recurrence';


-- ============================================================================
-- 6. gl_eudr_irc_indigenous_communities — Community registry
-- ============================================================================
RAISE NOTICE 'V109 [6/15]: Creating gl_eudr_irc_indigenous_communities...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_indigenous_communities (
    community_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    community_name              VARCHAR(500)    NOT NULL,
        -- Official or commonly recognized community name
    indigenous_group            VARCHAR(500)    NOT NULL,
        -- Name of the broader indigenous people or ethnic group
    territory_id                UUID            REFERENCES gl_eudr_irc_indigenous_territories(territory_id),
        -- Territory this community is associated with (nullable for unregistered territories)
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    contact_person              VARCHAR(200),
        -- Name of the community representative or liaison
    contact_email               VARCHAR(200),
        -- Email address for the community contact
    contact_phone               VARCHAR(50),
        -- Phone number for the community contact
    population                  INTEGER         CHECK (population >= 0),
        -- Estimated community population
    metadata                    JSONB           DEFAULT '{}',
        -- Additional community attributes (governance_structure, languages, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_comm_name ON gl_eudr_irc_indigenous_communities (community_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_comm_group ON gl_eudr_irc_indigenous_communities (indigenous_group);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_comm_territory ON gl_eudr_irc_indigenous_communities (territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_comm_country ON gl_eudr_irc_indigenous_communities (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_comm_population ON gl_eudr_irc_indigenous_communities (population DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_comm_tenant ON gl_eudr_irc_indigenous_communities (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_comm_created ON gl_eudr_irc_indigenous_communities (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_comm_metadata ON gl_eudr_irc_indigenous_communities USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_indigenous_communities IS 'Registry of indigenous communities with contact information, population, and territory linkage';
COMMENT ON COLUMN gl_eudr_irc_indigenous_communities.metadata IS 'Additional attributes: governance_structure, primary_languages, recognition_documents, etc.';


-- ============================================================================
-- 7. gl_eudr_irc_fpic_workflows — Workflow state tracking
-- ============================================================================
RAISE NOTICE 'V109 [7/15]: Creating gl_eudr_irc_fpic_workflows...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_fpic_workflows (
    workflow_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot requiring FPIC process
    territory_id                UUID            NOT NULL REFERENCES gl_eudr_irc_indigenous_territories(territory_id),
        -- Territory from which consent is sought
    current_state               VARCHAR(50)     NOT NULL DEFAULT 'initiated',
        -- Current workflow state
    required_steps              JSONB           DEFAULT '[]',
        -- [{ "step": "initial_contact", "required": true, "deadline": "2026-04-01" }, ...]
    completed_steps             JSONB           DEFAULT '[]',
        -- [{ "step": "initial_contact", "completed_at": "2026-03-15", "actor": "..." }, ...]
    compliance_status           VARCHAR(50)     NOT NULL DEFAULT 'in_progress',
        -- Overall workflow compliance determination
    started_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Workflow initiation timestamp
    completed_at                TIMESTAMPTZ,
        -- Workflow completion timestamp (null if in progress)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_irc_wf_state CHECK (current_state IN (
        'initiated', 'contact_established', 'information_shared',
        'consultation_scheduled', 'consultation_in_progress',
        'consent_requested', 'consent_granted', 'consent_denied',
        'consent_conditional', 'benefit_negotiation', 'completed',
        'withdrawn', 'suspended'
    )),
    CONSTRAINT chk_irc_wf_compliance CHECK (compliance_status IN (
        'in_progress', 'compliant', 'non_compliant', 'conditional',
        'expired', 'under_review'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_plot ON gl_eudr_irc_fpic_workflows (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_territory ON gl_eudr_irc_fpic_workflows (territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_state ON gl_eudr_irc_fpic_workflows (current_state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_compliance ON gl_eudr_irc_fpic_workflows (compliance_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_started ON gl_eudr_irc_fpic_workflows (started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_tenant ON gl_eudr_irc_fpic_workflows (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_created ON gl_eudr_irc_fpic_workflows (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_plot_territory ON gl_eudr_irc_fpic_workflows (plot_id, territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active workflows
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_active ON gl_eudr_irc_fpic_workflows (current_state, started_at DESC)
        WHERE completed_at IS NULL AND compliance_status = 'in_progress';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_req_steps ON gl_eudr_irc_fpic_workflows USING GIN (required_steps);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_wf_comp_steps ON gl_eudr_irc_fpic_workflows USING GIN (completed_steps);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_fpic_workflows IS 'FPIC workflow state machine tracking required steps, completion status, and compliance determination per plot-territory pair';
COMMENT ON COLUMN gl_eudr_irc_fpic_workflows.current_state IS 'Workflow state from initiation through consultation to consent determination or withdrawal';
COMMENT ON COLUMN gl_eudr_irc_fpic_workflows.compliance_status IS 'Overall FPIC compliance: in_progress, compliant (consent obtained), non_compliant (denied/missing), conditional, expired, under_review';


-- ============================================================================
-- 8. gl_eudr_irc_consultation_good_faith_scores — Consultation scoring
-- ============================================================================
RAISE NOTICE 'V109 [8/15]: Creating gl_eudr_irc_consultation_good_faith_scores...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_consultation_good_faith_scores (
    score_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    consultation_id             UUID            NOT NULL REFERENCES gl_eudr_irc_community_consultations(consultation_id),
        -- Reference to the consultation being scored
    score_components            JSONB           NOT NULL DEFAULT '{}',
        -- { "transparency": 85, "inclusivity": 90, "timing": 75,
        --   "language_accessibility": 80, "community_agency": 88,
        --   "information_adequacy": 92, "cultural_respect": 95 }
    composite_score             NUMERIC(5,2)    NOT NULL CHECK (composite_score >= 0 AND composite_score <= 100),
        -- Weighted average of all score components (0-100)
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the score was calculated
    tenant_id                   UUID            NOT NULL
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_gf_consultation ON gl_eudr_irc_consultation_good_faith_scores (consultation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_gf_composite ON gl_eudr_irc_consultation_good_faith_scores (composite_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_gf_calculated ON gl_eudr_irc_consultation_good_faith_scores (calculated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_gf_tenant ON gl_eudr_irc_consultation_good_faith_scores (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_gf_components ON gl_eudr_irc_consultation_good_faith_scores USING GIN (score_components);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_consultation_good_faith_scores IS 'Good-faith scoring of community consultations across transparency, inclusivity, timing, accessibility, and cultural respect dimensions';
COMMENT ON COLUMN gl_eudr_irc_consultation_good_faith_scores.composite_score IS 'Weighted composite score (0-100): threshold of 70 indicates adequate good faith per FPIC best practices';


-- ============================================================================
-- 9. gl_eudr_irc_territory_data_sources — Data source metadata
-- ============================================================================
RAISE NOTICE 'V109 [9/15]: Creating gl_eudr_irc_territory_data_sources...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_territory_data_sources (
    source_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name                 VARCHAR(200)    NOT NULL,
        -- Data source name (e.g., 'LandMark', 'RAISG', 'FUNAI', 'IWGIA')
    source_url                  TEXT,
        -- URL to the data source
    geographic_coverage         JSONB           DEFAULT '{}',
        -- { "regions": ["South America", "Southeast Asia"], "countries": ["BR", "ID", "PE"] }
    update_frequency            VARCHAR(50),
        -- How often the source is updated
    last_updated                DATE,
        -- Date the source was last refreshed
    reliability_score           NUMERIC(3,2)    CHECK (reliability_score >= 0 AND reliability_score <= 1),
        -- Source reliability rating (0.0 = unreliable, 1.0 = authoritative)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_irc_ds_frequency CHECK (update_frequency IN (
        'real_time', 'daily', 'weekly', 'monthly', 'quarterly',
        'semi_annual', 'annual', 'irregular', 'static'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ds_name ON gl_eudr_irc_territory_data_sources (source_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ds_frequency ON gl_eudr_irc_territory_data_sources (update_frequency);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ds_last_updated ON gl_eudr_irc_territory_data_sources (last_updated DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ds_reliability ON gl_eudr_irc_territory_data_sources (reliability_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ds_tenant ON gl_eudr_irc_territory_data_sources (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_ds_coverage ON gl_eudr_irc_territory_data_sources USING GIN (geographic_coverage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_territory_data_sources IS 'Metadata registry for indigenous territory data sources with geographic coverage, update frequency, and reliability scoring';
COMMENT ON COLUMN gl_eudr_irc_territory_data_sources.reliability_score IS 'Source reliability: 0.0 = unreliable/unverified, 1.0 = authoritative government or international body data';


-- ============================================================================
-- 10. gl_eudr_irc_country_fpic_requirements — Per-country FPIC legal frameworks
-- ============================================================================
RAISE NOTICE 'V109 [10/15]: Creating gl_eudr_irc_country_fpic_requirements...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_country_fpic_requirements (
    country_code                CHAR(2)         PRIMARY KEY,
        -- ISO 3166-1 alpha-2 country code
    ilo_169_ratified            BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the country has ratified ILO Convention 169
    ratification_date           DATE,
        -- Date of ILO 169 ratification (null if not ratified)
    fpic_in_national_law        BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether FPIC is enshrined in national legislation
    legal_framework             TEXT,
        -- Description of the applicable national/regional legal framework
    enforcement_level           VARCHAR(50)     NOT NULL DEFAULT 'weak',
        -- Assessment of enforcement effectiveness
    metadata                    JSONB           DEFAULT '{}',
        -- Additional legal context (constitutional provisions, case law, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_irc_cfr_enforcement CHECK (enforcement_level IN (
        'strong', 'moderate', 'weak', 'none', 'unknown'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_cfr_ilo169 ON gl_eudr_irc_country_fpic_requirements (ilo_169_ratified);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_cfr_fpic_law ON gl_eudr_irc_country_fpic_requirements (fpic_in_national_law);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_cfr_enforcement ON gl_eudr_irc_country_fpic_requirements (enforcement_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_cfr_tenant ON gl_eudr_irc_country_fpic_requirements (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for countries with strong FPIC frameworks
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_cfr_strong ON gl_eudr_irc_country_fpic_requirements (country_code)
        WHERE ilo_169_ratified = TRUE AND fpic_in_national_law = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_cfr_metadata ON gl_eudr_irc_country_fpic_requirements USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_country_fpic_requirements IS 'Per-country FPIC legal framework registry including ILO 169 ratification status, national law provisions, and enforcement assessment';
COMMENT ON COLUMN gl_eudr_irc_country_fpic_requirements.enforcement_level IS 'Enforcement effectiveness: strong (active judicial enforcement), moderate (sporadic), weak (laws exist but rarely enforced), none, unknown';


-- ============================================================================
-- 11. gl_eudr_irc_compliance_reports — Generated compliance reports
-- ============================================================================
RAISE NOTICE 'V109 [11/15]: Creating gl_eudr_irc_compliance_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_compliance_reports (
    report_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot this report covers
    report_type                 VARCHAR(50)     NOT NULL,
        -- Type of compliance report
    compliance_status           VARCHAR(50)     NOT NULL,
        -- Overall compliance determination
    findings                    JSONB           DEFAULT '[]',
        -- [{ "finding": "...", "severity": "high", "reference": "EUDR Art.3(b)" }, ...]
    recommendations             JSONB           DEFAULT '[]',
        -- [{ "action": "...", "priority": "urgent", "deadline": "2026-06-01" }, ...]
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Report generation timestamp
    tenant_id                   UUID            NOT NULL,

    CONSTRAINT chk_irc_rpt_type CHECK (report_type IN (
        'pre_sourcing_assessment', 'due_diligence_report', 'annual_review',
        'incident_report', 'remediation_plan', 'audit_response'
    )),
    CONSTRAINT chk_irc_rpt_compliance CHECK (compliance_status IN (
        'compliant', 'non_compliant', 'partially_compliant',
        'under_review', 'remediation_required'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rpt_plot ON gl_eudr_irc_compliance_reports (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rpt_type ON gl_eudr_irc_compliance_reports (report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rpt_compliance ON gl_eudr_irc_compliance_reports (compliance_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rpt_generated ON gl_eudr_irc_compliance_reports (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rpt_tenant ON gl_eudr_irc_compliance_reports (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for non-compliant reports
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rpt_non_compliant ON gl_eudr_irc_compliance_reports (generated_at DESC)
        WHERE compliance_status IN ('non_compliant', 'remediation_required');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rpt_findings ON gl_eudr_irc_compliance_reports USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rpt_recommendations ON gl_eudr_irc_compliance_reports USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_compliance_reports IS 'Generated indigenous rights compliance reports with findings, recommendations, and compliance determination';
COMMENT ON COLUMN gl_eudr_irc_compliance_reports.compliance_status IS 'compliant (FPIC obtained, no violations), non_compliant (missing FPIC or unresolved violations), partially_compliant (FPIC conditional or in progress), under_review, remediation_required';


-- ============================================================================
-- 12. gl_eudr_irc_territory_plot_associations — Many-to-many plot-territory links
-- ============================================================================
RAISE NOTICE 'V109 [12/15]: Creating gl_eudr_irc_territory_plot_associations...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_territory_plot_associations (
    association_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot
    territory_id                UUID            NOT NULL REFERENCES gl_eudr_irc_indigenous_territories(territory_id),
        -- Indigenous territory
    association_type            VARCHAR(50)     NOT NULL,
        -- Nature of the association
    intersection_geometry       GEOMETRY(POLYGON, 4326),
        -- PostGIS polygon of the spatial intersection between plot and territory (WGS-84)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_irc_assoc_type CHECK (association_type IN (
        'spatial_overlap', 'historical_claim', 'cultural_significance',
        'resource_dependency', 'transit_route', 'buffer_zone'
    )),
    CONSTRAINT uq_irc_plot_territory_type UNIQUE (plot_id, territory_id, association_type)
);

-- GIST spatial index on intersection geometry
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_assoc_geometry ON gl_eudr_irc_territory_plot_associations USING GIST (intersection_geometry);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_assoc_plot ON gl_eudr_irc_territory_plot_associations (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_assoc_territory ON gl_eudr_irc_territory_plot_associations (territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_assoc_type ON gl_eudr_irc_territory_plot_associations (association_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_assoc_tenant ON gl_eudr_irc_territory_plot_associations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_assoc_created ON gl_eudr_irc_territory_plot_associations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_assoc_plot_territory ON gl_eudr_irc_territory_plot_associations (plot_id, territory_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_territory_plot_associations IS 'Many-to-many linkage between production plots and indigenous territories with association type classification';
COMMENT ON COLUMN gl_eudr_irc_territory_plot_associations.association_type IS 'Association nature: spatial_overlap (geometric intersection), historical_claim, cultural_significance, resource_dependency, transit_route, buffer_zone';


-- ============================================================================
-- 13. gl_eudr_irc_fpic_verification_log — FPIC verification audit (hypertable)
-- ============================================================================
RAISE NOTICE 'V109 [13/15]: Creating gl_eudr_irc_fpic_verification_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_fpic_verification_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    verified_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the verification event (partition key)
    document_id                 UUID            NOT NULL,
        -- Reference to the FPIC document being verified
    verification_action         VARCHAR(50)     NOT NULL,
        -- Type of verification action performed
    previous_status             VARCHAR(50),
        -- Document verification status before this action
    new_status                  VARCHAR(50)     NOT NULL,
        -- Document verification status after this action
    verifier                    VARCHAR(200)    NOT NULL,
        -- User or system agent performing the verification
    verification_method         VARCHAR(50),
        -- Method used for verification
    confidence                  NUMERIC(3,2)    CHECK (confidence >= 0 AND confidence <= 1),
        -- Verification confidence score (0.0 to 1.0)
    notes                       TEXT,
        -- Notes accompanying the verification action
    evidence                    JSONB           DEFAULT '{}',
        -- Supporting evidence for the verification
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for audit trail integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, verified_at),

    CONSTRAINT chk_irc_fvl_action CHECK (verification_action IN (
        'initial_review', 'document_check', 'authority_confirmation',
        'field_verification', 'community_confirmation', 'expiry_check',
        'periodic_revalidation', 'rejection', 'reinstatement'
    )),
    CONSTRAINT chk_irc_fvl_method CHECK (verification_method IS NULL OR verification_method IN (
        'manual_review', 'automated_check', 'field_visit',
        'third_party_audit', 'community_attestation', 'blockchain_verification'
    ))
);

SELECT create_hypertable(
    'gl_eudr_irc_fpic_verification_log',
    'verified_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_document ON gl_eudr_irc_fpic_verification_log (document_id, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_action ON gl_eudr_irc_fpic_verification_log (verification_action, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_new_status ON gl_eudr_irc_fpic_verification_log (new_status, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_verifier ON gl_eudr_irc_fpic_verification_log (verifier, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_method ON gl_eudr_irc_fpic_verification_log (verification_method, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_confidence ON gl_eudr_irc_fpic_verification_log (confidence DESC, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_provenance ON gl_eudr_irc_fpic_verification_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_tenant ON gl_eudr_irc_fpic_verification_log (tenant_id, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_fvl_evidence ON gl_eudr_irc_fpic_verification_log USING GIN (evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_fpic_verification_log IS 'Immutable audit log for all FPIC document verification events with action, method, confidence, and provenance tracking';
COMMENT ON COLUMN gl_eudr_irc_fpic_verification_log.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- 14. gl_eudr_irc_overlap_analysis_log — Spatial overlap analysis audit (hypertable)
-- ============================================================================
RAISE NOTICE 'V109 [14/15]: Creating gl_eudr_irc_overlap_analysis_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_overlap_analysis_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    analyzed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the analysis event (partition key)
    plot_id                     UUID            NOT NULL,
        -- Production plot analyzed
    territories_checked         INTEGER         NOT NULL DEFAULT 0,
        -- Number of territories checked in this analysis
    overlaps_found              INTEGER         NOT NULL DEFAULT 0,
        -- Number of overlaps detected
    analysis_method             VARCHAR(50)     NOT NULL,
        -- Spatial analysis method used
    analysis_parameters         JSONB           DEFAULT '{}',
        -- { "buffer_m": 500, "min_overlap_pct": 0.01, "crs": "EPSG:4326" }
    execution_time_ms           INTEGER,
        -- Analysis execution time in milliseconds
    result_summary              JSONB           DEFAULT '{}',
        -- { "total_overlap_ha": 45.2, "max_overlap_pct": 23.5, "risk_levels": {...} }
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for analysis integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, analyzed_at),

    CONSTRAINT chk_irc_oal_method CHECK (analysis_method IN (
        'st_intersects', 'st_overlaps', 'st_contains', 'st_within',
        'buffer_analysis', 'proximity_analysis', 'composite'
    ))
);

SELECT create_hypertable(
    'gl_eudr_irc_overlap_analysis_log',
    'analyzed_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_oal_plot ON gl_eudr_irc_overlap_analysis_log (plot_id, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_oal_method ON gl_eudr_irc_overlap_analysis_log (analysis_method, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_oal_overlaps ON gl_eudr_irc_overlap_analysis_log (overlaps_found, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_oal_exec_time ON gl_eudr_irc_overlap_analysis_log (execution_time_ms DESC, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_oal_provenance ON gl_eudr_irc_overlap_analysis_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_oal_tenant ON gl_eudr_irc_overlap_analysis_log (tenant_id, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_oal_params ON gl_eudr_irc_overlap_analysis_log USING GIN (analysis_parameters);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_oal_summary ON gl_eudr_irc_overlap_analysis_log USING GIN (result_summary);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_overlap_analysis_log IS 'Immutable audit log for spatial overlap analysis runs with method, parameters, execution time, and result summaries';
COMMENT ON COLUMN gl_eudr_irc_overlap_analysis_log.analysis_method IS 'PostGIS spatial function used: st_intersects, st_overlaps, st_contains, st_within, buffer_analysis, proximity_analysis, composite';


-- ============================================================================
-- 15. gl_eudr_irc_rights_violation_events — Violation event stream (hypertable)
-- ============================================================================
RAISE NOTICE 'V109 [15/15]: Creating gl_eudr_irc_rights_violation_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_irc_rights_violation_events (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    event_timestamp             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)
    violation_id                UUID            NOT NULL,
        -- Reference to the parent violation record
    event_type                  VARCHAR(50)     NOT NULL,
        -- Type of violation lifecycle event
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

    CONSTRAINT chk_irc_rve_event_type CHECK (event_type IN (
        'detected', 'acknowledged', 'investigation_started',
        'evidence_added', 'severity_updated', 'escalated',
        'remediation_started', 'remediation_completed', 'resolved',
        'dismissed', 'reopened', 'comment_added'
    ))
);

SELECT create_hypertable(
    'gl_eudr_irc_rights_violation_events',
    'event_timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rve_violation ON gl_eudr_irc_rights_violation_events (violation_id, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rve_type ON gl_eudr_irc_rights_violation_events (event_type, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rve_actor ON gl_eudr_irc_rights_violation_events (actor, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rve_new_status ON gl_eudr_irc_rights_violation_events (new_status, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rve_provenance ON gl_eudr_irc_rights_violation_events (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rve_tenant ON gl_eudr_irc_rights_violation_events (tenant_id, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_irc_rve_details ON gl_eudr_irc_rights_violation_events USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_irc_rights_violation_events IS 'Immutable event stream for rights violation lifecycle tracking with actor, status transitions, and provenance hashing';
COMMENT ON COLUMN gl_eudr_irc_rights_violation_events.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Daily violation counts by severity
RAISE NOTICE 'V109: Creating continuous aggregate: daily_violation_counts...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_irc_daily_violation_counts
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', event_timestamp)   AS day,
        tenant_id,
        new_status,
        COUNT(*)                                AS event_count,
        COUNT(DISTINCT violation_id)            AS unique_violations
    FROM gl_eudr_irc_rights_violation_events
    GROUP BY day, tenant_id, new_status;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_irc_daily_violation_counts',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_irc_daily_violation_counts IS 'Daily rollup of rights violation events by status with event and unique violation counts';


-- Monthly FPIC compliance rates by country
RAISE NOTICE 'V109: Creating continuous aggregate: monthly_fpic_compliance...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_irc_monthly_fpic_compliance
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('30 days', verified_at)     AS month,
        tenant_id,
        verification_action,
        new_status,
        COUNT(*)                                AS verification_count,
        AVG(confidence)                         AS avg_confidence,
        COUNT(DISTINCT document_id)             AS unique_documents
    FROM gl_eudr_irc_fpic_verification_log
    GROUP BY month, tenant_id, verification_action, new_status;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_irc_monthly_fpic_compliance',
        start_offset => INTERVAL '60 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_irc_monthly_fpic_compliance IS 'Monthly rollup of FPIC verification events by action and status with confidence statistics';


-- ============================================================================
-- RETENTION POLICIES (5 years per EUDR Article 31)
-- ============================================================================

RAISE NOTICE 'V109: Creating retention policies (5 years per EUDR Article 31)...';

-- 5 years for FPIC verification logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_irc_fpic_verification_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for overlap analysis logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_irc_overlap_analysis_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for rights violation events
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_irc_rights_violation_events', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- GRANTS — greenlang_app role
-- ============================================================================

RAISE NOTICE 'V109: Granting permissions to greenlang_app...';

-- Regular tables
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_indigenous_territories TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_fpic_documents TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_land_rights_overlaps TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_community_consultations TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_rights_violations TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_indigenous_communities TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_fpic_workflows TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_consultation_good_faith_scores TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_territory_data_sources TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_country_fpic_requirements TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_compliance_reports TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_irc_territory_plot_associations TO greenlang_app;

-- Hypertables
GRANT SELECT, INSERT ON gl_eudr_irc_fpic_verification_log TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_irc_overlap_analysis_log TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_irc_rights_violation_events TO greenlang_app;

-- Continuous aggregates
GRANT SELECT ON gl_eudr_irc_daily_violation_counts TO greenlang_app;
GRANT SELECT ON gl_eudr_irc_monthly_fpic_compliance TO greenlang_app;

-- Read-only role (conditional)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT SELECT ON gl_eudr_irc_indigenous_territories TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_fpic_documents TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_land_rights_overlaps TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_community_consultations TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_rights_violations TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_indigenous_communities TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_fpic_workflows TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_consultation_good_faith_scores TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_territory_data_sources TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_country_fpic_requirements TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_compliance_reports TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_territory_plot_associations TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_fpic_verification_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_overlap_analysis_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_rights_violation_events TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_daily_violation_counts TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_irc_monthly_fpic_compliance TO greenlang_readonly;
    END IF;
END
$$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V109: AGENT-EUDR-021 Indigenous Rights Checker tables created successfully!';
RAISE NOTICE 'V109: Created 15 tables (12 regular + 3 hypertables), 2 continuous aggregates, ~131 indexes';
RAISE NOTICE 'V109: 3 GIST spatial indexes, 16 GIN indexes on JSONB, 7 partial indexes for active records';
RAISE NOTICE 'V109: Retention policies: 5y on all hypertables per EUDR Article 31';
RAISE NOTICE 'V109: Grants applied for greenlang_app and greenlang_readonly roles';

COMMIT;

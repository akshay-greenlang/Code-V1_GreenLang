-- =============================================================================
-- V177: PACK-027 Enterprise Net Zero - Assurance Records
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    012 of 015
-- Date:         March 2026
--
-- External assurance engagement management per ISO 14064-3, ISAE 3410, and
-- ISAE 3000. Tracks engagements (limited/reasonable), workpapers, evidence
-- packages, findings, and management responses for Big 4 audit readiness.
--
-- Tables (2):
--   1. pack027_enterprise_net_zero.gl_assurance_engagements
--   2. pack027_enterprise_net_zero.gl_assurance_workpapers
--
-- Previous: V176__PACK027_regulatory_compliance.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_assurance_engagements
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_assurance_engagements (
    engagement_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Provider
    provider                    VARCHAR(255)    NOT NULL,
    provider_type               VARCHAR(30)     DEFAULT 'BIG_4',
    lead_partner                VARCHAR(255),
    engagement_team_size        INTEGER,
    -- Engagement scope
    assurance_level             VARCHAR(30)     NOT NULL,
    assurance_standard          VARCHAR(50)     NOT NULL,
    scope                       TEXT            NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    reporting_period_start      DATE,
    reporting_period_end        DATE,
    -- Coverage
    scope1_in_scope             BOOLEAN         DEFAULT TRUE,
    scope2_in_scope             BOOLEAN         DEFAULT TRUE,
    scope3_in_scope             BOOLEAN         DEFAULT FALSE,
    scope3_categories           TEXT[]          DEFAULT '{}',
    entities_in_scope           TEXT[]          DEFAULT '{}',
    entity_coverage_pct         DECIMAL(6,2),
    emissions_coverage_pct      DECIMAL(6,2),
    -- Materiality
    materiality_threshold_pct   DECIMAL(6,2),
    materiality_threshold_tco2e DECIMAL(18,4),
    -- Timeline
    engagement_start_date       DATE,
    planning_complete_date      DATE,
    fieldwork_start_date        DATE,
    fieldwork_end_date          DATE,
    report_date                 DATE,
    -- Opinion
    opinion                     VARCHAR(30),
    opinion_basis               TEXT,
    modifications               TEXT,
    emphasis_of_matter          TEXT,
    -- Findings
    findings_count              INTEGER         DEFAULT 0,
    material_findings           INTEGER         DEFAULT 0,
    immaterial_findings         INTEGER         DEFAULT 0,
    observations                INTEGER         DEFAULT 0,
    recommendations             JSONB           DEFAULT '{}',
    -- Management response
    management_response         TEXT,
    remediation_committed       BOOLEAN         DEFAULT FALSE,
    remediation_deadline        DATE,
    -- Fees
    engagement_fee_usd          DECIMAL(18,2),
    actual_hours                DECIMAL(8,1),
    budgeted_hours              DECIMAL(8,1),
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PLANNED',
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_ae_assurance_level CHECK (
        assurance_level IN ('LIMITED', 'REASONABLE', 'COMBINED')
    ),
    CONSTRAINT chk_p027_ae_standard CHECK (
        assurance_standard IN ('ISAE_3410', 'ISAE_3000', 'ISO_14064_3', 'AA1000AS',
                                'PCAF_STANDARD', 'AICPA_AT_C_105', 'OTHER')
    ),
    CONSTRAINT chk_p027_ae_provider_type CHECK (
        provider_type IN ('BIG_4', 'NATIONAL_FIRM', 'SPECIALIST', 'CERTIFICATION_BODY', 'OTHER')
    ),
    CONSTRAINT chk_p027_ae_reporting_year CHECK (
        reporting_year >= 2020 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p027_ae_opinion CHECK (
        opinion IS NULL OR opinion IN (
            'UNMODIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER', 'PENDING'
        )
    ),
    CONSTRAINT chk_p027_ae_status CHECK (
        status IN ('PLANNED', 'ENGAGED', 'PLANNING', 'FIELDWORK', 'REVIEW',
                    'REPORTING', 'COMPLETED', 'CANCELLED')
    ),
    CONSTRAINT chk_p027_ae_coverage CHECK (
        (entity_coverage_pct IS NULL OR (entity_coverage_pct >= 0 AND entity_coverage_pct <= 100)) AND
        (emissions_coverage_pct IS NULL OR (emissions_coverage_pct >= 0 AND emissions_coverage_pct <= 100))
    ),
    CONSTRAINT chk_p027_ae_materiality CHECK (
        materiality_threshold_pct IS NULL OR (materiality_threshold_pct >= 0 AND materiality_threshold_pct <= 100)
    )
);

-- =============================================================================
-- Table 2: pack027_enterprise_net_zero.gl_assurance_workpapers
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_assurance_workpapers (
    workpaper_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    engagement_id               UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_assurance_engagements(engagement_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Workpaper identification
    workpaper_ref               VARCHAR(50)     NOT NULL,
    workpaper_type              VARCHAR(50)     NOT NULL,
    workpaper_title             VARCHAR(500)    NOT NULL,
    description                 TEXT,
    -- Scope reference
    scope_category              VARCHAR(30),
    entity_ref                  UUID,
    -- Evidence
    evidence_path               TEXT,
    evidence_hash               VARCHAR(64),
    evidence_type               VARCHAR(30),
    evidence_date               DATE,
    -- Content
    source_data_summary         JSONB           DEFAULT '{}',
    calculation_trace           JSONB           DEFAULT '{}',
    control_evidence            JSONB           DEFAULT '{}',
    sample_selection            JSONB           DEFAULT '{}',
    -- Review
    review_status               VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    prepared_by                 VARCHAR(255),
    prepared_date               DATE,
    reviewed_by                 VARCHAR(255),
    reviewed_date               DATE,
    partner_reviewed            BOOLEAN         DEFAULT FALSE,
    partner_review_date         DATE,
    -- Findings from this workpaper
    findings_raised             INTEGER         DEFAULT 0,
    finding_references          TEXT[]          DEFAULT '{}',
    -- Cross-references
    cross_references            TEXT[]          DEFAULT '{}',
    related_workpapers          UUID[]          DEFAULT '{}',
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_aw_type CHECK (
        workpaper_type IN ('PLANNING_MEMO', 'RISK_ASSESSMENT', 'MATERIALITY_CALC',
                            'SCOPE1_TESTING', 'SCOPE2_TESTING', 'SCOPE3_TESTING',
                            'DATA_ANALYTICS', 'SAMPLE_TESTING', 'RECALCULATION',
                            'CONTROL_TESTING', 'METHODOLOGY_REVIEW', 'EMISSION_FACTOR_REVIEW',
                            'CONSOLIDATION_REVIEW', 'MANAGEMENT_REPRESENTATION',
                            'ENGAGEMENT_LETTER', 'COMPLETION_MEMO', 'OPINION_DRAFT',
                            'SUMMARY_OF_FINDINGS', 'OTHER')
    ),
    CONSTRAINT chk_p027_aw_scope CHECK (
        scope_category IS NULL OR scope_category IN (
            'SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3',
            'CONSOLIDATION', 'DATA_QUALITY', 'METHODOLOGY', 'OVERALL'
        )
    ),
    CONSTRAINT chk_p027_aw_evidence_type CHECK (
        evidence_type IS NULL OR evidence_type IN (
            'DOCUMENT', 'SPREADSHEET', 'SCREENSHOT', 'EMAIL', 'CERTIFICATE',
            'INVOICE', 'METER_READING', 'SYSTEM_EXPORT', 'PHOTO', 'OTHER'
        )
    ),
    CONSTRAINT chk_p027_aw_review_status CHECK (
        review_status IN ('DRAFT', 'PREPARED', 'REVIEWED', 'PARTNER_REVIEWED', 'FINAL', 'SUPERSEDED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for gl_assurance_engagements
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_ae_company            ON pack027_enterprise_net_zero.gl_assurance_engagements(company_id);
CREATE INDEX idx_p027_ae_tenant             ON pack027_enterprise_net_zero.gl_assurance_engagements(tenant_id);
CREATE INDEX idx_p027_ae_provider           ON pack027_enterprise_net_zero.gl_assurance_engagements(provider);
CREATE INDEX idx_p027_ae_level              ON pack027_enterprise_net_zero.gl_assurance_engagements(assurance_level);
CREATE INDEX idx_p027_ae_standard           ON pack027_enterprise_net_zero.gl_assurance_engagements(assurance_standard);
CREATE INDEX idx_p027_ae_year               ON pack027_enterprise_net_zero.gl_assurance_engagements(reporting_year);
CREATE INDEX idx_p027_ae_opinion            ON pack027_enterprise_net_zero.gl_assurance_engagements(opinion);
CREATE INDEX idx_p027_ae_status             ON pack027_enterprise_net_zero.gl_assurance_engagements(status);
CREATE INDEX idx_p027_ae_report_date        ON pack027_enterprise_net_zero.gl_assurance_engagements(report_date);
CREATE INDEX idx_p027_ae_created            ON pack027_enterprise_net_zero.gl_assurance_engagements(created_at DESC);
CREATE INDEX idx_p027_ae_metadata           ON pack027_enterprise_net_zero.gl_assurance_engagements USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Indexes for gl_assurance_workpapers
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_aw_engagement         ON pack027_enterprise_net_zero.gl_assurance_workpapers(engagement_id);
CREATE INDEX idx_p027_aw_tenant             ON pack027_enterprise_net_zero.gl_assurance_workpapers(tenant_id);
CREATE INDEX idx_p027_aw_type               ON pack027_enterprise_net_zero.gl_assurance_workpapers(workpaper_type);
CREATE INDEX idx_p027_aw_scope              ON pack027_enterprise_net_zero.gl_assurance_workpapers(scope_category);
CREATE INDEX idx_p027_aw_review_status      ON pack027_enterprise_net_zero.gl_assurance_workpapers(review_status);
CREATE INDEX idx_p027_aw_prepared_by        ON pack027_enterprise_net_zero.gl_assurance_workpapers(prepared_by);
CREATE INDEX idx_p027_aw_reviewed_by        ON pack027_enterprise_net_zero.gl_assurance_workpapers(reviewed_by);
CREATE INDEX idx_p027_aw_entity             ON pack027_enterprise_net_zero.gl_assurance_workpapers(entity_ref);
CREATE INDEX idx_p027_aw_evidence_hash      ON pack027_enterprise_net_zero.gl_assurance_workpapers(evidence_hash);
CREATE INDEX idx_p027_aw_findings           ON pack027_enterprise_net_zero.gl_assurance_workpapers(findings_raised) WHERE findings_raised > 0;
CREATE INDEX idx_p027_aw_created            ON pack027_enterprise_net_zero.gl_assurance_workpapers(created_at DESC);
CREATE INDEX idx_p027_aw_calc_trace         ON pack027_enterprise_net_zero.gl_assurance_workpapers USING GIN(calculation_trace);
CREATE INDEX idx_p027_aw_metadata           ON pack027_enterprise_net_zero.gl_assurance_workpapers USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_assurance_engagements_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_assurance_engagements
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p027_assurance_workpapers_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_assurance_workpapers
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_assurance_engagements ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack027_enterprise_net_zero.gl_assurance_workpapers ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_ae_tenant_isolation
    ON pack027_enterprise_net_zero.gl_assurance_engagements
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_ae_service_bypass
    ON pack027_enterprise_net_zero.gl_assurance_engagements
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p027_aw_tenant_isolation
    ON pack027_enterprise_net_zero.gl_assurance_workpapers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_aw_service_bypass
    ON pack027_enterprise_net_zero.gl_assurance_workpapers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_assurance_engagements TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_assurance_workpapers TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_assurance_engagements IS
    'External assurance engagement records per ISO 14064-3/ISAE 3410/ISAE 3000 with provider details, scope, opinion, findings, and management responses.';
COMMENT ON TABLE pack027_enterprise_net_zero.gl_assurance_workpapers IS
    'Assurance workpaper records with evidence references, calculation traces, control testing, sample selection, and multi-level review tracking.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_engagements.engagement_id IS 'Unique assurance engagement identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_engagements.assurance_level IS 'Assurance level: LIMITED, REASONABLE, or COMBINED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_engagements.opinion IS 'Auditor opinion: UNMODIFIED, QUALIFIED, ADVERSE, DISCLAIMER, or PENDING.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_engagements.scope IS 'Textual description of the engagement scope.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_engagements.report_date IS 'Date of the assurance report/opinion.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_workpapers.workpaper_id IS 'Unique workpaper identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_workpapers.workpaper_type IS 'Workpaper type: PLANNING_MEMO, RISK_ASSESSMENT, SCOPE1_TESTING, RECALCULATION, etc.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_workpapers.evidence_path IS 'Storage path to supporting evidence document.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_workpapers.review_status IS 'Review pipeline: DRAFT, PREPARED, REVIEWED, PARTNER_REVIEWED, FINAL, SUPERSEDED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_assurance_workpapers.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

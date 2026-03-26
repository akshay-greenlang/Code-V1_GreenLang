-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V403 - Regulatory Compliance
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates regulatory compliance tables for tracking jurisdiction-specific
-- assurance requirements. Jurisdictions define the regulatory framework,
-- required assurance standard and level, mandatory scopes, effective dates,
-- and transition periods. Requirements map specific obligations per config
-- and jurisdiction with applicability reasoning and compliance status.
-- Compliance status provides an aggregate view per config-jurisdiction pair.
--
-- Tables (3):
--   1. ghg_assurance.gl_ap_jurisdictions
--   2. ghg_assurance.gl_ap_requirements
--   3. ghg_assurance.gl_ap_compliance_status
--
-- Also includes: indexes, RLS, comments.
-- Previous: V402__pack048_materiality_sampling.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_jurisdictions
-- =============================================================================
-- Reference table of regulatory jurisdictions with their assurance
-- requirements. Each jurisdiction defines the applicable standard,
-- required assurance level, mandatory scopes, effective dates, transition
-- timeline, and company size thresholds.

CREATE TABLE ghg_assurance.gl_ap_jurisdictions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    jurisdiction_code           TEXT            NOT NULL UNIQUE,
    jurisdiction_name           VARCHAR(255)    NOT NULL,
    region                      VARCHAR(50),
    assurance_standard          VARCHAR(100),
    assurance_level_required    VARCHAR(20)     NOT NULL DEFAULT 'VOLUNTARY',
    scopes_required             JSONB           DEFAULT '[]',
    effective_date              DATE,
    transition_end              DATE,
    company_size_threshold      TEXT,
    description                 TEXT,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_jur_level CHECK (
        assurance_level_required IN (
            'LIMITED', 'REASONABLE', 'EXAMINATION',
            'NONE', 'VOLUNTARY'
        )
    ),
    CONSTRAINT chk_p048_jur_transition CHECK (
        transition_end IS NULL OR effective_date IS NULL OR transition_end >= effective_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_jur_code             ON ghg_assurance.gl_ap_jurisdictions(jurisdiction_code);
CREATE INDEX idx_p048_jur_region           ON ghg_assurance.gl_ap_jurisdictions(region);
CREATE INDEX idx_p048_jur_level            ON ghg_assurance.gl_ap_jurisdictions(assurance_level_required);
CREATE INDEX idx_p048_jur_effective        ON ghg_assurance.gl_ap_jurisdictions(effective_date);
CREATE INDEX idx_p048_jur_active           ON ghg_assurance.gl_ap_jurisdictions(is_active) WHERE is_active = true;
CREATE INDEX idx_p048_jur_created          ON ghg_assurance.gl_ap_jurisdictions(created_at DESC);
CREATE INDEX idx_p048_jur_scopes           ON ghg_assurance.gl_ap_jurisdictions USING GIN(scopes_required);
CREATE INDEX idx_p048_jur_metadata         ON ghg_assurance.gl_ap_jurisdictions USING GIN(metadata);

-- Composite: region + level for regional analysis
CREATE INDEX idx_p048_jur_region_level     ON ghg_assurance.gl_ap_jurisdictions(region, assurance_level_required);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_jur_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_jurisdictions
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_requirements
-- =============================================================================
-- Specific regulatory requirements per configuration and jurisdiction.
-- Each requirement has an applicability assessment, compliance status,
-- gap description if non-compliant, and action items with deadlines.

CREATE TABLE ghg_assurance.gl_ap_requirements (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    jurisdiction_id             UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_jurisdictions(id) ON DELETE CASCADE,
    requirement_description     TEXT            NOT NULL,
    is_applicable               BOOLEAN         NOT NULL DEFAULT true,
    applicability_reason        TEXT,
    compliance_status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    gap_description             TEXT,
    action_required             TEXT,
    action_deadline             DATE,
    action_owner                UUID,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_req_status CHECK (
        compliance_status IN (
            'COMPLIANT', 'GAP', 'NOT_APPLICABLE', 'PENDING'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_req_tenant           ON ghg_assurance.gl_ap_requirements(tenant_id);
CREATE INDEX idx_p048_req_config           ON ghg_assurance.gl_ap_requirements(config_id);
CREATE INDEX idx_p048_req_jurisdiction     ON ghg_assurance.gl_ap_requirements(jurisdiction_id);
CREATE INDEX idx_p048_req_applicable       ON ghg_assurance.gl_ap_requirements(is_applicable);
CREATE INDEX idx_p048_req_status           ON ghg_assurance.gl_ap_requirements(compliance_status);
CREATE INDEX idx_p048_req_deadline         ON ghg_assurance.gl_ap_requirements(action_deadline);
CREATE INDEX idx_p048_req_owner            ON ghg_assurance.gl_ap_requirements(action_owner);
CREATE INDEX idx_p048_req_created          ON ghg_assurance.gl_ap_requirements(created_at DESC);
CREATE INDEX idx_p048_req_metadata         ON ghg_assurance.gl_ap_requirements USING GIN(metadata);

-- Composite: config + jurisdiction for requirement listing
CREATE INDEX idx_p048_req_config_jur       ON ghg_assurance.gl_ap_requirements(config_id, jurisdiction_id);

-- Composite: config + status for compliance overview
CREATE INDEX idx_p048_req_config_status    ON ghg_assurance.gl_ap_requirements(config_id, compliance_status);

-- Composite: jurisdiction + status for regulatory reporting
CREATE INDEX idx_p048_req_jur_status       ON ghg_assurance.gl_ap_requirements(jurisdiction_id, compliance_status);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_req_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_requirements
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_assurance.gl_ap_compliance_status
-- =============================================================================
-- Aggregate compliance status per configuration and jurisdiction. Provides
-- a summary view with overall compliance flag, gap count, action count,
-- and provenance for audit trail.

CREATE TABLE ghg_assurance.gl_ap_compliance_status (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    jurisdiction_id             UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_jurisdictions(id) ON DELETE CASCADE,
    overall_compliant           BOOLEAN         NOT NULL DEFAULT false,
    gaps_count                  INTEGER         NOT NULL DEFAULT 0,
    actions_count               INTEGER         NOT NULL DEFAULT 0,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_cs_gaps CHECK (
        gaps_count >= 0
    ),
    CONSTRAINT chk_p048_cs_actions CHECK (
        actions_count >= 0
    ),
    CONSTRAINT uq_p048_cs_config_jur UNIQUE (config_id, jurisdiction_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_cs_tenant            ON ghg_assurance.gl_ap_compliance_status(tenant_id);
CREATE INDEX idx_p048_cs_config            ON ghg_assurance.gl_ap_compliance_status(config_id);
CREATE INDEX idx_p048_cs_jurisdiction      ON ghg_assurance.gl_ap_compliance_status(jurisdiction_id);
CREATE INDEX idx_p048_cs_compliant         ON ghg_assurance.gl_ap_compliance_status(overall_compliant);
CREATE INDEX idx_p048_cs_assessed          ON ghg_assurance.gl_ap_compliance_status(assessed_at DESC);
CREATE INDEX idx_p048_cs_created           ON ghg_assurance.gl_ap_compliance_status(created_at DESC);
CREATE INDEX idx_p048_cs_metadata          ON ghg_assurance.gl_ap_compliance_status USING GIN(metadata);

-- Composite: tenant + config for scoped listing
CREATE INDEX idx_p048_cs_tenant_config     ON ghg_assurance.gl_ap_compliance_status(tenant_id, config_id);

-- Composite: config + compliant for compliance dashboard
CREATE INDEX idx_p048_cs_config_comp       ON ghg_assurance.gl_ap_compliance_status(config_id, overall_compliant);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_cs_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_compliance_status
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_jurisdictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_requirements ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_compliance_status ENABLE ROW LEVEL SECURITY;

-- Jurisdictions are shared reference data; service bypass only
CREATE POLICY p048_jur_service_bypass
    ON ghg_assurance.gl_ap_jurisdictions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Allow public read access to jurisdictions (reference data)
CREATE POLICY p048_jur_public_read
    ON ghg_assurance.gl_ap_jurisdictions
    FOR SELECT
    USING (TRUE);

CREATE POLICY p048_req_tenant_isolation
    ON ghg_assurance.gl_ap_requirements
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_req_service_bypass
    ON ghg_assurance.gl_ap_requirements
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_cs_tenant_isolation
    ON ghg_assurance.gl_ap_compliance_status
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_cs_service_bypass
    ON ghg_assurance.gl_ap_compliance_status
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_jurisdictions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_requirements TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_compliance_status TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_assurance.gl_ap_jurisdictions IS
    'Regulatory jurisdiction reference data with assurance standard, level, scope requirements, and transition timelines.';
COMMENT ON TABLE ghg_assurance.gl_ap_requirements IS
    'Specific regulatory requirements per configuration and jurisdiction with applicability and compliance assessment.';
COMMENT ON TABLE ghg_assurance.gl_ap_compliance_status IS
    'Aggregate compliance status per configuration-jurisdiction pair with gap and action counts.';

COMMENT ON COLUMN ghg_assurance.gl_ap_jurisdictions.jurisdiction_code IS 'Unique code, e.g. EU_CSRD, UK_SECR, US_SEC, AU_NGER, JP_ACT, SG_MAS, HK_HKEX, NZ_XRB, CA_CSSB, CH_TCFD, BR_CVM, ZA_JSE.';
COMMENT ON COLUMN ghg_assurance.gl_ap_jurisdictions.assurance_level_required IS 'LIMITED (negative assurance), REASONABLE (positive), EXAMINATION (US), NONE (no requirement), VOLUNTARY (market-driven).';
COMMENT ON COLUMN ghg_assurance.gl_ap_jurisdictions.scopes_required IS 'JSON array of mandatory scopes, e.g. ["SCOPE_1","SCOPE_2"] or ["SCOPE_1","SCOPE_2","SCOPE_3"].';
COMMENT ON COLUMN ghg_assurance.gl_ap_jurisdictions.transition_end IS 'End of transition period after which full requirements apply (e.g., EU CSRD reasonable assurance from 2028).';
COMMENT ON COLUMN ghg_assurance.gl_ap_jurisdictions.company_size_threshold IS 'Description of company size threshold, e.g. ">500 employees" or ">250 employees, >EUR 40M turnover".';
COMMENT ON COLUMN ghg_assurance.gl_ap_requirements.compliance_status IS 'COMPLIANT (meets requirement), GAP (does not meet), NOT_APPLICABLE (requirement excluded), PENDING (not yet assessed).';
COMMENT ON COLUMN ghg_assurance.gl_ap_compliance_status.overall_compliant IS 'True if all applicable requirements for this jurisdiction are COMPLIANT.';

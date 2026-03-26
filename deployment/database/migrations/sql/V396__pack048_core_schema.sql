-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V396 - Core Schema
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_assurance schema and foundational tables for GHG assurance
-- preparation. Tracks assurance configurations (standard selection, assurance
-- level, target scopes, jurisdiction) and engagement lifecycle management
-- (verifier selection, phase tracking, cost estimation, timeline).
--
-- Tables (2):
--   1. ghg_assurance.gl_ap_configurations
--   2. ghg_assurance.gl_ap_engagements
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V395__pack047_views_indexes_seed.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ghg_assurance;

SET search_path TO ghg_assurance, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_assurance.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_configurations
-- =============================================================================
-- Pack-level assurance configuration per organisation. Defines the assurance
-- standard (ISAE 3410, ISO 14064-3, AA1000AS v3, etc.), assurance level
-- (limited or reasonable), target scopes for verification, jurisdiction for
-- regulatory context, and configuration versioning with integrity hashing.

CREATE TABLE ghg_assurance.gl_ap_configurations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_name                 VARCHAR(255)    NOT NULL,
    config_version              INTEGER         NOT NULL DEFAULT 1,
    assurance_standard          VARCHAR(30)     NOT NULL,
    assurance_level             VARCHAR(20)     NOT NULL DEFAULT 'LIMITED',
    target_scopes               JSONB           NOT NULL DEFAULT '["SCOPE_1","SCOPE_2"]',
    jurisdiction                VARCHAR(50)     NOT NULL DEFAULT 'EU_CSRD',
    reporting_year              INTEGER,
    base_year                   INTEGER,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    config_hash                 TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p048_cfg_standard CHECK (
        assurance_standard IN (
            'ISAE_3410', 'ISO_14064_3', 'AA1000AS_V3',
            'ISAE_3000', 'SSAE_18', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p048_cfg_level CHECK (
        assurance_level IN ('LIMITED', 'REASONABLE')
    ),
    CONSTRAINT chk_p048_cfg_jurisdiction CHECK (
        jurisdiction IN (
            'EU_CSRD', 'UK_SECR', 'US_SEC', 'AU_NGER',
            'JP_ACT', 'SG_MAS', 'HK_HKEX', 'NZ_XRB',
            'CA_CSSB', 'CH_TCFD', 'BR_CVM', 'ZA_JSE'
        )
    ),
    CONSTRAINT chk_p048_cfg_version CHECK (
        config_version >= 1
    ),
    CONSTRAINT chk_p048_cfg_reporting_year CHECK (
        reporting_year IS NULL OR (reporting_year >= 2000 AND reporting_year <= 2100)
    ),
    CONSTRAINT chk_p048_cfg_base_year CHECK (
        base_year IS NULL OR (base_year >= 2000 AND base_year <= 2100)
    ),
    CONSTRAINT uq_p048_cfg_tenant_name UNIQUE (tenant_id, config_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_cfg_tenant           ON ghg_assurance.gl_ap_configurations(tenant_id);
CREATE INDEX idx_p048_cfg_standard         ON ghg_assurance.gl_ap_configurations(assurance_standard);
CREATE INDEX idx_p048_cfg_level            ON ghg_assurance.gl_ap_configurations(assurance_level);
CREATE INDEX idx_p048_cfg_jurisdiction     ON ghg_assurance.gl_ap_configurations(jurisdiction);
CREATE INDEX idx_p048_cfg_active           ON ghg_assurance.gl_ap_configurations(is_active) WHERE is_active = true;
CREATE INDEX idx_p048_cfg_created          ON ghg_assurance.gl_ap_configurations(created_at DESC);
CREATE INDEX idx_p048_cfg_scopes           ON ghg_assurance.gl_ap_configurations USING GIN(target_scopes);
CREATE INDEX idx_p048_cfg_metadata         ON ghg_assurance.gl_ap_configurations USING GIN(metadata);

-- Composite: tenant + active for filtered listing
CREATE INDEX idx_p048_cfg_tenant_active    ON ghg_assurance.gl_ap_configurations(tenant_id, is_active);

-- Composite: standard + jurisdiction for regulatory queries
CREATE INDEX idx_p048_cfg_std_juris        ON ghg_assurance.gl_ap_configurations(assurance_standard, jurisdiction);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_cfg_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_configurations
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_engagements
-- =============================================================================
-- Assurance engagement lifecycle tracking. Each engagement links to a
-- configuration and captures verifier details, phase progression (PLANNING
-- through CLOSEOUT), date tracking (planned vs actual), cost management,
-- and status workflow. An organisation may have multiple engagements per
-- configuration (e.g., annual assurance cycles).

CREATE TABLE ghg_assurance.gl_ap_engagements (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    engagement_name             VARCHAR(255)    NOT NULL,
    verifier_name               VARCHAR(255),
    verifier_contact            TEXT,
    engagement_phase            VARCHAR(30)     NOT NULL DEFAULT 'PLANNING',
    planned_start               DATE,
    planned_end                 DATE,
    actual_start                DATE,
    actual_end                  DATE,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    cost_estimate               NUMERIC(14,2),
    actual_cost                 NUMERIC(14,2),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p048_eng_phase CHECK (
        engagement_phase IN (
            'PLANNING', 'RISK_ASSESSMENT', 'FIELDWORK',
            'REPORTING', 'CLOSEOUT'
        )
    ),
    CONSTRAINT chk_p048_eng_status CHECK (
        status IN (
            'DRAFT', 'ACTIVE', 'IN_PROGRESS',
            'COMPLETED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p048_eng_planned_dates CHECK (
        planned_start IS NULL OR planned_end IS NULL OR planned_end >= planned_start
    ),
    CONSTRAINT chk_p048_eng_actual_dates CHECK (
        actual_start IS NULL OR actual_end IS NULL OR actual_end >= actual_start
    ),
    CONSTRAINT chk_p048_eng_cost_estimate CHECK (
        cost_estimate IS NULL OR cost_estimate >= 0
    ),
    CONSTRAINT chk_p048_eng_actual_cost CHECK (
        actual_cost IS NULL OR actual_cost >= 0
    ),
    CONSTRAINT uq_p048_eng_config_name UNIQUE (config_id, engagement_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_eng_tenant           ON ghg_assurance.gl_ap_engagements(tenant_id);
CREATE INDEX idx_p048_eng_config           ON ghg_assurance.gl_ap_engagements(config_id);
CREATE INDEX idx_p048_eng_phase            ON ghg_assurance.gl_ap_engagements(engagement_phase);
CREATE INDEX idx_p048_eng_status           ON ghg_assurance.gl_ap_engagements(status);
CREATE INDEX idx_p048_eng_planned_start    ON ghg_assurance.gl_ap_engagements(planned_start);
CREATE INDEX idx_p048_eng_planned_end      ON ghg_assurance.gl_ap_engagements(planned_end);
CREATE INDEX idx_p048_eng_verifier         ON ghg_assurance.gl_ap_engagements(verifier_name);
CREATE INDEX idx_p048_eng_created          ON ghg_assurance.gl_ap_engagements(created_at DESC);
CREATE INDEX idx_p048_eng_metadata         ON ghg_assurance.gl_ap_engagements USING GIN(metadata);

-- Composite: tenant + config for filtered queries
CREATE INDEX idx_p048_eng_tenant_config    ON ghg_assurance.gl_ap_engagements(tenant_id, config_id);

-- Composite: config + status for lifecycle queries
CREATE INDEX idx_p048_eng_config_status    ON ghg_assurance.gl_ap_engagements(config_id, status);

-- Composite: phase + status for pipeline monitoring
CREATE INDEX idx_p048_eng_phase_status     ON ghg_assurance.gl_ap_engagements(engagement_phase, status);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_eng_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_engagements
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_engagements ENABLE ROW LEVEL SECURITY;

CREATE POLICY p048_cfg_tenant_isolation
    ON ghg_assurance.gl_ap_configurations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_cfg_service_bypass
    ON ghg_assurance.gl_ap_configurations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_eng_tenant_isolation
    ON ghg_assurance.gl_ap_engagements
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_eng_service_bypass
    ON ghg_assurance.gl_ap_engagements
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA ghg_assurance TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_configurations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_engagements TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_assurance.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_assurance IS
    'PACK-048 GHG Assurance Prep Pack - Complete assurance preparation lifecycle including evidence packaging, readiness assessment, provenance chains, internal controls, verifier query management, materiality and sampling, regulatory compliance, cost estimation, and timeline management.';

COMMENT ON TABLE ghg_assurance.gl_ap_configurations IS
    'Organisation-level assurance configuration defining standard (ISAE 3410, ISO 14064-3, AA1000AS v3), assurance level, target scopes, and jurisdiction context.';
COMMENT ON TABLE ghg_assurance.gl_ap_engagements IS
    'Assurance engagement lifecycle tracking with verifier details, phase progression (PLANNING through CLOSEOUT), date and cost management.';

COMMENT ON COLUMN ghg_assurance.gl_ap_configurations.assurance_standard IS 'Assurance standard: ISAE_3410 (GHG-specific), ISO_14064_3 (verification), AA1000AS_V3 (AccountAbility), ISAE_3000 (general), SSAE_18 (US SOC), CUSTOM.';
COMMENT ON COLUMN ghg_assurance.gl_ap_configurations.assurance_level IS 'Level of assurance: LIMITED (negative assurance) or REASONABLE (positive assurance, higher cost/effort).';
COMMENT ON COLUMN ghg_assurance.gl_ap_configurations.target_scopes IS 'JSON array of emission scopes in assurance scope, e.g. ["SCOPE_1","SCOPE_2","SCOPE_3"].';
COMMENT ON COLUMN ghg_assurance.gl_ap_configurations.jurisdiction IS 'Regulatory jurisdiction determining assurance requirements: EU_CSRD, UK_SECR, US_SEC, etc.';
COMMENT ON COLUMN ghg_assurance.gl_ap_configurations.config_hash IS 'SHA-256 hash of configuration parameters for change detection and versioning.';
COMMENT ON COLUMN ghg_assurance.gl_ap_engagements.engagement_phase IS 'Current engagement phase: PLANNING, RISK_ASSESSMENT, FIELDWORK, REPORTING, CLOSEOUT.';
COMMENT ON COLUMN ghg_assurance.gl_ap_engagements.status IS 'Engagement status: DRAFT (setup), ACTIVE (approved), IN_PROGRESS (underway), COMPLETED (finalised), CANCELLED.';
COMMENT ON COLUMN ghg_assurance.gl_ap_engagements.cost_estimate IS 'Estimated total assurance engagement cost in base currency.';
COMMENT ON COLUMN ghg_assurance.gl_ap_engagements.actual_cost IS 'Actual incurred cost once engagement concludes.';

-- =============================================================================
-- V377: PACK-046 Intensity Metrics Pack - Denominator Registry
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the denominator definition registry and denominator value storage.
-- The registry holds 25+ standard denominators (revenue, FTE, area, production,
-- energy, etc.) plus custom denominators. Each denominator has sector
-- applicability, framework alignment (MANDATORY/RECOMMENDED/OPTIONAL per
-- framework), validation rules, and unit conversion factors.
--
-- Denominator values store actual data per organisation/period/entity with
-- data quality scoring, source provenance, and estimation flags.
--
-- Tables (2):
--   1. ghg_intensity.gl_im_denominator_definitions
--   2. ghg_intensity.gl_im_denominator_values
--
-- Also includes: indexes, RLS, comments.
-- Previous: V376__pack046_core_schema.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_denominator_definitions
-- =============================================================================
-- Registry of available denominators for intensity calculations. Contains
-- 25+ standard denominators pre-seeded in V385 covering financial (revenue,
-- AUM, lending), physical (production, vehicle-km, tonne-km), headcount
-- (FTE), area (GLA, floor area), and energy (MWh) categories. Custom
-- denominators can be added per organisation. Framework alignment maps
-- each denominator to its requirement level per reporting framework.

CREATE TABLE ghg_intensity.gl_im_denominator_definitions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    denominator_code            VARCHAR(50)     NOT NULL UNIQUE,
    name                        VARCHAR(255)    NOT NULL,
    unit                        VARCHAR(50)     NOT NULL,
    category                    VARCHAR(30)     NOT NULL,
    applicable_sectors          JSONB           NOT NULL DEFAULT '[]',
    framework_alignment         JSONB           NOT NULL DEFAULT '{}',
    validation_rules            JSONB           NOT NULL DEFAULT '{}',
    conversion_factors          JSONB           NOT NULL DEFAULT '{}',
    description                 TEXT,
    is_standard                 BOOLEAN         NOT NULL DEFAULT true,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_dd_category CHECK (
        category IN (
            'FINANCIAL', 'PHYSICAL', 'HEADCOUNT', 'AREA',
            'ENERGY', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p046_dd_code_format CHECK (
        denominator_code ~ '^DEN-[A-Z0-9_-]+$'
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_dd_code               ON ghg_intensity.gl_im_denominator_definitions(denominator_code);
CREATE INDEX idx_p046_dd_category           ON ghg_intensity.gl_im_denominator_definitions(category);
CREATE INDEX idx_p046_dd_standard           ON ghg_intensity.gl_im_denominator_definitions(is_standard);
CREATE INDEX idx_p046_dd_active             ON ghg_intensity.gl_im_denominator_definitions(is_active) WHERE is_active = true;
CREATE INDEX idx_p046_dd_created            ON ghg_intensity.gl_im_denominator_definitions(created_at DESC);
CREATE INDEX idx_p046_dd_sectors            ON ghg_intensity.gl_im_denominator_definitions USING GIN(applicable_sectors);
CREATE INDEX idx_p046_dd_frameworks         ON ghg_intensity.gl_im_denominator_definitions USING GIN(framework_alignment);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_dd_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_denominator_definitions
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_denominator_values
-- =============================================================================
-- Actual denominator data per organisation, configuration, period, and
-- optionally per entity (NULL entity_id means consolidated). Each value
-- carries data quality scoring (1-5 scale), source documentation, estimation
-- flags, and uncertainty percentage. The provenance hash links to the
-- original source data for audit trail.

CREATE TABLE ghg_intensity.gl_im_denominator_values (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_reporting_periods(id) ON DELETE CASCADE,
    denominator_id              UUID            NOT NULL REFERENCES ghg_intensity.gl_im_denominator_definitions(id),
    entity_id                   UUID,
    entity_name                 VARCHAR(255),
    value                       NUMERIC(20,6)   NOT NULL,
    unit                        VARCHAR(50)     NOT NULL,
    data_quality_score          INTEGER         NOT NULL DEFAULT 3,
    source_description          TEXT,
    source_document_ref         VARCHAR(255),
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    estimation_method           VARCHAR(100),
    uncertainty_pct             NUMERIC(10,6),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p046_dv_quality CHECK (
        data_quality_score BETWEEN 1 AND 5
    ),
    CONSTRAINT chk_p046_dv_value CHECK (
        value >= 0
    ),
    CONSTRAINT chk_p046_dv_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100)
    ),
    CONSTRAINT uq_p046_dv_org_period_denom_entity UNIQUE (org_id, config_id, period_id, denominator_id, entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_dv_tenant            ON ghg_intensity.gl_im_denominator_values(tenant_id);
CREATE INDEX idx_p046_dv_org               ON ghg_intensity.gl_im_denominator_values(org_id);
CREATE INDEX idx_p046_dv_config            ON ghg_intensity.gl_im_denominator_values(config_id);
CREATE INDEX idx_p046_dv_period            ON ghg_intensity.gl_im_denominator_values(period_id);
CREATE INDEX idx_p046_dv_denominator       ON ghg_intensity.gl_im_denominator_values(denominator_id);
CREATE INDEX idx_p046_dv_entity            ON ghg_intensity.gl_im_denominator_values(entity_id);
CREATE INDEX idx_p046_dv_quality           ON ghg_intensity.gl_im_denominator_values(data_quality_score);
CREATE INDEX idx_p046_dv_estimated         ON ghg_intensity.gl_im_denominator_values(is_estimated) WHERE is_estimated = true;
CREATE INDEX idx_p046_dv_created           ON ghg_intensity.gl_im_denominator_values(created_at DESC);
CREATE INDEX idx_p046_dv_metadata          ON ghg_intensity.gl_im_denominator_values USING GIN(metadata);

-- Composite: common lookup pattern (org + period + denominator)
CREATE INDEX idx_p046_dv_lookup            ON ghg_intensity.gl_im_denominator_values(org_id, period_id, denominator_id);

-- Composite: config + period for batch retrieval
CREATE INDEX idx_p046_dv_config_period     ON ghg_intensity.gl_im_denominator_values(config_id, period_id);

-- Composite: tenant + org for multi-tenant queries
CREATE INDEX idx_p046_dv_tenant_org        ON ghg_intensity.gl_im_denominator_values(tenant_id, org_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_dv_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_denominator_values
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_denominator_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_denominator_values ENABLE ROW LEVEL SECURITY;

-- Denominator definitions are shared (no tenant_id), so use permissive policy
CREATE POLICY p046_dd_read_all
    ON ghg_intensity.gl_im_denominator_definitions
    FOR SELECT
    USING (true);
CREATE POLICY p046_dd_service_write
    ON ghg_intensity.gl_im_denominator_definitions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p046_dv_tenant_isolation
    ON ghg_intensity.gl_im_denominator_values
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_dv_service_bypass
    ON ghg_intensity.gl_im_denominator_values
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_denominator_definitions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_denominator_values TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_intensity.gl_im_denominator_definitions IS
    'Registry of 25+ standard and custom denominators for intensity calculations, with sector applicability and framework alignment mappings.';
COMMENT ON TABLE ghg_intensity.gl_im_denominator_values IS
    'Actual denominator data per organisation/period/entity with data quality scoring, source provenance, and estimation flags.';

COMMENT ON COLUMN ghg_intensity.gl_im_denominator_definitions.denominator_code IS 'Unique code following DEN-{TYPE}-{UNIT} pattern, e.g. DEN-REV-EUR, DEN-FTE, DEN-GLA-M2.';
COMMENT ON COLUMN ghg_intensity.gl_im_denominator_definitions.category IS 'FINANCIAL (revenue, AUM), PHYSICAL (production, transport), HEADCOUNT (FTE), AREA (m2, ha), ENERGY (MWh), CUSTOM.';
COMMENT ON COLUMN ghg_intensity.gl_im_denominator_definitions.applicable_sectors IS 'JSON array of sector codes where this denominator is applicable, e.g. ["MANUFACTURING", "CEMENT"].';
COMMENT ON COLUMN ghg_intensity.gl_im_denominator_definitions.framework_alignment IS 'JSON map of framework to requirement level, e.g. {"ESRS_E1": "MANDATORY", "CDP": "RECOMMENDED"}.';
COMMENT ON COLUMN ghg_intensity.gl_im_denominator_definitions.conversion_factors IS 'JSON map of unit conversion factors for normalisation, e.g. {"EUR_to_USD": 1.08}.';
COMMENT ON COLUMN ghg_intensity.gl_im_denominator_values.entity_id IS 'NULL for consolidated (organisation-level) values; entity UUID for entity-level breakdowns.';
COMMENT ON COLUMN ghg_intensity.gl_im_denominator_values.data_quality_score IS 'GHG Protocol data quality scale: 1=highest (measured), 5=lowest (estimated/proxy).';
COMMENT ON COLUMN ghg_intensity.gl_im_denominator_values.provenance_hash IS 'SHA-256 hash linking to source data for complete audit trail.';

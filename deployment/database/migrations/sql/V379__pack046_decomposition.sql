-- =============================================================================
-- V379: PACK-046 Intensity Metrics Pack - LMDI Decomposition
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for Logarithmic Mean Divisia Index (LMDI) decomposition
-- analysis. LMDI decomposes total emission changes between two periods into
-- three additive effects: activity effect (scale), structure effect (mix),
-- and intensity effect (efficiency). This is the standard decomposition
-- method recommended by the IEA and used in SBTi SDA. The closure residual
-- should be approximately zero for LMDI (perfect decomposition).
--
-- Tables (2):
--   1. ghg_intensity.gl_im_decompositions
--   2. ghg_intensity.gl_im_decomposition_entities
--
-- Also includes: indexes, RLS, comments.
-- Previous: V378__pack046_intensity_calculations.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_decompositions
-- =============================================================================
-- Period-over-period LMDI decomposition at the organisation level. Compares
-- a base period against a comparison period to decompose total emission
-- change into activity, structure, and intensity effects. Supports both
-- additive (LMDI-I) and multiplicative (LMDI-II) methods. The closure
-- residual should approach zero for LMDI but is tracked for validation.

CREATE TABLE ghg_intensity.gl_im_decompositions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    base_period_id              UUID            NOT NULL REFERENCES ghg_intensity.gl_im_reporting_periods(id),
    comparison_period_id        UUID            NOT NULL REFERENCES ghg_intensity.gl_im_reporting_periods(id),
    denominator_code            VARCHAR(50)     NOT NULL,
    scope_inclusion             VARCHAR(50)     NOT NULL,
    method                      VARCHAR(50)     NOT NULL DEFAULT 'LMDI_I_ADDITIVE',
    total_change_tco2e          NUMERIC(20,6)   NOT NULL,
    activity_effect_tco2e       NUMERIC(20,6)   NOT NULL,
    structure_effect_tco2e      NUMERIC(20,6)   NOT NULL,
    intensity_effect_tco2e      NUMERIC(20,6)   NOT NULL,
    closure_residual_tco2e      NUMERIC(20,10),
    base_total_emissions        NUMERIC(20,6),
    comparison_total_emissions  NUMERIC(20,6),
    base_total_denominator      NUMERIC(20,6),
    comparison_total_denominator NUMERIC(20,6),
    decomposition_metadata      JSONB           NOT NULL DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_dec_method CHECK (
        method IN (
            'LMDI_I_ADDITIVE', 'LMDI_II_MULTIPLICATIVE',
            'SDA_ADDITIVE', 'SIMPLE_ADDITIVE'
        )
    ),
    CONSTRAINT chk_p046_dec_scope CHECK (
        scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p046_dec_periods CHECK (
        base_period_id != comparison_period_id
    ),
    CONSTRAINT chk_p046_dec_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_dec_tenant           ON ghg_intensity.gl_im_decompositions(tenant_id);
CREATE INDEX idx_p046_dec_org              ON ghg_intensity.gl_im_decompositions(org_id);
CREATE INDEX idx_p046_dec_config           ON ghg_intensity.gl_im_decompositions(config_id);
CREATE INDEX idx_p046_dec_base_period      ON ghg_intensity.gl_im_decompositions(base_period_id);
CREATE INDEX idx_p046_dec_comp_period      ON ghg_intensity.gl_im_decompositions(comparison_period_id);
CREATE INDEX idx_p046_dec_denom            ON ghg_intensity.gl_im_decompositions(denominator_code);
CREATE INDEX idx_p046_dec_scope            ON ghg_intensity.gl_im_decompositions(scope_inclusion);
CREATE INDEX idx_p046_dec_method           ON ghg_intensity.gl_im_decompositions(method);
CREATE INDEX idx_p046_dec_calculated       ON ghg_intensity.gl_im_decompositions(calculated_at DESC);
CREATE INDEX idx_p046_dec_created          ON ghg_intensity.gl_im_decompositions(created_at DESC);
CREATE INDEX idx_p046_dec_provenance       ON ghg_intensity.gl_im_decompositions(provenance_hash);

-- Composite: org + periods for comparison queries
CREATE INDEX idx_p046_dec_org_periods      ON ghg_intensity.gl_im_decompositions(org_id, base_period_id, comparison_period_id);

-- Composite: denominator + scope for filtered analysis
CREATE INDEX idx_p046_dec_denom_scope      ON ghg_intensity.gl_im_decompositions(denominator_code, scope_inclusion);

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_decomposition_entities
-- =============================================================================
-- Entity-level contributions to each decomposition effect. Each entity's
-- contribution to the activity, structure, and intensity effects is stored,
-- along with the base and comparison period emissions and denominator values
-- for auditability. The sum of entity contributions equals the organisation-
-- level effect values in the parent decomposition record.

CREATE TABLE ghg_intensity.gl_im_decomposition_entities (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    decomposition_id            UUID            NOT NULL REFERENCES ghg_intensity.gl_im_decompositions(id) ON DELETE CASCADE,
    entity_id                   UUID            NOT NULL,
    entity_name                 VARCHAR(255)    NOT NULL,
    activity_contribution_tco2e NUMERIC(20,6),
    structure_contribution_tco2e NUMERIC(20,6),
    intensity_contribution_tco2e NUMERIC(20,6),
    base_emissions_tco2e        NUMERIC(20,6),
    comparison_emissions_tco2e  NUMERIC(20,6),
    base_denominator            NUMERIC(20,6),
    comparison_denominator      NUMERIC(20,6),
    base_intensity              NUMERIC(20,10),
    comparison_intensity        NUMERIC(20,10),
    entity_weight_base          NUMERIC(10,6),
    entity_weight_comparison    NUMERIC(10,6),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_dce_base_emissions CHECK (
        base_emissions_tco2e IS NULL OR base_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p046_dce_comp_emissions CHECK (
        comparison_emissions_tco2e IS NULL OR comparison_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p046_dce_base_denom CHECK (
        base_denominator IS NULL OR base_denominator >= 0
    ),
    CONSTRAINT chk_p046_dce_comp_denom CHECK (
        comparison_denominator IS NULL OR comparison_denominator >= 0
    ),
    CONSTRAINT uq_p046_dce_decomp_entity UNIQUE (decomposition_id, entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_dce_decomposition    ON ghg_intensity.gl_im_decomposition_entities(decomposition_id);
CREATE INDEX idx_p046_dce_entity           ON ghg_intensity.gl_im_decomposition_entities(entity_id);
CREATE INDEX idx_p046_dce_created          ON ghg_intensity.gl_im_decomposition_entities(created_at DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_decompositions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_decomposition_entities ENABLE ROW LEVEL SECURITY;

CREATE POLICY p046_dec_tenant_isolation
    ON ghg_intensity.gl_im_decompositions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_dec_service_bypass
    ON ghg_intensity.gl_im_decompositions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Decomposition entities inherit access via decomposition_id FK
CREATE POLICY p046_dce_service_bypass
    ON ghg_intensity.gl_im_decomposition_entities
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_decompositions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_decomposition_entities TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_intensity.gl_im_decompositions IS
    'Period-over-period LMDI decomposition separating total emission change into activity (scale), structure (mix), and intensity (efficiency) effects.';
COMMENT ON TABLE ghg_intensity.gl_im_decomposition_entities IS
    'Entity-level contributions to each decomposition effect, summing to the organisation-level totals in the parent decomposition record.';

COMMENT ON COLUMN ghg_intensity.gl_im_decompositions.method IS 'Decomposition method: LMDI_I_ADDITIVE (recommended), LMDI_II_MULTIPLICATIVE, SDA_ADDITIVE, or SIMPLE_ADDITIVE.';
COMMENT ON COLUMN ghg_intensity.gl_im_decompositions.activity_effect_tco2e IS 'Emission change due to overall activity scale change (total denominator growth/decline).';
COMMENT ON COLUMN ghg_intensity.gl_im_decompositions.structure_effect_tco2e IS 'Emission change due to shifts in the mix of sub-activities (entity share changes).';
COMMENT ON COLUMN ghg_intensity.gl_im_decompositions.intensity_effect_tco2e IS 'Emission change due to efficiency improvements/declines at the entity level.';
COMMENT ON COLUMN ghg_intensity.gl_im_decompositions.closure_residual_tco2e IS 'Residual after summing all effects vs total change. Should be ~0 for LMDI; non-zero indicates rounding or data issues.';
COMMENT ON COLUMN ghg_intensity.gl_im_decomposition_entities.entity_weight_base IS 'Entity share of total denominator in the base period (0-1 fraction).';
COMMENT ON COLUMN ghg_intensity.gl_im_decomposition_entities.entity_weight_comparison IS 'Entity share of total denominator in the comparison period (0-1 fraction).';

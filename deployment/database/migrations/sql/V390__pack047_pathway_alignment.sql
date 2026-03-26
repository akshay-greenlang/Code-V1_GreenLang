-- =============================================================================
-- V390: PACK-047 GHG Emissions Benchmark Pack - Pathway Alignment
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for decarbonisation pathway management, annual waypoints,
-- and alignment result calculation. Pathways represent science-based
-- decarbonisation trajectories from IEA NZE, IPCC AR6 (C1/C2/C3), SBTi
-- SDA, OECM, TPI Carbon Performance, and CRREM. Waypoints store annual
-- intensity/emissions milestones. Alignment results calculate the gap
-- between an organisation's actual performance and the target pathway,
-- including overshoot year detection and convergence analysis.
--
-- Tables (3):
--   1. ghg_benchmark.gl_bm_pathways
--   2. ghg_benchmark.gl_bm_pathway_waypoints
--   3. ghg_benchmark.gl_bm_alignment_results
--
-- Also includes: indexes, RLS, comments.
-- Previous: V389__pack047_external_datasets.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_pathways
-- =============================================================================
-- Decarbonisation pathway definitions from authoritative sources. Each
-- pathway has a type (IEA NZE, IPCC AR6, SBTi SDA, etc.), sector scope,
-- temperature target (e.g., 1.5C, 2.0C), base and target years, source
-- reference, and metadata for source-specific parameters (scenario name,
-- model, region, probability).

CREATE TABLE ghg_benchmark.gl_bm_pathways (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    pathway_type                VARCHAR(30)     NOT NULL,
    pathway_name                VARCHAR(255)    NOT NULL,
    sector                      VARCHAR(100),
    temperature_target          NUMERIC(3,1)    NOT NULL,
    base_year                   INTEGER         NOT NULL,
    target_year                 INTEGER         NOT NULL,
    source_reference            TEXT,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_pw_type CHECK (
        pathway_type IN (
            'IEA_NZE', 'IPCC_AR6_C1', 'IPCC_AR6_C2', 'IPCC_AR6_C3',
            'SBTI_SDA', 'OECM', 'TPI_CP', 'CRREM'
        )
    ),
    CONSTRAINT chk_p047_pw_temp CHECK (
        temperature_target >= 1.0 AND temperature_target <= 4.0
    ),
    CONSTRAINT chk_p047_pw_base_year CHECK (
        base_year >= 2000 AND base_year <= 2100
    ),
    CONSTRAINT chk_p047_pw_target_year CHECK (
        target_year > base_year AND target_year <= 2100
    ),
    CONSTRAINT uq_p047_pw_tenant_name UNIQUE (tenant_id, pathway_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_pw_tenant            ON ghg_benchmark.gl_bm_pathways(tenant_id);
CREATE INDEX idx_p047_pw_type              ON ghg_benchmark.gl_bm_pathways(pathway_type);
CREATE INDEX idx_p047_pw_sector            ON ghg_benchmark.gl_bm_pathways(sector);
CREATE INDEX idx_p047_pw_temp              ON ghg_benchmark.gl_bm_pathways(temperature_target);
CREATE INDEX idx_p047_pw_base_year         ON ghg_benchmark.gl_bm_pathways(base_year);
CREATE INDEX idx_p047_pw_target_year       ON ghg_benchmark.gl_bm_pathways(target_year);
CREATE INDEX idx_p047_pw_active            ON ghg_benchmark.gl_bm_pathways(is_active) WHERE is_active = true;
CREATE INDEX idx_p047_pw_created           ON ghg_benchmark.gl_bm_pathways(created_at DESC);
CREATE INDEX idx_p047_pw_metadata          ON ghg_benchmark.gl_bm_pathways USING GIN(metadata);

-- Composite: type + sector for pathway selection
CREATE INDEX idx_p047_pw_type_sector       ON ghg_benchmark.gl_bm_pathways(pathway_type, sector);

-- Composite: tenant + type for tenant-scoped queries
CREATE INDEX idx_p047_pw_tenant_type       ON ghg_benchmark.gl_bm_pathways(tenant_id, pathway_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p047_pw_updated
    BEFORE UPDATE ON ghg_benchmark.gl_bm_pathways
    FOR EACH ROW EXECUTE FUNCTION ghg_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_pathway_waypoints
-- =============================================================================
-- Annual milestones along a decarbonisation pathway. Each waypoint stores
-- the target intensity and/or absolute emissions for a specific year.
-- Interpolated waypoints (between source data points) are flagged. Source
-- notes track the origin of each data point (e.g., "IEA WEO 2023 Table 3.4",
-- "IPCC AR6 WGIII Table SPM.1").

CREATE TABLE ghg_benchmark.gl_bm_pathway_waypoints (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    pathway_id                  UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_pathways(id) ON DELETE CASCADE,
    year                        INTEGER         NOT NULL,
    intensity_value             NUMERIC(20,10),
    emissions_value             NUMERIC(20,6),
    is_interpolated             BOOLEAN         NOT NULL DEFAULT false,
    source_note                 TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_wpw_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p047_wpw_intensity CHECK (
        intensity_value IS NULL OR intensity_value >= 0
    ),
    CONSTRAINT chk_p047_wpw_emissions CHECK (
        emissions_value IS NULL OR emissions_value >= 0
    ),
    CONSTRAINT chk_p047_wpw_has_value CHECK (
        intensity_value IS NOT NULL OR emissions_value IS NOT NULL
    ),
    CONSTRAINT uq_p047_wpw_pathway_year UNIQUE (pathway_id, year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_wpw_pathway          ON ghg_benchmark.gl_bm_pathway_waypoints(pathway_id);
CREATE INDEX idx_p047_wpw_year             ON ghg_benchmark.gl_bm_pathway_waypoints(year);
CREATE INDEX idx_p047_wpw_interpolated     ON ghg_benchmark.gl_bm_pathway_waypoints(is_interpolated);
CREATE INDEX idx_p047_wpw_created          ON ghg_benchmark.gl_bm_pathway_waypoints(created_at DESC);

-- Composite: pathway + year for ordered retrieval
CREATE INDEX idx_p047_wpw_pathway_year     ON ghg_benchmark.gl_bm_pathway_waypoints(pathway_id, year ASC);

-- =============================================================================
-- Table 3: ghg_benchmark.gl_bm_alignment_results
-- =============================================================================
-- Alignment analysis results comparing an organisation's emissions
-- performance against a target pathway. Calculates the absolute and
-- percentage gap between actual and target intensity for a given reporting
-- year. Includes years-to-convergence estimate (how many years at current
-- reduction rate before reaching pathway), alignment score (0-1), and
-- overshoot year detection (when cumulative emissions exceed the pathway
-- carbon budget).

CREATE TABLE ghg_benchmark.gl_bm_alignment_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    pathway_id                  UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_pathways(id) ON DELETE CASCADE,
    reporting_year              INTEGER         NOT NULL,
    org_intensity               NUMERIC(20,10)  NOT NULL,
    pathway_intensity           NUMERIC(20,10)  NOT NULL,
    gap_absolute                NUMERIC(20,10)  NOT NULL,
    gap_percentage              NUMERIC(8,4)    NOT NULL,
    years_to_convergence        NUMERIC(5,1),
    alignment_score             NUMERIC(5,3),
    overshoot_year              INTEGER,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'CALCULATED',
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_ar_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p047_ar_org_intensity CHECK (
        org_intensity >= 0
    ),
    CONSTRAINT chk_p047_ar_pathway_intensity CHECK (
        pathway_intensity >= 0
    ),
    CONSTRAINT chk_p047_ar_alignment CHECK (
        alignment_score IS NULL OR (alignment_score >= 0 AND alignment_score <= 1)
    ),
    CONSTRAINT chk_p047_ar_convergence CHECK (
        years_to_convergence IS NULL OR years_to_convergence >= 0
    ),
    CONSTRAINT chk_p047_ar_overshoot CHECK (
        overshoot_year IS NULL OR (overshoot_year >= 2000 AND overshoot_year <= 2200)
    ),
    CONSTRAINT chk_p047_ar_status CHECK (
        status IN ('CALCULATED', 'VERIFIED', 'PUBLISHED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p047_ar_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_ar_tenant            ON ghg_benchmark.gl_bm_alignment_results(tenant_id);
CREATE INDEX idx_p047_ar_config            ON ghg_benchmark.gl_bm_alignment_results(config_id);
CREATE INDEX idx_p047_ar_pathway           ON ghg_benchmark.gl_bm_alignment_results(pathway_id);
CREATE INDEX idx_p047_ar_year              ON ghg_benchmark.gl_bm_alignment_results(reporting_year);
CREATE INDEX idx_p047_ar_alignment         ON ghg_benchmark.gl_bm_alignment_results(alignment_score);
CREATE INDEX idx_p047_ar_status            ON ghg_benchmark.gl_bm_alignment_results(status);
CREATE INDEX idx_p047_ar_calculated        ON ghg_benchmark.gl_bm_alignment_results(calculated_at DESC);
CREATE INDEX idx_p047_ar_created           ON ghg_benchmark.gl_bm_alignment_results(created_at DESC);
CREATE INDEX idx_p047_ar_provenance        ON ghg_benchmark.gl_bm_alignment_results(provenance_hash);

-- Composite: config + pathway + year for dashboard queries
CREATE INDEX idx_p047_ar_cfg_pw_year       ON ghg_benchmark.gl_bm_alignment_results(config_id, pathway_id, reporting_year);

-- Composite: tenant + config for tenant-scoped listing
CREATE INDEX idx_p047_ar_tenant_config     ON ghg_benchmark.gl_bm_alignment_results(tenant_id, config_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_pathways ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_pathway_waypoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_alignment_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_pw_tenant_isolation
    ON ghg_benchmark.gl_bm_pathways
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_pw_service_bypass
    ON ghg_benchmark.gl_bm_pathways
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Pathway waypoints inherit access via pathway FK; service bypass only
CREATE POLICY p047_wpw_service_bypass
    ON ghg_benchmark.gl_bm_pathway_waypoints
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_ar_tenant_isolation
    ON ghg_benchmark.gl_bm_alignment_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_ar_service_bypass
    ON ghg_benchmark.gl_bm_alignment_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_pathways TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_pathway_waypoints TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_alignment_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_benchmark.gl_bm_pathways IS
    'Decarbonisation pathway definitions from IEA NZE, IPCC AR6, SBTi SDA, OECM, TPI CP, and CRREM with temperature targets and sector scope.';
COMMENT ON TABLE ghg_benchmark.gl_bm_pathway_waypoints IS
    'Annual intensity/emissions milestones along a pathway with interpolation flagging and source provenance.';
COMMENT ON TABLE ghg_benchmark.gl_bm_alignment_results IS
    'Alignment analysis results: gap to pathway, years to convergence, alignment score, and overshoot year detection.';

COMMENT ON COLUMN ghg_benchmark.gl_bm_pathways.pathway_type IS 'IEA_NZE (Net Zero Emissions), IPCC_AR6_C1/C2/C3 (IPCC categories), SBTI_SDA (Sectoral Decarbonisation Approach), OECM (One Earth Climate Model), TPI_CP (Transition Pathway Initiative Carbon Performance), CRREM (Carbon Risk Real Estate Monitor).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_pathways.temperature_target IS 'Target temperature outcome in degrees Celsius (e.g., 1.5, 1.8, 2.0).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_pathways.base_year IS 'Pathway base year from which decarbonisation trajectory begins.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_pathways.target_year IS 'Pathway target year for net-zero or specified reduction milestone.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_pathway_waypoints.is_interpolated IS 'True for waypoints calculated by linear/spline interpolation between source data points.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_pathway_waypoints.source_note IS 'Provenance reference for the data point, e.g. "IEA WEO 2023 Table 3.4" or "IPCC AR6 WGIII SPM.1".';
COMMENT ON COLUMN ghg_benchmark.gl_bm_alignment_results.gap_absolute IS 'org_intensity - pathway_intensity. Positive means above pathway (lagging), negative means below (leading).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_alignment_results.gap_percentage IS 'Percentage gap: (org_intensity - pathway_intensity) / pathway_intensity * 100.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_alignment_results.years_to_convergence IS 'Estimated years at current CARR before org reaches the pathway. NULL if diverging.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_alignment_results.alignment_score IS 'Normalised alignment score (0 = fully misaligned, 1 = fully aligned or better than pathway).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_alignment_results.overshoot_year IS 'Year when cumulative emissions are projected to exceed the pathway carbon budget. NULL if within budget.';

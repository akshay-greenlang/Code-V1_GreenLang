-- =============================================================================
-- V388: PACK-047 GHG Emissions Benchmark Pack - Normalisation
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates normalisation run management and normalised data storage tables.
-- Normalisation runs harmonise peer data across different scopes, GWP
-- versions, currencies, and reporting periods to enable like-for-like
-- comparison. Each run records the steps applied (GWP conversion, currency
-- PPP adjustment, scope harmonisation, period alignment) and tracks how
-- many records were processed vs adjusted. Normalised data stores the
-- before/after values per peer with adjustment details.
--
-- Tables (2):
--   1. ghg_benchmark.gl_bm_normalisation_runs
--   2. ghg_benchmark.gl_bm_normalised_data
--
-- Also includes: indexes, RLS, comments.
-- Previous: V387__pack047_peer_groups.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_normalisation_runs
-- =============================================================================
-- Normalisation run metadata. Each run converts peer data from heterogeneous
-- formats to a common basis for fair comparison. Target parameters define
-- the harmonisation goals: target scope (which scopes to include), target
-- GWP version (AR4/AR5/AR6), target currency (with PPP adjustment), and
-- target period (fiscal year alignment). Steps applied are tracked as JSON
-- for transparency and reproducibility.

CREATE TABLE ghg_benchmark.gl_bm_normalisation_runs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    peer_group_id               UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_peer_groups(id) ON DELETE CASCADE,
    target_scope                VARCHAR(50)     NOT NULL,
    target_gwp                  VARCHAR(10)     NOT NULL DEFAULT 'AR6',
    target_currency             VARCHAR(3)      NOT NULL DEFAULT 'EUR',
    target_period               INTEGER,
    steps_applied               JSONB           NOT NULL DEFAULT '[]',
    records_processed           INTEGER         NOT NULL DEFAULT 0,
    records_adjusted            INTEGER         NOT NULL DEFAULT 0,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'RUNNING',
    error_message               TEXT,
    processing_time_ms          INTEGER,
    run_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_nr_scope CHECK (
        target_scope IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p047_nr_gwp CHECK (
        target_gwp IN ('AR4', 'AR5', 'AR6')
    ),
    CONSTRAINT chk_p047_nr_currency CHECK (
        LENGTH(target_currency) = 3
    ),
    CONSTRAINT chk_p047_nr_period CHECK (
        target_period IS NULL OR (target_period >= 2000 AND target_period <= 2100)
    ),
    CONSTRAINT chk_p047_nr_records CHECK (
        records_processed >= 0 AND records_adjusted >= 0
    ),
    CONSTRAINT chk_p047_nr_adjusted_lte CHECK (
        records_adjusted <= records_processed
    ),
    CONSTRAINT chk_p047_nr_status CHECK (
        status IN ('RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),
    CONSTRAINT chk_p047_nr_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_nr_tenant            ON ghg_benchmark.gl_bm_normalisation_runs(tenant_id);
CREATE INDEX idx_p047_nr_config            ON ghg_benchmark.gl_bm_normalisation_runs(config_id);
CREATE INDEX idx_p047_nr_peer_group        ON ghg_benchmark.gl_bm_normalisation_runs(peer_group_id);
CREATE INDEX idx_p047_nr_scope             ON ghg_benchmark.gl_bm_normalisation_runs(target_scope);
CREATE INDEX idx_p047_nr_gwp               ON ghg_benchmark.gl_bm_normalisation_runs(target_gwp);
CREATE INDEX idx_p047_nr_status            ON ghg_benchmark.gl_bm_normalisation_runs(status);
CREATE INDEX idx_p047_nr_run_at            ON ghg_benchmark.gl_bm_normalisation_runs(run_at DESC);
CREATE INDEX idx_p047_nr_created           ON ghg_benchmark.gl_bm_normalisation_runs(created_at DESC);
CREATE INDEX idx_p047_nr_steps             ON ghg_benchmark.gl_bm_normalisation_runs USING GIN(steps_applied);

-- Composite: config + peer group for lookup
CREATE INDEX idx_p047_nr_config_pg         ON ghg_benchmark.gl_bm_normalisation_runs(config_id, peer_group_id);

-- Composite: tenant + status for operational queries
CREATE INDEX idx_p047_nr_tenant_status     ON ghg_benchmark.gl_bm_normalisation_runs(tenant_id, status);

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_normalised_data
-- =============================================================================
-- Normalised emissions data per peer entity. Stores original and normalised
-- values for scope 1, scope 2, scope 3, total emissions, and intensity.
-- Adjustments applied (GWP conversion factors, currency PPP ratios, scope
-- harmonisation flags) are tracked as JSON for complete transparency.
-- Quality downgrade flag indicates when normalisation increased uncertainty.

CREATE TABLE ghg_benchmark.gl_bm_normalised_data (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    normalisation_run_id        UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_normalisation_runs(id) ON DELETE CASCADE,
    peer_definition_id          UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_peer_definitions(id) ON DELETE CASCADE,
    original_scope1             NUMERIC(20,6),
    normalised_scope1           NUMERIC(20,6),
    original_scope2             NUMERIC(20,6),
    normalised_scope2           NUMERIC(20,6),
    original_scope3             NUMERIC(20,6),
    normalised_scope3           NUMERIC(20,6),
    normalised_total            NUMERIC(20,6),
    normalised_intensity        NUMERIC(20,10),
    adjustments_applied         JSONB           NOT NULL DEFAULT '{}',
    quality_downgrade           BOOLEAN         NOT NULL DEFAULT false,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_nd_orig_s1 CHECK (
        original_scope1 IS NULL OR original_scope1 >= 0
    ),
    CONSTRAINT chk_p047_nd_norm_s1 CHECK (
        normalised_scope1 IS NULL OR normalised_scope1 >= 0
    ),
    CONSTRAINT chk_p047_nd_orig_s2 CHECK (
        original_scope2 IS NULL OR original_scope2 >= 0
    ),
    CONSTRAINT chk_p047_nd_norm_s2 CHECK (
        normalised_scope2 IS NULL OR normalised_scope2 >= 0
    ),
    CONSTRAINT chk_p047_nd_orig_s3 CHECK (
        original_scope3 IS NULL OR original_scope3 >= 0
    ),
    CONSTRAINT chk_p047_nd_norm_s3 CHECK (
        normalised_scope3 IS NULL OR normalised_scope3 >= 0
    ),
    CONSTRAINT chk_p047_nd_total CHECK (
        normalised_total IS NULL OR normalised_total >= 0
    ),
    CONSTRAINT chk_p047_nd_intensity CHECK (
        normalised_intensity IS NULL OR normalised_intensity >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_nd_tenant            ON ghg_benchmark.gl_bm_normalised_data(tenant_id);
CREATE INDEX idx_p047_nd_norm_run          ON ghg_benchmark.gl_bm_normalised_data(normalisation_run_id);
CREATE INDEX idx_p047_nd_peer_def          ON ghg_benchmark.gl_bm_normalised_data(peer_definition_id);
CREATE INDEX idx_p047_nd_downgrade         ON ghg_benchmark.gl_bm_normalised_data(quality_downgrade) WHERE quality_downgrade = true;
CREATE INDEX idx_p047_nd_created           ON ghg_benchmark.gl_bm_normalised_data(created_at DESC);
CREATE INDEX idx_p047_nd_adjustments       ON ghg_benchmark.gl_bm_normalised_data USING GIN(adjustments_applied);

-- Composite: normalisation run + peer def for batch lookup
CREATE INDEX idx_p047_nd_run_peer          ON ghg_benchmark.gl_bm_normalised_data(normalisation_run_id, peer_definition_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_normalisation_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_normalised_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_nr_tenant_isolation
    ON ghg_benchmark.gl_bm_normalisation_runs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_nr_service_bypass
    ON ghg_benchmark.gl_bm_normalisation_runs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_nd_tenant_isolation
    ON ghg_benchmark.gl_bm_normalised_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_nd_service_bypass
    ON ghg_benchmark.gl_bm_normalised_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_normalisation_runs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_normalised_data TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_benchmark.gl_bm_normalisation_runs IS
    'Normalisation run metadata tracking harmonisation of peer data across scopes, GWP versions, currencies, and periods.';
COMMENT ON TABLE ghg_benchmark.gl_bm_normalised_data IS
    'Normalised emissions data per peer entity with original and adjusted values and full adjustment transparency.';

COMMENT ON COLUMN ghg_benchmark.gl_bm_normalisation_runs.target_scope IS 'Target scope inclusion for normalisation: harmonise all peers to this scope boundary.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_normalisation_runs.target_gwp IS 'Target GWP version for harmonisation. Peers using different AR versions are converted.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_normalisation_runs.target_currency IS 'Target currency for PPP-adjusted financial denominators (ISO 4217 3-letter code).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_normalisation_runs.steps_applied IS 'JSON array of normalisation steps: [{"step": "gwp_conversion", "from": "AR5", "to": "AR6", "factor": 1.028}].';
COMMENT ON COLUMN ghg_benchmark.gl_bm_normalised_data.adjustments_applied IS 'JSON of adjustments: {"gwp_factor": 1.028, "currency_ppp": 0.95, "scope_harmonised": true, "period_prorated": false}.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_normalised_data.quality_downgrade IS 'True when normalisation introduced additional uncertainty (e.g., scope estimation, proxy currency conversion).';

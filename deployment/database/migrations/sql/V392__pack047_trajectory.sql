-- =============================================================================
-- V392: PACK-047 GHG Emissions Benchmark Pack - Trajectory Analysis
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for emissions trajectory analysis and peer comparison.
-- Trajectories model an entity's historical emissions trend using compound
-- annual reduction rate (CARR), acceleration/deceleration detection, and
-- structural break identification. Trajectory comparisons rank entities
-- against peer groups on percentile rank, convergence rate, and gap-to-
-- median with trend direction (converging, diverging, stable).
--
-- Tables (2):
--   1. ghg_benchmark.gl_bm_trajectories
--   2. ghg_benchmark.gl_bm_trajectory_comparisons
--
-- Also includes: indexes, RLS, comments.
-- Previous: V391__pack047_itr.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_trajectories
-- =============================================================================
-- Entity emissions trajectory models. Each trajectory covers a start-to-end
-- year range and calculates the CARR (compound annual reduction rate) and
-- acceleration factor (second derivative of emissions trajectory). Structural
-- breaks detect significant changes in the reduction rate (e.g., due to
-- M&A, technology shifts, policy changes) stored as JSON with year, type,
-- and magnitude.

CREATE TABLE ghg_benchmark.gl_bm_trajectories (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    entity_name                 VARCHAR(255)    NOT NULL,
    entity_identifier           VARCHAR(100),
    start_year                  INTEGER         NOT NULL,
    end_year                    INTEGER         NOT NULL,
    data_points                 INTEGER         NOT NULL DEFAULT 0,
    carr                        NUMERIC(8,6)    NOT NULL,
    acceleration                NUMERIC(8,6),
    trend_direction             VARCHAR(20)     NOT NULL DEFAULT 'STABLE',
    trend_significance_p        NUMERIC(10,6),
    structural_breaks           JSONB           DEFAULT '[]',
    series_data                 JSONB           DEFAULT '[]',
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_tr_start CHECK (
        start_year >= 2000 AND start_year <= 2100
    ),
    CONSTRAINT chk_p047_tr_end CHECK (
        end_year > start_year AND end_year <= 2100
    ),
    CONSTRAINT chk_p047_tr_data_points CHECK (
        data_points >= 0
    ),
    CONSTRAINT chk_p047_tr_trend CHECK (
        trend_direction IN ('DECREASING', 'INCREASING', 'STABLE', 'INSUFFICIENT_DATA')
    ),
    CONSTRAINT chk_p047_tr_significance CHECK (
        trend_significance_p IS NULL OR (trend_significance_p >= 0 AND trend_significance_p <= 1)
    ),
    CONSTRAINT chk_p047_tr_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_tr_tenant            ON ghg_benchmark.gl_bm_trajectories(tenant_id);
CREATE INDEX idx_p047_tr_config            ON ghg_benchmark.gl_bm_trajectories(config_id);
CREATE INDEX idx_p047_tr_entity            ON ghg_benchmark.gl_bm_trajectories(entity_name);
CREATE INDEX idx_p047_tr_entity_id         ON ghg_benchmark.gl_bm_trajectories(entity_identifier);
CREATE INDEX idx_p047_tr_start             ON ghg_benchmark.gl_bm_trajectories(start_year);
CREATE INDEX idx_p047_tr_end               ON ghg_benchmark.gl_bm_trajectories(end_year);
CREATE INDEX idx_p047_tr_carr              ON ghg_benchmark.gl_bm_trajectories(carr);
CREATE INDEX idx_p047_tr_trend             ON ghg_benchmark.gl_bm_trajectories(trend_direction);
CREATE INDEX idx_p047_tr_calculated        ON ghg_benchmark.gl_bm_trajectories(calculated_at DESC);
CREATE INDEX idx_p047_tr_created           ON ghg_benchmark.gl_bm_trajectories(created_at DESC);
CREATE INDEX idx_p047_tr_provenance        ON ghg_benchmark.gl_bm_trajectories(provenance_hash);
CREATE INDEX idx_p047_tr_breaks            ON ghg_benchmark.gl_bm_trajectories USING GIN(structural_breaks);

-- Composite: config + entity for entity-level lookup
CREATE INDEX idx_p047_tr_config_entity     ON ghg_benchmark.gl_bm_trajectories(config_id, entity_name);

-- Composite: tenant + trend for filtered analysis
CREATE INDEX idx_p047_tr_tenant_trend      ON ghg_benchmark.gl_bm_trajectories(tenant_id, trend_direction);

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_trajectory_comparisons
-- =============================================================================
-- Trajectory comparisons rank an entity's emissions trajectory against a
-- peer group. Calculates percentile rank on CARR, rate rank (ordinal
-- position), convergence rate (how fast the entity approaches the peer
-- median), gap-to-median (absolute difference from peer group median CARR),
-- and gap trend direction (converging, diverging, or stable over time).

CREATE TABLE ghg_benchmark.gl_bm_trajectory_comparisons (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    trajectory_id               UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_trajectories(id) ON DELETE CASCADE,
    peer_group_id               UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_peer_groups(id) ON DELETE CASCADE,
    percentile_rank             NUMERIC(5,2)    NOT NULL,
    rate_rank                   INTEGER         NOT NULL,
    convergence_rate            NUMERIC(8,6),
    gap_to_median               NUMERIC(20,10),
    gap_trend                   TEXT            NOT NULL DEFAULT 'STABLE',
    peer_median_carr            NUMERIC(8,6),
    peer_p25_carr               NUMERIC(8,6),
    peer_p75_carr               NUMERIC(8,6),
    peer_best_carr              NUMERIC(8,6),
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    compared_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_tc_percentile CHECK (
        percentile_rank >= 0 AND percentile_rank <= 100
    ),
    CONSTRAINT chk_p047_tc_rate_rank CHECK (
        rate_rank >= 1
    ),
    CONSTRAINT chk_p047_tc_gap_trend CHECK (
        gap_trend IN ('CONVERGING', 'DIVERGING', 'STABLE')
    ),
    CONSTRAINT chk_p047_tc_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_tc_tenant            ON ghg_benchmark.gl_bm_trajectory_comparisons(tenant_id);
CREATE INDEX idx_p047_tc_trajectory        ON ghg_benchmark.gl_bm_trajectory_comparisons(trajectory_id);
CREATE INDEX idx_p047_tc_peer_group        ON ghg_benchmark.gl_bm_trajectory_comparisons(peer_group_id);
CREATE INDEX idx_p047_tc_percentile        ON ghg_benchmark.gl_bm_trajectory_comparisons(percentile_rank);
CREATE INDEX idx_p047_tc_rate_rank         ON ghg_benchmark.gl_bm_trajectory_comparisons(rate_rank);
CREATE INDEX idx_p047_tc_gap_trend         ON ghg_benchmark.gl_bm_trajectory_comparisons(gap_trend);
CREATE INDEX idx_p047_tc_compared          ON ghg_benchmark.gl_bm_trajectory_comparisons(compared_at DESC);
CREATE INDEX idx_p047_tc_created           ON ghg_benchmark.gl_bm_trajectory_comparisons(created_at DESC);
CREATE INDEX idx_p047_tc_provenance        ON ghg_benchmark.gl_bm_trajectory_comparisons(provenance_hash);

-- Composite: trajectory + peer group for specific comparison lookup
CREATE INDEX idx_p047_tc_traj_pg           ON ghg_benchmark.gl_bm_trajectory_comparisons(trajectory_id, peer_group_id);

-- Composite: peer group + percentile for ranking queries
CREATE INDEX idx_p047_tc_pg_percentile     ON ghg_benchmark.gl_bm_trajectory_comparisons(peer_group_id, percentile_rank);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_trajectories ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_trajectory_comparisons ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_tr_tenant_isolation
    ON ghg_benchmark.gl_bm_trajectories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_tr_service_bypass
    ON ghg_benchmark.gl_bm_trajectories
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_tc_tenant_isolation
    ON ghg_benchmark.gl_bm_trajectory_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_tc_service_bypass
    ON ghg_benchmark.gl_bm_trajectory_comparisons
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_trajectories TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_trajectory_comparisons TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_benchmark.gl_bm_trajectories IS
    'Entity emissions trajectory models with CARR, acceleration, trend direction, and structural break detection.';
COMMENT ON TABLE ghg_benchmark.gl_bm_trajectory_comparisons IS
    'Trajectory comparisons ranking entity CARR against peer group with percentile, rate rank, and convergence analysis.';

COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectories.carr IS 'Compound Annual Reduction Rate as decimal (e.g., -0.045 = 4.5% annual reduction). Negative = improving.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectories.acceleration IS 'Second derivative of trajectory: positive = accelerating reduction, negative = decelerating reduction.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectories.structural_breaks IS 'JSON array of detected structural breaks: [{"year": 2022, "type": "ACQUISITION", "magnitude": 0.15, "note": "Acquired SubCo"}].';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectories.trend_direction IS 'DECREASING (improving), INCREASING (worsening), STABLE (flat), INSUFFICIENT_DATA (<3 years).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectories.trend_significance_p IS 'Mann-Kendall trend test p-value. Values < 0.05 indicate statistically significant trend.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectory_comparisons.percentile_rank IS 'Entity percentile rank on CARR within peer group (0-100). Higher = faster reduction.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectory_comparisons.rate_rank IS 'Ordinal position in peer group sorted by CARR (1 = fastest reduction).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectory_comparisons.convergence_rate IS 'Rate at which the entity is closing the gap to peer median CARR (positive = converging).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectory_comparisons.gap_to_median IS 'Difference between entity CARR and peer group median CARR. Negative = faster than median.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_trajectory_comparisons.gap_trend IS 'Direction of the gap over time: CONVERGING (gap shrinking), DIVERGING (gap growing), STABLE (constant gap).';

-- =============================================================================
-- V270: PACK-035 Energy Benchmark Pack - Peer Comparison Tables
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    005 of 010
-- Date:         March 2026
--
-- Peer group definitions and comparison results for benchmarking facilities
-- against similar buildings. Includes percentile ranking, quartile banding,
-- ENERGY STAR scoring, z-score analysis, and comparison history tracking.
--
-- Tables (3):
--   1. pack035_energy_benchmark.peer_groups
--   2. pack035_energy_benchmark.peer_comparison_results
--   3. pack035_energy_benchmark.comparison_history
--
-- Previous: V269__pack035_energy_benchmark_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.peer_groups
-- =============================================================================
-- Peer group definitions characterised by building type, climate zone,
-- floor area range, vintage, and geography. Each group stores its
-- statistical distribution (mean, median, std dev, percentiles) derived
-- from the constituent building population.

CREATE TABLE pack035_energy_benchmark.peer_groups (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID,
    name                    VARCHAR(255)    NOT NULL,
    description             TEXT,
    building_type           VARCHAR(50)     NOT NULL,
    climate_zone            VARCHAR(20),
    floor_area_min_m2       DECIMAL(12, 2),
    floor_area_max_m2       DECIMAL(12, 2),
    vintage_min             INTEGER,
    vintage_max             INTEGER,
    country_code            CHAR(2),
    region                  VARCHAR(100),
    -- Statistical distribution
    sample_size             INTEGER         NOT NULL DEFAULT 0,
    mean_eui                DECIMAL(10, 4),
    median_eui              DECIMAL(10, 4),
    std_dev                 DECIMAL(10, 4),
    p10                     DECIMAL(10, 4),
    p25                     DECIMAL(10, 4),
    p50                     DECIMAL(10, 4),
    p75                     DECIMAL(10, 4),
    p90                     DECIMAL(10, 4),
    eui_unit                VARCHAR(20)     DEFAULT 'kWh/m2',
    -- Source tracking
    source                  VARCHAR(255),
    source_year             INTEGER,
    is_custom               BOOLEAN         DEFAULT false,
    is_active               BOOLEAN         DEFAULT true,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_pg_building_type CHECK (
        building_type IN (
            'OFFICE', 'RETAIL', 'WAREHOUSE', 'MANUFACTURING', 'HEALTHCARE',
            'EDUCATION', 'DATA_CENTER', 'HOTEL', 'RESTAURANT', 'MIXED_USE',
            'RESIDENTIAL_MULTIFAMILY', 'LABORATORY', 'LIBRARY', 'WORSHIP',
            'ENTERTAINMENT', 'SPORTS', 'PARKING', 'SME'
        )
    ),
    CONSTRAINT chk_p035_pg_area_range CHECK (
        floor_area_min_m2 IS NULL OR floor_area_max_m2 IS NULL
        OR floor_area_min_m2 <= floor_area_max_m2
    ),
    CONSTRAINT chk_p035_pg_vintage_range CHECK (
        vintage_min IS NULL OR vintage_max IS NULL OR vintage_min <= vintage_max
    ),
    CONSTRAINT chk_p035_pg_sample CHECK (
        sample_size >= 0
    ),
    CONSTRAINT chk_p035_pg_mean CHECK (
        mean_eui IS NULL OR mean_eui >= 0
    ),
    CONSTRAINT chk_p035_pg_median CHECK (
        median_eui IS NULL OR median_eui >= 0
    ),
    CONSTRAINT chk_p035_pg_std_dev CHECK (
        std_dev IS NULL OR std_dev >= 0
    ),
    CONSTRAINT chk_p035_pg_percentiles CHECK (
        (p10 IS NULL OR p25 IS NULL OR p10 <= p25) AND
        (p25 IS NULL OR p50 IS NULL OR p25 <= p50) AND
        (p50 IS NULL OR p75 IS NULL OR p50 <= p75) AND
        (p75 IS NULL OR p90 IS NULL OR p75 <= p90)
    )
);

-- Indexes
CREATE INDEX idx_p035_pg_tenant          ON pack035_energy_benchmark.peer_groups(tenant_id);
CREATE INDEX idx_p035_pg_building_type   ON pack035_energy_benchmark.peer_groups(building_type);
CREATE INDEX idx_p035_pg_climate         ON pack035_energy_benchmark.peer_groups(climate_zone);
CREATE INDEX idx_p035_pg_country         ON pack035_energy_benchmark.peer_groups(country_code);
CREATE INDEX idx_p035_pg_active          ON pack035_energy_benchmark.peer_groups(is_active);
CREATE INDEX idx_p035_pg_source          ON pack035_energy_benchmark.peer_groups(source);
CREATE INDEX idx_p035_pg_custom          ON pack035_energy_benchmark.peer_groups(is_custom);
CREATE INDEX idx_p035_pg_created         ON pack035_energy_benchmark.peer_groups(created_at DESC);

-- Trigger
CREATE TRIGGER trg_p035_pg_updated
    BEFORE UPDATE ON pack035_energy_benchmark.peer_groups
    FOR EACH ROW EXECUTE FUNCTION pack035_energy_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack035_energy_benchmark.peer_comparison_results
-- =============================================================================
-- Individual facility comparison results against a peer group. Stores
-- percentile rank, quartile band, ENERGY STAR equivalent score, z-score,
-- and gap-to-target metrics for benchmarking dashboards.

CREATE TABLE pack035_energy_benchmark.peer_comparison_results (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    peer_group_id           UUID            NOT NULL REFERENCES pack035_energy_benchmark.peer_groups(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    -- Facility EUI data
    facility_eui            DECIMAL(10, 4)  NOT NULL,
    normalised_eui          DECIMAL(10, 4),
    eui_type                VARCHAR(20)     DEFAULT 'SITE',
    -- Comparison results
    percentile_rank         DECIMAL(6, 3),
    quartile_band           INTEGER,
    energy_star_score       INTEGER,
    z_score                 DECIMAL(8, 4),
    distance_to_median      DECIMAL(10, 4),
    distance_to_top_quartile DECIMAL(10, 4),
    distance_to_best_practice DECIMAL(10, 4),
    -- Improvement potential
    savings_potential_kwh_m2 DECIMAL(10, 4),
    savings_potential_pct    DECIMAL(6, 3),
    estimated_savings_eur    DECIMAL(14, 4),
    estimated_co2_savings_kg DECIMAL(14, 4),
    -- Metadata
    compared_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_pcr_eui CHECK (
        facility_eui >= 0
    ),
    CONSTRAINT chk_p035_pcr_eui_type CHECK (
        eui_type IN ('SITE', 'SOURCE', 'PRIMARY', 'WEATHER_NORMALISED')
    ),
    CONSTRAINT chk_p035_pcr_percentile CHECK (
        percentile_rank IS NULL OR (percentile_rank >= 0 AND percentile_rank <= 100)
    ),
    CONSTRAINT chk_p035_pcr_quartile CHECK (
        quartile_band IS NULL OR (quartile_band >= 1 AND quartile_band <= 4)
    ),
    CONSTRAINT chk_p035_pcr_energy_star CHECK (
        energy_star_score IS NULL OR (energy_star_score >= 1 AND energy_star_score <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p035_pcr_facility       ON pack035_energy_benchmark.peer_comparison_results(facility_id);
CREATE INDEX idx_p035_pcr_peer_group     ON pack035_energy_benchmark.peer_comparison_results(peer_group_id);
CREATE INDEX idx_p035_pcr_tenant         ON pack035_energy_benchmark.peer_comparison_results(tenant_id);
CREATE INDEX idx_p035_pcr_percentile     ON pack035_energy_benchmark.peer_comparison_results(percentile_rank);
CREATE INDEX idx_p035_pcr_quartile       ON pack035_energy_benchmark.peer_comparison_results(quartile_band);
CREATE INDEX idx_p035_pcr_energy_star    ON pack035_energy_benchmark.peer_comparison_results(energy_star_score DESC);
CREATE INDEX idx_p035_pcr_compared       ON pack035_energy_benchmark.peer_comparison_results(compared_at DESC);
CREATE INDEX idx_p035_pcr_fac_pg         ON pack035_energy_benchmark.peer_comparison_results(facility_id, peer_group_id);

-- =============================================================================
-- Table 3: pack035_energy_benchmark.comparison_history
-- =============================================================================
-- Longitudinal tracking of a facility's peer comparison results over time
-- to identify rating trends, improvements, and regressions.

CREATE TABLE pack035_energy_benchmark.comparison_history (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    comparison_date         DATE            NOT NULL,
    peer_group_id           UUID            REFERENCES pack035_energy_benchmark.peer_groups(id) ON DELETE SET NULL,
    percentile              DECIMAL(6, 3),
    quartile_band           INTEGER,
    eui_value               DECIMAL(10, 4),
    rating_change           VARCHAR(20),
    previous_percentile     DECIMAL(6, 3),
    delta_percentile        DECIMAL(6, 3),
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_ch_percentile CHECK (
        percentile IS NULL OR (percentile >= 0 AND percentile <= 100)
    ),
    CONSTRAINT chk_p035_ch_quartile CHECK (
        quartile_band IS NULL OR (quartile_band >= 1 AND quartile_band <= 4)
    ),
    CONSTRAINT chk_p035_ch_rating_change CHECK (
        rating_change IS NULL OR rating_change IN (
            'IMPROVED', 'DECLINED', 'STABLE', 'NEW_ENTRY', 'PEER_GROUP_CHANGED'
        )
    )
);

-- Indexes
CREATE INDEX idx_p035_ch_facility        ON pack035_energy_benchmark.comparison_history(facility_id);
CREATE INDEX idx_p035_ch_tenant          ON pack035_energy_benchmark.comparison_history(tenant_id);
CREATE INDEX idx_p035_ch_date            ON pack035_energy_benchmark.comparison_history(comparison_date DESC);
CREATE INDEX idx_p035_ch_peer_group      ON pack035_energy_benchmark.comparison_history(peer_group_id);
CREATE INDEX idx_p035_ch_rating          ON pack035_energy_benchmark.comparison_history(rating_change);
CREATE INDEX idx_p035_ch_fac_date        ON pack035_energy_benchmark.comparison_history(facility_id, comparison_date DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack035_energy_benchmark.peer_comparison_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.comparison_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_pcr_tenant_isolation ON pack035_energy_benchmark.peer_comparison_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_pcr_service_bypass ON pack035_energy_benchmark.peer_comparison_results
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_ch_tenant_isolation ON pack035_energy_benchmark.comparison_history
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_ch_service_bypass ON pack035_energy_benchmark.comparison_history
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.peer_groups TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.peer_comparison_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.comparison_history TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.peer_groups IS
    'Peer group definitions with statistical distributions for building-type, climate-zone, and geography-matched benchmarking.';
COMMENT ON TABLE pack035_energy_benchmark.peer_comparison_results IS
    'Individual facility comparison results against a peer group: percentile, quartile, ENERGY STAR score, z-score.';
COMMENT ON TABLE pack035_energy_benchmark.comparison_history IS
    'Longitudinal tracking of peer comparison results to identify rating trends and improvements over time.';

COMMENT ON COLUMN pack035_energy_benchmark.peer_comparison_results.percentile_rank IS
    'Percentile rank within peer group (0=worst, 100=best). A rank of 75 means the facility is better than 75% of peers.';
COMMENT ON COLUMN pack035_energy_benchmark.peer_comparison_results.quartile_band IS
    'Quartile band: 1=top (best 25%), 2=second, 3=third, 4=bottom (worst 25%).';
COMMENT ON COLUMN pack035_energy_benchmark.peer_comparison_results.energy_star_score IS
    'ENERGY STAR equivalent score (1-100) based on peer group percentile mapping.';
COMMENT ON COLUMN pack035_energy_benchmark.peer_comparison_results.z_score IS
    'Z-score = (facility_eui - peer_mean) / peer_std_dev. Negative = better than average.';

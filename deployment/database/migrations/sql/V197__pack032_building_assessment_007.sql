-- =============================================================================
-- V197: PACK-032 Building Energy Assessment - Benchmarking & Performance
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    007 of 010
-- Date:         March 2026
--
-- Creates benchmarking and performance tracking tables including EUI,
-- CRREM alignment, energy consumption records (TimescaleDB hypertable),
-- and occupancy records.
--
-- Tables (3):
--   1. pack032_building_assessment.building_benchmarks
--   2. pack032_building_assessment.energy_consumption_records (hypertable)
--   3. pack032_building_assessment.occupancy_records
--
-- Previous: V196__pack032_building_assessment_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.building_benchmarks
-- =============================================================================
-- Annual building performance benchmarks including EUI, Energy Star, DEC,
-- CRREM pathway alignment, stranding year, and peer comparison.

CREATE TABLE pack032_building_assessment.building_benchmarks (
    benchmark_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    reporting_year          INTEGER         NOT NULL,
    eui_kwh_m2              NUMERIC(10,2),
    eui_weather_normalized  NUMERIC(10,2),
    energy_star_score       INTEGER,
    dec_operational_rating  VARCHAR(5),
    crrem_target_kgco2_m2   NUMERIC(10,4),
    crrem_actual_kgco2_m2   NUMERIC(10,4),
    crrem_aligned           BOOLEAN         DEFAULT FALSE,
    stranding_year          INTEGER,
    peer_percentile         NUMERIC(6,2),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_bm_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p032_bm_eui CHECK (
        eui_kwh_m2 IS NULL OR eui_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p032_bm_eui_wn CHECK (
        eui_weather_normalized IS NULL OR eui_weather_normalized >= 0
    ),
    CONSTRAINT chk_p032_bm_energy_star CHECK (
        energy_star_score IS NULL OR (energy_star_score >= 1 AND energy_star_score <= 100)
    ),
    CONSTRAINT chk_p032_bm_dec_rating CHECK (
        dec_operational_rating IS NULL OR dec_operational_rating IN ('A', 'B', 'C', 'D', 'E', 'F', 'G')
    ),
    CONSTRAINT chk_p032_bm_crrem_target CHECK (
        crrem_target_kgco2_m2 IS NULL OR crrem_target_kgco2_m2 >= 0
    ),
    CONSTRAINT chk_p032_bm_crrem_actual CHECK (
        crrem_actual_kgco2_m2 IS NULL OR crrem_actual_kgco2_m2 >= 0
    ),
    CONSTRAINT chk_p032_bm_stranding CHECK (
        stranding_year IS NULL OR (stranding_year >= 2020 AND stranding_year <= 2100)
    ),
    CONSTRAINT chk_p032_bm_peer_pctl CHECK (
        peer_percentile IS NULL OR (peer_percentile >= 0 AND peer_percentile <= 100)
    ),
    CONSTRAINT chk_p032_bm_unique_year UNIQUE (building_id, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_bm_building ON pack032_building_assessment.building_benchmarks(building_id);
CREATE INDEX idx_p032_bm_tenant   ON pack032_building_assessment.building_benchmarks(tenant_id);
CREATE INDEX idx_p032_bm_year     ON pack032_building_assessment.building_benchmarks(reporting_year DESC);
CREATE INDEX idx_p032_bm_crrem    ON pack032_building_assessment.building_benchmarks(crrem_aligned);
CREATE INDEX idx_p032_bm_strand   ON pack032_building_assessment.building_benchmarks(stranding_year);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_bm_updated
    BEFORE UPDATE ON pack032_building_assessment.building_benchmarks
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.energy_consumption_records
-- =============================================================================
-- Time-series energy consumption data by carrier with cost and CO2 emissions.
-- Converted to TimescaleDB hypertable for efficient time-series queries.

CREATE TABLE pack032_building_assessment.energy_consumption_records (
    record_id               UUID            NOT NULL DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    period                  TIMESTAMPTZ     NOT NULL,
    energy_carrier          VARCHAR(100)    NOT NULL,
    consumption_kwh         NUMERIC(14,2)   NOT NULL,
    cost_eur                NUMERIC(14,2),
    co2_kg                  NUMERIC(14,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_ecr_consumption CHECK (
        consumption_kwh >= 0
    ),
    CONSTRAINT chk_p032_ecr_cost CHECK (
        cost_eur IS NULL OR cost_eur >= 0
    ),
    CONSTRAINT chk_p032_ecr_co2 CHECK (
        co2_kg IS NULL OR co2_kg >= 0
    ),
    CONSTRAINT chk_p032_ecr_carrier CHECK (
        energy_carrier IN ('ELECTRICITY', 'NATURAL_GAS', 'DISTRICT_HEATING', 'DISTRICT_COOLING',
                            'OIL', 'LPG', 'BIOMASS', 'SOLAR_PV', 'SOLAR_THERMAL',
                            'WIND', 'OTHER')
    )
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable(
    'pack032_building_assessment.energy_consumption_records',
    'period',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_ecr_building ON pack032_building_assessment.energy_consumption_records(building_id, period DESC);
CREATE INDEX idx_p032_ecr_tenant   ON pack032_building_assessment.energy_consumption_records(tenant_id);
CREATE INDEX idx_p032_ecr_carrier  ON pack032_building_assessment.energy_consumption_records(energy_carrier);

-- =============================================================================
-- Table 3: pack032_building_assessment.occupancy_records
-- =============================================================================
-- Occupancy tracking records for energy normalisation and benchmarking.

CREATE TABLE pack032_building_assessment.occupancy_records (
    record_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    period                  TIMESTAMPTZ     NOT NULL,
    occupancy_pct           NUMERIC(6,2),
    headcount               INTEGER,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_or_occupancy CHECK (
        occupancy_pct IS NULL OR (occupancy_pct >= 0 AND occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p032_or_headcount CHECK (
        headcount IS NULL OR headcount >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_or_building ON pack032_building_assessment.occupancy_records(building_id, period DESC);
CREATE INDEX idx_p032_or_tenant   ON pack032_building_assessment.occupancy_records(tenant_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.building_benchmarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.energy_consumption_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.occupancy_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_bm_tenant_isolation
    ON pack032_building_assessment.building_benchmarks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_bm_service_bypass
    ON pack032_building_assessment.building_benchmarks
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_ecr_tenant_isolation
    ON pack032_building_assessment.energy_consumption_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_ecr_service_bypass
    ON pack032_building_assessment.energy_consumption_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_or_tenant_isolation
    ON pack032_building_assessment.occupancy_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_or_service_bypass
    ON pack032_building_assessment.occupancy_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.building_benchmarks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.energy_consumption_records TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.occupancy_records TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.building_benchmarks IS
    'Annual building performance benchmarks with EUI, Energy Star score, CRREM alignment, stranding year, and peer percentile.';

COMMENT ON TABLE pack032_building_assessment.energy_consumption_records IS
    'Time-series energy consumption data by carrier with cost and CO2 emissions. TimescaleDB hypertable with monthly chunks.';

COMMENT ON TABLE pack032_building_assessment.occupancy_records IS
    'Occupancy tracking records for energy normalisation and benchmarking.';

COMMENT ON COLUMN pack032_building_assessment.building_benchmarks.eui_kwh_m2 IS
    'Energy Use Intensity in kWh per m2 per year.';
COMMENT ON COLUMN pack032_building_assessment.building_benchmarks.eui_weather_normalized IS
    'Weather-normalised EUI adjusted using heating/cooling degree days.';
COMMENT ON COLUMN pack032_building_assessment.building_benchmarks.crrem_target_kgco2_m2 IS
    'CRREM pathway target carbon intensity in kgCO2/m2 for the reporting year.';
COMMENT ON COLUMN pack032_building_assessment.building_benchmarks.crrem_actual_kgco2_m2 IS
    'Actual carbon intensity in kgCO2/m2 for benchmarking against CRREM pathway.';
COMMENT ON COLUMN pack032_building_assessment.building_benchmarks.stranding_year IS
    'Projected year when building exceeds CRREM decarbonisation pathway (stranding risk).';
COMMENT ON COLUMN pack032_building_assessment.building_benchmarks.peer_percentile IS
    'Percentile ranking against peer buildings in the same category and region.';
COMMENT ON COLUMN pack032_building_assessment.building_benchmarks.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';

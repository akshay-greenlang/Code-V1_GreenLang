-- =============================================================================
-- V267: PACK-035 Energy Benchmark Pack - Benchmark Database Tables
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    002 of 010
-- Date:         March 2026
--
-- Benchmark reference data from multiple sources (CIBSE TM46, ENERGY STAR,
-- DIN V 18599, BPIE). Stores source metadata, building type mappings,
-- benchmark values by performance level, and industrial sector benchmarks.
--
-- Tables (4):
--   1. pack035_energy_benchmark.benchmark_sources
--   2. pack035_energy_benchmark.benchmark_building_types
--   3. pack035_energy_benchmark.benchmark_values
--   4. pack035_energy_benchmark.sector_benchmarks
--
-- Previous: V266__pack035_energy_benchmark_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.benchmark_sources
-- =============================================================================
-- Registry of benchmark data sources with version tracking and geographic scope.
-- Sources include: CIBSE TM46 (UK), ENERGY STAR (US), DIN V 18599 (DE),
-- BPIE (EU), TABULA/EPISCOPE (EU residential), NABERS (AU).

CREATE TABLE pack035_energy_benchmark.benchmark_sources (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name             VARCHAR(255)    NOT NULL,
    source_code             VARCHAR(50)     NOT NULL UNIQUE,
    source_version          VARCHAR(50),
    publisher               VARCHAR(255),
    effective_date          DATE            NOT NULL,
    expiry_date             DATE,
    country_scope           VARCHAR(10),
    region_scope            VARCHAR(100),
    description             TEXT,
    methodology_url         TEXT,
    data_vintage_year       INTEGER,
    is_active               BOOLEAN         DEFAULT true,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_src_dates CHECK (
        expiry_date IS NULL OR expiry_date > effective_date
    ),
    CONSTRAINT chk_p035_src_vintage CHECK (
        data_vintage_year IS NULL OR (data_vintage_year >= 1990 AND data_vintage_year <= 2100)
    )
);

-- Indexes
CREATE INDEX idx_p035_src_code           ON pack035_energy_benchmark.benchmark_sources(source_code);
CREATE INDEX idx_p035_src_country        ON pack035_energy_benchmark.benchmark_sources(country_scope);
CREATE INDEX idx_p035_src_active         ON pack035_energy_benchmark.benchmark_sources(is_active);
CREATE INDEX idx_p035_src_effective      ON pack035_energy_benchmark.benchmark_sources(effective_date);

-- =============================================================================
-- Table 2: pack035_energy_benchmark.benchmark_building_types
-- =============================================================================
-- Mapping between source-specific building type codes and the GreenLang
-- canonical building type taxonomy used in facility_profiles.

CREATE TABLE pack035_energy_benchmark.benchmark_building_types (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id               UUID            NOT NULL REFERENCES pack035_energy_benchmark.benchmark_sources(id) ON DELETE CASCADE,
    source_building_type    VARCHAR(100)    NOT NULL,
    source_type_code        VARCHAR(50),
    gl_building_type        VARCHAR(50)     NOT NULL,
    description             TEXT,
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_bbt_gl_type CHECK (
        gl_building_type IN (
            'OFFICE', 'RETAIL', 'WAREHOUSE', 'MANUFACTURING', 'HEALTHCARE',
            'EDUCATION', 'DATA_CENTER', 'HOTEL', 'RESTAURANT', 'MIXED_USE',
            'RESIDENTIAL_MULTIFAMILY', 'LABORATORY', 'LIBRARY', 'WORSHIP',
            'ENTERTAINMENT', 'SPORTS', 'PARKING', 'SME'
        )
    ),
    CONSTRAINT uq_p035_bbt_source_type UNIQUE (source_id, source_building_type)
);

-- Indexes
CREATE INDEX idx_p035_bbt_source         ON pack035_energy_benchmark.benchmark_building_types(source_id);
CREATE INDEX idx_p035_bbt_gl_type        ON pack035_energy_benchmark.benchmark_building_types(gl_building_type);
CREATE INDEX idx_p035_bbt_src_type       ON pack035_energy_benchmark.benchmark_building_types(source_building_type);

-- =============================================================================
-- Table 3: pack035_energy_benchmark.benchmark_values
-- =============================================================================
-- Benchmark EUI values by source, building type, and performance level.
-- Stores electricity and fossil fuel splits, site/source/primary energy,
-- CO2 intensity, and percentile distribution data.

CREATE TABLE pack035_energy_benchmark.benchmark_values (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id               UUID            NOT NULL REFERENCES pack035_energy_benchmark.benchmark_sources(id) ON DELETE CASCADE,
    building_type_id        UUID            NOT NULL REFERENCES pack035_energy_benchmark.benchmark_building_types(id) ON DELETE CASCADE,
    benchmark_level         VARCHAR(30)     NOT NULL,
    climate_zone            VARCHAR(20),
    electricity_kwh_m2      DECIMAL(10, 2),
    fossil_fuel_kwh_m2      DECIMAL(10, 2),
    total_site_kwh_m2       DECIMAL(10, 2),
    total_source_kwh_m2     DECIMAL(10, 2),
    primary_energy_kwh_m2   DECIMAL(10, 2),
    co2_kg_m2               DECIMAL(10, 4),
    percentile_10           DECIMAL(10, 2),
    percentile_25           DECIMAL(10, 2),
    percentile_50           DECIMAL(10, 2),
    percentile_75           DECIMAL(10, 2),
    percentile_90           DECIMAL(10, 2),
    sample_size             INTEGER,
    year_of_data            INTEGER,
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_bv_level CHECK (
        benchmark_level IN ('TYPICAL', 'GOOD_PRACTICE', 'BEST_PRACTICE', 'REGULATORY_MINIMUM', 'NZEB')
    ),
    CONSTRAINT chk_p035_bv_elec CHECK (
        electricity_kwh_m2 IS NULL OR electricity_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_bv_fossil CHECK (
        fossil_fuel_kwh_m2 IS NULL OR fossil_fuel_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_bv_site CHECK (
        total_site_kwh_m2 IS NULL OR total_site_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_bv_source CHECK (
        total_source_kwh_m2 IS NULL OR total_source_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_bv_primary CHECK (
        primary_energy_kwh_m2 IS NULL OR primary_energy_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_bv_co2 CHECK (
        co2_kg_m2 IS NULL OR co2_kg_m2 >= 0
    ),
    CONSTRAINT chk_p035_bv_sample CHECK (
        sample_size IS NULL OR sample_size >= 0
    ),
    CONSTRAINT chk_p035_bv_year CHECK (
        year_of_data IS NULL OR (year_of_data >= 1990 AND year_of_data <= 2100)
    ),
    CONSTRAINT chk_p035_bv_percentiles CHECK (
        (percentile_10 IS NULL OR percentile_25 IS NULL OR percentile_10 <= percentile_25) AND
        (percentile_25 IS NULL OR percentile_50 IS NULL OR percentile_25 <= percentile_50) AND
        (percentile_50 IS NULL OR percentile_75 IS NULL OR percentile_50 <= percentile_75) AND
        (percentile_75 IS NULL OR percentile_90 IS NULL OR percentile_75 <= percentile_90)
    )
);

-- Indexes
CREATE INDEX idx_p035_bv_source          ON pack035_energy_benchmark.benchmark_values(source_id);
CREATE INDEX idx_p035_bv_btype           ON pack035_energy_benchmark.benchmark_values(building_type_id);
CREATE INDEX idx_p035_bv_level           ON pack035_energy_benchmark.benchmark_values(benchmark_level);
CREATE INDEX idx_p035_bv_climate         ON pack035_energy_benchmark.benchmark_values(climate_zone);
CREATE INDEX idx_p035_bv_site_eui        ON pack035_energy_benchmark.benchmark_values(total_site_kwh_m2);
CREATE INDEX idx_p035_bv_year            ON pack035_energy_benchmark.benchmark_values(year_of_data);

-- =============================================================================
-- Table 4: pack035_energy_benchmark.sector_benchmarks
-- =============================================================================
-- Industrial/process-level energy intensity benchmarks for sector-specific
-- comparisons (e.g., kWh per tonne of product). Used for manufacturing
-- and industrial facility types.

CREATE TABLE pack035_energy_benchmark.sector_benchmarks (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sector                  VARCHAR(100)    NOT NULL,
    sub_sector              VARCHAR(100),
    product                 VARCHAR(255),
    unit                    VARCHAR(50)     NOT NULL,
    energy_intensity_typical    DECIMAL(14, 4),
    energy_intensity_good       DECIMAL(14, 4),
    energy_intensity_best       DECIMAL(14, 4),
    co2_intensity_typical       DECIMAL(14, 6),
    co2_intensity_good          DECIMAL(14, 6),
    co2_intensity_best          DECIMAL(14, 6),
    source_reference        VARCHAR(255)    NOT NULL,
    source_year             INTEGER,
    country_scope           VARCHAR(10),
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_sb_intensity_typ CHECK (
        energy_intensity_typical IS NULL OR energy_intensity_typical >= 0
    ),
    CONSTRAINT chk_p035_sb_intensity_good CHECK (
        energy_intensity_good IS NULL OR energy_intensity_good >= 0
    ),
    CONSTRAINT chk_p035_sb_intensity_best CHECK (
        energy_intensity_best IS NULL OR energy_intensity_best >= 0
    )
);

-- Indexes
CREATE INDEX idx_p035_sb_sector          ON pack035_energy_benchmark.sector_benchmarks(sector);
CREATE INDEX idx_p035_sb_sub_sector      ON pack035_energy_benchmark.sector_benchmarks(sub_sector);
CREATE INDEX idx_p035_sb_product         ON pack035_energy_benchmark.sector_benchmarks(product);
CREATE INDEX idx_p035_sb_source          ON pack035_energy_benchmark.sector_benchmarks(source_reference);
CREATE INDEX idx_p035_sb_country         ON pack035_energy_benchmark.sector_benchmarks(country_scope);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.benchmark_sources TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.benchmark_building_types TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.benchmark_values TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.sector_benchmarks TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.benchmark_sources IS
    'Registry of benchmark data sources (CIBSE TM46, ENERGY STAR, DIN V 18599, BPIE) with version tracking and geographic scope.';
COMMENT ON TABLE pack035_energy_benchmark.benchmark_building_types IS
    'Mapping between source-specific building type codes and the GreenLang canonical building type taxonomy.';
COMMENT ON TABLE pack035_energy_benchmark.benchmark_values IS
    'Benchmark EUI values by source, building type, and performance level with percentile distributions.';
COMMENT ON TABLE pack035_energy_benchmark.sector_benchmarks IS
    'Industrial/process energy intensity benchmarks per sector and product for manufacturing facility benchmarking.';

-- Seed data comments:
-- CIBSE TM46 (UK): Office general 95 kWh/m2 typical, 65 good practice
-- ENERGY STAR (US): Office 50th percentile ~188 kBtu/ft2 source EUI
-- DIN V 18599 (DE): Reference building method for EnEV compliance
-- BPIE (EU): Building Performance Institute Europe cross-country data
COMMENT ON COLUMN pack035_energy_benchmark.benchmark_values.benchmark_level IS
    'Performance tier: TYPICAL (median stock), GOOD_PRACTICE (top quartile), BEST_PRACTICE (top decile), REGULATORY_MINIMUM, NZEB (near-zero energy building).';
COMMENT ON COLUMN pack035_energy_benchmark.benchmark_values.total_site_kwh_m2 IS
    'Total site (delivered) energy use intensity in kWh per square metre per year.';
COMMENT ON COLUMN pack035_energy_benchmark.benchmark_values.total_source_kwh_m2 IS
    'Total source (primary) energy use intensity adjusted for grid losses and generation efficiency.';
COMMENT ON COLUMN pack035_energy_benchmark.sector_benchmarks.unit IS
    'Normalisation unit for intensity metric, e.g. kWh/tonne, kWh/unit, kWh/m3.';

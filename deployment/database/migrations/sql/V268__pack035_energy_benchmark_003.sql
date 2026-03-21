-- =============================================================================
-- V268: PACK-035 Energy Benchmark Pack - EUI Calculation Tables
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    003 of 010
-- Date:         March 2026
--
-- Energy consumption records, EUI (Energy Use Intensity) calculation results,
-- and energy carrier breakdown tables. Supports site, source, and primary
-- energy EUI with normalisation options (occupancy, weather, cost).
--
-- Tables (3):
--   1. pack035_energy_benchmark.energy_consumption_records
--   2. pack035_energy_benchmark.eui_calculations
--   3. pack035_energy_benchmark.energy_carrier_breakdown
--
-- Previous: V267__pack035_energy_benchmark_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.energy_consumption_records
-- =============================================================================
-- Raw energy consumption readings from meters, invoices, or estimates.
-- Supports multiple granularities (monthly, quarterly, annual) and
-- data quality scoring for provenance.

CREATE TABLE pack035_energy_benchmark.energy_consumption_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    meter_id                UUID            REFERENCES pack035_energy_benchmark.metering_points(id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    consumption_value       DECIMAL(14, 4)  NOT NULL,
    unit                    VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    energy_carrier          VARCHAR(50),
    is_estimated            BOOLEAN         DEFAULT false,
    estimation_method       VARCHAR(100),
    data_quality_score      DECIMAL(3, 2),
    source_document         VARCHAR(255),
    source_document_type    VARCHAR(50),
    cost_eur                DECIMAL(14, 4),
    cost_currency           VARCHAR(3)      DEFAULT 'EUR',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_ecr_dates CHECK (
        period_end > period_start
    ),
    CONSTRAINT chk_p035_ecr_consumption CHECK (
        consumption_value >= 0
    ),
    CONSTRAINT chk_p035_ecr_unit CHECK (
        unit IN ('kWh', 'MWh', 'GJ', 'therm', 'kBtu', 'MJ', 'm3', 'litre', 'kg', 'tonne')
    ),
    CONSTRAINT chk_p035_ecr_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 1)
    ),
    CONSTRAINT chk_p035_ecr_carrier CHECK (
        energy_carrier IS NULL OR energy_carrier IN (
            'ELECTRICITY', 'NATURAL_GAS', 'FUEL_OIL', 'LPG',
            'DISTRICT_HEATING', 'DISTRICT_COOLING', 'BIOMASS',
            'SOLAR_THERMAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p035_ecr_cost CHECK (
        cost_eur IS NULL OR cost_eur >= 0
    ),
    CONSTRAINT chk_p035_ecr_doc_type CHECK (
        source_document_type IS NULL OR source_document_type IN (
            'INVOICE', 'METER_READING', 'ESTIMATE', 'HALF_HOURLY_DATA', 'AMR', 'MANUAL', 'API'
        )
    )
);

-- Indexes
CREATE INDEX idx_p035_ecr_facility       ON pack035_energy_benchmark.energy_consumption_records(facility_id);
CREATE INDEX idx_p035_ecr_meter          ON pack035_energy_benchmark.energy_consumption_records(meter_id);
CREATE INDEX idx_p035_ecr_tenant         ON pack035_energy_benchmark.energy_consumption_records(tenant_id);
CREATE INDEX idx_p035_ecr_period         ON pack035_energy_benchmark.energy_consumption_records(period_start, period_end);
CREATE INDEX idx_p035_ecr_carrier        ON pack035_energy_benchmark.energy_consumption_records(energy_carrier);
CREATE INDEX idx_p035_ecr_estimated      ON pack035_energy_benchmark.energy_consumption_records(is_estimated);
CREATE INDEX idx_p035_ecr_quality        ON pack035_energy_benchmark.energy_consumption_records(data_quality_score);
CREATE INDEX idx_p035_ecr_created        ON pack035_energy_benchmark.energy_consumption_records(created_at DESC);

-- =============================================================================
-- Table 2: pack035_energy_benchmark.eui_calculations
-- =============================================================================
-- Calculated Energy Use Intensity (EUI) results per facility and period.
-- Stores site, source, and primary energy EUI along with normalised
-- variants (occupancy-adjusted, weather-normalised, cost-normalised).
-- Each calculation is signed with a provenance hash for audit trail.

CREATE TABLE pack035_energy_benchmark.eui_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    calculation_period      VARCHAR(20)     NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    accounting_boundary     VARCHAR(30)     NOT NULL DEFAULT 'WHOLE_BUILDING',
    floor_area_type         VARCHAR(30)     NOT NULL DEFAULT 'GIA',
    floor_area_m2           DECIMAL(12, 2)  NOT NULL,
    -- Energy totals
    site_energy_kwh         DECIMAL(14, 2),
    source_energy_kwh       DECIMAL(14, 2),
    primary_energy_kwh      DECIMAL(14, 2),
    -- EUI values
    site_eui_kwh_m2         DECIMAL(10, 4),
    source_eui_kwh_m2       DECIMAL(10, 4),
    primary_eui_kwh_m2      DECIMAL(10, 4),
    -- Normalised EUI variants
    cost_normalised_eui     DECIMAL(10, 4),
    occupancy_adjusted_eui  DECIMAL(10, 4),
    weather_normalised_eui  DECIMAL(10, 4),
    -- CO2 metrics
    co2_total_kg            DECIMAL(14, 4),
    co2_intensity_kg_m2     DECIMAL(10, 4),
    -- Metadata
    calculation_method      VARCHAR(100),
    data_completeness_pct   DECIMAL(5, 2),
    provenance_hash         VARCHAR(64)     NOT NULL,
    calculated_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_eui_dates CHECK (
        period_end > period_start
    ),
    CONSTRAINT chk_p035_eui_period CHECK (
        calculation_period IN ('MONTHLY', 'QUARTERLY', 'ANNUAL', 'ROLLING_12M', 'CUSTOM')
    ),
    CONSTRAINT chk_p035_eui_boundary CHECK (
        accounting_boundary IN ('WHOLE_BUILDING', 'LANDLORD', 'TENANT', 'COMMON_AREAS', 'PROCESS_ONLY')
    ),
    CONSTRAINT chk_p035_eui_area_type CHECK (
        floor_area_type IN ('GIA', 'NIA', 'GLA', 'TFA', 'CONDITIONED')
    ),
    CONSTRAINT chk_p035_eui_area CHECK (
        floor_area_m2 > 0
    ),
    CONSTRAINT chk_p035_eui_site CHECK (
        site_energy_kwh IS NULL OR site_energy_kwh >= 0
    ),
    CONSTRAINT chk_p035_eui_source CHECK (
        source_energy_kwh IS NULL OR source_energy_kwh >= 0
    ),
    CONSTRAINT chk_p035_eui_primary CHECK (
        primary_energy_kwh IS NULL OR primary_energy_kwh >= 0
    ),
    CONSTRAINT chk_p035_eui_site_eui CHECK (
        site_eui_kwh_m2 IS NULL OR site_eui_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_eui_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p035_eui_facility       ON pack035_energy_benchmark.eui_calculations(facility_id);
CREATE INDEX idx_p035_eui_tenant         ON pack035_energy_benchmark.eui_calculations(tenant_id);
CREATE INDEX idx_p035_eui_period         ON pack035_energy_benchmark.eui_calculations(calculation_period);
CREATE INDEX idx_p035_eui_dates          ON pack035_energy_benchmark.eui_calculations(period_start, period_end);
CREATE INDEX idx_p035_eui_site_val       ON pack035_energy_benchmark.eui_calculations(site_eui_kwh_m2);
CREATE INDEX idx_p035_eui_weather_norm   ON pack035_energy_benchmark.eui_calculations(weather_normalised_eui);
CREATE INDEX idx_p035_eui_calculated     ON pack035_energy_benchmark.eui_calculations(calculated_at DESC);

-- =============================================================================
-- Table 3: pack035_energy_benchmark.energy_carrier_breakdown
-- =============================================================================
-- Per-carrier energy breakdown linked to an EUI calculation, showing
-- consumption, percentage share, cost, emission factor, and CO2.

CREATE TABLE pack035_energy_benchmark.energy_carrier_breakdown (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    eui_calculation_id      UUID            NOT NULL REFERENCES pack035_energy_benchmark.eui_calculations(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    energy_carrier          VARCHAR(50)     NOT NULL,
    consumption_kwh         DECIMAL(14, 4)  NOT NULL,
    percentage_of_total     DECIMAL(6, 3),
    source_energy_kwh       DECIMAL(14, 4),
    primary_energy_kwh      DECIMAL(14, 4),
    primary_energy_factor   DECIMAL(6, 4),
    cost_eur                DECIMAL(14, 4),
    emission_factor_kg_co2_kwh DECIMAL(10, 6),
    co2_kg                  DECIMAL(14, 4),
    is_renewable            BOOLEAN         DEFAULT false,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_ecb_carrier CHECK (
        energy_carrier IN (
            'ELECTRICITY', 'NATURAL_GAS', 'FUEL_OIL', 'LPG',
            'DISTRICT_HEATING', 'DISTRICT_COOLING', 'BIOMASS',
            'SOLAR_THERMAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p035_ecb_consumption CHECK (
        consumption_kwh >= 0
    ),
    CONSTRAINT chk_p035_ecb_pct CHECK (
        percentage_of_total IS NULL OR (percentage_of_total >= 0 AND percentage_of_total <= 100)
    ),
    CONSTRAINT chk_p035_ecb_cost CHECK (
        cost_eur IS NULL OR cost_eur >= 0
    ),
    CONSTRAINT chk_p035_ecb_ef CHECK (
        emission_factor_kg_co2_kwh IS NULL OR emission_factor_kg_co2_kwh >= 0
    ),
    CONSTRAINT chk_p035_ecb_co2 CHECK (
        co2_kg IS NULL OR co2_kg >= 0
    )
);

-- Indexes
CREATE INDEX idx_p035_ecb_calc           ON pack035_energy_benchmark.energy_carrier_breakdown(eui_calculation_id);
CREATE INDEX idx_p035_ecb_tenant         ON pack035_energy_benchmark.energy_carrier_breakdown(tenant_id);
CREATE INDEX idx_p035_ecb_carrier        ON pack035_energy_benchmark.energy_carrier_breakdown(energy_carrier);
CREATE INDEX idx_p035_ecb_renewable      ON pack035_energy_benchmark.energy_carrier_breakdown(is_renewable);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack035_energy_benchmark.energy_consumption_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.eui_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.energy_carrier_breakdown ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_ecr_tenant_isolation ON pack035_energy_benchmark.energy_consumption_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_ecr_service_bypass ON pack035_energy_benchmark.energy_consumption_records
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_eui_tenant_isolation ON pack035_energy_benchmark.eui_calculations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_eui_service_bypass ON pack035_energy_benchmark.eui_calculations
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_ecb_tenant_isolation ON pack035_energy_benchmark.energy_carrier_breakdown
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_ecb_service_bypass ON pack035_energy_benchmark.energy_carrier_breakdown
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.energy_consumption_records TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.eui_calculations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.energy_carrier_breakdown TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.energy_consumption_records IS
    'Raw energy consumption readings from meters, invoices, or estimates with data quality scoring.';
COMMENT ON TABLE pack035_energy_benchmark.eui_calculations IS
    'Calculated EUI results per facility and period: site, source, primary energy with normalised variants.';
COMMENT ON TABLE pack035_energy_benchmark.energy_carrier_breakdown IS
    'Per-carrier energy breakdown linked to an EUI calculation with cost, emission factor, and CO2.';

COMMENT ON COLUMN pack035_energy_benchmark.eui_calculations.site_eui_kwh_m2 IS
    'Site (delivered) Energy Use Intensity = site_energy_kwh / floor_area_m2.';
COMMENT ON COLUMN pack035_energy_benchmark.eui_calculations.source_eui_kwh_m2 IS
    'Source (primary) Energy Use Intensity adjusting for grid losses (ENERGY STAR method).';
COMMENT ON COLUMN pack035_energy_benchmark.eui_calculations.weather_normalised_eui IS
    'EUI normalised using degree-day regression to TMY weather conditions.';
COMMENT ON COLUMN pack035_energy_benchmark.eui_calculations.provenance_hash IS
    'SHA-256 hash of input data and calculation parameters for audit provenance.';
COMMENT ON COLUMN pack035_energy_benchmark.energy_carrier_breakdown.primary_energy_factor IS
    'Primary energy factor used for conversion (e.g., 2.5 for electricity, 1.1 for gas per EN 15603).';

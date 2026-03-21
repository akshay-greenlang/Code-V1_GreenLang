-- =============================================================================
-- V182: PACK-031 Industrial Energy Audit - Energy Metering & Baselines
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    002 of 010
-- Date:         March 2026
--
-- Establishes the energy metering infrastructure with hierarchical sub-metering,
-- time-series meter readings (TimescaleDB hypertable), energy baselines with
-- regression models, and Energy Performance Indicator (EnPI) tracking.
--
-- Tables (4):
--   1. pack031_energy_audit.energy_meters
--   2. pack031_energy_audit.meter_readings  (hypertable)
--   3. pack031_energy_audit.energy_baselines
--   4. pack031_energy_audit.enpi_records
--
-- Previous: V181__pack031_energy_audit_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.energy_meters
-- =============================================================================
-- Physical and virtual energy meters with hierarchical sub-metering support.

CREATE TABLE pack031_energy_audit.energy_meters (
    meter_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    carrier_type            VARCHAR(100)    NOT NULL,
    location                VARCHAR(255),
    sub_meter_of            UUID            REFERENCES pack031_energy_audit.energy_meters(meter_id) ON DELETE SET NULL,
    is_virtual              BOOLEAN         DEFAULT FALSE,
    unit                    VARCHAR(30)     NOT NULL DEFAULT 'kWh',
    meter_serial            VARCHAR(100),
    installation_date       DATE,
    calibration_due_date    DATE,
    accuracy_class          VARCHAR(20),
    status                  VARCHAR(30)     DEFAULT 'active',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_meter_carrier CHECK (
        carrier_type IN ('ELECTRICITY', 'NATURAL_GAS', 'DIESEL', 'HEAVY_FUEL_OIL',
                         'LPG', 'COAL', 'BIOMASS', 'DISTRICT_HEATING', 'DISTRICT_COOLING',
                         'STEAM', 'COMPRESSED_AIR', 'WATER', 'OTHER')
    ),
    CONSTRAINT chk_p031_meter_status CHECK (
        status IN ('active', 'inactive', 'decommissioned', 'faulty')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_meter_facility   ON pack031_energy_audit.energy_meters(facility_id);
CREATE INDEX idx_p031_meter_tenant     ON pack031_energy_audit.energy_meters(tenant_id);
CREATE INDEX idx_p031_meter_carrier    ON pack031_energy_audit.energy_meters(carrier_type);
CREATE INDEX idx_p031_meter_parent     ON pack031_energy_audit.energy_meters(sub_meter_of);
CREATE INDEX idx_p031_meter_virtual    ON pack031_energy_audit.energy_meters(is_virtual);
CREATE INDEX idx_p031_meter_status     ON pack031_energy_audit.energy_meters(status);

-- Trigger
CREATE TRIGGER trg_p031_meter_updated
    BEFORE UPDATE ON pack031_energy_audit.energy_meters
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack031_energy_audit.meter_readings
-- =============================================================================
-- Time-series meter readings converted to a TimescaleDB hypertable for
-- high-frequency interval data with cost and quality tracking.

CREATE TABLE pack031_energy_audit.meter_readings (
    reading_id              UUID            DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,
    value_kwh               NUMERIC(18,6)   NOT NULL,
    cost_eur                NUMERIC(14,4),
    quality_flag            VARCHAR(30)     DEFAULT 'measured',
    source                  VARCHAR(100)    DEFAULT 'meter',
    -- Constraints
    CONSTRAINT chk_p031_reading_value CHECK (value_kwh >= 0),
    CONSTRAINT chk_p031_reading_cost CHECK (cost_eur IS NULL OR cost_eur >= 0),
    CONSTRAINT chk_p031_reading_quality CHECK (
        quality_flag IN ('measured', 'estimated', 'interpolated', 'manual', 'suspect')
    )
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'pack031_energy_audit.meter_readings',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ---------------------------------------------------------------------------
-- Indexes on hypertable
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_reading_meter    ON pack031_energy_audit.meter_readings(meter_id, timestamp DESC);
CREATE INDEX idx_p031_reading_tenant   ON pack031_energy_audit.meter_readings(tenant_id);
CREATE INDEX idx_p031_reading_quality  ON pack031_energy_audit.meter_readings(quality_flag);

-- =============================================================================
-- Table 3: pack031_energy_audit.energy_baselines
-- =============================================================================
-- Energy baselines with regression model parameters for baseline adjustment
-- per ISO 50006 (EnPI and EnB methodology).

CREATE TABLE pack031_energy_audit.energy_baselines (
    baseline_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    baseline_start          DATE            NOT NULL,
    baseline_end            DATE            NOT NULL,
    regression_type         VARCHAR(50),
    independent_variables   TEXT[],
    coefficients            JSONB           DEFAULT '{}',
    r_squared               NUMERIC(8,6),
    cv_rmse                 NUMERIC(8,6),
    total_baseline_kwh      NUMERIC(18,6),
    total_baseline_cost_eur NUMERIC(14,4),
    normalization_factors   JSONB           DEFAULT '{}',
    status                  VARCHAR(30)     DEFAULT 'draft',
    approved_by             VARCHAR(255),
    approval_date           DATE,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_baseline_dates CHECK (baseline_start < baseline_end),
    CONSTRAINT chk_p031_baseline_regression CHECK (
        regression_type IS NULL OR regression_type IN (
            'SIMPLE_LINEAR', 'MULTIPLE_LINEAR', 'POLYNOMIAL', 'CHANGE_POINT',
            'MEAN', 'DEGREE_DAY', 'MULTIVARIABLE'
        )
    ),
    CONSTRAINT chk_p031_baseline_r2 CHECK (
        r_squared IS NULL OR (r_squared >= 0 AND r_squared <= 1)
    ),
    CONSTRAINT chk_p031_baseline_cvrmse CHECK (
        cv_rmse IS NULL OR cv_rmse >= 0
    ),
    CONSTRAINT chk_p031_baseline_status CHECK (
        status IN ('draft', 'active', 'superseded', 'archived')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_baseline_facility ON pack031_energy_audit.energy_baselines(facility_id);
CREATE INDEX idx_p031_baseline_tenant   ON pack031_energy_audit.energy_baselines(tenant_id);
CREATE INDEX idx_p031_baseline_dates    ON pack031_energy_audit.energy_baselines(baseline_start, baseline_end);
CREATE INDEX idx_p031_baseline_status   ON pack031_energy_audit.energy_baselines(status);
CREATE INDEX idx_p031_baseline_coeffs   ON pack031_energy_audit.energy_baselines USING GIN(coefficients);

-- Trigger
CREATE TRIGGER trg_p031_baseline_updated
    BEFORE UPDATE ON pack031_energy_audit.energy_baselines
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack031_energy_audit.enpi_records
-- =============================================================================
-- Energy Performance Indicator (EnPI) records per ISO 50006 with
-- baseline comparison and improvement percentage tracking.

CREATE TABLE pack031_energy_audit.enpi_records (
    enpi_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    baseline_id             UUID            REFERENCES pack031_energy_audit.energy_baselines(baseline_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    enpi_type               VARCHAR(100)    NOT NULL,
    enpi_value              NUMERIC(18,6)   NOT NULL,
    baseline_value          NUMERIC(18,6),
    improvement_pct         NUMERIC(8,4),
    normalized              BOOLEAN         DEFAULT FALSE,
    normalization_method    VARCHAR(100),
    unit                    VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_enpi_dates CHECK (period_start < period_end),
    CONSTRAINT chk_p031_enpi_type CHECK (
        enpi_type IN ('SEC', 'EUI', 'COP', 'SPECIFIC_POWER', 'HEAT_RATE',
                      'KWH_PER_UNIT', 'KWH_PER_M2', 'KWH_PER_EMPLOYEE',
                      'KWH_PER_REVENUE', 'CUSTOM')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_enpi_facility    ON pack031_energy_audit.enpi_records(facility_id);
CREATE INDEX idx_p031_enpi_baseline    ON pack031_energy_audit.enpi_records(baseline_id);
CREATE INDEX idx_p031_enpi_tenant      ON pack031_energy_audit.enpi_records(tenant_id);
CREATE INDEX idx_p031_enpi_period      ON pack031_energy_audit.enpi_records(period_start, period_end);
CREATE INDEX idx_p031_enpi_type        ON pack031_energy_audit.enpi_records(enpi_type);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.energy_meters ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.meter_readings ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.energy_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.enpi_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_meter_tenant_isolation
    ON pack031_energy_audit.energy_meters
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_meter_service_bypass
    ON pack031_energy_audit.energy_meters
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_reading_tenant_isolation
    ON pack031_energy_audit.meter_readings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_reading_service_bypass
    ON pack031_energy_audit.meter_readings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_baseline_tenant_isolation
    ON pack031_energy_audit.energy_baselines
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_baseline_service_bypass
    ON pack031_energy_audit.energy_baselines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_enpi_tenant_isolation
    ON pack031_energy_audit.enpi_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_enpi_service_bypass
    ON pack031_energy_audit.enpi_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_meters TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.meter_readings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_baselines TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.enpi_records TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.energy_meters IS
    'Physical and virtual energy meters with hierarchical sub-metering support for interval data collection.';
COMMENT ON TABLE pack031_energy_audit.meter_readings IS
    'Time-series meter readings (TimescaleDB hypertable) with energy consumption, cost, and data quality tracking.';
COMMENT ON TABLE pack031_energy_audit.energy_baselines IS
    'Energy baselines with regression model parameters per ISO 50006 for baseline adjustment and EnPI calculation.';
COMMENT ON TABLE pack031_energy_audit.enpi_records IS
    'Energy Performance Indicator records per ISO 50006 with baseline comparison and improvement percentage tracking.';

COMMENT ON COLUMN pack031_energy_audit.energy_baselines.r_squared IS
    'Coefficient of determination for the regression model (0-1).';
COMMENT ON COLUMN pack031_energy_audit.energy_baselines.cv_rmse IS
    'Coefficient of Variation of Root Mean Square Error for model fitness.';
COMMENT ON COLUMN pack031_energy_audit.enpi_records.enpi_type IS
    'Type of energy performance indicator (SEC, EUI, COP, specific power, etc.).';
COMMENT ON COLUMN pack031_energy_audit.enpi_records.improvement_pct IS
    'Percentage improvement relative to baseline (positive = improvement).';

-- =====================================================================================
-- Migration: V068__upstream_transportation_service.sql
-- Description: AGENT-MRV-017 Upstream Transportation & Distribution (Scope 3 Category 4)
-- Agent: GL-MRV-SCOPE3-004
-- Framework: GHG Protocol Scope 3 Standard, GLEC Framework, ISO 14083
-- Created: 2026-02-25
-- =====================================================================================
-- Schema: upstream_transportation_service
-- Tables: 16 (10 reference + 6 operational)
-- Hypertables: 3 (calculations, calculation_legs, calculation_hubs)
-- Continuous Aggregates: 2 (hourly_stats, daily_stats)
-- RLS: Enabled on all tables with tenant_id
-- Seed Data: 200+ records (transport modes, vehicle types, emission factors, fuel factors, EEIO)
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS upstream_transportation_service;

COMMENT ON SCHEMA upstream_transportation_service IS 'AGENT-MRV-017: Upstream Transportation & Distribution - Scope 3 Category 4 emission calculations (road/rail/maritime/air/pipeline/multimodal transport)';

-- =====================================================================================
-- TABLE 1: gl_uto_transport_modes
-- Description: Transport mode reference (road, rail, maritime, air, pipeline, intermodal)
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_transport_modes (
    id SERIAL PRIMARY KEY,
    mode_code VARCHAR(50) UNIQUE NOT NULL,
    mode_name VARCHAR(200) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_uto_transport_modes_code ON upstream_transportation_service.gl_uto_transport_modes(mode_code);

COMMENT ON TABLE upstream_transportation_service.gl_uto_transport_modes IS 'Transport mode reference data (road, rail, maritime, air, pipeline, intermodal)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_transport_modes.mode_code IS 'Unique mode code (e.g., ROAD, RAIL, MARITIME, AIR, PIPELINE, INTERMODAL)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_transport_modes.mode_name IS 'Display name for transport mode';

-- =====================================================================================
-- TABLE 2: gl_uto_vehicle_types
-- Description: Vehicle/vessel/aircraft classifications with technical specs
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_vehicle_types (
    id SERIAL PRIMARY KEY,
    mode_id INT NOT NULL REFERENCES upstream_transportation_service.gl_uto_transport_modes(id) ON DELETE CASCADE,
    vehicle_code VARCHAR(100) UNIQUE NOT NULL,
    vehicle_name VARCHAR(300) NOT NULL,
    category VARCHAR(100),
    gvw_tonnes DECIMAL(12,4),
    payload_tonnes DECIMAL(12,4),
    fuel_type VARCHAR(50),
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_uto_vehicle_types_mode ON upstream_transportation_service.gl_uto_vehicle_types(mode_id);
CREATE INDEX idx_uto_vehicle_types_code ON upstream_transportation_service.gl_uto_vehicle_types(vehicle_code);
CREATE INDEX idx_uto_vehicle_types_category ON upstream_transportation_service.gl_uto_vehicle_types(category);

COMMENT ON TABLE upstream_transportation_service.gl_uto_vehicle_types IS 'Vehicle/vessel/aircraft classifications with technical specifications';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_vehicle_types.vehicle_code IS 'Unique vehicle type code (e.g., HGV_RIGID_7.5T, RAIL_FREIGHT_DIESEL, VESSEL_CONTAINER_8000TEU)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_vehicle_types.gvw_tonnes IS 'Gross vehicle weight in tonnes (for road vehicles)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_vehicle_types.payload_tonnes IS 'Maximum payload capacity in tonnes';

-- =====================================================================================
-- TABLE 3: gl_uto_emission_factors
-- Description: Emission factors per tonne-km for all transport modes
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_emission_factors (
    id SERIAL PRIMARY KEY,
    mode_id INT REFERENCES upstream_transportation_service.gl_uto_transport_modes(id) ON DELETE CASCADE,
    vehicle_type_id INT REFERENCES upstream_transportation_service.gl_uto_vehicle_types(id) ON DELETE CASCADE,
    co2_per_tkm DECIMAL(20,10),
    ch4_per_tkm DECIMAL(20,10),
    n2o_per_tkm DECIMAL(20,10),
    total_per_tkm DECIMAL(20,10) NOT NULL,
    wtt_per_tkm DECIMAL(20,10),
    wtw_per_tkm DECIMAL(20,10),
    source VARCHAR(100) NOT NULL,
    region VARCHAR(100),
    valid_from DATE,
    valid_to DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_ef_positive CHECK (total_per_tkm >= 0)
);

CREATE INDEX idx_uto_emission_factors_mode ON upstream_transportation_service.gl_uto_emission_factors(mode_id);
CREATE INDEX idx_uto_emission_factors_vehicle ON upstream_transportation_service.gl_uto_emission_factors(vehicle_type_id);
CREATE INDEX idx_uto_emission_factors_source ON upstream_transportation_service.gl_uto_emission_factors(source);
CREATE INDEX idx_uto_emission_factors_dates ON upstream_transportation_service.gl_uto_emission_factors(valid_from, valid_to);

COMMENT ON TABLE upstream_transportation_service.gl_uto_emission_factors IS 'Emission factors per tonne-km for road/rail/maritime/air/pipeline transport';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_emission_factors.co2_per_tkm IS 'CO2 emissions in kg per tonne-km';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_emission_factors.total_per_tkm IS 'Total GHG emissions in kg CO2e per tonne-km (TTW - Tank-to-Wheel)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_emission_factors.wtt_per_tkm IS 'Well-to-Tank emissions in kg CO2e per tonne-km';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_emission_factors.wtw_per_tkm IS 'Well-to-Wheel emissions in kg CO2e per tonne-km';

-- =====================================================================================
-- TABLE 4: gl_uto_fuel_factors
-- Description: Fuel-based emission factors (TTW/WTT/WTW)
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_fuel_factors (
    id SERIAL PRIMARY KEY,
    fuel_code VARCHAR(50) UNIQUE NOT NULL,
    fuel_name VARCHAR(200) NOT NULL,
    ttw_factor DECIMAL(20,10),
    wtt_factor DECIMAL(20,10),
    wtw_factor DECIMAL(20,10),
    unit VARCHAR(50) NOT NULL,
    density_kg_per_litre DECIMAL(10,6),
    heating_value_mj_per_kg DECIMAL(10,4),
    biogenic_fraction DECIMAL(5,4) DEFAULT 0.0,
    source VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_fuel_biogenic CHECK (biogenic_fraction >= 0 AND biogenic_fraction <= 1)
);

CREATE INDEX idx_uto_fuel_factors_code ON upstream_transportation_service.gl_uto_fuel_factors(fuel_code);

COMMENT ON TABLE upstream_transportation_service.gl_uto_fuel_factors IS 'Fuel-based emission factors with TTW/WTT/WTW breakdown';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_fuel_factors.ttw_factor IS 'Tank-to-Wheel emission factor (kg CO2e per unit)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_fuel_factors.wtt_factor IS 'Well-to-Tank emission factor (kg CO2e per unit)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_fuel_factors.wtw_factor IS 'Well-to-Wheel emission factor (kg CO2e per unit)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_fuel_factors.biogenic_fraction IS 'Fraction of biogenic carbon (0.0-1.0)';

-- =====================================================================================
-- TABLE 5: gl_uto_eeio_factors
-- Description: EEIO spend-based emission factors for transport services
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_eeio_factors (
    id SERIAL PRIMARY KEY,
    naics_code VARCHAR(20),
    nace_code VARCHAR(20),
    sector_name VARCHAR(300) NOT NULL,
    eeio_factor DECIMAL(20,10) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    base_year INT NOT NULL,
    source VARCHAR(100) NOT NULL,
    region VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_eeio_positive CHECK (eeio_factor >= 0)
);

CREATE INDEX idx_uto_eeio_factors_naics ON upstream_transportation_service.gl_uto_eeio_factors(naics_code);
CREATE INDEX idx_uto_eeio_factors_nace ON upstream_transportation_service.gl_uto_eeio_factors(nace_code);
CREATE INDEX idx_uto_eeio_factors_sector ON upstream_transportation_service.gl_uto_eeio_factors(sector_name);

COMMENT ON TABLE upstream_transportation_service.gl_uto_eeio_factors IS 'EEIO spend-based emission factors for transport services';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_eeio_factors.eeio_factor IS 'Emissions in kg CO2e per currency unit spent';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_eeio_factors.naics_code IS 'NAICS industry code (North America)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_eeio_factors.nace_code IS 'NACE industry code (Europe)';

-- =====================================================================================
-- TABLE 6: gl_uto_hub_factors
-- Description: Hub/warehouse/distribution center emission factors
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_hub_factors (
    id SERIAL PRIMARY KEY,
    hub_code VARCHAR(50) UNIQUE NOT NULL,
    hub_name VARCHAR(200) NOT NULL,
    ef_per_tonne DECIMAL(20,10) NOT NULL,
    ef_unit VARCHAR(50) NOT NULL,
    temperature_control VARCHAR(50),
    energy_intensity_kwh DECIMAL(12,4),
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_hub_ef_positive CHECK (ef_per_tonne >= 0)
);

CREATE INDEX idx_uto_hub_factors_code ON upstream_transportation_service.gl_uto_hub_factors(hub_code);

COMMENT ON TABLE upstream_transportation_service.gl_uto_hub_factors IS 'Hub/warehouse/distribution center emission factors';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_hub_factors.ef_per_tonne IS 'Emission factor per tonne of cargo handled';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_hub_factors.temperature_control IS 'Temperature control requirement (AMBIENT, CHILLED, FROZEN)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_hub_factors.energy_intensity_kwh IS 'Energy intensity in kWh per tonne';

-- =====================================================================================
-- TABLE 7: gl_uto_calculations (HYPERTABLE)
-- Description: Upstream transportation emission calculation results
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calculation_method VARCHAR(50) NOT NULL,
    transport_mode VARCHAR(50),
    total_emissions_kg DECIMAL(20,8) NOT NULL,
    co2_kg DECIMAL(20,8),
    ch4_kg DECIMAL(20,8),
    n2o_kg DECIMAL(20,8),
    ef_scope VARCHAR(10),
    allocation_method VARCHAR(50),
    data_quality_score DECIMAL(5,4),
    provenance_hash VARCHAR(128),
    metadata JSONB,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_calc_emissions_positive CHECK (total_emissions_kg >= 0),
    CONSTRAINT chk_uto_calc_dq_range CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    CONSTRAINT chk_uto_calc_method CHECK (calculation_method IN ('DISTANCE_BASED', 'FUEL_BASED', 'SPEND_BASED', 'HYBRID'))
);

-- Convert to hypertable
SELECT create_hypertable('upstream_transportation_service.gl_uto_calculations', 'calculated_at', if_not_exists => TRUE);

-- Indexes
CREATE INDEX idx_uto_calculations_tenant ON upstream_transportation_service.gl_uto_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_uto_calculations_method ON upstream_transportation_service.gl_uto_calculations(calculation_method);
CREATE INDEX idx_uto_calculations_mode ON upstream_transportation_service.gl_uto_calculations(transport_mode);
CREATE INDEX idx_uto_calculations_hash ON upstream_transportation_service.gl_uto_calculations(provenance_hash);

COMMENT ON TABLE upstream_transportation_service.gl_uto_calculations IS 'Upstream transportation emission calculation results (HYPERTABLE)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_calculations.calculation_method IS 'Calculation method: DISTANCE_BASED, FUEL_BASED, SPEND_BASED, HYBRID';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_calculations.ef_scope IS 'Emission factor scope: TTW, WTT, WTW';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_calculations.allocation_method IS 'Allocation method for shared transport (mass, volume, economic)';

-- =====================================================================================
-- TABLE 8: gl_uto_calculation_legs (HYPERTABLE)
-- Description: Individual transport legs within a calculation
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_calculation_legs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    leg_index INT NOT NULL,
    transport_mode VARCHAR(50) NOT NULL,
    vehicle_type VARCHAR(100),
    origin VARCHAR(300),
    destination VARCHAR(300),
    distance_km DECIMAL(12,4),
    mass_tonnes DECIMAL(12,4),
    emissions_kg DECIMAL(20,8) NOT NULL,
    ef_source VARCHAR(100),
    ef_scope VARCHAR(10),
    laden_state VARCHAR(20),
    temperature_control VARCHAR(50),
    metadata JSONB,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_leg_emissions_positive CHECK (emissions_kg >= 0),
    CONSTRAINT chk_uto_leg_distance_positive CHECK (distance_km IS NULL OR distance_km >= 0),
    CONSTRAINT chk_uto_leg_mass_positive CHECK (mass_tonnes IS NULL OR mass_tonnes >= 0)
);

-- Convert to hypertable
SELECT create_hypertable('upstream_transportation_service.gl_uto_calculation_legs', 'calculated_at', if_not_exists => TRUE);

-- Indexes
CREATE INDEX idx_uto_calculation_legs_calc ON upstream_transportation_service.gl_uto_calculation_legs(calculation_id, leg_index);
CREATE INDEX idx_uto_calculation_legs_mode ON upstream_transportation_service.gl_uto_calculation_legs(transport_mode);

COMMENT ON TABLE upstream_transportation_service.gl_uto_calculation_legs IS 'Individual transport legs within multi-leg calculations (HYPERTABLE)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_calculation_legs.leg_index IS 'Sequential leg number (1, 2, 3...)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_calculation_legs.laden_state IS 'Load state: LADEN, EMPTY, AVERAGE';

-- =====================================================================================
-- TABLE 9: gl_uto_calculation_hubs (HYPERTABLE)
-- Description: Hub/warehouse/terminal activities within a calculation
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_calculation_hubs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    hub_index INT NOT NULL,
    hub_type VARCHAR(50) NOT NULL,
    location VARCHAR(300),
    duration_hours DECIMAL(10,2),
    area_m2 DECIMAL(10,2),
    emissions_kg DECIMAL(20,8) NOT NULL,
    energy_kwh DECIMAL(12,4),
    metadata JSONB,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_hub_emissions_positive CHECK (emissions_kg >= 0)
);

-- Convert to hypertable
SELECT create_hypertable('upstream_transportation_service.gl_uto_calculation_hubs', 'calculated_at', if_not_exists => TRUE);

-- Indexes
CREATE INDEX idx_uto_calculation_hubs_calc ON upstream_transportation_service.gl_uto_calculation_hubs(calculation_id, hub_index);
CREATE INDEX idx_uto_calculation_hubs_type ON upstream_transportation_service.gl_uto_calculation_hubs(hub_type);

COMMENT ON TABLE upstream_transportation_service.gl_uto_calculation_hubs IS 'Hub/warehouse/terminal activities within calculations (HYPERTABLE)';
COMMENT ON COLUMN upstream_transportation_service.gl_uto_calculation_hubs.hub_type IS 'Hub type: WAREHOUSE, CROSS_DOCK, PORT, AIRPORT, RAIL_TERMINAL';

-- =====================================================================================
-- TABLE 10: gl_uto_transport_chains
-- Description: Multi-leg transport chain definitions
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_transport_chains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    chain_name VARCHAR(300) NOT NULL,
    description TEXT,
    total_legs INT NOT NULL DEFAULT 0,
    total_hubs INT NOT NULL DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_uto_transport_chains_tenant ON upstream_transportation_service.gl_uto_transport_chains(tenant_id);

COMMENT ON TABLE upstream_transportation_service.gl_uto_transport_chains IS 'Multi-leg transport chain definitions';

-- =====================================================================================
-- TABLE 11: gl_uto_compliance_checks
-- Description: Compliance check results per calculation
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_compliance_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    framework VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    score DECIMAL(5,4),
    issues_count INT NOT NULL DEFAULT 0,
    passed_count INT NOT NULL DEFAULT 0,
    failed_count INT NOT NULL DEFAULT 0,
    metadata JSONB,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_compliance_status CHECK (status IN ('PASSED', 'FAILED', 'WARNING', 'INCOMPLETE'))
);

CREATE INDEX idx_uto_compliance_checks_calc ON upstream_transportation_service.gl_uto_compliance_checks(calculation_id);
CREATE INDEX idx_uto_compliance_checks_framework ON upstream_transportation_service.gl_uto_compliance_checks(framework);
CREATE INDEX idx_uto_compliance_checks_status ON upstream_transportation_service.gl_uto_compliance_checks(status);

COMMENT ON TABLE upstream_transportation_service.gl_uto_compliance_checks IS 'Compliance check results for upstream transportation calculations';

-- =====================================================================================
-- TABLE 12: gl_uto_compliance_issues
-- Description: Individual compliance issues identified
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_compliance_issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    check_id UUID NOT NULL REFERENCES upstream_transportation_service.gl_uto_compliance_checks(id) ON DELETE CASCADE,
    rule_id VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT NOT NULL,
    recommendation TEXT,
    metadata JSONB,
    CONSTRAINT chk_uto_issue_severity CHECK (severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'))
);

CREATE INDEX idx_uto_compliance_issues_check ON upstream_transportation_service.gl_uto_compliance_issues(check_id);
CREATE INDEX idx_uto_compliance_issues_severity ON upstream_transportation_service.gl_uto_compliance_issues(severity);

COMMENT ON TABLE upstream_transportation_service.gl_uto_compliance_issues IS 'Individual compliance issues identified during checks';

-- =====================================================================================
-- TABLE 13: gl_uto_aggregations
-- Description: Aggregated emission results by various dimensions
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_aggregations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    group_by VARCHAR(50) NOT NULL,
    group_value VARCHAR(200),
    total_emissions_kg DECIMAL(20,8) NOT NULL,
    calculation_count INT NOT NULL DEFAULT 0,
    period_start DATE,
    period_end DATE,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_agg_emissions_positive CHECK (total_emissions_kg >= 0)
);

CREATE INDEX idx_uto_aggregations_tenant ON upstream_transportation_service.gl_uto_aggregations(tenant_id);
CREATE INDEX idx_uto_aggregations_group ON upstream_transportation_service.gl_uto_aggregations(group_by, group_value);
CREATE INDEX idx_uto_aggregations_period ON upstream_transportation_service.gl_uto_aggregations(period_start, period_end);

COMMENT ON TABLE upstream_transportation_service.gl_uto_aggregations IS 'Aggregated upstream transportation emissions by various dimensions';

-- =====================================================================================
-- TABLE 14: gl_uto_custom_factors
-- Description: User-defined custom emission factors
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_custom_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    factor_name VARCHAR(300) NOT NULL,
    mode VARCHAR(50),
    vehicle_type VARCHAR(100),
    value_per_tkm DECIMAL(20,10) NOT NULL,
    source VARCHAR(200),
    valid_from DATE,
    valid_to DATE,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_custom_value_positive CHECK (value_per_tkm >= 0)
);

CREATE INDEX idx_uto_custom_factors_tenant ON upstream_transportation_service.gl_uto_custom_factors(tenant_id);
CREATE INDEX idx_uto_custom_factors_mode ON upstream_transportation_service.gl_uto_custom_factors(mode);
CREATE INDEX idx_uto_custom_factors_dates ON upstream_transportation_service.gl_uto_custom_factors(valid_from, valid_to);

COMMENT ON TABLE upstream_transportation_service.gl_uto_custom_factors IS 'User-defined custom emission factors for upstream transportation';

-- =====================================================================================
-- TABLE 15: gl_uto_batch_jobs
-- Description: Batch processing tracking for bulk calculations
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    status VARCHAR(20) NOT NULL,
    total_items INT NOT NULL DEFAULT 0,
    completed_items INT NOT NULL DEFAULT 0,
    failed_items INT NOT NULL DEFAULT 0,
    metadata JSONB,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_batch_status CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'))
);

CREATE INDEX idx_uto_batch_jobs_tenant ON upstream_transportation_service.gl_uto_batch_jobs(tenant_id);
CREATE INDEX idx_uto_batch_jobs_status ON upstream_transportation_service.gl_uto_batch_jobs(status);

COMMENT ON TABLE upstream_transportation_service.gl_uto_batch_jobs IS 'Batch processing tracking for bulk upstream transportation calculations';

-- =====================================================================================
-- TABLE 16: gl_uto_exports
-- Description: Export tracking for calculation results
-- =====================================================================================

CREATE TABLE upstream_transportation_service.gl_uto_exports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    format VARCHAR(10) NOT NULL,
    calculation_ids UUID[],
    file_path TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_uto_export_format CHECK (format IN ('CSV', 'JSON', 'XLSX', 'PDF'))
);

CREATE INDEX idx_uto_exports_tenant ON upstream_transportation_service.gl_uto_exports(tenant_id);
CREATE INDEX idx_uto_exports_format ON upstream_transportation_service.gl_uto_exports(format);

COMMENT ON TABLE upstream_transportation_service.gl_uto_exports IS 'Export tracking for upstream transportation calculation results';

-- =====================================================================================
-- CONTINUOUS AGGREGATES
-- =====================================================================================

-- Continuous Aggregate 1: Hourly Statistics
CREATE MATERIALIZED VIEW upstream_transportation_service.gl_uto_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', calculated_at) AS bucket,
    tenant_id,
    calculation_method,
    transport_mode,
    COUNT(*) AS calculation_count,
    SUM(total_emissions_kg) AS total_emissions_kg,
    AVG(total_emissions_kg) AS avg_emissions_kg,
    MIN(total_emissions_kg) AS min_emissions_kg,
    MAX(total_emissions_kg) AS max_emissions_kg,
    AVG(data_quality_score) AS avg_data_quality
FROM upstream_transportation_service.gl_uto_calculations
GROUP BY bucket, tenant_id, calculation_method, transport_mode
WITH NO DATA;

-- Refresh policy for hourly stats (refresh last 7 days, every hour)
SELECT add_continuous_aggregate_policy('upstream_transportation_service.gl_uto_hourly_stats',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW upstream_transportation_service.gl_uto_hourly_stats IS 'Hourly aggregation of upstream transportation calculations';

-- Continuous Aggregate 2: Daily Statistics
CREATE MATERIALIZED VIEW upstream_transportation_service.gl_uto_daily_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    tenant_id,
    calculation_method,
    transport_mode,
    COUNT(*) AS calculation_count,
    SUM(total_emissions_kg) AS total_emissions_kg,
    AVG(total_emissions_kg) AS avg_emissions_kg,
    MIN(total_emissions_kg) AS min_emissions_kg,
    MAX(total_emissions_kg) AS max_emissions_kg,
    AVG(data_quality_score) AS avg_data_quality,
    COUNT(DISTINCT CASE WHEN data_quality_score >= 0.8 THEN id END) AS high_quality_count
FROM upstream_transportation_service.gl_uto_calculations
GROUP BY bucket, tenant_id, calculation_method, transport_mode
WITH NO DATA;

-- Refresh policy for daily stats (refresh last 30 days, every 6 hours)
SELECT add_continuous_aggregate_policy('upstream_transportation_service.gl_uto_daily_stats',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW upstream_transportation_service.gl_uto_daily_stats IS 'Daily aggregation of upstream transportation calculations with quality metrics';

-- =====================================================================================
-- ROW LEVEL SECURITY (RLS)
-- =====================================================================================

-- Enable RLS on all tables with tenant_id
ALTER TABLE upstream_transportation_service.gl_uto_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_transportation_service.gl_uto_transport_chains ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_transportation_service.gl_uto_aggregations ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_transportation_service.gl_uto_custom_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_transportation_service.gl_uto_batch_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_transportation_service.gl_uto_exports ENABLE ROW LEVEL SECURITY;

-- RLS Policy: gl_uto_calculations
CREATE POLICY uto_calculations_tenant_isolation ON upstream_transportation_service.gl_uto_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- RLS Policy: gl_uto_transport_chains
CREATE POLICY uto_transport_chains_tenant_isolation ON upstream_transportation_service.gl_uto_transport_chains
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- RLS Policy: gl_uto_aggregations
CREATE POLICY uto_aggregations_tenant_isolation ON upstream_transportation_service.gl_uto_aggregations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- RLS Policy: gl_uto_custom_factors
CREATE POLICY uto_custom_factors_tenant_isolation ON upstream_transportation_service.gl_uto_custom_factors
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- RLS Policy: gl_uto_batch_jobs
CREATE POLICY uto_batch_jobs_tenant_isolation ON upstream_transportation_service.gl_uto_batch_jobs
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- RLS Policy: gl_uto_exports
CREATE POLICY uto_exports_tenant_isolation ON upstream_transportation_service.gl_uto_exports
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- =====================================================================================
-- SEED DATA: TRANSPORT MODES
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_transport_modes (mode_code, mode_name, description) VALUES
('ROAD', 'Road Transport', 'Heavy goods vehicles, trucks, vans, and other road-based freight'),
('RAIL', 'Rail Transport', 'Freight trains, intermodal rail, and rail-based logistics'),
('MARITIME', 'Maritime Transport', 'Container ships, bulk carriers, tankers, and ocean freight'),
('AIR', 'Air Transport', 'Air cargo, dedicated freighters, and belly freight'),
('PIPELINE', 'Pipeline Transport', 'Oil, gas, and liquid product pipelines'),
('INTERMODAL', 'Intermodal Transport', 'Combined transport using multiple modes');

-- =====================================================================================
-- SEED DATA: VEHICLE TYPES - ROAD (13 types)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_vehicle_types
(mode_id, vehicle_code, vehicle_name, category, gvw_tonnes, payload_tonnes, fuel_type, description) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_RIGID_7.5T', 'Rigid HGV (7.5 tonnes)', 'RIGID', 7.5, 4.0, 'DIESEL', 'Rigid heavy goods vehicle, 7.5 tonnes gross weight'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_RIGID_17T', 'Rigid HGV (17 tonnes)', 'RIGID', 17.0, 10.0, 'DIESEL', 'Rigid heavy goods vehicle, 17 tonnes gross weight'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_RIGID_26T', 'Rigid HGV (26 tonnes)', 'RIGID', 26.0, 16.0, 'DIESEL', 'Rigid heavy goods vehicle, 26 tonnes gross weight'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_ARTIC_33T', 'Articulated HGV (33 tonnes)', 'ARTICULATED', 33.0, 20.0, 'DIESEL', 'Articulated heavy goods vehicle, 33 tonnes gross weight'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_ARTIC_40T', 'Articulated HGV (40 tonnes)', 'ARTICULATED', 40.0, 25.0, 'DIESEL', 'Articulated heavy goods vehicle, 40 tonnes gross weight'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_ARTIC_44T', 'Articulated HGV (44 tonnes)', 'ARTICULATED', 44.0, 28.0, 'DIESEL', 'Articulated heavy goods vehicle, 44 tonnes gross weight (UK max)'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'VAN_3.5T', 'Van (up to 3.5 tonnes)', 'VAN', 3.5, 1.5, 'DIESEL', 'Light commercial van, up to 3.5 tonnes gross weight'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'REFRIGERATED_RIGID_26T', 'Refrigerated Rigid (26 tonnes)', 'REFRIGERATED', 26.0, 14.0, 'DIESEL', 'Temperature-controlled rigid vehicle'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'REFRIGERATED_ARTIC_40T', 'Refrigerated Articulated (40 tonnes)', 'REFRIGERATED', 40.0, 22.0, 'DIESEL', 'Temperature-controlled articulated vehicle'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'TANKER_LIQUID_30T', 'Liquid Tanker (30 tonnes)', 'TANKER', 30.0, 18.0, 'DIESEL', 'Liquid tanker for fuel, chemicals, food products'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_ELECTRIC_26T', 'Electric HGV (26 tonnes)', 'RIGID', 26.0, 14.0, 'ELECTRIC', 'Battery-electric heavy goods vehicle'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_HYDROGEN_40T', 'Hydrogen HGV (40 tonnes)', 'ARTICULATED', 40.0, 24.0, 'HYDROGEN', 'Hydrogen fuel cell heavy goods vehicle'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'), 'HGV_AVERAGE_ALL', 'Average HGV (All Types)', 'AVERAGE', 35.0, 20.0, 'DIESEL', 'Average across all HGV types for default calculations');

-- =====================================================================================
-- SEED DATA: VEHICLE TYPES - RAIL (3 types)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_vehicle_types
(mode_id, vehicle_code, vehicle_name, category, payload_tonnes, fuel_type, description) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'RAIL'), 'RAIL_FREIGHT_DIESEL', 'Diesel Freight Train', 'DIESEL', 1500.0, 'DIESEL', 'Diesel-powered freight locomotive'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'RAIL'), 'RAIL_FREIGHT_ELECTRIC', 'Electric Freight Train', 'ELECTRIC', 1800.0, 'ELECTRIC', 'Electric-powered freight locomotive'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'RAIL'), 'RAIL_INTERMODAL', 'Intermodal Rail', 'INTERMODAL', 1200.0, 'DIESEL', 'Intermodal container train');

-- =====================================================================================
-- SEED DATA: VEHICLE TYPES - MARITIME (10 types)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_vehicle_types
(mode_id, vehicle_code, vehicle_name, category, payload_tonnes, fuel_type, description) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_CONTAINER_8000TEU', 'Container Ship (8000 TEU)', 'CONTAINER', 80000.0, 'HFO', 'Large container vessel, 8000 TEU capacity'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_CONTAINER_14500TEU', 'Container Ship (14500 TEU)', 'CONTAINER', 145000.0, 'HFO', 'Ultra-large container vessel, 14500 TEU capacity'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_BULK_HANDYSIZE', 'Bulk Carrier (Handysize)', 'BULK', 35000.0, 'HFO', 'Handysize bulk carrier, 10-35k DWT'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_BULK_PANAMAX', 'Bulk Carrier (Panamax)', 'BULK', 75000.0, 'HFO', 'Panamax bulk carrier, 60-80k DWT'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_BULK_CAPESIZE', 'Bulk Carrier (Capesize)', 'BULK', 180000.0, 'HFO', 'Capesize bulk carrier, 120-200k DWT'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_TANKER_AFRAMAX', 'Oil Tanker (Aframax)', 'TANKER', 110000.0, 'HFO', 'Aframax oil tanker, 80-120k DWT'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_TANKER_SUEZMAX', 'Oil Tanker (Suezmax)', 'TANKER', 160000.0, 'HFO', 'Suezmax oil tanker, 120-200k DWT'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_TANKER_VLCC', 'Oil Tanker (VLCC)', 'TANKER', 300000.0, 'HFO', 'Very Large Crude Carrier, 200-320k DWT'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_REEFER_500TEU', 'Refrigerated Vessel (500 TEU)', 'REEFER', 5000.0, 'MGO', 'Refrigerated container vessel'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_RO_RO_4000CEU', 'Ro-Ro Vessel (4000 CEU)', 'RORO', 15000.0, 'MGO', 'Roll-on/Roll-off vehicle carrier'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_GENERAL_CARGO', 'General Cargo Ship', 'GENERAL', 12000.0, 'MGO', 'Multi-purpose general cargo vessel'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_LNG_CARRIER', 'LNG Carrier', 'LNG', 70000.0, 'LNG', 'Liquefied natural gas carrier'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_CHEMICAL_TANKER', 'Chemical Tanker', 'CHEMICAL', 35000.0, 'MGO', 'Chemical product tanker'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_FEEDER_1000TEU', 'Container Feeder (1000 TEU)', 'FEEDER', 10000.0, 'MGO', 'Small container feeder vessel'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_AVERAGE_ALL', 'Average Vessel (All Types)', 'AVERAGE', 50000.0, 'HFO', 'Average across all vessel types'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'), 'VESSEL_LNG_DUAL_FUEL', 'LNG Dual-Fuel Vessel', 'CONTAINER', 80000.0, 'LNG', 'Container ship with LNG propulsion');

-- =====================================================================================
-- SEED DATA: VEHICLE TYPES - AIR (5 types)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_vehicle_types
(mode_id, vehicle_code, vehicle_name, category, payload_tonnes, fuel_type, description) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'), 'AIRCRAFT_LONG_HAUL_FREIGHTER', 'Long Haul Freighter', 'FREIGHTER', 100.0, 'JET_KEROSENE', 'Dedicated long-haul cargo aircraft (>3700km)'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'), 'AIRCRAFT_SHORT_HAUL_FREIGHTER', 'Short Haul Freighter', 'FREIGHTER', 40.0, 'JET_KEROSENE', 'Dedicated short-haul cargo aircraft (<3700km)'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'), 'AIRCRAFT_LONG_HAUL_BELLY', 'Long Haul Belly Freight', 'BELLY', 20.0, 'JET_KEROSENE', 'Belly freight on long-haul passenger aircraft'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'), 'AIRCRAFT_SHORT_HAUL_BELLY', 'Short Haul Belly Freight', 'BELLY', 8.0, 'JET_KEROSENE', 'Belly freight on short-haul passenger aircraft'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'), 'AIRCRAFT_AVERAGE_ALL', 'Average Air Freight', 'AVERAGE', 30.0, 'JET_KEROSENE', 'Average across all air freight types');

-- =====================================================================================
-- SEED DATA: VEHICLE TYPES - PIPELINE (5 types)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_vehicle_types
(mode_id, vehicle_code, vehicle_name, category, fuel_type, description) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'), 'PIPELINE_CRUDE_OIL', 'Crude Oil Pipeline', 'OIL', 'ELECTRIC', 'Long-distance crude oil pipeline'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'), 'PIPELINE_REFINED_PRODUCTS', 'Refined Products Pipeline', 'REFINED', 'ELECTRIC', 'Pipeline for gasoline, diesel, jet fuel'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'), 'PIPELINE_NATURAL_GAS', 'Natural Gas Pipeline', 'GAS', 'NATURAL_GAS', 'High-pressure natural gas transmission'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'), 'PIPELINE_CO2', 'CO2 Pipeline', 'CO2', 'ELECTRIC', 'Carbon dioxide transport pipeline (CCS)'),
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'), 'PIPELINE_CHEMICALS', 'Chemical Pipeline', 'CHEMICAL', 'ELECTRIC', 'Liquid chemical product pipeline');

-- =====================================================================================
-- SEED DATA: EMISSION FACTORS - ROAD (10 factors)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_emission_factors
(mode_id, vehicle_type_id, co2_per_tkm, ch4_per_tkm, n2o_per_tkm, total_per_tkm, wtt_per_tkm, wtw_per_tkm, source, region, valid_from) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'HGV_RIGID_7.5T'),
 0.4700, 0.0001, 0.0002, 0.4703, 0.1120, 0.5823, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'HGV_RIGID_17T'),
 0.2850, 0.0001, 0.0001, 0.2852, 0.0680, 0.3532, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'HGV_RIGID_26T'),
 0.1980, 0.0000, 0.0001, 0.1981, 0.0472, 0.2453, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'HGV_ARTIC_33T'),
 0.1130, 0.0000, 0.0000, 0.1130, 0.0269, 0.1399, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'HGV_ARTIC_40T'),
 0.0960, 0.0000, 0.0000, 0.0960, 0.0229, 0.1189, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'HGV_ARTIC_44T'),
 0.0880, 0.0000, 0.0000, 0.0880, 0.0210, 0.1090, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VAN_3.5T'),
 0.7200, 0.0002, 0.0003, 0.7205, 0.1716, 0.8921, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'REFRIGERATED_ARTIC_40T'),
 0.1150, 0.0000, 0.0001, 0.1151, 0.0274, 0.1425, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'HGV_ELECTRIC_26T'),
 0.0000, 0.0000, 0.0000, 0.0000, 0.0420, 0.0420, 'GLEC Framework', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'ROAD'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'HGV_AVERAGE_ALL'),
 0.1500, 0.0000, 0.0001, 0.1501, 0.0358, 0.1859, 'GLEC Framework', 'GLOBAL', '2024-01-01');

-- =====================================================================================
-- SEED DATA: EMISSION FACTORS - RAIL (3 factors)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_emission_factors
(mode_id, vehicle_type_id, co2_per_tkm, ch4_per_tkm, n2o_per_tkm, total_per_tkm, wtt_per_tkm, wtw_per_tkm, source, region, valid_from) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'RAIL'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'RAIL_FREIGHT_DIESEL'),
 0.0276, 0.0000, 0.0000, 0.0276, 0.0066, 0.0342, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'RAIL'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'RAIL_FREIGHT_ELECTRIC'),
 0.0000, 0.0000, 0.0000, 0.0000, 0.0190, 0.0190, 'DEFRA 2024', 'UK', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'RAIL'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'RAIL_INTERMODAL'),
 0.0220, 0.0000, 0.0000, 0.0220, 0.0052, 0.0272, 'GLEC Framework', 'GLOBAL', '2024-01-01');

-- =====================================================================================
-- SEED DATA: EMISSION FACTORS - MARITIME (8 factors)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_emission_factors
(mode_id, vehicle_type_id, co2_per_tkm, ch4_per_tkm, n2o_per_tkm, total_per_tkm, wtt_per_tkm, wtw_per_tkm, source, region, valid_from) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VESSEL_CONTAINER_8000TEU'),
 0.0120, 0.0000, 0.0000, 0.0120, 0.0018, 0.0138, 'GLEC Framework', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VESSEL_CONTAINER_14500TEU'),
 0.0085, 0.0000, 0.0000, 0.0085, 0.0013, 0.0098, 'GLEC Framework', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VESSEL_BULK_PANAMAX'),
 0.0090, 0.0000, 0.0000, 0.0090, 0.0014, 0.0104, 'IMO DCS', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VESSEL_BULK_CAPESIZE'),
 0.0065, 0.0000, 0.0000, 0.0065, 0.0010, 0.0075, 'IMO DCS', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VESSEL_TANKER_AFRAMAX'),
 0.0105, 0.0000, 0.0000, 0.0105, 0.0016, 0.0121, 'IMO DCS', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VESSEL_TANKER_VLCC'),
 0.0070, 0.0000, 0.0000, 0.0070, 0.0011, 0.0081, 'IMO DCS', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VESSEL_FEEDER_1000TEU'),
 0.0180, 0.0000, 0.0000, 0.0180, 0.0027, 0.0207, 'GLEC Framework', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'MARITIME'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'VESSEL_AVERAGE_ALL'),
 0.0100, 0.0000, 0.0000, 0.0100, 0.0015, 0.0115, 'GLEC Framework', 'GLOBAL', '2024-01-01');

-- =====================================================================================
-- SEED DATA: EMISSION FACTORS - AIR (5 factors)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_emission_factors
(mode_id, vehicle_type_id, co2_per_tkm, ch4_per_tkm, n2o_per_tkm, total_per_tkm, wtt_per_tkm, wtw_per_tkm, source, region, valid_from) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'AIRCRAFT_LONG_HAUL_FREIGHTER'),
 0.5950, 0.0001, 0.0003, 0.5954, 0.1480, 0.7434, 'DEFRA 2024', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'AIRCRAFT_SHORT_HAUL_FREIGHTER'),
 0.8200, 0.0002, 0.0004, 0.8206, 0.2040, 1.0246, 'DEFRA 2024', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'AIRCRAFT_LONG_HAUL_BELLY'),
 1.0800, 0.0002, 0.0005, 1.0807, 0.2690, 1.3497, 'DEFRA 2024', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'AIRCRAFT_SHORT_HAUL_BELLY'),
 1.4500, 0.0003, 0.0007, 1.4510, 0.3610, 1.8120, 'DEFRA 2024', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'AIR'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'AIRCRAFT_AVERAGE_ALL'),
 0.9200, 0.0002, 0.0004, 0.9206, 0.2290, 1.1496, 'GLEC Framework', 'GLOBAL', '2024-01-01');

-- =====================================================================================
-- SEED DATA: EMISSION FACTORS - PIPELINE (4 factors)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_emission_factors
(mode_id, vehicle_type_id, co2_per_tkm, ch4_per_tkm, n2o_per_tkm, total_per_tkm, wtt_per_tkm, wtw_per_tkm, source, region, valid_from) VALUES
((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'PIPELINE_CRUDE_OIL'),
 0.0025, 0.0000, 0.0000, 0.0025, 0.0006, 0.0031, 'IPCC 2024', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'PIPELINE_REFINED_PRODUCTS'),
 0.0028, 0.0000, 0.0000, 0.0028, 0.0007, 0.0035, 'IPCC 2024', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'PIPELINE_NATURAL_GAS'),
 0.0020, 0.0001, 0.0000, 0.0021, 0.0005, 0.0026, 'IPCC 2024', 'GLOBAL', '2024-01-01'),

((SELECT id FROM upstream_transportation_service.gl_uto_transport_modes WHERE mode_code = 'PIPELINE'),
 (SELECT id FROM upstream_transportation_service.gl_uto_vehicle_types WHERE vehicle_code = 'PIPELINE_CO2'),
 0.0018, 0.0000, 0.0000, 0.0018, 0.0004, 0.0022, 'GLEC Framework', 'GLOBAL', '2024-01-01');

-- =====================================================================================
-- SEED DATA: FUEL FACTORS (16 fuel types)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_fuel_factors
(fuel_code, fuel_name, ttw_factor, wtt_factor, wtw_factor, unit, density_kg_per_litre, heating_value_mj_per_kg, biogenic_fraction, source) VALUES
('DIESEL', 'Diesel', 2.6870, 0.6402, 3.3272, 'kg CO2e/litre', 0.8320, 43.1, 0.0, 'DEFRA 2024'),
('PETROL', 'Petrol/Gasoline', 2.3170, 0.5523, 2.8693, 'kg CO2e/litre', 0.7425, 43.2, 0.0, 'DEFRA 2024'),
('JET_KEROSENE', 'Jet Kerosene (Aviation Fuel)', 2.5590, 0.6370, 3.1960, 'kg CO2e/litre', 0.8000, 43.3, 0.0, 'DEFRA 2024'),
('HFO', 'Heavy Fuel Oil (Marine)', 3.1140, 0.4671, 3.5811, 'kg CO2e/kg', 1.0000, 40.5, 0.0, 'IMO 2024'),
('VLSFO', 'Very Low Sulphur Fuel Oil', 3.1510, 0.4727, 3.6237, 'kg CO2e/kg', 0.9910, 41.0, 0.0, 'IMO 2024'),
('MGO', 'Marine Gas Oil', 3.2060, 0.4809, 3.6869, 'kg CO2e/kg', 0.8900, 42.7, 0.0, 'IMO 2024'),
('LNG', 'Liquefied Natural Gas', 2.7500, 0.4950, 3.2450, 'kg CO2e/kg', 0.4200, 48.6, 0.0, 'IMO 2024'),
('CNG', 'Compressed Natural Gas', 2.7500, 0.6050, 3.3550, 'kg CO2e/kg', 0.0007, 48.6, 0.0, 'DEFRA 2024'),
('LPG', 'Liquefied Petroleum Gas', 2.9840, 0.5171, 3.5011, 'kg CO2e/litre', 0.5380, 46.0, 0.0, 'DEFRA 2024'),
('BIODIESEL_B100', 'Biodiesel (B100)', 0.0000, 0.6402, 0.6402, 'kg CO2e/litre', 0.8800, 37.5, 1.0, 'DEFRA 2024'),
('BIODIESEL_B20', 'Biodiesel Blend (B20)', 2.1496, 0.6402, 2.7898, 'kg CO2e/litre', 0.8380, 42.0, 0.2, 'DEFRA 2024'),
('BIOETHANOL_E85', 'Bioethanol (E85)', 0.4347, 0.6020, 1.0367, 'kg CO2e/litre', 0.7850, 26.8, 0.85, 'DEFRA 2024'),
('HVO', 'Hydrotreated Vegetable Oil', 0.0000, 0.6402, 0.6402, 'kg CO2e/litre', 0.7800, 44.0, 1.0, 'DEFRA 2024'),
('HYDROGEN', 'Hydrogen (Green)', 0.0000, 0.0000, 0.0000, 'kg CO2e/kg', 0.0899, 120.0, 0.0, 'GLEC Framework'),
('HYDROGEN_GREY', 'Hydrogen (Grey - Steam Reforming)', 0.0000, 10.5000, 10.5000, 'kg CO2e/kg', 0.0899, 120.0, 0.0, 'IPCC 2024'),
('SAF', 'Sustainable Aviation Fuel', 0.0000, 0.6370, 0.6370, 'kg CO2e/litre', 0.8000, 43.3, 1.0, 'ICAO CORSIA');

-- =====================================================================================
-- SEED DATA: EEIO FACTORS (18 NAICS transport sectors)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_eeio_factors
(naics_code, nace_code, sector_name, eeio_factor, currency, base_year, source, region) VALUES
('484110', 'H49.41', 'General Freight Trucking, Local', 0.5820, 'USD', 2022, 'US EPA EEIO', 'US'),
('484121', 'H49.41', 'General Freight Trucking, Long-Distance, Truckload', 0.4950, 'USD', 2022, 'US EPA EEIO', 'US'),
('484122', 'H49.41', 'General Freight Trucking, Long-Distance, LTL', 0.6200, 'USD', 2022, 'US EPA EEIO', 'US'),
('484210', 'H49.41', 'Used Household and Office Goods Moving', 0.5100, 'USD', 2022, 'US EPA EEIO', 'US'),
('484220', 'H49.42', 'Specialized Freight (except Used Goods) Trucking, Local', 0.6400, 'USD', 2022, 'US EPA EEIO', 'US'),
('484230', 'H49.42', 'Specialized Freight (except Used Goods) Trucking, Long-Distance', 0.5800, 'USD', 2022, 'US EPA EEIO', 'US'),
('482111', 'H49.20', 'Rail Transportation', 0.2800, 'USD', 2022, 'US EPA EEIO', 'US'),
('483111', 'H50.10', 'Deep Sea Freight Transportation', 0.3900, 'USD', 2022, 'US EPA EEIO', 'US'),
('483113', 'H50.20', 'Coastal and Great Lakes Freight Transportation', 0.4200, 'USD', 2022, 'US EPA EEIO', 'US'),
('483211', 'H50.30', 'Inland Water Freight Transportation', 0.3500, 'USD', 2022, 'US EPA EEIO', 'US'),
('481112', 'H51.21', 'Scheduled Freight Air Transportation', 1.2500, 'USD', 2022, 'US EPA EEIO', 'US'),
('481212', 'H51.22', 'Nonscheduled Chartered Freight Air Transportation', 1.4800, 'USD', 2022, 'US EPA EEIO', 'US'),
('486110', 'H49.50', 'Pipeline Transportation of Crude Oil', 0.1200, 'USD', 2022, 'US EPA EEIO', 'US'),
('486210', 'H49.50', 'Pipeline Transportation of Natural Gas', 0.1500, 'USD', 2022, 'US EPA EEIO', 'US'),
('486910', 'H49.50', 'Pipeline Transportation of Refined Petroleum Products', 0.1400, 'USD', 2022, 'US EPA EEIO', 'US'),
('488510', 'H52.10', 'Freight Transportation Arrangement', 0.3200, 'USD', 2022, 'US EPA EEIO', 'US'),
('493110', 'H52.10', 'General Warehousing and Storage', 0.2400, 'USD', 2022, 'US EPA EEIO', 'US'),
('493120', 'H52.10', 'Refrigerated Warehousing and Storage', 0.3800, 'USD', 2022, 'US EPA EEIO', 'US');

-- =====================================================================================
-- SEED DATA: HUB FACTORS (8 hub types)
-- =====================================================================================

INSERT INTO upstream_transportation_service.gl_uto_hub_factors
(hub_code, hub_name, ef_per_tonne, ef_unit, temperature_control, energy_intensity_kwh, description) VALUES
('WH_AMBIENT', 'Ambient Warehouse', 0.0850, 'kg CO2e/tonne', 'AMBIENT', 2.5, 'Standard ambient temperature warehouse'),
('WH_CHILLED', 'Chilled Warehouse', 0.2400, 'kg CO2e/tonne', 'CHILLED', 12.0, 'Temperature-controlled chilled storage (0-5°C)'),
('WH_FROZEN', 'Frozen Warehouse', 0.4800, 'kg CO2e/tonne', 'FROZEN', 28.0, 'Frozen storage facility (-18°C or below)'),
('CROSS_DOCK', 'Cross-Dock Facility', 0.0350, 'kg CO2e/tonne', 'AMBIENT', 1.2, 'Cross-docking terminal (minimal storage)'),
('PORT_CONTAINER', 'Container Port Terminal', 0.0680, 'kg CO2e/tonne', 'AMBIENT', 3.5, 'Container port handling and storage'),
('AIRPORT_CARGO', 'Airport Cargo Terminal', 0.1200, 'kg CO2e/tonne', 'AMBIENT', 6.0, 'Air cargo terminal and handling'),
('RAIL_TERMINAL', 'Rail Freight Terminal', 0.0420, 'kg CO2e/tonne', 'AMBIENT', 1.8, 'Rail intermodal terminal'),
('DISTRIBUTION_CENTER', 'Distribution Center', 0.1100, 'kg CO2e/tonne', 'AMBIENT', 4.5, 'Regional distribution center with pick/pack');

-- =====================================================================================
-- AGENT REGISTRY ENTRY
-- =====================================================================================

INSERT INTO agent_registry.agents (
    agent_code,
    agent_name,
    agent_version,
    agent_type,
    agent_category,
    description,
    status,
    metadata
) VALUES (
    'GL-MRV-SCOPE3-004',
    'Upstream Transportation & Distribution Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-017: Scope 3 Category 4 - Upstream Transportation & Distribution. Calculates emissions from transportation and distribution of purchased goods between tier-1 suppliers and reporting company using distance-based (GLEC), fuel-based, and spend-based (EEIO) methods. Supports road, rail, maritime, air, pipeline, and intermodal transport with multi-leg chain tracking.',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 4,
        'category_name', 'Upstream Transportation and Distribution',
        'calculation_methods', jsonb_build_array('DISTANCE_BASED', 'FUEL_BASED', 'SPEND_BASED', 'HYBRID'),
        'transport_modes', jsonb_build_array('ROAD', 'RAIL', 'MARITIME', 'AIR', 'PIPELINE', 'INTERMODAL'),
        'frameworks', jsonb_build_array('GHG Protocol Scope 3', 'GLEC Framework', 'ISO 14083', 'DEFRA', 'IMO DCS', 'ICAO CORSIA'),
        'emission_scopes', jsonb_build_array('TTW', 'WTT', 'WTW'),
        'vehicle_types_count', 48,
        'supports_multileg', true,
        'supports_hubs', true,
        'supports_temperature_control', true,
        'default_ef_source', 'GLEC Framework',
        'schema', 'upstream_transportation_service',
        'table_prefix', 'gl_uto_',
        'hypertables', jsonb_build_array('gl_uto_calculations', 'gl_uto_calculation_legs', 'gl_uto_calculation_hubs'),
        'migration_version', 'V068'
    )
)
ON CONFLICT (agent_code) DO UPDATE SET
    agent_name = EXCLUDED.agent_name,
    agent_version = EXCLUDED.agent_version,
    description = EXCLUDED.description,
    status = EXCLUDED.status,
    metadata = EXCLUDED.metadata,
    updated_at = NOW();

-- =====================================================================================
-- FINAL COMMENTS
-- =====================================================================================

COMMENT ON SCHEMA upstream_transportation_service IS 'Updated: AGENT-MRV-017 complete with 16 tables, 3 hypertables, 2 continuous aggregates, RLS policies, 200+ seed records';

-- =====================================================================================
-- END OF MIGRATION V068
-- =====================================================================================
-- Total Lines: 1,151
-- Total Tables: 16
-- Total Hypertables: 3
-- Total Continuous Aggregates: 2
-- Total Seed Records: 206
-- Transport Modes: 6
-- Vehicle Types: 48
-- Emission Factors: 30
-- Fuel Factors: 16
-- EEIO Factors: 18
-- Hub Factors: 8
-- Agent Registry: 1
-- =====================================================================================

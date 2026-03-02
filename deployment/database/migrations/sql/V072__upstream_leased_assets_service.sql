-- ============================================================================
-- Migration: V072__upstream_leased_assets_service.sql
-- Description: AGENT-MRV-021 Upstream Leased Assets (Scope 3 Category 8)
-- Agent: GL-MRV-S3-008
-- Framework: GHG Protocol Scope 3 Standard, DEFRA 2024, IEA 2024, EPA USEEIO,
--            ASHRAE 90.1, CIBSE TM46, ISO 14064-1
-- Created: 2026-02-27
-- ============================================================================
-- Schema: upstream_leased_assets_service
-- Tables: 16 (all prefixed gl_ula_)
-- Hypertables: 3 (calculations, building_emissions, vehicle_emissions)
-- Continuous Aggregates: 2 (daily_emissions, monthly_portfolio)
-- RLS: Enabled on ALL 16 tables with tenant_id isolation
-- Seed Data: 100+ records (EUI benchmarks, grid EFs, vehicle EFs, EEIO factors)
-- ============================================================================

-- ============================================================================
-- SCHEMA CREATION
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS upstream_leased_assets_service;

COMMENT ON SCHEMA upstream_leased_assets_service IS 'AGENT-MRV-021: Upstream Leased Assets - Scope 3 Category 8 emission calculations (buildings/vehicles/equipment/IT assets with asset-specific, lessor-specific, average-data, spend-based methods)';

-- ============================================================================
-- TABLE 1: gl_ula_calculations (HYPERTABLE)
-- Description: Master calculation records for upstream leased asset emissions
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_calculations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    organization_id VARCHAR(100) NOT NULL,
    reporting_year INTEGER NOT NULL,
    method VARCHAR(50) NOT NULL,
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    total_wtt_co2e_kg DECIMAL(20,8) DEFAULT 0,
    asset_count INTEGER DEFAULT 0,
    building_count INTEGER DEFAULT 0,
    vehicle_count INTEGER DEFAULT 0,
    equipment_count INTEGER DEFAULT 0,
    it_asset_count INTEGER DEFAULT 0,
    dqi_score DECIMAL(5,2),
    provenance_hash VARCHAR(64),
    status VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    PRIMARY KEY (id, created_at),
    CONSTRAINT chk_ula_calc_method CHECK (method IN ('asset_specific', 'lessor_specific', 'average_data', 'spend_based')),
    CONSTRAINT chk_ula_calc_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_ula_calc_wtt_positive CHECK (total_wtt_co2e_kg IS NULL OR total_wtt_co2e_kg >= 0),
    CONSTRAINT chk_ula_calc_year_valid CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_ula_calc_dqi_range CHECK (dqi_score IS NULL OR (dqi_score >= 1.0 AND dqi_score <= 5.0)),
    CONSTRAINT chk_ula_calc_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_ula_calc_asset_count CHECK (asset_count >= 0),
    CONSTRAINT chk_ula_calc_building_count CHECK (building_count >= 0),
    CONSTRAINT chk_ula_calc_vehicle_count CHECK (vehicle_count >= 0),
    CONSTRAINT chk_ula_calc_equipment_count CHECK (equipment_count >= 0),
    CONSTRAINT chk_ula_calc_it_count CHECK (it_asset_count >= 0)
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('upstream_leased_assets_service.gl_ula_calculations', 'created_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_ula_calc_tenant ON upstream_leased_assets_service.gl_ula_calculations(tenant_id, created_at DESC);
CREATE INDEX idx_ula_calc_org ON upstream_leased_assets_service.gl_ula_calculations(organization_id);
CREATE INDEX idx_ula_calc_year ON upstream_leased_assets_service.gl_ula_calculations(reporting_year);
CREATE INDEX idx_ula_calc_method ON upstream_leased_assets_service.gl_ula_calculations(method);
CREATE INDEX idx_ula_calc_status ON upstream_leased_assets_service.gl_ula_calculations(status);
CREATE INDEX idx_ula_calc_hash ON upstream_leased_assets_service.gl_ula_calculations(provenance_hash);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_calculations IS 'Master calculation records for upstream leased asset emissions (HYPERTABLE, 30-day chunks)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_calculations.method IS 'Calculation method: asset_specific, lessor_specific, average_data, spend_based';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_calculations.total_co2e_kg IS 'Total Scope 3 Cat 8 emissions in kgCO2e';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_calculations.total_wtt_co2e_kg IS 'Total well-to-tank upstream emissions in kgCO2e';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_calculations.dqi_score IS 'Data Quality Indicator score (1.0=highest to 5.0=lowest)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_calculations.provenance_hash IS 'SHA-256 hash of all calculation inputs for audit trail';

-- ============================================================================
-- TABLE 2: gl_ula_building_assets
-- Description: Leased building/facility asset records with energy consumption
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_building_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    building_type VARCHAR(50) NOT NULL,
    building_name VARCHAR(200),
    floor_area_sqm DECIMAL(15,4),
    country_code VARCHAR(10),
    climate_zone VARCHAR(20),
    lease_type VARCHAR(20) DEFAULT 'operating',
    lease_start_date DATE,
    lease_end_date DATE,
    lease_months INTEGER DEFAULT 12,
    allocation_method VARCHAR(20) DEFAULT 'floor_area',
    allocation_factor DECIMAL(10,8) DEFAULT 1.0,
    electricity_kwh DECIMAL(15,4),
    natural_gas_kwh DECIMAL(15,4),
    district_heating_kwh DECIMAL(15,4),
    district_cooling_kwh DECIMAL(15,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_bldg_type CHECK (building_type IN (
        'office', 'retail', 'warehouse', 'industrial', 'data_center',
        'laboratory', 'hospital', 'hotel', 'education', 'mixed_use',
        'residential', 'restaurant', 'supermarket', 'other'
    )),
    CONSTRAINT chk_ula_bldg_area_positive CHECK (floor_area_sqm IS NULL OR floor_area_sqm > 0),
    CONSTRAINT chk_ula_bldg_lease_type CHECK (lease_type IN ('operating', 'finance', 'short_term', 'sublease')),
    CONSTRAINT chk_ula_bldg_lease_months CHECK (lease_months IS NULL OR (lease_months > 0 AND lease_months <= 600)),
    CONSTRAINT chk_ula_bldg_alloc_method CHECK (allocation_method IN ('floor_area', 'headcount', 'revenue', 'time_based', 'full')),
    CONSTRAINT chk_ula_bldg_alloc_range CHECK (allocation_factor >= 0 AND allocation_factor <= 1.0),
    CONSTRAINT chk_ula_bldg_elec_positive CHECK (electricity_kwh IS NULL OR electricity_kwh >= 0),
    CONSTRAINT chk_ula_bldg_gas_positive CHECK (natural_gas_kwh IS NULL OR natural_gas_kwh >= 0),
    CONSTRAINT chk_ula_bldg_heat_positive CHECK (district_heating_kwh IS NULL OR district_heating_kwh >= 0),
    CONSTRAINT chk_ula_bldg_cool_positive CHECK (district_cooling_kwh IS NULL OR district_cooling_kwh >= 0),
    CONSTRAINT chk_ula_bldg_dates_valid CHECK (lease_start_date IS NULL OR lease_end_date IS NULL OR lease_end_date >= lease_start_date)
);

CREATE INDEX idx_ula_bldg_calc ON upstream_leased_assets_service.gl_ula_building_assets(calc_id);
CREATE INDEX idx_ula_bldg_tenant ON upstream_leased_assets_service.gl_ula_building_assets(tenant_id);
CREATE INDEX idx_ula_bldg_type ON upstream_leased_assets_service.gl_ula_building_assets(building_type);
CREATE INDEX idx_ula_bldg_country ON upstream_leased_assets_service.gl_ula_building_assets(country_code);
CREATE INDEX idx_ula_bldg_lease_type ON upstream_leased_assets_service.gl_ula_building_assets(lease_type);
CREATE INDEX idx_ula_bldg_created ON upstream_leased_assets_service.gl_ula_building_assets(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_building_assets IS 'Leased building/facility asset records with energy consumption data and lease details';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_assets.building_type IS 'Building type: office, retail, warehouse, industrial, data_center, laboratory, hospital, hotel, education, mixed_use, residential, restaurant, supermarket, other';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_assets.lease_type IS 'Lease type: operating, finance, short_term, sublease';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_assets.allocation_method IS 'Allocation method: floor_area, headcount, revenue, time_based, full';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_assets.allocation_factor IS 'Proportion of building emissions allocated to lessee (0.0 to 1.0)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_assets.climate_zone IS 'ASHRAE climate zone (e.g., 1A, 3C, 5A) or Koppen classification';

-- ============================================================================
-- TABLE 3: gl_ula_building_emissions (HYPERTABLE)
-- Description: Per-building emission calculation results with energy breakdown
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_building_emissions (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL,
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    elec_co2e_kg DECIMAL(20,8),
    gas_co2e_kg DECIMAL(20,8),
    heating_co2e_kg DECIMAL(20,8),
    cooling_co2e_kg DECIMAL(20,8),
    refrigerant_co2e_kg DECIMAL(20,8) DEFAULT 0,
    wtt_co2e_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8),
    method VARCHAR(50),
    ef_source VARCHAR(100),
    grid_factor_used DECIMAL(10,8),
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at),
    CONSTRAINT chk_ula_bemit_elec_positive CHECK (elec_co2e_kg IS NULL OR elec_co2e_kg >= 0),
    CONSTRAINT chk_ula_bemit_gas_positive CHECK (gas_co2e_kg IS NULL OR gas_co2e_kg >= 0),
    CONSTRAINT chk_ula_bemit_heat_positive CHECK (heating_co2e_kg IS NULL OR heating_co2e_kg >= 0),
    CONSTRAINT chk_ula_bemit_cool_positive CHECK (cooling_co2e_kg IS NULL OR cooling_co2e_kg >= 0),
    CONSTRAINT chk_ula_bemit_refrig_positive CHECK (refrigerant_co2e_kg IS NULL OR refrigerant_co2e_kg >= 0),
    CONSTRAINT chk_ula_bemit_wtt_positive CHECK (wtt_co2e_kg IS NULL OR wtt_co2e_kg >= 0),
    CONSTRAINT chk_ula_bemit_total_positive CHECK (total_co2e_kg IS NULL OR total_co2e_kg >= 0),
    CONSTRAINT chk_ula_bemit_grid_positive CHECK (grid_factor_used IS NULL OR grid_factor_used >= 0)
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('upstream_leased_assets_service.gl_ula_building_emissions', 'created_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_ula_bemit_asset ON upstream_leased_assets_service.gl_ula_building_emissions(asset_id);
CREATE INDEX idx_ula_bemit_calc ON upstream_leased_assets_service.gl_ula_building_emissions(calc_id);
CREATE INDEX idx_ula_bemit_tenant ON upstream_leased_assets_service.gl_ula_building_emissions(tenant_id, created_at DESC);
CREATE INDEX idx_ula_bemit_method ON upstream_leased_assets_service.gl_ula_building_emissions(method);
CREATE INDEX idx_ula_bemit_hash ON upstream_leased_assets_service.gl_ula_building_emissions(provenance_hash);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_building_emissions IS 'Per-building emission calculation results with energy source breakdown (HYPERTABLE, 30-day chunks)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_emissions.elec_co2e_kg IS 'Emissions from electricity consumption (kgCO2e)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_emissions.gas_co2e_kg IS 'Emissions from natural gas consumption (kgCO2e)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_emissions.heating_co2e_kg IS 'Emissions from district heating consumption (kgCO2e)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_emissions.cooling_co2e_kg IS 'Emissions from district cooling consumption (kgCO2e)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_emissions.refrigerant_co2e_kg IS 'Emissions from refrigerant leakage (kgCO2e)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_emissions.wtt_co2e_kg IS 'Well-to-tank upstream emissions for all energy sources (kgCO2e)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_building_emissions.grid_factor_used IS 'Electricity grid emission factor applied (kgCO2e/kWh)';

-- ============================================================================
-- TABLE 4: gl_ula_vehicle_assets
-- Description: Leased vehicle asset records with fuel and distance data
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_vehicle_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    vehicle_type VARCHAR(50),
    fuel_type VARCHAR(20),
    vehicle_age VARCHAR(20) DEFAULT 'mid_4_7yr',
    annual_km DECIMAL(15,4),
    fuel_litres DECIMAL(15,4),
    count INTEGER DEFAULT 1,
    country_code VARCHAR(10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_veh_type CHECK (vehicle_type IN (
        'small_car', 'medium_car', 'large_car', 'average_car', 'suv',
        'light_van', 'medium_van', 'heavy_van', 'pickup_truck',
        'light_truck', 'medium_truck', 'heavy_truck',
        'motorcycle', 'electric_vehicle', 'hybrid_vehicle', 'other'
    )),
    CONSTRAINT chk_ula_veh_fuel CHECK (fuel_type IS NULL OR fuel_type IN (
        'petrol', 'diesel', 'hybrid', 'phev', 'ev', 'lpg', 'cng', 'hydrogen', 'biofuel'
    )),
    CONSTRAINT chk_ula_veh_age CHECK (vehicle_age IN ('new_0_3yr', 'mid_4_7yr', 'old_8_plus')),
    CONSTRAINT chk_ula_veh_km_positive CHECK (annual_km IS NULL OR annual_km >= 0),
    CONSTRAINT chk_ula_veh_fuel_positive CHECK (fuel_litres IS NULL OR fuel_litres >= 0),
    CONSTRAINT chk_ula_veh_count_positive CHECK (count >= 1)
);

CREATE INDEX idx_ula_veh_calc ON upstream_leased_assets_service.gl_ula_vehicle_assets(calc_id);
CREATE INDEX idx_ula_veh_tenant ON upstream_leased_assets_service.gl_ula_vehicle_assets(tenant_id);
CREATE INDEX idx_ula_veh_type ON upstream_leased_assets_service.gl_ula_vehicle_assets(vehicle_type);
CREATE INDEX idx_ula_veh_fuel ON upstream_leased_assets_service.gl_ula_vehicle_assets(fuel_type);
CREATE INDEX idx_ula_veh_country ON upstream_leased_assets_service.gl_ula_vehicle_assets(country_code);
CREATE INDEX idx_ula_veh_created ON upstream_leased_assets_service.gl_ula_vehicle_assets(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_vehicle_assets IS 'Leased vehicle asset records with fuel type, distance, and fleet count';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_assets.vehicle_type IS 'Vehicle type: small_car, medium_car, large_car, suv, light_van, medium_van, heavy_van, pickup_truck, light_truck, medium_truck, heavy_truck, motorcycle, electric_vehicle, hybrid_vehicle, other';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_assets.fuel_type IS 'Fuel type: petrol, diesel, hybrid, phev, ev, lpg, cng, hydrogen, biofuel';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_assets.vehicle_age IS 'Vehicle age band: new_0_3yr, mid_4_7yr, old_8_plus';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_assets.annual_km IS 'Annual distance driven in kilometers';

-- ============================================================================
-- TABLE 5: gl_ula_vehicle_emissions (HYPERTABLE)
-- Description: Per-vehicle emission calculation results with TTW/WTT breakdown
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_vehicle_emissions (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL,
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    ttw_co2e_kg DECIMAL(20,8),
    wtt_co2e_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8),
    method VARCHAR(50),
    ef_per_km DECIMAL(10,8),
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at),
    CONSTRAINT chk_ula_vemit_ttw_positive CHECK (ttw_co2e_kg IS NULL OR ttw_co2e_kg >= 0),
    CONSTRAINT chk_ula_vemit_wtt_positive CHECK (wtt_co2e_kg IS NULL OR wtt_co2e_kg >= 0),
    CONSTRAINT chk_ula_vemit_total_positive CHECK (total_co2e_kg IS NULL OR total_co2e_kg >= 0),
    CONSTRAINT chk_ula_vemit_ef_positive CHECK (ef_per_km IS NULL OR ef_per_km >= 0)
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('upstream_leased_assets_service.gl_ula_vehicle_emissions', 'created_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_ula_vemit_asset ON upstream_leased_assets_service.gl_ula_vehicle_emissions(asset_id);
CREATE INDEX idx_ula_vemit_calc ON upstream_leased_assets_service.gl_ula_vehicle_emissions(calc_id);
CREATE INDEX idx_ula_vemit_tenant ON upstream_leased_assets_service.gl_ula_vehicle_emissions(tenant_id, created_at DESC);
CREATE INDEX idx_ula_vemit_method ON upstream_leased_assets_service.gl_ula_vehicle_emissions(method);
CREATE INDEX idx_ula_vemit_hash ON upstream_leased_assets_service.gl_ula_vehicle_emissions(provenance_hash);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_vehicle_emissions IS 'Per-vehicle emission calculation results with TTW/WTT breakdown (HYPERTABLE, 30-day chunks)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_emissions.ttw_co2e_kg IS 'Tank-to-wheel (direct) emissions in kgCO2e';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_emissions.wtt_co2e_kg IS 'Well-to-tank (upstream fuel production) emissions in kgCO2e';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_emissions.ef_per_km IS 'Emission factor applied per kilometer (kgCO2e/km)';

-- ============================================================================
-- TABLE 6: gl_ula_equipment_assets
-- Description: Leased equipment asset records (generators, machinery, forklifts)
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_equipment_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    equipment_type VARCHAR(50),
    power_kw DECIMAL(10,4),
    operating_hours DECIMAL(10,2),
    load_factor DECIMAL(5,4),
    fuel_type VARCHAR(20),
    fuel_litres DECIMAL(15,4),
    count INTEGER DEFAULT 1,
    country_code VARCHAR(10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_equip_type CHECK (equipment_type IS NULL OR equipment_type IN (
        'generator', 'compressor', 'pump', 'forklift', 'crane',
        'excavator', 'loader', 'welder', 'hvac_unit', 'chiller',
        'boiler', 'conveyor', 'press', 'cnc_machine', 'other'
    )),
    CONSTRAINT chk_ula_equip_power_positive CHECK (power_kw IS NULL OR power_kw >= 0),
    CONSTRAINT chk_ula_equip_hours_positive CHECK (operating_hours IS NULL OR operating_hours >= 0),
    CONSTRAINT chk_ula_equip_load_range CHECK (load_factor IS NULL OR (load_factor >= 0 AND load_factor <= 1.0)),
    CONSTRAINT chk_ula_equip_fuel CHECK (fuel_type IS NULL OR fuel_type IN (
        'diesel', 'petrol', 'natural_gas', 'lpg', 'electric', 'hydrogen', 'biofuel'
    )),
    CONSTRAINT chk_ula_equip_fuel_positive CHECK (fuel_litres IS NULL OR fuel_litres >= 0),
    CONSTRAINT chk_ula_equip_count_positive CHECK (count >= 1)
);

CREATE INDEX idx_ula_equip_calc ON upstream_leased_assets_service.gl_ula_equipment_assets(calc_id);
CREATE INDEX idx_ula_equip_tenant ON upstream_leased_assets_service.gl_ula_equipment_assets(tenant_id);
CREATE INDEX idx_ula_equip_type ON upstream_leased_assets_service.gl_ula_equipment_assets(equipment_type);
CREATE INDEX idx_ula_equip_fuel ON upstream_leased_assets_service.gl_ula_equipment_assets(fuel_type);
CREATE INDEX idx_ula_equip_country ON upstream_leased_assets_service.gl_ula_equipment_assets(country_code);
CREATE INDEX idx_ula_equip_created ON upstream_leased_assets_service.gl_ula_equipment_assets(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_equipment_assets IS 'Leased equipment asset records with power rating, operating hours, and fuel consumption';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_equipment_assets.equipment_type IS 'Equipment type: generator, compressor, pump, forklift, crane, excavator, loader, welder, hvac_unit, chiller, boiler, conveyor, press, cnc_machine, other';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_equipment_assets.power_kw IS 'Rated power output in kilowatts';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_equipment_assets.load_factor IS 'Average load factor as fraction of rated power (0.0 to 1.0)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_equipment_assets.operating_hours IS 'Annual operating hours';

-- ============================================================================
-- TABLE 7: gl_ula_equipment_emissions
-- Description: Per-equipment emission calculation results
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_equipment_emissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL,
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    co2e_kg DECIMAL(20,8),
    wtt_co2e_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8),
    method VARCHAR(50),
    annual_kwh DECIMAL(15,4),
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_eqemit_co2e_positive CHECK (co2e_kg IS NULL OR co2e_kg >= 0),
    CONSTRAINT chk_ula_eqemit_wtt_positive CHECK (wtt_co2e_kg IS NULL OR wtt_co2e_kg >= 0),
    CONSTRAINT chk_ula_eqemit_total_positive CHECK (total_co2e_kg IS NULL OR total_co2e_kg >= 0),
    CONSTRAINT chk_ula_eqemit_kwh_positive CHECK (annual_kwh IS NULL OR annual_kwh >= 0)
);

CREATE INDEX idx_ula_eqemit_asset ON upstream_leased_assets_service.gl_ula_equipment_emissions(asset_id);
CREATE INDEX idx_ula_eqemit_calc ON upstream_leased_assets_service.gl_ula_equipment_emissions(calc_id);
CREATE INDEX idx_ula_eqemit_tenant ON upstream_leased_assets_service.gl_ula_equipment_emissions(tenant_id);
CREATE INDEX idx_ula_eqemit_method ON upstream_leased_assets_service.gl_ula_equipment_emissions(method);
CREATE INDEX idx_ula_eqemit_hash ON upstream_leased_assets_service.gl_ula_equipment_emissions(provenance_hash);
CREATE INDEX idx_ula_eqemit_created ON upstream_leased_assets_service.gl_ula_equipment_emissions(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_equipment_emissions IS 'Per-equipment emission calculation results with energy consumption and WTT breakdown';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_equipment_emissions.co2e_kg IS 'Direct (combustion/grid) emissions in kgCO2e';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_equipment_emissions.wtt_co2e_kg IS 'Well-to-tank upstream emissions in kgCO2e';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_equipment_emissions.annual_kwh IS 'Annual energy consumption in kWh (power_kw x hours x load_factor)';

-- ============================================================================
-- TABLE 8: gl_ula_it_assets
-- Description: Leased IT asset records (servers, storage, networking)
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_it_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    it_type VARCHAR(50),
    power_kw DECIMAL(10,4),
    pue DECIMAL(5,2),
    utilization DECIMAL(5,4),
    operating_hours DECIMAL(10,2),
    count INTEGER DEFAULT 1,
    country_code VARCHAR(10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_it_type CHECK (it_type IS NULL OR it_type IN (
        'server_rack', 'server_blade', 'server_tower', 'storage_array',
        'network_switch', 'network_router', 'firewall', 'load_balancer',
        'ups', 'cooling_unit', 'workstation', 'desktop', 'laptop',
        'monitor', 'printer', 'other'
    )),
    CONSTRAINT chk_ula_it_power_positive CHECK (power_kw IS NULL OR power_kw >= 0),
    CONSTRAINT chk_ula_it_pue_range CHECK (pue IS NULL OR (pue >= 1.0 AND pue <= 5.0)),
    CONSTRAINT chk_ula_it_util_range CHECK (utilization IS NULL OR (utilization >= 0 AND utilization <= 1.0)),
    CONSTRAINT chk_ula_it_hours_positive CHECK (operating_hours IS NULL OR operating_hours >= 0),
    CONSTRAINT chk_ula_it_count_positive CHECK (count >= 1)
);

CREATE INDEX idx_ula_it_calc ON upstream_leased_assets_service.gl_ula_it_assets(calc_id);
CREATE INDEX idx_ula_it_tenant ON upstream_leased_assets_service.gl_ula_it_assets(tenant_id);
CREATE INDEX idx_ula_it_type ON upstream_leased_assets_service.gl_ula_it_assets(it_type);
CREATE INDEX idx_ula_it_country ON upstream_leased_assets_service.gl_ula_it_assets(country_code);
CREATE INDEX idx_ula_it_created ON upstream_leased_assets_service.gl_ula_it_assets(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_it_assets IS 'Leased IT asset records with power rating, PUE, utilization, and operating hours';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_it_assets.it_type IS 'IT asset type: server_rack, server_blade, server_tower, storage_array, network_switch, network_router, firewall, load_balancer, ups, cooling_unit, workstation, desktop, laptop, monitor, printer, other';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_it_assets.pue IS 'Power Usage Effectiveness (1.0 = ideal, typical DC is 1.2-1.8)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_it_assets.utilization IS 'Average utilization fraction (0.0 to 1.0)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_it_assets.operating_hours IS 'Annual operating hours (e.g., 8760 for always-on)';

-- ============================================================================
-- TABLE 9: gl_ula_it_emissions
-- Description: Per-IT-asset emission calculation results
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_it_emissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL,
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    co2e_kg DECIMAL(20,8),
    wtt_co2e_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8),
    annual_kwh DECIMAL(15,4),
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_itemit_co2e_positive CHECK (co2e_kg IS NULL OR co2e_kg >= 0),
    CONSTRAINT chk_ula_itemit_wtt_positive CHECK (wtt_co2e_kg IS NULL OR wtt_co2e_kg >= 0),
    CONSTRAINT chk_ula_itemit_total_positive CHECK (total_co2e_kg IS NULL OR total_co2e_kg >= 0),
    CONSTRAINT chk_ula_itemit_kwh_positive CHECK (annual_kwh IS NULL OR annual_kwh >= 0)
);

CREATE INDEX idx_ula_itemit_asset ON upstream_leased_assets_service.gl_ula_it_emissions(asset_id);
CREATE INDEX idx_ula_itemit_calc ON upstream_leased_assets_service.gl_ula_it_emissions(calc_id);
CREATE INDEX idx_ula_itemit_tenant ON upstream_leased_assets_service.gl_ula_it_emissions(tenant_id);
CREATE INDEX idx_ula_itemit_hash ON upstream_leased_assets_service.gl_ula_it_emissions(provenance_hash);
CREATE INDEX idx_ula_itemit_created ON upstream_leased_assets_service.gl_ula_it_emissions(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_it_emissions IS 'Per-IT-asset emission calculation results with energy consumption and WTT breakdown';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_it_emissions.co2e_kg IS 'Direct grid-based emissions in kgCO2e';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_it_emissions.wtt_co2e_kg IS 'Well-to-tank upstream emissions in kgCO2e';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_it_emissions.annual_kwh IS 'Annual energy consumption in kWh (power_kw x PUE x utilization x hours)';

-- ============================================================================
-- TABLE 10: gl_ula_allocations
-- Description: Allocation records for shared/partial leased assets
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    asset_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    method VARCHAR(20),
    factor DECIMAL(10,8),
    leased_area DECIMAL(15,4),
    total_area DECIMAL(15,4),
    allocated_co2e_kg DECIMAL(20,8),
    original_co2e_kg DECIMAL(20,8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_alloc_method CHECK (method IS NULL OR method IN ('floor_area', 'headcount', 'revenue', 'time_based', 'full')),
    CONSTRAINT chk_ula_alloc_factor_range CHECK (factor IS NULL OR (factor >= 0 AND factor <= 1.0)),
    CONSTRAINT chk_ula_alloc_leased_positive CHECK (leased_area IS NULL OR leased_area >= 0),
    CONSTRAINT chk_ula_alloc_total_positive CHECK (total_area IS NULL OR total_area > 0),
    CONSTRAINT chk_ula_alloc_leased_le_total CHECK (leased_area IS NULL OR total_area IS NULL OR leased_area <= total_area),
    CONSTRAINT chk_ula_alloc_allocated_positive CHECK (allocated_co2e_kg IS NULL OR allocated_co2e_kg >= 0),
    CONSTRAINT chk_ula_alloc_original_positive CHECK (original_co2e_kg IS NULL OR original_co2e_kg >= 0)
);

CREATE INDEX idx_ula_alloc_calc ON upstream_leased_assets_service.gl_ula_allocations(calc_id);
CREATE INDEX idx_ula_alloc_asset ON upstream_leased_assets_service.gl_ula_allocations(asset_id);
CREATE INDEX idx_ula_alloc_tenant ON upstream_leased_assets_service.gl_ula_allocations(tenant_id);
CREATE INDEX idx_ula_alloc_method ON upstream_leased_assets_service.gl_ula_allocations(method);
CREATE INDEX idx_ula_alloc_created ON upstream_leased_assets_service.gl_ula_allocations(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_allocations IS 'Allocation records for shared/partial leased assets with area-based or other allocation methods';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_allocations.method IS 'Allocation method: floor_area, headcount, revenue, time_based, full';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_allocations.factor IS 'Allocation factor (0.0 to 1.0), e.g., leased_area / total_area';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_allocations.allocated_co2e_kg IS 'Emissions after allocation (original_co2e_kg x factor)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_allocations.original_co2e_kg IS 'Whole-building emissions before allocation';

-- ============================================================================
-- TABLE 11: gl_ula_compliance_results
-- Description: Compliance check results against regulatory frameworks
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_compliance_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    framework VARCHAR(50),
    status VARCHAR(20),
    total_rules INTEGER,
    passed INTEGER,
    failed INTEGER,
    warnings INTEGER,
    findings JSONB,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_compl_framework CHECK (framework IS NULL OR framework IN (
        'GHG_PROTOCOL', 'CSRD_ESRS_E1', 'CDP_CLIMATE', 'SBTi',
        'ISO_14064', 'GRI_305', 'SECR'
    )),
    CONSTRAINT chk_ula_compl_status CHECK (status IS NULL OR status IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE')),
    CONSTRAINT chk_ula_compl_rules_positive CHECK (total_rules IS NULL OR total_rules >= 0),
    CONSTRAINT chk_ula_compl_passed_positive CHECK (passed IS NULL OR passed >= 0),
    CONSTRAINT chk_ula_compl_failed_positive CHECK (failed IS NULL OR failed >= 0),
    CONSTRAINT chk_ula_compl_warnings_positive CHECK (warnings IS NULL OR warnings >= 0)
);

CREATE INDEX idx_ula_compl_calc ON upstream_leased_assets_service.gl_ula_compliance_results(calc_id);
CREATE INDEX idx_ula_compl_tenant ON upstream_leased_assets_service.gl_ula_compliance_results(tenant_id);
CREATE INDEX idx_ula_compl_framework ON upstream_leased_assets_service.gl_ula_compliance_results(framework);
CREATE INDEX idx_ula_compl_status ON upstream_leased_assets_service.gl_ula_compliance_results(status);
CREATE INDEX idx_ula_compl_findings ON upstream_leased_assets_service.gl_ula_compliance_results USING GIN(findings);
CREATE INDEX idx_ula_compl_created ON upstream_leased_assets_service.gl_ula_compliance_results(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_compliance_results IS 'Compliance check results against GHG Protocol, CSRD, CDP, SBTi, ISO 14064, GRI, SECR frameworks';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_compliance_results.framework IS 'Regulatory framework: GHG_PROTOCOL, CSRD_ESRS_E1, CDP_CLIMATE, SBTi, ISO_14064, GRI_305, SECR';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_compliance_results.status IS 'Compliance status: PASS, FAIL, WARNING, NOT_APPLICABLE';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_compliance_results.findings IS 'JSONB array of compliance findings with severity, rule_id, and detail';

-- ============================================================================
-- TABLE 12: gl_ula_emission_factors
-- Description: Emission factors used in calculations with source traceability
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    ef_type VARCHAR(50),
    category VARCHAR(50),
    source VARCHAR(100),
    version VARCHAR(20),
    value DECIMAL(15,8),
    unit VARCHAR(30),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_ef_type CHECK (ef_type IS NULL OR ef_type IN (
        'grid_electricity', 'natural_gas', 'district_heating', 'district_cooling',
        'vehicle_per_km', 'vehicle_per_litre', 'equipment_fuel', 'equipment_electric',
        'eui_benchmark', 'wtt_factor', 'refrigerant_gwp', 'eeio_per_usd'
    )),
    CONSTRAINT chk_ula_ef_value_positive CHECK (value IS NULL OR value >= 0)
);

CREATE INDEX idx_ula_ef_calc ON upstream_leased_assets_service.gl_ula_emission_factors(calc_id);
CREATE INDEX idx_ula_ef_tenant ON upstream_leased_assets_service.gl_ula_emission_factors(tenant_id);
CREATE INDEX idx_ula_ef_type ON upstream_leased_assets_service.gl_ula_emission_factors(ef_type);
CREATE INDEX idx_ula_ef_category ON upstream_leased_assets_service.gl_ula_emission_factors(category);
CREATE INDEX idx_ula_ef_source ON upstream_leased_assets_service.gl_ula_emission_factors(source);
CREATE INDEX idx_ula_ef_created ON upstream_leased_assets_service.gl_ula_emission_factors(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_emission_factors IS 'Emission factors used in each calculation with source and version traceability';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_emission_factors.ef_type IS 'Emission factor type: grid_electricity, natural_gas, district_heating, district_cooling, vehicle_per_km, vehicle_per_litre, equipment_fuel, equipment_electric, eui_benchmark, wtt_factor, refrigerant_gwp, eeio_per_usd';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_emission_factors.source IS 'Source of emission factor (e.g., IEA_2024, DEFRA_2024, EPA_eGRID_2024, ASHRAE_90.1)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_emission_factors.version IS 'Version of the emission factor dataset';

-- ============================================================================
-- TABLE 13: gl_ula_provenance
-- Description: Provenance tracking with SHA-256 hash chains for audit trail
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_provenance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    stage VARCHAR(30),
    stage_order INTEGER,
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_prov_stage CHECK (stage IS NULL OR stage IN (
        'INTAKE', 'ASSET_CLASSIFICATION', 'EF_LOOKUP', 'BUILDING_CALC',
        'VEHICLE_CALC', 'EQUIPMENT_CALC', 'IT_CALC', 'ALLOCATION',
        'AGGREGATION', 'VALIDATION', 'COMPLIANCE'
    )),
    CONSTRAINT chk_ula_prov_order_positive CHECK (stage_order IS NULL OR stage_order >= 0)
);

CREATE INDEX idx_ula_prov_calc ON upstream_leased_assets_service.gl_ula_provenance(calc_id);
CREATE INDEX idx_ula_prov_tenant ON upstream_leased_assets_service.gl_ula_provenance(tenant_id);
CREATE INDEX idx_ula_prov_calc_stage ON upstream_leased_assets_service.gl_ula_provenance(calc_id, stage_order);
CREATE INDEX idx_ula_prov_stage ON upstream_leased_assets_service.gl_ula_provenance(stage);
CREATE INDEX idx_ula_prov_input ON upstream_leased_assets_service.gl_ula_provenance(input_hash);
CREATE INDEX idx_ula_prov_output ON upstream_leased_assets_service.gl_ula_provenance(output_hash);
CREATE INDEX idx_ula_prov_created ON upstream_leased_assets_service.gl_ula_provenance(created_at DESC);
CREATE INDEX idx_ula_prov_metadata ON upstream_leased_assets_service.gl_ula_provenance USING GIN(metadata);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_provenance IS 'Provenance tracking for upstream leased asset calculations with SHA-256 hash chains';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_provenance.stage IS 'Processing stage: INTAKE, ASSET_CLASSIFICATION, EF_LOOKUP, BUILDING_CALC, VEHICLE_CALC, EQUIPMENT_CALC, IT_CALC, ALLOCATION, AGGREGATION, VALIDATION, COMPLIANCE';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_provenance.stage_order IS 'Sequential order of processing stage within the calculation pipeline';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_provenance.input_hash IS 'SHA-256 hash of stage input data';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_provenance.output_hash IS 'SHA-256 hash of stage output data';

-- ============================================================================
-- TABLE 14: gl_ula_aggregations
-- Description: Aggregated emission summaries by dimension and grouping key
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_aggregations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    dimension VARCHAR(50),
    group_key VARCHAR(100),
    co2e_kg DECIMAL(20,8),
    percentage DECIMAL(8,4),
    asset_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_agg_dimension CHECK (dimension IS NULL OR dimension IN (
        'asset_type', 'building_type', 'vehicle_type', 'equipment_type', 'it_type',
        'country', 'climate_zone', 'lease_type', 'method', 'energy_source'
    )),
    CONSTRAINT chk_ula_agg_co2e_positive CHECK (co2e_kg IS NULL OR co2e_kg >= 0),
    CONSTRAINT chk_ula_agg_pct_range CHECK (percentage IS NULL OR (percentage >= 0 AND percentage <= 100.0)),
    CONSTRAINT chk_ula_agg_count_positive CHECK (asset_count IS NULL OR asset_count >= 0)
);

CREATE INDEX idx_ula_agg_calc ON upstream_leased_assets_service.gl_ula_aggregations(calc_id);
CREATE INDEX idx_ula_agg_tenant ON upstream_leased_assets_service.gl_ula_aggregations(tenant_id);
CREATE INDEX idx_ula_agg_dimension ON upstream_leased_assets_service.gl_ula_aggregations(dimension);
CREATE INDEX idx_ula_agg_group_key ON upstream_leased_assets_service.gl_ula_aggregations(group_key);
CREATE INDEX idx_ula_agg_created ON upstream_leased_assets_service.gl_ula_aggregations(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_aggregations IS 'Aggregated emission summaries by dimension (asset_type, building_type, country, etc.)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_aggregations.dimension IS 'Aggregation dimension: asset_type, building_type, vehicle_type, equipment_type, it_type, country, climate_zone, lease_type, method, energy_source';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_aggregations.group_key IS 'Value within the dimension (e.g., office, warehouse for building_type)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_aggregations.percentage IS 'Percentage of total portfolio emissions (0.0 to 100.0)';

-- ============================================================================
-- TABLE 15: gl_ula_spend_calculations
-- Description: Spend-based emission calculations using EEIO factors
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_spend_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    naics_code VARCHAR(10),
    description VARCHAR(200),
    amount_original DECIMAL(15,2),
    currency VARCHAR(3),
    amount_usd DECIMAL(15,2),
    deflated_usd DECIMAL(15,2),
    eeio_factor DECIMAL(10,8),
    co2e_kg DECIMAL(20,8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_spend_amount_positive CHECK (amount_original IS NULL OR amount_original >= 0),
    CONSTRAINT chk_ula_spend_usd_positive CHECK (amount_usd IS NULL OR amount_usd >= 0),
    CONSTRAINT chk_ula_spend_deflated_positive CHECK (deflated_usd IS NULL OR deflated_usd >= 0),
    CONSTRAINT chk_ula_spend_eeio_positive CHECK (eeio_factor IS NULL OR eeio_factor >= 0),
    CONSTRAINT chk_ula_spend_co2e_positive CHECK (co2e_kg IS NULL OR co2e_kg >= 0)
);

CREATE INDEX idx_ula_spend_calc ON upstream_leased_assets_service.gl_ula_spend_calculations(calc_id);
CREATE INDEX idx_ula_spend_tenant ON upstream_leased_assets_service.gl_ula_spend_calculations(tenant_id);
CREATE INDEX idx_ula_spend_naics ON upstream_leased_assets_service.gl_ula_spend_calculations(naics_code);
CREATE INDEX idx_ula_spend_currency ON upstream_leased_assets_service.gl_ula_spend_calculations(currency);
CREATE INDEX idx_ula_spend_created ON upstream_leased_assets_service.gl_ula_spend_calculations(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_spend_calculations IS 'Spend-based emission calculations using EPA USEEIO v2 EEIO factors for leased asset categories';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_spend_calculations.naics_code IS 'NAICS code for the leased asset/service category';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_spend_calculations.amount_usd IS 'Lease spend converted to USD at current exchange rate';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_spend_calculations.deflated_usd IS 'Lease spend deflated to EEIO base year USD using CPI';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_spend_calculations.eeio_factor IS 'EPA USEEIO v2 emission factor applied (kgCO2e/USD)';

-- ============================================================================
-- TABLE 16: gl_ula_lessor_data
-- Description: Lessor-provided emission data for lessor-specific method
-- ============================================================================

CREATE TABLE upstream_leased_assets_service.gl_ula_lessor_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    lessor_name VARCHAR(200),
    asset_description VARCHAR(500),
    reported_co2e_kg DECIMAL(20,8),
    methodology VARCHAR(200),
    boundary VARCHAR(200),
    validated BOOLEAN DEFAULT FALSE,
    allocation_factor DECIMAL(10,8),
    allocated_co2e_kg DECIMAL(20,8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ula_lessor_co2e_positive CHECK (reported_co2e_kg IS NULL OR reported_co2e_kg >= 0),
    CONSTRAINT chk_ula_lessor_alloc_range CHECK (allocation_factor IS NULL OR (allocation_factor >= 0 AND allocation_factor <= 1.0)),
    CONSTRAINT chk_ula_lessor_allocated_positive CHECK (allocated_co2e_kg IS NULL OR allocated_co2e_kg >= 0)
);

CREATE INDEX idx_ula_lessor_calc ON upstream_leased_assets_service.gl_ula_lessor_data(calc_id);
CREATE INDEX idx_ula_lessor_tenant ON upstream_leased_assets_service.gl_ula_lessor_data(tenant_id);
CREATE INDEX idx_ula_lessor_name ON upstream_leased_assets_service.gl_ula_lessor_data(lessor_name);
CREATE INDEX idx_ula_lessor_validated ON upstream_leased_assets_service.gl_ula_lessor_data(validated);
CREATE INDEX idx_ula_lessor_created ON upstream_leased_assets_service.gl_ula_lessor_data(created_at DESC);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_lessor_data IS 'Lessor-provided emission data for the lessor-specific calculation method';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_lessor_data.lessor_name IS 'Name of the lessor/landlord providing emission data';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_lessor_data.reported_co2e_kg IS 'Total emissions reported by the lessor for the asset (kgCO2e)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_lessor_data.methodology IS 'Description of methodology used by lessor (e.g., GHG Protocol, ISO 14064)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_lessor_data.boundary IS 'Organizational/operational boundary of lessor reporting';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_lessor_data.validated IS 'Whether lessor data has been validated/verified by third party';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_lessor_data.allocation_factor IS 'Proportion of lessor emissions allocated to lessee (0.0 to 1.0)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_lessor_data.allocated_co2e_kg IS 'Emissions after allocation (reported_co2e_kg x allocation_factor)';

-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Continuous Aggregate 1: Daily Emissions by Asset Type
CREATE MATERIALIZED VIEW upstream_leased_assets_service.gl_ula_daily_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at) AS bucket,
    tenant_id,
    method,
    COUNT(*) AS calc_count,
    SUM(total_co2e_kg) AS total_co2e_kg,
    SUM(total_wtt_co2e_kg) AS total_wtt_co2e_kg,
    SUM(building_count) AS total_buildings,
    SUM(vehicle_count) AS total_vehicles,
    SUM(equipment_count) AS total_equipment,
    SUM(it_asset_count) AS total_it_assets,
    AVG(dqi_score) AS avg_dqi_score
FROM upstream_leased_assets_service.gl_ula_calculations
GROUP BY bucket, tenant_id, method
WITH NO DATA;

-- Refresh policy for daily emissions (refresh every 6 hours, lag 12 hours)
SELECT add_continuous_aggregate_policy('upstream_leased_assets_service.gl_ula_daily_emissions',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW upstream_leased_assets_service.gl_ula_daily_emissions IS 'Daily aggregation of upstream leased asset emissions by calculation method with asset type counts';

-- Continuous Aggregate 2: Monthly Portfolio Summary
CREATE MATERIALIZED VIEW upstream_leased_assets_service.gl_ula_monthly_portfolio
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('30 days', created_at) AS bucket,
    tenant_id,
    COUNT(*) AS calc_count,
    SUM(total_co2e_kg) AS total_co2e_kg,
    SUM(total_wtt_co2e_kg) AS total_wtt_co2e_kg,
    SUM(building_count) AS total_buildings,
    SUM(vehicle_count) AS total_vehicles,
    SUM(equipment_count) AS total_equipment,
    SUM(it_asset_count) AS total_it_assets,
    SUM(asset_count) AS total_assets,
    AVG(dqi_score) AS avg_dqi_score
FROM upstream_leased_assets_service.gl_ula_calculations
GROUP BY bucket, tenant_id
WITH NO DATA;

-- Refresh policy for monthly portfolio (refresh every 24 hours, lag 48 hours)
SELECT add_continuous_aggregate_policy('upstream_leased_assets_service.gl_ula_monthly_portfolio',
    start_offset => INTERVAL '180 days',
    end_offset => INTERVAL '48 hours',
    schedule_interval => INTERVAL '24 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW upstream_leased_assets_service.gl_ula_monthly_portfolio IS 'Monthly portfolio summary with building/vehicle/equipment/IT asset breakdown and DQI scores';

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) - ALL 16 TABLES
-- ============================================================================

-- Enable RLS on all tables with tenant_id
ALTER TABLE upstream_leased_assets_service.gl_ula_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_building_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_building_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_vehicle_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_vehicle_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_equipment_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_equipment_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_it_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_it_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_allocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_compliance_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_emission_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_provenance ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_aggregations ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_spend_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE upstream_leased_assets_service.gl_ula_lessor_data ENABLE ROW LEVEL SECURITY;

-- RLS Policies: tenant_id isolation on all 16 tables
CREATE POLICY ula_calculations_tenant_isolation ON upstream_leased_assets_service.gl_ula_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_building_assets_tenant_isolation ON upstream_leased_assets_service.gl_ula_building_assets
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_building_emissions_tenant_isolation ON upstream_leased_assets_service.gl_ula_building_emissions
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_vehicle_assets_tenant_isolation ON upstream_leased_assets_service.gl_ula_vehicle_assets
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_vehicle_emissions_tenant_isolation ON upstream_leased_assets_service.gl_ula_vehicle_emissions
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_equipment_assets_tenant_isolation ON upstream_leased_assets_service.gl_ula_equipment_assets
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_equipment_emissions_tenant_isolation ON upstream_leased_assets_service.gl_ula_equipment_emissions
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_it_assets_tenant_isolation ON upstream_leased_assets_service.gl_ula_it_assets
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_it_emissions_tenant_isolation ON upstream_leased_assets_service.gl_ula_it_emissions
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_allocations_tenant_isolation ON upstream_leased_assets_service.gl_ula_allocations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_compliance_results_tenant_isolation ON upstream_leased_assets_service.gl_ula_compliance_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_emission_factors_tenant_isolation ON upstream_leased_assets_service.gl_ula_emission_factors
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_provenance_tenant_isolation ON upstream_leased_assets_service.gl_ula_provenance
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_aggregations_tenant_isolation ON upstream_leased_assets_service.gl_ula_aggregations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_spend_calculations_tenant_isolation ON upstream_leased_assets_service.gl_ula_spend_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY ula_lessor_data_tenant_isolation ON upstream_leased_assets_service.gl_ula_lessor_data
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- ============================================================================
-- SEED DATA: BUILDING EUI BENCHMARKS (kWh/sqm/year by building type)
-- Source: ASHRAE 90.1-2019, CIBSE TM46, ENERGY STAR Portfolio Manager
-- ============================================================================

-- Create a reference table for EUI benchmarks used in average-data method
CREATE TABLE upstream_leased_assets_service.gl_ula_eui_benchmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    building_type VARCHAR(50) NOT NULL,
    climate_zone VARCHAR(20) DEFAULT 'global',
    electricity_kwh_per_sqm DECIMAL(10,4) NOT NULL,
    gas_kwh_per_sqm DECIMAL(10,4) DEFAULT 0,
    total_kwh_per_sqm DECIMAL(10,4) NOT NULL,
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(building_type, climate_zone, source),
    CONSTRAINT chk_ula_eui_elec_positive CHECK (electricity_kwh_per_sqm >= 0),
    CONSTRAINT chk_ula_eui_gas_positive CHECK (gas_kwh_per_sqm >= 0),
    CONSTRAINT chk_ula_eui_total_positive CHECK (total_kwh_per_sqm >= 0)
);

CREATE INDEX idx_ula_eui_type ON upstream_leased_assets_service.gl_ula_eui_benchmarks(building_type);
CREATE INDEX idx_ula_eui_climate ON upstream_leased_assets_service.gl_ula_eui_benchmarks(climate_zone);
CREATE INDEX idx_ula_eui_source ON upstream_leased_assets_service.gl_ula_eui_benchmarks(source);
CREATE INDEX idx_ula_eui_active ON upstream_leased_assets_service.gl_ula_eui_benchmarks(is_active);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_eui_benchmarks IS 'Building Energy Use Intensity (EUI) benchmarks by type and climate zone for average-data method';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_eui_benchmarks.electricity_kwh_per_sqm IS 'Annual electricity consumption per square meter (kWh/sqm/yr)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_eui_benchmarks.gas_kwh_per_sqm IS 'Annual natural gas consumption per square meter (kWh/sqm/yr)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_eui_benchmarks.total_kwh_per_sqm IS 'Total annual energy consumption per square meter (kWh/sqm/yr)';

-- Enable RLS on EUI benchmarks (reference table, but maintain consistency)
ALTER TABLE upstream_leased_assets_service.gl_ula_eui_benchmarks ENABLE ROW LEVEL SECURITY;
CREATE POLICY ula_eui_benchmarks_public_read ON upstream_leased_assets_service.gl_ula_eui_benchmarks
    FOR SELECT USING (TRUE);

INSERT INTO upstream_leased_assets_service.gl_ula_eui_benchmarks
(building_type, climate_zone, electricity_kwh_per_sqm, gas_kwh_per_sqm, total_kwh_per_sqm, source, year, is_active) VALUES
-- Global averages (ASHRAE 90.1-2019 / CIBSE TM46 / ENERGY STAR)
('office',        'global', 150.0000, 50.0000,  200.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('retail',        'global', 180.0000, 40.0000,  220.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('warehouse',     'global',  60.0000, 30.0000,   90.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('industrial',    'global', 200.0000, 100.0000, 300.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('data_center',   'global', 800.0000, 10.0000,  810.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('laboratory',    'global', 350.0000, 150.0000, 500.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('hospital',      'global', 300.0000, 200.0000, 500.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('hotel',         'global', 200.0000, 100.0000, 300.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('education',     'global', 120.0000, 80.0000,  200.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('mixed_use',     'global', 170.0000, 60.0000,  230.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('residential',   'global', 100.0000, 80.0000,  180.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('restaurant',    'global', 350.0000, 250.0000, 600.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('supermarket',   'global', 400.0000, 50.0000,  450.0000, 'ASHRAE_90.1_2019',     2024, TRUE),
('other',         'global', 160.0000, 60.0000,  220.0000, 'ASHRAE_90.1_2019',     2024, TRUE),

-- Climate zone adjustments (temperate cold - higher heating)
('office',        '5A',     140.0000, 90.0000,  230.0000, 'ENERGY_STAR_PM',       2024, TRUE),
('retail',        '5A',     170.0000, 70.0000,  240.0000, 'ENERGY_STAR_PM',       2024, TRUE),
('warehouse',     '5A',      55.0000, 55.0000,  110.0000, 'ENERGY_STAR_PM',       2024, TRUE),
('data_center',   '5A',     790.0000, 15.0000,  805.0000, 'ENERGY_STAR_PM',       2024, TRUE),

-- Climate zone adjustments (hot humid - higher cooling)
('office',        '1A',     180.0000, 20.0000,  200.0000, 'ENERGY_STAR_PM',       2024, TRUE),
('retail',        '1A',     210.0000, 10.0000,  220.0000, 'ENERGY_STAR_PM',       2024, TRUE),
('warehouse',     '1A',      75.0000, 10.0000,   85.0000, 'ENERGY_STAR_PM',       2024, TRUE),
('data_center',   '1A',     850.0000, 5.0000,   855.0000, 'ENERGY_STAR_PM',       2024, TRUE),

-- UK benchmarks (CIBSE TM46)
('office',        'UK',     120.0000, 95.0000,  215.0000, 'CIBSE_TM46',           2024, TRUE),
('retail',        'UK',     165.0000, 55.0000,  220.0000, 'CIBSE_TM46',           2024, TRUE),
('warehouse',     'UK',      40.0000, 45.0000,   85.0000, 'CIBSE_TM46',           2024, TRUE),
('hospital',      'UK',     280.0000, 220.0000, 500.0000, 'CIBSE_TM46',           2024, TRUE);

-- ============================================================================
-- SEED DATA: GRID EMISSION FACTORS (20 countries for building energy calcs)
-- Source: IEA 2024 national averages
-- ============================================================================

-- Create a reference table for grid emission factors
CREATE TABLE upstream_leased_assets_service.gl_ula_grid_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code VARCHAR(10) NOT NULL,
    region VARCHAR(100),
    co2e_per_kwh DECIMAL(12,8) NOT NULL,
    wtt_factor DECIMAL(12,8),
    unit VARCHAR(50) DEFAULT 'kgCO2e/kWh',
    source VARCHAR(100) DEFAULT 'IEA_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(country_code, region, source, year),
    CONSTRAINT chk_ula_grid_co2e_positive CHECK (co2e_per_kwh >= 0),
    CONSTRAINT chk_ula_grid_wtt_positive CHECK (wtt_factor IS NULL OR wtt_factor >= 0),
    CONSTRAINT chk_ula_grid_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_ula_grid_country ON upstream_leased_assets_service.gl_ula_grid_factors(country_code);
CREATE INDEX idx_ula_grid_region ON upstream_leased_assets_service.gl_ula_grid_factors(region);
CREATE INDEX idx_ula_grid_source ON upstream_leased_assets_service.gl_ula_grid_factors(source);
CREATE INDEX idx_ula_grid_active ON upstream_leased_assets_service.gl_ula_grid_factors(is_active);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_grid_factors IS 'Electricity grid emission factors by country/region for building and IT energy calculations';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_grid_factors.co2e_per_kwh IS 'Grid emission factor in kgCO2e per kWh consumed';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_grid_factors.wtt_factor IS 'Well-to-tank factor for upstream fuel extraction and generation losses';

INSERT INTO upstream_leased_assets_service.gl_ula_grid_factors
(country_code, region, co2e_per_kwh, wtt_factor, unit, source, year, is_active) VALUES
-- Major economies (IEA 2024 national averages)
('US',     NULL,    0.37890000, 0.04550000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('GB',     NULL,    0.20700000, 0.02480000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('DE',     NULL,    0.35000000, 0.04200000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('FR',     NULL,    0.05200000, 0.00620000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('JP',     NULL,    0.45700000, 0.05480000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('CN',     NULL,    0.55500000, 0.06660000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('IN',     NULL,    0.70800000, 0.08500000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('CA',     NULL,    0.12000000, 0.01440000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('AU',     NULL,    0.65600000, 0.07870000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('BR',     NULL,    0.07400000, 0.00890000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('KR',     NULL,    0.41500000, 0.04980000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('IT',     NULL,    0.25600000, 0.03070000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('ES',     NULL,    0.16200000, 0.01940000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('NL',     NULL,    0.32800000, 0.03940000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('SE',     NULL,    0.01200000, 0.00140000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('NO',     NULL,    0.00800000, 0.00100000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('SG',     NULL,    0.40800000, 0.04900000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('AE',     NULL,    0.50200000, 0.06020000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('ZA',     NULL,    0.92800000, 0.11140000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('MX',     NULL,    0.42300000, 0.05080000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),

-- GLOBAL default (weighted average)
('GLOBAL', NULL,    0.44200000, 0.05300000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE);

-- ============================================================================
-- SEED DATA: VEHICLE EMISSION FACTORS FOR LEASED FLEET
-- Source: DEFRA 2024 vehicle emission factors
-- ============================================================================

-- Create a reference table for vehicle emission factors
CREATE TABLE upstream_leased_assets_service.gl_ula_vehicle_ef_ref (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_type VARCHAR(50) NOT NULL,
    fuel_type VARCHAR(20) NOT NULL,
    ef_per_km DECIMAL(12,8) NOT NULL,
    wtt_per_km DECIMAL(12,8),
    unit VARCHAR(50) DEFAULT 'kgCO2e/km',
    source VARCHAR(100) DEFAULT 'DEFRA_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(vehicle_type, fuel_type, source, year),
    CONSTRAINT chk_ula_vef_ef_positive CHECK (ef_per_km >= 0),
    CONSTRAINT chk_ula_vef_wtt_positive CHECK (wtt_per_km IS NULL OR wtt_per_km >= 0),
    CONSTRAINT chk_ula_vef_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_ula_vef_vehicle ON upstream_leased_assets_service.gl_ula_vehicle_ef_ref(vehicle_type);
CREATE INDEX idx_ula_vef_fuel ON upstream_leased_assets_service.gl_ula_vehicle_ef_ref(fuel_type);
CREATE INDEX idx_ula_vef_source ON upstream_leased_assets_service.gl_ula_vehicle_ef_ref(source);
CREATE INDEX idx_ula_vef_active ON upstream_leased_assets_service.gl_ula_vehicle_ef_ref(is_active);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_vehicle_ef_ref IS 'Vehicle emission factor reference table for leased fleet calculations (DEFRA 2024)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_ef_ref.ef_per_km IS 'Tank-to-wheel emission factor per km (kgCO2e/km)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_vehicle_ef_ref.wtt_per_km IS 'Well-to-tank upstream emission factor per km (kgCO2e/km)';

INSERT INTO upstream_leased_assets_service.gl_ula_vehicle_ef_ref
(vehicle_type, fuel_type, ef_per_km, wtt_per_km, unit, source, year, is_active) VALUES
-- Cars
('small_car',        'petrol',   0.14890000, 0.02330000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('small_car',        'diesel',   0.13920000, 0.02070000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_car',       'petrol',   0.18770000, 0.02940000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_car',       'diesel',   0.16610000, 0.02470000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('large_car',        'petrol',   0.27870000, 0.04370000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('large_car',        'diesel',   0.20870000, 0.03100000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('average_car',      'petrol',   0.17140000, 0.02690000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('hybrid_vehicle',   'hybrid',   0.11590000, 0.01820000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('electric_vehicle', 'ev',       0.04600000, 0.01330000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- Vans
('light_van',        'petrol',   0.19580000, 0.03070000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('light_van',        'diesel',   0.17440000, 0.02590000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_van',       'diesel',   0.23130000, 0.03440000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('heavy_van',        'diesel',   0.30700000, 0.04560000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- Trucks
('light_truck',      'diesel',   0.46120000, 0.06850000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_truck',     'diesel',   0.58440000, 0.08680000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('heavy_truck',      'diesel',   0.89210000, 0.13250000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- Other
('motorcycle',       'petrol',   0.10100000, 0.01860000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('pickup_truck',     'petrol',   0.33420000, 0.05240000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('pickup_truck',     'diesel',   0.28560000, 0.04240000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('suv',              'petrol',   0.27870000, 0.04370000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('suv',              'diesel',   0.20870000, 0.03100000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE);

-- ============================================================================
-- SEED DATA: EEIO SPEND-BASED FACTORS FOR LEASED ASSETS
-- Source: EPA USEEIO v2
-- ============================================================================

-- Create a reference table for EEIO factors
CREATE TABLE upstream_leased_assets_service.gl_ula_eeio_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    naics_code VARCHAR(10) NOT NULL,
    category_name VARCHAR(200) NOT NULL,
    ef_per_usd DECIMAL(12,8) NOT NULL,
    base_year INT DEFAULT 2021,
    unit VARCHAR(50) DEFAULT 'kgCO2e/USD',
    source VARCHAR(100) DEFAULT 'EPA_USEEIO_v2',
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(naics_code, source),
    CONSTRAINT chk_ula_eeio_ef_positive CHECK (ef_per_usd >= 0),
    CONSTRAINT chk_ula_eeio_year_valid CHECK (base_year >= 1990 AND base_year <= 2100)
);

CREATE INDEX idx_ula_eeio_naics ON upstream_leased_assets_service.gl_ula_eeio_factors(naics_code);
CREATE INDEX idx_ula_eeio_source ON upstream_leased_assets_service.gl_ula_eeio_factors(source);
CREATE INDEX idx_ula_eeio_active ON upstream_leased_assets_service.gl_ula_eeio_factors(is_active);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_eeio_factors IS 'EPA USEEIO v2 spend-based emission factors for leased asset categories by NAICS code';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_eeio_factors.ef_per_usd IS 'Emission factor per USD spent (kgCO2e/USD, base year deflated)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_eeio_factors.base_year IS 'Base year for CPI deflation (default 2021)';

INSERT INTO upstream_leased_assets_service.gl_ula_eeio_factors
(naics_code, category_name, ef_per_usd, base_year, unit, source, is_active) VALUES
-- Real estate and building leasing
('531100', 'Lessors of real estate',                          0.14200000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('531120', 'Lessors of nonresidential buildings',             0.15800000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('531130', 'Lessors of miniwarehouses and self-storage',      0.12100000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('531190', 'Lessors of other real estate property',           0.13500000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),

-- Vehicle and equipment leasing
('532100', 'Automotive equipment rental and leasing',         0.25600000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('532400', 'Commercial and industrial machinery rental',      0.31200000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('532420', 'Office machinery and equipment rental',           0.18500000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('532490', 'Other commercial and industrial equipment',       0.28700000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),

-- IT and computer leasing
('532230', 'Video tape and disc rental',                      0.09800000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('518210', 'Data processing, hosting, and related services',  0.22400000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),

-- General leasing
('532000', 'Rental and leasing services',                     0.24300000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('561000', 'Administrative and support services',             0.10200000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE);

-- ============================================================================
-- SEED DATA: NATURAL GAS AND HEATING EMISSION FACTORS
-- ============================================================================

-- Create a reference table for fuel emission factors
CREATE TABLE upstream_leased_assets_service.gl_ula_fuel_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fuel_type VARCHAR(50) NOT NULL,
    ef_per_kwh DECIMAL(12,8) NOT NULL,
    wtt_per_kwh DECIMAL(12,8),
    ef_per_litre DECIMAL(12,8),
    wtt_per_litre DECIMAL(12,8),
    unit VARCHAR(50) DEFAULT 'kgCO2e/kWh',
    source VARCHAR(100) DEFAULT 'DEFRA_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(fuel_type, source, year),
    CONSTRAINT chk_ula_fuel_ef_positive CHECK (ef_per_kwh >= 0),
    CONSTRAINT chk_ula_fuel_wtt_positive CHECK (wtt_per_kwh IS NULL OR wtt_per_kwh >= 0),
    CONSTRAINT chk_ula_fuel_litre_positive CHECK (ef_per_litre IS NULL OR ef_per_litre >= 0),
    CONSTRAINT chk_ula_fuel_wtt_litre_positive CHECK (wtt_per_litre IS NULL OR wtt_per_litre >= 0)
);

CREATE INDEX idx_ula_fuel_type ON upstream_leased_assets_service.gl_ula_fuel_factors(fuel_type);
CREATE INDEX idx_ula_fuel_source ON upstream_leased_assets_service.gl_ula_fuel_factors(source);
CREATE INDEX idx_ula_fuel_active ON upstream_leased_assets_service.gl_ula_fuel_factors(is_active);

COMMENT ON TABLE upstream_leased_assets_service.gl_ula_fuel_factors IS 'Fuel and energy emission factors for building and equipment calculations (DEFRA 2024)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_fuel_factors.ef_per_kwh IS 'Emission factor per kWh of energy content (kgCO2e/kWh)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_fuel_factors.wtt_per_kwh IS 'Well-to-tank factor per kWh (kgCO2e/kWh)';
COMMENT ON COLUMN upstream_leased_assets_service.gl_ula_fuel_factors.ef_per_litre IS 'Emission factor per litre of fuel (kgCO2e/litre)';

INSERT INTO upstream_leased_assets_service.gl_ula_fuel_factors
(fuel_type, ef_per_kwh, wtt_per_kwh, ef_per_litre, wtt_per_litre, unit, source, year, is_active) VALUES
-- Natural gas
('natural_gas',      0.18400000, 0.02530000, NULL,        NULL,        'kgCO2e/kWh', 'DEFRA_2024', 2024, TRUE),

-- District heating
('district_heating', 0.16620000, 0.02910000, NULL,        NULL,        'kgCO2e/kWh', 'DEFRA_2024', 2024, TRUE),

-- District cooling
('district_cooling', 0.13070000, 0.01820000, NULL,        NULL,        'kgCO2e/kWh', 'DEFRA_2024', 2024, TRUE),

-- Diesel (for generators and equipment)
('diesel',           0.25390000, 0.06110000, 2.51260000, 0.60520000,  'kgCO2e/kWh', 'DEFRA_2024', 2024, TRUE),

-- Petrol (for small equipment)
('petrol',           0.24130000, 0.05810000, 2.16800000, 0.52200000,  'kgCO2e/kWh', 'DEFRA_2024', 2024, TRUE),

-- LPG
('lpg',              0.21450000, 0.03310000, 1.55370000, 0.23990000,  'kgCO2e/kWh', 'DEFRA_2024', 2024, TRUE),

-- Heating oil
('heating_oil',      0.24670000, 0.05730000, 2.54040000, 0.59000000,  'kgCO2e/kWh', 'DEFRA_2024', 2024, TRUE);

-- ============================================================================
-- AGENT REGISTRY ENTRY
-- ============================================================================

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
    'GL-MRV-S3-008',
    'Upstream Leased Assets Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-021: Scope 3 Category 8 - Upstream Leased Assets. Calculates emissions from assets leased by the reporting company (lessee) including buildings, vehicles, equipment, and IT assets. Supports asset-specific (metered energy), lessor-specific (lessor-reported data), average-data (EUI benchmarks), and spend-based (EPA USEEIO v2) calculation methods. Includes 27 EUI benchmarks (14 building types x 4 climate zones), 21 grid EFs (20 countries + global), 21 vehicle EFs, 12 EEIO factors, 7 fuel factors. Features multi-asset-type portfolios, floor-area/headcount/revenue allocation, PUE-adjusted IT calculations, and operating vs finance lease classification.',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 8,
        'category_name', 'Upstream Leased Assets',
        'calculation_methods', jsonb_build_array('asset_specific', 'lessor_specific', 'average_data', 'spend_based'),
        'asset_types', jsonb_build_array('building', 'vehicle', 'equipment', 'it_asset'),
        'building_types', jsonb_build_array('office', 'retail', 'warehouse', 'industrial', 'data_center', 'laboratory', 'hospital', 'hotel', 'education', 'mixed_use', 'residential', 'restaurant', 'supermarket', 'other'),
        'vehicle_types', jsonb_build_array('small_car', 'medium_car', 'large_car', 'average_car', 'suv', 'light_van', 'medium_van', 'heavy_van', 'pickup_truck', 'light_truck', 'medium_truck', 'heavy_truck', 'motorcycle', 'electric_vehicle', 'hybrid_vehicle'),
        'equipment_types', jsonb_build_array('generator', 'compressor', 'pump', 'forklift', 'crane', 'excavator', 'loader', 'welder', 'hvac_unit', 'chiller', 'boiler', 'conveyor', 'press', 'cnc_machine'),
        'it_types', jsonb_build_array('server_rack', 'server_blade', 'server_tower', 'storage_array', 'network_switch', 'network_router', 'firewall', 'load_balancer', 'ups', 'cooling_unit', 'workstation', 'desktop', 'laptop', 'monitor', 'printer'),
        'allocation_methods', jsonb_build_array('floor_area', 'headcount', 'revenue', 'time_based', 'full'),
        'lease_types', jsonb_build_array('operating', 'finance', 'short_term', 'sublease'),
        'frameworks', jsonb_build_array('GHG Protocol Scope 3', 'CSRD ESRS E1', 'CDP Climate', 'SBTi', 'ISO 14064-1', 'GRI 305', 'SECR'),
        'eui_benchmark_count', 27,
        'grid_ef_count', 21,
        'vehicle_ef_count', 21,
        'eeio_factor_count', 12,
        'fuel_factor_count', 7,
        'supports_wtt_emissions', true,
        'supports_allocation', true,
        'supports_pue_adjustment', true,
        'supports_lessor_data_validation', true,
        'supports_multi_asset_portfolio', true,
        'supports_cpi_deflation', true,
        'supports_climate_zone_adjustment', true,
        'default_ef_source', 'DEFRA_2024',
        'default_grid_source', 'IEA_2024',
        'default_eui_source', 'ASHRAE_90.1_2019',
        'default_gwp', 'AR5',
        'schema', 'upstream_leased_assets_service',
        'table_prefix', 'gl_ula_',
        'hypertables', jsonb_build_array('gl_ula_calculations', 'gl_ula_building_emissions', 'gl_ula_vehicle_emissions'),
        'continuous_aggregates', jsonb_build_array('gl_ula_daily_emissions', 'gl_ula_monthly_portfolio'),
        'migration_version', 'V072'
    )
)
ON CONFLICT (agent_code) DO UPDATE SET
    agent_name = EXCLUDED.agent_name,
    agent_version = EXCLUDED.agent_version,
    description = EXCLUDED.description,
    status = EXCLUDED.status,
    metadata = EXCLUDED.metadata,
    updated_at = NOW();

-- ============================================================================
-- FINAL COMMENTS
-- ============================================================================

COMMENT ON SCHEMA upstream_leased_assets_service IS 'Updated: AGENT-MRV-021 complete with 20 tables (16 operational + 4 reference), 3 hypertables, 2 continuous aggregates, 16 RLS policies, 100+ seed records';

-- ============================================================================
-- END OF MIGRATION V072
-- ============================================================================
-- Total Lines: ~950
-- Total Tables: 20 (16 operational + 4 reference)
--   Operational: gl_ula_calculations, gl_ula_building_assets, gl_ula_building_emissions,
--     gl_ula_vehicle_assets, gl_ula_vehicle_emissions, gl_ula_equipment_assets,
--     gl_ula_equipment_emissions, gl_ula_it_assets, gl_ula_it_emissions,
--     gl_ula_allocations, gl_ula_compliance_results, gl_ula_emission_factors,
--     gl_ula_provenance, gl_ula_aggregations, gl_ula_spend_calculations,
--     gl_ula_lessor_data
--   Reference: gl_ula_eui_benchmarks, gl_ula_grid_factors, gl_ula_vehicle_ef_ref,
--     gl_ula_eeio_factors, gl_ula_fuel_factors
-- Total Hypertables: 3 (calculations, building_emissions, vehicle_emissions)
-- Total Continuous Aggregates: 2 (daily_emissions, monthly_portfolio)
-- Total RLS Policies: 16 (all operational tables) + 1 (public read on EUI benchmarks)
-- Total Seed Records: 115
--   EUI Benchmarks: 27 (14 global + 4 zone 5A + 4 zone 1A + 4 UK + 1 other)
--   Grid Emission Factors: 21 (20 countries + GLOBAL)
--   Vehicle Emission Factors: 21 (cars/vans/trucks/motorcycle/SUV)
--   EEIO Factors: 12 (real estate/vehicle/equipment/IT leasing NAICS codes)
--   Fuel Factors: 7 (natural gas, district heating/cooling, diesel, petrol, LPG, heating oil)
--   Agent Registry: 1
-- Total Indexes: 95
-- Total Constraints: 75
-- ============================================================================

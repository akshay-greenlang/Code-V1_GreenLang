-- ============================================================================
-- Migration: V077__downstream_leased_assets_service.sql
-- Description: AGENT-MRV-026 Downstream Leased Assets (Scope 3 Category 13)
-- Agent: GL-MRV-S3-013
-- Framework: GHG Protocol Scope 3 Standard, DEFRA 2024, IEA 2024, EPA eGRID,
--            ASHRAE 90.1, CIBSE TM46, EPA USEEIO v2, ISO 14064-1
-- Created: 2026-02-28
-- ============================================================================
-- Schema: downstream_leased_assets_service
-- Tables: 21 (10 reference + 8 operational + 3 supporting), all prefixed gl_dla_
-- Hypertables: 3 (calculations 7-day, compliance_checks 30-day, aggregations 30-day)
-- Continuous Aggregates: 2 (gl_dla_daily_by_asset_type, gl_dla_monthly_by_category)
-- RLS: 10 policies with app.current_tenant_id
-- Seed Data: ~188 records (40 building benchmarks, 56 vehicle EFs, 6 equipment,
--            7 IT asset, 38 grid EFs, 8 fuel EFs, 10 EEIO, 15 refrigerants,
--            8 vacancy factors)
-- ============================================================================
-- Category 13 = Downstream Leased Assets = assets OWNED by reporter and
--               LEASED TO others (reporter is LESSOR).
--               Mirror of Category 8 (Upstream Leased Assets) from lessor perspective.
-- ============================================================================

-- ============================================================================
-- SCHEMA CREATION
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS downstream_leased_assets_service;

COMMENT ON SCHEMA downstream_leased_assets_service IS
    'AGENT-MRV-026: Downstream Leased Assets - Scope 3 Category 13 emission calculations. '
    'Assets OWNED by reporter and LEASED TO others (reporter is LESSOR). '
    'Supports buildings, vehicles, equipment, IT assets with asset-specific, '
    'average-data, spend-based, and hybrid calculation methods.';

-- ============================================================================
-- REFERENCE TABLE 1: gl_dla_building_benchmarks
-- Description: Energy Use Intensity (EUI) benchmarks by building type and
--              climate zone. 8 building types x 5 ASHRAE climate zones = 40 rows.
--              Includes NABERS star rating and EPC grade mappings.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_building_benchmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    building_type VARCHAR(50) NOT NULL,
    climate_zone VARCHAR(50) NOT NULL,
    eui_kwh_per_sqm DECIMAL(10,4) NOT NULL,
    electricity_share DECIMAL(5,4) DEFAULT 0.60,
    gas_share DECIMAL(5,4) DEFAULT 0.30,
    steam_share DECIMAL(5,4) DEFAULT 0.05,
    cooling_share DECIMAL(5,4) DEFAULT 0.05,
    nabers_rating VARCHAR(10),
    epc_grade VARCHAR(5),
    source VARCHAR(100) NOT NULL DEFAULT 'ASHRAE_90.1_2019',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(building_type, climate_zone),
    CONSTRAINT chk_dla_bench_type CHECK (building_type IN (
        'office', 'retail', 'warehouse', 'industrial',
        'data_center', 'hospital', 'hotel', 'mixed_use'
    )),
    CONSTRAINT chk_dla_bench_zone CHECK (climate_zone IN (
        '1A_very_hot_humid', '2B_hot_dry', '3C_warm_marine',
        '4A_mixed_humid', '5A_cool_humid'
    )),
    CONSTRAINT chk_dla_bench_eui_positive CHECK (eui_kwh_per_sqm > 0),
    CONSTRAINT chk_dla_bench_shares_valid CHECK (
        electricity_share >= 0 AND electricity_share <= 1.0 AND
        gas_share >= 0 AND gas_share <= 1.0 AND
        steam_share >= 0 AND steam_share <= 1.0 AND
        cooling_share >= 0 AND cooling_share <= 1.0
    ),
    CONSTRAINT chk_dla_bench_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dla_bench_type ON downstream_leased_assets_service.gl_dla_building_benchmarks(building_type);
CREATE INDEX idx_dla_bench_zone ON downstream_leased_assets_service.gl_dla_building_benchmarks(climate_zone);
CREATE INDEX idx_dla_bench_active ON downstream_leased_assets_service.gl_dla_building_benchmarks(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_building_benchmarks IS 'EUI benchmarks by building type and ASHRAE climate zone (kWh/sqm/yr)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_building_benchmarks.eui_kwh_per_sqm IS 'Total Energy Use Intensity in kWh per sqm per year';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_building_benchmarks.electricity_share IS 'Fraction of EUI from electricity (0.0-1.0)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_building_benchmarks.nabers_rating IS 'Equivalent NABERS star rating (AU)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_building_benchmarks.epc_grade IS 'Equivalent EPC energy grade (A-G, EU/UK)';

-- ============================================================================
-- REFERENCE TABLE 2: gl_dla_vehicle_emission_factors
-- Description: Vehicle emission factors by type and fuel. 8 vehicle types x
--              7 fuel types = 56 rows. DEFRA 2024 per-km factors.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_vehicle_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_type VARCHAR(50) NOT NULL,
    fuel_type VARCHAR(20) NOT NULL,
    ef_per_km DECIMAL(10,8) NOT NULL,
    wtt_ef_per_km DECIMAL(10,8) NOT NULL DEFAULT 0,
    unit VARCHAR(30) DEFAULT 'kgCO2e/km',
    source VARCHAR(100) NOT NULL DEFAULT 'DEFRA_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(vehicle_type, fuel_type, source),
    CONSTRAINT chk_dla_vef_type CHECK (vehicle_type IN (
        'small_car', 'medium_car', 'large_car', 'suv',
        'light_van', 'heavy_van', 'light_truck', 'heavy_truck'
    )),
    CONSTRAINT chk_dla_vef_fuel CHECK (fuel_type IN (
        'petrol', 'diesel', 'hybrid', 'phev', 'ev', 'lpg', 'cng'
    )),
    CONSTRAINT chk_dla_vef_positive CHECK (ef_per_km >= 0 AND wtt_ef_per_km >= 0),
    CONSTRAINT chk_dla_vef_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dla_vef_type ON downstream_leased_assets_service.gl_dla_vehicle_emission_factors(vehicle_type);
CREATE INDEX idx_dla_vef_fuel ON downstream_leased_assets_service.gl_dla_vehicle_emission_factors(fuel_type);
CREATE INDEX idx_dla_vef_active ON downstream_leased_assets_service.gl_dla_vehicle_emission_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_vehicle_emission_factors IS 'Vehicle emission factors by type and fuel type (DEFRA 2024, kgCO2e/km)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_vehicle_emission_factors.ef_per_km IS 'Tank-to-wheel emission factor per km (kgCO2e/km)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_vehicle_emission_factors.wtt_ef_per_km IS 'Well-to-tank emission factor per km (kgCO2e/km)';

-- ============================================================================
-- REFERENCE TABLE 3: gl_dla_equipment_factors
-- Description: Equipment emission factors with rated power, fuel consumption
--              rate, and default load factor. 6 equipment types.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_equipment_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_type VARCHAR(50) NOT NULL,
    rated_power_kw DECIMAL(10,4),
    fuel_consumption_l_per_hr DECIMAL(10,6),
    default_load_factor DECIMAL(5,4) DEFAULT 0.50,
    default_fuel_type VARCHAR(20) DEFAULT 'diesel',
    ef_per_litre DECIMAL(10,6),
    wtt_ef_per_litre DECIMAL(10,6) DEFAULT 0,
    unit VARCHAR(30) DEFAULT 'kgCO2e/litre',
    source VARCHAR(100) NOT NULL DEFAULT 'DEFRA_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(equipment_type, source),
    CONSTRAINT chk_dla_equip_type CHECK (equipment_type IN (
        'generator', 'compressor', 'pump', 'forklift', 'crane', 'hvac_unit'
    )),
    CONSTRAINT chk_dla_equip_power_positive CHECK (rated_power_kw IS NULL OR rated_power_kw >= 0),
    CONSTRAINT chk_dla_equip_fuel_positive CHECK (fuel_consumption_l_per_hr IS NULL OR fuel_consumption_l_per_hr >= 0),
    CONSTRAINT chk_dla_equip_load_range CHECK (default_load_factor >= 0 AND default_load_factor <= 1.0),
    CONSTRAINT chk_dla_equip_ef_positive CHECK (ef_per_litre IS NULL OR ef_per_litre >= 0),
    CONSTRAINT chk_dla_equip_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dla_equip_type ON downstream_leased_assets_service.gl_dla_equipment_factors(equipment_type);
CREATE INDEX idx_dla_equip_active ON downstream_leased_assets_service.gl_dla_equipment_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_equipment_factors IS 'Equipment emission factors with rated power, fuel rate, and load factor defaults';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_equipment_factors.fuel_consumption_l_per_hr IS 'Fuel consumption rate in litres per operating hour at full load';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_equipment_factors.default_load_factor IS 'Default load factor (0.0-1.0) when actual load is unknown';

-- ============================================================================
-- REFERENCE TABLE 4: gl_dla_it_asset_factors
-- Description: IT asset power and emission defaults. 7 IT asset types with
--              default power_kw, PUE, and hours_per_year.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_it_asset_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    it_asset_type VARCHAR(50) NOT NULL,
    default_power_kw DECIMAL(10,6) NOT NULL,
    default_pue DECIMAL(5,4) DEFAULT 1.58,
    default_hours_per_year DECIMAL(8,2) DEFAULT 8760.00,
    source VARCHAR(100) NOT NULL DEFAULT 'Industry_Average_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(it_asset_type, source),
    CONSTRAINT chk_dla_it_type CHECK (it_asset_type IN (
        'server', 'storage', 'network_switch', 'router',
        'ups', 'cooling_unit', 'workstation'
    )),
    CONSTRAINT chk_dla_it_power_positive CHECK (default_power_kw > 0),
    CONSTRAINT chk_dla_it_pue_range CHECK (default_pue >= 1.0 AND default_pue <= 5.0),
    CONSTRAINT chk_dla_it_hours_range CHECK (default_hours_per_year >= 0 AND default_hours_per_year <= 8784),
    CONSTRAINT chk_dla_it_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dla_it_type ON downstream_leased_assets_service.gl_dla_it_asset_factors(it_asset_type);
CREATE INDEX idx_dla_it_active ON downstream_leased_assets_service.gl_dla_it_asset_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_it_asset_factors IS 'IT asset power defaults and PUE values for 7 IT asset types';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_it_asset_factors.default_power_kw IS 'Default power draw per unit in kW';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_it_asset_factors.default_pue IS 'Default Power Usage Effectiveness (1.0=ideal, industry avg ~1.58)';

-- ============================================================================
-- REFERENCE TABLE 5: gl_dla_grid_emission_factors
-- Description: Electricity grid emission factors by country and eGRID subregion.
--              12 countries + 26 eGRID subregions = 38 rows.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_grid_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_code VARCHAR(20) NOT NULL UNIQUE,
    region_name VARCHAR(200) NOT NULL,
    region_type VARCHAR(20) NOT NULL DEFAULT 'country',
    ef_kgco2e_per_kwh DECIMAL(10,8) NOT NULL,
    wtt_ef_kgco2e_per_kwh DECIMAL(10,8) DEFAULT 0,
    td_loss_factor DECIMAL(5,4) DEFAULT 0.05,
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_dla_grid_type CHECK (region_type IN ('country', 'egrid_subregion', 'state')),
    CONSTRAINT chk_dla_grid_ef_positive CHECK (ef_kgco2e_per_kwh >= 0),
    CONSTRAINT chk_dla_grid_wtt_positive CHECK (wtt_ef_kgco2e_per_kwh IS NULL OR wtt_ef_kgco2e_per_kwh >= 0),
    CONSTRAINT chk_dla_grid_td_range CHECK (td_loss_factor >= 0 AND td_loss_factor <= 0.50),
    CONSTRAINT chk_dla_grid_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dla_grid_code ON downstream_leased_assets_service.gl_dla_grid_emission_factors(region_code);
CREATE INDEX idx_dla_grid_type ON downstream_leased_assets_service.gl_dla_grid_emission_factors(region_type);
CREATE INDEX idx_dla_grid_active ON downstream_leased_assets_service.gl_dla_grid_emission_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_grid_emission_factors IS 'Electricity grid emission factors by country (IEA 2024) and eGRID subregion (EPA 2024)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_grid_emission_factors.ef_kgco2e_per_kwh IS 'Grid emission factor in kgCO2e per kWh consumed';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_grid_emission_factors.td_loss_factor IS 'Transmission and distribution loss factor (fraction)';

-- ============================================================================
-- REFERENCE TABLE 6: gl_dla_fuel_emission_factors
-- Description: Fuel combustion emission factors for 8 fuel types with
--              direct CO2e and WTT upstream factors.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_fuel_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fuel_type VARCHAR(30) NOT NULL UNIQUE,
    ef_kgco2e_per_kwh DECIMAL(10,8) NOT NULL,
    wtt_ef_kgco2e_per_kwh DECIMAL(10,8) DEFAULT 0,
    ef_kgco2e_per_litre DECIMAL(10,6),
    density_kg_per_litre DECIMAL(8,6),
    calorific_value_kwh_per_litre DECIMAL(8,4),
    source VARCHAR(100) NOT NULL DEFAULT 'DEFRA_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_dla_fuel_ef_positive CHECK (ef_kgco2e_per_kwh >= 0),
    CONSTRAINT chk_dla_fuel_wtt_positive CHECK (wtt_ef_kgco2e_per_kwh IS NULL OR wtt_ef_kgco2e_per_kwh >= 0),
    CONSTRAINT chk_dla_fuel_litre_positive CHECK (ef_kgco2e_per_litre IS NULL OR ef_kgco2e_per_litre >= 0),
    CONSTRAINT chk_dla_fuel_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dla_fuel_type ON downstream_leased_assets_service.gl_dla_fuel_emission_factors(fuel_type);
CREATE INDEX idx_dla_fuel_active ON downstream_leased_assets_service.gl_dla_fuel_emission_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_fuel_emission_factors IS 'Fuel combustion emission factors (DEFRA 2024) with CO2e per kWh and per litre';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_fuel_emission_factors.ef_kgco2e_per_kwh IS 'Emission factor per kWh of fuel energy (kgCO2e/kWh)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_fuel_emission_factors.ef_kgco2e_per_litre IS 'Emission factor per litre of fuel consumed (kgCO2e/litre)';

-- ============================================================================
-- REFERENCE TABLE 7: gl_dla_eeio_spend_factors
-- Description: EEIO spend-based emission factors for 10 NAICS leasing/rental
--              industry codes. EPA USEEIO v2.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_eeio_spend_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    naics_code VARCHAR(20) NOT NULL,
    category_name VARCHAR(200) NOT NULL,
    ef_per_usd DECIMAL(12,8) NOT NULL,
    base_year INT DEFAULT 2021,
    unit VARCHAR(30) DEFAULT 'kgCO2e/USD',
    source VARCHAR(100) DEFAULT 'EPA_USEEIO_v2',
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(naics_code, source),
    CONSTRAINT chk_dla_eeio_ef_positive CHECK (ef_per_usd >= 0),
    CONSTRAINT chk_dla_eeio_year_valid CHECK (base_year >= 1990 AND base_year <= 2100)
);

CREATE INDEX idx_dla_eeio_naics ON downstream_leased_assets_service.gl_dla_eeio_spend_factors(naics_code);
CREATE INDEX idx_dla_eeio_active ON downstream_leased_assets_service.gl_dla_eeio_spend_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_eeio_spend_factors IS 'EPA USEEIO v2 spend-based emission factors for leasing/rental NAICS codes';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_eeio_spend_factors.ef_per_usd IS 'Emission factor per USD of revenue (kgCO2e/USD, base year deflated)';

-- ============================================================================
-- REFERENCE TABLE 8: gl_dla_refrigerant_gwps
-- Description: Refrigerant Global Warming Potentials (IPCC AR6).
--              15 common refrigerants for building HVAC leakage estimates.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_refrigerant_gwps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    refrigerant VARCHAR(30) NOT NULL UNIQUE,
    chemical_formula VARCHAR(50),
    gwp_100yr INT NOT NULL,
    gwp_source VARCHAR(20) DEFAULT 'IPCC_AR6',
    ozone_depleting BOOLEAN DEFAULT FALSE,
    phase_down_schedule VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_dla_refrig_gwp_positive CHECK (gwp_100yr >= 0)
);

CREATE INDEX idx_dla_refrig_name ON downstream_leased_assets_service.gl_dla_refrigerant_gwps(refrigerant);
CREATE INDEX idx_dla_refrig_active ON downstream_leased_assets_service.gl_dla_refrigerant_gwps(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_refrigerant_gwps IS 'Refrigerant GWP values (IPCC AR6) for building HVAC leakage estimates';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_refrigerant_gwps.gwp_100yr IS '100-year Global Warming Potential relative to CO2';

-- ============================================================================
-- REFERENCE TABLE 9: gl_dla_vacancy_factors
-- Description: Base-load factors by building type for vacancy adjustment.
--              Represents fraction of full-load energy consumed when vacant.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_vacancy_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    building_type VARCHAR(50) NOT NULL UNIQUE,
    base_load_fraction DECIMAL(5,4) NOT NULL,
    description VARCHAR(200),
    source VARCHAR(100) DEFAULT 'ASHRAE_Guideline_14',
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_dla_vacancy_type CHECK (building_type IN (
        'office', 'retail', 'warehouse', 'industrial',
        'data_center', 'hospital', 'hotel', 'mixed_use'
    )),
    CONSTRAINT chk_dla_vacancy_fraction_range CHECK (base_load_fraction >= 0 AND base_load_fraction <= 1.0)
);

CREATE INDEX idx_dla_vacancy_type ON downstream_leased_assets_service.gl_dla_vacancy_factors(building_type);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_vacancy_factors IS 'Vacancy base-load fractions by building type (fraction of full-load energy when vacant)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_vacancy_factors.base_load_fraction IS 'Fraction of normal energy consumed when space is vacant (0.0-1.0)';

-- ============================================================================
-- REFERENCE TABLE 10: gl_dla_allocation_defaults
-- Description: Default allocation percentages and parameters by method.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_allocation_defaults (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    allocation_method VARCHAR(30) NOT NULL UNIQUE,
    description VARCHAR(300) NOT NULL,
    default_common_area_pct DECIMAL(5,4) DEFAULT 0.15,
    default_lessor_share DECIMAL(5,4) DEFAULT 0.00,
    requires_tenant_data BOOLEAN DEFAULT FALSE,
    data_quality_tier INT DEFAULT 2,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_dla_alloc_method CHECK (allocation_method IN (
        'floor_area', 'headcount', 'revenue', 'time_based', 'equal_split', 'full'
    )),
    CONSTRAINT chk_dla_alloc_common_range CHECK (default_common_area_pct >= 0 AND default_common_area_pct <= 1.0),
    CONSTRAINT chk_dla_alloc_lessor_range CHECK (default_lessor_share >= 0 AND default_lessor_share <= 1.0),
    CONSTRAINT chk_dla_alloc_dq_range CHECK (data_quality_tier >= 1 AND data_quality_tier <= 5)
);

CREATE INDEX idx_dla_alloc_method ON downstream_leased_assets_service.gl_dla_allocation_defaults(allocation_method);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_allocation_defaults IS 'Default allocation parameters by method for lessor/tenant emissions split';

-- ============================================================================
-- OPERATIONAL TABLE 11: gl_dla_calculations (HYPERTABLE)
-- Description: Master calculation records. 7-day chunk hypertable.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_calculations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    org_id VARCHAR(200) NOT NULL,
    reporting_year INT NOT NULL,
    method VARCHAR(50) NOT NULL,
    consolidation_approach VARCHAR(30) DEFAULT 'operational_control',
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    building_co2e_kg DECIMAL(20,8) DEFAULT 0,
    vehicle_co2e_kg DECIMAL(20,8) DEFAULT 0,
    equipment_co2e_kg DECIMAL(20,8) DEFAULT 0,
    it_asset_co2e_kg DECIMAL(20,8) DEFAULT 0,
    asset_count INT DEFAULT 0,
    dqi_score DECIMAL(5,2),
    provenance_hash VARCHAR(64),
    status VARCHAR(20) DEFAULT 'completed',
    is_deleted BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, calculated_at),
    CONSTRAINT chk_dla_calc_method CHECK (method IN (
        'asset_specific', 'average_data', 'spend_based', 'hybrid'
    )),
    CONSTRAINT chk_dla_calc_consol CHECK (consolidation_approach IN (
        'operational_control', 'financial_control', 'equity_share'
    )),
    CONSTRAINT chk_dla_calc_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_dla_calc_bldg_positive CHECK (building_co2e_kg IS NULL OR building_co2e_kg >= 0),
    CONSTRAINT chk_dla_calc_veh_positive CHECK (vehicle_co2e_kg IS NULL OR vehicle_co2e_kg >= 0),
    CONSTRAINT chk_dla_calc_equip_positive CHECK (equipment_co2e_kg IS NULL OR equipment_co2e_kg >= 0),
    CONSTRAINT chk_dla_calc_it_positive CHECK (it_asset_co2e_kg IS NULL OR it_asset_co2e_kg >= 0),
    CONSTRAINT chk_dla_calc_year_valid CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_dla_calc_dqi_range CHECK (dqi_score IS NULL OR (dqi_score >= 1.0 AND dqi_score <= 5.0)),
    CONSTRAINT chk_dla_calc_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_dla_calc_asset_count CHECK (asset_count >= 0)
);

-- Convert to hypertable (7-day chunks)
SELECT create_hypertable('downstream_leased_assets_service.gl_dla_calculations', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_dla_calc_tenant ON downstream_leased_assets_service.gl_dla_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_dla_calc_org ON downstream_leased_assets_service.gl_dla_calculations(org_id);
CREATE INDEX idx_dla_calc_year ON downstream_leased_assets_service.gl_dla_calculations(reporting_year);
CREATE INDEX idx_dla_calc_method ON downstream_leased_assets_service.gl_dla_calculations(method);
CREATE INDEX idx_dla_calc_status ON downstream_leased_assets_service.gl_dla_calculations(status);
CREATE INDEX idx_dla_calc_hash ON downstream_leased_assets_service.gl_dla_calculations(provenance_hash);
CREATE INDEX idx_dla_calc_deleted ON downstream_leased_assets_service.gl_dla_calculations(is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX idx_dla_calc_metadata ON downstream_leased_assets_service.gl_dla_calculations USING GIN(metadata);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_calculations IS 'Master calculation records for downstream leased asset emissions (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_calculations.method IS 'Calculation method: asset_specific, average_data, spend_based, hybrid';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_calculations.consolidation_approach IS 'GHG Protocol consolidation: operational_control, financial_control, equity_share';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_calculations.dqi_score IS 'Data Quality Indicator (1.0=highest to 5.0=lowest)';

-- ============================================================================
-- OPERATIONAL TABLE 12: gl_dla_asset_results
-- Description: Per-asset emission calculation results with category, type,
--              energy data, and CO2e values.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_asset_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    asset_category VARCHAR(30) NOT NULL,
    asset_type VARCHAR(50) NOT NULL,
    method VARCHAR(50) NOT NULL,
    floor_area_sqm DECIMAL(15,4),
    electricity_kwh DECIMAL(15,4),
    gas_kwh DECIMAL(15,4),
    steam_kwh DECIMAL(15,4),
    cooling_kwh DECIMAL(15,4),
    fuel_litres DECIMAL(15,4),
    distance_km DECIMAL(15,4),
    operating_hours DECIMAL(10,2),
    load_factor DECIMAL(5,4),
    power_kw DECIMAL(10,4),
    pue DECIMAL(5,4),
    quantity INT DEFAULT 1,
    allocation_method VARCHAR(30),
    allocation_factor DECIMAL(5,4) DEFAULT 1.0,
    pre_allocation_co2e_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    wtt_co2e_kg DECIMAL(20,8) DEFAULT 0,
    ef_source VARCHAR(100),
    grid_factor_used DECIMAL(10,8),
    dqi_score DECIMAL(5,2),
    provenance_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dla_ar_category CHECK (asset_category IN ('building', 'vehicle', 'equipment', 'it_asset')),
    CONSTRAINT chk_dla_ar_method CHECK (method IN ('asset_specific', 'average_data', 'spend_based', 'hybrid')),
    CONSTRAINT chk_dla_ar_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_dla_ar_wtt_positive CHECK (wtt_co2e_kg IS NULL OR wtt_co2e_kg >= 0),
    CONSTRAINT chk_dla_ar_alloc_range CHECK (allocation_factor >= 0 AND allocation_factor <= 1.0),
    CONSTRAINT chk_dla_ar_dqi_range CHECK (dqi_score IS NULL OR (dqi_score >= 1.0 AND dqi_score <= 5.0)),
    CONSTRAINT chk_dla_ar_quantity_positive CHECK (quantity >= 1)
);

CREATE INDEX idx_dla_ar_calc ON downstream_leased_assets_service.gl_dla_asset_results(calc_id);
CREATE INDEX idx_dla_ar_tenant ON downstream_leased_assets_service.gl_dla_asset_results(tenant_id, created_at DESC);
CREATE INDEX idx_dla_ar_category ON downstream_leased_assets_service.gl_dla_asset_results(asset_category);
CREATE INDEX idx_dla_ar_type ON downstream_leased_assets_service.gl_dla_asset_results(asset_type);
CREATE INDEX idx_dla_ar_method ON downstream_leased_assets_service.gl_dla_asset_results(method);
CREATE INDEX idx_dla_ar_hash ON downstream_leased_assets_service.gl_dla_asset_results(provenance_hash);
CREATE INDEX idx_dla_ar_metadata ON downstream_leased_assets_service.gl_dla_asset_results USING GIN(metadata);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_asset_results IS 'Per-asset emission results with category, type, energy consumption, and allocation';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_asset_results.pre_allocation_co2e_kg IS 'CO2e before allocation factor applied';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_asset_results.allocation_factor IS 'Fraction of emissions allocated to lessor (0.0-1.0)';

-- ============================================================================
-- OPERATIONAL TABLE 13: gl_dla_allocation_records
-- Description: Allocation audit trail showing method, tenant shares, and
--              common area handling.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_allocation_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    asset_result_id UUID,
    allocation_method VARCHAR(30) NOT NULL,
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    allocated_co2e_kg DECIMAL(20,8) NOT NULL,
    common_area_co2e_kg DECIMAL(20,8) DEFAULT 0,
    vacancy_co2e_kg DECIMAL(20,8) DEFAULT 0,
    common_area_pct DECIMAL(5,4) DEFAULT 0,
    vacancy_pct DECIMAL(5,4) DEFAULT 0,
    tenant_shares JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dla_alloc_method_valid CHECK (allocation_method IN (
        'floor_area', 'headcount', 'revenue', 'time_based', 'equal_split', 'full'
    )),
    CONSTRAINT chk_dla_alloc_co2e_positive CHECK (total_co2e_kg >= 0 AND allocated_co2e_kg >= 0),
    CONSTRAINT chk_dla_alloc_common_positive CHECK (common_area_co2e_kg >= 0),
    CONSTRAINT chk_dla_alloc_vacancy_positive CHECK (vacancy_co2e_kg >= 0),
    CONSTRAINT chk_dla_alloc_pct_range CHECK (
        common_area_pct >= 0 AND common_area_pct <= 1.0 AND
        vacancy_pct >= 0 AND vacancy_pct <= 1.0
    )
);

CREATE INDEX idx_dla_alloc_calc ON downstream_leased_assets_service.gl_dla_allocation_records(calc_id);
CREATE INDEX idx_dla_alloc_tenant ON downstream_leased_assets_service.gl_dla_allocation_records(tenant_id);
CREATE INDEX idx_dla_alloc_asset ON downstream_leased_assets_service.gl_dla_allocation_records(asset_result_id);
CREATE INDEX idx_dla_alloc_rec_method ON downstream_leased_assets_service.gl_dla_allocation_records(allocation_method);
CREATE INDEX idx_dla_alloc_shares ON downstream_leased_assets_service.gl_dla_allocation_records USING GIN(tenant_shares);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_allocation_records IS 'Allocation audit trail showing method, tenant shares, common area, and vacancy handling';

-- ============================================================================
-- OPERATIONAL TABLE 14: gl_dla_compliance_checks (HYPERTABLE)
-- Description: Compliance check results against 7 regulatory frameworks.
--              30-day chunk hypertable.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_compliance_checks (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calc_id UUID NOT NULL,
    framework VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    score DECIMAL(5,2),
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, checked_at),
    CONSTRAINT chk_dla_comp_framework CHECK (framework IN (
        'ghg_protocol', 'iso_14064', 'csrd_esrs', 'cdp', 'sbti', 'sb_253', 'gri'
    )),
    CONSTRAINT chk_dla_comp_status CHECK (status IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE')),
    CONSTRAINT chk_dla_comp_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 100))
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('downstream_leased_assets_service.gl_dla_compliance_checks', 'checked_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_dla_comp_tenant ON downstream_leased_assets_service.gl_dla_compliance_checks(tenant_id, checked_at DESC);
CREATE INDEX idx_dla_comp_calc ON downstream_leased_assets_service.gl_dla_compliance_checks(calc_id);
CREATE INDEX idx_dla_comp_framework ON downstream_leased_assets_service.gl_dla_compliance_checks(framework);
CREATE INDEX idx_dla_comp_status ON downstream_leased_assets_service.gl_dla_compliance_checks(status);
CREATE INDEX idx_dla_comp_findings ON downstream_leased_assets_service.gl_dla_compliance_checks USING GIN(findings);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_compliance_checks IS 'Compliance check results against 7 regulatory frameworks (HYPERTABLE, 30-day chunks)';

-- ============================================================================
-- OPERATIONAL TABLE 15: gl_dla_aggregations (HYPERTABLE)
-- Description: Period aggregations by asset category, method, and region.
--              30-day chunk hypertable.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_aggregations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    org_id VARCHAR(200),
    period VARCHAR(20) NOT NULL,
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    by_category JSONB DEFAULT '{}',
    by_method JSONB DEFAULT '{}',
    by_region JSONB DEFAULT '{}',
    asset_count INT DEFAULT 0,
    dqi_avg DECIMAL(5,2),
    metadata JSONB DEFAULT '{}',
    aggregated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, aggregated_at),
    CONSTRAINT chk_dla_agg_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_dla_agg_count_positive CHECK (asset_count >= 0),
    CONSTRAINT chk_dla_agg_dqi_range CHECK (dqi_avg IS NULL OR (dqi_avg >= 1.0 AND dqi_avg <= 5.0))
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('downstream_leased_assets_service.gl_dla_aggregations', 'aggregated_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_dla_agg_tenant ON downstream_leased_assets_service.gl_dla_aggregations(tenant_id, aggregated_at DESC);
CREATE INDEX idx_dla_agg_org ON downstream_leased_assets_service.gl_dla_aggregations(org_id);
CREATE INDEX idx_dla_agg_period ON downstream_leased_assets_service.gl_dla_aggregations(period);
CREATE INDEX idx_dla_agg_by_category ON downstream_leased_assets_service.gl_dla_aggregations USING GIN(by_category);
CREATE INDEX idx_dla_agg_by_method ON downstream_leased_assets_service.gl_dla_aggregations USING GIN(by_method);
CREATE INDEX idx_dla_agg_by_region ON downstream_leased_assets_service.gl_dla_aggregations USING GIN(by_region);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_aggregations IS 'Period aggregations of downstream leased asset emissions (HYPERTABLE, 30-day chunks)';

-- ============================================================================
-- OPERATIONAL TABLE 16: gl_dla_provenance_records
-- Description: SHA-256 hash chain for complete audit trail.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_provenance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calc_id UUID NOT NULL,
    stage VARCHAR(50) NOT NULL,
    stage_index INT NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64) NOT NULL,
    chain_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dla_prov_stage CHECK (stage IN (
        'VALIDATE', 'CLASSIFY', 'NORMALIZE', 'RESOLVE_EFS',
        'CALCULATE', 'ALLOCATE', 'AGGREGATE', 'COMPLIANCE',
        'PROVENANCE', 'SEAL'
    )),
    CONSTRAINT chk_dla_prov_index_positive CHECK (stage_index >= 0)
);

CREATE INDEX idx_dla_prov_tenant ON downstream_leased_assets_service.gl_dla_provenance_records(tenant_id);
CREATE INDEX idx_dla_prov_calc ON downstream_leased_assets_service.gl_dla_provenance_records(calc_id);
CREATE INDEX idx_dla_prov_calc_stage ON downstream_leased_assets_service.gl_dla_provenance_records(calc_id, stage_index);
CREATE INDEX idx_dla_prov_chain ON downstream_leased_assets_service.gl_dla_provenance_records(chain_hash);
CREATE INDEX idx_dla_prov_recorded ON downstream_leased_assets_service.gl_dla_provenance_records(recorded_at DESC);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_provenance_records IS 'SHA-256 provenance hash chain for downstream leased assets calculations';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_provenance_records.chain_hash IS 'SHA-256 hash chaining input_hash + output_hash + previous chain_hash';

-- ============================================================================
-- OPERATIONAL TABLE 17: gl_dla_audit_trail
-- Description: Operation audit log for all API actions.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    operation VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(200),
    user_id VARCHAR(200),
    old_value JSONB,
    new_value JSONB,
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dla_audit_operation CHECK (operation IN (
        'CREATE', 'READ', 'UPDATE', 'DELETE', 'CALCULATE',
        'BATCH_CALCULATE', 'COMPLIANCE_CHECK', 'PORTFOLIO_ANALYSIS'
    ))
);

CREATE INDEX idx_dla_audit_tenant ON downstream_leased_assets_service.gl_dla_audit_trail(tenant_id, created_at DESC);
CREATE INDEX idx_dla_audit_operation ON downstream_leased_assets_service.gl_dla_audit_trail(operation);
CREATE INDEX idx_dla_audit_entity ON downstream_leased_assets_service.gl_dla_audit_trail(entity_type, entity_id);
CREATE INDEX idx_dla_audit_user ON downstream_leased_assets_service.gl_dla_audit_trail(user_id);
CREATE INDEX idx_dla_audit_created ON downstream_leased_assets_service.gl_dla_audit_trail(created_at DESC);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_audit_trail IS 'Operation audit log for all downstream leased assets API actions';

-- ============================================================================
-- OPERATIONAL TABLE 18: gl_dla_batch_jobs
-- Description: Batch processing status tracking.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    batch_id VARCHAR(200) NOT NULL UNIQUE,
    org_id VARCHAR(200),
    reporting_year INT,
    total_assets INT NOT NULL,
    processed_assets INT DEFAULT 0,
    successful_assets INT DEFAULT 0,
    failed_assets INT DEFAULT 0,
    total_co2e_kg DECIMAL(20,8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    error_summary JSONB DEFAULT '[]',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dla_batch_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_dla_batch_total_positive CHECK (total_assets >= 1),
    CONSTRAINT chk_dla_batch_processed_range CHECK (processed_assets >= 0 AND processed_assets <= total_assets),
    CONSTRAINT chk_dla_batch_co2e_positive CHECK (total_co2e_kg >= 0)
);

CREATE INDEX idx_dla_batch_tenant ON downstream_leased_assets_service.gl_dla_batch_jobs(tenant_id, created_at DESC);
CREATE INDEX idx_dla_batch_id ON downstream_leased_assets_service.gl_dla_batch_jobs(batch_id);
CREATE INDEX idx_dla_batch_status ON downstream_leased_assets_service.gl_dla_batch_jobs(status);
CREATE INDEX idx_dla_batch_org ON downstream_leased_assets_service.gl_dla_batch_jobs(org_id);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_batch_jobs IS 'Batch processing status tracking for bulk downstream leased asset calculations';

-- ============================================================================
-- SUPPORTING TABLE 19: gl_dla_data_quality_scores
-- Description: 5-dimension data quality indicator scores.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_data_quality_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    asset_result_id UUID,
    technological_representativeness DECIMAL(3,1) NOT NULL,
    temporal_representativeness DECIMAL(3,1) NOT NULL,
    geographical_representativeness DECIMAL(3,1) NOT NULL,
    completeness DECIMAL(3,1) NOT NULL,
    reliability DECIMAL(3,1) NOT NULL,
    overall_dqi DECIMAL(3,1) NOT NULL,
    method_used VARCHAR(50),
    notes VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dla_dqi_tech_range CHECK (technological_representativeness >= 1.0 AND technological_representativeness <= 5.0),
    CONSTRAINT chk_dla_dqi_temp_range CHECK (temporal_representativeness >= 1.0 AND temporal_representativeness <= 5.0),
    CONSTRAINT chk_dla_dqi_geo_range CHECK (geographical_representativeness >= 1.0 AND geographical_representativeness <= 5.0),
    CONSTRAINT chk_dla_dqi_comp_range CHECK (completeness >= 1.0 AND completeness <= 5.0),
    CONSTRAINT chk_dla_dqi_rel_range CHECK (reliability >= 1.0 AND reliability <= 5.0),
    CONSTRAINT chk_dla_dqi_overall_range CHECK (overall_dqi >= 1.0 AND overall_dqi <= 5.0)
);

CREATE INDEX idx_dla_dqi_calc ON downstream_leased_assets_service.gl_dla_data_quality_scores(calc_id);
CREATE INDEX idx_dla_dqi_tenant ON downstream_leased_assets_service.gl_dla_data_quality_scores(tenant_id);
CREATE INDEX idx_dla_dqi_asset ON downstream_leased_assets_service.gl_dla_data_quality_scores(asset_result_id);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_data_quality_scores IS '5-dimension DQI scores (GHG Protocol pedigree matrix: tech, temporal, geo, completeness, reliability)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_data_quality_scores.overall_dqi IS 'Average of 5 dimensions (1.0=highest quality, 5.0=lowest)';

-- ============================================================================
-- SUPPORTING TABLE 20: gl_dla_uncertainty_results
-- Description: Monte Carlo and analytical uncertainty quantification.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_uncertainty_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    method VARCHAR(30) NOT NULL,
    iterations INT,
    confidence_level DECIMAL(5,4),
    mean_co2e_kg DECIMAL(20,8),
    std_dev_kg DECIMAL(20,8),
    ci_lower_kg DECIMAL(20,8),
    ci_upper_kg DECIMAL(20,8),
    cv_pct DECIMAL(8,4),
    metadata JSONB DEFAULT '{}',
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dla_unc_method CHECK (method IN ('monte_carlo', 'analytical', 'ipcc_tier_2')),
    CONSTRAINT chk_dla_unc_iterations CHECK (iterations IS NULL OR iterations > 0),
    CONSTRAINT chk_dla_unc_confidence CHECK (confidence_level IS NULL OR (confidence_level > 0 AND confidence_level <= 1)),
    CONSTRAINT chk_dla_unc_std_positive CHECK (std_dev_kg IS NULL OR std_dev_kg >= 0),
    CONSTRAINT chk_dla_unc_bounds CHECK (ci_lower_kg IS NULL OR ci_upper_kg IS NULL OR ci_lower_kg <= ci_upper_kg)
);

CREATE INDEX idx_dla_unc_calc ON downstream_leased_assets_service.gl_dla_uncertainty_results(calc_id);
CREATE INDEX idx_dla_unc_tenant ON downstream_leased_assets_service.gl_dla_uncertainty_results(tenant_id);
CREATE INDEX idx_dla_unc_method ON downstream_leased_assets_service.gl_dla_uncertainty_results(method);

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_uncertainty_results IS 'Monte Carlo and analytical uncertainty results for emission calculations';

-- ============================================================================
-- SUPPORTING TABLE 21: gl_dla_tenant_energy_data
-- Description: Collected tenant energy consumption data for asset-specific
--              calculations on leased-out buildings.
-- ============================================================================

CREATE TABLE downstream_leased_assets_service.gl_dla_tenant_energy_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    asset_result_id UUID,
    lessee_name VARCHAR(200),
    lessee_id VARCHAR(100),
    electricity_kwh DECIMAL(15,4),
    gas_kwh DECIMAL(15,4),
    steam_kwh DECIMAL(15,4),
    cooling_kwh DECIMAL(15,4),
    collection_date DATE,
    coverage_pct DECIMAL(5,4) DEFAULT 1.0,
    data_source VARCHAR(100),
    verified BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dla_ted_elec_positive CHECK (electricity_kwh IS NULL OR electricity_kwh >= 0),
    CONSTRAINT chk_dla_ted_gas_positive CHECK (gas_kwh IS NULL OR gas_kwh >= 0),
    CONSTRAINT chk_dla_ted_steam_positive CHECK (steam_kwh IS NULL OR steam_kwh >= 0),
    CONSTRAINT chk_dla_ted_cool_positive CHECK (cooling_kwh IS NULL OR cooling_kwh >= 0),
    CONSTRAINT chk_dla_ted_coverage_range CHECK (coverage_pct >= 0 AND coverage_pct <= 1.0)
);

CREATE INDEX idx_dla_ted_tenant ON downstream_leased_assets_service.gl_dla_tenant_energy_data(tenant_id);
CREATE INDEX idx_dla_ted_asset ON downstream_leased_assets_service.gl_dla_tenant_energy_data(asset_result_id);
CREATE INDEX idx_dla_ted_lessee ON downstream_leased_assets_service.gl_dla_tenant_energy_data(lessee_id);
CREATE INDEX idx_dla_ted_date ON downstream_leased_assets_service.gl_dla_tenant_energy_data(collection_date);
CREATE INDEX idx_dla_ted_verified ON downstream_leased_assets_service.gl_dla_tenant_energy_data(verified) WHERE verified = TRUE;

COMMENT ON TABLE downstream_leased_assets_service.gl_dla_tenant_energy_data IS 'Collected tenant energy consumption data for leased-out buildings';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_tenant_energy_data.coverage_pct IS 'Data coverage percentage (1.0=full year, 0.5=6 months, etc.)';
COMMENT ON COLUMN downstream_leased_assets_service.gl_dla_tenant_energy_data.verified IS 'Whether tenant-reported data has been verified/audited';

-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Continuous Aggregate 1: Daily emissions by asset type
CREATE MATERIALIZED VIEW downstream_leased_assets_service.gl_dla_daily_by_asset_type
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    tenant_id,
    method,
    COUNT(*) AS calc_count,
    SUM(total_co2e_kg) AS total_co2e_kg,
    SUM(building_co2e_kg) AS building_co2e_kg,
    SUM(vehicle_co2e_kg) AS vehicle_co2e_kg,
    SUM(equipment_co2e_kg) AS equipment_co2e_kg,
    SUM(it_asset_co2e_kg) AS it_asset_co2e_kg,
    SUM(asset_count) AS total_assets,
    AVG(dqi_score) AS avg_dqi_score
FROM downstream_leased_assets_service.gl_dla_calculations
WHERE is_deleted = FALSE
GROUP BY bucket, tenant_id, method
WITH NO DATA;

-- Refresh policy: every 6 hours, lag 12 hours, lookback 30 days
SELECT add_continuous_aggregate_policy('downstream_leased_assets_service.gl_dla_daily_by_asset_type',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW downstream_leased_assets_service.gl_dla_daily_by_asset_type IS 'Daily aggregation of downstream leased asset emissions by method with category subtotals';

-- Continuous Aggregate 2: Monthly emissions by category
CREATE MATERIALIZED VIEW downstream_leased_assets_service.gl_dla_monthly_by_category
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('30 days', calculated_at) AS bucket,
    tenant_id,
    method,
    COUNT(*) AS calc_count,
    SUM(total_co2e_kg) AS total_co2e_kg,
    SUM(building_co2e_kg) AS building_co2e_kg,
    SUM(vehicle_co2e_kg) AS vehicle_co2e_kg,
    SUM(equipment_co2e_kg) AS equipment_co2e_kg,
    SUM(it_asset_co2e_kg) AS it_asset_co2e_kg,
    SUM(asset_count) AS total_assets,
    AVG(dqi_score) AS avg_dqi_score
FROM downstream_leased_assets_service.gl_dla_calculations
WHERE is_deleted = FALSE
GROUP BY bucket, tenant_id, method
WITH NO DATA;

-- Refresh policy: every 24 hours, lag 2 days, lookback 365 days
SELECT add_continuous_aggregate_policy('downstream_leased_assets_service.gl_dla_monthly_by_category',
    start_offset => INTERVAL '365 days',
    end_offset => INTERVAL '2 days',
    schedule_interval => INTERVAL '24 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW downstream_leased_assets_service.gl_dla_monthly_by_category IS 'Monthly aggregation of downstream leased asset emissions by method with category subtotals';

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) - 10 policies
-- ============================================================================

ALTER TABLE downstream_leased_assets_service.gl_dla_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_asset_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_allocation_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_compliance_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_aggregations ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_provenance_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_audit_trail ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_batch_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_data_quality_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_leased_assets_service.gl_dla_tenant_energy_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY dla_calculations_tenant_isolation ON downstream_leased_assets_service.gl_dla_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_asset_results_tenant_isolation ON downstream_leased_assets_service.gl_dla_asset_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_allocation_records_tenant_isolation ON downstream_leased_assets_service.gl_dla_allocation_records
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_compliance_checks_tenant_isolation ON downstream_leased_assets_service.gl_dla_compliance_checks
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_aggregations_tenant_isolation ON downstream_leased_assets_service.gl_dla_aggregations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_provenance_records_tenant_isolation ON downstream_leased_assets_service.gl_dla_provenance_records
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_audit_trail_tenant_isolation ON downstream_leased_assets_service.gl_dla_audit_trail
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_batch_jobs_tenant_isolation ON downstream_leased_assets_service.gl_dla_batch_jobs
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_data_quality_tenant_isolation ON downstream_leased_assets_service.gl_dla_data_quality_scores
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dla_tenant_energy_tenant_isolation ON downstream_leased_assets_service.gl_dla_tenant_energy_data
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- ============================================================================
-- SEED DATA: BUILDING BENCHMARKS (8 types x 5 climate zones = 40 rows)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_building_benchmarks
(building_type, climate_zone, eui_kwh_per_sqm, electricity_share, gas_share, steam_share, cooling_share, nabers_rating, epc_grade, source, year) VALUES
-- Office
('office', '1A_very_hot_humid', 280.0000, 0.75, 0.05, 0.05, 0.15, '2.5', 'D', 'ASHRAE_90.1_2019', 2024),
('office', '2B_hot_dry',        250.0000, 0.70, 0.10, 0.05, 0.15, '3.0', 'D', 'ASHRAE_90.1_2019', 2024),
('office', '3C_warm_marine',    200.0000, 0.65, 0.20, 0.05, 0.10, '3.5', 'C', 'ASHRAE_90.1_2019', 2024),
('office', '4A_mixed_humid',    220.0000, 0.60, 0.25, 0.05, 0.10, '3.0', 'C', 'ASHRAE_90.1_2019', 2024),
('office', '5A_cool_humid',     240.0000, 0.55, 0.30, 0.10, 0.05, '2.5', 'D', 'ASHRAE_90.1_2019', 2024),
-- Retail
('retail', '1A_very_hot_humid', 320.0000, 0.80, 0.05, 0.00, 0.15, '2.0', 'E', 'ASHRAE_90.1_2019', 2024),
('retail', '2B_hot_dry',        290.0000, 0.75, 0.10, 0.00, 0.15, '2.5', 'D', 'ASHRAE_90.1_2019', 2024),
('retail', '3C_warm_marine',    240.0000, 0.70, 0.15, 0.05, 0.10, '3.0', 'D', 'ASHRAE_90.1_2019', 2024),
('retail', '4A_mixed_humid',    260.0000, 0.65, 0.20, 0.05, 0.10, '2.5', 'D', 'ASHRAE_90.1_2019', 2024),
('retail', '5A_cool_humid',     280.0000, 0.60, 0.25, 0.10, 0.05, '2.0', 'E', 'ASHRAE_90.1_2019', 2024),
-- Warehouse
('warehouse', '1A_very_hot_humid', 120.0000, 0.60, 0.15, 0.05, 0.20, '4.0', 'C', 'ASHRAE_90.1_2019', 2024),
('warehouse', '2B_hot_dry',        100.0000, 0.55, 0.20, 0.05, 0.20, '4.5', 'B', 'ASHRAE_90.1_2019', 2024),
('warehouse', '3C_warm_marine',     80.0000, 0.50, 0.30, 0.10, 0.10, '5.0', 'B', 'ASHRAE_90.1_2019', 2024),
('warehouse', '4A_mixed_humid',     90.0000, 0.50, 0.35, 0.10, 0.05, '4.5', 'B', 'ASHRAE_90.1_2019', 2024),
('warehouse', '5A_cool_humid',     110.0000, 0.45, 0.40, 0.10, 0.05, '4.0', 'C', 'ASHRAE_90.1_2019', 2024),
-- Industrial
('industrial', '1A_very_hot_humid', 350.0000, 0.70, 0.15, 0.05, 0.10, '2.0', 'E', 'CIBSE_TM46', 2024),
('industrial', '2B_hot_dry',        320.0000, 0.65, 0.20, 0.05, 0.10, '2.5', 'D', 'CIBSE_TM46', 2024),
('industrial', '3C_warm_marine',    280.0000, 0.60, 0.25, 0.05, 0.10, '3.0', 'D', 'CIBSE_TM46', 2024),
('industrial', '4A_mixed_humid',    300.0000, 0.55, 0.30, 0.05, 0.10, '2.5', 'D', 'CIBSE_TM46', 2024),
('industrial', '5A_cool_humid',     330.0000, 0.50, 0.35, 0.10, 0.05, '2.0', 'E', 'CIBSE_TM46', 2024),
-- Data Center
('data_center', '1A_very_hot_humid', 2500.0000, 0.95, 0.00, 0.00, 0.05, '2.0', 'E', 'Industry_Average_2024', 2024),
('data_center', '2B_hot_dry',        2200.0000, 0.95, 0.00, 0.00, 0.05, '2.5', 'D', 'Industry_Average_2024', 2024),
('data_center', '3C_warm_marine',    1800.0000, 0.95, 0.00, 0.00, 0.05, '3.5', 'C', 'Industry_Average_2024', 2024),
('data_center', '4A_mixed_humid',    2000.0000, 0.95, 0.00, 0.00, 0.05, '3.0', 'D', 'Industry_Average_2024', 2024),
('data_center', '5A_cool_humid',     1600.0000, 0.95, 0.00, 0.00, 0.05, '4.0', 'C', 'Industry_Average_2024', 2024),
-- Hospital
('hospital', '1A_very_hot_humid', 450.0000, 0.60, 0.20, 0.10, 0.10, '2.0', 'E', 'ASHRAE_90.1_2019', 2024),
('hospital', '2B_hot_dry',        420.0000, 0.55, 0.25, 0.10, 0.10, '2.5', 'D', 'ASHRAE_90.1_2019', 2024),
('hospital', '3C_warm_marine',    380.0000, 0.50, 0.30, 0.10, 0.10, '3.0', 'D', 'ASHRAE_90.1_2019', 2024),
('hospital', '4A_mixed_humid',    400.0000, 0.50, 0.30, 0.10, 0.10, '2.5', 'D', 'ASHRAE_90.1_2019', 2024),
('hospital', '5A_cool_humid',     430.0000, 0.45, 0.35, 0.15, 0.05, '2.0', 'E', 'ASHRAE_90.1_2019', 2024),
-- Hotel
('hotel', '1A_very_hot_humid', 300.0000, 0.65, 0.15, 0.05, 0.15, '2.5', 'D', 'CIBSE_TM46', 2024),
('hotel', '2B_hot_dry',        270.0000, 0.60, 0.20, 0.05, 0.15, '3.0', 'D', 'CIBSE_TM46', 2024),
('hotel', '3C_warm_marine',    230.0000, 0.55, 0.25, 0.05, 0.15, '3.5', 'C', 'CIBSE_TM46', 2024),
('hotel', '4A_mixed_humid',    250.0000, 0.55, 0.25, 0.10, 0.10, '3.0', 'D', 'CIBSE_TM46', 2024),
('hotel', '5A_cool_humid',     270.0000, 0.50, 0.30, 0.10, 0.10, '2.5', 'D', 'CIBSE_TM46', 2024),
-- Mixed Use
('mixed_use', '1A_very_hot_humid', 260.0000, 0.70, 0.10, 0.05, 0.15, '2.5', 'D', 'ASHRAE_90.1_2019', 2024),
('mixed_use', '2B_hot_dry',        235.0000, 0.65, 0.15, 0.05, 0.15, '3.0', 'D', 'ASHRAE_90.1_2019', 2024),
('mixed_use', '3C_warm_marine',    195.0000, 0.60, 0.20, 0.05, 0.15, '3.5', 'C', 'ASHRAE_90.1_2019', 2024),
('mixed_use', '4A_mixed_humid',    210.0000, 0.60, 0.25, 0.05, 0.10, '3.0', 'C', 'ASHRAE_90.1_2019', 2024),
('mixed_use', '5A_cool_humid',     230.0000, 0.55, 0.30, 0.10, 0.05, '2.5', 'D', 'ASHRAE_90.1_2019', 2024);

-- ============================================================================
-- SEED DATA: VEHICLE EMISSION FACTORS (8 types x 7 fuels = 56 rows)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_vehicle_emission_factors
(vehicle_type, fuel_type, ef_per_km, wtt_ef_per_km, unit, source, year) VALUES
-- Small Car
('small_car', 'petrol',  0.14890000, 0.03490000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('small_car', 'diesel',  0.13920000, 0.03260000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('small_car', 'hybrid',  0.10410000, 0.02440000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('small_car', 'phev',    0.06920000, 0.01810000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('small_car', 'ev',      0.04600000, 0.01330000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('small_car', 'lpg',     0.17850000, 0.01070000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('small_car', 'cng',     0.15820000, 0.02740000, 'kgCO2e/km', 'DEFRA_2024', 2024),
-- Medium Car
('medium_car', 'petrol',  0.18770000, 0.04400000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('medium_car', 'diesel',  0.16610000, 0.03890000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('medium_car', 'hybrid',  0.11590000, 0.02720000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('medium_car', 'phev',    0.07500000, 0.01960000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('medium_car', 'ev',      0.05300000, 0.01530000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('medium_car', 'lpg',     0.22520000, 0.01350000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('medium_car', 'cng',     0.19950000, 0.03460000, 'kgCO2e/km', 'DEFRA_2024', 2024),
-- Large Car
('large_car', 'petrol',  0.27870000, 0.06530000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('large_car', 'diesel',  0.20870000, 0.04890000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('large_car', 'hybrid',  0.16380000, 0.03840000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('large_car', 'phev',    0.09800000, 0.02560000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('large_car', 'ev',      0.06800000, 0.01960000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('large_car', 'lpg',     0.33440000, 0.02010000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('large_car', 'cng',     0.29610000, 0.05130000, 'kgCO2e/km', 'DEFRA_2024', 2024),
-- SUV
('suv', 'petrol',  0.23100000, 0.05410000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('suv', 'diesel',  0.19760000, 0.04630000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('suv', 'hybrid',  0.14900000, 0.03490000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('suv', 'phev',    0.09200000, 0.02410000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('suv', 'ev',      0.06200000, 0.01790000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('suv', 'lpg',     0.27720000, 0.01660000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('suv', 'cng',     0.24530000, 0.04250000, 'kgCO2e/km', 'DEFRA_2024', 2024),
-- Light Van
('light_van', 'petrol',  0.19930000, 0.04670000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_van', 'diesel',  0.17790000, 0.04170000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_van', 'hybrid',  0.14350000, 0.03360000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_van', 'phev',    0.09500000, 0.02490000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_van', 'ev',      0.06500000, 0.01880000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_van', 'lpg',     0.23920000, 0.01440000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_van', 'cng',     0.21170000, 0.03670000, 'kgCO2e/km', 'DEFRA_2024', 2024),
-- Heavy Van
('heavy_van', 'petrol',  0.26900000, 0.06300000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_van', 'diesel',  0.24310000, 0.05700000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_van', 'hybrid',  0.19450000, 0.04560000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_van', 'phev',    0.12900000, 0.03380000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_van', 'ev',      0.09100000, 0.02630000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_van', 'lpg',     0.32280000, 0.01940000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_van', 'cng',     0.28570000, 0.04950000, 'kgCO2e/km', 'DEFRA_2024', 2024),
-- Light Truck
('light_truck', 'petrol',  0.36100000, 0.08460000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_truck', 'diesel',  0.31200000, 0.07310000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_truck', 'hybrid',  0.26010000, 0.06100000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_truck', 'phev',    0.17260000, 0.04520000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_truck', 'ev',      0.12200000, 0.03520000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_truck', 'lpg',     0.43320000, 0.02600000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('light_truck', 'cng',     0.38340000, 0.06640000, 'kgCO2e/km', 'DEFRA_2024', 2024),
-- Heavy Truck
('heavy_truck', 'petrol',  0.88200000, 0.20670000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_truck', 'diesel',  0.76900000, 0.18020000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_truck', 'hybrid',  0.65870000, 0.15440000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_truck', 'phev',    0.46130000, 0.12080000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_truck', 'ev',      0.34500000, 0.09960000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_truck', 'lpg',     1.05840000, 0.06350000, 'kgCO2e/km', 'DEFRA_2024', 2024),
('heavy_truck', 'cng',     0.93690000, 0.16230000, 'kgCO2e/km', 'DEFRA_2024', 2024);

-- ============================================================================
-- SEED DATA: EQUIPMENT FACTORS (6 types)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_equipment_factors
(equipment_type, rated_power_kw, fuel_consumption_l_per_hr, default_load_factor, default_fuel_type, ef_per_litre, wtt_ef_per_litre, source, year) VALUES
('generator',  100.0000, 28.000000, 0.75, 'diesel', 2.68780, 0.63010, 'DEFRA_2024', 2024),
('compressor',  50.0000, 14.000000, 0.60, 'diesel', 2.68780, 0.63010, 'DEFRA_2024', 2024),
('pump',        30.0000,  8.400000, 0.55, 'diesel', 2.68780, 0.63010, 'DEFRA_2024', 2024),
('forklift',    25.0000,  7.000000, 0.50, 'diesel', 2.68780, 0.63010, 'DEFRA_2024', 2024),
('crane',      200.0000, 56.000000, 0.40, 'diesel', 2.68780, 0.63010, 'DEFRA_2024', 2024),
('hvac_unit',   15.0000,  4.200000, 0.65, 'diesel', 2.68780, 0.63010, 'DEFRA_2024', 2024);

-- ============================================================================
-- SEED DATA: IT ASSET FACTORS (7 types)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_it_asset_factors
(it_asset_type, default_power_kw, default_pue, default_hours_per_year, source, year) VALUES
('server',         0.500000, 1.58, 8760.00, 'Industry_Average_2024', 2024),
('storage',        0.200000, 1.58, 8760.00, 'Industry_Average_2024', 2024),
('network_switch', 0.080000, 1.58, 8760.00, 'Industry_Average_2024', 2024),
('router',         0.050000, 1.58, 8760.00, 'Industry_Average_2024', 2024),
('ups',            0.300000, 1.10, 8760.00, 'Industry_Average_2024', 2024),
('cooling_unit',   5.000000, 1.00, 8760.00, 'Industry_Average_2024', 2024),
('workstation',    0.150000, 1.20, 2080.00, 'Industry_Average_2024', 2024);

-- ============================================================================
-- SEED DATA: GRID EMISSION FACTORS (12 countries + 26 eGRID subregions = 38 rows)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_grid_emission_factors
(region_code, region_name, region_type, ef_kgco2e_per_kwh, wtt_ef_kgco2e_per_kwh, td_loss_factor, source, year) VALUES
-- Countries (IEA 2024)
('US',    'United States',     'country', 0.38600000, 0.04560000, 0.05, 'IEA_2024', 2024),
('GB',    'United Kingdom',    'country', 0.20700000, 0.02440000, 0.08, 'IEA_2024', 2024),
('DE',    'Germany',           'country', 0.35000000, 0.04130000, 0.04, 'IEA_2024', 2024),
('FR',    'France',            'country', 0.05200000, 0.00610000, 0.06, 'IEA_2024', 2024),
('JP',    'Japan',             'country', 0.45700000, 0.05390000, 0.05, 'IEA_2024', 2024),
('CN',    'China',             'country', 0.55500000, 0.06550000, 0.06, 'IEA_2024', 2024),
('IN',    'India',             'country', 0.70800000, 0.08350000, 0.19, 'IEA_2024', 2024),
('AU',    'Australia',         'country', 0.65600000, 0.07740000, 0.05, 'IEA_2024', 2024),
('CA',    'Canada',            'country', 0.12000000, 0.01420000, 0.08, 'IEA_2024', 2024),
('BR',    'Brazil',            'country', 0.07400000, 0.00870000, 0.16, 'IEA_2024', 2024),
('SG',    'Singapore',         'country', 0.40800000, 0.04810000, 0.03, 'IEA_2024', 2024),
('AE',    'United Arab Emirates', 'country', 0.56300000, 0.06640000, 0.08, 'IEA_2024', 2024),
-- eGRID Subregions (EPA eGRID 2024)
('AKGD',  'ASCC Alaska Grid',         'egrid_subregion', 0.43200000, 0.05100000, 0.07, 'EPA_eGRID_2024', 2024),
('AKMS',  'ASCC Miscellaneous',       'egrid_subregion', 0.23500000, 0.02770000, 0.06, 'EPA_eGRID_2024', 2024),
('AZNM',  'WECC Southwest',           'egrid_subregion', 0.38900000, 0.04590000, 0.05, 'EPA_eGRID_2024', 2024),
('CAMX',  'WECC California',          'egrid_subregion', 0.22800000, 0.02690000, 0.06, 'EPA_eGRID_2024', 2024),
('ERCT',  'ERCOT All',                'egrid_subregion', 0.37900000, 0.04470000, 0.05, 'EPA_eGRID_2024', 2024),
('FRCC',  'FRCC All',                 'egrid_subregion', 0.36800000, 0.04340000, 0.05, 'EPA_eGRID_2024', 2024),
('HIMS',  'HICC Miscellaneous',       'egrid_subregion', 0.56600000, 0.06680000, 0.06, 'EPA_eGRID_2024', 2024),
('HIOA',  'HICC Oahu',                'egrid_subregion', 0.63400000, 0.07480000, 0.06, 'EPA_eGRID_2024', 2024),
('MROE',  'MRO East',                 'egrid_subregion', 0.55900000, 0.06590000, 0.05, 'EPA_eGRID_2024', 2024),
('MROW',  'MRO West',                 'egrid_subregion', 0.45200000, 0.05330000, 0.05, 'EPA_eGRID_2024', 2024),
('NEWE',  'NPCC New England',         'egrid_subregion', 0.22100000, 0.02610000, 0.06, 'EPA_eGRID_2024', 2024),
('NWPP',  'WECC Northwest',           'egrid_subregion', 0.27200000, 0.03210000, 0.06, 'EPA_eGRID_2024', 2024),
('NYCW',  'NPCC NYC/Westchester',     'egrid_subregion', 0.24700000, 0.02910000, 0.06, 'EPA_eGRID_2024', 2024),
('NYLI',  'NPCC Long Island',         'egrid_subregion', 0.48800000, 0.05760000, 0.05, 'EPA_eGRID_2024', 2024),
('NYUP',  'NPCC Upstate NY',          'egrid_subregion', 0.09600000, 0.01130000, 0.07, 'EPA_eGRID_2024', 2024),
('PRMS',  'WECC Rockies',             'egrid_subregion', 0.51300000, 0.06050000, 0.06, 'EPA_eGRID_2024', 2024),
('RFCE',  'RFC East',                 'egrid_subregion', 0.30100000, 0.03550000, 0.05, 'EPA_eGRID_2024', 2024),
('RFCM',  'RFC Michigan',             'egrid_subregion', 0.49400000, 0.05830000, 0.05, 'EPA_eGRID_2024', 2024),
('RFCW',  'RFC West',                 'egrid_subregion', 0.46100000, 0.05440000, 0.05, 'EPA_eGRID_2024', 2024),
('RMPA',  'WECC S. Rockies',          'egrid_subregion', 0.55800000, 0.06580000, 0.06, 'EPA_eGRID_2024', 2024),
('SPNO',  'SPP North',                'egrid_subregion', 0.47700000, 0.05630000, 0.05, 'EPA_eGRID_2024', 2024),
('SPSO',  'SPP South',                'egrid_subregion', 0.41300000, 0.04870000, 0.05, 'EPA_eGRID_2024', 2024),
('SRMV',  'SERC Mississippi Valley',  'egrid_subregion', 0.33700000, 0.03980000, 0.05, 'EPA_eGRID_2024', 2024),
('SRMW',  'SERC Midwest',             'egrid_subregion', 0.65200000, 0.07690000, 0.05, 'EPA_eGRID_2024', 2024),
('SRSO',  'SERC South',               'egrid_subregion', 0.41600000, 0.04910000, 0.05, 'EPA_eGRID_2024', 2024),
('SRVC',  'SERC Virginia/Carolina',   'egrid_subregion', 0.30700000, 0.03620000, 0.05, 'EPA_eGRID_2024', 2024);

-- ============================================================================
-- SEED DATA: FUEL EMISSION FACTORS (8 fuel types)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_fuel_emission_factors
(fuel_type, ef_kgco2e_per_kwh, wtt_ef_kgco2e_per_kwh, ef_kgco2e_per_litre, density_kg_per_litre, calorific_value_kwh_per_litre, source, year) VALUES
('natural_gas',   0.18387000, 0.02350000, NULL,      NULL,      NULL,    'DEFRA_2024', 2024),
('diesel',        0.25320000, 0.05940000, 2.68780,   0.845000,  10.610,  'DEFRA_2024', 2024),
('petrol',        0.23190000, 0.05440000, 2.31480,   0.749000,  9.980,   'DEFRA_2024', 2024),
('lpg',           0.21440000, 0.01290000, 1.55730,   0.510000,  7.270,   'DEFRA_2024', 2024),
('fuel_oil',      0.26820000, 0.05260000, 3.17710,   0.940000,  11.840,  'DEFRA_2024', 2024),
('steam',         0.19400000, 0.02480000, NULL,      NULL,      NULL,    'DEFRA_2024', 2024),
('district_cooling', 0.15800000, 0.01860000, NULL,   NULL,      NULL,    'Industry_Average_2024', 2024),
('biofuel',       0.01150000, 0.05910000, 0.09950,   0.880000,  8.650,   'DEFRA_2024', 2024);

-- ============================================================================
-- SEED DATA: EEIO SPEND FACTORS (10 NAICS leasing/rental codes)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_eeio_spend_factors
(naics_code, category_name, ef_per_usd, base_year, source) VALUES
('531110', 'Lessors of Residential Buildings and Dwellings',   0.19200000, 2021, 'EPA_USEEIO_v2'),
('531120', 'Lessors of Nonresidential Buildings',              0.22800000, 2021, 'EPA_USEEIO_v2'),
('531130', 'Lessors of Miniwarehouses and Self-Storage',       0.15600000, 2021, 'EPA_USEEIO_v2'),
('531190', 'Lessors of Other Real Estate Property',            0.18400000, 2021, 'EPA_USEEIO_v2'),
('532100', 'Automotive Equipment Rental and Leasing',          0.25600000, 2021, 'EPA_USEEIO_v2'),
('532400', 'Commercial Equipment Rental and Leasing',          0.31200000, 2021, 'EPA_USEEIO_v2'),
('518210', 'Data Processing, Hosting, and Related Services',   0.14800000, 2021, 'EPA_USEEIO_v2'),
('532310', 'General Rental Centers',                           0.28400000, 2021, 'EPA_USEEIO_v2'),
('533110', 'Lessors of Nonfinancial Intangible Assets',        0.08200000, 2021, 'EPA_USEEIO_v2'),
('531210', 'Offices of Real Estate Agents and Brokers',        0.10500000, 2021, 'EPA_USEEIO_v2');

-- ============================================================================
-- SEED DATA: REFRIGERANT GWPs (15 common refrigerants, IPCC AR6)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_refrigerant_gwps
(refrigerant, chemical_formula, gwp_100yr, gwp_source, ozone_depleting, phase_down_schedule) VALUES
('R-32',    'CH2F2',                       771,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-125',   'CHF2CF3',                    3740,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-134a',  'CH2FCF3',                    1530,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-143a',  'CH3CF3',                     5810,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-152a',  'CH3CHF2',                     164,  'IPCC_AR6', FALSE, NULL),
('R-227ea', 'CF3CHFCF3',                  3600,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-404A',  'R125/143a/134a (44/52/4)',   4728,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-407C',  'R32/125/134a (23/25/52)',    1908,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-410A',  'R32/125 (50/50)',            2256,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-507A',  'R125/143a (50/50)',          4766,  'IPCC_AR6', FALSE, 'Kigali_2024'),
('R-22',    'CHClF2',                     1960,  'IPCC_AR6', TRUE,  'Montreal_2030'),
('R-290',   'C3H8 (Propane)',                3,  'IPCC_AR6', FALSE, NULL),
('R-600a',  'C4H10 (Isobutane)',             3,  'IPCC_AR6', FALSE, NULL),
('R-717',   'NH3 (Ammonia)',                 0,  'IPCC_AR6', FALSE, NULL),
('R-744',   'CO2',                           1,  'IPCC_AR6', FALSE, NULL);

-- ============================================================================
-- SEED DATA: VACANCY FACTORS (8 building types)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_vacancy_factors
(building_type, base_load_fraction, description, source) VALUES
('office',      0.35, 'Office base load: HVAC setback, lighting off, security systems',             'ASHRAE_Guideline_14'),
('retail',      0.25, 'Retail base load: minimal HVAC, security, refrigeration if applicable',      'ASHRAE_Guideline_14'),
('warehouse',   0.15, 'Warehouse base load: minimal ventilation, lighting off',                     'ASHRAE_Guideline_14'),
('industrial',  0.20, 'Industrial base load: process standby, ventilation, safety systems',         'ASHRAE_Guideline_14'),
('data_center', 0.80, 'Data center base load: cooling, UPS, network equipment always on',           'Industry_Average_2024'),
('hospital',    0.60, 'Hospital base load: critical systems, HVAC, medical equipment',              'ASHRAE_Guideline_14'),
('hotel',       0.30, 'Hotel base load: common areas, kitchen standby, HVAC setback',               'CIBSE_TM46'),
('mixed_use',   0.30, 'Mixed-use base load: weighted average of component types',                   'ASHRAE_Guideline_14');

-- ============================================================================
-- SEED DATA: ALLOCATION DEFAULTS (6 methods)
-- ============================================================================

INSERT INTO downstream_leased_assets_service.gl_dla_allocation_defaults
(allocation_method, description, default_common_area_pct, default_lessor_share, requires_tenant_data, data_quality_tier) VALUES
('floor_area',   'Allocate based on leased floor area as proportion of total',       0.15, 0.00, FALSE, 1),
('headcount',    'Allocate based on tenant headcount as proportion of total',         0.15, 0.00, TRUE,  2),
('revenue',      'Allocate based on lease revenue as proportion of total revenue',    0.15, 0.00, TRUE,  2),
('time_based',   'Allocate based on lease duration as proportion of reporting period', 0.15, 0.00, FALSE, 2),
('equal_split',  'Equal allocation among all tenants (common when data unavailable)', 0.15, 0.00, FALSE, 3),
('full',         'Full allocation to lessor (when tenant data is not available)',      0.00, 1.00, FALSE, 4);

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
    'GL-MRV-S3-013',
    'Downstream Leased Assets Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-026: Scope 3 Category 13 - Downstream Leased Assets. '
    'Calculates emissions from assets OWNED by the reporting company and LEASED TO other entities (reporter is LESSOR). '
    'Mirror of Category 8 from lessor perspective. Supports 4 asset categories: buildings (8 types, 5 climate zones, '
    'EUI benchmarks, vacancy handling), vehicles (8 types, 7 fuels), equipment (6 types), IT assets (7 types). '
    '4 calculation methods: asset-specific (metered tenant data), average-data (benchmarks), spend-based (EEIO), hybrid. '
    '6 allocation methods: floor_area, headcount, revenue, time_based, equal_split, full. '
    '7 compliance frameworks. 188+ seed records across 10 reference tables.',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 13,
        'category_name', 'Downstream Leased Assets',
        'reporter_role', 'LESSOR',
        'mirror_of', 'Category 8 (Upstream Leased Assets)',
        'asset_categories', jsonb_build_array('building', 'vehicle', 'equipment', 'it_asset'),
        'building_types', jsonb_build_array('office', 'retail', 'warehouse', 'industrial', 'data_center', 'hospital', 'hotel', 'mixed_use'),
        'vehicle_types', jsonb_build_array('small_car', 'medium_car', 'large_car', 'suv', 'light_van', 'heavy_van', 'light_truck', 'heavy_truck'),
        'equipment_types', jsonb_build_array('generator', 'compressor', 'pump', 'forklift', 'crane', 'hvac_unit'),
        'it_asset_types', jsonb_build_array('server', 'storage', 'network_switch', 'router', 'ups', 'cooling_unit', 'workstation'),
        'calculation_methods', jsonb_build_array('asset_specific', 'average_data', 'spend_based', 'hybrid'),
        'allocation_methods', jsonb_build_array('floor_area', 'headcount', 'revenue', 'time_based', 'equal_split', 'full'),
        'climate_zones', jsonb_build_array('1A_very_hot_humid', '2B_hot_dry', '3C_warm_marine', '4A_mixed_humid', '5A_cool_humid'),
        'frameworks', jsonb_build_array('GHG Protocol Scope 3', 'ISO 14064-1', 'CSRD ESRS E1', 'CDP Climate', 'SBTi', 'SB 253', 'GRI 305'),
        'building_benchmark_count', 40,
        'vehicle_ef_count', 56,
        'equipment_factor_count', 6,
        'it_asset_factor_count', 7,
        'grid_ef_count', 38,
        'fuel_ef_count', 8,
        'eeio_factor_count', 10,
        'refrigerant_count', 15,
        'vacancy_factor_count', 8,
        'total_seed_records', 188,
        'supports_vacancy_handling', true,
        'supports_allocation', true,
        'supports_tenant_data_collection', true,
        'default_ef_source', 'DEFRA_2024',
        'default_grid_source', 'IEA_2024',
        'default_gwp', 'AR6',
        'schema', 'downstream_leased_assets_service',
        'table_prefix', 'gl_dla_',
        'table_count', 21,
        'hypertables', jsonb_build_array('gl_dla_calculations', 'gl_dla_compliance_checks', 'gl_dla_aggregations'),
        'continuous_aggregates', jsonb_build_array('gl_dla_daily_by_asset_type', 'gl_dla_monthly_by_category'),
        'rls_policy_count', 10,
        'migration_version', 'V077'
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

COMMENT ON SCHEMA downstream_leased_assets_service IS
    'Updated: AGENT-MRV-026 complete with 21 tables, 3 hypertables, 2 continuous aggregates, '
    '10 RLS policies, 188+ seed records. Scope 3 Category 13 - Downstream Leased Assets.';

-- ============================================================================
-- END OF MIGRATION V077
-- ============================================================================
-- Total Lines: ~1100
-- Total Tables: 21 (10 reference + 8 operational + 3 supporting)
-- Total Hypertables: 3 (calculations 7-day, compliance_checks 30-day, aggregations 30-day)
-- Total Continuous Aggregates: 2 (gl_dla_daily_by_asset_type, gl_dla_monthly_by_category)
-- Total RLS Policies: 10
-- Total Seed Records: 188+
--   Building Benchmarks: 40 (8 types x 5 climate zones)
--   Vehicle Emission Factors: 56 (8 types x 7 fuels)
--   Equipment Factors: 6
--   IT Asset Factors: 7
--   Grid Emission Factors: 38 (12 countries + 26 eGRID subregions)
--   Fuel Emission Factors: 8
--   EEIO Spend Factors: 10
--   Refrigerant GWPs: 15
--   Vacancy Factors: 8
--   Allocation Defaults: 6 (not counted in 188 reference total)
--   Agent Registry: 1
-- ============================================================================

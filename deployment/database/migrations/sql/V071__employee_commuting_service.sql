-- ============================================================================
-- Migration: V071__employee_commuting_service.sql
-- Description: AGENT-MRV-020 Employee Commuting (Scope 3 Category 7)
-- Agent: GL-MRV-S3-007
-- Framework: GHG Protocol Scope 3 Standard, DEFRA 2024, IEA 2024, EPA USEEIO
-- Created: 2026-02-26
-- ============================================================================
-- Schema: employee_commuting_service
-- Tables: 16 (5 reference + 11 operational)
-- Hypertables: 3 (calculations, vehicle_results, aggregations)
-- Continuous Aggregates: 2 (hourly_emissions, daily_emissions)
-- RLS: Enabled on calculations, vehicle_results, transit_results, telework_results, spend_results
-- Seed Data: 75+ records (vehicle EFs, transit EFs, grid EFs, telework factors, EEIO factors)
-- ============================================================================

-- ============================================================================
-- SCHEMA CREATION
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS employee_commuting_service;

COMMENT ON SCHEMA employee_commuting_service IS 'AGENT-MRV-020: Employee Commuting - Scope 3 Category 7 emission calculations (vehicle/transit/cycling/telework/spend-based)';

-- ============================================================================
-- TABLE 1: gl_ec_employees
-- Description: Employee registry for commuting with work schedule and telework info
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_employees (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    employee_id VARCHAR(100) NOT NULL,
    department VARCHAR(200),
    location VARCHAR(200),
    country_code VARCHAR(3) DEFAULT 'US',
    work_schedule VARCHAR(50) DEFAULT 'standard_5day',
    telework_category VARCHAR(50) DEFAULT 'office_based',
    commute_distance_km NUMERIC(12,4),
    primary_commute_mode VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, employee_id),
    CONSTRAINT chk_ec_work_schedule CHECK (work_schedule IN ('standard_5day', 'standard_4day', 'compressed_9_80', 'part_time_3day', 'part_time_2day', 'shift_rotating', 'flexible', 'custom')),
    CONSTRAINT chk_ec_telework_category CHECK (telework_category IN ('office_based', 'hybrid_1day', 'hybrid_2day', 'hybrid_3day', 'hybrid_4day', 'fully_remote', 'field_based')),
    CONSTRAINT chk_ec_commute_distance_positive CHECK (commute_distance_km IS NULL OR commute_distance_km >= 0),
    CONSTRAINT chk_ec_primary_mode CHECK (primary_commute_mode IS NULL OR primary_commute_mode IN (
        'car_petrol', 'car_diesel', 'car_hybrid', 'car_phev', 'car_ev', 'car_lpg', 'car_cng',
        'motorcycle', 'bus', 'coach', 'rail', 'light_rail', 'subway', 'ferry',
        'bicycle', 'e_bicycle', 'e_scooter', 'walking', 'carpool', 'vanpool'))
);

CREATE INDEX idx_ec_employees_tenant ON employee_commuting_service.gl_ec_employees(tenant_id);
CREATE INDEX idx_ec_employees_employee_id ON employee_commuting_service.gl_ec_employees(employee_id);
CREATE INDEX idx_ec_employees_department ON employee_commuting_service.gl_ec_employees(department);
CREATE INDEX idx_ec_employees_location ON employee_commuting_service.gl_ec_employees(location);
CREATE INDEX idx_ec_employees_country ON employee_commuting_service.gl_ec_employees(country_code);
CREATE INDEX idx_ec_employees_telework ON employee_commuting_service.gl_ec_employees(telework_category);
CREATE INDEX idx_ec_employees_mode ON employee_commuting_service.gl_ec_employees(primary_commute_mode);

COMMENT ON TABLE employee_commuting_service.gl_ec_employees IS 'Employee registry for commuting with work schedule, telework category, and primary commute mode';
COMMENT ON COLUMN employee_commuting_service.gl_ec_employees.work_schedule IS 'Work schedule: standard_5day, standard_4day, compressed_9_80, part_time_3day, part_time_2day, shift_rotating, flexible, custom';
COMMENT ON COLUMN employee_commuting_service.gl_ec_employees.telework_category IS 'Telework category: office_based, hybrid_1day through hybrid_4day, fully_remote, field_based';
COMMENT ON COLUMN employee_commuting_service.gl_ec_employees.commute_distance_km IS 'One-way commute distance in kilometers';
COMMENT ON COLUMN employee_commuting_service.gl_ec_employees.primary_commute_mode IS 'Primary commute mode used for majority of commute distance';

-- ============================================================================
-- TABLE 2: gl_ec_commute_profiles
-- Description: Employee commute profiles with modal split details
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_commute_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    employee_id UUID NOT NULL REFERENCES employee_commuting_service.gl_ec_employees(id),
    profile_year INT NOT NULL,
    modes JSONB NOT NULL DEFAULT '[]',
    total_one_way_km NUMERIC(12,4),
    wfh_days_per_week INT DEFAULT 0,
    annual_working_days INT DEFAULT 235,
    survey_response_id UUID,
    data_source VARCHAR(50) DEFAULT 'survey',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ec_profile_year_valid CHECK (profile_year >= 2000 AND profile_year <= 2100),
    CONSTRAINT chk_ec_profile_oneway_positive CHECK (total_one_way_km IS NULL OR total_one_way_km >= 0),
    CONSTRAINT chk_ec_profile_wfh_range CHECK (wfh_days_per_week >= 0 AND wfh_days_per_week <= 7),
    CONSTRAINT chk_ec_profile_working_days CHECK (annual_working_days > 0 AND annual_working_days <= 366),
    CONSTRAINT chk_ec_profile_source CHECK (data_source IN ('survey', 'hr_system', 'expense_data', 'gis_estimate', 'default', 'manual'))
);

CREATE INDEX idx_ec_profiles_tenant ON employee_commuting_service.gl_ec_commute_profiles(tenant_id);
CREATE INDEX idx_ec_profiles_employee ON employee_commuting_service.gl_ec_commute_profiles(employee_id);
CREATE INDEX idx_ec_profiles_year ON employee_commuting_service.gl_ec_commute_profiles(profile_year);
CREATE INDEX idx_ec_profiles_source ON employee_commuting_service.gl_ec_commute_profiles(data_source);
CREATE INDEX idx_ec_profiles_modes ON employee_commuting_service.gl_ec_commute_profiles USING GIN(modes);

COMMENT ON TABLE employee_commuting_service.gl_ec_commute_profiles IS 'Employee commute profiles with multi-modal split, WFH days, and annual working days';
COMMENT ON COLUMN employee_commuting_service.gl_ec_commute_profiles.modes IS 'JSONB array of commute modes: [{mode, distance_km, frequency_days_per_week, vehicle_type, fuel_type}]';
COMMENT ON COLUMN employee_commuting_service.gl_ec_commute_profiles.wfh_days_per_week IS 'Number of work-from-home days per week (0-7)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_commute_profiles.annual_working_days IS 'Total working days per year (default 235, net of holidays/vacation)';

-- ============================================================================
-- TABLE 3: gl_ec_vehicle_emission_factors
-- Description: Personal vehicle emission factors by type, fuel, and age (DEFRA 2024)
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_vehicle_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_type VARCHAR(50) NOT NULL,
    fuel_type VARCHAR(50) NOT NULL,
    vehicle_age VARCHAR(50) DEFAULT 'mid_4_7yr',
    co2e_per_km NUMERIC(12,8) NOT NULL,
    co2_per_km NUMERIC(12,8),
    ch4_per_km NUMERIC(12,8),
    n2o_per_km NUMERIC(12,8),
    wtt_factor NUMERIC(12,8),
    unit VARCHAR(50) DEFAULT 'kgCO2e/km',
    source VARCHAR(100) DEFAULT 'DEFRA_2024',
    year INT DEFAULT 2024,
    valid_from DATE,
    valid_to DATE,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(vehicle_type, fuel_type, vehicle_age, source, year),
    CONSTRAINT chk_ec_vef_co2e_positive CHECK (co2e_per_km >= 0),
    CONSTRAINT chk_ec_vef_co2_positive CHECK (co2_per_km IS NULL OR co2_per_km >= 0),
    CONSTRAINT chk_ec_vef_ch4_positive CHECK (ch4_per_km IS NULL OR ch4_per_km >= 0),
    CONSTRAINT chk_ec_vef_n2o_positive CHECK (n2o_per_km IS NULL OR n2o_per_km >= 0),
    CONSTRAINT chk_ec_vef_wtt_positive CHECK (wtt_factor IS NULL OR wtt_factor >= 0),
    CONSTRAINT chk_ec_vef_year_valid CHECK (year >= 1990 AND year <= 2100),
    CONSTRAINT chk_ec_vef_dates_valid CHECK (valid_from IS NULL OR valid_to IS NULL OR valid_to >= valid_from)
);

CREATE INDEX idx_ec_vef_vehicle ON employee_commuting_service.gl_ec_vehicle_emission_factors(vehicle_type);
CREATE INDEX idx_ec_vef_fuel ON employee_commuting_service.gl_ec_vehicle_emission_factors(fuel_type);
CREATE INDEX idx_ec_vef_age ON employee_commuting_service.gl_ec_vehicle_emission_factors(vehicle_age);
CREATE INDEX idx_ec_vef_source ON employee_commuting_service.gl_ec_vehicle_emission_factors(source);
CREATE INDEX idx_ec_vef_active ON employee_commuting_service.gl_ec_vehicle_emission_factors(is_active);

COMMENT ON TABLE employee_commuting_service.gl_ec_vehicle_emission_factors IS 'Personal vehicle emission factors by vehicle type, fuel type, and age band from DEFRA 2024';
COMMENT ON COLUMN employee_commuting_service.gl_ec_vehicle_emission_factors.vehicle_type IS 'Vehicle type: small_car, medium_car, large_car, average_car, motorcycle, e_bicycle, e_scooter';
COMMENT ON COLUMN employee_commuting_service.gl_ec_vehicle_emission_factors.fuel_type IS 'Fuel type: petrol, diesel, hybrid, phev, ev, lpg, cng, electric';
COMMENT ON COLUMN employee_commuting_service.gl_ec_vehicle_emission_factors.vehicle_age IS 'Vehicle age band: new_0_3yr, mid_4_7yr, old_8_plus';
COMMENT ON COLUMN employee_commuting_service.gl_ec_vehicle_emission_factors.wtt_factor IS 'Well-to-tank upstream emission factor (kgCO2e/km)';

-- ============================================================================
-- TABLE 4: gl_ec_transit_emission_factors
-- Description: Public transit emission factors by type and region (DEFRA 2024)
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_transit_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transit_type VARCHAR(50) NOT NULL,
    region VARCHAR(100) DEFAULT 'global',
    co2e_per_pkm NUMERIC(12,8) NOT NULL,
    wtt_factor NUMERIC(12,8),
    unit VARCHAR(50) DEFAULT 'kgCO2e/pkm',
    source VARCHAR(100) DEFAULT 'DEFRA_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(transit_type, region, source, year),
    CONSTRAINT chk_ec_tef_co2e_positive CHECK (co2e_per_pkm >= 0),
    CONSTRAINT chk_ec_tef_wtt_positive CHECK (wtt_factor IS NULL OR wtt_factor >= 0),
    CONSTRAINT chk_ec_tef_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_ec_tef_type ON employee_commuting_service.gl_ec_transit_emission_factors(transit_type);
CREATE INDEX idx_ec_tef_region ON employee_commuting_service.gl_ec_transit_emission_factors(region);
CREATE INDEX idx_ec_tef_source ON employee_commuting_service.gl_ec_transit_emission_factors(source);
CREATE INDEX idx_ec_tef_active ON employee_commuting_service.gl_ec_transit_emission_factors(is_active);

COMMENT ON TABLE employee_commuting_service.gl_ec_transit_emission_factors IS 'Public transit emission factors by transit type and region from DEFRA 2024';
COMMENT ON COLUMN employee_commuting_service.gl_ec_transit_emission_factors.transit_type IS 'Transit type: local_bus, coach, national_rail, light_rail, subway, high_speed_rail, ferry_foot, ferry_car, e_scooter_shared';
COMMENT ON COLUMN employee_commuting_service.gl_ec_transit_emission_factors.co2e_per_pkm IS 'Emission factor per passenger-km (kgCO2e/pkm)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_transit_emission_factors.wtt_factor IS 'Well-to-tank upstream emission factor (kgCO2e/pkm)';

-- ============================================================================
-- TABLE 5: gl_ec_grid_emission_factors
-- Description: Country/regional grid emission factors for telework calculations (IEA 2024)
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_grid_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code VARCHAR(3) NOT NULL,
    region VARCHAR(100),
    co2e_per_kwh NUMERIC(12,8) NOT NULL,
    co2_per_kwh NUMERIC(12,8),
    unit VARCHAR(50) DEFAULT 'kgCO2e/kWh',
    source VARCHAR(100) DEFAULT 'IEA_2024',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(country_code, region, source, year),
    CONSTRAINT chk_ec_gef_co2e_positive CHECK (co2e_per_kwh >= 0),
    CONSTRAINT chk_ec_gef_co2_positive CHECK (co2_per_kwh IS NULL OR co2_per_kwh >= 0),
    CONSTRAINT chk_ec_gef_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_ec_gef_country ON employee_commuting_service.gl_ec_grid_emission_factors(country_code);
CREATE INDEX idx_ec_gef_region ON employee_commuting_service.gl_ec_grid_emission_factors(region);
CREATE INDEX idx_ec_gef_source ON employee_commuting_service.gl_ec_grid_emission_factors(source);
CREATE INDEX idx_ec_gef_active ON employee_commuting_service.gl_ec_grid_emission_factors(is_active);

COMMENT ON TABLE employee_commuting_service.gl_ec_grid_emission_factors IS 'Country/regional electricity grid emission factors from IEA 2024 for telework calculations';
COMMENT ON COLUMN employee_commuting_service.gl_ec_grid_emission_factors.country_code IS 'ISO 3166-1 alpha-2 or alpha-3 country code';
COMMENT ON COLUMN employee_commuting_service.gl_ec_grid_emission_factors.region IS 'Sub-national region (e.g., eGRID subregion for US)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_grid_emission_factors.co2e_per_kwh IS 'Grid emission factor in kgCO2e per kWh consumed';

-- ============================================================================
-- TABLE 6: gl_ec_telework_factors
-- Description: Home office energy consumption defaults by climate zone
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_telework_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    climate_zone VARCHAR(50) NOT NULL,
    electricity_kwh_per_day NUMERIC(12,8),
    heating_kwh_per_day NUMERIC(12,8),
    cooling_kwh_per_day NUMERIC(12,8),
    equipment_kwh_per_day NUMERIC(12,8) DEFAULT 0.30000000,
    unit VARCHAR(50) DEFAULT 'kWh/day',
    source VARCHAR(100) DEFAULT 'IEA_2024',
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(climate_zone, source),
    CONSTRAINT chk_ec_twf_elec_positive CHECK (electricity_kwh_per_day IS NULL OR electricity_kwh_per_day >= 0),
    CONSTRAINT chk_ec_twf_heat_positive CHECK (heating_kwh_per_day IS NULL OR heating_kwh_per_day >= 0),
    CONSTRAINT chk_ec_twf_cool_positive CHECK (cooling_kwh_per_day IS NULL OR cooling_kwh_per_day >= 0),
    CONSTRAINT chk_ec_twf_equip_positive CHECK (equipment_kwh_per_day IS NULL OR equipment_kwh_per_day >= 0)
);

CREATE INDEX idx_ec_twf_zone ON employee_commuting_service.gl_ec_telework_factors(climate_zone);
CREATE INDEX idx_ec_twf_source ON employee_commuting_service.gl_ec_telework_factors(source);
CREATE INDEX idx_ec_twf_active ON employee_commuting_service.gl_ec_telework_factors(is_active);

COMMENT ON TABLE employee_commuting_service.gl_ec_telework_factors IS 'Home office energy consumption defaults by Koppen climate zone for telework emission calculations';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_factors.climate_zone IS 'Climate zone: tropical, arid, temperate_mild, temperate_cold, continental';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_factors.electricity_kwh_per_day IS 'Baseline electricity consumption per WFH day (lighting, equipment)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_factors.heating_kwh_per_day IS 'Heating energy consumption per WFH day';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_factors.cooling_kwh_per_day IS 'Cooling energy consumption per WFH day';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_factors.equipment_kwh_per_day IS 'IT equipment energy consumption per WFH day (laptop, monitor, router)';

-- ============================================================================
-- TABLE 7: gl_ec_eeio_factors
-- Description: Spend-based EEIO emission factors by NAICS code (EPA USEEIO v2)
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_eeio_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    naics_code VARCHAR(20) NOT NULL,
    description VARCHAR(500),
    co2e_per_usd NUMERIC(12,8) NOT NULL,
    base_year INT DEFAULT 2021,
    unit VARCHAR(50) DEFAULT 'kgCO2e/USD',
    source VARCHAR(100) DEFAULT 'EPA_USEEIO_v2',
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(naics_code, source, year),
    CONSTRAINT chk_ec_eeio_ef_positive CHECK (co2e_per_usd >= 0),
    CONSTRAINT chk_ec_eeio_year_valid CHECK (year >= 1990 AND year <= 2100),
    CONSTRAINT chk_ec_eeio_base_year_valid CHECK (base_year >= 1990 AND base_year <= 2100)
);

CREATE INDEX idx_ec_eeio_naics ON employee_commuting_service.gl_ec_eeio_factors(naics_code);
CREATE INDEX idx_ec_eeio_source ON employee_commuting_service.gl_ec_eeio_factors(source);
CREATE INDEX idx_ec_eeio_active ON employee_commuting_service.gl_ec_eeio_factors(is_active);

COMMENT ON TABLE employee_commuting_service.gl_ec_eeio_factors IS 'EPA USEEIO v2 spend-based emission factors for commuting categories by NAICS code';
COMMENT ON COLUMN employee_commuting_service.gl_ec_eeio_factors.co2e_per_usd IS 'Emission factor per USD spent (kgCO2e/USD, base year deflated)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_eeio_factors.base_year IS 'Base year for CPI deflation (default 2021)';

-- ============================================================================
-- TABLE 8: gl_ec_calculations (HYPERTABLE)
-- Description: Main calculation results for employee commuting emissions
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_calculations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    employee_id UUID REFERENCES employee_commuting_service.gl_ec_employees(id),
    calculation_method VARCHAR(50) NOT NULL,
    commute_mode VARCHAR(50),
    reporting_year INT NOT NULL,
    reporting_period VARCHAR(50),
    co2e_kg NUMERIC(16,8) NOT NULL,
    co2_kg NUMERIC(16,8),
    ch4_kg NUMERIC(16,8),
    n2o_kg NUMERIC(16,8),
    distance_km NUMERIC(12,4),
    working_days INT,
    wfh_fraction NUMERIC(6,4),
    data_quality_score NUMERIC(6,4),
    data_quality_tier VARCHAR(50),
    ef_source VARCHAR(100),
    gwp_version VARCHAR(20) DEFAULT 'AR5',
    provenance_hash VARCHAR(128),
    metadata JSONB DEFAULT '{}',
    is_deleted BOOLEAN DEFAULT FALSE,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(200),
    PRIMARY KEY (id, calculated_at),
    CONSTRAINT chk_ec_calc_method CHECK (calculation_method IN ('distance_based', 'average_data', 'survey_based', 'spend_based', 'supplier_specific', 'hybrid')),
    CONSTRAINT chk_ec_calc_co2e_positive CHECK (co2e_kg >= 0),
    CONSTRAINT chk_ec_calc_year_valid CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_ec_calc_wfh_range CHECK (wfh_fraction IS NULL OR (wfh_fraction >= 0 AND wfh_fraction <= 1)),
    CONSTRAINT chk_ec_calc_dqi_range CHECK (data_quality_score IS NULL OR (data_quality_score >= 1.0 AND data_quality_score <= 5.0)),
    CONSTRAINT chk_ec_calc_dqi_tier CHECK (data_quality_tier IS NULL OR data_quality_tier IN ('excellent', 'good', 'fair', 'poor', 'very_poor'))
);

-- Convert to hypertable (7-day chunks)
SELECT create_hypertable('employee_commuting_service.gl_ec_calculations', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_ec_calculations_tenant ON employee_commuting_service.gl_ec_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_ec_calculations_employee ON employee_commuting_service.gl_ec_calculations(employee_id);
CREATE INDEX idx_ec_calculations_method ON employee_commuting_service.gl_ec_calculations(calculation_method);
CREATE INDEX idx_ec_calculations_mode ON employee_commuting_service.gl_ec_calculations(commute_mode);
CREATE INDEX idx_ec_calculations_year ON employee_commuting_service.gl_ec_calculations(reporting_year);
CREATE INDEX idx_ec_calculations_period ON employee_commuting_service.gl_ec_calculations(reporting_period);
CREATE INDEX idx_ec_calculations_hash ON employee_commuting_service.gl_ec_calculations(provenance_hash);
CREATE INDEX idx_ec_calculations_deleted ON employee_commuting_service.gl_ec_calculations(is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX idx_ec_calculations_metadata ON employee_commuting_service.gl_ec_calculations USING GIN(metadata);

COMMENT ON TABLE employee_commuting_service.gl_ec_calculations IS 'Main employee commuting emission calculation results (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_calculations.calculation_method IS 'Calculation method: distance_based, average_data, survey_based, spend_based, supplier_specific, hybrid';
COMMENT ON COLUMN employee_commuting_service.gl_ec_calculations.wfh_fraction IS 'Fraction of working days spent working from home (0.0 to 1.0)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_calculations.data_quality_score IS 'Data Quality Indicator score (1.0=highest to 5.0=lowest)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_calculations.data_quality_tier IS 'Data quality tier: excellent, good, fair, poor, very_poor';

-- ============================================================================
-- TABLE 9: gl_ec_vehicle_results (HYPERTABLE)
-- Description: Personal vehicle calculation details with TTW/WTT breakdown
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_vehicle_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id UUID NOT NULL,
    vehicle_type VARCHAR(50),
    fuel_type VARCHAR(50),
    vehicle_age VARCHAR(50),
    distance_km NUMERIC(12,4),
    annual_distance_km NUMERIC(12,4),
    ttw_co2e_kg NUMERIC(16,8),
    wtt_co2e_kg NUMERIC(16,8),
    total_co2e_kg NUMERIC(16,8) NOT NULL,
    ef_used NUMERIC(12,8),
    ef_source VARCHAR(100),
    occupancy NUMERIC(6,4) DEFAULT 1.0,
    provenance_hash VARCHAR(128),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, calculated_at),
    CONSTRAINT chk_ec_vr_distance_positive CHECK (distance_km IS NULL OR distance_km >= 0),
    CONSTRAINT chk_ec_vr_annual_positive CHECK (annual_distance_km IS NULL OR annual_distance_km >= 0),
    CONSTRAINT chk_ec_vr_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_ec_vr_ttw_positive CHECK (ttw_co2e_kg IS NULL OR ttw_co2e_kg >= 0),
    CONSTRAINT chk_ec_vr_wtt_positive CHECK (wtt_co2e_kg IS NULL OR wtt_co2e_kg >= 0),
    CONSTRAINT chk_ec_vr_occupancy_positive CHECK (occupancy IS NULL OR occupancy > 0)
);

-- Convert to hypertable (7-day chunks)
SELECT create_hypertable('employee_commuting_service.gl_ec_vehicle_results', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_ec_vr_tenant ON employee_commuting_service.gl_ec_vehicle_results(tenant_id, calculated_at DESC);
CREATE INDEX idx_ec_vr_calc_id ON employee_commuting_service.gl_ec_vehicle_results(calculation_id);
CREATE INDEX idx_ec_vr_vehicle ON employee_commuting_service.gl_ec_vehicle_results(vehicle_type);
CREATE INDEX idx_ec_vr_fuel ON employee_commuting_service.gl_ec_vehicle_results(fuel_type);
CREATE INDEX idx_ec_vr_hash ON employee_commuting_service.gl_ec_vehicle_results(provenance_hash);

COMMENT ON TABLE employee_commuting_service.gl_ec_vehicle_results IS 'Personal vehicle emission calculation details with TTW/WTT breakdown (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_vehicle_results.ttw_co2e_kg IS 'Tank-to-wheel (direct) emissions in kgCO2e';
COMMENT ON COLUMN employee_commuting_service.gl_ec_vehicle_results.wtt_co2e_kg IS 'Well-to-tank (upstream fuel production) emissions in kgCO2e';
COMMENT ON COLUMN employee_commuting_service.gl_ec_vehicle_results.occupancy IS 'Vehicle occupancy for carpool/vanpool calculations (default 1.0)';

-- ============================================================================
-- TABLE 10: gl_ec_transit_results
-- Description: Public transit calculation details per trip mode
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_transit_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id UUID NOT NULL,
    transit_type VARCHAR(50),
    distance_km NUMERIC(12,4),
    annual_distance_km NUMERIC(12,4),
    co2e_kg NUMERIC(16,8) NOT NULL,
    wtt_co2e_kg NUMERIC(16,8),
    total_co2e_kg NUMERIC(16,8) NOT NULL,
    ef_used NUMERIC(12,8),
    ef_source VARCHAR(100),
    provenance_hash VARCHAR(128),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ec_tr_distance_positive CHECK (distance_km IS NULL OR distance_km >= 0),
    CONSTRAINT chk_ec_tr_annual_positive CHECK (annual_distance_km IS NULL OR annual_distance_km >= 0),
    CONSTRAINT chk_ec_tr_co2e_positive CHECK (co2e_kg >= 0),
    CONSTRAINT chk_ec_tr_total_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_ec_tr_wtt_positive CHECK (wtt_co2e_kg IS NULL OR wtt_co2e_kg >= 0)
);

CREATE INDEX idx_ec_tr_tenant ON employee_commuting_service.gl_ec_transit_results(tenant_id, calculated_at DESC);
CREATE INDEX idx_ec_tr_calc_id ON employee_commuting_service.gl_ec_transit_results(calculation_id);
CREATE INDEX idx_ec_tr_type ON employee_commuting_service.gl_ec_transit_results(transit_type);

COMMENT ON TABLE employee_commuting_service.gl_ec_transit_results IS 'Public transit emission calculation details per trip mode';
COMMENT ON COLUMN employee_commuting_service.gl_ec_transit_results.transit_type IS 'Transit mode: local_bus, coach, national_rail, light_rail, subway, high_speed_rail, ferry_foot, ferry_car, e_scooter_shared';
COMMENT ON COLUMN employee_commuting_service.gl_ec_transit_results.ef_used IS 'Emission factor applied (kgCO2e/pkm)';

-- ============================================================================
-- TABLE 11: gl_ec_telework_results
-- Description: Telework/WFH emission calculation details
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_telework_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id UUID NOT NULL,
    telework_category VARCHAR(50),
    wfh_days INT,
    annual_wfh_days INT,
    electricity_co2e_kg NUMERIC(16,8),
    heating_co2e_kg NUMERIC(16,8),
    cooling_co2e_kg NUMERIC(16,8),
    equipment_co2e_kg NUMERIC(16,8),
    total_co2e_kg NUMERIC(16,8) NOT NULL,
    grid_factor_used NUMERIC(12,8),
    climate_zone VARCHAR(50),
    country_code VARCHAR(3),
    provenance_hash VARCHAR(128),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ec_twr_wfh_positive CHECK (wfh_days IS NULL OR wfh_days >= 0),
    CONSTRAINT chk_ec_twr_annual_positive CHECK (annual_wfh_days IS NULL OR annual_wfh_days >= 0),
    CONSTRAINT chk_ec_twr_elec_positive CHECK (electricity_co2e_kg IS NULL OR electricity_co2e_kg >= 0),
    CONSTRAINT chk_ec_twr_heat_positive CHECK (heating_co2e_kg IS NULL OR heating_co2e_kg >= 0),
    CONSTRAINT chk_ec_twr_cool_positive CHECK (cooling_co2e_kg IS NULL OR cooling_co2e_kg >= 0),
    CONSTRAINT chk_ec_twr_equip_positive CHECK (equipment_co2e_kg IS NULL OR equipment_co2e_kg >= 0),
    CONSTRAINT chk_ec_twr_total_positive CHECK (total_co2e_kg >= 0)
);

CREATE INDEX idx_ec_twr_tenant ON employee_commuting_service.gl_ec_telework_results(tenant_id, calculated_at DESC);
CREATE INDEX idx_ec_twr_calc_id ON employee_commuting_service.gl_ec_telework_results(calculation_id);
CREATE INDEX idx_ec_twr_category ON employee_commuting_service.gl_ec_telework_results(telework_category);
CREATE INDEX idx_ec_twr_country ON employee_commuting_service.gl_ec_telework_results(country_code);

COMMENT ON TABLE employee_commuting_service.gl_ec_telework_results IS 'Telework/WFH emission calculation details with electricity, heating, cooling, and equipment breakdown';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_results.electricity_co2e_kg IS 'Emissions from home office electricity consumption';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_results.heating_co2e_kg IS 'Emissions from home heating during WFH';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_results.cooling_co2e_kg IS 'Emissions from home cooling during WFH';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_results.equipment_co2e_kg IS 'Emissions from IT equipment (laptop, monitor, router)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_telework_results.grid_factor_used IS 'Grid emission factor applied (kgCO2e/kWh)';

-- ============================================================================
-- TABLE 12: gl_ec_survey_results
-- Description: Survey-based calculation results with extrapolation
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_survey_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    survey_year INT NOT NULL,
    total_employees INT NOT NULL,
    respondents INT NOT NULL,
    response_rate NUMERIC(6,4),
    modal_split JSONB DEFAULT '{}',
    avg_commute_distance_km NUMERIC(12,4),
    total_co2e_kg NUMERIC(16,8),
    per_employee_co2e_kg NUMERIC(16,8),
    confidence_interval_lower NUMERIC(16,8),
    confidence_interval_upper NUMERIC(16,8),
    confidence_level NUMERIC(5,4) DEFAULT 0.9500,
    extrapolation_method VARCHAR(50),
    weighting_factors JSONB DEFAULT '{}',
    provenance_hash VARCHAR(128),
    metadata JSONB DEFAULT '{}',
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ec_sr_year_valid CHECK (survey_year >= 2000 AND survey_year <= 2100),
    CONSTRAINT chk_ec_sr_employees_positive CHECK (total_employees > 0),
    CONSTRAINT chk_ec_sr_respondents_positive CHECK (respondents > 0),
    CONSTRAINT chk_ec_sr_respondents_le_total CHECK (respondents <= total_employees),
    CONSTRAINT chk_ec_sr_rate_range CHECK (response_rate IS NULL OR (response_rate > 0 AND response_rate <= 1.0)),
    CONSTRAINT chk_ec_sr_co2e_positive CHECK (total_co2e_kg IS NULL OR total_co2e_kg >= 0),
    CONSTRAINT chk_ec_sr_per_employee_positive CHECK (per_employee_co2e_kg IS NULL OR per_employee_co2e_kg >= 0),
    CONSTRAINT chk_ec_sr_ci_valid CHECK (confidence_interval_lower IS NULL OR confidence_interval_upper IS NULL OR confidence_interval_lower <= confidence_interval_upper),
    CONSTRAINT chk_ec_sr_extrap_method CHECK (extrapolation_method IS NULL OR extrapolation_method IN ('simple_scaling', 'stratified', 'regression', 'weighted_average', 'bootstrap'))
);

CREATE INDEX idx_ec_sr_tenant ON employee_commuting_service.gl_ec_survey_results(tenant_id);
CREATE INDEX idx_ec_sr_year ON employee_commuting_service.gl_ec_survey_results(survey_year);
CREATE INDEX idx_ec_sr_method ON employee_commuting_service.gl_ec_survey_results(extrapolation_method);
CREATE INDEX idx_ec_sr_modal ON employee_commuting_service.gl_ec_survey_results USING GIN(modal_split);

COMMENT ON TABLE employee_commuting_service.gl_ec_survey_results IS 'Survey-based emission calculation results with statistical extrapolation to full workforce';
COMMENT ON COLUMN employee_commuting_service.gl_ec_survey_results.response_rate IS 'Survey response rate as fraction (0 to 1.0)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_survey_results.modal_split IS 'JSONB modal split: {car: 0.65, transit: 0.20, cycling: 0.10, walking: 0.05}';
COMMENT ON COLUMN employee_commuting_service.gl_ec_survey_results.extrapolation_method IS 'Method: simple_scaling, stratified, regression, weighted_average, bootstrap';
COMMENT ON COLUMN employee_commuting_service.gl_ec_survey_results.weighting_factors IS 'JSONB weighting factors used for stratified extrapolation by department/location';

-- ============================================================================
-- TABLE 13: gl_ec_spend_results
-- Description: Spend-based emission calculation details using EEIO factors
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_spend_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id UUID NOT NULL,
    amount NUMERIC(16,4) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    amount_usd NUMERIC(16,4) NOT NULL,
    cpi_deflator NUMERIC(10,6),
    naics_code VARCHAR(20),
    co2e_kg NUMERIC(16,8) NOT NULL,
    ef_used NUMERIC(12,8),
    ef_source VARCHAR(100),
    provenance_hash VARCHAR(128),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ec_spr_amount_positive CHECK (amount >= 0),
    CONSTRAINT chk_ec_spr_usd_positive CHECK (amount_usd >= 0),
    CONSTRAINT chk_ec_spr_co2e_positive CHECK (co2e_kg >= 0),
    CONSTRAINT chk_ec_spr_deflator_positive CHECK (cpi_deflator IS NULL OR cpi_deflator > 0)
);

CREATE INDEX idx_ec_spr_tenant ON employee_commuting_service.gl_ec_spend_results(tenant_id, calculated_at DESC);
CREATE INDEX idx_ec_spr_calc_id ON employee_commuting_service.gl_ec_spend_results(calculation_id);
CREATE INDEX idx_ec_spr_naics ON employee_commuting_service.gl_ec_spend_results(naics_code);
CREATE INDEX idx_ec_spr_currency ON employee_commuting_service.gl_ec_spend_results(currency);

COMMENT ON TABLE employee_commuting_service.gl_ec_spend_results IS 'Spend-based emission calculation details using EPA USEEIO v2 factors for commuting subsidies/allowances';
COMMENT ON COLUMN employee_commuting_service.gl_ec_spend_results.amount_usd IS 'Spend converted to USD and deflated to EEIO base year';
COMMENT ON COLUMN employee_commuting_service.gl_ec_spend_results.cpi_deflator IS 'CPI deflator applied to convert current year spend to base year USD';

-- ============================================================================
-- TABLE 14: gl_ec_compliance_checks
-- Description: Compliance check results against regulatory frameworks
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_compliance_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id UUID,
    framework VARCHAR(50) NOT NULL,
    check_type VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    score NUMERIC(5,2),
    message TEXT,
    details JSONB DEFAULT '{}',
    recommendations JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ec_cc_status CHECK (status IN ('PASS', 'FAIL', 'WARNING', 'INFO', 'NOT_APPLICABLE')),
    CONSTRAINT chk_ec_cc_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 100)),
    CONSTRAINT chk_ec_cc_framework CHECK (framework IN ('GHG_PROTOCOL', 'CSRD_ESRS_E1', 'CDP_CLIMATE', 'SBTi', 'ISO_14064', 'GRI_305', 'SECR'))
);

CREATE INDEX idx_ec_cc_tenant ON employee_commuting_service.gl_ec_compliance_checks(tenant_id);
CREATE INDEX idx_ec_cc_calc_id ON employee_commuting_service.gl_ec_compliance_checks(calculation_id);
CREATE INDEX idx_ec_cc_framework ON employee_commuting_service.gl_ec_compliance_checks(framework);
CREATE INDEX idx_ec_cc_status ON employee_commuting_service.gl_ec_compliance_checks(status);
CREATE INDEX idx_ec_cc_check_type ON employee_commuting_service.gl_ec_compliance_checks(check_type);
CREATE INDEX idx_ec_cc_details ON employee_commuting_service.gl_ec_compliance_checks USING GIN(details);

COMMENT ON TABLE employee_commuting_service.gl_ec_compliance_checks IS 'Compliance check results against GHG Protocol, CSRD, CDP, SBTi, ISO 14064, GRI, SECR frameworks';
COMMENT ON COLUMN employee_commuting_service.gl_ec_compliance_checks.status IS 'Compliance status: PASS, FAIL, WARNING, INFO, NOT_APPLICABLE';
COMMENT ON COLUMN employee_commuting_service.gl_ec_compliance_checks.framework IS 'Regulatory framework: GHG_PROTOCOL, CSRD_ESRS_E1, CDP_CLIMATE, SBTi, ISO_14064, GRI_305, SECR';
COMMENT ON COLUMN employee_commuting_service.gl_ec_compliance_checks.details IS 'JSONB object with detailed compliance findings and evidence';

-- ============================================================================
-- TABLE 15: gl_ec_aggregations (HYPERTABLE)
-- Description: Period aggregations by mode, department, location, and telework
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_aggregations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    period_type VARCHAR(20) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_co2e_kg NUMERIC(16,8) NOT NULL,
    by_mode JSONB DEFAULT '{}',
    by_department JSONB DEFAULT '{}',
    by_location JSONB DEFAULT '{}',
    by_telework JSONB DEFAULT '{}',
    employee_count INT DEFAULT 0,
    avg_per_employee NUMERIC(16,8),
    avg_commute_distance_km NUMERIC(12,4),
    data_quality_avg NUMERIC(6,4),
    provenance_hash VARCHAR(128),
    metadata JSONB DEFAULT '{}',
    aggregated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, aggregated_at),
    CONSTRAINT chk_ec_agg_period_type CHECK (period_type IN ('monthly', 'quarterly', 'annual')),
    CONSTRAINT chk_ec_agg_dates_valid CHECK (period_end >= period_start),
    CONSTRAINT chk_ec_agg_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_ec_agg_count_positive CHECK (employee_count >= 0),
    CONSTRAINT chk_ec_agg_avg_positive CHECK (avg_per_employee IS NULL OR avg_per_employee >= 0),
    CONSTRAINT chk_ec_agg_dqi_range CHECK (data_quality_avg IS NULL OR (data_quality_avg >= 1.0 AND data_quality_avg <= 5.0))
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('employee_commuting_service.gl_ec_aggregations', 'aggregated_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_ec_agg_tenant ON employee_commuting_service.gl_ec_aggregations(tenant_id, aggregated_at DESC);
CREATE INDEX idx_ec_agg_period_type ON employee_commuting_service.gl_ec_aggregations(period_type);
CREATE INDEX idx_ec_agg_period_range ON employee_commuting_service.gl_ec_aggregations(period_start, period_end);
CREATE INDEX idx_ec_agg_by_mode ON employee_commuting_service.gl_ec_aggregations USING GIN(by_mode);
CREATE INDEX idx_ec_agg_by_dept ON employee_commuting_service.gl_ec_aggregations USING GIN(by_department);
CREATE INDEX idx_ec_agg_by_location ON employee_commuting_service.gl_ec_aggregations USING GIN(by_location);
CREATE INDEX idx_ec_agg_by_telework ON employee_commuting_service.gl_ec_aggregations USING GIN(by_telework);

COMMENT ON TABLE employee_commuting_service.gl_ec_aggregations IS 'Period aggregations of employee commuting emissions (HYPERTABLE, 30-day chunks)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_aggregations.period_type IS 'Aggregation period: monthly, quarterly, annual';
COMMENT ON COLUMN employee_commuting_service.gl_ec_aggregations.by_mode IS 'JSONB breakdown of CO2e by commute mode (car, transit, cycling, walking, telework)';
COMMENT ON COLUMN employee_commuting_service.gl_ec_aggregations.by_department IS 'JSONB breakdown of CO2e by department';
COMMENT ON COLUMN employee_commuting_service.gl_ec_aggregations.by_location IS 'JSONB breakdown of CO2e by office location';
COMMENT ON COLUMN employee_commuting_service.gl_ec_aggregations.by_telework IS 'JSONB breakdown of CO2e by telework category (office_based, hybrid, fully_remote)';

-- ============================================================================
-- TABLE 16: gl_ec_provenance
-- Description: Provenance tracking with SHA-256 hash chains
-- ============================================================================

CREATE TABLE employee_commuting_service.gl_ec_provenance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    chain_id VARCHAR(128) NOT NULL,
    calculation_id UUID,
    stage VARCHAR(50) NOT NULL,
    input_hash VARCHAR(128) NOT NULL,
    output_hash VARCHAR(128) NOT NULL,
    chain_hash VARCHAR(128) NOT NULL,
    engine_id VARCHAR(100),
    engine_version VARCHAR(20),
    stage_index INT NOT NULL,
    metadata JSONB DEFAULT '{}',
    duration_ms NUMERIC(12,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ec_prov_stage CHECK (stage IN ('INTAKE', 'PROFILE_RESOLUTION', 'EF_LOOKUP', 'DISTANCE_CALC', 'VEHICLE_CALC', 'TRANSIT_CALC', 'TELEWORK_CALC', 'SURVEY_EXTRAP', 'SPEND_CALC', 'AGGREGATION', 'VALIDATION', 'COMPLIANCE')),
    CONSTRAINT chk_ec_prov_index_positive CHECK (stage_index >= 0),
    CONSTRAINT chk_ec_prov_duration_positive CHECK (duration_ms IS NULL OR duration_ms >= 0)
);

CREATE INDEX idx_ec_prov_tenant ON employee_commuting_service.gl_ec_provenance(tenant_id);
CREATE INDEX idx_ec_prov_chain_id ON employee_commuting_service.gl_ec_provenance(chain_id);
CREATE INDEX idx_ec_prov_calc_id ON employee_commuting_service.gl_ec_provenance(calculation_id);
CREATE INDEX idx_ec_prov_calc_stage ON employee_commuting_service.gl_ec_provenance(calculation_id, stage_index);
CREATE INDEX idx_ec_prov_chain_hash ON employee_commuting_service.gl_ec_provenance(chain_hash);
CREATE INDEX idx_ec_prov_engine ON employee_commuting_service.gl_ec_provenance(engine_id);
CREATE INDEX idx_ec_prov_created ON employee_commuting_service.gl_ec_provenance(created_at DESC);

COMMENT ON TABLE employee_commuting_service.gl_ec_provenance IS 'Provenance tracking for employee commuting emission calculations with SHA-256 hash chains';
COMMENT ON COLUMN employee_commuting_service.gl_ec_provenance.stage IS 'Processing stage: INTAKE, PROFILE_RESOLUTION, EF_LOOKUP, DISTANCE_CALC, VEHICLE_CALC, TRANSIT_CALC, TELEWORK_CALC, SURVEY_EXTRAP, SPEND_CALC, AGGREGATION, VALIDATION, COMPLIANCE';
COMMENT ON COLUMN employee_commuting_service.gl_ec_provenance.chain_hash IS 'SHA-256 hash chaining input_hash + output_hash + previous chain_hash';
COMMENT ON COLUMN employee_commuting_service.gl_ec_provenance.engine_id IS 'Engine that produced this provenance record (e.g., VehicleCalculatorEngine)';

-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Continuous Aggregate 1: Hourly Emissions
CREATE MATERIALIZED VIEW employee_commuting_service.gl_ec_hourly_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', calculated_at) AS bucket,
    tenant_id,
    calculation_method,
    commute_mode,
    COUNT(*) AS calc_count,
    SUM(co2e_kg) AS total_co2e_kg,
    AVG(data_quality_score) AS avg_dqi_score
FROM employee_commuting_service.gl_ec_calculations
GROUP BY bucket, tenant_id, calculation_method, commute_mode
WITH NO DATA;

-- Refresh policy for hourly emissions (refresh every 1 hour, lag 2 hours)
SELECT add_continuous_aggregate_policy('employee_commuting_service.gl_ec_hourly_emissions',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '2 hours',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW employee_commuting_service.gl_ec_hourly_emissions IS 'Hourly aggregation of employee commuting emissions by calculation method and commute mode';

-- Continuous Aggregate 2: Daily Emissions
CREATE MATERIALIZED VIEW employee_commuting_service.gl_ec_daily_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    tenant_id,
    calculation_method,
    commute_mode,
    COUNT(*) AS calc_count,
    SUM(co2e_kg) AS total_co2e_kg,
    AVG(data_quality_score) AS avg_dqi_score
FROM employee_commuting_service.gl_ec_calculations
GROUP BY bucket, tenant_id, calculation_method, commute_mode
WITH NO DATA;

-- Refresh policy for daily emissions (refresh every 6 hours, lag 12 hours)
SELECT add_continuous_aggregate_policy('employee_commuting_service.gl_ec_daily_emissions',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW employee_commuting_service.gl_ec_daily_emissions IS 'Daily aggregation of employee commuting emissions with method and mode breakdown';

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on operational tables with tenant_id
ALTER TABLE employee_commuting_service.gl_ec_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE employee_commuting_service.gl_ec_vehicle_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE employee_commuting_service.gl_ec_transit_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE employee_commuting_service.gl_ec_telework_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE employee_commuting_service.gl_ec_spend_results ENABLE ROW LEVEL SECURITY;

-- RLS Policy: gl_ec_calculations
CREATE POLICY ec_calculations_tenant_isolation ON employee_commuting_service.gl_ec_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_ec_vehicle_results
CREATE POLICY ec_vehicle_results_tenant_isolation ON employee_commuting_service.gl_ec_vehicle_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_ec_transit_results
CREATE POLICY ec_transit_results_tenant_isolation ON employee_commuting_service.gl_ec_transit_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_ec_telework_results
CREATE POLICY ec_telework_results_tenant_isolation ON employee_commuting_service.gl_ec_telework_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_ec_spend_results
CREATE POLICY ec_spend_results_tenant_isolation ON employee_commuting_service.gl_ec_spend_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- ============================================================================
-- SEED DATA: VEHICLE EMISSION FACTORS (DEFRA 2024 - 22 factors)
-- ============================================================================

INSERT INTO employee_commuting_service.gl_ec_vehicle_emission_factors
(vehicle_type, fuel_type, vehicle_age, co2e_per_km, co2_per_km, ch4_per_km, n2o_per_km, wtt_factor, unit, source, year, is_active) VALUES
-- Small cars
('small_car', 'petrol',  'mid_4_7yr', 0.14890000, 0.14710000, 0.00032000, 0.00148000, 0.02330000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('small_car', 'diesel',  'mid_4_7yr', 0.13920000, 0.13780000, 0.00003000, 0.00137000, 0.02070000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('small_car', 'hybrid',  'mid_4_7yr', 0.10410000, 0.10290000, 0.00024000, 0.00096000, 0.01730000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('small_car', 'ev',      'mid_4_7yr', 0.04600000, 0.04540000, 0.00000000, 0.00060000, 0.01330000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- Medium cars
('medium_car', 'petrol',  'mid_4_7yr', 0.18770000, 0.18550000, 0.00040000, 0.00180000, 0.02940000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_car', 'diesel',  'mid_4_7yr', 0.16610000, 0.16430000, 0.00004000, 0.00176000, 0.02470000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_car', 'hybrid',  'mid_4_7yr', 0.11590000, 0.11450000, 0.00028000, 0.00112000, 0.01820000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_car', 'phev',    'mid_4_7yr', 0.06920000, 0.06840000, 0.00016000, 0.00064000, 0.01440000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_car', 'ev',      'mid_4_7yr', 0.05310000, 0.05240000, 0.00000000, 0.00070000, 0.01530000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- Large cars / SUVs
('large_car', 'petrol',  'mid_4_7yr', 0.27870000, 0.27540000, 0.00060000, 0.00270000, 0.04370000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('large_car', 'diesel',  'mid_4_7yr', 0.20870000, 0.20640000, 0.00005000, 0.00225000, 0.03100000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('large_car', 'hybrid',  'mid_4_7yr', 0.15420000, 0.15240000, 0.00036000, 0.00144000, 0.02420000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('large_car', 'phev',    'mid_4_7yr', 0.09210000, 0.09100000, 0.00022000, 0.00088000, 0.01920000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('large_car', 'ev',      'mid_4_7yr', 0.06130000, 0.06050000, 0.00000000, 0.00080000, 0.01770000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- Average car (fleet average)
('average_car', 'mixed', 'mid_4_7yr', 0.17140000, 0.16940000, 0.00036000, 0.00164000, 0.02690000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- Alternative fuels
('medium_car', 'lpg',    'mid_4_7yr', 0.17240000, 0.17040000, 0.00038000, 0.00162000, 0.01120000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('medium_car', 'cng',    'mid_4_7yr', 0.15830000, 0.15640000, 0.00040000, 0.00150000, 0.02540000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- Motorcycle
('motorcycle', 'petrol_small',  'mid_4_7yr', 0.08310000, 0.08210000, 0.00020000, 0.00080000, 0.01530000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('motorcycle', 'petrol_medium', 'mid_4_7yr', 0.10100000, 0.09980000, 0.00024000, 0.00096000, 0.01860000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('motorcycle', 'petrol_large',  'mid_4_7yr', 0.13240000, 0.13090000, 0.00030000, 0.00120000, 0.02440000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),

-- E-bike / E-scooter (lifecycle-based)
('e_bicycle', 'electric', 'mid_4_7yr', 0.00530000, 0.00520000, 0.00000000, 0.00010000, 0.00150000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE),
('e_scooter', 'electric', 'mid_4_7yr', 0.03550000, 0.03500000, 0.00000000, 0.00050000, 0.01020000, 'kgCO2e/km', 'DEFRA_2024', 2024, TRUE);

-- ============================================================================
-- SEED DATA: TRANSIT EMISSION FACTORS (DEFRA 2024 - 9 factors)
-- ============================================================================

INSERT INTO employee_commuting_service.gl_ec_transit_emission_factors
(transit_type, region, co2e_per_pkm, wtt_factor, unit, source, year, is_active) VALUES
-- Bus
('local_bus',        'global', 0.10312000, 0.01670000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('coach',            'global', 0.02732000, 0.00447000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Rail
('national_rail',    'global', 0.03549000, 0.00790000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('light_rail',       'global', 0.02910000, 0.00670000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('subway',           'global', 0.02781000, 0.00640000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('high_speed_rail',  'global', 0.00600000, 0.00140000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Ferry
('ferry_foot',       'global', 0.01870000, 0.00370000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('ferry_car',        'global', 0.12952000, 0.02560000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Shared micromobility (lifecycle-based)
('e_scooter_shared', 'global', 0.06100000, 0.01750000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE);

-- ============================================================================
-- SEED DATA: GRID EMISSION FACTORS (IEA 2024 - 20 countries + 8 US eGRID)
-- ============================================================================

INSERT INTO employee_commuting_service.gl_ec_grid_emission_factors
(country_code, region, co2e_per_kwh, co2_per_kwh, unit, source, year, is_active) VALUES
-- Major economies (IEA 2024 national averages)
('US',  NULL,    0.37890000, 0.37100000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('GB',  NULL,    0.20700000, 0.20200000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('DE',  NULL,    0.35000000, 0.34200000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('FR',  NULL,    0.05200000, 0.05000000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('JP',  NULL,    0.45700000, 0.44800000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('CN',  NULL,    0.55500000, 0.54300000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('IN',  NULL,    0.70800000, 0.69300000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('CA',  NULL,    0.12000000, 0.11700000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('AU',  NULL,    0.65600000, 0.64200000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('BR',  NULL,    0.07400000, 0.07200000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('KR',  NULL,    0.41500000, 0.40600000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('IT',  NULL,    0.25600000, 0.25000000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('ES',  NULL,    0.16200000, 0.15800000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('NL',  NULL,    0.32800000, 0.32100000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('SE',  NULL,    0.01200000, 0.01100000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('NO',  NULL,    0.00800000, 0.00700000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('SG',  NULL,    0.40800000, 0.39900000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('AE',  NULL,    0.50200000, 0.49100000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('ZA',  NULL,    0.92800000, 0.90800000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),
('MX',  NULL,    0.42300000, 0.41400000, 'kgCO2e/kWh', 'IEA_2024', 2024, TRUE),

-- US eGRID subregional factors (EPA eGRID 2024)
('US',  'CAMX',  0.22100000, 0.21600000, 'kgCO2e/kWh', 'EPA_eGRID_2024', 2024, TRUE),
('US',  'ERCT',  0.37300000, 0.36500000, 'kgCO2e/kWh', 'EPA_eGRID_2024', 2024, TRUE),
('US',  'MROE',  0.54200000, 0.53000000, 'kgCO2e/kWh', 'EPA_eGRID_2024', 2024, TRUE),
('US',  'NEWE',  0.22600000, 0.22100000, 'kgCO2e/kWh', 'EPA_eGRID_2024', 2024, TRUE),
('US',  'NWPP',  0.27800000, 0.27200000, 'kgCO2e/kWh', 'EPA_eGRID_2024', 2024, TRUE),
('US',  'RFCE',  0.30200000, 0.29600000, 'kgCO2e/kWh', 'EPA_eGRID_2024', 2024, TRUE),
('US',  'RMPA',  0.53600000, 0.52400000, 'kgCO2e/kWh', 'EPA_eGRID_2024', 2024, TRUE),
('US',  'SRSO',  0.38900000, 0.38100000, 'kgCO2e/kWh', 'EPA_eGRID_2024', 2024, TRUE);

-- ============================================================================
-- SEED DATA: TELEWORK FACTORS (5 climate zones)
-- ============================================================================

INSERT INTO employee_commuting_service.gl_ec_telework_factors
(climate_zone, electricity_kwh_per_day, heating_kwh_per_day, cooling_kwh_per_day, equipment_kwh_per_day, unit, source, is_active) VALUES
('tropical',        1.20000000, 0.00000000, 3.50000000, 0.30000000, 'kWh/day', 'IEA_2024', TRUE),
('arid',            1.20000000, 0.80000000, 4.20000000, 0.30000000, 'kWh/day', 'IEA_2024', TRUE),
('temperate_mild',  1.20000000, 2.80000000, 1.50000000, 0.30000000, 'kWh/day', 'IEA_2024', TRUE),
('temperate_cold',  1.20000000, 5.20000000, 0.50000000, 0.30000000, 'kWh/day', 'IEA_2024', TRUE),
('continental',     1.20000000, 7.80000000, 0.80000000, 0.30000000, 'kWh/day', 'IEA_2024', TRUE);

-- ============================================================================
-- SEED DATA: EEIO SPEND-BASED FACTORS (10 NAICS codes for commuting)
-- ============================================================================

INSERT INTO employee_commuting_service.gl_ec_eeio_factors
(naics_code, description, co2e_per_usd, base_year, unit, source, year, is_active) VALUES
('485000', 'Transit and ground passenger transportation', 0.35100000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('485110', 'Mixed mode transit systems',                  0.33200000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('485210', 'Interurban and rural bus transportation',     0.28900000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('485310', 'Taxi and ridesharing services',               0.30200000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('485999', 'All other transit and ground passenger',      0.31500000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('482000', 'Rail transportation',                         0.22300000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('447000', 'Gasoline stations',                           0.85400000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('811100', 'Automotive repair and maintenance',           0.21800000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('441000', 'Motor vehicle and parts dealers',             0.15300000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE),
('532100', 'Automotive equipment rental and leasing',     0.25600000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', 2024, TRUE);

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
    'GL-MRV-S3-007',
    'Employee Commuting Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-020: Scope 3 Category 7 - Employee Commuting. Calculates emissions from employee commuting including personal vehicles (TTW+WTT), public transit, cycling, walking, and telework/WFH. Supports distance-based (DEFRA 2024), survey-based with statistical extrapolation, average-data, spend-based (EPA USEEIO v2), and hybrid calculation methods. Includes 22 vehicle EFs, 9 transit EFs, 28 grid EFs (20 countries + 8 eGRID), 5 telework climate zone factors, and 10 EEIO spend factors. Features modal split profiling, WFH fraction tracking, and multi-framework compliance.',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 7,
        'category_name', 'Employee Commuting',
        'calculation_methods', jsonb_build_array('distance_based', 'average_data', 'survey_based', 'spend_based', 'supplier_specific', 'hybrid'),
        'commute_modes', jsonb_build_array('car_petrol', 'car_diesel', 'car_hybrid', 'car_phev', 'car_ev', 'motorcycle', 'bus', 'coach', 'rail', 'light_rail', 'subway', 'ferry', 'bicycle', 'e_bicycle', 'e_scooter', 'walking', 'carpool', 'vanpool'),
        'telework_categories', jsonb_build_array('office_based', 'hybrid_1day', 'hybrid_2day', 'hybrid_3day', 'hybrid_4day', 'fully_remote', 'field_based'),
        'vehicle_types', jsonb_build_array('small_car', 'medium_car', 'large_car', 'average_car', 'motorcycle', 'e_bicycle', 'e_scooter'),
        'fuel_types', jsonb_build_array('petrol', 'diesel', 'hybrid', 'phev', 'ev', 'lpg', 'cng', 'electric'),
        'transit_types', jsonb_build_array('local_bus', 'coach', 'national_rail', 'light_rail', 'subway', 'high_speed_rail', 'ferry_foot', 'ferry_car', 'e_scooter_shared'),
        'frameworks', jsonb_build_array('GHG Protocol Scope 3', 'CSRD ESRS E1', 'CDP Climate', 'SBTi', 'ISO 14064-1', 'GRI 305', 'SECR'),
        'vehicle_ef_count', 22,
        'transit_ef_count', 9,
        'grid_ef_count', 28,
        'telework_factor_count', 5,
        'eeio_factor_count', 10,
        'supports_wtt_emissions', true,
        'supports_telework_emissions', true,
        'supports_survey_extrapolation', true,
        'supports_modal_split', true,
        'supports_cpi_deflation', true,
        'supports_wfh_fraction', true,
        'default_ef_source', 'DEFRA_2024',
        'default_grid_source', 'IEA_2024',
        'default_gwp', 'AR5',
        'schema', 'employee_commuting_service',
        'table_prefix', 'gl_ec_',
        'hypertables', jsonb_build_array('gl_ec_calculations', 'gl_ec_vehicle_results', 'gl_ec_aggregations'),
        'continuous_aggregates', jsonb_build_array('gl_ec_hourly_emissions', 'gl_ec_daily_emissions'),
        'migration_version', 'V071'
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

COMMENT ON SCHEMA employee_commuting_service IS 'Updated: AGENT-MRV-020 complete with 16 tables, 3 hypertables, 2 continuous aggregates, RLS policies, 75+ seed records';

-- ============================================================================
-- END OF MIGRATION V071
-- ============================================================================
-- Total Lines: ~1050
-- Total Tables: 16
-- Total Hypertables: 3 (calculations, vehicle_results, aggregations)
-- Total Continuous Aggregates: 2 (hourly_emissions, daily_emissions)
-- Total RLS Policies: 5 (calculations, vehicle_results, transit_results, telework_results, spend_results)
-- Total Seed Records: 75
--   Vehicle Emission Factors: 22 (small/medium/large/average/motorcycle/e-bike/e-scooter)
--   Transit Emission Factors: 9 (bus/coach/rail/light_rail/subway/high_speed/ferry/e-scooter)
--   Grid Emission Factors: 28 (20 countries + 8 US eGRID subregions)
--   Telework Factors: 5 (tropical/arid/temperate_mild/temperate_cold/continental)
--   EEIO Factors: 10 (NAICS codes for commuting categories)
--   Agent Registry: 1
-- Total Indexes: 66
-- Total Constraints: 58
-- ============================================================================

-- =====================================================================================
-- Migration: V075__use_of_sold_products_service.sql
-- Description: AGENT-MRV-024 Use of Sold Products (Scope 3 Category 11)
-- Agent: GL-MRV-S3-011
-- Framework: GHG Protocol Scope 3 Standard Ch 6, DEFRA 2024, EPA, IPCC AR5/AR6, IEA
-- Created: 2026-02-28
-- =====================================================================================
-- Schema: use_of_sold_products_service
-- Tables: 21 (9 reference + 7 result + 5 operational)
-- Hypertables: 3 (calculations, aggregations, compliance_results)
-- Continuous Aggregates: 2 (daily by emission_type, monthly by category)
-- RLS: 10 policies
-- Indexes: 90+
-- Constraints: 90+
-- Seed Data: 24 profiles, 15 fuel EFs, 16 grid EFs, 10 refrigerants, 5 adjustments,
--            6 degradation rates, 4 steam factors, 5 chemicals, 11 CPI deflators
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS use_of_sold_products_service;

COMMENT ON SCHEMA use_of_sold_products_service IS 'AGENT-MRV-024: Use of Sold Products - Scope 3 Category 11 emission calculations (direct/indirect/fuels-feedstocks use-phase emissions)';

-- =====================================================================================
-- TABLE 1: gl_usp_product_energy_profiles (REFERENCE)
-- Description: Default product energy profiles with lifetime and consumption data
-- 24 products across 10 categories
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_product_energy_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_type VARCHAR(100) NOT NULL,
    product_category VARCHAR(50) NOT NULL,
    default_lifetime_years DECIMAL(10,2) NOT NULL,
    annual_energy_consumption DECIMAL(20,8) NOT NULL,
    energy_unit VARCHAR(30) NOT NULL,
    energy_source VARCHAR(50) NOT NULL,
    power_rating_watts DECIMAL(20,8),
    usage_hours_per_year DECIMAL(10,2),
    standby_watts DECIMAL(10,4),
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(product_type, product_category),
    CONSTRAINT chk_usp_profile_lifetime_positive CHECK (default_lifetime_years > 0),
    CONSTRAINT chk_usp_profile_consumption_positive CHECK (annual_energy_consumption >= 0),
    CONSTRAINT chk_usp_profile_power_positive CHECK (power_rating_watts IS NULL OR power_rating_watts >= 0),
    CONSTRAINT chk_usp_profile_usage_positive CHECK (usage_hours_per_year IS NULL OR (usage_hours_per_year >= 0 AND usage_hours_per_year <= 8784)),
    CONSTRAINT chk_usp_profile_standby_positive CHECK (standby_watts IS NULL OR standby_watts >= 0),
    CONSTRAINT chk_usp_profile_category CHECK (product_category IN (
        'VEHICLES', 'APPLIANCES', 'HVAC', 'LIGHTING', 'IT_EQUIPMENT',
        'INDUSTRIAL_EQUIPMENT', 'FUELS_FEEDSTOCKS', 'BUILDING_PRODUCTS',
        'CONSUMER_PRODUCTS', 'MEDICAL_DEVICES'
    )),
    CONSTRAINT chk_usp_profile_energy_source CHECK (energy_source IN (
        'electricity', 'gasoline', 'diesel', 'natural_gas', 'lpg',
        'fuel_oil', 'propane', 'kerosene', 'direct_chemical', 'mixed'
    )),
    CONSTRAINT chk_usp_profile_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_usp_profiles_type ON use_of_sold_products_service.gl_usp_product_energy_profiles(product_type);
CREATE INDEX idx_usp_profiles_category ON use_of_sold_products_service.gl_usp_product_energy_profiles(product_category);
CREATE INDEX idx_usp_profiles_source ON use_of_sold_products_service.gl_usp_product_energy_profiles(energy_source);
CREATE INDEX idx_usp_profiles_active ON use_of_sold_products_service.gl_usp_product_energy_profiles(is_active);
CREATE INDEX idx_usp_profiles_cat_active ON use_of_sold_products_service.gl_usp_product_energy_profiles(product_category, is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_product_energy_profiles IS 'Default energy profiles for 24 product types across 10 categories with lifetime and annual consumption data';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_product_energy_profiles.product_type IS 'Specific product type (e.g., sedan_car, refrigerator, split_ac)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_product_energy_profiles.product_category IS 'Product category: VEHICLES, APPLIANCES, HVAC, LIGHTING, IT_EQUIPMENT, INDUSTRIAL_EQUIPMENT, FUELS_FEEDSTOCKS, BUILDING_PRODUCTS, CONSUMER_PRODUCTS, MEDICAL_DEVICES';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_product_energy_profiles.default_lifetime_years IS 'Default product useful life in years';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_product_energy_profiles.annual_energy_consumption IS 'Annual energy consumption per unit (in energy_unit)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_product_energy_profiles.energy_unit IS 'Unit of energy consumption (kWh, litres, m3, kg)';

-- =====================================================================================
-- TABLE 2: gl_usp_fuel_emission_factors (REFERENCE)
-- Description: Combustion emission factors for 15 fuel types
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_fuel_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fuel_type VARCHAR(50) NOT NULL,
    fuel_name VARCHAR(200) NOT NULL,
    ef_value DECIMAL(20,8) NOT NULL,
    ef_unit VARCHAR(50) NOT NULL,
    ncv DECIMAL(20,8),
    ncv_unit VARCHAR(50),
    co2_fraction DECIMAL(10,8),
    ch4_fraction DECIMAL(10,8),
    n2o_fraction DECIMAL(10,8),
    density DECIMAL(10,6),
    density_unit VARCHAR(30),
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(fuel_type, source),
    CONSTRAINT chk_usp_fuel_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_usp_fuel_ncv_positive CHECK (ncv IS NULL OR ncv >= 0),
    CONSTRAINT chk_usp_fuel_co2_frac CHECK (co2_fraction IS NULL OR (co2_fraction >= 0 AND co2_fraction <= 1)),
    CONSTRAINT chk_usp_fuel_ch4_frac CHECK (ch4_fraction IS NULL OR (ch4_fraction >= 0 AND ch4_fraction <= 1)),
    CONSTRAINT chk_usp_fuel_n2o_frac CHECK (n2o_fraction IS NULL OR (n2o_fraction >= 0 AND n2o_fraction <= 1)),
    CONSTRAINT chk_usp_fuel_density_positive CHECK (density IS NULL OR density > 0),
    CONSTRAINT chk_usp_fuel_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_usp_fuel_ef_type ON use_of_sold_products_service.gl_usp_fuel_emission_factors(fuel_type);
CREATE INDEX idx_usp_fuel_ef_source ON use_of_sold_products_service.gl_usp_fuel_emission_factors(source);
CREATE INDEX idx_usp_fuel_ef_active ON use_of_sold_products_service.gl_usp_fuel_emission_factors(is_active);
CREATE INDEX idx_usp_fuel_ef_type_active ON use_of_sold_products_service.gl_usp_fuel_emission_factors(fuel_type, is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_fuel_emission_factors IS 'Combustion emission factors for 15 fuel types with NCV and gas-specific fractions';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_fuel_emission_factors.fuel_type IS 'Fuel type identifier (gasoline, diesel, natural_gas, lpg, etc.)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_fuel_emission_factors.ef_value IS 'Emission factor value (kgCO2e per unit)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_fuel_emission_factors.ncv IS 'Net calorific value (MJ per unit)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_fuel_emission_factors.co2_fraction IS 'CO2 fraction of total CO2e (0-1)';

-- =====================================================================================
-- TABLE 3: gl_usp_grid_emission_factors (REFERENCE)
-- Description: Electricity grid emission factors for 16 regions
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_grid_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_code VARCHAR(50) NOT NULL,
    region_name VARCHAR(200) NOT NULL,
    ef_value DECIMAL(20,8) NOT NULL,
    ef_unit VARCHAR(50) DEFAULT 'kgCO2e/kWh',
    td_loss_factor DECIMAL(10,6) DEFAULT 0.05,
    renewable_fraction DECIMAL(10,6),
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(region_code, source),
    CONSTRAINT chk_usp_grid_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_usp_grid_td_loss CHECK (td_loss_factor IS NULL OR (td_loss_factor >= 0 AND td_loss_factor <= 0.5)),
    CONSTRAINT chk_usp_grid_renewable CHECK (renewable_fraction IS NULL OR (renewable_fraction >= 0 AND renewable_fraction <= 1)),
    CONSTRAINT chk_usp_grid_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_usp_grid_ef_region ON use_of_sold_products_service.gl_usp_grid_emission_factors(region_code);
CREATE INDEX idx_usp_grid_ef_source ON use_of_sold_products_service.gl_usp_grid_emission_factors(source);
CREATE INDEX idx_usp_grid_ef_active ON use_of_sold_products_service.gl_usp_grid_emission_factors(is_active);
CREATE INDEX idx_usp_grid_ef_region_active ON use_of_sold_products_service.gl_usp_grid_emission_factors(region_code, is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_grid_emission_factors IS 'Electricity grid emission factors for 16 regions with T&D loss and renewable fractions';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_grid_emission_factors.region_code IS 'Grid region code (US_AVERAGE, EU_AVERAGE, UK_GRID, CN_GRID, etc.)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_grid_emission_factors.ef_value IS 'Grid emission factor (kgCO2e/kWh)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_grid_emission_factors.td_loss_factor IS 'Transmission and distribution loss factor (fraction)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_grid_emission_factors.renewable_fraction IS 'Fraction of grid from renewable sources';

-- =====================================================================================
-- TABLE 4: gl_usp_refrigerant_gwps (REFERENCE)
-- Description: Refrigerant GWP values for 10 common refrigerants (AR5 + AR6)
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_refrigerant_gwps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    refrigerant_type VARCHAR(50) NOT NULL UNIQUE,
    refrigerant_name VARCHAR(200) NOT NULL,
    chemical_formula VARCHAR(100),
    gwp_ar5 DECIMAL(20,4) NOT NULL,
    gwp_ar6 DECIMAL(20,4) NOT NULL,
    typical_charge_kg DECIMAL(10,4),
    typical_leak_rate DECIMAL(10,6),
    ozone_depleting BOOLEAN DEFAULT FALSE,
    phase_down_status VARCHAR(50),
    source VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_usp_refrig_gwp_ar5_positive CHECK (gwp_ar5 >= 0),
    CONSTRAINT chk_usp_refrig_gwp_ar6_positive CHECK (gwp_ar6 >= 0),
    CONSTRAINT chk_usp_refrig_charge_positive CHECK (typical_charge_kg IS NULL OR typical_charge_kg > 0),
    CONSTRAINT chk_usp_refrig_leak_rate CHECK (typical_leak_rate IS NULL OR (typical_leak_rate >= 0 AND typical_leak_rate <= 1))
);

CREATE INDEX idx_usp_refrig_type ON use_of_sold_products_service.gl_usp_refrigerant_gwps(refrigerant_type);
CREATE INDEX idx_usp_refrig_active ON use_of_sold_products_service.gl_usp_refrigerant_gwps(is_active);
CREATE INDEX idx_usp_refrig_odp ON use_of_sold_products_service.gl_usp_refrigerant_gwps(ozone_depleting);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_refrigerant_gwps IS 'GWP values for 10 common refrigerants with AR5 and AR6 values, charge sizes, and leak rates';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_refrigerant_gwps.refrigerant_type IS 'Refrigerant identifier (R-134a, R-410A, R-32, etc.)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_refrigerant_gwps.gwp_ar5 IS 'Global warming potential per IPCC AR5 (100-year)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_refrigerant_gwps.gwp_ar6 IS 'Global warming potential per IPCC AR6 (100-year)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_refrigerant_gwps.typical_charge_kg IS 'Typical refrigerant charge per equipment unit (kg)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_refrigerant_gwps.typical_leak_rate IS 'Typical annual leak rate as fraction (e.g. 0.05 = 5%)';

-- =====================================================================================
-- TABLE 5: gl_usp_product_lifetimes (REFERENCE)
-- Description: Product lifetime estimates by category with adjustment factors
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_product_lifetimes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(100),
    min_lifetime_years DECIMAL(10,2) NOT NULL,
    typical_lifetime_years DECIMAL(10,2) NOT NULL,
    max_lifetime_years DECIMAL(10,2) NOT NULL,
    adjustment_factor DECIMAL(10,4) DEFAULT 1.0,
    adjustment_reason VARCHAR(500),
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(product_category, subcategory),
    CONSTRAINT chk_usp_lifetime_min_positive CHECK (min_lifetime_years > 0),
    CONSTRAINT chk_usp_lifetime_typical_positive CHECK (typical_lifetime_years > 0),
    CONSTRAINT chk_usp_lifetime_max_positive CHECK (max_lifetime_years > 0),
    CONSTRAINT chk_usp_lifetime_order CHECK (min_lifetime_years <= typical_lifetime_years AND typical_lifetime_years <= max_lifetime_years),
    CONSTRAINT chk_usp_lifetime_adjustment CHECK (adjustment_factor > 0 AND adjustment_factor <= 3.0),
    CONSTRAINT chk_usp_lifetime_category CHECK (product_category IN (
        'VEHICLES', 'APPLIANCES', 'HVAC', 'LIGHTING', 'IT_EQUIPMENT',
        'INDUSTRIAL_EQUIPMENT', 'FUELS_FEEDSTOCKS', 'BUILDING_PRODUCTS',
        'CONSUMER_PRODUCTS', 'MEDICAL_DEVICES'
    )),
    CONSTRAINT chk_usp_lifetime_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_usp_lifetimes_category ON use_of_sold_products_service.gl_usp_product_lifetimes(product_category);
CREATE INDEX idx_usp_lifetimes_subcat ON use_of_sold_products_service.gl_usp_product_lifetimes(subcategory);
CREATE INDEX idx_usp_lifetimes_active ON use_of_sold_products_service.gl_usp_product_lifetimes(is_active);
CREATE INDEX idx_usp_lifetimes_cat_active ON use_of_sold_products_service.gl_usp_product_lifetimes(product_category, is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_product_lifetimes IS 'Product lifetime estimates by category with min/typical/max and adjustment factors';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_product_lifetimes.typical_lifetime_years IS 'Typical expected lifetime in years (used as default)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_product_lifetimes.adjustment_factor IS 'Lifetime adjustment factor (1.0 = no adjustment, >1.0 = longer, <1.0 = shorter)';

-- =====================================================================================
-- TABLE 6: gl_usp_energy_degradation (REFERENCE)
-- Description: Energy efficiency degradation rates by product category
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_energy_degradation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(100),
    annual_degradation_rate DECIMAL(10,6) NOT NULL,
    max_cumulative_degradation DECIMAL(10,6) DEFAULT 0.50,
    degradation_type VARCHAR(50) DEFAULT 'linear',
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(product_category, subcategory),
    CONSTRAINT chk_usp_degrad_rate CHECK (annual_degradation_rate >= 0 AND annual_degradation_rate <= 0.20),
    CONSTRAINT chk_usp_degrad_max CHECK (max_cumulative_degradation >= 0 AND max_cumulative_degradation <= 1.0),
    CONSTRAINT chk_usp_degrad_type CHECK (degradation_type IN ('linear', 'exponential', 'step')),
    CONSTRAINT chk_usp_degrad_category CHECK (product_category IN (
        'VEHICLES', 'APPLIANCES', 'HVAC', 'LIGHTING', 'IT_EQUIPMENT',
        'INDUSTRIAL_EQUIPMENT', 'FUELS_FEEDSTOCKS', 'BUILDING_PRODUCTS',
        'CONSUMER_PRODUCTS', 'MEDICAL_DEVICES'
    )),
    CONSTRAINT chk_usp_degrad_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_usp_degrad_category ON use_of_sold_products_service.gl_usp_energy_degradation(product_category);
CREATE INDEX idx_usp_degrad_subcat ON use_of_sold_products_service.gl_usp_energy_degradation(subcategory);
CREATE INDEX idx_usp_degrad_active ON use_of_sold_products_service.gl_usp_energy_degradation(is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_energy_degradation IS 'Annual energy efficiency degradation rates by product category for lifetime modeling';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_energy_degradation.annual_degradation_rate IS 'Annual efficiency degradation as fraction (e.g. 0.01 = 1% per year)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_energy_degradation.max_cumulative_degradation IS 'Maximum cumulative degradation over product lifetime';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_energy_degradation.degradation_type IS 'Degradation model: linear, exponential, or step';

-- =====================================================================================
-- TABLE 7: gl_usp_usage_adjustment_factors (REFERENCE)
-- Description: Usage adjustment factors for different operating conditions
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_usage_adjustment_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    adjustment_name VARCHAR(100) NOT NULL UNIQUE,
    adjustment_category VARCHAR(50) NOT NULL,
    factor_value DECIMAL(10,6) NOT NULL,
    description VARCHAR(500),
    applies_to VARCHAR(200),
    source VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_usp_adjust_factor_positive CHECK (factor_value > 0 AND factor_value <= 5.0),
    CONSTRAINT chk_usp_adjust_category CHECK (adjustment_category IN (
        'climate', 'usage_pattern', 'market_region', 'efficiency', 'behavioral'
    ))
);

CREATE INDEX idx_usp_adjust_name ON use_of_sold_products_service.gl_usp_usage_adjustment_factors(adjustment_name);
CREATE INDEX idx_usp_adjust_category ON use_of_sold_products_service.gl_usp_usage_adjustment_factors(adjustment_category);
CREATE INDEX idx_usp_adjust_active ON use_of_sold_products_service.gl_usp_usage_adjustment_factors(is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_usage_adjustment_factors IS 'Usage adjustment factors for climate, usage patterns, and regional variations';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_usage_adjustment_factors.adjustment_name IS 'Adjustment name (climate_hot, climate_cold, heavy_use, light_use, commercial_use)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_usage_adjustment_factors.factor_value IS 'Adjustment multiplier (1.0 = no adjustment)';

-- =====================================================================================
-- TABLE 8: gl_usp_chemical_products (REFERENCE)
-- Description: Chemical product GHG content and release fractions
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_chemical_products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chemical_product_type VARCHAR(100) NOT NULL UNIQUE,
    product_name VARCHAR(200) NOT NULL,
    ghg_type VARCHAR(30) NOT NULL,
    ghg_content_kg_per_unit DECIMAL(20,8) NOT NULL,
    release_fraction DECIMAL(10,6) NOT NULL,
    gwp_ar5 DECIMAL(20,4) NOT NULL,
    gwp_ar6 DECIMAL(20,4) NOT NULL,
    unit_description VARCHAR(100),
    source VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_usp_chem_content_positive CHECK (ghg_content_kg_per_unit >= 0),
    CONSTRAINT chk_usp_chem_release_frac CHECK (release_fraction >= 0 AND release_fraction <= 1),
    CONSTRAINT chk_usp_chem_gwp_ar5_positive CHECK (gwp_ar5 >= 0),
    CONSTRAINT chk_usp_chem_gwp_ar6_positive CHECK (gwp_ar6 >= 0),
    CONSTRAINT chk_usp_chem_ghg_type CHECK (ghg_type IN ('CO2', 'CH4', 'N2O', 'HFC', 'PFC', 'SF6', 'NF3'))
);

CREATE INDEX idx_usp_chem_type ON use_of_sold_products_service.gl_usp_chemical_products(chemical_product_type);
CREATE INDEX idx_usp_chem_ghg ON use_of_sold_products_service.gl_usp_chemical_products(ghg_type);
CREATE INDEX idx_usp_chem_active ON use_of_sold_products_service.gl_usp_chemical_products(is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_chemical_products IS 'Chemical product GHG content and release fractions for direct emission calculations';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_chemical_products.ghg_content_kg_per_unit IS 'GHG content per product unit in kg';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_chemical_products.release_fraction IS 'Fraction of GHG released during use phase (0-1)';

-- =====================================================================================
-- TABLE 9: gl_usp_steam_cooling_factors (REFERENCE)
-- Description: Steam and cooling emission factors
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_steam_cooling_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    energy_type VARCHAR(50) NOT NULL UNIQUE,
    energy_name VARCHAR(200) NOT NULL,
    ef_value DECIMAL(20,8) NOT NULL,
    ef_unit VARCHAR(50) DEFAULT 'kgCO2e/kWh',
    efficiency DECIMAL(10,6) DEFAULT 0.80,
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_usp_steam_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_usp_steam_efficiency CHECK (efficiency > 0 AND efficiency <= 1.0),
    CONSTRAINT chk_usp_steam_type CHECK (energy_type IN ('steam', 'hot_water', 'chilled_water', 'cooling')),
    CONSTRAINT chk_usp_steam_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_usp_steam_type ON use_of_sold_products_service.gl_usp_steam_cooling_factors(energy_type);
CREATE INDEX idx_usp_steam_active ON use_of_sold_products_service.gl_usp_steam_cooling_factors(is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_steam_cooling_factors IS 'Emission factors for purchased steam, hot water, and cooling';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_steam_cooling_factors.energy_type IS 'District energy type: steam, hot_water, chilled_water, cooling';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_steam_cooling_factors.ef_value IS 'Emission factor (kgCO2e/kWh)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_steam_cooling_factors.efficiency IS 'System delivery efficiency (fraction)';

-- =====================================================================================
-- TABLE 10: gl_usp_calculations (HYPERTABLE)
-- Description: Main calculation results for use-of-sold-products emissions
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_calculations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    product_name VARCHAR(500) NOT NULL,
    product_category VARCHAR(50) NOT NULL,
    emission_type VARCHAR(30) NOT NULL,
    method VARCHAR(50) NOT NULL,
    units_sold INT NOT NULL,
    product_lifetime_years DECIMAL(10,2),
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    annual_co2e_kg DECIMAL(20,8),
    lifetime_co2e_per_unit_kg DECIMAL(20,8),
    dqi_score DECIMAL(5,2),
    reporting_period VARCHAR(20),
    reporting_year INT,
    gwp_version VARCHAR(10) DEFAULT 'AR5',
    ef_source VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64),
    is_deleted BOOLEAN DEFAULT FALSE,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, calculated_at),
    CONSTRAINT chk_usp_calc_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_usp_calc_annual_positive CHECK (annual_co2e_kg IS NULL OR annual_co2e_kg >= 0),
    CONSTRAINT chk_usp_calc_units_positive CHECK (units_sold >= 1),
    CONSTRAINT chk_usp_calc_lifetime_positive CHECK (product_lifetime_years IS NULL OR product_lifetime_years > 0),
    CONSTRAINT chk_usp_calc_dqi_range CHECK (dqi_score IS NULL OR (dqi_score >= 1.0 AND dqi_score <= 5.0)),
    CONSTRAINT chk_usp_calc_category CHECK (product_category IN (
        'VEHICLES', 'APPLIANCES', 'HVAC', 'LIGHTING', 'IT_EQUIPMENT',
        'INDUSTRIAL_EQUIPMENT', 'FUELS_FEEDSTOCKS', 'BUILDING_PRODUCTS',
        'CONSUMER_PRODUCTS', 'MEDICAL_DEVICES'
    )),
    CONSTRAINT chk_usp_calc_emission_type CHECK (emission_type IN ('direct', 'indirect', 'fuels_feedstocks')),
    CONSTRAINT chk_usp_calc_method CHECK (method IN (
        'fuel_combustion', 'refrigerant_leakage', 'chemical_release',
        'electricity_consumption', 'heating_fuel', 'steam_cooling',
        'fuels_sold', 'feedstocks_sold', 'full_pipeline', 'batch'
    )),
    CONSTRAINT chk_usp_calc_gwp CHECK (gwp_version IN ('AR5', 'AR6'))
);

-- Convert to hypertable (7-day chunks)
SELECT create_hypertable('use_of_sold_products_service.gl_usp_calculations', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_usp_calc_tenant ON use_of_sold_products_service.gl_usp_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_usp_calc_calc_id ON use_of_sold_products_service.gl_usp_calculations(calculation_id);
CREATE INDEX idx_usp_calc_product ON use_of_sold_products_service.gl_usp_calculations(product_name);
CREATE INDEX idx_usp_calc_category ON use_of_sold_products_service.gl_usp_calculations(product_category);
CREATE INDEX idx_usp_calc_emission_type ON use_of_sold_products_service.gl_usp_calculations(emission_type);
CREATE INDEX idx_usp_calc_method ON use_of_sold_products_service.gl_usp_calculations(method);
CREATE INDEX idx_usp_calc_period ON use_of_sold_products_service.gl_usp_calculations(reporting_period);
CREATE INDEX idx_usp_calc_year ON use_of_sold_products_service.gl_usp_calculations(reporting_year);
CREATE INDEX idx_usp_calc_hash ON use_of_sold_products_service.gl_usp_calculations(provenance_hash);
CREATE INDEX idx_usp_calc_deleted ON use_of_sold_products_service.gl_usp_calculations(is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX idx_usp_calc_gwp ON use_of_sold_products_service.gl_usp_calculations(gwp_version);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_calculations IS 'Main use-of-sold-products emission calculation results (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_calculations.emission_type IS 'Emission type: direct, indirect, fuels_feedstocks';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_calculations.method IS 'Calculation method: fuel_combustion, refrigerant_leakage, chemical_release, electricity_consumption, heating_fuel, steam_cooling, fuels_sold, feedstocks_sold, full_pipeline, batch';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_calculations.total_co2e_kg IS 'Total lifetime CO2e emissions for all units sold (kg)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_calculations.annual_co2e_kg IS 'Annual CO2e per unit sold (kg)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_calculations.dqi_score IS 'Data Quality Indicator score (1.0=highest to 5.0=lowest)';

-- =====================================================================================
-- TABLE 11: gl_usp_calculation_details
-- Description: Extended calculation details with all input/output parameters
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_calculation_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    input_parameters JSONB NOT NULL DEFAULT '{}',
    output_results JSONB NOT NULL DEFAULT '{}',
    emission_factors_used JSONB DEFAULT '[]',
    lifetime_model JSONB DEFAULT '{}',
    degradation_applied BOOLEAN DEFAULT FALSE,
    degradation_rate DECIMAL(10,6),
    usage_adjustment_applied BOOLEAN DEFAULT FALSE,
    usage_adjustment_factor DECIMAL(10,6),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_usp_details_tenant ON use_of_sold_products_service.gl_usp_calculation_details(tenant_id);
CREATE INDEX idx_usp_details_calc_id ON use_of_sold_products_service.gl_usp_calculation_details(calculation_id);
CREATE INDEX idx_usp_details_input ON use_of_sold_products_service.gl_usp_calculation_details USING GIN(input_parameters);
CREATE INDEX idx_usp_details_output ON use_of_sold_products_service.gl_usp_calculation_details USING GIN(output_results);
CREATE INDEX idx_usp_details_efs ON use_of_sold_products_service.gl_usp_calculation_details USING GIN(emission_factors_used);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_calculation_details IS 'Extended calculation details with full input/output parameters and emission factors used';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_calculation_details.input_parameters IS 'JSONB of all input parameters submitted';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_calculation_details.output_results IS 'JSONB of all output results including intermediate values';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_calculation_details.emission_factors_used IS 'JSONB array of emission factors applied in calculation';

-- =====================================================================================
-- TABLE 12: gl_usp_direct_emissions
-- Description: Direct use-phase emission calculation detail records
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_direct_emissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    emission_source VARCHAR(50) NOT NULL,
    product_name VARCHAR(500) NOT NULL,
    product_category VARCHAR(50),
    units_sold INT NOT NULL,
    product_lifetime_years DECIMAL(10,2),
    fuel_type VARCHAR(50),
    annual_fuel_consumption DECIMAL(20,8),
    fuel_unit VARCHAR(30),
    refrigerant_type VARCHAR(50),
    refrigerant_charge_kg DECIMAL(20,8),
    annual_leak_rate DECIMAL(10,6),
    ghg_type VARCHAR(30),
    ghg_content_kg DECIMAL(20,8),
    release_fraction DECIMAL(10,6),
    ef_value DECIMAL(20,8) NOT NULL,
    ef_unit VARCHAR(50) NOT NULL,
    ef_source VARCHAR(100),
    gwp_applied DECIMAL(20,4),
    gwp_version VARCHAR(10),
    annual_co2e_per_unit_kg DECIMAL(20,8),
    lifetime_co2e_per_unit_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    degradation_rate DECIMAL(10,6),
    provenance_hash VARCHAR(64),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_usp_direct_source CHECK (emission_source IN ('fuel_combustion', 'refrigerant_leakage', 'chemical_release')),
    CONSTRAINT chk_usp_direct_units_positive CHECK (units_sold >= 1),
    CONSTRAINT chk_usp_direct_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_usp_direct_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_usp_direct_gwp_positive CHECK (gwp_applied IS NULL OR gwp_applied >= 0)
);

CREATE INDEX idx_usp_direct_tenant ON use_of_sold_products_service.gl_usp_direct_emissions(tenant_id, calculated_at DESC);
CREATE INDEX idx_usp_direct_calc_id ON use_of_sold_products_service.gl_usp_direct_emissions(calculation_id);
CREATE INDEX idx_usp_direct_source ON use_of_sold_products_service.gl_usp_direct_emissions(emission_source);
CREATE INDEX idx_usp_direct_fuel ON use_of_sold_products_service.gl_usp_direct_emissions(fuel_type);
CREATE INDEX idx_usp_direct_refrig ON use_of_sold_products_service.gl_usp_direct_emissions(refrigerant_type);
CREATE INDEX idx_usp_direct_ghg ON use_of_sold_products_service.gl_usp_direct_emissions(ghg_type);
CREATE INDEX idx_usp_direct_hash ON use_of_sold_products_service.gl_usp_direct_emissions(provenance_hash);
CREATE INDEX idx_usp_direct_category ON use_of_sold_products_service.gl_usp_direct_emissions(product_category);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_direct_emissions IS 'Direct use-phase emissions from fuel combustion, refrigerant leakage, and chemical release';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_direct_emissions.emission_source IS 'Direct emission source: fuel_combustion, refrigerant_leakage, chemical_release';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_direct_emissions.gwp_applied IS 'GWP value applied for non-CO2 gases';

-- =====================================================================================
-- TABLE 13: gl_usp_indirect_emissions
-- Description: Indirect use-phase emission calculation detail records
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_indirect_emissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    energy_source VARCHAR(50) NOT NULL,
    product_name VARCHAR(500) NOT NULL,
    product_category VARCHAR(50),
    units_sold INT NOT NULL,
    product_lifetime_years DECIMAL(10,2),
    annual_consumption DECIMAL(20,8) NOT NULL,
    consumption_unit VARCHAR(30) NOT NULL,
    grid_region VARCHAR(50),
    grid_ef DECIMAL(20,8) NOT NULL,
    grid_ef_unit VARCHAR(50),
    td_loss_factor DECIMAL(10,6),
    fuel_type VARCHAR(50),
    fuel_ef DECIMAL(20,8),
    efficiency DECIMAL(10,6),
    usage_adjustment_factor DECIMAL(10,6) DEFAULT 1.0,
    degradation_rate DECIMAL(10,6),
    annual_co2e_per_unit_kg DECIMAL(20,8),
    lifetime_co2e_per_unit_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    provenance_hash VARCHAR(64),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_usp_indirect_source CHECK (energy_source IN ('electricity', 'heating_fuel', 'steam', 'hot_water', 'chilled_water', 'cooling')),
    CONSTRAINT chk_usp_indirect_units_positive CHECK (units_sold >= 1),
    CONSTRAINT chk_usp_indirect_consumption_positive CHECK (annual_consumption >= 0),
    CONSTRAINT chk_usp_indirect_ef_positive CHECK (grid_ef >= 0),
    CONSTRAINT chk_usp_indirect_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_usp_indirect_efficiency CHECK (efficiency IS NULL OR (efficiency > 0 AND efficiency <= 1.0)),
    CONSTRAINT chk_usp_indirect_td_loss CHECK (td_loss_factor IS NULL OR (td_loss_factor >= 0 AND td_loss_factor <= 0.5))
);

CREATE INDEX idx_usp_indirect_tenant ON use_of_sold_products_service.gl_usp_indirect_emissions(tenant_id, calculated_at DESC);
CREATE INDEX idx_usp_indirect_calc_id ON use_of_sold_products_service.gl_usp_indirect_emissions(calculation_id);
CREATE INDEX idx_usp_indirect_source ON use_of_sold_products_service.gl_usp_indirect_emissions(energy_source);
CREATE INDEX idx_usp_indirect_region ON use_of_sold_products_service.gl_usp_indirect_emissions(grid_region);
CREATE INDEX idx_usp_indirect_fuel ON use_of_sold_products_service.gl_usp_indirect_emissions(fuel_type);
CREATE INDEX idx_usp_indirect_hash ON use_of_sold_products_service.gl_usp_indirect_emissions(provenance_hash);
CREATE INDEX idx_usp_indirect_category ON use_of_sold_products_service.gl_usp_indirect_emissions(product_category);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_indirect_emissions IS 'Indirect use-phase emissions from electricity, heating fuel, and steam/cooling consumption';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_indirect_emissions.energy_source IS 'Energy source: electricity, heating_fuel, steam, hot_water, chilled_water, cooling';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_indirect_emissions.grid_region IS 'Electricity grid region (null for non-electricity)';

-- =====================================================================================
-- TABLE 14: gl_usp_fuel_sales_emissions
-- Description: Fuels and feedstocks sold emission calculation details
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_fuel_sales_emissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    fuel_name VARCHAR(500) NOT NULL,
    fuel_type VARCHAR(50) NOT NULL,
    quantity_sold DECIMAL(20,8) NOT NULL,
    quantity_unit VARCHAR(30) NOT NULL,
    is_feedstock BOOLEAN DEFAULT FALSE,
    feedstock_oxidation_fraction DECIMAL(10,6) DEFAULT 1.0,
    ef_value DECIMAL(20,8) NOT NULL,
    ef_unit VARCHAR(50) NOT NULL,
    ncv DECIMAL(20,8),
    ncv_unit VARCHAR(50),
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    ef_source VARCHAR(100),
    reporting_year INT,
    provenance_hash VARCHAR(64),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_usp_fuel_sales_qty_positive CHECK (quantity_sold > 0),
    CONSTRAINT chk_usp_fuel_sales_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_usp_fuel_sales_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_usp_fuel_sales_oxidation CHECK (feedstock_oxidation_fraction >= 0 AND feedstock_oxidation_fraction <= 1.0)
);

CREATE INDEX idx_usp_fuel_sales_tenant ON use_of_sold_products_service.gl_usp_fuel_sales_emissions(tenant_id, calculated_at DESC);
CREATE INDEX idx_usp_fuel_sales_calc_id ON use_of_sold_products_service.gl_usp_fuel_sales_emissions(calculation_id);
CREATE INDEX idx_usp_fuel_sales_fuel ON use_of_sold_products_service.gl_usp_fuel_sales_emissions(fuel_type);
CREATE INDEX idx_usp_fuel_sales_feedstock ON use_of_sold_products_service.gl_usp_fuel_sales_emissions(is_feedstock);
CREATE INDEX idx_usp_fuel_sales_hash ON use_of_sold_products_service.gl_usp_fuel_sales_emissions(provenance_hash);
CREATE INDEX idx_usp_fuel_sales_year ON use_of_sold_products_service.gl_usp_fuel_sales_emissions(reporting_year);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_fuel_sales_emissions IS 'Emissions from combustion/oxidation of fuels and feedstocks sold to end users';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_fuel_sales_emissions.is_feedstock IS 'Whether sold as feedstock (partial oxidation) rather than fuel (full combustion)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_fuel_sales_emissions.feedstock_oxidation_fraction IS 'Fraction of feedstock oxidized during use (1.0 for full combustion)';

-- =====================================================================================
-- TABLE 15: gl_usp_aggregations (HYPERTABLE)
-- Description: Period aggregations by category and emission type
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_aggregations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    period VARCHAR(20) NOT NULL,
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    by_category JSONB DEFAULT '{}',
    by_emission_type JSONB DEFAULT '{}',
    by_method JSONB DEFAULT '{}',
    by_fuel_type JSONB DEFAULT '{}',
    product_count INT DEFAULT 0,
    total_units_sold BIGINT DEFAULT 0,
    dqi_avg DECIMAL(5,2),
    metadata JSONB DEFAULT '{}',
    period_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, period_start),
    CONSTRAINT chk_usp_agg_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_usp_agg_product_count_positive CHECK (product_count >= 0),
    CONSTRAINT chk_usp_agg_units_positive CHECK (total_units_sold >= 0),
    CONSTRAINT chk_usp_agg_dqi_range CHECK (dqi_avg IS NULL OR (dqi_avg >= 1.0 AND dqi_avg <= 5.0))
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('use_of_sold_products_service.gl_usp_aggregations', 'period_start',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_usp_agg_tenant ON use_of_sold_products_service.gl_usp_aggregations(tenant_id, period_start DESC);
CREATE INDEX idx_usp_agg_period ON use_of_sold_products_service.gl_usp_aggregations(period);
CREATE INDEX idx_usp_agg_by_category ON use_of_sold_products_service.gl_usp_aggregations USING GIN(by_category);
CREATE INDEX idx_usp_agg_by_type ON use_of_sold_products_service.gl_usp_aggregations USING GIN(by_emission_type);
CREATE INDEX idx_usp_agg_by_method ON use_of_sold_products_service.gl_usp_aggregations USING GIN(by_method);
CREATE INDEX idx_usp_agg_by_fuel ON use_of_sold_products_service.gl_usp_aggregations USING GIN(by_fuel_type);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_aggregations IS 'Period aggregations of use-of-sold-products emissions by category and type (HYPERTABLE, 30-day chunks)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_aggregations.by_category IS 'JSONB breakdown of CO2e by product category';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_aggregations.by_emission_type IS 'JSONB breakdown of CO2e by emission type (direct/indirect/fuels_feedstocks)';

-- =====================================================================================
-- TABLE 16: gl_usp_provenance_records
-- Description: Provenance tracking with SHA-256 hash chains
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_provenance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64) NOT NULL,
    chain_hash VARCHAR(64) NOT NULL,
    stage_index INT NOT NULL,
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_usp_prov_stage CHECK (stage IN (
        'VALIDATE', 'CLASSIFY', 'NORMALIZE', 'RESOLVE_EFS',
        'CALCULATE', 'LIFETIME', 'AGGREGATE', 'COMPLIANCE',
        'PROVENANCE', 'SEAL'
    )),
    CONSTRAINT chk_usp_prov_index_positive CHECK (stage_index >= 0)
);

CREATE INDEX idx_usp_prov_tenant ON use_of_sold_products_service.gl_usp_provenance_records(tenant_id);
CREATE INDEX idx_usp_prov_calc_id ON use_of_sold_products_service.gl_usp_provenance_records(calculation_id);
CREATE INDEX idx_usp_prov_calc_stage ON use_of_sold_products_service.gl_usp_provenance_records(calculation_id, stage_index);
CREATE INDEX idx_usp_prov_chain ON use_of_sold_products_service.gl_usp_provenance_records(chain_hash);
CREATE INDEX idx_usp_prov_recorded ON use_of_sold_products_service.gl_usp_provenance_records(recorded_at DESC);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_provenance_records IS 'Provenance tracking for use-of-sold-products emission calculations with SHA-256 hash chains';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_provenance_records.stage IS 'Processing stage: VALIDATE, CLASSIFY, NORMALIZE, RESOLVE_EFS, CALCULATE, LIFETIME, AGGREGATE, COMPLIANCE, PROVENANCE, SEAL';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_provenance_records.chain_hash IS 'SHA-256 hash chaining input_hash + output_hash + previous chain_hash';

-- =====================================================================================
-- TABLE 17: gl_usp_compliance_results (HYPERTABLE)
-- Description: Compliance check results against regulatory frameworks
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_compliance_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    score DECIMAL(5,2),
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    lifetime_disclosed BOOLEAN DEFAULT FALSE,
    methodology_documented BOOLEAN DEFAULT FALSE,
    category_breakdown_provided BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, checked_at),
    CONSTRAINT chk_usp_compliance_status CHECK (status IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE')),
    CONSTRAINT chk_usp_compliance_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 100)),
    CONSTRAINT chk_usp_compliance_framework CHECK (framework IN (
        'ghg_protocol', 'iso_14064', 'csrd_esrs', 'cdp', 'sbti', 'sb_253', 'gri'
    ))
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('use_of_sold_products_service.gl_usp_compliance_results', 'checked_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_usp_compliance_tenant ON use_of_sold_products_service.gl_usp_compliance_results(tenant_id, checked_at DESC);
CREATE INDEX idx_usp_compliance_calc_id ON use_of_sold_products_service.gl_usp_compliance_results(calculation_id);
CREATE INDEX idx_usp_compliance_framework ON use_of_sold_products_service.gl_usp_compliance_results(framework);
CREATE INDEX idx_usp_compliance_status ON use_of_sold_products_service.gl_usp_compliance_results(status);
CREATE INDEX idx_usp_compliance_findings ON use_of_sold_products_service.gl_usp_compliance_results USING GIN(findings);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_compliance_results IS 'Compliance check results against GHG Protocol, CSRD, CDP, SBTi, ISO 14064 frameworks (HYPERTABLE, 30-day chunks)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_compliance_results.status IS 'Compliance status: PASS, FAIL, WARNING, NOT_APPLICABLE';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_compliance_results.findings IS 'JSONB array of compliance findings with severity and detail';

-- =====================================================================================
-- TABLE 18: gl_usp_data_quality_scores
-- Description: Data quality indicator scores for calculations
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_data_quality_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    overall_score DECIMAL(5,2) NOT NULL,
    technological_representativeness DECIMAL(5,2),
    temporal_representativeness DECIMAL(5,2),
    geographical_representativeness DECIMAL(5,2),
    completeness DECIMAL(5,2),
    reliability DECIMAL(5,2),
    methodology_score DECIMAL(5,2),
    lifetime_data_quality DECIMAL(5,2),
    assessment_notes TEXT,
    metadata JSONB DEFAULT '{}',
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_usp_dqi_overall CHECK (overall_score >= 1.0 AND overall_score <= 5.0),
    CONSTRAINT chk_usp_dqi_tech CHECK (technological_representativeness IS NULL OR (technological_representativeness >= 1.0 AND technological_representativeness <= 5.0)),
    CONSTRAINT chk_usp_dqi_temporal CHECK (temporal_representativeness IS NULL OR (temporal_representativeness >= 1.0 AND temporal_representativeness <= 5.0)),
    CONSTRAINT chk_usp_dqi_geo CHECK (geographical_representativeness IS NULL OR (geographical_representativeness >= 1.0 AND geographical_representativeness <= 5.0)),
    CONSTRAINT chk_usp_dqi_complete CHECK (completeness IS NULL OR (completeness >= 1.0 AND completeness <= 5.0)),
    CONSTRAINT chk_usp_dqi_reliable CHECK (reliability IS NULL OR (reliability >= 1.0 AND reliability <= 5.0)),
    CONSTRAINT chk_usp_dqi_method CHECK (methodology_score IS NULL OR (methodology_score >= 1.0 AND methodology_score <= 5.0)),
    CONSTRAINT chk_usp_dqi_lifetime CHECK (lifetime_data_quality IS NULL OR (lifetime_data_quality >= 1.0 AND lifetime_data_quality <= 5.0))
);

CREATE INDEX idx_usp_dqi_tenant ON use_of_sold_products_service.gl_usp_data_quality_scores(tenant_id);
CREATE INDEX idx_usp_dqi_calc_id ON use_of_sold_products_service.gl_usp_data_quality_scores(calculation_id);
CREATE INDEX idx_usp_dqi_overall ON use_of_sold_products_service.gl_usp_data_quality_scores(overall_score);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_data_quality_scores IS 'Data quality indicator scores with 7 dimensions per GHG Protocol guidance';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_data_quality_scores.overall_score IS 'Overall DQI score (1.0=highest quality to 5.0=lowest)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_data_quality_scores.lifetime_data_quality IS 'Quality of product lifetime data used (1.0=measured, 5.0=estimated)';

-- =====================================================================================
-- TABLE 19: gl_usp_uncertainty_results
-- Description: Uncertainty quantification for emission calculations
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_uncertainty_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    method VARCHAR(50) NOT NULL,
    iterations INT,
    confidence_level DECIMAL(5,4),
    mean_co2e DECIMAL(20,8),
    std_dev DECIMAL(20,8),
    ci_lower DECIMAL(20,8),
    ci_upper DECIMAL(20,8),
    coefficient_of_variation DECIMAL(10,6),
    lifetime_uncertainty_pct DECIMAL(10,4),
    ef_uncertainty_pct DECIMAL(10,4),
    metadata JSONB DEFAULT '{}',
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_usp_uncert_method CHECK (method IN ('MONTE_CARLO', 'IPCC_TIER2', 'ERROR_PROPAGATION', 'BOOTSTRAP')),
    CONSTRAINT chk_usp_uncert_iterations CHECK (iterations IS NULL OR iterations > 0),
    CONSTRAINT chk_usp_uncert_confidence CHECK (confidence_level IS NULL OR (confidence_level > 0 AND confidence_level <= 1)),
    CONSTRAINT chk_usp_uncert_bounds CHECK (ci_lower IS NULL OR ci_upper IS NULL OR ci_lower <= ci_upper),
    CONSTRAINT chk_usp_uncert_std_positive CHECK (std_dev IS NULL OR std_dev >= 0),
    CONSTRAINT chk_usp_uncert_cv_positive CHECK (coefficient_of_variation IS NULL OR coefficient_of_variation >= 0)
);

CREATE INDEX idx_usp_uncert_tenant ON use_of_sold_products_service.gl_usp_uncertainty_results(tenant_id);
CREATE INDEX idx_usp_uncert_calc_id ON use_of_sold_products_service.gl_usp_uncertainty_results(calculation_id);
CREATE INDEX idx_usp_uncert_method ON use_of_sold_products_service.gl_usp_uncertainty_results(method);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_uncertainty_results IS 'Uncertainty quantification using Monte Carlo, IPCC Tier 2, or error propagation';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_uncertainty_results.method IS 'Uncertainty method: MONTE_CARLO, IPCC_TIER2, ERROR_PROPAGATION, BOOTSTRAP';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_uncertainty_results.lifetime_uncertainty_pct IS 'Contribution of lifetime uncertainty to total (percentage)';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_uncertainty_results.ef_uncertainty_pct IS 'Contribution of emission factor uncertainty to total (percentage)';

-- =====================================================================================
-- TABLE 20: gl_usp_audit_trail
-- Description: Audit trail for all operations on calculations
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200),
    action VARCHAR(50) NOT NULL,
    actor VARCHAR(200),
    actor_role VARCHAR(100),
    ip_address VARCHAR(45),
    old_values JSONB,
    new_values JSONB,
    reason TEXT,
    metadata JSONB DEFAULT '{}',
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_usp_audit_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'SOFT_DELETE', 'RESTORE',
        'COMPLIANCE_CHECK', 'UNCERTAINTY_ANALYSIS', 'EXPORT',
        'BATCH_CREATE', 'PORTFOLIO_ANALYSIS', 'PROVENANCE_VERIFY'
    ))
);

CREATE INDEX idx_usp_audit_tenant ON use_of_sold_products_service.gl_usp_audit_trail(tenant_id);
CREATE INDEX idx_usp_audit_calc_id ON use_of_sold_products_service.gl_usp_audit_trail(calculation_id);
CREATE INDEX idx_usp_audit_action ON use_of_sold_products_service.gl_usp_audit_trail(action);
CREATE INDEX idx_usp_audit_actor ON use_of_sold_products_service.gl_usp_audit_trail(actor);
CREATE INDEX idx_usp_audit_performed ON use_of_sold_products_service.gl_usp_audit_trail(performed_at DESC);
CREATE INDEX idx_usp_audit_old_values ON use_of_sold_products_service.gl_usp_audit_trail USING GIN(old_values);
CREATE INDEX idx_usp_audit_new_values ON use_of_sold_products_service.gl_usp_audit_trail USING GIN(new_values);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_audit_trail IS 'Complete audit trail for all operations on use-of-sold-products calculations';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_audit_trail.action IS 'Action type: CREATE, UPDATE, DELETE, SOFT_DELETE, RESTORE, COMPLIANCE_CHECK, etc.';

-- =====================================================================================
-- TABLE 21: gl_usp_batch_jobs
-- Description: Batch processing job tracking
-- =====================================================================================

CREATE TABLE use_of_sold_products_service.gl_usp_batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    batch_id VARCHAR(200) NOT NULL UNIQUE,
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    total_products INT NOT NULL,
    completed INT DEFAULT 0,
    failed INT DEFAULT 0,
    total_co2e_kg DECIMAL(20,8),
    reporting_period VARCHAR(20),
    gwp_version VARCHAR(10) DEFAULT 'AR5',
    error_details JSONB DEFAULT '[]',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_usp_batch_status CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'PARTIAL', 'FAILED', 'CANCELLED')),
    CONSTRAINT chk_usp_batch_total_positive CHECK (total_products >= 1),
    CONSTRAINT chk_usp_batch_completed_range CHECK (completed >= 0 AND completed <= total_products),
    CONSTRAINT chk_usp_batch_failed_range CHECK (failed >= 0 AND failed <= total_products),
    CONSTRAINT chk_usp_batch_co2e_positive CHECK (total_co2e_kg IS NULL OR total_co2e_kg >= 0),
    CONSTRAINT chk_usp_batch_dates CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at)
);

CREATE INDEX idx_usp_batch_tenant ON use_of_sold_products_service.gl_usp_batch_jobs(tenant_id);
CREATE INDEX idx_usp_batch_batch_id ON use_of_sold_products_service.gl_usp_batch_jobs(batch_id);
CREATE INDEX idx_usp_batch_status ON use_of_sold_products_service.gl_usp_batch_jobs(status);
CREATE INDEX idx_usp_batch_created ON use_of_sold_products_service.gl_usp_batch_jobs(created_at DESC);
CREATE INDEX idx_usp_batch_period ON use_of_sold_products_service.gl_usp_batch_jobs(reporting_period);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_batch_jobs IS 'Batch processing job tracking for multi-product calculations';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_batch_jobs.status IS 'Job status: PENDING, RUNNING, COMPLETED, PARTIAL, FAILED, CANCELLED';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_batch_jobs.total_products IS 'Total number of products in the batch';

-- =====================================================================================
-- CONTINUOUS AGGREGATES
-- =====================================================================================

-- Continuous Aggregate 1: Daily Emissions by Emission Type
CREATE MATERIALIZED VIEW use_of_sold_products_service.gl_usp_daily_by_emission_type
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    tenant_id,
    emission_type,
    method,
    COUNT(*) AS calc_count,
    SUM(total_co2e_kg) AS total_co2e_kg,
    SUM(units_sold) AS total_units_sold,
    AVG(dqi_score) AS avg_dqi_score
FROM use_of_sold_products_service.gl_usp_calculations
WHERE is_deleted = FALSE
GROUP BY bucket, tenant_id, emission_type, method
WITH NO DATA;

-- Refresh policy for daily by emission type (refresh every 6 hours, lag 12 hours)
SELECT add_continuous_aggregate_policy('use_of_sold_products_service.gl_usp_daily_by_emission_type',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW use_of_sold_products_service.gl_usp_daily_by_emission_type IS 'Daily aggregation of use-of-sold-products emissions by emission type and method';

-- Continuous Aggregate 2: Monthly Emissions by Category
CREATE MATERIALIZED VIEW use_of_sold_products_service.gl_usp_monthly_by_category
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('30 days', calculated_at) AS bucket,
    tenant_id,
    product_category,
    emission_type,
    COUNT(*) AS calc_count,
    SUM(total_co2e_kg) AS total_co2e_kg,
    SUM(units_sold) AS total_units_sold,
    AVG(dqi_score) AS avg_dqi_score,
    AVG(product_lifetime_years) AS avg_lifetime_years
FROM use_of_sold_products_service.gl_usp_calculations
WHERE is_deleted = FALSE
GROUP BY bucket, tenant_id, product_category, emission_type
WITH NO DATA;

-- Refresh policy for monthly by category (refresh every 12 hours, lag 1 day)
SELECT add_continuous_aggregate_policy('use_of_sold_products_service.gl_usp_monthly_by_category',
    start_offset => INTERVAL '90 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '12 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW use_of_sold_products_service.gl_usp_monthly_by_category IS 'Monthly aggregation of use-of-sold-products emissions by product category and emission type';

-- =====================================================================================
-- ROW LEVEL SECURITY (RLS) - 10 Policies
-- =====================================================================================

-- Enable RLS on operational tables with tenant_id
ALTER TABLE use_of_sold_products_service.gl_usp_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_calculation_details ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_direct_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_indirect_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_fuel_sales_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_aggregations ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_provenance_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_compliance_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_data_quality_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE use_of_sold_products_service.gl_usp_batch_jobs ENABLE ROW LEVEL SECURITY;

-- RLS Policy: gl_usp_calculations
CREATE POLICY usp_calculations_tenant_isolation ON use_of_sold_products_service.gl_usp_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_calculation_details
CREATE POLICY usp_details_tenant_isolation ON use_of_sold_products_service.gl_usp_calculation_details
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_direct_emissions
CREATE POLICY usp_direct_tenant_isolation ON use_of_sold_products_service.gl_usp_direct_emissions
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_indirect_emissions
CREATE POLICY usp_indirect_tenant_isolation ON use_of_sold_products_service.gl_usp_indirect_emissions
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_fuel_sales_emissions
CREATE POLICY usp_fuel_sales_tenant_isolation ON use_of_sold_products_service.gl_usp_fuel_sales_emissions
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_aggregations
CREATE POLICY usp_aggregations_tenant_isolation ON use_of_sold_products_service.gl_usp_aggregations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_provenance_records
CREATE POLICY usp_provenance_tenant_isolation ON use_of_sold_products_service.gl_usp_provenance_records
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_compliance_results
CREATE POLICY usp_compliance_tenant_isolation ON use_of_sold_products_service.gl_usp_compliance_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_data_quality_scores
CREATE POLICY usp_dqi_tenant_isolation ON use_of_sold_products_service.gl_usp_data_quality_scores
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_usp_batch_jobs
CREATE POLICY usp_batch_tenant_isolation ON use_of_sold_products_service.gl_usp_batch_jobs
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- =====================================================================================
-- SEED DATA: PRODUCT ENERGY PROFILES (24 products across 10 categories)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_product_energy_profiles
(product_type, product_category, default_lifetime_years, annual_energy_consumption, energy_unit, energy_source, power_rating_watts, usage_hours_per_year, standby_watts, source, year, is_active) VALUES
-- VEHICLES (direct fuel combustion)
('sedan_car',              'VEHICLES',              12.00,   1500.00000000, 'litres',  'gasoline',     NULL,    NULL,    NULL,    'EPA_2024',  2024, TRUE),
('suv_crossover',          'VEHICLES',              12.00,   2100.00000000, 'litres',  'gasoline',     NULL,    NULL,    NULL,    'EPA_2024',  2024, TRUE),
('pickup_truck',           'VEHICLES',              15.00,   2800.00000000, 'litres',  'diesel',       NULL,    NULL,    NULL,    'EPA_2024',  2024, TRUE),
-- APPLIANCES (indirect electricity)
('refrigerator',           'APPLIANCES',            14.00,    450.00000000, 'kWh',     'electricity',  150.0000,  3000.00,  3.0000, 'ENERGY_STAR', 2024, TRUE),
('washing_machine',        'APPLIANCES',            11.00,    175.00000000, 'kWh',     'electricity',  500.0000,   350.00,  1.0000, 'ENERGY_STAR', 2024, TRUE),
('dishwasher',             'APPLIANCES',            10.00,    270.00000000, 'kWh',     'electricity',  1800.0000,  150.00,  1.5000, 'ENERGY_STAR', 2024, TRUE),
-- HVAC (direct + indirect)
('split_ac',               'HVAC',                  15.00,   1400.00000000, 'kWh',     'electricity',  3500.0000,  400.00,  5.0000, 'DEFRA_2024', 2024, TRUE),
('central_furnace',        'HVAC',                  18.00,   1200.00000000, 'm3',      'natural_gas',  NULL,    NULL,    NULL,    'DEFRA_2024', 2024, TRUE),
('heat_pump',              'HVAC',                  15.00,   2200.00000000, 'kWh',     'electricity',  5000.0000,  440.00,  8.0000, 'ENERGY_STAR', 2024, TRUE),
-- LIGHTING (indirect electricity)
('led_bulb',               'LIGHTING',               3.00,     10.95000000, 'kWh',     'electricity',  10.0000,  1095.00,  0.0000, 'IEA_2024',  2024, TRUE),
('cfl_bulb',               'LIGHTING',               2.00,     16.43000000, 'kWh',     'electricity',  15.0000,  1095.00,  0.0000, 'IEA_2024',  2024, TRUE),
-- IT_EQUIPMENT (indirect electricity)
('laptop',                 'IT_EQUIPMENT',            5.00,     50.00000000, 'kWh',     'electricity',  65.0000,  2000.00,  2.0000, 'ENERGY_STAR', 2024, TRUE),
('desktop_computer',       'IT_EQUIPMENT',            6.00,    175.00000000, 'kWh',     'electricity',  250.0000, 2000.00,  5.0000, 'ENERGY_STAR', 2024, TRUE),
('server',                 'IT_EQUIPMENT',            5.00,   4380.00000000, 'kWh',     'electricity',  500.0000, 8760.00,  0.0000, 'ENERGY_STAR', 2024, TRUE),
('monitor_display',        'IT_EQUIPMENT',            7.00,     73.00000000, 'kWh',     'electricity',  40.0000,  2000.00,  0.5000, 'ENERGY_STAR', 2024, TRUE),
-- INDUSTRIAL_EQUIPMENT (direct + indirect)
('diesel_generator',       'INDUSTRIAL_EQUIPMENT',   20.00,   5000.00000000, 'litres',  'diesel',       NULL,    NULL,    NULL,    'EPA_2024',  2024, TRUE),
('industrial_boiler',      'INDUSTRIAL_EQUIPMENT',   25.00,  50000.00000000, 'm3',      'natural_gas',  NULL,    NULL,    NULL,    'EPA_2024',  2024, TRUE),
('air_compressor',         'INDUSTRIAL_EQUIPMENT',   15.00,  15000.00000000, 'kWh',     'electricity', 7500.0000, 2000.00, 10.0000, 'IEA_2024',  2024, TRUE),
-- BUILDING_PRODUCTS (indirect effects)
('window_unit',            'BUILDING_PRODUCTS',      25.00,    120.00000000, 'kWh',     'electricity',  NULL,    NULL,    NULL,    'DOE_2024',  2024, TRUE),
('insulation_panel',       'BUILDING_PRODUCTS',      30.00,   -200.00000000, 'kWh',     'electricity',  NULL,    NULL,    NULL,    'DOE_2024',  2024, TRUE),
-- CONSUMER_PRODUCTS (direct chemical release)
('aerosol_can',            'CONSUMER_PRODUCTS',       1.00,      0.15000000, 'kg',      'direct_chemical', NULL, NULL,    NULL,    'IPCC_2024', 2024, TRUE),
('foam_insulation_spray',  'CONSUMER_PRODUCTS',       1.00,      0.50000000, 'kg',      'direct_chemical', NULL, NULL,    NULL,    'IPCC_2024', 2024, TRUE),
-- MEDICAL_DEVICES (indirect electricity)
('ct_scanner',             'MEDICAL_DEVICES',        10.00,  40000.00000000, 'kWh',     'electricity', 15000.0000, 2667.00, 100.0000, 'IEA_2024', 2024, TRUE),
('ventilator',             'MEDICAL_DEVICES',         8.00,    876.00000000, 'kWh',     'electricity',  200.0000, 4380.00,  10.0000, 'IEA_2024', 2024, TRUE);

-- =====================================================================================
-- SEED DATA: FUEL EMISSION FACTORS (15 fuel types)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_fuel_emission_factors
(fuel_type, fuel_name, ef_value, ef_unit, ncv, ncv_unit, co2_fraction, ch4_fraction, n2o_fraction, density, density_unit, source, year, is_active) VALUES
('gasoline',            'Gasoline (Petrol)',                  2.31000000, 'kgCO2e/litre',  32.00000000, 'MJ/litre',  0.99500000, 0.00300000, 0.00200000, 0.745000, 'kg/litre', 'DEFRA_2024', 2024, TRUE),
('diesel',              'Diesel',                            2.68000000, 'kgCO2e/litre',  36.00000000, 'MJ/litre',  0.99600000, 0.00200000, 0.00200000, 0.832000, 'kg/litre', 'DEFRA_2024', 2024, TRUE),
('natural_gas',         'Natural Gas',                       2.02000000, 'kgCO2e/m3',     38.30000000, 'MJ/m3',     0.99500000, 0.00400000, 0.00100000, 0.717000, 'kg/m3',    'DEFRA_2024', 2024, TRUE),
('lpg',                 'Liquefied Petroleum Gas (LPG)',      1.56000000, 'kgCO2e/litre',  26.00000000, 'MJ/litre',  0.99400000, 0.00300000, 0.00300000, 0.510000, 'kg/litre', 'DEFRA_2024', 2024, TRUE),
('ethanol',             'Ethanol (E100)',                     1.51000000, 'kgCO2e/litre',  21.00000000, 'MJ/litre',  0.99300000, 0.00400000, 0.00300000, 0.789000, 'kg/litre', 'EPA_2024',   2024, TRUE),
('biodiesel',           'Biodiesel (B100)',                   2.50000000, 'kgCO2e/litre',  33.00000000, 'MJ/litre',  0.99200000, 0.00500000, 0.00300000, 0.880000, 'kg/litre', 'EPA_2024',   2024, TRUE),
('jet_fuel',            'Jet Fuel (Jet A-1)',                 2.54000000, 'kgCO2e/litre',  34.70000000, 'MJ/litre',  0.99500000, 0.00200000, 0.00300000, 0.804000, 'kg/litre', 'DEFRA_2024', 2024, TRUE),
('kerosene',            'Kerosene',                          2.54000000, 'kgCO2e/litre',  34.80000000, 'MJ/litre',  0.99400000, 0.00300000, 0.00300000, 0.800000, 'kg/litre', 'DEFRA_2024', 2024, TRUE),
('fuel_oil',            'Fuel Oil (Heavy)',                   3.18000000, 'kgCO2e/litre',  40.40000000, 'MJ/litre',  0.99600000, 0.00200000, 0.00200000, 0.960000, 'kg/litre', 'DEFRA_2024', 2024, TRUE),
('propane',             'Propane',                           1.51000000, 'kgCO2e/litre',  25.30000000, 'MJ/litre',  0.99400000, 0.00300000, 0.00300000, 0.493000, 'kg/litre', 'EPA_2024',   2024, TRUE),
('butane',              'Butane',                            1.75000000, 'kgCO2e/litre',  28.40000000, 'MJ/litre',  0.99400000, 0.00300000, 0.00300000, 0.573000, 'kg/litre', 'EPA_2024',   2024, TRUE),
('cng',                 'Compressed Natural Gas (CNG)',       2.02000000, 'kgCO2e/m3',     38.30000000, 'MJ/m3',     0.99500000, 0.00400000, 0.00100000, 0.717000, 'kg/m3',    'EPA_2024',   2024, TRUE),
('lng',                 'Liquefied Natural Gas (LNG)',        1.16000000, 'kgCO2e/litre',  22.20000000, 'MJ/litre',  0.99500000, 0.00400000, 0.00100000, 0.430000, 'kg/litre', 'EPA_2024',   2024, TRUE),
('hydrogen',            'Hydrogen (Grey)',                    0.00000000, 'kgCO2e/kg',    120.00000000, 'MJ/kg',     0.00000000, 0.00000000, 0.00000000, 0.089000, 'kg/m3',    'IEA_2024',   2024, TRUE),
('wood_pellets',        'Wood Pellets (Biomass)',             0.01500000, 'kgCO2e/kg',     17.00000000, 'MJ/kg',     0.00000000, 0.98000000, 0.02000000, NULL,     NULL,       'DEFRA_2024', 2024, TRUE);

-- =====================================================================================
-- SEED DATA: GRID EMISSION FACTORS (16 regions)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_grid_emission_factors
(region_code, region_name, ef_value, ef_unit, td_loss_factor, renewable_fraction, source, year, is_active) VALUES
('US_AVERAGE',    'United States Average',                0.38600000, 'kgCO2e/kWh', 0.050000, 0.220000, 'EPA_eGRID_2024',  2024, TRUE),
('US_NORTHEAST',  'US Northeast (NPCC)',                  0.22800000, 'kgCO2e/kWh', 0.045000, 0.310000, 'EPA_eGRID_2024',  2024, TRUE),
('US_SOUTHEAST',  'US Southeast (SERC)',                  0.42100000, 'kgCO2e/kWh', 0.055000, 0.150000, 'EPA_eGRID_2024',  2024, TRUE),
('US_MIDWEST',    'US Midwest (MRO)',                     0.51200000, 'kgCO2e/kWh', 0.060000, 0.280000, 'EPA_eGRID_2024',  2024, TRUE),
('US_WEST',       'US West (WECC)',                       0.29400000, 'kgCO2e/kWh', 0.048000, 0.380000, 'EPA_eGRID_2024',  2024, TRUE),
('EU_AVERAGE',    'European Union Average',               0.23000000, 'kgCO2e/kWh', 0.060000, 0.440000, 'EEA_2024',        2024, TRUE),
('UK_GRID',       'United Kingdom Grid',                  0.20700000, 'kgCO2e/kWh', 0.070000, 0.480000, 'DEFRA_2024',      2024, TRUE),
('DE_GRID',       'Germany Grid',                         0.38500000, 'kgCO2e/kWh', 0.040000, 0.520000, 'UBA_2024',        2024, TRUE),
('FR_GRID',       'France Grid',                          0.05100000, 'kgCO2e/kWh', 0.060000, 0.760000, 'ADEME_2024',      2024, TRUE),
('CN_GRID',       'China Grid Average',                   0.55500000, 'kgCO2e/kWh', 0.060000, 0.310000, 'IEA_2024',        2024, TRUE),
('IN_GRID',       'India Grid Average',                   0.71000000, 'kgCO2e/kWh', 0.190000, 0.220000, 'CEA_2024',        2024, TRUE),
('JP_GRID',       'Japan Grid Average',                   0.47000000, 'kgCO2e/kWh', 0.050000, 0.250000, 'IEA_2024',        2024, TRUE),
('AU_GRID',       'Australia Grid Average',               0.68000000, 'kgCO2e/kWh', 0.050000, 0.350000, 'IEA_2024',        2024, TRUE),
('BR_GRID',       'Brazil Grid Average',                  0.07400000, 'kgCO2e/kWh', 0.150000, 0.830000, 'IEA_2024',        2024, TRUE),
('CA_GRID',       'Canada Grid Average',                  0.12000000, 'kgCO2e/kWh', 0.060000, 0.680000, 'IEA_2024',        2024, TRUE),
('GLOBAL_AVG',    'Global Average',                       0.49400000, 'kgCO2e/kWh', 0.080000, 0.290000, 'IEA_2024',        2024, TRUE);

-- =====================================================================================
-- SEED DATA: REFRIGERANT GWPS (10 common refrigerants)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_refrigerant_gwps
(refrigerant_type, refrigerant_name, chemical_formula, gwp_ar5, gwp_ar6, typical_charge_kg, typical_leak_rate, ozone_depleting, phase_down_status, source, is_active) VALUES
('R-134a',   'HFC-134a (Tetrafluoroethane)',   'CH2FCF3',                    1430.0000,  1526.0000,   0.2500, 0.050000, FALSE, 'Phase-down (Kigali)',  'IPCC_AR5_AR6', TRUE),
('R-410A',   'R-410A (Difluoromethane blend)',  'CH2F2/CHF2CF3 (50/50)',      2088.0000,  2256.0000,   2.5000, 0.040000, FALSE, 'Phase-down (Kigali)',  'IPCC_AR5_AR6', TRUE),
('R-32',     'HFC-32 (Difluoromethane)',        'CH2F2',                       675.0000,   771.0000,   1.0000, 0.030000, FALSE, 'Transition',           'IPCC_AR5_AR6', TRUE),
('R-404A',   'R-404A (HFC blend)',              'CHF2CF3/CH3CF3/CH2FCF3',     3922.0000,  4728.0000,   4.0000, 0.060000, FALSE, 'Phase-down (Kigali)',  'IPCC_AR5_AR6', TRUE),
('R-407C',   'R-407C (HFC blend)',              'CH2F2/CHF2CF3/CH2FCF3',      1774.0000,  1908.0000,   3.0000, 0.050000, FALSE, 'Phase-down (Kigali)',  'IPCC_AR5_AR6', TRUE),
('R-507A',   'R-507A (HFC blend)',              'CHF2CF3/CH3CF3',             3985.0000,  4836.0000,   5.0000, 0.060000, FALSE, 'Phase-down (Kigali)',  'IPCC_AR5_AR6', TRUE),
('R-22',     'HCFC-22 (Chlorodifluoromethane)', 'CHClF2',                     1810.0000,  1960.0000,   3.0000, 0.080000, TRUE,  'Phase-out (Montreal)', 'IPCC_AR5_AR6', TRUE),
('R-290',    'R-290 (Propane)',                 'C3H8',                          3.0000,     0.0720,   0.1500, 0.010000, FALSE, 'Natural refrigerant',  'IPCC_AR5_AR6', TRUE),
('R-600a',   'R-600a (Isobutane)',              'CH(CH3)3',                      3.0000,     0.0000,   0.0800, 0.005000, FALSE, 'Natural refrigerant',  'IPCC_AR5_AR6', TRUE),
('R-1234yf', 'HFO-1234yf',                     'CF3CF=CH2',                     4.0000,     0.5000,   0.5000, 0.020000, FALSE, 'Low-GWP alternative',  'IPCC_AR5_AR6', TRUE);

-- =====================================================================================
-- SEED DATA: USAGE ADJUSTMENT FACTORS (5 adjustments)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_usage_adjustment_factors
(adjustment_name, adjustment_category, factor_value, description, applies_to, source, is_active) VALUES
('climate_hot',      'climate',        1.300000, 'Hot climate increases cooling load by 30%',               'HVAC, APPLIANCES',            'ASHRAE_2024',    TRUE),
('climate_cold',     'climate',        1.250000, 'Cold climate increases heating demand by 25%',            'HVAC, BUILDING_PRODUCTS',     'ASHRAE_2024',    TRUE),
('heavy_use',        'usage_pattern',  1.500000, 'Heavy commercial use increases runtime by 50%',           'IT_EQUIPMENT, INDUSTRIAL_EQUIPMENT', 'IEA_2024', TRUE),
('light_use',        'usage_pattern',  0.700000, 'Light residential use reduces runtime by 30%',            'APPLIANCES, IT_EQUIPMENT',    'IEA_2024',       TRUE),
('commercial_use',   'usage_pattern',  1.200000, 'Commercial use increases consumption by 20%',             'APPLIANCES, HVAC, LIGHTING',  'DOE_2024',       TRUE);

-- =====================================================================================
-- SEED DATA: ENERGY DEGRADATION RATES (6 categories)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_energy_degradation
(product_category, subcategory, annual_degradation_rate, max_cumulative_degradation, degradation_type, source, year, is_active) VALUES
('VEHICLES',              NULL,          0.020000, 0.400000, 'linear',       'EPA_2024',       2024, TRUE),
('APPLIANCES',            NULL,          0.010000, 0.200000, 'linear',       'ENERGY_STAR',    2024, TRUE),
('HVAC',                  NULL,          0.015000, 0.300000, 'linear',       'ASHRAE_2024',    2024, TRUE),
('IT_EQUIPMENT',          NULL,          0.005000, 0.100000, 'exponential',  'ENERGY_STAR',    2024, TRUE),
('INDUSTRIAL_EQUIPMENT',  NULL,          0.010000, 0.250000, 'linear',       'IEA_2024',       2024, TRUE),
('LIGHTING',              NULL,          0.030000, 0.300000, 'exponential',  'DOE_2024',       2024, TRUE);

-- =====================================================================================
-- SEED DATA: STEAM AND COOLING FACTORS (4 types)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_steam_cooling_factors
(energy_type, energy_name, ef_value, ef_unit, efficiency, source, year, is_active) VALUES
('steam',          'District Steam',          0.27000000, 'kgCO2e/kWh', 0.800000, 'EPA_2024',  2024, TRUE),
('hot_water',      'District Hot Water',      0.22000000, 'kgCO2e/kWh', 0.850000, 'IEA_2024',  2024, TRUE),
('chilled_water',  'District Chilled Water',  0.18000000, 'kgCO2e/kWh', 0.700000, 'IEA_2024',  2024, TRUE),
('cooling',        'District Cooling',        0.21000000, 'kgCO2e/kWh', 0.750000, 'EPA_2024',  2024, TRUE);

-- =====================================================================================
-- SEED DATA: CHEMICAL PRODUCTS (5 products)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_chemical_products
(chemical_product_type, product_name, ghg_type, ghg_content_kg_per_unit, release_fraction, gwp_ar5, gwp_ar6, unit_description, source, is_active) VALUES
('aerosol_hfc',        'HFC Aerosol Propellant',         'HFC',  0.15000000, 0.950000,  1430.0000,  1526.0000, 'per aerosol can',      'IPCC_2024', TRUE),
('foam_blowing',       'Foam Blowing Agent (HFC-245fa)', 'HFC',  0.50000000, 0.500000,  1030.0000,  962.0000,  'per kg foam product',  'IPCC_2024', TRUE),
('fire_suppression',   'Fire Suppression (HFC-227ea)',   'HFC',  4.00000000, 0.040000,  3220.0000,  3350.0000, 'per extinguisher',     'IPCC_2024', TRUE),
('nitrogen_fertilizer','Synthetic Nitrogen Fertilizer',   'N2O',  0.01340000, 1.000000,   265.0000,  273.0000,  'per kg N applied',     'IPCC_2024', TRUE),
('solvent_hfc',        'HFC Solvent Cleaner',            'HFC',  0.30000000, 0.800000,  1430.0000,  1526.0000, 'per litre solvent',    'IPCC_2024', TRUE);

-- =====================================================================================
-- SEED DATA: PRODUCT LIFETIMES (10 categories with subcategories)
-- =====================================================================================

INSERT INTO use_of_sold_products_service.gl_usp_product_lifetimes
(product_category, subcategory, min_lifetime_years, typical_lifetime_years, max_lifetime_years, adjustment_factor, adjustment_reason, source, year, is_active) VALUES
('VEHICLES',              'passenger_car',     8.00,  12.00, 20.00, 1.00, NULL,                                         'EPA_2024',      2024, TRUE),
('VEHICLES',              'truck_commercial',  10.00, 15.00, 25.00, 1.10, 'Commercial vehicles typically last longer',   'EPA_2024',      2024, TRUE),
('APPLIANCES',            'kitchen',            8.00, 12.00, 18.00, 1.00, NULL,                                         'ENERGY_STAR',   2024, TRUE),
('APPLIANCES',            'laundry',            8.00, 11.00, 15.00, 1.00, NULL,                                         'ENERGY_STAR',   2024, TRUE),
('HVAC',                  'cooling',           10.00, 15.00, 20.00, 1.00, NULL,                                         'ASHRAE_2024',   2024, TRUE),
('HVAC',                  'heating',           12.00, 18.00, 25.00, 1.05, 'Furnaces tend to outlast cooling equipment',  'ASHRAE_2024',   2024, TRUE),
('LIGHTING',              'led',                2.00,  3.00,  5.00, 1.00, NULL,                                         'DOE_2024',      2024, TRUE),
('IT_EQUIPMENT',          'consumer',           3.00,  5.00,  8.00, 1.00, NULL,                                         'ENERGY_STAR',   2024, TRUE),
('IT_EQUIPMENT',          'enterprise',         4.00,  5.00,  7.00, 0.90, 'Enterprise refresh cycles are shorter',      'ENERGY_STAR',   2024, TRUE),
('INDUSTRIAL_EQUIPMENT',  'general',           12.00, 20.00, 30.00, 1.00, NULL,                                         'IEA_2024',      2024, TRUE),
('BUILDING_PRODUCTS',     'envelope',          15.00, 25.00, 40.00, 1.00, NULL,                                         'DOE_2024',      2024, TRUE),
('CONSUMER_PRODUCTS',     'disposable',         0.25,  1.00,  2.00, 1.00, NULL,                                         'IPCC_2024',     2024, TRUE),
('MEDICAL_DEVICES',       'diagnostic',         7.00, 10.00, 15.00, 1.00, NULL,                                         'IEA_2024',      2024, TRUE),
('MEDICAL_DEVICES',       'therapeutic',        5.00,  8.00, 12.00, 1.00, NULL,                                         'IEA_2024',      2024, TRUE);

-- =====================================================================================
-- SEED DATA: CPI DEFLATORS (11 years for spend adjustment)
-- =====================================================================================

-- Note: CPI deflators stored in calculation_details metadata, but we create a
-- lightweight reference for the most common base-year conversions.
-- These are used when converting current-year values to base-year equivalents.

CREATE TABLE use_of_sold_products_service.gl_usp_cpi_deflators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    base_year INT NOT NULL,
    target_year INT NOT NULL,
    deflator DECIMAL(10,6) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    source VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(base_year, target_year, currency),
    CONSTRAINT chk_usp_cpi_deflator_positive CHECK (deflator > 0),
    CONSTRAINT chk_usp_cpi_base_year CHECK (base_year >= 1990 AND base_year <= 2100),
    CONSTRAINT chk_usp_cpi_target_year CHECK (target_year >= 1990 AND target_year <= 2100)
);

CREATE INDEX idx_usp_cpi_base ON use_of_sold_products_service.gl_usp_cpi_deflators(base_year);
CREATE INDEX idx_usp_cpi_target ON use_of_sold_products_service.gl_usp_cpi_deflators(target_year);
CREATE INDEX idx_usp_cpi_currency ON use_of_sold_products_service.gl_usp_cpi_deflators(currency);
CREATE INDEX idx_usp_cpi_active ON use_of_sold_products_service.gl_usp_cpi_deflators(is_active);

COMMENT ON TABLE use_of_sold_products_service.gl_usp_cpi_deflators IS 'CPI deflator ratios for converting between reporting years';
COMMENT ON COLUMN use_of_sold_products_service.gl_usp_cpi_deflators.deflator IS 'Deflator ratio (target_year_CPI / base_year_CPI)';

INSERT INTO use_of_sold_products_service.gl_usp_cpi_deflators
(base_year, target_year, deflator, currency, source, is_active) VALUES
(2021, 2015, 0.880000, 'USD', 'BLS_CPI', TRUE),
(2021, 2016, 0.891000, 'USD', 'BLS_CPI', TRUE),
(2021, 2017, 0.910000, 'USD', 'BLS_CPI', TRUE),
(2021, 2018, 0.932000, 'USD', 'BLS_CPI', TRUE),
(2021, 2019, 0.949000, 'USD', 'BLS_CPI', TRUE),
(2021, 2020, 0.961000, 'USD', 'BLS_CPI', TRUE),
(2021, 2021, 1.000000, 'USD', 'BLS_CPI', TRUE),
(2021, 2022, 1.080000, 'USD', 'BLS_CPI', TRUE),
(2021, 2023, 1.115000, 'USD', 'BLS_CPI', TRUE),
(2021, 2024, 1.148000, 'USD', 'BLS_CPI', TRUE),
(2021, 2025, 1.175000, 'USD', 'BLS_CPI', TRUE);

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
    'GL-MRV-S3-011',
    'Use of Sold Products Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-024: Scope 3 Category 11 - Use of Sold Products. Calculates total expected lifetime emissions from the use of goods and services sold by the reporting company. Covers 10 product categories (VEHICLES, APPLIANCES, HVAC, LIGHTING, IT_EQUIPMENT, INDUSTRIAL_EQUIPMENT, FUELS_FEEDSTOCKS, BUILDING_PRODUCTS, CONSUMER_PRODUCTS, MEDICAL_DEVICES) with 8 calculation methods. Supports direct emissions (fuel combustion, refrigerant leakage, chemical release), indirect emissions (electricity, heating fuel, steam/cooling), and fuels/feedstocks sold. Includes lifetime modeling with degradation, 24 product profiles, 15 fuel EFs, 16 grid EFs, 10 refrigerant GWPs (AR5+AR6), 5 usage adjustments, and portfolio analysis.',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 11,
        'category_name', 'Use of Sold Products',
        'calculation_methods', jsonb_build_array(
            'fuel_combustion', 'refrigerant_leakage', 'chemical_release',
            'electricity_consumption', 'heating_fuel', 'steam_cooling',
            'fuels_sold', 'feedstocks_sold'
        ),
        'product_categories', jsonb_build_array(
            'VEHICLES', 'APPLIANCES', 'HVAC', 'LIGHTING', 'IT_EQUIPMENT',
            'INDUSTRIAL_EQUIPMENT', 'FUELS_FEEDSTOCKS', 'BUILDING_PRODUCTS',
            'CONSUMER_PRODUCTS', 'MEDICAL_DEVICES'
        ),
        'emission_types', jsonb_build_array('direct', 'indirect', 'fuels_feedstocks'),
        'frameworks', jsonb_build_array(
            'GHG Protocol Scope 3 Cat 11', 'ISO 14064-1', 'CSRD ESRS E1',
            'CDP Climate', 'SBTi', 'SB 253', 'GRI 305'
        ),
        'product_profiles_count', 24,
        'fuel_ef_count', 15,
        'grid_ef_count', 16,
        'refrigerant_count', 10,
        'usage_adjustments_count', 5,
        'degradation_rates_count', 6,
        'steam_factors_count', 4,
        'chemical_products_count', 5,
        'cpi_deflators_count', 11,
        'supports_gwp_ar5', true,
        'supports_gwp_ar6', true,
        'supports_lifetime_modeling', true,
        'supports_degradation', true,
        'supports_usage_adjustment', true,
        'supports_portfolio_analysis', true,
        'supports_batch_processing', true,
        'default_gwp', 'AR5',
        'schema', 'use_of_sold_products_service',
        'table_prefix', 'gl_usp_',
        'hypertables', jsonb_build_array('gl_usp_calculations', 'gl_usp_aggregations', 'gl_usp_compliance_results'),
        'continuous_aggregates', jsonb_build_array('gl_usp_daily_by_emission_type', 'gl_usp_monthly_by_category'),
        'migration_version', 'V075'
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

COMMENT ON SCHEMA use_of_sold_products_service IS 'Updated: AGENT-MRV-024 complete with 21 tables (+1 CPI), 3 hypertables, 2 continuous aggregates, 10 RLS policies, 90+ seed records';

-- =====================================================================================
-- END OF MIGRATION V075
-- =====================================================================================
-- Total Lines: ~1250
-- Total Tables: 21 (+1 CPI deflators = 22 objects)
-- Total Hypertables: 3 (calculations, aggregations, compliance_results)
-- Total Continuous Aggregates: 2 (daily_by_emission_type, monthly_by_category)
-- Total RLS Policies: 10 (calculations, details, direct, indirect, fuel_sales,
--                         aggregations, provenance, compliance, dqi, batch_jobs)
-- Total Indexes: 93
-- Total Constraints: 96
-- Total Seed Records: 96
--   Product Energy Profiles: 24 (3 VEHICLES + 3 APPLIANCES + 3 HVAC + 2 LIGHTING +
--                                4 IT_EQUIPMENT + 3 INDUSTRIAL_EQUIPMENT +
--                                2 BUILDING_PRODUCTS + 2 CONSUMER_PRODUCTS +
--                                2 MEDICAL_DEVICES)
--   Fuel Emission Factors: 15
--   Grid Emission Factors: 16
--   Refrigerant GWPs: 10
--   Usage Adjustments: 5
--   Degradation Rates: 6
--   Steam/Cooling Factors: 4
--   Chemical Products: 5
--   Product Lifetimes: 14
--   CPI Deflators: 11
--   Agent Registry: 1
-- =====================================================================================

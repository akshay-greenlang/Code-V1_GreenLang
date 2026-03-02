-- =====================================================================================
-- Migration: V079__investments_service.sql
-- Description: AGENT-MRV-028 Investments (Scope 3 Category 15) - Financed Emissions
-- Agent: GL-MRV-S3-015
-- Framework: GHG Protocol Scope 3 Standard, PCAF Global Standard, TCFD, NZBA, CSRD,
--            CDP, SBTi Financial Sector, ISO 14064-1, GRI 305
-- Created: 2026-02-28
-- =====================================================================================
-- Schema: investments_service
-- Tables: 22 (10 reference + 9 operational + 3 supporting)
-- Hypertables: 3 (calculations 7d, compliance_checks 30d, aggregations 30d)
-- Continuous Aggregates: 2 (daily_financed_emissions, monthly_portfolio_summary)
-- Indexes: ~80
-- Seed Data: 200+ records (12 GICS sectors, 50+ countries, 8 PCAF asset classes,
--            building benchmarks, vehicle EFs, EEIO factors, currency rates,
--            sovereign data, carbon benchmarks, PCAF quality criteria)
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS investments_service;

COMMENT ON SCHEMA investments_service IS 'AGENT-MRV-028: Investments - Scope 3 Category 15 financed emissions calculations (PCAF attribution, building EUI, vehicle-specific, sovereign PPP)';

-- =====================================================================================
-- TABLE 1: gl_inv_sector_emission_factors
-- Description: Emission intensity factors by GICS sector for equity/bond estimation
-- Sources: MSCI, S&P Trucost, CDP, PCAF database
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_sector_emission_factors (
    id SERIAL PRIMARY KEY,
    sector VARCHAR(50) NOT NULL,
    gics_code VARCHAR(10) NOT NULL,
    ef_tco2e_per_m_revenue DECIMAL(12,4) NOT NULL,
    source VARCHAR(50) NOT NULL,
    year INT NOT NULL DEFAULT 2023,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(sector, gics_code, source, year),
    CONSTRAINT chk_inv_sec_ef_positive CHECK (ef_tco2e_per_m_revenue >= 0),
    CONSTRAINT chk_inv_sec_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_inv_sec_ef_sector ON investments_service.gl_inv_sector_emission_factors(sector);
CREATE INDEX idx_inv_sec_ef_gics ON investments_service.gl_inv_sector_emission_factors(gics_code);
CREATE INDEX idx_inv_sec_ef_source ON investments_service.gl_inv_sector_emission_factors(source);
CREATE INDEX idx_inv_sec_ef_active ON investments_service.gl_inv_sector_emission_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_sector_emission_factors IS 'GICS sector emission intensity factors for equity and bond PCAF attribution';
COMMENT ON COLUMN investments_service.gl_inv_sector_emission_factors.ef_tco2e_per_m_revenue IS 'tCO2e per million USD revenue (sector average)';

-- =====================================================================================
-- TABLE 2: gl_inv_country_emission_factors
-- Description: National GHG emissions, GDP PPP, and per-capita data for sovereign bonds
-- Sources: UNFCCC, World Bank, IEA, national inventories
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_country_emission_factors (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(3) NOT NULL,
    country_name VARCHAR(100) NOT NULL,
    total_ghg_mt DECIMAL(12,2) NOT NULL,
    gdp_ppp_billion_usd DECIMAL(14,2) NOT NULL,
    per_capita_tco2e DECIMAL(8,4) NOT NULL,
    lulucf_mt DECIMAL(12,2),
    source VARCHAR(50) NOT NULL,
    year INT NOT NULL DEFAULT 2022,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(country_code, source, year),
    CONSTRAINT chk_inv_ctry_ghg_positive CHECK (total_ghg_mt >= 0),
    CONSTRAINT chk_inv_ctry_gdp_positive CHECK (gdp_ppp_billion_usd > 0),
    CONSTRAINT chk_inv_ctry_percap_positive CHECK (per_capita_tco2e >= 0),
    CONSTRAINT chk_inv_ctry_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_inv_ctry_ef_code ON investments_service.gl_inv_country_emission_factors(country_code);
CREATE INDEX idx_inv_ctry_ef_name ON investments_service.gl_inv_country_emission_factors(country_name);
CREATE INDEX idx_inv_ctry_ef_source ON investments_service.gl_inv_country_emission_factors(source);
CREATE INDEX idx_inv_ctry_ef_year ON investments_service.gl_inv_country_emission_factors(year);
CREATE INDEX idx_inv_ctry_ef_active ON investments_service.gl_inv_country_emission_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_country_emission_factors IS 'National GHG emissions and GDP data for sovereign bond PCAF attribution';
COMMENT ON COLUMN investments_service.gl_inv_country_emission_factors.total_ghg_mt IS 'Total national GHG emissions in MtCO2e (excl LULUCF)';
COMMENT ON COLUMN investments_service.gl_inv_country_emission_factors.gdp_ppp_billion_usd IS 'GDP at purchasing power parity in billion USD';

-- =====================================================================================
-- TABLE 3: gl_inv_grid_emission_factors
-- Description: Grid electricity emission factors for CRE/mortgage building calculations
-- Sources: eGRID 2022, IEA 2023
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_grid_emission_factors (
    id SERIAL PRIMARY KEY,
    country VARCHAR(3) NOT NULL,
    region VARCHAR(50),
    ef_kgco2e_per_kwh DECIMAL(12,6) NOT NULL,
    source VARCHAR(50) NOT NULL,
    year INT NOT NULL DEFAULT 2022,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(country, region, source, year),
    CONSTRAINT chk_inv_grid_ef_positive CHECK (ef_kgco2e_per_kwh >= 0),
    CONSTRAINT chk_inv_grid_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_inv_grid_ef_country ON investments_service.gl_inv_grid_emission_factors(country);
CREATE INDEX idx_inv_grid_ef_region ON investments_service.gl_inv_grid_emission_factors(region);
CREATE INDEX idx_inv_grid_ef_source ON investments_service.gl_inv_grid_emission_factors(source);
CREATE INDEX idx_inv_grid_ef_active ON investments_service.gl_inv_grid_emission_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_grid_emission_factors IS 'Grid electricity emission factors for building-based financed emissions (CRE, mortgage)';

-- =====================================================================================
-- TABLE 4: gl_inv_building_benchmarks
-- Description: Building EUI benchmarks by property type and climate zone
-- Sources: ENERGY STAR Portfolio Manager, CBECS 2018, ASHRAE 90.1
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_building_benchmarks (
    id SERIAL PRIMARY KEY,
    property_type VARCHAR(30) NOT NULL,
    climate_zone VARCHAR(20),
    eui_kwh_per_m2 DECIMAL(10,4) NOT NULL,
    source VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(property_type, climate_zone, source),
    CONSTRAINT chk_inv_bldg_eui_positive CHECK (eui_kwh_per_m2 > 0)
);

CREATE INDEX idx_inv_bldg_bench_type ON investments_service.gl_inv_building_benchmarks(property_type);
CREATE INDEX idx_inv_bldg_bench_zone ON investments_service.gl_inv_building_benchmarks(climate_zone);
CREATE INDEX idx_inv_bldg_bench_source ON investments_service.gl_inv_building_benchmarks(source);
CREATE INDEX idx_inv_bldg_bench_active ON investments_service.gl_inv_building_benchmarks(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_building_benchmarks IS 'Building EUI benchmarks for CRE and mortgage financed emissions estimation';
COMMENT ON COLUMN investments_service.gl_inv_building_benchmarks.eui_kwh_per_m2 IS 'Energy Use Intensity in kWh per square metre per year';

-- =====================================================================================
-- TABLE 5: gl_inv_vehicle_emission_factors
-- Description: Vehicle emission factors by category and fuel type for auto loan portfolios
-- Sources: EPA, DEFRA 2024, ICCT
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_vehicle_emission_factors (
    id SERIAL PRIMARY KEY,
    vehicle_category VARCHAR(30) NOT NULL,
    fuel_type VARCHAR(20) NOT NULL,
    annual_emissions_kgco2e DECIMAL(10,2) NOT NULL,
    avg_distance_km INT NOT NULL DEFAULT 15000,
    ef_kgco2e_per_km DECIMAL(10,6) NOT NULL,
    source VARCHAR(50) NOT NULL DEFAULT 'DEFRA_2024',
    year INT NOT NULL DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(vehicle_category, fuel_type, source),
    CONSTRAINT chk_inv_veh_emissions_positive CHECK (annual_emissions_kgco2e >= 0),
    CONSTRAINT chk_inv_veh_distance_positive CHECK (avg_distance_km > 0),
    CONSTRAINT chk_inv_veh_ef_positive CHECK (ef_kgco2e_per_km >= 0)
);

CREATE INDEX idx_inv_veh_ef_category ON investments_service.gl_inv_vehicle_emission_factors(vehicle_category);
CREATE INDEX idx_inv_veh_ef_fuel ON investments_service.gl_inv_vehicle_emission_factors(fuel_type);
CREATE INDEX idx_inv_veh_ef_source ON investments_service.gl_inv_vehicle_emission_factors(source);
CREATE INDEX idx_inv_veh_ef_active ON investments_service.gl_inv_vehicle_emission_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_vehicle_emission_factors IS 'Vehicle emission factors for motor vehicle loan PCAF attribution';

-- =====================================================================================
-- TABLE 6: gl_inv_eeio_sector_factors
-- Description: EEIO-based emission factors for spend-based estimation
-- Sources: EPA USEEIO v2.1, EXIOBASE 3
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_eeio_sector_factors (
    id SERIAL PRIMARY KEY,
    sector VARCHAR(50) NOT NULL,
    gics_code VARCHAR(10) NOT NULL,
    ef_kgco2e_per_dollar DECIMAL(12,8) NOT NULL,
    source VARCHAR(50) NOT NULL,
    base_year INT DEFAULT 2021,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(sector, gics_code, source),
    CONSTRAINT chk_inv_eeio_ef_positive CHECK (ef_kgco2e_per_dollar >= 0),
    CONSTRAINT chk_inv_eeio_year_valid CHECK (base_year >= 1990 AND base_year <= 2100)
);

CREATE INDEX idx_inv_eeio_sector ON investments_service.gl_inv_eeio_sector_factors(sector);
CREATE INDEX idx_inv_eeio_gics ON investments_service.gl_inv_eeio_sector_factors(gics_code);
CREATE INDEX idx_inv_eeio_source ON investments_service.gl_inv_eeio_sector_factors(source);
CREATE INDEX idx_inv_eeio_active ON investments_service.gl_inv_eeio_sector_factors(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_eeio_sector_factors IS 'EEIO emission factors for spend-based financed emissions estimation';

-- =====================================================================================
-- TABLE 7: gl_inv_pcaf_data_quality
-- Description: PCAF data quality scoring criteria by asset class (scores 1-5)
-- Sources: PCAF Global GHG Standard 2nd Edition (2022)
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_pcaf_data_quality (
    id SERIAL PRIMARY KEY,
    asset_class VARCHAR(40) NOT NULL,
    score INT NOT NULL CHECK(score BETWEEN 1 AND 5),
    description TEXT NOT NULL,
    uncertainty_pct DECIMAL(5,2) NOT NULL,
    data_requirements TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(asset_class, score),
    CONSTRAINT chk_inv_pcaf_uncertainty_positive CHECK (uncertainty_pct >= 0)
);

CREATE INDEX idx_inv_pcaf_dq_class ON investments_service.gl_inv_pcaf_data_quality(asset_class);
CREATE INDEX idx_inv_pcaf_dq_score ON investments_service.gl_inv_pcaf_data_quality(score);
CREATE INDEX idx_inv_pcaf_dq_active ON investments_service.gl_inv_pcaf_data_quality(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_pcaf_data_quality IS 'PCAF data quality scoring criteria (1=highest to 5=lowest) per asset class';

-- =====================================================================================
-- TABLE 8: gl_inv_currency_rates
-- Description: Currency exchange rates to USD for multi-currency portfolios
-- Sources: World Bank, IMF, ECB
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_currency_rates (
    id SERIAL PRIMARY KEY,
    currency_code VARCHAR(3) NOT NULL,
    to_usd_rate DECIMAL(14,6) NOT NULL,
    year INT NOT NULL DEFAULT 2024,
    source VARCHAR(50) NOT NULL DEFAULT 'WORLD_BANK',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(currency_code, year, source),
    CONSTRAINT chk_inv_fx_rate_positive CHECK (to_usd_rate > 0),
    CONSTRAINT chk_inv_fx_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_inv_fx_currency ON investments_service.gl_inv_currency_rates(currency_code);
CREATE INDEX idx_inv_fx_year ON investments_service.gl_inv_currency_rates(year);
CREATE INDEX idx_inv_fx_active ON investments_service.gl_inv_currency_rates(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_currency_rates IS 'Currency exchange rates for multi-currency portfolio aggregation';

-- =====================================================================================
-- TABLE 9: gl_inv_sovereign_data
-- Description: Extended sovereign data for sovereign bond calculations
-- Sources: UNFCCC, World Bank, IMF WEO
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_sovereign_data (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(3) NOT NULL,
    country_name VARCHAR(100) NOT NULL,
    gdp_ppp_billion DECIMAL(14,2) NOT NULL,
    total_emissions_mt DECIMAL(12,2) NOT NULL,
    per_capita_tco2e DECIMAL(8,4) NOT NULL,
    population_million DECIMAL(10,2) NOT NULL,
    income_group VARCHAR(30),
    world_region VARCHAR(50),
    year INT NOT NULL DEFAULT 2022,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(country_code, year),
    CONSTRAINT chk_inv_sov_gdp_positive CHECK (gdp_ppp_billion > 0),
    CONSTRAINT chk_inv_sov_emissions_positive CHECK (total_emissions_mt >= 0),
    CONSTRAINT chk_inv_sov_pop_positive CHECK (population_million > 0)
);

CREATE INDEX idx_inv_sov_code ON investments_service.gl_inv_sovereign_data(country_code);
CREATE INDEX idx_inv_sov_region ON investments_service.gl_inv_sovereign_data(world_region);
CREATE INDEX idx_inv_sov_income ON investments_service.gl_inv_sovereign_data(income_group);
CREATE INDEX idx_inv_sov_year ON investments_service.gl_inv_sovereign_data(year);
CREATE INDEX idx_inv_sov_active ON investments_service.gl_inv_sovereign_data(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_sovereign_data IS 'Extended sovereign data for sovereign bond PCAF attribution calculations';

-- =====================================================================================
-- TABLE 10: gl_inv_carbon_benchmarks
-- Description: Sector carbon intensity benchmarks and Paris-aligned targets
-- Sources: SBTi, TPI, IEA NZE 2050
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_carbon_benchmarks (
    id SERIAL PRIMARY KEY,
    sector VARCHAR(50) NOT NULL,
    intensity_tco2e_per_m DECIMAL(12,4) NOT NULL,
    aligned_1_5c_target DECIMAL(12,4) NOT NULL,
    aligned_2c_target DECIMAL(12,4) NOT NULL,
    target_year INT NOT NULL DEFAULT 2030,
    source VARCHAR(50) NOT NULL DEFAULT 'SBTi',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(sector, target_year, source),
    CONSTRAINT chk_inv_bench_intensity_positive CHECK (intensity_tco2e_per_m >= 0),
    CONSTRAINT chk_inv_bench_15c_positive CHECK (aligned_1_5c_target >= 0),
    CONSTRAINT chk_inv_bench_2c_positive CHECK (aligned_2c_target >= 0)
);

CREATE INDEX idx_inv_bench_sector ON investments_service.gl_inv_carbon_benchmarks(sector);
CREATE INDEX idx_inv_bench_year ON investments_service.gl_inv_carbon_benchmarks(target_year);
CREATE INDEX idx_inv_bench_source ON investments_service.gl_inv_carbon_benchmarks(source);
CREATE INDEX idx_inv_bench_active ON investments_service.gl_inv_carbon_benchmarks(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_carbon_benchmarks IS 'Sector carbon intensity benchmarks and Paris-aligned targets for portfolio alignment';

-- =====================================================================================
-- TABLE 11: gl_inv_calculations (HYPERTABLE - 7-day chunks)
-- Description: Main financed emissions calculation results
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_calculations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset_class VARCHAR(40),
    method VARCHAR(30) NOT NULL,
    total_financed_emissions_kgco2e DECIMAL(18,6) NOT NULL,
    pcaf_data_quality INT,
    attribution_factor DECIMAL(12,8),
    outstanding_amount DECIMAL(18,2),
    company_name VARCHAR(300),
    sector VARCHAR(50),
    country VARCHAR(3),
    reporting_period VARCHAR(20),
    reporting_year INT,
    ef_source VARCHAR(100),
    currency VARCHAR(3) DEFAULT 'USD',
    metadata JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64),
    is_deleted BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (id, calculated_at),
    CONSTRAINT chk_inv_calc_emissions_positive CHECK (total_financed_emissions_kgco2e >= 0),
    CONSTRAINT chk_inv_calc_method CHECK (method IN (
        'pcaf_attribution', 'investment_specific', 'average_data',
        'spend_based', 'building_eui', 'vehicle_specific', 'sovereign_ppp'
    )),
    CONSTRAINT chk_inv_calc_pcaf_range CHECK (pcaf_data_quality IS NULL OR (pcaf_data_quality >= 1 AND pcaf_data_quality <= 5)),
    CONSTRAINT chk_inv_calc_attr_range CHECK (attribution_factor IS NULL OR (attribution_factor >= 0 AND attribution_factor <= 1))
);

SELECT create_hypertable('investments_service.gl_inv_calculations', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_inv_calc_tenant ON investments_service.gl_inv_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_inv_calc_asset ON investments_service.gl_inv_calculations(asset_class);
CREATE INDEX idx_inv_calc_method ON investments_service.gl_inv_calculations(method);
CREATE INDEX idx_inv_calc_sector ON investments_service.gl_inv_calculations(sector);
CREATE INDEX idx_inv_calc_country ON investments_service.gl_inv_calculations(country);
CREATE INDEX idx_inv_calc_period ON investments_service.gl_inv_calculations(reporting_period);
CREATE INDEX idx_inv_calc_year ON investments_service.gl_inv_calculations(reporting_year);
CREATE INDEX idx_inv_calc_hash ON investments_service.gl_inv_calculations(provenance_hash);
CREATE INDEX idx_inv_calc_deleted ON investments_service.gl_inv_calculations(is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX idx_inv_calc_tenant_asset ON investments_service.gl_inv_calculations(tenant_id, asset_class, calculated_at DESC);
CREATE INDEX idx_inv_calc_tenant_sector ON investments_service.gl_inv_calculations(tenant_id, sector, calculated_at DESC);

COMMENT ON TABLE investments_service.gl_inv_calculations IS 'Main financed emissions calculation results (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN investments_service.gl_inv_calculations.method IS 'Calculation method: pcaf_attribution, investment_specific, average_data, spend_based, building_eui, vehicle_specific, sovereign_ppp';
COMMENT ON COLUMN investments_service.gl_inv_calculations.pcaf_data_quality IS 'PCAF data quality score (1=highest to 5=lowest)';

-- =====================================================================================
-- TABLE 12: gl_inv_investment_results
-- Description: Per-position investment calculation details
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_investment_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    investment_id VARCHAR(100) NOT NULL,
    asset_class VARCHAR(40) NOT NULL,
    company_name VARCHAR(300),
    financed_emissions DECIMAL(18,6) NOT NULL,
    attribution_factor DECIMAL(12,8) NOT NULL,
    pcaf_score INT,
    carbon_intensity DECIMAL(14,6),
    outstanding_amount DECIMAL(18,2),
    sector VARCHAR(50),
    country VARCHAR(3),
    scope1_tco2e DECIMAL(18,6),
    scope2_tco2e DECIMAL(18,6),
    scope3_tco2e DECIMAL(18,6),
    data_source VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_inv_result_emissions_positive CHECK (financed_emissions >= 0),
    CONSTRAINT chk_inv_result_attr_range CHECK (attribution_factor >= 0 AND attribution_factor <= 1),
    CONSTRAINT chk_inv_result_pcaf_range CHECK (pcaf_score IS NULL OR (pcaf_score >= 1 AND pcaf_score <= 5))
);

CREATE INDEX idx_inv_result_calc_id ON investments_service.gl_inv_investment_results(calculation_id);
CREATE INDEX idx_inv_result_inv_id ON investments_service.gl_inv_investment_results(investment_id);
CREATE INDEX idx_inv_result_asset ON investments_service.gl_inv_investment_results(asset_class);
CREATE INDEX idx_inv_result_sector ON investments_service.gl_inv_investment_results(sector);
CREATE INDEX idx_inv_result_country ON investments_service.gl_inv_investment_results(country);
CREATE INDEX idx_inv_result_pcaf ON investments_service.gl_inv_investment_results(pcaf_score);

COMMENT ON TABLE investments_service.gl_inv_investment_results IS 'Per-position financed emissions results linked to parent calculation';

-- =====================================================================================
-- TABLE 13: gl_inv_portfolio_aggregations
-- Description: Portfolio-level aggregation with WACI and carbon metrics
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_portfolio_aggregations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    portfolio_name VARCHAR(200),
    total_financed_emissions DECIMAL(18,6) NOT NULL,
    total_aum DECIMAL(18,2) NOT NULL,
    waci DECIMAL(14,6),
    carbon_footprint DECIMAL(14,6),
    position_count INT NOT NULL DEFAULT 0,
    coverage_ratio DECIMAL(5,4),
    weighted_pcaf_score DECIMAL(3,2),
    by_asset_class JSONB DEFAULT '{}',
    by_sector JSONB DEFAULT '{}',
    by_country JSONB DEFAULT '{}',
    consolidation_approach VARCHAR(30),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_inv_port_emissions_positive CHECK (total_financed_emissions >= 0),
    CONSTRAINT chk_inv_port_aum_positive CHECK (total_aum > 0),
    CONSTRAINT chk_inv_port_coverage_range CHECK (coverage_ratio IS NULL OR (coverage_ratio >= 0 AND coverage_ratio <= 1))
);

CREATE INDEX idx_inv_port_agg_calc_id ON investments_service.gl_inv_portfolio_aggregations(calculation_id);
CREATE INDEX idx_inv_port_agg_name ON investments_service.gl_inv_portfolio_aggregations(portfolio_name);
CREATE INDEX idx_inv_port_agg_waci ON investments_service.gl_inv_portfolio_aggregations(waci);

COMMENT ON TABLE investments_service.gl_inv_portfolio_aggregations IS 'Portfolio-level financed emissions aggregation with WACI and carbon metrics';

-- =====================================================================================
-- TABLE 14: gl_inv_compliance_checks (HYPERTABLE - 30-day chunks)
-- Description: Multi-framework compliance check results
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_compliance_checks (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    calculation_id UUID,
    tenant_id UUID NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    framework VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL,
    score DECIMAL(5,4),
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    pcaf_score_met BOOLEAN,
    attribution_valid BOOLEAN,
    coverage_sufficient BOOLEAN,
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (id, checked_at),
    CONSTRAINT chk_inv_comp_status CHECK (status IN ('pass', 'fail', 'warning', 'not_applicable')),
    CONSTRAINT chk_inv_comp_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 1)),
    CONSTRAINT chk_inv_comp_framework CHECK (framework IN (
        'ghg_protocol', 'pcaf', 'tcfd', 'nzba', 'iso_14064', 'csrd_esrs', 'cdp', 'sbti', 'gri'
    ))
);

SELECT create_hypertable('investments_service.gl_inv_compliance_checks', 'checked_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_inv_comp_calc_id ON investments_service.gl_inv_compliance_checks(calculation_id);
CREATE INDEX idx_inv_comp_tenant ON investments_service.gl_inv_compliance_checks(tenant_id, checked_at DESC);
CREATE INDEX idx_inv_comp_framework ON investments_service.gl_inv_compliance_checks(framework);
CREATE INDEX idx_inv_comp_status ON investments_service.gl_inv_compliance_checks(status);

COMMENT ON TABLE investments_service.gl_inv_compliance_checks IS 'Multi-framework compliance check results (HYPERTABLE, 30-day chunks)';

-- =====================================================================================
-- TABLE 15: gl_inv_aggregations (HYPERTABLE - 30-day chunks)
-- Description: Time-series aggregated financed emissions by period
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_aggregations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ,
    period_type VARCHAR(20) NOT NULL,
    total_financed_emissions DECIMAL(18,6) NOT NULL,
    total_aum DECIMAL(18,2),
    waci DECIMAL(14,6),
    position_count INT DEFAULT 0,
    by_asset_class JSONB DEFAULT '{}',
    by_sector JSONB DEFAULT '{}',
    by_country JSONB DEFAULT '{}',
    weighted_pcaf_score DECIMAL(3,2),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, period_start),
    CONSTRAINT chk_inv_agg_emissions_positive CHECK (total_financed_emissions >= 0),
    CONSTRAINT chk_inv_agg_period_type CHECK (period_type IN ('daily', 'weekly', 'monthly', 'quarterly', 'annual'))
);

SELECT create_hypertable('investments_service.gl_inv_aggregations', 'period_start',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_inv_agg_tenant ON investments_service.gl_inv_aggregations(tenant_id, period_start DESC);
CREATE INDEX idx_inv_agg_period_type ON investments_service.gl_inv_aggregations(period_type);
CREATE INDEX idx_inv_agg_period_range ON investments_service.gl_inv_aggregations(period_start, period_end);

COMMENT ON TABLE investments_service.gl_inv_aggregations IS 'Time-series aggregated financed emissions (HYPERTABLE, 30-day chunks)';

-- =====================================================================================
-- TABLE 16: gl_inv_provenance_records
-- Description: SHA-256 provenance chain records for audit trail
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_provenance_records (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    record_type VARCHAR(50) NOT NULL,
    record_id UUID,
    stage_name VARCHAR(50),
    sha256_hash VARCHAR(64) NOT NULL,
    parent_hash VARCHAR(64),
    input_snapshot JSONB,
    output_snapshot JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inv_prov_record_id ON investments_service.gl_inv_provenance_records(record_id);
CREATE INDEX idx_inv_prov_type ON investments_service.gl_inv_provenance_records(record_type);
CREATE INDEX idx_inv_prov_hash ON investments_service.gl_inv_provenance_records(sha256_hash);
CREATE INDEX idx_inv_prov_parent ON investments_service.gl_inv_provenance_records(parent_hash);
CREATE INDEX idx_inv_prov_created ON investments_service.gl_inv_provenance_records(created_at);

COMMENT ON TABLE investments_service.gl_inv_provenance_records IS 'SHA-256 provenance chain records for financed emissions audit trail';

-- =====================================================================================
-- TABLE 17: gl_inv_audit_trail
-- Description: Audit trail for all investment operations
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_audit_trail (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    operation VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    actor VARCHAR(200),
    details JSONB DEFAULT '{}',
    ip_address VARCHAR(45),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inv_audit_tenant ON investments_service.gl_inv_audit_trail(tenant_id, created_at DESC);
CREATE INDEX idx_inv_audit_operation ON investments_service.gl_inv_audit_trail(operation);
CREATE INDEX idx_inv_audit_entity ON investments_service.gl_inv_audit_trail(entity_type, entity_id);
CREATE INDEX idx_inv_audit_created ON investments_service.gl_inv_audit_trail(created_at);

COMMENT ON TABLE investments_service.gl_inv_audit_trail IS 'Audit trail for all investment financed emissions operations';

-- =====================================================================================
-- TABLE 18: gl_inv_batch_jobs
-- Description: Batch processing job tracking
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_batch_jobs (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    total_positions INT NOT NULL DEFAULT 0,
    processed_positions INT NOT NULL DEFAULT 0,
    failed_positions INT NOT NULL DEFAULT 0,
    total_financed_emissions DECIMAL(18,6),
    error_details JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    CONSTRAINT chk_inv_batch_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_inv_batch_positions_positive CHECK (total_positions >= 0 AND processed_positions >= 0 AND failed_positions >= 0)
);

CREATE INDEX idx_inv_batch_tenant ON investments_service.gl_inv_batch_jobs(tenant_id, created_at DESC);
CREATE INDEX idx_inv_batch_status ON investments_service.gl_inv_batch_jobs(status);
CREATE INDEX idx_inv_batch_created ON investments_service.gl_inv_batch_jobs(created_at);

COMMENT ON TABLE investments_service.gl_inv_batch_jobs IS 'Batch processing job tracking for financed emissions calculations (up to 50,000 positions)';

-- =====================================================================================
-- TABLE 19: gl_inv_portfolio_positions
-- Description: Stored portfolio positions for ongoing monitoring
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_portfolio_positions (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    portfolio_id UUID,
    portfolio_name VARCHAR(200),
    isin VARCHAR(12),
    investment_id VARCHAR(100),
    asset_class VARCHAR(40) NOT NULL,
    outstanding_amount DECIMAL(18,2) NOT NULL,
    sector VARCHAR(50),
    country VARCHAR(3),
    company_name VARCHAR(300),
    evic DECIMAL(18,2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_inv_pos_amount_positive CHECK (outstanding_amount > 0)
);

CREATE INDEX idx_inv_pos_tenant ON investments_service.gl_inv_portfolio_positions(tenant_id);
CREATE INDEX idx_inv_pos_portfolio ON investments_service.gl_inv_portfolio_positions(portfolio_id);
CREATE INDEX idx_inv_pos_isin ON investments_service.gl_inv_portfolio_positions(isin);
CREATE INDEX idx_inv_pos_asset ON investments_service.gl_inv_portfolio_positions(asset_class);
CREATE INDEX idx_inv_pos_sector ON investments_service.gl_inv_portfolio_positions(sector);
CREATE INDEX idx_inv_pos_country ON investments_service.gl_inv_portfolio_positions(country);
CREATE INDEX idx_inv_pos_active ON investments_service.gl_inv_portfolio_positions(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE investments_service.gl_inv_portfolio_positions IS 'Stored portfolio positions for ongoing financed emissions monitoring';

-- =====================================================================================
-- TABLE 20: gl_inv_data_quality_scores
-- Description: Multi-dimensional data quality assessment scores
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_data_quality_scores (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    pcaf_score INT NOT NULL,
    temporal DECIMAL(3,2) NOT NULL,
    geographical DECIMAL(3,2) NOT NULL,
    technological DECIMAL(3,2) NOT NULL,
    completeness DECIMAL(3,2) NOT NULL,
    reliability DECIMAL(3,2) NOT NULL,
    methodology VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_inv_dqs_pcaf CHECK (pcaf_score >= 1 AND pcaf_score <= 5),
    CONSTRAINT chk_inv_dqs_temporal CHECK (temporal >= 1.0 AND temporal <= 5.0),
    CONSTRAINT chk_inv_dqs_geographical CHECK (geographical >= 1.0 AND geographical <= 5.0),
    CONSTRAINT chk_inv_dqs_technological CHECK (technological >= 1.0 AND technological <= 5.0),
    CONSTRAINT chk_inv_dqs_completeness CHECK (completeness >= 1.0 AND completeness <= 5.0),
    CONSTRAINT chk_inv_dqs_reliability CHECK (reliability >= 1.0 AND reliability <= 5.0)
);

CREATE INDEX idx_inv_dqs_calc_id ON investments_service.gl_inv_data_quality_scores(calculation_id);
CREATE INDEX idx_inv_dqs_pcaf ON investments_service.gl_inv_data_quality_scores(pcaf_score);

COMMENT ON TABLE investments_service.gl_inv_data_quality_scores IS 'Multi-dimensional data quality assessment with PCAF scoring';

-- =====================================================================================
-- TABLE 21: gl_inv_uncertainty_results
-- Description: Uncertainty analysis results for financed emissions
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_uncertainty_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    method VARCHAR(30) NOT NULL,
    mean_kgco2e DECIMAL(18,6),
    lower_bound DECIMAL(18,6) NOT NULL,
    upper_bound DECIMAL(18,6) NOT NULL,
    confidence_level DECIMAL(3,2) NOT NULL DEFAULT 0.95,
    iterations INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_inv_unc_bounds CHECK (upper_bound >= lower_bound),
    CONSTRAINT chk_inv_unc_confidence CHECK (confidence_level > 0 AND confidence_level < 1),
    CONSTRAINT chk_inv_unc_method CHECK (method IN ('monte_carlo', 'analytical', 'pcaf_uncertainty'))
);

CREATE INDEX idx_inv_unc_calc_id ON investments_service.gl_inv_uncertainty_results(calculation_id);
CREATE INDEX idx_inv_unc_method ON investments_service.gl_inv_uncertainty_results(method);

COMMENT ON TABLE investments_service.gl_inv_uncertainty_results IS 'Uncertainty analysis results for financed emissions calculations';

-- =====================================================================================
-- TABLE 22: gl_inv_carbon_intensity
-- Description: Portfolio carbon intensity metrics (WACI, revenue, financed, physical)
-- =====================================================================================

CREATE TABLE investments_service.gl_inv_carbon_intensity (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    waci DECIMAL(14,6),
    revenue_intensity DECIMAL(14,6),
    financed_intensity DECIMAL(14,6),
    physical_intensity DECIMAL(14,6),
    by_sector JSONB DEFAULT '{}',
    by_asset_class JSONB DEFAULT '{}',
    benchmark_comparison JSONB,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inv_ci_calc_id ON investments_service.gl_inv_carbon_intensity(calculation_id);
CREATE INDEX idx_inv_ci_waci ON investments_service.gl_inv_carbon_intensity(waci);

COMMENT ON TABLE investments_service.gl_inv_carbon_intensity IS 'Portfolio carbon intensity metrics including WACI, revenue, financed, and physical intensity';

-- =====================================================================================
-- CONTINUOUS AGGREGATE 1: Daily financed emissions summary
-- =====================================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS investments_service.gl_inv_daily_financed_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    tenant_id,
    asset_class,
    method,
    COUNT(*) AS calculation_count,
    SUM(total_financed_emissions_kgco2e) AS total_emissions_kgco2e,
    AVG(total_financed_emissions_kgco2e) AS avg_emissions_kgco2e,
    AVG(pcaf_data_quality) AS avg_pcaf_score,
    SUM(outstanding_amount) AS total_outstanding
FROM investments_service.gl_inv_calculations
WHERE is_deleted = FALSE
GROUP BY bucket, tenant_id, asset_class, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy('investments_service.gl_inv_daily_financed_emissions',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW investments_service.gl_inv_daily_financed_emissions IS 'Daily financed emissions summary continuous aggregate';

-- =====================================================================================
-- CONTINUOUS AGGREGATE 2: Monthly portfolio summary
-- =====================================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS investments_service.gl_inv_monthly_portfolio_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('30 days', calculated_at) AS bucket,
    tenant_id,
    COUNT(*) AS calculation_count,
    SUM(total_financed_emissions_kgco2e) AS total_emissions_kgco2e,
    AVG(pcaf_data_quality) AS avg_pcaf_score,
    SUM(outstanding_amount) AS total_outstanding,
    COUNT(DISTINCT asset_class) AS asset_class_count,
    COUNT(DISTINCT sector) AS sector_count,
    COUNT(DISTINCT country) AS country_count
FROM investments_service.gl_inv_calculations
WHERE is_deleted = FALSE
GROUP BY bucket, tenant_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('investments_service.gl_inv_monthly_portfolio_summary',
    start_offset => INTERVAL '60 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW investments_service.gl_inv_monthly_portfolio_summary IS 'Monthly portfolio financed emissions summary continuous aggregate';

-- =====================================================================================
-- SEED DATA: SECTOR EMISSION FACTORS (12 GICS sectors)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_sector_emission_factors
(sector, gics_code, ef_tco2e_per_m_revenue, source, year) VALUES
('energy',                   '10', 850.4200, 'MSCI_Trucost_2023', 2023),
('materials',                '15', 624.8100, 'MSCI_Trucost_2023', 2023),
('industrials',              '20', 182.3400, 'MSCI_Trucost_2023', 2023),
('consumer_discretionary',   '25',  78.5600, 'MSCI_Trucost_2023', 2023),
('consumer_staples',         '30', 142.7300, 'MSCI_Trucost_2023', 2023),
('health_care',              '35',  52.4100, 'MSCI_Trucost_2023', 2023),
('financials',               '40',  12.3500, 'MSCI_Trucost_2023', 2023),
('information_technology',   '45',  18.9200, 'MSCI_Trucost_2023', 2023),
('communication_services',   '50',  22.6700, 'MSCI_Trucost_2023', 2023),
('utilities',                '55', 1245.6800, 'MSCI_Trucost_2023', 2023),
('real_estate',              '60',  68.4300, 'MSCI_Trucost_2023', 2023),
('sovereign',                '99',   0.0000, 'N/A', 2023);

-- =====================================================================================
-- SEED DATA: COUNTRY EMISSION FACTORS (52 countries)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_country_emission_factors
(country_code, country_name, total_ghg_mt, gdp_ppp_billion_usd, per_capita_tco2e, lulucf_mt, source, year) VALUES
-- North America
('USA', 'United States',       5222.40, 25462.70, 15.5200, -762.00, 'UNFCCC_2023', 2022),
('CAN', 'Canada',               670.40,  2139.84, 17.2800, -178.00, 'UNFCCC_2023', 2022),
('MEX', 'Mexico',               492.30,  2616.90,  3.7600,  -68.00, 'UNFCCC_2023', 2022),
-- Europe
('GBR', 'United Kingdom',       341.20,  3376.00,  5.0500,  -16.80, 'UNFCCC_2023', 2022),
('DEU', 'Germany',              674.30,  4857.00,  8.0900,  -25.40, 'UNFCCC_2023', 2022),
('FRA', 'France',               299.70,  3677.00,  4.3900,  -33.20, 'UNFCCC_2023', 2022),
('ITA', 'Italy',                327.80,  2972.00,  5.5500,  -26.80, 'UNFCCC_2023', 2022),
('ESP', 'Spain',                245.60,  2201.00,  5.1800,  -28.40, 'UNFCCC_2023', 2022),
('NLD', 'Netherlands',          147.20,  1149.00,  8.4100,   -3.20, 'UNFCCC_2023', 2022),
('BEL', 'Belgium',               99.80,   685.00,  8.6200,   -1.40, 'UNFCCC_2023', 2022),
('AUT', 'Austria',               66.40,   572.00,  7.3800,  -10.20, 'UNFCCC_2023', 2022),
('CHE', 'Switzerland',           36.40,   649.00,  4.1800,   -2.60, 'UNFCCC_2023', 2022),
('SWE', 'Sweden',                38.80,   629.00,  3.7100,  -35.40, 'UNFCCC_2023', 2022),
('NOR', 'Norway',                41.20,   443.00,  7.5100,  -20.80, 'UNFCCC_2023', 2022),
('DNK', 'Denmark',               31.40,   390.00,  5.3500,   -4.60, 'UNFCCC_2023', 2022),
('FIN', 'Finland',               36.20,   304.00,  6.5200,  -18.40, 'UNFCCC_2023', 2022),
('IRL', 'Ireland',               58.60,   577.00, 11.5600,   -3.80, 'UNFCCC_2023', 2022),
('POL', 'Poland',               320.40,  1548.00,  8.4600,  -28.60, 'UNFCCC_2023', 2022),
('PRT', 'Portugal',              48.60,   405.00,  4.7200,   -8.40, 'UNFCCC_2023', 2022),
('GRC', 'Greece',                58.80,   361.00,  5.5800,   -3.20, 'UNFCCC_2023', 2022),
-- Asia-Pacific
('CHN', 'China',               11472.00, 30330.00,  8.1300, -540.00, 'IEA_2023', 2022),
('JPN', 'Japan',               1064.00,  5710.00,  8.5000,  -47.60, 'UNFCCC_2023', 2022),
('KOR', 'South Korea',          616.40,  2734.00, 11.9200,  -38.20, 'UNFCCC_2023', 2022),
('IND', 'India',               3347.00, 11870.00,  2.3500, -318.00, 'IEA_2023', 2022),
('AUS', 'Australia',            464.80,  1653.00, 17.8400, -162.40, 'UNFCCC_2023', 2022),
('NZL', 'New Zealand',           62.40,   248.00, 12.1200,  -23.60, 'UNFCCC_2023', 2022),
('SGP', 'Singapore',             47.80,   637.00,  8.2400,    0.00, 'IEA_2023', 2022),
('HKG', 'Hong Kong',             31.40,   504.00,  4.2100,    0.00, 'IEA_2023', 2022),
('TWN', 'Taiwan',               268.40,  1448.00, 11.4200,   -8.60, 'IEA_2023', 2022),
('IDN', 'Indonesia',            691.20,  3996.00,  2.5200, -322.00, 'IEA_2023', 2022),
('THA', 'Thailand',             261.40,  1390.00,  3.7200,  -62.40, 'IEA_2023', 2022),
('MYS', 'Malaysia',             254.20,  1063.00,  7.6200,  -84.20, 'IEA_2023', 2022),
('PHL', 'Philippines',          167.80,  1138.00,  1.5000,  -42.60, 'IEA_2023', 2022),
('VNM', 'Vietnam',              323.60,  1278.00,  3.2600,  -28.40, 'IEA_2023', 2022),
-- Middle East
('SAU', 'Saudi Arabia',         588.40,  1869.00, 16.4200,   -2.40, 'IEA_2023', 2022),
('ARE', 'United Arab Emirates', 193.60,   729.00, 19.4200,   -0.80, 'IEA_2023', 2022),
('QAT', 'Qatar',                 98.20,   265.00, 33.5800,   -0.20, 'IEA_2023', 2022),
-- South America
('BRA', 'Brazil',               478.20,  3780.00,  2.2200, -242.00, 'UNFCCC_2023', 2022),
('ARG', 'Argentina',            192.40,  1126.00,  4.2000,  -38.60, 'UNFCCC_2023', 2022),
('CHL', 'Chile',                 84.60,   523.00,  4.3200,  -52.40, 'UNFCCC_2023', 2022),
('COL', 'Colombia',             118.40,   960.00,  2.2800,  -48.60, 'UNFCCC_2023', 2022),
-- Africa
('ZAF', 'South Africa',        434.80,   943.00,  7.2200,  -18.40, 'IEA_2023', 2022),
('NGA', 'Nigeria',              126.40,  1268.00,  0.5800,  -62.40, 'IEA_2023', 2022),
('EGY', 'Egypt',                282.60,  1561.00,  2.6500,  -12.80, 'IEA_2023', 2022),
('KEN', 'Kenya',                 24.80,   300.00,  0.4600,  -12.40, 'IEA_2023', 2022),
('MAR', 'Morocco',               68.20,   346.00,  1.8200,   -8.60, 'IEA_2023', 2022),
-- Other
('RUS', 'Russia',              1580.40,  4365.00, 10.8200, -420.00, 'IEA_2023', 2022),
('TUR', 'Turkey',               418.60,  3212.00,  4.9200,  -42.80, 'IEA_2023', 2022),
('ISR', 'Israel',                68.40,   478.00,  7.2200,   -2.40, 'IEA_2023', 2022),
('PAK', 'Pakistan',             215.40,  1505.00,  0.9200,  -28.60, 'IEA_2023', 2022),
('BGD', 'Bangladesh',           108.60,  1266.00,  0.6200,  -14.20, 'IEA_2023', 2022),
('LKA', 'Sri Lanka',             23.40,   305.00,  1.0600,   -6.80, 'IEA_2023', 2022);

-- =====================================================================================
-- SEED DATA: GRID EMISSION FACTORS (30 records - eGRID + international)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_grid_emission_factors
(country, region, ef_kgco2e_per_kwh, source, year) VALUES
-- US eGRID subregions (2022)
('USA', 'AKGD',  0.437500, 'eGRID_2022', 2022),
('USA', 'AZNM',  0.401200, 'eGRID_2022', 2022),
('USA', 'CAMX',  0.228600, 'eGRID_2022', 2022),
('USA', 'ERCT',  0.373800, 'eGRID_2022', 2022),
('USA', 'FRCC',  0.376400, 'eGRID_2022', 2022),
('USA', 'MROE',  0.513200, 'eGRID_2022', 2022),
('USA', 'MROW',  0.431800, 'eGRID_2022', 2022),
('USA', 'NEWE',  0.219400, 'eGRID_2022', 2022),
('USA', 'NWPP',  0.268200, 'eGRID_2022', 2022),
('USA', 'NYCW',  0.242100, 'eGRID_2022', 2022),
('USA', 'NYLI',  0.315200, 'eGRID_2022', 2022),
('USA', 'NYUP',  0.135600, 'eGRID_2022', 2022),
('USA', 'RFCE',  0.310200, 'eGRID_2022', 2022),
('USA', 'RFCM',  0.548600, 'eGRID_2022', 2022),
('USA', 'RFCW',  0.504800, 'eGRID_2022', 2022),
('USA', 'RMPA',  0.468200, 'eGRID_2022', 2022),
('USA', 'SPNO',  0.456800, 'eGRID_2022', 2022),
('USA', 'SPSO',  0.408600, 'eGRID_2022', 2022),
('USA', 'SRMV',  0.354200, 'eGRID_2022', 2022),
('USA', 'SRMW',  0.618400, 'eGRID_2022', 2022),
('USA', 'SRSO',  0.381200, 'eGRID_2022', 2022),
('USA', 'SRTV',  0.396800, 'eGRID_2022', 2022),
('USA', 'SRVC',  0.322400, 'eGRID_2022', 2022),
-- US national average
('USA', NULL,    0.386100, 'eGRID_2022', 2022),
-- International (IEA 2023)
('GBR', NULL,    0.207000, 'IEA_2023', 2023),
('DEU', NULL,    0.338000, 'IEA_2023', 2023),
('FRA', NULL,    0.056000, 'IEA_2023', 2023),
('JPN', NULL,    0.457000, 'IEA_2023', 2023),
('AUS', NULL,    0.656000, 'IEA_2023', 2023),
('CAN', NULL,    0.120000, 'IEA_2023', 2023);

-- =====================================================================================
-- SEED DATA: BUILDING BENCHMARKS (12 property types)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_building_benchmarks
(property_type, climate_zone, eui_kwh_per_m2, source) VALUES
-- Commercial property types (ENERGY STAR / CBECS)
('office',        '4A',    256.0000, 'ENERGY_STAR'),
('office',        'mixed', 248.0000, 'ENERGY_STAR'),
('retail',        '4A',    312.0000, 'ENERGY_STAR'),
('retail',        'mixed', 298.0000, 'ENERGY_STAR'),
('industrial',    '4A',    178.0000, 'CBECS_2018'),
('industrial',    'mixed', 172.0000, 'CBECS_2018'),
('multifamily',   '4A',    198.0000, 'ENERGY_STAR'),
('multifamily',   'mixed', 192.0000, 'ENERGY_STAR'),
('hotel',         '4A',    362.0000, 'ENERGY_STAR'),
('hotel',         'mixed', 348.0000, 'ENERGY_STAR'),
('data_center',   '4A',   1248.0000, 'ENERGY_STAR'),
('data_center',   'mixed',1220.0000, 'ENERGY_STAR'),
('warehouse',     '4A',    112.0000, 'CBECS_2018'),
('warehouse',     'mixed', 108.0000, 'CBECS_2018'),
('hospital',      '4A',    586.0000, 'ENERGY_STAR'),
('hospital',      'mixed', 572.0000, 'ENERGY_STAR'),
-- Residential property types
('single_family', '4A',    148.0000, 'RECS_2020'),
('single_family', 'mixed', 142.0000, 'RECS_2020'),
('townhouse',     '4A',    128.0000, 'RECS_2020'),
('townhouse',     'mixed', 124.0000, 'RECS_2020'),
('apartment',     '4A',    112.0000, 'RECS_2020'),
('apartment',     'mixed', 108.0000, 'RECS_2020'),
('condo',         '4A',    118.0000, 'RECS_2020'),
('condo',         'mixed', 114.0000, 'RECS_2020');

-- =====================================================================================
-- SEED DATA: VEHICLE EMISSION FACTORS (15 vehicle categories)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_vehicle_emission_factors
(vehicle_category, fuel_type, annual_emissions_kgco2e, avg_distance_km, ef_kgco2e_per_km) VALUES
('passenger_car_small',  'gasoline', 2142.00, 15000, 0.142800),
('passenger_car_small',  'diesel',   1986.00, 15000, 0.132400),
('passenger_car_medium', 'gasoline', 2856.00, 15000, 0.190400),
('passenger_car_medium', 'diesel',   2571.00, 15000, 0.171400),
('passenger_car_large',  'gasoline', 3624.00, 15000, 0.241600),
('passenger_car_large',  'diesel',   3198.00, 15000, 0.213200),
('suv_small',            'gasoline', 3042.00, 15000, 0.202800),
('suv_large',            'gasoline', 4284.00, 15000, 0.285600),
('suv_large',            'diesel',   3852.00, 15000, 0.256800),
('light_truck',          'gasoline', 4572.00, 18000, 0.254000),
('light_truck',          'diesel',   4104.00, 18000, 0.228000),
('heavy_truck',          'diesel',  12852.00, 25000, 0.514080),
('motorcycle',           'gasoline',  984.00, 8000,  0.123000),
('electric_vehicle',     'electric',  462.00, 15000, 0.030800),
('hybrid',               'gasoline', 1686.00, 15000, 0.112400),
('plug_in_hybrid',       'gasoline', 1128.00, 15000, 0.075200);

-- =====================================================================================
-- SEED DATA: EEIO SECTOR FACTORS (12 GICS sectors)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_eeio_sector_factors
(sector, gics_code, ef_kgco2e_per_dollar, source, base_year) VALUES
('energy',                   '10', 0.85042000, 'EPA_USEEIO_v2.1', 2021),
('materials',                '15', 0.62481000, 'EPA_USEEIO_v2.1', 2021),
('industrials',              '20', 0.18234000, 'EPA_USEEIO_v2.1', 2021),
('consumer_discretionary',   '25', 0.07856000, 'EPA_USEEIO_v2.1', 2021),
('consumer_staples',         '30', 0.14273000, 'EPA_USEEIO_v2.1', 2021),
('health_care',              '35', 0.05241000, 'EPA_USEEIO_v2.1', 2021),
('financials',               '40', 0.01235000, 'EPA_USEEIO_v2.1', 2021),
('information_technology',   '45', 0.01892000, 'EPA_USEEIO_v2.1', 2021),
('communication_services',   '50', 0.02267000, 'EPA_USEEIO_v2.1', 2021),
('utilities',                '55', 1.24568000, 'EPA_USEEIO_v2.1', 2021),
('real_estate',              '60', 0.06843000, 'EPA_USEEIO_v2.1', 2021),
('sovereign',                '99', 0.00000000, 'N/A', 2021);

-- =====================================================================================
-- SEED DATA: PCAF DATA QUALITY CRITERIA (8 asset classes x 5 scores = 40 records)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_pcaf_data_quality
(asset_class, score, description, uncertainty_pct, data_requirements) VALUES
-- Listed Equity / Corporate Bonds
('listed_equity',   1, 'Reported emissions verified by third party', 10.00, 'Verified Scope 1+2 from company reporting, EVIC from financial data'),
('listed_equity',   2, 'Reported emissions not verified', 20.00, 'Unverified Scope 1+2 from company reporting, EVIC from financial data'),
('listed_equity',   3, 'Emissions estimated from physical activity data', 30.00, 'Physical activity data (energy, fuel), emission factors'),
('listed_equity',   4, 'Emissions estimated from economic activity data', 40.00, 'Revenue and sector-average emission factors'),
('listed_equity',   5, 'Emissions estimated from sector average', 50.00, 'Asset class and sector only'),
('corporate_bond',  1, 'Reported emissions verified by third party', 10.00, 'Verified Scope 1+2 from issuer reporting, EVIC from financial data'),
('corporate_bond',  2, 'Reported emissions not verified', 20.00, 'Unverified Scope 1+2 from issuer reporting'),
('corporate_bond',  3, 'Emissions estimated from physical activity data', 30.00, 'Physical activity data and emission factors'),
('corporate_bond',  4, 'Emissions estimated from economic activity data', 40.00, 'Revenue and sector-average emission factors'),
('corporate_bond',  5, 'Emissions estimated from sector average', 50.00, 'Sector and asset class only'),
-- Private Equity / Business Loans
('private_equity',  1, 'Reported emissions verified by third party', 15.00, 'Verified company emissions, audited equity+debt'),
('private_equity',  2, 'Reported emissions not verified', 25.00, 'Unverified company emissions'),
('private_equity',  3, 'Emissions from physical activity data', 35.00, 'Energy/fuel data with emission factors'),
('private_equity',  4, 'Estimated from revenue-based factors', 45.00, 'Revenue and sector EEIO factors'),
('private_equity',  5, 'Estimated from sector averages', 55.00, 'Sector and region estimates'),
-- Project Finance
('project_finance', 1, 'Direct measurement of project emissions', 10.00, 'CEMS data or measured project emissions'),
('project_finance', 2, 'Project-specific activity data', 20.00, 'Project energy/fuel consumption data'),
('project_finance', 3, 'Estimated from similar projects', 35.00, 'Benchmark data from comparable projects'),
('project_finance', 4, 'Estimated from project type averages', 45.00, 'Project type and capacity data'),
('project_finance', 5, 'Estimated from sector averages', 55.00, 'Sector-level average intensity'),
-- Commercial Real Estate
('commercial_real_estate', 1, 'Actual building energy consumption data', 10.00, 'Metered energy data, actual utility bills'),
('commercial_real_estate', 2, 'Partially actual building data', 20.00, 'Partial meter data plus estimates'),
('commercial_real_estate', 3, 'Building EPC + floor area', 30.00, 'Energy Performance Certificate and floor area'),
('commercial_real_estate', 4, 'Estimated from building type benchmarks', 40.00, 'Building type, location, floor area'),
('commercial_real_estate', 5, 'Estimated from sector average EUI', 50.00, 'Floor area only or estimated floor area'),
-- Mortgage
('mortgage', 1, 'Actual energy consumption from utility data', 10.00, 'Metered household energy data'),
('mortgage', 2, 'Energy from EPC + floor area', 20.00, 'EPC rating and floor area'),
('mortgage', 3, 'Estimated from building type + location', 30.00, 'Property type, location, floor area'),
('mortgage', 4, 'Estimated from regional averages', 40.00, 'Property type and region'),
('mortgage', 5, 'Estimated from national averages', 50.00, 'Property type only'),
-- Motor Vehicle Loan
('motor_vehicle_loan', 1, 'Actual fuel consumption data', 10.00, 'Telematics or actual fuel purchase data'),
('motor_vehicle_loan', 2, 'Make/model specific emission factor', 20.00, 'Vehicle make, model, year'),
('motor_vehicle_loan', 3, 'Vehicle category emission factor', 30.00, 'Vehicle category and fuel type'),
('motor_vehicle_loan', 4, 'Estimated from vehicle type average', 40.00, 'Vehicle type (car, SUV, truck)'),
('motor_vehicle_loan', 5, 'Estimated from fleet average', 50.00, 'Asset class average only'),
-- Sovereign Bond
('sovereign_bond', 1, 'National inventory (verified/reviewed)', 10.00, 'UNFCCC reviewed national GHG inventory'),
('sovereign_bond', 2, 'National inventory (submitted)', 15.00, 'UNFCCC submitted national inventory'),
('sovereign_bond', 3, 'Estimated from IEA/World Bank data', 25.00, 'IEA or World Bank emissions estimate'),
('sovereign_bond', 4, 'Estimated with partial data', 35.00, 'Partial emissions data with gap-filling'),
('sovereign_bond', 5, 'Estimated from regional averages', 45.00, 'Regional per-capita or GDP-based estimate');

-- =====================================================================================
-- SEED DATA: CURRENCY RATES (20 currencies)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_currency_rates
(currency_code, to_usd_rate, year, source) VALUES
('USD', 1.000000, 2024, 'WORLD_BANK'),
('EUR', 1.085000, 2024, 'ECB'),
('GBP', 1.268000, 2024, 'WORLD_BANK'),
('JPY', 0.006700, 2024, 'WORLD_BANK'),
('CHF', 1.128000, 2024, 'WORLD_BANK'),
('CAD', 0.742000, 2024, 'WORLD_BANK'),
('AUD', 0.654000, 2024, 'WORLD_BANK'),
('NZD', 0.612000, 2024, 'WORLD_BANK'),
('CNY', 0.138000, 2024, 'WORLD_BANK'),
('HKD', 0.128000, 2024, 'WORLD_BANK'),
('SGD', 0.746000, 2024, 'WORLD_BANK'),
('KRW', 0.000752, 2024, 'WORLD_BANK'),
('INR', 0.012000, 2024, 'WORLD_BANK'),
('BRL', 0.200000, 2024, 'WORLD_BANK'),
('MXN', 0.058400, 2024, 'WORLD_BANK'),
('ZAR', 0.055200, 2024, 'WORLD_BANK'),
('SAR', 0.266600, 2024, 'WORLD_BANK'),
('AED', 0.272200, 2024, 'WORLD_BANK'),
('SEK', 0.095600, 2024, 'WORLD_BANK'),
('NOK', 0.093800, 2024, 'WORLD_BANK');

-- =====================================================================================
-- SEED DATA: SOVEREIGN DATA (same 52 countries, extended fields)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_sovereign_data
(country_code, country_name, gdp_ppp_billion, total_emissions_mt, per_capita_tco2e, population_million, income_group, world_region, year) VALUES
('USA', 'United States',       25462.70, 5222.40, 15.5200, 336.50, 'HIGH',         'NORTH_AMERICA', 2022),
('CAN', 'Canada',               2139.84,  670.40, 17.2800,  38.80, 'HIGH',         'NORTH_AMERICA', 2022),
('MEX', 'Mexico',               2616.90,  492.30,  3.7600, 131.00, 'UPPER_MIDDLE', 'LATIN_AMERICA', 2022),
('GBR', 'United Kingdom',       3376.00,  341.20,  5.0500,  67.60, 'HIGH',         'EUROPE', 2022),
('DEU', 'Germany',              4857.00,  674.30,  8.0900,  83.40, 'HIGH',         'EUROPE', 2022),
('FRA', 'France',               3677.00,  299.70,  4.3900,  68.30, 'HIGH',         'EUROPE', 2022),
('ITA', 'Italy',                2972.00,  327.80,  5.5500,  59.10, 'HIGH',         'EUROPE', 2022),
('ESP', 'Spain',                2201.00,  245.60,  5.1800,  47.40, 'HIGH',         'EUROPE', 2022),
('NLD', 'Netherlands',          1149.00,  147.20,  8.4100,  17.50, 'HIGH',         'EUROPE', 2022),
('BEL', 'Belgium',               685.00,   99.80,  8.6200,  11.60, 'HIGH',         'EUROPE', 2022),
('AUT', 'Austria',               572.00,   66.40,  7.3800,   9.00, 'HIGH',         'EUROPE', 2022),
('CHE', 'Switzerland',           649.00,   36.40,  4.1800,   8.70, 'HIGH',         'EUROPE', 2022),
('SWE', 'Sweden',                629.00,   38.80,  3.7100,  10.50, 'HIGH',         'EUROPE', 2022),
('NOR', 'Norway',                443.00,   41.20,  7.5100,   5.50, 'HIGH',         'EUROPE', 2022),
('DNK', 'Denmark',               390.00,   31.40,  5.3500,   5.90, 'HIGH',         'EUROPE', 2022),
('FIN', 'Finland',               304.00,   36.20,  6.5200,   5.60, 'HIGH',         'EUROPE', 2022),
('IRL', 'Ireland',               577.00,   58.60, 11.5600,   5.10, 'HIGH',         'EUROPE', 2022),
('POL', 'Poland',               1548.00,  320.40,  8.4600,  37.90, 'HIGH',         'EUROPE', 2022),
('PRT', 'Portugal',              405.00,   48.60,  4.7200,  10.30, 'HIGH',         'EUROPE', 2022),
('GRC', 'Greece',                361.00,   58.80,  5.5800,  10.50, 'HIGH',         'EUROPE', 2022),
('CHN', 'China',               30330.00, 11472.00,  8.1300, 1412.00, 'UPPER_MIDDLE', 'EAST_ASIA', 2022),
('JPN', 'Japan',                5710.00, 1064.00,  8.5000, 125.20, 'HIGH',         'EAST_ASIA', 2022),
('KOR', 'South Korea',          2734.00,  616.40, 11.9200,  51.70, 'HIGH',         'EAST_ASIA', 2022),
('IND', 'India',               11870.00, 3347.00,  2.3500, 1425.00, 'LOWER_MIDDLE', 'SOUTH_ASIA', 2022),
('AUS', 'Australia',            1653.00,  464.80, 17.8400,  26.10, 'HIGH',         'OCEANIA', 2022),
('NZL', 'New Zealand',           248.00,   62.40, 12.1200,   5.20, 'HIGH',         'OCEANIA', 2022),
('SGP', 'Singapore',             637.00,   47.80,  8.2400,   5.80, 'HIGH',         'SOUTHEAST_ASIA', 2022),
('HKG', 'Hong Kong',             504.00,   31.40,  4.2100,   7.50, 'HIGH',         'EAST_ASIA', 2022),
('TWN', 'Taiwan',               1448.00,  268.40, 11.4200,  23.50, 'HIGH',         'EAST_ASIA', 2022),
('IDN', 'Indonesia',            3996.00,  691.20,  2.5200, 275.50, 'LOWER_MIDDLE', 'SOUTHEAST_ASIA', 2022),
('THA', 'Thailand',             1390.00,  261.40,  3.7200,  71.80, 'UPPER_MIDDLE', 'SOUTHEAST_ASIA', 2022),
('MYS', 'Malaysia',             1063.00,  254.20,  7.6200,  33.40, 'UPPER_MIDDLE', 'SOUTHEAST_ASIA', 2022),
('PHL', 'Philippines',          1138.00,  167.80,  1.5000, 115.60, 'LOWER_MIDDLE', 'SOUTHEAST_ASIA', 2022),
('VNM', 'Vietnam',              1278.00,  323.60,  3.2600,  99.50, 'LOWER_MIDDLE', 'SOUTHEAST_ASIA', 2022),
('SAU', 'Saudi Arabia',         1869.00,  588.40, 16.4200,  36.40, 'HIGH',         'MIDDLE_EAST', 2022),
('ARE', 'United Arab Emirates',  729.00,  193.60, 19.4200,  10.00, 'HIGH',         'MIDDLE_EAST', 2022),
('QAT', 'Qatar',                 265.00,   98.20, 33.5800,   2.90, 'HIGH',         'MIDDLE_EAST', 2022),
('BRA', 'Brazil',               3780.00,  478.20,  2.2200, 215.30, 'UPPER_MIDDLE', 'LATIN_AMERICA', 2022),
('ARG', 'Argentina',            1126.00,  192.40,  4.2000,  46.20, 'UPPER_MIDDLE', 'LATIN_AMERICA', 2022),
('CHL', 'Chile',                 523.00,   84.60,  4.3200,  19.60, 'HIGH',         'LATIN_AMERICA', 2022),
('COL', 'Colombia',              960.00,  118.40,  2.2800,  52.20, 'UPPER_MIDDLE', 'LATIN_AMERICA', 2022),
('ZAF', 'South Africa',          943.00,  434.80,  7.2200,  60.40, 'UPPER_MIDDLE', 'AFRICA', 2022),
('NGA', 'Nigeria',              1268.00,  126.40,  0.5800, 218.50, 'LOWER_MIDDLE', 'AFRICA', 2022),
('EGY', 'Egypt',                1561.00,  282.60,  2.6500, 106.70, 'LOWER_MIDDLE', 'AFRICA', 2022),
('KEN', 'Kenya',                 300.00,   24.80,  0.4600,  55.10, 'LOWER_MIDDLE', 'AFRICA', 2022),
('MAR', 'Morocco',               346.00,   68.20,  1.8200,  37.50, 'LOWER_MIDDLE', 'AFRICA', 2022),
('RUS', 'Russia',               4365.00, 1580.40, 10.8200, 146.00, 'UPPER_MIDDLE', 'EUROPE', 2022),
('TUR', 'Turkey',               3212.00,  418.60,  4.9200,  85.30, 'UPPER_MIDDLE', 'EUROPE', 2022),
('ISR', 'Israel',                478.00,   68.40,  7.2200,   9.60, 'HIGH',         'MIDDLE_EAST', 2022),
('PAK', 'Pakistan',             1505.00,  215.40,  0.9200, 235.80, 'LOWER_MIDDLE', 'SOUTH_ASIA', 2022),
('BGD', 'Bangladesh',           1266.00,  108.60,  0.6200, 172.00, 'LOWER_MIDDLE', 'SOUTH_ASIA', 2022),
('LKA', 'Sri Lanka',             305.00,   23.40,  1.0600,  22.20, 'LOWER_MIDDLE', 'SOUTH_ASIA', 2022);

-- =====================================================================================
-- SEED DATA: CARBON BENCHMARKS (12 sectors with 2030 targets)
-- =====================================================================================

INSERT INTO investments_service.gl_inv_carbon_benchmarks
(sector, intensity_tco2e_per_m, aligned_1_5c_target, aligned_2c_target, target_year, source) VALUES
('energy',                   850.4200, 425.2100, 552.7700, 2030, 'SBTi'),
('materials',                624.8100, 312.4050, 406.1300, 2030, 'SBTi'),
('industrials',              182.3400,  91.1700, 118.5200, 2030, 'SBTi'),
('consumer_discretionary',    78.5600,  39.2800,  51.0600, 2030, 'SBTi'),
('consumer_staples',         142.7300,  71.3650,  92.7700, 2030, 'SBTi'),
('health_care',               52.4100,  26.2050,  34.0700, 2030, 'SBTi'),
('financials',                12.3500,   6.1750,   8.0300, 2030, 'SBTi'),
('information_technology',    18.9200,   9.4600,  12.2980, 2030, 'SBTi'),
('communication_services',    22.6700,  11.3350,  14.7400, 2030, 'SBTi'),
('utilities',               1245.6800, 622.8400, 809.6900, 2030, 'SBTi'),
('real_estate',               68.4300,  34.2150,  44.4800, 2030, 'SBTi'),
('sovereign',                  0.0000,   0.0000,   0.0000, 2030, 'N/A');

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
    'GL-MRV-S3-015',
    'Investments Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-028: Scope 3 Category 15 - Investments / Financed Emissions. Calculates financed emissions using PCAF Global Standard for 8 asset classes (listed equity, private equity, corporate bonds, business loans, project finance, CRE, mortgages, motor vehicle loans) plus sovereign bonds. Supports PCAF attribution (outstanding/EVIC), building EUI, vehicle-specific, and sovereign PPP methods. WACI, carbon footprint, temperature alignment. 9-framework compliance (GHG Protocol, PCAF, TCFD, NZBA, CSRD, CDP, SBTi, ISO 14064, GRI). 12 GICS sector EFs, 52 country EFs, 30 grid EFs, 24 building benchmarks, 16 vehicle EFs, 12 EEIO factors, 40 PCAF quality criteria, 20 currency rates, 52 sovereign records, 12 carbon benchmarks.',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 15,
        'category_name', 'Investments',
        'calculation_methods', jsonb_build_array('pcaf_attribution', 'investment_specific', 'average_data', 'spend_based', 'building_eui', 'vehicle_specific', 'sovereign_ppp'),
        'asset_classes', jsonb_build_array(
            'listed_equity', 'private_equity', 'corporate_bond', 'business_loan',
            'project_finance', 'commercial_real_estate', 'mortgage',
            'motor_vehicle_loan', 'sovereign_bond'
        ),
        'frameworks', jsonb_build_array('GHG Protocol Scope 3', 'PCAF Global Standard', 'TCFD', 'NZBA', 'CSRD ESRS E1', 'CDP Climate', 'SBTi Financial Sector', 'ISO 14064-1', 'GRI 305'),
        'sector_ef_count', 12,
        'country_ef_count', 52,
        'grid_ef_count', 30,
        'building_benchmark_count', 24,
        'vehicle_ef_count', 16,
        'eeio_factor_count', 12,
        'pcaf_quality_criteria', 40,
        'currency_rate_count', 20,
        'sovereign_data_count', 52,
        'carbon_benchmark_count', 12,
        'total_seed_records', 270,
        'supports_pcaf_scoring', true,
        'supports_waci', true,
        'supports_temperature_alignment', true,
        'supports_portfolio_analysis', true,
        'supports_batch_50k', true,
        'supports_sovereign_bonds', true,
        'default_ef_source', 'MSCI_Trucost_2023',
        'schema', 'investments_service',
        'table_prefix', 'gl_inv_',
        'hypertables', jsonb_build_array('gl_inv_calculations', 'gl_inv_compliance_checks', 'gl_inv_aggregations'),
        'continuous_aggregates', jsonb_build_array('gl_inv_daily_financed_emissions', 'gl_inv_monthly_portfolio_summary'),
        'migration_version', 'V079'
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

COMMENT ON SCHEMA investments_service IS 'Updated: AGENT-MRV-028 complete with 22 tables, 3 hypertables, 2 continuous aggregates, 270+ seed records';

-- =====================================================================================
-- END OF MIGRATION V079
-- =====================================================================================
-- Total Lines: ~1200
-- Total Tables: 22 (10 reference + 9 operational + 3 supporting)
-- Total Hypertables: 3 (calculations 7d, compliance_checks 30d, aggregations 30d)
-- Total Continuous Aggregates: 2 (daily_financed_emissions, monthly_portfolio_summary)
-- Total Indexes: 80
-- Total Seed Records: 270
--   Sector Emission Factors: 12 (GICS sectors)
--   Country Emission Factors: 52 (UNFCCC/IEA)
--   Grid Emission Factors: 30 (23 eGRID + 7 international)
--   Building Benchmarks: 24 (12 property types x 2 zones)
--   Vehicle Emission Factors: 16 (11 categories x fuel types)
--   EEIO Sector Factors: 12 (GICS sectors)
--   PCAF Data Quality Criteria: 40 (8 asset classes x 5 scores)
--   Currency Rates: 20 (major currencies)
--   Sovereign Data: 52 (countries with GDP/emissions/population)
--   Carbon Benchmarks: 12 (sectors with 1.5C/2C targets)
--   Agent Registry: 1
-- =====================================================================================

-- =====================================================================================
-- Migration: V074__processing_sold_products_service.sql
-- Description: AGENT-MRV-023 Processing of Sold Products (Scope 3 Category 10)
-- Agent: GL-MRV-S3-010
-- Framework: GHG Protocol Scope 3 Standard, ISO 14064-1, CSRD ESRS E1,
--            CDP Climate, SBTi FLAG/SDA, SB 253, GRI 305
-- Created: 2026-02-28
-- =====================================================================================
-- Schema: gl_psp
-- Tables: 19 (10 reference + 9 operational)
-- Hypertables: 3 (calculations, aggregations, compliance_results)
-- Continuous Aggregates: 2 (daily_emissions_by_category, monthly_emissions_by_method)
-- RLS: Enabled on 10 operational tables with tenant_id isolation
-- Seed Data: 95+ records (processing EFs, energy intensity, grid EFs, fuel EFs,
--            EEIO factors, processing chains, currencies, CPI deflators)
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS gl_psp;

SET search_path TO gl_psp, public;

COMMENT ON SCHEMA gl_psp IS 'AGENT-MRV-023: Processing of Sold Products - Scope 3 Category 10 emission calculations (intermediate product processing by downstream customers, site-specific / average-data / spend-based / hybrid methods)';

-- =====================================================================================
-- TABLE 1: gl_psp_processing_emission_factors (REFERENCE)
-- Description: Processing emission factors by product category and processing type
-- Source: DEFRA 2024, EPA, Ecoinvent 3.10, IEA 2024, Industry averages
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_processing_emission_factors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    category VARCHAR(64) NOT NULL,
    processing_type VARCHAR(64),
    ef_value NUMERIC(20,8) NOT NULL,
    ef_unit VARCHAR(32) NOT NULL DEFAULT 'kgCO2e/tonne',
    source VARCHAR(64) NOT NULL,
    region VARCHAR(16),
    reference_year INTEGER NOT NULL,
    uncertainty NUMERIC(10,4),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_pef_category CHECK (category IN (
        'METALS_FERROUS', 'METALS_NON_FERROUS', 'PLASTICS_THERMOPLASTIC',
        'PLASTICS_THERMOSET', 'CHEMICALS', 'FOOD_INGREDIENTS', 'TEXTILES',
        'ELECTRONICS', 'GLASS_CERAMICS', 'WOOD_PAPER', 'MINERALS', 'AGRICULTURAL'
    )),
    CONSTRAINT chk_psp_pef_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_psp_pef_uncertainty_range CHECK (uncertainty IS NULL OR (uncertainty >= 0 AND uncertainty <= 1.0)),
    CONSTRAINT chk_psp_pef_year_valid CHECK (reference_year >= 1990 AND reference_year <= 2100),
    CONSTRAINT uq_psp_pef_cat_type_src_region UNIQUE (category, processing_type, source, region, tenant_id)
);

CREATE INDEX idx_psp_pef_category ON gl_psp.gl_psp_processing_emission_factors(category);
CREATE INDEX idx_psp_pef_processing_type ON gl_psp.gl_psp_processing_emission_factors(processing_type);
CREATE INDEX idx_psp_pef_source ON gl_psp.gl_psp_processing_emission_factors(source);
CREATE INDEX idx_psp_pef_region ON gl_psp.gl_psp_processing_emission_factors(region);
CREATE INDEX idx_psp_pef_tenant ON gl_psp.gl_psp_processing_emission_factors(tenant_id);
CREATE INDEX idx_psp_pef_year ON gl_psp.gl_psp_processing_emission_factors(reference_year);
CREATE INDEX idx_psp_pef_cat_region ON gl_psp.gl_psp_processing_emission_factors(category, region);

COMMENT ON TABLE gl_psp.gl_psp_processing_emission_factors IS 'Processing emission factors by product category (12 categories) and processing type. Source: DEFRA 2024, EPA, Ecoinvent 3.10, IEA 2024';
COMMENT ON COLUMN gl_psp.gl_psp_processing_emission_factors.category IS 'Product category: METALS_FERROUS, METALS_NON_FERROUS, PLASTICS_THERMOPLASTIC, PLASTICS_THERMOSET, CHEMICALS, FOOD_INGREDIENTS, TEXTILES, ELECTRONICS, GLASS_CERAMICS, WOOD_PAPER, MINERALS, AGRICULTURAL';
COMMENT ON COLUMN gl_psp.gl_psp_processing_emission_factors.processing_type IS 'Specific processing type (e.g., MACHINING, INJECTION_MOLDING, CHEMICAL_SYNTHESIS)';
COMMENT ON COLUMN gl_psp.gl_psp_processing_emission_factors.ef_value IS 'Emission factor value in kgCO2e per unit (default per tonne)';
COMMENT ON COLUMN gl_psp.gl_psp_processing_emission_factors.ef_unit IS 'Emission factor unit (e.g., kgCO2e/tonne, kgCO2e/kWh)';
COMMENT ON COLUMN gl_psp.gl_psp_processing_emission_factors.uncertainty IS 'Uncertainty as fraction (0.0 to 1.0), e.g., 0.15 = +/-15%';

-- =====================================================================================
-- TABLE 2: gl_psp_energy_intensity_factors (REFERENCE)
-- Description: Energy intensity by processing type in kWh/tonne
-- Source: IEA Industrial Energy Efficiency, DOE Manufacturing Energy Bandwidth Studies
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_energy_intensity_factors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    processing_type VARCHAR(64) NOT NULL,
    low NUMERIC(20,8) NOT NULL,
    mid NUMERIC(20,8) NOT NULL,
    high NUMERIC(20,8) NOT NULL,
    default_value NUMERIC(20,8) NOT NULL,
    unit VARCHAR(32) NOT NULL DEFAULT 'kWh/tonne',
    source VARCHAR(64) NOT NULL,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_eif_low_positive CHECK (low >= 0),
    CONSTRAINT chk_psp_eif_mid_positive CHECK (mid >= 0),
    CONSTRAINT chk_psp_eif_high_positive CHECK (high >= 0),
    CONSTRAINT chk_psp_eif_default_positive CHECK (default_value >= 0),
    CONSTRAINT chk_psp_eif_range_order CHECK (low <= mid AND mid <= high),
    CONSTRAINT chk_psp_eif_default_in_range CHECK (default_value >= low AND default_value <= high),
    CONSTRAINT uq_psp_eif_type_tenant UNIQUE (processing_type, tenant_id)
);

CREATE INDEX idx_psp_eif_processing_type ON gl_psp.gl_psp_energy_intensity_factors(processing_type);
CREATE INDEX idx_psp_eif_tenant ON gl_psp.gl_psp_energy_intensity_factors(tenant_id);
CREATE INDEX idx_psp_eif_source ON gl_psp.gl_psp_energy_intensity_factors(source);

COMMENT ON TABLE gl_psp.gl_psp_energy_intensity_factors IS 'Energy intensity factors by processing type with low/mid/high ranges. Source: IEA Industrial Energy Efficiency, DOE Manufacturing Bandwidth Studies';
COMMENT ON COLUMN gl_psp.gl_psp_energy_intensity_factors.processing_type IS 'Processing type (e.g., MACHINING, CASTING, INJECTION_MOLDING, EXTRUSION)';
COMMENT ON COLUMN gl_psp.gl_psp_energy_intensity_factors.low IS 'Low-end energy intensity in kWh/tonne (best available technology)';
COMMENT ON COLUMN gl_psp.gl_psp_energy_intensity_factors.mid IS 'Mid-range energy intensity in kWh/tonne (industry average)';
COMMENT ON COLUMN gl_psp.gl_psp_energy_intensity_factors.high IS 'High-end energy intensity in kWh/tonne (older/less efficient technology)';
COMMENT ON COLUMN gl_psp.gl_psp_energy_intensity_factors.default_value IS 'Default energy intensity value used when specific data unavailable';

-- =====================================================================================
-- TABLE 3: gl_psp_grid_emission_factors (REFERENCE)
-- Description: Grid emission factors by country/region for electricity used in processing
-- Source: IEA 2024, eGRID 2024, DEFRA 2024
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_grid_emission_factors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    region VARCHAR(16) NOT NULL,
    country_name VARCHAR(128),
    ef_value NUMERIC(20,8) NOT NULL,
    ef_unit VARCHAR(32) NOT NULL DEFAULT 'kgCO2e/kWh',
    reference_year INTEGER NOT NULL,
    source VARCHAR(64) NOT NULL,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_gef_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_psp_gef_year_valid CHECK (reference_year >= 1990 AND reference_year <= 2100),
    CONSTRAINT uq_psp_gef_region_year_tenant UNIQUE (region, reference_year, tenant_id)
);

CREATE INDEX idx_psp_gef_region ON gl_psp.gl_psp_grid_emission_factors(region);
CREATE INDEX idx_psp_gef_tenant ON gl_psp.gl_psp_grid_emission_factors(tenant_id);
CREATE INDEX idx_psp_gef_source ON gl_psp.gl_psp_grid_emission_factors(source);
CREATE INDEX idx_psp_gef_year ON gl_psp.gl_psp_grid_emission_factors(reference_year);
CREATE INDEX idx_psp_gef_region_year ON gl_psp.gl_psp_grid_emission_factors(region, reference_year);

COMMENT ON TABLE gl_psp.gl_psp_grid_emission_factors IS 'Grid emission factors by country/region for electricity used in downstream processing. Source: IEA 2024, eGRID 2024, DEFRA 2024';
COMMENT ON COLUMN gl_psp.gl_psp_grid_emission_factors.region IS 'Country ISO 3166-1 alpha-2 code or region code (e.g., US, GB, EU, GLOBAL)';
COMMENT ON COLUMN gl_psp.gl_psp_grid_emission_factors.ef_value IS 'Grid emission factor in kgCO2e/kWh';
COMMENT ON COLUMN gl_psp.gl_psp_grid_emission_factors.country_name IS 'Full country or region name';

-- =====================================================================================
-- TABLE 4: gl_psp_fuel_emission_factors (REFERENCE)
-- Description: Fuel emission factors for thermal energy used in processing
-- Source: DEFRA 2024, EPA 2024, IPCC AR5
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_fuel_emission_factors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    fuel_type VARCHAR(32) NOT NULL,
    ef_value NUMERIC(20,8) NOT NULL,
    ef_unit VARCHAR(32) NOT NULL,
    source VARCHAR(64) NOT NULL,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_fef_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_psp_fef_fuel_type CHECK (fuel_type IN (
        'NATURAL_GAS', 'DIESEL', 'FUEL_OIL', 'LPG', 'COAL', 'BIOMASS',
        'BIOGAS', 'HYDROGEN', 'ELECTRICITY', 'STEAM'
    )),
    CONSTRAINT uq_psp_fef_fuel_tenant UNIQUE (fuel_type, tenant_id)
);

CREATE INDEX idx_psp_fef_fuel_type ON gl_psp.gl_psp_fuel_emission_factors(fuel_type);
CREATE INDEX idx_psp_fef_tenant ON gl_psp.gl_psp_fuel_emission_factors(tenant_id);
CREATE INDEX idx_psp_fef_source ON gl_psp.gl_psp_fuel_emission_factors(source);

COMMENT ON TABLE gl_psp.gl_psp_fuel_emission_factors IS 'Fuel emission factors for thermal energy used in downstream processing. Source: DEFRA 2024, EPA 2024, IPCC AR5';
COMMENT ON COLUMN gl_psp.gl_psp_fuel_emission_factors.fuel_type IS 'Fuel type: NATURAL_GAS, DIESEL, FUEL_OIL, LPG, COAL, BIOMASS, BIOGAS, HYDROGEN, ELECTRICITY, STEAM';
COMMENT ON COLUMN gl_psp.gl_psp_fuel_emission_factors.ef_value IS 'Emission factor value (units as specified in ef_unit)';
COMMENT ON COLUMN gl_psp.gl_psp_fuel_emission_factors.ef_unit IS 'Emission factor unit (e.g., kgCO2e/kWh, kgCO2e/litre, kgCO2e/tonne)';

-- =====================================================================================
-- TABLE 5: gl_psp_eeio_sector_factors (REFERENCE)
-- Description: EEIO spend-based emission factors by NAICS sector
-- Source: EPA USEEIO v2.0
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_eeio_sector_factors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    naics_code VARCHAR(8) NOT NULL,
    sector_name VARCHAR(128) NOT NULL,
    ef_value NUMERIC(20,8) NOT NULL,
    ef_unit VARCHAR(32) NOT NULL DEFAULT 'kgCO2e/USD',
    margin NUMERIC(10,4),
    source VARCHAR(64) NOT NULL,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_eeio_ef_positive CHECK (ef_value >= 0),
    CONSTRAINT chk_psp_eeio_margin_range CHECK (margin IS NULL OR (margin >= 0 AND margin <= 1.0)),
    CONSTRAINT uq_psp_eeio_naics_tenant UNIQUE (naics_code, tenant_id)
);

CREATE INDEX idx_psp_eeio_naics ON gl_psp.gl_psp_eeio_sector_factors(naics_code);
CREATE INDEX idx_psp_eeio_sector ON gl_psp.gl_psp_eeio_sector_factors(sector_name);
CREATE INDEX idx_psp_eeio_tenant ON gl_psp.gl_psp_eeio_sector_factors(tenant_id);
CREATE INDEX idx_psp_eeio_source ON gl_psp.gl_psp_eeio_sector_factors(source);

COMMENT ON TABLE gl_psp.gl_psp_eeio_sector_factors IS 'EPA USEEIO v2.0 spend-based emission factors for processing sectors by NAICS code. Used for spend-based calculation method';
COMMENT ON COLUMN gl_psp.gl_psp_eeio_sector_factors.naics_code IS 'NAICS industry code (e.g., 331110 Iron and Steel Mills)';
COMMENT ON COLUMN gl_psp.gl_psp_eeio_sector_factors.sector_name IS 'Sector name matching the NAICS code';
COMMENT ON COLUMN gl_psp.gl_psp_eeio_sector_factors.ef_value IS 'Emission factor per USD spent (kgCO2e/USD, base year deflated)';
COMMENT ON COLUMN gl_psp.gl_psp_eeio_sector_factors.margin IS 'Producer margin fraction for margin removal (0.0 to 1.0)';

-- =====================================================================================
-- TABLE 6: gl_psp_processing_chains (REFERENCE)
-- Description: Multi-step processing chain definitions with JSONB step arrays
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_processing_chains (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chain_type VARCHAR(64) NOT NULL,
    chain_name VARCHAR(128) NOT NULL,
    steps JSONB NOT NULL,
    combined_ef NUMERIC(20,8),
    description TEXT,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_pc_combined_ef_positive CHECK (combined_ef IS NULL OR combined_ef >= 0),
    CONSTRAINT chk_psp_pc_steps_array CHECK (jsonb_typeof(steps) = 'array'),
    CONSTRAINT uq_psp_pc_type_name_tenant UNIQUE (chain_type, chain_name, tenant_id)
);

CREATE INDEX idx_psp_pc_chain_type ON gl_psp.gl_psp_processing_chains(chain_type);
CREATE INDEX idx_psp_pc_chain_name ON gl_psp.gl_psp_processing_chains(chain_name);
CREATE INDEX idx_psp_pc_tenant ON gl_psp.gl_psp_processing_chains(tenant_id);
CREATE INDEX idx_psp_pc_steps ON gl_psp.gl_psp_processing_chains USING GIN(steps);

COMMENT ON TABLE gl_psp.gl_psp_processing_chains IS 'Multi-step processing chain definitions for complex downstream processing sequences. Each chain contains ordered JSONB step arrays';
COMMENT ON COLUMN gl_psp.gl_psp_processing_chains.chain_type IS 'Chain type identifier (e.g., STEEL_FABRICATION, PLASTIC_MOLDING_ASSEMBLY)';
COMMENT ON COLUMN gl_psp.gl_psp_processing_chains.chain_name IS 'Human-readable chain name';
COMMENT ON COLUMN gl_psp.gl_psp_processing_chains.steps IS 'JSONB array of processing steps [{step, process, ef_value, unit, energy_kwh}]';
COMMENT ON COLUMN gl_psp.gl_psp_processing_chains.combined_ef IS 'Pre-calculated combined emission factor for all steps (kgCO2e/tonne)';

-- =====================================================================================
-- TABLE 7: gl_psp_intermediate_products (REFERENCE)
-- Description: Product registry for intermediate products sold to downstream processors
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_intermediate_products (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    product_id VARCHAR(64) NOT NULL,
    product_name VARCHAR(256),
    category VARCHAR(64) NOT NULL,
    default_processing_type VARCHAR(64),
    default_unit VARCHAR(16),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_ip_category CHECK (category IN (
        'METALS_FERROUS', 'METALS_NON_FERROUS', 'PLASTICS_THERMOPLASTIC',
        'PLASTICS_THERMOSET', 'CHEMICALS', 'FOOD_INGREDIENTS', 'TEXTILES',
        'ELECTRONICS', 'GLASS_CERAMICS', 'WOOD_PAPER', 'MINERALS', 'AGRICULTURAL'
    )),
    CONSTRAINT uq_psp_ip_product_tenant UNIQUE (product_id, tenant_id)
);

CREATE INDEX idx_psp_ip_product_id ON gl_psp.gl_psp_intermediate_products(product_id);
CREATE INDEX idx_psp_ip_category ON gl_psp.gl_psp_intermediate_products(category);
CREATE INDEX idx_psp_ip_processing_type ON gl_psp.gl_psp_intermediate_products(default_processing_type);
CREATE INDEX idx_psp_ip_tenant ON gl_psp.gl_psp_intermediate_products(tenant_id);

COMMENT ON TABLE gl_psp.gl_psp_intermediate_products IS 'Registry of intermediate products sold to downstream customers for further processing. Links products to categories and default processing types';
COMMENT ON COLUMN gl_psp.gl_psp_intermediate_products.product_id IS 'Unique product identifier (e.g., SKU, internal product code)';
COMMENT ON COLUMN gl_psp.gl_psp_intermediate_products.category IS 'Product category: 12 categories from METALS_FERROUS to AGRICULTURAL';
COMMENT ON COLUMN gl_psp.gl_psp_intermediate_products.default_processing_type IS 'Default processing type applied to this product by downstream customers';
COMMENT ON COLUMN gl_psp.gl_psp_intermediate_products.default_unit IS 'Default measurement unit (e.g., tonne, kg, m3, unit)';

-- =====================================================================================
-- TABLE 8: gl_psp_customer_processing_data (OPERATIONAL)
-- Description: Customer-provided processing data for site-specific calculations
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_customer_processing_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    customer_name VARCHAR(256),
    product_id VARCHAR(64),
    processing_emissions_per_unit NUMERIC(20,8),
    energy_per_unit_kwh NUMERIC(20,8),
    fuel_type VARCHAR(32),
    fuel_per_unit NUMERIC(20,8),
    country VARCHAR(16),
    data_year INTEGER,
    verified BOOLEAN DEFAULT FALSE,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_cpd_emissions_positive CHECK (processing_emissions_per_unit IS NULL OR processing_emissions_per_unit >= 0),
    CONSTRAINT chk_psp_cpd_energy_positive CHECK (energy_per_unit_kwh IS NULL OR energy_per_unit_kwh >= 0),
    CONSTRAINT chk_psp_cpd_fuel_positive CHECK (fuel_per_unit IS NULL OR fuel_per_unit >= 0),
    CONSTRAINT chk_psp_cpd_fuel_type CHECK (fuel_type IS NULL OR fuel_type IN (
        'NATURAL_GAS', 'DIESEL', 'FUEL_OIL', 'LPG', 'COAL', 'BIOMASS',
        'BIOGAS', 'HYDROGEN', 'ELECTRICITY', 'STEAM'
    )),
    CONSTRAINT chk_psp_cpd_year_valid CHECK (data_year IS NULL OR (data_year >= 1990 AND data_year <= 2100)),
    CONSTRAINT uq_psp_cpd_cust_prod_year_tenant UNIQUE (customer_id, product_id, data_year, tenant_id)
);

CREATE INDEX idx_psp_cpd_customer_id ON gl_psp.gl_psp_customer_processing_data(customer_id);
CREATE INDEX idx_psp_cpd_customer_name ON gl_psp.gl_psp_customer_processing_data(customer_name);
CREATE INDEX idx_psp_cpd_product_id ON gl_psp.gl_psp_customer_processing_data(product_id);
CREATE INDEX idx_psp_cpd_country ON gl_psp.gl_psp_customer_processing_data(country);
CREATE INDEX idx_psp_cpd_fuel_type ON gl_psp.gl_psp_customer_processing_data(fuel_type);
CREATE INDEX idx_psp_cpd_data_year ON gl_psp.gl_psp_customer_processing_data(data_year);
CREATE INDEX idx_psp_cpd_verified ON gl_psp.gl_psp_customer_processing_data(verified);
CREATE INDEX idx_psp_cpd_tenant ON gl_psp.gl_psp_customer_processing_data(tenant_id);
CREATE INDEX idx_psp_cpd_cust_prod ON gl_psp.gl_psp_customer_processing_data(customer_id, product_id);

COMMENT ON TABLE gl_psp.gl_psp_customer_processing_data IS 'Customer-provided processing data for site-specific (supplier-specific) Scope 3 Cat 10 calculations. Includes energy, fuel, and direct emissions per unit';
COMMENT ON COLUMN gl_psp.gl_psp_customer_processing_data.customer_id IS 'Downstream customer identifier';
COMMENT ON COLUMN gl_psp.gl_psp_customer_processing_data.processing_emissions_per_unit IS 'Customer-reported processing emissions per product unit (kgCO2e/unit)';
COMMENT ON COLUMN gl_psp.gl_psp_customer_processing_data.energy_per_unit_kwh IS 'Electricity consumption per product unit (kWh/unit)';
COMMENT ON COLUMN gl_psp.gl_psp_customer_processing_data.fuel_type IS 'Primary fuel type used in processing';
COMMENT ON COLUMN gl_psp.gl_psp_customer_processing_data.fuel_per_unit IS 'Fuel consumption per product unit (unit varies by fuel type)';
COMMENT ON COLUMN gl_psp.gl_psp_customer_processing_data.verified IS 'Whether customer data has been independently verified';

-- =====================================================================================
-- TABLE 9: gl_psp_currencies (REFERENCE)
-- Description: Currency conversion rates to USD for spend-based calculations
-- Source: IMF, World Bank, Central Banks
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_currencies (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    currency_code VARCHAR(8) NOT NULL,
    currency_name VARCHAR(64),
    usd_rate NUMERIC(20,8) NOT NULL,
    reference_date DATE NOT NULL,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_cur_rate_positive CHECK (usd_rate > 0),
    CONSTRAINT uq_psp_cur_code_date_tenant UNIQUE (currency_code, reference_date, tenant_id)
);

CREATE INDEX idx_psp_cur_code ON gl_psp.gl_psp_currencies(currency_code);
CREATE INDEX idx_psp_cur_tenant ON gl_psp.gl_psp_currencies(tenant_id);
CREATE INDEX idx_psp_cur_date ON gl_psp.gl_psp_currencies(reference_date);

COMMENT ON TABLE gl_psp.gl_psp_currencies IS 'Currency conversion rates to USD for spend-based emission calculations';
COMMENT ON COLUMN gl_psp.gl_psp_currencies.currency_code IS 'ISO 4217 currency code (e.g., USD, EUR, GBP)';
COMMENT ON COLUMN gl_psp.gl_psp_currencies.usd_rate IS 'Exchange rate: 1 unit of this currency = X USD';
COMMENT ON COLUMN gl_psp.gl_psp_currencies.reference_date IS 'Date of exchange rate';

-- =====================================================================================
-- TABLE 10: gl_psp_cpi_deflators (REFERENCE)
-- Description: CPI deflation indices for spend-based cost normalization
-- Source: US Bureau of Labor Statistics, World Bank
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_cpi_deflators (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    year INTEGER NOT NULL,
    cpi_index NUMERIC(10,4) NOT NULL,
    base_year INTEGER NOT NULL DEFAULT 2024,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_cpi_year_valid CHECK (year >= 1990 AND year <= 2100),
    CONSTRAINT chk_psp_cpi_index_positive CHECK (cpi_index > 0),
    CONSTRAINT chk_psp_cpi_base_year_valid CHECK (base_year >= 1990 AND base_year <= 2100),
    CONSTRAINT uq_psp_cpi_year_base_tenant UNIQUE (year, base_year, tenant_id)
);

CREATE INDEX idx_psp_cpi_year ON gl_psp.gl_psp_cpi_deflators(year);
CREATE INDEX idx_psp_cpi_tenant ON gl_psp.gl_psp_cpi_deflators(tenant_id);
CREATE INDEX idx_psp_cpi_base_year ON gl_psp.gl_psp_cpi_deflators(base_year);

COMMENT ON TABLE gl_psp.gl_psp_cpi_deflators IS 'Consumer Price Index deflation indices for normalizing spend data to base year 2024 USD. Source: BLS, World Bank';
COMMENT ON COLUMN gl_psp.gl_psp_cpi_deflators.year IS 'Calendar year of the CPI index value';
COMMENT ON COLUMN gl_psp.gl_psp_cpi_deflators.cpi_index IS 'CPI index value relative to base year (base year = 100.0)';
COMMENT ON COLUMN gl_psp.gl_psp_cpi_deflators.base_year IS 'Base year for CPI normalization (default 2024)';

-- =====================================================================================
-- TABLE 11: gl_psp_calculations (OPERATIONAL / HYPERTABLE)
-- Description: Main calculation results for processing of sold products
-- Hypertable on calculated_at for time-series optimization
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_calculations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL,
    reporting_year INTEGER,
    calculation_method VARCHAR(32) NOT NULL,
    total_emissions_kg NUMERIC(20,8) NOT NULL,
    total_emissions_tco2e NUMERIC(20,8) NOT NULL,
    num_products INTEGER,
    dqi_score NUMERIC(10,4),
    uncertainty_lower NUMERIC(20,8),
    uncertainty_upper NUMERIC(20,8),
    provenance_hash VARCHAR(128),
    tenant_id UUID NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_calc_method CHECK (calculation_method IN (
        'SITE_SPECIFIC', 'AVERAGE_DATA', 'SPEND_BASED', 'HYBRID'
    )),
    CONSTRAINT chk_psp_calc_emissions_positive CHECK (total_emissions_kg >= 0),
    CONSTRAINT chk_psp_calc_tco2e_positive CHECK (total_emissions_tco2e >= 0),
    CONSTRAINT chk_psp_calc_num_products_positive CHECK (num_products IS NULL OR num_products >= 0),
    CONSTRAINT chk_psp_calc_dqi_range CHECK (dqi_score IS NULL OR (dqi_score >= 1.0 AND dqi_score <= 5.0)),
    CONSTRAINT chk_psp_calc_uncertainty_order CHECK (
        uncertainty_lower IS NULL OR uncertainty_upper IS NULL OR uncertainty_lower <= uncertainty_upper
    ),
    CONSTRAINT chk_psp_calc_year_valid CHECK (reporting_year IS NULL OR (reporting_year >= 1990 AND reporting_year <= 2100)),
    PRIMARY KEY (id, calculated_at)
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('gl_psp.gl_psp_calculations', 'calculated_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_psp_calc_org ON gl_psp.gl_psp_calculations(org_id, calculated_at DESC);
CREATE INDEX idx_psp_calc_tenant ON gl_psp.gl_psp_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_psp_calc_method ON gl_psp.gl_psp_calculations(calculation_method);
CREATE INDEX idx_psp_calc_year ON gl_psp.gl_psp_calculations(reporting_year);
CREATE INDEX idx_psp_calc_org_year ON gl_psp.gl_psp_calculations(org_id, reporting_year);
CREATE INDEX idx_psp_calc_provenance ON gl_psp.gl_psp_calculations(provenance_hash);
CREATE INDEX idx_psp_calc_dqi ON gl_psp.gl_psp_calculations(dqi_score);

COMMENT ON TABLE gl_psp.gl_psp_calculations IS 'Main calculation results for Scope 3 Category 10 - Processing of Sold Products. TimescaleDB hypertable on calculated_at (30-day chunks)';
COMMENT ON COLUMN gl_psp.gl_psp_calculations.org_id IS 'Organization UUID (reporting entity)';
COMMENT ON COLUMN gl_psp.gl_psp_calculations.calculation_method IS 'Calculation method: SITE_SPECIFIC, AVERAGE_DATA, SPEND_BASED, HYBRID';
COMMENT ON COLUMN gl_psp.gl_psp_calculations.total_emissions_kg IS 'Total processing emissions in kgCO2e';
COMMENT ON COLUMN gl_psp.gl_psp_calculations.total_emissions_tco2e IS 'Total processing emissions in tCO2e (tonnes)';
COMMENT ON COLUMN gl_psp.gl_psp_calculations.dqi_score IS 'Data Quality Indicator score (1.0 = highest quality, 5.0 = lowest)';
COMMENT ON COLUMN gl_psp.gl_psp_calculations.provenance_hash IS 'SHA-256 provenance hash for complete audit trail';

-- =====================================================================================
-- TABLE 12: gl_psp_calculation_details (OPERATIONAL)
-- Description: Per-product breakdown of calculation results
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_calculation_details (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    calculation_id UUID NOT NULL,
    product_id VARCHAR(64),
    category VARCHAR(64),
    processing_type VARCHAR(64),
    quantity NUMERIC(20,8),
    unit VARCHAR(16),
    emissions_kg NUMERIC(20,8) NOT NULL,
    ef_used NUMERIC(20,8),
    ef_source VARCHAR(64),
    method VARCHAR(32),
    dqi_score NUMERIC(10,4),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_cd_category CHECK (category IS NULL OR category IN (
        'METALS_FERROUS', 'METALS_NON_FERROUS', 'PLASTICS_THERMOPLASTIC',
        'PLASTICS_THERMOSET', 'CHEMICALS', 'FOOD_INGREDIENTS', 'TEXTILES',
        'ELECTRONICS', 'GLASS_CERAMICS', 'WOOD_PAPER', 'MINERALS', 'AGRICULTURAL'
    )),
    CONSTRAINT chk_psp_cd_method CHECK (method IS NULL OR method IN (
        'SITE_SPECIFIC', 'AVERAGE_DATA', 'SPEND_BASED', 'HYBRID'
    )),
    CONSTRAINT chk_psp_cd_emissions_positive CHECK (emissions_kg >= 0),
    CONSTRAINT chk_psp_cd_quantity_positive CHECK (quantity IS NULL OR quantity >= 0),
    CONSTRAINT chk_psp_cd_ef_positive CHECK (ef_used IS NULL OR ef_used >= 0),
    CONSTRAINT chk_psp_cd_dqi_range CHECK (dqi_score IS NULL OR (dqi_score >= 1.0 AND dqi_score <= 5.0))
);

CREATE INDEX idx_psp_cd_calculation ON gl_psp.gl_psp_calculation_details(calculation_id);
CREATE INDEX idx_psp_cd_product ON gl_psp.gl_psp_calculation_details(product_id);
CREATE INDEX idx_psp_cd_category ON gl_psp.gl_psp_calculation_details(category);
CREATE INDEX idx_psp_cd_processing_type ON gl_psp.gl_psp_calculation_details(processing_type);
CREATE INDEX idx_psp_cd_method ON gl_psp.gl_psp_calculation_details(method);
CREATE INDEX idx_psp_cd_ef_source ON gl_psp.gl_psp_calculation_details(ef_source);
CREATE INDEX idx_psp_cd_tenant ON gl_psp.gl_psp_calculation_details(tenant_id);
CREATE INDEX idx_psp_cd_created ON gl_psp.gl_psp_calculation_details(created_at DESC);

COMMENT ON TABLE gl_psp.gl_psp_calculation_details IS 'Per-product breakdown of Scope 3 Cat 10 calculation results with emission factor traceability';
COMMENT ON COLUMN gl_psp.gl_psp_calculation_details.calculation_id IS 'FK to gl_psp_calculations.id (parent calculation)';
COMMENT ON COLUMN gl_psp.gl_psp_calculation_details.product_id IS 'Product identifier from gl_psp_intermediate_products';
COMMENT ON COLUMN gl_psp.gl_psp_calculation_details.emissions_kg IS 'Product-level emissions in kgCO2e';
COMMENT ON COLUMN gl_psp.gl_psp_calculation_details.ef_used IS 'Emission factor value applied in this calculation';
COMMENT ON COLUMN gl_psp.gl_psp_calculation_details.ef_source IS 'Source of the emission factor used (e.g., DEFRA_2024, EPA_USEEIO_v2, CUSTOMER)';

-- =====================================================================================
-- TABLE 13: gl_psp_aggregations (OPERATIONAL / HYPERTABLE)
-- Description: Aggregated results by period with category/method/country breakdowns
-- Hypertable on period_start for time-series optimization
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_aggregations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    total_emissions_tco2e NUMERIC(20,8) NOT NULL,
    by_category JSONB DEFAULT '{}',
    by_method JSONB DEFAULT '{}',
    by_country JSONB DEFAULT '{}',
    num_calculations INTEGER,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_agg_emissions_positive CHECK (total_emissions_tco2e >= 0),
    CONSTRAINT chk_psp_agg_period_valid CHECK (period_end >= period_start),
    CONSTRAINT chk_psp_agg_num_calc_positive CHECK (num_calculations IS NULL OR num_calculations >= 0),
    PRIMARY KEY (id, period_start)
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('gl_psp.gl_psp_aggregations', 'period_start',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_psp_agg_org ON gl_psp.gl_psp_aggregations(org_id, period_start DESC);
CREATE INDEX idx_psp_agg_tenant ON gl_psp.gl_psp_aggregations(tenant_id, period_start DESC);
CREATE INDEX idx_psp_agg_period ON gl_psp.gl_psp_aggregations(period_start, period_end);
CREATE INDEX idx_psp_agg_by_category ON gl_psp.gl_psp_aggregations USING GIN(by_category);
CREATE INDEX idx_psp_agg_by_method ON gl_psp.gl_psp_aggregations USING GIN(by_method);
CREATE INDEX idx_psp_agg_by_country ON gl_psp.gl_psp_aggregations USING GIN(by_country);

COMMENT ON TABLE gl_psp.gl_psp_aggregations IS 'Period aggregations of Scope 3 Cat 10 emissions with breakdowns by category, method, and country. TimescaleDB hypertable on period_start (30-day chunks)';
COMMENT ON COLUMN gl_psp.gl_psp_aggregations.org_id IS 'Organization UUID (reporting entity)';
COMMENT ON COLUMN gl_psp.gl_psp_aggregations.by_category IS 'JSONB breakdown of emissions by product category';
COMMENT ON COLUMN gl_psp.gl_psp_aggregations.by_method IS 'JSONB breakdown of emissions by calculation method';
COMMENT ON COLUMN gl_psp.gl_psp_aggregations.by_country IS 'JSONB breakdown of emissions by processing country';

-- =====================================================================================
-- TABLE 14: gl_psp_provenance_records (OPERATIONAL)
-- Description: Provenance chain hashes for complete audit trail (SHA-256)
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_provenance_records (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    calculation_id UUID NOT NULL,
    stage VARCHAR(32) NOT NULL,
    input_hash VARCHAR(128) NOT NULL,
    output_hash VARCHAR(128) NOT NULL,
    chain_hash VARCHAR(128) NOT NULL,
    merkle_root VARCHAR(128),
    metadata JSONB DEFAULT '{}',
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_prov_stage CHECK (stage IN (
        'VALIDATE', 'CLASSIFY', 'NORMALIZE', 'RESOLVE_EFS',
        'CALCULATE', 'ALLOCATE', 'AGGREGATE', 'COMPLIANCE',
        'PROVENANCE', 'SEAL'
    ))
);

CREATE INDEX idx_psp_prov_calculation ON gl_psp.gl_psp_provenance_records(calculation_id);
CREATE INDEX idx_psp_prov_stage ON gl_psp.gl_psp_provenance_records(stage);
CREATE INDEX idx_psp_prov_tenant ON gl_psp.gl_psp_provenance_records(tenant_id);
CREATE INDEX idx_psp_prov_input ON gl_psp.gl_psp_provenance_records(input_hash);
CREATE INDEX idx_psp_prov_output ON gl_psp.gl_psp_provenance_records(output_hash);
CREATE INDEX idx_psp_prov_chain ON gl_psp.gl_psp_provenance_records(chain_hash);
CREATE INDEX idx_psp_prov_merkle ON gl_psp.gl_psp_provenance_records(merkle_root);
CREATE INDEX idx_psp_prov_created ON gl_psp.gl_psp_provenance_records(created_at DESC);
CREATE INDEX idx_psp_prov_metadata ON gl_psp.gl_psp_provenance_records USING GIN(metadata);

COMMENT ON TABLE gl_psp.gl_psp_provenance_records IS 'SHA-256 provenance chain records for Scope 3 Cat 10 calculations across 10 pipeline stages';
COMMENT ON COLUMN gl_psp.gl_psp_provenance_records.stage IS 'Processing stage: VALIDATE, CLASSIFY, NORMALIZE, RESOLVE_EFS, CALCULATE, ALLOCATE, AGGREGATE, COMPLIANCE, PROVENANCE, SEAL';
COMMENT ON COLUMN gl_psp.gl_psp_provenance_records.input_hash IS 'SHA-256 hash of stage input data';
COMMENT ON COLUMN gl_psp.gl_psp_provenance_records.output_hash IS 'SHA-256 hash of stage output data';
COMMENT ON COLUMN gl_psp.gl_psp_provenance_records.chain_hash IS 'SHA-256 hash chaining input_hash + output_hash + previous chain_hash';
COMMENT ON COLUMN gl_psp.gl_psp_provenance_records.merkle_root IS 'Merkle tree root hash for batch provenance verification';

-- =====================================================================================
-- TABLE 15: gl_psp_compliance_results (OPERATIONAL / HYPERTABLE)
-- Description: Compliance check results against multiple regulatory frameworks
-- Hypertable on checked_at for time-series optimization
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_compliance_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    framework VARCHAR(32) NOT NULL,
    status VARCHAR(16) NOT NULL,
    rules_checked INTEGER,
    rules_passed INTEGER,
    rules_failed INTEGER,
    findings JSONB DEFAULT '[]',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_compl_framework CHECK (framework IN (
        'GHG_PROTOCOL', 'ISO_14064', 'CSRD_ESRS', 'CDP', 'SBTI', 'SB_253', 'GRI_305'
    )),
    CONSTRAINT chk_psp_compl_status CHECK (status IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE')),
    CONSTRAINT chk_psp_compl_rules_positive CHECK (rules_checked IS NULL OR rules_checked >= 0),
    CONSTRAINT chk_psp_compl_passed_positive CHECK (rules_passed IS NULL OR rules_passed >= 0),
    CONSTRAINT chk_psp_compl_failed_positive CHECK (rules_failed IS NULL OR rules_failed >= 0),
    CONSTRAINT chk_psp_compl_rules_sum CHECK (
        rules_checked IS NULL OR rules_passed IS NULL OR rules_failed IS NULL
        OR (rules_passed + rules_failed <= rules_checked)
    ),
    PRIMARY KEY (id, checked_at)
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('gl_psp.gl_psp_compliance_results', 'checked_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_psp_compl_calculation ON gl_psp.gl_psp_compliance_results(calculation_id);
CREATE INDEX idx_psp_compl_framework ON gl_psp.gl_psp_compliance_results(framework);
CREATE INDEX idx_psp_compl_status ON gl_psp.gl_psp_compliance_results(status);
CREATE INDEX idx_psp_compl_tenant ON gl_psp.gl_psp_compliance_results(tenant_id, checked_at DESC);
CREATE INDEX idx_psp_compl_findings ON gl_psp.gl_psp_compliance_results USING GIN(findings);
CREATE INDEX idx_psp_compl_calc_framework ON gl_psp.gl_psp_compliance_results(calculation_id, framework);

COMMENT ON TABLE gl_psp.gl_psp_compliance_results IS 'Compliance check results for Scope 3 Cat 10 against 7 regulatory frameworks. TimescaleDB hypertable on checked_at (30-day chunks)';
COMMENT ON COLUMN gl_psp.gl_psp_compliance_results.framework IS 'Regulatory framework: GHG_PROTOCOL, ISO_14064, CSRD_ESRS, CDP, SBTI, SB_253, GRI_305';
COMMENT ON COLUMN gl_psp.gl_psp_compliance_results.status IS 'Compliance status: PASS, FAIL, WARNING, NOT_APPLICABLE';
COMMENT ON COLUMN gl_psp.gl_psp_compliance_results.findings IS 'JSONB array of compliance findings [{rule_id, severity, message, recommendation}]';

-- =====================================================================================
-- TABLE 16: gl_psp_data_quality_scores (OPERATIONAL)
-- Description: Data Quality Indicator (DQI) scores for calculation inputs
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_data_quality_scores (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    calculation_id UUID NOT NULL,
    reliability NUMERIC(10,4),
    completeness NUMERIC(10,4),
    temporal NUMERIC(10,4),
    geographical NUMERIC(10,4),
    technological NUMERIC(10,4),
    overall NUMERIC(10,4),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_dqs_reliability_range CHECK (reliability IS NULL OR (reliability >= 1.0 AND reliability <= 5.0)),
    CONSTRAINT chk_psp_dqs_completeness_range CHECK (completeness IS NULL OR (completeness >= 1.0 AND completeness <= 5.0)),
    CONSTRAINT chk_psp_dqs_temporal_range CHECK (temporal IS NULL OR (temporal >= 1.0 AND temporal <= 5.0)),
    CONSTRAINT chk_psp_dqs_geographical_range CHECK (geographical IS NULL OR (geographical >= 1.0 AND geographical <= 5.0)),
    CONSTRAINT chk_psp_dqs_technological_range CHECK (technological IS NULL OR (technological >= 1.0 AND technological <= 5.0)),
    CONSTRAINT chk_psp_dqs_overall_range CHECK (overall IS NULL OR (overall >= 1.0 AND overall <= 5.0)),
    CONSTRAINT uq_psp_dqs_calc_tenant UNIQUE (calculation_id, tenant_id)
);

CREATE INDEX idx_psp_dqs_calculation ON gl_psp.gl_psp_data_quality_scores(calculation_id);
CREATE INDEX idx_psp_dqs_tenant ON gl_psp.gl_psp_data_quality_scores(tenant_id);
CREATE INDEX idx_psp_dqs_overall ON gl_psp.gl_psp_data_quality_scores(overall);
CREATE INDEX idx_psp_dqs_created ON gl_psp.gl_psp_data_quality_scores(created_at DESC);

COMMENT ON TABLE gl_psp.gl_psp_data_quality_scores IS '5-dimension Data Quality Indicator scores for Scope 3 Cat 10 calculations (GHG Protocol DQI framework)';
COMMENT ON COLUMN gl_psp.gl_psp_data_quality_scores.reliability IS 'Data reliability score (1.0 = verified data, 5.0 = non-qualified estimates)';
COMMENT ON COLUMN gl_psp.gl_psp_data_quality_scores.completeness IS 'Data completeness score (1.0 = all data, 5.0 = highly incomplete)';
COMMENT ON COLUMN gl_psp.gl_psp_data_quality_scores.temporal IS 'Temporal representativeness (1.0 = same year, 5.0 = >15 years old)';
COMMENT ON COLUMN gl_psp.gl_psp_data_quality_scores.geographical IS 'Geographical representativeness (1.0 = same country, 5.0 = global average)';
COMMENT ON COLUMN gl_psp.gl_psp_data_quality_scores.technological IS 'Technological representativeness (1.0 = exact process, 5.0 = generic)';
COMMENT ON COLUMN gl_psp.gl_psp_data_quality_scores.overall IS 'Weighted overall DQI score (average of 5 dimensions)';

-- =====================================================================================
-- TABLE 17: gl_psp_uncertainty_results (OPERATIONAL)
-- Description: Uncertainty analysis results (Monte Carlo, parametric, analytical)
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_uncertainty_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    calculation_id UUID NOT NULL,
    method VARCHAR(32) NOT NULL,
    mean_value NUMERIC(20,8) NOT NULL,
    std_dev NUMERIC(20,8),
    ci_lower NUMERIC(20,8),
    ci_upper NUMERIC(20,8),
    confidence_level NUMERIC(10,4),
    iterations INTEGER,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_unc_method CHECK (method IN (
        'MONTE_CARLO', 'PARAMETRIC', 'ANALYTICAL', 'PEDIGREE_MATRIX'
    )),
    CONSTRAINT chk_psp_unc_mean_positive CHECK (mean_value >= 0),
    CONSTRAINT chk_psp_unc_stddev_positive CHECK (std_dev IS NULL OR std_dev >= 0),
    CONSTRAINT chk_psp_unc_ci_order CHECK (ci_lower IS NULL OR ci_upper IS NULL OR ci_lower <= ci_upper),
    CONSTRAINT chk_psp_unc_confidence_range CHECK (confidence_level IS NULL OR (confidence_level > 0 AND confidence_level <= 1.0)),
    CONSTRAINT chk_psp_unc_iterations_positive CHECK (iterations IS NULL OR iterations > 0),
    CONSTRAINT uq_psp_unc_calc_method_tenant UNIQUE (calculation_id, method, tenant_id)
);

CREATE INDEX idx_psp_unc_calculation ON gl_psp.gl_psp_uncertainty_results(calculation_id);
CREATE INDEX idx_psp_unc_method ON gl_psp.gl_psp_uncertainty_results(method);
CREATE INDEX idx_psp_unc_tenant ON gl_psp.gl_psp_uncertainty_results(tenant_id);
CREATE INDEX idx_psp_unc_created ON gl_psp.gl_psp_uncertainty_results(created_at DESC);

COMMENT ON TABLE gl_psp.gl_psp_uncertainty_results IS 'Uncertainty analysis results for Scope 3 Cat 10 calculations using Monte Carlo, parametric, analytical, or pedigree matrix methods';
COMMENT ON COLUMN gl_psp.gl_psp_uncertainty_results.method IS 'Uncertainty analysis method: MONTE_CARLO, PARAMETRIC, ANALYTICAL, PEDIGREE_MATRIX';
COMMENT ON COLUMN gl_psp.gl_psp_uncertainty_results.mean_value IS 'Mean emission value from uncertainty analysis (kgCO2e)';
COMMENT ON COLUMN gl_psp.gl_psp_uncertainty_results.ci_lower IS 'Lower bound of confidence interval (kgCO2e)';
COMMENT ON COLUMN gl_psp.gl_psp_uncertainty_results.ci_upper IS 'Upper bound of confidence interval (kgCO2e)';
COMMENT ON COLUMN gl_psp.gl_psp_uncertainty_results.confidence_level IS 'Confidence level for interval (e.g., 0.95 = 95%)';
COMMENT ON COLUMN gl_psp.gl_psp_uncertainty_results.iterations IS 'Number of Monte Carlo iterations (NULL for analytical methods)';

-- =====================================================================================
-- TABLE 18: gl_psp_audit_trail (OPERATIONAL)
-- Description: Audit log for all processing of sold products operations
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_audit_trail (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entity_type VARCHAR(64) NOT NULL,
    entity_id UUID,
    action VARCHAR(16) NOT NULL,
    actor_id UUID,
    changes JSONB DEFAULT '{}',
    ip_address VARCHAR(45),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_audit_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'RECALCULATE',
        'VERIFY', 'APPROVE', 'REJECT', 'EXPORT', 'ARCHIVE'
    )),
    CONSTRAINT chk_psp_audit_entity_type CHECK (entity_type IN (
        'processing_emission_factor', 'energy_intensity_factor', 'grid_emission_factor',
        'fuel_emission_factor', 'eeio_sector_factor', 'processing_chain',
        'intermediate_product', 'customer_processing_data', 'calculation',
        'calculation_detail', 'aggregation', 'compliance_result',
        'data_quality_score', 'uncertainty_result', 'provenance_record', 'batch_job'
    ))
);

CREATE INDEX idx_psp_audit_entity ON gl_psp.gl_psp_audit_trail(entity_type, entity_id);
CREATE INDEX idx_psp_audit_action ON gl_psp.gl_psp_audit_trail(action);
CREATE INDEX idx_psp_audit_actor ON gl_psp.gl_psp_audit_trail(actor_id);
CREATE INDEX idx_psp_audit_tenant ON gl_psp.gl_psp_audit_trail(tenant_id);
CREATE INDEX idx_psp_audit_created ON gl_psp.gl_psp_audit_trail(created_at DESC);
CREATE INDEX idx_psp_audit_changes ON gl_psp.gl_psp_audit_trail USING GIN(changes);
CREATE INDEX idx_psp_audit_tenant_created ON gl_psp.gl_psp_audit_trail(tenant_id, created_at DESC);

COMMENT ON TABLE gl_psp.gl_psp_audit_trail IS 'Comprehensive audit log for all Scope 3 Cat 10 operations (CRUD, calculations, compliance checks, exports)';
COMMENT ON COLUMN gl_psp.gl_psp_audit_trail.entity_type IS 'Entity type being audited (16 entity types across all PSP tables)';
COMMENT ON COLUMN gl_psp.gl_psp_audit_trail.action IS 'Action type: CREATE, UPDATE, DELETE, CALCULATE, RECALCULATE, VERIFY, APPROVE, REJECT, EXPORT, ARCHIVE';
COMMENT ON COLUMN gl_psp.gl_psp_audit_trail.changes IS 'JSONB object with before/after values for UPDATE actions, parameters for CALCULATE actions';
COMMENT ON COLUMN gl_psp.gl_psp_audit_trail.ip_address IS 'IPv4 or IPv6 address of the actor (up to 45 characters for IPv6)';

-- =====================================================================================
-- TABLE 19: gl_psp_batch_jobs (OPERATIONAL)
-- Description: Batch job tracking for bulk processing calculations
-- =====================================================================================

CREATE TABLE gl_psp.gl_psp_batch_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    job_name VARCHAR(256) NOT NULL,
    status VARCHAR(16) NOT NULL,
    total_items INTEGER,
    completed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_details JSONB DEFAULT '[]',
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_psp_bj_status CHECK (status IN (
        'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'PARTIAL'
    )),
    CONSTRAINT chk_psp_bj_total_positive CHECK (total_items IS NULL OR total_items >= 0),
    CONSTRAINT chk_psp_bj_completed_positive CHECK (completed_items IS NULL OR completed_items >= 0),
    CONSTRAINT chk_psp_bj_failed_positive CHECK (failed_items IS NULL OR failed_items >= 0),
    CONSTRAINT chk_psp_bj_items_sum CHECK (
        total_items IS NULL OR completed_items IS NULL OR failed_items IS NULL
        OR (completed_items + failed_items <= total_items)
    ),
    CONSTRAINT chk_psp_bj_dates_order CHECK (
        started_at IS NULL OR completed_at IS NULL OR completed_at >= started_at
    )
);

CREATE INDEX idx_psp_bj_status ON gl_psp.gl_psp_batch_jobs(status);
CREATE INDEX idx_psp_bj_tenant ON gl_psp.gl_psp_batch_jobs(tenant_id);
CREATE INDEX idx_psp_bj_started ON gl_psp.gl_psp_batch_jobs(started_at DESC);
CREATE INDEX idx_psp_bj_completed ON gl_psp.gl_psp_batch_jobs(completed_at DESC);
CREATE INDEX idx_psp_bj_created ON gl_psp.gl_psp_batch_jobs(created_at DESC);
CREATE INDEX idx_psp_bj_error_details ON gl_psp.gl_psp_batch_jobs USING GIN(error_details);
CREATE INDEX idx_psp_bj_tenant_status ON gl_psp.gl_psp_batch_jobs(tenant_id, status);
CREATE INDEX idx_psp_bj_active ON gl_psp.gl_psp_batch_jobs(status) WHERE status IN ('PENDING', 'RUNNING');

COMMENT ON TABLE gl_psp.gl_psp_batch_jobs IS 'Batch job tracking for bulk Scope 3 Cat 10 calculations with progress monitoring and error reporting';
COMMENT ON COLUMN gl_psp.gl_psp_batch_jobs.status IS 'Job status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, PARTIAL';
COMMENT ON COLUMN gl_psp.gl_psp_batch_jobs.total_items IS 'Total number of products/items in the batch';
COMMENT ON COLUMN gl_psp.gl_psp_batch_jobs.error_details IS 'JSONB array of error details [{item_id, error_code, message, timestamp}]';

-- =====================================================================================
-- CONTINUOUS AGGREGATES
-- =====================================================================================

-- Continuous Aggregate 1: Daily Emissions by Category
CREATE MATERIALIZED VIEW gl_psp.gl_psp_daily_emissions_by_category
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', c.calculated_at) AS bucket,
    c.tenant_id,
    cd.category,
    COUNT(*) AS calc_count,
    SUM(cd.emissions_kg) AS total_emissions_kg,
    SUM(cd.emissions_kg) / 1000.0 AS total_emissions_tco2e,
    AVG(cd.dqi_score) AS avg_dqi_score,
    COUNT(DISTINCT cd.product_id) AS unique_products
FROM gl_psp.gl_psp_calculations c
JOIN gl_psp.gl_psp_calculation_details cd ON cd.calculation_id = c.id
GROUP BY bucket, c.tenant_id, cd.category
WITH NO DATA;

-- Refresh policy for daily category emissions (refresh every 6 hours, lag 12 hours)
SELECT add_continuous_aggregate_policy('gl_psp.gl_psp_daily_emissions_by_category',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW gl_psp.gl_psp_daily_emissions_by_category IS 'Daily aggregation of Scope 3 Cat 10 processing emissions by product category with DQI scores and product counts';

-- Continuous Aggregate 2: Monthly Emissions by Method
CREATE MATERIALIZED VIEW gl_psp.gl_psp_monthly_emissions_by_method
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('30 days', calculated_at) AS bucket,
    tenant_id,
    calculation_method,
    COUNT(*) AS calc_count,
    SUM(total_emissions_kg) AS total_emissions_kg,
    SUM(total_emissions_tco2e) AS total_emissions_tco2e,
    AVG(dqi_score) AS avg_dqi_score,
    SUM(num_products) AS total_products,
    AVG(total_emissions_tco2e) AS avg_emissions_tco2e
FROM gl_psp.gl_psp_calculations
GROUP BY bucket, tenant_id, calculation_method
WITH NO DATA;

-- Refresh policy for monthly method emissions (refresh every 12 hours, lag 1 day)
SELECT add_continuous_aggregate_policy('gl_psp.gl_psp_monthly_emissions_by_method',
    start_offset => INTERVAL '90 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '12 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW gl_psp.gl_psp_monthly_emissions_by_method IS 'Monthly aggregation of Scope 3 Cat 10 processing emissions by calculation method (SITE_SPECIFIC, AVERAGE_DATA, SPEND_BASED, HYBRID)';

-- =====================================================================================
-- ROW LEVEL SECURITY (RLS) - ALL OPERATIONAL TABLES
-- =====================================================================================

-- Enable RLS on all tables with tenant_id
ALTER TABLE gl_psp.gl_psp_processing_emission_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_energy_intensity_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_customer_processing_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_calculation_details ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_aggregations ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_provenance_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_compliance_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_data_quality_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_psp.gl_psp_audit_trail ENABLE ROW LEVEL SECURITY;

-- RLS Policies: tenant_id isolation on all operational tables
CREATE POLICY psp_pef_tenant_isolation ON gl_psp.gl_psp_processing_emission_factors
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_eif_tenant_isolation ON gl_psp.gl_psp_energy_intensity_factors
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_cpd_tenant_isolation ON gl_psp.gl_psp_customer_processing_data
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_calc_tenant_isolation ON gl_psp.gl_psp_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_cd_tenant_isolation ON gl_psp.gl_psp_calculation_details
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_agg_tenant_isolation ON gl_psp.gl_psp_aggregations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_prov_tenant_isolation ON gl_psp.gl_psp_provenance_records
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_compl_tenant_isolation ON gl_psp.gl_psp_compliance_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_dqs_tenant_isolation ON gl_psp.gl_psp_data_quality_scores
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY psp_audit_tenant_isolation ON gl_psp.gl_psp_audit_trail
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- =====================================================================================
-- SEED DATA: PROCESSING EMISSION FACTORS (12 product categories)
-- Source: DEFRA 2024, Ecoinvent 3.10, EPA, Industry averages
-- =====================================================================================

INSERT INTO gl_psp.gl_psp_processing_emission_factors
(category, processing_type, ef_value, ef_unit, source, region, reference_year, uncertainty, tenant_id) VALUES
-- Metals - Ferrous (steel processing: machining, forging, casting)
('METALS_FERROUS',          'MACHINING',            280.00000000, 'kgCO2e/tonne', 'DEFRA_2024',      'GLOBAL', 2024, 0.1500, '00000000-0000-0000-0000-000000000000'::UUID),
-- Metals - Non-Ferrous (aluminium, copper processing)
('METALS_NON_FERROUS',      'SMELTING',             520.00000000, 'kgCO2e/tonne', 'ECOINVENT_3.10',  'GLOBAL', 2024, 0.1200, '00000000-0000-0000-0000-000000000000'::UUID),
-- Plastics - Thermoplastic (injection molding, extrusion)
('PLASTICS_THERMOPLASTIC',  'INJECTION_MOLDING',    350.00000000, 'kgCO2e/tonne', 'DEFRA_2024',      'GLOBAL', 2024, 0.1000, '00000000-0000-0000-0000-000000000000'::UUID),
-- Plastics - Thermoset (curing, compression molding)
('PLASTICS_THERMOSET',      'COMPRESSION_MOLDING',  420.00000000, 'kgCO2e/tonne', 'ECOINVENT_3.10',  'GLOBAL', 2024, 0.1300, '00000000-0000-0000-0000-000000000000'::UUID),
-- Chemicals (chemical synthesis, distillation, blending)
('CHEMICALS',               'CHEMICAL_SYNTHESIS',   380.00000000, 'kgCO2e/tonne', 'EPA_2024',        'GLOBAL', 2024, 0.2000, '00000000-0000-0000-0000-000000000000'::UUID),
-- Food Ingredients (milling, drying, mixing, pasteurizing)
('FOOD_INGREDIENTS',        'FOOD_PROCESSING',      150.00000000, 'kgCO2e/tonne', 'DEFRA_2024',      'GLOBAL', 2024, 0.1800, '00000000-0000-0000-0000-000000000000'::UUID),
-- Textiles (spinning, weaving, dyeing, finishing)
('TEXTILES',                'TEXTILE_PROCESSING',   290.00000000, 'kgCO2e/tonne', 'ECOINVENT_3.10',  'GLOBAL', 2024, 0.1500, '00000000-0000-0000-0000-000000000000'::UUID),
-- Electronics (SMT assembly, wave soldering, testing)
('ELECTRONICS',             'PCB_ASSEMBLY',         450.00000000, 'kgCO2e/tonne', 'EPA_2024',        'GLOBAL', 2024, 0.2000, '00000000-0000-0000-0000-000000000000'::UUID),
-- Glass & Ceramics (melting, forming, annealing)
('GLASS_CERAMICS',          'GLASS_FORMING',        480.00000000, 'kgCO2e/tonne', 'ECOINVENT_3.10',  'GLOBAL', 2024, 0.1400, '00000000-0000-0000-0000-000000000000'::UUID),
-- Wood & Paper (pulping, papermaking, converting)
('WOOD_PAPER',              'PAPERMAKING',          210.00000000, 'kgCO2e/tonne', 'DEFRA_2024',      'GLOBAL', 2024, 0.1200, '00000000-0000-0000-0000-000000000000'::UUID),
-- Minerals (crushing, grinding, calcining)
('MINERALS',                'CALCINING',            560.00000000, 'kgCO2e/tonne', 'ECOINVENT_3.10',  'GLOBAL', 2024, 0.1600, '00000000-0000-0000-0000-000000000000'::UUID),
-- Agricultural (cleaning, sorting, drying, milling)
('AGRICULTURAL',            'GRAIN_MILLING',        110.00000000, 'kgCO2e/tonne', 'DEFRA_2024',      'GLOBAL', 2024, 0.2000, '00000000-0000-0000-0000-000000000000'::UUID);

-- =====================================================================================
-- SEED DATA: ENERGY INTENSITY FACTORS (18 processing types)
-- Source: IEA Industrial Energy Efficiency, DOE Manufacturing Bandwidth Studies
-- =====================================================================================

INSERT INTO gl_psp.gl_psp_energy_intensity_factors
(processing_type, low, mid, high, default_value, unit, source, tenant_id) VALUES
('MACHINING',               150.00000000, 280.00000000,  450.00000000,  280.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('CASTING',                 300.00000000, 500.00000000,  750.00000000,  500.00000000, 'kWh/tonne', 'DOE_BANDWIDTH',   '00000000-0000-0000-0000-000000000000'::UUID),
('FORGING',                 200.00000000, 380.00000000,  600.00000000,  380.00000000, 'kWh/tonne', 'DOE_BANDWIDTH',   '00000000-0000-0000-0000-000000000000'::UUID),
('SMELTING',                800.00000000, 1200.00000000, 1800.00000000, 1200.00000000, 'kWh/tonne', 'IEA_2024',       '00000000-0000-0000-0000-000000000000'::UUID),
('INJECTION_MOLDING',       250.00000000, 420.00000000,  650.00000000,  420.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('EXTRUSION',               180.00000000, 320.00000000,  500.00000000,  320.00000000, 'kWh/tonne', 'DOE_BANDWIDTH',   '00000000-0000-0000-0000-000000000000'::UUID),
('BLOW_MOLDING',            200.00000000, 350.00000000,  550.00000000,  350.00000000, 'kWh/tonne', 'DOE_BANDWIDTH',   '00000000-0000-0000-0000-000000000000'::UUID),
('COMPRESSION_MOLDING',     280.00000000, 480.00000000,  720.00000000,  480.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('CHEMICAL_SYNTHESIS',      350.00000000, 600.00000000,  900.00000000,  600.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('DISTILLATION',            400.00000000, 700.00000000,  1100.00000000, 700.00000000, 'kWh/tonne', 'DOE_BANDWIDTH',   '00000000-0000-0000-0000-000000000000'::UUID),
('FOOD_PROCESSING',         100.00000000, 200.00000000,  350.00000000,  200.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('TEXTILE_PROCESSING',      250.00000000, 420.00000000,  650.00000000,  420.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('PCB_ASSEMBLY',            350.00000000, 580.00000000,  850.00000000,  580.00000000, 'kWh/tonne', 'DOE_BANDWIDTH',   '00000000-0000-0000-0000-000000000000'::UUID),
('GLASS_FORMING',           500.00000000, 800.00000000,  1200.00000000, 800.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('PAPERMAKING',             150.00000000, 280.00000000,  450.00000000,  280.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('CALCINING',               600.00000000, 950.00000000,  1400.00000000, 950.00000000, 'kWh/tonne', 'DOE_BANDWIDTH',   '00000000-0000-0000-0000-000000000000'::UUID),
('GRAIN_MILLING',           80.00000000,  150.00000000,  250.00000000,  150.00000000, 'kWh/tonne', 'IEA_2024',        '00000000-0000-0000-0000-000000000000'::UUID),
('TEXTILE_FINISHING',       280.00000000, 420.00000000,  600.00000000,  420.00000000, 'kWh/tonne', 'DOE_BANDWIDTH',   '00000000-0000-0000-0000-000000000000'::UUID);

-- =====================================================================================
-- SEED DATA: GRID EMISSION FACTORS (16 countries/regions)
-- Source: IEA 2024, eGRID 2024, DEFRA 2024
-- =====================================================================================

INSERT INTO gl_psp.gl_psp_grid_emission_factors
(region, country_name, ef_value, ef_unit, reference_year, source, tenant_id) VALUES
('US',     'United States',       0.41700000, 'kgCO2e/kWh', 2024, 'eGRID_2024',  '00000000-0000-0000-0000-000000000000'::UUID),
('GB',     'United Kingdom',      0.21200000, 'kgCO2e/kWh', 2024, 'DEFRA_2024',  '00000000-0000-0000-0000-000000000000'::UUID),
('DE',     'Germany',             0.36400000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('FR',     'France',              0.05690000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('JP',     'Japan',               0.45700000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('CN',     'China',               0.55700000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('IN',     'India',               0.70800000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('KR',     'South Korea',         0.41500000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('BR',     'Brazil',              0.07400000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('CA',     'Canada',              0.12000000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('AU',     'Australia',           0.61000000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('MX',     'Mexico',              0.43100000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('ZA',     'South Africa',        0.92800000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('PL',     'Poland',              0.63500000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('EU',     'European Union (27)', 0.25500000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID),
('GLOBAL', 'Global Average',      0.47500000, 'kgCO2e/kWh', 2024, 'IEA_2024',    '00000000-0000-0000-0000-000000000000'::UUID);

-- =====================================================================================
-- SEED DATA: FUEL EMISSION FACTORS (6 primary fuel types)
-- Source: DEFRA 2024, EPA 2024, IPCC AR5
-- =====================================================================================

INSERT INTO gl_psp.gl_psp_fuel_emission_factors
(fuel_type, ef_value, ef_unit, source, tenant_id) VALUES
('NATURAL_GAS', 2.02400000, 'kgCO2e/m3',    'DEFRA_2024', '00000000-0000-0000-0000-000000000000'::UUID),
('DIESEL',      2.68760000, 'kgCO2e/litre',  'DEFRA_2024', '00000000-0000-0000-0000-000000000000'::UUID),
('FUEL_OIL',    3.17800000, 'kgCO2e/litre',  'DEFRA_2024', '00000000-0000-0000-0000-000000000000'::UUID),
('LPG',         1.55370000, 'kgCO2e/litre',  'DEFRA_2024', '00000000-0000-0000-0000-000000000000'::UUID),
('COAL',        2450.00000000, 'kgCO2e/tonne', 'EPA_2024',  '00000000-0000-0000-0000-000000000000'::UUID),
('BIOMASS',     0.01500000, 'kgCO2e/kWh',    'IPCC_AR5',   '00000000-0000-0000-0000-000000000000'::UUID);

-- =====================================================================================
-- SEED DATA: EEIO SECTOR FACTORS (12 processing-related NAICS sectors)
-- Source: EPA USEEIO v2.0
-- =====================================================================================

INSERT INTO gl_psp.gl_psp_eeio_sector_factors
(naics_code, sector_name, ef_value, ef_unit, margin, source, tenant_id) VALUES
('331110', 'Iron and Steel Mills and Ferroalloy Manufacturing',              0.78200000, 'kgCO2e/USD', 0.1200, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('331313', 'Alumina Refining and Primary Aluminum Production',               1.12000000, 'kgCO2e/USD', 0.1000, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('326110', 'Plastics Packaging Materials and Unlaminated Film Manufacturing', 0.52400000, 'kgCO2e/USD', 0.1500, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('325110', 'Petrochemical Manufacturing',                                    0.89600000, 'kgCO2e/USD', 0.1100, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('311210', 'Flour Milling and Malt Manufacturing',                           0.31200000, 'kgCO2e/USD', 0.1800, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('313110', 'Fiber, Yarn, and Thread Mills',                                  0.45600000, 'kgCO2e/USD', 0.1400, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('334410', 'Semiconductor and Other Electronic Component Manufacturing',     0.28900000, 'kgCO2e/USD', 0.2000, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('327211', 'Flat Glass Manufacturing',                                       0.67800000, 'kgCO2e/USD', 0.1300, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('322110', 'Pulp Mills',                                                     0.58300000, 'kgCO2e/USD', 0.1200, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('327310', 'Cement and Concrete Product Manufacturing',                      0.94100000, 'kgCO2e/USD', 0.1000, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('311119', 'Other Animal Food Manufacturing',                                0.38500000, 'kgCO2e/USD', 0.1600, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID),
('332710', 'Machine Shops',                                                  0.34200000, 'kgCO2e/USD', 0.1700, 'EPA_USEEIO_v2', '00000000-0000-0000-0000-000000000000'::UUID);

-- =====================================================================================
-- SEED DATA: PROCESSING CHAINS (8 multi-step chain definitions)
-- =====================================================================================

INSERT INTO gl_psp.gl_psp_processing_chains
(chain_type, chain_name, steps, combined_ef, description, tenant_id) VALUES
('STEEL_FABRICATION', 'Steel Sheet to Automotive Part',
    '[{"step": 1, "process": "CUTTING", "ef_value": 45.0, "unit": "kgCO2e/tonne", "energy_kwh": 80},
      {"step": 2, "process": "STAMPING", "ef_value": 65.0, "unit": "kgCO2e/tonne", "energy_kwh": 120},
      {"step": 3, "process": "WELDING", "ef_value": 85.0, "unit": "kgCO2e/tonne", "energy_kwh": 150},
      {"step": 4, "process": "SURFACE_TREATMENT", "ef_value": 55.0, "unit": "kgCO2e/tonne", "energy_kwh": 95}]'::JSONB,
    250.00000000, 'Steel sheet processed into automotive body panels via cutting, stamping, welding, and surface treatment',
    '00000000-0000-0000-0000-000000000000'::UUID),

('ALUMINUM_EXTRUSION', 'Aluminum Billet to Profile',
    '[{"step": 1, "process": "PREHEATING", "ef_value": 120.0, "unit": "kgCO2e/tonne", "energy_kwh": 200},
      {"step": 2, "process": "EXTRUSION", "ef_value": 180.0, "unit": "kgCO2e/tonne", "energy_kwh": 320},
      {"step": 3, "process": "AGING", "ef_value": 60.0, "unit": "kgCO2e/tonne", "energy_kwh": 100},
      {"step": 4, "process": "ANODIZING", "ef_value": 95.0, "unit": "kgCO2e/tonne", "energy_kwh": 160}]'::JSONB,
    455.00000000, 'Aluminum billet extruded into structural profiles with aging and anodizing finish',
    '00000000-0000-0000-0000-000000000000'::UUID),

('PLASTIC_MOLDING_ASSEMBLY', 'Plastic Pellets to Consumer Product',
    '[{"step": 1, "process": "DRYING", "ef_value": 25.0, "unit": "kgCO2e/tonne", "energy_kwh": 45},
      {"step": 2, "process": "INJECTION_MOLDING", "ef_value": 180.0, "unit": "kgCO2e/tonne", "energy_kwh": 420},
      {"step": 3, "process": "TRIMMING", "ef_value": 15.0, "unit": "kgCO2e/tonne", "energy_kwh": 25},
      {"step": 4, "process": "ASSEMBLY", "ef_value": 30.0, "unit": "kgCO2e/tonne", "energy_kwh": 50}]'::JSONB,
    250.00000000, 'Thermoplastic pellets processed via injection molding into assembled consumer products',
    '00000000-0000-0000-0000-000000000000'::UUID),

('CHEMICAL_FORMULATION', 'Chemical Intermediates to Formulated Product',
    '[{"step": 1, "process": "BLENDING", "ef_value": 40.0, "unit": "kgCO2e/tonne", "energy_kwh": 70},
      {"step": 2, "process": "REACTION", "ef_value": 200.0, "unit": "kgCO2e/tonne", "energy_kwh": 350},
      {"step": 3, "process": "DISTILLATION", "ef_value": 150.0, "unit": "kgCO2e/tonne", "energy_kwh": 700},
      {"step": 4, "process": "PACKAGING", "ef_value": 20.0, "unit": "kgCO2e/tonne", "energy_kwh": 35}]'::JSONB,
    410.00000000, 'Chemical intermediates blended, reacted, distilled, and packaged into formulated products',
    '00000000-0000-0000-0000-000000000000'::UUID),

('FOOD_PRODUCTION', 'Flour to Baked Goods',
    '[{"step": 1, "process": "MIXING", "ef_value": 15.0, "unit": "kgCO2e/tonne", "energy_kwh": 30},
      {"step": 2, "process": "PROOFING", "ef_value": 10.0, "unit": "kgCO2e/tonne", "energy_kwh": 20},
      {"step": 3, "process": "BAKING", "ef_value": 85.0, "unit": "kgCO2e/tonne", "energy_kwh": 180},
      {"step": 4, "process": "COOLING_PACKAGING", "ef_value": 25.0, "unit": "kgCO2e/tonne", "energy_kwh": 50}]'::JSONB,
    135.00000000, 'Flour and ingredients processed into baked goods via mixing, proofing, baking, and packaging',
    '00000000-0000-0000-0000-000000000000'::UUID),

('TEXTILE_GARMENT', 'Fabric to Finished Garment',
    '[{"step": 1, "process": "CUTTING", "ef_value": 20.0, "unit": "kgCO2e/tonne", "energy_kwh": 35},
      {"step": 2, "process": "SEWING", "ef_value": 40.0, "unit": "kgCO2e/tonne", "energy_kwh": 70},
      {"step": 3, "process": "DYEING", "ef_value": 160.0, "unit": "kgCO2e/tonne", "energy_kwh": 280},
      {"step": 4, "process": "FINISHING", "ef_value": 80.0, "unit": "kgCO2e/tonne", "energy_kwh": 140}]'::JSONB,
    300.00000000, 'Fabric processed into finished garments via cutting, sewing, dyeing, and finishing',
    '00000000-0000-0000-0000-000000000000'::UUID),

('ELECTRONICS_ASSEMBLY', 'Components to Assembled PCB',
    '[{"step": 1, "process": "SMT_PLACEMENT", "ef_value": 120.0, "unit": "kgCO2e/tonne", "energy_kwh": 200},
      {"step": 2, "process": "REFLOW_SOLDERING", "ef_value": 180.0, "unit": "kgCO2e/tonne", "energy_kwh": 300},
      {"step": 3, "process": "INSPECTION", "ef_value": 25.0, "unit": "kgCO2e/tonne", "energy_kwh": 40},
      {"step": 4, "process": "CONFORMAL_COATING", "ef_value": 45.0, "unit": "kgCO2e/tonne", "energy_kwh": 75}]'::JSONB,
    370.00000000, 'Electronic components assembled onto PCBs via SMT, reflow soldering, and conformal coating',
    '00000000-0000-0000-0000-000000000000'::UUID),

('GLASS_CONTAINER', 'Glass Batch to Container',
    '[{"step": 1, "process": "MELTING", "ef_value": 280.0, "unit": "kgCO2e/tonne", "energy_kwh": 500},
      {"step": 2, "process": "FORMING", "ef_value": 80.0, "unit": "kgCO2e/tonne", "energy_kwh": 140},
      {"step": 3, "process": "ANNEALING", "ef_value": 60.0, "unit": "kgCO2e/tonne", "energy_kwh": 100},
      {"step": 4, "process": "INSPECTION_PACKING", "ef_value": 15.0, "unit": "kgCO2e/tonne", "energy_kwh": 25}]'::JSONB,
    435.00000000, 'Glass batch materials melted, formed, annealed, and inspected into finished containers',
    '00000000-0000-0000-0000-000000000000'::UUID);

-- =====================================================================================
-- SEED DATA: CURRENCY CONVERSION RATES (12 currencies)
-- Source: IMF SDR rates, World Bank, as of 2024-12-31
-- =====================================================================================

INSERT INTO gl_psp.gl_psp_currencies
(currency_code, currency_name, usd_rate, reference_date, tenant_id) VALUES
('USD', 'US Dollar',             1.00000000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('EUR', 'Euro',                  1.10400000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('GBP', 'British Pound',         1.27300000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('JPY', 'Japanese Yen',          0.00710000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('CAD', 'Canadian Dollar',       0.74200000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('AUD', 'Australian Dollar',     0.68100000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('CHF', 'Swiss Franc',           1.12800000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('CNY', 'Chinese Yuan',          0.14100000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('INR', 'Indian Rupee',          0.01200000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('BRL', 'Brazilian Real',        0.20500000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('KRW', 'South Korean Won',      0.00077000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID),
('SEK', 'Swedish Krona',         0.09600000, '2024-12-31', '00000000-0000-0000-0000-000000000000'::UUID);

-- =====================================================================================
-- SEED DATA: CPI DEFLATORS (11 years, 2015-2025, base year 2024)
-- Source: US Bureau of Labor Statistics, World Bank
-- =====================================================================================

INSERT INTO gl_psp.gl_psp_cpi_deflators
(year, cpi_index, base_year, tenant_id) VALUES
(2015,  76.8000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2016,  78.0000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2017,  79.6000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2018,  81.6000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2019,  83.0000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2020,  84.0000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2021,  87.9000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2022,  95.0000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2023,  98.2000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2024, 100.0000, 2024, '00000000-0000-0000-0000-000000000000'::UUID),
(2025, 102.5000, 2024, '00000000-0000-0000-0000-000000000000'::UUID);

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
    'GL-MRV-S3-010',
    'Processing of Sold Products Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-023: Scope 3 Category 10 - Processing of Sold Products. Calculates emissions from downstream processing of intermediate products sold to customers. Supports site-specific (customer-provided energy/fuel data), average-data (category EFs + energy intensity), spend-based (EPA USEEIO v2 EEIO factors with CPI deflation and margin removal), and hybrid calculation methods. Includes 12 product category EFs, 18 energy intensity factors, 16 grid EFs, 6 fuel EFs, 12 EEIO sector factors, 8 multi-step processing chains, 12 currencies, 11 CPI deflators. 5-dimension DQI scoring, Monte Carlo uncertainty analysis, SHA-256 provenance chains, 7-framework compliance (GHG Protocol, ISO 14064, CSRD ESRS, CDP, SBTi, SB 253, GRI 305).',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 10,
        'category_name', 'Processing of Sold Products',
        'calculation_methods', jsonb_build_array('SITE_SPECIFIC', 'AVERAGE_DATA', 'SPEND_BASED', 'HYBRID'),
        'product_categories', jsonb_build_array(
            'METALS_FERROUS', 'METALS_NON_FERROUS', 'PLASTICS_THERMOPLASTIC',
            'PLASTICS_THERMOSET', 'CHEMICALS', 'FOOD_INGREDIENTS', 'TEXTILES',
            'ELECTRONICS', 'GLASS_CERAMICS', 'WOOD_PAPER', 'MINERALS', 'AGRICULTURAL'
        ),
        'processing_types', jsonb_build_array(
            'MACHINING', 'CASTING', 'FORGING', 'SMELTING', 'INJECTION_MOLDING',
            'EXTRUSION', 'BLOW_MOLDING', 'COMPRESSION_MOLDING', 'CHEMICAL_SYNTHESIS',
            'DISTILLATION', 'FOOD_PROCESSING', 'TEXTILE_PROCESSING', 'PCB_ASSEMBLY',
            'GLASS_FORMING', 'PAPERMAKING', 'CALCINING', 'GRAIN_MILLING', 'TEXTILE_FINISHING'
        ),
        'uncertainty_methods', jsonb_build_array('MONTE_CARLO', 'PARAMETRIC', 'ANALYTICAL', 'PEDIGREE_MATRIX'),
        'dqi_dimensions', jsonb_build_array('reliability', 'completeness', 'temporal', 'geographical', 'technological'),
        'frameworks', jsonb_build_array('GHG Protocol Scope 3', 'ISO 14064-1', 'CSRD ESRS E1', 'CDP Climate', 'SBTi', 'SB 253', 'GRI 305'),
        'processing_ef_count', 12,
        'energy_intensity_count', 18,
        'grid_ef_count', 16,
        'fuel_ef_count', 6,
        'eeio_factor_count', 12,
        'processing_chain_count', 8,
        'currency_count', 12,
        'cpi_deflator_count', 11,
        'supports_multi_step_chains', true,
        'supports_customer_specific_data', true,
        'supports_cpi_deflation', true,
        'supports_margin_removal', true,
        'supports_dqi_scoring', true,
        'supports_uncertainty_analysis', true,
        'supports_provenance_hashing', true,
        'supports_batch_processing', true,
        'default_ef_source', 'DEFRA_2024',
        'default_gwp', 'AR5',
        'schema', 'gl_psp',
        'table_prefix', 'gl_psp_',
        'hypertables', jsonb_build_array('gl_psp_calculations', 'gl_psp_aggregations', 'gl_psp_compliance_results'),
        'continuous_aggregates', jsonb_build_array('gl_psp_daily_emissions_by_category', 'gl_psp_monthly_emissions_by_method'),
        'migration_version', 'V074'
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

COMMENT ON SCHEMA gl_psp IS 'Updated: AGENT-MRV-023 complete with 19 tables, 3 hypertables, 2 continuous aggregates, 10 RLS policies, 95+ seed records';

-- =====================================================================================
-- END OF MIGRATION V074
-- =====================================================================================
-- Total Lines: ~1200
-- Total Tables: 19 (10 reference + 9 operational)
-- Total Hypertables: 3 (calculations, aggregations, compliance_results)
-- Total Continuous Aggregates: 2 (daily_emissions_by_category, monthly_emissions_by_method)
-- Total RLS Policies: 10 (processing_emission_factors, energy_intensity_factors,
--                         customer_processing_data, calculations, calculation_details,
--                         aggregations, provenance_records, compliance_results,
--                         data_quality_scores, audit_trail)
-- Total Indexes: 82
-- Total Constraints: 93 (CHECK + UNIQUE + PK)
-- Total Seed Records: 95
--   Processing Emission Factors: 12 (METALS_FERROUS through AGRICULTURAL)
--   Energy Intensity Factors: 18 (MACHINING through TEXTILE_FINISHING)
--   Grid Emission Factors: 16 (US/GB/DE/FR/JP/CN/IN/KR/BR/CA/AU/MX/ZA/PL/EU/GLOBAL)
--   Fuel Emission Factors: 6 (NATURAL_GAS/DIESEL/FUEL_OIL/LPG/COAL/BIOMASS)
--   EEIO Sector Factors: 12 (NAICS processing sectors)
--   Processing Chains: 8 (Steel/Aluminum/Plastic/Chemical/Food/Textile/Electronics/Glass)
--   Currency Rates: 12 (USD/EUR/GBP/JPY/CAD/AUD/CHF/CNY/INR/BRL/KRW/SEK)
--   CPI Deflators: 11 (2015-2025, base year 2024)
--   Agent Registry: 1
-- =====================================================================================

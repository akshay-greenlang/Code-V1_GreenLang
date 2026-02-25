-- ============================================================================
-- Migration: V066__capital_goods_service.sql
-- Description: Capital Goods Agent (AGENT-MRV-015) - Scope 3 Category 2
-- Author: GL-BackendDeveloper
-- Date: 2026-02-25
-- Dependencies: V065__purchased_goods_services.sql
-- ============================================================================
-- This migration creates the complete database schema for the Capital Goods
-- Agent, supporting EEIO spend-based, average-data, supplier-specific, and
-- hybrid calculation methods per GHG Protocol Scope 3 Category 2.
-- ============================================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS capital_goods_service;

-- ============================================================================
-- REFERENCE TABLES
-- ============================================================================

-- Asset category reference (8 primary categories + subcategories)
CREATE TABLE capital_goods_service.cg_asset_categories (
    category_id VARCHAR(50) PRIMARY KEY,
    category_name VARCHAR(200) NOT NULL,
    parent_category_id VARCHAR(50),
    scope3_category INTEGER NOT NULL DEFAULT 2 CHECK (scope3_category = 2),
    description TEXT,
    typical_useful_life_years INTEGER CHECK (typical_useful_life_years > 0),
    capitalization_threshold_usd DECIMAL(15,2),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_parent_category FOREIGN KEY (parent_category_id)
        REFERENCES capital_goods_service.cg_asset_categories(category_id)
);

CREATE INDEX idx_cg_asset_categories_parent ON capital_goods_service.cg_asset_categories(parent_category_id);
CREATE INDEX idx_cg_asset_categories_active ON capital_goods_service.cg_asset_categories(is_active);

COMMENT ON TABLE capital_goods_service.cg_asset_categories IS 'Asset category reference with 8 primary categories plus subcategories';

-- EEIO emission factors by NAICS code
CREATE TABLE capital_goods_service.cg_eeio_emission_factors (
    factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    naics_code VARCHAR(10) NOT NULL,
    naics_description TEXT,
    sector_name VARCHAR(200),
    emission_factor_kg_co2e_per_usd DECIMAL(15,6) NOT NULL CHECK (emission_factor_kg_co2e_per_usd >= 0),
    source VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL CHECK (year >= 2000 AND year <= 2100),
    region VARCHAR(10) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    scope_coverage VARCHAR(50),
    uncertainty_percentage DECIMAL(5,2) CHECK (uncertainty_percentage >= 0),
    data_quality_score DECIMAL(3,2) CHECK (data_quality_score >= 1.0 AND data_quality_score <= 5.0),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cg_eeio_tenant ON capital_goods_service.cg_eeio_emission_factors(tenant_id);
CREATE INDEX idx_cg_eeio_naics ON capital_goods_service.cg_eeio_emission_factors(naics_code);
CREATE INDEX idx_cg_eeio_year_region ON capital_goods_service.cg_eeio_emission_factors(year, region);
CREATE INDEX idx_cg_eeio_active ON capital_goods_service.cg_eeio_emission_factors(is_active);

COMMENT ON TABLE capital_goods_service.cg_eeio_emission_factors IS 'EEIO emission factors by NAICS code for spend-based calculations';

-- Physical emission factors by material type
CREATE TABLE capital_goods_service.cg_physical_emission_factors (
    factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    material_type VARCHAR(100) NOT NULL,
    material_subcategory VARCHAR(100),
    emission_factor_kg_co2e_per_unit DECIMAL(15,6) NOT NULL CHECK (emission_factor_kg_co2e_per_unit >= 0),
    unit VARCHAR(20) NOT NULL,
    source VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL CHECK (year >= 2000 AND year <= 2100),
    region VARCHAR(10) NOT NULL,
    lifecycle_stage VARCHAR(50),
    uncertainty_percentage DECIMAL(5,2) CHECK (uncertainty_percentage >= 0),
    data_quality_score DECIMAL(3,2) CHECK (data_quality_score >= 1.0 AND data_quality_score <= 5.0),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cg_physical_tenant ON capital_goods_service.cg_physical_emission_factors(tenant_id);
CREATE INDEX idx_cg_physical_material ON capital_goods_service.cg_physical_emission_factors(material_type);
CREATE INDEX idx_cg_physical_year_region ON capital_goods_service.cg_physical_emission_factors(year, region);
CREATE INDEX idx_cg_physical_active ON capital_goods_service.cg_physical_emission_factors(is_active);

COMMENT ON TABLE capital_goods_service.cg_physical_emission_factors IS 'Physical emission factors by material type for average-data calculations';

-- Supplier-specific emission factors
CREATE TABLE capital_goods_service.cg_supplier_emission_factors (
    factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    supplier_id VARCHAR(100) NOT NULL,
    supplier_name VARCHAR(200),
    asset_category_id VARCHAR(50),
    emission_factor_kg_co2e_per_unit DECIMAL(15,6) NOT NULL CHECK (emission_factor_kg_co2e_per_unit >= 0),
    unit VARCHAR(20) NOT NULL,
    scope_coverage VARCHAR(50),
    verification_status VARCHAR(50),
    verification_date DATE,
    data_source VARCHAR(100),
    year INTEGER NOT NULL CHECK (year >= 2000 AND year <= 2100),
    valid_from DATE NOT NULL,
    valid_until DATE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_supplier_category FOREIGN KEY (asset_category_id)
        REFERENCES capital_goods_service.cg_asset_categories(category_id)
);

CREATE INDEX idx_cg_supplier_tenant ON capital_goods_service.cg_supplier_emission_factors(tenant_id);
CREATE INDEX idx_cg_supplier_id ON capital_goods_service.cg_supplier_emission_factors(supplier_id);
CREATE INDEX idx_cg_supplier_category ON capital_goods_service.cg_supplier_emission_factors(asset_category_id);
CREATE INDEX idx_cg_supplier_active ON capital_goods_service.cg_supplier_emission_factors(is_active);

COMMENT ON TABLE capital_goods_service.cg_supplier_emission_factors IS 'Supplier-specific emission factors for capital goods';

-- Classification mappings (NAICS/NACE/ISIC/UNSPSC)
CREATE TABLE capital_goods_service.cg_classification_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    naics_code VARCHAR(10),
    nace_code VARCHAR(10),
    isic_code VARCHAR(10),
    unspsc_code VARCHAR(10),
    asset_category_id VARCHAR(50),
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_mapping_category FOREIGN KEY (asset_category_id)
        REFERENCES capital_goods_service.cg_asset_categories(category_id)
);

CREATE INDEX idx_cg_mapping_naics ON capital_goods_service.cg_classification_mappings(naics_code);
CREATE INDEX idx_cg_mapping_nace ON capital_goods_service.cg_classification_mappings(nace_code);
CREATE INDEX idx_cg_mapping_isic ON capital_goods_service.cg_classification_mappings(isic_code);
CREATE INDEX idx_cg_mapping_unspsc ON capital_goods_service.cg_classification_mappings(unspsc_code);
CREATE INDEX idx_cg_mapping_category ON capital_goods_service.cg_classification_mappings(asset_category_id);

COMMENT ON TABLE capital_goods_service.cg_classification_mappings IS 'Cross-mapping between NAICS/NACE/ISIC/UNSPSC classification systems';

-- Margin factors for purchaser-to-producer price conversion
CREATE TABLE capital_goods_service.cg_margin_factors (
    margin_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    sector_code VARCHAR(10) NOT NULL,
    sector_name VARCHAR(200),
    margin_percentage DECIMAL(5,2) NOT NULL CHECK (margin_percentage >= -100 AND margin_percentage <= 100),
    region VARCHAR(10) NOT NULL,
    year INTEGER NOT NULL CHECK (year >= 2000 AND year <= 2100),
    source VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cg_margin_tenant ON capital_goods_service.cg_margin_factors(tenant_id);
CREATE INDEX idx_cg_margin_sector ON capital_goods_service.cg_margin_factors(sector_code);
CREATE INDEX idx_cg_margin_year_region ON capital_goods_service.cg_margin_factors(year, region);

COMMENT ON TABLE capital_goods_service.cg_margin_factors IS 'Sector margin percentages for purchaser-to-producer price conversion';

-- ============================================================================
-- OPERATIONAL TABLES
-- ============================================================================

-- Registered capital assets
CREATE TABLE capital_goods_service.cg_capital_assets (
    asset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    asset_code VARCHAR(100) NOT NULL,
    asset_name VARCHAR(200) NOT NULL,
    category_id VARCHAR(50) NOT NULL,
    subcategory_id VARCHAR(50),
    capex_amount_usd DECIMAL(15,2) NOT NULL CHECK (capex_amount_usd >= 0),
    acquisition_date DATE NOT NULL,
    useful_life_years INTEGER CHECK (useful_life_years > 0),
    capitalization_policy VARCHAR(50),
    supplier_id VARCHAR(100),
    supplier_name VARCHAR(200),
    naics_code VARCHAR(10),
    quantity DECIMAL(15,3),
    unit VARCHAR(20),
    physical_description TEXT,
    location_id VARCHAR(100),
    department VARCHAR(100),
    project_id VARCHAR(100),
    metadata JSONB,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_asset_category FOREIGN KEY (category_id)
        REFERENCES capital_goods_service.cg_asset_categories(category_id),
    CONSTRAINT fk_cg_asset_subcategory FOREIGN KEY (subcategory_id)
        REFERENCES capital_goods_service.cg_asset_categories(category_id)
);

CREATE INDEX idx_cg_assets_tenant ON capital_goods_service.cg_capital_assets(tenant_id);
CREATE INDEX idx_cg_assets_code ON capital_goods_service.cg_capital_assets(asset_code);
CREATE INDEX idx_cg_assets_category ON capital_goods_service.cg_capital_assets(category_id);
CREATE INDEX idx_cg_assets_acquisition ON capital_goods_service.cg_capital_assets(acquisition_date);
CREATE INDEX idx_cg_assets_supplier ON capital_goods_service.cg_capital_assets(supplier_id);
CREATE INDEX idx_cg_assets_active ON capital_goods_service.cg_capital_assets(is_active);

COMMENT ON TABLE capital_goods_service.cg_capital_assets IS 'Registered capital assets with CAPEX and lifecycle data';

-- Calculation results
CREATE TABLE capital_goods_service.cg_calculations (
    calc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    reporting_period_start DATE NOT NULL,
    reporting_period_end DATE NOT NULL,
    calculation_method VARCHAR(50) NOT NULL,
    total_emissions_kg_co2e DECIMAL(15,3) NOT NULL CHECK (total_emissions_kg_co2e >= 0),
    total_co2_kg DECIMAL(15,3) CHECK (total_co2_kg >= 0),
    total_ch4_kg DECIMAL(15,3) CHECK (total_ch4_kg >= 0),
    total_n2o_kg DECIMAL(15,3) CHECK (total_n2o_kg >= 0),
    gwp_source VARCHAR(50) NOT NULL,
    asset_count INTEGER NOT NULL CHECK (asset_count >= 0),
    total_capex_usd DECIMAL(15,2) CHECK (total_capex_usd >= 0),
    provenance_hash VARCHAR(64) NOT NULL,
    calculation_metadata JSONB,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cg_calc_tenant ON capital_goods_service.cg_calculations(tenant_id);
CREATE INDEX idx_cg_calc_period ON capital_goods_service.cg_calculations(reporting_period_start, reporting_period_end);
CREATE INDEX idx_cg_calc_method ON capital_goods_service.cg_calculations(calculation_method);
CREATE INDEX idx_cg_calc_hash ON capital_goods_service.cg_calculations(provenance_hash);

COMMENT ON TABLE capital_goods_service.cg_calculations IS 'Capital goods calculation results by reporting period';

-- Per-asset calculation details
CREATE TABLE capital_goods_service.cg_calculation_details (
    detail_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    asset_id UUID NOT NULL,
    emissions_kg_co2e DECIMAL(15,3) NOT NULL CHECK (emissions_kg_co2e >= 0),
    emission_factor DECIMAL(15,6) NOT NULL,
    emission_factor_unit VARCHAR(50),
    emission_factor_source VARCHAR(100),
    calculation_method VARCHAR(50) NOT NULL,
    data_quality_score DECIMAL(3,2) CHECK (data_quality_score >= 1.0 AND data_quality_score <= 5.0),
    uncertainty_percentage DECIMAL(5,2) CHECK (uncertainty_percentage >= 0),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_detail_calc FOREIGN KEY (calc_id)
        REFERENCES capital_goods_service.cg_calculations(calc_id) ON DELETE CASCADE,
    CONSTRAINT fk_cg_detail_asset FOREIGN KEY (asset_id)
        REFERENCES capital_goods_service.cg_capital_assets(asset_id)
);

CREATE INDEX idx_cg_detail_calc ON capital_goods_service.cg_calculation_details(calc_id);
CREATE INDEX idx_cg_detail_asset ON capital_goods_service.cg_calculation_details(asset_id);
CREATE INDEX idx_cg_detail_method ON capital_goods_service.cg_calculation_details(calculation_method);

COMMENT ON TABLE capital_goods_service.cg_calculation_details IS 'Per-asset calculation details for transparency';

-- Spend-based calculation results
CREATE TABLE capital_goods_service.cg_spend_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    asset_id UUID NOT NULL,
    capex_amount_usd DECIMAL(15,2) NOT NULL,
    margin_adjusted_amount_usd DECIMAL(15,2),
    naics_code VARCHAR(10) NOT NULL,
    emission_factor_kg_co2e_per_usd DECIMAL(15,6) NOT NULL,
    emission_factor_source VARCHAR(100),
    emissions_kg_co2e DECIMAL(15,3) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_spend_calc FOREIGN KEY (calc_id)
        REFERENCES capital_goods_service.cg_calculations(calc_id) ON DELETE CASCADE,
    CONSTRAINT fk_cg_spend_asset FOREIGN KEY (asset_id)
        REFERENCES capital_goods_service.cg_capital_assets(asset_id)
);

CREATE INDEX idx_cg_spend_calc ON capital_goods_service.cg_spend_results(calc_id);
CREATE INDEX idx_cg_spend_asset ON capital_goods_service.cg_spend_results(asset_id);
CREATE INDEX idx_cg_spend_naics ON capital_goods_service.cg_spend_results(naics_code);

COMMENT ON TABLE capital_goods_service.cg_spend_results IS 'Spend-based (EEIO) calculation results';

-- Average-data calculation results
CREATE TABLE capital_goods_service.cg_average_data_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    asset_id UUID NOT NULL,
    material_type VARCHAR(100) NOT NULL,
    quantity DECIMAL(15,3) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    emission_factor_kg_co2e_per_unit DECIMAL(15,6) NOT NULL,
    emission_factor_source VARCHAR(100),
    emissions_kg_co2e DECIMAL(15,3) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_avg_calc FOREIGN KEY (calc_id)
        REFERENCES capital_goods_service.cg_calculations(calc_id) ON DELETE CASCADE,
    CONSTRAINT fk_cg_avg_asset FOREIGN KEY (asset_id)
        REFERENCES capital_goods_service.cg_capital_assets(asset_id)
);

CREATE INDEX idx_cg_avg_calc ON capital_goods_service.cg_average_data_results(calc_id);
CREATE INDEX idx_cg_avg_asset ON capital_goods_service.cg_average_data_results(asset_id);
CREATE INDEX idx_cg_avg_material ON capital_goods_service.cg_average_data_results(material_type);

COMMENT ON TABLE capital_goods_service.cg_average_data_results IS 'Average-data (physical) calculation results';

-- Supplier-specific results
CREATE TABLE capital_goods_service.cg_supplier_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    asset_id UUID NOT NULL,
    supplier_id VARCHAR(100) NOT NULL,
    quantity DECIMAL(15,3) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    emission_factor_kg_co2e_per_unit DECIMAL(15,6) NOT NULL,
    emission_factor_source VARCHAR(100),
    verification_status VARCHAR(50),
    emissions_kg_co2e DECIMAL(15,3) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_supp_calc FOREIGN KEY (calc_id)
        REFERENCES capital_goods_service.cg_calculations(calc_id) ON DELETE CASCADE,
    CONSTRAINT fk_cg_supp_asset FOREIGN KEY (asset_id)
        REFERENCES capital_goods_service.cg_capital_assets(asset_id)
);

CREATE INDEX idx_cg_supp_calc ON capital_goods_service.cg_supplier_results(calc_id);
CREATE INDEX idx_cg_supp_asset ON capital_goods_service.cg_supplier_results(asset_id);
CREATE INDEX idx_cg_supp_supplier ON capital_goods_service.cg_supplier_results(supplier_id);

COMMENT ON TABLE capital_goods_service.cg_supplier_results IS 'Supplier-specific calculation results';

-- Hybrid aggregation results
CREATE TABLE capital_goods_service.cg_hybrid_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    total_emissions_kg_co2e DECIMAL(15,3) NOT NULL,
    supplier_specific_emissions DECIMAL(15,3),
    supplier_specific_percentage DECIMAL(5,2),
    average_data_emissions DECIMAL(15,3),
    average_data_percentage DECIMAL(5,2),
    spend_based_emissions DECIMAL(15,3),
    spend_based_percentage DECIMAL(5,2),
    data_quality_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_hybrid_calc FOREIGN KEY (calc_id)
        REFERENCES capital_goods_service.cg_calculations(calc_id) ON DELETE CASCADE
);

CREATE INDEX idx_cg_hybrid_calc ON capital_goods_service.cg_hybrid_results(calc_id);

COMMENT ON TABLE capital_goods_service.cg_hybrid_results IS 'Hybrid calculation aggregation with method breakdown';

-- Compliance checks
CREATE TABLE capital_goods_service.cg_compliance_checks (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    framework VARCHAR(50) NOT NULL,
    check_type VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('PASS', 'FAIL', 'WARNING', 'INFO')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')),
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_compliance_calc FOREIGN KEY (calc_id)
        REFERENCES capital_goods_service.cg_calculations(calc_id) ON DELETE CASCADE
);

CREATE INDEX idx_cg_compliance_calc ON capital_goods_service.cg_compliance_checks(calc_id);
CREATE INDEX idx_cg_compliance_framework ON capital_goods_service.cg_compliance_checks(framework);
CREATE INDEX idx_cg_compliance_status ON capital_goods_service.cg_compliance_checks(status);

COMMENT ON TABLE capital_goods_service.cg_compliance_checks IS 'Compliance check results per framework';

-- Hot-spot analyses
CREATE TABLE capital_goods_service.cg_hot_spot_analyses (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    dimension VARCHAR(50) NOT NULL,
    dimension_value VARCHAR(200) NOT NULL,
    emissions_kg_co2e DECIMAL(15,3) NOT NULL,
    percentage_of_total DECIMAL(5,2) NOT NULL,
    asset_count INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cg_hotspot_calc FOREIGN KEY (calc_id)
        REFERENCES capital_goods_service.cg_calculations(calc_id) ON DELETE CASCADE
);

CREATE INDEX idx_cg_hotspot_calc ON capital_goods_service.cg_hot_spot_analyses(calc_id);
CREATE INDEX idx_cg_hotspot_dimension ON capital_goods_service.cg_hot_spot_analyses(dimension);
CREATE INDEX idx_cg_hotspot_rank ON capital_goods_service.cg_hot_spot_analyses(rank);

COMMENT ON TABLE capital_goods_service.cg_hot_spot_analyses IS 'Hot-spot analysis results by dimension';

-- Audit trail
CREATE TABLE capital_goods_service.cg_audit_entries (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    actor VARCHAR(100) NOT NULL,
    changes JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cg_audit_tenant ON capital_goods_service.cg_audit_entries(tenant_id);
CREATE INDEX idx_cg_audit_entity ON capital_goods_service.cg_audit_entries(entity_type, entity_id);
CREATE INDEX idx_cg_audit_timestamp ON capital_goods_service.cg_audit_entries(timestamp);

COMMENT ON TABLE capital_goods_service.cg_audit_entries IS 'Audit trail for all capital goods operations';

-- ============================================================================
-- HYPERTABLES (TimescaleDB)
-- ============================================================================

-- Calculation events hypertable
CREATE TABLE capital_goods_service.cg_calculation_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calc_id UUID NOT NULL,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    emissions_kg_co2e DECIMAL(15,3),
    asset_count INTEGER,
    calculation_method VARCHAR(50),
    metadata JSONB
);

SELECT create_hypertable('capital_goods_service.cg_calculation_events', 'event_time', chunk_time_interval => INTERVAL '7 days');

CREATE INDEX idx_cg_calc_events_tenant ON capital_goods_service.cg_calculation_events(tenant_id, event_time DESC);
CREATE INDEX idx_cg_calc_events_type ON capital_goods_service.cg_calculation_events(event_type, event_time DESC);

COMMENT ON TABLE capital_goods_service.cg_calculation_events IS 'TimescaleDB hypertable for calculation event time-series';

-- Asset events hypertable
CREATE TABLE capital_goods_service.cg_asset_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    asset_id UUID NOT NULL,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    event_details JSONB
);

SELECT create_hypertable('capital_goods_service.cg_asset_events', 'event_time', chunk_time_interval => INTERVAL '7 days');

CREATE INDEX idx_cg_asset_events_tenant ON capital_goods_service.cg_asset_events(tenant_id, event_time DESC);
CREATE INDEX idx_cg_asset_events_asset ON capital_goods_service.cg_asset_events(asset_id, event_time DESC);

COMMENT ON TABLE capital_goods_service.cg_asset_events IS 'TimescaleDB hypertable for asset lifecycle events';

-- Compliance events hypertable
CREATE TABLE capital_goods_service.cg_compliance_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calc_id UUID NOT NULL,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    framework VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    check_count INTEGER,
    pass_count INTEGER,
    fail_count INTEGER,
    metadata JSONB
);

SELECT create_hypertable('capital_goods_service.cg_compliance_events', 'event_time', chunk_time_interval => INTERVAL '7 days');

CREATE INDEX idx_cg_compliance_events_tenant ON capital_goods_service.cg_compliance_events(tenant_id, event_time DESC);
CREATE INDEX idx_cg_compliance_events_framework ON capital_goods_service.cg_compliance_events(framework, event_time DESC);

COMMENT ON TABLE capital_goods_service.cg_compliance_events IS 'TimescaleDB hypertable for compliance check events';

-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Hourly calculation statistics
CREATE MATERIALIZED VIEW capital_goods_service.cg_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    tenant_id,
    calculation_method,
    COUNT(*) AS calculation_count,
    SUM(emissions_kg_co2e) AS total_emissions_kg_co2e,
    AVG(emissions_kg_co2e) AS avg_emissions_kg_co2e,
    SUM(asset_count) AS total_assets
FROM capital_goods_service.cg_calculation_events
GROUP BY bucket, tenant_id, calculation_method
WITH NO DATA;

SELECT add_continuous_aggregate_policy('capital_goods_service.cg_hourly_calculation_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

COMMENT ON MATERIALIZED VIEW capital_goods_service.cg_hourly_calculation_stats IS 'Hourly aggregation of calculation metrics';

-- Daily emission totals by category
CREATE MATERIALIZED VIEW capital_goods_service.cg_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', ae.event_time) AS bucket,
    ae.tenant_id,
    ca.category_id,
    ca.category_name,
    COUNT(DISTINCT ae.asset_id) AS asset_count,
    SUM(cd.emissions_kg_co2e) AS total_emissions_kg_co2e,
    AVG(cd.data_quality_score) AS avg_data_quality_score
FROM capital_goods_service.cg_asset_events ae
JOIN capital_goods_service.cg_capital_assets ca ON ae.asset_id = ca.asset_id
LEFT JOIN capital_goods_service.cg_calculation_details cd ON ca.asset_id = cd.asset_id
GROUP BY bucket, ae.tenant_id, ca.category_id, ca.category_name
WITH NO DATA;

SELECT add_continuous_aggregate_policy('capital_goods_service.cg_daily_emission_totals',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

COMMENT ON MATERIALIZED VIEW capital_goods_service.cg_daily_emission_totals IS 'Daily emission totals by asset category';

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

ALTER TABLE capital_goods_service.cg_eeio_emission_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_physical_emission_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_supplier_emission_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_margin_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_capital_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_audit_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_calculation_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_asset_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE capital_goods_service.cg_compliance_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY cg_eeio_tenant_isolation ON capital_goods_service.cg_eeio_emission_factors
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_physical_tenant_isolation ON capital_goods_service.cg_physical_emission_factors
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_supplier_tenant_isolation ON capital_goods_service.cg_supplier_emission_factors
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_margin_tenant_isolation ON capital_goods_service.cg_margin_factors
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_assets_tenant_isolation ON capital_goods_service.cg_capital_assets
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_calc_tenant_isolation ON capital_goods_service.cg_calculations
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_audit_tenant_isolation ON capital_goods_service.cg_audit_entries
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_calc_events_tenant_isolation ON capital_goods_service.cg_calculation_events
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_asset_events_tenant_isolation ON capital_goods_service.cg_asset_events
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

CREATE POLICY cg_compliance_events_tenant_isolation ON capital_goods_service.cg_compliance_events
    USING (tenant_id::TEXT = current_setting('app.current_tenant', TRUE));

-- ============================================================================
-- SEED DATA
-- ============================================================================

-- 8 primary asset categories with subcategories
INSERT INTO capital_goods_service.cg_asset_categories (category_id, category_name, parent_category_id, description, typical_useful_life_years, capitalization_threshold_usd) VALUES
('CAT-BUILDINGS', 'Buildings and Structures', NULL, 'Office buildings, warehouses, production facilities', 30, 10000),
('CAT-BUILDINGS-OFFICE', 'Office Buildings', 'CAT-BUILDINGS', 'Administrative and office buildings', 30, 10000),
('CAT-BUILDINGS-PRODUCTION', 'Production Facilities', 'CAT-BUILDINGS', 'Manufacturing and production buildings', 30, 10000),
('CAT-MACHINERY', 'Machinery and Equipment', NULL, 'Production machinery, processing equipment', 10, 5000),
('CAT-MACHINERY-PRODUCTION', 'Production Machinery', 'CAT-MACHINERY', 'Manufacturing equipment', 10, 5000),
('CAT-MACHINERY-MATERIAL', 'Material Handling Equipment', 'CAT-MACHINERY', 'Forklifts, conveyors, cranes', 8, 2500),
('CAT-VEHICLES', 'Vehicles and Fleet', NULL, 'Company vehicles, trucks, fleet equipment', 5, 5000),
('CAT-VEHICLES-LIGHT', 'Light Duty Vehicles', 'CAT-VEHICLES', 'Cars, vans, light trucks', 5, 5000),
('CAT-VEHICLES-HEAVY', 'Heavy Duty Vehicles', 'CAT-VEHICLES', 'Trucks, trailers, heavy equipment', 7, 10000),
('CAT-IT', 'IT Equipment and Infrastructure', NULL, 'Servers, computers, network equipment', 3, 1000),
('CAT-IT-SERVERS', 'Servers and Data Centers', 'CAT-IT', 'Server hardware, data center equipment', 5, 5000),
('CAT-IT-END-USER', 'End User Devices', 'CAT-IT', 'Computers, laptops, tablets', 3, 500),
('CAT-FURNITURE', 'Furniture and Fixtures', NULL, 'Office furniture, fixtures', 7, 500),
('CAT-LAND-IMPROVEMENTS', 'Land Improvements', NULL, 'Parking lots, landscaping, fencing', 15, 5000),
('CAT-LEASEHOLD', 'Leasehold Improvements', NULL, 'Improvements to leased property', 10, 5000),
('CAT-OTHER', 'Other Capital Goods', NULL, 'Miscellaneous capital assets', 5, 1000);

-- 30+ EEIO factors for capital goods NAICS sectors (2023 US data)
INSERT INTO capital_goods_service.cg_eeio_emission_factors (tenant_id, naics_code, naics_description, sector_name, emission_factor_kg_co2e_per_usd, source, year, region, data_quality_score) VALUES
('00000000-0000-0000-0000-000000000000', '2361', 'Residential building construction', 'Construction', 0.285, 'USEEIO', 2023, 'US', 3.5),
('00000000-0000-0000-0000-000000000000', '2362', 'Nonresidential building construction', 'Construction', 0.312, 'USEEIO', 2023, 'US', 3.5),
('00000000-0000-0000-0000-000000000000', '333111', 'Farm machinery and equipment manufacturing', 'Machinery Manufacturing', 0.198, 'USEEIO', 2023, 'US', 3.8),
('00000000-0000-0000-0000-000000000000', '333120', 'Construction machinery manufacturing', 'Machinery Manufacturing', 0.215, 'USEEIO', 2023, 'US', 3.8),
('00000000-0000-0000-0000-000000000000', '333131', 'Mining machinery and equipment manufacturing', 'Machinery Manufacturing', 0.225, 'USEEIO', 2023, 'US', 3.7),
('00000000-0000-0000-0000-000000000000', '333241', 'Food product machinery manufacturing', 'Machinery Manufacturing', 0.192, 'USEEIO', 2023, 'US', 3.6),
('00000000-0000-0000-0000-000000000000', '333318', 'Other commercial and service industry machinery', 'Machinery Manufacturing', 0.188, 'USEEIO', 2023, 'US', 3.6),
('00000000-0000-0000-0000-000000000000', '333515', 'Cutting tool and machine tool accessory manufacturing', 'Machinery Manufacturing', 0.205, 'USEEIO', 2023, 'US', 3.5),
('00000000-0000-0000-0000-000000000000', '333611', 'Turbine and turbine generator set units manufacturing', 'Machinery Manufacturing', 0.235, 'USEEIO', 2023, 'US', 3.9),
('00000000-0000-0000-0000-000000000000', '334111', 'Electronic computer manufacturing', 'Computer Manufacturing', 0.165, 'USEEIO', 2023, 'US', 4.0),
('00000000-0000-0000-0000-000000000000', '334112', 'Computer storage device manufacturing', 'Computer Manufacturing', 0.172, 'USEEIO', 2023, 'US', 3.9),
('00000000-0000-0000-0000-000000000000', '334210', 'Telephone apparatus manufacturing', 'Communications Equipment', 0.158, 'USEEIO', 2023, 'US', 3.8),
('00000000-0000-0000-0000-000000000000', '334220', 'Radio and television broadcasting and wireless communications equipment', 'Communications Equipment', 0.162, 'USEEIO', 2023, 'US', 3.8),
('00000000-0000-0000-0000-000000000000', '334290', 'Other communications equipment manufacturing', 'Communications Equipment', 0.155, 'USEEIO', 2023, 'US', 3.7),
('00000000-0000-0000-0000-000000000000', '334413', 'Semiconductor and related device manufacturing', 'Electronics', 0.245, 'USEEIO', 2023, 'US', 4.1),
('00000000-0000-0000-0000-000000000000', '335311', 'Power, distribution, and specialty transformer manufacturing', 'Electrical Equipment', 0.218, 'USEEIO', 2023, 'US', 3.7),
('00000000-0000-0000-0000-000000000000', '335312', 'Motor and generator manufacturing', 'Electrical Equipment', 0.208, 'USEEIO', 2023, 'US', 3.7),
('00000000-0000-0000-0000-000000000000', '335313', 'Switchgear and switchboard apparatus manufacturing', 'Electrical Equipment', 0.195, 'USEEIO', 2023, 'US', 3.6),
('00000000-0000-0000-0000-000000000000', '336111', 'Automobile manufacturing', 'Motor Vehicles', 0.325, 'USEEIO', 2023, 'US', 4.2),
('00000000-0000-0000-0000-000000000000', '336112', 'Light truck and utility vehicle manufacturing', 'Motor Vehicles', 0.338, 'USEEIO', 2023, 'US', 4.2),
('00000000-0000-0000-0000-000000000000', '336120', 'Heavy duty truck manufacturing', 'Motor Vehicles', 0.352, 'USEEIO', 2023, 'US', 4.1),
('00000000-0000-0000-0000-000000000000', '337121', 'Upholstered household furniture manufacturing', 'Furniture', 0.168, 'USEEIO', 2023, 'US', 3.4),
('00000000-0000-0000-0000-000000000000', '337122', 'Nonupholstered wood household furniture manufacturing', 'Furniture', 0.175, 'USEEIO', 2023, 'US', 3.4),
('00000000-0000-0000-0000-000000000000', '337127', 'Institutional furniture manufacturing', 'Furniture', 0.182, 'USEEIO', 2023, 'US', 3.5),
('00000000-0000-0000-0000-000000000000', '337214', 'Office furniture (except wood) manufacturing', 'Furniture', 0.178, 'USEEIO', 2023, 'US', 3.5),
('00000000-0000-0000-0000-000000000000', '541310', 'Architectural services', 'Professional Services', 0.045, 'USEEIO', 2023, 'US', 3.2),
('00000000-0000-0000-0000-000000000000', '541330', 'Engineering services', 'Professional Services', 0.052, 'USEEIO', 2023, 'US', 3.2),
('00000000-0000-0000-0000-000000000000', '541511', 'Custom computer programming services', 'Software Services', 0.028, 'USEEIO', 2023, 'US', 3.0),
('00000000-0000-0000-0000-000000000000', '541512', 'Computer systems design services', 'Software Services', 0.032, 'USEEIO', 2023, 'US', 3.0),
('00000000-0000-0000-0000-000000000000', '541990', 'All other professional, scientific, and technical services', 'Professional Services', 0.038, 'USEEIO', 2023, 'US', 3.1);

-- 20+ physical emission factors for common materials
INSERT INTO capital_goods_service.cg_physical_emission_factors (tenant_id, material_type, material_subcategory, emission_factor_kg_co2e_per_unit, unit, source, year, region, lifecycle_stage, data_quality_score) VALUES
('00000000-0000-0000-0000-000000000000', 'Steel', 'Structural Steel', 1.85, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 4.2),
('00000000-0000-0000-0000-000000000000', 'Steel', 'Reinforcing Steel', 1.92, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 4.1),
('00000000-0000-0000-0000-000000000000', 'Concrete', 'Ready-Mix Concrete', 0.135, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 3.8),
('00000000-0000-0000-0000-000000000000', 'Aluminum', 'Primary Aluminum', 8.24, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 4.3),
('00000000-0000-0000-0000-000000000000', 'Aluminum', 'Recycled Aluminum', 0.52, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 4.0),
('00000000-0000-0000-0000-000000000000', 'Copper', 'Primary Copper', 2.95, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 4.1),
('00000000-0000-0000-0000-000000000000', 'Copper', 'Recycled Copper', 0.68, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 3.9),
('00000000-0000-0000-0000-000000000000', 'Glass', 'Float Glass', 0.85, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 3.7),
('00000000-0000-0000-0000-000000000000', 'Plastic', 'Polyethylene (PE)', 1.95, 'kg', 'Ecoinvent', 2023, 'Global', 'Cradle-to-Gate', 3.8),
('00000000-0000-0000-0000-000000000000', 'Plastic', 'Polypropylene (PP)', 2.02, 'kg', 'Ecoinvent', 2023, 'Global', 'Cradle-to-Gate', 3.8),
('00000000-0000-0000-0000-000000000000', 'Plastic', 'Polyvinyl Chloride (PVC)', 2.15, 'kg', 'Ecoinvent', 2023, 'Global', 'Cradle-to-Gate', 3.7),
('00000000-0000-0000-0000-000000000000', 'Wood', 'Softwood Lumber', 0.25, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 3.5),
('00000000-0000-0000-0000-000000000000', 'Wood', 'Hardwood Lumber', 0.28, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 3.5),
('00000000-0000-0000-0000-000000000000', 'Wood', 'Plywood', 0.58, 'kg', 'ICE Database', 2023, 'Global', 'Cradle-to-Gate', 3.4),
('00000000-0000-0000-0000-000000000000', 'Electronics', 'Desktop Computer', 350, 'unit', 'DEFRA', 2023, 'UK', 'Cradle-to-Gate', 3.9),
('00000000-0000-0000-0000-000000000000', 'Electronics', 'Laptop Computer', 220, 'unit', 'DEFRA', 2023, 'UK', 'Cradle-to-Gate', 3.9),
('00000000-0000-0000-0000-000000000000', 'Electronics', 'Server (Rack Unit)', 1250, 'unit', 'DEFRA', 2023, 'UK', 'Cradle-to-Gate', 4.0),
('00000000-0000-0000-0000-000000000000', 'Electronics', 'Network Switch (24-port)', 185, 'unit', 'DEFRA', 2023, 'UK', 'Cradle-to-Gate', 3.7),
('00000000-0000-0000-0000-000000000000', 'Machinery', 'Industrial Pump', 1850, 'unit', 'Ecoinvent', 2023, 'Global', 'Cradle-to-Gate', 3.5),
('00000000-0000-0000-0000-000000000000', 'Machinery', 'Electric Motor (10kW)', 425, 'unit', 'Ecoinvent', 2023, 'Global', 'Cradle-to-Gate', 3.6),
('00000000-0000-0000-0000-000000000000', 'Furniture', 'Office Desk', 85, 'unit', 'DEFRA', 2023, 'UK', 'Cradle-to-Gate', 3.3),
('00000000-0000-0000-0000-000000000000', 'Furniture', 'Office Chair', 45, 'unit', 'DEFRA', 2023, 'UK', 'Cradle-to-Gate', 3.3);

-- 20+ margin factors for capital sectors
INSERT INTO capital_goods_service.cg_margin_factors (tenant_id, sector_code, sector_name, margin_percentage, region, year, source) VALUES
('00000000-0000-0000-0000-000000000000', '2361', 'Residential building construction', 12.5, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '2362', 'Nonresidential building construction', 15.2, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '333', 'Machinery Manufacturing', 8.7, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '334', 'Computer and Electronic Product Manufacturing', 6.5, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '335', 'Electrical Equipment Manufacturing', 9.2, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '336', 'Transportation Equipment Manufacturing', 7.8, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '337', 'Furniture and Related Product Manufacturing', 11.3, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '423', 'Merchant Wholesalers, Durable Goods', 18.5, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '441', 'Motor Vehicle and Parts Dealers', 14.2, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '443', 'Electronics and Appliance Stores', 16.8, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '444', 'Building Material and Garden Equipment', 19.5, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '541310', 'Architectural services', 22.0, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '541330', 'Engineering services', 20.5, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '541511', 'Custom computer programming services', 25.0, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '541512', 'Computer systems design services', 23.5, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '238', 'Specialty Trade Contractors', 13.8, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '331', 'Primary Metal Manufacturing', 5.5, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '332', 'Fabricated Metal Product Manufacturing', 8.2, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '326', 'Plastics and Rubber Products Manufacturing', 7.5, 'US', 2023, 'BEA Use Tables'),
('00000000-0000-0000-0000-000000000000', '327', 'Nonmetallic Mineral Product Manufacturing', 10.5, 'US', 2023, 'BEA Use Tables');

-- 50+ classification mappings
INSERT INTO capital_goods_service.cg_classification_mappings (naics_code, nace_code, isic_code, unspsc_code, asset_category_id, description) VALUES
('2361', 'F41.20', '4100', '72101500', 'CAT-BUILDINGS', 'Residential building construction'),
('2362', 'F41.20', '4100', '72101600', 'CAT-BUILDINGS', 'Nonresidential building construction'),
('333111', 'C28.30', '2824', '21101500', 'CAT-MACHINERY', 'Farm machinery and equipment manufacturing'),
('333120', 'C28.92', '2824', '23151500', 'CAT-MACHINERY', 'Construction machinery manufacturing'),
('333131', 'C28.92', '2824', '23161500', 'CAT-MACHINERY', 'Mining machinery manufacturing'),
('333241', 'C28.93', '2825', '21101700', 'CAT-MACHINERY', 'Food product machinery manufacturing'),
('334111', 'C26.20', '2620', '43211500', 'CAT-IT', 'Electronic computer manufacturing'),
('334112', 'C26.20', '2620', '43211700', 'CAT-IT', 'Computer storage device manufacturing'),
('334210', 'C26.30', '2630', '43191500', 'CAT-IT', 'Telephone apparatus manufacturing'),
('334220', 'C26.30', '2630', '43191600', 'CAT-IT', 'Broadcasting and wireless communications equipment'),
('334413', 'C26.11', '2610', '31151500', 'CAT-IT', 'Semiconductor manufacturing'),
('335311', 'C27.11', '2710', '40151500', 'CAT-MACHINERY', 'Power transformer manufacturing'),
('335312', 'C27.11', '2710', '40151600', 'CAT-MACHINERY', 'Motor and generator manufacturing'),
('335313', 'C27.12', '2710', '40151700', 'CAT-MACHINERY', 'Switchgear manufacturing'),
('336111', 'C29.10', '2910', '25101500', 'CAT-VEHICLES', 'Automobile manufacturing'),
('336112', 'C29.10', '2910', '25101600', 'CAT-VEHICLES', 'Light truck manufacturing'),
('336120', 'C29.10', '2910', '25101700', 'CAT-VEHICLES', 'Heavy truck manufacturing'),
('337121', 'C31.01', '3100', '56101500', 'CAT-FURNITURE', 'Upholstered furniture manufacturing'),
('337122', 'C31.02', '3100', '56101600', 'CAT-FURNITURE', 'Wood furniture manufacturing'),
('337127', 'C31.01', '3100', '56101700', 'CAT-FURNITURE', 'Institutional furniture manufacturing'),
('337214', 'C31.01', '3100', '56101800', 'CAT-FURNITURE', 'Office furniture manufacturing'),
('333611', 'C28.11', '2811', '40101500', 'CAT-MACHINERY', 'Turbine manufacturing'),
('333515', 'C28.41', '2841', '23141500', 'CAT-MACHINERY', 'Cutting tool manufacturing'),
('333318', 'C28.99', '2826', '24101500', 'CAT-MACHINERY', 'Commercial machinery manufacturing'),
('541310', 'M71.11', '7110', '80111500', NULL, 'Architectural services'),
('541330', 'M71.12', '7110', '80111600', NULL, 'Engineering services'),
('541511', 'J62.01', '6201', '81111500', NULL, 'Computer programming services'),
('541512', 'J62.02', '6202', '81111600', NULL, 'Computer systems design services'),
('331110', 'C24.10', '2410', '11101500', 'CAT-MACHINERY', 'Iron and steel mills'),
('332310', 'C25.11', '2511', '30181500', 'CAT-MACHINERY', 'Plate work and fabricated structural product'),
('326199', 'C22.29', '2220', '30141500', 'CAT-MACHINERY', 'Plastics product manufacturing'),
('327310', 'C23.51', '2395', '30171500', 'CAT-BUILDINGS', 'Cement manufacturing'),
('327320', 'C23.61', '2396', '30171600', 'CAT-BUILDINGS', 'Ready-mix concrete manufacturing'),
('238', 'F43', '4300', '72151500', 'CAT-BUILDINGS', 'Specialty trade contractors'),
('333912', 'C28.22', '2822', '20111500', 'CAT-MACHINERY', 'Air and gas compressor manufacturing'),
('333991', 'C28.29', '2829', '21111500', 'CAT-MACHINERY', 'Power-driven handtool manufacturing'),
('333993', 'C28.93', '2825', '23151600', 'CAT-MACHINERY', 'Packaging machinery manufacturing'),
('333994', 'C28.94', '2826', '24111500', 'CAT-MACHINERY', 'Industrial process furnace and oven manufacturing'),
('333995', 'C28.95', '2824', '21111600', 'CAT-MACHINERY', 'Fluid power cylinder and actuator manufacturing'),
('334310', 'C26.51', '2651', '41111500', 'CAT-IT', 'Audio and video equipment manufacturing'),
('334511', 'C26.70', '2670', '41111600', 'CAT-IT', 'Search, detection, and navigation instruments'),
('334513', 'C26.51', '2651', '41111700', 'CAT-IT', 'Industrial process variable instruments'),
('334514', 'C26.51', '2651', '42111500', 'CAT-IT', 'Totalizing fluid meter and counting device'),
('334515', 'C26.60', '2660', '42111600', 'CAT-IT', 'Electricity and signal testing instruments'),
('335110', 'C27.11', '2710', '39111500', 'CAT-MACHINERY', 'Electric lamp bulb manufacturing'),
('335121', 'C27.40', '2740', '39111600', 'CAT-MACHINERY', 'Residential electric lighting fixture manufacturing'),
('335122', 'C27.40', '2740', '39111700', 'CAT-MACHINERY', 'Commercial electric lighting fixture manufacturing'),
('335210', 'C27.51', '2750', '52111500', 'CAT-MACHINERY', 'Small electrical appliance manufacturing'),
('335220', 'C27.51', '2750', '52111600', 'CAT-MACHINERY', 'Major household appliance manufacturing'),
('336211', 'C29.32', '2930', '25101800', 'CAT-VEHICLES', 'Motor vehicle body manufacturing'),
('336212', 'C29.20', '2920', '25101900', 'CAT-VEHICLES', 'Truck trailer manufacturing');

-- ============================================================================
-- AGENT REGISTRY ENTRY
-- ============================================================================

INSERT INTO agent_registry.agents (
    id, name, version, category, subcategory, status, description,
    input_schema, output_schema, configuration_schema,
    capabilities, dependencies, tags, health_check_endpoint,
    created_at, updated_at
) VALUES (
    'GL-MRV-SCOPE3-002',
    'Capital Goods Agent',
    '1.0.0',
    'MRV',
    'Scope 3',
    'active',
    'Capital Goods Agent (Scope 3 Category 2) - Calculates emissions from capital goods purchased using EEIO spend-based, average-data physical, supplier-specific, and hybrid methods per GHG Protocol',
    '{"type": "object", "properties": {"assets": {"type": "array"}, "reporting_period": {"type": "object"}, "calculation_method": {"type": "string"}}}',
    '{"type": "object", "properties": {"total_emissions_kg_co2e": {"type": "number"}, "calculation_method": {"type": "string"}, "provenance_hash": {"type": "string"}}}',
    '{"type": "object", "properties": {"gwp_source": {"type": "string"}, "eeio_database": {"type": "string"}, "margin_adjustment": {"type": "boolean"}}}',
    ARRAY['capital_goods_calculation', 'eeio_spend_based', 'average_data_method', 'supplier_specific_method', 'hybrid_calculation', 'hot_spot_analysis', 'compliance_checking'],
    ARRAY['GL-FOUND-003', 'GL-FOUND-004', 'GL-FOUND-005', 'GL-DATA-003'],
    ARRAY['scope3', 'category2', 'capital-goods', 'ghg-protocol', 'eeio', 'spend-based', 'hybrid'],
    '/health/capital-goods',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- ============================================================================
-- COMPLETION
-- ============================================================================

COMMENT ON SCHEMA capital_goods_service IS 'Capital Goods Agent (AGENT-MRV-015) - Complete schema with 16 tables, 3 hypertables, 2 continuous aggregates, RLS policies, and seed data';

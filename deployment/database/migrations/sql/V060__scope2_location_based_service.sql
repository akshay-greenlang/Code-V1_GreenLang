-- ==========================================================================
-- V060: Scope 2 Location-Based Emissions Service Schema
-- AGENT-MRV-009 (GL-MRV-SCOPE2-009)
--
-- Tables: 12 (s2l_ prefix)
-- Hypertables: 3
-- Continuous Aggregates: 2
-- Indexes: 30+
--
-- GHG Protocol Scope 2 Guidance (2015) — Location-based method
-- Supports eGRID (US 26 subregions), IEA (130+ countries), EU EEA, DEFRA
--
-- Author: GreenLang Platform Team
-- Date: February 2026
-- ==========================================================================

-- ==========================================================================
-- Table 1: s2l_facilities — Facility registration with grid region
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_facilities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    facility_id VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    facility_type VARCHAR(50) NOT NULL DEFAULT 'office',
    country_code VARCHAR(3) NOT NULL,
    grid_region_id VARCHAR(100),
    egrid_subregion VARCHAR(10),
    latitude DECIMAL(10, 7),
    longitude DECIMAL(10, 7),
    address TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, facility_id)
);

COMMENT ON TABLE s2l_facilities IS 'Scope 2 location-based facility registration with grid region assignment';
COMMENT ON COLUMN s2l_facilities.facility_type IS 'Type: office, warehouse, manufacturing, retail, data_center, hospital, school, other';
COMMENT ON COLUMN s2l_facilities.egrid_subregion IS 'US EPA eGRID subregion (e.g., CAMX, ERCT) — US facilities only';

-- ==========================================================================
-- Table 2: s2l_grid_regions — Grid region definitions
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_grid_regions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    region_type VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    country_code VARCHAR(3),
    subregion_code VARCHAR(20),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE s2l_grid_regions IS 'Grid region definitions for emission factor assignment';
COMMENT ON COLUMN s2l_grid_regions.region_type IS 'Type: country, subregion, state, custom';
COMMENT ON COLUMN s2l_grid_regions.source IS 'Source: egrid, iea, eu_eea, defra, national, custom';

-- ==========================================================================
-- Table 3: s2l_grid_emission_factors — Emission factor database
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_grid_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(100) NOT NULL REFERENCES s2l_grid_regions(region_id),
    source VARCHAR(50) NOT NULL,
    year INTEGER NOT NULL,
    co2_kg_per_mwh DECIMAL(12, 6) NOT NULL,
    ch4_kg_per_mwh DECIMAL(12, 6) DEFAULT 0,
    n2o_kg_per_mwh DECIMAL(12, 6) DEFAULT 0,
    total_co2e_kg_per_mwh DECIMAL(12, 6),
    data_quality_tier VARCHAR(20) DEFAULT 'tier_2',
    uncertainty_pct DECIMAL(8, 4),
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(region_id, source, year)
);

COMMENT ON TABLE s2l_grid_emission_factors IS 'Grid emission factors from authoritative sources (eGRID, IEA, EU EEA, DEFRA, IPCC)';
COMMENT ON COLUMN s2l_grid_emission_factors.co2_kg_per_mwh IS 'CO2 emission factor in kg CO2 per MWh';
COMMENT ON COLUMN s2l_grid_emission_factors.data_quality_tier IS 'IPCC data quality: tier_1, tier_2, tier_3';

-- ==========================================================================
-- Table 4: s2l_energy_consumption — Energy consumption records
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_energy_consumption (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    facility_id VARCHAR(100) NOT NULL,
    energy_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(16, 6) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    data_source VARCHAR(30) DEFAULT 'invoice',
    meter_id VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE s2l_energy_consumption IS 'Energy consumption records for electricity, steam, heating, cooling';
COMMENT ON COLUMN s2l_energy_consumption.energy_type IS 'Type: electricity, steam, heating, cooling';
COMMENT ON COLUMN s2l_energy_consumption.unit IS 'Unit: kwh, mwh, gj, mmbtu, therms';
COMMENT ON COLUMN s2l_energy_consumption.data_source IS 'Source: meter, invoice, estimate, benchmark';

-- ==========================================================================
-- Table 5: s2l_td_loss_factors — T&D loss factors by region
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_td_loss_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code VARCHAR(3) NOT NULL,
    td_loss_pct DECIMAL(8, 4) NOT NULL,
    source VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL,
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(country_code, source, year)
);

COMMENT ON TABLE s2l_td_loss_factors IS 'Transmission & distribution loss factors by country/region';
COMMENT ON COLUMN s2l_td_loss_factors.td_loss_pct IS 'T&D loss as decimal fraction (e.g., 0.05 = 5%)';

-- ==========================================================================
-- Table 6: s2l_calculations — Calculation results
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calculation_id VARCHAR(100) NOT NULL,
    facility_id VARCHAR(100) NOT NULL,
    energy_type VARCHAR(20) NOT NULL,
    consumption_value DECIMAL(16, 6) NOT NULL,
    consumption_unit VARCHAR(20) NOT NULL,
    grid_region VARCHAR(100),
    emission_factor_source VARCHAR(50),
    ef_co2e_per_mwh DECIMAL(12, 6),
    td_loss_pct DECIMAL(8, 4) DEFAULT 0,
    total_co2e_kg DECIMAL(16, 6) NOT NULL,
    total_co2e_tonnes DECIMAL(16, 6) NOT NULL,
    gwp_source VARCHAR(20) DEFAULT 'AR5',
    calculation_method VARCHAR(50) DEFAULT 'ipcc_tier_1',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, calculation_id)
);

COMMENT ON TABLE s2l_calculations IS 'Scope 2 location-based calculation results with provenance';
COMMENT ON COLUMN s2l_calculations.gwp_source IS 'GWP source: AR4, AR5, AR6, AR6_20YR';
COMMENT ON COLUMN s2l_calculations.provenance_hash IS 'SHA-256 hash of complete calculation provenance chain';

-- ==========================================================================
-- Table 7: s2l_calculation_details — Per-gas/per-energy details
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_calculation_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id VARCHAR(100) NOT NULL,
    gas VARCHAR(10) NOT NULL,
    emission_kg DECIMAL(16, 6) NOT NULL,
    gwp_factor DECIMAL(10, 4) NOT NULL,
    co2e_kg DECIMAL(16, 6) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE s2l_calculation_details IS 'Per-gas emission breakdown (CO2, CH4, N2O) for each calculation';
COMMENT ON COLUMN s2l_calculation_details.gas IS 'Greenhouse gas: co2, ch4, n2o';

-- ==========================================================================
-- Table 8: s2l_meter_readings — Meter reading data
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_meter_readings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    facility_id VARCHAR(100) NOT NULL,
    meter_id VARCHAR(100) NOT NULL,
    reading_date DATE NOT NULL,
    reading_value DECIMAL(16, 6) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    energy_type VARCHAR(20) NOT NULL,
    is_estimated BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE s2l_meter_readings IS 'Energy meter reading data for consumption tracking';

-- ==========================================================================
-- Table 9: s2l_custom_factors — User-supplied custom factors
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_custom_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    region_id VARCHAR(100) NOT NULL,
    name VARCHAR(255),
    co2_kg_per_mwh DECIMAL(12, 6) NOT NULL,
    ch4_kg_per_mwh DECIMAL(12, 6) DEFAULT 0,
    n2o_kg_per_mwh DECIMAL(12, 6) DEFAULT 0,
    year INTEGER NOT NULL,
    data_quality_tier VARCHAR(20) DEFAULT 'tier_3',
    justification TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, region_id, year)
);

COMMENT ON TABLE s2l_custom_factors IS 'User-supplied custom emission factors with quality tracking';

-- ==========================================================================
-- Table 10: s2l_compliance_records — Compliance check results
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_compliance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calculation_id VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    score DECIMAL(5, 2),
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE s2l_compliance_records IS 'Multi-framework regulatory compliance check results';
COMMENT ON COLUMN s2l_compliance_records.framework IS 'Framework: ghg_protocol_scope2, ipcc_2006, iso_14064, csrd_esrs, epa_ghgrp, defra, cdp';
COMMENT ON COLUMN s2l_compliance_records.status IS 'Status: compliant, non_compliant, partial, not_assessed';

-- ==========================================================================
-- Table 11: s2l_audit_entries — Provenance/audit trail
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_audit_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calculation_id VARCHAR(100),
    action VARCHAR(50) NOT NULL,
    stage VARCHAR(50),
    hash_value VARCHAR(64) NOT NULL,
    previous_hash VARCHAR(64),
    details JSONB DEFAULT '{}',
    performed_by VARCHAR(100),
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE s2l_audit_entries IS 'SHA-256 provenance chain audit trail for calculation reproducibility';

-- ==========================================================================
-- Table 12: s2l_factor_updates — Factor version tracking
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_factor_updates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(100) NOT NULL,
    source VARCHAR(50) NOT NULL,
    old_co2e_per_mwh DECIMAL(12, 6),
    new_co2e_per_mwh DECIMAL(12, 6) NOT NULL,
    old_year INTEGER,
    new_year INTEGER NOT NULL,
    change_reason TEXT,
    updated_by VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE s2l_factor_updates IS 'Emission factor version tracking for audit and rollback';

-- ==========================================================================
-- Hypertable 1: s2l_calculation_events — Calculation time-series
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_calculation_events (
    time TIMESTAMPTZ NOT NULL,
    tenant_id UUID NOT NULL,
    calculation_id VARCHAR(100) NOT NULL,
    facility_id VARCHAR(100) NOT NULL,
    energy_type VARCHAR(20) NOT NULL,
    total_co2e_kg DECIMAL(16, 6) NOT NULL,
    total_co2e_tonnes DECIMAL(16, 6) NOT NULL,
    grid_region VARCHAR(100),
    gwp_source VARCHAR(20),
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('s2l_calculation_events', 'time', if_not_exists => TRUE);

COMMENT ON TABLE s2l_calculation_events IS 'Time-series of Scope 2 location-based calculation events';

-- ==========================================================================
-- Hypertable 2: s2l_consumption_events_ts — Consumption time-series
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_consumption_events_ts (
    time TIMESTAMPTZ NOT NULL,
    tenant_id UUID NOT NULL,
    facility_id VARCHAR(100) NOT NULL,
    energy_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(16, 6) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    data_source VARCHAR(30),
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('s2l_consumption_events_ts', 'time', if_not_exists => TRUE);

COMMENT ON TABLE s2l_consumption_events_ts IS 'Time-series of energy consumption events';

-- ==========================================================================
-- Hypertable 3: s2l_compliance_events — Compliance time-series
-- ==========================================================================
CREATE TABLE IF NOT EXISTS s2l_compliance_events (
    time TIMESTAMPTZ NOT NULL,
    tenant_id UUID NOT NULL,
    calculation_id VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    score DECIMAL(5, 2),
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('s2l_compliance_events', 'time', if_not_exists => TRUE);

COMMENT ON TABLE s2l_compliance_events IS 'Time-series of compliance check events';

-- ==========================================================================
-- Continuous Aggregate 1: Hourly calculation statistics
-- ==========================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS s2l_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    tenant_id,
    energy_type,
    COUNT(*) AS calculation_count,
    SUM(total_co2e_tonnes) AS total_co2e_tonnes,
    AVG(total_co2e_tonnes) AS avg_co2e_tonnes,
    MIN(total_co2e_tonnes) AS min_co2e_tonnes,
    MAX(total_co2e_tonnes) AS max_co2e_tonnes
FROM s2l_calculation_events
GROUP BY bucket, tenant_id, energy_type
WITH NO DATA;

-- ==========================================================================
-- Continuous Aggregate 2: Daily emission totals
-- ==========================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS s2l_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    tenant_id,
    facility_id,
    energy_type,
    COUNT(*) AS calculation_count,
    SUM(total_co2e_tonnes) AS total_co2e_tonnes,
    MIN(total_co2e_tonnes) AS min_co2e_tonnes,
    MAX(total_co2e_tonnes) AS max_co2e_tonnes
FROM s2l_calculation_events
GROUP BY bucket, tenant_id, facility_id, energy_type
WITH NO DATA;

-- ==========================================================================
-- Indexes: Core tables
-- ==========================================================================

-- s2l_facilities
CREATE INDEX IF NOT EXISTS idx_s2l_facilities_tenant ON s2l_facilities(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2l_facilities_country ON s2l_facilities(country_code);
CREATE INDEX IF NOT EXISTS idx_s2l_facilities_region ON s2l_facilities(grid_region_id);
CREATE INDEX IF NOT EXISTS idx_s2l_facilities_egrid ON s2l_facilities(egrid_subregion) WHERE egrid_subregion IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_s2l_facilities_active ON s2l_facilities(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_s2l_facilities_type ON s2l_facilities(facility_type);

-- s2l_grid_regions
CREATE INDEX IF NOT EXISTS idx_s2l_grid_regions_type ON s2l_grid_regions(region_type);
CREATE INDEX IF NOT EXISTS idx_s2l_grid_regions_source ON s2l_grid_regions(source);
CREATE INDEX IF NOT EXISTS idx_s2l_grid_regions_country ON s2l_grid_regions(country_code);

-- s2l_grid_emission_factors
CREATE INDEX IF NOT EXISTS idx_s2l_gef_region ON s2l_grid_emission_factors(region_id);
CREATE INDEX IF NOT EXISTS idx_s2l_gef_source ON s2l_grid_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_s2l_gef_year ON s2l_grid_emission_factors(year);
CREATE INDEX IF NOT EXISTS idx_s2l_gef_region_year ON s2l_grid_emission_factors(region_id, year);

-- s2l_energy_consumption
CREATE INDEX IF NOT EXISTS idx_s2l_consumption_tenant ON s2l_energy_consumption(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2l_consumption_facility ON s2l_energy_consumption(facility_id);
CREATE INDEX IF NOT EXISTS idx_s2l_consumption_type ON s2l_energy_consumption(energy_type);
CREATE INDEX IF NOT EXISTS idx_s2l_consumption_period ON s2l_energy_consumption(period_start, period_end);

-- s2l_td_loss_factors
CREATE INDEX IF NOT EXISTS idx_s2l_td_country ON s2l_td_loss_factors(country_code);

-- s2l_calculations
CREATE INDEX IF NOT EXISTS idx_s2l_calc_tenant ON s2l_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2l_calc_facility ON s2l_calculations(facility_id);
CREATE INDEX IF NOT EXISTS idx_s2l_calc_type ON s2l_calculations(energy_type);
CREATE INDEX IF NOT EXISTS idx_s2l_calc_date ON s2l_calculations(calculated_at);
CREATE INDEX IF NOT EXISTS idx_s2l_calc_provenance ON s2l_calculations(provenance_hash);

-- s2l_calculation_details
CREATE INDEX IF NOT EXISTS idx_s2l_details_calc ON s2l_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_s2l_details_gas ON s2l_calculation_details(gas);

-- s2l_meter_readings
CREATE INDEX IF NOT EXISTS idx_s2l_meter_tenant ON s2l_meter_readings(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2l_meter_facility ON s2l_meter_readings(facility_id);
CREATE INDEX IF NOT EXISTS idx_s2l_meter_date ON s2l_meter_readings(reading_date);

-- s2l_custom_factors
CREATE INDEX IF NOT EXISTS idx_s2l_custom_tenant ON s2l_custom_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2l_custom_region ON s2l_custom_factors(region_id);
CREATE INDEX IF NOT EXISTS idx_s2l_custom_active ON s2l_custom_factors(is_active) WHERE is_active = TRUE;

-- s2l_compliance_records
CREATE INDEX IF NOT EXISTS idx_s2l_compliance_tenant ON s2l_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2l_compliance_calc ON s2l_compliance_records(calculation_id);
CREATE INDEX IF NOT EXISTS idx_s2l_compliance_framework ON s2l_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_s2l_compliance_status ON s2l_compliance_records(status);

-- s2l_audit_entries
CREATE INDEX IF NOT EXISTS idx_s2l_audit_tenant ON s2l_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2l_audit_calc ON s2l_audit_entries(calculation_id);
CREATE INDEX IF NOT EXISTS idx_s2l_audit_action ON s2l_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_s2l_audit_date ON s2l_audit_entries(performed_at);

-- GIN indexes on JSONB metadata columns
CREATE INDEX IF NOT EXISTS idx_s2l_facilities_meta ON s2l_facilities USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_s2l_calculations_meta ON s2l_calculations USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_s2l_consumption_meta ON s2l_energy_consumption USING GIN(metadata);

-- ==========================================================================
-- Hypertable indexes
-- ==========================================================================
CREATE INDEX IF NOT EXISTS idx_s2l_calc_events_tenant ON s2l_calculation_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_s2l_calc_events_facility ON s2l_calculation_events(facility_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_s2l_calc_events_type ON s2l_calculation_events(energy_type, time DESC);

CREATE INDEX IF NOT EXISTS idx_s2l_cons_events_tenant ON s2l_consumption_events_ts(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_s2l_cons_events_facility ON s2l_consumption_events_ts(facility_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_s2l_comp_events_tenant ON s2l_compliance_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_s2l_comp_events_framework ON s2l_compliance_events(framework, time DESC);

-- ==========================================================================
-- Continuous Aggregate Refresh Policies
-- ==========================================================================
SELECT add_continuous_aggregate_policy('s2l_hourly_calculation_stats',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('s2l_daily_emission_totals',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ==========================================================================
-- Retention Policies (5 years)
-- ==========================================================================
SELECT add_retention_policy('s2l_calculation_events', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('s2l_consumption_events_ts', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('s2l_compliance_events', INTERVAL '5 years', if_not_exists => TRUE);

-- ==========================================================================
-- Seed Data: Grid Regions (eGRID subregions + major countries)
-- ==========================================================================

-- EPA eGRID 2022 subregions (26)
INSERT INTO s2l_grid_regions (region_id, name, region_type, source, country_code, subregion_code, description)
VALUES
    ('EGRID-AKGD', 'ASCC Alaska Grid', 'subregion', 'egrid', 'US', 'AKGD', 'Alaska Systems Coordinating Council - Alaska Grid'),
    ('EGRID-AKMS', 'ASCC Miscellaneous', 'subregion', 'egrid', 'US', 'AKMS', 'Alaska Systems Coordinating Council - Miscellaneous'),
    ('EGRID-AZNM', 'WECC Southwest', 'subregion', 'egrid', 'US', 'AZNM', 'Western Electricity Coordinating Council - Southwest'),
    ('EGRID-CAMX', 'WECC California', 'subregion', 'egrid', 'US', 'CAMX', 'Western Electricity Coordinating Council - California'),
    ('EGRID-ERCT', 'ERCOT All', 'subregion', 'egrid', 'US', 'ERCT', 'Electric Reliability Council of Texas'),
    ('EGRID-FRCC', 'FRCC All', 'subregion', 'egrid', 'US', 'FRCC', 'Florida Reliability Coordinating Council'),
    ('EGRID-HIMS', 'HICC Miscellaneous', 'subregion', 'egrid', 'US', 'HIMS', 'Hawaiian Islands Coordinating Council - Miscellaneous'),
    ('EGRID-HIOA', 'HICC Oahu', 'subregion', 'egrid', 'US', 'HIOA', 'Hawaiian Islands Coordinating Council - Oahu'),
    ('EGRID-MROE', 'MRO East', 'subregion', 'egrid', 'US', 'MROE', 'Midwest Reliability Organization - East'),
    ('EGRID-MROW', 'MRO West', 'subregion', 'egrid', 'US', 'MROW', 'Midwest Reliability Organization - West'),
    ('EGRID-NEWE', 'NPCC New England', 'subregion', 'egrid', 'US', 'NEWE', 'Northeast Power Coordinating Council - New England'),
    ('EGRID-NWPP', 'WECC Northwest', 'subregion', 'egrid', 'US', 'NWPP', 'Western Electricity Coordinating Council - Northwest'),
    ('EGRID-NYCW', 'NPCC NYC/Westchester', 'subregion', 'egrid', 'US', 'NYCW', 'Northeast Power Coordinating Council - NYC/Westchester'),
    ('EGRID-NYLI', 'NPCC Long Island', 'subregion', 'egrid', 'US', 'NYLI', 'Northeast Power Coordinating Council - Long Island'),
    ('EGRID-NYUP', 'NPCC Upstate NY', 'subregion', 'egrid', 'US', 'NYUP', 'Northeast Power Coordinating Council - Upstate New York'),
    ('EGRID-PRMS', 'PREPA Miscellaneous', 'subregion', 'egrid', 'US', 'PRMS', 'Puerto Rico Electric Power Authority'),
    ('EGRID-RFCE', 'RFC East', 'subregion', 'egrid', 'US', 'RFCE', 'ReliabilityFirst Corporation - East'),
    ('EGRID-RFCM', 'RFC Michigan', 'subregion', 'egrid', 'US', 'RFCM', 'ReliabilityFirst Corporation - Michigan'),
    ('EGRID-RFCW', 'RFC West', 'subregion', 'egrid', 'US', 'RFCW', 'ReliabilityFirst Corporation - West'),
    ('EGRID-RMPA', 'WECC Rockies', 'subregion', 'egrid', 'US', 'RMPA', 'Western Electricity Coordinating Council - Rockies'),
    ('EGRID-SPNO', 'SPP North', 'subregion', 'egrid', 'US', 'SPNO', 'Southwest Power Pool - North'),
    ('EGRID-SPSO', 'SPP South', 'subregion', 'egrid', 'US', 'SPSO', 'Southwest Power Pool - South'),
    ('EGRID-SRMV', 'SERC Mississippi Valley', 'subregion', 'egrid', 'US', 'SRMV', 'SERC Reliability Corporation - Mississippi Valley'),
    ('EGRID-SRMW', 'SERC Midwest', 'subregion', 'egrid', 'US', 'SRMW', 'SERC Reliability Corporation - Midwest'),
    ('EGRID-SRSO', 'SERC South', 'subregion', 'egrid', 'US', 'SRSO', 'SERC Reliability Corporation - South'),
    ('EGRID-SRTV', 'SERC Tennessee Valley', 'subregion', 'egrid', 'US', 'SRTV', 'SERC Reliability Corporation - Tennessee Valley'),
    ('EGRID-SRVC', 'SERC Virginia/Carolina', 'subregion', 'egrid', 'US', 'SRVC', 'SERC Reliability Corporation - Virginia/Carolina')
ON CONFLICT (region_id) DO NOTHING;

-- Major IEA countries (30 largest economies)
INSERT INTO s2l_grid_regions (region_id, name, region_type, source, country_code, description)
VALUES
    ('IEA-US', 'United States', 'country', 'iea', 'US', 'IEA grid factor for United States'),
    ('IEA-CN', 'China', 'country', 'iea', 'CN', 'IEA grid factor for China'),
    ('IEA-IN', 'India', 'country', 'iea', 'IN', 'IEA grid factor for India'),
    ('IEA-JP', 'Japan', 'country', 'iea', 'JP', 'IEA grid factor for Japan'),
    ('IEA-DE', 'Germany', 'country', 'iea', 'DE', 'IEA grid factor for Germany'),
    ('IEA-GB', 'United Kingdom', 'country', 'iea', 'GB', 'IEA/DEFRA grid factor for United Kingdom'),
    ('IEA-FR', 'France', 'country', 'iea', 'FR', 'IEA grid factor for France'),
    ('IEA-IT', 'Italy', 'country', 'iea', 'IT', 'IEA grid factor for Italy'),
    ('IEA-CA', 'Canada', 'country', 'iea', 'CA', 'IEA grid factor for Canada'),
    ('IEA-KR', 'South Korea', 'country', 'iea', 'KR', 'IEA grid factor for South Korea'),
    ('IEA-AU', 'Australia', 'country', 'iea', 'AU', 'IEA grid factor for Australia'),
    ('IEA-BR', 'Brazil', 'country', 'iea', 'BR', 'IEA grid factor for Brazil'),
    ('IEA-MX', 'Mexico', 'country', 'iea', 'MX', 'IEA grid factor for Mexico'),
    ('IEA-ES', 'Spain', 'country', 'iea', 'ES', 'IEA grid factor for Spain'),
    ('IEA-NL', 'Netherlands', 'country', 'iea', 'NL', 'IEA grid factor for Netherlands'),
    ('IEA-SE', 'Sweden', 'country', 'iea', 'SE', 'IEA grid factor for Sweden'),
    ('IEA-NO', 'Norway', 'country', 'iea', 'NO', 'IEA grid factor for Norway'),
    ('IEA-PL', 'Poland', 'country', 'iea', 'PL', 'IEA grid factor for Poland'),
    ('IEA-ZA', 'South Africa', 'country', 'iea', 'ZA', 'IEA grid factor for South Africa'),
    ('IEA-SG', 'Singapore', 'country', 'iea', 'SG', 'IEA grid factor for Singapore')
ON CONFLICT (region_id) DO NOTHING;

-- EU 27 member states
INSERT INTO s2l_grid_regions (region_id, name, region_type, source, country_code, description)
VALUES
    ('EU-AT', 'Austria', 'country', 'eu_eea', 'AT', 'EU EEA grid factor for Austria'),
    ('EU-BE', 'Belgium', 'country', 'eu_eea', 'BE', 'EU EEA grid factor for Belgium'),
    ('EU-BG', 'Bulgaria', 'country', 'eu_eea', 'BG', 'EU EEA grid factor for Bulgaria'),
    ('EU-HR', 'Croatia', 'country', 'eu_eea', 'HR', 'EU EEA grid factor for Croatia'),
    ('EU-CY', 'Cyprus', 'country', 'eu_eea', 'CY', 'EU EEA grid factor for Cyprus'),
    ('EU-CZ', 'Czech Republic', 'country', 'eu_eea', 'CZ', 'EU EEA grid factor for Czech Republic'),
    ('EU-DK', 'Denmark', 'country', 'eu_eea', 'DK', 'EU EEA grid factor for Denmark'),
    ('EU-EE', 'Estonia', 'country', 'eu_eea', 'EE', 'EU EEA grid factor for Estonia'),
    ('EU-FI', 'Finland', 'country', 'eu_eea', 'FI', 'EU EEA grid factor for Finland'),
    ('EU-FR', 'France', 'country', 'eu_eea', 'FR', 'EU EEA grid factor for France'),
    ('EU-DE', 'Germany', 'country', 'eu_eea', 'DE', 'EU EEA grid factor for Germany'),
    ('EU-GR', 'Greece', 'country', 'eu_eea', 'GR', 'EU EEA grid factor for Greece'),
    ('EU-HU', 'Hungary', 'country', 'eu_eea', 'HU', 'EU EEA grid factor for Hungary'),
    ('EU-IE', 'Ireland', 'country', 'eu_eea', 'IE', 'EU EEA grid factor for Ireland'),
    ('EU-IT', 'Italy', 'country', 'eu_eea', 'IT', 'EU EEA grid factor for Italy'),
    ('EU-LV', 'Latvia', 'country', 'eu_eea', 'LV', 'EU EEA grid factor for Latvia'),
    ('EU-LT', 'Lithuania', 'country', 'eu_eea', 'LT', 'EU EEA grid factor for Lithuania'),
    ('EU-LU', 'Luxembourg', 'country', 'eu_eea', 'LU', 'EU EEA grid factor for Luxembourg'),
    ('EU-MT', 'Malta', 'country', 'eu_eea', 'MT', 'EU EEA grid factor for Malta'),
    ('EU-NL', 'Netherlands', 'country', 'eu_eea', 'NL', 'EU EEA grid factor for Netherlands'),
    ('EU-PL', 'Poland', 'country', 'eu_eea', 'PL', 'EU EEA grid factor for Poland'),
    ('EU-PT', 'Portugal', 'country', 'eu_eea', 'PT', 'EU EEA grid factor for Portugal'),
    ('EU-RO', 'Romania', 'country', 'eu_eea', 'RO', 'EU EEA grid factor for Romania'),
    ('EU-SK', 'Slovakia', 'country', 'eu_eea', 'SK', 'EU EEA grid factor for Slovakia'),
    ('EU-SI', 'Slovenia', 'country', 'eu_eea', 'SI', 'EU EEA grid factor for Slovenia'),
    ('EU-ES', 'Spain', 'country', 'eu_eea', 'ES', 'EU EEA grid factor for Spain'),
    ('EU-SE', 'Sweden', 'country', 'eu_eea', 'SE', 'EU EEA grid factor for Sweden')
ON CONFLICT (region_id) DO NOTHING;

-- ==========================================================================
-- Migration complete
-- ==========================================================================

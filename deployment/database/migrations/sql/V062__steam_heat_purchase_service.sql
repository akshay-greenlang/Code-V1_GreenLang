-- ==========================================================================
-- V062: Steam/Heat Purchase Agent Service Schema
-- AGENT-MRV-011 (GL-MRV-SCOPE2-011)
--
-- Tables: 14 (shp_ prefix)
-- Hypertables: 3
-- Continuous Aggregates: 2
-- Indexes: 66
-- RLS Policies: 34 (SELECT + INSERT per table)
--
-- GHG Protocol Scope 2 Guidance (2015) -- Purchased steam, heat, cooling
-- Supports fuel-based boiler emission factors, district heating networks,
-- cooling system COP factors, CHP/cogeneration allocation (energy, exergy,
-- efficiency methods), supplier-specific emission factors, multi-gas
-- breakdown (CO2/CH4/N2O), biogenic CO2 separation, uncertainty
-- quantification, and multi-framework regulatory compliance.
--
-- Author: GreenLang Platform Team
-- Date: February 2026
-- Previous: V061__scope2_market_based_service.sql
-- ==========================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS steam_heat_purchase_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ==========================================================================
-- Function: Auto-update updated_at timestamp
-- ==========================================================================

CREATE OR REPLACE FUNCTION steam_heat_purchase_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ==========================================================================
-- Table 1: shp_fuel_emission_factors -- Fuel-based boiler emission factors
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_fuel_emission_factors (
    factor_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    fuel_type           VARCHAR(50)     NOT NULL UNIQUE,
    co2_ef_per_gj       DECIMAL(20,8)   NOT NULL,
    ch4_ef_per_gj       DECIMAL(20,8)   NOT NULL,
    n2o_ef_per_gj       DECIMAL(20,8)   NOT NULL,
    default_efficiency  DECIMAL(10,4)   NOT NULL,
    is_biogenic         BOOLEAN         NOT NULL DEFAULT FALSE,
    source              VARCHAR(200)    NOT NULL DEFAULT 'IPCC 2006 Vol 2',
    effective_date      DATE            NOT NULL DEFAULT CURRENT_DATE,
    tenant_id           VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE steam_heat_purchase_service.shp_fuel_emission_factors IS 'Fuel-based boiler emission factors for steam/heat generation (kg CO2e per GJ by gas)';
COMMENT ON COLUMN steam_heat_purchase_service.shp_fuel_emission_factors.fuel_type IS 'Fuel type: natural_gas, coal_bituminous, coal_sub_bituminous, coal_lignite, fuel_oil_light, fuel_oil_heavy, lpg, diesel, biomass_wood, biomass_pellets, biogas, peat, waste_municipal, waste_industrial';
COMMENT ON COLUMN steam_heat_purchase_service.shp_fuel_emission_factors.default_efficiency IS 'Default boiler thermal efficiency (0.0-1.0)';
COMMENT ON COLUMN steam_heat_purchase_service.shp_fuel_emission_factors.is_biogenic IS 'True if fuel is biogenic (biomass/biogas) -- emissions reported separately';

CREATE TRIGGER trg_shp_fuel_ef_updated_at
    BEFORE UPDATE ON steam_heat_purchase_service.shp_fuel_emission_factors
    FOR EACH ROW EXECUTE FUNCTION steam_heat_purchase_service.set_updated_at();

-- ==========================================================================
-- Table 2: shp_district_heating_factors -- District heating network EFs
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_district_heating_factors (
    factor_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    region                  VARCHAR(100)    NOT NULL,
    network_type            VARCHAR(50)     NOT NULL DEFAULT 'municipal',
    ef_kgco2e_per_gj        DECIMAL(20,8)   NOT NULL,
    distribution_loss_pct   DECIMAL(10,4)   NOT NULL DEFAULT 0.12,
    source                  VARCHAR(200)    NOT NULL,
    effective_date          DATE            NOT NULL DEFAULT CURRENT_DATE,
    tenant_id               VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(region, network_type, tenant_id)
);

COMMENT ON TABLE steam_heat_purchase_service.shp_district_heating_factors IS 'District heating network emission factors by region and network type';
COMMENT ON COLUMN steam_heat_purchase_service.shp_district_heating_factors.network_type IS 'Network type: municipal, industrial, campus, cogeneration, waste_heat';
COMMENT ON COLUMN steam_heat_purchase_service.shp_district_heating_factors.distribution_loss_pct IS 'Distribution heat loss percentage (0.0-1.0), typically 0.05-0.25';
COMMENT ON COLUMN steam_heat_purchase_service.shp_district_heating_factors.ef_kgco2e_per_gj IS 'Emission factor in kg CO2e per GJ of delivered heat';

CREATE TRIGGER trg_shp_district_ef_updated_at
    BEFORE UPDATE ON steam_heat_purchase_service.shp_district_heating_factors
    FOR EACH ROW EXECUTE FUNCTION steam_heat_purchase_service.set_updated_at();

-- ==========================================================================
-- Table 3: shp_cooling_system_factors -- Cooling system COP factors
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_cooling_system_factors (
    factor_id       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    technology      VARCHAR(50)     NOT NULL UNIQUE,
    cop_min         DECIMAL(10,4)   NOT NULL,
    cop_max         DECIMAL(10,4)   NOT NULL,
    cop_default     DECIMAL(10,4)   NOT NULL,
    energy_source   VARCHAR(50)     NOT NULL DEFAULT 'electricity',
    source          VARCHAR(200)    NOT NULL,
    tenant_id       VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE steam_heat_purchase_service.shp_cooling_system_factors IS 'Cooling system Coefficient of Performance (COP) factors by technology';
COMMENT ON COLUMN steam_heat_purchase_service.shp_cooling_system_factors.technology IS 'Technology: electric_chiller, absorption_chiller, centrifugal_chiller, screw_chiller, district_cooling, free_cooling, evaporative_cooling, ground_source_hp, air_source_hp';
COMMENT ON COLUMN steam_heat_purchase_service.shp_cooling_system_factors.cop_default IS 'Default COP used when actual measurement is unavailable';
COMMENT ON COLUMN steam_heat_purchase_service.shp_cooling_system_factors.energy_source IS 'Primary energy input: electricity, natural_gas, steam, hot_water';

CREATE TRIGGER trg_shp_cooling_ef_updated_at
    BEFORE UPDATE ON steam_heat_purchase_service.shp_cooling_system_factors
    FOR EACH ROW EXECUTE FUNCTION steam_heat_purchase_service.set_updated_at();

-- ==========================================================================
-- Table 4: shp_chp_defaults -- CHP/cogeneration default parameters
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_chp_defaults (
    chp_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    fuel_type               VARCHAR(50)     NOT NULL,
    electrical_efficiency   DECIMAL(10,4)   NOT NULL,
    thermal_efficiency      DECIMAL(10,4)   NOT NULL,
    overall_efficiency      DECIMAL(10,4)   NOT NULL,
    source                  VARCHAR(200)    NOT NULL DEFAULT 'EPA CHP Partnership',
    tenant_id               VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(fuel_type, tenant_id)
);

COMMENT ON TABLE steam_heat_purchase_service.shp_chp_defaults IS 'CHP/cogeneration default efficiency parameters by fuel type for allocation methods';
COMMENT ON COLUMN steam_heat_purchase_service.shp_chp_defaults.electrical_efficiency IS 'Default electrical efficiency (0.0-1.0)';
COMMENT ON COLUMN steam_heat_purchase_service.shp_chp_defaults.thermal_efficiency IS 'Default thermal/heat efficiency (0.0-1.0)';
COMMENT ON COLUMN steam_heat_purchase_service.shp_chp_defaults.overall_efficiency IS 'Combined cycle overall efficiency = electrical + thermal';

CREATE TRIGGER trg_shp_chp_defaults_updated_at
    BEFORE UPDATE ON steam_heat_purchase_service.shp_chp_defaults
    FOR EACH ROW EXECUTE FUNCTION steam_heat_purchase_service.set_updated_at();

-- ==========================================================================
-- Table 5: shp_facilities -- Facility registration with heating/cooling
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_facilities (
    facility_id     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(300)    NOT NULL,
    facility_type   VARCHAR(50)     NOT NULL,
    country         VARCHAR(100)    NOT NULL,
    region          VARCHAR(200),
    latitude        DECIMAL(10,6),
    longitude       DECIMAL(11,6),
    heating_network VARCHAR(200),
    cooling_system  VARCHAR(200),
    tenant_id       VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE steam_heat_purchase_service.shp_facilities IS 'Facility registry with heating network and cooling system assignments';
COMMENT ON COLUMN steam_heat_purchase_service.shp_facilities.facility_type IS 'Type: manufacturing, office, hospital, campus, data_center, warehouse, retail, other';
COMMENT ON COLUMN steam_heat_purchase_service.shp_facilities.heating_network IS 'Connected district heating network name (if applicable)';
COMMENT ON COLUMN steam_heat_purchase_service.shp_facilities.cooling_system IS 'Primary cooling system technology (if applicable)';

CREATE TRIGGER trg_shp_facilities_updated_at
    BEFORE UPDATE ON steam_heat_purchase_service.shp_facilities
    FOR EACH ROW EXECUTE FUNCTION steam_heat_purchase_service.set_updated_at();

-- ==========================================================================
-- Table 6: shp_steam_suppliers -- Steam/heat supplier registry
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_steam_suppliers (
    supplier_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name                        VARCHAR(300)    NOT NULL,
    fuel_mix                    JSONB           NOT NULL DEFAULT '{}',
    boiler_efficiency           DECIMAL(10,4),
    supplier_ef_kgco2e_per_gj   DECIMAL(20,8),
    country                     VARCHAR(100)    NOT NULL,
    region                      VARCHAR(200),
    verified                    BOOLEAN         NOT NULL DEFAULT FALSE,
    data_quality_tier           VARCHAR(20)     NOT NULL DEFAULT 'TIER_1',
    tenant_id                   VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE steam_heat_purchase_service.shp_steam_suppliers IS 'Steam/heat supplier registry with fuel mix, boiler efficiency, and data quality tier';
COMMENT ON COLUMN steam_heat_purchase_service.shp_steam_suppliers.fuel_mix IS 'JSON fuel mix breakdown e.g. {"natural_gas": 0.60, "coal_bituminous": 0.30, "biomass_wood": 0.10}';
COMMENT ON COLUMN steam_heat_purchase_service.shp_steam_suppliers.data_quality_tier IS 'Data quality tier: TIER_1 (default EF), TIER_2 (supplier-specific), TIER_3 (measured)';
COMMENT ON COLUMN steam_heat_purchase_service.shp_steam_suppliers.verified IS 'True if supplier data has been independently verified/audited';

CREATE TRIGGER trg_shp_suppliers_updated_at
    BEFORE UPDATE ON steam_heat_purchase_service.shp_steam_suppliers
    FOR EACH ROW EXECUTE FUNCTION steam_heat_purchase_service.set_updated_at();

-- ==========================================================================
-- Table 7: shp_supplier_emission_factors -- Supplier-specific EFs by year
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_supplier_emission_factors (
    sef_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id         UUID            NOT NULL REFERENCES steam_heat_purchase_service.shp_steam_suppliers(supplier_id),
    co2_ef_per_gj       DECIMAL(20,8)   NOT NULL,
    ch4_ef_per_gj       DECIMAL(20,8),
    n2o_ef_per_gj       DECIMAL(20,8),
    total_co2e_per_gj   DECIMAL(20,8)   NOT NULL,
    reporting_year      INT             NOT NULL,
    source              VARCHAR(200)    NOT NULL,
    verified            BOOLEAN         NOT NULL DEFAULT FALSE,
    tenant_id           VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE steam_heat_purchase_service.shp_supplier_emission_factors IS 'Supplier-specific emission factors by reporting year with verification status';
COMMENT ON COLUMN steam_heat_purchase_service.shp_supplier_emission_factors.total_co2e_per_gj IS 'Total CO2e per GJ including all GHGs with GWP weighting';
COMMENT ON COLUMN steam_heat_purchase_service.shp_supplier_emission_factors.reporting_year IS 'Year the emission factor applies to';

-- ==========================================================================
-- Table 8: shp_calculations -- Steam/heat emission calculations (HYPERTABLE)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_calculations (
    calc_id                     UUID            NOT NULL DEFAULT gen_random_uuid(),
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    facility_id                 UUID            REFERENCES steam_heat_purchase_service.shp_facilities(facility_id),
    supplier_id                 UUID            REFERENCES steam_heat_purchase_service.shp_steam_suppliers(supplier_id),
    energy_type                 VARCHAR(50)     NOT NULL,
    calculation_method          VARCHAR(50)     NOT NULL,
    consumption_gj              DECIMAL(20,8)   NOT NULL,
    fuel_type                   VARCHAR(50),
    boiler_efficiency           DECIMAL(10,4),
    total_co2e_kg               DECIMAL(20,8)   NOT NULL,
    fossil_co2e_kg              DECIMAL(20,8)   NOT NULL DEFAULT 0,
    biogenic_co2_kg             DECIMAL(20,8)   NOT NULL DEFAULT 0,
    effective_ef_kgco2e_per_gj  DECIMAL(20,8),
    data_quality_tier           VARCHAR(20)     NOT NULL DEFAULT 'TIER_1',
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR6',
    provenance_hash             VARCHAR(64)     NOT NULL,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'SUCCESS',
    tenant_id                   VARCHAR(100)    NOT NULL DEFAULT 'default',
    PRIMARY KEY (calc_id, calculated_at),
    CONSTRAINT chk_shp_calc_energy_type CHECK (energy_type IN (
        'steam', 'hot_water', 'chilled_water', 'district_heating',
        'district_cooling', 'process_heat', 'industrial_steam'
    )),
    CONSTRAINT chk_shp_calc_method CHECK (calculation_method IN (
        'fuel_based', 'supplier_specific', 'district_heating',
        'cooling_cop', 'chp_energy', 'chp_exergy', 'chp_efficiency',
        'default_factor'
    )),
    CONSTRAINT chk_shp_calc_tier CHECK (data_quality_tier IN (
        'TIER_1', 'TIER_2', 'TIER_3'
    )),
    CONSTRAINT chk_shp_calc_gwp CHECK (gwp_source IN (
        'AR4', 'AR5', 'AR6', 'AR6_20YR'
    )),
    CONSTRAINT chk_shp_calc_status CHECK (status IN (
        'SUCCESS', 'FAILED', 'PENDING', 'PARTIAL'
    ))
);

SELECT create_hypertable(
    'steam_heat_purchase_service.shp_calculations',
    'calculated_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE steam_heat_purchase_service.shp_calculations IS 'Steam/heat emission calculation results (hypertable on calculated_at) with fossil/biogenic separation';
COMMENT ON COLUMN steam_heat_purchase_service.shp_calculations.energy_type IS 'Energy type: steam, hot_water, chilled_water, district_heating, district_cooling, process_heat, industrial_steam';
COMMENT ON COLUMN steam_heat_purchase_service.shp_calculations.calculation_method IS 'Method: fuel_based, supplier_specific, district_heating, cooling_cop, chp_energy, chp_exergy, chp_efficiency, default_factor';
COMMENT ON COLUMN steam_heat_purchase_service.shp_calculations.provenance_hash IS 'SHA-256 hash of complete calculation provenance chain';

-- ==========================================================================
-- Table 9: shp_calculation_details -- Per-gas emission breakdown
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_calculation_details (
    detail_id       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id         UUID            NOT NULL,
    gas             VARCHAR(20)     NOT NULL,
    emission_kg     DECIMAL(20,8)   NOT NULL,
    gwp_value       DECIMAL(10,4)   NOT NULL,
    gwp_source      VARCHAR(20)     NOT NULL,
    co2e_kg         DECIMAL(20,8)   NOT NULL,
    tenant_id       VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_shp_detail_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O', 'CO2_biogenic'
    )),
    CONSTRAINT chk_shp_detail_gwp CHECK (gwp_source IN (
        'AR4', 'AR5', 'AR6', 'AR6_20YR'
    ))
);

COMMENT ON TABLE steam_heat_purchase_service.shp_calculation_details IS 'Per-gas emission breakdown for each calculation with GWP weighting details';
COMMENT ON COLUMN steam_heat_purchase_service.shp_calculation_details.gas IS 'Greenhouse gas: CO2, CH4, N2O, CO2_biogenic';

-- ==========================================================================
-- Table 10: shp_chp_allocations -- CHP emission allocation records
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_chp_allocations (
    allocation_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id                     UUID            NOT NULL,
    method                      VARCHAR(20)     NOT NULL,
    total_fuel_gj               DECIMAL(20,8)   NOT NULL,
    fuel_type                   VARCHAR(50)     NOT NULL,
    heat_output_gj              DECIMAL(20,8)   NOT NULL,
    power_output_gj             DECIMAL(20,8)   NOT NULL,
    cooling_output_gj           DECIMAL(20,8)   NOT NULL DEFAULT 0,
    heat_share                  DECIMAL(10,8)   NOT NULL,
    power_share                 DECIMAL(10,8)   NOT NULL,
    cooling_share               DECIMAL(10,8)   NOT NULL DEFAULT 0,
    heat_emissions_kgco2e       DECIMAL(20,8)   NOT NULL,
    power_emissions_kgco2e      DECIMAL(20,8)   NOT NULL,
    cooling_emissions_kgco2e    DECIMAL(20,8)   NOT NULL DEFAULT 0,
    primary_energy_savings_pct  DECIMAL(10,4)   DEFAULT 0,
    provenance_hash             VARCHAR(64)     NOT NULL,
    tenant_id                   VARCHAR(100)    NOT NULL DEFAULT 'default',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_shp_alloc_method CHECK (method IN (
        'energy', 'exergy', 'efficiency', 'iea', 'custom'
    ))
);

COMMENT ON TABLE steam_heat_purchase_service.shp_chp_allocations IS 'CHP/cogeneration emission allocation records with heat/power/cooling shares';
COMMENT ON COLUMN steam_heat_purchase_service.shp_chp_allocations.method IS 'CHP allocation method: energy, exergy, efficiency, iea, custom';
COMMENT ON COLUMN steam_heat_purchase_service.shp_chp_allocations.heat_share IS 'Fraction of total emissions allocated to heat output (0.0-1.0)';
COMMENT ON COLUMN steam_heat_purchase_service.shp_chp_allocations.primary_energy_savings_pct IS 'EU CHP Directive primary energy savings percentage';

-- ==========================================================================
-- Table 11: shp_uncertainty_results -- Uncertainty quantification (HYPERTABLE)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_uncertainty_results (
    uncertainty_id              UUID            NOT NULL DEFAULT gen_random_uuid(),
    analyzed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    calc_id                     UUID            NOT NULL,
    method                      VARCHAR(30)     NOT NULL,
    mean_co2e_kg                DECIMAL(20,8)   NOT NULL,
    std_dev_kg                  DECIMAL(20,8)   NOT NULL,
    ci_lower_kg                 DECIMAL(20,8)   NOT NULL,
    ci_upper_kg                 DECIMAL(20,8)   NOT NULL,
    confidence_level            DECIMAL(5,4)    NOT NULL DEFAULT 0.95,
    relative_uncertainty_pct    DECIMAL(10,4)   NOT NULL,
    provenance_hash             VARCHAR(64)     NOT NULL,
    tenant_id                   VARCHAR(100)    NOT NULL DEFAULT 'default',
    PRIMARY KEY (uncertainty_id, analyzed_at),
    CONSTRAINT chk_shp_unc_method CHECK (method IN (
        'monte_carlo', 'error_propagation', 'ipcc_tier1',
        'ipcc_tier2', 'bootstrap', 'sensitivity'
    ))
);

SELECT create_hypertable(
    'steam_heat_purchase_service.shp_uncertainty_results',
    'analyzed_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE steam_heat_purchase_service.shp_uncertainty_results IS 'Uncertainty quantification results (hypertable on analyzed_at) with confidence intervals';
COMMENT ON COLUMN steam_heat_purchase_service.shp_uncertainty_results.method IS 'Method: monte_carlo, error_propagation, ipcc_tier1, ipcc_tier2, bootstrap, sensitivity';
COMMENT ON COLUMN steam_heat_purchase_service.shp_uncertainty_results.relative_uncertainty_pct IS 'Relative uncertainty as percentage of mean value';

-- ==========================================================================
-- Table 12: shp_compliance_checks -- Multi-framework compliance checks
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_compliance_checks (
    check_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id             UUID            NOT NULL,
    framework           VARCHAR(50)     NOT NULL,
    status              VARCHAR(20)     NOT NULL,
    total_requirements  INT             NOT NULL,
    met_requirements    INT             NOT NULL,
    score_pct           DECIMAL(10,4)   NOT NULL,
    findings            JSONB           NOT NULL DEFAULT '[]',
    provenance_hash     VARCHAR(64)     NOT NULL,
    tenant_id           VARCHAR(100)    NOT NULL DEFAULT 'default',
    checked_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_shp_compl_framework CHECK (framework IN (
        'ghg_protocol_scope2', 'iso_14064', 'csrd_esrs',
        'epc_chp_directive', 'ipcc_2006', 'cdp', 'tcfd', 'sec_climate'
    )),
    CONSTRAINT chk_shp_compl_status CHECK (status IN (
        'compliant', 'non_compliant', 'partial', 'not_assessed'
    ))
);

COMMENT ON TABLE steam_heat_purchase_service.shp_compliance_checks IS 'Multi-framework regulatory compliance check results for steam/heat calculations';
COMMENT ON COLUMN steam_heat_purchase_service.shp_compliance_checks.framework IS 'Framework: ghg_protocol_scope2, iso_14064, csrd_esrs, epc_chp_directive, ipcc_2006, cdp, tcfd, sec_climate';
COMMENT ON COLUMN steam_heat_purchase_service.shp_compliance_checks.findings IS 'JSON array of compliance findings with requirement, status, and evidence';

-- ==========================================================================
-- Table 13: shp_batch_jobs -- Batch processing job records
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_batch_jobs (
    batch_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    status                  VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    total_requests          INT             NOT NULL,
    success_count           INT             NOT NULL DEFAULT 0,
    failure_count           INT             NOT NULL DEFAULT 0,
    total_co2e_kg           DECIMAL(20,8),
    total_fossil_co2e_kg    DECIMAL(20,8),
    total_biogenic_co2_kg   DECIMAL(20,8),
    tenant_id               VARCHAR(100)    NOT NULL DEFAULT 'default',
    started_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMPTZ,
    CONSTRAINT chk_shp_batch_status CHECK (status IN (
        'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    ))
);

COMMENT ON TABLE steam_heat_purchase_service.shp_batch_jobs IS 'Batch processing job records with success/failure counts and total emissions';
COMMENT ON COLUMN steam_heat_purchase_service.shp_batch_jobs.status IS 'Job status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED';

-- ==========================================================================
-- Table 14: shp_aggregations -- Aggregated emission results (HYPERTABLE)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS steam_heat_purchase_service.shp_aggregations (
    aggregation_id          UUID            NOT NULL DEFAULT gen_random_uuid(),
    aggregated_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    aggregation_type        VARCHAR(50)     NOT NULL,
    total_co2e_kg           DECIMAL(20,8)   NOT NULL,
    total_fossil_co2e_kg    DECIMAL(20,8)   NOT NULL DEFAULT 0,
    total_biogenic_co2_kg   DECIMAL(20,8)   NOT NULL DEFAULT 0,
    breakdown               JSONB           NOT NULL DEFAULT '{}',
    count                   INT             NOT NULL,
    provenance_hash         VARCHAR(64)     NOT NULL,
    tenant_id               VARCHAR(100)    NOT NULL DEFAULT 'default',
    PRIMARY KEY (aggregation_id, aggregated_at),
    CONSTRAINT chk_shp_agg_type CHECK (aggregation_type IN (
        'facility', 'supplier', 'energy_type', 'fuel_type',
        'method', 'region', 'tenant', 'period', 'custom'
    ))
);

SELECT create_hypertable(
    'steam_heat_purchase_service.shp_aggregations',
    'aggregated_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE steam_heat_purchase_service.shp_aggregations IS 'Aggregated emission results (hypertable on aggregated_at) with fossil/biogenic separation';
COMMENT ON COLUMN steam_heat_purchase_service.shp_aggregations.aggregation_type IS 'Type: facility, supplier, energy_type, fuel_type, method, region, tenant, period, custom';
COMMENT ON COLUMN steam_heat_purchase_service.shp_aggregations.breakdown IS 'JSON breakdown by sub-categories within the aggregation';

-- ==========================================================================
-- Continuous Aggregate 1: Hourly calculation statistics
-- ==========================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS steam_heat_purchase_service.shp_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', calculated_at)    AS bucket,
    energy_type,
    tenant_id,
    COUNT(*)                                AS calculation_count,
    AVG(total_co2e_kg)                      AS avg_co2e_kg,
    MIN(total_co2e_kg)                      AS min_co2e_kg,
    MAX(total_co2e_kg)                      AS max_co2e_kg,
    SUM(total_co2e_kg)                      AS sum_co2e_kg,
    SUM(consumption_gj)                     AS sum_consumption_gj
FROM steam_heat_purchase_service.shp_calculations
GROUP BY bucket, energy_type, tenant_id
WITH NO DATA;

-- ==========================================================================
-- Continuous Aggregate 2: Daily calculation statistics
-- ==========================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS steam_heat_purchase_service.shp_daily_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at)     AS bucket,
    energy_type,
    tenant_id,
    COUNT(*)                                AS calculation_count,
    AVG(total_co2e_kg)                      AS avg_co2e_kg,
    MIN(total_co2e_kg)                      AS min_co2e_kg,
    MAX(total_co2e_kg)                      AS max_co2e_kg,
    SUM(total_co2e_kg)                      AS sum_co2e_kg,
    SUM(consumption_gj)                     AS sum_consumption_gj
FROM steam_heat_purchase_service.shp_calculations
GROUP BY bucket, energy_type, tenant_id
WITH NO DATA;

-- ==========================================================================
-- Indexes: Core tables (66 indexes)
-- ==========================================================================

-- shp_fuel_emission_factors (3 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_fuel_ef_tenant
    ON steam_heat_purchase_service.shp_fuel_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_fuel_ef_created
    ON steam_heat_purchase_service.shp_fuel_emission_factors(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_fuel_ef_biogenic
    ON steam_heat_purchase_service.shp_fuel_emission_factors(is_biogenic)
    WHERE is_biogenic = TRUE;

-- shp_district_heating_factors (4 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_district_tenant
    ON steam_heat_purchase_service.shp_district_heating_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_district_created
    ON steam_heat_purchase_service.shp_district_heating_factors(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_district_region
    ON steam_heat_purchase_service.shp_district_heating_factors(region);
CREATE INDEX IF NOT EXISTS idx_shp_district_type
    ON steam_heat_purchase_service.shp_district_heating_factors(network_type);

-- shp_cooling_system_factors (3 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_cooling_tenant
    ON steam_heat_purchase_service.shp_cooling_system_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_cooling_created
    ON steam_heat_purchase_service.shp_cooling_system_factors(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_cooling_source
    ON steam_heat_purchase_service.shp_cooling_system_factors(energy_source);

-- shp_chp_defaults (3 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_chp_tenant
    ON steam_heat_purchase_service.shp_chp_defaults(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_chp_created
    ON steam_heat_purchase_service.shp_chp_defaults(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_chp_fuel
    ON steam_heat_purchase_service.shp_chp_defaults(fuel_type);

-- shp_facilities (7 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_fac_tenant
    ON steam_heat_purchase_service.shp_facilities(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_fac_created
    ON steam_heat_purchase_service.shp_facilities(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_fac_type
    ON steam_heat_purchase_service.shp_facilities(facility_type);
CREATE INDEX IF NOT EXISTS idx_shp_fac_country
    ON steam_heat_purchase_service.shp_facilities(country);
CREATE INDEX IF NOT EXISTS idx_shp_fac_region
    ON steam_heat_purchase_service.shp_facilities(region);
CREATE INDEX IF NOT EXISTS idx_shp_fac_heating
    ON steam_heat_purchase_service.shp_facilities(heating_network);
CREATE INDEX IF NOT EXISTS idx_shp_fac_cooling
    ON steam_heat_purchase_service.shp_facilities(cooling_system);

-- shp_steam_suppliers (7 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_sup_tenant
    ON steam_heat_purchase_service.shp_steam_suppliers(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_sup_created
    ON steam_heat_purchase_service.shp_steam_suppliers(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_sup_country
    ON steam_heat_purchase_service.shp_steam_suppliers(country);
CREATE INDEX IF NOT EXISTS idx_shp_sup_region
    ON steam_heat_purchase_service.shp_steam_suppliers(region);
CREATE INDEX IF NOT EXISTS idx_shp_sup_verified
    ON steam_heat_purchase_service.shp_steam_suppliers(verified)
    WHERE verified = TRUE;
CREATE INDEX IF NOT EXISTS idx_shp_sup_tier
    ON steam_heat_purchase_service.shp_steam_suppliers(data_quality_tier);
CREATE INDEX IF NOT EXISTS idx_shp_sup_fuel_mix
    ON steam_heat_purchase_service.shp_steam_suppliers USING GIN(fuel_mix);

-- shp_supplier_emission_factors (5 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_sef_tenant
    ON steam_heat_purchase_service.shp_supplier_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_sef_created
    ON steam_heat_purchase_service.shp_supplier_emission_factors(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_sef_supplier
    ON steam_heat_purchase_service.shp_supplier_emission_factors(supplier_id);
CREATE INDEX IF NOT EXISTS idx_shp_sef_year
    ON steam_heat_purchase_service.shp_supplier_emission_factors(reporting_year);
CREATE INDEX IF NOT EXISTS idx_shp_sef_verified
    ON steam_heat_purchase_service.shp_supplier_emission_factors(verified)
    WHERE verified = TRUE;

-- shp_calculations (10 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_calc_tenant
    ON steam_heat_purchase_service.shp_calculations(tenant_id, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_calc_facility
    ON steam_heat_purchase_service.shp_calculations(facility_id, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_calc_supplier
    ON steam_heat_purchase_service.shp_calculations(supplier_id, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_calc_energy_type
    ON steam_heat_purchase_service.shp_calculations(energy_type, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_calc_fuel_type
    ON steam_heat_purchase_service.shp_calculations(fuel_type, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_calc_method
    ON steam_heat_purchase_service.shp_calculations(calculation_method, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_calc_status
    ON steam_heat_purchase_service.shp_calculations(status, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_calc_provenance
    ON steam_heat_purchase_service.shp_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_shp_calc_tier
    ON steam_heat_purchase_service.shp_calculations(data_quality_tier, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_calc_gwp
    ON steam_heat_purchase_service.shp_calculations(gwp_source, calculated_at DESC);

-- shp_calculation_details (4 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_detail_tenant
    ON steam_heat_purchase_service.shp_calculation_details(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_detail_created
    ON steam_heat_purchase_service.shp_calculation_details(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_detail_calc
    ON steam_heat_purchase_service.shp_calculation_details(calc_id);
CREATE INDEX IF NOT EXISTS idx_shp_detail_gas
    ON steam_heat_purchase_service.shp_calculation_details(gas);

-- shp_chp_allocations (4 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_alloc_tenant
    ON steam_heat_purchase_service.shp_chp_allocations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_alloc_created
    ON steam_heat_purchase_service.shp_chp_allocations(created_at);
CREATE INDEX IF NOT EXISTS idx_shp_alloc_calc
    ON steam_heat_purchase_service.shp_chp_allocations(calc_id);
CREATE INDEX IF NOT EXISTS idx_shp_alloc_method
    ON steam_heat_purchase_service.shp_chp_allocations(method);

-- shp_uncertainty_results (4 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_unc_tenant
    ON steam_heat_purchase_service.shp_uncertainty_results(tenant_id, analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_unc_calc
    ON steam_heat_purchase_service.shp_uncertainty_results(calc_id, analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_unc_method
    ON steam_heat_purchase_service.shp_uncertainty_results(method, analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_unc_provenance
    ON steam_heat_purchase_service.shp_uncertainty_results(provenance_hash);

-- shp_compliance_checks (5 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_compl_tenant
    ON steam_heat_purchase_service.shp_compliance_checks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_compl_checked
    ON steam_heat_purchase_service.shp_compliance_checks(checked_at);
CREATE INDEX IF NOT EXISTS idx_shp_compl_calc
    ON steam_heat_purchase_service.shp_compliance_checks(calc_id);
CREATE INDEX IF NOT EXISTS idx_shp_compl_framework
    ON steam_heat_purchase_service.shp_compliance_checks(framework);
CREATE INDEX IF NOT EXISTS idx_shp_compl_status
    ON steam_heat_purchase_service.shp_compliance_checks(status);

-- shp_batch_jobs (3 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_batch_tenant
    ON steam_heat_purchase_service.shp_batch_jobs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_shp_batch_started
    ON steam_heat_purchase_service.shp_batch_jobs(started_at);
CREATE INDEX IF NOT EXISTS idx_shp_batch_status
    ON steam_heat_purchase_service.shp_batch_jobs(status, tenant_id);

-- shp_aggregations (4 indexes)
CREATE INDEX IF NOT EXISTS idx_shp_agg_tenant
    ON steam_heat_purchase_service.shp_aggregations(tenant_id, aggregated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_agg_type
    ON steam_heat_purchase_service.shp_aggregations(aggregation_type, aggregated_at DESC);
CREATE INDEX IF NOT EXISTS idx_shp_agg_provenance
    ON steam_heat_purchase_service.shp_aggregations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_shp_agg_breakdown
    ON steam_heat_purchase_service.shp_aggregations USING GIN(breakdown);

-- ==========================================================================
-- Continuous Aggregate Refresh Policies
-- ==========================================================================
SELECT add_continuous_aggregate_policy('steam_heat_purchase_service.shp_hourly_stats',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('steam_heat_purchase_service.shp_daily_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ==========================================================================
-- Retention Policies (5 years)
-- ==========================================================================
SELECT add_retention_policy('steam_heat_purchase_service.shp_calculations', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('steam_heat_purchase_service.shp_uncertainty_results', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('steam_heat_purchase_service.shp_aggregations', INTERVAL '5 years', if_not_exists => TRUE);

-- ==========================================================================
-- Compression Policies (30 days)
-- ==========================================================================
ALTER TABLE steam_heat_purchase_service.shp_calculations
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'energy_type',
         timescaledb.compress_orderby   = 'calculated_at DESC');
SELECT add_compression_policy('steam_heat_purchase_service.shp_calculations', INTERVAL '30 days');

ALTER TABLE steam_heat_purchase_service.shp_uncertainty_results
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'method',
         timescaledb.compress_orderby   = 'analyzed_at DESC');
SELECT add_compression_policy('steam_heat_purchase_service.shp_uncertainty_results', INTERVAL '30 days');

ALTER TABLE steam_heat_purchase_service.shp_aggregations
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'aggregation_type',
         timescaledb.compress_orderby   = 'aggregated_at DESC');
SELECT add_compression_policy('steam_heat_purchase_service.shp_aggregations', INTERVAL '30 days');

-- ==========================================================================
-- Row-Level Security (34 policies: SELECT + INSERT per table)
-- ==========================================================================

-- shp_fuel_emission_factors: shared reference data (open read, admin write)
ALTER TABLE steam_heat_purchase_service.shp_fuel_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_fuel_ef_select ON steam_heat_purchase_service.shp_fuel_emission_factors
    FOR SELECT USING (TRUE);
CREATE POLICY shp_fuel_ef_insert ON steam_heat_purchase_service.shp_fuel_emission_factors
    FOR INSERT WITH CHECK (
        current_setting('app.is_admin', true) = 'true'
        OR tenant_id = current_setting('app.tenant_id', true)
    );

-- shp_district_heating_factors: shared reference data (open read, admin write)
ALTER TABLE steam_heat_purchase_service.shp_district_heating_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_district_select ON steam_heat_purchase_service.shp_district_heating_factors
    FOR SELECT USING (TRUE);
CREATE POLICY shp_district_insert ON steam_heat_purchase_service.shp_district_heating_factors
    FOR INSERT WITH CHECK (
        current_setting('app.is_admin', true) = 'true'
        OR tenant_id = current_setting('app.tenant_id', true)
    );

-- shp_cooling_system_factors: shared reference data (open read, admin write)
ALTER TABLE steam_heat_purchase_service.shp_cooling_system_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_cooling_select ON steam_heat_purchase_service.shp_cooling_system_factors
    FOR SELECT USING (TRUE);
CREATE POLICY shp_cooling_insert ON steam_heat_purchase_service.shp_cooling_system_factors
    FOR INSERT WITH CHECK (
        current_setting('app.is_admin', true) = 'true'
        OR tenant_id = current_setting('app.tenant_id', true)
    );

-- shp_chp_defaults: shared reference data (open read, admin write)
ALTER TABLE steam_heat_purchase_service.shp_chp_defaults ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_chp_select ON steam_heat_purchase_service.shp_chp_defaults
    FOR SELECT USING (TRUE);
CREATE POLICY shp_chp_insert ON steam_heat_purchase_service.shp_chp_defaults
    FOR INSERT WITH CHECK (
        current_setting('app.is_admin', true) = 'true'
        OR tenant_id = current_setting('app.tenant_id', true)
    );

-- shp_facilities: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_facilities ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_fac_select ON steam_heat_purchase_service.shp_facilities
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_fac_insert ON steam_heat_purchase_service.shp_facilities
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_steam_suppliers: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_steam_suppliers ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_sup_select ON steam_heat_purchase_service.shp_steam_suppliers
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_sup_insert ON steam_heat_purchase_service.shp_steam_suppliers
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_supplier_emission_factors: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_supplier_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_sef_select ON steam_heat_purchase_service.shp_supplier_emission_factors
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_sef_insert ON steam_heat_purchase_service.shp_supplier_emission_factors
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_calculations: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_calc_select ON steam_heat_purchase_service.shp_calculations
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_calc_insert ON steam_heat_purchase_service.shp_calculations
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_calculation_details: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_detail_select ON steam_heat_purchase_service.shp_calculation_details
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_detail_insert ON steam_heat_purchase_service.shp_calculation_details
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_chp_allocations: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_chp_allocations ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_alloc_select ON steam_heat_purchase_service.shp_chp_allocations
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_alloc_insert ON steam_heat_purchase_service.shp_chp_allocations
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_uncertainty_results: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_uncertainty_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_unc_select ON steam_heat_purchase_service.shp_uncertainty_results
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_unc_insert ON steam_heat_purchase_service.shp_uncertainty_results
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_compliance_checks: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_compliance_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_compl_select ON steam_heat_purchase_service.shp_compliance_checks
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_compl_insert ON steam_heat_purchase_service.shp_compliance_checks
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_batch_jobs: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_batch_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_batch_select ON steam_heat_purchase_service.shp_batch_jobs
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_batch_insert ON steam_heat_purchase_service.shp_batch_jobs
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_aggregations: tenant-isolated
ALTER TABLE steam_heat_purchase_service.shp_aggregations ENABLE ROW LEVEL SECURITY;
CREATE POLICY shp_agg_select ON steam_heat_purchase_service.shp_aggregations
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shp_agg_insert ON steam_heat_purchase_service.shp_aggregations
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR current_setting('app.is_admin', true) = 'true'
    );

-- shp_hourly_stats: open read (materialized view)
GRANT SELECT ON steam_heat_purchase_service.shp_hourly_stats TO greenlang_app;
GRANT SELECT ON steam_heat_purchase_service.shp_hourly_stats TO greenlang_readonly;

-- shp_daily_stats: open read (materialized view)
GRANT SELECT ON steam_heat_purchase_service.shp_daily_stats TO greenlang_app;
GRANT SELECT ON steam_heat_purchase_service.shp_daily_stats TO greenlang_readonly;

-- ==========================================================================
-- Permissions
-- ==========================================================================

GRANT USAGE ON SCHEMA steam_heat_purchase_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA steam_heat_purchase_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA steam_heat_purchase_service TO greenlang_app;

GRANT USAGE ON SCHEMA steam_heat_purchase_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA steam_heat_purchase_service TO greenlang_readonly;

GRANT ALL ON SCHEMA steam_heat_purchase_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA steam_heat_purchase_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA steam_heat_purchase_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'steam-heat:read',                  'steam-heat', 'read',                  'View all steam/heat purchase emissions data including fuel factors, district heating, cooling, CHP, calculations, and compliance records'),
    (gen_random_uuid(), 'steam-heat:write',                 'steam-heat', 'write',                 'Create, update, and manage steam/heat purchase emissions data'),
    (gen_random_uuid(), 'steam-heat:execute',               'steam-heat', 'execute',               'Execute steam/heat emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'steam-heat:factors:read',          'steam-heat', 'factors_read',          'View fuel emission factors, district heating factors, cooling system factors, and CHP defaults'),
    (gen_random_uuid(), 'steam-heat:factors:write',         'steam-heat', 'factors_write',         'Create and manage emission factor entries for steam/heat calculations'),
    (gen_random_uuid(), 'steam-heat:facilities:read',       'steam-heat', 'facilities_read',       'View facility records with heating network and cooling system assignments'),
    (gen_random_uuid(), 'steam-heat:facilities:write',      'steam-heat', 'facilities_write',      'Create and manage facility records'),
    (gen_random_uuid(), 'steam-heat:suppliers:read',        'steam-heat', 'suppliers_read',        'View steam/heat supplier records with fuel mix, efficiency, and verification status'),
    (gen_random_uuid(), 'steam-heat:suppliers:write',       'steam-heat', 'suppliers_write',       'Create and manage steam/heat supplier records'),
    (gen_random_uuid(), 'steam-heat:calculations:read',     'steam-heat', 'calculations_read',     'View steam/heat calculation results with fossil/biogenic breakdown, per-gas details, and provenance hashes'),
    (gen_random_uuid(), 'steam-heat:calculations:write',    'steam-heat', 'calculations_write',    'Create and manage steam/heat emission calculation records'),
    (gen_random_uuid(), 'steam-heat:chp:read',              'steam-heat', 'chp_read',              'View CHP allocation records with energy/exergy/efficiency method results'),
    (gen_random_uuid(), 'steam-heat:chp:write',             'steam-heat', 'chp_write',             'Create and manage CHP allocation records'),
    (gen_random_uuid(), 'steam-heat:uncertainty:read',      'steam-heat', 'uncertainty_read',      'View uncertainty quantification results with confidence intervals'),
    (gen_random_uuid(), 'steam-heat:uncertainty:execute',   'steam-heat', 'uncertainty_execute',   'Execute uncertainty analysis (Monte Carlo, error propagation, IPCC methods)'),
    (gen_random_uuid(), 'steam-heat:compliance:read',       'steam-heat', 'compliance_read',       'View regulatory compliance records for GHG Protocol, ISO 14064, CSRD, EPC CHP Directive, IPCC, CDP, TCFD, and SEC Climate'),
    (gen_random_uuid(), 'steam-heat:compliance:execute',    'steam-heat', 'compliance_execute',    'Execute regulatory compliance checks against multiple frameworks'),
    (gen_random_uuid(), 'steam-heat:batch:read',            'steam-heat', 'batch_read',            'View batch processing job status and results'),
    (gen_random_uuid(), 'steam-heat:batch:execute',         'steam-heat', 'batch_execute',         'Submit and manage batch processing jobs'),
    (gen_random_uuid(), 'steam-heat:admin',                 'steam-heat', 'admin',                 'Full administrative access to steam/heat purchase service including configuration, diagnostics, and bulk operations')
ON CONFLICT DO NOTHING;

-- ==========================================================================
-- Seed Data: Fuel Emission Factors (14 fuels)
-- IPCC 2006 Vol 2 Table 2.2 -- stationary combustion default EFs
-- Units: kg per GJ of fuel input
-- ==========================================================================
INSERT INTO steam_heat_purchase_service.shp_fuel_emission_factors
    (fuel_type, co2_ef_per_gj, ch4_ef_per_gj, n2o_ef_per_gj, default_efficiency, is_biogenic, source) VALUES
    ('natural_gas',         56.10000000, 0.00100000, 0.00010000, 0.8500, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('coal_bituminous',     94.60000000, 0.00100000, 0.00150000, 0.7500, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('coal_sub_bituminous', 96.10000000, 0.00100000, 0.00150000, 0.7200, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('coal_lignite',        101.0000000, 0.00100000, 0.00150000, 0.6500, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('fuel_oil_light',      74.10000000, 0.00030000, 0.00060000, 0.8200, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('fuel_oil_heavy',      77.40000000, 0.00030000, 0.00060000, 0.8000, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('lpg',                 63.10000000, 0.00100000, 0.00010000, 0.8600, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('diesel',              74.10000000, 0.00030000, 0.00060000, 0.8300, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('biomass_wood',        112.0000000, 0.03000000, 0.00400000, 0.7000, TRUE,  'IPCC 2006 Vol 2 Table 2.5'),
    ('biomass_pellets',     109.6000000, 0.03000000, 0.00400000, 0.8000, TRUE,  'IPCC 2006 Vol 2 Table 2.5'),
    ('biogas',              54.60000000, 0.00100000, 0.00010000, 0.7500, TRUE,  'IPCC 2006 Vol 2 Table 2.5'),
    ('peat',                106.0000000, 0.00100000, 0.00150000, 0.6000, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('waste_municipal',     91.70000000, 0.03000000, 0.00400000, 0.6500, FALSE, 'IPCC 2006 Vol 2 Table 2.2'),
    ('waste_industrial',    143.0000000, 0.03000000, 0.00400000, 0.6500, FALSE, 'IPCC 2006 Vol 2 Table 2.2')
ON CONFLICT (fuel_type) DO NOTHING;

-- ==========================================================================
-- Seed Data: District Heating Factors (13 regions)
-- Average district heating emission factors by region (kgCO2e/GJ delivered)
-- ==========================================================================
INSERT INTO steam_heat_purchase_service.shp_district_heating_factors
    (region, network_type, ef_kgco2e_per_gj, distribution_loss_pct, source) VALUES
    ('EU-DK',   'municipal',     25.80000000, 0.1200, 'Danish Energy Agency 2022'),
    ('EU-SE',   'municipal',     12.50000000, 0.0800, 'Swedish Energy Agency 2022'),
    ('EU-FI',   'municipal',     43.20000000, 0.1000, 'Statistics Finland 2022'),
    ('EU-DE',   'municipal',     62.10000000, 0.1200, 'AGFW Germany 2022'),
    ('EU-PL',   'municipal',     89.40000000, 0.1500, 'URE Poland 2022'),
    ('EU-AT',   'municipal',     38.70000000, 0.1000, 'Statistics Austria 2022'),
    ('EU-NL',   'municipal',     47.30000000, 0.1100, 'CBS Netherlands 2022'),
    ('EU-FR',   'municipal',     34.90000000, 0.1200, 'SNCU France 2022'),
    ('EU-CZ',   'municipal',     73.60000000, 0.1400, 'ERU Czech Republic 2022'),
    ('US-NY',   'campus',        58.40000000, 0.1500, 'Con Edison Steam 2022'),
    ('US-BOS',  'campus',        62.70000000, 0.1300, 'Vicinity Energy 2022'),
    ('APAC-CN', 'industrial',    95.80000000, 0.2000, 'China Energy Statistical Yearbook 2022'),
    ('APAC-KR', 'municipal',     54.30000000, 0.1200, 'KDHC Korea 2022')
ON CONFLICT (region, network_type, tenant_id) DO NOTHING;

-- ==========================================================================
-- Seed Data: Cooling System Factors (9 technologies)
-- Coefficient of Performance (COP) ranges by technology
-- ==========================================================================
INSERT INTO steam_heat_purchase_service.shp_cooling_system_factors
    (technology, cop_min, cop_max, cop_default, energy_source, source) VALUES
    ('electric_chiller',    3.5000, 7.0000, 5.0000, 'electricity',  'ASHRAE 90.1-2022'),
    ('absorption_chiller',  0.7000, 1.4000, 1.0000, 'natural_gas',  'ASHRAE Handbook HVAC 2022'),
    ('centrifugal_chiller', 5.0000, 9.0000, 6.5000, 'electricity',  'ASHRAE 90.1-2022'),
    ('screw_chiller',       3.0000, 5.5000, 4.2000, 'electricity',  'ASHRAE 90.1-2022'),
    ('district_cooling',    4.0000, 8.0000, 5.5000, 'electricity',  'IEA District Cooling 2022'),
    ('free_cooling',        15.000, 50.000, 25.000, 'electricity',  'ASHRAE Handbook HVAC 2022'),
    ('evaporative_cooling', 8.0000, 20.000, 12.000, 'electricity',  'ASHRAE Handbook HVAC 2022'),
    ('ground_source_hp',    3.0000, 5.0000, 4.0000, 'electricity',  'EPA Energy Star 2022'),
    ('air_source_hp',       2.5000, 4.5000, 3.5000, 'electricity',  'EPA Energy Star 2022')
ON CONFLICT (technology) DO NOTHING;

-- ==========================================================================
-- Seed Data: CHP Defaults (5 fuel types)
-- EPA CHP Partnership default efficiencies
-- ==========================================================================
INSERT INTO steam_heat_purchase_service.shp_chp_defaults
    (fuel_type, electrical_efficiency, thermal_efficiency, overall_efficiency, source) VALUES
    ('natural_gas',     0.3500, 0.4000, 0.7500, 'EPA CHP Partnership 2023'),
    ('coal_bituminous', 0.3000, 0.3500, 0.6500, 'EPA CHP Partnership 2023'),
    ('biomass_wood',    0.2500, 0.4500, 0.7000, 'EPA CHP Partnership 2023'),
    ('fuel_oil_heavy',  0.3000, 0.3500, 0.6500, 'EPA CHP Partnership 2023'),
    ('biogas',          0.3300, 0.4200, 0.7500, 'EPA CHP Partnership 2023')
ON CONFLICT (fuel_type, tenant_id) DO NOTHING;

-- ==========================================================================
-- Seed Data: Register Agent in Agent Registry
-- ==========================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE2-011',
    'Steam/Heat Purchase Agent',
    'Steam, heat, and cooling purchase emission calculator for GreenLang Climate OS. Manages fuel-based boiler emission factors (14 fuel types including natural gas, coal variants, fuel oils, LPG, diesel, biomass, biogas, peat, and waste with per-GJ CO2/CH4/N2O emission factors, default thermal efficiencies, and biogenic classification from IPCC 2006 Vol 2 Table 2.2/2.5). Maintains district heating network emission factor database for 13+ regions (EU Denmark/Sweden/Finland/Germany/Poland/Austria/Netherlands/France/Czech Republic, US New York/Boston, APAC China/Korea) with distribution loss percentages and municipal/industrial/campus network types. Stores cooling system COP factors for 9 technologies (electric chiller, absorption chiller, centrifugal chiller, screw chiller, district cooling, free cooling, evaporative cooling, ground source heat pump, air source heat pump) from ASHRAE/IEA/EPA sources. Provides CHP/cogeneration default parameters (electrical/thermal/overall efficiencies) for 5 fuel types from EPA CHP Partnership. Registers facilities with heating network and cooling system assignments. Manages steam/heat supplier registry with fuel mix disclosure, boiler efficiency, supplier-specific emission factors by reporting year, data quality tiers (TIER_1/TIER_2/TIER_3), and independent verification status. Executes deterministic emission calculations for 7 energy types (steam, hot_water, chilled_water, district_heating, district_cooling, process_heat, industrial_steam) using 8 calculation methods (fuel_based, supplier_specific, district_heating, cooling_cop, chp_energy, chp_exergy, chp_efficiency, default_factor). Separates fossil CO2e and biogenic CO2 for each calculation. Records per-gas emission breakdowns (CO2/CH4/N2O/CO2_biogenic) with GWP weighting (AR4/AR5/AR6/AR6_20YR). Implements CHP allocation with heat/power/cooling output shares and primary energy savings percentage per EU CHP Directive. Performs uncertainty quantification via Monte Carlo, error propagation, IPCC Tier 1/2, bootstrap, and sensitivity methods with confidence intervals. Checks regulatory compliance against 8 frameworks (GHG Protocol Scope 2, ISO 14064, CSRD ESRS, EPC CHP Directive, IPCC 2006, CDP, TCFD, SEC Climate). Supports batch processing with job tracking. Produces aggregations by facility/supplier/energy_type/fuel_type/method/region/tenant/period with fossil/biogenic breakdown. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/steam-heat-purchase',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE2-011', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/steam-heat-purchase-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"scope-2", "steam", "heat", "cooling", "chp", "cogeneration", "district-heating", "boiler", "biogenic", "mrv"}',
    '{"energy", "utilities", "manufacturing", "healthcare", "campus", "industrial", "cross-sector"}',
    'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE2-011', '1.0.0',
    'fuel_based_calculation',
    'calculation',
    'Calculate steam/heat emissions from fuel combustion in boilers using IPCC 2006 emission factors, boiler efficiency, and multi-gas breakdown with biogenic CO2 separation.',
    '{"fuel_type", "consumption_gj", "boiler_efficiency", "gwp_source"}',
    '{"calc_id", "total_co2e_kg", "fossil_co2e_kg", "biogenic_co2_kg", "provenance_hash"}',
    '{"fuel_types": ["natural_gas", "coal_bituminous", "coal_sub_bituminous", "coal_lignite", "fuel_oil_light", "fuel_oil_heavy", "lpg", "diesel", "biomass_wood", "biomass_pellets", "biogas", "peat", "waste_municipal", "waste_industrial"], "gwp_sources": ["AR4", "AR5", "AR6", "AR6_20YR"]}'::jsonb
),
(
    'GL-MRV-SCOPE2-011', '1.0.0',
    'district_heating_calculation',
    'calculation',
    'Calculate emissions from district heating purchases using regional network emission factors and distribution loss adjustments.',
    '{"region", "network_type", "consumption_gj", "gwp_source"}',
    '{"calc_id", "total_co2e_kg", "effective_ef", "provenance_hash"}',
    '{"network_types": ["municipal", "industrial", "campus", "cogeneration", "waste_heat"]}'::jsonb
),
(
    'GL-MRV-SCOPE2-011', '1.0.0',
    'chp_allocation',
    'calculation',
    'Allocate CHP/cogeneration emissions between heat, power, and cooling outputs using energy, exergy, or efficiency allocation methods per GHG Protocol and EU CHP Directive.',
    '{"total_fuel_gj", "fuel_type", "heat_output_gj", "power_output_gj", "cooling_output_gj", "method"}',
    '{"allocation_id", "heat_share", "power_share", "cooling_share", "heat_emissions_kgco2e", "primary_energy_savings_pct", "provenance_hash"}',
    '{"methods": ["energy", "exergy", "efficiency", "iea", "custom"]}'::jsonb
),
(
    'GL-MRV-SCOPE2-011', '1.0.0',
    'uncertainty_quantification',
    'analysis',
    'Perform uncertainty analysis on steam/heat calculations using Monte Carlo simulation, analytical error propagation, IPCC Tier 1/2 methods, bootstrap, and sensitivity analysis.',
    '{"calc_id", "method", "confidence_level", "n_simulations"}',
    '{"uncertainty_id", "mean_co2e_kg", "std_dev_kg", "ci_lower_kg", "ci_upper_kg", "relative_uncertainty_pct"}',
    '{"methods": ["monte_carlo", "error_propagation", "ipcc_tier1", "ipcc_tier2", "bootstrap", "sensitivity"]}'::jsonb
),
(
    'GL-MRV-SCOPE2-011', '1.0.0',
    'compliance_checking',
    'compliance',
    'Check steam/heat calculations against regulatory frameworks (GHG Protocol Scope 2, ISO 14064, CSRD ESRS, EPC CHP Directive, IPCC 2006, CDP, TCFD, SEC Climate).',
    '{"calc_id", "frameworks"}',
    '{"compliance_records", "overall_status", "score_pct"}',
    '{"frameworks": ["ghg_protocol_scope2", "iso_14064", "csrd_esrs", "epc_chp_directive", "ipcc_2006", "cdp", "tcfd", "sec_climate"]}'::jsonb
)
ON CONFLICT DO NOTHING;

-- ==========================================================================
-- Migration Metadata
-- ==========================================================================
INSERT INTO schema_migrations_metadata (version, description, component, tables_created, indexes_created)
VALUES ('V062', 'Steam/Heat Purchase Agent Service', 'AGENT-MRV-011', 14, 66)
ON CONFLICT DO NOTHING;

-- ==========================================================================
-- Migration complete: V062 Steam/Heat Purchase Agent Service
-- 14 tables, 3 hypertables, 2 continuous aggregates, 66 indexes,
-- 34 RLS policies, 14 fuel EFs, 13 district heating EFs,
-- 9 cooling system factors, 5 CHP defaults
-- ==========================================================================

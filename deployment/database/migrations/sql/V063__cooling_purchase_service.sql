-- V063__cooling_purchase_service.sql
-- AGENT-MRV-012: Cooling Purchase Agent Database Schema
-- Scope 2 (purchased cooling from electric chillers, absorption chillers, district cooling, free cooling, TES)
-- Pattern: V062 (steam_heat_purchase) with cp_ prefix
-- Tables: 14 tables, 3 hypertables, 2 continuous aggregates
-- Author: GL-BackendDeveloper
-- Date: 2026-02-22

-- ============================================================================
-- DIMENSION TABLES (7)
-- ============================================================================

-- 1. Cooling Technologies (18 rows)
CREATE TABLE cp_cooling_technologies (
    id SERIAL PRIMARY KEY,
    technology_key VARCHAR(60) UNIQUE NOT NULL,
    display_name VARCHAR(120) NOT NULL,
    category VARCHAR(40) NOT NULL CHECK (category IN ('electric', 'absorption', 'free_cooling', 'tes', 'district')),
    compressor_type VARCHAR(30),
    condenser_type VARCHAR(20),
    cop_min NUMERIC(8,4) NOT NULL,
    cop_max NUMERIC(8,4) NOT NULL,
    cop_default NUMERIC(8,4) NOT NULL,
    iplv NUMERIC(8,4),
    energy_source VARCHAR(60) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_cp_cooling_technologies_tenant ON cp_cooling_technologies(tenant_id);
CREATE INDEX idx_cp_cooling_technologies_category ON cp_cooling_technologies(category);
CREATE INDEX idx_cp_cooling_technologies_energy_source ON cp_cooling_technologies(energy_source);

ALTER TABLE cp_cooling_technologies ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_cooling_technologies_tenant_isolation ON cp_cooling_technologies
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 2. District Cooling Factors (12 rows)
CREATE TABLE cp_district_cooling_factors (
    id SERIAL PRIMARY KEY,
    region_key VARCHAR(40) UNIQUE NOT NULL,
    display_name VARCHAR(120) NOT NULL,
    ef_kgco2e_per_gj NUMERIC(10,4) NOT NULL,
    technology_mix VARCHAR(200),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_cp_district_cooling_factors_tenant ON cp_district_cooling_factors(tenant_id);
CREATE INDEX idx_cp_district_cooling_factors_region ON cp_district_cooling_factors(region_key);

ALTER TABLE cp_district_cooling_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_district_cooling_factors_tenant_isolation ON cp_district_cooling_factors
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 3. Heat Source Factors for Absorption Chillers (11 rows)
CREATE TABLE cp_heat_source_factors (
    id SERIAL PRIMARY KEY,
    heat_source_key VARCHAR(40) UNIQUE NOT NULL,
    display_name VARCHAR(120) NOT NULL,
    ef_kgco2e_per_gj NUMERIC(10,4) NOT NULL,
    is_zero_emission BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_cp_heat_source_factors_tenant ON cp_heat_source_factors(tenant_id);
CREATE INDEX idx_cp_heat_source_factors_source ON cp_heat_source_factors(heat_source_key);

ALTER TABLE cp_heat_source_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_heat_source_factors_tenant_isolation ON cp_heat_source_factors
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 4. Refrigerant Data (11 rows)
CREATE TABLE cp_refrigerant_data (
    id SERIAL PRIMARY KEY,
    refrigerant_key VARCHAR(30) UNIQUE NOT NULL,
    display_name VARCHAR(60) NOT NULL,
    gwp_ar5 NUMERIC(10,2) NOT NULL,
    gwp_ar6 NUMERIC(10,2) NOT NULL,
    common_use VARCHAR(200),
    phase_down VARCHAR(200),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_cp_refrigerant_data_tenant ON cp_refrigerant_data(tenant_id);
CREATE INDEX idx_cp_refrigerant_data_refrigerant ON cp_refrigerant_data(refrigerant_key);

ALTER TABLE cp_refrigerant_data ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_refrigerant_data_tenant_isolation ON cp_refrigerant_data
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 5. Part Load Curves (AHRI 550/590 standard)
CREATE TABLE cp_part_load_curves (
    id SERIAL PRIMARY KEY,
    technology_id INT NOT NULL REFERENCES cp_cooling_technologies(id) ON DELETE CASCADE,
    load_pct NUMERIC(5,2) NOT NULL CHECK (load_pct BETWEEN 0 AND 100),
    cop_multiplier NUMERIC(6,4) NOT NULL,
    weighting NUMERIC(6,4) NOT NULL CHECK (weighting BETWEEN 0 AND 1),
    standard VARCHAR(40) NOT NULL DEFAULT 'AHRI_550_590',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_cp_part_load_curves_technology ON cp_part_load_curves(technology_id);
CREATE INDEX idx_cp_part_load_curves_tenant ON cp_part_load_curves(tenant_id);
CREATE INDEX idx_cp_part_load_curves_load ON cp_part_load_curves(load_pct);

ALTER TABLE cp_part_load_curves ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_part_load_curves_tenant_isolation ON cp_part_load_curves
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 6. Facilities
CREATE TABLE cp_facilities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    facility_type VARCHAR(40) NOT NULL,
    tenant_id UUID NOT NULL,
    cooling_demand_kwh_th NUMERIC(18,4),
    location VARCHAR(200),
    latitude NUMERIC(10,6),
    longitude NUMERIC(11,6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cp_facilities_tenant ON cp_facilities(tenant_id);
CREATE INDEX idx_cp_facilities_type ON cp_facilities(facility_type);
CREATE INDEX idx_cp_facilities_location ON cp_facilities(location);

ALTER TABLE cp_facilities ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_facilities_tenant_isolation ON cp_facilities
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 7. Cooling Suppliers (district cooling, chiller vendors)
CREATE TABLE cp_cooling_suppliers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    technology VARCHAR(60),
    cop_rated NUMERIC(8,4),
    iplv_rated NUMERIC(8,4),
    refrigerant VARCHAR(30),
    charge_kg NUMERIC(12,4),
    annual_leak_rate NUMERIC(6,4),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cp_cooling_suppliers_tenant ON cp_cooling_suppliers(tenant_id);
CREATE INDEX idx_cp_cooling_suppliers_technology ON cp_cooling_suppliers(technology);
CREATE INDEX idx_cp_cooling_suppliers_refrigerant ON cp_cooling_suppliers(refrigerant);

ALTER TABLE cp_cooling_suppliers ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_cooling_suppliers_tenant_isolation ON cp_cooling_suppliers
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- ============================================================================
-- HYPERTABLES (3)
-- ============================================================================

-- 8. Cooling Calculations (main hypertable)
CREATE TABLE cp_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_type VARCHAR(40) NOT NULL CHECK (calculation_type IN ('electric', 'absorption', 'district', 'free_cooling', 'tes')),
    cooling_output_kwh_th NUMERIC(18,4) NOT NULL,
    energy_input_kwh NUMERIC(18,4),
    cop_used NUMERIC(8,4),
    emissions_kgco2e NUMERIC(18,8) NOT NULL,
    calculation_tier VARCHAR(10),
    gwp_source VARCHAR(10),
    technology VARCHAR(60),
    region VARCHAR(40),
    facility_id UUID REFERENCES cp_facilities(id),
    supplier_id UUID REFERENCES cp_cooling_suppliers(id),
    tenant_id UUID NOT NULL,
    provenance_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('cp_calculations', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_cp_calculations_tenant_time ON cp_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_cp_calculations_type ON cp_calculations(calculation_type);
CREATE INDEX idx_cp_calculations_technology ON cp_calculations(technology);
CREATE INDEX idx_cp_calculations_facility ON cp_calculations(facility_id);
CREATE INDEX idx_cp_calculations_supplier ON cp_calculations(supplier_id);
CREATE INDEX idx_cp_calculations_region ON cp_calculations(region);
CREATE INDEX idx_cp_calculations_provenance ON cp_calculations(provenance_hash);

ALTER TABLE cp_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_calculations_tenant_isolation ON cp_calculations
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 9. Uncertainty Results (hypertable)
CREATE TABLE cp_uncertainty_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    method VARCHAR(20) NOT NULL CHECK (method IN ('monte_carlo', 'analytical')),
    mean_emissions NUMERIC(18,8),
    std_dev NUMERIC(18,8),
    ci_lower NUMERIC(18,8),
    ci_upper NUMERIC(18,8),
    confidence_level NUMERIC(4,2),
    iterations INT,
    p5 NUMERIC(18,8),
    p25 NUMERIC(18,8),
    p50 NUMERIC(18,8),
    p75 NUMERIC(18,8),
    p95 NUMERIC(18,8),
    cv NUMERIC(8,4),
    tenant_id UUID NOT NULL,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('cp_uncertainty_results', 'analyzed_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_cp_uncertainty_results_calculation ON cp_uncertainty_results(calculation_id);
CREATE INDEX idx_cp_uncertainty_results_tenant_time ON cp_uncertainty_results(tenant_id, analyzed_at DESC);
CREATE INDEX idx_cp_uncertainty_results_method ON cp_uncertainty_results(method);

ALTER TABLE cp_uncertainty_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_uncertainty_results_tenant_isolation ON cp_uncertainty_results
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 10. Aggregations (hypertable)
CREATE TABLE cp_aggregations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregation_type VARCHAR(30) NOT NULL,
    group_key VARCHAR(200) NOT NULL,
    total_emissions_kgco2e NUMERIC(18,8),
    calculation_count INT,
    provenance_hash VARCHAR(64),
    tenant_id UUID NOT NULL,
    aggregated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('cp_aggregations', 'aggregated_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_cp_aggregations_tenant_time ON cp_aggregations(tenant_id, aggregated_at DESC);
CREATE INDEX idx_cp_aggregations_type ON cp_aggregations(aggregation_type);
CREATE INDEX idx_cp_aggregations_group ON cp_aggregations(group_key);

ALTER TABLE cp_aggregations ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_aggregations_tenant_isolation ON cp_aggregations
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- ============================================================================
-- REGULAR TABLES (4)
-- ============================================================================

-- 11. Calculation Details
CREATE TABLE cp_calculation_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    component VARCHAR(60),
    value NUMERIC(18,8),
    unit VARCHAR(30),
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cp_calculation_details_calculation ON cp_calculation_details(calculation_id);
CREATE INDEX idx_cp_calculation_details_component ON cp_calculation_details(component);

-- 12. TES (Thermal Energy Storage) Calculations
CREATE TABLE cp_tes_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    tes_type VARCHAR(30),
    capacity_kwh_th NUMERIC(18,4),
    charge_energy_kwh NUMERIC(18,4),
    round_trip_efficiency NUMERIC(6,4),
    grid_ef_charge NUMERIC(10,6),
    grid_ef_peak NUMERIC(10,6),
    emission_savings_kgco2e NUMERIC(18,8),
    peak_emissions_avoided_kgco2e NUMERIC(18,8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cp_tes_calculations_calculation ON cp_tes_calculations(calculation_id);
CREATE INDEX idx_cp_tes_calculations_type ON cp_tes_calculations(tes_type);

-- 13. Compliance Checks
CREATE TABLE cp_compliance_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    framework VARCHAR(40) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pass', 'fail', 'warning', 'not_applicable')),
    requirements_total INT,
    requirements_met INT,
    score NUMERIC(5,2),
    findings JSONB DEFAULT '[]',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cp_compliance_checks_calculation ON cp_compliance_checks(calculation_id);
CREATE INDEX idx_cp_compliance_checks_framework ON cp_compliance_checks(framework);
CREATE INDEX idx_cp_compliance_checks_status ON cp_compliance_checks(status);

-- 14. Batch Jobs
CREATE TABLE cp_batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    total_calculations INT,
    completed INT DEFAULT 0,
    failed INT DEFAULT 0,
    total_emissions_kgco2e NUMERIC(18,8),
    processing_time_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_cp_batch_jobs_tenant ON cp_batch_jobs(tenant_id);
CREATE INDEX idx_cp_batch_jobs_status ON cp_batch_jobs(status);
CREATE INDEX idx_cp_batch_jobs_created ON cp_batch_jobs(created_at DESC);

ALTER TABLE cp_batch_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_batch_jobs_tenant_isolation ON cp_batch_jobs
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- ============================================================================
-- CONTINUOUS AGGREGATES (2)
-- ============================================================================

-- 15. Hourly Stats
CREATE MATERIALIZED VIEW cp_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', calculated_at) AS bucket,
    tenant_id,
    calculation_type,
    COUNT(*) AS calculation_count,
    SUM(emissions_kgco2e) AS total_emissions_kgco2e,
    AVG(cop_used) AS avg_cop,
    SUM(cooling_output_kwh_th) AS total_cooling_kwh_th,
    AVG(cooling_output_kwh_th) AS avg_cooling_kwh_th,
    MIN(calculated_at) AS first_calculation,
    MAX(calculated_at) AS last_calculation
FROM cp_calculations
GROUP BY bucket, tenant_id, calculation_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('cp_hourly_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX idx_cp_hourly_stats_tenant_time ON cp_hourly_stats(tenant_id, bucket DESC);
CREATE INDEX idx_cp_hourly_stats_type ON cp_hourly_stats(calculation_type);

-- 16. Daily Stats
CREATE MATERIALIZED VIEW cp_daily_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    tenant_id,
    calculation_type,
    COUNT(*) AS calculation_count,
    SUM(emissions_kgco2e) AS total_emissions_kgco2e,
    AVG(cop_used) AS avg_cop,
    SUM(cooling_output_kwh_th) AS total_cooling_kwh_th,
    AVG(cooling_output_kwh_th) AS avg_cooling_kwh_th,
    MIN(calculated_at) AS first_calculation,
    MAX(calculated_at) AS last_calculation
FROM cp_calculations
GROUP BY bucket, tenant_id, calculation_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('cp_daily_stats',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_cp_daily_stats_tenant_time ON cp_daily_stats(tenant_id, bucket DESC);
CREATE INDEX idx_cp_daily_stats_type ON cp_daily_stats(calculation_type);

-- ============================================================================
-- SEED DATA: Cooling Technologies (18 rows)
-- ============================================================================

INSERT INTO cp_cooling_technologies (technology_key, display_name, category, compressor_type, condenser_type, cop_min, cop_max, cop_default, iplv, energy_source) VALUES
-- Electric Chillers (8)
('air_cooled_scroll', 'Air-Cooled Scroll Chiller', 'electric', 'scroll', 'air', 2.5, 3.2, 2.8, 3.5, 'electricity'),
('air_cooled_screw', 'Air-Cooled Screw Chiller', 'electric', 'screw', 'air', 2.8, 3.5, 3.1, 3.8, 'electricity'),
('water_cooled_centrifugal', 'Water-Cooled Centrifugal Chiller', 'electric', 'centrifugal', 'water', 5.5, 7.5, 6.5, 8.2, 'electricity'),
('water_cooled_screw', 'Water-Cooled Screw Chiller', 'electric', 'screw', 'water', 4.5, 5.5, 5.0, 6.0, 'electricity'),
('magnetic_bearing_centrifugal', 'Magnetic Bearing Centrifugal Chiller', 'electric', 'centrifugal', 'water', 6.5, 9.0, 7.8, 10.0, 'electricity'),
('vrf_heat_pump', 'VRF Heat Pump System', 'electric', 'scroll', 'air', 3.0, 4.5, 3.8, 4.8, 'electricity'),
('evaporative_cooled_scroll', 'Evaporative-Cooled Scroll Chiller', 'electric', 'scroll', 'evaporative', 3.5, 4.2, 3.8, 4.5, 'electricity'),
('reciprocating_chiller', 'Reciprocating Chiller', 'electric', 'reciprocating', 'air', 2.2, 2.8, 2.5, 3.0, 'electricity'),

-- Absorption Chillers (4)
('single_effect_absorption', 'Single-Effect Absorption Chiller', 'absorption', NULL, NULL, 0.65, 0.75, 0.70, NULL, 'natural_gas'),
('double_effect_absorption', 'Double-Effect Absorption Chiller', 'absorption', NULL, NULL, 1.1, 1.3, 1.2, NULL, 'natural_gas'),
('triple_effect_absorption', 'Triple-Effect Absorption Chiller', 'absorption', NULL, NULL, 1.5, 1.7, 1.6, NULL, 'natural_gas'),
('waste_heat_absorption', 'Waste Heat-Driven Absorption Chiller', 'absorption', NULL, NULL, 0.65, 0.75, 0.70, NULL, 'waste_heat'),

-- Free Cooling (3)
('waterside_economizer', 'Waterside Economizer', 'free_cooling', NULL, NULL, 20.0, 40.0, 30.0, NULL, 'ambient'),
('airside_economizer', 'Airside Economizer', 'free_cooling', NULL, NULL, 15.0, 35.0, 25.0, NULL, 'ambient'),
('dry_cooler', 'Dry Cooler Free Cooling', 'free_cooling', NULL, NULL, 12.0, 25.0, 18.0, NULL, 'ambient'),

-- TES (2)
('ice_storage', 'Ice Thermal Energy Storage', 'tes', NULL, NULL, 2.8, 3.5, 3.1, NULL, 'electricity'),
('chilled_water_storage', 'Chilled Water Thermal Energy Storage', 'tes', NULL, NULL, 3.2, 4.0, 3.6, NULL, 'electricity'),

-- District Cooling (1)
('district_cooling', 'District Cooling Network', 'district', NULL, NULL, 4.0, 6.0, 5.0, NULL, 'district_network');

-- ============================================================================
-- SEED DATA: District Cooling Factors (12 rows)
-- ============================================================================

INSERT INTO cp_district_cooling_factors (region_key, display_name, ef_kgco2e_per_gj, technology_mix, notes) VALUES
('middle_east_gcc', 'Middle East (GCC)', 18.5, 'Electric chillers (80%), Absorption (20%)', 'Gulf Cooperation Council region'),
('singapore', 'Singapore', 12.3, 'Electric chillers (90%), Free cooling (10%)', 'Urban district cooling'),
('sweden_stockholm', 'Sweden (Stockholm)', 2.5, 'Free cooling (60%), Heat pumps (40%)', 'Low-carbon district cooling'),
('france_paris', 'France (Paris)', 8.7, 'Electric chillers (70%), Free cooling (30%)', 'Paris district cooling network'),
('us_new_york', 'United States (New York)', 14.2, 'Electric chillers (85%), Free cooling (15%)', 'NYC district cooling'),
('canada_toronto', 'Canada (Toronto)', 6.8, 'Free cooling (50%), Electric chillers (50%)', 'Deep lake water cooling'),
('japan_tokyo', 'Japan (Tokyo)', 15.6, 'Electric chillers (95%), Cogeneration (5%)', 'Tokyo district cooling'),
('korea_seoul', 'South Korea (Seoul)', 16.1, 'Electric chillers (90%), Absorption (10%)', 'Seoul district cooling'),
('china_beijing', 'China (Beijing)', 22.4, 'Electric chillers (85%), Absorption (15%)', 'Beijing district cooling'),
('india_delhi', 'India (Delhi)', 28.7, 'Electric chillers (95%), Solar absorption (5%)', 'Delhi district cooling'),
('australia_sydney', 'Australia (Sydney)', 18.9, 'Electric chillers (90%), Seawater cooling (10%)', 'Sydney CBD district cooling'),
('uae_dubai', 'UAE (Dubai)', 21.3, 'Electric chillers (75%), Absorption (25%)', 'Dubai district cooling');

-- ============================================================================
-- SEED DATA: Heat Source Factors for Absorption Chillers (11 rows)
-- ============================================================================

INSERT INTO cp_heat_source_factors (heat_source_key, display_name, ef_kgco2e_per_gj, is_zero_emission, notes) VALUES
('natural_gas', 'Natural Gas', 56.1, FALSE, 'Combustion for heat'),
('fuel_oil', 'Fuel Oil', 77.4, FALSE, 'Heavy fuel oil'),
('diesel', 'Diesel', 74.1, FALSE, 'Light diesel oil'),
('lpg', 'Liquefied Petroleum Gas (LPG)', 63.1, FALSE, 'Propane/butane'),
('biomass', 'Biomass', 0.0, TRUE, 'Carbon-neutral biomass'),
('waste_heat_industrial', 'Industrial Waste Heat', 0.0, TRUE, 'Recovered process heat'),
('waste_heat_cogeneration', 'Cogeneration Waste Heat', 0.0, TRUE, 'CHP exhaust heat'),
('solar_thermal', 'Solar Thermal', 0.0, TRUE, 'Solar collectors'),
('geothermal', 'Geothermal', 0.0, TRUE, 'Ground source heat'),
('district_heat', 'District Heating Network', 25.3, FALSE, 'Average district heat'),
('electric_resistance', 'Electric Resistance Heating', 180.5, FALSE, 'Direct electric heating (high EF)');

-- ============================================================================
-- SEED DATA: Refrigerant Data (11 rows)
-- ============================================================================

INSERT INTO cp_refrigerant_data (refrigerant_key, display_name, gwp_ar5, gwp_ar6, common_use, phase_down) VALUES
('R-134a', 'R-134a (HFC-134a)', 1430.00, 1530.00, 'Common in older chillers', 'EU F-Gas phase-down'),
('R-410A', 'R-410A (HFC-410A)', 2088.00, 2265.00, 'Air conditioning, VRF systems', 'EU F-Gas phase-down'),
('R-32', 'R-32 (HFC-32)', 675.00, 771.00, 'Modern VRF, split systems', 'Lower GWP alternative'),
('R-407C', 'R-407C (HFC-407C)', 1774.00, 1923.00, 'Retrofit for R-22', 'EU F-Gas phase-down'),
('R-513A', 'R-513A (Opteon XP10)', 631.00, 573.00, 'Low-GWP replacement for R-134a', 'HFO blend'),
('R-1234yf', 'R-1234yf (HFO-1234yf)', 4.00, 1.00, 'Very low GWP, automotive AC', 'Next-gen refrigerant'),
('R-1234ze', 'R-1234ze (HFO-1234ze)', 6.00, 1.00, 'Centrifugal chillers', 'Next-gen refrigerant'),
('R-717', 'R-717 (Ammonia)', 0.00, 0.00, 'Industrial refrigeration', 'Natural refrigerant'),
('R-744', 'R-744 (CO2)', 1.00, 1.00, 'Transcritical systems', 'Natural refrigerant'),
('R-290', 'R-290 (Propane)', 3.00, 3.00, 'Small systems, residential', 'Natural refrigerant'),
('R-600a', 'R-600a (Isobutane)', 3.00, 3.00, 'Domestic refrigeration', 'Natural refrigerant');

-- ============================================================================
-- SEED DATA: Part-Load Curves (AHRI 550/590 - 4 points per electric chiller)
-- ============================================================================

-- Air-Cooled Scroll (id=1)
INSERT INTO cp_part_load_curves (technology_id, load_pct, cop_multiplier, weighting, standard) VALUES
(1, 100.00, 1.000, 0.01, 'AHRI_550_590'),
(1, 75.00, 1.040, 0.42, 'AHRI_550_590'),
(1, 50.00, 1.080, 0.45, 'AHRI_550_590'),
(1, 25.00, 1.050, 0.12, 'AHRI_550_590');

-- Air-Cooled Screw (id=2)
INSERT INTO cp_part_load_curves (technology_id, load_pct, cop_multiplier, weighting, standard) VALUES
(2, 100.00, 1.000, 0.01, 'AHRI_550_590'),
(2, 75.00, 1.050, 0.42, 'AHRI_550_590'),
(2, 50.00, 1.090, 0.45, 'AHRI_550_590'),
(2, 25.00, 1.060, 0.12, 'AHRI_550_590');

-- Water-Cooled Centrifugal (id=3)
INSERT INTO cp_part_load_curves (technology_id, load_pct, cop_multiplier, weighting, standard) VALUES
(3, 100.00, 1.000, 0.01, 'AHRI_550_590'),
(3, 75.00, 1.080, 0.42, 'AHRI_550_590'),
(3, 50.00, 1.130, 0.45, 'AHRI_550_590'),
(3, 25.00, 1.100, 0.12, 'AHRI_550_590');

-- Water-Cooled Screw (id=4)
INSERT INTO cp_part_load_curves (technology_id, load_pct, cop_multiplier, weighting, standard) VALUES
(4, 100.00, 1.000, 0.01, 'AHRI_550_590'),
(4, 75.00, 1.070, 0.42, 'AHRI_550_590'),
(4, 50.00, 1.110, 0.45, 'AHRI_550_590'),
(4, 25.00, 1.080, 0.12, 'AHRI_550_590');

-- Magnetic Bearing Centrifugal (id=5)
INSERT INTO cp_part_load_curves (technology_id, load_pct, cop_multiplier, weighting, standard) VALUES
(5, 100.00, 1.000, 0.01, 'AHRI_550_590'),
(5, 75.00, 1.100, 0.42, 'AHRI_550_590'),
(5, 50.00, 1.180, 0.45, 'AHRI_550_590'),
(5, 25.00, 1.150, 0.12, 'AHRI_550_590');

-- VRF Heat Pump (id=6)
INSERT INTO cp_part_load_curves (technology_id, load_pct, cop_multiplier, weighting, standard) VALUES
(6, 100.00, 1.000, 0.01, 'AHRI_550_590'),
(6, 75.00, 1.060, 0.42, 'AHRI_550_590'),
(6, 50.00, 1.100, 0.45, 'AHRI_550_590'),
(6, 25.00, 1.070, 0.12, 'AHRI_550_590');

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE cp_cooling_technologies IS 'AGENT-MRV-012: Cooling technology reference data (electric, absorption, free cooling, TES, district)';
COMMENT ON TABLE cp_district_cooling_factors IS 'AGENT-MRV-012: Emission factors for district cooling networks by region';
COMMENT ON TABLE cp_heat_source_factors IS 'AGENT-MRV-012: Emission factors for heat sources driving absorption chillers';
COMMENT ON TABLE cp_refrigerant_data IS 'AGENT-MRV-012: Refrigerant GWP data (AR5/AR6) for leakage calculations';
COMMENT ON TABLE cp_part_load_curves IS 'AGENT-MRV-012: Part-load performance curves (AHRI 550/590 IPLV)';
COMMENT ON TABLE cp_facilities IS 'AGENT-MRV-012: Facility master data with cooling demand';
COMMENT ON TABLE cp_cooling_suppliers IS 'AGENT-MRV-012: Cooling supplier/vendor data (district cooling, chiller manufacturers)';
COMMENT ON TABLE cp_calculations IS 'AGENT-MRV-012: Main cooling emissions calculations (hypertable on calculated_at)';
COMMENT ON TABLE cp_uncertainty_results IS 'AGENT-MRV-012: Monte Carlo and analytical uncertainty quantification (hypertable)';
COMMENT ON TABLE cp_aggregations IS 'AGENT-MRV-012: Pre-computed aggregations by technology/region/facility (hypertable)';
COMMENT ON TABLE cp_calculation_details IS 'AGENT-MRV-012: Detailed calculation components and intermediate values';
COMMENT ON TABLE cp_tes_calculations IS 'AGENT-MRV-012: Thermal energy storage calculations with load shifting benefits';
COMMENT ON TABLE cp_compliance_checks IS 'AGENT-MRV-012: Compliance validation against GHG Protocol/ISO 14064/CSRD/SECR';
COMMENT ON TABLE cp_batch_jobs IS 'AGENT-MRV-012: Batch processing jobs for bulk cooling calculations';

COMMENT ON MATERIALIZED VIEW cp_hourly_stats IS 'AGENT-MRV-012: Hourly aggregated cooling statistics (continuous aggregate)';
COMMENT ON MATERIALIZED VIEW cp_daily_stats IS 'AGENT-MRV-012: Daily aggregated cooling statistics (continuous aggregate)';

-- ============================================================================
-- END V063__cooling_purchase_service.sql
-- ============================================================================

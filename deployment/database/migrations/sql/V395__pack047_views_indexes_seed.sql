-- =============================================================================
-- V395: PACK-047 GHG Emissions Benchmark Pack - Views, Indexes & Seed Data
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates analytical views, materialized views, additional composite indexes
-- for common query patterns, and seed data. Views provide pre-joined
-- perspectives for dashboards, alignment monitoring, portfolio analytics,
-- and risk scoring. Seed data populates IEA NZE, IPCC AR6, SBTi SDA
-- pathways with sector waypoints, PCAF quality descriptions, GWP conversion
-- factors, and sector benchmark reference values.
--
-- Materialized Views (4):
--   1. ghg_benchmark.gl_bm_v_peer_summary
--   2. ghg_benchmark.gl_bm_v_alignment_dashboard
--   3. ghg_benchmark.gl_bm_v_portfolio_summary
--   4. ghg_benchmark.gl_bm_v_transition_risk_dashboard
--
-- Also includes: additional indexes, seed data (~30 sectors + pathways),
-- grants, comments.
-- Previous: V394__pack047_transition_risk_quality.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Materialized View 1: ghg_benchmark.gl_bm_v_peer_summary
-- =============================================================================
-- Peer group statistical summary: count of active peers, median intensity,
-- P25, P75, min, max from the latest normalised data per peer group.

CREATE MATERIALIZED VIEW ghg_benchmark.gl_bm_v_peer_summary AS
SELECT
    pg.id                       AS peer_group_id,
    pg.tenant_id,
    pg.config_id,
    pg.group_name,
    pg.sector_code,
    pg.sector_system,
    pg.size_band,
    pg.peer_count,
    pg.quality_score            AS group_quality,
    -- Summary statistics from latest normalised data
    COUNT(DISTINCT nd.peer_definition_id) AS normalised_peer_count,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY nd.normalised_intensity) AS median_intensity,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY nd.normalised_intensity) AS p25_intensity,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY nd.normalised_intensity) AS p75_intensity,
    MIN(nd.normalised_intensity) AS min_intensity,
    MAX(nd.normalised_intensity) AS max_intensity,
    AVG(nd.normalised_intensity) AS mean_intensity,
    STDDEV(nd.normalised_intensity) AS stddev_intensity,
    COUNT(CASE WHEN nd.quality_downgrade THEN 1 END) AS downgraded_count,
    NOW()                       AS materialized_at
FROM ghg_benchmark.gl_bm_peer_groups pg
LEFT JOIN ghg_benchmark.gl_bm_normalisation_runs nr
    ON pg.id = nr.peer_group_id
    AND nr.status = 'COMPLETED'
    AND nr.run_at = (
        SELECT MAX(nr2.run_at)
        FROM ghg_benchmark.gl_bm_normalisation_runs nr2
        WHERE nr2.peer_group_id = pg.id
          AND nr2.status = 'COMPLETED'
    )
LEFT JOIN ghg_benchmark.gl_bm_normalised_data nd
    ON nr.id = nd.normalisation_run_id
WHERE pg.is_active = true
GROUP BY pg.id, pg.tenant_id, pg.config_id, pg.group_name,
         pg.sector_code, pg.sector_system, pg.size_band,
         pg.peer_count, pg.quality_score;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p047_vps_pk
    ON ghg_benchmark.gl_bm_v_peer_summary(peer_group_id);
CREATE INDEX idx_p047_vps_tenant
    ON ghg_benchmark.gl_bm_v_peer_summary(tenant_id);
CREATE INDEX idx_p047_vps_config
    ON ghg_benchmark.gl_bm_v_peer_summary(config_id);
CREATE INDEX idx_p047_vps_sector
    ON ghg_benchmark.gl_bm_v_peer_summary(sector_code);

-- =============================================================================
-- Materialized View 2: ghg_benchmark.gl_bm_v_alignment_dashboard
-- =============================================================================
-- Latest alignment results per pathway showing current gap, score, and
-- convergence status with pathway metadata.

CREATE MATERIALIZED VIEW ghg_benchmark.gl_bm_v_alignment_dashboard AS
SELECT
    ar.id                       AS alignment_id,
    ar.tenant_id,
    ar.config_id,
    ar.pathway_id,
    pw.pathway_type,
    pw.pathway_name,
    pw.sector                   AS pathway_sector,
    pw.temperature_target,
    ar.reporting_year,
    ar.org_intensity,
    ar.pathway_intensity,
    ar.gap_absolute,
    ar.gap_percentage,
    ar.years_to_convergence,
    ar.alignment_score,
    ar.overshoot_year,
    ar.status,
    ar.provenance_hash,
    ar.calculated_at,
    NOW()                       AS materialized_at
FROM ghg_benchmark.gl_bm_alignment_results ar
JOIN ghg_benchmark.gl_bm_pathways pw ON ar.pathway_id = pw.id
WHERE ar.calculated_at = (
    SELECT MAX(ar2.calculated_at)
    FROM ghg_benchmark.gl_bm_alignment_results ar2
    WHERE ar2.config_id = ar.config_id
      AND ar2.pathway_id = ar.pathway_id
);

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p047_vad_pk
    ON ghg_benchmark.gl_bm_v_alignment_dashboard(alignment_id);
CREATE INDEX idx_p047_vad_tenant
    ON ghg_benchmark.gl_bm_v_alignment_dashboard(tenant_id);
CREATE INDEX idx_p047_vad_config
    ON ghg_benchmark.gl_bm_v_alignment_dashboard(config_id);
CREATE INDEX idx_p047_vad_pathway_type
    ON ghg_benchmark.gl_bm_v_alignment_dashboard(pathway_type);
CREATE INDEX idx_p047_vad_temp
    ON ghg_benchmark.gl_bm_v_alignment_dashboard(temperature_target);
CREATE INDEX idx_p047_vad_score
    ON ghg_benchmark.gl_bm_v_alignment_dashboard(alignment_score);

-- =============================================================================
-- Materialized View 3: ghg_benchmark.gl_bm_v_portfolio_summary
-- =============================================================================
-- Latest portfolio benchmark results with portfolio metadata.

CREATE MATERIALIZED VIEW ghg_benchmark.gl_bm_v_portfolio_summary AS
SELECT
    pr.id                       AS result_id,
    pr.tenant_id,
    pf.id                       AS portfolio_id,
    pf.config_id,
    pf.portfolio_name,
    pf.portfolio_type,
    pf.aum,
    pf.currency,
    pf.holding_count,
    pf.coverage_pct             AS portfolio_coverage,
    pf.pcaf_quality             AS portfolio_pcaf,
    pr.benchmark_index,
    pr.waci,
    pr.carbon_footprint,
    pr.carbon_intensity,
    pr.total_financed_emissions,
    pr.tracking_error,
    pr.coverage_pct             AS result_coverage,
    pr.pcaf_quality             AS result_pcaf,
    pr.yoy_change_pct,
    pr.sector_attribution,
    pr.top_contributors,
    pr.status,
    pr.provenance_hash,
    pr.calculated_at,
    NOW()                       AS materialized_at
FROM ghg_benchmark.gl_bm_portfolio_results pr
JOIN ghg_benchmark.gl_bm_portfolios pf ON pr.portfolio_id = pf.id
WHERE pf.is_active = true
  AND pr.calculated_at = (
      SELECT MAX(pr2.calculated_at)
      FROM ghg_benchmark.gl_bm_portfolio_results pr2
      WHERE pr2.portfolio_id = pr.portfolio_id
  );

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p047_vps2_pk
    ON ghg_benchmark.gl_bm_v_portfolio_summary(result_id);
CREATE INDEX idx_p047_vps2_tenant
    ON ghg_benchmark.gl_bm_v_portfolio_summary(tenant_id);
CREATE INDEX idx_p047_vps2_portfolio
    ON ghg_benchmark.gl_bm_v_portfolio_summary(portfolio_id);
CREATE INDEX idx_p047_vps2_config
    ON ghg_benchmark.gl_bm_v_portfolio_summary(config_id);
CREATE INDEX idx_p047_vps2_waci
    ON ghg_benchmark.gl_bm_v_portfolio_summary(waci);

-- =============================================================================
-- Materialized View 4: ghg_benchmark.gl_bm_v_transition_risk_dashboard
-- =============================================================================
-- Latest transition risk scores per entity with risk level classification.

CREATE MATERIALIZED VIEW ghg_benchmark.gl_bm_v_transition_risk_dashboard AS
SELECT
    tr.id                       AS risk_id,
    tr.tenant_id,
    tr.config_id,
    tr.entity_name,
    tr.entity_identifier,
    tr.composite_score,
    tr.carbon_budget_score,
    tr.stranding_score,
    tr.regulatory_score,
    tr.competitive_score,
    tr.financial_score,
    tr.stranding_year,
    tr.carbon_price_exposure,
    tr.overshoot_probability,
    tr.risk_trajectory,
    tr.risk_level,
    tr.provenance_hash,
    tr.calculated_at,
    NOW()                       AS materialized_at
FROM ghg_benchmark.gl_bm_transition_risk tr
WHERE tr.calculated_at = (
    SELECT MAX(tr2.calculated_at)
    FROM ghg_benchmark.gl_bm_transition_risk tr2
    WHERE tr2.config_id = tr.config_id
      AND tr2.entity_name = tr.entity_name
);

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p047_vtrd_pk
    ON ghg_benchmark.gl_bm_v_transition_risk_dashboard(risk_id);
CREATE INDEX idx_p047_vtrd_tenant
    ON ghg_benchmark.gl_bm_v_transition_risk_dashboard(tenant_id);
CREATE INDEX idx_p047_vtrd_config
    ON ghg_benchmark.gl_bm_v_transition_risk_dashboard(config_id);
CREATE INDEX idx_p047_vtrd_entity
    ON ghg_benchmark.gl_bm_v_transition_risk_dashboard(entity_name);
CREATE INDEX idx_p047_vtrd_composite
    ON ghg_benchmark.gl_bm_v_transition_risk_dashboard(composite_score);
CREATE INDEX idx_p047_vtrd_level
    ON ghg_benchmark.gl_bm_v_transition_risk_dashboard(risk_level);
CREATE INDEX idx_p047_vtrd_trajectory
    ON ghg_benchmark.gl_bm_v_transition_risk_dashboard(risk_trajectory);

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Cross-table: peer data by group + year + verification
CREATE INDEX idx_p047_pdata_grp_yr_verif
    ON ghg_benchmark.gl_bm_peer_data(peer_definition_id, reporting_year, verification_status);

-- Cross-table: alignment by config + year for time series
CREATE INDEX idx_p047_ar_config_year
    ON ghg_benchmark.gl_bm_alignment_results(config_id, reporting_year);

-- Cross-table: ITR by config + method + temperature for distribution
CREATE INDEX idx_p047_itr_cfg_method_temp
    ON ghg_benchmark.gl_bm_itr_calculations(config_id, method, implied_temperature);

-- Cross-table: trajectory comparisons by peer group + gap trend
CREATE INDEX idx_p047_tc_pg_trend
    ON ghg_benchmark.gl_bm_trajectory_comparisons(peer_group_id, gap_trend);

-- Cross-table: holdings by portfolio + asset class + pcaf for PCAF reporting
CREATE INDEX idx_p047_hd_port_class_pcaf
    ON ghg_benchmark.gl_bm_holdings(portfolio_id, asset_class, pcaf_score);

-- Cross-table: transition risk by config + trajectory + level
CREATE INDEX idx_p047_trs_cfg_traj_level
    ON ghg_benchmark.gl_bm_transition_risk(config_id, risk_trajectory, risk_level);

-- Cross-table: data quality by config + composite for improvement prioritisation
CREATE INDEX idx_p047_dqs_cfg_composite
    ON ghg_benchmark.gl_bm_data_quality_scores(config_id, composite_score DESC);

-- Cross-table: external data by source + sector + year for sector aggregation
CREATE INDEX idx_p047_ed_src_sector_year
    ON ghg_benchmark.gl_bm_external_data(source_id, sector_code, reporting_year);

-- =============================================================================
-- Seed Data: IEA NZE Pathway Waypoints (Power Sector)
-- =============================================================================
-- IEA Net Zero Emissions by 2050 scenario - Power sector intensity pathway.
-- Source: IEA World Energy Outlook 2023, Net Zero by 2050 Roadmap (2021 update).
-- Values in gCO2/kWh for electricity generation.

-- Note: Pathway and waypoint seed data requires tenant context. These INSERT
-- statements use a placeholder approach that should be executed within
-- application migration context. The reference data below documents the
-- canonical pathway values for PACK-047.

-- =============================================================================
-- Seed Data: PCAF Quality Score Descriptions
-- =============================================================================

CREATE TABLE IF NOT EXISTS ghg_benchmark.gl_bm_pcaf_quality_reference (
    score                       INTEGER         PRIMARY KEY,
    name                        VARCHAR(50)     NOT NULL,
    description                 TEXT            NOT NULL,
    data_source                 TEXT            NOT NULL,
    uncertainty_range           VARCHAR(30)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

INSERT INTO ghg_benchmark.gl_bm_pcaf_quality_reference (score, name, description, data_source, uncertainty_range)
VALUES
(1, 'Audited/Verified', 'Verified emissions data from the borrower or investee company. Third-party assurance (limited or reasonable).', 'Direct measurement, continuous monitoring, verified GHG inventory', '<10%'),
(2, 'Reported', 'Reported emissions from the borrower or investee company, not yet verified. Self-reported through CDP, annual reports, or direct engagement.', 'Company-reported GHG inventory, CDP questionnaire response', '10-20%'),
(3, 'Physical Activity', 'Emissions estimated using physical activity data (e.g., energy consumption, production output) and emission factors.', 'Physical activity data (kWh, GJ, tonnes) with region-specific emission factors', '20-40%'),
(4, 'Revenue-Based', 'Emissions estimated using economic activity data (e.g., revenue, assets) and sector-average emission factors.', 'Revenue or asset values with sector-average EEIO emission factors', '40-60%'),
(5, 'Estimated', 'Emissions estimated using broad sector averages or proxy data with no entity-specific information.', 'Sector average emissions, proxy from comparable entities', '>60%');

-- =============================================================================
-- Seed Data: GWP Conversion Factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS ghg_benchmark.gl_bm_gwp_conversion_factors (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    gas_name                    VARCHAR(50)     NOT NULL,
    gas_formula                 VARCHAR(30)     NOT NULL,
    ar4_gwp100                  NUMERIC(10,2)   NOT NULL,
    ar5_gwp100                  NUMERIC(10,2)   NOT NULL,
    ar6_gwp100                  NUMERIC(10,2)   NOT NULL,
    ar4_to_ar5_factor           NUMERIC(10,6)   NOT NULL,
    ar4_to_ar6_factor           NUMERIC(10,6)   NOT NULL,
    ar5_to_ar6_factor           NUMERIC(10,6)   NOT NULL,
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

INSERT INTO ghg_benchmark.gl_bm_gwp_conversion_factors (gas_name, gas_formula, ar4_gwp100, ar5_gwp100, ar6_gwp100, ar4_to_ar5_factor, ar4_to_ar6_factor, ar5_to_ar6_factor, notes)
VALUES
('Carbon Dioxide', 'CO2', 1, 1, 1, 1.000000, 1.000000, 1.000000, 'Reference gas, GWP always 1'),
('Methane (fossil)', 'CH4', 25, 28, 29.8, 1.120000, 1.192000, 1.064286, 'AR6 includes climate-carbon feedback for fossil CH4'),
('Methane (biogenic)', 'CH4-bio', 25, 28, 27.0, 1.120000, 1.080000, 0.964286, 'AR6 separates biogenic methane (lower GWP)'),
('Nitrous Oxide', 'N2O', 298, 265, 273, 0.889262, 0.916107, 1.030189, 'AR6 GWP updated from latest atmospheric chemistry'),
('HFC-134a', 'CF3CH2F', 1430, 1300, 1526, 0.909091, 1.067133, 1.173846, 'Common refrigerant, GWP revised upward in AR6'),
('HFC-32', 'CH2F2', 675, 677, 771, 1.002963, 1.142222, 1.138847, 'Low-GWP alternative refrigerant'),
('Sulfur Hexafluoride', 'SF6', 22800, 23500, 25200, 1.030702, 1.105263, 1.072340, 'Electrical insulation gas, highest GWP'),
('Nitrogen Trifluoride', 'NF3', 17200, 16100, 17400, 0.936047, 1.011628, 1.080745, 'Semiconductor manufacturing gas'),
('PFC-14', 'CF4', 7390, 6630, 7380, 0.897159, 0.998647, 1.113122, 'Aluminium smelting byproduct'),
('PFC-116', 'C2F6', 12200, 11100, 12400, 0.909836, 1.016393, 1.117117, 'Aluminium smelting byproduct');

-- =============================================================================
-- Seed Data: Sector Benchmark Reference Values
-- =============================================================================
-- Reference intensity values for 30+ sectors based on publicly available
-- data from IEA, TPI, CDP, and SBTi. These serve as baseline benchmarks
-- for organisations that lack sufficient peer data.

CREATE TABLE IF NOT EXISTS ghg_benchmark.gl_bm_sector_benchmarks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sector_code                 VARCHAR(50)     NOT NULL,
    sector_name                 VARCHAR(255)    NOT NULL,
    intensity_unit              VARCHAR(100)    NOT NULL,
    source                      VARCHAR(100)    NOT NULL,
    source_year                 INTEGER         NOT NULL,
    -- Percentile distribution
    p10_intensity               NUMERIC(20,10),
    p25_intensity               NUMERIC(20,10),
    median_intensity            NUMERIC(20,10)  NOT NULL,
    mean_intensity              NUMERIC(20,10),
    p75_intensity               NUMERIC(20,10),
    p90_intensity               NUMERIC(20,10),
    best_in_class               NUMERIC(20,10),
    -- Paris-aligned targets
    paris_2030_target           NUMERIC(20,10),
    paris_2050_target           NUMERIC(20,10),
    -- Metadata
    peer_count                  INTEGER,
    geographic_scope            VARCHAR(50)     NOT NULL DEFAULT 'GLOBAL',
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

INSERT INTO ghg_benchmark.gl_bm_sector_benchmarks (
    sector_code, sector_name, intensity_unit, source, source_year,
    p10_intensity, p25_intensity, median_intensity, mean_intensity,
    p75_intensity, p90_intensity, best_in_class,
    paris_2030_target, paris_2050_target, peer_count, geographic_scope, notes
) VALUES
-- Power Generation
('POWER_GENERATION', 'Power Generation', 'gCO2/kWh', 'IEA_WEO_2023', 2023,
 30.0, 120.0, 350.0, 420.0, 600.0, 850.0, 5.0,
 138.0, 0.0, 2500, 'GLOBAL', 'IEA NZE scenario: 138 gCO2/kWh by 2030, near-zero by 2050'),
-- Cement
('CEMENT', 'Cement Production', 'kgCO2/t clinker', 'TPI_2024', 2024,
 550.0, 620.0, 680.0, 700.0, 780.0, 880.0, 480.0,
 520.0, 130.0, 85, 'GLOBAL', 'TPI benchmarks for cement sector. SBTi SDA pathway target'),
-- Steel
('STEEL', 'Steel Production', 'tCO2/t crude steel', 'TPI_2024', 2024,
 0.4, 0.8, 1.4, 1.6, 2.0, 2.6, 0.2,
 1.0, 0.14, 60, 'GLOBAL', 'TPI benchmarks for steel. Includes BF-BOF and EAF producers'),
-- Aluminium
('ALUMINIUM', 'Aluminium Production', 'tCO2/t aluminium', 'IEA_2023', 2023,
 2.0, 4.0, 8.0, 9.5, 14.0, 18.0, 1.5,
 5.6, 1.3, 40, 'GLOBAL', 'Primary aluminium. Wide range due to electricity source variation'),
-- Pulp & Paper
('PULP_PAPER', 'Pulp & Paper', 'tCO2/t paper', 'SBTi_SDA', 2023,
 0.15, 0.25, 0.45, 0.55, 0.75, 1.10, 0.08,
 0.32, 0.10, 55, 'GLOBAL', 'SBTi SDA sector pathway for pulp and paper'),
-- Real Estate (Commercial)
('REAL_ESTATE', 'Commercial Real Estate', 'kgCO2/m2', 'CRREM_2024', 2024,
 15.0, 30.0, 55.0, 65.0, 90.0, 140.0, 5.0,
 30.0, 5.0, 1200, 'EU', 'CRREM decarbonisation pathways for EU commercial buildings'),
-- Transport (Freight)
('FREIGHT_TRANSPORT', 'Freight Transport', 'gCO2/tkm', 'GLEC_2023', 2023,
 5.0, 15.0, 45.0, 62.0, 90.0, 150.0, 2.0,
 25.0, 5.0, 300, 'GLOBAL', 'GLEC Framework. Wide range: rail (5) to air freight (600)'),
-- Transport (Passenger)
('PASSENGER_TRANSPORT', 'Passenger Transport', 'gCO2/pkm', 'SBTi_SDA', 2023,
 3.0, 20.0, 60.0, 85.0, 120.0, 200.0, 1.0,
 35.0, 5.0, 200, 'GLOBAL', 'SBTi SDA for passenger transport. Aviation and road combined'),
-- Banking
('BANKING', 'Banking (Financed Emissions)', 'tCO2e/MEUR lent', 'PCAF_2024', 2024,
 10.0, 30.0, 75.0, 100.0, 150.0, 250.0, 5.0,
 45.0, 10.0, 150, 'GLOBAL', 'PCAF methodology. Financed emissions per EUR million lending'),
-- Asset Management
('ASSET_MANAGEMENT', 'Asset Management (WACI)', 'tCO2e/MEUR revenue', 'TCFD_2024', 2024,
 20.0, 50.0, 120.0, 160.0, 230.0, 400.0, 8.0,
 70.0, 15.0, 250, 'GLOBAL', 'WACI for diversified equity portfolios per TCFD methodology'),
-- Insurance
('INSURANCE', 'Insurance (Underwriting)', 'tCO2e/MEUR premiums', 'PCAF_2024', 2024,
 15.0, 40.0, 90.0, 120.0, 180.0, 300.0, 8.0,
 55.0, 12.0, 80, 'GLOBAL', 'PCAF for insurance underwriting portfolios'),
-- Oil & Gas
('OIL_GAS', 'Oil & Gas', 'kgCO2e/boe', 'TPI_2024', 2024,
 25.0, 35.0, 50.0, 55.0, 70.0, 95.0, 15.0,
 30.0, 5.0, 100, 'GLOBAL', 'TPI upstream Scope 1+2. Excludes Scope 3 combustion'),
-- Chemicals
('CHEMICALS', 'Chemical Production', 'tCO2/t product', 'SBTi_SDA', 2023,
 0.3, 0.6, 1.2, 1.5, 2.2, 3.5, 0.15,
 0.8, 0.2, 70, 'GLOBAL', 'SBTi SDA for diversified chemicals. Product-weighted'),
-- Mining
('MINING', 'Mining Operations', 'tCO2e/MEUR revenue', 'CDP_2024', 2024,
 100.0, 200.0, 400.0, 550.0, 750.0, 1200.0, 50.0,
 250.0, 60.0, 120, 'GLOBAL', 'CDP-reported. Scope 1+2 intensity. Highly variable by commodity'),
-- Food & Beverage
('FOOD', 'Food Production', 'tCO2e/t product', 'CDP_2024', 2024,
 0.2, 0.5, 1.0, 1.3, 1.8, 3.0, 0.08,
 0.7, 0.15, 180, 'GLOBAL', 'CDP questionnaire responses. Includes agricultural supply chain'),
-- Hospitality
('HOSPITALITY', 'Hospitality', 'kgCO2e/room-night', 'HCMI_2024', 2024,
 5.0, 12.0, 25.0, 30.0, 45.0, 70.0, 3.0,
 15.0, 3.0, 200, 'GLOBAL', 'Hotel Carbon Measurement Initiative. Per occupied room-night'),
-- Data Center
('DATA_CENTER', 'Data Centers', 'kgCO2/MWh IT load', 'CDP_2024', 2024,
 10.0, 40.0, 150.0, 200.0, 350.0, 500.0, 2.0,
 60.0, 5.0, 100, 'GLOBAL', 'Scope 2 per IT load. PUE-adjusted. Renewable procurement reduces significantly'),
-- Healthcare
('HEALTHCARE', 'Healthcare', 'kgCO2e/bed-day', 'NHS_2023', 2023,
 15.0, 30.0, 55.0, 65.0, 90.0, 130.0, 8.0,
 35.0, 8.0, 80, 'EU', 'NHS England benchmarks adapted for EU healthcare. Per occupied bed-day'),
-- Retail
('RETAIL', 'Retail Operations', 'kgCO2e/m2', 'GRESB_2024', 2024,
 20.0, 40.0, 70.0, 85.0, 120.0, 180.0, 10.0,
 45.0, 8.0, 300, 'GLOBAL', 'GRESB retail sector. Per m2 GLA. Includes refrigerant leakage'),
-- Manufacturing (General)
('MANUFACTURING', 'General Manufacturing', 'tCO2e/MEUR revenue', 'CDP_2024', 2024,
 20.0, 60.0, 150.0, 200.0, 350.0, 600.0, 10.0,
 90.0, 20.0, 500, 'GLOBAL', 'CDP-reported across manufacturing sub-sectors'),
-- Agriculture
('AGRICULTURE', 'Agriculture', 'tCO2e/ha', 'FAO_2023', 2023,
 0.5, 1.5, 3.5, 4.5, 6.5, 10.0, 0.2,
 2.5, 0.5, 150, 'GLOBAL', 'FAO GLEAM methodology. Includes land use, livestock, and crop emissions'),
-- Forestry
('FORESTRY', 'Forestry & Logging', 'tCO2e/ha', 'FAO_2023', 2023,
 -5.0, -2.0, 0.5, 1.0, 3.0, 6.0, -10.0,
 -1.0, -3.0, 50, 'GLOBAL', 'Net emissions including sequestration. Negative = carbon sink'),
-- Utilities
('UTILITIES', 'Utilities (Multi)', 'tCO2e/MEUR revenue', 'CDP_2024', 2024,
 50.0, 150.0, 350.0, 450.0, 700.0, 1100.0, 20.0,
 200.0, 40.0, 200, 'GLOBAL', 'Combined electricity, gas, water utilities. Scope 1+2'),
-- Services
('SERVICES', 'Professional Services', 'tCO2e/FTE', 'CDP_2024', 2024,
 0.5, 1.0, 2.5, 3.5, 5.0, 8.0, 0.2,
 1.5, 0.3, 300, 'GLOBAL', 'Office-based services. Scope 1+2 per FTE. Travel in Scope 3'),
-- Office
('OFFICE', 'Office Buildings', 'kgCO2/m2', 'CRREM_2024', 2024,
 10.0, 20.0, 40.0, 50.0, 70.0, 110.0, 3.0,
 22.0, 3.0, 800, 'EU', 'CRREM office pathway for EU. Per m2 net lettable area'),
-- ICT
('ICT', 'Information & Communication Technology', 'tCO2e/MEUR revenue', 'CDP_2024', 2024,
 5.0, 15.0, 40.0, 55.0, 80.0, 150.0, 2.0,
 25.0, 5.0, 200, 'GLOBAL', 'ICT sector including hardware, software, telecoms. Revenue-based'),
-- Beverage
('BEVERAGE', 'Beverage Production', 'kgCO2e/hectolitre', 'CDP_2024', 2024,
 5.0, 10.0, 20.0, 25.0, 35.0, 55.0, 2.0,
 12.0, 3.0, 80, 'GLOBAL', 'Brewing and soft drinks. Per hectolitre of packaged product'),
-- Commercial (Buildings)
('COMMERCIAL', 'Commercial Buildings', 'kgCO2/m2', 'CRREM_2024', 2024,
 12.0, 25.0, 50.0, 60.0, 85.0, 130.0, 4.0,
 28.0, 4.0, 600, 'EU', 'CRREM commercial mixed-use. Per m2 GLA');

-- =============================================================================
-- Seed Data: IEA NZE Global Pathway Reference
-- =============================================================================
-- These reference values document the IEA NZE global CO2 emissions pathway
-- (GtCO2) for seed into pathway_waypoints via application migration context.

CREATE TABLE IF NOT EXISTS ghg_benchmark.gl_bm_pathway_reference (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    pathway_type                VARCHAR(30)     NOT NULL,
    pathway_name                VARCHAR(255)    NOT NULL,
    sector                      VARCHAR(100),
    temperature_target          NUMERIC(3,1)    NOT NULL,
    year                        INTEGER         NOT NULL,
    intensity_value             NUMERIC(20,10),
    emissions_value_gt          NUMERIC(10,3),
    unit                        VARCHAR(50)     NOT NULL,
    source_reference            TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

INSERT INTO ghg_benchmark.gl_bm_pathway_reference (
    pathway_type, pathway_name, sector, temperature_target, year,
    intensity_value, emissions_value_gt, unit, source_reference
) VALUES
-- IEA NZE - Power Sector (gCO2/kWh)
('IEA_NZE', 'IEA NZE 2050 - Power', 'POWER_GENERATION', 1.5, 2020, 460.0, NULL, 'gCO2/kWh', 'IEA Net Zero by 2050 Roadmap, 2021'),
('IEA_NZE', 'IEA NZE 2050 - Power', 'POWER_GENERATION', 1.5, 2025, 340.0, NULL, 'gCO2/kWh', 'IEA Net Zero by 2050 Roadmap, 2021'),
('IEA_NZE', 'IEA NZE 2050 - Power', 'POWER_GENERATION', 1.5, 2030, 138.0, NULL, 'gCO2/kWh', 'IEA Net Zero by 2050 Roadmap, 2021'),
('IEA_NZE', 'IEA NZE 2050 - Power', 'POWER_GENERATION', 1.5, 2035, 50.0, NULL, 'gCO2/kWh', 'IEA Net Zero by 2050 Roadmap, 2021'),
('IEA_NZE', 'IEA NZE 2050 - Power', 'POWER_GENERATION', 1.5, 2040, 15.0, NULL, 'gCO2/kWh', 'IEA Net Zero by 2050 Roadmap, 2021'),
('IEA_NZE', 'IEA NZE 2050 - Power', 'POWER_GENERATION', 1.5, 2045, 3.0, NULL, 'gCO2/kWh', 'IEA Net Zero by 2050 Roadmap, 2021'),
('IEA_NZE', 'IEA NZE 2050 - Power', 'POWER_GENERATION', 1.5, 2050, 0.0, NULL, 'gCO2/kWh', 'IEA Net Zero by 2050 Roadmap, 2021'),
-- IEA NZE - Industry (tCO2/t crude steel equivalent)
('IEA_NZE', 'IEA NZE 2050 - Steel', 'STEEL', 1.5, 2020, 1.40, NULL, 'tCO2/t steel', 'IEA Iron and Steel Technology Roadmap, 2020'),
('IEA_NZE', 'IEA NZE 2050 - Steel', 'STEEL', 1.5, 2025, 1.25, NULL, 'tCO2/t steel', 'IEA Iron and Steel Technology Roadmap, 2020'),
('IEA_NZE', 'IEA NZE 2050 - Steel', 'STEEL', 1.5, 2030, 1.00, NULL, 'tCO2/t steel', 'IEA Iron and Steel Technology Roadmap, 2020'),
('IEA_NZE', 'IEA NZE 2050 - Steel', 'STEEL', 1.5, 2035, 0.65, NULL, 'tCO2/t steel', 'IEA Iron and Steel Technology Roadmap, 2020'),
('IEA_NZE', 'IEA NZE 2050 - Steel', 'STEEL', 1.5, 2040, 0.35, NULL, 'tCO2/t steel', 'IEA Iron and Steel Technology Roadmap, 2020'),
('IEA_NZE', 'IEA NZE 2050 - Steel', 'STEEL', 1.5, 2050, 0.14, NULL, 'tCO2/t steel', 'IEA Iron and Steel Technology Roadmap, 2020'),
-- IEA NZE - Cement (kgCO2/t clinker)
('IEA_NZE', 'IEA NZE 2050 - Cement', 'CEMENT', 1.5, 2020, 680.0, NULL, 'kgCO2/t clinker', 'IEA Cement Technology Roadmap, 2018 update'),
('IEA_NZE', 'IEA NZE 2050 - Cement', 'CEMENT', 1.5, 2025, 640.0, NULL, 'kgCO2/t clinker', 'IEA Cement Technology Roadmap, 2018 update'),
('IEA_NZE', 'IEA NZE 2050 - Cement', 'CEMENT', 1.5, 2030, 520.0, NULL, 'kgCO2/t clinker', 'IEA Cement Technology Roadmap, 2018 update'),
('IEA_NZE', 'IEA NZE 2050 - Cement', 'CEMENT', 1.5, 2035, 380.0, NULL, 'kgCO2/t clinker', 'IEA Cement Technology Roadmap, 2018 update'),
('IEA_NZE', 'IEA NZE 2050 - Cement', 'CEMENT', 1.5, 2040, 250.0, NULL, 'kgCO2/t clinker', 'IEA Cement Technology Roadmap, 2018 update'),
('IEA_NZE', 'IEA NZE 2050 - Cement', 'CEMENT', 1.5, 2050, 130.0, NULL, 'kgCO2/t clinker', 'IEA Cement Technology Roadmap, 2018 update'),
-- IEA NZE - Transport (gCO2/pkm for passenger)
('IEA_NZE', 'IEA NZE 2050 - Passenger Transport', 'PASSENGER_TRANSPORT', 1.5, 2020, 85.0, NULL, 'gCO2/pkm', 'IEA Transport Outlook, 2023'),
('IEA_NZE', 'IEA NZE 2050 - Passenger Transport', 'PASSENGER_TRANSPORT', 1.5, 2025, 70.0, NULL, 'gCO2/pkm', 'IEA Transport Outlook, 2023'),
('IEA_NZE', 'IEA NZE 2050 - Passenger Transport', 'PASSENGER_TRANSPORT', 1.5, 2030, 45.0, NULL, 'gCO2/pkm', 'IEA Transport Outlook, 2023'),
('IEA_NZE', 'IEA NZE 2050 - Passenger Transport', 'PASSENGER_TRANSPORT', 1.5, 2040, 15.0, NULL, 'gCO2/pkm', 'IEA Transport Outlook, 2023'),
('IEA_NZE', 'IEA NZE 2050 - Passenger Transport', 'PASSENGER_TRANSPORT', 1.5, 2050, 3.0, NULL, 'gCO2/pkm', 'IEA Transport Outlook, 2023'),
-- IEA NZE - Buildings (kgCO2/m2)
('IEA_NZE', 'IEA NZE 2050 - Buildings', 'REAL_ESTATE', 1.5, 2020, 55.0, NULL, 'kgCO2/m2', 'IEA Buildings Outlook, 2023'),
('IEA_NZE', 'IEA NZE 2050 - Buildings', 'REAL_ESTATE', 1.5, 2025, 45.0, NULL, 'kgCO2/m2', 'IEA Buildings Outlook, 2023'),
('IEA_NZE', 'IEA NZE 2050 - Buildings', 'REAL_ESTATE', 1.5, 2030, 30.0, NULL, 'kgCO2/m2', 'IEA Buildings Outlook, 2023'),
('IEA_NZE', 'IEA NZE 2050 - Buildings', 'REAL_ESTATE', 1.5, 2040, 10.0, NULL, 'kgCO2/m2', 'IEA Buildings Outlook, 2023'),
('IEA_NZE', 'IEA NZE 2050 - Buildings', 'REAL_ESTATE', 1.5, 2050, 2.0, NULL, 'kgCO2/m2', 'IEA Buildings Outlook, 2023'),
-- IPCC AR6 C1 - Global (GtCO2)
('IPCC_AR6_C1', 'IPCC AR6 C1 (1.5C no/limited overshoot)', NULL, 1.5, 2020, NULL, 36.4, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C1', 'IPCC AR6 C1 (1.5C no/limited overshoot)', NULL, 1.5, 2025, NULL, 31.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C1', 'IPCC AR6 C1 (1.5C no/limited overshoot)', NULL, 1.5, 2030, NULL, 21.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C1', 'IPCC AR6 C1 (1.5C no/limited overshoot)', NULL, 1.5, 2035, NULL, 13.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C1', 'IPCC AR6 C1 (1.5C no/limited overshoot)', NULL, 1.5, 2040, NULL, 8.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C1', 'IPCC AR6 C1 (1.5C no/limited overshoot)', NULL, 1.5, 2050, NULL, 0.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
-- IPCC AR6 C2 - Global (GtCO2)
('IPCC_AR6_C2', 'IPCC AR6 C2 (1.5C high overshoot)', NULL, 1.5, 2020, NULL, 36.4, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C2', 'IPCC AR6 C2 (1.5C high overshoot)', NULL, 1.5, 2030, NULL, 25.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C2', 'IPCC AR6 C2 (1.5C high overshoot)', NULL, 1.5, 2040, NULL, 12.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C2', 'IPCC AR6 C2 (1.5C high overshoot)', NULL, 1.5, 2050, NULL, -2.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1. Negative = net removal'),
-- IPCC AR6 C3 - Global (GtCO2)
('IPCC_AR6_C3', 'IPCC AR6 C3 (2.0C)', NULL, 2.0, 2020, NULL, 36.4, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C3', 'IPCC AR6 C3 (2.0C)', NULL, 2.0, 2030, NULL, 30.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C3', 'IPCC AR6 C3 (2.0C)', NULL, 2.0, 2040, NULL, 20.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
('IPCC_AR6_C3', 'IPCC AR6 C3 (2.0C)', NULL, 2.0, 2050, NULL, 10.0, 'GtCO2', 'IPCC AR6 WGIII SPM, Table SPM.1'),
-- SBTi SDA - Power
('SBTI_SDA', 'SBTi SDA - Power Generation', 'POWER_GENERATION', 1.5, 2020, 460.0, NULL, 'gCO2/kWh', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Power Generation', 'POWER_GENERATION', 1.5, 2025, 338.0, NULL, 'gCO2/kWh', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Power Generation', 'POWER_GENERATION', 1.5, 2030, 136.0, NULL, 'gCO2/kWh', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Power Generation', 'POWER_GENERATION', 1.5, 2035, 50.0, NULL, 'gCO2/kWh', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Power Generation', 'POWER_GENERATION', 1.5, 2050, 0.0, NULL, 'gCO2/kWh', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
-- SBTi SDA - Cement
('SBTI_SDA', 'SBTi SDA - Cement', 'CEMENT', 1.5, 2020, 680.0, NULL, 'kgCO2/t clinker', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Cement', 'CEMENT', 1.5, 2030, 520.0, NULL, 'kgCO2/t clinker', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Cement', 'CEMENT', 1.5, 2050, 130.0, NULL, 'kgCO2/t clinker', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
-- SBTi SDA - Steel
('SBTI_SDA', 'SBTi SDA - Steel', 'STEEL', 1.5, 2020, 1.40, NULL, 'tCO2/t steel', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Steel', 'STEEL', 1.5, 2030, 1.00, NULL, 'tCO2/t steel', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Steel', 'STEEL', 1.5, 2050, 0.14, NULL, 'tCO2/t steel', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
-- SBTi SDA - Aluminium
('SBTI_SDA', 'SBTi SDA - Aluminium', 'ALUMINIUM', 1.5, 2020, 8.00, NULL, 'tCO2/t aluminium', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Aluminium', 'ALUMINIUM', 1.5, 2030, 5.60, NULL, 'tCO2/t aluminium', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Aluminium', 'ALUMINIUM', 1.5, 2050, 1.30, NULL, 'tCO2/t aluminium', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
-- SBTi SDA - Pulp & Paper
('SBTI_SDA', 'SBTi SDA - Pulp & Paper', 'PULP_PAPER', 1.5, 2020, 0.45, NULL, 'tCO2/t paper', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Pulp & Paper', 'PULP_PAPER', 1.5, 2030, 0.32, NULL, 'tCO2/t paper', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Pulp & Paper', 'PULP_PAPER', 1.5, 2050, 0.10, NULL, 'tCO2/t paper', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
-- SBTi SDA - Passenger Transport
('SBTI_SDA', 'SBTi SDA - Passenger Transport', 'PASSENGER_TRANSPORT', 1.5, 2020, 85.0, NULL, 'gCO2/pkm', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Passenger Transport', 'PASSENGER_TRANSPORT', 1.5, 2030, 45.0, NULL, 'gCO2/pkm', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Passenger Transport', 'PASSENGER_TRANSPORT', 1.5, 2050, 5.0, NULL, 'gCO2/pkm', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
-- SBTi SDA - Freight Transport
('SBTI_SDA', 'SBTi SDA - Freight Transport', 'FREIGHT_TRANSPORT', 1.5, 2020, 45.0, NULL, 'gCO2/tkm', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Freight Transport', 'FREIGHT_TRANSPORT', 1.5, 2030, 25.0, NULL, 'gCO2/tkm', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
('SBTI_SDA', 'SBTi SDA - Freight Transport', 'FREIGHT_TRANSPORT', 1.5, 2050, 5.0, NULL, 'gCO2/tkm', 'SBTi Sectoral Decarbonisation Approach, v2.0'),
-- SBTi SDA - Real Estate (via CRREM)
('SBTI_SDA', 'SBTi SDA - Buildings', 'REAL_ESTATE', 1.5, 2020, 55.0, NULL, 'kgCO2/m2', 'SBTi Sectoral Decarbonisation Approach, v2.0 (CRREM-aligned)'),
('SBTI_SDA', 'SBTi SDA - Buildings', 'REAL_ESTATE', 1.5, 2030, 30.0, NULL, 'kgCO2/m2', 'SBTi Sectoral Decarbonisation Approach, v2.0 (CRREM-aligned)'),
('SBTI_SDA', 'SBTi SDA - Buildings', 'REAL_ESTATE', 1.5, 2050, 5.0, NULL, 'kgCO2/m2', 'SBTi Sectoral Decarbonisation Approach, v2.0 (CRREM-aligned)');

-- =============================================================================
-- Grants for Views and Reference Tables
-- =============================================================================
GRANT SELECT ON ghg_benchmark.gl_bm_v_peer_summary TO PUBLIC;
GRANT SELECT ON ghg_benchmark.gl_bm_v_alignment_dashboard TO PUBLIC;
GRANT SELECT ON ghg_benchmark.gl_bm_v_portfolio_summary TO PUBLIC;
GRANT SELECT ON ghg_benchmark.gl_bm_v_transition_risk_dashboard TO PUBLIC;
GRANT SELECT ON ghg_benchmark.gl_bm_pcaf_quality_reference TO PUBLIC;
GRANT SELECT ON ghg_benchmark.gl_bm_gwp_conversion_factors TO PUBLIC;
GRANT SELECT ON ghg_benchmark.gl_bm_sector_benchmarks TO PUBLIC;
GRANT SELECT ON ghg_benchmark.gl_bm_pathway_reference TO PUBLIC;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON MATERIALIZED VIEW ghg_benchmark.gl_bm_v_peer_summary IS
    'Peer group statistical summary: count, median, P25, P75, min, max intensity from latest normalised data. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';
COMMENT ON MATERIALIZED VIEW ghg_benchmark.gl_bm_v_alignment_dashboard IS
    'Latest alignment results per pathway with gap, score, convergence, and overshoot year. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';
COMMENT ON MATERIALIZED VIEW ghg_benchmark.gl_bm_v_portfolio_summary IS
    'Latest portfolio benchmark results: WACI, footprint, intensity, financed emissions with portfolio metadata. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';
COMMENT ON MATERIALIZED VIEW ghg_benchmark.gl_bm_v_transition_risk_dashboard IS
    'Latest transition risk scores per entity with composite score, sub-dimensions, and risk level. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';

COMMENT ON TABLE ghg_benchmark.gl_bm_pcaf_quality_reference IS
    'PACK-047: PCAF data quality score reference (1-5) with descriptions, data sources, and uncertainty ranges.';
COMMENT ON TABLE ghg_benchmark.gl_bm_gwp_conversion_factors IS
    'PACK-047: GWP conversion factors for AR4, AR5, AR6 with inter-version conversion ratios for major greenhouse gases.';
COMMENT ON TABLE ghg_benchmark.gl_bm_sector_benchmarks IS
    'PACK-047: Reference sector benchmark intensities from IEA, TPI, CDP, GRESB, CRREM, SBTi for 28 sectors with percentile distribution and Paris-aligned targets.';
COMMENT ON TABLE ghg_benchmark.gl_bm_pathway_reference IS
    'PACK-047: Reference pathway waypoints for IEA NZE (4 sectors), IPCC AR6 C1/C2/C3 (global), and SBTi SDA (9 sectors) with source provenance.';

-- Table-level comments for all PACK-047 tables
COMMENT ON TABLE ghg_benchmark.gl_bm_configurations IS
    'PACK-047: Organisation-level benchmark configuration with sector, scope alignment, pathway enablement, and GWP version.';
COMMENT ON TABLE ghg_benchmark.gl_bm_reporting_periods IS
    'PACK-047: Reporting periods for benchmark analyses with DRAFT/ACTIVE/CLOSED lifecycle.';
COMMENT ON TABLE ghg_benchmark.gl_bm_peer_groups IS
    'PACK-047: Peer group definitions with sector classification system, size band, geographic weighting, and quality scoring.';
COMMENT ON TABLE ghg_benchmark.gl_bm_peer_definitions IS
    'PACK-047: Individual peer entities with similarity scoring, outlier detection, and source dataset tracking.';
COMMENT ON TABLE ghg_benchmark.gl_bm_peer_data IS
    'PACK-047: Annual emissions data per peer with scope breakdown, PCAF scoring, and GWP version.';
COMMENT ON TABLE ghg_benchmark.gl_bm_normalisation_runs IS
    'PACK-047: Normalisation run metadata harmonising peer data across scopes, GWP, currencies, and periods.';
COMMENT ON TABLE ghg_benchmark.gl_bm_normalised_data IS
    'PACK-047: Normalised emissions per peer with original/adjusted values and adjustment transparency.';
COMMENT ON TABLE ghg_benchmark.gl_bm_external_sources IS
    'PACK-047: External benchmark data source registry (CDP, TPI, GRESB, CRREM, ISS ESG) with cache TTL.';
COMMENT ON TABLE ghg_benchmark.gl_bm_external_data IS
    'PACK-047: Ingested external benchmark data with entity identification, metrics, and raw JSON preservation.';
COMMENT ON TABLE ghg_benchmark.gl_bm_data_cache IS
    'PACK-047: Cache layer for external API responses with TTL expiry and hit counting.';
COMMENT ON TABLE ghg_benchmark.gl_bm_pathways IS
    'PACK-047: Decarbonisation pathway definitions from IEA NZE, IPCC AR6, SBTi SDA, OECM, TPI CP, CRREM.';
COMMENT ON TABLE ghg_benchmark.gl_bm_pathway_waypoints IS
    'PACK-047: Annual pathway milestones (intensity/emissions) with interpolation flagging.';
COMMENT ON TABLE ghg_benchmark.gl_bm_alignment_results IS
    'PACK-047: Pathway alignment analysis with gap, convergence, alignment score, and overshoot detection.';
COMMENT ON TABLE ghg_benchmark.gl_bm_itr_calculations IS
    'PACK-047: Implied Temperature Rating using budget-based, sector-relative, or rate-of-reduction methods.';
COMMENT ON TABLE ghg_benchmark.gl_bm_itr_portfolio IS
    'PACK-047: Portfolio-level ITR aggregation with coverage and weighted quality tracking.';
COMMENT ON TABLE ghg_benchmark.gl_bm_trajectories IS
    'PACK-047: Entity emissions trajectory with CARR, acceleration, trend direction, and structural breaks.';
COMMENT ON TABLE ghg_benchmark.gl_bm_trajectory_comparisons IS
    'PACK-047: Trajectory ranking against peer group with percentile, convergence rate, and gap trend.';
COMMENT ON TABLE ghg_benchmark.gl_bm_portfolios IS
    'PACK-047: Portfolio definitions for carbon benchmarking with AUM, PCAF coverage, and benchmark index.';
COMMENT ON TABLE ghg_benchmark.gl_bm_holdings IS
    'PACK-047: Individual holdings with PCAF asset class, emissions, ownership share, and data quality.';
COMMENT ON TABLE ghg_benchmark.gl_bm_portfolio_results IS
    'PACK-047: Portfolio benchmark results: WACI, carbon footprint, intensity, financed emissions, sector attribution.';
COMMENT ON TABLE ghg_benchmark.gl_bm_transition_risk IS
    'PACK-047: Transition risk scoring with carbon budget, stranding, regulatory, competitive, and financial sub-dimensions.';
COMMENT ON TABLE ghg_benchmark.gl_bm_data_quality_scores IS
    'PACK-047: Multi-dimensional data quality assessment (temporal, geographic, technological, completeness, reliability).';

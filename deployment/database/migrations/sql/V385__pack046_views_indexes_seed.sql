-- =============================================================================
-- V385: PACK-046 Intensity Metrics Pack - Views, Indexes & Seed Data
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates analytical views, materialized views, additional composite indexes
-- for common query patterns, and seed data. Views provide pre-joined
-- perspectives for dashboards and reporting. Seed data populates the 25
-- standard denominator definitions covering financial, physical, headcount,
-- area, and energy categories across all major reporting frameworks.
--
-- Views (3):
--   1. ghg_intensity.v_intensity_latest
--   2. ghg_intensity.v_intensity_with_targets
--   3. ghg_intensity.v_intensity_benchmark_summary
--
-- Materialized Views (1):
--   4. ghg_intensity.mv_intensity_dashboard
--
-- Also includes: additional indexes, seed data (25 denominators), grants,
-- comments.
-- Previous: V384__pack046_disclosures.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- View 1: ghg_intensity.v_intensity_latest
-- =============================================================================
-- Latest intensity calculation per organisation, denominator, scope, and
-- entity. Uses DISTINCT ON with period_end DESC ordering to select the most
-- recent period's calculation. Joins denominator definitions and reporting
-- period metadata for complete dashboard display.

CREATE OR REPLACE VIEW ghg_intensity.v_intensity_latest AS
SELECT DISTINCT ON (ic.org_id, ic.denominator_id, ic.scope_inclusion, ic.entity_id)
    ic.id                       AS calculation_id,
    ic.tenant_id,
    ic.org_id,
    ic.config_id,
    ic.period_id,
    ic.denominator_id,
    ic.entity_id,
    ic.entity_name,
    ic.scope_inclusion,
    ic.emissions_tco2e,
    ic.denominator_value,
    ic.intensity_value,
    ic.intensity_unit,
    ic.yoy_change_pct,
    ic.data_quality_combined,
    ic.scope_coverage_pct,
    dd.denominator_code,
    dd.name                     AS denominator_name,
    dd.unit                     AS denominator_unit,
    dd.category                 AS denominator_category,
    rp.period_label,
    rp.period_start,
    rp.period_end,
    rp.is_base_year,
    rp.status                   AS period_status,
    ic.provenance_hash,
    ic.calculated_at,
    ic.updated_at
FROM ghg_intensity.gl_im_calculations ic
JOIN ghg_intensity.gl_im_denominator_definitions dd ON ic.denominator_id = dd.id
JOIN ghg_intensity.gl_im_reporting_periods rp ON ic.period_id = rp.id
ORDER BY ic.org_id, ic.denominator_id, ic.scope_inclusion, ic.entity_id, rp.period_end DESC;

-- =============================================================================
-- View 2: ghg_intensity.v_intensity_with_targets
-- =============================================================================
-- Intensity calculations joined with target progress data. Shows current
-- intensity alongside target intensity, variance, and status for each
-- period where both calculation and target tracking exist.

CREATE OR REPLACE VIEW ghg_intensity.v_intensity_with_targets AS
SELECT
    ic.id                       AS calculation_id,
    ic.tenant_id,
    ic.org_id,
    ic.config_id,
    ic.period_id,
    ic.entity_id,
    ic.scope_inclusion,
    ic.emissions_tco2e,
    ic.denominator_value,
    ic.intensity_value,
    ic.intensity_unit,
    ic.yoy_change_pct,
    ic.data_quality_combined,
    dd.denominator_code,
    dd.name                     AS denominator_name,
    rp.period_label,
    rp.period_end,
    -- Target data
    it.id                       AS target_id,
    it.target_name,
    it.target_type,
    it.pathway,
    it.base_year_intensity,
    it.target_intensity          AS final_target_intensity,
    itp.target_intensity         AS period_target_intensity,
    itp.actual_intensity,
    itp.variance,
    itp.variance_pct,
    itp.status                   AS target_status,
    itp.pct_of_target_achieved,
    itp.remaining_annual_rate_pct,
    itp.carbon_budget_remaining,
    ic.provenance_hash,
    ic.calculated_at
FROM ghg_intensity.gl_im_calculations ic
JOIN ghg_intensity.gl_im_denominator_definitions dd ON ic.denominator_id = dd.id
JOIN ghg_intensity.gl_im_reporting_periods rp ON ic.period_id = rp.id
LEFT JOIN ghg_intensity.gl_im_target_progress itp ON ic.period_id = itp.period_id
LEFT JOIN ghg_intensity.gl_im_targets it ON itp.target_id = it.id
    AND it.denominator_code = dd.denominator_code
    AND it.scope_inclusion = ic.scope_inclusion
    AND it.is_active = true;

-- =============================================================================
-- View 3: ghg_intensity.v_intensity_benchmark_summary
-- =============================================================================
-- Benchmark results with peer group context. Shows organisation's intensity,
-- percentile rank, and all peer statistics alongside the peer group
-- metadata (name, sector, peer count).

CREATE OR REPLACE VIEW ghg_intensity.v_intensity_benchmark_summary AS
SELECT
    ibr.id                      AS benchmark_result_id,
    ibr.tenant_id,
    ibr.org_id,
    ibr.config_id,
    ibr.period_id,
    ibr.peer_group_id,
    ibr.denominator_code,
    ibr.scope_inclusion,
    ibr.org_intensity,
    ibr.percentile_rank,
    ibr.peer_mean,
    ibr.peer_median,
    ibr.peer_p10,
    ibr.peer_p25,
    ibr.peer_p75,
    ibr.peer_p90,
    ibr.peer_best,
    ibr.peer_worst,
    ibr.gap_to_average,
    ibr.gap_to_best,
    ibr.gap_to_target,
    ibr.gap_to_average_pct,
    ibr.gap_to_best_pct,
    ipg.group_name,
    ipg.sector,
    ipg.sub_sector,
    ipg.geography,
    ipg.peer_count,
    ipg.data_vintage_year,
    rp.period_label,
    rp.period_end,
    ibr.normalisation_adjustments,
    ibr.provenance_hash,
    ibr.calculated_at
FROM ghg_intensity.gl_im_benchmark_results ibr
JOIN ghg_intensity.gl_im_peer_groups ipg ON ibr.peer_group_id = ipg.id
JOIN ghg_intensity.gl_im_reporting_periods rp ON ibr.period_id = rp.id;

-- =============================================================================
-- Materialized View: ghg_intensity.mv_intensity_dashboard
-- =============================================================================
-- Pre-computed dashboard summary per organisation showing latest intensity
-- metrics across all denominators and scopes, with trend direction,
-- target status, and data quality indicators. Refresh on new calculations.

CREATE MATERIALIZED VIEW ghg_intensity.mv_intensity_dashboard AS
SELECT
    ic.org_id,
    ic.tenant_id,
    ic.config_id,
    dd.denominator_code,
    dd.name                     AS denominator_name,
    dd.category                 AS denominator_category,
    ic.scope_inclusion,
    ic.entity_id,
    ic.entity_name,
    -- Latest values
    ic.intensity_value,
    ic.intensity_unit,
    ic.emissions_tco2e,
    ic.denominator_value,
    ic.yoy_change_pct,
    ic.data_quality_combined,
    -- Period info
    rp.period_label,
    rp.period_end,
    -- Time series trend
    ts.trend_direction,
    ts.carr_pct,
    ts.data_points              AS series_data_points,
    -- Provenance
    ic.provenance_hash,
    ic.calculated_at,
    NOW()                       AS materialized_at
FROM ghg_intensity.gl_im_calculations ic
JOIN ghg_intensity.gl_im_denominator_definitions dd ON ic.denominator_id = dd.id
JOIN ghg_intensity.gl_im_reporting_periods rp ON ic.period_id = rp.id
LEFT JOIN ghg_intensity.gl_im_time_series ts
    ON ic.org_id = ts.org_id
    AND ic.config_id = ts.config_id
    AND dd.denominator_code = ts.denominator_code
    AND ic.scope_inclusion = ts.scope_inclusion
    AND COALESCE(ic.entity_id, '00000000-0000-0000-0000-000000000000'::UUID)
        = COALESCE(ts.entity_id, '00000000-0000-0000-0000-000000000000'::UUID)
WHERE rp.status IN ('CALCULATED', 'VERIFIED', 'PUBLISHED')
  AND ic.calculated_at = (
      SELECT MAX(ic2.calculated_at)
      FROM ghg_intensity.gl_im_calculations ic2
      WHERE ic2.org_id = ic.org_id
        AND ic2.config_id = ic.config_id
        AND ic2.denominator_id = ic.denominator_id
        AND ic2.scope_inclusion = ic.scope_inclusion
        AND COALESCE(ic2.entity_id, '00000000-0000-0000-0000-000000000000'::UUID)
            = COALESCE(ic.entity_id, '00000000-0000-0000-0000-000000000000'::UUID)
  );

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p046_mv_dash_pk
    ON ghg_intensity.mv_intensity_dashboard(org_id, config_id, denominator_code, scope_inclusion, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::UUID));
CREATE INDEX idx_p046_mv_dash_tenant
    ON ghg_intensity.mv_intensity_dashboard(tenant_id);
CREATE INDEX idx_p046_mv_dash_org
    ON ghg_intensity.mv_intensity_dashboard(org_id);
CREATE INDEX idx_p046_mv_dash_denom
    ON ghg_intensity.mv_intensity_dashboard(denominator_code);
CREATE INDEX idx_p046_mv_dash_scope
    ON ghg_intensity.mv_intensity_dashboard(scope_inclusion);
CREATE INDEX idx_p046_mv_dash_trend
    ON ghg_intensity.mv_intensity_dashboard(trend_direction);
CREATE INDEX idx_p046_mv_dash_period
    ON ghg_intensity.mv_intensity_dashboard(period_end DESC);

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Cross-table: disclosure completeness dashboard
CREATE INDEX idx_p046_df_org_status ON ghg_intensity.gl_im_disclosure_frameworks(org_id, status);

-- Cross-table: uncertainty by data quality for improvement prioritisation
CREATE INDEX idx_p046_ua_dq_unc ON ghg_intensity.gl_im_uncertainty_assessments(data_quality_combined, combined_uncertainty_pct DESC);

-- Cross-table: target progress alerts by org
CREATE INDEX idx_p046_prg_org_status ON ghg_intensity.gl_im_target_progress(tenant_id, status)
    WHERE status IN ('AT_RISK', 'OFF_TRACK');

-- Cross-table: peer data aggregation by group and year
CREATE INDEX idx_p046_pd_group_year_denom ON ghg_intensity.gl_im_peer_data(peer_group_id, reporting_year, denominator_code);

-- Cross-table: decomposition trend (org + denominator + scope)
CREATE INDEX idx_p046_dec_org_denom_scope ON ghg_intensity.gl_im_decompositions(org_id, denominator_code, scope_inclusion);

-- =============================================================================
-- Seed Data: 25 Standard Denominator Definitions
-- =============================================================================
-- Standard denominators covering financial, physical, headcount, area, and
-- energy categories. Applicable sectors and framework alignment are pre-
-- configured. Custom denominators can be added per organisation.

INSERT INTO ghg_intensity.gl_im_denominator_definitions (
    denominator_code, name, unit, category, applicable_sectors,
    framework_alignment, description, is_standard, is_active
) VALUES
-- Financial denominators
(
    'DEN-REV-EUR', 'Net Revenue', 'MEUR', 'FINANCIAL',
    '["ALL"]'::JSONB,
    '{"ESRS_E1": "MANDATORY", "CDP": "MANDATORY", "SEC": "OPTIONAL", "GRI": "RECOMMENDED", "IFRS_S2": "MANDATORY"}'::JSONB,
    'Net revenue in millions of euros. Most universal denominator for financial intensity across all sectors.',
    true, true
),
(
    'DEN-AUM-EUR', 'Assets Under Management', 'MEUR', 'FINANCIAL',
    '["ASSET_MANAGEMENT"]'::JSONB,
    '{"PCAF": "MANDATORY", "CDP": "RECOMMENDED", "TCFD": "RECOMMENDED"}'::JSONB,
    'Total assets under management in millions of euros. PCAF-mandated for asset managers.',
    true, true
),
(
    'DEN-LEND-EUR', 'Lending Portfolio', 'MEUR', 'FINANCIAL',
    '["BANKING"]'::JSONB,
    '{"PCAF": "MANDATORY", "CDP": "RECOMMENDED"}'::JSONB,
    'Outstanding lending portfolio value in millions of euros. PCAF-mandated for banks.',
    true, true
),
(
    'DEN-PREM-EUR', 'Insurance Premiums', 'MEUR', 'FINANCIAL',
    '["INSURANCE"]'::JSONB,
    '{"PCAF": "MANDATORY", "CDP": "RECOMMENDED"}'::JSONB,
    'Gross written premiums in millions of euros. PCAF-mandated for insurers.',
    true, true
),
-- Headcount denominator
(
    'DEN-FTE', 'Full-Time Equivalents', 'FTE', 'HEADCOUNT',
    '["SERVICES", "OFFICE", "ALL"]'::JSONB,
    '{"CDP": "RECOMMENDED", "GRI": "RECOMMENDED", "ESRS_E1": "OPTIONAL"}'::JSONB,
    'Full-time equivalent headcount. Common denominator for service and office-based organisations.',
    true, true
),
-- Area denominators
(
    'DEN-GLA-M2', 'Gross Leasable Area', 'm2', 'AREA',
    '["REAL_ESTATE", "COMMERCIAL"]'::JSONB,
    '{"GRESB": "MANDATORY", "CRREM": "MANDATORY", "CDP": "RECOMMENDED"}'::JSONB,
    'Gross leasable area in square metres. GRESB and CRREM mandatory for real estate.',
    true, true
),
(
    'DEN-FLOOR-M2', 'Floor Area', 'm2', 'AREA',
    '["ALL"]'::JSONB,
    '{"ISO_50001": "RECOMMENDED", "ASHRAE": "RECOMMENDED"}'::JSONB,
    'Conditioned floor area in square metres. Universal area metric for building energy intensity.',
    true, true
),
(
    'DEN-HA', 'Hectares', 'ha', 'AREA',
    '["AGRICULTURE", "FORESTRY"]'::JSONB,
    '{"SBTI_FLAG": "MANDATORY", "CDP": "RECOMMENDED"}'::JSONB,
    'Land area in hectares. SBTi FLAG-mandated for agriculture and forestry sectors.',
    true, true
),
(
    'DEN-COOLED-M2', 'Cooled Space', 'm2', 'AREA',
    '["DATA_CENTER", "RETAIL"]'::JSONB,
    '{"SECTOR": "OPTIONAL"}'::JSONB,
    'Cooled floor area in square metres. Data center and cold-chain retail intensity metric.',
    true, true
),
-- Physical production denominators
(
    'DEN-PROD-T', 'Production Output (tonnes)', 't', 'PHYSICAL',
    '["MANUFACTURING", "MINING", "CEMENT", "STEEL"]'::JSONB,
    '{"SBTI_SDA": "MANDATORY", "CDP": "RECOMMENDED", "ISO_14064": "RECOMMENDED"}'::JSONB,
    'Total production output in tonnes. SBTi SDA-mandated for heavy industry.',
    true, true
),
(
    'DEN-UNITS', 'Units Produced', 'units', 'PHYSICAL',
    '["DISCRETE_MANUFACTURING"]'::JSONB,
    '{"ISO_14064": "RECOMMENDED"}'::JSONB,
    'Discrete production units. For discrete manufacturing (automotive, electronics, etc.).',
    true, true
),
(
    'DEN-CLINKER-T', 'Cement (Clinker) Produced', 't clinker', 'PHYSICAL',
    '["CEMENT"]'::JSONB,
    '{"SBTI_SDA": "MANDATORY", "TPI": "MANDATORY"}'::JSONB,
    'Clinker production in tonnes. Sector-specific SBTi SDA and TPI denominator for cement.',
    true, true
),
(
    'DEN-STEEL-T', 'Crude Steel Produced', 't crude steel', 'PHYSICAL',
    '["STEEL"]'::JSONB,
    '{"SBTI_SDA": "MANDATORY", "TPI": "MANDATORY"}'::JSONB,
    'Crude steel production in tonnes. Sector-specific SBTi SDA and TPI denominator for steel.',
    true, true
),
(
    'DEN-ALUM-T', 'Aluminium Produced', 't aluminium', 'PHYSICAL',
    '["ALUMINIUM"]'::JSONB,
    '{"SBTI_SDA": "MANDATORY"}'::JSONB,
    'Primary aluminium production in tonnes. SBTi SDA sector pathway denominator.',
    true, true
),
(
    'DEN-PAPER-T', 'Paper Produced', 't paper', 'PHYSICAL',
    '["PULP_PAPER"]'::JSONB,
    '{"SBTI_SDA": "MANDATORY"}'::JSONB,
    'Paper and paperboard production in tonnes. SBTi SDA sector pathway denominator.',
    true, true
),
(
    'DEN-PROD-FOOD-T', 'Tonnes of Product (Food)', 't', 'PHYSICAL',
    '["FOOD", "BEVERAGE"]'::JSONB,
    '{"SBTI_SDA": "MANDATORY"}'::JSONB,
    'Food and beverage production in tonnes. SBTi SDA sector pathway denominator.',
    true, true
),
-- Transport denominators
(
    'DEN-VKM', 'Vehicle-Kilometres', 'vkm', 'PHYSICAL',
    '["TRANSPORT"]'::JSONB,
    '{"GLEC": "MANDATORY", "CDP": "RECOMMENDED"}'::JSONB,
    'Vehicle-kilometres travelled. GLEC Framework mandatory for general transport.',
    true, true
),
(
    'DEN-TKM', 'Tonne-Kilometres', 'tkm', 'PHYSICAL',
    '["FREIGHT_TRANSPORT"]'::JSONB,
    '{"GLEC": "MANDATORY", "SBTI_SDA": "MANDATORY"}'::JSONB,
    'Tonne-kilometres for freight transport. GLEC and SBTi SDA mandatory for freight.',
    true, true
),
(
    'DEN-PKM', 'Passenger-Kilometres', 'pkm', 'PHYSICAL',
    '["PASSENGER_TRANSPORT"]'::JSONB,
    '{"SBTI_SDA": "MANDATORY", "CDP": "RECOMMENDED"}'::JSONB,
    'Passenger-kilometres for passenger transport. SBTi SDA mandatory for aviation, rail, bus.',
    true, true
),
-- Energy denominator
(
    'DEN-ELEC-MWH', 'Electricity Generated', 'MWh', 'ENERGY',
    '["POWER_GENERATION"]'::JSONB,
    '{"SBTI_SDA": "MANDATORY", "TPI": "MANDATORY", "CDP": "MANDATORY"}'::JSONB,
    'Net electricity generated in MWh. SBTi SDA and TPI mandatory for power generation.',
    true, true
),
-- Sector-specific denominators
(
    'DEN-BED-DAY', 'Bed-Days', 'bed-day', 'PHYSICAL',
    '["HEALTHCARE", "HOSPITALITY"]'::JSONB,
    '{"SECTOR": "RECOMMENDED"}'::JSONB,
    'Occupied bed-days. Healthcare and hospitality sector-specific intensity metric.',
    true, true
),
(
    'DEN-CUSTOMERS', 'Number of Customers', 'customers', 'PHYSICAL',
    '["UTILITIES", "TELECOM"]'::JSONB,
    '{"SECTOR": "OPTIONAL"}'::JSONB,
    'Total customer accounts served. Utilities and telecom intensity metric.',
    true, true
),
(
    'DEN-DATA-TB', 'Data Throughput', 'TB', 'PHYSICAL',
    '["DATA_CENTER", "ICT"]'::JSONB,
    '{"SECTOR": "OPTIONAL"}'::JSONB,
    'Data throughput in terabytes. Data center and ICT efficiency metric.',
    true, true
),
(
    'DEN-MEALS', 'Meals Served', 'meals', 'PHYSICAL',
    '["HOSPITALITY", "FOOD_SERVICE"]'::JSONB,
    '{"SECTOR": "OPTIONAL"}'::JSONB,
    'Total meals served. Hospitality and food service intensity metric.',
    true, true
),
(
    'DEN-ROOM-NIGHTS', 'Nights Sold', 'room-nights', 'PHYSICAL',
    '["HOSPITALITY"]'::JSONB,
    '{"GRESB": "RECOMMENDED", "HCMI": "MANDATORY"}'::JSONB,
    'Room-nights sold. HCMI (Hotel Carbon Measurement Initiative) mandatory for hospitality.',
    true, true
);

-- =============================================================================
-- Grants for Views
-- =============================================================================
GRANT SELECT ON ghg_intensity.v_intensity_latest TO PUBLIC;
GRANT SELECT ON ghg_intensity.v_intensity_with_targets TO PUBLIC;
GRANT SELECT ON ghg_intensity.v_intensity_benchmark_summary TO PUBLIC;
GRANT SELECT ON ghg_intensity.mv_intensity_dashboard TO PUBLIC;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON VIEW ghg_intensity.v_intensity_latest IS
    'Latest intensity calculation per organisation, denominator, scope, and entity. Uses most recent period_end for each combination.';
COMMENT ON VIEW ghg_intensity.v_intensity_with_targets IS
    'Intensity calculations joined with target progress data showing current intensity alongside target, variance, and status.';
COMMENT ON VIEW ghg_intensity.v_intensity_benchmark_summary IS
    'Benchmark results with peer group context: percentile rank, summary statistics, and gap metrics.';
COMMENT ON MATERIALIZED VIEW ghg_intensity.mv_intensity_dashboard IS
    'Pre-computed dashboard summary per organisation with latest intensity, trend direction, CARR, and data quality. Refresh with REFRESH MATERIALIZED VIEW CONCURRENTLY.';

-- =============================================================================
-- Table and Column Comments (all tables)
-- =============================================================================
COMMENT ON TABLE ghg_intensity.gl_im_configurations IS
    'PACK-046: Organisation-level intensity metrics configuration defining sector, scope, and calculation settings.';
COMMENT ON TABLE ghg_intensity.gl_im_reporting_periods IS
    'PACK-046: Reporting periods for intensity calculations with lifecycle status and base year flagging.';
COMMENT ON TABLE ghg_intensity.gl_im_denominator_definitions IS
    'PACK-046: Registry of 25 standard and custom denominators with sector applicability and framework alignment.';
COMMENT ON TABLE ghg_intensity.gl_im_denominator_values IS
    'PACK-046: Denominator data per organisation/period/entity with quality scoring and provenance.';
COMMENT ON TABLE ghg_intensity.gl_im_calculations IS
    'PACK-046: Calculated intensity metrics (emissions/denominator) with full provenance and data quality.';
COMMENT ON TABLE ghg_intensity.gl_im_time_series IS
    'PACK-046: Pre-computed intensity time series for trending with CARR and Mann-Kendall significance.';
COMMENT ON TABLE ghg_intensity.gl_im_decompositions IS
    'PACK-046: LMDI decomposition results separating activity, structure, and intensity effects.';
COMMENT ON TABLE ghg_intensity.gl_im_decomposition_entities IS
    'PACK-046: Entity-level contributions to each LMDI decomposition effect.';
COMMENT ON TABLE ghg_intensity.gl_im_peer_groups IS
    'PACK-046: Benchmark peer group definitions with sector and geography selection criteria.';
COMMENT ON TABLE ghg_intensity.gl_im_peer_data IS
    'PACK-046: External peer intensity data from CDP, TPI, GRESB, CRREM, and custom sources.';
COMMENT ON TABLE ghg_intensity.gl_im_benchmark_results IS
    'PACK-046: Benchmark comparison results with percentile rank, summary statistics, and gap metrics.';
COMMENT ON TABLE ghg_intensity.gl_im_targets IS
    'PACK-046: SBTi SDA and custom intensity reduction targets with sector convergence pathways.';
COMMENT ON TABLE ghg_intensity.gl_im_target_pathways IS
    'PACK-046: Annual target pathway milestones with expected intensity and cumulative reduction.';
COMMENT ON TABLE ghg_intensity.gl_im_target_progress IS
    'PACK-046: Actual vs target tracking with ON_TRACK/AT_RISK/OFF_TRACK/AHEAD status.';
COMMENT ON TABLE ghg_intensity.gl_im_scenarios IS
    'PACK-046: Scenario analysis definitions for intensity projection with Monte Carlo parameters.';
COMMENT ON TABLE ghg_intensity.gl_im_scenario_results IS
    'PACK-046: Scenario simulation results per projection year with percentile distribution.';
COMMENT ON TABLE ghg_intensity.gl_im_monte_carlo_runs IS
    'PACK-046: Monte Carlo simulation run metadata with convergence and performance tracking.';
COMMENT ON TABLE ghg_intensity.gl_im_uncertainty_assessments IS
    'PACK-046: Uncertainty quantification using IPCC Tier 1/2 propagation with confidence intervals.';
COMMENT ON TABLE ghg_intensity.gl_im_data_quality_log IS
    'PACK-046: Data quality scoring per element for trend analysis and assurance support.';
COMMENT ON TABLE ghg_intensity.gl_im_disclosure_frameworks IS
    'PACK-046: Configured disclosure frameworks per organisation with mandatory/optional and deadlines.';
COMMENT ON TABLE ghg_intensity.gl_im_disclosure_mappings IS
    'PACK-046: Individual field mappings to framework requirements with validation and override.';
COMMENT ON TABLE ghg_intensity.gl_im_disclosure_packages IS
    'PACK-046: Generated disclosure packages (MD/HTML/PDF/JSON/XBRL) with completeness tracking.';

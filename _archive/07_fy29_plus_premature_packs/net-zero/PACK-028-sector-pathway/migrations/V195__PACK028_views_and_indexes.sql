-- =============================================================================
-- V195: PACK-028 Sector Pathway Pack - Views & Performance Indexes
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    015 of 015
-- Date:         March 2026
--
-- Sector pathway dashboard views and comprehensive performance indexes
-- across all PACK-028 tables for production-grade query performance.
--
-- Views (3):
--   1. pack028_sector_pathway.v_sector_pathway_dashboard
--   2. pack028_sector_pathway.v_technology_roadmap_summary
--   3. pack028_sector_pathway.v_benchmark_comparison
--
-- Indexes: 80+ composite and conditional indexes.
--
-- Previous: V194__PACK028_technology_adoption.sql
-- =============================================================================

-- =============================================================================
-- View 1: v_sector_pathway_dashboard
-- =============================================================================
-- Consolidated sector pathway dashboard with current intensity, pathway
-- targets, convergence status, technology progress, and benchmark position
-- for executive-level sector transition monitoring.

CREATE OR REPLACE VIEW pack028_sector_pathway.v_sector_pathway_dashboard AS
SELECT
    sc.classification_id,
    sc.tenant_id,
    sc.company_id,
    sc.primary_sector,
    sc.primary_sector_code,
    sc.nace_code,
    sc.gics_code,
    sc.sda_eligible,
    sc.sda_sector,
    sc.flag_eligible,
    sc.iea_sector,
    sc.carbon_intensity_profile,
    sc.validation_status         AS classification_status,
    -- Latest intensity metric
    im.metric_type               AS intensity_metric_type,
    im.intensity_value           AS current_intensity,
    im.intensity_unit            AS intensity_unit,
    im.reporting_year            AS intensity_year,
    im.yoy_change_pct            AS intensity_yoy_change,
    im.data_quality_score        AS intensity_dq_score,
    im.verification_status       AS intensity_verification,
    -- Primary pathway
    sp.pathway_name,
    sp.pathway_type,
    sp.pathway_source,
    sp.scenario                  AS pathway_scenario,
    sp.temperature_target,
    sp.base_year                 AS pathway_base_year,
    sp.base_year_intensity       AS pathway_base_intensity,
    sp.target_year               AS pathway_target_year,
    sp.target_intensity          AS pathway_target_intensity,
    sp.target_reduction_pct      AS pathway_reduction_pct,
    sp.interim_2030_intensity,
    sp.interim_2050_intensity    AS final_target_intensity,
    sp.convergence_model,
    sp.sbti_aligned,
    sp.iea_aligned,
    sp.pathway_status,
    -- Latest convergence analysis
    cv.intensity_gap,
    cv.intensity_gap_pct,
    cv.gap_direction,
    cv.gap_severity,
    cv.time_to_convergence_years,
    cv.convergence_feasible,
    cv.required_annual_reduction_pct,
    cv.acceleration_needed_pct,
    cv.risk_level                AS convergence_risk,
    cv.probability_on_track,
    cv.percentile_rank,
    cv.sbti_aligned              AS convergence_sbti_aligned,
    cv.iea_aligned               AS convergence_iea_aligned,
    -- Latest benchmark
    bm.percentile_overall        AS benchmark_percentile,
    bm.performance_tier,
    bm.gap_to_leader_pct,
    bm.iea_alignment_score       AS benchmark_iea_score,
    bm.sda_alignment_score       AS benchmark_sda_score,
    bm.benchmark_composite_score,
    -- Technology roadmap summary
    tr_agg.total_technologies,
    tr_agg.technologies_deployed,
    tr_agg.technologies_in_pilot,
    tr_agg.total_capex_planned,
    tr_agg.avg_trl,
    -- Abatement summary
    al_agg.total_levers,
    al_agg.total_abatement_tco2e,
    al_agg.levers_in_progress,
    al_agg.avg_cost_per_tco2e,
    -- SBTi SDA compliance
    ssp.submission_status        AS sbti_submission_status,
    ssp.near_term_compliant,
    ssp.long_term_compliant,
    ssp.overall_criteria_pct     AS sbti_criteria_pct
FROM pack028_sector_pathway.gl_sector_classifications sc
-- Latest intensity metric
LEFT JOIN LATERAL (
    SELECT * FROM pack028_sector_pathway.gl_sector_intensity_metrics
    WHERE company_id = sc.company_id AND sector_code = sc.primary_sector_code
    ORDER BY reporting_year DESC LIMIT 1
) im ON TRUE
-- Primary active pathway
LEFT JOIN LATERAL (
    SELECT * FROM pack028_sector_pathway.gl_sector_pathways
    WHERE company_id = sc.company_id AND sector_code = sc.primary_sector_code
          AND is_active = TRUE AND is_primary = TRUE
    ORDER BY created_at DESC LIMIT 1
) sp ON TRUE
-- Latest convergence analysis
LEFT JOIN LATERAL (
    SELECT * FROM pack028_sector_pathway.gl_sector_convergence
    WHERE company_id = sc.company_id AND sector_code = sc.primary_sector_code
    ORDER BY analysis_date DESC LIMIT 1
) cv ON TRUE
-- Latest benchmark
LEFT JOIN LATERAL (
    SELECT * FROM pack028_sector_pathway.gl_sector_benchmarks
    WHERE company_id = sc.company_id AND sector_code = sc.primary_sector_code
    ORDER BY benchmark_year DESC LIMIT 1
) bm ON TRUE
-- Technology roadmap aggregate
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS total_technologies,
        COUNT(*) FILTER (WHERE implementation_status IN ('DEPLOYED', 'MATURE')) AS technologies_deployed,
        COUNT(*) FILTER (WHERE implementation_status = 'PILOT')         AS technologies_in_pilot,
        SUM(total_capex_usd)                                            AS total_capex_planned,
        AVG(current_trl)                                                AS avg_trl
    FROM pack028_sector_pathway.gl_technology_roadmaps
    WHERE company_id = sc.company_id AND sector_code = sc.primary_sector_code AND is_active = TRUE
) tr_agg ON TRUE
-- Abatement lever aggregate
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS total_levers,
        SUM(abatement_tco2e_annual)                                     AS total_abatement_tco2e,
        COUNT(*) FILTER (WHERE implementation_status = 'IN_PROGRESS')   AS levers_in_progress,
        AVG(cost_per_tco2e)                                             AS avg_cost_per_tco2e
    FROM pack028_sector_pathway.gl_sector_abatement_levers
    WHERE company_id = sc.company_id AND sector_code = sc.primary_sector_code AND is_active = TRUE
) al_agg ON TRUE
-- SBTi SDA pathway status
LEFT JOIN LATERAL (
    SELECT * FROM pack028_sector_pathway.gl_sbti_sector_pathways
    WHERE company_id = sc.company_id AND sda_sector = sc.sda_sector AND is_active = TRUE
    ORDER BY created_at DESC LIMIT 1
) ssp ON TRUE
WHERE sc.is_primary = TRUE
  AND sc.validation_status != 'REJECTED';

-- =============================================================================
-- View 2: v_technology_roadmap_summary
-- =============================================================================
-- Technology roadmap summary with deployment status, IEA milestone alignment,
-- cost tracking, and abatement contribution per technology.

CREATE OR REPLACE VIEW pack028_sector_pathway.v_technology_roadmap_summary AS
SELECT
    tr.roadmap_id,
    tr.tenant_id,
    tr.company_id,
    tr.sector,
    tr.sector_code,
    tr.technology_name,
    tr.technology_code,
    tr.technology_category,
    tr.current_trl,
    tr.target_trl,
    tr.adoption_model,
    tr.current_penetration_pct,
    tr.target_penetration_2030_pct,
    tr.target_penetration_2050_pct,
    tr.adoption_start_year,
    tr.implementation_status,
    tr.total_capex_usd,
    tr.marginal_abatement_cost,
    tr.abatement_potential_tco2e,
    tr.abatement_share_pct,
    tr.technology_risk,
    tr.priority,
    tr.iea_milestone_ref,
    tr.iea_milestone_status,
    tr.dependencies_met,
    tr.region,
    tr.is_active,
    -- Latest adoption tracking
    tat.current_trl              AS tracked_trl,
    tat.deployment_status        AS tracked_deploy_status,
    tat.deployment_progress_pct,
    tat.actual_abatement_tco2e,
    tat.planned_abatement_tco2e,
    tat.actual_capex_usd,
    tat.planned_capex_usd,
    tat.actual_cost_per_tco2e,
    tat.risk_level               AS tracked_risk,
    tat.all_dependencies_met,
    tat.reporting_year           AS latest_tracking_year,
    -- Calculated fields
    CASE
        WHEN tr.abatement_potential_tco2e > 0 AND tat.actual_abatement_tco2e IS NOT NULL
        THEN ROUND((tat.actual_abatement_tco2e / tr.abatement_potential_tco2e * 100)::NUMERIC, 1)
        ELSE 0
    END AS abatement_achievement_pct,
    CASE
        WHEN tr.total_capex_usd > 0 AND tat.actual_capex_usd IS NOT NULL
        THEN ROUND((tat.actual_capex_usd / tr.total_capex_usd * 100)::NUMERIC, 1)
        ELSE 0
    END AS capex_utilization_pct,
    CASE
        WHEN tr.iea_milestone_status IS NOT NULL
        THEN CASE tr.iea_milestone_status
            WHEN 'ON_TRACK' THEN 'GREEN'
            WHEN 'ACHIEVED' THEN 'GREEN'
            WHEN 'BEHIND' THEN 'AMBER'
            WHEN 'WELL_BEHIND' THEN 'RED'
            WHEN 'NOT_STARTED' THEN 'GREY'
            ELSE 'GREY'
        END
        ELSE 'GREY'
    END AS iea_rag_status
FROM pack028_sector_pathway.gl_technology_roadmaps tr
LEFT JOIN LATERAL (
    SELECT * FROM pack028_sector_pathway.gl_technology_adoption_tracking
    WHERE roadmap_id = tr.roadmap_id AND is_active = TRUE
    ORDER BY reporting_year DESC, reporting_quarter DESC NULLS LAST LIMIT 1
) tat ON TRUE
WHERE tr.is_active = TRUE;

-- =============================================================================
-- View 3: v_benchmark_comparison
-- =============================================================================
-- Sector benchmark comparison with peer positioning, pathway alignment
-- scores, and performance trend for benchmarking dashboards.

CREATE OR REPLACE VIEW pack028_sector_pathway.v_benchmark_comparison AS
SELECT
    bm.benchmark_id,
    bm.tenant_id,
    bm.company_id,
    bm.sector,
    bm.sector_code,
    bm.intensity_metric,
    bm.benchmark_year,
    -- Company position
    bm.company_intensity,
    bm.company_intensity_unit,
    bm.company_trend_3yr_pct,
    bm.company_trend_5yr_pct,
    bm.company_rank,
    bm.total_peer_count,
    -- Percentile rankings
    bm.percentile_overall,
    bm.percentile_by_region,
    bm.percentile_by_size,
    bm.percentile_sbti_peers,
    -- Peer statistics
    bm.peer_mean_intensity,
    bm.peer_median_intensity,
    bm.peer_p10_intensity,
    bm.peer_p90_intensity,
    -- Gap analysis
    bm.leader_intensity,
    bm.leader_name,
    bm.gap_to_leader_pct,
    bm.gap_to_leader_absolute,
    bm.top_decile_avg_intensity,
    bm.gap_to_top_decile_pct,
    -- SBTi & IEA alignment
    bm.sbti_peer_count,
    bm.sbti_peer_avg_intensity,
    bm.gap_to_sbti_avg_pct,
    bm.iea_pathway_intensity,
    bm.iea_scenario,
    bm.gap_to_iea_pct,
    bm.iea_alignment_score,
    bm.sda_pathway_intensity,
    bm.gap_to_sda_pct,
    bm.sda_alignment_score,
    -- Performance
    bm.benchmark_composite_score,
    bm.performance_tier,
    bm.benchmark_trend,
    bm.region,
    bm.size_category,
    -- RAG status
    CASE bm.performance_tier
        WHEN 'LEADER' THEN 'GREEN'
        WHEN 'FRONT_RUNNER' THEN 'GREEN'
        WHEN 'ALIGNED' THEN 'AMBER'
        WHEN 'LAGGING' THEN 'RED'
        WHEN 'SIGNIFICANTLY_BEHIND' THEN 'RED'
        ELSE 'GREY'
    END AS performance_rag,
    -- Improvement needed
    CASE
        WHEN bm.gap_to_iea_pct IS NOT NULL AND bm.gap_to_iea_pct > 0
        THEN ROUND(bm.gap_to_iea_pct, 1)
        ELSE 0
    END AS iea_gap_to_close_pct,
    CASE
        WHEN bm.gap_to_sda_pct IS NOT NULL AND bm.gap_to_sda_pct > 0
        THEN ROUND(bm.gap_to_sda_pct, 1)
        ELSE 0
    END AS sda_gap_to_close_pct
FROM pack028_sector_pathway.gl_sector_benchmarks bm;

-- =============================================================================
-- Performance Indexes (Composite & Conditional)
-- =============================================================================
-- Multi-column composite indexes for common query patterns across all tables.

-- Sector classifications: multi-code lookups
CREATE INDEX idx_p028_sc_nace_gics          ON pack028_sector_pathway.gl_sector_classifications(nace_code, gics_code);
CREATE INDEX idx_p028_sc_company_sda        ON pack028_sector_pathway.gl_sector_classifications(company_id, sda_eligible) WHERE sda_eligible = TRUE;
CREATE INDEX idx_p028_sc_company_flag       ON pack028_sector_pathway.gl_sector_classifications(company_id, flag_eligible) WHERE flag_eligible = TRUE;
CREATE INDEX idx_p028_sc_tenant_primary     ON pack028_sector_pathway.gl_sector_classifications(tenant_id, is_primary) WHERE is_primary = TRUE;
CREATE INDEX idx_p028_sc_sector_intensity   ON pack028_sector_pathway.gl_sector_classifications(primary_sector_code, carbon_intensity_profile);

-- Intensity metrics: time series queries
CREATE INDEX idx_p028_im_company_sector_ts  ON pack028_sector_pathway.gl_sector_intensity_metrics(company_id, sector_code, metric_type, reporting_year DESC);
CREATE INDEX idx_p028_im_sector_year_int    ON pack028_sector_pathway.gl_sector_intensity_metrics(sector_code, reporting_year, intensity_value);
CREATE INDEX idx_p028_im_base_sector        ON pack028_sector_pathway.gl_sector_intensity_metrics(sector_code, is_base_year) WHERE is_base_year = TRUE;
CREATE INDEX idx_p028_im_verified_year      ON pack028_sector_pathway.gl_sector_intensity_metrics(verification_status, reporting_year) WHERE verification_status != 'UNVERIFIED';
CREATE INDEX idx_p028_im_quality_low        ON pack028_sector_pathway.gl_sector_intensity_metrics(data_quality_level, sector_code) WHERE data_quality_level >= 4;

-- Sector pathways: scenario-sector-active lookups
CREATE INDEX idx_p028_sp_company_active_pri ON pack028_sector_pathway.gl_sector_pathways(company_id, is_active, is_primary) WHERE is_active = TRUE AND is_primary = TRUE;
CREATE INDEX idx_p028_sp_sector_scenario    ON pack028_sector_pathway.gl_sector_pathways(sector_code, scenario, pathway_status);
CREATE INDEX idx_p028_sp_source_type        ON pack028_sector_pathway.gl_sector_pathways(pathway_source, pathway_type);
CREATE INDEX idx_p028_sp_sector_region      ON pack028_sector_pathway.gl_sector_pathways(sector_code, region);
CREATE INDEX idx_p028_sp_approved           ON pack028_sector_pathway.gl_sector_pathways(approved_at DESC) WHERE pathway_status = 'APPROVED';

-- Convergence: risk and gap tracking
CREATE INDEX idx_p028_cv_company_risk       ON pack028_sector_pathway.gl_sector_convergence(company_id, risk_level, analysis_year);
CREATE INDEX idx_p028_cv_sector_gap         ON pack028_sector_pathway.gl_sector_convergence(sector_code, gap_severity, analysis_year);
CREATE INDEX idx_p028_cv_scenario_gap       ON pack028_sector_pathway.gl_sector_convergence(scenario, gap_severity);
CREATE INDEX idx_p028_cv_acceleration       ON pack028_sector_pathway.gl_sector_convergence(acceleration_needed_pct DESC NULLS LAST) WHERE acceleration_needed_pct > 0;
CREATE INDEX idx_p028_cv_infeasible         ON pack028_sector_pathway.gl_sector_convergence(company_id) WHERE convergence_feasible = FALSE;

-- Technology roadmaps: deployment pipeline
CREATE INDEX idx_p028_tr_company_impl       ON pack028_sector_pathway.gl_technology_roadmaps(company_id, implementation_status);
CREATE INDEX idx_p028_tr_sector_cat_status  ON pack028_sector_pathway.gl_technology_roadmaps(sector_code, technology_category, implementation_status);
CREATE INDEX idx_p028_tr_trl_sector         ON pack028_sector_pathway.gl_technology_roadmaps(current_trl, sector_code);
CREATE INDEX idx_p028_tr_priority_active    ON pack028_sector_pathway.gl_technology_roadmaps(priority, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_tr_capex_desc         ON pack028_sector_pathway.gl_technology_roadmaps(total_capex_usd DESC NULLS LAST);
CREATE INDEX idx_p028_tr_abatement_desc     ON pack028_sector_pathway.gl_technology_roadmaps(abatement_potential_tco2e DESC NULLS LAST);

-- Abatement levers: waterfall and cost queries
CREATE INDEX idx_p028_al_company_waterfall  ON pack028_sector_pathway.gl_sector_abatement_levers(company_id, pathway_id, waterfall_order) WHERE is_active = TRUE;
CREATE INDEX idx_p028_al_sector_cat_cost    ON pack028_sector_pathway.gl_sector_abatement_levers(sector_code, lever_category, cost_per_tco2e);
CREATE INDEX idx_p028_al_cost_curve         ON pack028_sector_pathway.gl_sector_abatement_levers(cost_per_tco2e ASC, abatement_tco2e_annual DESC) WHERE is_active = TRUE;
CREATE INDEX idx_p028_al_company_impl       ON pack028_sector_pathway.gl_sector_abatement_levers(company_id, implementation_status);
CREATE INDEX idx_p028_al_maturity_cat       ON pack028_sector_pathway.gl_sector_abatement_levers(maturity, lever_category);

-- Benchmarks: performance ranking
CREATE INDEX idx_p028_bm_sector_year_perc   ON pack028_sector_pathway.gl_sector_benchmarks(sector_code, benchmark_year, percentile_overall DESC);
CREATE INDEX idx_p028_bm_tier_sector        ON pack028_sector_pathway.gl_sector_benchmarks(performance_tier, sector_code, benchmark_year);
CREATE INDEX idx_p028_bm_company_latest     ON pack028_sector_pathway.gl_sector_benchmarks(company_id, benchmark_year DESC);
CREATE INDEX idx_p028_bm_leader_gap_desc    ON pack028_sector_pathway.gl_sector_benchmarks(gap_to_leader_pct DESC NULLS LAST) WHERE gap_to_leader_pct IS NOT NULL;
CREATE INDEX idx_p028_bm_behind_iea        ON pack028_sector_pathway.gl_sector_benchmarks(gap_to_iea_pct DESC NULLS LAST) WHERE gap_to_iea_pct > 0;

-- Scenario comparisons: analysis queries
CREATE INDEX idx_p028_scc_company_active_dt ON pack028_sector_pathway.gl_scenario_comparisons(company_id, is_active, analysis_date DESC) WHERE is_active = TRUE;
CREATE INDEX idx_p028_scc_sector_type       ON pack028_sector_pathway.gl_scenario_comparisons(sector_code, comparison_type);
CREATE INDEX idx_p028_scc_approved_dt       ON pack028_sector_pathway.gl_scenario_comparisons(approved_at DESC) WHERE comparison_status = 'APPROVED';

-- SBTi sector pathways: compliance pipeline
CREATE INDEX idx_p028_ssp_sector_ambition   ON pack028_sector_pathway.gl_sbti_sector_pathways(sda_sector, ambition_level);
CREATE INDEX idx_p028_ssp_company_active    ON pack028_sector_pathway.gl_sbti_sector_pathways(company_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_ssp_non_compliant_nt  ON pack028_sector_pathway.gl_sbti_sector_pathways(near_term_compliant) WHERE near_term_compliant = FALSE;
CREATE INDEX idx_p028_ssp_non_compliant_lt  ON pack028_sector_pathway.gl_sbti_sector_pathways(long_term_compliant) WHERE long_term_compliant = FALSE;
CREATE INDEX idx_p028_ssp_pending_sub       ON pack028_sector_pathway.gl_sbti_sector_pathways(submission_status) WHERE submission_status IN ('NOT_SUBMITTED', 'COMMITTED');

-- IEA milestones: progress tracking
CREATE INDEX idx_p028_iem_sector_year_cat   ON pack028_sector_pathway.gl_iea_technology_milestones(sector_code, target_year, milestone_category);
CREATE INDEX idx_p028_iem_company_behind    ON pack028_sector_pathway.gl_iea_technology_milestones(company_id, company_on_track) WHERE company_on_track = FALSE;
CREATE INDEX idx_p028_iem_critical_behind   ON pack028_sector_pathway.gl_iea_technology_milestones(sector_code) WHERE is_critical_milestone = TRUE AND global_progress_status IN ('BEHIND', 'WELL_BEHIND');
CREATE INDEX idx_p028_iem_region_sector     ON pack028_sector_pathway.gl_iea_technology_milestones(region, sector_code, target_year);

-- IPCC pathways: scenario-sector lookups
CREATE INDEX idx_p028_ipp_ssp_sector        ON pack028_sector_pathway.gl_ipcc_sector_pathways(ssp_scenario, sector_code);
CREATE INDEX idx_p028_ipp_temp_sector       ON pack028_sector_pathway.gl_ipcc_sector_pathways(temperature_outcome, sector_code);
CREATE INDEX idx_p028_ipp_ref_active        ON pack028_sector_pathway.gl_ipcc_sector_pathways(is_reference_data, is_active) WHERE is_reference_data = TRUE AND is_active = TRUE;

-- Emission factors: sector-source lookups
CREATE INDEX idx_p028_sef_sector_source_yr  ON pack028_sector_pathway.gl_sector_emission_factors(sector_code, source, data_vintage DESC);
CREATE INDEX idx_p028_sef_default_sector    ON pack028_sector_pathway.gl_sector_emission_factors(sector_code, is_default, is_active) WHERE is_default = TRUE AND is_active = TRUE;
CREATE INDEX idx_p028_sef_category_region   ON pack028_sector_pathway.gl_sector_emission_factors(factor_category, region);

-- Activity data: time series and quality
CREATE INDEX idx_p028_sad_company_type_yr   ON pack028_sector_pathway.gl_sector_activity_data(company_id, activity_type, reporting_year DESC);
CREATE INDEX idx_p028_sad_sector_type       ON pack028_sector_pathway.gl_sector_activity_data(sector_code, activity_type);
CREATE INDEX idx_p028_sad_low_quality       ON pack028_sector_pathway.gl_sector_activity_data(data_quality, sector_code) WHERE data_quality IN ('ESTIMATED', 'PROXY', 'DEFAULT');

-- Technology catalog: sector-category lookups
CREATE INDEX idx_p028_stc_sector_trl        ON pack028_sector_pathway.gl_sector_technology_catalog(sector_code, current_trl DESC);
CREATE INDEX idx_p028_stc_readiness_sector  ON pack028_sector_pathway.gl_sector_technology_catalog(commercial_readiness, sector_code);

-- Scenario definitions: group and type
CREATE INDEX idx_p028_sd_company_active     ON pack028_sector_pathway.gl_scenario_definitions(company_id, scenario_status) WHERE scenario_status = 'ACTIVE';
CREATE INDEX idx_p028_sd_group_order        ON pack028_sector_pathway.gl_scenario_definitions(comparison_group_id, comparison_order);

-- Scenario parameters: scenario-sector-category
CREATE INDEX idx_p028_spm_scen_sect_cat     ON pack028_sector_pathway.gl_scenario_parameters(scenario_def_id, sector_code, parameter_category);
CREATE INDEX idx_p028_spm_key_drivers       ON pack028_sector_pathway.gl_scenario_parameters(scenario_def_id, sensitivity_rank) WHERE is_key_driver = TRUE;

-- Scenario results: latest completed
CREATE INDEX idx_p028_sr_latest_completed   ON pack028_sector_pathway.gl_scenario_results(company_id, scenario_def_id, is_latest) WHERE is_latest = TRUE AND computation_status = 'COMPLETED';
CREATE INDEX idx_p028_sr_sector_alignment   ON pack028_sector_pathway.gl_scenario_results(sector_code, sbti_alignment_score DESC NULLS LAST);

-- Technology adoption: progress pipeline
CREATE INDEX idx_p028_tat_company_yr_status ON pack028_sector_pathway.gl_technology_adoption_tracking(company_id, reporting_year, deployment_status);
CREATE INDEX idx_p028_tat_sector_cat_yr     ON pack028_sector_pathway.gl_technology_adoption_tracking(sector_code, technology_category, reporting_year DESC);
CREATE INDEX idx_p028_tat_roadmap_latest    ON pack028_sector_pathway.gl_technology_adoption_tracking(roadmap_id, reporting_year DESC) WHERE is_active = TRUE;
CREATE INDEX idx_p028_tat_variance_high     ON pack028_sector_pathway.gl_technology_adoption_tracking(abatement_variance_pct) WHERE abatement_variance_pct IS NOT NULL AND ABS(abatement_variance_pct) > 20;
CREATE INDEX idx_p028_tat_over_budget       ON pack028_sector_pathway.gl_technology_adoption_tracking(capex_variance_pct) WHERE capex_variance_pct IS NOT NULL AND capex_variance_pct > 10;

-- =============================================================================
-- Comments on Views
-- =============================================================================
COMMENT ON VIEW pack028_sector_pathway.v_sector_pathway_dashboard IS
    'Consolidated sector pathway dashboard with current intensity, pathway targets, convergence status, technology progress, abatement summary, benchmark position, and SBTi SDA compliance for executive-level monitoring.';

COMMENT ON VIEW pack028_sector_pathway.v_technology_roadmap_summary IS
    'Technology roadmap summary combining planned roadmap with latest adoption tracking data, IEA milestone alignment, cost tracking, and abatement achievement metrics.';

COMMENT ON VIEW pack028_sector_pathway.v_benchmark_comparison IS
    'Sector benchmark comparison with peer positioning, percentile rankings, SBTi/IEA alignment scores, gap analysis, performance tier classification, and RAG status indicators.';

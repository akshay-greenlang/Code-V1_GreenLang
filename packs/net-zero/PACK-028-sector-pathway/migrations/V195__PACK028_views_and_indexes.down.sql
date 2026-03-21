-- =============================================================================
-- V195 DOWN: Drop all views and performance indexes
-- =============================================================================

-- Drop views first
DROP VIEW IF EXISTS pack028_sector_pathway.v_benchmark_comparison;
DROP VIEW IF EXISTS pack028_sector_pathway.v_technology_roadmap_summary;
DROP VIEW IF EXISTS pack028_sector_pathway.v_sector_pathway_dashboard;

-- Drop composite/conditional performance indexes
-- Sector classifications
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sc_nace_gics;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sc_company_sda;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sc_company_flag;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sc_tenant_primary;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sc_sector_intensity;

-- Intensity metrics
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_im_company_sector_ts;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_im_sector_year_int;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_im_base_sector;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_im_verified_year;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_im_quality_low;

-- Sector pathways
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sp_company_active_pri;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sp_sector_scenario;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sp_source_type;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sp_sector_region;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sp_approved;

-- Convergence
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_cv_company_risk;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_cv_sector_gap;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_cv_scenario_gap;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_cv_acceleration;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_cv_infeasible;

-- Technology roadmaps
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tr_company_impl;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tr_sector_cat_status;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tr_trl_sector;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tr_priority_active;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tr_capex_desc;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tr_abatement_desc;

-- Abatement levers
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_al_company_waterfall;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_al_sector_cat_cost;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_al_cost_curve;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_al_company_impl;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_al_maturity_cat;

-- Benchmarks
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_bm_sector_year_perc;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_bm_tier_sector;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_bm_company_latest;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_bm_leader_gap_desc;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_bm_behind_iea;

-- Scenario comparisons
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_scc_company_active_dt;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_scc_sector_type;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_scc_approved_dt;

-- SBTi sector pathways
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_ssp_sector_ambition;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_ssp_company_active;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_ssp_non_compliant_nt;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_ssp_non_compliant_lt;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_ssp_pending_sub;

-- IEA milestones
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_iem_sector_year_cat;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_iem_company_behind;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_iem_critical_behind;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_iem_region_sector;

-- IPCC pathways
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_ipp_ssp_sector;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_ipp_temp_sector;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_ipp_ref_active;

-- Emission factors
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sef_sector_source_yr;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sef_default_sector;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sef_category_region;

-- Activity data
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sad_company_type_yr;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sad_sector_type;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sad_low_quality;

-- Technology catalog
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_stc_sector_trl;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_stc_readiness_sector;

-- Scenario definitions
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sd_company_active;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sd_group_order;

-- Scenario parameters
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_spm_scen_sect_cat;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_spm_key_drivers;

-- Scenario results
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sr_latest_completed;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_sr_sector_alignment;

-- Technology adoption
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tat_company_yr_status;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tat_sector_cat_yr;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tat_roadmap_latest;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tat_variance_high;
DROP INDEX IF EXISTS pack028_sector_pathway.idx_p028_tat_over_budget;

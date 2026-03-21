-- =============================================================================
-- V180 DOWN: Drop all views and performance indexes
-- =============================================================================

-- Drop views first
DROP VIEW IF EXISTS pack027_enterprise_net_zero.vw_regulatory_calendar;
DROP VIEW IF EXISTS pack027_enterprise_net_zero.vw_supply_chain_heatmap;
DROP VIEW IF EXISTS pack027_enterprise_net_zero.vw_enterprise_dashboard;

-- Drop composite/conditional performance indexes
-- Enterprise profiles
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ep_sector_country;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ep_sector_employees;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ep_country_revenue;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ep_tenant_active;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ep_boundary_status;

-- Entity hierarchy
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_eh_parent_active;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_eh_child_active;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_eh_tenant_active;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_eh_type_control;

-- Intercompany transactions
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ict_pending_elim;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ict_unmatched;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ict_from_year;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ict_to_year;

-- Baselines
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_bl_tenant_year;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_bl_company_base;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_bl_year_dq;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_bl_verified_year;

-- SBTi targets
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_st_company_type;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_st_pathway_ambition;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_st_validation_type;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_st_submitted;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_st_reval_due;

-- Scenarios
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_sm_company_active;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_sm_type_pathway;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_sm_company_completed;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_sm_comparison_active;

-- Carbon pricing
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cp_company_active;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cp_type_active;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cp_regulatory_active;

-- Carbon liabilities
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cl_company_year_q;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cl_status_year;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cl_ebitda_desc;

-- Scope 4
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_s4_company_year;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_s4_type_verified;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_s4_avoided_desc;

-- Supply chain
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_sct_company_year_tier;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_sct_emissions_desc;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_se_company_tier;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_se_company_engage;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_se_overdue;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_se_risk_country;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_se_spend_emissions;

-- Financial integration
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cpl_company_year_bu;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cpl_cbam_year;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cpl_cost_desc;

-- Carbon assets
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ca_company_active;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ca_type_vintage;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ca_expiring;

-- Climate risks
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cr_company_type;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cr_severity_status;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cr_high_critical;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cr_unmitigated;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cr_impact_desc;

-- Asset risk exposure
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_are_company_type;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_are_country_physical;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_are_high_physical;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_are_high_transition;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_are_stranded_high;

-- Regulatory filings
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_rf_company_framework;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_rf_deadline_status;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_rf_overdue;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_rf_assurance_pending;

-- Compliance gaps
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cg_company_open;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cg_framework_severity;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cg_overdue;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_cg_critical_open;

-- Assurance engagements
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ae_company_year;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ae_status_level;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_ae_active_engagement;

-- Workpapers
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_aw_engagement_type;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_aw_pending_review;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_aw_scope_review;

-- Board reports
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_br_company_year_q;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_br_status_date;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_br_at_risk;

-- Data quality
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_dq_company_year_cat;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_dq_low_quality;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_dq_completeness_low;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_dq_accuracy_low;
DROP INDEX IF EXISTS pack027_enterprise_net_zero.idx_p027_dq_entity_year;

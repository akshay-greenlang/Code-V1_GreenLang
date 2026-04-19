-- =============================================================================
-- V210 DOWN: Drop all views and performance indexes
-- =============================================================================

-- Drop views first
DROP VIEW IF EXISTS pack029_interim_targets.v_milestone_status;
DROP VIEW IF EXISTS pack029_interim_targets.v_variance_dashboard;
DROP VIEW IF EXISTS pack029_interim_targets.v_progress_summary;

-- ---------------------------------------------------------------------------
-- Drop composite/conditional performance indexes
-- ---------------------------------------------------------------------------

-- Interim Targets
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_it_org_scope_year;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_it_org_active_scope;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_it_sbti_method_path;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_it_validated_scope;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_it_approval_pending;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_it_superseded;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_it_target_type_scope;

-- Annual Pathways
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ap_org_scope_year_act;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ap_target_year_type;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ap_budget_year;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ap_locked_active;

-- Quarterly Milestones
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_qm_org_scope_yr_qtr;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_qm_target_yr_status;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_qm_pending_year;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_qm_deferred;

-- Actual Performance
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_perf_org_scope_yr_qtr;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_perf_target_yr;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_perf_scope_cat_yr;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_perf_dq_scope;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_perf_verified_yr;

-- Variance Analysis
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_va_org_scope_yr_act;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_va_target_severity;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_va_off_track_severity;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_va_consec_miss;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_va_decomp_scope;

-- Corrective Actions
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ca_org_status_year;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ca_target_status;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ca_cost_effectiveness;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ca_deploy_schedule;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ca_progress_active;

-- Progress Alerts
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_pa_org_type_sev;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_pa_target_type;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_pa_reason_severity;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_pa_escalation_due;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_pa_recurring_org;

-- Initiative Schedule
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_is_org_cat_status;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_is_target_start_yr;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_is_schedule_risk;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_is_critical_delayed;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_is_permits_pending;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_is_pilot_active;

-- Carbon Budget Allocation
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_cba_org_scope_yr_act;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_cba_target_year;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_cba_overshoot_scope;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_cba_rebal_strategy;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_cba_locked_approved;

-- Reporting Periods
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_rp_org_year_active;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_rp_framework_year;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_rp_overdue;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_rp_incomplete_scope;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_rp_published_year;

-- Validation Results
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_vr_org_criteria_date;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_vr_run_criteria;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_vr_fail_category;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_vr_mandatory_fail;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_vr_improved;

-- Assurance Evidence
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ae_org_type_year;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ae_tier_complete;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ae_review_pending;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ae_quality_low;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ae_retention_due;

-- Trend Forecasts
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_tf_org_scope_yr_lat;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_tf_method_scenario;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_tf_accuracy_check;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_tf_off_target;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_tf_high_confidence;

-- SBTi Submissions
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ss_org_scope_status;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ss_pathway_method;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ss_commitment_pub;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ss_expiring;
DROP INDEX IF EXISTS pack029_interim_targets.idx_p029_ss_resub_pending;

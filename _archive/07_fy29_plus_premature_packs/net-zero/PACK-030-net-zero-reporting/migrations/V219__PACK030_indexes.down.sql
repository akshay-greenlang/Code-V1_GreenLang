-- =============================================================================
-- V219 DOWN: Drop PACK-030 performance composite indexes
-- =============================================================================
-- Note: Only drops the composite indexes added in V219.
-- Per-table indexes created in V211-V218 are dropped with their tables.

-- Reports composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_org_fw_yr_status;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_org_fw_latest_act;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_org_status_yr;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_fw_status_yr;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_org_published_fw;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_org_review_pending;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_org_draft_fw;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_completeness_fw;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_version_chain;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_approved_pending;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_archived_yr;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rpt_multi_fw;

-- Sections composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_rpt_type_lang_act;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_rpt_order_act;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_rpt_review_pend;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_rpt_needs_rev;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_rpt_low_consist;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_rpt_depth_parent;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_rpt_has_citations;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_version_chain;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_translated_lang;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_sec_word_count;

-- Metrics composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_rpt_scope_name;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_rpt_source_scope;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_rpt_category;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_rpt_pack_scope;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_rpt_verified;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_rpt_unverified;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_rpt_xbrl_tagged;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_rpt_high_uncert;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_fw_element;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_met_variance_prior;

-- Framework schemas composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fs_fw_type_current;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fs_fw_eff_date;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fs_source_org;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fs_field_count;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fs_version_chain;

-- Framework mappings composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fm_src_tgt_type_act;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fm_src_met_tgt_fw;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fm_tgt_met_src_fw;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fm_high_confidence;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fm_low_confidence;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fm_needs_conversion;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fm_most_used;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fm_last_used;

-- Framework deadlines composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fd_org_fw_yr;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fd_upcoming_30d;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fd_upcoming_90d;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fd_org_overdue;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fd_extended;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_fd_submitted_yr;

-- Narratives composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_nar_org_fw_sec_lang;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_nar_org_fw_audience;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_nar_org_fw_tone;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_nar_org_low_consist;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_nar_org_ai_unreviewed;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_nar_org_high_usage;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_nar_template_fw_sec;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_nar_version_chain;

-- Translations composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_tr_org_src_tgt_lang;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_tr_org_type_src;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_tr_low_quality;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_tr_high_quality;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_tr_citations_broken;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_tr_terminology_issue;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_tr_machine_unreviewed;

-- Assurance evidence composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_rpt_type_tier;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_rpt_std_level;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_org_incomplete;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_org_rejected;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_bundle_order;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_methodology_missing;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_control_adequate;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_high_quality;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_ae_low_quality;

-- Audit trail composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_org_cat_time;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_org_type_time;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_rpt_cat_time;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_actor_time;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_actor_type_time;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_resource_time;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_fw_type_time;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_session;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_approval_events;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_at_data_events;

-- Data lineage composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dl_rpt_src_metric;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dl_rpt_pack_metric;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dl_rpt_app_metric;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dl_org_src_type;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dl_rpt_dq_tier;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dl_rpt_low_conf;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dl_rpt_complex_xform;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dl_rpt_period;

-- XBRL tags composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_xb_rpt_fw_element;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_xb_rpt_fw_valid;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_xb_rpt_fw_invalid;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_xb_rpt_numeric;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_xb_rpt_text;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_xb_rpt_extensions;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_xb_ns_element;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_xb_context_period;

-- Validation results composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_rpt_sev_cat_open;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_rpt_val_type_sev;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_rpt_fw_cat;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_rpt_blocking_open;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_rpt_auto_fix_pend;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_org_critical_open;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_run_severity;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_run_category;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_resolution_method;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_vr_false_positive;

-- Config & dashboard composite indexes
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rc_org_fw_active;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rc_fw_auto_gen;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_rc_org_assurance;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dv_org_type_default;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dv_org_public_active;
DROP INDEX IF EXISTS pack030_nz_reporting.idx_p030_dv_creator_active;

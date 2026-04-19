-- =============================================================================
-- V219: PACK-030 Net Zero Reporting Pack - Performance Indexes
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    009 of 015
-- Date:         March 2026
--
-- Comprehensive composite and conditional performance indexes across all
-- PACK-030 tables for production-grade query performance. Supplements the
-- per-table indexes created in V211-V218 with cross-table query optimizations.
--
-- Indexes: 350+ total (per-table created earlier + composite indexes here)
--
-- Previous: V218__PACK030_config_tables.sql
-- =============================================================================

-- =============================================================================
-- Reports: Multi-Framework & Cross-Year Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_rpt_org_fw_yr_status    ON pack030_nz_reporting.gl_nz_reports(organization_id, framework, reporting_year, status);
CREATE INDEX idx_p030_rpt_org_fw_latest_act   ON pack030_nz_reporting.gl_nz_reports(organization_id, framework) WHERE is_latest = TRUE AND is_active = TRUE;
CREATE INDEX idx_p030_rpt_org_status_yr       ON pack030_nz_reporting.gl_nz_reports(organization_id, status, reporting_year DESC);
CREATE INDEX idx_p030_rpt_fw_status_yr        ON pack030_nz_reporting.gl_nz_reports(framework, status, reporting_year DESC);
CREATE INDEX idx_p030_rpt_org_published_fw    ON pack030_nz_reporting.gl_nz_reports(organization_id, framework, published_at DESC) WHERE status = 'PUBLISHED';
CREATE INDEX idx_p030_rpt_org_review_pending  ON pack030_nz_reporting.gl_nz_reports(organization_id) WHERE status = 'REVIEW';
CREATE INDEX idx_p030_rpt_org_draft_fw        ON pack030_nz_reporting.gl_nz_reports(organization_id, framework) WHERE status = 'DRAFT';
CREATE INDEX idx_p030_rpt_completeness_fw     ON pack030_nz_reporting.gl_nz_reports(framework, data_completeness_pct) WHERE data_completeness_pct < 100;
CREATE INDEX idx_p030_rpt_version_chain       ON pack030_nz_reporting.gl_nz_reports(previous_version_id) WHERE previous_version_id IS NOT NULL;
CREATE INDEX idx_p030_rpt_approved_pending    ON pack030_nz_reporting.gl_nz_reports(organization_id, framework) WHERE status = 'APPROVED' AND published_at IS NULL;
CREATE INDEX idx_p030_rpt_archived_yr         ON pack030_nz_reporting.gl_nz_reports(organization_id, reporting_year) WHERE archived = TRUE;
CREATE INDEX idx_p030_rpt_multi_fw            ON pack030_nz_reporting.gl_nz_reports(organization_id, reporting_year) WHERE framework = 'MULTI_FRAMEWORK';

-- =============================================================================
-- Report Sections: Content & Review Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_sec_rpt_type_lang_act   ON pack030_nz_reporting.gl_nz_report_sections(report_id, section_type, language) WHERE is_active = TRUE;
CREATE INDEX idx_p030_sec_rpt_order_act       ON pack030_nz_reporting.gl_nz_report_sections(report_id, section_order ASC) WHERE is_active = TRUE AND is_latest = TRUE;
CREATE INDEX idx_p030_sec_rpt_review_pend     ON pack030_nz_reporting.gl_nz_report_sections(report_id) WHERE review_status = 'PENDING_REVIEW';
CREATE INDEX idx_p030_sec_rpt_needs_rev       ON pack030_nz_reporting.gl_nz_report_sections(report_id) WHERE review_status = 'NEEDS_REVISION';
CREATE INDEX idx_p030_sec_rpt_low_consist     ON pack030_nz_reporting.gl_nz_report_sections(report_id) WHERE consistency_score IS NOT NULL AND consistency_score < 80;
CREATE INDEX idx_p030_sec_rpt_depth_parent    ON pack030_nz_reporting.gl_nz_report_sections(report_id, depth_level, parent_section_id);
CREATE INDEX idx_p030_sec_rpt_has_citations   ON pack030_nz_reporting.gl_nz_report_sections(report_id) WHERE citation_count > 0;
CREATE INDEX idx_p030_sec_version_chain       ON pack030_nz_reporting.gl_nz_report_sections(report_id, section_type, version_number DESC);
CREATE INDEX idx_p030_sec_translated_lang     ON pack030_nz_reporting.gl_nz_report_sections(report_id, language, source_language) WHERE is_translated = TRUE;
CREATE INDEX idx_p030_sec_word_count          ON pack030_nz_reporting.gl_nz_report_sections(report_id, word_count DESC);

-- =============================================================================
-- Report Metrics: Scope & Source Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_met_rpt_scope_name      ON pack030_nz_reporting.gl_nz_report_metrics(report_id, scope, metric_name) WHERE is_active = TRUE;
CREATE INDEX idx_p030_met_rpt_source_scope    ON pack030_nz_reporting.gl_nz_report_metrics(report_id, source_system, scope);
CREATE INDEX idx_p030_met_rpt_category        ON pack030_nz_reporting.gl_nz_report_metrics(report_id, metric_category);
CREATE INDEX idx_p030_met_rpt_pack_scope      ON pack030_nz_reporting.gl_nz_report_metrics(report_id, source_pack, scope) WHERE source_pack IS NOT NULL;
CREATE INDEX idx_p030_met_rpt_verified        ON pack030_nz_reporting.gl_nz_report_metrics(report_id) WHERE verification_status IN ('EXTERNAL_LIMITED', 'EXTERNAL_REASONABLE');
CREATE INDEX idx_p030_met_rpt_unverified      ON pack030_nz_reporting.gl_nz_report_metrics(report_id) WHERE verification_status = 'UNVERIFIED';
CREATE INDEX idx_p030_met_rpt_xbrl_tagged     ON pack030_nz_reporting.gl_nz_report_metrics(report_id) WHERE xbrl_tag IS NOT NULL;
CREATE INDEX idx_p030_met_rpt_high_uncert     ON pack030_nz_reporting.gl_nz_report_metrics(report_id) WHERE uncertainty_confidence_pct IS NOT NULL AND uncertainty_confidence_pct < 80;
CREATE INDEX idx_p030_met_fw_element          ON pack030_nz_reporting.gl_nz_report_metrics(framework_element_ref) WHERE framework_element_ref IS NOT NULL;
CREATE INDEX idx_p030_met_variance_prior      ON pack030_nz_reporting.gl_nz_report_metrics(report_id) WHERE variance_pct_from_prior IS NOT NULL AND ABS(variance_pct_from_prior) > 10;

-- =============================================================================
-- Framework Schemas: Lookup & Lifecycle Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_fs_fw_type_current      ON pack030_nz_reporting.gl_nz_framework_schemas(framework, schema_type, version) WHERE is_current = TRUE AND is_active = TRUE;
CREATE INDEX idx_p030_fs_fw_eff_date          ON pack030_nz_reporting.gl_nz_framework_schemas(framework, effective_date DESC);
CREATE INDEX idx_p030_fs_source_org           ON pack030_nz_reporting.gl_nz_framework_schemas(source_organization);
CREATE INDEX idx_p030_fs_field_count          ON pack030_nz_reporting.gl_nz_framework_schemas(framework, total_field_count DESC);
CREATE INDEX idx_p030_fs_version_chain        ON pack030_nz_reporting.gl_nz_framework_schemas(previous_version_id) WHERE previous_version_id IS NOT NULL;

-- =============================================================================
-- Framework Mappings: Cross-Framework Query Indexes
-- =============================================================================
CREATE INDEX idx_p030_fm_src_tgt_type_act     ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, target_framework, mapping_type) WHERE is_active = TRUE;
CREATE INDEX idx_p030_fm_src_met_tgt_fw       ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, source_metric, target_framework);
CREATE INDEX idx_p030_fm_tgt_met_src_fw       ON pack030_nz_reporting.gl_nz_framework_mappings(target_framework, target_metric, source_framework);
CREATE INDEX idx_p030_fm_high_confidence      ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, target_framework) WHERE confidence_score >= 90;
CREATE INDEX idx_p030_fm_low_confidence       ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, target_framework) WHERE confidence_score IS NOT NULL AND confidence_score < 70;
CREATE INDEX idx_p030_fm_needs_conversion     ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, target_framework) WHERE unit_conversion_required = TRUE;
CREATE INDEX idx_p030_fm_most_used            ON pack030_nz_reporting.gl_nz_framework_mappings(usage_count DESC) WHERE is_active = TRUE;
CREATE INDEX idx_p030_fm_last_used            ON pack030_nz_reporting.gl_nz_framework_mappings(last_used_at DESC) WHERE last_used_at IS NOT NULL;

-- =============================================================================
-- Framework Deadlines: Timeline & Status Indexes
-- =============================================================================
CREATE INDEX idx_p030_fd_org_fw_yr            ON pack030_nz_reporting.gl_nz_framework_deadlines(organization_id, framework, reporting_year);
CREATE INDEX idx_p030_fd_upcoming_30d         ON pack030_nz_reporting.gl_nz_framework_deadlines(deadline_date) WHERE deadline_date BETWEEN CURRENT_DATE AND (CURRENT_DATE + INTERVAL '30 days') AND submission_status != 'SUBMITTED';
CREATE INDEX idx_p030_fd_upcoming_90d         ON pack030_nz_reporting.gl_nz_framework_deadlines(deadline_date) WHERE deadline_date BETWEEN CURRENT_DATE AND (CURRENT_DATE + INTERVAL '90 days') AND submission_status != 'SUBMITTED';
CREATE INDEX idx_p030_fd_org_overdue          ON pack030_nz_reporting.gl_nz_framework_deadlines(organization_id, framework) WHERE deadline_date < CURRENT_DATE AND submission_status IN ('NOT_SUBMITTED', 'IN_PROGRESS');
CREATE INDEX idx_p030_fd_extended             ON pack030_nz_reporting.gl_nz_framework_deadlines(organization_id) WHERE extension_granted = TRUE;
CREATE INDEX idx_p030_fd_submitted_yr         ON pack030_nz_reporting.gl_nz_framework_deadlines(organization_id, reporting_year) WHERE submission_status = 'SUBMITTED';

-- =============================================================================
-- Narratives: Reuse & Quality Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_nar_org_fw_sec_lang     ON pack030_nz_reporting.gl_nz_narratives(organization_id, framework, section_type, language) WHERE is_active = TRUE AND is_latest = TRUE;
CREATE INDEX idx_p030_nar_org_fw_audience     ON pack030_nz_reporting.gl_nz_narratives(organization_id, framework, audience);
CREATE INDEX idx_p030_nar_org_fw_tone         ON pack030_nz_reporting.gl_nz_narratives(organization_id, framework, tone);
CREATE INDEX idx_p030_nar_org_low_consist     ON pack030_nz_reporting.gl_nz_narratives(organization_id) WHERE consistency_score IS NOT NULL AND consistency_score < 80;
CREATE INDEX idx_p030_nar_org_ai_unreviewed   ON pack030_nz_reporting.gl_nz_narratives(organization_id, framework) WHERE generation_method IN ('AI_GENERATED', 'AI_ASSISTED') AND human_reviewed = FALSE;
CREATE INDEX idx_p030_nar_org_high_usage      ON pack030_nz_reporting.gl_nz_narratives(organization_id, usage_count DESC) WHERE usage_count > 0;
CREATE INDEX idx_p030_nar_template_fw_sec     ON pack030_nz_reporting.gl_nz_narratives(framework, section_type, audience) WHERE is_template = TRUE AND is_active = TRUE;
CREATE INDEX idx_p030_nar_version_chain       ON pack030_nz_reporting.gl_nz_narratives(previous_version_id) WHERE previous_version_id IS NOT NULL;

-- =============================================================================
-- Translations: Quality & Review Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_tr_org_src_tgt_lang     ON pack030_nz_reporting.gl_nz_translations(organization_id, source_language, target_language) WHERE is_active = TRUE;
CREATE INDEX idx_p030_tr_org_type_src         ON pack030_nz_reporting.gl_nz_translations(organization_id, source_type, source_id);
CREATE INDEX idx_p030_tr_low_quality          ON pack030_nz_reporting.gl_nz_translations(organization_id) WHERE quality_score IS NOT NULL AND quality_score < 80;
CREATE INDEX idx_p030_tr_high_quality         ON pack030_nz_reporting.gl_nz_translations(organization_id, target_language) WHERE quality_score >= 95 AND is_approved = TRUE;
CREATE INDEX idx_p030_tr_citations_broken     ON pack030_nz_reporting.gl_nz_translations(organization_id) WHERE citations_preserved = FALSE;
CREATE INDEX idx_p030_tr_terminology_issue    ON pack030_nz_reporting.gl_nz_translations(organization_id) WHERE terminology_consistent = FALSE;
CREATE INDEX idx_p030_tr_machine_unreviewed   ON pack030_nz_reporting.gl_nz_translations(organization_id) WHERE translator_type = 'MACHINE' AND human_reviewed = FALSE;

-- =============================================================================
-- Assurance Evidence: Audit Readiness Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_ae_rpt_type_tier        ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id, evidence_type, evidence_tier) WHERE is_active = TRUE;
CREATE INDEX idx_p030_ae_rpt_std_level        ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id, assurance_standard, assurance_level);
CREATE INDEX idx_p030_ae_org_incomplete       ON pack030_nz_reporting.gl_nz_assurance_evidence(organization_id, report_id) WHERE completeness_pct < 100 AND is_active = TRUE;
CREATE INDEX idx_p030_ae_org_rejected         ON pack030_nz_reporting.gl_nz_assurance_evidence(organization_id) WHERE review_outcome = 'REJECTED' AND is_active = TRUE;
CREATE INDEX idx_p030_ae_bundle_order         ON pack030_nz_reporting.gl_nz_assurance_evidence(bundle_id, bundle_order ASC) WHERE bundle_id IS NOT NULL;
CREATE INDEX idx_p030_ae_methodology_missing  ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id) WHERE methodology_documented = FALSE AND evidence_type IN ('CALCULATION_SHEET', 'EMISSION_FACTOR');
CREATE INDEX idx_p030_ae_control_adequate     ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id) WHERE control_evidence_adequate = FALSE;
CREATE INDEX idx_p030_ae_high_quality         ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id) WHERE overall_quality_score >= 90;
CREATE INDEX idx_p030_ae_low_quality          ON pack030_nz_reporting.gl_nz_assurance_evidence(report_id) WHERE overall_quality_score IS NOT NULL AND overall_quality_score < 50;

-- =============================================================================
-- Audit Trail: Forensic & Compliance Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_at_org_cat_time         ON pack030_nz_reporting.gl_nz_audit_trail(organization_id, event_category, created_at DESC);
CREATE INDEX idx_p030_at_org_type_time        ON pack030_nz_reporting.gl_nz_audit_trail(organization_id, event_type, created_at DESC);
CREATE INDEX idx_p030_at_rpt_cat_time         ON pack030_nz_reporting.gl_nz_audit_trail(report_id, event_category, created_at DESC);
CREATE INDEX idx_p030_at_actor_time           ON pack030_nz_reporting.gl_nz_audit_trail(actor_id, created_at DESC);
CREATE INDEX idx_p030_at_actor_type_time      ON pack030_nz_reporting.gl_nz_audit_trail(actor_type, created_at DESC);
CREATE INDEX idx_p030_at_resource_time        ON pack030_nz_reporting.gl_nz_audit_trail(resource_type, resource_id, created_at DESC);
CREATE INDEX idx_p030_at_fw_type_time         ON pack030_nz_reporting.gl_nz_audit_trail(framework, event_type, created_at DESC) WHERE framework IS NOT NULL;
CREATE INDEX idx_p030_at_session              ON pack030_nz_reporting.gl_nz_audit_trail(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_p030_at_approval_events      ON pack030_nz_reporting.gl_nz_audit_trail(report_id, created_at DESC) WHERE event_type IN ('REPORT_APPROVED', 'REPORT_REJECTED', 'REPORT_PUBLISHED');
CREATE INDEX idx_p030_at_data_events          ON pack030_nz_reporting.gl_nz_audit_trail(report_id, created_at DESC) WHERE event_category IN ('DATA', 'METRIC');

-- =============================================================================
-- Data Lineage: Source Tracing Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_dl_rpt_src_metric       ON pack030_nz_reporting.gl_nz_data_lineage(report_id, source_system, metric_name) WHERE is_active = TRUE;
CREATE INDEX idx_p030_dl_rpt_pack_metric      ON pack030_nz_reporting.gl_nz_data_lineage(report_id, source_pack, metric_name) WHERE source_pack IS NOT NULL;
CREATE INDEX idx_p030_dl_rpt_app_metric       ON pack030_nz_reporting.gl_nz_data_lineage(report_id, source_app, metric_name) WHERE source_app IS NOT NULL;
CREATE INDEX idx_p030_dl_org_src_type         ON pack030_nz_reporting.gl_nz_data_lineage(organization_id, source_type);
CREATE INDEX idx_p030_dl_rpt_dq_tier          ON pack030_nz_reporting.gl_nz_data_lineage(report_id, data_quality_tier);
CREATE INDEX idx_p030_dl_rpt_low_conf         ON pack030_nz_reporting.gl_nz_data_lineage(report_id) WHERE confidence_level IS NOT NULL AND confidence_level < 80;
CREATE INDEX idx_p030_dl_rpt_complex_xform    ON pack030_nz_reporting.gl_nz_data_lineage(report_id) WHERE transformation_count > 3;
CREATE INDEX idx_p030_dl_rpt_period           ON pack030_nz_reporting.gl_nz_data_lineage(report_id, data_period_start, data_period_end);

-- =============================================================================
-- XBRL Tags: Taxonomy & Validation Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_xb_rpt_fw_element       ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id, taxonomy_framework, xbrl_element) WHERE is_active = TRUE;
CREATE INDEX idx_p030_xb_rpt_fw_valid         ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id, taxonomy_framework) WHERE validation_status = 'VALID';
CREATE INDEX idx_p030_xb_rpt_fw_invalid       ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id, taxonomy_framework) WHERE validation_status = 'INVALID';
CREATE INDEX idx_p030_xb_rpt_numeric          ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id) WHERE tag_type = 'NUMERIC';
CREATE INDEX idx_p030_xb_rpt_text             ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id) WHERE tag_type = 'TEXT';
CREATE INDEX idx_p030_xb_rpt_extensions       ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id, taxonomy_framework) WHERE is_extension = TRUE;
CREATE INDEX idx_p030_xb_ns_element           ON pack030_nz_reporting.gl_nz_xbrl_tags(xbrl_namespace, xbrl_element);
CREATE INDEX idx_p030_xb_context_period       ON pack030_nz_reporting.gl_nz_xbrl_tags(context_period_start, context_period_end) WHERE context_period_start IS NOT NULL;

-- =============================================================================
-- Validation Results: Issue Tracking Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_vr_rpt_sev_cat_open     ON pack030_nz_reporting.gl_nz_validation_results(report_id, severity, validation_category) WHERE resolved = FALSE;
CREATE INDEX idx_p030_vr_rpt_val_type_sev     ON pack030_nz_reporting.gl_nz_validation_results(report_id, validation_type, severity) WHERE resolved = FALSE;
CREATE INDEX idx_p030_vr_rpt_fw_cat           ON pack030_nz_reporting.gl_nz_validation_results(report_id, framework, validation_category) WHERE framework IS NOT NULL;
CREATE INDEX idx_p030_vr_rpt_blocking_open    ON pack030_nz_reporting.gl_nz_validation_results(report_id, severity) WHERE blocking = TRUE AND resolved = FALSE;
CREATE INDEX idx_p030_vr_rpt_auto_fix_pend    ON pack030_nz_reporting.gl_nz_validation_results(report_id, validation_category) WHERE auto_fixable = TRUE AND auto_fix_applied = FALSE;
CREATE INDEX idx_p030_vr_org_critical_open    ON pack030_nz_reporting.gl_nz_validation_results(organization_id) WHERE severity = 'CRITICAL' AND resolved = FALSE;
CREATE INDEX idx_p030_vr_run_severity         ON pack030_nz_reporting.gl_nz_validation_results(validation_run_id, severity);
CREATE INDEX idx_p030_vr_run_category         ON pack030_nz_reporting.gl_nz_validation_results(validation_run_id, validation_category);
CREATE INDEX idx_p030_vr_resolution_method    ON pack030_nz_reporting.gl_nz_validation_results(resolution_method) WHERE resolved = TRUE;
CREATE INDEX idx_p030_vr_false_positive       ON pack030_nz_reporting.gl_nz_validation_results(report_id) WHERE result_status = 'FALSE_POSITIVE';

-- =============================================================================
-- Config & Dashboard: Lookup Composite Indexes
-- =============================================================================
CREATE INDEX idx_p030_rc_org_fw_active        ON pack030_nz_reporting.gl_nz_report_config(organization_id, framework) WHERE is_active = TRUE;
CREATE INDEX idx_p030_rc_fw_auto_gen          ON pack030_nz_reporting.gl_nz_report_config(framework) WHERE auto_generate = TRUE AND is_active = TRUE;
CREATE INDEX idx_p030_rc_org_assurance        ON pack030_nz_reporting.gl_nz_report_config(organization_id) WHERE assurance_standard IS NOT NULL;

CREATE INDEX idx_p030_dv_org_type_default     ON pack030_nz_reporting.gl_nz_dashboard_views(organization_id, view_type) WHERE is_default = TRUE AND is_active = TRUE;
CREATE INDEX idx_p030_dv_org_public_active    ON pack030_nz_reporting.gl_nz_dashboard_views(organization_id) WHERE is_public = TRUE AND is_active = TRUE;
CREATE INDEX idx_p030_dv_creator_active       ON pack030_nz_reporting.gl_nz_dashboard_views(created_by) WHERE is_active = TRUE;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON INDEX pack030_nz_reporting.idx_p030_rpt_org_fw_yr_status IS 'Multi-column composite for filtering reports by organization, framework, year, and status.';
COMMENT ON INDEX pack030_nz_reporting.idx_p030_vr_rpt_sev_cat_open IS 'Composite for finding unresolved validation issues by severity and category.';
COMMENT ON INDEX pack030_nz_reporting.idx_p030_fd_upcoming_30d IS 'Conditional index for framework deadlines within 30 days that have not been submitted.';

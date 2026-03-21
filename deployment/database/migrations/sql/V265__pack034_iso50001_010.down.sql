-- =============================================================================
-- V265 DOWN: Drop PACK-034 audit trail, views, functions, and audit triggers
-- =============================================================================

-- Drop audit triggers on status columns
DROP TRIGGER IF EXISTS trg_p034_assess_status_audit ON pack034_iso50001.compliance_assessments;
DROP TRIGGER IF EXISTS trg_p034_ap_status_audit ON pack034_iso50001.action_plans;
DROP TRIGGER IF EXISTS trg_p034_nc_status_audit ON pack034_iso50001.nonconformities;
DROP TRIGGER IF EXISTS trg_p034_bl_status_audit ON pack034_iso50001.energy_baselines;
DROP TRIGGER IF EXISTS trg_p034_seu_status_audit ON pack034_iso50001.significant_energy_uses;
DROP TRIGGER IF EXISTS trg_p034_ems_status_audit ON pack034_iso50001.energy_management_systems;

-- Drop audit trigger function
DROP FUNCTION IF EXISTS pack034_iso50001.fn_audit_status_change();

-- Drop utility functions
DROP FUNCTION IF EXISTS pack034_iso50001.fn_update_compliance_score(UUID);
DROP FUNCTION IF EXISTS pack034_iso50001.fn_check_cusum_alert(UUID);
DROP FUNCTION IF EXISTS pack034_iso50001.fn_calculate_enpi_improvement(UUID);

-- Drop views
DROP VIEW IF EXISTS pack034_iso50001.v_action_plan_progress;
DROP VIEW IF EXISTS pack034_iso50001.v_compliance_summary;
DROP VIEW IF EXISTS pack034_iso50001.v_cusum_status;
DROP VIEW IF EXISTS pack034_iso50001.v_enpi_performance;
DROP VIEW IF EXISTS pack034_iso50001.v_seu_pareto;
DROP VIEW IF EXISTS pack034_iso50001.v_enms_overview;

-- Drop RLS policies on audit trail
DROP POLICY IF EXISTS p034_audit_service_bypass ON pack034_iso50001.pack034_audit_trail;
DROP POLICY IF EXISTS p034_audit_tenant_isolation ON pack034_iso50001.pack034_audit_trail;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.pack034_audit_trail DISABLE ROW LEVEL SECURITY;

-- Drop audit trail table
DROP TABLE IF EXISTS pack034_iso50001.pack034_audit_trail CASCADE;

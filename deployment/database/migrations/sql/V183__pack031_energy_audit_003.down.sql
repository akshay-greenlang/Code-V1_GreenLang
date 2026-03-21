-- =============================================================================
-- V183 DOWN: Drop energy audit tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_checklist_service_bypass ON pack031_energy_audit.en16247_checklists;
DROP POLICY IF EXISTS p031_checklist_tenant_isolation ON pack031_energy_audit.en16247_checklists;
DROP POLICY IF EXISTS p031_enduse_service_bypass ON pack031_energy_audit.energy_end_uses;
DROP POLICY IF EXISTS p031_enduse_tenant_isolation ON pack031_energy_audit.energy_end_uses;
DROP POLICY IF EXISTS p031_finding_service_bypass ON pack031_energy_audit.audit_findings;
DROP POLICY IF EXISTS p031_finding_tenant_isolation ON pack031_energy_audit.audit_findings;
DROP POLICY IF EXISTS p031_audit_service_bypass ON pack031_energy_audit.energy_audits;
DROP POLICY IF EXISTS p031_audit_tenant_isolation ON pack031_energy_audit.energy_audits;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.en16247_checklists DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_end_uses DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.audit_findings DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_audits DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p031_audit_updated ON pack031_energy_audit.energy_audits;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.en16247_checklists CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_end_uses CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.audit_findings CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_audits CASCADE;

-- =============================================================================
-- V262 DOWN: Drop PACK-034 compliance tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_cora_service_bypass ON pack034_iso50001.corrective_actions;
DROP POLICY IF EXISTS p034_cora_tenant_isolation ON pack034_iso50001.corrective_actions;
DROP POLICY IF EXISTS p034_nc_service_bypass ON pack034_iso50001.nonconformities;
DROP POLICY IF EXISTS p034_nc_tenant_isolation ON pack034_iso50001.nonconformities;
DROP POLICY IF EXISTS p034_cs_service_bypass ON pack034_iso50001.clause_scores;
DROP POLICY IF EXISTS p034_cs_tenant_isolation ON pack034_iso50001.clause_scores;
DROP POLICY IF EXISTS p034_assess_service_bypass ON pack034_iso50001.compliance_assessments;
DROP POLICY IF EXISTS p034_assess_tenant_isolation ON pack034_iso50001.compliance_assessments;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.corrective_actions DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.nonconformities DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.clause_scores DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.compliance_assessments DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_cora_updated ON pack034_iso50001.corrective_actions;
DROP TRIGGER IF EXISTS trg_p034_nc_updated ON pack034_iso50001.nonconformities;
DROP TRIGGER IF EXISTS trg_p034_cs_updated ON pack034_iso50001.clause_scores;
DROP TRIGGER IF EXISTS trg_p034_assess_updated ON pack034_iso50001.compliance_assessments;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.corrective_actions CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.nonconformities CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.clause_scores CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.compliance_assessments CASCADE;

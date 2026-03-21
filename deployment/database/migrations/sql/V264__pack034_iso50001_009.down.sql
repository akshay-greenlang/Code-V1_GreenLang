-- =============================================================================
-- V264 DOWN: Drop PACK-034 performance tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_vr_service_bypass ON pack034_iso50001.verification_results;
DROP POLICY IF EXISTS p034_vr_tenant_isolation ON pack034_iso50001.verification_results;
DROP POLICY IF EXISTS p034_ta_service_bypass ON pack034_iso50001.trend_analyses;
DROP POLICY IF EXISTS p034_ta_tenant_isolation ON pack034_iso50001.trend_analyses;
DROP POLICY IF EXISTS p034_pr_service_bypass ON pack034_iso50001.performance_reports;
DROP POLICY IF EXISTS p034_pr_tenant_isolation ON pack034_iso50001.performance_reports;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.verification_results DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.trend_analyses DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.performance_reports DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_vr_updated ON pack034_iso50001.verification_results;
DROP TRIGGER IF EXISTS trg_p034_ta_updated ON pack034_iso50001.trend_analyses;
DROP TRIGGER IF EXISTS trg_p034_pr_updated ON pack034_iso50001.performance_reports;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.verification_results CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.trend_analyses CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.performance_reports CASCADE;

-- =============================================================================
-- V200 DOWN: Drop PACK-032 audit trail, compliance, and views
-- =============================================================================

-- Drop views first (no FK dependencies)
DROP VIEW IF EXISTS pack032_building_assessment.v_compliance_dashboard;
DROP VIEW IF EXISTS pack032_building_assessment.v_retrofit_portfolio;
DROP VIEW IF EXISTS pack032_building_assessment.v_portfolio_benchmarks;
DROP VIEW IF EXISTS pack032_building_assessment.v_building_performance_summary;

-- Drop RLS policies
DROP POLICY IF EXISTS p032_cr_service_bypass ON pack032_building_assessment.compliance_records;
DROP POLICY IF EXISTS p032_cr_tenant_isolation ON pack032_building_assessment.compliance_records;
DROP POLICY IF EXISTS p032_audit_service_bypass ON pack032_building_assessment.pack032_audit_trail;
DROP POLICY IF EXISTS p032_audit_tenant_isolation ON pack032_building_assessment.pack032_audit_trail;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.compliance_records DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.pack032_audit_trail DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_cr_updated ON pack032_building_assessment.compliance_records;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack032_building_assessment.compliance_records CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.pack032_audit_trail CASCADE;

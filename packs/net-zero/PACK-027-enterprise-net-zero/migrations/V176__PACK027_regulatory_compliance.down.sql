-- =============================================================================
-- V176 DOWN: Drop gl_regulatory_filings and gl_compliance_gaps tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_cg_service_bypass ON pack027_enterprise_net_zero.gl_compliance_gaps;
DROP POLICY IF EXISTS p027_cg_tenant_isolation ON pack027_enterprise_net_zero.gl_compliance_gaps;
DROP POLICY IF EXISTS p027_rf_service_bypass ON pack027_enterprise_net_zero.gl_regulatory_filings;
DROP POLICY IF EXISTS p027_rf_tenant_isolation ON pack027_enterprise_net_zero.gl_regulatory_filings;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_compliance_gaps DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_regulatory_filings DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p027_compliance_gaps_updated ON pack027_enterprise_net_zero.gl_compliance_gaps;
DROP TRIGGER IF EXISTS trg_p027_regulatory_filings_updated ON pack027_enterprise_net_zero.gl_regulatory_filings;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_compliance_gaps CASCADE;
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_regulatory_filings CASCADE;

-- =============================================================================
-- V177 DOWN: Drop gl_assurance_engagements and gl_assurance_workpapers tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_aw_service_bypass ON pack027_enterprise_net_zero.gl_assurance_workpapers;
DROP POLICY IF EXISTS p027_aw_tenant_isolation ON pack027_enterprise_net_zero.gl_assurance_workpapers;
DROP POLICY IF EXISTS p027_ae_service_bypass ON pack027_enterprise_net_zero.gl_assurance_engagements;
DROP POLICY IF EXISTS p027_ae_tenant_isolation ON pack027_enterprise_net_zero.gl_assurance_engagements;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_assurance_workpapers DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_assurance_engagements DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p027_assurance_workpapers_updated ON pack027_enterprise_net_zero.gl_assurance_workpapers;
DROP TRIGGER IF EXISTS trg_p027_assurance_engagements_updated ON pack027_enterprise_net_zero.gl_assurance_engagements;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_assurance_workpapers CASCADE;
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_assurance_engagements CASCADE;

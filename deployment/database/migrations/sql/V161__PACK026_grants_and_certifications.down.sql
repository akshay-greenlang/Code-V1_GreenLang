-- =============================================================================
-- V161 DOWN: Drop grant_programs, grant_applications, and certifications tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p026_cert_service_bypass ON pack026_sme_net_zero.certifications;
DROP POLICY IF EXISTS p026_cert_tenant_isolation ON pack026_sme_net_zero.certifications;
DROP POLICY IF EXISTS p026_ga_service_bypass ON pack026_sme_net_zero.grant_applications;
DROP POLICY IF EXISTS p026_ga_tenant_isolation ON pack026_sme_net_zero.grant_applications;

-- Disable RLS
ALTER TABLE IF EXISTS pack026_sme_net_zero.certifications DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack026_sme_net_zero.grant_applications DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p026_certifications_updated ON pack026_sme_net_zero.certifications;
DROP TRIGGER IF EXISTS trg_p026_grant_applications_updated ON pack026_sme_net_zero.grant_applications;
DROP TRIGGER IF EXISTS trg_p026_grant_programs_updated ON pack026_sme_net_zero.grant_programs;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack026_sme_net_zero.certifications CASCADE;
DROP TABLE IF EXISTS pack026_sme_net_zero.grant_applications CASCADE;
DROP TABLE IF EXISTS pack026_sme_net_zero.grant_programs CASCADE;

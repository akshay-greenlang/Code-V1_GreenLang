-- =============================================================================
-- V163 DOWN: Drop annual_reviews and quarterly_snapshots tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p026_qs_service_bypass ON pack026_sme_net_zero.quarterly_snapshots;
DROP POLICY IF EXISTS p026_qs_tenant_isolation ON pack026_sme_net_zero.quarterly_snapshots;
DROP POLICY IF EXISTS p026_ar_service_bypass ON pack026_sme_net_zero.annual_reviews;
DROP POLICY IF EXISTS p026_ar_tenant_isolation ON pack026_sme_net_zero.annual_reviews;

-- Disable RLS
ALTER TABLE IF EXISTS pack026_sme_net_zero.quarterly_snapshots DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack026_sme_net_zero.annual_reviews DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p026_quarterly_snapshots_updated ON pack026_sme_net_zero.quarterly_snapshots;
DROP TRIGGER IF EXISTS trg_p026_annual_reviews_updated ON pack026_sme_net_zero.annual_reviews;

-- Drop tables
DROP TABLE IF EXISTS pack026_sme_net_zero.quarterly_snapshots CASCADE;
DROP TABLE IF EXISTS pack026_sme_net_zero.annual_reviews CASCADE;

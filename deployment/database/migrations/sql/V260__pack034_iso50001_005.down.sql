-- =============================================================================
-- V260 DOWN: Drop PACK-034 CUSUM tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_ca_service_bypass ON pack034_iso50001.cusum_alerts;
DROP POLICY IF EXISTS p034_ca_tenant_isolation ON pack034_iso50001.cusum_alerts;
DROP POLICY IF EXISTS p034_cdp_service_bypass ON pack034_iso50001.cusum_data_points;
DROP POLICY IF EXISTS p034_cdp_tenant_isolation ON pack034_iso50001.cusum_data_points;
DROP POLICY IF EXISTS p034_cm_service_bypass ON pack034_iso50001.cusum_monitors;
DROP POLICY IF EXISTS p034_cm_tenant_isolation ON pack034_iso50001.cusum_monitors;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.cusum_alerts DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.cusum_data_points DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.cusum_monitors DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_ca_updated ON pack034_iso50001.cusum_alerts;
DROP TRIGGER IF EXISTS trg_p034_cdp_updated ON pack034_iso50001.cusum_data_points;
DROP TRIGGER IF EXISTS trg_p034_cm_updated ON pack034_iso50001.cusum_monitors;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.cusum_alerts CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.cusum_data_points CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.cusum_monitors CASCADE;

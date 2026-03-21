-- =============================================================================
-- V175 DOWN: Drop gl_climate_risks and gl_asset_risk_exposure tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_are_service_bypass ON pack027_enterprise_net_zero.gl_asset_risk_exposure;
DROP POLICY IF EXISTS p027_are_tenant_isolation ON pack027_enterprise_net_zero.gl_asset_risk_exposure;
DROP POLICY IF EXISTS p027_cr_service_bypass ON pack027_enterprise_net_zero.gl_climate_risks;
DROP POLICY IF EXISTS p027_cr_tenant_isolation ON pack027_enterprise_net_zero.gl_climate_risks;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_asset_risk_exposure DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_climate_risks DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p027_asset_risk_updated ON pack027_enterprise_net_zero.gl_asset_risk_exposure;
DROP TRIGGER IF EXISTS trg_p027_climate_risks_updated ON pack027_enterprise_net_zero.gl_climate_risks;

-- Drop tables
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_asset_risk_exposure CASCADE;
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_climate_risks CASCADE;

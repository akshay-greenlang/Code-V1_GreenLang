-- =============================================================================
-- V179 DOWN: Drop gl_data_quality_scores table
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_dq_service_bypass ON pack027_enterprise_net_zero.gl_data_quality_scores;
DROP POLICY IF EXISTS p027_dq_tenant_isolation ON pack027_enterprise_net_zero.gl_data_quality_scores;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_data_quality_scores DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p027_data_quality_updated ON pack027_enterprise_net_zero.gl_data_quality_scores;

-- Drop table
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_data_quality_scores CASCADE;

-- =============================================================================
-- V208 DOWN: Drop gl_trend_forecasts
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_tf_service_bypass ON pack029_interim_targets.gl_trend_forecasts;
DROP POLICY IF EXISTS p029_tf_tenant_isolation ON pack029_interim_targets.gl_trend_forecasts;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_trend_forecasts DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_trend_forecasts_updated ON pack029_interim_targets.gl_trend_forecasts;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_trend_forecasts CASCADE;

-- =============================================================================
-- V199 DOWN: Drop gl_actual_performance
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_perf_service_bypass ON pack029_interim_targets.gl_actual_performance;
DROP POLICY IF EXISTS p029_perf_tenant_isolation ON pack029_interim_targets.gl_actual_performance;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_actual_performance DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_actual_performance_updated ON pack029_interim_targets.gl_actual_performance;

-- Drop table (hypertable)
DROP TABLE IF EXISTS pack029_interim_targets.gl_actual_performance CASCADE;

-- =============================================================================
-- V263 DOWN: Drop PACK-034 monitoring tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_mr_service_bypass ON pack034_iso50001.meter_readings;
DROP POLICY IF EXISTS p034_mr_tenant_isolation ON pack034_iso50001.meter_readings;
DROP POLICY IF EXISTS p034_ms_service_bypass ON pack034_iso50001.monitoring_schedules;
DROP POLICY IF EXISTS p034_ms_tenant_isolation ON pack034_iso50001.monitoring_schedules;
DROP POLICY IF EXISTS p034_mh_service_bypass ON pack034_iso50001.meter_hierarchy;
DROP POLICY IF EXISTS p034_mh_tenant_isolation ON pack034_iso50001.meter_hierarchy;
DROP POLICY IF EXISTS p034_mp_service_bypass ON pack034_iso50001.metering_plans;
DROP POLICY IF EXISTS p034_mp_tenant_isolation ON pack034_iso50001.metering_plans;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.meter_readings DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.monitoring_schedules DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.meter_hierarchy DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.metering_plans DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_mr_updated ON pack034_iso50001.meter_readings;
DROP TRIGGER IF EXISTS trg_p034_ms_updated ON pack034_iso50001.monitoring_schedules;
DROP TRIGGER IF EXISTS trg_p034_mh_updated ON pack034_iso50001.meter_hierarchy;
DROP TRIGGER IF EXISTS trg_p034_mp_updated ON pack034_iso50001.metering_plans;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.meter_readings CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.monitoring_schedules CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.meter_hierarchy CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.metering_plans CASCADE;

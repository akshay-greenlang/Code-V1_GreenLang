-- =============================================================================
-- V266 DOWN: Drop PACK-035 schema, facility profiles, metering points
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p035_mp_service_bypass ON pack035_energy_benchmark.metering_points;
DROP POLICY IF EXISTS p035_mp_tenant_isolation ON pack035_energy_benchmark.metering_points;
DROP POLICY IF EXISTS p035_fp_service_bypass ON pack035_energy_benchmark.facility_profiles;
DROP POLICY IF EXISTS p035_fp_tenant_isolation ON pack035_energy_benchmark.facility_profiles;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.metering_points DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.facility_profiles DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p035_fp_updated ON pack035_energy_benchmark.facility_profiles;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.metering_points CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.facility_profiles CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack035_energy_benchmark.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack035_energy_benchmark CASCADE;

-- =============================================================================
-- V256 DOWN: Drop PACK-034 schema, core EnMS tables, triggers, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_bounds_service_bypass ON pack034_iso50001.enms_boundaries;
DROP POLICY IF EXISTS p034_bounds_tenant_isolation ON pack034_iso50001.enms_boundaries;
DROP POLICY IF EXISTS p034_scope_service_bypass ON pack034_iso50001.enms_scope;
DROP POLICY IF EXISTS p034_scope_tenant_isolation ON pack034_iso50001.enms_scope;
DROP POLICY IF EXISTS p034_ems_service_bypass ON pack034_iso50001.energy_management_systems;
DROP POLICY IF EXISTS p034_ems_tenant_isolation ON pack034_iso50001.energy_management_systems;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.enms_boundaries DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.enms_scope DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.energy_management_systems DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_bounds_updated ON pack034_iso50001.enms_boundaries;
DROP TRIGGER IF EXISTS trg_p034_scope_updated ON pack034_iso50001.enms_scope;
DROP TRIGGER IF EXISTS trg_p034_ems_updated ON pack034_iso50001.energy_management_systems;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.enms_boundaries CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.enms_scope CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.energy_management_systems CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack034_iso50001.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack034_iso50001 CASCADE;

-- =============================================================================
-- V181 DOWN: Drop PACK-031 schema, facility tables, trigger, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_contact_service_bypass ON pack031_energy_audit.facility_contacts;
DROP POLICY IF EXISTS p031_contact_tenant_isolation ON pack031_energy_audit.facility_contacts;
DROP POLICY IF EXISTS p031_carrier_service_bypass ON pack031_energy_audit.energy_carriers;
DROP POLICY IF EXISTS p031_carrier_tenant_isolation ON pack031_energy_audit.energy_carriers;
DROP POLICY IF EXISTS p031_fac_service_bypass ON pack031_energy_audit.energy_audit_facilities;
DROP POLICY IF EXISTS p031_fac_tenant_isolation ON pack031_energy_audit.energy_audit_facilities;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.facility_contacts DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_carriers DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_audit_facilities DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p031_fac_updated ON pack031_energy_audit.energy_audit_facilities;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.facility_contacts CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_carriers CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_audit_facilities CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack031_energy_audit.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack031_energy_audit CASCADE;

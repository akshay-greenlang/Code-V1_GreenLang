-- =============================================================================
-- V190 DOWN: Drop process mapping, audit trail, and views
-- =============================================================================

-- Drop views first
DROP VIEW IF EXISTS pack031_energy_audit.v_compliance_dashboard CASCADE;
DROP VIEW IF EXISTS pack031_energy_audit.v_equipment_efficiency_gaps CASCADE;
DROP VIEW IF EXISTS pack031_energy_audit.v_savings_portfolio CASCADE;
DROP VIEW IF EXISTS pack031_energy_audit.v_facility_energy_summary CASCADE;

-- Drop RLS policies
DROP POLICY IF EXISTS p031_trail_service_bypass ON pack031_energy_audit.pack031_audit_trail;
DROP POLICY IF EXISTS p031_trail_tenant_isolation ON pack031_energy_audit.pack031_audit_trail;
DROP POLICY IF EXISTS p031_flow_service_bypass ON pack031_energy_audit.energy_flows;
DROP POLICY IF EXISTS p031_flow_tenant_isolation ON pack031_energy_audit.energy_flows;
DROP POLICY IF EXISTS p031_node_service_bypass ON pack031_energy_audit.process_nodes;
DROP POLICY IF EXISTS p031_node_tenant_isolation ON pack031_energy_audit.process_nodes;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.pack031_audit_trail DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_flows DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.process_nodes DISABLE ROW LEVEL SECURITY;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.pack031_audit_trail CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_flows CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.process_nodes CASCADE;

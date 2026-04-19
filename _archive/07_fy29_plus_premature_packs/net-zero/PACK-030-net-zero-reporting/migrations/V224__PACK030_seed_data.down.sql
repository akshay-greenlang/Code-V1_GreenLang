-- =============================================================================
-- V224 DOWN: Remove PACK-030 seed data
-- =============================================================================

-- Remove framework deadlines seed data
DELETE FROM pack030_nz_reporting.gl_nz_framework_deadlines
WHERE tenant_id = '00000000-0000-0000-0000-000000000001'::UUID;

-- Remove framework mappings seed data
DELETE FROM pack030_nz_reporting.gl_nz_framework_mappings
WHERE tenant_id = '00000000-0000-0000-0000-000000000001'::UUID;

-- Remove framework schemas seed data
DELETE FROM pack030_nz_reporting.gl_nz_framework_schemas
WHERE tenant_id = '00000000-0000-0000-0000-000000000001'::UUID;

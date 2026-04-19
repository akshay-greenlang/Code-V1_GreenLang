-- =============================================================================
-- V181 DOWN: Drop PACK-028 schema, gl_sector_classifications, trigger, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p028_sc_service_bypass ON pack028_sector_pathway.gl_sector_classifications;
DROP POLICY IF EXISTS p028_sc_tenant_isolation ON pack028_sector_pathway.gl_sector_classifications;

-- Disable RLS
ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_classifications DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p028_sector_classifications_updated ON pack028_sector_pathway.gl_sector_classifications;

-- Drop table
DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_classifications CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack028_sector_pathway.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack028_sector_pathway CASCADE;

-- =============================================================================
-- V153 DOWN: Drop sector_pathways and sector_alignment tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_sa_service_bypass ON pack025_race_to_zero.sector_alignment;
DROP POLICY IF EXISTS p025_sa_tenant_isolation ON pack025_race_to_zero.sector_alignment;
DROP POLICY IF EXISTS p025_spw_service_write ON pack025_race_to_zero.sector_pathways;
DROP POLICY IF EXISTS p025_spw_read_all ON pack025_race_to_zero.sector_pathways;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.sector_alignment DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.sector_pathways DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_sector_alignment_updated ON pack025_race_to_zero.sector_alignment;
DROP TRIGGER IF EXISTS trg_p025_sector_pathways_updated ON pack025_race_to_zero.sector_pathways;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack025_race_to_zero.sector_alignment CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.sector_pathways CASCADE;

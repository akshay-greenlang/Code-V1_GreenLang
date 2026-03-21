-- =============================================================================
-- V150 DOWN: Drop interim_targets and emission_pathways tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_ep_service_bypass ON pack025_race_to_zero.emission_pathways;
DROP POLICY IF EXISTS p025_ep_tenant_isolation ON pack025_race_to_zero.emission_pathways;
DROP POLICY IF EXISTS p025_it_service_bypass ON pack025_race_to_zero.interim_targets;
DROP POLICY IF EXISTS p025_it_tenant_isolation ON pack025_race_to_zero.interim_targets;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.emission_pathways DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.interim_targets DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_emission_pathways_updated ON pack025_race_to_zero.emission_pathways;
DROP TRIGGER IF EXISTS trg_p025_interim_targets_updated ON pack025_race_to_zero.interim_targets;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack025_race_to_zero.emission_pathways CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.interim_targets CASCADE;

-- =============================================================================
-- V186 DOWN: Drop gl_sector_abatement_levers
-- =============================================================================

DROP POLICY IF EXISTS p028_al_service_bypass ON pack028_sector_pathway.gl_sector_abatement_levers;
DROP POLICY IF EXISTS p028_al_tenant_isolation ON pack028_sector_pathway.gl_sector_abatement_levers;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_abatement_levers DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_abatement_levers_updated ON pack028_sector_pathway.gl_sector_abatement_levers;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_abatement_levers CASCADE;

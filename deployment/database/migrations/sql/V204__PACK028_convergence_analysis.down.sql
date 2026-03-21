-- =============================================================================
-- V184 DOWN: Drop gl_sector_convergence
-- =============================================================================

DROP POLICY IF EXISTS p028_cv_service_bypass ON pack028_sector_pathway.gl_sector_convergence;
DROP POLICY IF EXISTS p028_cv_tenant_isolation ON pack028_sector_pathway.gl_sector_convergence;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_convergence DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_sector_convergence_updated ON pack028_sector_pathway.gl_sector_convergence;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_convergence CASCADE;

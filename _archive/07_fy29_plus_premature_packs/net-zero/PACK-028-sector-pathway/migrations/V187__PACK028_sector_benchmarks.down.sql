-- =============================================================================
-- V187 DOWN: Drop gl_sector_benchmarks
-- =============================================================================

DROP POLICY IF EXISTS p028_bm_service_bypass ON pack028_sector_pathway.gl_sector_benchmarks;
DROP POLICY IF EXISTS p028_bm_tenant_isolation ON pack028_sector_pathway.gl_sector_benchmarks;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_benchmarks DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_sector_benchmarks_updated ON pack028_sector_pathway.gl_sector_benchmarks;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_benchmarks CASCADE;

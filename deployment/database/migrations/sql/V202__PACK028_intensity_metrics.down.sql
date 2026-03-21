-- =============================================================================
-- V182 DOWN: Drop gl_sector_intensity_metrics
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p028_im_service_bypass ON pack028_sector_pathway.gl_sector_intensity_metrics;
DROP POLICY IF EXISTS p028_im_tenant_isolation ON pack028_sector_pathway.gl_sector_intensity_metrics;

-- Disable RLS
ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_intensity_metrics DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p028_intensity_metrics_updated ON pack028_sector_pathway.gl_sector_intensity_metrics;

-- Drop table
DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_intensity_metrics CASCADE;

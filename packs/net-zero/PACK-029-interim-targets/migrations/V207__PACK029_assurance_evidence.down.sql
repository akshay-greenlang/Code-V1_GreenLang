-- =============================================================================
-- V207 DOWN: Drop gl_assurance_evidence
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_ae_service_bypass ON pack029_interim_targets.gl_assurance_evidence;
DROP POLICY IF EXISTS p029_ae_tenant_isolation ON pack029_interim_targets.gl_assurance_evidence;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_assurance_evidence DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_assurance_evidence_updated ON pack029_interim_targets.gl_assurance_evidence;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_assurance_evidence CASCADE;

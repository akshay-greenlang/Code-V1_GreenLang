-- =============================================================================
-- V145 Rollback: Drop Verification Packages
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_pkg_updated_at ON pack024_carbon_neutral.pack024_verification_packages;
DROP TRIGGER IF EXISTS trg_pack024_doc_updated_at ON pack024_carbon_neutral.pack024_package_documentation;
DROP TRIGGER IF EXISTS trg_pack024_finding_updated_at ON pack024_carbon_neutral.pack024_package_review_findings;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_package_audit_trail CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_package_review_findings CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_package_documentation CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_verification_packages CASCADE;

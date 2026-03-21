-- =============================================================================
-- V131 Down: Rollback Scope 3 Screening and Coverage Tracking
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_supp_updated_at ON pack023_sbti_alignment.pack023_scope3_supplier_engagement;
DROP TRIGGER IF EXISTS trg_pk_cov_updated_at ON pack023_sbti_alignment.pack023_scope3_coverage_analysis;
DROP TRIGGER IF EXISTS trg_pk_cat_updated_at ON pack023_sbti_alignment.pack023_scope3_category_details;
DROP TRIGGER IF EXISTS trg_pk_s3_updated_at ON pack023_sbti_alignment.pack023_scope3_materiality_screening;

-- Drop hypertables
SELECT drop_chunks('pack023_sbti_alignment.pack023_scope3_materiality_screening', INTERVAL '1 day');
SELECT detach_hypertable('pack023_sbti_alignment.pack023_scope3_materiality_screening', if_exists => TRUE);

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_scope3_supplier_engagement;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_scope3_coverage_analysis;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_scope3_category_details;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_scope3_materiality_screening;

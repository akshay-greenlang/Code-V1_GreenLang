-- =============================================================================
-- V132 Down: Rollback SDA Sector Convergence Pathway Records
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_mile_updated_at ON pack023_sbti_alignment.pack023_sda_annual_milestones;
DROP TRIGGER IF EXISTS trg_pk_bench_updated_at ON pack023_sbti_alignment.pack023_sda_sector_benchmarks;
DROP TRIGGER IF EXISTS trg_pk_sda_updated_at ON pack023_sbti_alignment.pack023_sda_sector_pathways;

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sda_annual_milestones;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sda_sector_benchmarks;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sda_sector_pathways;

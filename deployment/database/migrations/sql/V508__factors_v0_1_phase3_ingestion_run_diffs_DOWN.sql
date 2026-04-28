-- =============================================================================
-- V508: GreenLang Factors v0.1 - Phase 3 Ingestion Run Diffs (DOWN)
-- =============================================================================
-- Description: Reverse of V508 FORWARD. Drops the indexes, table, and ENUM
--              in reverse dependency order. Idempotent (`IF EXISTS`).
--
--              Ordering note: V508 must be DROPped BEFORE V507 because the
--              table FK from `ingestion_run_diffs.run_id` references
--              `ingestion_runs.run_id`. The alembic chain (0009 down before
--              0008 down) enforces this.
--
-- Authority:
--   - V508 FORWARD: V508__factors_v0_1_phase3_ingestion_run_diffs.sql
--
-- Wave: Phase 3 / Wave 1.0
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-28
-- =============================================================================

SET search_path TO factors_v0_1, public;

DROP INDEX IF EXISTS factors_v0_1.ingestion_run_diffs_factor_urn_idx;
DROP INDEX IF EXISTS factors_v0_1.ingestion_run_diffs_run_kind_idx;

DROP TABLE IF EXISTS factors_v0_1.ingestion_run_diffs;

DROP TYPE IF EXISTS factors_v0_1.ingestion_diff_kind;

-- =============================================================================
-- End V508 DOWN
-- =============================================================================

-- =============================================================================
-- V507: GreenLang Factors v0.1 - Phase 3 Ingestion Runs (DOWN)
-- =============================================================================
-- Description: Reverse of V507 FORWARD. Drops the trigger, trigger function,
--              indexes, table, and ENUM in reverse dependency order. Uses
--              IF EXISTS throughout so the script is idempotent and safe to
--              re-run on a partially-rolled-back schema.
--
--              IMPORTANT: dropping `factors_v0_1.ingestion_runs` cascades
--              the CASCADE-FK from `factors_v0_1.ingestion_run_diffs`, so
--              this DOWN migration MUST run AFTER V508 DOWN. Operator
--              runbook enforces ordering via alembic 0009 -> 0008.
--
-- Authority:
--   - V507 FORWARD: V507__factors_v0_1_phase3_ingestion_runs.sql
--   - PHASE_3_PLAN.md §"Run-status enum (formal)"
--
-- Wave: Phase 3 / Wave 1.0
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-28
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- Drop trigger first; the function it invokes is then unreferenced.
DROP TRIGGER IF EXISTS ingestion_runs_last_updated_at_trg
    ON factors_v0_1.ingestion_runs;

DROP FUNCTION IF EXISTS factors_v0_1.ingestion_runs_set_last_updated_at();

-- Indexes ride with the table on DROP TABLE, but we drop them explicitly
-- so a partially-rolled-back schema (table already gone, indexes lingering)
-- still cleans up.
DROP INDEX IF EXISTS factors_v0_1.ingestion_runs_started_at_desc_idx;
DROP INDEX IF EXISTS factors_v0_1.ingestion_runs_batch_id_idx;
DROP INDEX IF EXISTS factors_v0_1.ingestion_runs_status_idx;
DROP INDEX IF EXISTS factors_v0_1.ingestion_runs_source_urn_idx;

DROP TABLE IF EXISTS factors_v0_1.ingestion_runs;

-- ENUM is dropped last (no longer referenced after the table is gone).
DROP TYPE IF EXISTS factors_v0_1.ingestion_run_status;

-- =============================================================================
-- End V507 DOWN
-- =============================================================================

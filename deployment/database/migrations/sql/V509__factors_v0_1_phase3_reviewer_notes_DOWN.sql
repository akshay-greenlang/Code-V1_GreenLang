-- =============================================================================
-- V509: GreenLang Factors v0.1 - Phase 3 source_artifacts.reviewer_notes (DOWN)
-- =============================================================================
-- Description: Reverse of V509 FORWARD. Drops the `reviewer_notes` JSONB
--              column from `factors_v0_1.source_artifacts`. Idempotent:
--              wrapped in a DO $$ block that checks
--              information_schema.columns first so applying the down
--              migration twice (or against a database that never had the
--              column) is a no-op.
--
--              Note: this drops the entire JSONB payload. If a prior run
--              recorded reviewer sign-offs or manual-correction trails in
--              this column, they MUST be exported to an out-of-band audit
--              store before this migration runs in production.
--
-- Authority:
--   - V509 FORWARD: V509__factors_v0_1_phase3_reviewer_notes.sql
--
-- Wave: Phase 3 / Wave 2.5
-- Postgres target: 16+
-- Author: GL-DataIntegrationEngineer
-- Created: 2026-04-28
-- =============================================================================

SET search_path TO factors_v0_1, public;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'reviewer_notes'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts
            DROP COLUMN reviewer_notes;
    END IF;
END
$$;

-- =============================================================================
-- End V509 DOWN
-- =============================================================================

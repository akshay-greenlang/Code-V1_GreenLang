-- =============================================================================
-- V505: GreenLang Factors v0.1 Phase 2 — Aliases + Source Artifacts (REVERSE)
-- =============================================================================
-- Description: Exact inverse of V505__factors_v0_1_phase2_aliases_artifacts.sql.
--              Drops `source_artifacts` and `factor_aliases`. Does NOT
--              touch the geography CHECK constraints — those are owned by
--              V501_additive (a peer migration that lives on its own
--              upgrade/downgrade boundary).
--
--              Drop order respects dependencies: source_artifacts is
--              referenced by V503.provenance_edges (FK), so V503 must be
--              downgraded first; factor_aliases has no inbound FKs.
--
-- Phase: 2 / TaskCreate #7 / WS7-T1
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Drop source_artifacts (referenced by V503.provenance_edges via FK; that
--    migration must already be downgraded for this to succeed).
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS factors_v0_1.source_artifacts_version_idx;
DROP INDEX IF EXISTS factors_v0_1.source_artifacts_source_idx;
DROP TABLE IF EXISTS factors_v0_1.source_artifacts;

-- -----------------------------------------------------------------------------
-- 2. Drop factor_aliases.
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS factors_v0_1.factor_aliases_urn_idx;
DROP TABLE IF EXISTS factors_v0_1.factor_aliases;

-- =============================================================================
-- End V505 DOWN
-- =============================================================================

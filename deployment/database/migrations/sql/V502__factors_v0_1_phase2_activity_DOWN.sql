-- =============================================================================
-- V502: GreenLang Factors v0.1 Alpha — Phase 2 Activity Taxonomy (REVERSE)
-- =============================================================================
-- Description: Reverse migration for V502__factors_v0_1_phase2_activity.sql.
--              Drops the activity table and its indexes. The CASCADE on
--              the table covers the self-referencing parent_urn FK and
--              any v0.1 reference-data dependents.
--
-- Authority:
--   - GreenLang_Climate_OS Final Product Definition Roadmap (CTO doc) §6.1
--   - Phase 2 plan: docs/factors/PHASE_2_PLAN.md §2.3 (WS5)
--
-- Wave: Phase 2 / TaskCreate #5 / WS5-T1
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Drop indexes (idempotent, safe even if already gone).
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS factors_v0_1.activity_taxonomy_idx;
DROP INDEX IF EXISTS factors_v0_1.activity_parent_idx;

-- -----------------------------------------------------------------------------
-- 2. Drop the activity table (CASCADE covers self-FK and any consumers).
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS factors_v0_1.activity CASCADE;

-- =============================================================================
-- End V502 DOWN
-- =============================================================================

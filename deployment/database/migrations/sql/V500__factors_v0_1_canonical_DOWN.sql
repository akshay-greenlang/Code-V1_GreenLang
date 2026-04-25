-- =============================================================================
-- V500: GreenLang Factors v0.1 Alpha — Canonical Schema (REVERSE / DOWN)
-- =============================================================================
-- Description: Reverse migration for V500__factors_v0_1_canonical.sql.
--              Drops the factor_publish_log audit table, the factor
--              immutability trigger and its function, the factor table,
--              the four registry tables (factor_pack / unit / geography /
--              methodology / source), and finally the schema itself.
--
--              CASCADE on the schema covers any unforeseen dependent
--              objects, but explicit ordered drops make the intent clear.
--
-- Authority:
--   - GreenLang_Climate_OS Final Product Definition Roadmap (CTO doc) §6.1
--   - Schema $id: https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json
--   - Freeze date: 2026-04-25 (config/schemas/FACTOR_RECORD_V0_1_FREEZE.md)
--
-- Wave: B / TaskCreate #2 / WS1-T2
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-25
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Append-only audit table (no FKs into it; safe to drop first).
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS factors_v0_1.factor_publish_log;

-- -----------------------------------------------------------------------------
-- 2. Factor immutability trigger and its backing function.
-- -----------------------------------------------------------------------------
DROP TRIGGER IF EXISTS factor_no_mutate_after_publish ON factors_v0_1.factor;
DROP FUNCTION IF EXISTS factors_v0_1.factor_immutable_trigger();

-- -----------------------------------------------------------------------------
-- 3. Factor table (FKs into source / factor_pack / unit / geography / methodology).
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS factors_v0_1.factor;

-- -----------------------------------------------------------------------------
-- 4. Factor pack (FK into source).
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS factors_v0_1.factor_pack;

-- -----------------------------------------------------------------------------
-- 5. Standalone registries (no inter-FKs except geography.parent_urn -> self).
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS factors_v0_1.unit;
DROP TABLE IF EXISTS factors_v0_1.geography;
DROP TABLE IF EXISTS factors_v0_1.methodology;
DROP TABLE IF EXISTS factors_v0_1.source;

-- -----------------------------------------------------------------------------
-- 6. Drop the schema itself (CASCADE for any residual dependent objects).
-- -----------------------------------------------------------------------------
DROP SCHEMA IF EXISTS factors_v0_1 CASCADE;

-- =============================================================================
-- End V500 DOWN
-- =============================================================================

-- =============================================================================
-- V501: GreenLang Factors v0.1 Alpha - Phase 2 ADDITIVE schema bumps (DOWN)
-- =============================================================================
-- Description: Reverse migration for V501__factors_v0_1_phase2_additive.sql.
--              Restores the V500 geography.type CHECK enum (no basin/tenant)
--              and the V500 geography_urn_pattern regex. Drops the
--              'seed_source' marker columns on geography / unit / methodology.
--
-- Pre-condition: caller has already run the matching Alembic 0002 downgrade
-- (or has otherwise removed any rows with type IN ('basin', 'tenant')); if
-- residual basin/tenant rows exist, restoring the V500 narrower CHECK will
-- fail loudly. This is BY DESIGN — Phase 2 ontology rollback is a two-step
-- (data → DDL) process.
--
-- Wave: F / TaskCreate #3+#4+#6 / WS3+WS4+WS6
-- Author: GL-FormulaLibraryCurator
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Drop the seed_source partial indexes and columns.
-- -----------------------------------------------------------------------------

DROP INDEX IF EXISTS factors_v0_1.geography_seed_source_idx;
DROP INDEX IF EXISTS factors_v0_1.unit_seed_source_idx;
DROP INDEX IF EXISTS factors_v0_1.methodology_seed_source_idx;

ALTER TABLE factors_v0_1.geography  DROP COLUMN IF EXISTS seed_source;
ALTER TABLE factors_v0_1.unit       DROP COLUMN IF EXISTS seed_source;
ALTER TABLE factors_v0_1.methodology DROP COLUMN IF EXISTS seed_source;

-- -----------------------------------------------------------------------------
-- 2. Restore the V500 narrower geography_urn_pattern CHECK regex.
-- -----------------------------------------------------------------------------

ALTER TABLE factors_v0_1.geography
    DROP CONSTRAINT IF EXISTS geography_urn_pattern;

ALTER TABLE factors_v0_1.geography
    ADD CONSTRAINT geography_urn_pattern CHECK (
        urn ~ '^urn:gl:geo:(global|country|subregion|state_or_province|grid_zone|bidding_zone|balancing_authority):[a-zA-Z0-9._-]+$'
    );

-- -----------------------------------------------------------------------------
-- 3. Restore the V500 narrower geography.type CHECK enum (no basin/tenant).
-- -----------------------------------------------------------------------------

ALTER TABLE factors_v0_1.geography
    DROP CONSTRAINT IF EXISTS geography_type_check;

ALTER TABLE factors_v0_1.geography
    ADD CONSTRAINT geography_type_check CHECK (type IN (
        'global',
        'country',
        'subregion',
        'state_or_province',
        'grid_zone',
        'bidding_zone',
        'balancing_authority'
    ));

-- =============================================================================
-- End V501 DOWN (Phase 2 additive rollback)
-- =============================================================================

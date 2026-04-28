-- =============================================================================
-- V501: GreenLang Factors v0.1 Alpha - Phase 2 ADDITIVE schema bumps (FORWARD)
-- =============================================================================
-- Description: Phase 2 ontology-seed prerequisites. Strictly ADDITIVE — does
--              not modify any data, does not remove any existing constraint
--              option, does not touch the FROZEN factor_record_v0_1 JSON
--              schema. Two surgical changes:
--
--                1) Extend factors_v0_1.geography.type CHECK to additionally
--                   accept 'basin' and 'tenant' geographies (CTO Phase 2
--                   spec §2.3 'Geography (WS3)' — required types: global,
--                   country, state/province, grid subregion, bidding zone,
--                   balancing authority, basin, tenant-defined).
--
--                2) Extend the matching geography_urn_pattern CHECK so the
--                   URN regex on geography.urn admits 'basin' and 'tenant'
--                   as valid <type> segments (otherwise INSERTing a
--                   'urn:gl:geo:basin:amazon' row would fail the URN regex
--                   even though the type enum allows it).
--
--              Also adds a 'seed_source' column (NULLABLE, DEFAULT NULL) on
--              the three Phase 2 ontology tables so the data migration in
--              Alembic 0002 can mark Phase 2 seed rows for downgrade-safe
--              deletion. Production-ingested rows leave 'seed_source' NULL
--              and are untouched by the data migration's downgrade.
--
-- Authority:
--   - GreenLang_Climate_OS Final Product Definition Roadmap (CTO doc) §6.1
--   - PHASE_2_PLAN.md §2.3 (Geography / Unit / Methodology seeds)
--   - Phase 2 brief (2026-04-27): basin + tenant additive to geography
--   - factor_record_v0_1.schema.json is FROZEN — this migration does NOT
--     change the JSON schema. The schema's geography_urn regex remains
--     authoritative for record-level validation; rows of type 'basin' or
--     'tenant' are NOT addressable by alpha-published factors (no factor
--     record carries such a geography_urn in v0.1). They exist in the
--     ontology table for upstream tooling and v0.5+ extension paths.
--
-- Wave: F / TaskCreate #3+#4+#6 / WS3+WS4+WS6 (Phase 2 ontology seeds)
-- Postgres target: 16+
-- Author: GL-FormulaLibraryCurator
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Extend the geography.type CHECK enum.
--    PostgreSQL has no ALTER CHECK; drop + recreate by name. The constraint
--    name 'geography_type_check' is the default Postgres assigns to an
--    unnamed CHECK on column 'type' in table 'geography'; we use the same
--    name idempotently with IF EXISTS for forward-replay safety.
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
        'balancing_authority',
        'basin',
        'tenant'
    ));

-- -----------------------------------------------------------------------------
-- 2. Extend the geography_urn_pattern CHECK regex to accept 'basin' and
--    'tenant' as the <type> segment. Drop the existing named constraint
--    from V500 and recreate with the broader regex. The geography.urn
--    UNIQUE constraint is a separate object (geography_urn_key); only the
--    regex CHECK is replaced here.
-- -----------------------------------------------------------------------------

ALTER TABLE factors_v0_1.geography
    DROP CONSTRAINT IF EXISTS geography_urn_pattern;

ALTER TABLE factors_v0_1.geography
    ADD CONSTRAINT geography_urn_pattern CHECK (
        urn ~ '^urn:gl:geo:(global|country|subregion|state_or_province|grid_zone|bidding_zone|balancing_authority|basin|tenant):[a-zA-Z0-9._-]+$'
    );

-- -----------------------------------------------------------------------------
-- 3. Add a 'seed_source' marker column on the three Phase 2 ontology tables
--    so the Alembic data migration in revision 0002 can safely DELETE only
--    the rows it inserted on downgrade. Production rows ingested via the
--    publish pipeline leave 'seed_source' NULL and are NOT touched.
-- -----------------------------------------------------------------------------

ALTER TABLE factors_v0_1.geography
    ADD COLUMN IF NOT EXISTS seed_source TEXT;

ALTER TABLE factors_v0_1.unit
    ADD COLUMN IF NOT EXISTS seed_source TEXT;

ALTER TABLE factors_v0_1.methodology
    ADD COLUMN IF NOT EXISTS seed_source TEXT;

CREATE INDEX IF NOT EXISTS geography_seed_source_idx
    ON factors_v0_1.geography (seed_source)
    WHERE seed_source IS NOT NULL;
CREATE INDEX IF NOT EXISTS unit_seed_source_idx
    ON factors_v0_1.unit (seed_source)
    WHERE seed_source IS NOT NULL;
CREATE INDEX IF NOT EXISTS methodology_seed_source_idx
    ON factors_v0_1.methodology (seed_source)
    WHERE seed_source IS NOT NULL;

-- =============================================================================
-- End V501 (Phase 2 additive)
-- =============================================================================

-- =============================================================================
-- V506: GreenLang Factors v0.1 Alpha - Phase 2 Additive Contract Fields (FORWARD)
-- =============================================================================
-- Description: Strictly ADDITIVE Postgres column adds backing the 5 new
--              OPTIONAL fields promoted into the v0.1 public contract per
--              CHANGELOG `## v0.1 - 2026-04-27 - additive`:
--
--                  activity_taxonomy_urn  - URN of the activity taxonomy entry
--                  confidence             - methodology-lead confidence in [0,1]
--                  created_at_pre_publish - staging-window create timestamp
--                  updated_at_pre_publish - staging-window last-edit timestamp
--                  superseded_by_urn      - reverse pointer to successor URN
--
--              Field-name caveat:
--              -----------------
--              The public JSON Schema and the Pydantic mirror name the two
--              new lifecycle timestamps `created_at` and `updated_at`. We
--              CANNOT reuse those exact column names on `factors_v0_1.factor`
--              because V500 already declared a `created_at` column (the row-
--              creation timestamp, defaulted via `NOW()`). To avoid silently
--              overloading that column with two different semantics, the
--              backing columns added here are named `created_at_pre_publish`
--              and `updated_at_pre_publish`. The repository / API layer maps
--              the public contract names to these columns when reading from
--              the JSONB blob (which keeps the canonical names).
--
--              Provenance gate (V500's `factor_review_required_fields` etc.)
--              is unchanged. The post-publish immutability trigger
--              (`factor_immutable_trigger`) is also unchanged: the new
--              columns are metadata-only and not in the trigger's blocked
--              set, so corrections that flip `superseded_by_urn` on a prior
--              factor (i.e. setting the reverse pointer on the OLD record
--              once a successor is published) are permitted.
--
-- Authority:
--   - GreenLang_Climate_OS Final Product Definition Roadmap (CTO doc) §6.1
--   - CHANGELOG anchor: ## v0.1 - 2026-04-27 - additive
--   - PHASE_2_PLAN.md (LifecycleFields group)
--   - Schema $id (UNCHANGED): https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json
--
-- Wave: Phase 2 / WS9-A / contract-fields amendment
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- Add five OPTIONAL columns to factors_v0_1.factor.
-- All five are NULLABLE. CHECK constraints enforce the same patterns / ranges
-- the JSON Schema declares. The activity_taxonomy_urn FK references
-- factors_v0_1.activity(urn) (created in V502 / WS5).
-- -----------------------------------------------------------------------------
ALTER TABLE factors_v0_1.factor
    ADD COLUMN activity_taxonomy_urn TEXT
        CONSTRAINT factor_activity_urn_pattern CHECK (
            activity_taxonomy_urn IS NULL
            OR activity_taxonomy_urn ~ '^urn:gl:activity:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*)?$'
        )
        REFERENCES factors_v0_1.activity(urn),
    ADD COLUMN confidence NUMERIC(3,2)
        CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    ADD COLUMN created_at_pre_publish TIMESTAMPTZ,
    ADD COLUMN updated_at_pre_publish TIMESTAMPTZ,
    ADD COLUMN superseded_by_urn TEXT
        CONSTRAINT factor_superseded_by_urn_pattern CHECK (
            superseded_by_urn IS NULL
            OR superseded_by_urn ~ '^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$'
        );

-- Partial indexes — only index rows that actually carry the field. Both
-- indexes are bounded by the catalog size (~1.5k rows in v0.1) so they
-- stay tiny in memory.
CREATE INDEX factor_activity_urn_idx
    ON factors_v0_1.factor (activity_taxonomy_urn)
    WHERE activity_taxonomy_urn IS NOT NULL;

CREATE INDEX factor_superseded_by_urn_idx
    ON factors_v0_1.factor (superseded_by_urn)
    WHERE superseded_by_urn IS NOT NULL;

-- =============================================================================
-- End V506 FORWARD
-- =============================================================================

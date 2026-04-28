-- =============================================================================
-- V505: GreenLang Factors v0.1 Phase 2 — Aliases + Source Artifacts (FORWARD)
-- =============================================================================
-- Description: Phase 2 / WS7 additive migration that creates:
--                * `factors_v0_1.factor_aliases` — legacy `EF:...` id ->
--                  canonical factor URN map (Phase 2 §2.4 / table #9).
--                * `factors_v0_1.source_artifacts` — raw immutable byte
--                  references (sha256 + uri + parser metadata; Phase 2
--                  §2.4 / table #2).
--
--              All operations are additive against V500/V501/V502; the
--              V500 schema itself is FROZEN and untouched.
--
-- Coordination note (collision with V501_additive):
--   The Phase 2 brief originally allocated V501 to factor_aliases +
--   source_artifacts AND to the `geography.type` extension that adds
--   'basin' and 'tenant'. WS3/WS4/WS6 shipped their V501_additive
--   migration first, which already performs the geography enum +
--   geography_urn_pattern extension AND adds a `seed_source` marker
--   column on the ontology tables. This migration therefore:
--     - SKIPS the geography enum / regex extension (already done by V501).
--     - ONLY creates the two new tables originally specified for V501
--       in the WS7 brief (factor_aliases, source_artifacts).
--     - Sits at slot V505 (next free) to avoid the V501 collision.
--
-- Authority:
--   - GreenLang Factors Phase 2 master plan (docs/factors/PHASE_2_PLAN.md)
--     §2.4 (Postgres canonical storage / WS7), table inventory rows
--     #2 (source_artifacts) and #9 (factor_aliases).
--   - CTO Phase 2 brief (TaskCreate #7 / WS7-T1).
--
-- Phase: 2 / TaskCreate #7 / WS7-T1
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. factor_aliases — legacy `EF:...` id -> canonical factor URN.
--    Append-only by convention (no UPDATE/DELETE in production); retired
--    rows carry a non-null `retired_at`.
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.factor_aliases (
    pk_id            BIGSERIAL PRIMARY KEY,
    urn              TEXT NOT NULL REFERENCES factors_v0_1.factor(urn),
    legacy_id        TEXT NOT NULL UNIQUE,
    kind             TEXT NOT NULL DEFAULT 'EF'
        CHECK (kind IN ('EF', 'custom')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retired_at       TIMESTAMPTZ
);

CREATE INDEX factor_aliases_urn_idx
    ON factors_v0_1.factor_aliases (urn);

-- -----------------------------------------------------------------------------
-- 2. source_artifacts — raw immutable bytes referenced by factors.
--    Each row is the canonical pointer for a sha256 (UNIQUE) and carries
--    the parser identity that produced its extracted records.
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.source_artifacts (
    pk_id            BIGSERIAL PRIMARY KEY,
    sha256           TEXT NOT NULL UNIQUE
        CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    source_urn       TEXT NOT NULL REFERENCES factors_v0_1.source(urn),
    source_version   TEXT NOT NULL,
    uri              TEXT NOT NULL,
    content_type     TEXT,
    size_bytes       BIGINT
        CHECK (size_bytes IS NULL OR size_bytes > 0),
    parser_id        TEXT,
    parser_version   TEXT,
    parser_commit    TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata         JSONB
);

CREATE INDEX source_artifacts_source_idx
    ON factors_v0_1.source_artifacts (source_urn);

CREATE INDEX source_artifacts_version_idx
    ON factors_v0_1.source_artifacts (source_urn, source_version);

-- =============================================================================
-- End V505
-- =============================================================================

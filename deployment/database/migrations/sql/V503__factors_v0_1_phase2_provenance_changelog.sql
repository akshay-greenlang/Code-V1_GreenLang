-- =============================================================================
-- V503: GreenLang Factors v0.1 Phase 2 — Provenance + Changelog (FORWARD)
-- =============================================================================
-- Description: Phase 2 / WS7 migration that:
--                * creates `factors_v0_1.provenance_edges` linking each
--                  factor URN to one or more `source_artifacts` rows
--                  (factor -> raw artifact lineage).
--                * creates `factors_v0_1.changelog_events` — append-only
--                  schema/record audit covering schema bumps, factor
--                  publish/supersede/deprecate, source registry changes,
--                  pack releases and migration application.
--
--              V502 (activity table) is owned by WS5 and is NOT part of
--              this migration. provenance_edges does not depend on
--              activity, so V503 is safe to apply on top of V502.
--
-- Authority:
--   - GreenLang Factors Phase 2 master plan §2.4 (table inventory rows
--     #10 provenance_edges and #11 changelog_events).
--   - CTO Phase 2 brief (TaskCreate #7 / WS7-T1).
--
-- Phase: 2 / TaskCreate #7 / WS7-T1
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. provenance_edges — factor URN -> source_artifact lineage.
--    edge_type captures the relation kind (extraction, derivation,
--    correction, supersedes); UNIQUE on (factor_urn, source_artifact_pk,
--    row_ref, edge_type) prevents duplicate edges.
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.provenance_edges (
    pk_id              BIGSERIAL PRIMARY KEY,
    factor_urn         TEXT NOT NULL REFERENCES factors_v0_1.factor(urn),
    source_artifact_pk BIGINT NOT NULL REFERENCES factors_v0_1.source_artifacts(pk_id),
    row_ref            TEXT NOT NULL,
    edge_type          TEXT NOT NULL DEFAULT 'extraction'
        CHECK (edge_type IN ('extraction', 'derivation', 'correction', 'supersedes')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (factor_urn, source_artifact_pk, row_ref, edge_type)
);

CREATE INDEX provenance_edges_factor_idx
    ON factors_v0_1.provenance_edges (factor_urn);

CREATE INDEX provenance_edges_artifact_idx
    ON factors_v0_1.provenance_edges (source_artifact_pk);

-- -----------------------------------------------------------------------------
-- 2. changelog_events — append-only schema / record audit log.
--    Covers schema bumps and lifecycle transitions across the catalog.
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.changelog_events (
    pk_id              BIGSERIAL PRIMARY KEY,
    event_type         TEXT NOT NULL CHECK (event_type IN (
        'schema_change',
        'factor_publish',
        'factor_supersede',
        'factor_deprecate',
        'source_add',
        'source_deprecate',
        'pack_release',
        'migration_apply'
    )),
    schema_version     TEXT,
    subject_urn        TEXT,
    change_class       TEXT
        CHECK (change_class IN ('additive', 'breaking', 'deprecated', 'removed')
               OR change_class IS NULL),
    migration_note_uri TEXT,
    actor              TEXT NOT NULL,
    occurred_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata           JSONB
);

CREATE INDEX changelog_events_subject_idx
    ON factors_v0_1.changelog_events (subject_urn);

CREATE INDEX changelog_events_type_idx
    ON factors_v0_1.changelog_events (event_type, occurred_at DESC);

-- =============================================================================
-- End V503
-- =============================================================================

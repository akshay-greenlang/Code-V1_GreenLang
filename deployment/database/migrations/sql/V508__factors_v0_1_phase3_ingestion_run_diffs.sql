-- =============================================================================
-- V508: GreenLang Factors v0.1 - Phase 3 Ingestion Run Diffs (FORWARD)
-- =============================================================================
-- Description: Creates the per-row diff table produced by stage 6 of the
--              ingestion pipeline. One row per individual diff entry so the
--              MD/JSON diff blob (referenced by `ingestion_runs.diff_*_uri`)
--              is reproducible directly from these rows — methodology-lead
--              reviewers can pivot or filter without re-fetching the diff
--              artefact.
--
--              Two database objects are created:
--
--                1. `factors_v0_1.ingestion_diff_kind` ENUM — the closed
--                   set of diff classifications produced by `StagingDiff`
--                   (greenlang/factors/release/alpha_publisher.py) at
--                   stage 6:
--
--                       added                  - new factor URN
--                       removed                - factor present in prod but
--                                                missing from staging run
--                       changed                - same URN, attribute drift
--                       supersedes             - new URN supersedes prior
--                       unchanged              - no-op (counted, not always
--                                                emitted to MD diff)
--                       parser_version_change  - parser identity changed
--                                                without a value change
--                       licence_change         - licence_class drift
--                       methodology_change     - methodology_urn drift
--                       removal_candidate      - flagged for human review;
--                                                never auto-deleted
--
--                2. `factors_v0_1.ingestion_run_diffs` — append-only diff
--                   detail with CASCADE delete on the parent run (so an
--                   aborted run cleans up its row-level audit trail).
--
--              The `factor_urn` column intentionally has NO foreign key to
--              `factors_v0_1.factor.urn` because the diff can reference:
--                * a factor that has not yet been published (added)
--                * a factor that has been removed from production (removed)
--                * a not-yet-staged successor (supersedes preview)
--
-- Authority:
--   - PHASE_3_PLAN.md §"Dedupe / supersede / diff rules"
--   - greenlang/factors/release/alpha_publisher.py (StagingDiff source)
--
-- Wave: Phase 3 / Wave 1.0 / TaskCreate #27
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-28
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Diff-kind ENUM.
-- -----------------------------------------------------------------------------
CREATE TYPE factors_v0_1.ingestion_diff_kind AS ENUM (
    'added',
    'removed',
    'changed',
    'supersedes',
    'unchanged',
    'parser_version_change',
    'licence_change',
    'methodology_change',
    'removal_candidate'
);

-- -----------------------------------------------------------------------------
-- 2. ingestion_run_diffs — one row per individual diff entry.
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.ingestion_run_diffs (
    pk_id              BIGSERIAL PRIMARY KEY,

    run_id             UUID NOT NULL
        REFERENCES factors_v0_1.ingestion_runs(run_id) ON DELETE CASCADE,

    kind               factors_v0_1.ingestion_diff_kind NOT NULL,

    -- The factor URN this diff entry relates to. NO FK by design; see
    -- header for the three lifecycle states this URN may represent.
    factor_urn         TEXT NOT NULL,

    -- For supersedes / changed kinds: the prior URN being replaced.
    prior_factor_urn   TEXT,

    -- For changed kind: the dotted attribute path that drifted.
    -- e.g. `value`, `unit`, `methodology_urn`, `licence.class`.
    attribute_changed  TEXT,

    -- For changed kind: the prior + new payload values. JSONB for shape
    -- flexibility (numeric, string, nested object all fit).
    prior_value        JSONB,
    new_value          JSONB,

    generated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 3. Indexes — operator-query workload.
-- -----------------------------------------------------------------------------
-- Composite index supports the two most common review queries:
--   "show me everything that changed in run X" (run_id alone) and
--   "show me only `added` rows in run X" (run_id + kind).
CREATE INDEX ingestion_run_diffs_run_kind_idx
    ON factors_v0_1.ingestion_run_diffs (run_id, kind);

-- Lookup all diff entries that ever touched a given URN (cross-run audit).
CREATE INDEX ingestion_run_diffs_factor_urn_idx
    ON factors_v0_1.ingestion_run_diffs (factor_urn);

-- =============================================================================
-- End V508 FORWARD
-- =============================================================================

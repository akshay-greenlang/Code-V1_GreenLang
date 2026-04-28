-- =============================================================================
-- V507: GreenLang Factors v0.1 - Phase 3 Ingestion Runs (FORWARD)
-- =============================================================================
-- Description: Creates the canonical Postgres backing table for the seven-
--              stage ingestion pipeline state machine introduced in Phase 3.
--
--              Two database objects are created:
--
--                1. `factors_v0_1.ingestion_run_status` ENUM — the formal
--                   run-status state machine. Values are sequenced through
--                   the pipeline:
--
--                       created -> fetched -> parsed -> normalized
--                               -> validated -> deduped -> staged
--                               -> review_required -> published
--
--                   Terminal failure / branch states: rejected, failed,
--                   rolled_back. The Python `Enum` mirror lives in
--                   greenlang/factors/ingestion/state.py.
--
--                2. `factors_v0_1.ingestion_runs` — one row per ingestion
--                   run. Carries the artifact reference, parser identity,
--                   operator identity (with bot/human pattern guard),
--                   approver identity (humans only), diff URIs, run
--                   counters, structured error JSON, and last-updated-at
--                   timestamp maintained by an UPDATE trigger.
--
--              Operator policy:
--                * `operator` MUST match `bot:<id>` or `human:<email>`.
--                * `approved_by` if set MUST be a `human:<email>`. Bots
--                  cannot approve a publish action.
--                * Any row landing in a terminal published state
--                  (`status` in {published, rolled_back}) MUST carry a
--                  non-null `approved_by`.
--
--              Indexes optimise the typical operator queries:
--                - "all runs for a source" (source_urn)
--                - "what's stuck in review_required" (status)
--                - "recent runs in a publish batch" (batch_id)
--                - "latest activity" (started_at DESC)
--
-- Authority:
--   - PHASE_3_PLAN.md §"The seven-stage pipeline contract"
--   - PHASE_3_PLAN.md §"Run-status enum (formal)"
--   - CTO Phase 3 brief (2026-04-28)
--
-- Wave: Phase 3 / Wave 1.0 / TaskCreate #27
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-28
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Run-status ENUM. Order matters for human readability; Postgres stores
--    the underlying OID, so reordering values would be a breaking schema
--    change (would require ALTER TYPE ... RENAME VALUE).
-- -----------------------------------------------------------------------------
CREATE TYPE factors_v0_1.ingestion_run_status AS ENUM (
    'created',
    'fetched',
    'parsed',
    'normalized',
    'validated',
    'deduped',
    'staged',
    'review_required',
    'published',
    'rejected',
    'failed',
    'rolled_back'
);

-- -----------------------------------------------------------------------------
-- 2. ingestion_runs — one row per pipeline invocation.
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.ingestion_runs (
    run_id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source identity.
    source_urn                 TEXT NOT NULL
        REFERENCES factors_v0_1.source(urn),
    source_version             TEXT NOT NULL,

    -- Lifecycle timestamps + state.
    started_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    status                     factors_v0_1.ingestion_run_status NOT NULL
                                   DEFAULT 'created',
    -- The pipeline stage currently being executed (NULL once the run
    -- terminates). Constrained to the seven canonical stage names.
    current_stage              TEXT
        CONSTRAINT ingestion_runs_current_stage_chk CHECK (
            current_stage IS NULL
            OR current_stage IN (
                'fetch', 'parse', 'normalize', 'validate',
                'dedupe', 'stage', 'publish'
            )
        ),

    -- Stage 1 output: the immutable artifact reference. RESTRICTed so
    -- a published run's artifact cannot be deleted out from under it.
    artifact_pk_id             BIGINT
        REFERENCES factors_v0_1.source_artifacts(pk_id) ON DELETE RESTRICT,

    -- Parser identity (mirrored onto each artifact for redundancy).
    parser_module              TEXT,
    parser_function            TEXT,
    parser_version             TEXT,
    parser_commit              TEXT,

    -- Operator identity. Bots MUST follow `bot:<lowercase-slug>`;
    -- humans MUST follow `human:<email>`.
    operator                   TEXT NOT NULL
        CONSTRAINT ingestion_runs_operator_pattern_chk CHECK (
            operator ~ '^(bot:[a-z0-9_.-]+|human:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})$'
        ),

    -- Publish coordinates.
    batch_id                   UUID,
    approved_by                TEXT,
    approved_at                TIMESTAMPTZ,

    -- Stage 6 outputs (diff artefacts referenced by URI; the row-level
    -- detail lives in factors_v0_1.ingestion_run_diffs).
    diff_json_uri              TEXT,
    diff_md_uri                TEXT,

    -- Run counters (set at end of stage 6).
    accepted_count             INT,
    rejected_count             INT,
    supersedes_count           INT,
    unchanged_count            INT,
    removal_candidate_count    INT,

    -- Terminal-failure detail. Free-form structured JSON; populated when
    -- status in {rejected, failed}.
    error_json                 JSONB,

    -- Trigger-maintained.
    last_updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Approver-policy guards.
    -- Bots are not allowed to approve a publish; if approved_by is set
    -- it MUST be a human:<email>.
    CONSTRAINT ingestion_runs_approver_must_be_human_chk CHECK (
        approved_by IS NULL
        OR approved_by ~ '^human:.+@.+\..+$'
    ),
    -- Terminal-publish states REQUIRE an approver.
    CONSTRAINT ingestion_runs_terminal_publish_requires_approver_chk CHECK (
        status NOT IN ('published', 'rolled_back')
        OR approved_by IS NOT NULL
    )
);

-- -----------------------------------------------------------------------------
-- 3. Indexes — operator-query workload.
-- -----------------------------------------------------------------------------
CREATE INDEX ingestion_runs_source_urn_idx
    ON factors_v0_1.ingestion_runs (source_urn);

CREATE INDEX ingestion_runs_status_idx
    ON factors_v0_1.ingestion_runs (status);

CREATE INDEX ingestion_runs_batch_id_idx
    ON factors_v0_1.ingestion_runs (batch_id)
    WHERE batch_id IS NOT NULL;

CREATE INDEX ingestion_runs_started_at_desc_idx
    ON factors_v0_1.ingestion_runs (started_at DESC);

-- -----------------------------------------------------------------------------
-- 4. Trigger to maintain last_updated_at on every UPDATE.
--    Function is namespaced to factors_v0_1 to avoid colliding with any
--    pre-existing `set_last_updated_at` helper in `public`.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION factors_v0_1.ingestion_runs_set_last_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ingestion_runs_last_updated_at_trg
    BEFORE UPDATE ON factors_v0_1.ingestion_runs
    FOR EACH ROW
    EXECUTE FUNCTION factors_v0_1.ingestion_runs_set_last_updated_at();

-- =============================================================================
-- End V507 FORWARD
-- =============================================================================

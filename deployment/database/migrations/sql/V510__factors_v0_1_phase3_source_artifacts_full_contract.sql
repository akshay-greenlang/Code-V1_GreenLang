-- =============================================================================
-- V510: GreenLang Factors v0.1 - Phase 3 source_artifacts full Phase 3 contract (FORWARD)
-- =============================================================================
-- Description: Phase 3 audit gap C — extends `factors_v0_1.source_artifacts`
--              with the additional contract columns the Phase 3 plan
--              §"Artifact storage contract" requires on every certified
--              factor's lineage. V505 created the table with 11 columns;
--              this revision adds the 7 still-missing ones so the runner's
--              `upsert_source_artifact()` can land the full 16+ field row:
--
--                  V505 already had: sha256, source_urn, source_version,
--                                    uri, content_type, size_bytes,
--                                    parser_id, parser_version,
--                                    parser_commit, ingested_at, metadata
--
--                  V510 adds:        source_url, source_publication_date,
--                                    parser_module, parser_function,
--                                    operator, licence_class,
--                                    redistribution_class,
--                                    ingestion_run_id, status
--
--              `uri`         remains the canonical raw_bytes_uri column
--                            (the Phase 3 contract uses a different name
--                            but the on-disk column is unchanged).
--              `size_bytes`  remains the canonical bytes_size column.
--              `ingested_at` remains the canonical fetched_at column.
--
--              All ALTER TABLE operations are wrapped in idempotent DO
--              blocks so re-running V510 against an already-migrated DB
--              is a no-op (matters for the SQLite mirror parity test).
--
-- Authority:
--   - PHASE_3_PLAN.md §"Artifact storage contract" (16-field row).
--   - Phase 3 audit gap C — full source_artifacts row on every run.
--
-- Wave: Phase 3 / audit close-out
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-28
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Additive columns. Each ADD COLUMN is idempotent via information_schema.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'source_url'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN source_url TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'source_publication_date'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN source_publication_date DATE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'parser_module'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN parser_module TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'parser_function'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN parser_function TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'operator'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN operator TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'licence_class'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN licence_class TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'redistribution_class'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN redistribution_class TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'ingestion_run_id'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts ADD COLUMN ingestion_run_id TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'status'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts
            ADD COLUMN status TEXT NOT NULL DEFAULT 'fetched'
                CHECK (status IN (
                    'fetched','parsed','normalized','validated',
                    'deduped','staged','review_required','published',
                    'rejected','failed','rolled_back'
                ));
    END IF;
END
$$;

-- -----------------------------------------------------------------------------
-- 2. Index on ingestion_run_id so the runner can do "find every artifact
--    for run X" lookups without a sequential scan.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'factors_v0_1'
          AND tablename  = 'source_artifacts'
          AND indexname  = 'source_artifacts_ingestion_run_idx'
    ) THEN
        CREATE INDEX source_artifacts_ingestion_run_idx
            ON factors_v0_1.source_artifacts (ingestion_run_id)
            WHERE ingestion_run_id IS NOT NULL;
    END IF;
END
$$;

-- =============================================================================
-- End V510 FORWARD
-- =============================================================================

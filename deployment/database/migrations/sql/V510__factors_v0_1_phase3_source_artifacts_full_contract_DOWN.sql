-- =============================================================================
-- V510: GreenLang Factors v0.1 - Phase 3 source_artifacts full contract (DOWN)
-- =============================================================================
-- Description: Reverts V510 by dropping the additive columns + index.
--              ALTER TABLE ... DROP COLUMN IF EXISTS keeps the rollback
--              idempotent.
-- =============================================================================

SET search_path TO factors_v0_1, public;

DROP INDEX IF EXISTS factors_v0_1.source_artifacts_ingestion_run_idx;

ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS status;
ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS ingestion_run_id;
ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS redistribution_class;
ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS licence_class;
ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS operator;
ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS parser_function;
ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS parser_module;
ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS source_publication_date;
ALTER TABLE factors_v0_1.source_artifacts DROP COLUMN IF EXISTS source_url;

-- =============================================================================
-- End V510 DOWN
-- =============================================================================

-- =============================================================================
-- V506: GreenLang Factors v0.1 Alpha - Phase 2 Additive Contract Fields (DOWN)
-- =============================================================================
-- Description: Reverse of V506 FORWARD. Drops the two partial indexes and
--              the five additive columns added by the Phase 2 contract-
--              fields amendment.
--
--              IMPORTANT: this DOWN migration LOSES data stored in those
--              columns. Run only when the schema rollback is intentional
--              (e.g. amending the public contract in v0.2 with a different
--              shape). The corresponding CHANGELOG entry must be reverted
--              in the same operator action.
--
-- Authority:
--   - V506 FORWARD: V506__factors_v0_1_phase2_contract_fields.sql
--   - CHANGELOG anchor: ## v0.1 - 2026-04-27 - additive
--
-- Wave: Phase 2 / WS9-A / contract-fields amendment
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- Drop the partial indexes BEFORE the columns they reference.
DROP INDEX IF EXISTS factors_v0_1.factor_superseded_by_urn_idx;
DROP INDEX IF EXISTS factors_v0_1.factor_activity_urn_idx;

-- Drop the five additive columns. ALTER TABLE ... DROP COLUMN cascades
-- the CHECK constraints + foreign-key constraint automatically, so we do
-- not need to drop them explicitly.
ALTER TABLE factors_v0_1.factor
    DROP COLUMN IF EXISTS superseded_by_urn,
    DROP COLUMN IF EXISTS updated_at_pre_publish,
    DROP COLUMN IF EXISTS created_at_pre_publish,
    DROP COLUMN IF EXISTS confidence,
    DROP COLUMN IF EXISTS activity_taxonomy_urn;

-- =============================================================================
-- End V506 DOWN
-- =============================================================================

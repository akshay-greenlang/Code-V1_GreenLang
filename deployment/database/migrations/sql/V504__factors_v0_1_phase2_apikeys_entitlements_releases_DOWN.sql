-- =============================================================================
-- V504: GreenLang Factors v0.1 Phase 2 — Reverse (api_keys, entitlements,
--                                                   release_manifests)
-- =============================================================================
-- Description: Exact inverse of V504__factors_v0_1_phase2_apikeys_entitlements_releases.sql.
--              All three tables are leaf tables (no inbound FKs) so they
--              can be dropped in any order; we drop in reverse-create
--              order for clarity.
--
-- Phase: 2 / TaskCreate #7 / WS7-T1
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Drop release_manifests.
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS factors_v0_1.release_manifests_released_at_idx;
DROP TABLE IF EXISTS factors_v0_1.release_manifests;

-- -----------------------------------------------------------------------------
-- 2. Drop entitlements.
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS factors_v0_1.entitlements_source_idx;
DROP INDEX IF EXISTS factors_v0_1.entitlements_tenant_idx;
DROP TABLE IF EXISTS factors_v0_1.entitlements;

-- -----------------------------------------------------------------------------
-- 3. Drop api_keys.
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS factors_v0_1.api_keys_tenant_idx;
DROP TABLE IF EXISTS factors_v0_1.api_keys;

-- =============================================================================
-- End V504 DOWN
-- =============================================================================

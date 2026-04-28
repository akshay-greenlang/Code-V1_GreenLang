-- =============================================================================
-- V503: GreenLang Factors v0.1 Phase 2 — Provenance + Changelog (REVERSE)
-- =============================================================================
-- Description: Exact inverse of V503__factors_v0_1_phase2_provenance_changelog.sql.
--              Drops `changelog_events` first (no inbound FKs), then
--              `provenance_edges` (FKs out to factor and source_artifacts;
--              no inbound FKs).
--
-- Phase: 2 / TaskCreate #7 / WS7-T1
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Drop changelog_events.
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS factors_v0_1.changelog_events_type_idx;
DROP INDEX IF EXISTS factors_v0_1.changelog_events_subject_idx;
DROP TABLE IF EXISTS factors_v0_1.changelog_events;

-- -----------------------------------------------------------------------------
-- 2. Drop provenance_edges.
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS factors_v0_1.provenance_edges_artifact_idx;
DROP INDEX IF EXISTS factors_v0_1.provenance_edges_factor_idx;
DROP TABLE IF EXISTS factors_v0_1.provenance_edges;

-- =============================================================================
-- End V503 DOWN
-- =============================================================================

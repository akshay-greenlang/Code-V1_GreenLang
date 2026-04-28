-- =============================================================================
-- V502: GreenLang Factors v0.1 Alpha — Phase 2 Activity Taxonomy (FORWARD)
-- =============================================================================
-- Description: Adds the canonical `activity` ontology table to the
--              factors_v0_1 schema. The activity table is the registry
--              of activity-taxonomy entries (IPCC, GHGP, HS/CN, CPC, NACE,
--              NAICS, SIC, PACT, freight, CBAM, PCF, refrigerants,
--              agriculture, waste, land-use). Each row is uniquely
--              addressed by a `urn:gl:activity:<taxonomy>:<code>` URN.
--
--              In v0.1 the table is reference data only. The frozen
--              `factor_record_v0_1` schema does NOT yet require an
--              activity_taxonomy_urn FK on `factor`; promotion to a
--              first-class FK is deferred to v0.2.
--
-- Authority:
--   - GreenLang_Climate_OS Final Product Definition Roadmap (CTO doc) §6.1
--   - Phase 2 plan: docs/factors/PHASE_2_PLAN.md §2.3 (WS5)
--   - Phase 2 release-templates: docs/factors/release-templates/
--
-- Wave: Phase 2 / TaskCreate #5 / WS5-T1
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- Activity ontology — every activity-taxonomy entry usable as an
-- emissions activity reference (IPCC categories, GHGP scopes, HS/CN
-- chapters, etc.). Hierarchical via parent_urn (self-FK).
-- -----------------------------------------------------------------------------
CREATE TABLE activity (
    pk_id            BIGSERIAL PRIMARY KEY,
    urn              TEXT NOT NULL UNIQUE
        CONSTRAINT activity_urn_pattern
        CHECK (urn ~ '^urn:gl:activity:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*)?$'),
    taxonomy         TEXT NOT NULL CHECK (taxonomy IN (
        'ipcc',
        'ghgp',
        'hs-cn',
        'cpc',
        'nace',
        'naics',
        'sic',
        'pact',
        'freight',
        'cbam',
        'pcf',
        'refrigerants',
        'agriculture',
        'waste',
        'land-use'
    )),
    code             TEXT NOT NULL,
    name             TEXT NOT NULL,
    description      TEXT,
    parent_urn       TEXT REFERENCES activity(urn),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (taxonomy, code)
);

-- Indexes for the activity table
CREATE INDEX activity_taxonomy_idx ON activity (taxonomy);
CREATE INDEX activity_parent_idx   ON activity (parent_urn);

-- =============================================================================
-- End V502 FORWARD
-- =============================================================================

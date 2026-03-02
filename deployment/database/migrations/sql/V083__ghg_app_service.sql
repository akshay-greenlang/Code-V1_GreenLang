-- =============================================================================
-- V083: GL-GHG-APP v1.0 Application Service Schema
-- =============================================================================
-- Application: GL-GHG-APP (GHG Protocol Corporate Accounting & Reporting Standard)
-- Date:        March 2026
--
-- Application-level tables for corporate GHG accounting, inventory
-- management, emission aggregation, reporting, verification, and
-- target tracking per the GHG Protocol Corporate Standard.
--
-- EXTENDS:
--   V051-V070: AGENT-MRV calculation agents (Scope 1-3)
--   V080: Scope 3 Category Mapper Service
--
-- These tables sit in the ghg_app schema and aggregate results from the
-- underlying MRV agent schemas for all scopes.  They provide the
-- user-facing application layer including organizations, entity hierarchy,
-- inventory boundaries, base years, GHG inventories, emission data entries,
-- reports, verification records, targets, settings, and audit trail.
-- =============================================================================
-- Tables (12):
--   1.  organizations             - Top-level reporting organizations
--   2.  entities                  - Entity hierarchy (subsidiary/facility/operation)
--   3.  inventory_boundaries      - Organizational + operational boundary
--   4.  base_years                - Base year definition + emissions snapshot
--   5.  recalculations            - Base year recalculation events
--   6.  inventories               - Annual GHG inventories (central object)
--   7.  emission_entries          - Activity data / emission data entries (hypertable)
--   8.  reports                   - Generated reports
--   9.  verification_records      - Verification / assurance records
--  10.  targets                   - Emission reduction targets
--  11.  settings                  - Application settings KV store
--  12.  audit_trail               - Audit trail for all entities (hypertable)
--
-- Continuous Aggregates (1):
--   1.  monthly_emission_summary  - Monthly emission aggregates
--
-- Also includes: 14+ indexes (B-tree), update triggers, security grants,
-- and comments.
-- Previous: V082__eudr_app_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS ghg_app;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION ghg_app.update_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: ghg_app.organizations
-- =============================================================================
-- Top-level reporting organizations performing GHG accounting.
-- Each organization defines its own entity hierarchy, boundary,
-- and inventories per the GHG Protocol Corporate Standard.

CREATE TABLE ghg_app.organizations (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name              VARCHAR(500)    NOT NULL,
    industry          VARCHAR(200),
    country_iso3      CHAR(3),
    description       TEXT,
    metadata          JSONB           DEFAULT '{}',
    created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE ghg_app.organizations IS
    'Top-level reporting organizations for GHG Protocol corporate accounting.';

CREATE TRIGGER trg_org_updated
    BEFORE UPDATE ON ghg_app.organizations
    FOR EACH ROW EXECUTE FUNCTION ghg_app.update_timestamp();

-- =============================================================================
-- Table 2: ghg_app.entities
-- =============================================================================
-- Organizational entities forming the reporting hierarchy.
-- Entity types: SUBSIDIARY, FACILITY, OPERATION.
-- Used for consolidation under equity share / financial control /
-- operational control approaches (GHG Protocol Ch 3).

CREATE TABLE ghg_app.entities (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            UUID            NOT NULL REFERENCES ghg_app.organizations(id) ON DELETE CASCADE,
    parent_id         UUID            REFERENCES ghg_app.entities(id) ON DELETE SET NULL,
    name              VARCHAR(500)    NOT NULL,
    entity_type       VARCHAR(50)     NOT NULL
                      CHECK (entity_type IN ('SUBSIDIARY', 'FACILITY', 'OPERATION')),
    ownership_pct     DECIMAL(5,2)    DEFAULT 100.00
                      CHECK (ownership_pct >= 0 AND ownership_pct <= 100),
    country_iso3      CHAR(3),
    employees         INTEGER         DEFAULT 0 CHECK (employees >= 0),
    revenue_usd       DECIMAL(15,2)   DEFAULT 0 CHECK (revenue_usd >= 0),
    floor_area_m2     DECIMAL(12,2)   CHECK (floor_area_m2 IS NULL OR floor_area_m2 >= 0),
    production_units  DECIMAL(15,2)   CHECK (production_units IS NULL OR production_units >= 0),
    production_unit_name VARCHAR(100),
    active            BOOLEAN         DEFAULT TRUE,
    metadata          JSONB           DEFAULT '{}',
    created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE ghg_app.entities IS
    'Organizational entities (subsidiaries, facilities, operations) forming the reporting hierarchy.';

CREATE TRIGGER trg_entity_updated
    BEFORE UPDATE ON ghg_app.entities
    FOR EACH ROW EXECUTE FUNCTION ghg_app.update_timestamp();

-- =============================================================================
-- Table 3: ghg_app.inventory_boundaries
-- =============================================================================
-- Organizational and operational boundary for GHG inventories.
-- Defines consolidation approach, included scopes, base year,
-- and exclusions per GHG Protocol Chapters 3 and 4.

CREATE TABLE ghg_app.inventory_boundaries (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES ghg_app.organizations(id) ON DELETE CASCADE,
    consolidation_approach  VARCHAR(50)     NOT NULL
                            CHECK (consolidation_approach IN (
                                'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE'
                            )),
    scopes                  TEXT[]          NOT NULL DEFAULT '{SCOPE_1,SCOPE_2}',
    base_year               INTEGER         NOT NULL CHECK (base_year >= 1990 AND base_year <= 2100),
    reporting_year          INTEGER         NOT NULL CHECK (reporting_year >= 1990 AND reporting_year <= 2100),
    entity_ids              UUID[]          DEFAULT '{}',
    exclusions              JSONB           DEFAULT '[]',
    justification           TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_reporting_ge_base CHECK (reporting_year >= base_year)
);

COMMENT ON TABLE ghg_app.inventory_boundaries IS
    'Organizational and operational boundary for GHG inventories per GHG Protocol Ch 3-4.';

CREATE TRIGGER trg_boundary_updated
    BEFORE UPDATE ON ghg_app.inventory_boundaries
    FOR EACH ROW EXECUTE FUNCTION ghg_app.update_timestamp();

-- =============================================================================
-- Table 4: ghg_app.base_years
-- =============================================================================
-- Base year definition with emissions snapshot per GHG Protocol Ch 6.
-- The base year is the reference point for tracking emissions trends
-- and must be recalculated under specific structural change triggers.

CREATE TABLE ghg_app.base_years (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES ghg_app.organizations(id) ON DELETE CASCADE,
    year                        INTEGER         NOT NULL CHECK (year >= 1990 AND year <= 2100),
    scope1_tco2e                DECIMAL(15,3)   DEFAULT 0 CHECK (scope1_tco2e >= 0),
    scope2_location_tco2e       DECIMAL(15,3)   DEFAULT 0 CHECK (scope2_location_tco2e >= 0),
    scope2_market_tco2e         DECIMAL(15,3)   DEFAULT 0 CHECK (scope2_market_tco2e >= 0),
    scope3_tco2e                DECIMAL(15,3)   DEFAULT 0 CHECK (scope3_tco2e >= 0),
    total_tco2e                 DECIMAL(15,3)   DEFAULT 0 CHECK (total_tco2e >= 0),
    justification               TEXT            NOT NULL,
    locked                      BOOLEAN         DEFAULT FALSE,
    locked_at                   TIMESTAMPTZ,
    locked_by                   VARCHAR(200),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, year)
);

COMMENT ON TABLE ghg_app.base_years IS
    'Base year emissions snapshots per GHG Protocol Ch 6 for trend tracking.';

-- =============================================================================
-- Table 5: ghg_app.recalculations
-- =============================================================================
-- Base year recalculation events triggered by structural changes
-- (mergers, acquisitions, divestitures, methodology changes).

CREATE TABLE ghg_app.recalculations (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    base_year_id      UUID            NOT NULL REFERENCES ghg_app.base_years(id) ON DELETE CASCADE,
    trigger_type      VARCHAR(100)    NOT NULL,
    original_total    DECIMAL(15,3)   CHECK (original_total >= 0),
    new_total         DECIMAL(15,3)   CHECK (new_total >= 0),
    change_pct        DECIMAL(8,4),
    reason            TEXT            NOT NULL,
    affected_scopes   TEXT[]          DEFAULT '{}',
    approved_by       VARCHAR(200),
    created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE ghg_app.recalculations IS
    'Base year recalculation events per GHG Protocol Ch 6 structural change triggers.';

-- =============================================================================
-- Table 6: ghg_app.inventories
-- =============================================================================
-- Annual GHG inventories -- the central object aggregating all scope
-- emissions, intensity metrics, uncertainty, and completeness for an
-- organization-year.

CREATE TABLE ghg_app.inventories (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES ghg_app.organizations(id) ON DELETE CASCADE,
    boundary_id             UUID            REFERENCES ghg_app.inventory_boundaries(id) ON DELETE SET NULL,
    year                    INTEGER         NOT NULL CHECK (year >= 1990 AND year <= 2100),
    status                  VARCHAR(50)     DEFAULT 'draft'
                            CHECK (status IN ('draft', 'in_review', 'approved', 'verified', 'published')),
    -- Scope totals (tCO2e)
    scope1_tco2e            DECIMAL(15,3)   DEFAULT 0 CHECK (scope1_tco2e >= 0),
    scope2_location_tco2e   DECIMAL(15,3)   DEFAULT 0 CHECK (scope2_location_tco2e >= 0),
    scope2_market_tco2e     DECIMAL(15,3)   DEFAULT 0 CHECK (scope2_market_tco2e >= 0),
    scope3_tco2e            DECIMAL(15,3)   DEFAULT 0 CHECK (scope3_tco2e >= 0),
    total_tco2e             DECIMAL(15,3)   DEFAULT 0 CHECK (total_tco2e >= 0),
    biogenic_co2_tco2e      DECIMAL(15,3)   DEFAULT 0 CHECK (biogenic_co2_tco2e >= 0),
    -- Breakdowns (JSONB for flexible structure)
    scope1_by_gas           JSONB           DEFAULT '{}',
    scope1_by_category      JSONB           DEFAULT '{}',
    scope2_by_method        JSONB           DEFAULT '{}',
    scope3_by_category      JSONB           DEFAULT '{}',
    by_entity               JSONB           DEFAULT '{}',
    by_country              JSONB           DEFAULT '{}',
    -- Intensity & quality
    intensity_metrics       JSONB           DEFAULT '[]',
    uncertainty             JSONB           DEFAULT '{}',
    data_quality_score      DECIMAL(5,2)    DEFAULT 0 CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    completeness_pct        DECIMAL(5,2)    DEFAULT 0 CHECK (completeness_pct >= 0 AND completeness_pct <= 100),
    -- Metadata
    methodology_notes       TEXT,
    provenance_hash         CHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, year)
);

COMMENT ON TABLE ghg_app.inventories IS
    'Annual GHG inventories aggregating all scope emissions, metrics, and quality assessments.';

CREATE TRIGGER trg_inventory_updated
    BEFORE UPDATE ON ghg_app.inventories
    FOR EACH ROW EXECUTE FUNCTION ghg_app.update_timestamp();

-- =============================================================================
-- Table 7: ghg_app.emission_entries
-- =============================================================================
-- Individual emission data entries submitted by users and processed by
-- MRV agents.  This is a TimescaleDB hypertable partitioned on created_at
-- for time-series querying and retention policies.

CREATE TABLE ghg_app.emission_entries (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id      UUID            NOT NULL REFERENCES ghg_app.inventories(id) ON DELETE CASCADE,
    entity_id         UUID            REFERENCES ghg_app.entities(id) ON DELETE SET NULL,
    scope             VARCHAR(30)     NOT NULL,
    category          VARCHAR(100)    NOT NULL,
    sub_category      VARCHAR(200),
    -- Activity data
    activity_data     DECIMAL(15,6),
    activity_unit     VARCHAR(50),
    -- Emission factor
    emission_factor   DECIMAL(15,8),
    ef_source         VARCHAR(200),
    ef_unit           VARCHAR(100),
    -- Calculated emissions
    emissions_tco2e   DECIMAL(15,6)   CHECK (emissions_tco2e IS NULL OR emissions_tco2e >= 0),
    gas               VARCHAR(20)     DEFAULT 'CO2',
    -- Quality & methodology
    data_quality_tier VARCHAR(20)     DEFAULT 'TIER_2'
                      CHECK (data_quality_tier IN ('TIER_1', 'TIER_2', 'TIER_3')),
    methodology       VARCHAR(200),
    notes             TEXT,
    provenance_hash   CHAR(64),
    -- Period
    period_start      DATE,
    period_end        DATE,
    -- Timestamps
    created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE ghg_app.emission_entries IS
    'Individual emission data entries submitted by users and processed by MRV agents.';

-- Convert to hypertable for time-series partitioning
SELECT create_hypertable('ghg_app.emission_entries', 'created_at', if_not_exists => TRUE);

-- =============================================================================
-- Table 8: ghg_app.reports
-- =============================================================================
-- Generated GHG inventory reports in various formats (JSON, CSV, Excel, PDF).

CREATE TABLE ghg_app.reports (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id      UUID            NOT NULL REFERENCES ghg_app.inventories(id) ON DELETE CASCADE,
    format            VARCHAR(20)     NOT NULL
                      CHECK (format IN ('json', 'csv', 'excel', 'pdf')),
    sections          TEXT[]          DEFAULT '{}',
    file_path         VARCHAR(1000),
    file_size_bytes   BIGINT          CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),
    provenance_hash   CHAR(64),
    generated_by      VARCHAR(200),
    generated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE ghg_app.reports IS
    'Generated GHG inventory reports in JSON, CSV, Excel, or PDF format.';

-- =============================================================================
-- Table 9: ghg_app.verification_records
-- =============================================================================
-- Verification and assurance records tracking the full lifecycle from
-- internal review through external third-party assurance.

CREATE TABLE ghg_app.verification_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id                UUID            NOT NULL REFERENCES ghg_app.inventories(id) ON DELETE CASCADE,
    level                       VARCHAR(50)     NOT NULL
                                CHECK (level IN (
                                    'INTERNAL_REVIEW', 'LIMITED_ASSURANCE', 'REASONABLE_ASSURANCE'
                                )),
    status                      VARCHAR(50)     NOT NULL DEFAULT 'pending'
                                CHECK (status IN (
                                    'pending', 'in_review', 'submitted', 'approved',
                                    'rejected', 'verified'
                                )),
    -- Reviewer / verifier info
    reviewer_id                 VARCHAR(200),
    verifier_name               VARCHAR(500),
    verifier_organization       VARCHAR(500),
    verifier_accreditation      VARCHAR(200),
    -- Findings and statement
    findings                    JSONB           DEFAULT '[]',
    statement                   JSONB,
    opinion                     VARCHAR(50)
                                CHECK (opinion IS NULL OR opinion IN (
                                    'unqualified', 'qualified', 'adverse', 'disclaimer'
                                )),
    -- Timestamps
    started_at                  TIMESTAMPTZ,
    submitted_at                TIMESTAMPTZ,
    completed_at                TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE ghg_app.verification_records IS
    'Verification and assurance records for GHG inventories (internal review + external assurance).';

-- =============================================================================
-- Table 10: ghg_app.targets
-- =============================================================================
-- Emission reduction targets (absolute or intensity-based) with SBTi
-- alignment validation.

CREATE TABLE ghg_app.targets (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES ghg_app.organizations(id) ON DELETE CASCADE,
    name                    VARCHAR(500)    DEFAULT '',
    target_type             VARCHAR(20)     NOT NULL
                            CHECK (target_type IN ('ABSOLUTE', 'INTENSITY')),
    scope                   VARCHAR(30)     NOT NULL,
    base_year               INTEGER         NOT NULL CHECK (base_year >= 1990 AND base_year <= 2100),
    base_year_emissions     DECIMAL(15,3)   CHECK (base_year_emissions >= 0),
    target_year             INTEGER         NOT NULL CHECK (target_year >= 1990 AND target_year <= 2100),
    reduction_pct           DECIMAL(5,2)    NOT NULL
                            CHECK (reduction_pct > 0 AND reduction_pct <= 100),
    -- SBTi alignment
    sbti_aligned            BOOLEAN         DEFAULT FALSE,
    sbti_pathway            VARCHAR(50),
    -- Current progress
    current_emissions       DECIMAL(15,3)   CHECK (current_emissions IS NULL OR current_emissions >= 0),
    current_year            INTEGER         CHECK (current_year IS NULL OR (current_year >= 1990 AND current_year <= 2100)),
    current_progress_pct    DECIMAL(5,2)    DEFAULT 0,
    -- Status
    status                  VARCHAR(50)     DEFAULT 'active'
                            CHECK (status IN ('active', 'achieved', 'expired', 'cancelled')),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_target_after_base CHECK (target_year > base_year)
);

COMMENT ON TABLE ghg_app.targets IS
    'Emission reduction targets (absolute/intensity) with SBTi pathway alignment.';

CREATE TRIGGER trg_target_updated
    BEFORE UPDATE ON ghg_app.targets
    FOR EACH ROW EXECUTE FUNCTION ghg_app.update_timestamp();

-- =============================================================================
-- Table 11: ghg_app.settings
-- =============================================================================
-- Application settings key-value store per organization.

CREATE TABLE ghg_app.settings (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            UUID            REFERENCES ghg_app.organizations(id) ON DELETE CASCADE,
    key               VARCHAR(200)    NOT NULL,
    value             JSONB           NOT NULL,
    updated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, key)
);

COMMENT ON TABLE ghg_app.settings IS
    'Application settings key-value store per organization.';

-- =============================================================================
-- Table 12: ghg_app.audit_trail
-- =============================================================================
-- Audit trail for all entity actions in the GHG application.
-- TimescaleDB hypertable for time-series querying and retention.

CREATE TABLE ghg_app.audit_trail (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type       VARCHAR(50)     NOT NULL,
    entity_id         UUID            NOT NULL,
    action            VARCHAR(50)     NOT NULL,
    actor             VARCHAR(200),
    details           JSONB           DEFAULT '{}',
    created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE ghg_app.audit_trail IS
    'Audit trail for all entity actions in the GHG application.';

-- Convert to hypertable for time-series partitioning
SELECT create_hypertable('ghg_app.audit_trail', 'created_at', if_not_exists => TRUE);

-- =============================================================================
-- Continuous Aggregate: Monthly Emission Summary
-- =============================================================================
-- Aggregates emission entries by month, inventory, scope, and category
-- for dashboard time-series charts and trend analysis.

CREATE MATERIALIZED VIEW ghg_app.monthly_emission_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', created_at)    AS bucket,
    inventory_id,
    scope,
    category,
    SUM(emissions_tco2e)                  AS total_tco2e,
    COUNT(*)                              AS entry_count,
    AVG(CASE
        WHEN data_quality_tier = 'TIER_1' THEN 1
        WHEN data_quality_tier = 'TIER_2' THEN 2
        WHEN data_quality_tier = 'TIER_3' THEN 3
        ELSE 1
    END)                                  AS avg_quality_tier
FROM ghg_app.emission_entries
GROUP BY bucket, inventory_id, scope, category;

COMMENT ON MATERIALIZED VIEW ghg_app.monthly_emission_summary IS
    'Monthly aggregation of emission entries for time-series dashboards.';

-- =============================================================================
-- Indexes
-- =============================================================================

-- Table 2: entities
CREATE INDEX idx_ghg_entities_org
    ON ghg_app.entities(org_id);
CREATE INDEX idx_ghg_entities_parent
    ON ghg_app.entities(parent_id);
CREATE INDEX idx_ghg_entities_type
    ON ghg_app.entities(entity_type);

-- Table 3: inventory_boundaries
CREATE INDEX idx_ghg_boundaries_org
    ON ghg_app.inventory_boundaries(org_id);

-- Table 4: base_years
CREATE INDEX idx_ghg_baseyears_org
    ON ghg_app.base_years(org_id);

-- Table 5: recalculations
CREATE INDEX idx_ghg_recalc_baseyear
    ON ghg_app.recalculations(base_year_id);

-- Table 6: inventories
CREATE INDEX idx_ghg_inventories_org
    ON ghg_app.inventories(org_id);
CREATE INDEX idx_ghg_inventories_year
    ON ghg_app.inventories(year);
CREATE INDEX idx_ghg_inventories_status
    ON ghg_app.inventories(status);

-- Table 7: emission_entries (hypertable -- additional indexes)
CREATE INDEX idx_ghg_entries_inventory
    ON ghg_app.emission_entries(inventory_id, created_at DESC);
CREATE INDEX idx_ghg_entries_scope
    ON ghg_app.emission_entries(scope);
CREATE INDEX idx_ghg_entries_category
    ON ghg_app.emission_entries(category);
CREATE INDEX idx_ghg_entries_entity
    ON ghg_app.emission_entries(entity_id);

-- Table 8: reports
CREATE INDEX idx_ghg_reports_inventory
    ON ghg_app.reports(inventory_id);

-- Table 9: verification_records
CREATE INDEX idx_ghg_verification_inventory
    ON ghg_app.verification_records(inventory_id);
CREATE INDEX idx_ghg_verification_status
    ON ghg_app.verification_records(status);

-- Table 10: targets
CREATE INDEX idx_ghg_targets_org
    ON ghg_app.targets(org_id);
CREATE INDEX idx_ghg_targets_status
    ON ghg_app.targets(status);

-- Table 12: audit_trail (hypertable -- additional indexes)
CREATE INDEX idx_ghg_audit_entity
    ON ghg_app.audit_trail(entity_type, entity_id, created_at DESC);
CREATE INDEX idx_ghg_audit_actor
    ON ghg_app.audit_trail(actor);

-- =============================================================================
-- GIN Index on JSONB columns for flexible querying
-- =============================================================================

CREATE INDEX idx_ghg_inventories_scope1_gas_gin
    ON ghg_app.inventories USING GIN (scope1_by_gas);
CREATE INDEX idx_ghg_inventories_scope3_cat_gin
    ON ghg_app.inventories USING GIN (scope3_by_category);
CREATE INDEX idx_ghg_inventories_by_entity_gin
    ON ghg_app.inventories USING GIN (by_entity);
CREATE INDEX idx_ghg_inventories_by_country_gin
    ON ghg_app.inventories USING GIN (by_country);
CREATE INDEX idx_ghg_verification_findings_gin
    ON ghg_app.verification_records USING GIN (findings);
CREATE INDEX idx_ghg_audit_details_gin
    ON ghg_app.audit_trail USING GIN (details);
CREATE INDEX idx_ghg_entities_metadata_gin
    ON ghg_app.entities USING GIN (metadata);
CREATE INDEX idx_ghg_org_metadata_gin
    ON ghg_app.organizations USING GIN (metadata);

-- =============================================================================
-- Security: Grant permissions to application role
-- =============================================================================

GRANT USAGE ON SCHEMA ghg_app TO greenlang_app;
GRANT ALL ON ALL TABLES IN SCHEMA ghg_app TO greenlang_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ghg_app TO greenlang_app;

-- Read-only access for reporting/analytics role
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA ghg_app TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA ghg_app TO greenlang_readonly;
    END IF;
END
$$;

-- =============================================================================
-- End of V083: GL-GHG-APP v1.0 Application Service Schema
-- =============================================================================
-- Summary:
--   12 tables created (2 hypertables: emission_entries, audit_trail)
--   1 continuous aggregate (monthly_emission_summary)
--   5 update triggers
--   22 B-tree indexes
--   8 GIN indexes on JSONB columns
--   Security grants for greenlang_app and greenlang_readonly
-- =============================================================================

-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V421 - Consolidation
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates consolidation tables for aggregating entity-level emissions into
-- corporate totals. Consolidation runs process approved entity submissions,
-- apply equity percentages or control-based inclusion, compute per-entity
-- adjustments, and produce scope-level consolidated totals with provenance.
--
-- Tables (3):
--   1. ghg_consolidation.gl_cons_consolidation_runs
--   2. ghg_consolidation.gl_cons_entity_adjustments
--   3. ghg_consolidation.gl_cons_consolidated_totals
--
-- Also includes: indexes, RLS, constraints, comments.
-- Previous: V420__pack050_data_collection.sql
-- Next:     V422__pack050_eliminations.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_consolidation_runs
-- =============================================================================
-- A consolidation run represents one execution of the entity-to-corporate
-- aggregation process. Multiple runs can exist per period (draft, final,
-- restated). Each run captures the approach, boundary, completeness,
-- timing, and a provenance hash of all inputs for reproducibility.

CREATE TABLE ghg_consolidation.gl_cons_consolidation_runs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_boundaries(id) ON DELETE CASCADE,
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    run_number                  INTEGER         NOT NULL DEFAULT 1,
    run_type                    VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approach                    ghg_consolidation.consolidation_approach NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    status                      VARCHAR(20)     NOT NULL DEFAULT 'IN_PROGRESS',
    entities_included           INTEGER         NOT NULL DEFAULT 0,
    entities_excluded           INTEGER         NOT NULL DEFAULT 0,
    completeness_pct            NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    scope1_total_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope2_location_total_tco2e NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope2_market_total_tco2e   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope3_total_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    total_tco2e                 NUMERIC(20,6)   NOT NULL DEFAULT 0,
    eliminations_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    net_total_tco2e             NUMERIC(20,6)   NOT NULL DEFAULT 0,
    biogenic_total_tco2e        NUMERIC(20,6)   NOT NULL DEFAULT 0,
    input_provenance_hash       VARCHAR(64),
    output_provenance_hash      VARCHAR(64),
    provenance_hash             VARCHAR(64),
    started_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at                TIMESTAMPTZ,
    approved_at                 TIMESTAMPTZ,
    approved_by                 UUID,
    error_message               TEXT,
    run_config                  JSONB           DEFAULT '{}',
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_cr_dates CHECK (
        reporting_period_end > reporting_period_start
    ),
    CONSTRAINT chk_p050_cr_run_num CHECK (
        run_number >= 1 AND run_number <= 999
    ),
    CONSTRAINT chk_p050_cr_type CHECK (
        run_type IN ('DRAFT', 'PRELIMINARY', 'FINAL', 'RESTATED', 'AUDIT')
    ),
    CONSTRAINT chk_p050_cr_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'FAILED', 'APPROVED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p050_cr_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p050_cr_ent_incl CHECK (entities_included >= 0),
    CONSTRAINT chk_p050_cr_ent_excl CHECK (entities_excluded >= 0),
    CONSTRAINT chk_p050_cr_s1 CHECK (scope1_total_tco2e >= 0),
    CONSTRAINT chk_p050_cr_s2l CHECK (scope2_location_total_tco2e >= 0),
    CONSTRAINT chk_p050_cr_s2m CHECK (scope2_market_total_tco2e >= 0),
    CONSTRAINT chk_p050_cr_s3 CHECK (scope3_total_tco2e >= 0),
    CONSTRAINT chk_p050_cr_total CHECK (total_tco2e >= 0),
    CONSTRAINT chk_p050_cr_elim CHECK (eliminations_tco2e >= 0),
    CONSTRAINT chk_p050_cr_net CHECK (net_total_tco2e >= 0),
    CONSTRAINT chk_p050_cr_biogenic CHECK (biogenic_total_tco2e >= 0),
    CONSTRAINT uq_p050_cr_boundary_num UNIQUE (boundary_id, run_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_cr_tenant          ON ghg_consolidation.gl_cons_consolidation_runs(tenant_id);
CREATE INDEX idx_p050_cr_boundary        ON ghg_consolidation.gl_cons_consolidation_runs(boundary_id);
CREATE INDEX idx_p050_cr_period          ON ghg_consolidation.gl_cons_consolidation_runs(reporting_period_start, reporting_period_end);
CREATE INDEX idx_p050_cr_type            ON ghg_consolidation.gl_cons_consolidation_runs(run_type);
CREATE INDEX idx_p050_cr_status          ON ghg_consolidation.gl_cons_consolidation_runs(status);
CREATE INDEX idx_p050_cr_approach        ON ghg_consolidation.gl_cons_consolidation_runs(approach);
CREATE INDEX idx_p050_cr_approved        ON ghg_consolidation.gl_cons_consolidation_runs(boundary_id, status)
    WHERE status = 'APPROVED';
CREATE INDEX idx_p050_cr_final           ON ghg_consolidation.gl_cons_consolidation_runs(boundary_id, run_type)
    WHERE run_type = 'FINAL';
CREATE INDEX idx_p050_cr_completed       ON ghg_consolidation.gl_cons_consolidation_runs(completed_at)
    WHERE completed_at IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_consolidation_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_cr_tenant_isolation ON ghg_consolidation.gl_cons_consolidation_runs
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_entity_adjustments
-- =============================================================================
-- Per-entity emission adjustments within a consolidation run. Each row
-- captures the raw (100%) emissions for an entity, the equity or control
-- percentage applied, and the resulting adjusted emissions. Supports
-- multiple adjustment types (equity share, proportional, pro rata, etc.).

CREATE TABLE ghg_consolidation.gl_cons_entity_adjustments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_consolidation_runs(id) ON DELETE CASCADE,
    entity_id                   UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    inclusion_id                UUID            REFERENCES ghg_consolidation.gl_cons_entity_inclusions(id) ON DELETE SET NULL,
    scope                       VARCHAR(20)     NOT NULL,
    category                    VARCHAR(50),
    raw_emissions_tco2e         NUMERIC(20,6)   NOT NULL DEFAULT 0,
    equity_pct                  NUMERIC(10,4)   NOT NULL DEFAULT 100.0000,
    adjusted_emissions_tco2e    NUMERIC(20,6)   NOT NULL DEFAULT 0,
    adjustment_type             VARCHAR(30)     NOT NULL DEFAULT 'EQUITY_SHARE',
    adjustment_factor           NUMERIC(10,6)   NOT NULL DEFAULT 1.000000,
    adjustment_reason           TEXT,
    data_quality_tier           INTEGER,
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    submission_id               UUID            REFERENCES ghg_consolidation.gl_cons_entity_submissions(id) ON DELETE SET NULL,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_ea_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3')
    ),
    CONSTRAINT chk_p050_ea_raw CHECK (raw_emissions_tco2e >= 0),
    CONSTRAINT chk_p050_ea_equity CHECK (
        equity_pct >= 0 AND equity_pct <= 100
    ),
    CONSTRAINT chk_p050_ea_adjusted CHECK (adjusted_emissions_tco2e >= 0),
    CONSTRAINT chk_p050_ea_type CHECK (
        adjustment_type IN (
            'EQUITY_SHARE', 'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL',
            'PROPORTIONAL', 'PRO_RATA', 'MANUAL_OVERRIDE', 'NONE'
        )
    ),
    CONSTRAINT chk_p050_ea_factor CHECK (
        adjustment_factor >= 0 AND adjustment_factor <= 1
    ),
    CONSTRAINT chk_p050_ea_dq_tier CHECK (
        data_quality_tier IS NULL OR (data_quality_tier >= 1 AND data_quality_tier <= 5)
    ),
    CONSTRAINT uq_p050_ea_run_entity_scope_cat UNIQUE (run_id, entity_id, scope, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_ea_tenant          ON ghg_consolidation.gl_cons_entity_adjustments(tenant_id);
CREATE INDEX idx_p050_ea_run             ON ghg_consolidation.gl_cons_entity_adjustments(run_id);
CREATE INDEX idx_p050_ea_entity          ON ghg_consolidation.gl_cons_entity_adjustments(entity_id);
CREATE INDEX idx_p050_ea_scope           ON ghg_consolidation.gl_cons_entity_adjustments(scope);
CREATE INDEX idx_p050_ea_type            ON ghg_consolidation.gl_cons_entity_adjustments(adjustment_type);
CREATE INDEX idx_p050_ea_inclusion       ON ghg_consolidation.gl_cons_entity_adjustments(inclusion_id)
    WHERE inclusion_id IS NOT NULL;
CREATE INDEX idx_p050_ea_submission      ON ghg_consolidation.gl_cons_entity_adjustments(submission_id)
    WHERE submission_id IS NOT NULL;
CREATE INDEX idx_p050_ea_run_scope       ON ghg_consolidation.gl_cons_entity_adjustments(run_id, scope);
CREATE INDEX idx_p050_ea_run_entity      ON ghg_consolidation.gl_cons_entity_adjustments(run_id, entity_id);
CREATE INDEX idx_p050_ea_estimated       ON ghg_consolidation.gl_cons_entity_adjustments(run_id)
    WHERE is_estimated = true;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_entity_adjustments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_ea_tenant_isolation ON ghg_consolidation.gl_cons_entity_adjustments
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_consolidation.gl_cons_consolidated_totals
-- =============================================================================
-- Scope-level consolidated totals for a consolidation run. Aggregates
-- all entity adjustments into corporate-level totals by scope and
-- optional category. Each total carries entity count and provenance
-- hash for audit reproducibility.

CREATE TABLE ghg_consolidation.gl_cons_consolidated_totals (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_consolidation_runs(id) ON DELETE CASCADE,
    scope                       VARCHAR(20)     NOT NULL,
    category                    VARCHAR(50),
    category_description        VARCHAR(255),
    total_tco2e                 NUMERIC(20,6)   NOT NULL DEFAULT 0,
    raw_total_tco2e             NUMERIC(20,6)   NOT NULL DEFAULT 0,
    adjustment_total_tco2e      NUMERIC(20,6)   NOT NULL DEFAULT 0,
    elimination_total_tco2e     NUMERIC(20,6)   NOT NULL DEFAULT 0,
    net_total_tco2e             NUMERIC(20,6)   NOT NULL DEFAULT 0,
    biogenic_tco2e              NUMERIC(20,6)   NOT NULL DEFAULT 0,
    entity_count                INTEGER         NOT NULL DEFAULT 0,
    estimated_entity_count      INTEGER         NOT NULL DEFAULT 0,
    avg_data_quality_tier       NUMERIC(10,4),
    weighted_data_quality_score NUMERIC(10,4),
    completeness_pct            NUMERIC(10,4),
    provenance_hash             VARCHAR(64),
    methodology_summary         TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_ct_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3', 'TOTAL')
    ),
    CONSTRAINT chk_p050_ct_total CHECK (total_tco2e >= 0),
    CONSTRAINT chk_p050_ct_raw CHECK (raw_total_tco2e >= 0),
    CONSTRAINT chk_p050_ct_adjustment CHECK (adjustment_total_tco2e >= 0),
    CONSTRAINT chk_p050_ct_elim CHECK (elimination_total_tco2e >= 0),
    CONSTRAINT chk_p050_ct_net CHECK (net_total_tco2e >= 0),
    CONSTRAINT chk_p050_ct_biogenic CHECK (biogenic_tco2e >= 0),
    CONSTRAINT chk_p050_ct_entity_count CHECK (entity_count >= 0),
    CONSTRAINT chk_p050_ct_est_count CHECK (estimated_entity_count >= 0),
    CONSTRAINT chk_p050_ct_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p050_ct_dq_score CHECK (
        weighted_data_quality_score IS NULL OR (weighted_data_quality_score >= 0 AND weighted_data_quality_score <= 100)
    ),
    CONSTRAINT uq_p050_ct_run_scope_cat UNIQUE (run_id, scope, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_ct_tenant          ON ghg_consolidation.gl_cons_consolidated_totals(tenant_id);
CREATE INDEX idx_p050_ct_run             ON ghg_consolidation.gl_cons_consolidated_totals(run_id);
CREATE INDEX idx_p050_ct_scope           ON ghg_consolidation.gl_cons_consolidated_totals(scope);
CREATE INDEX idx_p050_ct_category        ON ghg_consolidation.gl_cons_consolidated_totals(category)
    WHERE category IS NOT NULL;
CREATE INDEX idx_p050_ct_run_scope       ON ghg_consolidation.gl_cons_consolidated_totals(run_id, scope);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_consolidated_totals ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_ct_tenant_isolation ON ghg_consolidation.gl_cons_consolidated_totals
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_consolidation.gl_cons_consolidation_runs IS
    'PACK-050: Consolidation runs (5 types, 5 statuses) with scope totals, eliminations, and provenance.';
COMMENT ON TABLE ghg_consolidation.gl_cons_entity_adjustments IS
    'PACK-050: Per-entity emission adjustments with equity/control percentage and 7 adjustment types.';
COMMENT ON TABLE ghg_consolidation.gl_cons_consolidated_totals IS
    'PACK-050: Scope-level consolidated totals with raw, adjusted, eliminated, and net emissions.';

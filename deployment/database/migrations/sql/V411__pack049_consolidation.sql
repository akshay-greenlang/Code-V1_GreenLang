-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V411 - Consolidation
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates consolidation tables for aggregating site-level emissions into
-- corporate totals. Consolidation runs process approved site submissions,
-- apply ownership percentages and boundary rules, eliminate intercompany
-- transactions, and reconcile against prior-period results.
--
-- Tables (4):
--   1. ghg_multisite.gl_ms_consolidation_runs
--   2. ghg_multisite.gl_ms_site_totals
--   3. ghg_multisite.gl_ms_elimination_entries
--   4. ghg_multisite.gl_ms_reconciliation_results
--
-- Also includes: indexes, RLS, comments.
-- Previous: V410__pack049_regional_factors.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_consolidation_runs
-- =============================================================================
-- A consolidation run represents one execution of the site-to-corporate
-- aggregation process. Multiple runs can exist per period (draft, final,
-- restated). Each run captures the consolidation approach, completeness,
-- and total emissions.

CREATE TABLE ghg_multisite.gl_ms_consolidation_runs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    boundary_id                 UUID            REFERENCES ghg_multisite.gl_ms_boundary_definitions(id) ON DELETE SET NULL,
    run_number                  INTEGER         NOT NULL DEFAULT 1,
    run_type                    VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    status                      VARCHAR(20)     NOT NULL DEFAULT 'IN_PROGRESS',
    consolidation_approach      VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    sites_included              INTEGER         NOT NULL DEFAULT 0,
    sites_excluded              INTEGER         NOT NULL DEFAULT 0,
    completeness_pct            NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    scope1_total_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope2_location_total_tco2e NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope2_market_total_tco2e   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope3_total_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    total_tco2e                 NUMERIC(20,6)   NOT NULL DEFAULT 0,
    eliminations_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    net_total_tco2e             NUMERIC(20,6)   NOT NULL DEFAULT 0,
    started_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at                TIMESTAMPTZ,
    approved_at                 TIMESTAMPTZ,
    approved_by                 UUID,
    provenance_hash             VARCHAR(64),
    error_message               TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_crun_type CHECK (
        run_type IN ('DRAFT', 'PRELIMINARY', 'FINAL', 'RESTATED', 'AUDIT')
    ),
    CONSTRAINT chk_p049_crun_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'FAILED', 'APPROVED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p049_crun_approach CHECK (
        consolidation_approach IN (
            'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE'
        )
    ),
    CONSTRAINT chk_p049_crun_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p049_crun_sites_incl CHECK (sites_included >= 0),
    CONSTRAINT chk_p049_crun_sites_excl CHECK (sites_excluded >= 0),
    CONSTRAINT chk_p049_crun_scope1 CHECK (scope1_total_tco2e >= 0),
    CONSTRAINT chk_p049_crun_scope2l CHECK (scope2_location_total_tco2e >= 0),
    CONSTRAINT chk_p049_crun_scope2m CHECK (scope2_market_total_tco2e >= 0),
    CONSTRAINT chk_p049_crun_scope3 CHECK (scope3_total_tco2e >= 0),
    CONSTRAINT chk_p049_crun_total CHECK (total_tco2e >= 0),
    CONSTRAINT chk_p049_crun_elim CHECK (eliminations_tco2e >= 0),
    CONSTRAINT chk_p049_crun_net CHECK (net_total_tco2e >= 0),
    CONSTRAINT chk_p049_crun_num CHECK (run_number >= 1 AND run_number <= 999),
    CONSTRAINT uq_p049_crun_period_num UNIQUE (period_id, run_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_crun_tenant        ON ghg_multisite.gl_ms_consolidation_runs(tenant_id);
CREATE INDEX idx_p049_crun_config        ON ghg_multisite.gl_ms_consolidation_runs(config_id);
CREATE INDEX idx_p049_crun_period        ON ghg_multisite.gl_ms_consolidation_runs(period_id);
CREATE INDEX idx_p049_crun_boundary      ON ghg_multisite.gl_ms_consolidation_runs(boundary_id)
    WHERE boundary_id IS NOT NULL;
CREATE INDEX idx_p049_crun_type          ON ghg_multisite.gl_ms_consolidation_runs(run_type);
CREATE INDEX idx_p049_crun_status        ON ghg_multisite.gl_ms_consolidation_runs(status);
CREATE INDEX idx_p049_crun_approach      ON ghg_multisite.gl_ms_consolidation_runs(consolidation_approach);
CREATE INDEX idx_p049_crun_approved      ON ghg_multisite.gl_ms_consolidation_runs(period_id, status)
    WHERE status = 'APPROVED';
CREATE INDEX idx_p049_crun_final         ON ghg_multisite.gl_ms_consolidation_runs(period_id, run_type)
    WHERE run_type = 'FINAL';

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_consolidation_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_crun_tenant_isolation ON ghg_multisite.gl_ms_consolidation_runs
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_site_totals
-- =============================================================================
-- Per-site emission totals within a consolidation run. Each row captures
-- the gross and consolidated emissions for one site, including the ownership
-- percentage and consolidation percentage applied.

CREATE TABLE ghg_multisite.gl_ms_site_totals (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_consolidation_runs(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    submission_id               UUID            REFERENCES ghg_multisite.gl_ms_site_submissions(id) ON DELETE SET NULL,
    inclusion_status            VARCHAR(20)     NOT NULL DEFAULT 'INCLUDED',
    ownership_pct               NUMERIC(10,4)   NOT NULL DEFAULT 100.0000,
    consolidation_pct           NUMERIC(10,4)   NOT NULL DEFAULT 100.0000,
    -- Gross emissions (before consolidation percentage)
    gross_scope1_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    gross_scope2_location_tco2e NUMERIC(20,6)   NOT NULL DEFAULT 0,
    gross_scope2_market_tco2e   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    gross_scope3_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    gross_total_tco2e           NUMERIC(20,6)   NOT NULL DEFAULT 0,
    -- Consolidated emissions (after consolidation percentage)
    consol_scope1_tco2e         NUMERIC(20,6)   NOT NULL DEFAULT 0,
    consol_scope2_location_tco2e NUMERIC(20,6)  NOT NULL DEFAULT 0,
    consol_scope2_market_tco2e  NUMERIC(20,6)   NOT NULL DEFAULT 0,
    consol_scope3_tco2e         NUMERIC(20,6)   NOT NULL DEFAULT 0,
    consol_total_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    data_quality_score          NUMERIC(10,4),
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    estimation_coverage_pct     NUMERIC(10,4),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_st_inclusion CHECK (
        inclusion_status IN ('INCLUDED', 'EXCLUDED', 'PARTIAL', 'ESTIMATED')
    ),
    CONSTRAINT chk_p049_st_ownership CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_p049_st_consol CHECK (
        consolidation_pct >= 0 AND consolidation_pct <= 100
    ),
    CONSTRAINT chk_p049_st_g_s1 CHECK (gross_scope1_tco2e >= 0),
    CONSTRAINT chk_p049_st_g_s2l CHECK (gross_scope2_location_tco2e >= 0),
    CONSTRAINT chk_p049_st_g_s2m CHECK (gross_scope2_market_tco2e >= 0),
    CONSTRAINT chk_p049_st_g_s3 CHECK (gross_scope3_tco2e >= 0),
    CONSTRAINT chk_p049_st_g_tot CHECK (gross_total_tco2e >= 0),
    CONSTRAINT chk_p049_st_c_s1 CHECK (consol_scope1_tco2e >= 0),
    CONSTRAINT chk_p049_st_c_s2l CHECK (consol_scope2_location_tco2e >= 0),
    CONSTRAINT chk_p049_st_c_s2m CHECK (consol_scope2_market_tco2e >= 0),
    CONSTRAINT chk_p049_st_c_s3 CHECK (consol_scope3_tco2e >= 0),
    CONSTRAINT chk_p049_st_c_tot CHECK (consol_total_tco2e >= 0),
    CONSTRAINT chk_p049_st_dq CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p049_st_est_cov CHECK (
        estimation_coverage_pct IS NULL OR (estimation_coverage_pct >= 0 AND estimation_coverage_pct <= 100)
    ),
    CONSTRAINT uq_p049_st_run_site UNIQUE (run_id, site_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_st_tenant          ON ghg_multisite.gl_ms_site_totals(tenant_id);
CREATE INDEX idx_p049_st_run             ON ghg_multisite.gl_ms_site_totals(run_id);
CREATE INDEX idx_p049_st_site            ON ghg_multisite.gl_ms_site_totals(site_id);
CREATE INDEX idx_p049_st_submission      ON ghg_multisite.gl_ms_site_totals(submission_id)
    WHERE submission_id IS NOT NULL;
CREATE INDEX idx_p049_st_inclusion       ON ghg_multisite.gl_ms_site_totals(inclusion_status);
CREATE INDEX idx_p049_st_included        ON ghg_multisite.gl_ms_site_totals(run_id, inclusion_status)
    WHERE inclusion_status = 'INCLUDED';
CREATE INDEX idx_p049_st_estimated       ON ghg_multisite.gl_ms_site_totals(run_id)
    WHERE is_estimated = true;
CREATE INDEX idx_p049_st_quality         ON ghg_multisite.gl_ms_site_totals(data_quality_score)
    WHERE data_quality_score IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_site_totals ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_st_tenant_isolation ON ghg_multisite.gl_ms_site_totals
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_multisite.gl_ms_elimination_entries
-- =============================================================================
-- Intercompany emission eliminations to avoid double counting within the
-- consolidated boundary. Each entry removes a specified quantity of
-- emissions from the consolidation total.

CREATE TABLE ghg_multisite.gl_ms_elimination_entries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_consolidation_runs(id) ON DELETE CASCADE,
    source_site_id              UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    target_site_id              UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    elimination_type            VARCHAR(30)     NOT NULL,
    scope                       VARCHAR(10)     NOT NULL,
    description                 TEXT            NOT NULL,
    gross_amount_tco2e          NUMERIC(20,6)   NOT NULL,
    elimination_amount_tco2e    NUMERIC(20,6)   NOT NULL,
    net_amount_tco2e            NUMERIC(20,6)   NOT NULL,
    evidence_ref                VARCHAR(500),
    is_automatic                BOOLEAN         NOT NULL DEFAULT false,
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_ee_type CHECK (
        elimination_type IN (
            'INTERNAL_ENERGY', 'INTERNAL_TRANSPORT', 'INTERNAL_WASTE',
            'INTERCOMPANY_SALES', 'SHARED_SERVICES', 'OTHER'
        )
    ),
    CONSTRAINT chk_p049_ee_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    ),
    CONSTRAINT chk_p049_ee_gross CHECK (gross_amount_tco2e >= 0),
    CONSTRAINT chk_p049_ee_elim CHECK (elimination_amount_tco2e >= 0),
    CONSTRAINT chk_p049_ee_net CHECK (net_amount_tco2e >= 0),
    CONSTRAINT chk_p049_ee_no_self CHECK (source_site_id != target_site_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_ee_tenant          ON ghg_multisite.gl_ms_elimination_entries(tenant_id);
CREATE INDEX idx_p049_ee_run             ON ghg_multisite.gl_ms_elimination_entries(run_id);
CREATE INDEX idx_p049_ee_source          ON ghg_multisite.gl_ms_elimination_entries(source_site_id);
CREATE INDEX idx_p049_ee_target          ON ghg_multisite.gl_ms_elimination_entries(target_site_id);
CREATE INDEX idx_p049_ee_type            ON ghg_multisite.gl_ms_elimination_entries(elimination_type);
CREATE INDEX idx_p049_ee_scope           ON ghg_multisite.gl_ms_elimination_entries(scope);
CREATE INDEX idx_p049_ee_auto            ON ghg_multisite.gl_ms_elimination_entries(run_id, is_automatic)
    WHERE is_automatic = true;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_elimination_entries ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_ee_tenant_isolation ON ghg_multisite.gl_ms_elimination_entries
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 4: ghg_multisite.gl_ms_reconciliation_results
-- =============================================================================
-- Period-over-period reconciliation comparing the current consolidation
-- run against a prior period or base year. Captures absolute and
-- percentage changes and flags significant variances.

CREATE TABLE ghg_multisite.gl_ms_reconciliation_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    current_run_id              UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_consolidation_runs(id) ON DELETE CASCADE,
    comparison_run_id           UUID            REFERENCES ghg_multisite.gl_ms_consolidation_runs(id) ON DELETE SET NULL,
    comparison_type             VARCHAR(30)     NOT NULL DEFAULT 'PRIOR_PERIOD',
    scope                       VARCHAR(20)     NOT NULL DEFAULT 'TOTAL',
    current_value_tco2e         NUMERIC(20,6)   NOT NULL,
    comparison_value_tco2e      NUMERIC(20,6)   NOT NULL,
    absolute_change_tco2e       NUMERIC(20,6)   NOT NULL,
    percentage_change           NUMERIC(10,4)   NOT NULL,
    is_significant              BOOLEAN         NOT NULL DEFAULT false,
    significance_threshold_pct  NUMERIC(10,4)   NOT NULL DEFAULT 5.0000,
    variance_explanation        TEXT,
    contributing_factors        JSONB           DEFAULT '[]',
    site_id                     UUID            REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE SET NULL,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_rr_comp_type CHECK (
        comparison_type IN (
            'PRIOR_PERIOD', 'BASE_YEAR', 'BUDGET', 'TARGET', 'BENCHMARK'
        )
    ),
    CONSTRAINT chk_p049_rr_scope CHECK (
        scope IN (
            'SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET',
            'SCOPE_3', 'TOTAL', 'NET_TOTAL'
        )
    ),
    CONSTRAINT chk_p049_rr_current CHECK (current_value_tco2e >= 0),
    CONSTRAINT chk_p049_rr_comparison CHECK (comparison_value_tco2e >= 0),
    CONSTRAINT chk_p049_rr_threshold CHECK (
        significance_threshold_pct >= 0 AND significance_threshold_pct <= 100
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_rr_tenant          ON ghg_multisite.gl_ms_reconciliation_results(tenant_id);
CREATE INDEX idx_p049_rr_current_run     ON ghg_multisite.gl_ms_reconciliation_results(current_run_id);
CREATE INDEX idx_p049_rr_comparison_run  ON ghg_multisite.gl_ms_reconciliation_results(comparison_run_id)
    WHERE comparison_run_id IS NOT NULL;
CREATE INDEX idx_p049_rr_type            ON ghg_multisite.gl_ms_reconciliation_results(comparison_type);
CREATE INDEX idx_p049_rr_scope           ON ghg_multisite.gl_ms_reconciliation_results(scope);
CREATE INDEX idx_p049_rr_significant     ON ghg_multisite.gl_ms_reconciliation_results(current_run_id)
    WHERE is_significant = true;
CREATE INDEX idx_p049_rr_site            ON ghg_multisite.gl_ms_reconciliation_results(site_id)
    WHERE site_id IS NOT NULL;
CREATE INDEX idx_p049_rr_factors         ON ghg_multisite.gl_ms_reconciliation_results USING gin(contributing_factors);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_reconciliation_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_rr_tenant_isolation ON ghg_multisite.gl_ms_reconciliation_results
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_multisite.gl_ms_consolidation_runs IS
    'PACK-049: Consolidation runs (5 types, 5 statuses) with scope totals, eliminations, and net result.';
COMMENT ON TABLE ghg_multisite.gl_ms_site_totals IS
    'PACK-049: Per-site gross and consolidated emissions with ownership/consolidation percentages applied.';
COMMENT ON TABLE ghg_multisite.gl_ms_elimination_entries IS
    'PACK-049: Intercompany elimination entries (6 types) to prevent double counting.';
COMMENT ON TABLE ghg_multisite.gl_ms_reconciliation_results IS
    'PACK-049: Period-over-period reconciliation (5 comparison types) with significance assessment.';

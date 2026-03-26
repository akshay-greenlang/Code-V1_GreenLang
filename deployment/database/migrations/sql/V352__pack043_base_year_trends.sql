-- =============================================================================
-- V352: PACK-043 Scope 3 Complete Pack - Base Year Management & Multi-Year Trends
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates base year management and multi-year trend analysis tables. Supports
-- base year establishment per GHG Protocol, base year recalculation triggers
-- (structural changes, methodology updates, errors), per-category base year
-- data, and multi-year time series for trend analysis and target tracking.
-- Implements GHG Protocol requirements for base year recalculation policy
-- including significance thresholds and recalculation triggers.
--
-- Tables (4):
--   1. ghg_accounting_scope3_complete.base_years
--   2. ghg_accounting_scope3_complete.base_year_categories
--   3. ghg_accounting_scope3_complete.recalculations
--   4. ghg_accounting_scope3_complete.multi_year_data
--
-- Enums (1):
--   1. ghg_accounting_scope3_complete.recalculation_trigger_type
--
-- Also includes: indexes, RLS, comments.
-- Previous: V351__pack043_climate_risk.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- ---------------------------------------------------------------------------
-- Enum: recalculation_trigger_type
-- ---------------------------------------------------------------------------
-- Reasons for recalculating the base year per GHG Protocol guidance.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'recalculation_trigger_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.recalculation_trigger_type AS ENUM (
            'STRUCTURAL_CHANGE',      -- M&A, divestiture, outsourcing
            'METHODOLOGY_CHANGE',     -- Updated calculation methodology
            'DATA_IMPROVEMENT',       -- Better data quality or availability
            'ERROR_CORRECTION',       -- Correction of previously reported errors
            'BOUNDARY_CHANGE',        -- Change in organizational boundary
            'EF_UPDATE',              -- Emission factor database update
            'POLICY_CHANGE',          -- Change in recalculation policy
            'SIGNIFICANCE_TRIGGER'    -- Cumulative threshold exceeded
        );
    END IF;
END;
$$;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.base_years
-- =============================================================================
-- Base year establishment for Scope 3 inventory per GHG Protocol. Each
-- record defines a base year with total Scope 3 emissions, methodology
-- version, recalculation policy (including significance threshold), and
-- the date the base year was established. An organization typically has
-- one active base year at a time.

CREATE TABLE ghg_accounting_scope3_complete.base_years (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    -- Base year
    base_year                   INTEGER         NOT NULL,
    total_scope3_tco2e          DECIMAL(15,3)   NOT NULL,
    total_scope1_2_tco2e        DECIMAL(15,3),
    total_all_scopes_tco2e      DECIMAL(15,3),
    -- Methodology
    methodology_version         VARCHAR(100)    NOT NULL DEFAULT 'GHG_PROTOCOL_SCOPE3_V1',
    gwp_source                  VARCHAR(20)     DEFAULT 'AR5',
    boundary_approach           VARCHAR(30)     DEFAULT 'OPERATIONAL_CONTROL',
    -- Categories summary
    categories_included         INTEGER         NOT NULL DEFAULT 0,
    categories_excluded         INTEGER         NOT NULL DEFAULT 0,
    -- Recalculation policy
    significance_threshold_pct  DECIMAL(5,2)    NOT NULL DEFAULT 5.00,
    recalculation_policy        TEXT,
    cumulative_change_pct       DECIMAL(8,2)    DEFAULT 0,
    recalculation_count         INTEGER         DEFAULT 0,
    last_recalculation_date     TIMESTAMPTZ,
    -- Establishment
    established_date            DATE            NOT NULL DEFAULT CURRENT_DATE,
    established_by              VARCHAR(255),
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    superseded_by               UUID,
    superseded_date             DATE,
    -- Verification
    verified                    BOOLEAN         NOT NULL DEFAULT false,
    verified_by                 VARCHAR(255),
    verified_at                 TIMESTAMPTZ,
    assurance_level             ghg_accounting_scope3_complete.assurance_level,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_by_year CHECK (base_year >= 1990 AND base_year <= 2100),
    CONSTRAINT chk_p043_by_total CHECK (total_scope3_tco2e >= 0),
    CONSTRAINT chk_p043_by_scope12 CHECK (total_scope1_2_tco2e IS NULL OR total_scope1_2_tco2e >= 0),
    CONSTRAINT chk_p043_by_all CHECK (total_all_scopes_tco2e IS NULL OR total_all_scopes_tco2e >= 0),
    CONSTRAINT chk_p043_by_methodology CHECK (
        methodology_version IN ('GHG_PROTOCOL_SCOPE3_V1', 'ISO_14064_1_2018', 'CUSTOM')
    ),
    CONSTRAINT chk_p043_by_gwp CHECK (
        gwp_source IS NULL OR gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p043_by_boundary CHECK (
        boundary_approach IS NULL OR boundary_approach IN (
            'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE'
        )
    ),
    CONSTRAINT chk_p043_by_included CHECK (
        categories_included >= 0 AND categories_included <= 15
    ),
    CONSTRAINT chk_p043_by_excluded CHECK (
        categories_excluded >= 0 AND categories_excluded <= 15
    ),
    CONSTRAINT chk_p043_by_threshold CHECK (
        significance_threshold_pct >= 0 AND significance_threshold_pct <= 100
    ),
    CONSTRAINT chk_p043_by_recount CHECK (recalculation_count IS NULL OR recalculation_count >= 0),
    CONSTRAINT uq_p043_by_org_year UNIQUE (org_id, base_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_by_tenant             ON ghg_accounting_scope3_complete.base_years(tenant_id);
CREATE INDEX idx_p043_by_org               ON ghg_accounting_scope3_complete.base_years(org_id);
CREATE INDEX idx_p043_by_year              ON ghg_accounting_scope3_complete.base_years(base_year);
CREATE INDEX idx_p043_by_active            ON ghg_accounting_scope3_complete.base_years(is_active) WHERE is_active = true;
CREATE INDEX idx_p043_by_verified          ON ghg_accounting_scope3_complete.base_years(verified) WHERE verified = true;
CREATE INDEX idx_p043_by_methodology       ON ghg_accounting_scope3_complete.base_years(methodology_version);
CREATE INDEX idx_p043_by_created           ON ghg_accounting_scope3_complete.base_years(created_at DESC);

-- Composite: org + active base year (should be at most one)
CREATE INDEX idx_p043_by_org_active        ON ghg_accounting_scope3_complete.base_years(org_id)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_by_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.base_years
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.base_year_categories
-- =============================================================================
-- Per-category breakdown of base year emissions. Each of the 15 categories
-- has a base year tCO2e value, methodology tier, and data quality rating.
-- This granularity supports per-category recalculation and trend analysis.

CREATE TABLE ghg_accounting_scope3_complete.base_year_categories (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    base_year_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.base_years(id) ON DELETE CASCADE,
    -- Category
    category                    ghg_accounting_scope3_complete.scope3_category_type NOT NULL,
    -- Emissions
    tco2e                       DECIMAL(15,3)   NOT NULL DEFAULT 0,
    pct_of_total                DECIMAL(5,2),
    -- Methodology
    tier                        ghg_accounting_scope3_complete.maturity_level NOT NULL DEFAULT 'LEVEL_1',
    methodology_detail          VARCHAR(200),
    -- Data quality
    dqr                         DECIMAL(3,1)    NOT NULL DEFAULT 5.0,
    primary_data_pct            DECIMAL(5,2)    DEFAULT 0,
    data_source                 VARCHAR(200),
    -- Inclusion
    included                    BOOLEAN         NOT NULL DEFAULT true,
    exclusion_reason            TEXT,
    -- Recalculated values
    original_tco2e              DECIMAL(15,3),
    is_recalculated             BOOLEAN         NOT NULL DEFAULT false,
    recalculation_date          TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_byc_tco2e CHECK (tco2e >= 0),
    CONSTRAINT chk_p043_byc_pct CHECK (
        pct_of_total IS NULL OR (pct_of_total >= 0 AND pct_of_total <= 100)
    ),
    CONSTRAINT chk_p043_byc_dqr CHECK (dqr >= 1.0 AND dqr <= 5.0),
    CONSTRAINT chk_p043_byc_primary CHECK (
        primary_data_pct IS NULL OR (primary_data_pct >= 0 AND primary_data_pct <= 100)
    ),
    CONSTRAINT chk_p043_byc_original CHECK (original_tco2e IS NULL OR original_tco2e >= 0),
    CONSTRAINT chk_p043_byc_exclusion CHECK (
        included = true OR exclusion_reason IS NOT NULL
    ),
    CONSTRAINT uq_p043_byc_base_category UNIQUE (base_year_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_byc_tenant            ON ghg_accounting_scope3_complete.base_year_categories(tenant_id);
CREATE INDEX idx_p043_byc_base_year         ON ghg_accounting_scope3_complete.base_year_categories(base_year_id);
CREATE INDEX idx_p043_byc_category          ON ghg_accounting_scope3_complete.base_year_categories(category);
CREATE INDEX idx_p043_byc_tco2e             ON ghg_accounting_scope3_complete.base_year_categories(tco2e DESC);
CREATE INDEX idx_p043_byc_tier              ON ghg_accounting_scope3_complete.base_year_categories(tier);
CREATE INDEX idx_p043_byc_dqr              ON ghg_accounting_scope3_complete.base_year_categories(dqr);
CREATE INDEX idx_p043_byc_included          ON ghg_accounting_scope3_complete.base_year_categories(included) WHERE included = true;
CREATE INDEX idx_p043_byc_recalculated      ON ghg_accounting_scope3_complete.base_year_categories(is_recalculated) WHERE is_recalculated = true;
CREATE INDEX idx_p043_byc_created           ON ghg_accounting_scope3_complete.base_year_categories(created_at DESC);

-- Composite: base year + ranked categories
CREATE INDEX idx_p043_byc_base_ranked       ON ghg_accounting_scope3_complete.base_year_categories(base_year_id, tco2e DESC)
    WHERE included = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_byc_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.base_year_categories
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.recalculations
-- =============================================================================
-- Base year recalculation events. Each record documents a recalculation
-- trigger, the original and recalculated values, the percentage impact,
-- and whether the significance threshold was met. Provides a full audit
-- trail of all base year adjustments per GHG Protocol requirements.

CREATE TABLE ghg_accounting_scope3_complete.recalculations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    base_year_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.base_years(id) ON DELETE CASCADE,
    -- Trigger
    trigger_type                ghg_accounting_scope3_complete.recalculation_trigger_type NOT NULL,
    trigger_description         TEXT            NOT NULL,
    trigger_date                DATE            NOT NULL,
    -- Scope
    affected_categories         ghg_accounting_scope3_complete.scope3_category_type[],
    -- Values
    original_tco2e              DECIMAL(15,3)   NOT NULL,
    recalculated_tco2e          DECIMAL(15,3)   NOT NULL,
    impact_tco2e                DECIMAL(15,3)   GENERATED ALWAYS AS (
        recalculated_tco2e - original_tco2e
    ) STORED,
    impact_pct                  DECIMAL(8,2)    GENERATED ALWAYS AS (
        CASE WHEN original_tco2e > 0
            THEN ROUND(((recalculated_tco2e - original_tco2e) / original_tco2e * 100)::NUMERIC, 2)
            ELSE 0
        END
    ) STORED,
    -- Significance
    significance_met            BOOLEAN         NOT NULL DEFAULT false,
    significance_threshold_pct  DECIMAL(5,2),
    -- Methodology
    methodology_change          TEXT,
    ef_changes                  JSONB           DEFAULT '[]',
    boundary_changes            JSONB           DEFAULT '[]',
    -- Decision
    rationale                   TEXT            NOT NULL,
    decision                    VARCHAR(30)     NOT NULL DEFAULT 'RECALCULATE',
    decided_by                  VARCHAR(255),
    decided_at                  TIMESTAMPTZ,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    applied_date                DATE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_rc_original CHECK (original_tco2e >= 0),
    CONSTRAINT chk_p043_rc_recalculated CHECK (recalculated_tco2e >= 0),
    CONSTRAINT chk_p043_rc_threshold CHECK (
        significance_threshold_pct IS NULL OR (significance_threshold_pct >= 0 AND significance_threshold_pct <= 100)
    ),
    CONSTRAINT chk_p043_rc_decision CHECK (
        decision IN ('RECALCULATE', 'NO_CHANGE', 'DEFER', 'PARTIAL_ADJUSTMENT')
    ),
    CONSTRAINT chk_p043_rc_status CHECK (
        status IN ('DRAFT', 'UNDER_REVIEW', 'APPROVED', 'COMPLETED', 'REJECTED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_rc_tenant             ON ghg_accounting_scope3_complete.recalculations(tenant_id);
CREATE INDEX idx_p043_rc_base_year          ON ghg_accounting_scope3_complete.recalculations(base_year_id);
CREATE INDEX idx_p043_rc_trigger            ON ghg_accounting_scope3_complete.recalculations(trigger_type);
CREATE INDEX idx_p043_rc_trigger_date       ON ghg_accounting_scope3_complete.recalculations(trigger_date DESC);
CREATE INDEX idx_p043_rc_significance       ON ghg_accounting_scope3_complete.recalculations(significance_met) WHERE significance_met = true;
CREATE INDEX idx_p043_rc_decision           ON ghg_accounting_scope3_complete.recalculations(decision);
CREATE INDEX idx_p043_rc_status             ON ghg_accounting_scope3_complete.recalculations(status);
CREATE INDEX idx_p043_rc_created            ON ghg_accounting_scope3_complete.recalculations(created_at DESC);
CREATE INDEX idx_p043_rc_categories         ON ghg_accounting_scope3_complete.recalculations USING GIN(affected_categories);

-- Composite: base year + chronological recalculations
CREATE INDEX idx_p043_rc_base_chrono        ON ghg_accounting_scope3_complete.recalculations(base_year_id, trigger_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_rc_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.recalculations
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.multi_year_data
-- =============================================================================
-- Multi-year time series of Scope 3 emissions per category. Each row
-- represents one category for one reporting year, enabling year-over-year
-- trend analysis, growth rate calculations, and recalculation tracking.

CREATE TABLE ghg_accounting_scope3_complete.multi_year_data (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    -- Year and category
    reporting_year              INTEGER         NOT NULL,
    category                    ghg_accounting_scope3_complete.scope3_category_type NOT NULL,
    -- Emissions
    tco2e                       DECIMAL(15,3)   NOT NULL DEFAULT 0,
    pct_of_total_scope3         DECIMAL(5,2),
    -- Methodology
    tier                        ghg_accounting_scope3_complete.maturity_level NOT NULL DEFAULT 'LEVEL_1',
    methodology_detail          VARCHAR(200),
    -- Data quality
    dqr                         DECIMAL(3,1)    NOT NULL DEFAULT 5.0,
    primary_data_pct            DECIMAL(5,2)    DEFAULT 0,
    -- Year-over-year
    prior_year_tco2e            DECIMAL(15,3),
    yoy_change_pct              DECIMAL(8,2),
    yoy_change_tco2e            DECIMAL(15,3),
    -- Base year comparison
    base_year_tco2e             DECIMAL(15,3),
    vs_base_year_pct            DECIMAL(8,2),
    -- Recalculation
    recalculated                BOOLEAN         NOT NULL DEFAULT false,
    original_tco2e              DECIMAL(15,3),
    methodology_adjusted        BOOLEAN         NOT NULL DEFAULT false,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_myd_year CHECK (reporting_year >= 1990 AND reporting_year <= 2100),
    CONSTRAINT chk_p043_myd_tco2e CHECK (tco2e >= 0),
    CONSTRAINT chk_p043_myd_pct CHECK (
        pct_of_total_scope3 IS NULL OR (pct_of_total_scope3 >= 0 AND pct_of_total_scope3 <= 100)
    ),
    CONSTRAINT chk_p043_myd_dqr CHECK (dqr >= 1.0 AND dqr <= 5.0),
    CONSTRAINT chk_p043_myd_primary CHECK (
        primary_data_pct IS NULL OR (primary_data_pct >= 0 AND primary_data_pct <= 100)
    ),
    CONSTRAINT chk_p043_myd_prior CHECK (prior_year_tco2e IS NULL OR prior_year_tco2e >= 0),
    CONSTRAINT chk_p043_myd_base CHECK (base_year_tco2e IS NULL OR base_year_tco2e >= 0),
    CONSTRAINT chk_p043_myd_original CHECK (original_tco2e IS NULL OR original_tco2e >= 0),
    CONSTRAINT uq_p043_myd_org_year_cat UNIQUE (org_id, reporting_year, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_myd_tenant            ON ghg_accounting_scope3_complete.multi_year_data(tenant_id);
CREATE INDEX idx_p043_myd_org              ON ghg_accounting_scope3_complete.multi_year_data(org_id);
CREATE INDEX idx_p043_myd_year             ON ghg_accounting_scope3_complete.multi_year_data(reporting_year);
CREATE INDEX idx_p043_myd_category         ON ghg_accounting_scope3_complete.multi_year_data(category);
CREATE INDEX idx_p043_myd_tco2e            ON ghg_accounting_scope3_complete.multi_year_data(tco2e DESC);
CREATE INDEX idx_p043_myd_tier             ON ghg_accounting_scope3_complete.multi_year_data(tier);
CREATE INDEX idx_p043_myd_dqr             ON ghg_accounting_scope3_complete.multi_year_data(dqr);
CREATE INDEX idx_p043_myd_recalculated     ON ghg_accounting_scope3_complete.multi_year_data(recalculated) WHERE recalculated = true;
CREATE INDEX idx_p043_myd_created          ON ghg_accounting_scope3_complete.multi_year_data(created_at DESC);

-- Composite: org + year for annual total
CREATE INDEX idx_p043_myd_org_year          ON ghg_accounting_scope3_complete.multi_year_data(org_id, reporting_year DESC);

-- Composite: org + category + year for trend chart
CREATE INDEX idx_p043_myd_org_cat_year      ON ghg_accounting_scope3_complete.multi_year_data(org_id, category, reporting_year);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_myd_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.multi_year_data
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.base_years ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.base_year_categories ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.recalculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.multi_year_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_by_tenant_isolation ON ghg_accounting_scope3_complete.base_years
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_by_service_bypass ON ghg_accounting_scope3_complete.base_years
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_byc_tenant_isolation ON ghg_accounting_scope3_complete.base_year_categories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_byc_service_bypass ON ghg_accounting_scope3_complete.base_year_categories
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_rc_tenant_isolation ON ghg_accounting_scope3_complete.recalculations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_rc_service_bypass ON ghg_accounting_scope3_complete.recalculations
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_myd_tenant_isolation ON ghg_accounting_scope3_complete.multi_year_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_myd_service_bypass ON ghg_accounting_scope3_complete.multi_year_data
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.base_years TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.base_year_categories TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.recalculations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.multi_year_data TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3_complete.base_years IS
    'Base year establishment per GHG Protocol with total emissions, methodology, recalculation policy (significance threshold), and verification.';
COMMENT ON TABLE ghg_accounting_scope3_complete.base_year_categories IS
    'Per-category base year emissions with methodology tier, DQR, and recalculation tracking for 15 Scope 3 categories.';
COMMENT ON TABLE ghg_accounting_scope3_complete.recalculations IS
    'Base year recalculation log documenting trigger, original/recalculated values, significance test, and decision rationale.';
COMMENT ON TABLE ghg_accounting_scope3_complete.multi_year_data IS
    'Multi-year Scope 3 time series per category for trend analysis with YoY change, base year comparison, and recalculation flags.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.base_years.significance_threshold_pct IS 'Percentage threshold (typically 5%) above which changes trigger recalculation per GHG Protocol.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.base_years.cumulative_change_pct IS 'Cumulative percentage change since base year establishment (triggers recalculation if > threshold).';

COMMENT ON COLUMN ghg_accounting_scope3_complete.recalculations.impact_tco2e IS 'Generated column: recalculated - original (positive = increase).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.recalculations.impact_pct IS 'Generated column: ((recalculated - original) / original) * 100.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.recalculations.significance_met IS 'Whether the recalculation impact exceeds the significance threshold.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.multi_year_data.yoy_change_pct IS 'Year-over-year percentage change vs prior year.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.multi_year_data.vs_base_year_pct IS 'Percentage change vs base year value for target tracking.';

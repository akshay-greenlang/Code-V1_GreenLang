-- =============================================================================
-- V350: PACK-043 Scope 3 Complete Pack - Supplier Programme Management
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates supplier programme management tables for structured Scope 3
-- reduction through value chain engagement. Supports supplier-level targets,
-- climate commitment tracking (SBTs, RE100, CDP), annual progress reporting,
-- multi-dimensional scorecards (emission, quality, engagement, commitment),
-- and overall programme performance metrics with ROI tracking.
--
-- Tables (5):
--   1. ghg_accounting_scope3_complete.supplier_targets
--   2. ghg_accounting_scope3_complete.supplier_commitments
--   3. ghg_accounting_scope3_complete.supplier_progress
--   4. ghg_accounting_scope3_complete.supplier_scorecards
--   5. ghg_accounting_scope3_complete.programme_metrics
--
-- Enums (1):
--   1. ghg_accounting_scope3_complete.commitment_type
--
-- Also includes: indexes, RLS, comments.
-- Previous: V349__pack043_sbti_targets.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- ---------------------------------------------------------------------------
-- Enum: commitment_type
-- ---------------------------------------------------------------------------
-- Types of climate commitments a supplier may hold.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'commitment_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.commitment_type AS ENUM (
            'SBTI_COMMITTED',       -- Committed to SBTi but not yet validated
            'SBTI_VALIDATED',       -- SBTi targets validated
            'NET_ZERO',             -- Net-zero commitment
            'RE100',                -- 100% renewable electricity commitment
            'CARBON_NEUTRAL',       -- Carbon neutrality claim
            'CDP_DISCLOSURE',       -- Disclosing through CDP Supply Chain
            'CLIMATE_PLEDGE',       -- Amazon Climate Pledge or similar
            'INTERNAL_TARGET',      -- Internal reduction target (not SBTi)
            'OTHER'                 -- Other climate commitment
        );
    END IF;
END;
$$;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.supplier_targets
-- =============================================================================
-- Per-supplier emission reduction targets set by the reporting organisation
-- as part of its supplier engagement programme. Each target defines a base
-- year, baseline emissions, target year, and required reduction percentage.

CREATE TABLE ghg_accounting_scope3_complete.supplier_targets (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    supplier_id                 UUID            NOT NULL,
    -- Target definition
    target_name                 VARCHAR(500),
    target_reduction_pct        DECIMAL(5,2)    NOT NULL,
    -- Base year
    base_year                   INTEGER         NOT NULL,
    base_year_tco2e             DECIMAL(15,3)   NOT NULL,
    -- Target year
    target_year                 INTEGER         NOT NULL,
    target_tco2e                DECIMAL(15,3),
    -- Scope
    target_scope                VARCHAR(50)     NOT NULL DEFAULT 'SCOPE_1_2',
    includes_scope3             BOOLEAN         NOT NULL DEFAULT false,
    -- Alignment
    sbti_aligned                BOOLEAN         NOT NULL DEFAULT false,
    alignment_level             VARCHAR(30),
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    communicated_date           DATE,
    accepted_date               DATE,
    -- Progress
    latest_reported_tco2e       DECIMAL(15,3),
    latest_reported_year        INTEGER,
    current_reduction_pct       DECIMAL(5,2),
    on_track                    BOOLEAN,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_st_reduction CHECK (
        target_reduction_pct >= 0 AND target_reduction_pct <= 100
    ),
    CONSTRAINT chk_p043_st_base_year CHECK (base_year >= 2000 AND base_year <= 2100),
    CONSTRAINT chk_p043_st_base_tco2e CHECK (base_year_tco2e >= 0),
    CONSTRAINT chk_p043_st_target_year CHECK (target_year > base_year AND target_year <= 2100),
    CONSTRAINT chk_p043_st_target_tco2e CHECK (target_tco2e IS NULL OR target_tco2e >= 0),
    CONSTRAINT chk_p043_st_scope CHECK (
        target_scope IN ('SCOPE_1_2', 'SCOPE_1_2_3', 'SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    ),
    CONSTRAINT chk_p043_st_alignment CHECK (
        alignment_level IS NULL OR alignment_level IN ('1.5C', 'WELL_BELOW_2C', '2C', 'NOT_ALIGNED')
    ),
    CONSTRAINT chk_p043_st_status CHECK (
        status IN ('DRAFT', 'ACTIVE', 'ON_HOLD', 'ACHIEVED', 'MISSED', 'WITHDRAWN')
    ),
    CONSTRAINT chk_p043_st_latest CHECK (
        latest_reported_tco2e IS NULL OR latest_reported_tco2e >= 0
    ),
    CONSTRAINT chk_p043_st_current_pct CHECK (
        current_reduction_pct IS NULL OR (current_reduction_pct >= -100 AND current_reduction_pct <= 100)
    ),
    CONSTRAINT uq_p043_st_supplier_target_year UNIQUE (supplier_id, target_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_st_tenant             ON ghg_accounting_scope3_complete.supplier_targets(tenant_id);
CREATE INDEX idx_p043_st_supplier           ON ghg_accounting_scope3_complete.supplier_targets(supplier_id);
CREATE INDEX idx_p043_st_base_year          ON ghg_accounting_scope3_complete.supplier_targets(base_year);
CREATE INDEX idx_p043_st_target_year        ON ghg_accounting_scope3_complete.supplier_targets(target_year);
CREATE INDEX idx_p043_st_reduction          ON ghg_accounting_scope3_complete.supplier_targets(target_reduction_pct DESC);
CREATE INDEX idx_p043_st_sbti               ON ghg_accounting_scope3_complete.supplier_targets(sbti_aligned) WHERE sbti_aligned = true;
CREATE INDEX idx_p043_st_status             ON ghg_accounting_scope3_complete.supplier_targets(status);
CREATE INDEX idx_p043_st_on_track           ON ghg_accounting_scope3_complete.supplier_targets(on_track);
CREATE INDEX idx_p043_st_created            ON ghg_accounting_scope3_complete.supplier_targets(created_at DESC);

-- Composite: tenant + active targets
CREATE INDEX idx_p043_st_tenant_active      ON ghg_accounting_scope3_complete.supplier_targets(tenant_id, supplier_id)
    WHERE status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_st_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.supplier_targets
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.supplier_commitments
-- =============================================================================
-- Climate commitments held by suppliers (SBTi, RE100, CDP, etc.). Tracks
-- commitment type, date, target year, verification status, and evidence.

CREATE TABLE ghg_accounting_scope3_complete.supplier_commitments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    supplier_id                 UUID            NOT NULL,
    -- Commitment
    commitment_type             ghg_accounting_scope3_complete.commitment_type NOT NULL,
    commitment_name             VARCHAR(500),
    commitment_date             DATE            NOT NULL,
    target_year                 INTEGER,
    -- Details
    commitment_details          TEXT,
    reduction_target_pct        DECIMAL(5,2),
    base_year                   INTEGER,
    -- Verification
    verified                    BOOLEAN         NOT NULL DEFAULT false,
    verification_date           DATE,
    verification_body           VARCHAR(255),
    source                      VARCHAR(500),
    source_url                  TEXT,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    expiry_date                 DATE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_sc_target_year CHECK (
        target_year IS NULL OR (target_year >= 2020 AND target_year <= 2100)
    ),
    CONSTRAINT chk_p043_sc_reduction CHECK (
        reduction_target_pct IS NULL OR (reduction_target_pct >= 0 AND reduction_target_pct <= 100)
    ),
    CONSTRAINT chk_p043_sc_base_year CHECK (
        base_year IS NULL OR (base_year >= 2000 AND base_year <= 2100)
    ),
    CONSTRAINT chk_p043_sc_status CHECK (
        status IN ('ACTIVE', 'EXPIRED', 'WITHDRAWN', 'SUPERSEDED', 'PENDING_VERIFICATION')
    ),
    CONSTRAINT uq_p043_sc_supplier_type UNIQUE (supplier_id, commitment_type, commitment_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_sc2_tenant            ON ghg_accounting_scope3_complete.supplier_commitments(tenant_id);
CREATE INDEX idx_p043_sc2_supplier          ON ghg_accounting_scope3_complete.supplier_commitments(supplier_id);
CREATE INDEX idx_p043_sc2_type              ON ghg_accounting_scope3_complete.supplier_commitments(commitment_type);
CREATE INDEX idx_p043_sc2_date              ON ghg_accounting_scope3_complete.supplier_commitments(commitment_date DESC);
CREATE INDEX idx_p043_sc2_target_year       ON ghg_accounting_scope3_complete.supplier_commitments(target_year);
CREATE INDEX idx_p043_sc2_verified          ON ghg_accounting_scope3_complete.supplier_commitments(verified) WHERE verified = true;
CREATE INDEX idx_p043_sc2_status            ON ghg_accounting_scope3_complete.supplier_commitments(status);
CREATE INDEX idx_p043_sc2_created           ON ghg_accounting_scope3_complete.supplier_commitments(created_at DESC);

-- Composite: supplier + active commitments
CREATE INDEX idx_p043_sc2_sup_active        ON ghg_accounting_scope3_complete.supplier_commitments(supplier_id, commitment_type)
    WHERE status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_sc2_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.supplier_commitments
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.supplier_progress
-- =============================================================================
-- Annual progress reports from suppliers against their targets. Tracks
-- reported emissions, reduction percentage, data quality level, and
-- verification status for each reporting year.

CREATE TABLE ghg_accounting_scope3_complete.supplier_progress (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    supplier_id                 UUID            NOT NULL,
    -- Reporting period
    reporting_year              INTEGER         NOT NULL,
    reporting_period_start      DATE,
    reporting_period_end        DATE,
    -- Emissions
    reported_tco2e              DECIMAL(15,3)   NOT NULL,
    scope1_tco2e                DECIMAL(15,3),
    scope2_tco2e                DECIMAL(15,3),
    scope3_tco2e                DECIMAL(15,3),
    -- Reduction
    reduction_pct               DECIMAL(5,2),
    reduction_vs_base_year      DECIMAL(5,2),
    -- Data quality
    data_quality_level          VARCHAR(20)     NOT NULL DEFAULT 'AVERAGE_DATA',
    primary_data_pct            DECIMAL(5,2),
    completeness_pct            DECIMAL(5,2),
    -- Verification
    verified                    BOOLEAN         NOT NULL DEFAULT false,
    verification_body           VARCHAR(255),
    verification_level          VARCHAR(30),
    -- Methodology
    methodology                 VARCHAR(200),
    boundary_approach           VARCHAR(50),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_sp_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_p043_sp_dates CHECK (
        reporting_period_start IS NULL OR reporting_period_end IS NULL OR
        reporting_period_start <= reporting_period_end
    ),
    CONSTRAINT chk_p043_sp_tco2e CHECK (reported_tco2e >= 0),
    CONSTRAINT chk_p043_sp_scope1 CHECK (scope1_tco2e IS NULL OR scope1_tco2e >= 0),
    CONSTRAINT chk_p043_sp_scope2 CHECK (scope2_tco2e IS NULL OR scope2_tco2e >= 0),
    CONSTRAINT chk_p043_sp_scope3 CHECK (scope3_tco2e IS NULL OR scope3_tco2e >= 0),
    CONSTRAINT chk_p043_sp_reduction CHECK (
        reduction_pct IS NULL OR (reduction_pct >= -1000 AND reduction_pct <= 100)
    ),
    CONSTRAINT chk_p043_sp_quality CHECK (
        data_quality_level IN ('SPEND_BASED', 'AVERAGE_DATA', 'SUPPLIER_SPECIFIC', 'VERIFIED')
    ),
    CONSTRAINT chk_p043_sp_primary CHECK (
        primary_data_pct IS NULL OR (primary_data_pct >= 0 AND primary_data_pct <= 100)
    ),
    CONSTRAINT chk_p043_sp_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p043_sp_verification CHECK (
        verification_level IS NULL OR verification_level IN (
            'SELF_DECLARED', 'LIMITED_ASSURANCE', 'REASONABLE_ASSURANCE', 'THIRD_PARTY'
        )
    ),
    CONSTRAINT uq_p043_sp_supplier_year UNIQUE (supplier_id, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_sp_tenant             ON ghg_accounting_scope3_complete.supplier_progress(tenant_id);
CREATE INDEX idx_p043_sp_supplier           ON ghg_accounting_scope3_complete.supplier_progress(supplier_id);
CREATE INDEX idx_p043_sp_year               ON ghg_accounting_scope3_complete.supplier_progress(reporting_year);
CREATE INDEX idx_p043_sp_tco2e              ON ghg_accounting_scope3_complete.supplier_progress(reported_tco2e DESC);
CREATE INDEX idx_p043_sp_reduction          ON ghg_accounting_scope3_complete.supplier_progress(reduction_pct);
CREATE INDEX idx_p043_sp_quality            ON ghg_accounting_scope3_complete.supplier_progress(data_quality_level);
CREATE INDEX idx_p043_sp_verified           ON ghg_accounting_scope3_complete.supplier_progress(verified) WHERE verified = true;
CREATE INDEX idx_p043_sp_created            ON ghg_accounting_scope3_complete.supplier_progress(created_at DESC);

-- Composite: supplier + year for trend analysis
CREATE INDEX idx_p043_sp_sup_year           ON ghg_accounting_scope3_complete.supplier_progress(supplier_id, reporting_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_sp_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.supplier_progress
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.supplier_scorecards
-- =============================================================================
-- Multi-dimensional supplier scorecard aggregating emission performance,
-- data quality, engagement level, and climate commitment strength into a
-- composite score with tier classification.

CREATE TABLE ghg_accounting_scope3_complete.supplier_scorecards (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    supplier_id                 UUID            NOT NULL,
    -- Assessment
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessment_period_year      INTEGER         NOT NULL,
    -- Dimensional scores (0-100 each)
    emission_score              DECIMAL(5,2)    NOT NULL DEFAULT 0,
    quality_score               DECIMAL(5,2)    NOT NULL DEFAULT 0,
    engagement_score            DECIMAL(5,2)    NOT NULL DEFAULT 0,
    commitment_score            DECIMAL(5,2)    NOT NULL DEFAULT 0,
    -- Composite
    overall_score               DECIMAL(5,2)    NOT NULL DEFAULT 0,
    previous_score              DECIMAL(5,2),
    score_change                DECIMAL(5,2),
    -- Tier
    tier_classification         VARCHAR(20)     NOT NULL DEFAULT 'STANDARD',
    previous_tier               VARCHAR(20),
    -- Weights used
    emission_weight             DECIMAL(3,2)    NOT NULL DEFAULT 0.35,
    quality_weight              DECIMAL(3,2)    NOT NULL DEFAULT 0.25,
    engagement_weight           DECIMAL(3,2)    NOT NULL DEFAULT 0.20,
    commitment_weight           DECIMAL(3,2)    NOT NULL DEFAULT 0.20,
    -- Context
    supplier_spend_usd          NUMERIC(18,2),
    supplier_emission_tco2e     DECIMAL(15,3),
    supplier_emission_pct       DECIMAL(5,2),
    -- Actions
    improvement_actions         JSONB           DEFAULT '[]',
    recognition                 VARCHAR(100),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_ssc_emission CHECK (emission_score >= 0 AND emission_score <= 100),
    CONSTRAINT chk_p043_ssc_quality CHECK (quality_score >= 0 AND quality_score <= 100),
    CONSTRAINT chk_p043_ssc_engagement CHECK (engagement_score >= 0 AND engagement_score <= 100),
    CONSTRAINT chk_p043_ssc_commitment CHECK (commitment_score >= 0 AND commitment_score <= 100),
    CONSTRAINT chk_p043_ssc_overall CHECK (overall_score >= 0 AND overall_score <= 100),
    CONSTRAINT chk_p043_ssc_year CHECK (assessment_period_year >= 2000 AND assessment_period_year <= 2100),
    CONSTRAINT chk_p043_ssc_tier CHECK (
        tier_classification IN ('LEADER', 'ADVANCED', 'STANDARD', 'DEVELOPING', 'LAGGARD')
    ),
    CONSTRAINT chk_p043_ssc_weights CHECK (
        emission_weight + quality_weight + engagement_weight + commitment_weight BETWEEN 0.99 AND 1.01
    ),
    CONSTRAINT chk_p043_ssc_spend CHECK (supplier_spend_usd IS NULL OR supplier_spend_usd >= 0),
    CONSTRAINT chk_p043_ssc_emission_tco2e CHECK (supplier_emission_tco2e IS NULL OR supplier_emission_tco2e >= 0),
    CONSTRAINT chk_p043_ssc_emission_pct CHECK (
        supplier_emission_pct IS NULL OR (supplier_emission_pct >= 0 AND supplier_emission_pct <= 100)
    ),
    CONSTRAINT uq_p043_ssc_supplier_year UNIQUE (supplier_id, assessment_period_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_ssc_tenant            ON ghg_accounting_scope3_complete.supplier_scorecards(tenant_id);
CREATE INDEX idx_p043_ssc_supplier          ON ghg_accounting_scope3_complete.supplier_scorecards(supplier_id);
CREATE INDEX idx_p043_ssc_date              ON ghg_accounting_scope3_complete.supplier_scorecards(assessment_date DESC);
CREATE INDEX idx_p043_ssc_year              ON ghg_accounting_scope3_complete.supplier_scorecards(assessment_period_year);
CREATE INDEX idx_p043_ssc_overall           ON ghg_accounting_scope3_complete.supplier_scorecards(overall_score DESC);
CREATE INDEX idx_p043_ssc_tier              ON ghg_accounting_scope3_complete.supplier_scorecards(tier_classification);
CREATE INDEX idx_p043_ssc_emission          ON ghg_accounting_scope3_complete.supplier_scorecards(emission_score DESC);
CREATE INDEX idx_p043_ssc_created           ON ghg_accounting_scope3_complete.supplier_scorecards(created_at DESC);
CREATE INDEX idx_p043_ssc_actions           ON ghg_accounting_scope3_complete.supplier_scorecards USING GIN(improvement_actions);

-- Composite: tenant + tier for programme reporting
CREATE INDEX idx_p043_ssc_tenant_tier       ON ghg_accounting_scope3_complete.supplier_scorecards(tenant_id, tier_classification, overall_score DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_ssc_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.supplier_scorecards
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_accounting_scope3_complete.programme_metrics
-- =============================================================================
-- Aggregate supplier programme performance metrics by period. Tracks overall
-- programme KPIs: enrolled suppliers, response rates, average reduction,
-- programme costs, and ROI. Used for programme-level dashboard and reporting.

CREATE TABLE ghg_accounting_scope3_complete.programme_metrics (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    programme_id                UUID,
    -- Period
    period                      VARCHAR(20)     NOT NULL,
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    -- Supplier counts
    enrolled_suppliers          INTEGER         NOT NULL DEFAULT 0,
    responding_suppliers        INTEGER         NOT NULL DEFAULT 0,
    response_rate_pct           DECIMAL(5,2)    GENERATED ALWAYS AS (
        CASE WHEN enrolled_suppliers > 0
            THEN ROUND(((responding_suppliers::DECIMAL / enrolled_suppliers) * 100)::NUMERIC, 2)
            ELSE 0
        END
    ) STORED,
    suppliers_with_targets      INTEGER         DEFAULT 0,
    suppliers_on_track          INTEGER         DEFAULT 0,
    -- Emissions
    total_supplier_tco2e        DECIMAL(15,3),
    avg_reduction_pct           DECIMAL(5,2),
    total_reduction_tco2e       DECIMAL(15,3),
    -- Data quality
    suppliers_tier3             INTEGER         DEFAULT 0,
    avg_data_quality_score      DECIMAL(3,1),
    primary_data_coverage_pct   DECIMAL(5,2),
    -- Commitments
    suppliers_with_sbti         INTEGER         DEFAULT 0,
    suppliers_with_commitments  INTEGER         DEFAULT 0,
    -- Programme economics
    programme_cost              NUMERIC(14,2),
    cost_per_supplier           NUMERIC(14,2),
    cost_per_tco2e_reduced      NUMERIC(14,2),
    programme_roi               DECIMAL(8,2),
    -- Tier distribution
    tier_leader_count           INTEGER         DEFAULT 0,
    tier_advanced_count         INTEGER         DEFAULT 0,
    tier_standard_count         INTEGER         DEFAULT 0,
    tier_developing_count       INTEGER         DEFAULT 0,
    tier_laggard_count          INTEGER         DEFAULT 0,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_pm_period CHECK (period_start < period_end),
    CONSTRAINT chk_p043_pm_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_p043_pm_enrolled CHECK (enrolled_suppliers >= 0),
    CONSTRAINT chk_p043_pm_responding CHECK (responding_suppliers >= 0 AND responding_suppliers <= enrolled_suppliers),
    CONSTRAINT chk_p043_pm_targets CHECK (suppliers_with_targets IS NULL OR suppliers_with_targets >= 0),
    CONSTRAINT chk_p043_pm_on_track CHECK (suppliers_on_track IS NULL OR suppliers_on_track >= 0),
    CONSTRAINT chk_p043_pm_tco2e CHECK (total_supplier_tco2e IS NULL OR total_supplier_tco2e >= 0),
    CONSTRAINT chk_p043_pm_cost CHECK (programme_cost IS NULL OR programme_cost >= 0),
    CONSTRAINT chk_p043_pm_sbti CHECK (suppliers_with_sbti IS NULL OR suppliers_with_sbti >= 0),
    CONSTRAINT uq_p043_pm_tenant_period UNIQUE (tenant_id, programme_id, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_pm_tenant             ON ghg_accounting_scope3_complete.programme_metrics(tenant_id);
CREATE INDEX idx_p043_pm_programme          ON ghg_accounting_scope3_complete.programme_metrics(programme_id);
CREATE INDEX idx_p043_pm_year               ON ghg_accounting_scope3_complete.programme_metrics(reporting_year);
CREATE INDEX idx_p043_pm_period             ON ghg_accounting_scope3_complete.programme_metrics(period_start, period_end);
CREATE INDEX idx_p043_pm_enrolled           ON ghg_accounting_scope3_complete.programme_metrics(enrolled_suppliers DESC);
CREATE INDEX idx_p043_pm_reduction          ON ghg_accounting_scope3_complete.programme_metrics(avg_reduction_pct DESC);
CREATE INDEX idx_p043_pm_roi               ON ghg_accounting_scope3_complete.programme_metrics(programme_roi DESC);
CREATE INDEX idx_p043_pm_created            ON ghg_accounting_scope3_complete.programme_metrics(created_at DESC);

-- Composite: tenant + year for YoY comparison
CREATE INDEX idx_p043_pm_tenant_year        ON ghg_accounting_scope3_complete.programme_metrics(tenant_id, reporting_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_pm_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.programme_metrics
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.supplier_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.supplier_commitments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.supplier_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.supplier_scorecards ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.programme_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_st_tenant_isolation ON ghg_accounting_scope3_complete.supplier_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_st_service_bypass ON ghg_accounting_scope3_complete.supplier_targets
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_sc2_tenant_isolation ON ghg_accounting_scope3_complete.supplier_commitments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_sc2_service_bypass ON ghg_accounting_scope3_complete.supplier_commitments
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_sp_tenant_isolation ON ghg_accounting_scope3_complete.supplier_progress
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_sp_service_bypass ON ghg_accounting_scope3_complete.supplier_progress
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_ssc_tenant_isolation ON ghg_accounting_scope3_complete.supplier_scorecards
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_ssc_service_bypass ON ghg_accounting_scope3_complete.supplier_scorecards
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_pm_tenant_isolation ON ghg_accounting_scope3_complete.programme_metrics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_pm_service_bypass ON ghg_accounting_scope3_complete.programme_metrics
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.supplier_targets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.supplier_commitments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.supplier_progress TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.supplier_scorecards TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.programme_metrics TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3_complete.supplier_targets IS
    'Per-supplier emission reduction targets with base year, target year, and progress tracking for supplier engagement programme.';
COMMENT ON TABLE ghg_accounting_scope3_complete.supplier_commitments IS
    'Climate commitments held by suppliers (SBTi, RE100, CDP, etc.) with verification status and evidence source.';
COMMENT ON TABLE ghg_accounting_scope3_complete.supplier_progress IS
    'Annual supplier progress reports with reported emissions, reduction percentage, data quality, and verification status.';
COMMENT ON TABLE ghg_accounting_scope3_complete.supplier_scorecards IS
    'Multi-dimensional supplier scorecard (emission, quality, engagement, commitment) with composite score and tier classification.';
COMMENT ON TABLE ghg_accounting_scope3_complete.programme_metrics IS
    'Aggregate supplier programme KPIs by period with response rates, reductions, cost, and ROI for programme-level reporting.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.supplier_scorecards.tier_classification IS 'Supplier tier: LEADER, ADVANCED, STANDARD, DEVELOPING, LAGGARD based on composite score.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.supplier_scorecards.overall_score IS 'Weighted composite score: (emission * 0.35) + (quality * 0.25) + (engagement * 0.20) + (commitment * 0.20).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.programme_metrics.response_rate_pct IS 'Generated column: (responding / enrolled) * 100.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.programme_metrics.programme_roi IS 'Programme return on investment: value of emission reductions / programme cost.';

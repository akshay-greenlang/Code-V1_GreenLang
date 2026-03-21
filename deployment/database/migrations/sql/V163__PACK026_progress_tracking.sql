-- =============================================================================
-- V163: PACK-026 SME Net Zero - Progress Tracking
-- =============================================================================
-- Pack:         PACK-026 (SME Net Zero Pack)
-- Migration:    006 of 008
-- Date:         March 2026
--
-- Annual review records tracking actual vs target emissions with variance
-- analysis, on-track status, and financial outcomes. Quarterly snapshots
-- for lightweight progress monitoring of spend and quick wins.
--
-- Tables (2):
--   1. pack026_sme_net_zero.annual_reviews
--   2. pack026_sme_net_zero.quarterly_snapshots
--
-- Previous: V162__PACK026_accounting_integration.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack026_sme_net_zero.annual_reviews
-- =============================================================================
-- Annual emission review records comparing actual performance against targets
-- with scope-level breakdowns, variance analysis, on-track status, and
-- financial outcome tracking (grants received, cost savings realized).

CREATE TABLE pack026_sme_net_zero.annual_reviews (
    review_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    target_id               UUID            REFERENCES pack026_sme_net_zero.sme_targets(target_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    review_year             INTEGER         NOT NULL,
    -- Actual emissions
    actual_scope1           DECIMAL(18,4)   NOT NULL DEFAULT 0,
    actual_scope2           DECIMAL(18,4)   NOT NULL DEFAULT 0,
    actual_scope3           DECIMAL(18,4)   DEFAULT 0,
    actual_total            DECIMAL(18,4)   GENERATED ALWAYS AS (
        actual_scope1 + actual_scope2 + COALESCE(actual_scope3, 0)
    ) STORED,
    -- Target comparison
    target_total            DECIMAL(18,4),
    variance_tco2e          DECIMAL(18,4),
    variance_pct            DECIMAL(8,3),
    -- Status
    on_track_status         VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    reduction_from_baseline_pct DECIMAL(8,3),
    annual_reduction_rate   DECIMAL(8,3),
    -- Intensity
    intensity_per_employee  DECIMAL(18,6),
    intensity_per_revenue   DECIMAL(18,8),
    -- Actions
    actions_completed_count INTEGER         DEFAULT 0,
    actions_in_progress_count INTEGER       DEFAULT 0,
    total_actions_count     INTEGER         DEFAULT 0,
    -- Financial
    grants_received_eur     DECIMAL(14,2)   DEFAULT 0,
    cost_savings_realized_eur DECIMAL(14,2) DEFAULT 0,
    total_investment_eur    DECIMAL(14,2)   DEFAULT 0,
    roi_pct                 DECIMAL(8,3),
    -- Data quality
    data_quality_tier       VARCHAR(10),
    data_quality_score      DECIMAL(5,2),
    data_completeness_pct   DECIMAL(6,2),
    -- Review
    reviewed_by             VARCHAR(255),
    review_date             DATE,
    review_notes            TEXT,
    -- Metadata
    warnings                TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_ar_on_track CHECK (
        on_track_status IN ('ON_TRACK', 'CAUTION', 'OFF_TRACK', 'ACHIEVED', 'PENDING')
    ),
    CONSTRAINT chk_p026_ar_review_year CHECK (
        review_year >= 2020 AND review_year <= 2100
    ),
    CONSTRAINT chk_p026_ar_emissions_non_neg CHECK (
        actual_scope1 >= 0 AND actual_scope2 >= 0
        AND (actual_scope3 IS NULL OR actual_scope3 >= 0)
    ),
    CONSTRAINT chk_p026_ar_data_tier CHECK (
        data_quality_tier IS NULL OR data_quality_tier IN ('BRONZE', 'SILVER', 'GOLD')
    ),
    CONSTRAINT chk_p026_ar_quality_score CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p026_ar_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p026_ar_grants_non_neg CHECK (
        grants_received_eur IS NULL OR grants_received_eur >= 0
    ),
    CONSTRAINT chk_p026_ar_savings_non_neg CHECK (
        cost_savings_realized_eur IS NULL OR cost_savings_realized_eur >= 0
    ),
    CONSTRAINT uq_p026_ar_sme_year UNIQUE (sme_id, review_year)
);

-- ---------------------------------------------------------------------------
-- Indexes for annual_reviews
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_ar_sme              ON pack026_sme_net_zero.annual_reviews(sme_id);
CREATE INDEX idx_p026_ar_target           ON pack026_sme_net_zero.annual_reviews(target_id);
CREATE INDEX idx_p026_ar_tenant           ON pack026_sme_net_zero.annual_reviews(tenant_id);
CREATE INDEX idx_p026_ar_year             ON pack026_sme_net_zero.annual_reviews(review_year);
CREATE INDEX idx_p026_ar_sme_year         ON pack026_sme_net_zero.annual_reviews(sme_id, review_year);
CREATE INDEX idx_p026_ar_on_track         ON pack026_sme_net_zero.annual_reviews(on_track_status);
CREATE INDEX idx_p026_ar_data_tier        ON pack026_sme_net_zero.annual_reviews(data_quality_tier);
CREATE INDEX idx_p026_ar_created          ON pack026_sme_net_zero.annual_reviews(created_at DESC);
CREATE INDEX idx_p026_ar_metadata         ON pack026_sme_net_zero.annual_reviews USING GIN(metadata);

-- =============================================================================
-- Table 2: pack026_sme_net_zero.quarterly_snapshots
-- =============================================================================
-- Lightweight quarterly progress snapshots tracking energy, travel, and
-- procurement spend alongside quick win implementation progress.

CREATE TABLE pack026_sme_net_zero.quarterly_snapshots (
    snapshot_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    quarter                 INTEGER         NOT NULL,
    year                    INTEGER         NOT NULL,
    -- Spend tracking
    energy_spend_eur        DECIMAL(14,2),
    travel_spend_eur        DECIMAL(14,2),
    procurement_spend_eur   DECIMAL(14,2),
    total_operational_spend_eur DECIMAL(14,2),
    -- Emission estimates
    estimated_scope1_tco2e  DECIMAL(14,4),
    estimated_scope2_tco2e  DECIMAL(14,4),
    estimated_scope3_tco2e  DECIMAL(14,4),
    estimated_total_tco2e   DECIMAL(14,4)   GENERATED ALWAYS AS (
        COALESCE(estimated_scope1_tco2e, 0) + COALESCE(estimated_scope2_tco2e, 0) + COALESCE(estimated_scope3_tco2e, 0)
    ) STORED,
    -- Quick wins progress
    quick_wins_progress_pct DECIMAL(6,2),
    quick_wins_completed    INTEGER         DEFAULT 0,
    quick_wins_in_progress  INTEGER         DEFAULT 0,
    quick_wins_planned      INTEGER         DEFAULT 0,
    -- Comparison
    spend_vs_previous_quarter_pct DECIMAL(8,3),
    emissions_vs_previous_quarter_pct DECIMAL(8,3),
    -- Energy metrics
    electricity_kwh         DECIMAL(14,2),
    gas_kwh                 DECIMAL(14,2),
    water_m3                DECIMAL(14,2),
    -- Notes
    highlights              TEXT,
    concerns                TEXT,
    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_qs_quarter CHECK (
        quarter >= 1 AND quarter <= 4
    ),
    CONSTRAINT chk_p026_qs_year CHECK (
        year >= 2020 AND year <= 2100
    ),
    CONSTRAINT chk_p026_qs_spend_non_neg CHECK (
        (energy_spend_eur IS NULL OR energy_spend_eur >= 0)
        AND (travel_spend_eur IS NULL OR travel_spend_eur >= 0)
        AND (procurement_spend_eur IS NULL OR procurement_spend_eur >= 0)
    ),
    CONSTRAINT chk_p026_qs_progress CHECK (
        quick_wins_progress_pct IS NULL OR (quick_wins_progress_pct >= 0 AND quick_wins_progress_pct <= 100)
    ),
    CONSTRAINT uq_p026_qs_sme_quarter_year UNIQUE (sme_id, quarter, year)
);

-- ---------------------------------------------------------------------------
-- Indexes for quarterly_snapshots
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_qs_sme              ON pack026_sme_net_zero.quarterly_snapshots(sme_id);
CREATE INDEX idx_p026_qs_tenant           ON pack026_sme_net_zero.quarterly_snapshots(tenant_id);
CREATE INDEX idx_p026_qs_year             ON pack026_sme_net_zero.quarterly_snapshots(year);
CREATE INDEX idx_p026_qs_quarter_year     ON pack026_sme_net_zero.quarterly_snapshots(year, quarter);
CREATE INDEX idx_p026_qs_sme_year         ON pack026_sme_net_zero.quarterly_snapshots(sme_id, year, quarter);
CREATE INDEX idx_p026_qs_progress         ON pack026_sme_net_zero.quarterly_snapshots(quick_wins_progress_pct);
CREATE INDEX idx_p026_qs_created          ON pack026_sme_net_zero.quarterly_snapshots(created_at DESC);
CREATE INDEX idx_p026_qs_metadata         ON pack026_sme_net_zero.quarterly_snapshots USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p026_annual_reviews_updated
    BEFORE UPDATE ON pack026_sme_net_zero.annual_reviews
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p026_quarterly_snapshots_updated
    BEFORE UPDATE ON pack026_sme_net_zero.quarterly_snapshots
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack026_sme_net_zero.annual_reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack026_sme_net_zero.quarterly_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY p026_ar_tenant_isolation
    ON pack026_sme_net_zero.annual_reviews
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_ar_service_bypass
    ON pack026_sme_net_zero.annual_reviews
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p026_qs_tenant_isolation
    ON pack026_sme_net_zero.quarterly_snapshots
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_qs_service_bypass
    ON pack026_sme_net_zero.quarterly_snapshots
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.annual_reviews TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.quarterly_snapshots TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack026_sme_net_zero.annual_reviews IS
    'Annual emission review records comparing actual performance against targets with scope-level breakdowns, variance analysis, on-track status, and financial outcomes.';
COMMENT ON TABLE pack026_sme_net_zero.quarterly_snapshots IS
    'Lightweight quarterly progress snapshots tracking energy, travel, and procurement spend alongside quick win implementation progress.';

COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.review_id IS 'Unique annual review identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.review_year IS 'Year under review.';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.actual_total IS 'Generated total actual emissions across all scopes (tCO2e).';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.target_total IS 'Target emissions for this year from the net-zero plan (tCO2e).';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.variance_pct IS 'Percentage variance from target (negative = better than target).';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.on_track_status IS 'Progress status: ON_TRACK, CAUTION, OFF_TRACK, ACHIEVED, PENDING.';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.grants_received_eur IS 'Total grant funding received during the review year (EUR).';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.cost_savings_realized_eur IS 'Total cost savings from decarbonization actions during the review year (EUR).';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.actions_completed_count IS 'Number of quick-win actions completed during the review year.';
COMMENT ON COLUMN pack026_sme_net_zero.annual_reviews.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack026_sme_net_zero.quarterly_snapshots.snapshot_id IS 'Unique quarterly snapshot identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.quarterly_snapshots.quarter IS 'Calendar quarter (1-4).';
COMMENT ON COLUMN pack026_sme_net_zero.quarterly_snapshots.energy_spend_eur IS 'Total energy spend for the quarter (EUR).';
COMMENT ON COLUMN pack026_sme_net_zero.quarterly_snapshots.travel_spend_eur IS 'Total travel and transport spend for the quarter (EUR).';
COMMENT ON COLUMN pack026_sme_net_zero.quarterly_snapshots.procurement_spend_eur IS 'Total procurement spend for the quarter (EUR).';
COMMENT ON COLUMN pack026_sme_net_zero.quarterly_snapshots.quick_wins_progress_pct IS 'Overall progress on selected quick wins (0-100%).';

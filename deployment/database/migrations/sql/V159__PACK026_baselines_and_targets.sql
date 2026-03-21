-- =============================================================================
-- V159: PACK-026 SME Net Zero - Baselines & Targets
-- =============================================================================
-- Pack:         PACK-026 (SME Net Zero Pack)
-- Migration:    002 of 008
-- Date:         March 2026
--
-- SME emission baselines with scope-level breakdowns, intensity metrics, and
-- industry benchmarking. Net-zero targets with interim and long-term reduction
-- percentages, ACA pathway alignment, and compliance status tracking.
--
-- Tables (2):
--   1. pack026_sme_net_zero.sme_baselines
--   2. pack026_sme_net_zero.sme_targets
--
-- Previous: V158__PACK026_sme_net_zero_schema_and_profiles.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack026_sme_net_zero.sme_baselines
-- =============================================================================
-- SME emission baselines with scope-level breakdowns, per-employee intensity,
-- and industry average comparison for benchmarking.

CREATE TABLE pack026_sme_net_zero.sme_baselines (
    baseline_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                          UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    tenant_id                       UUID            NOT NULL,
    baseline_year                   INTEGER         NOT NULL,
    data_tier                       VARCHAR(10)     NOT NULL DEFAULT 'BRONZE',
    -- Scope emissions
    scope1_tco2e                    DECIMAL(18,4)   NOT NULL DEFAULT 0,
    scope2_tco2e                    DECIMAL(18,4)   NOT NULL DEFAULT 0,
    scope3_tco2e                    DECIMAL(18,4)   DEFAULT 0,
    total_tco2e                     DECIMAL(18,4)   GENERATED ALWAYS AS (
        scope1_tco2e + scope2_tco2e + COALESCE(scope3_tco2e, 0)
    ) STORED,
    -- Intensity metrics
    intensity_per_employee          DECIMAL(18,6),
    intensity_per_revenue           DECIMAL(18,8),
    intensity_per_sqm               DECIMAL(18,6),
    -- Benchmarking
    industry_avg_comparison_percentile DECIMAL(6,2),
    industry_avg_tco2e_per_employee DECIMAL(18,6),
    -- Scope 1 breakdown
    scope1_stationary_tco2e         DECIMAL(18,4),
    scope1_mobile_tco2e             DECIMAL(18,4),
    scope1_refrigerant_tco2e        DECIMAL(18,4),
    -- Scope 2 breakdown
    scope2_electricity_tco2e        DECIMAL(18,4),
    scope2_heating_tco2e            DECIMAL(18,4),
    scope2_method                   VARCHAR(20)     DEFAULT 'LOCATION',
    -- Scope 3 breakdown (simplified for SMEs)
    scope3_purchased_goods_tco2e    DECIMAL(18,4),
    scope3_travel_tco2e             DECIMAL(18,4),
    scope3_commuting_tco2e          DECIMAL(18,4),
    scope3_waste_tco2e              DECIMAL(18,4),
    scope3_upstream_transport_tco2e DECIMAL(18,4),
    -- Quality
    data_quality_score              DECIMAL(5,2),
    methodology_notes               TEXT,
    verification_status             VARCHAR(30)     DEFAULT 'unverified',
    verified_by                     VARCHAR(255),
    verified_date                   DATE,
    -- Metadata
    warnings                        TEXT[]          DEFAULT '{}',
    metadata                        JSONB           DEFAULT '{}',
    provenance_hash                 VARCHAR(64),
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_bl_data_tier CHECK (
        data_tier IN ('BRONZE', 'SILVER', 'GOLD')
    ),
    CONSTRAINT chk_p026_bl_baseline_year CHECK (
        baseline_year >= 2015 AND baseline_year <= 2100
    ),
    CONSTRAINT chk_p026_bl_emissions_non_neg CHECK (
        scope1_tco2e >= 0 AND scope2_tco2e >= 0
        AND (scope3_tco2e IS NULL OR scope3_tco2e >= 0)
    ),
    CONSTRAINT chk_p026_bl_intensity_non_neg CHECK (
        intensity_per_employee IS NULL OR intensity_per_employee >= 0
    ),
    CONSTRAINT chk_p026_bl_percentile CHECK (
        industry_avg_comparison_percentile IS NULL
        OR (industry_avg_comparison_percentile >= 0 AND industry_avg_comparison_percentile <= 100)
    ),
    CONSTRAINT chk_p026_bl_scope2_method CHECK (
        scope2_method IS NULL OR scope2_method IN ('LOCATION', 'MARKET')
    ),
    CONSTRAINT chk_p026_bl_verification CHECK (
        verification_status IN ('unverified', 'self_verified', 'third_party_verified', 'pending')
    ),
    CONSTRAINT chk_p026_bl_quality_score CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT uq_p026_bl_sme_year UNIQUE (sme_id, baseline_year)
);

-- ---------------------------------------------------------------------------
-- Indexes for sme_baselines
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_bl_sme              ON pack026_sme_net_zero.sme_baselines(sme_id);
CREATE INDEX idx_p026_bl_tenant           ON pack026_sme_net_zero.sme_baselines(tenant_id);
CREATE INDEX idx_p026_bl_year             ON pack026_sme_net_zero.sme_baselines(baseline_year);
CREATE INDEX idx_p026_bl_sme_year         ON pack026_sme_net_zero.sme_baselines(sme_id, baseline_year);
CREATE INDEX idx_p026_bl_data_tier        ON pack026_sme_net_zero.sme_baselines(data_tier);
CREATE INDEX idx_p026_bl_verification     ON pack026_sme_net_zero.sme_baselines(verification_status);
CREATE INDEX idx_p026_bl_percentile       ON pack026_sme_net_zero.sme_baselines(industry_avg_comparison_percentile);
CREATE INDEX idx_p026_bl_created          ON pack026_sme_net_zero.sme_baselines(created_at DESC);
CREATE INDEX idx_p026_bl_metadata         ON pack026_sme_net_zero.sme_baselines USING GIN(metadata);

-- =============================================================================
-- Table 2: pack026_sme_net_zero.sme_targets
-- =============================================================================
-- SME net-zero targets with interim (2030) and long-term (2050) reduction
-- percentages, pathway type (ACA default), and compliance tracking.

CREATE TABLE pack026_sme_net_zero.sme_targets (
    target_id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                          UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    tenant_id                       UUID            NOT NULL,
    baseline_id                     UUID            REFERENCES pack026_sme_net_zero.sme_baselines(baseline_id) ON DELETE SET NULL,
    -- Target years and reductions
    target_year_interim             INTEGER         NOT NULL DEFAULT 2030,
    target_year_longterm            INTEGER         NOT NULL DEFAULT 2050,
    interim_reduction_pct           DECIMAL(6,2)    NOT NULL,
    longterm_reduction_pct          DECIMAL(6,2)    NOT NULL DEFAULT 90.00,
    annual_reduction_rate_pct       DECIMAL(6,3),
    -- Pathway
    pathway_type                    VARCHAR(30)     NOT NULL DEFAULT 'ACA',
    pathway_details                 JSONB           DEFAULT '{}',
    -- Absolute targets
    interim_target_tco2e            DECIMAL(18,4),
    longterm_target_tco2e           DECIMAL(18,4),
    -- Compliance
    compliance_status               VARCHAR(30)     NOT NULL DEFAULT 'pending',
    sbti_aligned                    BOOLEAN         DEFAULT FALSE,
    climate_hub_committed           BOOLEAN         DEFAULT FALSE,
    -- Offset policy
    residual_offset_pct             DECIMAL(6,2)    DEFAULT 0,
    offset_approach                 VARCHAR(50),
    -- Governance
    board_approved                  BOOLEAN         DEFAULT FALSE,
    approval_date                   DATE,
    public_commitment               BOOLEAN         DEFAULT FALSE,
    public_commitment_url           TEXT,
    -- Metadata
    warnings                        TEXT[]          DEFAULT '{}',
    metadata                        JSONB           DEFAULT '{}',
    provenance_hash                 VARCHAR(64),
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_tgt_pathway CHECK (
        pathway_type IN ('ACA', 'SDA', 'ABSOLUTE_CONTRACTION', 'ECONOMIC_INTENSITY', 'CUSTOM')
    ),
    CONSTRAINT chk_p026_tgt_compliance CHECK (
        compliance_status IN ('pending', 'on_track', 'needs_attention', 'off_track', 'achieved', 'withdrawn')
    ),
    CONSTRAINT chk_p026_tgt_interim_year CHECK (
        target_year_interim >= 2025 AND target_year_interim <= 2040
    ),
    CONSTRAINT chk_p026_tgt_longterm_year CHECK (
        target_year_longterm >= 2040 AND target_year_longterm <= 2060
    ),
    CONSTRAINT chk_p026_tgt_year_order CHECK (
        target_year_interim < target_year_longterm
    ),
    CONSTRAINT chk_p026_tgt_interim_pct CHECK (
        interim_reduction_pct >= 0 AND interim_reduction_pct <= 100
    ),
    CONSTRAINT chk_p026_tgt_longterm_pct CHECK (
        longterm_reduction_pct >= 0 AND longterm_reduction_pct <= 100
    ),
    CONSTRAINT chk_p026_tgt_longterm_ge_interim CHECK (
        longterm_reduction_pct >= interim_reduction_pct
    ),
    CONSTRAINT chk_p026_tgt_residual CHECK (
        residual_offset_pct IS NULL OR (residual_offset_pct >= 0 AND residual_offset_pct <= 100)
    ),
    CONSTRAINT chk_p026_tgt_offset_approach CHECK (
        offset_approach IS NULL OR offset_approach IN (
            'NONE', 'CARBON_CREDITS', 'DIRECT_REMOVAL', 'NATURE_BASED', 'TECHNOLOGY_BASED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for sme_targets
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_tgt_sme             ON pack026_sme_net_zero.sme_targets(sme_id);
CREATE INDEX idx_p026_tgt_tenant          ON pack026_sme_net_zero.sme_targets(tenant_id);
CREATE INDEX idx_p026_tgt_baseline        ON pack026_sme_net_zero.sme_targets(baseline_id);
CREATE INDEX idx_p026_tgt_interim_year    ON pack026_sme_net_zero.sme_targets(target_year_interim);
CREATE INDEX idx_p026_tgt_longterm_year   ON pack026_sme_net_zero.sme_targets(target_year_longterm);
CREATE INDEX idx_p026_tgt_pathway         ON pack026_sme_net_zero.sme_targets(pathway_type);
CREATE INDEX idx_p026_tgt_compliance      ON pack026_sme_net_zero.sme_targets(compliance_status);
CREATE INDEX idx_p026_tgt_sbti            ON pack026_sme_net_zero.sme_targets(sbti_aligned);
CREATE INDEX idx_p026_tgt_climate_hub     ON pack026_sme_net_zero.sme_targets(climate_hub_committed);
CREATE INDEX idx_p026_tgt_created         ON pack026_sme_net_zero.sme_targets(created_at DESC);
CREATE INDEX idx_p026_tgt_pathway_detail  ON pack026_sme_net_zero.sme_targets USING GIN(pathway_details);
CREATE INDEX idx_p026_tgt_metadata        ON pack026_sme_net_zero.sme_targets USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p026_sme_baselines_updated
    BEFORE UPDATE ON pack026_sme_net_zero.sme_baselines
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p026_sme_targets_updated
    BEFORE UPDATE ON pack026_sme_net_zero.sme_targets
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack026_sme_net_zero.sme_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack026_sme_net_zero.sme_targets ENABLE ROW LEVEL SECURITY;

CREATE POLICY p026_bl_tenant_isolation
    ON pack026_sme_net_zero.sme_baselines
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_bl_service_bypass
    ON pack026_sme_net_zero.sme_baselines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p026_tgt_tenant_isolation
    ON pack026_sme_net_zero.sme_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_tgt_service_bypass
    ON pack026_sme_net_zero.sme_targets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.sme_baselines TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.sme_targets TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack026_sme_net_zero.sme_baselines IS
    'SME emission baselines with scope-level breakdowns, per-employee intensity metrics, and industry average benchmarking.';
COMMENT ON TABLE pack026_sme_net_zero.sme_targets IS
    'SME net-zero targets with interim and long-term reduction percentages, ACA pathway alignment, and compliance status.';

COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.baseline_id IS 'Unique baseline identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.baseline_year IS 'Emissions baseline reference year.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.data_tier IS 'Data quality tier: BRONZE (estimates), SILVER (spend-based), GOLD (activity-based).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.scope1_tco2e IS 'Scope 1 direct emissions in tCO2e.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.scope2_tco2e IS 'Scope 2 indirect emissions in tCO2e.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.scope3_tco2e IS 'Scope 3 value chain emissions in tCO2e (simplified categories for SMEs).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.total_tco2e IS 'Generated total emissions across all scopes.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.intensity_per_employee IS 'Emission intensity per full-time equivalent employee (tCO2e/FTE).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.industry_avg_comparison_percentile IS 'Percentile ranking vs industry average (0=worst, 100=best).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_baselines.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack026_sme_net_zero.sme_targets.target_id IS 'Unique target identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_targets.target_year_interim IS 'Interim target year (default 2030).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_targets.target_year_longterm IS 'Long-term net-zero target year (default 2050).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_targets.interim_reduction_pct IS 'Percent reduction from baseline by interim year.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_targets.longterm_reduction_pct IS 'Percent reduction from baseline by long-term year (typically 90%+).';
COMMENT ON COLUMN pack026_sme_net_zero.sme_targets.pathway_type IS 'Decarbonization pathway: ACA (Absolute Contraction Approach), SDA, etc.';
COMMENT ON COLUMN pack026_sme_net_zero.sme_targets.compliance_status IS 'Target compliance status: pending, on_track, needs_attention, off_track, achieved, withdrawn.';

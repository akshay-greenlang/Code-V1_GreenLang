-- =============================================================================
-- V150: PACK-025 Race to Zero - Interim Targets & Emission Pathways
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    003 of 010
-- Date:         March 2026
--
-- Interim target validation with scope-level emissions breakdowns, pathway
-- alignment scoring, temperature scoring, and IPCC compliance assessment.
-- Emission pathway projections vs actuals with variance tracking.
--
-- Tables (2):
--   1. pack025_race_to_zero.interim_targets
--   2. pack025_race_to_zero.emission_pathways
--
-- Previous: V149__pack025_race_to_zero_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.interim_targets
-- =============================================================================
-- Interim net-zero targets with scope-level emissions, reduction percentages,
-- pathway alignment, temperature scoring, and IPCC compliance.

CREATE TABLE pack025_race_to_zero.interim_targets (
    target_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    pledge_id               UUID            NOT NULL REFERENCES pack025_race_to_zero.pledges(pledge_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    target_year             INTEGER         NOT NULL DEFAULT 2030,
    target_emissions_s1     DECIMAL(18,4),
    target_emissions_s2     DECIMAL(18,4),
    target_emissions_s3     DECIMAL(18,4),
    total_target_tco2e      DECIMAL(18,4)   GENERATED ALWAYS AS (
        COALESCE(target_emissions_s1, 0) + COALESCE(target_emissions_s2, 0) + COALESCE(target_emissions_s3, 0)
    ) STORED,
    baseline_year           INTEGER         NOT NULL,
    baseline_emissions      DECIMAL(18,4)   NOT NULL,
    reduction_pct           DECIMAL(6,2)    NOT NULL,
    annual_reduction_rate   DECIMAL(6,3),
    pathway_alignment       VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    temperature_score       DECIMAL(4,2),
    ipcc_compliance         BOOLEAN         DEFAULT FALSE,
    methodology             VARCHAR(100)    NOT NULL DEFAULT 'ABSOLUTE_CONTRACTION',
    fair_share_assessment   JSONB           DEFAULT '{}',
    scope1_coverage_pct     DECIMAL(6,2)    DEFAULT 100,
    scope2_coverage_pct     DECIMAL(6,2)    DEFAULT 100,
    scope3_coverage_pct     DECIMAL(6,2),
    intensity_metric        VARCHAR(100),
    intensity_target_value  DECIMAL(18,6),
    validation_status       VARCHAR(30)     DEFAULT 'pending',
    validation_date         DATE,
    warnings                TEXT[]          DEFAULT '{}',
    errors                  TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_it_alignment CHECK (
        pathway_alignment IN ('ALIGNED', 'PARTIALLY_ALIGNED', 'MISALIGNED', 'AHEAD', 'PENDING')
    ),
    CONSTRAINT chk_p025_it_validation CHECK (
        validation_status IN ('pending', 'validated', 'rejected', 'conditional')
    ),
    CONSTRAINT chk_p025_it_target_year CHECK (
        target_year >= 2025 AND target_year <= 2060
    ),
    CONSTRAINT chk_p025_it_baseline_year CHECK (
        baseline_year >= 2000 AND baseline_year <= 2100
    ),
    CONSTRAINT chk_p025_it_reduction_pct CHECK (
        reduction_pct >= 0 AND reduction_pct <= 100
    ),
    CONSTRAINT chk_p025_it_temp_score CHECK (
        temperature_score IS NULL OR (temperature_score >= 1.0 AND temperature_score <= 5.0)
    ),
    CONSTRAINT chk_p025_it_emissions_non_neg CHECK (
        baseline_emissions >= 0
    ),
    CONSTRAINT chk_p025_it_methodology CHECK (
        methodology IN ('ABSOLUTE_CONTRACTION', 'SDA', 'ECONOMIC_INTENSITY', 'PHYSICAL_INTENSITY')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for interim_targets
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_it_org             ON pack025_race_to_zero.interim_targets(org_id);
CREATE INDEX idx_p025_it_pledge          ON pack025_race_to_zero.interim_targets(pledge_id);
CREATE INDEX idx_p025_it_tenant          ON pack025_race_to_zero.interim_targets(tenant_id);
CREATE INDEX idx_p025_it_target_year     ON pack025_race_to_zero.interim_targets(target_year);
CREATE INDEX idx_p025_it_org_year        ON pack025_race_to_zero.interim_targets(org_id, target_year);
CREATE INDEX idx_p025_it_alignment       ON pack025_race_to_zero.interim_targets(pathway_alignment);
CREATE INDEX idx_p025_it_temp            ON pack025_race_to_zero.interim_targets(temperature_score);
CREATE INDEX idx_p025_it_ipcc            ON pack025_race_to_zero.interim_targets(ipcc_compliance);
CREATE INDEX idx_p025_it_validation      ON pack025_race_to_zero.interim_targets(validation_status);
CREATE INDEX idx_p025_it_created         ON pack025_race_to_zero.interim_targets(created_at DESC);
CREATE INDEX idx_p025_it_metadata        ON pack025_race_to_zero.interim_targets USING GIN(metadata);
CREATE INDEX idx_p025_it_fair_share      ON pack025_race_to_zero.interim_targets USING GIN(fair_share_assessment);

-- =============================================================================
-- Table 2: pack025_race_to_zero.emission_pathways
-- =============================================================================
-- Year-by-year emission pathway projections vs actuals with variance tracking
-- across all three scopes.

CREATE TABLE pack025_race_to_zero.emission_pathways (
    pathway_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    target_id               UUID            REFERENCES pack025_race_to_zero.interim_targets(target_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    year                    INTEGER         NOT NULL,
    projected_emissions_s1  DECIMAL(18,4),
    projected_emissions_s2  DECIMAL(18,4),
    projected_emissions_s3  DECIMAL(18,4),
    total_projected_tco2e   DECIMAL(18,4)   GENERATED ALWAYS AS (
        COALESCE(projected_emissions_s1, 0) + COALESCE(projected_emissions_s2, 0) + COALESCE(projected_emissions_s3, 0)
    ) STORED,
    actual_emissions_s1     DECIMAL(18,4),
    actual_emissions_s2     DECIMAL(18,4),
    actual_emissions_s3     DECIMAL(18,4),
    total_actual_tco2e      DECIMAL(18,4)   GENERATED ALWAYS AS (
        COALESCE(actual_emissions_s1, 0) + COALESCE(actual_emissions_s2, 0) + COALESCE(actual_emissions_s3, 0)
    ) STORED,
    variance_pct            DECIMAL(8,3),
    pathway_source          VARCHAR(50)     DEFAULT 'IEA_NZE',
    data_quality_score      DECIMAL(5,2),
    is_projection           BOOLEAN         DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_ep_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p025_ep_source CHECK (
        pathway_source IN ('IEA_NZE', 'IPCC_AR6', 'TPI', 'SBTi_SDA', 'CUSTOM')
    ),
    CONSTRAINT chk_p025_ep_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT uq_p025_ep_org_year UNIQUE (org_id, year, pathway_source)
);

-- ---------------------------------------------------------------------------
-- Indexes for emission_pathways
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_ep_org             ON pack025_race_to_zero.emission_pathways(org_id);
CREATE INDEX idx_p025_ep_target          ON pack025_race_to_zero.emission_pathways(target_id);
CREATE INDEX idx_p025_ep_tenant          ON pack025_race_to_zero.emission_pathways(tenant_id);
CREATE INDEX idx_p025_ep_year            ON pack025_race_to_zero.emission_pathways(year);
CREATE INDEX idx_p025_ep_org_year        ON pack025_race_to_zero.emission_pathways(org_id, year);
CREATE INDEX idx_p025_ep_source          ON pack025_race_to_zero.emission_pathways(pathway_source);
CREATE INDEX idx_p025_ep_projection      ON pack025_race_to_zero.emission_pathways(is_projection);
CREATE INDEX idx_p025_ep_created         ON pack025_race_to_zero.emission_pathways(created_at DESC);
CREATE INDEX idx_p025_ep_metadata        ON pack025_race_to_zero.emission_pathways USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_interim_targets_updated
    BEFORE UPDATE ON pack025_race_to_zero.interim_targets
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

CREATE TRIGGER trg_p025_emission_pathways_updated
    BEFORE UPDATE ON pack025_race_to_zero.emission_pathways
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.interim_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.emission_pathways ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_it_tenant_isolation
    ON pack025_race_to_zero.interim_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_it_service_bypass
    ON pack025_race_to_zero.interim_targets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_ep_tenant_isolation
    ON pack025_race_to_zero.emission_pathways
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_ep_service_bypass
    ON pack025_race_to_zero.emission_pathways
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.interim_targets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.emission_pathways TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.interim_targets IS
    'Interim net-zero targets with scope-level emissions, pathway alignment, temperature scoring, and IPCC compliance.';
COMMENT ON TABLE pack025_race_to_zero.emission_pathways IS
    'Year-by-year emission pathway projections vs actuals with variance tracking across all scopes.';

COMMENT ON COLUMN pack025_race_to_zero.interim_targets.target_id IS 'Unique target identifier.';
COMMENT ON COLUMN pack025_race_to_zero.interim_targets.pathway_alignment IS 'Alignment with 1.5C pathway: ALIGNED, PARTIALLY_ALIGNED, MISALIGNED, AHEAD, PENDING.';
COMMENT ON COLUMN pack025_race_to_zero.interim_targets.temperature_score IS 'Implied temperature rise score (1.0-5.0 degrees C).';
COMMENT ON COLUMN pack025_race_to_zero.interim_targets.ipcc_compliance IS 'Whether target meets IPCC AR6 45% reduction by 2030 requirement.';
COMMENT ON COLUMN pack025_race_to_zero.emission_pathways.pathway_id IS 'Unique pathway point identifier.';
COMMENT ON COLUMN pack025_race_to_zero.emission_pathways.variance_pct IS 'Percentage variance between projected and actual emissions.';

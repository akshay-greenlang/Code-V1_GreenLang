-- =============================================================================
-- V274: PACK-035 Energy Benchmark Pack - Gap Analysis Tables
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    009 of 010
-- Date:         March 2026
--
-- Gap analysis between actual facility performance and benchmark targets.
-- Supports end-use disaggregation, improvement targeting, and integration
-- with PACK-033 (Quick Wins Identifier) for measure linking.
--
-- Tables (3):
--   1. pack035_energy_benchmark.gap_analyses
--   2. pack035_energy_benchmark.end_use_gaps
--   3. pack035_energy_benchmark.improvement_targets
--
-- Previous: V273__pack035_energy_benchmark_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.gap_analyses
-- =============================================================================
-- Top-level gap analysis records comparing a facility's actual EUI
-- against a selected benchmark target. Stores overall gap magnitude
-- and the disaggregation method used to break down by end use.

CREATE TABLE pack035_energy_benchmark.gap_analyses (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    analysis_date           DATE            NOT NULL,
    analysis_name           VARCHAR(255),
    -- Benchmark reference
    benchmark_source_id     UUID            REFERENCES pack035_energy_benchmark.benchmark_sources(id) ON DELETE SET NULL,
    benchmark_level         VARCHAR(30),
    peer_group_id           UUID            REFERENCES pack035_energy_benchmark.peer_groups(id) ON DELETE SET NULL,
    benchmark_target_eui    DECIMAL(10, 4)  NOT NULL,
    -- Facility performance
    actual_eui              DECIMAL(10, 4)  NOT NULL,
    eui_type                VARCHAR(20)     DEFAULT 'SITE',
    -- Gap results
    overall_gap_kwh_m2      DECIMAL(10, 4)  NOT NULL,
    overall_gap_pct         DECIMAL(8, 4)   NOT NULL,
    overall_gap_kwh_annual  DECIMAL(14, 4),
    overall_gap_cost_eur    DECIMAL(14, 4),
    overall_gap_co2_kg      DECIMAL(14, 4),
    -- Method
    disaggregation_method   VARCHAR(50)     NOT NULL DEFAULT 'PROPORTIONAL',
    floor_area_m2           DECIMAL(12, 2),
    energy_cost_eur_kwh     DECIMAL(8, 6),
    co2_factor_kg_kwh       DECIMAL(10, 6),
    -- Status
    status                  VARCHAR(20)     DEFAULT 'DRAFT',
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    -- Metadata
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_ga_actual CHECK (
        actual_eui >= 0
    ),
    CONSTRAINT chk_p035_ga_target CHECK (
        benchmark_target_eui >= 0
    ),
    CONSTRAINT chk_p035_ga_eui_type CHECK (
        eui_type IN ('SITE', 'SOURCE', 'PRIMARY', 'WEATHER_NORMALISED')
    ),
    CONSTRAINT chk_p035_ga_method CHECK (
        disaggregation_method IN (
            'PROPORTIONAL', 'SUB_METERED', 'END_USE_SPLIT', 'ASHRAE_TYPICAL',
            'CIBSE_TM22', 'SIMULATION', 'HYBRID', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p035_ga_level CHECK (
        benchmark_level IS NULL OR benchmark_level IN (
            'TYPICAL', 'GOOD_PRACTICE', 'BEST_PRACTICE', 'REGULATORY_MINIMUM', 'NZEB', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p035_ga_status CHECK (
        status IN ('DRAFT', 'REVIEWED', 'APPROVED', 'ARCHIVED')
    )
);

-- Indexes
CREATE INDEX idx_p035_ga_facility        ON pack035_energy_benchmark.gap_analyses(facility_id);
CREATE INDEX idx_p035_ga_tenant          ON pack035_energy_benchmark.gap_analyses(tenant_id);
CREATE INDEX idx_p035_ga_date            ON pack035_energy_benchmark.gap_analyses(analysis_date DESC);
CREATE INDEX idx_p035_ga_source          ON pack035_energy_benchmark.gap_analyses(benchmark_source_id);
CREATE INDEX idx_p035_ga_peer            ON pack035_energy_benchmark.gap_analyses(peer_group_id);
CREATE INDEX idx_p035_ga_status          ON pack035_energy_benchmark.gap_analyses(status);
CREATE INDEX idx_p035_ga_gap_pct         ON pack035_energy_benchmark.gap_analyses(overall_gap_pct DESC);
CREATE INDEX idx_p035_ga_fac_date        ON pack035_energy_benchmark.gap_analyses(facility_id, analysis_date DESC);

-- Trigger
CREATE TRIGGER trg_p035_ga_updated
    BEFORE UPDATE ON pack035_energy_benchmark.gap_analyses
    FOR EACH ROW EXECUTE FUNCTION pack035_energy_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack035_energy_benchmark.end_use_gaps
-- =============================================================================
-- Disaggregated gap analysis by end-use category (heating, cooling,
-- lighting, etc.). Each row shows the benchmark vs actual for a
-- specific end use, the gap magnitude, improvement potential, and
-- optional link to PACK-033 Quick Wins measures.

CREATE TABLE pack035_energy_benchmark.end_use_gaps (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    gap_analysis_id         UUID            NOT NULL REFERENCES pack035_energy_benchmark.gap_analyses(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    end_use_category        VARCHAR(50)     NOT NULL,
    end_use_label           VARCHAR(100),
    -- Benchmark vs actual
    benchmark_kwh_m2        DECIMAL(10, 4)  NOT NULL,
    actual_kwh_m2           DECIMAL(10, 4)  NOT NULL,
    gap_kwh_m2              DECIMAL(10, 4)  NOT NULL,
    gap_pct                 DECIMAL(8, 4),
    gap_kwh_annual          DECIMAL(14, 4),
    -- Financial impact
    annual_cost_gap_eur     DECIMAL(14, 4),
    annual_co2_gap_kg       DECIMAL(14, 4),
    -- Improvement potential
    improvement_potential_pct DECIMAL(6, 3),
    achievable_target_kwh_m2 DECIMAL(10, 4),
    priority_rank           INTEGER,
    priority_score          DECIMAL(6, 3),
    -- PACK-033 integration
    linked_pack033_measures JSONB           DEFAULT '[]',
    linked_measure_count    INTEGER         DEFAULT 0,
    linked_savings_kwh      DECIMAL(14, 4),
    -- Data quality
    data_source             VARCHAR(50),
    confidence_level        VARCHAR(20)     DEFAULT 'MEDIUM',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_eug_category CHECK (
        end_use_category IN (
            'HEATING', 'COOLING', 'LIGHTING', 'VENTILATION', 'DHW',
            'PLUG_LOADS', 'CATERING', 'IT_EQUIPMENT', 'VERTICAL_TRANSPORT',
            'PROCESS', 'REFRIGERATION', 'HUMIDIFICATION', 'SOLAR_GAINS',
            'BASELOAD', 'OTHER'
        )
    ),
    CONSTRAINT chk_p035_eug_benchmark CHECK (
        benchmark_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_eug_actual CHECK (
        actual_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_eug_rank CHECK (
        priority_rank IS NULL OR priority_rank >= 1
    ),
    CONSTRAINT chk_p035_eug_confidence CHECK (
        confidence_level IN ('HIGH', 'MEDIUM', 'LOW', 'ESTIMATED')
    ),
    CONSTRAINT chk_p035_eug_data_source CHECK (
        data_source IS NULL OR data_source IN (
            'SUB_METER', 'BMS', 'ESTIMATE', 'SIMULATION', 'PROPORTIONAL', 'AUDIT'
        )
    )
);

-- Indexes
CREATE INDEX idx_p035_eug_analysis       ON pack035_energy_benchmark.end_use_gaps(gap_analysis_id);
CREATE INDEX idx_p035_eug_tenant         ON pack035_energy_benchmark.end_use_gaps(tenant_id);
CREATE INDEX idx_p035_eug_category       ON pack035_energy_benchmark.end_use_gaps(end_use_category);
CREATE INDEX idx_p035_eug_priority       ON pack035_energy_benchmark.end_use_gaps(priority_rank);
CREATE INDEX idx_p035_eug_gap            ON pack035_energy_benchmark.end_use_gaps(gap_kwh_m2 DESC);
CREATE INDEX idx_p035_eug_measures       ON pack035_energy_benchmark.end_use_gaps USING GIN(linked_pack033_measures);

-- =============================================================================
-- Table 3: pack035_energy_benchmark.improvement_targets
-- =============================================================================
-- Improvement targets per facility and end-use category derived from
-- gap analysis. Tracks current performance, target EUI, timeline,
-- estimated savings, and implementation status.

CREATE TABLE pack035_energy_benchmark.improvement_targets (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    gap_analysis_id         UUID            REFERENCES pack035_energy_benchmark.gap_analyses(id) ON DELETE SET NULL,
    end_use_category        VARCHAR(50)     NOT NULL,
    target_name             VARCHAR(255),
    -- Current vs target
    current_kwh_m2          DECIMAL(10, 4)  NOT NULL,
    target_kwh_m2           DECIMAL(10, 4)  NOT NULL,
    interim_target_kwh_m2   DECIMAL(10, 4),
    target_year             INTEGER         NOT NULL,
    interim_target_year     INTEGER,
    reduction_pct           DECIMAL(6, 3)   NOT NULL,
    -- Savings estimates
    estimated_savings_kwh   DECIMAL(14, 4),
    estimated_savings_eur   DECIMAL(14, 4),
    estimated_co2_savings_kg DECIMAL(14, 4),
    estimated_capex_eur     DECIMAL(14, 4),
    payback_years           DECIMAL(6, 2),
    -- Status
    status                  VARCHAR(20)     NOT NULL DEFAULT 'PROPOSED',
    progress_pct            DECIMAL(5, 2)   DEFAULT 0,
    actual_kwh_m2           DECIMAL(10, 4),
    on_track                BOOLEAN,
    -- Metadata
    owner                   VARCHAR(255),
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_it_category CHECK (
        end_use_category IN (
            'HEATING', 'COOLING', 'LIGHTING', 'VENTILATION', 'DHW',
            'PLUG_LOADS', 'CATERING', 'IT_EQUIPMENT', 'VERTICAL_TRANSPORT',
            'PROCESS', 'REFRIGERATION', 'BASELOAD', 'TOTAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p035_it_current CHECK (
        current_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_it_target CHECK (
        target_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_it_reduction CHECK (
        reduction_pct >= 0 AND reduction_pct <= 100
    ),
    CONSTRAINT chk_p035_it_year CHECK (
        target_year >= 2020 AND target_year <= 2100
    ),
    CONSTRAINT chk_p035_it_status CHECK (
        status IN ('PROPOSED', 'APPROVED', 'IN_PROGRESS', 'ACHIEVED', 'MISSED', 'CANCELLED')
    ),
    CONSTRAINT chk_p035_it_progress CHECK (
        progress_pct IS NULL OR (progress_pct >= 0 AND progress_pct <= 100)
    ),
    CONSTRAINT chk_p035_it_payback CHECK (
        payback_years IS NULL OR payback_years >= 0
    )
);

-- Indexes
CREATE INDEX idx_p035_it_facility        ON pack035_energy_benchmark.improvement_targets(facility_id);
CREATE INDEX idx_p035_it_tenant          ON pack035_energy_benchmark.improvement_targets(tenant_id);
CREATE INDEX idx_p035_it_analysis        ON pack035_energy_benchmark.improvement_targets(gap_analysis_id);
CREATE INDEX idx_p035_it_category        ON pack035_energy_benchmark.improvement_targets(end_use_category);
CREATE INDEX idx_p035_it_year            ON pack035_energy_benchmark.improvement_targets(target_year);
CREATE INDEX idx_p035_it_status          ON pack035_energy_benchmark.improvement_targets(status);
CREATE INDEX idx_p035_it_on_track        ON pack035_energy_benchmark.improvement_targets(on_track);
CREATE INDEX idx_p035_it_reduction       ON pack035_energy_benchmark.improvement_targets(reduction_pct DESC);

-- Trigger
CREATE TRIGGER trg_p035_it_updated
    BEFORE UPDATE ON pack035_energy_benchmark.improvement_targets
    FOR EACH ROW EXECUTE FUNCTION pack035_energy_benchmark.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack035_energy_benchmark.gap_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.end_use_gaps ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.improvement_targets ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_ga_tenant_isolation ON pack035_energy_benchmark.gap_analyses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_ga_service_bypass ON pack035_energy_benchmark.gap_analyses
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_eug_tenant_isolation ON pack035_energy_benchmark.end_use_gaps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_eug_service_bypass ON pack035_energy_benchmark.end_use_gaps
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_it_tenant_isolation ON pack035_energy_benchmark.improvement_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_it_service_bypass ON pack035_energy_benchmark.improvement_targets
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.gap_analyses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.end_use_gaps TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.improvement_targets TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.gap_analyses IS
    'Top-level gap analysis comparing facility EUI against benchmark targets with disaggregation method.';
COMMENT ON TABLE pack035_energy_benchmark.end_use_gaps IS
    'End-use disaggregated gap analysis (heating, cooling, lighting, etc.) with PACK-033 measure linking.';
COMMENT ON TABLE pack035_energy_benchmark.improvement_targets IS
    'Improvement targets per facility and end-use category with timeline, savings estimates, and progress tracking.';

COMMENT ON COLUMN pack035_energy_benchmark.gap_analyses.disaggregation_method IS
    'Method for splitting total gap into end uses: PROPORTIONAL (ratio split), SUB_METERED (from meters), CIBSE_TM22 (CIBSE method), SIMULATION (dynamic model).';
COMMENT ON COLUMN pack035_energy_benchmark.end_use_gaps.linked_pack033_measures IS
    'JSON array of PACK-033 Quick Wins measure IDs linked to this end-use gap for implementation planning.';
COMMENT ON COLUMN pack035_energy_benchmark.improvement_targets.on_track IS
    'Whether the target is on track based on latest actual performance vs interpolated trajectory.';

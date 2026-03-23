-- =============================================================================
-- V288: PACK-037 Demand Response Pack - Baseline Calculation
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    003 of 010
-- Date:         March 2026
--
-- Baseline calculation engine tables for Customer Baseline Load (CBL)
-- determination. Supports multiple baseline methodologies (CAISO 10-of-10,
-- PJM symmetric additive adjustment, ISO-NE 5CP, NYISO average day,
-- regression-based). Stores interval data, baseline calculations,
-- adjustments, comparisons, and methodology optimization results.
--
-- Tables (6):
--   1. pack037_demand_response.dr_baseline_methodologies
--   2. pack037_demand_response.dr_interval_data
--   3. pack037_demand_response.dr_baseline_calculations
--   4. pack037_demand_response.dr_baseline_adjustments
--   5. pack037_demand_response.dr_baseline_comparisons
--   6. pack037_demand_response.dr_baseline_optimizations
--
-- Previous: V287__pack037_demand_response_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.dr_baseline_methodologies
-- =============================================================================
-- Reference table defining available baseline calculation methodologies
-- with their parameters, applicability rules, and regulatory approvals.

CREATE TABLE pack037_demand_response.dr_baseline_methodologies (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    methodology_code        VARCHAR(50)     NOT NULL UNIQUE,
    methodology_name        VARCHAR(255)    NOT NULL,
    description             TEXT            NOT NULL,
    category                VARCHAR(50)     NOT NULL,
    lookback_days           INTEGER         NOT NULL,
    excluded_day_types      VARCHAR(100),
    weather_adjustment      BOOLEAN         DEFAULT false,
    symmetric_adjustment    BOOLEAN         DEFAULT false,
    adjustment_cap_pct      NUMERIC(5,2),
    applicable_regions      TEXT[],
    applicable_load_types   TEXT[],
    min_data_days           INTEGER         NOT NULL DEFAULT 10,
    interval_resolution_min INTEGER         NOT NULL DEFAULT 15,
    calculation_formula     TEXT,
    regulatory_reference    TEXT,
    is_default              BOOLEAN         DEFAULT false,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_bm_category CHECK (
        category IN (
            'AVERAGING', 'REGRESSION', 'MATCHING_DAY', 'METER_BEFORE_AFTER',
            'MAXIMUM_BASE', 'ROLLING_AVERAGE', 'COMPARABLE_DAY', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p037_bm_lookback CHECK (
        lookback_days >= 1 AND lookback_days <= 365
    ),
    CONSTRAINT chk_p037_bm_adj_cap CHECK (
        adjustment_cap_pct IS NULL OR (adjustment_cap_pct >= 0 AND adjustment_cap_pct <= 100)
    ),
    CONSTRAINT chk_p037_bm_interval CHECK (
        interval_resolution_min IN (1, 5, 15, 30, 60)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_bm_category        ON pack037_demand_response.dr_baseline_methodologies(category);
CREATE INDEX idx_p037_bm_default         ON pack037_demand_response.dr_baseline_methodologies(is_default);
CREATE INDEX idx_p037_bm_regions         ON pack037_demand_response.dr_baseline_methodologies USING GIN(applicable_regions);

-- =============================================================================
-- Table 2: pack037_demand_response.dr_interval_data
-- =============================================================================
-- High-resolution interval load data used for baseline calculation and
-- event performance measurement. Supports 1/5/15/30/60-minute intervals.
-- Uses BRIN indexing for efficient time-series queries and optional
-- TimescaleDB hypertable conversion.

CREATE TABLE pack037_demand_response.dr_interval_data (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    meter_id                VARCHAR(100)    NOT NULL,
    interval_start          TIMESTAMPTZ     NOT NULL,
    interval_end            TIMESTAMPTZ     NOT NULL,
    demand_kw               NUMERIC(12,4),
    energy_kwh              NUMERIC(14,6),
    power_factor            NUMERIC(5,4),
    reactive_kvar           NUMERIC(12,4),
    interval_minutes        INTEGER         NOT NULL DEFAULT 15,
    data_quality            VARCHAR(20)     NOT NULL DEFAULT 'ACTUAL',
    source                  VARCHAR(30)     NOT NULL DEFAULT 'METER',
    is_event_interval       BOOLEAN         DEFAULT false,
    is_baseline_eligible    BOOLEAN         DEFAULT true,
    temperature_f           NUMERIC(6,2),
    humidity_pct            NUMERIC(5,2),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_id_demand CHECK (
        demand_kw IS NULL OR demand_kw >= 0
    ),
    CONSTRAINT chk_p037_id_energy CHECK (
        energy_kwh IS NULL OR energy_kwh >= 0
    ),
    CONSTRAINT chk_p037_id_pf CHECK (
        power_factor IS NULL OR (power_factor >= 0 AND power_factor <= 1)
    ),
    CONSTRAINT chk_p037_id_interval CHECK (
        interval_minutes IN (1, 5, 15, 30, 60)
    ),
    CONSTRAINT chk_p037_id_quality CHECK (
        data_quality IN ('ACTUAL', 'ESTIMATED', 'INTERPOLATED', 'MISSING', 'VALIDATED', 'EXCLUDED')
    ),
    CONSTRAINT chk_p037_id_source CHECK (
        source IN ('METER', 'SCADA', 'EMS', 'TELEMETRY', 'MANUAL', 'ESTIMATED')
    ),
    CONSTRAINT chk_p037_id_period CHECK (
        interval_end > interval_start
    ),
    CONSTRAINT uq_p037_id_meter_start UNIQUE (meter_id, interval_start)
);

-- ---------------------------------------------------------------------------
-- Indexes (BRIN for time-series, B-tree for lookups)
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_id_facility        ON pack037_demand_response.dr_interval_data(facility_profile_id);
CREATE INDEX idx_p037_id_tenant          ON pack037_demand_response.dr_interval_data(tenant_id);
CREATE INDEX idx_p037_id_meter           ON pack037_demand_response.dr_interval_data(meter_id);
CREATE INDEX idx_p037_id_start           ON pack037_demand_response.dr_interval_data USING BRIN(interval_start);
CREATE INDEX idx_p037_id_quality         ON pack037_demand_response.dr_interval_data(data_quality);
CREATE INDEX idx_p037_id_event           ON pack037_demand_response.dr_interval_data(is_event_interval);
CREATE INDEX idx_p037_id_baseline_elig   ON pack037_demand_response.dr_interval_data(is_baseline_eligible);

-- Composite: facility + meter + time for baseline window lookups
CREATE INDEX idx_p037_id_fac_meter_ts    ON pack037_demand_response.dr_interval_data(facility_profile_id, meter_id, interval_start DESC);

-- ---------------------------------------------------------------------------
-- TimescaleDB hypertable conversion (if extension is available)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'pack037_demand_response.dr_interval_data',
            'interval_start',
            chunk_time_interval => INTERVAL '1 month',
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
        RAISE NOTICE 'TimescaleDB hypertable created for dr_interval_data';
    ELSE
        RAISE NOTICE 'TimescaleDB not available - dr_interval_data remains a standard table with BRIN index';
    END IF;
END;
$$;

-- =============================================================================
-- Table 3: pack037_demand_response.dr_baseline_calculations
-- =============================================================================
-- Calculated Customer Baseline Load (CBL) results for each event or
-- settlement period. Links to the methodology used and stores the
-- per-interval baseline values as a JSONB array.

CREATE TABLE pack037_demand_response.dr_baseline_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    methodology_id          UUID            NOT NULL REFERENCES pack037_demand_response.dr_baseline_methodologies(id),
    event_date              DATE            NOT NULL,
    baseline_type           VARCHAR(30)     NOT NULL DEFAULT 'EVENT',
    lookback_start          DATE            NOT NULL,
    lookback_end            DATE            NOT NULL,
    days_used               INTEGER         NOT NULL,
    days_excluded           INTEGER         DEFAULT 0,
    baseline_peak_kw        NUMERIC(12,4)   NOT NULL,
    baseline_avg_kw         NUMERIC(12,4),
    baseline_total_kwh      NUMERIC(16,4),
    adjustment_applied      BOOLEAN         DEFAULT false,
    adjustment_factor       NUMERIC(8,5)    DEFAULT 1.0,
    adjusted_peak_kw        NUMERIC(12,4),
    weather_adjusted        BOOLEAN         DEFAULT false,
    interval_baselines      JSONB           NOT NULL DEFAULT '[]',
    data_completeness_pct   NUMERIC(5,2),
    calculation_status      VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    calculated_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_bc_type CHECK (
        baseline_type IN ('EVENT', 'TEST', 'SETTLEMENT', 'VERIFICATION', 'FORECAST')
    ),
    CONSTRAINT chk_p037_bc_lookback CHECK (
        lookback_end >= lookback_start
    ),
    CONSTRAINT chk_p037_bc_days CHECK (
        days_used >= 1
    ),
    CONSTRAINT chk_p037_bc_peak CHECK (
        baseline_peak_kw >= 0
    ),
    CONSTRAINT chk_p037_bc_status CHECK (
        calculation_status IN ('CALCULATED', 'ADJUSTED', 'VERIFIED', 'DISPUTED', 'FINAL')
    ),
    CONSTRAINT chk_p037_bc_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_bc_facility        ON pack037_demand_response.dr_baseline_calculations(facility_profile_id);
CREATE INDEX idx_p037_bc_tenant          ON pack037_demand_response.dr_baseline_calculations(tenant_id);
CREATE INDEX idx_p037_bc_methodology     ON pack037_demand_response.dr_baseline_calculations(methodology_id);
CREATE INDEX idx_p037_bc_event_date      ON pack037_demand_response.dr_baseline_calculations(event_date DESC);
CREATE INDEX idx_p037_bc_type            ON pack037_demand_response.dr_baseline_calculations(baseline_type);
CREATE INDEX idx_p037_bc_status          ON pack037_demand_response.dr_baseline_calculations(calculation_status);
CREATE INDEX idx_p037_bc_calculated      ON pack037_demand_response.dr_baseline_calculations(calculated_at DESC);
CREATE INDEX idx_p037_bc_created         ON pack037_demand_response.dr_baseline_calculations(created_at DESC);

-- Composite: facility + event date for event-specific baseline lookup
CREATE INDEX idx_p037_bc_fac_date        ON pack037_demand_response.dr_baseline_calculations(facility_profile_id, event_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_bc_updated
    BEFORE UPDATE ON pack037_demand_response.dr_baseline_calculations
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack037_demand_response.dr_baseline_adjustments
-- =============================================================================
-- Adjustment records applied to baselines including symmetric additive
-- adjustments, weather normalization, and day-of adjustments required
-- by various ISO/RTO methodologies.

CREATE TABLE pack037_demand_response.dr_baseline_adjustments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_calculation_id UUID            NOT NULL REFERENCES pack037_demand_response.dr_baseline_calculations(id) ON DELETE CASCADE,
    adjustment_type         VARCHAR(50)     NOT NULL,
    adjustment_reason       TEXT            NOT NULL,
    pre_adjustment_kw       NUMERIC(12,4)   NOT NULL,
    adjustment_value_kw     NUMERIC(12,4)   NOT NULL,
    post_adjustment_kw      NUMERIC(12,4)   NOT NULL,
    adjustment_pct          NUMERIC(8,4),
    adjustment_window_start TIMESTAMPTZ,
    adjustment_window_end   TIMESTAMPTZ,
    weather_variable        VARCHAR(30),
    weather_value           NUMERIC(8,2),
    approved_by             VARCHAR(255),
    approved_at             TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_ba_type CHECK (
        adjustment_type IN (
            'SYMMETRIC_ADDITIVE', 'WEATHER_NORMALIZATION', 'DAY_OF_ADJUSTMENT',
            'REGRESSION_CORRECTION', 'MANUAL_OVERRIDE', 'OCCUPANCY_ADJUSTMENT',
            'PRODUCTION_ADJUSTMENT', 'CAP_ADJUSTMENT'
        )
    ),
    CONSTRAINT chk_p037_ba_pre CHECK (
        pre_adjustment_kw >= 0
    ),
    CONSTRAINT chk_p037_ba_post CHECK (
        post_adjustment_kw >= 0
    ),
    CONSTRAINT chk_p037_ba_weather CHECK (
        weather_variable IS NULL OR weather_variable IN (
            'TEMPERATURE_F', 'TEMPERATURE_C', 'HUMIDITY_PCT', 'DEW_POINT_F',
            'COOLING_DEGREE_HOURS', 'HEATING_DEGREE_HOURS', 'THI'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_ba_baseline        ON pack037_demand_response.dr_baseline_adjustments(baseline_calculation_id);
CREATE INDEX idx_p037_ba_type            ON pack037_demand_response.dr_baseline_adjustments(adjustment_type);
CREATE INDEX idx_p037_ba_created         ON pack037_demand_response.dr_baseline_adjustments(created_at DESC);

-- =============================================================================
-- Table 5: pack037_demand_response.dr_baseline_comparisons
-- =============================================================================
-- Side-by-side comparison of baseline values calculated using different
-- methodologies for the same event, enabling methodology selection
-- and accuracy analysis.

CREATE TABLE pack037_demand_response.dr_baseline_comparisons (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    comparison_date         DATE            NOT NULL,
    baseline_a_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_baseline_calculations(id),
    baseline_b_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_baseline_calculations(id),
    baseline_a_peak_kw      NUMERIC(12,4)   NOT NULL,
    baseline_b_peak_kw      NUMERIC(12,4)   NOT NULL,
    difference_kw           NUMERIC(12,4)   NOT NULL,
    difference_pct          NUMERIC(8,4)    NOT NULL,
    recommended_method      UUID,
    recommendation_reason   TEXT,
    actual_demand_kw        NUMERIC(12,4),
    accuracy_a_pct          NUMERIC(8,4),
    accuracy_b_pct          NUMERIC(8,4),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_bcomp_diff CHECK (
        baseline_a_id != baseline_b_id
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_bcomp_facility     ON pack037_demand_response.dr_baseline_comparisons(facility_profile_id);
CREATE INDEX idx_p037_bcomp_tenant       ON pack037_demand_response.dr_baseline_comparisons(tenant_id);
CREATE INDEX idx_p037_bcomp_date         ON pack037_demand_response.dr_baseline_comparisons(comparison_date DESC);
CREATE INDEX idx_p037_bcomp_a            ON pack037_demand_response.dr_baseline_comparisons(baseline_a_id);
CREATE INDEX idx_p037_bcomp_b            ON pack037_demand_response.dr_baseline_comparisons(baseline_b_id);

-- =============================================================================
-- Table 6: pack037_demand_response.dr_baseline_optimizations
-- =============================================================================
-- Records of baseline methodology optimization runs that evaluate
-- which methodology minimises error for a given facility and season.

CREATE TABLE pack037_demand_response.dr_baseline_optimizations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    optimization_date       DATE            NOT NULL,
    season                  VARCHAR(20)     NOT NULL,
    methodologies_tested    INTEGER         NOT NULL,
    best_methodology_id     UUID            REFERENCES pack037_demand_response.dr_baseline_methodologies(id),
    best_mae_kw             NUMERIC(12,4),
    best_mape_pct           NUMERIC(8,4),
    best_rmse_kw            NUMERIC(12,4),
    best_bias_kw            NUMERIC(12,4),
    runner_up_methodology_id UUID           REFERENCES pack037_demand_response.dr_baseline_methodologies(id),
    results_detail          JSONB           NOT NULL DEFAULT '{}',
    recommendation          TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_bo_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p037_bo_tested CHECK (
        methodologies_tested >= 1
    ),
    CONSTRAINT chk_p037_bo_mape CHECK (
        best_mape_pct IS NULL OR (best_mape_pct >= 0 AND best_mape_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_bo_facility        ON pack037_demand_response.dr_baseline_optimizations(facility_profile_id);
CREATE INDEX idx_p037_bo_tenant          ON pack037_demand_response.dr_baseline_optimizations(tenant_id);
CREATE INDEX idx_p037_bo_date            ON pack037_demand_response.dr_baseline_optimizations(optimization_date DESC);
CREATE INDEX idx_p037_bo_season          ON pack037_demand_response.dr_baseline_optimizations(season);
CREATE INDEX idx_p037_bo_best_method     ON pack037_demand_response.dr_baseline_optimizations(best_methodology_id);
CREATE INDEX idx_p037_bo_mape            ON pack037_demand_response.dr_baseline_optimizations(best_mape_pct);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_baseline_methodologies ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_interval_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_baseline_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_baseline_adjustments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_baseline_comparisons ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_baseline_optimizations ENABLE ROW LEVEL SECURITY;

-- Baseline methodologies are shared reference data
CREATE POLICY p037_bm_read_all ON pack037_demand_response.dr_baseline_methodologies
    FOR SELECT USING (TRUE);
CREATE POLICY p037_bm_service_bypass ON pack037_demand_response.dr_baseline_methodologies
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_id_tenant_isolation ON pack037_demand_response.dr_interval_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_id_service_bypass ON pack037_demand_response.dr_interval_data
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_bc_tenant_isolation ON pack037_demand_response.dr_baseline_calculations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_bc_service_bypass ON pack037_demand_response.dr_baseline_calculations
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_ba_service_bypass ON pack037_demand_response.dr_baseline_adjustments
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_bcomp_tenant_isolation ON pack037_demand_response.dr_baseline_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_bcomp_service_bypass ON pack037_demand_response.dr_baseline_comparisons
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_bo_tenant_isolation ON pack037_demand_response.dr_baseline_optimizations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_bo_service_bypass ON pack037_demand_response.dr_baseline_optimizations
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_baseline_methodologies TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_interval_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_baseline_calculations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_baseline_adjustments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_baseline_comparisons TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_baseline_optimizations TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.dr_baseline_methodologies IS
    'Reference table of available baseline calculation methodologies with parameters, applicability rules, and regulatory approvals.';
COMMENT ON TABLE pack037_demand_response.dr_interval_data IS
    'High-resolution interval load data at 1/5/15/30/60-minute granularity for baseline calculation and event performance measurement.';
COMMENT ON TABLE pack037_demand_response.dr_baseline_calculations IS
    'Calculated Customer Baseline Load (CBL) results linking methodology, lookback window, and per-interval baseline values.';
COMMENT ON TABLE pack037_demand_response.dr_baseline_adjustments IS
    'Adjustment records applied to baselines including symmetric additive, weather normalization, and day-of adjustments.';
COMMENT ON TABLE pack037_demand_response.dr_baseline_comparisons IS
    'Side-by-side comparison of baseline values from different methodologies for accuracy analysis.';
COMMENT ON TABLE pack037_demand_response.dr_baseline_optimizations IS
    'Methodology optimization runs evaluating which baseline approach minimises error for a facility and season.';

COMMENT ON COLUMN pack037_demand_response.dr_baseline_methodologies.methodology_code IS 'Unique methodology code (e.g., CAISO_10_OF_10, PJM_SYM_ADD, ISONE_5CP).';
COMMENT ON COLUMN pack037_demand_response.dr_baseline_methodologies.lookback_days IS 'Number of lookback days used to calculate the baseline.';
COMMENT ON COLUMN pack037_demand_response.dr_baseline_methodologies.symmetric_adjustment IS 'Whether the methodology uses symmetric additive adjustment (PJM-style).';
COMMENT ON COLUMN pack037_demand_response.dr_baseline_methodologies.adjustment_cap_pct IS 'Maximum percentage adjustment allowed (e.g., 20% cap on symmetric adjustment).';

COMMENT ON COLUMN pack037_demand_response.dr_interval_data.is_event_interval IS 'Whether this interval falls within a DR event window (excluded from future baselines).';
COMMENT ON COLUMN pack037_demand_response.dr_interval_data.is_baseline_eligible IS 'Whether this interval can be used in baseline calculations (non-event, valid quality).';
COMMENT ON COLUMN pack037_demand_response.dr_interval_data.temperature_f IS 'Outdoor air temperature in Fahrenheit at the time of this interval, for weather-adjusted baselines.';

COMMENT ON COLUMN pack037_demand_response.dr_baseline_calculations.interval_baselines IS 'JSONB array of per-interval baseline kW values for the event window.';
COMMENT ON COLUMN pack037_demand_response.dr_baseline_calculations.adjustment_factor IS 'Multiplicative or additive adjustment factor applied to the raw baseline.';
COMMENT ON COLUMN pack037_demand_response.dr_baseline_calculations.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack037_demand_response.dr_baseline_adjustments.adjustment_type IS 'Type of adjustment: SYMMETRIC_ADDITIVE, WEATHER_NORMALIZATION, DAY_OF_ADJUSTMENT, etc.';

COMMENT ON COLUMN pack037_demand_response.dr_baseline_optimizations.best_mae_kw IS 'Mean Absolute Error in kW for the best-performing methodology.';
COMMENT ON COLUMN pack037_demand_response.dr_baseline_optimizations.best_mape_pct IS 'Mean Absolute Percentage Error for the best-performing methodology.';
COMMENT ON COLUMN pack037_demand_response.dr_baseline_optimizations.best_rmse_kw IS 'Root Mean Square Error in kW for the best-performing methodology.';

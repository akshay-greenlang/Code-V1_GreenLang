-- =============================================================================
-- V279: PACK-036 Utility Analysis Pack - Demand Profiles & Interval Data
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    004 of 010
-- Date:         March 2026
--
-- Tables for interval (smart meter) data, demand profile analysis, and
-- peak event tracking. Uses TimescaleDB hypertable for gl_interval_data
-- when available, and BRIN indexes for efficient time-series queries.
--
-- Tables (3):
--   1. pack036_utility_analysis.gl_interval_data
--   2. pack036_utility_analysis.gl_demand_profiles
--   3. pack036_utility_analysis.gl_peak_events
--
-- Previous: V278__pack036_rate_structures.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_interval_data
-- =============================================================================
-- High-resolution interval (smart meter) data at 5/15/30/60-minute
-- granularity. This is the primary time-series table for demand analysis,
-- load profiling, and TOU cost allocation. Uses BRIN indexing and optional
-- TimescaleDB hypertable conversion for scalability.

CREATE TABLE pack036_utility_analysis.gl_interval_data (
    interval_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    meter_id                VARCHAR(100)    NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,
    demand_kw               NUMERIC(12,4),
    energy_kwh              NUMERIC(14,6),
    power_factor            NUMERIC(5,4),
    reactive_kvar           NUMERIC(12,4),
    voltage_v               NUMERIC(8,2),
    current_a               NUMERIC(10,4),
    interval_minutes        INTEGER         NOT NULL DEFAULT 15,
    data_quality            VARCHAR(20)     DEFAULT 'ACTUAL',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_id_demand CHECK (
        demand_kw IS NULL OR demand_kw >= 0
    ),
    CONSTRAINT chk_p036_id_energy CHECK (
        energy_kwh IS NULL OR energy_kwh >= 0
    ),
    CONSTRAINT chk_p036_id_pf CHECK (
        power_factor IS NULL OR (power_factor >= 0 AND power_factor <= 1)
    ),
    CONSTRAINT chk_p036_id_interval CHECK (
        interval_minutes IN (1, 5, 15, 30, 60)
    ),
    CONSTRAINT chk_p036_id_quality CHECK (
        data_quality IN ('ACTUAL', 'ESTIMATED', 'INTERPOLATED', 'MISSING', 'VALIDATED')
    ),
    -- Prevent duplicate readings for the same meter at the same timestamp
    CONSTRAINT uq_p036_id_meter_ts UNIQUE (meter_id, timestamp)
);

-- ---------------------------------------------------------------------------
-- Indexes (BRIN for time-series, B-tree for lookups)
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_id_tenant         ON pack036_utility_analysis.gl_interval_data(tenant_id);
CREATE INDEX idx_p036_id_facility       ON pack036_utility_analysis.gl_interval_data(facility_id);
CREATE INDEX idx_p036_id_meter          ON pack036_utility_analysis.gl_interval_data(meter_id);
CREATE INDEX idx_p036_id_timestamp      ON pack036_utility_analysis.gl_interval_data USING BRIN(timestamp);
CREATE INDEX idx_p036_id_quality        ON pack036_utility_analysis.gl_interval_data(data_quality);

-- Composite: facility + meter + timestamp for demand profile queries
CREATE INDEX idx_p036_id_fac_meter_ts   ON pack036_utility_analysis.gl_interval_data(facility_id, meter_id, timestamp DESC);

-- ---------------------------------------------------------------------------
-- TimescaleDB hypertable conversion (if extension is available)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'pack036_utility_analysis.gl_interval_data',
            'timestamp',
            chunk_time_interval => INTERVAL '1 month',
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
        RAISE NOTICE 'TimescaleDB hypertable created for gl_interval_data';
    ELSE
        RAISE NOTICE 'TimescaleDB not available - gl_interval_data remains a standard table with BRIN index';
    END IF;
END;
$$;

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_demand_profiles
-- =============================================================================
-- Aggregated demand profile summaries per facility for a given period.
-- Captures peak/average/minimum demand, load factor, and peak timing.

CREATE TABLE pack036_utility_analysis.gl_demand_profiles (
    profile_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    meter_id                VARCHAR(100),
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    peak_demand_kw          NUMERIC(12,4)   NOT NULL,
    avg_demand_kw           NUMERIC(12,4)   NOT NULL,
    min_demand_kw           NUMERIC(12,4),
    load_factor_pct         NUMERIC(6,2),
    peak_timestamp          TIMESTAMPTZ,
    peak_period             VARCHAR(30),
    base_load_kw            NUMERIC(12,4),
    total_energy_kwh        NUMERIC(16,4),
    on_peak_energy_kwh      NUMERIC(16,4),
    off_peak_energy_kwh     NUMERIC(16,4),
    on_peak_pct             NUMERIC(6,2),
    data_completeness_pct   NUMERIC(6,2),
    intervals_count         INTEGER,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_dp_period CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p036_dp_peak CHECK (
        peak_demand_kw >= 0
    ),
    CONSTRAINT chk_p036_dp_avg CHECK (
        avg_demand_kw >= 0
    ),
    CONSTRAINT chk_p036_dp_min CHECK (
        min_demand_kw IS NULL OR min_demand_kw >= 0
    ),
    CONSTRAINT chk_p036_dp_lf CHECK (
        load_factor_pct IS NULL OR (load_factor_pct >= 0 AND load_factor_pct <= 100)
    ),
    CONSTRAINT chk_p036_dp_base_load CHECK (
        base_load_kw IS NULL OR base_load_kw >= 0
    ),
    CONSTRAINT chk_p036_dp_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p036_dp_on_peak_pct CHECK (
        on_peak_pct IS NULL OR (on_peak_pct >= 0 AND on_peak_pct <= 100)
    ),
    CONSTRAINT chk_p036_dp_peak_period CHECK (
        peak_period IS NULL OR peak_period IN (
            'ON_PEAK', 'MID_PEAK', 'OFF_PEAK', 'SUPER_OFF_PEAK'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_dp_tenant         ON pack036_utility_analysis.gl_demand_profiles(tenant_id);
CREATE INDEX idx_p036_dp_facility       ON pack036_utility_analysis.gl_demand_profiles(facility_id);
CREATE INDEX idx_p036_dp_meter          ON pack036_utility_analysis.gl_demand_profiles(meter_id);
CREATE INDEX idx_p036_dp_period_start   ON pack036_utility_analysis.gl_demand_profiles(period_start DESC);
CREATE INDEX idx_p036_dp_peak           ON pack036_utility_analysis.gl_demand_profiles(peak_demand_kw DESC);
CREATE INDEX idx_p036_dp_load_factor    ON pack036_utility_analysis.gl_demand_profiles(load_factor_pct);
CREATE INDEX idx_p036_dp_created        ON pack036_utility_analysis.gl_demand_profiles(created_at DESC);

-- Composite: facility + period for time-series profile lookup
CREATE INDEX idx_p036_dp_fac_period     ON pack036_utility_analysis.gl_demand_profiles(facility_id, period_start DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_dp_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_demand_profiles
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_peak_events
-- =============================================================================
-- Individual peak demand events detected in interval data. Tracks the
-- magnitude, duration, TOU period, cost impact, and avoidability of
-- each peak event for demand response and peak shaving analysis.

CREATE TABLE pack036_utility_analysis.gl_peak_events (
    event_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_demand_profiles(profile_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,
    demand_kw               NUMERIC(12,4)   NOT NULL,
    duration_minutes        INTEGER         NOT NULL DEFAULT 15,
    peak_type               VARCHAR(30)     NOT NULL,
    tou_period              VARCHAR(30),
    cost_impact_eur         NUMERIC(14,2),
    monthly_demand_rank     INTEGER,
    avoidable               BOOLEAN         DEFAULT false,
    avoidance_strategy      TEXT,
    weather_driven          BOOLEAN,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_pe_demand CHECK (
        demand_kw >= 0
    ),
    CONSTRAINT chk_p036_pe_duration CHECK (
        duration_minutes > 0
    ),
    CONSTRAINT chk_p036_pe_peak_type CHECK (
        peak_type IN (
            'MONTHLY_PEAK', 'ANNUAL_PEAK', 'SEASONAL_PEAK', 'RATCHET_PEAK',
            'COINCIDENT_PEAK', 'SYSTEM_PEAK', 'NEAR_PEAK'
        )
    ),
    CONSTRAINT chk_p036_pe_tou CHECK (
        tou_period IS NULL OR tou_period IN (
            'ON_PEAK', 'MID_PEAK', 'OFF_PEAK', 'SUPER_OFF_PEAK',
            'CRITICAL_PEAK', 'SHOULDER'
        )
    ),
    CONSTRAINT chk_p036_pe_rank CHECK (
        monthly_demand_rank IS NULL OR monthly_demand_rank >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_pe_profile        ON pack036_utility_analysis.gl_peak_events(profile_id);
CREATE INDEX idx_p036_pe_tenant         ON pack036_utility_analysis.gl_peak_events(tenant_id);
CREATE INDEX idx_p036_pe_timestamp      ON pack036_utility_analysis.gl_peak_events(timestamp DESC);
CREATE INDEX idx_p036_pe_demand         ON pack036_utility_analysis.gl_peak_events(demand_kw DESC);
CREATE INDEX idx_p036_pe_peak_type      ON pack036_utility_analysis.gl_peak_events(peak_type);
CREATE INDEX idx_p036_pe_avoidable      ON pack036_utility_analysis.gl_peak_events(avoidable);
CREATE INDEX idx_p036_pe_cost           ON pack036_utility_analysis.gl_peak_events(cost_impact_eur DESC);

-- Composite: avoidable peaks sorted by cost impact for optimization
CREATE INDEX idx_p036_pe_avoid_cost     ON pack036_utility_analysis.gl_peak_events(cost_impact_eur DESC)
    WHERE avoidable = true;

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_interval_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_demand_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_peak_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_id_tenant_isolation
    ON pack036_utility_analysis.gl_interval_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_id_service_bypass
    ON pack036_utility_analysis.gl_interval_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_dp_tenant_isolation
    ON pack036_utility_analysis.gl_demand_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_dp_service_bypass
    ON pack036_utility_analysis.gl_demand_profiles
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_pe_tenant_isolation
    ON pack036_utility_analysis.gl_peak_events
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_pe_service_bypass
    ON pack036_utility_analysis.gl_peak_events
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_interval_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_demand_profiles TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_peak_events TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_interval_data IS
    'High-resolution interval (smart meter) data at 5/15/30/60-minute granularity for demand analysis and load profiling.';

COMMENT ON TABLE pack036_utility_analysis.gl_demand_profiles IS
    'Aggregated demand profile summaries per facility with peak/average/minimum demand, load factor, and peak timing.';

COMMENT ON TABLE pack036_utility_analysis.gl_peak_events IS
    'Individual peak demand events with magnitude, duration, cost impact, and avoidability assessment.';

COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.interval_id IS
    'Unique identifier for the interval reading.';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.facility_id IS
    'Reference to the facility this interval data belongs to.';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.meter_id IS
    'Smart meter identifier for this interval data stream.';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.timestamp IS
    'Timestamp for the start of this measurement interval.';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.demand_kw IS
    'Demand in kW for this interval.';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.energy_kwh IS
    'Energy consumption in kWh for this interval.';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.power_factor IS
    'Power factor for this interval (0 to 1).';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.reactive_kvar IS
    'Reactive power in kVAR for this interval.';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.interval_minutes IS
    'Duration of the measurement interval in minutes (1, 5, 15, 30, or 60).';
COMMENT ON COLUMN pack036_utility_analysis.gl_interval_data.data_quality IS
    'Data quality flag: ACTUAL, ESTIMATED, INTERPOLATED, MISSING, VALIDATED.';
COMMENT ON COLUMN pack036_utility_analysis.gl_demand_profiles.profile_id IS
    'Unique identifier for the demand profile.';
COMMENT ON COLUMN pack036_utility_analysis.gl_demand_profiles.load_factor_pct IS
    'Load factor percentage (average demand / peak demand * 100). Higher = flatter load.';
COMMENT ON COLUMN pack036_utility_analysis.gl_demand_profiles.base_load_kw IS
    'Estimated base load in kW (minimum consistent demand, typically overnight).';
COMMENT ON COLUMN pack036_utility_analysis.gl_demand_profiles.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_peak_events.peak_type IS
    'Peak event classification: MONTHLY_PEAK, ANNUAL_PEAK, SEASONAL_PEAK, RATCHET_PEAK, COINCIDENT_PEAK, SYSTEM_PEAK, NEAR_PEAK.';
COMMENT ON COLUMN pack036_utility_analysis.gl_peak_events.avoidable IS
    'Whether this peak event could have been avoided through load management.';
COMMENT ON COLUMN pack036_utility_analysis.gl_peak_events.cost_impact_eur IS
    'Estimated cost impact of this peak event on demand charges in EUR.';

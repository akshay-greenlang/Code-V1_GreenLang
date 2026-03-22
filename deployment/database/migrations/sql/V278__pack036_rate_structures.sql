-- =============================================================================
-- V278: PACK-036 Utility Analysis Pack - Rate Structures & Comparisons
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    003 of 010
-- Date:         March 2026
--
-- Tables for utility rate structure modelling including tiered rates,
-- time-of-use schedules, demand charges, and rate comparison analyses.
-- Supports complex tariff structures across multiple jurisdictions for
-- rate optimization and tariff switching recommendations.
--
-- Tables (5):
--   1. pack036_utility_analysis.gl_rate_structures
--   2. pack036_utility_analysis.gl_rate_tiers
--   3. pack036_utility_analysis.gl_tou_schedules
--   4. pack036_utility_analysis.gl_demand_charges
--   5. pack036_utility_analysis.gl_rate_comparisons
--
-- Previous: V277__pack036_bill_errors.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_rate_structures
-- =============================================================================
-- Master rate/tariff definitions from utilities. Each rate has a type,
-- jurisdiction, effective/expiry dates, fixed charges, and optional
-- power factor and voltage adjustments.

CREATE TABLE pack036_utility_analysis.gl_rate_structures (
    rate_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    rate_name               VARCHAR(255)    NOT NULL,
    rate_code               VARCHAR(50),
    rate_type               VARCHAR(30)     NOT NULL,
    utility_name            VARCHAR(255)    NOT NULL,
    jurisdiction            VARCHAR(100),
    effective_date          DATE            NOT NULL,
    expiry_date             DATE,
    fixed_charges_monthly   NUMERIC(12,2)   DEFAULT 0,
    minimum_bill            NUMERIC(12,2),
    pf_adjustment_threshold NUMERIC(5,4),
    voltage_discount_pct    NUMERIC(6,4),
    currency                VARCHAR(3)      NOT NULL DEFAULT 'EUR',
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_rs_rate_type CHECK (
        rate_type IN (
            'FLAT', 'TIERED', 'TOU', 'TIERED_TOU', 'DEMAND',
            'REAL_TIME', 'INTERRUPTIBLE', 'SEASONAL', 'DYNAMIC'
        )
    ),
    CONSTRAINT chk_p036_rs_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'STEAM',
            'CHILLED_WATER', 'DISTRICT_HEATING', 'DISTRICT_COOLING'
        )
    ),
    CONSTRAINT chk_p036_rs_dates CHECK (
        expiry_date IS NULL OR expiry_date > effective_date
    ),
    CONSTRAINT chk_p036_rs_fixed CHECK (
        fixed_charges_monthly >= 0
    ),
    CONSTRAINT chk_p036_rs_min_bill CHECK (
        minimum_bill IS NULL OR minimum_bill >= 0
    ),
    CONSTRAINT chk_p036_rs_pf_threshold CHECK (
        pf_adjustment_threshold IS NULL OR (pf_adjustment_threshold >= 0 AND pf_adjustment_threshold <= 1)
    ),
    CONSTRAINT chk_p036_rs_voltage_disc CHECK (
        voltage_discount_pct IS NULL OR (voltage_discount_pct >= 0 AND voltage_discount_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_rs_tenant         ON pack036_utility_analysis.gl_rate_structures(tenant_id);
CREATE INDEX idx_p036_rs_rate_type      ON pack036_utility_analysis.gl_rate_structures(rate_type);
CREATE INDEX idx_p036_rs_utility        ON pack036_utility_analysis.gl_rate_structures(utility_name);
CREATE INDEX idx_p036_rs_jurisdiction   ON pack036_utility_analysis.gl_rate_structures(jurisdiction);
CREATE INDEX idx_p036_rs_effective      ON pack036_utility_analysis.gl_rate_structures(effective_date DESC);
CREATE INDEX idx_p036_rs_commodity      ON pack036_utility_analysis.gl_rate_structures(commodity);
CREATE INDEX idx_p036_rs_active         ON pack036_utility_analysis.gl_rate_structures(is_active);
CREATE INDEX idx_p036_rs_created        ON pack036_utility_analysis.gl_rate_structures(created_at DESC);
CREATE INDEX idx_p036_rs_metadata       ON pack036_utility_analysis.gl_rate_structures USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_rs_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_rate_structures
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_rate_tiers
-- =============================================================================
-- Tiered (block) rate definitions within a rate structure. Each tier
-- defines a consumption range and the per-kWh rate for that block.

CREATE TABLE pack036_utility_analysis.gl_rate_tiers (
    tier_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    rate_id                 UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_rate_structures(rate_id) ON DELETE CASCADE,
    tier_number             INTEGER         NOT NULL,
    lower_kwh               NUMERIC(14,4)   NOT NULL DEFAULT 0,
    upper_kwh               NUMERIC(14,4),
    rate_per_kwh            NUMERIC(12,8)   NOT NULL,
    season                  VARCHAR(20),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_rt_tier_num CHECK (
        tier_number >= 1
    ),
    CONSTRAINT chk_p036_rt_lower CHECK (
        lower_kwh >= 0
    ),
    CONSTRAINT chk_p036_rt_upper CHECK (
        upper_kwh IS NULL OR upper_kwh > lower_kwh
    ),
    CONSTRAINT chk_p036_rt_rate CHECK (
        rate_per_kwh >= 0
    ),
    CONSTRAINT chk_p036_rt_season CHECK (
        season IS NULL OR season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ANNUAL')
    ),
    CONSTRAINT uq_p036_rt_rate_tier UNIQUE (rate_id, tier_number, season)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_rt_rate           ON pack036_utility_analysis.gl_rate_tiers(rate_id);
CREATE INDEX idx_p036_rt_season         ON pack036_utility_analysis.gl_rate_tiers(season);
CREATE INDEX idx_p036_rt_tier           ON pack036_utility_analysis.gl_rate_tiers(tier_number);

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_tou_schedules
-- =============================================================================
-- Time-of-use period definitions within a rate structure. Each schedule
-- entry defines a TOU period (on-peak, off-peak, etc.) with hour ranges,
-- applicable days, and per-kWh rate.

CREATE TABLE pack036_utility_analysis.gl_tou_schedules (
    schedule_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    rate_id                 UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_rate_structures(rate_id) ON DELETE CASCADE,
    period                  VARCHAR(30)     NOT NULL,
    start_hour              INTEGER         NOT NULL,
    end_hour                INTEGER         NOT NULL,
    days                    VARCHAR(30)     NOT NULL DEFAULT 'WEEKDAY',
    season                  VARCHAR(20),
    rate_per_kwh            NUMERIC(12,8)   NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_ts_period CHECK (
        period IN (
            'ON_PEAK', 'MID_PEAK', 'OFF_PEAK', 'SUPER_OFF_PEAK',
            'CRITICAL_PEAK', 'SHOULDER'
        )
    ),
    CONSTRAINT chk_p036_ts_start_hour CHECK (
        start_hour >= 0 AND start_hour <= 23
    ),
    CONSTRAINT chk_p036_ts_end_hour CHECK (
        end_hour >= 0 AND end_hour <= 24
    ),
    CONSTRAINT chk_p036_ts_days CHECK (
        days IN ('WEEKDAY', 'WEEKEND', 'ALL', 'MONDAY', 'TUESDAY', 'WEDNESDAY',
                 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY', 'HOLIDAY')
    ),
    CONSTRAINT chk_p036_ts_season CHECK (
        season IS NULL OR season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ANNUAL')
    ),
    CONSTRAINT chk_p036_ts_rate CHECK (
        rate_per_kwh >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_ts_rate           ON pack036_utility_analysis.gl_tou_schedules(rate_id);
CREATE INDEX idx_p036_ts_period         ON pack036_utility_analysis.gl_tou_schedules(period);
CREATE INDEX idx_p036_ts_season         ON pack036_utility_analysis.gl_tou_schedules(season);
CREATE INDEX idx_p036_ts_days           ON pack036_utility_analysis.gl_tou_schedules(days);

-- =============================================================================
-- Table 4: pack036_utility_analysis.gl_demand_charges
-- =============================================================================
-- Demand charge components within a rate structure. Supports multiple
-- demand types (on-peak, off-peak, ratchet), seasonal variations, and
-- minimum demand thresholds.

CREATE TABLE pack036_utility_analysis.gl_demand_charges (
    charge_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    rate_id                 UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_rate_structures(rate_id) ON DELETE CASCADE,
    demand_type             VARCHAR(30)     NOT NULL,
    rate_per_kw             NUMERIC(12,6)   NOT NULL,
    ratchet_pct             NUMERIC(6,4),
    minimum_kw              NUMERIC(12,4),
    maximum_kw              NUMERIC(12,4),
    season                  VARCHAR(20),
    measurement_window_min  INTEGER         DEFAULT 15,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_dc_demand_type CHECK (
        demand_type IN (
            'ON_PEAK', 'OFF_PEAK', 'MID_PEAK', 'MAX', 'COINCIDENT',
            'NON_COINCIDENT', 'RATCHET', 'FACILITIES', 'DISTRIBUTION'
        )
    ),
    CONSTRAINT chk_p036_dc_rate CHECK (
        rate_per_kw >= 0
    ),
    CONSTRAINT chk_p036_dc_ratchet CHECK (
        ratchet_pct IS NULL OR (ratchet_pct >= 0 AND ratchet_pct <= 100)
    ),
    CONSTRAINT chk_p036_dc_min_kw CHECK (
        minimum_kw IS NULL OR minimum_kw >= 0
    ),
    CONSTRAINT chk_p036_dc_max_kw CHECK (
        maximum_kw IS NULL OR maximum_kw >= 0
    ),
    CONSTRAINT chk_p036_dc_season CHECK (
        season IS NULL OR season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ANNUAL')
    ),
    CONSTRAINT chk_p036_dc_window CHECK (
        measurement_window_min IN (5, 15, 30, 60)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_dc_rate           ON pack036_utility_analysis.gl_demand_charges(rate_id);
CREATE INDEX idx_p036_dc_demand_type    ON pack036_utility_analysis.gl_demand_charges(demand_type);
CREATE INDEX idx_p036_dc_season         ON pack036_utility_analysis.gl_demand_charges(season);

-- =============================================================================
-- Table 5: pack036_utility_analysis.gl_rate_comparisons
-- =============================================================================
-- Rate comparison analysis results. Compares a facility's current rate
-- against alternative rates to identify the optimal tariff and projected
-- annual savings.

CREATE TABLE pack036_utility_analysis.gl_rate_comparisons (
    comparison_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    analysis_date           DATE            NOT NULL DEFAULT CURRENT_DATE,
    current_rate_id         UUID            REFERENCES pack036_utility_analysis.gl_rate_structures(rate_id) ON DELETE SET NULL,
    optimal_rate_id         UUID            REFERENCES pack036_utility_analysis.gl_rate_structures(rate_id) ON DELETE SET NULL,
    current_annual_cost_eur NUMERIC(14,2),
    optimal_annual_cost_eur NUMERIC(14,2),
    annual_savings_eur      NUMERIC(14,2)   NOT NULL DEFAULT 0,
    savings_pct             NUMERIC(6,2),
    analysis_period_months  INTEGER         DEFAULT 12,
    rates_evaluated         INTEGER         DEFAULT 0,
    recommendation          TEXT,
    confidence_level        VARCHAR(20)     DEFAULT 'MEDIUM',
    status                  VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_rc_confidence CHECK (
        confidence_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERIFIED')
    ),
    CONSTRAINT chk_p036_rc_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'APPROVED', 'IMPLEMENTED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p036_rc_rates_eval CHECK (
        rates_evaluated >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_rc_tenant         ON pack036_utility_analysis.gl_rate_comparisons(tenant_id);
CREATE INDEX idx_p036_rc_facility       ON pack036_utility_analysis.gl_rate_comparisons(facility_id);
CREATE INDEX idx_p036_rc_analysis_date  ON pack036_utility_analysis.gl_rate_comparisons(analysis_date DESC);
CREATE INDEX idx_p036_rc_current_rate   ON pack036_utility_analysis.gl_rate_comparisons(current_rate_id);
CREATE INDEX idx_p036_rc_optimal_rate   ON pack036_utility_analysis.gl_rate_comparisons(optimal_rate_id);
CREATE INDEX idx_p036_rc_savings        ON pack036_utility_analysis.gl_rate_comparisons(annual_savings_eur DESC);
CREATE INDEX idx_p036_rc_status         ON pack036_utility_analysis.gl_rate_comparisons(status);
CREATE INDEX idx_p036_rc_created        ON pack036_utility_analysis.gl_rate_comparisons(created_at DESC);

-- Composite: facility + date for historical comparison lookup
CREATE INDEX idx_p036_rc_fac_date       ON pack036_utility_analysis.gl_rate_comparisons(facility_id, analysis_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_rc_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_rate_comparisons
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_rate_structures ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_rate_tiers ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_tou_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_demand_charges ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_rate_comparisons ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_rs_tenant_isolation
    ON pack036_utility_analysis.gl_rate_structures
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_rs_service_bypass
    ON pack036_utility_analysis.gl_rate_structures
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_rt_tenant_isolation
    ON pack036_utility_analysis.gl_rate_tiers
    USING (rate_id IN (
        SELECT rate_id FROM pack036_utility_analysis.gl_rate_structures
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p036_rt_service_bypass
    ON pack036_utility_analysis.gl_rate_tiers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_ts_tenant_isolation
    ON pack036_utility_analysis.gl_tou_schedules
    USING (rate_id IN (
        SELECT rate_id FROM pack036_utility_analysis.gl_rate_structures
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p036_ts_service_bypass
    ON pack036_utility_analysis.gl_tou_schedules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_dc_tenant_isolation
    ON pack036_utility_analysis.gl_demand_charges
    USING (rate_id IN (
        SELECT rate_id FROM pack036_utility_analysis.gl_rate_structures
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p036_dc_service_bypass
    ON pack036_utility_analysis.gl_demand_charges
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_rc_tenant_isolation
    ON pack036_utility_analysis.gl_rate_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_rc_service_bypass
    ON pack036_utility_analysis.gl_rate_comparisons
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_rate_structures TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_rate_tiers TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_tou_schedules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_demand_charges TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_rate_comparisons TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_rate_structures IS
    'Master rate/tariff definitions from utilities with type, jurisdiction, effective dates, fixed charges, and adjustment parameters.';

COMMENT ON TABLE pack036_utility_analysis.gl_rate_tiers IS
    'Tiered (block) rate definitions within a rate structure with consumption ranges and per-kWh pricing.';

COMMENT ON TABLE pack036_utility_analysis.gl_tou_schedules IS
    'Time-of-use period definitions with hour ranges, applicable days, seasonal variation, and per-kWh rates.';

COMMENT ON TABLE pack036_utility_analysis.gl_demand_charges IS
    'Demand charge components within a rate structure with demand types, ratchet clauses, and seasonal variations.';

COMMENT ON TABLE pack036_utility_analysis.gl_rate_comparisons IS
    'Rate comparison analysis results comparing current vs optimal tariff with projected annual savings.';

COMMENT ON COLUMN pack036_utility_analysis.gl_rate_structures.rate_id IS
    'Unique identifier for the rate structure.';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_structures.rate_type IS
    'Rate structure type: FLAT, TIERED, TOU, TIERED_TOU, DEMAND, REAL_TIME, INTERRUPTIBLE, SEASONAL, DYNAMIC.';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_structures.pf_adjustment_threshold IS
    'Power factor threshold below which penalty charges apply (e.g., 0.90).';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_structures.voltage_discount_pct IS
    'Percentage discount for high-voltage (primary metering) customers.';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_structures.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_tiers.tier_number IS
    'Tier position in the block rate schedule (1 = first block).';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_tiers.lower_kwh IS
    'Lower bound of consumption for this tier (inclusive).';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_tiers.upper_kwh IS
    'Upper bound of consumption for this tier (exclusive). NULL for unlimited top tier.';
COMMENT ON COLUMN pack036_utility_analysis.gl_tou_schedules.period IS
    'TOU period type: ON_PEAK, MID_PEAK, OFF_PEAK, SUPER_OFF_PEAK, CRITICAL_PEAK, SHOULDER.';
COMMENT ON COLUMN pack036_utility_analysis.gl_demand_charges.demand_type IS
    'Demand charge type: ON_PEAK, OFF_PEAK, MAX, COINCIDENT, RATCHET, FACILITIES, DISTRIBUTION.';
COMMENT ON COLUMN pack036_utility_analysis.gl_demand_charges.ratchet_pct IS
    'Ratchet percentage - minimum demand billed as percentage of historical peak (e.g., 80%).';
COMMENT ON COLUMN pack036_utility_analysis.gl_demand_charges.measurement_window_min IS
    'Demand measurement interval in minutes (5, 15, 30, or 60).';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_comparisons.annual_savings_eur IS
    'Projected annual savings from switching to the optimal rate in EUR.';
COMMENT ON COLUMN pack036_utility_analysis.gl_rate_comparisons.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';

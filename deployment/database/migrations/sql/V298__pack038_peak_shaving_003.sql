-- =============================================================================
-- V298: PACK-038 Peak Shaving Pack - Demand Charges & Tariff Structures
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Tariff structure modeling and demand charge calculation tables. Captures
-- utility rate schedules with tiered demand charges, time-of-use periods,
-- ratchet clauses, and enables tariff comparison analysis for optimal
-- rate selection and demand charge reduction strategies.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_tariff_structures
--   2. pack038_peak_shaving.ps_demand_charges
--   3. pack038_peak_shaving.ps_charge_components
--   4. pack038_peak_shaving.ps_marginal_values
--   5. pack038_peak_shaving.ps_tariff_comparisons
--
-- Previous: V297__pack038_peak_shaving_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_tariff_structures
-- =============================================================================
-- Utility tariff rate schedule definitions with demand charge tiers,
-- time-of-use periods, ratchet clauses, and seasonal variations. Serves
-- as the master reference for demand charge calculations.

CREATE TABLE pack038_peak_shaving.ps_tariff_structures (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    tariff_code             VARCHAR(50)     NOT NULL,
    tariff_name             VARCHAR(255)    NOT NULL,
    utility_name            VARCHAR(255)    NOT NULL,
    utility_id              VARCHAR(50),
    state_province          VARCHAR(50),
    country_code            CHAR(2)         NOT NULL DEFAULT 'US',
    iso_rto_region          VARCHAR(30),
    rate_structure          VARCHAR(30)     NOT NULL DEFAULT 'TOU',
    customer_class          VARCHAR(30)     NOT NULL DEFAULT 'COMMERCIAL',
    voltage_level           VARCHAR(30),
    min_demand_kw           NUMERIC(12,3),
    max_demand_kw           NUMERIC(12,3),
    effective_date          DATE            NOT NULL,
    expiration_date         DATE,
    tiers                   JSONB           NOT NULL DEFAULT '[]',
    tou_periods             JSONB           NOT NULL DEFAULT '[]',
    ratchet_clause          JSONB           DEFAULT '{}',
    seasonal_adjustments    JSONB           DEFAULT '{}',
    power_factor_adjustment JSONB           DEFAULT '{}',
    demand_window_minutes   INTEGER         NOT NULL DEFAULT 15,
    billing_cycle_days      INTEGER         DEFAULT 30,
    minimum_bill_amount     NUMERIC(10,2),
    facilities_charge       NUMERIC(10,2),
    transformer_discount_pct NUMERIC(5,2),
    tax_rate_pct            NUMERIC(5,4),
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    source_url              VARCHAR(500),
    source_document         VARCHAR(255),
    tariff_status           VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ts_rate_structure CHECK (
        rate_structure IN (
            'FLAT', 'TOU', 'TIERED', 'TOU_TIERED', 'REAL_TIME_PRICING',
            'CRITICAL_PEAK_PRICING', 'VARIABLE_PEAK_PRICING', 'DEMAND_RESPONSE'
        )
    ),
    CONSTRAINT chk_p038_ts_customer_class CHECK (
        customer_class IN (
            'RESIDENTIAL', 'SMALL_COMMERCIAL', 'COMMERCIAL', 'LARGE_COMMERCIAL',
            'INDUSTRIAL', 'AGRICULTURAL', 'STREET_LIGHTING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_ts_voltage CHECK (
        voltage_level IS NULL OR voltage_level IN (
            'SECONDARY', 'PRIMARY', 'SUBTRANSMISSION', 'TRANSMISSION'
        )
    ),
    CONSTRAINT chk_p038_ts_region CHECK (
        iso_rto_region IS NULL OR iso_rto_region IN (
            'PJM', 'ERCOT', 'CAISO', 'ISO_NE', 'NYISO', 'MISO', 'SPP',
            'UK_NGESO', 'DE_TENNET', 'DE_AMPRION', 'DE_50HZ', 'DE_TRANSNET',
            'FR_RTE', 'NL_TENNET', 'ES_REE', 'IT_TERNA', 'AU_AEMO',
            'JP_TEPCO', 'JP_KEPCO', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_ts_demand_window CHECK (
        demand_window_minutes IN (5, 15, 30, 60)
    ),
    CONSTRAINT chk_p038_ts_status CHECK (
        tariff_status IN ('DRAFT', 'ACTIVE', 'SUPERSEDED', 'EXPIRED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p038_ts_min_demand CHECK (
        min_demand_kw IS NULL OR min_demand_kw >= 0
    ),
    CONSTRAINT chk_p038_ts_max_demand CHECK (
        max_demand_kw IS NULL OR max_demand_kw > 0
    ),
    CONSTRAINT chk_p038_ts_dates CHECK (
        expiration_date IS NULL OR effective_date <= expiration_date
    ),
    CONSTRAINT chk_p038_ts_tax CHECK (
        tax_rate_pct IS NULL OR (tax_rate_pct >= 0 AND tax_rate_pct <= 50)
    ),
    CONSTRAINT chk_p038_ts_discount CHECK (
        transformer_discount_pct IS NULL OR (transformer_discount_pct >= 0 AND transformer_discount_pct <= 100)
    ),
    CONSTRAINT uq_p038_ts_tariff_code UNIQUE (tenant_id, utility_name, tariff_code, effective_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ts_tenant          ON pack038_peak_shaving.ps_tariff_structures(tenant_id);
CREATE INDEX idx_p038_ts_tariff_code     ON pack038_peak_shaving.ps_tariff_structures(tariff_code);
CREATE INDEX idx_p038_ts_utility         ON pack038_peak_shaving.ps_tariff_structures(utility_name);
CREATE INDEX idx_p038_ts_region          ON pack038_peak_shaving.ps_tariff_structures(iso_rto_region);
CREATE INDEX idx_p038_ts_rate_structure  ON pack038_peak_shaving.ps_tariff_structures(rate_structure);
CREATE INDEX idx_p038_ts_customer_class  ON pack038_peak_shaving.ps_tariff_structures(customer_class);
CREATE INDEX idx_p038_ts_effective       ON pack038_peak_shaving.ps_tariff_structures(effective_date DESC);
CREATE INDEX idx_p038_ts_status          ON pack038_peak_shaving.ps_tariff_structures(tariff_status);
CREATE INDEX idx_p038_ts_country         ON pack038_peak_shaving.ps_tariff_structures(country_code);
CREATE INDEX idx_p038_ts_tiers           ON pack038_peak_shaving.ps_tariff_structures USING GIN(tiers);
CREATE INDEX idx_p038_ts_tou             ON pack038_peak_shaving.ps_tariff_structures USING GIN(tou_periods);
CREATE INDEX idx_p038_ts_ratchet         ON pack038_peak_shaving.ps_tariff_structures USING GIN(ratchet_clause);
CREATE INDEX idx_p038_ts_created         ON pack038_peak_shaving.ps_tariff_structures(created_at DESC);

-- Active tariffs by utility for rate comparison
CREATE INDEX idx_p038_ts_active_utility  ON pack038_peak_shaving.ps_tariff_structures(utility_name, tariff_code)
    WHERE tariff_status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_ts_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_tariff_structures
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_demand_charges
-- =============================================================================
-- Calculated demand charges per billing period for each load profile.
-- Links peak events to the tariff structure and computes the actual
-- demand charge components including TOU, tiered, and ratchet charges.

CREATE TABLE pack038_peak_shaving.ps_demand_charges (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tariff_id               UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_tariff_structures(id),
    peak_event_id           UUID            REFERENCES pack038_peak_shaving.ps_peak_events(id),
    tenant_id               UUID            NOT NULL,
    billing_period_start    DATE            NOT NULL,
    billing_period_end      DATE            NOT NULL,
    billing_demand_kw       NUMERIC(12,3)   NOT NULL,
    actual_peak_kw          NUMERIC(12,3)   NOT NULL,
    on_peak_demand_kw       NUMERIC(12,3),
    mid_peak_demand_kw      NUMERIC(12,3),
    off_peak_demand_kw      NUMERIC(12,3),
    ratchet_demand_kw       NUMERIC(12,3),
    ratchet_source_month    DATE,
    pf_adjusted_demand_kw   NUMERIC(12,3),
    contract_demand_kw      NUMERIC(12,3),
    total_demand_charge     NUMERIC(12,2)   NOT NULL,
    on_peak_charge          NUMERIC(12,2),
    mid_peak_charge         NUMERIC(12,2),
    off_peak_charge         NUMERIC(12,2),
    facilities_charge       NUMERIC(12,2),
    excess_demand_charge    NUMERIC(12,2),
    pf_penalty_charge       NUMERIC(12,2),
    ratchet_adder           NUMERIC(12,2),
    total_energy_charge     NUMERIC(12,2),
    total_bill              NUMERIC(12,2),
    demand_pct_of_bill      NUMERIC(7,4),
    effective_rate_per_kw   NUMERIC(10,4),
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    billing_status          VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_dc_billing_demand CHECK (
        billing_demand_kw >= 0
    ),
    CONSTRAINT chk_p038_dc_actual_peak CHECK (
        actual_peak_kw >= 0
    ),
    CONSTRAINT chk_p038_dc_total_charge CHECK (
        total_demand_charge >= 0
    ),
    CONSTRAINT chk_p038_dc_demand_pct CHECK (
        demand_pct_of_bill IS NULL OR (demand_pct_of_bill >= 0 AND demand_pct_of_bill <= 100)
    ),
    CONSTRAINT chk_p038_dc_dates CHECK (
        billing_period_start <= billing_period_end
    ),
    CONSTRAINT chk_p038_dc_status CHECK (
        billing_status IN ('ESTIMATED', 'CALCULATED', 'ACTUAL', 'RECONCILED', 'DISPUTED')
    ),
    CONSTRAINT uq_p038_dc_profile_period UNIQUE (profile_id, tariff_id, billing_period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_dc_profile         ON pack038_peak_shaving.ps_demand_charges(profile_id);
CREATE INDEX idx_p038_dc_tariff          ON pack038_peak_shaving.ps_demand_charges(tariff_id);
CREATE INDEX idx_p038_dc_peak_event      ON pack038_peak_shaving.ps_demand_charges(peak_event_id);
CREATE INDEX idx_p038_dc_tenant          ON pack038_peak_shaving.ps_demand_charges(tenant_id);
CREATE INDEX idx_p038_dc_billing_start   ON pack038_peak_shaving.ps_demand_charges(billing_period_start DESC);
CREATE INDEX idx_p038_dc_total_charge    ON pack038_peak_shaving.ps_demand_charges(total_demand_charge DESC);
CREATE INDEX idx_p038_dc_billing_demand  ON pack038_peak_shaving.ps_demand_charges(billing_demand_kw DESC);
CREATE INDEX idx_p038_dc_status          ON pack038_peak_shaving.ps_demand_charges(billing_status);
CREATE INDEX idx_p038_dc_created         ON pack038_peak_shaving.ps_demand_charges(created_at DESC);

-- Composite: profile + period for billing history
CREATE INDEX idx_p038_dc_profile_period  ON pack038_peak_shaving.ps_demand_charges(profile_id, billing_period_start DESC, total_demand_charge DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_dc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_demand_charges
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_charge_components
-- =============================================================================
-- Granular demand charge component breakdown per billing period. Each
-- row represents a single charge line item (e.g., summer on-peak tier 1,
-- facilities charge, excess demand penalty) for detailed bill analysis.

CREATE TABLE pack038_peak_shaving.ps_charge_components (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    demand_charge_id        UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_demand_charges(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    component_type          VARCHAR(30)     NOT NULL,
    component_name          VARCHAR(100)    NOT NULL,
    tou_period              VARCHAR(20),
    tier_number             INTEGER,
    tier_lower_kw           NUMERIC(12,3),
    tier_upper_kw           NUMERIC(12,3),
    applicable_demand_kw    NUMERIC(12,3)   NOT NULL,
    rate_per_kw             NUMERIC(10,4)   NOT NULL,
    charge_amount           NUMERIC(12,2)   NOT NULL,
    season                  VARCHAR(20),
    is_ratchet_based        BOOLEAN         DEFAULT false,
    is_pf_adjusted          BOOLEAN         DEFAULT false,
    calculation_notes       TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cc_type CHECK (
        component_type IN (
            'ON_PEAK_DEMAND', 'MID_PEAK_DEMAND', 'OFF_PEAK_DEMAND',
            'NON_COINCIDENT_DEMAND', 'COINCIDENT_DEMAND',
            'FACILITIES_CHARGE', 'EXCESS_DEMAND', 'RATCHET_DEMAND',
            'PF_PENALTY', 'PF_CREDIT', 'TRANSFORMER_DISCOUNT',
            'DEMAND_CREDIT', 'STANDBY_DEMAND', 'SUPPLEMENTAL_DEMAND',
            'INTERRUPTIBLE_CREDIT', 'TAX', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cc_tou CHECK (
        tou_period IS NULL OR tou_period IN (
            'ON_PEAK', 'MID_PEAK', 'OFF_PEAK', 'SUPER_PEAK',
            'CRITICAL_PEAK', 'SHOULDER', 'ALL_HOURS'
        )
    ),
    CONSTRAINT chk_p038_cc_season CHECK (
        season IS NULL OR season IN ('SUMMER', 'WINTER', 'SHOULDER', 'SPRING', 'FALL')
    ),
    CONSTRAINT chk_p038_cc_demand CHECK (
        applicable_demand_kw >= 0
    ),
    CONSTRAINT chk_p038_cc_tier CHECK (
        tier_number IS NULL OR tier_number >= 1
    ),
    CONSTRAINT chk_p038_cc_tier_bounds CHECK (
        tier_lower_kw IS NULL OR tier_upper_kw IS NULL OR tier_lower_kw <= tier_upper_kw
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cc_demand_charge   ON pack038_peak_shaving.ps_charge_components(demand_charge_id);
CREATE INDEX idx_p038_cc_tenant          ON pack038_peak_shaving.ps_charge_components(tenant_id);
CREATE INDEX idx_p038_cc_type            ON pack038_peak_shaving.ps_charge_components(component_type);
CREATE INDEX idx_p038_cc_tou             ON pack038_peak_shaving.ps_charge_components(tou_period);
CREATE INDEX idx_p038_cc_season          ON pack038_peak_shaving.ps_charge_components(season);
CREATE INDEX idx_p038_cc_charge          ON pack038_peak_shaving.ps_charge_components(charge_amount DESC);
CREATE INDEX idx_p038_cc_created         ON pack038_peak_shaving.ps_charge_components(created_at DESC);

-- ---------------------------------------------------------------------------
-- (No trigger needed - no updated_at column)
-- ---------------------------------------------------------------------------

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_marginal_values
-- =============================================================================
-- Marginal value of peak demand reduction at each kW level. Captures
-- the incremental savings per kW of peak reduction considering demand
-- charge tiers, ratchet effects, and coincident peak charges.

CREATE TABLE pack038_peak_shaving.ps_marginal_values (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tariff_id               UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_tariff_structures(id),
    tenant_id               UUID            NOT NULL,
    analysis_period_start   DATE            NOT NULL,
    analysis_period_end     DATE            NOT NULL,
    demand_level_kw         NUMERIC(12,3)   NOT NULL,
    marginal_rate_per_kw    NUMERIC(10,4)   NOT NULL,
    marginal_annual_value   NUMERIC(12,2),
    includes_ratchet_value  BOOLEAN         DEFAULT false,
    ratchet_multiplier      NUMERIC(5,3),
    includes_cp_value       BOOLEAN         DEFAULT false,
    cp_value_per_kw         NUMERIC(10,4),
    includes_transmission   BOOLEAN         DEFAULT false,
    transmission_value_per_kw NUMERIC(10,4),
    total_marginal_value    NUMERIC(12,2)   NOT NULL,
    tou_period              VARCHAR(20),
    season                  VARCHAR(20),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_mv_demand CHECK (
        demand_level_kw >= 0
    ),
    CONSTRAINT chk_p038_mv_marginal CHECK (
        marginal_rate_per_kw >= 0
    ),
    CONSTRAINT chk_p038_mv_total CHECK (
        total_marginal_value >= 0
    ),
    CONSTRAINT chk_p038_mv_ratchet_mult CHECK (
        ratchet_multiplier IS NULL OR (ratchet_multiplier >= 0 AND ratchet_multiplier <= 12)
    ),
    CONSTRAINT chk_p038_mv_dates CHECK (
        analysis_period_start <= analysis_period_end
    ),
    CONSTRAINT chk_p038_mv_tou CHECK (
        tou_period IS NULL OR tou_period IN (
            'ON_PEAK', 'MID_PEAK', 'OFF_PEAK', 'SUPER_PEAK',
            'CRITICAL_PEAK', 'SHOULDER', 'ALL_HOURS'
        )
    ),
    CONSTRAINT chk_p038_mv_season CHECK (
        season IS NULL OR season IN ('SUMMER', 'WINTER', 'SHOULDER', 'SPRING', 'FALL')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_mv_profile         ON pack038_peak_shaving.ps_marginal_values(profile_id);
CREATE INDEX idx_p038_mv_tariff          ON pack038_peak_shaving.ps_marginal_values(tariff_id);
CREATE INDEX idx_p038_mv_tenant          ON pack038_peak_shaving.ps_marginal_values(tenant_id);
CREATE INDEX idx_p038_mv_demand_level    ON pack038_peak_shaving.ps_marginal_values(demand_level_kw);
CREATE INDEX idx_p038_mv_total_value     ON pack038_peak_shaving.ps_marginal_values(total_marginal_value DESC);
CREATE INDEX idx_p038_mv_period          ON pack038_peak_shaving.ps_marginal_values(analysis_period_start DESC);
CREATE INDEX idx_p038_mv_created         ON pack038_peak_shaving.ps_marginal_values(created_at DESC);

-- Composite: profile + demand level for marginal value curve rendering
CREATE INDEX idx_p038_mv_profile_demand  ON pack038_peak_shaving.ps_marginal_values(profile_id, demand_level_kw);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_mv_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_marginal_values
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_tariff_comparisons
-- =============================================================================
-- Side-by-side comparison of demand charges under different tariff
-- schedules for the same load profile. Used for rate optimization
-- and utility tariff selection recommendations.

CREATE TABLE pack038_peak_shaving.ps_tariff_comparisons (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    comparison_name         VARCHAR(255)    NOT NULL,
    base_tariff_id          UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_tariff_structures(id),
    alt_tariff_id           UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_tariff_structures(id),
    comparison_period_start DATE            NOT NULL,
    comparison_period_end   DATE            NOT NULL,
    base_annual_demand_cost NUMERIC(12,2)   NOT NULL,
    alt_annual_demand_cost  NUMERIC(12,2)   NOT NULL,
    demand_cost_difference  NUMERIC(12,2),
    base_annual_energy_cost NUMERIC(12,2),
    alt_annual_energy_cost  NUMERIC(12,2),
    energy_cost_difference  NUMERIC(12,2),
    base_total_cost         NUMERIC(12,2),
    alt_total_cost          NUMERIC(12,2),
    total_savings           NUMERIC(12,2),
    savings_pct             NUMERIC(7,4),
    switching_fee           NUMERIC(10,2),
    net_savings             NUMERIC(12,2),
    recommendation          VARCHAR(30)     NOT NULL DEFAULT 'EVALUATE',
    analysis_notes          TEXT,
    monthly_breakdown       JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_tc_dates CHECK (
        comparison_period_start <= comparison_period_end
    ),
    CONSTRAINT chk_p038_tc_base_cost CHECK (
        base_annual_demand_cost >= 0
    ),
    CONSTRAINT chk_p038_tc_alt_cost CHECK (
        alt_annual_demand_cost >= 0
    ),
    CONSTRAINT chk_p038_tc_recommendation CHECK (
        recommendation IN ('SWITCH', 'STAY', 'EVALUATE', 'CONDITIONAL')
    ),
    CONSTRAINT chk_p038_tc_different_tariffs CHECK (
        base_tariff_id != alt_tariff_id
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_tc_profile         ON pack038_peak_shaving.ps_tariff_comparisons(profile_id);
CREATE INDEX idx_p038_tc_tenant          ON pack038_peak_shaving.ps_tariff_comparisons(tenant_id);
CREATE INDEX idx_p038_tc_base_tariff     ON pack038_peak_shaving.ps_tariff_comparisons(base_tariff_id);
CREATE INDEX idx_p038_tc_alt_tariff      ON pack038_peak_shaving.ps_tariff_comparisons(alt_tariff_id);
CREATE INDEX idx_p038_tc_savings         ON pack038_peak_shaving.ps_tariff_comparisons(total_savings DESC);
CREATE INDEX idx_p038_tc_recommendation  ON pack038_peak_shaving.ps_tariff_comparisons(recommendation);
CREATE INDEX idx_p038_tc_created         ON pack038_peak_shaving.ps_tariff_comparisons(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_tc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_tariff_comparisons
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_tariff_structures ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_demand_charges ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_charge_components ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_marginal_values ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_tariff_comparisons ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_ts_tenant_isolation
    ON pack038_peak_shaving.ps_tariff_structures
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ts_service_bypass
    ON pack038_peak_shaving.ps_tariff_structures
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_dc_tenant_isolation
    ON pack038_peak_shaving.ps_demand_charges
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_dc_service_bypass
    ON pack038_peak_shaving.ps_demand_charges
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_cc_tenant_isolation
    ON pack038_peak_shaving.ps_charge_components
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cc_service_bypass
    ON pack038_peak_shaving.ps_charge_components
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_mv_tenant_isolation
    ON pack038_peak_shaving.ps_marginal_values
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_mv_service_bypass
    ON pack038_peak_shaving.ps_marginal_values
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_tc_tenant_isolation
    ON pack038_peak_shaving.ps_tariff_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_tc_service_bypass
    ON pack038_peak_shaving.ps_tariff_comparisons
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_tariff_structures TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_demand_charges TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_charge_components TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_marginal_values TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_tariff_comparisons TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.ps_tariff_structures IS
    'Utility tariff rate schedule definitions with demand charge tiers, TOU periods, ratchet clauses, and power factor adjustments.';
COMMENT ON TABLE pack038_peak_shaving.ps_demand_charges IS
    'Calculated demand charges per billing period linking peak events to tariff structures with TOU, tiered, and ratchet components.';
COMMENT ON TABLE pack038_peak_shaving.ps_charge_components IS
    'Granular demand charge component breakdown per billing period for detailed bill analysis and savings attribution.';
COMMENT ON TABLE pack038_peak_shaving.ps_marginal_values IS
    'Marginal value of peak demand reduction at each kW level considering tiers, ratchets, and coincident peak charges.';
COMMENT ON TABLE pack038_peak_shaving.ps_tariff_comparisons IS
    'Side-by-side tariff comparison analysis for rate optimization and utility tariff selection recommendations.';

COMMENT ON COLUMN pack038_peak_shaving.ps_tariff_structures.tiers IS 'JSON array of demand charge tiers: [{tier: 1, lower_kw: 0, upper_kw: 500, rate_per_kw: 12.50}, ...].';
COMMENT ON COLUMN pack038_peak_shaving.ps_tariff_structures.tou_periods IS 'JSON array of TOU periods: [{name: "ON_PEAK", start_hour: 12, end_hour: 18, months: [6,7,8,9], days: "WEEKDAY"}, ...].';
COMMENT ON COLUMN pack038_peak_shaving.ps_tariff_structures.ratchet_clause IS 'JSON ratchet clause definition: {type: "PERCENTAGE", pct: 80, lookback_months: 11, billing_months: [6,7,8,9]}.';
COMMENT ON COLUMN pack038_peak_shaving.ps_tariff_structures.demand_window_minutes IS 'Demand averaging window in minutes (5, 15, 30, or 60). Most US utilities use 15-minute demand.';

COMMENT ON COLUMN pack038_peak_shaving.ps_demand_charges.billing_demand_kw IS 'The demand used for billing which may differ from actual peak due to ratchets, PF adjustments, or contract demand.';
COMMENT ON COLUMN pack038_peak_shaving.ps_demand_charges.ratchet_demand_kw IS 'Ratcheted demand from a prior billing period that sets a floor for current billing demand.';
COMMENT ON COLUMN pack038_peak_shaving.ps_demand_charges.demand_pct_of_bill IS 'Percentage of total bill attributable to demand charges (0-100).';
COMMENT ON COLUMN pack038_peak_shaving.ps_demand_charges.effective_rate_per_kw IS 'Effective blended demand rate per kW after all adjustments.';

COMMENT ON COLUMN pack038_peak_shaving.ps_marginal_values.ratchet_multiplier IS 'Ratchet duration multiplier (e.g., 11 months means each kW saved avoids 11 months of ratcheted charges).';
COMMENT ON COLUMN pack038_peak_shaving.ps_marginal_values.total_marginal_value IS 'Total marginal value of 1 kW demand reduction including all charge components and ratchet effects.';

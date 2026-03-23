-- =============================================================================
-- V293: PACK-037 Demand Response Pack - Revenue & Financial Analysis
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    008 of 010
-- Date:         March 2026
--
-- Revenue tracking, financial analysis, and economic optimization tables.
-- Covers revenue streams, forecasts, settlements, penalties, ROI analysis,
-- and what-if scenario modelling for DR program participation decisions.
--
-- Tables (6):
--   1. pack037_demand_response.dr_revenue_streams
--   2. pack037_demand_response.dr_revenue_forecasts
--   3. pack037_demand_response.dr_settlements
--   4. pack037_demand_response.dr_penalties
--   5. pack037_demand_response.dr_roi_analysis
--   6. pack037_demand_response.dr_what_if_scenarios
--
-- Previous: V292__pack037_demand_response_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.dr_revenue_streams
-- =============================================================================
-- Individual revenue line items from DR program participation. Each
-- stream represents a distinct payment type (capacity, energy,
-- availability, performance bonus) from a specific program.

CREATE TABLE pack037_demand_response.dr_revenue_streams (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_program_enrollment(id),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    stream_type             VARCHAR(50)     NOT NULL,
    program_code            VARCHAR(50)     NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    committed_kw            NUMERIC(12,4)   NOT NULL,
    delivered_kw            NUMERIC(12,4),
    rate_applied            NUMERIC(12,6)   NOT NULL,
    rate_unit               VARCHAR(30)     NOT NULL,
    base_amount             NUMERIC(14,2)   NOT NULL,
    multiplier_applied      NUMERIC(6,3)    DEFAULT 1.0,
    bonus_amount            NUMERIC(14,2)   DEFAULT 0,
    gross_amount            NUMERIC(14,2)   NOT NULL,
    tax_amount              NUMERIC(14,2)   DEFAULT 0,
    net_amount              NUMERIC(14,2)   NOT NULL,
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    payment_status          VARCHAR(20)     NOT NULL DEFAULT 'ACCRUED',
    invoice_number          VARCHAR(100),
    payment_date            DATE,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_rs_type CHECK (
        stream_type IN (
            'CAPACITY_PAYMENT', 'ENERGY_PAYMENT', 'AVAILABILITY_PAYMENT',
            'PERFORMANCE_BONUS', 'RESERVATION_FEE', 'ANCILLARY_PAYMENT',
            'FREQUENCY_REGULATION', 'DEMAND_CHARGE_SAVINGS',
            'AVOIDED_CAPACITY_CHARGE', 'INCENTIVE_REBATE'
        )
    ),
    CONSTRAINT chk_p037_rs_dates CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p037_rs_committed CHECK (
        committed_kw > 0
    ),
    CONSTRAINT chk_p037_rs_rate CHECK (
        rate_applied >= 0
    ),
    CONSTRAINT chk_p037_rs_base CHECK (
        base_amount >= 0
    ),
    CONSTRAINT chk_p037_rs_gross CHECK (
        gross_amount >= 0
    ),
    CONSTRAINT chk_p037_rs_net CHECK (
        net_amount >= 0
    ),
    CONSTRAINT chk_p037_rs_status CHECK (
        payment_status IN (
            'ACCRUED', 'INVOICED', 'PAID', 'PARTIAL', 'DISPUTED',
            'WRITTEN_OFF', 'REVERSED'
        )
    ),
    CONSTRAINT chk_p037_rs_unit CHECK (
        rate_unit IN (
            'USD_PER_KW_YEAR', 'USD_PER_KW_MONTH', 'USD_PER_KW_DAY',
            'USD_PER_KWH', 'USD_PER_MWH', 'USD_PER_MW_HOUR',
            'EUR_PER_KW_YEAR', 'EUR_PER_KW_MONTH', 'EUR_PER_KWH',
            'EUR_PER_MWH', 'GBP_PER_KW_YEAR', 'GBP_PER_MWH',
            'FLAT', 'PCT_OF_SAVINGS'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_rs_enrollment      ON pack037_demand_response.dr_revenue_streams(enrollment_id);
CREATE INDEX idx_p037_rs_facility        ON pack037_demand_response.dr_revenue_streams(facility_profile_id);
CREATE INDEX idx_p037_rs_tenant          ON pack037_demand_response.dr_revenue_streams(tenant_id);
CREATE INDEX idx_p037_rs_type            ON pack037_demand_response.dr_revenue_streams(stream_type);
CREATE INDEX idx_p037_rs_program         ON pack037_demand_response.dr_revenue_streams(program_code);
CREATE INDEX idx_p037_rs_period          ON pack037_demand_response.dr_revenue_streams(period_start DESC);
CREATE INDEX idx_p037_rs_status          ON pack037_demand_response.dr_revenue_streams(payment_status);
CREATE INDEX idx_p037_rs_net             ON pack037_demand_response.dr_revenue_streams(net_amount DESC);
CREATE INDEX idx_p037_rs_created         ON pack037_demand_response.dr_revenue_streams(created_at DESC);

-- Composite: facility + type + period for revenue reporting
CREATE INDEX idx_p037_rs_fac_type_period ON pack037_demand_response.dr_revenue_streams(facility_profile_id, stream_type, period_start DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_rs_updated
    BEFORE UPDATE ON pack037_demand_response.dr_revenue_streams
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack037_demand_response.dr_revenue_forecasts
-- =============================================================================
-- Revenue forecasts for upcoming DR seasons based on committed capacity,
-- expected events, market conditions, and historical performance.

CREATE TABLE pack037_demand_response.dr_revenue_forecasts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    enrollment_id           UUID            REFERENCES pack037_demand_response.dr_program_enrollment(id),
    tenant_id               UUID            NOT NULL,
    forecast_name           VARCHAR(255)    NOT NULL,
    forecast_type           VARCHAR(30)     NOT NULL,
    forecast_year           INTEGER         NOT NULL,
    forecast_season         VARCHAR(20)     NOT NULL,
    program_code            VARCHAR(50),
    committed_kw            NUMERIC(12,4)   NOT NULL,
    expected_events         INTEGER         NOT NULL,
    expected_hours          NUMERIC(10,2),
    avg_performance_assumption NUMERIC(6,4) NOT NULL DEFAULT 1.0,
    capacity_revenue_low    NUMERIC(14,2),
    capacity_revenue_mid    NUMERIC(14,2)   NOT NULL,
    capacity_revenue_high   NUMERIC(14,2),
    energy_revenue_low      NUMERIC(14,2),
    energy_revenue_mid      NUMERIC(14,2),
    energy_revenue_high     NUMERIC(14,2),
    total_revenue_low       NUMERIC(14,2)   NOT NULL,
    total_revenue_mid       NUMERIC(14,2)   NOT NULL,
    total_revenue_high      NUMERIC(14,2)   NOT NULL,
    expected_penalties      NUMERIC(14,2)   DEFAULT 0,
    net_revenue_forecast    NUMERIC(14,2)   NOT NULL,
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    confidence_level_pct    NUMERIC(5,2),
    assumptions             JSONB           NOT NULL DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_rf_type CHECK (
        forecast_type IN ('BUDGET', 'ESTIMATE', 'OPTIMISTIC', 'PESSIMISTIC', 'SCENARIO')
    ),
    CONSTRAINT chk_p037_rf_year CHECK (
        forecast_year >= 2020 AND forecast_year <= 2100
    ),
    CONSTRAINT chk_p037_rf_season CHECK (
        forecast_season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p037_rf_committed CHECK (
        committed_kw > 0
    ),
    CONSTRAINT chk_p037_rf_events CHECK (
        expected_events >= 0
    ),
    CONSTRAINT chk_p037_rf_perf CHECK (
        avg_performance_assumption > 0 AND avg_performance_assumption <= 2.0
    ),
    CONSTRAINT chk_p037_rf_confidence CHECK (
        confidence_level_pct IS NULL OR (confidence_level_pct >= 0 AND confidence_level_pct <= 100)
    ),
    CONSTRAINT chk_p037_rf_revenue_range CHECK (
        total_revenue_low <= total_revenue_mid AND total_revenue_mid <= total_revenue_high
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_rf_facility        ON pack037_demand_response.dr_revenue_forecasts(facility_profile_id);
CREATE INDEX idx_p037_rf_enrollment      ON pack037_demand_response.dr_revenue_forecasts(enrollment_id);
CREATE INDEX idx_p037_rf_tenant          ON pack037_demand_response.dr_revenue_forecasts(tenant_id);
CREATE INDEX idx_p037_rf_type            ON pack037_demand_response.dr_revenue_forecasts(forecast_type);
CREATE INDEX idx_p037_rf_year            ON pack037_demand_response.dr_revenue_forecasts(forecast_year, forecast_season);
CREATE INDEX idx_p037_rf_program         ON pack037_demand_response.dr_revenue_forecasts(program_code);
CREATE INDEX idx_p037_rf_net             ON pack037_demand_response.dr_revenue_forecasts(net_revenue_forecast DESC);
CREATE INDEX idx_p037_rf_created         ON pack037_demand_response.dr_revenue_forecasts(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_rf_updated
    BEFORE UPDATE ON pack037_demand_response.dr_revenue_forecasts
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack037_demand_response.dr_settlements
-- =============================================================================
-- Settlement records from ISOs/RTOs/aggregators for DR event participation.
-- Links events to financial settlements with reconciliation tracking.

CREATE TABLE pack037_demand_response.dr_settlements (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID            REFERENCES pack037_demand_response.dr_events(id),
    enrollment_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_program_enrollment(id),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    settlement_type         VARCHAR(30)     NOT NULL,
    settlement_period_start DATE            NOT NULL,
    settlement_period_end   DATE            NOT NULL,
    program_code            VARCHAR(50)     NOT NULL,
    settled_kw              NUMERIC(12,4)   NOT NULL,
    settled_mwh             NUMERIC(14,6),
    settlement_price        NUMERIC(12,6),
    price_unit              VARCHAR(30),
    gross_payment           NUMERIC(14,2)   NOT NULL,
    adjustments             NUMERIC(14,2)   DEFAULT 0,
    penalties               NUMERIC(14,2)   DEFAULT 0,
    net_payment             NUMERIC(14,2)   NOT NULL,
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    settlement_status       VARCHAR(20)     NOT NULL DEFAULT 'PRELIMINARY',
    iso_settlement_id       VARCHAR(100),
    invoice_reference       VARCHAR(100),
    settled_at              TIMESTAMPTZ,
    paid_at                 TIMESTAMPTZ,
    dispute_flag            BOOLEAN         DEFAULT false,
    dispute_reason          TEXT,
    dispute_resolution      TEXT,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_set_type CHECK (
        settlement_type IN (
            'CAPACITY', 'ENERGY', 'PERFORMANCE', 'AVAILABILITY',
            'PENALTY', 'RECONCILIATION', 'TRUE_UP', 'FINAL'
        )
    ),
    CONSTRAINT chk_p037_set_dates CHECK (
        settlement_period_end >= settlement_period_start
    ),
    CONSTRAINT chk_p037_set_settled_kw CHECK (
        settled_kw >= 0
    ),
    CONSTRAINT chk_p037_set_status CHECK (
        settlement_status IN (
            'PRELIMINARY', 'INITIAL', 'REVISED', 'FINAL', 'DISPUTED',
            'RESOLVED', 'PAID', 'WRITTEN_OFF'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_set_event          ON pack037_demand_response.dr_settlements(event_id);
CREATE INDEX idx_p037_set_enrollment     ON pack037_demand_response.dr_settlements(enrollment_id);
CREATE INDEX idx_p037_set_facility       ON pack037_demand_response.dr_settlements(facility_profile_id);
CREATE INDEX idx_p037_set_tenant         ON pack037_demand_response.dr_settlements(tenant_id);
CREATE INDEX idx_p037_set_type           ON pack037_demand_response.dr_settlements(settlement_type);
CREATE INDEX idx_p037_set_program        ON pack037_demand_response.dr_settlements(program_code);
CREATE INDEX idx_p037_set_status         ON pack037_demand_response.dr_settlements(settlement_status);
CREATE INDEX idx_p037_set_period         ON pack037_demand_response.dr_settlements(settlement_period_start DESC);
CREATE INDEX idx_p037_set_net            ON pack037_demand_response.dr_settlements(net_payment DESC);
CREATE INDEX idx_p037_set_dispute        ON pack037_demand_response.dr_settlements(dispute_flag);
CREATE INDEX idx_p037_set_created        ON pack037_demand_response.dr_settlements(created_at DESC);

-- Composite: disputed settlements for resolution tracking
CREATE INDEX idx_p037_set_disputed       ON pack037_demand_response.dr_settlements(settlement_period_start DESC)
    WHERE dispute_flag = true AND settlement_status = 'DISPUTED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_set_updated
    BEFORE UPDATE ON pack037_demand_response.dr_settlements
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack037_demand_response.dr_penalties
-- =============================================================================
-- Penalty records assessed for non-performance, under-delivery, or
-- non-compliance during DR events. Links to settlement for financial
-- reconciliation.

CREATE TABLE pack037_demand_response.dr_penalties (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID            REFERENCES pack037_demand_response.dr_events(id),
    enrollment_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_program_enrollment(id),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    penalty_type            VARCHAR(50)     NOT NULL,
    program_code            VARCHAR(50)     NOT NULL,
    penalty_date            DATE            NOT NULL,
    committed_kw            NUMERIC(12,4)   NOT NULL,
    delivered_kw            NUMERIC(12,4),
    shortfall_kw            NUMERIC(12,4)   NOT NULL,
    penalty_rate            NUMERIC(12,4)   NOT NULL,
    penalty_unit            VARCHAR(30)     NOT NULL,
    base_penalty            NUMERIC(14,2)   NOT NULL,
    escalation_multiplier   NUMERIC(6,3)    DEFAULT 1.0,
    total_penalty           NUMERIC(14,2)   NOT NULL,
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    penalty_status          VARCHAR(20)     NOT NULL DEFAULT 'ASSESSED',
    waiver_requested        BOOLEAN         DEFAULT false,
    waiver_granted          BOOLEAN,
    waiver_reason           TEXT,
    settlement_id           UUID            REFERENCES pack037_demand_response.dr_settlements(id),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_pen2_type CHECK (
        penalty_type IN (
            'UNDER_DELIVERY', 'NON_RESPONSE', 'NON_COMPLIANCE',
            'TELEMETRY_FAILURE', 'LATE_NOTIFICATION', 'TEST_FAILURE',
            'CAPACITY_DEFICIENCY', 'AVAILABILITY_SHORTFALL'
        )
    ),
    CONSTRAINT chk_p037_pen2_shortfall CHECK (
        shortfall_kw >= 0
    ),
    CONSTRAINT chk_p037_pen2_rate CHECK (
        penalty_rate >= 0
    ),
    CONSTRAINT chk_p037_pen2_total CHECK (
        total_penalty >= 0
    ),
    CONSTRAINT chk_p037_pen2_status CHECK (
        penalty_status IN (
            'ASSESSED', 'DISPUTED', 'WAIVED', 'REDUCED', 'FINAL', 'PAID'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_pen2_event         ON pack037_demand_response.dr_penalties(event_id);
CREATE INDEX idx_p037_pen2_enrollment    ON pack037_demand_response.dr_penalties(enrollment_id);
CREATE INDEX idx_p037_pen2_facility      ON pack037_demand_response.dr_penalties(facility_profile_id);
CREATE INDEX idx_p037_pen2_tenant        ON pack037_demand_response.dr_penalties(tenant_id);
CREATE INDEX idx_p037_pen2_type          ON pack037_demand_response.dr_penalties(penalty_type);
CREATE INDEX idx_p037_pen2_date          ON pack037_demand_response.dr_penalties(penalty_date DESC);
CREATE INDEX idx_p037_pen2_status        ON pack037_demand_response.dr_penalties(penalty_status);
CREATE INDEX idx_p037_pen2_total         ON pack037_demand_response.dr_penalties(total_penalty DESC);
CREATE INDEX idx_p037_pen2_settlement    ON pack037_demand_response.dr_penalties(settlement_id);
CREATE INDEX idx_p037_pen2_created       ON pack037_demand_response.dr_penalties(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_pen2_updated
    BEFORE UPDATE ON pack037_demand_response.dr_penalties
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack037_demand_response.dr_roi_analysis
-- =============================================================================
-- Return on investment analysis for DR program participation comparing
-- revenue, costs (implementation, operational, opportunity), and net
-- benefit over time.

CREATE TABLE pack037_demand_response.dr_roi_analysis (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    enrollment_id           UUID            REFERENCES pack037_demand_response.dr_program_enrollment(id),
    tenant_id               UUID            NOT NULL,
    analysis_name           VARCHAR(255)    NOT NULL,
    analysis_date           DATE            NOT NULL,
    analysis_period_years   INTEGER         NOT NULL DEFAULT 1,
    program_code            VARCHAR(50),
    -- Revenue
    total_revenue           NUMERIC(14,2)   NOT NULL,
    capacity_revenue        NUMERIC(14,2)   DEFAULT 0,
    energy_revenue          NUMERIC(14,2)   DEFAULT 0,
    demand_charge_savings   NUMERIC(14,2)   DEFAULT 0,
    other_revenue           NUMERIC(14,2)   DEFAULT 0,
    -- Costs
    total_costs             NUMERIC(14,2)   NOT NULL,
    implementation_cost     NUMERIC(14,2)   DEFAULT 0,
    metering_cost           NUMERIC(14,2)   DEFAULT 0,
    automation_cost         NUMERIC(14,2)   DEFAULT 0,
    operational_cost        NUMERIC(14,2)   DEFAULT 0,
    comfort_cost            NUMERIC(14,2)   DEFAULT 0,
    production_loss_cost    NUMERIC(14,2)   DEFAULT 0,
    penalty_cost            NUMERIC(14,2)   DEFAULT 0,
    aggregator_fee          NUMERIC(14,2)   DEFAULT 0,
    -- Analysis
    net_benefit             NUMERIC(14,2)   NOT NULL,
    roi_pct                 NUMERIC(8,2),
    payback_months          NUMERIC(8,2),
    benefit_cost_ratio      NUMERIC(8,4),
    npv                     NUMERIC(14,2),
    irr_pct                 NUMERIC(8,4),
    discount_rate_pct       NUMERIC(6,4)    DEFAULT 8.0,
    levelized_cost_per_kw   NUMERIC(10,4),
    revenue_per_kw_year     NUMERIC(10,4),
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_roi_period CHECK (
        analysis_period_years >= 1 AND analysis_period_years <= 30
    ),
    CONSTRAINT chk_p037_roi_revenue CHECK (
        total_revenue >= 0
    ),
    CONSTRAINT chk_p037_roi_costs CHECK (
        total_costs >= 0
    ),
    CONSTRAINT chk_p037_roi_discount CHECK (
        discount_rate_pct IS NULL OR (discount_rate_pct >= 0 AND discount_rate_pct <= 50)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_roi_facility       ON pack037_demand_response.dr_roi_analysis(facility_profile_id);
CREATE INDEX idx_p037_roi_enrollment     ON pack037_demand_response.dr_roi_analysis(enrollment_id);
CREATE INDEX idx_p037_roi_tenant         ON pack037_demand_response.dr_roi_analysis(tenant_id);
CREATE INDEX idx_p037_roi_date           ON pack037_demand_response.dr_roi_analysis(analysis_date DESC);
CREATE INDEX idx_p037_roi_program        ON pack037_demand_response.dr_roi_analysis(program_code);
CREATE INDEX idx_p037_roi_roi            ON pack037_demand_response.dr_roi_analysis(roi_pct DESC);
CREATE INDEX idx_p037_roi_net            ON pack037_demand_response.dr_roi_analysis(net_benefit DESC);
CREATE INDEX idx_p037_roi_created        ON pack037_demand_response.dr_roi_analysis(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_roi_updated
    BEFORE UPDATE ON pack037_demand_response.dr_roi_analysis
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 6: pack037_demand_response.dr_what_if_scenarios
-- =============================================================================
-- What-if scenario modelling for DR program participation decisions.
-- Evaluates different capacity commitments, program combinations,
-- and market conditions to optimize portfolio revenue.

CREATE TABLE pack037_demand_response.dr_what_if_scenarios (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    scenario_name           VARCHAR(255)    NOT NULL,
    scenario_description    TEXT,
    scenario_type           VARCHAR(30)     NOT NULL,
    base_scenario_id        UUID            REFERENCES pack037_demand_response.dr_what_if_scenarios(id),
    -- Scenario parameters
    target_year             INTEGER         NOT NULL,
    programs_considered     JSONB           NOT NULL DEFAULT '[]',
    committed_capacity_kw   NUMERIC(12,4)   NOT NULL,
    assumed_events          INTEGER,
    assumed_performance     NUMERIC(6,4)    DEFAULT 1.0,
    market_price_assumption VARCHAR(30),
    automation_investment   NUMERIC(14,2)   DEFAULT 0,
    -- Scenario results
    projected_revenue       NUMERIC(14,2),
    projected_costs         NUMERIC(14,2),
    projected_net_benefit   NUMERIC(14,2),
    projected_roi_pct       NUMERIC(8,2),
    risk_score              NUMERIC(5,2),
    recommendation          TEXT,
    is_selected             BOOLEAN         DEFAULT false,
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p037_wis_type CHECK (
        scenario_type IN (
            'BASELINE', 'OPTIMISTIC', 'PESSIMISTIC', 'GROWTH',
            'NEW_PROGRAM', 'PROGRAM_EXIT', 'CAPACITY_INCREASE',
            'AUTOMATION_UPGRADE', 'DER_ADDITION', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p037_wis_year CHECK (
        target_year >= 2020 AND target_year <= 2100
    ),
    CONSTRAINT chk_p037_wis_capacity CHECK (
        committed_capacity_kw > 0
    ),
    CONSTRAINT chk_p037_wis_performance CHECK (
        assumed_performance > 0 AND assumed_performance <= 2.0
    ),
    CONSTRAINT chk_p037_wis_market CHECK (
        market_price_assumption IS NULL OR market_price_assumption IN (
            'CURRENT', 'LOW', 'MEDIUM', 'HIGH', 'HISTORICAL_AVG', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p037_wis_risk CHECK (
        risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_wis_facility       ON pack037_demand_response.dr_what_if_scenarios(facility_profile_id);
CREATE INDEX idx_p037_wis_tenant         ON pack037_demand_response.dr_what_if_scenarios(tenant_id);
CREATE INDEX idx_p037_wis_type           ON pack037_demand_response.dr_what_if_scenarios(scenario_type);
CREATE INDEX idx_p037_wis_year           ON pack037_demand_response.dr_what_if_scenarios(target_year);
CREATE INDEX idx_p037_wis_base           ON pack037_demand_response.dr_what_if_scenarios(base_scenario_id);
CREATE INDEX idx_p037_wis_selected       ON pack037_demand_response.dr_what_if_scenarios(is_selected);
CREATE INDEX idx_p037_wis_roi            ON pack037_demand_response.dr_what_if_scenarios(projected_roi_pct DESC);
CREATE INDEX idx_p037_wis_created        ON pack037_demand_response.dr_what_if_scenarios(created_at DESC);
CREATE INDEX idx_p037_wis_programs       ON pack037_demand_response.dr_what_if_scenarios USING GIN(programs_considered);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_wis_updated
    BEFORE UPDATE ON pack037_demand_response.dr_what_if_scenarios
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_revenue_streams ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_revenue_forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_settlements ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_penalties ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_roi_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_what_if_scenarios ENABLE ROW LEVEL SECURITY;

CREATE POLICY p037_rs_tenant_isolation ON pack037_demand_response.dr_revenue_streams
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_rs_service_bypass ON pack037_demand_response.dr_revenue_streams
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_rf_tenant_isolation ON pack037_demand_response.dr_revenue_forecasts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_rf_service_bypass ON pack037_demand_response.dr_revenue_forecasts
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_set_tenant_isolation ON pack037_demand_response.dr_settlements
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_set_service_bypass ON pack037_demand_response.dr_settlements
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_pen2_tenant_isolation ON pack037_demand_response.dr_penalties
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_pen2_service_bypass ON pack037_demand_response.dr_penalties
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_roi_tenant_isolation ON pack037_demand_response.dr_roi_analysis
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_roi_service_bypass ON pack037_demand_response.dr_roi_analysis
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_wis_tenant_isolation ON pack037_demand_response.dr_what_if_scenarios
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_wis_service_bypass ON pack037_demand_response.dr_what_if_scenarios
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_revenue_streams TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_revenue_forecasts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_settlements TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_penalties TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_roi_analysis TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_what_if_scenarios TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.dr_revenue_streams IS
    'Individual revenue line items from DR program participation by payment type (capacity, energy, availability, bonus).';
COMMENT ON TABLE pack037_demand_response.dr_revenue_forecasts IS
    'Revenue forecasts for upcoming DR seasons based on committed capacity, expected events, and market conditions.';
COMMENT ON TABLE pack037_demand_response.dr_settlements IS
    'Settlement records from ISOs/RTOs/aggregators for DR event participation with reconciliation tracking.';
COMMENT ON TABLE pack037_demand_response.dr_penalties IS
    'Penalty records assessed for non-performance or under-delivery with waiver and dispute management.';
COMMENT ON TABLE pack037_demand_response.dr_roi_analysis IS
    'ROI analysis comparing revenue, costs (implementation, operational, opportunity), and net benefit over time.';
COMMENT ON TABLE pack037_demand_response.dr_what_if_scenarios IS
    'What-if scenario modelling for DR program participation decisions optimizing capacity commitments and program mix.';

COMMENT ON COLUMN pack037_demand_response.dr_revenue_streams.stream_type IS 'Revenue type: CAPACITY_PAYMENT, ENERGY_PAYMENT, AVAILABILITY_PAYMENT, DEMAND_CHARGE_SAVINGS, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_revenue_streams.payment_status IS 'Payment status: ACCRUED, INVOICED, PAID, PARTIAL, DISPUTED, WRITTEN_OFF, REVERSED.';
COMMENT ON COLUMN pack037_demand_response.dr_revenue_streams.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack037_demand_response.dr_revenue_forecasts.forecast_type IS 'Forecast scenario: BUDGET, ESTIMATE, OPTIMISTIC, PESSIMISTIC, SCENARIO.';
COMMENT ON COLUMN pack037_demand_response.dr_revenue_forecasts.total_revenue_low IS 'Low-case (P10) total revenue forecast.';
COMMENT ON COLUMN pack037_demand_response.dr_revenue_forecasts.total_revenue_high IS 'High-case (P90) total revenue forecast.';

COMMENT ON COLUMN pack037_demand_response.dr_settlements.settlement_status IS 'Settlement lifecycle: PRELIMINARY, INITIAL, REVISED, FINAL, DISPUTED, RESOLVED, PAID, WRITTEN_OFF.';

COMMENT ON COLUMN pack037_demand_response.dr_roi_analysis.roi_pct IS 'Return on investment percentage ((net_benefit / total_costs) * 100).';
COMMENT ON COLUMN pack037_demand_response.dr_roi_analysis.npv IS 'Net Present Value of DR participation cash flows.';
COMMENT ON COLUMN pack037_demand_response.dr_roi_analysis.irr_pct IS 'Internal Rate of Return for DR participation investment.';
COMMENT ON COLUMN pack037_demand_response.dr_roi_analysis.levelized_cost_per_kw IS 'Total cost per kW of curtailment capacity over the analysis period.';

COMMENT ON COLUMN pack037_demand_response.dr_what_if_scenarios.scenario_type IS 'Scenario category: BASELINE, OPTIMISTIC, GROWTH, NEW_PROGRAM, DER_ADDITION, CUSTOM, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_what_if_scenarios.risk_score IS 'Quantified risk score 0-100 (higher = more risk) based on market volatility and performance uncertainty.';

-- =============================================================================
-- V319: PACK-040 M&V Pack - Savings Calculations
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for energy savings calculation including reporting periods,
-- avoided energy use, cost savings, cumulative savings tracking, and savings
-- summaries. Implements the core IPMVP savings equation:
-- Savings = Adjusted Baseline - Reporting Period Actual +/- Non-Routine Adj.
--
-- Tables (5):
--   1. pack040_mv.mv_savings_periods
--   2. pack040_mv.mv_avoided_energy
--   3. pack040_mv.mv_cost_savings
--   4. pack040_mv.mv_cumulative_savings
--   5. pack040_mv.mv_savings_summaries
--
-- Previous: V318__pack040_mv_003.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.mv_savings_periods
-- =============================================================================
-- Defines reporting periods for savings calculation. Each period represents
-- a time interval (month, quarter, year) over which savings are calculated
-- by comparing adjusted baseline predictions to actual consumption.

CREATE TABLE pack040_mv.mv_savings_periods (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    baseline_id                 UUID            NOT NULL REFERENCES pack040_mv.mv_baselines(id) ON DELETE CASCADE,
    period_name                 VARCHAR(255)    NOT NULL,
    period_type                 VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    period_number               INTEGER         NOT NULL DEFAULT 1,
    reporting_year              INTEGER         NOT NULL,
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    num_days                    INTEGER         NOT NULL,
    -- Energy values
    adjusted_baseline_kwh       NUMERIC(18,3)   NOT NULL,
    actual_consumption_kwh      NUMERIC(18,3)   NOT NULL,
    routine_adjustment_kwh      NUMERIC(18,3)   NOT NULL DEFAULT 0,
    nonroutine_adjustment_kwh   NUMERIC(18,3)   NOT NULL DEFAULT 0,
    -- Weather data
    period_hdd                  NUMERIC(10,3),
    period_cdd                  NUMERIC(10,3),
    avg_temperature_f           NUMERIC(8,3),
    -- Production data
    period_production           NUMERIC(18,3),
    production_unit             VARCHAR(50),
    -- Savings
    avoided_energy_kwh          NUMERIC(18,3)   NOT NULL,
    normalized_savings_kwh      NUMERIC(18,3),
    savings_pct                 NUMERIC(8,4),
    -- Demand
    baseline_demand_kw          NUMERIC(12,3),
    actual_demand_kw            NUMERIC(12,3),
    demand_savings_kw           NUMERIC(12,3),
    -- Data quality
    data_completeness_pct       NUMERIC(5,2),
    num_estimated_intervals     INTEGER         DEFAULT 0,
    num_missing_intervals       INTEGER         DEFAULT 0,
    -- Status
    period_status               VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_sp_period_type CHECK (
        period_type IN (
            'MONTHLY', 'QUARTERLY', 'SEMI_ANNUALLY', 'ANNUALLY',
            'BILLING_PERIOD', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_sp_dates CHECK (
        period_start < period_end
    ),
    CONSTRAINT chk_p040_sp_num_days CHECK (
        num_days >= 1 AND num_days <= 366
    ),
    CONSTRAINT chk_p040_sp_period_num CHECK (
        period_number >= 1 AND period_number <= 12
    ),
    CONSTRAINT chk_p040_sp_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p040_sp_actual CHECK (
        actual_consumption_kwh >= 0
    ),
    CONSTRAINT chk_p040_sp_baseline CHECK (
        adjusted_baseline_kwh >= 0
    ),
    CONSTRAINT chk_p040_sp_completeness CHECK (
        data_completeness_pct IS NULL OR
        (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p040_sp_status CHECK (
        period_status IN (
            'CALCULATED', 'REVIEWED', 'APPROVED', 'REVISED',
            'PRELIMINARY', 'FINAL', 'DISPUTED'
        )
    ),
    CONSTRAINT chk_p040_sp_estimated CHECK (
        num_estimated_intervals IS NULL OR num_estimated_intervals >= 0
    ),
    CONSTRAINT chk_p040_sp_missing CHECK (
        num_missing_intervals IS NULL OR num_missing_intervals >= 0
    ),
    CONSTRAINT uq_p040_sp_project_period UNIQUE (project_id, baseline_id, period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_sp_tenant            ON pack040_mv.mv_savings_periods(tenant_id);
CREATE INDEX idx_p040_sp_project           ON pack040_mv.mv_savings_periods(project_id);
CREATE INDEX idx_p040_sp_baseline          ON pack040_mv.mv_savings_periods(baseline_id);
CREATE INDEX idx_p040_sp_type              ON pack040_mv.mv_savings_periods(period_type);
CREATE INDEX idx_p040_sp_year              ON pack040_mv.mv_savings_periods(reporting_year);
CREATE INDEX idx_p040_sp_period            ON pack040_mv.mv_savings_periods(period_start, period_end);
CREATE INDEX idx_p040_sp_status            ON pack040_mv.mv_savings_periods(period_status);
CREATE INDEX idx_p040_sp_created           ON pack040_mv.mv_savings_periods(created_at DESC);

-- Composite: project + approved periods for reporting
CREATE INDEX idx_p040_sp_project_approved  ON pack040_mv.mv_savings_periods(project_id, period_start DESC)
    WHERE period_status IN ('APPROVED', 'FINAL');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_sp_updated
    BEFORE UPDATE ON pack040_mv.mv_savings_periods
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_avoided_energy
-- =============================================================================
-- Detailed avoided energy calculations per ECM per reporting period. While
-- savings_periods stores project-level totals, this table tracks ECM-level
-- avoided energy for projects with multiple measures, supporting Option A/B
-- retrofit isolation calculations.

CREATE TABLE pack040_mv.mv_avoided_energy (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    savings_period_id           UUID            NOT NULL REFERENCES pack040_mv.mv_savings_periods(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    ipmvp_option                VARCHAR(10)     NOT NULL DEFAULT 'OPTION_C',
    -- Energy values
    baseline_energy_kwh         NUMERIC(18,3)   NOT NULL,
    adjusted_baseline_kwh       NUMERIC(18,3)   NOT NULL,
    reporting_energy_kwh        NUMERIC(18,3)   NOT NULL,
    avoided_energy_kwh          NUMERIC(18,3)   NOT NULL,
    normalized_savings_kwh      NUMERIC(18,3),
    -- For Option A: stipulated values
    stipulated_value_kwh        NUMERIC(18,3),
    measured_parameter_value    NUMERIC(18,6),
    measured_parameter_unit     VARCHAR(50),
    stipulated_parameter_name   VARCHAR(100),
    stipulated_parameter_value  NUMERIC(18,6),
    -- Percentages
    savings_pct_of_baseline     NUMERIC(8,4),
    savings_pct_of_guaranteed   NUMERIC(8,4),
    ecm_contribution_pct        NUMERIC(8,4),
    -- Interactive effects
    interactive_adjustment_kwh  NUMERIC(18,3)   DEFAULT 0,
    net_avoided_energy_kwh      NUMERIC(18,3),
    -- Uncertainty
    savings_uncertainty_kwh     NUMERIC(18,3),
    savings_uncertainty_pct     NUMERIC(8,4),
    confidence_level_pct        NUMERIC(5,2)    DEFAULT 90.0,
    -- Status
    calculation_method          VARCHAR(50)     NOT NULL DEFAULT 'ADJUSTED_BASELINE',
    verification_status         VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_ae_option CHECK (
        ipmvp_option IN ('OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D')
    ),
    CONSTRAINT chk_p040_ae_calc_method CHECK (
        calculation_method IN (
            'ADJUSTED_BASELINE', 'REGRESSION_PREDICT', 'KEY_PARAMETER',
            'ALL_PARAMETER', 'SIMULATION', 'STIPULATED', 'ENGINEERING_ESTIMATE'
        )
    ),
    CONSTRAINT chk_p040_ae_verif_status CHECK (
        verification_status IN (
            'CALCULATED', 'REVIEWED', 'VERIFIED', 'DISPUTED', 'REVISED'
        )
    ),
    CONSTRAINT chk_p040_ae_baseline CHECK (
        baseline_energy_kwh >= 0
    ),
    CONSTRAINT chk_p040_ae_reporting CHECK (
        reporting_energy_kwh >= 0
    ),
    CONSTRAINT chk_p040_ae_savings_pct CHECK (
        savings_pct_of_baseline IS NULL OR
        (savings_pct_of_baseline >= -100 AND savings_pct_of_baseline <= 100)
    ),
    CONSTRAINT chk_p040_ae_contribution CHECK (
        ecm_contribution_pct IS NULL OR
        (ecm_contribution_pct >= 0 AND ecm_contribution_pct <= 100)
    ),
    CONSTRAINT chk_p040_ae_uncertainty CHECK (
        savings_uncertainty_pct IS NULL OR savings_uncertainty_pct >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_ae_tenant            ON pack040_mv.mv_avoided_energy(tenant_id);
CREATE INDEX idx_p040_ae_project           ON pack040_mv.mv_avoided_energy(project_id);
CREATE INDEX idx_p040_ae_period            ON pack040_mv.mv_avoided_energy(savings_period_id);
CREATE INDEX idx_p040_ae_ecm               ON pack040_mv.mv_avoided_energy(ecm_id);
CREATE INDEX idx_p040_ae_option            ON pack040_mv.mv_avoided_energy(ipmvp_option);
CREATE INDEX idx_p040_ae_verif             ON pack040_mv.mv_avoided_energy(verification_status);
CREATE INDEX idx_p040_ae_created           ON pack040_mv.mv_avoided_energy(created_at DESC);

-- Composite: project + ECM verified savings
CREATE INDEX idx_p040_ae_project_ecm_ver   ON pack040_mv.mv_avoided_energy(project_id, ecm_id)
    WHERE verification_status = 'VERIFIED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_ae_updated
    BEFORE UPDATE ON pack040_mv.mv_avoided_energy
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_cost_savings
-- =============================================================================
-- Energy cost savings calculations based on avoided energy and utility rate
-- structures. Includes energy charges, demand charges, and any applicable
-- incentive payments or penalty avoidance values.

CREATE TABLE pack040_mv.mv_cost_savings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    savings_period_id           UUID            NOT NULL REFERENCES pack040_mv.mv_savings_periods(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    currency_code               VARCHAR(3)      NOT NULL DEFAULT 'USD',
    -- Energy cost savings
    avoided_energy_kwh          NUMERIC(18,3)   NOT NULL,
    energy_rate_per_kwh         NUMERIC(12,6),
    blended_rate_per_kwh        NUMERIC(12,6),
    energy_charge_savings       NUMERIC(18,2)   NOT NULL DEFAULT 0,
    -- Demand cost savings
    demand_savings_kw           NUMERIC(12,3),
    demand_rate_per_kw          NUMERIC(12,4),
    demand_charge_savings       NUMERIC(18,2)   NOT NULL DEFAULT 0,
    -- Other savings
    reactive_power_savings      NUMERIC(18,2)   DEFAULT 0,
    power_factor_penalty_avoided NUMERIC(18,2)  DEFAULT 0,
    maintenance_cost_savings    NUMERIC(18,2)   DEFAULT 0,
    water_cost_savings          NUMERIC(18,2)   DEFAULT 0,
    other_cost_savings          NUMERIC(18,2)   DEFAULT 0,
    -- Incentives
    utility_incentive           NUMERIC(18,2)   DEFAULT 0,
    tax_incentive               NUMERIC(18,2)   DEFAULT 0,
    carbon_credit_value         NUMERIC(18,2)   DEFAULT 0,
    -- Totals
    total_cost_savings          NUMERIC(18,2)   NOT NULL,
    total_cost_savings_annualized NUMERIC(18,2),
    -- Emissions
    avoided_co2_kg              NUMERIC(18,3),
    emission_factor_kg_per_kwh  NUMERIC(10,6),
    -- Rate source
    rate_schedule_name          VARCHAR(255),
    rate_source                 VARCHAR(50)     NOT NULL DEFAULT 'UTILITY_BILL',
    rate_effective_date         DATE,
    -- Status
    calculation_status          VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_cs_rate_source CHECK (
        rate_source IN (
            'UTILITY_BILL', 'RATE_SCHEDULE', 'BLENDED_AVERAGE',
            'CONTRACT_RATE', 'MARKET_RATE', 'ESTIMATED', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p040_cs_status CHECK (
        calculation_status IN (
            'CALCULATED', 'REVIEWED', 'APPROVED', 'REVISED', 'ESTIMATED'
        )
    ),
    CONSTRAINT chk_p040_cs_energy_rate CHECK (
        energy_rate_per_kwh IS NULL OR energy_rate_per_kwh >= 0
    ),
    CONSTRAINT chk_p040_cs_demand_rate CHECK (
        demand_rate_per_kw IS NULL OR demand_rate_per_kw >= 0
    ),
    CONSTRAINT chk_p040_cs_emission CHECK (
        emission_factor_kg_per_kwh IS NULL OR emission_factor_kg_per_kwh >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_cs_tenant            ON pack040_mv.mv_cost_savings(tenant_id);
CREATE INDEX idx_p040_cs_project           ON pack040_mv.mv_cost_savings(project_id);
CREATE INDEX idx_p040_cs_period            ON pack040_mv.mv_cost_savings(savings_period_id);
CREATE INDEX idx_p040_cs_ecm               ON pack040_mv.mv_cost_savings(ecm_id);
CREATE INDEX idx_p040_cs_status            ON pack040_mv.mv_cost_savings(calculation_status);
CREATE INDEX idx_p040_cs_created           ON pack040_mv.mv_cost_savings(created_at DESC);

-- Composite: project + approved cost savings
CREATE INDEX idx_p040_cs_project_approved  ON pack040_mv.mv_cost_savings(project_id, savings_period_id)
    WHERE calculation_status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_cs_updated
    BEFORE UPDATE ON pack040_mv.mv_cost_savings
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_cumulative_savings
-- =============================================================================
-- Running cumulative savings totals across all reporting periods for multi-year
-- tracking. Stores year-to-date and life-to-date accumulated energy and cost
-- savings for performance contract verification and executive reporting.

CREATE TABLE pack040_mv.mv_cumulative_savings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    as_of_date                  DATE            NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    -- Year-to-date energy
    ytd_avoided_energy_kwh      NUMERIC(18,3)   NOT NULL DEFAULT 0,
    ytd_normalized_savings_kwh  NUMERIC(18,3)   DEFAULT 0,
    ytd_periods_count           INTEGER         NOT NULL DEFAULT 0,
    -- Year-to-date cost
    ytd_energy_cost_savings     NUMERIC(18,2)   NOT NULL DEFAULT 0,
    ytd_demand_cost_savings     NUMERIC(18,2)   DEFAULT 0,
    ytd_total_cost_savings      NUMERIC(18,2)   NOT NULL DEFAULT 0,
    -- Year-to-date emissions
    ytd_avoided_co2_kg          NUMERIC(18,3)   DEFAULT 0,
    -- Life-to-date energy
    ltd_avoided_energy_kwh      NUMERIC(22,3)   NOT NULL DEFAULT 0,
    ltd_normalized_savings_kwh  NUMERIC(22,3)   DEFAULT 0,
    ltd_periods_count           INTEGER         NOT NULL DEFAULT 0,
    -- Life-to-date cost
    ltd_energy_cost_savings     NUMERIC(22,2)   NOT NULL DEFAULT 0,
    ltd_demand_cost_savings     NUMERIC(22,2)   DEFAULT 0,
    ltd_total_cost_savings      NUMERIC(22,2)   NOT NULL DEFAULT 0,
    -- Life-to-date emissions
    ltd_avoided_co2_kg          NUMERIC(22,3)   DEFAULT 0,
    -- Guaranteed savings tracking
    guaranteed_savings_kwh      NUMERIC(18,3),
    guaranteed_pct_achieved     NUMERIC(8,4),
    guaranteed_cost_savings     NUMERIC(18,2),
    guaranteed_cost_pct_achieved NUMERIC(8,4),
    shortfall_kwh               NUMERIC(18,3),
    shortfall_cost              NUMERIC(18,2),
    -- ROI
    cumulative_implementation_cost NUMERIC(18,2),
    simple_payback_achieved     BOOLEAN         NOT NULL DEFAULT false,
    payback_date                DATE,
    roi_pct                     NUMERIC(8,4),
    currency_code               VARCHAR(3)      NOT NULL DEFAULT 'USD',
    -- Status
    summary_status              VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_cum_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p040_cum_ytd_count CHECK (
        ytd_periods_count >= 0
    ),
    CONSTRAINT chk_p040_cum_ltd_count CHECK (
        ltd_periods_count >= 0
    ),
    CONSTRAINT chk_p040_cum_guaranteed_pct CHECK (
        guaranteed_pct_achieved IS NULL OR guaranteed_pct_achieved >= 0
    ),
    CONSTRAINT chk_p040_cum_roi CHECK (
        roi_pct IS NULL OR roi_pct >= -100
    ),
    CONSTRAINT chk_p040_cum_status CHECK (
        summary_status IN (
            'CALCULATED', 'REVIEWED', 'APPROVED', 'REVISED', 'FINAL'
        )
    ),
    CONSTRAINT uq_p040_cum_project_date UNIQUE (project_id, as_of_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_cum_tenant           ON pack040_mv.mv_cumulative_savings(tenant_id);
CREATE INDEX idx_p040_cum_project          ON pack040_mv.mv_cumulative_savings(project_id);
CREATE INDEX idx_p040_cum_date             ON pack040_mv.mv_cumulative_savings(as_of_date DESC);
CREATE INDEX idx_p040_cum_year             ON pack040_mv.mv_cumulative_savings(reporting_year);
CREATE INDEX idx_p040_cum_status           ON pack040_mv.mv_cumulative_savings(summary_status);
CREATE INDEX idx_p040_cum_payback          ON pack040_mv.mv_cumulative_savings(simple_payback_achieved) WHERE simple_payback_achieved = true;
CREATE INDEX idx_p040_cum_created          ON pack040_mv.mv_cumulative_savings(created_at DESC);

-- Composite: project + latest cumulative
CREATE INDEX idx_p040_cum_project_latest   ON pack040_mv.mv_cumulative_savings(project_id, as_of_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_cum_updated
    BEFORE UPDATE ON pack040_mv.mv_cumulative_savings
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_savings_summaries
-- =============================================================================
-- High-level savings summaries aggregating all ECMs for a project across
-- specified time ranges. Used for executive dashboards, performance contract
-- reporting, and annual M&V summary reports.

CREATE TABLE pack040_mv.mv_savings_summaries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    summary_name                VARCHAR(255)    NOT NULL,
    summary_type                VARCHAR(30)     NOT NULL DEFAULT 'ANNUAL',
    summary_period_start        DATE            NOT NULL,
    summary_period_end          DATE            NOT NULL,
    -- ECM counts
    total_ecm_count             INTEGER         NOT NULL DEFAULT 0,
    verified_ecm_count          INTEGER         NOT NULL DEFAULT 0,
    -- Energy savings
    total_avoided_energy_kwh    NUMERIC(18,3)   NOT NULL DEFAULT 0,
    total_normalized_savings_kwh NUMERIC(18,3)  DEFAULT 0,
    total_demand_savings_kw     NUMERIC(12,3)   DEFAULT 0,
    savings_pct_of_baseline     NUMERIC(8,4),
    savings_pct_of_guaranteed   NUMERIC(8,4),
    -- Cost savings
    total_energy_cost_savings   NUMERIC(18,2)   NOT NULL DEFAULT 0,
    total_demand_cost_savings   NUMERIC(18,2)   DEFAULT 0,
    total_other_savings         NUMERIC(18,2)   DEFAULT 0,
    total_cost_savings          NUMERIC(18,2)   NOT NULL DEFAULT 0,
    currency_code               VARCHAR(3)      NOT NULL DEFAULT 'USD',
    -- Emissions
    total_avoided_co2_kg        NUMERIC(18,3)   DEFAULT 0,
    -- Uncertainty
    combined_uncertainty_kwh    NUMERIC(18,3),
    combined_uncertainty_pct    NUMERIC(8,4),
    savings_significant         BOOLEAN,
    -- Performance contract
    meets_guarantee             BOOLEAN,
    guarantee_margin_kwh        NUMERIC(18,3),
    guarantee_margin_pct        NUMERIC(8,4),
    -- Status
    summary_status              VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_ss_type CHECK (
        summary_type IN (
            'MONTHLY', 'QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL',
            'CONTRACT_YEAR', 'LIFE_TO_DATE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_ss_dates CHECK (
        summary_period_start < summary_period_end
    ),
    CONSTRAINT chk_p040_ss_ecm_count CHECK (
        total_ecm_count >= 0 AND verified_ecm_count >= 0 AND
        verified_ecm_count <= total_ecm_count
    ),
    CONSTRAINT chk_p040_ss_status CHECK (
        summary_status IN (
            'DRAFT', 'CALCULATED', 'REVIEWED', 'APPROVED', 'FINAL', 'REVISED'
        )
    ),
    CONSTRAINT chk_p040_ss_uncertainty CHECK (
        combined_uncertainty_pct IS NULL OR combined_uncertainty_pct >= 0
    ),
    CONSTRAINT uq_p040_ss_project_period UNIQUE (project_id, summary_type, summary_period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_ss_tenant            ON pack040_mv.mv_savings_summaries(tenant_id);
CREATE INDEX idx_p040_ss_project           ON pack040_mv.mv_savings_summaries(project_id);
CREATE INDEX idx_p040_ss_type              ON pack040_mv.mv_savings_summaries(summary_type);
CREATE INDEX idx_p040_ss_period            ON pack040_mv.mv_savings_summaries(summary_period_start, summary_period_end);
CREATE INDEX idx_p040_ss_status            ON pack040_mv.mv_savings_summaries(summary_status);
CREATE INDEX idx_p040_ss_guarantee         ON pack040_mv.mv_savings_summaries(meets_guarantee);
CREATE INDEX idx_p040_ss_created           ON pack040_mv.mv_savings_summaries(created_at DESC);

-- Composite: project + approved summaries
CREATE INDEX idx_p040_ss_project_approved  ON pack040_mv.mv_savings_summaries(project_id, summary_period_start DESC)
    WHERE summary_status IN ('APPROVED', 'FINAL');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_ss_updated
    BEFORE UPDATE ON pack040_mv.mv_savings_summaries
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_savings_periods ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_avoided_energy ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_cost_savings ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_cumulative_savings ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_savings_summaries ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_sp_tenant_isolation
    ON pack040_mv.mv_savings_periods
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_sp_service_bypass
    ON pack040_mv.mv_savings_periods
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_ae_tenant_isolation
    ON pack040_mv.mv_avoided_energy
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_ae_service_bypass
    ON pack040_mv.mv_avoided_energy
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_cs_tenant_isolation
    ON pack040_mv.mv_cost_savings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_cs_service_bypass
    ON pack040_mv.mv_cost_savings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_cum_tenant_isolation
    ON pack040_mv.mv_cumulative_savings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_cum_service_bypass
    ON pack040_mv.mv_cumulative_savings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_ss_tenant_isolation
    ON pack040_mv.mv_savings_summaries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_ss_service_bypass
    ON pack040_mv.mv_savings_summaries
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_savings_periods TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_avoided_energy TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_cost_savings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_cumulative_savings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_savings_summaries TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.mv_savings_periods IS
    'Reporting periods for savings calculation with adjusted baseline, actual consumption, and avoided energy values.';
COMMENT ON TABLE pack040_mv.mv_avoided_energy IS
    'Per-ECM avoided energy calculations with IPMVP option-specific methods, interactive effects, and uncertainty bounds.';
COMMENT ON TABLE pack040_mv.mv_cost_savings IS
    'Energy cost savings based on avoided energy, rate schedules, demand charges, and ancillary savings.';
COMMENT ON TABLE pack040_mv.mv_cumulative_savings IS
    'Year-to-date and life-to-date cumulative savings for multi-year performance contract tracking.';
COMMENT ON TABLE pack040_mv.mv_savings_summaries IS
    'High-level savings summaries aggregating all ECMs with guarantee tracking, uncertainty, and compliance status.';

COMMENT ON COLUMN pack040_mv.mv_savings_periods.avoided_energy_kwh IS 'Avoided energy = adjusted baseline - actual consumption (IPMVP savings equation).';
COMMENT ON COLUMN pack040_mv.mv_savings_periods.normalized_savings_kwh IS 'Savings normalized to standard conditions (e.g., TMY weather) for year-over-year comparison.';
COMMENT ON COLUMN pack040_mv.mv_savings_periods.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_avoided_energy.stipulated_value_kwh IS 'Stipulated energy value for IPMVP Option A non-measured parameters.';
COMMENT ON COLUMN pack040_mv.mv_avoided_energy.interactive_adjustment_kwh IS 'Adjustment for interactive effects between ECMs (e.g., lighting-HVAC interaction).';

COMMENT ON COLUMN pack040_mv.mv_cost_savings.blended_rate_per_kwh IS 'Blended average energy rate including all charges divided by total consumption.';
COMMENT ON COLUMN pack040_mv.mv_cost_savings.avoided_co2_kg IS 'CO2 emissions avoided through energy savings for GHG reporting.';

COMMENT ON COLUMN pack040_mv.mv_cumulative_savings.guaranteed_pct_achieved IS 'Percentage of guaranteed savings achieved (actual/guaranteed * 100).';
COMMENT ON COLUMN pack040_mv.mv_cumulative_savings.simple_payback_achieved IS 'Whether cumulative cost savings have exceeded implementation cost.';

-- =============================================================================
-- V312: PACK-039 Energy Monitoring Pack - Energy Budgeting
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates energy budgeting tables for planning, tracking, and variance
-- analysis of energy consumption and costs. Includes budget definitions,
-- period-level budget allocations, variance tracking against actuals,
-- rolling forecasts based on consumption trends, and budget alert
-- configurations for proactive management.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_budgets
--   2. pack039_energy_monitoring.em_budget_periods
--   3. pack039_energy_monitoring.em_variance_records
--   4. pack039_energy_monitoring.em_rolling_forecasts
--   5. pack039_energy_monitoring.em_budget_alerts
--
-- Previous: V311__pack039_energy_monitoring_006.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_budgets
-- =============================================================================
-- Top-level energy budget definitions for facilities, buildings, or
-- departments. Each budget defines the annual or multi-year energy and
-- cost plan with target reduction percentages, baseline references, and
-- approval workflow. Budgets serve as the benchmark for variance analysis
-- and trigger alerts when consumption deviates from plan.

CREATE TABLE pack039_energy_monitoring.em_budgets (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    account_id              UUID            REFERENCES pack039_energy_monitoring.em_tenant_accounts(id),
    budget_name             VARCHAR(255)    NOT NULL,
    budget_code             VARCHAR(50)     NOT NULL,
    budget_type             VARCHAR(30)     NOT NULL DEFAULT 'ANNUAL',
    energy_type             VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    fiscal_year             INTEGER         NOT NULL,
    budget_start_date       DATE            NOT NULL,
    budget_end_date         DATE            NOT NULL,
    total_energy_budget_kwh NUMERIC(15,3)   NOT NULL,
    energy_unit             VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    total_cost_budget       NUMERIC(12,2)   NOT NULL,
    cost_currency           VARCHAR(3)      NOT NULL DEFAULT 'USD',
    baseline_energy_kwh     NUMERIC(15,3),
    baseline_year           INTEGER,
    target_reduction_pct    NUMERIC(7,4),
    target_enpi_value       NUMERIC(15,6),
    enpi_unit               VARCHAR(50),
    budget_methodology      VARCHAR(30)     NOT NULL DEFAULT 'HISTORICAL',
    allocation_method       VARCHAR(30)     NOT NULL DEFAULT 'MONTHLY_FLAT',
    weather_normalized      BOOLEAN         NOT NULL DEFAULT false,
    production_adjusted     BOOLEAN         NOT NULL DEFAULT false,
    assumed_hdd             NUMERIC(10,2),
    assumed_cdd             NUMERIC(10,2),
    assumed_production      NUMERIC(15,3),
    production_unit         VARCHAR(50),
    assumed_occupancy_pct   NUMERIC(5,2),
    assumed_rate_per_kwh    NUMERIC(10,6),
    assumptions             JSONB           DEFAULT '{}',
    contingency_pct         NUMERIC(5,2)    DEFAULT 0,
    ytd_actual_energy_kwh   NUMERIC(15,3)   NOT NULL DEFAULT 0,
    ytd_actual_cost         NUMERIC(12,2)   NOT NULL DEFAULT 0,
    ytd_variance_energy_pct NUMERIC(10,4),
    ytd_variance_cost_pct   NUMERIC(10,4),
    budget_status           VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approval_status         VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    revision_number         INTEGER         NOT NULL DEFAULT 1,
    previous_budget_id      UUID,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p039_bg_type CHECK (
        budget_type IN (
            'ANNUAL', 'MULTI_YEAR', 'QUARTERLY', 'PROJECT', 'ROLLING'
        )
    ),
    CONSTRAINT chk_p039_bg_energy_type CHECK (
        energy_type IN (
            'ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'CHILLED_WATER',
            'HOT_WATER', 'TOTAL_ENERGY', 'PRIMARY_ENERGY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_bg_methodology CHECK (
        budget_methodology IN (
            'HISTORICAL', 'ZERO_BASED', 'WEATHER_NORMALIZED',
            'REGRESSION_BASED', 'BENCHMARK', 'TOP_DOWN',
            'BOTTOM_UP', 'HYBRID', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_bg_alloc_method CHECK (
        allocation_method IN (
            'MONTHLY_FLAT', 'MONTHLY_SEASONAL', 'MONTHLY_HISTORICAL',
            'WEEKLY', 'DAILY', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_bg_budget_status CHECK (
        budget_status IN (
            'DRAFT', 'ACTIVE', 'UNDER_REVIEW', 'CLOSED',
            'SUPERSEDED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p039_bg_approval CHECK (
        approval_status IN (
            'DRAFT', 'PENDING_REVIEW', 'APPROVED', 'REJECTED', 'SUPERSEDED'
        )
    ),
    CONSTRAINT chk_p039_bg_energy CHECK (
        total_energy_budget_kwh > 0
    ),
    CONSTRAINT chk_p039_bg_cost CHECK (
        total_cost_budget > 0
    ),
    CONSTRAINT chk_p039_bg_reduction CHECK (
        target_reduction_pct IS NULL OR (target_reduction_pct >= -100 AND target_reduction_pct <= 100)
    ),
    CONSTRAINT chk_p039_bg_occupancy CHECK (
        assumed_occupancy_pct IS NULL OR (assumed_occupancy_pct >= 0 AND assumed_occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p039_bg_contingency CHECK (
        contingency_pct IS NULL OR (contingency_pct >= 0 AND contingency_pct <= 50)
    ),
    CONSTRAINT chk_p039_bg_revision CHECK (
        revision_number >= 1
    ),
    CONSTRAINT chk_p039_bg_dates CHECK (
        budget_start_date <= budget_end_date
    ),
    CONSTRAINT uq_p039_bg_tenant_code_rev UNIQUE (tenant_id, budget_code, revision_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_bg_tenant          ON pack039_energy_monitoring.em_budgets(tenant_id);
CREATE INDEX idx_p039_bg_facility        ON pack039_energy_monitoring.em_budgets(facility_id);
CREATE INDEX idx_p039_bg_account         ON pack039_energy_monitoring.em_budgets(account_id);
CREATE INDEX idx_p039_bg_code            ON pack039_energy_monitoring.em_budgets(budget_code);
CREATE INDEX idx_p039_bg_type            ON pack039_energy_monitoring.em_budgets(budget_type);
CREATE INDEX idx_p039_bg_energy_type     ON pack039_energy_monitoring.em_budgets(energy_type);
CREATE INDEX idx_p039_bg_fiscal_year     ON pack039_energy_monitoring.em_budgets(fiscal_year);
CREATE INDEX idx_p039_bg_status          ON pack039_energy_monitoring.em_budgets(budget_status);
CREATE INDEX idx_p039_bg_approval        ON pack039_energy_monitoring.em_budgets(approval_status);
CREATE INDEX idx_p039_bg_start_date      ON pack039_energy_monitoring.em_budgets(budget_start_date);
CREATE INDEX idx_p039_bg_created         ON pack039_energy_monitoring.em_budgets(created_at DESC);
CREATE INDEX idx_p039_bg_assumptions     ON pack039_energy_monitoring.em_budgets USING GIN(assumptions);

-- Composite: active budgets by facility and year
CREATE INDEX idx_p039_bg_fac_active      ON pack039_energy_monitoring.em_budgets(facility_id, fiscal_year)
    WHERE budget_status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_bg_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_budgets
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_budget_periods
-- =============================================================================
-- Period-level budget allocations (monthly, weekly, or daily) within a
-- budget. Each period has budgeted energy, cost, and relevant variable
-- assumptions. Period budgets enable granular variance tracking and
-- seasonal adjustment of energy expectations.

CREATE TABLE pack039_energy_monitoring.em_budget_periods (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    budget_id               UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_budgets(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    period_type             VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    period_label            VARCHAR(50),
    budgeted_energy_kwh     NUMERIC(15,3)   NOT NULL,
    budgeted_cost           NUMERIC(12,2)   NOT NULL,
    budgeted_demand_kw      NUMERIC(12,3),
    budgeted_enpi_value     NUMERIC(15,6),
    assumed_hdd             NUMERIC(8,2),
    assumed_cdd             NUMERIC(8,2),
    assumed_production      NUMERIC(15,3),
    assumed_occupancy_pct   NUMERIC(5,2),
    assumed_operating_days  INTEGER,
    assumed_rate_per_kwh    NUMERIC(10,6),
    seasonal_factor         NUMERIC(5,4)    DEFAULT 1.0,
    actual_energy_kwh       NUMERIC(15,3),
    actual_cost             NUMERIC(12,2),
    actual_demand_kw        NUMERIC(12,3),
    actual_enpi_value       NUMERIC(15,6),
    variance_energy_kwh     NUMERIC(15,3),
    variance_energy_pct     NUMERIC(10,4),
    variance_cost           NUMERIC(12,2),
    variance_cost_pct       NUMERIC(10,4),
    variance_status         VARCHAR(20),
    is_locked               BOOLEAN         NOT NULL DEFAULT false,
    locked_at               TIMESTAMPTZ,
    locked_by               UUID,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_bp_period_type CHECK (
        period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY')
    ),
    CONSTRAINT chk_p039_bp_energy CHECK (
        budgeted_energy_kwh >= 0
    ),
    CONSTRAINT chk_p039_bp_cost CHECK (
        budgeted_cost >= 0
    ),
    CONSTRAINT chk_p039_bp_occupancy CHECK (
        assumed_occupancy_pct IS NULL OR (assumed_occupancy_pct >= 0 AND assumed_occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p039_bp_seasonal CHECK (
        seasonal_factor IS NULL OR (seasonal_factor >= 0 AND seasonal_factor <= 5)
    ),
    CONSTRAINT chk_p039_bp_operating_days CHECK (
        assumed_operating_days IS NULL OR (assumed_operating_days >= 0 AND assumed_operating_days <= 366)
    ),
    CONSTRAINT chk_p039_bp_variance_status CHECK (
        variance_status IS NULL OR variance_status IN (
            'ON_BUDGET', 'UNDER_BUDGET', 'OVER_BUDGET',
            'WARNING', 'CRITICAL', 'NOT_YET'
        )
    ),
    CONSTRAINT chk_p039_bp_dates CHECK (
        period_start <= period_end
    ),
    CONSTRAINT uq_p039_bp_budget_period UNIQUE (budget_id, period_type, period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_bp_budget          ON pack039_energy_monitoring.em_budget_periods(budget_id);
CREATE INDEX idx_p039_bp_tenant          ON pack039_energy_monitoring.em_budget_periods(tenant_id);
CREATE INDEX idx_p039_bp_period_type     ON pack039_energy_monitoring.em_budget_periods(period_type);
CREATE INDEX idx_p039_bp_period_start    ON pack039_energy_monitoring.em_budget_periods(period_start DESC);
CREATE INDEX idx_p039_bp_variance_status ON pack039_energy_monitoring.em_budget_periods(variance_status);
CREATE INDEX idx_p039_bp_locked          ON pack039_energy_monitoring.em_budget_periods(is_locked) WHERE is_locked = false;
CREATE INDEX idx_p039_bp_created         ON pack039_energy_monitoring.em_budget_periods(created_at DESC);

-- Composite: budget + monthly periods for variance dashboard
CREATE INDEX idx_p039_bp_budget_monthly  ON pack039_energy_monitoring.em_budget_periods(budget_id, period_start DESC)
    WHERE period_type = 'MONTHLY';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_bp_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_budget_periods
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_variance_records
-- =============================================================================
-- Detailed variance analysis records comparing actual to budgeted energy
-- consumption. Each record decomposes the total variance into contributing
-- factors (weather, production, occupancy, rate, behavioral) to explain
-- why actual deviated from budget. Supports both automatic decomposition
-- and manual commentary.

CREATE TABLE pack039_energy_monitoring.em_variance_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    budget_period_id        UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_budget_periods(id) ON DELETE CASCADE,
    budget_id               UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_budgets(id),
    tenant_id               UUID            NOT NULL,
    analysis_date           DATE            NOT NULL DEFAULT CURRENT_DATE,
    total_variance_kwh      NUMERIC(15,3)   NOT NULL,
    total_variance_pct      NUMERIC(10,4)   NOT NULL,
    total_variance_cost     NUMERIC(12,2),
    weather_variance_kwh    NUMERIC(15,3),
    weather_variance_pct    NUMERIC(10,4),
    production_variance_kwh NUMERIC(15,3),
    production_variance_pct NUMERIC(10,4),
    occupancy_variance_kwh  NUMERIC(15,3),
    occupancy_variance_pct  NUMERIC(10,4),
    rate_variance           NUMERIC(12,2),
    rate_variance_pct       NUMERIC(10,4),
    behavioral_variance_kwh NUMERIC(15,3),
    behavioral_variance_pct NUMERIC(10,4),
    equipment_variance_kwh  NUMERIC(15,3),
    equipment_variance_pct  NUMERIC(10,4),
    unexplained_variance_kwh NUMERIC(15,3),
    unexplained_variance_pct NUMERIC(10,4),
    actual_hdd              NUMERIC(8,2),
    budgeted_hdd            NUMERIC(8,2),
    actual_cdd              NUMERIC(8,2),
    budgeted_cdd            NUMERIC(8,2),
    actual_production       NUMERIC(15,3),
    budgeted_production     NUMERIC(15,3),
    actual_occupancy_pct    NUMERIC(5,2),
    budgeted_occupancy_pct  NUMERIC(5,2),
    variance_severity       VARCHAR(20)     NOT NULL DEFAULT 'NORMAL',
    corrective_actions      JSONB           DEFAULT '[]',
    commentary              TEXT,
    reviewed_by             UUID,
    reviewed_at             TIMESTAMPTZ,
    decomposition_method    VARCHAR(30)     DEFAULT 'REGRESSION',
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_vrc_severity CHECK (
        variance_severity IN ('NORMAL', 'MINOR', 'MODERATE', 'MAJOR', 'CRITICAL')
    ),
    CONSTRAINT chk_p039_vrc_method CHECK (
        decomposition_method IS NULL OR decomposition_method IN (
            'REGRESSION', 'PROPORTIONAL', 'SEQUENTIAL', 'MANUAL', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p039_vrc_occupancy_actual CHECK (
        actual_occupancy_pct IS NULL OR (actual_occupancy_pct >= 0 AND actual_occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p039_vrc_occupancy_budget CHECK (
        budgeted_occupancy_pct IS NULL OR (budgeted_occupancy_pct >= 0 AND budgeted_occupancy_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_vrc_budget_period  ON pack039_energy_monitoring.em_variance_records(budget_period_id);
CREATE INDEX idx_p039_vrc_budget         ON pack039_energy_monitoring.em_variance_records(budget_id);
CREATE INDEX idx_p039_vrc_tenant         ON pack039_energy_monitoring.em_variance_records(tenant_id);
CREATE INDEX idx_p039_vrc_analysis_date  ON pack039_energy_monitoring.em_variance_records(analysis_date DESC);
CREATE INDEX idx_p039_vrc_total_var_pct  ON pack039_energy_monitoring.em_variance_records(total_variance_pct DESC);
CREATE INDEX idx_p039_vrc_severity       ON pack039_energy_monitoring.em_variance_records(variance_severity);
CREATE INDEX idx_p039_vrc_reviewed       ON pack039_energy_monitoring.em_variance_records(reviewed_by) WHERE reviewed_by IS NULL;
CREATE INDEX idx_p039_vrc_created        ON pack039_energy_monitoring.em_variance_records(created_at DESC);
CREATE INDEX idx_p039_vrc_actions        ON pack039_energy_monitoring.em_variance_records USING GIN(corrective_actions);
CREATE INDEX idx_p039_vrc_details        ON pack039_energy_monitoring.em_variance_records USING GIN(details);

-- Composite: major variances for management review
CREATE INDEX idx_p039_vrc_major          ON pack039_energy_monitoring.em_variance_records(budget_id, analysis_date DESC)
    WHERE variance_severity IN ('MAJOR', 'CRITICAL');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_vrc_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_variance_records
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_rolling_forecasts
-- =============================================================================
-- Rolling energy and cost forecasts based on actual consumption trends,
-- weather predictions, and production schedules. Updated periodically
-- to provide forward-looking budget compliance projections. Enables
-- early warning when year-end projections exceed budget thresholds.

CREATE TABLE pack039_energy_monitoring.em_rolling_forecasts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    budget_id               UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_budgets(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    forecast_date           DATE            NOT NULL,
    forecast_horizon_months INTEGER         NOT NULL DEFAULT 12,
    forecast_method         VARCHAR(30)     NOT NULL DEFAULT 'TREND',
    forecast_period_start   DATE            NOT NULL,
    forecast_period_end     DATE            NOT NULL,
    forecasted_energy_kwh   NUMERIC(15,3)   NOT NULL,
    forecasted_cost         NUMERIC(12,2)   NOT NULL,
    forecasted_demand_kw    NUMERIC(12,3),
    forecasted_enpi_value   NUMERIC(15,6),
    confidence_low_kwh      NUMERIC(15,3),
    confidence_high_kwh     NUMERIC(15,3),
    confidence_level_pct    NUMERIC(5,2)    DEFAULT 80.0,
    ytd_actual_energy_kwh   NUMERIC(15,3),
    ytd_actual_cost         NUMERIC(12,2),
    remaining_budget_kwh    NUMERIC(15,3),
    remaining_budget_cost   NUMERIC(12,2),
    projected_year_end_kwh  NUMERIC(15,3)   NOT NULL,
    projected_year_end_cost NUMERIC(12,2)   NOT NULL,
    projected_variance_pct  NUMERIC(10,4),
    budget_compliance       VARCHAR(20)     NOT NULL DEFAULT 'ON_TRACK',
    weather_outlook         JSONB           DEFAULT '{}',
    production_outlook      JSONB           DEFAULT '{}',
    model_accuracy_mape_pct NUMERIC(8,4),
    monthly_breakdown       JSONB           DEFAULT '[]',
    assumptions             JSONB           DEFAULT '{}',
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_rf_method CHECK (
        forecast_method IN (
            'TREND', 'REGRESSION', 'SEASONAL_DECOMPOSITION',
            'ARIMA', 'EXPONENTIAL_SMOOTHING', 'MACHINE_LEARNING',
            'WEATHER_NORMALIZED', 'MANUAL', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p039_rf_compliance CHECK (
        budget_compliance IN (
            'ON_TRACK', 'AT_RISK', 'OVER_BUDGET', 'UNDER_BUDGET', 'UNKNOWN'
        )
    ),
    CONSTRAINT chk_p039_rf_horizon CHECK (
        forecast_horizon_months >= 1 AND forecast_horizon_months <= 60
    ),
    CONSTRAINT chk_p039_rf_confidence CHECK (
        confidence_level_pct IS NULL OR (confidence_level_pct >= 50 AND confidence_level_pct <= 99)
    ),
    CONSTRAINT chk_p039_rf_energy CHECK (
        forecasted_energy_kwh >= 0 AND projected_year_end_kwh >= 0
    ),
    CONSTRAINT chk_p039_rf_cost CHECK (
        forecasted_cost >= 0 AND projected_year_end_cost >= 0
    ),
    CONSTRAINT chk_p039_rf_mape CHECK (
        model_accuracy_mape_pct IS NULL OR model_accuracy_mape_pct >= 0
    ),
    CONSTRAINT chk_p039_rf_dates CHECK (
        forecast_period_start <= forecast_period_end
    ),
    CONSTRAINT uq_p039_rf_budget_forecast_date UNIQUE (budget_id, forecast_date, forecast_method)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_rf_budget          ON pack039_energy_monitoring.em_rolling_forecasts(budget_id);
CREATE INDEX idx_p039_rf_tenant          ON pack039_energy_monitoring.em_rolling_forecasts(tenant_id);
CREATE INDEX idx_p039_rf_forecast_date   ON pack039_energy_monitoring.em_rolling_forecasts(forecast_date DESC);
CREATE INDEX idx_p039_rf_method          ON pack039_energy_monitoring.em_rolling_forecasts(forecast_method);
CREATE INDEX idx_p039_rf_compliance      ON pack039_energy_monitoring.em_rolling_forecasts(budget_compliance);
CREATE INDEX idx_p039_rf_var_pct         ON pack039_energy_monitoring.em_rolling_forecasts(projected_variance_pct DESC);
CREATE INDEX idx_p039_rf_created         ON pack039_energy_monitoring.em_rolling_forecasts(created_at DESC);
CREATE INDEX idx_p039_rf_monthly         ON pack039_energy_monitoring.em_rolling_forecasts USING GIN(monthly_breakdown);

-- Composite: latest forecast per budget
CREATE INDEX idx_p039_rf_budget_latest   ON pack039_energy_monitoring.em_rolling_forecasts(budget_id, forecast_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_rf_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_rolling_forecasts
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_budget_alerts
-- =============================================================================
-- Alert configurations and triggered alerts for budget variance monitoring.
-- Defines thresholds for energy and cost variance alerts at various levels
-- (warning, critical) with notification routing and escalation rules.
-- Tracks alert lifecycle from trigger through acknowledgment to resolution.

CREATE TABLE pack039_energy_monitoring.em_budget_alerts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    budget_id               UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_budgets(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    alert_name              VARCHAR(255)    NOT NULL,
    alert_type              VARCHAR(30)     NOT NULL DEFAULT 'VARIANCE',
    alert_metric            VARCHAR(30)     NOT NULL DEFAULT 'ENERGY',
    threshold_warning_pct   NUMERIC(7,4)    NOT NULL DEFAULT 5.0,
    threshold_critical_pct  NUMERIC(7,4)    NOT NULL DEFAULT 10.0,
    threshold_direction     VARCHAR(10)     NOT NULL DEFAULT 'OVER',
    check_frequency         VARCHAR(20)     NOT NULL DEFAULT 'DAILY',
    current_severity        VARCHAR(20),
    current_variance_pct    NUMERIC(10,4),
    is_triggered            BOOLEAN         NOT NULL DEFAULT false,
    triggered_at            TIMESTAMPTZ,
    triggered_value         NUMERIC(15,3),
    acknowledged_at         TIMESTAMPTZ,
    acknowledged_by         UUID,
    resolved_at             TIMESTAMPTZ,
    resolved_by             UUID,
    resolution_notes        TEXT,
    notification_channels   JSONB           DEFAULT '[]',
    notification_recipients JSONB           DEFAULT '[]',
    escalation_after_hours  INTEGER         DEFAULT 24,
    escalation_recipients   JSONB           DEFAULT '[]',
    snooze_until            TIMESTAMPTZ,
    trigger_count           INTEGER         NOT NULL DEFAULT 0,
    last_checked_at         TIMESTAMPTZ,
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ba_type CHECK (
        alert_type IN (
            'VARIANCE', 'FORECAST', 'TREND', 'THRESHOLD',
            'CUMULATIVE', 'RATE_OF_CHANGE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_ba_metric CHECK (
        alert_metric IN (
            'ENERGY', 'COST', 'DEMAND', 'ENPI', 'ALL'
        )
    ),
    CONSTRAINT chk_p039_ba_direction CHECK (
        threshold_direction IN ('OVER', 'UNDER', 'BOTH')
    ),
    CONSTRAINT chk_p039_ba_frequency CHECK (
        check_frequency IN ('HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY')
    ),
    CONSTRAINT chk_p039_ba_severity CHECK (
        current_severity IS NULL OR current_severity IN ('WARNING', 'CRITICAL', 'RESOLVED')
    ),
    CONSTRAINT chk_p039_ba_warning CHECK (
        threshold_warning_pct > 0
    ),
    CONSTRAINT chk_p039_ba_critical CHECK (
        threshold_critical_pct > 0
    ),
    CONSTRAINT chk_p039_ba_thresholds CHECK (
        threshold_warning_pct <= threshold_critical_pct
    ),
    CONSTRAINT chk_p039_ba_escalation CHECK (
        escalation_after_hours IS NULL OR escalation_after_hours > 0
    ),
    CONSTRAINT chk_p039_ba_trigger_count CHECK (
        trigger_count >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ba_budget          ON pack039_energy_monitoring.em_budget_alerts(budget_id);
CREATE INDEX idx_p039_ba_tenant          ON pack039_energy_monitoring.em_budget_alerts(tenant_id);
CREATE INDEX idx_p039_ba_type            ON pack039_energy_monitoring.em_budget_alerts(alert_type);
CREATE INDEX idx_p039_ba_metric          ON pack039_energy_monitoring.em_budget_alerts(alert_metric);
CREATE INDEX idx_p039_ba_triggered       ON pack039_energy_monitoring.em_budget_alerts(is_triggered) WHERE is_triggered = true;
CREATE INDEX idx_p039_ba_severity        ON pack039_energy_monitoring.em_budget_alerts(current_severity);
CREATE INDEX idx_p039_ba_enabled         ON pack039_energy_monitoring.em_budget_alerts(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_ba_triggered_at    ON pack039_energy_monitoring.em_budget_alerts(triggered_at DESC);
CREATE INDEX idx_p039_ba_last_checked    ON pack039_energy_monitoring.em_budget_alerts(last_checked_at);
CREATE INDEX idx_p039_ba_created         ON pack039_energy_monitoring.em_budget_alerts(created_at DESC);
CREATE INDEX idx_p039_ba_recipients      ON pack039_energy_monitoring.em_budget_alerts USING GIN(notification_recipients);

-- Composite: active triggered alerts for operations dashboard
CREATE INDEX idx_p039_ba_active_trig     ON pack039_energy_monitoring.em_budget_alerts(current_severity, triggered_at DESC)
    WHERE is_triggered = true AND is_enabled = true AND resolved_at IS NULL;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ba_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_budget_alerts
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_budgets ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_budget_periods ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_variance_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_rolling_forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_budget_alerts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_bg_tenant_isolation
    ON pack039_energy_monitoring.em_budgets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_bg_service_bypass
    ON pack039_energy_monitoring.em_budgets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_bp_tenant_isolation
    ON pack039_energy_monitoring.em_budget_periods
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_bp_service_bypass
    ON pack039_energy_monitoring.em_budget_periods
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_vrc_tenant_isolation
    ON pack039_energy_monitoring.em_variance_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_vrc_service_bypass
    ON pack039_energy_monitoring.em_variance_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_rf_tenant_isolation
    ON pack039_energy_monitoring.em_rolling_forecasts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_rf_service_bypass
    ON pack039_energy_monitoring.em_rolling_forecasts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ba_tenant_isolation
    ON pack039_energy_monitoring.em_budget_alerts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ba_service_bypass
    ON pack039_energy_monitoring.em_budget_alerts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_budgets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_budget_periods TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_variance_records TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_rolling_forecasts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_budget_alerts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.em_budgets IS
    'Energy budget definitions with annual targets, reduction goals, methodology, and approval workflow.';
COMMENT ON TABLE pack039_energy_monitoring.em_budget_periods IS
    'Period-level budget allocations (monthly/weekly) with seasonal factors, assumptions, and variance tracking.';
COMMENT ON TABLE pack039_energy_monitoring.em_variance_records IS
    'Detailed variance analysis decomposing budget deviations into weather, production, occupancy, rate, and behavioral factors.';
COMMENT ON TABLE pack039_energy_monitoring.em_rolling_forecasts IS
    'Rolling energy and cost forecasts with confidence intervals, year-end projections, and budget compliance status.';
COMMENT ON TABLE pack039_energy_monitoring.em_budget_alerts IS
    'Budget variance alert definitions with threshold triggers, notification routing, and escalation configuration.';

COMMENT ON COLUMN pack039_energy_monitoring.em_budgets.budget_methodology IS 'Budgeting approach: HISTORICAL, ZERO_BASED, WEATHER_NORMALIZED, REGRESSION_BASED, BENCHMARK, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_budgets.target_reduction_pct IS 'Target energy reduction percentage vs baseline. Negative values indicate planned increase.';
COMMENT ON COLUMN pack039_energy_monitoring.em_budgets.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_budget_periods.seasonal_factor IS 'Multiplier for seasonal energy distribution (e.g., 1.3 for summer, 0.8 for spring).';
COMMENT ON COLUMN pack039_energy_monitoring.em_budget_periods.variance_status IS 'Period budget status: ON_BUDGET, UNDER_BUDGET, OVER_BUDGET, WARNING, CRITICAL.';

COMMENT ON COLUMN pack039_energy_monitoring.em_variance_records.weather_variance_kwh IS 'Energy variance attributable to weather differences (actual vs budgeted HDD/CDD).';
COMMENT ON COLUMN pack039_energy_monitoring.em_variance_records.unexplained_variance_kwh IS 'Residual variance not explained by any identified factor.';

COMMENT ON COLUMN pack039_energy_monitoring.em_rolling_forecasts.budget_compliance IS 'Projected year-end budget compliance: ON_TRACK, AT_RISK, OVER_BUDGET, UNDER_BUDGET.';
COMMENT ON COLUMN pack039_energy_monitoring.em_rolling_forecasts.confidence_level_pct IS 'Prediction interval confidence level (e.g., 80% means 80% probability actual falls within range).';

COMMENT ON COLUMN pack039_energy_monitoring.em_budget_alerts.threshold_direction IS 'Alert direction: OVER (exceeding budget), UNDER (below budget), BOTH.';
COMMENT ON COLUMN pack039_energy_monitoring.em_budget_alerts.escalation_after_hours IS 'Hours after triggering before escalation to senior recipients.';

-- =============================================================================
-- V323: PACK-040 M&V Pack - Persistence Tracking
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for long-term savings persistence tracking including
-- persistence records, degradation analysis, re-commissioning triggers,
-- performance guarantees, and persistence alerts. Supports multi-year
-- savings verification for ESCO/EPC performance contracts.
--
-- Tables (5):
--   1. pack040_mv.mv_persistence_records
--   2. pack040_mv.mv_degradation_analysis
--   3. pack040_mv.mv_recommissioning_triggers
--   4. pack040_mv.mv_performance_guarantees
--   5. pack040_mv.mv_persistence_alerts
--
-- Previous: V322__pack040_mv_007.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.mv_persistence_records
-- =============================================================================
-- Year-over-year savings persistence records tracking how savings from
-- each ECM perform over time. Calculates persistence factors comparing
-- actual Year N savings to expected Year N savings. Detects degradation
-- trends and provides early warning of savings loss.

CREATE TABLE pack040_mv.mv_persistence_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    -- Period
    tracking_year               INTEGER         NOT NULL,
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    years_since_installation    NUMERIC(5,2)    NOT NULL,
    -- Expected savings (from Year 1 or adjusted)
    expected_savings_kwh        NUMERIC(18,3)   NOT NULL,
    expected_savings_adjusted   BOOLEAN         NOT NULL DEFAULT false,
    adjustment_reason           TEXT,
    -- Actual savings
    actual_savings_kwh          NUMERIC(18,3)   NOT NULL,
    normalized_savings_kwh      NUMERIC(18,3),
    -- Persistence metrics
    persistence_factor          NUMERIC(6,4)    NOT NULL,
    persistence_pct             NUMERIC(8,4)    NOT NULL,
    savings_retained_pct        NUMERIC(8,4),
    savings_lost_kwh            NUMERIC(18,3),
    savings_lost_pct            NUMERIC(8,4),
    -- Year-over-year comparison
    previous_year_savings_kwh   NUMERIC(18,3),
    yoy_change_kwh              NUMERIC(18,3),
    yoy_change_pct              NUMERIC(8,4),
    -- Trend
    trend_direction             VARCHAR(20)     NOT NULL DEFAULT 'STABLE',
    trend_slope                 NUMERIC(12,6),
    trend_r_squared             NUMERIC(8,6),
    trend_projected_year_n      NUMERIC(18,3),
    -- Uncertainty
    savings_uncertainty_kwh     NUMERIC(18,3),
    savings_uncertainty_pct     NUMERIC(8,4),
    -- Status
    persistence_status          VARCHAR(20)     NOT NULL DEFAULT 'ON_TRACK',
    requires_investigation      BOOLEAN         NOT NULL DEFAULT false,
    investigation_notes         TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_pr_year CHECK (
        tracking_year >= 2000 AND tracking_year <= 2100
    ),
    CONSTRAINT chk_p040_pr_dates CHECK (
        period_start < period_end
    ),
    CONSTRAINT chk_p040_pr_years_since CHECK (
        years_since_installation >= 0 AND years_since_installation <= 50
    ),
    CONSTRAINT chk_p040_pr_expected CHECK (
        expected_savings_kwh >= 0
    ),
    CONSTRAINT chk_p040_pr_factor CHECK (
        persistence_factor >= 0 AND persistence_factor <= 2.0
    ),
    CONSTRAINT chk_p040_pr_pct CHECK (
        persistence_pct >= 0 AND persistence_pct <= 200
    ),
    CONSTRAINT chk_p040_pr_trend CHECK (
        trend_direction IN (
            'IMPROVING', 'STABLE', 'SLIGHT_DECLINE', 'DECLINING',
            'RAPID_DECLINE', 'STEP_CHANGE', 'RECOVERY'
        )
    ),
    CONSTRAINT chk_p040_pr_status CHECK (
        persistence_status IN (
            'ON_TRACK', 'MARGINAL', 'BELOW_TARGET', 'CRITICAL',
            'FAILED', 'RECOVERING', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p040_pr_retained CHECK (
        savings_retained_pct IS NULL OR
        (savings_retained_pct >= 0 AND savings_retained_pct <= 200)
    ),
    CONSTRAINT uq_p040_pr_project_year UNIQUE (project_id, ecm_id, tracking_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_pr_tenant            ON pack040_mv.mv_persistence_records(tenant_id);
CREATE INDEX idx_p040_pr_project           ON pack040_mv.mv_persistence_records(project_id);
CREATE INDEX idx_p040_pr_ecm               ON pack040_mv.mv_persistence_records(ecm_id);
CREATE INDEX idx_p040_pr_year              ON pack040_mv.mv_persistence_records(tracking_year);
CREATE INDEX idx_p040_pr_period            ON pack040_mv.mv_persistence_records(period_start, period_end);
CREATE INDEX idx_p040_pr_status            ON pack040_mv.mv_persistence_records(persistence_status);
CREATE INDEX idx_p040_pr_trend             ON pack040_mv.mv_persistence_records(trend_direction);
CREATE INDEX idx_p040_pr_investigate       ON pack040_mv.mv_persistence_records(requires_investigation) WHERE requires_investigation = true;
CREATE INDEX idx_p040_pr_created           ON pack040_mv.mv_persistence_records(created_at DESC);

-- Composite: project + declining persistence
CREATE INDEX idx_p040_pr_project_decline   ON pack040_mv.mv_persistence_records(project_id, tracking_year DESC)
    WHERE persistence_status IN ('BELOW_TARGET', 'CRITICAL', 'FAILED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_pr_updated
    BEFORE UPDATE ON pack040_mv.mv_persistence_records
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_degradation_analysis
-- =============================================================================
-- Detailed degradation analysis modeling the rate at which savings decline
-- over time. Supports linear, exponential, and step-change degradation
-- models. Used to project future savings and determine when savings will
-- fall below acceptable thresholds.

CREATE TABLE pack040_mv.mv_degradation_analysis (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    analysis_date               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Analysis scope
    analysis_period_start       DATE            NOT NULL,
    analysis_period_end         DATE            NOT NULL,
    num_data_years              INTEGER         NOT NULL,
    -- Degradation model
    degradation_model           VARCHAR(30)     NOT NULL DEFAULT 'LINEAR',
    degradation_rate_pct_year   NUMERIC(8,4)    NOT NULL,
    degradation_rate_kwh_year   NUMERIC(18,3),
    -- Linear model: S(t) = S0 - rate * t
    linear_intercept            NUMERIC(18,6),
    linear_slope                NUMERIC(18,6),
    linear_r_squared            NUMERIC(8,6),
    -- Exponential model: S(t) = S0 * exp(-lambda * t)
    exponential_s0              NUMERIC(18,6),
    exponential_lambda          NUMERIC(10,6),
    exponential_r_squared       NUMERIC(8,6),
    exponential_half_life_years NUMERIC(6,2),
    -- Step change detection
    step_change_detected        BOOLEAN         NOT NULL DEFAULT false,
    step_change_date            DATE,
    step_change_magnitude_kwh   NUMERIC(18,3),
    step_change_magnitude_pct   NUMERIC(8,4),
    step_change_cause           TEXT,
    -- Projections
    projected_year_5_savings_kwh  NUMERIC(18,3),
    projected_year_10_savings_kwh NUMERIC(18,3),
    projected_zero_savings_year   INTEGER,
    projected_below_80pct_year    INTEGER,
    projected_below_50pct_year    INTEGER,
    -- Causes
    primary_degradation_cause   VARCHAR(50),
    contributing_factors        JSONB           DEFAULT '[]',
    -- Recommendations
    recommissioning_recommended BOOLEAN         NOT NULL DEFAULT false,
    estimated_recovery_kwh      NUMERIC(18,3),
    estimated_recovery_cost     NUMERIC(18,2),
    recovery_roi_pct            NUMERIC(8,4),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_da_dates CHECK (
        analysis_period_start < analysis_period_end
    ),
    CONSTRAINT chk_p040_da_years CHECK (
        num_data_years >= 2
    ),
    CONSTRAINT chk_p040_da_model CHECK (
        degradation_model IN (
            'LINEAR', 'EXPONENTIAL', 'STEP_CHANGE', 'PIECEWISE_LINEAR',
            'NO_DEGRADATION', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_da_rate CHECK (
        degradation_rate_pct_year >= -50 AND degradation_rate_pct_year <= 100
    ),
    CONSTRAINT chk_p040_da_r_squared_lin CHECK (
        linear_r_squared IS NULL OR (linear_r_squared >= 0 AND linear_r_squared <= 1)
    ),
    CONSTRAINT chk_p040_da_r_squared_exp CHECK (
        exponential_r_squared IS NULL OR (exponential_r_squared >= 0 AND exponential_r_squared <= 1)
    ),
    CONSTRAINT chk_p040_da_half_life CHECK (
        exponential_half_life_years IS NULL OR exponential_half_life_years > 0
    ),
    CONSTRAINT chk_p040_da_cause CHECK (
        primary_degradation_cause IS NULL OR primary_degradation_cause IN (
            'EQUIPMENT_AGING', 'MAINTENANCE_NEGLECT', 'OPERATIONAL_DRIFT',
            'BEHAVIORAL_REVERSION', 'CONTROLS_OVERRIDE', 'EQUIPMENT_FAILURE',
            'LOAD_INCREASE', 'WEATHER_SHIFT', 'OCCUPANCY_CHANGE',
            'NONE_DETECTED', 'UNKNOWN', 'MULTIPLE'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_da_tenant            ON pack040_mv.mv_degradation_analysis(tenant_id);
CREATE INDEX idx_p040_da_project           ON pack040_mv.mv_degradation_analysis(project_id);
CREATE INDEX idx_p040_da_ecm               ON pack040_mv.mv_degradation_analysis(ecm_id);
CREATE INDEX idx_p040_da_model             ON pack040_mv.mv_degradation_analysis(degradation_model);
CREATE INDEX idx_p040_da_cause             ON pack040_mv.mv_degradation_analysis(primary_degradation_cause);
CREATE INDEX idx_p040_da_recom             ON pack040_mv.mv_degradation_analysis(recommissioning_recommended) WHERE recommissioning_recommended = true;
CREATE INDEX idx_p040_da_step              ON pack040_mv.mv_degradation_analysis(step_change_detected) WHERE step_change_detected = true;
CREATE INDEX idx_p040_da_created           ON pack040_mv.mv_degradation_analysis(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_da_updated
    BEFORE UPDATE ON pack040_mv.mv_degradation_analysis
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_recommissioning_triggers
-- =============================================================================
-- Re-commissioning trigger events generated when savings persistence falls
-- below configured thresholds. Tracks trigger conditions, recommended actions,
-- and resolution status for maintaining savings performance.

CREATE TABLE pack040_mv.mv_recommissioning_triggers (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    persistence_record_id       UUID            REFERENCES pack040_mv.mv_persistence_records(id) ON DELETE SET NULL,
    degradation_analysis_id     UUID            REFERENCES pack040_mv.mv_degradation_analysis(id) ON DELETE SET NULL,
    -- Trigger details
    trigger_date                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    trigger_type                VARCHAR(50)     NOT NULL DEFAULT 'PERSISTENCE_BELOW_THRESHOLD',
    trigger_severity            VARCHAR(20)     NOT NULL DEFAULT 'WARNING',
    trigger_description         TEXT            NOT NULL,
    -- Threshold details
    threshold_type              VARCHAR(50)     NOT NULL,
    threshold_value             NUMERIC(10,4)   NOT NULL,
    actual_value                NUMERIC(10,4)   NOT NULL,
    exceedance_pct              NUMERIC(8,4),
    -- Impact
    estimated_savings_loss_kwh  NUMERIC(18,3),
    estimated_cost_impact       NUMERIC(18,2),
    -- Recommended actions
    recommended_actions         JSONB           NOT NULL DEFAULT '[]',
    estimated_recovery_kwh      NUMERIC(18,3),
    estimated_recovery_cost     NUMERIC(18,2),
    estimated_recovery_timeline_days INTEGER,
    -- Resolution
    trigger_status              VARCHAR(20)     NOT NULL DEFAULT 'OPEN',
    assigned_to                 VARCHAR(255),
    acknowledged_by             VARCHAR(255),
    acknowledged_at             TIMESTAMPTZ,
    resolved_by                 VARCHAR(255),
    resolved_at                 TIMESTAMPTZ,
    resolution_description      TEXT,
    resolution_effectiveness    VARCHAR(20),
    actual_recovery_kwh         NUMERIC(18,3),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_rt_type CHECK (
        trigger_type IN (
            'PERSISTENCE_BELOW_THRESHOLD', 'DEGRADATION_RATE_HIGH',
            'STEP_CHANGE_DETECTED', 'YOY_DECLINE_EXCEEDED',
            'GUARANTEE_AT_RISK', 'CUMULATIVE_LOSS_HIGH',
            'EQUIPMENT_PERFORMANCE', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p040_rt_severity CHECK (
        trigger_severity IN ('INFO', 'WARNING', 'CRITICAL', 'EMERGENCY')
    ),
    CONSTRAINT chk_p040_rt_threshold CHECK (
        threshold_type IN (
            'PERSISTENCE_FACTOR', 'ANNUAL_DEGRADATION_RATE',
            'CUMULATIVE_SAVINGS_LOSS', 'YOY_CHANGE',
            'GUARANTEE_MARGIN', 'PERFORMANCE_FACTOR'
        )
    ),
    CONSTRAINT chk_p040_rt_status CHECK (
        trigger_status IN (
            'OPEN', 'ACKNOWLEDGED', 'INVESTIGATING', 'ACTION_PLANNED',
            'IN_PROGRESS', 'RESOLVED', 'CLOSED', 'DEFERRED'
        )
    ),
    CONSTRAINT chk_p040_rt_effectiveness CHECK (
        resolution_effectiveness IS NULL OR resolution_effectiveness IN (
            'FULL_RECOVERY', 'PARTIAL_RECOVERY', 'NO_RECOVERY',
            'ONGOING', 'NOT_APPLICABLE'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_rt_tenant            ON pack040_mv.mv_recommissioning_triggers(tenant_id);
CREATE INDEX idx_p040_rt_project           ON pack040_mv.mv_recommissioning_triggers(project_id);
CREATE INDEX idx_p040_rt_ecm               ON pack040_mv.mv_recommissioning_triggers(ecm_id);
CREATE INDEX idx_p040_rt_type              ON pack040_mv.mv_recommissioning_triggers(trigger_type);
CREATE INDEX idx_p040_rt_severity          ON pack040_mv.mv_recommissioning_triggers(trigger_severity);
CREATE INDEX idx_p040_rt_status            ON pack040_mv.mv_recommissioning_triggers(trigger_status);
CREATE INDEX idx_p040_rt_date              ON pack040_mv.mv_recommissioning_triggers(trigger_date DESC);
CREATE INDEX idx_p040_rt_created           ON pack040_mv.mv_recommissioning_triggers(created_at DESC);

-- Composite: project + open triggers
CREATE INDEX idx_p040_rt_project_open      ON pack040_mv.mv_recommissioning_triggers(project_id, trigger_severity DESC)
    WHERE trigger_status IN ('OPEN', 'ACKNOWLEDGED', 'INVESTIGATING');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_rt_updated
    BEFORE UPDATE ON pack040_mv.mv_recommissioning_triggers
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_performance_guarantees
-- =============================================================================
-- Performance guarantee tracking for ESCO/EPC contracts. Records guaranteed
-- savings levels, actual performance, shortfall/surplus calculations, and
-- financial settlement details for multi-year performance contracts.

CREATE TABLE pack040_mv.mv_performance_guarantees (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    guarantee_name              VARCHAR(255)    NOT NULL,
    contract_year               INTEGER         NOT NULL,
    contract_year_start         DATE            NOT NULL,
    contract_year_end           DATE            NOT NULL,
    -- Guaranteed values
    guaranteed_savings_kwh      NUMERIC(18,3)   NOT NULL,
    guaranteed_savings_pct      NUMERIC(8,4),
    guaranteed_cost_savings     NUMERIC(18,2),
    guaranteed_demand_savings_kw NUMERIC(12,3),
    -- Actual performance
    actual_savings_kwh          NUMERIC(18,3),
    actual_savings_pct          NUMERIC(8,4),
    actual_cost_savings         NUMERIC(18,2),
    actual_demand_savings_kw    NUMERIC(12,3),
    -- Comparison
    savings_surplus_kwh         NUMERIC(18,3),
    savings_shortfall_kwh       NUMERIC(18,3),
    performance_ratio           NUMERIC(6,4),
    meets_guarantee             BOOLEAN,
    -- Financial settlement
    shortfall_payment           NUMERIC(18,2),
    surplus_sharing_payment     NUMERIC(18,2),
    net_settlement              NUMERIC(18,2),
    settlement_method           VARCHAR(50),
    settlement_status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    settlement_date             DATE,
    -- Adjustments
    guarantee_adjustment_kwh    NUMERIC(18,3),
    adjustment_reason           TEXT,
    adjusted_guarantee_kwh      NUMERIC(18,3),
    -- Dispute
    is_disputed                 BOOLEAN         NOT NULL DEFAULT false,
    dispute_description         TEXT,
    dispute_resolution          TEXT,
    currency_code               VARCHAR(3)      NOT NULL DEFAULT 'USD',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_pg_year CHECK (
        contract_year >= 1 AND contract_year <= 30
    ),
    CONSTRAINT chk_p040_pg_dates CHECK (
        contract_year_start < contract_year_end
    ),
    CONSTRAINT chk_p040_pg_guaranteed CHECK (
        guaranteed_savings_kwh >= 0
    ),
    CONSTRAINT chk_p040_pg_performance CHECK (
        performance_ratio IS NULL OR performance_ratio >= 0
    ),
    CONSTRAINT chk_p040_pg_settlement_method CHECK (
        settlement_method IS NULL OR settlement_method IN (
            'GUARANTEED_SAVINGS', 'SHARED_SAVINGS', 'FIXED_PAYMENT',
            'PERFORMANCE_BONUS', 'HYBRID', 'NONE'
        )
    ),
    CONSTRAINT chk_p040_pg_settlement_status CHECK (
        settlement_status IN (
            'PENDING', 'CALCULATED', 'INVOICED', 'PAID',
            'DISPUTED', 'SETTLED', 'WAIVED'
        )
    ),
    CONSTRAINT uq_p040_pg_project_year UNIQUE (project_id, contract_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_pg_tenant            ON pack040_mv.mv_performance_guarantees(tenant_id);
CREATE INDEX idx_p040_pg_project           ON pack040_mv.mv_performance_guarantees(project_id);
CREATE INDEX idx_p040_pg_year              ON pack040_mv.mv_performance_guarantees(contract_year);
CREATE INDEX idx_p040_pg_period            ON pack040_mv.mv_performance_guarantees(contract_year_start, contract_year_end);
CREATE INDEX idx_p040_pg_meets             ON pack040_mv.mv_performance_guarantees(meets_guarantee);
CREATE INDEX idx_p040_pg_disputed          ON pack040_mv.mv_performance_guarantees(is_disputed) WHERE is_disputed = true;
CREATE INDEX idx_p040_pg_settlement        ON pack040_mv.mv_performance_guarantees(settlement_status);
CREATE INDEX idx_p040_pg_created           ON pack040_mv.mv_performance_guarantees(created_at DESC);

-- Composite: project + unsettled guarantees
CREATE INDEX idx_p040_pg_project_unsettled ON pack040_mv.mv_performance_guarantees(project_id, contract_year)
    WHERE settlement_status IN ('PENDING', 'CALCULATED', 'DISPUTED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_pg_updated
    BEFORE UPDATE ON pack040_mv.mv_performance_guarantees
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_persistence_alerts
-- =============================================================================
-- Alert records for savings persistence monitoring. Generated when
-- persistence metrics cross configured thresholds. Supports multi-channel
-- notification with escalation.

CREATE TABLE pack040_mv.mv_persistence_alerts (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    -- Alert details
    alert_type                  VARCHAR(50)     NOT NULL,
    alert_severity              VARCHAR(20)     NOT NULL DEFAULT 'WARNING',
    alert_title                 VARCHAR(255)    NOT NULL,
    alert_message               TEXT            NOT NULL,
    alert_timestamp             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Metric details
    metric_name                 VARCHAR(100)    NOT NULL,
    metric_value                NUMERIC(18,6)   NOT NULL,
    threshold_value             NUMERIC(18,6)   NOT NULL,
    threshold_direction         VARCHAR(10)     NOT NULL DEFAULT 'BELOW',
    -- Context
    tracking_year               INTEGER,
    period_start                DATE,
    period_end                  DATE,
    persistence_factor          NUMERIC(6,4),
    degradation_rate_pct        NUMERIC(8,4),
    -- Notification
    notification_channels       VARCHAR(20)[]   DEFAULT '{EMAIL}',
    notification_sent           BOOLEAN         NOT NULL DEFAULT false,
    notification_sent_at        TIMESTAMPTZ,
    recipients                  JSONB           DEFAULT '[]',
    -- Escalation
    escalation_level            INTEGER         NOT NULL DEFAULT 0,
    escalated_at                TIMESTAMPTZ,
    escalation_reason           TEXT,
    -- Status
    alert_status                VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    acknowledged_by             VARCHAR(255),
    acknowledged_at             TIMESTAMPTZ,
    resolved_by                 VARCHAR(255),
    resolved_at                 TIMESTAMPTZ,
    resolution_notes            TEXT,
    auto_resolved               BOOLEAN         NOT NULL DEFAULT false,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_pa_type CHECK (
        alert_type IN (
            'PERSISTENCE_DECLINE', 'SAVINGS_SHORTFALL', 'DEGRADATION_HIGH',
            'STEP_CHANGE', 'GUARANTEE_AT_RISK', 'DATA_QUALITY_LOW',
            'CALIBRATION_DUE', 'REPORT_DUE', 'COMPLIANCE_DEADLINE',
            'RECOMMISSIONING_NEEDED', 'BASELINE_DRIFT'
        )
    ),
    CONSTRAINT chk_p040_pa_severity CHECK (
        alert_severity IN ('INFO', 'WARNING', 'CRITICAL', 'EMERGENCY')
    ),
    CONSTRAINT chk_p040_pa_direction CHECK (
        threshold_direction IN ('ABOVE', 'BELOW', 'EQUALS')
    ),
    CONSTRAINT chk_p040_pa_status CHECK (
        alert_status IN (
            'ACTIVE', 'ACKNOWLEDGED', 'INVESTIGATING',
            'RESOLVED', 'SUPPRESSED', 'EXPIRED'
        )
    ),
    CONSTRAINT chk_p040_pa_escalation CHECK (
        escalation_level >= 0 AND escalation_level <= 5
    ),
    CONSTRAINT chk_p040_pa_factor CHECK (
        persistence_factor IS NULL OR
        (persistence_factor >= 0 AND persistence_factor <= 2.0)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_pa_tenant            ON pack040_mv.mv_persistence_alerts(tenant_id);
CREATE INDEX idx_p040_pa_project           ON pack040_mv.mv_persistence_alerts(project_id);
CREATE INDEX idx_p040_pa_ecm               ON pack040_mv.mv_persistence_alerts(ecm_id);
CREATE INDEX idx_p040_pa_type              ON pack040_mv.mv_persistence_alerts(alert_type);
CREATE INDEX idx_p040_pa_severity          ON pack040_mv.mv_persistence_alerts(alert_severity);
CREATE INDEX idx_p040_pa_status            ON pack040_mv.mv_persistence_alerts(alert_status);
CREATE INDEX idx_p040_pa_timestamp         ON pack040_mv.mv_persistence_alerts(alert_timestamp DESC);
CREATE INDEX idx_p040_pa_created           ON pack040_mv.mv_persistence_alerts(created_at DESC);

-- Composite: project + active alerts
CREATE INDEX idx_p040_pa_project_active    ON pack040_mv.mv_persistence_alerts(project_id, alert_severity DESC)
    WHERE alert_status IN ('ACTIVE', 'ACKNOWLEDGED', 'INVESTIGATING');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_pa_updated
    BEFORE UPDATE ON pack040_mv.mv_persistence_alerts
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_persistence_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_degradation_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_recommissioning_triggers ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_performance_guarantees ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_persistence_alerts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_pr_tenant_isolation
    ON pack040_mv.mv_persistence_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_pr_service_bypass
    ON pack040_mv.mv_persistence_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_da_tenant_isolation
    ON pack040_mv.mv_degradation_analysis
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_da_service_bypass
    ON pack040_mv.mv_degradation_analysis
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_rt_tenant_isolation
    ON pack040_mv.mv_recommissioning_triggers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_rt_service_bypass
    ON pack040_mv.mv_recommissioning_triggers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_pg_tenant_isolation
    ON pack040_mv.mv_performance_guarantees
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_pg_service_bypass
    ON pack040_mv.mv_performance_guarantees
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_pa_tenant_isolation
    ON pack040_mv.mv_persistence_alerts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_pa_service_bypass
    ON pack040_mv.mv_persistence_alerts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_persistence_records TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_degradation_analysis TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_recommissioning_triggers TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_performance_guarantees TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_persistence_alerts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.mv_persistence_records IS
    'Year-over-year savings persistence tracking with persistence factors, trend analysis, and degradation detection.';
COMMENT ON TABLE pack040_mv.mv_degradation_analysis IS
    'Savings degradation modeling (linear, exponential, step-change) with projections and re-commissioning recommendations.';
COMMENT ON TABLE pack040_mv.mv_recommissioning_triggers IS
    'Re-commissioning trigger events when savings fall below thresholds, with recommended actions and resolution tracking.';
COMMENT ON TABLE pack040_mv.mv_performance_guarantees IS
    'ESCO/EPC performance guarantee tracking with guaranteed vs. actual savings, shortfall calculations, and settlement.';
COMMENT ON TABLE pack040_mv.mv_persistence_alerts IS
    'Alert records for persistence monitoring with multi-channel notification, escalation, and resolution tracking.';

COMMENT ON COLUMN pack040_mv.mv_persistence_records.persistence_factor IS 'Ratio of actual to expected savings (1.0 = full persistence, <0.80 triggers re-commissioning).';
COMMENT ON COLUMN pack040_mv.mv_persistence_records.trend_direction IS 'Savings trend: IMPROVING, STABLE, SLIGHT_DECLINE, DECLINING, RAPID_DECLINE, STEP_CHANGE.';
COMMENT ON COLUMN pack040_mv.mv_persistence_records.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_degradation_analysis.degradation_rate_pct_year IS 'Annual savings degradation rate as percentage of Year 1 savings.';
COMMENT ON COLUMN pack040_mv.mv_degradation_analysis.exponential_half_life_years IS 'Years until savings degrade to 50% of initial value (exponential model).';
COMMENT ON COLUMN pack040_mv.mv_degradation_analysis.primary_degradation_cause IS 'Root cause: EQUIPMENT_AGING, MAINTENANCE_NEGLECT, OPERATIONAL_DRIFT, BEHAVIORAL_REVERSION, etc.';

COMMENT ON COLUMN pack040_mv.mv_performance_guarantees.performance_ratio IS 'Actual savings / guaranteed savings (>1.0 = exceeding guarantee).';
COMMENT ON COLUMN pack040_mv.mv_performance_guarantees.settlement_method IS 'Contract settlement type: GUARANTEED_SAVINGS, SHARED_SAVINGS, PERFORMANCE_BONUS.';

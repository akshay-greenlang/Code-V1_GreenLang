-- =============================================================================
-- V292: PACK-037 Demand Response Pack - Performance Tracking
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    007 of 010
-- Date:         March 2026
--
-- Performance tracking and compliance reporting tables for ongoing DR
-- program participation. Covers per-event performance records, seasonal
-- summaries, multi-year trend analysis, compliance reports, and
-- performance alert management.
--
-- Tables (5):
--   1. pack037_demand_response.dr_performance_events
--   2. pack037_demand_response.dr_season_summaries
--   3. pack037_demand_response.dr_performance_trends
--   4. pack037_demand_response.dr_compliance_reports
--   5. pack037_demand_response.dr_performance_alerts
--
-- Previous: V291__pack037_demand_response_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.dr_performance_events
-- =============================================================================
-- Consolidated performance scorecard for each DR event combining baseline,
-- actual delivery, settlement, payment, and penalty outcomes.

CREATE TABLE pack037_demand_response.dr_performance_events (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID            NOT NULL REFERENCES pack037_demand_response.dr_events(id) ON DELETE CASCADE,
    enrollment_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_program_enrollment(id),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    event_date              DATE            NOT NULL,
    event_type              VARCHAR(30)     NOT NULL,
    program_code            VARCHAR(50)     NOT NULL,
    committed_kw            NUMERIC(12,4)   NOT NULL,
    baseline_kw             NUMERIC(12,4)   NOT NULL,
    actual_demand_kw        NUMERIC(12,4)   NOT NULL,
    delivered_kw            NUMERIC(12,4)   NOT NULL,
    delivered_mwh           NUMERIC(14,6),
    performance_ratio       NUMERIC(6,4)    NOT NULL,
    compliance_flag         BOOLEAN         NOT NULL,
    response_time_min       NUMERIC(8,2),
    event_duration_min      NUMERIC(8,2),
    data_quality_score      NUMERIC(5,2),
    payment_earned          NUMERIC(14,2),
    penalty_incurred        NUMERIC(14,2)   DEFAULT 0,
    net_revenue             NUMERIC(14,2),
    currency_code           CHAR(3)         DEFAULT 'USD',
    weather_temperature_f   NUMERIC(6,2),
    score_rank              INTEGER,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_pev_type CHECK (
        event_type IN (
            'CAPACITY', 'ENERGY', 'EMERGENCY', 'TEST', 'ECONOMIC',
            'RELIABILITY', 'ANCILLARY', 'VOLUNTARY'
        )
    ),
    CONSTRAINT chk_p037_pev_committed CHECK (
        committed_kw > 0
    ),
    CONSTRAINT chk_p037_pev_baseline CHECK (
        baseline_kw >= 0
    ),
    CONSTRAINT chk_p037_pev_actual CHECK (
        actual_demand_kw >= 0
    ),
    CONSTRAINT chk_p037_pev_delivered CHECK (
        delivered_kw >= 0
    ),
    CONSTRAINT chk_p037_pev_perf CHECK (
        performance_ratio >= 0
    ),
    CONSTRAINT chk_p037_pev_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT uq_p037_pev_event UNIQUE (event_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_pev_event          ON pack037_demand_response.dr_performance_events(event_id);
CREATE INDEX idx_p037_pev_enrollment     ON pack037_demand_response.dr_performance_events(enrollment_id);
CREATE INDEX idx_p037_pev_facility       ON pack037_demand_response.dr_performance_events(facility_profile_id);
CREATE INDEX idx_p037_pev_tenant         ON pack037_demand_response.dr_performance_events(tenant_id);
CREATE INDEX idx_p037_pev_date           ON pack037_demand_response.dr_performance_events(event_date DESC);
CREATE INDEX idx_p037_pev_type           ON pack037_demand_response.dr_performance_events(event_type);
CREATE INDEX idx_p037_pev_program        ON pack037_demand_response.dr_performance_events(program_code);
CREATE INDEX idx_p037_pev_compliance     ON pack037_demand_response.dr_performance_events(compliance_flag);
CREATE INDEX idx_p037_pev_perf           ON pack037_demand_response.dr_performance_events(performance_ratio DESC);
CREATE INDEX idx_p037_pev_revenue        ON pack037_demand_response.dr_performance_events(net_revenue DESC);
CREATE INDEX idx_p037_pev_created        ON pack037_demand_response.dr_performance_events(created_at DESC);

-- Composite: facility + program + date for performance history
CREATE INDEX idx_p037_pev_fac_prg_date   ON pack037_demand_response.dr_performance_events(facility_profile_id, program_code, event_date DESC);

-- Non-compliant events for monitoring
CREATE INDEX idx_p037_pev_non_compliant  ON pack037_demand_response.dr_performance_events(event_date DESC)
    WHERE compliance_flag = false;

-- =============================================================================
-- Table 2: pack037_demand_response.dr_season_summaries
-- =============================================================================
-- Aggregated seasonal performance summaries for each facility enrollment.
-- Captures total events, delivery rates, revenue, penalties, and
-- overall compliance across a DR season.

CREATE TABLE pack037_demand_response.dr_season_summaries (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_program_enrollment(id),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    season_year             INTEGER         NOT NULL,
    season                  VARCHAR(20)     NOT NULL,
    program_code            VARCHAR(50)     NOT NULL,
    committed_kw            NUMERIC(12,4)   NOT NULL,
    total_events            INTEGER         NOT NULL DEFAULT 0,
    events_responded        INTEGER         NOT NULL DEFAULT 0,
    events_compliant        INTEGER         NOT NULL DEFAULT 0,
    events_non_compliant    INTEGER         NOT NULL DEFAULT 0,
    events_opted_out        INTEGER         NOT NULL DEFAULT 0,
    events_cancelled        INTEGER         NOT NULL DEFAULT 0,
    avg_performance_ratio   NUMERIC(6,4),
    min_performance_ratio   NUMERIC(6,4),
    max_performance_ratio   NUMERIC(6,4),
    total_curtailment_mwh   NUMERIC(14,4),
    avg_response_time_min   NUMERIC(8,2),
    total_hours_curtailed   NUMERIC(10,2),
    availability_pct        NUMERIC(5,2),
    compliance_rate_pct     NUMERIC(5,2),
    total_revenue           NUMERIC(14,2),
    total_penalties         NUMERIC(14,2)   DEFAULT 0,
    net_revenue             NUMERIC(14,2),
    revenue_per_kw          NUMERIC(10,4),
    currency_code           CHAR(3)         DEFAULT 'USD',
    season_grade            CHAR(2),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_ss_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p037_ss_year CHECK (
        season_year >= 2020 AND season_year <= 2100
    ),
    CONSTRAINT chk_p037_ss_committed CHECK (
        committed_kw > 0
    ),
    CONSTRAINT chk_p037_ss_events CHECK (
        total_events >= 0
    ),
    CONSTRAINT chk_p037_ss_responded CHECK (
        events_responded >= 0 AND events_responded <= total_events
    ),
    CONSTRAINT chk_p037_ss_compliant CHECK (
        events_compliant >= 0
    ),
    CONSTRAINT chk_p037_ss_availability CHECK (
        availability_pct IS NULL OR (availability_pct >= 0 AND availability_pct <= 100)
    ),
    CONSTRAINT chk_p037_ss_compliance_rate CHECK (
        compliance_rate_pct IS NULL OR (compliance_rate_pct >= 0 AND compliance_rate_pct <= 100)
    ),
    CONSTRAINT chk_p037_ss_grade CHECK (
        season_grade IS NULL OR season_grade IN ('A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F')
    ),
    CONSTRAINT uq_p037_ss_enrollment_season UNIQUE (enrollment_id, season_year, season)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_ss_enrollment      ON pack037_demand_response.dr_season_summaries(enrollment_id);
CREATE INDEX idx_p037_ss_facility        ON pack037_demand_response.dr_season_summaries(facility_profile_id);
CREATE INDEX idx_p037_ss_tenant          ON pack037_demand_response.dr_season_summaries(tenant_id);
CREATE INDEX idx_p037_ss_season          ON pack037_demand_response.dr_season_summaries(season_year, season);
CREATE INDEX idx_p037_ss_program         ON pack037_demand_response.dr_season_summaries(program_code);
CREATE INDEX idx_p037_ss_compliance      ON pack037_demand_response.dr_season_summaries(compliance_rate_pct DESC);
CREATE INDEX idx_p037_ss_revenue         ON pack037_demand_response.dr_season_summaries(net_revenue DESC);
CREATE INDEX idx_p037_ss_grade           ON pack037_demand_response.dr_season_summaries(season_grade);
CREATE INDEX idx_p037_ss_created         ON pack037_demand_response.dr_season_summaries(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_ss_updated
    BEFORE UPDATE ON pack037_demand_response.dr_season_summaries
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack037_demand_response.dr_performance_trends
-- =============================================================================
-- Multi-year performance trend analysis tracking rolling averages,
-- year-over-year changes, and trajectory metrics for strategic
-- decision-making and continuous improvement.

CREATE TABLE pack037_demand_response.dr_performance_trends (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    trend_period            VARCHAR(20)     NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    metric_name             VARCHAR(50)     NOT NULL,
    metric_value            NUMERIC(14,4)   NOT NULL,
    previous_period_value   NUMERIC(14,4),
    change_pct              NUMERIC(8,4),
    rolling_avg_3_period    NUMERIC(14,4),
    rolling_avg_12_period   NUMERIC(14,4),
    trend_direction         VARCHAR(10),
    trend_significance      VARCHAR(20),
    forecast_next_period    NUMERIC(14,4),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_pt_period CHECK (
        trend_period IN ('MONTHLY', 'QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL')
    ),
    CONSTRAINT chk_p037_pt_dates CHECK (
        period_end > period_start
    ),
    CONSTRAINT chk_p037_pt_metric CHECK (
        metric_name IN (
            'AVG_PERFORMANCE_RATIO', 'COMPLIANCE_RATE', 'AVAILABILITY_RATE',
            'AVG_RESPONSE_TIME', 'TOTAL_CURTAILMENT_MWH', 'NET_REVENUE',
            'REVENUE_PER_KW', 'PENALTY_RATE', 'EVENTS_PER_PERIOD',
            'AVG_CURTAILMENT_KW', 'DATA_QUALITY_SCORE'
        )
    ),
    CONSTRAINT chk_p037_pt_direction CHECK (
        trend_direction IS NULL OR trend_direction IN ('UP', 'DOWN', 'FLAT')
    ),
    CONSTRAINT chk_p037_pt_significance CHECK (
        trend_significance IS NULL OR trend_significance IN (
            'STRONG_POSITIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'STRONG_NEGATIVE'
        )
    ),
    CONSTRAINT uq_p037_pt_fac_period_metric UNIQUE (facility_profile_id, trend_period, period_start, metric_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_pt_facility        ON pack037_demand_response.dr_performance_trends(facility_profile_id);
CREATE INDEX idx_p037_pt_tenant          ON pack037_demand_response.dr_performance_trends(tenant_id);
CREATE INDEX idx_p037_pt_period          ON pack037_demand_response.dr_performance_trends(trend_period);
CREATE INDEX idx_p037_pt_start           ON pack037_demand_response.dr_performance_trends(period_start DESC);
CREATE INDEX idx_p037_pt_metric          ON pack037_demand_response.dr_performance_trends(metric_name);
CREATE INDEX idx_p037_pt_direction       ON pack037_demand_response.dr_performance_trends(trend_direction);
CREATE INDEX idx_p037_pt_created         ON pack037_demand_response.dr_performance_trends(created_at DESC);

-- Composite: facility + metric + period for trend line queries
CREATE INDEX idx_p037_pt_fac_metric      ON pack037_demand_response.dr_performance_trends(facility_profile_id, metric_name, period_start DESC);

-- =============================================================================
-- Table 4: pack037_demand_response.dr_compliance_reports
-- =============================================================================
-- Generated compliance reports for DR programs submitted to ISOs, RTOs,
-- utilities, and regulators. Tracks report generation, submission,
-- and approval workflow.

CREATE TABLE pack037_demand_response.dr_compliance_reports (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_program_enrollment(id),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    report_type             VARCHAR(50)     NOT NULL,
    report_period           VARCHAR(20)     NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    program_code            VARCHAR(50)     NOT NULL,
    report_title            VARCHAR(255)    NOT NULL,
    summary                 TEXT,
    committed_kw            NUMERIC(12,4),
    total_events            INTEGER,
    compliance_rate_pct     NUMERIC(5,2),
    total_curtailment_mwh   NUMERIC(14,4),
    total_revenue           NUMERIC(14,2),
    total_penalties         NUMERIC(14,2),
    report_status           VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    submitted_at            TIMESTAMPTZ,
    submitted_to            VARCHAR(255),
    approved_at             TIMESTAMPTZ,
    approved_by             VARCHAR(255),
    report_file_path        TEXT,
    report_format           VARCHAR(20)     DEFAULT 'PDF',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_cr_type CHECK (
        report_type IN (
            'MONTHLY_PERFORMANCE', 'SEASONAL_SUMMARY', 'ANNUAL_COMPLIANCE',
            'SETTLEMENT_REPORT', 'TEST_RESULTS', 'CAPACITY_VERIFICATION',
            'INCIDENT_REPORT', 'AUDIT_RESPONSE', 'REGULATORY_FILING'
        )
    ),
    CONSTRAINT chk_p037_cr_period CHECK (
        report_period IN ('MONTHLY', 'QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL', 'AD_HOC')
    ),
    CONSTRAINT chk_p037_cr_dates CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p037_cr_status CHECK (
        report_status IN (
            'DRAFT', 'REVIEW', 'APPROVED', 'SUBMITTED', 'ACCEPTED',
            'REJECTED', 'REVISION_REQUESTED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p037_cr_format CHECK (
        report_format IS NULL OR report_format IN ('PDF', 'EXCEL', 'CSV', 'XML', 'JSON')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_cr_enrollment      ON pack037_demand_response.dr_compliance_reports(enrollment_id);
CREATE INDEX idx_p037_cr_facility        ON pack037_demand_response.dr_compliance_reports(facility_profile_id);
CREATE INDEX idx_p037_cr_tenant          ON pack037_demand_response.dr_compliance_reports(tenant_id);
CREATE INDEX idx_p037_cr_type            ON pack037_demand_response.dr_compliance_reports(report_type);
CREATE INDEX idx_p037_cr_program         ON pack037_demand_response.dr_compliance_reports(program_code);
CREATE INDEX idx_p037_cr_status          ON pack037_demand_response.dr_compliance_reports(report_status);
CREATE INDEX idx_p037_cr_period_start    ON pack037_demand_response.dr_compliance_reports(period_start DESC);
CREATE INDEX idx_p037_cr_generated       ON pack037_demand_response.dr_compliance_reports(generated_at DESC);
CREATE INDEX idx_p037_cr_created         ON pack037_demand_response.dr_compliance_reports(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_cr_updated
    BEFORE UPDATE ON pack037_demand_response.dr_compliance_reports
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack037_demand_response.dr_performance_alerts
-- =============================================================================
-- Performance alerts triggered when DR metrics breach thresholds.
-- Covers under-performance, compliance risk, equipment issues,
-- and revenue impact warnings.

CREATE TABLE pack037_demand_response.dr_performance_alerts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    enrollment_id           UUID            REFERENCES pack037_demand_response.dr_program_enrollment(id),
    event_id                UUID            REFERENCES pack037_demand_response.dr_events(id),
    tenant_id               UUID            NOT NULL,
    alert_type              VARCHAR(50)     NOT NULL,
    severity                VARCHAR(10)     NOT NULL,
    alert_title             VARCHAR(255)    NOT NULL,
    alert_message           TEXT            NOT NULL,
    metric_name             VARCHAR(50),
    metric_value            NUMERIC(14,4),
    threshold_value         NUMERIC(14,4),
    breach_direction        VARCHAR(10),
    recommended_action      TEXT,
    acknowledged            BOOLEAN         DEFAULT false,
    acknowledged_by         VARCHAR(255),
    acknowledged_at         TIMESTAMPTZ,
    resolved                BOOLEAN         DEFAULT false,
    resolved_at             TIMESTAMPTZ,
    resolution_notes        TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_pa_type CHECK (
        alert_type IN (
            'UNDER_PERFORMANCE', 'NON_COMPLIANCE', 'CAPACITY_SHORTFALL',
            'RESPONSE_TIME_BREACH', 'TELEMETRY_FAILURE', 'EQUIPMENT_FAULT',
            'BASELINE_ANOMALY', 'REVENUE_RISK', 'CONTRACT_EXPIRY',
            'DER_SOC_LOW', 'DER_DEGRADATION', 'DATA_QUALITY'
        )
    ),
    CONSTRAINT chk_p037_pa_severity CHECK (
        severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p037_pa_breach CHECK (
        breach_direction IS NULL OR breach_direction IN ('ABOVE', 'BELOW')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_pa_facility        ON pack037_demand_response.dr_performance_alerts(facility_profile_id);
CREATE INDEX idx_p037_pa_enrollment      ON pack037_demand_response.dr_performance_alerts(enrollment_id);
CREATE INDEX idx_p037_pa_event           ON pack037_demand_response.dr_performance_alerts(event_id);
CREATE INDEX idx_p037_pa_tenant          ON pack037_demand_response.dr_performance_alerts(tenant_id);
CREATE INDEX idx_p037_pa_type            ON pack037_demand_response.dr_performance_alerts(alert_type);
CREATE INDEX idx_p037_pa_severity        ON pack037_demand_response.dr_performance_alerts(severity);
CREATE INDEX idx_p037_pa_acknowledged    ON pack037_demand_response.dr_performance_alerts(acknowledged);
CREATE INDEX idx_p037_pa_resolved        ON pack037_demand_response.dr_performance_alerts(resolved);
CREATE INDEX idx_p037_pa_created         ON pack037_demand_response.dr_performance_alerts(created_at DESC);

-- Composite: unresolved alerts by severity for operations dashboard
CREATE INDEX idx_p037_pa_unresolved      ON pack037_demand_response.dr_performance_alerts(severity, created_at DESC)
    WHERE resolved = false;

-- Composite: facility + unacknowledged for notification queue
CREATE INDEX idx_p037_pa_fac_unack       ON pack037_demand_response.dr_performance_alerts(facility_profile_id, severity)
    WHERE acknowledged = false AND resolved = false;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_pa_updated
    BEFORE UPDATE ON pack037_demand_response.dr_performance_alerts
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_performance_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_season_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_performance_trends ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_compliance_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_performance_alerts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p037_pev_tenant_isolation ON pack037_demand_response.dr_performance_events
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_pev_service_bypass ON pack037_demand_response.dr_performance_events
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_ss_tenant_isolation ON pack037_demand_response.dr_season_summaries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_ss_service_bypass ON pack037_demand_response.dr_season_summaries
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_pt_tenant_isolation ON pack037_demand_response.dr_performance_trends
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_pt_service_bypass ON pack037_demand_response.dr_performance_trends
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_cr_tenant_isolation ON pack037_demand_response.dr_compliance_reports
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_cr_service_bypass ON pack037_demand_response.dr_compliance_reports
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_pa_tenant_isolation ON pack037_demand_response.dr_performance_alerts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_pa_service_bypass ON pack037_demand_response.dr_performance_alerts
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_performance_events TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_season_summaries TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_performance_trends TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_compliance_reports TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_performance_alerts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.dr_performance_events IS
    'Consolidated performance scorecard per DR event with baseline, delivery, settlement, payment, and penalty outcomes.';
COMMENT ON TABLE pack037_demand_response.dr_season_summaries IS
    'Aggregated seasonal performance summaries per enrollment with total events, delivery rates, revenue, and compliance.';
COMMENT ON TABLE pack037_demand_response.dr_performance_trends IS
    'Multi-year performance trend analysis with rolling averages, year-over-year changes, and trajectory metrics.';
COMMENT ON TABLE pack037_demand_response.dr_compliance_reports IS
    'Generated compliance reports for DR programs submitted to ISOs, RTOs, utilities, and regulators.';
COMMENT ON TABLE pack037_demand_response.dr_performance_alerts IS
    'Performance alerts triggered when DR metrics breach thresholds for under-performance, compliance risk, or equipment issues.';

COMMENT ON COLUMN pack037_demand_response.dr_performance_events.performance_ratio IS 'Ratio of delivered to committed kW (1.0 = 100% delivery).';
COMMENT ON COLUMN pack037_demand_response.dr_performance_events.net_revenue IS 'Net revenue after penalties (payment_earned - penalty_incurred).';
COMMENT ON COLUMN pack037_demand_response.dr_performance_events.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack037_demand_response.dr_season_summaries.season_grade IS 'Performance grade for the season: A+, A, A-, B+, B, B-, C+, C, C-, D, F.';
COMMENT ON COLUMN pack037_demand_response.dr_season_summaries.revenue_per_kw IS 'Net revenue per committed kW for ROI analysis.';
COMMENT ON COLUMN pack037_demand_response.dr_season_summaries.availability_pct IS 'Percentage of time the facility was available for DR dispatch.';

COMMENT ON COLUMN pack037_demand_response.dr_performance_trends.trend_direction IS 'Direction of the trend: UP (improving), DOWN (declining), FLAT (stable).';
COMMENT ON COLUMN pack037_demand_response.dr_performance_trends.trend_significance IS 'Significance level of the trend: STRONG_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, STRONG_NEGATIVE.';

COMMENT ON COLUMN pack037_demand_response.dr_compliance_reports.report_type IS 'Report type: MONTHLY_PERFORMANCE, SEASONAL_SUMMARY, ANNUAL_COMPLIANCE, SETTLEMENT_REPORT, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_compliance_reports.report_status IS 'Report workflow: DRAFT, REVIEW, APPROVED, SUBMITTED, ACCEPTED, REJECTED, etc.';

COMMENT ON COLUMN pack037_demand_response.dr_performance_alerts.alert_type IS 'Alert category: UNDER_PERFORMANCE, NON_COMPLIANCE, CAPACITY_SHORTFALL, DER_SOC_LOW, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_performance_alerts.severity IS 'Alert severity: LOW, MEDIUM, HIGH, CRITICAL.';

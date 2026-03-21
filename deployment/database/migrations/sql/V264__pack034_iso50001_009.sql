-- =============================================================================
-- V264: PACK-034 ISO 50001 Energy Management System - Performance Tables
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    009 of 010
-- Date:         March 2026
--
-- Creates performance reporting, trend analysis, and M&V verification tables
-- per ISO 50001 Clause 9.1 and IPMVP Options A-D. Supports management review
-- data requirements and energy performance improvement demonstration.
--
-- Tables (3):
--   1. pack034_iso50001.performance_reports
--   2. pack034_iso50001.trend_analyses
--   3. pack034_iso50001.verification_results
--
-- Previous: V263__pack034_iso50001_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.performance_reports
-- =============================================================================
-- Periodic performance reports summarizing energy consumption, cost, emissions,
-- and EnPI performance for management review and continual improvement.

CREATE TABLE pack034_iso50001.performance_reports (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    report_period_start         DATE            NOT NULL,
    report_period_end           DATE            NOT NULL,
    report_type                 VARCHAR(30)     NOT NULL,
    total_energy_kwh            DECIMAL(18,4),
    total_cost                  DECIMAL(14,2),
    total_emissions_tco2e       DECIMAL(14,4),
    enpi_summary_json           JSONB           DEFAULT '{}',
    generated_by                UUID,
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_pr_type CHECK (
        report_type IN ('monthly', 'quarterly', 'annual', 'management_review')
    ),
    CONSTRAINT chk_p034_pr_period CHECK (
        report_period_start < report_period_end
    ),
    CONSTRAINT chk_p034_pr_energy CHECK (
        total_energy_kwh IS NULL OR total_energy_kwh >= 0
    ),
    CONSTRAINT chk_p034_pr_cost CHECK (
        total_cost IS NULL OR total_cost >= 0
    ),
    CONSTRAINT chk_p034_pr_emissions CHECK (
        total_emissions_tco2e IS NULL OR total_emissions_tco2e >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_pr_enms            ON pack034_iso50001.performance_reports(enms_id);
CREATE INDEX idx_p034_pr_period          ON pack034_iso50001.performance_reports(report_period_start, report_period_end);
CREATE INDEX idx_p034_pr_type            ON pack034_iso50001.performance_reports(report_type);
CREATE INDEX idx_p034_pr_energy          ON pack034_iso50001.performance_reports(total_energy_kwh);
CREATE INDEX idx_p034_pr_generated       ON pack034_iso50001.performance_reports(generated_at DESC);
CREATE INDEX idx_p034_pr_enpi            ON pack034_iso50001.performance_reports USING GIN(enpi_summary_json);
CREATE INDEX idx_p034_pr_created         ON pack034_iso50001.performance_reports(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_pr_updated
    BEFORE UPDATE ON pack034_iso50001.performance_reports
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.trend_analyses
-- =============================================================================
-- Trend analysis records for year-over-year, rolling 12-month, regression,
-- and forecast analyses of energy performance metrics.

CREATE TABLE pack034_iso50001.trend_analyses (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id                   UUID            NOT NULL REFERENCES pack034_iso50001.performance_reports(id) ON DELETE CASCADE,
    analysis_type               VARCHAR(20)     NOT NULL,
    metric_name                 VARCHAR(255)    NOT NULL,
    period_count                INTEGER         NOT NULL,
    trend_direction             VARCHAR(20)     NOT NULL,
    trend_slope                 DECIMAL(12,8),
    confidence_pct              DECIMAL(6,2),
    data_points_json            JSONB           DEFAULT '[]',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_ta_type CHECK (
        analysis_type IN ('yoy', 'rolling12', 'regression', 'forecast')
    ),
    CONSTRAINT chk_p034_ta_direction CHECK (
        trend_direction IN ('improving', 'stable', 'degrading')
    ),
    CONSTRAINT chk_p034_ta_period CHECK (
        period_count > 0
    ),
    CONSTRAINT chk_p034_ta_confidence CHECK (
        confidence_pct IS NULL OR (confidence_pct >= 0 AND confidence_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_ta_report          ON pack034_iso50001.trend_analyses(report_id);
CREATE INDEX idx_p034_ta_type            ON pack034_iso50001.trend_analyses(analysis_type);
CREATE INDEX idx_p034_ta_metric          ON pack034_iso50001.trend_analyses(metric_name);
CREATE INDEX idx_p034_ta_direction       ON pack034_iso50001.trend_analyses(trend_direction);
CREATE INDEX idx_p034_ta_confidence      ON pack034_iso50001.trend_analyses(confidence_pct DESC);
CREATE INDEX idx_p034_ta_data_points     ON pack034_iso50001.trend_analyses USING GIN(data_points_json);
CREATE INDEX idx_p034_ta_created         ON pack034_iso50001.trend_analyses(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_ta_updated
    BEFORE UPDATE ON pack034_iso50001.trend_analyses
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.verification_results
-- =============================================================================
-- Measurement and verification (M&V) results per IPMVP Options A-D,
-- calculating verified energy savings with uncertainty quantification.

CREATE TABLE pack034_iso50001.verification_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id                   UUID            NOT NULL REFERENCES pack034_iso50001.performance_reports(id) ON DELETE CASCADE,
    verification_method         VARCHAR(20)     NOT NULL,
    baseline_consumption        DECIMAL(18,4)   NOT NULL,
    reporting_consumption       DECIMAL(18,4)   NOT NULL,
    adjusted_consumption        DECIMAL(18,4),
    verified_savings_kwh        DECIMAL(18,4)   NOT NULL,
    uncertainty_pct             DECIMAL(8,4),
    confidence_level            DECIMAL(6,2),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_vr_method CHECK (
        verification_method IN ('option_a', 'option_b', 'option_c', 'option_d')
    ),
    CONSTRAINT chk_p034_vr_baseline CHECK (
        baseline_consumption >= 0
    ),
    CONSTRAINT chk_p034_vr_reporting CHECK (
        reporting_consumption >= 0
    ),
    CONSTRAINT chk_p034_vr_adjusted CHECK (
        adjusted_consumption IS NULL OR adjusted_consumption >= 0
    ),
    CONSTRAINT chk_p034_vr_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100)
    ),
    CONSTRAINT chk_p034_vr_confidence CHECK (
        confidence_level IS NULL OR (confidence_level >= 0 AND confidence_level <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_vr_report          ON pack034_iso50001.verification_results(report_id);
CREATE INDEX idx_p034_vr_method          ON pack034_iso50001.verification_results(verification_method);
CREATE INDEX idx_p034_vr_savings         ON pack034_iso50001.verification_results(verified_savings_kwh DESC);
CREATE INDEX idx_p034_vr_uncertainty     ON pack034_iso50001.verification_results(uncertainty_pct);
CREATE INDEX idx_p034_vr_confidence      ON pack034_iso50001.verification_results(confidence_level DESC);
CREATE INDEX idx_p034_vr_created         ON pack034_iso50001.verification_results(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_vr_updated
    BEFORE UPDATE ON pack034_iso50001.verification_results
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.performance_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.trend_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.verification_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_pr_tenant_isolation
    ON pack034_iso50001.performance_reports
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_pr_service_bypass
    ON pack034_iso50001.performance_reports
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_ta_tenant_isolation
    ON pack034_iso50001.trend_analyses
    USING (report_id IN (
        SELECT id FROM pack034_iso50001.performance_reports
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_ta_service_bypass
    ON pack034_iso50001.trend_analyses
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_vr_tenant_isolation
    ON pack034_iso50001.verification_results
    USING (report_id IN (
        SELECT id FROM pack034_iso50001.performance_reports
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_vr_service_bypass
    ON pack034_iso50001.verification_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.performance_reports TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.trend_analyses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.verification_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.performance_reports IS
    'Periodic performance reports summarizing energy consumption, cost, emissions, and EnPI performance for management review.';

COMMENT ON TABLE pack034_iso50001.trend_analyses IS
    'Trend analysis records for YoY, rolling 12-month, regression, and forecast analyses of energy metrics.';

COMMENT ON TABLE pack034_iso50001.verification_results IS
    'M&V results per IPMVP Options A-D calculating verified energy savings with uncertainty quantification.';

COMMENT ON COLUMN pack034_iso50001.performance_reports.report_type IS
    'Report type: monthly (operational), quarterly (management), annual (strategic), management_review (ISO 50001 Clause 9.3).';
COMMENT ON COLUMN pack034_iso50001.performance_reports.enpi_summary_json IS
    'JSON summary of all EnPI values for the reporting period (e.g., {"enpi_name": {"value": 1.5, "target": 1.3}}).';
COMMENT ON COLUMN pack034_iso50001.performance_reports.total_emissions_tco2e IS
    'Total greenhouse gas emissions in tonnes CO2e associated with energy consumption.';
COMMENT ON COLUMN pack034_iso50001.trend_analyses.analysis_type IS
    'Analysis type: yoy (year-over-year), rolling12 (rolling 12-month), regression (statistical), forecast (predictive).';
COMMENT ON COLUMN pack034_iso50001.trend_analyses.trend_direction IS
    'Trend direction: improving (energy performance getting better), stable, degrading (getting worse).';
COMMENT ON COLUMN pack034_iso50001.trend_analyses.trend_slope IS
    'Slope of the trend line. Negative = improving for consumption metrics.';
COMMENT ON COLUMN pack034_iso50001.trend_analyses.data_points_json IS
    'JSON array of data points used in the trend analysis (e.g., [{"date": "2026-01", "value": 1500}]).';
COMMENT ON COLUMN pack034_iso50001.verification_results.verification_method IS
    'IPMVP Option: A (retrofit isolation, key parameter), B (retrofit isolation, all parameters), C (whole facility), D (calibrated simulation).';
COMMENT ON COLUMN pack034_iso50001.verification_results.verified_savings_kwh IS
    'Verified energy savings in kWh (baseline adjusted - reporting consumption).';
COMMENT ON COLUMN pack034_iso50001.verification_results.uncertainty_pct IS
    'Uncertainty of the savings estimate as a percentage (e.g., 10 = +/- 10%).';
COMMENT ON COLUMN pack034_iso50001.verification_results.confidence_level IS
    'Statistical confidence level for the savings estimate (e.g., 90 = 90% confidence).';

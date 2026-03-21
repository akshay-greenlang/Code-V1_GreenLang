-- =============================================================================
-- V247: PACK-033 Quick Wins Identifier - Financial Analysis
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    002 of 010
-- Date:         March 2026
--
-- Creates financial analysis tables for payback calculations, NPV, IRR, and
-- scenario modeling of quick-win energy efficiency actions.
--
-- Tables (2):
--   1. pack033_quick_wins.payback_analyses
--   2. pack033_quick_wins.financial_scenarios
--
-- Previous: V246__pack033_quick_wins_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.payback_analyses
-- =============================================================================
-- Detailed financial analysis per action including simple/discounted payback,
-- NPV, IRR, ROI, LCOE, and tax incentive adjustments.

CREATE TABLE pack033_quick_wins.payback_analyses (
    analysis_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                 UUID            NOT NULL REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE CASCADE,
    action_id               UUID,
    implementation_cost     NUMERIC(14,2)   NOT NULL,
    annual_savings          NUMERIC(14,2)   NOT NULL,
    simple_payback_years    NUMERIC(8,2),
    discounted_payback_years NUMERIC(8,2),
    npv                     NUMERIC(16,2),
    irr                     NUMERIC(8,4),
    roi_pct                 NUMERIC(8,2),
    lcoe                    NUMERIC(10,4),
    discount_rate           NUMERIC(6,4)    NOT NULL DEFAULT 0.08,
    analysis_period_years   INTEGER         NOT NULL DEFAULT 10,
    utility_escalation_rate NUMERIC(6,4)    DEFAULT 0.03,
    tax_incentive_amount    NUMERIC(14,2)   DEFAULT 0,
    depreciation_benefit    NUMERIC(14,2)   DEFAULT 0,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_pa_impl_cost CHECK (
        implementation_cost >= 0
    ),
    CONSTRAINT chk_p033_pa_annual_savings CHECK (
        annual_savings >= 0
    ),
    CONSTRAINT chk_p033_pa_simple_payback CHECK (
        simple_payback_years IS NULL OR simple_payback_years >= 0
    ),
    CONSTRAINT chk_p033_pa_disc_payback CHECK (
        discounted_payback_years IS NULL OR discounted_payback_years >= 0
    ),
    CONSTRAINT chk_p033_pa_discount_rate CHECK (
        discount_rate >= 0 AND discount_rate <= 1
    ),
    CONSTRAINT chk_p033_pa_analysis_period CHECK (
        analysis_period_years >= 1 AND analysis_period_years <= 50
    ),
    CONSTRAINT chk_p033_pa_escalation CHECK (
        utility_escalation_rate IS NULL OR (utility_escalation_rate >= -0.1 AND utility_escalation_rate <= 0.5)
    ),
    CONSTRAINT chk_p033_pa_tax_incentive CHECK (
        tax_incentive_amount IS NULL OR tax_incentive_amount >= 0
    ),
    CONSTRAINT chk_p033_pa_depreciation CHECK (
        depreciation_benefit IS NULL OR depreciation_benefit >= 0
    ),
    CONSTRAINT chk_p033_pa_irr CHECK (
        irr IS NULL OR (irr >= -1 AND irr <= 10)
    ),
    CONSTRAINT chk_p033_pa_roi CHECK (
        roi_pct IS NULL OR roi_pct >= -100
    ),
    CONSTRAINT chk_p033_pa_lcoe CHECK (
        lcoe IS NULL OR lcoe >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_pa_scan          ON pack033_quick_wins.payback_analyses(scan_id);
CREATE INDEX idx_p033_pa_action        ON pack033_quick_wins.payback_analyses(action_id);
CREATE INDEX idx_p033_pa_npv           ON pack033_quick_wins.payback_analyses(npv DESC);
CREATE INDEX idx_p033_pa_irr           ON pack033_quick_wins.payback_analyses(irr DESC);
CREATE INDEX idx_p033_pa_payback       ON pack033_quick_wins.payback_analyses(simple_payback_years);
CREATE INDEX idx_p033_pa_created       ON pack033_quick_wins.payback_analyses(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_pa_updated
    BEFORE UPDATE ON pack033_quick_wins.payback_analyses
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.financial_scenarios
-- =============================================================================
-- What-if financial scenarios for sensitivity analysis, each linked to a
-- payback analysis with varied discount/escalation/inflation rates.

CREATE TABLE pack033_quick_wins.financial_scenarios (
    scenario_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id             UUID            NOT NULL REFERENCES pack033_quick_wins.payback_analyses(analysis_id) ON DELETE CASCADE,
    scenario_name           VARCHAR(255)    NOT NULL,
    discount_rate           NUMERIC(6,4)    NOT NULL,
    escalation_rate         NUMERIC(6,4)    NOT NULL,
    inflation_rate          NUMERIC(6,4)    NOT NULL DEFAULT 0.02,
    npv                     NUMERIC(16,2),
    irr                     NUMERIC(8,4),
    payback_years           NUMERIC(8,2),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_fs_discount CHECK (
        discount_rate >= 0 AND discount_rate <= 1
    ),
    CONSTRAINT chk_p033_fs_escalation CHECK (
        escalation_rate >= -0.1 AND escalation_rate <= 0.5
    ),
    CONSTRAINT chk_p033_fs_inflation CHECK (
        inflation_rate >= -0.1 AND inflation_rate <= 0.5
    ),
    CONSTRAINT chk_p033_fs_payback CHECK (
        payback_years IS NULL OR payback_years >= 0
    ),
    CONSTRAINT chk_p033_fs_irr CHECK (
        irr IS NULL OR (irr >= -1 AND irr <= 10)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_fs_analysis      ON pack033_quick_wins.financial_scenarios(analysis_id);
CREATE INDEX idx_p033_fs_npv           ON pack033_quick_wins.financial_scenarios(npv DESC);
CREATE INDEX idx_p033_fs_created       ON pack033_quick_wins.financial_scenarios(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_fs_updated
    BEFORE UPDATE ON pack033_quick_wins.financial_scenarios
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.payback_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.financial_scenarios ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_pa_tenant_isolation
    ON pack033_quick_wins.payback_analyses
    USING (scan_id IN (
        SELECT scan_id FROM pack033_quick_wins.quick_wins_scans
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_pa_service_bypass
    ON pack033_quick_wins.payback_analyses
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_fs_tenant_isolation
    ON pack033_quick_wins.financial_scenarios
    USING (analysis_id IN (
        SELECT pa.analysis_id FROM pack033_quick_wins.payback_analyses pa
        JOIN pack033_quick_wins.quick_wins_scans qs ON pa.scan_id = qs.scan_id
        WHERE qs.tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_fs_service_bypass
    ON pack033_quick_wins.financial_scenarios
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.payback_analyses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.financial_scenarios TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.payback_analyses IS
    'Detailed financial analysis per action including simple/discounted payback, NPV, IRR, ROI, LCOE, and tax incentive adjustments.';

COMMENT ON TABLE pack033_quick_wins.financial_scenarios IS
    'What-if financial scenarios for sensitivity analysis with varied discount, escalation, and inflation rates.';

COMMENT ON COLUMN pack033_quick_wins.payback_analyses.analysis_id IS
    'Unique identifier for the financial analysis.';
COMMENT ON COLUMN pack033_quick_wins.payback_analyses.simple_payback_years IS
    'Simple payback period in years (implementation_cost / annual_savings).';
COMMENT ON COLUMN pack033_quick_wins.payback_analyses.discounted_payback_years IS
    'Discounted payback period accounting for time value of money.';
COMMENT ON COLUMN pack033_quick_wins.payback_analyses.npv IS
    'Net Present Value of the action over the analysis period.';
COMMENT ON COLUMN pack033_quick_wins.payback_analyses.irr IS
    'Internal Rate of Return as a decimal (e.g., 0.15 = 15%).';
COMMENT ON COLUMN pack033_quick_wins.payback_analyses.lcoe IS
    'Levelized Cost of Energy savings (cost per kWh saved over lifetime).';
COMMENT ON COLUMN pack033_quick_wins.payback_analyses.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack033_quick_wins.financial_scenarios.scenario_name IS
    'Descriptive name for the scenario (e.g., Optimistic, Base Case, Pessimistic).';

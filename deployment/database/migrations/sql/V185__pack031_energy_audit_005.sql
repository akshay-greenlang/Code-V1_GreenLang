-- =============================================================================
-- V185: PACK-031 Industrial Energy Audit - Energy Savings & ECMs
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    005 of 010
-- Date:         March 2026
--
-- Energy conservation measures (ECMs) with IPMVP measurement and
-- verification plans, savings verification records, and financial
-- analysis (NPV, IRR, payback, LCOE, ROI).
--
-- Tables (4):
--   1. pack031_energy_audit.energy_savings_measures
--   2. pack031_energy_audit.ipmvp_plans
--   3. pack031_energy_audit.savings_verifications
--   4. pack031_energy_audit.financial_analyses
--
-- Previous: V184__pack031_energy_audit_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.energy_savings_measures
-- =============================================================================
-- Energy conservation measures with baseline, expected savings,
-- implementation cost, lifecycle, and status tracking.

CREATE TABLE pack031_energy_audit.energy_savings_measures (
    measure_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    audit_id                UUID            REFERENCES pack031_energy_audit.energy_audits(audit_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(500)    NOT NULL,
    category                VARCHAR(100)    NOT NULL,
    system_affected         VARCHAR(100),
    description             TEXT,
    baseline_kwh            NUMERIC(14,4),
    expected_savings_kwh    NUMERIC(14,4),
    savings_pct             NUMERIC(8,4),
    confidence_level        NUMERIC(5,2),
    implementation_cost_eur NUMERIC(14,4),
    annual_maintenance_eur  NUMERIC(12,4),
    lifetime_years          INTEGER,
    complexity              VARCHAR(20)     DEFAULT 'medium',
    priority                VARCHAR(20)     DEFAULT 'medium',
    status                  VARCHAR(30)     DEFAULT 'proposed',
    implementation_date     DATE,
    completion_date         DATE,
    responsible_party       VARCHAR(255),
    co2_savings_tonnes      NUMERIC(12,4),
    funding_source          VARCHAR(100),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_measure_category CHECK (
        category IN ('HVAC', 'LIGHTING', 'COMPRESSED_AIR', 'STEAM', 'MOTORS_DRIVES',
                     'PUMPS', 'FANS', 'PROCESS_HEAT', 'INSULATION', 'CONTROLS',
                     'HEAT_RECOVERY', 'POWER_FACTOR', 'BUILDING_ENVELOPE',
                     'RENEWABLE_ENERGY', 'COGENERATION', 'DEMAND_RESPONSE', 'OTHER')
    ),
    CONSTRAINT chk_p031_measure_baseline CHECK (
        baseline_kwh IS NULL OR baseline_kwh >= 0
    ),
    CONSTRAINT chk_p031_measure_savings CHECK (
        expected_savings_kwh IS NULL OR expected_savings_kwh >= 0
    ),
    CONSTRAINT chk_p031_measure_savings_pct CHECK (
        savings_pct IS NULL OR (savings_pct >= 0 AND savings_pct <= 100)
    ),
    CONSTRAINT chk_p031_measure_confidence CHECK (
        confidence_level IS NULL OR (confidence_level >= 0 AND confidence_level <= 100)
    ),
    CONSTRAINT chk_p031_measure_cost CHECK (
        implementation_cost_eur IS NULL OR implementation_cost_eur >= 0
    ),
    CONSTRAINT chk_p031_measure_lifetime CHECK (
        lifetime_years IS NULL OR (lifetime_years >= 0 AND lifetime_years <= 50)
    ),
    CONSTRAINT chk_p031_measure_complexity CHECK (
        complexity IN ('simple', 'medium', 'complex', 'major_project')
    ),
    CONSTRAINT chk_p031_measure_priority CHECK (
        priority IN ('critical', 'high', 'medium', 'low')
    ),
    CONSTRAINT chk_p031_measure_status CHECK (
        status IN ('proposed', 'evaluated', 'approved', 'funded', 'in_progress',
                   'implemented', 'verified', 'rejected', 'deferred')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_measure_facility ON pack031_energy_audit.energy_savings_measures(facility_id);
CREATE INDEX idx_p031_measure_audit    ON pack031_energy_audit.energy_savings_measures(audit_id);
CREATE INDEX idx_p031_measure_tenant   ON pack031_energy_audit.energy_savings_measures(tenant_id);
CREATE INDEX idx_p031_measure_category ON pack031_energy_audit.energy_savings_measures(category);
CREATE INDEX idx_p031_measure_system   ON pack031_energy_audit.energy_savings_measures(system_affected);
CREATE INDEX idx_p031_measure_priority ON pack031_energy_audit.energy_savings_measures(priority);
CREATE INDEX idx_p031_measure_status   ON pack031_energy_audit.energy_savings_measures(status);
CREATE INDEX idx_p031_measure_savings  ON pack031_energy_audit.energy_savings_measures(expected_savings_kwh DESC);
CREATE INDEX idx_p031_measure_cost     ON pack031_energy_audit.energy_savings_measures(implementation_cost_eur);
CREATE INDEX idx_p031_measure_created  ON pack031_energy_audit.energy_savings_measures(created_at DESC);

-- Trigger
CREATE TRIGGER trg_p031_measure_updated
    BEFORE UPDATE ON pack031_energy_audit.energy_savings_measures
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack031_energy_audit.ipmvp_plans
-- =============================================================================
-- International Performance Measurement and Verification Protocol (IPMVP)
-- plans defining M&V approach, measurement boundary, and baseline/post periods.

CREATE TABLE pack031_energy_audit.ipmvp_plans (
    plan_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    measure_id              UUID            NOT NULL REFERENCES pack031_energy_audit.energy_savings_measures(measure_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    ipmvp_option            VARCHAR(20)     NOT NULL,
    measurement_boundary    TEXT,
    baseline_period_start   DATE            NOT NULL,
    baseline_period_end     DATE            NOT NULL,
    post_period_start       DATE,
    post_period_end         DATE,
    key_parameters          JSONB           DEFAULT '{}',
    static_factors          JSONB           DEFAULT '{}',
    interactive_effects     TEXT,
    metering_equipment      TEXT,
    sampling_plan           TEXT,
    uncertainty_budget_pct  NUMERIC(5,2),
    mv_cost_eur             NUMERIC(12,4),
    reporting_frequency     VARCHAR(30),
    status                  VARCHAR(30)     DEFAULT 'draft',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_ipmvp_option CHECK (
        ipmvp_option IN ('OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D')
    ),
    CONSTRAINT chk_p031_ipmvp_baseline_dates CHECK (
        baseline_period_start < baseline_period_end
    ),
    CONSTRAINT chk_p031_ipmvp_post_dates CHECK (
        post_period_start IS NULL OR post_period_end IS NULL OR post_period_start < post_period_end
    ),
    CONSTRAINT chk_p031_ipmvp_status CHECK (
        status IN ('draft', 'approved', 'active', 'completed', 'cancelled')
    )
);

-- Indexes
CREATE INDEX idx_p031_ipmvp_measure    ON pack031_energy_audit.ipmvp_plans(measure_id);
CREATE INDEX idx_p031_ipmvp_tenant     ON pack031_energy_audit.ipmvp_plans(tenant_id);
CREATE INDEX idx_p031_ipmvp_option     ON pack031_energy_audit.ipmvp_plans(ipmvp_option);
CREATE INDEX idx_p031_ipmvp_status     ON pack031_energy_audit.ipmvp_plans(status);

-- Trigger
CREATE TRIGGER trg_p031_ipmvp_updated
    BEFORE UPDATE ON pack031_energy_audit.ipmvp_plans
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack031_energy_audit.savings_verifications
-- =============================================================================
-- Post-implementation savings verification records with measured vs
-- expected comparison, confidence intervals, and routine adjustments.

CREATE TABLE pack031_energy_audit.savings_verifications (
    verification_id         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    measure_id              UUID            NOT NULL REFERENCES pack031_energy_audit.energy_savings_measures(measure_id) ON DELETE CASCADE,
    plan_id                 UUID            REFERENCES pack031_energy_audit.ipmvp_plans(plan_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    verification_date       DATE            NOT NULL,
    reporting_period_start  DATE,
    reporting_period_end    DATE,
    verified_savings_kwh    NUMERIC(14,4)   NOT NULL,
    verified_savings_eur    NUMERIC(14,4),
    expected_savings_kwh    NUMERIC(14,4),
    savings_realization_pct NUMERIC(8,4),
    confidence_interval_pct NUMERIC(5,2),
    adjustments             JSONB           DEFAULT '{}',
    non_routine_adjustments JSONB           DEFAULT '{}',
    verifier_name           VARCHAR(255),
    status                  VARCHAR(30)     DEFAULT 'draft',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_verify_savings CHECK (
        verified_savings_kwh >= 0
    ),
    CONSTRAINT chk_p031_verify_confidence CHECK (
        confidence_interval_pct IS NULL OR (confidence_interval_pct >= 0 AND confidence_interval_pct <= 100)
    ),
    CONSTRAINT chk_p031_verify_status CHECK (
        status IN ('draft', 'reviewed', 'approved', 'disputed')
    )
);

-- Indexes
CREATE INDEX idx_p031_verify_measure   ON pack031_energy_audit.savings_verifications(measure_id);
CREATE INDEX idx_p031_verify_plan      ON pack031_energy_audit.savings_verifications(plan_id);
CREATE INDEX idx_p031_verify_tenant    ON pack031_energy_audit.savings_verifications(tenant_id);
CREATE INDEX idx_p031_verify_date      ON pack031_energy_audit.savings_verifications(verification_date);
CREATE INDEX idx_p031_verify_status    ON pack031_energy_audit.savings_verifications(status);

-- =============================================================================
-- Table 4: pack031_energy_audit.financial_analyses
-- =============================================================================
-- Financial analysis of energy conservation measures: NPV, IRR,
-- payback, discounted payback, LCOE, and ROI calculations.

CREATE TABLE pack031_energy_audit.financial_analyses (
    analysis_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    measure_id              UUID            NOT NULL REFERENCES pack031_energy_audit.energy_savings_measures(measure_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    npv_eur                 NUMERIC(14,4),
    irr_pct                 NUMERIC(8,4),
    simple_payback_years    NUMERIC(8,2),
    discounted_payback_years NUMERIC(8,2),
    lcoe_eur_kwh            NUMERIC(10,6),
    roi_pct                 NUMERIC(8,4),
    discount_rate           NUMERIC(6,4)    NOT NULL DEFAULT 0.08,
    energy_price_escalation NUMERIC(6,4)    DEFAULT 0.03,
    analysis_period_years   INTEGER         DEFAULT 20,
    annual_cash_flows       JSONB           DEFAULT '[]',
    sensitivity_results     JSONB           DEFAULT '{}',
    currency                VARCHAR(3)      DEFAULT 'EUR',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_finance_discount CHECK (
        discount_rate >= 0 AND discount_rate <= 1
    ),
    CONSTRAINT chk_p031_finance_escalation CHECK (
        energy_price_escalation IS NULL OR (energy_price_escalation >= -0.5 AND energy_price_escalation <= 0.5)
    ),
    CONSTRAINT chk_p031_finance_period CHECK (
        analysis_period_years IS NULL OR (analysis_period_years >= 1 AND analysis_period_years <= 50)
    )
);

-- Indexes
CREATE INDEX idx_p031_finance_measure  ON pack031_energy_audit.financial_analyses(measure_id);
CREATE INDEX idx_p031_finance_tenant   ON pack031_energy_audit.financial_analyses(tenant_id);
CREATE INDEX idx_p031_finance_npv      ON pack031_energy_audit.financial_analyses(npv_eur DESC);
CREATE INDEX idx_p031_finance_irr      ON pack031_energy_audit.financial_analyses(irr_pct DESC);
CREATE INDEX idx_p031_finance_payback  ON pack031_energy_audit.financial_analyses(simple_payback_years);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.energy_savings_measures ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.ipmvp_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.savings_verifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.financial_analyses ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_measure_tenant_isolation ON pack031_energy_audit.energy_savings_measures
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_measure_service_bypass ON pack031_energy_audit.energy_savings_measures
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_ipmvp_tenant_isolation ON pack031_energy_audit.ipmvp_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_ipmvp_service_bypass ON pack031_energy_audit.ipmvp_plans
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_verify_tenant_isolation ON pack031_energy_audit.savings_verifications
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_verify_service_bypass ON pack031_energy_audit.savings_verifications
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_finance_tenant_isolation ON pack031_energy_audit.financial_analyses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_finance_service_bypass ON pack031_energy_audit.financial_analyses
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_savings_measures TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.ipmvp_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.savings_verifications TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.financial_analyses TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.energy_savings_measures IS
    'Energy conservation measures with baseline, expected savings, implementation cost, lifecycle, and status tracking.';
COMMENT ON TABLE pack031_energy_audit.ipmvp_plans IS
    'IPMVP measurement and verification plans defining M&V approach, boundary, and baseline/post periods.';
COMMENT ON TABLE pack031_energy_audit.savings_verifications IS
    'Post-implementation savings verification records with measured vs expected comparison and confidence intervals.';
COMMENT ON TABLE pack031_energy_audit.financial_analyses IS
    'Financial analysis of ECMs: NPV, IRR, simple/discounted payback, LCOE, and ROI with sensitivity results.';

COMMENT ON COLUMN pack031_energy_audit.ipmvp_plans.ipmvp_option IS
    'IPMVP Option: A (retrofit isolation key parameter), B (retrofit isolation all parameter), C (whole facility), D (calibrated simulation).';
COMMENT ON COLUMN pack031_energy_audit.financial_analyses.lcoe_eur_kwh IS
    'Levelized Cost of Energy saved in EUR per kWh.';

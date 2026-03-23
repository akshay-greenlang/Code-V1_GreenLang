-- =============================================================================
-- V304: PACK-038 Peak Shaving Pack - Financial Analysis
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Financial modeling tables for peak shaving investments. Covers
-- comprehensive financial models, incentive capture, revenue stacking
-- from multiple value streams, cash flow projections, and sensitivity
-- analysis for BESS and peak shaving capital investments.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_financial_models
--   2. pack038_peak_shaving.ps_incentive_capture
--   3. pack038_peak_shaving.ps_revenue_stacking
--   4. pack038_peak_shaving.ps_cash_flow_projections
--   5. pack038_peak_shaving.ps_sensitivity_results
--
-- Seed Data: ITC rates, SGIP step rates, state incentive programs
--
-- Previous: V303__pack038_peak_shaving_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_financial_models
-- =============================================================================
-- Comprehensive financial model for peak shaving investments (BESS,
-- PF correction, load shifting equipment). Computes NPV, IRR, payback,
-- and lifecycle economics considering degradation, escalation, and
-- incentives.

CREATE TABLE pack038_peak_shaving.ps_financial_models (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    bess_id                 UUID            REFERENCES pack038_peak_shaving.ps_bess_configurations(id),
    tenant_id               UUID            NOT NULL,
    model_name              VARCHAR(255)    NOT NULL,
    model_type              VARCHAR(30)     NOT NULL,
    analysis_period_years   INTEGER         NOT NULL DEFAULT 15,
    discount_rate_pct       NUMERIC(5,3)    NOT NULL DEFAULT 7.000,
    inflation_rate_pct      NUMERIC(5,3)    NOT NULL DEFAULT 2.500,
    demand_charge_escalation_pct NUMERIC(5,3) DEFAULT 3.000,
    energy_cost_escalation_pct NUMERIC(5,3) DEFAULT 2.500,
    -- Capital costs
    capital_cost             NUMERIC(12,2)   NOT NULL,
    installation_cost       NUMERIC(12,2),
    engineering_cost        NUMERIC(10,2),
    permitting_cost         NUMERIC(10,2),
    interconnection_cost    NUMERIC(10,2),
    total_installed_cost    NUMERIC(12,2)   NOT NULL,
    -- Incentives
    federal_itc_pct         NUMERIC(5,2),
    federal_itc_amount      NUMERIC(12,2),
    state_incentive_amount  NUMERIC(12,2),
    utility_rebate_amount   NUMERIC(12,2),
    total_incentives        NUMERIC(12,2),
    net_cost_after_incentives NUMERIC(12,2),
    -- Annual costs
    annual_om_cost          NUMERIC(10,2),
    annual_insurance_cost   NUMERIC(10,2),
    annual_property_tax     NUMERIC(10,2),
    augmentation_reserve    NUMERIC(10,2),
    total_annual_cost       NUMERIC(10,2),
    -- Annual revenue/savings
    demand_charge_savings   NUMERIC(12,2)   NOT NULL,
    cp_charge_savings       NUMERIC(12,2),
    pf_penalty_savings      NUMERIC(12,2),
    energy_arbitrage_value  NUMERIC(12,2),
    ancillary_services_value NUMERIC(12,2),
    resilience_value        NUMERIC(12,2),
    total_annual_value      NUMERIC(12,2)   NOT NULL,
    -- Financial metrics
    simple_payback_years    NUMERIC(6,2),
    discounted_payback_years NUMERIC(6,2),
    npv                     NUMERIC(14,2),
    irr_pct                 NUMERIC(7,4),
    mirr_pct                NUMERIC(7,4),
    roi_pct                 NUMERIC(7,4),
    lcoe_per_kwh            NUMERIC(10,4),
    benefit_cost_ratio      NUMERIC(6,3),
    profitability_index     NUMERIC(6,3),
    -- Tax
    depreciation_method     VARCHAR(20),
    macrs_years             INTEGER,
    tax_rate_pct            NUMERIC(5,2),
    tax_savings_year1       NUMERIC(12,2),
    -- Status
    model_status            VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    assumptions             JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_fm_type CHECK (
        model_type IN (
            'BESS_PEAK_SHAVING', 'PF_CORRECTION', 'LOAD_SHIFTING',
            'COMBINED_BESS_PF', 'SOLAR_PLUS_STORAGE', 'DEMAND_MANAGEMENT',
            'COMPREHENSIVE'
        )
    ),
    CONSTRAINT chk_p038_fm_period CHECK (
        analysis_period_years >= 1 AND analysis_period_years <= 30
    ),
    CONSTRAINT chk_p038_fm_discount CHECK (
        discount_rate_pct >= 0 AND discount_rate_pct <= 30
    ),
    CONSTRAINT chk_p038_fm_capital CHECK (
        capital_cost >= 0
    ),
    CONSTRAINT chk_p038_fm_installed CHECK (
        total_installed_cost >= 0
    ),
    CONSTRAINT chk_p038_fm_savings CHECK (
        demand_charge_savings >= 0
    ),
    CONSTRAINT chk_p038_fm_annual_value CHECK (
        total_annual_value >= 0
    ),
    CONSTRAINT chk_p038_fm_payback CHECK (
        simple_payback_years IS NULL OR simple_payback_years >= 0
    ),
    CONSTRAINT chk_p038_fm_itc CHECK (
        federal_itc_pct IS NULL OR (federal_itc_pct >= 0 AND federal_itc_pct <= 100)
    ),
    CONSTRAINT chk_p038_fm_depreciation CHECK (
        depreciation_method IS NULL OR depreciation_method IN (
            'MACRS', 'STRAIGHT_LINE', 'DECLINING_BALANCE', 'BONUS', 'NONE'
        )
    ),
    CONSTRAINT chk_p038_fm_macrs CHECK (
        macrs_years IS NULL OR macrs_years IN (5, 7, 10, 15, 20)
    ),
    CONSTRAINT chk_p038_fm_status CHECK (
        model_status IN ('DRAFT', 'PRELIMINARY', 'FINAL', 'APPROVED', 'ARCHIVED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_fm_profile         ON pack038_peak_shaving.ps_financial_models(profile_id);
CREATE INDEX idx_p038_fm_bess            ON pack038_peak_shaving.ps_financial_models(bess_id);
CREATE INDEX idx_p038_fm_tenant          ON pack038_peak_shaving.ps_financial_models(tenant_id);
CREATE INDEX idx_p038_fm_type            ON pack038_peak_shaving.ps_financial_models(model_type);
CREATE INDEX idx_p038_fm_npv             ON pack038_peak_shaving.ps_financial_models(npv DESC);
CREATE INDEX idx_p038_fm_irr             ON pack038_peak_shaving.ps_financial_models(irr_pct DESC);
CREATE INDEX idx_p038_fm_payback         ON pack038_peak_shaving.ps_financial_models(simple_payback_years);
CREATE INDEX idx_p038_fm_status          ON pack038_peak_shaving.ps_financial_models(model_status);
CREATE INDEX idx_p038_fm_created         ON pack038_peak_shaving.ps_financial_models(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_fm_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_financial_models
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_incentive_capture
-- =============================================================================
-- Tracking of available and captured incentives for peak shaving
-- investments including federal ITC, state programs, utility rebates,
-- and performance-based incentives.

CREATE TABLE pack038_peak_shaving.ps_incentive_capture (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    financial_model_id      UUID            REFERENCES pack038_peak_shaving.ps_financial_models(id) ON DELETE CASCADE,
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id),
    tenant_id               UUID            NOT NULL,
    incentive_name          VARCHAR(255)    NOT NULL,
    incentive_type          VARCHAR(30)     NOT NULL,
    program_name            VARCHAR(255),
    program_administrator   VARCHAR(255),
    jurisdiction            VARCHAR(50)     NOT NULL,
    country_code            CHAR(2)         NOT NULL DEFAULT 'US',
    incentive_rate_per_kw   NUMERIC(10,4),
    incentive_rate_per_kwh  NUMERIC(10,4),
    incentive_pct           NUMERIC(5,2),
    incentive_amount        NUMERIC(12,2)   NOT NULL,
    max_incentive_amount    NUMERIC(12,2),
    step_number             INTEGER,
    step_budget_remaining   NUMERIC(12,2),
    application_deadline    DATE,
    reservation_date        DATE,
    approval_date           DATE,
    payment_date            DATE,
    disbursement_type       VARCHAR(20)     NOT NULL DEFAULT 'LUMP_SUM',
    performance_requirement BOOLEAN         DEFAULT false,
    performance_period_years INTEGER,
    eligible_technologies   JSONB           DEFAULT '[]',
    application_status      VARCHAR(20)     NOT NULL DEFAULT 'AVAILABLE',
    application_notes       TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ic_type CHECK (
        incentive_type IN (
            'FEDERAL_ITC', 'FEDERAL_PTC', 'STATE_REBATE', 'STATE_TAX_CREDIT',
            'UTILITY_REBATE', 'UTILITY_PERFORMANCE', 'SGIP', 'MACRS_BONUS',
            'GRANT', 'LOAN_SUBSIDY', 'PROPERTY_TAX_EXEMPT', 'SALES_TAX_EXEMPT',
            'OTHER'
        )
    ),
    CONSTRAINT chk_p038_ic_amount CHECK (
        incentive_amount >= 0
    ),
    CONSTRAINT chk_p038_ic_max CHECK (
        max_incentive_amount IS NULL OR max_incentive_amount >= 0
    ),
    CONSTRAINT chk_p038_ic_pct CHECK (
        incentive_pct IS NULL OR (incentive_pct >= 0 AND incentive_pct <= 100)
    ),
    CONSTRAINT chk_p038_ic_disbursement CHECK (
        disbursement_type IN ('LUMP_SUM', 'ANNUAL', 'QUARTERLY', 'PERFORMANCE_BASED', 'MILESTONE')
    ),
    CONSTRAINT chk_p038_ic_status CHECK (
        application_status IN (
            'AVAILABLE', 'RESEARCHING', 'APPLYING', 'SUBMITTED', 'RESERVED',
            'APPROVED', 'RECEIVED', 'DENIED', 'EXPIRED', 'NOT_ELIGIBLE'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ic_financial       ON pack038_peak_shaving.ps_incentive_capture(financial_model_id);
CREATE INDEX idx_p038_ic_profile         ON pack038_peak_shaving.ps_incentive_capture(profile_id);
CREATE INDEX idx_p038_ic_tenant          ON pack038_peak_shaving.ps_incentive_capture(tenant_id);
CREATE INDEX idx_p038_ic_type            ON pack038_peak_shaving.ps_incentive_capture(incentive_type);
CREATE INDEX idx_p038_ic_jurisdiction    ON pack038_peak_shaving.ps_incentive_capture(jurisdiction);
CREATE INDEX idx_p038_ic_amount          ON pack038_peak_shaving.ps_incentive_capture(incentive_amount DESC);
CREATE INDEX idx_p038_ic_status          ON pack038_peak_shaving.ps_incentive_capture(application_status);
CREATE INDEX idx_p038_ic_deadline        ON pack038_peak_shaving.ps_incentive_capture(application_deadline);
CREATE INDEX idx_p038_ic_created         ON pack038_peak_shaving.ps_incentive_capture(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_ic_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_incentive_capture
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_revenue_stacking
-- =============================================================================
-- Revenue stacking analysis showing how multiple value streams from a
-- single BESS asset combine to improve project economics. Includes
-- demand charge reduction, CP avoidance, energy arbitrage, ancillary
-- services, and resilience value.

CREATE TABLE pack038_peak_shaving.ps_revenue_stacking (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    financial_model_id      UUID            REFERENCES pack038_peak_shaving.ps_financial_models(id) ON DELETE CASCADE,
    bess_id                 UUID            REFERENCES pack038_peak_shaving.ps_bess_configurations(id),
    tenant_id               UUID            NOT NULL,
    analysis_year           INTEGER         NOT NULL,
    value_stream            VARCHAR(50)     NOT NULL,
    annual_value            NUMERIC(12,2)   NOT NULL,
    hours_allocated         NUMERIC(8,2),
    cycles_allocated        NUMERIC(8,2),
    energy_allocated_kwh    NUMERIC(15,3),
    capacity_allocated_kw   NUMERIC(12,3),
    utilization_pct         NUMERIC(5,2),
    priority_rank           INTEGER,
    conflict_with           VARCHAR(50),
    overlap_resolution      VARCHAR(30),
    revenue_certainty       VARCHAR(20)     NOT NULL DEFAULT 'HIGH',
    contract_based          BOOLEAN         DEFAULT false,
    contract_term_years     INTEGER,
    market_rate_source      VARCHAR(100),
    escalation_rate_pct     NUMERIC(5,3),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_rs_year CHECK (
        analysis_year >= 2020 AND analysis_year <= 2060
    ),
    CONSTRAINT chk_p038_rs_stream CHECK (
        value_stream IN (
            'DEMAND_CHARGE_REDUCTION', 'CP_AVOIDANCE', 'RATCHET_AVOIDANCE',
            'PF_CORRECTION', 'ENERGY_ARBITRAGE', 'FREQUENCY_REGULATION',
            'SPINNING_RESERVE', 'NON_SPINNING_RESERVE', 'CAPACITY_MARKET',
            'RESILIENCE', 'BACKUP_POWER', 'SOLAR_SELF_CONSUMPTION',
            'TOU_OPTIMIZATION', 'DEMAND_RESPONSE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_rs_value CHECK (
        annual_value >= 0
    ),
    CONSTRAINT chk_p038_rs_utilization CHECK (
        utilization_pct IS NULL OR (utilization_pct >= 0 AND utilization_pct <= 100)
    ),
    CONSTRAINT chk_p038_rs_certainty CHECK (
        revenue_certainty IN ('HIGH', 'MEDIUM', 'LOW', 'SPECULATIVE')
    ),
    CONSTRAINT chk_p038_rs_overlap CHECK (
        overlap_resolution IS NULL OR overlap_resolution IN (
            'TIME_SEPARATED', 'PRIORITY_BASED', 'PROPORTIONAL', 'EXCLUSIVE'
        )
    ),
    CONSTRAINT chk_p038_rs_rank CHECK (
        priority_rank IS NULL OR priority_rank >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_rs_financial       ON pack038_peak_shaving.ps_revenue_stacking(financial_model_id);
CREATE INDEX idx_p038_rs_bess            ON pack038_peak_shaving.ps_revenue_stacking(bess_id);
CREATE INDEX idx_p038_rs_tenant          ON pack038_peak_shaving.ps_revenue_stacking(tenant_id);
CREATE INDEX idx_p038_rs_year            ON pack038_peak_shaving.ps_revenue_stacking(analysis_year);
CREATE INDEX idx_p038_rs_stream          ON pack038_peak_shaving.ps_revenue_stacking(value_stream);
CREATE INDEX idx_p038_rs_value           ON pack038_peak_shaving.ps_revenue_stacking(annual_value DESC);
CREATE INDEX idx_p038_rs_certainty       ON pack038_peak_shaving.ps_revenue_stacking(revenue_certainty);
CREATE INDEX idx_p038_rs_created         ON pack038_peak_shaving.ps_revenue_stacking(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_rs_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_revenue_stacking
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_cash_flow_projections
-- =============================================================================
-- Year-by-year cash flow projections for peak shaving investments
-- including degradation-adjusted savings, escalating costs, incentive
-- timing, and cumulative NPV.

CREATE TABLE pack038_peak_shaving.ps_cash_flow_projections (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    financial_model_id      UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_financial_models(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    projection_year         INTEGER         NOT NULL,
    calendar_year           INTEGER         NOT NULL,
    -- Costs
    capital_expenditure     NUMERIC(12,2)   DEFAULT 0,
    operating_cost          NUMERIC(10,2)   DEFAULT 0,
    augmentation_cost       NUMERIC(10,2)   DEFAULT 0,
    insurance_cost          NUMERIC(10,2)   DEFAULT 0,
    decommissioning_cost    NUMERIC(10,2)   DEFAULT 0,
    total_cost              NUMERIC(12,2)   NOT NULL,
    -- Revenue/Savings
    demand_charge_savings   NUMERIC(12,2)   DEFAULT 0,
    cp_charge_savings       NUMERIC(12,2)   DEFAULT 0,
    pf_savings              NUMERIC(12,2)   DEFAULT 0,
    energy_arbitrage        NUMERIC(12,2)   DEFAULT 0,
    ancillary_services      NUMERIC(12,2)   DEFAULT 0,
    other_revenue           NUMERIC(12,2)   DEFAULT 0,
    total_revenue           NUMERIC(12,2)   NOT NULL,
    -- Incentives & Tax
    incentive_received      NUMERIC(12,2)   DEFAULT 0,
    depreciation_benefit    NUMERIC(12,2)   DEFAULT 0,
    tax_benefit             NUMERIC(12,2)   DEFAULT 0,
    -- Net
    net_cash_flow           NUMERIC(14,2)   NOT NULL,
    cumulative_cash_flow    NUMERIC(14,2)   NOT NULL,
    discount_factor         NUMERIC(8,6),
    present_value           NUMERIC(14,2),
    cumulative_npv          NUMERIC(14,2),
    -- BESS state
    bess_soh_pct            NUMERIC(5,2),
    bess_capacity_kwh       NUMERIC(12,3),
    cycles_this_year        NUMERIC(8,2),
    degradation_adjustment  NUMERIC(5,4),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cf_proj_year CHECK (
        projection_year >= 0 AND projection_year <= 30
    ),
    CONSTRAINT chk_p038_cf_cal_year CHECK (
        calendar_year >= 2020 AND calendar_year <= 2060
    ),
    CONSTRAINT chk_p038_cf_soh CHECK (
        bess_soh_pct IS NULL OR (bess_soh_pct >= 0 AND bess_soh_pct <= 100)
    ),
    CONSTRAINT chk_p038_cf_degradation CHECK (
        degradation_adjustment IS NULL OR (degradation_adjustment >= 0 AND degradation_adjustment <= 1.0)
    ),
    CONSTRAINT uq_p038_cf_model_year UNIQUE (financial_model_id, projection_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cf_financial       ON pack038_peak_shaving.ps_cash_flow_projections(financial_model_id);
CREATE INDEX idx_p038_cf_tenant          ON pack038_peak_shaving.ps_cash_flow_projections(tenant_id);
CREATE INDEX idx_p038_cf_year            ON pack038_peak_shaving.ps_cash_flow_projections(projection_year);
CREATE INDEX idx_p038_cf_npv             ON pack038_peak_shaving.ps_cash_flow_projections(cumulative_npv DESC);
CREATE INDEX idx_p038_cf_created         ON pack038_peak_shaving.ps_cash_flow_projections(created_at DESC);

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_sensitivity_results
-- =============================================================================
-- Sensitivity and scenario analysis results showing how key input
-- assumptions affect financial outcomes. Tests variations in demand
-- charges, BESS costs, degradation rates, incentives, and other
-- parameters.

CREATE TABLE pack038_peak_shaving.ps_sensitivity_results (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    financial_model_id      UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_financial_models(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    scenario_name           VARCHAR(255)    NOT NULL,
    scenario_type           VARCHAR(30)     NOT NULL,
    variable_name           VARCHAR(100)    NOT NULL,
    variable_unit           VARCHAR(30),
    base_value              NUMERIC(14,4)   NOT NULL,
    test_value              NUMERIC(14,4)   NOT NULL,
    change_pct              NUMERIC(7,4),
    -- Resulting metrics
    result_npv              NUMERIC(14,2)   NOT NULL,
    result_irr_pct          NUMERIC(7,4),
    result_payback_years    NUMERIC(6,2),
    result_annual_savings   NUMERIC(12,2),
    result_lcoe             NUMERIC(10,4),
    -- Delta from base case
    npv_delta               NUMERIC(14,2),
    npv_delta_pct           NUMERIC(7,4),
    irr_delta_pct           NUMERIC(7,4),
    payback_delta_years     NUMERIC(6,2),
    -- Tornado chart data
    sensitivity_rank        INTEGER,
    elasticity              NUMERIC(8,4),
    -- Breakeven
    breakeven_value         NUMERIC(14,4),
    breakeven_achievable    BOOLEAN,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_sr_type CHECK (
        scenario_type IN (
            'SINGLE_VARIABLE', 'MULTI_VARIABLE', 'MONTE_CARLO',
            'BEST_CASE', 'WORST_CASE', 'BREAKEVEN'
        )
    ),
    CONSTRAINT chk_p038_sr_rank CHECK (
        sensitivity_rank IS NULL OR sensitivity_rank >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_sr_financial       ON pack038_peak_shaving.ps_sensitivity_results(financial_model_id);
CREATE INDEX idx_p038_sr_tenant          ON pack038_peak_shaving.ps_sensitivity_results(tenant_id);
CREATE INDEX idx_p038_sr_type            ON pack038_peak_shaving.ps_sensitivity_results(scenario_type);
CREATE INDEX idx_p038_sr_variable        ON pack038_peak_shaving.ps_sensitivity_results(variable_name);
CREATE INDEX idx_p038_sr_npv             ON pack038_peak_shaving.ps_sensitivity_results(result_npv DESC);
CREATE INDEX idx_p038_sr_rank            ON pack038_peak_shaving.ps_sensitivity_results(sensitivity_rank);
CREATE INDEX idx_p038_sr_created         ON pack038_peak_shaving.ps_sensitivity_results(created_at DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_financial_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_incentive_capture ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_revenue_stacking ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_cash_flow_projections ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_sensitivity_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_fm_tenant_isolation
    ON pack038_peak_shaving.ps_financial_models
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_fm_service_bypass
    ON pack038_peak_shaving.ps_financial_models
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_ic_tenant_isolation
    ON pack038_peak_shaving.ps_incentive_capture
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ic_service_bypass
    ON pack038_peak_shaving.ps_incentive_capture
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_rs_tenant_isolation
    ON pack038_peak_shaving.ps_revenue_stacking
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_rs_service_bypass
    ON pack038_peak_shaving.ps_revenue_stacking
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_cf_tenant_isolation
    ON pack038_peak_shaving.ps_cash_flow_projections
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cf_service_bypass
    ON pack038_peak_shaving.ps_cash_flow_projections
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_sr_tenant_isolation
    ON pack038_peak_shaving.ps_sensitivity_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_sr_service_bypass
    ON pack038_peak_shaving.ps_sensitivity_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_financial_models TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_incentive_capture TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_revenue_stacking TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_cash_flow_projections TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_sensitivity_results TO PUBLIC;

-- =============================================================================
-- Seed Data: Federal ITC Rates (2024-2035, IRA Section 48/48E)
-- =============================================================================
INSERT INTO pack038_peak_shaving.ps_incentive_capture (tenant_id, profile_id, incentive_name, incentive_type, program_name, program_administrator, jurisdiction, country_code, incentive_pct, incentive_amount, disbursement_type, application_status, application_notes) VALUES
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'IRA Section 48 ITC - Base Rate (2024-2032)', 'FEDERAL_ITC', 'Investment Tax Credit', 'IRS', 'US-FEDERAL', 'US', 6.00, 0, 'LUMP_SUM', 'AVAILABLE', 'Base ITC rate for commercial storage systems <1MW. Prevailing wage/apprenticeship required for >1MW to qualify for full rate.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'IRA Section 48 ITC - Full Rate (2024-2032)', 'FEDERAL_ITC', 'Investment Tax Credit', 'IRS', 'US-FEDERAL', 'US', 30.00, 0, 'LUMP_SUM', 'AVAILABLE', 'Full ITC rate for storage meeting prevailing wage and apprenticeship requirements. Stand-alone storage eligible from 2023+.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'IRA Section 48 ITC - Full + Energy Community (2024-2032)', 'FEDERAL_ITC', 'Investment Tax Credit', 'IRS', 'US-FEDERAL', 'US', 40.00, 0, 'LUMP_SUM', 'AVAILABLE', 'Full ITC + 10% energy community adder for projects in brownfield or fossil fuel community census tracts.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'IRA Section 48 ITC - Full + Low Income (2024-2032)', 'FEDERAL_ITC', 'Investment Tax Credit', 'IRS', 'US-FEDERAL', 'US', 50.00, 0, 'LUMP_SUM', 'AVAILABLE', 'Full ITC + 10-20% low-income community adder for qualified low-income projects (competitive allocation).');

-- =============================================================================
-- Seed Data: California SGIP Step Rates
-- =============================================================================
INSERT INTO pack038_peak_shaving.ps_incentive_capture (tenant_id, profile_id, incentive_name, incentive_type, program_name, program_administrator, jurisdiction, country_code, incentive_rate_per_kwh, incentive_amount, step_number, disbursement_type, performance_requirement, performance_period_years, application_status, application_notes) VALUES
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'SGIP Large Storage Step 1', 'SGIP', 'Self-Generation Incentive Program', 'CA IOUs (PG&E, SCE, SDG&E)', 'US-CA', 'US', 0.50, 0, 1, 'PERFORMANCE_BASED', true, 5, 'AVAILABLE', 'SGIP large storage (>10kW) Step 1 rate. 50% upfront, 50% performance-based over 5 years.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'SGIP Large Storage Step 2', 'SGIP', 'Self-Generation Incentive Program', 'CA IOUs (PG&E, SCE, SDG&E)', 'US-CA', 'US', 0.40, 0, 2, 'PERFORMANCE_BASED', true, 5, 'AVAILABLE', 'SGIP large storage Step 2 rate. Activated after Step 1 budget exhaustion.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'SGIP Large Storage Step 3', 'SGIP', 'Self-Generation Incentive Program', 'CA IOUs (PG&E, SCE, SDG&E)', 'US-CA', 'US', 0.30, 0, 3, 'PERFORMANCE_BASED', true, 5, 'AVAILABLE', 'SGIP large storage Step 3 rate. Activated after Step 2 budget exhaustion.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'SGIP Equity Storage', 'SGIP', 'Self-Generation Incentive Program', 'CA IOUs (PG&E, SCE, SDG&E)', 'US-CA', 'US', 1.00, 0, 1, 'PERFORMANCE_BASED', true, 5, 'AVAILABLE', 'SGIP equity budget for storage in disadvantaged communities or low-income customers. Double the base incentive.');

-- =============================================================================
-- Seed Data: State Incentive Programs
-- =============================================================================
INSERT INTO pack038_peak_shaving.ps_incentive_capture (tenant_id, profile_id, incentive_name, incentive_type, program_name, program_administrator, jurisdiction, country_code, incentive_rate_per_kwh, incentive_rate_per_kw, incentive_amount, max_incentive_amount, disbursement_type, application_status, application_notes) VALUES
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'MA SMART Storage Adder', 'STATE_REBATE', 'SMART Program', 'MA DOER', 'US-MA', 'US', NULL, 200.00, 0, 500000, 'ANNUAL', 'AVAILABLE', 'Massachusetts SMART program storage adder for behind-the-meter storage paired with solar.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'NY VDER Storage Value', 'STATE_REBATE', 'Value of DER', 'NYSERDA', 'US-NY', 'US', NULL, 350.00, 0, 1000000, 'ANNUAL', 'AVAILABLE', 'New York Value of Distributed Energy Resources (VDER) compensation for storage dispatch during system peak.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'NJ Clean Energy Storage Incentive', 'STATE_REBATE', 'NJ Storage Incentive Program', 'NJ BPU', 'US-NJ', 'US', 0.25, NULL, 0, 750000, 'PERFORMANCE_BASED', 'AVAILABLE', 'New Jersey behind-the-meter storage incentive program with performance-based payments.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'CT BESS Incentive', 'STATE_REBATE', 'CT Storage Solutions', 'CT Green Bank', 'US-CT', 'US', 0.20, NULL, 0, 500000, 'LUMP_SUM', 'AVAILABLE', 'Connecticut energy storage incentive for commercial and industrial customers.'),
('00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000', 'OR Solar + Storage Rebate', 'STATE_REBATE', 'Oregon Solar + Storage Rebate', 'Energy Trust of Oregon', 'US-OR', 'US', 0.16, NULL, 0, 250000, 'LUMP_SUM', 'AVAILABLE', 'Oregon energy storage rebate for commercial solar-plus-storage installations.');

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.ps_financial_models IS
    'Comprehensive financial models for peak shaving investments with NPV, IRR, payback, and lifecycle economics.';
COMMENT ON TABLE pack038_peak_shaving.ps_incentive_capture IS
    'Available and captured incentives including federal ITC, state programs, utility rebates, and SGIP for peak shaving investments.';
COMMENT ON TABLE pack038_peak_shaving.ps_revenue_stacking IS
    'Revenue stacking analysis combining demand charge reduction, CP avoidance, energy arbitrage, and ancillary services value streams.';
COMMENT ON TABLE pack038_peak_shaving.ps_cash_flow_projections IS
    'Year-by-year cash flow projections with degradation-adjusted savings, escalating costs, and cumulative NPV.';
COMMENT ON TABLE pack038_peak_shaving.ps_sensitivity_results IS
    'Sensitivity and scenario analysis showing how input assumptions affect financial outcomes with tornado chart data.';

COMMENT ON COLUMN pack038_peak_shaving.ps_financial_models.npv IS 'Net Present Value of the investment over the analysis period at the specified discount rate.';
COMMENT ON COLUMN pack038_peak_shaving.ps_financial_models.irr_pct IS 'Internal Rate of Return - the discount rate that makes NPV equal to zero.';
COMMENT ON COLUMN pack038_peak_shaving.ps_financial_models.benefit_cost_ratio IS 'Ratio of total discounted benefits to total discounted costs. Values > 1.0 indicate positive investment.';
COMMENT ON COLUMN pack038_peak_shaving.ps_revenue_stacking.value_stream IS 'Revenue/savings source: DEMAND_CHARGE_REDUCTION, CP_AVOIDANCE, ENERGY_ARBITRAGE, FREQUENCY_REGULATION, etc.';
COMMENT ON COLUMN pack038_peak_shaving.ps_revenue_stacking.revenue_certainty IS 'Certainty of the revenue stream: HIGH (contract/regulated), MEDIUM (market-based), LOW (variable), SPECULATIVE.';
COMMENT ON COLUMN pack038_peak_shaving.ps_sensitivity_results.elasticity IS 'Sensitivity elasticity: percentage change in NPV per percentage change in the variable. Higher absolute values indicate higher sensitivity.';

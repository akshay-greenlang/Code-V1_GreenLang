-- =============================================================================
-- V351: PACK-043 Scope 3 Complete Pack - Climate Risk Quantification
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates climate risk quantification tables for Scope 3 value chain risk
-- assessment aligned with TCFD recommendations. Supports transition risks
-- (policy, technology, market, reputation), physical risks (acute and
-- chronic hazards to supplier locations), climate-related opportunities,
-- and financial impact modelling across multiple scenarios (IEA NZE, NGFS).
--
-- Tables (5):
--   1. ghg_accounting_scope3_complete.risk_assessments
--   2. ghg_accounting_scope3_complete.transition_risks
--   3. ghg_accounting_scope3_complete.physical_risks
--   4. ghg_accounting_scope3_complete.opportunities
--   5. ghg_accounting_scope3_complete.financial_impacts
--
-- Also includes: indexes, RLS, comments.
-- Previous: V350__pack043_supplier_programme.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.risk_assessments
-- =============================================================================
-- Top-level climate risk assessment header for a Scope 3 inventory. Defines
-- the assessment scope, carbon price assumption, time horizons, and climate
-- scenario used (IEA NZE 2050, NGFS orderly/disorderly, etc.).

CREATE TABLE ghg_accounting_scope3_complete.risk_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Assessment
    assessment_name             VARCHAR(500)    NOT NULL,
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessment_version          INTEGER         NOT NULL DEFAULT 1,
    assessor                    VARCHAR(255),
    -- Carbon price
    carbon_price_usd            DECIMAL(10,2)   NOT NULL,
    carbon_price_year           INTEGER,
    carbon_price_source         VARCHAR(200),
    -- Time horizons
    time_horizon_years          INTEGER         NOT NULL DEFAULT 10,
    short_term_years            INTEGER         DEFAULT 3,
    medium_term_years           INTEGER         DEFAULT 10,
    long_term_years             INTEGER         DEFAULT 30,
    -- Scenario
    scenario                    VARCHAR(100)    NOT NULL DEFAULT 'IEA_NZE_2050',
    scenario_description        TEXT,
    warming_target_c            DECIMAL(3,1),
    -- Scope
    categories_assessed         ghg_accounting_scope3_complete.scope3_category_type[],
    suppliers_assessed          INTEGER,
    total_exposure_tco2e        DECIMAL(15,3),
    -- Results summary
    total_transition_risk_usd   NUMERIC(18,2),
    total_physical_risk_usd     NUMERIC(18,2),
    total_opportunity_usd       NUMERIC(18,2),
    net_risk_usd                NUMERIC(18,2),
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    -- Metadata
    methodology                 TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_ra_carbon_price CHECK (carbon_price_usd >= 0),
    CONSTRAINT chk_p043_ra_carbon_year CHECK (
        carbon_price_year IS NULL OR (carbon_price_year >= 2020 AND carbon_price_year <= 2100)
    ),
    CONSTRAINT chk_p043_ra_horizon CHECK (time_horizon_years > 0 AND time_horizon_years <= 100),
    CONSTRAINT chk_p043_ra_short CHECK (short_term_years IS NULL OR (short_term_years > 0 AND short_term_years <= 10)),
    CONSTRAINT chk_p043_ra_medium CHECK (medium_term_years IS NULL OR (medium_term_years > 0 AND medium_term_years <= 30)),
    CONSTRAINT chk_p043_ra_long CHECK (long_term_years IS NULL OR (long_term_years > 0 AND long_term_years <= 100)),
    CONSTRAINT chk_p043_ra_scenario CHECK (
        scenario IN (
            'IEA_NZE_2050', 'IEA_APS', 'IEA_STEPS', 'IEA_SDS',
            'NGFS_ORDERLY', 'NGFS_DISORDERLY', 'NGFS_HOT_HOUSE',
            'IPCC_SSP1_1.9', 'IPCC_SSP1_2.6', 'IPCC_SSP2_4.5',
            'IPCC_SSP3_7.0', 'IPCC_SSP5_8.5', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p043_ra_warming CHECK (
        warming_target_c IS NULL OR (warming_target_c >= 1.0 AND warming_target_c <= 6.0)
    ),
    CONSTRAINT chk_p043_ra_suppliers CHECK (suppliers_assessed IS NULL OR suppliers_assessed >= 0),
    CONSTRAINT chk_p043_ra_exposure CHECK (total_exposure_tco2e IS NULL OR total_exposure_tco2e >= 0),
    CONSTRAINT chk_p043_ra_version CHECK (assessment_version >= 1),
    CONSTRAINT chk_p043_ra_status CHECK (
        status IN ('DRAFT', 'IN_PROGRESS', 'COMPLETED', 'REVIEWED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p043_ra_inventory_version UNIQUE (inventory_id, assessment_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_ra_tenant             ON ghg_accounting_scope3_complete.risk_assessments(tenant_id);
CREATE INDEX idx_p043_ra_inventory          ON ghg_accounting_scope3_complete.risk_assessments(inventory_id);
CREATE INDEX idx_p043_ra_date               ON ghg_accounting_scope3_complete.risk_assessments(assessment_date DESC);
CREATE INDEX idx_p043_ra_scenario           ON ghg_accounting_scope3_complete.risk_assessments(scenario);
CREATE INDEX idx_p043_ra_carbon_price       ON ghg_accounting_scope3_complete.risk_assessments(carbon_price_usd);
CREATE INDEX idx_p043_ra_status             ON ghg_accounting_scope3_complete.risk_assessments(status);
CREATE INDEX idx_p043_ra_created            ON ghg_accounting_scope3_complete.risk_assessments(created_at DESC);
CREATE INDEX idx_p043_ra_categories         ON ghg_accounting_scope3_complete.risk_assessments USING GIN(categories_assessed);

-- Composite: inventory + latest
CREATE INDEX idx_p043_ra_inv_latest         ON ghg_accounting_scope3_complete.risk_assessments(inventory_id, assessment_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_ra_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.risk_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.transition_risks
-- =============================================================================
-- Transition risks to the organisation's Scope 3 value chain. Each risk is
-- classified by type (policy, technology, market, reputation), linked to an
-- exposure in tCO2e, and quantified as a financial impact in USD with
-- probability and time horizon per TCFD recommendations.

CREATE TABLE ghg_accounting_scope3_complete.transition_risks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.risk_assessments(id) ON DELETE CASCADE,
    -- Risk identification
    risk_name                   VARCHAR(500)    NOT NULL,
    risk_type                   VARCHAR(30)     NOT NULL,
    description                 TEXT,
    -- Exposure
    category                    ghg_accounting_scope3_complete.scope3_category_type,
    exposure_tco2e              DECIMAL(15,3)   NOT NULL DEFAULT 0,
    exposure_pct_of_total       DECIMAL(5,2),
    -- Financial impact
    financial_impact_usd        NUMERIC(18,2)   NOT NULL DEFAULT 0,
    impact_type                 VARCHAR(30)     DEFAULT 'COST_INCREASE',
    impact_recurrence           VARCHAR(20)     DEFAULT 'ANNUAL',
    -- Probability and timing
    probability                 DECIMAL(3,2)    NOT NULL DEFAULT 0.50,
    time_horizon                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM_TERM',
    expected_onset_year         INTEGER,
    -- Risk rating
    likelihood                  VARCHAR(20)     DEFAULT 'POSSIBLE',
    consequence                 VARCHAR(20)     DEFAULT 'MODERATE',
    risk_rating                 VARCHAR(20),
    -- Mitigation
    mitigation_strategy         TEXT,
    mitigation_cost_usd         NUMERIC(14,2),
    residual_risk_usd           NUMERIC(18,2),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_tr_type CHECK (
        risk_type IN ('POLICY', 'TECHNOLOGY', 'MARKET', 'REPUTATION', 'LEGAL')
    ),
    CONSTRAINT chk_p043_tr_exposure CHECK (exposure_tco2e >= 0),
    CONSTRAINT chk_p043_tr_exposure_pct CHECK (
        exposure_pct_of_total IS NULL OR (exposure_pct_of_total >= 0 AND exposure_pct_of_total <= 100)
    ),
    CONSTRAINT chk_p043_tr_impact CHECK (financial_impact_usd >= 0),
    CONSTRAINT chk_p043_tr_impact_type CHECK (
        impact_type IS NULL OR impact_type IN (
            'COST_INCREASE', 'REVENUE_LOSS', 'ASSET_IMPAIRMENT', 'STRANDED_ASSET',
            'COMPLIANCE_COST', 'LITIGATION', 'REMEDIATION'
        )
    ),
    CONSTRAINT chk_p043_tr_recurrence CHECK (
        impact_recurrence IS NULL OR impact_recurrence IN ('ONE_TIME', 'ANNUAL', 'PERIODIC')
    ),
    CONSTRAINT chk_p043_tr_probability CHECK (probability >= 0 AND probability <= 1),
    CONSTRAINT chk_p043_tr_horizon CHECK (
        time_horizon IN ('SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM')
    ),
    CONSTRAINT chk_p043_tr_onset CHECK (
        expected_onset_year IS NULL OR (expected_onset_year >= 2024 AND expected_onset_year <= 2100)
    ),
    CONSTRAINT chk_p043_tr_likelihood CHECK (
        likelihood IS NULL OR likelihood IN ('RARE', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'ALMOST_CERTAIN')
    ),
    CONSTRAINT chk_p043_tr_consequence CHECK (
        consequence IS NULL OR consequence IN ('INSIGNIFICANT', 'MINOR', 'MODERATE', 'MAJOR', 'CATASTROPHIC')
    ),
    CONSTRAINT chk_p043_tr_rating CHECK (
        risk_rating IS NULL OR risk_rating IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p043_tr_mitigation CHECK (mitigation_cost_usd IS NULL OR mitigation_cost_usd >= 0),
    CONSTRAINT chk_p043_tr_residual CHECK (residual_risk_usd IS NULL OR residual_risk_usd >= 0)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_tr_tenant             ON ghg_accounting_scope3_complete.transition_risks(tenant_id);
CREATE INDEX idx_p043_tr_assessment         ON ghg_accounting_scope3_complete.transition_risks(assessment_id);
CREATE INDEX idx_p043_tr_type               ON ghg_accounting_scope3_complete.transition_risks(risk_type);
CREATE INDEX idx_p043_tr_category           ON ghg_accounting_scope3_complete.transition_risks(category);
CREATE INDEX idx_p043_tr_exposure           ON ghg_accounting_scope3_complete.transition_risks(exposure_tco2e DESC);
CREATE INDEX idx_p043_tr_impact             ON ghg_accounting_scope3_complete.transition_risks(financial_impact_usd DESC);
CREATE INDEX idx_p043_tr_probability        ON ghg_accounting_scope3_complete.transition_risks(probability DESC);
CREATE INDEX idx_p043_tr_horizon            ON ghg_accounting_scope3_complete.transition_risks(time_horizon);
CREATE INDEX idx_p043_tr_rating             ON ghg_accounting_scope3_complete.transition_risks(risk_rating);
CREATE INDEX idx_p043_tr_created            ON ghg_accounting_scope3_complete.transition_risks(created_at DESC);

-- Composite: assessment + top risks
CREATE INDEX idx_p043_tr_assess_top         ON ghg_accounting_scope3_complete.transition_risks(assessment_id, financial_impact_usd DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_tr_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.transition_risks
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.physical_risks
-- =============================================================================
-- Physical climate risks to suppliers and value chain assets. Each risk is
-- linked to a supplier and location, classifying the hazard type (flood,
-- drought, wildfire, etc.) with probability and financial impact.

CREATE TABLE ghg_accounting_scope3_complete.physical_risks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.risk_assessments(id) ON DELETE CASCADE,
    -- Risk identification
    risk_name                   VARCHAR(500)    NOT NULL,
    risk_type                   VARCHAR(20)     NOT NULL,
    description                 TEXT,
    -- Location
    supplier_id                 UUID,
    supplier_name               VARCHAR(500),
    location                    VARCHAR(500)    NOT NULL,
    latitude                    DECIMAL(10,7),
    longitude                   DECIMAL(10,7),
    country                     VARCHAR(3),
    -- Hazard
    hazard_type                 VARCHAR(50)     NOT NULL,
    hazard_severity             VARCHAR(20)     DEFAULT 'MODERATE',
    return_period_years         INTEGER,
    -- Financial impact
    probability                 DECIMAL(3,2)    NOT NULL DEFAULT 0.50,
    financial_impact_usd        NUMERIC(18,2)   NOT NULL DEFAULT 0,
    impact_type                 VARCHAR(30)     DEFAULT 'SUPPLY_DISRUPTION',
    disruption_days             INTEGER,
    -- Time horizon
    time_horizon                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM_TERM',
    projected_change_by_2050    VARCHAR(30),
    -- Adaptation
    adaptation_strategy         TEXT,
    adaptation_cost_usd         NUMERIC(14,2),
    residual_risk_usd           NUMERIC(18,2),
    insurance_coverage_usd      NUMERIC(18,2),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_pr_type CHECK (
        risk_type IN ('ACUTE', 'CHRONIC')
    ),
    CONSTRAINT chk_p043_pr_hazard CHECK (
        hazard_type IN (
            'FLOOD', 'CYCLONE', 'WILDFIRE', 'EXTREME_HEAT', 'EXTREME_COLD',
            'DROUGHT', 'SEA_LEVEL_RISE', 'WATER_STRESS', 'STORM_SURGE',
            'LANDSLIDE', 'PERMAFROST_THAW', 'PRECIPITATION_CHANGE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p043_pr_severity CHECK (
        hazard_severity IS NULL OR hazard_severity IN ('LOW', 'MODERATE', 'HIGH', 'EXTREME')
    ),
    CONSTRAINT chk_p043_pr_return CHECK (return_period_years IS NULL OR return_period_years > 0),
    CONSTRAINT chk_p043_pr_probability CHECK (probability >= 0 AND probability <= 1),
    CONSTRAINT chk_p043_pr_impact CHECK (financial_impact_usd >= 0),
    CONSTRAINT chk_p043_pr_impact_type CHECK (
        impact_type IS NULL OR impact_type IN (
            'SUPPLY_DISRUPTION', 'ASSET_DAMAGE', 'PRODUCTION_LOSS',
            'INCREASED_COST', 'QUALITY_IMPACT', 'LOGISTICS_DISRUPTION'
        )
    ),
    CONSTRAINT chk_p043_pr_disruption CHECK (disruption_days IS NULL OR disruption_days >= 0),
    CONSTRAINT chk_p043_pr_horizon CHECK (
        time_horizon IN ('SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM')
    ),
    CONSTRAINT chk_p043_pr_adaptation CHECK (adaptation_cost_usd IS NULL OR adaptation_cost_usd >= 0),
    CONSTRAINT chk_p043_pr_residual CHECK (residual_risk_usd IS NULL OR residual_risk_usd >= 0),
    CONSTRAINT chk_p043_pr_insurance CHECK (insurance_coverage_usd IS NULL OR insurance_coverage_usd >= 0)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_pr_tenant             ON ghg_accounting_scope3_complete.physical_risks(tenant_id);
CREATE INDEX idx_p043_pr_assessment         ON ghg_accounting_scope3_complete.physical_risks(assessment_id);
CREATE INDEX idx_p043_pr_type               ON ghg_accounting_scope3_complete.physical_risks(risk_type);
CREATE INDEX idx_p043_pr_supplier           ON ghg_accounting_scope3_complete.physical_risks(supplier_id);
CREATE INDEX idx_p043_pr_country            ON ghg_accounting_scope3_complete.physical_risks(country);
CREATE INDEX idx_p043_pr_hazard             ON ghg_accounting_scope3_complete.physical_risks(hazard_type);
CREATE INDEX idx_p043_pr_severity           ON ghg_accounting_scope3_complete.physical_risks(hazard_severity);
CREATE INDEX idx_p043_pr_impact             ON ghg_accounting_scope3_complete.physical_risks(financial_impact_usd DESC);
CREATE INDEX idx_p043_pr_probability        ON ghg_accounting_scope3_complete.physical_risks(probability DESC);
CREATE INDEX idx_p043_pr_horizon            ON ghg_accounting_scope3_complete.physical_risks(time_horizon);
CREATE INDEX idx_p043_pr_created            ON ghg_accounting_scope3_complete.physical_risks(created_at DESC);

-- Composite: assessment + top physical risks
CREATE INDEX idx_p043_pr_assess_top         ON ghg_accounting_scope3_complete.physical_risks(assessment_id, financial_impact_usd DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_pr_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.physical_risks
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.opportunities
-- =============================================================================
-- Climate-related opportunities per TCFD. Identifies potential value creation
-- from climate action in the Scope 3 value chain (e.g., low-carbon products,
-- supply chain efficiency, new markets).

CREATE TABLE ghg_accounting_scope3_complete.opportunities (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.risk_assessments(id) ON DELETE CASCADE,
    -- Opportunity
    opportunity_name            VARCHAR(500)    NOT NULL,
    opportunity_type            VARCHAR(50)     NOT NULL,
    description                 TEXT,
    -- Value
    potential_value_usd         NUMERIC(18,2)   NOT NULL DEFAULT 0,
    value_type                  VARCHAR(30)     DEFAULT 'REVENUE_GROWTH',
    annual_value_usd            NUMERIC(18,2),
    -- Feasibility
    probability                 DECIMAL(3,2)    NOT NULL DEFAULT 0.50,
    implementation_cost_usd     NUMERIC(14,2),
    payback_years               DECIMAL(4,1),
    -- Timeline
    time_horizon                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM_TERM',
    readiness_level             VARCHAR(20)     DEFAULT 'MODERATE',
    -- Scope 3 impact
    category_impact             ghg_accounting_scope3_complete.scope3_category_type,
    reduction_potential_tco2e   DECIMAL(15,3),
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'IDENTIFIED',
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_opp_type CHECK (
        opportunity_type IN (
            'RESOURCE_EFFICIENCY', 'ENERGY_SOURCE', 'PRODUCTS_SERVICES',
            'MARKETS', 'RESILIENCE', 'SUPPLY_CHAIN_OPTIMIZATION',
            'CIRCULAR_ECONOMY', 'LOW_CARBON_PRODUCTS', 'OTHER'
        )
    ),
    CONSTRAINT chk_p043_opp_value CHECK (potential_value_usd >= 0),
    CONSTRAINT chk_p043_opp_value_type CHECK (
        value_type IS NULL OR value_type IN (
            'REVENUE_GROWTH', 'COST_REDUCTION', 'RISK_REDUCTION',
            'BRAND_VALUE', 'MARKET_SHARE', 'RESILIENCE'
        )
    ),
    CONSTRAINT chk_p043_opp_annual CHECK (annual_value_usd IS NULL OR annual_value_usd >= 0),
    CONSTRAINT chk_p043_opp_probability CHECK (probability >= 0 AND probability <= 1),
    CONSTRAINT chk_p043_opp_impl_cost CHECK (implementation_cost_usd IS NULL OR implementation_cost_usd >= 0),
    CONSTRAINT chk_p043_opp_payback CHECK (payback_years IS NULL OR payback_years > 0),
    CONSTRAINT chk_p043_opp_horizon CHECK (
        time_horizon IN ('SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM')
    ),
    CONSTRAINT chk_p043_opp_readiness CHECK (
        readiness_level IS NULL OR readiness_level IN ('LOW', 'MODERATE', 'HIGH', 'READY')
    ),
    CONSTRAINT chk_p043_opp_reduction CHECK (reduction_potential_tco2e IS NULL OR reduction_potential_tco2e >= 0),
    CONSTRAINT chk_p043_opp_status CHECK (
        status IN ('IDENTIFIED', 'EVALUATING', 'APPROVED', 'IN_PROGRESS', 'REALIZED', 'DEFERRED', 'REJECTED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_opp_tenant            ON ghg_accounting_scope3_complete.opportunities(tenant_id);
CREATE INDEX idx_p043_opp_assessment        ON ghg_accounting_scope3_complete.opportunities(assessment_id);
CREATE INDEX idx_p043_opp_type              ON ghg_accounting_scope3_complete.opportunities(opportunity_type);
CREATE INDEX idx_p043_opp_value             ON ghg_accounting_scope3_complete.opportunities(potential_value_usd DESC);
CREATE INDEX idx_p043_opp_probability       ON ghg_accounting_scope3_complete.opportunities(probability DESC);
CREATE INDEX idx_p043_opp_horizon           ON ghg_accounting_scope3_complete.opportunities(time_horizon);
CREATE INDEX idx_p043_opp_status            ON ghg_accounting_scope3_complete.opportunities(status);
CREATE INDEX idx_p043_opp_category          ON ghg_accounting_scope3_complete.opportunities(category_impact);
CREATE INDEX idx_p043_opp_created           ON ghg_accounting_scope3_complete.opportunities(created_at DESC);

-- Composite: assessment + highest value opportunities
CREATE INDEX idx_p043_opp_assess_top        ON ghg_accounting_scope3_complete.opportunities(assessment_id, potential_value_usd DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_opp_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.opportunities
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_accounting_scope3_complete.financial_impacts
-- =============================================================================
-- Net present value analysis of climate risks and opportunities across
-- multiple time horizons and scenarios. Aggregates transition risks,
-- physical risks, and opportunities into discounted cash flows.

CREATE TABLE ghg_accounting_scope3_complete.financial_impacts (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.risk_assessments(id) ON DELETE CASCADE,
    -- Scenario
    scenario                    VARCHAR(100)    NOT NULL,
    scenario_description        TEXT,
    -- NPV values
    npv_10yr                    NUMERIC(18,2)   NOT NULL DEFAULT 0,
    npv_20yr                    NUMERIC(18,2),
    npv_30yr                    NUMERIC(18,2),
    -- Discount rate
    discount_rate               DECIMAL(5,3)    NOT NULL DEFAULT 0.080,
    -- Breakdown
    transition_risk_npv         NUMERIC(18,2),
    physical_risk_npv           NUMERIC(18,2),
    opportunity_npv             NUMERIC(18,2),
    net_impact_npv              NUMERIC(18,2),
    -- Carbon cost
    carbon_cost_10yr            NUMERIC(18,2),
    carbon_cost_20yr            NUMERIC(18,2),
    carbon_cost_30yr            NUMERIC(18,2),
    -- Revenue at risk
    revenue_at_risk_pct         DECIMAL(5,2),
    ebitda_at_risk_pct          DECIMAL(5,2),
    -- Sensitivity
    sensitivity_to_carbon_price DECIMAL(8,2),
    breakeven_carbon_price      DECIMAL(10,2),
    -- Metadata
    assumptions                 JSONB           DEFAULT '[]',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_fi_discount CHECK (discount_rate > 0 AND discount_rate < 1),
    CONSTRAINT chk_p043_fi_revenue CHECK (
        revenue_at_risk_pct IS NULL OR (revenue_at_risk_pct >= 0 AND revenue_at_risk_pct <= 100)
    ),
    CONSTRAINT chk_p043_fi_ebitda CHECK (
        ebitda_at_risk_pct IS NULL OR (ebitda_at_risk_pct >= 0 AND ebitda_at_risk_pct <= 100)
    ),
    CONSTRAINT chk_p043_fi_breakeven CHECK (breakeven_carbon_price IS NULL OR breakeven_carbon_price >= 0),
    CONSTRAINT uq_p043_fi_assessment_scenario UNIQUE (assessment_id, scenario)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_fi_tenant             ON ghg_accounting_scope3_complete.financial_impacts(tenant_id);
CREATE INDEX idx_p043_fi_assessment         ON ghg_accounting_scope3_complete.financial_impacts(assessment_id);
CREATE INDEX idx_p043_fi_scenario           ON ghg_accounting_scope3_complete.financial_impacts(scenario);
CREATE INDEX idx_p043_fi_npv_10yr           ON ghg_accounting_scope3_complete.financial_impacts(npv_10yr DESC);
CREATE INDEX idx_p043_fi_net_impact         ON ghg_accounting_scope3_complete.financial_impacts(net_impact_npv);
CREATE INDEX idx_p043_fi_revenue_risk       ON ghg_accounting_scope3_complete.financial_impacts(revenue_at_risk_pct DESC);
CREATE INDEX idx_p043_fi_created            ON ghg_accounting_scope3_complete.financial_impacts(created_at DESC);

-- Composite: assessment + scenario comparison
CREATE INDEX idx_p043_fi_assess_scenario    ON ghg_accounting_scope3_complete.financial_impacts(assessment_id, scenario);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_fi_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.financial_impacts
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.risk_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.transition_risks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.physical_risks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.opportunities ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.financial_impacts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_ra_tenant_isolation ON ghg_accounting_scope3_complete.risk_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_ra_service_bypass ON ghg_accounting_scope3_complete.risk_assessments
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_tr_tenant_isolation ON ghg_accounting_scope3_complete.transition_risks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_tr_service_bypass ON ghg_accounting_scope3_complete.transition_risks
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_pr_tenant_isolation ON ghg_accounting_scope3_complete.physical_risks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_pr_service_bypass ON ghg_accounting_scope3_complete.physical_risks
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_opp_tenant_isolation ON ghg_accounting_scope3_complete.opportunities
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_opp_service_bypass ON ghg_accounting_scope3_complete.opportunities
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_fi_tenant_isolation ON ghg_accounting_scope3_complete.financial_impacts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_fi_service_bypass ON ghg_accounting_scope3_complete.financial_impacts
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.risk_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.transition_risks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.physical_risks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.opportunities TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.financial_impacts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3_complete.risk_assessments IS
    'Climate risk assessment header per TCFD with carbon price, time horizons, scenario, and risk/opportunity totals.';
COMMENT ON TABLE ghg_accounting_scope3_complete.transition_risks IS
    'Transition risks (policy, technology, market, reputation) to the Scope 3 value chain with financial impact and probability.';
COMMENT ON TABLE ghg_accounting_scope3_complete.physical_risks IS
    'Physical climate risks (acute/chronic) to supplier locations with hazard type, financial impact, and adaptation strategy.';
COMMENT ON TABLE ghg_accounting_scope3_complete.opportunities IS
    'Climate-related opportunities (efficiency, products, markets) with potential value, probability, and payback analysis.';
COMMENT ON TABLE ghg_accounting_scope3_complete.financial_impacts IS
    'Net present value analysis across scenarios aggregating transition risks, physical risks, and opportunities into discounted cash flows.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.risk_assessments.scenario IS 'Climate scenario: IEA (NZE, APS, STEPS, SDS), NGFS (Orderly, Disorderly, Hot House), IPCC SSPs.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.transition_risks.risk_type IS 'TCFD transition risk type: POLICY, TECHNOLOGY, MARKET, REPUTATION, LEGAL.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.physical_risks.risk_type IS 'Physical risk type: ACUTE (event-driven) or CHRONIC (longer-term shifts).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.physical_risks.hazard_type IS 'Climate hazard: FLOOD, CYCLONE, WILDFIRE, EXTREME_HEAT, DROUGHT, SEA_LEVEL_RISE, WATER_STRESS, etc.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.financial_impacts.npv_10yr IS 'Net present value of total climate risk/opportunity over 10-year horizon.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.financial_impacts.breakeven_carbon_price IS 'Carbon price at which climate opportunities equal climate risks (USD/tCO2e).';

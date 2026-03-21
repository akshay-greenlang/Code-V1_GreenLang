-- =============================================================================
-- V140: PACK-024-carbon-neutral-003: Carbon Credit Inventory
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for carbon credit portfolio management including holdings,
-- trading history, validation, retirement tracking, and compliance with
-- standard requirements (VCS, Gold Standard, CDM, Article 6, etc).
--
-- EXTENDS:
--   V139: Carbon Management Plans
--
-- These tables provide comprehensive carbon credit management with full audit
-- trail and market-standard compliance tracking.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_credit_inventory           - Credit holdings
--   2. pack024_carbon_neutral.pack024_credit_transactions        - Trading history
--   3. pack024_carbon_neutral.pack024_credit_validation          - Compliance validation
--   4. pack024_carbon_neutral.pack024_additionality_assessment   - Additionality checks
--
-- Also includes: 55+ indexes, update triggers, security grants, and comments.
-- Previous: V139__pack024_carbon_neutral_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_credit_inventory
-- =============================================================================
-- Carbon credit portfolio with holdings, provenance, and compliance status.

CREATE TABLE pack024_carbon_neutral.pack024_credit_inventory (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    credit_type             VARCHAR(50)     NOT NULL,
    project_name            VARCHAR(500),
    project_id              VARCHAR(100),
    project_location        VARCHAR(255),
    project_country         VARCHAR(3),
    project_vintage_start   INTEGER,
    project_vintage_end     INTEGER,
    standard                VARCHAR(50)     NOT NULL,
    standard_version        VARCHAR(20),
    methodology             VARCHAR(255),
    certifying_body         VARCHAR(255),
    certification_date      DATE,
    batch_number            VARCHAR(100),
    serial_number_start     VARCHAR(100),
    serial_number_end       VARCHAR(100),
    units_held              DECIMAL(18,2)   NOT NULL,
    unit_type               VARCHAR(30)     DEFAULT 'tCO2e',
    issue_date              DATE            NOT NULL,
    expiry_date             DATE,
    validity_status         VARCHAR(30),
    purchase_date           DATE,
    purchase_price_usd      DECIMAL(18,4),
    total_cost_usd          DECIMAL(18,2),
    cost_per_unit_usd       DECIMAL(18,4),
    carbon_footprint_covered DECIMAL(18,2),
    coverage_percentage     DECIMAL(6,2),
    project_description     TEXT,
    co_benefits             TEXT[],
    sdg_alignment           TEXT[],
    social_impact_claims    TEXT[],
    environmental_additionality BOOLEAN,
    additionality_description TEXT,
    buffer_pool_credits     BOOLEAN         DEFAULT FALSE,
    buffer_pool_percentage  DECIMAL(6,2),
    buffer_pool_reason      TEXT,
    permanence_guarantee    VARCHAR(50),
    permanence_years        INTEGER,
    leakage_risk            VARCHAR(30),
    leakage_mitigation      TEXT,
    contract_terms          JSONB           DEFAULT '{}',
    insurance_coverage      BOOLEAN         DEFAULT FALSE,
    insurance_provider      VARCHAR(255),
    insurance_expiry        DATE,
    portfolio_role          VARCHAR(50),
    retirement_eligible     BOOLEAN         DEFAULT TRUE,
    retired_units           DECIMAL(18,2)   DEFAULT 0,
    retired_percentage      DECIMAL(6,2)    DEFAULT 0,
    retirement_scheduled    BOOLEAN         DEFAULT FALSE,
    scheduled_retirement_date DATE,
    monetization_strategy   VARCHAR(100),
    market_value_usd        DECIMAL(18,2),
    liquidity_rating        VARCHAR(30),
    holding_status          VARCHAR(30)     DEFAULT 'active',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_inv_credit_type CHECK (
        credit_type IN ('RENEWABLE_ENERGY', 'ENERGY_EFFICIENCY', 'FORESTRY_REFORESTATION',
                        'WASTE_MANAGEMENT', 'METHANE_REDUCTION', 'CLEAN_COOKSTOVE',
                        'WIND_POWER', 'HYDRO_POWER', 'SOLAR_POWER', 'GEOTHERMAL', 'OTHER')
    ),
    CONSTRAINT chk_pack024_inv_standard CHECK (
        standard IN ('VCS', 'GOLD_STANDARD', 'CDM', 'ACR', 'CAR', 'ARTICLE_6', 'OTHER')
    ),
    CONSTRAINT chk_pack024_inv_units_non_neg CHECK (
        units_held >= 0
    ),
    CONSTRAINT chk_pack024_inv_coverage_valid CHECK (
        coverage_percentage IS NULL OR (coverage_percentage >= 0 AND coverage_percentage <= 100)
    ),
    CONSTRAINT chk_pack024_inv_retired_valid CHECK (
        retired_units >= 0 AND retired_units <= units_held
    )
);

-- Indexes
CREATE INDEX idx_pack024_inv_org ON pack024_carbon_neutral.pack024_credit_inventory(org_id);
CREATE INDEX idx_pack024_inv_tenant ON pack024_carbon_neutral.pack024_credit_inventory(tenant_id);
CREATE INDEX idx_pack024_inv_type ON pack024_carbon_neutral.pack024_credit_inventory(credit_type);
CREATE INDEX idx_pack024_inv_standard ON pack024_carbon_neutral.pack024_credit_inventory(standard);
CREATE INDEX idx_pack024_inv_project_id ON pack024_carbon_neutral.pack024_credit_inventory(project_id);
CREATE INDEX idx_pack024_inv_country ON pack024_carbon_neutral.pack024_credit_inventory(project_country);
CREATE INDEX idx_pack024_inv_status ON pack024_carbon_neutral.pack024_credit_inventory(holding_status);
CREATE INDEX idx_pack024_inv_validity ON pack024_carbon_neutral.pack024_credit_inventory(validity_status);
CREATE INDEX idx_pack024_inv_retirement_eligible ON pack024_carbon_neutral.pack024_credit_inventory(retirement_eligible);
CREATE INDEX idx_pack024_inv_retired_units ON pack024_carbon_neutral.pack024_credit_inventory(retired_units);
CREATE INDEX idx_pack024_inv_issue_date ON pack024_carbon_neutral.pack024_credit_inventory(issue_date);
CREATE INDEX idx_pack024_inv_expiry_date ON pack024_carbon_neutral.pack024_credit_inventory(expiry_date);
CREATE INDEX idx_pack024_inv_purchase_date ON pack024_carbon_neutral.pack024_credit_inventory(purchase_date);
CREATE INDEX idx_pack024_inv_cost_per_unit ON pack024_carbon_neutral.pack024_credit_inventory(cost_per_unit_usd);
CREATE INDEX idx_pack024_inv_additionality ON pack024_carbon_neutral.pack024_credit_inventory(environmental_additionality);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_inv_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_credit_inventory
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_credit_transactions
-- =============================================================================
-- Trading and transaction history for carbon credits with full audit trail.

CREATE TABLE pack024_carbon_neutral.pack024_credit_transactions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    credit_inventory_id     UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_credit_inventory(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    transaction_date        DATE            NOT NULL,
    transaction_type        VARCHAR(50)     NOT NULL,
    transaction_reason      VARCHAR(100),
    transaction_status      VARCHAR(30)     DEFAULT 'completed',
    units_transacted        DECIMAL(18,2)   NOT NULL,
    unit_type               VARCHAR(30)     DEFAULT 'tCO2e',
    price_per_unit_usd      DECIMAL(18,4),
    total_transaction_usd   DECIMAL(18,2),
    counterparty_name       VARCHAR(255),
    counterparty_type       VARCHAR(50),
    counterparty_country    VARCHAR(3),
    broker_involved         BOOLEAN         DEFAULT FALSE,
    broker_name             VARCHAR(255),
    broker_fee_usd          DECIMAL(18,2),
    contract_reference      VARCHAR(100),
    payment_terms           VARCHAR(100),
    payment_date            DATE,
    payment_method          VARCHAR(50),
    receipt_confirmation    VARCHAR(100),
    exchange_used           VARCHAR(100),
    market_price_usd        DECIMAL(18,4),
    price_variance_usd      DECIMAL(18,4),
    price_variance_pct      DECIMAL(6,2),
    settlement_date         DATE,
    settlement_method       VARCHAR(100),
    delivery_method         VARCHAR(100),
    transfer_status         VARCHAR(30),
    registry_transfer_date  DATE,
    registry_reference      VARCHAR(100),
    risk_factors            TEXT[],
    hedging_applied         BOOLEAN         DEFAULT FALSE,
    hedging_instrument      VARCHAR(100),
    hedging_amount_usd      DECIMAL(18,2),
    tax_implications        JSONB           DEFAULT '{}',
    regulatory_compliance   BOOLEAN         DEFAULT TRUE,
    compliance_notes        TEXT,
    approval_status         VARCHAR(30),
    approved_by             VARCHAR(255),
    approval_date           DATE,
    audit_notes             TEXT,
    created_by              VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_trans_type CHECK (
        transaction_type IN ('PURCHASE', 'SALE', 'RETIREMENT', 'TRANSFER', 'DONATION', 'ADJUSTMENT')
    ),
    CONSTRAINT chk_pack024_trans_units_non_neg CHECK (
        units_transacted > 0
    ),
    CONSTRAINT chk_pack024_trans_status CHECK (
        transaction_status IN ('PENDING', 'COMPLETED', 'FAILED', 'CANCELLED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_trans_inventory_id ON pack024_carbon_neutral.pack024_credit_transactions(credit_inventory_id);
CREATE INDEX idx_pack024_trans_org ON pack024_carbon_neutral.pack024_credit_transactions(org_id);
CREATE INDEX idx_pack024_trans_tenant ON pack024_carbon_neutral.pack024_credit_transactions(tenant_id);
CREATE INDEX idx_pack024_trans_date ON pack024_carbon_neutral.pack024_credit_transactions(transaction_date DESC);
CREATE INDEX idx_pack024_trans_type ON pack024_carbon_neutral.pack024_credit_transactions(transaction_type);
CREATE INDEX idx_pack024_trans_status ON pack024_carbon_neutral.pack024_credit_transactions(transaction_status);
CREATE INDEX idx_pack024_trans_counterparty ON pack024_carbon_neutral.pack024_credit_transactions(counterparty_name);
CREATE INDEX idx_pack024_trans_price_per_unit ON pack024_carbon_neutral.pack024_credit_transactions(price_per_unit_usd);
CREATE INDEX idx_pack024_trans_settlement ON pack024_carbon_neutral.pack024_credit_transactions(settlement_date);
CREATE INDEX idx_pack024_trans_registry_transfer ON pack024_carbon_neutral.pack024_credit_transactions(registry_transfer_date);
CREATE INDEX idx_pack024_trans_compliance ON pack024_carbon_neutral.pack024_credit_transactions(regulatory_compliance);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_trans_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_credit_transactions
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_credit_validation
-- =============================================================================
-- Validation tracking for carbon credit compliance with standards.

CREATE TABLE pack024_carbon_neutral.pack024_credit_validation (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    credit_inventory_id     UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_credit_inventory(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    validation_date         DATE            NOT NULL,
    validation_type         VARCHAR(50)     NOT NULL,
    validator_organization  VARCHAR(255),
    validator_name          VARCHAR(255),
    validation_status       VARCHAR(30),
    findings_summary        TEXT,
    critical_issues         TEXT[],
    major_issues            TEXT[],
    minor_issues            TEXT[],
    issues_resolved         BOOLEAN         DEFAULT FALSE,
    resolution_evidence     TEXT[],
    standard_compliance     BOOLEAN         DEFAULT TRUE,
    compliance_percentage   DECIMAL(6,2),
    authenticity_verified   BOOLEAN         DEFAULT TRUE,
    authenticity_evidence   TEXT,
    issued_date_verified    BOOLEAN         DEFAULT TRUE,
    expiry_verification     BOOLEAN         DEFAULT TRUE,
    unit_count_verified     BOOLEAN         DEFAULT TRUE,
    serial_number_verified  BOOLEAN         DEFAULT TRUE,
    ownership_verified      BOOLEAN         DEFAULT TRUE,
    no_double_counting      BOOLEAN         DEFAULT TRUE,
    no_double_counting_evidence TEXT,
    additionality_verified  BOOLEAN         DEFAULT TRUE,
    permanent_reduction     BOOLEAN         DEFAULT TRUE,
    impact_claims_verified  BOOLEAN         DEFAULT TRUE,
    impact_claims_evidence  TEXT,
    social_safeguards_verified BOOLEAN      DEFAULT TRUE,
    environmental_safeguards_verified BOOLEAN DEFAULT TRUE,
    conflict_of_interest_check BOOLEAN      DEFAULT TRUE,
    validation_methodology  VARCHAR(255),
    documentation_complete  BOOLEAN         DEFAULT TRUE,
    missing_documentation   TEXT[],
    recommendation          VARCHAR(500),
    approved_for_use        BOOLEAN         DEFAULT FALSE,
    approval_date           DATE,
    approved_by             VARCHAR(255),
    next_validation_date    DATE,
    validation_frequency    VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_val_type CHECK (
        validation_type IN ('INITIAL', 'PERIODIC', 'PRE_RETIREMENT', 'DUE_DILIGENCE', 'AUDIT')
    ),
    CONSTRAINT chk_pack024_val_compliance CHECK (
        compliance_percentage IS NULL OR (compliance_percentage >= 0 AND compliance_percentage <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pack024_val_inventory_id ON pack024_carbon_neutral.pack024_credit_validation(credit_inventory_id);
CREATE INDEX idx_pack024_val_org ON pack024_carbon_neutral.pack024_credit_validation(org_id);
CREATE INDEX idx_pack024_val_tenant ON pack024_carbon_neutral.pack024_credit_validation(tenant_id);
CREATE INDEX idx_pack024_val_date ON pack024_carbon_neutral.pack024_credit_validation(validation_date DESC);
CREATE INDEX idx_pack024_val_type ON pack024_carbon_neutral.pack024_credit_validation(validation_type);
CREATE INDEX idx_pack024_val_status ON pack024_carbon_neutral.pack024_credit_validation(validation_status);
CREATE INDEX idx_pack024_val_compliance_pct ON pack024_carbon_neutral.pack024_credit_validation(compliance_percentage);
CREATE INDEX idx_pack024_val_approved ON pack024_carbon_neutral.pack024_credit_validation(approved_for_use);
CREATE INDEX idx_pack024_val_authenticity ON pack024_carbon_neutral.pack024_credit_validation(authenticity_verified);
CREATE INDEX idx_pack024_val_additionality ON pack024_carbon_neutral.pack024_credit_validation(additionality_verified);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_val_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_credit_validation
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_additionality_assessment
-- =============================================================================
-- Additionality assessment tracking for environmental integrity verification.

CREATE TABLE pack024_carbon_neutral.pack024_additionality_assessment (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    credit_inventory_id     UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_credit_inventory(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    assessment_methodology  VARCHAR(100)    NOT NULL,
    baseline_scenario       TEXT,
    project_scenario        TEXT,
    with_project_emissions  DECIMAL(18,4),
    without_project_emissions DECIMAL(18,4),
    additional_emission_reduction DECIMAL(18,4),
    reduction_percentage    DECIMAL(6,2),
    financial_viability_analysis BOOLEAN,
    financial_analysis_summary TEXT,
    investment_required_usd DECIMAL(18,2),
    expected_returns_usd    DECIMAL(18,2),
    expected_roi_percentage DECIMAL(6,2),
    baseline_scenario_credibility DECIMAL(5,2),
    project_scenario_credibility DECIMAL(5,2),
    common_practice_analysis BOOLEAN,
    common_practice_evidence TEXT,
    barrier_analysis        BOOLEAN,
    barriers_identified     TEXT[],
    technology_barriers     TEXT,
    capital_barriers        TEXT,
    market_barriers         TEXT,
    institutional_barriers  TEXT,
    additionality_claim     BOOLEAN,
    additionality_claim_basis TEXT,
    real_leakage_risk       BOOLEAN         DEFAULT FALSE,
    leakage_factors         TEXT[],
    leakage_mitigation_measures TEXT[],
    permanence_period_years INTEGER,
    permanence_risk_factors TEXT[],
    permanence_safeguards   TEXT[],
    sectoral_policies_analyzed BOOLEAN,
    policy_references       TEXT[],
    policy_impact_on_additionality TEXT,
    market_dynamics_analysis BOOLEAN,
    market_analysis_summary TEXT,
    assessment_conclusion   VARCHAR(30),
    assessment_confidence   DECIMAL(5,2),
    assessor_name           VARCHAR(255),
    assessment_comments     TEXT,
    critical_assumptions    JSONB           DEFAULT '{}',
    sensitivity_analysis    JSONB           DEFAULT '{}',
    peer_review_conducted   BOOLEAN         DEFAULT FALSE,
    peer_reviewer_name      VARCHAR(255),
    peer_review_date        DATE,
    peer_review_approval    BOOLEAN         DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_add_methodology CHECK (
        assessment_methodology IN ('FINANCIAL_ANALYSIS', 'BARRIER_ANALYSIS', 'COMMON_PRACTICE', 'COMBINED', 'OTHER')
    ),
    CONSTRAINT chk_pack024_add_emission_order CHECK (
        with_project_emissions IS NULL OR without_project_emissions IS NULL OR
        without_project_emissions >= with_project_emissions
    ),
    CONSTRAINT chk_pack024_add_conclusion CHECK (
        assessment_conclusion IN ('ADDITIONAL', 'NOT_ADDITIONAL', 'UNCERTAIN')
    )
);

-- Indexes
CREATE INDEX idx_pack024_add_inventory_id ON pack024_carbon_neutral.pack024_additionality_assessment(credit_inventory_id);
CREATE INDEX idx_pack024_add_org ON pack024_carbon_neutral.pack024_additionality_assessment(org_id);
CREATE INDEX idx_pack024_add_tenant ON pack024_carbon_neutral.pack024_additionality_assessment(tenant_id);
CREATE INDEX idx_pack024_add_date ON pack024_carbon_neutral.pack024_additionality_assessment(assessment_date DESC);
CREATE INDEX idx_pack024_add_methodology ON pack024_carbon_neutral.pack024_additionality_assessment(assessment_methodology);
CREATE INDEX idx_pack024_add_conclusion ON pack024_carbon_neutral.pack024_additionality_assessment(assessment_conclusion);
CREATE INDEX idx_pack024_add_confidence ON pack024_carbon_neutral.pack024_additionality_assessment(assessment_confidence);
CREATE INDEX idx_pack024_add_leakage_risk ON pack024_carbon_neutral.pack024_additionality_assessment(real_leakage_risk);
CREATE INDEX idx_pack024_add_peer_reviewed ON pack024_carbon_neutral.pack024_additionality_assessment(peer_review_conducted);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_add_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_additionality_assessment
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack024_carbon_neutral TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack024_carbon_neutral TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack024_carbon_neutral TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack024_carbon_neutral.pack024_credit_inventory IS
'Carbon credit portfolio with holdings, provenance, compliance status, and standard certification tracking (VCS, Gold Standard, CDM, Article 6, etc).';

COMMENT ON TABLE pack024_carbon_neutral.pack024_credit_transactions IS
'Trading and transaction history for carbon credits with full audit trail, counterparty tracking, and settlement status monitoring.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_credit_validation IS
'Validation tracking for carbon credit compliance with standards including authenticity, additionality, permanence, and impact claims verification.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_additionality_assessment IS
'Additionality assessment tracking for environmental integrity verification with financial analysis, barrier analysis, and common practice assessment.';
